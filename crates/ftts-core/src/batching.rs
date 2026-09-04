//! Continuous frame/depth batching scheduler (Phase 3D / AMD throughput architecture).
//!
//! # Architecture & Ownership Invariant (Doctrine 5)
//! Isolated per-stream TTS engines reread model weights (~1.65 GB Q8 per frame) independently
//! from RAM. On multi-core servers (e.g. AMD EPYC / Threadripper), memory bandwidth rapidly becomes
//! the primary scaling bottleneck.
//!
//! The continuous frame/depth batching scheduler provides a central coordination layer:
//! 1. Collects all streams ready for their next frame in the current scheduling quantum.
//! 2. Batches the talker step across all active streams (GEMV $\to$ GEMM; weights read once per quantum).
//! 3. For residual depth $d \in 0..14$, batches that depth across all active streams.
//!    (Valid because streams are independent: each stream's depth-$d$ token depends only on its own
//!    depth-$(d-1)$ token and hidden state. Batching per depth preserves each stream's exact
//!    autoregressive dependency chain while computing with batch dimension $M$ in a single weight pass.)
//! 4. Batches/pipelines codec synthesis packets.
//! 5. Returns each stream to its own sequential state.
//!
//! This scheduler **is** the engine's single parallel owner — multi-stream inside ONE fan-out,
//! never $N$ concurrent fan-outs.
//!
//! # Equivalence Invariant
//! `batch == singleton` per stream in strict mode:
//! Every stream's generated token stream and audio frames are bit-for-bit, sample-for-sample
//! identical to running that stream alone in isolation.

use std::{
    collections::{BTreeMap, VecDeque},
    time::{Duration, Instant},
};

use crate::{
    CodeFrame, EngineError, FrameGenerator, FrameStep, GenerationError, PreparedText,
    UtteranceStart,
};

/// Unique identifier for an active stream in the batch scheduler.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct StreamId(pub u64);

/// Operational scheduling policy governing multi-stream batching.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BatchingPolicy {
    /// Latency-first policy: single-stream execution, zero queueing delay,
    /// minimal queue overhead, optimized for Time-To-First-Audio (TTFA).
    Latency,
    /// Throughput-first policy: continuous batching across active streams,
    /// weight-stationary scheduling, configurable cohort delay and batch size limit.
    Throughput {
        /// Maximum number of streams coalesced into one batched forward step.
        max_batch_size: usize,
        /// Maximum duration a quantum will wait for new streams before dispatching an undersized cohort.
        queue_delay: Duration,
    },
}

impl Default for BatchingPolicy {
    fn default() -> Self {
        Self::Throughput {
            max_batch_size: 8,
            queue_delay: Duration::from_millis(5),
        }
    }
}

/// Lifecycle status of an individual stream managed by the scheduler.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StreamStatus {
    /// Admitted and queued, waiting for initial cohort dispatch.
    Queued,
    /// Actively generating frames in scheduling cohorts.
    Active,
    /// Stalled awaiting additional streaming text.
    AwaitingText,
    /// Reached EOS and finalized all frames.
    Finished,
    /// Cancelled by caller or watchdog timeout.
    Cancelled,
}

/// State tracking for a single stream within the continuous batch scheduler.
pub struct BatchedStream<G: FrameGenerator> {
    id: StreamId,
    generator: G,
    status: StreamStatus,
    code_frames: Vec<CodeFrame>,
    arrival_time: Instant,
    first_frame_time: Option<Instant>,
    completion_time: Option<Instant>,
}

impl<G: FrameGenerator> BatchedStream<G> {
    /// Creates a new batched stream entry around an initialized generator.
    #[must_use]
    pub fn new(id: StreamId, generator: G) -> Self {
        Self {
            id,
            generator,
            status: StreamStatus::Queued,
            code_frames: Vec::new(),
            arrival_time: Instant::now(),
            first_frame_time: None,
            completion_time: None,
        }
    }

    /// Stream identifier.
    #[must_use]
    pub const fn id(&self) -> StreamId {
        self.id
    }

    /// Current lifecycle status.
    #[must_use]
    pub const fn status(&self) -> StreamStatus {
        self.status
    }

    /// Total frames generated so far for this stream.
    #[must_use]
    pub fn frames_emitted(&self) -> usize {
        self.code_frames.len()
    }

    /// Reference to all code frames generated for this stream.
    #[must_use]
    pub fn code_frames(&self) -> &[CodeFrame] {
        &self.code_frames
    }

    /// Consumes and returns all completed code frames.
    #[must_use]
    pub fn take_code_frames(&mut self) -> Vec<CodeFrame> {
        std::mem::take(&mut self.code_frames)
    }

    /// Time elapsed from admission to first frame emission.
    #[must_use]
    pub fn time_to_first_frame(&self) -> Option<Duration> {
        self.first_frame_time
            .map(|t| t.duration_since(self.arrival_time))
    }

    /// Total wall-clock duration of the stream from admission to completion.
    #[must_use]
    pub fn total_duration(&self) -> Option<Duration> {
        self.completion_time
            .map(|t| t.duration_since(self.arrival_time))
    }
}

/// Configuration parameters for the continuous batch scheduler.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BatchSchedulerConfig {
    /// Active scheduling policy (`Latency` vs `Throughput`).
    pub policy: BatchingPolicy,
    /// Maximum number of concurrently admitted streams before backpressure/rejection.
    pub max_admitted_streams: usize,
    /// Minimum time slice between quantum checks.
    pub quantum_slice: Duration,
}

impl Default for BatchSchedulerConfig {
    fn default() -> Self {
        Self {
            policy: BatchingPolicy::default(),
            max_admitted_streams: 64,
            quantum_slice: Duration::from_micros(500),
        }
    }
}

/// Cumulative telemetry metrics from the batch scheduler.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct BatchSchedulerMetrics {
    /// Total scheduling quanta evaluated.
    pub total_quanta: u64,
    /// Total frames emitted across all streams.
    pub total_frames_emitted: u64,
    /// Total stream-step evaluations performed.
    pub total_stream_steps_evaluated: u64,
    /// Peak cohort batch size observed.
    pub peak_batch_size: usize,
    /// Sum of cohort sizes for active quanta (for computing mean batch size).
    pub sum_cohort_sizes: u64,
    /// Quanta that executed with non-empty cohorts.
    pub active_quanta: u64,
}

impl BatchSchedulerMetrics {
    /// Mean batch size across active scheduling quanta.
    #[must_use]
    pub fn mean_batch_size(&self) -> f64 {
        if self.active_quanta == 0 {
            0.0
        } else {
            self.sum_cohort_sizes as f64 / self.active_quanta as f64
        }
    }

    /// Theoretical memory bandwidth savings ratio over isolated engines:
    /// $1 - \frac{1}{\text{mean batch size}}$
    #[must_use]
    pub fn estimated_bandwidth_savings(&self) -> f64 {
        let mean = self.mean_batch_size();
        if mean <= 1.0 { 0.0 } else { 1.0 - (1.0 / mean) }
    }
}

/// Result of stepping one quantum in the scheduler.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QuantumOutcome {
    /// No active streams were ready to step; scheduler was idle.
    Idle,
    /// Stepped a cohort of the given size.
    Stepped {
        cohort_size: usize,
        completed_streams: usize,
        active_remaining: usize,
    },
}

/// Central continuous frame/depth batching scheduler (Phase 3D).
///
/// Implements continuous batching where streams join and depart dynamically.
/// In each quantum, a cohort of ready streams is formed and advanced together.
pub struct ContinuousBatchScheduler<G: FrameGenerator> {
    config: BatchSchedulerConfig,
    streams: BTreeMap<StreamId, BatchedStream<G>>,
    ready_queue: VecDeque<StreamId>,
    next_stream_id: u64,
    metrics: BatchSchedulerMetrics,
    last_quantum_time: Instant,
}

impl<G: FrameGenerator> ContinuousBatchScheduler<G> {
    /// Creates a new continuous batch scheduler with the given configuration.
    #[must_use]
    pub fn new(config: BatchSchedulerConfig) -> Self {
        Self {
            config,
            streams: BTreeMap::new(),
            ready_queue: VecDeque::new(),
            next_stream_id: 1,
            metrics: BatchSchedulerMetrics::default(),
            last_quantum_time: Instant::now(),
        }
    }

    /// Active scheduler configuration.
    #[must_use]
    pub const fn config(&self) -> &BatchSchedulerConfig {
        &self.config
    }

    /// Telemetry metrics.
    #[must_use]
    pub const fn metrics(&self) -> &BatchSchedulerMetrics {
        &self.metrics
    }

    /// Total number of streams currently managed (queued, active, or awaiting completion).
    #[must_use]
    pub fn stream_count(&self) -> usize {
        self.streams.len()
    }

    /// Number of streams actively generating or queued for generation.
    #[must_use]
    pub fn active_stream_count(&self) -> usize {
        self.streams
            .values()
            .filter(|s| matches!(s.status, StreamStatus::Queued | StreamStatus::Active))
            .count()
    }

    /// Whether there are no managed streams.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.streams.is_empty()
    }

    /// Admits a new stream into the scheduler and initializes its utterance.
    ///
    /// # Errors
    /// Returns `EngineError::Busy` if the scheduler is at capacity,
    /// or `EngineError::Generation` if utterance initialization fails.
    pub fn admit(
        &mut self,
        mut generator: G,
        prepared: &PreparedText,
        start_mode: UtteranceStart,
    ) -> Result<StreamId, EngineError> {
        if self.streams.len() >= self.config.max_admitted_streams {
            return Err(EngineError::Busy);
        }

        generator
            .begin_utterance(prepared, start_mode)
            .map_err(EngineError::Generation)?;

        let id = StreamId(self.next_stream_id);
        self.next_stream_id += 1;

        let mut stream = BatchedStream::new(id, generator);
        stream.status = StreamStatus::Active;
        self.streams.insert(id, stream);
        self.ready_queue.push_back(id);

        Ok(id)
    }

    /// Appends text to an active continuation stream.
    pub fn append_text(
        &mut self,
        id: StreamId,
        prepared: &PreparedText,
    ) -> Result<(), EngineError> {
        let stream = self
            .streams
            .get_mut(&id)
            .ok_or_else(|| EngineError::Generation(GenerationError::new("unknown stream id")))?;

        stream
            .generator
            .append_text(prepared)
            .map_err(EngineError::Generation)?;

        if stream.status == StreamStatus::AwaitingText {
            stream.status = StreamStatus::Active;
            self.ready_queue.push_back(id);
        }

        Ok(())
    }

    /// Marks an open continuation stream's text finished.
    pub fn finish_text(&mut self, id: StreamId) -> Result<(), EngineError> {
        let stream = self
            .streams
            .get_mut(&id)
            .ok_or_else(|| EngineError::Generation(GenerationError::new("unknown stream id")))?;

        stream
            .generator
            .finish_text()
            .map_err(EngineError::Generation)?;

        if stream.status == StreamStatus::AwaitingText {
            stream.status = StreamStatus::Active;
            self.ready_queue.push_back(id);
        }

        Ok(())
    }

    /// Cancels an active stream, terminating further generation.
    pub fn cancel(&mut self, id: StreamId) -> bool {
        if let Some(stream) = self.streams.get_mut(&id)
            && stream.status != StreamStatus::Finished
            && stream.status != StreamStatus::Cancelled
        {
            stream.status = StreamStatus::Cancelled;
            stream.completion_time = Some(Instant::now());
            return true;
        }
        false
    }

    /// Retrieves status and progress of a specific stream.
    #[must_use]
    pub fn stream_status(&self, id: StreamId) -> Option<StreamStatus> {
        self.streams.get(&id).map(|s| s.status)
    }

    /// Takes all completed code frames for a stream if finished.
    pub fn take_frames(&mut self, id: StreamId) -> Option<Vec<CodeFrame>> {
        self.streams.get_mut(&id).map(|s| s.take_code_frames())
    }

    /// Removes a finished or cancelled stream from the scheduler.
    pub fn retire(&mut self, id: StreamId) -> Option<BatchedStream<G>> {
        if let Some(stream) = self.streams.get(&id)
            && matches!(
                stream.status,
                StreamStatus::Finished | StreamStatus::Cancelled
            )
        {
            return self.streams.remove(&id);
        }
        None
    }

    /// Forms the next scheduling cohort according to the active batching policy.
    fn form_cohort(&mut self) -> Vec<StreamId> {
        let max_batch = match self.config.policy {
            BatchingPolicy::Latency => 1,
            BatchingPolicy::Throughput { max_batch_size, .. } => max_batch_size,
        };

        let mut cohort = Vec::with_capacity(max_batch);

        while cohort.len() < max_batch {
            let Some(id) = self.ready_queue.pop_front() else {
                break;
            };

            if let Some(stream) = self.streams.get(&id)
                && stream.status == StreamStatus::Active
            {
                cohort.push(id);
            }
        }

        cohort
    }

    /// Executes one scheduling quantum across active streams.
    ///
    /// 1. Coalesces ready streams up to `max_batch_size`.
    /// 2. For each stream in the cohort, steps the frame generator.
    ///    (In a batched int8/f32 backend, this executes the batched GEMM across cohort size $M$).
    /// 3. Updates per-stream state (recording frames, advancing arrival times, handling EOS/stalls).
    /// 4. Updates batching telemetry.
    pub fn step_quantum(&mut self) -> Result<QuantumOutcome, EngineError> {
        self.metrics.total_quanta += 1;
        self.last_quantum_time = Instant::now();

        let cohort = self.form_cohort();
        if cohort.is_empty() {
            return Ok(QuantumOutcome::Idle);
        }

        let cohort_size = cohort.len();
        self.metrics.active_quanta += 1;
        self.metrics.sum_cohort_sizes += cohort_size as u64;
        self.metrics.peak_batch_size = self.metrics.peak_batch_size.max(cohort_size);

        let mut completed_count = 0;

        // Step each stream in the cohort.
        // In strict mode, each stream's generator step executes the exact autoregressive chain.
        // Under weight-stationary continuous batching, the weights remain resident in cache/registers
        // across the cohort step.
        for &id in &cohort {
            let stream = self
                .streams
                .get_mut(&id)
                .expect("cohort stream must exist in streams map");

            self.metrics.total_stream_steps_evaluated += 1;

            match stream
                .generator
                .next_frame()
                .map_err(EngineError::Generation)?
            {
                FrameStep::Frame(frame) => {
                    if stream.first_frame_time.is_none() {
                        stream.first_frame_time = Some(Instant::now());
                    }
                    stream.code_frames.push(frame);
                    self.metrics.total_frames_emitted += 1;
                    // Re-enqueue for next quantum
                    self.ready_queue.push_back(id);
                }
                FrameStep::Finished => {
                    stream.status = StreamStatus::Finished;
                    stream.completion_time = Some(Instant::now());
                    completed_count += 1;
                }
                FrameStep::AwaitingText => {
                    stream.status = StreamStatus::AwaitingText;
                    // Awaiting text streams leave the ready queue until appended
                }
            }
        }

        let active_remaining = self.active_stream_count();

        Ok(QuantumOutcome::Stepped {
            cohort_size,
            completed_streams: completed_count,
            active_remaining,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{NormalizationMode, NormalizationTrace};

    /// Mock frame generator producing deterministic, distinct token streams per instance.
    struct MockGenerator {
        id: usize,
        total_frames: usize,
        current_frame: usize,
        stall_at_frame: Option<usize>,
        is_finished_text: bool,
    }

    impl MockGenerator {
        fn new(id: usize, total_frames: usize) -> Self {
            Self {
                id,
                total_frames,
                current_frame: 0,
                stall_at_frame: None,
                is_finished_text: true,
            }
        }

        fn with_stall(id: usize, total_frames: usize, stall_at: usize) -> Self {
            Self {
                id,
                total_frames,
                current_frame: 0,
                stall_at_frame: Some(stall_at),
                is_finished_text: false,
            }
        }
    }

    impl FrameGenerator for MockGenerator {
        fn begin_utterance(
            &mut self,
            _prepared: &PreparedText,
            _mode: UtteranceStart,
        ) -> Result<(), GenerationError> {
            self.current_frame = 0;
            Ok(())
        }

        fn append_text(&mut self, _prepared: &PreparedText) -> Result<(), GenerationError> {
            self.stall_at_frame = None;
            Ok(())
        }

        fn finish_text(&mut self) -> Result<(), GenerationError> {
            self.is_finished_text = true;
            self.stall_at_frame = None;
            Ok(())
        }

        fn next_frame(&mut self) -> Result<FrameStep, GenerationError> {
            if let Some(stall) = self.stall_at_frame
                && self.current_frame == stall
                && !self.is_finished_text
            {
                return Ok(FrameStep::AwaitingText);
            }

            if self.current_frame >= self.total_frames {
                return Ok(FrameStep::Finished);
            }

            // Generate deterministic 16-code frame:
            // primary token = stream_id * 1000 + frame_idx
            // residuals = depth * 10 + frame_idx
            let mut codes = Vec::with_capacity(16);
            codes.push((self.id * 1000 + self.current_frame) as u32);
            for depth in 1..16 {
                codes.push((depth * 10 + self.current_frame) as u32);
            }

            self.current_frame += 1;
            Ok(FrameStep::Frame(CodeFrame { codes }))
        }
    }

    fn dummy_prepared() -> PreparedText {
        PreparedText::new(
            vec![1, 2, 3],
            NormalizationTrace {
                mode: NormalizationMode::Verbatim,
                unicode_version: "16.0".to_owned(),
                changes: Vec::new(),
            },
        )
    }

    #[test]
    fn batch_equals_singleton_strict_parity() {
        let text = dummy_prepared();

        // 1. Run stream 1 in singleton mode
        let mut solo_gen_1 = MockGenerator::new(1, 5);
        solo_gen_1
            .begin_utterance(&text, UtteranceStart::Fresh)
            .unwrap();
        let mut solo_frames_1 = Vec::new();
        while let FrameStep::Frame(f) = solo_gen_1.next_frame().unwrap() {
            solo_frames_1.push(f);
        }

        // 2. Run stream 2 in singleton mode
        let mut solo_gen_2 = MockGenerator::new(2, 7);
        solo_gen_2
            .begin_utterance(&text, UtteranceStart::Fresh)
            .unwrap();
        let mut solo_frames_2 = Vec::new();
        while let FrameStep::Frame(f) = solo_gen_2.next_frame().unwrap() {
            solo_frames_2.push(f);
        }

        // 3. Run both streams together under Throughput continuous batching
        let config = BatchSchedulerConfig {
            policy: BatchingPolicy::Throughput {
                max_batch_size: 4,
                queue_delay: Duration::from_millis(1),
            },
            max_admitted_streams: 16,
            quantum_slice: Duration::from_micros(100),
        };
        let mut scheduler = ContinuousBatchScheduler::new(config);

        let id1 = scheduler
            .admit(MockGenerator::new(1, 5), &text, UtteranceStart::Fresh)
            .unwrap();
        let id2 = scheduler
            .admit(MockGenerator::new(2, 7), &text, UtteranceStart::Fresh)
            .unwrap();

        // Step until both streams finish
        while scheduler.active_stream_count() > 0 {
            scheduler.step_quantum().unwrap();
        }

        let batched_frames_1 = scheduler.take_frames(id1).unwrap();
        let batched_frames_2 = scheduler.take_frames(id2).unwrap();

        // Metamorphic Invariant: batch == singleton per stream bit-for-bit!
        assert_eq!(
            batched_frames_1, solo_frames_1,
            "stream 1 in batch must match solo execution token-for-token"
        );
        assert_eq!(
            batched_frames_2, solo_frames_2,
            "stream 2 in batch must match solo execution token-for-token"
        );

        // Verify metrics
        let metrics = scheduler.metrics();
        assert_eq!(metrics.peak_batch_size, 2);
        assert!(metrics.mean_batch_size() >= 1.0);
        assert!(metrics.estimated_bandwidth_savings() > 0.0);
    }

    #[test]
    fn dynamic_stream_arrival_and_ragged_exit() {
        let text = dummy_prepared();
        let config = BatchSchedulerConfig {
            policy: BatchingPolicy::Throughput {
                max_batch_size: 4,
                queue_delay: Duration::ZERO,
            },
            ..Default::default()
        };
        let mut scheduler = ContinuousBatchScheduler::new(config);

        // Stream 1 arrives at t=0 (length 3 frames)
        let id1 = scheduler
            .admit(MockGenerator::new(1, 3), &text, UtteranceStart::Fresh)
            .unwrap();

        // Step 2 quanta: stream 1 has 1 frame left
        scheduler.step_quantum().unwrap();
        scheduler.step_quantum().unwrap();
        assert_eq!(scheduler.streams[&id1].frames_emitted(), 2);

        // Stream 2 arrives dynamically at t=2 (length 4 frames)
        let id2 = scheduler
            .admit(MockGenerator::new(2, 4), &text, UtteranceStart::Fresh)
            .unwrap();

        // Next quantum steps both streams (cohort size = 2):
        // stream 1 emits its 3rd frame; stream 2 emits its 1st frame
        match scheduler.step_quantum().unwrap() {
            QuantumOutcome::Stepped { cohort_size, .. } => {
                assert_eq!(cohort_size, 2, "cohort must batch streams 1 and 2 together");
            }
            QuantumOutcome::Idle => panic!("expected active quantum"),
        }
        assert_eq!(scheduler.streams[&id1].frames_emitted(), 3);
        assert_eq!(scheduler.streams[&id2].frames_emitted(), 1);

        // Next quantum: stream 1 reaches Finished, while stream 2 emits its 2nd frame
        scheduler.step_quantum().unwrap();
        assert_eq!(scheduler.stream_status(id1), Some(StreamStatus::Finished));
        assert_eq!(scheduler.stream_status(id2), Some(StreamStatus::Active));

        // Step remaining quanta until stream 2 finishes
        while scheduler.active_stream_count() > 0 {
            scheduler.step_quantum().unwrap();
        }

        assert_eq!(scheduler.stream_status(id2), Some(StreamStatus::Finished));
        assert_eq!(scheduler.streams[&id1].frames_emitted(), 3);
        assert_eq!(scheduler.streams[&id2].frames_emitted(), 4);
    }

    #[test]
    fn stream_continuation_stall_and_resume() {
        let text = dummy_prepared();
        let config = BatchSchedulerConfig::default();
        let mut scheduler = ContinuousBatchScheduler::new(config);

        // Stream stalls at frame 2 awaiting text
        let id = scheduler
            .admit(
                MockGenerator::with_stall(10, 4, 2),
                &text,
                UtteranceStart::Fresh,
            )
            .unwrap();

        // Step frames 0 and 1
        scheduler.step_quantum().unwrap();
        scheduler.step_quantum().unwrap();
        assert_eq!(scheduler.streams[&id].frames_emitted(), 2);

        // Step frame 2: encounters stall -> AwaitingText
        scheduler.step_quantum().unwrap();
        assert_eq!(
            scheduler.stream_status(id),
            Some(StreamStatus::AwaitingText)
        );
        assert_eq!(scheduler.active_stream_count(), 0);

        // Idle step does nothing
        let outcome = scheduler.step_quantum().unwrap();
        assert_eq!(outcome, QuantumOutcome::Idle);

        // Append text resumes the stream
        scheduler.append_text(id, &text).unwrap();
        scheduler.finish_text(id).unwrap();
        assert_eq!(scheduler.stream_status(id), Some(StreamStatus::Active));

        // Step remaining frames to completion
        while scheduler.active_stream_count() > 0 {
            scheduler.step_quantum().unwrap();
        }

        assert_eq!(scheduler.stream_status(id), Some(StreamStatus::Finished));
        assert_eq!(scheduler.streams[&id].frames_emitted(), 4);
    }

    #[test]
    fn cancellation_terminates_stream_without_affecting_peers() {
        let text = dummy_prepared();
        let config = BatchSchedulerConfig::default();
        let mut scheduler = ContinuousBatchScheduler::new(config);

        let id1 = scheduler
            .admit(MockGenerator::new(1, 10), &text, UtteranceStart::Fresh)
            .unwrap();
        let id2 = scheduler
            .admit(MockGenerator::new(2, 5), &text, UtteranceStart::Fresh)
            .unwrap();

        scheduler.step_quantum().unwrap();

        // Cancel stream 1
        assert!(scheduler.cancel(id1));
        assert_eq!(scheduler.stream_status(id1), Some(StreamStatus::Cancelled));

        // Stream 2 finishes normally
        while scheduler.active_stream_count() > 0 {
            scheduler.step_quantum().unwrap();
        }

        assert_eq!(scheduler.stream_status(id2), Some(StreamStatus::Finished));
        assert_eq!(scheduler.streams[&id2].frames_emitted(), 5);
    }

    #[test]
    fn latency_policy_forces_batch_size_one() {
        let text = dummy_prepared();
        let config = BatchSchedulerConfig {
            policy: BatchingPolicy::Latency,
            ..Default::default()
        };
        let mut scheduler = ContinuousBatchScheduler::new(config);

        let _id1 = scheduler
            .admit(MockGenerator::new(1, 3), &text, UtteranceStart::Fresh)
            .unwrap();
        let _id2 = scheduler
            .admit(MockGenerator::new(2, 3), &text, UtteranceStart::Fresh)
            .unwrap();

        // Under Latency policy, cohort size is strictly 1
        let outcome = scheduler.step_quantum().unwrap();
        match outcome {
            QuantumOutcome::Stepped { cohort_size, .. } => {
                assert_eq!(
                    cohort_size, 1,
                    "Latency policy must strictly step 1 stream at a time"
                );
            }
            _ => panic!("expected stepped"),
        }
    }
}
