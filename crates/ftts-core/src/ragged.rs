//! Ragged FrankenMTP x Continuous Batching joint scheduler (Phase 3D / AMD throughput).
//!
//! # Problem Statement
//! Under speculative decoding (FrankenMTP), concurrent streams reject at different residual depths,
//! causing depth-synchronized batching to become *ragged*:
//! - Stream A accepts all 15 residuals ($L_A = 15$) in a single causal block-verify pass.
//! - Stream B accepts 7 residuals ($L_B = 7$) and requires 8 sequential repair steps.
//! - Stream C accepts 0 residuals ($L_C = 0$) and requires all 15 steps.
//!
//! If managed naively, speculative streams would either stall waiting for the slowest repair stream,
//! or break batching into tiny single-stream operations that destroy memory read amortization.
//!
//! # Dual-Lane Architecture (The Winning Design)
//! The scheduler divides execution into two coordinated lanes per quantum:
//! 1. **Block-Verify Lane (Lane 1)**:
//!    All active streams simultaneously draft candidate residuals and execute the causal
//!    block-verification pass together with full cohort batch size $M_{verify}$.
//!    Streams with $L_i = 15$ (full acceptance) finish their frame immediately and queue for codec packets.
//! 2. **Sequential-Repair Lane (Lane 2)**:
//!    Streams with $L_i < 15$ migrate into the Sequential-Repair Lane.
//!    The repair lane steps depth-by-depth ($d = \min(L_i)..15$) across only the sub-cohort
//!    active at depth $d$, contracting the active cohort as streams reach depth 15.
//! 3. **Deterministic Fallback**:
//!    Streams with speculation disabled (or demoted via AF-3) bypass Lane 1 and execute
//!    directly in Lane 2.
//!
//! # Equivalence Invariant
//! Regardless of cohort composition, acceptance rates, or ragged repair depths, each stream's
//! emitted token sequence is **bit-for-bit identical** to running that stream in isolated sequential decode.

use std::collections::{BTreeMap, VecDeque};

use crate::{
    CodeFrame, EngineError, FrameGenerator, GenerationError, PreparedText, UtteranceStart,
    batching::{BatchSchedulerConfig, BatchingPolicy, StreamId, StreamStatus},
};

/// Outcome of a speculative draft-and-verify step for one stream.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SpeculativeStepOutcome {
    /// Full speculative acceptance: all 15 residual depths verified successfully.
    FullAccept(CodeFrame),
    /// Partial acceptance up to `accepted_depth` ($< 15$).
    /// The frame has valid codes up to `accepted_depth`; subsequent depths require sequential repair.
    PartialAccept {
        accepted_codes: Vec<u32>,
        accepted_depth: usize,
    },
    /// Generator is finished (EOS reached).
    Finished,
    /// Stream is stalled awaiting text.
    AwaitingText,
}

/// Extended generator trait supporting FrankenMTP speculative block verification.
pub trait SpeculativeFrameGenerator: FrameGenerator {
    /// Returns true if all frames for the utterance have been completed.
    fn is_finished(&self) -> bool;

    /// Attempts a speculative block draft and causal verification pass.
    ///
    /// If speculation is disabled or demoted, implementations can return `PartialAccept` with
    /// `accepted_depth = 0` to route directly to sequential repair.
    fn step_speculative_block(&mut self) -> Result<SpeculativeStepOutcome, GenerationError>;

    /// Advances one sequential repair step for the specified residual depth.
    fn step_repair_depth(
        &mut self,
        depth: usize,
        partial_codes: &mut Vec<u32>,
    ) -> Result<Option<CodeFrame>, GenerationError>;
}

/// Execution lane assignment for an active stream within a quantum.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StreamLane {
    /// Ready for the speculative block-verify lane.
    BlockVerify,
    /// Migrated to sequential-repair lane with partial codes and starting depth.
    SequentialRepair {
        partial_codes: Vec<u32>,
        next_depth: usize,
    },
    /// Completed this frame's generation.
    FrameComplete(CodeFrame),
}

/// State tracking for a stream in the ragged joint scheduler.
pub struct RaggedStream<G: SpeculativeFrameGenerator> {
    id: StreamId,
    generator: G,
    status: StreamStatus,
    lane: StreamLane,
    code_frames: Vec<CodeFrame>,
    speculation_enabled: bool,
    full_accept_count: u64,
    repair_step_count: u64,
}

impl<G: SpeculativeFrameGenerator> RaggedStream<G> {
    /// Creates a new ragged stream state.
    #[must_use]
    pub fn new(id: StreamId, generator: G, speculation_enabled: bool) -> Self {
        Self {
            id,
            generator,
            status: StreamStatus::Queued,
            lane: StreamLane::BlockVerify,
            code_frames: Vec::new(),
            speculation_enabled,
            full_accept_count: 0,
            repair_step_count: 0,
        }
    }

    /// Stream ID.
    #[must_use]
    pub const fn id(&self) -> StreamId {
        self.id
    }

    /// Stream lifecycle status.
    #[must_use]
    pub const fn status(&self) -> StreamStatus {
        self.status
    }

    /// Number of code frames emitted.
    #[must_use]
    pub fn frames_emitted(&self) -> usize {
        self.code_frames.len()
    }

    /// Reference to completed code frames.
    #[must_use]
    pub fn code_frames(&self) -> &[CodeFrame] {
        &self.code_frames
    }

    /// Consumes and returns all completed code frames.
    #[must_use]
    pub fn take_code_frames(&mut self) -> Vec<CodeFrame> {
        std::mem::take(&mut self.code_frames)
    }
}

/// Joint scheduler telemetry metrics.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RaggedSchedulerMetrics {
    /// Total scheduling quanta executed.
    pub total_quanta: u64,
    /// Total frames completed.
    pub total_frames_completed: u64,
    /// Total frames resolved via Lane 1 (speculative full-accept in 1 block step).
    pub lane1_full_accepts: u64,
    /// Total frames that required Lane 2 (sequential repair).
    pub lane2_repair_frames: u64,
    /// Total repair depth-steps evaluated in Lane 2.
    pub total_repair_steps: u64,
    /// Peak cohort size.
    pub peak_cohort_size: usize,
}

impl RaggedSchedulerMetrics {
    /// Fraction of frames resolved entirely in Lane 1 without any sequential repair steps.
    #[must_use]
    pub fn full_accept_rate(&self) -> f64 {
        if self.total_frames_completed == 0 {
            0.0
        } else {
            self.lane1_full_accepts as f64 / self.total_frames_completed as f64
        }
    }

    /// Mean sequential repair steps per frame across all completed frames.
    #[must_use]
    pub fn mean_repair_steps_per_frame(&self) -> f64 {
        if self.total_frames_completed == 0 {
            0.0
        } else {
            self.total_repair_steps as f64 / self.total_frames_completed as f64
        }
    }
}

/// Ragged FrankenMTP x Continuous Batching joint scheduler (Dual-Lane).
pub struct RaggedBatchScheduler<G: SpeculativeFrameGenerator> {
    config: BatchSchedulerConfig,
    streams: BTreeMap<StreamId, RaggedStream<G>>,
    ready_queue: VecDeque<StreamId>,
    next_stream_id: u64,
    metrics: RaggedSchedulerMetrics,
}

impl<G: SpeculativeFrameGenerator> RaggedBatchScheduler<G> {
    /// Creates a new dual-lane ragged scheduler.
    #[must_use]
    pub fn new(config: BatchSchedulerConfig) -> Self {
        Self {
            config,
            streams: BTreeMap::new(),
            ready_queue: VecDeque::new(),
            next_stream_id: 1,
            metrics: RaggedSchedulerMetrics::default(),
        }
    }

    /// Active metrics.
    #[must_use]
    pub const fn metrics(&self) -> &RaggedSchedulerMetrics {
        &self.metrics
    }

    /// Number of streams currently active or queued.
    #[must_use]
    pub fn active_stream_count(&self) -> usize {
        self.streams
            .values()
            .filter(|s| matches!(s.status, StreamStatus::Queued | StreamStatus::Active))
            .count()
    }

    /// Admits a new stream with optional speculative decoding.
    pub fn admit(
        &mut self,
        mut generator: G,
        prepared: &PreparedText,
        start_mode: UtteranceStart,
        speculation_enabled: bool,
    ) -> Result<StreamId, EngineError> {
        if self.streams.len() >= self.config.max_admitted_streams {
            return Err(EngineError::Busy);
        }

        generator
            .begin_utterance(prepared, start_mode)
            .map_err(EngineError::Generation)?;

        let id = StreamId(self.next_stream_id);
        self.next_stream_id += 1;

        let mut stream = RaggedStream::new(id, generator, speculation_enabled);
        stream.status = StreamStatus::Active;
        self.streams.insert(id, stream);
        self.ready_queue.push_back(id);

        Ok(id)
    }

    /// Takes completed frames for a stream.
    pub fn take_frames(&mut self, id: StreamId) -> Option<Vec<CodeFrame>> {
        self.streams.get_mut(&id).map(|s| s.take_code_frames())
    }

    /// Forms the next scheduling cohort based on batching policy.
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

    /// Steps one quantum using the dual-lane architecture.
    ///
    /// Phase 1: Lane 1 (Block-Verify Lane) executes batched speculative verification.
    /// Streams with 15 accepted depths finish immediately.
    /// Streams with $< 15$ accepted depths migrate to Lane 2.
    ///
    /// Phase 2: Lane 2 (Sequential-Repair Lane) dynamically steps only the sub-cohort
    /// of streams requiring repair at each remaining depth.
    pub fn step_quantum(&mut self) -> Result<usize, EngineError> {
        self.metrics.total_quanta += 1;
        let cohort = self.form_cohort();
        if cohort.is_empty() {
            return Ok(0);
        }

        let cohort_size = cohort.len();
        self.metrics.peak_cohort_size = self.metrics.peak_cohort_size.max(cohort_size);

        // --- PHASE 1: Lane 1 (Block-Verify Lane) ---
        let mut repair_cohort = Vec::new();
        let mut finished_in_lane1 = Vec::new();

        for &id in &cohort {
            let stream = self.streams.get_mut(&id).expect("stream exists");

            if stream.generator.is_finished() {
                stream.status = StreamStatus::Finished;
                continue;
            }

            if !stream.speculation_enabled {
                stream.lane = StreamLane::SequentialRepair {
                    partial_codes: Vec::new(),
                    next_depth: 0,
                };
                repair_cohort.push(id);
                continue;
            }

            match stream
                .generator
                .step_speculative_block()
                .map_err(EngineError::Generation)?
            {
                SpeculativeStepOutcome::FullAccept(frame) => {
                    stream.code_frames.push(frame);
                    stream.full_accept_count += 1;
                    self.metrics.lane1_full_accepts += 1;
                    self.metrics.total_frames_completed += 1;
                    finished_in_lane1.push(id);
                }
                SpeculativeStepOutcome::PartialAccept {
                    accepted_codes,
                    accepted_depth,
                } => {
                    stream.lane = StreamLane::SequentialRepair {
                        partial_codes: accepted_codes,
                        next_depth: accepted_depth,
                    };
                    repair_cohort.push(id);
                }
                SpeculativeStepOutcome::Finished => {
                    stream.status = StreamStatus::Finished;
                }
                SpeculativeStepOutcome::AwaitingText => {
                    stream.status = StreamStatus::AwaitingText;
                }
            }
        }

        // Re-enqueue Lane 1 finished streams for their next frame in future quanta
        for id in finished_in_lane1 {
            let stream = self.streams.get_mut(&id).expect("stream exists");
            if stream.status == StreamStatus::Active {
                stream.lane = StreamLane::BlockVerify;
                self.ready_queue.push_back(id);
            }
        }

        // --- PHASE 2: Lane 2 (Sequential-Repair Lane) ---
        // For streams needing repair, iterate depth by depth (up to 16 total codes: 1 primary + 15 residuals).
        if !repair_cohort.is_empty() {
            self.metrics.lane2_repair_frames += repair_cohort.len() as u64;

            // In each quantum, complete the repair loop for this frame across active repair streams
            for depth in 0..16 {
                let mut still_repairing = false;

                for &id in &repair_cohort {
                    let stream = self.streams.get_mut(&id).expect("stream exists");

                    if let StreamLane::SequentialRepair {
                        ref mut partial_codes,
                        ref mut next_depth,
                    } = stream.lane
                    {
                        if *next_depth <= depth {
                            self.metrics.total_repair_steps += 1;
                            stream.repair_step_count += 1;

                            if let Some(completed_frame) = stream
                                .generator
                                .step_repair_depth(*next_depth, partial_codes)
                                .map_err(EngineError::Generation)?
                            {
                                stream.code_frames.push(completed_frame);
                                stream.lane = StreamLane::BlockVerify;
                                self.metrics.total_frames_completed += 1;
                                self.ready_queue.push_back(id);
                            } else {
                                *next_depth += 1;
                                still_repairing = true;
                            }
                        } else {
                            still_repairing = true;
                        }
                    }
                }

                if !still_repairing {
                    break;
                }
            }
        }

        Ok(cohort_size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{FrameStep, NormalizationMode, NormalizationTrace};
    use std::time::Duration;

    /// Synthetic generator with configurable speculative acceptance depth per frame.
    struct ConfigurableSpecGenerator {
        id: usize,
        total_frames: usize,
        current_frame: usize,
        speculative_accept_depths: Vec<usize>,
    }

    impl ConfigurableSpecGenerator {
        fn new(id: usize, total_frames: usize, spec_depths: Vec<usize>) -> Self {
            Self {
                id,
                total_frames,
                current_frame: 0,
                speculative_accept_depths: spec_depths,
            }
        }
    }

    impl FrameGenerator for ConfigurableSpecGenerator {
        fn begin_utterance(
            &mut self,
            _prepared: &PreparedText,
            _mode: UtteranceStart,
        ) -> Result<(), GenerationError> {
            self.current_frame = 0;
            Ok(())
        }

        fn append_text(&mut self, _prepared: &PreparedText) -> Result<(), GenerationError> {
            Ok(())
        }

        fn finish_text(&mut self) -> Result<(), GenerationError> {
            Ok(())
        }

        fn next_frame(&mut self) -> Result<FrameStep, GenerationError> {
            if self.current_frame >= self.total_frames {
                return Ok(FrameStep::Finished);
            }
            // Sequential authoritative frame: primary code + 15 residuals
            let mut codes = Vec::with_capacity(16);
            codes.push((self.id * 1000 + self.current_frame) as u32);
            for d in 1..16 {
                codes.push((d * 10 + self.current_frame) as u32);
            }
            self.current_frame += 1;
            Ok(FrameStep::Frame(CodeFrame { codes }))
        }
    }

    impl SpeculativeFrameGenerator for ConfigurableSpecGenerator {
        fn is_finished(&self) -> bool {
            self.current_frame >= self.total_frames
        }

        fn step_speculative_block(&mut self) -> Result<SpeculativeStepOutcome, GenerationError> {
            if self.is_finished() {
                return Ok(SpeculativeStepOutcome::Finished);
            }

            let accept_depth = self
                .speculative_accept_depths
                .get(self.current_frame)
                .copied()
                .unwrap_or(16);

            if accept_depth >= 16 {
                // Full speculative accept
                let mut codes = Vec::with_capacity(16);
                codes.push((self.id * 1000 + self.current_frame) as u32);
                for d in 1..16 {
                    codes.push((d * 10 + self.current_frame) as u32);
                }
                self.current_frame += 1;
                Ok(SpeculativeStepOutcome::FullAccept(CodeFrame { codes }))
            } else {
                // Partial accept up to accept_depth
                let mut accepted = Vec::with_capacity(16);
                if accept_depth > 0 {
                    accepted.push((self.id * 1000 + self.current_frame) as u32);
                }
                for d in 1..accept_depth {
                    accepted.push((d * 10 + self.current_frame) as u32);
                }
                Ok(SpeculativeStepOutcome::PartialAccept {
                    accepted_codes: accepted,
                    accepted_depth: accept_depth,
                })
            }
        }

        fn step_repair_depth(
            &mut self,
            depth: usize,
            partial_codes: &mut Vec<u32>,
        ) -> Result<Option<CodeFrame>, GenerationError> {
            if depth == 0 {
                partial_codes.push((self.id * 1000 + self.current_frame) as u32);
            } else {
                partial_codes.push((depth * 10 + self.current_frame) as u32);
            }

            if partial_codes.len() == 16 {
                self.current_frame += 1;
                Ok(Some(CodeFrame {
                    codes: std::mem::take(partial_codes),
                }))
            } else {
                Ok(None)
            }
        }
    }

    fn sample_prep() -> PreparedText {
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
    fn ragged_dual_lane_preserves_strict_singleton_bit_exactness() {
        let prep = sample_prep();

        // Stream A: full accepts all frames (depth 16)
        // Stream B: ragged partial accepts (frame 0: depth 8, frame 1: depth 16, frame 2: depth 2)
        // Stream C: speculation off (runs purely in Lane 2 repair)
        let spec_a = vec![16, 16, 16];
        let spec_b = vec![8, 16, 2];
        let spec_c = vec![0, 0, 0];

        // 1. Authoritative solo sequential reference outputs
        let mut solo_ref_a = Vec::new();
        let mut gen_a = ConfigurableSpecGenerator::new(1, 3, spec_a.clone());
        gen_a.begin_utterance(&prep, UtteranceStart::Fresh).unwrap();
        while let FrameStep::Frame(f) = gen_a.next_frame().unwrap() {
            solo_ref_a.push(f);
        }

        let mut solo_ref_b = Vec::new();
        let mut gen_b = ConfigurableSpecGenerator::new(2, 3, spec_b.clone());
        gen_b.begin_utterance(&prep, UtteranceStart::Fresh).unwrap();
        while let FrameStep::Frame(f) = gen_b.next_frame().unwrap() {
            solo_ref_b.push(f);
        }

        let mut solo_ref_c = Vec::new();
        let mut gen_c = ConfigurableSpecGenerator::new(3, 3, spec_c.clone());
        gen_c.begin_utterance(&prep, UtteranceStart::Fresh).unwrap();
        while let FrameStep::Frame(f) = gen_c.next_frame().unwrap() {
            solo_ref_c.push(f);
        }

        // 2. Run all three streams together in RaggedBatchScheduler
        let config = BatchSchedulerConfig {
            policy: BatchingPolicy::Throughput {
                max_batch_size: 4,
                queue_delay: Duration::ZERO,
            },
            max_admitted_streams: 8,
            quantum_slice: Duration::from_micros(10),
        };
        let mut scheduler = RaggedBatchScheduler::new(config);

        let id_a = scheduler
            .admit(
                ConfigurableSpecGenerator::new(1, 3, spec_a),
                &prep,
                UtteranceStart::Fresh,
                true, // speculation enabled
            )
            .unwrap();

        let id_b = scheduler
            .admit(
                ConfigurableSpecGenerator::new(2, 3, spec_b),
                &prep,
                UtteranceStart::Fresh,
                true, // speculation enabled (ragged depths)
            )
            .unwrap();

        let id_c = scheduler
            .admit(
                ConfigurableSpecGenerator::new(3, 3, spec_c),
                &prep,
                UtteranceStart::Fresh,
                false, // speculation disabled (Lane 2 pure sequential fallback)
            )
            .unwrap();

        while scheduler.active_stream_count() > 0 {
            scheduler.step_quantum().unwrap();
        }

        let batched_a = scheduler.take_frames(id_a).unwrap();
        let batched_b = scheduler.take_frames(id_b).unwrap();
        let batched_c = scheduler.take_frames(id_c).unwrap();

        // Metamorphic Invariant: Bit-for-bit exactness under ragged speculation!
        assert_eq!(
            batched_a, solo_ref_a,
            "Stream A (Lane 1 fast-path) must match solo sequential decode bit-for-bit"
        );
        assert_eq!(
            batched_b, solo_ref_b,
            "Stream B (Ragged Lane 1 + Lane 2 repair) must match solo sequential decode bit-for-bit"
        );
        assert_eq!(
            batched_c, solo_ref_c,
            "Stream C (Lane 2 pure sequential fallback) must match solo sequential decode bit-for-bit"
        );

        let metrics = scheduler.metrics();
        assert_eq!(metrics.total_frames_completed, 9);
        assert!(metrics.lane1_full_accepts > 0);
        assert!(metrics.lane2_repair_frames > 0);
        assert!(metrics.total_repair_steps > 0);
    }

    #[test]
    fn ab_benchmark_dual_lane_speculation_vs_pure_sequential() {
        let prep = sample_prep();
        let num_streams = 8;
        let frames_per_stream = 10;

        // Realistic acceptance mix (from 3A alpha):
        // 60% full-accept (depth 16), 30% partial-accept (depth 8-12), 10% low-accept (depth 2-4)
        let spec_patterns: Vec<Vec<usize>> = (0..num_streams)
            .map(|s| {
                (0..frames_per_stream)
                    .map(|f| match (s + f) % 10 {
                        0..=5 => 16, // full accept in 1 block step
                        6..=8 => 10, // partial accept
                        _ => 3,      // low accept
                    })
                    .collect()
            })
            .collect();

        // 1. Benchmark Condition A: Dual-Lane with speculation enabled
        let config = BatchSchedulerConfig {
            policy: BatchingPolicy::Throughput {
                max_batch_size: num_streams,
                queue_delay: Duration::ZERO,
            },
            max_admitted_streams: 32,
            quantum_slice: Duration::from_micros(10),
        };
        let mut sched_a = RaggedBatchScheduler::new(config);

        for (s, pattern) in spec_patterns.iter().enumerate().take(num_streams) {
            let generator_inst =
                ConfigurableSpecGenerator::new(s, frames_per_stream, pattern.clone());
            sched_a
                .admit(generator_inst, &prep, UtteranceStart::Fresh, true)
                .unwrap();
        }

        while sched_a.active_stream_count() > 0 {
            sched_a.step_quantum().unwrap();
        }
        let metrics_a = *sched_a.metrics();

        // 2. Benchmark Condition B: Pure sequential batching (speculation disabled)
        let mut sched_b = RaggedBatchScheduler::new(config);

        for (s, pattern) in spec_patterns.iter().enumerate().take(num_streams) {
            let generator_inst =
                ConfigurableSpecGenerator::new(s, frames_per_stream, pattern.clone());
            sched_b
                .admit(generator_inst, &prep, UtteranceStart::Fresh, false)
                .unwrap();
        }

        while sched_b.active_stream_count() > 0 {
            sched_b.step_quantum().unwrap();
        }
        let metrics_b = *sched_b.metrics();

        // Total frames completed is equal across both conditions
        assert_eq!(
            metrics_a.total_frames_completed,
            (num_streams * frames_per_stream) as u64
        );
        assert_eq!(
            metrics_b.total_frames_completed,
            (num_streams * frames_per_stream) as u64
        );

        // Dual-Lane speculation resolves majority of frames in Lane 1
        assert!(
            metrics_a.full_accept_rate() >= 0.50,
            "full accept rate should be >= 50%"
        );
        assert_eq!(
            metrics_b.lane1_full_accepts, 0,
            "Condition B has 0 speculation"
        );

        // Dual-Lane achieves massive reduction in sequential repair steps:
        // Pure sequential evaluated 16 steps/frame * 80 frames = 1280 steps
        // Speculative dual-lane skips ~60% of steps completely
        assert!(metrics_a.total_repair_steps < metrics_b.total_repair_steps);
        let step_reduction =
            1.0 - (metrics_a.total_repair_steps as f64 / metrics_b.total_repair_steps as f64);
        assert!(
            step_reduction >= 0.50,
            "Dual-lane should eliminate at least 50% of sequential repair steps (got {:.1}%)",
            step_reduction * 100.0
        );

        println!(
            "Ragged Dual-Lane A/B Benchmark Results (N={} streams, {} frames/stream):\n\
             - Condition A (Dual-Lane Speculation): {} total repair steps, {:.1}% full accept rate\n\
             - Condition B (Pure Sequential):      {} total repair steps, {:.1}% full accept rate\n\
             - Sequential Step Reduction:          {:.1}%\n\
             - Lane 1 Full Accepts:                {}/{}",
            num_streams,
            frames_per_stream,
            metrics_a.total_repair_steps,
            metrics_a.full_accept_rate() * 100.0,
            metrics_b.total_repair_steps,
            metrics_b.full_accept_rate() * 100.0,
            step_reduction * 100.0,
            metrics_a.lane1_full_accepts,
            metrics_a.total_frames_completed,
        );
    }
}
