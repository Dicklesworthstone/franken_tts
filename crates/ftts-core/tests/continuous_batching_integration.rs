//! Integration tests and throughput measurements for Continuous Frame/Depth Batching (Phase 3D).

use std::time::{Duration, Instant};

use ftts_core::{
    CodeFrame, FrameGenerator, FrameStep, GenerationError, NormalizationMode, NormalizationTrace,
    PreparedText, UtteranceStart,
    batching::{BatchSchedulerConfig, BatchingPolicy, ContinuousBatchScheduler},
};

/// Deterministic synthetic workload generator simulating realistic talker and microdecoder frame costs.
struct SyntheticWorkloadGenerator {
    stream_id: usize,
    total_frames: usize,
    current_frame: usize,
    work_cycles: usize,
}

impl SyntheticWorkloadGenerator {
    fn new(stream_id: usize, total_frames: usize, work_cycles: usize) -> Self {
        Self {
            stream_id,
            total_frames,
            current_frame: 0,
            work_cycles,
        }
    }
}

impl FrameGenerator for SyntheticWorkloadGenerator {
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

        // Simulate frame compute (e.g. arithmetic workload)
        let mut accum = 0u64;
        for i in 0..self.work_cycles {
            accum = accum.wrapping_add((i ^ self.stream_id) as u64);
        }
        std::hint::black_box(accum);

        let mut codes = Vec::with_capacity(16);
        codes.push(((self.stream_id * 1000 + self.current_frame) as u32) ^ (accum as u32 & 0x7));
        for depth in 1..16 {
            codes.push((depth * 100 + self.current_frame) as u32);
        }

        self.current_frame += 1;
        Ok(FrameStep::Frame(CodeFrame { codes }))
    }
}

fn sample_prepared() -> PreparedText {
    PreparedText::new(
        vec![101, 102, 103],
        NormalizationTrace {
            mode: NormalizationMode::Verbatim,
            unicode_version: "16.0".to_owned(),
            changes: Vec::new(),
        },
    )
}

#[test]
fn continuous_batching_multi_stream_bit_exactness() {
    let prepared = sample_prepared();
    let stream_counts = [1, 2, 4, 8];

    for &num_streams in &stream_counts {
        // 1. Run all streams in isolated singleton mode
        let mut solo_outputs = Vec::new();
        for s in 0..num_streams {
            let frames_count = 5 + (s % 4); // variable length per stream
            let mut solo_gen = SyntheticWorkloadGenerator::new(s, frames_count, 500);
            solo_gen
                .begin_utterance(&prepared, UtteranceStart::Fresh)
                .unwrap();
            let mut frames = Vec::new();
            while let FrameStep::Frame(f) = solo_gen.next_frame().unwrap() {
                frames.push(f);
            }
            solo_outputs.push(frames);
        }

        // 2. Run all streams under continuous batching
        let config = BatchSchedulerConfig {
            policy: BatchingPolicy::Throughput {
                max_batch_size: num_streams,
                queue_delay: Duration::from_micros(100),
            },
            max_admitted_streams: 32,
            quantum_slice: Duration::from_micros(10),
        };
        let mut scheduler = ContinuousBatchScheduler::new(config);

        let mut ids = Vec::new();
        for s in 0..num_streams {
            let frames_count = 5 + (s % 4);
            let generator_inst = SyntheticWorkloadGenerator::new(s, frames_count, 500);
            let id = scheduler
                .admit(generator_inst, &prepared, UtteranceStart::Fresh)
                .unwrap();
            ids.push(id);
        }

        while scheduler.active_stream_count() > 0 {
            scheduler.step_quantum().unwrap();
        }

        // Verify exact token parity for every single stream
        for (s, &id) in ids.iter().enumerate() {
            let batched_frames = scheduler.take_frames(id).unwrap();
            assert_eq!(
                batched_frames, solo_outputs[s],
                "stream {s} in batch size {num_streams} failed strict token equivalence"
            );
        }
    }
}

#[test]
fn continuous_batching_throughput_scaling_and_metrics() {
    let prepared = sample_prepared();
    let num_streams = 8;
    let frames_per_stream = 10;
    let work_cycles = 1000;

    // Run isolated sequential baseline
    let solo_start = Instant::now();
    for s in 0..num_streams {
        let mut generator_inst = SyntheticWorkloadGenerator::new(s, frames_per_stream, work_cycles);
        generator_inst.begin_utterance(&prepared, UtteranceStart::Fresh)
            .unwrap();
        while let FrameStep::Frame(_) = generator_inst.next_frame().unwrap() {}
    }
    let solo_duration = solo_start.elapsed();

    // Run continuous batching scheduler
    let config = BatchSchedulerConfig {
        policy: BatchingPolicy::Throughput {
            max_batch_size: num_streams,
            queue_delay: Duration::ZERO,
        },
        max_admitted_streams: 32,
        quantum_slice: Duration::from_micros(10),
    };
    let mut scheduler = ContinuousBatchScheduler::new(config);

    let batch_start = Instant::now();
    for s in 0..num_streams {
        let generator_inst = SyntheticWorkloadGenerator::new(s, frames_per_stream, work_cycles);
        scheduler
            .admit(generator_inst, &prepared, UtteranceStart::Fresh)
            .unwrap();
    }

    while scheduler.active_stream_count() > 0 {
        scheduler.step_quantum().unwrap();
    }
    let batch_duration = batch_start.elapsed();

    let metrics = scheduler.metrics();
    assert_eq!(metrics.peak_batch_size, num_streams);
    assert_eq!(metrics.total_frames_emitted, (num_streams * frames_per_stream) as u64);
    assert!(metrics.mean_batch_size() >= 1.0);

    let bandwidth_savings = metrics.estimated_bandwidth_savings();
    // For batch size 8, estimated memory bandwidth savings = 1 - 1/8 = 87.5%
    assert!(
        bandwidth_savings >= 0.70,
        "bandwidth savings ({bandwidth_savings:.2}) should exceed 70% for batch 8"
    );

    println!(
        "Throughput Measurement (N={num_streams} streams, {frames_per_stream} frames/stream):\n\
         - Isolated Sequential Duration: {:?}\n\
         - Continuous Batching Duration: {:?}\n\
         - Mean Batch Size: {:.2}\n\
         - Theoretical Weight Read Amortization: {:.1}%\n\
         - Total Quanta: {}",
        solo_duration,
        batch_duration,
        metrics.mean_batch_size(),
        bandwidth_savings * 100.0,
        metrics.total_quanta,
    );
}
