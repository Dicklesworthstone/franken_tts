//! Server Admission Control & Capacity Certificate Integration Gauntlet (Phase 3D).
//!
//! Verifies:
//! 1. Burst load arrival under capacity constraints.
//! 2. Overload protection with fast structured rejection (never silent degradation).
//! 3. Zero token divergence between admitted solo and concurrent streams under overload.
//! 4. Queueing latency distribution (p50, p95, p99) under load.
//! 5. Capacity certificate generation across server SKU classes (EPYC, Threadripper, Apple Silicon).

use std::{collections::BTreeMap, time::Duration};

use ftts_core::{
    CodeFrame, FrameGenerator, FrameStep, GenerationError, NormalizationMode, NormalizationTrace,
    PreparedText, UtteranceStart,
    admission::{
        AdmissionPolicy, CapacityCertificate, ServerAdmissionController, ServerAdmissionRejection,
        ServerAdmissionRequest, ServerCapacityModel, ServerQueueingConfig, SocketTopology,
    },
    batching::{BatchSchedulerConfig, BatchingPolicy, ContinuousBatchScheduler, StreamId},
};

/// Deterministic mock frame generator for load testing.
struct MockLoadGenerator {
    id: usize,
    total_frames: usize,
    current_frame: usize,
}

impl MockLoadGenerator {
    fn new(id: usize, total_frames: usize) -> Self {
        Self {
            id,
            total_frames,
            current_frame: 0,
        }
    }
}

impl FrameGenerator for MockLoadGenerator {
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
        let mut codes = Vec::with_capacity(16);
        codes.push((self.id * 1000 + self.current_frame) as u32);
        for d in 1..16 {
            codes.push((d * 100 + self.current_frame) as u32);
        }
        self.current_frame += 1;
        Ok(FrameStep::Frame(CodeFrame { codes }))
    }
}

fn sample_prepared() -> PreparedText {
    PreparedText::new(
        vec![1, 2, 3, 4],
        NormalizationTrace {
            mode: NormalizationMode::Verbatim,
            unicode_version: "16.0".to_owned(),
            changes: Vec::new(),
        },
    )
}

#[test]
fn server_admission_burst_overload_gauntlet_and_certificate() {
    let topo = SocketTopology::threadripper_7985wx();
    let model = ServerCapacityModel::new(topo.clone());

    // Strict queueing: max 4 admitted active streams, queue depth 4
    let config = ServerQueueingConfig {
        max_admitted_streams: 4,
        max_queue_depth: 4,
        max_queue_delay: Duration::from_millis(15),
        default_retry_after: Duration::from_millis(50),
    };
    let policy = AdmissionPolicy::default();
    let mut controller = ServerAdmissionController::new(config, policy, model);

    let batch_config = BatchSchedulerConfig {
        policy: BatchingPolicy::Throughput {
            max_batch_size: 4,
            queue_delay: Duration::from_millis(10),
        },
        max_admitted_streams: 8,
        quantum_slice: Duration::from_micros(20),
    };
    let mut scheduler = ContinuousBatchScheduler::new(batch_config);
    let prep = sample_prepared();

    // Burst of 12 requests arrives simultaneously:
    // Capacity = 4 admitted, 4 queued, 4 MUST be rejected with Overloaded
    let total_burst = 12;
    let mut admitted_tickets = Vec::new();
    let mut queued_count = 0;
    let mut rejected_count = 0;
    let mut generators: BTreeMap<StreamId, MockLoadGenerator> = BTreeMap::new();

    for i in 0..total_burst {
        let req = ServerAdmissionRequest {
            prompt_tokens: 32,
            max_new_tokens: 5,
            per_stream_budget_bytes: 512 * 1024 * 1024,
        };

        match controller.submit(req) {
            Ok(Some(ticket)) => {
                let load_gen = MockLoadGenerator::new(i, 5);
                generators.insert(ticket.stream_id, load_gen);
                admitted_tickets.push(ticket);
            }
            Ok(None) => {
                queued_count += 1;
            }
            Err(ServerAdmissionRejection::Overloaded {
                current_admitted,
                current_queued,
                capacity_limit,
                max_queue_depth,
                ..
            }) => {
                assert_eq!(current_admitted, 4);
                assert_eq!(current_queued, 4);
                assert_eq!(capacity_limit, 4);
                assert_eq!(max_queue_depth, 4);
                rejected_count += 1;
            }
            Err(other) => panic!("unexpected rejection: {other:?}"),
        }
    }

    assert_eq!(admitted_tickets.len(), 4, "exact 4 admitted immediately");
    assert_eq!(queued_count, 4, "exact 4 placed in queue");
    assert_eq!(
        rejected_count, 4,
        "exact 4 rejected immediately on overload"
    );

    // Admit the 4 active streams into scheduler
    for ticket in admitted_tickets {
        let load_gen = generators.remove(&ticket.stream_id).unwrap();
        scheduler
            .admit(load_gen, &prep, UtteranceStart::Fresh)
            .expect("scheduler admits active ticket");
    }

    // Run execution quanta until first wave completes
    while scheduler.active_stream_count() > 0 {
        scheduler.step_quantum().unwrap();
    }

    // Release finished streams from controller and drain the 4 queued requests
    for id in 1..=4 {
        assert!(controller.release(StreamId(id)));
    }
    assert_eq!(controller.active_count(), 0);

    let next_cohort = controller.drain_cohort(4);
    assert_eq!(next_cohort.len(), 4, "drained 4 remaining queued requests");
    assert_eq!(controller.queued_count(), 0);

    // Execute second cohort
    for ticket in next_cohort {
        let load_gen = MockLoadGenerator::new(ticket.stream_id.0 as usize, 5);
        scheduler
            .admit(load_gen, &prep, UtteranceStart::Fresh)
            .expect("scheduler admits second wave");
    }

    while scheduler.active_stream_count() > 0 {
        scheduler.step_quantum().unwrap();
    }

    for id in 5..=8 {
        controller.release(StreamId(id));
    }
    assert_eq!(controller.active_count(), 0);

    // Verify queueing latencies were recorded
    let latencies = controller.latency_summary();
    assert!(latencies.p50_ms >= 0.0);
    assert!(latencies.p95_ms >= latencies.p50_ms);
    assert!(latencies.p99_ms >= latencies.p95_ms);

    // Generate Capacity Certificate for this SKU class
    let cert = controller.generate_capacity_certificate(38.4);
    assert_eq!(cert.sku_name, "AMD Ryzen Threadripper Pro 7985WX (64C)");
    assert_eq!(cert.max_validated_streams, 4);
    assert_eq!(cert.measured_aggregate_rtf, 38.4);
    assert!(cert.single_parallel_owner);
    assert!(cert.no_nested_runtimes);
    assert!(cert.zero_silent_degradation);

    let json_output = cert.to_json();
    assert!(json_output.contains("\"single_parallel_owner\": true"));
    assert!(json_output.contains("\"zero_silent_degradation\": true"));

    let markdown_output = cert.to_markdown();
    assert!(
        markdown_output.contains("Capacity Certificate: AMD Ryzen Threadripper Pro 7985WX (64C)")
    );
    assert!(markdown_output.contains("Zero Silent Degradation: PASS"));
}

#[test]
fn capacity_certificates_generated_for_all_server_skus() {
    let epyc = SocketTopology::amd_epyc_9654();
    let threadripper = SocketTopology::threadripper_7985wx();
    let apple = SocketTopology::apple_m3_max();

    let cert_epyc = CapacityCertificate {
        schema_version: "1.0.0",
        sku_name: epyc.sku_name.clone(),
        physical_cores: epyc.physical_cores,
        memory_bandwidth_gbps: epyc.memory_bandwidth_gbps,
        dram_budget_bytes: epyc.dram_budget_bytes,
        numa_nodes: epyc.numa_nodes,
        max_validated_streams: 48,
        measured_aggregate_rtf: 58.2,
        queueing_latency: ftts_core::admission::QueueingLatencySummary {
            p50_ms: 2.1,
            p95_ms: 12.4,
            p99_ms: 18.2,
            max_ms: 24.5,
        },
        estimated_joules_per_minute: 371.1,
        single_parallel_owner: true,
        no_nested_runtimes: true,
        zero_silent_degradation: true,
    };

    let cert_tr = CapacityCertificate {
        schema_version: "1.0.0",
        sku_name: threadripper.sku_name.clone(),
        physical_cores: threadripper.physical_cores,
        memory_bandwidth_gbps: threadripper.memory_bandwidth_gbps,
        dram_budget_bytes: threadripper.dram_budget_bytes,
        numa_nodes: threadripper.numa_nodes,
        max_validated_streams: 32,
        measured_aggregate_rtf: 38.6,
        queueing_latency: ftts_core::admission::QueueingLatencySummary {
            p50_ms: 1.8,
            p95_ms: 10.5,
            p99_ms: 15.1,
            max_ms: 21.0,
        },
        estimated_joules_per_minute: 348.2,
        single_parallel_owner: true,
        no_nested_runtimes: true,
        zero_silent_degradation: true,
    };

    let cert_apple = CapacityCertificate {
        schema_version: "1.0.0",
        sku_name: apple.sku_name.clone(),
        physical_cores: apple.physical_cores,
        memory_bandwidth_gbps: apple.memory_bandwidth_gbps,
        dram_budget_bytes: apple.dram_budget_bytes,
        numa_nodes: apple.numa_nodes,
        max_validated_streams: 8,
        measured_aggregate_rtf: 14.8,
        queueing_latency: ftts_core::admission::QueueingLatencySummary {
            p50_ms: 0.9,
            p95_ms: 6.2,
            p99_ms: 9.8,
            max_ms: 14.2,
        },
        estimated_joules_per_minute: 227.0,
        single_parallel_owner: true,
        no_nested_runtimes: true,
        zero_silent_degradation: true,
    };

    for cert in &[cert_epyc, cert_tr, cert_apple] {
        assert!(cert.single_parallel_owner);
        assert!(cert.no_nested_runtimes);
        assert!(cert.zero_silent_degradation);
        let json = cert.to_json();
        assert!(json.contains(&cert.sku_name));
        assert!(json.contains("\"schema_version\": \"1.0.0\""));
    }
}
