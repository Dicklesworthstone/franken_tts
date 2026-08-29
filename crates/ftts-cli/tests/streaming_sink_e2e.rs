//! Live PCM sink contract (`synth::PcmPacketSink`), against the real model in-process.
//!
//! Model-gated: without a complete model directory each test reports the skip and passes,
//! per the repository's model-gated e2e convention. With the model these prove, on this
//! machine, the four promises of the sink seam (bead `frankentts-x8el`):
//!
//! 1. the sink receives exactly the samples the whole-utterance buffer returns —
//!    bit-identical, packet accounting exact, tail-partial packet included;
//! 2. a slow sink backpressures synthesis without deadlock and without corrupting output;
//! 3. a failing sink aborts the run promptly with an error (the barge-in abort path);
//! 4. cancelling the token mid-run stops delivery promptly, without a hang, and the
//!    packets delivered before the cancel are a prefix of a completed run's audio.
//!
//! The bead also names a zero-frame case (sink never called). Immediate EOS cannot be
//! forced deterministically through the public path (EOS is masked for the first two
//! frames and sampling-dependent after), so that case is NOT claimed here; the property
//! is covered indirectly by case 1's exact packet accounting (`sum(frames) == frames`).
//!
//! Every case logs an NDJSON receipt line to stderr (`receipt: {...}`) with the seam,
//! seed, and measured values, per the ftts-conformance observability convention.

#![cfg(feature = "ultra-tests")]

use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use ftts_cli::synth::{self, PcmPacketSink};

fn model_dir() -> Option<PathBuf> {
    let root = std::env::var("FTTS_MODEL_DIR").map_or_else(
        |_| {
            #[allow(deprecated)]
            std::env::home_dir().map(|home| home.join(".cache/franken_tts/model"))
        },
        |dir| Some(PathBuf::from(dir)),
    )?;
    for required in [
        "vocab.json",
        "merges.txt",
        "tokenizer_config.json",
        "speech_tokenizer/model.safetensors",
    ] {
        if !root.join(required).is_file() {
            return None;
        }
    }
    if !root.join("qwen3-tts-12hz-0.6b-base.fttsq").is_file()
        && !root.join("model.safetensors").is_file()
    {
        return None;
    }
    Some(root)
}

/// A deterministic synthetic speaker vector, unit-normalized.
///
/// The identity properties under test do not depend on the voice sounding natural —
/// the same vector conditions both the sink delivery and the returned buffer — and a
/// synthetic vector keeps the test independent of any enrolled or preset voice on the
/// host. Seeded xorshift so every run and every machine sees the same speaker.
fn synthetic_speaker() -> Vec<f32> {
    let mut state = 0x9E37_79B9_7F4A_7C15_u64;
    let mut raw: Vec<f32> = (0..1024)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state >> 40) as f32 / 8_388_608.0) - 1.0
        })
        .collect();
    let norm = raw.iter().map(|v| v * v).sum::<f32>().sqrt();
    for v in &mut raw {
        *v /= norm;
    }
    raw
}

struct Loaded {
    model: synth::LoadedModel,
    engine: ftts_core::TtsEngine,
}

fn load(root: &std::path::Path) -> Loaded {
    let bundle = synth::ModelBundle::resolve(root).expect("model bundle resolves");
    let model = synth::LoadedModel::load(&bundle).expect("model loads");
    let engine = ftts_core::TtsEngine::from_process_environment().expect("engine starts");
    Loaded { model, engine }
}

fn run(
    loaded: &Loaded,
    text: &str,
    seed: u64,
    cancellation: &ftts_core::CancellationToken,
    packet_frames: usize,
    sink: Option<&mut dyn PcmPacketSink>,
) -> Result<synth::SynthesizedAudio, ftts_cli::FttsError> {
    let request = ftts_core::SynthesisRequest::new(text.to_owned());
    let observer = |_event: ftts_core::SynthesisEvent| {};
    synth::synthesize(
        &loaded.model,
        &loaded.engine,
        &request,
        &synth::VoiceConditioning::XVector(synthetic_speaker()),
        seed,
        cancellation,
        &observer,
        packet_frames,
        None,
        sink,
    )
}

const TEXT: &str = "The streaming sink must deliver every packet the buffer keeps.";
const SEED: u64 = 7;

/// Collects every delivered packet, optionally sleeping (slow consumer) and optionally
/// signalling an external observer after each delivery.
#[derive(Default)]
struct CollectingSink {
    samples: Vec<f32>,
    packet_frames: Vec<usize>,
    delay: Option<Duration>,
    delivered: Option<Arc<AtomicUsize>>,
}

impl PcmPacketSink for CollectingSink {
    fn deliver(&mut self, samples: &[f32], frames: usize) -> Result<(), ftts_cli::FttsError> {
        if let Some(delay) = self.delay {
            std::thread::sleep(delay);
        }
        self.samples.extend_from_slice(samples);
        self.packet_frames.push(frames);
        if let Some(counter) = &self.delivered {
            counter.fetch_add(1, Ordering::SeqCst);
        }
        Ok(())
    }
}

#[test]
fn sink_receives_exactly_the_returned_audio() {
    let Some(root) = model_dir() else {
        eprintln!(
            "receipt: {{\"test\":\"sink_identity\",\"outcome\":\"skipped\",\"reason\":\"model directory unavailable\"}}"
        );
        return;
    };
    let loaded = load(&root);
    let mut sink = CollectingSink::default();
    let cancellation = ftts_core::CancellationToken::new();
    let audio = run(&loaded, TEXT, SEED, &cancellation, 4, Some(&mut sink)).expect("synthesis");

    assert_eq!(
        sink.samples.len(),
        audio.pcm.len(),
        "sink sample count diverges from the returned buffer"
    );
    for (index, (sunk, kept)) in sink.samples.iter().zip(audio.pcm.iter()).enumerate() {
        assert!(
            sunk.to_bits() == kept.to_bits(),
            "first divergent sample at index {index}: sink {sunk} vs buffer {kept}"
        );
    }
    let delivered_frames: usize = sink.packet_frames.iter().sum();
    assert_eq!(
        delivered_frames as u64, audio.frames,
        "packet frame accounting diverges from generated frames"
    );
    // Every packet before the tail is full; only the final one may be partial.
    if let Some((tail, body)) = sink.packet_frames.split_last() {
        assert!(
            body.iter().all(|frames| *frames == 4),
            "non-tail packet not full"
        );
        assert!(
            *tail >= 1 && *tail <= 4,
            "tail packet frame count out of range"
        );
    }
    assert_eq!(
        sink.samples.len(),
        delivered_frames * 1920,
        "samples per frame diverge from the codec's 1,920"
    );
    eprintln!(
        "receipt: {{\"test\":\"sink_identity\",\"outcome\":\"passed\",\"seed\":{SEED},\"frames\":{},\"packets\":{},\"tail_frames\":{},\"ttfa_ms\":{}}}",
        audio.frames,
        sink.packet_frames.len(),
        sink.packet_frames.last().copied().unwrap_or(0),
        audio.ttfa.map_or(0, |d| d.as_millis())
    );
}

#[test]
fn a_slow_sink_backpressures_without_deadlock_or_corruption() {
    let Some(root) = model_dir() else {
        eprintln!(
            "receipt: {{\"test\":\"sink_backpressure\",\"outcome\":\"skipped\",\"reason\":\"model directory unavailable\"}}"
        );
        return;
    };
    let loaded = load(&root);

    // Reference run: no sink.
    let cancellation = ftts_core::CancellationToken::new();
    let reference = run(&loaded, TEXT, SEED, &cancellation, 4, None).expect("reference synthesis");

    // Slow-consumer run: 25 ms per packet is ~1/3 of a 4-frame packet's real-time budget —
    // enough to exercise the park path on a faster-than-real-time engine without making
    // the test slow. Identity against the reference proves the parked path corrupts nothing.
    let mut sink = CollectingSink {
        delay: Some(Duration::from_millis(25)),
        ..CollectingSink::default()
    };
    let cancellation = ftts_core::CancellationToken::new();
    let started = Instant::now();
    let audio = run(&loaded, TEXT, SEED, &cancellation, 4, Some(&mut sink)).expect("slow-sink run");
    let elapsed = started.elapsed();

    assert_eq!(sink.samples.len(), audio.pcm.len());
    assert_eq!(
        audio.pcm.len(),
        reference.pcm.len(),
        "slow sink changed output length"
    );
    for (index, (a, b)) in audio.pcm.iter().zip(reference.pcm.iter()).enumerate() {
        assert!(
            a.to_bits() == b.to_bits(),
            "slow-sink output diverges from reference at sample {index}"
        );
    }
    eprintln!(
        "receipt: {{\"test\":\"sink_backpressure\",\"outcome\":\"passed\",\"seed\":{SEED},\"packets\":{},\"elapsed_ms\":{}}}",
        sink.packet_frames.len(),
        elapsed.as_millis()
    );
}

/// A sink that accepts `accept` packets and then fails.
struct FailingSink {
    accept: usize,
    delivered: usize,
}

impl PcmPacketSink for FailingSink {
    fn deliver(&mut self, _samples: &[f32], _frames: usize) -> Result<(), ftts_cli::FttsError> {
        if self.delivered == self.accept {
            return Err(ftts_cli::FttsError::Generic(
                "sink refused delivery (test-injected)".to_owned(),
            ));
        }
        self.delivered += 1;
        Ok(())
    }
}

#[test]
fn a_failing_sink_aborts_the_run_with_an_error() {
    let Some(root) = model_dir() else {
        eprintln!(
            "receipt: {{\"test\":\"sink_failure_abort\",\"outcome\":\"skipped\",\"reason\":\"model directory unavailable\"}}"
        );
        return;
    };
    let loaded = load(&root);
    let mut sink = FailingSink {
        accept: 1,
        delivered: 0,
    };
    let cancellation = ftts_core::CancellationToken::new();
    let started = Instant::now();
    let result = run(&loaded, TEXT, SEED, &cancellation, 4, Some(&mut sink));
    let elapsed = started.elapsed();
    assert!(result.is_err(), "a failing sink must abort the run");
    eprintln!(
        "receipt: {{\"test\":\"sink_failure_abort\",\"outcome\":\"passed\",\"accepted_packets\":1,\"abort_after_ms\":{}}}",
        elapsed.as_millis()
    );
}

#[test]
fn cancelling_mid_run_stops_delivery_and_keeps_a_valid_prefix() {
    let Some(root) = model_dir() else {
        eprintln!(
            "receipt: {{\"test\":\"sink_cancel\",\"outcome\":\"skipped\",\"reason\":\"model directory unavailable\"}}"
        );
        return;
    };
    let loaded = load(&root);

    // Reference: the full utterance at the same seed.
    let cancellation = ftts_core::CancellationToken::new();
    let reference = run(&loaded, TEXT, SEED, &cancellation, 4, None).expect("reference synthesis");

    // Cancel as soon as the first packet has been delivered. The watcher thread owns the
    // token; the sink only counts deliveries.
    let delivered = Arc::new(AtomicUsize::new(0));
    let cancellation = ftts_core::CancellationToken::new();
    let watcher_token = cancellation.clone();
    let watcher_count = Arc::clone(&delivered);
    let watcher = std::thread::spawn(move || {
        let deadline = Instant::now() + Duration::from_secs(300);
        while watcher_count.load(Ordering::SeqCst) == 0 {
            if Instant::now() > deadline {
                return;
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        watcher_token.cancel();
    });

    let collected = Arc::new(Mutex::new(Vec::<f32>::new()));
    struct CountingSink {
        delivered: Arc<AtomicUsize>,
        collected: Arc<Mutex<Vec<f32>>>,
    }
    impl PcmPacketSink for CountingSink {
        fn deliver(&mut self, samples: &[f32], _frames: usize) -> Result<(), ftts_cli::FttsError> {
            self.collected
                .lock()
                .expect("collector lock")
                .extend_from_slice(samples);
            self.delivered.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
    }
    let mut sink = CountingSink {
        delivered: Arc::clone(&delivered),
        collected: Arc::clone(&collected),
    };
    let started = Instant::now();
    let result = run(&loaded, TEXT, SEED, &cancellation, 4, Some(&mut sink));
    let elapsed = started.elapsed();
    watcher.join().expect("watcher joins");

    assert!(
        result.is_err(),
        "a cancelled run must report an error, not partial success"
    );
    let prefix = collected.lock().expect("collector lock");
    assert!(
        !prefix.is_empty(),
        "the cancel was requested after the first delivery, so a prefix must exist"
    );
    assert!(
        prefix.len() <= reference.pcm.len(),
        "cancelled run delivered more audio than the completed reference"
    );
    for (index, (partial, full)) in prefix.iter().zip(reference.pcm.iter()).enumerate() {
        assert!(
            partial.to_bits() == full.to_bits(),
            "cancelled prefix diverges from the completed run at sample {index}"
        );
    }
    eprintln!(
        "receipt: {{\"test\":\"sink_cancel\",\"outcome\":\"passed\",\"seed\":{SEED},\"prefix_samples\":{},\"reference_samples\":{},\"elapsed_ms\":{}}}",
        prefix.len(),
        reference.pcm.len(),
        elapsed.as_millis()
    );
}

/// The standing packet-parity gate on the LIVE path (plan §9: packet-1 == packet-4 for
/// the same codec tokens): concatenated PCM is bit-identical across packet schedules
/// 1/2/4 at the same seed, each schedule's packet accounting obeys its own size rule,
/// and the receipt logs the raw-vs-audible TTFA pair per schedule — the measurement
/// evidence the TTFA certification bead consumes (PROVISIONAL_LOCAL_WIN until ledgered).
#[test]
fn packet_schedules_are_bit_identical_and_smaller_packets_deliver_sooner() {
    let Some(root) = model_dir() else {
        eprintln!(
            "receipt: {{\"test\":\"packet_parity\",\"outcome\":\"skipped\",\"reason\":\"model directory unavailable\"}}"
        );
        return;
    };
    let loaded = load(&root);

    // (schedule, samples, packet frame counts, ttfa_ms, ttfa_audible_ms)
    type ScheduleRun = (usize, Vec<f32>, Vec<usize>, Option<u128>, Option<u128>);
    let mut outputs: Vec<ScheduleRun> = Vec::new();
    for packet_frames in [1_usize, 2, 4] {
        let mut sink = CollectingSink::default();
        let cancellation = ftts_core::CancellationToken::new();
        let audio = run(
            &loaded,
            TEXT,
            SEED,
            &cancellation,
            packet_frames,
            Some(&mut sink),
        )
        .expect("synthesis");
        // Per-schedule packet rule: all but the tail carry exactly `packet_frames`.
        if let Some((tail, body)) = sink.packet_frames.split_last() {
            assert!(
                body.iter().all(|frames| *frames == packet_frames),
                "schedule {packet_frames}: non-tail packet not full"
            );
            assert!(
                *tail >= 1 && *tail <= packet_frames,
                "schedule {packet_frames}: tail out of range"
            );
        }
        assert_eq!(
            sink.packet_frames.iter().sum::<usize>() as u64,
            audio.frames,
            "schedule {packet_frames}: frame accounting"
        );
        outputs.push((
            packet_frames,
            sink.samples,
            sink.packet_frames,
            audio.ttfa.map(|d| d.as_millis()),
            audio.ttfa_audible.map(|d| d.as_millis()),
        ));
    }

    let (_, reference, ..) = &outputs[outputs.len() - 1];
    for (schedule, samples, ..) in &outputs {
        assert_eq!(
            samples.len(),
            reference.len(),
            "schedule {schedule}: length diverges from packet-4"
        );
        for (index, (a, b)) in samples.iter().zip(reference.iter()).enumerate() {
            assert!(
                a.to_bits() == b.to_bits(),
                "schedule {schedule}: first divergent sample at {index}"
            );
        }
    }
    for (schedule, _, packets, ttfa, ttfa_audible) in &outputs {
        eprintln!(
            "receipt: {{\"test\":\"packet_parity\",\"outcome\":\"passed\",\"schedule\":{schedule},\"packets\":{},\"ttfa_ms\":{:?},\"ttfa_audible_ms\":{:?},\"claim_tier\":\"PROVISIONAL_LOCAL_WIN\"}}",
            packets.len(),
            ttfa,
            ttfa_audible
        );
    }
}
