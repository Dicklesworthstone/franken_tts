//! Warm-engine byte-identity contract: state reused across utterances changes no bytes.
//!
//! Model-gated (bead frankentts-wlvg): without a complete model directory the tests report the
//! skip and pass, per the repository's model-gated e2e convention. With the model they prove,
//! in one process, the invariant the persistent warm engine must hold:
//!
//! 1. the int8 route is built once — not ready before the first utterance, ready after, and
//!    every later utterance borrows it (`LoadedModel::int8_route_ready`);
//! 2. a repeated utterance in a warmed process is bit-identical to the same text+seed on a
//!    freshly loaded model — no ring-buffer samples, sampler RNG, or KV rows leak across
//!    utterances, and route lending produces exactly what a fresh route build would;
//! 3. two different texts interleave before the repeat, so the repeat cannot pass by accident
//!    of adjacency;
//! 4. the soak test (ignored by default; run with
//!    `cargo test --release --test warm_engine_e2e -- --ignored`)
//!    drives 100 utterances and asserts flat RSS via `ps` samples plus per-utterance wall
//!    times on stdout.

#![cfg(feature = "ultra-tests")]

use std::path::PathBuf;
use std::time::Instant;

use ftts_cli::synth::{self, LoadedModel, ModelBundle, SPEAKER_VECTOR_BYTES};
use ftts_core::{CancellationToken, SynthesisEvent, SynthesisRequest, TtsEngine};

const TEXT_A: &str = "Please call Stella. Ask her to bring these things with her from the store.";
const TEXT_B: &str = "When the sunlight strikes raindrops in the air, they act as a prism.";

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
        "qwen3-tts-12hz-0.6b-base.fttsq",
    ] {
        if !root.join(required).is_file() {
            return None;
        }
    }
    Some(root)
}

/// The shipped default preset's x-vector: a real enrolled speaker vector so generation behaves
/// as it does in production. An absent `default.spk` skips the test rather than synthesizing
/// with a fabricated vector that could hit the zero-frame refusal.
fn default_speaker(root: &std::path::Path) -> Option<Vec<f32>> {
    let path = root.join("default.spk");
    let bytes = std::fs::read(&path).ok()?;
    assert_eq!(
        bytes.len(),
        SPEAKER_VECTOR_BYTES,
        "{} is not a speaker vector",
        path.display()
    );
    Some(
        bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|quad| f32::from_le_bytes(*quad))
            .collect(),
    )
}

fn synthesize_utterance(
    label: &str,
    model: &LoadedModel,
    engine: &TtsEngine,
    speaker: &synth::VoiceConditioning,
    text: &str,
    seed: u64,
) -> ftts_cli::synth::SynthesizedAudio {
    let cancellation = CancellationToken::new();
    let observer = |event: SynthesisEvent| {
        let _ = event; // lifecycle events are not under test here
    };
    let started = Instant::now();
    let audio = synth::synthesize(
        model,
        engine,
        &SynthesisRequest::new(text),
        speaker,
        seed,
        &cancellation,
        &observer,
        4,
        None,
        None,
    )
    .expect("utterance synthesizes");
    println!(
        "{{\"receipt\":\"warm_engine\",\"label\":\"{label}\",\"text_chars\":{},\"seed\":{},\"frames\":{},\"ttfa_ms\":{:?},\"wall_ms\":{}}}",
        text.len(),
        seed,
        audio.frames,
        audio.ttfa.map(|d| d.as_millis() as u64),
        started.elapsed().as_millis() as u64,
    );
    audio
}

fn same_samples(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b.iter())
            .all(|(x, y)| x.to_bits() == y.to_bits())
}

#[test]
fn warm_reuse_is_bit_identical_to_a_fresh_model() {
    let Some(root) = model_dir() else {
        eprintln!("SKIP: no complete model directory; warm-engine identity untested here");
        return;
    };
    let Some(speaker) = default_speaker(&root) else {
        eprintln!("SKIP: no default.spk speaker vector beside the model");
        return;
    };
    let bundle = ModelBundle::resolve(&root).expect("bundle resolves");
    let engine = TtsEngine::from_process_environment().expect("engine starts");

    // A warms up across two different texts; B stays cold until the final comparison.
    let warm = LoadedModel::load(&bundle).expect("warm model loads");
    let fresh = LoadedModel::load(&bundle).expect("fresh model loads");
    assert!(!warm.int8_route_ready(), "route must start unbuilt");

    let first = synthesize_utterance(
        "warm_first_a",
        &warm,
        &engine,
        &synth::VoiceConditioning::XVector(speaker.clone()),
        TEXT_A,
        0,
    );
    assert!(warm.int8_route_ready(), "first utterance builds the route");

    let _interleaved = synthesize_utterance(
        "warm_b",
        &warm,
        &engine,
        &synth::VoiceConditioning::XVector(speaker.clone()),
        TEXT_B,
        0,
    );
    let repeat = synthesize_utterance(
        "warm_repeat_a",
        &warm,
        &engine,
        &synth::VoiceConditioning::XVector(speaker.clone()),
        TEXT_A,
        0,
    );
    assert_eq!(repeat.frames, first.frames, "same text+seed, same frames");
    assert!(
        same_samples(&repeat.pcm, &first.pcm),
        "warmed-process repeat diverged from its own first utterance: state leaked across \
         utterances (stale ring buffer, RNG, or KV)"
    );

    // The heart of the bead: a brand-new model (its own route build, own codec state) produces
    // byte-identical audio for the same text and seed.
    let cold = synthesize_utterance(
        "fresh_a",
        &fresh,
        &engine,
        &synth::VoiceConditioning::XVector(speaker.clone()),
        TEXT_A,
        0,
    );
    assert!(
        same_samples(&cold.pcm, &first.pcm),
        "fresh-model synthesis diverged from warmed-process synthesis of the same text+seed"
    );
}

#[test]
#[ignore = "soak: run with `cargo test --release --test warm_engine_e2e -- --ignored`; ~100 warm utterances"]
fn soaked_hundred_utterances_hold_rss_flat() {
    let Some(root) = model_dir() else {
        eprintln!("SKIP: no complete model directory");
        return;
    };
    let Some(speaker) = default_speaker(&root) else {
        eprintln!("SKIP: no default.spk speaker vector beside the model");
        return;
    };
    let bundle = ModelBundle::resolve(&root).expect("bundle resolves");
    let model = LoadedModel::load(&bundle).expect("model loads");
    let engine = TtsEngine::from_process_environment().expect("engine starts");
    let pid = std::process::id();

    let mut rss_samples: Vec<u64> = Vec::new();
    for utterance in 0..100u64 {
        let text = if utterance % 2 == 0 { TEXT_A } else { TEXT_B };
        synthesize_utterance(
            "soak",
            &model,
            &engine,
            &synth::VoiceConditioning::XVector(speaker.clone()),
            text,
            utterance % 7,
        );
        if utterance % 10 == 9 {
            let output = std::process::Command::new("ps")
                .args(["-o", "rss=", "-p", &pid.to_string()])
                .output()
                .expect("ps runs");
            let rss_kb: u64 = String::from_utf8_lossy(&output.stdout)
                .trim()
                .parse()
                .expect("ps prints a number");
            rss_samples.push(rss_kb);
            println!("{{\"receipt\":\"soak_rss\",\"utterance\":{utterance},\"rss_kb\":{rss_kb}}}");
        }
    }

    let low = rss_samples.iter().copied().min().expect("samples exist");
    let high = rss_samples.iter().copied().max().expect("samples exist");
    let drift_mb = (high - low) / 1024;
    println!("{{\"receipt\":\"soak_drift_mb\",\"drift_mb\":{drift_mb}}}");
    assert!(
        drift_mb < 256,
        "RSS drifted {drift_mb} MiB across the soak; retained per-utterance allocations are \
         leaking"
    );
}

/// The dcfn certification bench: warm in-process TTFA under the interactive profile.
///
/// Protocol per frankentts-dcfn (doctrine 8): WARM means utterance N >= 2 in one process with
/// the int8 route already built — the first utterance of each class is a discarded warmup. The
/// interactive profile is one 80 ms frame per packet (`packet_frames = 1`). The product metric
/// is `ttfa_audible` (first delivered packet whose samples cross AUDIBLE_FLOOR); `ttfa` (first
/// byte) is reported alongside because the two definitions can differ by leading silence. A
/// pinned seed keeps every run's frame count identical, so the only varying quantity is time.
/// The corpus is a short first clause and a long paragraph, reported SEPARATELY: TTFA and RTF
/// are different products and short-utterance numbers must never be averaged into long-form
/// ones. This test asserts nothing about the absolute number — the 200 ms figure is a target,
/// not a gate; it exists to produce the receipts the PERF_LEDGER entry cites.
#[test]
#[ignore = "ttfa certification: cargo test --release --test warm_engine_e2e ttfa_certification -- --ignored; ~50 warm utterances"]
fn ttfa_certification_warm_interactive_produces_the_dcfn_receipts() {
    const RUNS_PER_CLASS: usize = 24;
    const SEED: u64 = 0x5EED_0001;
    const SHORT: &str = "Please call Stella.";
    const LONG: &str = "Please call Stella. Ask her to bring these things with her from the store: \
six spoons of fresh snow peas, five thick slabs of blue cheese, and maybe a snack for her brother \
Bob. We also need a small plastic snake and a big toy frog for the kids. She can scoop these \
things into three red bags, and we will go meet her Wednesday at the train station. When the \
sunlight strikes raindrops in the air, they act as a prism and form a rainbow.";

    let Some(root) = model_dir() else {
        eprintln!("SKIP: no complete model directory");
        return;
    };
    let Some(speaker) = default_speaker(&root) else {
        eprintln!("SKIP: no default.spk speaker vector beside the model");
        return;
    };
    let bundle = ModelBundle::resolve(&root).expect("bundle resolves");
    let model = LoadedModel::load(&bundle).expect("model loads");
    let engine = TtsEngine::from_process_environment().expect("engine starts");

    let cancellation = CancellationToken::new();
    let observer = |event: SynthesisEvent| {
        let _ = event;
    };

    for (class, text) in [("short", SHORT), ("long", LONG)] {
        // Discarded warmup: builds the int8 route on first use and warms allocator paths, so
        // every measured run is utterance N >= 2 in this process.
        synth::synthesize(
            &model,
            &engine,
            &SynthesisRequest::new(text),
            &synth::VoiceConditioning::XVector(speaker.to_vec()),
            SEED,
            &cancellation,
            &observer,
            1,
            None,
            None,
        )
        .expect("warmup utterance synthesizes");

        let mut audible_ms: Vec<f64> = Vec::with_capacity(RUNS_PER_CLASS);
        let mut raw_ms: Vec<f64> = Vec::with_capacity(RUNS_PER_CLASS);
        for run in 0..RUNS_PER_CLASS {
            let started = Instant::now();
            let audio = synth::synthesize(
                &model,
                &engine,
                &SynthesisRequest::new(text),
                &synth::VoiceConditioning::XVector(speaker.clone()),
                SEED,
                &cancellation,
                &observer,
                1,
                None,
                None,
            )
            .expect("measured utterance synthesizes");
            let wall = started.elapsed().as_secs_f64() * 1000.0;
            let audible = audio
                .ttfa_audible
                .expect("interactive packets deliver audible audio")
                .as_secs_f64()
                * 1000.0;
            let raw = audio.ttfa.map_or(f64::NAN, |d| d.as_secs_f64() * 1000.0);
            let audio_ms = audio.pcm.len() as f64 / 24.0; // 24 kHz f32 samples → ms.
            audible_ms.push(audible);
            raw_ms.push(raw);
            println!(
                "{{\"receipt\":\"ttfa_cert\",\"class\":\"{class}\",\"run\":{run},\"seed\":{SEED},\
\"frames\":{},\"ttfa_audible_ms\":{audible:.3},\"ttfa_first_byte_ms\":{raw:.3},\"wall_ms\":{wall:.3},\
\"audio_ms\":{audio_ms:.3},\"rtf\":{:.4}}}",
                audio.frames,
                wall / audio_ms
            );
        }

        let mean = audible_ms.iter().sum::<f64>() / RUNS_PER_CLASS as f64;
        let variance = audible_ms
            .iter()
            .map(|value| (value - mean) * (value - mean))
            .sum::<f64>()
            / (RUNS_PER_CLASS - 1) as f64;
        let cv_pct = variance.sqrt() / mean * 100.0;
        let raw_mean = raw_ms.iter().sum::<f64>() / RUNS_PER_CLASS as f64;
        println!(
            "{{\"receipt\":\"ttfa_summary\",\"class\":\"{class}\",\"runs\":{RUNS_PER_CLASS},\
\"packet_frames\":1,\"ttfa_audible_mean_ms\":{mean:.3},\"ttfa_audible_cv_pct\":{cv_pct:.2},\
\"ttfa_first_byte_mean_ms\":{raw_mean:.3}}}"
        );
    }
}

/// Metamorphic invariant (frankentts-v-metamorphic-0wq #2): the packet size is a delivery
/// dial, not an arithmetic one. The same text and seed decoded one frame at a time must be
/// bit-identical to the same stream decoded in four-frame packets — codec state is carried
/// across `stream_push` calls, so a packet boundary that leaked into the arithmetic would show
/// up here as divergent samples rather than as a failed accounting assert.
#[test]
fn packet_one_equals_packet_four_bit_for_bit() {
    const TEXT: &str = "Please call Stella. Ask her to bring these things with her.";
    let Some(root) = model_dir() else {
        eprintln!("SKIP: no complete model directory");
        return;
    };
    let Some(speaker) = default_speaker(&root) else {
        eprintln!("SKIP: no default.spk speaker vector beside the model");
        return;
    };
    let bundle = ModelBundle::resolve(&root).expect("bundle resolves");
    let model = LoadedModel::load(&bundle).expect("model loads");
    let engine = TtsEngine::from_process_environment().expect("engine starts");
    let cancellation = CancellationToken::new();
    let observer = |event: SynthesisEvent| {
        let _ = event;
    };

    let mut by_packet: Vec<(usize, Vec<u32>)> = Vec::new();
    for packet_frames in [1usize, 4] {
        let audio = synth::synthesize(
            &model,
            &engine,
            &SynthesisRequest::new(TEXT),
            &synth::VoiceConditioning::XVector(speaker.to_vec()),
            0x5EED_0002,
            &cancellation,
            &observer,
            packet_frames,
            None,
            None,
        )
        .expect("utterance synthesizes");
        by_packet.push((
            audio.frames as usize,
            audio.pcm.iter().map(|value| value.to_bits()).collect(),
        ));
    }

    let (frames_one, pcm_one) = (&by_packet[0].0, &by_packet[0].1);
    let (frames_four, pcm_four) = (&by_packet[1].0, &by_packet[1].1);
    assert_eq!(
        frames_one, frames_four,
        "packet size changed the frame count"
    );
    assert_eq!(
        pcm_one, pcm_four,
        "packet size changed the samples: streaming arithmetic depends on the \
         packet boundary, which breaks the streaming==batch contract"
    );
}
