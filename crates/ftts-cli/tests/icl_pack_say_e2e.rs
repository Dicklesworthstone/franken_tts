//! End-to-end proof for bead `frankentts-6hdc` tranche 2: a QUALITY-mode `.ftvoice` pack
//! enrolls through `ftts enroll --mode quality`, inspects cleanly, and SYNTHESIZES through
//! `ftts say --voice pack.ftvoice` — exercising the whole streaming-ICL chain (icl_header,
//! per-frame codec sums, wrapped-transcript tokenization, `assemble_prompt`, generation,
//! codec decode) against the real model.
//!
//! Model-gated: every step needs the pinned weights; absence reports an honest skip receipt
//! and passes, never a counterfeit green.
//!
//! The synthetic reference is a voiced-harmonic contour (not silence): enrollment's VAD must
//! find speech energy in it, and the codec encoder must produce real tokens from it. Quality
//! of the clone is explicitly NOT claimed here — this pins MECHANISM (runs, emits parseable
//! WAV, deterministic exit codes), while listening-tier quality belongs to its own protocol.

#![cfg(feature = "ultra-tests")]

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Mutex, MutexGuard, OnceLock};

use ftts_core::audio::WavWriter;

fn emit_skip(test: &str, reason: &str) {
    eprintln!(
        "receipt: {{\"test\":\"{test}\",\"outcome\":\"skipped\",\"reason\":\"{reason}\",\
         \"contract\":\"ICL/pack-say-e2e\",\"seam\":\"enroll.icl_say\"}}"
    );
}

/// Each real-model case creates its own native kernel team. Running both cases concurrently
/// oversubscribes small CI workers and can make a healthy suite trip an inactivity watchdog.
fn model_test_guard() -> MutexGuard<'static, ()> {
    static MODEL_TEST: OnceLock<Mutex<()>> = OnceLock::new();
    MODEL_TEST
        .get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Same resolution order as the other model-gated CLI e2es: FTTS_MODEL_DIR, then the user
/// cache, then the git-ignored truth-pack snapshot.
fn model_dir() -> Option<PathBuf> {
    let mut candidates: Vec<PathBuf> = Vec::new();
    if let Ok(dir) = std::env::var("FTTS_MODEL_DIR") {
        candidates.push(PathBuf::from(dir));
    }
    #[allow(deprecated)]
    let home = std::env::home_dir();
    if let Some(home) = home {
        candidates.push(home.join(".cache/franken_tts/model"));
    }
    candidates
        .push(Path::new(env!("CARGO_MANIFEST_DIR")).join("../../docs/truth-pack/snapshots/hf"));
    candidates.into_iter().find(|root| {
        [
            "vocab.json",
            "merges.txt",
            "tokenizer_config.json",
            "speech_tokenizer/model.safetensors",
        ]
        .iter()
        .all(|required| root.join(required).is_file())
            && (root.join("qwen3-tts-12hz-0.6b-base.fttsq").is_file()
                || root.join("model.safetensors").is_file())
    })
}

fn scratch_dir(label: &str) -> PathBuf {
    // Keep artifacts for post-mortem inspection, but never let Cargo's parallel test runner
    // make one case delete or overwrite another case's live model output.
    let dir = std::env::temp_dir().join(format!(
        "ftts-icl-say-e2e-{label}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |duration| duration.as_nanos())
    ));
    std::fs::create_dir_all(&dir).expect("scratch dir");
    dir
}

/// Deterministic SYLLABIC contour: a voiced-harmonic carrier under a ~2.5 Hz speech-like
/// envelope with near-zero valleys, so the enrollment VAD's hysteresis sees a real
/// pause-floor/voiced-energy spread (a constant tone sits at one energy and never crosses
/// the open threshold — the exit-8 gate correctly refused that fixture).
fn write_reference_wav(path: &Path, seconds: usize, seed: u64) {
    let sample_rate = ftts_core::audio::SAMPLE_RATE_HZ as usize;
    let samples: Vec<f32> = (0..sample_rate * seconds)
        .map(|index| {
            let t = index as f32 / sample_rate as f32;
            let base = (seed as f32 * 13.7) + 90.0;
            let wobble = (2.0 * std::f32::consts::PI * 3.0 * t).sin() * 12.0;
            let f0 = base + wobble;
            let phase = 2.0 * std::f32::consts::PI * f0 * t;
            let carrier = 0.35 * phase.sin()
                + 0.18 * (2.0 * phase).sin()
                + 0.09 * (3.0 * phase).sin()
                + 0.04 * ((index % 97) as f32 / 97.0 - 0.5);
            let envelope = (2.0 * std::f32::consts::PI * 2.5 * t).sin().powi(2);
            carrier * envelope
        })
        .collect();
    let file = std::fs::File::create(path).expect("reference wav");
    let mut writer = WavWriter::new(file, ftts_core::audio::SAMPLE_RATE_HZ).expect("writer");
    writer.write_samples(&samples).expect("samples");
    writer.finish().expect("finish");
}

/// `--model`, `--no-resident`: both are PER-SUBCOMMAND flags and go AFTER the subcommand.
fn run(mut command: Command, what: &str) -> std::process::Output {
    eprintln!("starting ICL end-to-end phase: {what}");
    let output = command
        .output()
        .unwrap_or_else(|error| panic!("spawn {what}: {error}"));
    eprintln!("finished ICL end-to-end phase: {what} ({})", output.status);
    output
}

#[test]
fn quality_pack_enrolls_inspects_and_synthesizes_end_to_end() {
    let _model_test = model_test_guard();
    const TEST: &str = "quality_pack_enrolls_inspects_and_synthesizes_end_to_end";
    let Some(model_root) = model_dir() else {
        emit_skip(TEST, "no pinned model weights on this host");
        return;
    };
    let model = if model_root.join("qwen3-tts-12hz-0.6b-base.fttsq").is_file() {
        model_root.join("qwen3-tts-12hz-0.6b-base.fttsq")
    } else {
        model_root.join("model.safetensors")
    };
    let dir = scratch_dir("quality");

    // 1. Reference audio with real spectral structure.
    let reference = dir.join("reference.wav");
    write_reference_wav(&reference, 4, 7);

    // 2. QUALITY-mode enrollment: transcript + consent land in a Portable pack.
    let pack = dir.join("voice.ftvoice");
    let transcript = "Please call Stella. Ask her to bring these things with her.";
    let enroll = run(
        {
            let mut c = Command::new(env!("CARGO_BIN_EXE_ftts"));
            c.args([
                "enroll",
                reference.to_str().expect("utf8 reference"),
                "--model",
                model.to_str().expect("utf8 model"),
                "-o",
                pack.to_str().expect("utf8 pack"),
                "--mode",
                "quality",
                "--transcript-text",
                transcript,
                "--consent-attest",
                "--no-denoise",
            ]);
            c
        },
        "enroll",
    );
    assert!(
        enroll.status.success(),
        "enroll failed ({}): {}",
        enroll.status,
        String::from_utf8_lossy(&enroll.stderr)
    );
    assert!(pack.is_file(), "enroll wrote no pack");

    // 3. Inspect renders it as a portable, attested, identity-carrying pack. No model flag:
    // inspection reads only the pack bytes.
    let inspect = run(
        {
            let mut c = Command::new(env!("CARGO_BIN_EXE_ftts"));
            c.args(["voice", "inspect", pack.to_str().expect("utf8 pack")]);
            c
        },
        "inspect",
    );
    assert!(inspect.status.success(), "inspect failed");
    let stdout = String::from_utf8_lossy(&inspect.stdout);
    let event_line = stdout.lines().last().expect("json event line");
    let event: serde_json::Value = serde_json::from_str(event_line).expect("json");
    assert_eq!(event["event"], "voice_inspect");
    assert_eq!(event["status"], "ok");
    assert_eq!(event["profile"], "portable");
    assert_eq!(event["codec_codes_present"], true);
    assert_eq!(event["transcript_present"], true);

    // 4. THE TRANCHES-1+2 PROOF: the pack synthesizes through the streaming-ICL path.

    let out_wav = dir.join("icl_out.wav");
    let say = run(
        {
            let mut c = Command::new(env!("CARGO_BIN_EXE_ftts"));
            let say_stdout = dir.join("say.stdout.ndjson");
            c.stdout(std::fs::File::create(&say_stdout).expect("say stdout file"));
            c.args([
                "say",
                "--no-resident",
                "--model",
                model.to_str().expect("utf8 model"),
                "--voice",
                pack.to_str().expect("utf8 pack"),
                "-o",
                out_wav.to_str().expect("utf8 wav"),
                "Hello from the reference continuation.",
            ]);
            c
        },
        "say",
    );
    // Post-mortem FIRST: how far did generation get before any abort?
    let say_log = std::fs::read_to_string(dir.join("say.stdout.ndjson")).unwrap_or_default();
    eprintln!("post-mortem: {} events; last 5:", say_log.lines().count());
    for line in say_log.lines().rev().take(5) {
        eprintln!("  {line}");
    }
    assert!(
        say.status.success(),
        "ICL say failed ({}): {}",
        say.status,
        String::from_utf8_lossy(&say.stderr)
    );
    let wav = std::fs::read(&out_wav).expect("output wav");
    assert!(
        wav.len() > 44,
        "output wav holds no data beyond RIFF header"
    );
    assert_eq!(&wav[..4], b"RIFF", "output must be a parseable WAV");
}

#[test]
fn quick_pack_control_synthesizes_without_icl() {
    let _model_test = model_test_guard();
    // CONTROL for the ICL overflow: identical fixture and flow, but QUICK mode —
    // embedding-only conditioning. If this passes while the ICL test aborts, the fault is
    // in the streaming-ICL path, not in enrollment or the shared pipeline.
    const TEST: &str = "quick_pack_control_synthesizes_without_icl";
    let Some(model_root) = model_dir() else {
        emit_skip(TEST, "no pinned model weights on this host");
        return;
    };
    let model = if model_root.join("qwen3-tts-12hz-0.6b-base.fttsq").is_file() {
        model_root.join("qwen3-tts-12hz-0.6b-base.fttsq")
    } else {
        model_root.join("model.safetensors")
    };
    let dir = scratch_dir("quick");
    let reference = dir.join("reference.wav");
    write_reference_wav(&reference, 4, 7);
    let pack = dir.join("quick.ftvoice");
    let enroll = run(
        {
            let mut c = Command::new(env!("CARGO_BIN_EXE_ftts"));
            c.args([
                "enroll",
                reference.to_str().expect("utf8 reference"),
                "--model",
                model.to_str().expect("utf8 model"),
                "-o",
                pack.to_str().expect("utf8 pack"),
                "--mode",
                "quick",
                "--consent-attest",
                "--no-denoise",
            ]);
            c
        },
        "enroll",
    );
    assert!(enroll.status.success(), "enroll failed: {}", enroll.status);
    let out_wav = dir.join("quick_out.wav");
    let say = run(
        {
            let mut c = Command::new(env!("CARGO_BIN_EXE_ftts"));
            c.args([
                "say",
                "--no-resident",
                "--model",
                model.to_str().expect("utf8 model"),
                "--voice",
                pack.to_str().expect("utf8 pack"),
                "-o",
                out_wav.to_str().expect("utf8 wav"),
                "Hello from the reference continuation.",
            ]);
            c
        },
        "say",
    );
    assert!(say.status.success(), "quick say failed: {}", say.status);
}
