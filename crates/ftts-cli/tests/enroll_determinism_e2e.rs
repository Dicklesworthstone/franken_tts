//! Metamorphic invariant: identical voice input → identical enrolled voice bytes.
//!
//! The hash-stability half of `frankentts-v-metamorphic-0wq`'s invariant list. Enrollment
//! is a pure function of the reference audio and the pinned model weights — decode,
//! optional cleanup, log-mel, speaker encoder, raw little-endian serialization carry no
//! clock, no randomness, and no paths — so enrolling the SAME reference twice MUST
//! produce byte-identical `.spk` files. A single differing byte means hidden state
//! (thread-count-dependent reductions, uninitialized scratch, time-based fields) leaked
//! into the pipeline, and every future voice-card or provenance feature that hashes the
//! pack would inherit the flake.
//!
//! Negative control included: two DIFFERENT references must produce different vectors,
//! so the equality above cannot pass against a constant.
//!
//! Model-gated: enrollment needs the real speaker encoder; absence reports an honest
//! skip and passes.

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use ftts_core::audio::WavWriter;

/// Skip receipts go to stderr as NDJSON in the same shape the conformance crate emits —
/// a skipped test says why, on the record.
fn emit_skip(test: &str, reason: &str) {
    eprintln!(
        "receipt: {{\"test\":\"{test}\",\"outcome\":\"skipped\",\"reason\":\"{reason}\",\
         \"contract\":\"Metamorphic/voice-hash-stability\",\"seam\":\"enroll.voice_pack\"}}"
    );
}

/// Canonical locations first (FTTS_MODEL_DIR, then ~/.cache/franken_tts/model), then the
/// git-ignored in-tree truth-pack snapshot that rsyncs to workers — enrollment must be
/// measurable wherever its checkpoint already is.
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

/// A scratch directory for this test's artifacts. Never deleted — small, uniquely named.
fn scratch_dir(label: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "ftts-enroll-determinism-{label}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos())
    ));
    std::fs::create_dir_all(&dir).expect("scratch dir");
    dir
}

/// Writes a mono 24 kHz WAV with a deterministic synthesized contour: `seed` picks the
/// harmonics so different seeds produce genuinely different voice-shaped signals while
/// the same seed reproduces the exact same PCM.
fn write_reference_wav(path: &std::path::Path, seconds: usize, seed: u64) {
    let sample_rate = ftts_core::audio::SAMPLE_RATE_HZ as usize;
    let samples: Vec<f32> = (0..sample_rate * seconds)
        .map(|index| {
            let t = index as f32 / sample_rate as f32;
            let base = (seed as f32 * 13.7) + 90.0;
            let wobble = (2.0 * std::f32::consts::PI * 3.0 * t).sin() * 12.0;
            let f0 = base + wobble;
            let phase = 2.0 * std::f32::consts::PI * f0 * t;
            0.35 * phase.sin()
                + 0.18 * (2.0 * phase).sin()
                + 0.09 * (3.0 * phase).sin()
                + 0.04 * ((index % 97) as f32 / 97.0 - 0.5)
        })
        .collect();
    let file = std::fs::File::create(path).expect("reference wav");
    let mut writer = WavWriter::new(file, ftts_core::audio::SAMPLE_RATE_HZ).expect("wav writer");
    writer.write_samples(&samples).expect("write samples");
    writer.finish().expect("finish wav");
}
fn enroll(reference: &std::path::Path, output: &std::path::Path) {
    let output = Command::new(env!("CARGO_BIN_EXE_ftts"))
        .args([
            "enroll",
            reference.to_str().expect("utf8 reference"),
            "-o",
            output.to_str().expect("utf8 output"),
            "--no-denoise",
            "--model",
            // The in-tree staged snapshot: the same resolution the gating above found.
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../docs/truth-pack/snapshots/hf/qwen3-tts-12hz-0.6b-base.fttsq")
                .to_str()
                .expect("utf8 model"),
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .expect("ftts spawns");
    assert!(
        output.status.success(),
        "enroll failed with {} for {}: {}",
        output.status,
        reference.display(),
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn identical_reference_enrolls_byte_identical_voice_bytes() {
    const TEST: &str = "identical_reference_enrolls_byte_identical_voice_bytes";
    if model_dir().is_none() {
        emit_skip(TEST, "model not present");
        return;
    }

    let scratch = scratch_dir("same");
    let reference = scratch.join("reference.wav");
    write_reference_wav(&reference, 4, 11);

    // Enroll the SAME input twice, into separate outputs, in separate processes.
    let first = scratch.join("first.spk");
    let second = scratch.join("second.spk");
    enroll(&reference, &first);
    enroll(&reference, &second);

    let first_bytes = std::fs::read(&first).expect("first spk");
    let second_bytes = std::fs::read(&second).expect("second spk");
    assert_eq!(
        first_bytes.len(),
        ftts_cli::synth::SPEAKER_VECTOR_BYTES,
        "enrolled voice is not a raw {}-byte vector",
        ftts_cli::synth::SPEAKER_VECTOR_BYTES
    );
    assert_eq!(
        first_bytes, second_bytes,
        "identical reference produced DIFFERENT voice bytes — nondeterminism leaked into \
         the enrollment pipeline (decode, cleanup, mel, encoder, or serialization)"
    );

    // Negative control: a different voice-shaped input must not enroll to the same bytes.
    let other = scratch.join("other.wav");
    write_reference_wav(&other, 4, 9001);
    let third = scratch.join("third.spk");
    enroll(&other, &third);
    let third_bytes = std::fs::read(&third).expect("third spk");
    assert_ne!(
        first_bytes, third_bytes,
        "different references produced IDENTICAL voice bytes; the equality check above \
         would have been vacuous"
    );

    eprintln!(
        "receipt: {{\"test\":\"{TEST}\",\"outcome\":\"passed\",\"bytes\":{},\"note\":\
         \"two-process enroll of one wav is byte-stable; distinct wav diverges\"}}",
        first_bytes.len()
    );
}
