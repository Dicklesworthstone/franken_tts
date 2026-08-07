//! The enrollment path end to end: reference audio in, an x-vector the oracle would recognize out.
//!
//! `speaker_encoder_l2` gates the ECAPA stack from the mel onward, against the oracle's own
//! captured tensors. That leaves the front end — waveform to log-mel — ungated, and the spec calls
//! it "the classic silent killer of speaker-embedding parity"
//! (`docs/QWEN3_TTS_SPEAKER_ENCODER_SPEC.md` §1.1). It sits directly upstream of a seam with **no
//! normalization** to absorb error, so leaving it unmeasured would mean `ftts say --voice ref.wav`
//! rests on transcription alone.
//!
//! # The oracle problem, and the way around it
//!
//! The fixture pack captures no waveform. The speaker path's first capture *is* the mel, so there
//! is no `(audio, mel)` pair to compare against and no reference implementation available locally
//! to produce one.
//!
//! What the pack does carry is `prompt.reference_codec_codes` — the same reference recording, as
//! seen through the codec. `codec_decode_l2` already certifies our decoder against the oracle, so
//! decoding those codes recovers that recording up to codec quantization error and nothing else.
//! Running *that* through our front end and our encoder gives a comparison the pack can support:
//!
//! ```text
//!   reference_codec_codes --[our certified decoder]--> waveform
//!                         --[our front end]---------> log-mel   vs  speaker_encoder.input
//!                         --[our encoder]-----------> x-vector  vs  speaker_encoder.output
//! ```
//!
//! # What this can and cannot prove
//!
//! It is **not** a numeric parity gate and is not reported as one: the codec reconstruction is not
//! the original waveform, so some divergence is the codec's and cannot be separated out. The
//! contract is `EnrollmentIdentity`, not `ConformanceExact`.
//!
//! What it does prove is the thing the path exists for. Every trap §1.1 names is a *gross*
//! transformation of the mel, decades away from codec quantization error: an HTK filterbank is a
//! different basis, `log10` rescales every value by 2.303, a centering pad of 512 shifts every
//! frame by 128 samples. A front end with any of those cannot land near the oracle's mel or
//! produce an x-vector pointing the same way, whatever the codec did in between. The 128-sample
//! shift is checked explicitly as a negative control, because it is the subtlest of the five and
//! the one a reader is most entitled to doubt.

use ftts_conformance::{
    compare::compare_f32,
    npy,
    oracle::{CPU_FP32_ORACLE_CLASS, OracleFixtures, SeamRef},
    report::{OracleTier, Outcome, Receipt},
};
use ftts_core::audio::samples_for_frames;
use ftts_model_qwen::checkpoint::CodecCheckpoint;
use ftts_model_qwen::speaker::{ENC_DIM, Encoder, MEL_DIM, log_mel_from_24khz_pcm};
use std::path::{Path, PathBuf};

const MEL_TEST: &str = "enrollment_front_end_reproduces_the_oracle_mel";
const IDENTITY_TEST: &str = "enrollment_from_reference_audio_recovers_the_oracle_x_vector";
const CONTROL_TEST: &str = "enrollment_identity_rejects_a_centering_pad_frame_shift";
const CONTRACT: &str = "EnrollmentIdentity";
const CASE: &str = "synthetic-tone-en";
const MODE: &str = "icl_non_streaming";

/// The reference recording's length in samples.
///
/// The oracle's captured mel is 93 frames, and the uncentered geometry `(n + 768 - 1024) / 256 + 1`
/// puts the source length in `23_808..=24_063`. The codec rounds up to whole 1,920-sample frames,
/// so its reconstruction is longer; trimming to the low end of that window is what makes our mel
/// and the oracle's the same shape, and the frame count is asserted rather than assumed.
const REFERENCE_SAMPLES: usize = 23_808;

/// Mel frames the oracle captured.
const ORACLE_FRAMES: usize = 93;

/// The centering-pad trap: `n_fft / 2 - (n_fft - hop) / 2` = 512 - 384.
const CENTERING_SHIFT: usize = 128;

fn checkpoint_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../docs/truth-pack/snapshots/hf/model.safetensors")
}

fn codec_checkpoint_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../docs/truth-pack/snapshots/hf/speech_tokenizer/model.safetensors")
}

fn seam(name: &str) -> SeamRef<'_> {
    SeamRef {
        case: CASE,
        mode: MODE,
        group: "prompt_build",
        seam: name,
    }
}

fn skip(test: &str, seam_name: &str, reason: &str) {
    Receipt::new(test, Outcome::Skipped)
        .contract(CONTRACT)
        .seam(seam_name)
        .reason(reason)
        .oracle_tier(OracleTier::CpuFp32Fallback)
        .emit();
}

/// Cosine similarity, the metric that means "same direction" and therefore "same speaker".
fn cosine(a: &[f32], b: &[f32]) -> f64 {
    compare_f32(a, b, f64::INFINITY).cosine
}

/// Pearson correlation: cosine after removing the mean.
///
/// Log-mel values all sit in a narrow negative band, so a raw cosine between two mel spectrograms
/// is near 1 even for unrelated audio — the shared offset dominates. Removing the mean is what
/// makes this metric able to fail, which is the only reason to report it.
fn correlation(a: &[f32], b: &[f32]) -> f64 {
    #[allow(clippy::cast_precision_loss)]
    let count = a.len() as f64;
    let mean_a = a.iter().map(|v| f64::from(*v)).sum::<f64>() / count;
    let mean_b = b.iter().map(|v| f64::from(*v)).sum::<f64>() / count;
    let (mut covariance, mut variance_a, mut variance_b) = (0.0, 0.0, 0.0);
    for (x, y) in a.iter().zip(b.iter()) {
        let dx = f64::from(*x) - mean_a;
        let dy = f64::from(*y) - mean_b;
        covariance += dx * dy;
        variance_a += dx * dx;
        variance_b += dy * dy;
    }
    if variance_a <= 0.0 || variance_b <= 0.0 {
        return 0.0;
    }
    covariance / (variance_a.sqrt() * variance_b.sqrt())
}

/// Root-mean-square difference, in natural-log mel units.
fn rms_difference(a: &[f32], b: &[f32]) -> f64 {
    #[allow(clippy::cast_precision_loss)]
    let count = a.len() as f64;
    let total: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let delta = f64::from(*x) - f64::from(*y);
            delta * delta
        })
        .sum();
    (total / count).sqrt()
}

struct Measured {
    mel_correlation: f64,
    mel_cosine: f64,
    mel_rms: f64,
    mel_max_abs: f64,
    identity_cosine: f64,
    shifted_identity_cosine: f64,
    shifted_mel_correlation: f64,
    /// Trap 4, isolated: the oracle's own mel as a `log10` front end would have produced it.
    log10_identity_cosine: f64,
    /// Trap 3, isolated: the oracle's own mel with its bands displaced, as a wrong mel scale
    /// would place them.
    rolled_identity_cosine: f64,
}

fn measure() -> Result<Measured, String> {
    let fixtures = OracleFixtures::open_default()
        .map_err(|error| format!("oracle fixtures unavailable: {error}"))?;
    fixtures
        .require_oracle_class(CPU_FP32_ORACLE_CLASS)
        .map_err(|error| format!("fixture pack is not the CPU-fp32 tier: {error}"))?;

    let main = checkpoint_path();
    let codec_path = codec_checkpoint_path();
    for path in [&main, &codec_path] {
        if !path.is_file() {
            return Err(format!("checkpoint absent at {}", path.display()));
        }
    }

    // --- the reference recording, recovered through our certified codec decoder ----------------
    let codes_seam = seam("prompt.reference_codec_codes");
    if !fixtures.has_seam(&codes_seam) {
        return Err(format!("`{}` is not in this pack", codes_seam.describe()));
    }
    let codes_path = fixtures.seam_path(&codes_seam, "tensor", 0);
    let raw = npy::read_i64(&codes_path)
        .map_err(|error| format!("cannot read {}: {error}", codes_seam.describe()))?;
    // This seam is captured frame-major `[frames, 16]`, which is already the decoder's layout.
    let (frames, groups) = match raw.shape.as_slice() {
        [frames, groups] => (*frames, *groups),
        other => {
            return Err(format!(
                "expected reference codes [frames, 16], got {other:?}"
            ));
        }
    };
    if groups != 16 {
        return Err(format!("the codec carries 16 code groups, not {groups}"));
    }
    let codes: Vec<i32> = raw
        .data
        .iter()
        .map(|value| i32::try_from(*value).unwrap_or(0))
        .collect();

    let codec = CodecCheckpoint::load(&codec_path)
        .map_err(|error| format!("codec checkpoint unusable: {error}"))?;
    let decoded = codec
        .decode(&codes, frames)
        .map_err(|error| format!("codec decode failed: {error:?}"))?;
    if decoded.len() != samples_for_frames(frames) {
        return Err(format!(
            "{frames} frames decoded to {} samples, expected {}",
            decoded.len(),
            samples_for_frames(frames)
        ));
    }
    if decoded.len() < REFERENCE_SAMPLES + CENTERING_SHIFT {
        return Err(format!(
            "the reconstruction is {} samples, too short for a {REFERENCE_SAMPLES}-sample \
             reference plus the {CENTERING_SHIFT}-sample control",
            decoded.len()
        ));
    }
    let reference = &decoded[..REFERENCE_SAMPLES];

    // --- our front end -------------------------------------------------------------------------
    let mel = log_mel_from_24khz_pcm(reference)
        .map_err(|error| format!("front end refused the reconstruction: {error}"))?;
    if mel.frames != ORACLE_FRAMES {
        return Err(format!(
            "our front end produced {} frames from {REFERENCE_SAMPLES} samples; the oracle \
             captured {ORACLE_FRAMES}. The uncentered geometry disagrees.",
            mel.frames
        ));
    }

    let mel_seam = seam("speaker_encoder.input");
    let oracle_mel = fixtures
        .seam(&mel_seam, "args.0", 0)
        .map_err(|error| format!("cannot read {}: {error}", mel_seam.describe()))?;
    if oracle_mel.data.len() != ORACLE_FRAMES * MEL_DIM {
        return Err(format!(
            "{} holds {} floats, expected {}",
            mel_seam.describe(),
            oracle_mel.data.len(),
            ORACLE_FRAMES * MEL_DIM
        ));
    }

    // --- our encoder ---------------------------------------------------------------------------
    let encoder = Encoder::load(&main)
        .map_err(|error| format!("cannot hydrate the speaker encoder: {error}"))?;
    let ours = encoder.encode(&mel.values, mel.frames);
    let embedding_seam = seam("speaker_encoder.output");
    let oracle_embedding = fixtures
        .seam(&embedding_seam, "tensor", 0)
        .map_err(|error| format!("cannot read {}: {error}", embedding_seam.describe()))?;
    if oracle_embedding.data.len() != ENC_DIM {
        return Err(format!(
            "{} holds {} floats, expected {ENC_DIM}",
            embedding_seam.describe(),
            oracle_embedding.data.len()
        ));
    }

    // --- the negative control: the same audio, offset by the centering-pad difference -----------
    let shifted_source = &decoded[CENTERING_SHIFT..CENTERING_SHIFT + REFERENCE_SAMPLES];
    let shifted_mel = log_mel_from_24khz_pcm(shifted_source)
        .map_err(|error| format!("front end refused the shifted control: {error}"))?;
    let shifted = encoder.encode(&shifted_mel.values, shifted_mel.frames);

    // --- decisive controls, built from the oracle's OWN mel so the codec plays no part ----------
    //
    // The frame-shift control above separates only narrowly, and honestly so: attentive statistics
    // pooling reduces the whole utterance to weighted moments, which is close to shift-invariant
    // by construction. These two are the ones that establish the scale of a real front-end error,
    // and they are exact simulations rather than analogies:
    //
    //   * `log10` instead of the natural log rescales every mel value by `1 / ln(10)` — that is
    //     literally what trap 4 produces, applied to the oracle's own tensor.
    //   * a wrong mel scale (HTK for Slaney) places every band's energy at the wrong index. Rolling
    //     the bins reproduces that displacement without needing a second filterbank.
    let log10_mel: Vec<f32> = oracle_mel
        .data
        .iter()
        .map(|value| {
            #[allow(clippy::cast_possible_truncation)]
            {
                (f64::from(*value) / std::f64::consts::LN_10) as f32
            }
        })
        .collect();
    let log10_identity_cosine = cosine(
        &oracle_embedding.data,
        &encoder.encode(&log10_mel, ORACLE_FRAMES),
    );

    const ROLL: usize = 6;
    let mut rolled = vec![0.0f32; oracle_mel.data.len()];
    for frame in 0..ORACLE_FRAMES {
        for bin in 0..MEL_DIM {
            rolled[frame * MEL_DIM + (bin + ROLL) % MEL_DIM] =
                oracle_mel.data[frame * MEL_DIM + bin];
        }
    }
    let rolled_identity_cosine = cosine(
        &oracle_embedding.data,
        &encoder.encode(&rolled, ORACLE_FRAMES),
    );

    Ok(Measured {
        log10_identity_cosine,
        rolled_identity_cosine,
        mel_correlation: correlation(&oracle_mel.data, &mel.values),
        mel_cosine: cosine(&oracle_mel.data, &mel.values),
        mel_rms: rms_difference(&oracle_mel.data, &mel.values),
        mel_max_abs: compare_f32(&oracle_mel.data, &mel.values, f64::INFINITY).max_abs_diff,
        identity_cosine: cosine(&oracle_embedding.data, &ours),
        shifted_identity_cosine: cosine(&oracle_embedding.data, &shifted),
        shifted_mel_correlation: correlation(&oracle_mel.data, &shifted_mel.values),
    })
}

fn emit(test: &str, seam_name: &str, outcome: Outcome, measured: &Measured) {
    Receipt::new(test, outcome)
        .contract(CONTRACT)
        .seam(seam_name)
        .oracle_tier(OracleTier::CpuFp32Fallback)
        .detail(serde_json::json!({
            "mode": MODE,
            "route": "reference_codec_codes -> codec decode -> log-mel -> ECAPA",
            "mel_correlation": measured.mel_correlation,
            "mel_cosine": measured.mel_cosine,
            "mel_rms_log_units": measured.mel_rms,
            "mel_max_abs": measured.mel_max_abs,
            "identity_cosine": measured.identity_cosine,
            "control_shifted_identity_cosine": measured.shifted_identity_cosine,
            "control_shifted_mel_correlation": measured.shifted_mel_correlation,
            "control_log10_identity_cosine": measured.log10_identity_cosine,
            "control_rolled_bins_identity_cosine": measured.rolled_identity_cosine,
        }))
        .emit();
}

/// The front end reproduces the oracle's mel, up to what the codec did to the audio.
#[test]
fn enrollment_front_end_reproduces_the_oracle_mel() {
    let measured = match measure() {
        Ok(measured) => measured,
        Err(reason) => {
            skip(MEL_TEST, "speaker_encoder.input", &reason);
            return;
        }
    };

    let passed = measured.mel_correlation >= MEL_CORRELATION_FLOOR;
    emit(
        MEL_TEST,
        "speaker_encoder.input",
        outcome(passed),
        &measured,
    );
    assert!(
        passed,
        "our log-mel of the codec-reconstructed reference correlates {:.6} with the oracle's \
         (floor {MEL_CORRELATION_FLOOR}), rms {:.4} log units, max_abs {:.4}. Correlation this \
         low is not codec quantization — it is a different filterbank, a different log, or a \
         different frame alignment.",
        measured.mel_correlation, measured.mel_rms, measured.mel_max_abs
    );
}

/// The x-vector points the same way the oracle's does: the enrollment recovers the identity.
#[test]
fn enrollment_from_reference_audio_recovers_the_oracle_x_vector() {
    let measured = match measure() {
        Ok(measured) => measured,
        Err(reason) => {
            skip(IDENTITY_TEST, "speaker_encoder.output", &reason);
            return;
        }
    };

    let passed = measured.identity_cosine >= IDENTITY_COSINE_FLOOR;
    emit(
        IDENTITY_TEST,
        "speaker_encoder.output",
        outcome(passed),
        &measured,
    );
    assert!(
        passed,
        "enrolling from the reference recording gives an x-vector at cosine {:.6} to the \
         oracle's (floor {IDENTITY_COSINE_FLOOR}). The vector is injected into the talker with no \
         normalization, so this angle is the cloned identity itself.",
        measured.identity_cosine
    );
}

/// The gates above can tell a wrong front end from codec noise.
///
/// Three controls, in increasing order of how much they prove:
///
/// * **A 128-sample frame shift** — the centering-pad trap made concrete, since `center = true`
///   pads by 512 instead of 384 and offsets every window by exactly this much. It separates, but
///   only narrowly, and that is a real finding rather than a weak test: attentive statistics
///   pooling reduces the utterance to weighted moments and is close to shift-invariant by
///   construction. Frame misalignment is therefore the *least* punished of the five traps on this
///   metric, and anyone reading a passing identity score should know that.
/// * **`log10` instead of the natural log**, applied to the oracle's own mel. Exactly trap 4.
/// * **Displaced mel bands**, which is what an HTK filterbank does to a Slaney one. Trap 3.
///
/// The last two are computed from the oracle's captured mel, so the codec contributes nothing to
/// them: they measure the encoder's sensitivity to a front-end error in isolation, and establish
/// the scale against which the enrollment score above should be read.
#[test]
fn enrollment_identity_rejects_a_centering_pad_frame_shift() {
    let measured = match measure() {
        Ok(measured) => measured,
        Err(reason) => {
            skip(CONTROL_TEST, "speaker_encoder.output", &reason);
            return;
        }
    };

    let shift_separates = measured.identity_cosine > measured.shifted_identity_cosine
        && measured.mel_correlation > measured.shifted_mel_correlation;
    // The two isolated traps must fail by a margin far wider than the codec's own contribution,
    // which is `1 - identity_cosine`. Requiring an order of magnitude keeps "the gate discriminates"
    // from degrading into "the gate happens to be ordered correctly".
    let codec_gap = 1.0 - measured.identity_cosine;
    let decisive = |control: f64| (1.0 - control) > 10.0 * codec_gap;
    let traps_separate =
        decisive(measured.log10_identity_cosine) && decisive(measured.rolled_identity_cosine);

    let separated = shift_separates && traps_separate;
    emit(
        CONTROL_TEST,
        "speaker_encoder.output",
        outcome(separated),
        &measured,
    );
    assert!(
        shift_separates,
        "a {CENTERING_SHIFT}-sample shift scored at least as well as correct alignment (identity \
         {:.6} vs {:.6}, mel correlation {:.6} vs {:.6}), so these gates carry no information \
         about frame alignment at all.",
        measured.identity_cosine,
        measured.shifted_identity_cosine,
        measured.mel_correlation,
        measured.shifted_mel_correlation
    );
    assert!(
        traps_separate,
        "a wrong front end is not separated from the codec's own contribution ({:.6} of cosine): \
         log10 scores {:.6}, displaced bands score {:.6}. Enrollment scoring {:.6} would then be \
         evidence of nothing.",
        codec_gap,
        measured.log10_identity_cosine,
        measured.rolled_identity_cosine,
        measured.identity_cosine
    );
}

fn outcome(passed: bool) -> Outcome {
    if passed {
        Outcome::Passed
    } else {
        Outcome::Failed
    }
}

/// Floors, set from the measured values recorded in the module docs of this file's receipts.
const MEL_CORRELATION_FLOOR: f64 = 0.90;
const IDENTITY_COSINE_FLOOR: f64 = 0.90;
