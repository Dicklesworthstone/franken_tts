//! The `ftts say` synthesis path: text in, 24 kHz PCM out.
//!
//! This module is where the CLI stops describing the pipeline and runs it. It resolves a
//! checkpoint bundle, hydrates the talker and the codec, tokenizes and wraps the text, derives the
//! prompt header, drives [`TtsEngine::synthesize`] over the real [`QwenGenerator`], and hands the
//! generated codes to the codec decoder. What comes back is `f32` samples; the WAV writing lives
//! in `ftts-core::audio` and the sink policy in [`crate::AudioOutput`].
//!
//! # Why the text is prepared before the engine runs
//!
//! [`TtsEngine::synthesize`] owns text preparation, and normally that is where tokenization
//! happens. Here it happens once, up front, and the engine is handed a preparer that returns that
//! exact result. The reason is the cold text embedding: it is `[151936, 2048]`, and materializing
//! it whole to serve a fifteen-token utterance would cost 1.24 GB. The gather needs the token ids,
//! the generator needs the gathered table, and the generator must exist before `synthesize` is
//! called — so the ids have to be known first. The engine still receives, verbatim, the
//! `PreparedText` a fresh call would have produced; nothing is skipped, only ordered.
//!
//! # Speaker conditioning is derived, never invented
//!
//! An x-vector prompt conditions on a 1,024-wide speaker embedding. A voice source may be either
//! a precomputed raw vector (1,024 little-endian `f32`, 4,096 bytes) or reference audio decoded
//! through the pinned 24 kHz log-mel front end and ECAPA encoder. Neither path accepts a
//! fabricated vector.

use crate::error::FttsError;
use ftts_core::{
    CancellationToken, EngineError, FrameGenerator, GenerationError, NormalizationOptions,
    NormalizationTrace, PreparedText, SynthesisObserver, SynthesisRequest, TextPreparationError,
    TextPreparer, TtsEngine,
};
use ftts_model_qwen::checkpoint::{
    CODEC_LANGUAGE_ENGLISH_ID, CheckpointError, CodecCheckpoint, TALKER_HIDDEN, TalkerCheckpoint,
};
use ftts_model_qwen::generate::{QwenGenerator, QwenGeneratorConfig};
use ftts_model_qwen::microdecoder::MicrodecoderConfig;
use ftts_model_qwen::prompt::{CloneMode, PromptMode};
use ftts_model_qwen::sampler::SamplingMode;
use ftts_model_qwen::speaker::{
    Encoder as SpeakerEncoder, SPEAKER_SAMPLE_RATE_HZ, log_mel_from_24khz_pcm,
};
use ftts_model_qwen::talker::TalkerConfig;
use ftts_model_qwen::tokenizer::{QwenTokenizer, TokenizerFiles};
use std::fs;
use std::fs::OpenOptions;
use std::path::{Path, PathBuf};
use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::DecoderOptions;
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;
use symphonia::default::{get_codecs, get_probe};

/// Bytes in a speaker-vector file: 1,024 little-endian `f32`.
pub const SPEAKER_VECTOR_BYTES: usize = TALKER_HIDDEN * 4;

const CANONICAL_MODEL_BASENAME: &str = "qwen3-tts-12hz-0.6b-base.fttsq";

fn checkpoint_error(error: CheckpointError) -> FttsError {
    FttsError::ArtifactFormat(error.to_string())
}

/// The model resources `ftts say` needs, located relative to one model path.
#[derive(Clone, Debug)]
pub struct ModelBundle {
    /// Directory holding the artifact sidecars and tokenizer files.
    pub root: PathBuf,
    /// The raw main checkpoint, retained for enrollment-only components that have not yet gained
    /// a canonical-artifact accessor.
    pub main: PathBuf,
    /// The portable main-weight artifact selected for synthesis, when present.
    pub canonical_main: Option<PathBuf>,
    /// The codec decoder checkpoint.
    pub codec: PathBuf,
}

impl ModelBundle {
    /// Resolve a bundle from `--model`, which may name the directory, a `.fttsq`, or
    /// `model.safetensors`.
    ///
    /// # Errors
    ///
    /// [`FttsError::ModelNotFound`] naming the exact missing file, so a partial download is
    /// diagnosable without guessing which of the four is absent.
    pub fn resolve(model: &Path) -> Result<Self, FttsError> {
        let root = if model.is_dir() {
            model.to_path_buf()
        } else {
            model
                .parent()
                .ok_or_else(|| {
                    FttsError::ModelNotFound(format!(
                        "model path {} has no parent directory",
                        model.display()
                    ))
                })?
                .to_path_buf()
        };
        let canonical_main = if model.is_dir() {
            let canonical = root.join(CANONICAL_MODEL_BASENAME);
            if canonical.is_file() {
                Some(canonical)
            } else {
                None
            }
        } else if model.extension().and_then(|extension| extension.to_str()) == Some("fttsq") {
            Some(model.to_path_buf())
        } else {
            None
        };
        let main = root.join("model.safetensors");
        let codec = root.join("speech_tokenizer/model.safetensors");
        let (main_label, main_path) = match canonical_main.as_ref() {
            Some(path) => ("canonical talker artifact", path),
            None => ("talker checkpoint", &main),
        };
        for (label, path) in [
            (main_label, main_path),
            ("codec checkpoint", &codec),
            ("tokenizer vocabulary", &root.join("vocab.json")),
            ("tokenizer merges", &root.join("merges.txt")),
            ("tokenizer config", &root.join("tokenizer_config.json")),
        ] {
            if !path.is_file() {
                return Err(FttsError::ModelNotFound(format!(
                    "{label} is missing at {}; `ftts say` needs a complete model directory \
                     ({CANONICAL_MODEL_BASENAME} or model.safetensors, \
                     speech_tokenizer/model.safetensors, \
                     vocab.json, merges.txt, tokenizer_config.json)",
                    path.display()
                )));
            }
        }
        Ok(Self {
            root,
            main,
            canonical_main,
            codec,
        })
    }
}

/// Every weight and table one `say` needs, hydrated once.
pub struct LoadedModel {
    talker: TalkerCheckpoint,
    codec: CodecCheckpoint,
    tokenizer: QwenTokenizer,
    /// A second mapping of the canonical artifact, when the bundle carries one. The int8 route
    /// hydrates its Q8 tables straight from this (proven byte-identical to requantizing the
    /// widened f32 copies, scales included); mapping the file twice costs nothing beyond page
    /// cache the first mapping already warmed.
    artifact: Option<ftts_artifacts::fttsq::MappedFttsq>,
}

impl LoadedModel {
    /// Hydrate the bundle. This reads gigabytes and is the slow step of a cold run.
    ///
    /// # Errors
    ///
    /// If any checkpoint or tokenizer file is unreadable or not the pinned model.
    pub fn load(bundle: &ModelBundle) -> Result<Self, FttsError> {
        let read = |name: &str| -> Result<String, FttsError> {
            let path = bundle.root.join(name);
            fs::read_to_string(&path).map_err(|error| {
                FttsError::ArtifactFormat(format!("cannot read {}: {error}", path.display()))
            })
        };
        let vocab = read("vocab.json")?;
        let merges = read("merges.txt")?;
        let config = read("tokenizer_config.json")?;

        // The three heavyweight hydrations are independent, so the codec checkpoint and the
        // tokenizer build overlap the talker load instead of queueing behind it. Each result is
        // computed exactly as it was serially; only wall time changes.
        let (talker, codec, tokenizer) = std::thread::scope(|scope| {
            let codec = scope.spawn(|| CodecCheckpoint::load(&bundle.codec));
            let tokenizer = scope.spawn(|| {
                QwenTokenizer::from_files_using_environment(TokenizerFiles {
                    vocab_json: &vocab,
                    merges_txt: &merges,
                    tokenizer_config_json: &config,
                })
            });
            let talker = match bundle.canonical_main.as_deref() {
                // The elision mirrors the generator's own hydration decision for this process:
                // stacks that will run artifact-native int8 skip their dead f32 projections
                // (~2.5 GB of widening nobody reads).
                Some(path) => TalkerCheckpoint::load_fttsq_elided(
                    path,
                    ftts_model_qwen::generate::hot_elision_from_environment(),
                ),
                None => TalkerCheckpoint::load(&bundle.main),
            };
            (
                talker,
                codec.join().expect("codec loader panicked"),
                tokenizer.join().expect("tokenizer builder panicked"),
            )
        });
        let tokenizer = tokenizer
            .map_err(|error| FttsError::ArtifactFormat(format!("tokenizer unusable: {error}")))?;

        Ok(Self {
            talker: talker.map_err(checkpoint_error)?,
            codec: codec.map_err(checkpoint_error)?,
            tokenizer,
            artifact: bundle
                .canonical_main
                .as_deref()
                .map(|path| {
                    ftts_artifacts::fttsq::MappedFttsq::open(path).map_err(|error| {
                        FttsError::ArtifactFormat(format!(
                            "cannot map canonical artifact {}: {error}",
                            path.display()
                        ))
                    })
                })
                .transpose()?,
        })
    }
}

/// Read a precomputed 1,024-wide speaker vector.
///
/// See the module docs on why this is a raw vector rather than a `.ftvoice` pack.
///
/// # Errors
///
/// [`FttsError::Input`] when the file is unreadable or is not exactly
/// [`SPEAKER_VECTOR_BYTES`] bytes — a truncated vector would otherwise be padded with silence and
/// change the voice in a way only listening could detect.
pub fn read_speaker_vector(path: &Path) -> Result<Vec<f32>, FttsError> {
    let bytes = fs::read(path).map_err(|error| {
        FttsError::Input(format!(
            "cannot read speaker vector {}: {error}",
            path.display()
        ))
    })?;
    if bytes.len() != SPEAKER_VECTOR_BYTES {
        return Err(FttsError::Input(format!(
            "speaker vector {} is {} bytes; `ftts say --voice` expects exactly {} \
             ({TALKER_HIDDEN} little-endian f32)",
            path.display(),
            bytes.len(),
            SPEAKER_VECTOR_BYTES
        )));
    }
    let vector: Vec<f32> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|quad| f32::from_le_bytes(*quad))
        .collect();
    if let Some(index) = vector.iter().position(|value| !value.is_finite()) {
        return Err(FttsError::Input(format!(
            "speaker vector {} holds a non-finite value at index {index}; it would poison every \
             prefill position it is summed into",
            path.display()
        )));
    }
    Ok(vector)
}

/// Derive an x-vector from an enrolled raw vector or a real reference recording.
/// What a `--denoise` enrollment measured, so the CLI can report the effect rather than assert it.
#[derive(Clone, Copy, Debug)]
pub struct DenoiseReport {
    /// Pause floor of the decoded reference, before denoising.
    pub before_dbfs: f32,
    /// Pause floor after denoising.
    pub after_dbfs: f32,
}

/// Derive a speaker vector from a voice source: a raw x-vector file, or reference audio.
///
/// Passing `Some` for `denoise` opts the reference into [`denoise_reference`] and fills the slot
/// with the measured before/after pause floor.
///
/// # Errors
///
/// When the source cannot be read, decoded, resampled, or encoded into a finite x-vector.
pub fn speaker_from_voice(
    bundle: &ModelBundle,
    path: &Path,
    denoise: Option<&mut Option<DenoiseReport>>,
) -> Result<Vec<f32>, FttsError> {
    let bytes = fs::read(path).map_err(|error| {
        FttsError::Input(format!(
            "cannot read voice source {}: {error}",
            path.display()
        ))
    })?;
    if bytes.len() == SPEAKER_VECTOR_BYTES {
        return decode_speaker_vector(path, &bytes);
    }
    let pcm = decode_reference_audio_any(path)?;
    let pcm = match denoise {
        Some(report) => {
            let before = pause_floor_dbfs(&pcm);
            let cleaned = denoise_reference(&pcm);
            *report = Some(DenoiseReport {
                before_dbfs: before,
                after_dbfs: pause_floor_dbfs(&cleaned),
            });
            cleaned
        }
        None => pcm,
    };
    let mel = log_mel_from_24khz_pcm(&pcm)
        .map_err(|error| FttsError::Input(format!("cannot extract speaker features: {error}")))?;
    let encoder = match bundle.canonical_main.as_deref() {
        Some(artifact) => SpeakerEncoder::load_fttsq(artifact),
        None => SpeakerEncoder::load(&bundle.main),
    }
    .map_err(checkpoint_error)?;
    let vector = encoder.encode(&mel.values, mel.frames);
    if vector.iter().all(|value| value.is_finite()) {
        Ok(vector)
    } else {
        Err(FttsError::Input(
            "speaker encoder produced a non-finite x-vector; refusing to condition synthesis"
                .to_owned(),
        ))
    }
}

/// Write a raw x-vector without replacing an existing enrollment result.
pub fn write_speaker_vector_new(path: &Path, vector: &[f32]) -> Result<(), FttsError> {
    if vector.len() != TALKER_HIDDEN {
        return Err(FttsError::Input(format!(
            "cannot write {}-wide speaker vector; expected {TALKER_HIDDEN}",
            vector.len()
        )));
    }
    if let Some(index) = vector.iter().position(|value| !value.is_finite()) {
        return Err(FttsError::Input(format!(
            "cannot write speaker vector with a non-finite value at index {index}"
        )));
    }
    let mut bytes = Vec::with_capacity(SPEAKER_VECTOR_BYTES);
    for value in vector {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    use std::io::Write;
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|error| {
            FttsError::Input(format!(
                "cannot create enrolled voice {} without overwriting an existing file: {error}",
                path.display()
            ))
        })?;
    file.write_all(&bytes).map_err(|error| {
        FttsError::Input(format!(
            "cannot write enrolled voice {}: {error}",
            path.display()
        ))
    })
}

fn decode_speaker_vector(path: &Path, bytes: &[u8]) -> Result<Vec<f32>, FttsError> {
    let vector: Vec<f32> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|quad| f32::from_le_bytes(*quad))
        .collect();
    if let Some(index) = vector.iter().position(|value| !value.is_finite()) {
        return Err(FttsError::Input(format!(
            "speaker vector {} holds a non-finite value at index {index}; it would poison every \
             prefill position it is summed into",
            path.display()
        )));
    }
    Ok(vector)
}

/// Container formats the embedded decoder does not read; these route through a system decoder,
/// mirroring how output encoding shells out — synthesis and enrollment themselves never depend
/// on one.
const SYSTEM_DECODED_EXTENSIONS: [&str; 6] = ["m4a", "mp3", "aac", "mp4", "ogg", "opus"];

/// Decodes reference audio of any supported container to mono f32 PCM.
///
/// WAV and FLAC decode through the embedded pure-Rust path. Compressed containers (m4a, mp3, …)
/// are first transcoded to a temporary WAV by the first system decoder found — `afconvert` on
/// macOS, then `ffmpeg` — with a clear error naming both tools when neither exists.
fn decode_reference_audio_any(path: &Path) -> Result<Vec<f32>, FttsError> {
    let extension = path
        .extension()
        .and_then(|extension| extension.to_str())
        .map(str::to_ascii_lowercase);
    let needs_system_decoder = extension
        .as_deref()
        .is_some_and(|extension| SYSTEM_DECODED_EXTENSIONS.contains(&extension));
    if !needs_system_decoder {
        return decode_reference_audio(path);
    }

    let staging = std::env::temp_dir().join(format!(
        "ftts-enroll-{}-{}.wav",
        std::process::id(),
        path.file_stem()
            .and_then(|stem| stem.to_str())
            .unwrap_or("reference")
    ));
    let attempts: &[(&str, Vec<&std::ffi::OsStr>)] = &[
        // Both decoders are told to resample to the speaker encoder's pinned 24 kHz mono here
        // rather than leaving the source rate intact: phone and Mac voice memos default to
        // 44.1/48 kHz, and a transcode that preserves them would only move the failure to the
        // enrollment rate check (frankentts-gra).
        (
            "afconvert",
            vec![
                "-f".as_ref(),
                "WAVE".as_ref(),
                "-d".as_ref(),
                "LEI16@24000".as_ref(),
                "-c".as_ref(),
                "1".as_ref(),
                path.as_os_str(),
                staging.as_os_str(),
            ],
        ),
        (
            "ffmpeg",
            vec![
                "-y".as_ref(),
                "-loglevel".as_ref(),
                "error".as_ref(),
                "-i".as_ref(),
                path.as_os_str(),
                "-acodec".as_ref(),
                "pcm_s16le".as_ref(),
                "-ar".as_ref(),
                "24000".as_ref(),
                "-ac".as_ref(),
                "1".as_ref(),
                staging.as_os_str(),
            ],
        ),
    ];
    let mut ran = false;
    for (tool, arguments) in attempts {
        match std::process::Command::new(tool).args(arguments).status() {
            Ok(status) if status.success() => {
                ran = true;
                break;
            }
            Ok(status) => {
                let _ = fs::remove_file(&staging);
                return Err(FttsError::Input(format!(
                    "{tool} failed decoding reference audio {} (exit {status})",
                    path.display()
                )));
            }
            Err(_) => continue, // tool not installed; try the next one
        }
    }
    if !ran {
        return Err(FttsError::Input(format!(
            "reference audio {} is a compressed container and no system decoder was found; \
             install afconvert (macOS) or ffmpeg, or supply WAV/FLAC",
            path.display()
        )));
    }
    let decoded = decode_reference_audio(&staging);
    let _ = fs::remove_file(&staging);
    decoded
}

fn decode_reference_audio(path: &Path) -> Result<Vec<f32>, FttsError> {
    let file = fs::File::open(path).map_err(|error| {
        FttsError::Input(format!(
            "cannot open reference audio {}: {error}",
            path.display()
        ))
    })?;
    let mut hint = Hint::new();
    if let Some(extension) = path.extension().and_then(|extension| extension.to_str()) {
        hint.with_extension(extension);
    }
    let stream = MediaSourceStream::new(Box::new(file), Default::default());
    let probed = get_probe()
        .format(
            &hint,
            stream,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )
        .map_err(|error| {
            FttsError::Input(format!(
                "cannot identify reference audio {}: {error}",
                path.display()
            ))
        })?;
    let mut format = probed.format;
    let track = format.default_track().ok_or_else(|| {
        FttsError::Input(format!(
            "reference audio {} has no default audio track",
            path.display()
        ))
    })?;
    let track_id = track.id;
    let mut decoder = get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .map_err(|error| {
            FttsError::Input(format!(
                "cannot decode reference audio {}: {error}",
                path.display()
            ))
        })?;
    let mut sample_rate = None;
    let mut mono = Vec::new();
    loop {
        let packet = match format.next_packet() {
            Ok(packet) => packet,
            Err(SymphoniaError::IoError(error))
                if error.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                break;
            }
            Err(error) => {
                return Err(FttsError::Input(format!(
                    "cannot read reference audio {}: {error}",
                    path.display()
                )));
            }
        };
        if packet.track_id() != track_id {
            continue;
        }
        let decoded = decoder.decode(&packet).map_err(|error| {
            FttsError::Input(format!(
                "cannot decode reference audio {}: {error}",
                path.display()
            ))
        })?;
        let spec = *decoded.spec();
        match sample_rate {
            Some(rate) if rate != spec.rate => {
                return Err(FttsError::Input(format!(
                    "reference audio {} changed sample rate mid-stream ({rate} to {} Hz)",
                    path.display(),
                    spec.rate
                )));
            }
            None => sample_rate = Some(spec.rate),
            Some(_) => {}
        }
        let channels = spec.channels.count();
        let mut samples = SampleBuffer::<f32>::new(decoded.capacity() as u64, spec);
        samples.copy_interleaved_ref(decoded);
        for frame in samples.samples().chunks_exact(channels) {
            mono.push(frame.iter().sum::<f32>() / channels as f32);
        }
    }
    let rate = sample_rate.ok_or_else(|| {
        FttsError::Input(format!(
            "reference audio {} contains no decodable samples",
            path.display()
        ))
    })?;
    if mono.is_empty() {
        return Err(FttsError::Input(format!(
            "reference audio {} contains no PCM samples",
            path.display()
        )));
    }
    let pcm = resample_to_speaker_rate(mono, rate);
    // Downsampling shortens the signal, and a clip of a few samples at a high source rate can
    // round to nothing. The mel front end would then see an empty slice, so the emptiness check
    // has to be made against the PCM actually handed on, not only against what was decoded.
    if pcm.is_empty() {
        return Err(FttsError::Input(format!(
            "reference audio {} is too short to resample from {rate} Hz to \
             {SPEAKER_SAMPLE_RATE_HZ} Hz; supply a longer recording",
            path.display()
        )));
    }
    Ok(pcm)
}

/// Resamples decoded mono PCM to the speaker encoder's pinned rate.
///
/// Compressed references already arrive at 24 kHz because the system decoder is told to convert
/// (`frankentts-gra`), but a `.wav` or `.flac` is read directly and can be any rate — 44.1 and 48
/// kHz being what every phone, Mac voice memo, and DAW export actually produces. Refusing those
/// pushed the identical resample onto the user as an `ffmpeg` incantation, so it happens here.
///
/// Audio already at the pinned rate is returned untouched, so this cannot perturb any existing
/// enrollment: it only turns a former hard error into a working path.
///
/// Windowed-sinc (Lanczos-3) with the kernel cutoff clamped to the lower of the two rates, which
/// is what suppresses aliasing on the common downsampling direction. Taps are normalized by their
/// own sum so DC gain stays 1 even where the window runs off the ends of the signal.
fn resample_to_speaker_rate(mono: Vec<f32>, from_rate: u32) -> Vec<f32> {
    if from_rate == SPEAKER_SAMPLE_RATE_HZ {
        return mono;
    }
    const LOBES: f64 = 3.0;
    let ratio = f64::from(SPEAKER_SAMPLE_RATE_HZ) / f64::from(from_rate);
    let cutoff = ratio.min(1.0);
    let half = (LOBES / cutoff).ceil() as isize;
    let out_len = ((mono.len() as f64) * ratio).round() as usize;

    let mut out = Vec::with_capacity(out_len);
    for index in 0..out_len {
        let center = index as f64 / ratio;
        let first = center.floor() as isize - half + 1;
        let mut acc = 0.0_f64;
        let mut norm = 0.0_f64;
        for tap in first..first + 2 * half {
            if tap < 0 {
                continue;
            }
            let Some(sample) = mono.get(tap as usize) else {
                break;
            };
            let weight = lanczos_tap(center - tap as f64, cutoff, LOBES);
            acc += weight * f64::from(*sample);
            norm += weight;
        }
        out.push(if norm.abs() > 1e-12 {
            (acc / norm) as f32
        } else {
            0.0
        });
    }
    out
}

/// STFT window for reference denoising: 512 samples is ~21 ms at the pinned 24 kHz, long enough
/// to resolve a noise floor between words and short enough not to smear plosives.
const DENOISE_FRAME: usize = 512;

/// Three-quarter overlap. Hann at hop `N/4` overlap-adds smoothly, which is what keeps gain
/// changes from becoming audible frame edges.
const DENOISE_HOP: usize = DENOISE_FRAME / 4;

/// Decision-directed smoothing for the a priori SNR. Ephraim and Malah's 0.98 is calmer but lags
/// onsets; 0.92 tracks a voice that starts and stops mid-recording.
const DD_ALPHA: f32 = 0.92;

/// Floor gain, −25 dB. OM-LSA never gates a bin fully closed: leaving a quiet, *stationary* bed
/// is what stops residual noise from flickering into musical tones.
const OMLSA_GAIN_FLOOR: f32 = 0.056_234_13;

/// Smoothing for IMCRA's power spectrum, and for its recursive noise average.
const IMCRA_ALPHA_S: f32 = 0.9;
const IMCRA_ALPHA_D: f32 = 0.85;

/// Minimum-tracking window: `U` subwindows of `V` frames. 8 × 15 frames ≈ 0.64 s at this hop,
/// long enough to span a syllable so speech cannot masquerade as the floor.
const IMCRA_SUBWINDOWS: usize = 8;
const IMCRA_SUBWINDOW_LEN: usize = 15;

/// Bias compensation for the minimum of a smoothed periodogram, and the decision thresholds that
/// separate "this bin is noise" from "this bin has speech" (Cohen 2003, §III).
const IMCRA_B_MIN: f32 = 1.66;
const IMCRA_GAMMA0: f32 = 4.6;
const IMCRA_GAMMA1: f32 = 3.0;
const IMCRA_ZETA0: f32 = 1.67;

/// Single-channel speech enhancement by OM-LSA with IMCRA noise tracking.
///
/// Enrollment noise is not cosmetic: it is encoded into the x-vector and then reproduced in every
/// utterance the cloned voice speaks (measured — cleaning a real 53 s reference dropped the
/// synthesized output's pause floor by 3.7 dB). This removes the stationary part of it.
///
/// **Why this and not spectral subtraction.** Subtracting an estimated noise magnitude minimises
/// squared error in the *spectrum*, which is the wrong objective for something a listener judges
/// and a speaker encoder reads: it punches holes in low-SNR bins, producing musical noise, and it
/// removes real signal along with the noise (measured here at 33% of peak burst energy before
/// this replaced it). Three pieces fix that, and they compose:
///
/// 1. **MMSE-LSA** (Ephraim & Malah 1985) estimates the *log* amplitude, matching how loudness is
///    perceived, and yields the gain `ξ/(1+ξ) · exp(½·E₁(ν))`. The exponential-integral term is
///    what makes it gentle where the a posteriori SNR is uncertain instead of gating hard.
/// 2. **Decision-directed a priori SNR** (same paper) smooths ξ across frames using the previous
///    frame's own estimate. This is the specific mechanism that suppresses musical noise: isolated
///    noise peaks never get a confident ξ, so they are never sharply attenuated *or* passed.
/// 3. **IMCRA** (Cohen 2003) tracks the noise floor by minimum statistics with a two-pass
///    speech-presence exclusion, so the floor keeps adapting through speech and follows
///    nonstationary room tone — where a fixed quantile estimate silently drifts wrong.
///
/// OM-LSA (Cohen & Berdugo 2001) then blends the LSA gain toward the floor by the speech-presence
/// probability, `G = G_LSA^p · G_min^(1−p)`, so bins that are probably noise are attenuated to a
/// constant bed rather than tracked.
///
/// This is the state of the art among methods that need no trained weights. Neural enhancers
/// (DeepFilterNet and friends) do beat it, at the cost of shipping and running another model —
/// which is not a trade this CLI should make silently for an enrollment preprocessing step.
///
/// Deliberately conservative even so: the speaker encoder reads breath, sibilance, and room as
/// part of identity, so this stays opt-in (`--denoise`) per the project's doctrine that a lever
/// which can damage speaker identity ships behind a named switch until blind listening clears it.
///
/// Phase is preserved untouched; only per-bin magnitude is scaled.
fn denoise_reference(pcm: &[f32]) -> Vec<f32> {
    if pcm.len() < DENOISE_FRAME {
        return pcm.to_vec();
    }
    let mut planner = rustfft::FftPlanner::<f32>::new();
    let forward = planner.plan_fft_forward(DENOISE_FRAME);
    let inverse = planner.plan_fft_inverse(DENOISE_FRAME);

    let window: Vec<f32> = (0..DENOISE_FRAME)
        .map(|n| {
            let phase = std::f32::consts::TAU * n as f32 / DENOISE_FRAME as f32;
            0.5 - 0.5 * phase.cos()
        })
        .collect();

    let bins = DENOISE_FRAME / 2 + 1;
    let starts: Vec<usize> = (0..=pcm.len() - DENOISE_FRAME)
        .step_by(DENOISE_HOP)
        .collect();

    let mut tracker = ImcraTracker::new(bins);
    // Decision-directed state: last frame's gain and a posteriori SNR, per bin.
    let mut prev_gain = vec![1.0_f32; bins];
    let mut prev_gamma = vec![1.0_f32; bins];

    let mut out = vec![0.0_f32; pcm.len()];
    let mut weight = vec![0.0_f32; pcm.len()];

    for &start in &starts {
        let mut frame: Vec<rustfft::num_complex::Complex<f32>> = (0..DENOISE_FRAME)
            .map(|n| rustfft::num_complex::Complex::new(pcm[start + n] * window[n], 0.0))
            .collect();
        forward.process(&mut frame);

        let power: Vec<f32> = (0..bins).map(|bin| frame[bin].norm_sqr()).collect();
        let presence = tracker.update(&power);

        for bin in 0..bins {
            let noise = tracker.noise[bin].max(1e-12);
            let gamma = (power[bin] / noise).min(1e6);
            // Decision-directed a priori SNR: last frame's estimate, blended with this frame's
            // maximum-likelihood one.
            let xi = (DD_ALPHA * prev_gain[bin].powi(2) * prev_gamma[bin]
                + (1.0 - DD_ALPHA) * (gamma - 1.0).max(0.0))
            .max(1e-6);

            let nu = (xi / (1.0 + xi)) * gamma;
            let lsa = (xi / (1.0 + xi)) * (0.5 * exponential_integral_e1(nu)).exp();
            let lsa = lsa.clamp(OMLSA_GAIN_FLOOR, 1.0);

            // OM-LSA: interpolate in the log domain between the LSA gain and the floor.
            let p = presence[bin].clamp(0.0, 1.0);
            let gain = lsa.powf(p) * OMLSA_GAIN_FLOOR.powf(1.0 - p);

            prev_gain[bin] = gain;
            prev_gamma[bin] = gamma;

            frame[bin] *= gain;
            // The spectrum of a real signal is conjugate-symmetric; scaling only the first half
            // and letting the mirror drift would synthesize a complex signal.
            let mirror = DENOISE_FRAME - bin;
            if bin > 0 && mirror < DENOISE_FRAME {
                frame[mirror] *= gain;
            }
        }

        inverse.process(&mut frame);
        let scale = 1.0 / DENOISE_FRAME as f32;
        for n in 0..DENOISE_FRAME {
            out[start + n] += frame[n].re * scale * window[n];
            weight[start + n] += window[n] * window[n];
        }
    }

    for (sample, energy) in out.iter_mut().zip(weight.iter()) {
        if *energy > 1e-6 {
            *sample /= *energy;
        }
    }
    // The head and tail lie outside any full window and keep their original samples rather than
    // a partially-normalized reconstruction.
    let covered =
        starts.first().copied().unwrap_or(0)..starts.last().map_or(0, |last| last + DENOISE_FRAME);
    for (index, sample) in out.iter_mut().enumerate() {
        if !covered.contains(&index) || weight[index] <= 1e-6 {
            *sample = pcm[index];
        }
    }
    out
}

/// IMCRA noise tracker: minimum statistics with two-pass speech exclusion.
///
/// The whole point is that the noise floor must keep updating *while someone is speaking*. A
/// tracker that only learns during silence drifts as the room changes; one that learns from every
/// frame absorbs the voice into its own noise estimate and then subtracts it.
struct ImcraTracker {
    bins: usize,
    /// Recursively averaged noise power — the estimate everything downstream divides by.
    noise: Vec<f32>,
    /// First-pass smoothed power and its running/subwindow minima.
    smoothed: Vec<f32>,
    running_min: Vec<f32>,
    subwindow_min: Vec<f32>,
    history: Vec<Vec<f32>>,
    /// Second pass, computed over bins the first pass called noise-only.
    smoothed_refined: Vec<f32>,
    running_min_refined: Vec<f32>,
    subwindow_min_refined: Vec<f32>,
    history_refined: Vec<Vec<f32>>,
    frame: usize,
    subwindow_slot: usize,
}

impl ImcraTracker {
    fn new(bins: usize) -> Self {
        Self {
            bins,
            noise: vec![0.0; bins],
            smoothed: vec![0.0; bins],
            running_min: vec![f32::MAX; bins],
            subwindow_min: vec![f32::MAX; bins],
            history: vec![vec![f32::MAX; bins]; IMCRA_SUBWINDOWS],
            smoothed_refined: vec![0.0; bins],
            running_min_refined: vec![f32::MAX; bins],
            subwindow_min_refined: vec![f32::MAX; bins],
            history_refined: vec![vec![f32::MAX; bins]; IMCRA_SUBWINDOWS],
            frame: 0,
            subwindow_slot: 0,
        }
    }

    /// Feed one frame's power spectrum; return per-bin speech-presence probability.
    fn update(&mut self, power: &[f32]) -> Vec<f32> {
        if self.frame == 0 {
            // Seed every state from the first frame rather than from zero, so the first second of
            // audio is not enhanced against a meaningless floor.
            self.noise.copy_from_slice(power);
            self.smoothed.copy_from_slice(power);
            self.smoothed_refined.copy_from_slice(power);
            self.running_min.copy_from_slice(power);
            self.subwindow_min.copy_from_slice(power);
            self.running_min_refined.copy_from_slice(power);
            self.subwindow_min_refined.copy_from_slice(power);
        }

        // Pass 1: smooth in frequency, then in time, then track the minimum.
        let smoothed_freq = smooth_across_bins(power);
        for bin in 0..self.bins {
            self.smoothed[bin] =
                IMCRA_ALPHA_S * self.smoothed[bin] + (1.0 - IMCRA_ALPHA_S) * smoothed_freq[bin];
        }
        track_minimum(
            &self.smoothed,
            &mut self.running_min,
            &mut self.subwindow_min,
        );

        // Which bins look like noise only? Both the instantaneous and the smoothed power have to
        // sit close to the tracked minimum.
        let mut noise_only = vec![false; self.bins];
        for bin in 0..self.bins {
            let floor = (IMCRA_B_MIN * self.running_min[bin]).max(1e-12);
            let gamma_min = power[bin] / floor;
            let zeta = self.smoothed[bin] / floor;
            noise_only[bin] = gamma_min < IMCRA_GAMMA0 && zeta < IMCRA_ZETA0;
        }

        // Pass 2: repeat the smoothing using only those bins, which keeps a strong harmonic from
        // dragging its neighbours' floor upward.
        let refined_freq = smooth_across_bins_masked(power, &noise_only, &self.smoothed_refined);
        for bin in 0..self.bins {
            self.smoothed_refined[bin] = IMCRA_ALPHA_S * self.smoothed_refined[bin]
                + (1.0 - IMCRA_ALPHA_S) * refined_freq[bin];
        }
        track_minimum(
            &self.smoothed_refined,
            &mut self.running_min_refined,
            &mut self.subwindow_min_refined,
        );

        // Speech-absence probability from the refined statistics, then presence.
        let mut presence = vec![0.0_f32; self.bins];
        for bin in 0..self.bins {
            let floor = (IMCRA_B_MIN * self.running_min_refined[bin]).max(1e-12);
            let gamma_min = power[bin] / floor;
            let zeta = self.smoothed_refined[bin] / floor;
            let q = if gamma_min <= 1.0 || zeta < IMCRA_ZETA0 {
                1.0
            } else if gamma_min >= IMCRA_GAMMA1 {
                0.0
            } else {
                (IMCRA_GAMMA1 - gamma_min) / (IMCRA_GAMMA1 - 1.0)
            };
            presence[bin] = (1.0 - q).clamp(0.0, 1.0);
        }

        // Recursive noise update, slowed wherever speech is likely present.
        for bin in 0..self.bins {
            let alpha = IMCRA_ALPHA_D + (1.0 - IMCRA_ALPHA_D) * presence[bin];
            self.noise[bin] = alpha * self.noise[bin] + (1.0 - alpha) * power[bin];
        }

        self.frame += 1;
        if self.frame % IMCRA_SUBWINDOW_LEN == 0 {
            self.roll_subwindow();
        }
        presence
    }

    /// Close the current subwindow and recompute each running minimum over the retained ones.
    fn roll_subwindow(&mut self) {
        self.history[self.subwindow_slot].copy_from_slice(&self.subwindow_min);
        self.history_refined[self.subwindow_slot].copy_from_slice(&self.subwindow_min_refined);
        self.subwindow_slot = (self.subwindow_slot + 1) % IMCRA_SUBWINDOWS;

        for bin in 0..self.bins {
            let mut best = f32::MAX;
            let mut best_refined = f32::MAX;
            for slot in 0..IMCRA_SUBWINDOWS {
                best = best.min(self.history[slot][bin]);
                best_refined = best_refined.min(self.history_refined[slot][bin]);
            }
            self.running_min[bin] = best.min(self.smoothed[bin]);
            self.running_min_refined[bin] = best_refined.min(self.smoothed_refined[bin]);
            self.subwindow_min[bin] = self.smoothed[bin];
            self.subwindow_min_refined[bin] = self.smoothed_refined[bin];
        }
    }
}

/// Running and subwindow minima, updated in place for one frame.
fn track_minimum(smoothed: &[f32], running: &mut [f32], subwindow: &mut [f32]) {
    for bin in 0..smoothed.len() {
        running[bin] = running[bin].min(smoothed[bin]);
        subwindow[bin] = subwindow[bin].min(smoothed[bin]);
    }
}

/// Three-bin Hann smoothing across frequency, which is what makes the minimum statistic stable
/// enough to track without a long window.
fn smooth_across_bins(power: &[f32]) -> Vec<f32> {
    const WEIGHTS: [f32; 3] = [0.25, 0.5, 0.25];
    (0..power.len())
        .map(|bin| {
            let mut acc = 0.0;
            let mut norm = 0.0;
            for (offset, weight) in (-1_isize..=1).zip(WEIGHTS) {
                let index = bin as isize + offset;
                if index >= 0 && (index as usize) < power.len() {
                    acc += weight * power[index as usize];
                    norm += weight;
                }
            }
            acc / norm
        })
        .collect()
}

/// As [`smooth_across_bins`], but averaging only over bins currently believed to be noise.
///
/// Where a whole neighbourhood is speech there is nothing to average, so the previous refined
/// estimate carries forward rather than being replaced by the speech power.
fn smooth_across_bins_masked(power: &[f32], noise_only: &[bool], previous: &[f32]) -> Vec<f32> {
    const WEIGHTS: [f32; 3] = [0.25, 0.5, 0.25];
    (0..power.len())
        .map(|bin| {
            let mut acc = 0.0;
            let mut norm = 0.0;
            for (offset, weight) in (-1_isize..=1).zip(WEIGHTS) {
                let index = bin as isize + offset;
                if index >= 0 && (index as usize) < power.len() && noise_only[index as usize] {
                    acc += weight * power[index as usize];
                    norm += weight;
                }
            }
            if norm > 0.0 {
                acc / norm
            } else {
                previous[bin]
            }
        })
        .collect()
}

/// The exponential integral `E₁(x) = ∫ₓ^∞ e^{−t}/t dt`, for `x > 0`.
///
/// This is the term that makes MMSE-LSA gentle rather than gating: it grows without bound as the
/// a priori SNR falls, so the log-amplitude estimate backs off smoothly instead of snapping shut.
/// Abramowitz & Stegun 5.1.53 below 1 and 5.1.56 above it; both are accurate to ~2e-7, far inside
/// what a spectral gain needs.
fn exponential_integral_e1(x: f32) -> f32 {
    if x <= 0.0 {
        // ν ≤ 0 cannot arise from a non-negative SNR, but a denormal would otherwise return NaN
        // and poison the frame.
        return 0.0;
    }
    let x = f64::from(x);
    let value = if x < 1.0 {
        // A&S 5.1.53: E₁(x) + ln x = polynomial in x.
        const A: [f64; 6] = [
            -0.577_215_664_9,
            0.999_991_93,
            -0.249_910_55,
            0.055_199_68,
            -0.009_760_04,
            0.001_079_857,
        ];
        let mut acc = 0.0;
        for (power, coefficient) in A.iter().enumerate() {
            acc += coefficient * x.powi(power as i32);
        }
        acc - x.ln()
    } else {
        // A&S 5.1.56: x·e^x·E₁(x) = rational in x.
        const A: [f64; 4] = [8.573_328_74, 18.059_016_97, 8.634_760_89, 0.267_773_734];
        const B: [f64; 4] = [9.573_322_34, 25.632_956_15, 21.099_653_08, 3.958_496_93];
        let numerator = x.powi(4) + A[0] * x.powi(3) + A[1] * x * x + A[2] * x + A[3];
        let denominator = x.powi(4) + B[0] * x.powi(3) + B[1] * x * x + B[2] * x + B[3];
        (numerator / denominator) / (x * x.exp())
    };
    value as f32
}

/// Root-mean-square of the quietest decile of 50 ms windows, in dBFS.
///
/// Whole-signal RMS hides pause noise behind the speech that dominates it, so enrollment
/// diagnostics report the floor between words — the part a denoise actually moves.
fn pause_floor_dbfs(pcm: &[f32]) -> f32 {
    let span = (SPEAKER_SAMPLE_RATE_HZ as usize) / 20;
    if pcm.len() < span {
        return f32::NEG_INFINITY;
    }
    let mut windows: Vec<f32> = pcm
        .chunks_exact(span)
        .map(|chunk| {
            (chunk
                .iter()
                .map(|s| f64::from(*s) * f64::from(*s))
                .sum::<f64>()
                / chunk.len() as f64)
                .sqrt() as f32
        })
        .filter(|rms| *rms > 0.0)
        .collect();
    if windows.is_empty() {
        return f32::NEG_INFINITY;
    }
    windows.sort_by(f32::total_cmp);
    let keep = (windows.len() / 10).max(1);
    let mean = windows[..keep].iter().sum::<f32>() / keep as f32;
    20.0 * mean.log10()
}

/// One Lanczos tap: a sinc lowpass at `cutoff`, windowed by a wider sinc over `lobes`.
fn lanczos_tap(offset: f64, cutoff: f64, lobes: f64) -> f64 {
    let scaled = cutoff * offset;
    if scaled.abs() >= lobes {
        return 0.0;
    }
    sinc(scaled) * sinc(scaled / lobes)
}

/// Normalized sinc, `sin(pi x) / (pi x)`, with the removable singularity at zero filled in.
fn sinc(x: f64) -> f64 {
    if x.abs() < 1e-12 {
        return 1.0;
    }
    let scaled = std::f64::consts::PI * x;
    scaled.sin() / scaled
}

/// Hands the engine a `PreparedText` that was computed before the weights were borrowed.
struct PreparedPassThrough {
    prepared: PreparedText,
}

impl TextPreparer for PreparedPassThrough {
    fn prepare(
        &self,
        _text: &str,
        _options: &NormalizationOptions,
    ) -> Result<PreparedText, TextPreparationError> {
        Ok(PreparedText::new(
            self.prepared.token_ids.clone(),
            NormalizationTrace {
                mode: self.prepared.normalization_trace.mode,
                unicode_version: self.prepared.normalization_trace.unicode_version.clone(),
                changes: self.prepared.normalization_trace.changes.clone(),
            },
        ))
    }
}

/// A completed synthesis: the codes the talker produced and the audio they decode to.
pub struct SynthesizedAudio {
    /// Codec frames generated before the stop.
    pub frames: u64,
    /// Token ids that entered the model path, including the assistant wrapper.
    pub prepared_token_count: usize,
    /// Mono 24 kHz samples in `[-1, 1]`.
    pub pcm: Vec<f32>,
}

/// Run one utterance end to end: text, codes, PCM.
///
/// # Errors
///
/// Engine refusals (admission, budget, cancellation) and model refusals are mapped to their CLI
/// exit classes; a zero-frame generation is reported rather than written out as an empty file.
#[allow(clippy::too_many_arguments)]
pub fn synthesize(
    model: &LoadedModel,
    engine: &TtsEngine,
    request: &SynthesisRequest,
    speaker: &[f32],
    seed: u64,
    cancellation: &CancellationToken,
    observer: &dyn SynthesisObserver,
) -> Result<SynthesizedAudio, FttsError> {
    // 1. Text, once — see the module docs on ordering.
    let prepared_raw = model
        .tokenizer
        .prepare(&request.text, &request.normalization_options)
        .map_err(|error| FttsError::Input(format!("text preparation failed: {error}")))?;
    let wrapped = TalkerCheckpoint::wrap_target_ids(&prepared_raw.token_ids);
    let prepared = PreparedText::new(wrapped.clone(), prepared_raw.normalization_trace);

    // 2. The cold-embedding rows this utterance can reach, and nothing else.
    let ids = TalkerCheckpoint::utterance_text_ids(&wrapped);
    let table = model
        .talker
        .gather_text_rows(&ids)
        .map_err(checkpoint_error)?;

    // 3. The prompt header, derived from checkpoint tensors and the caller's speaker vector.
    let header = model
        .talker
        .xvector_header(&table, speaker, CODEC_LANGUAGE_ENGLISH_ID)
        .map_err(checkpoint_error)?;
    let tts_eos = model.talker.tts_eos(&table);

    // 4. Borrowed weights for the generator.
    let talker_layers = model.talker.talker_layer_weights();
    let micro_layers = model.talker.microdecoder_layer_weights();
    let residual = model.talker.residual_embedding_slices();
    let heads = model.talker.microdecoder_head_slices();
    // The microdecoder's internal tables cover depths 2..=15: the first fourteen of the same
    // fifteen-table set the talker feedback path uses.
    let micro_residual = &residual[..residual.len() - 1];

    let mut generator = QwenGenerator::new_with_artifact(
        QwenGeneratorConfig {
            talker_config: TalkerConfig::default(),
            talker_weights: model.talker.talker_weights(&talker_layers),
            text: model.talker.text_weights(&table),
            feedback: model.talker.feedback_tables(&residual),
            microdecoder_config: MicrodecoderConfig::default(),
            microdecoder_weights: model.talker.microdecoder_weights(
                &micro_layers,
                micro_residual,
                &heads,
            ),
            prompt_mode: PromptMode {
                clone_mode: CloneMode::XVector,
                non_streaming_mode: false,
            },
            header,
            tts_eos,
            reference: None,
            // The PRODUCT samples, exactly as the pinned upstream runtime does
            // (generation_config.json: do_sample=true, T=0.9, top_k=50, repetition_penalty=1.05,
            // subtalker likewise); canonical greedy remains the conformance decoder only. The p7r
            // forensics that certified this path: our talker draw stack matched torch's choices
            // code-for-code for seven straight frames from the same prefill, the silence defect was
            // the subtalker being forced greedy under a sampled talker (a measured silence
            // attractor the reference reproduces in that mismatched configuration), and with the
            // subtalker sampling per depth the engine's utterance envelope matches the reference's
            // sampled runs (peak frame RMS 0.086 with trailing silence). Determinism scope: build +
            // ISA + sampler version + seed, 16 draws per frame.
            sampling_mode: SamplingMode::Production,
            seed,
        },
        model.artifact.as_ref(),
    );

    // 5. The engine owns admission, the budget, cancellation, and the frame loop — and the
    // codec decodes IN PARALLEL with it: a tee on the generator feeds every produced frame
    // through a bounded channel to a scoped codec worker driving the streaming decoder.
    // Streamed output is bit-identical to offline decode under every packet schedule (the
    // standing streaming==batch gate), so this overlap changes wall time and nothing else.
    // Deadlock shape: the worker only ever blocks on `recv` (it always drains), and the
    // generator only ever blocks on `send` when the worker is more than 256 frames behind —
    // bounded, and covered by the engine's rolling frame budget if the worker wedges.
    let preparer = PreparedPassThrough { prepared };
    let (frame_tx, frame_rx) = std::sync::mpsc::sync_channel::<ftts_core::CodeFrame>(256);
    let codec = &model.codec;
    let (result, pcm) = std::thread::scope(
        |scope| -> Result<(ftts_core::SynthesisResult, Vec<f32>), FttsError> {
            let worker = scope.spawn(move || -> Result<Vec<f32>, FttsError> {
                // Overlap for real: this thread's int8 ops run serially on a spare core
                // instead of contending for the generator's worker team.
                ftts_kernels::team::bypass_team_on_this_thread();
                const PACKET_FRAMES: usize = 4;
                let mut state = codec.stream_state();
                let mut pcm = Vec::new();
                // `stream_push` REPLACES its output buffer with one packet's samples (see the
                // streaming==offline test), so packets decode into a scratch and append here.
                let mut packet_pcm = Vec::new();
                let mut packet: Vec<i32> = Vec::with_capacity(16 * PACKET_FRAMES);
                let mut packet_frames = 0_usize;
                while let Ok(frame) = frame_rx.recv() {
                    if frame.codes.len() != 16 {
                        return Err(FttsError::Generic(format!(
                            "generated frame carries {} codes, expected 16",
                            frame.codes.len()
                        )));
                    }
                    for code in &frame.codes {
                        packet.push(i32::try_from(*code).map_err(|_| {
                            FttsError::Generic(format!(
                                "generated code {code} does not fit the codec's i32"
                            ))
                        })?);
                    }
                    packet_frames += 1;
                    if packet_frames == PACKET_FRAMES {
                        codec
                            .stream_push(&mut state, &packet, packet_frames, &mut packet_pcm)
                            .map_err(checkpoint_error)?;
                        pcm.extend_from_slice(&packet_pcm);
                        packet.clear();
                        packet_frames = 0;
                    }
                }
                if packet_frames > 0 {
                    codec
                        .stream_push(&mut state, &packet, packet_frames, &mut packet_pcm)
                        .map_err(checkpoint_error)?;
                    pcm.extend_from_slice(&packet_pcm);
                }
                Ok(pcm)
            });

            let mut tee = TeeGenerator {
                inner: &mut generator,
                frames: frame_tx,
            };
            let result = engine
                .synthesize(
                    request.clone(),
                    &preparer,
                    &mut tee as &mut dyn FrameGenerator,
                    cancellation,
                    observer,
                )
                .map_err(engine_error);
            drop(tee); // closes the channel; the worker drains the tail packet and exits
            let pcm = worker.join().expect("codec worker must not panic")?;
            Ok((result?, pcm))
        },
    )?;

    if result.code_frames.is_empty() {
        return Err(FttsError::Generic(
            "the talker stopped before emitting a frame; there is no audio to write. This is a \
             model or prompt problem, not an output problem — check the speaker vector and the \
             text"
                .to_owned(),
        ));
    }

    Ok(SynthesizedAudio {
        frames: result.generated_frames,
        prepared_token_count: result.prepared_token_count,
        pcm,
    })
}

/// Forwards a generator's frames unchanged while teeing each one to the codec worker.
///
/// A send only fails when the worker has already died with its own error; surfacing a
/// generation error here aborts the engine loop early, and the worker's real failure is
/// reported at join.
struct TeeGenerator<'a> {
    inner: &'a mut dyn FrameGenerator,
    frames: std::sync::mpsc::SyncSender<ftts_core::CodeFrame>,
}

impl FrameGenerator for TeeGenerator<'_> {
    fn begin_utterance(&mut self, prepared: &PreparedText) -> Result<(), GenerationError> {
        self.inner.begin_utterance(prepared)
    }

    fn next_frame(&mut self) -> Result<Option<ftts_core::CodeFrame>, GenerationError> {
        let frame = self.inner.next_frame()?;
        if let Some(frame) = &frame
            && self.frames.send(frame.clone()).is_err()
        {
            return Err(GenerationError::new(
                "the codec worker stopped accepting frames; its error follows at join",
            ));
        }
        Ok(frame)
    }
}

/// Map an engine refusal onto the CLI's exit-code contract.
fn engine_error(error: EngineError) -> FttsError {
    match error {
        EngineError::BudgetExceeded(_) => FttsError::BudgetTimeout(error.to_string()),
        EngineError::ResourceAdmission(_) => FttsError::BudgetTimeout(error.to_string()),
        EngineError::TextPreparation(_) => FttsError::Input(error.to_string()),
        other => FttsError::Generic(other.to_string()),
    }
}

/// A model-side failure, for callers that need the engine's own error type.
#[must_use]
pub fn generation_error(message: &str) -> GenerationError {
    GenerationError::new(message)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Audio already at the pinned rate must come back untouched — the resample path is additive
    /// and may not perturb any enrollment that worked before it existed.
    #[test]
    fn audio_at_the_pinned_rate_is_returned_bit_for_bit() {
        let pcm: Vec<f32> = (0..4_096)
            .map(|n| (n as f32 * 0.017).sin() * 0.4 + (n as f32 * 0.31).sin() * 0.05)
            .collect();
        let out = resample_to_speaker_rate(pcm.clone(), SPEAKER_SAMPLE_RATE_HZ);
        assert_eq!(out.len(), pcm.len());
        for (index, (a, b)) in out.iter().zip(pcm.iter()).enumerate() {
            assert!(
                a.to_bits() == b.to_bits(),
                "sample {index} was altered at the pinned rate"
            );
        }
    }

    /// The denoiser has to do both halves of its job: drop the noise floor between bursts, and
    /// leave the signal itself standing. A filter that achieves the first by attenuating
    /// everything would pass a floor-only check while destroying the voice it was meant to clean.
    #[test]
    fn denoise_lowers_the_floor_between_bursts_without_eating_the_signal() {
        const TONE_HZ: f64 = 700.0;
        let samples = SPEAKER_SAMPLE_RATE_HZ as usize * 2;
        // Deterministic hiss, so the assertion cannot flake on a lucky seed.
        let mut state = 0x2545_F491_4F6C_DD1D_u64;
        let mut noise = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state >> 40) as f32 / 2048.0) - 0.5
        };

        // Half-second bursts of tone alternating with silence, all of it under hiss.
        let clean: Vec<f32> = (0..samples)
            .map(|n| {
                let t = n as f64 / f64::from(SPEAKER_SAMPLE_RATE_HZ);
                let speaking = (n / (SPEAKER_SAMPLE_RATE_HZ as usize / 2)) % 2 == 0;
                if speaking {
                    (std::f64::consts::TAU * TONE_HZ * t).sin() as f32 * 0.35
                } else {
                    0.0
                }
            })
            .collect();
        let noisy: Vec<f32> = clean.iter().map(|s| s + noise() * 0.02).collect();

        let cleaned = denoise_reference(&noisy);
        assert_eq!(cleaned.len(), noisy.len(), "denoise must preserve length");
        assert!(
            cleaned.iter().all(|s| s.is_finite()),
            "denoise produced a non-finite sample"
        );

        let before = pause_floor_dbfs(&noisy);
        let after = pause_floor_dbfs(&cleaned);
        assert!(
            after < before - 3.0,
            "expected the pause floor to drop by >3 dB, got {before:.1} -> {after:.1} dBFS"
        );

        // Energy inside a burst must survive. Compare the loudest quarter-second of each.
        let span = SPEAKER_SAMPLE_RATE_HZ as usize / 4;
        let peak_rms = |pcm: &[f32]| {
            pcm.chunks_exact(span)
                .map(|c| (c.iter().map(|s| s * s).sum::<f32>() / c.len() as f32).sqrt())
                .fold(0.0_f32, f32::max)
        };
        let kept = peak_rms(&cleaned) / peak_rms(&noisy);
        assert!(
            kept > 0.7,
            "denoise removed too much of the signal: peak RMS kept {kept:.3} of the original"
        );
    }

    /// A clip too short to survive its own downsample must round to nothing here rather than
    /// reaching the mel front end as an empty slice — which is why `decode_reference_audio`
    /// re-checks emptiness against the resampled PCM instead of only the decoded PCM.
    #[test]
    fn a_clip_shorter_than_its_downsample_ratio_resamples_to_nothing() {
        let out = resample_to_speaker_rate(vec![0.25], 192_000);
        assert!(
            out.is_empty(),
            "one sample at 192 kHz is less than half an output sample at \
             {SPEAKER_SAMPLE_RATE_HZ} Hz, so it cannot produce one"
        );
    }

    /// A tone that survives the resample proves the kernel is a real lowpass and not a decimator:
    /// 48 kHz is the rate every phone and Mac voice memo records at, and a 1 kHz tone sits well
    /// inside the 12 kHz band that survives the trip to 24 kHz.
    #[test]
    fn a_48k_tone_resamples_to_24k_with_its_shape_intact() {
        const SOURCE_HZ: u32 = 48_000;
        const TONE_HZ: f64 = 1_000.0;
        let samples = SOURCE_HZ as usize; // one second
        let pcm: Vec<f32> = (0..samples)
            .map(|n| {
                let t = n as f64 / f64::from(SOURCE_HZ);
                (std::f64::consts::TAU * TONE_HZ * t).sin() as f32
            })
            .collect();

        let out = resample_to_speaker_rate(pcm, SOURCE_HZ);

        let expected_len = SPEAKER_SAMPLE_RATE_HZ as usize;
        assert!(
            out.len().abs_diff(expected_len) <= 1,
            "expected ~{expected_len} samples at {SPEAKER_SAMPLE_RATE_HZ} Hz, got {}",
            out.len()
        );

        // Compare against the tone sampled directly at the target rate, ignoring the window's
        // run-up at each end where the kernel is truncated by the signal boundary.
        let skip = 64;
        let interior = out.len() - skip;
        let mut worst = 0.0_f32;
        for (index, sample) in out.iter().enumerate().take(interior).skip(skip) {
            let t = index as f64 / f64::from(SPEAKER_SAMPLE_RATE_HZ);
            let ideal = (std::f64::consts::TAU * TONE_HZ * t).sin() as f32;
            worst = worst.max((sample - ideal).abs());
        }
        assert!(
            worst < 0.02,
            "resampled tone drifted from the analytic reference by {worst}"
        );
    }

    #[test]
    fn a_short_speaker_vector_is_refused_rather_than_padded() {
        let dir = std::env::temp_dir().join("ftts-synth-tests");
        fs::create_dir_all(&dir).expect("temp dir");
        let path = dir.join("short.spk");
        fs::write(&path, vec![0u8; 64]).expect("write");
        let error = read_speaker_vector(&path).expect_err("a short vector must be refused");
        let message = error.to_string();
        assert!(message.contains("64 bytes"), "{message}");
        assert!(message.contains("4096"), "{message}");
    }

    #[test]
    fn a_non_finite_speaker_vector_is_refused() {
        let dir = std::env::temp_dir().join("ftts-synth-tests");
        fs::create_dir_all(&dir).expect("temp dir");
        let path = dir.join("nan.spk");
        let mut bytes = vec![0u8; SPEAKER_VECTOR_BYTES];
        bytes[0..4].copy_from_slice(&f32::NAN.to_le_bytes());
        fs::write(&path, &bytes).expect("write");
        let error = read_speaker_vector(&path).expect_err("NaN must be refused");
        assert!(error.to_string().contains("index 0"), "{error}");
    }

    #[test]
    fn a_well_formed_speaker_vector_reads_back_exactly() {
        let dir = std::env::temp_dir().join("ftts-synth-tests");
        fs::create_dir_all(&dir).expect("temp dir");
        let path = dir.join("good.spk");
        let expected: Vec<f32> = (0..TALKER_HIDDEN).map(|i| i as f32 * 0.001).collect();
        let mut bytes = Vec::with_capacity(SPEAKER_VECTOR_BYTES);
        for value in &expected {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        fs::write(&path, &bytes).expect("write");
        assert_eq!(read_speaker_vector(&path).expect("read"), expected);
    }

    #[test]
    fn enrollment_writer_refuses_overwrite_and_preserves_the_vector() {
        let path = std::env::temp_dir().join(format!(
            "ftts-enroll-{}-{}.spk",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("clock")
                .as_nanos()
        ));
        let expected: Vec<f32> = (0..TALKER_HIDDEN)
            .map(|index| index as f32 * 0.125)
            .collect();
        write_speaker_vector_new(&path, &expected).expect("initial enrollment write");
        assert_eq!(
            read_speaker_vector(&path).expect("read enrolled vector"),
            expected
        );
        let error = write_speaker_vector_new(&path, &[0.0; TALKER_HIDDEN])
            .expect_err("an enrollment must never replace an existing voice");
        assert!(error.to_string().contains("without overwriting"), "{error}");
    }

    #[test]
    fn wav_reference_decodes_to_mono_24khz_pcm() {
        let path = std::env::temp_dir().join(format!(
            "ftts-reference-{}-{}.wav",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("clock")
                .as_nanos()
        ));
        let pcm: Vec<f32> = (0..1_920)
            .map(|index| (index as f32 / 1_920.0 * std::f32::consts::TAU).sin() * 0.25)
            .collect();
        fs::write(
            &path,
            ftts_core::audio::encode_wav(&pcm, SPEAKER_SAMPLE_RATE_HZ),
        )
        .expect("write reference WAV");
        let decoded = decode_reference_audio(&path).expect("decode reference WAV");
        assert_eq!(decoded.len(), pcm.len());
        assert!(decoded.iter().all(|sample| sample.is_finite()));
    }

    #[test]
    fn a_bundle_names_the_file_that_is_actually_missing() {
        // An agent that gets "model not found" for a directory holding three of four files cannot
        // act on it; the message must name the one that is absent.
        let dir = std::env::temp_dir().join("ftts-bundle-tests-empty");
        fs::create_dir_all(&dir).expect("temp dir");
        let error = ModelBundle::resolve(&dir).expect_err("an empty directory is not a bundle");
        assert!(error.to_string().contains("model.safetensors"), "{error}");
    }

    #[test]
    fn a_complete_bundle_prefers_its_canonical_artifact_for_synthesis() {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock after epoch")
            .as_nanos();
        let dir = std::env::temp_dir().join(format!(
            "ftts-bundle-canonical-{}-{nonce}",
            std::process::id()
        ));
        fs::create_dir_all(dir.join("speech_tokenizer")).expect("create bundle sidecar directory");
        for name in [
            CANONICAL_MODEL_BASENAME,
            "speech_tokenizer/model.safetensors",
            "vocab.json",
            "merges.txt",
            "tokenizer_config.json",
        ] {
            fs::write(dir.join(name), []).expect("write bundle fixture sidecar");
        }

        let expected_artifact = dir.join(CANONICAL_MODEL_BASENAME);
        let bundle = ModelBundle::resolve(&dir).expect("complete canonical bundle resolves");
        assert_eq!(
            bundle.canonical_main.as_deref(),
            Some(expected_artifact.as_path())
        );
        assert!(
            !bundle.main.exists(),
            "canonical synthesis must not require the raw main checkpoint"
        );

        let explicit = ModelBundle::resolve(&expected_artifact)
            .expect("an explicit canonical artifact resolves against its sidecars");
        assert_eq!(
            explicit.canonical_main.as_deref(),
            Some(expected_artifact.as_path())
        );
    }
}
