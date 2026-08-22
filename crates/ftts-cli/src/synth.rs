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
use ftts_model_qwen::generate::{prepare_int8_route, Int8Route, QwenGenerator, QwenGeneratorConfig};
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
    /// The checkpoint's own digest-verified mapping of the canonical artifact, shared so the
    /// int8 route hydrates its Q8 tables from it (proven byte-identical to requantizing the
    /// widened f32 copies, scales included). Never re-opened: every `MappedFttsq::open`
    /// re-verifies the whole artifact's digests, ~1.3 GB of hashing.
    artifact: Option<std::sync::Arc<ftts_artifacts::fttsq::MappedFttsq>>,
    /// The fused int8 route, built once per loaded model and lent to every utterance after.
    ///
    /// Building the route is a ~0.46 GB Q8 copy of the hot weights (~146 ms measured); doing it
    /// per utterance taxed every warm synthesis — exactly what the wasm engine fixed by caching
    /// `hydrated.int8` across presses. `None` inside the lock means "the optimized route is off
    /// for this process" (`FTTS_INT8=0`), cached so the env is read once rather than re-litigated
    /// per utterance. The route borrows nothing from the artifact once built, so it outlives any
    /// one generator. Cold one-shot runs build it here exactly where they previously built it
    /// inside [`QwenGenerator::new_with_artifact`] — same inputs, same bytes, same total work.
    int8_route: std::sync::OnceLock<Option<std::sync::Arc<Int8Route>>>,
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
        let talker = talker.map_err(checkpoint_error)?;
        // Shared, not re-opened: a second MappedFttsq::open would re-verify the whole artifact's
        // digests (~1.3 GB of hashing) for a mapping the checkpoint already carries.
        let artifact = talker.artifact().cloned();
        Ok(Self {
            talker,
            codec: codec.map_err(checkpoint_error)?,
            tokenizer,
            artifact,
            // Runtime-input initializer (needs the hydrated weights + artifact), so OnceLock
            // rather than LazyLock per the project's lazy-cell rule.
            int8_route: std::sync::OnceLock::new(),
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

/// Which reference-cleanup stages to run, and where each reports what it measured.
///
/// Both default to off. Every stage changes the enrolled identity, so none of them is applied on
/// a user's behalf — a lever that can alter who the clone sounds like is opted into by name.
#[derive(Default)]
pub struct ReferenceCleanup<'a> {
    /// Spectral-subtract the stationary noise floor.
    pub denoise: Option<&'a mut Option<DenoiseReport>>,
    /// Remove late reverberation.
    pub dereverb: Option<&'a mut Option<DereverbReport>>,
}

/// Derive a speaker vector from a voice source: a raw x-vector file, or reference audio.
///
/// Passing `Some` for a cleanup slot opts the reference into that stage and fills the slot with
/// what it measured. Dereverberation runs first: it is a linear operation on the observed signal,
/// so applying it before the noise floor is estimated keeps that estimate from being fitted to a
/// signal the next stage is about to change.
///
/// # Errors
///
/// When the source cannot be read, decoded, resampled, or encoded into a finite x-vector.
pub fn speaker_from_voice(
    bundle: &ModelBundle,
    path: &Path,
    cleanup: ReferenceCleanup<'_>,
) -> Result<Vec<f32>, FttsError> {
    let bytes = fs::read(path).map_err(|error| {
        FttsError::Input(format!(
            "cannot read voice source {}: {error}",
            path.display()
        ))
    })?;
    // Size alone must not decide: a perfectly plausible 4,096-byte audio clip would
    // otherwise reinterpret as garbage floats and enroll a nonsense voice without any
    // error. A recognizable audio container magic wins over the size sniff.
    let looks_like_audio = bytes.len() >= 12
        && (bytes.starts_with(b"RIFF")
            || bytes.starts_with(b"fLaC")
            || bytes.starts_with(b"ID3")
            || bytes.starts_with(b"OggS")
            || &bytes[4..8] == b"ftyp");
    // A voice-card image (PNG or JPEG) IS a voice: decode it directly, no explicit
    // `ftts card import` step needed. Card decoding never touches the model bundle.
    // Checked BEFORE the raw-vector size branch for the same reason the audio sniff
    // exists: a 4,096-byte image would otherwise reinterpret as garbage floats.
    // The JPEG sniff is three bytes (SOI plus the next marker's FF), not two: every
    // real JPEG has it, and a bare FF D8 matches one legitimate .spk in 65,536 —
    // which would then FAIL to load through the image path instead of speaking.
    let looks_like_image = bytes.starts_with(&[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A])
        || bytes.starts_with(&[0xFF, 0xD8, 0xFF]);
    if looks_like_image {
        let (_, vector) = crate::card::decode_card(&bytes)?;
        return Ok(vector);
    }
    if bytes.len() == SPEAKER_VECTOR_BYTES && !looks_like_audio {
        return decode_speaker_vector(path, &bytes);
    }
    let pcm = decode_reference_audio_any(path)?;
    speaker_from_reference_pcm(bundle, pcm, cleanup)
}

/// Derive a speaker vector from already-decoded mono 24 kHz reference PCM.
///
/// The PCM-level half of [`speaker_from_voice`], split out so hosts that capture audio
/// themselves (the iOS app through `ftts-ffi`) run exactly the enrollment pipeline the
/// CLI runs — same cleanup stages, same encoder, same finiteness refusal — instead of a
/// parallel one that quietly skips the denoiser.
///
/// # Errors
///
/// When feature extraction or encoding fails, or the encoder yields a non-finite vector.
pub fn speaker_from_reference_pcm(
    bundle: &ModelBundle,
    pcm: Vec<f32>,
    cleanup: ReferenceCleanup<'_>,
) -> Result<Vec<f32>, FttsError> {
    let ReferenceCleanup { denoise, dereverb } = cleanup;
    let pcm = match dereverb {
        Some(report) => {
            let before = reverb_time_s(&pcm);
            let dried = dereverb_reference(&pcm);
            let after = reverb_time_s(&dried);
            if let (Some(before), Some(after)) = (before, after) {
                *report = Some(DereverbReport {
                    before_rt60_s: before,
                    after_rt60_s: after,
                });
            }
            dried
        }
        None => pcm,
    };
    let pcm = match denoise {
        Some(report) => {
            let before = pause_floor_dbfs(&pcm);
            let cleaned = match neural_denoise_reference(bundle, &pcm)? {
                Some(cleaned) => cleaned,
                None => denoise_reference(&pcm),
            };
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

/// Replaces an existing enrolled voice, keeping the displaced one alongside it.
///
/// Enrollment is cheap to redo but a voice is not always cheap to re-record, and the reference a
/// `.spk` came from may be long gone. The previous vector is copied to `<path>.bak` before the new
/// one lands, so a mistaken overwrite is one `mv` away from undone rather than unrecoverable.
///
/// # Errors
///
/// When the vector is malformed, the backup cannot be written, or the file cannot be replaced.
pub fn replace_speaker_vector(path: &Path, vector: &[f32]) -> Result<PathBuf, FttsError> {
    let backup = path.with_extension("spk.bak");
    fs::copy(path, &backup).map_err(|error| {
        FttsError::Input(format!(
            "cannot back up the existing voice {} to {}: {error}",
            path.display(),
            backup.display()
        ))
    })?;
    // Write the replacement to a sibling first, then rename over the target: a crash mid-write
    // must not leave a half-written vector where a valid voice used to be.
    let staging = path.with_extension("spk.incoming");
    if staging.exists() {
        fs::remove_file(&staging).map_err(|error| {
            FttsError::Input(format!(
                "cannot clear the stale staging file {}: {error}",
                staging.display()
            ))
        })?;
    }
    write_speaker_vector_new(&staging, vector)?;
    fs::rename(&staging, path).map_err(|error| {
        FttsError::Input(format!(
            "cannot replace {} with the new voice: {error}",
            path.display()
        ))
    })?;
    Ok(backup)
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

/// A user-owned staging directory for temporary artifacts (transcoded references,
/// materialized preset voices).
///
/// This deliberately avoids the shared system temp dir: on Linux, `/tmp` is world-writable,
/// the staging names here are predictable (pid + stem), and the files are written by
/// external decoders that happily follow a pre-planted symlink — so another local user
/// could truncate an arbitrary victim-writable file, or swap content between our write and
/// read. A directory under the user's own cache root closes the whole class. Falls back to
/// the system temp dir only when no home directory exists at all.
pub(crate) fn private_staging_dir() -> std::io::Result<PathBuf> {
    #[allow(deprecated)] // un-deprecated in current Rust; the lint fires on older stables
    let base = std::env::home_dir()
        .map(|home| home.join(".cache/franken_tts"))
        .unwrap_or_else(std::env::temp_dir);
    let dir = base.join("staging");
    fs::create_dir_all(&dir)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(&dir, fs::Permissions::from_mode(0o700))?;
    }
    Ok(dir)
}

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

    let staging_dir = private_staging_dir()
        .map_err(|error| FttsError::Generic(format!("cannot create staging directory: {error}")))?;
    let staging = staging_dir.join(format!(
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
    resample_lanczos(&mono, from_rate, SPEAKER_SAMPLE_RATE_HZ)
}

/// The Lanczos-6 core, rate-agnostic: also carries the denoiser's 24 <-> 48 kHz round trip.
fn resample_lanczos(mono: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
    if from_rate == to_rate {
        return mono.to_vec();
    }
    // Six lobes, not three: with the cutoff at the output Nyquist a 3-lobe kernel's transition
    // band sits inside the passband — measured -2.2 dB at 10 kHz for 48->24 kHz and only ~18 dB
    // of alias rejection, right where the speaker encoder reads sibilance. Six lobes halves the
    // transition width and pushes rejection past 40 dB for double the (still trivial) tap count.
    const LOBES: f64 = 6.0;
    let ratio = f64::from(to_rate) / f64::from(from_rate);
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

/// Floor gain, −35 dB. OM-LSA never gates a bin fully closed: leaving a quiet, *stationary* bed
/// is what stops residual noise from flickering into musical tones.
///
/// This is the right knob for "deeper pauses", and the only safe one — it applies where the
/// presence probability has already decided a bin is noise, so lowering it buys silence between
/// words without touching a bin the estimator thinks holds voice. Reaching for a more aggressive
/// *noise estimate* instead is what damages speech.
const OMLSA_GAIN_FLOOR: f32 = 0.017_782_79;

/// Frames per block over which each bin's noise floor is estimated. ~1.4 s at this hop: long
/// enough that speech is sparse within a block, short enough to follow room tone that drifts.
const NOISE_BLOCK_FRAMES: usize = 256;

/// Prior probability that a bin holds no speech, used in the likelihood ratio. Slightly above a
/// half so that ambiguous bins lean toward suppression rather than passing noise through.
const SPEECH_ABSENCE_PRIOR: f32 = 0.6;

/// Bias compensation for reading a low quantile of an exponentially distributed power as its
/// mean: for `Exp(mu)` the q-quantile is `-mu ln(1-q)`, so the 10th percentile UNDERSTATES the
/// mean 9.49x. Without this factor, pure-pause frames measure a posteriori SNRs of ~7-10 against
/// the uncorrected floor, the presence estimator reads them as speech, and the gain never
/// approaches the floor (measured: 2.5 dB of pause reduction instead of ~20). This is the same
/// role IMCRA's B_min plays for its minimum statistic, recomputed for the quantile used here.
#[allow(clippy::excessive_precision)]
const NOISE_QUANTILE_BIAS: f32 = 9.491_221; // 1 / -ln(1 - NOISE_INIT_QUANTILE)

/// Quantile of each bin's power, across the whole recording, taken as its initial noise floor.
/// Speech is sparse in time, so a low quantile of a bin is the room rather than the voice.
const NOISE_INIT_QUANTILE: f32 = 0.1;

/// Single-channel speech enhancement: MMSE-LSA gains, decision-directed SNR, OM-LSA presence
/// weighting, over a noise floor initialised offline and then tracked recursively.
///
/// Enrollment noise is not cosmetic: it is encoded into the x-vector and then reproduced in every
/// utterance the cloned voice speaks (measured — cleaning a real 53 s reference dropped the
/// synthesized output's pause floor by 19.5 dB). This removes the stationary part of it.
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
/// 3. **OM-LSA** (Cohen & Berdugo 2001) blends that gain toward the floor by the speech-presence
///    probability, `G = G_LSA^p · G_min^(1−p)`, so bins that are probably noise settle to a
///    constant bed rather than being tracked.
///
/// **Why the noise floor is initialised offline rather than by minimum statistics.** IMCRA's
/// online minimum tracking exists because a streaming denoiser cannot see the future. Enrollment
/// can: the file is already on disk, so a low quantile of each bin over the whole recording is a
/// better starting floor than any causal estimator's, with none of the convergence transient.
/// An earlier revision here did run minimum statistics, and it is instructive why that was
/// removed rather than debugged: seeded from frame 0 of a reference that opens on speech, the
/// refined minimum locked above the speech level, which drove the presence probability to zero,
/// which unfroze the noise update, which let the noise estimate absorb the voice — a positive
/// feedback that left ~10% of every burst after the first. Speech presence here instead comes
/// from the likelihood ratio in ξ and ν, which is self-correcting: it cannot conclude "no speech"
/// about a bin whose own a priori SNR is high.
///
/// Nonstationary noise is still tracked, by the recursive average that the presence probability
/// gates — the offline quantile only sets where that average starts.
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
/// Where `ftts pull` lands the neural denoiser, relative to the model root.
pub const DENOISE_ARTIFACT_RELPATH: &str = "denoise/fastenhancer-s-48k.safetensors";

/// The FastEnhancer denoise path: 24 kHz reference up to the model's native 48 kHz,
/// through the ported network, back down to 24 kHz.
///
/// Returns `Ok(None)` when the neural route is unavailable (artifact not pulled) or the
/// user forced the classic engine with `FTTS_DENOISE_ENGINE=omlsa` — the caller falls back
/// to the spectral-subtraction reference. A *present but malformed* artifact is an error,
/// not a silent fallback: repairing it is one `ftts pull --force` away, and quietly handing
/// the user a different denoiser than the one they pulled would misreport what cleaned
/// their reference.
///
/// The network runs at 48 kHz because that is what its checkpoint was trained on (with
/// low-pass augmentation down to 24 kHz content, so band-limited input is in-distribution).
/// The round trip uses the same Lanczos-6 resampler enrollment already trusts, and the
/// input is zero-padded to the model's hop grid so the STFT round trip returns every
/// sample, then trimmed back to the original length.
fn neural_denoise_reference(
    bundle: &ModelBundle,
    pcm24k: &[f32],
) -> Result<Option<Vec<f32>>, FttsError> {
    let path = bundle.root.join(DENOISE_ARTIFACT_RELPATH);
    if !path.is_file() {
        return Ok(None);
    }
    if std::env::var("FTTS_DENOISE_ENGINE").is_ok_and(|v| v.eq_ignore_ascii_case("omlsa")) {
        return Ok(None);
    }
    let enhancer = ftts_artifacts::enhance_loader::open_enhancer(&path).map_err(|error| {
        FttsError::ArtifactFormat(format!(
            "denoiser artifact {} is unreadable ({error}); re-fetch it with `ftts pull --force`",
            path.display()
        ))
    })?;
    Ok(Some(enhancer.enhance_24k(pcm24k)))
}

/// Denoise arbitrary mono 24 kHz PCM through the pulled neural denoiser, if present.
///
/// `Ok(None)` when the neural route is unavailable — the caller keeps the original.
/// Split out for hosts (the iOS app through `ftts-ffi`) that clean synthesized OUTPUT;
/// enrollment keeps its own opt-in path through [`ReferenceCleanup`].
///
/// # Errors
///
/// When the artifact is present but unreadable — a broken denoiser is repaired, not
/// silently swapped for a different one.
pub fn denoise_pcm_24k(bundle: &ModelBundle, pcm: &[f32]) -> Result<Option<Vec<f32>>, FttsError> {
    neural_denoise_reference(bundle, pcm)
}

fn denoise_reference(pcm: &[f32]) -> Vec<f32> {
    // A floor estimated from a handful of frames is just the clip's own spectrum; below ~a
    // quarter second there is nothing honest to subtract, so the clip passes through untouched
    // (a single-frame "estimate" measured as flattening the whole clip toward the gain floor).
    const DENOISE_MIN_FRAMES: usize = 32;
    if pcm.len() < DENOISE_FRAME + (DENOISE_MIN_FRAMES - 1) * DENOISE_HOP {
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

    // Pass 1: every frame's power spectrum. Only the magnitudes are kept; retaining the complex
    // frames would save the second FFT at 8× the memory, which a several-minute reference feels.
    let mut powers: Vec<Vec<f32>> = Vec::with_capacity(starts.len());
    let mut scratch: Vec<rustfft::num_complex::Complex<f32>> =
        vec![rustfft::num_complex::Complex::new(0.0, 0.0); DENOISE_FRAME];
    for &start in &starts {
        for (slot, n) in scratch.iter_mut().zip(0..DENOISE_FRAME) {
            *slot = rustfft::num_complex::Complex::new(pcm[start + n] * window[n], 0.0);
        }
        forward.process(&mut scratch);
        powers.push((0..bins).map(|bin| scratch[bin].norm_sqr()).collect());
    }

    // Noise floor per bin: the MINIMUM over blocks of a low within-block quantile.
    //
    // There is deliberately no feedback here. A recursive noise average has to be gated by a
    // speech-presence estimate, which in turn divides by the noise — and any error in that loop
    // compounds: one speech frame admitted into the floor raises it, which lowers the presence
    // estimate, which admits more speech. Both earlier revisions of this function died that way.
    // Reading the whole file at once removes the loop rather than tuning it.
    //
    // The min-over-blocks reduction is load-bearing: a bare per-block quantile assumes speech is
    // sparse WITHIN every 1.4 s block, and a bin that stays voiced across one whole block (a held
    // vowel, a low harmonic mid-sentence) would have its own signal adopted as that block's floor
    // and be gated to the floor gain — measured at ~28 dB of deletion on a sustained tone. Taking
    // the minimum across blocks only requires the bin to be quiet somewhere in the recording,
    // which is what "noise floor" actually means.
    let blocks = powers.len().div_ceil(NOISE_BLOCK_FRAMES);
    let mut noise = vec![f32::INFINITY; bins];
    let mut column: Vec<f32> = Vec::with_capacity(NOISE_BLOCK_FRAMES);
    for block in 0..blocks {
        let span = block * NOISE_BLOCK_FRAMES..((block + 1) * NOISE_BLOCK_FRAMES).min(powers.len());
        for (bin, slot) in noise.iter_mut().enumerate() {
            column.clear();
            column.extend(powers[span.clone()].iter().map(|frame| frame[bin]));
            column.sort_by(f32::total_cmp);
            let rank = ((column.len() as f32 - 1.0) * NOISE_INIT_QUANTILE).round() as usize;
            *slot = slot.min(column[rank].max(1e-12) * NOISE_QUANTILE_BIAS);
        }
    }

    // Decision-directed state: last frame's gain and a posteriori SNR, per bin.
    let mut prev_gain = vec![1.0_f32; bins];
    let mut prev_gamma = vec![1.0_f32; bins];

    let mut out = vec![0.0_f32; pcm.len()];
    let mut weight = vec![0.0_f32; pcm.len()];

    for (index, &start) in starts.iter().enumerate() {
        let mut frame: Vec<rustfft::num_complex::Complex<f32>> = (0..DENOISE_FRAME)
            .map(|n| rustfft::num_complex::Complex::new(pcm[start + n] * window[n], 0.0))
            .collect();
        forward.process(&mut frame);
        let power = &powers[index];

        for bin in 0..bins {
            let gamma = (power[bin] / noise[bin]).min(1e6);
            let xi = (DD_ALPHA * prev_gain[bin].powi(2) * prev_gamma[bin]
                + (1.0 - DD_ALPHA) * (gamma - 1.0).max(0.0))
            .max(1e-6);

            let nu = (xi / (1.0 + xi)) * gamma;
            let lsa =
                ((xi / (1.0 + xi)) * (0.5 * exponential_integral_e1(nu)).exp()).clamp(0.0, 1.0);

            // Speech-presence probability by the likelihood ratio (Ephraim & Malah's signal
            // presence uncertainty). High ξ with a matching ν drives this to 1, so a bin that is
            // plainly speech can never be talked into being noise.
            let odds = SPEECH_ABSENCE_PRIOR / (1.0 - SPEECH_ABSENCE_PRIOR);
            let presence = 1.0 / (1.0 + odds * (1.0 + xi) * (-nu).exp());
            let presence = presence.clamp(0.0, 1.0);

            let gain = (lsa.max(OMLSA_GAIN_FLOOR).powf(presence)
                * OMLSA_GAIN_FLOOR.powf(1.0 - presence))
            .clamp(OMLSA_GAIN_FLOOR, 1.0);

            prev_gain[bin] = gain;
            prev_gamma[bin] = gamma;

            frame[bin] *= gain;
            let mirror = DENOISE_FRAME - bin;
            // DC (bin 0) has no mirror, and Nyquist (bin N/2) *is* its own mirror — scaling it
            // through this branch as well would apply `gain` twice there.
            if mirror != bin && mirror < DENOISE_FRAME {
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

    // A sample under-covered by the window stack cannot be normalized honestly: near the edges
    // out[n] is dominated by circular-convolution leakage from the rest of the frame, and
    // dividing that by a window energy as small as ~2e-6 manufactures a spike (measured 1.5x
    // input peak with a bare non-zero guard). Anything below a tenth of the steady-state COLA
    // sum (1.5 for periodic Hann at hop N/4) keeps the original PCM instead.
    const WOLA_MIN_WEIGHT: f32 = 0.15;
    for (sample, energy) in out.iter_mut().zip(weight.iter()) {
        if *energy > WOLA_MIN_WEIGHT {
            *sample /= *energy;
        }
    }
    // The head and tail lie outside any fully-stacked window and keep their original samples
    // rather than a partially-normalized reconstruction.
    let covered =
        starts.first().copied().unwrap_or(0)..starts.last().map_or(0, |last| last + DENOISE_FRAME);
    for (index, sample) in out.iter_mut().enumerate() {
        if !covered.contains(&index) || weight[index] <= WOLA_MIN_WEIGHT {
            *sample = pcm[index];
        }
    }
    out
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
            0.001_078_57,
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

/// Dereverberation runs its own STFT, deliberately coarser in time than the denoiser's.
///
/// The two want opposite things. Denoising wants short frames so a gain change lands inside a
/// phoneme; prediction wants each frame to cover enough of the room's tail that a tractable
/// number of taps can span it. At a 5.3 ms hop, a 24-tap filter reaches 128 ms — against an
/// 810 ms reverb that removed 0.01 s of RT60, i.e. nothing (measured). A 10.7 ms hop with 40 taps
/// reaches ~427 ms, which is the fraction of the tail single-channel prediction can model without
/// the covariance becoming both enormous and ill-conditioned.
const DEREVERB_FRAME: usize = 1024;
const DEREVERB_HOP: usize = 256;

/// Prediction taps: ~427 ms of tail at [`DEREVERB_HOP`].
const DEREVERB_TAPS: usize = 40;

/// Frames skipped before prediction starts, so the direct sound and its early reflections are
/// never predictable from the regressor and therefore never subtracted. This delay is the whole
/// reason WPE dereverberates instead of just whitening the voice.
const DEREVERB_DELAY: usize = 2;

/// Alternations between "estimate the speech variance" and "re-fit the filter". The variance
/// estimate is what makes the fit ignore loud speech frames and key on the tail; two passes are
/// enough to converge in practice, three leaves margin.
const DEREVERB_ITERATIONS: usize = 3;

/// Diagonal loading on the covariance, relative to its own trace. Silent bins are rank-deficient
/// and would otherwise produce an arbitrary filter that injects noise instead of removing tail.
const DEREVERB_LOADING: f64 = 1e-4;

/// What a `--dereverb` enrollment measured, so the CLI reports the effect rather than asserting it.
#[derive(Clone, Copy, Debug)]
pub struct DereverbReport {
    /// Reverberation time equivalent of the reference, before dereverberation.
    pub before_rt60_s: f32,
    /// The same measure afterwards.
    pub after_rt60_s: f32,
}

/// Blind single-channel dereverberation by Weighted Prediction Error (Nakatani et al., 2010).
///
/// # Why this is a separate lever from `--denoise`
///
/// Reverb is *convolutive*: the microphone hears the voice convolved with the room's impulse
/// response. Denoising subtracts an *additive* stationary floor. The two do not overlap at all,
/// which is why running the denoiser on a reverberant reference moves the noise floor by 0.0 dB
/// and leaves the wetness untouched — measured, on exactly the recording that prompted this.
///
/// # Why it matters for enrollment specifically
///
/// The speaker encoder cannot separate voice from room, so a wet reference enrolls the room as
/// part of the speaker's identity and every utterance the clone speaks is rendered in that room
/// (measured: a 0.81 s reference produced a 0.79 s clone; a 0.66 s reference produced 0.68 s).
/// Drying the reference is therefore not cosmetic — it changes who the model thinks it is
/// imitating.
///
/// # The method
///
/// Late reverberation at frame `t` is, by construction, a linear function of the *past* of the
/// same signal: it is what earlier sound has decayed into. So per frequency bin, fit a linear
/// predictor from frames `t-D-L+1 ..= t-D` and subtract what it predicts. The delay `D` is what
/// protects the direct path: the speech itself is not predictable at that lag, the room's tail is.
///
/// The weighting is the "WPE" part and the reason it beats plain linear prediction. Each frame is
/// divided by the current estimate of the speech power there, so loud vowels — where the residual
/// is dominated by speech, not tail — stop dominating the fit. Estimating that power needs the
/// dereverberated signal, which needs the filter, so the two alternate for a few iterations.
///
/// Only late reverberation is removed. Early reflections arrive inside the protected delay by
/// design, so a very close, very live room is improved less than a distant one.
fn dereverb_reference(pcm: &[f32]) -> Vec<f32> {
    if pcm.len() < DEREVERB_FRAME * 4 {
        return pcm.to_vec();
    }
    let mut planner = rustfft::FftPlanner::<f32>::new();
    let forward = planner.plan_fft_forward(DEREVERB_FRAME);
    let inverse = planner.plan_fft_inverse(DEREVERB_FRAME);

    let window: Vec<f32> = (0..DEREVERB_FRAME)
        .map(|n| {
            let phase = std::f32::consts::TAU * n as f32 / DEREVERB_FRAME as f32;
            0.5 - 0.5 * phase.cos()
        })
        .collect();

    let bins = DEREVERB_FRAME / 2 + 1;
    let starts: Vec<usize> = (0..=pcm.len() - DEREVERB_FRAME)
        .step_by(DEREVERB_HOP)
        .collect();
    let frames = starts.len();
    if frames <= DEREVERB_DELAY + DEREVERB_TAPS + 2 {
        return pcm.to_vec();
    }

    // Observed spectra, kept complex: prediction needs phase, unlike the magnitude-only denoiser.
    let mut observed: Vec<Vec<Complex64>> = Vec::with_capacity(frames);
    let mut scratch: Vec<rustfft::num_complex::Complex<f32>> =
        vec![rustfft::num_complex::Complex::new(0.0, 0.0); DEREVERB_FRAME];
    for &start in &starts {
        for (slot, n) in scratch.iter_mut().zip(0..DEREVERB_FRAME) {
            *slot = rustfft::num_complex::Complex::new(pcm[start + n] * window[n], 0.0);
        }
        forward.process(&mut scratch);
        observed.push(
            scratch[..bins]
                .iter()
                .map(|value| Complex64::new(f64::from(value.re), f64::from(value.im)))
                .collect(),
        );
    }

    let mut desired = observed.clone();
    for _ in 0..DEREVERB_ITERATIONS {
        for bin in 0..bins {
            // Speech power per frame, from the current estimate. The floor keeps a silent frame
            // from receiving unbounded weight and hijacking the fit.
            let mut power: Vec<f64> = (0..frames).map(|t| desired[t][bin].norm_sqr()).collect();
            let mean = power.iter().sum::<f64>() / frames as f64;
            let floor = (mean * 1e-6).max(1e-12);
            for value in &mut power {
                *value = value.max(floor);
            }

            let taps = DEREVERB_TAPS;
            let mut covariance = vec![Complex64::new(0.0, 0.0); taps * taps];
            let mut cross = vec![Complex64::new(0.0, 0.0); taps];
            for t in (DEREVERB_DELAY + taps)..frames {
                let weight = 1.0 / power[t];
                // Regressor: the observed signal at increasing lag past the protected delay.
                let regressor: Vec<Complex64> = (0..taps)
                    .map(|lag| observed[t - DEREVERB_DELAY - lag][bin])
                    .collect();
                for row in 0..taps {
                    let scaled = regressor[row] * weight;
                    for column in row..taps {
                        covariance[row * taps + column] += scaled * regressor[column].conj();
                    }
                    cross[row] += scaled * observed[t][bin].conj();
                }
            }
            // Hermitian: fill the lower triangle from the upper one that was accumulated.
            for row in 0..taps {
                for column in 0..row {
                    covariance[row * taps + column] = covariance[column * taps + row].conj();
                }
            }
            let trace: f64 = (0..taps).map(|i| covariance[i * taps + i].re).sum();
            if trace <= 0.0 {
                continue;
            }
            let loading = trace / taps as f64 * DEREVERB_LOADING;
            for i in 0..taps {
                covariance[i * taps + i] += Complex64::new(loading, 0.0);
            }

            let Some(filter) = solve_complex_system(&mut covariance, &mut cross, taps) else {
                continue;
            };
            for t in 0..frames {
                if t < DEREVERB_DELAY + taps {
                    desired[t][bin] = observed[t][bin];
                    continue;
                }
                let mut tail = Complex64::new(0.0, 0.0);
                for (lag, coefficient) in filter.iter().enumerate() {
                    tail += coefficient.conj() * observed[t - DEREVERB_DELAY - lag][bin];
                }
                desired[t][bin] = observed[t][bin] - tail;
            }
        }
    }

    // Overlap-add the dereverberated spectra back, mirroring the conjugate half so the inverse
    // transform yields a real signal.
    let mut out = vec![0.0_f32; pcm.len()];
    let mut weight = vec![0.0_f32; pcm.len()];
    for (index, &start) in starts.iter().enumerate() {
        let mut frame = vec![rustfft::num_complex::Complex::new(0.0_f32, 0.0); DEREVERB_FRAME];
        for bin in 0..bins {
            let value = desired[index][bin];
            #[allow(clippy::cast_possible_truncation)]
            let value = rustfft::num_complex::Complex::new(value.re as f32, value.im as f32);
            frame[bin] = value;
            let mirror = DEREVERB_FRAME - bin;
            if mirror != bin && mirror < DEREVERB_FRAME {
                frame[mirror] = value.conj();
            }
        }
        inverse.process(&mut frame);
        let scale = 1.0 / DEREVERB_FRAME as f32;
        for n in 0..DEREVERB_FRAME {
            out[start + n] += frame[n].re * scale * window[n];
            weight[start + n] += window[n] * window[n];
        }
    }
    for (sample, energy) in out.iter_mut().zip(weight.iter()) {
        if *energy > 1e-6 {
            *sample /= *energy;
        }
    }
    let covered =
        starts.first().copied().unwrap_or(0)..starts.last().map_or(0, |last| last + DEREVERB_FRAME);
    for (index, sample) in out.iter_mut().enumerate() {
        if !covered.contains(&index) || weight[index] <= 1e-6 {
            *sample = pcm[index];
        }
    }
    out
}

/// Double-precision complex scalar for the normal equations.
///
/// The covariance is accumulated over thousands of frames and then inverted; doing that in f32
/// loses conditioning on quiet bins, which is where a bad filter does the most audible damage.
type Complex64 = rustfft::num_complex::Complex<f64>;

/// Solves `a x = b` by Gaussian elimination with partial pivoting, consuming both.
///
/// Returns `None` when the system is singular to working precision, which the caller treats as
/// "leave this bin alone" rather than as a failure — a bin with no energy has no tail to remove.
fn solve_complex_system(
    a: &mut [Complex64],
    b: &mut [Complex64],
    n: usize,
) -> Option<Vec<Complex64>> {
    for column in 0..n {
        let (pivot, magnitude) = (column..n).fold((column, 0.0_f64), |best, row| {
            let candidate = a[row * n + column].norm_sqr();
            if candidate > best.1 {
                (row, candidate)
            } else {
                best
            }
        });
        if magnitude <= f64::MIN_POSITIVE {
            return None;
        }
        if pivot != column {
            for k in 0..n {
                a.swap(pivot * n + k, column * n + k);
            }
            b.swap(pivot, column);
        }
        let diagonal = a[column * n + column];
        for row in (column + 1)..n {
            let factor = a[row * n + column] / diagonal;
            if factor == Complex64::new(0.0, 0.0) {
                continue;
            }
            for k in column..n {
                let value = a[column * n + k] * factor;
                a[row * n + k] -= value;
            }
            let value = b[column] * factor;
            b[row] -= value;
        }
    }
    let mut solution = vec![Complex64::new(0.0, 0.0); n];
    for row in (0..n).rev() {
        let mut accumulator = b[row];
        for k in (row + 1)..n {
            accumulator -= a[row * n + k] * solution[k];
        }
        solution[row] = accumulator / a[row * n + row];
    }
    Some(solution)
}

/// Reverberation time equivalent, in seconds, from the decay following speech offsets.
///
/// Reverb leaves no trace in a noise floor — what it does is stretch the energy envelope after
/// every stop. Measuring the median decay slope across offsets is what separates "the room rings"
/// from "the microphone hisses", two problems whose fixes have nothing in common. Returns `None`
/// when the audio has no clear offsets to measure.
fn reverb_time_s(pcm: &[f32]) -> Option<f32> {
    let hop = (SPEAKER_SAMPLE_RATE_HZ as usize) / 100; // 10 ms
    if pcm.len() < hop * 32 {
        return None;
    }
    let envelope: Vec<f32> = pcm
        .chunks_exact(hop)
        .map(|chunk| {
            let energy = chunk.iter().map(|s| s * s).sum::<f32>() / chunk.len() as f32;
            10.0 * (energy + 1e-9).log10()
        })
        .collect();
    let peak = envelope.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let span = 15_usize; // 150 ms of decay
    let mut slopes: Vec<f32> = Vec::new();
    for index in 1..envelope.len().saturating_sub(span) {
        if envelope[index] < peak - 25.0 || envelope[index] <= envelope[index - 1] {
            continue;
        }
        let drop = envelope[index] - envelope[index + span - 1];
        if drop < 6.0 {
            continue;
        }
        slopes.push(drop / (span as f32 * 0.01));
    }
    if slopes.is_empty() {
        return None;
    }
    slopes.sort_by(f32::total_cmp);
    let median = slopes[slopes.len() / 2];
    (median > 0.0).then(|| 60.0 / median)
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
    /// Time from synthesis start (prompt work + prefill + first frames) to the first decoded
    /// packet of PCM existing — and, when a [`PcmPacketSink`] is attached, to that packet
    /// having been DELIVERED through it, so live callers get an honest delivery time. `None`
    /// when the run produced no audio. Time-to-first-audio and real-time factor are different
    /// products (doctrine: report them separately); this is the TTFA half, excluding model
    /// load, which the `load` stage event already bounds.
    pub ttfa: Option<std::time::Duration>,
}

/// Receives each decoded PCM packet on the codec worker thread, the moment it exists.
///
/// `samples` are one packet's mono 24 kHz samples in `[-1, 1]` — `frames` × 1,920 of them,
/// fewer never; only the final packet of an utterance may carry fewer than the configured
/// packet's frames. Delivery happens before the packet joins the whole-utterance buffer, so
/// a consumer that blocks in `deliver` backpressures the codec worker, which backpressures
/// the generator through the bounded frame channel — that chain is the intended flow
/// control for live consumers, not a hazard. A `deliver` error ends the run: the worker
/// carries it to its join exactly like a codec failure, and the engine loop aborts on the
/// closed frame channel.
pub trait PcmPacketSink: Send {
    /// Consumes one decoded packet.
    ///
    /// # Errors
    ///
    /// Any error aborts the synthesis run; the whole-utterance buffer is not returned.
    fn deliver(&mut self, samples: &[f32], frames: usize) -> Result<(), FttsError>;
}

/// Run one utterance end to end: text, codes, PCM.
///
/// `pcm_sink`, when present, receives every decoded packet during synthesis (see
/// [`PcmPacketSink`]); the completed [`SynthesizedAudio`] is returned identically either
/// way, so the sink is purely additive observation with backpressure.
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
    pcm_sink: Option<&mut dyn PcmPacketSink>,
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

    // The fused int8 tables are a process asset, not an utterance one: the first utterance on
    // this LoadedModel pays the ~146 ms Q8-table build and every later one borrows the same
    // Arc (the wasm engine has cached `hydrated.int8` across presses since its first release).
    // new_with_artifact would rebuild identical tables per call from the same weights and
    // artifact; lending the prepared ones changes no bytes — the route is built from exactly
    // the config below, and assemble() is shared by both constructors.
    let generator_config = QwenGeneratorConfig {
        talker_config: TalkerConfig::default(),
        talker_weights: model.talker.talker_weights(&talker_layers),
        text: model.talker.text_weights(&table),
        feedback: model.talker.feedback_tables(&residual),
        microdecoder_config: MicrodecoderConfig::default(),
        microdecoder_weights: model.talker.microdecoder_weights(&micro_layers, micro_residual, &heads),
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
    };
    let int8 = model.int8_route.get_or_init(|| {
        ftts_kernels::route::optimized_default("FTTS_INT8")
            .then(|| {
                prepare_int8_route(
                    &generator_config.talker_config,
                    &generator_config.talker_weights,
                    &generator_config.microdecoder_config,
                    &generator_config.microdecoder_weights,
                    model.artifact.as_deref(),
                )
            })
            .flatten()
            .map(std::sync::Arc::new)
    });
    let mut generator = QwenGenerator::new_with_prepared_int8(generator_config, int8.clone());

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
    let synthesis_started = std::time::Instant::now();
    let (result, pcm, ttfa) = std::thread::scope(
        |scope| -> Result<
            (
                ftts_core::SynthesisResult,
                Vec<f32>,
                Option<std::time::Duration>,
            ),
            FttsError,
        > {
            let mut pcm_sink = pcm_sink;
            let worker = scope.spawn(
                move || -> Result<(Vec<f32>, Option<std::time::Duration>), FttsError> {
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
                    let mut first_audio_at: Option<std::time::Duration> = None;
                    // One decoded packet leaves the worker: live delivery first (a blocking or
                    // failing sink is the flow-control/abort contract on `PcmPacketSink`), then
                    // the whole-utterance buffer, then the TTFA mark — so with a sink attached
                    // `ttfa` is a DELIVERY time, not merely "the samples exist".
                    let emit_packet = |sink: &mut Option<&mut dyn PcmPacketSink>,
                                       pcm: &mut Vec<f32>,
                                       packet_pcm: &[f32],
                                       frames: usize,
                                       first_audio_at: &mut Option<std::time::Duration>|
                     -> Result<(), FttsError> {
                        if let Some(sink) = sink.as_mut() {
                            sink.deliver(packet_pcm, frames)?;
                        }
                        pcm.extend_from_slice(packet_pcm);
                        first_audio_at.get_or_insert_with(|| synthesis_started.elapsed());
                        Ok(())
                    };
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
                            emit_packet(
                                &mut pcm_sink,
                                &mut pcm,
                                &packet_pcm,
                                packet_frames,
                                &mut first_audio_at,
                            )?;
                            packet.clear();
                            packet_frames = 0;
                        }
                    }
                    if packet_frames > 0 {
                        codec
                            .stream_push(&mut state, &packet, packet_frames, &mut packet_pcm)
                            .map_err(checkpoint_error)?;
                        emit_packet(
                            &mut pcm_sink,
                            &mut pcm,
                            &packet_pcm,
                            packet_frames,
                            &mut first_audio_at,
                        )?;
                    }
                    Ok((pcm, first_audio_at))
                },
            );

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
            let worker_outcome = worker.join().expect("codec worker must not panic");
            // Error precedence: a worker `Err` is always the ROOT cause. A worker merely
            // starved by an engine failure does not error — its recv loop ends on the
            // disconnect and it returns the partial PCM it decoded — so the only way the
            // worker carries an error is a genuine codec/format/sink failure, which is
            // also what killed the engine loop through the tee's failed send. Reporting
            // the engine's induced "worker stopped accepting frames" instead would bury
            // the sink's abort reason — the barge-in path's actual signal.
            match (result, worker_outcome) {
                (_, Err(worker_error)) => Err(worker_error),
                (Err(engine_failure), Ok(_)) => Err(engine_failure),
                (Ok(result), Ok((pcm, ttfa))) => Ok((result, pcm, ttfa)),
            }
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
        ttfa,
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

    /// `E₁` sets the MMSE-LSA gain at every bin of every frame, so a mistyped coefficient would
    /// quietly bias the whole denoiser instead of failing. References computed from the
    /// convergent series `−γ − ln x + Σ (−1)^{k+1} x^k /(k·k!)`, a different algorithm from the
    /// rational fits under test.
    ///
    /// The tolerance is tight on purpose: a single-digit slip in the fifth-order coefficient
    /// perturbs `E₁(0.9)` by ~7.6e-7, which a looser bound would wave through.
    #[test]
    fn the_exponential_integral_matches_its_series_expansion() {
        // (x, E₁(x)) — the series branch, x < 1.
        for (x, expected) in [
            (0.1_f32, 1.822_923_9_f32),
            (0.5, 0.559_773_6),
            (0.9, 0.260_183_94),
        ] {
            let actual = exponential_integral_e1(x);
            let relative = ((actual - expected) / expected).abs();
            assert!(
                relative < 1e-6,
                "E1({x}) = {actual} but the series gives {expected} (relative {relative:e})"
            );
        }

        // E₁ is positive and strictly decreasing; the two branches must agree where they meet.
        let below = exponential_integral_e1(0.999_9);
        let above = exponential_integral_e1(1.000_1);
        assert!(
            below > above && (below - above).abs() < 1e-4,
            "the series and rational branches disagree across x = 1: {below} vs {above}"
        );
        assert_eq!(
            exponential_integral_e1(0.0),
            0.0,
            "a non-positive argument must not produce NaN"
        );
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
            ((state >> 40) as f32 / 16_777_216.0) - 0.5
        };

        // Half-second bursts of tone alternating with silence, all of it under hiss.
        let clean: Vec<f32> = (0..samples)
            .map(|n| {
                let t = n as f64 / f64::from(SPEAKER_SAMPLE_RATE_HZ);
                let speaking = (n / (SPEAKER_SAMPLE_RATE_HZ as usize / 2)).is_multiple_of(2);
                if speaking {
                    (std::f64::consts::TAU * TONE_HZ * t).sin() as f32 * 0.35
                } else {
                    0.0
                }
            })
            .collect();
        // Hiss at ~-26 dBFS against a 0.35 tone (~17 dB SNR): the audible-voice-memo regime
        // this lever exists for. (An earlier revision's generator bug made the "hiss" 4000x
        // louder than the signal, and the assertions below were calibrated against artifacts.)
        let noisy: Vec<f32> = clean.iter().map(|s| s + noise() * 0.1).collect();

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

    /// A reference that opens on speech, with no leading room tone to learn from, must keep that
    /// opening — most voice memos start the moment recording does.
    ///
    /// The probe is deliberately voice-*like*: a harmonic stack with vibrato and an amplitude
    /// envelope. That matters, because a steady unvarying partial is spectrally what a hum is,
    /// and suppressing it is correct behaviour rather than a bug — an earlier version of this
    /// test used a bare sine and was measuring the denoiser doing its job.
    #[test]
    fn denoise_keeps_a_reference_that_opens_on_speech() {
        let rate = SPEAKER_SAMPLE_RATE_HZ as usize;
        let mut state = 0x9E37_79B9_7F4A_7C15_u64;
        let mut hiss = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state >> 40) as f32 / 16_777_216.0) - 0.5
        };
        let burst = rate / 4;
        let noisy: Vec<f32> = (0..rate * 3)
            .map(|n| {
                let t = n as f64 / rate as f64;
                let index = n / burst;
                let voice = if index.is_multiple_of(2) {
                    // Vibrato and a syllable envelope keep the partials moving, which is what
                    // distinguishes a voice from a tone to any minimum/quantile noise estimator.
                    let vibrato = 1.0 + 0.03 * (std::f64::consts::TAU * 5.5 * t).sin();
                    let phase = (n % burst) as f32 / burst as f32;
                    let envelope = (std::f32::consts::PI * phase).sin();
                    let f0 = 140.0 * vibrato * (1.0 + 0.15 * (index / 2) as f64);
                    (1..=10)
                        .map(|h| {
                            let a = 0.3 / h as f32;
                            (std::f64::consts::TAU * f0 * h as f64 * t).sin() as f32 * a
                        })
                        .sum::<f32>()
                        * envelope
                } else {
                    0.0
                };
                voice + hiss() * 0.02
            })
            .collect();

        let cleaned = denoise_reference(&noisy);
        let rms = |pcm: &[f32]| (pcm.iter().map(|s| s * s).sum::<f32>() / pcm.len() as f32).sqrt();
        let kept = rms(&cleaned[..burst]) / rms(&noisy[..burst]);
        assert!(
            kept > 0.7,
            "the opening burst kept only {kept:.3} of its energy; the noise floor is being \
             seeded from speech the estimator has not yet learned to exclude"
        );
    }

    /// Denoising must not buy a quiet floor by dulling the voice.
    ///
    /// High frequencies are where this fails first and where it matters most: broadband hiss
    /// overlaps sibilance almost exactly, so a suppressor tuned by overall SNR happily trades
    /// away 4–10 kHz — and the speaker encoder reads sibilance as identity, so that trade shows
    /// up as a duller *and less recognizable* clone rather than merely a duller one.
    ///
    /// The probe is broadband speech-like content: every band must survive comparably, so the
    /// assertion is on the spread across bands, not on any single band's absolute retention.
    #[test]
    fn denoise_does_not_preferentially_zap_high_frequencies() {
        let rate = SPEAKER_SAMPLE_RATE_HZ as usize;
        let mut state = 0xDEAD_BEEF_1234_5678_u64;
        let mut hiss = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state >> 40) as f32 / 16_777_216.0) - 0.5
        };
        // Equal-amplitude tones spanning the band, gated into syllable-like bursts so the
        // estimator treats them as speech rather than as hum.
        let probes: [f64; 5] = [300.0, 1_200.0, 3_000.0, 6_000.0, 9_000.0];
        let burst = rate / 4;
        let noisy: Vec<f32> = (0..rate * 3)
            .map(|n| {
                let t = n as f64 / rate as f64;
                let voice = if (n / burst).is_multiple_of(2) {
                    let phase = (n % burst) as f32 / burst as f32;
                    let envelope = (std::f32::consts::PI * phase).sin();
                    probes
                        .iter()
                        .map(|hz| (std::f64::consts::TAU * hz * t).sin() as f32 * 0.12)
                        .sum::<f32>()
                        * envelope
                } else {
                    0.0
                };
                voice + hiss() * 0.02
            })
            .collect();

        let cleaned = denoise_reference(&noisy);

        // Per-probe retention, measured by projecting each band onto its own tone (a one-bin
        // Goertzel-style correlation) inside a burst.
        let span = burst / 2..burst;
        let energy_at = |pcm: &[f32], hz: f64| -> f32 {
            let (mut re, mut im) = (0.0_f64, 0.0_f64);
            for (offset, sample) in pcm[span.clone()].iter().enumerate() {
                let t = (span.start + offset) as f64 / rate as f64;
                let angle = std::f64::consts::TAU * hz * t;
                re += f64::from(*sample) * angle.cos();
                im += f64::from(*sample) * angle.sin();
            }
            (re.hypot(im) / span.len() as f64) as f32
        };

        let retention: Vec<f32> = probes
            .iter()
            .map(|hz| energy_at(&cleaned, *hz) / energy_at(&noisy, *hz).max(1e-9))
            .collect();
        let low = retention[0];
        for (hz, kept) in probes.iter().zip(retention.iter()) {
            assert!(
                *kept > 0.5,
                "{hz} Hz retained only {kept:.3}; the denoiser is eating the band, not the noise \
                 (all bands: {retention:?})"
            );
            assert!(
                *kept > low * 0.6,
                "{hz} Hz retained {kept:.3} against {low:.3} at 300 Hz — high frequencies are \
                 being attenuated preferentially, which is how sibilance and speaker identity go \
                 (all bands: {retention:?})"
            );
        }
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
