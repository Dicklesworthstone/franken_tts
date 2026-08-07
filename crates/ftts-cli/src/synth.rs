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
use symphonia::core::audio::{SampleBuffer, Signal};
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
        let tokenizer = QwenTokenizer::from_files_using_environment(TokenizerFiles {
            vocab_json: &vocab,
            merges_txt: &merges,
            tokenizer_config_json: &config,
        })
        .map_err(|error| FttsError::ArtifactFormat(format!("tokenizer unusable: {error}")))?;

        Ok(Self {
            talker: match bundle.canonical_main.as_deref() {
                Some(path) => TalkerCheckpoint::load_fttsq(path).map_err(checkpoint_error)?,
                None => TalkerCheckpoint::load(&bundle.main).map_err(checkpoint_error)?,
            },
            codec: CodecCheckpoint::load(&bundle.codec).map_err(checkpoint_error)?,
            tokenizer,
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
        .chunks_exact(4)
        .map(|quad| f32::from_le_bytes([quad[0], quad[1], quad[2], quad[3]]))
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
pub fn speaker_from_voice(bundle: &ModelBundle, path: &Path) -> Result<Vec<f32>, FttsError> {
    let bytes = fs::read(path).map_err(|error| {
        FttsError::Input(format!(
            "cannot read voice source {}: {error}",
            path.display()
        ))
    })?;
    if bytes.len() == SPEAKER_VECTOR_BYTES {
        return decode_speaker_vector(path, &bytes);
    }
    let pcm = decode_reference_audio(path)?;
    let mel = log_mel_from_24khz_pcm(&pcm)
        .map_err(|error| FttsError::Input(format!("cannot extract speaker features: {error}")))?;
    let encoder = SpeakerEncoder::load(&bundle.main).map_err(checkpoint_error)?;
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
        .chunks_exact(4)
        .map(|quad| f32::from_le_bytes([quad[0], quad[1], quad[2], quad[3]]))
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
    if rate != SPEAKER_SAMPLE_RATE_HZ {
        return Err(FttsError::Input(format!(
            "reference audio {} is {rate} Hz; the pinned speaker encoder requires {SPEAKER_SAMPLE_RATE_HZ} Hz",
            path.display()
        )));
    }
    if mono.is_empty() {
        return Err(FttsError::Input(format!(
            "reference audio {} contains no PCM samples",
            path.display()
        )));
    }
    Ok(mono)
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

    let mut generator = QwenGenerator::new(QwenGeneratorConfig {
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
        // Greedy is reproducible and consumes no RNG state; `--seed` is still reported so a later
        // switch to the production sampler does not change the event contract.
        sampling_mode: SamplingMode::CanonicalGreedy,
        seed,
    });

    // 5. The engine owns admission, the budget, cancellation, and the frame loop.
    let preparer = PreparedPassThrough { prepared };
    let result = engine
        .synthesize(
            request.clone(),
            &preparer,
            &mut generator as &mut dyn FrameGenerator,
            cancellation,
            observer,
        )
        .map_err(engine_error)?;

    if result.code_frames.is_empty() {
        return Err(FttsError::Generic(
            "the talker stopped before emitting a frame; there is no audio to write. This is a \
             model or prompt problem, not an output problem — check the speaker vector and the \
             text"
                .to_owned(),
        ));
    }

    // 6. Codes to PCM. The codec wants frame-major `i32` groups.
    let frames = result.code_frames.len();
    let mut codes = Vec::with_capacity(frames * 16);
    for frame in &result.code_frames {
        if frame.codes.len() != 16 {
            return Err(FttsError::Generic(format!(
                "generated frame carries {} codes, expected 16",
                frame.codes.len()
            )));
        }
        for code in &frame.codes {
            codes.push(i32::try_from(*code).map_err(|_| {
                FttsError::Generic(format!(
                    "generated code {code} does not fit the codec's i32"
                ))
            })?);
        }
    }
    let pcm = model
        .codec
        .decode(&codes, frames)
        .map_err(checkpoint_error)?;

    Ok(SynthesizedAudio {
        frames: result.generated_frames,
        prepared_token_count: result.prepared_token_count,
        pcm,
    })
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
            "model.safetensors",
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
        assert_eq!(bundle.main, dir.join("model.safetensors"));

        let explicit = ModelBundle::resolve(&expected_artifact)
            .expect("an explicit canonical artifact resolves against its sidecars");
        assert_eq!(
            explicit.canonical_main.as_deref(),
            Some(expected_artifact.as_path())
        );
    }
}
