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
//! # The speaker vector is required, and is not invented
//!
//! An x-vector prompt conditions on a 1,024-wide speaker embedding. Computing one from reference
//! audio is the ECAPA speaker encoder (`frankentts-p1-speaker-ga6`), which is not implemented. So
//! `--voice` here reads a **precomputed** speaker vector: 1,024 little-endian `f32`, 4,096 bytes
//! exactly. This is not the `.ftvoice` format (`frankentts-p4-ftvoice-format-x0p`) and does not
//! pretend to be; it is the minimum honest input that lets the rest of the pipeline run end to
//! end today. Synthesizing with a zeroed or fabricated vector would produce confident nonsense, so
//! there is no default.

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
use ftts_model_qwen::talker::TalkerConfig;
use ftts_model_qwen::tokenizer::{QwenTokenizer, TokenizerFiles};
use std::fs;
use std::path::{Path, PathBuf};

/// Bytes in a speaker-vector file: 1,024 little-endian `f32`.
pub const SPEAKER_VECTOR_BYTES: usize = TALKER_HIDDEN * 4;

fn checkpoint_error(error: CheckpointError) -> FttsError {
    FttsError::ArtifactFormat(error.to_string())
}

/// The four files `ftts say` needs, located relative to one model path.
#[derive(Clone, Debug)]
pub struct ModelBundle {
    /// Directory holding `model.safetensors` and the tokenizer files.
    pub root: PathBuf,
    /// The talker/microdecoder checkpoint.
    pub main: PathBuf,
    /// The codec decoder checkpoint.
    pub codec: PathBuf,
}

impl ModelBundle {
    /// Resolve a bundle from `--model`, which may name the directory or `model.safetensors`.
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
        let main = root.join("model.safetensors");
        let codec = root.join("speech_tokenizer/model.safetensors");
        for (label, path) in [
            ("talker checkpoint", &main),
            ("codec checkpoint", &codec),
            ("tokenizer vocabulary", &root.join("vocab.json")),
            ("tokenizer merges", &root.join("merges.txt")),
            ("tokenizer config", &root.join("tokenizer_config.json")),
        ] {
            if !path.is_file() {
                return Err(FttsError::ModelNotFound(format!(
                    "{label} is missing at {}; `ftts say` needs a complete pinned checkpoint \
                     directory (model.safetensors, speech_tokenizer/model.safetensors, \
                     vocab.json, merges.txt, tokenizer_config.json)",
                    path.display()
                )));
            }
        }
        Ok(Self { root, main, codec })
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
            talker: TalkerCheckpoint::load(&bundle.main).map_err(checkpoint_error)?,
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
    fn a_bundle_names_the_file_that_is_actually_missing() {
        // An agent that gets "model not found" for a directory holding three of four files cannot
        // act on it; the message must name the one that is absent.
        let dir = std::env::temp_dir().join("ftts-bundle-tests-empty");
        fs::create_dir_all(&dir).expect("temp dir");
        let error = ModelBundle::resolve(&dir).expect_err("an empty directory is not a bundle");
        assert!(error.to_string().contains("model.safetensors"), "{error}");
    }
}
