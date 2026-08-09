//! franken_tts in the browser: the Qwen3-TTS pipeline behind `wasm-bindgen`.
//!
//! The engine here is the same library the CLI runs — same checkpoint hydration, same
//! generator, same codec — loaded from in-memory buffers because wasm has no filesystem.
//! Everything stays on the caller's thread: the KernelTeam is compiled out on wasm32
//! (`ftts_kernels::team::armed` returns `None`), hydration takes its serial walks, and the
//! int8 route hydrates artifact-natively from the digest-verified buffer, so hot weights are
//! never widened to f32 — the difference between fitting in a browser heap and not.
//!
//! JS-facing conventions: methods return plain data (`Float32Array` PCM, JSON strings) and
//! errors surface as thrown strings, never panics — a panic in wasm is an opaque
//! `unreachable`, so every fallible edge maps to `Result<_, JsValue>`.

use ftts_core::{FrameGenerator, NormalizationOptions, PreparedText, TextPreparer};
use ftts_model_qwen::checkpoint::{
    CODEC_LANGUAGE_ENGLISH_ID, CodecCheckpoint, TALKER_HIDDEN, TalkerCheckpoint,
};
use ftts_model_qwen::generate::{QwenGenerator, QwenGeneratorConfig};
use ftts_model_qwen::microdecoder::MicrodecoderConfig;
use ftts_model_qwen::prompt::{CloneMode, PromptMode};
use ftts_model_qwen::sampler::SamplingMode;
use ftts_model_qwen::speaker::{Encoder as SpeakerEncoder, log_mel_from_24khz_pcm};
use ftts_model_qwen::talker::TalkerConfig;
use ftts_model_qwen::tokenizer::{QwenTokenizer, TokenizerFiles};
use std::path::Path;
use std::sync::Arc;
use wasm_bindgen::prelude::*;

/// The built-in voices, mirrored from the CLI's preset table (same files, same names).
const PRESET_VOICES: &[(&str, &str, &[u8])] = &[
    (
        "matt",
        "warm, easy, masculine — the out-of-box default",
        include_bytes!("../../ftts-cli/presets/matt.spk"),
    ),
    (
        "james",
        "natural, conversational, masculine",
        include_bytes!("../../ftts-cli/presets/james.spk"),
    ),
    (
        "leo",
        "relaxed, resonant, masculine",
        include_bytes!("../../ftts-cli/presets/leo.spk"),
    ),
    (
        "robert",
        "steady, measured, masculine",
        include_bytes!("../../ftts-cli/presets/robert.spk"),
    ),
    (
        "judy",
        "bright, articulate, feminine",
        include_bytes!("../../ftts-cli/presets/judy.spk"),
    ),
    (
        "aria",
        "clear, warm, feminine",
        include_bytes!("../../ftts-cli/presets/aria.spk"),
    ),
    (
        "ember",
        "the same character a few semitones deeper",
        include_bytes!("../../ftts-cli/presets/ember.spk"),
    ),
];

fn js_error(context: &str, error: impl std::fmt::Display) -> JsValue {
    JsValue::from_str(&format!("{context}: {error}"))
}

/// Names and one-line characters of the built-in voices, as JSON.
#[wasm_bindgen]
#[must_use]
pub fn presets() -> String {
    let list: Vec<serde_json::Value> = PRESET_VOICES
        .iter()
        .map(|(name, character, _)| serde_json::json!({ "name": name, "character": character }))
        .collect();
    serde_json::Value::Array(list).to_string()
}

/// The 1,024-float x-vector of a built-in voice.
///
/// # Errors
///
/// Throws when the name is not a built-in.
#[wasm_bindgen]
pub fn preset_vector(name: &str) -> Result<Vec<f32>, JsValue> {
    let (_, _, bytes) = PRESET_VOICES
        .iter()
        .find(|(preset, _, _)| *preset == name)
        .ok_or_else(|| js_error("unknown preset", name))?;
    Ok(bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|chunk| f32::from_le_bytes(*chunk))
        .collect())
}

/// The loaded model: talker+microdecoder from the canonical artifact, codec from the raw
/// speech-tokenizer checkpoint, tokenizer from its three text files.
#[wasm_bindgen]
pub struct WasmEngine {
    talker: TalkerCheckpoint,
    codec: CodecCheckpoint,
    tokenizer: QwenTokenizer,
    artifact: Arc<ftts_artifacts::fttsq::MappedFttsq>,
    speaker_encoder: Option<SpeakerEncoder>,
}

#[wasm_bindgen]
impl WasmEngine {
    /// Hydrate the engine from in-memory buffers.
    ///
    /// `fttsq` is the canonical quantized artifact (digest-verified here before any tensor is
    /// read); `codec` is `speech_tokenizer/model.safetensors`; the three strings are the
    /// tokenizer files, byte-for-byte as pulled.
    ///
    /// # Errors
    ///
    /// Throws with the failing stage named: artifact verification, codec parse, hydration, or
    /// tokenizer construction.
    #[wasm_bindgen(constructor)]
    #[allow(clippy::needless_pass_by_value)]
    #[cfg(not(unix))]
    pub fn new(
        fttsq: Vec<u8>,
        codec: Vec<u8>,
        vocab_json: String,
        merges_txt: String,
        tokenizer_config_json: String,
    ) -> Result<WasmEngine, JsValue> {
        let artifact = Arc::new(
            ftts_artifacts::fttsq::MappedFttsq::from_bytes(fttsq)
                .map_err(|error| js_error("artifact rejected", error))?,
        );

        let label = Path::new("browser://model.fttsq");
        let talker = TalkerCheckpoint::load_fttsq_mapped(
            Arc::clone(&artifact),
            label,
            ftts_model_qwen::generate::hot_elision_from_environment(),
        )
        .map_err(|error| js_error("talker hydration failed", error))?;

        let codec_file = ftts_artifacts::safetensors::SafetensorsFile::from_bytes(codec)
            .map_err(|error| js_error("codec checkpoint rejected", error))?;
        let codec =
            CodecCheckpoint::load_from_file(&codec_file, Path::new("browser://codec.safetensors"))
                .map_err(|error| js_error("codec hydration failed", error))?;

        let tokenizer = QwenTokenizer::from_files_using_environment(TokenizerFiles {
            vocab_json: &vocab_json,
            merges_txt: &merges_txt,
            tokenizer_config_json: &tokenizer_config_json,
        })
        .map_err(|error| js_error("tokenizer unusable", error))?;

        Ok(WasmEngine {
            talker,
            codec,
            tokenizer,
            artifact,
            speaker_encoder: None,
        })
    }

    /// The unix stub: byte-based construction is the wasm path; native hosts use the CLI.
    ///
    /// # Errors
    ///
    /// Always throws — it exists so the crate compiles in native workspace gates.
    #[wasm_bindgen(constructor)]
    #[allow(clippy::needless_pass_by_value)]
    #[cfg(unix)]
    pub fn new(
        fttsq: Vec<u8>,
        codec: Vec<u8>,
        vocab_json: String,
        merges_txt: String,
        tokenizer_config_json: String,
    ) -> Result<WasmEngine, JsValue> {
        let _ = (fttsq, codec, vocab_json, merges_txt, tokenizer_config_json);
        Err(js_error(
            "unsupported host",
            "byte-based engine construction is the wasm path; native hosts use the CLI",
        ))
    }

    /// Synthesize `text` with a 1,024-float speaker vector; returns mono 24 kHz PCM in
    /// `[-1, 1]`.
    ///
    /// The production sampling stack (seeded, deterministic per build+seed) drives both the
    /// talker and the subtalker, exactly as the CLI's `say`. `max_frames` bounds runaway
    /// generation; 0 selects the CLI's text-proportional backstop.
    ///
    /// # Errors
    ///
    /// Throws on an ill-shaped speaker vector, text preparation failure, or a decode error.
    pub fn synthesize(
        &self,
        text: &str,
        speaker: &[f32],
        seed: u64,
        max_frames: u32,
    ) -> Result<Vec<f32>, JsValue> {
        if speaker.len() != TALKER_HIDDEN {
            return Err(js_error(
                "speaker vector must be exactly 1,024 floats",
                speaker.len(),
            ));
        }
        let prepared_raw = self
            .tokenizer
            .prepare(text, &NormalizationOptions::default())
            .map_err(|error| js_error("text preparation failed", error))?;
        let wrapped = TalkerCheckpoint::wrap_target_ids(&prepared_raw.token_ids);
        let prepared = PreparedText::new(wrapped.clone(), prepared_raw.normalization_trace);

        let ids = TalkerCheckpoint::utterance_text_ids(&wrapped);
        let table = self
            .talker
            .gather_text_rows(&ids)
            .map_err(|error| js_error("text embedding gather failed", error))?;
        let header = self
            .talker
            .xvector_header(&table, speaker, CODEC_LANGUAGE_ENGLISH_ID)
            .map_err(|error| js_error("prompt header derivation failed", error))?;
        let tts_eos = self.talker.tts_eos(&table);

        let talker_layers = self.talker.talker_layer_weights();
        let micro_layers = self.talker.microdecoder_layer_weights();
        let residual = self.talker.residual_embedding_slices();
        let heads = self.talker.microdecoder_head_slices();
        let micro_residual = &residual[..residual.len() - 1];

        let mut generator = QwenGenerator::new_with_artifact(
            QwenGeneratorConfig {
                talker_config: TalkerConfig::default(),
                talker_weights: self.talker.talker_weights(&talker_layers),
                text: self.talker.text_weights(&table),
                feedback: self.talker.feedback_tables(&residual),
                microdecoder_config: MicrodecoderConfig::default(),
                microdecoder_weights: self.talker.microdecoder_weights(
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
                sampling_mode: SamplingMode::Production,
                seed,
            },
            Some(&self.artifact),
        );

        generator
            .begin_utterance(&prepared)
            .map_err(|error| js_error("prefill failed", error))?;

        // The CLI's text-proportional EOS backstop, so a runaway prompt cannot spin forever.
        let cap = if max_frames == 0 {
            wrapped.len() * 4 + 64
        } else {
            max_frames as usize
        };

        const PACKET_FRAMES: usize = 4;
        let mut state = self.codec.stream_state();
        let mut pcm: Vec<f32> = Vec::new();
        let mut packet_pcm: Vec<f32> = Vec::new();
        let mut packet: Vec<i32> = Vec::with_capacity(16 * PACKET_FRAMES);
        let mut packet_frames = 0usize;
        let mut flush = |state: &mut _,
                         packet: &mut Vec<i32>,
                         frames: &mut usize,
                         pcm: &mut Vec<f32>|
         -> Result<(), JsValue> {
            if *frames == 0 {
                return Ok(());
            }
            self.codec
                .stream_push(state, packet, *frames, &mut packet_pcm)
                .map_err(|error| js_error("codec decode failed", error))?;
            pcm.extend_from_slice(&packet_pcm);
            packet.clear();
            *frames = 0;
            Ok(())
        };

        for _ in 0..cap {
            match generator
                .next_frame()
                .map_err(|error| js_error("generation failed", error))?
            {
                Some(frame) => {
                    for code in &frame.codes {
                        packet.push(i32::try_from(*code).map_err(|error| {
                            js_error("generated code does not fit the codec's i32", error)
                        })?);
                    }
                    packet_frames += 1;
                    if packet_frames == PACKET_FRAMES {
                        flush(&mut state, &mut packet, &mut packet_frames, &mut pcm)?;
                    }
                }
                None => break,
            }
        }
        flush(&mut state, &mut packet, &mut packet_frames, &mut pcm)?;
        Ok(pcm)
    }

    /// Enroll a voice from mono 24 kHz PCM in `[-1, 1]`; returns the 1,024-float x-vector.
    ///
    /// The speaker encoder hydrates lazily from the artifact on first use and is cached.
    ///
    /// # Errors
    ///
    /// Throws when the PCM is too short for the mel front end or hydration fails.
    pub fn enroll(&mut self, pcm: &[f32]) -> Result<Vec<f32>, JsValue> {
        if self.speaker_encoder.is_none() {
            let encoder = SpeakerEncoder::load_fttsq_mapped(
                &self.artifact,
                Path::new("browser://model.fttsq"),
            )
            .map_err(|error| js_error("speaker encoder hydration failed", error))?;
            self.speaker_encoder = Some(encoder);
        }
        let mel = log_mel_from_24khz_pcm(pcm)
            .map_err(|error| js_error("cannot extract speaker features", error))?;
        let encoder = self.speaker_encoder.as_ref().expect("cached just above");
        let vector = encoder.encode(&mel.values, mel.frames);
        if vector.iter().all(|value| value.is_finite()) {
            Ok(vector)
        } else {
            Err(js_error(
                "enrollment refused",
                "speaker encoder produced a non-finite x-vector",
            ))
        }
    }
}

/// Times the per-frame kernel schedule at the model's real shapes; returns a JSON report.
///
/// This is the Spike-B RTF proxy: one talker step (28 layers x fused qkv/o/gate_up/down at
/// m=1) plus fifteen sequential microdecoder steps (5 layers each), all through the same
/// `linear_q8` the armed route dispatches. Codec cost is excluded (its dense fall-through is
/// BLAS-on-macOS, scalar here) — the report says so rather than pretending.
#[wasm_bindgen]
#[must_use]
#[allow(clippy::missing_panics_doc)]
pub fn bench_frame_kernels(rounds: u32) -> String {
    use ftts_kernels::int8::{Int8Tier, QuantizedMatrix, linear_q8};

    // (n, k, calls_per_frame): talker fused shapes x28, micro fused shapes x(15*5).
    let schedule: &[(usize, usize, usize)] = &[
        (4096, 1024, 28), // talker qkv (2048 q + 2x1024 kv)
        (1024, 2048, 28), // talker o_proj
        (6144, 1024, 28), // talker gate||up
        (1024, 3072, 28), // talker down
        (4096, 1024, 75), // micro qkv x 15 depths x 5 layers
        (1024, 2048, 75), // micro o
        (6144, 1024, 75), // micro gate||up
        (1024, 3072, 75), // micro down
    ];
    let mut matrices: Vec<(QuantizedMatrix, usize)> = Vec::new();
    for &(n, k, calls) in schedule {
        let data: Vec<i8> = (0..n * k)
            .map(|index| (((index * 37 + 11) % 255) as i32 - 127) as i8)
            .collect();
        let scales = vec![0.01f32; n];
        matrices.push((QuantizedMatrix { data, scales, n, k }, calls));
    }
    let x: Vec<i8> = (0..8192)
        .map(|i| (((i * 29 + 5) % 255) as i32 - 127) as i8)
        .collect();
    let mut out = vec![0.0f32; 8192];

    let now = || js_sys::Date::now();
    let mut per_frame_ms = f64::MAX;
    for _ in 0..rounds.max(1) {
        let started = now();
        for (matrix, calls) in &matrices {
            for _ in 0..*calls {
                linear_q8(
                    &x[..matrix.k],
                    &[0.02],
                    matrix,
                    None,
                    1,
                    &mut out[..matrix.n],
                    Int8Tier::Scalar,
                );
            }
        }
        per_frame_ms = per_frame_ms.min(now() - started);
    }
    let frame_budget_ms = 80.0;
    serde_json::json!({
        "per_frame_talker_micro_ms": per_frame_ms,
        "frame_budget_ms": frame_budget_ms,
        "rtf_estimate_talker_micro_only": frame_budget_ms / per_frame_ms,
        "excluded": "codec + attention + norms + sampling; codec dense runs scalar on wasm",
        "tier": "scalar (LLVM autovec; build with +simd128 for WASM SIMD)",
        "rounds": rounds,
    })
    .to_string()
}
