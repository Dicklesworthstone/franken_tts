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
use ftts_model_qwen::tokenizer::QwenTokenizer;
#[cfg(not(unix))]
use ftts_model_qwen::tokenizer::TokenizerFiles;
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

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console, js_name = error)]
    fn console_error(message: &str);
}

/// Routes Rust panics to `console.error` with the real message and location — without this a
/// release-wasm panic surfaces as an opaque `RuntimeError: unreachable`.
#[wasm_bindgen(start)]
pub fn install_panic_hook() {
    std::panic::set_hook(Box::new(|info| {
        console_error(&format!("ftts-wasm panic: {info}"));
    }));
}

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

/// Which int8 route this build actually dispatches in the browser.
///
/// Exposed because the browser has no environment variables and no `robot backends`: without a
/// way to ask, a wasm build that silently fell back to the scalar loop would look exactly like
/// one running the SIMD128 island, and that difference is most of the frame time.
#[wasm_bindgen]
#[must_use]
pub fn int8_route() -> String {
    ftts_kernels::int8::Int8Tier::dispatch().as_str().to_owned()
}

/// Times one int8 GEMV at a real model reduction length and returns nanoseconds per dot.
///
/// A kernel benchmark rather than an end-to-end one on purpose: it isolates the thing the SIMD
/// island changes, needs no 2 GB model, and runs in a second. `tier` takes the route names from
/// [`ftts_kernels::int8::Int8Tier::as_str`] so a caller can A/B `scalar` against `wasm-simd128`
/// in the same process — same allocator, same warm caches, same engine — which is the only way
/// the ratio means anything.
///
/// Timing is the caller's job: wasm has no clock, so this returns after `rounds` passes and the
/// JS side divides by its own `performance.now()` delta.
///
/// # Errors
///
/// Throws when `tier` is not a route this build can execute.
#[wasm_bindgen]
pub fn bench_int8_gemv(tier: &str, k: usize, n: usize, rounds: usize) -> Result<f32, JsValue> {
    use ftts_kernels::int8::{Int8Tier, QuantizedMatrix, linear_q8, quantize_row_q8};

    let route = Int8Tier::available()
        .into_iter()
        .find(|candidate| candidate.as_str() == tier)
        .ok_or_else(|| js_error("unavailable int8 route", tier))?;

    // Deterministic operands spanning the full [-127, 127] range, so the measurement cannot be
    // flattered by a sparse or small-magnitude matrix.
    let mut state = 0x2545_F491_4F6C_DD1D_u64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        ((state >> 33) as f32 / f32::from(u16::MAX)) - 0.5
    };
    let weight: Vec<f32> = (0..n * k).map(|_| next()).collect();
    let matrix = QuantizedMatrix::quantize(&weight, n, k);
    let activation: Vec<f32> = (0..k).map(|_| next()).collect();
    let mut x_q = vec![0_i8; k];
    let scale = quantize_row_q8(&activation, &mut x_q);
    let mut out = vec![0.0_f32; n];

    for _ in 0..rounds {
        linear_q8(&x_q, &[scale], &matrix, None, 1, &mut out, route);
    }
    // Returned so the optimizer cannot delete the loop it was asked to time.
    Ok(out.iter().sum())
}

/// Publishes the control block Workers park on, before the team has a size.
///
/// Call from the engine Worker — never the page's main thread, where `atomic.wait` traps. Each
/// Worker then instantiates this same module against the same `WebAssembly.Memory` and calls
/// [`worker_loop_entry`] with its index. Once they report parked, call [`arm_worker_team`] with
/// the count that actually started.
#[cfg(not(unix))]
#[wasm_bindgen]
pub fn publish_team_block() {
    ftts_kernels::team::publish_wasm_block();
}

/// Sizes the team once the Workers that will serve it have confirmed they are parked.
///
/// `partitions` counts the dispatcher too, so pass `readyWorkers + 1`. Anything <= 1 arms
/// nothing and the engine runs serially — the correct outcome on a browser without
/// `SharedArrayBuffer`, and on one where every Worker failed to start.
#[cfg(not(unix))]
#[wasm_bindgen]
pub fn arm_worker_team(partitions: usize) {
    ftts_kernels::team::arm_wasm_team(partitions);
}

/// The body a spawned Worker runs; never returns.
///
/// # Errors
///
/// Throws if called before [`arm_worker_team`] published the control block, which is a host
/// sequencing bug — parking on a block that does not exist would hang silently instead.
#[cfg(not(unix))]
#[wasm_bindgen]
pub fn worker_loop_entry(worker: usize) -> Result<(), JsValue> {
    if worker == 0 {
        return Err(js_error("worker index must be >= 1", "0 is the dispatcher"));
    }
    ftts_kernels::team::wasm_worker_loop(worker);
    Ok(())
}

/// How many partitions the int8 team is running with; 1 means serial.
#[cfg(not(unix))]
#[must_use]
#[wasm_bindgen]
pub fn worker_team_width() -> usize {
    ftts_kernels::team::partitions()
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
    #[cfg(not(unix))]
    denoiser: Option<ftts_kernels::enhance::Enhancer>,
}

/// The FastEnhancer-S denoiser weights, embedded like the preset voices: 0.8 MB of
/// inference-form safetensors (see `docs/DENOISER.md` for the pin). Embedding them keeps
/// browser enrollment cleanup working with no extra fetch, no CORS surface, and no
/// divergence between what the CLI and the tab run.
#[cfg(not(unix))]
const DENOISER_SAFETENSORS: &[u8] =
    include_bytes!("../assets/fastenhancer-s-48k-denoise.safetensors");

/// Model bytes accumulated directly inside wasm linear memory, a slice at a time.
///
/// # Why this exists
///
/// Passing the artifact as a `Vec<u8>` makes wasm-bindgen copy it out of the JS `ArrayBuffer`
/// into linear memory, so a 1.3 GB model is **live twice** at the moment of construction. Desktop
/// Chrome absorbs 2.6 GB; an iPhone does not — the tab is reclaimed and the page "crashes while
/// loading". Streaming OPFS slices straight into a buffer that already lives in wasm keeps the JS
/// heap at one slice and the artifact at exactly one copy.
///
/// Capacity is reserved once, exactly, up front. That is the load-bearing detail: a `Vec` that
/// grows by doubling would transiently hold 1.3 GB *plus* its 2.6 GB successor while copying —
/// worse than the problem being solved. `try_reserve_exact` also turns an allocation failure into
/// a thrown error naming the number of bytes, rather than the opaque `unreachable` a wasm abort
/// shows the user.
#[wasm_bindgen]
pub struct ModelStaging {
    fttsq: Vec<u8>,
    codec: Vec<u8>,
}

#[wasm_bindgen]
impl ModelStaging {
    /// Reserve exact room for both files before any bytes arrive.
    ///
    /// # Errors
    ///
    /// Throws when linear memory cannot be reserved, naming the byte count that failed — the
    /// honest signal on a device that simply does not have the memory.
    #[wasm_bindgen(constructor)]
    pub fn new(fttsq_bytes: usize, codec_bytes: usize) -> Result<ModelStaging, JsValue> {
        let mut fttsq = Vec::new();
        fttsq
            .try_reserve_exact(fttsq_bytes)
            .map_err(|_| js_error("cannot reserve model memory (bytes)", fttsq_bytes))?;
        let mut codec = Vec::new();
        codec
            .try_reserve_exact(codec_bytes)
            .map_err(|_| js_error("cannot reserve codec memory (bytes)", codec_bytes))?;
        Ok(ModelStaging { fttsq, codec })
    }

    /// Append one slice of the artifact, in order.
    ///
    /// # Errors
    ///
    /// Throws if the slice would exceed the reserved capacity, which means the caller's manifest
    /// and its download disagree — better caught here than as a corrupt tensor later.
    pub fn push_fttsq(&mut self, chunk: &[u8]) -> Result<(), JsValue> {
        append_exact(&mut self.fttsq, chunk, "artifact")
    }

    /// Append one slice of the codec checkpoint, in order.
    ///
    /// # Errors
    ///
    /// As [`ModelStaging::push_fttsq`].
    pub fn push_codec(&mut self, chunk: &[u8]) -> Result<(), JsValue> {
        append_exact(&mut self.codec, chunk, "codec")
    }

    /// Bytes accepted so far, so the caller can drive a progress bar without tracking it twice.
    #[must_use]
    pub fn filled(&self) -> usize {
        self.fttsq.len() + self.codec.len()
    }
}

/// Appends without ever letting the vector reallocate.
fn append_exact(target: &mut Vec<u8>, chunk: &[u8], what: &str) -> Result<(), JsValue> {
    if target.len() + chunk.len() > target.capacity() {
        return Err(js_error(
            "more bytes than reserved for",
            format!(
                "{what}: {} + {} exceeds {}",
                target.len(),
                chunk.len(),
                target.capacity()
            ),
        ));
    }
    target.extend_from_slice(chunk);
    Ok(())
}

#[wasm_bindgen]
impl WasmEngine {
    /// Hydrate from bytes already resident in wasm memory, consuming the staging buffer.
    ///
    /// The streaming counterpart of [`WasmEngine::new`]: identical hydration, but the artifact is
    /// moved rather than copied, so the peak is one copy instead of two.
    ///
    /// # Errors
    ///
    /// Throws when staging is incomplete, or with the failing hydration stage named.
    #[cfg(not(unix))]
    #[allow(clippy::needless_pass_by_value)]
    pub fn from_staging(
        staging: ModelStaging,
        vocab_json: String,
        merges_txt: String,
        tokenizer_config_json: String,
    ) -> Result<WasmEngine, JsValue> {
        if staging.fttsq.len() != staging.fttsq.capacity()
            || staging.codec.len() != staging.codec.capacity()
        {
            return Err(js_error(
                "staging is incomplete",
                format!(
                    "artifact {}/{}, codec {}/{}",
                    staging.fttsq.len(),
                    staging.fttsq.capacity(),
                    staging.codec.len(),
                    staging.codec.capacity()
                ),
            ));
        }
        let ModelStaging { fttsq, codec } = staging;
        Self::hydrate(
            fttsq,
            codec,
            &vocab_json,
            &merges_txt,
            &tokenizer_config_json,
        )
    }

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
        Self::hydrate(
            fttsq,
            codec,
            &vocab_json,
            &merges_txt,
            &tokenizer_config_json,
        )
    }

    /// The hydration both constructors share, taking ownership of bytes already in wasm memory.
    #[cfg(not(unix))]
    fn hydrate(
        fttsq: Vec<u8>,
        codec: Vec<u8>,
        vocab_json: &str,
        merges_txt: &str,
        tokenizer_config_json: &str,
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
        // Free the codec's source bytes the instant they stop being needed.
        //
        // `CodecCheckpoint` materializes every tensor into owned `Vec<f32>`, so the moment it is
        // built the ~0.68 GB of BF16 behind `codec_file` is dead weight — yet Rust would hold it
        // until this function returns, which is exactly when the high-water mark occurs: artifact
        // (1.31 GB) + codec source (0.68) + widened codec f32 (~1.36) ≈ 3.35 GB. That is more than
        // iOS Safari gives a tab, and it is why hydration dies there while the download now
        // survives. Dropping here removes 0.68 GB from the peak at no cost to anything.
        drop(codec_file);

        let tokenizer = QwenTokenizer::from_files_using_environment(TokenizerFiles {
            vocab_json,
            merges_txt,
            tokenizer_config_json,
        })
        .map_err(|error| js_error("tokenizer unusable", error))?;

        Ok(WasmEngine {
            talker,
            codec,
            tokenizer,
            artifact,
            speaker_encoder: None,
            #[cfg(not(unix))]
            denoiser: None,
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

        let clock = js_sys::Date::now;
        let prefill_started = clock();
        generator
            .begin_utterance(&prepared)
            .map_err(|error| js_error("prefill failed", error))?;
        let prefill_ms = clock() - prefill_started;
        let mut frames_ms = 0.0f64;
        let mut codec_ms = 0.0f64;

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
            let codec_started = js_sys::Date::now();
            self.codec
                .stream_push(state, packet, *frames, &mut packet_pcm)
                .map_err(|error| js_error("codec decode failed", error))?;
            codec_ms += js_sys::Date::now() - codec_started;
            pcm.extend_from_slice(&packet_pcm);
            packet.clear();
            *frames = 0;
            Ok(())
        };

        for _ in 0..cap {
            let frame_started = clock();
            let step = generator
                .next_frame()
                .map_err(|error| js_error("generation failed", error))?;
            frames_ms += clock() - frame_started;
            match step {
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
        // Stage split to the console: the number that says WHERE wasm time goes.
        console_error(&format!(
            "ftts-wasm timing: prefill {prefill_ms:.0}ms, talker+micro {frames_ms:.0}ms, codec {codec_ms:.0}ms, {} samples",
            pcm.len()
        ));
        Ok(pcm)
    }

    /// Enroll a voice from mono 24 kHz PCM in `[-1, 1]`; returns the 1,024-float x-vector.
    ///
    /// The reference is denoised first with the embedded FastEnhancer-S port — the same
    /// automatic cleanup the CLI applies — so a laptop-mic recording enrolls without its
    /// room hiss. Use [`WasmEngine::enroll_raw`] to skip cleanup.
    ///
    /// The speaker encoder hydrates lazily from the artifact on first use and is cached,
    /// as is the denoiser.
    ///
    /// # Errors
    ///
    /// Throws when the PCM is too short for the mel front end or hydration fails.
    pub fn enroll(&mut self, pcm: &[f32]) -> Result<Vec<f32>, JsValue> {
        // The in-memory safetensors backing only exists off-unix (`SafetensorsFile::from_bytes`),
        // like every other browser-only path in this crate. The unix compilation exists to keep
        // the workspace gate honest, not to serve users — there `enroll` is the raw path.
        #[cfg(not(unix))]
        {
            if self.denoiser.is_none() {
                let file = ftts_artifacts::safetensors::SafetensorsFile::from_bytes(
                    DENOISER_SAFETENSORS.to_vec(),
                )
                .map_err(|error| {
                    js_error(
                        "embedded denoiser weights are malformed",
                        format!("{error:?}"),
                    )
                })?;
                let enhancer = ftts_artifacts::enhance_loader::enhancer_from_safetensors(&file)
                    .map_err(|error| js_error("embedded denoiser weights are malformed", error))?;
                self.denoiser = Some(enhancer);
            }
            let cleaned = self
                .denoiser
                .as_ref()
                .expect("cached just above")
                .enhance_24k(pcm);
            self.enroll_raw(&cleaned)
        }
        #[cfg(unix)]
        self.enroll_raw(pcm)
    }

    /// Enroll exactly the PCM given, with no noise cleanup.
    ///
    /// # Errors
    ///
    /// Throws when the PCM is too short for the mel front end or hydration fails.
    pub fn enroll_raw(&mut self, pcm: &[f32]) -> Result<Vec<f32>, JsValue> {
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
        .map(|i| (((i * 29 + 5) % 255) - 127) as i8)
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
