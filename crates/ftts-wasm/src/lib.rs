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

use ftts_model_qwen::checkpoint::CodecCheckpoint;
use wasm_bindgen::prelude::*;

// The hydration-from-bytes and synthesis surface exists only where there is no filesystem
// (everything `#[cfg(not(unix))]` below); these imports are gated with it so the workspace's
// native `--all-targets` pass stays warning-free.
#[cfg(not(unix))]
use ftts_core::{
    FrameGenerator, FrameStep, NormalizationOptions, PreparedText, TextPreparer, UtteranceStart,
};
#[cfg(not(unix))]
use ftts_model_qwen::checkpoint::{
    CODEC_LANGUAGE_ENGLISH_ID, TALKER_HIDDEN, TalkerCheckpoint, TextEmbeddingTable,
};
#[cfg(not(unix))]
use ftts_model_qwen::generate::{QwenGenerator, QwenGeneratorConfig};
#[cfg(not(unix))]
use ftts_model_qwen::microdecoder::MicrodecoderConfig;
#[cfg(not(unix))]
use ftts_model_qwen::prompt::{CloneMode, PromptMode};
#[cfg(not(unix))]
use ftts_model_qwen::sampler::SamplingMode;
#[cfg(not(unix))]
use ftts_model_qwen::speaker::{Encoder as SpeakerEncoder, log_mel_from_24khz_pcm};
#[cfg(not(unix))]
use ftts_model_qwen::talker::TalkerConfig;
#[cfg(not(unix))]
use ftts_model_qwen::tokenizer::{QwenTokenizer, TokenizerFiles};
#[cfg(not(unix))]
use std::path::Path;
#[cfg(not(unix))]
use std::sync::Arc;

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

/// Wasm linear memory in bytes, read from the engine itself.
///
/// The JS side can only sample this between calls, so a spike that happens INSIDE one synchronous
/// synthesize call is invisible there — and synthesis was measured to add 0.56 GB somewhere in
/// that call. This reads the page count directly, which is exact and costs one instruction.
#[cfg(target_arch = "wasm32")]
fn linear_memory_bytes() -> f64 {
    (core::arch::wasm32::memory_size(0) as f64) * 65_536.0
}
// Exists only where its (not-unix) callers do; without the unix bound it is dead code on the
// native hosts the workspace `--all-targets` check compiles.
#[cfg(all(not(target_arch = "wasm32"), not(unix)))]
fn linear_memory_bytes() -> f64 {
    0.0
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
        // Top 16 bits → [0, 1) → [-0.5, 0.5). The earlier `>> 33` kept 31 bits, producing
        // values up to ~32767.5 — all positive and huge, the opposite of the full-sign
        // small-magnitude spread this comment promises (int8.rs:1090 has the honest form).
        ((state >> 48) as f32 / f32::from(u16::MAX)) - 0.5
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
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn publish_team_block() {
    ftts_kernels::team::publish_wasm_block();
}

/// Sizes the team once the Workers that will serve it have confirmed they are parked.
///
/// `partitions` counts the dispatcher too, so pass `readyWorkers + 1`. Anything <= 1 arms
/// nothing and the engine runs serially — the correct outcome on a browser without
/// `SharedArrayBuffer`, and on one where every Worker failed to start.
#[cfg(target_arch = "wasm32")]
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
#[cfg(target_arch = "wasm32")]
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
///
/// Gated off unix with the rest of the byte-oriented surface: native hosts run the CLI's
/// engine, and the workspace `--all-targets` check compiles this crate natively.
#[cfg(not(unix))]
#[wasm_bindgen]
pub struct WasmEngine {
    talker: TalkerCheckpoint,
    codec: CodecCheckpoint,
    tokenizer: QwenTokenizer,
    /// The fused W8A8 tables, built ONCE at hydration and lent to every utterance.
    ///
    /// This field is what replaced the artifact handle. Fusing QKV and gate‖up is a copy by
    /// definition, ~0.46 GB of Q8 duplicating bytes already in the staged artifact — and it used
    /// to happen on every synthesize call, landing on top of an already-peaked heap that wasm can
    /// never shrink. Built here instead, the artifact becomes dead the moment it exists, so it is
    /// released and the 0.69 GB hot prefix goes back to the allocator for the codec and the
    /// talker's widened tensors to reuse.
    int8: Option<Arc<ftts_model_qwen::generate::Int8Route>>,
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
/// # Why the two files are staged in sequence rather than together
///
/// The order is the whole optimization. Staging both raw files first and hydrating afterwards
/// means that at the moment the codec finishes widening, THREE allocations are live at once:
///
/// ```text
/// artifact raw   1.31 GB   (staged, still untouched)
/// codec raw      0.68 GB   (source, not yet droppable)
/// codec f32     ~1.36 GB   (CodecCheckpoint owns Vec<f32> for every tensor)
///                =======
/// peak          ~3.35 GB
/// ```
///
/// Dropping the codec source after the fact does not help: the peak has already happened. iOS
/// Safari reclaims the tab somewhere below that, which is why the page died the instant it said
/// "hydrating" even though the download had just succeeded.
///
/// Hydrating the codec BEFORE the artifact is staged removes the largest term from the peak
/// instead of from the aftermath:
///
/// ```text
/// phase 1   codec raw 0.68 + codec f32 1.36  = 2.04 GB, then the source drops -> 1.36 GB
/// phase 2   codec f32 1.36 + artifact 1.31   = 2.67 GB
/// ```
///
/// 2.67 GB against 3.35 GB, and nothing is ever read twice. The caller therefore drives:
/// `new(codec_bytes)` -> `push_codec` xN -> `finish_codec()` -> `reserve_fttsq(bytes)` ->
/// `push_fttsq` xN -> `WasmEngine::from_staging`. Each step refuses to run out of order.
#[wasm_bindgen]
pub struct ModelStaging {
    /// Codec tensors, widened as they arrive and moved into the checkpoint by `finish_codec`.
    ///
    /// Never a staged file. Holding the codec's 0.46 GB of source bytes to build a 0.457 GB
    /// checkpoint out of them meant paying for it twice, and wasm cannot hand the first copy
    /// back — linear memory only grows. Each tensor is widened on arrival and its incoming bytes
    /// released immediately, so the high-water mark is the finished set plus one tensor.
    codec_tensors: ftts_model_qwen::checkpoint::WidenedTensors,
    /// Bytes accepted into `codec_tensors`, for progress only.
    codec_filled: usize,
    /// Present only after `finish_codec`. Its presence IS the phase marker.
    codec: Option<CodecCheckpoint>,
    /// Reserved by `reserve_fttsq`, which refuses to run before the codec is hydrated.
    fttsq: Vec<u8>,
    /// The talker side, complete and artifact-free, once `finish_artifact` has run.
    ///
    /// Gated like the type it holds: hydration-from-bytes exists only where there is no
    /// filesystem, and the workspace `--all-targets` check compiles this crate natively.
    #[cfg(not(unix))]
    hydrated: Option<HydratedArtifact>,
    /// Set when the caller reserved only the artifact's hot prefix, holding the 622 MB cold text
    /// embedding out of linear memory. Checked against the artifact's own directory at hydration,
    /// so a caller that truncates at the wrong offset is refused rather than trusted. Carries the
    /// artifact's FULL declared length, because the directory's ranges must be validated against
    /// the real artifact rather than the bytes in hand — the elided section runs past the prefix
    /// by construction, so bounding by what is present rejects every prefix outright.
    cold_elided: Option<u64>,
    /// Bytes freed by `finish_codec`, so `filled` can keep reporting monotonic download progress
    /// after the source it was counting has gone away.
    codec_retired: usize,
}

#[wasm_bindgen]
impl ModelStaging {
    /// Reserve exact room for the codec checkpoint, the first file staged.
    ///
    /// The artifact is deliberately NOT reserved here — see the type-level note. Reserving it now
    /// would put its 1.31 GB back into the codec-hydration peak and undo the whole point.
    ///
    /// # Errors
    ///
    /// Throws when linear memory cannot be reserved, naming the byte count that failed — the
    /// honest signal on a device that simply does not have the memory.
    #[wasm_bindgen(constructor)]
    #[allow(clippy::new_without_default)] // a JS-facing constructor; Default has no JS caller
    pub fn new() -> ModelStaging {
        // Reserves NOTHING. Both buffers are claimed explicitly, by the caller, in the order the
        // caller wants them laid out — which is the whole lever now that wasm memory can only
        // grow. Reserving the codec here would pin it below the artifact permanently, and the
        // codec's buffer is the one that becomes a hole.
        ModelStaging {
            codec_tensors: ftts_model_qwen::checkpoint::WidenedTensors::new(),
            codec_filled: 0,
            codec: None,
            fttsq: Vec::new(),
            #[cfg(not(unix))]
            hydrated: None,
            cold_elided: None,
            codec_retired: 0,
        }
    }

    /// Hydrate the talker from the staged artifact, then RELEASE the artifact.
    ///
    /// The ordering step, and the reason this is a separate call rather than part of the final
    /// assembly. Building the fused int8 tables is what makes the artifact's 0.69 GB hot prefix
    /// dead — and wasm memory only ever grows, so the hole that release leaves is worth exactly
    /// whatever gets allocated after it. Running this BEFORE the codec is staged lets the codec's
    /// 0.46 GB of source and 0.457 GB of checkpoint land INSIDE that hole rather than on top of
    /// it. With the codec first, the same work peaked at 2.10 GB.
    ///
    /// # Errors
    ///
    /// Throws when staging is incomplete, the prefix does not match the artifact's own declared
    /// layout, or hydration fails — each named.
    ///
    /// Gated off unix like `finish_codec` below: [`HydratedArtifact`] is the byte-oriented
    /// loader's product and exists only where there is no filesystem.
    #[cfg(not(unix))]
    pub fn finish_artifact(&mut self) -> Result<(), JsValue> {
        if self.hydrated.is_some() {
            return Err(js_error(
                "artifact already hydrated",
                "finish_artifact twice",
            ));
        }
        if self.fttsq.len() != self.fttsq.capacity() {
            return Err(js_error(
                "artifact staging is incomplete",
                format!("{}/{}", self.fttsq.len(), self.fttsq.capacity()),
            ));
        }
        // `take` moves the bytes out, so the staging buffer is not a second owner keeping the
        // payload alive past the release this whole step exists to perform.
        let fttsq = std::mem::take(&mut self.fttsq);
        self.hydrated = Some(HydratedArtifact::build(fttsq, self.cold_elided)?);
        Ok(())
    }

    /// Hand over one codec tensor's little-endian f32 bytes, widened on arrival.
    ///
    /// There is no staged file and no reservation. The codec used to arrive as a 0.46 GB
    /// safetensors buffer that a 0.457 GB checkpoint was then built out of — paying for the same
    /// weights twice, in a heap that can never shrink. Tensor by tensor, the caller's chunk is
    /// widened and released immediately, so the peak is the finished set plus one tensor.
    ///
    /// # Errors
    ///
    /// Throws once the codec is hydrated, or when the byte count is not a whole number of f32 —
    /// a truncated push, which must not be rounded down into a silently short tensor.
    pub fn push_codec_tensor(&mut self, name: &str, bytes: &[u8]) -> Result<(), JsValue> {
        if self.codec.is_some() {
            return Err(js_error(
                "codec already hydrated",
                "push_codec_tensor after finish",
            ));
        }
        self.codec_tensors
            .insert_f32_le(name, bytes)
            .map_err(|error| js_error("codec tensor rejected", error))?;
        self.codec_filled += bytes.len();
        Ok(())
    }

    /// Parse and widen the codec, then free its source bytes.
    ///
    /// This is the phase boundary. After it returns, the 0.68 GB of BF16 is gone and only the
    /// widened checkpoint remains, so the artifact can be staged against a much lower floor.
    ///
    /// # Errors
    ///
    /// Throws when the staged bytes are short of what was reserved, or when the checkpoint does
    /// not parse.
    ///
    /// Gated off unix for the same reason as the constructors below: `SafetensorsFile::from_bytes`
    /// is the byte-oriented loader that exists only where there is no filesystem. Native hosts go
    /// through the CLI's path-based loader instead, so this arm simply does not exist there — and
    /// the workspace's `--all-targets` check compiles this crate natively, which is what caught it.
    #[cfg(not(unix))]
    pub fn finish_codec(&mut self) -> Result<(), JsValue> {
        if self.codec.is_some() {
            return Err(js_error("codec already hydrated", "finish_codec twice"));
        }
        if self.codec_tensors.is_empty() {
            return Err(js_error(
                "no codec tensors were staged",
                "push_codec_tensor before finish_codec",
            ));
        }
        self.codec_retired = self.codec_filled;
        // Each tensor is MOVED out of the store as the checkpoint claims it, so the two never
        // hold the same weights at once. A name the checkpoint asks for and the caller never
        // pushed surfaces here as a missing-tensor error naming it, which is the honest failure
        // for an incomplete stream — never a zero-filled stand-in.
        let codec = CodecCheckpoint::load_from_store(
            &self.codec_tensors,
            Path::new("browser://codec.safetensors"),
        )
        .map_err(|error| js_error("codec hydration failed", error))?;
        self.codec = Some(codec);
        Ok(())
    }

    /// Reserve exact room for the artifact.
    ///
    /// # Why this no longer insists the codec goes first
    ///
    /// It used to, because the codec's source was the whole 0.68 GB file and holding it beside the
    /// artifact restored a ~3.35 GB peak. Both halves of that changed. The caller now stages only
    /// the codec's DECODER (~0.46 GB, since every tensor `CodecCheckpoint` reads is `decoder.*`),
    /// and the artifact is now a hot prefix rather than the whole file — so holding both is
    /// ~1.15 GB, below the peak the old ordering reached anyway.
    ///
    /// Ordering now matters for a different reason, and it points the other way. wasm memory can
    /// only grow, so a freed buffer is not returned; it is a hole, and a hole is only worth having
    /// where something later can land in it. The codec's source is freed the moment the checkpoint
    /// is built, and the checkpoint is ~0.04 GB — so that 0.46 GB hole wants to be the LAST large
    /// allocation, where the talker's 0.34 GB of widened tensors can reuse it. Reserving the
    /// artifact first puts it there. Codec-first instead committed the hole before the artifact
    /// ever grew, and the artifact could not fit in it, so the page paid for both.
    ///
    /// Measured: 1.64 GB committed codec-first, and the artifact streamed while already carrying
    /// ~1.19 GB — which is the phase an iPhone died in.
    ///
    /// # Errors
    ///
    /// When the reservation fails.
    pub fn reserve_fttsq(&mut self, fttsq_bytes: usize) -> Result<(), JsValue> {
        self.fttsq
            .try_reserve_exact(fttsq_bytes)
            .map_err(|_| js_error("cannot reserve model memory (bytes)", fttsq_bytes))
    }

    /// Reserve room for the artifact's HOT PREFIX only, leaving the cold text embedding on disk.
    ///
    /// The cold section is 622 MB — 47% of the artifact — and an utterance touches a few hundred
    /// of its 4 KB rows. A native host pays for only those rows because it maps the file; wasm has
    /// no mapping, and its heap is grow-only, so every staged byte is resident for the life of the
    /// page. On a device with a ~2 GB per-tab ceiling that ballast is the difference between
    /// running and being killed.
    ///
    /// `hot_bytes` must be exactly the artifact's cold-section offset. It is not taken on trust:
    /// hydration re-derives it from the staged directory and refuses a mismatch, so a caller that
    /// truncates at the wrong place gets a named error instead of silently wrong tensors.
    ///
    /// # Errors
    ///
    /// As [`ModelStaging::reserve_fttsq`].
    pub fn reserve_fttsq_hot_prefix(
        &mut self,
        hot_bytes: usize,
        full_bytes: u64,
    ) -> Result<(), JsValue> {
        self.cold_elided = Some(full_bytes);
        self.reserve_fttsq(hot_bytes)
    }

    /// Append one slice of the artifact, in order.
    ///
    /// # Errors
    ///
    /// Throws if the slice would exceed the reserved capacity, or if no reservation was made.
    pub fn push_fttsq(&mut self, chunk: &[u8]) -> Result<(), JsValue> {
        append_exact(&mut self.fttsq, chunk, "artifact")
    }

    /// Bytes accepted so far, so the caller can drive a progress bar without tracking it twice.
    ///
    /// Counts the retired codec source rather than the live buffer, so the number keeps rising
    /// across the phase boundary instead of collapsing when the source is freed.
    #[must_use]
    pub fn filled(&self) -> usize {
        self.codec_retired + self.codec_filled + self.fttsq.len()
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

#[cfg(not(unix))]
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
        let ModelStaging {
            codec, hydrated, ..
        } = staging;
        let mut hydrated = hydrated.ok_or_else(|| {
            js_error(
                "artifact was never hydrated",
                "call finish_artifact() before from_staging",
            )
        })?;
        let codec = codec.ok_or_else(|| {
            js_error(
                "codec was never hydrated",
                "call finish_codec() before from_staging",
            )
        })?;
        let tokenizer = QwenTokenizer::from_files_using_environment(TokenizerFiles {
            vocab_json: &vocab_json,
            merges_txt: &merges_txt,
            tokenizer_config_json: &tokenizer_config_json,
        })
        .map_err(|error| js_error("tokenizer unusable", error))?;
        Ok(WasmEngine {
            talker: hydrated.talker.take().expect("built by finish_artifact"),
            codec,
            tokenizer,
            int8: hydrated.int8.take(),
            speaker_encoder: hydrated.speaker_encoder.take(),
            #[cfg(not(unix))]
            denoiser: None,
        })
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

    /// The whole-buffer constructor's hydration: parse the codec, then delegate.
    ///
    /// This path necessarily holds both raw files at once — that is inherent to being handed two
    /// complete buffers — so it carries the ~3.35 GB peak described on [`ModelStaging`]. It is
    /// kept for callers with memory to spare (desktop Chrome, tests); the streaming path is what
    /// the site uses and what a phone can survive.
    #[cfg(not(unix))]
    fn hydrate(
        fttsq: Vec<u8>,
        codec: Vec<u8>,
        vocab_json: &str,
        merges_txt: &str,
        tokenizer_config_json: &str,
    ) -> Result<WasmEngine, JsValue> {
        let codec_file = ftts_artifacts::safetensors::SafetensorsFile::from_bytes(codec)
            .map_err(|error| js_error("codec checkpoint rejected", error))?;
        let codec =
            CodecCheckpoint::load_from_file(&codec_file, Path::new("browser://codec.safetensors"))
                .map_err(|error| js_error("codec hydration failed", error))?;
        drop(codec_file);
        Self::hydrate_with_codec(
            fttsq,
            codec,
            None,
            vocab_json,
            merges_txt,
            tokenizer_config_json,
        )
    }

    /// Hydration from an artifact buffer plus an ALREADY-widened codec.
    ///
    /// Taking the codec pre-built is what lets the streaming path keep the codec's source bytes
    /// out of the artifact's memory window entirely (see [`ModelStaging`]).
    #[cfg(not(unix))]
    fn hydrate_with_codec(
        fttsq: Vec<u8>,
        codec: CodecCheckpoint,
        cold_elided: Option<u64>,
        vocab_json: &str,
        merges_txt: &str,
        tokenizer_config_json: &str,
    ) -> Result<WasmEngine, JsValue> {
        // The whole-buffer path: run the artifact phase and the codec phase back to back. It
        // carries a higher peak than the streaming path by construction, because both raw files
        // are already in hand; see `ModelStaging` for the ordering that a phone needs.
        let mut hydrated = HydratedArtifact::build(fttsq, cold_elided)?;
        let tokenizer = QwenTokenizer::from_files_using_environment(TokenizerFiles {
            vocab_json,
            merges_txt,
            tokenizer_config_json,
        })
        .map_err(|error| js_error("tokenizer unusable", error))?;
        Ok(WasmEngine {
            talker: hydrated.talker.take().expect("built above"),
            codec,
            tokenizer,
            int8: hydrated.int8.take(),
            speaker_encoder: hydrated.speaker_encoder.take(),
            #[cfg(not(unix))]
            denoiser: None,
        })
    }
}

/// The talker side of hydration, complete and no longer holding the artifact.
///
/// Exists as its own step because ORDER is the lever. Building the fused int8 tables is what
/// makes the artifact's 0.69 GB hot prefix dead, and wasm memory only grows — so the hole that
/// release leaves is worth exactly as much as whatever gets allocated after it. Running this
/// BEFORE the codec is staged lets the codec's 0.46 GB source and 0.457 GB checkpoint land inside
/// that hole instead of on top of it. Measured the wrong way round, the peak was 2.10 GB.
#[cfg(not(unix))]
struct HydratedArtifact {
    talker: Option<TalkerCheckpoint>,
    int8: Option<Arc<ftts_model_qwen::generate::Int8Route>>,
    speaker_encoder: Option<SpeakerEncoder>,
}

#[cfg(not(unix))]
impl HydratedArtifact {
    fn build(fttsq: Vec<u8>, cold_elided: Option<u64>) -> Result<Self, JsValue> {
        let staged_bytes = fttsq.len() as u64;
        let artifact = Arc::new(match cold_elided {
            Some(full_bytes) => {
                ftts_artifacts::fttsq::MappedFttsq::from_prefix_bytes(fttsq, full_bytes)
                    .map_err(|error| js_error("artifact prefix rejected", error))?
            }
            None => ftts_artifacts::fttsq::MappedFttsq::from_bytes(fttsq)
                .map_err(|error| js_error("artifact rejected", error))?,
        });

        // Re-derive the truncation point from the directory the artifact itself declares, and
        // insist the caller stopped exactly there. Without this the JS side's byte offset is an
        // unchecked assumption about the file layout, and the failure it would produce — hot
        // tensors silently missing their tails — is the kind that reads as a model bug rather
        // than a loader bug. `prefix_len_omitting` also returns None if a future layout puts a
        // retained section after the cold one, which turns a silent 622 MB regression into a
        // refusal to load.
        if cold_elided.is_some() {
            let expected = artifact
                .reader()
                .prefix_len_omitting(&[ftts_artifacts::fttsq::AccessClass::ColdTextEmbedding]);
            match expected {
                Some(len) if len == staged_bytes => {}
                Some(len) => {
                    return Err(js_error(
                        "hot-prefix length does not match the artifact layout (expected/staged)",
                        format!("{len}/{staged_bytes}"),
                    ));
                }
                None => {
                    return Err(js_error(
                        "artifact layout does not permit eliding the cold text embedding",
                        "a retained section starts after the cold section",
                    ));
                }
            }

            // The provided-row path widens bf16, while the mapped path also accepts f32 and q8
            // cold tables. The directory survives truncation (it sits at the head of the file), so
            // the stored dtype can be checked here rather than assumed — an f32 or q8 cold table
            // would otherwise be read at the wrong stride and produce fluent wrong audio.
            let cold = artifact
                .reader()
                .tensor("talker.model.text_embedding.weight")
                .ok_or_else(|| {
                    js_error(
                        "artifact declares no cold text embedding",
                        "cannot elide what is not there",
                    )
                })?;
            if cold.dtype != ftts_artifacts::fttsq::StoredDtype::Bf16 {
                return Err(js_error(
                    "cold text embedding must be bf16 to be supplied row-wise",
                    format!("{:?}", cold.dtype),
                ));
            }
        }

        let label = Path::new("browser://model.fttsq");
        let talker = TalkerCheckpoint::load_fttsq_mapped(
            Arc::clone(&artifact),
            label,
            ftts_model_qwen::generate::hot_elision_from_environment(),
        )
        .map_err(|error| js_error("talker hydration failed", error))?;

        // ── everything the artifact is still needed for, done here, once ────────────────────
        //
        // Both readers of the artifact's hot bytes run now so the bytes can be released: the fused
        // W8A8 tables (~0.46 GB of Q8, previously rebuilt on EVERY synthesize call and stacked on
        // top of a heap wasm cannot shrink) and the speaker encoder (~18 MB, previously lazy on
        // first enrollment — cheap enough to pay always, and paying it is what lets voice cloning
        // keep working after the release).
        let mut talker = talker;
        let int8 = {
            let talker_layers = talker.talker_layer_weights();
            let micro_layers = talker.microdecoder_layer_weights();
            let residual = talker.residual_embedding_slices();
            let heads = talker.microdecoder_head_slices();
            let micro_residual = &residual[..residual.len() - 1];
            ftts_model_qwen::generate::prepare_int8_route(
                &TalkerConfig::default(),
                &talker.talker_weights(&talker_layers),
                &MicrodecoderConfig::default(),
                &talker.microdecoder_weights(&micro_layers, micro_residual, &heads),
                Some(&artifact),
            )
            .map(Arc::new)
        };
        let speaker_encoder = SpeakerEncoder::load_fttsq_mapped(&artifact, label).ok();

        // Drop this scope's handle first, then the checkpoint's, so the payload actually goes.
        // `release_artifact` reports whether it was the last one; a handle left outstanding would
        // look exactly like the saving working, right up until the device ran out anyway.
        drop(artifact);
        let released = talker.release_artifact();
        console_error(&format!(
            "ftts-wasm hydrate: artifact released={released}, int8 route {}, encoder {}",
            if int8.is_some() { "built" } else { "absent" },
            if speaker_encoder.is_some() {
                "built"
            } else {
                "absent"
            },
        ));

        Ok(Self {
            talker: Some(talker),
            int8,
            speaker_encoder,
        })
    }
}

// `#[wasm_bindgen]` is not decoration here: without it every method below stops being exported to
// JS while still compiling perfectly. Splitting the original impl block to make room for the
// artifact phase silently dropped `text_row_ids` and `synthesize_with_text_rows` from the glue,
// and the only symptom was "engine.text_row_ids is not a function" at the first press.
#[wasm_bindgen]
#[cfg(not(unix))]
impl WasmEngine {
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
        self.synthesize_inner(text, speaker, seed, max_frames, None, 0)
    }

    /// The cold-text-embedding ids this exact text will need, in the order the rows must arrive.
    ///
    /// The caller reads these rows out of the artifact itself (in the browser: out of OPFS) and
    /// hands them to [`WasmEngine::synthesize_with_text_rows`]. Ids come from the engine rather
    /// than from the caller's own tokenization on purpose — the two must agree exactly, and the
    /// only way to guarantee that is to ask the tokenizer that will actually run.
    ///
    /// # Errors
    ///
    /// Throws when text preparation fails.
    pub fn text_row_ids(&self, text: &str) -> Result<Vec<u32>, JsValue> {
        let prepared_raw = self
            .tokenizer
            .prepare(text, &NormalizationOptions::default())
            .map_err(|error| js_error("text preparation failed", error))?;
        let wrapped = TalkerCheckpoint::wrap_target_ids(&prepared_raw.token_ids);
        let mut ids = TalkerCheckpoint::utterance_text_ids(&wrapped);
        // Ascending-and-distinct is the wire contract: the caller reads rows in this order and
        // `from_provided_rows` rebuilds the same order independently. Sorting here means the
        // caller never has to know that rule, only to preserve the order it was given.
        ids.sort_unstable();
        ids.dedup();
        Ok(ids)
    }

    /// [`WasmEngine::synthesize`] for an engine whose artifact omits the cold text embedding.
    ///
    /// `rows_bf16` is one bf16 row per id from [`WasmEngine::text_row_ids`], in that order,
    /// concatenated. Roughly 4 KB per distinct token, so a long utterance costs a couple of MB of
    /// reads against 622 MB of permanently resident linear memory saved.
    ///
    /// # Errors
    ///
    /// As [`WasmEngine::synthesize`], plus a refusal when the rows do not match the ids.
    pub fn synthesize_with_text_rows(
        &self,
        text: &str,
        speaker: &[f32],
        seed: u64,
        max_frames: u32,
        rows_bf16: &[u8],
        packet_frames: u32,
    ) -> Result<Vec<f32>, JsValue> {
        self.synthesize_inner(
            text,
            speaker,
            seed,
            max_frames,
            Some(rows_bf16),
            packet_frames,
        )
    }

    fn synthesize_inner(
        &self,
        text: &str,
        speaker: &[f32],
        seed: u64,
        max_frames: u32,
        provided_rows: Option<&[u8]>,
        packet_frames: u32,
    ) -> Result<Vec<f32>, JsValue> {
        if speaker.len() != TALKER_HIDDEN {
            return Err(js_error(
                "speaker vector must be exactly 1,024 floats",
                speaker.len(),
            ));
        }
        let mem_at_entry = linear_memory_bytes();
        let prepared_raw = self
            .tokenizer
            .prepare(text, &NormalizationOptions::default())
            .map_err(|error| js_error("text preparation failed", error))?;
        let wrapped = TalkerCheckpoint::wrap_target_ids(&prepared_raw.token_ids);
        let prepared = PreparedText::new(wrapped.clone(), prepared_raw.normalization_trace);

        let ids = TalkerCheckpoint::utterance_text_ids(&wrapped);
        // Directly comparable to the CLI's `prepared_token_count`. If these disagree the two are
        // conditioning on different text, and every audio comparison downstream is meaningless.
        console_error(&format!(
            "ftts-wasm tokens: prepared {} wrapped {} distinct_rows {}",
            prepared_raw.token_ids.len(),
            wrapped.len(),
            {
                let mut d = ids.clone();
                d.sort_unstable();
                d.dedup();
                d.len()
            }
        ));
        let table = match provided_rows {
            Some(rows) => TextEmbeddingTable::from_provided_rows(&ids, rows)
                .map_err(|error| js_error("provided text rows rejected", error))?,
            None => self
                .talker
                .gather_text_rows(&ids)
                .map_err(|error| js_error("text embedding gather failed", error))?,
        };
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

        let mut generator = QwenGenerator::new_with_prepared_int8(
            QwenGeneratorConfig {
                talker_config: TalkerConfig::default(),
                talker_weights: self.talker.talker_weights(&talker_layers),
                text: self.talker.text_weights(&table),
                cold_rows: None,
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
            self.int8.clone(),
        );

        let clock = js_sys::Date::now;
        let mem_after_generator = linear_memory_bytes();
        let prefill_started = clock();
        generator
            .begin_utterance(&prepared, UtteranceStart::Fresh)
            .map_err(|error| js_error("prefill failed", error))?;
        let prefill_ms = clock() - prefill_started;
        let mem_after_prefill = linear_memory_bytes();
        let mut frames_ms = 0.0f64;
        let mut codec_ms = 0.0f64;

        // The CLI's text-proportional EOS backstop, so a runaway prompt cannot spin forever.
        let cap = if max_frames == 0 {
            wrapped.len() * 4 + 64
        } else {
            max_frames as usize
        };

        // How many frames the codec decodes per call. Bigger packets mean a larger `m` in every
        // codec GEMM, which is the regime blocking pays off in; the cost is that a packet's
        // activations scale with it. Output does not change either way — the streaming==batch
        // gate holds under every packet schedule — so this is purely a speed/memory dial, and
        // `0` keeps the long-standing default rather than silently moving it.
        let packet_target = if packet_frames == 0 {
            4
        } else {
            packet_frames as usize
        };
        let mut state = self.codec.stream_state();
        let mut pcm: Vec<f32> = Vec::new();
        let mut packet_pcm: Vec<f32> = Vec::new();
        let mut packet: Vec<i32> = Vec::with_capacity(16 * packet_target);
        let mut emitted_in_packet = 0usize;
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

        let mut emitted = 0usize;
        for _ in 0..cap {
            let frame_started = clock();
            let step = generator
                .next_frame()
                .map_err(|error| js_error("generation failed", error))?;
            frames_ms += clock() - frame_started;
            match step {
                FrameStep::Frame(frame) => {
                    // The first frames' codes, for cross-target comparison. Audio can only say
                    // THAT two builds diverge; codes say whether the divergence is upstream of the
                    // codec (token generation) or inside it, which is the difference between
                    // hunting the talker and hunting the vocoder.
                    if emitted < 3 {
                        console_error(&format!("ftts-wasm codes[{emitted}]: {:?}", frame.codes));
                    }
                    emitted += 1;
                    for code in &frame.codes {
                        packet.push(i32::try_from(*code).map_err(|error| {
                            js_error("generated code does not fit the codec's i32", error)
                        })?);
                    }
                    emitted_in_packet += 1;
                    if emitted_in_packet == packet_target {
                        flush(&mut state, &mut packet, &mut emitted_in_packet, &mut pcm)?;
                    }
                }
                FrameStep::Finished => break,
                // `AwaitingText` is legal only for continuation utterances, and this path always
                // starts `Fresh` with no append_text/finish_text calls — reaching it here would
                // mean a contract break upstream. Fail loudly rather than silently truncating
                // the utterance's tail.
                FrameStep::AwaitingText => {
                    return Err(js_error(
                        "generation stalled awaiting text",
                        "a fresh utterance must never await text",
                    ));
                }
            }
        }
        flush(&mut state, &mut packet, &mut emitted_in_packet, &mut pcm)?;
        // Stage split to the console: the number that says WHERE wasm time goes.
        console_error(&format!(
            "ftts-wasm timing: prefill {prefill_ms:.0}ms, talker+micro {frames_ms:.0}ms, codec {codec_ms:.0}ms, {} samples",
            pcm.len()
        ));
        // Where synthesis's memory actually goes, measured inside the call that spends it.
        console_error(&format!(
            "ftts-wasm memory: entry {:.2} GB, after generator {:.2} GB, after prefill {:.2} GB, end {:.2} GB",
            mem_at_entry / 1e9,
            mem_after_generator / 1e9,
            mem_after_prefill / 1e9,
            linear_memory_bytes() / 1e9,
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
            // Hydration builds this eagerly precisely so the artifact can be released; reaching
            // here means it did not, and the bytes it would need are gone.
            return Err(js_error(
                "speaker encoder unavailable",
                "the artifact was released before the encoder was built",
            ));
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
