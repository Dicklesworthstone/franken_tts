//! `QwenGenerator`: the model side of `ftts_core`'s [`FrameGenerator`] seam.
//!
//! `begin_utterance` runs the cold path once per request — wrapper stripping, cold text embedding,
//! the SiLU text projection, exact prompt assembly, and the full talker prefill — and parks the
//! last position's hidden state and primary logits. `next_frame` then advances one frame at a
//! time: sample `c0`, run the 15-step microdecoder for `c1..=c15`, and feed the 16 code embeddings
//! (plus one consumed trailing-text hidden, or `tts_pad` once the text stream is exhausted) back
//! through a single-position talker step.
//!
//! Everything heavy is borrowed: checkpoint hydration owns the tensors and this type holds `&[f32]`
//! views, so an utterance never clones a weight table.

use ftts_core::{
    CodeFrame, FrameGenerator, FrameStep, GenerationError, PreparedText, UtteranceStart,
};

use crate::microdecoder::{
    self, FrameState, MicroLayerQuant, MicrodecoderConfig, MicrodecoderWeights, RESIDUAL_VOCAB,
    RopeTable,
};
use crate::prompt::{
    self, CloneMode, HiddenState, PromptAssemblyInput, PromptError, PromptHeader, PromptMode,
};
use crate::sampler::{CODEC_EOS_TOKEN_ID, QwenSampler, SamplingMode};
use crate::talker::{
    self, CODE_GROUP_COUNT, PRIMARY_CODE_VOCAB_SIZE, RotaryRows, TalkerConfig, TalkerKvCache,
    TalkerLayerQuant, TalkerWeights,
};
use ftts_artifacts::fttsq::{MappedFttsq, StoredDtype};
use ftts_kernels::int8::{QuantLinearMode, QuantizedMatrix};

/// The talker's pinned mRoPE base. The microdecoder and codec each use a different theta; crossing
/// them is a silent correctness failure, so the constant lives next to its only call sites.
const MROPE_THETA: f32 = 1.0e6;

/// Cross-target seam taps for the DISC-006 hunt (frankentts-p16p): when enabled, the generator
/// emits one stable line per frame — hashes of the tensors that cross engine seams — so a native
/// run and a wasm run of the same pinned inputs can be diffed line-by-line to name the first
/// operator whose bits diverge. Hashes are FNV-1a over little-endian `f32` bit patterns, which is
/// target-independent given identical bits. Off by default everywhere: native opts in through
/// `FTTS_DEBUG_TAPS=1`, a wasm host calls [`set_taps_enabled`] (the CLI's file-output path never
/// pays anything).
static TAP_ENABLED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
static TAP_SINK: std::sync::OnceLock<Box<dyn Fn(&str) + Send + Sync>> = std::sync::OnceLock::new();
/// Native processes opt in through the environment; read once.
#[cfg(not(target_arch = "wasm32"))]
static NATIVE_TAPS_REQUESTED: std::sync::LazyLock<bool> =
    std::sync::LazyLock::new(|| std::env::var_os("FTTS_DEBUG_TAPS").is_some());

/// Enables or disables tap emission for this process.
pub fn set_taps_enabled(enabled: bool) {
    TAP_ENABLED.store(enabled, std::sync::atomic::Ordering::Relaxed);
}

/// Installs a custom emission sink (the wasm bindings route lines to `console.error`; the
/// default writes to stderr). Idempotent per process: only the first install wins.
pub fn install_tap_sink(sink: Box<dyn Fn(&str) + Send + Sync>) {
    let _ = TAP_SINK.set(sink);
}

fn taps_active() -> bool {
    if TAP_ENABLED.load(std::sync::atomic::Ordering::Relaxed) {
        return true;
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        *NATIVE_TAPS_REQUESTED
    }
    #[cfg(target_arch = "wasm32")]
    {
        false
    }
}

fn tap_emit(line: &str) {
    if !taps_active() {
        return;
    }
    match TAP_SINK.get() {
        Some(sink) => sink(line),
        None => eprintln!("{line}"),
    }
}

/// FNV-1a over the little-endian bit patterns of a slice — identical bits, identical hash, on
/// every target.
fn tap_hash_f32(values: &[f32]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for value in values {
        for byte in value.to_bits().to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x100_0000_01b3);
        }
    }
    hash
}

/// The same hash over 16-bit code groups.
fn tap_hash_u32(values: &[u32]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for value in values {
        for byte in value.to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x100_0000_01b3);
        }
    }
    hash
}

/// Reads one Q8 tensor out of a canonical artifact as an executable [`QuantizedMatrix`].
///
/// The artifact's payload IS the canonical quantization — the converter and the runtime share one
/// quantizer, pinned byte-identical by a cross-crate test — so consuming it directly skips both
/// the bf16→f32 widen and the f32→int8 requantize the runtime route would otherwise pay at
/// startup. `None` (wrong dtype, missing scales, geometry mismatch) sends the caller to the
/// runtime-quantization fallback rather than failing hydration.
pub(crate) fn q8_from_artifact(
    artifact: &MappedFttsq,
    name: &str,
    n: usize,
    k: usize,
) -> Option<QuantizedMatrix> {
    let entry = artifact.reader().tensor(name)?;
    if entry.dtype != StoredDtype::Q8 {
        return None;
    }
    let scales_name = entry.scales.clone()?;
    let data_bytes = artifact.tensor_bytes(name).ok()?;
    if data_bytes.len() != n.checked_mul(k)? {
        return None;
    }
    let scales_bytes = artifact.tensor_bytes(&scales_name).ok()?;
    if scales_bytes.len() != n.checked_mul(4)? {
        return None;
    }
    // Q8 payload bytes are the two's-complement values themselves; this cast-copy is the whole
    // hydration cost (a memcpy), replacing a widen + max-scan + rounding pass per row.
    let data = data_bytes.iter().map(|&byte| byte as i8).collect();
    let scales = scales_bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|chunk| f32::from_le_bytes(*chunk))
        .collect();
    Some(QuantizedMatrix { data, scales, n, k })
}

/// The seven projections of one attention layer, artifact-native, in the same fused shape
/// [`TalkerLayerQuant::quantize`] produces. `None` if any tensor is absent, letting mixed or
/// older artifacts fall back to runtime quantization for the whole layer.
fn fused_layer_from_artifact(
    artifact: &MappedFttsq,
    base: &str,
    hidden: usize,
    q_width: usize,
    kv_width: usize,
    intermediate: usize,
) -> Option<(
    QuantizedMatrix,
    QuantizedMatrix,
    QuantizedMatrix,
    QuantizedMatrix,
)> {
    let q = q8_from_artifact(
        artifact,
        &format!("{base}.self_attn.q_proj.weight"),
        q_width,
        hidden,
    )?;
    let k = q8_from_artifact(
        artifact,
        &format!("{base}.self_attn.k_proj.weight"),
        kv_width,
        hidden,
    )?;
    let v = q8_from_artifact(
        artifact,
        &format!("{base}.self_attn.v_proj.weight"),
        kv_width,
        hidden,
    )?;
    let o = q8_from_artifact(
        artifact,
        &format!("{base}.self_attn.o_proj.weight"),
        hidden,
        q_width,
    )?;
    let gate = q8_from_artifact(
        artifact,
        &format!("{base}.mlp.gate_proj.weight"),
        intermediate,
        hidden,
    )?;
    let up = q8_from_artifact(
        artifact,
        &format!("{base}.mlp.up_proj.weight"),
        intermediate,
        hidden,
    )?;
    let down = q8_from_artifact(
        artifact,
        &format!("{base}.mlp.down_proj.weight"),
        hidden,
        intermediate,
    )?;
    Some((
        QuantizedMatrix::concat_rows(&[&q, &k, &v]),
        o,
        QuantizedMatrix::concat_rows(&[&gate, &up]),
        down,
    ))
}

/// Every talker layer's fused int8 tables read artifact-natively, or `None` if any tensor is
/// missing. Public for the hydration parity gate (`examples/artifact_q8_hydration.rs`).
#[must_use]
pub fn talker_layers_from_artifact(
    artifact: &MappedFttsq,
    config: &TalkerConfig,
    layer_count: usize,
) -> Option<Vec<TalkerLayerQuant>> {
    (0..layer_count)
        .map(|index| {
            fused_layer_from_artifact(
                artifact,
                &format!("talker.model.layers.{index}"),
                config.hidden_size,
                config.query_width(),
                config.kv_width(),
                config.intermediate_size,
            )
            .map(|(qkv, o_proj, gate_up, down_proj)| TalkerLayerQuant {
                qkv,
                o_proj,
                gate_up,
                down_proj,
            })
        })
        .collect()
}

/// Every microdecoder layer's fused int8 tables read artifact-natively, or `None` if any tensor
/// is missing. Public for the hydration parity gate.
#[must_use]
pub fn micro_layers_from_artifact(
    artifact: &MappedFttsq,
    config: &MicrodecoderConfig,
    layer_count: usize,
) -> Option<Vec<MicroLayerQuant>> {
    (0..layer_count)
        .map(|index| {
            fused_layer_from_artifact(
                artifact,
                &format!("talker.code_predictor.model.layers.{index}"),
                config.hidden_size,
                config.q_width(),
                config.kv_width(),
                config.intermediate_size,
            )
            .map(|(qkv, o_proj, gate_up, down_proj)| MicroLayerQuant {
                qkv,
                o_proj,
                gate_up,
                down_proj,
            })
        })
        .collect()
}

/// The hot-projection elision the CURRENT process environment permits.
///
/// This is the load-time mirror of the generator's own hydration decision: a stack's f32
/// projections may be skipped exactly when that stack will run int8 with artifact-native tables
/// (route armed for it, artifact hydration not kill-switched). Callers pass the result to
/// [`crate::checkpoint::TalkerCheckpoint::load_fttsq_elided`]; the checkpoint additionally
/// verifies per tensor that the artifact really carries the Q8 payload before eliding it.
#[must_use]
pub fn hot_elision_from_environment() -> crate::checkpoint::HotElision {
    if !ftts_kernels::route::optimized_default("FTTS_INT8") || !artifact_q8_enabled() {
        return crate::checkpoint::HotElision::default();
    }
    let scope = std::env::var("FTTS_INT8_SCOPE").unwrap_or_default();
    let (talker, micro) = match scope.as_str() {
        "talker" => (true, false),
        "micro" => (false, true),
        _ => (true, true),
    };
    crate::checkpoint::HotElision {
        talker,
        micro,
        // The per-depth heads feed only the int8 coarse-score path, which reads the
        // artifact's Q8 bytes natively (frankentts-x7bt); widening them is pure startup
        // cost whenever the artifact verifiably carries all fifteen as Q8 — the load-time
        // closure enforces that all-or-nothing and keeps widened heads otherwise.
        micro_heads: micro,
    }
}

/// `FTTS_ARTIFACT_Q8=0` forces the widen-then-requantize hydration even when a canonical
/// artifact is available — the A/B and forensics switch for artifact-native hydration.
fn artifact_q8_enabled() -> bool {
    !matches!(
        std::env::var("FTTS_ARTIFACT_Q8").as_deref(),
        Ok("0") | Ok("off") | Ok("false")
    )
}
/// The cold text embedding table and the learned biased SiLU text projection.
///
/// Production widths are `(embed_width, intermediate, hidden) = (2048, 2048, 1024)`; the fields
/// stay explicit so tiny-geometry tests exercise the same code path.
#[derive(Clone, Copy, Debug)]
pub struct TextEmbeddingWeights<'a> {
    /// Compact row-major `[gathered.len(), embed_width]` cold-embedding rows, one per entry of
    /// `gathered`, in the same order. Compact rather than vocab-strided: a full-stride table is
    /// 1.24 GB of address space per utterance, which wasm linear memory must actually commit.
    pub table: &'a [f32],
    /// The token ids materialized in `table`, ascending. Lookups binary-search this.
    pub gathered: &'a [u32],
    /// Width of one embedding row.
    pub embed_width: usize,
    /// Projection fc1 weight, `[intermediate, embed_width]`.
    pub fc1_weight: &'a [f32],
    /// Projection fc1 bias, `[intermediate]`.
    pub fc1_bias: &'a [f32],
    /// Projection fc2 weight, `[hidden, intermediate]`.
    pub fc2_weight: &'a [f32],
    /// Projection fc2 bias, `[hidden]`.
    pub fc2_bias: &'a [f32],
}

/// The 16 feedback embedding tables summed into every next-frame talker input.
///
/// Table 0 is the talker's own codec embedding (`[3072, hidden]`); tables 1..=15 are the residual
/// feedback tables (`[2048, hidden]` each). These are talker-input tables, distinct from the
/// microdecoder's internal per-depth embeddings.
#[derive(Clone, Debug)]
pub struct FeedbackTables<'a> {
    /// `[PRIMARY_CODE_VOCAB_SIZE, hidden]`, indexed by `c0`.
    pub talker_codec: &'a [f32],
    /// Fifteen `[RESIDUAL_VOCAB, hidden]` tables, indexed by `c1..=c15` in depth order.
    pub residual: Vec<&'a [f32]>,
}

/// Reference-voice conditioning for ICL clone mode.
#[derive(Clone, Debug)]
pub struct ReferencePrompt {
    /// The reference transcript with its official assistant wrapper still attached.
    pub wrapped_ids: Vec<u32>,
    /// Per-frame sums of the sixteen reference-code embeddings, already computed.
    pub codec: Vec<HiddenState>,
}

/// A source of cold text-embedding rows for token ids that were NOT part of the initial
/// utterance gather — the continuation-append analog of the wasm playground's cold-row
/// injection. The generator batch-gathers missing ids through this at `append_text`
/// time, on the synthesis thread between frames (never inside the frame hot loop), and
/// keeps them in a private overlay consulted when the compact primary table misses.
/// Absent a source, an append reaching ungathered ids keeps today's fail-closed error.
pub trait ColdTextRows {
    /// Returns `(sorted ids, row-major rows at the text embed width)` covering `ids`.
    ///
    /// # Errors
    ///
    /// If any id cannot be produced; partial coverage is an error, not a silent gap.
    fn gather_rows(&self, ids: &[u32]) -> Result<(Vec<u32>, Vec<f32>), GenerationError>;
}

impl std::fmt::Debug for dyn ColdTextRows + '_ {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("ColdTextRows")
    }
}

/// Everything a [`QwenGenerator`] borrows or owns for its lifetime.
pub struct QwenGeneratorConfig<'a> {
    pub talker_config: TalkerConfig,
    pub talker_weights: TalkerWeights<'a>,
    pub text: TextEmbeddingWeights<'a>,
    /// Cold-row source for continuation appends; `None` keeps appends restricted to the
    /// initially gathered id set (fail-closed).
    pub cold_rows: Option<&'a dyn ColdTextRows>,
    pub feedback: FeedbackTables<'a>,
    pub microdecoder_config: MicrodecoderConfig,
    pub microdecoder_weights: MicrodecoderWeights<'a>,
    pub prompt_mode: PromptMode,
    pub header: PromptHeader,
    pub tts_eos: HiddenState,
    pub reference: Option<ReferencePrompt>,
    pub sampling_mode: SamplingMode,
    /// Seed for the production sampler; canonical greedy never consumes RNG state.
    pub seed: u64,
}

/// Per-utterance autoregressive state, dropped and rebuilt by every `begin_utterance`.
#[derive(Debug, PartialEq)]
struct UtteranceState {
    /// Final-norm hidden of the newest talker position, `[hidden]`.
    pending_hidden: Vec<f32>,
    /// Primary-code logits of the newest talker position, `[3072]`.
    pending_logits: Vec<f32>,
    /// One projected text hidden consumed per generated frame; `tts_pad` afterwards.
    trailing_text_hidden: Vec<HiddenState>,
    next_position: usize,
    frames_emitted: usize,
    /// Group-0 codes only — residuals never enter the repetition-penalty history.
    group_zero_history: Vec<u32>,
    finished: bool,
    /// Whether this utterance accepts text chunks and holds its terminal EOS back.
    continuation: bool,
    /// The terminal EOS marker is inside the trailing stream: true for fresh utterances from
    /// the start, and for continuations once `finish_text` releases it.
    text_finished: bool,
}

/// The armed W8A8 int8 route: quantized projection tables for both stacks plus the dot tier.
///
/// Built once at construction when the `FTTS_INT8=1` kill-switch is armed (staged levers 2a+2b);
/// `None` leaves the f32 reference path structurally untouched. Norms, QK-Norm, rotary,
/// attention, per-depth heads/embeddings, the final norms, and the primary head remain f32 in
/// both stacks per the fixed doctrine recipe.
#[derive(Debug)]
pub struct Int8Route {
    talker: Vec<TalkerLayerQuant>,
    micro: Vec<MicroLayerQuant>,
    /// Quantized per-depth heads for the int8+f32-refine scoring path (empty when the
    /// microdecoder stack is not armed).
    micro_heads: Vec<ftts_kernels::int8::QuantizedMatrix>,
    mode: QuantLinearMode,
}

/// Build the fused W8A8 tables ONCE, so they can outlive the artifact they were read from.
///
/// # Why this is separable, and why it matters on a phone
///
/// Fusing QKV and gate‖up into one matrix each is a copy by definition — the whole point is that
/// the three projections end up contiguous. Read artifact-natively that copy is ~0.46 GB of Q8,
/// duplicating bytes that are already resident in the staged artifact, and it was being rebuilt on
/// every single synthesize call, landing on top of the existing high-water mark. Measured in
/// WebKit: entry 1.61 GB, after the generator 2.07 GB, and the tab died there.
///
/// The route owns everything it returns — `Vec<TalkerLayerQuant>`, `Vec<MicroLayerQuant>`,
/// `Vec<QuantizedMatrix>` — and borrows nothing from `artifact`. So a caller can build it during
/// hydration, drop the artifact, and keep synthesizing: the fused tables are the only reader of
/// those hot bytes once they exist.
#[must_use]
pub fn prepare_int8_route(
    talker_config: &TalkerConfig,
    talker_weights: &TalkerWeights<'_>,
    microdecoder_config: &MicrodecoderConfig,
    microdecoder_weights: &MicrodecoderWeights<'_>,
    artifact: Option<&MappedFttsq>,
) -> Option<Int8Route> {
    ftts_kernels::route::optimized_default("FTTS_INT8").then(|| {
        // FTTS_INT8_SCOPE narrows the lever for sensitivity attribution: `talker` or
        // `micro` quantizes one stack and leaves the other on the f32 reference. An empty
        // table below means "this stack stays f32" at the branch sites.
        let scope = std::env::var("FTTS_INT8_SCOPE").unwrap_or_default();
        let (arm_talker, arm_micro) = match scope.as_str() {
            "talker" => (true, false),
            "micro" => (false, true),
            _ => (true, true),
        };
        let artifact = artifact.filter(|_| artifact_q8_enabled());
        Int8Route {
            talker: if arm_talker {
                talker_weights
                    .layers
                    .iter()
                    .enumerate()
                    .map(|(index, layer)| {
                        artifact
                            .and_then(|artifact| {
                                let talker = &*talker_config;
                                fused_layer_from_artifact(
                                    artifact,
                                    &format!("talker.model.layers.{index}"),
                                    talker.hidden_size,
                                    talker.query_width(),
                                    talker.kv_width(),
                                    talker.intermediate_size,
                                )
                            })
                            .map(|(qkv, o_proj, gate_up, down_proj)| TalkerLayerQuant {
                                qkv,
                                o_proj,
                                gate_up,
                                down_proj,
                            })
                            .unwrap_or_else(|| TalkerLayerQuant::quantize(&*talker_config, layer))
                    })
                    .collect()
            } else {
                Vec::new()
            },
            micro: if arm_micro {
                microdecoder_weights
                    .layers
                    .iter()
                    .enumerate()
                    .map(|(index, layer)| {
                        artifact
                            .and_then(|artifact| {
                                let micro = &*microdecoder_config;
                                fused_layer_from_artifact(
                                    artifact,
                                    &format!("talker.code_predictor.model.layers.{index}"),
                                    micro.hidden_size,
                                    micro.q_width(),
                                    micro.kv_width(),
                                    micro.intermediate_size,
                                )
                            })
                            .map(|(qkv, o_proj, gate_up, down_proj)| MicroLayerQuant {
                                qkv,
                                o_proj,
                                gate_up,
                                down_proj,
                            })
                            .unwrap_or_else(|| {
                                MicroLayerQuant::quantize(&*microdecoder_config, layer)
                            })
                    })
                    .collect()
            } else {
                Vec::new()
            },
            micro_heads: if arm_micro {
                (0..microdecoder_weights.heads.len())
                    .map(|head| {
                        // A --micro-q8 artifact carries the head verbatim as per-row Q8
                        // (frankentts-x7bt): consuming those bytes skips the requantize
                        // pass and makes the stored bytes authoritative. The fallback is
                        // bit-identical — quantizing the widened form of the same rows
                        // reproduces scale (max|row| = 127·scale) and values exactly — so
                        // this is a startup-cost lever, never a numerics one.
                        artifact
                            .and_then(|artifact| {
                                q8_from_artifact(
                                    artifact,
                                    &format!("talker.code_predictor.lm_head.{head}.weight"),
                                    RESIDUAL_VOCAB,
                                    microdecoder_config.hidden_size,
                                )
                            })
                            .unwrap_or_else(|| {
                                ftts_kernels::int8::QuantizedMatrix::quantize(
                                    microdecoder_weights.heads[head],
                                    RESIDUAL_VOCAB,
                                    microdecoder_config.hidden_size,
                                )
                            })
                    })
                    .collect()
            } else {
                Vec::new()
            },
            mode: ftts_kernels::int8::quant_mode_from_environment(),
        }
    })
}

/// The Qwen3-TTS Base implementation of [`FrameGenerator`].
#[derive(Debug)]
pub struct QwenGenerator<'a> {
    talker_config: TalkerConfig,
    talker_weights: TalkerWeights<'a>,
    text: TextEmbeddingWeights<'a>,
    /// Cold-row source for continuation appends (see [`ColdTextRows`]).
    cold_rows: Option<&'a dyn ColdTextRows>,
    /// Overlay for rows gathered after construction: sorted ids and their row-major
    /// embeddings at `text.embed_width`, consulted when the primary table misses.
    overlay_ids: Vec<u32>,
    overlay_rows: Vec<f32>,
    feedback: FeedbackTables<'a>,
    microdecoder_config: MicrodecoderConfig,
    microdecoder_weights: MicrodecoderWeights<'a>,
    microdecoder_rope: RopeTable,
    prompt_mode: PromptMode,
    header: PromptHeader,
    tts_eos: HiddenState,
    reference: Option<ReferencePrompt>,
    sampler: QwenSampler,
    sampling_mode: SamplingMode,
    kv: TalkerKvCache,
    frame_state: FrameState,
    utterance: Option<UtteranceState>,
    /// Shared rather than owned: the engine builds these once during hydration and lends the
    /// same tables to every utterance, which is what lets the artifact they came from be dropped.
    int8: Option<std::sync::Arc<Int8Route>>,
}

impl<'a> QwenGenerator<'a> {
    /// Builds a generator over borrowed weights.
    ///
    /// # Panics
    ///
    /// Panics when the feedback tables or text projection disagree with the talker geometry —
    /// these are checkpoint-hydration bugs, not runtime conditions.
    #[must_use]
    pub fn new(config: QwenGeneratorConfig<'a>) -> Self {
        Self::new_with_artifact(config, None)
    }

    /// [`QwenGenerator::new`] with the fused int8 tables supplied rather than built.
    ///
    /// The memory-frugal entry point, and the one the browser uses. Building the route costs a
    /// ~0.46 GB Q8 copy of the hot weights; doing it per utterance rebuilt that copy on top of an
    /// already-peaked heap, and wasm memory never shrinks, so every press ratcheted the tab
    /// upward. A caller that prepared the route once with [`prepare_int8_route`] lends the same
    /// tables here for free — and, because the route borrows nothing from the artifact, can have
    /// dropped the artifact entirely by now.
    #[must_use]
    pub fn new_with_prepared_int8(
        config: QwenGeneratorConfig<'a>,
        int8: Option<std::sync::Arc<Int8Route>>,
    ) -> Self {
        Self::assemble(config, int8)
    }

    /// [`QwenGenerator::new`], additionally offered the mapped canonical artifact the checkpoint
    /// was hydrated from.
    ///
    /// When the int8 route is armed and the artifact carries a layer's Q8 tensors, that layer's
    /// projections hydrate directly from the artifact payload — the canonical quantization
    /// itself — instead of re-quantizing widened f32 copies. Payload bytes are identical to the
    /// runtime quantizer's output by the shared-quantizer contract; scales are the converter's
    /// own (the requantize round trip can drift a scale by an ulp, so the artifact is the more
    /// canonical of the two). `FTTS_ARTIFACT_Q8=0` forces the old path for A/B.
    #[must_use]
    pub fn new_with_artifact(
        config: QwenGeneratorConfig<'a>,
        artifact: Option<&MappedFttsq>,
    ) -> Self {
        let int8 = ftts_kernels::route::optimized_default("FTTS_INT8")
            .then(|| {
                prepare_int8_route(
                    &config.talker_config,
                    &config.talker_weights,
                    &config.microdecoder_config,
                    &config.microdecoder_weights,
                    artifact,
                )
            })
            .flatten()
            .map(std::sync::Arc::new);
        Self::assemble(config, int8)
    }

    /// Validate the config and build everything that is not the int8 route.
    ///
    /// Split out so the route can either be built here or handed in already built — the two
    /// constructors differ in nothing else, and duplicating a dozen shape assertions between them
    /// is how the two paths would quietly stop agreeing.
    fn assemble(config: QwenGeneratorConfig<'a>, int8: Option<std::sync::Arc<Int8Route>>) -> Self {
        let hidden = config.talker_config.hidden_size;
        assert_eq!(
            config.feedback.talker_codec.len(),
            PRIMARY_CODE_VOCAB_SIZE * hidden,
            "feedback table 0 must be [{PRIMARY_CODE_VOCAB_SIZE}, hidden]"
        );
        assert_eq!(
            config.feedback.residual.len(),
            CODE_GROUP_COUNT - 1,
            "expected {} residual feedback tables",
            CODE_GROUP_COUNT - 1
        );
        for table in &config.feedback.residual {
            assert_eq!(
                table.len(),
                RESIDUAL_VOCAB * hidden,
                "each residual feedback table must be [{RESIDUAL_VOCAB}, hidden]"
            );
        }
        assert_eq!(
            config.text.fc2_bias.len(),
            hidden,
            "text projection must land on the talker width"
        );
        assert!(
            config.text.embed_width > 0
                && config
                    .text
                    .table
                    .len()
                    .is_multiple_of(config.text.embed_width),
            "text embedding table must be [vocab, embed_width]"
        );
        assert_eq!(config.header.tts_pad.len(), hidden, "header width");
        assert_eq!(config.tts_eos.len(), hidden, "tts_eos width");
        assert_eq!(
            config.microdecoder_config.hidden_size, hidden,
            "the microdecoder conditions on talker hiddens, so widths must agree"
        );

        let microdecoder_rope = RopeTable::new(&config.microdecoder_config);
        let frame_state = FrameState::new(&config.microdecoder_config);

        // Staged levers 2a+2b: runtime W8A8, armed only by the explicit kill-switch. The default
        // path quantizes nothing and calls the untouched f32 reference functions.
        // The optimized route is the library default; `FTTS_INT8=0` or a conformance
        // reference-pin selects the f32 reference instead (ftts_kernels::route).

        Self {
            talker_config: config.talker_config,
            talker_weights: config.talker_weights,
            text: config.text,
            feedback: config.feedback,
            microdecoder_config: config.microdecoder_config,
            microdecoder_weights: config.microdecoder_weights,
            microdecoder_rope,
            prompt_mode: config.prompt_mode,
            header: config.header,
            tts_eos: config.tts_eos,
            reference: config.reference,
            sampler: QwenSampler::seeded(config.seed),
            sampling_mode: config.sampling_mode,
            kv: TalkerKvCache::new(),
            frame_state,
            utterance: None,
            cold_rows: config.cold_rows,
            overlay_ids: Vec::new(),
            overlay_rows: Vec::new(),
            int8,
        }
    }

    /// Talker positions currently cached, exposed for geometry tests.
    #[must_use]
    pub fn cached_positions(&self) -> usize {
        self.kv.len()
    }

    /// Embeds token ids through the cold table and projects them to talker width.
    /// Makes every id in `ids` resolvable by [`Self::project_text_ids`], batch-gathering
    /// the misses through the cold-row source into the overlay.
    ///
    /// Runs on the synthesis thread between frames (the `append_text` path), never in
    /// the frame hot loop; the gather is one bounded read per missing id from the cold
    /// table. Partial coverage from the source is an error — a silently missing row
    /// would surface later as fluent wrong audio.
    fn ensure_text_rows(&mut self, ids: &[u32]) -> Result<(), GenerationError> {
        let mut missing: Vec<u32> = ids
            .iter()
            .copied()
            .filter(|id| {
                self.text.gathered.binary_search(id).is_err()
                    && self.overlay_ids.binary_search(id).is_err()
            })
            .collect();
        if missing.is_empty() {
            return Ok(());
        }
        missing.sort_unstable();
        missing.dedup();
        let Some(source) = self.cold_rows else {
            return Err(GenerationError::new(format!(
                "append reaches {} text id(s) outside the gathered table (first: {}) and no \
                 cold-row source is attached; construct the generator with `cold_rows` to \
                 support continuation appends over the open vocabulary",
                missing.len(),
                missing[0]
            )));
        };
        let (got_ids, rows) = source.gather_rows(&missing)?;
        let width = self.text.embed_width;
        if got_ids.len() * width != rows.len() {
            return Err(GenerationError::new(format!(
                "cold-row source returned {} rows worth of data for {} ids at width {width}",
                rows.len() / width.max(1),
                got_ids.len()
            )));
        }
        for id in &missing {
            if got_ids.binary_search(id).is_err() {
                return Err(GenerationError::new(format!(
                    "cold-row source did not cover text id {id}"
                )));
            }
        }
        for (slot, id) in got_ids.iter().enumerate() {
            if let Err(at) = self.overlay_ids.binary_search(id) {
                self.overlay_ids.insert(at, *id);
                let row = &rows[slot * width..(slot + 1) * width];
                self.overlay_rows
                    .splice(at * width..at * width, row.iter().copied());
            }
        }
        Ok(())
    }

    fn project_text_ids(&self, ids: &[u32]) -> Result<Vec<HiddenState>, GenerationError> {
        let embed_width = self.text.embed_width;
        let mut embedded = Vec::with_capacity(ids.len() * embed_width);
        for &id in ids {
            if let Ok(slot) = self.text.gathered.binary_search(&id) {
                embedded.extend_from_slice(
                    &self.text.table[slot * embed_width..(slot + 1) * embed_width],
                );
                continue;
            }
            // Rows gathered after construction (continuation appends) live in the overlay.
            let Ok(slot) = self.overlay_ids.binary_search(&id) else {
                return Err(GenerationError::new(format!(
                    "text token {id} was not gathered into the compact embedding table; the \
                     utterance id set must cover every id the prompt can reach, or a cold-row \
                     source must be attached for continuation appends"
                )));
            };
            embedded.extend_from_slice(
                &self.overlay_rows[slot * embed_width..(slot + 1) * embed_width],
            );
        }

        let hidden = self.talker_config.hidden_size;
        let intermediate = self.text.fc1_bias.len();
        let mut projected = vec![0.0_f32; ids.len() * hidden];
        if !ids.is_empty() {
            talker::project_text_rows(
                &embedded,
                ids.len(),
                embed_width,
                intermediate,
                hidden,
                self.text.fc1_weight,
                self.text.fc1_bias,
                self.text.fc2_weight,
                self.text.fc2_bias,
                &mut projected,
            );
        }
        Ok(projected.chunks(hidden).map(<[f32]>::to_vec).collect())
    }

    /// Begins an utterance from an externally assembled prefill.
    ///
    /// This is the conformance seam for the exact ladder: the oracle fixtures capture the fully
    /// assembled `[seq, hidden]` talker input (`kwargs.inputs_embeds`) and the trailing text
    /// stream, so parity tests can drive the real prefill/decode path from the oracle's own
    /// prompt without re-deriving header construction. `prefill` is row-major `[seq, hidden]`,
    /// consumed at positions `0..seq`. Production callers use [`FrameGenerator::begin_utterance`].
    pub fn begin_with_prefill(
        &mut self,
        prefill: &[f32],
        seq: usize,
        trailing_text_hidden: Vec<HiddenState>,
    ) -> Result<(), GenerationError> {
        let hidden_size = self.talker_config.hidden_size;
        if seq == 0 || prefill.len() != seq * hidden_size {
            return Err(GenerationError::new(format!(
                "prefill must be a non-empty [seq, {hidden_size}] buffer; got {} values for seq {seq}",
                prefill.len()
            )));
        }
        self.run_prefill(prefill.to_vec(), seq, trailing_text_hidden);
        Ok(())
    }

    /// Runs the talker over an assembled prefill and parks the newest position's state.
    fn run_prefill(&mut self, mut hidden: Vec<f32>, seq: usize, trailing: Vec<HiddenState>) {
        self.utterance = None;
        self.kv.clear();
        let hidden_size = self.talker_config.hidden_size;
        let positions: Vec<i64> = (0..seq as i64).collect();
        let (cos, sin) = talker::mrope_rows(&positions, self.talker_config.head_dim, MROPE_THETA);
        let mask = causal_mask(seq);
        // A [3072] buffer selects the last-row-only head: prefill consumes only the newest
        // position's logits, and the projected row is byte-identical to the full-head form.
        let mut logits = vec![0.0_f32; PRIMARY_CODE_VOCAB_SIZE];
        let rotary = RotaryRows {
            cos: &cos,
            sin: &sin,
        };
        match self.int8.as_ref().filter(|route| !route.talker.is_empty()) {
            Some(route) => talker::forward_talker_q8(
                &self.talker_config,
                &self.talker_weights,
                &route.talker,
                rotary,
                &mask,
                &mut hidden,
                seq,
                &mut self.kv,
                &mut logits,
                route.mode,
            ),
            None => talker::forward_talker(
                &self.talker_config,
                &self.talker_weights,
                rotary,
                &mask,
                &mut hidden,
                seq,
                &mut self.kv,
                &mut logits,
            ),
        }
        self.utterance = Some(UtteranceState {
            pending_hidden: hidden[(seq - 1) * hidden_size..].to_vec(),
            pending_logits: logits,
            trailing_text_hidden: trailing,
            next_position: seq,
            frames_emitted: 0,
            group_zero_history: Vec::new(),
            finished: false,
            // Fresh semantics by default; `begin_utterance` overrides for continuations.
            continuation: false,
            text_finished: true,
        });
        if taps_active() {
            let parked = self.utterance.as_ref().expect("just constructed above");
            tap_emit(&format!(
                "ftts-tap prefill ph={:016x} plog={:016x}",
                tap_hash_f32(&parked.pending_hidden),
                tap_hash_f32(&parked.pending_logits),
            ));
        }
    }
}

fn generation_error(error: impl std::fmt::Display) -> GenerationError {
    GenerationError::new(error.to_string())
}

/// The additive causal mask for a `seq`-position prefill over an empty cache.
fn causal_mask(seq: usize) -> Vec<f32> {
    let mut mask = vec![0.0_f32; seq * seq];
    for query in 0..seq {
        for key in query + 1..seq {
            mask[query * seq + key] = f32::NEG_INFINITY;
        }
    }
    mask
}

impl FrameGenerator for QwenGenerator<'_> {
    fn begin_utterance(
        &mut self,
        prepared: &PreparedText,
        mode: UtteranceStart,
    ) -> Result<(), GenerationError> {
        if matches!(mode, UtteranceStart::Continuation)
            && (self.prompt_mode.clone_mode == CloneMode::Icl || !self.prompt_mode.streaming())
        {
            return Err(GenerationError::new(
                "continuations require streaming x-vector assembly; ICL and non-streaming \
                 prompts reject them",
            ));
        }
        self.utterance = None;

        let ids = prompt::extract_prompt_text_ids(
            &prepared.token_ids,
            self.reference.as_ref().map(|r| r.wrapped_ids.as_slice()),
        )
        .map_err(generation_error)?;
        let target_text = self.project_text_ids(&ids.target)?;
        let reference_text = match &ids.reference {
            Some(reference_ids) => Some(self.project_text_ids(reference_ids)?),
            None => None,
        };

        let assembly = prompt::assemble_prompt(PromptAssemblyInput {
            mode: self.prompt_mode,
            header: self.header.clone(),
            target_text,
            reference_text,
            reference_codec: self.reference.as_ref().map(|r| r.codec.clone()),
            tts_eos: self.tts_eos.clone(),
            hold_tts_eos: matches!(mode, UtteranceStart::Continuation),
        })
        .map_err(generation_error)?;

        let hidden_size = self.talker_config.hidden_size;
        let seq = assembly.prefill.len();
        if seq == 0 {
            return Err(GenerationError::new("assembled prompt prefill is empty"));
        }
        let mut hidden = Vec::with_capacity(seq * hidden_size);
        for state in &assembly.prefill {
            if state.len() != hidden_size {
                return Err(generation_error(PromptError::WidthMismatch {
                    expected: hidden_size,
                    actual: state.len(),
                }));
            }
            hidden.extend_from_slice(state);
        }

        self.run_prefill(hidden, seq, assembly.trailing_text_hidden);
        if matches!(mode, UtteranceStart::Continuation)
            && let Some(utterance) = self.utterance.as_mut()
        {
            utterance.continuation = true;
            utterance.text_finished = false;
        }
        Ok(())
    }

    fn append_text(&mut self, prepared: &PreparedText) -> Result<(), GenerationError> {
        if self.utterance.is_none() {
            return Err(GenerationError::new(
                "append_text called before begin_utterance",
            ));
        }
        let (continuation, text_finished, finished) = match &self.utterance {
            Some(utterance) => (
                utterance.continuation,
                utterance.text_finished,
                utterance.finished,
            ),
            None => unreachable!("checked above"),
        };
        if finished {
            return Err(GenerationError::new(
                "append_text after the model emitted codec EOS; the utterance is over",
            ));
        }
        if !continuation {
            return Err(GenerationError::new(
                "append_text on a fresh utterance whose terminal EOS already rode in the prompt",
            ));
        }
        if text_finished {
            return Err(GenerationError::new(
                "append_text after finish_text: the terminal EOS is already reachable",
            ));
        }
        let ids =
            prompt::extract_prompt_text_ids(&prepared.token_ids, None).map_err(generation_error)?;
        self.ensure_text_rows(&ids.target)?;
        let rows = self.project_text_ids(&ids.target)?;
        // Appended rows join the trailing stream and are consumed at their index — position
        // numbering (`next_position`, the KV cache) is untouched by construction, which the
        // chunked-versus-whole test pins bit-for-bit.
        if let Some(utterance) = self.utterance.as_mut() {
            utterance.trailing_text_hidden.extend(rows);
        }
        Ok(())
    }

    fn finish_text(&mut self) -> Result<(), GenerationError> {
        let Some(utterance) = self.utterance.as_mut() else {
            return Err(GenerationError::new(
                "finish_text called before begin_utterance",
            ));
        };
        if !utterance.continuation {
            return Err(GenerationError::new(
                "finish_text on a fresh utterance: its terminal EOS was never held back",
            ));
        }
        if utterance.text_finished {
            return Err(GenerationError::new(
                "finish_text called twice on one utterance",
            ));
        }
        utterance.trailing_text_hidden.push(self.tts_eos.clone());
        utterance.text_finished = true;
        Ok(())
    }

    fn next_frame(&mut self) -> Result<FrameStep, GenerationError> {
        let Some(utterance) = self.utterance.as_mut() else {
            return Err(GenerationError::new(
                "next_frame called before begin_utterance",
            ));
        };
        if utterance.finished {
            return Ok(FrameStep::Finished);
        }
        // ENTRY GATE, before anything stateful: this call's feedback step would consume
        // trailing text row `frames_emitted` (see the `.get(utterance.frames_emitted)`
        // read below). If that row does not exist yet on an open continuation, proceeding
        // would substitute `tts_pad` — the model's "text is over" signal — and wind the
        // utterance down early because the caller's LLM was slow. Stall instead. Placed
        // before `select_talker` so no RNG is drawn: a stalled-then-resumed run stays
        // bit-identical to a never-stalled one, which the parity gate pins.
        if utterance.continuation
            && !utterance.text_finished
            && utterance.frames_emitted >= utterance.trailing_text_hidden.len()
        {
            return Ok(FrameStep::AwaitingText);
        }
        let entry_logits_hash = tap_hash_f32(&utterance.pending_logits);
        let frame_index = utterance.frames_emitted;
        let primary = self
            .sampler
            .select_talker(
                &utterance.pending_logits,
                &utterance.group_zero_history,
                self.sampling_mode,
            )
            .map_err(generation_error)?;
        if primary == CODEC_EOS_TOKEN_ID {
            utterance.finished = true;
            return Ok(FrameStep::Finished);
        }

        // The subtalker follows the run's decode mode: greedy conformance stays zero-RNG, while
        // production SAMPLES every residual depth exactly as upstream's subtalker_dosample=true
        // does — greedy residuals under a sampled talker sit in a measured silence attractor
        // (frankentts-p7r; the reference reproduces it in that mismatched configuration).
        let sampler = &mut self.sampler;
        let sampling_mode = self.sampling_mode;
        // A non-finite residual logit (corrupt checkpoint, numeric blowup) must surface as a
        // `GenerationError` like every other failure in this function, not a panic. The selector
        // signature is infallible, so the first failure is parked here and re-raised after the
        // frame call returns; the fallback index 0 is never emitted because the error wins.
        let mut sampler_failure: Option<crate::sampler::SamplerError> = None;
        let select = |logits: &[f32]| match sampling_mode {
            SamplingMode::CanonicalGreedy => microdecoder::argmax(logits),
            SamplingMode::Production => {
                match sampler.select_microdecoder(logits, SamplingMode::Production) {
                    Ok(code) => code as usize,
                    Err(error) => {
                        sampler_failure.get_or_insert(error);
                        0
                    }
                }
            }
        };
        let residuals = match self.int8.as_ref().filter(|route| !route.micro.is_empty()) {
            Some(route) => microdecoder::decode_frame_with_selector_q8(
                &self.microdecoder_config,
                &self.microdecoder_rope,
                &self.microdecoder_weights,
                &microdecoder::MicroQuantRoute {
                    layers: &route.micro,
                    heads: (!route.micro_heads.is_empty()).then_some(route.micro_heads.as_slice()),
                    mode: route.mode,
                },
                &mut self.frame_state,
                &utterance.pending_hidden,
                primary as usize,
                select,
            ),
            None => microdecoder::decode_frame_with_selector(
                &self.microdecoder_config,
                &self.microdecoder_rope,
                &self.microdecoder_weights,
                &mut self.frame_state,
                &utterance.pending_hidden,
                primary as usize,
                select,
            ),
        };
        if let Some(error) = sampler_failure {
            return Err(generation_error(error));
        }
        let mut codes = Vec::with_capacity(CODE_GROUP_COUNT);
        codes.push(primary);
        codes.extend(residuals.iter().map(|&code| code as u32));

        // Feedback: sum the 16 code embeddings onto one trailing-text hidden (or tts_pad).
        let mut rows: Vec<&[f32]> = Vec::with_capacity(CODE_GROUP_COUNT);
        for (depth, &code) in codes.iter().enumerate() {
            let table = if depth == 0 {
                self.feedback.talker_codec
            } else {
                self.feedback.residual[depth - 1]
            };
            let hidden = self.talker_config.hidden_size;
            rows.push(&table[code as usize * hidden..(code as usize + 1) * hidden]);
        }
        let text_row = utterance
            .trailing_text_hidden
            .get(utterance.frames_emitted)
            .map(Vec::as_slice);
        let mut next_input = vec![0.0_f32; self.talker_config.hidden_size];
        talker::form_frame_input(&rows, text_row, &self.header.tts_pad, &mut next_input);

        let (cos, sin) = talker::mrope_rows(
            &[utterance.next_position as i64],
            self.talker_config.head_dim,
            MROPE_THETA,
        );
        let mask = vec![0.0_f32; self.kv.len() + 1];
        let mut logits = vec![0.0_f32; PRIMARY_CODE_VOCAB_SIZE];
        let rotary = RotaryRows {
            cos: &cos,
            sin: &sin,
        };
        match self.int8.as_ref().filter(|route| !route.talker.is_empty()) {
            Some(route) => talker::forward_talker_q8(
                &self.talker_config,
                &self.talker_weights,
                &route.talker,
                rotary,
                &mask,
                &mut next_input,
                1,
                &mut self.kv,
                &mut logits,
                route.mode,
            ),
            None => talker::forward_talker(
                &self.talker_config,
                &self.talker_weights,
                rotary,
                &mask,
                &mut next_input,
                1,
                &mut self.kv,
                &mut logits,
            ),
        }

        utterance.pending_hidden = next_input;
        utterance.pending_logits = logits;
        utterance.next_position += 1;
        utterance.frames_emitted += 1;
        utterance.group_zero_history.push(primary);
        if taps_active() {
            tap_emit(&format!(
                "ftts-tap f={frame_index} pl={entry_logits_hash:016x} p={primary} \
                 r={:016x} nh={:016x} nl={:016x}",
                tap_hash_u32(&codes),
                tap_hash_f32(&utterance.pending_hidden),
                tap_hash_f32(&utterance.pending_logits),
            ));
        }
        Ok(FrameStep::Frame(CodeFrame { codes }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::microdecoder::{LayerWeights, RESIDUAL_DEPTHS};
    use crate::prompt::CloneMode;
    use crate::talker::{TALKER_LAYER_COUNT, TalkerLayerWeights};
    use ftts_core::{NormalizationMode, NormalizationTrace};

    /// Tiny but structurally complete geometry: talker width 8 with the mandatory 28 layers and
    /// the hardcoded 3072/2048 vocabularies, microdecoder width 8 with 2 layers.
    const HIDDEN: usize = 8;

    fn talker_config() -> TalkerConfig {
        TalkerConfig {
            hidden_size: HIDDEN,
            intermediate_size: 4,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            head_dim: 4,
            rms_norm_eps: 1e-6,
            qk_norm_eps: 1e-6,
        }
    }

    fn microdecoder_config() -> MicrodecoderConfig {
        MicrodecoderConfig {
            hidden_size: HIDDEN,
            num_layers: 2,
            num_q_heads: 4,
            num_kv_heads: 2,
            head_dim: 4,
            intermediate_size: 16,
            rope_theta: 1.0e6,
            rms_eps: 1e-6,
        }
    }

    /// Owns every tensor the generator borrows. Zeroed projections make each transformer layer an
    /// identity over the residual stream, so the test controls outputs through the heads alone.
    struct TinyWeights {
        norm: Vec<f32>,
        talker_q: Vec<f32>,
        talker_kv: Vec<f32>,
        head_norm: Vec<f32>,
        talker_o: Vec<f32>,
        talker_mlp: Vec<f32>,
        talker_down: Vec<f32>,
        codec_head: Vec<f32>,
        micro_q: Vec<f32>,
        micro_kv: Vec<f32>,
        micro_o: Vec<f32>,
        micro_mlp: Vec<f32>,
        micro_down: Vec<f32>,
        talker_codec_embedding: Vec<f32>,
        residual_feedback: Vec<Vec<f32>>,
        micro_residual_embeddings: Vec<Vec<f32>>,
        micro_heads: Vec<Vec<f32>>,
        text_table: Vec<f32>,
        text_fc1: Vec<f32>,
        text_fc1_bias: Vec<f32>,
        text_fc2: Vec<f32>,
        text_fc2_bias: Vec<f32>,
    }

    impl TinyWeights {
        fn new(eos_logit: f32) -> Box<Self> {
            let talker = talker_config();
            let micro = microdecoder_config();
            let mut codec_head = vec![0.0_f32; PRIMARY_CODE_VOCAB_SIZE * HIDDEN];
            if eos_logit != 0.0 {
                let row = CODEC_EOS_TOKEN_ID as usize * HIDDEN;
                codec_head[row..row + HIDDEN].fill(eos_logit);
            }
            Box::new(Self {
                norm: vec![1.0; HIDDEN],
                talker_q: vec![0.0; talker.query_width() * HIDDEN],
                talker_kv: vec![0.0; talker.kv_width() * HIDDEN],
                head_norm: vec![1.0; talker.head_dim],
                talker_o: vec![0.0; HIDDEN * talker.query_width()],
                talker_mlp: vec![0.0; talker.intermediate_size * HIDDEN],
                talker_down: vec![0.0; HIDDEN * talker.intermediate_size],
                codec_head,
                micro_q: vec![0.0; micro.q_width() * HIDDEN],
                micro_kv: vec![0.0; micro.kv_width() * HIDDEN],
                micro_o: vec![0.0; HIDDEN * micro.q_width()],
                micro_mlp: vec![0.0; micro.intermediate_size * HIDDEN],
                micro_down: vec![0.0; HIDDEN * micro.intermediate_size],
                talker_codec_embedding: vec![0.0; PRIMARY_CODE_VOCAB_SIZE * HIDDEN],
                residual_feedback: vec![vec![0.0; RESIDUAL_VOCAB * HIDDEN]; CODE_GROUP_COUNT - 1],
                micro_residual_embeddings: vec![
                    vec![0.0; RESIDUAL_VOCAB * HIDDEN];
                    RESIDUAL_DEPTHS - 1
                ],
                micro_heads: vec![vec![0.0; RESIDUAL_VOCAB * HIDDEN]; RESIDUAL_DEPTHS],
                text_table: (0..4 * 4).map(|v| 0.01 * v as f32).collect(),
                text_fc1: vec![0.1; 4 * 4],
                text_fc1_bias: vec![0.0; 4],
                text_fc2: vec![0.1; HIDDEN * 4],
                text_fc2_bias: vec![0.5; HIDDEN],
            })
        }

        fn talker_layer(&self) -> TalkerLayerWeights<'_> {
            TalkerLayerWeights {
                input_layernorm: &self.norm,
                q_proj: &self.talker_q,
                k_proj: &self.talker_kv,
                v_proj: &self.talker_kv,
                q_norm: &self.head_norm,
                k_norm: &self.head_norm,
                o_proj: &self.talker_o,
                post_attention_layernorm: &self.norm,
                gate_proj: &self.talker_mlp,
                up_proj: &self.talker_mlp,
                down_proj: &self.talker_down,
            }
        }

        fn micro_layer(&self) -> LayerWeights<'_> {
            LayerWeights {
                input_norm: &self.norm,
                q_proj: &self.micro_q,
                k_proj: &self.micro_kv,
                v_proj: &self.micro_kv,
                q_norm: &self.head_norm,
                k_norm: &self.head_norm,
                o_proj: &self.micro_o,
                post_attention_norm: &self.norm,
                gate_proj: &self.micro_mlp,
                up_proj: &self.micro_mlp,
                down_proj: &self.micro_down,
            }
        }
    }

    fn header() -> PromptHeader {
        PromptHeader {
            role: vec![vec![0.5; HIDDEN]; 3],
            codec_prefill: vec![vec![0.25; HIDDEN]; 3],
            tts_bos: vec![0.5; HIDDEN],
            tts_pad: vec![0.5; HIDDEN],
        }
    }

    fn generator<'w>(
        weights: &'w TinyWeights,
        micro_layers: &'w [LayerWeights<'w>],
        micro_residual: &'w [&'w [f32]],
        micro_heads: &'w [&'w [f32]],
        residual_feedback: Vec<&'w [f32]>,
        seed: u64,
        mode: SamplingMode,
    ) -> QwenGenerator<'w> {
        QwenGenerator::new(QwenGeneratorConfig {
            talker_config: talker_config(),
            talker_weights: TalkerWeights {
                layers: vec![weights.talker_layer(); TALKER_LAYER_COUNT],
                final_norm: &weights.norm,
                codec_head: &weights.codec_head,
            },
            text: TextEmbeddingWeights {
                table: &weights.text_table,
                gathered: &[0, 1, 2, 3],
                embed_width: 4,
                fc1_weight: &weights.text_fc1,
                fc1_bias: &weights.text_fc1_bias,
                fc2_weight: &weights.text_fc2,
                fc2_bias: &weights.text_fc2_bias,
            },
            cold_rows: None,
            feedback: FeedbackTables {
                talker_codec: &weights.talker_codec_embedding,
                residual: residual_feedback,
            },
            microdecoder_config: microdecoder_config(),
            microdecoder_weights: MicrodecoderWeights {
                layers: micro_layers,
                talker_codec_embedding: &weights.talker_codec_embedding,
                residual_embeddings: crate::microdecoder::ResidualEmbeddings::Widened(
                    micro_residual,
                ),
                heads: micro_heads,
                final_norm: &weights.norm,
            },
            prompt_mode: PromptMode {
                clone_mode: CloneMode::XVector,
                non_streaming_mode: false,
            },
            header: header(),
            tts_eos: vec![0.5; HIDDEN],
            reference: None,
            sampling_mode: mode,
            seed,
        })
    }

    fn prepared(body: &[u32]) -> PreparedText {
        let mut ids = vec![151_644, 77_091, 198];
        ids.extend_from_slice(body);
        ids.extend_from_slice(&[151_645, 198, 151_644, 77_091, 198]);
        PreparedText::new(
            ids,
            NormalizationTrace {
                mode: NormalizationMode::Verbatim,
                unicode_version: "16.0".to_owned(),
                changes: Vec::new(),
            },
        )
    }

    fn drive(
        weights: &TinyWeights,
        seed: u64,
        mode: SamplingMode,
        max_frames: usize,
    ) -> Vec<CodeFrame> {
        let micro_layers = vec![weights.micro_layer(); 2];
        let micro_residual: Vec<&[f32]> = weights
            .micro_residual_embeddings
            .iter()
            .map(Vec::as_slice)
            .collect();
        let micro_heads: Vec<&[f32]> = weights.micro_heads.iter().map(Vec::as_slice).collect();
        let residual_feedback: Vec<&[f32]> = weights
            .residual_feedback
            .iter()
            .map(Vec::as_slice)
            .collect();
        let mut generator = generator(
            weights,
            &micro_layers,
            &micro_residual,
            &micro_heads,
            residual_feedback,
            seed,
            mode,
        );
        generator
            .begin_utterance(&prepared(&[1, 2]), UtteranceStart::Fresh)
            .expect("valid tiny prompt");
        let mut frames = Vec::new();
        for _ in 0..max_frames {
            match generator.next_frame().expect("frame generation succeeds") {
                FrameStep::Frame(frame) => frames.push(frame),
                FrameStep::Finished => break,
                FrameStep::AwaitingText => panic!("fresh utterance must never await text"),
            }
        }
        frames
    }

    #[test]
    fn next_frame_before_begin_utterance_is_an_error_not_a_panic() {
        let weights = TinyWeights::new(0.0);
        let micro_layers = vec![weights.micro_layer(); 2];
        let micro_residual: Vec<&[f32]> = weights
            .micro_residual_embeddings
            .iter()
            .map(Vec::as_slice)
            .collect();
        let micro_heads: Vec<&[f32]> = weights.micro_heads.iter().map(Vec::as_slice).collect();
        let residual_feedback: Vec<&[f32]> = weights
            .residual_feedback
            .iter()
            .map(Vec::as_slice)
            .collect();
        let mut generator = generator(
            &weights,
            &micro_layers,
            &micro_residual,
            &micro_heads,
            residual_feedback,
            1,
            SamplingMode::CanonicalGreedy,
        );
        assert!(generator.next_frame().is_err());
    }

    #[test]
    fn malformed_wrapper_surfaces_as_a_generation_error() {
        let weights = TinyWeights::new(0.0);
        let micro_layers = vec![weights.micro_layer(); 2];
        let micro_residual: Vec<&[f32]> = weights
            .micro_residual_embeddings
            .iter()
            .map(Vec::as_slice)
            .collect();
        let micro_heads: Vec<&[f32]> = weights.micro_heads.iter().map(Vec::as_slice).collect();
        let residual_feedback: Vec<&[f32]> = weights
            .residual_feedback
            .iter()
            .map(Vec::as_slice)
            .collect();
        let mut generator = generator(
            &weights,
            &micro_layers,
            &micro_residual,
            &micro_heads,
            residual_feedback,
            1,
            SamplingMode::CanonicalGreedy,
        );
        let bad = PreparedText::new(
            vec![1, 2, 3],
            NormalizationTrace {
                mode: NormalizationMode::Verbatim,
                unicode_version: "16.0".to_owned(),
                changes: Vec::new(),
            },
        );
        assert!(
            generator
                .begin_utterance(&bad, UtteranceStart::Fresh)
                .is_err()
        );
    }

    #[test]
    fn frames_have_sixteen_codes_and_the_kv_cache_grows_one_position_per_frame() {
        let weights = TinyWeights::new(0.0);
        let micro_layers = vec![weights.micro_layer(); 2];
        let micro_residual: Vec<&[f32]> = weights
            .micro_residual_embeddings
            .iter()
            .map(Vec::as_slice)
            .collect();
        let micro_heads: Vec<&[f32]> = weights.micro_heads.iter().map(Vec::as_slice).collect();
        let residual_feedback: Vec<&[f32]> = weights
            .residual_feedback
            .iter()
            .map(Vec::as_slice)
            .collect();
        let mut generator = generator(
            &weights,
            &micro_layers,
            &micro_residual,
            &micro_heads,
            residual_feedback,
            1,
            SamplingMode::CanonicalGreedy,
        );
        generator
            .begin_utterance(&prepared(&[1, 2]), UtteranceStart::Fresh)
            .expect("valid tiny prompt");
        // XVector streaming with a 2-token target: role (3) + summed header (2) + one target/BOS
        // position.
        let prefill_len = generator.cached_positions();
        assert_eq!(prefill_len, 6);

        let frame = match generator.next_frame().expect("frame generation succeeds") {
            FrameStep::Frame(frame) => frame,
            other => panic!("all-zero heads cannot reach EOS or stall, got {other:?}"),
        };
        assert_eq!(frame.codes.len(), CODE_GROUP_COUNT);
        assert!((frame.codes[0] as usize) < PRIMARY_CODE_VOCAB_SIZE);
        assert!(
            frame
                .codes
                .iter()
                .skip(1)
                .all(|&code| (code as usize) < RESIDUAL_VOCAB)
        );
        assert_eq!(generator.cached_positions(), prefill_len + 1);
    }

    #[test]
    fn eos_ends_the_utterance_after_min_new_tokens_and_none_is_sticky() {
        // A positive EOS head row makes EOS the argmax; the min-new-tokens processor masks it for
        // the first two frames, so exactly two frames are emitted.
        let weights = TinyWeights::new(1.0);
        let frames = drive(&weights, 1, SamplingMode::CanonicalGreedy, 10);
        assert_eq!(frames.len(), crate::sampler::TALKER_MIN_NEW_TOKENS);

        let micro_layers = vec![weights.micro_layer(); 2];
        let micro_residual: Vec<&[f32]> = weights
            .micro_residual_embeddings
            .iter()
            .map(Vec::as_slice)
            .collect();
        let micro_heads: Vec<&[f32]> = weights.micro_heads.iter().map(Vec::as_slice).collect();
        let residual_feedback: Vec<&[f32]> = weights
            .residual_feedback
            .iter()
            .map(Vec::as_slice)
            .collect();
        let mut generator = generator(
            &weights,
            &micro_layers,
            &micro_residual,
            &micro_heads,
            residual_feedback,
            1,
            SamplingMode::CanonicalGreedy,
        );
        generator
            .begin_utterance(&prepared(&[1, 2]), UtteranceStart::Fresh)
            .expect("valid tiny prompt");
        while matches!(
            generator.next_frame().expect("frames succeed"),
            FrameStep::Frame(_)
        ) {}
        assert_eq!(
            generator
                .next_frame()
                .expect("finished utterance is stable"),
            FrameStep::Finished,
            "EOS must be sticky until the next begin_utterance"
        );
    }

    #[test]
    fn same_seed_and_artifact_produce_byte_identical_code_streams() {
        let weights = TinyWeights::new(1.0);
        let left = drive(&weights, 42, SamplingMode::Production, 6);
        let right = drive(&weights, 42, SamplingMode::Production, 6);
        assert!(!left.is_empty());
        assert_eq!(left, right);
    }

    #[test]
    fn begin_utterance_resets_state_for_a_fresh_utterance() {
        let weights = TinyWeights::new(0.0);
        let micro_layers = vec![weights.micro_layer(); 2];
        let micro_residual: Vec<&[f32]> = weights
            .micro_residual_embeddings
            .iter()
            .map(Vec::as_slice)
            .collect();
        let micro_heads: Vec<&[f32]> = weights.micro_heads.iter().map(Vec::as_slice).collect();
        let residual_feedback: Vec<&[f32]> = weights
            .residual_feedback
            .iter()
            .map(Vec::as_slice)
            .collect();
        let mut generator = generator(
            &weights,
            &micro_layers,
            &micro_residual,
            &micro_heads,
            residual_feedback,
            1,
            SamplingMode::CanonicalGreedy,
        );
        generator
            .begin_utterance(&prepared(&[1, 2]), UtteranceStart::Fresh)
            .expect("valid tiny prompt");
        assert!(
            matches!(
                generator.next_frame().expect("frame generation succeeds"),
                FrameStep::Frame(_)
            ),
            "frame emitted"
        );

        generator
            .begin_utterance(&prepared(&[1, 2]), UtteranceStart::Fresh)
            .expect("second utterance");
        assert_eq!(
            generator.cached_positions(),
            6,
            "a new utterance must not inherit the previous KV prefix"
        );
    }

    #[test]
    fn full_utterance_streamed_kv_matches_causally_replayed_prefill() {
        // The cached decode path must leave precisely the same talker state as rebuilding the
        // prompt plus every emitted-frame input in one causal prefill. This is deliberately a
        // real multi-frame utterance: EOS is masked for the first two generated frames, then ends
        // the run on the third sampling decision.
        let mut weights = TinyWeights::new(1.0);
        weights.talker_q.fill(0.05);
        weights.talker_kv.fill(0.1);
        // Keep attention observable: a zero output projection would still compare the cached
        // key/value buffers, but would not prove that later positions consume them.
        weights.talker_o.fill(0.025);
        let micro_layers = vec![weights.micro_layer(); 2];
        let micro_residual: Vec<&[f32]> = weights
            .micro_residual_embeddings
            .iter()
            .map(Vec::as_slice)
            .collect();
        let micro_heads: Vec<&[f32]> = weights.micro_heads.iter().map(Vec::as_slice).collect();
        let residual_feedback: Vec<&[f32]> = weights
            .residual_feedback
            .iter()
            .map(Vec::as_slice)
            .collect();
        let mut streamed = generator(
            &weights,
            &micro_layers,
            &micro_residual,
            &micro_heads,
            residual_feedback.clone(),
            1,
            SamplingMode::CanonicalGreedy,
        );
        let initial_prefill = vec![0.25; 3 * HIDDEN];
        let trailing = vec![vec![0.5; HIDDEN], vec![0.75; HIDDEN]];
        streamed
            .begin_with_prefill(&initial_prefill, 3, trailing.clone())
            .expect("valid test prefill");

        let mut frames = Vec::new();
        while let FrameStep::Frame(frame) = streamed
            .next_frame()
            .expect("streamed frame generation succeeds")
        {
            frames.push(frame);
        }
        assert_eq!(frames.len(), crate::sampler::TALKER_MIN_NEW_TOKENS);

        let mut replay_prefill = initial_prefill;
        for (frame_index, frame) in frames.iter().enumerate() {
            let mut rows = Vec::with_capacity(CODE_GROUP_COUNT);
            for (depth, &code) in frame.codes.iter().enumerate() {
                let table = if depth == 0 {
                    &weights.talker_codec_embedding
                } else {
                    &weights.residual_feedback[depth - 1]
                };
                rows.push(&table[code as usize * HIDDEN..(code as usize + 1) * HIDDEN]);
            }
            let mut input = vec![0.0; HIDDEN];
            talker::form_frame_input(
                &rows,
                trailing.get(frame_index).map(Vec::as_slice),
                &streamed.header.tts_pad,
                &mut input,
            );
            replay_prefill.extend_from_slice(&input);
        }

        let mut replayed = generator(
            &weights,
            &micro_layers,
            &micro_residual,
            &micro_heads,
            residual_feedback,
            1,
            SamplingMode::CanonicalGreedy,
        );
        replayed
            .begin_with_prefill(&replay_prefill, 3 + frames.len(), trailing)
            .expect("replayed prefill is valid");
        let replayed_utterance = replayed.utterance.as_mut().expect("replayed state");
        replayed_utterance.frames_emitted = frames.len();
        replayed_utterance.group_zero_history = frames.iter().map(|frame| frame.codes[0]).collect();

        assert_eq!(
            streamed.kv, replayed.kv,
            "cached single-position decode and causal replay must retain identical KV buffers"
        );
        // The streamed generator's loop ended BY taking the terminal EOS decision from its parked
        // logits, so its `finished` flag is already set while the freshly replayed state has not
        // decided yet. Comparing full states across that boundary would demand the impossible;
        // instead make the replayed side take the same decision, prove it is the same decision,
        // and only then require the two states to be identical in every field.
        assert!(
            replayed.next_frame().expect("replayed terminal decision") == FrameStep::Finished,
            "the replayed state must make the same EOS decision"
        );
        assert!(
            streamed
                .next_frame()
                .expect("streamed EOS must stay sticky")
                == FrameStep::Finished,
        );
        assert_eq!(
            streamed.utterance, replayed.utterance,
            "cached decode and causal replay must park the same next-frame state"
        );
        assert_eq!(
            streamed.kv, replayed.kv,
            "the aligned terminal decisions must not have touched the KV buffers"
        );
    }

    // ── frankentts-g6an: the continuation API ────────────────────────────────────────────

    /// Builds a generator over the tiny synthetic weights for continuation tests.
    fn continuation_generator<'w>(
        weights: &'w TinyWeights,
        micro_layers: &'w [LayerWeights<'w>],
        micro_residual: &'w [&'w [f32]],
        micro_heads: &'w [&'w [f32]],
    ) -> QwenGenerator<'w> {
        let residual_feedback: Vec<&'w [f32]> = weights
            .residual_feedback
            .iter()
            .map(Vec::as_slice)
            .collect();
        generator(
            weights,
            micro_layers,
            micro_residual,
            micro_heads,
            residual_feedback,
            7,
            SamplingMode::CanonicalGreedy,
        )
    }

    /// A fixed synthetic cold-row source over `(ids, rows)` pairs at embed width 4.
    struct FixedColdRows {
        ids: Vec<u32>,
        rows: Vec<f32>,
    }

    impl ColdTextRows for FixedColdRows {
        fn gather_rows(&self, ids: &[u32]) -> Result<(Vec<u32>, Vec<f32>), GenerationError> {
            let mut got_ids = Vec::new();
            let mut rows = Vec::new();
            for &id in ids {
                let Ok(slot) = self.ids.binary_search(&id) else {
                    return Err(GenerationError::new(format!("no cold row for id {id}")));
                };
                got_ids.push(id);
                rows.extend_from_slice(&self.rows[slot * 4..(slot + 1) * 4]);
            }
            Ok((got_ids, rows))
        }
    }

    /// `generator()` with a caller-supplied text table, gather set, and cold-row source —
    /// the cold-row gate needs the two runs to differ ONLY in where id coverage comes from.
    #[allow(clippy::too_many_arguments)] // test helper: explicit beats a params struct here
    fn generator_with_text<'w>(
        weights: &'w TinyWeights,
        micro_layers: &'w [LayerWeights<'w>],
        micro_residual: &'w [&'w [f32]],
        micro_heads: &'w [&'w [f32]],
        residual_feedback: Vec<&'w [f32]>,
        text_table: &'w [f32],
        text_gathered: &'w [u32],
        cold_rows: Option<&'w dyn ColdTextRows>,
    ) -> QwenGenerator<'w> {
        QwenGenerator::new(QwenGeneratorConfig {
            talker_config: talker_config(),
            talker_weights: TalkerWeights {
                layers: vec![weights.talker_layer(); TALKER_LAYER_COUNT],
                final_norm: &weights.norm,
                codec_head: &weights.codec_head,
            },
            text: TextEmbeddingWeights {
                table: text_table,
                gathered: text_gathered,
                embed_width: 4,
                fc1_weight: &weights.text_fc1,
                fc1_bias: &weights.text_fc1_bias,
                fc2_weight: &weights.text_fc2,
                fc2_bias: &weights.text_fc2_bias,
            },
            cold_rows,
            feedback: FeedbackTables {
                talker_codec: &weights.talker_codec_embedding,
                residual: residual_feedback,
            },
            microdecoder_config: microdecoder_config(),
            microdecoder_weights: MicrodecoderWeights {
                layers: micro_layers,
                talker_codec_embedding: &weights.talker_codec_embedding,
                residual_embeddings: crate::microdecoder::ResidualEmbeddings::Widened(
                    micro_residual,
                ),
                heads: micro_heads,
                final_norm: &weights.norm,
            },
            prompt_mode: PromptMode {
                clone_mode: CloneMode::XVector,
                non_streaming_mode: false,
            },
            header: header(),
            tts_eos: vec![0.5; HIDDEN],
            reference: None,
            sampling_mode: SamplingMode::CanonicalGreedy,
            seed: 7,
        })
    }

    /// The cold-row gate (bead frankentts-edz0's re-gather seam): a continuation append
    /// reaching an id OUTSIDE the initial gather resolves through the ColdTextRows source
    /// and produces frames bit-identical to a whole-text run whose table simply included
    /// that id from the start. Also pins fail-closed behavior without a source.
    #[test]
    fn appends_outside_the_initial_gather_resolve_through_the_cold_row_source() {
        let weights = TinyWeights::new(1.0);
        let micro_layers = vec![weights.micro_layer(); 2];
        let micro_residual: Vec<&[f32]> = weights
            .micro_residual_embeddings
            .iter()
            .map(Vec::as_slice)
            .collect();
        let micro_heads: Vec<&[f32]> = weights.micro_heads.iter().map(Vec::as_slice).collect();
        let residual_feedback: Vec<&[f32]> = weights
            .residual_feedback
            .iter()
            .map(Vec::as_slice)
            .collect();

        // Row for id 4, absent from TinyWeights' table (ids 0..=3). Kept deliberately in
        // the same magnitude family as the base rows.
        let extended_table: Vec<f32> = weights
            .text_table
            .iter()
            .copied()
            .chain([0.16, 0.17, 0.18, 0.19])
            .collect();
        let extended_gathered: Vec<u32> = vec![0, 1, 2, 3, 4];

        let mut whole = generator_with_text(
            &weights,
            &micro_layers,
            &micro_residual,
            &micro_heads,
            residual_feedback.clone(),
            &extended_table,
            &extended_gathered,
            None,
        );
        whole
            .begin_utterance(&prepared(&[1, 4, 2]), UtteranceStart::Fresh)
            .expect("whole-text begin");
        let mut whole_frames = Vec::new();
        for _ in 0..32 {
            match whole.next_frame().expect("whole frame") {
                FrameStep::Frame(frame) => whole_frames.push(frame),
                FrameStep::Finished => break,
                FrameStep::AwaitingText => panic!("whole run must not stall"),
            }
        }

        // The continuation's PRIMARY table never learns id 4; the source supplies it.
        let source = FixedColdRows {
            ids: vec![4],
            rows: vec![0.16, 0.17, 0.18, 0.19],
        };
        let mut cont = generator_with_text(
            &weights,
            &micro_layers,
            &micro_residual,
            &micro_heads,
            residual_feedback.clone(),
            &weights.text_table,
            &[0, 1, 2, 3],
            Some(&source),
        );
        cont.begin_utterance(&prepared(&[1]), UtteranceStart::Continuation)
            .expect("continuation begin");
        let mut cont_frames = Vec::new();
        let mut fed = 0;
        for _ in 0..64 {
            match cont.next_frame().expect("continuation frame") {
                FrameStep::Frame(frame) => cont_frames.push(frame),
                FrameStep::Finished => break,
                FrameStep::AwaitingText => match fed {
                    0 => {
                        cont.append_text(&prepared(&[4])).expect("cold-id append");
                        fed = 1;
                    }
                    1 => {
                        cont.append_text(&prepared(&[2])).expect("warm-id append");
                        fed = 2;
                    }
                    _ => cont.finish_text().expect("finish"),
                },
            }
        }
        assert_eq!(
            cont_frames, whole_frames,
            "cold-row-resolved continuation diverged from the extended-table whole run"
        );

        // Fail-closed without a source: the same cold append is a clear error, and the
        // utterance survives to keep speaking its admitted text.
        let mut bare = generator_with_text(
            &weights,
            &micro_layers,
            &micro_residual,
            &micro_heads,
            residual_feedback,
            &weights.text_table,
            &[0, 1, 2, 3],
            None,
        );
        bare.begin_utterance(&prepared(&[1]), UtteranceStart::Continuation)
            .expect("continuation begin");
        let error = bare
            .append_text(&prepared(&[4]))
            .expect_err("cold append without a source must fail closed");
        assert!(
            error.to_string().contains("cold-row"),
            "error names the missing capability: {error}"
        );
        assert!(
            matches!(
                bare.next_frame().expect("still runs"),
                FrameStep::AwaitingText
            ),
            "the utterance survives a refused append"
        );
    }

    /// The stall gate: a continuation that CATCHES UP (AwaitingText returned, repeatedly)
    /// and then resumes on appended text produces bit-identical frames to the whole-text
    /// run. Repeated stalled polls must be side-effect-free — no RNG draw, no KV growth,
    /// no position advance — or the resumed run would diverge. This is the generator-seam
    /// half of the standing chunked==whole invariant (bead frankentts-e0wr); the engine-
    /// level wait loop is proven separately in ftts-core.
    #[test]
    fn a_stalled_then_resumed_continuation_matches_the_whole_text_run() {
        let weights = TinyWeights::new(1.0);
        let micro_layers = vec![weights.micro_layer(); 2];
        let micro_residual: Vec<&[f32]> = weights
            .micro_residual_embeddings
            .iter()
            .map(Vec::as_slice)
            .collect();
        let micro_heads: Vec<&[f32]> = weights.micro_heads.iter().map(Vec::as_slice).collect();
        let residual_feedback: Vec<&[f32]> = weights
            .residual_feedback
            .iter()
            .map(Vec::as_slice)
            .collect();

        let mut whole = generator(
            &weights,
            &micro_layers,
            &micro_residual,
            &micro_heads,
            residual_feedback.clone(),
            7,
            SamplingMode::CanonicalGreedy,
        );
        whole
            .begin_utterance(&prepared(&[1, 2, 3]), UtteranceStart::Fresh)
            .expect("whole-text begin");

        // Continuation starts with ONLY the prefill token: zero trailing headroom, so the
        // very first frame catches up and must stall.
        let mut stalled = generator(
            &weights,
            &micro_layers,
            &micro_residual,
            &micro_heads,
            residual_feedback,
            7,
            SamplingMode::CanonicalGreedy,
        );
        stalled
            .begin_utterance(&prepared(&[1]), UtteranceStart::Continuation)
            .expect("continuation begin");

        let positions_before = stalled.cached_positions();
        for poll in 0..3 {
            assert_eq!(
                stalled.next_frame().expect("stalled poll succeeds"),
                FrameStep::AwaitingText,
                "poll {poll}: an exhausted open continuation must stall, not consume pad"
            );
        }
        assert_eq!(
            stalled.cached_positions(),
            positions_before,
            "stalled polls must not grow the KV cache"
        );

        stalled
            .append_text(&prepared(&[2, 3]))
            .expect("append resumes the stream");
        stalled.finish_text().expect("finish the text stream");

        let mut stalled_frames = Vec::new();
        for _ in 0..6 {
            match stalled.next_frame().expect("resumed frame") {
                FrameStep::Frame(frame) => stalled_frames.push(frame),
                FrameStep::Finished => break,
                FrameStep::AwaitingText => panic!("finished stream must not stall again"),
            }
        }
        let mut whole_frames = Vec::new();
        for _ in 0..stalled_frames.len() {
            match whole.next_frame().expect("whole frame") {
                FrameStep::Frame(frame) => whole_frames.push(frame),
                FrameStep::Finished => break,
                FrameStep::AwaitingText => panic!("whole-text utterance must never await text"),
            }
        }
        assert_eq!(
            stalled_frames, whole_frames,
            "a stalled-then-resumed run diverged from the never-stalled run"
        );
    }

    /// Shared driver for the parity gates: run a continuation with a scripted feed
    /// (`chunks[0]` starts the utterance; later chunks are appended on demand whenever
    /// the generator stalls), and assert its frames equal the whole-text run of the
    /// concatenation. Every stall point is exercised by construction when chunks are
    /// single tokens: each boundary waits in AwaitingText until fed.
    fn assert_drip_feed_matches_whole(chunks: &[&[u32]], label: &str) {
        let weights = TinyWeights::new(1.0);
        let micro_layers = vec![weights.micro_layer(); 2];
        let micro_residual: Vec<&[f32]> = weights
            .micro_residual_embeddings
            .iter()
            .map(Vec::as_slice)
            .collect();
        let micro_heads: Vec<&[f32]> = weights.micro_heads.iter().map(Vec::as_slice).collect();
        let residual_feedback: Vec<&[f32]> = weights
            .residual_feedback
            .iter()
            .map(Vec::as_slice)
            .collect();

        let whole_ids: Vec<u32> = chunks
            .iter()
            .flat_map(|chunk| chunk.iter().copied())
            .collect();
        let mut whole = generator(
            &weights,
            &micro_layers,
            &micro_residual,
            &micro_heads,
            residual_feedback.clone(),
            7,
            SamplingMode::CanonicalGreedy,
        );
        whole
            .begin_utterance(&prepared(&whole_ids), UtteranceStart::Fresh)
            .expect("whole-text begin");
        let mut whole_frames = Vec::new();
        for _ in 0..64 {
            match whole.next_frame().expect("whole frame") {
                FrameStep::Frame(frame) => whole_frames.push(frame),
                FrameStep::Finished => break,
                FrameStep::AwaitingText => panic!("{label}: whole-text run stalled"),
            }
        }

        let mut drip = generator(
            &weights,
            &micro_layers,
            &micro_residual,
            &micro_heads,
            residual_feedback,
            7,
            SamplingMode::CanonicalGreedy,
        );
        drip.begin_utterance(&prepared(chunks[0]), UtteranceStart::Continuation)
            .expect("continuation begin");
        let mut pending = chunks[1..].iter();
        let mut finished_text = false;
        let mut drip_frames = Vec::new();
        let mut stalls = 0_usize;
        for _ in 0..512 {
            match drip.next_frame().expect("drip frame") {
                FrameStep::Frame(frame) => drip_frames.push(frame),
                FrameStep::Finished => break,
                FrameStep::AwaitingText => {
                    stalls += 1;
                    match pending.next() {
                        Some(chunk) => drip.append_text(&prepared(chunk)).expect("append"),
                        None => {
                            assert!(!finished_text, "{label}: stalled after finish_text");
                            drip.finish_text().expect("finish");
                            finished_text = true;
                        }
                    }
                }
            }
        }
        assert_eq!(
            drip_frames, whole_frames,
            "{label}: drip-fed codes diverged from whole-text codes (stalls={stalls})"
        );
        // The stall-coverage claim holds only when the model consumed the whole feed:
        // an early EOS (legal for these synthetic weights) ends the run before later
        // boundaries exist, so requiring a stall per boundary there would be false.
        if finished_text {
            assert!(
                stalls >= chunks.len().saturating_sub(1),
                "{label}: expected a stall per boundary, saw {stalls}"
            );
        }
    }

    /// Gate: a stall at EVERY consecutive frame boundary (single-token drip) changes
    /// nothing — the strongest form of the stall metamorphic (bead frankentts-hsio).
    #[test]
    fn single_token_drip_feed_stalling_at_every_boundary_matches_whole_text() {
        assert_drip_feed_matches_whole(&[&[1], &[2], &[3], &[0]], "every-boundary drip");
    }

    /// Gate: EOS-equivalence timings (bead frankentts-hsio). finish_text before the
    /// first frame, and finish_text delivered from inside a stall, both reproduce the
    /// whole-text run exactly; mid-generation finish is covered by
    /// `chunked_appends_match_whole_text_bit_for_bit`.
    #[test]
    fn finish_text_timing_is_equivalence_preserving() {
        // (a) Finish before the first frame == Fresh run of the same text.
        assert_drip_feed_matches_whole(&[&[1]], "finish-before-first-frame");
        // (c) Finish from inside a stall: the driver above always finishes from a stall
        // when the feed runs dry, so a two-chunk case exercises append-then-stall-then-
        // finish; assert_drip_feed_matches_whole verified (a) with zero appends.
        assert_drip_feed_matches_whole(&[&[1], &[2]], "finish-during-stall");
    }

    /// Gate: seeded boundary fuzz (bead frankentts-hsio) — random token sequences split
    /// at random boundaries, every case must match its whole-text run bit for bit. LCG
    /// seeding so any failure names a reproducible case in its label.
    #[test]
    fn random_chunk_boundaries_always_match_the_whole_text_run() {
        let mut state: u64 = 0x00C0_FFEE_D00D_5EED;
        let mut next = move |bound: u64| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (state >> 33) % bound
        };
        for case in 0..24 {
            let token_count = 2 + next(6) as usize; // 2..=7 tokens
            // The synthetic table gathers exactly ids 0..=3 (see `gathered` above).
            let ids: Vec<u32> = (0..token_count).map(|_| next(4) as u32).collect();
            // Random split points: each boundary independently starts a new chunk.
            let mut chunks: Vec<Vec<u32>> = vec![vec![ids[0]]];
            for &id in &ids[1..] {
                if next(2) == 0 {
                    chunks.push(Vec::new());
                }
                chunks.last_mut().expect("nonempty").push(id);
            }
            let chunk_slices: Vec<&[u32]> = chunks.iter().map(Vec::as_slice).collect();
            assert_drip_feed_matches_whole(&chunk_slices, &format!("fuzz case {case} ids={ids:?}"));
        }
    }

    /// THE exactness gate: chunk-fed synthesis is bit-identical to whole-text synthesis when
    /// the token stream is identical and each append lands before the frame that would consume
    /// it. Positions, KV, and every code must agree — appending text changes the trailing-row
    /// supply and nothing else.
    #[test]
    fn chunked_appends_match_whole_text_bit_for_bit() {
        let weights = TinyWeights::new(1.0);
        let micro_layers = vec![weights.micro_layer(); 2];
        let micro_residual: Vec<&[f32]> = weights
            .micro_residual_embeddings
            .iter()
            .map(Vec::as_slice)
            .collect();
        let micro_heads: Vec<&[f32]> = weights.micro_heads.iter().map(Vec::as_slice).collect();
        let residual_feedback: Vec<&[f32]> = weights
            .residual_feedback
            .iter()
            .map(Vec::as_slice)
            .collect();

        // Whole text [1, 2, 3]: first row rides in prefill, trailing = [r2, r3, eos].
        let mut whole = generator(
            &weights,
            &micro_layers,
            &micro_residual,
            &micro_heads,
            residual_feedback.clone(),
            7,
            SamplingMode::CanonicalGreedy,
        );
        whole
            .begin_utterance(&prepared(&[1, 2, 3]), UtteranceStart::Fresh)
            .expect("whole-text begin");

        // Chunked: chunk 1 = [1, 2] (two trailing rows of headroom), then [3] arrives while
        // the second frame is the next consumer, then the stream finishes.
        let mut chunked = generator(
            &weights,
            &micro_layers,
            &micro_residual,
            &micro_heads,
            residual_feedback,
            7,
            SamplingMode::CanonicalGreedy,
        );
        chunked
            .begin_utterance(&prepared(&[1, 2]), UtteranceStart::Continuation)
            .expect("continuation begin");
        let mut chunked_frames = Vec::new();
        for frame_index in 0..6 {
            if frame_index == 1 {
                chunked
                    .append_text(&prepared(&[3]))
                    .expect("append before the consuming frame");
            }
            if frame_index == 2 {
                chunked.finish_text().expect("finish the text stream");
            }
            match chunked.next_frame().expect("chunked frame") {
                FrameStep::Frame(frame) => chunked_frames.push(frame),
                FrameStep::Finished => break,
                FrameStep::AwaitingText => panic!("chunked run stalled: append landed late"),
            }
        }

        let mut whole_frames = Vec::new();
        for _ in 0..chunked_frames.len() {
            match whole.next_frame().expect("whole frame") {
                FrameStep::Frame(frame) => whole_frames.push(frame),
                FrameStep::Finished => break,
                FrameStep::AwaitingText => panic!("whole-text utterance must never await text"),
            }
        }

        assert_eq!(
            chunked_frames, whole_frames,
            "chunk-fed codes diverged from whole-text codes"
        );
        // Position invariance: the append changed nothing about mRoPE numbering.
        assert_eq!(
            chunked.cached_positions(),
            whole.cached_positions(),
            "appended text must not shift talker positions"
        );
        assert_eq!(
            chunked.utterance.as_ref().expect("state").next_position,
            whole.utterance.as_ref().expect("state").next_position,
        );
    }

    #[test]
    fn continuation_holds_the_terminal_until_finish_text() {
        let weights = TinyWeights::new(1.0);
        let micro_layers = vec![weights.micro_layer(); 2];
        let micro_residual: Vec<&[f32]> = weights
            .micro_residual_embeddings
            .iter()
            .map(Vec::as_slice)
            .collect();
        let micro_heads: Vec<&[f32]> = weights.micro_heads.iter().map(Vec::as_slice).collect();
        let mut cont_gen =
            continuation_generator(&weights, &micro_layers, &micro_residual, &micro_heads);
        cont_gen
            .begin_utterance(&prepared(&[1]), UtteranceStart::Continuation)
            .expect("continuation begin");
        let state = cont_gen.utterance.as_ref().expect("state exists");
        assert!(state.continuation && !state.text_finished);
        let eos = vec![0.5; HIDDEN];
        assert!(
            state
                .trailing_text_hidden
                .last()
                .is_none_or(|row| row != &eos),
            "the terminal EOS must be held out of the stream"
        );

        cont_gen
            .finish_text()
            .expect("finish releases the terminal");
        let state = cont_gen.utterance.as_ref().expect("state exists");
        assert!(state.text_finished);
        assert_eq!(
            state.trailing_text_hidden.last().expect("terminal present"),
            &eos,
            "finish_text must append exactly the configured tts_eos row"
        );
    }

    #[test]
    fn continuation_errors_are_clean_not_panics() {
        let weights = TinyWeights::new(1.0);
        let micro_layers = vec![weights.micro_layer(); 2];
        let micro_residual: Vec<&[f32]> = weights
            .micro_residual_embeddings
            .iter()
            .map(Vec::as_slice)
            .collect();
        let micro_heads: Vec<&[f32]> = weights.micro_heads.iter().map(Vec::as_slice).collect();
        let mut cont_gen =
            continuation_generator(&weights, &micro_layers, &micro_residual, &micro_heads);
        assert!(
            cont_gen.append_text(&prepared(&[1])).is_err(),
            "before begin"
        );
        assert!(cont_gen.finish_text().is_err(), "finish before begin");

        cont_gen
            .begin_utterance(&prepared(&[1]), UtteranceStart::Fresh)
            .expect("fresh begin");
        assert!(
            cont_gen.append_text(&prepared(&[2])).is_err(),
            "append on a fresh utterance is rejected"
        );
        assert!(
            cont_gen.finish_text().is_err(),
            "finish on a fresh utterance is rejected"
        );

        cont_gen
            .begin_utterance(&prepared(&[1]), UtteranceStart::Continuation)
            .expect("continuation begin");
        cont_gen.finish_text().expect("first finish");
        assert!(cont_gen.finish_text().is_err(), "double finish is rejected");
        assert!(
            cont_gen.append_text(&prepared(&[2])).is_err(),
            "append after finish"
        );
    }

    #[test]
    fn non_streaming_prompts_reject_continuations() {
        let weights = TinyWeights::new(1.0);
        let micro_layers = vec![weights.micro_layer(); 2];
        let micro_residual: Vec<&[f32]> = weights
            .micro_residual_embeddings
            .iter()
            .map(Vec::as_slice)
            .collect();
        let micro_heads: Vec<&[f32]> = weights.micro_heads.iter().map(Vec::as_slice).collect();
        let residual_feedback: Vec<&[f32]> = weights
            .residual_feedback
            .iter()
            .map(Vec::as_slice)
            .collect();
        let mut cont_gen = QwenGenerator::new(QwenGeneratorConfig {
            talker_config: talker_config(),
            talker_weights: TalkerWeights {
                layers: vec![weights.talker_layer(); TALKER_LAYER_COUNT],
                final_norm: &weights.norm,
                codec_head: &weights.codec_head,
            },
            text: TextEmbeddingWeights {
                table: &weights.text_table,
                gathered: &[0, 1, 2, 3],
                embed_width: 4,
                fc1_weight: &weights.text_fc1,
                fc1_bias: &weights.text_fc1_bias,
                fc2_weight: &weights.text_fc2,
                fc2_bias: &weights.text_fc2_bias,
            },
            cold_rows: None,
            feedback: FeedbackTables {
                talker_codec: &weights.talker_codec_embedding,
                residual: residual_feedback,
            },
            microdecoder_config: microdecoder_config(),
            microdecoder_weights: MicrodecoderWeights {
                layers: &micro_layers,
                talker_codec_embedding: &weights.talker_codec_embedding,
                residual_embeddings: crate::microdecoder::ResidualEmbeddings::Widened(
                    &micro_residual,
                ),
                heads: &micro_heads,
                final_norm: &weights.norm,
            },
            prompt_mode: PromptMode {
                clone_mode: CloneMode::XVector,
                non_streaming_mode: true,
            },
            header: header(),
            tts_eos: vec![0.5; HIDDEN],
            reference: None,
            sampling_mode: SamplingMode::CanonicalGreedy,
            seed: 7,
        });
        let error = cont_gen
            .begin_utterance(&prepared(&[1]), UtteranceStart::Continuation)
            .expect_err("non-streaming must reject continuations");
        assert!(
            error.to_string().contains("streaming x-vector"),
            "error should name the requirement: {error}"
        );
    }

    /// Cost receipt for the session's per-chunk append path (bead frankentts-g6an): the
    /// gather + projection must fit inside one 80 ms frame budget with room to spare. Measured
    /// on the tiny synthetic weights — the real-weights receipt lands with the session e2e;
    /// this one exists to catch a gross algorithmic regression (an accidental O(n²) copy).
    #[test]
    fn append_cost_fits_inside_one_frame_budget() {
        let weights = TinyWeights::new(1.0);
        let micro_layers = vec![weights.micro_layer(); 2];
        let micro_residual: Vec<&[f32]> = weights
            .micro_residual_embeddings
            .iter()
            .map(Vec::as_slice)
            .collect();
        let micro_heads: Vec<&[f32]> = weights.micro_heads.iter().map(Vec::as_slice).collect();
        let mut cont_gen =
            continuation_generator(&weights, &micro_layers, &micro_residual, &micro_heads);
        cont_gen
            .begin_utterance(&prepared(&[1]), UtteranceStart::Continuation)
            .expect("begin");
        let chunk: Vec<u32> = std::iter::repeat_n(2_u32, 25).collect();
        let chunk = prepared(&chunk);
        let started = std::time::Instant::now();
        cont_gen.append_text(&chunk).expect("append succeeds");
        let elapsed = started.elapsed();
        println!(
            "append receipt (tiny synthetic weights, 25 tokens): {elapsed:?}; frame budget 80 ms"
        );
        assert!(
            elapsed < std::time::Duration::from_millis(80),
            "append cost {elapsed:?} exceeded the 80 ms frame budget"
        );
    }
}
