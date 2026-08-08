//! The Qwen3-TTS talker decoder layer: QK-Norm GQA attention + SwiGLU MLP, in safe f32.
//!
//! This is the first code in the project that computes the model rather than describing it. It is
//! deliberately the reference shape: one layer, f32, no packing, no threads. Correctness against
//! the CPU-fp32 oracle fixtures comes first and the fast tiers are judged against it (G1 > G2).
//!
//! Geometry, all from the pinned config rather than the plan text: 28 layers, hidden 1024,
//! intermediate 3072, 16 query heads and 8 KV heads of head_dim 128 — so attention is 2048 wide,
//! wider than the hidden state, and GQA repeats each KV head across two query heads.
//!
//! Two details that a port gets wrong silently, both settled by the OQ-3/OQ-4 addendum:
//!
//! * **QK-Norm is present.** An RMSNorm over `head_dim` only, applied AFTER the q/k projection and
//!   reshape and BEFORE rotary, eps 1e-6, weight-only. Omitting it still produces plausible audio.
//! * **Everything is bias-free** except `text_projection`. `attention_bias` is false in the config,
//!   so there are no QKV or O biases to load, and a port that allocates them will not notice.

use ftts_kernels::f32ref::{
    self, F32LinearAccumulation, F32RmsNormArithmetic, F32SiluArithmetic, F32SoftmaxArithmetic,
};
use ftts_kernels::int8::{Int8Tier, QuantizedMatrix, linear_q8_dynamic};

/// Talker geometry. Field values come from `talker_config` in the pinned `config.json`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TalkerConfig {
    /// Model width.
    pub hidden_size: usize,
    /// SwiGLU inner width.
    pub intermediate_size: usize,
    /// Query head count.
    pub num_attention_heads: usize,
    /// Key/value head count; GQA when smaller than the query head count.
    pub num_key_value_heads: usize,
    /// Per-head width.
    pub head_dim: usize,
    /// RMSNorm epsilon for the layer norms.
    pub rms_norm_eps: f32,
    /// RMSNorm epsilon for QK-Norm.
    pub qk_norm_eps: f32,
}

impl Default for TalkerConfig {
    fn default() -> Self {
        Self {
            hidden_size: 1024,
            intermediate_size: 3072,
            num_attention_heads: 16,
            num_key_value_heads: 8,
            head_dim: 128,
            rms_norm_eps: 1e-6,
            qk_norm_eps: 1e-6,
        }
    }
}

impl TalkerConfig {
    /// Total width of the query projection: `num_attention_heads * head_dim`.
    #[must_use]
    pub const fn query_width(&self) -> usize {
        self.num_attention_heads * self.head_dim
    }

    /// Total width of each key/value projection.
    #[must_use]
    pub const fn kv_width(&self) -> usize {
        self.num_key_value_heads * self.head_dim
    }

    /// How many query heads share one KV head.
    #[must_use]
    pub const fn kv_group(&self) -> usize {
        self.num_attention_heads / self.num_key_value_heads
    }
}

/// One layer's weights, borrowed. Every matrix is `[out, in]` as `nn.Linear` stores it.
#[derive(Clone, Copy, Debug)]
pub struct TalkerLayerWeights<'a> {
    /// Pre-attention RMSNorm weight, `[hidden]`.
    pub input_layernorm: &'a [f32],
    /// Query projection, `[query_width, hidden]`.
    pub q_proj: &'a [f32],
    /// Key projection, `[kv_width, hidden]`.
    pub k_proj: &'a [f32],
    /// Value projection, `[kv_width, hidden]`.
    pub v_proj: &'a [f32],
    /// QK-Norm weight for queries, `[head_dim]`.
    pub q_norm: &'a [f32],
    /// QK-Norm weight for keys, `[head_dim]`.
    pub k_norm: &'a [f32],
    /// Output projection, `[hidden, query_width]`.
    pub o_proj: &'a [f32],
    /// Post-attention RMSNorm weight, `[hidden]`.
    pub post_attention_layernorm: &'a [f32],
    /// SwiGLU gate projection, `[intermediate, hidden]`.
    pub gate_proj: &'a [f32],
    /// SwiGLU up projection, `[intermediate, hidden]`.
    pub up_proj: &'a [f32],
    /// SwiGLU down projection, `[hidden, intermediate]`.
    pub down_proj: &'a [f32],
}

/// Per-layer key/value cache, grown one decode step at a time.
///
/// Stored as `[position][kv_width]` so a decode step appends contiguously; this is the reference
/// layout, chosen for obviousness rather than for locality.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct KvCache {
    keys: Vec<f32>,
    values: Vec<f32>,
    positions: usize,
}

impl KvCache {
    /// An empty cache.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of cached positions.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.positions
    }

    /// Whether nothing is cached yet.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.positions == 0
    }

    /// Drop every cached position, keeping the allocation.
    pub fn clear(&mut self) {
        self.keys.clear();
        self.values.clear();
        self.positions = 0;
    }

    fn append(&mut self, key: &[f32], value: &[f32]) {
        self.keys.extend_from_slice(key);
        self.values.extend_from_slice(value);
        self.positions += 1;
    }
}

/// Rotary tables for the positions being computed, already collapsed to one axis.
///
/// The oracle emits `position_embeddings` as `[3, batch, seq, head_dim]`; collapse the three axes
/// with [`f32ref::mrope_interleave`] before calling, or use [`collapse_mrope`].
#[derive(Clone, Copy, Debug)]
pub struct RotaryRows<'a> {
    /// `[seq, head_dim]`.
    pub cos: &'a [f32],
    /// `[seq, head_dim]`.
    pub sin: &'a [f32],
}

/// Collapse an oracle `[3, seq, head_dim]` rotary tensor into `[seq, head_dim]`.
///
/// The reference takes the first half of each axis, interleaves the three axes per
/// `rope_scaling.interleaved`, then concatenates the result with itself to refill `head_dim`.
///
/// # Panics
///
/// Panics if `axes` rows are not `seq * head_dim`, or `head_dim` is odd.
#[must_use]
pub fn collapse_mrope(
    axes: [&[f32]; 3],
    seq: usize,
    head_dim: usize,
    sections: [usize; 3],
) -> Vec<f32> {
    assert!(head_dim.is_multiple_of(2), "head_dim must be even");
    for axis in axes {
        assert_eq!(axis.len(), seq * head_dim, "axis must be [seq, head_dim]");
    }
    let half = head_dim / 2;
    let mut out = vec![0.0f32; seq * head_dim];
    let mut collapsed = vec![0.0f32; half];
    for position in 0..seq {
        let rows = [
            &axes[0][position * head_dim..position * head_dim + half],
            &axes[1][position * head_dim..position * head_dim + half],
            &axes[2][position * head_dim..position * head_dim + half],
        ];
        f32ref::mrope_interleave(rows, sections, &mut collapsed);
        let base = position * head_dim;
        out[base..base + half].copy_from_slice(&collapsed);
        out[base + half..base + head_dim].copy_from_slice(&collapsed);
    }
    out
}

/// Run one decoder layer over `seq` positions, appending to `cache`.
///
/// `hidden` is `[seq, hidden_size]` and is updated in place. `mask` is the additive attention mask
/// as the reference builds it, `[seq, cache_len + seq]` — already containing `-inf` where a query
/// may not attend, so causality and left-padding are expressed entirely by the caller.
///
/// # Panics
///
/// Panics if any slice length disagrees with `config` and `seq`.
pub fn forward_layer(
    config: &TalkerConfig,
    weights: &TalkerLayerWeights<'_>,
    rotary: RotaryRows<'_>,
    mask: &[f32],
    hidden: &mut [f32],
    seq: usize,
    cache: &mut KvCache,
) {
    forward_layer_with_linear_accumulation(
        config,
        weights,
        rotary,
        mask,
        hidden,
        seq,
        cache,
        F32LinearAccumulation::Scalar,
    );
}

/// Run one decoder layer with an explicitly selected f32 projection reduction order.
///
/// [`forward_layer`] is the normal scalar reference path. This entry point makes the reduction
/// order observable to the CPU-fp32 layer fixture without changing that path.
#[allow(clippy::too_many_arguments)]
pub fn forward_layer_with_linear_accumulation(
    config: &TalkerConfig,
    weights: &TalkerLayerWeights<'_>,
    rotary: RotaryRows<'_>,
    mask: &[f32],
    hidden: &mut [f32],
    seq: usize,
    cache: &mut KvCache,
    accumulation: F32LinearAccumulation,
) {
    forward_layer_with_arithmetic(
        config,
        weights,
        rotary,
        mask,
        hidden,
        seq,
        cache,
        accumulation,
        F32RmsNormArithmetic::ScalarReciprocalSqrt,
        F32SiluArithmetic::Divide,
        F32SoftmaxArithmetic::ReciprocalMultiply,
        F32LinearAccumulation::Scalar,
    );
}

/// Run one decoder layer with explicitly selected f32 projection and RMSNorm arithmetic.
///
/// [`forward_layer`] is the normal scalar reference path. This entry point is used only to
/// identify a CPU-fp32 arithmetic-order mismatch at a fixture seam.
#[allow(clippy::too_many_arguments)]
pub fn forward_layer_with_arithmetic(
    config: &TalkerConfig,
    weights: &TalkerLayerWeights<'_>,
    rotary: RotaryRows<'_>,
    mask: &[f32],
    hidden: &mut [f32],
    seq: usize,
    cache: &mut KvCache,
    accumulation: F32LinearAccumulation,
    rms_arithmetic: F32RmsNormArithmetic,
    silu_arithmetic: F32SiluArithmetic,
    softmax_arithmetic: F32SoftmaxArithmetic,
    attention_accumulation: F32LinearAccumulation,
) {
    let hidden_size = config.hidden_size;
    let head_dim = config.head_dim;
    let query_width = config.query_width();
    let kv_width = config.kv_width();
    assert_eq!(
        hidden.len(),
        seq * hidden_size,
        "hidden must be [seq, hidden]"
    );
    assert_eq!(
        rotary.cos.len(),
        seq * head_dim,
        "cos must be [seq, head_dim]"
    );
    assert_eq!(
        rotary.sin.len(),
        seq * head_dim,
        "sin must be [seq, head_dim]"
    );

    let past = cache.len();
    let total = past + seq;
    assert_eq!(mask.len(), seq * total, "mask must be [seq, past + seq]");

    // ── Attention block ────────────────────────────────────────────────────────────────────────
    let mut normed = vec![0.0f32; seq * hidden_size];
    f32ref::rms_norm_with_arithmetic(
        hidden,
        weights.input_layernorm,
        config.rms_norm_eps,
        seq,
        hidden_size,
        rms_arithmetic,
        &mut normed,
    );

    let mut queries = vec![0.0f32; seq * query_width];
    let mut keys = vec![0.0f32; seq * kv_width];
    let mut values = vec![0.0f32; seq * kv_width];
    f32ref::linear_with_accumulation(
        &normed,
        weights.q_proj,
        None,
        seq,
        hidden_size,
        query_width,
        accumulation,
        &mut queries,
    );
    f32ref::linear_with_accumulation(
        &normed,
        weights.k_proj,
        None,
        seq,
        hidden_size,
        kv_width,
        accumulation,
        &mut keys,
    );
    f32ref::linear_with_accumulation(
        &normed,
        weights.v_proj,
        None,
        seq,
        hidden_size,
        kv_width,
        accumulation,
        &mut values,
    );

    // QK-Norm over head_dim, then rotary — in that order. Values are neither normed nor rotated.
    let mut scratch = vec![0.0f32; head_dim];
    for position in 0..seq {
        let cos = &rotary.cos[position * head_dim..position * head_dim + head_dim];
        let sin = &rotary.sin[position * head_dim..position * head_dim + head_dim];
        for head in 0..config.num_attention_heads {
            let offset = position * query_width + head * head_dim;
            let row = &mut queries[offset..offset + head_dim];
            f32ref::rms_norm_with_arithmetic(
                row,
                weights.q_norm,
                config.qk_norm_eps,
                1,
                head_dim,
                rms_arithmetic,
                &mut scratch,
            );
            row.copy_from_slice(&scratch);
            f32ref::apply_rope_in_place(row, cos, sin);
        }
        for head in 0..config.num_key_value_heads {
            let offset = position * kv_width + head * head_dim;
            let row = &mut keys[offset..offset + head_dim];
            f32ref::rms_norm_with_arithmetic(
                row,
                weights.k_norm,
                config.qk_norm_eps,
                1,
                head_dim,
                rms_arithmetic,
                &mut scratch,
            );
            row.copy_from_slice(&scratch);
            f32ref::apply_rope_in_place(row, cos, sin);
        }
    }

    for position in 0..seq {
        cache.append(
            &keys[position * kv_width..position * kv_width + kv_width],
            &values[position * kv_width..position * kv_width + kv_width],
        );
    }

    let mut context = vec![0.0f32; seq * query_width];
    f32ref::gqa_attention_with_arithmetic(
        &queries,
        &cache.keys,
        &cache.values,
        mask,
        seq,
        total,
        config.num_attention_heads,
        config.num_key_value_heads,
        head_dim,
        softmax_arithmetic,
        attention_accumulation,
        &mut context,
    );

    let mut attention = vec![0.0f32; seq * hidden_size];
    f32ref::linear_with_accumulation(
        &context,
        weights.o_proj,
        None,
        seq,
        query_width,
        hidden_size,
        accumulation,
        &mut attention,
    );
    for (state, delta) in hidden.iter_mut().zip(&attention) {
        *state += *delta;
    }

    // ── MLP block ──────────────────────────────────────────────────────────────────────────────
    f32ref::rms_norm_with_arithmetic(
        hidden,
        weights.post_attention_layernorm,
        config.rms_norm_eps,
        seq,
        hidden_size,
        rms_arithmetic,
        &mut normed,
    );

    let intermediate = config.intermediate_size;
    let mut gate = vec![0.0f32; seq * intermediate];
    let mut up = vec![0.0f32; seq * intermediate];
    f32ref::linear_with_accumulation(
        &normed,
        weights.gate_proj,
        None,
        seq,
        hidden_size,
        intermediate,
        accumulation,
        &mut gate,
    );
    f32ref::linear_with_accumulation(
        &normed,
        weights.up_proj,
        None,
        seq,
        hidden_size,
        intermediate,
        accumulation,
        &mut up,
    );
    f32ref::silu_mul_in_place_with_arithmetic(&mut gate, &up, silu_arithmetic);

    let mut down = vec![0.0f32; seq * hidden_size];
    f32ref::linear_with_accumulation(
        &gate,
        weights.down_proj,
        None,
        seq,
        intermediate,
        hidden_size,
        accumulation,
        &mut down,
    );
    for (state, delta) in hidden.iter_mut().zip(&down) {
        *state += *delta;
    }
}

/// Number of identical decoder layers in the pinned 0.6B Base talker.
pub const TALKER_LAYER_COUNT: usize = 28;

/// Primary-code vocabulary width. The residual-code heads are 2,048 wide; this one is not.
pub const PRIMARY_CODE_VOCAB_SIZE: usize = 3_072;

/// Number of acoustic-code embeddings summed into every next-frame talker input.
pub const CODE_GROUP_COUNT: usize = 16;

/// All weights required after input embeddings have been assembled.
///
/// The safe f32 reference deliberately borrows every slice: checkpoint hydration owns the BF16
/// bytes and widens only the tiles an operation consumes. Holding 28 borrowed layer records here
/// avoids an accidental model-wide f32 expansion while still making the full graph executable.
#[derive(Clone, Debug)]
pub struct TalkerWeights<'a> {
    /// The 28 decoder layers in checkpoint order.
    pub layers: Vec<TalkerLayerWeights<'a>>,
    /// Final RMSNorm weight, `[hidden]`.
    pub final_norm: &'a [f32],
    /// Primary-code head, `[3072, hidden]`, bias-free.
    pub codec_head: &'a [f32],
}

/// The 28 independent growing KV caches for a single talker sequence.
#[derive(Clone, Debug, PartialEq)]
pub struct TalkerKvCache {
    layers: Vec<KvCache>,
}

impl TalkerKvCache {
    /// Creates one empty cache for every talker layer.
    #[must_use]
    pub fn new() -> Self {
        Self {
            layers: vec![KvCache::new(); TALKER_LAYER_COUNT],
        }
    }

    /// Clears all layers while retaining their allocations for the next utterance.
    pub fn clear(&mut self) {
        for cache in &mut self.layers {
            cache.clear();
        }
    }

    /// The cached sequence length, after checking that every layer agrees.
    ///
    /// A disagreement is a caller bug: cache state cannot be repaired by dropping a layer's
    /// history without silently changing the autoregressive model.
    #[must_use]
    pub fn len(&self) -> usize {
        let Some((first, rest)) = self.layers.split_first() else {
            return 0;
        };
        assert!(
            rest.iter().all(|cache| cache.len() == first.len()),
            "talker KV layers disagree on their cached sequence length"
        );
        first.len()
    }

    /// Whether no layer contains an autoregressive prefix.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl Default for TalkerKvCache {
    fn default() -> Self {
        Self::new()
    }
}

/// Runs the complete 28-layer talker and its final primary-code head.
///
/// `hidden` is the already assembled `[seq, 1024]` input. It is updated in place to the final
/// normalized hidden state; `logits` receives `[seq, 3072]`. `mask` is the additive causal and
/// left-padding mask for this call, with shape `[seq, cache_len + seq]`.
///
/// This is intentionally the f32 correctness baseline. It has no quantization, packing, or
/// scheduler assumptions, so a future kernel route can be compared against this complete graph.
#[allow(clippy::too_many_arguments)]
pub fn forward_talker(
    config: &TalkerConfig,
    weights: &TalkerWeights<'_>,
    rotary: RotaryRows<'_>,
    mask: &[f32],
    hidden: &mut [f32],
    seq: usize,
    cache: &mut TalkerKvCache,
    logits: &mut [f32],
) {
    assert_eq!(
        weights.layers.len(),
        TALKER_LAYER_COUNT,
        "the Base talker has exactly 28 layers"
    );
    assert_eq!(
        cache.layers.len(),
        TALKER_LAYER_COUNT,
        "the cache must contain one entry per talker layer"
    );
    assert_eq!(hidden.len(), seq * config.hidden_size, "hidden shape");
    assert_eq!(
        weights.final_norm.len(),
        config.hidden_size,
        "final norm shape"
    );
    assert_eq!(
        weights.codec_head.len(),
        PRIMARY_CODE_VOCAB_SIZE * config.hidden_size,
        "codec head shape"
    );
    assert_eq!(logits.len(), seq * PRIMARY_CODE_VOCAB_SIZE, "logit shape");

    for (layer, layer_cache) in weights.layers.iter().zip(&mut cache.layers) {
        forward_layer(config, layer, rotary, mask, hidden, seq, layer_cache);
    }

    let mut normalized = vec![0.0f32; hidden.len()];
    f32ref::rms_norm(
        hidden,
        weights.final_norm,
        config.rms_norm_eps,
        seq,
        config.hidden_size,
        &mut normalized,
    );
    hidden.copy_from_slice(&normalized);
    f32ref::linear(
        hidden,
        weights.codec_head,
        None,
        seq,
        config.hidden_size,
        PRIMARY_CODE_VOCAB_SIZE,
        logits,
    );
}

/// One talker layer's seven projection matrices, quantized W8 per-output-channel symmetric.
///
/// This is the Phase-2 staged lever 2a (`FTTS_INT8=1`): only the GEMMs quantize. Norm weights,
/// QK-Norm, rotary, attention arithmetic, residual adds, the final norm, and the primary head
/// all stay f32, per the fixed doctrine recipe. Built once at generator construction from the
/// same borrowed f32 tensors the reference path reads; the f32 path itself is untouched.
#[derive(Clone, Debug)]
pub struct TalkerLayerQuant {
    /// Query projection, `[query_width, hidden]`.
    pub q_proj: QuantizedMatrix,
    /// Key projection, `[kv_width, hidden]`.
    pub k_proj: QuantizedMatrix,
    /// Value projection, `[kv_width, hidden]`.
    pub v_proj: QuantizedMatrix,
    /// Output projection, `[hidden, query_width]`.
    pub o_proj: QuantizedMatrix,
    /// SwiGLU gate projection, `[intermediate, hidden]`.
    pub gate_proj: QuantizedMatrix,
    /// SwiGLU up projection, `[intermediate, hidden]`.
    pub up_proj: QuantizedMatrix,
    /// SwiGLU down projection, `[hidden, intermediate]`.
    pub down_proj: QuantizedMatrix,
}

impl TalkerLayerQuant {
    /// Quantizes one layer's projections with the canonical symmetric Q8 recipe.
    #[must_use]
    pub fn quantize(config: &TalkerConfig, weights: &TalkerLayerWeights<'_>) -> Self {
        let hidden = config.hidden_size;
        let query_width = config.query_width();
        let kv_width = config.kv_width();
        let intermediate = config.intermediate_size;
        Self {
            q_proj: QuantizedMatrix::quantize(weights.q_proj, query_width, hidden),
            k_proj: QuantizedMatrix::quantize(weights.k_proj, kv_width, hidden),
            v_proj: QuantizedMatrix::quantize(weights.v_proj, kv_width, hidden),
            o_proj: QuantizedMatrix::quantize(weights.o_proj, hidden, query_width),
            gate_proj: QuantizedMatrix::quantize(weights.gate_proj, intermediate, hidden),
            up_proj: QuantizedMatrix::quantize(weights.up_proj, intermediate, hidden),
            down_proj: QuantizedMatrix::quantize(weights.down_proj, hidden, intermediate),
        }
    }
}

/// [`forward_layer`] with the seven projections running W8A8 int8 and everything else f32.
///
/// The reference f32 path is structurally unchanged when this function is not called; the
/// kill-switch branch lives at the `forward_talker_q8` call site, not inside the layer.
///
/// # Panics
///
/// Panics on any shape mismatch, exactly as [`forward_layer`] does.
#[allow(clippy::too_many_arguments)]
pub fn forward_layer_q8(
    config: &TalkerConfig,
    weights: &TalkerLayerWeights<'_>,
    quant: &TalkerLayerQuant,
    rotary: RotaryRows<'_>,
    mask: &[f32],
    hidden: &mut [f32],
    seq: usize,
    cache: &mut KvCache,
    tier: Int8Tier,
) {
    let hidden_size = config.hidden_size;
    let head_dim = config.head_dim;
    let query_width = config.query_width();
    let kv_width = config.kv_width();
    assert_eq!(
        hidden.len(),
        seq * hidden_size,
        "hidden must be [seq, hidden]"
    );
    assert_eq!(
        rotary.cos.len(),
        seq * head_dim,
        "cos must be [seq, head_dim]"
    );
    assert_eq!(
        rotary.sin.len(),
        seq * head_dim,
        "sin must be [seq, head_dim]"
    );
    let past = cache.len();
    let total = past + seq;
    assert_eq!(mask.len(), seq * total, "mask must be [seq, past + seq]");

    // ── Attention block ─────────────────────────────────────────────────────────────────────
    let mut normed = vec![0.0f32; seq * hidden_size];
    f32ref::rms_norm(
        hidden,
        weights.input_layernorm,
        config.rms_norm_eps,
        seq,
        hidden_size,
        &mut normed,
    );

    let mut queries = vec![0.0f32; seq * query_width];
    let mut keys = vec![0.0f32; seq * kv_width];
    let mut values = vec![0.0f32; seq * kv_width];
    linear_q8_dynamic(&normed, &quant.q_proj, None, seq, &mut queries, tier);
    linear_q8_dynamic(&normed, &quant.k_proj, None, seq, &mut keys, tier);
    linear_q8_dynamic(&normed, &quant.v_proj, None, seq, &mut values, tier);

    // QK-Norm over head_dim, then rotary — identical to the f32 reference.
    let mut scratch = vec![0.0f32; head_dim];
    for position in 0..seq {
        let cos = &rotary.cos[position * head_dim..position * head_dim + head_dim];
        let sin = &rotary.sin[position * head_dim..position * head_dim + head_dim];
        for head in 0..config.num_attention_heads {
            let offset = position * query_width + head * head_dim;
            let row = &mut queries[offset..offset + head_dim];
            f32ref::rms_norm(row, weights.q_norm, config.qk_norm_eps, 1, head_dim, &mut scratch);
            row.copy_from_slice(&scratch);
            f32ref::apply_rope_in_place(row, cos, sin);
        }
        for head in 0..config.num_key_value_heads {
            let offset = position * kv_width + head * head_dim;
            let row = &mut keys[offset..offset + head_dim];
            f32ref::rms_norm(row, weights.k_norm, config.qk_norm_eps, 1, head_dim, &mut scratch);
            row.copy_from_slice(&scratch);
            f32ref::apply_rope_in_place(row, cos, sin);
        }
    }

    for position in 0..seq {
        cache.append(
            &keys[position * kv_width..position * kv_width + kv_width],
            &values[position * kv_width..position * kv_width + kv_width],
        );
    }

    let mut context = vec![0.0f32; seq * query_width];
    f32ref::gqa_attention(
        &queries,
        &cache.keys,
        &cache.values,
        mask,
        seq,
        total,
        config.num_attention_heads,
        config.num_key_value_heads,
        head_dim,
        &mut context,
    );

    let mut attention = vec![0.0f32; seq * hidden_size];
    linear_q8_dynamic(&context, &quant.o_proj, None, seq, &mut attention, tier);
    for (state, delta) in hidden.iter_mut().zip(&attention) {
        *state += *delta;
    }

    // ── MLP block ───────────────────────────────────────────────────────────────────────────
    f32ref::rms_norm(
        hidden,
        weights.post_attention_layernorm,
        config.rms_norm_eps,
        seq,
        hidden_size,
        &mut normed,
    );

    let intermediate = config.intermediate_size;
    let mut gate = vec![0.0f32; seq * intermediate];
    let mut up = vec![0.0f32; seq * intermediate];
    linear_q8_dynamic(&normed, &quant.gate_proj, None, seq, &mut gate, tier);
    linear_q8_dynamic(&normed, &quant.up_proj, None, seq, &mut up, tier);
    f32ref::silu_mul_in_place(&mut gate, &up);

    let mut down = vec![0.0f32; seq * hidden_size];
    linear_q8_dynamic(&gate, &quant.down_proj, None, seq, &mut down, tier);
    for (state, delta) in hidden.iter_mut().zip(&down) {
        *state += *delta;
    }
}

/// [`forward_talker`] with every layer's projections running W8A8 int8.
///
/// The final norm and the primary-code head stay f32: sampling-logit boundaries are in the
/// protected high-precision set. Callers arm this path explicitly (the `FTTS_INT8=1`
/// kill-switch in `QwenGenerator`); the f32 route remains the default and is untouched.
///
/// # Panics
///
/// Panics on any geometry mismatch, exactly as [`forward_talker`] does.
#[allow(clippy::too_many_arguments)]
pub fn forward_talker_q8(
    config: &TalkerConfig,
    weights: &TalkerWeights<'_>,
    quant: &[TalkerLayerQuant],
    rotary: RotaryRows<'_>,
    mask: &[f32],
    hidden: &mut [f32],
    seq: usize,
    cache: &mut TalkerKvCache,
    logits: &mut [f32],
    tier: Int8Tier,
) {
    assert_eq!(
        weights.layers.len(),
        TALKER_LAYER_COUNT,
        "the Base talker has exactly 28 layers"
    );
    assert_eq!(
        quant.len(),
        TALKER_LAYER_COUNT,
        "the quantized route needs all 28 layers quantized"
    );
    assert_eq!(hidden.len(), seq * config.hidden_size, "hidden shape");
    assert_eq!(logits.len(), seq * PRIMARY_CODE_VOCAB_SIZE, "logit shape");

    for ((layer, layer_quant), layer_cache) in
        weights.layers.iter().zip(quant).zip(&mut cache.layers)
    {
        forward_layer_q8(
            config, layer, layer_quant, rotary, mask, hidden, seq, layer_cache, tier,
        );
    }

    let mut normalized = vec![0.0f32; hidden.len()];
    f32ref::rms_norm(
        hidden,
        weights.final_norm,
        config.rms_norm_eps,
        seq,
        config.hidden_size,
        &mut normalized,
    );
    hidden.copy_from_slice(&normalized);
    f32ref::linear(
        hidden,
        weights.codec_head,
        None,
        seq,
        config.hidden_size,
        PRIMARY_CODE_VOCAB_SIZE,
        logits,
    );
}

/// The resolved OQ-4 left-padding position rule for one prompt.
///
/// Real elements receive `cumsum(mask) - 1`; pads receive `1`. All three mRoPE axes carry each
/// resulting scalar position, so this function deliberately produces one stream rather than three.
#[must_use]
pub fn left_padded_positions(attention_mask: &[bool]) -> Vec<i64> {
    let mut seen = 0i64;
    attention_mask
        .iter()
        .map(|real| {
            if *real {
                seen += 1;
                seen - 1
            } else {
                1
            }
        })
        .collect()
}

/// `rope_deltas` for a normal left-padded prompt.
#[must_use]
pub fn rope_delta_for_left_padding(attention_mask: &[bool]) -> i64 {
    -(attention_mask.iter().filter(|real| !**real).count() as i64)
}

/// Builds full-width half-split RoPE rows for scalar causal positions.
///
/// `apply_multimodal_rotary_pos_emb` selects mRoPE sections and then uses `rotate_half`; because
/// every axis has the same position for this model, section selection leaves the values unchanged.
/// The returned rows therefore match the source's doubled-half `cos`/`sin` representation exactly.
#[must_use]
pub fn mrope_rows(positions: &[i64], head_dim: usize, theta: f32) -> (Vec<f32>, Vec<f32>) {
    assert!(head_dim.is_multiple_of(2), "head_dim must be even");
    let half = head_dim / 2;
    let mut cos = vec![0.0f32; positions.len() * head_dim];
    let mut sin = vec![0.0f32; positions.len() * head_dim];
    for (row, position) in positions.iter().enumerate() {
        for pair in 0..half {
            let exponent = (2 * pair) as f32 / head_dim as f32;
            let angle = *position as f32 * theta.powf(-exponent);
            let offset = row * head_dim + pair;
            cos[offset] = angle.cos();
            cos[offset + half] = cos[offset];
            sin[offset] = angle.sin();
            sin[offset + half] = sin[offset];
        }
    }
    (cos, sin)
}

/// Applies the learned biased `2048 -> 2048 -> 1024` SiLU text projection.
///
/// Widths are explicit because this f32 reference is also exercised at tiny dimensions in unit
/// tests. The production caller supplies `(input_width, intermediate_width, output_width) =
/// (2048, 2048, 1024)` and validates all four checkpoint tensor shapes before calling it.
/// `input` and `output` are row-major `[rows, width]` buffers.
#[allow(clippy::too_many_arguments)]
pub fn project_text_rows(
    input: &[f32],
    rows: usize,
    input_width: usize,
    intermediate_width: usize,
    output_width: usize,
    fc1_weight: &[f32],
    fc1_bias: &[f32],
    fc2_weight: &[f32],
    fc2_bias: &[f32],
    output: &mut [f32],
) {
    assert_eq!(
        input.len(),
        rows * input_width,
        "text projection input shape"
    );
    assert_eq!(
        fc1_weight.len(),
        intermediate_width * input_width,
        "fc1 weight shape"
    );
    assert_eq!(fc1_bias.len(), intermediate_width, "fc1 bias shape");
    assert_eq!(
        fc2_weight.len(),
        output_width * intermediate_width,
        "fc2 weight shape"
    );
    assert_eq!(fc2_bias.len(), output_width, "fc2 bias shape");
    assert_eq!(
        output.len(),
        rows * output_width,
        "text projection output shape"
    );

    let mut activated = vec![0.0f32; rows * intermediate_width];
    f32ref::linear(
        input,
        fc1_weight,
        Some(fc1_bias),
        rows,
        input_width,
        intermediate_width,
        &mut activated,
    );
    for value in &mut activated {
        *value = *value / (1.0 + (-*value).exp());
    }
    f32ref::linear(
        &activated,
        fc2_weight,
        Some(fc2_bias),
        rows,
        intermediate_width,
        output_width,
        output,
    );
}

/// Forms the next-frame talker input from the prior frame's 16 code embeddings and text stream.
///
/// The text state is consumed one projected hidden per generated frame. When the stream is
/// exhausted, callers pass `None`, which selects `tts_pad_embed`; this is not merely prefill
/// setup, but the normal steady-state path after trailing text ends.
pub fn form_frame_input(
    code_embeddings: &[&[f32]],
    trailing_text_hidden: Option<&[f32]>,
    tts_pad_embed: &[f32],
    output: &mut [f32],
) {
    assert_eq!(
        code_embeddings.len(),
        CODE_GROUP_COUNT,
        "each generated frame has exactly 16 code groups"
    );
    assert_eq!(output.len(), tts_pad_embed.len(), "frame-input width");
    output.copy_from_slice(trailing_text_hidden.unwrap_or(tts_pad_embed));
    for embedding in code_embeddings {
        assert_eq!(embedding.len(), output.len(), "code embedding width");
        for (result, value) in output.iter_mut().zip(*embedding) {
            *result += *value;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Norm, query, key/value, output, and MLP buffers for one probe layer.
    type ProbeWeights = (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>);

    /// Weights that make the layer an identity-ish probe: zeroed projections leave the residual.
    fn zero_weights(config: &TalkerConfig) -> ProbeWeights {
        (
            vec![1.0; config.hidden_size],
            vec![0.0; config.query_width() * config.hidden_size],
            vec![0.0; config.kv_width() * config.hidden_size],
            vec![1.0; config.head_dim],
            vec![0.0; config.intermediate_size * config.hidden_size],
        )
    }

    #[test]
    fn zeroed_projections_leave_the_residual_untouched() {
        // Every projection zero => attention and MLP contribute nothing, so the residual stream is
        // returned unchanged. This pins the residual wiring independently of the math.
        let config = TalkerConfig::default();
        let (norm, q, kv, head_norm, mlp) = zero_weights(&config);
        let o = vec![0.0f32; config.hidden_size * config.query_width()];
        let down = vec![0.0f32; config.hidden_size * config.intermediate_size];
        let weights = TalkerLayerWeights {
            input_layernorm: &norm,
            q_proj: &q,
            k_proj: &kv,
            v_proj: &kv,
            q_norm: &head_norm,
            k_norm: &head_norm,
            o_proj: &o,
            post_attention_layernorm: &norm,
            gate_proj: &mlp,
            up_proj: &mlp,
            down_proj: &down,
        };

        let seq = 3;
        let mut hidden: Vec<f32> = (0..seq * config.hidden_size)
            .map(|i| (i % 7) as f32)
            .collect();
        let original = hidden.clone();
        let cos = vec![1.0f32; seq * config.head_dim];
        let sin = vec![0.0f32; seq * config.head_dim];
        let mut mask = vec![0.0f32; seq * seq];
        for query in 0..seq {
            for key in query + 1..seq {
                mask[query * seq + key] = f32::NEG_INFINITY;
            }
        }
        let mut cache = KvCache::new();
        forward_layer(
            &config,
            &weights,
            RotaryRows {
                cos: &cos,
                sin: &sin,
            },
            &mask,
            &mut hidden,
            seq,
            &mut cache,
        );

        assert_eq!(hidden, original);
        assert_eq!(cache.len(), seq);
    }

    #[test]
    fn prefill_then_decode_grows_the_cache_and_attends_over_it() {
        // A decode step must see every prefilled position: run 4 prefill positions, then one step,
        // and assert the mask width the layer demands is the full cache.
        let config = TalkerConfig::default();
        let (norm, q, kv, head_norm, mlp) = zero_weights(&config);
        let o = vec![0.0f32; config.hidden_size * config.query_width()];
        let down = vec![0.0f32; config.hidden_size * config.intermediate_size];
        let weights = TalkerLayerWeights {
            input_layernorm: &norm,
            q_proj: &q,
            k_proj: &kv,
            v_proj: &kv,
            q_norm: &head_norm,
            k_norm: &head_norm,
            o_proj: &o,
            post_attention_layernorm: &norm,
            gate_proj: &mlp,
            up_proj: &mlp,
            down_proj: &down,
        };

        let mut cache = KvCache::new();
        let prefill = 4;
        let mut hidden = vec![0.5f32; prefill * config.hidden_size];
        let cos = vec![1.0f32; prefill * config.head_dim];
        let sin = vec![0.0f32; prefill * config.head_dim];
        let mut mask = vec![0.0f32; prefill * prefill];
        for query in 0..prefill {
            for key in query + 1..prefill {
                mask[query * prefill + key] = f32::NEG_INFINITY;
            }
        }
        forward_layer(
            &config,
            &weights,
            RotaryRows {
                cos: &cos,
                sin: &sin,
            },
            &mask,
            &mut hidden,
            prefill,
            &mut cache,
        );
        assert_eq!(cache.len(), prefill);

        let mut step = vec![0.25f32; config.hidden_size];
        let step_cos = vec![1.0f32; config.head_dim];
        let step_sin = vec![0.0f32; config.head_dim];
        let step_mask = vec![0.0f32; prefill + 1];
        forward_layer(
            &config,
            &weights,
            RotaryRows {
                cos: &step_cos,
                sin: &step_sin,
            },
            &step_mask,
            &mut step,
            1,
            &mut cache,
        );
        assert_eq!(cache.len(), prefill + 1);

        cache.clear();
        assert!(cache.is_empty());
    }

    #[test]
    fn cached_prefill_is_bit_exact_to_position_by_position_replay() {
        // The usual cache-length checks cannot detect a cache whose values were appended in a
        // different order, or a prefill path that computes a subtly different causal row from
        // decode. Use non-zero attention projections and distinct inputs so the last row really
        // depends on its prefix, then compare every final row, logit, and per-layer KV buffer.
        let config = TalkerConfig {
            hidden_size: 4,
            intermediate_size: 3,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            head_dim: 2,
            rms_norm_eps: 1e-6,
            qk_norm_eps: 1e-6,
        };
        let patterned = |len: usize, scale: f32| {
            (0..len)
                .map(|index| (index as f32 % 5.0 - 2.0) * scale)
                .collect::<Vec<_>>()
        };
        let norm = vec![1.0f32; config.hidden_size];
        let q = patterned(config.query_width() * config.hidden_size, 0.07);
        let k = patterned(config.kv_width() * config.hidden_size, 0.05);
        let v = patterned(config.kv_width() * config.hidden_size, 0.09);
        let head_norm = vec![1.0f32; config.head_dim];
        let o = patterned(config.hidden_size * config.query_width(), 0.06);
        // Keep the MLP inert: this isolates a context-sensitive attention path while still
        // exercising the complete 28-layer cache schedule and primary-code head.
        let mlp = vec![0.0f32; config.intermediate_size * config.hidden_size];
        let down = vec![0.0f32; config.hidden_size * config.intermediate_size];
        let layer = TalkerLayerWeights {
            input_layernorm: &norm,
            q_proj: &q,
            k_proj: &k,
            v_proj: &v,
            q_norm: &head_norm,
            k_norm: &head_norm,
            o_proj: &o,
            post_attention_layernorm: &norm,
            gate_proj: &mlp,
            up_proj: &mlp,
            down_proj: &down,
        };
        let head = patterned(PRIMARY_CODE_VOCAB_SIZE * config.hidden_size, 0.02);
        let weights = TalkerWeights {
            layers: vec![layer; TALKER_LAYER_COUNT],
            final_norm: &norm,
            codec_head: &head,
        };
        let seq = 4;
        let input = vec![
            0.5_f32, -1.0, 2.0, 0.25, 1.5, 0.75, -0.5, 2.5, -0.25, 1.0, 0.5, -2.0, 3.0, -1.5, 0.75,
            0.5,
        ];

        let (batch_cos, batch_sin) = mrope_rows(&[0, 1, 2, 3], config.head_dim, 1.0e6);
        let mut batch_mask = vec![0.0f32; seq * seq];
        for query in 0..seq {
            for key in query + 1..seq {
                batch_mask[query * seq + key] = f32::NEG_INFINITY;
            }
        }
        let mut batch_hidden = input.clone();
        let mut batch_logits = vec![0.0f32; seq * PRIMARY_CODE_VOCAB_SIZE];
        let mut batch_cache = TalkerKvCache::new();
        forward_talker(
            &config,
            &weights,
            RotaryRows {
                cos: &batch_cos,
                sin: &batch_sin,
            },
            &batch_mask,
            &mut batch_hidden,
            seq,
            &mut batch_cache,
            &mut batch_logits,
        );

        let mut replay_hidden = Vec::with_capacity(input.len());
        let mut replay_logits = Vec::with_capacity(batch_logits.len());
        let mut replay_cache = TalkerKvCache::new();
        for position in 0..seq {
            let mut row =
                input[position * config.hidden_size..(position + 1) * config.hidden_size].to_vec();
            let (cos, sin) = mrope_rows(&[position as i64], config.head_dim, 1.0e6);
            let mut logits = vec![0.0f32; PRIMARY_CODE_VOCAB_SIZE];
            let mask = vec![0.0f32; replay_cache.len() + 1];
            forward_talker(
                &config,
                &weights,
                RotaryRows {
                    cos: &cos,
                    sin: &sin,
                },
                &mask,
                &mut row,
                1,
                &mut replay_cache,
                &mut logits,
            );
            replay_hidden.extend_from_slice(&row);
            replay_logits.extend_from_slice(&logits);
        }

        assert_eq!(
            batch_hidden, replay_hidden,
            "prefill and replay hidden states"
        );
        assert_eq!(batch_logits, replay_logits, "prefill and replay logits");
        assert_eq!(batch_cache.len(), seq);
        assert_eq!(batch_cache.len(), replay_cache.len());
        for (layer, (batch, replay)) in batch_cache
            .layers
            .iter()
            .zip(&replay_cache.layers)
            .enumerate()
        {
            assert_eq!(batch.positions, replay.positions, "layer {layer} KV length");
            assert_eq!(batch.keys, replay.keys, "layer {layer} cached keys");
            assert_eq!(batch.values, replay.values, "layer {layer} cached values");
        }

        let mut isolated_last = input[(seq - 1) * config.hidden_size..].to_vec();
        let (cos, sin) = mrope_rows(&[(seq - 1) as i64], config.head_dim, 1.0e6);
        let mut isolated_logits = vec![0.0f32; PRIMARY_CODE_VOCAB_SIZE];
        let mut isolated_cache = TalkerKvCache::new();
        forward_talker(
            &config,
            &weights,
            RotaryRows {
                cos: &cos,
                sin: &sin,
            },
            &[0.0],
            &mut isolated_last,
            1,
            &mut isolated_cache,
            &mut isolated_logits,
        );
        assert_ne!(
            &batch_hidden[(seq - 1) * config.hidden_size..],
            isolated_last,
            "the test must remain context-sensitive rather than passing on an identity cache"
        );
    }

    #[test]
    fn left_padded_prompt_reproduces_the_unpadded_rows_bit_exactly() {
        // The resolved OQ-4 addendum's mandated padded-prompt test: an unpadded-only port passes
        // every fixture and then fails batched/padded production. Left pads receive position 1 and
        // an additive -inf mask column; with `exp(-inf) = 0` contributing exactly nothing to the
        // attention sums, the real rows of a left-padded prompt must be BIT-IDENTICAL to the same
        // prompt run unpadded — any drift here means the mask or the position schedule leaks pad
        // state into real positions.
        let config = TalkerConfig {
            hidden_size: 4,
            intermediate_size: 3,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            head_dim: 2,
            rms_norm_eps: 1e-6,
            qk_norm_eps: 1e-6,
        };
        let patterned = |len: usize, scale: f32| {
            (0..len)
                .map(|index| (index as f32 % 5.0 - 2.0) * scale)
                .collect::<Vec<_>>()
        };
        let norm = vec![1.0f32; config.hidden_size];
        let q = patterned(config.query_width() * config.hidden_size, 0.07);
        let k = patterned(config.kv_width() * config.hidden_size, 0.05);
        let v = patterned(config.kv_width() * config.hidden_size, 0.09);
        let head_norm = vec![1.0f32; config.head_dim];
        let o = patterned(config.hidden_size * config.query_width(), 0.06);
        let mlp = patterned(config.intermediate_size * config.hidden_size, 0.04);
        let down = patterned(config.hidden_size * config.intermediate_size, 0.03);
        let layer = TalkerLayerWeights {
            input_layernorm: &norm,
            q_proj: &q,
            k_proj: &k,
            v_proj: &v,
            q_norm: &head_norm,
            k_norm: &head_norm,
            o_proj: &o,
            post_attention_layernorm: &norm,
            gate_proj: &mlp,
            up_proj: &mlp,
            down_proj: &down,
        };
        let head = patterned(PRIMARY_CODE_VOCAB_SIZE * config.hidden_size, 0.02);
        let weights = TalkerWeights {
            layers: vec![layer; TALKER_LAYER_COUNT],
            final_norm: &norm,
            codec_head: &head,
        };

        let real = 3usize;
        let pads = 2usize;
        let padded_seq = pads + real;
        let real_input = vec![
            0.5_f32, -1.0, 2.0, 0.25, 1.5, 0.75, -0.5, 2.5, -0.25, 1.0, 0.5, -2.0,
        ];

        // Unpadded run: positions 0..real, plain causal mask.
        let attention_flags: Vec<bool> = vec![true; real];
        let unpadded_positions = left_padded_positions(&attention_flags);
        assert_eq!(unpadded_positions, vec![0, 1, 2]);
        let (cos, sin) = mrope_rows(&unpadded_positions, config.head_dim, 1.0e6);
        let mut unpadded_mask = vec![0.0f32; real * real];
        for query in 0..real {
            for key in query + 1..real {
                unpadded_mask[query * real + key] = f32::NEG_INFINITY;
            }
        }
        let mut unpadded_hidden = real_input.clone();
        let mut unpadded_logits = vec![0.0f32; real * PRIMARY_CODE_VOCAB_SIZE];
        let mut unpadded_cache = TalkerKvCache::new();
        forward_talker(
            &config,
            &weights,
            RotaryRows {
                cos: &cos,
                sin: &sin,
            },
            &unpadded_mask,
            &mut unpadded_hidden,
            real,
            &mut unpadded_cache,
            &mut unpadded_logits,
        );

        // Left-padded run: pad rows first, the resolved position rule, and -inf pad columns.
        let padded_flags: Vec<bool> = [vec![false; pads], vec![true; real]].concat();
        let padded_positions = left_padded_positions(&padded_flags);
        assert_eq!(
            padded_positions,
            vec![1, 1, 0, 1, 2],
            "pads take position 1; real elements restart the causal stream at 0"
        );
        assert_eq!(rope_delta_for_left_padding(&padded_flags), -(pads as i64));
        let (padded_cos, padded_sin) = mrope_rows(&padded_positions, config.head_dim, 1.0e6);
        let mut padded_mask = vec![0.0f32; padded_seq * padded_seq];
        for query in 0..padded_seq {
            for key in 0..padded_seq {
                let causal_future = key > query;
                let pad_column = key < pads;
                // Pads may see themselves (softmax needs one finite row entry); real queries
                // must never see a pad.
                if (causal_future || pad_column) && !(query < pads && key == query) {
                    padded_mask[query * padded_seq + key] = f32::NEG_INFINITY;
                }
            }
        }
        let mut padded_hidden = [patterned(pads * config.hidden_size, 0.11), real_input].concat();
        let mut padded_logits = vec![0.0f32; padded_seq * PRIMARY_CODE_VOCAB_SIZE];
        let mut padded_cache = TalkerKvCache::new();
        forward_talker(
            &config,
            &weights,
            RotaryRows {
                cos: &padded_cos,
                sin: &padded_sin,
            },
            &padded_mask,
            &mut padded_hidden,
            padded_seq,
            &mut padded_cache,
            &mut padded_logits,
        );

        assert_eq!(
            &padded_hidden[pads * config.hidden_size..],
            &unpadded_hidden[..],
            "real hidden rows must be bit-identical between padded and unpadded prompts"
        );
        assert_eq!(
            &padded_logits[pads * PRIMARY_CODE_VOCAB_SIZE..],
            &unpadded_logits[..],
            "real logit rows must be bit-identical between padded and unpadded prompts"
        );
    }

    #[test]
    fn collapse_mrope_doubles_the_half_row() {
        // head_dim 8 => half 4; the collapsed half must be repeated to refill head_dim.
        let seq = 2;
        let head_dim = 8;
        let axis: Vec<f32> = (0..seq * head_dim).map(|v| v as f32).collect();
        let out = collapse_mrope([&axis, &axis, &axis], seq, head_dim, [24, 20, 20]);
        assert_eq!(out.len(), seq * head_dim);
        for position in 0..seq {
            let base = position * head_dim;
            assert_eq!(out[base..base + 4], out[base + 4..base + 8]);
            // And the half itself is the axis's own first half at that position.
            assert_eq!(out[base..base + 4], axis[base..base + 4]);
        }
    }

    #[test]
    fn left_padding_positions_and_rope_delta_are_exact() {
        let mask = [false, false, true, true, true];
        assert_eq!(left_padded_positions(&mask), vec![1, 1, 0, 1, 2]);
        assert_eq!(rope_delta_for_left_padding(&mask), -2);

        let (cos, sin) = mrope_rows(&[0, 1], 4, 1_000_000.0);
        assert_eq!(cos.len(), 8);
        assert_eq!(sin.len(), 8);
        // Position zero has identity rotary rows, and every full row repeats its half because
        // the upstream transform applies rotate_half after the doubled-half construction.
        assert_eq!(cos[..4], [1.0; 4]);
        assert_eq!(sin[..4], [0.0; 4]);
        assert_eq!(cos[4], cos[6]);
        assert_eq!(cos[5], cos[7]);
        assert_eq!(sin[4], sin[6]);
        assert_eq!(sin[5], sin[7]);
    }

    #[test]
    fn text_projection_and_per_frame_text_fallback_follow_the_model_order() {
        let input = [1.0f32, -1.0];
        let fc1 = [1.0f32, 0.0, 0.0, 1.0];
        let fc2 = [1.0f32, 1.0];
        let mut projected = [0.0f32; 1];
        project_text_rows(
            &input,
            1,
            2,
            2,
            1,
            &fc1,
            &[0.0, 0.0],
            &fc2,
            &[0.0],
            &mut projected,
        );
        let expected = 1.0 / (1.0 + (-1.0f32).exp()) - 1.0 / (1.0 + 1.0f32.exp());
        assert_eq!(projected, [expected]);

        let code_embedding = [0.25f32, -0.5];
        let code_embeddings = vec![&code_embedding[..]; CODE_GROUP_COUNT];
        let mut frame = [0.0f32; 2];
        form_frame_input(&code_embeddings, None, &[1.0, 2.0], &mut frame);
        assert_eq!(frame, [5.0, -6.0]);
        form_frame_input(&code_embeddings, Some(&[3.0, 4.0]), &[1.0, 2.0], &mut frame);
        assert_eq!(frame, [7.0, -4.0]);
    }

    #[test]
    fn complete_talker_executes_all_28_layers_and_primary_head() {
        // Small geometry keeps this structural test fast while exercising the real 28-layer
        // schedule, per-layer KV ownership, final norm, and 3,072-way primary-code head.
        let config = TalkerConfig {
            hidden_size: 4,
            intermediate_size: 3,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            head_dim: 2,
            rms_norm_eps: 1e-6,
            qk_norm_eps: 1e-6,
        };
        let norm = vec![1.0f32; config.hidden_size];
        let q = vec![0.0f32; config.query_width() * config.hidden_size];
        let kv = vec![0.0f32; config.kv_width() * config.hidden_size];
        let head_norm = vec![1.0f32; config.head_dim];
        let o = vec![0.0f32; config.hidden_size * config.query_width()];
        let mlp = vec![0.0f32; config.intermediate_size * config.hidden_size];
        let down = vec![0.0f32; config.hidden_size * config.intermediate_size];
        let layer = TalkerLayerWeights {
            input_layernorm: &norm,
            q_proj: &q,
            k_proj: &kv,
            v_proj: &kv,
            q_norm: &head_norm,
            k_norm: &head_norm,
            o_proj: &o,
            post_attention_layernorm: &norm,
            gate_proj: &mlp,
            up_proj: &mlp,
            down_proj: &down,
        };
        let head = vec![0.0f32; PRIMARY_CODE_VOCAB_SIZE * config.hidden_size];
        let weights = TalkerWeights {
            layers: vec![layer; TALKER_LAYER_COUNT],
            final_norm: &norm,
            codec_head: &head,
        };
        let mut hidden = vec![1.0f32, 2.0, 3.0, 4.0];
        let (cos, sin) = mrope_rows(&[0], config.head_dim, 1_000_000.0);
        let mut cache = TalkerKvCache::new();
        let mut logits = vec![0.0f32; PRIMARY_CODE_VOCAB_SIZE];

        forward_talker(
            &config,
            &weights,
            RotaryRows {
                cos: &cos,
                sin: &sin,
            },
            &[0.0],
            &mut hidden,
            1,
            &mut cache,
            &mut logits,
        );

        let mean = (1.0f32 + 4.0 + 9.0 + 16.0) / 4.0;
        let scale = (mean + config.rms_norm_eps).sqrt().recip();
        assert_eq!(hidden, vec![scale, 2.0 * scale, 3.0 * scale, 4.0 * scale]);
        assert!(logits.iter().all(|value| *value == 0.0));
        assert_eq!(cache.len(), 1);
    }
}
