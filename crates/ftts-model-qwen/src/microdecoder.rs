//! The Residual-Code Microdecoder: the 15-step sequential loop that produces codes 1..15.
//!
//! This is the authoritative baseline. Every later lever — the hot-packed engine, FrankenMTP's
//! speculative block verify, Q4 depths — must reproduce *this* bit-for-bit under greedy decode.
//! It is written for obviousness, not speed (G1 > G2).
//!
//! # The index map (OQ-5), which is the whole correctness surface
//!
//! One frame is **16 positions**, and getting the head placement off by one still produces
//! plausible audio, so it is pinned here and in the tests:
//!
//! | position | embedding source | scored by | produces |
//! |---|---|---|---|
//! | 0 | the talker's hidden state (conditioning) | **nothing** | — |
//! | 1 | the **talker's** codec embedding (vocab 3072) of `c0` | `lm_head[0]` | `c1` |
//! | 2..=15 | `code_predictor.codec_embedding[p - 2]` (vocab 2048) of `c_{p-1}` | `lm_head[p-1]` | `c_p` |
//!
//! **Two traps.** Position 1 *is* scored — the plan's "one conditioning position plus 15 residual
//! positions" reads as though `c0`'s slot were unscored, and it is not. And position 1's embedding
//! comes from the **talker's** table (vocab 3072), not from a `code_predictor` table (vocab 2048);
//! they are different tables of different widths.
//!
//! # Everything else that is pinned
//!
//! * Mask: plain 16×16 lower-triangular causal. All five layers are `full_attention` with
//!   `sliding_window: null` — there is no sliding window anywhere in this stack.
//! * Rotary: **plain** RoPE, theta 1e6, a 16-position table. This is a *third* rotary table,
//!   distinct from the talker's mRoPE and from the codec's theta-10000 RoPE. The three are never
//!   shared; [`RopeTable`] here is deliberately its own type so they cannot be crossed by accident.
//! * KV state resets at every frame boundary. Frame *N* must not see frame *N-1*'s microdecoder KV.
//! * Each sampled residual conditions the next depth, so the depths are **autoregressively
//!   dependent**. Any "the depths are independent" shortcut is invalid.
//!
//! # What parity may claim
//!
//! Per OQ-5 §6, the training-mode causal forward is equivalent to this loop *in exact arithmetic*
//! only: a seq-16 batched matmul and 15 `m=1` GEMVs can differ in the last ULPs by reduction order.
//! Strict-mode acceptance therefore compares **argmax / token ids**, never logit bits, unless a
//! verify kernel reproduces this loop's reduction order exactly.

use ftts_kernels::int8::{QuantLinearMode, QuantizedMatrix, quant_linear};

/// Number of code groups per frame: one primary code plus 15 residuals.
pub const CODES_PER_FRAME: usize = 16;

/// Positions in one microdecoder frame: conditioning + 15 residual slots.
pub const FRAME_POSITIONS: usize = 16;

/// Residual depths actually predicted by this loop.
pub const RESIDUAL_DEPTHS: usize = 15;

/// Vocabulary of the talker's codec embedding, used at position 1 only.
pub const TALKER_CODEC_VOCAB: usize = 3072;

/// Vocabulary of each per-depth `code_predictor` embedding and head.
pub const RESIDUAL_VOCAB: usize = 2048;

/// Microdecoder geometry, from `code_predictor_config` in the pinned `config.json`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MicrodecoderConfig {
    /// Model width.
    pub hidden_size: usize,
    /// Depth of the tiny transformer.
    pub num_layers: usize,
    /// Query heads.
    pub num_q_heads: usize,
    /// Key/value heads.
    pub num_kv_heads: usize,
    /// Per-head width.
    pub head_dim: usize,
    /// SwiGLU inner width.
    pub intermediate_size: usize,
    /// Rotary base. Plain RoPE, no scaling.
    pub rope_theta: f32,
    /// RMSNorm epsilon, inside the square root.
    pub rms_eps: f32,
}

impl Default for MicrodecoderConfig {
    fn default() -> Self {
        Self {
            hidden_size: 1024,
            num_layers: 5,
            num_q_heads: 16,
            num_kv_heads: 8,
            head_dim: 128,
            intermediate_size: 3072,
            rope_theta: 1.0e6,
            rms_eps: 1e-6,
        }
    }
}

impl MicrodecoderConfig {
    /// Query projection width.
    #[must_use]
    pub const fn q_width(&self) -> usize {
        self.num_q_heads * self.head_dim
    }

    /// Key/value projection width.
    #[must_use]
    pub const fn kv_width(&self) -> usize {
        self.num_kv_heads * self.head_dim
    }

    /// Query heads sharing one KV head.
    #[must_use]
    pub const fn q_per_kv(&self) -> usize {
        self.num_q_heads / self.num_kv_heads
    }
}

/// Which table a position's embedding comes from, and which head scores it.
///
/// Returned by [`position_role`] so the index map is a value the tests can assert on rather than
/// control flow buried in a loop.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PositionRole {
    /// Position 0: the talker's hidden state conditions the frame and is never scored.
    Conditioning,
    /// Position 1: embed `c0` with the **talker's** table, score with `lm_head[0]`.
    PrimaryCodeEmbedding {
        /// Head index that scores this position.
        head: usize,
    },
    /// Positions 2..=15: embed `c_{p-1}` with `codec_embedding[p - 2]`, score with `lm_head[p - 1]`.
    ResidualEmbedding {
        /// Per-depth embedding table index.
        table: usize,
        /// Head index that scores this position.
        head: usize,
    },
}

/// The role of one position in the 16-position frame.
///
/// # Panics
///
/// Panics when `position` is outside the frame.
#[must_use]
pub fn position_role(position: usize) -> PositionRole {
    assert!(
        position < FRAME_POSITIONS,
        "position {position} is outside the {FRAME_POSITIONS}-position frame"
    );
    match position {
        0 => PositionRole::Conditioning,
        // Position 1 IS scored, by head 0, and its embedding is the TALKER's table.
        1 => PositionRole::PrimaryCodeEmbedding { head: 0 },
        p => PositionRole::ResidualEmbedding {
            table: p - 2,
            head: p - 1,
        },
    }
}

/// Root-mean-square normalization, weight-only, epsilon inside the square root.
///
/// # Panics
///
/// Panics when `x` and `weight` differ in length.
#[must_use]
fn rms_norm(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    assert_eq!(x.len(), weight.len(), "rms_norm weight width mismatch");
    let mean_square = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let scale = 1.0 / (mean_square + eps).sqrt();
    x.iter()
        .zip(weight.iter())
        .map(|(v, w)| v * scale * w)
        .collect()
}

/// SiLU: `x * sigmoid(x)`.
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// Row-major `[out, in]` matrix-vector product, matching `nn.Linear.weight`.
///
/// # Panics
///
/// Panics when `weight` is not `out.len() * x.len()` elements.
fn matvec(weight: &[f32], x: &[f32], out: &mut [f32]) {
    assert_eq!(
        weight.len(),
        out.len() * x.len(),
        "matvec shape mismatch: {} for [{}, {}]",
        weight.len(),
        out.len(),
        x.len()
    );
    for (o, slot) in out.iter_mut().enumerate() {
        let row = &weight[o * x.len()..(o + 1) * x.len()];
        *slot = row.iter().zip(x.iter()).map(|(w, v)| w * v).sum();
    }
}

/// Plain RoPE over 16 positions — the microdecoder's own table.
///
/// Deliberately a distinct type from the talker's mRoPE table and the codec's theta-10000 table.
/// Three rotary schemes coexist in this model and crossing them is a silent correctness failure.
#[derive(Clone, Debug)]
pub struct RopeTable {
    head_dim: usize,
    cos: Vec<f32>,
    sin: Vec<f32>,
}

impl RopeTable {
    /// Builds the 16-position table.
    ///
    /// # Panics
    ///
    /// Panics when `head_dim` is odd.
    #[must_use]
    pub fn new(config: &MicrodecoderConfig) -> Self {
        assert!(
            config.head_dim.is_multiple_of(2),
            "head_dim must be even for RoPE"
        );
        let half = config.head_dim / 2;
        let mut cos = vec![0.0_f32; FRAME_POSITIONS * config.head_dim];
        let mut sin = vec![0.0_f32; FRAME_POSITIONS * config.head_dim];
        for position in 0..FRAME_POSITIONS {
            for i in 0..half {
                let freq = 1.0
                    / config
                        .rope_theta
                        .powf(2.0 * i as f32 / config.head_dim as f32);
                let angle = position as f32 * freq;
                let base = position * config.head_dim;
                // `cat(freqs, freqs)`: the second half repeats the first.
                cos[base + i] = angle.cos();
                cos[base + i + half] = angle.cos();
                sin[base + i] = angle.sin();
                sin[base + i + half] = angle.sin();
            }
        }
        Self {
            head_dim: config.head_dim,
            cos,
            sin,
        }
    }

    /// The `cos`/`sin` row for one position, each `head_dim` wide.
    ///
    /// Exposed so a parity test can feed the ORACLE's own `position_embeddings` through the same
    /// layer code, which separates "is our rotary table right" from "is our layer math right".
    ///
    /// # Panics
    ///
    /// Panics when `position` is outside the frame.
    #[must_use]
    pub fn row(&self, position: usize) -> (&[f32], &[f32]) {
        assert!(position < FRAME_POSITIONS, "position outside the frame");
        let span = position * self.head_dim..(position + 1) * self.head_dim;
        (&self.cos[span.clone()], &self.sin[span])
    }

    /// Applies `x = x * cos + rotate_half(x) * sin` for one head at one position.
    ///
    /// # Panics
    ///
    /// Panics when `x` is not one head wide or `position` is outside the frame.
    pub fn apply(&self, x: &mut [f32], position: usize) {
        let (cos, sin) = self.row(position);
        apply_rope_row(x, cos, sin);
    }
}

/// Applies `x = x * cos + rotate_half(x) * sin` for one head, with explicit rotary rows.
///
/// `rotate_half` is the half-split convention (`[-x2, x1]`), not pairwise interleaving.
///
/// # Panics
///
/// Panics when `x`, `cos`, and `sin` do not all have the same even length.
pub fn apply_rope_row(x: &mut [f32], cos: &[f32], sin: &[f32]) {
    assert_eq!(x.len(), cos.len(), "rope cos width mismatch");
    assert_eq!(x.len(), sin.len(), "rope sin width mismatch");
    assert!(x.len().is_multiple_of(2), "head width must be even");
    let half = x.len() / 2;
    let original = x.to_vec();
    for i in 0..half {
        x[i] = original[i] * cos[i] - original[i + half] * sin[i];
        x[i + half] = original[i + half] * cos[i + half] + original[i] * sin[i + half];
    }
}

/// One microdecoder layer's weights, all bias-free.
#[derive(Clone, Copy, Debug)]
pub struct LayerWeights<'a> {
    /// Pre-attention RMSNorm weight, `[hidden]`.
    pub input_norm: &'a [f32],
    /// Query projection, `[q_width, hidden]`.
    pub q_proj: &'a [f32],
    /// Key projection, `[kv_width, hidden]`.
    pub k_proj: &'a [f32],
    /// Value projection, `[kv_width, hidden]`.
    pub v_proj: &'a [f32],
    /// QK-Norm weight for queries, `[head_dim]`.
    pub q_norm: &'a [f32],
    /// QK-Norm weight for keys, `[head_dim]`.
    pub k_norm: &'a [f32],
    /// Output projection, `[hidden, q_width]`.
    pub o_proj: &'a [f32],
    /// Pre-MLP RMSNorm weight, `[hidden]`.
    pub post_attention_norm: &'a [f32],
    /// SwiGLU gate projection, `[intermediate, hidden]`.
    pub gate_proj: &'a [f32],
    /// SwiGLU up projection, `[intermediate, hidden]`.
    pub up_proj: &'a [f32],
    /// SwiGLU down projection, `[hidden, intermediate]`.
    pub down_proj: &'a [f32],
}

/// Per-frame key/value state, head-major, reset at every frame boundary.
#[derive(Clone, Debug)]
pub struct FrameKvState {
    num_kv_heads: usize,
    head_dim: usize,
    len: usize,
    keys: Vec<Vec<f32>>,
    values: Vec<Vec<f32>>,
}

impl FrameKvState {
    /// Allocates state sized for one frame.
    #[must_use]
    pub fn new(config: &MicrodecoderConfig) -> Self {
        let capacity = FRAME_POSITIONS * config.head_dim;
        Self {
            num_kv_heads: config.num_kv_heads,
            head_dim: config.head_dim,
            len: 0,
            keys: (0..config.num_kv_heads)
                .map(|_| Vec::with_capacity(capacity))
                .collect(),
            values: (0..config.num_kv_heads)
                .map(|_| Vec::with_capacity(capacity))
                .collect(),
        }
    }

    /// Cached positions in the current frame.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Whether the frame has no cached positions.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Drops every cached position, keeping allocations.
    ///
    /// **Called at every frame boundary.** Frame *N* seeing frame *N-1*'s keys is a silent
    /// correctness failure that degrades slowly rather than crashing.
    pub fn reset(&mut self) {
        for head in 0..self.num_kv_heads {
            self.keys[head].clear();
            self.values[head].clear();
        }
        self.len = 0;
    }

    fn push(&mut self, key: &[f32], value: &[f32]) {
        let width = self.num_kv_heads * self.head_dim;
        assert_eq!(key.len(), width, "kv key width mismatch");
        assert_eq!(value.len(), width, "kv value width mismatch");
        for head in 0..self.num_kv_heads {
            let span = head * self.head_dim..(head + 1) * self.head_dim;
            self.keys[head].extend_from_slice(&key[span.clone()]);
            self.values[head].extend_from_slice(&value[span]);
        }
        self.len += 1;
    }
}

/// Per-layer KV state for one frame.
///
/// **Each layer keeps its own keys and values.** Sharing one buffer across the five layers is a
/// bug that does not crash: every layer would attend over its predecessors' keys, the cache would
/// grow by `layers * positions`, and the audio would merely be wrong.
/// `the_loop_leaves_exactly_one_frame_cached_per_layer` pins this.
#[derive(Clone, Debug)]
pub struct FrameState {
    layers: Vec<FrameKvState>,
}

impl FrameState {
    /// Allocates one cache per layer.
    #[must_use]
    pub fn new(config: &MicrodecoderConfig) -> Self {
        Self {
            layers: (0..config.num_layers)
                .map(|_| FrameKvState::new(config))
                .collect(),
        }
    }

    /// Clears every layer's cache. Called at each frame boundary.
    pub fn reset(&mut self) {
        for layer in &mut self.layers {
            layer.reset();
        }
    }

    /// Cached positions in one layer.
    #[must_use]
    pub fn len(&self, layer: usize) -> usize {
        self.layers[layer].len()
    }

    /// Whether a layer has nothing cached.
    #[must_use]
    pub fn is_empty(&self, layer: usize) -> bool {
        self.layers[layer].is_empty()
    }

    /// Number of layers this state covers.
    #[must_use]
    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }
}

/// Runs one layer over a single position, appending to the frame's KV state.
///
/// The 16×16 causal mask is expressed structurally: a position attends to the cache as it stands
/// after its own key/value are appended, which is exactly the lower-triangular row.
///
/// # Panics
///
/// Panics on any shape mismatch against `config`.
pub fn layer_step(
    config: &MicrodecoderConfig,
    rope: &RopeTable,
    weights: &LayerWeights<'_>,
    hidden: &[f32],
    position: usize,
    state: &mut FrameKvState,
) -> Vec<f32> {
    let (cos, sin) = rope.row(position);
    layer_step_with_rotary(config, weights, cos, sin, hidden, state)
}

/// [`layer_step`] with the rotary rows supplied explicitly rather than from a [`RopeTable`].
///
/// The position is implied by the state: this element attends to the cache as it stands after its
/// own key/value are appended, which is exactly the causal row.
///
/// # Panics
///
/// Panics on any shape mismatch against `config`.
pub fn layer_step_with_rotary(
    config: &MicrodecoderConfig,
    weights: &LayerWeights<'_>,
    cos: &[f32],
    sin: &[f32],
    hidden: &[f32],
    state: &mut FrameKvState,
) -> Vec<f32> {
    let normed = rms_norm(hidden, weights.input_norm, config.rms_eps);

    let mut q = vec![0.0_f32; config.q_width()];
    let mut k = vec![0.0_f32; config.kv_width()];
    let mut v = vec![0.0_f32; config.kv_width()];
    matvec(weights.q_proj, &normed, &mut q);
    matvec(weights.k_proj, &normed, &mut k);
    matvec(weights.v_proj, &normed, &mut v);

    // QK-Norm per head over head_dim, then plain RoPE. Values are never normalized or rotated.
    for head in 0..config.num_q_heads {
        let span = head * config.head_dim..(head + 1) * config.head_dim;
        let normed_head = rms_norm(&q[span.clone()], weights.q_norm, config.rms_eps);
        q[span.clone()].copy_from_slice(&normed_head);
        apply_rope_row(&mut q[span], cos, sin);
    }
    for head in 0..config.num_kv_heads {
        let span = head * config.head_dim..(head + 1) * config.head_dim;
        let normed_head = rms_norm(&k[span.clone()], weights.k_norm, config.rms_eps);
        k[span.clone()].copy_from_slice(&normed_head);
        apply_rope_row(&mut k[span], cos, sin);
    }

    state.push(&k, &v);

    // One definition of the attention reduction, shared with the q8 and blocked schedules —
    // a drifting private copy here would silently desynchronize the f32 authority from the
    // routes whose bit-identity proofs cite it.
    let mut context = vec![0.0_f32; config.q_width()];
    attend_one_position(config, &q, state, &mut context);

    let mut attn_out = vec![0.0_f32; config.hidden_size];
    matvec(weights.o_proj, &context, &mut attn_out);
    let residual: Vec<f32> = hidden
        .iter()
        .zip(attn_out.iter())
        .map(|(h, a)| h + a)
        .collect();

    let normed = rms_norm(&residual, weights.post_attention_norm, config.rms_eps);
    let mut gate = vec![0.0_f32; config.intermediate_size];
    let mut up = vec![0.0_f32; config.intermediate_size];
    matvec(weights.gate_proj, &normed, &mut gate);
    matvec(weights.up_proj, &normed, &mut up);
    for (g, u) in gate.iter_mut().zip(up.iter()) {
        *g = silu(*g) * u;
    }
    let mut mlp_out = vec![0.0_f32; config.hidden_size];
    matvec(weights.down_proj, &gate, &mut mlp_out);

    residual
        .iter()
        .zip(mlp_out.iter())
        .map(|(r, m)| r + m)
        .collect()
}

/// One microdecoder layer's seven projections, quantized W8 per-output-channel symmetric.
///
/// Phase-2 staged lever 2b: the body GEMMs only. Per-depth embeddings, per-depth heads, all
/// norms, rotary, and attention stay f32 — the heads are a separately gated lever, and the
/// embeddings are gathers, not GEMMs.
#[derive(Clone, Debug)]
pub struct MicroLayerQuant {
    /// Fused Q‖K‖V projection, `[q_width + 2 * kv_width, hidden]` — one dispatch, per-row
    /// scales unchanged, byte-identical to the split form.
    pub qkv: QuantizedMatrix,
    /// Output projection, `[hidden, q_width]`.
    pub o_proj: QuantizedMatrix,
    /// Fused SwiGLU gate‖up projection, `[2 * intermediate, hidden]`.
    pub gate_up: QuantizedMatrix,
    /// SwiGLU down projection, `[hidden, intermediate]`.
    pub down_proj: QuantizedMatrix,
}

impl MicroLayerQuant {
    /// Quantizes one layer's projections with the canonical symmetric Q8 recipe.
    #[must_use]
    pub fn quantize(config: &MicrodecoderConfig, weights: &LayerWeights<'_>) -> Self {
        let hidden = config.hidden_size;
        let q = QuantizedMatrix::quantize(weights.q_proj, config.q_width(), hidden);
        let k = QuantizedMatrix::quantize(weights.k_proj, config.kv_width(), hidden);
        let v = QuantizedMatrix::quantize(weights.v_proj, config.kv_width(), hidden);
        let gate = QuantizedMatrix::quantize(weights.gate_proj, config.intermediate_size, hidden);
        let up = QuantizedMatrix::quantize(weights.up_proj, config.intermediate_size, hidden);
        Self {
            qkv: QuantizedMatrix::concat_rows(&[&q, &k, &v]),
            o_proj: QuantizedMatrix::quantize(weights.o_proj, hidden, config.q_width()),
            gate_up: QuantizedMatrix::concat_rows(&[&gate, &up]),
            down_proj: QuantizedMatrix::quantize(
                weights.down_proj,
                hidden,
                config.intermediate_size,
            ),
        }
    }
}

/// Reusable buffers for [`layer_step_q8`], one set per thread.
///
/// The q8 layer step runs 75 times per frame; without reuse each call makes ~8 short-lived
/// allocations. Buffer reuse changes no arithmetic — every buffer is fully overwritten (or
/// explicitly cleared) before it is read.
struct MicroStepScratch {
    normed: Vec<f32>,
    qkv: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    context: Vec<f32>,
    attn_out: Vec<f32>,
    residual: Vec<f32>,
    gate_up: Vec<f32>,
    mlp_out: Vec<f32>,
    scores: Vec<f32>,
}

impl MicroStepScratch {
    const fn new() -> Self {
        Self {
            normed: Vec::new(),
            qkv: Vec::new(),
            q: Vec::new(),
            k: Vec::new(),
            context: Vec::new(),
            attn_out: Vec::new(),
            residual: Vec::new(),
            gate_up: Vec::new(),
            mlp_out: Vec::new(),
            scores: Vec::new(),
        }
    }
}

thread_local! {
    static MICRO_STEP_SCRATCH: std::cell::RefCell<MicroStepScratch> =
        const { std::cell::RefCell::new(MicroStepScratch::new()) };
}

fn resize_filled(buffer: &mut Vec<f32>, len: usize) {
    buffer.clear();
    buffer.resize(len, 0.0);
}

/// [`layer_step_with_rotary`] with the seven projections running W8A8 int8.
///
/// Norms, QK-Norm, rotary, attention, and residual adds are byte-for-byte the f32 reference's
/// arithmetic; only the projection matmuls differ.
///
/// # Panics
///
/// Panics on any shape mismatch against `config`.
#[allow(clippy::too_many_arguments)]
pub fn layer_step_q8(
    config: &MicrodecoderConfig,
    weights: &LayerWeights<'_>,
    quant: &MicroLayerQuant,
    cos: &[f32],
    sin: &[f32],
    hidden: &[f32],
    state: &mut FrameKvState,
    mode: QuantLinearMode,
) -> Vec<f32> {
    MICRO_STEP_SCRATCH.with(|scratch| {
        let scratch = &mut *scratch.borrow_mut();
        let (q_width, kv_width) = (config.q_width(), config.kv_width());

        resize_filled(&mut scratch.normed, hidden.len());
        rms_norm_into(
            hidden,
            weights.input_norm,
            config.rms_eps,
            &mut scratch.normed,
        );

        // Single position: the fused Q‖K‖V output row splits into the role buffers without a
        // second pass over memory beyond these copies.
        resize_filled(&mut scratch.qkv, q_width + 2 * kv_width);
        quant_linear(mode, &scratch.normed, &quant.qkv, None, 1, &mut scratch.qkv);
        scratch.q.clear();
        scratch.q.extend_from_slice(&scratch.qkv[..q_width]);
        scratch.k.clear();
        scratch
            .k
            .extend_from_slice(&scratch.qkv[q_width..q_width + kv_width]);
        // `v` reuses the tail of `qkv` in place: nothing below mutates it before the push.
        let v = &scratch.qkv[q_width + kv_width..];

        for head in 0..config.num_q_heads {
            let span = head * config.head_dim..(head + 1) * config.head_dim;
            resize_filled(&mut scratch.scores, config.head_dim);
            rms_norm_into(
                &scratch.q[span.clone()],
                weights.q_norm,
                config.rms_eps,
                &mut scratch.scores,
            );
            scratch.q[span.clone()].copy_from_slice(&scratch.scores);
            apply_rope_row(&mut scratch.q[span], cos, sin);
        }
        for head in 0..config.num_kv_heads {
            let span = head * config.head_dim..(head + 1) * config.head_dim;
            resize_filled(&mut scratch.scores, config.head_dim);
            rms_norm_into(
                &scratch.k[span.clone()],
                weights.k_norm,
                config.rms_eps,
                &mut scratch.scores,
            );
            scratch.k[span.clone()].copy_from_slice(&scratch.scores);
            apply_rope_row(&mut scratch.k[span], cos, sin);
        }

        state.push(&scratch.k, v);

        resize_filled(&mut scratch.context, q_width);
        attend_one_position(config, &scratch.q, state, &mut scratch.context);

        resize_filled(&mut scratch.attn_out, config.hidden_size);
        quant_linear(
            mode,
            &scratch.context,
            &quant.o_proj,
            None,
            1,
            &mut scratch.attn_out,
        );
        scratch.residual.clear();
        scratch.residual.extend(
            hidden
                .iter()
                .zip(scratch.attn_out.iter())
                .map(|(h, a)| h + a),
        );

        resize_filled(&mut scratch.normed, scratch.residual.len());
        rms_norm_into(
            &scratch.residual,
            weights.post_attention_norm,
            config.rms_eps,
            &mut scratch.normed,
        );
        resize_filled(&mut scratch.gate_up, 2 * config.intermediate_size);
        quant_linear(
            mode,
            &scratch.normed,
            &quant.gate_up,
            None,
            1,
            &mut scratch.gate_up,
        );
        let (gate, up) = scratch.gate_up.split_at_mut(config.intermediate_size);
        for (g, u) in gate.iter_mut().zip(up.iter()) {
            *g = silu(*g) * u;
        }
        resize_filled(&mut scratch.mlp_out, config.hidden_size);
        let gate = &scratch.gate_up[..config.intermediate_size];
        quant_linear(mode, gate, &quant.down_proj, None, 1, &mut scratch.mlp_out);

        scratch
            .residual
            .iter()
            .zip(scratch.mlp_out.iter())
            .map(|(r, m)| r + m)
            .collect()
    })
}

/// [`rms_norm`] writing into a caller-provided buffer instead of allocating.
///
/// Same arithmetic, same order: sum of squares left to right, one reciprocal square root, then
/// `v * scale * w` per element.
fn rms_norm_into(x: &[f32], weight: &[f32], eps: f32, out: &mut [f32]) {
    assert_eq!(x.len(), weight.len(), "rms_norm weight width mismatch");
    assert_eq!(x.len(), out.len(), "rms_norm output width mismatch");
    let mean_square = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let scale = 1.0 / (mean_square + eps).sqrt();
    for ((slot, v), w) in out.iter_mut().zip(x).zip(weight) {
        *slot = v * scale * w;
    }
}

/// [`layer_step_q8`] over all [`FRAME_POSITIONS`] positions with the four projections hoisted
/// into single `m = FRAME_POSITIONS` dispatches.
///
/// This is the packed seq-16 kernel the FrankenMTP verifier was written against: the sequential
/// decoder reads each layer's 15.7 MB of int8 body weights once per residual depth, fifteen times
/// per frame; this reads them once. Only the *schedule* changes — every projection is the same
/// `quant_linear` over the same `QuantizedMatrix`, and the norms, QK-Norm, rotary, and causal
/// attention still run per position in position order against the same [`FrameKvState`].
///
/// **Bit-identity is by construction, not by measurement.** `linear_q8_dynamic` quantizes each
/// row of `x` independently (its own `quantize_row_q8` scale) and accumulates each output element
/// in i32, so one `m = 16` dispatch is byte-for-byte sixteen `m = 1` dispatches. Batching cannot
/// perturb the arithmetic, which is why this is a pure schedule change rather than a numerics
/// lever. `microdecoder_block_matches_sequential_q8` is the standing proof.
///
/// `hidden` is `[FRAME_POSITIONS, hidden_size]` row-major; the return has the same shape.
///
/// # Panics
///
/// Panics when `hidden` is not exactly `FRAME_POSITIONS * config.hidden_size` long.
#[must_use]
pub fn layer_block_q8(
    config: &MicrodecoderConfig,
    weights: &LayerWeights<'_>,
    quant: &MicroLayerQuant,
    rope: &RopeTable,
    hidden: &[f32],
    mode: QuantLinearMode,
) -> Vec<f32> {
    let width = config.hidden_size;
    assert_eq!(
        hidden.len(),
        FRAME_POSITIONS * width,
        "block input must be [{FRAME_POSITIONS}, {width}]"
    );
    let (q_width, kv_width) = (config.q_width(), config.kv_width());
    let qkv_width = q_width + 2 * kv_width;

    // Hoisted dispatch 1 of 4: Q‖K‖V for every position in one pass over the fused matrix.
    let mut normed = Vec::with_capacity(FRAME_POSITIONS * width);
    for position in 0..FRAME_POSITIONS {
        let row = &hidden[position * width..(position + 1) * width];
        normed.extend_from_slice(&rms_norm(row, weights.input_norm, config.rms_eps));
    }
    let mut qkv = vec![0.0_f32; FRAME_POSITIONS * qkv_width];
    quant_linear(mode, &normed, &quant.qkv, None, FRAME_POSITIONS, &mut qkv);

    // Per-position work: QK-Norm, rotary, and causal attention stay in position order against a
    // single frame cache, so the reduction order the reference defines is preserved exactly.
    let mut state = FrameKvState::new(config);
    let mut context = vec![0.0_f32; FRAME_POSITIONS * q_width];
    for position in 0..FRAME_POSITIONS {
        let row = &qkv[position * qkv_width..(position + 1) * qkv_width];
        let mut q = row[..q_width].to_vec();
        let mut k = row[q_width..q_width + kv_width].to_vec();
        let v = row[q_width + kv_width..].to_vec();
        let (cos, sin) = rope.row(position);

        for head in 0..config.num_q_heads {
            let span = head * config.head_dim..(head + 1) * config.head_dim;
            let normed_head = rms_norm(&q[span.clone()], weights.q_norm, config.rms_eps);
            q[span.clone()].copy_from_slice(&normed_head);
            apply_rope_row(&mut q[span], cos, sin);
        }
        for head in 0..config.num_kv_heads {
            let span = head * config.head_dim..(head + 1) * config.head_dim;
            let normed_head = rms_norm(&k[span.clone()], weights.k_norm, config.rms_eps);
            k[span.clone()].copy_from_slice(&normed_head);
            apply_rope_row(&mut k[span], cos, sin);
        }
        state.push(&k, &v);

        let out = &mut context[position * q_width..(position + 1) * q_width];
        attend_one_position(config, &q, &state, out);
    }

    // Hoisted dispatches 2, 3 and 4: output projection, fused SwiGLU gate‖up, and down.
    let mut attn_out = vec![0.0_f32; FRAME_POSITIONS * width];
    quant_linear(
        mode,
        &context,
        &quant.o_proj,
        None,
        FRAME_POSITIONS,
        &mut attn_out,
    );
    let residual: Vec<f32> = hidden
        .iter()
        .zip(attn_out.iter())
        .map(|(h, a)| h + a)
        .collect();

    let mut normed = Vec::with_capacity(FRAME_POSITIONS * width);
    for position in 0..FRAME_POSITIONS {
        let row = &residual[position * width..(position + 1) * width];
        normed.extend_from_slice(&rms_norm(row, weights.post_attention_norm, config.rms_eps));
    }
    let mut gate_up = vec![0.0_f32; FRAME_POSITIONS * 2 * config.intermediate_size];
    quant_linear(
        mode,
        &normed,
        &quant.gate_up,
        None,
        FRAME_POSITIONS,
        &mut gate_up,
    );

    let mut activated = vec![0.0_f32; FRAME_POSITIONS * config.intermediate_size];
    for position in 0..FRAME_POSITIONS {
        let row = &gate_up[position * 2 * config.intermediate_size
            ..(position + 1) * 2 * config.intermediate_size];
        let (gate, up) = row.split_at(config.intermediate_size);
        let out = &mut activated
            [position * config.intermediate_size..(position + 1) * config.intermediate_size];
        for ((slot, g), u) in out.iter_mut().zip(gate.iter()).zip(up.iter()) {
            *slot = silu(*g) * u;
        }
    }

    let mut mlp_out = vec![0.0_f32; FRAME_POSITIONS * width];
    quant_linear(
        mode,
        &activated,
        &quant.down_proj,
        None,
        FRAME_POSITIONS,
        &mut mlp_out,
    );

    residual
        .iter()
        .zip(mlp_out.iter())
        .map(|(r, m)| r + m)
        .collect()
}

/// Grouped-query softmax attention for one query row against the frame cache.
///
/// Lifted verbatim out of [`layer_step_q8`] so the sequential and blocked schedules share one
/// definition — the reduction order here is load-bearing for their bit-identity.
fn attend_one_position(
    config: &MicrodecoderConfig,
    q: &[f32],
    state: &FrameKvState,
    context: &mut [f32],
) {
    let scale = 1.0 / (config.head_dim as f32).sqrt();
    let positions = state.len();
    context.fill(0.0);
    for head in 0..config.num_q_heads {
        let kv_head = head / config.q_per_kv();
        let query = &q[head * config.head_dim..(head + 1) * config.head_dim];
        let keys = &state.keys[kv_head];
        let values = &state.values[kv_head];

        let mut scores = vec![0.0_f32; positions];
        for (p, score) in scores.iter_mut().enumerate() {
            let key = &keys[p * config.head_dim..(p + 1) * config.head_dim];
            *score = query
                .iter()
                .zip(key.iter())
                .map(|(a, b)| a * b)
                .sum::<f32>()
                * scale;
        }
        let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut total = 0.0_f32;
        for score in &mut scores {
            *score = (*score - max).exp();
            total += *score;
        }
        let out = &mut context[head * config.head_dim..(head + 1) * config.head_dim];
        for (p, weight) in scores.iter().enumerate() {
            let normalized = weight / total;
            let value = &values[p * config.head_dim..(p + 1) * config.head_dim];
            for (slot, v) in out.iter_mut().zip(value.iter()) {
                *slot += normalized * v;
            }
        }
    }
}

/// The armed microdecoder quantization route: body layers, optional heads, and the op mode.
#[derive(Clone, Copy, Debug)]
pub struct MicroQuantRoute<'a> {
    /// One quantized body per layer.
    pub layers: &'a [MicroLayerQuant],
    /// Quantized per-depth heads for the int8+refine scoring path; `None` keeps heads f32.
    pub heads: Option<&'a [QuantizedMatrix]>,
    /// Which quantized linear op class the route runs.
    pub mode: QuantLinearMode,
}

/// Candidates recomputed exactly in f32 after int8 head scoring.
///
/// The production sampler draws from the top 50; 96 leaves a wide margin for int8 rank noise
/// around the cut, so the true f32 top-50 lands inside the refined set in practice (a corpus
/// diagnostic, not a per-frame guarantee — this route only arms with the optimized product
/// path, never under a reference pin).
const HEAD_REFINE_CANDIDATES: usize = 96;

/// Scores one residual head via the int8 kernel, then rebuilds an exact-f32 logit row for the
/// top candidates.
///
/// The skill's "int8 head + top-K refine" structural lever: the 2,048-way head matvec is the
/// microdecoder's second-largest per-step cost after the body, and the sampler only ever looks
/// at the top of the row. Scoring runs W8A8 (through the worker team); the highest
/// [`HEAD_REFINE_CANDIDATES`] logits are then recomputed with the reference's own per-row
/// scalar dot, so every value the sampler can select is bit-identical to the full-f32 row.
/// Unrefined positions are `-inf` and unpickable.
fn score_head_refined(
    head_q8: &QuantizedMatrix,
    head_f32: &[f32],
    normed: &[f32],
    mode: QuantLinearMode,
    logits: &mut [f32],
) {
    let hidden = normed.len();
    let vocab = head_q8.n;
    // The f32 route's matvec asserts its geometry; this route must too, or a truncated or
    // mis-shaped head silently scores a partial vocabulary (tokens past `n` become unreachable
    // with no panic — audio degrades with nothing to point at).
    assert_eq!(
        logits.len(),
        vocab,
        "refined head must cover the full vocabulary"
    );
    assert_eq!(
        head_f32.len(),
        vocab * hidden,
        "refined f32 head must be [vocab, hidden]"
    );
    assert_eq!(
        head_q8.k, hidden,
        "refined q8 head reduction width mismatch"
    );
    let mut coarse = vec![0.0_f32; vocab];
    quant_linear(mode, normed, head_q8, None, 1, &mut coarse);

    // Top-M indices by coarse logit: selection sort over a running boundary is O(vocab * M)
    // with M=96 — cheaper and simpler than sorting 2,048 floats, and allocation-light.
    let mut candidates: Vec<usize> = Vec::with_capacity(HEAD_REFINE_CANDIDATES);
    let mut kept = vec![false; vocab];
    for _ in 0..HEAD_REFINE_CANDIDATES.min(vocab) {
        let mut best: Option<usize> = None;
        for token in 0..vocab {
            if !kept[token] && best.is_none_or(|current| coarse[token] > coarse[current]) {
                best = Some(token);
            }
        }
        let token = best.expect("vocab is non-empty");
        kept[token] = true;
        candidates.push(token);
    }

    logits.fill(f32::NEG_INFINITY);
    for &token in &candidates {
        // The reference matvec's per-row arithmetic: one left-to-right scalar dot. The refined
        // value is therefore exactly the full-f32 row's value at this token.
        let row = &head_f32[token * hidden..(token + 1) * hidden];
        let mut sum = 0.0_f32;
        for index in 0..hidden {
            sum += row[index] * normed[index];
        }
        logits[token] = sum;
    }
}

/// Greedy selection: the lowest index wins a tie, matching `argmax` in the reference.
///
/// # Panics
///
/// Panics on an empty slice — an empty logit row is a wiring bug, not a decode outcome.
#[must_use]
pub fn argmax(logits: &[f32]) -> usize {
    assert!(!logits.is_empty(), "argmax over an empty logit row");
    let mut best = 0_usize;
    for (index, value) in logits.iter().enumerate().skip(1) {
        if *value > logits[best] {
            best = index;
        }
    }
    best
}

/// Everything the 15-step loop needs, borrowed for one frame.
#[derive(Clone, Copy, Debug)]
pub struct MicrodecoderWeights<'a> {
    /// The five layers, in order.
    pub layers: &'a [LayerWeights<'a>],
    /// The **talker's** codec embedding, `[TALKER_CODEC_VOCAB, hidden]`. Position 1 only.
    pub talker_codec_embedding: &'a [f32],
    /// Per-depth embeddings, 14 tables of `[RESIDUAL_VOCAB, hidden]`, for positions 2..=15.
    pub residual_embeddings: &'a [&'a [f32]],
    /// Per-depth heads, 15 tables of `[RESIDUAL_VOCAB, hidden]`.
    pub heads: &'a [&'a [f32]],
    /// Final RMSNorm weight applied before each head, `[hidden]`.
    pub final_norm: &'a [f32],
}

/// Logits from one teacher-forced FrankenMTP block verification pass.
///
/// Rows are ordered by residual depth: row `j` is `lm_head[j]` evaluated at sequence position
/// `j + 1`, and predicts `c{j + 1}`. The supplied `c15` draft id has no input position in a
/// 16-position frame; it is retained so [`accepted_greedy_prefix_len`] can compare all 15 draft
/// ids against their verifier rows.
#[derive(Clone, Debug, PartialEq)]
pub struct DraftVerification {
    logits: Vec<Vec<f32>>,
}

impl DraftVerification {
    /// One 2,048-way logit row per residual depth, ordered `c1..=c15`.
    #[must_use]
    pub fn logits(&self) -> &[Vec<f32>] {
        &self.logits
    }

    /// The strict/greedy verifier ids, one per residual depth.
    #[must_use]
    pub fn greedy_codes(&self) -> Vec<usize> {
        self.logits.iter().map(|row| argmax(row)).collect()
    }

    /// Number of leading draft ids accepted by strict greedy verification.
    ///
    /// Strict FrankenMTP accepts a draft id only when it equals the verifier's argmax at that
    /// depth. A mismatch invalidates the entire remaining suffix because its causal context has
    /// changed.
    ///
    /// # Panics
    ///
    /// Panics unless `drafted_codes` contains all 15 residual ids.
    #[must_use]
    pub fn accepted_greedy_prefix_len(&self, drafted_codes: &[usize]) -> usize {
        assert_eq!(
            drafted_codes.len(),
            RESIDUAL_DEPTHS,
            "expected {RESIDUAL_DEPTHS} drafted residual codes"
        );
        self.logits
            .iter()
            .zip(drafted_codes)
            .take_while(|(logits, drafted)| argmax(logits) == **drafted)
            .count()
    }
}

const TRANSITION_BUCKETS: usize = 64;
const EMPTY_TRANSITION_CODE: u16 = u16::MAX;

/// One bounded counter for the previous-frame residual drafter.
///
/// The draft engine intentionally uses a fixed direct-mapped sketch rather than an unbounded
/// token-pair table. Collisions only make its proposals worse; they cannot alter strict output,
/// because every proposal is checked by [`verify_frame_draft`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct TransitionBucket {
    source: u16,
    target: u16,
    count: u32,
}

impl TransitionBucket {
    const EMPTY: Self = Self {
        source: EMPTY_TRANSITION_CODE,
        target: EMPTY_TRANSITION_CODE,
        count: 0,
    };
}

/// Fixed-memory drafter #1 for greedy FrankenMTP decoding.
///
/// It first proposes the prior frame's residuals. A 64-entry, direct-mapped transition sketch per
/// residual depth learns recurring `previous_frame_code -> next_frame_code` transitions as exact
/// sequential outputs arrive. The sketch is strictly a proposal source: a collision, cold start,
/// or bad guess can only cause verifier rejection and repair.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FrankenMtpDrafter {
    previous_frame: Option<[usize; RESIDUAL_DEPTHS]>,
    transitions: [[TransitionBucket; TRANSITION_BUCKETS]; RESIDUAL_DEPTHS],
}

impl FrankenMtpDrafter {
    /// Starts cold; the first draft is all-zero and is expected to be verified or repaired.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            previous_frame: None,
            transitions: [[TransitionBucket::EMPTY; TRANSITION_BUCKETS]; RESIDUAL_DEPTHS],
        }
    }

    /// Proposes one residual code at each of the fifteen depths without allocating.
    #[must_use]
    pub fn draft(&self) -> [usize; RESIDUAL_DEPTHS] {
        let Some(previous) = self.previous_frame else {
            return [0; RESIDUAL_DEPTHS];
        };
        std::array::from_fn(|depth| {
            let source = previous[depth];
            let bucket = &self.transitions[depth][source % TRANSITION_BUCKETS];
            if usize::from(bucket.source) == source && bucket.count > 0 {
                usize::from(bucket.target)
            } else {
                source
            }
        })
    }

    /// Learns from a repaired or fully verified strict output before the next frame is drafted.
    ///
    /// # Panics
    ///
    /// Panics unless every supplied residual id is in the pinned 2,048-token vocabulary.
    pub fn observe(&mut self, codes: &[usize; RESIDUAL_DEPTHS]) {
        for (depth, &code) in codes.iter().enumerate() {
            assert!(
                code < RESIDUAL_VOCAB,
                "observed c{}={code} is outside residual vocab {RESIDUAL_VOCAB}",
                depth + 1
            );
        }
        if let Some(previous) = self.previous_frame {
            for depth in 0..RESIDUAL_DEPTHS {
                let source = previous[depth];
                let target = codes[depth];
                let bucket = &mut self.transitions[depth][source % TRANSITION_BUCKETS];
                if usize::from(bucket.source) == source && usize::from(bucket.target) == target {
                    bucket.count = bucket.count.saturating_add(1);
                } else if bucket.count == 0 {
                    *bucket = TransitionBucket {
                        source: u16::try_from(source).expect("residual source fits u16"),
                        target: u16::try_from(target).expect("residual target fits u16"),
                        count: 1,
                    };
                } else {
                    bucket.count -= 1;
                }
            }
        }
        self.previous_frame = Some(*codes);
    }
}

impl Default for FrankenMtpDrafter {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of one strict greedy FrankenMTP proposal, verification, and possible repair.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GreedySpeculativeFrame {
    /// The drafter proposal, ordered `c1..=c15`.
    pub drafted_codes: [usize; RESIDUAL_DEPTHS],
    /// The verified strict output, ordered `c1..=c15`.
    pub codes: [usize; RESIDUAL_DEPTHS],
    /// Number of leading proposal ids accepted by the verifier.
    pub accepted_prefix_len: usize,
    /// Whether a rejection required the authoritative sequential repair path.
    pub repaired: bool,
}

/// Runs drafter #1 with strict greedy verification and an exact sequential fallback.
///
/// A complete accepted proposal is returned directly. At the first mismatch the entire
/// authoritative sequential frame is regenerated; this conservative v1 repair never emits a
/// stale verified suffix after the causal context changes. A later suffix-only repair may reuse
/// the accepted prefix, but must preserve these exact output semantics.
///
/// `route` selects the numerics for **both** verification and repair. `Some` runs the packed
/// seq-16 kernel ([`verify_frame_draft_q8`]), which is where the speculation actually pays: the
/// verifier reads each layer's int8 body once for the whole frame instead of once per residual
/// depth. `None` keeps the f32 reference schedule.
///
/// Speculation is a *scheduling* device, never a quality one: whatever the drafter proposes, the
/// emitted codes equal what the same route's sequential greedy decode would have emitted, which
/// `speculation_never_changes_the_emitted_codes` pins for both routes.
///
/// This is greedy-only by construction — acceptance compares against the verifier's argmax. The
/// production sampler (T 0.9, top-k 50) needs the sampled-mode acceptance rule
/// (`frankentts-k-oq19-sampled-rule-w4q`) before speculation can reach the shipping path.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn decode_frame_greedy_speculative(
    config: &MicrodecoderConfig,
    rope: &RopeTable,
    weights: &MicrodecoderWeights<'_>,
    route: Option<&MicroQuantRoute<'_>>,
    state: &mut FrameState,
    drafter: &mut FrankenMtpDrafter,
    talker_hidden: &[f32],
    primary_code: usize,
) -> GreedySpeculativeFrame {
    let drafted_codes = drafter.draft();

    // Verification and repair MUST run the same numerics. Verifying a draft under int8 and then
    // repairing it under f32 would make an accepted frame and a repaired frame come from two
    // different routes, so whether a frame was drafted well would leak into the audio — a
    // speculation mechanism is only sound while it cannot change the output.
    let verification = match route {
        Some(route) => verify_frame_draft_q8(
            config,
            rope,
            weights,
            route,
            talker_hidden,
            primary_code,
            &drafted_codes,
        ),
        None => verify_frame_draft(
            config,
            rope,
            weights,
            talker_hidden,
            primary_code,
            &drafted_codes,
        ),
    };
    let accepted_prefix_len = verification.accepted_greedy_prefix_len(&drafted_codes);
    // Postcondition, uniform across BOTH exits: `state` leaves this function EMPTY. The
    // verifier used its own per-layer caches, so on a full accept the caller's state would
    // otherwise keep the previous frame's entries; and if the repair path left the frame's
    // 16 positions cached while the accept path left none, whether the draft was lucky
    // would leak to the caller through cache state — the exact class of divergence this
    // module exists to make impossible. Every decode entry resets on the way in as well;
    // the reset here is what makes that a guarantee instead of a convention.
    let (codes, repaired) = if accepted_prefix_len == RESIDUAL_DEPTHS {
        state.reset();
        (drafted_codes, false)
    } else {
        let sequential = match route {
            Some(route) => decode_frame_with_selector_q8(
                config,
                rope,
                weights,
                route,
                state,
                talker_hidden,
                primary_code,
                argmax,
            ),
            None => decode_frame_greedy(config, rope, weights, state, talker_hidden, primary_code),
        };
        // Discard the repair pass's cached positions so both exits agree (see above).
        state.reset();
        (
            sequential
                .try_into()
                .expect("sequential microdecoder emits exactly 15 residual codes"),
            true,
        )
    };
    drafter.observe(&codes);
    GreedySpeculativeFrame {
        drafted_codes,
        codes,
        accepted_prefix_len,
        repaired,
    }
}

/// Reads one row of a row-major `[vocab, hidden]` embedding table.
///
/// # Panics
///
/// Panics when `token` is outside the table.
fn embedding_row(table: &[f32], token: usize, hidden: usize) -> Vec<f32> {
    let start = token * hidden;
    assert!(
        start + hidden <= table.len(),
        "token {token} is outside a [{}, {hidden}] embedding table",
        table.len() / hidden
    );
    table[start..start + hidden].to_vec()
}

fn assert_frame_shapes(
    config: &MicrodecoderConfig,
    weights: &MicrodecoderWeights<'_>,
    talker_hidden: &[f32],
) {
    assert_eq!(
        weights.layers.len(),
        config.num_layers,
        "expected {} microdecoder layers",
        config.num_layers
    );
    assert_eq!(
        weights.heads.len(),
        RESIDUAL_DEPTHS,
        "expected {RESIDUAL_DEPTHS} per-depth heads"
    );
    assert_eq!(
        weights.residual_embeddings.len(),
        RESIDUAL_DEPTHS - 1,
        "expected {} per-depth embedding tables (positions 2..=15)",
        RESIDUAL_DEPTHS - 1
    );
    assert_eq!(
        talker_hidden.len(),
        config.hidden_size,
        "talker hidden width mismatch"
    );
}

/// Verifies one drafted residual block through the 16-position causal microdecoder forward.
///
/// This is the scalar reference schedule for the seq-16 FrankenMTP verifier: inputs are prepared
/// for all positions, then each of the five shared body layers consumes positions 0 through 15
/// under a fresh lower-triangular KV cache. It deliberately calls the same scalar layer primitive
/// as [`decode_frame_greedy`], preserving its reduction order while a later packed kernel replaces
/// this implementation. It is therefore a correctness seam, not a performance claim.
///
/// `drafted_codes` is the drafter's proposed `c1..=c15`. Position 0 is the frozen talker hidden
/// state; position 1 embeds `c0` using the talker's table; positions 2 through 15 embed drafted
/// `c1..=c14` through their own residual tables. Every position 1 through 15 is scored by its own
/// residual head. `c15` is not embedded because no position follows it, but its verifier row is
/// still returned and participates in acceptance.
///
/// # Panics
///
/// Panics when the draft is not exactly 15 codes, when weight-table counts differ from the pinned
/// geometry, or on any tensor shape or token-range mismatch.
#[must_use]
pub fn verify_frame_draft(
    config: &MicrodecoderConfig,
    rope: &RopeTable,
    weights: &MicrodecoderWeights<'_>,
    talker_hidden: &[f32],
    primary_code: usize,
    drafted_codes: &[usize],
) -> DraftVerification {
    assert_frame_shapes(config, weights, talker_hidden);
    assert_eq!(
        drafted_codes.len(),
        RESIDUAL_DEPTHS,
        "expected {RESIDUAL_DEPTHS} drafted residual codes"
    );
    for (depth, code) in drafted_codes.iter().enumerate() {
        assert!(
            *code < RESIDUAL_VOCAB,
            "drafted c{}={code} is outside residual vocab {RESIDUAL_VOCAB}",
            depth + 1
        );
    }

    // Flattened [position, hidden] storage is the fixed seq-16 interface the packed verifier
    // shares; `frame_input_block` is the single definition of the position/table index map, so
    // the scalar reference and `verify_frame_draft_q8` can never disagree about layout.
    let mut hidden = frame_input_block(config, weights, talker_hidden, primary_code, drafted_codes);

    // Layer-major order is the training-mode block forward. Each layer owns an independent
    // 16-position causal cache, exactly as in the sequential position-major decoder.
    for layer in weights.layers {
        let mut state = FrameKvState::new(config);
        let mut next = Vec::with_capacity(FRAME_POSITIONS * config.hidden_size);
        for position in 0..FRAME_POSITIONS {
            let start = position * config.hidden_size;
            next.extend_from_slice(&layer_step(
                config,
                rope,
                layer,
                &hidden[start..start + config.hidden_size],
                position,
                &mut state,
            ));
        }
        hidden = next;
    }

    DraftVerification {
        logits: score_frame_heads(config, weights, None, &hidden),
    }
}

/// [`verify_frame_draft`] on the quantized route, with each layer running the packed seq-16
/// kernel instead of a per-position loop.
///
/// This is where FrankenMTP's speedup actually lives. The scalar verifier is a correctness seam:
/// it walks 16 positions through `layer_step`, so it re-reads each layer's weights once per
/// position and costs *more* than the sequential decode it replaces. Routing the five layers
/// through [`layer_block_q8`] collapses that to one pass over the body per frame — the 1.18 GB
/// of per-frame int8 weight traffic the microdecoder spends re-reading its 78.7 MB body fifteen
/// times becomes 78.7 MB read once.
///
/// Head scoring mirrors [`decode_frame_with_selector_q8`] exactly: quantized heads take the
/// int8+top-K-refine path, `None` keeps the reference f32 matvec, so a draft verified here is
/// scored the same way the sequential decoder would score it.
///
/// # Panics
///
/// Panics when the route does not carry every layer, when the draft is not exactly
/// [`RESIDUAL_DEPTHS`] codes, or on any shape or token-range mismatch.
#[must_use]
pub fn verify_frame_draft_q8(
    config: &MicrodecoderConfig,
    rope: &RopeTable,
    weights: &MicrodecoderWeights<'_>,
    route: &MicroQuantRoute<'_>,
    talker_hidden: &[f32],
    primary_code: usize,
    drafted_codes: &[usize],
) -> DraftVerification {
    assert_frame_shapes(config, weights, talker_hidden);
    assert_eq!(
        route.layers.len(),
        config.num_layers,
        "the quantized route needs every microdecoder layer quantized"
    );
    if let Some(heads) = route.heads {
        assert_eq!(
            heads.len(),
            RESIDUAL_DEPTHS,
            "the quantized head route needs every per-depth head quantized"
        );
    }
    assert_eq!(
        drafted_codes.len(),
        RESIDUAL_DEPTHS,
        "expected {RESIDUAL_DEPTHS} drafted residual codes"
    );
    for (depth, code) in drafted_codes.iter().enumerate() {
        assert!(
            *code < RESIDUAL_VOCAB,
            "drafted c{}={code} is outside residual vocab {RESIDUAL_VOCAB}",
            depth + 1
        );
    }

    let mut hidden = frame_input_block(config, weights, talker_hidden, primary_code, drafted_codes);

    // One pass over each layer's body for the whole frame, versus fifteen in the sequential
    // schedule. Layer-major order and the per-layer fresh cache match the scalar verifier.
    for (layer, quant) in weights.layers.iter().zip(route.layers) {
        hidden = layer_block_q8(config, layer, quant, rope, &hidden, route.mode);
    }

    DraftVerification {
        logits: score_frame_heads(
            config,
            weights,
            route.heads.map(|heads| (heads, route.mode)),
            &hidden,
        ),
    }
}

/// Builds the `[FRAME_POSITIONS, hidden]` verifier input: talker state, `c0`, then the draft.
///
/// Shared by both verifier schedules so the position/table index map has exactly one definition.
fn frame_input_block(
    config: &MicrodecoderConfig,
    weights: &MicrodecoderWeights<'_>,
    talker_hidden: &[f32],
    primary_code: usize,
    drafted_codes: &[usize],
) -> Vec<f32> {
    let mut hidden = Vec::with_capacity(FRAME_POSITIONS * config.hidden_size);
    hidden.extend_from_slice(talker_hidden);
    hidden.extend_from_slice(&embedding_row(
        weights.talker_codec_embedding,
        primary_code,
        config.hidden_size,
    ));
    for position in 2..FRAME_POSITIONS {
        let table = match position_role(position) {
            PositionRole::ResidualEmbedding { table, .. } => table,
            role => panic!("position {position} must use a residual embedding, got {role:?}"),
        };
        hidden.extend_from_slice(&embedding_row(
            weights.residual_embeddings[table],
            drafted_codes[position - 2],
            config.hidden_size,
        ));
    }
    hidden
}

/// Scores positions 1..=15 of a finished verifier block through their per-depth heads.
fn score_frame_heads(
    config: &MicrodecoderConfig,
    weights: &MicrodecoderWeights<'_>,
    quant_heads: Option<(&[QuantizedMatrix], QuantLinearMode)>,
    hidden: &[f32],
) -> Vec<Vec<f32>> {
    let mut logits = Vec::with_capacity(RESIDUAL_DEPTHS);
    for position in 1..FRAME_POSITIONS {
        let head = match position_role(position) {
            PositionRole::PrimaryCodeEmbedding { head }
            | PositionRole::ResidualEmbedding { head, .. } => head,
            PositionRole::Conditioning => unreachable!("position 0 is not scored"),
        };
        let start = position * config.hidden_size;
        let normed = rms_norm(
            &hidden[start..start + config.hidden_size],
            weights.final_norm,
            config.rms_eps,
        );
        let mut row = vec![0.0_f32; RESIDUAL_VOCAB];
        match quant_heads {
            Some((heads, mode)) => {
                score_head_refined(&heads[head], weights.heads[head], &normed, mode, &mut row);
            }
            None => matvec(weights.heads[head], &normed, &mut row),
        }
        logits.push(row);
    }
    logits
}

/// Runs the full 15-step sequential loop for one frame under greedy decode.
///
/// `talker_hidden` is the talker's output for this frame and occupies position 0. `primary_code` is
/// `c0`, already sampled by the talker. Returns the 15 residual codes `c1..=c15`.
///
/// The loop is autoregressive by construction: each sampled residual is embedded as the next
/// position's input, so no depth can be computed without its predecessor.
///
/// # Panics
///
/// Panics when the weight tables do not have the expected counts, or on any shape mismatch.
#[must_use]
pub fn decode_frame_greedy(
    config: &MicrodecoderConfig,
    rope: &RopeTable,
    weights: &MicrodecoderWeights<'_>,
    state: &mut FrameState,
    talker_hidden: &[f32],
    primary_code: usize,
) -> Vec<usize> {
    decode_frame_with_selector(
        config,
        rope,
        weights,
        state,
        talker_hidden,
        primary_code,
        argmax,
    )
}

/// The sequential loop with the per-depth code selection supplied by the caller.
///
/// This is how production subtalker SAMPLING runs the same authoritative loop: upstream ships
/// `subtalker_dosample=true` (T 0.9, top-k 50), and forcing greedy residuals under a sampled
/// talker is measurably pathological — the pinned reference in exactly that configuration emits
/// seven silent leading frames before a lucky talker draw escapes the silence attractor
/// (measured 2026-08-08, x-vector mode, seed 0), which is the silence `frankentts-p7r` observed
/// end-to-end. Greedy conformance decode keeps its zero-RNG selector via
/// [`decode_frame_greedy`]; everything else — reset semantics, per-depth tables, autoregressive
/// feedback — is identical by construction.
///
/// # Panics
///
/// Panics when the weight tables do not have the expected counts, or on any shape mismatch.
pub fn decode_frame_with_selector(
    config: &MicrodecoderConfig,
    rope: &RopeTable,
    weights: &MicrodecoderWeights<'_>,
    state: &mut FrameState,
    talker_hidden: &[f32],
    primary_code: usize,
    select: impl FnMut(&[f32]) -> usize,
) -> Vec<usize> {
    decode_frame_with_selector_inner(
        config,
        rope,
        weights,
        None,
        state,
        talker_hidden,
        primary_code,
        select,
    )
}

/// [`decode_frame_with_selector`] with the five body layers running W8A8 int8.
///
/// The per-depth heads and embeddings stay f32. Reset semantics, the index map, and the
/// autoregressive feedback are shared with the reference loop by construction — both routes run
/// the same function body, differing only in the layer-step projection kernel.
///
/// # Panics
///
/// Panics when `quant` does not cover every layer, or as [`decode_frame_with_selector`] does.
#[allow(clippy::too_many_arguments)]
pub fn decode_frame_with_selector_q8(
    config: &MicrodecoderConfig,
    rope: &RopeTable,
    weights: &MicrodecoderWeights<'_>,
    route: &MicroQuantRoute<'_>,
    state: &mut FrameState,
    talker_hidden: &[f32],
    primary_code: usize,
    select: impl FnMut(&[f32]) -> usize,
) -> Vec<usize> {
    assert_eq!(
        route.layers.len(),
        config.num_layers,
        "the quantized route needs every microdecoder layer quantized"
    );
    if let Some(heads) = route.heads {
        assert_eq!(
            heads.len(),
            RESIDUAL_DEPTHS,
            "the quantized head route needs every per-depth head quantized"
        );
    }
    decode_frame_with_selector_inner(
        config,
        rope,
        weights,
        Some(route),
        state,
        talker_hidden,
        primary_code,
        select,
    )
}

#[allow(clippy::too_many_arguments)]
/// `FTTS_SPEC_PROBE=1` measures the speculative-sampling precondition without changing any
/// output: a shadow drafter proposes each frame, and the loop logs the production-sampling
/// probability each draft would have been accepted with (NDJSON on stderr, one line per
/// depth). Bead w4q's go/no-go is decided by this number, per §7.5's warning that a partial
/// accept can cost more than the sequential loop it replaces.
fn spec_probe_enabled() -> bool {
    use std::sync::OnceLock;
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("FTTS_SPEC_PROBE").as_deref() == Ok("1"))
}

thread_local! {
    static PROBE_DRAFTER: std::cell::RefCell<FrankenMtpDrafter> =
        const { std::cell::RefCell::new(FrankenMtpDrafter::new()) };
}

// Matches `layer_step_q8`: the microdecoder's shared entry points carry the model's full context
// (config, rope table, weights, route, cache, conditioning) and bundling them into a struct would
// buy nothing but an extra indirection on the hot path.
#[allow(clippy::too_many_arguments)]
fn decode_frame_with_selector_inner(
    config: &MicrodecoderConfig,
    rope: &RopeTable,
    weights: &MicrodecoderWeights<'_>,
    quant: Option<&MicroQuantRoute<'_>>,
    state: &mut FrameState,
    talker_hidden: &[f32],
    primary_code: usize,
    mut select: impl FnMut(&[f32]) -> usize,
) -> Vec<usize> {
    assert_frame_shapes(config, weights, talker_hidden);

    // Frame boundary: frame N must not see frame N-1's keys.
    state.reset();

    let probe_draft =
        spec_probe_enabled().then(|| PROBE_DRAFTER.with(|drafter| drafter.borrow().draft()));
    let mut codes = Vec::with_capacity(RESIDUAL_DEPTHS);
    let mut previous_code = primary_code;

    for position in 0..FRAME_POSITIONS {
        let role = position_role(position);
        let mut hidden = match role {
            PositionRole::Conditioning => talker_hidden.to_vec(),
            PositionRole::PrimaryCodeEmbedding { .. } => {
                // The TALKER's table (vocab 3072), not a code_predictor table.
                embedding_row(
                    weights.talker_codec_embedding,
                    previous_code,
                    config.hidden_size,
                )
            }
            PositionRole::ResidualEmbedding { table, .. } => embedding_row(
                weights.residual_embeddings[table],
                previous_code,
                config.hidden_size,
            ),
        };

        for (index, layer) in weights.layers.iter().enumerate() {
            hidden = match quant {
                Some(route) => {
                    let (quant_layers, mode) = (route.layers, route.mode);
                    let (cos, sin) = rope.row(position);
                    layer_step_q8(
                        config,
                        layer,
                        &quant_layers[index],
                        cos,
                        sin,
                        &hidden,
                        &mut state.layers[index],
                        mode,
                    )
                }
                None => layer_step(
                    config,
                    rope,
                    layer,
                    &hidden,
                    position,
                    &mut state.layers[index],
                ),
            };
        }

        let head = match role {
            // Position 0 conditions the frame and is never scored.
            PositionRole::Conditioning => continue,
            PositionRole::PrimaryCodeEmbedding { head }
            | PositionRole::ResidualEmbedding { head, .. } => head,
        };

        let normed = rms_norm(&hidden, weights.final_norm, config.rms_eps);
        let mut logits = vec![0.0_f32; RESIDUAL_VOCAB];
        match quant.and_then(|route| route.heads.map(|heads| (heads, route.mode))) {
            Some((heads, mode)) => score_head_refined(
                &heads[head],
                weights.heads[head],
                &normed,
                mode,
                &mut logits,
            ),
            None => matvec(weights.heads[head], &normed, &mut logits),
        }
        if let Some(draft) = &probe_draft {
            let depth = position - 1;
            let acceptance = crate::sampler::production_probability(&logits, draft[depth]);
            eprintln!("{{\"probe\":\"spec\",\"depth\":{depth},\"p_draft\":{acceptance:.6}}}");
        }
        let code = select(&logits);
        assert!(
            code < RESIDUAL_VOCAB,
            "selector returned {code}, outside the residual vocab"
        );
        codes.push(code);
        previous_code = code;
    }

    if probe_draft.is_some()
        && let Ok(frame_codes) = <[usize; RESIDUAL_DEPTHS]>::try_from(codes.as_slice())
    {
        PROBE_DRAFTER.with(|drafter| drafter.borrow_mut().observe(&frame_codes));
    }
    codes
}

#[cfg(test)]
mod tests {
    use super::*;

    fn weights_of(len: usize, seed: u32) -> Vec<f32> {
        let mut state = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
        (0..len)
            .map(|_| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (state >> 8) as f32 / f32::from(u16::MAX) / 256.0 - 0.5
            })
            .collect()
    }

    fn tiny() -> MicrodecoderConfig {
        MicrodecoderConfig {
            hidden_size: 8,
            num_layers: 2,
            num_q_heads: 4,
            num_kv_heads: 2,
            head_dim: 4,
            intermediate_size: 16,
            rope_theta: 1.0e6,
            rms_eps: 1e-6,
        }
    }

    /// Teacher-forced sequential reference: position-major, with each layer retaining its own
    /// causal cache. The block verifier must produce precisely these rows for the same draft.
    fn sequential_draft_logits(
        config: &MicrodecoderConfig,
        rope: &RopeTable,
        weights: &MicrodecoderWeights<'_>,
        talker_hidden: &[f32],
        primary_code: usize,
        drafted_codes: &[usize],
    ) -> Vec<Vec<f32>> {
        assert_eq!(drafted_codes.len(), RESIDUAL_DEPTHS);
        let mut state = FrameState::new(config);
        state.reset();
        let mut logits = Vec::with_capacity(RESIDUAL_DEPTHS);

        for position in 0..FRAME_POSITIONS {
            let role = position_role(position);
            let mut hidden = match role {
                PositionRole::Conditioning => talker_hidden.to_vec(),
                PositionRole::PrimaryCodeEmbedding { .. } => embedding_row(
                    weights.talker_codec_embedding,
                    primary_code,
                    config.hidden_size,
                ),
                PositionRole::ResidualEmbedding { table, .. } => embedding_row(
                    weights.residual_embeddings[table],
                    drafted_codes[position - 2],
                    config.hidden_size,
                ),
            };

            for (index, layer) in weights.layers.iter().enumerate() {
                hidden = layer_step(
                    config,
                    rope,
                    layer,
                    &hidden,
                    position,
                    &mut state.layers[index],
                );
            }

            let head = match role {
                PositionRole::Conditioning => continue,
                PositionRole::PrimaryCodeEmbedding { head }
                | PositionRole::ResidualEmbedding { head, .. } => head,
            };
            let normed = rms_norm(&hidden, weights.final_norm, config.rms_eps);
            let mut row = vec![0.0_f32; RESIDUAL_VOCAB];
            matvec(weights.heads[head], &normed, &mut row);
            logits.push(row);
        }

        logits
    }

    #[test]
    fn pinned_geometry_matches_the_code_predictor_config() {
        let config = MicrodecoderConfig::default();
        assert_eq!(config.num_layers, 5, "the microdecoder is 5 layers");
        assert_eq!(config.q_width(), 2048);
        assert_eq!(config.kv_width(), 1024);
        assert_eq!(CODES_PER_FRAME, 16);
        assert_eq!(RESIDUAL_DEPTHS, 15);
        assert_eq!(
            FRAME_POSITIONS,
            1 + RESIDUAL_DEPTHS,
            "one conditioning position plus 15 residual slots"
        );
    }

    /// TRAP 1: position 1 is scored. An off-by-one in head placement still produces audio.
    #[test]
    fn position_one_is_scored_by_head_zero() {
        assert_eq!(position_role(0), PositionRole::Conditioning);
        assert_eq!(
            position_role(1),
            PositionRole::PrimaryCodeEmbedding { head: 0 },
            "position 1 must be SCORED, by head 0 — it is not a second conditioning slot"
        );
    }

    /// TRAP 2: position 1 embeds through the talker's table, which is a different width.
    #[test]
    fn position_one_uses_the_talker_table_and_the_rest_use_per_depth_tables() {
        assert!(
            matches!(position_role(1), PositionRole::PrimaryCodeEmbedding { .. }),
            "position 1 must not read a code_predictor table"
        );
        assert_ne!(
            TALKER_CODEC_VOCAB, RESIDUAL_VOCAB,
            "the two tables have different vocabularies; conflating them is silent"
        );
        assert_eq!(
            position_role(2),
            PositionRole::ResidualEmbedding { table: 0, head: 1 }
        );
        assert_eq!(
            position_role(15),
            PositionRole::ResidualEmbedding {
                table: 13,
                head: 14
            }
        );
    }

    #[test]
    fn every_residual_depth_is_scored_exactly_once_by_a_distinct_head() {
        let mut heads: Vec<usize> = (0..FRAME_POSITIONS)
            .filter_map(|p| match position_role(p) {
                PositionRole::Conditioning => None,
                PositionRole::PrimaryCodeEmbedding { head }
                | PositionRole::ResidualEmbedding { head, .. } => Some(head),
            })
            .collect();
        assert_eq!(heads.len(), RESIDUAL_DEPTHS, "15 scored positions");
        heads.sort_unstable();
        heads.dedup();
        assert_eq!(
            heads,
            (0..RESIDUAL_DEPTHS).collect::<Vec<_>>(),
            "heads 0..14 each score exactly one position"
        );
    }

    #[test]
    fn embedding_tables_cover_positions_two_through_fifteen_without_gaps() {
        let mut tables: Vec<usize> = (2..FRAME_POSITIONS)
            .map(|p| match position_role(p) {
                PositionRole::ResidualEmbedding { table, .. } => table,
                other => panic!("position {p} should embed per-depth, got {other:?}"),
            })
            .collect();
        tables.sort_unstable();
        assert_eq!(tables, (0..RESIDUAL_DEPTHS - 1).collect::<Vec<_>>());
    }

    #[test]
    fn argmax_takes_the_lowest_index_on_a_tie() {
        assert_eq!(
            argmax(&[0.0, 1.0, 1.0]),
            1,
            "ties resolve to the lower index"
        );
        assert_eq!(argmax(&[3.0, 1.0, 2.0]), 0);
        assert_eq!(argmax(&[-5.0, -1.0]), 1);
    }

    #[test]
    fn rope_is_the_microdecoders_own_sixteen_position_table() {
        let config = MicrodecoderConfig::default();
        let rope = RopeTable::new(&config);
        // Position 0 is the identity.
        let original = weights_of(config.head_dim, 5);
        let mut x = original.clone();
        rope.apply(&mut x, 0);
        for (a, b) in original.iter().zip(x.iter()) {
            assert!((a - b).abs() < 1e-6, "position 0 must not rotate");
        }
        // A later position does rotate, and preserves the norm.
        let mut y = original.clone();
        rope.apply(&mut y, 7);
        let before = original.iter().map(|v| v * v).sum::<f32>();
        let after = y.iter().map(|v| v * v).sum::<f32>();
        assert!((before - after).abs() < 1e-4, "rotation must preserve norm");
        assert!(
            original
                .iter()
                .zip(y.iter())
                .any(|(a, b)| (a - b).abs() > 1e-6),
            "position 7 must actually rotate"
        );
    }

    #[test]
    fn kv_state_resets_at_the_frame_boundary() {
        let config = tiny();
        let mut state = FrameKvState::new(&config);
        state.push(
            &weights_of(config.kv_width(), 1),
            &weights_of(config.kv_width(), 2),
        );
        state.push(
            &weights_of(config.kv_width(), 3),
            &weights_of(config.kv_width(), 4),
        );
        assert_eq!(state.len(), 2);
        state.reset();
        assert!(
            state.is_empty(),
            "frame N must not see frame N-1's microdecoder KV"
        );
    }

    #[test]
    fn a_layer_step_preserves_width_and_grows_the_cache_by_one() {
        let config = tiny();
        let rope = RopeTable::new(&config);
        let owned = TestLayer::new(&config);
        let layer = owned.borrow();
        let mut state = FrameKvState::new(&config);
        let hidden = weights_of(config.hidden_size, 9);
        let out = layer_step(&config, &rope, &layer, &hidden, 0, &mut state);
        assert_eq!(out.len(), config.hidden_size);
        assert_eq!(state.len(), 1);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn the_loop_leaves_exactly_one_frame_cached_per_layer() {
        let config = tiny();
        let rope = RopeTable::new(&config);
        let bundle = TestBundle::new(&config);
        let (layers, embeddings, heads) = bundle.views();
        let weights = bundle.weights(&layers, &embeddings, &heads);
        let mut state = FrameState::new(&config);
        // Pre-dirty layer 0 to prove decode_frame_greedy resets the state itself.
        state.layers[0].push(
            &weights_of(config.kv_width(), 77),
            &weights_of(config.kv_width(), 78),
        );

        let hidden = weights_of(config.hidden_size, 30);
        let codes = decode_frame_greedy(&config, &rope, &weights, &mut state, &hidden, 5);

        assert_eq!(codes.len(), RESIDUAL_DEPTHS, "15 residual codes per frame");
        assert!(codes.iter().all(|c| *c < RESIDUAL_VOCAB));
        assert_eq!(state.layer_count(), config.num_layers);
        for layer in 0..config.num_layers {
            assert_eq!(
                state.len(layer),
                FRAME_POSITIONS,
                "layer {layer} must cache exactly this frame's 16 positions — sharing one \
                 buffer across layers silently multiplies the cache and corrupts attention"
            );
        }
    }

    /// The depths are autoregressively dependent: changing `c0` must be able to change later codes.
    ///
    /// A "depths are independent" shortcut would leave the tail identical.
    #[test]
    fn a_different_primary_code_can_change_later_residuals() {
        let config = tiny();
        let rope = RopeTable::new(&config);
        let bundle = TestBundle::new(&config);
        let (layers, embeddings, heads) = bundle.views();
        let weights = bundle.weights(&layers, &embeddings, &heads);
        let hidden = weights_of(config.hidden_size, 31);

        let mut state_a = FrameState::new(&config);
        let a = decode_frame_greedy(&config, &rope, &weights, &mut state_a, &hidden, 0);
        let mut state_b = FrameState::new(&config);
        let b = decode_frame_greedy(&config, &rope, &weights, &mut state_b, &hidden, 11);

        // The conditioning differs, so the frames must not be forced to agree.
        assert_eq!(a.len(), b.len());
        assert!(
            a != b || bundle.degenerate,
            "c0 must be able to influence the residual stream: {a:?} vs {b:?}"
        );
    }

    #[test]
    fn greedy_decode_is_deterministic() {
        let config = tiny();
        let rope = RopeTable::new(&config);
        let bundle = TestBundle::new(&config);
        let (layers, embeddings, heads) = bundle.views();
        let weights = bundle.weights(&layers, &embeddings, &heads);
        let hidden = weights_of(config.hidden_size, 32);

        let mut first_state = FrameState::new(&config);
        let first = decode_frame_greedy(&config, &rope, &weights, &mut first_state, &hidden, 3);
        let mut second_state = FrameState::new(&config);
        let second = decode_frame_greedy(&config, &rope, &weights, &mut second_state, &hidden, 3);
        assert_eq!(first, second, "greedy decode must be reproducible");
    }

    #[test]
    fn block_verifier_logits_match_teacher_forced_sequential_rows() {
        let config = tiny();
        let rope = RopeTable::new(&config);
        let bundle = TestBundle::new(&config);
        let (layers, embeddings, heads) = bundle.views();
        let weights = bundle.weights(&layers, &embeddings, &heads);
        let talker_hidden = weights_of(config.hidden_size, 33);
        let drafted_codes: Vec<usize> = (0..RESIDUAL_DEPTHS)
            .map(|depth| (depth * 137 + 11) % RESIDUAL_VOCAB)
            .collect();

        let expected =
            sequential_draft_logits(&config, &rope, &weights, &talker_hidden, 17, &drafted_codes);
        let verified =
            verify_frame_draft(&config, &rope, &weights, &talker_hidden, 17, &drafted_codes);

        assert_eq!(
            verified.logits(),
            expected.as_slice(),
            "every verifier row must equal the sequential engine for the same causal prefix"
        );
    }

    #[test]
    fn block_verifier_accepts_the_complete_sequential_greedy_draft() {
        let config = tiny();
        let rope = RopeTable::new(&config);
        let bundle = TestBundle::new(&config);
        let (layers, embeddings, heads) = bundle.views();
        let weights = bundle.weights(&layers, &embeddings, &heads);
        let talker_hidden = weights_of(config.hidden_size, 34);
        let mut state = FrameState::new(&config);
        let sequential =
            decode_frame_greedy(&config, &rope, &weights, &mut state, &talker_hidden, 19);
        let verified =
            verify_frame_draft(&config, &rope, &weights, &talker_hidden, 19, &sequential);

        assert_eq!(verified.greedy_codes(), sequential);
        assert_eq!(
            verified.accepted_greedy_prefix_len(&sequential),
            RESIDUAL_DEPTHS,
            "the exact sequential greedy trajectory must be fully accepted"
        );
    }

    #[test]
    fn block_verifier_rejects_the_suffix_after_its_first_mismatched_draft_id() {
        let config = tiny();
        let rope = RopeTable::new(&config);
        let bundle = TestBundle::new(&config);
        let (layers, embeddings, heads) = bundle.views();
        let weights = bundle.weights(&layers, &embeddings, &heads);
        let talker_hidden = weights_of(config.hidden_size, 35);
        let mut state = FrameState::new(&config);
        let sequential =
            decode_frame_greedy(&config, &rope, &weights, &mut state, &talker_hidden, 23);
        let verified =
            verify_frame_draft(&config, &rope, &weights, &talker_hidden, 23, &sequential);
        let mut mismatched = sequential;
        const FIRST_MISMATCH: usize = 6;
        mismatched[FIRST_MISMATCH] = (mismatched[FIRST_MISMATCH] + 1) % RESIDUAL_VOCAB;

        assert_eq!(
            verified.accepted_greedy_prefix_len(&mismatched),
            FIRST_MISMATCH,
            "a rejected id must invalidate its causal suffix rather than being accepted out of order"
        );
    }

    #[test]
    fn drafter_uses_previous_frame_then_learns_bounded_transitions() {
        let mut drafter = FrankenMtpDrafter::new();
        assert_eq!(drafter.draft(), [0; RESIDUAL_DEPTHS], "cold draft");

        let first = [10; RESIDUAL_DEPTHS];
        let second = [20; RESIDUAL_DEPTHS];
        drafter.observe(&first);
        assert_eq!(drafter.draft(), first, "previous-frame proposal");
        drafter.observe(&second);
        drafter.observe(&first);

        assert_eq!(
            drafter.draft(),
            second,
            "the bounded sketch must prefer its observed 10-to-20 transition over a raw copy"
        );
    }

    #[test]
    fn speculative_greedy_full_accept_matches_the_authoritative_frame() {
        let config = tiny();
        let rope = RopeTable::new(&config);
        let bundle = TestBundle::new(&config);
        let (layers, embeddings, heads) = bundle.views();
        let weights = bundle.weights(&layers, &embeddings, &heads);
        let talker_hidden = weights_of(config.hidden_size, 37);
        let mut sequential_state = FrameState::new(&config);
        let sequential: [usize; RESIDUAL_DEPTHS] = decode_frame_greedy(
            &config,
            &rope,
            &weights,
            &mut sequential_state,
            &talker_hidden,
            31,
        )
        .try_into()
        .expect("sequential decode has fifteen residuals");

        let mut drafter = FrankenMtpDrafter::new();
        drafter.observe(&sequential);
        let mut speculative_state = FrameState::new(&config);
        let result = decode_frame_greedy_speculative(
            &config,
            &rope,
            &weights,
            None,
            &mut speculative_state,
            &mut drafter,
            &talker_hidden,
            31,
        );

        assert_eq!(result.drafted_codes, sequential);
        assert_eq!(result.accepted_prefix_len, RESIDUAL_DEPTHS);
        assert!(!result.repaired, "a fully accepted draft needs no repair");
        assert_eq!(result.codes, sequential, "strict output must be exact");
    }

    #[test]
    fn speculative_greedy_repair_discards_a_stale_rejected_suffix() {
        let config = tiny();
        let rope = RopeTable::new(&config);
        let bundle = TestBundle::new(&config);
        let (layers, embeddings, heads) = bundle.views();
        let weights = bundle.weights(&layers, &embeddings, &heads);
        let talker_hidden = weights_of(config.hidden_size, 38);
        let mut sequential_state = FrameState::new(&config);
        let sequential: [usize; RESIDUAL_DEPTHS] = decode_frame_greedy(
            &config,
            &rope,
            &weights,
            &mut sequential_state,
            &talker_hidden,
            41,
        )
        .try_into()
        .expect("sequential decode has fifteen residuals");

        const FIRST_MISMATCH: usize = 6;
        let mut stale_draft = sequential;
        stale_draft[FIRST_MISMATCH] = (stale_draft[FIRST_MISMATCH] + 1) % RESIDUAL_VOCAB;
        let mut drafter = FrankenMtpDrafter::new();
        drafter.previous_frame = Some(stale_draft);
        let mut speculative_state = FrameState::new(&config);
        let result = decode_frame_greedy_speculative(
            &config,
            &rope,
            &weights,
            None,
            &mut speculative_state,
            &mut drafter,
            &talker_hidden,
            41,
        );

        assert_eq!(result.drafted_codes, stale_draft);
        assert_eq!(result.accepted_prefix_len, FIRST_MISMATCH);
        assert!(result.repaired, "the mismatched draft must enter repair");
        assert_eq!(
            result.codes, sequential,
            "repair must use the authoritative frame"
        );
        assert_ne!(
            result.codes[FIRST_MISMATCH..],
            stale_draft[FIRST_MISMATCH..],
            "the rejected causal suffix must not be emitted from the stale draft"
        );
    }

    #[test]
    #[should_panic(expected = "drafted c15=2048 is outside residual vocab 2048")]
    fn block_verifier_rejects_an_out_of_range_last_draft_id() {
        let config = tiny();
        let rope = RopeTable::new(&config);
        let bundle = TestBundle::new(&config);
        let (layers, embeddings, heads) = bundle.views();
        let weights = bundle.weights(&layers, &embeddings, &heads);
        let talker_hidden = weights_of(config.hidden_size, 36);
        let mut drafted_codes = vec![0; RESIDUAL_DEPTHS];
        drafted_codes[RESIDUAL_DEPTHS - 1] = RESIDUAL_VOCAB;

        let _ = verify_frame_draft(&config, &rope, &weights, &talker_hidden, 29, &drafted_codes);
    }

    struct TestLayer {
        input_norm: Vec<f32>,
        q_proj: Vec<f32>,
        k_proj: Vec<f32>,
        v_proj: Vec<f32>,
        q_norm: Vec<f32>,
        k_norm: Vec<f32>,
        o_proj: Vec<f32>,
        post_attention_norm: Vec<f32>,
        gate_proj: Vec<f32>,
        up_proj: Vec<f32>,
        down_proj: Vec<f32>,
    }

    impl TestLayer {
        fn new(config: &MicrodecoderConfig) -> Self {
            Self {
                input_norm: vec![1.0; config.hidden_size],
                q_proj: weights_of(config.q_width() * config.hidden_size, 1),
                k_proj: weights_of(config.kv_width() * config.hidden_size, 2),
                v_proj: weights_of(config.kv_width() * config.hidden_size, 3),
                q_norm: vec![1.0; config.head_dim],
                k_norm: vec![1.0; config.head_dim],
                o_proj: weights_of(config.hidden_size * config.q_width(), 4),
                post_attention_norm: vec![1.0; config.hidden_size],
                gate_proj: weights_of(config.intermediate_size * config.hidden_size, 5),
                up_proj: weights_of(config.intermediate_size * config.hidden_size, 6),
                down_proj: weights_of(config.hidden_size * config.intermediate_size, 7),
            }
        }

        fn borrow(&self) -> LayerWeights<'_> {
            LayerWeights {
                input_norm: &self.input_norm,
                q_proj: &self.q_proj,
                k_proj: &self.k_proj,
                v_proj: &self.v_proj,
                q_norm: &self.q_norm,
                k_norm: &self.k_norm,
                o_proj: &self.o_proj,
                post_attention_norm: &self.post_attention_norm,
                gate_proj: &self.gate_proj,
                up_proj: &self.up_proj,
                down_proj: &self.down_proj,
            }
        }
    }

    /// The blocked seq-16 schedule must be **byte-identical** to the sequential one, not merely
    /// close. Both run `layer_step_q8`'s arithmetic; the only difference is that
    /// [`layer_block_q8`] hoists the four projections into `m = FRAME_POSITIONS` dispatches,
    /// which reads each layer's int8 body once per frame instead of once per residual depth.
    ///
    /// This holds by construction — `linear_q8_dynamic` derives a separate activation scale per
    /// row and accumulates every output element in i32 — so any failure here is a wiring bug in
    /// the block's row indexing or cache ordering, never a tolerance question.
    #[test]
    fn microdecoder_block_matches_sequential_q8_bit_for_bit() {
        let config = tiny();
        let rope = RopeTable::new(&config);
        let layer = TestLayer::new(&config);
        let borrowed = layer.borrow();
        let quant = MicroLayerQuant::quantize(&config, &borrowed);

        // Every dispatchable tier, not just scalar. NeonSdot is what actually runs on Apple
        // Silicon, so proving only `Int8Tier::Scalar` would leave the shipped route unproven —
        // the same reason `q8_layer_is_bit_identical_across_every_available_tier` sweeps the
        // talker seam.
        for tier in ftts_kernels::int8::Int8Tier::available() {
            let mode = QuantLinearMode::W8A8(tier);
            for seed in 0..6_u32 {
                let hidden = weights_of(FRAME_POSITIONS * config.hidden_size, 4_000 + seed);

                // Sequential: one position at a time through a single retained frame cache.
                let mut state = FrameKvState::new(&config);
                let mut sequential = Vec::with_capacity(FRAME_POSITIONS * config.hidden_size);
                for position in 0..FRAME_POSITIONS {
                    let (cos, sin) = rope.row(position);
                    let row =
                        &hidden[position * config.hidden_size..(position + 1) * config.hidden_size];
                    sequential.extend_from_slice(&layer_step_q8(
                        &config, &borrowed, &quant, cos, sin, row, &mut state, mode,
                    ));
                }

                let blocked = layer_block_q8(&config, &borrowed, &quant, &rope, &hidden, mode);

                assert_eq!(
                    blocked.len(),
                    sequential.len(),
                    "{tier:?}/seed {seed}: block output shape must match the sequential schedule"
                );
                for (index, (block, step)) in blocked.iter().zip(sequential.iter()).enumerate() {
                    assert!(
                        block.to_bits() == step.to_bits(),
                        "{tier:?}/seed {seed}: element {index} diverged — block {block:?} vs \
                         sequential {step:?}; batching must never perturb the arithmetic"
                    );
                }
            }
        }
    }

    /// Speculation is a scheduling device, not a quality one. Whatever the drafter proposes —
    /// a perfect guess, one stale code, or pure garbage — the codes that come out must equal
    /// what the same route's sequential greedy decode emits.
    ///
    /// Run on **both** routes, because the failure this guards against is asymmetric: verifying
    /// under one set of numerics and repairing under another would make the audio depend on how
    /// lucky the drafter got, and only the quantized route can express that mistake.
    #[test]
    fn speculation_never_changes_the_emitted_codes() {
        let config = tiny();
        let rope = RopeTable::new(&config);
        let bundle = TestBundle::new(&config);
        let (layers, embeddings, heads) = bundle.views();
        let weights = bundle.weights(&layers, &embeddings, &heads);
        let quant: Vec<MicroLayerQuant> = layers
            .iter()
            .map(|layer| MicroLayerQuant::quantize(&config, layer))
            .collect();

        for tier in ftts_kernels::int8::Int8Tier::available() {
            let q8 = MicroQuantRoute {
                layers: &quant,
                heads: None,
                mode: QuantLinearMode::W8A8(tier),
            };

            for (label, route) in [("f32", None), ("q8", Some(&q8))] {
                for seed in 0..3_u32 {
                    let talker_hidden = weights_of(config.hidden_size, 880 + seed);
                    let primary_code = (seed as usize * 29) % TALKER_CODEC_VOCAB;

                    // The authoritative answer for this route.
                    let mut reference_state = FrameState::new(&config);
                    let expected: Vec<usize> = match route {
                        Some(route) => decode_frame_with_selector_q8(
                            &config,
                            &rope,
                            &weights,
                            route,
                            &mut reference_state,
                            &talker_hidden,
                            primary_code,
                            argmax,
                        ),
                        None => decode_frame_greedy(
                            &config,
                            &rope,
                            &weights,
                            &mut reference_state,
                            &talker_hidden,
                            primary_code,
                        ),
                    };

                    // Three drafter states: cold (no history), exactly right, and deliberately
                    // wrong — the accept, the full-accept, and the repair paths respectively.
                    let exact: [usize; RESIDUAL_DEPTHS] = expected
                        .clone()
                        .try_into()
                        .expect("greedy decode emits fifteen residuals");
                    let mut wrong = exact;
                    wrong[RESIDUAL_DEPTHS / 2] = (wrong[RESIDUAL_DEPTHS / 2] + 7) % RESIDUAL_VOCAB;

                    let mut cold = FrankenMtpDrafter::new();
                    let mut right = FrankenMtpDrafter::new();
                    right.observe(&exact);
                    let mut stale = FrankenMtpDrafter::new();
                    stale.previous_frame = Some(wrong);

                    for (draft_label, drafter) in [
                        ("cold", &mut cold),
                        ("exact", &mut right),
                        ("stale", &mut stale),
                    ] {
                        let mut state = FrameState::new(&config);
                        let result = decode_frame_greedy_speculative(
                            &config,
                            &rope,
                            &weights,
                            route,
                            &mut state,
                            drafter,
                            &talker_hidden,
                            primary_code,
                        );
                        assert_eq!(
                            result.codes.to_vec(),
                            expected,
                            "{tier:?}/{label}/{draft_label}/seed {seed}: speculation changed the \
                             emitted codes"
                        );
                        assert_eq!(
                            result.repaired,
                            result.accepted_prefix_len != RESIDUAL_DEPTHS,
                            "{tier:?}/{label}/{draft_label}/seed {seed}: repair flag must track \
                             whether the whole draft was accepted"
                        );
                        // Beyond the invariant above, each drafter state must reach the branch it
                        // was built to exercise — otherwise this test could pass while never
                        // running the accept path or never running the repair path at all.
                        match draft_label {
                            "exact" => assert!(
                                !result.repaired && result.accepted_prefix_len == RESIDUAL_DEPTHS,
                                "{tier:?}/{label}: a draft equal to the sequential output must be \
                                 accepted whole, got prefix {} repaired={}",
                                result.accepted_prefix_len,
                                result.repaired
                            ),
                            "stale" => assert!(
                                result.repaired,
                                "{tier:?}/{label}: a draft that differs at depth {} must fall back \
                                 to the authoritative sequential path",
                                RESIDUAL_DEPTHS / 2
                            ),
                            _ => {}
                        }
                    }
                }
            }
        }
    }

    /// The packed verifier must reproduce the scalar verifier's rows exactly when both run the
    /// same quantized body — the schedule is the only difference, and
    /// [`layer_block_q8`]'s bit-identity makes that a construction, not a tolerance.
    ///
    /// Guards the composition the speedup depends on: if this ever drifts, a drafted block would
    /// be accepted or repaired on different evidence than the sequential decoder would produce,
    /// which is a correctness bug and not a speed/quality trade.
    #[test]
    fn packed_verifier_matches_the_scalar_verifier_on_the_same_quantized_body() {
        let config = tiny();
        let rope = RopeTable::new(&config);
        // Distinct weights per layer. Identical layers would let a route that reused layer 0's
        // quantized body for every layer — or paired the two lists in the wrong order — pass a
        // test whose entire job is catching exactly that wiring bug.
        let layers: Vec<TestLayer> = (0..config.num_layers)
            .map(|layer| TestLayer::new_seeded(&config, 500 + layer as u32 * 8))
            .collect();
        let borrowed: Vec<LayerWeights<'_>> = layers.iter().map(TestLayer::borrow).collect();
        let quant: Vec<MicroLayerQuant> = borrowed
            .iter()
            .map(|layer| MicroLayerQuant::quantize(&config, layer))
            .collect();

        let talker_codec = weights_of(TALKER_CODEC_VOCAB * config.hidden_size, 11);
        let residual_tables: Vec<Vec<f32>> = (0..RESIDUAL_DEPTHS - 1)
            .map(|depth| weights_of(RESIDUAL_VOCAB * config.hidden_size, 20 + depth as u32))
            .collect();
        let head_tables: Vec<Vec<f32>> = (0..RESIDUAL_DEPTHS)
            .map(|depth| weights_of(RESIDUAL_VOCAB * config.hidden_size, 40 + depth as u32))
            .collect();
        let residual_refs: Vec<&[f32]> = residual_tables.iter().map(Vec::as_slice).collect();
        let head_refs: Vec<&[f32]> = head_tables.iter().map(Vec::as_slice).collect();
        let final_norm = vec![1.0_f32; config.hidden_size];

        let weights = MicrodecoderWeights {
            layers: &borrowed,
            talker_codec_embedding: &talker_codec,
            residual_embeddings: &residual_refs,
            heads: &head_refs,
            final_norm: &final_norm,
        };
        // Sweep tier × seed. Pinning this to scalar would prove the schedule only on a route the
        // product never dispatches.
        let cases = ftts_kernels::int8::Int8Tier::available()
            .into_iter()
            .flat_map(|tier| (0..4_u32).map(move |seed| (tier, seed)));

        for (tier, seed) in cases {
            let mode = QuantLinearMode::W8A8(tier);
            let route = MicroQuantRoute {
                layers: &quant,
                heads: None,
                mode,
            };
            let talker_hidden = weights_of(config.hidden_size, 700 + seed);
            let drafted: Vec<usize> = (0..RESIDUAL_DEPTHS)
                .map(|depth| (depth * 137 + seed as usize * 31) % RESIDUAL_VOCAB)
                .collect();
            let primary_code = (seed as usize * 17) % TALKER_CODEC_VOCAB;

            // Scalar schedule over the same quantized body, position by position.
            let mut hidden =
                frame_input_block(&config, &weights, &talker_hidden, primary_code, &drafted);
            for (layer, layer_quant) in borrowed.iter().zip(quant.iter()) {
                let mut state = FrameKvState::new(&config);
                let mut next = Vec::with_capacity(FRAME_POSITIONS * config.hidden_size);
                for position in 0..FRAME_POSITIONS {
                    let (cos, sin) = rope.row(position);
                    let row =
                        &hidden[position * config.hidden_size..(position + 1) * config.hidden_size];
                    next.extend_from_slice(&layer_step_q8(
                        &config,
                        layer,
                        layer_quant,
                        cos,
                        sin,
                        row,
                        &mut state,
                        mode,
                    ));
                }
                hidden = next;
            }
            let expected = score_frame_heads(&config, &weights, None, &hidden);

            let packed = verify_frame_draft_q8(
                &config,
                &rope,
                &weights,
                &route,
                &talker_hidden,
                primary_code,
                &drafted,
            );

            assert_eq!(
                packed.logits().len(),
                expected.len(),
                "{tier:?}/seed {seed}: row count"
            );
            for (depth, (fast, slow)) in packed.logits().iter().zip(expected.iter()).enumerate() {
                for (token, (a, b)) in fast.iter().zip(slow.iter()).enumerate() {
                    assert!(
                        a.to_bits() == b.to_bits(),
                        "{tier:?}/seed {seed}: depth {depth} token {token} diverged — packed \
                         {a:?} vs sequential {b:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn q8_block_verifier_accepts_the_sequential_q8_draft_with_refined_heads() {
        // Independent index-map coverage for the QUANTIZED route with head refine armed: the
        // sequential engine builds its inputs one position at a time, so agreement with the
        // packed verifier here cannot come from a shared block-construction helper. The
        // packed-vs-scalar sweep above runs `heads: None`; this is the only test where the
        // q8 head-refine path crosses the block verifier.
        let config = tiny();
        let rope = RopeTable::new(&config);
        let layers: Vec<TestLayer> = (0..config.num_layers)
            .map(|layer| TestLayer::new_seeded(&config, 800 + layer as u32 * 8))
            .collect();
        let borrowed: Vec<LayerWeights<'_>> = layers.iter().map(TestLayer::borrow).collect();
        let quant: Vec<MicroLayerQuant> = borrowed
            .iter()
            .map(|layer| MicroLayerQuant::quantize(&config, layer))
            .collect();
        let talker_codec = weights_of(TALKER_CODEC_VOCAB * config.hidden_size, 61);
        let residual_tables: Vec<Vec<f32>> = (0..RESIDUAL_DEPTHS - 1)
            .map(|depth| weights_of(RESIDUAL_VOCAB * config.hidden_size, 70 + depth as u32))
            .collect();
        let head_tables: Vec<Vec<f32>> = (0..RESIDUAL_DEPTHS)
            .map(|depth| weights_of(RESIDUAL_VOCAB * config.hidden_size, 90 + depth as u32))
            .collect();
        let residual_refs: Vec<&[f32]> = residual_tables.iter().map(Vec::as_slice).collect();
        let head_refs: Vec<&[f32]> = head_tables.iter().map(Vec::as_slice).collect();
        let final_norm = vec![1.0_f32; config.hidden_size];
        let weights = MicrodecoderWeights {
            layers: &borrowed,
            talker_codec_embedding: &talker_codec,
            residual_embeddings: &residual_refs,
            heads: &head_refs,
            final_norm: &final_norm,
        };
        let heads_q8: Vec<ftts_kernels::int8::QuantizedMatrix> = head_tables
            .iter()
            .map(|head| {
                ftts_kernels::int8::QuantizedMatrix::quantize(
                    head,
                    RESIDUAL_VOCAB,
                    config.hidden_size,
                )
            })
            .collect();

        for tier in ftts_kernels::int8::Int8Tier::available() {
            let route = MicroQuantRoute {
                layers: &quant,
                heads: Some(&heads_q8),
                mode: QuantLinearMode::W8A8(tier),
            };
            for seed in 0..3_u32 {
                let talker_hidden = weights_of(config.hidden_size, 950 + seed);
                let primary = (seed as usize * 29) % TALKER_CODEC_VOCAB;
                let mut state = FrameState::new(&config);
                let sequential = decode_frame_with_selector_q8(
                    &config,
                    &rope,
                    &weights,
                    &route,
                    &mut state,
                    &talker_hidden,
                    primary,
                    argmax,
                );
                let verified = verify_frame_draft_q8(
                    &config,
                    &rope,
                    &weights,
                    &route,
                    &talker_hidden,
                    primary,
                    &sequential,
                );
                assert_eq!(
                    verified.greedy_codes(),
                    sequential,
                    "tier {} seed {seed}: verifier rows must reproduce the sequential q8 \
                     trajectory",
                    tier.as_str()
                );
                assert_eq!(
                    verified.accepted_greedy_prefix_len(&sequential),
                    RESIDUAL_DEPTHS,
                    "tier {} seed {seed}: the exact sequential q8 greedy draft must be fully \
                     accepted",
                    tier.as_str()
                );
            }
        }
    }

    #[test]
    fn refined_head_scoring_reproduces_the_full_f32_top_50_exactly() {
        // The lever's correctness claim: every logit the sampler can pick (top 50 of 2,048)
        // is present in the refined row with its exact full-f32 value. Checked across seeds
        // at the real vocab width.
        let hidden_size = 64_usize;
        let vocab = RESIDUAL_VOCAB;
        for seed in 0..12_u32 {
            let head = weights_of(vocab * hidden_size, 300 + seed);
            let normed = weights_of(hidden_size, 900 + seed);
            let head_q8 = ftts_kernels::int8::QuantizedMatrix::quantize(&head, vocab, hidden_size);

            let mut full = vec![0.0_f32; vocab];
            matvec(&head, &normed, &mut full);

            let mut refined = vec![0.0_f32; vocab];
            score_head_refined(
                &head_q8,
                &head,
                &normed,
                ftts_kernels::int8::QuantLinearMode::W8A8(ftts_kernels::int8::Int8Tier::Scalar),
                &mut refined,
            );

            // Full-precision top 50 by the same deterministic rule the selector family uses.
            let mut order: Vec<usize> = (0..vocab).collect();
            order.sort_by(|&a, &b| full[b].partial_cmp(&full[a]).unwrap().then(a.cmp(&b)));
            for &token in &order[..50] {
                assert_eq!(
                    refined[token].to_bits(),
                    full[token].to_bits(),
                    "seed {seed}: top-50 token {token} missing or inexact in the refined row"
                );
            }
            assert_eq!(
                argmax(&refined),
                argmax(&full),
                "seed {seed}: argmax drifted"
            );
        }
    }

    #[test]
    fn q8_layer_step_is_bit_identical_across_every_available_tier() {
        // The tier law at the model seam: the same quantized layer step must produce
        // bit-identical f32 hiddens on scalar, autovec, and (where present) neon-sdot, because
        // the i32 accumulators are exactly equal and the dequant order is fixed.
        let config = tiny();
        let layer = TestLayer::new(&config);
        let weights = layer.borrow();
        let quant = MicroLayerQuant::quantize(&config, &weights);
        let rope = RopeTable::new(&config);
        let hidden = weights_of(config.hidden_size, 42);

        let mut reference = None;
        for tier in ftts_kernels::int8::Int8Tier::available() {
            let mut state = FrameKvState::new(&config);
            let (cos, sin) = rope.row(0);
            let out = layer_step_q8(
                &config,
                &weights,
                &quant,
                cos,
                sin,
                &hidden,
                &mut state,
                ftts_kernels::int8::QuantLinearMode::W8A8(tier),
            );
            match &reference {
                None => reference = Some(out),
                Some(expected) => {
                    assert_eq!(expected.len(), out.len());
                    for (index, (a, b)) in expected.iter().zip(&out).enumerate() {
                        assert_eq!(
                            a.to_bits(),
                            b.to_bits(),
                            "tier {} differs at element {index}",
                            tier.as_str()
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn q8_layer_step_tracks_the_f32_reference_on_tiny_geometry() {
        // Not a parity claim — W8A8 is lossy by design and its production gate is Contract B.
        // This bounds gross wiring failures (wrong scale order, transposed quant layout).
        let config = tiny();
        let layer = TestLayer::new(&config);
        let weights = layer.borrow();
        let quant = MicroLayerQuant::quantize(&config, &weights);
        let rope = RopeTable::new(&config);
        let hidden = weights_of(config.hidden_size, 42);

        let mut f32_state = FrameKvState::new(&config);
        let expected = layer_step(&config, &rope, &weights, &hidden, 0, &mut f32_state);

        let mut q8_state = FrameKvState::new(&config);
        let (cos, sin) = rope.row(0);
        let actual = layer_step_q8(
            &config,
            &weights,
            &quant,
            cos,
            sin,
            &hidden,
            &mut q8_state,
            ftts_kernels::int8::QuantLinearMode::W8A8(ftts_kernels::int8::Int8Tier::Autovec),
        );

        let dot = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>();
        let cosine = dot(&expected, &actual)
            / (dot(&expected, &expected).sqrt() * dot(&actual, &actual).sqrt());
        assert!(
            cosine > 0.99,
            "W8A8 layer step diverged from the f32 reference: cosine {cosine}"
        );
    }

    /// Owns a whole microdecoder's worth of small test weights.
    struct TestBundle {
        layers_owned: Vec<TestLayer>,
        talker_codec_embedding: Vec<f32>,
        residual_embeddings: Vec<Vec<f32>>,
        heads: Vec<Vec<f32>>,
        final_norm: Vec<f32>,
        /// True when the random heads happen to make every depth pick the same code regardless of
        /// conditioning, which would make the autoregression test vacuous rather than failing.
        degenerate: bool,
    }

    impl TestBundle {
        fn new(config: &MicrodecoderConfig) -> Self {
            let layers_owned = (0..config.num_layers)
                .map(|i| TestLayer::new_seeded(config, 100 + i as u32))
                .collect();
            Self {
                layers_owned,
                talker_codec_embedding: weights_of(TALKER_CODEC_VOCAB * config.hidden_size, 200),
                residual_embeddings: (0..RESIDUAL_DEPTHS - 1)
                    .map(|i| weights_of(RESIDUAL_VOCAB * config.hidden_size, 300 + i as u32))
                    .collect(),
                heads: (0..RESIDUAL_DEPTHS)
                    .map(|i| weights_of(RESIDUAL_VOCAB * config.hidden_size, 400 + i as u32))
                    .collect(),
                final_norm: vec![1.0; config.hidden_size],
                degenerate: false,
            }
        }

        /// Borrowed views the caller must keep alive; assembled into [`MicrodecoderWeights`]
        /// by [`TestBundle::weights`].
        fn views(&self) -> (Vec<LayerWeights<'_>>, Vec<&[f32]>, Vec<&[f32]>) {
            (
                self.layers_owned.iter().map(TestLayer::borrow).collect(),
                self.residual_embeddings.iter().map(Vec::as_slice).collect(),
                self.heads.iter().map(Vec::as_slice).collect(),
            )
        }

        fn weights<'a>(
            &'a self,
            layers: &'a [LayerWeights<'a>],
            embeddings: &'a [&'a [f32]],
            heads: &'a [&'a [f32]],
        ) -> MicrodecoderWeights<'a> {
            MicrodecoderWeights {
                layers,
                talker_codec_embedding: &self.talker_codec_embedding,
                residual_embeddings: embeddings,
                heads,
                final_norm: &self.final_norm,
            }
        }
    }

    impl TestLayer {
        fn new_seeded(config: &MicrodecoderConfig, seed: u32) -> Self {
            Self {
                input_norm: vec![1.0; config.hidden_size],
                q_proj: weights_of(config.q_width() * config.hidden_size, seed),
                k_proj: weights_of(config.kv_width() * config.hidden_size, seed + 1),
                v_proj: weights_of(config.kv_width() * config.hidden_size, seed + 2),
                q_norm: vec![1.0; config.head_dim],
                k_norm: vec![1.0; config.head_dim],
                o_proj: weights_of(config.hidden_size * config.q_width(), seed + 3),
                post_attention_norm: vec![1.0; config.hidden_size],
                gate_proj: weights_of(config.intermediate_size * config.hidden_size, seed + 4),
                up_proj: weights_of(config.intermediate_size * config.hidden_size, seed + 5),
                down_proj: weights_of(config.hidden_size * config.intermediate_size, seed + 6),
            }
        }
    }
}
