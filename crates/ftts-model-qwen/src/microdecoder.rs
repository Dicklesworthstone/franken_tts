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
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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
        assert!(config.head_dim % 2 == 0, "head_dim must be even for RoPE");
        let half = config.head_dim / 2;
        let mut cos = vec![0.0_f32; FRAME_POSITIONS * config.head_dim];
        let mut sin = vec![0.0_f32; FRAME_POSITIONS * config.head_dim];
        for position in 0..FRAME_POSITIONS {
            for i in 0..half {
                let freq =
                    1.0 / config.rope_theta.powf(2.0 * i as f32 / config.head_dim as f32);
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

    /// Applies `x = x * cos + rotate_half(x) * sin` for one head at one position.
    ///
    /// # Panics
    ///
    /// Panics when `x` is not one head wide or `position` is outside the frame.
    pub fn apply(&self, x: &mut [f32], position: usize) {
        assert_eq!(x.len(), self.head_dim, "rope input is not one head wide");
        assert!(position < FRAME_POSITIONS, "position outside the frame");
        let half = self.head_dim / 2;
        let base = position * self.head_dim;
        let original = x.to_vec();
        for i in 0..half {
            let (c_lo, s_lo) = (self.cos[base + i], self.sin[base + i]);
            let (c_hi, s_hi) = (self.cos[base + i + half], self.sin[base + i + half]);
            x[i] = original[i] * c_lo - original[i + half] * s_lo;
            x[i + half] = original[i + half] * c_hi + original[i] * s_hi;
        }
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
        rope.apply(&mut q[span], position);
    }
    for head in 0..config.num_kv_heads {
        let span = head * config.head_dim..(head + 1) * config.head_dim;
        let normed_head = rms_norm(&k[span.clone()], weights.k_norm, config.rms_eps);
        k[span.clone()].copy_from_slice(&normed_head);
        rope.apply(&mut k[span], position);
    }

    state.push(&k, &v);

    let scale = 1.0 / (config.head_dim as f32).sqrt();
    let positions = state.len();
    let mut context = vec![0.0_f32; config.q_width()];
    for head in 0..config.num_q_heads {
        let kv_head = head / config.q_per_kv();
        let query = &q[head * config.head_dim..(head + 1) * config.head_dim];
        let keys = &state.keys[kv_head];
        let values = &state.values[kv_head];

        let mut scores = vec![0.0_f32; positions];
        for (p, score) in scores.iter_mut().enumerate() {
            let key = &keys[p * config.head_dim..(p + 1) * config.head_dim];
            *score = query.iter().zip(key.iter()).map(|(a, b)| a * b).sum::<f32>() * scale;
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
    state: &mut FrameKvState,
    talker_hidden: &[f32],
    primary_code: usize,
) -> Vec<usize> {
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

    // Frame boundary: frame N must not see frame N-1's keys.
    state.reset();

    let mut codes = Vec::with_capacity(RESIDUAL_DEPTHS);
    let mut previous_code = primary_code;

    for position in 0..FRAME_POSITIONS {
        let role = position_role(position);
        let mut hidden = match role {
            PositionRole::Conditioning => talker_hidden.to_vec(),
            PositionRole::PrimaryCodeEmbedding { .. } => {
                // The TALKER's table (vocab 3072), not a code_predictor table.
                embedding_row(weights.talker_codec_embedding, previous_code, config.hidden_size)
            }
            PositionRole::ResidualEmbedding { table, .. } => embedding_row(
                weights.residual_embeddings[table],
                previous_code,
                config.hidden_size,
            ),
        };

        for layer in weights.layers {
            hidden = layer_step(config, rope, layer, &hidden, position, state);
        }

        let head = match role {
            // Position 0 conditions the frame and is never scored.
            PositionRole::Conditioning => continue,
            PositionRole::PrimaryCodeEmbedding { head }
            | PositionRole::ResidualEmbedding { head, .. } => head,
        };

        let normed = rms_norm(&hidden, weights.final_norm, config.rms_eps);
        let mut logits = vec![0.0_f32; RESIDUAL_VOCAB];
        matvec(weights.heads[head], &normed, &mut logits);
        let code = argmax(&logits);
        codes.push(code);
        previous_code = code;
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
            PositionRole::ResidualEmbedding { table: 13, head: 14 }
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
        assert_eq!(argmax(&[0.0, 1.0, 1.0]), 1, "ties resolve to the lower index");
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
    fn the_loop_emits_fifteen_codes_in_range_and_resets_its_own_state() {
        let config = tiny();
        let rope = RopeTable::new(&config);
        let bundle = TestBundle::new(&config);
        let (layers, embeddings, heads) = bundle.views();
        let weights = bundle.weights(&layers, &embeddings, &heads);
        let mut state = FrameKvState::new(&config);
        // Pre-dirty the state to prove decode_frame_greedy resets it itself.
        state.push(
            &weights_of(config.kv_width(), 77),
            &weights_of(config.kv_width(), 78),
        );

        let hidden = weights_of(config.hidden_size, 30);
        let codes = decode_frame_greedy(&config, &rope, &weights, &mut state, &hidden, 5);

        assert_eq!(codes.len(), RESIDUAL_DEPTHS, "15 residual codes per frame");
        assert!(codes.iter().all(|c| *c < RESIDUAL_VOCAB));
        assert_eq!(
            state.len(),
            FRAME_POSITIONS,
            "the frame leaves exactly its own 16 positions cached"
        );
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

        let mut state_a = FrameKvState::new(&config);
        let a = decode_frame_greedy(&config, &rope, &weights, &mut state_a, &hidden, 0);
        let mut state_b = FrameKvState::new(&config);
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

        let mut first_state = FrameKvState::new(&config);
        let first = decode_frame_greedy(&config, &rope, &weights, &mut first_state, &hidden, 3);
        let mut second_state = FrameKvState::new(&config);
        let second = decode_frame_greedy(&config, &rope, &weights, &mut second_state, &hidden, 3);
        assert_eq!(first, second, "greedy decode must be reproducible");
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
