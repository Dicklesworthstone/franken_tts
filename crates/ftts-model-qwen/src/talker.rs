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

use ftts_kernels::f32ref;

/// Talker geometry. Field values come from `talker_config` in the pinned `config.json`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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
#[derive(Clone, Debug, Default)]
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
pub fn collapse_mrope(axes: [&[f32]; 3], seq: usize, head_dim: usize, sections: [usize; 3]) -> Vec<f32> {
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
    let hidden_size = config.hidden_size;
    let head_dim = config.head_dim;
    let query_width = config.query_width();
    let kv_width = config.kv_width();
    assert_eq!(hidden.len(), seq * hidden_size, "hidden must be [seq, hidden]");
    assert_eq!(rotary.cos.len(), seq * head_dim, "cos must be [seq, head_dim]");
    assert_eq!(rotary.sin.len(), seq * head_dim, "sin must be [seq, head_dim]");

    let past = cache.len();
    let total = past + seq;
    assert_eq!(mask.len(), seq * total, "mask must be [seq, past + seq]");

    // ── Attention block ────────────────────────────────────────────────────────────────────────
    let mut normed = vec![0.0f32; seq * hidden_size];
    f32ref::rms_norm(hidden, weights.input_layernorm, config.rms_norm_eps, seq, hidden_size, &mut normed);

    let mut queries = vec![0.0f32; seq * query_width];
    let mut keys = vec![0.0f32; seq * kv_width];
    let mut values = vec![0.0f32; seq * kv_width];
    f32ref::linear(&normed, weights.q_proj, None, seq, hidden_size, query_width, &mut queries);
    f32ref::linear(&normed, weights.k_proj, None, seq, hidden_size, kv_width, &mut keys);
    f32ref::linear(&normed, weights.v_proj, None, seq, hidden_size, kv_width, &mut values);

    // QK-Norm over head_dim, then rotary — in that order. Values are neither normed nor rotated.
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

    let scaling = (head_dim as f32).sqrt().recip();
    let mut context = vec![0.0f32; seq * query_width];
    let mut scores = vec![0.0f32; total];
    for position in 0..seq {
        for head in 0..config.num_attention_heads {
            let kv_head = head / config.kv_group();
            let query = &queries[position * query_width + head * head_dim..][..head_dim];

            for key_position in 0..total {
                let key = &cache.keys[key_position * kv_width + kv_head * head_dim..][..head_dim];
                let mut dot = 0.0f32;
                for index in 0..head_dim {
                    dot += query[index] * key[index];
                }
                scores[key_position] = dot * scaling + mask[position * total + key_position];
            }
            f32ref::softmax_rows(&mut scores, 1, total);

            let out = &mut context[position * query_width + head * head_dim..][..head_dim];
            out.fill(0.0);
            for key_position in 0..total {
                let weight = scores[key_position];
                let value = &cache.values[key_position * kv_width + kv_head * head_dim..][..head_dim];
                for index in 0..head_dim {
                    out[index] += weight * value[index];
                }
            }
        }
    }

    let mut attention = vec![0.0f32; seq * hidden_size];
    f32ref::linear(&context, weights.o_proj, None, seq, query_width, hidden_size, &mut attention);
    for (state, delta) in hidden.iter_mut().zip(&attention) {
        *state += *delta;
    }

    // ── MLP block ──────────────────────────────────────────────────────────────────────────────
    f32ref::rms_norm(hidden, weights.post_attention_layernorm, config.rms_norm_eps, seq, hidden_size, &mut normed);

    let intermediate = config.intermediate_size;
    let mut gate = vec![0.0f32; seq * intermediate];
    let mut up = vec![0.0f32; seq * intermediate];
    f32ref::linear(&normed, weights.gate_proj, None, seq, hidden_size, intermediate, &mut gate);
    f32ref::linear(&normed, weights.up_proj, None, seq, hidden_size, intermediate, &mut up);
    f32ref::silu_mul_in_place(&mut gate, &up);

    let mut down = vec![0.0f32; seq * hidden_size];
    f32ref::linear(&gate, weights.down_proj, None, seq, intermediate, hidden_size, &mut down);
    for (state, delta) in hidden.iter_mut().zip(&down) {
        *state += *delta;
    }
}

/// Number of identical decoder layers in the pinned 0.6B Base talker.
pub const TALKER_LAYER_COUNT: usize = 28;

/// Primary-code vocabulary width. The residual-code heads are 2,048 wide; this one is not.
pub const PRIMARY_CODE_VOCAB_SIZE: usize = 3_072;

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
#[derive(Clone, Debug)]
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
    assert_eq!(weights.final_norm.len(), config.hidden_size, "final norm shape");
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Weights that make the layer an identity-ish probe: zeroed projections leave the residual.
    fn zero_weights(config: &TalkerConfig) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
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
        let mut hidden: Vec<f32> = (0..seq * config.hidden_size).map(|i| (i % 7) as f32).collect();
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
            RotaryRows { cos: &cos, sin: &sin },
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
            RotaryRows { cos: &cos, sin: &sin },
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
            RotaryRows { cos: &step_cos, sin: &step_sin },
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
