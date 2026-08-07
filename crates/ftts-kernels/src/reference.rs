//! f32 reference kernels for the Qwen3-TTS talker.
//!
//! These are the *certified scalar baseline*: the bit-identical fallbacks every future SIMD tier
//! is proved against, and the arithmetic the Contract-A ladder compares to the oracle. Correctness
//! first (G1 > G2) — nothing here is vectorised, nothing is fused, and the reduction order is
//! fixed and documented because a different order is a different answer at f32.
//!
//! Every routine takes and returns plain slices. No tensor objects, no allocation beyond the
//! caller's output buffer, so the same code can later be called from inside a persistent decode
//! loop without allocator traffic.
//!
//! # Model facts these encode (OQ-3 / OQ-4, resolved — do not re-litigate)
//!
//! * RMSNorm is **weight-only** with **f32 variance accumulation** and eps 1e-6.
//! * QK-Norm is an RMSNorm over `head_dim` (128) **only**, applied *after* the q/k projection and
//!   reshape and *before* RoPE.
//! * `attention_bias = false` everywhere: no biases on q/k/v/o or on the MLP.
//! * The MLP is `down(SiLU(gate(x)) * up(x))`.
//! * mRoPE's three axes receive the **same scalar causal index** at every sequence element. It is a
//!   3-D representation of one causal position stream, not a modality-aware schedule. Sections
//!   `[24, 20, 20]` partition the 64 rotary pairs; theta is 1e6.
//!
//! Bead: `frankentts-p1-talker-z2w`.

/// RMSNorm epsilon used by every Qwen3-TTS norm.
pub const RMS_NORM_EPS: f32 = 1e-6;

/// The three mRoPE sections, in pairs (they sum to `head_dim / 2` = 64).
pub const MROPE_SECTIONS: [usize; 3] = [24, 20, 20];

/// Root-mean-square layer norm, weight-only.
///
/// `out[i] = x[i] / sqrt(mean(x^2) + eps) * weight[i]`
///
/// The variance is accumulated in f32 in index order. That order is part of the contract: summing
/// 1024 squares in a different order gives a different f32 result, and Contract A at the CPU tier
/// is an exact compare.
///
/// # Panics
///
/// If `out`, `x` and `weight` do not all have the same length.
pub fn rms_norm(out: &mut [f32], x: &[f32], weight: &[f32], eps: f32) {
    assert_eq!(x.len(), weight.len(), "rms_norm: weight must match width");
    assert_eq!(out.len(), x.len(), "rms_norm: out must match width");

    let mut sum_squares = 0.0f32;
    for value in x {
        sum_squares += value * value;
    }
    let mean = sum_squares / x.len() as f32;
    let scale = 1.0 / (mean + eps).sqrt();
    for index in 0..x.len() {
        out[index] = x[index] * scale * weight[index];
    }
}

/// In-place RMSNorm over each `head_dim`-wide head of a q or k projection.
///
/// This is QK-Norm: one shared `weight` of length `head_dim`, applied independently to every head
/// of every position, after the projection and reshape and **before** RoPE.
pub fn qk_norm(heads: &mut [f32], head_dim: usize, weight: &[f32], eps: f32) {
    assert_eq!(weight.len(), head_dim, "qk_norm: weight must be head_dim");
    assert_eq!(
        heads.len() % head_dim,
        0,
        "qk_norm: buffer must be a whole number of heads"
    );
    let mut scratch = vec![0.0f32; head_dim];
    for head in heads.chunks_mut(head_dim) {
        rms_norm(&mut scratch, head, weight, eps);
        head.copy_from_slice(&scratch);
    }
}

/// SiLU (a.k.a. swish): `x * sigmoid(x)`.
///
/// Computed as `x / (1 + exp(-x))` — the form the reference uses. A mathematically equivalent
/// rearrangement is not bit-equivalent, so the expression is fixed here rather than left to
/// whichever identity looks tidier at the call site.
#[must_use]
pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// The SwiGLU feed-forward block: `down(SiLU(gate(x)) * up(x))`, all bias-free.
///
/// `gate` and `up` are `[intermediate, hidden]` row-major; `down` is `[hidden, intermediate]`.
///
/// # Panics
///
/// If any weight matrix disagrees with the implied dimensions.
pub fn swiglu_mlp(
    out: &mut [f32],
    x: &[f32],
    gate_weight: &[f32],
    up_weight: &[f32],
    down_weight: &[f32],
    hidden: usize,
    intermediate: usize,
) {
    assert_eq!(x.len(), hidden, "swiglu: input width");
    assert_eq!(out.len(), hidden, "swiglu: output width");
    assert_eq!(
        gate_weight.len(),
        intermediate * hidden,
        "swiglu: gate shape"
    );
    assert_eq!(up_weight.len(), intermediate * hidden, "swiglu: up shape");
    assert_eq!(
        down_weight.len(),
        hidden * intermediate,
        "swiglu: down shape"
    );

    let mut activated = vec![0.0f32; intermediate];
    for row in 0..intermediate {
        let base = row * hidden;
        let mut gate_acc = 0.0f32;
        let mut up_acc = 0.0f32;
        for column in 0..hidden {
            gate_acc += gate_weight[base + column] * x[column];
            up_acc += up_weight[base + column] * x[column];
        }
        activated[row] = silu(gate_acc) * up_acc;
    }
    for row in 0..hidden {
        let base = row * intermediate;
        let mut acc = 0.0f32;
        for column in 0..intermediate {
            acc += down_weight[base + column] * activated[column];
        }
        out[row] = acc;
    }
}

/// `out = weight * x` for a row-major `[rows, cols]` weight and a `cols`-long vector.
///
/// The bias-free GEMV every talker projection is. Accumulation is in f32 in column order, which is
/// what the reference does; this is the reduction order the exact-compare contract pins.
pub fn matvec(out: &mut [f32], weight: &[f32], x: &[f32], rows: usize, cols: usize) {
    assert_eq!(weight.len(), rows * cols, "matvec: weight shape");
    assert_eq!(x.len(), cols, "matvec: input width");
    assert_eq!(out.len(), rows, "matvec: output width");
    for row in 0..rows {
        let base = row * cols;
        let mut acc = 0.0f32;
        for column in 0..cols {
            acc += weight[base + column] * x[column];
        }
        out[row] = acc;
    }
}

/// Build the mRoPE cos/sin table for one scalar causal position.
///
/// Both OQ-4 facts are encoded here, and both are easy to get wrong:
///
/// 1. **All three axes receive the same scalar index.** `position` is one number, not a triple.
///    The 3-D shape is a representation of one causal stream. An implementation that advances the
///    axes independently produces plausible-looking output that diverges from the oracle.
/// 2. **The sections partition the rotary pairs**, `[24, 20, 20]` of the 64 pairs at
///    `head_dim = 128`. Because every axis carries the same index, the partition does not change
///    the values here — but it fixes which frequency belongs to which pair, and that ordering is
///    what the fixture pins.
///
/// `cos` and `sin` are filled with `head_dim / 2` entries each: one per rotary pair.
///
/// # Panics
///
/// If the buffers are not `head_dim / 2` long, or the sections do not sum to `head_dim / 2`.
pub fn mrope_table(cos: &mut [f32], sin: &mut [f32], position: i64, head_dim: usize, theta: f32) {
    let pairs = head_dim / 2;
    assert_eq!(cos.len(), pairs, "mrope: cos length");
    assert_eq!(sin.len(), pairs, "mrope: sin length");
    assert_eq!(
        MROPE_SECTIONS.iter().sum::<usize>(),
        pairs,
        "mrope: sections must partition the rotary pairs"
    );

    let position = position as f32;
    for pair in 0..pairs {
        // inv_freq for pair p is theta^(-2p/head_dim); the exponent uses the pair index across the
        // whole head, not an index restarted per section.
        let exponent = (2 * pair) as f32 / head_dim as f32;
        let inv_freq = theta.powf(-exponent);
        let angle = position * inv_freq;
        cos[pair] = angle.cos();
        sin[pair] = angle.sin();
    }
}

/// Apply the interleaved rotary transform to one head in place.
///
/// Interleaved means pair `p` is `(head[2p], head[2p+1])`, rotated as
/// `(x*cos - y*sin, x*sin + y*cos)`. The alternative "half-split" convention pairs `i` with
/// `i + head_dim/2`; using the wrong one still produces finite, plausible activations, which is
/// exactly why the fixture comparison is the only real check.
///
/// # Panics
///
/// If `head` is not twice the length of `cos`/`sin`.
pub fn apply_rope_interleaved(head: &mut [f32], cos: &[f32], sin: &[f32]) {
    let pairs = cos.len();
    assert_eq!(sin.len(), pairs, "rope: cos/sin length mismatch");
    assert_eq!(
        head.len(),
        pairs * 2,
        "rope: head must be 2x the pair count"
    );
    for pair in 0..pairs {
        let x = head[2 * pair];
        let y = head[2 * pair + 1];
        head[2 * pair] = x * cos[pair] - y * sin[pair];
        head[2 * pair + 1] = x * sin[pair] + y * cos[pair];
    }
}

/// Numerically stable softmax over `values`, in place.
///
/// Max-subtraction is not an optimisation here: attention logits at f32 overflow `exp` without it,
/// and the reference subtracts the max, so matching it is part of the contract.
pub fn softmax(values: &mut [f32]) {
    let mut max = f32::NEG_INFINITY;
    for value in values.iter() {
        if *value > max {
            max = *value;
        }
    }
    if !max.is_finite() {
        // An all -inf row (a fully masked query) would otherwise produce NaN. The reference yields
        // zeros; propagating NaN here would poison the whole layer and be blamed on a kernel.
        for value in values.iter_mut() {
            *value = 0.0;
        }
        return;
    }
    let mut sum = 0.0f32;
    for value in values.iter_mut() {
        *value = (*value - max).exp();
        sum += *value;
    }
    for value in values.iter_mut() {
        *value /= sum;
    }
}

/// Grouped-query attention for one query position against a KV cache.
///
/// * `query` is `[q_heads, head_dim]`.
/// * `keys` and `values` are `[kv_len, kv_heads, head_dim]`.
/// * `mask` is `kv_len` additive logits (`0.0` to attend, `-inf` to forbid).
/// * Query head `h` reads KV head `h / (q_heads / kv_heads)` — 16 query heads over 8 KV heads
///   means each KV head serves two query heads, which is why the cache is half the size the query
///   head count suggests.
///
/// # Panics
///
/// If the head counts are not divisible or the buffers disagree with the shapes.
#[allow(clippy::too_many_arguments)]
pub fn gqa_attention(
    out: &mut [f32],
    query: &[f32],
    keys: &[f32],
    values: &[f32],
    mask: &[f32],
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    kv_len: usize,
) {
    assert!(
        kv_heads > 0 && q_heads % kv_heads == 0,
        "gqa: head grouping"
    );
    assert_eq!(query.len(), q_heads * head_dim, "gqa: query shape");
    assert_eq!(keys.len(), kv_len * kv_heads * head_dim, "gqa: key shape");
    assert_eq!(
        values.len(),
        kv_len * kv_heads * head_dim,
        "gqa: value shape"
    );
    assert_eq!(mask.len(), kv_len, "gqa: mask length");
    assert_eq!(out.len(), q_heads * head_dim, "gqa: output shape");

    let group = q_heads / kv_heads;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut logits = vec![0.0f32; kv_len];

    for q_head in 0..q_heads {
        let kv_head = q_head / group;
        let q_base = q_head * head_dim;

        for position in 0..kv_len {
            let k_base = (position * kv_heads + kv_head) * head_dim;
            let mut dot = 0.0f32;
            for lane in 0..head_dim {
                dot += query[q_base + lane] * keys[k_base + lane];
            }
            logits[position] = dot * scale + mask[position];
        }
        softmax(&mut logits);

        let out_base = q_head * head_dim;
        for lane in 0..head_dim {
            out[out_base + lane] = 0.0;
        }
        for position in 0..kv_len {
            let weight = logits[position];
            let v_base = (position * kv_heads + kv_head) * head_dim;
            for lane in 0..head_dim {
                out[out_base + lane] += weight * values[v_base + lane];
            }
        }
    }
}

/// Causal mRoPE position ids under **left padding**.
///
/// `p[j] = cumsum(mask)[j] - 1` for real elements and `1` for pad. This is mandatory and is the
/// single most likely thing to be silently wrong: an unpadded-only port matches every fixture and
/// then fails the moment a batch contains a shorter sequence, because the padded prefix shifts
/// every real position. `rope_deltas` for the normal prompt is `-left_pad_count`.
///
/// `mask[j]` is `true` for a real token.
pub fn causal_positions_left_padded(mask: &[bool]) -> Vec<i64> {
    let mut positions = Vec::with_capacity(mask.len());
    let mut seen = 0i64;
    for real in mask {
        if *real {
            seen += 1;
            positions.push(seen - 1);
        } else {
            positions.push(1);
        }
    }
    positions
}

/// `rope_deltas` for a left-padded prompt: the negated pad count.
#[must_use]
pub fn rope_delta_for(mask: &[bool]) -> i64 {
    -(mask.iter().filter(|real| !**real).count() as i64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rms_norm_matches_its_definition() {
        let x = [1.0f32, 2.0, 3.0, 4.0];
        let weight = [1.0f32; 4];
        let mut out = [0.0f32; 4];
        rms_norm(&mut out, &x, &weight, RMS_NORM_EPS);

        let mean = (1.0 + 4.0 + 9.0 + 16.0) / 4.0f32;
        let scale = 1.0 / (mean + RMS_NORM_EPS).sqrt();
        for index in 0..4 {
            assert!((out[index] - x[index] * scale).abs() < 1e-6);
        }
    }

    #[test]
    fn rms_norm_is_scale_equivariant_only_through_the_weight() {
        // Doubling the input leaves RMSNorm's output unchanged (it normalises the scale away);
        // doubling the weight doubles it. Getting these backwards is a silent, plausible bug.
        let x = [0.5f32, -1.5, 2.0, 0.25];
        let doubled: Vec<f32> = x.iter().map(|v| v * 2.0).collect();
        let weight = [1.0f32; 4];
        let mut a = [0.0f32; 4];
        let mut b = [0.0f32; 4];
        rms_norm(&mut a, &x, &weight, 0.0);
        rms_norm(&mut b, &doubled, &weight, 0.0);
        for index in 0..4 {
            assert!(
                (a[index] - b[index]).abs() < 1e-5,
                "scale must normalise away"
            );
        }
    }

    #[test]
    fn qk_norm_normalises_each_head_independently() {
        let head_dim = 4;
        let weight = [1.0f32; 4];
        // Two heads with very different magnitudes: if the norm were applied across the whole
        // buffer instead of per head, the second head would keep its inflated scale.
        let mut heads = vec![1.0, 1.0, 1.0, 1.0, 100.0, 100.0, 100.0, 100.0];
        qk_norm(&mut heads, head_dim, &weight, 0.0);
        for value in &heads {
            assert!(
                (value - 1.0).abs() < 1e-4,
                "each head normalises to unit RMS"
            );
        }
    }

    #[test]
    fn silu_has_its_defining_values() {
        assert!((silu(0.0) - 0.0).abs() < 1e-7);
        // SiLU is not monotone: it dips below zero before recovering.
        assert!(silu(-1.0) < 0.0);
        assert!(silu(-10.0).abs() < 1e-3, "far negative decays to zero");
        assert!(
            (silu(10.0) - 10.0).abs() < 1e-3,
            "far positive approaches identity"
        );
    }

    #[test]
    fn mrope_gives_all_three_axes_the_same_scalar_position() {
        // OQ-4's correction, asserted directly: the table is a function of ONE index. Building it
        // per axis with differing indices is the failure this test exists to catch.
        let head_dim = 128;
        let pairs = head_dim / 2;
        let mut cos_a = vec![0.0f32; pairs];
        let mut sin_a = vec![0.0f32; pairs];
        let mut cos_b = vec![0.0f32; pairs];
        let mut sin_b = vec![0.0f32; pairs];
        mrope_table(&mut cos_a, &mut sin_a, 7, head_dim, 1e6);
        mrope_table(&mut cos_b, &mut sin_b, 7, head_dim, 1e6);
        assert_eq!(cos_a, cos_b);
        assert_eq!(sin_a, sin_b);
        assert_eq!(MROPE_SECTIONS.iter().sum::<usize>(), pairs);
    }

    #[test]
    fn mrope_position_zero_is_the_identity_rotation() {
        let head_dim = 128;
        let pairs = head_dim / 2;
        let mut cos = vec![0.0f32; pairs];
        let mut sin = vec![0.0f32; pairs];
        mrope_table(&mut cos, &mut sin, 0, head_dim, 1e6);
        assert!(cos.iter().all(|c| (c - 1.0).abs() < 1e-7));
        assert!(sin.iter().all(|s| s.abs() < 1e-7));

        let mut head: Vec<f32> = (0..head_dim).map(|i| i as f32).collect();
        let original = head.clone();
        apply_rope_interleaved(&mut head, &cos, &sin);
        assert_eq!(head, original, "position 0 must not rotate anything");
    }

    #[test]
    fn rope_is_interleaved_not_half_split() {
        // A quarter turn on pair 0 only. Interleaved rotates (head[0], head[1]); the half-split
        // convention would rotate (head[0], head[head_dim/2]). Both are finite and plausible.
        let pairs = 2;
        let cos = [0.0f32, 1.0];
        let sin = [1.0f32, 0.0];
        let mut head = [1.0f32, 0.0, 5.0, 6.0];
        apply_rope_interleaved(&mut head, &cos, &sin);
        assert!((head[0] - 0.0).abs() < 1e-6);
        assert!(
            (head[1] - 1.0).abs() < 1e-6,
            "pair 0 rotated into its partner lane"
        );
        assert!((head[2] - 5.0).abs() < 1e-6, "pair 1 untouched");
        assert!((head[3] - 6.0).abs() < 1e-6);
    }

    #[test]
    fn rope_preserves_the_norm_of_every_pair() {
        // Rotation is orthogonal: whatever the angle, pair magnitudes are invariant. A sign error
        // in the transform breaks this immediately.
        let head_dim = 8;
        let pairs = head_dim / 2;
        let mut cos = vec![0.0f32; pairs];
        let mut sin = vec![0.0f32; pairs];
        mrope_table(&mut cos, &mut sin, 3, head_dim, 1e6);
        let original = [1.0f32, 2.0, -3.0, 0.5, 4.0, -1.0, 0.25, 8.0];
        let mut head = original;
        apply_rope_interleaved(&mut head, &cos, &sin);
        for pair in 0..pairs {
            let before = original[2 * pair].hypot(original[2 * pair + 1]);
            let after = head[2 * pair].hypot(head[2 * pair + 1]);
            assert!(
                (before - after).abs() < 1e-5,
                "pair {pair} changed magnitude"
            );
        }
    }

    #[test]
    fn softmax_sums_to_one_and_respects_masking() {
        let mut values = [1.0f32, 2.0, 3.0];
        softmax(&mut values);
        let sum: f32 = values.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        assert!(values[2] > values[1] && values[1] > values[0]);

        let mut masked = [1.0f32, f32::NEG_INFINITY];
        softmax(&mut masked);
        assert!((masked[0] - 1.0).abs() < 1e-6);
        assert_eq!(masked[1], 0.0, "a masked position must contribute nothing");
    }

    #[test]
    fn a_fully_masked_row_yields_zeros_rather_than_nan() {
        let mut values = [f32::NEG_INFINITY; 3];
        softmax(&mut values);
        assert!(values.iter().all(|v| *v == 0.0));
    }

    #[test]
    fn gqa_shares_each_kv_head_across_its_query_group() {
        // 4 query heads over 2 KV heads: heads 0,1 read KV 0 and heads 2,3 read KV 1. With one
        // KV position the attention weight is 1, so each query head must return its KV head's
        // value verbatim — which makes a wrong grouping obvious.
        let (q_heads, kv_heads, head_dim, kv_len) = (4, 2, 2, 1);
        let query = vec![1.0f32; q_heads * head_dim];
        let keys = vec![0.0f32; kv_len * kv_heads * head_dim];
        let values = vec![10.0, 11.0, 20.0, 21.0];
        let mask = vec![0.0f32; kv_len];
        let mut out = vec![0.0f32; q_heads * head_dim];
        gqa_attention(
            &mut out, &query, &keys, &values, &mask, q_heads, kv_heads, head_dim, kv_len,
        );
        assert_eq!(&out[0..2], &[10.0, 11.0]);
        assert_eq!(&out[2..4], &[10.0, 11.0]);
        assert_eq!(&out[4..6], &[20.0, 21.0]);
        assert_eq!(&out[6..8], &[20.0, 21.0]);
    }

    #[test]
    fn gqa_masking_excludes_a_position_entirely() {
        let (q_heads, kv_heads, head_dim, kv_len) = (1, 1, 2, 2);
        let query = vec![1.0f32, 0.0];
        let keys = vec![1.0f32, 0.0, 1.0, 0.0];
        let values = vec![5.0f32, 5.0, 99.0, 99.0];
        let mask = vec![0.0f32, f32::NEG_INFINITY];
        let mut out = vec![0.0f32; 2];
        gqa_attention(
            &mut out, &query, &keys, &values, &mask, q_heads, kv_heads, head_dim, kv_len,
        );
        assert!((out[0] - 5.0).abs() < 1e-5, "masked value must not leak in");
        assert!((out[1] - 5.0).abs() < 1e-5);
    }

    #[test]
    fn left_padding_shifts_every_real_position() {
        // The bug this prevents: an unpadded-only port passes every fixture and then produces
        // wrong positions the first time a batch contains a shorter sequence.
        let mask = [false, false, true, true, true];
        assert_eq!(causal_positions_left_padded(&mask), vec![1, 1, 0, 1, 2]);
        assert_eq!(rope_delta_for(&mask), -2);

        let unpadded = [true, true, true];
        assert_eq!(causal_positions_left_padded(&unpadded), vec![0, 1, 2]);
        assert_eq!(rope_delta_for(&unpadded), 0);
    }

    #[test]
    fn swiglu_applies_the_gate_to_the_gate_branch_only() {
        // down(SiLU(gate(x)) * up(x)). Swapping which branch is activated is a classic port bug
        // that still produces finite output of the right shape.
        let (hidden, intermediate) = (2, 2);
        let x = [1.0f32, 0.0];
        let gate = [1.0f32, 0.0, 0.0, 1.0];
        let up = [2.0f32, 0.0, 0.0, 2.0];
        let down = [1.0f32, 0.0, 0.0, 1.0];
        let mut out = [0.0f32; 2];
        swiglu_mlp(&mut out, &x, &gate, &up, &down, hidden, intermediate);

        // gate(x) = [1, 0]; up(x) = [2, 0]; SiLU(1)*2 and SiLU(0)*0 = 0.
        assert!((out[0] - silu(1.0) * 2.0).abs() < 1e-6);
        assert!(out[1].abs() < 1e-6);
    }

    #[test]
    fn matvec_is_row_major() {
        let weight = [1.0f32, 2.0, 3.0, 4.0];
        let x = [1.0f32, 1.0];
        let mut out = [0.0f32; 2];
        matvec(&mut out, &weight, &x, 2, 2);
        assert_eq!(out, [3.0, 7.0]);
    }
}
