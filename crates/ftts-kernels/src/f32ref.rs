//! f32 reference kernels: the correctness baseline every optimized tier must reproduce.
//!
//! These are deliberately the obvious implementations. They exist so that a SIMD or int8 kernel has
//! something bit-comparable to be judged against (G1 > G2 — parity first, speed second), and so the
//! first end-to-end forward can be brought up without any unsafe at all. Nothing here is on the hot
//! path yet; nothing here should be "optimized" in place. When a fast tier lands it lands beside
//! these, with a test asserting the two agree.
//!
//! Accumulation is f32 to match the reference stack's CPU fp32 tier. In particular, RMSNorm widens
//! BF16 inputs to f32 and accumulates its variance in f32, exactly as the resolved QK-Norm contract
//! requires.

/// Reduction order used by [`linear_with_accumulation`].
///
/// The scalar order is the f32 reference used by production code. The lane orders are retained so
/// the CPU-fp32 fixture test can identify whether a BLAS-style partial reduction is responsible
/// for a layer-level arithmetic divergence.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum F32LinearAccumulation {
    /// One left-to-right f32 accumulator.
    Scalar,
    /// Four independent f32 partial accumulators, reduced in lane order.
    Lanes4,
    /// Eight independent f32 partial accumulators, reduced in lane order.
    Lanes8,
    /// Four FMA partial accumulators, reduced in lane order.
    FusedLanes4,
    /// Eight FMA partial accumulators, reduced in lane order.
    FusedLanes8,
}

/// Arithmetic used by [`rms_norm_with_arithmetic`] to form RMSNorm's scale.
///
/// The scalar reciprocal-square-root path is the f32 reference used by production code. The other
/// modes make the exact CPU-fp32 fixture able to discriminate reduction precision and reciprocal
/// placement without changing that normal path.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum F32RmsNormArithmetic {
    /// Left-to-right f32 reduction and `sqrt(value).recip()`.
    ScalarReciprocalSqrt,
    /// Left-to-right f32 reduction and `1.0 / sqrt(value)`.
    ScalarDivideSqrt,
    /// Four f32 partial sums, then `sqrt(value).recip()`.
    Lanes4ReciprocalSqrt,
    /// Eight f32 partial sums, then `sqrt(value).recip()`.
    Lanes8ReciprocalSqrt,
    /// f64 reduction and scale calculation, narrowed only at the final scale.
    F64ReciprocalSqrt,
}

/// Association used by [`silu_mul_in_place_with_arithmetic`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum F32SiluArithmetic {
    /// `x / (1 + exp(-x))`.
    Divide,
    /// `x * (1 / (1 + exp(-x)))`, matching `x * sigmoid(x)` association.
    MultiplyReciprocal,
}

/// Normalization form used by [`softmax_rows_with_arithmetic`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum F32SoftmaxArithmetic {
    /// Form one reciprocal then multiply every exponent by it.
    ReciprocalMultiply,
    /// Divide every exponent by the sum directly.
    Divide,
}

/// Row-major matrix-vector/matrix-matrix product in the layout PyTorch `Linear` stores.
///
/// `x` is `[m, k]`, `weight` is `[n, k]` (out-features major, as `nn.Linear` stores it), and the
/// result is `[m, n]`. Bias is optional because every attention/MLP projection in this model is
/// bias-free; only `text_projection` carries one.
///
/// # Panics
///
/// Panics if any slice length disagrees with `m`, `k`, `n`.
pub fn linear(
    x: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    m: usize,
    k: usize,
    n: usize,
    out: &mut [f32],
) {
    linear_with_accumulation(x, weight, bias, m, k, n, F32LinearAccumulation::Scalar, out);
}

/// Same operation as [`linear`], with an explicitly chosen f32 dot-product reduction order.
///
/// This exists for parity forensics. The normal [`linear`] entry point remains the scalar,
/// left-to-right reference.
#[allow(clippy::too_many_arguments)]
pub fn linear_with_accumulation(
    x: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    m: usize,
    k: usize,
    n: usize,
    accumulation: F32LinearAccumulation,
    out: &mut [f32],
) {
    assert_eq!(x.len(), m * k, "x must be [m, k]");
    assert_eq!(weight.len(), n * k, "weight must be [n, k]");
    assert_eq!(out.len(), m * n, "out must be [m, n]");
    if let Some(bias) = bias {
        assert_eq!(bias.len(), n, "bias must be [n]");
    }

    for row in 0..m {
        let x_row = &x[row * k..row * k + k];
        for col in 0..n {
            let w_row = &weight[col * k..col * k + k];
            let sum = dot_with_accumulation(x_row, w_row, accumulation);
            out[row * n + col] = bias.map_or(sum, |b| sum + b[col]);
        }
    }
}

fn dot_with_accumulation(x: &[f32], weight: &[f32], accumulation: F32LinearAccumulation) -> f32 {
    assert_eq!(x.len(), weight.len(), "dot-product inputs must match");
    match accumulation {
        F32LinearAccumulation::Scalar => {
            let mut sum = 0.0f32;
            for index in 0..x.len() {
                sum += x[index] * weight[index];
            }
            sum
        }
        F32LinearAccumulation::Lanes4
        | F32LinearAccumulation::Lanes8
        | F32LinearAccumulation::FusedLanes4
        | F32LinearAccumulation::FusedLanes8 => {
            let lanes = match accumulation {
                F32LinearAccumulation::Lanes4 => 4,
                F32LinearAccumulation::Lanes8 => 8,
                F32LinearAccumulation::FusedLanes4 => 4,
                F32LinearAccumulation::FusedLanes8 => 8,
                F32LinearAccumulation::Scalar => unreachable!("scalar handled above"),
            };
            let mut partial = [0.0f32; 8];
            for index in 0..x.len() {
                let lane = index % lanes;
                partial[lane] = match accumulation {
                    F32LinearAccumulation::FusedLanes4 | F32LinearAccumulation::FusedLanes8 => {
                        x[index].mul_add(weight[index], partial[lane])
                    }
                    F32LinearAccumulation::Scalar
                    | F32LinearAccumulation::Lanes4
                    | F32LinearAccumulation::Lanes8 => partial[lane] + x[index] * weight[index],
                };
            }
            let mut sum = 0.0f32;
            for value in &partial[..lanes] {
                sum += *value;
            }
            sum
        }
    }
}

/// Qwen3 RMSNorm: `x * rsqrt(mean(x^2) + eps) * weight`, weight-only, no centering.
///
/// # Panics
///
/// Panics if `x` is not `rows * dim` elements or `weight` is not `dim`.
pub fn rms_norm(x: &[f32], weight: &[f32], eps: f32, rows: usize, dim: usize, out: &mut [f32]) {
    rms_norm_with_arithmetic(
        x,
        weight,
        eps,
        rows,
        dim,
        F32RmsNormArithmetic::ScalarReciprocalSqrt,
        out,
    );
}

/// Same operation as [`rms_norm`], with an explicitly selected reduction and scale calculation.
///
/// This entry point is for CPU-fp32 parity forensics; [`rms_norm`] remains the normal scalar f32
/// reference path.
pub fn rms_norm_with_arithmetic(
    x: &[f32],
    weight: &[f32],
    eps: f32,
    rows: usize,
    dim: usize,
    arithmetic: F32RmsNormArithmetic,
    out: &mut [f32],
) {
    assert_eq!(x.len(), rows * dim, "x must be [rows, dim]");
    assert_eq!(weight.len(), dim, "weight must be [dim]");
    assert_eq!(out.len(), rows * dim, "out must be [rows, dim]");

    for row in 0..rows {
        let src = &x[row * dim..row * dim + dim];
        let scale = rms_scale(src, eps, arithmetic);
        for index in 0..dim {
            out[row * dim + index] = src[index] * scale * weight[index];
        }
    }
}

fn rms_scale(src: &[f32], eps: f32, arithmetic: F32RmsNormArithmetic) -> f32 {
    match arithmetic {
        F32RmsNormArithmetic::ScalarReciprocalSqrt => {
            let sum = sum_squares_f32(src, 1);
            (sum / src.len() as f32 + eps).sqrt().recip()
        }
        F32RmsNormArithmetic::ScalarDivideSqrt => {
            let sum = sum_squares_f32(src, 1);
            1.0f32 / (sum / src.len() as f32 + eps).sqrt()
        }
        F32RmsNormArithmetic::Lanes4ReciprocalSqrt => {
            let sum = sum_squares_f32(src, 4);
            (sum / src.len() as f32 + eps).sqrt().recip()
        }
        F32RmsNormArithmetic::Lanes8ReciprocalSqrt => {
            let sum = sum_squares_f32(src, 8);
            (sum / src.len() as f32 + eps).sqrt().recip()
        }
        F32RmsNormArithmetic::F64ReciprocalSqrt => {
            let mut sum = 0.0f64;
            for value in src {
                let value = f64::from(*value);
                sum += value * value;
            }
            (sum / src.len() as f64 + f64::from(eps)).sqrt().recip() as f32
        }
    }
}

fn sum_squares_f32(src: &[f32], lanes: usize) -> f32 {
    let mut partial = [0.0f32; 8];
    for (index, value) in src.iter().enumerate() {
        partial[index % lanes] += *value * *value;
    }
    let mut sum = 0.0f32;
    for value in &partial[..lanes] {
        sum += *value;
    }
    sum
}

/// SwiGLU's elementwise half: `silu(gate) * up`, written into `gate`.
pub fn silu_mul_in_place(gate: &mut [f32], up: &[f32]) {
    silu_mul_in_place_with_arithmetic(gate, up, F32SiluArithmetic::Divide);
}

/// Same operation as [`silu_mul_in_place`], with an explicitly chosen f32 association.
pub fn silu_mul_in_place_with_arithmetic(
    gate: &mut [f32],
    up: &[f32],
    arithmetic: F32SiluArithmetic,
) {
    assert_eq!(gate.len(), up.len(), "gate and up must match");
    for (g, u) in gate.iter_mut().zip(up) {
        let x = *g;
        let denominator = 1.0 + (-x).exp();
        let silu = match arithmetic {
            F32SiluArithmetic::Divide => x / denominator,
            F32SiluArithmetic::MultiplyReciprocal => x * denominator.recip(),
        };
        *g = silu * u;
    }
}

/// In-place row-wise softmax in f32, max-subtracted for stability.
pub fn softmax_rows(x: &mut [f32], rows: usize, cols: usize) {
    softmax_rows_with_arithmetic(x, rows, cols, F32SoftmaxArithmetic::ReciprocalMultiply);
}

/// Same operation as [`softmax_rows`], with an explicitly selected normalization form.
pub fn softmax_rows_with_arithmetic(
    x: &mut [f32],
    rows: usize,
    cols: usize,
    arithmetic: F32SoftmaxArithmetic,
) {
    assert_eq!(x.len(), rows * cols, "x must be [rows, cols]");
    for row in 0..rows {
        let slice = &mut x[row * cols..row * cols + cols];
        let mut max = f32::NEG_INFINITY;
        for value in slice.iter() {
            if *value > max {
                max = *value;
            }
        }
        let mut sum = 0.0f32;
        for value in slice.iter_mut() {
            *value = (*value - max).exp();
            sum += *value;
        }
        for value in slice.iter_mut() {
            *value = match arithmetic {
                F32SoftmaxArithmetic::ReciprocalMultiply => *value * sum.recip(),
                F32SoftmaxArithmetic::Divide => *value / sum,
            };
        }
    }
}

/// Grouped-query attention for row-major f32 tensors.
///
/// `queries` and `out` are `[query_positions, q_heads, head_dim]`; `keys` and `values` are
/// `[key_positions, kv_heads, head_dim]`; `additive_mask` is `[query_positions, key_positions]`.
/// Query head `h` reads key/value head `h / (q_heads / kv_heads)`, matching Qwen3-TTS's 16 query
/// heads over 8 KV heads. The reduction order is scalar and fixed so an ISA-specific kernel has a
/// direct f32 reference to compare against.
///
/// # Panics
///
/// Panics if the dimensions disagree or query heads are not evenly grouped over KV heads.
#[allow(clippy::too_many_arguments)]
pub fn gqa_attention(
    queries: &[f32],
    keys: &[f32],
    values: &[f32],
    additive_mask: &[f32],
    query_positions: usize,
    key_positions: usize,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    out: &mut [f32],
) {
    gqa_attention_with_softmax(
        queries,
        keys,
        values,
        additive_mask,
        query_positions,
        key_positions,
        q_heads,
        kv_heads,
        head_dim,
        F32SoftmaxArithmetic::ReciprocalMultiply,
        out,
    );
}

/// Same operation as [`gqa_attention`], with an explicitly selected softmax normalization form.
#[allow(clippy::too_many_arguments)]
pub fn gqa_attention_with_softmax(
    queries: &[f32],
    keys: &[f32],
    values: &[f32],
    additive_mask: &[f32],
    query_positions: usize,
    key_positions: usize,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    softmax_arithmetic: F32SoftmaxArithmetic,
    out: &mut [f32],
) {
    gqa_attention_with_arithmetic(
        queries,
        keys,
        values,
        additive_mask,
        query_positions,
        key_positions,
        q_heads,
        kv_heads,
        head_dim,
        softmax_arithmetic,
        F32LinearAccumulation::Scalar,
        out,
    );
}

/// Same operation as [`gqa_attention`], with selected softmax and dot-product reduction forms.
#[allow(clippy::too_many_arguments)]
pub fn gqa_attention_with_arithmetic(
    queries: &[f32],
    keys: &[f32],
    values: &[f32],
    additive_mask: &[f32],
    query_positions: usize,
    key_positions: usize,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    softmax_arithmetic: F32SoftmaxArithmetic,
    accumulation: F32LinearAccumulation,
    out: &mut [f32],
) {
    assert!(kv_heads > 0, "at least one KV head is required");
    assert_eq!(
        q_heads % kv_heads,
        0,
        "query heads must divide evenly into KV groups"
    );
    assert_eq!(
        queries.len(),
        query_positions * q_heads * head_dim,
        "queries must be [query_positions, q_heads, head_dim]"
    );
    assert_eq!(
        keys.len(),
        key_positions * kv_heads * head_dim,
        "keys must be [key_positions, kv_heads, head_dim]"
    );
    assert_eq!(
        values.len(),
        key_positions * kv_heads * head_dim,
        "values must be [key_positions, kv_heads, head_dim]"
    );
    assert_eq!(
        additive_mask.len(),
        query_positions * key_positions,
        "mask must be [query_positions, key_positions]"
    );
    assert_eq!(
        out.len(),
        query_positions * q_heads * head_dim,
        "out must be [query_positions, q_heads, head_dim]"
    );

    let scale = (head_dim as f32).sqrt().recip();
    let kv_group = q_heads / kv_heads;
    let mut scores = vec![0.0f32; key_positions];

    for query_position in 0..query_positions {
        let mask =
            &additive_mask[query_position * key_positions..(query_position + 1) * key_positions];
        for q_head in 0..q_heads {
            let kv_head = q_head / kv_group;
            let query_base = (query_position * q_heads + q_head) * head_dim;
            let query = &queries[query_base..query_base + head_dim];

            for (key_position, score) in scores.iter_mut().enumerate() {
                let key_base = (key_position * kv_heads + kv_head) * head_dim;
                let key = &keys[key_base..key_base + head_dim];
                let dot = dot_with_accumulation(query, key, accumulation);
                *score = dot * scale + mask[key_position];
            }
            softmax_rows_with_arithmetic(&mut scores, 1, key_positions, softmax_arithmetic);

            let out_base = query_base;
            attention_weighted_sum(
                &scores,
                values,
                kv_head,
                kv_heads,
                head_dim,
                accumulation,
                &mut out[out_base..out_base + head_dim],
            );
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn attention_weighted_sum(
    scores: &[f32],
    values: &[f32],
    kv_head: usize,
    kv_heads: usize,
    head_dim: usize,
    accumulation: F32LinearAccumulation,
    out: &mut [f32],
) {
    let lanes = match accumulation {
        F32LinearAccumulation::Scalar => 1,
        F32LinearAccumulation::Lanes4 | F32LinearAccumulation::FusedLanes4 => 4,
        F32LinearAccumulation::Lanes8 | F32LinearAccumulation::FusedLanes8 => 8,
    };
    for lane in 0..head_dim {
        let mut partial = [0.0f32; 8];
        for (key_position, weight) in scores.iter().copied().enumerate() {
            let value = values[(key_position * kv_heads + kv_head) * head_dim + lane];
            let partial_index = key_position % lanes;
            partial[partial_index] = match accumulation {
                F32LinearAccumulation::FusedLanes4 | F32LinearAccumulation::FusedLanes8 => {
                    weight.mul_add(value, partial[partial_index])
                }
                F32LinearAccumulation::Scalar
                | F32LinearAccumulation::Lanes4
                | F32LinearAccumulation::Lanes8 => partial[partial_index] + weight * value,
            };
        }
        let mut sum = 0.0f32;
        for value in &partial[..lanes] {
            sum += *value;
        }
        out[lane] = sum;
    }
}

/// Collapse the three mRoPE axes into one `cos`/`sin` row using the checkpoint's INTERLEAVED rule.
///
/// The pinned config sets `rope_scaling.interleaved = true`, which selects a different branch from
/// the familiar section-split one — a difference that is numerically invisible whenever the three
/// axes carry equal positions (which OQ-4 says they always do here, all three receiving the same
/// scalar causal index) and therefore exactly the kind of thing a port gets wrong and only discovers
/// against a batched or genuinely multimodal input. It is implemented faithfully regardless.
///
/// `axes` is the first half of each axis's row, `[3][half]`; `out` receives `[half]`. Element `j`
/// takes axis `j % 3` while `j` lies in `1..sections[1..].max() * 3`, and axis 0 elsewhere.
///
/// # Panics
///
/// Panics if `out` is not `half` long or an axis row is short.
pub fn mrope_interleave(axes: [&[f32]; 3], sections: [usize; 3], out: &mut [f32]) {
    let half = out.len();
    for axis in axes {
        assert!(
            axis.len() >= half,
            "axis row shorter than the half-dimension"
        );
    }

    // Start from axis 0 everywhere, then overwrite the strided lanes from axes 1 and 2, exactly as
    // the reference does with its `x_t[..., beg:end:3] = x[beg, ..., beg:end:3]` assignments.
    out.copy_from_slice(&axes[0][..half]);
    let modality_num = 3usize;
    for (axis_index, section) in sections.iter().enumerate().skip(1) {
        let end = section * modality_num;
        let mut lane = axis_index;
        while lane < end && lane < half {
            out[lane] = axes[axis_index][lane];
            lane += modality_num;
        }
    }
}

/// Apply rotary embeddings to one head row in the `rotate_half` layout.
///
/// `row` is `[head_dim]`; `cos` and `sin` are the full `[head_dim]` rows (the doubled half). The
/// transform is `x*cos + rotate_half(x)*sin` where `rotate_half` maps `[a, b] -> [-b, a]` over the
/// two halves.
///
/// # Panics
///
/// Panics if `cos`/`sin` do not match `row`, or if `head_dim` is odd.
pub fn apply_rope_in_place(row: &mut [f32], cos: &[f32], sin: &[f32]) {
    let dim = row.len();
    assert_eq!(cos.len(), dim, "cos must match head_dim");
    assert_eq!(sin.len(), dim, "sin must match head_dim");
    assert!(dim.is_multiple_of(2), "head_dim must be even");

    let half = dim / 2;
    let original: Vec<f32> = row.to_vec();
    for index in 0..dim {
        let rotated = if index < half {
            -original[index + half]
        } else {
            original[index - half]
        };
        row[index] = original[index] * cos[index] + rotated * sin[index];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_matches_a_hand_computed_product() {
        // x = [[1, 2, 3]], weight = [[1, 0, -1], [2, 2, 2]] -> [1*1 + 2*0 + 3*-1, 2+4+6] = [-2, 12]
        let x = [1.0, 2.0, 3.0];
        let weight = [1.0, 0.0, -1.0, 2.0, 2.0, 2.0];
        let mut out = [0.0; 2];
        linear(&x, &weight, None, 1, 3, 2, &mut out);
        assert_eq!(out, [-2.0, 12.0]);

        let mut biased = [0.0; 2];
        linear(&x, &weight, Some(&[10.0, -12.0]), 1, 3, 2, &mut biased);
        assert_eq!(biased, [8.0, 0.0]);
    }

    #[test]
    fn rms_norm_normalizes_and_scales() {
        // mean(x^2) for [3, 4] is 12.5; rsqrt(12.5 + 0) ~ 0.2828427
        let x = [3.0f32, 4.0];
        let weight = [1.0f32, 1.0];
        let mut out = [0.0; 2];
        rms_norm(&x, &weight, 0.0, 1, 2, &mut out);
        let expected = 12.5f32.sqrt().recip();
        assert!((out[0] - 3.0 * expected).abs() < 1e-6);
        assert!((out[1] - 4.0 * expected).abs() < 1e-6);

        // The weight is applied per element, after scaling.
        let mut weighted = [0.0; 2];
        rms_norm(&x, &[2.0, 0.5], 0.0, 1, 2, &mut weighted);
        assert!((weighted[0] - 3.0 * expected * 2.0).abs() < 1e-6);
        assert!((weighted[1] - 4.0 * expected * 0.5).abs() < 1e-6);
    }

    #[test]
    fn silu_mul_matches_the_definition() {
        let mut gate = [0.0f32, 1.0, -1.0];
        let up = [1.0f32, 2.0, 3.0];
        silu_mul_in_place(&mut gate, &up);
        assert_eq!(gate[0], 0.0);
        let silu_one = 1.0f32 / (1.0 + (-1.0f32).exp());
        assert!((gate[1] - silu_one * 2.0).abs() < 1e-6);
        let silu_neg = -1.0f32 / (1.0 + 1.0f32.exp());
        assert!((gate[2] - silu_neg * 3.0).abs() < 1e-6);
    }

    #[test]
    fn softmax_rows_sums_to_one_and_is_shift_invariant() {
        let mut x = [1.0f32, 2.0, 3.0, 101.0, 102.0, 103.0];
        softmax_rows(&mut x, 2, 3);
        let first: f32 = x[..3].iter().sum();
        let second: f32 = x[3..].iter().sum();
        assert!((first - 1.0).abs() < 1e-6);
        assert!((second - 1.0).abs() < 1e-6);
        // Rows differing by a constant shift must produce identical distributions.
        for index in 0..3 {
            assert!((x[index] - x[index + 3]).abs() < 1e-6);
        }
    }

    #[test]
    fn gqa_maps_each_query_head_to_its_kv_group() {
        let (query_positions, key_positions, q_heads, kv_heads, head_dim) = (1, 1, 4, 2, 2);
        let queries = vec![0.0f32; query_positions * q_heads * head_dim];
        let keys = vec![0.0f32; key_positions * kv_heads * head_dim];
        let values = [10.0f32, 11.0, 20.0, 21.0];
        let mut out = vec![0.0f32; query_positions * q_heads * head_dim];

        gqa_attention(
            &queries,
            &keys,
            &values,
            &[0.0],
            query_positions,
            key_positions,
            q_heads,
            kv_heads,
            head_dim,
            &mut out,
        );

        assert_eq!(&out[0..2], &[10.0, 11.0]);
        assert_eq!(&out[2..4], &[10.0, 11.0]);
        assert_eq!(&out[4..6], &[20.0, 21.0]);
        assert_eq!(&out[6..8], &[20.0, 21.0]);
    }

    #[test]
    fn gqa_honors_the_additive_causal_mask() {
        let (query_positions, key_positions, q_heads, kv_heads, head_dim) = (2, 2, 1, 1, 2);
        let queries = vec![0.0f32; query_positions * q_heads * head_dim];
        let keys = vec![0.0f32; key_positions * kv_heads * head_dim];
        let values = [2.0f32, 4.0, 10.0, 20.0];
        let mask = [0.0f32, f32::NEG_INFINITY, 0.0, 0.0];
        let mut out = vec![0.0f32; query_positions * q_heads * head_dim];

        gqa_attention(
            &queries,
            &keys,
            &values,
            &mask,
            query_positions,
            key_positions,
            q_heads,
            kv_heads,
            head_dim,
            &mut out,
        );

        assert_eq!(&out[0..2], &[2.0, 4.0]);
        assert_eq!(&out[2..4], &[6.0, 12.0]);
    }

    #[test]
    fn rope_rotates_a_known_pair() {
        // head_dim 2, cos = [0, 0], sin = [1, 1]: [a, b] -> [-b, a]
        let mut row = [3.0f32, 5.0];
        apply_rope_in_place(&mut row, &[0.0, 0.0], &[1.0, 1.0]);
        assert_eq!(row, [-5.0, 3.0]);

        // Identity when cos = 1, sin = 0.
        let mut same = [3.0f32, 5.0];
        apply_rope_in_place(&mut same, &[1.0, 1.0], &[0.0, 0.0]);
        assert_eq!(same, [3.0, 5.0]);
    }

    #[test]
    fn mrope_interleave_is_identity_when_all_axes_agree() {
        // OQ-4: all three axes carry the same scalar causal index in this model, so the interleave
        // must be a no-op on equal axes. If it is not, the lane arithmetic is wrong.
        let axis: Vec<f32> = (0..64).map(|value| value as f32).collect();
        let mut out = vec![0.0f32; 64];
        mrope_interleave([&axis, &axis, &axis], [24, 20, 20], &mut out);
        assert_eq!(out, axis);
    }

    #[test]
    fn mrope_interleave_selects_the_documented_lanes() {
        let zeros = vec![0.0f32; 64];
        let ones = vec![1.0f32; 64];
        let twos = vec![2.0f32; 64];
        let mut out = vec![0.0f32; 64];
        mrope_interleave([&zeros, &ones, &twos], [24, 20, 20], &mut out);

        // Lanes 1, 4, .. < 60 come from axis 1; lanes 2, 5, .. < 60 from axis 2; the rest stay 0,
        // including every lane at or above 60.
        for (lane, value) in out.iter().enumerate() {
            let expected = if lane < 60 && lane % 3 == 1 {
                1.0
            } else if lane < 60 && lane % 3 == 2 {
                2.0
            } else {
                0.0
            };
            assert_eq!(*value, expected, "lane {lane}");
        }
    }
}
