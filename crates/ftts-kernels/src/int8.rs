//! Int8 W8A8 kernels: symmetric per-output-channel Q8 weights times per-row Q8 activations.
//!
//! This is the Phase-2/3A quantized projection route for the talker and microdecoder GEMMs.
//! The numeric contract is S8S8: weights quantized by the canonical symmetric recipe
//! (`scale = max|row| / 127`, ties-to-even, `[-127, 127]`, `-128` never emitted — identical to
//! `ftts-artifacts::converter::quantize_output_channel_q8`, byte-for-byte, asserted by a
//! cross-crate test in `ftts-model-qwen`), activations quantized dynamically per row with the
//! same recipe. Accumulation is exact i32; the two f32 scales are applied once, after
//! accumulation, in a fixed multiplication order shared by every tier.
//!
//! ## Tier law
//!
//! Every tier of [`dot_i32`] is *exactly equal in i32* to [`Int8Tier::Scalar`] on every input —
//! integer addition is associative, and the overflow selftest proves the all-extreme reduction
//! fits i32 at every census binding K. A tier is only dispatchable after
//! [`crate::selftest::run_selftest`] has executed its all-extreme proof rows through the real
//! kernel function on the running silicon. Do not add a tier here without extending the selftest.
//!
//! Inherited prior NE-INH-003 (re-verify per toolchain): on Apple M4, LLVM autovectorization of
//! the scalar shape beat a hand SDOT micro-tile at m=1. Both routes therefore ship; dispatch
//! preference is decided by measurement (`FTTS_INT8_TIER` forces a route for A/B), never by
//! assumption.

/// Largest absolute Q8 byte the canonical symmetric recipe emits.
pub const Q8_MAX_ABS: i8 = 127;

/// Quantizes one row (weight output channel or activation row) with the canonical symmetric
/// Q8 recipe.
///
/// The returned scale is `max(abs(row)) / 127`; all-zero rows use the explicit scale `1.0` and
/// emit zero bytes. Values are clamped to `[-127, 127]` and rounded ties-to-even; `-128` is never
/// emitted. This is the same arithmetic as the offline converter's
/// `quantize_output_channel_q8`, restated here because the artifact crate depends on this one.
///
/// # Panics
///
/// Panics if `output.len() != row.len()` or a value is non-finite. A NaN/inf activation reaching
/// the quantizer means the f32 graph upstream is already corrupt; refusing loudly here beats
/// synthesizing garbage audio quietly.
pub fn quantize_row_q8(row: &[f32], output: &mut [i8]) -> f32 {
    assert_eq!(output.len(), row.len(), "quantize output length mismatch");
    let mut maximum = 0.0_f32;
    for (index, &value) in row.iter().enumerate() {
        assert!(
            value.is_finite(),
            "non-finite value {value} at index {index} reached the Q8 quantizer"
        );
        maximum = maximum.max(value.abs());
    }
    if maximum == 0.0 {
        output.fill(0);
        return 1.0;
    }
    let scale = maximum / 127.0;
    for (&value, slot) in row.iter().zip(output.iter_mut()) {
        let rounded = (value / scale).clamp(-127.0, 127.0).round_ties_even();
        // The clamp bounds the conversion inside i8, and the symmetric contract additionally
        // excludes the otherwise-representable -128.
        *slot = rounded as i8;
    }
    scale
}

/// A weight matrix quantized with per-output-channel symmetric Q8 scales.
///
/// Layout is the `nn.Linear` layout the checkpoint stores: `data` is `[n, k]` row-major with one
/// f32 scale per output row. Quantized once at hydration; the borrowed f32 tensor is untouched.
#[derive(Clone, Debug)]
pub struct QuantizedMatrix {
    /// Q8 bytes, `[n, k]` row-major, each value in `[-127, 127]`.
    pub data: Vec<i8>,
    /// One symmetric scale per output row, `[n]`.
    pub scales: Vec<f32>,
    /// Output rows.
    pub n: usize,
    /// Reduction length of one output element.
    pub k: usize,
}

impl QuantizedMatrix {
    /// Quantizes an `[n, k]` f32 weight matrix one output channel at a time.
    ///
    /// # Panics
    ///
    /// Panics if `weight.len() != n * k` or any value is non-finite.
    #[must_use]
    pub fn quantize(weight: &[f32], n: usize, k: usize) -> Self {
        assert_eq!(weight.len(), n * k, "weight must be [n, k]");
        let mut data = vec![0_i8; n * k];
        let mut scales = vec![0.0_f32; n];
        for ((weight_row, data_row), scale) in weight
            .chunks_exact(k)
            .zip(data.chunks_exact_mut(k))
            .zip(scales.iter_mut())
        {
            *scale = quantize_row_q8(weight_row, data_row);
        }
        Self { data, scales, n, k }
    }
}

/// An executable int8 dot-product route.
///
/// Every variant is exactly equal in i32 to `Scalar` on every input. `NeonSdot` exists only on
/// aarch64 builds with the `neon-dotprod` feature and is dispatchable only where the CPU reports
/// FEAT_DotProd at runtime.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Int8Tier {
    /// Portable left-to-right checked-free scalar loop; the reference every tier must equal.
    Scalar,
    /// Portable eight-lane loop shaped for LLVM autovectorization (NE-INH-003's measured winner
    /// class at m=1 on Apple M4 — a live dispatch candidate, not just a fallback).
    Autovec,
    /// Hand SDOT island (aarch64 + FEAT_DotProd), four 16-byte accumulator streams.
    NeonSdot,
}

impl Int8Tier {
    /// Stable machine-readable route name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Scalar => "scalar",
            Self::Autovec => "autovec",
            Self::NeonSdot => "neon-sdot",
        }
    }

    /// Every tier this build can execute on the running silicon, scalar first.
    #[must_use]
    pub fn available() -> Vec<Self> {
        let mut tiers = vec![Self::Scalar, Self::Autovec];
        if neon_sdot_available() {
            tiers.push(Self::NeonSdot);
        }
        tiers
    }

    /// The route the int8 path dispatches by default, honoring the `FTTS_INT8_TIER` override.
    ///
    /// The override exists for interleaved A/B measurement (`scalar` / `autovec` / `neon-sdot`);
    /// an unavailable or unrecognized override falls back to the measured default rather than
    /// panicking mid-synthesis. Until a per-shape KernelPlan lands, the default is `NeonSdot`
    /// where FEAT_DotProd exists, else `Scalar`. Measured on M4 Pro (2026-08-08, shape bench,
    /// noisy shared host, indicative): plain `Scalar` autovectorizes to ~50 GB/s and ties SDOT
    /// at m=1 — NE-INH-003 reconfirmed — while the hand-shaped `Autovec` lane loop defeats the
    /// vectorizer and loses ~15x; it stays only as an A/B datapoint.
    #[must_use]
    pub fn dispatch() -> Self {
        match std::env::var("FTTS_INT8_TIER").as_deref() {
            Ok("scalar") => Self::Scalar,
            Ok("autovec") => Self::Autovec,
            Ok("neon-sdot") if neon_sdot_available() => Self::NeonSdot,
            _ if neon_sdot_available() => Self::NeonSdot,
            _ => Self::Scalar,
        }
    }
}

/// Whether the SDOT island is compiled in and the CPU reports FEAT_DotProd.
#[must_use]
pub fn neon_sdot_available() -> bool {
    #[cfg(all(target_arch = "aarch64", feature = "neon-dotprod"))]
    {
        neon_dotprod::available()
    }
    #[cfg(not(all(target_arch = "aarch64", feature = "neon-dotprod")))]
    {
        false
    }
}

/// Exact i32 dot product of two Q8 rows over the selected route.
///
/// # Panics
///
/// Panics if the lengths differ, or if `NeonSdot` is requested where it is not executable.
#[must_use]
pub fn dot_i32(a: &[i8], b: &[i8], tier: Int8Tier) -> i32 {
    assert_eq!(a.len(), b.len(), "int8 dot inputs must match");
    match tier {
        Int8Tier::Scalar => dot_i32_scalar(a, b),
        Int8Tier::Autovec => dot_i32_autovec(a, b),
        Int8Tier::NeonSdot => dot_i32_neon_or_panic(a, b),
    }
}

#[cfg(all(target_arch = "aarch64", feature = "neon-dotprod"))]
fn dot_i32_neon_or_panic(a: &[i8], b: &[i8]) -> i32 {
    assert!(
        neon_dotprod::available(),
        "neon-sdot route selected without FEAT_DotProd"
    );
    neon_dotprod::dot_i32(a, b)
}

#[cfg(not(all(target_arch = "aarch64", feature = "neon-dotprod")))]
fn dot_i32_neon_or_panic(_a: &[i8], _b: &[i8]) -> i32 {
    panic!("neon-sdot route selected on a build without the island");
}

fn dot_i32_scalar(a: &[i8], b: &[i8]) -> i32 {
    let mut sum = 0_i32;
    for index in 0..a.len() {
        sum += i32::from(a[index]) * i32::from(b[index]);
    }
    sum
}

/// Eight independent i32 lanes over fixed-width chunks; LLVM autovectorizes this shape into
/// widening multiply-accumulate sequences (and SDOT where the target baseline carries it).
/// Integer addition is associative, so the result is exactly [`dot_i32_scalar`]'s.
fn dot_i32_autovec(a: &[i8], b: &[i8]) -> i32 {
    const LANES: usize = 8;
    let mut lanes = [0_i32; LANES];
    let chunks = a.len() / LANES;
    for chunk in 0..chunks {
        let base = chunk * LANES;
        for lane in 0..LANES {
            lanes[lane] += i32::from(a[base + lane]) * i32::from(b[base + lane]);
        }
    }
    let mut sum: i32 = lanes.iter().sum();
    for index in chunks * LANES..a.len() {
        sum += i32::from(a[index]) * i32::from(b[index]);
    }
    sum
}

#[cfg(all(target_arch = "aarch64", feature = "neon-dotprod"))]
mod neon_dotprod {
    //! The audited SDOT island. Named per the crate law: feature-gated, runtime-detected,
    //! bit-identical scalar fallback in the parent module, every load bounds-checked by loop
    //! structure, every unsafe operation carrying a SAFETY note.

    use core::arch::aarch64::{vaddq_s32, vaddvq_s32, vdotq_s32, vdupq_n_s32, vld1q_s8};

    /// Whether the running CPU reports FEAT_DotProd.
    #[must_use]
    pub fn available() -> bool {
        std::arch::is_aarch64_feature_detected!("dotprod")
    }

    /// Exact i32 dot product via SDOT, four accumulator streams over 64-byte blocks.
    ///
    /// # Panics
    ///
    /// Panics (in the caller) unless [`available`] returned true; lengths are asserted equal by
    /// [`super::dot_i32`].
    #[must_use]
    pub fn dot_i32(a: &[i8], b: &[i8]) -> i32 {
        debug_assert!(available(), "SDOT island entered without FEAT_DotProd");
        // SAFETY: `dot_i32_sdot` requires NEON + FEAT_DotProd, which `available()` has confirmed
        // on this CPU at every dispatch site (asserted in `super::dot_i32`, debug-asserted here).
        unsafe { dot_i32_sdot(a, b) }
    }

    #[target_feature(enable = "neon,dotprod")]
    unsafe fn dot_i32_sdot(a: &[i8], b: &[i8]) -> i32 {
        let len = a.len();
        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();
        let mut acc0 = vdupq_n_s32(0);
        let mut acc1 = vdupq_n_s32(0);
        let mut acc2 = vdupq_n_s32(0);
        let mut acc3 = vdupq_n_s32(0);
        let mut index = 0_usize;
        while index + 64 <= len {
            // SAFETY: `index + 64 <= len` bounds all four 16-byte loads inside both slices,
            // whose lengths are equal by the caller's assertion. `vld1q_s8` has no alignment
            // requirement beyond byte alignment.
            unsafe {
                acc0 = vdotq_s32(acc0, vld1q_s8(a_ptr.add(index)), vld1q_s8(b_ptr.add(index)));
                acc1 = vdotq_s32(
                    acc1,
                    vld1q_s8(a_ptr.add(index + 16)),
                    vld1q_s8(b_ptr.add(index + 16)),
                );
                acc2 = vdotq_s32(
                    acc2,
                    vld1q_s8(a_ptr.add(index + 32)),
                    vld1q_s8(b_ptr.add(index + 32)),
                );
                acc3 = vdotq_s32(
                    acc3,
                    vld1q_s8(a_ptr.add(index + 48)),
                    vld1q_s8(b_ptr.add(index + 48)),
                );
            }
            index += 64;
        }
        while index + 16 <= len {
            // SAFETY: `index + 16 <= len` bounds this 16-byte load inside both slices.
            unsafe {
                acc0 = vdotq_s32(acc0, vld1q_s8(a_ptr.add(index)), vld1q_s8(b_ptr.add(index)));
            }
            index += 16;
        }
        let mut sum = vaddvq_s32(vaddq_s32(vaddq_s32(acc0, acc1), vaddq_s32(acc2, acc3)));
        while index < len {
            sum += i32::from(a[index]) * i32::from(b[index]);
            index += 1;
        }
        sum
    }
}

/// W8A8 linear: quantized activations `[m, k]` times a [`QuantizedMatrix`] `[n, k]`, producing
/// f32 `[m, n]`.
///
/// `x_scales` carries one dynamic activation scale per row of `x_q`. The i32 accumulator is
/// exact on every tier; dequantization applies `acc as f32 * (x_scale * w_scale)` in exactly
/// that order on every tier, so the f32 output of any two tiers is bit-identical, not merely
/// close. Bias (only `text_projection` carries one) is added after dequantization.
///
/// # Panics
///
/// Panics on any shape mismatch.
#[allow(clippy::too_many_arguments)]
pub fn linear_q8(
    x_q: &[i8],
    x_scales: &[f32],
    weight: &QuantizedMatrix,
    bias: Option<&[f32]>,
    m: usize,
    out: &mut [f32],
    tier: Int8Tier,
) {
    let (n, k) = (weight.n, weight.k);
    assert_eq!(x_q.len(), m * k, "x_q must be [m, k]");
    assert_eq!(x_scales.len(), m, "x_scales must be [m]");
    assert_eq!(out.len(), m * n, "out must be [m, n]");
    if let Some(bias) = bias {
        assert_eq!(bias.len(), n, "bias must be [n]");
    }
    for row in 0..m {
        let x_row = &x_q[row * k..(row + 1) * k];
        let x_scale = x_scales[row];
        let out_row = &mut out[row * n..(row + 1) * n];
        for col in 0..n {
            let w_row = &weight.data[col * k..(col + 1) * k];
            let acc = dot_i32(x_row, w_row, tier);
            let value = acc as f32 * (x_scale * weight.scales[col]);
            out_row[col] = bias.map_or(value, |b| value + b[col]);
        }
    }
}

/// Quantizes an f32 activation matrix `[m, k]` per row and runs [`linear_q8`].
///
/// This is the drop-in W8A8 counterpart of `f32ref::linear`: same `[m, k] × [n, k]ᵀ → [m, n]`
/// layout, same bias placement. The row quantization is the canonical symmetric recipe.
///
/// # Panics
///
/// Panics on any shape mismatch or a non-finite activation.
pub fn linear_q8_dynamic(
    x: &[f32],
    weight: &QuantizedMatrix,
    bias: Option<&[f32]>,
    m: usize,
    out: &mut [f32],
    tier: Int8Tier,
) {
    let k = weight.k;
    assert_eq!(x.len(), m * k, "x must be [m, k]");
    let mut x_q = vec![0_i8; m * k];
    let mut x_scales = vec![0.0_f32; m];
    for ((x_row, q_row), scale) in x
        .chunks_exact(k)
        .zip(x_q.chunks_exact_mut(k))
        .zip(x_scales.iter_mut())
    {
        *scale = quantize_row_q8(x_row, q_row);
    }
    linear_q8(&x_q, &x_scales, weight, bias, m, out, tier);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic pseudo-random Q8 bytes (SplitMix64), full `[-127, 127]` range.
    fn pseudo_random_q8(len: usize, seed: u64) -> Vec<i8> {
        let mut state = seed;
        (0..len)
            .map(|_| {
                state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
                let mut z = state;
                z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
                z ^= z >> 31;
                // Map to [-127, 127]; never -128, matching the converter contract.
                ((z % 255) as i32 - 127) as i8
            })
            .collect()
    }

    /// The model's real decode GEMV shapes: (n, k) per §7 of the plan.
    const MODEL_SHAPES: &[(usize, usize)] = &[
        (2048, 1024), // q_proj / per-depth heads
        (1024, 1024), // k_proj / v_proj
        (1024, 2048), // o_proj
        (3072, 1024), // gate/up_proj, primary head
        (1024, 3072), // down_proj (binding talker K)
    ];

    #[test]
    fn every_tier_is_exactly_equal_in_i32_at_every_model_shape() {
        for &(n, k) in MODEL_SHAPES {
            let a = pseudo_random_q8(k, 0x5eed_0001 ^ (n as u64) << 20 ^ k as u64);
            let w = pseudo_random_q8(n * k, 0x5eed_0002 ^ (n as u64) << 20 ^ k as u64);
            for row in [0, n / 2, n - 1] {
                let w_row = &w[row * k..(row + 1) * k];
                let reference = dot_i32(&a, w_row, Int8Tier::Scalar);
                for tier in Int8Tier::available() {
                    assert_eq!(
                        dot_i32(&a, w_row, tier),
                        reference,
                        "tier {} diverged at shape {n}x{k} row {row}",
                        tier.as_str()
                    );
                }
            }
        }
    }

    #[test]
    fn every_tier_survives_the_all_extreme_reduction_at_the_binding_census_k() {
        // 127 * 127 * 8192 = 132,120,576 — the S8S8 all-extreme envelope at the largest census K.
        for k in [2048_usize, 3072, 4608, 7168, 8192] {
            let a = vec![127_i8; k];
            let b = vec![127_i8; k];
            let negative = vec![-127_i8; k];
            let expected = 127_i64 * 127 * k as i64;
            for tier in Int8Tier::available() {
                assert_eq!(
                    i64::from(dot_i32(&a, &b, tier)),
                    expected,
                    "positive all-extreme diverged on {} at K={k}",
                    tier.as_str()
                );
                assert_eq!(
                    i64::from(dot_i32(&a, &negative, tier)),
                    -expected,
                    "negative all-extreme diverged on {} at K={k}",
                    tier.as_str()
                );
            }
        }
    }

    #[test]
    fn tail_lengths_that_defeat_block_boundaries_stay_exact() {
        // Exercise every SDOT path: <16 (pure tail), 16..64 (single-block loop), 64+tail.
        for len in [1_usize, 7, 15, 16, 17, 63, 64, 65, 100, 129] {
            let a = pseudo_random_q8(len, tail_seed(len));
            let b = pseudo_random_q8(len, tail_seed(len) ^ 1);
            let reference = dot_i32(&a, &b, Int8Tier::Scalar);
            for tier in Int8Tier::available() {
                assert_eq!(
                    dot_i32(&a, &b, tier),
                    reference,
                    "len={len} {}",
                    tier.as_str()
                );
            }
        }
    }

    #[test]
    fn quantizer_matches_the_canonical_converter_semantics() {
        // Ties-to-even, clamp, zero-row scale, and the -128 exclusion. The cross-crate
        // byte-identity test against `ftts-artifacts` lives in `ftts-model-qwen`.
        let row = [
            -127.0_f32, -126.5, -125.5, -1.5, -0.5, 0.5, 1.5, 125.5, 126.5, 127.0,
        ];
        let mut q = [0_i8; 10];
        let scale = quantize_row_q8(&row, &mut q);
        assert_eq!(scale.to_bits(), 1.0_f32.to_bits());
        assert_eq!(q, [-127, -126, -126, -2, 0, 0, 2, 126, 126, 127]);

        let zeros = [0.0_f32; 4];
        let mut qz = [1_i8; 4];
        assert_eq!(
            quantize_row_q8(&zeros, &mut qz).to_bits(),
            1.0_f32.to_bits()
        );
        assert_eq!(qz, [0, 0, 0, 0]);

        let matrix = QuantizedMatrix::quantize(&[2.0, -1.0, 0.0, 3.0], 2, 2);
        assert_eq!(matrix.scales[0].to_bits(), (2.0_f32 / 127.0).to_bits());
        assert_eq!(matrix.scales[1].to_bits(), (3.0_f32 / 127.0).to_bits());
        assert!(matrix.data.iter().all(|&b| b != -128));
    }

    #[test]
    fn dynamic_w8a8_linear_tracks_the_f32_reference_within_quant_error() {
        // Not a parity claim — a sanity bound that the dequant plumbing is wired correctly.
        let (n, k) = (64_usize, 128_usize);
        let mut weight = vec![0.0_f32; n * k];
        let mut x = vec![0.0_f32; k];
        let mut state = 0x1234_5678_u64;
        let mut next = || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            ((state >> 33) as f32 / (1u64 << 31) as f32) - 1.0
        };
        for value in weight.iter_mut() {
            *value = next();
        }
        for value in x.iter_mut() {
            *value = next();
        }
        let quantized = QuantizedMatrix::quantize(&weight, n, k);
        let mut out_q8 = vec![0.0_f32; n];
        linear_q8_dynamic(&x, &quantized, None, 1, &mut out_q8, Int8Tier::Autovec);

        let mut out_f32 = vec![0.0_f32; n];
        crate::f32ref::linear(&x, &weight, None, 1, k, n, &mut out_f32);

        let dot = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>();
        let cosine = dot(&out_q8, &out_f32)
            / (dot(&out_q8, &out_q8).sqrt() * dot(&out_f32, &out_f32).sqrt());
        assert!(
            cosine > 0.999,
            "W8A8 dequant plumbing is broken: cosine {cosine}"
        );
    }

    #[test]
    fn tiers_produce_bit_identical_f32_output_not_merely_close() {
        let (n, k) = (256_usize, 1024_usize);
        let weight: Vec<f32> = pseudo_random_q8(n * k, 77)
            .iter()
            .map(|&b| f32::from(b) / 64.0)
            .collect();
        let x: Vec<f32> = pseudo_random_q8(k, 78)
            .iter()
            .map(|&b| f32::from(b) / 64.0)
            .collect();
        let quantized = QuantizedMatrix::quantize(&weight, n, k);
        let mut reference = vec![0.0_f32; n];
        linear_q8_dynamic(&x, &quantized, None, 1, &mut reference, Int8Tier::Scalar);
        for tier in Int8Tier::available() {
            let mut out = vec![0.0_f32; n];
            linear_q8_dynamic(&x, &quantized, None, 1, &mut out, tier);
            for (index, (a, b)) in reference.iter().zip(&out).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "tier {} f32 output differs at {index}",
                    tier.as_str()
                );
            }
        }
    }

    fn tail_seed(len: usize) -> u64 {
        0x7a11_0000 ^ len as u64
    }
}
