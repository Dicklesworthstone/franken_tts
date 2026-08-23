//! W4A8 int4 weights for the microdecoder — packed two-per-byte, unpacked in registers.
//!
//! # Why the microdecoder, and why now
//!
//! Doctrine #2 sends int4 to the microdecoder FIRST, and the measurement now agrees. Its 5-layer
//! body is re-read **fifteen times per frame** — the single largest repeated read in the model —
//! so halving its weight bytes attacks the one place cache residency is plausibly winnable:
//! roughly 79 MB of Q8 becomes ~40 MB of Q4, which is the difference between spilling to DRAM
//! every depth step and staying resident across all fifteen.
//!
//! Until 2026-08-10 this was a rounding error: the codec was 92% of browser frame time and the
//! talker+microdecoder 7.9%. After the packed GEMM and the kernel team took the codec down 13x,
//! the split is codec 65% / talker+micro 33% — so this now targets a third of the frame.
//!
//! # The quantization contract, and how it differs from Q8
//!
//! Symmetric, per-output-channel, ties-to-even — the same shape as
//! [`crate::int8::quantize_row_q8`], with one deliberate asymmetry preserved: the most negative
//! representable value is never emitted. Q8 excludes -128 and keeps [-127, 127]; Q4 excludes -8
//! and keeps **[-7, 7]**. That symmetry is what makes `-w` exactly representable whenever `w` is,
//! so negating a row negates its quantization exactly, and it keeps the accumulator's worst case
//! symmetric.
//!
//! The cost is real and must not be glossed: 15 levels instead of 255. Quantization error is ~17x
//! larger per weight, which is precisely why doctrine #2 gates this behind BOTH a per-ISA speed
//! test that includes unpack cost AND a blind-listening equivalence test. **This module ships the
//! arithmetic, not the decision.** Nothing routes to it until those gates are run.
//!
//! # Packing
//!
//! Two nibbles per byte, low nibble first, along `k`. A row of odd length pads its final high
//! nibble with the BIASED zero (`8`), not a raw `0`.
//!
//! That distinction matters and is easy to get backwards. A raw `0` nibble decodes to `0 - 8 = -8`,
//! the largest negative weight in the range — so zero-initialized padding is not neutral, it is
//! maximally *non*-neutral. Today nothing reads past `k` and it would not matter, but the entire
//! reason to store int4 is a future SIMD unpack that processes whole BYTES, and such a kernel would
//! silently fold that `-8` into the last accumulator. Padding with the biased zero makes the pad
//! decode to `0.0` and keeps any whole-byte kernel correct by construction.
//!
//! Storing the nibble biased by +8 (so [-7, 7] becomes [1, 15]) makes unpacking a shift-and-mask
//! with no sign extension, and the bias cancels exactly in the dot product — see
//! [`dot_i32_q4`], where it becomes a single correction term computed from the activation sum.

/// Nibbles per packed byte.
const PER_BYTE: usize = 2;

/// The bias added to every nibble so the stored value is unsigned `[1, 15]`.
///
/// Chosen so unpacking never needs sign extension: `(byte & 0xF) as i32 - BIAS` recovers the
/// signed weight with one subtract, and across a whole dot product the subtraction can be hoisted
/// into one correction term rather than paid per element.
const BIAS: i32 = 8;

/// A weight matrix quantized to symmetric int4, packed two values per byte.
///
/// Layout mirrors [`crate::int8::QuantizedMatrix`]: `[n, k]` row-major in the checkpoint's own
/// `nn.Linear` orientation, one f32 scale per output channel, so no transpose is ever materialized.
#[derive(Clone, Debug, PartialEq)]
pub struct QuantizedMatrixQ4 {
    /// `n * k.div_ceil(2)` bytes: row-major, two biased nibbles per byte, low nibble first.
    pub data: Vec<u8>,
    /// One scale per output channel.
    pub scales: Vec<f32>,
    pub n: usize,
    pub k: usize,
}

impl QuantizedMatrixQ4 {
    /// Quantizes an `[n, k]` f32 weight matrix.
    ///
    /// # Panics
    ///
    /// If `weight.len() != n * k`, or a weight is non-finite — a NaN reaching the quantizer means
    /// the graph upstream is already corrupt, and refusing loudly beats baking it into an artifact.
    #[must_use]
    pub fn quantize(weight: &[f32], n: usize, k: usize) -> Self {
        assert_eq!(weight.len(), n * k, "weight must be [n, k]");
        let packed_row = k.div_ceil(PER_BYTE);
        // Initialized to the BIASED zero in both nibbles (`0x88`), never to `0x00`: an unwritten
        // nibble must decode to 0.0, and a raw zero nibble decodes to -8. See the packing note in
        // the module docs.
        let mut data = vec![0x88_u8; n * packed_row];
        let mut scales = Vec::with_capacity(n);

        for row in 0..n {
            let source = &weight[row * k..row * k + k];
            let mut maximum = 0.0_f32;
            for (index, &value) in source.iter().enumerate() {
                assert!(
                    value.is_finite(),
                    "non-finite value {value} at index {index} reached the Q4 quantizer"
                );
                maximum = maximum.max(value.abs());
            }
            // A zero row quantizes to the zero row it already is; scale 1.0 keeps the dequantized
            // result exactly zero rather than introducing a NaN through a zero divisor.
            let scale = if maximum == 0.0 { 0.0 } else { maximum / 7.0 };
            scales.push(if scale == 0.0 { 1.0 } else { scale });

            let target = &mut data[row * packed_row..(row + 1) * packed_row];
            if scale == 0.0 {
                // Every nibble is the biased zero, so the row dequantizes to exact zeros.
                target.fill(((BIAS as u8) << 4) | BIAS as u8);
                continue;
            }
            for (index, &value) in source.iter().enumerate() {
                // Ties-to-even and a clamp that excludes -8, matching the Q8 contract's exclusion
                // of -128: symmetry is what makes negation exact.
                let level = (value / scale).clamp(-7.0, 7.0).round_ties_even() as i32;
                let biased = (level + BIAS) as u8;
                let byte = &mut target[index / PER_BYTE];
                if index % PER_BYTE == 0 {
                    *byte = (*byte & 0xF0) | biased;
                } else {
                    *byte = (*byte & 0x0F) | (biased << 4);
                }
            }
            // An odd k leaves the final high nibble as the buffer's initialization value,
            // which is the BIASED zero (0x88 fills both nibbles), so the pad decodes to
            // exactly 0.0 — see the init comment above and the module docs. No current
            // reader touches it (`dot_i32_q4` and `dequantize_row` both stop at k), and a
            // future whole-byte SIMD kernel may consume it freely as long as the matching
            // activation pad is 0, since biased-zero times anything contributes nothing.
        }

        Self { data, scales, n, k }
    }

    /// Dequantizes one output channel back to f32, for parity comparison against the f32 weights.
    #[must_use]
    pub fn dequantize_row(&self, row: usize) -> Vec<f32> {
        let packed_row = self.k.div_ceil(PER_BYTE);
        let bytes = &self.data[row * packed_row..(row + 1) * packed_row];
        let scale = self.scales[row];
        (0..self.k)
            .map(|index| {
                let byte = bytes[index / PER_BYTE];
                let nibble = if index % PER_BYTE == 0 {
                    i32::from(byte & 0x0F)
                } else {
                    i32::from(byte >> 4)
                };
                #[allow(clippy::cast_precision_loss)]
                {
                    (nibble - BIAS) as f32 * scale
                }
            })
            .collect()
    }

    /// Bytes of weight storage, the number this lever exists to shrink.
    #[must_use]
    pub fn packed_bytes(&self) -> usize {
        self.data.len()
    }
}

/// Exact i32 dot product of an int8 activation row against one packed int4 weight row.
///
/// # The bias cancellation, which is the whole trick
///
/// Each stored nibble is `w + 8`. Expanding the dot product:
///
/// ```text
///   sum_i x[i] * w[i]  ==  sum_i x[i] * (nibble[i] - 8)
///                      ==  sum_i x[i] * nibble[i]  -  8 * sum_i x[i]
/// ```
///
/// So the per-element subtraction disappears: accumulate against the *unsigned* nibbles, then
/// apply one correction of `8 * sum(x)` at the end. That leaves the inner loop as mask, shift,
/// multiply-add — no sign extension, no per-element bias — which is what makes unpacking cheap
/// enough to be worth the halved bytes (NE-INH-004's escape clause is exactly this: int4 pays off
/// only if the unpack folds into the MAC).
///
/// Accumulation is exact i32 throughout; scales are applied once by the caller, after.
///
/// # Panics
///
/// If `packed` is too short for `k` values.
#[must_use]
pub fn dot_i32_q4(x: &[i8], packed: &[u8], k: usize) -> i32 {
    dot_i32_q4_tier(x, packed, k, Q4Tier::dispatch())
}

/// [`linear_q4`] through an explicit tier, for interleaved A/B measurement.
pub fn linear_q4_tier(
    x_q: &[i8],
    x_scales: &[f32],
    weight: &QuantizedMatrixQ4,
    bias: Option<&[f32]>,
    m: usize,
    out: &mut [f32],
    tier: Q4Tier,
) {
    let (n, k) = (weight.n, weight.k);
    assert_eq!(x_q.len(), m * k, "activations must be [m, k]");
    assert_eq!(out.len(), m * n, "out must be [m, n]");
    let packed_row = k.div_ceil(PER_BYTE);
    for row in 0..m {
        let x_row = &x_q[row * k..row * k + k];
        for column in 0..n {
            let w_row = &weight.data[column * packed_row..(column + 1) * packed_row];
            let accumulated = dot_i32_q4_tier(x_row, w_row, k, tier);
            #[allow(clippy::cast_precision_loss)]
            let value = accumulated as f32 * (x_scales[row] * weight.scales[column]);
            out[row * n + column] = bias.map_or(value, |values| value + values[column]);
        }
    }
}

#[cfg(all(target_arch = "aarch64", feature = "neon-dotprod"))]
mod neon_q4 {
    //! The audited in-register unpack island for W4A8 — NE-INH-004's escape clause and
    //! NE-005's sole retry predicate. One 16-byte load per packed block; nibbles split by
    //! shift/mask INSIDE the vector registers; both lanes feed SDOT against the signed
    //! activations. The +8 storage bias cancels exactly as in the scalar path (nibbles
    //! accumulate unsigned; one `8*sum(x)` correction at the end), so the result is
    //! bit-identical to [`super::dot_i32_q4_scalar_from`] by integer exactness.
    //!
    //! # The deinterleave that makes it correct
    //!
    //! Packed byte j holds elements (2j, 2j+1): low nibbles are the EVEN activations of a
    //! block's 32-element span, high nibbles the ODD ones. A contiguous activation load
    //! would therefore mispair every element with its neighbor's nibble. Two loads plus a
    //! `vuzp1q`/`vuzp2q` deinterleave produce exactly the even and odd lanes SDOT needs;
    //! the final horizontal add makes lane grouping irrelevant to the exact total.

    use super::{BIAS, PER_BYTE};
    use core::arch::aarch64::{
        vaddlvq_s8, vaddvq_s32, vandq_u8, vdotq_s32, vdupq_n_s32, vdupq_n_u8, vld1q_s8, vld1q_u8,
        vreinterpretq_s8_u8, vshrq_n_u8, vuzp1q_s8, vuzp2q_s8,
    };
    /// Whether the running CPU reports FEAT_DotProd (SDOT is this island's MAC).
    #[must_use]
    pub fn available() -> bool {
        std::arch::is_aarch64_feature_detected!("dotprod")
    }

    /// Exact i32 biased-nibble dot via in-register unpack + SDOT.
    ///
    /// # Panics
    ///
    /// Panics (in the caller) unless [`available`] returned true; length relations are
    /// asserted by [`super::dot_i32_q4_tier`].
    #[must_use]
    pub fn dot_i32(x: &[i8], packed: &[u8], k: usize) -> i32 {
        debug_assert!(available(), "q4 neon island entered without FEAT_DotProd");
        // SAFETY: `dot_i32_sdot` requires NEON + FEAT_DotProd, which `available()` has
        // confirmed on this CPU at the dispatch site (asserted in the caller).
        unsafe { dot_i32_sdot(x, packed, k) }
    }

    // SAFETY: callers must have confirmed FEAT_DotProd via `available()`. All loads are
    // bounded by loop structure: packed blocks run while `block*16 + 16 <= pairs` where
    // `pairs <= packed.len()`, and each block's two x loads touch `[base, base+32)` with
    // `base + 32 = block*32 + 32 <= pairs*2 <= k <= x.len()`; the scalar tail touches
    // nothing past `k`. No read passes either slice's end.
    #[target_feature(enable = "neon,dotprod")]
    unsafe fn dot_i32_sdot(x: &[i8], packed: &[u8], k: usize) -> i32 {
        let pairs = k / PER_BYTE;
        let blocks = pairs / 16;
        let low_mask = vdupq_n_u8(0x0F);
        let mut unsigned_acc = vdupq_n_s32(0);
        let mut activation_sum = 0_i32;

        let x_ptr = x.as_ptr();
        let packed_ptr = packed.as_ptr();
        let mut block = 0_usize;
        while block < blocks {
            let byte_base = block * 16;
            let x_base = block * 32;
            // SAFETY: see the function-level note — both loads are bounded by the loop
            // condition and the caller's length assertions; NEON loads need no alignment.
            unsafe {
                let bytes = vld1q_u8(packed_ptr.add(byte_base));
                let low_nibbles = vandq_u8(bytes, low_mask);
                let high_nibbles = vshrq_n_u8(bytes, 4);
                let x_first = vld1q_s8(x_ptr.add(x_base));
                let x_second = vld1q_s8(x_ptr.add(x_base + 16));
                let evens = vuzp1q_s8(x_first, x_second);
                let odds = vuzp2q_s8(x_first, x_second);
                unsigned_acc = vdotq_s32(unsigned_acc, evens, vreinterpretq_s8_u8(low_nibbles));
                unsigned_acc = vdotq_s32(unsigned_acc, odds, vreinterpretq_s8_u8(high_nibbles));
                activation_sum += i32::from(vaddlvq_s8(evens)) + i32::from(vaddlvq_s8(odds));
            }
            block += 1;
        }

        let mut unsigned_total = vaddvq_s32(unsigned_acc);
        for pair in blocks * 16..pairs {
            let byte = packed[pair];
            let low = i32::from(byte & 0x0F);
            let high = i32::from(byte >> 4);
            let first = i32::from(x[pair * PER_BYTE]);
            let second = i32::from(x[pair * PER_BYTE + 1]);
            unsigned_total += first * low + second * high;
            activation_sum += first + second;
        }
        if k % PER_BYTE == 1 {
            let byte = packed[pairs];
            let value = i32::from(x[k - 1]);
            unsigned_total += value * i32::from(byte & 0x0F);
            activation_sum += value;
        }

        unsigned_total - BIAS * activation_sum
    }
}

/// Dispatchable implementations of [`dot_i32_q4`]'s arithmetic, for interleaved gate
/// measurement and so the SIMD island can be exercised independently of autodetection.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Q4Tier {
    /// The plain mask/shift/MAC loop; LLVM's autovectorizer provably cannot fold it
    /// (NE-005 measured it 22x under SDOT int8 at microdecoder shapes).
    Scalar,
    /// In-register unpack on FEAT_DotProd hardware: one 16-byte load per packed block,
    /// nibbles split by shift/mask inside the vector registers, fed straight to SDOT.
    NeonSdot,
}

impl Q4Tier {
    /// Human-readable route name, matching the Int8Tier reporting convention.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Scalar => "scalar",
            Self::NeonSdot => "neon-sdot",
        }
    }

    /// Every tier this build can execute on this machine.
    #[must_use]
    pub fn available() -> Vec<Self> {
        // The mut is structural: the vector extends only under the aarch64+dotprod cfg,
        // so scalar-only targets would otherwise flag it.
        #[allow(unused_mut)]
        let mut tiers = vec![Self::Scalar];
        #[cfg(all(target_arch = "aarch64", feature = "neon-dotprod"))]
        if neon_q4::available() {
            tiers.push(Self::NeonSdot);
        }
        tiers
    }

    /// The fastest available tier. Auto-routing inside this module changes nothing about
    /// doctrine #2's routing gate: nothing outside the module constructs a
    /// [`QuantizedMatrixQ4`], so production selection remains gated exactly as before.
    #[must_use]
    pub fn dispatch() -> Self {
        #[cfg(all(target_arch = "aarch64", feature = "neon-dotprod"))]
        if neon_q4::available() {
            return Self::NeonSdot;
        }
        Self::Scalar
    }
}

/// [`dot_i32_q4`] through an explicit tier, for interleaved A/B measurement.
///
/// # Panics
///
/// Panics when the selected tier is unavailable on this machine.
#[must_use]
pub fn dot_i32_q4_tier(x: &[i8], packed: &[u8], k: usize, tier: Q4Tier) -> i32 {
    assert!(
        packed.len() >= k.div_ceil(PER_BYTE),
        "packed row shorter than k nibbles"
    );
    assert!(x.len() >= k, "activation row shorter than k");
    match tier {
        Q4Tier::Scalar => dot_i32_q4_scalar_from(x, packed, k),
        #[cfg(all(target_arch = "aarch64", feature = "neon-dotprod"))]
        Q4Tier::NeonSdot => {
            assert!(
                neon_q4::available(),
                "neon-sdot q4 route selected without FEAT_DotProd"
            );
            neon_q4::dot_i32(x, packed, k)
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon-dotprod")))]
        Q4Tier::NeonSdot => unreachable!("neon-sdot q4 tier not built into this target"),
    }
}

/// The scalar reference every other tier must reproduce bit-for-bit: the original
/// mask/shift/MAC loop with the single end-of-row bias correction.
#[must_use]
fn dot_i32_q4_scalar_from(x: &[i8], packed: &[u8], k: usize) -> i32 {
    let mut unsigned_accumulator = 0_i32;
    let mut activation_sum = 0_i32;

    let pairs = k / PER_BYTE;
    for pair in 0..pairs {
        let byte = packed[pair];
        let low = i32::from(byte & 0x0F);
        let high = i32::from(byte >> 4);
        let first = i32::from(x[pair * PER_BYTE]);
        let second = i32::from(x[pair * PER_BYTE + 1]);
        unsigned_accumulator += first * low + second * high;
        activation_sum += first + second;
    }
    if k % PER_BYTE == 1 {
        let byte = packed[pairs];
        let value = i32::from(x[k - 1]);
        unsigned_accumulator += value * i32::from(byte & 0x0F);
        activation_sum += value;
    }

    unsigned_accumulator - BIAS * activation_sum
}

/// W4A8 linear: `out[m, n] = x[m, k] @ weight^T`, scales applied once per element.
///
/// # Panics
///
/// On shape mismatch.
pub fn linear_q4(
    x_q: &[i8],
    x_scales: &[f32],
    weight: &QuantizedMatrixQ4,
    bias: Option<&[f32]>,
    m: usize,
    out: &mut [f32],
) {
    let (n, k) = (weight.n, weight.k);
    assert_eq!(x_q.len(), m * k, "activations must be [m, k]");
    assert_eq!(x_scales.len(), m, "one activation scale per row");
    assert_eq!(out.len(), m * n, "out must be [m, n]");
    // Every sibling kernel pins this; without it a too-long bias is silently
    // prefix-consumed and a too-short one panics mid-row with a bare index message.
    if let Some(bias) = bias {
        assert_eq!(bias.len(), n, "bias must have one entry per output channel");
    }
    let packed_row = k.div_ceil(PER_BYTE);

    for row in 0..m {
        let x_row = &x_q[row * k..row * k + k];
        for column in 0..n {
            let w_row = &weight.data[column * packed_row..(column + 1) * packed_row];
            let accumulated = dot_i32_q4(x_row, w_row, k);
            #[allow(clippy::cast_precision_loss)]
            let value = accumulated as f32 * (x_scales[row] * weight.scales[column]);
            out[row * n + column] = bias.map_or(value, |values| value + values[column]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::int8::quantize_row_q8;

    fn deterministic(count: usize, seed: u64) -> Vec<f32> {
        let mut state = seed | 1;
        (0..count)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                ((state >> 40) as f32 / 8192.0) - 0.5
            })
            .collect()
    }

    /// The bias-cancellation identity must hold EXACTLY, not approximately.
    ///
    /// This is the load-bearing claim of the whole module: accumulating against biased nibbles and
    /// correcting once at the end must equal the straightforward signed dot. If it drifts by even
    /// one integer the error is silent and shows up as audio artifacts much later.
    #[test]
    fn bias_cancellation_is_exact_against_a_signed_reference() {
        for (index, &k) in [1_usize, 2, 3, 7, 8, 15, 64, 127, 1024].iter().enumerate() {
            let weight = deterministic(k, 0x4B1A_0000 + index as u64);
            let matrix = QuantizedMatrixQ4::quantize(&weight, 1, k);
            let activation = deterministic(k, 0xA0C7_0000 + index as u64);
            let mut x_q = vec![0_i8; k];
            quantize_row_q8(&activation, &mut x_q);

            // Reference: dequantize the nibbles to signed levels and dot them plainly.
            let packed_row = k.div_ceil(PER_BYTE);
            let mut expected = 0_i32;
            for (position, &activation) in x_q.iter().enumerate().take(k) {
                let byte = matrix.data[position / PER_BYTE];
                let nibble = if position % PER_BYTE == 0 {
                    i32::from(byte & 0x0F)
                } else {
                    i32::from(byte >> 4)
                };
                expected += i32::from(activation) * (nibble - BIAS);
            }
            assert_eq!(
                dot_i32_q4(&x_q, &matrix.data[..packed_row], k),
                expected,
                "k={k}: biased accumulation with a single correction diverged from the signed dot"
            );
        }
    }

    /// Every quantized level stays inside the symmetric range, and -8 is never emitted.
    #[test]
    fn levels_are_symmetric_and_never_emit_negative_eight() {
        let k = 4096;
        // Deliberately includes the extremes and values that round exactly onto .5 boundaries.
        let mut weight = deterministic(k, 0xD00D);
        weight[0] = 1.0;
        weight[1] = -1.0;
        weight[2] = 0.0;
        let matrix = QuantizedMatrixQ4::quantize(&weight, 1, k);
        let scale = matrix.scales[0];
        for position in 0..k {
            let byte = matrix.data[position / PER_BYTE];
            let nibble = if position % PER_BYTE == 0 {
                i32::from(byte & 0x0F)
            } else {
                i32::from(byte >> 4)
            };
            let level = nibble - BIAS;
            assert!(
                (-7..=7).contains(&level),
                "level {level} outside the symmetric range at {position}"
            );
        }
        // Negation must be exact, which is the property the -8 exclusion buys.
        let negated: Vec<f32> = weight.iter().map(|value| -value).collect();
        let mirror = QuantizedMatrixQ4::quantize(&negated, 1, k);
        assert!((mirror.scales[0] - scale).abs() <= f32::EPSILON * scale.max(1.0));
        // Value equality, not bit equality: a zero weight dequantizes to +0.0 on both sides, and
        // `-(+0.0)` is `-0.0`, whose bits differ while the value does not. For every non-zero
        // level, f32 equality here is still exact — the levels are small integers times a shared
        // scale, so no rounding can hide a mismatch.
        let forward = matrix.dequantize_row(0);
        let backward = mirror.dequantize_row(0);
        for position in 0..k {
            assert!(
                forward[position] == -backward[position],
                "negation was not exact at {position}: {} vs {}",
                forward[position],
                backward[position]
            );
        }
    }

    /// The padding nibble of an odd-length row must decode to ZERO, not to -8.
    ///
    /// Nothing reads past `k` today, so this cannot bite yet. It is pinned because the point of
    /// int4 is a SIMD unpack over whole bytes, and such a kernel WILL read the pad; if it decodes
    /// to -8 the last accumulator is silently wrong in a way no shape or size check would catch.
    #[test]
    fn the_padding_nibble_of_an_odd_row_is_a_neutral_zero() {
        for k in [1_usize, 3, 5, 7, 65] {
            let weight = deterministic(k, 0x0DD0_0000 + k as u64);
            let matrix = QuantizedMatrixQ4::quantize(&weight, 1, k);
            let last = *matrix.data.last().expect("a packed row");
            let pad = i32::from(last >> 4) - BIAS;
            assert_eq!(pad, 0, "k={k}: padding nibble decodes to {pad}, not 0");
        }
    }

    /// Q4 must be materially smaller than Q8 — the entire premise of the lever.
    #[test]
    fn packed_storage_is_half_of_q8() {
        let (n, k) = (2048, 1024);
        let weight = deterministic(n * k, 0xFEED);
        let q4 = QuantizedMatrixQ4::quantize(&weight, n, k);
        assert_eq!(q4.packed_bytes(), n * k / 2);
        let q8 = crate::int8::QuantizedMatrix::quantize(&weight, n, k);
        assert_eq!(q4.packed_bytes() * 2, q8.data.len());
    }

    /// The packed layout decodes identically through the kernel and through a naive
    /// independent unpacking — bit-for-bit, no tolerance. (The loose-tolerance prose
    /// that used to sit here described `quantization_error_is_bounded_by_the_level_step`
    /// below, not this test.)
    #[test]
    fn linear_q4_matches_an_independent_nibble_unpack_bit_for_bit() {
        // `linear_q4` itself had zero coverage: `dot_i32_q4` was tested, but not the packed
        // row stride, the multi-row walk, the activation scales, or the bias path. The
        // reference here unpacks nibbles directly from storage (independently of
        // `dot_i32_q4`) and reproduces the kernel's exact arithmetic order, so equality is
        // bit-for-bit, not approximate. Odd k exercises the padded final nibble.
        for (m, n, k, seed) in [
            (1_usize, 4_usize, 16_usize, 1_u64),
            (2, 5, 13, 2),
            (3, 7, 31, 3),
        ] {
            let weight_f32 = deterministic(n * k, seed * 100 + 7);
            let weight = QuantizedMatrixQ4::quantize(&weight_f32, n, k);
            let bias: Vec<f32> = deterministic(n, seed * 100 + 11);
            let mut x_q = Vec::with_capacity(m * k);
            let mut x_scales = Vec::with_capacity(m);
            for row in 0..m {
                let activation = deterministic(k, seed * 100 + 13 + row as u64);
                let mut quantized = vec![0_i8; k];
                let scale = quantize_row_q8(&activation, &mut quantized);
                x_q.extend_from_slice(&quantized);
                x_scales.push(scale);
            }

            let mut out = vec![0.0_f32; m * n];
            linear_q4(&x_q, &x_scales, &weight, Some(&bias), m, &mut out);

            let packed_row = k.div_ceil(PER_BYTE);
            for row in 0..m {
                for column in 0..n {
                    let bytes = &weight.data[column * packed_row..(column + 1) * packed_row];
                    let mut accumulated = 0_i32;
                    for index in 0..k {
                        let byte = bytes[index / PER_BYTE];
                        let nibble = if index % PER_BYTE == 0 {
                            i32::from(byte & 0x0F)
                        } else {
                            i32::from(byte >> 4)
                        };
                        accumulated += i32::from(x_q[row * k + index]) * (nibble - BIAS);
                    }
                    #[allow(clippy::cast_precision_loss)]
                    let expected =
                        accumulated as f32 * (x_scales[row] * weight.scales[column]) + bias[column];
                    assert_eq!(
                        out[row * n + column].to_bits(),
                        expected.to_bits(),
                        "m={m} n={n} k={k} row={row} column={column}"
                    );
                }
            }
        }
    }

    #[test]
    #[should_panic(expected = "bias must have one entry per output channel")]
    fn linear_q4_refuses_a_missized_bias() {
        let weight = QuantizedMatrixQ4::quantize(&deterministic(8, 5), 2, 4);
        let mut out = vec![0.0_f32; 2];
        linear_q4(&[1, 2, 3, 4], &[1.0], &weight, Some(&[0.5]), 1, &mut out);
    }

    #[test]
    fn quantization_error_is_bounded_by_the_level_step() {
        let k = 8192;
        let weight = deterministic(k, 0xBEEF);
        let matrix = QuantizedMatrixQ4::quantize(&weight, 1, k);
        let restored = matrix.dequantize_row(0);
        let scale = matrix.scales[0];
        for (position, (&original, &back)) in weight.iter().zip(restored.iter()).enumerate() {
            assert!(
                (original - back).abs() <= scale * 0.5 + f32::EPSILON * 8.0,
                "position {position}: |{original} - {back}| exceeds half a level ({scale})"
            );
        }
    }

    /// Every dispatchable tier must reproduce the scalar dot bit-for-bit — across whole
    /// blocks (k a multiple of 32), block remainders (16-byte pairs), odd nibbles, and
    /// combinations of all three. The SIMD island reads the odd-row pad's byte freely by
    /// design only when k is even; these shapes pin the tail arithmetic either way.
    #[test]
    fn every_available_q4_tier_reproduces_the_scalar_dot_bit_for_bit() {
        let tiers = Q4Tier::available();
        assert!(tiers.contains(&Q4Tier::Scalar));
        for (index, &(m, n, k)) in [
            (1_usize, 3_usize, 1_usize),
            (1, 4, 15),
            (1, 4, 16),
            (1, 4, 17),
            (1, 4, 31),
            (1, 4, 32),
            (1, 4, 33),
            (2, 5, 63),
            (1, 2048, 1024),
            (1, 3072, 1024),
        ]
        .iter()
        .enumerate()
        {
            let weight = deterministic(n * k, 0x51ED_0000 + index as u64);
            let activation = deterministic(m * k, 0xA0C7_0000 + index as u64);
            let mut x_q = vec![0_i8; m * k];
            // quantize_row_q8 scales ONE row; multi-row activations get one scale each by
            // quantizing per row, exactly what production's dynamic quantization does.
            let mut scales = vec![0.0_f32; m];
            for row in 0..m {
                scales[row] = quantize_row_q8(
                    &activation[row * k..(row + 1) * k],
                    &mut x_q[row * k..(row + 1) * k],
                );
            }
            let matrix = QuantizedMatrixQ4::quantize(&weight, n, k);
            for tier in &tiers {
                let mut out = vec![0.0_f32; m * n];
                linear_q4_tier(&x_q, &scales, &matrix, None, m, &mut out, *tier);
                let mut reference = vec![0.0_f32; m * n];
                linear_q4_tier(
                    &x_q,
                    &scales,
                    &matrix,
                    None,
                    m,
                    &mut reference,
                    Q4Tier::Scalar,
                );
                assert_eq!(
                    out, reference,
                    "tier {tier:?} diverged from scalar at m={m} n={n} k={k}"
                );
            }
        }
    }
}
