//! SLEEF's 1-ulp `sinf` and `expf`, ported to safe scalar Rust.
//!
// The coefficient literals below are bit-faithful copies of upstream SLEEF's. Truncating them to
// clippy's taste or substituting `std::f32::consts` values would silently change the arithmetic
// this module exists to reproduce exactly (AGENTS.md doctrine #8: no silent numerics changes).
#![allow(clippy::excessive_precision, clippy::approx_constant)]
//!
//! # Why this exists
//!
//! The pinned CPU-fp32 oracle does not evaluate elementwise transcendentals with the platform's
//! scalar libm. Its CPU kernels run through a vectorized `Vectorized<float>`, and on AArch64 that
//! type's `sin` and `exp` dispatch to SLEEF's `Sleef_sinf4_u10` / `Sleef_expf4_u10` — routines that
//! are accurate to 1 ulp rather than correctly rounded. `codec_snake_bisect` proved by measurement
//! that this is the whole remaining question at the SnakeBeta seam: every other operation there is
//! a correctly-rounded f32 `*`, `+` or `/` with no freedom at all, and *widening* `sin` or `exp`
//! toward the true value moves us further from the oracle, not closer. That direction is only
//! possible if the target itself is a ~1-ulp routine.
//!
//! So this module answers the question the bisect posed: it is the candidate implementation, in
//! pure portable Rust, that an Accelerate `vvsinf` call could only ever approximate.
//!
//! # What is ported, and what is not
//!
//! Both routines here are the per-lane arithmetic of SLEEF's AArch64 (`advsimd`) kernels with
//! `ENABLE_FMA_SP` on, which is what a PyTorch AArch64 build compiles. Every lane of those kernels
//! is an independent branch-free expression, so evaluating one element at a time is faithful — with
//! one exception, recorded here rather than hidden: `xsinf_u1` switches its whole vector to a
//! Payne–Hanek reduction when *any* lane exceeds [`TRIGRANGEMAX2_F`], and that branch is NOT ported.
//! [`sinf_u10`] falls back to a correctly-rounded f64 evaluation above that threshold and
//! [`sinf_u10_in_fast_range`] lets a caller assert it never got there.
//!
//! [`sinf_u10`] is also the default route's SnakeBeta sine (`FTTS_FAST_SNAKE`, via
//! [`snake_beta_frame_major`]); the parity harness selects these routines through
//! [`crate::f32ref::F32Transcendental`].

/// `1 / π`, rounded once to f32 — SLEEF's `M_1_PIf`.
const M_1_PI_F: f32 = 0.318_309_886_183_790_671_537_767_526_745_028_724_f32;
/// The three-part Cody–Waite split of π used by the medium-range reduction.
const PI_A2_F: f32 = 3.141_479_492_187_5;
const PI_B2_F: f32 = 0.000_113_159_418_106_079_101_56;
const PI_C2_F: f32 = 1.984_187_258_941_005_893_6e-9;
/// Above this magnitude SLEEF abandons the Cody–Waite reduction for Payne–Hanek.
pub const TRIGRANGEMAX2_F: f32 = 125.0;

/// `1 / ln 2`, rounded once to f32 — SLEEF's `R_LN2f`.
const R_LN2_F: f32 = 1.442_695_040_888_963_407_359_924_681_001_892_137_4_f32;
/// The two-part split of `ln 2`.
const L2U_F: f32 = 0.693_145_751_953_125;
const L2L_F: f32 = 1.428_606_765_330_187_045e-6;

/// A number held as an unevaluated sum of two f32s — SLEEF's `vfloat2`.
///
/// The high part carries the value, the low part the rounding error the high part dropped. Every
/// helper below is one of SLEEF's `df*` primitives under its own name; the FMA forms are used
/// because AArch64 always has FMA and SLEEF compiles `ENABLE_FMA_SP` there.
#[derive(Clone, Copy, Debug)]
struct Df {
    high: f32,
    low: f32,
}

/// `dfadd2_vf2_vf_vf` — Knuth's two-sum, which needs no ordering assumption.
#[inline(always)]
fn df_two_sum(x: f32, y: f32) -> Df {
    let high = x + y;
    let v = high - x;
    let low = (x - (high - v)) + (y - v);
    Df { high, low }
}

/// `dfadd_vf2_vf_vf` — Dekker's fast two-sum, valid only because `|x| >= |y|`.
#[inline(always)]
fn df_fast_two_sum(x: f32, y: f32) -> Df {
    let high = x + y;
    Df {
        high,
        low: (x - high) + y,
    }
}

/// `dfadd_vf2_vf2_vf` — fast two-sum of a double-float and a float.
#[inline(always)]
fn df_add_f32(x: Df, y: f32) -> Df {
    let high = x.high + y;
    Df {
        high,
        low: ((x.high - high) + y) + x.low,
    }
}

/// `dfadd_vf2_vf_vf2` — fast two-sum of a float and a double-float.
#[inline(always)]
fn df_add_to_f32(x: f32, y: Df) -> Df {
    let high = x + y.high;
    Df {
        high,
        low: ((x - high) + y.high) + y.low,
    }
}

/// `dfsqu_vf2_vf2` — the square of a double-float, FMA form.
#[inline(always)]
fn df_square(x: Df) -> Df {
    let high = x.high * x.high;
    Df {
        high,
        low: (x.high + x.high).mul_add(x.low, x.high.mul_add(x.high, -high)),
    }
}

/// `dfmul_vf2_vf2_vf2` — the product of two double-floats, FMA form.
#[inline(always)]
fn df_mul(x: Df, y: Df) -> Df {
    let high = x.high * y.high;
    let mut low = x.high.mul_add(y.high, -high);
    low = x.low.mul_add(y.high, low);
    low = x.high.mul_add(y.low, low);
    Df { high, low }
}

/// `dfmul_vf_vf2_vf2` — the same product, rounded down to a single f32, FMA form.
#[inline(always)]
fn df_mul_to_f32(x: Df, y: Df) -> f32 {
    x.high
        .mul_add(y.high, x.low.mul_add(y.high, x.high * y.low))
}

/// True when `x` takes SLEEF's Cody–Waite branch, the only one ported here.
#[must_use]
pub fn sinf_u10_in_fast_range(x: f32) -> bool {
    x.abs() < TRIGRANGEMAX2_F
}

/// `Sleef_sinf_u10` — `sin(d)` to within 1 ulp.
///
/// Outside [`sinf_u10_in_fast_range`] this returns a correctly-rounded result instead of SLEEF's
/// Payne–Hanek branch, which is a deliberate documented divergence and not SLEEF's answer.
#[must_use]
pub fn sinf_u10(d: f32) -> f32 {
    if !sinf_u10_in_fast_range(d) {
        return f64::from(d).sin() as f32;
    }
    sinf_u10_fast_range(d)
}

/// [`sinf_u10`]'s Cody–Waite branch alone: equal to it for every `d` with
/// [`sinf_u10_in_fast_range`], meaningless (but defined) outside that range.
///
/// Branch-free apart from selects, so a loop over it vectorizes; `#[inline(always)]` so every
/// `mul_add` lands inside the caller's target features (one `vfmadd`, not a libm `fmaf` call).
#[inline(always)]
fn sinf_u10_fast_range(d: f32) -> f32 {
    let scaled = (d * M_1_PI_F).round_ties_even();
    let quadrant = scaled as i32;

    // The reduced argument, carried as a double-float so the three Cody–Waite terms do not lose
    // the low bits that decide the last ulp of the result.
    let reduced = scaled.mul_add(-PI_A2_F, d);
    let mut reduced = df_two_sum(reduced, scaled * -PI_B2_F);
    reduced = df_add_f32(reduced, scaled * -PI_C2_F);

    let argument = reduced;
    let square = df_square(reduced);

    let mut poly = 2.608_315_980_978_659_354_150_3e-6_f32;
    poly = poly.mul_add(square.high, -0.000_198_106_907_191_686_332_225_8);
    poly = poly.mul_add(square.high, 0.008_333_078_585_565_090_179_443_36);

    let series = df_add_to_f32(
        1.0,
        df_mul(
            df_fast_two_sum(-0.166_666_597_127_914_428_710_938, poly * square.high),
            square,
        ),
    );
    let result = df_mul_to_f32(argument, series);

    if d == 0.0 {
        // `sin(-0.0)` is `-0.0`, which the polynomial's sign flip would not produce.
        return d;
    }
    if quadrant & 1 == 0 { result } else { -result }
}

/// SnakeBeta over frame-major data with [`sinf_u10`] as the sine:
/// `values[f * C + c] += scale[c] * sin(values[f * C + c] * alpha[c])²` for `C = alpha.len()`.
///
/// `alpha` and `scale` are the per-channel constants (`exp(alpha_log)` and
/// `1 / (exp(beta_log) + 1e-9)`), precomputed by the caller. The expression and its rounding
/// order are exactly the per-element form the codec's fast SnakeBeta always used; this function
/// only decides how it is compiled. Rows whose arguments are all in [`sinf_u10_in_fast_range`]
/// (all of them, in practice) run the branch-free loop; any other row takes the per-element
/// [`sinf_u10`] walk — the same value either way.
///
/// Why it is here and dispatched: on x86-64 the release binaries target baseline SSE2, where every
/// `f32::mul_add` in the polynomial lowers to a call into libm's `fmaf` — about eleven calls per
/// element — and the loop cannot vectorize. The `avx2,fma` / `avx512f` instantiations compile the
/// same body with hardware FMA. `mul_add` is a single correctly rounded fused operation on every
/// path (libm `fmaf` and `vfmadd` agree exactly), and lanes are independent, so all
/// instantiations are **bit-identical** (pinned by `snake_kernel_levels_are_bit_identical`).
///
/// # Panics
///
/// If `scale.len() != alpha.len()` or `values.len()` is not a multiple of `alpha.len()`.
pub fn snake_beta_frame_major(values: &mut [f32], alpha: &[f32], scale: &[f32]) {
    assert_eq!(scale.len(), alpha.len(), "SnakeBeta scale width");
    let channels = alpha.len();
    if channels == 0 {
        assert!(values.is_empty(), "SnakeBeta values without channels");
        return;
    }
    assert!(
        values.len().is_multiple_of(channels),
        "SnakeBeta values shape"
    );
    #[cfg(all(target_arch = "x86_64", feature = "x86-f32"))]
    {
        if std::arch::is_x86_feature_detected!("avx512f") {
            // SAFETY: AVX-512F confirmed on this CPU just above (it implies FMA).
            unsafe { x86_snake::avx512(values, alpha, scale) };
            return;
        }
        if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
        {
            // SAFETY: AVX2 and FMA confirmed on this CPU just above.
            unsafe { x86_snake::avx2_fma(values, alpha, scale) };
            return;
        }
    }
    snake_beta_rows(values, alpha, scale);
}

/// The shared SnakeBeta body; see [`snake_beta_frame_major`].
#[inline(always)]
fn snake_beta_rows(values: &mut [f32], alpha: &[f32], scale: &[f32]) {
    for row in values.chunks_exact_mut(alpha.len()) {
        let in_range = row
            .iter()
            .zip(alpha)
            .all(|(&value, &a)| sinf_u10_in_fast_range(value * a));
        if in_range {
            for ((value, &a), &s) in row.iter_mut().zip(alpha).zip(scale) {
                let sine = sinf_u10_fast_range(*value * a);
                *value += s * (sine * sine);
            }
        } else {
            for ((value, &a), &s) in row.iter_mut().zip(alpha).zip(scale) {
                let sine = sinf_u10(*value * a);
                *value += s * (sine * sine);
            }
        }
    }
}

#[cfg(all(target_arch = "x86_64", feature = "x86-f32"))]
mod x86_snake {
    //! Target-feature instantiations of [`super::snake_beta_rows`]. No intrinsics; bit-identity is
    //! argued on [`super::snake_beta_frame_major`].

    /// # Safety
    ///
    /// The CPU must support AVX2 and FMA.
    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn avx2_fma(values: &mut [f32], alpha: &[f32], scale: &[f32]) {
        super::snake_beta_rows(values, alpha, scale);
    }

    /// # Safety
    ///
    /// The CPU must support AVX-512F (and FMA, which every AVX-512F part has).
    #[target_feature(enable = "avx512f,fma")]
    pub(super) unsafe fn avx512(values: &mut [f32], alpha: &[f32], scale: &[f32]) {
        super::snake_beta_rows(values, alpha, scale);
    }
}

/// `Sleef_expf_u10` — `exp(d)` to within 1 ulp. SLEEF ships no lower-accuracy `expf`.
#[must_use]
pub fn expf_u10(d: f32) -> f32 {
    let exponent = (d * R_LN2_F).round_ties_even() as i32;
    let scaled = exponent as f32;

    let mut reduced = scaled.mul_add(-L2U_F, d);
    reduced = scaled.mul_add(-L2L_F, reduced);

    let mut poly = 0.000_198_527_617_612_853_646_278_381_f32;
    poly = poly.mul_add(reduced, 0.001_393_043_552_525_341_510_772_71);
    poly = poly.mul_add(reduced, 0.008_333_360_776_305_198_669_433_59);
    poly = poly.mul_add(reduced, 0.041_666_485_369_205_474_853_515_6);
    poly = poly.mul_add(reduced, 0.166_666_671_633_720_397_949_219);
    poly = poly.mul_add(reduced, 0.5);

    let mantissa = 1.0 + (reduced * reduced).mul_add(poly, reduced);
    let result = ldexp2(mantissa, exponent);

    if d < -104.0 {
        return 0.0;
    }
    if d > 100.0 {
        return f32::INFINITY;
    }
    result
}

/// `vldexp2_vf_vf_vi2` — `x * 2^exponent`, split in half so neither factor can overflow.
fn ldexp2(x: f32, exponent: i32) -> f32 {
    let half = exponent >> 1;
    x * pow2i(half) * pow2i(exponent - half)
}

/// `vpow2i_vf_vi2` — `2^exponent` built directly out of the exponent field.
fn pow2i(exponent: i32) -> f32 {
    f32::from_bits(((exponent + 0x7f) << 23) as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Distance in representable f32 steps between a candidate and the correctly-rounded value.
    ///
    /// This is the port's own correctness proof and it is independent of any oracle: a routine
    /// documented at 1 ulp that is transcribed wrongly does not stay within 1 ulp, it lands
    /// hundreds or millions of steps away. Passing this says the algorithm is SLEEF's; only the
    /// parity harness can say whether SLEEF is what the oracle ran.
    fn ulp_distance(candidate: f32, exact: f64) -> i64 {
        let rounded = exact as f32;
        assert!(
            candidate.is_finite() && rounded.is_finite(),
            "finite inputs"
        );
        let ordered = |value: f32| -> i64 {
            let bits = i64::from(value.to_bits() as i32);
            if bits < 0 {
                i64::from(i32::MIN) - bits
            } else {
                bits
            }
        };
        (ordered(candidate) - ordered(rounded)).abs()
    }

    /// A deterministic even spread of `count` values over `[-limit, limit]`.
    fn sweep(limit: f32, count: u32) -> impl Iterator<Item = f32> {
        (0..count).map(move |step| {
            let unit = f64::from(step) / f64::from(count - 1);
            ((unit * 2.0 - 1.0) * f64::from(limit)) as f32
        })
    }

    #[test]
    fn sinf_u10_is_within_one_ulp_over_the_cody_waite_range() {
        let mut worst = 0;
        for x in sweep(TRIGRANGEMAX2_F * 0.999, 40_001) {
            worst = worst.max(ulp_distance(sinf_u10(x), f64::from(x).sin()));
        }
        assert!(worst <= 1, "sinf_u10 drifted {worst} ulps from correct");
    }

    #[test]
    fn sinf_u10_is_within_one_ulp_near_zero_where_the_seam_lives() {
        let mut worst = 0;
        for x in sweep(8.0, 60_001) {
            worst = worst.max(ulp_distance(sinf_u10(x), f64::from(x).sin()));
        }
        assert!(worst <= 1, "sinf_u10 drifted {worst} ulps near zero");
    }

    #[test]
    fn expf_u10_is_within_one_ulp() {
        let mut worst = 0;
        for x in sweep(80.0, 60_001) {
            worst = worst.max(ulp_distance(expf_u10(x), f64::from(x).exp()));
        }
        assert!(worst <= 1, "expf_u10 drifted {worst} ulps from correct");
    }

    #[test]
    fn the_exact_cases_stay_exact() {
        assert_eq!(sinf_u10(0.0), 0.0);
        assert!(sinf_u10(-0.0).is_sign_negative());
        assert_eq!(expf_u10(0.0), 1.0);
        assert_eq!(expf_u10(-200.0), 0.0);
        assert_eq!(expf_u10(200.0), f32::INFINITY);
    }

    #[test]
    fn snake_kernel_levels_are_bit_identical() {
        // The dispatched kernel (whatever this CPU selects) and the portable body must agree with
        // the plain per-element `sinf_u10` form to the bit: in-range rows, a row with one
        // Payne–Hanek-range argument (the per-element fallback), signed zeros, and a NaN.
        let channels = 96;
        let alpha: Vec<f32> = (0..channels).map(|c| 0.5 + c as f32 * 0.07).collect();
        let scale: Vec<f32> = (0..channels)
            .map(|c| 1.0 / (0.3 + c as f32 * 0.01))
            .collect();
        let mut values: Vec<f32> = sweep(9.0, 40 * channels as u32).collect();
        values[3 * channels + 5] = 400.0; // out of the Cody–Waite range after scaling
        values[7 * channels] = -0.0;
        values[7 * channels + 1] = 0.0;
        values[11 * channels + 2] = f32::NAN;
        let mut expected = values.clone();
        for row in expected.chunks_exact_mut(channels) {
            for ((value, &a), &s) in row.iter_mut().zip(&alpha).zip(&scale) {
                let sine = sinf_u10(*value * a);
                *value += s * (sine * sine);
            }
        }
        let mut portable = values.clone();
        snake_beta_rows(&mut portable, &alpha, &scale);
        let mut dispatched = values;
        snake_beta_frame_major(&mut dispatched, &alpha, &scale);
        for (index, ((e, p), d)) in expected.iter().zip(&portable).zip(&dispatched).enumerate() {
            assert_eq!(e.to_bits(), p.to_bits(), "portable differs at {index}");
            assert_eq!(e.to_bits(), d.to_bits(), "dispatched differs at {index}");
        }
    }

    #[test]
    fn the_payne_hanek_range_is_flagged_rather_than_claimed() {
        assert!(sinf_u10_in_fast_range(124.9));
        assert!(!sinf_u10_in_fast_range(125.0));
        assert!(!sinf_u10_in_fast_range(f32::NAN));
    }
}
