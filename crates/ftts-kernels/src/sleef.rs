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
//! Nothing in this module is on the production path. It is selected only by the parity harness,
//! through [`crate::f32ref::F32Transcendental`].

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
fn df_two_sum(x: f32, y: f32) -> Df {
    let high = x + y;
    let v = high - x;
    let low = (x - (high - v)) + (y - v);
    Df { high, low }
}

/// `dfadd_vf2_vf_vf` — Dekker's fast two-sum, valid only because `|x| >= |y|`.
fn df_fast_two_sum(x: f32, y: f32) -> Df {
    let high = x + y;
    Df {
        high,
        low: (x - high) + y,
    }
}

/// `dfadd_vf2_vf2_vf` — fast two-sum of a double-float and a float.
fn df_add_f32(x: Df, y: f32) -> Df {
    let high = x.high + y;
    Df {
        high,
        low: ((x.high - high) + y) + x.low,
    }
}

/// `dfadd_vf2_vf_vf2` — fast two-sum of a float and a double-float.
fn df_add_to_f32(x: f32, y: Df) -> Df {
    let high = x + y.high;
    Df {
        high,
        low: ((x - high) + y.high) + y.low,
    }
}

/// `dfsqu_vf2_vf2` — the square of a double-float, FMA form.
fn df_square(x: Df) -> Df {
    let high = x.high * x.high;
    Df {
        high,
        low: (x.high + x.high).mul_add(x.low, x.high.mul_add(x.high, -high)),
    }
}

/// `dfmul_vf2_vf2_vf2` — the product of two double-floats, FMA form.
fn df_mul(x: Df, y: Df) -> Df {
    let high = x.high * y.high;
    let mut low = x.high.mul_add(y.high, -high);
    low = x.low.mul_add(y.high, low);
    low = x.high.mul_add(y.low, low);
    Df { high, low }
}

/// `dfmul_vf_vf2_vf2` — the same product, rounded down to a single f32, FMA form.
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
    fn the_payne_hanek_range_is_flagged_rather_than_claimed() {
        assert!(sinf_u10_in_fast_range(124.9));
        assert!(!sinf_u10_in_fast_range(125.0));
        assert!(!sinf_u10_in_fast_range(f32::NAN));
    }
}
