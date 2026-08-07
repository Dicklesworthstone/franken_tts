//! Bisect probe: what does the final SnakeBeta's residual divergence consist of?
//!
//! `codec_decoder.block_05` is `decoder.decoder.5`, the final BigVGAN SnakeBeta. It is the cleanest
//! seam left in the codec: purely elementwise, oracle-fed, no reduction anywhere in it, so every
//! bit of its divergence is created by the formula itself and nothing is inherited.
//!
//! Its divergence is also SPARSE in a way the other open seams are not — 42_042 of 2_580_480
//! elements, 1.6%, against the ConvNeXt block's 87%. A form that is merely a different summation
//! order diverges nearly everywhere; a divergence this sparse is the signature of a transcendental
//! whose two implementations agree on most inputs and part company on a few. There are exactly
//! three transcendentals here — `exp` on the two log-scale parameters, and `sin` on the scaled
//! input — so the probe evaluates each in f64 and rounds once, which is what a correctly-rounded
//! implementation would be indistinguishable from at f32 output width.
//!
//! Each candidate changes exactly ONE thing away from production. It is a reporting probe, not a
//! gate: it asserts only that the candidates ran.
//!
//! # Measured, and it names the remaining obstacle
//!
//! Over the seam's 2_580_480 elements on the pinned macOS oracle:
//!
//!   production                7.629e-6    42_042 over tolerance
//!   division_not_recip        7.629e-6    42_042
//!   fused_scale_multiply_add  7.629e-6   168_185
//!   exp_f64                   7.629e-6    56_184
//!   sin_f64                   7.629e-6    42_629
//!   sin_and_exp_f64           7.629e-6    56_807
//!   sin_input_product_f64     7.629e-6   108_673
//!   all_f64                   3.815e-6   201_833
//!
//! Production is the best candidate and every widening is worse, which is a stronger statement than
//! it looks. This seam is fed the oracle's own input, so nothing is inherited; and `*`, `+` and `/`
//! in f32 are correctly rounded and therefore have no freedom at all. `exp` and `sin` are the ONLY
//! degrees of freedom in this expression, so the 42_042 divergent elements are, necessarily, inputs
//! where the reference's `sin`/`exp` and ours round apart.
//!
//! The direction of the f64 rows is what identifies the reference. Widening `exp` alone costs 14k
//! elements and widening the `x * alpha` product costs 66k, so the reference's transcendentals are
//! NOT correctly rounded — our f32 libm is already closer to them than the true value is. That is
//! the signature of a vectorized ~1-ulp implementation, i.e. the Sleef routines PyTorch's ARM NEON
//! `Vectorized<float>` dispatches `sin` and `exp` to. `all_f64` halving `max_abs` while quintupling
//! the divergent count is the same story from the other side: computing more accurately moves us
//! away from the target, because the target is not accuracy.
//!
//! So `block_05` is exactly reachable, but only by reproducing `Sleef_sinf_u10` and `Sleef_expf_u10`
//! bit-for-bit rather than by any rearrangement of the surrounding f32 algebra. That is worth doing
//! beyond this seam: the same SnakeBeta runs inside every residual unit of `block_01`–`block_04`,
//! which are four of the five largest open seams in the codec.
//!
//! Two smaller results also land here. `division_not_recip` is bit-identical to production, so
//! `f32::recip` and `1.0 / x` are confirmed interchangeable and the production comment's choice is
//! free. `fused_scale_multiply_add` is 4x worse by element count, confirming by measurement that the
//! un-fused `scale * (sine * sine)` association the production code already argues for is correct.

use ftts_artifacts::safetensors::SafetensorsFile;
use ftts_conformance::npy::NpyArray;
use ftts_conformance::{
    compare::compare_f32,
    oracle::{CPU_TIER_TOLERANCE, CPU_TIER_TOLERANCE_SOURCE, OracleFixtures, SeamRef},
    report::{OracleTier, Outcome, Receipt},
};
use std::path::{Path, PathBuf};

const TEST_NAME: &str = "codec_snake_beta_bisect";

const SEAM: &str = "codec_decoder.block_05";
const CHANNELS: usize = 96;
/// The reference's `1.0 / (beta + 0.000000001)`.
const BETA_FLOOR: f32 = 1e-9;

/// Which part of the expression is evaluated in f64 and rounded once.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Widen {
    /// Everything in f32. Production.
    None,
    /// Only `exp` on the two log-scale parameters.
    Exp,
    /// Only `sin`.
    Sin,
    /// Both `sin` and `exp`.
    SinAndExp,
    /// `sin`, `exp`, and the `x * alpha` product feeding `sin`.
    SinExpAndProduct,
    /// The whole expression, including the reciprocal and the residual add.
    All,
}

/// How the scaled square is combined with the input.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Combine {
    /// `x + scale * (sine * sine)`, with `scale` from `recip`. Production.
    ReciprocalThenMultiply,
    /// The same, with `scale` from an explicit `1.0 / (beta + floor)` division.
    DivisionThenMultiply,
    /// `sine.mul_add(sine * scale, x)`, the form a fused kernel would reach for.
    FusedMultiplyAdd,
}

struct Variant {
    name: &'static str,
    widen: Widen,
    combine: Combine,
}

impl Variant {
    const fn production() -> Self {
        Self {
            name: "production",
            widen: Widen::None,
            combine: Combine::ReciprocalThenMultiply,
        }
    }
}

fn snake_beta(
    values: &[f32],
    frames: usize,
    alpha_log: &[f32],
    beta_log: &[f32],
    variant: &Variant,
) -> Vec<f32> {
    let mut out = values.to_vec();
    for channel in 0..CHANNELS {
        let widen_exp = matches!(
            variant.widen,
            Widen::Exp | Widen::SinAndExp | Widen::SinExpAndProduct | Widen::All
        );
        let (alpha, beta) = if widen_exp {
            (
                f64::from(alpha_log[channel]).exp(),
                f64::from(beta_log[channel]).exp(),
            )
        } else {
            (
                f64::from(alpha_log[channel].exp()),
                f64::from(beta_log[channel].exp()),
            )
        };

        if variant.widen == Widen::All {
            let scale = 1.0 / (beta + f64::from(BETA_FLOOR));
            for frame in 0..frames {
                let index = frame * CHANNELS + channel;
                let wide = f64::from(values[index]);
                let sine = (wide * alpha).sin();
                out[index] = (wide + scale * (sine * sine)) as f32;
            }
            continue;
        }

        let alpha = alpha as f32;
        let beta = beta as f32;
        let scale = match variant.combine {
            Combine::DivisionThenMultiply => 1.0f32 / (beta + BETA_FLOOR),
            _ => (beta + BETA_FLOOR).recip(),
        };
        let widen_sin = matches!(
            variant.widen,
            Widen::Sin | Widen::SinAndExp | Widen::SinExpAndProduct
        );
        for frame in 0..frames {
            let index = frame * CHANNELS + channel;
            let value = values[index];
            let sine = if widen_sin {
                let scaled = if variant.widen == Widen::SinExpAndProduct {
                    f64::from(value) * f64::from(alpha)
                } else {
                    f64::from(value * alpha)
                };
                scaled.sin() as f32
            } else {
                (value * alpha).sin()
            };
            out[index] = match variant.combine {
                Combine::FusedMultiplyAdd => sine.mul_add(sine * scale, value),
                _ => value + scale * (sine * sine),
            };
        }
    }
    out
}

fn checkpoint_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../docs/truth-pack/snapshots/hf/speech_tokenizer/model.safetensors")
}

fn skip(reason: &str) {
    Receipt::new(TEST_NAME, Outcome::Skipped)
        .contract("ConformanceExact/L2")
        .seam(SEAM)
        .reason(reason)
        .tolerance(CPU_TIER_TOLERANCE, CPU_TIER_TOLERANCE_SOURCE)
        .oracle_tier(OracleTier::CpuFp32Fallback)
        .emit();
}

fn widen_tensor(file: &SafetensorsFile, name: &str) -> Vec<f32> {
    let view = file
        .view(name)
        .unwrap_or_else(|| panic!("checkpoint is missing `{name}`"));
    (0..view.len())
        .map(|index| {
            view.get_f32(index)
                .unwrap_or_else(|| panic!("`{name}` index {index} out of range"))
        })
        .collect()
}

fn seam(name: &str) -> SeamRef<'_> {
    SeamRef {
        case: "synthetic-tone-en",
        mode: "icl_non_streaming",
        group: "codec_decode",
        seam: name,
    }
}

/// Fixture conv tensors are channel-major `[1, channels, frames]`.
fn to_time_major(array: &NpyArray) -> (Vec<f32>, usize, usize) {
    let (channels, frames) = match array.shape.as_slice() {
        [1, channels, frames] => (*channels, *frames),
        other => panic!("expected [1, C, T], got {other:?}"),
    };
    let mut out = vec![0.0f32; channels * frames];
    for channel in 0..channels {
        for frame in 0..frames {
            out[frame * channels + channel] = array.data[channel * frames + frame];
        }
    }
    (out, channels, frames)
}

fn to_channel_major(values: &[f32], frames: usize, channels: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; values.len()];
    for frame in 0..frames {
        for channel in 0..channels {
            out[channel * frames + frame] = values[frame * channels + channel];
        }
    }
    out
}

fn report(candidate: &str, expected: &NpyArray, actual: &[f32]) {
    let comparison = compare_f32(&expected.data, actual, CPU_TIER_TOLERANCE);
    eprintln!(
        "CODEC_SNAKE_BISECT candidate={candidate} max_abs={:.16e} over_tolerance={}/{} \
         cosine={:.12}",
        comparison.max_abs_diff, comparison.over_tolerance, comparison.len, comparison.cosine,
    );
    Receipt::new(TEST_NAME, Outcome::Passed)
        .contract("ConformanceExact/L2")
        .seam(SEAM)
        .reason(candidate)
        .tolerance(CPU_TIER_TOLERANCE, CPU_TIER_TOLERANCE_SOURCE)
        .oracle_tier(OracleTier::CpuFp32Fallback)
        .detail(comparison.to_json())
        .emit();
}

#[test]
fn codec_snake_beta_bisect() {
    let fixtures = match OracleFixtures::open_default() {
        Ok(fixtures) => fixtures,
        Err(error) => {
            skip(&format!("fixtures unavailable: {error}"));
            return;
        }
    };
    let checkpoint = checkpoint_path();
    if !checkpoint.is_file() {
        skip(&format!(
            "pinned checkpoint absent at {}",
            checkpoint.display()
        ));
        return;
    }
    if !fixtures.has_seam(&seam(&format!("{SEAM}.input"))) {
        skip("codec_decode seams absent from the fixture pack");
        return;
    }

    let file = SafetensorsFile::open(&checkpoint).expect("pinned speech tokenizer opens");
    let alpha_log = widen_tensor(&file, "decoder.decoder.5.alpha");
    let beta_log = widen_tensor(&file, "decoder.decoder.5.beta");
    assert_eq!(alpha_log.len(), CHANNELS, "SnakeBeta alpha width");

    let input = fixtures
        .seam(&seam(&format!("{SEAM}.input")), "args.0", 0)
        .expect("SnakeBeta input");
    let expected = fixtures
        .seam(&seam(&format!("{SEAM}.output")), "tensor", 0)
        .expect("SnakeBeta output");
    let (time_major, channels, frames) = to_time_major(&input);
    assert_eq!(channels, CHANNELS, "SnakeBeta input width");

    let candidates = [
        Variant::production(),
        Variant {
            name: "division_not_recip",
            combine: Combine::DivisionThenMultiply,
            ..Variant::production()
        },
        Variant {
            name: "fused_scale_multiply_add",
            combine: Combine::FusedMultiplyAdd,
            ..Variant::production()
        },
        Variant {
            name: "exp_f64",
            widen: Widen::Exp,
            ..Variant::production()
        },
        Variant {
            name: "sin_f64",
            widen: Widen::Sin,
            ..Variant::production()
        },
        Variant {
            name: "sin_and_exp_f64",
            widen: Widen::SinAndExp,
            ..Variant::production()
        },
        Variant {
            name: "sin_input_product_f64",
            widen: Widen::SinExpAndProduct,
            ..Variant::production()
        },
        Variant {
            name: "all_f64",
            widen: Widen::All,
            ..Variant::production()
        },
    ];

    for variant in &candidates {
        let actual = snake_beta(&time_major, frames, &alpha_log, &beta_log, variant);
        report(
            variant.name,
            &expected,
            &to_channel_major(&actual, frames, CHANNELS),
        );
    }
}
