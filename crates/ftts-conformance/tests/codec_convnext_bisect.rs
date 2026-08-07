//! Bisect probe: which op inside the codec's ConvNeXt block owns its residual divergence?
//!
//! After the convolution GEMM went exact (`codec_gemm_bisect`, ratchet round 4), the two largest
//! remaining codec seams are both the same block: `codec_decoder.upsample_0_1` (1.699e-6) and
//! `codec_decoder.upsample_1_1` (2.480e-5). Both are oracle-fed — the pack captures the block's own
//! input — so the whole divergence is created inside these five ops:
//!
//!   depthwise Conv1d (kernel 7, groups = channels)  ->  LayerNorm  ->  pwconv1 (`nn.Linear`)
//!   ->  GELU  ->  pwconv2 (`nn.Linear`)  ->  gamma-scaled residual add
//!
//! The checkpoint says which rule each one falls under: `dwconv.conv.weight` is `[1024, 1, 7]`, a
//! convolution, while `pwconv1/pwconv2` are 2-D and so are genuine `nn.Linear` — the sites the
//! round-4 note records as REJECTED when routed through the convolution's BLAS. That rejection was
//! measured on the whole pipeline at once; this probe re-measures it here, alone, alongside the
//! forms the other three ops could take.
//!
//! Each candidate changes exactly ONE op away from the production block, so a candidate's number is
//! attributable. It is a reporting probe, not a gate: it asserts only that the candidates ran.
//!
//! # Measured, and the result is negative
//!
//! Over the seam's 28_672 elements on the pinned macOS oracle:
//!
//!   production                    1.699e-6   24_932 over tolerance
//!   dwconv_bias_trailing          2.027e-6   24_895
//!   dwconv_bias_seeded_blas       2.027e-6   24_899
//!   layernorm_welford_moments     1.609e-6   24_956
//!   layernorm_mean_of_squares     1.431e-6   24_841
//!   layernorm_fused_scale_shift   1.907e-6   24_983
//!   layernorm_welford_and_fused   1.669e-6   24_953
//!   pwconv_blas_bias_trailing     2.146e-6   24_943
//!   pwconv_blas_bias_seeded       1.490e-6   24_695
//!   gelu_f64_erf                  1.669e-6   24_855
//!   layernorm_eps_1e5             1.192e-5   26_855
//!
//! Every arithmetic-form candidate sits in the same 1.4e-6 – 2.1e-6 band and ~87% of the block's
//! elements diverge under all of them. Compare round 4, where the discriminating candidate was 0.0
//! against 1.2e-5: there the op form WAS the answer, and it announced itself. Here nothing does, so
//! this probe forecloses hypotheses rather than confirming one — the depthwise bias placement, its
//! reduction backend, the LayerNorm moment algorithm, the LayerNorm affine form, and the `erf`
//! precision are each individually NOT what this seam is made of.
//!
//! The one candidate that DID separate is the epsilon, and it separated the wrong way: `1e-5` — the
//! only norm epsilon the pinned `decoder_config` actually names — is 7x worse than production's
//! `1e-6`. So the ConvNeXt default this block was written against is now confirmed by measurement
//! instead of assumed, and the residual is not an eps mismatch either.
//!
//! `layernorm_mean_of_squares` and `pwconv_blas_bias_seeded` are nominally the best two, and
//! neither was adopted: a ulp-scale wiggle on one seam is not evidence of the right form, and
//! `pwconv_blas_bias_seeded` is the site the round-4 pipeline run already REJECTED (it moved
//! `upsample_1_1` 2.480e-5 -> 2.861e-5). Adopting on this probe alone would be overfitting to it.
//!
//! What this leaves standing is the round-4 note's own leading suspect: that these `nn.Linear`
//! sites lower to a batched 3-D `baddbmm` whose blocking none of the candidates here reproduces,
//! in which case the residue is not attributable to any ONE op and this whole axis is the wrong
//! cut. Testing that needs a `baddbmm`-shaped call, which `linear_with_accumulation` cannot
//! currently issue.

#![feature(float_erf)]

use ftts_artifacts::safetensors::SafetensorsFile;
use ftts_conformance::npy::NpyArray;
use ftts_conformance::{
    compare::compare_f32,
    oracle::{CPU_TIER_TOLERANCE, CPU_TIER_TOLERANCE_SOURCE, OracleFixtures, SeamRef},
    report::{OracleTier, Outcome, Receipt},
};
use ftts_kernels::f32ref;
use std::path::{Path, PathBuf};

const TEST_NAME: &str = "codec_convnext_op_bisect";

/// `decoder.upsample.0.1`: the first latent-upsample ConvNeXt block.
const CHANNELS: usize = 1_024;
const KERNEL: usize = 7;
const SEAM: &str = "codec_decoder.upsample_0_1";
const LAYER_NORM_EPS: f32 = 1e-6;

/// How the depthwise convolution reduces and where its bias lands.
// The `Bias` prefix is the taxonomy: every variant is named by where the bias enters the
// reduction, which is the whole question this bisect asks.
#[allow(clippy::enum_variant_names)]
#[derive(Clone, Copy, PartialEq, Eq)]
enum Depthwise {
    /// Bias seeds the accumulator, then the taps are added left to right. Production.
    BiasSeededScalar,
    /// The taps are summed first and the bias added after.
    BiasTrailingScalar,
    /// One per-channel `[frames, 7]` GEMM with the bias seeded at `beta = 1`, which is what a
    /// grouped `slow_conv2d` issues per group.
    BiasSeededBlas,
}

/// How LayerNorm computes its row moments.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Moments {
    /// Sum for the mean, then a second pass over the centered squares. Production.
    TwoPass,
    /// One pass of Welford, which is what ATen's `RowwiseMoments` runs.
    Welford,
    /// `E[x^2] - E[x]^2` in one pass.
    MeanOfSquares,
}

/// How LayerNorm applies those moments.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Affine {
    /// `(x - mean) * rstd * weight + bias`. Production.
    Centered,
    /// `(x * rstd + (-mean * rstd)) * weight + bias`, the fused scale/shift form.
    FusedScaleShift,
}

/// How the two pointwise `nn.Linear` projections reduce.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Pointwise {
    /// Scalar left-to-right, bias trailing. Production.
    Scalar,
    /// Platform BLAS at `beta = 0`, bias trailing.
    Blas,
    /// Platform BLAS with the bias seeded at `beta = 1`, as `addmm` would.
    BlasBiasSeeded,
}

/// How GELU's `erf` is evaluated.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Gelu {
    /// `0.5 * x * (1 + erff(x * FRAC_1_SQRT_2))` in f32. Production.
    F32Erf,
    /// The same expression evaluated in f64 and rounded once, which is what a correctly-rounded
    /// vectorized `erf` would be indistinguishable from at f32 output width.
    F64Erf,
}

struct Variant {
    name: &'static str,
    depthwise: Depthwise,
    moments: Moments,
    affine: Affine,
    pointwise: Pointwise,
    gelu: Gelu,
    /// The LayerNorm epsilon. Production uses the ConvNeXt default; the pinned `decoder_config`
    /// names only `rms_norm_eps = 1e-5` and says nothing about this block, so the alternative is
    /// worth one measurement — an eps mismatch would show up as exactly this kind of diffuse,
    /// few-ulp, everywhere-at-once residual.
    eps: f32,
}

impl Variant {
    const fn production() -> Self {
        Self {
            name: "production",
            depthwise: Depthwise::BiasSeededScalar,
            moments: Moments::TwoPass,
            affine: Affine::Centered,
            pointwise: Pointwise::Scalar,
            gelu: Gelu::F32Erf,
            eps: LAYER_NORM_EPS,
        }
    }
}

struct ConvNextWeights {
    depthwise_weight: Vec<f32>,
    depthwise_bias: Vec<f32>,
    norm_weight: Vec<f32>,
    norm_bias: Vec<f32>,
    pwconv1: Vec<f32>,
    pwconv1_bias: Vec<f32>,
    pwconv2: Vec<f32>,
    pwconv2_bias: Vec<f32>,
    gamma: Vec<f32>,
}

fn forward(input: &[f32], frames: usize, weights: &ConvNextWeights, variant: &Variant) -> Vec<f32> {
    let mut hidden = vec![0.0f32; input.len()];
    depthwise(
        input,
        frames,
        &weights.depthwise_weight,
        &weights.depthwise_bias,
        variant.depthwise,
        &mut hidden,
    );
    layer_norm(
        &mut hidden,
        frames,
        &weights.norm_weight,
        &weights.norm_bias,
        variant.moments,
        variant.affine,
        variant.eps,
    );

    let mut expanded = vec![0.0f32; frames * 4 * CHANNELS];
    pointwise(
        &hidden,
        &weights.pwconv1,
        &weights.pwconv1_bias,
        frames,
        CHANNELS,
        4 * CHANNELS,
        variant.pointwise,
        &mut expanded,
    );
    for value in &mut expanded {
        *value = match variant.gelu {
            Gelu::F32Erf => {
                0.5 * *value * (1.0 + (*value * core::f32::consts::FRAC_1_SQRT_2).erf())
            }
            Gelu::F64Erf => {
                let wide = f64::from(*value);
                (0.5 * wide * (1.0 + (wide * core::f64::consts::FRAC_1_SQRT_2).erf())) as f32
            }
        };
    }
    pointwise(
        &expanded,
        &weights.pwconv2,
        &weights.pwconv2_bias,
        frames,
        4 * CHANNELS,
        CHANNELS,
        variant.pointwise,
        &mut hidden,
    );

    let mut output = input.to_vec();
    for frame in 0..frames {
        for channel in 0..CHANNELS {
            let index = frame * CHANNELS + channel;
            output[index] += weights.gamma[channel] * hidden[index];
        }
    }
    output
}

fn depthwise(
    input: &[f32],
    frames: usize,
    weight: &[f32],
    bias: &[f32],
    form: Depthwise,
    output: &mut [f32],
) {
    if form == Depthwise::BiasSeededBlas {
        // One group per channel: unfold that channel's causal window into `[frames, 7]` columns and
        // let the BLAS reduce them, with the channel's single bias seeding the accumulator.
        let mut columns = vec![0.0f32; frames * KERNEL];
        let mut column_output = vec![0.0f32; frames];
        for channel in 0..CHANNELS {
            columns.fill(0.0);
            for frame in 0..frames {
                for tap in 0..KERNEL {
                    if let Some(source) = frame.checked_sub(KERNEL - 1 - tap) {
                        columns[frame * KERNEL + tap] = input[source * CHANNELS + channel];
                    }
                }
            }
            f32ref::linear_with_accumulation(
                &columns,
                &weight[channel * KERNEL..(channel + 1) * KERNEL],
                Some(&bias[channel..=channel]),
                frames,
                KERNEL,
                1,
                f32ref::F32LinearAccumulation::AccelerateBiasSeeded,
                &mut column_output,
            );
            for frame in 0..frames {
                output[frame * CHANNELS + channel] = column_output[frame];
            }
        }
        return;
    }

    for frame in 0..frames {
        for channel in 0..CHANNELS {
            let seed = match form {
                Depthwise::BiasSeededScalar => bias[channel],
                _ => 0.0,
            };
            let mut total = seed;
            for tap in 0..KERNEL {
                if let Some(source) = frame.checked_sub(KERNEL - 1 - tap) {
                    total += input[source * CHANNELS + channel] * weight[channel * KERNEL + tap];
                }
            }
            output[frame * CHANNELS + channel] = match form {
                Depthwise::BiasTrailingScalar => total + bias[channel],
                _ => total,
            };
        }
    }
}

fn layer_norm(
    values: &mut [f32],
    frames: usize,
    weight: &[f32],
    bias: &[f32],
    moments: Moments,
    affine: Affine,
    eps: f32,
) {
    for frame in 0..frames {
        let row = &mut values[frame * CHANNELS..(frame + 1) * CHANNELS];
        let width = CHANNELS as f32;
        let (mean, variance) = match moments {
            Moments::TwoPass => {
                let mut sum = 0.0f32;
                for value in row.iter() {
                    sum += *value;
                }
                let mean = sum / width;
                let mut variance = 0.0f32;
                for value in row.iter() {
                    let centered = *value - mean;
                    variance += centered * centered;
                }
                (mean, variance / width)
            }
            Moments::Welford => {
                let mut mean = 0.0f32;
                let mut squared = 0.0f32;
                for (index, value) in row.iter().enumerate() {
                    let count = index as f32 + 1.0;
                    let delta = *value - mean;
                    mean += delta / count;
                    squared += delta * (*value - mean);
                }
                (mean, squared / width)
            }
            Moments::MeanOfSquares => {
                let mut sum = 0.0f32;
                let mut squares = 0.0f32;
                for value in row.iter() {
                    sum += *value;
                    squares += *value * *value;
                }
                let mean = sum / width;
                (mean, squares / width - mean * mean)
            }
        };
        let rstd = (variance + eps).sqrt().recip();
        match affine {
            Affine::Centered => {
                for channel in 0..CHANNELS {
                    row[channel] = (row[channel] - mean) * rstd * weight[channel] + bias[channel];
                }
            }
            Affine::FusedScaleShift => {
                let shift = -mean * rstd;
                for channel in 0..CHANNELS {
                    row[channel] = (row[channel] * rstd + shift) * weight[channel] + bias[channel];
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn pointwise(
    input: &[f32],
    weight: &[f32],
    bias: &[f32],
    frames: usize,
    k: usize,
    n: usize,
    form: Pointwise,
    output: &mut [f32],
) {
    let accumulation = match form {
        Pointwise::Scalar => f32ref::F32LinearAccumulation::Scalar,
        Pointwise::Blas => f32ref::F32LinearAccumulation::Accelerate,
        Pointwise::BlasBiasSeeded => f32ref::F32LinearAccumulation::AccelerateBiasSeeded,
    };
    f32ref::linear_with_accumulation(
        input,
        weight,
        Some(bias),
        frames,
        k,
        n,
        accumulation,
        output,
    );
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

fn widen(file: &SafetensorsFile, name: &str) -> Vec<f32> {
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
        "CODEC_CONVNEXT_BISECT candidate={candidate} max_abs={:.16e} over_tolerance={}/{} \
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
fn codec_convnext_op_bisect() {
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
    let prefix = "decoder.upsample.0.1";
    let weights = ConvNextWeights {
        depthwise_weight: widen(&file, &format!("{prefix}.dwconv.conv.weight")),
        depthwise_bias: widen(&file, &format!("{prefix}.dwconv.conv.bias")),
        norm_weight: widen(&file, &format!("{prefix}.norm.weight")),
        norm_bias: widen(&file, &format!("{prefix}.norm.bias")),
        pwconv1: widen(&file, &format!("{prefix}.pwconv1.weight")),
        pwconv1_bias: widen(&file, &format!("{prefix}.pwconv1.bias")),
        pwconv2: widen(&file, &format!("{prefix}.pwconv2.weight")),
        pwconv2_bias: widen(&file, &format!("{prefix}.pwconv2.bias")),
        gamma: widen(&file, &format!("{prefix}.gamma")),
    };

    let input = fixtures
        .seam(&seam(&format!("{SEAM}.input")), "args.0", 0)
        .expect("ConvNeXt input");
    let expected = fixtures
        .seam(&seam(&format!("{SEAM}.output")), "tensor", 0)
        .expect("ConvNeXt output");
    let (time_major, channels, frames) = to_time_major(&input);
    assert_eq!(channels, CHANNELS, "ConvNeXt input width");

    let candidates = [
        Variant::production(),
        Variant {
            name: "dwconv_bias_trailing",
            depthwise: Depthwise::BiasTrailingScalar,
            ..Variant::production()
        },
        Variant {
            name: "dwconv_bias_seeded_blas",
            depthwise: Depthwise::BiasSeededBlas,
            ..Variant::production()
        },
        Variant {
            name: "layernorm_welford_moments",
            moments: Moments::Welford,
            ..Variant::production()
        },
        Variant {
            name: "layernorm_mean_of_squares",
            moments: Moments::MeanOfSquares,
            ..Variant::production()
        },
        Variant {
            name: "layernorm_fused_scale_shift",
            affine: Affine::FusedScaleShift,
            ..Variant::production()
        },
        Variant {
            name: "layernorm_welford_and_fused",
            moments: Moments::Welford,
            affine: Affine::FusedScaleShift,
            ..Variant::production()
        },
        Variant {
            name: "pwconv_blas_bias_trailing",
            pointwise: Pointwise::Blas,
            ..Variant::production()
        },
        Variant {
            name: "pwconv_blas_bias_seeded",
            pointwise: Pointwise::BlasBiasSeeded,
            ..Variant::production()
        },
        Variant {
            name: "layernorm_eps_1e5",
            eps: 1e-5,
            ..Variant::production()
        },
        Variant {
            name: "gelu_f64_erf",
            gelu: Gelu::F64Erf,
            ..Variant::production()
        },
    ];

    for variant in &candidates {
        let actual = forward(&time_major, frames, &weights, variant);
        report(
            variant.name,
            &expected,
            &to_channel_major(&actual, frames, CHANNELS),
        );
    }
}
