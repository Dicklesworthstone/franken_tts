//! Where the talker layer-00 CPU-fp32 residual is *born*, by selective f64 promotion.
//!
//! The fixture pack captures `talker.layer_NN.input` and `talker.layer_NN.output` and nothing in
//! between, so every previous attempt to localize layer 00's `max_abs = 9.54e-6` had to guess: the
//! variant sweep in `talker_layer_l2` tries candidate f32 reduction orders and reports which lands
//! closest, which tells you what the oracle *might* have done but never which operation the
//! residual came from. A guess that lands closer is not an attribution.
//!
//! This test localizes without needing an internal seam. Each operation family in the layer is
//! promoted to f64 in turn — the reduction's rounding removed rather than reordered — and the seam
//! is re-measured. That splits the divergence into named parts:
//!
//! * promoting family X and watching `max_abs` **collapse** means our f32 rounding in X is what
//!   the oracle did not have, i.e. X carries the residual;
//! * promoting family X and watching `max_abs` **not move** rules X out, permanently, with no
//!   appeal to which lane order looked closest;
//! * promoting **everything** gives the layer's f64 value, so `‖oracle − f64‖` is how far the
//!   *oracle itself* sits from exact arithmetic. If that distance is the same size as the residual
//!   we are chasing, then the oracle's own f32 rounding is the whole gap, and no improvement in
//!   our accuracy can reach `0.0` — only reproducing its exact operation order can.
//!
//! That last row is the one that decides whether bit-exactness at this tier is an engineering task
//! or a category error, which is the open question on `frankentts-p1-talker-z2w`.
//!
//! Model-gated: skips with SUCCESS and a named reason when the fixtures or the pinned checkpoint
//! are absent, never folding a skip into green.

use ftts_artifacts::safetensors::SafetensorsFile;
use ftts_conformance::npy::NpyArray;
use ftts_conformance::{
    compare::compare_f32,
    oracle::{OracleFixtures, SeamRef},
    report::{OracleTier, Outcome, Receipt},
};
use ftts_kernels::f32ref::{
    F32LinearAccumulation, F32RmsNormArithmetic, F32SiluArithmetic, F32SoftmaxArithmetic,
};
use ftts_model_qwen::talker::{
    KvCache, RotaryRows, TalkerConfig, TalkerLayerWeights, collapse_mrope,
    forward_layer_with_arithmetic,
};
use std::path::{Path, PathBuf};

const TEST_NAME: &str = "contract_a_l2_talker_layer_00_residual_attribution";
const CONTRACT: &str = "ConformanceExact/L2";
const SEAM: &str = "talker.layer_00.output";

/// The all-f32 residual this attribution explains, frozen by `talker_layer_l2`.
const BASELINE_MAX_ABS: f64 = 9.536_743_164_062_5e-6;

/// Largest magnitude in the oracle's `talker.layer_00.output`.
const OUTPUT_SCALE: f64 = 16.365_345;

/// How much of the baseline residual a family must remove before we call it the carrier.
///
/// Set well away from both ends: a family that removes over half the divergence is doing the work,
/// and one that removes under a tenth is noise. Nothing measured here sits between the two.
const CARRIER_SHARE: f64 = 0.5;
const BYSTANDER_SHARE: f64 = 0.1;

fn checkpoint_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../docs/truth-pack/snapshots/hf/model.safetensors")
}

fn skip(reason: &str) {
    Receipt::new(TEST_NAME, Outcome::Skipped)
        .contract(CONTRACT)
        .seam(SEAM)
        .reason(reason)
        .oracle_tier(OracleTier::CpuFp32Fallback)
        .emit();
}

fn widen(file: &SafetensorsFile, name: &str) -> Result<Vec<f32>, String> {
    let view = file
        .view(name)
        .ok_or_else(|| format!("checkpoint is missing `{name}`"))?;
    (0..view.len())
        .map(|index| {
            view.get_f32(index)
                .ok_or_else(|| format!("`{name}` index {index} out of range"))
        })
        .collect()
}

/// Splits the oracle's `[3, 1, seq, head_dim]` rotary tensor into its three mRoPE axes.
fn split_axes(array: &NpyArray, seq: usize, head_dim: usize) -> [Vec<f32>; 3] {
    let stride = seq * head_dim;
    assert_eq!(
        array.data.len(),
        3 * stride,
        "rotary tensor should be [3, 1, {seq}, {head_dim}], got {}",
        array.shape_string()
    );
    [
        array.data[..stride].to_vec(),
        array.data[stride..2 * stride].to_vec(),
        array.data[2 * stride..].to_vec(),
    ]
}

/// One point in the promotion sweep: which operation families keep f32 rounding, and which do not.
#[derive(Clone, Copy)]
struct Promotion {
    /// Human-readable name of the family promoted (or `"none"` / `"all"`).
    family: &'static str,
    projections: F32LinearAccumulation,
    rms: F32RmsNormArithmetic,
    silu: F32SiluArithmetic,
    softmax: F32SoftmaxArithmetic,
    attention: F32LinearAccumulation,
}

/// The all-f32 reference arithmetic, i.e. exactly what production `forward_layer` runs.
const ALL_F32: Promotion = Promotion {
    family: "none (production f32)",
    projections: F32LinearAccumulation::Scalar,
    rms: F32RmsNormArithmetic::ScalarReciprocalSqrt,
    silu: F32SiluArithmetic::Divide,
    softmax: F32SoftmaxArithmetic::ReciprocalMultiply,
    attention: F32LinearAccumulation::Scalar,
};

fn sweep() -> Vec<Promotion> {
    vec![
        ALL_F32,
        Promotion {
            family: "projections (QKV, O, gate, up, down)",
            projections: F32LinearAccumulation::WidenedF64,
            ..ALL_F32
        },
        Promotion {
            family: "rmsnorm (input, post-attention, QK-Norm)",
            rms: F32RmsNormArithmetic::WIDENED_F64,
            ..ALL_F32
        },
        Promotion {
            family: "silu",
            silu: F32SiluArithmetic::WidenedF64,
            ..ALL_F32
        },
        Promotion {
            family: "softmax",
            softmax: F32SoftmaxArithmetic::WidenedF64,
            ..ALL_F32
        },
        Promotion {
            family: "attention matmuls (QK^T, weights x V)",
            attention: F32LinearAccumulation::WidenedF64,
            ..ALL_F32
        },
        Promotion {
            family: "all",
            projections: F32LinearAccumulation::WidenedF64,
            rms: F32RmsNormArithmetic::WIDENED_F64,
            silu: F32SiluArithmetic::WidenedF64,
            softmax: F32SoftmaxArithmetic::WidenedF64,
            attention: F32LinearAccumulation::WidenedF64,
            ..ALL_F32
        },
    ]
}

/// Everything one run of the layer needs, read once and reused across the sweep.
struct Inputs {
    config: TalkerConfig,
    hidden_in: Vec<f32>,
    expected: Vec<f32>,
    mask: Vec<f32>,
    cos_rows: Vec<f32>,
    sin_rows: Vec<f32>,
    seq: usize,
}

fn load() -> Result<(Inputs, Vec<Vec<f32>>), String> {
    let fixtures =
        OracleFixtures::open_default().map_err(|error| format!("fixtures unavailable: {error}"))?;
    let checkpoint = checkpoint_path();
    if !checkpoint.is_file() {
        return Err(format!(
            "pinned checkpoint absent at {}",
            checkpoint.display()
        ));
    }
    let input_seam = SeamRef {
        case: "synthetic-tone-en",
        mode: "icl_non_streaming",
        group: "talker_free_running",
        seam: "talker.layer_00.input",
    };
    let output_seam = SeamRef {
        seam: SEAM,
        ..input_seam
    };
    if !fixtures.has_seam(&input_seam) {
        return Err(format!("seam absent: {}", input_seam.describe()));
    }

    let hidden_in = fixtures
        .seam(&input_seam, "args.0", 0)
        .map_err(|error| format!("cannot read the layer input: {error}"))?;
    let expected = fixtures
        .seam(&output_seam, "0", 0)
        .map_err(|error| format!("cannot read the layer output: {error}"))?;
    let cos = fixtures
        .seam(&input_seam, "kwargs.position_embeddings.0", 0)
        .map_err(|error| format!("cannot read rotary cos: {error}"))?;
    let sin = fixtures
        .seam(&input_seam, "kwargs.position_embeddings.1", 0)
        .map_err(|error| format!("cannot read rotary sin: {error}"))?;
    let mask = fixtures
        .seam(&input_seam, "kwargs.attention_mask", 0)
        .map_err(|error| format!("cannot read the attention mask: {error}"))?;

    let config = TalkerConfig::default();
    let seq = hidden_in.data.len() / config.hidden_size;
    assert_eq!(
        hidden_in.data.len(),
        seq * config.hidden_size,
        "layer input should be a whole number of {}-wide rows",
        config.hidden_size
    );

    let cos_axes = split_axes(&cos, seq, config.head_dim);
    let sin_axes = split_axes(&sin, seq, config.head_dim);
    let sections = [24usize, 20, 20];
    let cos_rows = collapse_mrope(
        [&cos_axes[0], &cos_axes[1], &cos_axes[2]],
        seq,
        config.head_dim,
        sections,
    );
    let sin_rows = collapse_mrope(
        [&sin_axes[0], &sin_axes[1], &sin_axes[2]],
        seq,
        config.head_dim,
        sections,
    );

    let file = SafetensorsFile::open(&checkpoint)
        .map_err(|error| format!("cannot open {}: {error}", checkpoint.display()))?;
    let prefix = "talker.model.layers.0";
    let weights = [
        "input_layernorm.weight",
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.q_norm.weight",
        "self_attn.k_norm.weight",
        "self_attn.o_proj.weight",
        "post_attention_layernorm.weight",
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "mlp.down_proj.weight",
    ]
    .iter()
    .map(|name| widen(&file, &format!("{prefix}.{name}")))
    .collect::<Result<Vec<_>, _>>()?;

    Ok((
        Inputs {
            config,
            hidden_in: hidden_in.data,
            expected: expected.data,
            mask: mask.data,
            cos_rows,
            sin_rows,
            seq,
        },
        weights,
    ))
}

fn run(inputs: &Inputs, weights: &[Vec<f32>], promotion: Promotion) -> Vec<f32> {
    let layer = TalkerLayerWeights {
        input_layernorm: &weights[0],
        q_proj: &weights[1],
        k_proj: &weights[2],
        v_proj: &weights[3],
        q_norm: &weights[4],
        k_norm: &weights[5],
        o_proj: &weights[6],
        post_attention_layernorm: &weights[7],
        gate_proj: &weights[8],
        up_proj: &weights[9],
        down_proj: &weights[10],
    };
    let mut hidden = inputs.hidden_in.clone();
    let mut cache = KvCache::new();
    forward_layer_with_arithmetic(
        &inputs.config,
        &layer,
        RotaryRows {
            cos: &inputs.cos_rows,
            sin: &inputs.sin_rows,
        },
        &inputs.mask,
        &mut hidden,
        inputs.seq,
        &mut cache,
        promotion.projections,
        promotion.rms,
        promotion.silu,
        promotion.softmax,
        promotion.attention,
    );
    assert_eq!(
        cache.len(),
        inputs.seq,
        "the layer must cache every prefill position"
    );
    hidden
}

#[test]
fn contract_a_l2_talker_layer_00_residual_attribution() {
    let (inputs, weights) = match load() {
        Ok(loaded) => loaded,
        Err(reason) => {
            skip(&reason);
            return;
        }
    };

    let promotions = sweep();
    let mut outputs = Vec::with_capacity(promotions.len());
    let mut rows = Vec::with_capacity(promotions.len());
    for promotion in &promotions {
        let candidate = run(&inputs, &weights, *promotion);
        let comparison = compare_f32(&inputs.expected, &candidate, f64::INFINITY);
        eprintln!(
            "ft7 CPU fp32 ATTRIBUTION promoted={:<38} max_abs={:.6e} relative={:.3e} cosine={:.12}",
            promotion.family,
            comparison.max_abs_diff,
            comparison.max_abs_diff / OUTPUT_SCALE,
            comparison.cosine,
        );
        rows.push(serde_json::json!({
            "promoted": promotion.family,
            "max_abs_vs_oracle": comparison.max_abs_diff,
            "relative_vs_oracle": comparison.max_abs_diff / OUTPUT_SCALE,
            "cosine_vs_oracle": comparison.cosine,
        }));
        outputs.push(candidate);
    }

    let baseline = &outputs[0];
    let exact = outputs.last().expect("the sweep ends with the all-f64 run");
    let baseline_vs_oracle = compare_f32(&inputs.expected, baseline, f64::INFINITY);
    let exact_vs_oracle = compare_f32(&inputs.expected, exact, f64::INFINITY);
    let baseline_vs_exact = compare_f32(exact, baseline, f64::INFINITY);

    // The decisive triangle. `oracle_vs_exact` is the oracle's own distance from f64 arithmetic;
    // if it is the same size as the residual, the oracle is as rounded as we are and no accuracy
    // improvement on our side can reach 0.0.
    eprintln!(
        "ft7 CPU fp32 ATTRIBUTION TRIANGLE ours_f32_vs_oracle={:.6e} oracle_vs_f64={:.6e} \
         ours_f32_vs_f64={:.6e}",
        baseline_vs_oracle.max_abs_diff,
        exact_vs_oracle.max_abs_diff,
        baseline_vs_exact.max_abs_diff,
    );

    assert_eq!(
        baseline_vs_oracle.max_abs_diff, BASELINE_MAX_ABS,
        "the sweep must start from the same f32 arithmetic `talker_layer_l2` froze; if this moved, \
         the production layer changed and every attribution below is about different code"
    );

    // Split the families into carriers and bystanders by how much of the baseline residual each
    // removes on its own. This is the attribution, stated as an assertion rather than a printout.
    let mut carriers = Vec::new();
    let mut bystanders = Vec::new();
    for (promotion, candidate) in promotions.iter().zip(&outputs).skip(1).take(5) {
        let residual = compare_f32(&inputs.expected, candidate, f64::INFINITY).max_abs_diff;
        let removed = 1.0 - residual / baseline_vs_oracle.max_abs_diff;
        assert!(
            !(BYSTANDER_SHARE..=CARRIER_SHARE).contains(&removed),
            "promoting `{}` removed {:.1}% of the residual, which is neither a carrier (>{:.0}%) \
             nor a bystander (<{:.0}%). The attribution this test states no longer holds and must \
             be re-derived, not re-tuned",
            promotion.family,
            removed * 100.0,
            CARRIER_SHARE * 100.0,
            BYSTANDER_SHARE * 100.0,
        );
        if removed > CARRIER_SHARE {
            carriers.push((promotion.family, removed));
        } else {
            bystanders.push((promotion.family, removed));
        }
    }
    for (family, removed) in &carriers {
        eprintln!("ft7 CPU fp32 ATTRIBUTION carrier: {family} removes {:.1}%", removed * 100.0);
    }
    assert!(
        !carriers.is_empty(),
        "no single operation family accounts for the residual; it is spread across families and \
         the sweep must be extended before anything here can be called attributed"
    );

    Receipt::new(TEST_NAME, Outcome::Passed)
        .contract(CONTRACT)
        .seam(SEAM)
        .oracle_tier(OracleTier::CpuFp32Fallback)
        .detail(serde_json::json!({
            "output_scale": OUTPUT_SCALE,
            "promotions": rows,
            "ours_f32_vs_oracle": baseline_vs_oracle.max_abs_diff,
            "oracle_vs_f64": exact_vs_oracle.max_abs_diff,
            "ours_f32_vs_f64": baseline_vs_exact.max_abs_diff,
            "carriers": carriers.iter().map(|(family, removed)| serde_json::json!({
                "family": family,
                "residual_removed": removed,
            })).collect::<Vec<_>>(),
            "bystanders": bystanders.iter().map(|(family, removed)| serde_json::json!({
                "family": family,
                "residual_removed": removed,
            })).collect::<Vec<_>>(),
        }))
        .emit();
}
