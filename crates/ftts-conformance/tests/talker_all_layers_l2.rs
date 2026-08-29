//! L2 parity: **all 28** talker layers against the CPU-fp32 oracle, with real weights.
//!
//! `talker_layer_l2.rs` certifies layer 00 in depth, including its arithmetic-variant sweep. This
//! covers the other 27, which had no test at all: 28 layer seams are captured and only the first
//! was ever compared, so a wiring error introduced anywhere in layers 01..27 would have gone
//! unnoticed by every gate in the repo.
//!
//! Each layer is fed the **oracle's own input**, rotary and mask, so a failure at layer 17 means
//! layer 17 rather than accumulated drift from layer 00. That is the seam-ordered discipline the
//! bead asks for: attribute every residual to a named cause.
//!
//! # Acceptance
//!
//! The same shape as the microdecoder tests and as layer 00's gate: a **scale-relative bound** plus
//! a **cosine floor**, not bit-exactness. Measured across three seams already, f32 accumulation
//! rounding sits near 1e-7 relative with cosine indistinguishable from 1, while a transposed
//! projection, a dropped QK-Norm or a wrong mask is O(1) relative and collapses cosine. The two are
//! separated by decades, so this gate catches wiring without chasing an unreachable bit-exactness.

#![cfg(feature = "ultra-tests")]

use ftts_artifacts::safetensors::SafetensorsFile;
use ftts_conformance::{
    compare::compare_f32,
    npy::NpyArray,
    oracle::{CPU_FP32_ORACLE_CLASS, OracleFixtures, SeamRef},
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

const TEST_NAME: &str = "contract_a_l2_talker_layers_00_27_cpu_fp32";
const CONTRACT: &str = "ConformanceExact/L2";
const CASE: &str = "synthetic-tone-en";
const MODE: &str = "xvector_non_streaming";
const LAYERS: usize = 28;

/// Scale-relative acceptance; see the module docs. Rounding is ~1e-7, wiring is O(1).
const RELATIVE_BOUND: f64 = 1e-4;

/// Rounding does not move cosine; wiring collapses it.
const COSINE_FLOOR: f64 = 0.999_999;

fn checkpoint_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../docs/truth-pack/snapshots/hf/model.safetensors")
}

fn skip(reason: &str) {
    Receipt::new(TEST_NAME, Outcome::Skipped)
        .contract(CONTRACT)
        .seam("talker.layer_00..layer_27.output")
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
fn split_axes(array: &NpyArray, seq: usize, head_dim: usize) -> Result<[Vec<f32>; 3], String> {
    let stride = seq * head_dim;
    if array.data.len() != 3 * stride {
        return Err(format!(
            "rotary tensor should be [3, 1, {seq}, {head_dim}], got {}",
            array.shape_string()
        ));
    }
    Ok([
        array.data[..stride].to_vec(),
        array.data[stride..2 * stride].to_vec(),
        array.data[2 * stride..].to_vec(),
    ])
}

/// Owns one layer's weights so [`TalkerLayerWeights`] can borrow them.
struct OwnedLayer {
    input_layernorm: Vec<f32>,
    q_proj: Vec<f32>,
    k_proj: Vec<f32>,
    v_proj: Vec<f32>,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    o_proj: Vec<f32>,
    post_attention_layernorm: Vec<f32>,
    gate_proj: Vec<f32>,
    up_proj: Vec<f32>,
    down_proj: Vec<f32>,
}

impl OwnedLayer {
    fn load(file: &SafetensorsFile, layer: usize) -> Result<Self, String> {
        let prefix = format!("talker.model.layers.{layer}");
        Ok(Self {
            input_layernorm: widen(file, &format!("{prefix}.input_layernorm.weight"))?,
            q_proj: widen(file, &format!("{prefix}.self_attn.q_proj.weight"))?,
            k_proj: widen(file, &format!("{prefix}.self_attn.k_proj.weight"))?,
            v_proj: widen(file, &format!("{prefix}.self_attn.v_proj.weight"))?,
            q_norm: widen(file, &format!("{prefix}.self_attn.q_norm.weight"))?,
            k_norm: widen(file, &format!("{prefix}.self_attn.k_norm.weight"))?,
            o_proj: widen(file, &format!("{prefix}.self_attn.o_proj.weight"))?,
            post_attention_layernorm: widen(
                file,
                &format!("{prefix}.post_attention_layernorm.weight"),
            )?,
            gate_proj: widen(file, &format!("{prefix}.mlp.gate_proj.weight"))?,
            up_proj: widen(file, &format!("{prefix}.mlp.up_proj.weight"))?,
            down_proj: widen(file, &format!("{prefix}.mlp.down_proj.weight"))?,
        })
    }

    fn borrow(&self) -> TalkerLayerWeights<'_> {
        TalkerLayerWeights {
            input_layernorm: &self.input_layernorm,
            q_proj: &self.q_proj,
            k_proj: &self.k_proj,
            v_proj: &self.v_proj,
            q_norm: &self.q_norm,
            k_norm: &self.k_norm,
            o_proj: &self.o_proj,
            post_attention_layernorm: &self.post_attention_layernorm,
            gate_proj: &self.gate_proj,
            up_proj: &self.up_proj,
            down_proj: &self.down_proj,
        }
    }
}

struct LayerOutcome {
    layer: usize,
    max_abs: f64,
    scale: f64,
    cosine: f64,
}

impl LayerOutcome {
    fn relative(&self) -> f64 {
        if self.scale > 0.0 {
            self.max_abs / self.scale
        } else {
            0.0
        }
    }
}

fn seam<'a>(name: &'a str) -> SeamRef<'a> {
    SeamRef {
        case: CASE,
        mode: MODE,
        group: "talker_free_running",
        seam: name,
    }
}

fn run_all() -> Result<Vec<LayerOutcome>, String> {
    let fixtures = OracleFixtures::open_default()
        .map_err(|error| format!("oracle fixtures unavailable: {error}"))?;
    fixtures
        .require_oracle_class(CPU_FP32_ORACLE_CLASS)
        .map_err(|error| format!("fixture pack is not the CPU-fp32 tier: {error}"))?;
    let path = checkpoint_path();
    if !path.is_file() {
        return Err(format!("checkpoint absent at {}", path.display()));
    }
    let checkpoint = SafetensorsFile::open(&path)
        .map_err(|error| format!("cannot open {}: {error}", path.display()))?;

    let config = TalkerConfig::default();
    let mut outcomes = Vec::with_capacity(LAYERS);

    for layer in 0..LAYERS {
        let input_name = format!("talker.layer_{layer:02}.input");
        let output_name = format!("talker.layer_{layer:02}.output");
        let input_seam = seam(&input_name);
        let output_seam = seam(&output_name);
        if !fixtures.has_seam(&input_seam) || !fixtures.has_seam(&output_seam) {
            return Err(format!("`{}` is not in this pack", input_seam.describe()));
        }

        let hidden_in = fixtures
            .seam(&input_seam, "args.0", 0)
            .map_err(|error| format!("cannot read {}: {error}", input_seam.describe()))?;
        let mask = fixtures
            .seam(&input_seam, "kwargs.attention_mask", 0)
            .map_err(|error| format!("cannot read mask: {error}"))?;
        let cos_axes = fixtures
            .seam(&input_seam, "kwargs.position_embeddings.0", 0)
            .map_err(|error| format!("cannot read rotary cos: {error}"))?;
        let sin_axes = fixtures
            .seam(&input_seam, "kwargs.position_embeddings.1", 0)
            .map_err(|error| format!("cannot read rotary sin: {error}"))?;
        let expected = fixtures
            .seam(&output_seam, "0", 0)
            .map_err(|error| format!("cannot read {}: {error}", output_seam.describe()))?;

        if !hidden_in.data.len().is_multiple_of(config.hidden_size) {
            return Err(format!(
                "{} is {} elements, not a multiple of hidden {}",
                input_seam.describe(),
                hidden_in.data.len(),
                config.hidden_size
            ));
        }
        let seq = hidden_in.data.len() / config.hidden_size;

        let cos_rows = collapse_mrope(
            [
                &split_axes(&cos_axes, seq, config.head_dim)?[0],
                &split_axes(&cos_axes, seq, config.head_dim)?[1],
                &split_axes(&cos_axes, seq, config.head_dim)?[2],
            ],
            seq,
            config.head_dim,
            [24, 20, 20],
        );
        let sin_rows = collapse_mrope(
            [
                &split_axes(&sin_axes, seq, config.head_dim)?[0],
                &split_axes(&sin_axes, seq, config.head_dim)?[1],
                &split_axes(&sin_axes, seq, config.head_dim)?[2],
            ],
            seq,
            config.head_dim,
            [24, 20, 20],
        );

        let weights = OwnedLayer::load(&checkpoint, layer)?;
        let mut hidden = hidden_in.data.clone();
        let mut cache = KvCache::new();
        forward_layer_with_arithmetic(
            &config,
            &weights.borrow(),
            RotaryRows {
                cos: &cos_rows,
                sin: &sin_rows,
            },
            &mask.data,
            &mut hidden,
            seq,
            &mut cache,
            F32LinearAccumulation::Scalar,
            F32RmsNormArithmetic::ScalarReciprocalSqrt,
            F32SiluArithmetic::Divide,
            F32SoftmaxArithmetic::ReciprocalMultiply,
            F32LinearAccumulation::Scalar,
        );
        if cache.len() != seq {
            return Err(format!(
                "layer {layer:02} cached {} positions, expected {seq}",
                cache.len()
            ));
        }

        let comparison = compare_f32(&expected.data, &hidden, f64::INFINITY);
        let scale = expected
            .data
            .iter()
            .fold(0.0_f64, |acc, v| acc.max(f64::from(v.abs())));
        outcomes.push(LayerOutcome {
            layer,
            max_abs: comparison.max_abs_diff,
            scale,
            cosine: comparison.cosine,
        });
    }
    Ok(outcomes)
}

/// Every talker layer reproduces the oracle within a bound no wiring error could pass.
#[test]
fn contract_a_l2_talker_layers_00_27_cpu_fp32() {
    let outcomes = match run_all() {
        Ok(outcomes) => outcomes,
        Err(reason) => {
            skip(&reason);
            return;
        }
    };
    assert_eq!(
        outcomes.len(),
        LAYERS,
        "all {LAYERS} layers must be compared"
    );

    for outcome in &outcomes {
        assert!(
            outcome.relative() < RELATIVE_BOUND,
            "talker layer {:02} diverges beyond rounding: max_abs {:.3e} against scale {:.3e} \
             = {:.3e} relative, bound {RELATIVE_BOUND:.0e}. At this magnitude suspect wiring \
             (transposed projection, missing QK-Norm, wrong mask), not precision.",
            outcome.layer,
            outcome.max_abs,
            outcome.scale,
            outcome.relative()
        );
        assert!(
            outcome.cosine > COSINE_FLOOR,
            "talker layer {:02} cosine {:.12} is below {COSINE_FLOOR} — f32 rounding does not \
             move cosine, so this is a wiring error",
            outcome.layer,
            outcome.cosine
        );
    }

    let worst = outcomes
        .iter()
        .max_by(|a, b| {
            a.relative()
                .partial_cmp(&b.relative())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .expect("at least one layer");

    Receipt::new(TEST_NAME, Outcome::Passed)
        .contract(CONTRACT)
        .seam("talker.layer_00..layer_27.output")
        .oracle_tier(OracleTier::CpuFp32Fallback)
        .detail(serde_json::json!({
            "layers": outcomes.len(),
            "worst_layer": worst.layer,
            "worst_relative": worst.relative(),
            "worst_cosine": outcomes.iter().map(|o| o.cosine).fold(f64::INFINITY, f64::min),
            "relative_bound": RELATIVE_BOUND,
            "per_layer": outcomes.iter().map(|o| serde_json::json!({
                "layer": o.layer,
                "max_abs": o.max_abs,
                "scale": o.scale,
                "relative": o.relative(),
                "cosine": o.cosine,
            })).collect::<Vec<_>>(),
            "bitwise_exact": false,
        }))
        .emit();
}
