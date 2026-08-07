//! L2 parity: one talker decoder layer against the CPU-fp32 oracle seam.
//!
//! The mandate is that a stage without a fixture comparison is not done. This is that comparison
//! for `ftts_model_qwen::talker::forward_layer`: feed layer 00 the oracle's own captured input,
//! rotary tables and attention mask, run our layer with the real checkpoint weights, and compare
//! against the oracle's captured output.
//!
//! Model-gated twice over — it needs both the fixture set and the pinned checkpoint — and skips
//! with SUCCESS and a named reason when either is absent, never folding a skip into green.

use ftts_artifacts::safetensors::SafetensorsFile;
use ftts_conformance::npy::NpyArray;
use ftts_conformance::{
    compare::compare_f32,
    oracle::{
        CPU_TIER_TOLERANCE, CPU_TIER_TOLERANCE_SOURCE, OracleFixtures, SeamRef, compare_exactly,
    },
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

const TEST_NAME: &str = "contract_a_l2_talker_layer_00_cpu_fp32_exact";
const FIRST_OBSERVED_MAX_ABS: f64 = 9.536_743_164_062_5e-6;
const FIRST_OBSERVED_OVER_TOLERANCE: usize = 27_166;

/// Largest magnitude in the oracle's `talker.layer_00.output`, so `max_abs` can be read against
/// the scale it belongs to instead of as a bare number.
const OUTPUT_SCALE: f64 = 16.365_345;

/// Scale-relative acceptance. Measured rounding sits near 6e-7 relative; a transposed projection,
/// a dropped QK-Norm or a wrong mask is O(1), so this separates cause from cause by decades.
///
/// This is the assertion that actually gates wiring. The frozen `FIRST_OBSERVED_*` equalities
/// below pin the exact arithmetic against silent change; they do not, on their own, distinguish a
/// rounding difference from a broken layer.
const RELATIVE_BOUND: f64 = 1e-4;

/// Rounding does not move cosine; wiring collapses it.
const COSINE_FLOOR: f64 = 0.999_999;

/// The pinned talker checkpoint, alongside the truth-pack snapshots.
fn checkpoint_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../docs/truth-pack/snapshots/hf/model.safetensors")
}

fn skip(reason: &str) {
    Receipt::new(TEST_NAME, Outcome::Skipped)
        .contract("ConformanceExact/L2")
        .seam("talker.layer_00.output")
        .reason(reason)
        .tolerance(CPU_TIER_TOLERANCE, CPU_TIER_TOLERANCE_SOURCE)
        .oracle_tier(OracleTier::CpuFp32Fallback)
        .emit();
}

/// Widen a whole tensor to `f32` through the accessor, which is how the engine reads BF16.
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

/// The oracle stores `[3, batch, seq, head_dim]`; split it into the three axis rows.
fn split_axes(array: &NpyArray, seq: usize, head_dim: usize) -> [Vec<f32>; 3] {
    let stride = seq * head_dim;
    assert_eq!(
        array.data.len(),
        3 * stride,
        "rotary tensor should be [3, 1, seq, head_dim], got {}",
        array.shape_string()
    );
    [
        array.data[..stride].to_vec(),
        array.data[stride..2 * stride].to_vec(),
        array.data[2 * stride..].to_vec(),
    ]
}

// One argument per arithmetic-mode axis the sweep varies; a struct would only rename them.
#[allow(clippy::too_many_arguments)]
fn run_layer(
    config: &TalkerConfig,
    weights: &TalkerLayerWeights<'_>,
    rotary: RotaryRows<'_>,
    mask: &[f32],
    hidden_in: &[f32],
    seq: usize,
    accumulation: F32LinearAccumulation,
    rms_arithmetic: F32RmsNormArithmetic,
    silu_arithmetic: F32SiluArithmetic,
    softmax_arithmetic: F32SoftmaxArithmetic,
    attention_accumulation: F32LinearAccumulation,
) -> Vec<f32> {
    let mut hidden = hidden_in.to_vec();
    let mut cache = KvCache::new();
    forward_layer_with_arithmetic(
        config,
        weights,
        rotary,
        mask,
        &mut hidden,
        seq,
        &mut cache,
        accumulation,
        rms_arithmetic,
        silu_arithmetic,
        softmax_arithmetic,
        attention_accumulation,
    );
    assert_eq!(
        cache.len(),
        seq,
        "the layer must cache every prefill position"
    );
    hidden
}

#[test]
fn contract_a_l2_talker_layer_00_cpu_fp32_exact() {
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

    let input_seam = SeamRef {
        case: "synthetic-tone-en",
        mode: "icl_non_streaming",
        group: "talker_free_running",
        seam: "talker.layer_00.input",
    };
    let output_seam = SeamRef {
        seam: "talker.layer_00.output",
        ..input_seam
    };
    if !fixtures.has_seam(&input_seam) {
        skip(&format!("seam absent: {}", input_seam.describe()));
        return;
    }

    // Step 000 is the prefill pass: [1, seq, hidden].
    let hidden_in = fixtures
        .seam(&input_seam, "args.0", 0)
        .expect("layer input");
    let expected = fixtures.seam(&output_seam, "0", 0).expect("layer output");
    let cos = fixtures
        .seam(&input_seam, "kwargs.position_embeddings.0", 0)
        .expect("cos");
    let sin = fixtures
        .seam(&input_seam, "kwargs.position_embeddings.1", 0)
        .expect("sin");
    let mask = fixtures
        .seam(&input_seam, "kwargs.attention_mask", 0)
        .expect("mask");

    let config = TalkerConfig::default();
    let hidden_size = config.hidden_size;
    let head_dim = config.head_dim;
    let seq = hidden_in.data.len() / hidden_size;
    assert_eq!(
        hidden_in.data.len(),
        seq * hidden_size,
        "layer input should be a whole number of {hidden_size}-wide rows"
    );
    assert_eq!(
        mask.data.len(),
        seq * seq,
        "prefill mask should be [seq, seq], got {}",
        mask.shape_string()
    );

    let file = SafetensorsFile::open(&checkpoint).expect("pinned checkpoint opens");
    let prefix = "talker.model.layers.0";
    let input_layernorm = widen(&file, &format!("{prefix}.input_layernorm.weight"))
        .expect("input layer norm must be present and readable");
    let q_proj = widen(&file, &format!("{prefix}.self_attn.q_proj.weight"))
        .expect("query projection must be present and readable");
    let k_proj = widen(&file, &format!("{prefix}.self_attn.k_proj.weight"))
        .expect("key projection must be present and readable");
    let v_proj = widen(&file, &format!("{prefix}.self_attn.v_proj.weight"))
        .expect("value projection must be present and readable");
    let q_norm = widen(&file, &format!("{prefix}.self_attn.q_norm.weight"))
        .expect("query norm must be present and readable");
    let k_norm = widen(&file, &format!("{prefix}.self_attn.k_norm.weight"))
        .expect("key norm must be present and readable");
    let o_proj = widen(&file, &format!("{prefix}.self_attn.o_proj.weight"))
        .expect("output projection must be present and readable");
    let post_attention_layernorm =
        widen(&file, &format!("{prefix}.post_attention_layernorm.weight"))
            .expect("post-attention layer norm must be present and readable");
    let gate_proj = widen(&file, &format!("{prefix}.mlp.gate_proj.weight"))
        .expect("gate projection must be present and readable");
    let up_proj = widen(&file, &format!("{prefix}.mlp.up_proj.weight"))
        .expect("up projection must be present and readable");
    let down_proj = widen(&file, &format!("{prefix}.mlp.down_proj.weight"))
        .expect("down projection must be present and readable");

    let weights = TalkerLayerWeights {
        input_layernorm: &input_layernorm,
        q_proj: &q_proj,
        k_proj: &k_proj,
        v_proj: &v_proj,
        q_norm: &q_norm,
        k_norm: &k_norm,
        o_proj: &o_proj,
        post_attention_layernorm: &post_attention_layernorm,
        gate_proj: &gate_proj,
        up_proj: &up_proj,
        down_proj: &down_proj,
    };

    let cos_axes = split_axes(&cos, seq, head_dim);
    let sin_axes = split_axes(&sin, seq, head_dim);
    let sections = [24usize, 20, 20];
    let cos_rows = collapse_mrope(
        [&cos_axes[0], &cos_axes[1], &cos_axes[2]],
        seq,
        head_dim,
        sections,
    );
    let sin_rows = collapse_mrope(
        [&sin_axes[0], &sin_axes[1], &sin_axes[2]],
        seq,
        head_dim,
        sections,
    );

    let rotary = RotaryRows {
        cos: &cos_rows,
        sin: &sin_rows,
    };
    let hidden = run_layer(
        &config,
        &weights,
        rotary,
        &mask.data,
        &hidden_in.data,
        seq,
        F32LinearAccumulation::Scalar,
        F32RmsNormArithmetic::ScalarReciprocalSqrt,
        F32SiluArithmetic::Divide,
        F32SoftmaxArithmetic::ReciprocalMultiply,
        F32LinearAccumulation::Scalar,
    );
    for accumulation in [F32LinearAccumulation::Lanes4, F32LinearAccumulation::Lanes8] {
        let candidate = run_layer(
            &config,
            &weights,
            rotary,
            &mask.data,
            &hidden_in.data,
            seq,
            accumulation,
            F32RmsNormArithmetic::ScalarReciprocalSqrt,
            F32SiluArithmetic::Divide,
            F32SoftmaxArithmetic::ReciprocalMultiply,
            F32LinearAccumulation::Scalar,
        );
        let comparison = compare_f32(&expected.data, &candidate, CPU_TIER_TOLERANCE);
        eprintln!(
            "ft7 CPU fp32 GEMM_BISECT accumulation={accumulation:?} max_abs={:.16e} over_tolerance={}/{} cosine={:.12}",
            comparison.max_abs_diff, comparison.over_tolerance, comparison.len, comparison.cosine,
        );
    }
    let candidate = run_layer(
        &config,
        &weights,
        rotary,
        &mask.data,
        &hidden_in.data,
        seq,
        F32LinearAccumulation::Accelerate,
        F32RmsNormArithmetic::ScalarReciprocalSqrt,
        F32SiluArithmetic::Divide,
        F32SoftmaxArithmetic::ReciprocalMultiply,
        F32LinearAccumulation::Scalar,
    );
    let comparison = compare_f32(&expected.data, &candidate, CPU_TIER_TOLERANCE);
    eprintln!(
        "ft7 CPU fp32 ACCELERATE_SGEMM_BISECT max_abs={:.16e} over_tolerance={}/{} cosine={:.12}",
        comparison.max_abs_diff, comparison.over_tolerance, comparison.len, comparison.cosine,
    );
    let candidate = run_layer(
        &config,
        &weights,
        rotary,
        &mask.data,
        &hidden_in.data,
        seq,
        F32LinearAccumulation::Accelerate,
        F32RmsNormArithmetic::Lanes4ReciprocalSqrt,
        F32SiluArithmetic::Divide,
        F32SoftmaxArithmetic::Divide,
        F32LinearAccumulation::Scalar,
    );
    let comparison = compare_f32(&expected.data, &candidate, CPU_TIER_TOLERANCE);
    eprintln!(
        "ft7 CPU fp32 ACCELERATE_GEMM_RMSNORM_SOFTMAX_BISECT max_abs={:.16e} over_tolerance={}/{} cosine={:.12}",
        comparison.max_abs_diff, comparison.over_tolerance, comparison.len, comparison.cosine,
    );
    let candidate = run_layer(
        &config,
        &weights,
        rotary,
        &mask.data,
        &hidden_in.data,
        seq,
        F32LinearAccumulation::Accelerate,
        F32RmsNormArithmetic::Lanes4ReciprocalSqrt,
        F32SiluArithmetic::Divide,
        F32SoftmaxArithmetic::Divide,
        F32LinearAccumulation::Accelerate,
    );
    let comparison = compare_f32(&expected.data, &candidate, CPU_TIER_TOLERANCE);
    eprintln!(
        "ft7 CPU fp32 ACCELERATE_FULL_ATTENTION_BISECT max_abs={:.16e} over_tolerance={}/{} cosine={:.12}",
        comparison.max_abs_diff, comparison.over_tolerance, comparison.len, comparison.cosine,
    );
    for accumulation in [
        F32LinearAccumulation::FusedLanes4,
        F32LinearAccumulation::FusedLanes8,
    ] {
        let candidate = run_layer(
            &config,
            &weights,
            rotary,
            &mask.data,
            &hidden_in.data,
            seq,
            accumulation,
            F32RmsNormArithmetic::Lanes4ReciprocalSqrt,
            F32SiluArithmetic::Divide,
            F32SoftmaxArithmetic::ReciprocalMultiply,
            F32LinearAccumulation::Scalar,
        );
        let comparison = compare_f32(&expected.data, &candidate, CPU_TIER_TOLERANCE);
        eprintln!(
            "ft7 CPU fp32 GEMM_RMSNORM_BISECT accumulation={accumulation:?} rms=Lanes4ReciprocalSqrt max_abs={:.16e} over_tolerance={}/{} cosine={:.12}",
            comparison.max_abs_diff, comparison.over_tolerance, comparison.len, comparison.cosine,
        );
    }
    for rms_arithmetic in [
        F32RmsNormArithmetic::ScalarDivideSqrt,
        F32RmsNormArithmetic::Lanes4ReciprocalSqrt,
        F32RmsNormArithmetic::Lanes8ReciprocalSqrt,
        F32RmsNormArithmetic::F64ReciprocalSqrt,
    ] {
        let candidate = run_layer(
            &config,
            &weights,
            rotary,
            &mask.data,
            &hidden_in.data,
            seq,
            F32LinearAccumulation::Scalar,
            rms_arithmetic,
            F32SiluArithmetic::Divide,
            F32SoftmaxArithmetic::ReciprocalMultiply,
            F32LinearAccumulation::Scalar,
        );
        let comparison = compare_f32(&expected.data, &candidate, CPU_TIER_TOLERANCE);
        eprintln!(
            "ft7 CPU fp32 RMSNORM_BISECT arithmetic={rms_arithmetic:?} max_abs={:.16e} over_tolerance={}/{} cosine={:.12}",
            comparison.max_abs_diff, comparison.over_tolerance, comparison.len, comparison.cosine,
        );
    }
    let candidate = run_layer(
        &config,
        &weights,
        rotary,
        &mask.data,
        &hidden_in.data,
        seq,
        F32LinearAccumulation::Scalar,
        F32RmsNormArithmetic::Lanes4ReciprocalSqrt,
        F32SiluArithmetic::MultiplyReciprocal,
        F32SoftmaxArithmetic::ReciprocalMultiply,
        F32LinearAccumulation::Scalar,
    );
    let comparison = compare_f32(&expected.data, &candidate, CPU_TIER_TOLERANCE);
    eprintln!(
        "ft7 CPU fp32 SILU_RMSNORM_BISECT silu=MultiplyReciprocal rms=Lanes4ReciprocalSqrt max_abs={:.16e} over_tolerance={}/{} cosine={:.12}",
        comparison.max_abs_diff, comparison.over_tolerance, comparison.len, comparison.cosine,
    );

    let candidate = run_layer(
        &config,
        &weights,
        rotary,
        &mask.data,
        &hidden_in.data,
        seq,
        F32LinearAccumulation::Scalar,
        F32RmsNormArithmetic::Lanes4ReciprocalSqrt,
        F32SiluArithmetic::Divide,
        F32SoftmaxArithmetic::Divide,
        F32LinearAccumulation::Scalar,
    );
    let comparison = compare_f32(&expected.data, &candidate, CPU_TIER_TOLERANCE);
    eprintln!(
        "ft7 CPU fp32 SOFTMAX_RMSNORM_BISECT softmax=Divide rms=Lanes4ReciprocalSqrt max_abs={:.16e} over_tolerance={}/{} cosine={:.12}",
        comparison.max_abs_diff, comparison.over_tolerance, comparison.len, comparison.cosine,
    );
    for attention_accumulation in [
        F32LinearAccumulation::Lanes4,
        F32LinearAccumulation::Lanes8,
        F32LinearAccumulation::FusedLanes4,
    ] {
        let candidate = run_layer(
            &config,
            &weights,
            rotary,
            &mask.data,
            &hidden_in.data,
            seq,
            F32LinearAccumulation::Scalar,
            F32RmsNormArithmetic::Lanes4ReciprocalSqrt,
            F32SiluArithmetic::Divide,
            F32SoftmaxArithmetic::Divide,
            attention_accumulation,
        );
        let comparison = compare_f32(&expected.data, &candidate, CPU_TIER_TOLERANCE);
        eprintln!(
            "ft7 CPU fp32 ATTENTION_RMSNORM_SOFTMAX_BISECT attention={attention_accumulation:?} rms=Lanes4ReciprocalSqrt softmax=Divide max_abs={:.16e} over_tolerance={}/{} cosine={:.12}",
            comparison.max_abs_diff, comparison.over_tolerance, comparison.len, comparison.cosine,
        );
    }

    let comparison = compare_f32(&expected.data, &hidden, CPU_TIER_TOLERANCE);
    eprintln!(
        "ft7 CPU fp32 stage={} max_abs={:.16e} over_tolerance={}/{} cosine={:.12}",
        output_seam.describe(),
        comparison.max_abs_diff,
        comparison.over_tolerance,
        comparison.len,
        comparison.cosine,
    );
    if comparison.holds() {
        compare_exactly(&output_seam.describe(), &expected, &hidden)
            .expect("a zero-difference summary must satisfy the exact comparator");
        Receipt::new(TEST_NAME, Outcome::Failed)
            .contract("ConformanceExact/L2")
            .seam(output_seam.describe())
            .reason("unexpected exact pass: remove the recorded XFAIL only after reviewing the arithmetic change")
            .tolerance(CPU_TIER_TOLERANCE, CPU_TIER_TOLERANCE_SOURCE)
            .oracle_tier(OracleTier::CpuFp32Fallback)
            .detail(comparison.to_json())
            .emit();
        assert!(
            !comparison.holds(),
            "talker layer 00 unexpectedly achieved exact CPU-fp32 parity"
        );
    }

    // The real gate: rounding-scale divergence is acceptable, wiring-scale divergence is not.
    // Without this the test only froze two numbers and would have passed a transposed projection
    // that happened to reproduce them, or failed loudly for a harmless last-bit change.
    let relative = comparison.max_abs_diff / OUTPUT_SCALE;
    assert!(
        relative < RELATIVE_BOUND,
        "talker layer 00 diverges beyond rounding: max_abs {:.3e} against scale {OUTPUT_SCALE} \
         = {relative:.3e} relative, bound {RELATIVE_BOUND:.0e}. At this magnitude suspect wiring \
         (transposed projection, missing QK-Norm, wrong mask), not precision.",
        comparison.max_abs_diff
    );
    assert!(
        comparison.cosine > COSINE_FLOOR,
        "talker layer 00 cosine {:.12} is below {COSINE_FLOOR} — f32 rounding does not move \
         cosine, so this is a wiring error, not an arithmetic one",
        comparison.cosine
    );

    let report = compare_exactly(&output_seam.describe(), &expected, &hidden)
        .expect_err("a nonzero exact comparison must return its localized report");
    // The divergence is attributed, not left open: 9.54e-6 against a scale of 16.37 is 5.83e-7
    // relative, i.e. 4.9 ULP of f32 and 0.15x the sqrt(K)*eps a K=1024 accumulation costs. The
    // same cause, at the same scale, is measured at the microdecoder head seam (0.72x budget) and
    // across microdecoder layers 00..04 (0.05x budget) — three independent seams, all rounding.
    // `over_tolerance = 27166` is an artefact of comparing against a tolerance of exactly 0.0,
    // not evidence of breadth.
    Receipt::new(TEST_NAME, Outcome::ExpectedFailure)
        .contract("ConformanceExact/L2")
        .seam(output_seam.describe())
        .reason(
            "CPU-fp32 accumulation rounding: max_abs 9.54e-6 / scale 16.37 = 5.83e-7 relative \
             (4.9 ULP, 0.15x the sqrt(1024)*eps budget), cosine 0.9999999999999. Exact compare is \
             not achievable in f32 at this tier; the scale-relative bound and cosine floor are \
             the acceptance criteria that gate wiring.",
        )
        .tolerance(CPU_TIER_TOLERANCE, CPU_TIER_TOLERANCE_SOURCE)
        .oracle_tier(OracleTier::CpuFp32Fallback)
        .detail(comparison.to_json())
        .emit();
    eprintln!("XFAIL[{TEST_NAME}]: {report}");
    assert_eq!(comparison.max_abs_diff, FIRST_OBSERVED_MAX_ABS);
    assert_eq!(comparison.over_tolerance, FIRST_OBSERVED_OVER_TOLERANCE);
}
