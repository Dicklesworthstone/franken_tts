//! L2 parity: the microdecoder's five layers against the CPU-fp32 oracle, with real weights.
//!
//! `microdecoder_head_l2.rs` proves the head projection; this proves the stack that feeds it.
//! For each of the five layers the oracle captured the full 16-position frame — input hidden
//! states, the additive attention mask, and `position_embeddings` — so each layer is fed the
//! **oracle's own input** and compared against the oracle's own output. That decoupling is the
//! point: a layer-3 failure means layer 3, not accumulated drift from layer 0.
//!
//! Two things are separated deliberately:
//!
//! * **Our rotary table is checked on its own** against the oracle's `position_embeddings`, so
//!   "is our RoPE right" never hides inside "is our layer right".
//! * **The layers are then run with the oracle's rotary rows**, isolating the layer arithmetic
//!   (RMSNorm, QK-Norm, GQA, SwiGLU) from the table that feeds it.
//!
//! # Claim tier
//!
//! Bit-exactness is **not** claimed and is recorded as XFAIL, for the reason already measured at
//! the head seam: an f32 dot product over K=1024 costs about `sqrt(K) * eps * |x|`, and the
//! observed divergence sits at that scale rather than at a wiring scale. What *is* asserted is
//! that every layer stays within a scale-relative bound tight enough that a real wiring error —
//! a transposed projection, a missing QK-Norm, a wrong mask — could not pass it. Cosine is
//! asserted too, because a wiring error collapses it while rounding does not.

use ftts_artifacts::safetensors::SafetensorsFile;
use ftts_conformance::{
    compare::compare_f32,
    oracle::{CPU_FP32_ORACLE_CLASS, OracleFixtures, SeamRef},
    report::{OracleTier, Outcome, Receipt},
    xfail,
};
use ftts_model_qwen::microdecoder::{
    FRAME_POSITIONS, FrameKvState, LayerWeights, MicrodecoderConfig, RopeTable,
    layer_step_with_rotary,
};
use std::path::{Path, PathBuf};

const LAYER_TEST: &str = "contract_a_l2_microdecoder_layers_00_04_cpu_fp32";
const ROTARY_TEST: &str = "contract_a_l2_microdecoder_rotary_table_matches_oracle";
const BITWISE_TEST: &str = "contract_a_l2_microdecoder_layers_bitwise_cpu_fp32";
const CONTRACT: &str = "ConformanceExact/L2";
const CASE: &str = "synthetic-tone-en";
const MODE: &str = "xvector_non_streaming";
const LAYERS: usize = 5;
const HIDDEN: usize = 1024;
const HEAD_DIM: usize = 128;

const LEDGER: &str = "frankentts-p1-microdecoder-xst (bead comment: f32 accumulation vs CPU tier)";

/// Scale-relative bound. Rounding over K=1024 lands near `sqrt(K) * eps` ≈ 3.8e-6 relative; a
/// transposed matrix or a dropped norm is O(1) relative, so this separates them by decades.
const RELATIVE_BOUND: f64 = 1e-4;

/// A wiring error collapses cosine; f32 rounding does not move it below this.
const COSINE_FLOOR: f64 = 0.999_999;

fn checkpoint_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../docs/truth-pack/snapshots/hf/model.safetensors")
}

fn skip(test: &str, reason: &str) {
    Receipt::new(test, Outcome::Skipped)
        .contract(CONTRACT)
        .seam("microdecoder.layer_NN.output")
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

/// Every f32 buffer one layer needs, owned so [`LayerWeights`] can borrow it.
struct OwnedLayer {
    input_norm: Vec<f32>,
    q_proj: Vec<f32>,
    k_proj: Vec<f32>,
    v_proj: Vec<f32>,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    o_proj: Vec<f32>,
    post_attention_norm: Vec<f32>,
    gate_proj: Vec<f32>,
    up_proj: Vec<f32>,
    down_proj: Vec<f32>,
}

impl OwnedLayer {
    fn load(file: &SafetensorsFile, layer: usize) -> Result<Self, String> {
        let base = format!("talker.code_predictor.model.layers.{layer}");
        Ok(Self {
            input_norm: widen(file, &format!("{base}.input_layernorm.weight"))?,
            q_proj: widen(file, &format!("{base}.self_attn.q_proj.weight"))?,
            k_proj: widen(file, &format!("{base}.self_attn.k_proj.weight"))?,
            v_proj: widen(file, &format!("{base}.self_attn.v_proj.weight"))?,
            q_norm: widen(file, &format!("{base}.self_attn.q_norm.weight"))?,
            k_norm: widen(file, &format!("{base}.self_attn.k_norm.weight"))?,
            o_proj: widen(file, &format!("{base}.self_attn.o_proj.weight"))?,
            post_attention_norm: widen(file, &format!("{base}.post_attention_layernorm.weight"))?,
            gate_proj: widen(file, &format!("{base}.mlp.gate_proj.weight"))?,
            up_proj: widen(file, &format!("{base}.mlp.up_proj.weight"))?,
            down_proj: widen(file, &format!("{base}.mlp.down_proj.weight"))?,
        })
    }

    fn borrow(&self) -> LayerWeights<'_> {
        LayerWeights {
            input_norm: &self.input_norm,
            q_proj: &self.q_proj,
            k_proj: &self.k_proj,
            v_proj: &self.v_proj,
            q_norm: &self.q_norm,
            k_norm: &self.k_norm,
            o_proj: &self.o_proj,
            post_attention_norm: &self.post_attention_norm,
            gate_proj: &self.gate_proj,
            up_proj: &self.up_proj,
            down_proj: &self.down_proj,
        }
    }
}

/// One layer's measured divergence.
struct LayerOutcome {
    layer: usize,
    max_abs: f64,
    scale: f64,
    cosine: f64,
}

impl LayerOutcome {
    /// `max_abs` expressed against the magnitude of the tensor being compared.
    fn relative(&self) -> f64 {
        if self.scale > 0.0 {
            self.max_abs / self.scale
        } else {
            0.0
        }
    }
}

fn seam<'a>(group: &'a str, name: &'a str) -> SeamRef<'a> {
    SeamRef {
        case: CASE,
        mode: MODE,
        group,
        seam: name,
    }
}

/// Runs all five layers against the oracle, or returns why it could not.
fn run_layers() -> Result<Vec<LayerOutcome>, String> {
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

    let config = MicrodecoderConfig::default();
    let mut outcomes = Vec::with_capacity(LAYERS);

    for layer in 0..LAYERS {
        let input_name = format!("microdecoder.layer_{layer:02}.input");
        let output_name = format!("microdecoder.layer_{layer:02}.output");
        let input_seam = seam("teacher_forced_frame_0000", &input_name);
        let output_seam = seam("teacher_forced_frame_0000", &output_name);
        if !fixtures.has_seam(&input_seam) || !fixtures.has_seam(&output_seam) {
            return Err(format!("`{}` is not in this pack", input_seam.describe()));
        }

        let hidden_in = fixtures
            .seam(&input_seam, "args.0", 0)
            .map_err(|error| format!("cannot read {}: {error}", input_seam.describe()))?;
        let cos = fixtures
            .seam(&input_seam, "kwargs.position_embeddings.0", 0)
            .map_err(|error| format!("cannot read rotary cos: {error}"))?;
        let sin = fixtures
            .seam(&input_seam, "kwargs.position_embeddings.1", 0)
            .map_err(|error| format!("cannot read rotary sin: {error}"))?;
        let expected = fixtures
            .seam(&output_seam, "0", 0)
            .map_err(|error| format!("cannot read {}: {error}", output_seam.describe()))?;

        if hidden_in.data.len() != FRAME_POSITIONS * HIDDEN {
            return Err(format!(
                "{} is {} elements, expected {}",
                input_seam.describe(),
                hidden_in.data.len(),
                FRAME_POSITIONS * HIDDEN
            ));
        }
        if cos.data.len() != FRAME_POSITIONS * HEAD_DIM {
            return Err(format!(
                "rotary cos is {} elements, expected {}",
                cos.data.len(),
                FRAME_POSITIONS * HEAD_DIM
            ));
        }

        let weights = OwnedLayer::load(&checkpoint, layer)?;
        let borrowed = weights.borrow();

        // Fresh KV per layer, and the frame is replayed position by position: this is the
        // sequential loop, not a batched matmul, which is exactly the path being certified.
        let mut state = FrameKvState::new(&config);
        let mut ours = Vec::with_capacity(FRAME_POSITIONS * HIDDEN);
        for position in 0..FRAME_POSITIONS {
            let hidden = &hidden_in.data[position * HIDDEN..(position + 1) * HIDDEN];
            let cos_row = &cos.data[position * HEAD_DIM..(position + 1) * HEAD_DIM];
            let sin_row = &sin.data[position * HEAD_DIM..(position + 1) * HEAD_DIM];
            let out =
                layer_step_with_rotary(&config, &borrowed, cos_row, sin_row, hidden, &mut state);
            ours.extend_from_slice(&out);
        }

        let comparison = compare_f32(&expected.data, &ours, f64::INFINITY);
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

/// Our rotary table must reproduce the oracle's `position_embeddings`.
#[test]
fn contract_a_l2_microdecoder_rotary_table_matches_oracle() {
    let fixtures = match OracleFixtures::open_default() {
        Ok(fixtures) => fixtures,
        Err(error) => {
            skip(
                ROTARY_TEST,
                &format!("oracle fixtures unavailable: {error}"),
            );
            return;
        }
    };
    let input_seam = seam("teacher_forced_frame_0000", "microdecoder.layer_00.input");
    if !fixtures.has_seam(&input_seam) {
        skip(ROTARY_TEST, "layer_00.input is not in this pack");
        return;
    }
    let cos = match fixtures.seam(&input_seam, "kwargs.position_embeddings.0", 0) {
        Ok(cos) => cos,
        Err(error) => {
            skip(ROTARY_TEST, &format!("cannot read rotary cos: {error}"));
            return;
        }
    };
    let sin = match fixtures.seam(&input_seam, "kwargs.position_embeddings.1", 0) {
        Ok(sin) => sin,
        Err(error) => {
            skip(ROTARY_TEST, &format!("cannot read rotary sin: {error}"));
            return;
        }
    };

    let config = MicrodecoderConfig::default();
    let table = RopeTable::new(&config);
    let (mut worst_cos, mut worst_sin) = (0.0_f64, 0.0_f64);
    for position in 0..FRAME_POSITIONS {
        let (ours_cos, ours_sin) = table.row(position);
        for i in 0..HEAD_DIM {
            let index = position * HEAD_DIM + i;
            worst_cos = worst_cos.max(f64::from((ours_cos[i] - cos.data[index]).abs()));
            worst_sin = worst_sin.max(f64::from((ours_sin[i] - sin.data[index]).abs()));
        }
    }

    assert!(
        worst_cos < 1e-6 && worst_sin < 1e-6,
        "our plain-RoPE table (theta 1e6, 16 positions) diverges from the oracle's \
         position_embeddings: max |dcos| {worst_cos:.3e}, max |dsin| {worst_sin:.3e}. \
         A wrong theta or an interleaved-vs-half-split convention shows up here."
    );

    Receipt::new(ROTARY_TEST, Outcome::Passed)
        .contract(CONTRACT)
        .seam("microdecoder.layer_00.input/position_embeddings")
        .oracle_tier(OracleTier::CpuFp32Fallback)
        .detail(serde_json::json!({
            "positions": FRAME_POSITIONS,
            "max_abs_cos": worst_cos,
            "max_abs_sin": worst_sin,
        }))
        .emit();
}

/// All five layers reproduce the oracle within a bound no wiring error could pass.
#[test]
fn contract_a_l2_microdecoder_layers_00_04_cpu_fp32() {
    let outcomes = match run_layers() {
        Ok(outcomes) => outcomes,
        Err(reason) => {
            skip(LAYER_TEST, &reason);
            return;
        }
    };
    assert_eq!(outcomes.len(), LAYERS, "all five layers must be compared");

    for outcome in &outcomes {
        assert!(
            outcome.relative() < RELATIVE_BOUND,
            "layer {:02} diverges beyond rounding: max_abs {:.3e} against scale {:.3e} \
             = {:.3e} relative, bound {RELATIVE_BOUND:.0e}. At this magnitude suspect wiring \
             (transposed projection, missing QK-Norm, wrong mask), not precision.",
            outcome.layer,
            outcome.max_abs,
            outcome.scale,
            outcome.relative()
        );
        assert!(
            outcome.cosine > COSINE_FLOOR,
            "layer {:02} cosine {:.9} is below {COSINE_FLOOR} — rounding does not move cosine, \
             so this is a wiring error",
            outcome.layer,
            outcome.cosine
        );
    }

    Receipt::new(LAYER_TEST, Outcome::Passed)
        .contract(CONTRACT)
        .seam("microdecoder.layer_00..layer_04.output")
        .oracle_tier(OracleTier::CpuFp32Fallback)
        .detail(serde_json::json!({
            "layers": outcomes.len(),
            "per_layer": outcomes.iter().map(|o| serde_json::json!({
                "layer": o.layer,
                "max_abs": o.max_abs,
                "scale": o.scale,
                "relative": o.relative(),
                "cosine": o.cosine,
            })).collect::<Vec<_>>(),
            "relative_bound": RELATIVE_BOUND,
            "bitwise_exact": false,
        }))
        .emit();
}

/// Bit-exactness at the layer seams: XFAIL with its measured magnitude, never asserted.
#[test]
fn contract_a_l2_microdecoder_layers_bitwise_cpu_fp32() {
    let outcomes = match run_layers() {
        Ok(outcomes) => outcomes,
        Err(reason) => {
            skip(BITWISE_TEST, &reason);
            return;
        }
    };
    let worst = outcomes
        .iter()
        .map(|outcome| outcome.max_abs)
        .fold(0.0_f64, f64::max);
    let worst_relative = outcomes
        .iter()
        .map(LayerOutcome::relative)
        .fold(0.0_f64, f64::max);

    let still_diverging = xfail(BITWISE_TEST, CONTRACT, LEDGER, || {
        if worst == 0.0 {
            Ok(())
        } else {
            let expected = (HIDDEN as f64).sqrt() * f64::from(f32::EPSILON);
            Err(format!(
                "microdecoder layers are not bit-exact against the CPU-tier oracle: worst max_abs \
                 {worst:.3e}, worst relative {worst_relative:.3e} over {LAYERS} layers. An f32 \
                 accumulation over K={HIDDEN} costs about sqrt(K)*eps = {expected:.3e} relative, \
                 so this is {:.2}x the rounding budget — the same cause measured at the head seam \
                 and at talker layer 00 (e481fa8), not wiring.",
                worst_relative / expected
            ))
        }
    });

    assert!(
        still_diverging,
        "the microdecoder layers became bit-exact — retire the ledger entry and promote this to a \
         plain assertion"
    );
}
