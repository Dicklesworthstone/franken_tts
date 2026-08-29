//! L2 parity: the microdecoder's per-depth heads against the oracle, with the real checkpoint.
//!
//! This is the weight-gated half of `frankentts-p1-microdecoder-xst`. The companion test
//! `microdecoder_l2.rs` proves the head-to-code *wiring* using only fixtures; this one runs the
//! actual projection `lm_head[d-1] @ head_d.input` through the BF16 accessor and compares against
//! the oracle's captured `head_d.output`.
//!
//! # Two claims, deliberately separated
//!
//! **Argmax is exact and is asserted.** For every depth in every prompt mode, our projection's
//! argmax equals the oracle's, and equals `codec_codes[d]`. This is the claim OQ-5 §6 licenses:
//! the sequential loop and a batched forward agree in exact arithmetic only, so strict acceptance
//! compares token ids, never logit bits.
//!
//! **Bit-exactness does not hold, and is recorded as XFAIL rather than asserted or hidden.** The
//! contract addendum says the CPU tier's measured nondeterminism floor is `max_abs 0.0` at every
//! observed seam, i.e. exact compare. It does not hold here. The divergence is attributed to a
//! named cause rather than waved at:
//!
//! * observed worst `max_abs` = 4.005e-5 over all 60 comparisons, against a logit scale of
//!   `max|logit|` ≈ 14.6 — so 2.7e-6 relative to scale, about **23 ULP** of f32 (eps 1.19e-7);
//! * that is what a 1024-term f32 dot product should cost: `sqrt(K) * eps * |logit|` with K = 1024
//!   is 5.57e-5, and the observed error is **0.72x** that bound. The magnitude is explained by
//!   accumulation rounding, not by wiring;
//! * f64 accumulation shrinks but does not remove it, and neither does a naive sequential f32 sum,
//!   so it is not our reduction order alone;
//! * the argmax is unaffected at every depth in every mode — all 60 select the oracle's token.
//!
//! Beware the per-element relative error, which reaches 6.7e-3: that is an artefact of dividing by
//! near-zero logits, not evidence of a large error. `max_abs` against the logit scale is the
//! meaningful figure. The same shape of divergence is already recorded for talker layer 00
//! (e481fa8), so this is a property of the CPU-tier capture rather than of one seam.
//!
//! XFAIL, never SKIP: the check keeps executing, so the day the divergence disappears — a fixture
//! regeneration, or a matching accumulation order — the unexpected pass fails loudly and this
//! ledger entry gets retired instead of quietly rotting.

#![cfg(feature = "ultra-tests")]

use ftts_artifacts::safetensors::SafetensorsFile;
use ftts_conformance::{
    npy,
    oracle::{CPU_FP32_ORACLE_CLASS, OracleFixtures, SeamRef},
    report::{OracleTier, Outcome, Receipt},
    xfail,
};
use ftts_model_qwen::microdecoder::{RESIDUAL_DEPTHS, RESIDUAL_VOCAB, argmax};
use std::path::{Path, PathBuf};

const ARGMAX_TEST: &str = "contract_a_l2_microdecoder_head_projection_argmax_cpu_fp32_exact";
const BITWISE_TEST: &str = "contract_a_l2_microdecoder_head_projection_bitwise_cpu_fp32";
const CONTRACT: &str = "ConformanceExact/L2";
const CASE: &str = "synthetic-tone-en";
const MODES: [&str; 4] = [
    "xvector_non_streaming",
    "xvector_streaming",
    "icl_non_streaming",
    "icl_streaming",
];
const HIDDEN: usize = 1024;

/// Observed magnitude of the largest logit in this fixture set, used to express `max_abs` as a
/// fraction of scale instead of as a bare number.
const LOGIT_SCALE: f64 = 14.6;

/// Where the divergence below is recorded, so an unexpected pass names its ledger entry.
const LEDGER: &str =
    "frankentts-p1-microdecoder-xst (bead comment: head projection ULP divergence)";

fn checkpoint_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../docs/truth-pack/snapshots/hf/model.safetensors")
}

fn skip(test: &str, reason: &str) {
    Receipt::new(test, Outcome::Skipped)
        .contract(CONTRACT)
        .seam("microdecoder.head_NN.output")
        .reason(reason)
        .oracle_tier(OracleTier::CpuFp32Fallback)
        .emit();
}

/// Widens a whole BF16 tensor to f32 through the accessor, which is how the engine reads weights.
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

/// `out[o] = sum_i weight[o * HIDDEN + i] * x[i]`, the projection under test.
fn project(weight: &[f32], x: &[f32]) -> Vec<f32> {
    weight
        .chunks_exact(x.len())
        .map(|row| row.iter().zip(x.iter()).map(|(w, v)| w * v).sum())
        .collect()
}

/// One depth's comparison, in one mode.
struct DepthOutcome {
    ours: usize,
    oracle: usize,
    expected_code: usize,
    max_abs: f64,
    max_rel: f64,
}

/// Runs every depth in every mode, or returns the reason it could not.
fn run_all() -> Result<Vec<DepthOutcome>, String> {
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

    let mut outcomes = Vec::new();
    for mode in MODES {
        let codes_seam = SeamRef {
            case: CASE,
            mode,
            group: "talker_free_running",
            seam: "talker.codec_codes",
        };
        if !fixtures.has_seam(&codes_seam) {
            return Err(format!("`{}` is not in this pack", codes_seam.describe()));
        }
        let codes = npy::read_i64(&fixtures.seam_path(&codes_seam, "tensor", 0))
            .map_err(|error| format!("cannot read {}: {error}", codes_seam.describe()))?;

        for depth in 1..=RESIDUAL_DEPTHS {
            let input_name = format!("microdecoder.head_{depth:02}.input");
            let output_name = format!("microdecoder.head_{depth:02}.output");
            let input_seam = SeamRef {
                case: CASE,
                mode,
                group: "teacher_forced_frame_0000",
                seam: &input_name,
            };
            let output_seam = SeamRef {
                case: CASE,
                mode,
                group: "teacher_forced_frame_0000",
                seam: &output_name,
            };
            if !fixtures.has_seam(&input_seam) || !fixtures.has_seam(&output_seam) {
                return Err(format!("`{}` is not in this pack", input_seam.describe()));
            }

            let x = fixtures
                .seam(&input_seam, "args.0", 0)
                .map_err(|error| format!("cannot read {}: {error}", input_seam.describe()))?;
            let expected = fixtures
                .seam(&output_seam, "tensor", 0)
                .map_err(|error| format!("cannot read {}: {error}", output_seam.describe()))?;

            if x.data.len() != HIDDEN {
                return Err(format!(
                    "{} is {} wide, expected {HIDDEN}",
                    input_seam.describe(),
                    x.data.len()
                ));
            }
            if expected.data.len() != RESIDUAL_VOCAB {
                return Err(format!(
                    "{} is {} wide, expected {RESIDUAL_VOCAB}",
                    output_seam.describe(),
                    expected.data.len()
                ));
            }

            // Head d scores position d, so it is `lm_head[d - 1]`.
            let weight_name = format!("talker.code_predictor.lm_head.{}.weight", depth - 1);
            let weight = widen(&checkpoint, &weight_name)?;
            if weight.len() != RESIDUAL_VOCAB * HIDDEN {
                return Err(format!(
                    "`{weight_name}` has {} elements, expected {}",
                    weight.len(),
                    RESIDUAL_VOCAB * HIDDEN
                ));
            }

            let ours = project(&weight, &x.data);
            let (mut max_abs, mut max_rel) = (0.0_f64, 0.0_f64);
            for (a, b) in ours.iter().zip(expected.data.iter()) {
                let diff = f64::from(*a) - f64::from(*b);
                max_abs = max_abs.max(diff.abs());
                let scale = f64::from(a.abs()).max(f64::from(b.abs()));
                if scale > 0.0 {
                    max_rel = max_rel.max(diff.abs() / scale);
                }
            }

            outcomes.push(DepthOutcome {
                ours: argmax(&ours),
                oracle: argmax(&expected.data),
                expected_code: usize::try_from(codes.data[depth])
                    .map_err(|_| format!("negative code id at depth {depth}"))?,
                max_abs,
                max_rel,
            });
        }
    }
    Ok(outcomes)
}

/// The licensed claim: our projection selects the same token as the oracle, at every depth.
#[test]
fn contract_a_l2_microdecoder_head_projection_argmax_cpu_fp32_exact() {
    let outcomes = match run_all() {
        Ok(outcomes) => outcomes,
        Err(reason) => {
            skip(ARGMAX_TEST, &reason);
            return;
        }
    };

    assert_eq!(
        outcomes.len(),
        MODES.len() * RESIDUAL_DEPTHS,
        "every depth in every mode must be compared"
    );
    let mut worst_abs = 0.0_f64;
    for (index, outcome) in outcomes.iter().enumerate() {
        let depth = index % RESIDUAL_DEPTHS + 1;
        let mode = MODES[index / RESIDUAL_DEPTHS];
        assert_eq!(
            outcome.ours, outcome.oracle,
            "depth {depth} mode `{mode}`: our projection selects {} but the oracle's own logits \
             select {} — the head weight or its index is wrong, not merely imprecise",
            outcome.ours, outcome.oracle
        );
        assert_eq!(
            outcome.ours, outcome.expected_code,
            "depth {depth} mode `{mode}`: selected {} but the frame recorded c{depth} = {}",
            outcome.ours, outcome.expected_code
        );
        worst_abs = worst_abs.max(outcome.max_abs);
    }

    Receipt::new(ARGMAX_TEST, Outcome::Passed)
        .contract(CONTRACT)
        .seam("microdecoder.head_NN.output")
        .oracle_tier(OracleTier::CpuFp32Fallback)
        .detail(serde_json::json!({
            "comparisons": outcomes.len(),
            "claim": "argmax exact vs oracle logits and vs recorded codec_codes",
            "worst_max_abs_logit_diff": worst_abs,
            "bitwise_exact": false,
        }))
        .emit();
}

/// The unlicensed claim: bit-exactness. Recorded as XFAIL with its measured magnitude.
#[test]
fn contract_a_l2_microdecoder_head_projection_bitwise_cpu_fp32() {
    let outcomes = match run_all() {
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
    let worst_rel = outcomes
        .iter()
        .map(|outcome| outcome.max_rel)
        .fold(0.0_f64, f64::max);

    let still_diverging = xfail(BITWISE_TEST, CONTRACT, LEDGER, || {
        if worst == 0.0 {
            Ok(())
        } else {
            // sqrt(K) * eps * |logit| is what a K-term f32 dot product should cost; reporting the
            // ratio makes the claim checkable instead of asserted.
            let k = HIDDEN as f64;
            let expected = k.sqrt() * f64::from(f32::EPSILON) * LOGIT_SCALE;
            Err(format!(
                "head projection is not bit-exact against the CPU-tier oracle: max_abs {worst:.3e} \
                 over {} comparisons ({:.1} ULP of f32 relative to a logit scale of {LOGIT_SCALE}). \
                 That is {:.2}x the sqrt(K)*eps*|logit| = {expected:.3e} an f32 dot product over \
                 K={HIDDEN} should cost, so the magnitude is accumulation rounding, not wiring. \
                 Argmax is unaffected at every depth. Per-element max_rel reaches {worst_rel:.3e}, \
                 which is near-zero logits inflating a ratio, not a large error.",
                outcomes.len(),
                (worst / LOGIT_SCALE) / f64::from(f32::EPSILON),
                worst / expected
            ))
        }
    });

    assert!(
        still_diverging,
        "the head projection became bit-exact — retire the ledger entry and promote this to a \
         plain assertion"
    );
}
