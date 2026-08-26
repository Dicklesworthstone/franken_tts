//! AWQ + GPTQ validation on captured activations (bead `frankentts-ukk6`,
//! tranche-4 gate precursor): the calibrated-int4 core in
//! `ftts_artifacts::awq` must beat plain round-to-nearest Q4 on REAL
//! teacher-forced microdecoder activations, and the greedy token must survive.
//!
//! For every depth × mode: load the captured input vector `x` and head `W`,
//! compute reference logits `W·x`, then compare two 4-bit variants —
//! per-row RTN, and AWQ-scaled GPTQ (scales grid-searched over all four
//! modes' calibration vectors for that depth). Receipts carry relative error
//! and top-k overlap for the listening pair (`frankentts-4tgm`).
//!
//! Model-gated twice: needs the pinned checkpoint and the ft7 fixture pack;
//! each absence reports an honest skip and passes.

use std::path::{Path, PathBuf};

use ftts_artifacts::awq::{awq_best_alpha, gptq_inverse_hessian, gptq_round_matrix};
use ftts_artifacts::safetensors::SafetensorsFile;
use ftts_conformance::oracle::{CPU_FP32_ORACLE_CLASS, OracleFixtures, SeamRef};
use ftts_conformance::report::{Outcome, Receipt};
use ftts_model_qwen::microdecoder::{RESIDUAL_DEPTHS, RESIDUAL_VOCAB};

const CONTRACT: &str = "ConformanceExact/L2";
const CASE: &str = "synthetic-tone-en";
const MODES: [&str; 4] = [
    "xvector_non_streaming",
    "xvector_streaming",
    "icl_non_streaming",
    "icl_streaming",
];
const HIDDEN: usize = 1024;

fn checkpoint_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../docs/truth-pack/snapshots/hf/model.safetensors")
}

fn fixtures_pack() -> Result<OracleFixtures, String> {
    match OracleFixtures::open_default() {
        Ok(fixtures) => Ok(fixtures),
        Err(home_error) => {
            let staged = Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../docs/truth-pack/snapshots/ft7-cpu-fp32-r1");
            OracleFixtures::open(&staged).map_err(|error| {
                format!(
                    "oracle fixtures unavailable: {home_error}; staged copy at {} also unusable: {error}",
                    staged.display()
                )
            })
        }
    }
}

fn skip(test: &str, reason: &str) {
    Receipt::new(test, Outcome::Skipped)
        .contract(CONTRACT)
        .seam("microdecoder.q4_awq")
        .reason(reason)
        .emit();
}

fn widen(file: &SafetensorsFile, name: &str) -> Result<Vec<f32>, String> {
    let view = file
        .view(name)
        .ok_or_else(|| format!("checkpoint is missing `{name}`"))?;
    if view.len() != RESIDUAL_VOCAB * HIDDEN {
        return Err(format!(
            "`{name}` holds {} elements, expected {RESIDUAL_VOCAB}x{HIDDEN}",
            view.len()
        ));
    }
    (0..view.len())
        .map(|index| {
            view.get_f32(index)
                .ok_or_else(|| format!("`{name}` element {index} unreadable"))
        })
        .collect()
}

fn project(weight: &[f32], x: &[f32]) -> Vec<f32> {
    weight
        .chunks_exact(x.len())
        .map(|row| row.iter().zip(x.iter()).map(|(w, v)| w * v).sum())
        .collect()
}

fn ranking(logits: &[f32]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..logits.len()).collect();
    order.sort_by(|&a, &b| {
        logits[b]
            .partial_cmp(&logits[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    order
}

/// Per-row symmetric 4-bit round-to-nearest quantize/dequantize of a
/// row-major `[n, k]` matrix — the baseline the calibrated path must beat.
fn rtn_q4_dequantized(weight_row_major: &[f32], n: usize, k: usize) -> Vec<f32> {
    let levels = 8.0_f32;
    let mut out = vec![0.0_f32; n * k];
    for row in 0..n {
        let src = &weight_row_major[row * k..(row + 1) * k];
        let max_abs = src.iter().fold(0.0_f32, |acc, &value| acc.max(value.abs()));
        let scale = if max_abs > 0.0 { max_abs / levels } else { 1.0 };
        for (j, &w) in src.iter().enumerate() {
            out[row * k + j] = (w / scale).round().clamp(-levels, levels) * scale;
        }
    }
    out
}

/// The calibrated variant: AWQ scales grid-searched over ALL modes' vectors at
/// this depth, weights rescaled, GPTQ-compensated against the scaled
/// calibration Hessian, dequantized; effective logits use the scaled target
/// activation (the runtime fold of the scales into the preceding op).
fn awq_gptq_logits(
    weight_row_major: &[f32],
    n: usize,
    k: usize,
    calib_x: &[Vec<f32>],
    target_x: &[f32],
) -> Vec<f32> {
    let (_alpha, scales) = awq_best_alpha(weight_row_major, n, k, calib_x, 4, 0.1);

    let rescale_weight = |w: f32, j: usize| w / scales[j];
    let scale_activation = |v: f32, j: usize| v * scales[j];

    let mut rescaled_w = vec![0.0_f32; n * k];
    for row in 0..n {
        for j in 0..k {
            rescaled_w[row * k + j] = rescale_weight(weight_row_major[row * k + j], j);
        }
    }

    let scaled = |x: &[f32]| -> Vec<f32> {
        x.iter()
            .enumerate()
            .map(|(j, &value)| scale_activation(value, j))
            .collect()
    };
    let scaled_calib: Vec<Vec<f32>> = calib_x.iter().map(|x| scaled(x)).collect();
    let scaled_target = scaled(target_x);

    let mut hessian = vec![0.0_f64; k * k];
    for x in &scaled_calib {
        for i in 0..k {
            for j in 0..k {
                hessian[i * k + j] += f64::from(x[i]) * f64::from(x[j]);
            }
        }
    }
    let inverse = gptq_inverse_hessian(&hessian, k, 0.01)
        .expect("calibration Hessian invertible after damping");
    let rounded = gptq_round_matrix(&rescaled_w, n, k, &inverse, 4);
    project(&rounded, &scaled_target)
}

#[test]
fn q4_awq_beats_rtn_on_captured_microdecoder_activations() {
    const TEST: &str = "q4_awq_beats_rtn_on_captured_microdecoder_activations";
    let fixtures = match fixtures_pack() {
        Ok(fixtures) => fixtures,
        Err(reason) => {
            skip(TEST, &reason);
            return;
        }
    };
    if fixtures.oracle_class() != CPU_FP32_ORACLE_CLASS {
        skip(TEST, "fixture pack is not the CPU-fp32 tier");
        return;
    }
    let path = checkpoint_path();
    if !path.is_file() {
        skip(TEST, &format!("checkpoint absent at {}", path.display()));
        return;
    }
    let checkpoint = match SafetensorsFile::open(&path) {
        Ok(file) => file,
        Err(error) => {
            skip(TEST, &format!("cannot open checkpoint: {error}"));
            return;
        }
    };

    let mut worst_ratio = 0.0_f64;
    let mut total_greedy_moves_rtn = 0_usize;
    let mut total_greedy_moves_awq = 0_usize;
    let mut cases = 0_usize;

    for mode in MODES {
        // Calibration for each depth pools ALL modes' vectors: the pack is
        // frozen, so evaluating on a member of its own calibration pool is by
        // construction here (leave-one-out belongs to a bigger corpus).
        for depth in 1..=RESIDUAL_DEPTHS {
            let mut calib: Vec<Vec<f32>> = Vec::with_capacity(MODES.len());
            let mut target: Option<Vec<f32>> = None;
            for m in MODES {
                let seam = SeamRef {
                    case: CASE,
                    mode: m,
                    group: "teacher_forced_frame_0000",
                    seam: &format!("microdecoder.head_{depth:02}.input"),
                };
                if !fixtures.has_seam(&seam) {
                    skip(TEST, &format!("{} not in pack", seam.describe()));
                    return;
                }
                let Ok(x) = fixtures.seam(&seam, "args.0", 0) else {
                    skip(TEST, &format!("cannot read {}", seam.describe()));
                    return;
                };
                if m == mode {
                    target = Some(x.data.clone());
                }
                calib.push(x.data);
            }
            let Some(target_x) = target else {
                skip(TEST, &format!("no target vector for mode {mode}"));
                return;
            };

            let weight_name = format!("talker.code_predictor.lm_head.{}.weight", depth - 1);
            let weight = match widen(&checkpoint, &weight_name) {
                Ok(weight) => weight,
                Err(reason) => {
                    skip(TEST, &reason);
                    return;
                }
            };

            let reference = project(&weight, &target_x);
            let rtn_logits = project(
                &rtn_q4_dequantized(&weight, RESIDUAL_VOCAB, HIDDEN),
                &target_x,
            );
            let awq_logits = awq_gptq_logits(&weight, RESIDUAL_VOCAB, HIDDEN, &calib, &target_x);

            let rel_error = |logits: &[f32]| -> f64 {
                let num: f64 = reference
                    .iter()
                    .zip(logits)
                    .map(|(r, q)| (f64::from(*r) - f64::from(*q)).powi(2))
                    .sum();
                let den: f64 = reference.iter().map(|&r| f64::from(r) * f64::from(r)).sum();
                if den == 0.0 { 0.0 } else { num / den }
            };
            let ratio = rel_error(&awq_logits) / rel_error(&rtn_logits).max(1e-30);
            worst_ratio = worst_ratio.max(ratio);
            cases += 1;

            let ref_order = ranking(&reference);
            let greedy_ref = ref_order[0];
            for (label, logits) in [("rtn", &rtn_logits), ("awq", &awq_logits)] {
                if ranking(logits)[0] != greedy_ref {
                    match label {
                        "rtn" => total_greedy_moves_rtn += 1,
                        _ => total_greedy_moves_awq += 1,
                    }
                }
            }
            eprintln!(
                "{{\"receipt\":\"q4_awq\",\"depth\":{depth},\"mode\":\"{mode}\",\
\"err_ratio_vs_rtn\":{ratio:.4}}}"
            );
        }
    }

    assert!(
        cases >= RESIDUAL_DEPTHS,
        "expected at least one case per depth, ran {cases}"
    );
    assert!(
        worst_ratio <= 1.10,
        "AWQ+GPTQ relative-error ratio vs RTN exceeded 1.10 on some depth×mode: {worst_ratio}"
    );
    assert_eq!(
        total_greedy_moves_rtn, 0,
        "RTN baseline itself moved the greedy token; fixture or threshold drift"
    );
    assert_eq!(
        total_greedy_moves_awq, 0,
        "AWQ+GPTQ moved the greedy token on the frozen corpus"
    );
    Receipt::new(TEST, Outcome::Passed)
        .contract(CONTRACT)
        .seam("microdecoder.q4_awq")
        .detail(serde_json::json!({"cases": cases, "worst_err_ratio": worst_ratio}))
        .emit();
}
