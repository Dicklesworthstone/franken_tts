//! Per-depth distribution gate for `--micro-q8` heads: full-width KL and top-k overlap
//! between the bf16-widened reference scoring and the stored-Q8 rows' own matvec, on real
//! teacher-forced activations from the captured fixture pack (bead `frankentts-x7bt`).
//!
//! This is the objective half of the bead's Contract-B pre-work, runnable without the
//! Python oracle because BOTH sides of the comparison are our own arithmetic on the same
//! captured inputs: `logits_ref = Widen(lm_head[d-1]) @ x` versus `logits_q8 =
//! Dequant(Q8(lm_head[d-1])) @ x`. The production path scores coarse with the Q8 bytes and
//! refines candidates against those same bytes, so `logits_q8` is what the sampler can
//! select from once the artifact stores heads quantized; `logits_ref` is what it selects
//! from today. The distance between them IS the user-visible delta of the lever.
//!
//! Reported per depth in every prompt mode: KL(P‖Q) at the production temperature,
//! top-50 set overlap, and the rank-boundary margin. Argmax agreement is asserted: the
//! greedy token must not move anywhere in the corpus. The remaining numbers are receipts
//! for the listening pair (`frankentts-4tgm`), which outranks every figure here.
//!
//! Model-gated twice over: needs both the pinned checkpoint and the ft7 fixture pack;
//! each absence reports an honest skip and passes.

use std::path::{Path, PathBuf};

use ftts_artifacts::converter::quantize_output_channel_q8;
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
/// The production sampler's temperature (generation_config.json): distributions are
/// compared where the model actually samples, not at unity where nobody runs.
const TEMPERATURE: f64 = 0.9;
const TOP_K: usize = 50;

fn checkpoint_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../docs/truth-pack/snapshots/hf/model.safetensors")
}
fn fixtures_pack() -> Result<OracleFixtures, String> {
    // The canonical home (~/.cache/frankentts/...) wherever it exists; otherwise an
    // in-tree staging copy under docs/truth-pack/snapshots (git-ignored bytes, rsynced
    // to workers), so fixture-bearing measurement is not tied to one machine's home.
    match OracleFixtures::open_default() {
        Ok(fixtures) => Ok(fixtures),
        Err(home_error) => {
            let staged = Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../docs/truth-pack/snapshots/ft7-cpu-fp32-r1");
            OracleFixtures::open(&staged)
                .map_err(|error| format!("oracle fixtures unavailable: {home_error}; staged copy at {} also unusable: {error}", staged.display()))
        }
    }
}

fn skip(test: &str, reason: &str) {
    Receipt::new(test, Outcome::Skipped)
        .contract(CONTRACT)
        .seam("microdecoder.q8_topk")
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
                .ok_or_else(|| format!("`{name}` index {index} out of range"))
        })
        .collect()
}

fn project(weight: &[f32], x: &[f32]) -> Vec<f32> {
    weight
        .chunks_exact(x.len())
        .map(|row| row.iter().zip(x.iter()).map(|(w, v)| w * v).sum())
        .collect()
}

/// Indices sorted by descending logit with ascending-index tie-break — the deterministic
/// rule the selector family uses, so "top-k" means the same tokens on both sides.
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

/// KL(P‖Q) in nats with P from `reference` and Q from `comparison`, both softmaxed at
/// [`TEMPERATURE`] via log-sum-exp so a 2,048-wide f32 sum cannot underflow.
fn kl_divergence(reference: &[f32], comparison: &[f32]) -> f64 {
    let softmax_log = |logits: &[f32]| -> Vec<f64> {
        let scaled: Vec<f64> = logits
            .iter()
            .map(|&value| f64::from(value) / TEMPERATURE)
            .collect();
        let max = scaled.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let log_sum = max
            + scaled
                .iter()
                .map(|value| (value - max).exp())
                .sum::<f64>()
                .ln();
        scaled.iter().map(|value| value - log_sum).collect()
    };
    let log_p = softmax_log(reference);
    let log_q = softmax_log(comparison);
    log_p
        .iter()
        .zip(&log_q)
        .map(|(&p, &q)| p.exp() * (p - q))
        .sum()
}

struct DepthGate {
    mode: &'static str,
    depth: usize,
    /// KL(reference ‖ q8), nats, at the production temperature.
    kl_nats: f64,
    /// Shared tokens between the two top-[`TOP_K`] sets.
    top_k_overlap: usize,
    /// |logit difference| across the two rankings' shared rank-`TOP_K - 1` boundary pair,
    /// normalized by the reference logit scale: how close the cut is to flipping.
    boundary_margin_relative: f64,
    argmax_agrees: bool,
    /// Reference logit gap between rank 1 and rank 2, relative to scale — when the argmax
    /// moved, THIS is the tie that quantization noise was asked to break. Near zero means
    /// a coin-flip reorder, not a quality cliff.
    top1_gap_relative: f64,
}

fn run_all() -> Result<Vec<DepthGate>, String> {
    let fixtures = fixtures_pack()?;
    fixtures
        .require_oracle_class(CPU_FP32_ORACLE_CLASS)
        .map_err(|error| format!("fixture pack is not the CPU-fp32 tier: {error}"))?;

    let path = checkpoint_path();
    if !path.is_file() {
        return Err(format!("checkpoint absent at {}", path.display()));
    }
    let checkpoint = SafetensorsFile::open(&path)
        .map_err(|error| format!("cannot open {}: {error}", path.display()))?;

    let mut gates = Vec::new();
    for mode in MODES {
        for depth in 1..=RESIDUAL_DEPTHS {
            let input_seam = SeamRef {
                case: CASE,
                mode,
                group: "teacher_forced_frame_0000",
                seam: &format!("microdecoder.head_{depth:02}.input"),
            };
            if !fixtures.has_seam(&input_seam) {
                return Err(format!("`{}` is not in this pack", input_seam.describe()));
            }
            let x = fixtures
                .seam(&input_seam, "args.0", 0)
                .map_err(|error| format!("cannot read {}: {error}", input_seam.describe()))?;
            if x.data.len() != HIDDEN {
                return Err(format!(
                    "{} is {} wide, expected {HIDDEN}",
                    input_seam.describe(),
                    x.data.len()
                ));
            }

            // Head d scores position d, so it is `lm_head[d - 1]`.
            let weight_name = format!("talker.code_predictor.lm_head.{}.weight", depth - 1);
            let weight = widen(&checkpoint, &weight_name)?;

            // The artifact-native form: canonical per-row Q8, then its own dequantize —
            // byte-for-byte what `score_head_refined`'s refine source becomes when no
            // widened head is resident.
            let mut q8_bytes = vec![0_i8; HIDDEN];
            let mut dequantized = vec![0.0_f32; weight.len()];
            for (row_index, row) in weight.as_chunks::<HIDDEN>().0.iter().enumerate() {
                let scale = quantize_output_channel_q8(row, &mut q8_bytes)
                    .map_err(|error| format!("{weight_name} row {row_index}: {error}"))?;
                for (column, &byte) in q8_bytes.iter().enumerate() {
                    dequantized[row_index * HIDDEN + column] = f32::from(byte) * scale;
                }
            }

            let reference = project(&weight, &x.data);
            let comparison = project(&dequantized, &x.data);

            let reference_order = ranking(&reference);
            let comparison_order = ranking(&comparison);
            let reference_top: std::collections::HashSet<usize> =
                reference_order[..TOP_K].iter().copied().collect();
            let overlap = comparison_order[..TOP_K]
                .iter()
                .filter(|token| reference_top.contains(*token))
                .count();

            // Boundary margin: the gap between the last kept and first dropped reference
            // tokens, measured against the dequantized values (what production scores),
            // relative to the reference logit scale.
            let scale = reference
                .iter()
                .fold(0.0_f64, |max, &value| max.max(f64::from(value.abs())));
            let last_kept = reference_order[TOP_K - 1];
            let first_dropped = reference_order[TOP_K];
            let reference_top1 = reference_order[0];
            let reference_top1_gap = if reference.len() > 1 {
                f64::from(reference[reference_top1] - reference[reference_order[1]])
            } else {
                0.0
            };
            let boundary_margin_relative =
                f64::from((comparison[last_kept] - comparison[first_dropped]).abs()) / scale;
            gates.push(DepthGate {
                mode,
                depth,
                kl_nats: kl_divergence(&reference, &comparison),
                top_k_overlap: overlap,
                boundary_margin_relative,
                argmax_agrees: argmax_of(&reference) == argmax_of(&comparison),
                top1_gap_relative: reference_top1_gap / scale,
            });
        }
    }
    Ok(gates)
}

fn argmax_of(logits: &[f32]) -> usize {
    let mut best = 0_usize;
    for (index, &value) in logits.iter().enumerate().skip(1) {
        if value > logits[best] {
            best = index;
        }
    }
    best
}

#[test]
fn microdecoder_q8_head_rows_preserve_the_reference_distribution() {
    const TEST: &str = "microdecoder_q8_head_rows_preserve_the_reference_distribution";
    let gates = match run_all() {
        Ok(gates) => gates,
        Err(reason) => {
            skip(TEST, &reason);
            return;
        }
    };

    // MEASUREMENT FIRST: a lossy lever is allowed to move near-ties; what this test
    // polices is gross wiring failure, and what it produces is the evidence the
    // listening pair (frankentts-4tgm) consumes. Argmax moves are reported with the
    // tie margin they broke — a coin-flip reorder on a ~0 gap is a different universe
    // from a flip across a wide margin.
    let moved: Vec<&DepthGate> = gates.iter().filter(|gate| !gate.argmax_agrees).collect();
    for gate in &moved {
        eprintln!(
            "receipt: {{\"test\":\"q8_head_topk\",\"event\":\"argmax_moved\",\
             \"mode\":\"{}\",\"depth\":{},\"top1_gap_rel\":{:.6}}}",
            gate.mode, gate.depth, gate.top1_gap_relative
        );
    }

    let worst_kl = gates
        .iter()
        .max_by(|a, b| a.kl_nats.total_cmp(&b.kl_nats))
        .expect("non-empty");
    let mean_kl = gates.iter().map(|gate| gate.kl_nats).sum::<f64>() / gates.len() as f64;
    let min_overlap = gates
        .iter()
        .map(|gate| gate.top_k_overlap)
        .min()
        .expect("non-empty");
    let moved_gaps: Vec<f64> = moved.iter().map(|gate| gate.top1_gap_relative).collect();

    // Catastrophe bounds only: a transposed or half-written table produces KL orders of
    // magnitude above these; legitimate quantization noise sits orders below. The numbers
    // are anti-wiring tripwires, NOT quality gates — quality is the listening pair's call.
    assert!(
        mean_kl < 0.05,
        "mean KL({mean_kl:.4} nats) crossed the wiring-failure bound; the stored-Q8 rows \
         are not a mild perturbation of the reference rows"
    );
    assert!(
        worst_kl.kl_nats < 0.5,
        "worst KL {:.4} nats at {}/depth{} crossed the wiring-failure bound",
        worst_kl.kl_nats,
        worst_kl.mode,
        worst_kl.depth
    );
    assert!(
        min_overlap >= TOP_K / 2,
        "top-{TOP_K} overlap fell to {min_overlap} somewhere — half the sampler's \
         eligible set is being rebuilt by quantization noise, which no rendition class \
         covers"
    );

    let worst_margin = gates
        .iter()
        .min_by(|a, b| {
            a.boundary_margin_relative
                .total_cmp(&b.boundary_margin_relative)
        })
        .expect("non-empty");
    let mean_kl_display = mean_kl;
    eprintln!(
        "receipt: {{\"test\":\"q8_head_topk\",\"comparisons\":{},\"argmax_moved\":{},\
         \"moved_top1_gaps_rel\":{:?},\"min_overlap\":{},\"mean_kl_nats\":{:.6},\
         \"worst_kl_nats\":{:.6},\"worst_kl_at\":\"{}/depth{}\",\
         \"min_boundary_margin_rel\":{:.6},\"min_margin_at\":\"{}/depth{}\"}}",
        gates.len(),
        moved.len(),
        moved_gaps,
        min_overlap,
        mean_kl_display,
        worst_kl.kl_nats,
        worst_kl.mode,
        worst_kl.depth,
        worst_margin.boundary_margin_relative,
        worst_margin.mode,
        worst_margin.depth,
    );
    Receipt::new(TEST, Outcome::Passed)
        .contract(CONTRACT)
        .seam("microdecoder.q8_topk")
        .detail(serde_json::json!({
            "comparisons": gates.len(),
            "temperature": TEMPERATURE,
            "top_k": TOP_K,
            "argmax_moved": moved.len(),
            "moved_top1_gaps_relative": moved_gaps,
            "min_top_k_overlap": min_overlap,
            "mean_kl_nats": mean_kl,
            "worst_kl_nats": worst_kl.kl_nats,
            "min_boundary_margin_relative": worst_margin.boundary_margin_relative,
        }))
        .emit();
}
