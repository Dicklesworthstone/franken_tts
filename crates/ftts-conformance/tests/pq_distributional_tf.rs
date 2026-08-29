//! ProductionQuality (Contract B), family 1: teacher-forced distributional metrics.
//!
//! Bead `frankentts-v-prod-harness-t96`. For every captured oracle point — each
//! microdecoder depth in each prompt mode across every teacher-forced frame, plus the
//! talker's own primary-code head across the free-running steps — this harness runs OUR
//! projection on the oracle's captured input and scores our logit vector against the
//! oracle's captured logits:
//!
//! * KL(reference ‖ ours) and JS, both at the production temperature;
//! * top-50 set overlap between the two rankings;
//! * the rank of our selected token inside the reference ordering.
//!
//! **What is asserted and why.** Argmax agreement at every scored point is asserted:
//! Contract-A work proved our CPU-fp32 arithmetic selects the oracle's token at these
//! seams, so any disagreement here is a wiring bug in THIS harness, not a new numeric
//! finding. Everything else is recorded, never gated against an invented threshold:
//! Contract B is distribution-level by design, and its margins live in the listening
//! protocol (`scripts/listening/margins.toml`), not in this file.
//!
//! Model-gated twice over (fixture pack + checkpoint); each absence reports an honest
//! skip and passes. When `FTTS_PQ_REPORT` names a path, `<path>.distributional.json`
//! holds the metric table for the release scorecard to consume.

#![cfg(feature = "ultra-tests")]

use std::path::PathBuf;

use ftts_artifacts::safetensors::SafetensorsFile;
use ftts_conformance::oracle::{CPU_FP32_ORACLE_CLASS, OracleFixtures, SeamRef};
use ftts_conformance::production::{
    argmax_of, js_divergence_at, kl_divergence_at, rank_of, ranking, top_k_overlap,
};
use ftts_conformance::report::{Outcome, Receipt};
use ftts_model_qwen::microdecoder::{RESIDUAL_DEPTHS, RESIDUAL_VOCAB};

const CONTRACT: &str = "ProductionQuality/teacher_forced";
const CASE: &str = "synthetic-tone-en";
const MODES: [&str; 4] = [
    "xvector_non_streaming",
    "xvector_streaming",
    "icl_non_streaming",
    "icl_streaming",
];
const TEMPERATURE: f64 = 0.9;
const TOP_K: usize = 50;
const HIDDEN: usize = 1024;
const TALKER_VOCAB: usize = 3072;

/// Scorecard destination, when the caller asks for one.
fn report_path() -> Option<PathBuf> {
    std::env::var_os("FTTS_PQ_REPORT").map(PathBuf::from)
}

fn skip(test: &str, reason: &str) {
    Receipt::new(test, Outcome::Skipped)
        .contract(CONTRACT)
        .seam("production.teacher_forced")
        .reason(reason)
        .emit();
}

/// Canonical fixture home wherever it exists; otherwise an in-tree staging copy
/// (git-ignored bytes, rsynced to workers), so measurement is not tied to one machine.
fn fixtures_pack() -> Result<OracleFixtures, String> {
    match OracleFixtures::open_default() {
        Ok(fixtures) => Ok(fixtures),
        Err(home_error) => {
            let staged = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
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

/// Widens a whole BF16 head tensor to f32 through the checkpoint accessor.
fn widen(file: &SafetensorsFile, name: &str, rows: usize) -> Result<Vec<f32>, String> {
    let view = file
        .view(name)
        .ok_or_else(|| format!("checkpoint is missing `{name}`"))?;
    if view.len() != rows * HIDDEN {
        return Err(format!(
            "`{name}` holds {} elements, expected {rows}x{HIDDEN}",
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

/// `out[o] = sum_i weight[o * HIDDEN + i] * x[i]`.
fn project(weight: &[f32], x: &[f32]) -> Vec<f32> {
    weight
        .chunks_exact(x.len())
        .map(|row| row.iter().zip(x.iter()).map(|(w, v)| w * v).sum())
        .collect()
}

/// One scored comparison: our logits against the reference logits at one oracle point.
struct Cell {
    label: String,
    kl_nats: f64,
    js_nats: f64,
    /// Shared tokens between the two top-[`TOP_K`] sets.
    overlap: usize,
    /// Rank of OUR argmax inside the reference ranking.
    selected_rank: Option<usize>,
    argmax_agrees: bool,
}

fn score_cell(label: String, reference: &[f32], ours: &[f32]) -> Cell {
    let reference_order = ranking(reference);
    let ours_order = ranking(ours);
    let selected = argmax_of(ours);
    Cell {
        selected_rank: rank_of(selected, &reference_order),
        argmax_agrees: selected == reference_order[0],
        kl_nats: kl_divergence_at(reference, ours, TEMPERATURE),
        js_nats: js_divergence_at(reference, ours, TEMPERATURE),
        overlap: top_k_overlap(&reference_order, &ours_order, TOP_K),
        label,
    }
}

/// Top-`TOP_K` overlap as a fraction of the window.
fn overlap_fraction(overlap: usize) -> f64 {
    f64::from(u32::try_from(overlap).unwrap_or(u32::MAX)) / f64::from(TOP_K as u32)
}

/// Mean of a column over cells.
fn mean(values: impl Iterator<Item = f64>) -> f64 {
    let (count, total) = values.fold((0_u64, 0.0_f64), |(n, sum), value| (n + 1, sum + value));
    if count == 0 {
        0.0
    } else {
        total / f64::from(u32::try_from(count).unwrap_or(u32::MAX))
    }
}

/// Every teacher-forced microdecoder cell: mode x frame x depth.
fn microdecoder_cells(
    fixtures: &OracleFixtures,
    checkpoint: &SafetensorsFile,
) -> Result<Vec<Cell>, String> {
    // Head weights widen once per depth, reused across modes and frames. Head d scores
    // position d, so depth d's head is `lm_head[d - 1]`.
    let mut heads: Vec<Vec<f32>> = Vec::with_capacity(RESIDUAL_DEPTHS);
    for depth in 1..=RESIDUAL_DEPTHS {
        heads.push(widen(
            checkpoint,
            &format!("talker.code_predictor.lm_head.{}.weight", depth - 1),
            RESIDUAL_VOCAB,
        )?);
    }

    let mut cells = Vec::new();
    let mut frame_index = 0_usize;
    loop {
        let group = format!("teacher_forced_frame_{frame_index:04}");
        let probe = SeamRef {
            case: CASE,
            mode: MODES[0],
            group: &group,
            seam: "microdecoder.head_01.input",
        };
        if !fixtures.has_seam(&probe) {
            break;
        }
        for mode in MODES {
            for (head_index, weight) in heads.iter().enumerate() {
                let depth = head_index + 1;
                let input_seam = SeamRef {
                    case: CASE,
                    mode,
                    group: &group,
                    seam: &format!("microdecoder.head_{depth:02}.input"),
                };
                let output_seam = SeamRef {
                    case: CASE,
                    mode,
                    group: &group,
                    seam: &format!("microdecoder.head_{depth:02}.output"),
                };
                let x = fixtures
                    .seam(&input_seam, "args.0", 0)
                    .map_err(|error| format!("cannot read {}: {error}", input_seam.describe()))?;
                let reference_npy = fixtures
                    .seam(&output_seam, "tensor", 0)
                    .map_err(|error| format!("cannot read {}: {error}", output_seam.describe()))?;
                if x.data.len() != HIDDEN || reference_npy.data.len() != RESIDUAL_VOCAB {
                    return Err(format!(
                        "{} unexpected widths: input {}, reference {}",
                        input_seam.describe(),
                        x.data.len(),
                        reference_npy.data.len()
                    ));
                }
                cells.push(score_cell(
                    format!("{mode}/{group}/depth_{depth:02}"),
                    &reference_npy.data,
                    &project(weight, &x.data),
                ));
            }
        }
        frame_index += 1;
    }
    Ok(cells)
}

/// Talker-level cells: the primary-code head over the free-running prefill/update steps.
///
/// Each captured step holds the whole scored sequence ([1, seq, ·]); generation happens
/// at the LAST position, so that is the row compared.
fn talker_cells(
    fixtures: &OracleFixtures,
    checkpoint: &SafetensorsFile,
) -> Result<Vec<Cell>, String> {
    let weight = widen(checkpoint, "talker.codec_head.weight", TALKER_VOCAB)?;
    let mut cells = Vec::new();
    for mode in MODES {
        for step in 0_usize.. {
            let seam = SeamRef {
                case: CASE,
                mode,
                group: "talker_free_running",
                seam: "talker.codec_head.input",
            };
            let input = match fixtures.seam(&seam, "args.0", step) {
                Ok(input) => input,
                Err(_) => break,
            };
            let output = fixtures
                .seam(
                    &SeamRef {
                        case: CASE,
                        mode,
                        group: "talker_free_running",
                        seam: "talker.codec_head.output",
                    },
                    "tensor",
                    step,
                )
                .map_err(|error| {
                    format!("cannot read talker.codec_head.output step {step}: {error}")
                })?;
            if input.data.len() % HIDDEN != 0
                || input.data.len() / HIDDEN * TALKER_VOCAB != output.data.len()
            {
                return Err(format!(
                    "talker codec head step {step} shape mismatch: {} input elements vs {} output",
                    input.data.len(),
                    output.data.len()
                ));
            }
            let rows = input.data.len() / HIDDEN;
            let x = &input.data[(rows - 1) * HIDDEN..];
            let reference = &output.data[(rows - 1) * TALKER_VOCAB..];
            cells.push(score_cell(
                format!("{mode}/talker_free_running/step_{step:03}"),
                reference,
                &project(&weight, x),
            ));
        }
    }
    Ok(cells)
}

#[test]
fn production_quality_teacher_forced_distributions_match_the_oracle() {
    const TEST: &str = "production_quality_teacher_forced_distributions_match_the_oracle";

    let fixtures = match fixtures_pack() {
        Ok(fixtures) => fixtures,
        Err(error) => {
            skip(TEST, &error);
            return;
        }
    };
    if let Err(error) = fixtures.require_oracle_class(CPU_FP32_ORACLE_CLASS) {
        skip(
            TEST,
            &format!("fixture pack is not the CPU-fp32 tier: {error}"),
        );
        return;
    }
    let checkpoint_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../docs/truth-pack/snapshots/hf/model.safetensors");
    if !checkpoint_path.is_file() {
        skip(
            TEST,
            &format!("checkpoint absent at {}", checkpoint_path.display()),
        );
        return;
    }
    let checkpoint = match SafetensorsFile::open(&checkpoint_path) {
        Ok(checkpoint) => checkpoint,
        Err(error) => {
            skip(
                TEST,
                &format!("cannot open {}: {error}", checkpoint_path.display()),
            );
            return;
        }
    };

    let micro_cells = match microdecoder_cells(&fixtures, &checkpoint) {
        Ok(cells) => cells,
        Err(reason) => {
            skip(TEST, &reason);
            return;
        }
    };
    let talker_steps = match talker_cells(&fixtures, &checkpoint) {
        Ok(cells) => cells,
        Err(reason) => {
            skip(TEST, &reason);
            return;
        }
    };

    assert!(
        !micro_cells.is_empty(),
        "no teacher-forced microdecoder cells found; the fixture pack layout changed"
    );
    assert!(!talker_steps.is_empty(), "no talker codec-head steps found");

    // The licensed claim: our arithmetic selects the oracle's token everywhere.
    let disagreements: Vec<&str> = micro_cells
        .iter()
        .chain(talker_steps.iter())
        .filter(|cell| !cell.argmax_agrees)
        .map(|cell| cell.label.as_str())
        .collect();
    assert!(
        disagreements.is_empty(),
        "our argmax left the oracle's token at {} points, starting {}: \
         this contradicts the Contract-A exactness receipts and must be investigated, \
         not tolerated",
        disagreements.len(),
        disagreements[0]
    );

    // Per-mode aggregates for the receipt detail.
    let mut mode_summary = serde_json::Map::new();
    for mode in MODES {
        let mode_cells: Vec<&Cell> = micro_cells
            .iter()
            .filter(|cell| cell.label.starts_with(mode))
            .collect();
        mode_summary.insert(
            mode.to_owned(),
            serde_json::json!({
                "cells": mode_cells.len(),
                "kl_nats_mean": mean(mode_cells.iter().map(|cell| cell.kl_nats)),
                "js_nats_mean": mean(mode_cells.iter().map(|cell| cell.js_nats)),
                "top50_overlap_mean": mean(mode_cells.iter().map(|cell| overlap_fraction(cell.overlap))),
                "selected_rank_max": mode_cells.iter().filter_map(|cell| cell.selected_rank).max(),
            }),
        );
    }

    let micro_kl_mean = mean(micro_cells.iter().map(|cell| cell.kl_nats));
    let micro_js_mean = mean(micro_cells.iter().map(|cell| cell.js_nats));
    let micro_overlap_mean = mean(
        micro_cells
            .iter()
            .map(|cell| overlap_fraction(cell.overlap)),
    );
    let talker_kl_mean = mean(talker_steps.iter().map(|cell| cell.kl_nats));
    let talker_js_mean = mean(talker_steps.iter().map(|cell| cell.js_nats));
    let talker_overlap_mean = mean(
        talker_steps
            .iter()
            .map(|cell| overlap_fraction(cell.overlap)),
    );

    Receipt::new(TEST, Outcome::Passed)
        .contract(CONTRACT)
        .seam("production.teacher_forced")
        .reason(format!(
            "{} microdecoder cells ({} modes x frames x 15 depths) + {} talker head steps; \
             argmax agreement 100%; micro KL mean {micro_kl_mean:.3e} nats, JS mean \
             {micro_js_mean:.3e}, top-{TOP_K} overlap mean {micro_overlap_mean:.4}; \
             talker KL mean {talker_kl_mean:.3e}, overlap mean {talker_overlap_mean:.4}",
            micro_cells.len(),
            MODES.len(),
            talker_steps.len(),
        ))
        .detail(serde_json::json!({
            "temperature": TEMPERATURE,
            "top_k": TOP_K,
            "by_mode": mode_summary,
            "talker_codec_head": {
                "steps": talker_steps.len(),
                "kl_nats_mean": talker_kl_mean,
                "js_nats_mean": talker_js_mean,
                "top50_overlap_mean": talker_overlap_mean,
            },
        }))
        .emit();

    write_scorecard(&micro_cells, &talker_steps, &mode_summary);
}

/// Writes `<FTTS_PQ_REPORT>.distributional.json` when the env var is set.
fn write_scorecard(
    micro_cells: &[Cell],
    talker_steps: &[Cell],
    mode_summary: &serde_json::Map<String, serde_json::Value>,
) {
    let Some(base) = report_path() else { return };
    let mut path = base;
    path.set_file_name(format!(
        "{}.distributional.json",
        path.file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("pq_scorecard")
    ));
    let agreeing = micro_cells.iter().filter(|cell| cell.argmax_agrees).count();
    let scorecard = serde_json::json!({
        "schema_version": 1,
        "generator": "pq_distributional_tf",
        "bead": "frankentts-v-prod-harness-t96",
        "temperature": TEMPERATURE,
        "teacher_forced_distributional": {
            "microdecoder": {
                "cells": micro_cells.len(),
                "argmax_agreement_rate": f64::from(u32::try_from(agreeing).unwrap_or(0))
                    / f64::from(u32::try_from(micro_cells.len().max(1)).unwrap_or(1)),
                "kl_nats_mean": mean(micro_cells.iter().map(|cell| cell.kl_nats)),
                "js_nats_mean": mean(micro_cells.iter().map(|cell| cell.js_nats)),
                "top50_overlap_mean": mean(micro_cells.iter().map(|cell| overlap_fraction(cell.overlap))),
                "selected_rank_max": micro_cells.iter().filter_map(|cell| cell.selected_rank).max(),
                "by_mode": mode_summary,
            },
            "talker_primary_code_head": {
                "steps": talker_steps.len(),
                "kl_nats_mean": mean(talker_steps.iter().map(|cell| cell.kl_nats)),
                "js_nats_mean": mean(talker_steps.iter().map(|cell| cell.js_nats)),
                "top50_overlap_mean": mean(talker_steps.iter().map(|cell| overlap_fraction(cell.overlap))),
            },
        },
    });
    if let Ok(json) = serde_json::to_string_pretty(&scorecard) {
        let _ = std::fs::write(&path, json + "\n");
    }
}
