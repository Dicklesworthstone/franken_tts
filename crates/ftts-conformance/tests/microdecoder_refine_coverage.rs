//! Full-corpus receipt for the head-refine window claim: on every captured teacher-forced
//! activation in the fixture pack, the int8 coarse pass's top-[`HEAD_REFINE_CANDIDATES`]
//! set contains the sampler's entire top-k of the exact-f32 row (bead `frankentts-0ged`).
//!
//! The production path scores 2,048-way logits coarsely in int8 and recomputes exactly the
//! top [`HEAD_REFINE_CANDIDATES`] in f32; anything outside that window is `-inf`. The
//! inline justification ("96 leaves a wide margin around the cut") was until now an
//! unmeasured assertion. This test measures it across all four prompt modes × fifteen
//! depths using the production coarse kernel itself (`quant_linear`, W8A8, dynamic
//! per-row activation quantization) against the widened bf16 reference.
//!
//! Asserted: coverage is complete (all 50 sampled-eligible tokens inside the 96-window)
//! everywhere, and the worst observed coarse rank is reported so the window has a measured
//! margin, not a vibe. Model-gated: skips honestly without checkpoint or fixtures.

#![cfg(feature = "ultra-tests")]

use std::path::{Path, PathBuf};

use ftts_artifacts::safetensors::SafetensorsFile;
use ftts_conformance::oracle::{CPU_FP32_ORACLE_CLASS, OracleFixtures, SeamRef};
use ftts_conformance::report::{Outcome, Receipt};
use ftts_kernels::int8::{Int8Tier, KernelPlanV0, QuantLinearMode, QuantizedMatrix, quant_linear};
use ftts_model_qwen::microdecoder::{HEAD_REFINE_CANDIDATES, RESIDUAL_DEPTHS, RESIDUAL_VOCAB};
use ftts_model_qwen::sampler::TOP_K;

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

fn skip(test: &str, reason: &str) {
    Receipt::new(test, Outcome::Skipped)
        .contract(CONTRACT)
        .seam("microdecoder.refine_coverage")
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

/// Descending-value order with ascending-index ties — the same deterministic rule the
/// selector family and `score_head_refined`'s own selection loop implement.
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

struct Coverage {
    mode: &'static str,
    depth: usize,
    /// Every f32-top-k token appeared in the coarse window when true.
    covered: bool,
    /// Highest coarse rank any f32-top-k token reached (1-based; the window is
    /// [`HEAD_REFINE_CANDIDATES`] deep).
    worst_rank_in_window: usize,
}

fn fixtures_pack() -> Result<OracleFixtures, String> {
    // Canonical home first, then the git-ignored in-tree staging copy (rsynced to
    // workers) — same resolution order as microdecoder_q8_topk.
    match OracleFixtures::open_default() {
        Ok(fixtures) => Ok(fixtures),
        Err(home_error) => {
            let staged = Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../docs/truth-pack/snapshots/ft7-cpu-fp32-r1");
            OracleFixtures::open(&staged).map_err(|error| {
                format!(
                    "oracle fixtures unavailable: {home_error}; staged copy at {} also \
                     unusable: {error}",
                    staged.display()
                )
            })
        }
    }
}

fn run_all() -> Result<Vec<Coverage>, String> {
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

    let mode_w8a8 = QuantLinearMode::W8A8(KernelPlanV0::pinned(Int8Tier::Scalar));
    let mut coverages = Vec::new();
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

            let weight_name = format!("talker.code_predictor.lm_head.{}.weight", depth - 1);
            let weight = widen(&checkpoint, &weight_name)?;

            // The sampler's eligibility set: top-k of the exact-f32 row.
            let reference = project(&weight, &x.data);
            let eligible: std::collections::HashSet<usize> =
                ranking(&reference)[..TOP_K].iter().copied().collect();

            // The production coarse pass: the real kernel, the real quantization recipe.
            let head_q8 = QuantizedMatrix::quantize(&weight, RESIDUAL_VOCAB, HIDDEN);
            let mut coarse = vec![0.0_f32; RESIDUAL_VOCAB];
            quant_linear(mode_w8a8, &x.data, &head_q8, None, 1, &mut coarse);

            let coarse_order = ranking(&coarse);
            let window: std::collections::HashSet<usize> = coarse_order
                .iter()
                .copied()
                .take(HEAD_REFINE_CANDIDATES)
                .collect();
            let mut worst_rank_in_window = 0_usize;
            for (rank, token) in coarse_order
                .into_iter()
                .take(HEAD_REFINE_CANDIDATES)
                .enumerate()
            {
                if eligible.contains(&token) {
                    worst_rank_in_window = worst_rank_in_window.max(rank + 1);
                }
            }
            let covered = eligible.is_subset(&window);
            coverages.push(Coverage {
                mode,
                depth,
                covered,
                worst_rank_in_window,
            });
        }
    }
    Ok(coverages)
}

#[test]
fn refine_window_covers_the_sampler_top_k_across_the_corpus() {
    const TEST: &str = "refine_window_covers_the_sampler_top_k_across_the_corpus";
    let coverages = match run_all() {
        Ok(coverages) => coverages,
        Err(reason) => {
            skip(TEST, &reason);
            return;
        }
    };

    let misses: Vec<String> = coverages
        .iter()
        .filter(|coverage| !coverage.covered)
        .map(|coverage| format!("{}/depth{}", coverage.mode, coverage.depth))
        .collect();
    assert!(
        misses.is_empty(),
        "int8 coarse scoring dropped sampler-eligible tokens out of the \
         {HEAD_REFINE_CANDIDATES}-wide refine window at {} — the constant no longer \
         covers the sampler and must grow (or top-k shrink) before this ships",
        misses.join(", ")
    );

    let worst = coverages
        .iter()
        .max_by_key(|coverage| coverage.worst_rank_in_window)
        .expect("non-empty");
    let mean_rank = coverages
        .iter()
        .map(|c| c.worst_rank_in_window)
        .sum::<usize>() as f64
        / coverages.len() as f64;
    eprintln!(
        "receipt: {{\"test\":\"refine_coverage\",\"comparisons\":{},\"window\":{},\"top_k\":{},\
         \"misses\":0,\"worst_eligible_rank\":{},\"worst_at\":\"{}/depth{}\",\
         \"mean_worst_rank\":{mean_rank:.2}}}",
        coverages.len(),
        HEAD_REFINE_CANDIDATES,
        TOP_K,
        worst.worst_rank_in_window,
        worst.mode,
        worst.depth,
    );
    Receipt::new(TEST, Outcome::Passed)
        .contract(CONTRACT)
        .seam("microdecoder.refine_coverage")
        .detail(serde_json::json!({
            "comparisons": coverages.len(),
            "window": HEAD_REFINE_CANDIDATES,
            "top_k": TOP_K,
            "misses": 0,
            "worst_eligible_rank": worst.worst_rank_in_window,
            "mean_worst_rank": mean_rank,
        }))
        .emit();
}
