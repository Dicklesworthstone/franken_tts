//! L3 discrete parity: the talker's primary-code **argmax is EXACT** through our forward.
//!
//! Bead `frankentts-p1-talker-z2w`'s remaining criterion. The L2 story is settled and ledgered
//! (DISC-002): our scalar f32 forward differs from the pinned oracle's Accelerate-SGEMM +
//! SLEEF-vectorized stack by f32 accumulation rounding — ~5.8e-7 relative at worst, cosine
//! indistinguishable from 1, measured across four independent seam families. Per the equivalence
//! tiers, continuous activations are held to a measured budget; **the discrete artifact is where
//! EXACT belongs**. This test is that gate, at two scopes:
//!
//! 1. **Head seam** (both captured steps): our `codec_head` GEMM over the oracle's own
//!    post-final-norm hidden must produce logits whose argmax equals the argmax of the oracle's
//!    captured logits — the prefill row and the decode-step row both.
//! 2. **Full talker stack** (prefill): the oracle's assembled layer-00 input driven through all
//!    28 of our layers, our final norm, and our head — accumulated rounding included — must
//!    still select the oracle's primary code. This is the strongest discrete claim the pack
//!    supports for the talker in isolation: rounding may move logits in the last few ULPs, but
//!    if it ever crosses a decision boundary this fails loudly.
//!
//! The logit tensors are additionally reported (not gated) at their measured relative error so
//! the receipt records how much margin the argmax decision actually had.
//!
//! Model-gated twice (fixtures + pinned checkpoint); absent inputs produce a loud skip receipt.

#![cfg(feature = "ultra-tests")]

use ftts_artifacts::safetensors::SafetensorsFile;
use ftts_conformance::npy::NpyArray;
use ftts_conformance::oracle::{CPU_FP32_ORACLE_CLASS, OracleFixtures, SeamRef};
use ftts_conformance::report::{OracleTier, Outcome, Receipt};
use ftts_model_qwen::talker::{
    PRIMARY_CODE_VOCAB_SIZE, RotaryRows, TalkerConfig, TalkerKvCache, TalkerLayerWeights,
    TalkerWeights, collapse_mrope, forward_talker,
};
use serde_json::json;
use std::path::{Path, PathBuf};

const TEST_NAME: &str = "contract_a_l3_talker_primary_code_argmax_exact";
const CONTRACT: &str = "ConformanceExact/L3";
const CASE: &str = "synthetic-tone-en";
const MODE: &str = "icl_non_streaming";
const GROUP: &str = "talker_free_running";
const LAYERS: usize = 28;

fn checkpoint_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../docs/truth-pack/snapshots/hf/model.safetensors")
}

fn skip(reason: &str) {
    Receipt::new(TEST_NAME, Outcome::Skipped)
        .contract(CONTRACT)
        .seam("talker.codec_head.output.argmax")
        .reason(reason)
        .oracle_tier(OracleTier::CpuFp32Fallback)
        .emit();
}

fn seam(name: &str) -> SeamRef<'_> {
    SeamRef {
        case: CASE,
        mode: MODE,
        group: GROUP,
        seam: name,
    }
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

/// First index of the maximum, torch `argmax` semantics.
fn argmax(row: &[f32]) -> usize {
    let mut best = 0usize;
    for (index, &value) in row.iter().enumerate() {
        if value > row[best] {
            best = index;
        }
    }
    best
}

/// Worst elementwise |diff| / max|expected| over a tensor, for the receipt.
fn relative_error(expected: &[f32], actual: &[f32]) -> f64 {
    let scale = expected
        .iter()
        .fold(0.0f64, |acc, &value| acc.max(f64::from(value.abs())));
    let max_abs = expected
        .iter()
        .zip(actual)
        .fold(0.0f64, |acc, (&expected, &actual)| {
            acc.max(f64::from((expected - actual).abs()))
        });
    if scale > 0.0 { max_abs / scale } else { 0.0 }
}

struct OwnedLayer {
    tensors: [Vec<f32>; 11],
}

impl OwnedLayer {
    fn load(file: &SafetensorsFile, layer: usize) -> Self {
        let prefix = format!("talker.model.layers.{layer}");
        Self {
            tensors: [
                widen(file, &format!("{prefix}.input_layernorm.weight")),
                widen(file, &format!("{prefix}.self_attn.q_proj.weight")),
                widen(file, &format!("{prefix}.self_attn.k_proj.weight")),
                widen(file, &format!("{prefix}.self_attn.v_proj.weight")),
                widen(file, &format!("{prefix}.self_attn.q_norm.weight")),
                widen(file, &format!("{prefix}.self_attn.k_norm.weight")),
                widen(file, &format!("{prefix}.self_attn.o_proj.weight")),
                widen(file, &format!("{prefix}.post_attention_layernorm.weight")),
                widen(file, &format!("{prefix}.mlp.gate_proj.weight")),
                widen(file, &format!("{prefix}.mlp.up_proj.weight")),
                widen(file, &format!("{prefix}.mlp.down_proj.weight")),
            ],
        }
    }

    fn borrow(&self) -> TalkerLayerWeights<'_> {
        let [
            input_layernorm,
            q_proj,
            k_proj,
            v_proj,
            q_norm,
            k_norm,
            o_proj,
            post_attention_layernorm,
            gate_proj,
            up_proj,
            down_proj,
        ] = &self.tensors;
        TalkerLayerWeights {
            input_layernorm,
            q_proj,
            k_proj,
            v_proj,
            q_norm,
            k_norm,
            o_proj,
            post_attention_layernorm,
            gate_proj,
            up_proj,
            down_proj,
        }
    }
}

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

#[test]
fn talker_primary_code_argmax_is_exact() {
    let fixtures = match OracleFixtures::open_default() {
        Ok(fixtures) => fixtures,
        Err(error) => {
            skip(&format!("fixtures unavailable: {error}"));
            return;
        }
    };
    fixtures
        .require_oracle_class(CPU_FP32_ORACLE_CLASS)
        .expect("pack is the CPU-fp32 tier");
    let checkpoint = checkpoint_path();
    if !checkpoint.is_file() {
        skip(&format!(
            "pinned checkpoint absent at {}",
            checkpoint.display()
        ));
        return;
    }
    let file = SafetensorsFile::open(&checkpoint).expect("pinned checkpoint opens");
    file.advise_random();

    let config = TalkerConfig::default();
    let hidden_size = config.hidden_size;
    let head_dim = config.head_dim;
    let codec_head = widen(&file, "talker.codec_head.weight");
    let final_norm = widen(&file, "talker.model.norm.weight");

    // --- Gate 1: head seam, argmax exact at every captured step -------------------------------
    let mut head_margin = Vec::new();
    for step in 0..2usize {
        let head_input = fixtures
            .seam(&seam("talker.codec_head.input"), "args.0", step)
            .expect("post-final-norm hidden captured");
        let expected_logits = fixtures
            .seam(&seam("talker.codec_head.output"), "tensor", step)
            .expect("head logits captured");
        let rows = head_input.data.len() / hidden_size;
        assert_eq!(
            expected_logits.data.len(),
            rows * PRIMARY_CODE_VOCAB_SIZE,
            "step {step}: captured logits disagree with the captured head input rows"
        );
        let mut logits = vec![0.0f32; rows * PRIMARY_CODE_VOCAB_SIZE];
        ftts_kernels::f32ref::linear(
            &head_input.data,
            &codec_head,
            None,
            rows,
            hidden_size,
            PRIMARY_CODE_VOCAB_SIZE,
            &mut logits,
        );
        // The decision row: the last position is what sampling reads.
        let ours = &logits[(rows - 1) * PRIMARY_CODE_VOCAB_SIZE..];
        let oracle = &expected_logits.data[(rows - 1) * PRIMARY_CODE_VOCAB_SIZE..];
        let our_pick = argmax(ours);
        let oracle_pick = argmax(oracle);
        assert_eq!(
            our_pick, oracle_pick,
            "step {step}: head-seam argmax diverged — our GEMM picks {our_pick}, the oracle's \
             captured logits pick {oracle_pick}"
        );
        head_margin.push(json!({
            "step": step,
            "argmax": our_pick,
            "relative_error": relative_error(oracle, ours),
        }));
    }

    // --- Gate 2: full 28-layer stack + final norm + head, accumulated, argmax exact -----------
    let input_seam = seam("talker.layer_00.input");
    let hidden_in = fixtures
        .seam(&input_seam, "args.0", 0)
        .expect("assembled prefill hidden captured");
    let mask = fixtures
        .seam(&input_seam, "kwargs.attention_mask", 0)
        .expect("prefill mask captured");
    let cos = fixtures
        .seam(&input_seam, "kwargs.position_embeddings.0", 0)
        .expect("rotary cos captured");
    let sin = fixtures
        .seam(&input_seam, "kwargs.position_embeddings.1", 0)
        .expect("rotary sin captured");
    let oracle_logits = fixtures
        .seam(&seam("talker.codec_head.output"), "tensor", 0)
        .expect("prefill logits captured");

    let seq = hidden_in.data.len() / hidden_size;
    assert_eq!(mask.data.len(), seq * seq, "prefill mask is [seq, seq]");
    let sections = [24usize, 20, 20];
    let cos_axes = split_axes(&cos, seq, head_dim);
    let sin_axes = split_axes(&sin, seq, head_dim);
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

    let layers: Vec<OwnedLayer> = (0..LAYERS)
        .map(|layer| OwnedLayer::load(&file, layer))
        .collect();
    let weights = TalkerWeights {
        layers: layers.iter().map(OwnedLayer::borrow).collect(),
        final_norm: &final_norm,
        codec_head: &codec_head,
    };

    let mut hidden = hidden_in.data.clone();
    let mut cache = TalkerKvCache::new();
    let mut logits = vec![0.0f32; seq * PRIMARY_CODE_VOCAB_SIZE];
    forward_talker(
        &config,
        &weights,
        RotaryRows {
            cos: &cos_rows,
            sin: &sin_rows,
        },
        &mask.data,
        &mut hidden,
        seq,
        &mut cache,
        &mut logits,
    );

    let ours = &logits[(seq - 1) * PRIMARY_CODE_VOCAB_SIZE..];
    let oracle = &oracle_logits.data[(seq - 1) * PRIMARY_CODE_VOCAB_SIZE..];
    let our_pick = argmax(ours);
    let oracle_pick = argmax(oracle);
    let full_stack_relative = relative_error(oracle, ours);
    assert_eq!(
        our_pick, oracle_pick,
        "full-stack argmax diverged after 28 accumulated layers — ours {our_pick}, oracle \
         {oracle_pick}, logits relative error {full_stack_relative:e}; the ledgered rounding \
         (DISC-002) crossed a decision boundary and is no longer harmless"
    );

    Receipt::new(TEST_NAME, Outcome::Passed)
        .contract(CONTRACT)
        .seam("talker.codec_head.output.argmax")
        .reason(format!(
            "argmax EXACT at the head seam (both captured steps) and through the full \
             28-layer accumulated stack (pick {our_pick}); full-stack logits relative error \
             {full_stack_relative:e} (reported, not gated; activation budget is DISC-002)"
        ))
        .tolerance(0.0, "discrete rung: argmax equality, no epsilon applies")
        .oracle_tier(OracleTier::CpuFp32Fallback)
        .detail(json!({
            "head_seam": head_margin,
            "full_stack": {
                "argmax": our_pick,
                "relative_error": full_stack_relative,
                "seq": seq,
            },
        }))
        .emit();
}
