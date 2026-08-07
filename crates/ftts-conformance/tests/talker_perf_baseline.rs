//! PERF baseline: ms/step for the safe-Rust f32 talker, against the census one-read floor.
//!
//! The z2w addendum's final criterion: "a baseline PERF_LEDGER row for the safe-Rust talker
//! (ms/frame prefill+decode, DRAM bytes/frame vs one-read floor) — the honest baseline Phase-3B
//! must beat." This is that measurement, not a gate: it is `#[ignore]`d so the suite never
//! times anything under arbitrary load, and it is run manually with the exact command recorded
//! in the ledger row it feeds (docs/PERF_LEDGER.md).
//!
//! What is timed:
//! - **prefill**: the oracle's captured 28-token assembled prompt through all 28 layers + final
//!   norm + head, exactly the argmax test's forward, KV populated;
//! - **decode steps**: N single-position steps over the growing KV cache — the shape that
//!   dominates real-time factor. Inputs are real prefill hidden rows (not synthetic zeros), so
//!   denormal behaviour matches production.
//!
//! Reported per step: wall ms, implied weight-traffic GB/s against the f32-widened talker body
//! (2x the checkpoint's 893.5 MB BF16 one-read floor — the reference widens at the accessor),
//! and the cv% across decode steps, because a row with cv > 5 is not admissible under the
//! ledger's own rules and must be re-measured in a quiet window instead of recorded.

use ftts_artifacts::safetensors::SafetensorsFile;
use ftts_conformance::npy::NpyArray;
use ftts_conformance::oracle::{CPU_FP32_ORACLE_CLASS, OracleFixtures, SeamRef};
use ftts_conformance::report::{OracleTier, Outcome, Receipt};
use ftts_model_qwen::talker::{
    PRIMARY_CODE_VOCAB_SIZE, RotaryRows, TalkerConfig, TalkerKvCache, TalkerLayerWeights,
    TalkerWeights, collapse_mrope, forward_talker, mrope_rows,
};
use serde_json::json;
use std::path::{Path, PathBuf};
use std::time::Instant;

const TEST_NAME: &str = "perf_baseline_talker_f32_reference";
const CASE: &str = "synthetic-tone-en";
const MODE: &str = "icl_non_streaming";
const GROUP: &str = "talker_free_running";
const LAYERS: usize = 28;
const DECODE_STEPS: usize = 8;

/// Talker body + primary head, BF16 checkpoint bytes (EXECUTION_CENSUS.json .components.talker:
/// body 880_934_912 + codec_embedding 6_291_456 + primary_head 6_291_456).
const TALKER_ONE_READ_FLOOR_BF16: u64 = 893_517_824;

fn checkpoint_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../docs/truth-pack/snapshots/hf/model.safetensors")
}

fn skip(reason: &str) {
    Receipt::new(TEST_NAME, Outcome::Skipped)
        .contract("PerfBaseline/talker")
        .seam("talker.decode_step.wall_ms")
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
        "rotary is [3, 1, seq, head_dim]"
    );
    [
        array.data[..stride].to_vec(),
        array.data[stride..2 * stride].to_vec(),
        array.data[2 * stride..].to_vec(),
    ]
}

#[test]
#[ignore = "manual perf baseline; run alone in a quiet window and record the row in docs/PERF_LEDGER.md"]
fn talker_f32_reference_baseline() {
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

    let config = TalkerConfig::default();
    let hidden_size = config.hidden_size;
    let head_dim = config.head_dim;

    let file = SafetensorsFile::open(&checkpoint).expect("pinned checkpoint opens");
    file.advise_random();
    let hydrate_start = Instant::now();
    let codec_head = widen(&file, "talker.codec_head.weight");
    let final_norm = widen(&file, "talker.model.norm.weight");
    let layers: Vec<OwnedLayer> = (0..LAYERS)
        .map(|layer| OwnedLayer::load(&file, layer))
        .collect();
    let hydrate_ms = hydrate_start.elapsed().as_secs_f64() * 1e3;
    let weights = TalkerWeights {
        layers: layers.iter().map(OwnedLayer::borrow).collect(),
        final_norm: &final_norm,
        codec_head: &codec_head,
    };

    // Prefill: the oracle's captured assembled prompt, exactly as talker_argmax_l3 runs it.
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
    let seq = hidden_in.data.len() / hidden_size;
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

    let mut hidden = hidden_in.data.clone();
    let mut cache = TalkerKvCache::new();
    let mut logits = vec![0.0f32; seq * PRIMARY_CODE_VOCAB_SIZE];
    let prefill_start = Instant::now();
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
    let prefill_ms = prefill_start.elapsed().as_secs_f64() * 1e3;

    // Decode steps: single positions over the growing cache, inputs = real hidden rows.
    let mut step_ms = Vec::with_capacity(DECODE_STEPS);
    let mut step_logits = vec![0.0f32; PRIMARY_CODE_VOCAB_SIZE];
    for step in 0..DECODE_STEPS {
        let mut row = hidden[(step % seq) * hidden_size..(step % seq + 1) * hidden_size].to_vec();
        let position = (seq + step) as i64;
        let (step_cos, step_sin) = mrope_rows(&[position], head_dim, 1.0e6);
        let step_mask = vec![0.0f32; cache.len() + 1];
        let start = Instant::now();
        forward_talker(
            &config,
            &weights,
            RotaryRows {
                cos: &step_cos,
                sin: &step_sin,
            },
            &step_mask,
            &mut row,
            1,
            &mut cache,
            &mut step_logits,
        );
        step_ms.push(start.elapsed().as_secs_f64() * 1e3);
    }

    let mean = step_ms.iter().sum::<f64>() / step_ms.len() as f64;
    let variance =
        step_ms.iter().map(|ms| (ms - mean).powi(2)).sum::<f64>() / (step_ms.len() - 1) as f64;
    let cv_percent = variance.sqrt() / mean * 100.0;
    // The reference widens BF16 to f32 at hydration, so each decode step streams 2x the
    // checkpoint bytes through DRAM (first order; KV and activations are noise at this scale).
    let f32_bytes_per_step = TALKER_ONE_READ_FLOOR_BF16 as f64 * 2.0;
    let implied_gb_per_s = f32_bytes_per_step / (mean / 1e3) / 1e9;

    println!("hydrate: {hydrate_ms:.0} ms");
    println!("prefill (seq {seq}): {prefill_ms:.1} ms");
    for (step, ms) in step_ms.iter().enumerate() {
        println!("decode step {step}: {ms:.1} ms");
    }
    println!(
        "decode mean {mean:.1} ms/step, cv {cv_percent:.1}%, implied f32 weight traffic \
         {implied_gb_per_s:.2} GB/s (floor: {TALKER_ONE_READ_FLOOR_BF16} BF16 bytes/step, \
         f32-widened 2x)"
    );

    Receipt::new(TEST_NAME, Outcome::Passed)
        .contract("PerfBaseline/talker")
        .seam("talker.decode_step.wall_ms")
        .reason(format!(
            "f32 reference talker: prefill(seq {seq}) {prefill_ms:.1} ms; decode mean \
             {mean:.1} ms/step over {DECODE_STEPS} steps, cv {cv_percent:.1}%; measurement row \
             admissible only if cv <= 5"
        ))
        .oracle_tier(OracleTier::CpuFp32Fallback)
        .detail(json!({
            "hydrate_ms": hydrate_ms,
            "prefill_ms": prefill_ms,
            "prefill_seq": seq,
            "decode_step_ms": step_ms,
            "decode_mean_ms": mean,
            "cv_percent": cv_percent,
            "one_read_floor_bf16_bytes": TALKER_ONE_READ_FLOOR_BF16,
            "implied_f32_traffic_gb_per_s": implied_gb_per_s,
        }))
        .emit();
}
