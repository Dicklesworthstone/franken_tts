//! Engine-level e2e: `TtsEngine::synthesize` driving `QwenGenerator` with the REAL pinned weights.
//!
//! This is the first assembled-forward receipt for frankentts-p1-e2e-miy: the engine's admission,
//! budget, and decode loop run against the real 28-layer talker and 5-layer microdecoder hydrated
//! from the pinned checkpoint, seeded with the ft7 oracle's own assembled prompt
//! (`talker.input.input/kwargs.inputs_embeds`), and the first generated frame is compared
//! code-for-code against the oracle's captured `talker.codec_codes`.
//!
//! Prompt-header construction is deliberately NOT re-derived here — it belongs to the prompt and
//! speaker beads. Feeding the oracle's prefill through `QwenGenerator::begin_with_prefill` scopes
//! this receipt to what it actually proves: talker prefill + canonical-greedy `c0` + the 15-step
//! microdecoder, end-to-end under the engine, at the CPU-fp32 tier.
//!
//! Model-gated twice (fixtures + checkpoint); absent inputs produce a loud skip receipt, never a
//! silent green.

use ftts_artifacts::safetensors::SafetensorsFile;
use ftts_conformance::npy;
use ftts_conformance::oracle::{FixtureError, OracleFixtures, SeamRef};
use ftts_conformance::report::{OracleTier, Outcome, Receipt};
use ftts_core::{
    CancellationToken, CodeFrame, EngineConfig, FrameGenerator, GenerationError,
    NormalizationOptions, NormalizationTrace, PreparedText, SynthesisEvent, SynthesisRequest,
    TextPreparationError, TextPreparer, TtsEngine,
};
use ftts_model_qwen::generate::{
    FeedbackTables, QwenGenerator, QwenGeneratorConfig, TextEmbeddingWeights,
};
use ftts_model_qwen::microdecoder::{
    LayerWeights, MicrodecoderConfig, MicrodecoderWeights, RESIDUAL_DEPTHS,
};
use ftts_model_qwen::prompt::{CloneMode, PromptHeader, PromptMode};
use ftts_model_qwen::sampler::SamplingMode;
use ftts_model_qwen::talker::{
    TalkerConfig, TalkerLayerWeights, TalkerWeights, CODE_GROUP_COUNT, TALKER_LAYER_COUNT,
};
use std::path::{Path, PathBuf};
use std::time::Duration;

const TEST_NAME: &str = "contract_a_engine_e2e_first_frame_cpu_fp32_exact";
const CASE: &str = "synthetic-tone-en";
const MODE: &str = "xvector_streaming";
const GROUP: &str = "talker_free_running";
const HIDDEN: usize = 1024;

/// The pinned talker checkpoint, alongside the truth-pack snapshots.
fn checkpoint_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../docs/truth-pack/snapshots/hf/model.safetensors")
}

fn skip(reason: &str) {
    Receipt::new(TEST_NAME, Outcome::Skipped)
        .contract("ConformanceExact/e2e")
        .seam("engine.synthesize.first_frame")
        .reason(reason)
        .oracle_tier(OracleTier::CpuFp32Fallback)
        .emit();
}

/// Widen a whole tensor to `f32` through the accessor, which is how the engine reads BF16.
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

/// Every widened tensor the generator borrows, owned for the test's lifetime.
struct Hydrated {
    talker_layers: Vec<[Vec<f32>; 11]>,
    talker_final_norm: Vec<f32>,
    codec_head: Vec<f32>,
    talker_codec_embedding: Vec<f32>,
    residual_embeddings: Vec<Vec<f32>>,
    micro_layers: Vec<[Vec<f32>; 11]>,
    micro_final_norm: Vec<f32>,
    micro_heads: Vec<Vec<f32>>,
    text_stub_table: Vec<f32>,
    text_stub_fc1: Vec<f32>,
    text_stub_fc1_bias: Vec<f32>,
    text_stub_fc2: Vec<f32>,
    text_stub_fc2_bias: Vec<f32>,
    tts_pad: Vec<f32>,
}

/// The eleven per-layer tensors, in [`TalkerLayerWeights`] field order.
fn layer_tensors(file: &SafetensorsFile, prefix: &str) -> [Vec<f32>; 11] {
    [
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
    ]
}

fn borrow_layer(tensors: &[Vec<f32>; 11]) -> TalkerLayerWeights<'_> {
    TalkerLayerWeights {
        input_layernorm: &tensors[0],
        q_proj: &tensors[1],
        k_proj: &tensors[2],
        v_proj: &tensors[3],
        q_norm: &tensors[4],
        k_norm: &tensors[5],
        o_proj: &tensors[6],
        post_attention_layernorm: &tensors[7],
        gate_proj: &tensors[8],
        up_proj: &tensors[9],
        down_proj: &tensors[10],
    }
}

fn borrow_micro_layer(tensors: &[Vec<f32>; 11]) -> LayerWeights<'_> {
    LayerWeights {
        input_norm: &tensors[0],
        q_proj: &tensors[1],
        k_proj: &tensors[2],
        v_proj: &tensors[3],
        q_norm: &tensors[4],
        k_norm: &tensors[5],
        o_proj: &tensors[6],
        post_attention_norm: &tensors[7],
        gate_proj: &tensors[8],
        up_proj: &tensors[9],
        down_proj: &tensors[10],
    }
}

impl Hydrated {
    fn load(file: &SafetensorsFile, tts_pad: Vec<f32>) -> Self {
        Self {
            talker_layers: (0..TALKER_LAYER_COUNT)
                .map(|layer| layer_tensors(file, &format!("talker.model.layers.{layer}")))
                .collect(),
            talker_final_norm: widen(file, "talker.model.norm.weight"),
            codec_head: widen(file, "talker.codec_head.weight"),
            talker_codec_embedding: widen(file, "talker.model.codec_embedding.weight"),
            residual_embeddings: (0..CODE_GROUP_COUNT - 1)
                .map(|table| {
                    widen(
                        file,
                        &format!("talker.code_predictor.model.codec_embedding.{table}.weight"),
                    )
                })
                .collect(),
            micro_layers: (0..MicrodecoderConfig::default().num_layers)
                .map(|layer| {
                    layer_tensors(file, &format!("talker.code_predictor.model.layers.{layer}"))
                })
                .collect(),
            micro_final_norm: widen(file, "talker.code_predictor.model.norm.weight"),
            micro_heads: (0..RESIDUAL_DEPTHS)
                .map(|head| widen(file, &format!("talker.code_predictor.lm_head.{head}.weight")))
                .collect(),
            // The text path is unused behind `begin_with_prefill`; minimal well-shaped stubs.
            text_stub_table: vec![0.0; 2],
            text_stub_fc1: vec![0.0; 2],
            text_stub_fc1_bias: vec![0.0; 1],
            text_stub_fc2: vec![0.0; HIDDEN],
            text_stub_fc2_bias: vec![0.0; HIDDEN],
            tts_pad,
        }
    }

    fn generator<'a>(
        &'a self,
        talker_layers: &'a [TalkerLayerWeights<'a>],
        micro_layers: &'a [LayerWeights<'a>],
        micro_residual: &'a [&'a [f32]],
        micro_heads: &'a [&'a [f32]],
    ) -> QwenGenerator<'a> {
        QwenGenerator::new(QwenGeneratorConfig {
            talker_config: TalkerConfig::default(),
            talker_weights: TalkerWeights {
                layers: talker_layers.to_vec(),
                final_norm: &self.talker_final_norm,
                codec_head: &self.codec_head,
            },
            text: TextEmbeddingWeights {
                table: &self.text_stub_table,
                embed_width: 2,
                fc1_weight: &self.text_stub_fc1,
                fc1_bias: &self.text_stub_fc1_bias,
                fc2_weight: &self.text_stub_fc2,
                fc2_bias: &self.text_stub_fc2_bias,
            },
            feedback: FeedbackTables {
                talker_codec: &self.talker_codec_embedding,
                residual: self.residual_embeddings.iter().map(Vec::as_slice).collect(),
            },
            microdecoder_config: MicrodecoderConfig::default(),
            microdecoder_weights: MicrodecoderWeights {
                layers: micro_layers,
                talker_codec_embedding: &self.talker_codec_embedding,
                // The microdecoder's internal tables for positions 2..=15 are the first 14 of the
                // same per-depth set the feedback path uses.
                residual_embeddings: micro_residual,
                heads: micro_heads,
                final_norm: &self.micro_final_norm,
            },
            prompt_mode: PromptMode {
                clone_mode: CloneMode::XVector,
                non_streaming_mode: false,
            },
            header: PromptHeader {
                role: vec![vec![0.0; HIDDEN]; 3],
                codec_prefill: vec![vec![0.0; HIDDEN]; 2],
                tts_bos: vec![0.0; HIDDEN],
                tts_pad: self.tts_pad.clone(),
            },
            tts_eos: vec![0.0; HIDDEN],
            reference: None,
            sampling_mode: SamplingMode::CanonicalGreedy,
            seed: 0,
        })
    }
}

/// Feeds the engine the fixture's wrapped prompt ids without re-running normalization.
struct FixtureTextPreparer {
    token_ids: Vec<u32>,
}

impl TextPreparer for FixtureTextPreparer {
    fn prepare(
        &self,
        _text: &str,
        options: &NormalizationOptions,
    ) -> Result<PreparedText, TextPreparationError> {
        Ok(PreparedText::new(
            self.token_ids.clone(),
            NormalizationTrace {
                mode: options.mode,
                unicode_version: "fixture".to_owned(),
                changes: Vec::new(),
            },
        ))
    }
}

/// `FrameGenerator` adapter that seeds the oracle's assembled prefill instead of re-deriving
/// prompt assembly, then defers every frame to the real `QwenGenerator`.
struct OraclePromptGenerator<'a> {
    inner: QwenGenerator<'a>,
    prefill: Vec<f32>,
    seq: usize,
    trailing_text_hidden: Vec<Vec<f32>>,
}

impl FrameGenerator for OraclePromptGenerator<'_> {
    fn begin_utterance(&mut self, _prepared: &PreparedText) -> Result<(), GenerationError> {
        self.inner
            .begin_with_prefill(&self.prefill, self.seq, self.trailing_text_hidden.clone())
    }

    fn next_frame(&mut self) -> Result<Option<CodeFrame>, GenerationError> {
        self.inner.next_frame()
    }
}

#[test]
fn engine_synthesize_reproduces_the_oracle_first_frame_exactly() {
    let fixtures = match OracleFixtures::open_default() {
        Ok(fixtures) => fixtures,
        Err(FixtureError::PackAbsent { path }) => {
            skip(&format!("oracle fixture pack absent at {path}"));
            return;
        }
        Err(error) => panic!("fixture pack unusable: {error}"),
    };
    let checkpoint = checkpoint_path();
    if !checkpoint.exists() {
        skip(&format!(
            "pinned weights absent at {}; run docs/truth-pack/fetch-truth-pack.sh --with-weights",
            checkpoint.display()
        ));
        return;
    }

    let input_seam = SeamRef {
        case: CASE,
        mode: MODE,
        group: GROUP,
        seam: "talker.input.input",
    };
    let prefill = fixtures
        .seam(&input_seam, "kwargs.inputs_embeds", 0)
        .expect("assembled prefill captured");
    assert_eq!(prefill.shape.len(), 3, "prefill is [1, seq, hidden]");
    let seq = prefill.shape[1];
    assert_eq!(prefill.shape[2], HIDDEN, "prefill width");
    let trailing = fixtures
        .seam(&input_seam, "kwargs.trailing_text_hidden", 0)
        .expect("trailing text stream captured");
    let tts_pad = fixtures
        .seam(&input_seam, "kwargs.tts_pad_embed", 0)
        .expect("tts_pad embedding captured");
    let expected_codes = npy::read_i64(&fixtures.seam_path(
        &SeamRef {
            case: CASE,
            mode: MODE,
            group: GROUP,
            seam: "talker.codec_codes",
        },
        "tensor",
        0,
    ))
    .expect("oracle first-frame codes captured");
    assert_eq!(
        expected_codes.data.len(),
        CODE_GROUP_COUNT,
        "the oracle frame has 16 codes"
    );
    let prompt_ids = npy::read_i64(&fixtures.seam_path(
        &SeamRef {
            case: CASE,
            mode: MODE,
            group: "prompt_build",
            seam: "prompt.text_ids",
        },
        "tensor",
        0,
    ))
    .expect("wrapped prompt ids captured");

    let file = SafetensorsFile::open(&checkpoint).expect("pinned checkpoint maps and parses");
    file.advise_random();
    let hydrated = Hydrated::load(&file, tts_pad.data.clone());
    let talker_layers: Vec<TalkerLayerWeights<'_>> =
        hydrated.talker_layers.iter().map(borrow_layer).collect();
    let micro_layers: Vec<LayerWeights<'_>> =
        hydrated.micro_layers.iter().map(borrow_micro_layer).collect();
    let micro_residual: Vec<&[f32]> = hydrated.residual_embeddings[..RESIDUAL_DEPTHS - 1]
        .iter()
        .map(Vec::as_slice)
        .collect();
    let micro_heads: Vec<&[f32]> = hydrated.micro_heads.iter().map(Vec::as_slice).collect();

    let trailing_rows: Vec<Vec<f32>> = trailing.data.chunks(HIDDEN).map(<[f32]>::to_vec).collect();
    let mut generator = OraclePromptGenerator {
        inner: hydrated.generator(&talker_layers, &micro_layers, &micro_residual, &micro_heads),
        prefill: prefill.data.clone(),
        seq,
        trailing_text_hidden: trailing_rows,
    };

    let engine = TtsEngine::new(EngineConfig {
        // The f32 reference forward is deliberately unoptimized; give it room.
        synthesis_stage_budget: Duration::from_secs(600),
        ..EngineConfig::default()
    })
    .expect("engine constructs");
    let preparer = FixtureTextPreparer {
        token_ids: prompt_ids.data.iter().map(|&id| id as u32).collect(),
    };
    let result = engine
        .synthesize(
            SynthesisRequest {
                text: String::new(),
                normalization_options: NormalizationOptions::default(),
                trace_normalization: false,
            },
            &preparer,
            &mut generator,
            &CancellationToken::new(),
            &|_: SynthesisEvent| {},
        )
        .expect("synthesis completes within budget");

    assert!(
        result.generated_frames > 0,
        "the engine loop must emit at least the oracle-verified first frame"
    );
    let first: Vec<i64> = result.code_frames[0]
        .codes
        .iter()
        .map(|&code| i64::from(code))
        .collect();
    assert_eq!(
        first, expected_codes.data,
        "first generated frame must match the oracle's 16 codes exactly"
    );

    Receipt::new(TEST_NAME, Outcome::Passed)
        .contract("ConformanceExact/e2e")
        .seam("engine.synthesize.first_frame")
        .oracle_tier(OracleTier::CpuFp32Fallback)
        .emit();
}
