//! Engine-level e2e: `TtsEngine::synthesize` driving `QwenGenerator` with the REAL pinned weights.
//!
//! This is the assembled-forward receipt for frankentts-p1-e2e-miy: the engine's admission,
//! budget, and decode loop run against the real 28-layer talker and 5-layer microdecoder hydrated
//! from the pinned checkpoint, seeded with the ft7 oracle's own assembled prompt
//! (`talker.input.input/kwargs.inputs_embeds`), and the generated codes are compared
//! code-for-code against the oracle's captured `talker.codec_codes` — for the **whole utterance**,
//! every frame through the stop, not merely the first.
//!
//! # What is covered
//!
//! 1. **Every captured frame, not just the first.** The mode manifest's `generated_frames` sets the
//!    length; the codes tensor must agree with it, and the engine's frames must match code for
//!    code, with the first divergence localized by frame and depth.
//! 2. **The capture's own digest.** The codes are hashed the way the capture hashed them
//!    (little-endian `int64`, `[frames, 16]`) and checked against `generated_codes_sha256` — one
//!    value no partial comparison can accidentally satisfy.
//! 3. **One step past the codes.** Upstream drops the final step's frame, but it *kept that step's
//!    logits*. Running our production sampler over the oracle's own `talker.codec_head.output` at
//!    that step yields the code the reference would have emitted, and the engine must produce it.
//!    This is what reaches the decode loop's feedback path — frame codes summed onto the next
//!    input, KV appended, mRoPE advanced — which frame-0-only parity cannot touch.
//!
//! # What this pack cannot cover: the EOS stop
//!
//! This capture contains **no EOS**, so there is no stop here to be parity with, and the test says
//! so in its receipt rather than manufacturing the claim. Three facts force that reading:
//!
//! - The pinned corpus caps generation at `max_new_tokens = 2` — the script's own minimum. The
//!   reference's `min_new_tokens = 2` processor masks EOS while fewer than two tokens have been
//!   generated, so EOS is *structurally unreachable* within this capture.
//! - The captured step-1 logits put EOS (2150) at 5.5 against a winning code at 26.5. Nothing near
//!   a stop.
//! - Upstream truncates at the first EOS **found in the emitted codes** (`has_stop_token`); with no
//!   EOS drawn it keeps every frame. So the codes — not the frame count — decide.
//!
//! In particular `generated_frames = 1` under a 2-token cap is **not** evidence of an early stop.
//! Upstream drops the last step's frame because that step's microdecoder never runs, so an N-step
//! capture always reports N-1 frames. A "frames < cap ⟹ stopped at EOS" heuristic reads this pack
//! as EOS-terminated and is simply wrong; the code below checks codebook 0 for EOS instead, and
//! only then asserts that the engine stops where the reference stopped.
//!
//! Closing this gap needs a recapture with a cap high enough for the utterance to end on its own,
//! not a change here.
//!
//! Prompt-header construction is deliberately NOT re-derived here — it belongs to the prompt and
//! speaker beads. Feeding the oracle's prefill through `QwenGenerator::begin_with_prefill` scopes
//! this receipt to what it actually proves: talker prefill + canonical-greedy `c0` + the 15-step
//! microdecoder + the feedback step into the next frame, end-to-end under the engine, at the
//! CPU-fp32 tier. Not the EOS stop — see above.
//!
//! Model-gated twice (fixtures + checkpoint); absent inputs produce a loud skip receipt, never a
//! silent green.

#![cfg(feature = "ultra-tests")]

use ftts_artifacts::safetensors::SafetensorsFile;
use ftts_conformance::npy;
use ftts_conformance::oracle::{FixtureError, OracleFixtures, SeamRef};
use ftts_conformance::report::{OracleTier, Outcome, Receipt};
use ftts_core::{
    CancellationToken, EngineConfig, FrameGenerator, FrameStep, GenerationError,
    NormalizationOptions, NormalizationTrace, PreparedText, SynthesisEvent, SynthesisRequest,
    TextPreparationError, TextPreparer, TtsEngine,
};
use ftts_model_qwen::generate::{
    FeedbackResidual, FeedbackTables, QwenGenerator, QwenGeneratorConfig, TextEmbeddingWeights,
};
use ftts_model_qwen::microdecoder::{
    LayerWeights, MicrodecoderConfig, MicrodecoderWeights, RESIDUAL_DEPTHS,
};
use ftts_model_qwen::prompt::{CloneMode, PromptHeader, PromptMode};
use ftts_model_qwen::sampler::{CODEC_EOS_TOKEN_ID, QwenSampler, SamplingMode};
use ftts_model_qwen::talker::{
    CODE_GROUP_COUNT, TALKER_LAYER_COUNT, TalkerConfig, TalkerLayerWeights, TalkerWeights,
};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

const TEST_NAME: &str = "contract_a_engine_e2e_full_utterance_cpu_fp32_exact";
const SEAM: &str = "engine.synthesize.full_utterance";
const CASE: &str = "synthetic-tone-en";
const MODE: &str = "xvector_streaming";
const GROUP: &str = "talker_free_running";
const HIDDEN: usize = 1024;

/// The pinned talker checkpoint, alongside the truth-pack snapshots.
fn checkpoint_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../docs/truth-pack/snapshots/hf/model.safetensors")
}

/// Where the digest-pinned corpus that fixed this capture's token cap lives.
fn corpus_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../docs/conformance/oracle_corpus.json")
}

fn skip(reason: &str) {
    Receipt::new(TEST_NAME, Outcome::Skipped)
        .contract("ConformanceExact/e2e")
        .seam(SEAM)
        .reason(reason)
        .oracle_tier(OracleTier::CpuFp32Fallback)
        .emit();
}

/// Records the frame ceiling the engine admitted this utterance under.
///
/// Without it, "the engine emitted exactly N frames" cannot distinguish a genuine EOS stop from
/// the decode loop simply running out of its budgeted frames at N.
#[derive(Default)]
struct CeilingObserver {
    predicted_max_frames: AtomicU64,
}

impl ftts_core::SynthesisObserver for CeilingObserver {
    fn on_event(&self, event: SynthesisEvent) {
        if let SynthesisEvent::ResourceAdmission {
            admitted: true,
            predicted_max_frames,
            ..
        } = event
        {
            self.predicted_max_frames
                .store(predicted_max_frames, Ordering::Relaxed);
        }
    }
}

/// The token cap recorded for `case` in the corpus, but only if the corpus is byte-identical to
/// the one the pack was captured from.
///
/// Returning `None` on any mismatch is deliberate: a cap read from a corpus that has since drifted
/// would silently turn "the oracle stopped early" into an unfounded claim.
fn corpus_max_new_tokens(fixtures: &OracleFixtures, case: &str) -> Option<usize> {
    let provenance = fixtures.provenance().ok()?;
    let pinned = provenance["corpus_sha256"].as_str()?;
    let bytes = std::fs::read(corpus_path()).ok()?;
    let actual = format!("{:x}", Sha256::digest(&bytes));
    if actual != pinned {
        return None;
    }
    let corpus: serde_json::Value = serde_json::from_slice(&bytes).ok()?;
    corpus["cases"]
        .as_array()?
        .iter()
        .find(|entry| entry["id"].as_str() == Some(case))?["max_new_tokens"]
        .as_u64()
        .and_then(|value| usize::try_from(value).ok())
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
    text_stub_gathered: Vec<u32>,
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
                .map(|head| {
                    widen(
                        file,
                        &format!("talker.code_predictor.lm_head.{head}.weight"),
                    )
                })
                .collect(),
            // The text path is unused behind `begin_with_prefill`; minimal well-shaped stubs.
            text_stub_table: vec![0.0; 2],
            text_stub_gathered: vec![0],
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
        self.generator_with_mode(
            talker_layers,
            micro_layers,
            micro_residual,
            micro_heads,
            SamplingMode::CanonicalGreedy,
        )
    }

    fn generator_with_mode<'a>(
        &'a self,
        talker_layers: &'a [TalkerLayerWeights<'a>],
        micro_layers: &'a [LayerWeights<'a>],
        micro_residual: &'a [&'a [f32]],
        micro_heads: &'a [&'a [f32]],
        sampling_mode: SamplingMode,
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
                gathered: &self.text_stub_gathered,
                embed_width: 2,
                fc1_weight: &self.text_stub_fc1,
                fc1_bias: &self.text_stub_fc1_bias,
                fc2_weight: &self.text_stub_fc2,
                fc2_bias: &self.text_stub_fc2_bias,
            },
            cold_rows: None,
            feedback: FeedbackTables {
                talker_codec: &self.talker_codec_embedding,
                residual: FeedbackResidual::Widened(
                    self.residual_embeddings.iter().map(Vec::as_slice).collect(),
                ),
            },
            microdecoder_config: MicrodecoderConfig::default(),
            microdecoder_weights: MicrodecoderWeights {
                layers: micro_layers,
                talker_codec_embedding: &self.talker_codec_embedding,
                // The microdecoder's internal tables for positions 2..=15 are the first 14 of the
                // same per-depth set the feedback path uses.
                residual_embeddings: ftts_model_qwen::microdecoder::ResidualEmbeddings::Widened(
                    micro_residual.to_vec(),
                ),
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
            sampling_mode,
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
    fn begin_utterance(
        &mut self,
        _prepared: &PreparedText,
        _mode: ftts_core::UtteranceStart,
    ) -> Result<(), GenerationError> {
        self.inner
            .begin_with_prefill(&self.prefill, self.seq, self.trailing_text_hidden.clone())
    }

    fn append_text(&mut self, _prepared: &PreparedText) -> Result<(), GenerationError> {
        Err(GenerationError::new(
            "the oracle-prompt fixture does not model text appends",
        ))
    }

    fn finish_text(&mut self) -> Result<(), GenerationError> {
        Err(GenerationError::new(
            "the oracle-prompt fixture does not model text appends",
        ))
    }

    fn next_frame(&mut self) -> Result<FrameStep, GenerationError> {
        self.inner.next_frame()
    }
}

#[test]
fn engine_synthesize_reproduces_the_whole_oracle_utterance_exactly() {
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
    .expect("oracle utterance codes captured");

    // The manifest is the pack's own statement of how long the utterance was; the codes tensor must
    // agree with it, or one of the two is describing a different run.
    let manifest = fixtures
        .mode_manifest(CASE, MODE)
        .expect("per-mode manifest readable");
    let expected_frames = manifest.generated_frames;
    assert!(expected_frames > 0, "the oracle captured no frames");
    assert_eq!(
        expected_codes.shape,
        vec![expected_frames, CODE_GROUP_COUNT],
        "captured codes must be [frames, 16] and agree with the manifest's frame count"
    );

    // Does this capture actually contain a stop? The upstream assembler only truncates at the
    // first EOS *present in the emitted codes* (`is_stop_token` over codebook 0); when no EOS was
    // drawn it keeps every frame it has. So the codes themselves — not the frame count — are the
    // only sound witness that the utterance ended rather than ran out of budget.
    //
    // The frame count cannot stand in for that. Upstream drops the final step's frame (its
    // microdecoder never runs, so `hid[-1] is None`), which makes `generated_frames` one *less*
    // than the steps taken. A pack capped at N steps therefore reports N-1 frames and would look
    // "stopped early" to any count-versus-cap heuristic, EOS or not.
    let captured_eos = expected_codes
        .data
        .chunks(CODE_GROUP_COUNT)
        .any(|frame| frame[0] == i64::from(CODEC_EOS_TOKEN_ID));
    let cap = corpus_max_new_tokens(&fixtures, CASE);
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
    let micro_layers: Vec<LayerWeights<'_>> = hydrated
        .micro_layers
        .iter()
        .map(borrow_micro_layer)
        .collect();
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

    // Bound the run just above the capture rather than leaning on the 8,192-frame default.
    //
    // Neither side necessarily emits EOS (see the module docs), so an unbounded engine would decode
    // thousands of fp32 frames to prove nothing beyond what the capture covers. Two frames of slack
    // is exactly what the assertions need: one to reach the post-capture step whose logits the
    // oracle kept, and one more so the ceiling is never what ended the utterance.
    let frame_ceiling = expected_frames as u64 + 2;
    let engine = TtsEngine::new(EngineConfig {
        // The f32 reference forward is deliberately unoptimized; give it room.
        synthesis_stage_budget: Duration::from_secs(600),
        admission: ftts_core::admission::AdmissionPolicy {
            max_new_tokens: frame_ceiling,
            ..ftts_core::admission::AdmissionPolicy::default()
        },
        ..EngineConfig::default()
    })
    .expect("engine constructs");
    let preparer = FixtureTextPreparer {
        token_ids: prompt_ids.data.iter().map(|&id| id as u32).collect(),
    };
    let observer = CeilingObserver::default();
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
            &observer,
            None,
        )
        .expect("synthesis completes within budget");

    // The ceiling must have had slack, or every count assertion below is vacuous: a loop that ran
    // out of permitted frames is indistinguishable from one that chose to stop.
    let ceiling = observer.predicted_max_frames.load(Ordering::Relaxed);
    assert!(
        ceiling > expected_frames as u64,
        "admitted frame ceiling {ceiling} leaves no slack above the oracle's {expected_frames} \
         frames, so nothing about the stop could be concluded"
    );

    assert_eq!(
        result.code_frames.len() as u64,
        result.generated_frames,
        "generated_frames must agree with the frames actually carried"
    );

    // Compare the overlap FIRST, so a divergence is reported where it happened rather than as a
    // frame-count mismatch. A count difference is usually a *consequence* of codes going wrong
    // several frames earlier, and "emitted 20, expected 255" does not say where.
    let actual: Vec<i64> = result
        .code_frames
        .iter()
        .take(expected_frames)
        .flat_map(|frame| {
            assert_eq!(
                frame.codes.len(),
                CODE_GROUP_COUNT,
                "every frame carries 16 codes"
            );
            frame.codes.iter().map(|&code| i64::from(code))
        })
        .collect();
    if let Some((index, (&ours, &theirs))) = actual
        .iter()
        .zip(expected_codes.data.iter())
        .enumerate()
        .find(|(_, (ours, theirs))| ours != theirs)
    {
        panic!(
            "code divergence at frame {} depth {}: ours {ours}, oracle {theirs} \
             (matched {index} codes = {} full frames before diverging)",
            index / CODE_GROUP_COUNT,
            index % CODE_GROUP_COUNT,
            index / CODE_GROUP_COUNT
        );
    }

    // Only now the length: with the overlap proven identical, a count difference is exactly a stop
    // disagreement and nothing else.
    assert!(
        result.generated_frames >= expected_frames as u64,
        "codes agree over all {} frames the engine produced, but it stopped there while the \
         oracle continued to {expected_frames} frames — the engine ends the utterance early",
        result.generated_frames
    );
    assert_eq!(
        actual, expected_codes.data,
        "whole-utterance codes must match the oracle exactly"
    );

    // The capture's own digest, computed the way the capture computed it.
    let mut hasher = Sha256::new();
    for code in &actual {
        hasher.update(code.to_le_bytes());
    }
    assert_eq!(
        format!("{:x}", hasher.finalize()),
        manifest.generated_codes_sha256,
        "captured-utterance code digest must match the capture's `generated_codes_sha256`"
    );

    // One step further than the codes go. The capture kept the *logits* of the step whose frame it
    // dropped, so the decode loop's feedback path — frame codes summed onto the next input, KV
    // appended, mRoPE advanced — can still be judged, and that path is exactly what frame-0-only
    // parity cannot reach.
    let head_seam = SeamRef {
        case: CASE,
        mode: MODE,
        group: GROUP,
        seam: "talker.codec_head.output",
    };
    if let Ok(next_logits) = fixtures.seam(&head_seam, "tensor", expected_frames) {
        let history: Vec<u32> = expected_codes
            .data
            .chunks(CODE_GROUP_COUNT)
            .map(|frame| frame[0] as u32)
            .collect();
        // The production sampler on the oracle's own logits: this judges our processor stack
        // (repetition penalty, min-new-tokens, suppression) as well as the forward.
        let expected_next = QwenSampler::seeded(0)
            .select_talker(&next_logits.data, &history, SamplingMode::CanonicalGreedy)
            .expect("oracle logits are a well-formed talker row");
        assert!(
            result.generated_frames > expected_frames as u64,
            "the oracle captured logits for step {expected_frames}, so the engine must have \
             reached that step"
        );
        assert_eq!(
            result.code_frames[expected_frames].codes[0], expected_next,
            "primary code at the first post-capture step must match the oracle's own logits; \
             a mismatch here is the decode loop's feedback path, not the prefill"
        );
    }

    // What kind of stop this capture can and cannot witness.
    if captured_eos {
        // The capture really does end at EOS, so the engine must end there too — and stay ended.
        assert_eq!(
            result.generated_frames, expected_frames as u64,
            "the oracle's codes end at EOS after {expected_frames} frames; the engine emitted {}",
            result.generated_frames
        );
        assert!(
            matches!(
                generator
                    .next_frame()
                    .expect("polling a finished generator is not an error"),
                FrameStep::Finished
            ),
            "after EOS the generator must stay finished, not resume"
        );
        Receipt::new(TEST_NAME, Outcome::Passed)
            .contract("ConformanceExact/e2e")
            .seam(SEAM)
            .oracle_tier(OracleTier::CpuFp32Fallback)
            .emit();
    } else {
        // No EOS was ever drawn, so this pack has no stop to be parity with. Upstream's
        // `has_stop_token` is false and it kept every frame it had; the utterance was bounded by
        // the corpus cap. Saying so in the receipt is the point — a green here must not be read
        // as "the engine stops where the reference stops", which remains unproven at this tier.
        Receipt::new(TEST_NAME, Outcome::Passed)
            .contract("ConformanceExact/e2e")
            .seam(SEAM)
            .reason(&*format!(
                "codes parity exact over all {expected_frames} captured frame(s) plus the \
                 post-capture primary; EOS-stop parity NOT covered — this capture drew no EOS \
                 (corpus max_new_tokens {cap:?}), so it contains no stop to compare against"
            ))
            .oracle_tier(OracleTier::CpuFp32Fallback)
            .emit();
    }
}

/// p7r bisect probe: the engine loop under Production sampling, from the oracle's own prefill.
///
/// Prints every frame's 16 codes so the silence-shaped failure can be attributed: degenerate or
/// repeating c0 means the loop/history/logits interaction is wrong; plausible diverse c0 (the
/// family the reference draws under sampling) with silent audio means the greedy-residuals-under-
/// sampled-c0 combination or the codec input is at fault. Run manually:
/// `cargo test -p ftts-conformance --test engine_e2e_ft7 -- --ignored --nocapture`.
#[test]
#[ignore = "manual p7r diagnostic; prints per-frame codes under Production sampling"]
fn production_mode_codes_probe() {
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
        skip("pinned weights absent");
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
    let seq = prefill.shape[1];
    let trailing = fixtures
        .seam(&input_seam, "kwargs.trailing_text_hidden", 0)
        .expect("trailing text stream captured");
    let tts_pad = fixtures
        .seam(&input_seam, "kwargs.tts_pad_embed", 0)
        .expect("tts_pad embedding captured");

    let file = SafetensorsFile::open(&checkpoint).expect("pinned checkpoint maps and parses");
    file.advise_random();
    let hydrated = Hydrated::load(&file, tts_pad.data.clone());
    let talker_layers: Vec<TalkerLayerWeights<'_>> =
        hydrated.talker_layers.iter().map(borrow_layer).collect();
    let micro_layers: Vec<LayerWeights<'_>> = hydrated
        .micro_layers
        .iter()
        .map(borrow_micro_layer)
        .collect();
    let micro_residual: Vec<&[f32]> = hydrated.residual_embeddings[..RESIDUAL_DEPTHS - 1]
        .iter()
        .map(Vec::as_slice)
        .collect();
    let micro_heads: Vec<&[f32]> = hydrated.micro_heads.iter().map(Vec::as_slice).collect();
    let trailing_rows: Vec<Vec<f32>> = trailing.data.chunks(HIDDEN).map(<[f32]>::to_vec).collect();

    let mut generator = hydrated.generator_with_mode(
        &talker_layers,
        &micro_layers,
        &micro_residual,
        &micro_heads,
        SamplingMode::Production,
    );
    generator
        .begin_with_prefill(&prefill.data, seq, trailing_rows)
        .expect("prefill accepted");
    for frame in 0..14 {
        match generator.next_frame().expect("frame generation succeeds") {
            FrameStep::Frame(code_frame) => {
                eprintln!("frame {frame:02}: {:?}", code_frame.codes);
            }
            FrameStep::Finished => {
                eprintln!("frame {frame:02}: EOS drawn — utterance ended");
                break;
            }
            FrameStep::AwaitingText => unreachable!("fresh utterance never awaits text"),
        }
    }
}
