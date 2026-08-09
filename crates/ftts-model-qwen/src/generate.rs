//! `QwenGenerator`: the model side of `ftts_core`'s [`FrameGenerator`] seam.
//!
//! `begin_utterance` runs the cold path once per request — wrapper stripping, cold text embedding,
//! the SiLU text projection, exact prompt assembly, and the full talker prefill — and parks the
//! last position's hidden state and primary logits. `next_frame` then advances one frame at a
//! time: sample `c0`, run the 15-step microdecoder for `c1..=c15`, and feed the 16 code embeddings
//! (plus one consumed trailing-text hidden, or `tts_pad` once the text stream is exhausted) back
//! through a single-position talker step.
//!
//! Everything heavy is borrowed: checkpoint hydration owns the tensors and this type holds `&[f32]`
//! views, so an utterance never clones a weight table.

use ftts_core::{CodeFrame, FrameGenerator, GenerationError, PreparedText};

use crate::microdecoder::{
    self, FrameState, MicroLayerQuant, MicrodecoderConfig, MicrodecoderWeights, RESIDUAL_VOCAB,
    RopeTable,
};
use crate::prompt::{
    self, HiddenState, PromptAssemblyInput, PromptError, PromptHeader, PromptMode,
};
use crate::sampler::{CODEC_EOS_TOKEN_ID, QwenSampler, SamplingMode};
use crate::talker::{
    self, CODE_GROUP_COUNT, PRIMARY_CODE_VOCAB_SIZE, RotaryRows, TalkerConfig, TalkerKvCache,
    TalkerLayerQuant, TalkerWeights,
};
use ftts_artifacts::fttsq::{MappedFttsq, StoredDtype};
use ftts_kernels::int8::{QuantLinearMode, QuantizedMatrix};

/// The talker's pinned mRoPE base. The microdecoder and codec each use a different theta; crossing
/// them is a silent correctness failure, so the constant lives next to its only call sites.
const MROPE_THETA: f32 = 1.0e6;

/// Reads one Q8 tensor out of a canonical artifact as an executable [`QuantizedMatrix`].
///
/// The artifact's payload IS the canonical quantization — the converter and the runtime share one
/// quantizer, pinned byte-identical by a cross-crate test — so consuming it directly skips both
/// the bf16→f32 widen and the f32→int8 requantize the runtime route would otherwise pay at
/// startup. `None` (wrong dtype, missing scales, geometry mismatch) sends the caller to the
/// runtime-quantization fallback rather than failing hydration.
fn q8_from_artifact(
    artifact: &MappedFttsq,
    name: &str,
    n: usize,
    k: usize,
) -> Option<QuantizedMatrix> {
    let entry = artifact.reader().tensor(name)?;
    if entry.dtype != StoredDtype::Q8 {
        return None;
    }
    let scales_name = entry.scales.clone()?;
    let data_bytes = artifact.tensor_bytes(name).ok()?;
    if data_bytes.len() != n.checked_mul(k)? {
        return None;
    }
    let scales_bytes = artifact.tensor_bytes(&scales_name).ok()?;
    if scales_bytes.len() != n.checked_mul(4)? {
        return None;
    }
    // Q8 payload bytes are the two's-complement values themselves; this cast-copy is the whole
    // hydration cost (a memcpy), replacing a widen + max-scan + rounding pass per row.
    let data = data_bytes.iter().map(|&byte| byte as i8).collect();
    let scales = scales_bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|chunk| f32::from_le_bytes(*chunk))
        .collect();
    Some(QuantizedMatrix { data, scales, n, k })
}

/// The seven projections of one attention layer, artifact-native, in the same fused shape
/// [`TalkerLayerQuant::quantize`] produces. `None` if any tensor is absent, letting mixed or
/// older artifacts fall back to runtime quantization for the whole layer.
fn fused_layer_from_artifact(
    artifact: &MappedFttsq,
    base: &str,
    hidden: usize,
    q_width: usize,
    kv_width: usize,
    intermediate: usize,
) -> Option<(
    QuantizedMatrix,
    QuantizedMatrix,
    QuantizedMatrix,
    QuantizedMatrix,
)> {
    let q = q8_from_artifact(
        artifact,
        &format!("{base}.self_attn.q_proj.weight"),
        q_width,
        hidden,
    )?;
    let k = q8_from_artifact(
        artifact,
        &format!("{base}.self_attn.k_proj.weight"),
        kv_width,
        hidden,
    )?;
    let v = q8_from_artifact(
        artifact,
        &format!("{base}.self_attn.v_proj.weight"),
        kv_width,
        hidden,
    )?;
    let o = q8_from_artifact(
        artifact,
        &format!("{base}.self_attn.o_proj.weight"),
        hidden,
        q_width,
    )?;
    let gate = q8_from_artifact(
        artifact,
        &format!("{base}.mlp.gate_proj.weight"),
        intermediate,
        hidden,
    )?;
    let up = q8_from_artifact(
        artifact,
        &format!("{base}.mlp.up_proj.weight"),
        intermediate,
        hidden,
    )?;
    let down = q8_from_artifact(
        artifact,
        &format!("{base}.mlp.down_proj.weight"),
        hidden,
        intermediate,
    )?;
    Some((
        QuantizedMatrix::concat_rows(&[&q, &k, &v]),
        o,
        QuantizedMatrix::concat_rows(&[&gate, &up]),
        down,
    ))
}

/// Every talker layer's fused int8 tables read artifact-natively, or `None` if any tensor is
/// missing. Public for the hydration parity gate (`examples/artifact_q8_hydration.rs`).
#[must_use]
pub fn talker_layers_from_artifact(
    artifact: &MappedFttsq,
    config: &TalkerConfig,
    layer_count: usize,
) -> Option<Vec<TalkerLayerQuant>> {
    (0..layer_count)
        .map(|index| {
            fused_layer_from_artifact(
                artifact,
                &format!("talker.model.layers.{index}"),
                config.hidden_size,
                config.query_width(),
                config.kv_width(),
                config.intermediate_size,
            )
            .map(|(qkv, o_proj, gate_up, down_proj)| TalkerLayerQuant {
                qkv,
                o_proj,
                gate_up,
                down_proj,
            })
        })
        .collect()
}

/// Every microdecoder layer's fused int8 tables read artifact-natively, or `None` if any tensor
/// is missing. Public for the hydration parity gate.
#[must_use]
pub fn micro_layers_from_artifact(
    artifact: &MappedFttsq,
    config: &MicrodecoderConfig,
    layer_count: usize,
) -> Option<Vec<MicroLayerQuant>> {
    (0..layer_count)
        .map(|index| {
            fused_layer_from_artifact(
                artifact,
                &format!("talker.code_predictor.model.layers.{index}"),
                config.hidden_size,
                config.q_width(),
                config.kv_width(),
                config.intermediate_size,
            )
            .map(|(qkv, o_proj, gate_up, down_proj)| MicroLayerQuant {
                qkv,
                o_proj,
                gate_up,
                down_proj,
            })
        })
        .collect()
}

/// `FTTS_ARTIFACT_Q8=0` forces the widen-then-requantize hydration even when a canonical
/// artifact is available — the A/B and forensics switch for artifact-native hydration.
fn artifact_q8_enabled() -> bool {
    !matches!(
        std::env::var("FTTS_ARTIFACT_Q8").as_deref(),
        Ok("0") | Ok("off") | Ok("false")
    )
}

/// The cold text embedding table and the learned biased SiLU text projection.
///
/// Production widths are `(embed_width, intermediate, hidden) = (2048, 2048, 1024)`; the fields
/// stay explicit so tiny-geometry tests exercise the same code path.
#[derive(Clone, Copy, Debug)]
pub struct TextEmbeddingWeights<'a> {
    /// Row-major `[vocab, embed_width]` cold text embedding. Only requested rows are read.
    pub table: &'a [f32],
    /// Width of one embedding row.
    pub embed_width: usize,
    /// Projection fc1 weight, `[intermediate, embed_width]`.
    pub fc1_weight: &'a [f32],
    /// Projection fc1 bias, `[intermediate]`.
    pub fc1_bias: &'a [f32],
    /// Projection fc2 weight, `[hidden, intermediate]`.
    pub fc2_weight: &'a [f32],
    /// Projection fc2 bias, `[hidden]`.
    pub fc2_bias: &'a [f32],
}

/// The 16 feedback embedding tables summed into every next-frame talker input.
///
/// Table 0 is the talker's own codec embedding (`[3072, hidden]`); tables 1..=15 are the residual
/// feedback tables (`[2048, hidden]` each). These are talker-input tables, distinct from the
/// microdecoder's internal per-depth embeddings.
#[derive(Clone, Debug)]
pub struct FeedbackTables<'a> {
    /// `[PRIMARY_CODE_VOCAB_SIZE, hidden]`, indexed by `c0`.
    pub talker_codec: &'a [f32],
    /// Fifteen `[RESIDUAL_VOCAB, hidden]` tables, indexed by `c1..=c15` in depth order.
    pub residual: Vec<&'a [f32]>,
}

/// Reference-voice conditioning for ICL clone mode.
#[derive(Clone, Debug)]
pub struct ReferencePrompt {
    /// The reference transcript with its official assistant wrapper still attached.
    pub wrapped_ids: Vec<u32>,
    /// Per-frame sums of the sixteen reference-code embeddings, already computed.
    pub codec: Vec<HiddenState>,
}

/// Everything a [`QwenGenerator`] borrows or owns for its lifetime.
#[derive(Clone, Debug)]
pub struct QwenGeneratorConfig<'a> {
    pub talker_config: TalkerConfig,
    pub talker_weights: TalkerWeights<'a>,
    pub text: TextEmbeddingWeights<'a>,
    pub feedback: FeedbackTables<'a>,
    pub microdecoder_config: MicrodecoderConfig,
    pub microdecoder_weights: MicrodecoderWeights<'a>,
    pub prompt_mode: PromptMode,
    pub header: PromptHeader,
    pub tts_eos: HiddenState,
    /// Required for ICL clone mode, ignored for x-vector.
    pub reference: Option<ReferencePrompt>,
    pub sampling_mode: SamplingMode,
    /// Seed for the production sampler; canonical greedy never consumes RNG state.
    pub seed: u64,
}

/// Per-utterance autoregressive state, dropped and rebuilt by every `begin_utterance`.
#[derive(Debug, PartialEq)]
struct UtteranceState {
    /// Final-norm hidden of the newest talker position, `[hidden]`.
    pending_hidden: Vec<f32>,
    /// Primary-code logits of the newest talker position, `[3072]`.
    pending_logits: Vec<f32>,
    /// One projected text hidden consumed per generated frame; `tts_pad` afterwards.
    trailing_text_hidden: Vec<HiddenState>,
    next_position: usize,
    frames_emitted: usize,
    /// Group-0 codes only — residuals never enter the repetition-penalty history.
    group_zero_history: Vec<u32>,
    finished: bool,
}

/// The armed W8A8 int8 route: quantized projection tables for both stacks plus the dot tier.
///
/// Built once at construction when the `FTTS_INT8=1` kill-switch is armed (staged levers 2a+2b);
/// `None` leaves the f32 reference path structurally untouched. Norms, QK-Norm, rotary,
/// attention, per-depth heads/embeddings, the final norms, and the primary head remain f32 in
/// both stacks per the fixed doctrine recipe.
#[derive(Debug)]
struct Int8Route {
    talker: Vec<TalkerLayerQuant>,
    micro: Vec<MicroLayerQuant>,
    /// Quantized per-depth heads for the int8+f32-refine scoring path (empty when the
    /// microdecoder stack is not armed).
    micro_heads: Vec<ftts_kernels::int8::QuantizedMatrix>,
    mode: QuantLinearMode,
}

/// The Qwen3-TTS Base implementation of [`FrameGenerator`].
#[derive(Debug)]
pub struct QwenGenerator<'a> {
    talker_config: TalkerConfig,
    talker_weights: TalkerWeights<'a>,
    text: TextEmbeddingWeights<'a>,
    feedback: FeedbackTables<'a>,
    microdecoder_config: MicrodecoderConfig,
    microdecoder_weights: MicrodecoderWeights<'a>,
    microdecoder_rope: RopeTable,
    prompt_mode: PromptMode,
    header: PromptHeader,
    tts_eos: HiddenState,
    reference: Option<ReferencePrompt>,
    sampler: QwenSampler,
    sampling_mode: SamplingMode,
    kv: TalkerKvCache,
    frame_state: FrameState,
    utterance: Option<UtteranceState>,
    int8: Option<Int8Route>,
}

impl<'a> QwenGenerator<'a> {
    /// Builds a generator over borrowed weights.
    ///
    /// # Panics
    ///
    /// Panics when the feedback tables or text projection disagree with the talker geometry —
    /// these are checkpoint-hydration bugs, not runtime conditions.
    #[must_use]
    pub fn new(config: QwenGeneratorConfig<'a>) -> Self {
        Self::new_with_artifact(config, None)
    }

    /// [`QwenGenerator::new`], additionally offered the mapped canonical artifact the checkpoint
    /// was hydrated from.
    ///
    /// When the int8 route is armed and the artifact carries a layer's Q8 tensors, that layer's
    /// projections hydrate directly from the artifact payload — the canonical quantization
    /// itself — instead of re-quantizing widened f32 copies. Payload bytes are identical to the
    /// runtime quantizer's output by the shared-quantizer contract; scales are the converter's
    /// own (the requantize round trip can drift a scale by an ulp, so the artifact is the more
    /// canonical of the two). `FTTS_ARTIFACT_Q8=0` forces the old path for A/B.
    #[must_use]
    pub fn new_with_artifact(
        config: QwenGeneratorConfig<'a>,
        artifact: Option<&MappedFttsq>,
    ) -> Self {
        let hidden = config.talker_config.hidden_size;
        assert_eq!(
            config.feedback.talker_codec.len(),
            PRIMARY_CODE_VOCAB_SIZE * hidden,
            "feedback table 0 must be [{PRIMARY_CODE_VOCAB_SIZE}, hidden]"
        );
        assert_eq!(
            config.feedback.residual.len(),
            CODE_GROUP_COUNT - 1,
            "expected {} residual feedback tables",
            CODE_GROUP_COUNT - 1
        );
        for table in &config.feedback.residual {
            assert_eq!(
                table.len(),
                RESIDUAL_VOCAB * hidden,
                "each residual feedback table must be [{RESIDUAL_VOCAB}, hidden]"
            );
        }
        assert_eq!(
            config.text.fc2_bias.len(),
            hidden,
            "text projection must land on the talker width"
        );
        assert!(
            config.text.embed_width > 0
                && config
                    .text
                    .table
                    .len()
                    .is_multiple_of(config.text.embed_width),
            "text embedding table must be [vocab, embed_width]"
        );
        assert_eq!(config.header.tts_pad.len(), hidden, "header width");
        assert_eq!(config.tts_eos.len(), hidden, "tts_eos width");
        assert_eq!(
            config.microdecoder_config.hidden_size, hidden,
            "the microdecoder conditions on talker hiddens, so widths must agree"
        );

        let microdecoder_rope = RopeTable::new(&config.microdecoder_config);
        let frame_state = FrameState::new(&config.microdecoder_config);

        // Staged levers 2a+2b: runtime W8A8, armed only by the explicit kill-switch. The default
        // path quantizes nothing and calls the untouched f32 reference functions.
        // The optimized route is the library default; `FTTS_INT8=0` or a conformance
        // reference-pin selects the f32 reference instead (ftts_kernels::route).
        let int8 = ftts_kernels::route::optimized_default("FTTS_INT8").then(|| {
            // FTTS_INT8_SCOPE narrows the lever for sensitivity attribution: `talker` or
            // `micro` quantizes one stack and leaves the other on the f32 reference. An empty
            // table below means "this stack stays f32" at the branch sites.
            let scope = std::env::var("FTTS_INT8_SCOPE").unwrap_or_default();
            let (arm_talker, arm_micro) = match scope.as_str() {
                "talker" => (true, false),
                "micro" => (false, true),
                _ => (true, true),
            };
            let artifact = artifact.filter(|_| artifact_q8_enabled());
            Int8Route {
                talker: if arm_talker {
                    config
                        .talker_weights
                        .layers
                        .iter()
                        .enumerate()
                        .map(|(index, layer)| {
                            artifact
                                .and_then(|artifact| {
                                    let talker = &config.talker_config;
                                    fused_layer_from_artifact(
                                        artifact,
                                        &format!("talker.model.layers.{index}"),
                                        talker.hidden_size,
                                        talker.query_width(),
                                        talker.kv_width(),
                                        talker.intermediate_size,
                                    )
                                })
                                .map(|(qkv, o_proj, gate_up, down_proj)| TalkerLayerQuant {
                                    qkv,
                                    o_proj,
                                    gate_up,
                                    down_proj,
                                })
                                .unwrap_or_else(|| {
                                    TalkerLayerQuant::quantize(&config.talker_config, layer)
                                })
                        })
                        .collect()
                } else {
                    Vec::new()
                },
                micro: if arm_micro {
                    config
                        .microdecoder_weights
                        .layers
                        .iter()
                        .enumerate()
                        .map(|(index, layer)| {
                            artifact
                                .and_then(|artifact| {
                                    let micro = &config.microdecoder_config;
                                    fused_layer_from_artifact(
                                        artifact,
                                        &format!("talker.code_predictor.model.layers.{index}"),
                                        micro.hidden_size,
                                        micro.q_width(),
                                        micro.kv_width(),
                                        micro.intermediate_size,
                                    )
                                })
                                .map(|(qkv, o_proj, gate_up, down_proj)| MicroLayerQuant {
                                    qkv,
                                    o_proj,
                                    gate_up,
                                    down_proj,
                                })
                                .unwrap_or_else(|| {
                                    MicroLayerQuant::quantize(&config.microdecoder_config, layer)
                                })
                        })
                        .collect()
                } else {
                    Vec::new()
                },
                micro_heads: if arm_micro {
                    config
                        .microdecoder_weights
                        .heads
                        .iter()
                        .map(|head| {
                            ftts_kernels::int8::QuantizedMatrix::quantize(
                                head,
                                RESIDUAL_VOCAB,
                                config.microdecoder_config.hidden_size,
                            )
                        })
                        .collect()
                } else {
                    Vec::new()
                },
                mode: ftts_kernels::int8::quant_mode_from_environment(),
            }
        });

        Self {
            talker_config: config.talker_config,
            talker_weights: config.talker_weights,
            text: config.text,
            feedback: config.feedback,
            microdecoder_config: config.microdecoder_config,
            microdecoder_weights: config.microdecoder_weights,
            microdecoder_rope,
            prompt_mode: config.prompt_mode,
            header: config.header,
            tts_eos: config.tts_eos,
            reference: config.reference,
            sampler: QwenSampler::seeded(config.seed),
            sampling_mode: config.sampling_mode,
            kv: TalkerKvCache::new(),
            frame_state,
            utterance: None,
            int8,
        }
    }

    /// Talker positions currently cached, exposed for geometry tests.
    #[must_use]
    pub fn cached_positions(&self) -> usize {
        self.kv.len()
    }

    /// Embeds token ids through the cold table and projects them to talker width.
    fn project_text_ids(&self, ids: &[u32]) -> Result<Vec<HiddenState>, GenerationError> {
        let embed_width = self.text.embed_width;
        let vocab = self.text.table.len() / embed_width;
        let mut embedded = Vec::with_capacity(ids.len() * embed_width);
        for &id in ids {
            let row = id as usize;
            if row >= vocab {
                return Err(GenerationError::new(format!(
                    "text token {id} is outside the [{vocab}, {embed_width}] embedding table"
                )));
            }
            embedded
                .extend_from_slice(&self.text.table[row * embed_width..(row + 1) * embed_width]);
        }

        let hidden = self.talker_config.hidden_size;
        let intermediate = self.text.fc1_bias.len();
        let mut projected = vec![0.0_f32; ids.len() * hidden];
        if !ids.is_empty() {
            talker::project_text_rows(
                &embedded,
                ids.len(),
                embed_width,
                intermediate,
                hidden,
                self.text.fc1_weight,
                self.text.fc1_bias,
                self.text.fc2_weight,
                self.text.fc2_bias,
                &mut projected,
            );
        }
        Ok(projected.chunks(hidden).map(<[f32]>::to_vec).collect())
    }

    /// Begins an utterance from an externally assembled prefill.
    ///
    /// This is the conformance seam for the exact ladder: the oracle fixtures capture the fully
    /// assembled `[seq, hidden]` talker input (`kwargs.inputs_embeds`) and the trailing text
    /// stream, so parity tests can drive the real prefill/decode path from the oracle's own
    /// prompt without re-deriving header construction. `prefill` is row-major `[seq, hidden]`,
    /// consumed at positions `0..seq`. Production callers use [`FrameGenerator::begin_utterance`].
    pub fn begin_with_prefill(
        &mut self,
        prefill: &[f32],
        seq: usize,
        trailing_text_hidden: Vec<HiddenState>,
    ) -> Result<(), GenerationError> {
        let hidden_size = self.talker_config.hidden_size;
        if seq == 0 || prefill.len() != seq * hidden_size {
            return Err(GenerationError::new(format!(
                "prefill must be a non-empty [seq, {hidden_size}] buffer; got {} values for seq {seq}",
                prefill.len()
            )));
        }
        self.run_prefill(prefill.to_vec(), seq, trailing_text_hidden);
        Ok(())
    }

    /// Runs the talker over an assembled prefill and parks the newest position's state.
    fn run_prefill(&mut self, mut hidden: Vec<f32>, seq: usize, trailing: Vec<HiddenState>) {
        self.utterance = None;
        self.kv.clear();
        let hidden_size = self.talker_config.hidden_size;
        let positions: Vec<i64> = (0..seq as i64).collect();
        let (cos, sin) = talker::mrope_rows(&positions, self.talker_config.head_dim, MROPE_THETA);
        let mask = causal_mask(seq);
        let mut logits = vec![0.0_f32; seq * PRIMARY_CODE_VOCAB_SIZE];
        let rotary = RotaryRows {
            cos: &cos,
            sin: &sin,
        };
        match self.int8.as_ref().filter(|route| !route.talker.is_empty()) {
            Some(route) => talker::forward_talker_q8(
                &self.talker_config,
                &self.talker_weights,
                &route.talker,
                rotary,
                &mask,
                &mut hidden,
                seq,
                &mut self.kv,
                &mut logits,
                route.mode,
            ),
            None => talker::forward_talker(
                &self.talker_config,
                &self.talker_weights,
                rotary,
                &mask,
                &mut hidden,
                seq,
                &mut self.kv,
                &mut logits,
            ),
        }
        self.utterance = Some(UtteranceState {
            pending_hidden: hidden[(seq - 1) * hidden_size..].to_vec(),
            pending_logits: logits[(seq - 1) * PRIMARY_CODE_VOCAB_SIZE..].to_vec(),
            trailing_text_hidden: trailing,
            next_position: seq,
            frames_emitted: 0,
            group_zero_history: Vec::new(),
            finished: false,
        });
    }
}

fn generation_error(error: impl std::fmt::Display) -> GenerationError {
    GenerationError::new(error.to_string())
}

/// The additive causal mask for a `seq`-position prefill over an empty cache.
fn causal_mask(seq: usize) -> Vec<f32> {
    let mut mask = vec![0.0_f32; seq * seq];
    for query in 0..seq {
        for key in query + 1..seq {
            mask[query * seq + key] = f32::NEG_INFINITY;
        }
    }
    mask
}

impl FrameGenerator for QwenGenerator<'_> {
    fn begin_utterance(&mut self, prepared: &PreparedText) -> Result<(), GenerationError> {
        self.utterance = None;

        let ids = prompt::extract_prompt_text_ids(
            &prepared.token_ids,
            self.reference.as_ref().map(|r| r.wrapped_ids.as_slice()),
        )
        .map_err(generation_error)?;
        let target_text = self.project_text_ids(&ids.target)?;
        let reference_text = match &ids.reference {
            Some(reference_ids) => Some(self.project_text_ids(reference_ids)?),
            None => None,
        };

        let assembly = prompt::assemble_prompt(PromptAssemblyInput {
            mode: self.prompt_mode,
            header: self.header.clone(),
            target_text,
            reference_text,
            reference_codec: self.reference.as_ref().map(|r| r.codec.clone()),
            tts_eos: self.tts_eos.clone(),
        })
        .map_err(generation_error)?;

        let hidden_size = self.talker_config.hidden_size;
        let seq = assembly.prefill.len();
        if seq == 0 {
            return Err(GenerationError::new("assembled prompt prefill is empty"));
        }
        let mut hidden = Vec::with_capacity(seq * hidden_size);
        for state in &assembly.prefill {
            if state.len() != hidden_size {
                return Err(generation_error(PromptError::WidthMismatch {
                    expected: hidden_size,
                    actual: state.len(),
                }));
            }
            hidden.extend_from_slice(state);
        }

        self.run_prefill(hidden, seq, assembly.trailing_text_hidden);
        Ok(())
    }

    fn next_frame(&mut self) -> Result<Option<CodeFrame>, GenerationError> {
        let Some(utterance) = self.utterance.as_mut() else {
            return Err(GenerationError::new(
                "next_frame called before begin_utterance",
            ));
        };
        if utterance.finished {
            return Ok(None);
        }

        let primary = self
            .sampler
            .select_talker(
                &utterance.pending_logits,
                &utterance.group_zero_history,
                self.sampling_mode,
            )
            .map_err(generation_error)?;
        if primary == CODEC_EOS_TOKEN_ID {
            utterance.finished = true;
            return Ok(None);
        }

        // The subtalker follows the run's decode mode: greedy conformance stays zero-RNG, while
        // production SAMPLES every residual depth exactly as upstream's subtalker_dosample=true
        // does — greedy residuals under a sampled talker sit in a measured silence attractor
        // (frankentts-p7r; the reference reproduces it in that mismatched configuration).
        let sampler = &mut self.sampler;
        let sampling_mode = self.sampling_mode;
        let select = |logits: &[f32]| match sampling_mode {
            SamplingMode::CanonicalGreedy => microdecoder::argmax(logits),
            SamplingMode::Production => sampler
                .select_microdecoder(logits, SamplingMode::Production)
                .expect("RESIDUAL_VOCAB rows are well-formed microdecoder logits")
                as usize,
        };
        let residuals = match self.int8.as_ref().filter(|route| !route.micro.is_empty()) {
            Some(route) => microdecoder::decode_frame_with_selector_q8(
                &self.microdecoder_config,
                &self.microdecoder_rope,
                &self.microdecoder_weights,
                &microdecoder::MicroQuantRoute {
                    layers: &route.micro,
                    heads: (!route.micro_heads.is_empty()).then_some(route.micro_heads.as_slice()),
                    mode: route.mode,
                },
                &mut self.frame_state,
                &utterance.pending_hidden,
                primary as usize,
                select,
            ),
            None => microdecoder::decode_frame_with_selector(
                &self.microdecoder_config,
                &self.microdecoder_rope,
                &self.microdecoder_weights,
                &mut self.frame_state,
                &utterance.pending_hidden,
                primary as usize,
                select,
            ),
        };
        let mut codes = Vec::with_capacity(CODE_GROUP_COUNT);
        codes.push(primary);
        codes.extend(residuals.iter().map(|&code| code as u32));

        // Feedback: sum the 16 code embeddings onto one trailing-text hidden (or tts_pad).
        let mut rows: Vec<&[f32]> = Vec::with_capacity(CODE_GROUP_COUNT);
        for (depth, &code) in codes.iter().enumerate() {
            let table = if depth == 0 {
                self.feedback.talker_codec
            } else {
                self.feedback.residual[depth - 1]
            };
            let hidden = self.talker_config.hidden_size;
            rows.push(&table[code as usize * hidden..(code as usize + 1) * hidden]);
        }
        let text_row = utterance
            .trailing_text_hidden
            .get(utterance.frames_emitted)
            .map(Vec::as_slice);
        let mut next_input = vec![0.0_f32; self.talker_config.hidden_size];
        talker::form_frame_input(&rows, text_row, &self.header.tts_pad, &mut next_input);

        let (cos, sin) = talker::mrope_rows(
            &[utterance.next_position as i64],
            self.talker_config.head_dim,
            MROPE_THETA,
        );
        let mask = vec![0.0_f32; self.kv.len() + 1];
        let mut logits = vec![0.0_f32; PRIMARY_CODE_VOCAB_SIZE];
        let rotary = RotaryRows {
            cos: &cos,
            sin: &sin,
        };
        match self.int8.as_ref().filter(|route| !route.talker.is_empty()) {
            Some(route) => talker::forward_talker_q8(
                &self.talker_config,
                &self.talker_weights,
                &route.talker,
                rotary,
                &mask,
                &mut next_input,
                1,
                &mut self.kv,
                &mut logits,
                route.mode,
            ),
            None => talker::forward_talker(
                &self.talker_config,
                &self.talker_weights,
                rotary,
                &mask,
                &mut next_input,
                1,
                &mut self.kv,
                &mut logits,
            ),
        }

        utterance.pending_hidden = next_input;
        utterance.pending_logits = logits;
        utterance.next_position += 1;
        utterance.frames_emitted += 1;
        utterance.group_zero_history.push(primary);
        Ok(Some(CodeFrame { codes }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::microdecoder::{LayerWeights, RESIDUAL_DEPTHS};
    use crate::prompt::CloneMode;
    use crate::talker::{TALKER_LAYER_COUNT, TalkerLayerWeights};
    use ftts_core::{NormalizationMode, NormalizationTrace};

    /// Tiny but structurally complete geometry: talker width 8 with the mandatory 28 layers and
    /// the hardcoded 3072/2048 vocabularies, microdecoder width 8 with 2 layers.
    const HIDDEN: usize = 8;

    fn talker_config() -> TalkerConfig {
        TalkerConfig {
            hidden_size: HIDDEN,
            intermediate_size: 4,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            head_dim: 4,
            rms_norm_eps: 1e-6,
            qk_norm_eps: 1e-6,
        }
    }

    fn microdecoder_config() -> MicrodecoderConfig {
        MicrodecoderConfig {
            hidden_size: HIDDEN,
            num_layers: 2,
            num_q_heads: 4,
            num_kv_heads: 2,
            head_dim: 4,
            intermediate_size: 16,
            rope_theta: 1.0e6,
            rms_eps: 1e-6,
        }
    }

    /// Owns every tensor the generator borrows. Zeroed projections make each transformer layer an
    /// identity over the residual stream, so the test controls outputs through the heads alone.
    struct TinyWeights {
        norm: Vec<f32>,
        talker_q: Vec<f32>,
        talker_kv: Vec<f32>,
        head_norm: Vec<f32>,
        talker_o: Vec<f32>,
        talker_mlp: Vec<f32>,
        talker_down: Vec<f32>,
        codec_head: Vec<f32>,
        micro_q: Vec<f32>,
        micro_kv: Vec<f32>,
        micro_o: Vec<f32>,
        micro_mlp: Vec<f32>,
        micro_down: Vec<f32>,
        talker_codec_embedding: Vec<f32>,
        residual_feedback: Vec<Vec<f32>>,
        micro_residual_embeddings: Vec<Vec<f32>>,
        micro_heads: Vec<Vec<f32>>,
        text_table: Vec<f32>,
        text_fc1: Vec<f32>,
        text_fc1_bias: Vec<f32>,
        text_fc2: Vec<f32>,
        text_fc2_bias: Vec<f32>,
    }

    impl TinyWeights {
        fn new(eos_logit: f32) -> Box<Self> {
            let talker = talker_config();
            let micro = microdecoder_config();
            let mut codec_head = vec![0.0_f32; PRIMARY_CODE_VOCAB_SIZE * HIDDEN];
            if eos_logit != 0.0 {
                let row = CODEC_EOS_TOKEN_ID as usize * HIDDEN;
                codec_head[row..row + HIDDEN].fill(eos_logit);
            }
            Box::new(Self {
                norm: vec![1.0; HIDDEN],
                talker_q: vec![0.0; talker.query_width() * HIDDEN],
                talker_kv: vec![0.0; talker.kv_width() * HIDDEN],
                head_norm: vec![1.0; talker.head_dim],
                talker_o: vec![0.0; HIDDEN * talker.query_width()],
                talker_mlp: vec![0.0; talker.intermediate_size * HIDDEN],
                talker_down: vec![0.0; HIDDEN * talker.intermediate_size],
                codec_head,
                micro_q: vec![0.0; micro.q_width() * HIDDEN],
                micro_kv: vec![0.0; micro.kv_width() * HIDDEN],
                micro_o: vec![0.0; HIDDEN * micro.q_width()],
                micro_mlp: vec![0.0; micro.intermediate_size * HIDDEN],
                micro_down: vec![0.0; HIDDEN * micro.intermediate_size],
                talker_codec_embedding: vec![0.0; PRIMARY_CODE_VOCAB_SIZE * HIDDEN],
                residual_feedback: vec![vec![0.0; RESIDUAL_VOCAB * HIDDEN]; CODE_GROUP_COUNT - 1],
                micro_residual_embeddings: vec![
                    vec![0.0; RESIDUAL_VOCAB * HIDDEN];
                    RESIDUAL_DEPTHS - 1
                ],
                micro_heads: vec![vec![0.0; RESIDUAL_VOCAB * HIDDEN]; RESIDUAL_DEPTHS],
                text_table: (0..4 * 4).map(|v| 0.01 * v as f32).collect(),
                text_fc1: vec![0.1; 4 * 4],
                text_fc1_bias: vec![0.0; 4],
                text_fc2: vec![0.1; HIDDEN * 4],
                text_fc2_bias: vec![0.5; HIDDEN],
            })
        }

        fn talker_layer(&self) -> TalkerLayerWeights<'_> {
            TalkerLayerWeights {
                input_layernorm: &self.norm,
                q_proj: &self.talker_q,
                k_proj: &self.talker_kv,
                v_proj: &self.talker_kv,
                q_norm: &self.head_norm,
                k_norm: &self.head_norm,
                o_proj: &self.talker_o,
                post_attention_layernorm: &self.norm,
                gate_proj: &self.talker_mlp,
                up_proj: &self.talker_mlp,
                down_proj: &self.talker_down,
            }
        }

        fn micro_layer(&self) -> LayerWeights<'_> {
            LayerWeights {
                input_norm: &self.norm,
                q_proj: &self.micro_q,
                k_proj: &self.micro_kv,
                v_proj: &self.micro_kv,
                q_norm: &self.head_norm,
                k_norm: &self.head_norm,
                o_proj: &self.micro_o,
                post_attention_norm: &self.norm,
                gate_proj: &self.micro_mlp,
                up_proj: &self.micro_mlp,
                down_proj: &self.micro_down,
            }
        }
    }

    fn header() -> PromptHeader {
        PromptHeader {
            role: vec![vec![0.5; HIDDEN]; 3],
            codec_prefill: vec![vec![0.25; HIDDEN]; 3],
            tts_bos: vec![0.5; HIDDEN],
            tts_pad: vec![0.5; HIDDEN],
        }
    }

    fn generator<'w>(
        weights: &'w TinyWeights,
        micro_layers: &'w [LayerWeights<'w>],
        micro_residual: &'w [&'w [f32]],
        micro_heads: &'w [&'w [f32]],
        residual_feedback: Vec<&'w [f32]>,
        seed: u64,
        mode: SamplingMode,
    ) -> QwenGenerator<'w> {
        QwenGenerator::new(QwenGeneratorConfig {
            talker_config: talker_config(),
            talker_weights: TalkerWeights {
                layers: vec![weights.talker_layer(); TALKER_LAYER_COUNT],
                final_norm: &weights.norm,
                codec_head: &weights.codec_head,
            },
            text: TextEmbeddingWeights {
                table: &weights.text_table,
                embed_width: 4,
                fc1_weight: &weights.text_fc1,
                fc1_bias: &weights.text_fc1_bias,
                fc2_weight: &weights.text_fc2,
                fc2_bias: &weights.text_fc2_bias,
            },
            feedback: FeedbackTables {
                talker_codec: &weights.talker_codec_embedding,
                residual: residual_feedback,
            },
            microdecoder_config: microdecoder_config(),
            microdecoder_weights: MicrodecoderWeights {
                layers: micro_layers,
                talker_codec_embedding: &weights.talker_codec_embedding,
                residual_embeddings: micro_residual,
                heads: micro_heads,
                final_norm: &weights.norm,
            },
            prompt_mode: PromptMode {
                clone_mode: CloneMode::XVector,
                non_streaming_mode: false,
            },
            header: header(),
            tts_eos: vec![0.5; HIDDEN],
            reference: None,
            sampling_mode: mode,
            seed,
        })
    }

    fn prepared(body: &[u32]) -> PreparedText {
        let mut ids = vec![151_644, 77_091, 198];
        ids.extend_from_slice(body);
        ids.extend_from_slice(&[151_645, 198, 151_644, 77_091, 198]);
        PreparedText::new(
            ids,
            NormalizationTrace {
                mode: NormalizationMode::Verbatim,
                unicode_version: "16.0".to_owned(),
                changes: Vec::new(),
            },
        )
    }

    fn drive(
        weights: &TinyWeights,
        seed: u64,
        mode: SamplingMode,
        max_frames: usize,
    ) -> Vec<CodeFrame> {
        let micro_layers = vec![weights.micro_layer(); 2];
        let micro_residual: Vec<&[f32]> = weights
            .micro_residual_embeddings
            .iter()
            .map(Vec::as_slice)
            .collect();
        let micro_heads: Vec<&[f32]> = weights.micro_heads.iter().map(Vec::as_slice).collect();
        let residual_feedback: Vec<&[f32]> = weights
            .residual_feedback
            .iter()
            .map(Vec::as_slice)
            .collect();
        let mut generator = generator(
            weights,
            &micro_layers,
            &micro_residual,
            &micro_heads,
            residual_feedback,
            seed,
            mode,
        );
        generator
            .begin_utterance(&prepared(&[1, 2]))
            .expect("valid tiny prompt");
        let mut frames = Vec::new();
        for _ in 0..max_frames {
            match generator.next_frame().expect("frame generation succeeds") {
                Some(frame) => frames.push(frame),
                None => break,
            }
        }
        frames
    }

    #[test]
    fn next_frame_before_begin_utterance_is_an_error_not_a_panic() {
        let weights = TinyWeights::new(0.0);
        let micro_layers = vec![weights.micro_layer(); 2];
        let micro_residual: Vec<&[f32]> = weights
            .micro_residual_embeddings
            .iter()
            .map(Vec::as_slice)
            .collect();
        let micro_heads: Vec<&[f32]> = weights.micro_heads.iter().map(Vec::as_slice).collect();
        let residual_feedback: Vec<&[f32]> = weights
            .residual_feedback
            .iter()
            .map(Vec::as_slice)
            .collect();
        let mut generator = generator(
            &weights,
            &micro_layers,
            &micro_residual,
            &micro_heads,
            residual_feedback,
            1,
            SamplingMode::CanonicalGreedy,
        );
        assert!(generator.next_frame().is_err());
    }

    #[test]
    fn malformed_wrapper_surfaces_as_a_generation_error() {
        let weights = TinyWeights::new(0.0);
        let micro_layers = vec![weights.micro_layer(); 2];
        let micro_residual: Vec<&[f32]> = weights
            .micro_residual_embeddings
            .iter()
            .map(Vec::as_slice)
            .collect();
        let micro_heads: Vec<&[f32]> = weights.micro_heads.iter().map(Vec::as_slice).collect();
        let residual_feedback: Vec<&[f32]> = weights
            .residual_feedback
            .iter()
            .map(Vec::as_slice)
            .collect();
        let mut generator = generator(
            &weights,
            &micro_layers,
            &micro_residual,
            &micro_heads,
            residual_feedback,
            1,
            SamplingMode::CanonicalGreedy,
        );
        let bad = PreparedText::new(
            vec![1, 2, 3],
            NormalizationTrace {
                mode: NormalizationMode::Verbatim,
                unicode_version: "16.0".to_owned(),
                changes: Vec::new(),
            },
        );
        assert!(generator.begin_utterance(&bad).is_err());
    }

    #[test]
    fn frames_have_sixteen_codes_and_the_kv_cache_grows_one_position_per_frame() {
        let weights = TinyWeights::new(0.0);
        let micro_layers = vec![weights.micro_layer(); 2];
        let micro_residual: Vec<&[f32]> = weights
            .micro_residual_embeddings
            .iter()
            .map(Vec::as_slice)
            .collect();
        let micro_heads: Vec<&[f32]> = weights.micro_heads.iter().map(Vec::as_slice).collect();
        let residual_feedback: Vec<&[f32]> = weights
            .residual_feedback
            .iter()
            .map(Vec::as_slice)
            .collect();
        let mut generator = generator(
            &weights,
            &micro_layers,
            &micro_residual,
            &micro_heads,
            residual_feedback,
            1,
            SamplingMode::CanonicalGreedy,
        );
        generator
            .begin_utterance(&prepared(&[1, 2]))
            .expect("valid tiny prompt");
        // XVector streaming with a 2-token target: role (3) + summed header (2) + one target/BOS
        // position.
        let prefill_len = generator.cached_positions();
        assert_eq!(prefill_len, 6);

        let frame = generator
            .next_frame()
            .expect("frame generation succeeds")
            .expect("all-zero heads cannot reach EOS");
        assert_eq!(frame.codes.len(), CODE_GROUP_COUNT);
        assert!((frame.codes[0] as usize) < PRIMARY_CODE_VOCAB_SIZE);
        assert!(
            frame
                .codes
                .iter()
                .skip(1)
                .all(|&code| (code as usize) < RESIDUAL_VOCAB)
        );
        assert_eq!(generator.cached_positions(), prefill_len + 1);
    }

    #[test]
    fn eos_ends_the_utterance_after_min_new_tokens_and_none_is_sticky() {
        // A positive EOS head row makes EOS the argmax; the min-new-tokens processor masks it for
        // the first two frames, so exactly two frames are emitted.
        let weights = TinyWeights::new(1.0);
        let frames = drive(&weights, 1, SamplingMode::CanonicalGreedy, 10);
        assert_eq!(frames.len(), crate::sampler::TALKER_MIN_NEW_TOKENS);

        let micro_layers = vec![weights.micro_layer(); 2];
        let micro_residual: Vec<&[f32]> = weights
            .micro_residual_embeddings
            .iter()
            .map(Vec::as_slice)
            .collect();
        let micro_heads: Vec<&[f32]> = weights.micro_heads.iter().map(Vec::as_slice).collect();
        let residual_feedback: Vec<&[f32]> = weights
            .residual_feedback
            .iter()
            .map(Vec::as_slice)
            .collect();
        let mut generator = generator(
            &weights,
            &micro_layers,
            &micro_residual,
            &micro_heads,
            residual_feedback,
            1,
            SamplingMode::CanonicalGreedy,
        );
        generator
            .begin_utterance(&prepared(&[1, 2]))
            .expect("valid tiny prompt");
        while generator.next_frame().expect("frames succeed").is_some() {}
        assert_eq!(
            generator
                .next_frame()
                .expect("finished utterance is stable"),
            None,
            "EOS must be sticky until the next begin_utterance"
        );
    }

    #[test]
    fn same_seed_and_artifact_produce_byte_identical_code_streams() {
        let weights = TinyWeights::new(1.0);
        let left = drive(&weights, 42, SamplingMode::Production, 6);
        let right = drive(&weights, 42, SamplingMode::Production, 6);
        assert!(!left.is_empty());
        assert_eq!(left, right);
    }

    #[test]
    fn begin_utterance_resets_state_for_a_fresh_utterance() {
        let weights = TinyWeights::new(0.0);
        let micro_layers = vec![weights.micro_layer(); 2];
        let micro_residual: Vec<&[f32]> = weights
            .micro_residual_embeddings
            .iter()
            .map(Vec::as_slice)
            .collect();
        let micro_heads: Vec<&[f32]> = weights.micro_heads.iter().map(Vec::as_slice).collect();
        let residual_feedback: Vec<&[f32]> = weights
            .residual_feedback
            .iter()
            .map(Vec::as_slice)
            .collect();
        let mut generator = generator(
            &weights,
            &micro_layers,
            &micro_residual,
            &micro_heads,
            residual_feedback,
            1,
            SamplingMode::CanonicalGreedy,
        );
        generator
            .begin_utterance(&prepared(&[1, 2]))
            .expect("valid tiny prompt");
        generator
            .next_frame()
            .expect("frame generation succeeds")
            .expect("frame emitted");

        generator
            .begin_utterance(&prepared(&[1, 2]))
            .expect("second utterance");
        assert_eq!(
            generator.cached_positions(),
            6,
            "a new utterance must not inherit the previous KV prefix"
        );
    }

    #[test]
    fn full_utterance_streamed_kv_matches_causally_replayed_prefill() {
        // The cached decode path must leave precisely the same talker state as rebuilding the
        // prompt plus every emitted-frame input in one causal prefill. This is deliberately a
        // real multi-frame utterance: EOS is masked for the first two generated frames, then ends
        // the run on the third sampling decision.
        let mut weights = TinyWeights::new(1.0);
        weights.talker_q.fill(0.05);
        weights.talker_kv.fill(0.1);
        // Keep attention observable: a zero output projection would still compare the cached
        // key/value buffers, but would not prove that later positions consume them.
        weights.talker_o.fill(0.025);
        let micro_layers = vec![weights.micro_layer(); 2];
        let micro_residual: Vec<&[f32]> = weights
            .micro_residual_embeddings
            .iter()
            .map(Vec::as_slice)
            .collect();
        let micro_heads: Vec<&[f32]> = weights.micro_heads.iter().map(Vec::as_slice).collect();
        let residual_feedback: Vec<&[f32]> = weights
            .residual_feedback
            .iter()
            .map(Vec::as_slice)
            .collect();
        let mut streamed = generator(
            &weights,
            &micro_layers,
            &micro_residual,
            &micro_heads,
            residual_feedback.clone(),
            1,
            SamplingMode::CanonicalGreedy,
        );
        let initial_prefill = vec![0.25; 3 * HIDDEN];
        let trailing = vec![vec![0.5; HIDDEN], vec![0.75; HIDDEN]];
        streamed
            .begin_with_prefill(&initial_prefill, 3, trailing.clone())
            .expect("valid test prefill");

        let mut frames = Vec::new();
        while let Some(frame) = streamed
            .next_frame()
            .expect("streamed frame generation succeeds")
        {
            frames.push(frame);
        }
        assert_eq!(frames.len(), crate::sampler::TALKER_MIN_NEW_TOKENS);

        let mut replay_prefill = initial_prefill;
        for (frame_index, frame) in frames.iter().enumerate() {
            let mut rows = Vec::with_capacity(CODE_GROUP_COUNT);
            for (depth, &code) in frame.codes.iter().enumerate() {
                let table = if depth == 0 {
                    &weights.talker_codec_embedding
                } else {
                    &weights.residual_feedback[depth - 1]
                };
                rows.push(&table[code as usize * HIDDEN..(code as usize + 1) * HIDDEN]);
            }
            let mut input = vec![0.0; HIDDEN];
            talker::form_frame_input(
                &rows,
                trailing.get(frame_index).map(Vec::as_slice),
                &streamed.header.tts_pad,
                &mut input,
            );
            replay_prefill.extend_from_slice(&input);
        }

        let mut replayed = generator(
            &weights,
            &micro_layers,
            &micro_residual,
            &micro_heads,
            residual_feedback,
            1,
            SamplingMode::CanonicalGreedy,
        );
        replayed
            .begin_with_prefill(&replay_prefill, 3 + frames.len(), trailing)
            .expect("replayed prefill is valid");
        let replayed_utterance = replayed.utterance.as_mut().expect("replayed state");
        replayed_utterance.frames_emitted = frames.len();
        replayed_utterance.group_zero_history = frames.iter().map(|frame| frame.codes[0]).collect();

        assert_eq!(
            streamed.kv, replayed.kv,
            "cached single-position decode and causal replay must retain identical KV buffers"
        );
        // The streamed generator's loop ended BY taking the terminal EOS decision from its parked
        // logits, so its `finished` flag is already set while the freshly replayed state has not
        // decided yet. Comparing full states across that boundary would demand the impossible;
        // instead make the replayed side take the same decision, prove it is the same decision,
        // and only then require the two states to be identical in every field.
        assert!(
            replayed
                .next_frame()
                .expect("replayed terminal decision")
                .is_none(),
            "the replayed state must make the same EOS decision"
        );
        assert!(
            streamed
                .next_frame()
                .expect("streamed EOS must stay sticky")
                .is_none(),
        );
        assert_eq!(
            streamed.utterance, replayed.utterance,
            "cached decode and causal replay must park the same next-frame state"
        );
        assert_eq!(
            streamed.kv, replayed.kv,
            "the aligned terminal decisions must not have touched the KV buffers"
        );
    }
}
