//! The Mimi speech-tokenizer ENCODER: 24 kHz reference audio in, 16 codec codes per frame out.
//!
//! This is the ICL enrollment path's missing arithmetic (bead `frankentts-p1-codec-encoder-snt`):
//! the quality cloning path prompts the talker with the reference *as codec tokens*, and those
//! tokens must be the ones the official stack would produce — a mismatch here poisons the prompt
//! and everything downstream of enrollment. Ordinary synthesis never touches this module; it is
//! enrollment-only and stays off the hot path, so every loop below chooses clarity and exactness
//! over speed.
//!
//! # Reference
//!
//! `Qwen3TTSTokenizerV2Encoder` is `transformers.MimiModel` (pinned `4.57.3`,
//! `models/mimi/modeling_mimi.py`) with the decode branches set to `None`; the wrapper's
//! `encode()` keeps the first 16 of 32 quantizers (`encoder_valid_num_quantizers`, `MOD:983`) and
//! trims the frame axis to `ceil(samples / 1920)`. Geometry and operator order live in
//! `docs/QWEN3_TTS_CODEC_SPEC.md` §8; the traps that section names are re-stated at their code
//! sites below. Config values are the checkpoint's `encoder_config`
//! (`speech_tokenizer/config.json`): `use_causal_conv true`, `pad_mode "constant"`,
//! GELU, LayerNorm-with-bias (`norm_eps 1e-5`), LayerScale, RoPE theta 10 000, 8 heads.
//!
//! # The two port-critical findings source reading added over spec §8
//!
//! 1. **Causal convs pad LEFT-only.** With `use_causal_conv: true` every `MimiConv1d` takes
//!    `padding_total = k̂ − stride` entirely on the LEFT plus the ceil-to-frame `extra_padding`
//!    on the RIGHT (`_pad1d((padding_total, extra))`, `modeling_mimi.py:332`). The asymmetric
//!    left/right split the spec's padding paragraph describes is the NON-causal branch and is
//!    dead for this checkpoint.
//! 2. **The sliding window is NOT applied on the oracle path.** `MimiTransformerModel` builds its
//!    mask with plain `create_causal_mask`, which never consults `config.sliding_window`; only
//!    the flash-attention branch passes the window. The pinned CPU oracle therefore runs PLAIN
//!    causal attention, and so does this port. (Window 250 in the config is real but inert here —
//!    an upstream eager/flash inconsistency worth knowing about, not accommodating.)
//!
//! # Layout
//!
//! Activations are time-major `[frames, channels]`, matching [`crate::codec`]. PyTorch's conv
//! weights `[out, in, k]` flatten row-major to exactly the `[out, in·k]` GEMM operand the im2col
//! below builds columns for (`column[c·k + tap]`), so weights are used as stored. Convolutions go
//! through the same platform-BLAS `beta = 1` bias-seeded GEMM the codec decoder proved exact
//! against its oracle (`ftts-conformance/tests/codec_gemm_bisect.rs`); the reference issues these
//! convs through the identical `slow_conv2d` im2col path.

use std::path::Path;

use ftts_kernels::f32ref;

use crate::checkpoint::{CheckpointError, FileStore, TensorStore};
use crate::codec::{MaterializedCodebook, codec_rope_rows, gelu};
use ftts_artifacts::safetensors::SafetensorsFile;

/// Input/output sample rate the encoder is defined at.
pub const ENCODER_SAMPLE_RATE_HZ: u32 = 24_000;
/// Samples per emitted frame: the total conv-stack downsample (4·5·6·8·2 = 1920 at 24 kHz).
pub const ENCODE_DOWNSAMPLE_RATE: usize = 1_920;
/// Codec code groups the product consumes per frame (first 16 of the checkpoint's 32).
pub const CODE_GROUPS: usize = 16;

const HIDDEN: usize = 512;
const HEADS: usize = 8;
const HEAD_DIM: usize = 64;
const INTERMEDIATE: usize = 2_048;
const TRANSFORMER_LAYERS: usize = 8;
const NORM_EPS: f32 = 1e-5;
const VQ_DIM: usize = 256;
const CODEBOOK_SIZE: usize = 2_048;
const ACOUSTIC_GROUPS: usize = CODE_GROUPS - 1;
/// `1 / sqrt(head_dim)`, formed exactly as the reference's `1 / math.sqrt(64)`.
const ATTENTION_SCALING: f32 = 0.125;

/// Refusals at the audio-to-codes boundary.
#[derive(Debug, PartialEq, Eq)]
pub enum SpeechEncodeError {
    /// The reference is shorter than one emitted frame.
    AudioTooShort { samples: usize },
    /// A non-finite sample reached the encoder.
    NonFiniteAudio { index: usize },
}

impl core::fmt::Display for SpeechEncodeError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::AudioTooShort { samples } => write!(
                formatter,
                "reference audio has {samples} samples; ICL encoding needs at least one \
                 {ENCODE_DOWNSAMPLE_RATE}-sample frame"
            ),
            Self::NonFiniteAudio { index } => {
                write!(formatter, "reference audio sample {index} is not finite")
            }
        }
    }
}

impl std::error::Error for SpeechEncodeError {}

/// How a conv pads, per the reference's `pad_mode` argument.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PadMode {
    /// `pad_mode "constant"` — zeros (every stack conv; the checkpoint's config-wide mode).
    Constant,
    /// `pad_mode "replicate"` — edge value (ONLY the model-level downsample, `MOD` sets it
    /// explicitly when constructing `self.downsample`).
    Replicate,
}

/// One `MimiConv1d`: weights as stored plus the causal padding arithmetic.
struct EncoderConv {
    /// `[out, in·k]` row-major, exactly the flattened torch `[out, in, k]`.
    weight: Vec<f32>,
    bias: Option<Vec<f32>>,
    in_channels: usize,
    out_channels: usize,
    kernel: usize,
    stride: usize,
    pad_mode: PadMode,
}

impl EncoderConv {
    /// Output frame count for `frames` input frames, mirroring `_get_extra_padding_for_conv1d`:
    /// `n = ceil((L − k + padding_total) / s + 1) − 1`, output = `n + 1 = div_ceil(L − k + pt, s) + 1`.
    fn output_frames(&self, frames: usize) -> usize {
        let padding_total = self.kernel - self.stride;
        // Enrollment audio is always at least one frame; the stem (k=7) needs L >= 1 after the
        // guard in `encode_24khz_pcm`, and every later stage only shrinks by its stride.
        let numerator = frames + padding_total - self.kernel;
        numerator.div_ceil(self.stride) + 1
    }

    /// The reference forward: left-pad `padding_total`, right-pad the ceil-to-frame extra, then
    /// one im2col GEMM with the bias seeded under `beta = 1` (the codec's proven-exact form).
    fn forward(&self, input: &[f32], frames: usize) -> Vec<f32> {
        assert_eq!(input.len(), frames * self.in_channels, "conv input shape");
        let padding_total = self.kernel - self.stride;
        let out_frames = self.output_frames(frames);
        // ideal_length − length, always in [0, stride): how far the input misses a whole frame.
        let extra = ((out_frames - 1) * self.stride + self.kernel - padding_total) - frames;
        let padded_frames = padding_total + frames + extra;

        let mut padded = vec![0.0f32; padded_frames * self.in_channels];
        padded[padding_total * self.in_channels..(padding_total + frames) * self.in_channels]
            .copy_from_slice(input);
        if self.pad_mode == PadMode::Replicate {
            // torch `replicate` pad repeats the edge frame; `constant` leaves the zeros above.
            let (first, rest) = padded.split_at_mut(padding_total * self.in_channels);
            for row in first.chunks_mut(self.in_channels) {
                row.copy_from_slice(&rest[..self.in_channels]);
            }
            if extra > 0 {
                let last_source = (padding_total + frames - 1) * self.in_channels
                    ..(padding_total + frames) * self.in_channels;
                let last = padded[last_source].to_vec();
                for row in padded[(padding_total + frames) * self.in_channels..]
                    .chunks_mut(self.in_channels)
                {
                    row.copy_from_slice(&last);
                }
            }
        }

        let reduction = self.in_channels * self.kernel;
        let mut columns = vec![0.0f32; out_frames * reduction];
        for frame in 0..out_frames {
            let start = frame * self.stride;
            let target = &mut columns[frame * reduction..][..reduction];
            for tap in 0..self.kernel {
                let source = &padded[(start + tap) * self.in_channels..][..self.in_channels];
                for (channel, &value) in source.iter().enumerate() {
                    target[channel * self.kernel + tap] = value;
                }
            }
        }
        let mut output = vec![0.0f32; out_frames * self.out_channels];
        f32ref::linear_with_accumulation(
            &columns,
            &self.weight,
            self.bias.as_deref(),
            out_frames,
            reduction,
            self.out_channels,
            f32ref::F32LinearAccumulation::AccelerateBiasSeededRowInvariant,
            &mut output,
        );
        output
    }
}

/// One SEANet residual unit: pre-activation `ELU → conv(k3) → ELU → conv(k1)`, identity shortcut,
/// additive join, NO trailing activation (`MimiResnetBlock`, `use_conv_shortcut` false).
struct ResidualUnit {
    conv_a: EncoderConv,
    conv_b: EncoderConv,
}

impl ResidualUnit {
    fn forward(&self, input: &[f32], frames: usize) -> Vec<f32> {
        let mut hidden = input.to_vec();
        elu_in_place(&mut hidden);
        let hidden = self.conv_a.forward(&hidden, frames);
        let mut hidden_frames_checked = hidden;
        elu_in_place(&mut hidden_frames_checked);
        let branch = self.conv_b.forward(&hidden_frames_checked, frames);
        debug_assert_eq!(branch.len(), input.len(), "residual unit preserves shape");
        branch
            .iter()
            .zip(input.iter())
            .map(|(b, r)| r + b)
            .collect()
    }
}

/// One encoder-transformer layer's owned weights (LayerNorm-with-bias, bias-free projections,
/// GELU MLP, LayerScale on both branches — deliberately NOT the decode-path transformer, which
/// is RMSNorm/SiLU/16-head; spec §8 warns they cannot share a kernel).
struct TransformerLayer {
    input_norm_weight: Vec<f32>,
    input_norm_bias: Vec<f32>,
    q_proj: Vec<f32>,
    k_proj: Vec<f32>,
    v_proj: Vec<f32>,
    o_proj: Vec<f32>,
    attn_scale: Vec<f32>,
    post_norm_weight: Vec<f32>,
    post_norm_bias: Vec<f32>,
    fc1: Vec<f32>,
    fc2: Vec<f32>,
    mlp_scale: Vec<f32>,
}

/// One residual-VQ branch: shared input projection plus per-level codebooks.
struct RvqBranch {
    /// `input_proj.weight` `[256, 512]` (the trailing conv axis of 1 dropped), bias-free.
    input_proj: Vec<f32>,
    codebooks: Vec<MaterializedCodebook>,
    /// Per-level `‖e‖²` rows, precomputed once for the distance form below.
    codebook_norms: Vec<Vec<f32>>,
}

impl RvqBranch {
    /// `MimiResidualVectorQuantizer.encode`: project once, then per level pick the nearest
    /// centroid of the RESIDUAL and subtract its reconstruction.
    ///
    /// Distance is the reference's mm form — `torch.cdist` at this size routes through
    /// `‖x‖² − 2x·e + ‖e‖²` (clamped at zero; the final sqrt is monotone and skipped). Ties take
    /// the LOWEST index, as `argmin` does.
    fn encode(
        &self,
        embeddings: &[f32],
        frames: usize,
        codes: &mut [u32],
        group_stride: usize,
        first_group: usize,
    ) {
        let mut projected = vec![0.0f32; frames * VQ_DIM];
        f32ref::linear_with_accumulation(
            embeddings,
            &self.input_proj,
            None,
            frames,
            HIDDEN,
            VQ_DIM,
            f32ref::F32LinearAccumulation::AccelerateBiasSeededRowInvariant,
            &mut projected,
        );
        let mut residual = projected;
        let mut dots = vec![0.0f32; frames * CODEBOOK_SIZE];
        for (level, codebook) in self.codebooks.iter().enumerate() {
            let entries = codebook.entries();
            let norms = &self.codebook_norms[level];
            f32ref::linear_with_accumulation(
                &residual,
                entries,
                None,
                frames,
                VQ_DIM,
                CODEBOOK_SIZE,
                f32ref::F32LinearAccumulation::AccelerateBiasSeededRowInvariant,
                &mut dots,
            );
            for frame in 0..frames {
                let row = &residual[frame * VQ_DIM..][..VQ_DIM];
                let x_norm: f32 = row.iter().map(|v| v * v).sum();
                let dot_row = &dots[frame * CODEBOOK_SIZE..][..CODEBOOK_SIZE];
                let mut best = 0usize;
                let mut best_distance = f32::INFINITY;
                for (index, (&dot, &e_norm)) in dot_row.iter().zip(norms.iter()).enumerate() {
                    let distance = (x_norm - 2.0 * dot + e_norm).max(0.0);
                    if distance < best_distance {
                        best_distance = distance;
                        best = index;
                    }
                }
                codes[frame * group_stride + first_group + level] = best as u32;
                let centroid = &entries[best * VQ_DIM..][..VQ_DIM];
                let target = &mut residual[frame * VQ_DIM..][..VQ_DIM];
                for (value, &e) in target.iter_mut().zip(centroid.iter()) {
                    *value -= e;
                }
            }
        }
    }
}

/// The complete encoder: SEANet stack, transformer, model-level downsample, split RVQ.
pub struct SpeechEncoder {
    stem: EncoderConv,
    stages: Vec<(ResidualUnit, EncoderConv)>,
    final_conv: EncoderConv,
    transformer: Vec<TransformerLayer>,
    downsample: EncoderConv,
    semantic: RvqBranch,
    acoustic: RvqBranch,
}

impl SpeechEncoder {
    /// Hydrate from `speech_tokenizer/model.safetensors` (the same file the codec decoder loads;
    /// the encoder's 225 tensors ship F32 with weight norm already fused into plain weights).
    ///
    /// # Errors
    ///
    /// When the file cannot be opened or a tensor is missing or mis-shaped.
    pub fn load(path: &Path) -> Result<Self, CheckpointError> {
        let store = crate::checkpoint::open(path)?;
        Self::load_from_file(&store, path)
    }

    /// [`SpeechEncoder::load`] over an already-parsed checkpoint.
    ///
    /// # Errors
    ///
    /// As [`SpeechEncoder::load`].
    pub fn load_from_file(file: &SafetensorsFile, label: &Path) -> Result<Self, CheckpointError> {
        let store = FileStore(file);
        let take = |name: &str| store.take_widened(label, name);
        let conv = |prefix: &str,
                    out_channels: usize,
                    in_channels: usize,
                    kernel: usize,
                    stride: usize,
                    with_bias: bool,
                    pad_mode: PadMode|
         -> Result<EncoderConv, CheckpointError> {
            let weight = take(&format!("{prefix}.weight"))?;
            expect_len(
                &format!("{prefix}.weight"),
                &weight,
                out_channels * in_channels * kernel,
            )?;
            let bias = if with_bias {
                let bias = take(&format!("{prefix}.bias"))?;
                expect_len(&format!("{prefix}.bias"), &bias, out_channels)?;
                Some(bias)
            } else {
                None
            };
            Ok(EncoderConv {
                weight,
                bias,
                in_channels,
                out_channels,
                kernel,
                stride,
                pad_mode,
            })
        };

        let stem = conv(
            "encoder.encoder.layers.0.conv",
            64,
            1,
            7,
            1,
            true,
            PadMode::Constant,
        )?;
        // Stage layout from `MimiEncoder.__init__`: per reversed ratio [4, 5, 6, 8] the module
        // list gains [resblock, ELU, downsample-conv], so parametered layers sit at
        // 1/3, 4/6, 7/9, 10/12, with the parameterless ELUs at 2, 5, 8, 11 and 13.
        let mut stages = Vec::new();
        for (stage, (ratio, channels)) in [(4usize, 64usize), (5, 128), (6, 256), (8, 512)]
            .into_iter()
            .enumerate()
        {
            let res_layer = 3 * stage + 1;
            let down_layer = 3 * stage + 3;
            let unit = ResidualUnit {
                conv_a: conv(
                    &format!("encoder.encoder.layers.{res_layer}.block.1.conv"),
                    channels / 2,
                    channels,
                    3,
                    1,
                    true,
                    PadMode::Constant,
                )?,
                conv_b: conv(
                    &format!("encoder.encoder.layers.{res_layer}.block.3.conv"),
                    channels,
                    channels / 2,
                    1,
                    1,
                    true,
                    PadMode::Constant,
                )?,
            };
            let down = conv(
                &format!("encoder.encoder.layers.{down_layer}.conv"),
                channels * 2,
                channels,
                2 * ratio,
                ratio,
                true,
                PadMode::Constant,
            )?;
            stages.push((unit, down));
        }
        let final_conv = conv(
            "encoder.encoder.layers.14.conv",
            HIDDEN,
            1_024,
            3,
            1,
            true,
            PadMode::Constant,
        )?;

        let mut transformer = Vec::with_capacity(TRANSFORMER_LAYERS);
        for layer in 0..TRANSFORMER_LAYERS {
            let base = format!("encoder.encoder_transformer.layers.{layer}");
            let sized = |name: String, expected: usize| -> Result<Vec<f32>, CheckpointError> {
                let tensor = take(&name)?;
                expect_len(&name, &tensor, expected)?;
                Ok(tensor)
            };
            transformer.push(TransformerLayer {
                input_norm_weight: sized(format!("{base}.input_layernorm.weight"), HIDDEN)?,
                input_norm_bias: sized(format!("{base}.input_layernorm.bias"), HIDDEN)?,
                q_proj: sized(format!("{base}.self_attn.q_proj.weight"), HIDDEN * HIDDEN)?,
                k_proj: sized(format!("{base}.self_attn.k_proj.weight"), HIDDEN * HIDDEN)?,
                v_proj: sized(format!("{base}.self_attn.v_proj.weight"), HIDDEN * HIDDEN)?,
                o_proj: sized(format!("{base}.self_attn.o_proj.weight"), HIDDEN * HIDDEN)?,
                attn_scale: sized(format!("{base}.self_attn_layer_scale.scale"), HIDDEN)?,
                post_norm_weight: sized(format!("{base}.post_attention_layernorm.weight"), HIDDEN)?,
                post_norm_bias: sized(format!("{base}.post_attention_layernorm.bias"), HIDDEN)?,
                fc1: sized(format!("{base}.mlp.fc1.weight"), INTERMEDIATE * HIDDEN)?,
                fc2: sized(format!("{base}.mlp.fc2.weight"), HIDDEN * INTERMEDIATE)?,
                mlp_scale: sized(format!("{base}.mlp_layer_scale.scale"), HIDDEN)?,
            });
        }

        let downsample = conv(
            "encoder.downsample.conv",
            HIDDEN,
            HIDDEN,
            4,
            2,
            false,
            PadMode::Replicate,
        )?;

        let branch = |prefix: &str, levels: usize| -> Result<RvqBranch, CheckpointError> {
            let input_proj = take(&format!("{prefix}.input_proj.weight"))?;
            expect_len(
                &format!("{prefix}.input_proj.weight"),
                &input_proj,
                VQ_DIM * HIDDEN,
            )?;
            let mut codebooks = Vec::with_capacity(levels);
            let mut codebook_norms = Vec::with_capacity(levels);
            for level in 0..levels {
                // Encoder-side names are `embed_sum` / `cluster_usage` — the decoder's are
                // `embedding_sum`; spec §8 pins this as "same hazard, different tensor names".
                let sum = take(&format!("{prefix}.layers.{level}.codebook.embed_sum"))?;
                let usage = take(&format!("{prefix}.layers.{level}.codebook.cluster_usage"))?;
                let codebook =
                    MaterializedCodebook::from_unnormalized(&sum, &usage, CODEBOOK_SIZE, VQ_DIM)
                        .map_err(CheckpointError::Codec)?;
                let norms = codebook
                    .entries()
                    .chunks(VQ_DIM)
                    .map(|row| row.iter().map(|v| v * v).sum())
                    .collect();
                codebooks.push(codebook);
                codebook_norms.push(norms);
            }
            Ok(RvqBranch {
                input_proj,
                codebooks,
                codebook_norms,
            })
        };
        let semantic = branch("encoder.quantizer.semantic_residual_vector_quantizer", 1)?;
        let acoustic = branch(
            "encoder.quantizer.acoustic_residual_vector_quantizer",
            ACOUSTIC_GROUPS,
        )?;

        Ok(Self {
            stem,
            stages,
            final_conv,
            transformer,
            downsample,
            semantic,
            acoustic,
        })
    }

    /// Encode mono 24 kHz PCM to `[frames, 16]` codec codes, frames-major — the exact ids the
    /// pinned reference's `encode()` returns (semantic group first), trimmed to
    /// `ceil(samples / 1920)` frames as the wrapper does.
    ///
    /// # Errors
    ///
    /// When the audio is shorter than one frame or carries a non-finite sample.
    pub fn encode_24khz_pcm(&self, pcm: &[f32]) -> Result<Vec<u32>, SpeechEncodeError> {
        if pcm.len() < ENCODE_DOWNSAMPLE_RATE {
            return Err(SpeechEncodeError::AudioTooShort { samples: pcm.len() });
        }
        if let Some(index) = pcm.iter().position(|value| !value.is_finite()) {
            return Err(SpeechEncodeError::NonFiniteAudio { index });
        }

        // SEANet stack: stem, then per stage resblock → ELU → strided downsample, then the
        // closing ELU → final conv. `frames` tracks the time axis through every stride.
        let mut frames = pcm.len();
        let mut hidden = self.stem.forward(pcm, frames);
        for (unit, down) in &self.stages {
            hidden = unit.forward(&hidden, frames);
            elu_in_place(&mut hidden);
            let next_frames = down.output_frames(frames);
            hidden = down.forward(&hidden, frames);
            frames = next_frames;
        }
        elu_in_place(&mut hidden);
        hidden = self.final_conv.forward(&hidden, frames);

        for layer in &self.transformer {
            transformer_layer_in_place(layer, &mut hidden, frames);
        }

        let out_frames = self.downsample.output_frames(frames);
        let embeddings = self.downsample.forward(&hidden, frames);
        let mut codes = vec![0u32; out_frames * CODE_GROUPS];
        // Split RVQ: BOTH branches encode the same embeddings — the acoustic branch does NOT
        // see the semantic residual (`MimiSplitResidualVectorQuantizer.encode` passes
        // `embeddings` to each), only its own chain of 15.
        self.semantic
            .encode(&embeddings, out_frames, &mut codes, CODE_GROUPS, 0);
        self.acoustic
            .encode(&embeddings, out_frames, &mut codes, CODE_GROUPS, 1);

        // The wrapper trims to `ceil(mask_sum / 1920)`; with a full mask that is the natural
        // frame count, so a mismatch means the padding arithmetic above drifted — fail loudly.
        let expected = pcm.len().div_ceil(ENCODE_DOWNSAMPLE_RATE);
        assert_eq!(
            out_frames, expected,
            "conv-chain frame count diverged from ceil(samples/1920)"
        );
        Ok(codes)
    }
}

/// torch CPU ELU (alpha 1): `x > 0 ? x : expm1(x)` — the kernel uses `expm1`, not `exp − 1`.
fn elu_in_place(values: &mut [f32]) {
    for value in values.iter_mut() {
        if *value <= 0.0 {
            *value = value.exp_m1();
        }
    }
}

/// `nn.LayerNorm` over the channel axis: biased variance, `(x − mean) / sqrt(var + eps) · w + b`.
fn layer_norm(input: &[f32], weight: &[f32], bias: &[f32], output: &mut [f32]) {
    let n = weight.len();
    debug_assert_eq!(input.len(), n);
    let mean = input.iter().sum::<f32>() / n as f32;
    let variance = input.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n as f32;
    let rstd = (variance + NORM_EPS).sqrt().recip();
    for index in 0..n {
        output[index] = (input[index] - mean) * rstd * weight[index] + bias[index];
    }
}

/// One `MimiTransformerLayer` over the full sequence, plain-causal (see module doc, finding 2).
fn transformer_layer_in_place(layer: &TransformerLayer, hidden: &mut [f32], frames: usize) {
    let mut normed = vec![0.0f32; frames * HIDDEN];
    for frame in 0..frames {
        layer_norm(
            &hidden[frame * HIDDEN..][..HIDDEN],
            &layer.input_norm_weight,
            &layer.input_norm_bias,
            &mut normed[frame * HIDDEN..][..HIDDEN],
        );
    }

    let project = |weight: &[f32]| -> Vec<f32> {
        let mut out = vec![0.0f32; frames * HIDDEN];
        f32ref::linear_with_accumulation(
            &normed,
            weight,
            None,
            frames,
            HIDDEN,
            HIDDEN,
            f32ref::F32LinearAccumulation::AccelerateBiasSeededRowInvariant,
            &mut out,
        );
        out
    };
    let mut queries = project(&layer.q_proj);
    let mut keys = project(&layer.k_proj);
    let values = project(&layer.v_proj);

    // RoPE: same theta-10000 half-duplicated rows as the codec decoder; positions are 0..frames.
    let mut cos = vec![0.0f32; HEAD_DIM];
    let mut sin = vec![0.0f32; HEAD_DIM];
    for frame in 0..frames {
        codec_rope_rows(frame, HEAD_DIM, &mut cos, &mut sin);
        for head in 0..HEADS {
            let offset = frame * HIDDEN + head * HEAD_DIM;
            rotate_in_place(&mut queries[offset..offset + HEAD_DIM], &cos, &sin);
            rotate_in_place(&mut keys[offset..offset + HEAD_DIM], &cos, &sin);
        }
    }

    let mut context = vec![0.0f32; frames * HIDDEN];
    let mut scores = vec![0.0f32; frames];
    for head in 0..HEADS {
        let head_offset = head * HEAD_DIM;
        for frame in 0..frames {
            let query = &queries[frame * HIDDEN + head_offset..][..HEAD_DIM];
            let visible = frame + 1;
            for (position, score) in scores[..visible].iter_mut().enumerate() {
                let key = &keys[position * HIDDEN + head_offset..][..HEAD_DIM];
                let mut dot = 0.0f32;
                for lane in 0..HEAD_DIM {
                    dot += query[lane] * key[lane];
                }
                *score = dot * ATTENTION_SCALING;
            }
            f32ref::softmax_rows_with_arithmetic(
                &mut scores[..visible],
                1,
                visible,
                f32ref::F32SoftmaxArithmetic::Divide,
            );
            let target = &mut context[frame * HIDDEN + head_offset..][..HEAD_DIM];
            target.fill(0.0);
            for (position, &score) in scores[..visible].iter().enumerate() {
                let value = &values[position * HIDDEN + head_offset..][..HEAD_DIM];
                for lane in 0..HEAD_DIM {
                    target[lane] += score * value[lane];
                }
            }
        }
    }

    let mut attention_out = vec![0.0f32; frames * HIDDEN];
    f32ref::linear_with_accumulation(
        &context,
        &layer.o_proj,
        None,
        frames,
        HIDDEN,
        HIDDEN,
        f32ref::F32LinearAccumulation::AccelerateBiasSeededRowInvariant,
        &mut attention_out,
    );
    for frame in 0..frames {
        for channel in 0..HIDDEN {
            hidden[frame * HIDDEN + channel] +=
                layer.attn_scale[channel] * attention_out[frame * HIDDEN + channel];
        }
    }

    for frame in 0..frames {
        layer_norm(
            &hidden[frame * HIDDEN..][..HIDDEN],
            &layer.post_norm_weight,
            &layer.post_norm_bias,
            &mut normed[frame * HIDDEN..][..HIDDEN],
        );
    }
    let mut up = vec![0.0f32; frames * INTERMEDIATE];
    f32ref::linear_with_accumulation(
        &normed,
        &layer.fc1,
        None,
        frames,
        HIDDEN,
        INTERMEDIATE,
        f32ref::F32LinearAccumulation::AccelerateBiasSeededRowInvariant,
        &mut up,
    );
    for value in up.iter_mut() {
        *value = gelu(*value);
    }
    let mut down = vec![0.0f32; frames * HIDDEN];
    f32ref::linear_with_accumulation(
        &up,
        &layer.fc2,
        None,
        frames,
        INTERMEDIATE,
        HIDDEN,
        f32ref::F32LinearAccumulation::AccelerateBiasSeededRowInvariant,
        &mut down,
    );
    for frame in 0..frames {
        for channel in 0..HIDDEN {
            hidden[frame * HIDDEN + channel] +=
                layer.mlp_scale[channel] * down[frame * HIDDEN + channel];
        }
    }
}

/// LLaMA-convention rotary application over one head: `x·cos + rotate_half(x)·sin`.
fn rotate_in_place(head: &mut [f32], cos: &[f32], sin: &[f32]) {
    let half = HEAD_DIM / 2;
    let mut rotated = [0.0f32; HEAD_DIM];
    for index in 0..half {
        rotated[index] = -head[index + half];
        rotated[index + half] = head[index];
    }
    for index in 0..HEAD_DIM {
        head[index] = head[index] * cos[index] + rotated[index] * sin[index];
    }
}

fn expect_len(name: &str, tensor: &[f32], expected: usize) -> Result<(), CheckpointError> {
    if tensor.len() == expected {
        Ok(())
    } else {
        Err(CheckpointError::TensorShape {
            tensor: name.to_owned(),
            expected,
            actual: tensor.len(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn conv(kernel: usize, stride: usize) -> EncoderConv {
        EncoderConv {
            weight: vec![0.0; kernel],
            bias: None,
            in_channels: 1,
            out_channels: 1,
            kernel,
            stride,
            pad_mode: PadMode::Constant,
        }
    }

    #[test]
    fn causal_output_length_matches_the_reference_formula() {
        // Stride-1 convs preserve length exactly (padding_total = k − 1).
        for kernel in [1usize, 3, 7] {
            for frames in [1usize, 5, 1_920, 264_000] {
                assert_eq!(
                    conv(kernel, 1).output_frames(frames),
                    frames,
                    "k={kernel} L={frames}"
                );
            }
        }
        // Strided stage convs land on ceil(L / stride) — the ceil-to-frame extra pad's purpose.
        for (ratio, frames) in [(4usize, 264_000usize), (5, 66_000), (6, 13_200), (8, 2_200)] {
            assert_eq!(
                conv(2 * ratio, ratio).output_frames(frames),
                frames.div_ceil(ratio)
            );
        }
        assert_eq!(conv(4, 2).output_frames(275), 138, "model-level downsample");
        // The jfk-length chain: 264000 samples -> 138 frames == ceil(264000 / 1920).
        let chain = [(8usize, 4usize), (10, 5), (12, 6), (16, 8), (4, 2)];
        let mut frames = 264_000usize;
        for (kernel, stride) in chain {
            frames = conv(kernel, stride).output_frames(frames);
        }
        assert_eq!(frames, 264_000usize.div_ceil(ENCODE_DOWNSAMPLE_RATE));
    }

    #[test]
    fn strided_conv_places_taps_like_the_reference() {
        // k=4, s=2, causal: padding_total = 2 on the left, input [1,2,3,4] + extra 0.
        // Output frame 0 reads padded[0..4] = [0,0,1,2]; frame 1 reads [1,2,3,4].
        let unit = EncoderConv {
            weight: vec![1.0, 1.0, 1.0, 1.0],
            bias: None,
            in_channels: 1,
            out_channels: 1,
            kernel: 4,
            stride: 2,
            pad_mode: PadMode::Constant,
        };
        let out = unit.forward(&[1.0, 2.0, 3.0, 4.0], 4);
        assert_eq!(out, vec![3.0, 10.0]);
    }

    #[test]
    fn replicate_padding_repeats_the_edge_frames() {
        let unit = EncoderConv {
            weight: vec![1.0, 1.0, 1.0, 1.0],
            bias: None,
            in_channels: 1,
            out_channels: 1,
            kernel: 4,
            stride: 2,
            pad_mode: PadMode::Replicate,
        };
        // padded = [1,1,1,2,3] + extra 1 replicated -> [1,1,1,2,3,3]; frames read [1,1,1,2],[1,2,3,3].
        let out = unit.forward(&[1.0, 2.0, 3.0], 3);
        assert_eq!(out, vec![5.0, 9.0]);
    }

    #[test]
    fn elu_uses_expm1_below_zero_and_identity_above() {
        let mut values = [1.5f32, 0.0, -1.0];
        elu_in_place(&mut values);
        assert_eq!(values[0], 1.5);
        assert_eq!(values[1], 0.0);
        assert_eq!(values[2], (-1.0f32).exp_m1());
    }

    #[test]
    fn layer_norm_matches_a_hand_computation() {
        let input = [1.0f32, 2.0, 3.0, 4.0];
        let weight = [1.0f32; 4];
        let bias = [0.5f32; 4];
        let mut output = [0.0f32; 4];
        layer_norm(&input, &weight, &bias, &mut output);
        let mean = 2.5f32;
        let variance = 1.25f32;
        let rstd = (variance + NORM_EPS).sqrt().recip();
        for (index, &value) in input.iter().enumerate() {
            assert_eq!(output[index], (value - mean) * rstd + 0.5);
        }
    }

    #[test]
    fn nearest_centroid_ties_take_the_lowest_index() {
        // Two identical centroids: argmin must return the first.
        let sum = vec![1.0f32; 2 * VQ_DIM]
            .into_iter()
            .chain(vec![0.0f32; (CODEBOOK_SIZE - 2) * VQ_DIM])
            .collect::<Vec<_>>();
        let usage = vec![1.0f32; CODEBOOK_SIZE];
        let codebook = MaterializedCodebook::from_unnormalized(&sum, &usage, CODEBOOK_SIZE, VQ_DIM)
            .expect("codebook");
        let norms: Vec<f32> = codebook
            .entries()
            .chunks(VQ_DIM)
            .map(|row| row.iter().map(|v| v * v).sum())
            .collect();
        let branch = RvqBranch {
            input_proj: identity_projection(),
            codebooks: vec![codebook],
            codebook_norms: vec![norms],
        };
        let embeddings = vec![1.0f32; HIDDEN];
        let mut codes = vec![u32::MAX; CODE_GROUPS];
        branch.encode(&embeddings, 1, &mut codes, CODE_GROUPS, 0);
        assert_eq!(codes[0], 0, "tie must resolve to the lowest index");
    }

    /// `[256, 512]` projection taking the first 256 channels, for quantizer tests.
    fn identity_projection() -> Vec<f32> {
        let mut projection = vec![0.0f32; VQ_DIM * HIDDEN];
        for row in 0..VQ_DIM {
            projection[row * HIDDEN + row] = 1.0;
        }
        projection
    }
}
