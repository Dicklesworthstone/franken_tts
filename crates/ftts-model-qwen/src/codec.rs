//! Safe f32 Qwen3-TTS 12 Hz speech-tokenizer decoder primitives.
//!
//! This is the correctness baseline for the codec path: SplitRVQ dequantization, the eight-layer
//! sliding-window pre-transformer, causal convolution, and SnakeBeta. The shipped checkpoint keeps
//! codec weights in f32, including the RVQ codebooks, so this module deliberately does no
//! quantization. Optimized kernels must reproduce this arithmetic before they become selectable.

use ftts_kernels::f32ref;
use ftts_kernels::int8::{KernelPlanV0, QuantizedMatrix, autotuned_plan, linear_q8_dynamic};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

/// The number of codec groups emitted per 80 ms frame.
pub const CODEC_GROUPS: usize = 16;
/// All live codec codebooks have this many entries; `semantic_codebook_size = 4096` is dead config.
pub const CODEBOOK_SIZE: usize = 2_048;
/// Residual-vector codebook width. It is not the 512-wide codec latent.
pub const CODEBOOK_DIM: usize = 256;
/// The codec latent width after each SplitRVQ output projection.
pub const CODEC_LATENT_DIM: usize = 512;
/// The pre-transformer's input/output width before its input projection.
pub const PRE_TRANSFORMER_LATENT_DIM: usize = 1_024;
/// The pre-transformer's hidden width.
pub const PRE_TRANSFORMER_HIDDEN_DIM: usize = 512;
/// The pre-transformer's attention width, deliberately wider than its hidden width.
pub const PRE_TRANSFORMER_ATTENTION_DIM: usize = 1_024;
/// The fixed attention window retained by every codec transformer layer.
pub const PRE_TRANSFORMER_WINDOW: usize = 72;
/// Codec RMSNorm epsilon. This differs from the talker epsilon.
pub const CODEC_RMS_NORM_EPS: f32 = 1e-5;
/// The codec's independent, plain RoPE base.
pub const CODEC_ROPE_THETA: f32 = 10_000.0;
/// Codebook normalisation floor from `EuclideanCodebook.decode`.
pub const CODEBOOK_USAGE_EPSILON: f32 = 1e-5;
/// `decoder.0`: 1024 input channels times a width-seven convolution.
pub const CODEC_MAX_INT8_REDUCTION_K: usize = 7_168;
/// Saturated U8S8 accumulation at the binding codec reduction length.
pub const CODEC_MAX_U8S8_ACCUMULATOR: i64 =
    u8::MAX as i64 * i8::MAX as i64 * CODEC_MAX_INT8_REDUCTION_K as i64;

/// Exact f32 geometry for the Qwen3-TTS 12 Hz codec decoder.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CodecConfig {
    /// Code groups per frame.
    pub groups: usize,
    /// Entries per codebook.
    pub codebook_size: usize,
    /// Codebook vector width.
    pub codebook_dim: usize,
    /// SplitRVQ output width.
    pub codec_latent_dim: usize,
    /// Pre-transformer input/output width.
    pub transformer_latent_dim: usize,
    /// Pre-transformer hidden width.
    pub transformer_hidden_dim: usize,
    /// Number of equal Q and KV heads; codec attention is not GQA.
    pub attention_heads: usize,
    /// Per-head width.
    pub attention_head_dim: usize,
    /// SwiGLU intermediate width.
    pub transformer_intermediate_dim: usize,
    /// Retained causal attention window.
    pub sliding_window: usize,
    /// Weight-only RMSNorm epsilon.
    pub rms_norm_eps: f32,
}

use crate::taps::{tap_emit, tap_hash_f32, taps_active};

impl Default for CodecConfig {
    fn default() -> Self {
        Self {
            groups: CODEC_GROUPS,
            codebook_size: CODEBOOK_SIZE,
            codebook_dim: CODEBOOK_DIM,
            codec_latent_dim: CODEC_LATENT_DIM,
            transformer_latent_dim: PRE_TRANSFORMER_LATENT_DIM,
            transformer_hidden_dim: PRE_TRANSFORMER_HIDDEN_DIM,
            attention_heads: 16,
            attention_head_dim: 64,
            transformer_intermediate_dim: 1_024,
            sliding_window: PRE_TRANSFORMER_WINDOW,
            rms_norm_eps: CODEC_RMS_NORM_EPS,
        }
    }
}

impl CodecConfig {
    /// Total Q/K/V projection width.
    #[must_use]
    pub const fn attention_dim(self) -> usize {
        self.attention_heads * self.attention_head_dim
    }
}

/// A named refusal from codec input or weight validation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CodecError {
    /// An unnormalised codebook does not match the configured geometry.
    CodebookShape,
    /// A codebook usage row is non-finite and therefore cannot be materialised safely.
    NonFiniteClusterUsage { row: usize },
    /// A codec id lies outside the sole valid range, `0..CODEBOOK_SIZE`.
    InvalidCode {
        frame: usize,
        group: usize,
        code: i32,
    },
    /// A caller supplied a code layout inconsistent with the exact 16-group decoder.
    CodeLayout,
    /// A requested U8S8 reduction cannot safely accumulate in i32.
    I32AccumulatorOverflow { reduction_k: usize },
}

/// A high-precision codebook materialised from upstream's unnormalised storage.
///
/// The checkpoint stores `embedding_sum` rather than lookup-ready vectors. Materialising the
/// division once preserves the reference's `clamp(cluster_usage, 1e-5)` semantics while keeping
/// the decode loop free of an avoidable branch and division per emitted code.
#[derive(Clone, Debug, PartialEq)]
pub struct MaterializedCodebook {
    entries: Vec<f32>,
    rows: usize,
    dim: usize,
}

impl MaterializedCodebook {
    /// Materialise `embedding_sum / max(cluster_usage, 1e-5)` row by row.
    pub fn from_unnormalized(
        embedding_sum: &[f32],
        cluster_usage: &[f32],
        rows: usize,
        dim: usize,
    ) -> Result<Self, CodecError> {
        if embedding_sum.len() != rows * dim || cluster_usage.len() != rows {
            return Err(CodecError::CodebookShape);
        }
        let mut entries = vec![0.0f32; embedding_sum.len()];
        for row in 0..rows {
            let usage = cluster_usage[row];
            if !usage.is_finite() {
                return Err(CodecError::NonFiniteClusterUsage { row });
            }
            let divisor = usage.max(CODEBOOK_USAGE_EPSILON);
            let input = &embedding_sum[row * dim..row * dim + dim];
            let output = &mut entries[row * dim..row * dim + dim];
            for index in 0..dim {
                output[index] = input[index] / divisor;
            }
        }
        Ok(Self { entries, rows, dim })
    }

    /// Add one code vector to `output`.
    fn add_code(
        &self,
        code: i32,
        frame: usize,
        group: usize,
        output: &mut [f32],
    ) -> Result<(), CodecError> {
        if code < 0
            || usize::try_from(code)
                .ok()
                .is_none_or(|entry| entry >= self.rows)
            || output.len() != self.dim
        {
            return Err(CodecError::InvalidCode { frame, group, code });
        }
        let entry = code as usize;
        let source = &self.entries[entry * self.dim..entry * self.dim + self.dim];
        for index in 0..self.dim {
            output[index] += source[index];
        }
        Ok(())
    }

    /// The materialised lookup table as `[rows, dim]`.
    #[must_use]
    pub fn entries(&self) -> &[f32] {
        &self.entries
    }
}

/// Borrowed SplitRVQ decode weights. The first branch handles group zero; the rest handles groups 1–15.
#[derive(Clone, Copy, Debug)]
pub struct SplitResidualVectorQuantizer<'a> {
    /// The semantic group-zero codebook.
    pub first_codebook: &'a MaterializedCodebook,
    /// The fifteen acoustic codebooks, in code-group order.
    pub rest_codebooks: &'a [MaterializedCodebook],
    /// `rvq_first.output_proj.weight`, `[512, 256]`, bias-free.
    pub first_output_proj: &'a [f32],
    /// `rvq_rest.output_proj.weight`, `[512, 256]`, bias-free.
    pub rest_output_proj: &'a [f32],
}

impl SplitResidualVectorQuantizer<'_> {
    /// Decode `[frames, 16]` identity-mapped codec ids to `[frames, 512]` latents.
    ///
    /// The two output-projected branches are added, exactly as the upstream split quantizer does.
    pub fn decode(
        &self,
        config: CodecConfig,
        codes: &[i32],
        frames: usize,
        output: &mut [f32],
    ) -> Result<(), CodecError> {
        if codes.len() != frames * config.groups
            || self.rest_codebooks.len() != config.groups.saturating_sub(1)
            || self.first_codebook.rows != config.codebook_size
            || self.first_codebook.dim != config.codebook_dim
            || self.first_output_proj.len() != config.codec_latent_dim * config.codebook_dim
            || self.rest_output_proj.len() != config.codec_latent_dim * config.codebook_dim
            || output.len() != frames * config.codec_latent_dim
        {
            return Err(CodecError::CodeLayout);
        }

        let mut first = vec![0.0f32; frames * config.codebook_dim];
        let mut rest = vec![0.0f32; frames * config.codebook_dim];
        for frame in 0..frames {
            self.first_codebook.add_code(
                codes[frame * config.groups],
                frame,
                0,
                &mut first[frame * config.codebook_dim..(frame + 1) * config.codebook_dim],
            )?;
            for (index, codebook) in self.rest_codebooks.iter().enumerate() {
                codebook.add_code(
                    codes[frame * config.groups + index + 1],
                    frame,
                    index + 1,
                    &mut rest[frame * config.codebook_dim..(frame + 1) * config.codebook_dim],
                )?;
            }
        }

        // Both `output_proj` are `Conv1d` with kernel 1 in the checkpoint —
        // `decoder.quantizer.rvq_{first,rest}.output_proj.weight` carry the 3-D shape
        // `[512, 256, 1]`, not the 2-D shape an `nn.Linear` would. So they follow the convolution
        // rule measured in `causal_conv1d`: the reduction belongs to the platform BLAS. At kernel 1
        // the `im2col` unfolding is the identity, so the GEMM is issued directly on the codebook
        // sums; both are bias-free, so it runs at `beta = 0`. Off macOS this degrades to the scalar
        // reduction, which is what these calls did before.
        let mut first_projected = vec![0.0f32; output.len()];
        let mut rest_projected = vec![0.0f32; output.len()];
        f32ref::linear_with_accumulation(
            &first,
            self.first_output_proj,
            None,
            frames,
            config.codebook_dim,
            config.codec_latent_dim,
            if ftts_kernels::f32ref::canonical_norm_requested() {
                f32ref::F32LinearAccumulation::Scalar
            } else {
                f32ref::F32LinearAccumulation::AccelerateRowInvariant
            },
            &mut first_projected,
        );
        f32ref::linear_with_accumulation(
            &rest,
            self.rest_output_proj,
            None,
            frames,
            config.codebook_dim,
            config.codec_latent_dim,
            if ftts_kernels::f32ref::canonical_norm_requested() {
                f32ref::F32LinearAccumulation::Scalar
            } else {
                f32ref::F32LinearAccumulation::AccelerateRowInvariant
            },
            &mut rest_projected,
        );
        for index in 0..output.len() {
            output[index] = first_projected[index] + rest_projected[index];
        }
        Ok(())
    }
}

/// Fill the plain codec RoPE rows for one absolute position.
///
/// The reference uses `rotate_half`, so each pair-frequency row is duplicated into the two halves
/// of `cos` and `sin`. This is intentionally separate from both talker mRoPE and microdecoder RoPE.
pub fn codec_rope_rows(position: usize, head_dim: usize, cos: &mut [f32], sin: &mut [f32]) {
    assert!(
        head_dim.is_multiple_of(2),
        "codec RoPE requires an even head dimension"
    );
    assert_eq!(cos.len(), head_dim, "codec RoPE cos width");
    assert_eq!(sin.len(), head_dim, "codec RoPE sin width");
    let half = head_dim / 2;
    let canonical = ftts_kernels::f32ref::canonical_norm_requested();
    for index in 0..half {
        let angle = if canonical {
            let inv_freq =
                ftts_kernels::f32ref::canonical_rope_inv_freq(CODEC_ROPE_THETA, index, head_dim);
            position as f32 * inv_freq
        } else {
            let exponent = (2 * index) as f32 / head_dim as f32;
            // The reference forms `inv_freq = 1 / theta^(2i/d)` once and multiplies by the
            // position; dividing the position directly lands on different bits.
            let inv_freq = CODEC_ROPE_THETA.powf(exponent).recip();
            position as f32 * inv_freq
        };
        let (sine, cosine) = if canonical {
            ftts_kernels::f32ref::canonical_sin_cos_f32(angle)
        } else {
            (angle.sin(), angle.cos())
        };
        cos[index] = cosine;
        cos[index + half] = cosine;
        sin[index] = sine;
        sin[index + half] = sine;
    }
}

/// One codec pre-transformer layer's borrowed f32 weights.
#[derive(Clone, Copy, Debug)]
pub struct CodecTransformerLayerWeights<'a> {
    /// Input RMSNorm scale, `[512]`.
    pub input_layernorm: &'a [f32],
    /// Query, key, and value projections, each `[1024, 512]` and bias-free.
    pub q_proj: &'a [f32],
    pub k_proj: &'a [f32],
    pub v_proj: &'a [f32],
    /// Attention output projection, `[512, 1024]`, bias-free.
    pub o_proj: &'a [f32],
    /// Learned LayerScale on the attention branch, `[512]`.
    pub self_attn_layer_scale: &'a [f32],
    /// Post-attention RMSNorm scale, `[512]`.
    pub post_attention_layernorm: &'a [f32],
    /// Bias-free SwiGLU projections.
    pub gate_proj: &'a [f32],
    pub up_proj: &'a [f32],
    pub down_proj: &'a [f32],
    /// Learned LayerScale on the MLP branch, `[512]`.
    pub mlp_layer_scale: &'a [f32],
}

/// Bounded per-layer K/V state for causal codec decoding.
#[derive(Clone, Debug)]
pub struct CodecKvCache {
    keys: Vec<f32>,
    values: Vec<f32>,
    positions: usize,
    width: usize,
    capacity: usize,
}

impl CodecKvCache {
    /// Make an empty cache retaining at most the configured sliding window.
    #[must_use]
    pub fn new(config: CodecConfig) -> Self {
        Self {
            keys: Vec::with_capacity(config.sliding_window * config.attention_dim()),
            values: Vec::with_capacity(config.sliding_window * config.attention_dim()),
            positions: 0,
            width: config.attention_dim(),
            capacity: config.sliding_window,
        }
    }

    /// Number of live attention positions.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.positions
    }

    /// Whether no attention positions are live yet.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.positions == 0
    }

    /// Drop all positions while retaining allocation for the next utterance.
    pub fn clear(&mut self) {
        self.keys.clear();
        self.values.clear();
        self.positions = 0;
    }

    fn push(&mut self, key: &[f32], value: &[f32]) {
        assert_eq!(key.len(), self.width, "codec key width");
        assert_eq!(value.len(), self.width, "codec value width");
        if self.positions == self.capacity {
            self.keys.drain(..self.width);
            self.values.drain(..self.width);
            self.positions -= 1;
        }
        self.keys.extend_from_slice(key);
        self.values.extend_from_slice(value);
        self.positions += 1;
    }
}

/// Run a single causal pre-transformer step, mutating its 512-wide residual state.
///
/// Codec attention has 16 Q and 16 KV heads (no GQA, no QK-Norm). Both residual branches carry
/// mandatory LayerScale; omitting either produces plausible but fixture-divergent audio.
pub fn forward_codec_transformer_step(
    config: CodecConfig,
    weights: CodecTransformerLayerWeights<'_>,
    position: usize,
    state: &mut [f32],
    cache: &mut CodecKvCache,
) {
    let hidden = config.transformer_hidden_dim;
    let attention = config.attention_dim();
    let intermediate = config.transformer_intermediate_dim;
    assert_eq!(state.len(), hidden, "codec transformer state width");
    assert_eq!(cache.width, attention, "codec KV cache geometry");
    assert_eq!(
        cache.capacity, config.sliding_window,
        "codec KV cache window"
    );

    // Lanes-4 squared-sum and divide-normalized softmax measurably track the reference's CPU
    // reductions closest on this tier (talker layer-00 bisect evidence, mirrored by the codec L2
    // ratchet); scalar forms sit an order of magnitude further from the oracle.
    let mut normed = vec![0.0f32; hidden];
    f32ref::rms_norm_with_arithmetic(
        state,
        weights.input_layernorm,
        config.rms_norm_eps,
        1,
        hidden,
        f32ref::F32RmsNormArithmetic::Lanes4ReciprocalSqrt,
        &mut normed,
    );
    let mut query = vec![0.0f32; attention];
    let mut key = vec![0.0f32; attention];
    let mut value = vec![0.0f32; attention];
    dense_linear(
        DenseClass::Transformer,
        &normed,
        weights.q_proj,
        None,
        1,
        hidden,
        attention,
        &mut query,
    );
    dense_linear(
        DenseClass::Transformer,
        &normed,
        weights.k_proj,
        None,
        1,
        hidden,
        attention,
        &mut key,
    );
    dense_linear(
        DenseClass::Transformer,
        &normed,
        weights.v_proj,
        None,
        1,
        hidden,
        attention,
        &mut value,
    );

    let mut cos = vec![0.0f32; config.attention_head_dim];
    let mut sin = vec![0.0f32; config.attention_head_dim];
    codec_rope_rows(position, config.attention_head_dim, &mut cos, &mut sin);
    for head in 0..config.attention_heads {
        let offset = head * config.attention_head_dim;
        f32ref::apply_rope_in_place(
            &mut query[offset..offset + config.attention_head_dim],
            &cos,
            &sin,
        );
        f32ref::apply_rope_in_place(
            &mut key[offset..offset + config.attention_head_dim],
            &cos,
            &sin,
        );
    }
    cache.push(&key, &value);

    let mut context = vec![0.0f32; attention];
    let mut scores = vec![0.0f32; cache.len()];
    let scaling = (config.attention_head_dim as f32).sqrt().recip();
    for head in 0..config.attention_heads {
        let head_offset = head * config.attention_head_dim;
        for (cache_position, score) in scores.iter_mut().enumerate() {
            let cached_key = &cache.keys[cache_position * attention + head_offset..]
                [..config.attention_head_dim];
            let mut dot = 0.0f32;
            for lane in 0..config.attention_head_dim {
                dot += query[head_offset + lane] * cached_key[lane];
            }
            *score = dot * scaling;
        }
        f32ref::softmax_rows_with_arithmetic(
            &mut scores,
            1,
            cache.len(),
            f32ref::F32SoftmaxArithmetic::Divide,
        );
        let target = &mut context[head_offset..head_offset + config.attention_head_dim];
        target.fill(0.0);
        for (cache_position, &score) in scores.iter().enumerate() {
            let cached_value = &cache.values[cache_position * attention + head_offset..]
                [..config.attention_head_dim];
            for lane in 0..config.attention_head_dim {
                target[lane] += score * cached_value[lane];
            }
        }
    }

    let mut attention_output = vec![0.0f32; hidden];
    dense_linear(
        DenseClass::Transformer,
        &context,
        weights.o_proj,
        None,
        1,
        attention,
        hidden,
        &mut attention_output,
    );
    for index in 0..hidden {
        state[index] += weights.self_attn_layer_scale[index] * attention_output[index];
    }

    f32ref::rms_norm_with_arithmetic(
        state,
        weights.post_attention_layernorm,
        config.rms_norm_eps,
        1,
        hidden,
        f32ref::F32RmsNormArithmetic::Lanes4ReciprocalSqrt,
        &mut normed,
    );
    let mut gate = vec![0.0f32; intermediate];
    let mut up = vec![0.0f32; intermediate];
    dense_linear(
        DenseClass::Transformer,
        &normed,
        weights.gate_proj,
        None,
        1,
        hidden,
        intermediate,
        &mut gate,
    );
    dense_linear(
        DenseClass::Transformer,
        &normed,
        weights.up_proj,
        None,
        1,
        hidden,
        intermediate,
        &mut up,
    );
    f32ref::silu_mul_in_place(&mut gate, &up);
    let mut mlp_output = vec![0.0f32; hidden];
    dense_linear(
        DenseClass::Transformer,
        &gate,
        weights.down_proj,
        None,
        1,
        intermediate,
        hidden,
        &mut mlp_output,
    );
    for index in 0..hidden {
        state[index] += weights.mlp_layer_scale[index] * mlp_output[index];
    }
}

/// Borrowed pre-transformer weights surrounding its eight causal layers.
#[derive(Clone, Debug)]
pub struct CodecPreTransformerWeights<'a> {
    /// Biased 1024→512 input projection.
    pub input_proj: &'a [f32],
    pub input_proj_bias: &'a [f32],
    /// Eight sliding-window layers.
    pub layers: &'a [CodecTransformerLayerWeights<'a>],
    /// Final RMSNorm scale.
    pub final_norm: &'a [f32],
    /// Biased 512→1024 output projection.
    pub output_proj: &'a [f32],
    pub output_proj_bias: &'a [f32],
}

/// Decode codec latents through the streaming pre-transformer.
///
/// `caches` is reused across packets. Calling this once per frame or once for the whole utterance
/// executes the same per-frame operations in the same order.
pub fn forward_codec_pre_transformer(
    config: CodecConfig,
    weights: &CodecPreTransformerWeights<'_>,
    start_position: usize,
    input: &[f32],
    frames: usize,
    caches: &mut [CodecKvCache],
    output: &mut [f32],
) {
    assert_eq!(
        input.len(),
        frames * config.transformer_latent_dim,
        "codec pre-transformer input"
    );
    assert_eq!(
        output.len(),
        frames * config.transformer_latent_dim,
        "codec pre-transformer output"
    );
    assert_eq!(
        weights.layers.len(),
        caches.len(),
        "one cache per codec transformer layer"
    );
    assert_eq!(
        weights.input_proj.len(),
        config.transformer_hidden_dim * config.transformer_latent_dim
    );
    assert_eq!(weights.input_proj_bias.len(), config.transformer_hidden_dim);
    assert_eq!(weights.final_norm.len(), config.transformer_hidden_dim);
    assert_eq!(
        weights.output_proj.len(),
        config.transformer_latent_dim * config.transformer_hidden_dim
    );
    assert_eq!(
        weights.output_proj_bias.len(),
        config.transformer_latent_dim
    );
    let mut hidden = vec![0.0f32; frames * config.transformer_hidden_dim];
    dense_linear(
        DenseClass::Transformer,
        input,
        weights.input_proj,
        Some(weights.input_proj_bias),
        frames,
        config.transformer_latent_dim,
        config.transformer_hidden_dim,
        &mut hidden,
    );
    for frame in 0..frames {
        let state = &mut hidden
            [frame * config.transformer_hidden_dim..(frame + 1) * config.transformer_hidden_dim];
        for (layer, cache) in weights.layers.iter().copied().zip(caches.iter_mut()) {
            forward_codec_transformer_step(config, layer, start_position + frame, state, cache);
        }
    }
    let mut normed = vec![0.0f32; hidden.len()];
    f32ref::rms_norm_with_arithmetic(
        &hidden,
        weights.final_norm,
        config.rms_norm_eps,
        frames,
        config.transformer_hidden_dim,
        f32ref::F32RmsNormArithmetic::Lanes4ReciprocalSqrt,
        &mut normed,
    );
    dense_linear(
        DenseClass::Transformer,
        &normed,
        weights.output_proj,
        Some(weights.output_proj_bias),
        frames,
        config.transformer_hidden_dim,
        config.transformer_latent_dim,
        output,
    );
}

/// The armed codec W8A8 route: dispatch tier plus a per-weight quantization memo.
///
/// The memo is keyed by the weight slice's address and length, with the first and last element
/// bits stored as a fingerprint: checkpoint tensors live for the process, but if an entry's
/// fingerprint ever disagrees (a dropped checkpoint's allocation reused), the entry is
/// requantized rather than trusted. This stays a private implementation detail of
/// [`dense_linear`]; the artifact-fed `.fttspack` route replaces it when that lands.
/// One memoized quantization: the weight's first/last element bits plus its Q8 form.
type QuantMemoEntry = Arc<(u32, u32, QuantizedMatrix)>;

struct CodecInt8Route {
    plan: KernelPlanV0,
    transformer: bool,
    convnext: bool,
    memo: Mutex<HashMap<(usize, usize, usize, usize), QuantMemoEntry>>,
}

/// Which family of dense projections a [`dense_linear`] call belongs to.
///
/// Kept coarse on purpose: the two families run at very different sample rates (the transformer
/// at 12.5 Hz latent rate, ConvNeXt inside the upsample cascade approaching 24 kHz), so they
/// differ in both cost share and quantization sensitivity. `FTTS_INT8_CODEC` accepts
/// `1`/`all`, `transformer`, or `convnext` so the two can be measured separately.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum DenseClass {
    /// Pre-transformer projections and the eight transformer layers.
    Transformer,
    /// ConvNeXt pointwise projections inside the upsample stages.
    ConvNext,
}

/// `FTTS_INT8_CODEC=1` arms W8A8 for the codec's dense projections (staged lever 4).
///
/// Default off: the scalar f32 reduction is the oracle-parity route (see the negative-evidence
/// note above [`causal_conv1d`]). Armed, the dense projections trade that exactness for int8
/// speed; the convolutions keep their parity-pinned BLAS route either way, and the RVQ
/// codebooks are gathers, untouched by any quantization. Read once per process.
fn codec_int8_route() -> Option<&'static CodecInt8Route> {
    static ROUTE: OnceLock<Option<CodecInt8Route>> = OnceLock::new();
    ROUTE
        .get_or_init(|| {
            // Default: NO codec int8 — the dense fall-through's named BLAS reduction measured
            // faster than the convnext int8 arm in every interleaved round (20.5/21.1/21.2 vs
            // 21.1/22.1/22.1 ms/frame, 2026-08-09) while tracking the oracle's own addmm
            // arithmetic more closely (the codec_decode_l2 ratchet tightened at all eight
            // transformer seams). Quantizing the codec now costs speed AND fidelity; the arms
            // stay opt-in for A/B.
            let env_value = std::env::var("FTTS_INT8_CODEC").ok();
            let (transformer, convnext) = codec_int8_route_decision(
                ftts_kernels::route::reference_pinned(),
                env_value.as_deref(),
            )?;
            Some(CodecInt8Route {
                plan: autotuned_plan(),
                transformer,
                convnext,
                memo: Mutex::new(HashMap::new()),
            })
        })
        .as_ref()
}

/// The `FTTS_INT8_CODEC` arming decision, as a pure function so its full matrix is testable.
///
/// A pinned reference route (every conformance entry point pins before synthesizing — see
/// [`ftts_kernels::route::pin_reference`]) always disarms the codec: the oracle comparison must
/// never measure the optimized arithmetic it exists to certify against (frankentts-7ozx: this
/// used to read the variable raw, so an exported `FTTS_INT8_CODEC` silently armed the codec
/// inside oracle-pinned suites and moved the ratchet's seams). One default on every target:
/// no codec int8 unless asked for.
fn codec_int8_route_decision(
    reference_pinned: bool,
    env_value: Option<&str>,
) -> Option<(bool, bool)> {
    if reference_pinned {
        return None;
    }
    match env_value {
        Some("1" | "all") => Some((true, true)),
        Some("transformer") => Some((true, false)),
        Some("convnext") => Some((false, true)),
        _ => None,
    }
}

/// Every dense (non-convolution) codec projection goes through here.
///
/// Unarmed, this is byte-for-byte `f32ref::linear` — the scalar reference reduction the parity
/// suite pins. Armed (`FTTS_INT8_CODEC=1`), the weight is quantized once (canonical symmetric
/// per-output-channel Q8) and the activations per row, so streaming and offline decode still
/// agree bit-for-bit: per-row activation quantization is batching-invariant.
#[allow(clippy::too_many_arguments)]
fn dense_linear(
    class: DenseClass,
    x: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    m: usize,
    k: usize,
    n: usize,
    out: &mut [f32],
) {
    let armed = codec_int8_route().filter(|route| match class {
        DenseClass::Transformer => route.transformer,
        DenseClass::ConvNext => route.convnext,
    });
    let Some(route) = armed else {
        // The reference's dense projections are `nn.Linear` — torch `addmm`, a bias-seeded
        // `beta = 1` BLAS GEMM. The convolution sites measured that exact named form byte-exact
        // against the oracle while every scalar/laned reduction diverged (see the GEMM-bisect
        // table above `causal_conv`), so the f32 dense fall-through asks for the same reduction
        // by name. Off macOS the request degrades to the scalar reference — correct, not exact.
        f32ref::linear_with_accumulation(
            x,
            weight,
            bias,
            m,
            k,
            n,
            if ftts_kernels::f32ref::canonical_norm_requested() {
                f32ref::F32LinearAccumulation::Scalar
            } else {
                f32ref::F32LinearAccumulation::AccelerateBiasSeededRowInvariant
            },
            out,
        );
        return;
    };

    // The shape belongs in the key: an address can be reused by a different tensor (observed
    // with constant-filled tiny test weights whose fingerprints also matched), and a memo hit
    // with the wrong (n, k) would reinterpret the matrix.
    let key = (weight.as_ptr() as usize, weight.len(), n, k);
    let fingerprint = (
        weight.first().map_or(0, |value| value.to_bits()),
        weight.last().map_or(0, |value| value.to_bits()),
    );
    let cached = {
        let memo = route.memo.lock().expect("codec int8 memo poisoned");
        memo.get(&key).cloned()
    };
    let quantized = match cached {
        Some(entry) if (entry.0, entry.1) == fingerprint => entry,
        _ => {
            let entry = Arc::new((
                fingerprint.0,
                fingerprint.1,
                QuantizedMatrix::quantize(weight, n, k),
            ));
            route
                .memo
                .lock()
                .expect("codec int8 memo poisoned")
                .insert(key, Arc::clone(&entry));
            entry
        }
    };
    // Offline decode arrives with m = frames (batch regime); streaming pushes small m.
    let tier = if m > 4 {
        route.plan.batch_gemm
    } else {
        route.plan.decode_gemv
    };
    linear_q8_dynamic(x, &quantized.2, bias, m, out, tier);
}

/// BigVGAN SnakeBeta in time-major `[frames, channels]` storage.
///
/// Upstream stores `alpha` and `beta` as logs. Both must be exponentiated before the reciprocal.
pub fn snake_beta_in_place(values: &mut [f32], frames: usize, alpha_log: &[f32], beta_log: &[f32]) {
    let channels = alpha_log.len();
    assert_eq!(beta_log.len(), channels, "SnakeBeta beta width");
    assert_eq!(values.len(), frames * channels, "SnakeBeta values shape");
    // Canonical parity mode runs the gate-aware scalar walk on every engine;
    // the SLEEF fast path is a default-route-only lever.
    if !ftts_kernels::f32ref::canonical_norm_requested() && fast_snake_armed() {
        snake_beta_fast(values, frames, alpha_log, beta_log);
        return;
    }
    let canonical = ftts_kernels::f32ref::canonical_norm_requested();
    let exp = |v: f32| -> f32 {
        if canonical {
            ftts_kernels::f32ref::canonical_exp_f32(v)
        } else {
            v.exp()
        }
    };
    let sine_of = |v: f32| -> f32 {
        if canonical {
            ftts_kernels::f32ref::canonical_sin_cos_f32(v).0
        } else {
            v.sin()
        }
    };
    for channel in 0..channels {
        let alpha = exp(alpha_log[channel]);
        let beta = exp(beta_log[channel]);
        let scale = (beta + 1e-9).recip();
        for frame in 0..frames {
            let index = frame * channels + channel;
            let sine = sine_of(values[index] * alpha);
            // The reference squares `sin` first (`pow(sin, 2)`) and then applies the reciprocal;
            // associating the scale into the first product lands on different bits.
            values[index] += scale * (sine * sine);
        }
    }
}

/// `FTTS_FAST_SNAKE=1` swaps SnakeBeta's system-libm `sin` for the crate's SLEEF-shaped 1-ulp
/// polynomial and walks the data frame-major (unit stride) with precomputed per-channel
/// constants. Default off: the scalar libm walk above is the parity reference. Read once.
fn fast_snake_armed() -> bool {
    static ARMED: OnceLock<bool> = OnceLock::new();
    *ARMED.get_or_init(|| match std::env::var("FTTS_FAST_SNAKE").as_deref() {
        Ok("1" | "on" | "true") => !ftts_kernels::route::reference_pinned(),
        Ok("0" | "off" | "false") => false,
        // Unset: follow the master optimized-route default.
        _ => ftts_kernels::route::optimized_default("FTTS_INT8"),
    })
}

fn snake_beta_fast(values: &mut [f32], frames: usize, alpha_log: &[f32], beta_log: &[f32]) {
    let channels = alpha_log.len();
    // Per-channel constants once, instead of once per (channel, frame) pair.
    let canonical = ftts_kernels::f32ref::canonical_norm_requested();
    let alpha: Vec<f32> = alpha_log
        .iter()
        .map(|log| {
            if canonical {
                ftts_kernels::f32ref::canonical_exp_f32(*log)
            } else {
                log.exp()
            }
        })
        .collect();
    let scale: Vec<f32> = beta_log
        .iter()
        .map(|log| {
            let beta = if canonical {
                ftts_kernels::f32ref::canonical_exp_f32(*log)
            } else {
                log.exp()
            };
            (beta + 1e-9).recip()
        })
        .collect();
    for frame in 0..frames {
        let row = &mut values[frame * channels..(frame + 1) * channels];
        for ((value, &a), &s) in row.iter_mut().zip(&alpha).zip(&scale) {
            let sine = ftts_kernels::sleef::sinf_u10(*value * a);
            *value += s * (sine * sine);
        }
    }
}

// NEGATIVE EVIDENCE — the platform-BLAS reduction does NOT generalize from the convolutions to the
// codec's dense projections, and the difference is op-shaped rather than bias-shaped.
//
// The two convolutions below reach exact by issuing the reference's own GEMM (see `causal_conv1d`).
// The obvious next step is to route every `f32ref::linear` here the same way, on the reasoning that
// a biased `nn.Linear` lowers to `addmm` — also a `beta = 1` bias-seeded BLAS call. That was tried
// against the pinned oracle and REJECTED; it moved seams the wrong way:
//
//   bias-free projections (transformer q/k/v/o, gate/up/down), routed at `beta = 0`:
//     transformer_layer_02   2.861e-6 -> 3.815e-6
//     transformer_layer_03   2.384e-7 -> 7.749e-7
//     transformer_layer_05   1.192e-7 -> 8.941e-7
//   biased `nn.Linear` (ConvNeXt pwconv1/pwconv2), routed bias-seeded at `beta = 1`:
//     upsample_1_1           2.480e-5 -> 2.861e-5
//
// So "carries a bias" is the wrong discriminator. `Conv1d`/`ConvTranspose1d` go through
// `slow_conv2d`/`slow_conv_transpose2d`, whose GEMM shape and blocking the convolutions below now
// reproduce exactly; `nn.Linear` does not land on that same blocking, and why is still open — a
// batched 3-D `baddbmm` lowering is the leading suspect. Until that is measured, the dense
// projections keep the scalar reduction, which is the closer of the two.
//
// That earlier run left one question open — whether the `k = 1` projections follow the conv rule or
// the `nn.Linear` rule — because it moved them together with the dense sites. The pinned checkpoint
// settles which is which: `decoder.quantizer.rvq_{first,rest}.output_proj.weight` carry the 3-D
// shape `[512, 256, 1]` and so are `Conv1d`, whereas `decoder.pre_transformer.{input,output}_proj`
// are 2-D and so are genuinely `nn.Linear`. Routing ONLY the two RVQ projections at `beta = 0`
// (below, in `SplitResidualVectorQuantizer::decode`) took `rvq+pre_conv+input_proj` from 1.639e-7 to
// 5.960e-8 and left every other seam bit-identical, which is the isolation the earlier run lacked.
// `decode_codec_offline[xvector]` moved the wrong way by a ulp (4.005e-8 -> 5.402e-8) while
// `[icl]` improved (2.198e-7 -> 1.788e-7); both are pinned.

/// Reference causal Conv1d for time-major `[frames, channels]` data.
///
/// Weights retain PyTorch's `[out_channel, in_channel, kernel]` checkpoint layout.
// The argument list mirrors the reference convolution's checkpoint-facing signature; bundling
// them into a struct is a refactor for the codec engine bead, not the parity reference.
#[allow(clippy::too_many_arguments)]
pub fn causal_conv1d(
    input: &[f32],
    frames: usize,
    input_channels: usize,
    weight: &[f32],
    bias: Option<&[f32]>,
    output_channels: usize,
    kernel: usize,
    dilation: usize,
    output: &mut [f32],
) {
    assert_eq!(
        input.len(),
        frames * input_channels,
        "causal conv input shape"
    );
    assert_eq!(
        weight.len(),
        output_channels * input_channels * kernel,
        "causal conv weight shape"
    );
    assert_eq!(
        output.len(),
        frames * output_channels,
        "causal conv output shape"
    );
    assert!(
        kernel > 0 && dilation > 0,
        "causal conv geometry must be positive"
    );
    if let Some(bias) = bias {
        assert_eq!(bias.len(), output_channels, "causal conv bias shape");
    }
    // `slow_conv2d_update_output_frame` issues this convolution as ONE GEMM over an `im2col`
    // unfolding: columns ordered `[in_channel, tap]`-major and zero-padded on the left, with the
    // bias reaching the GEMM as its `beta = 1` accumulator seed rather than as a trailing add.
    //
    // Reproducing that call is what takes this seam to exact. Measured on the pinned oracle at
    // `codec_decoder.block_00` (1024 -> 1536, kernel 7, K = 7168, the codec's binding worst case),
    // over 86_016 elements:
    //
    //   scalar left-to-right dot           max_abs 1.240e-5   83_663 over tolerance
    //   4-lane / 8-lane partial sums       max_abs 3.815e-5  ~84_300 over tolerance
    //   BLAS, bias added after `beta = 0`  max_abs 1.335e-5   83_351 over tolerance
    //   BLAS, bias seeded under `beta = 1` max_abs 0.0             0 over tolerance
    //
    // Only the last is exact, and the `beta = 0` row is what rules out "any BLAS will do": the
    // seeding is load-bearing, not incidental. The reduction order therefore belongs to the
    // platform BLAS, so `linear_with_accumulation` is asked for it by name. Off macOS that request
    // degrades to the scalar reduction, which is the first row above — correct, not exact. See
    // `ftts-conformance/tests/codec_gemm_bisect.rs`, which is the probe that measured this.
    let reduction = input_channels * kernel;
    let mut columns = vec![0.0f32; frames * reduction];
    for frame in 0..frames {
        for tap in 0..kernel {
            let past = (kernel - 1 - tap) * dilation;
            let Some(source_frame) = frame.checked_sub(past) else {
                // Left padding. The zero column must be materialized, not skipped: the reference's
                // GEMM reduces over the full `K`, and a shortened reduction blocks differently.
                continue;
            };
            let source = &input[source_frame * input_channels..][..input_channels];
            let target = &mut columns[frame * reduction..][..reduction];
            for (input_channel, &value) in source.iter().enumerate() {
                target[input_channel * kernel + tap] = value;
            }
        }
    }
    f32ref::linear_with_accumulation(
        &columns,
        weight,
        bias,
        frames,
        reduction,
        output_channels,
        if ftts_kernels::f32ref::canonical_norm_requested() {
            f32ref::F32LinearAccumulation::Scalar
        } else {
            f32ref::F32LinearAccumulation::AccelerateBiasSeededRowInvariant
        },
        output,
    );
}

/// Rearrange `[in_channel, out_channel, tap]` into the `[n, k]` the column GEMM wants.
///
/// A pure permutation — every value survives unchanged, so any caller that supplies the result
/// instead of re-deriving it gets bit-identical output by construction, not by tolerance.
///
/// Worth hoisting because the operand is INVARIANT: the same upsample weights were permuted once
/// per packet, roughly a dozen times per utterance, purely to be consumed identically each time.
#[must_use]
pub fn column_layout(
    weight: &[f32],
    input_channels: usize,
    output_channels: usize,
    kernel: usize,
) -> Vec<f32> {
    let mut column_weight = vec![0.0f32; output_channels * kernel * input_channels];
    for input_channel in 0..input_channels {
        for output_channel in 0..output_channels {
            for tap in 0..kernel {
                let source = (input_channel * output_channels + output_channel) * kernel + tap;
                let target = (output_channel * kernel + tap) * input_channels + input_channel;
                column_weight[target] = weight[source];
            }
        }
    }
    column_weight
}

/// Reference causal ConvTranspose1d for time-major data.
///
/// PyTorch stores transposed-convolution weights `[in_channel, out_channel, kernel]`. The Qwen
/// causal wrapper trims exactly `kernel - stride` samples from the right, leaving `frames * stride`.
#[allow(clippy::too_many_arguments)]
pub fn causal_transpose_conv1d(
    input: &[f32],
    frames: usize,
    input_channels: usize,
    weight: &[f32],
    bias: Option<&[f32]>,
    output_channels: usize,
    kernel: usize,
    stride: usize,
    output: &mut [f32],
) {
    assert_eq!(
        input.len(),
        frames * input_channels,
        "causal tconv input shape"
    );
    assert_eq!(
        weight.len(),
        input_channels * output_channels * kernel,
        "causal tconv weight shape"
    );
    assert_eq!(
        output.len(),
        frames * stride * output_channels,
        "causal tconv output shape"
    );
    assert!(kernel >= stride && stride > 0, "causal tconv geometry");
    if let Some(bias) = bias {
        assert_eq!(bias.len(), output_channels, "causal tconv bias shape");
    }
    // `slow_conv_transpose2d` reduces in two levels: ONE GEMM over the input channels materializes
    // the whole `[frames, output_channel * tap]` column tensor, then `col2im` sums those columns
    // into the output in ascending tap order. Bias is added only after both levels, matching the
    // reference's trailing `output.add_(bias)` — unlike the forward convolution above, this GEMM
    // runs at `beta = 0` with no bias in it.
    //
    // The column GEMM's reduction order belongs to the platform BLAS for the same measured reason
    // the forward convolution's does (see `causal_conv1d`); a scalar left-to-right sum over the
    // input channels is correct but not bit-exact. Off macOS this degrades to that scalar sum.
    //
    // The reference's weight is `[in_channel, out_channel, tap]`, which is `[k, n]`; the GEMM
    // helper wants `[n, k]`, so the columns' operand is transposed once up front.
    let column_weight = column_layout(weight, input_channels, output_channels, kernel);
    causal_transpose_conv1d_columns(
        input,
        frames,
        input_channels,
        &column_weight,
        bias,
        output_channels,
        kernel,
        stride,
        output,
    );
}

/// [`causal_transpose_conv1d`] for a caller that already holds the column layout.
///
/// Identical arithmetic; the only difference is who owns the permutation. Streaming decode reuses
/// one layout for the whole utterance instead of rebuilding it per packet.
#[allow(clippy::too_many_arguments)]
pub fn causal_transpose_conv1d_columns(
    input: &[f32],
    frames: usize,
    input_channels: usize,
    column_weight: &[f32],
    bias: Option<&[f32]>,
    output_channels: usize,
    kernel: usize,
    stride: usize,
    output: &mut [f32],
) {
    let column_width = output_channels * kernel;
    let mut columns = vec![0.0f32; frames * column_width];
    f32ref::linear_with_accumulation(
        input,
        column_weight,
        None,
        frames,
        input_channels,
        column_width,
        if ftts_kernels::f32ref::canonical_norm_requested() {
            f32ref::F32LinearAccumulation::Scalar
        } else {
            f32ref::F32LinearAccumulation::AccelerateRowInvariant
        },
        &mut columns,
    );

    // `col2im`: sum each input frame's column into the output samples its taps land on, ascending.
    let kept_frames = frames * stride;
    for output_frame in 0..kept_frames {
        for output_channel in 0..output_channels {
            let mut total = 0.0f32;
            for tap in 0..kernel {
                // The input frame whose `tap` lands on this output sample, when one exists.
                let Some(shifted) = output_frame.checked_sub(tap) else {
                    continue;
                };
                if !shifted.is_multiple_of(stride) {
                    continue;
                }
                let frame = shifted / stride;
                if frame >= frames {
                    continue;
                }
                total += columns[frame * column_width + output_channel * kernel + tap];
            }
            output[output_frame * output_channels + output_channel] =
                bias.map_or(total, |values| total + values[output_channel]);
        }
    }
}

/// Persistent left context for one causal Conv1d stage.
#[derive(Clone, Debug)]
pub struct CausalConvStream {
    input_channels: usize,
    kernel: usize,
    dilation: usize,
    history: Vec<f32>,
}

impl CausalConvStream {
    /// Construct an empty stream. Its retained context is `(kernel - 1) * dilation` input frames.
    #[must_use]
    pub fn new(input_channels: usize, kernel: usize, dilation: usize) -> Self {
        assert!(
            input_channels > 0 && kernel > 0 && dilation > 0,
            "causal stream geometry"
        );
        Self {
            input_channels,
            kernel,
            dilation,
            history: Vec::with_capacity((kernel - 1) * dilation * input_channels),
        }
    }

    /// Process a packet and emit only its newly produced output frames.
    pub fn push(
        &mut self,
        input: &[f32],
        frames: usize,
        weight: &[f32],
        bias: Option<&[f32]>,
        output_channels: usize,
        output: &mut Vec<f32>,
    ) {
        assert_eq!(
            input.len(),
            frames * self.input_channels,
            "causal stream input shape"
        );
        let history_frames = self.history.len() / self.input_channels;
        let mut joined = Vec::with_capacity(self.history.len() + input.len());
        joined.extend_from_slice(&self.history);
        joined.extend_from_slice(input);
        let joined_frames = history_frames + frames;
        let mut all = vec![0.0f32; joined_frames * output_channels];
        causal_conv1d(
            &joined,
            joined_frames,
            self.input_channels,
            weight,
            bias,
            output_channels,
            self.kernel,
            self.dilation,
            &mut all,
        );
        output.clear();
        output.extend_from_slice(&all[history_frames * output_channels..]);
        let retain = ((self.kernel - 1) * self.dilation).min(joined_frames);
        self.history.clear();
        self.history
            .extend_from_slice(&joined[(joined_frames - retain) * self.input_channels..]);
    }

    /// Discard retained left context without releasing its allocation.
    pub fn clear(&mut self) {
        self.history.clear();
    }
}

/// Persistent causal right-trimmed ConvTranspose1d state.
///
/// The Qwen codec emits exactly `stride` finalized samples for every new input frame.  Contributions
/// to those samples come only from the current and earlier input frames, so the stream retains the
/// `(kernel - 1) / stride` input frames that can still reach a future sample and recomputes each
/// packet through [`causal_transpose_conv1d`].  Carrying *summed* partial outputs instead would
/// finalize each sample in the opposite tap order from the offline reduction and therefore land on
/// different bits; retaining the inputs keeps the two paths bit-identical.
#[derive(Clone, Debug)]
pub struct CausalTransposeConvStream {
    input_channels: usize,
    output_channels: usize,
    kernel: usize,
    stride: usize,
    /// The retained input frames, time-major `[retained, input_channels]`.
    history: Vec<f32>,
    /// The GEMM-ready weight permutation, derived once on first use and reused for the utterance.
    ///
    /// The weight is invariant, so rebuilding this per packet was pure repetition: roughly a
    /// dozen permutations of the same 8.4 MB upsample weight for one utterance, each feeding an
    /// identical GEMM. Cached here rather than in the checkpoint so the layout lives exactly as
    /// long as the decode that needs it, which keeps hydration's high-water mark untouched — the
    /// browser has no room to spare there.
    columns: Vec<f32>,
}

impl CausalTransposeConvStream {
    /// Construct an empty stream for one causal transposed convolution.
    #[must_use]
    pub fn new(
        input_channels: usize,
        output_channels: usize,
        kernel: usize,
        stride: usize,
    ) -> Self {
        assert!(
            input_channels > 0 && output_channels > 0 && kernel >= stride && stride > 0,
            "causal transposed stream geometry"
        );
        Self {
            input_channels,
            output_channels,
            kernel,
            stride,
            history: Vec::with_capacity(((kernel - 1) / stride) * input_channels),
            columns: Vec::new(),
        }
    }

    /// Discard retained left context without releasing its allocation.
    ///
    /// The cached column layout goes too. A cleared stream may legitimately be re-driven with a
    /// different weight, and a layout that outlived its weight would decode the next utterance
    /// with the previous one — silently, and with perfectly plausible audio.
    pub fn clear(&mut self) {
        self.history.clear();
        self.columns.clear();
    }

    /// Process a packet and write its finalized output frames to `output`.
    ///
    /// Weights retain PyTorch's `[in_channel, out_channel, kernel]` layout.  `output` receives
    /// time-major `[frames * stride, output_channels]` samples, exactly matching
    /// [`causal_transpose_conv1d`] on the concatenation of all packets seen so far.
    pub fn push(
        &mut self,
        input: &[f32],
        frames: usize,
        weight: &[f32],
        bias: Option<&[f32]>,
        output: &mut Vec<f32>,
    ) {
        assert_eq!(
            input.len(),
            frames * self.input_channels,
            "causal transposed stream input shape"
        );
        assert_eq!(
            weight.len(),
            self.input_channels * self.output_channels * self.kernel,
            "causal transposed stream weight shape"
        );
        if let Some(bias) = bias {
            assert_eq!(
                bias.len(),
                self.output_channels,
                "causal transposed stream bias shape"
            );
        }

        let history_frames = self.history.len() / self.input_channels;
        let mut joined = Vec::with_capacity(self.history.len() + input.len());
        joined.extend_from_slice(&self.history);
        joined.extend_from_slice(input);
        let joined_frames = history_frames + frames;
        let mut all = vec![0.0f32; joined_frames * self.stride * self.output_channels];
        // Derive the column layout once per stream. `is_empty` is the whole guard: a stream is
        // built for one utterance and driven with one weight, so the first packet fills this and
        // every later packet reuses it. The values are a permutation of the same weight either
        // way, so the GEMM sees identical operands and the output stays bit-identical to the
        // offline path — the streaming==batch gate still holds.
        if self.columns.is_empty() {
            self.columns = column_layout(
                weight,
                self.input_channels,
                self.output_channels,
                self.kernel,
            );
        }
        causal_transpose_conv1d_columns(
            &joined,
            joined_frames,
            self.input_channels,
            &self.columns,
            bias,
            self.output_channels,
            self.kernel,
            self.stride,
            &mut all,
        );
        output.clear();
        output.extend_from_slice(&all[history_frames * self.stride * self.output_channels..]);
        let retain = ((self.kernel - 1) / self.stride).min(joined_frames);
        self.history.clear();
        self.history
            .extend_from_slice(&joined[(joined_frames - retain) * self.input_channels..]);
    }
}

/// Persistent left context for one causal depthwise Conv1d stage.
#[derive(Clone, Debug)]
struct CausalDepthwiseConvStream {
    channels: usize,
    kernel: usize,
    history: Vec<f32>,
}

impl CausalDepthwiseConvStream {
    fn new(channels: usize, kernel: usize) -> Self {
        assert!(channels > 0 && kernel > 0, "depthwise stream geometry");
        Self {
            channels,
            kernel,
            history: Vec::with_capacity((kernel - 1) * channels),
        }
    }

    fn clear(&mut self) {
        self.history.clear();
    }

    fn push(
        &mut self,
        input: &[f32],
        frames: usize,
        weight: &[f32],
        bias: &[f32],
        output: &mut Vec<f32>,
    ) {
        assert_eq!(
            input.len(),
            frames * self.channels,
            "depthwise stream input shape"
        );
        let history_frames = self.history.len() / self.channels;
        let mut joined = Vec::with_capacity(self.history.len() + input.len());
        joined.extend_from_slice(&self.history);
        joined.extend_from_slice(input);
        let joined_frames = history_frames + frames;
        let mut all = vec![0.0f32; joined_frames * self.channels];
        causal_depthwise_conv1d(
            &joined,
            joined_frames,
            self.channels,
            weight,
            bias,
            self.kernel,
            &mut all,
        );
        output.clear();
        output.extend_from_slice(&all[history_frames * self.channels..]);
        let retain = (self.kernel - 1).min(joined_frames);
        self.history.clear();
        self.history
            .extend_from_slice(&joined[(joined_frames - retain) * self.channels..]);
    }
}

/// Refuse an int8 kernel whose U8S8 accumulation cannot fit in i32.
pub fn require_i32_safe_u8s8_reduction(reduction_k: usize) -> Result<(), CodecError> {
    let worst = i64::from(u8::MAX) * i64::from(i8::MAX) * reduction_k as i64;
    if worst > i64::from(i32::MAX) {
        return Err(CodecError::I32AccumulatorOverflow { reduction_k });
    }
    Ok(())
}

/// Borrowed causal Conv1d weights in their original PyTorch layout.
#[derive(Clone, Copy, Debug)]
pub struct CausalConvWeights<'a> {
    /// `[out_channel, in_channel, kernel]`.
    pub weight: &'a [f32],
    /// Conv1d bias, `[out_channel]`.
    pub bias: &'a [f32],
    pub input_channels: usize,
    pub output_channels: usize,
    pub kernel: usize,
    pub dilation: usize,
}

impl CausalConvWeights<'_> {
    fn forward(&self, input: &[f32], frames: usize) -> Vec<f32> {
        let mut output = vec![0.0f32; frames * self.output_channels];
        causal_conv1d(
            input,
            frames,
            self.input_channels,
            self.weight,
            Some(self.bias),
            self.output_channels,
            self.kernel,
            self.dilation,
            &mut output,
        );
        output
    }
}

/// Borrowed causal ConvTranspose1d weights in their original PyTorch layout.
#[derive(Clone, Copy, Debug)]
pub struct CausalTransposeConvWeights<'a> {
    /// `[in_channel, out_channel, kernel]`.
    pub weight: &'a [f32],
    /// ConvTranspose1d bias, `[out_channel]`.
    pub bias: &'a [f32],
    pub input_channels: usize,
    pub output_channels: usize,
    pub kernel: usize,
    pub stride: usize,
}

impl CausalTransposeConvWeights<'_> {
    fn forward(&self, input: &[f32], frames: usize) -> Vec<f32> {
        let mut output = vec![0.0f32; frames * self.stride * self.output_channels];
        causal_transpose_conv1d(
            input,
            frames,
            self.input_channels,
            self.weight,
            Some(self.bias),
            self.output_channels,
            self.kernel,
            self.stride,
            &mut output,
        );
        output
    }
}

/// One causal ConvNeXt block between the two latent upsample stages.
#[derive(Clone, Copy, Debug)]
pub struct CodecConvNextWeights<'a> {
    /// Depthwise Conv1d `[channels, 1, 7]` and its bias.
    pub depthwise_weight: &'a [f32],
    pub depthwise_bias: &'a [f32],
    /// Per-time-step LayerNorm parameters, each `[channels]`.
    pub norm_weight: &'a [f32],
    pub norm_bias: &'a [f32],
    /// Biased pointwise 1024→4096 and 4096→1024 linear layers.
    pub pwconv1: &'a [f32],
    pub pwconv1_bias: &'a [f32],
    pub pwconv2: &'a [f32],
    pub pwconv2_bias: &'a [f32],
    /// Learned residual gamma, `[channels]`.
    pub gamma: &'a [f32],
}

/// One `upsample.{0,1}` stage: causal transposed convolution then causal ConvNeXt.
#[derive(Clone, Copy, Debug)]
pub struct CodecUpsampleStageWeights<'a> {
    pub transposed: CausalTransposeConvWeights<'a>,
    pub convnext: CodecConvNextWeights<'a>,
}

/// One SnakeBeta → causal-conv → SnakeBeta → causal-conv residual unit.
#[derive(Clone, Copy, Debug)]
pub struct CodecResidualUnitWeights<'a> {
    pub first_alpha_log: &'a [f32],
    pub first_beta_log: &'a [f32],
    pub first_conv: CausalConvWeights<'a>,
    pub second_alpha_log: &'a [f32],
    pub second_beta_log: &'a [f32],
    pub second_conv: CausalConvWeights<'a>,
}

/// One of the four BigVGAN decoder upsampling blocks.
#[derive(Clone, Copy, Debug)]
pub struct CodecDecoderBlockWeights<'a> {
    pub alpha_log: &'a [f32],
    pub beta_log: &'a [f32],
    pub transposed: CausalTransposeConvWeights<'a>,
    /// The exactly three residual units at dilations 1, 3, and 9.
    pub residual_units: [CodecResidualUnitWeights<'a>; 3],
}

/// Complete borrowed f32 decoder weights after SplitRVQ.
#[derive(Clone, Debug)]
pub struct CodecDecoderWeights<'a> {
    /// Causal 512→1024 convolution before the transformer.
    pub pre_conv: CausalConvWeights<'a>,
    pub pre_transformer: CodecPreTransformerWeights<'a>,
    /// Two 2× latent upsample stages.
    pub latent_upsample: [CodecUpsampleStageWeights<'a>; 2],
    /// Causal 1024→1536 convolution before BigVGAN blocks.
    pub decoder_input: CausalConvWeights<'a>,
    /// 8×, 5×, 4×, 3× BigVGAN blocks.
    pub decoder_blocks: [CodecDecoderBlockWeights<'a>; 4],
    /// Final 96-channel SnakeBeta and causal 96→1 output convolution.
    pub final_alpha_log: &'a [f32],
    pub final_beta_log: &'a [f32],
    pub final_conv: CausalConvWeights<'a>,
}

/// Exact f32 GELU (`approximate='none'`), used by codec ConvNeXt blocks.
///
/// The reference scales by the constant `M_SQRT1_2` rather than dividing by `sqrt(2)`; the two land
/// on different bits because `1/sqrt(2)` is not exactly representable.
#[must_use]
pub fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + (x * core::f32::consts::FRAC_1_SQRT_2).erf())
}

fn causal_depthwise_conv1d(
    input: &[f32],
    frames: usize,
    channels: usize,
    weight: &[f32],
    bias: &[f32],
    kernel: usize,
    output: &mut [f32],
) {
    assert_eq!(input.len(), frames * channels, "depthwise input shape");
    assert_eq!(weight.len(), channels * kernel, "depthwise weight shape");
    assert_eq!(bias.len(), channels, "depthwise bias shape");
    assert_eq!(output.len(), frames * channels, "depthwise output shape");
    for frame in 0..frames {
        for channel in 0..channels {
            let mut total = bias[channel];
            for tap in 0..kernel {
                if let Some(source_frame) = frame.checked_sub(kernel - 1 - tap) {
                    total +=
                        input[source_frame * channels + channel] * weight[channel * kernel + tap];
                }
            }
            output[frame * channels + channel] = total;
        }
    }
}

fn layer_norm_in_place(values: &mut [f32], frames: usize, weight: &[f32], bias: &[f32], eps: f32) {
    let channels = weight.len();
    assert_eq!(bias.len(), channels, "LayerNorm bias width");
    assert_eq!(values.len(), frames * channels, "LayerNorm values shape");
    for frame in 0..frames {
        let row = &mut values[frame * channels..(frame + 1) * channels];
        let mut mean = 0.0f32;
        for value in row.iter() {
            mean += *value;
        }
        mean /= channels as f32;
        let mut variance = 0.0f32;
        for value in row.iter() {
            let centered = *value - mean;
            variance += centered * centered;
        }
        let scale = (variance / channels as f32 + eps).sqrt().recip();
        for channel in 0..channels {
            row[channel] = (row[channel] - mean) * scale * weight[channel] + bias[channel];
        }
    }
}

/// Run one causal ConvNeXt block. Public so the conformance harness can gate this exact seam.
#[must_use]
pub fn forward_convnext(
    input: &[f32],
    frames: usize,
    channels: usize,
    weights: CodecConvNextWeights<'_>,
) -> Vec<f32> {
    assert_eq!(input.len(), frames * channels, "ConvNeXt input shape");
    assert_eq!(weights.norm_weight.len(), channels, "ConvNeXt norm width");
    assert_eq!(weights.gamma.len(), channels, "ConvNeXt gamma width");
    let mut hidden = vec![0.0f32; input.len()];
    causal_depthwise_conv1d(
        input,
        frames,
        channels,
        weights.depthwise_weight,
        weights.depthwise_bias,
        7,
        &mut hidden,
    );
    layer_norm_in_place(
        &mut hidden,
        frames,
        weights.norm_weight,
        weights.norm_bias,
        1e-6,
    );
    let mut expanded = vec![0.0f32; frames * 4 * channels];
    dense_linear(
        DenseClass::ConvNext,
        &hidden,
        weights.pwconv1,
        Some(weights.pwconv1_bias),
        frames,
        channels,
        4 * channels,
        &mut expanded,
    );
    for value in &mut expanded {
        *value = gelu(*value);
    }
    dense_linear(
        DenseClass::ConvNext,
        &expanded,
        weights.pwconv2,
        Some(weights.pwconv2_bias),
        frames,
        4 * channels,
        channels,
        &mut hidden,
    );
    let mut output = input.to_vec();
    for frame in 0..frames {
        for channel in 0..channels {
            let index = frame * channels + channel;
            output[index] += weights.gamma[channel] * hidden[index];
        }
    }
    output
}

fn forward_residual_unit(
    input: &[f32],
    frames: usize,
    channels: usize,
    weights: CodecResidualUnitWeights<'_>,
) -> Vec<f32> {
    let mut hidden = input.to_vec();
    snake_beta_in_place(
        &mut hidden,
        frames,
        weights.first_alpha_log,
        weights.first_beta_log,
    );
    hidden = weights.first_conv.forward(&hidden, frames);
    snake_beta_in_place(
        &mut hidden,
        frames,
        weights.second_alpha_log,
        weights.second_beta_log,
    );
    hidden = weights.second_conv.forward(&hidden, frames);
    assert_eq!(
        hidden.len(),
        frames * channels,
        "residual unit output shape"
    );
    for index in 0..hidden.len() {
        hidden[index] += input[index];
    }
    hidden
}

/// Run one BigVGAN upsampling block, returning its output and new frame count.
/// Public so the conformance harness can gate this exact seam.
#[must_use]
pub fn forward_decoder_block(
    input: &[f32],
    frames: usize,
    input_channels: usize,
    weights: CodecDecoderBlockWeights<'_>,
) -> (Vec<f32>, usize) {
    let mut hidden = input.to_vec();
    snake_beta_in_place(&mut hidden, frames, weights.alpha_log, weights.beta_log);
    hidden = weights.transposed.forward(&hidden, frames);
    let output_frames = frames * weights.transposed.stride;
    let output_channels = weights.transposed.output_channels;
    assert_eq!(
        hidden.len(),
        output_frames * output_channels,
        "decoder tconv output shape"
    );
    assert_eq!(
        weights.transposed.input_channels, input_channels,
        "decoder tconv input width"
    );
    for unit in weights.residual_units {
        hidden = forward_residual_unit(&hidden, output_frames, output_channels, unit);
    }
    (hidden, output_frames)
}

/// Retained causal context for a codec ConvNeXt block.
#[derive(Clone, Debug)]
struct CodecConvNextStream {
    depthwise: CausalDepthwiseConvStream,
}

impl CodecConvNextStream {
    fn new(channels: usize) -> Self {
        Self {
            depthwise: CausalDepthwiseConvStream::new(channels, 7),
        }
    }

    fn clear(&mut self) {
        self.depthwise.clear();
    }

    fn push(
        &mut self,
        input: &[f32],
        frames: usize,
        weights: CodecConvNextWeights<'_>,
        output: &mut Vec<f32>,
    ) {
        let channels = weights.norm_weight.len();
        assert_eq!(
            input.len(),
            frames * channels,
            "streaming ConvNeXt input shape"
        );
        self.depthwise.push(
            input,
            frames,
            weights.depthwise_weight,
            weights.depthwise_bias,
            output,
        );
        layer_norm_in_place(output, frames, weights.norm_weight, weights.norm_bias, 1e-6);
        let mut expanded = vec![0.0f32; frames * 4 * channels];
        dense_linear(
            DenseClass::ConvNext,
            output,
            weights.pwconv1,
            Some(weights.pwconv1_bias),
            frames,
            channels,
            4 * channels,
            &mut expanded,
        );
        for value in &mut expanded {
            *value = gelu(*value);
        }
        let mut residual = vec![0.0f32; input.len()];
        dense_linear(
            DenseClass::ConvNext,
            &expanded,
            weights.pwconv2,
            Some(weights.pwconv2_bias),
            frames,
            4 * channels,
            channels,
            &mut residual,
        );
        output.clear();
        output.reserve(residual.len());
        for index in 0..residual.len() {
            output.push(input[index] + weights.gamma[index % channels] * residual[index]);
        }
    }
}

/// Retained causal context for one BigVGAN residual unit.
#[derive(Clone, Debug)]
struct CodecResidualUnitStream {
    first_conv: CausalConvStream,
    second_conv: CausalConvStream,
}

impl CodecResidualUnitStream {
    fn new(weights: CodecResidualUnitWeights<'_>) -> Self {
        Self {
            first_conv: CausalConvStream::new(
                weights.first_conv.input_channels,
                weights.first_conv.kernel,
                weights.first_conv.dilation,
            ),
            second_conv: CausalConvStream::new(
                weights.second_conv.input_channels,
                weights.second_conv.kernel,
                weights.second_conv.dilation,
            ),
        }
    }

    fn clear(&mut self) {
        self.first_conv.clear();
        self.second_conv.clear();
    }

    fn push(
        &mut self,
        input: &[f32],
        frames: usize,
        weights: CodecResidualUnitWeights<'_>,
        output: &mut Vec<f32>,
    ) {
        let mut hidden = input.to_vec();
        snake_beta_in_place(
            &mut hidden,
            frames,
            weights.first_alpha_log,
            weights.first_beta_log,
        );
        self.first_conv.push(
            &hidden,
            frames,
            weights.first_conv.weight,
            Some(weights.first_conv.bias),
            weights.first_conv.output_channels,
            output,
        );
        snake_beta_in_place(
            output,
            frames,
            weights.second_alpha_log,
            weights.second_beta_log,
        );
        hidden.clear();
        self.second_conv.push(
            output,
            frames,
            weights.second_conv.weight,
            Some(weights.second_conv.bias),
            weights.second_conv.output_channels,
            &mut hidden,
        );
        assert_eq!(hidden.len(), input.len(), "streaming residual output shape");
        output.clear();
        output.reserve(hidden.len());
        for index in 0..hidden.len() {
            output.push(hidden[index] + input[index]);
        }
    }
}

/// Retained causal context for one BigVGAN transposed-convolution block.
#[derive(Clone, Debug)]
struct CodecDecoderBlockStream {
    transposed: CausalTransposeConvStream,
    residual_units: [CodecResidualUnitStream; 3],
}

impl CodecDecoderBlockStream {
    fn new(weights: CodecDecoderBlockWeights<'_>) -> Self {
        Self {
            transposed: CausalTransposeConvStream::new(
                weights.transposed.input_channels,
                weights.transposed.output_channels,
                weights.transposed.kernel,
                weights.transposed.stride,
            ),
            residual_units: weights.residual_units.map(CodecResidualUnitStream::new),
        }
    }

    fn clear(&mut self) {
        self.transposed.clear();
        for unit in &mut self.residual_units {
            unit.clear();
        }
    }

    fn push(
        &mut self,
        input: &[f32],
        frames: usize,
        weights: CodecDecoderBlockWeights<'_>,
        output: &mut Vec<f32>,
    ) -> usize {
        let mut hidden = input.to_vec();
        snake_beta_in_place(&mut hidden, frames, weights.alpha_log, weights.beta_log);
        self.transposed.push(
            &hidden,
            frames,
            weights.transposed.weight,
            Some(weights.transposed.bias),
            output,
        );
        let output_frames = frames * weights.transposed.stride;
        for (unit_state, unit_weights) in self.residual_units.iter_mut().zip(weights.residual_units)
        {
            hidden.clear();
            hidden.extend_from_slice(output);
            unit_state.push(&hidden, output_frames, unit_weights, output);
        }
        output_frames
    }
}

/// Retained causal context for a latent 2× upsample plus ConvNeXt stage.
#[derive(Clone, Debug)]
struct CodecUpsampleStageStream {
    transposed: CausalTransposeConvStream,
    convnext: CodecConvNextStream,
}

impl CodecUpsampleStageStream {
    fn new(weights: CodecUpsampleStageWeights<'_>) -> Self {
        Self {
            transposed: CausalTransposeConvStream::new(
                weights.transposed.input_channels,
                weights.transposed.output_channels,
                weights.transposed.kernel,
                weights.transposed.stride,
            ),
            convnext: CodecConvNextStream::new(weights.transposed.output_channels),
        }
    }

    fn clear(&mut self) {
        self.transposed.clear();
        self.convnext.clear();
    }

    fn push(
        &mut self,
        input: &[f32],
        frames: usize,
        weights: CodecUpsampleStageWeights<'_>,
        output: &mut Vec<f32>,
    ) -> usize {
        let mut hidden = Vec::new();
        self.transposed.push(
            input,
            frames,
            weights.transposed.weight,
            Some(weights.transposed.bias),
            &mut hidden,
        );
        let output_frames = frames * weights.transposed.stride;
        self.convnext
            .push(&hidden, output_frames, weights.convnext, output);
        output_frames
    }
}

/// Stateful exact-f32 codec decoder for packeted token frames.
///
/// Each call emits all PCM samples made final by the supplied codec frames. The state preserves
/// the causal convolution contexts, transposed-convolution overlap, and per-layer 72-position KV
/// caches so concatenated packets match [`decode_codec_offline`] on the same token stream.
#[derive(Clone, Debug)]
pub struct CodecStreamingState {
    config: CodecConfig,
    frames_seen: usize,
    pre_conv: CausalConvStream,
    transformer_caches: Vec<CodecKvCache>,
    latent_upsample: [CodecUpsampleStageStream; 2],
    decoder_input: CausalConvStream,
    decoder_blocks: [CodecDecoderBlockStream; 4],
    final_conv: CausalConvStream,
}

impl CodecStreamingState {
    /// Allocate persistent state for one decoder instance, retaining no checkpoint data.
    #[must_use]
    pub fn new(config: CodecConfig, weights: &CodecDecoderWeights<'_>) -> Self {
        Self {
            config,
            frames_seen: 0,
            pre_conv: CausalConvStream::new(
                weights.pre_conv.input_channels,
                weights.pre_conv.kernel,
                weights.pre_conv.dilation,
            ),
            transformer_caches: vec![
                CodecKvCache::new(config);
                weights.pre_transformer.layers.len()
            ],
            latent_upsample: weights.latent_upsample.map(CodecUpsampleStageStream::new),
            decoder_input: CausalConvStream::new(
                weights.decoder_input.input_channels,
                weights.decoder_input.kernel,
                weights.decoder_input.dilation,
            ),
            decoder_blocks: weights.decoder_blocks.map(CodecDecoderBlockStream::new),
            final_conv: CausalConvStream::new(
                weights.final_conv.input_channels,
                weights.final_conv.kernel,
                weights.final_conv.dilation,
            ),
        }
    }

    /// Reset this state for a fresh utterance without releasing its retained allocations.
    pub fn clear(&mut self) {
        self.frames_seen = 0;
        self.pre_conv.clear();
        for cache in &mut self.transformer_caches {
            cache.clear();
        }
        for stage in &mut self.latent_upsample {
            stage.clear();
        }
        self.decoder_input.clear();
        for block in &mut self.decoder_blocks {
            block.clear();
        }
        self.final_conv.clear();
    }

    /// Prime this state with an ICL reference's codec codes, discarding their waveform.
    ///
    /// The official ICL path (`INF:612-631`, spec §5.3) PREPENDS the reference codes to the
    /// generated codes, decodes the concatenation as one sequence, and cuts the leading
    /// `round(ref_len / total_len * wav_len)` samples in the waveform domain. On this decoder the
    /// proportional rule collapses to exactness (§5.3's derived paragraph): every frame emits
    /// exactly 1,920 samples with full-context causal state, so `wav_len == total_len * 1920` and
    /// the cut is `ref_len * 1920` — precisely the samples this method decodes and drops. Pushing
    /// the generated frames afterwards therefore produces the official ICL waveform, bit for bit,
    /// with no concatenation buffer and no post-hoc trim.
    ///
    /// The primed state is the `.ftvoice-cache` seam: [`CodecStreamingState`] is `Clone`, so a
    /// caller may prime once per voice, keep the clone, and start every utterance from it —
    /// skipping the reference re-decode that would otherwise sit inside TTFA.
    ///
    /// Returns the number of reference samples discarded (`frames * 1920`).
    ///
    /// # Errors
    ///
    /// As [`CodecStreamingState::push`], on an invalid code id.
    pub fn prime_reference(
        &mut self,
        quantizer: SplitResidualVectorQuantizer<'_>,
        weights: &CodecDecoderWeights<'_>,
        codes: &[i32],
        frames: usize,
    ) -> Result<usize, CodecError> {
        let mut discarded = Vec::new();
        self.push(quantizer, weights, codes, frames, &mut discarded)?;
        Ok(discarded.len())
    }

    /// Decode a nonempty packet of `frames` code rows and write its finalized PCM to `output`.
    pub fn push(
        &mut self,
        quantizer: SplitResidualVectorQuantizer<'_>,
        weights: &CodecDecoderWeights<'_>,
        codes: &[i32],
        frames: usize,
        output: &mut Vec<f32>,
    ) -> Result<(), CodecError> {
        assert!(frames > 0, "streaming codec packets must contain a frame");
        let mut hidden = vec![0.0f32; frames * self.config.codec_latent_dim];
        quantizer.decode(self.config, codes, frames, &mut hidden)?;

        let mut next = Vec::new();
        self.pre_conv.push(
            &hidden,
            frames,
            weights.pre_conv.weight,
            Some(weights.pre_conv.bias),
            weights.pre_conv.output_channels,
            &mut next,
        );
        hidden = next;
        next = vec![0.0f32; frames * self.config.transformer_latent_dim];
        forward_codec_pre_transformer(
            self.config,
            &weights.pre_transformer,
            self.frames_seen,
            &hidden,
            frames,
            &mut self.transformer_caches,
            &mut next,
        );
        self.frames_seen += frames;
        hidden = next;
        next = Vec::new();

        let mut current_frames = frames;
        for (stage_state, stage_weights) in
            self.latent_upsample.iter_mut().zip(weights.latent_upsample)
        {
            current_frames = stage_state.push(&hidden, current_frames, stage_weights, &mut next);
            core::mem::swap(&mut hidden, &mut next);
        }
        self.decoder_input.push(
            &hidden,
            current_frames,
            weights.decoder_input.weight,
            Some(weights.decoder_input.bias),
            weights.decoder_input.output_channels,
            &mut next,
        );
        core::mem::swap(&mut hidden, &mut next);

        let mut waveform_frames = current_frames;
        for (block_state, block_weights) in
            self.decoder_blocks.iter_mut().zip(weights.decoder_blocks)
        {
            waveform_frames = block_state.push(&hidden, waveform_frames, block_weights, &mut next);
            core::mem::swap(&mut hidden, &mut next);
        }
        snake_beta_in_place(
            &mut hidden,
            waveform_frames,
            weights.final_alpha_log,
            weights.final_beta_log,
        );
        self.final_conv.push(
            &hidden,
            waveform_frames,
            weights.final_conv.weight,
            Some(weights.final_conv.bias),
            weights.final_conv.output_channels,
            output,
        );
        assert_eq!(
            output.len(),
            waveform_frames,
            "streaming codec output must be mono"
        );
        for sample in output {
            *sample = sample.clamp(-1.0, 1.0);
        }
        Ok(())
    }
}

/// Decode an entire fixed codec-token stream to clamped PCM with the exact safe f32 graph.
///
/// This is deliberately the whole-sequence reference path. The full packet-at-a-time engine will
/// reuse [`CausalConvStream`] for every convolutional seam and retain the same 72-position K/V
/// state; it must prove equality against this function rather than against upstream's 25-frame
/// chunk approximation.
pub fn decode_codec_offline(
    config: CodecConfig,
    quantizer: SplitResidualVectorQuantizer<'_>,
    weights: &CodecDecoderWeights<'_>,
    codes: &[i32],
    frames: usize,
) -> Result<Vec<f32>, CodecError> {
    let mut codec_latents = vec![0.0f32; frames * config.codec_latent_dim];
    quantizer.decode(config, codes, frames, &mut codec_latents)?;
    let tap_block = |tag: &str, buf: &[f32]| {
        if taps_active() {
            tap_emit(&format!(
                "ftts-tapC b={tag} h={:016x}",
                tap_hash_f32(buf),
            ));
        }
    };
    tap_block("quant", &codec_latents);
    let pre_transformer_input = weights.pre_conv.forward(&codec_latents, frames);
    let mut caches = vec![CodecKvCache::new(config); weights.pre_transformer.layers.len()];
    let mut hidden = vec![0.0f32; frames * config.transformer_latent_dim];
    forward_codec_pre_transformer(
        config,
        &weights.pre_transformer,
        0,
        &pre_transformer_input,
        frames,
        &mut caches,
        &mut hidden,
    );

    let mut latent_frames = frames;
    for stage in weights.latent_upsample {
        hidden = stage.transposed.forward(&hidden, latent_frames);
        latent_frames *= stage.transposed.stride;
        hidden = forward_convnext(
            &hidden,
            latent_frames,
            stage.transposed.output_channels,
            stage.convnext,
        );
    }
    hidden = weights.decoder_input.forward(&hidden, latent_frames);
    let mut waveform_frames = latent_frames;
    let mut channels = weights.decoder_input.output_channels;
    for block in weights.decoder_blocks {
        let (next, next_frames) = forward_decoder_block(&hidden, waveform_frames, channels, block);
        hidden = next;
        waveform_frames = next_frames;
        channels = block.transposed.output_channels;
    }
    snake_beta_in_place(
        &mut hidden,
        waveform_frames,
        weights.final_alpha_log,
        weights.final_beta_log,
    );
    hidden = weights.final_conv.forward(&hidden, waveform_frames);
    assert_eq!(
        hidden.len(),
        waveform_frames,
        "final codec output must be mono"
    );
    for sample in &mut hidden {
        *sample = sample.clamp(-1.0, 1.0);
    }
    Ok(hidden)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small_config() -> CodecConfig {
        CodecConfig {
            groups: 2,
            codebook_size: 2,
            codebook_dim: 2,
            codec_latent_dim: 2,
            transformer_latent_dim: 2,
            transformer_hidden_dim: 2,
            attention_heads: 1,
            attention_head_dim: 2,
            transformer_intermediate_dim: 2,
            sliding_window: 2,
            rms_norm_eps: 1e-5,
        }
    }

    #[test]
    fn codebook_materialization_clamps_usage_before_lookup() {
        let table =
            MaterializedCodebook::from_unnormalized(&[2.0, 4.0, 3.0, 6.0], &[2.0, 1e-8], 2, 2)
                .expect("well-shaped table materializes");
        assert_eq!(&table.entries()[..2], &[1.0, 2.0]);
        assert_eq!(&table.entries()[2..], &[300_000.0, 600_000.0]);
    }

    #[test]
    fn split_rvq_adds_semantic_and_acoustic_projected_branches() {
        let config = small_config();
        let first =
            MaterializedCodebook::from_unnormalized(&[2.0, 4.0, 0.0, 0.0], &[2.0, 1.0], 2, 2)
                .expect("first table");
        let rest =
            MaterializedCodebook::from_unnormalized(&[3.0, 6.0, 0.0, 0.0], &[3.0, 1.0], 2, 2)
                .expect("rest table");
        let quantizer = SplitResidualVectorQuantizer {
            first_codebook: &first,
            rest_codebooks: std::slice::from_ref(&rest),
            first_output_proj: &[1.0, 0.0, 0.0, 1.0],
            rest_output_proj: &[1.0, 0.0, 0.0, 1.0],
        };
        let mut output = [0.0f32; 2];
        quantizer
            .decode(config, &[0, 0], 1, &mut output)
            .expect("valid ids");
        assert_eq!(output, [2.0, 4.0]);
    }

    #[test]
    fn codec_rope_is_identity_at_zero_and_uses_theta_10000() {
        let mut cos = [0.0f32; 4];
        let mut sin = [0.0f32; 4];
        codec_rope_rows(0, 4, &mut cos, &mut sin);
        assert_eq!(cos, [1.0; 4]);
        assert_eq!(sin, [0.0; 4]);
        codec_rope_rows(1, 4, &mut cos, &mut sin);
        assert!((cos[0] - 1.0f32.cos()).abs() < 1e-6);
        assert!((cos[1] - (1.0f32 / 100.0).cos()).abs() < 1e-6);
        assert_eq!(cos[0], cos[2]);
        assert_eq!(sin[1], sin[3]);
    }

    #[test]
    fn snake_beta_exponentiates_log_parameters() {
        let mut values = [core::f32::consts::FRAC_PI_2];
        snake_beta_in_place(&mut values, 1, &[0.0], &[0.0]);
        assert!((values[0] - (core::f32::consts::FRAC_PI_2 + 1.0)).abs() < 1e-5);
    }

    #[test]
    fn causal_conv_streaming_equals_one_shot() {
        let input = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        let weight = [1.0f32, 2.0, 3.0];
        let mut offline = [0.0f32; 5];
        causal_conv1d(&input, 5, 1, &weight, None, 1, 3, 1, &mut offline);
        assert_eq!(offline, [3.0, 8.0, 14.0, 20.0, 26.0]);
        let mut stream = CausalConvStream::new(1, 3, 1);
        let mut first = Vec::new();
        let mut second = Vec::new();
        stream.push(&input[..2], 2, &weight, None, 1, &mut first);
        stream.push(&input[2..], 3, &weight, None, 1, &mut second);
        first.extend(second);
        assert_eq!(first, offline);
    }

    #[test]
    fn causal_transpose_trims_the_right_padding() {
        let input = [1.0f32, 2.0];
        let weight = [1.0f32, 10.0, 100.0, 1_000.0];
        let mut output = [0.0f32; 4];
        causal_transpose_conv1d(&input, 2, 1, &weight, None, 1, 4, 2, &mut output);
        assert_eq!(output, [1.0, 10.0, 102.0, 1_020.0]);
    }

    #[test]
    fn causal_transpose_streaming_equals_one_shot_with_overlap() {
        let input = [1.0f32, 2.0, 3.0];
        let weight = [1.0f32, 10.0, 100.0, 1_000.0];
        let mut offline = [0.0f32; 6];
        causal_transpose_conv1d(&input, 3, 1, &weight, None, 1, 4, 2, &mut offline);

        let mut stream = CausalTransposeConvStream::new(1, 1, 4, 2);
        let mut packet = Vec::new();
        let mut packeted = Vec::new();
        stream.push(&input[..1], 1, &weight, None, &mut packet);
        packeted.extend_from_slice(&packet);
        stream.push(&input[1..], 2, &weight, None, &mut packet);
        packeted.extend_from_slice(&packet);

        assert_eq!(packeted, offline);
        stream.clear();
        stream.push(&input, 3, &weight, None, &mut packet);
        assert_eq!(packet, offline);
    }

    #[test]
    fn codec_k_7168_has_the_documented_i32_headroom() {
        require_i32_safe_u8s8_reduction(CODEC_MAX_INT8_REDUCTION_K).expect("binding K fits i32");
        assert_eq!(CODEC_MAX_U8S8_ACCUMULATOR, 232_135_680);
        assert!(CODEC_MAX_U8S8_ACCUMULATOR < i64::from(i32::MAX));
        assert_eq!(
            require_i32_safe_u8s8_reduction(CODEC_MAX_INT8_REDUCTION_K * 10),
            Err(CodecError::I32AccumulatorOverflow {
                reduction_k: CODEC_MAX_INT8_REDUCTION_K * 10
            })
        );
    }

    #[test]
    fn transformer_keeps_a_bounded_cache_and_layerscale_branches() {
        let config = small_config();
        let norm = [1.0f32; 2];
        let zeros_hidden = [0.0f32; 4];
        let zeros_attention = [0.0f32; 4];
        let zero_down = [0.0f32; 4];
        let scales = [0.01f32; 2];
        let weights = CodecTransformerLayerWeights {
            input_layernorm: &norm,
            q_proj: &zeros_hidden,
            k_proj: &zeros_hidden,
            v_proj: &zeros_hidden,
            o_proj: &zeros_attention,
            self_attn_layer_scale: &scales,
            post_attention_layernorm: &norm,
            gate_proj: &zeros_hidden,
            up_proj: &zeros_hidden,
            down_proj: &zero_down,
            mlp_layer_scale: &scales,
        };
        let mut cache = CodecKvCache::new(config);
        let mut state = [3.0f32, -4.0];
        for position in 0..3 {
            forward_codec_transformer_step(config, weights, position, &mut state, &mut cache);
        }
        assert_eq!(state, [3.0, -4.0]);
        assert_eq!(
            cache.len(),
            2,
            "cache evicts exactly the oldest position at the window"
        );
    }

    #[test]
    fn pre_transformer_packet_execution_is_exactly_whole_sequence_execution() {
        let config = small_config();
        let norm = [1.0f32; 2];
        let identity = [1.0f32, 0.0, 0.0, 1.0];
        let zero = [0.0f32; 4];
        let scale = [1.0f32; 2];
        let layer = CodecTransformerLayerWeights {
            input_layernorm: &norm,
            q_proj: &identity,
            k_proj: &identity,
            v_proj: &identity,
            o_proj: &identity,
            self_attn_layer_scale: &scale,
            post_attention_layernorm: &norm,
            gate_proj: &zero,
            up_proj: &zero,
            down_proj: &zero,
            mlp_layer_scale: &scale,
        };
        let layers = [layer];
        let weights = CodecPreTransformerWeights {
            input_proj: &identity,
            input_proj_bias: &[0.0; 2],
            layers: &layers,
            final_norm: &norm,
            output_proj: &identity,
            output_proj_bias: &[0.0; 2],
        };
        let input = [1.0f32, -2.0, 3.0, 4.0, -5.0, 6.0];
        let mut whole_sequence = [0.0f32; 6];
        let mut whole_cache = [CodecKvCache::new(config)];
        forward_codec_pre_transformer(
            config,
            &weights,
            0,
            &input,
            3,
            &mut whole_cache,
            &mut whole_sequence,
        );

        let mut packet_cache = [CodecKvCache::new(config)];
        let mut packeted = [0.0f32; 6];
        for frame in 0..3 {
            forward_codec_pre_transformer(
                config,
                &weights,
                frame,
                &input[frame * 2..(frame + 1) * 2],
                1,
                &mut packet_cache,
                &mut packeted[frame * 2..(frame + 1) * 2],
            );
        }
        assert_eq!(packeted, whole_sequence);
    }

    #[test]
    fn offline_codec_graph_emits_exactly_1920_samples_per_frame() {
        let config = small_config();
        let first = MaterializedCodebook::from_unnormalized(&[1.0, 0.0, 0.0, 0.0], &[1.0; 2], 2, 2)
            .expect("first codebook");
        let rest = MaterializedCodebook::from_unnormalized(&[0.0; 4], &[1.0; 2], 2, 2)
            .expect("rest codebook");
        let rest_codebooks = [rest];
        let quantizer = SplitResidualVectorQuantizer {
            first_codebook: &first,
            rest_codebooks: &rest_codebooks,
            first_output_proj: &[1.0, 0.0, 0.0, 1.0],
            rest_output_proj: &[0.0; 4],
        };

        let two_channels = [0.0f32; 2];
        let two_by_two = [1.0f32, 0.0, 0.0, 1.0];
        let pre_conv_weight = [
            0.0f32, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        let pre_conv = CausalConvWeights {
            weight: &pre_conv_weight,
            bias: &two_channels,
            input_channels: 2,
            output_channels: 2,
            kernel: 3,
            dilation: 1,
        };
        let pre_transformer = CodecPreTransformerWeights {
            input_proj: &two_by_two,
            input_proj_bias: &two_channels,
            layers: &[],
            final_norm: &[1.0; 2],
            output_proj: &two_by_two,
            output_proj_bias: &two_channels,
        };
        let latent_transposed_weight = [0.1f32, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let latent_transposed = CausalTransposeConvWeights {
            weight: &latent_transposed_weight,
            bias: &two_channels,
            input_channels: 2,
            output_channels: 2,
            kernel: 2,
            stride: 2,
        };
        let convnext = CodecConvNextWeights {
            depthwise_weight: &[0.0; 14],
            depthwise_bias: &two_channels,
            norm_weight: &[1.0; 2],
            norm_bias: &two_channels,
            pwconv1: &[0.0; 16],
            pwconv1_bias: &[0.0; 8],
            pwconv2: &[0.0; 16],
            pwconv2_bias: &two_channels,
            gamma: &two_channels,
        };
        let latent_upsample = [
            CodecUpsampleStageWeights {
                transposed: latent_transposed,
                convnext,
            },
            CodecUpsampleStageWeights {
                transposed: latent_transposed,
                convnext,
            },
        ];
        let decoder_input_weight = [
            0.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        let decoder_input = CausalConvWeights {
            weight: &decoder_input_weight,
            bias: &[0.0],
            input_channels: 2,
            output_channels: 1,
            kernel: 7,
            dilation: 1,
        };
        let one_channel = [0.0f32; 1];
        let residual_at_dilation = |dilation| CodecResidualUnitWeights {
            first_alpha_log: &one_channel,
            first_beta_log: &one_channel,
            first_conv: CausalConvWeights {
                weight: &[0.0; 7],
                bias: &one_channel,
                input_channels: 1,
                output_channels: 1,
                kernel: 7,
                dilation,
            },
            second_alpha_log: &one_channel,
            second_beta_log: &one_channel,
            second_conv: CausalConvWeights {
                weight: &[0.0],
                bias: &one_channel,
                input_channels: 1,
                output_channels: 1,
                kernel: 1,
                dilation: 1,
            },
        };
        let residual_units = [
            residual_at_dilation(1),
            residual_at_dilation(3),
            residual_at_dilation(9),
        ];
        let block_8_weight = [
            0.1f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        let block_5_weight = [0.1f32, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0];
        let block_4_weight = [0.1f32, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0];
        let block_3_weight = [0.1f32, 0.0, 0.0, 0.1, 0.0, 0.0];
        let decoder_blocks = [
            CodecDecoderBlockWeights {
                alpha_log: &one_channel,
                beta_log: &one_channel,
                transposed: CausalTransposeConvWeights {
                    weight: &block_8_weight,
                    bias: &one_channel,
                    input_channels: 1,
                    output_channels: 1,
                    kernel: 16,
                    stride: 8,
                },
                residual_units,
            },
            CodecDecoderBlockWeights {
                alpha_log: &one_channel,
                beta_log: &one_channel,
                transposed: CausalTransposeConvWeights {
                    weight: &block_5_weight,
                    bias: &one_channel,
                    input_channels: 1,
                    output_channels: 1,
                    kernel: 10,
                    stride: 5,
                },
                residual_units,
            },
            CodecDecoderBlockWeights {
                alpha_log: &one_channel,
                beta_log: &one_channel,
                transposed: CausalTransposeConvWeights {
                    weight: &block_4_weight,
                    bias: &one_channel,
                    input_channels: 1,
                    output_channels: 1,
                    kernel: 8,
                    stride: 4,
                },
                residual_units,
            },
            CodecDecoderBlockWeights {
                alpha_log: &one_channel,
                beta_log: &one_channel,
                transposed: CausalTransposeConvWeights {
                    weight: &block_3_weight,
                    bias: &one_channel,
                    input_channels: 1,
                    output_channels: 1,
                    kernel: 6,
                    stride: 3,
                },
                residual_units,
            },
        ];
        let weights = CodecDecoderWeights {
            pre_conv,
            pre_transformer,
            latent_upsample,
            decoder_input,
            decoder_blocks,
            final_alpha_log: &one_channel,
            final_beta_log: &one_channel,
            final_conv: CausalConvWeights {
                weight: &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1],
                bias: &one_channel,
                input_channels: 1,
                output_channels: 1,
                kernel: 7,
                dilation: 1,
            },
        };

        let codes = [0, 0, 0, 0];
        let waveform = decode_codec_offline(config, quantizer, &weights, &codes, 2)
            .expect("well-shaped model graph");
        assert_eq!(waveform.len(), 3_840);
        assert!(waveform.iter().all(|sample| sample.is_finite()));
        assert!(
            waveform
                .iter()
                .any(|sample| sample.is_normal() || sample.is_subnormal())
        );

        let mut stream = CodecStreamingState::new(config, &weights);
        let mut packet = Vec::new();
        let mut packeted = Vec::new();
        stream
            .push(quantizer, &weights, &codes[..2], 1, &mut packet)
            .expect("first codec packet");
        packeted.extend_from_slice(&packet);
        stream
            .push(quantizer, &weights, &codes[2..], 1, &mut packet)
            .expect("second codec packet");
        packeted.extend_from_slice(&packet);
        assert_eq!(packeted, waveform);

        stream.clear();
        stream
            .push(quantizer, &weights, &codes, 2, &mut packet)
            .expect("reset codec stream");
        assert_eq!(packet, waveform);
    }

    #[test]
    fn int8_codec_route_decision_matrix() {
        // The pin always wins: an exported FTTS_INT8_CODEC must never arm the codec inside an
        // oracle-pinned run (frankentts-7ozx — the raw env read used to leak exactly there).
        for env in [
            None,
            Some("0"),
            Some("1"),
            Some("all"),
            Some("transformer"),
            Some("convnext"),
            Some("yes"),
        ] {
            assert_eq!(
                codec_int8_route_decision(true, env),
                None,
                "env {env:?}: a pinned reference route must disarm the codec"
            );
        }
        // Unpinned: the documented opt-in values, and everything else disarmed.
        assert_eq!(codec_int8_route_decision(false, None), None);
        assert_eq!(codec_int8_route_decision(false, Some("0")), None);
        assert_eq!(codec_int8_route_decision(false, Some("garbage")), None);
        assert_eq!(
            codec_int8_route_decision(false, Some("1")),
            Some((true, true))
        );
        assert_eq!(
            codec_int8_route_decision(false, Some("all")),
            Some((true, true))
        );
        assert_eq!(
            codec_int8_route_decision(false, Some("transformer")),
            Some((true, false))
        );
        assert_eq!(
            codec_int8_route_decision(false, Some("convnext")),
            Some((false, true))
        );
    }
}
