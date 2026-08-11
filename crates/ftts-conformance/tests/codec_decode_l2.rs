//! L2 parity: every captured codec-decoder seam against the CPU-fp32 oracle.
//!
// The float literals in this file are pinned observed divergences; truncating them to clippy's
// taste would un-pin the measurements (AGENTS.md doctrine #8: no silent numerics changes).
#![allow(clippy::excessive_precision)]
//!
//! The ft7 pack captures the codec decoder at stage granularity: the SplitRVQ input codes, each of
//! the eight pre-transformer layers, both latent upsample stages, all seven `decoder.*` children,
//! and the decoder's own waveform output. Each stage here is fed the oracle's *captured input* —
//! never our own previous stage's output — so a divergence names the exact stage that computed it.
//!
//! Like the talker L2 harness, exact CPU-fp32 parity remains open: every seam currently diverges
//! at libm/accumulation-order magnitude (max_abs 1e-7..1e-5, cosine ≈ 1). Each seam's observed
//! divergence is therefore pinned as a recorded XFAIL and this test is a ratchet: an arithmetic
//! change that moves any seam's divergence — including achieving an exact pass — fails the test
//! until the change is reviewed and the pin moved deliberately.
//!
//! Model-gated twice over — it needs both the fixture set and the pinned speech-tokenizer
//! checkpoint — and skips with SUCCESS and a named reason when either is absent.

use ftts_artifacts::safetensors::SafetensorsFile;
use ftts_conformance::npy::NpyArray;
use ftts_conformance::{
    compare::compare_f32,
    oracle::{
        CPU_TIER_TOLERANCE, CPU_TIER_TOLERANCE_SOURCE, OracleFixtures, SeamRef, compare_exactly,
    },
    report::{OracleTier, Outcome, Receipt},
};
use ftts_kernels::f32ref;
use ftts_model_qwen::codec::{
    CausalConvWeights, CausalTransposeConvWeights, CodecConfig, CodecConvNextWeights,
    CodecDecoderBlockWeights, CodecDecoderWeights, CodecKvCache, CodecPreTransformerWeights,
    CodecResidualUnitWeights, CodecStreamingState, CodecTransformerLayerWeights,
    CodecUpsampleStageWeights, MaterializedCodebook, SplitResidualVectorQuantizer, causal_conv1d,
    causal_transpose_conv1d, codec_rope_rows, decode_codec_offline, forward_codec_transformer_step,
    forward_convnext, forward_decoder_block, snake_beta_in_place, transpose_conv_weight,
};
use std::path::{Path, PathBuf};

const TEST_NAME: &str = "contract_a_l2_codec_decoder_stages_cpu_fp32_exact";

/// Observed CPU-fp32 arithmetic divergence per seam, pinned exactly like the talker L2 ratchet:
/// exact parity remains open, so each seam's `(max_abs_diff, over_tolerance)` is frozen and any
/// arithmetic change that moves it — better or worse — must be reviewed before the pin moves.
///
/// Ratchet round 4 (the convolution GEMM). `codec_gemm_bisect` measured `block_00` — one causal
/// Conv1d, oracle-fed, no activation or norm anywhere near it — five ways, and exactly one was
/// exact: the platform BLAS issued over an `im2col` unfolding with the bias seeded as a `beta = 1`
/// accumulator, which is the call `slow_conv2d` itself makes. Scalar left-to-right was 1.240e-5,
/// 4-lane and 8-lane 3.815e-5, and the same BLAS with the bias added after a `beta = 0` product
/// 1.335e-5 — so the seeding is load-bearing, not incidental. Rewriting `causal_conv1d` and
/// `causal_transpose_conv1d` into that form retired three seams from this table outright:
///
///   codec_decoder.block_00       2.861e-6 -> EXACT
///   codec_decoder.upsample_0_0   2.384e-7 -> EXACT
///   codec_decoder.upsample_1_0   2.861e-6 -> EXACT
///
/// Every remaining entry below is a seam that is still downstream of something un-retired, so most
/// moved by a ulp or two as their inputs changed. Two moved the wrong way and are recorded rather
/// than smoothed over: `block_02`/`block_03` gained a ulp of `max_abs`, and `block_06` holds its
/// `max_abs` but diverges over more elements (9_110 -> 15_125). The end-to-end waveform improved
/// (`decode_codec_offline[icl]` 2.384e-7 -> 2.198e-7).
///
/// The dense projections were tried the same way and REJECTED — see the negative-evidence note in
/// `ftts-model-qwen/src/codec.rs`; `nn.Linear` does not land on the convolution's blocking, so the
/// transformer seams here are unchanged in form and only shifted by their inputs.
///
/// Ratchet round 5 (the RVQ output projections). Round 4 left one measurement open: whether the
/// `k = 1` projections follow the convolution rule or the `nn.Linear` rule. The pinned checkpoint
/// answers it directly — `decoder.quantizer.rvq_{first,rest}.output_proj.weight` carry the 3-D
/// shape `[512, 256, 1]`, so they are `Conv1d`, while `decoder.pre_transformer.{input,output}_proj`
/// are 2-D and so are genuinely `nn.Linear`. Routing only the two RVQ projections through the
/// convolution's BLAS (bias-free, `beta = 0`) tightened the first seam in the graph:
///
///   rvq+pre_conv+input_proj      1.639e-7 -> 5.960e-8   (6_674 -> 5_234 elements)
///   decode_codec_offline[icl]    2.198e-7 -> 1.788e-7   (26_194 -> 25_749)
///
/// Every other seam is bit-identical, which is itself the isolation: the oracle-fed transformer
/// stages do not see this input. One entry moved the wrong way and is recorded rather than smoothed
/// over — `decode_codec_offline[xvector]` 4.005e-8 -> 5.402e-8 over 1_904 of 1_920 samples, a
/// ulp-scale wobble on the shorter of the two waveforms while the longer one improved.
///
/// What is left in `rvq+pre_conv+input_proj` is therefore `pre_transformer.input_proj`, a real
/// `nn.Linear`, plus the codebook accumulation order — `pre_conv` itself went exact in round 4.
const PINNED_DIVERGENCE: &[(&str, f64, usize)] = &[
    ("rvq+pre_conv+input_proj", 5.9604644775390625e-8, 5_234),
    ("codec_rope.cos", 5.9604644775390625e-8, 80),
    ("codec_rope.sin", 5.9604644775390625e-8, 24),
    (
        "codec_decoder.transformer_layer_00.output",
        5.9604644775390625e-8,
        4_332,
    ),
    (
        "codec_decoder.transformer_layer_01.output",
        4.4703483581542969e-8,
        4_244,
    ),
    (
        "codec_decoder.transformer_layer_02.output",
        2.8610229492187500e-6,
        4_552,
    ),
    (
        "codec_decoder.transformer_layer_03.output",
        1.3411045074462891e-7,
        4_895,
    ),
    (
        "codec_decoder.transformer_layer_04.output",
        1.3411045074462891e-7,
        4_742,
    ),
    (
        "codec_decoder.transformer_layer_05.output",
        1.1920928955078125e-7,
        4_588,
    ),
    (
        "codec_decoder.transformer_layer_06.output",
        5.9604644775390625e-8,
        3_214,
    ),
    (
        "codec_decoder.transformer_layer_07.output",
        6.7055225372314453e-8,
        4_161,
    ),
    ("final_norm+output_proj", 8.9406967163085938e-8, 13_462),
    ("codec_decoder.upsample_0_1", 1.4901161193847656e-6, 24_695),
    ("codec_decoder.upsample_1_1", 2.8610229492187500e-5, 49_656),
    (
        "codec_decoder.block_01.output",
        4.7683715820312500e-6,
        319_209,
    ),
    (
        "codec_decoder.block_02.output",
        4.0531158447265625e-6,
        791_259,
    ),
    (
        "codec_decoder.block_03.output",
        2.5033950805664062e-6,
        1_495_609,
    ),
    (
        "codec_decoder.block_04.output",
        1.3351440429687500e-5,
        2_463_507,
    ),
    ("codec_decoder.block_05", 7.6293945312500000e-6, 42_042),
    ("codec_decoder.block_06", 2.2351741790771484e-8, 15_125),
    ("decode_codec_offline[icl]", 1.8626451492309570e-7, 25_671),
    // Moved 5.402e-8/1_904 -> 4.610e-8/1_877 when the Accelerate M=1 GEMV path was pinned to the
    // M>=2 GEMM kernel for streaming==offline M-invariance (this capture decodes a single frame,
    // so its conv GEMMs ran M=1 offline); the shift is toward the oracle.
    (
        "decode_codec_offline[xvector]",
        5.2154064178466797e-8,
        1_891,
    ),
];

/// The pinned speech-tokenizer checkpoint, alongside the truth-pack snapshots.
fn checkpoint_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../docs/truth-pack/snapshots/hf/speech_tokenizer/model.safetensors")
}

fn skip(reason: &str) {
    Receipt::new(TEST_NAME, Outcome::Skipped)
        .contract("ConformanceExact/L2")
        .seam("codec_decode")
        .reason(reason)
        .tolerance(CPU_TIER_TOLERANCE, CPU_TIER_TOLERANCE_SOURCE)
        .oracle_tier(OracleTier::CpuFp32Fallback)
        .emit();
}

/// Widen a whole tensor to `f32` through the accessor.
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

/// Minimal reader for the oracle's `int64` arrays (codes, cache positions), which the shared
/// float32-only `npy` reader refuses by design.
fn read_i64_npy(path: &Path) -> (Vec<usize>, Vec<i64>) {
    let bytes = std::fs::read(path)
        .unwrap_or_else(|error| panic!("cannot read `{}`: {error}", path.display()));
    assert_eq!(&bytes[..6], b"\x93NUMPY", "not an npy file");
    let (header_len, body_start) = match bytes[6] {
        1 => (
            usize::from(u16::from_le_bytes([bytes[8], bytes[9]])),
            10usize,
        ),
        2 => (
            u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize,
            12usize,
        ),
        other => panic!("unsupported npy version {other}"),
    };
    let header = std::str::from_utf8(&bytes[body_start..body_start + header_len])
        .expect("npy header is ASCII");
    assert!(
        header.contains("'descr': '<i8'"),
        "expected little-endian int64, header: {header}"
    );
    assert!(
        header.contains("'fortran_order': False"),
        "Fortran order unsupported, header: {header}"
    );
    let shape_text = header
        .split("'shape': (")
        .nth(1)
        .and_then(|rest| rest.split(')').next())
        .expect("npy header has a shape tuple");
    let shape: Vec<usize> = shape_text
        .split(',')
        .filter_map(|piece| piece.trim().parse::<usize>().ok())
        .collect();
    let data: Vec<i64> = bytes[body_start + header_len..]
        .as_chunks::<8>()
        .0
        .iter()
        .map(|chunk| i64::from_le_bytes(*chunk))
        .collect();
    assert_eq!(
        data.len(),
        shape.iter().product::<usize>(),
        "npy payload length disagrees with its shape"
    );
    (shape, data)
}

/// Fixture conv tensors are channel-major `[1, channels, frames]`; our engine is time-major.
fn to_time_major(array: &NpyArray) -> (Vec<f32>, usize, usize) {
    let (channels, frames) = match array.shape.as_slice() {
        [1, channels, frames] => (*channels, *frames),
        other => panic!("expected [1, C, T], got {other:?}"),
    };
    let mut out = vec![0.0f32; channels * frames];
    for channel in 0..channels {
        for frame in 0..frames {
            out[frame * channels + channel] = array.data[channel * frames + frame];
        }
    }
    (out, channels, frames)
}

/// Transpose our time-major output back to the oracle's channel-major layout for comparison.
fn to_channel_major(values: &[f32], frames: usize, channels: usize) -> Vec<f32> {
    assert_eq!(values.len(), frames * channels, "transpose shape");
    let mut out = vec![0.0f32; values.len()];
    for frame in 0..frames {
        for channel in 0..channels {
            out[channel * frames + frame] = values[frame * channels + channel];
        }
    }
    out
}

/// Load the oracle's `[1, 16, frames]` group-major codes as our frame-major `i32` layout.
fn load_codes(fixtures: &OracleFixtures, seam: &SeamRef<'_>) -> (Vec<i32>, usize) {
    let path = fixtures.seam_path(seam, "args.0", 0);
    let (shape, raw) = read_i64_npy(&path);
    let (groups, frames) = match shape.as_slice() {
        [1, groups, frames] => (*groups, *frames),
        other => panic!("expected codes [1, 16, T], got {other:?}"),
    };
    assert_eq!(groups, 16, "the codec always carries 16 code groups");
    let mut codes = vec![0i32; frames * groups];
    for group in 0..groups {
        for frame in 0..frames {
            codes[frame * groups + group] =
                i32::try_from(raw[group * frames + frame]).expect("codec id fits i32");
        }
    }
    (codes, frames)
}

/// One layer's owned f32 weights, so the borrowed layer struct has something to point into.
struct OwnedLayer {
    input_layernorm: Vec<f32>,
    q_proj: Vec<f32>,
    k_proj: Vec<f32>,
    v_proj: Vec<f32>,
    o_proj: Vec<f32>,
    self_attn_layer_scale: Vec<f32>,
    post_attention_layernorm: Vec<f32>,
    gate_proj: Vec<f32>,
    up_proj: Vec<f32>,
    down_proj: Vec<f32>,
    mlp_layer_scale: Vec<f32>,
}

impl OwnedLayer {
    fn load(file: &SafetensorsFile, index: usize) -> Self {
        let prefix = format!("decoder.pre_transformer.layers.{index}");
        Self {
            input_layernorm: widen(file, &format!("{prefix}.input_layernorm.weight")),
            q_proj: widen(file, &format!("{prefix}.self_attn.q_proj.weight")),
            k_proj: widen(file, &format!("{prefix}.self_attn.k_proj.weight")),
            v_proj: widen(file, &format!("{prefix}.self_attn.v_proj.weight")),
            o_proj: widen(file, &format!("{prefix}.self_attn.o_proj.weight")),
            self_attn_layer_scale: widen(file, &format!("{prefix}.self_attn_layer_scale.scale")),
            post_attention_layernorm: widen(
                file,
                &format!("{prefix}.post_attention_layernorm.weight"),
            ),
            gate_proj: widen(file, &format!("{prefix}.mlp.gate_proj.weight")),
            up_proj: widen(file, &format!("{prefix}.mlp.up_proj.weight")),
            down_proj: widen(file, &format!("{prefix}.mlp.down_proj.weight")),
            mlp_layer_scale: widen(file, &format!("{prefix}.mlp_layer_scale.scale")),
        }
    }

    fn borrow(&self) -> CodecTransformerLayerWeights<'_> {
        CodecTransformerLayerWeights {
            input_layernorm: &self.input_layernorm,
            q_proj: &self.q_proj,
            k_proj: &self.k_proj,
            v_proj: &self.v_proj,
            o_proj: &self.o_proj,
            self_attn_layer_scale: &self.self_attn_layer_scale,
            post_attention_layernorm: &self.post_attention_layernorm,
            gate_proj: &self.gate_proj,
            up_proj: &self.up_proj,
            down_proj: &self.down_proj,
            mlp_layer_scale: &self.mlp_layer_scale,
        }
    }
}

/// Owned SnakeBeta + causal conv pair for one residual-unit half.
struct OwnedConv {
    weight: Vec<f32>,
    bias: Vec<f32>,
}

impl OwnedConv {
    fn load(file: &SafetensorsFile, prefix: &str) -> Self {
        Self {
            weight: widen(file, &format!("{prefix}.weight")),
            bias: widen(file, &format!("{prefix}.bias")),
        }
    }
}

struct OwnedResidualUnit {
    first_alpha: Vec<f32>,
    first_beta: Vec<f32>,
    first_conv: OwnedConv,
    second_alpha: Vec<f32>,
    second_beta: Vec<f32>,
    second_conv: OwnedConv,
}

impl OwnedResidualUnit {
    fn load(file: &SafetensorsFile, block: usize, unit: usize) -> Self {
        let prefix = format!("decoder.decoder.{block}.block.{unit}");
        Self {
            first_alpha: widen(file, &format!("{prefix}.act1.alpha")),
            first_beta: widen(file, &format!("{prefix}.act1.beta")),
            first_conv: OwnedConv::load(file, &format!("{prefix}.conv1.conv")),
            second_alpha: widen(file, &format!("{prefix}.act2.alpha")),
            second_beta: widen(file, &format!("{prefix}.act2.beta")),
            second_conv: OwnedConv::load(file, &format!("{prefix}.conv2.conv")),
        }
    }

    fn borrow(&self, channels: usize, dilation: usize) -> CodecResidualUnitWeights<'_> {
        CodecResidualUnitWeights {
            first_alpha_log: &self.first_alpha,
            first_beta_log: &self.first_beta,
            first_conv: CausalConvWeights {
                weight: &self.first_conv.weight,
                bias: &self.first_conv.bias,
                input_channels: channels,
                output_channels: channels,
                kernel: 7,
                dilation,
            },
            second_alpha_log: &self.second_alpha,
            second_beta_log: &self.second_beta,
            second_conv: CausalConvWeights {
                weight: &self.second_conv.weight,
                bias: &self.second_conv.bias,
                input_channels: channels,
                output_channels: channels,
                kernel: 1,
                dilation: 1,
            },
        }
    }
}

struct OwnedBlock {
    alpha: Vec<f32>,
    beta: Vec<f32>,
    transposed: OwnedConv,
    transposed_column_weight: Vec<f32>,
    units: [OwnedResidualUnit; 3],
    input_channels: usize,
    output_channels: usize,
    kernel: usize,
    stride: usize,
}

impl OwnedBlock {
    fn load(
        file: &SafetensorsFile,
        block: usize,
        input_channels: usize,
        output_channels: usize,
        kernel: usize,
        stride: usize,
    ) -> Self {
        let prefix = format!("decoder.decoder.{block}");
        let transposed = OwnedConv::load(file, &format!("{prefix}.block.1.conv"));
        let transposed_column_weight =
            transpose_conv_weight(&transposed.weight, input_channels, output_channels, kernel);
        Self {
            alpha: widen(file, &format!("{prefix}.block.0.alpha")),
            beta: widen(file, &format!("{prefix}.block.0.beta")),
            transposed,
            transposed_column_weight,
            units: [
                OwnedResidualUnit::load(file, block, 2),
                OwnedResidualUnit::load(file, block, 3),
                OwnedResidualUnit::load(file, block, 4),
            ],
            input_channels,
            output_channels,
            kernel,
            stride,
        }
    }

    fn borrow(&self) -> CodecDecoderBlockWeights<'_> {
        CodecDecoderBlockWeights {
            alpha_log: &self.alpha,
            beta_log: &self.beta,
            transposed: CausalTransposeConvWeights {
                weight: &self.transposed.weight,
                column_weight: &self.transposed_column_weight,
                bias: &self.transposed.bias,
                input_channels: self.input_channels,
                output_channels: self.output_channels,
                kernel: self.kernel,
                stride: self.stride,
            },
            residual_units: [
                self.units[0].borrow(self.output_channels, 1),
                self.units[1].borrow(self.output_channels, 3),
                self.units[2].borrow(self.output_channels, 9),
            ],
        }
    }
}

struct OwnedUpsampleStage {
    transposed: OwnedConv,
    transposed_column_weight: Vec<f32>,
    dwconv: OwnedConv,
    norm_weight: Vec<f32>,
    norm_bias: Vec<f32>,
    pwconv1: OwnedConv,
    pwconv2: OwnedConv,
    gamma: Vec<f32>,
}

impl OwnedUpsampleStage {
    fn load(file: &SafetensorsFile, stage: usize) -> Self {
        let prefix = format!("decoder.upsample.{stage}");
        let transposed = OwnedConv::load(file, &format!("{prefix}.0.conv"));
        let transposed_column_weight = transpose_conv_weight(&transposed.weight, 1024, 1024, 2);
        Self {
            transposed,
            transposed_column_weight,
            dwconv: OwnedConv::load(file, &format!("{prefix}.1.dwconv.conv")),
            norm_weight: widen(file, &format!("{prefix}.1.norm.weight")),
            norm_bias: widen(file, &format!("{prefix}.1.norm.bias")),
            pwconv1: OwnedConv::load(file, &format!("{prefix}.1.pwconv1")),
            pwconv2: OwnedConv::load(file, &format!("{prefix}.1.pwconv2")),
            gamma: widen(file, &format!("{prefix}.1.gamma")),
        }
    }

    fn borrow_transposed(&self) -> CausalTransposeConvWeights<'_> {
        CausalTransposeConvWeights {
            weight: &self.transposed.weight,
            column_weight: &self.transposed_column_weight,
            bias: &self.transposed.bias,
            input_channels: 1024,
            output_channels: 1024,
            kernel: 2,
            stride: 2,
        }
    }

    fn borrow_convnext(&self) -> CodecConvNextWeights<'_> {
        CodecConvNextWeights {
            depthwise_weight: &self.dwconv.weight,
            depthwise_bias: &self.dwconv.bias,
            norm_weight: &self.norm_weight,
            norm_bias: &self.norm_bias,
            pwconv1: &self.pwconv1.weight,
            pwconv1_bias: &self.pwconv1.bias,
            pwconv2: &self.pwconv2.weight,
            pwconv2_bias: &self.pwconv2.bias,
            gamma: &self.gamma,
        }
    }
}

/// Every owned codec weight, loaded once for all stage checks.
struct OwnedCodec {
    first_codebook: MaterializedCodebook,
    rest_codebooks: Vec<MaterializedCodebook>,
    first_output_proj: Vec<f32>,
    rest_output_proj: Vec<f32>,
    pre_conv: OwnedConv,
    input_proj: OwnedConv,
    layers: Vec<OwnedLayer>,
    final_norm: Vec<f32>,
    output_proj: OwnedConv,
    upsample: [OwnedUpsampleStage; 2],
    decoder_input: OwnedConv,
    blocks: [OwnedBlock; 4],
    final_alpha: Vec<f32>,
    final_beta: Vec<f32>,
    final_conv: OwnedConv,
}

impl OwnedCodec {
    fn load(file: &SafetensorsFile) -> Self {
        let materialize = |prefix: &str| {
            let sum = widen(file, &format!("{prefix}.embedding_sum"));
            let usage = widen(file, &format!("{prefix}.cluster_usage"));
            MaterializedCodebook::from_unnormalized(&sum, &usage, 2_048, 256)
                .unwrap_or_else(|error| panic!("codebook `{prefix}` refuses: {error:?}"))
        };
        Self {
            first_codebook: materialize("decoder.quantizer.rvq_first.vq.layers.0._codebook"),
            rest_codebooks: (0..15)
                .map(|layer| {
                    materialize(&format!(
                        "decoder.quantizer.rvq_rest.vq.layers.{layer}._codebook"
                    ))
                })
                .collect(),
            first_output_proj: widen(file, "decoder.quantizer.rvq_first.output_proj.weight"),
            rest_output_proj: widen(file, "decoder.quantizer.rvq_rest.output_proj.weight"),
            pre_conv: OwnedConv::load(file, "decoder.pre_conv.conv"),
            input_proj: OwnedConv::load(file, "decoder.pre_transformer.input_proj"),
            layers: (0..8).map(|index| OwnedLayer::load(file, index)).collect(),
            final_norm: widen(file, "decoder.pre_transformer.norm.weight"),
            output_proj: OwnedConv::load(file, "decoder.pre_transformer.output_proj"),
            upsample: [
                OwnedUpsampleStage::load(file, 0),
                OwnedUpsampleStage::load(file, 1),
            ],
            decoder_input: OwnedConv::load(file, "decoder.decoder.0.conv"),
            blocks: [
                OwnedBlock::load(file, 1, 1_536, 768, 16, 8),
                OwnedBlock::load(file, 2, 768, 384, 10, 5),
                OwnedBlock::load(file, 3, 384, 192, 8, 4),
                OwnedBlock::load(file, 4, 192, 96, 6, 3),
            ],
            final_alpha: widen(file, "decoder.decoder.5.alpha"),
            final_beta: widen(file, "decoder.decoder.5.beta"),
            final_conv: OwnedConv::load(file, "decoder.decoder.6.conv"),
        }
    }

    fn quantizer(&self) -> SplitResidualVectorQuantizer<'_> {
        SplitResidualVectorQuantizer {
            first_codebook: &self.first_codebook,
            rest_codebooks: &self.rest_codebooks,
            first_output_proj: &self.first_output_proj,
            rest_output_proj: &self.rest_output_proj,
        }
    }

    fn pre_conv(&self) -> CausalConvWeights<'_> {
        CausalConvWeights {
            weight: &self.pre_conv.weight,
            bias: &self.pre_conv.bias,
            input_channels: 512,
            output_channels: 1_024,
            kernel: 3,
            dilation: 1,
        }
    }

    fn decoder_input(&self) -> CausalConvWeights<'_> {
        CausalConvWeights {
            weight: &self.decoder_input.weight,
            bias: &self.decoder_input.bias,
            input_channels: 1_024,
            output_channels: 1_536,
            kernel: 7,
            dilation: 1,
        }
    }

    fn final_conv(&self) -> CausalConvWeights<'_> {
        CausalConvWeights {
            weight: &self.final_conv.weight,
            bias: &self.final_conv.bias,
            input_channels: 96,
            output_channels: 1,
            kernel: 7,
            dilation: 1,
        }
    }

    fn decoder_weights<'a>(
        &'a self,
        layers: &'a [CodecTransformerLayerWeights<'a>],
    ) -> CodecDecoderWeights<'a> {
        CodecDecoderWeights {
            pre_conv: self.pre_conv(),
            pre_transformer: CodecPreTransformerWeights {
                input_proj: &self.input_proj.weight,
                input_proj_bias: &self.input_proj.bias,
                layers,
                final_norm: &self.final_norm,
                output_proj: &self.output_proj.weight,
                output_proj_bias: &self.output_proj.bias,
            },
            latent_upsample: [
                CodecUpsampleStageWeights {
                    transposed: self.upsample[0].borrow_transposed(),
                    convnext: self.upsample[0].borrow_convnext(),
                },
                CodecUpsampleStageWeights {
                    transposed: self.upsample[1].borrow_transposed(),
                    convnext: self.upsample[1].borrow_convnext(),
                },
            ],
            decoder_input: self.decoder_input(),
            decoder_blocks: [
                self.blocks[0].borrow(),
                self.blocks[1].borrow(),
                self.blocks[2].borrow(),
                self.blocks[3].borrow(),
            ],
            final_alpha_log: &self.final_alpha,
            final_beta_log: &self.final_beta,
            final_conv: self.final_conv(),
        }
    }
}

fn seam<'a>(mode: &'a str, name: &'a str) -> SeamRef<'a> {
    SeamRef {
        case: "synthetic-tone-en",
        mode,
        group: "codec_decode",
        seam: name,
    }
}

/// Compare one seam, recording rather than panicking, so a single run localizes every stage.
/// Gate one seam under the pinned-XFAIL ratchet, recording rather than panicking so a single run
/// reports every stage. Shape mismatches still panic inside the comparator — those are wiring
/// bugs, not tolerance questions.
fn check(failures: &mut Vec<String>, name: &str, expected: &NpyArray, actual: &[f32]) {
    let comparison = compare_f32(&expected.data, actual, CPU_TIER_TOLERANCE);
    if comparison.holds() {
        compare_exactly(name, expected, actual)
            .expect("a zero-difference summary must satisfy the exact comparator");
        if PINNED_DIVERGENCE
            .iter()
            .any(|(pinned, _, _)| *pinned == name)
        {
            Receipt::new(TEST_NAME, Outcome::Failed)
                .contract("ConformanceExact/L2")
                .seam(name)
                .reason(
                    "unexpected exact pass: remove the recorded XFAIL only after reviewing the \
                     arithmetic change",
                )
                .tolerance(CPU_TIER_TOLERANCE, CPU_TIER_TOLERANCE_SOURCE)
                .oracle_tier(OracleTier::CpuFp32Fallback)
                .detail(comparison.to_json())
                .emit();
            failures.push(format!(
                "{name}: unexpectedly exact — review, then move the pin"
            ));
        } else {
            eprintln!(
                "codec L2 parity: {name} — {} elements, exact",
                comparison.len
            );
        }
        return;
    }
    Receipt::new(TEST_NAME, Outcome::ExpectedFailure)
        .contract("ConformanceExact/L2")
        .seam(name)
        .reason("observed CPU-fp32 arithmetic divergence; exact parity remains open")
        .tolerance(CPU_TIER_TOLERANCE, CPU_TIER_TOLERANCE_SOURCE)
        .oracle_tier(OracleTier::CpuFp32Fallback)
        .detail(comparison.to_json())
        .emit();
    eprintln!(
        "XFAIL[{name}]: max_abs={:.16e} over_tolerance={}/{} cosine={:.12}",
        comparison.max_abs_diff, comparison.over_tolerance, comparison.len, comparison.cosine,
    );
    match PINNED_DIVERGENCE
        .iter()
        .find(|(pinned, _, _)| *pinned == name)
    {
        Some((_, max_abs, over))
            if *max_abs == comparison.max_abs_diff && *over == comparison.over_tolerance => {}
        Some((_, max_abs, over)) => failures.push(format!(
            "{name}: divergence moved off its pin — pinned (max_abs {max_abs:.16e}, over \
             {over}), observed (max_abs {:.16e}, over {})",
            comparison.max_abs_diff, comparison.over_tolerance,
        )),
        None => failures.push(format!(
            "{name}: no pinned divergence — observed (\"{name}\", {:.16e}, {})",
            comparison.max_abs_diff, comparison.over_tolerance,
        )),
    }
}

#[test]
fn contract_a_l2_codec_decoder_stages_cpu_fp32_exact() {
    let fixtures = match OracleFixtures::open_default() {
        Ok(fixtures) => fixtures,
        Err(error) => {
            skip(&format!("fixtures unavailable: {error}"));
            return;
        }
    };
    let checkpoint = checkpoint_path();
    if !checkpoint.is_file() {
        skip(&format!(
            "pinned checkpoint absent at {}",
            checkpoint.display()
        ));
        return;
    }
    let mode = "icl_non_streaming";
    if !fixtures.has_seam(&seam(mode, "codec_decoder.input.input")) {
        skip("codec_decode seams absent from the fixture pack");
        return;
    }

    let mut failures: Vec<String> = Vec::new();
    let file = SafetensorsFile::open(&checkpoint).expect("pinned speech tokenizer opens");
    let owned = OwnedCodec::load(&file);
    let layer_weights: Vec<CodecTransformerLayerWeights<'_>> =
        owned.layers.iter().map(OwnedLayer::borrow).collect();
    let config = CodecConfig::default();

    let (codes, frames) = load_codes(&fixtures, &seam(mode, "codec_decoder.input.input"));

    // Stage: SplitRVQ + pre_conv + input_proj, gated at the oracle's layer-00 input.
    let mut latents = vec![0.0f32; frames * config.codec_latent_dim];
    owned
        .quantizer()
        .decode(config, &codes, frames, &mut latents)
        .expect("oracle codes are valid ids");
    let mut pre_conv_out = vec![0.0f32; frames * 1_024];
    causal_conv1d(
        &latents,
        frames,
        512,
        &owned.pre_conv.weight,
        Some(&owned.pre_conv.bias),
        1_024,
        3,
        1,
        &mut pre_conv_out,
    );
    let mut projected = vec![0.0f32; frames * 512];
    f32ref::linear(
        &pre_conv_out,
        &owned.input_proj.weight,
        Some(&owned.input_proj.bias),
        frames,
        1_024,
        512,
        &mut projected,
    );
    let layer_00_input = fixtures
        .seam(
            &seam(mode, "codec_decoder.transformer_layer_00.input"),
            "args.0",
            0,
        )
        .expect("layer 00 input");
    check(
        &mut failures,
        "rvq+pre_conv+input_proj",
        &layer_00_input,
        &projected,
    );

    // Stage: our plain theta-10000 RoPE against the oracle's captured tables.
    let rope_cos = fixtures
        .seam(
            &seam(mode, "codec_decoder.transformer_layer_00.input"),
            "kwargs.position_embeddings.0",
            0,
        )
        .expect("rope cos");
    let rope_sin = fixtures
        .seam(
            &seam(mode, "codec_decoder.transformer_layer_00.input"),
            "kwargs.position_embeddings.1",
            0,
        )
        .expect("rope sin");
    let head_dim = config.attention_head_dim;
    let mut our_cos = vec![0.0f32; frames * head_dim];
    let mut our_sin = vec![0.0f32; frames * head_dim];
    for frame in 0..frames {
        codec_rope_rows(
            frame,
            head_dim,
            &mut our_cos[frame * head_dim..(frame + 1) * head_dim],
            &mut our_sin[frame * head_dim..(frame + 1) * head_dim],
        );
    }
    check(&mut failures, "codec_rope.cos", &rope_cos, &our_cos);
    check(&mut failures, "codec_rope.sin", &rope_sin, &our_sin);

    // Stage: each pre-transformer layer, stepwise, from the oracle's own layer input.
    for (layer, &weights) in layer_weights.iter().enumerate() {
        let input_name = format!("codec_decoder.transformer_layer_{layer:02}.input");
        let output_name = format!("codec_decoder.transformer_layer_{layer:02}.output");
        let input = fixtures
            .seam(&seam(mode, &input_name), "args.0", 0)
            .expect("layer input");
        let expected = fixtures
            .seam(&seam(mode, &output_name), "tensor", 0)
            .expect("layer output");
        let mut state = input.data.clone();
        let mut cache = CodecKvCache::new(config);
        for frame in 0..frames {
            forward_codec_transformer_step(
                config,
                weights,
                frame,
                &mut state[frame * 512..(frame + 1) * 512],
                &mut cache,
            );
        }
        check(&mut failures, &output_name, &expected, &state);
    }

    // Stage: final norm + output projection, gated at the first upsample stage's input.
    let layer_07_output = fixtures
        .seam(
            &seam(mode, "codec_decoder.transformer_layer_07.output"),
            "tensor",
            0,
        )
        .expect("layer 07 output");
    let mut normed = vec![0.0f32; frames * 512];
    f32ref::rms_norm_with_arithmetic(
        &layer_07_output.data,
        &owned.final_norm,
        config.rms_norm_eps,
        frames,
        512,
        f32ref::F32RmsNormArithmetic::Lanes4ReciprocalSqrt,
        &mut normed,
    );
    let mut transformer_out = vec![0.0f32; frames * 1_024];
    f32ref::linear(
        &normed,
        &owned.output_proj.weight,
        Some(&owned.output_proj.bias),
        frames,
        512,
        1_024,
        &mut transformer_out,
    );
    let upsample_0_input = fixtures
        .seam(&seam(mode, "codec_decoder.upsample_0_0.input"), "args.0", 0)
        .expect("upsample 0 input");
    check(
        &mut failures,
        "final_norm+output_proj",
        &upsample_0_input,
        &to_channel_major(&transformer_out, frames, 1_024),
    );

    // Stage: both latent upsample stages, transposed conv and ConvNeXt separately.
    for stage in 0..2 {
        let tconv_name = format!("codec_decoder.upsample_{stage}_0");
        let input = fixtures
            .seam(&seam(mode, &format!("{tconv_name}.input")), "args.0", 0)
            .expect("tconv input");
        let expected = fixtures
            .seam(&seam(mode, &format!("{tconv_name}.output")), "tensor", 0)
            .expect("tconv output");
        let (time_major, channels, in_frames) = to_time_major(&input);
        let mut out = vec![0.0f32; in_frames * 2 * 1_024];
        causal_transpose_conv1d(
            &time_major,
            in_frames,
            channels,
            &owned.upsample[stage].transposed.weight,
            Some(&owned.upsample[stage].transposed.bias),
            1_024,
            2,
            2,
            &mut out,
        );
        check(
            &mut failures,
            &tconv_name,
            &expected,
            &to_channel_major(&out, in_frames * 2, 1_024),
        );

        let convnext_name = format!("codec_decoder.upsample_{stage}_1");
        let input = fixtures
            .seam(&seam(mode, &format!("{convnext_name}.input")), "args.0", 0)
            .expect("convnext input");
        let expected = fixtures
            .seam(&seam(mode, &format!("{convnext_name}.output")), "tensor", 0)
            .expect("convnext output");
        let (time_major, channels, in_frames) = to_time_major(&input);
        let out = forward_convnext(
            &time_major,
            in_frames,
            channels,
            owned.upsample[stage].borrow_convnext(),
        );
        check(
            &mut failures,
            &convnext_name,
            &expected,
            &to_channel_major(&out, in_frames, channels),
        );
    }

    // Stage: decoder.0, the 1024×7 input convolution — the binding K=7168 worst case.
    let block_00_input = fixtures
        .seam(&seam(mode, "codec_decoder.block_00.input"), "args.0", 0)
        .expect("block 00 input");
    let block_00_output = fixtures
        .seam(&seam(mode, "codec_decoder.block_00.output"), "tensor", 0)
        .expect("block 00 output");
    let (time_major, _, latent_frames) = to_time_major(&block_00_input);
    let mut decoder_in = vec![0.0f32; latent_frames * 1_536];
    causal_conv1d(
        &time_major,
        latent_frames,
        1_024,
        &owned.decoder_input.weight,
        Some(&owned.decoder_input.bias),
        1_536,
        7,
        1,
        &mut decoder_in,
    );
    check(
        &mut failures,
        "codec_decoder.block_00",
        &block_00_output,
        &to_channel_major(&decoder_in, latent_frames, 1_536),
    );

    // Stage: the four BigVGAN upsampling blocks.
    for block in 0..4 {
        let input_name = format!("codec_decoder.block_{:02}.input", block + 1);
        let output_name = format!("codec_decoder.block_{:02}.output", block + 1);
        let input = fixtures
            .seam(&seam(mode, &input_name), "args.0", 0)
            .expect("block input");
        let expected = fixtures
            .seam(&seam(mode, &output_name), "tensor", 0)
            .expect("block output");
        let (time_major, channels, in_frames) = to_time_major(&input);
        let (out, out_frames) = forward_decoder_block(
            &time_major,
            in_frames,
            channels,
            owned.blocks[block].borrow(),
        );
        check(
            &mut failures,
            &output_name,
            &expected,
            &to_channel_major(&out, out_frames, owned.blocks[block].output_channels),
        );
    }

    // Stage: decoder.5, the final 96-channel SnakeBeta.
    let block_05_input = fixtures
        .seam(&seam(mode, "codec_decoder.block_05.input"), "args.0", 0)
        .expect("block 05 input");
    let block_05_output = fixtures
        .seam(&seam(mode, "codec_decoder.block_05.output"), "tensor", 0)
        .expect("block 05 output");
    let (mut time_major, channels, pcm_frames) = to_time_major(&block_05_input);
    snake_beta_in_place(
        &mut time_major,
        pcm_frames,
        &owned.final_alpha,
        &owned.final_beta,
    );
    check(
        &mut failures,
        "codec_decoder.block_05",
        &block_05_output,
        &to_channel_major(&time_major, pcm_frames, channels),
    );

    // Stage: decoder.6, the 96→1 output convolution.
    let block_06_input = fixtures
        .seam(&seam(mode, "codec_decoder.block_06.input"), "args.0", 0)
        .expect("block 06 input");
    let block_06_output = fixtures
        .seam(&seam(mode, "codec_decoder.block_06.output"), "tensor", 0)
        .expect("block 06 output");
    let (time_major, _, pcm_frames) = to_time_major(&block_06_input);
    let mut mono = vec![0.0f32; pcm_frames];
    causal_conv1d(
        &time_major,
        pcm_frames,
        96,
        &owned.final_conv.weight,
        Some(&owned.final_conv.bias),
        1,
        7,
        1,
        &mut mono,
    );
    check(
        &mut failures,
        "codec_decoder.block_06",
        &block_06_output,
        &mono,
    );

    // Whole-graph: the oracle's own codes through `decode_codec_offline`, against the decoder's
    // captured waveform output.
    let expected_waveform = fixtures
        .seam(&seam(mode, "codec_decoder.input.output"), "tensor", 0)
        .expect("decoder waveform output");
    let max_abs = expected_waveform
        .data
        .iter()
        .fold(0.0f32, |acc, value| acc.max(value.abs()));
    assert!(
        max_abs <= 1.0,
        "oracle decoder output exceeds ±1 ({max_abs}); our clamp would silently diverge — \
         the clamp seam needs its own gate before this comparison is meaningful"
    );
    let waveform = decode_codec_offline(
        config,
        owned.quantizer(),
        &owned.decoder_weights(&layer_weights),
        &codes,
        frames,
    )
    .expect("oracle codes decode");
    check(
        &mut failures,
        "decode_codec_offline[icl]",
        &expected_waveform,
        &waveform,
    );

    // Whole-graph again on the x-vector capture, against the final generated waveform.
    let xvector = "xvector_non_streaming";
    let (codes, frames) = load_codes(&fixtures, &seam(xvector, "codec_decoder.input.input"));
    let expected_waveform = fixtures
        .seam(&seam(xvector, "codec.generated_waveform"), "tensor", 0)
        .expect("generated waveform");
    let waveform = decode_codec_offline(
        config,
        owned.quantizer(),
        &owned.decoder_weights(&layer_weights),
        &codes,
        frames,
    )
    .expect("oracle codes decode");
    check(
        &mut failures,
        "decode_codec_offline[xvector]",
        &expected_waveform,
        &waveform,
    );

    assert!(
        failures.is_empty(),
        "codec decoder seams moved off the pinned CPU-fp32 divergence ratchet: {failures:#?}"
    );
}

/// The bead's standing gate (frankentts-p1-codec-hu7): **streaming == offline, BIT-IDENTICAL.**
///
/// The named reference is our whole-sequence offline decode (per the binding addendum — never the
/// official 25-frame-context chunker, whose behaviour past 300 frames is an upstream artifact).
/// The same 14-frame ICL code stream is pushed through [`CodecStreamingState`] under four packet
/// schedules — per-frame, packet-4, one whole-utterance packet, and an irregular fuzz schedule —
/// and every one must reproduce the offline PCM bit-for-bit and leave identical `clear()`-reset
/// behaviour for the next utterance. Chunking must not change arithmetic: any divergence here is
/// a wiring bug in the ring buffers or retained KV, never a tolerance question.
#[test]
fn streaming_decode_is_bit_identical_to_offline_across_packet_schedules() {
    let fixtures = match OracleFixtures::open_default() {
        Ok(fixtures) => fixtures,
        Err(error) => {
            skip(&format!("fixtures unavailable: {error}"));
            return;
        }
    };
    let checkpoint = checkpoint_path();
    if !checkpoint.is_file() {
        skip(&format!(
            "pinned checkpoint absent at {}",
            checkpoint.display()
        ));
        return;
    }
    let mode = "icl_non_streaming";
    if !fixtures.has_seam(&seam(mode, "codec_decoder.input.input")) {
        skip("codec_decode seams absent from the fixture pack");
        return;
    }

    let file = SafetensorsFile::open(&checkpoint).expect("pinned speech tokenizer opens");
    let owned = OwnedCodec::load(&file);
    let layer_weights: Vec<CodecTransformerLayerWeights<'_>> =
        owned.layers.iter().map(OwnedLayer::borrow).collect();
    let config = CodecConfig::default();
    let weights = owned.decoder_weights(&layer_weights);
    let (codes, frames) = load_codes(&fixtures, &seam(mode, "codec_decoder.input.input"));
    assert!(
        frames >= 8,
        "the packet schedules below need a multi-frame stream; this capture has {frames}"
    );

    let offline = decode_codec_offline(config, owned.quantizer(), &weights, &codes, frames)
        .expect("oracle codes decode offline");

    let schedules: [(&str, Vec<usize>); 6] = [
        ("packet-1", vec![1; frames]),
        ("packet-4", {
            let mut packets = vec![4; frames / 4];
            if frames % 4 > 0 {
                packets.push(frames % 4);
            }
            packets
        }),
        ("whole-utterance", vec![frames]),
        (
            "fuzz-3-1-5-2-3",
            vec![3, 1, 5, 2, frames.saturating_sub(11)],
        ),
        ("packet-2", vec![2; frames / 2]),
        ("tail-single", vec![frames - 1, 1]),
    ];
    let mut state = CodecStreamingState::new(config, &weights);
    let mut verdicts: Vec<String> = Vec::new();
    for (name, schedule) in &schedules {
        assert_eq!(
            schedule.iter().sum::<usize>(),
            frames,
            "schedule {name} must cover the stream exactly"
        );
        state.clear();
        let mut streamed = Vec::with_capacity(offline.len());
        let mut packet_pcm = Vec::new();
        let mut cursor = 0usize;
        for &packet in schedule {
            state
                .push(
                    owned.quantizer(),
                    &weights,
                    &codes[cursor * 16..(cursor + packet) * 16],
                    packet,
                    &mut packet_pcm,
                )
                .expect("streaming packet decodes");
            streamed.extend_from_slice(&packet_pcm);
            cursor += packet;
        }
        assert_eq!(
            streamed.len(),
            offline.len(),
            "schedule {name}: streamed sample count"
        );
        if let Some(index) = streamed
            .iter()
            .zip(&offline)
            .position(|(streamed, offline)| streamed.to_bits() != offline.to_bits())
        {
            let divergent = streamed
                .iter()
                .zip(&offline)
                .filter(|(streamed, offline)| streamed.to_bits() != offline.to_bits())
                .count();
            let max_abs = streamed
                .iter()
                .zip(&offline)
                .fold(0.0f32, |acc, (s, o)| acc.max((s - o).abs()));
            verdicts.push(format!(
                "{name}: FIRST divergence at sample {index} (frame {}), {divergent}/{} \
                 samples differ, max_abs {max_abs:e} (streamed {:e} vs offline {:e})",
                index / 1920,
                streamed.len(),
                streamed[index],
                offline[index]
            ));
        } else {
            eprintln!("schedule {name}: bit-identical over {frames} frames");
        }
    }

    assert!(
        verdicts.is_empty(),
        "streaming is NOT bit-identical to offline — ring-buffer/retained-KV state or a \
         diverging streaming code path (a whole-utterance-packet failure means the streaming \
         ops themselves differ from offline; a packet-only failure means carried state): \
         {verdicts:#?}"
    );
    Receipt::new(
        "contract_a_codec_streaming_equals_offline_bit_identical",
        Outcome::Passed,
    )
    .contract("ConformanceExact/streaming")
    .seam("codec.streaming_vs_offline.pcm")
    .reason(format!(
        "streaming == offline BIT-IDENTICAL over {frames} frames under 4 packet schedules \
         (packet-1, packet-4, whole-utterance, fuzz), state reused across schedules via clear()"
    ))
    .tolerance(
        0.0,
        "structural gate: same ops, same order — zero tolerance by construction",
    )
    .oracle_tier(OracleTier::CpuFp32Fallback)
    .emit();
}
