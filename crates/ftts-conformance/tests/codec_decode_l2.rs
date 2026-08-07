//! L2 parity: every captured codec-decoder seam against the CPU-fp32 oracle.
//!
//! The ft7 pack captures the codec decoder at stage granularity: the SplitRVQ input codes, each of
//! the eight pre-transformer layers, both latent upsample stages, all seven `decoder.*` children,
//! and the decoder's own waveform output. Each stage here is fed the oracle's *captured input* —
//! never our own previous stage's output — so a divergence names the exact stage that computed it.
//!
//! Model-gated twice over — it needs both the fixture set and the pinned speech-tokenizer
//! checkpoint — and skips with SUCCESS and a named reason when either is absent.

use ftts_artifacts::safetensors::SafetensorsFile;
use ftts_conformance::npy::NpyArray;
use ftts_conformance::oracle::{OracleFixtures, SeamRef, compare_exactly};
use ftts_kernels::f32ref;
use ftts_model_qwen::codec::{
    CausalConvWeights, CausalTransposeConvWeights, CodecConfig, CodecConvNextWeights,
    CodecDecoderBlockWeights, CodecDecoderWeights, CodecKvCache, CodecPreTransformerWeights,
    CodecResidualUnitWeights, CodecTransformerLayerWeights, CodecUpsampleStageWeights,
    MaterializedCodebook, SplitResidualVectorQuantizer, causal_conv1d, causal_transpose_conv1d,
    codec_rope_rows, decode_codec_offline, forward_codec_transformer_step, forward_convnext,
    forward_decoder_block, snake_beta_in_place,
};
use std::path::{Path, PathBuf};

/// The pinned speech-tokenizer checkpoint, alongside the truth-pack snapshots.
fn checkpoint_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../docs/truth-pack/snapshots/hf/speech_tokenizer/model.safetensors")
}

fn skip(reason: &str) {
    eprintln!("SKIP[model-gated]: codec decoder L2 parity — {reason}");
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
        .chunks_exact(8)
        .map(|chunk| i64::from_le_bytes(chunk.try_into().expect("eight bytes")))
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
        Self {
            alpha: widen(file, &format!("{prefix}.block.0.alpha")),
            beta: widen(file, &format!("{prefix}.block.0.beta")),
            transposed: OwnedConv::load(file, &format!("{prefix}.block.1.conv")),
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
        Self {
            transposed: OwnedConv::load(file, &format!("{prefix}.0.conv")),
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

fn check(name: &str, expected: &NpyArray, actual: &[f32]) {
    match compare_exactly(name, expected, actual) {
        Ok(comparison) => eprintln!("codec L2 parity: {name} — {} elements, exact", comparison.len),
        Err(report) => panic!("{report}"),
    }
}

#[test]
fn codec_decoder_stages_match_the_cpu_oracle_seams() {
    let fixtures = match OracleFixtures::open_default() {
        Ok(fixtures) => fixtures,
        Err(error) => {
            skip(&format!("fixtures unavailable: {error}"));
            return;
        }
    };
    let checkpoint = checkpoint_path();
    if !checkpoint.is_file() {
        skip(&format!("pinned checkpoint absent at {}", checkpoint.display()));
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
        .seam(&seam(mode, "codec_decoder.transformer_layer_00.input"), "args.0", 0)
        .expect("layer 00 input");
    check("rvq+pre_conv+input_proj", &layer_00_input, &projected);

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
    check("codec_rope.cos", &rope_cos, &our_cos);
    check("codec_rope.sin", &rope_sin, &our_sin);

    // Stage: each pre-transformer layer, stepwise, from the oracle's own layer input.
    for layer in 0..8 {
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
                layer_weights[layer],
                frame,
                &mut state[frame * 512..(frame + 1) * 512],
                &mut cache,
            );
        }
        check(&output_name, &expected, &state);
    }

    // Stage: final norm + output projection, gated at the first upsample stage's input.
    let layer_07_output = fixtures
        .seam(&seam(mode, "codec_decoder.transformer_layer_07.output"), "tensor", 0)
        .expect("layer 07 output");
    let mut normed = vec![0.0f32; frames * 512];
    f32ref::rms_norm(
        &layer_07_output.data,
        &owned.final_norm,
        config.rms_norm_eps,
        frames,
        512,
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
        let (out, out_frames) =
            forward_decoder_block(&time_major, in_frames, channels, owned.blocks[block].borrow());
        check(
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
    check("codec_decoder.block_06", &block_06_output, &mono);

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
    check("decode_codec_offline[icl]", &expected_waveform, &waveform);

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
    check("decode_codec_offline[xvector]", &expected_waveform, &waveform);
}
