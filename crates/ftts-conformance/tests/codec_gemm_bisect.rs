//! Bisect probe: is the codec convolution seam's residual divergence purely a GEMM reduction order?
//!
//! `codec_decoder.block_00` is the cleanest convolution in the pack — one causal Conv1d
//! (1024 -> 1536, kernel 7) fed the oracle's own captured input, with no activation, normalization
//! or residual anywhere near it. Its unfolded reduction length K = 7168 is also the codec's binding
//! worst case, so whatever arithmetic explains this seam explains every other convolution.
//!
//! Our reference reduces that unfolded row with one scalar left-to-right f32 accumulator. The
//! reference stack unfolds identically (`[in_channel, tap]`-major columns) but reduces with BLAS —
//! on the pinned macOS oracle, Accelerate's SGEMM. This test runs the same seam three ways and
//! reports each one's divergence, so the follow-on question ("can a portable kernel reach exact, or
//! does exact require the oracle's own BLAS?") is answered with a measurement rather than a guess.
//!
//! It is a reporting probe, not a gate: it asserts only that the candidates ran.

use ftts_artifacts::safetensors::SafetensorsFile;
use ftts_conformance::npy::NpyArray;
use ftts_conformance::{
    compare::compare_f32,
    oracle::{CPU_TIER_TOLERANCE, CPU_TIER_TOLERANCE_SOURCE, OracleFixtures, SeamRef},
    report::{OracleTier, Outcome, Receipt},
};
use ftts_kernels::f32ref;
use std::path::{Path, PathBuf};

const TEST_NAME: &str = "codec_conv_gemm_reduction_bisect";

/// `decoder.0`: the binding codec convolution.
const INPUT_CHANNELS: usize = 1_024;
const OUTPUT_CHANNELS: usize = 1_536;
const KERNEL: usize = 7;
const REDUCTION_K: usize = INPUT_CHANNELS * KERNEL;

fn checkpoint_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../docs/truth-pack/snapshots/hf/speech_tokenizer/model.safetensors")
}

fn skip(reason: &str) {
    Receipt::new(TEST_NAME, Outcome::Skipped)
        .contract("ConformanceExact/L2")
        .seam("codec_decoder.block_00")
        .reason(reason)
        .tolerance(CPU_TIER_TOLERANCE, CPU_TIER_TOLERANCE_SOURCE)
        .oracle_tier(OracleTier::CpuFp32Fallback)
        .emit();
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

fn seam(name: &str) -> SeamRef<'_> {
    SeamRef {
        case: "synthetic-tone-en",
        mode: "icl_non_streaming",
        group: "codec_decode",
        seam: name,
    }
}

/// Fixture conv tensors are channel-major `[1, channels, frames]`.
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

fn to_channel_major(values: &[f32], frames: usize, channels: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; values.len()];
    for frame in 0..frames {
        for channel in 0..channels {
            out[channel * frames + frame] = values[frame * channels + channel];
        }
    }
    out
}

/// Unfold the causal convolution exactly as the reference's `im2col` does: one row per output
/// frame, columns ordered `[in_channel, tap]`, left-padded with zeros.
fn unfold(input: &[f32], frames: usize) -> Vec<f32> {
    let mut columns = vec![0.0f32; frames * REDUCTION_K];
    for frame in 0..frames {
        for input_channel in 0..INPUT_CHANNELS {
            for tap in 0..KERNEL {
                if let Some(source) = frame.checked_sub(KERNEL - 1 - tap) {
                    columns[frame * REDUCTION_K + input_channel * KERNEL + tap] =
                        input[source * INPUT_CHANNELS + input_channel];
                }
            }
        }
    }
    columns
}

fn report(candidate: &str, expected: &NpyArray, actual: &[f32]) {
    let comparison = compare_f32(&expected.data, actual, CPU_TIER_TOLERANCE);
    eprintln!(
        "CODEC_CONV_GEMM_BISECT candidate={candidate} max_abs={:.16e} over_tolerance={}/{} \
         cosine={:.12}",
        comparison.max_abs_diff, comparison.over_tolerance, comparison.len, comparison.cosine,
    );
    Receipt::new(TEST_NAME, Outcome::Passed)
        .contract("ConformanceExact/L2")
        .seam("codec_decoder.block_00")
        .reason(candidate)
        .tolerance(CPU_TIER_TOLERANCE, CPU_TIER_TOLERANCE_SOURCE)
        .oracle_tier(OracleTier::CpuFp32Fallback)
        .detail(comparison.to_json())
        .emit();
}

#[test]
fn codec_conv_gemm_reduction_bisect() {
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
    if !fixtures.has_seam(&seam("codec_decoder.block_00.input")) {
        skip("codec_decode seams absent from the fixture pack");
        return;
    }

    let file = SafetensorsFile::open(&checkpoint).expect("pinned speech tokenizer opens");
    let weight = widen(&file, "decoder.decoder.0.conv.weight");
    let bias = widen(&file, "decoder.decoder.0.conv.bias");
    assert_eq!(
        weight.len(),
        OUTPUT_CHANNELS * REDUCTION_K,
        "conv weight shape"
    );
    assert_eq!(bias.len(), OUTPUT_CHANNELS, "conv bias shape");

    let input = fixtures
        .seam(&seam("codec_decoder.block_00.input"), "args.0", 0)
        .expect("block 00 input");
    let expected = fixtures
        .seam(&seam("codec_decoder.block_00.output"), "tensor", 0)
        .expect("block 00 output");
    let (time_major, channels, frames) = to_time_major(&input);
    assert_eq!(channels, INPUT_CHANNELS, "block 00 input width");
    let columns = unfold(&time_major, frames);

    // Candidate 1: the production scalar reduction, bias seeded (today's pinned form).
    let mut scalar = vec![0.0f32; frames * OUTPUT_CHANNELS];
    f32ref::linear_with_accumulation(
        &columns,
        &weight,
        Some(&bias),
        frames,
        REDUCTION_K,
        OUTPUT_CHANNELS,
        f32ref::F32LinearAccumulation::Scalar,
        &mut scalar,
    );
    report(
        "scalar_unfolded",
        &expected,
        &to_channel_major(&scalar, frames, OUTPUT_CHANNELS),
    );

    // Candidates 2 and 3: the lane-partitioned reductions a SIMD kernel would produce.
    for (name, accumulation) in [
        ("lanes4_unfolded", f32ref::F32LinearAccumulation::Lanes4),
        ("lanes8_unfolded", f32ref::F32LinearAccumulation::Lanes8),
    ] {
        let mut candidate = vec![0.0f32; frames * OUTPUT_CHANNELS];
        f32ref::linear_with_accumulation(
            &columns,
            &weight,
            Some(&bias),
            frames,
            REDUCTION_K,
            OUTPUT_CHANNELS,
            accumulation,
            &mut candidate,
        );
        report(
            name,
            &expected,
            &to_channel_major(&candidate, frames, OUTPUT_CHANNELS),
        );
    }

    // Candidates 4 and 5: the oracle's own BLAS, with the bias added after a `beta = 0` product
    // and seeded into a `beta = 1` one. The second is literally the call the reference convolution
    // issues. Both fall back to the scalar reduction off macOS, where they simply restate
    // candidate 1.
    for (name, accumulation) in [
        (
            "accelerate_bias_trailing",
            f32ref::F32LinearAccumulation::Accelerate,
        ),
        (
            "accelerate_bias_seeded_beta1",
            f32ref::F32LinearAccumulation::AccelerateBiasSeeded,
        ),
    ] {
        let mut candidate = vec![0.0f32; frames * OUTPUT_CHANNELS];
        f32ref::linear_with_accumulation(
            &columns,
            &weight,
            Some(&bias),
            frames,
            REDUCTION_K,
            OUTPUT_CHANNELS,
            accumulation,
            &mut candidate,
        );
        report(
            name,
            &expected,
            &to_channel_major(&candidate, frames, OUTPUT_CHANNELS),
        );
    }
}
