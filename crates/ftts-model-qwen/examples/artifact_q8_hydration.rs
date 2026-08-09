//! Parity gate + A/B timer for artifact-native Q8 hydration.
//!
//! Compares the two ways an armed int8 route can obtain its `QuantizedMatrix` tables from a
//! canonical `.fttsq`:
//!
//! 1. today's route: widen every Q8 tensor to f32 at checkpoint load, then requantize at
//!    generator hydration (`TalkerLayerQuant::quantize`), and
//! 2. artifact-native: read the Q8 payload and scales straight out of the mapped artifact.
//!
//! For every talker and microdecoder layer it asserts the fused payload BYTES are identical and
//! reports the worst scale difference in ulps (the widen-then-requantize round trip can move a
//! scale by an ulp; the artifact's scale is the converter's canonical one). It then reports the
//! wall time of each hydration so the startup claim rests on a same-process A/B, not on
//! subtracting stage logs.
//!
//! ```sh
//! cargo run --release -p ftts-model-qwen --example artifact_q8_hydration -- \
//!     path/to/qwen3-tts-12hz-0.6b-base.fttsq
//! ```

use ftts_artifacts::fttsq::MappedFttsq;
use ftts_model_qwen::checkpoint::TalkerCheckpoint;
use ftts_model_qwen::microdecoder::{MicroLayerQuant, MicrodecoderConfig};
use ftts_model_qwen::talker::{TalkerConfig, TalkerLayerQuant};
use std::path::Path;
use std::time::Instant;

fn ulp_distance(a: f32, b: f32) -> u32 {
    if a == b {
        return 0;
    }
    let (a, b) = (a.to_bits() as i64, b.to_bits() as i64);
    (a - b).unsigned_abs() as u32
}

fn compare(
    label: &str,
    native: &ftts_kernels::int8::QuantizedMatrix,
    requant: &ftts_kernels::int8::QuantizedMatrix,
    worst_ulp: &mut u32,
) {
    assert_eq!(
        (native.n, native.k),
        (requant.n, requant.k),
        "{label}: geometry"
    );
    assert_eq!(native.data, requant.data, "{label}: payload bytes differ");
    for (row, (a, b)) in native.scales.iter().zip(requant.scales.iter()).enumerate() {
        let ulp = ulp_distance(*a, *b);
        if ulp > *worst_ulp {
            *worst_ulp = ulp;
            if ulp > 1 {
                eprintln!("{label} row {row}: scale {a:?} vs {b:?} = {ulp} ulps");
            }
        }
    }
}

fn main() {
    let artifact_path = std::env::args()
        .nth(1)
        .expect("usage: artifact_q8_hydration ARTIFACT.fttsq");
    let artifact_path = Path::new(&artifact_path);

    let load_started = Instant::now();
    let checkpoint = TalkerCheckpoint::load_fttsq(artifact_path).expect("hydrate checkpoint");
    let widen_elapsed = load_started.elapsed();
    let artifact = MappedFttsq::open(artifact_path).expect("map artifact");

    let talker_config = TalkerConfig::default();
    let micro_config = MicrodecoderConfig::default();
    let talker_layers = checkpoint.talker_layer_weights();
    let micro_layers = checkpoint.microdecoder_layer_weights();

    // Requantize hydration (the incumbent), timed alone.
    let requant_started = Instant::now();
    let requant_talker: Vec<TalkerLayerQuant> = talker_layers
        .iter()
        .map(|layer| TalkerLayerQuant::quantize(&talker_config, layer))
        .collect();
    let requant_micro: Vec<MicroLayerQuant> = micro_layers
        .iter()
        .map(|layer| MicroLayerQuant::quantize(&micro_config, layer))
        .collect();
    let requant_elapsed = requant_started.elapsed();

    // Artifact-native hydration (the lever), timed alone.
    let native_started = Instant::now();
    let native_talker = ftts_model_qwen::generate::talker_layers_from_artifact(
        &artifact,
        &talker_config,
        requant_talker.len(),
    )
    .expect("artifact carries every talker layer");
    let native_micro = ftts_model_qwen::generate::micro_layers_from_artifact(
        &artifact,
        &micro_config,
        requant_micro.len(),
    )
    .expect("artifact carries every microdecoder layer");
    let native_elapsed = native_started.elapsed();

    let mut worst_ulp = 0_u32;
    for (index, (native, requant)) in native_talker.iter().zip(&requant_talker).enumerate() {
        compare(&format!("talker.{index}.qkv"), &native.qkv, &requant.qkv, &mut worst_ulp);
        compare(&format!("talker.{index}.o"), &native.o_proj, &requant.o_proj, &mut worst_ulp);
        compare(&format!("talker.{index}.gate_up"), &native.gate_up, &requant.gate_up, &mut worst_ulp);
        compare(&format!("talker.{index}.down"), &native.down_proj, &requant.down_proj, &mut worst_ulp);
    }
    for (index, (native, requant)) in native_micro.iter().zip(&requant_micro).enumerate() {
        compare(&format!("micro.{index}.qkv"), &native.qkv, &requant.qkv, &mut worst_ulp);
        compare(&format!("micro.{index}.o"), &native.o_proj, &requant.o_proj, &mut worst_ulp);
        compare(&format!("micro.{index}.gate_up"), &native.gate_up, &requant.gate_up, &mut worst_ulp);
        compare(&format!("micro.{index}.down"), &native.down_proj, &requant.down_proj, &mut worst_ulp);
    }

    println!("layers: talker {} micro {}", native_talker.len(), native_micro.len());
    println!("payload bytes: IDENTICAL (asserted)");
    println!("worst scale ulp distance: {worst_ulp}");
    println!("checkpoint f32 hydration (widen): {widen_elapsed:?}");
    println!("requantize hydration: {requant_elapsed:?}");
    println!("artifact-native hydration: {native_elapsed:?}");
}
