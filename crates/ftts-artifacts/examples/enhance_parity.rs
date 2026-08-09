//! Parity harness: run the Rust FastEnhancer on a raw f32 LE mono 48 kHz input and dump
//! the enhanced waveform as raw f32 LE, for byte-level comparison against the pinned
//! PyTorch oracle dumps.
//!
//! Usage: enhance_parity <weights.safetensors> <input.f32> <output.f32>

use std::io::{Read, Write};

fn main() {
    let mut args = std::env::args().skip(1);
    let weights = args.next().expect("weights path");
    let input = args.next().expect("input raw-f32 path");
    let output = args.next().expect("output raw-f32 path");

    let enhancer = ftts_artifacts::enhance_loader::open_enhancer(&weights)
        .unwrap_or_else(|error| panic!("cannot load {weights}: {error}"));

    let mut bytes = Vec::new();
    std::fs::File::open(&input)
        .and_then(|mut f| f.read_to_end(&mut bytes))
        .unwrap_or_else(|error| panic!("cannot read {input}: {error}"));
    assert!(bytes.len() % 4 == 0, "raw f32 input must be 4-byte aligned");
    let wav: Vec<f32> = bytes
        .as_chunks::<4>().0.iter()
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    let started = std::time::Instant::now();
    let out = enhancer.enhance_48k(&wav);
    let elapsed = started.elapsed();
    let audio_seconds = wav.len() as f64 / 48_000.0;
    eprintln!(
        "enhanced {:.2}s of audio in {:.3}s (rtf {:.4})",
        audio_seconds,
        elapsed.as_secs_f64(),
        elapsed.as_secs_f64() / audio_seconds,
    );

    let mut out_bytes = Vec::with_capacity(out.len() * 4);
    for v in &out {
        out_bytes.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::File::create(&output)
        .and_then(|mut f| f.write_all(&out_bytes))
        .unwrap_or_else(|error| panic!("cannot write {output}: {error}"));
}
