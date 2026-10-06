//! Per-level f32 packed-GEMM bench at the codec's real dense shapes (single thread).
//!
//! Every [`F32GemmLevel`] is bit-identical (pinned by `packed_gemm` tests), so this measures speed
//! only. Rounds are interleaved across levels in the same thermal window and a cv% is printed so
//! an incoherent capture is visible instead of averaged away. Numbers are PROVISIONAL_LOCAL_WIN
//! candidates comparing routes inside this tree, never a pinned-incumbent ratio.
//!
//! ```sh
//! cargo run --release --locked -p ftts-kernels --example f32_gemm_bench
//! BENCH_LEVEL=x86-avx512 cargo run --release --locked -p ftts-kernels --example f32_gemm_bench
//! ```
//!
//! `BENCH_LEVEL` restricts the run to one level, so a level can be timed in a process that has
//! never executed another (rules out cross-level state such as dirty-upper-register penalties).

use ftts_kernels::packed_gemm::{F32GemmLevel, linear_packed_at};
use std::hint::black_box;
use std::time::Instant;

/// (label, m, k, n): one 4-frame packet's worth of rows at each codec dense geometry.
const SHAPES: &[(&str, usize, usize, usize)] = &[
    ("decoder.0 conv k7 1024->1536", 16, 7168, 1536),
    ("decoder.1 unit conv1 k7 768", 128, 5376, 768),
    ("decoder.1 unit conv2 k1 768", 128, 768, 768),
    ("decoder.2 unit conv1 k7 384", 640, 2688, 384),
    ("decoder.3 unit conv1 k7 192", 2560, 1344, 192),
    ("decoder.4 unit conv1 k7 96", 7680, 672, 96),
    ("transformer qkv 512->1024", 4, 512, 1024),
    ("single-row GEMV 512->1024", 1, 512, 1024),
];

const ROUNDS: usize = 6;

fn deterministic(len: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((state >> 40) as f32 / (1u64 << 24) as f32) - 0.5
        })
        .collect()
}

fn main() {
    let levels: Vec<F32GemmLevel> = F32GemmLevel::available()
        .into_iter()
        .filter(|level| std::env::var("BENCH_LEVEL").map_or(true, |only| only == level.as_str()))
        .collect();
    for &(label, m, k, n) in SHAPES {
        let x = deterministic(m * k, 1);
        let weight = deterministic(n * k, 2);
        let bias = deterministic(n, 3);
        let mut out = vec![0.0_f32; m * n];
        let mut samples = vec![Vec::new(); levels.len()];
        for round in 0..=ROUNDS {
            for (slot, &level) in levels.iter().enumerate() {
                let start = Instant::now();
                linear_packed_at(level, &x, &weight, Some(&bias), m, k, n, &mut out);
                black_box(&out);
                if round > 0 {
                    samples[slot].push(start.elapsed().as_secs_f64() * 1e3);
                }
            }
        }
        let flops = 2.0 * (m * k * n) as f64;
        for (slot, &level) in levels.iter().enumerate() {
            let mean = samples[slot].iter().sum::<f64>() / ROUNDS as f64;
            let variance = samples[slot]
                .iter()
                .map(|s| (s - mean) * (s - mean))
                .sum::<f64>()
                / ROUNDS as f64;
            println!(
                "{label:<30} {:<11} {mean:9.2} ms  cv {:5.1}%  {:6.1} GFLOP/s",
                level.as_str(),
                100.0 * variance.sqrt() / mean,
                flops / (mean * 1e6),
            );
        }
    }
}
