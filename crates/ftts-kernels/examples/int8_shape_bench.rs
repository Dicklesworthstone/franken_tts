//! Interleaved W8A8-vs-f32 GEMV/GEMM bench at the model's exact decode shapes.
//!
//! This is the per-shape bench required by `frankentts-k-int8-kernels-qhy` and the seed data for
//! the KernelPlan (`frankentts-b-w8a8-bench-1mu`). It measures the kernel seam only — activation
//! quantization included on the W8A8 side, allocation excluded on both sides — with interleaved
//! same-thermal-window rounds (NE-INH-007) and a cv% report so an incoherent capture is visible
//! instead of averaged away.
//!
//! Numbers printed here are PROVISIONAL_LOCAL_WIN candidates at best: they compare routes inside
//! this tree and never claim a pinned-incumbent ratio.
//!
//! ```sh
//! cargo run --release --locked -p ftts-kernels --example int8_shape_bench
//! ```

use ftts_kernels::f32ref;
use ftts_kernels::int8::{Int8Tier, QuantizedMatrix, linear_q8, quantize_row_q8};
use std::hint::black_box;
use std::time::Instant;

/// (label, n, k) — the seven distinct decode-path projections plus the seq-16 verify regime.
const SHAPES: &[(&str, usize, usize)] = &[
    ("q_proj/head 2048x1024", 2048, 1024),
    ("k/v_proj  1024x1024", 1024, 1024),
    ("o_proj    1024x2048", 1024, 2048),
    ("gate/up   3072x1024", 3072, 1024),
    ("down_proj 1024x3072", 1024, 3072),
];

const ROUNDS: usize = 12;
const WARMUP_ROUNDS: usize = 2;

fn pseudo_random_f32(len: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            ((z >> 40) as f32 / (1u64 << 23) as f32) - 1.0
        })
        .collect()
}

struct Stats {
    mean_us: f64,
    cv_percent: f64,
}

fn stats(samples: &[f64]) -> Stats {
    let mean = samples.iter().sum::<f64>() / samples.len() as f64;
    let variance =
        samples.iter().map(|s| (s - mean) * (s - mean)).sum::<f64>() / samples.len() as f64;
    Stats {
        mean_us: mean,
        cv_percent: 100.0 * variance.sqrt() / mean,
    }
}

#[allow(clippy::too_many_lines)]
fn main() {
    let tiers: Vec<Int8Tier> = Int8Tier::available();
    println!("int8 shape bench — tiers available: {:?}", {
        tiers.iter().map(|t| t.as_str()).collect::<Vec<_>>()
    });
    println!(
        "interleaved rounds={ROUNDS} (+{WARMUP_ROUNDS} warmup); per-sample = mean over calls in one round; cv% over rounds"
    );

    for &m in &[1_usize, 16] {
        println!(
            "\n== m = {m} {} ==",
            if m == 1 {
                "(decode GEMV)"
            } else {
                "(seq-16 verify GEMM)"
            }
        );
        for &(label, n, k) in SHAPES {
            let calls: usize = (32 / m).max(2);
            let weight = pseudo_random_f32(n * k, 0xbe0_0001 ^ (n as u64) << 20 ^ k as u64);
            let x = pseudo_random_f32(m * k, 0xbe0_0002 ^ (m as u64) << 32 ^ k as u64);
            let quantized = QuantizedMatrix::quantize(&weight, n, k);
            let mut out = vec![0.0_f32; m * n];
            let mut x_q = vec![0_i8; m * k];
            let mut x_scales = vec![0.0_f32; m];

            // One arm per route, all interleaved inside every round.
            let mut f32_samples = Vec::with_capacity(ROUNDS);
            let mut tier_samples: Vec<Vec<f64>> =
                tiers.iter().map(|_| Vec::with_capacity(ROUNDS)).collect();

            for round in 0..ROUNDS + WARMUP_ROUNDS {
                // f32 arm
                let start = Instant::now();
                for _ in 0..calls {
                    f32ref::linear(
                        black_box(&x),
                        black_box(&weight),
                        None,
                        m,
                        k,
                        n,
                        black_box(&mut out),
                    );
                }
                let f32_us = start.elapsed().as_secs_f64() * 1e6 / calls as f64;

                // W8A8 arms, including dynamic activation quantization each call.
                let mut this_round = Vec::with_capacity(tiers.len());
                for &tier in &tiers {
                    let start = Instant::now();
                    for _ in 0..calls {
                        for ((x_row, q_row), scale) in x
                            .chunks_exact(k)
                            .zip(x_q.chunks_exact_mut(k))
                            .zip(x_scales.iter_mut())
                        {
                            *scale = quantize_row_q8(black_box(x_row), q_row);
                        }
                        linear_q8(
                            black_box(&x_q),
                            black_box(&x_scales),
                            black_box(&quantized),
                            None,
                            m,
                            black_box(&mut out),
                            tier,
                        );
                    }
                    this_round.push(start.elapsed().as_secs_f64() * 1e6 / calls as f64);
                }

                if round >= WARMUP_ROUNDS {
                    f32_samples.push(f32_us);
                    for (samples, sample) in tier_samples.iter_mut().zip(&this_round) {
                        samples.push(*sample);
                    }
                }
            }

            let f32_stats = stats(&f32_samples);
            // The f32 reference loops rows outermost, so it streams the weight matrix once per
            // activation row (m times per call); the q8 kernel is weight-stationary and streams
            // it exactly once per call. The column reports actual weight bytes moved per second.
            let f32_bytes = (n * k * 4 * m) as f64;
            println!(
                "{label}  f32     {:9.1} us  cv {:4.1}%  ({:5.1} GB/s weight-stream)",
                f32_stats.mean_us,
                f32_stats.cv_percent,
                f32_bytes / (f32_stats.mean_us * 1e-6) / 1e9,
            );
            for (tier, samples) in tiers.iter().zip(&tier_samples) {
                let tier_stats = stats(samples);
                let q8_bytes = (n * k) as f64;
                let verdict = if tier_stats.cv_percent > 5.0 || f32_stats.cv_percent > 5.0 {
                    "REFUSED (cv>5%)"
                } else {
                    ""
                };
                println!(
                    "{label}  q8 {:9} {:9.1} us  cv {:4.1}%  ({:5.1} GB/s weight-stream)  x{:.2} vs f32 {verdict}",
                    tier.as_str(),
                    tier_stats.mean_us,
                    tier_stats.cv_percent,
                    q8_bytes / (tier_stats.mean_us * 1e-6) / 1e9,
                    f32_stats.mean_us / tier_stats.mean_us,
                );
            }
        }
    }
    println!(
        "\nNOTE: ratios above compare routes inside this tree (self-comparison = maintenance),\nnever a pinned incumbent. cv%>5 rows are refused, not averaged."
    );
}
