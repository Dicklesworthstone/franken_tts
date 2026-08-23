//! Doctrine #2 gate (a) for int4: is W4A8 actually FASTER than W8A8, unpack cost included?
//!
//! Run: `cargo run --release -p ftts-kernels --example int4_speed_gate`
//!
//! # Why this exists before any listening test
//!
//! int4 halves the microdecoder's weight bytes, and its 5-layer body is re-read fifteen times per
//! frame, so cache residency is the whole thesis. But the unpack is not free: every nibble costs a
//! mask and a shift that Q8 does not pay. Doctrine #2 requires this measured on the real ISA, at
//! real shapes, INCLUDING that cost — and it comes first, because a lever that is slower does not
//! deserve anyone's ears.
//!
//! # What is compared, and why two comparisons rather than one
//!
//! * **scalar vs scalar** isolates the memory-traffic thesis from implementation quality. Both
//!   sides are plain Rust with no intrinsics, so a win here is the halved bytes talking.
//! * **int4 vs the DISPATCHED int8 route** is the shipping question. On this machine that route is
//!   NEON SDOT — a hand-written SIMD island — against a scalar int4 kernel with no SIMD unpack.
//!   Losing that comparison does NOT kill int4; it says int4 needs its own in-register unpack
//!   before it can compete, which is exactly NE-INH-004's documented escape clause.
//!
//! Shapes are the microdecoder's own: hidden 1024, intermediate 3072, 16 Q / 8 KV heads at
//! head_dim 128, per `MicrodecoderConfig::default`. Every projection runs at m = 1, the decode
//! geometry, because the microdecoder is fifteen sequential single-token steps.

use std::time::Instant;

use ftts_kernels::int4::{Q4Tier, QuantizedMatrixQ4, linear_q4, linear_q4_tier};
use ftts_kernels::int8::{Int8Tier, QuantizedMatrix, linear_q8, quantize_row_q8};

/// Deterministic operands spanning the representable range, so neither side is flattered by a
/// sparse or small-magnitude matrix.
fn deterministic(count: usize, seed: u64) -> Vec<f32> {
    let mut state = seed | 1;
    (0..count)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state >> 40) as f32 / 4096.0) - 0.5
        })
        .collect()
}

fn main() {
    // (label, n, k) for one microdecoder layer, at decode geometry m = 1.
    const HIDDEN: usize = 1024;
    const INTERMEDIATE: usize = 3072;
    const Q_WIDTH: usize = 16 * 128;
    const KV_WIDTH: usize = 8 * 128;
    let projections: &[(&str, usize, usize)] = &[
        ("q_proj", Q_WIDTH, HIDDEN),
        ("k_proj", KV_WIDTH, HIDDEN),
        ("v_proj", KV_WIDTH, HIDDEN),
        ("o_proj", HIDDEN, Q_WIDTH),
        ("gate_up", INTERMEDIATE * 2, HIDDEN),
        ("down_proj", HIDDEN, INTERMEDIATE),
    ];

    // Fifteen depths x five layers is how often this body is walked per frame.
    const ROUNDS: usize = 15 * 5;

    let dispatched = Int8Tier::dispatch();
    println!("dispatched int8 route: {}", dispatched.as_str());
    println!("shapes: microdecoder layer at m=1, {ROUNDS} rounds (15 depths x 5 layers)\n");
    // The ratio columns are SPEED ratios (q4's throughput relative to the q8 variant):
    // t_q8 / t_q4, so 1.0 means parity and 0.05 means q4 runs at 5% of q8's speed.
    println!(
        "{:<10} {:>6} {:>6} {:>10} {:>10} {:>10} {:>9} {:>9}",
        "proj", "n", "k", "q8-scalar", "q4-scalar", "q8-route", "spd/q8scl", "spd/route"
    );

    let mut total_q8_scalar = 0.0_f64;
    let mut total_q4 = 0.0_f64;
    let mut total_q8_route = 0.0_f64;
    let mut total_q4_neon = 0.0_f64;
    let q4_neon_dispatched = Q4Tier::dispatch();
    println!("dispatched int4 route: {}", q4_neon_dispatched.as_str());
    for &(label, n, k) in projections {
        let weight = deterministic(n * k, 0x51ED_0000 + n as u64);
        let activation = deterministic(k, 0xA0C7_0000 + k as u64);
        let mut x_q = vec![0_i8; k];
        let scale = quantize_row_q8(&activation, &mut x_q);

        let q8 = QuantizedMatrix::quantize(&weight, n, k);
        let q4 = QuantizedMatrixQ4::quantize(&weight, n, k);
        let mut out = vec![0.0_f32; n];

        // Warm the caches for whichever side runs first, so the ordering does not decide it.
        linear_q8(&x_q, &[scale], &q8, None, 1, &mut out, Int8Tier::Scalar);
        linear_q4(&x_q, &[scale], &q4, None, 1, &mut out);

        // INTERLEAVED repeats, reporting the minimum of each variant.
        //
        // A first attempt timed the three variants in sequential blocks and produced 98.60 ms for
        // k_proj against 5.14 ms for v_proj — identical 1024x1024 shapes, 19x apart. That is
        // scheduler and thermal noise, and it is precisely what doctrine #8's same-thermal-window
        // rule exists to prevent. Interleaving puts all three variants in the same window on every
        // repeat, and the minimum is the least noise-contaminated estimator of a deterministic
        // kernel: noise only ever adds time.
        const REPEATS: usize = 7;
        let mut q8_scalar = f64::MAX;
        let mut q4_ms = f64::MAX;
        let mut q8_route = f64::MAX;
        let mut q4_neon = f64::MAX;
        for _ in 0..REPEATS {
            let started = Instant::now();
            for _ in 0..ROUNDS {
                linear_q8(&x_q, &[scale], &q8, None, 1, &mut out, Int8Tier::Scalar);
            }
            q8_scalar = q8_scalar.min(started.elapsed().as_secs_f64() * 1000.0);

            let started = Instant::now();
            for _ in 0..ROUNDS {
                linear_q4(&x_q, &[scale], &q4, None, 1, &mut out);
            }
            q4_ms = q4_ms.min(started.elapsed().as_secs_f64() * 1000.0);

            let started = Instant::now();
            for _ in 0..ROUNDS {
                linear_q8(&x_q, &[scale], &q8, None, 1, &mut out, dispatched);
            }
            q8_route = q8_route.min(started.elapsed().as_secs_f64() * 1000.0);

            if q4_neon_dispatched == Q4Tier::NeonSdot {
                let started = Instant::now();
                for _ in 0..ROUNDS {
                    linear_q4_tier(&x_q, &[scale], &q4, None, 1, &mut out, Q4Tier::NeonSdot);
                }
                q4_neon = q4_neon.min(started.elapsed().as_secs_f64() * 1000.0);
            }
        }

        total_q8_scalar += q8_scalar;
        total_q4 += q4_ms;
        total_q8_route += q8_route;
        if q4_neon_dispatched == Q4Tier::NeonSdot {
            total_q4_neon += q4_neon;
        }

        if q4_neon_dispatched == Q4Tier::NeonSdot {
            println!(
                "{label:<10} {n:>6} {k:>6} {q8_scalar:>9.2}m {q4_ms:>9.2}m {q8_route:>9.2}m \
                 {q4_neon:>9.2}m {:>8.2}x {:>8.2}x {:>8.2}x",
                q8_scalar / q4_ms,
                q8_route / q4_ms,
                q8_route / q4_neon
            );
        } else {
            println!(
                "{label:<10} {n:>6} {k:>6} {q8_scalar:>9.2}m {q4_ms:>9.2}m {q8_route:>9.2}m \
                 {:>8.2}x {:>8.2}x",
                q8_scalar / q4_ms,
                q8_route / q4_ms
            );
        }
    }

    println!(
        "\ntotals (one layer, {ROUNDS} rounds): q8-scalar {total_q8_scalar:.1} ms, \
         q4-scalar {total_q4:.1} ms, q8-route {total_q8_route:.1} ms"
    );
    if q4_neon_dispatched == Q4Tier::NeonSdot {
        println!(
            "int4-neon vs shipping int8: {:.2}x  ({})",
            total_q8_route / total_q4_neon,
            if total_q4_neon < total_q8_route {
                "FASTER - gate (a) PASSES for the in-register unpack; the listening gate is \
                 now worth running"
            } else {
                "SLOWER - gate (a) FAILS even with the in-register unpack; NE-005 records a \
                 second strike"
            }
        );
    }

    // Bytes are the thesis; report them so the ratio can be read against the traffic it saves.
    let q8_bytes: usize = projections.iter().map(|(_, n, k)| n * k).sum();
    println!(
        "\nweight bytes per layer: q8 {:.1} MB, q4 {:.1} MB",
        q8_bytes as f64 / 1e6,
        q8_bytes as f64 / 2e6
    );
}
