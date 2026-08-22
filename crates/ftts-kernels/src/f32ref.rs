//! f32 reference kernels: the correctness baseline every optimized tier must reproduce.
//!
//! These are deliberately the obvious implementations. They exist so that a SIMD or int8 kernel has
//! something bit-comparable to be judged against (G1 > G2 — parity first, speed second), and so the
//! first end-to-end forward can be brought up without any unsafe at all. Nothing here is on the hot
//! path yet; nothing here should be "optimized" in place. When a fast tier lands it lands beside
//! these, with a test asserting the two agree.
//!
//! Accumulation is f32 to match the reference stack's CPU fp32 tier. In particular, RMSNorm widens
//! BF16 inputs to f32 and accumulates its variance in f32, exactly as the resolved QK-Norm contract
//! requires.

/// Reduction order used by [`linear_with_accumulation`].
///
/// The scalar order is the f32 reference used by production code. The lane orders are retained so
/// the CPU-fp32 fixture test can identify whether a BLAS-style partial reduction is responsible
/// for a layer-level arithmetic divergence.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum F32LinearAccumulation {
    /// One left-to-right f32 accumulator.
    Scalar,
    /// Four independent f32 partial accumulators, reduced in lane order.
    Lanes4,
    /// Eight independent f32 partial accumulators, reduced in lane order.
    Lanes8,
    /// Four FMA partial accumulators, reduced in lane order.
    FusedLanes4,
    /// Eight FMA partial accumulators, reduced in lane order.
    FusedLanes8,
    /// macOS Accelerate SGEMM, selected only by the CPU-fp32 parity harness.
    ///
    /// On every other target, this deliberately falls back to [`Self::Scalar`].
    Accelerate,
    /// [`Self::Accelerate`], with M = 1 calls pinned onto the M >= 2 GEMM kernel.
    ///
    /// See [`Self::AccelerateBiasSeededRowInvariant`] for the streaming == offline rationale;
    /// this is the same pinning for the `beta = 0` route (the codec's RVQ projections).
    AccelerateRowInvariant,
    /// macOS Accelerate SGEMM over a bias-seeded output, issued with `beta = 1`.
    ///
    /// This is the exact call `slow_conv2d_update_output_frame` makes for a convolution with a
    /// bias, and it differs from [`Self::Accelerate`] — which adds the bias after a `beta = 0`
    /// product — whenever the BLAS blocks its reduction. Like the other lane orders, it exists so
    /// the CPU-fp32 fixture can attribute a convolution's divergence; it falls back to
    /// [`Self::Scalar`] with a trailing bias on every non-macOS target.
    AccelerateBiasSeeded,
    /// [`Self::AccelerateBiasSeeded`], with M = 1 calls pinned onto the M >= 2 GEMM kernel.
    ///
    /// Accelerate routes M = 1 to a GEMV kernel whose reduction order differs from its (measured
    /// row-invariant) M >= 2 GEMM kernel. Seams whose streaming variant must equal whole-sequence
    /// decode bit-for-bit — the codec convolutions — need every M on the same kernel path, and
    /// accept drifting a single-frame call away from the oracle's own GEMV bits to get it. Seams
    /// the ORACLE itself computes at M = 1 (the speaker-encoder embedding head) must NOT use
    /// this: the GEMV path is the oracle-matching one there.
    AccelerateBiasSeededRowInvariant,
    /// One f64 accumulator, narrowed to f32 only at the store.
    ///
    /// Not a candidate for what the oracle did — it is an *attribution probe*. Every f32 lane
    /// order above is one guess at the oracle's reduction; this one removes the reduction's
    /// rounding entirely, so the residual it leaves at a seam is the part of that seam's
    /// divergence that a reduction order cannot explain. See `talker_layer_attribution`.
    WidenedF64,
}

/// Arithmetic used by [`rms_norm_with_arithmetic`] to form RMSNorm's scale.
///
/// The scalar reciprocal-square-root path is the f32 reference used by production code. The other
/// modes make the exact CPU-fp32 fixture able to discriminate reduction precision and reciprocal
/// placement without changing that normal path.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum F32RmsNormArithmetic {
    /// Left-to-right f32 reduction and `sqrt(value).recip()`.
    ScalarReciprocalSqrt,
    /// Left-to-right f32 reduction and `1.0 / sqrt(value)`.
    ScalarDivideSqrt,
    /// Four f32 partial sums, then `sqrt(value).recip()`.
    Lanes4ReciprocalSqrt,
    /// Eight f32 partial sums, then `sqrt(value).recip()`.
    Lanes8ReciprocalSqrt,
    /// Sixteen f32 partial sums, then `sqrt(value).recip()`.
    Lanes16ReciprocalSqrt,
    /// Thirty-two f32 partial sums, then `sqrt(value).recip()`.
    Lanes32ReciprocalSqrt,
    /// The reference stack's own cascade reduction over a **4-wide** vector, then
    /// `sqrt(value).recip()`. See [`torch_cascade_sum`].
    TorchCascade4ReciprocalSqrt,
    /// The reference stack's cascade reduction over an **8-wide** vector — the width an ARM build
    /// with `AT_BUILD_ARM_VEC256_WITH_SLEEF` uses, which the pinned oracle reports.
    TorchCascade8ReciprocalSqrt,
    /// f64 reduction and scale calculation, narrowed only at the final scale.
    F64ReciprocalSqrt,
}

impl F32RmsNormArithmetic {
    /// The variant that removes this operation's f32 reduction rounding, for attribution probes.
    pub const WIDENED_F64: Self = Self::F64ReciprocalSqrt;
}

/// Association used by [`silu_mul_in_place_with_arithmetic`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum F32SiluArithmetic {
    /// `x / (1 + exp(-x))`.
    Divide,
    /// `x * (1 / (1 + exp(-x)))`, matching `x * sigmoid(x)` association.
    MultiplyReciprocal,
    /// The whole expression in f64, narrowed only at the store — an attribution probe, not a
    /// candidate for what the oracle did.
    WidenedF64,
}

/// Normalization form used by [`softmax_rows_with_arithmetic`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum F32SoftmaxArithmetic {
    /// Form one reciprocal then multiply every exponent by it.
    ReciprocalMultiply,
    /// Divide every exponent by the sum directly.
    Divide,
    /// Exponentiate, sum and normalize in f64, narrowing only at the store — an attribution
    /// probe, not a candidate for what the oracle did.
    WidenedF64,
}

/// Row-major matrix-vector/matrix-matrix product in the layout PyTorch `Linear` stores.
///
/// `x` is `[m, k]`, `weight` is `[n, k]` (out-features major, as `nn.Linear` stores it), and the
/// result is `[m, n]`. Bias is optional because every attention/MLP projection in this model is
/// bias-free; only `text_projection` carries one.
///
/// # Panics
///
/// Panics if any slice length disagrees with `m`, `k`, `n`.
pub fn linear(
    x: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    m: usize,
    k: usize,
    n: usize,
    out: &mut [f32],
) {
    linear_with_accumulation(x, weight, bias, m, k, n, F32LinearAccumulation::Scalar, out);
}

/// Same operation as [`linear`], with an explicitly chosen f32 dot-product reduction order.
///
/// This exists for parity forensics. The normal [`linear`] entry point remains the scalar,
/// left-to-right reference.
#[allow(clippy::too_many_arguments)]
pub fn linear_with_accumulation(
    x: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    m: usize,
    k: usize,
    n: usize,
    accumulation: F32LinearAccumulation,
    out: &mut [f32],
) {
    assert_eq!(x.len(), m * k, "x must be [m, k]");
    assert_eq!(weight.len(), n * k, "weight must be [n, k]");
    assert_eq!(out.len(), m * n, "out must be [m, n]");
    if let Some(bias) = bias {
        assert_eq!(bias.len(), n, "bias must be [n]");
    }

    if matches!(
        accumulation,
        F32LinearAccumulation::AccelerateBiasSeeded
            | F32LinearAccumulation::AccelerateBiasSeededRowInvariant
    ) {
        // Seed `out` with the bias and let the BLAS accumulate onto it, exactly as the reference
        // convolution's `beta = 1` GEMM does.
        match bias {
            Some(bias) => {
                for row in out.chunks_exact_mut(n) {
                    row.copy_from_slice(bias);
                }
            }
            None => out.fill(0.0),
        }
        let row_invariant = accumulation == F32LinearAccumulation::AccelerateBiasSeededRowInvariant;
        if accelerate_sgemm(x, weight, m, k, n, 1.0, row_invariant, out) {
            return;
        }
        out.fill(0.0);
    }

    if matches!(
        accumulation,
        F32LinearAccumulation::Accelerate | F32LinearAccumulation::AccelerateRowInvariant
    ) && accelerate_sgemm(
        x,
        weight,
        m,
        k,
        n,
        0.0,
        accumulation == F32LinearAccumulation::AccelerateRowInvariant,
        out,
    ) {
        if let Some(bias) = bias {
            for row in out.chunks_exact_mut(n) {
                for (value, offset) in row.iter_mut().zip(bias) {
                    *value += offset;
                }
            }
        }
        return;
    }

    // The BLAS-less fall-through: no platform GEMM was available (or none was asked for).
    //
    // Every dense op in the codec — both convolutions via im2col, the ConvNeXt pointwise pair, the
    // transformer's q/k/v/o and FFN — reaches this one function, so the kernel chosen here is the
    // codec's entire arithmetic budget. In the browser that budget measured 89.1 s of a 97.3 s
    // frame (92%), because a per-element dot product re-reads the activation row once per output
    // column and reuses no weight at all.
    //
    // `linear_packed` is the register-tiled, panel-packed replacement, and it is safe to
    // substitute HERE specifically because of what it does not change: each output element still
    // accumulates over ascending `k` into its own slot, one add at a time, so it is bit-identical
    // to the `Scalar` reduction (`packed_matches_scalar_bit_for_bit`).
    //
    // Two regimes, and only one of them changes any bits:
    //
    //   * native non-macOS — a denied BLAS request already degrades to lanes = 1, i.e. the scalar
    //     order. Packed reproduces it exactly, so this is a pure speed change with NO numerics
    //     change and nothing to ledger.
    //   * wasm — the fall-through used eight independent partial chains (a non-reference order,
    //     adopted because a single scalar chain cannot be autovectorized). Packed replaces that
    //     with the reference's own order, so the browser moves CLOSER to the oracle while getting
    //     faster. A speed lever that tightens parity rather than loosening it.
    //
    // `m == 1` keeps the dot path: with one row there is nothing to amortize a packed panel over,
    // and NE-004 measured register blocking as neutral at that geometry.
    //
    // The gate below also keeps the forensic probe orders OFF this path. `Lanes4/8`,
    // `FusedLanes4/8`, and `WidenedF64` exist solely so the parity harness can reproduce a
    // specific non-scalar reduction order; routing them through packed/team would silently
    // hand every probe the Scalar order and turn the attribution sweep into Scalar-vs-Scalar
    // under five labels. Only the orders packed provably reproduces may take the shortcut:
    // `Scalar` itself, and the denied-BLAS `Accelerate*` degradations documented above.
    let packed_preserves_order = matches!(
        accumulation,
        F32LinearAccumulation::Scalar
            | F32LinearAccumulation::Accelerate
            | F32LinearAccumulation::AccelerateRowInvariant
            | F32LinearAccumulation::AccelerateBiasSeeded
            | F32LinearAccumulation::AccelerateBiasSeededRowInvariant
    );
    if m > 1 && packed_preserves_order {
        // Hand it to the team when one is armed. This is the codec's entire arithmetic budget —
        // 92% of browser frame time — and it ran on one thread while every worker sat parked.
        //
        // Partitioning is a pure speed knob: stripes are disjoint columns, no reduction is split,
        // so the parallel result is bit-identical per element to the serial packed kernel. The
        // work must also be big enough to pay for a dispatch, hence the floor; and a worker thread
        // that is itself inside a kernel must never re-dispatch (`thread_bypassed`).
        const TEAM_FLOOR: usize = 64 * 1024;
        if m * n >= TEAM_FLOOR
            && !crate::team::thread_bypassed()
            && let Some(team) = crate::team::armed()
        {
            team.linear_f32(x, weight, bias, m, k, n, out);
            return;
        }
        crate::packed_gemm::linear_packed(x, weight, bias, m, k, n, out);
        return;
    }
    // A single-row call — the interactive profile's per-frame codec GEMV, which is most of the
    // browser's remaining frame budget — takes the same team route when one is armed. The exact-
    // ness argument is identical to the batch arm above: column stripes never split a reduction,
    // and packed's m = 1 shapes are pinned bit-equal to this very scalar dot by
    // `packed_matches_scalar_bit_for_bit`. NE-004 does not cover this lever: it measured
    // single-thread register blocking on the int8 GEMV as neutral, whereas this is worker
    // parallelism on the f32 path — a different kernel on a different axis. Without an armed
    // team the scalar dot below stays, exactly as before. The floor is computed over n * k
    // because single-row work scales with depth, not with rows; the old m * n form could never
    // admit one row at any real geometry.
    if m == 1 && packed_preserves_order {
        const GEMV_FLOOR: usize = 64 * 1024;
        if n * k >= GEMV_FLOOR
            && !crate::team::thread_bypassed()
            && let Some(team) = crate::team::armed()
        {
            team.linear_f32(x, weight, bias, m, k, n, out);
            return;
        }
    }

    for row in 0..m {
        let x_row = &x[row * k..row * k + k];
        for col in 0..n {
            let w_row = &weight[col * k..col * k + k];
            let sum = dot_with_accumulation(x_row, w_row, accumulation);
            out[row * n + col] = bias.map_or(sum, |b| sum + b[col]);
        }
    }
}

fn dot_with_accumulation(x: &[f32], weight: &[f32], accumulation: F32LinearAccumulation) -> f32 {
    assert_eq!(x.len(), weight.len(), "dot-product inputs must match");
    match accumulation {
        F32LinearAccumulation::Scalar => {
            let mut sum = 0.0f32;
            for index in 0..x.len() {
                sum += x[index] * weight[index];
            }
            sum
        }
        F32LinearAccumulation::WidenedF64 => {
            let mut sum = 0.0f64;
            for index in 0..x.len() {
                sum += f64::from(x[index]) * f64::from(weight[index]);
            }
            sum as f32
        }
        F32LinearAccumulation::Lanes4
        | F32LinearAccumulation::Lanes8
        | F32LinearAccumulation::FusedLanes4
        | F32LinearAccumulation::FusedLanes8
        | F32LinearAccumulation::Accelerate
        | F32LinearAccumulation::AccelerateRowInvariant
        | F32LinearAccumulation::AccelerateBiasSeeded
        | F32LinearAccumulation::AccelerateBiasSeededRowInvariant => {
            let lanes = match accumulation {
                F32LinearAccumulation::Lanes4 => 4,
                F32LinearAccumulation::Lanes8 => 8,
                F32LinearAccumulation::FusedLanes4 => 4,
                F32LinearAccumulation::FusedLanes8 => 8,
                F32LinearAccumulation::Accelerate
                | F32LinearAccumulation::AccelerateRowInvariant
                | F32LinearAccumulation::AccelerateBiasSeeded
                | F32LinearAccumulation::AccelerateBiasSeededRowInvariant => {
                    // A denied BLAS request degrades to lanes = 1 (the scalar order) on native
                    // targets — pinned behavior for Linux parity. On wasm32 that scalar f32
                    // reduction chain cannot be autovectorized (f32 addition is not
                    // reassociable) and profiled as 71% of ALL synthesis time; eight partial
                    // chains give the compiler independent accumulators. Both orders live in
                    // the same "correct, not exact" contract the denied-BLAS path already
                    // declares.
                    // PARITY EXPERIMENT: wasm takes the native reduction order.
                    //
                    // Eight partial chains autovectorize where one cannot, but f32 addition is
                    // not associative, so the two orders give different sums — and these sums
                    // feed the codec head's logits, where a one-ulp difference flips a sampled
                    // token and the whole utterance diverges. Measured browser-vs-CLI on the same
                    // text/voice/seed: 0.4% identical samples, -4.5 dB SNR at best alignment,
                    // 40 frames against 41. Same words, different performance.
                    1
                }
                F32LinearAccumulation::Scalar | F32LinearAccumulation::WidenedF64 => {
                    unreachable!("scalar and widened orders are handled above")
                }
            };
            let mut partial = [0.0f32; 8];
            for index in 0..x.len() {
                let lane = index % lanes;
                partial[lane] = match accumulation {
                    F32LinearAccumulation::FusedLanes4 | F32LinearAccumulation::FusedLanes8 => {
                        x[index].mul_add(weight[index], partial[lane])
                    }
                    F32LinearAccumulation::Scalar
                    | F32LinearAccumulation::Lanes4
                    | F32LinearAccumulation::Lanes8
                    | F32LinearAccumulation::Accelerate
                    | F32LinearAccumulation::AccelerateRowInvariant
                    | F32LinearAccumulation::AccelerateBiasSeeded
                    | F32LinearAccumulation::AccelerateBiasSeededRowInvariant
                    | F32LinearAccumulation::WidenedF64 => partial[lane] + x[index] * weight[index],
                };
            }
            let mut sum = 0.0f32;
            for value in &partial[..lanes] {
                sum += *value;
            }
            sum
        }
    }
}

/// Attempts the same row-major SGEMM backend recorded by the pinned macOS CPU-fp32 oracle.
///
/// The test-only [`F32LinearAccumulation::Accelerate`] selector keeps this foreign call out of the
/// normal safe-Rust reference path. Targets without the opt-in macOS backend return `false`, so
/// their scalar fallback remains bit-for-bit the ordinary reference implementation.
#[cfg(all(feature = "accelerate-sgemm", target_os = "macos"))]
#[allow(clippy::too_many_arguments)]
fn accelerate_sgemm(
    x: &[f32],
    weight: &[f32],
    m: usize,
    k: usize,
    n: usize,
    beta: f32,
    row_invariant: bool,
    out: &mut [f32],
) -> bool {
    // Accelerate routes M = 1 to a GEMV kernel whose reduction order differs from its M >= 2
    // GEMM kernel in the last ulps, while M >= 2 is row-invariant (measured: M of 2, 3, 4 and 14
    // produce bit-identical rows). A streaming decode presents the same convolution at M = packet
    // while offline presents M = utterance, so without this pinning the streaming == offline gate
    // fails on every 1-frame packet. Under `row_invariant`, present M = 1 as a duplicated-row
    // M = 2 call and keep row 0, so every M reduces on the same kernel path. Seams the ORACLE
    // itself computes at M = 1 (the speaker-encoder embedding head) must NOT request this:
    // the GEMV bits ARE the oracle's bits there.
    if row_invariant && m == 1 {
        let mut doubled_x = Vec::with_capacity(2 * k);
        doubled_x.extend_from_slice(x);
        doubled_x.extend_from_slice(x);
        let mut doubled_out = Vec::with_capacity(2 * n);
        doubled_out.extend_from_slice(out);
        doubled_out.extend_from_slice(out);
        if !accelerate_sgemm(&doubled_x, weight, 2, k, n, beta, false, &mut doubled_out) {
            return false;
        }
        out.copy_from_slice(&doubled_out[..n]);
        return true;
    }
    let m = i32::try_from(m).expect("SGEMM rows fit CBLAS i32 dimensions");
    let k = i32::try_from(k).expect("SGEMM reduction fits CBLAS i32 dimensions");
    let n = i32::try_from(n).expect("SGEMM columns fit CBLAS i32 dimensions");
    // SAFETY: `linear_with_accumulation` proves the row-major slice lengths before this call.
    // `x` is M×K, `weight` is N×K and is passed transposed, and `out` is M×N. All pointers remain
    // valid and non-overlapping for the full synchronous CBLAS call.
    unsafe {
        cblas_sgemm(
            CBLAS_ROW_MAJOR,
            CBLAS_NO_TRANSPOSE,
            CBLAS_TRANSPOSE,
            m,
            n,
            k,
            1.0,
            x.as_ptr(),
            k,
            weight.as_ptr(),
            k,
            beta,
            out.as_mut_ptr(),
            n,
        );
    }
    true
}

// The stub must mirror the real Accelerate entry's eight parameters exactly so both cfg
// arms are call-site identical; the arity is the contract, not a design smell.
#[allow(clippy::too_many_arguments)]
#[cfg(not(all(feature = "accelerate-sgemm", target_os = "macos")))]
fn accelerate_sgemm(
    _x: &[f32],
    _weight: &[f32],
    _m: usize,
    _k: usize,
    _n: usize,
    _beta: f32,
    _row_invariant: bool,
    _out: &mut [f32],
) -> bool {
    false
}

#[cfg(all(feature = "accelerate-sgemm", target_os = "macos"))]
const CBLAS_ROW_MAJOR: i32 = 101;
#[cfg(all(feature = "accelerate-sgemm", target_os = "macos"))]
const CBLAS_NO_TRANSPOSE: i32 = 111;
#[cfg(all(feature = "accelerate-sgemm", target_os = "macos"))]
const CBLAS_TRANSPOSE: i32 = 112;

#[cfg(all(feature = "accelerate-sgemm", target_os = "macos"))]
#[link(name = "Accelerate", kind = "framework")]
unsafe extern "C" {
    fn cblas_sgemm(
        order: i32,
        trans_a: i32,
        trans_b: i32,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        a: *const f32,
        lda: i32,
        b: *const f32,
        ldb: i32,
        beta: f32,
        c: *mut f32,
        ldc: i32,
    );
}

/// Which `sin`/`exp` implementation an elementwise parity probe evaluates.
///
/// The reference stack does not call the scalar libm for large tensors: its CPU elementwise kernels
/// dispatch through a vectorized `Vectorized<float>`, whose transcendentals are ~1-ulp routines
/// rather than correctly-rounded ones. `codec_snake_bisect` established that the SnakeBeta seam's
/// entire residual divergence lives in exactly these two functions — every other operation in that
/// expression is a correctly-rounded f32 `*`, `+` or `/` with no freedom at all — so identifying
/// *which* vectorized routine the pinned oracle used is the whole remaining question there.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum F32Transcendental {
    /// Rust's scalar `f32::sin` / `f32::exp`, i.e. the platform libm. The production reference.
    ScalarLibm,
    /// macOS Accelerate vForce (`vvsinf` / `vvexpf`), selected only by the parity harness.
    ///
    /// On every other target this deliberately falls back to [`Self::ScalarLibm`].
    AccelerateVForce,
    /// SLEEF's 1-ulp `Sleef_sinf_u10` / `Sleef_expf_u10`, ported to safe Rust in [`crate::sleef`].
    ///
    /// This is the routine an AArch64 `Vectorized<float>` actually dispatches to, so it is the
    /// candidate the vForce probe was only ever standing in for — and unlike vForce it is portable
    /// and could therefore be adopted into production if it measures exact.
    SleefU10,
}

/// Fills `out` with `sin(x)` under the selected implementation.
///
/// # Panics
///
/// Panics if `out` is not the same length as `x`.
pub fn sin_with(x: &[f32], implementation: F32Transcendental, out: &mut [f32]) {
    assert_eq!(x.len(), out.len(), "sin output must match its input");
    if implementation == F32Transcendental::AccelerateVForce && vforce_sin(x, out) {
        return;
    }
    if implementation == F32Transcendental::SleefU10 {
        for (value, target) in x.iter().zip(out.iter_mut()) {
            *target = crate::sleef::sinf_u10(*value);
        }
        return;
    }
    for (value, target) in x.iter().zip(out.iter_mut()) {
        *target = value.sin();
    }
}

/// Fills `out` with `exp(x)` under the selected implementation.
///
/// # Panics
///
/// Panics if `out` is not the same length as `x`.
pub fn exp_with(x: &[f32], implementation: F32Transcendental, out: &mut [f32]) {
    assert_eq!(x.len(), out.len(), "exp output must match its input");
    if implementation == F32Transcendental::AccelerateVForce && vforce_exp(x, out) {
        return;
    }
    if implementation == F32Transcendental::SleefU10 {
        for (value, target) in x.iter().zip(out.iter_mut()) {
            *target = crate::sleef::expf_u10(*value);
        }
        return;
    }
    for (value, target) in x.iter().zip(out.iter_mut()) {
        *target = value.exp();
    }
}

#[cfg(all(feature = "accelerate-sgemm", target_os = "macos"))]
fn vforce_sin(x: &[f32], out: &mut [f32]) -> bool {
    let count = i32::try_from(x.len()).expect("vForce length fits i32");
    // SAFETY: `sin_with` proved the two slices have equal length, and they are distinct live
    // allocations for the duration of this synchronous call.
    unsafe { vvsinf(out.as_mut_ptr(), x.as_ptr(), &raw const count) };
    true
}

#[cfg(all(feature = "accelerate-sgemm", target_os = "macos"))]
fn vforce_exp(x: &[f32], out: &mut [f32]) -> bool {
    let count = i32::try_from(x.len()).expect("vForce length fits i32");
    // SAFETY: as in `vforce_sin`.
    unsafe { vvexpf(out.as_mut_ptr(), x.as_ptr(), &raw const count) };
    true
}

#[cfg(not(all(feature = "accelerate-sgemm", target_os = "macos")))]
fn vforce_sin(_x: &[f32], _out: &mut [f32]) -> bool {
    false
}

#[cfg(not(all(feature = "accelerate-sgemm", target_os = "macos")))]
fn vforce_exp(_x: &[f32], _out: &mut [f32]) -> bool {
    false
}

#[cfg(all(feature = "accelerate-sgemm", target_os = "macos"))]
#[link(name = "Accelerate", kind = "framework")]
unsafe extern "C" {
    fn vvsinf(out: *mut f32, x: *const f32, count: *const i32);
    fn vvexpf(out: *mut f32, x: *const f32, count: *const i32);
}

/// Qwen3 RMSNorm: `x * rsqrt(mean(x^2) + eps) * weight`, weight-only, no centering.
///
/// # Panics
///
/// Panics if `x` is not `rows * dim` elements or `weight` is not `dim`.
pub fn rms_norm(x: &[f32], weight: &[f32], eps: f32, rows: usize, dim: usize, out: &mut [f32]) {
    rms_norm_with_arithmetic(
        x,
        weight,
        eps,
        rows,
        dim,
        F32RmsNormArithmetic::ScalarReciprocalSqrt,
        out,
    );
}

/// Same operation as [`rms_norm`], with an explicitly selected reduction and scale calculation.
///
/// This entry point is for CPU-fp32 parity forensics; [`rms_norm`] remains the normal scalar f32
/// reference path.
pub fn rms_norm_with_arithmetic(
    x: &[f32],
    weight: &[f32],
    eps: f32,
    rows: usize,
    dim: usize,
    arithmetic: F32RmsNormArithmetic,
    out: &mut [f32],
) {
    assert_eq!(x.len(), rows * dim, "x must be [rows, dim]");
    assert_eq!(weight.len(), dim, "weight must be [dim]");
    assert_eq!(out.len(), rows * dim, "out must be [rows, dim]");

    for row in 0..rows {
        let src = &x[row * dim..row * dim + dim];
        let scale = rms_scale(src, eps, arithmetic);
        for index in 0..dim {
            out[row * dim + index] = src[index] * scale * weight[index];
        }
    }
}

fn rms_scale(src: &[f32], eps: f32, arithmetic: F32RmsNormArithmetic) -> f32 {
    match arithmetic {
        F32RmsNormArithmetic::ScalarReciprocalSqrt => {
            let sum = sum_squares_f32(src, 1);
            (sum / src.len() as f32 + eps).sqrt().recip()
        }
        F32RmsNormArithmetic::ScalarDivideSqrt => {
            let sum = sum_squares_f32(src, 1);
            1.0f32 / (sum / src.len() as f32 + eps).sqrt()
        }
        F32RmsNormArithmetic::Lanes4ReciprocalSqrt => {
            let sum = sum_squares_f32(src, 4);
            (sum / src.len() as f32 + eps).sqrt().recip()
        }
        F32RmsNormArithmetic::Lanes8ReciprocalSqrt => {
            let sum = sum_squares_f32(src, 8);
            (sum / src.len() as f32 + eps).sqrt().recip()
        }
        F32RmsNormArithmetic::Lanes16ReciprocalSqrt => {
            let sum = sum_squares_f32(src, 16);
            (sum / src.len() as f32 + eps).sqrt().recip()
        }
        F32RmsNormArithmetic::Lanes32ReciprocalSqrt => {
            let sum = sum_squares_f32(src, 32);
            (sum / src.len() as f32 + eps).sqrt().recip()
        }
        F32RmsNormArithmetic::TorchCascade4ReciprocalSqrt => {
            let sum = torch_cascade_sum(src, 4, |value| value * value);
            (sum / src.len() as f32 + eps).sqrt().recip()
        }
        F32RmsNormArithmetic::TorchCascade8ReciprocalSqrt => {
            let sum = torch_cascade_sum(src, 8, |value| value * value);
            (sum / src.len() as f32 + eps).sqrt().recip()
        }
        F32RmsNormArithmetic::F64ReciprocalSqrt => {
            let mut sum = 0.0f64;
            for value in src {
                let value = f64::from(*value);
                sum += value * value;
            }
            (sum / src.len() as f64 + f64::from(eps)).sqrt().recip() as f32
        }
    }
}

fn sum_squares_f32(src: &[f32], lanes: usize) -> f32 {
    let mut partial = [0.0f32; 32];
    for (index, value) in src.iter().enumerate() {
        partial[index % lanes] += *value * *value;
    }
    let mut sum = 0.0f32;
    for value in &partial[..lanes] {
        sum += *value;
    }
    sum
}

/// The reference stack's contiguous-inner-dimension f32 sum, transcribed operation for operation.
///
/// Every other reduction offered here is a *guess* at the oracle's order — "four accumulators
/// landed closer, so perhaps it used four". This one is not a guess: it is the shape PyTorch's
/// `SumKernel.cpp` actually reduces with, and it is materially different from any flat lane
/// interleave, so a flat lane order that lands close can still never land on it.
///
/// Three nested structures, outermost first:
///
/// 1. **Vector lanes.** The row is walked `width` elements at a time and each lane keeps its own
///    running sum. `width` is `Vectorized<float>::size()`, 8 on an ARM build with
///    `AT_BUILD_ARM_VEC256_WITH_SLEEF` and 4 without, which is why it is a parameter here rather
///    than a constant: it is the one part of the shape the provenance does not pin.
/// 2. **Instruction-level parallelism.** `row_sum` splits the vector stream into `ILP = 4`
///    independent chains, reduced `p0 += p1; p0 += p2; p0 += p3` at the end.
/// 3. **Cascade.** Each chain is not a flat running sum but a `LEVELS = 4` deep cascade: level 0
///    absorbs `level_step` vectors and is then drained into level 1, level 1 into level 2 when its
///    own counter wraps, and so on. This is what bounds the reduction's error growth, and it makes
///    the partial-sum magnitudes — and therefore the rounding — differ from a flat sum even at
///    identical lane counts.
///
/// The final horizontal fold is left-to-right over the lanes, as `vectorized_inner_sum` does when
/// it stores the accumulator and sums the array.
///
/// `transform` is applied to each element before accumulation, so RMSNorm's `pow(2).mean(-1)` can
/// square in the same pass the reference's separate `pow(2)` tensor would have.
///
/// # Panics
///
/// Panics if `width` is zero.
// Index loops keep the reference's exact accumulation order (level, chain, lane) visible;
// iterator rewrites would obscure the summation-order argument this port is documenting.
#[allow(clippy::needless_range_loop)]
pub fn torch_cascade_sum(src: &[f32], width: usize, transform: impl Fn(f32) -> f32) -> f32 {
    assert!(width > 0, "vector width must be positive");
    const ILP: usize = 4;
    const LEVELS: usize = 4;

    let vector_count = src.len() / width;
    let vector = |index: usize, lane: usize| transform(src[index * width + lane]);

    // `multi_row_sum(in_data, row_stride = col_stride * ILP, col_stride, size = vector_count / ILP)`
    let size = vector_count / ILP;
    let level_power = ceil_log2(size).div_euclid(LEVELS).max(4);
    let level_step = 1usize << level_power;
    let level_mask = level_step - 1;

    let mut acc = vec![[0.0f32; ILP].map(|_| vec![0.0f32; width]); LEVELS];
    let mut index = 0usize;
    while index + level_step <= size {
        for _ in 0..level_step {
            for chain in 0..ILP {
                for lane in 0..width {
                    acc[0][chain][lane] += vector(index * ILP + chain, lane);
                }
            }
            index += 1;
        }
        for level in 1..LEVELS {
            for chain in 0..ILP {
                for lane in 0..width {
                    acc[level][chain][lane] += acc[level - 1][chain][lane];
                    acc[level - 1][chain][lane] = 0.0;
                }
            }
            if index & (level_mask << (level * level_power)) != 0 {
                break;
            }
        }
    }
    while index < size {
        for chain in 0..ILP {
            for lane in 0..width {
                acc[0][chain][lane] += vector(index * ILP + chain, lane);
            }
        }
        index += 1;
    }
    for level in 1..LEVELS {
        for chain in 0..ILP {
            for lane in 0..width {
                acc[level][chain][lane] += acc[level - 1][chain][lane];
            }
        }
    }

    // `row_sum`: absorb the vectors `multi_row_sum` could not group, then fold the ILP chains.
    let mut partial = acc.swap_remove(LEVELS - 1);
    for leftover in size * ILP..vector_count {
        for lane in 0..width {
            partial[0][lane] += vector(leftover, lane);
        }
    }
    for chain in 1..ILP {
        for lane in 0..width {
            partial[0][lane] += partial[chain][lane];
        }
    }

    // `vectorized_inner_sum`: the elements past the last whole vector are summed first, then the
    // lanes are folded into that running total left to right.
    let mut sum = 0.0f32;
    for index in vector_count * width..src.len() {
        sum += transform(src[index]);
    }
    for lane in 0..width {
        sum += partial[0][lane];
    }
    sum
}

/// `c10::llvm::CeilLog2` for the sizes this reduction sees: `ceil(log2(value))`, zero for `0` and `1`.
fn ceil_log2(value: usize) -> usize {
    if value <= 1 {
        return 0;
    }
    usize::BITS as usize - (value - 1).leading_zeros() as usize
}

/// SwiGLU's elementwise half: `silu(gate) * up`, written into `gate`.
pub fn silu_mul_in_place(gate: &mut [f32], up: &[f32]) {
    silu_mul_in_place_with_arithmetic(gate, up, F32SiluArithmetic::Divide);
}

/// Same operation as [`silu_mul_in_place`], with an explicitly chosen f32 association.
pub fn silu_mul_in_place_with_arithmetic(
    gate: &mut [f32],
    up: &[f32],
    arithmetic: F32SiluArithmetic,
) {
    assert_eq!(gate.len(), up.len(), "gate and up must match");
    for (g, u) in gate.iter_mut().zip(up) {
        let x = *g;
        if arithmetic == F32SiluArithmetic::WidenedF64 {
            let wide = f64::from(x);
            *g = (wide / (1.0 + (-wide).exp()) * f64::from(*u)) as f32;
            continue;
        }
        let denominator = 1.0 + (-x).exp();
        let silu = match arithmetic {
            F32SiluArithmetic::Divide => x / denominator,
            F32SiluArithmetic::MultiplyReciprocal => x * denominator.recip(),
            F32SiluArithmetic::WidenedF64 => unreachable!("handled above"),
        };
        *g = silu * u;
    }
}

/// In-place row-wise softmax in f32, max-subtracted for stability.
pub fn softmax_rows(x: &mut [f32], rows: usize, cols: usize) {
    softmax_rows_with_arithmetic(x, rows, cols, F32SoftmaxArithmetic::ReciprocalMultiply);
}

/// Same operation as [`softmax_rows`], with an explicitly selected normalization form.
pub fn softmax_rows_with_arithmetic(
    x: &mut [f32],
    rows: usize,
    cols: usize,
    arithmetic: F32SoftmaxArithmetic,
) {
    assert_eq!(x.len(), rows * cols, "x must be [rows, cols]");
    for row in 0..rows {
        let slice = &mut x[row * cols..row * cols + cols];
        let mut max = f32::NEG_INFINITY;
        for value in slice.iter() {
            if *value > max {
                max = *value;
            }
        }
        if arithmetic == F32SoftmaxArithmetic::WidenedF64 {
            let max = f64::from(max);
            let mut wide = Vec::with_capacity(slice.len());
            let mut sum = 0.0f64;
            for value in slice.iter() {
                let exponent = (f64::from(*value) - max).exp();
                sum += exponent;
                wide.push(exponent);
            }
            for (value, exponent) in slice.iter_mut().zip(wide) {
                *value = (exponent / sum) as f32;
            }
            continue;
        }
        let mut sum = 0.0f32;
        for value in slice.iter_mut() {
            *value = (*value - max).exp();
            sum += *value;
        }
        for value in slice.iter_mut() {
            *value = match arithmetic {
                F32SoftmaxArithmetic::ReciprocalMultiply => *value * sum.recip(),
                F32SoftmaxArithmetic::Divide => *value / sum,
                F32SoftmaxArithmetic::WidenedF64 => unreachable!("handled above"),
            };
        }
    }
}

/// Grouped-query attention for row-major f32 tensors.
///
/// `queries` and `out` are `[query_positions, q_heads, head_dim]`; `keys` and `values` are
/// `[key_positions, kv_heads, head_dim]`; `additive_mask` is `[query_positions, key_positions]`.
/// Query head `h` reads key/value head `h / (q_heads / kv_heads)`, matching Qwen3-TTS's 16 query
/// heads over 8 KV heads. The reduction order is scalar and fixed so an ISA-specific kernel has a
/// direct f32 reference to compare against.
///
/// # Panics
///
/// Panics if the dimensions disagree or query heads are not evenly grouped over KV heads.
#[allow(clippy::too_many_arguments)]
pub fn gqa_attention(
    queries: &[f32],
    keys: &[f32],
    values: &[f32],
    additive_mask: &[f32],
    query_positions: usize,
    key_positions: usize,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    out: &mut [f32],
) {
    gqa_attention_with_softmax(
        queries,
        keys,
        values,
        additive_mask,
        query_positions,
        key_positions,
        q_heads,
        kv_heads,
        head_dim,
        F32SoftmaxArithmetic::ReciprocalMultiply,
        out,
    );
}

/// Same operation as [`gqa_attention`], with an explicitly selected softmax normalization form.
#[allow(clippy::too_many_arguments)]
pub fn gqa_attention_with_softmax(
    queries: &[f32],
    keys: &[f32],
    values: &[f32],
    additive_mask: &[f32],
    query_positions: usize,
    key_positions: usize,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    softmax_arithmetic: F32SoftmaxArithmetic,
    out: &mut [f32],
) {
    gqa_attention_with_arithmetic(
        queries,
        keys,
        values,
        additive_mask,
        query_positions,
        key_positions,
        q_heads,
        kv_heads,
        head_dim,
        softmax_arithmetic,
        F32LinearAccumulation::Scalar,
        out,
    );
}

/// Same operation as [`gqa_attention`], with selected softmax and dot-product reduction forms.
#[allow(clippy::too_many_arguments)]
pub fn gqa_attention_with_arithmetic(
    queries: &[f32],
    keys: &[f32],
    values: &[f32],
    additive_mask: &[f32],
    query_positions: usize,
    key_positions: usize,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    softmax_arithmetic: F32SoftmaxArithmetic,
    accumulation: F32LinearAccumulation,
    out: &mut [f32],
) {
    assert!(kv_heads > 0, "at least one KV head is required");
    assert_eq!(
        q_heads % kv_heads,
        0,
        "query heads must divide evenly into KV groups"
    );
    assert_eq!(
        queries.len(),
        query_positions * q_heads * head_dim,
        "queries must be [query_positions, q_heads, head_dim]"
    );
    assert_eq!(
        keys.len(),
        key_positions * kv_heads * head_dim,
        "keys must be [key_positions, kv_heads, head_dim]"
    );
    assert_eq!(
        values.len(),
        key_positions * kv_heads * head_dim,
        "values must be [key_positions, kv_heads, head_dim]"
    );
    assert_eq!(
        additive_mask.len(),
        query_positions * key_positions,
        "mask must be [query_positions, key_positions]"
    );
    assert_eq!(
        out.len(),
        query_positions * q_heads * head_dim,
        "out must be [query_positions, q_heads, head_dim]"
    );

    if accumulation == F32LinearAccumulation::Accelerate
        && accelerate_gqa_attention(
            queries,
            keys,
            values,
            additive_mask,
            query_positions,
            key_positions,
            q_heads,
            kv_heads,
            head_dim,
            softmax_arithmetic,
            out,
        )
    {
        return;
    }

    // Team partitioning over query heads: the workers run the SAME extracted per-head loop
    // (`gqa_attention_head_range_with_arithmetic`), heads are independent, and no reduction
    // crosses a head — so the partitioned result is bit-identical to the serial reference
    // (`attention_partitioning_is_bit_exact` in team.rs). Gated to the default arithmetic
    // pair because `AttentionJob` runs exactly that; the forensic softmax and accumulation
    // probe orders must keep the serial path, same rule as the packed-GEMM gate in
    // `linear_with_accumulation`. The floor keeps decode steps over short contexts off the
    // dispatch (mul-add count ≈ heads × qp × kp × dim).
    const TEAM_ATTENTION_FLOOR_MADDS: usize = 512 * 1024;
    if softmax_arithmetic == F32SoftmaxArithmetic::ReciprocalMultiply
        && accumulation == F32LinearAccumulation::Scalar
        && q_heads
            .saturating_mul(query_positions)
            .saturating_mul(key_positions)
            .saturating_mul(head_dim)
            >= TEAM_ATTENTION_FLOOR_MADDS
        && !crate::team::thread_bypassed()
        && let Some(team) = crate::team::armed()
    {
        team.gqa_attention(
            queries,
            keys,
            values,
            additive_mask,
            query_positions,
            key_positions,
            q_heads,
            kv_heads,
            head_dim,
            out,
        );
        return;
    }

    gqa_attention_head_range_with_arithmetic(
        queries,
        keys,
        values,
        additive_mask,
        query_positions,
        key_positions,
        q_heads,
        kv_heads,
        head_dim,
        softmax_arithmetic,
        accumulation,
        0..q_heads,
        out,
    );
}

/// The scalar GQA loop restricted to `q_head_range`, writing only those heads' output spans.
///
/// This is the SAME loop [`gqa_attention_with_arithmetic`] runs — extracted, not duplicated —
/// so a partitioned caller (the worker team) composes the identical arithmetic per head and the
/// full-range serial call remains the reference. Heads are independent: no reduction crosses a
/// head, which is why partitioning here is bit-exact rather than merely close.
///
/// # Panics
///
/// Panics if the range exceeds `q_heads`. Full shape validation is the full-range caller's job;
/// partitioned callers must have validated once before splitting.
#[allow(clippy::too_many_arguments)]
pub fn gqa_attention_head_range_with_arithmetic(
    queries: &[f32],
    keys: &[f32],
    values: &[f32],
    additive_mask: &[f32],
    query_positions: usize,
    key_positions: usize,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    softmax_arithmetic: F32SoftmaxArithmetic,
    accumulation: F32LinearAccumulation,
    q_head_range: std::ops::Range<usize>,
    out: &mut [f32],
) {
    assert!(
        out.len() >= query_positions * q_heads * head_dim,
        "attention output must hold [query_positions, q_heads, head_dim]"
    );
    let out = out.as_mut_ptr();
    // SAFETY: the pointer comes from the `&mut` slice above, whose length was just checked to
    // cover every index the head range can reach, and it is not used after this call.
    unsafe {
        gqa_attention_head_range_into(
            queries,
            keys,
            values,
            additive_mask,
            query_positions,
            key_positions,
            q_heads,
            kv_heads,
            head_dim,
            softmax_arithmetic,
            accumulation,
            q_head_range,
            out,
        );
    }
}

/// The head-range attention loop, writing through a raw output pointer.
///
/// This exists so parallel workers never have to materialize a `&mut [f32]` over the whole output
/// while a sibling worker holds one too. Disjoint *writes* are not enough for that to be sound:
/// two live `&mut` into the same allocation is undefined behaviour whatever the access pattern,
/// and `rustc` marks `&mut` parameters `noalias`, so it is the optimizer — not just the model —
/// that the overlap would mislead. Here each worker turns the pointer into a `&mut` covering
/// exactly the one head span it is about to write, and those spans are disjoint by construction.
///
/// # Safety
///
/// `out` must be valid for writes across `[query_positions, q_heads, head_dim]`, and no other
/// reference may alias the `head_dim` spans this call's `q_head_range` writes for its duration.
// SAFETY: discharged by both callers — the safe wrapper asserts `out.len()` and holds the only
// `&mut`; the team gives each worker a disjoint `q_head_range` whose `head_dim` spans cannot
// overlap, and blocks until every partition reports done, so `out` outlives all writes.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn gqa_attention_head_range_into(
    queries: &[f32],
    keys: &[f32],
    values: &[f32],
    additive_mask: &[f32],
    query_positions: usize,
    key_positions: usize,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    softmax_arithmetic: F32SoftmaxArithmetic,
    accumulation: F32LinearAccumulation,
    q_head_range: std::ops::Range<usize>,
    out: *mut f32,
) {
    assert!(q_head_range.end <= q_heads, "head range exceeds q_heads");
    let scale = (head_dim as f32).sqrt().recip();
    let kv_group = q_heads / kv_heads;
    // Per-thread scratch, not a per-call `vec!`: this runs per attention dispatch on the
    // steady-state decode path (doctrine: no allocator activity there). Grows monotonically
    // to the longest context this thread has scored.
    thread_local! {
        static SCORES_SCRATCH: std::cell::RefCell<Vec<f32>> =
            const { std::cell::RefCell::new(Vec::new()) };
    }
    SCORES_SCRATCH.with(|scratch| {
        let mut scores_guard = scratch.borrow_mut();
        if scores_guard.len() < key_positions {
            scores_guard.resize(key_positions, 0.0);
        }
        let scores = &mut scores_guard[..key_positions];

        for query_position in 0..query_positions {
            let mask = &additive_mask
                [query_position * key_positions..(query_position + 1) * key_positions];
            for q_head in q_head_range.clone() {
                let kv_head = q_head / kv_group;
                let query_base = (query_position * q_heads + q_head) * head_dim;
                let query = &queries[query_base..query_base + head_dim];
                for (key_position, score) in scores.iter_mut().enumerate() {
                    let key_base = (key_position * kv_heads + kv_head) * head_dim;
                    let key = &keys[key_base..key_base + head_dim];
                    let dot = dot_with_accumulation(query, key, accumulation);
                    *score = dot * scale + mask[key_position];
                }
                softmax_rows_with_arithmetic(scores, 1, key_positions, softmax_arithmetic);

                // SAFETY: `query_base` indexes [query_position, q_head, head_dim] inside the bounds
                // the caller guaranteed, and this borrow spans only this head — the one span this
                // partition owns, disjoint from every other partition's.
                let head_out =
                    unsafe { std::slice::from_raw_parts_mut(out.add(query_base), head_dim) };
                attention_weighted_sum(
                    scores,
                    values,
                    kv_head,
                    kv_heads,
                    head_dim,
                    accumulation,
                    head_out,
                );
            }
        }
    });
}

/// Executes the two attention matrix products through the exact macOS SGEMM candidate.
///
/// This is intentionally an L2-parity probe rather than the normal attention route. The gather
/// buffers present the strided GQA heads as the row-major matrices consumed by CBLAS; the scalar
/// path above remains the cross-platform reference and the fallback when this candidate is off.
#[cfg(all(feature = "accelerate-sgemm", target_os = "macos"))]
#[allow(clippy::too_many_arguments)]
fn accelerate_gqa_attention(
    queries: &[f32],
    keys: &[f32],
    values: &[f32],
    additive_mask: &[f32],
    query_positions: usize,
    key_positions: usize,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    softmax_arithmetic: F32SoftmaxArithmetic,
    out: &mut [f32],
) -> bool {
    let scale = (head_dim as f32).sqrt().recip();
    let kv_group = q_heads / kv_heads;
    let mut query_matrix = vec![0.0f32; query_positions * head_dim];
    let mut key_matrix = vec![0.0f32; key_positions * head_dim];
    let mut value_transpose = vec![0.0f32; head_dim * key_positions];
    let mut scores = vec![0.0f32; query_positions * key_positions];
    let mut context = vec![0.0f32; query_positions * head_dim];

    for q_head in 0..q_heads {
        let kv_head = q_head / kv_group;
        for query_position in 0..query_positions {
            let query_base = (query_position * q_heads + q_head) * head_dim;
            query_matrix[query_position * head_dim..(query_position + 1) * head_dim]
                .copy_from_slice(&queries[query_base..query_base + head_dim]);
        }
        for key_position in 0..key_positions {
            let key_base = (key_position * kv_heads + kv_head) * head_dim;
            key_matrix[key_position * head_dim..(key_position + 1) * head_dim]
                .copy_from_slice(&keys[key_base..key_base + head_dim]);
            for lane in 0..head_dim {
                value_transpose[lane * key_positions + key_position] = values[key_base + lane];
            }
        }

        if !accelerate_sgemm(
            &query_matrix,
            &key_matrix,
            query_positions,
            head_dim,
            key_positions,
            0.0,
            false,
            &mut scores,
        ) {
            return false;
        }
        for query_position in 0..query_positions {
            let score_row =
                &mut scores[query_position * key_positions..(query_position + 1) * key_positions];
            let mask = &additive_mask
                [query_position * key_positions..(query_position + 1) * key_positions];
            for (score, mask_value) in score_row.iter_mut().zip(mask) {
                *score = *score * scale + mask_value;
            }
            softmax_rows_with_arithmetic(score_row, 1, key_positions, softmax_arithmetic);
        }
        if !accelerate_sgemm(
            &scores,
            &value_transpose,
            query_positions,
            key_positions,
            head_dim,
            0.0,
            false,
            &mut context,
        ) {
            return false;
        }
        for query_position in 0..query_positions {
            let out_base = (query_position * q_heads + q_head) * head_dim;
            out[out_base..out_base + head_dim].copy_from_slice(
                &context[query_position * head_dim..(query_position + 1) * head_dim],
            );
        }
    }
    true
}

#[cfg(not(all(feature = "accelerate-sgemm", target_os = "macos")))]
#[allow(clippy::too_many_arguments)]
fn accelerate_gqa_attention(
    _queries: &[f32],
    _keys: &[f32],
    _values: &[f32],
    _additive_mask: &[f32],
    _query_positions: usize,
    _key_positions: usize,
    _q_heads: usize,
    _kv_heads: usize,
    _head_dim: usize,
    _softmax_arithmetic: F32SoftmaxArithmetic,
    _out: &mut [f32],
) -> bool {
    false
}

#[allow(clippy::too_many_arguments)]
fn attention_weighted_sum(
    scores: &[f32],
    values: &[f32],
    kv_head: usize,
    kv_heads: usize,
    head_dim: usize,
    accumulation: F32LinearAccumulation,
    out: &mut [f32],
) {
    if accumulation == F32LinearAccumulation::WidenedF64 {
        for lane in 0..head_dim {
            let mut sum = 0.0f64;
            for (key_position, weight) in scores.iter().copied().enumerate() {
                let value = values[(key_position * kv_heads + kv_head) * head_dim + lane];
                sum += f64::from(weight) * f64::from(value);
            }
            out[lane] = sum as f32;
        }
        return;
    }
    let lanes = match accumulation {
        F32LinearAccumulation::Scalar => 1,
        F32LinearAccumulation::Lanes4 | F32LinearAccumulation::FusedLanes4 => 4,
        F32LinearAccumulation::Lanes8 | F32LinearAccumulation::FusedLanes8 => 8,
        F32LinearAccumulation::Accelerate
        | F32LinearAccumulation::AccelerateRowInvariant
        | F32LinearAccumulation::AccelerateBiasSeeded
        | F32LinearAccumulation::AccelerateBiasSeededRowInvariant => 1,
        F32LinearAccumulation::WidenedF64 => unreachable!("handled above"),
    };
    for lane in 0..head_dim {
        let mut partial = [0.0f32; 8];
        for (key_position, weight) in scores.iter().copied().enumerate() {
            let value = values[(key_position * kv_heads + kv_head) * head_dim + lane];
            let partial_index = key_position % lanes;
            partial[partial_index] = match accumulation {
                F32LinearAccumulation::FusedLanes4 | F32LinearAccumulation::FusedLanes8 => {
                    weight.mul_add(value, partial[partial_index])
                }
                F32LinearAccumulation::Scalar
                | F32LinearAccumulation::Lanes4
                | F32LinearAccumulation::Lanes8
                | F32LinearAccumulation::Accelerate
                | F32LinearAccumulation::AccelerateRowInvariant
                | F32LinearAccumulation::AccelerateBiasSeeded
                | F32LinearAccumulation::AccelerateBiasSeededRowInvariant
                | F32LinearAccumulation::WidenedF64 => partial[partial_index] + weight * value,
            };
        }
        let mut sum = 0.0f32;
        for value in &partial[..lanes] {
            sum += *value;
        }
        out[lane] = sum;
    }
}

/// Collapse the three mRoPE axes into one `cos`/`sin` row using the checkpoint's INTERLEAVED rule.
///
/// The pinned config sets `rope_scaling.interleaved = true`, which selects a different branch from
/// the familiar section-split one — a difference that is numerically invisible whenever the three
/// axes carry equal positions (which OQ-4 says they always do here, all three receiving the same
/// scalar causal index) and therefore exactly the kind of thing a port gets wrong and only discovers
/// against a batched or genuinely multimodal input. It is implemented faithfully regardless.
///
/// `axes` is the first half of each axis's row, `[3][half]`; `out` receives `[half]`. Element `j`
/// takes axis `j % 3` while `j` lies in `1..sections[1..].max() * 3`, and axis 0 elsewhere.
///
/// # Panics
///
/// Panics if `out` is not `half` long or an axis row is short.
pub fn mrope_interleave(axes: [&[f32]; 3], sections: [usize; 3], out: &mut [f32]) {
    let half = out.len();
    for axis in axes {
        assert!(
            axis.len() >= half,
            "axis row shorter than the half-dimension"
        );
    }

    // Start from axis 0 everywhere, then overwrite the strided lanes from axes 1 and 2, exactly as
    // the reference does with its `x_t[..., beg:end:3] = x[beg, ..., beg:end:3]` assignments.
    out.copy_from_slice(&axes[0][..half]);
    let modality_num = 3usize;
    for (axis_index, section) in sections.iter().enumerate().skip(1) {
        let end = section * modality_num;
        let mut lane = axis_index;
        while lane < end && lane < half {
            out[lane] = axes[axis_index][lane];
            lane += modality_num;
        }
    }
}

/// Apply rotary embeddings to one head row in the `rotate_half` layout.
///
/// `row` is `[head_dim]`; `cos` and `sin` are the full `[head_dim]` rows (the doubled half). The
/// transform is `x*cos + rotate_half(x)*sin` where `rotate_half` maps `[a, b] -> [-b, a]` over the
/// two halves.
///
/// # Panics
///
/// Panics if `cos`/`sin` do not match `row`, or if `head_dim` is odd.
pub fn apply_rope_in_place(row: &mut [f32], cos: &[f32], sin: &[f32]) {
    let dim = row.len();
    assert_eq!(cos.len(), dim, "cos must match head_dim");
    assert_eq!(sin.len(), dim, "sin must match head_dim");
    assert!(dim.is_multiple_of(2), "head_dim must be even");

    let half = dim / 2;
    let original: Vec<f32> = row.to_vec();
    for index in 0..dim {
        let rotated = if index < half {
            -original[index + half]
        } else {
            original[index - half]
        };
        row[index] = original[index] * cos[index] + rotated * sin[index];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_matches_a_hand_computed_product() {
        // x = [[1, 2, 3]], weight = [[1, 0, -1], [2, 2, 2]] -> [1*1 + 2*0 + 3*-1, 2+4+6] = [-2, 12]
        let x = [1.0, 2.0, 3.0];
        let weight = [1.0, 0.0, -1.0, 2.0, 2.0, 2.0];
        let mut out = [0.0; 2];
        linear(&x, &weight, None, 1, 3, 2, &mut out);
        assert_eq!(out, [-2.0, 12.0]);

        let mut biased = [0.0; 2];
        linear(&x, &weight, Some(&[10.0, -12.0]), 1, 3, 2, &mut biased);
        assert_eq!(biased, [8.0, 0.0]);
    }

    #[test]
    fn torch_cascade_sum_agrees_with_the_flat_sum_when_rounding_cannot_intervene() {
        // Powers of two below 2^24 add without rounding, so every reduction order must produce the
        // same total. This proves the cascade's bookkeeping — its drains, its ILP chains, its tail
        // handling — visits each element exactly once, independently of any parity claim.
        for length in [1usize, 7, 8, 15, 16, 31, 128, 1024, 3072] {
            for width in [4usize, 8] {
                let values: Vec<f32> = (0..length).map(|index| (index % 8) as f32).collect();
                let flat: f32 = values.iter().sum();
                assert_eq!(
                    torch_cascade_sum(&values, width, |value| value),
                    flat,
                    "length {length}, width {width}"
                );
            }
        }
    }

    #[test]
    fn torch_cascade_sum_applies_its_transform_before_accumulating() {
        let values = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(torch_cascade_sum(&values, 4, |value| value * value), 55.0);
    }

    #[test]
    fn torch_cascade_sum_differs_from_a_flat_sum_once_rounding_matters() {
        // A large leading term followed by many small ones is exactly the case a cascade exists to
        // improve: the flat sum loses every small term into the large accumulator, the cascade does
        // not. If these ever agreed, the transcription would have collapsed into a flat sum and the
        // parity sweep would be comparing one order against itself.
        let mut values = vec![1.0f32; 1024];
        values[0] = 1.0e8;
        let flat = values.iter().fold(0.0f32, |sum, value| sum + value);
        assert_ne!(torch_cascade_sum(&values, 8, |value| value), flat);
    }

    #[test]
    fn ceil_log2_matches_its_definition() {
        assert_eq!(ceil_log2(0), 0);
        assert_eq!(ceil_log2(1), 0);
        assert_eq!(ceil_log2(2), 1);
        assert_eq!(ceil_log2(3), 2);
        assert_eq!(ceil_log2(32), 5);
        assert_eq!(ceil_log2(33), 6);
    }

    #[test]
    fn rms_norm_normalizes_and_scales() {
        // mean(x^2) for [3, 4] is 12.5; rsqrt(12.5 + 0) ~ 0.2828427
        let x = [3.0f32, 4.0];
        let weight = [1.0f32, 1.0];
        let mut out = [0.0; 2];
        rms_norm(&x, &weight, 0.0, 1, 2, &mut out);
        let expected = 12.5f32.sqrt().recip();
        assert!((out[0] - 3.0 * expected).abs() < 1e-6);
        assert!((out[1] - 4.0 * expected).abs() < 1e-6);

        // The weight is applied per element, after scaling.
        let mut weighted = [0.0; 2];
        rms_norm(&x, &[2.0, 0.5], 0.0, 1, 2, &mut weighted);
        assert!((weighted[0] - 3.0 * expected * 2.0).abs() < 1e-6);
        assert!((weighted[1] - 4.0 * expected * 0.5).abs() < 1e-6);
    }

    #[test]
    fn silu_mul_matches_the_definition() {
        let mut gate = [0.0f32, 1.0, -1.0];
        let up = [1.0f32, 2.0, 3.0];
        silu_mul_in_place(&mut gate, &up);
        assert_eq!(gate[0], 0.0);
        let silu_one = 1.0f32 / (1.0 + (-1.0f32).exp());
        assert!((gate[1] - silu_one * 2.0).abs() < 1e-6);
        let silu_neg = -1.0f32 / (1.0 + 1.0f32.exp());
        assert!((gate[2] - silu_neg * 3.0).abs() < 1e-6);
    }

    #[test]
    fn softmax_rows_sums_to_one_and_is_shift_invariant() {
        let mut x = [1.0f32, 2.0, 3.0, 101.0, 102.0, 103.0];
        softmax_rows(&mut x, 2, 3);
        let first: f32 = x[..3].iter().sum();
        let second: f32 = x[3..].iter().sum();
        assert!((first - 1.0).abs() < 1e-6);
        assert!((second - 1.0).abs() < 1e-6);
        // Rows differing by a constant shift must produce identical distributions.
        for index in 0..3 {
            assert!((x[index] - x[index + 3]).abs() < 1e-6);
        }
    }

    #[test]
    fn gqa_maps_each_query_head_to_its_kv_group() {
        let (query_positions, key_positions, q_heads, kv_heads, head_dim) = (1, 1, 4, 2, 2);
        let queries = vec![0.0f32; query_positions * q_heads * head_dim];
        let keys = vec![0.0f32; key_positions * kv_heads * head_dim];
        let values = [10.0f32, 11.0, 20.0, 21.0];
        let mut out = vec![0.0f32; query_positions * q_heads * head_dim];

        gqa_attention(
            &queries,
            &keys,
            &values,
            &[0.0],
            query_positions,
            key_positions,
            q_heads,
            kv_heads,
            head_dim,
            &mut out,
        );

        assert_eq!(&out[0..2], &[10.0, 11.0]);
        assert_eq!(&out[2..4], &[10.0, 11.0]);
        assert_eq!(&out[4..6], &[20.0, 21.0]);
        assert_eq!(&out[6..8], &[20.0, 21.0]);
    }

    #[test]
    fn gqa_honors_the_additive_causal_mask() {
        let (query_positions, key_positions, q_heads, kv_heads, head_dim) = (2, 2, 1, 1, 2);
        let queries = vec![0.0f32; query_positions * q_heads * head_dim];
        let keys = vec![0.0f32; key_positions * kv_heads * head_dim];
        let values = [2.0f32, 4.0, 10.0, 20.0];
        let mask = [0.0f32, f32::NEG_INFINITY, 0.0, 0.0];
        let mut out = vec![0.0f32; query_positions * q_heads * head_dim];

        gqa_attention(
            &queries,
            &keys,
            &values,
            &mask,
            query_positions,
            key_positions,
            q_heads,
            kv_heads,
            head_dim,
            &mut out,
        );

        assert_eq!(&out[0..2], &[2.0, 4.0]);
        assert_eq!(&out[2..4], &[6.0, 12.0]);
    }

    #[test]
    fn rope_rotates_a_known_pair() {
        // head_dim 2, cos = [0, 0], sin = [1, 1]: [a, b] -> [-b, a]
        let mut row = [3.0f32, 5.0];
        apply_rope_in_place(&mut row, &[0.0, 0.0], &[1.0, 1.0]);
        assert_eq!(row, [-5.0, 3.0]);

        // Identity when cos = 1, sin = 0.
        let mut same = [3.0f32, 5.0];
        apply_rope_in_place(&mut same, &[1.0, 1.0], &[0.0, 0.0]);
        assert_eq!(same, [3.0, 5.0]);
    }

    #[test]
    fn mrope_interleave_is_identity_when_all_axes_agree() {
        // OQ-4: all three axes carry the same scalar causal index in this model, so the interleave
        // must be a no-op on equal axes. If it is not, the lane arithmetic is wrong.
        let axis: Vec<f32> = (0..64).map(|value| value as f32).collect();
        let mut out = vec![0.0f32; 64];
        mrope_interleave([&axis, &axis, &axis], [24, 20, 20], &mut out);
        assert_eq!(out, axis);
    }

    #[test]
    fn mrope_interleave_selects_the_documented_lanes() {
        let zeros = vec![0.0f32; 64];
        let ones = vec![1.0f32; 64];
        let twos = vec![2.0f32; 64];
        let mut out = vec![0.0f32; 64];
        mrope_interleave([&zeros, &ones, &twos], [24, 20, 20], &mut out);

        // Lanes 1, 4, .. < 60 come from axis 1; lanes 2, 5, .. < 60 from axis 2; the rest stay 0,
        // including every lane at or above 60.
        for (lane, value) in out.iter().enumerate() {
            let expected = if lane < 60 && lane % 3 == 1 {
                1.0
            } else if lane < 60 && lane % 3 == 2 {
                2.0
            } else {
                0.0
            };
            assert_eq!(*value, expected, "lane {lane}");
        }
    }
    #[test]
    fn single_row_codec_gemv_matches_the_bypassed_scalar_bits_through_the_team() {
        // The interactive profile's per-frame codec call: m = 1 at the binding worst-case
        // im2col reduction (block_00, kernel 7 x 1024 -> 1536, so k = 7168). With the native
        // team self-armed this must dispatch across workers; bypassed, it must fall through to
        // the scalar dot. Both routes write the same bits — that is the whole claim of the new
        // arm in `linear_with_accumulation`.
        let (m, k, n) = (1usize, 7168usize, 1536usize);
        assert!(
            n * k >= 64 * 1024,
            "the shape must clear the GEMV dispatch floor or the test pins nothing"
        );
        let mut state = 0x51ED_5EED_u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f32 / 8_388_608.0 * 2.0 - 1.0
        };
        let x: Vec<f32> = (0..m * k).map(|_| next()).collect();
        let weight: Vec<f32> = (0..n * k).map(|_| next()).collect();
        let bias: Vec<f32> = (0..n).map(|_| next()).collect();

        let mut via_team = vec![0.0f32; m * n];
        linear_with_accumulation(
            &x,
            &weight,
            Some(&bias),
            m,
            k,
            n,
            F32LinearAccumulation::AccelerateBiasSeededRowInvariant,
            &mut via_team,
        );

        let mut bypassed = vec![0.0f32; m * n];
        crate::team::with_team_bypassed(|| {
            linear_with_accumulation(
                &x,
                &weight,
                Some(&bias),
                m,
                k,
                n,
                F32LinearAccumulation::AccelerateBiasSeededRowInvariant,
                &mut bypassed,
            );
        });

        assert_eq!(
            via_team.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            bypassed.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "the team-routed GEMV diverged from the bypassed scalar dot"
        );
    }
}
