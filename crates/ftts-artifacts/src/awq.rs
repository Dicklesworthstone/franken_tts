//! Activation-aware weight quantization for the microdecoder tables (bead
//! `frankentts-ukk6`).
//!
//! Two ideas stacked on top of round-to-nearest, both driven by CALIBRATION
//! activations captured teacher-forced from the frozen conformance corpus:
//!
//! 1. **AWQ-style per-channel scales** ([`awq_scales`]): a handful of salient
//!    input channels dominate the quantization error of every output row they
//!    touch. Scaling those channels UP before quantization (and folding the
//!    reciprocal into the activation side, which is exact at runtime because
//!    the fold lands in an already-fused preceding op) preserves them at the
//!    expense of channels nobody activates. The scale strength is a single
//!    exponent picked by grid search against the calibration set.
//!
//! 2. **GPTQ error-compensating rounding** ([`gptq_round_matrix`]): after
//!    choosing which way a column rounds, propagate its rounding error into
//!    the not-yet-rounded columns through the inverse-Hessian Cholesky factor,
//!    so later columns round to COMPENSATE earlier mistakes instead of adding
//!    them. Same bit budget, strictly less expected output error.
//!
//! Everything here is offline/converter-side: the runtime sees ordinary packed
//! matrices plus (eventually) the folded scales. Nothing in this module runs
//! during synthesis.
//!
//! # Numerical contract
//!
//! - Weights are row-major `[n, k]`: one quantizable row per output element.
//! - Calibration activations are `[m, k]`: m vectors collected at the op's
//!   input across the corpus; the Hessian is their second moment `XᵀX`.
//! - All output rows share one Hessian (the input distribution is per-op, not
//!   per-row) and one scale vector (scales act on INPUT channels).

/// Accumulates the per-channel second moment of calibration activations —
/// the diagonal of the GPTQ Hessian and the AWQ saliency signal.
#[derive(Clone, Debug)]
pub struct CalibrationStats {
    /// Sum over calibration vectors of x_j², per channel.
    pub second_moment: Vec<f64>,
    samples: usize,
}

impl CalibrationStats {
    /// Empty stats for `channels` input channels.
    #[must_use]
    pub fn new(channels: usize) -> Self {
        Self {
            second_moment: vec![0.0_f64; channels],
            samples: 0,
        }
    }

    /// Absorbs one calibration vector of length `channels`.
    ///
    /// # Panics
    ///
    /// Panics when the vector length disagrees with the channel count.
    pub fn observe(&mut self, x: &[f32]) {
        assert_eq!(x.len(), self.second_moment.len(), "channel count mismatch");
        for (moment, &value) in self.second_moment.iter_mut().zip(x) {
            *moment += f64::from(value) * f64::from(value);
        }
        self.samples += 1;
    }

    /// How many calibration vectors have been absorbed.
    #[must_use]
    pub fn samples(&self) -> usize {
        self.samples
    }

    /// Mean second moment per channel (= E[x²]); the AWQ saliency signal.
    ///
    /// Channels with zero observed energy get the smallest positive mean value
    /// so a dead channel can never win the saliency ranking.
    #[must_use]
    pub fn mean_second_moment(&self) -> Vec<f64> {
        let denominator = self.samples.max(1) as f64;
        let mut means: Vec<f64> = self.second_moment.iter().map(|m| m / denominator).collect();
        let floor = means.iter().cloned().fold(f64::INFINITY, f64::min);
        let floor = if floor.is_finite() && floor > 0.0 { floor } else { 1.0 };
        for mean in &mut means {
            if *mean <= 0.0 {
                *mean = floor;
            }
        }
        means
    }
}

/// AWQ-style per-input-channel scales: `s_j ∝ E[x_j²]^α`, normalized to keep
/// the geometric mean at 1 (so the folded activation rescaling neither grows
/// nor shrinks overall magnitudes).
///
/// `alpha` is the protection exponent — 0 disables scaling entirely, larger
/// values concentrate more range on salient channels. Callers grid-search it
/// ([`awq_best_alpha`]).
#[must_use]
pub fn awq_scales(saliency: &[f64], alpha: f64) -> Vec<f32> {
    let mut scales: Vec<f64> = saliency
        .iter()
        .map(|&moment| {
            // Saliency values are non-negative by construction; guard anyway so a
            // stray NaN or negative cannot poison the geometric mean below.
            let value = if moment.is_finite() && *moment > 0.0 { moment } else { 1.0 };
            value.powf(alpha)
        })
        .collect();
    let count = scales.len() as f64;
    let log_mean = scales.iter().map(|s| s.ln()).sum::<f64>() / count;
    for scale in &mut scales {
        *scale = (*scale / log_mean.exp()).max(1e-6);
    }
    scales.into_iter().map(|s| s as f32).collect()
}

/// Squared output error of per-row RTN quantization at `bits` after applying
/// input-side `scales` (activation multiplied by s, weight divided by s).
#[must_use]
pub fn rescaled_quantization_error(
    weight_row_major: &[f32],
    n: usize,
    k: usize,
    calib_x: &[Vec<f32>],
    scales: &[f32],
    bits: u32,
) -> f64 {
    let levels = (1_u64 << (bits - 1)) as f32; // symmetric range [-levels, levels]
    let mut total = 0.0_f64;
    let mut quantized = vec![0.0_f32; n * k];
    for row in 0..n {
        let src = &weight_row_major[row * k..(row + 1) * k];
        let max_abs = src
            .iter()
            .enumerate()
            .map(|(j, &w)| w.abs() / scales[j])
            .fold(0.0_f32, f32::max);
        if max_abs == 0.0 {
            continue;
        }
        let scale = max_abs / levels;
        for (j, &w) in src.iter().enumerate() {
            let q = (w / scales[j] / scale).round().clamp(-levels, levels);
            quantized[row * k + j] = q * scale * scales[j];
        }
    }
    for x in calib_x {
        for row in 0..n {
            let w_src = &weight_row_major[row * k..(row + 1) * k];
            let w_q = &quantized[row * k..(row + 1) * k];
            let mut exact = 0.0_f64;
            let mut approx = 0.0_f64;
            for j in 0..k {
                let scaled = f64::from(x[j]) * f64::from(scales[j]);
                exact += scaled * f64::from(w_src[j]);
                approx += scaled * f64::from(w_q[j]);
            }
            total += (exact - approx) * (exact - approx);
        }
    }
    total
}

/// Grid-searches the AWQ exponent against calibration data and returns
/// `(best_alpha, best_scales)`.
///
/// Error model: for each candidate α, quantize the rescaled weights
/// `W·diag(1/s)` per-row RTN at `bits`, dequantize, and measure the squared
/// output error over the calibration rows. Pure and deterministic.
///
/// # Panics
///
/// Panics when shapes disagree (weights `[n,k]`, activations `[m,k]`).
pub fn awq_best_alpha(
    weight_row_major: &[f32],
    n: usize,
    k: usize,
    calib_x: &[Vec<f32>],
    bits: u32,
    grid_step: f64,
) -> (f64, Vec<f32>) {
    assert_eq!(weight_row_major.len(), n * k, "weight shape mismatch");
    assert!(!calib_x.is_empty(), "calibration set is empty");
    let mut stats = CalibrationStats::new(k);
    for x in calib_x {
        stats.observe(x);
    }
    let saliency = stats.mean_second_moment();

    let mut best_alpha = 0.0_f64;
    let mut best_scales = awq_scales(&saliency, 0.0);
    let mut best_error = f64::INFINITY;
    let steps = (1.0 / grid_step).round() as usize;
    for step in 0..=steps {
        let alpha = step as f64 * grid_step;
        let scales = awq_scales(&saliency, alpha);
        let error = rescaled_quantization_error(weight_row_major, n, k, calib_x, &scales, bits);
        if error < best_error {
            best_error = error;
            best_alpha = alpha;
            best_scales = scales;
        }
    }
    (best_alpha, best_scales)
}

/// Computes the inverse of the `k×k` second-moment matrix `XᵀX` via damped
/// Cholesky solve — the matrix GPTQ's column sweep propagates errors through.
///
/// The Hessian is damped by `damping × trace(H)/k` on the diagonal before
/// inversion (standard regularization keeping the solve finite when some
/// channels carry almost no calibration energy).
///
/// Returns `None` when damping still leaves the matrix singular (a degenerate
/// calibration set).
#[must_use]
pub fn gptq_inverse_hessian(hessian: &[f64], k: usize, damping: f64) -> Option<Vec<f64>> {
    let trace: f64 = (0..k).map(|i| hessian[i * k + i]).sum::<f64>();
    let damp = damping * trace / k as f64;
    let mut a = hessian.to_vec();
    for i in 0..k {
        a[i * k + i] += damp;
    }

    // Lower Cholesky L with A = L·Lᵀ.
    let mut l = vec![0.0_f64; k * k];
    for i in 0..k {
        for j in 0..=i {
            let mut sum = a[i * k + j];
            for p in 0..j {
                sum -= l[i * k + p] * l[j * k + p];
            }
            if i == j {
                if sum <= 0.0 || !sum.is_finite() {
                    return None;
                }
                l[i * k + i] = sum.sqrt();
            } else {
                l[i * k + j] = sum / l[j * k + j];
            }
        }
    }

    // Solve A·Z = I column-wise: L·y = e_col, then Lᵀ·z = y.
    let mut z = vec![0.0_f64; k * k];
    let mut y = vec![0.0_f64; k];
    for col in 0..k {
        for i in 0..k {
            let mut sum = if i == col { 1.0 } else { 0.0 };
            for p in 0..i {
                sum -= l[i * k + p] * y[p];
            }
            y[i] = sum / l[i * k + i];
        }
        for i in (0..k).rev() {
            let mut sum = y[i];
            for p in (i + 1)..k {
                sum -= l[p * k + i] * z[p * k + col];
            }
            z[i * k + col] = sum / l[i * k + i];
        }
    }
    Some(z)
}

/// GPTQ error-compensating rounding of a row-major `[n, k]` weight against the
/// inverted calibration Hessian ([`gptq_inverse_hessian`]).
///
/// Columns are swept RIGHT to LEFT. Rounding column j's error is propagated
/// through the inverse-Hessian coefficients into the still-unrounded columns,
/// so subsequent columns compensate rather than accumulate. Every row shares
/// one sweep (one input distribution per op); each row's quantization scale is
/// fixed once from its incoming magnitudes so the sweep never rewrites
/// already-rounded columns.
///
/// Returns the rounded-and-dequantized weights in `[n, k]` layout.
///
/// # Panics
///
/// Panics if `inverse_hessian` is not `k×k` or shapes disagree.
#[must_use]
pub fn gptq_round_matrix(
    weight_row_major: &[f32],
    n: usize,
    k: usize,
    inverse_hessian: &[f64],
    bits: u32,
) -> Vec<f32> {
    assert_eq!(inverse_hessian.len(), k * k, "inverse hessian shape");
    let levels = (1_u64 << (bits - 1)) as f64;
    let mut w: Vec<f64> = weight_row_major.iter().map(f64::from).collect();

    // Per-row scales fixed ONCE from incoming magnitudes: re-deriving them
    // mid-sweep would silently rewrite already-rounded columns.
    let row_scales: Vec<f64> = (0..n)
        .map(|row| {
            let max_abs = (0..k)
                .map(|j| w[row * k + j].abs())
                .fold(0.0_f64, f64::max);
            if max_abs == 0.0 { 1.0 } else { max_abs / levels }
        })
        .collect();

    for col in (0..k).rev() {
        let pivot = inverse_hessian[col * k + col];
        for row in 0..n {
            let index = row * k + col;
            let scale = row_scales[row];
            let q = (w[index] / scale).round().clamp(-levels, levels);
            let err = (w[index] - q * scale) / pivot;
            w[index] = q * scale;
            for j in 0..col {
                w[row * k + j] -= err * inverse_hessian[col * k + j];
            }
        }
    }

    w.into_iter().map(|value| value as f32).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seeded(seed: u64, len: usize) -> Vec<f32> {
        let mut state = seed | 1;
        (0..len)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (state % 20_000) as f32 / 10_000.0 - 1.0
            })
            .collect()
    }

    #[test]
    fn awq_scales_concentrate_on_salient_channels() {
        let mut saliency = vec![1.0_f64; 8];
        saliency[3] = 400.0; // one hot channel
        let scales = awq_scales(&saliency, 0.6);
        assert!(scales[3] > scales[0], "salient channel must scale up");
        assert!(
            scales[3] > 4.0,
            "strong saliency should move the scale substantially, got {}",
            scales[3]
        );
        let log_mean = scales.iter().map(|s| f64::from(*s).ln()).sum::<f64>() / 8.0;
        assert!(log_mean.abs() < 0.5, "geometric mean stays pinned near 1");
    }

    #[test]
    fn awq_grid_reduces_error_against_naive_rtn_on_skewed_calibration() {
        let (n, k) = (16_usize, 32_usize);
        let mut weight = seeded(11, n * k);
        for row in 0..n {
            weight[row * k + 5] *= 20.0; // weights echo the skewed channel
        }
        let calib: Vec<Vec<f32>> = (0..12)
            .map(|i| {
                let mut x = seeded(100 + i, k);
                x[5] *= 20.0;
                x
            })
            .collect();

        let (best_alpha, scales) = awq_best_alpha(&weight, n, k, &calib, 4, 0.05);
        let error_best = rescaled_quantization_error(&weight, n, k, &calib, &scales, 4);
        let error_rtn = rescaled_quantization_error(
            &weight,
            n,
            k,
            &calib,
            &vec![1.0; k],
            4,
        );
        assert!(
            error_best < error_rtn,
            "grid result ({best_alpha}) must beat naive RTN: {error_best} !< {error_rtn}"
        );
    }

    #[test]
    fn gptq_never_increases_error_vs_rtn_on_random_matrices() {
        let (n, k) = (8_usize, 16_usize);
        let weight = seeded(42, n * k);
        let calib: Vec<Vec<f32>> = (0..24).map(|i| seeded(500 + i, k)).collect();
        let mut stats = CalibrationStats::new(k);
        for x in &calib {
            stats.observe(x);
        }
        let mut hessian = vec![0.0_f64; k * k];
        for x in &calib {
            for i in 0..k {
                for j in 0..k {
                    hessian[i * k + j] += f64::from(x[i]) * f64::from(x[j]);
                }
            }
        }
        let inverse = gptq_inverse_hessian(&hessian, k, 0.01).expect("well-conditioned");
        let gptq = gptq_round_matrix(&weight, n, k, &inverse, 4);

        let levels = 8.0_f32;
        let mut rtn = vec![0.0_f32; n * k];
        for row in 0..n {
            let src = &weight[row * k..(row + 1) * k];
            let max_abs = src.iter().fold(0.0_f32, f32::max);
            let scale = max_abs / levels;
            for (j, &w) in src.iter().enumerate() {
                rtn[row * k + j] = (w / scale).round().clamp(-levels, levels) * scale;
            }
        }
        let error_of = |q: &[f32]| -> f64 {
            let mut total = 0.0;
            for x in &calib {
                for row in 0..n {
                    let (exact_sum, approx_sum) = (0..k).fold((0.0_f64, 0.0_f64), |acc, j| {
                        (
                            acc.0 + f64::from(x[j]) * f64::from(weight[row * k + j]),
                            acc.1 + f64::from(x[j]) * f64::from(q[row * k + j]),
                        )
                    });
                    total += (exact_sum - approx_sum).powi(2);
                }
            }
            total
        };
        assert!(
            error_of(&gptq) <= error_of(&rtn) * 1.05,
            "gptq error {} must not exceed rtn error {} beyond tolerance",
            error_of(&gptq),
            error_of(&rtn)
        );
    }

    #[test]
    fn inverse_hessian_solves_identity_for_diagonal_input() {
        // H = diag(2, 8): H⁻¹ = diag(1/2, 1/8) up to damping noise on the zeros.
        let hessian = vec![2.0, 0.0, 0.0, 8.0];
        let z = gptq_inverse_hessian(&hessian, 2, 0.0001).expect("invertible");
        assert!((z[0] - 0.5).abs() < 1e-6, "z00={}", z[0]);
        assert!((z[3] - 0.125).abs() < 1e-6, "z11={}", z[3]);
    }
}
