//! ProductionQuality (Contract B) measurement primitives.
//!
//! The listening protocol ([`scripts/listening/`]) *analyses* measurements; this module
//! *produces* them. Every function here is a pure computation over data a harness already
//! holds — logits, token streams, transcripts, PCM — so each one is unit-testable without
//! the model and every harness number traces to one of these definitions.
//!
//! Definitions are fixed by the consuming gates, not invented here:
//!
//! * Distributional divergences compare softmaxes at the **production temperature**
//!   (`generation_config.json`, 0.9), because that is where the model actually samples.
//! * WER and the structural word-error counts are word-level edit statistics between the
//!   normalized input text and the normalized transcript of the rendered audio.
//! * The long-form drift statistic is the late-minus-early RMS decline of an utterance,
//!   the exact quantity the `longform_drift` non-inferiority family pairs on.

/// Softmax of `logits / temperature` in log space, via log-sum-exp.
///
/// Log space keeps a 2,048-wide f32 sum from underflowing; callers exponentiate per term
/// where they need probabilities.
#[must_use]
pub fn softmax_log_at(logits: &[f32], temperature: f64) -> Vec<f64> {
    assert!(temperature > 0.0, "temperature must be positive");
    let scaled: Vec<f64> = logits
        .iter()
        .map(|&value| f64::from(value) / temperature)
        .collect();
    let max = scaled.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let log_sum = max
        + scaled
            .iter()
            .map(|&value| (value - max).exp())
            .sum::<f64>()
            .ln();
    scaled.into_iter().map(|value| value - log_sum).collect()
}

/// KL(P‖Q) in nats, P from `reference` and Q from `comparison`, both softmaxed at
/// `temperature`.
///
/// Terms with vanishing P contribute nothing (0·log 0 = 0). A Q that underflows to no
/// mass where P concentrates saturates the sum at a huge finite value — log-space
/// softmax keeps every finite logit representable — and only a −inf logit yields a
/// literal +inf term. Callers comparing across routes read the magnitude, not the
/// finiteness.
#[must_use]
pub fn kl_divergence_at(reference: &[f32], comparison: &[f32], temperature: f64) -> f64 {
    debug_assert_eq!(reference.len(), comparison.len(), "logit widths must match");
    let log_p = softmax_log_at(reference, temperature);
    let log_q = softmax_log_at(comparison, temperature);
    log_p
        .iter()
        .zip(&log_q)
        .map(|(&p, &q)| p.exp() * (p - q))
        .sum()
}

/// Jensen–Shannon divergence in nats: the symmetrized, bounded (≤ ln 2) half of KL.
///
/// Reported alongside KL because KL can explode on a single near-zero Q while JS stays
/// interpretable; together they separate "one tail lost" from "distribution moved".
#[must_use]
pub fn js_divergence_at(reference: &[f32], comparison: &[f32], temperature: f64) -> f64 {
    debug_assert_eq!(reference.len(), comparison.len(), "logit widths must match");
    let log_p = softmax_log_at(reference, temperature);
    let log_q = softmax_log_at(comparison, temperature);
    // M = (P+Q)/2 in probability space; log M = logaddexp(p, q) − ln 2.
    let ln_half = (0.5_f64).ln();
    let mut kl_pm = 0.0;
    let mut kl_qm = 0.0;
    for (&p, &q) in log_p.iter().zip(&log_q) {
        let (pp, qp) = (p.exp(), q.exp());
        let log_m = add_logs(p, q) + ln_half;
        if pp > 0.0 {
            kl_pm += pp * (p - log_m);
        }
        if qp > 0.0 {
            kl_qm += qp * (q - log_m);
        }
    }
    0.5 * (kl_pm + kl_qm)
}

/// `ln(e^a + e^b)` without overflow.
fn add_logs(a: f64, b: f64) -> f64 {
    let max = a.max(b);
    if max == f64::NEG_INFINITY {
        return max;
    }
    max + ((a - max).exp() + (b - max).exp()).ln()
}

/// Indices sorted by descending logit with ascending-index tie-break — the deterministic
/// rule the selector family uses, so "top-k" means the same tokens on both sides.
#[must_use]
pub fn ranking(logits: &[f32]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..logits.len()).collect();
    order.sort_by(|&a, &b| {
        logits[b]
            .partial_cmp(&logits[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    order
}

/// Index of the largest logit; ties break toward the lowest index, matching [`ranking`].
#[must_use]
pub fn argmax_of(logits: &[f32]) -> usize {
    ranking(logits)[0]
}

/// Shared tokens between the heads of two [`ranking`] orders.
#[must_use]
pub fn top_k_overlap(a_order: &[usize], b_order: &[usize], k: usize) -> usize {
    let head_a: std::collections::HashSet<usize> = a_order.iter().take(k).copied().collect();
    b_order
        .iter()
        .take(k)
        .filter(|token| head_a.contains(*token))
        .count()
}

/// Zero-based rank of `token` within a [`ranking`] order, or `None` when absent.
#[must_use]
pub fn rank_of(token: usize, order: &[usize]) -> Option<usize> {
    order.iter().position(|&candidate| candidate == token)
}

/// First index where two token streams part, or `None` when they agree throughout.
#[must_use]
pub fn first_divergence(a: &[u32], b: &[u32]) -> Option<usize> {
    a.iter().zip(b).position(|(x, y)| x != y)
}

/// Lowercases, strips punctuation, and whitespace-splits transcript text for word metrics.
///
/// Numbers stay verbatim: "1963" vs "nineteen sixty-three" is a real intelligibility
/// difference the ASR scorer may or may not normalize, and hiding it would flatter the
/// route under test.
#[must_use]
pub fn normalize_words(text: &str) -> Vec<String> {
    text.split_whitespace()
        .map(|word| {
            word.chars()
                .filter_map(|c| {
                    if c.is_alphanumeric() {
                        Some(c.to_ascii_lowercase())
                    } else {
                        None
                    }
                })
                .collect::<String>()
        })
        .filter(|word| !word.is_empty())
        .collect()
}

/// Word-level edit statistics from a Levenshtein alignment of reference to hypothesis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct WordEditStats {
    /// Total edits (substitutions + deletions + insertions); the WER numerator.
    pub distance: usize,
    /// Reference words aligned to a different hypothesis word.
    pub substitutions: usize,
    /// Reference words with no hypothesis alignment — read as *skipped* words.
    pub deletions: usize,
    /// Hypothesis words with no reference alignment.
    pub insertions: usize,
}

impl WordEditStats {
    /// WER against a reference of `reference_len` words. A zero-length reference yields
    /// `None`: an empty truth measures nothing.
    #[must_use]
    pub fn wer(&self, reference_len: usize) -> Option<f64> {
        if reference_len == 0 {
            return None;
        }
        Some(
            f64::from(u32::try_from(self.distance).unwrap_or(u32::MAX))
                / f64::from(reference_len as u32),
        )
    }
}

/// Levenshtein alignment with operation counts.
///
/// Backtracking prefers substitution over split insert+delete, keeping the op breakdown
/// stable across equal-cost paths.
#[must_use]
pub fn word_edit_stats(reference: &[String], hypothesis: &[String]) -> WordEditStats {
    let rows = reference.len();
    let cols = hypothesis.len();
    if rows == 0 {
        return WordEditStats {
            distance: cols,
            substitutions: 0,
            deletions: 0,
            insertions: cols,
        };
    }
    if cols == 0 {
        return WordEditStats {
            distance: rows,
            substitutions: 0,
            deletions: rows,
            insertions: 0,
        };
    }

    let mut dp = vec![0_usize; (rows + 1) * (cols + 1)];
    let at = |row: usize, col: usize| row * (cols + 1) + col;
    for row in 0..=rows {
        dp[at(row, 0)] = row;
    }
    for col in 0..=cols {
        dp[at(0, col)] = col;
    }
    for row in 1..=rows {
        for col in 1..=cols {
            let substitute_cost = usize::from(reference[row - 1] != hypothesis[col - 1]);
            dp[at(row, col)] = (dp[at(row - 1, col - 1)] + substitute_cost)
                .min(dp[at(row - 1, col)] + 1)
                .min(dp[at(row, col - 1)] + 1);
        }
    }

    let mut stats = WordEditStats {
        distance: dp[at(rows, cols)],
        ..WordEditStats::default()
    };
    let mut row = rows;
    let mut col = cols;
    while row > 0 || col > 0 {
        if row > 0
            && col > 0
            && dp[at(row, col)] == dp[at(row - 1, col - 1)]
            && reference[row - 1] == hypothesis[col - 1]
        {
            row -= 1;
            col -= 1;
        } else if row > 0 && col > 0 && dp[at(row, col)] == dp[at(row - 1, col - 1)] + 1 {
            stats.substitutions += 1;
            row -= 1;
            col -= 1;
        } else if row > 0 && dp[at(row, col)] == dp[at(row - 1, col)] + 1 {
            stats.deletions += 1;
            row -= 1;
        } else {
            stats.insertions += 1;
            col -= 1;
        }
    }
    stats
}

/// Immediate echo repeats in a hypothesis: positions where a word duplicates its
/// predecessor.
///
/// Legitimate doubled words ("very very") exist in language, so this is reported as a
/// rate alongside WER rather than folded into it — the paired structural family decides
/// what matters.
#[must_use]
pub fn immediate_repetitions(hypothesis: &[String]) -> usize {
    hypothesis
        .windows(2)
        .filter(|pair| pair[0] == pair[1])
        .count()
}

/// Frame-level RMS of PCM in dBFS (`20·log10(rms)`, silence-floored at −120 dB).
#[must_use]
pub fn frame_rms_db(pcm: &[f32], samples_per_frame: usize) -> Vec<f64> {
    pcm.chunks(samples_per_frame)
        .map(|frame| {
            let mean_square = frame
                .iter()
                .map(|&s| f64::from(s) * f64::from(s))
                .sum::<f64>()
                / frame.len() as f64;
            (mean_square.sqrt().max(1.0e-6)).log10() * 20.0
        })
        .collect()
}

/// Late-minus-early energy decline of one utterance, in dB.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DriftStats {
    /// Mean frame RMS (dBFS) over the first third of frames.
    pub early_rms_db: f64,
    /// Mean frame RMS (dBFS) over the last third of frames.
    pub late_rms_db: f64,
    /// `late − early`; strongly negative values are fade-outs, the shape long-form
    /// degradation takes.
    pub decline_db: f64,
}

/// Splits `pcm` into frames of `samples_per_frame` and compares the first third's mean
/// RMS to the last third's.
///
/// `None` for utterances shorter than three frames: thirds need somewhere to stand.
#[must_use]
pub fn longform_drift(pcm: &[f32], samples_per_frame: usize) -> Option<DriftStats> {
    let rms = frame_rms_db(pcm, samples_per_frame);
    if rms.len() < 3 {
        return None;
    }
    let third = rms.len() / 3;
    let mean = |slice: &[f64]| slice.iter().sum::<f64>() / slice.len() as f64;
    let early_rms_db = mean(&rms[..third]);
    let late_rms_db = mean(&rms[rms.len() - third..]);
    Some(DriftStats {
        early_rms_db,
        late_rms_db,
        decline_db: late_rms_db - early_rms_db,
    })
}

/// Utterance duration in milliseconds per reference word.
///
/// A crude but honest prosody proxy: a route that babbles repeats or freezes inflates
/// it, one that truncates deflates it. `None` when either count is zero.
#[must_use]
pub fn duration_per_word_ms(samples: usize, sample_rate: u32, words: usize) -> Option<f64> {
    if words == 0 || sample_rate == 0 {
        return None;
    }
    Some(
        f64::from(u32::try_from(samples).ok()?) * 1000.0
            / (f64::from(sample_rate) * f64::from(words as u32)),
    )
}

/// Prosody contour of one utterance, from frame-wise fundamental-frequency estimates.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PitchContour {
    /// Median voiced f0 in Hz.
    pub median_f0_hz: f64,
    /// 12·log2(p95 / p5) over voiced frames — the utterance's pitch excursion.
    pub range_semitones: Option<f64>,
    /// `median(late third) − median(early third)` of voiced f0, in semitones — the
    /// pitch analog of [`DriftStats::decline_db`]; strongly negative is a sagging tail.
    pub late_early_drift_semitones: Option<f64>,
    /// Voiced frames over analyzed frames — how much of the utterance carried pitch.
    pub voicing_ratio: f64,
}

/// Estimates the pitch contour by normalized autocorrelation per 40 ms window, 13.3 ms
/// hop, lag range 60–400 Hz.
///
/// A frame counts as voiced when its autocorrelation clarity exceeds 0.5 and its RMS
/// clears `silence_floor` linear amplitude; unvoiced frames are skipped, not zeroed.
#[must_use]
pub fn pitch_contour(pcm: &[f32], sample_rate: u32, silence_floor: f64) -> Option<PitchContour> {
    if sample_rate == 0 || pcm.is_empty() {
        return None;
    }
    let rate = f64::from(sample_rate);
    let window = (rate * 0.04).round() as usize; // two periods at the low end
    let hop = (rate / 75.0).round() as usize; // ~13.3 ms
    let min_lag = (rate / 400.0).floor() as usize;
    let max_lag = (rate / 60.0).ceil() as usize;
    if pcm.len() < window + max_lag {
        return None;
    }

    let mut voiced_f0: Vec<f64> = Vec::new();
    let mut frame_index = 0_usize;
    while frame_index * hop + window <= pcm.len() {
        let frame = &pcm[frame_index * hop..frame_index * hop + window];
        let mean: f64 = frame.iter().map(|&s| f64::from(s)).sum::<f64>() / frame.len() as f64;
        let centered: Vec<f64> = frame.iter().map(|&s| f64::from(s) - mean).collect();
        let energy: f64 = centered.iter().map(|s| s * s).sum();
        let rms = (energy / frame.len() as f64).sqrt();
        if rms >= silence_floor && energy > 0.0 {
            // Normalized autocorrelation over the lag search range; the denominator
            // shortens with lag so long lags are not structurally penalized.
            let mut best_clarity = 0.0;
            let mut best_lag = 0_usize;
            for lag in min_lag..=max_lag.min(window - 1) {
                let mut dot = 0.0;
                let mut tail_energy = 0.0;
                for index in 0..window - lag {
                    dot += centered[index] * centered[index + lag];
                    tail_energy += centered[index + lag] * centered[index + lag];
                }
                if tail_energy == 0.0 {
                    continue;
                }
                let clarity = dot / tail_energy.sqrt();
                if clarity > best_clarity {
                    best_clarity = clarity;
                    best_lag = lag;
                }
            }
            if best_clarity > 0.5 && best_lag > 0 {
                voiced_f0.push(rate / f64::from(best_lag as u32));
            }
        }
        frame_index += 1;
    }

    if voiced_f0.is_empty() {
        return None;
    }
    let mut sorted = voiced_f0.clone();
    sorted.sort_by(f64::total_cmp);
    let median = |values: &[f64]| -> f64 {
        let mid = values.len() / 2;
        if values.len().is_multiple_of(2) {
            (values[mid - 1] + values[mid]) / 2.0
        } else {
            values[mid]
        }
    };
    let percentile = |values: &[f64], fraction: f64| -> f64 {
        let index = ((values.len() - 1) as f64 * fraction)
            .round()
            .clamp(0.0, (values.len() - 1) as f64);
        values[index as usize]
    };
    let p5 = percentile(&sorted, 0.05);
    let p95 = percentile(&sorted, 0.95);
    let third = sorted.len() / 3;
    let drift = if third > 0 && sorted.len() >= 3 {
        let early = median(&sorted[..third]);
        let late = median(&sorted[sorted.len() - third..]);
        Some(12.0 * (late / early).log2())
    } else {
        None
    };
    Some(PitchContour {
        median_f0_hz: median(&sorted),
        range_semitones: Some(12.0 * (p95 / p5).log2()),
        late_early_drift_semitones: drift,
        voicing_ratio: f64::from(u32::try_from(voiced_f0.len()).unwrap_or(u32::MAX))
            / f64::from(u32::try_from(frame_index).unwrap_or(1)),
    })
}

/// Tail-risk summary metrics (AF-2 release gate).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TailRiskSummary {
    /// Sample mean across all items.
    pub mean: f64,
    /// Sample median (50th percentile).
    pub median: f64,
    /// 90th percentile.
    pub p90: f64,
    /// 95th percentile.
    pub p95: f64,
    /// 99th percentile.
    pub p99: f64,
    /// Conditional Value at Risk at alpha = 0.10 (mean of the worst 10% tail).
    pub cvar_10: f64,
    /// Conditional Value at Risk at alpha = 0.05 (mean of the worst 5% tail).
    pub cvar_05: f64,
    /// Extreme Value Theory (EVT) Generalized Pareto tail estimate at p = 0.99, or `None` if under the sample floor.
    pub evt_p99x: Option<f64>,
    /// Number of samples analyzed.
    pub sample_count: usize,
}

/// Computes the Conditional Value at Risk (CVaR) over the worst `alpha` fraction of `samples`.
///
/// If `higher_is_worse` is true (e.g. WER, loss), evaluates the upper tail.
/// If `higher_is_worse` is false (e.g. cosine similarity), evaluates the lower tail.
/// Returns `None` if `samples` is empty or `alpha <= 0.0` or `alpha > 1.0`.
#[must_use]
pub fn cvar_alpha(samples: &[f64], alpha: f64, higher_is_worse: bool) -> Option<f64> {
    if samples.is_empty() || alpha <= 0.0 || alpha > 1.0 {
        return None;
    }
    let mut sorted: Vec<f64> = samples.iter().copied().filter(|v| v.is_finite()).collect();
    if sorted.is_empty() {
        return None;
    }
    sorted.sort_by(f64::total_cmp);
    let n = sorted.len();
    let k = ((n as f64 * alpha).round() as usize).clamp(1, n);

    if higher_is_worse {
        let tail = &sorted[n - k..];
        Some(tail.iter().sum::<f64>() / tail.len() as f64)
    } else {
        let tail = &sorted[..k];
        Some(tail.iter().sum::<f64>() / tail.len() as f64)
    }
}

/// Fits a Generalized Pareto Distribution (GPD) via method of moments on excesses over threshold `u`,
/// estimating the extreme quantile at probability `p`.
///
/// Returns `None` if `samples.len() < 20` or excess count `< 5` (sample-size floor per honesty doctrine).
#[must_use]
pub fn evt_gpd_tail_quantile(samples: &[f64], p: f64, threshold_quantile: f64) -> Option<f64> {
    if samples.len() < 20 || p <= threshold_quantile || p >= 1.0 || threshold_quantile <= 0.0 {
        return None;
    }
    let mut sorted: Vec<f64> = samples.iter().copied().filter(|v| v.is_finite()).collect();
    if sorted.len() < 20 {
        return None;
    }
    sorted.sort_by(f64::total_cmp);
    let n = sorted.len();

    let u_idx = ((n - 1) as f64 * threshold_quantile).round() as usize;
    let u = sorted[u_idx];

    let excesses: Vec<f64> = sorted.iter().copied().filter(|&x| x > u).map(|x| x - u).collect();
    let k = excesses.len();
    if k < 5 {
        return None;
    }

    let mean_excess = excesses.iter().sum::<f64>() / k as f64;
    let var_excess = excesses
        .iter()
        .map(|&y| (y - mean_excess) * (y - mean_excess))
        .sum::<f64>()
        / (k - 1) as f64;

    if mean_excess <= 0.0 || var_excess <= 0.0 {
        return None;
    }

    let ratio = (mean_excess * mean_excess) / var_excess;
    let xi = 0.5 * (1.0 - ratio);
    let sigma = mean_excess * (1.0 - xi);

    if sigma <= 0.0 {
        // Fallback to exponential tail
        let factor = (k as f64) / (n as f64 * (1.0 - p));
        if factor <= 0.0 {
            return None;
        }
        return Some(u + mean_excess * factor.ln());
    }

    if xi.abs() < 1.0e-4 {
        // Exponential tail limit
        let factor = (k as f64) / (n as f64 * (1.0 - p));
        if factor <= 0.0 {
            return None;
        }
        Some(u + sigma * factor.ln())
    } else {
        let prob_ratio = (n as f64 / k as f64) * (1.0 - p);
        if prob_ratio <= 0.0 {
            return None;
        }
        let term = prob_ratio.powf(-xi);
        Some(u + (sigma / xi) * (term - 1.0))
    }
}

/// Computes the comprehensive tail-risk summary over a sample vector.
#[must_use]
pub fn compute_tail_risk(samples: &[f64], higher_is_worse: bool) -> Option<TailRiskSummary> {
    let mut sorted: Vec<f64> = samples.iter().copied().filter(|v| v.is_finite()).collect();
    if sorted.is_empty() {
        return None;
    }
    sorted.sort_by(f64::total_cmp);
    let n = sorted.len();

    let mean = sorted.iter().sum::<f64>() / n as f64;
    let percentile = |frac: f64| -> f64 {
        let idx = ((n - 1) as f64 * frac).round() as usize;
        sorted[idx.min(n - 1)]
    };

    let median = percentile(0.50);
    let p90 = percentile(0.90);
    let p95 = percentile(0.95);
    let p99 = percentile(0.99);

    let cvar_10 = cvar_alpha(&sorted, 0.10, higher_is_worse)?;
    let cvar_05 = cvar_alpha(&sorted, 0.05, higher_is_worse)?;
    let evt_p99x = if higher_is_worse {
        evt_gpd_tail_quantile(&sorted, 0.99, 0.85)
    } else {
        // Invert to upper tail for EVT then invert result
        let inverted: Vec<f64> = sorted.iter().map(|&x| -x).collect();
        evt_gpd_tail_quantile(&inverted, 0.99, 0.85).map(|val| -val)
    };

    Some(TailRiskSummary {
        mean,
        median,
        p90,
        p95,
        p99,
        cvar_10,
        cvar_05,
        evt_p99x,
        sample_count: n,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1.0e-9;

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1.0e-9 * (1.0 + a.abs().max(b.abs()))
    }

    #[test]
    fn identical_distributions_have_zero_divergence() {
        let logits = [0.3_f32, -1.2, 4.0, 2.2, -0.7];
        assert!(close(kl_divergence_at(&logits, &logits, 0.9), 0.0));
        assert!(close(js_divergence_at(&logits, &logits, 0.9), 0.0));
    }

    #[test]
    fn kl_is_non_negative_and_js_bounded_by_ln2() {
        let reference = [5.0_f32, 1.0, 0.1, -2.0];
        let comparison = [-2.0_f32, 0.1, 1.0, 5.0];
        let kl = kl_divergence_at(&reference, &comparison, 0.9);
        let js = js_divergence_at(&reference, &comparison, 0.9);
        assert!(kl > 0.0, "distinct distributions must have positive KL");
        assert!(
            js > 0.0 && js <= std::f64::consts::LN_2 + EPS,
            "JS must be in [0, ln 2]: {js}"
        );
    }

    #[test]
    fn temperature_flattens_and_sharpens_as_documented() {
        let peaked = [8.0_f32, 0.0, 0.0];
        let flat = [0.0_f32; 3];
        // Higher temperature moves both distributions toward uniform, shrinking their gap.
        let sharp_gap = kl_divergence_at(&peaked, &flat, 0.1);
        let soft_gap = kl_divergence_at(&peaked, &flat, 4.0);
        assert!(
            sharp_gap > soft_gap,
            "lower temperature must widen the divergence"
        );
    }

    #[test]
    fn softmax_log_normalizes_exactly() {
        let log_p = softmax_log_at(&[1.0_f32, 2.0, 3.0], 1.0);
        let total: f64 = log_p.iter().map(|value| value.exp()).sum();
        assert!(
            close(total, 1.0),
            "probabilities must sum to 1, got {total}"
        );
    }

    #[test]
    fn kl_saturates_when_comparison_mass_underflows_where_reference_concentrates() {
        let sharp = [50.0_f32, 0.0];
        let shifted = [0.0_f32, 50.0];
        let kl = kl_divergence_at(&sharp, &shifted, 1.0);
        // Log-space softmax keeps the lost tail representable, so this saturates near
        // the logit gap (≈50 nats) instead of reaching a literal infinity — which is
        // exactly why route comparisons read magnitudes rather than finiteness.
        assert!(kl > 40.0, "a 50-nat mass displacement must dominate: {kl}");
    }

    #[test]
    fn ranking_orders_by_descending_logit_then_ascending_index() {
        assert_eq!(ranking(&[1.0_f32, 3.0, 3.0, 0.5]), vec![1, 2, 0, 3]);
        assert_eq!(argmax_of(&[1.0, 3.0, 3.0]), 1);
    }

    #[test]
    fn top_k_overlap_counts_shared_heads() {
        let a = vec![7, 3, 9, 1];
        let b = vec![3, 7, 5, 2];
        assert_eq!(top_k_overlap(&a, &b, 2), 2);
        assert_eq!(top_k_overlap(&a, &b, 1), 0);
        assert_eq!(top_k_overlap(&a, &b, 4), 2);
    }

    #[test]
    fn rank_of_finds_positions_or_nothing() {
        assert_eq!(rank_of(9, &[7, 3, 9, 1]), Some(2));
        assert_eq!(rank_of(4, &[7, 3, 9, 1]), None);
    }

    #[test]
    fn first_divergence_localizes_the_parting_point() {
        assert_eq!(first_divergence(&[1, 2, 3], &[1, 2, 3]), None);
        assert_eq!(first_divergence(&[1, 2, 3], &[1, 9, 3]), Some(1));
        assert_eq!(first_divergence(&[], &[]), None);
    }

    #[test]
    fn normalize_words_lowercases_strips_punctuation_keeps_numbers() {
        assert_eq!(
            normalize_words("In 1963, forty-two volunteers joined!"),
            vec!["in", "1963", "fortytwo", "volunteers", "joined"]
        );
    }

    #[test]
    fn wer_matches_hand_computed_edit_distance() {
        let reference = normalize_words("the quick brown fox jumps");
        let hypothesis = normalize_words("the quick brown dog jumps jumps");
        let stats = word_edit_stats(&reference, &hypothesis);
        // fox→dog substitution, plus one inserted "jumps": distance 2 of 5 words.
        assert_eq!(stats.distance, 2);
        assert_eq!(stats.substitutions, 1);
        assert_eq!(stats.insertions, 1);
        assert_eq!(stats.deletions, 0);
        assert!(close(stats.wer(5).expect("nonempty"), 0.4));
    }

    #[test]
    fn skipped_words_surface_as_deletions() {
        let reference = normalize_words("one two three four");
        let hypothesis = normalize_words("one three four");
        let stats = word_edit_stats(&reference, &hypothesis);
        assert_eq!(stats.distance, 1);
        assert_eq!(stats.deletions, 1, "the dropped 'two' must be a deletion");
    }

    #[test]
    fn empty_reference_yields_no_wer_instead_of_a_fake_number() {
        let stats = word_edit_stats(&[], &normalize_words("something"));
        assert_eq!(stats.wer(0), None);
    }

    #[test]
    fn immediate_repetitions_count_consecutive_echoes() {
        assert_eq!(
            immediate_repetitions(&["a".into(), "a".into(), "b".into(), "b".into()]),
            2
        );
        assert_eq!(immediate_repetitions(&["a".into(), "b".into()]), 0);
    }

    #[test]
    fn drift_measures_late_minus_early_decline() {
        // 6 frames of 4 samples: loud, loud, loud, quiet, quiet, quiet.
        let mut pcm = Vec::new();
        for frame in 0..6 {
            let amplitude = if frame < 3 { 0.5 } else { 0.05 };
            pcm.extend(std::iter::repeat_n(amplitude, 4));
        }
        let drift = longform_drift(&pcm, 4).expect("six frames");
        assert!(
            drift.decline_db < -19.9,
            "a 10x amplitude drop is ~-20 dB: {:?}",
            drift
        );
        assert!(drift.early_rms_db > drift.late_rms_db);
    }

    #[test]
    fn drift_refuses_too_short_utterances() {
        let pcm = vec![0.1_f32; 8];
        assert!(
            longform_drift(&pcm, 4).is_none(),
            "two frames cannot form thirds"
        );
    }

    #[test]
    fn duration_per_word_divides_honestly() {
        // 24_000 samples at 24 kHz = 1000 ms; 4 words → 250 ms/word.
        assert!(close(
            duration_per_word_ms(24_000, 24_000, 4).expect("valid"),
            250.0
        ));
        assert_eq!(duration_per_word_ms(24_000, 24_000, 0), None);
    }

    fn sine(f0_hz: f64, seconds: f64, rate: u32, amplitude: f32) -> Vec<f32> {
        let count = (rate as f64 * seconds) as usize;
        (0..count)
            .map(|n| {
                amplitude
                    * (2.0 * std::f64::consts::PI * f0_hz * n as f64 / f64::from(rate)).sin() as f32
            })
            .collect()
    }

    #[test]
    fn pitch_contour_tracks_a_pure_tone() {
        let contour = pitch_contour(&sine(200.0, 1.5, 24_000, 0.4), 24_000, 0.01).expect("voiced");
        assert!(
            (contour.median_f0_hz - 200.0).abs() < 4.0,
            "200 Hz tone must land within one lag step: {}",
            contour.median_f0_hz
        );
        assert!(
            contour.voicing_ratio > 0.9,
            "a pure tone is voiced throughout"
        );
        assert!(
            contour.range_semitones.expect("range") < 2.0,
            "constant pitch has negligible excursion"
        );
    }

    #[test]
    fn pitch_contour_refuses_silence() {
        let silence = vec![0.0_f32; 48_000];
        assert!(pitch_contour(&silence, 24_000, 0.01).is_none());
    }

    #[test]
    fn pitch_contour_measures_a_pitch_step_in_semitones() {
        let mut pcm = sine(150.0, 0.75, 24_000, 0.4);
        pcm.extend(sine(300.0, 0.75, 24_000, 0.4));
        let contour = pitch_contour(&pcm, 24_000, 0.01).expect("voiced");
        let drift = contour
            .late_early_drift_semitones
            .expect("two thirds of frames exist");
        assert!(
            (drift - 12.0).abs() < 1.0,
            "150→300 Hz is exactly +12 semitones: {drift}"
        );
    }

    #[test]
    fn pitch_contour_needs_enough_audio() {
        assert!(pitch_contour(&sine(200.0, 0.01, 24_000, 0.4), 24_000, 0.01).is_none());
    }

    #[test]
    fn cvar_computes_expected_shortfall_on_uniform_distribution() {
        // Uniform 0 to 100 with 100 samples: [1.0, 2.0, ..., 100.0]
        let samples: Vec<f64> = (1..=100).map(|x| x as f64).collect();

        // Higher is worse: top 10% is 91..=100 -> mean is 95.5
        let cvar10_upper = cvar_alpha(&samples, 0.10, true).expect("cvar10 upper");
        assert!((cvar10_upper - 95.5).abs() < 1e-6);

        // Lower is worse: bottom 10% is 1..=10 -> mean is 5.5
        let cvar10_lower = cvar_alpha(&samples, 0.10, false).expect("cvar10 lower");
        assert!((cvar10_lower - 5.5).abs() < 1e-6);
    }

    #[test]
    fn evt_gpd_quantile_recovers_exponential_tail() {
        // Exponential distribution with mean scale lambda = 1.0
        // F(x) = 1 - exp(-x), Quantile Q(p) = -ln(1 - p)
        // Q(0.99) = -ln(0.01) ≈ 4.60517
        let n = 200;
        let samples: Vec<f64> = (1..=n)
            .map(|i| {
                let p = (i as f64 - 0.5) / n as f64;
                -(1.0 - p).ln()
            })
            .collect();

        let est_p99 = evt_gpd_tail_quantile(&samples, 0.99, 0.85).expect("evt estimate");
        let true_p99 = -(0.01f64).ln();
        assert!(
            (est_p99 - true_p99).abs() < 0.25,
            "EVT GPD estimate {est_p99} must recover true exponential quantile {true_p99}"
        );
    }

    #[test]
    fn tail_risk_summary_reports_complete_envelope_and_refuses_empty() {
        assert!(compute_tail_risk(&[], true).is_none());

        let samples: Vec<f64> = (1..=100).map(|x| x as f64).collect();
        let summary = compute_tail_risk(&samples, true).expect("summary");
        assert_eq!(summary.sample_count, 100);
        assert!((summary.mean - 50.5).abs() < 1e-6);
        assert!((summary.median - 50.5).abs() < 1e-6);
        assert!((summary.cvar_10 - 95.5).abs() < 1e-6);
        assert!((summary.cvar_05 - 98.0).abs() < 1e-6);
    }
}

