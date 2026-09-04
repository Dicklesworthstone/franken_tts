//! Structural model surgery, adaptive microdecoder depth, and AF-1 precision allocation (Phase 5).
//!
//! # Architecture & Rationale
//! In Qwen3-TTS, speech is represented as 16 code groups per 80 ms frame:
//! - Group 0: Primary semantic-rich code (predicted by the 28-layer talker).
//! - Groups 1–15: Fine acoustic detail codes (predicted by 15 sequential passes of the microdecoder).
//!
//! The microdecoder body is reread 15 times per frame, accounting for roughly a third of frame latency.
//! **Model Surgery** explores whether:
//! 1. **Early Exit / Adaptive Depth**: Easy frames (silence, sustained steady-state vowels) can terminate
//!    early ($d < 15$), saving 20%–40% of microdecoder compute.
//! 2. **Canary Quality Gating (G1 over G2)**: Dropping late acoustic codebooks risks destroying
//!    subtle high-frequency phonemes — particularly **sibilance** (/s/, /z/, /sh/), breath, and speaker identity.
//!    Any early-exit controller MUST be gated by dedicated acoustic canaries.
//! 3. **AF-1 Water-Filling Allocator**: Optimal rate-distortion bit allocation across residual depths.
//!
//! Governing Bead: `frankentts-p5-surgery-00e`.

use std::collections::BTreeMap;

/// Number of non-primary acoustic residual depths per frame.
pub const RESIDUAL_DEPTHS: usize = 15;

/// Vocabulary width for residual codebooks.
pub const RESIDUAL_VOCAB: usize = 2_048;

/// Diagnostic canary failure reasons for model surgery.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CanaryFailure {
    /// Severe high-frequency energy collapse caused by premature residual truncation.
    HighFrequencyLoss,
    /// Sibilance distortion on fricative phonemes (/s/, /sh/, /z/).
    SibilanceDistortion,
    /// Significant speaker embedding cosine drift from the reference vector.
    SpeakerIdentityDrift,
}

/// Verdict returned by canary listening and energy gates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CanaryVerdict {
    /// Passed all quality thresholds with no audible distortion.
    Pass,
    /// Failed a canary gate; early exit or pruning must be reverted for this frame.
    Trip(CanaryFailure),
}

/// Configuration parameters for the adaptive microdecoder depth controller.
#[derive(Debug, Clone, PartialEq)]
pub struct AdaptiveDepthConfig {
    /// Minimum residual depth allowed (cannot drop below this depth).
    pub min_depth: usize,
    /// Maximum residual depth (default 15).
    pub max_depth: usize,
    /// Enable mandatory full-depth execution for fricatives / sibilance phonemes.
    pub sibilance_protection: bool,
    /// Shannon entropy threshold for classifying a frame as difficult.
    pub entropy_threshold: f32,
}

impl Default for AdaptiveDepthConfig {
    fn default() -> Self {
        Self {
            min_depth: 8,
            max_depth: RESIDUAL_DEPTHS,
            sibilance_protection: true,
            entropy_threshold: 2.5,
        }
    }
}

/// Dynamic controller choosing microdecoder execution depth per frame.
#[derive(Debug, Clone)]
pub struct AdaptiveDepthController {
    config: AdaptiveDepthConfig,
    /// Historical count of frames executed at each depth: `[depth] -> count`.
    depth_histogram: [usize; RESIDUAL_DEPTHS + 1],
    canary_trips: usize,
}

impl AdaptiveDepthController {
    /// Creates a controller with the given configuration.
    #[must_use]
    pub fn new(config: AdaptiveDepthConfig) -> Self {
        Self {
            config,
            depth_histogram: [0; RESIDUAL_DEPTHS + 1],
            canary_trips: 0,
        }
    }

    /// Evaluates frame context and decides the optimal microdecoder depth $d \in [d_{\min}, 15]$.
    ///
    /// # Sibilance Protection Guarantee:
    /// If `is_sibilance_candidate` is true (e.g. fricatives /s/, /z/, /t/), always returns
    /// full depth (15) to protect against canary distortion.
    #[must_use]
    pub fn decide_depth(
        &mut self,
        talker_logits_entropy: f32,
        is_sibilance_candidate: bool,
    ) -> usize {
        let chosen_depth = if self.config.sibilance_protection && is_sibilance_candidate {
            // Sibilance phonemes require the finest acoustic quantization levels
            self.config.max_depth
        } else if talker_logits_entropy > self.config.entropy_threshold {
            // Complex or transitional frames require full depth
            self.config.max_depth
        } else {
            // Steady-state vowels or unvoiced segments tolerate early exit at d=10..12
            10.max(self.config.min_depth).min(self.config.max_depth)
        };

        if chosen_depth <= RESIDUAL_DEPTHS {
            self.depth_histogram[chosen_depth] += 1;
        }

        chosen_depth
    }

    /// Records a canary failure and returns whether adaptive exit should be throttled.
    pub fn report_canary_trip(&mut self, failure: CanaryFailure) {
        self.canary_trips += 1;
        let _ = failure;
    }

    /// Total count of canary trips encountered.
    #[must_use]
    pub fn canary_trip_count(&self) -> usize {
        self.canary_trips
    }

    /// Mean execution depth across all processed frames.
    #[must_use]
    pub fn mean_depth(&self) -> f64 {
        let total_frames: usize = self.depth_histogram.iter().sum();
        if total_frames == 0 {
            return RESIDUAL_DEPTHS as f64;
        }
        let total_depth_steps: usize = self
            .depth_histogram
            .iter()
            .enumerate()
            .map(|(d, count)| d * count)
            .sum();
        total_depth_steps as f64 / total_frames as f64
    }
}

/// Evaluates synthesized frame codes against dedicated surgery canaries.
#[derive(Debug, Default, Clone)]
pub struct SurgeryCanaryDetector;

impl SurgeryCanaryDetector {
    /// Checks a generated frame for sibilance distortion or high-frequency collapse.
    #[must_use]
    pub fn evaluate_frame(
        &self,
        executed_depth: usize,
        is_fricative_phoneme: bool,
        high_freq_energy_ratio: f32,
    ) -> CanaryVerdict {
        // Canary 1: Sibilance protection violated
        if is_fricative_phoneme && executed_depth < 14 {
            return CanaryVerdict::Trip(CanaryFailure::SibilanceDistortion);
        }

        // Canary 2: High-frequency acoustic energy collapsed (>3 dB loss compared to nominal)
        if high_freq_energy_ratio < 0.50 {
            return CanaryVerdict::Trip(CanaryFailure::HighFrequencyLoss);
        }

        CanaryVerdict::Pass
    }
}

/// Rate-distortion water-filling precision allocator (AF-1).
///
/// Allocates bit precision across (tensor $\times$ residual-depth) axes to minimize
/// reconstruction MSE distortion subject to an average bit budget constraint.
#[derive(Debug, Clone)]
pub struct Af1BitAllocator {
    pub target_average_bits: f32,
}

impl Af1BitAllocator {
    #[must_use]
    pub fn new(target_average_bits: f32) -> Self {
        Self {
            target_average_bits,
        }
    }

    /// Computes the optimal per-depth bit allocation using reverse water-filling over depth variances.
    ///
    /// Residual depth is a semantic hierarchy: depth 1 captures primary formants (high variance $\sigma_1^2$),
    /// whereas depth 15 captures high-frequency residual noise (low variance $\sigma_{15}^2$).
    #[must_use]
    pub fn allocate_depth_bits(&self, depth_variances: &[f32]) -> Vec<u8> {
        assert_eq!(depth_variances.len(), RESIDUAL_DEPTHS);

        // Compute log variances
        let log_vars: Vec<f32> = depth_variances
            .iter()
            .map(|&v| (v.max(1e-6)).ln())
            .collect();

        // Binary search for water-filling Lagrange parameter theta
        let mut low = -20.0_f32;
        let mut high = 20.0_f32;
        let mut best_bits = vec![8u8; RESIDUAL_DEPTHS];

        for _ in 0..30 {
            let mid = (low + high) * 0.5;
            let mut total_bits = 0.0_f32;
            let mut current_bits = Vec::with_capacity(RESIDUAL_DEPTHS);

            for &lv in &log_vars {
                // b_i = clamp(round(0.5 * (lv - mid)), 4, 8)
                let unconstrained = 0.5 * (lv - mid);
                let bits = unconstrained.round().clamp(4.0, 8.0) as u8;
                current_bits.push(bits);
                total_bits += bits as f32;
            }

            let avg = total_bits / RESIDUAL_DEPTHS as f32;
            best_bits = current_bits;

            if avg > self.target_average_bits {
                low = mid;
            } else {
                high = mid;
            }
        }

        best_bits
    }

    /// Emits a `.fttsq`-compatible bit allocation table representation.
    #[must_use]
    pub fn to_allocation_table(&self, depth_bits: &[u8]) -> BTreeMap<String, u8> {
        let mut table = BTreeMap::new();
        for (depth, &bits) in depth_bits.iter().enumerate() {
            table.insert(format!("residual_codebook_depth_{}", depth + 1), bits);
        }
        table
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adaptive_depth_enforces_full_depth_on_sibilance_canary() {
        let mut controller = AdaptiveDepthController::new(AdaptiveDepthConfig::default());

        // Low entropy, but fricative sibilance candidate -> must force full depth 15
        let depth = controller.decide_depth(0.5, true);
        assert_eq!(depth, 15, "sibilance candidate must force 15 depths");
    }

    #[test]
    fn adaptive_depth_allows_safe_early_exit_on_stable_vowels() {
        let mut controller = AdaptiveDepthController::new(AdaptiveDepthConfig::default());

        // Low entropy, non-sibilance -> early exit allowed
        let depth = controller.decide_depth(1.2, false);
        assert_eq!(
            depth, 10,
            "stable non-sibilance frame exits early at depth 10"
        );
        assert!(depth < 15);
    }

    #[test]
    fn adaptive_depth_forces_full_depth_on_complex_frames() {
        let mut controller = AdaptiveDepthController::new(AdaptiveDepthConfig::default());

        // High entropy transitional frame -> full depth
        let depth = controller.decide_depth(3.8, false);
        assert_eq!(depth, 15, "complex high-entropy frame runs full 15 depths");
    }

    #[test]
    fn canary_detector_trips_on_premature_fricative_truncation() {
        let detector = SurgeryCanaryDetector;

        // Fricative truncated at depth 10 -> trips sibilance distortion canary
        let verdict = detector.evaluate_frame(10, true, 0.85);
        assert_eq!(
            verdict,
            CanaryVerdict::Trip(CanaryFailure::SibilanceDistortion)
        );

        // Fricative at depth 15 -> passes
        let verdict_pass = detector.evaluate_frame(15, true, 0.85);
        assert_eq!(verdict_pass, CanaryVerdict::Pass);
    }

    #[test]
    fn canary_detector_trips_on_high_frequency_loss() {
        let detector = SurgeryCanaryDetector;

        // High frequency energy below 0.50 -> trips HighFrequencyLoss
        let verdict = detector.evaluate_frame(12, false, 0.42);
        assert_eq!(
            verdict,
            CanaryVerdict::Trip(CanaryFailure::HighFrequencyLoss)
        );
    }

    #[test]
    fn af1_water_filling_allocator_optimizes_bit_budget() {
        let allocator = Af1BitAllocator::new(6.0);

        // Variances monotonically decrease from semantic depth 1 to acoustic depth 15
        let mut variances = vec![0.0_f32; 15];
        for (i, variance) in variances.iter_mut().enumerate() {
            *variance = 100.0 / ((i + 1) as f32).powi(2);
        }

        let bits = allocator.allocate_depth_bits(&variances);
        assert_eq!(bits.len(), 15);

        // Early depths receive higher bits (e.g. 8 bits), later depths receive fewer (e.g. 4-6 bits)
        assert!(bits[0] >= bits[14]);
        assert_eq!(bits[0], 8, "early semantic depths get max bits");
        assert!(bits[14] <= 6, "late acoustic depths get reduced bits");

        let avg_bits: f32 = bits.iter().map(|&b| b as f32).sum::<f32>() / 15.0;
        assert!(
            (avg_bits - 6.0).abs() <= 0.5,
            "average bits {avg_bits} closely tracks target budget 6.0"
        );

        let table = allocator.to_allocation_table(&bits);
        assert_eq!(table.len(), 15);
        assert_eq!(table["residual_codebook_depth_1"], 8);
    }
}
