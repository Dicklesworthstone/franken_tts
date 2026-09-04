//! Alien-Artifact Family AF-3: Anytime-valid sequential test (e-process) reliability monitor
//! for FrankenMTP speculative decode.
//!
//! # Alien-Artifact Contract
//! - **Family**: AF-3
//! - **Consumer**: FrankenMTP speculative decode loop (`FTTS_SPEC_MTP` kill-switch)
//! - **Deletion Condition**: FrankenMTP abandoned or removed
//! - **Fallback**: Authoritative sequential microdecoder (exactness Tier 1)
//!
//! # Statistical Foundation (Ville's Inequality)
//! An e-process is a sequence of non-negative random variables $(E_t)_{t \ge 0}$ with $E_0 = 1$
//! that forms a supermartingale under the null hypothesis $H_0$:
//! $$\mathbb{E}_{H_0}[E_t \mid \mathcal{F}_{t-1}] \le E_{t-1}$$
//!
//! By Ville's inequality (the maximal inequality for non-negative supermartingales):
//! $$\mathbb{P}_{H_0}\left(\exists t \ge 1: E_t \ge \frac{1}{\alpha}\right) \le \alpha$$
//!
//! This provides an anytime-valid safety guarantee: at ANY stopping time $\tau$ (e.g. per-token,
//! per-frame, or across sessions), the probability of a false alarm under $H_0$ is strictly bounded
//! by $\alpha$, with no multiple-testing correction or pre-fixed horizon required.
//!
//! # Hypotheses & Update Rule
//! - $H_0$: Drafter anomaly rate is bounded by $p_0$ ($\mathbb{P}(Y_t = 1) \le p_0$).
//! - $H_1$: Drafter is misbehaving / corrupted ($\mathbb{P}(Y_t = 1) > p_0$).
//!
//! At step $t$ with anomaly indicator $Y_t \in \{0, 1\}$:
//! $$E_t = \max\left(0, E_{t-1} \cdot \left(1 + \lambda (Y_t - p_0)\right)\right)$$
//! where $\lambda \in (0, 1/p_0)$ is the betting parameter.
//!
//! If $E_t \ge 1/\alpha$, the test rejects $H_0$, sets the `alarmed` flag, and triggers an
//! automatic, irrevocable fallback to authoritative sequential decoding.
//!
//! When $\alpha \le 0$, speculation is disabled unconditionally (the deterministic fallback,
//! wired first).

use std::sync::atomic::{AtomicI8, Ordering};

/// Override switch for testing AF-3 behavior without environment variable races.
static AF3_MONITOR_OVERRIDE: AtomicI8 = AtomicI8::new(-1);

/// Sets the test override for the AF-3 monitor:
/// - `Some(true)`: forces monitor to remain healthy (alarm disabled)
/// - `Some(false)`: forces monitor into immediate demotion
/// - `None`: restores standard e-process resolution
pub fn set_af3_monitor_override(override_value: Option<bool>) {
    let val = match override_value {
        Some(true) => 1,
        Some(false) => 0,
        None => -1,
    };
    AF3_MONITOR_OVERRIDE.store(val, Ordering::Relaxed);
}

/// Configuration parameters for the AF-3 e-process reliability monitor.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FrankenMtpEProcessConfig {
    /// Nominal misbehavior / anomaly probability upper bound under $H_0$ ($0 < p_0 < 1$).
    pub p0: f64,
    /// Martingale betting parameter $\lambda \in (0, 1/p_0)$.
    pub lambda: f64,
    /// Risk level $\alpha \in (0, 1)$; alarm threshold is $1/\alpha$.
    /// When $\alpha \le 0$, speculation is unconditionally disabled (deterministic fallback).
    pub alpha: f64,
}

impl Default for FrankenMtpEProcessConfig {
    fn default() -> Self {
        Self {
            p0: 0.10,
            lambda: 2.0,
            alpha: 0.01,
        }
    }
}

impl FrankenMtpEProcessConfig {
    /// Constructs a validated configuration.
    ///
    /// # Panics
    /// Panics if $p_0 \le 0$, $p_0 \ge 1$, or $\lambda \le 0$ or $\lambda \ge 1/p_0$.
    #[must_use]
    pub fn new(p0: f64, lambda: f64, alpha: f64) -> Self {
        assert!(p0 > 0.0 && p0 < 1.0, "p0 must be in (0, 1), got {p0}");
        assert!(
            lambda > 0.0 && lambda < (1.0 / p0),
            "lambda must be in (0, 1/p0), got {lambda} with 1/p0={}",
            1.0 / p0
        );
        Self { p0, lambda, alpha }
    }

    /// Resolves configuration from environment variables, falling back to defaults:
    /// - `FTTS_AF3_P0`: nominal anomaly rate (default: 0.10)
    /// - `FTTS_AF3_LAMBDA`: betting fraction (default: 2.0)
    /// - `FTTS_AF3_ALPHA`: significance level (default: 0.01; set to 0 to disable speculation)
    #[must_use]
    pub fn from_env() -> Self {
        let p0 = std::env::var("FTTS_AF3_P0")
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
            .filter(|&v| v > 0.0 && v < 1.0)
            .unwrap_or(0.10);

        let max_lambda = 1.0 / p0;
        let lambda = std::env::var("FTTS_AF3_LAMBDA")
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
            .filter(|&v| v > 0.0 && v < max_lambda)
            .unwrap_or(2.0);

        let alpha = std::env::var("FTTS_AF3_ALPHA")
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
            .unwrap_or(0.01);

        Self { p0, lambda, alpha }
    }

    /// Alarm threshold $1/\alpha$. Returns `f64::INFINITY` if $\alpha \le 0$.
    #[must_use]
    pub fn threshold(&self) -> f64 {
        if self.alpha <= 0.0 {
            f64::INFINITY
        } else {
            1.0 / self.alpha
        }
    }
}

/// Decision emitted by the AF-3 monitor after observing a speculative proposal.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MonitorDecision {
    /// Speculation is healthy; e-value remains strictly below $1/\alpha$.
    Healthy { e_value: f64 },
    /// Alarm triggered on this step: e-value crossed $1/\alpha$.
    /// Speculation demotes to authoritative sequential decode immediately.
    Alarm { e_value: f64, steps: u64 },
    /// Speculation was previously alarmed and remains demoted.
    Demoted,
    /// Speculation is disabled by configuration ($\alpha \le 0$, deterministic fallback).
    Disabled,
}

/// The anytime-valid sequential test (e-process) monitor for FrankenMTP (AF-3).
#[derive(Clone, Debug, PartialEq)]
pub struct FrankenMtpEProcessMonitor {
    config: FrankenMtpEProcessConfig,
    e_value: f64,
    steps: u64,
    anomalies: u64,
    alarmed: bool,
}

impl Default for FrankenMtpEProcessMonitor {
    fn default() -> Self {
        Self::new(FrankenMtpEProcessConfig::default())
    }
}

impl FrankenMtpEProcessMonitor {
    /// Creates a new monitor with the given configuration.
    #[must_use]
    pub fn new(config: FrankenMtpEProcessConfig) -> Self {
        Self {
            config,
            e_value: 1.0,
            steps: 0,
            anomalies: 0,
            alarmed: false,
        }
    }

    /// Creates a new monitor using configuration from environment variables.
    #[must_use]
    pub fn from_env() -> Self {
        Self::new(FrankenMtpEProcessConfig::from_env())
    }

    /// Resets the monitor back to its initial state ($E_0 = 1.0$, 0 steps, unalarmed).
    pub fn reset(&mut self) {
        self.e_value = 1.0;
        self.steps = 0;
        self.anomalies = 0;
        self.alarmed = false;
    }

    /// Whether speculation has been demoted (due to alarm or $\alpha \le 0$).
    #[must_use]
    pub fn is_demoted(&self) -> bool {
        match AF3_MONITOR_OVERRIDE.load(Ordering::Relaxed) {
            1 => false,
            0 => true,
            _ => self.alarmed || self.config.alpha <= 0.0,
        }
    }

    /// Current e-value $E_t$.
    #[must_use]
    pub fn e_value(&self) -> f64 {
        self.e_value
    }

    /// Total speculative steps observed so far.
    #[must_use]
    pub fn steps(&self) -> u64 {
        self.steps
    }

    /// Total anomaly events observed.
    #[must_use]
    pub fn anomalies(&self) -> u64 {
        self.anomalies
    }

    /// Reference to the active configuration.
    #[must_use]
    pub const fn config(&self) -> &FrankenMtpEProcessConfig {
        &self.config
    }

    /// Observes one speculative outcome (anomaly indicator $Y_t \in \{0, 1\}$)
    /// and updates the e-process state.
    pub fn observe(&mut self, is_anomaly: bool) -> MonitorDecision {
        if self.config.alpha <= 0.0 {
            return MonitorDecision::Disabled;
        }

        if let Some(forced) = match AF3_MONITOR_OVERRIDE.load(Ordering::Relaxed) {
            1 => Some(MonitorDecision::Healthy {
                e_value: self.e_value,
            }),
            0 => Some(MonitorDecision::Demoted),
            _ => None,
        } {
            return forced;
        }

        if self.alarmed {
            return MonitorDecision::Demoted;
        }

        self.steps += 1;
        let y = if is_anomaly {
            self.anomalies += 1;
            1.0
        } else {
            0.0
        };

        // Multiplier: 1 + lambda * (Y_t - p0)
        let multiplier = 1.0 + self.config.lambda * (y - self.config.p0);
        self.e_value = (self.e_value * multiplier).max(0.0);

        let threshold = self.config.threshold();
        if self.e_value >= threshold {
            self.alarmed = true;
            MonitorDecision::Alarm {
                e_value: self.e_value,
                steps: self.steps,
            }
        } else {
            MonitorDecision::Healthy {
                e_value: self.e_value,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_has_valid_mathematical_bounds() {
        let config = FrankenMtpEProcessConfig::default();
        assert_eq!(config.p0, 0.10);
        assert_eq!(config.lambda, 2.0);
        assert_eq!(config.alpha, 0.01);
        assert_eq!(config.threshold(), 100.0);

        // Multiplier for anomaly Y=1: 1 + 2 * (1 - 0.1) = 2.8 > 1
        let anomaly_mult = 1.0 + config.lambda * (1.0 - config.p0);
        assert!((anomaly_mult - 2.8).abs() < 1e-6);

        // Multiplier for normal Y=0: 1 + 2 * (0 - 0.1) = 0.8 < 1
        let normal_mult = 1.0 + config.lambda * (0.0 - config.p0);
        assert!((normal_mult - 0.8).abs() < 1e-6);
    }

    #[test]
    fn deterministic_fallback_when_alpha_zero() {
        let config = FrankenMtpEProcessConfig::new(0.10, 2.0, 0.0);
        let mut monitor = FrankenMtpEProcessMonitor::new(config);
        assert!(
            monitor.is_demoted(),
            "alpha <= 0 must immediately demote speculation"
        );

        let decision = monitor.observe(false);
        assert_eq!(decision, MonitorDecision::Disabled);
    }

    #[test]
    fn fault_injection_broken_drafter_trips_alarm_in_bounded_steps() {
        let config = FrankenMtpEProcessConfig::new(0.10, 2.0, 0.01); // threshold = 100.0
        let mut monitor = FrankenMtpEProcessMonitor::new(config);

        // Step 0: E0 = 1.0
        // Step 1: 1.0 * 2.8 = 2.8
        // Step 2: 2.8 * 2.8 = 7.84
        // Step 3: 7.84 * 2.8 = 21.952
        // Step 4: 21.952 * 2.8 = 61.4656
        // Step 5: 61.4656 * 2.8 = 172.10368 >= 100.0 -> Alarm!
        for step in 1..=4 {
            let decision = monitor.observe(true);
            assert!(
                matches!(decision, MonitorDecision::Healthy { .. }),
                "step {step} should still be below threshold"
            );
            assert!(!monitor.is_demoted());
        }

        let alarm_decision = monitor.observe(true);
        match alarm_decision {
            MonitorDecision::Alarm { e_value, steps } => {
                assert_eq!(steps, 5);
                assert!(e_value >= 100.0);
            }
            other => panic!("expected Alarm, got {other:?}"),
        }
        assert!(monitor.is_demoted(), "monitor must now be demoted");

        // Subsequent steps return Demoted
        let next_decision = monitor.observe(false);
        assert_eq!(next_decision, MonitorDecision::Demoted);
        assert!(monitor.is_demoted());
    }

    #[test]
    fn healthy_stream_contracts_e_value_and_never_alarms() {
        let config = FrankenMtpEProcessConfig::default();
        let mut monitor = FrankenMtpEProcessMonitor::new(config);

        for _ in 0..100 {
            let decision = monitor.observe(false);
            assert!(matches!(decision, MonitorDecision::Healthy { .. }));
        }

        assert!(monitor.e_value() < 1e-6);
        assert!(!monitor.is_demoted());
        assert_eq!(monitor.steps(), 100);
        assert_eq!(monitor.anomalies(), 0);
    }

    #[test]
    fn reset_restores_initial_state() {
        let config = FrankenMtpEProcessConfig::default();
        let mut monitor = FrankenMtpEProcessMonitor::new(config);

        for _ in 0..6 {
            monitor.observe(true);
        }
        assert!(monitor.is_demoted());

        monitor.reset();
        assert!(!monitor.is_demoted());
        assert_eq!(monitor.e_value(), 1.0);
        assert_eq!(monitor.steps(), 0);
        assert_eq!(monitor.anomalies(), 0);
    }

    #[test]
    fn ville_inequality_false_alarm_rate_under_null() {
        // Calibration test: under H0 with true anomaly rate p <= p0,
        // Ville's inequality guarantees P(exists t: Et >= 1/alpha) <= alpha.
        // We simulate 1,000 independent streams of length 50 under H0 (p = 0.05 < p0 = 0.10).
        let config = FrankenMtpEProcessConfig::new(0.10, 2.0, 0.05); // threshold = 20.0
        let trials = 1000;
        let horizon = 50;
        let mut false_alarms = 0;

        // Simple LCG pseudo-random for deterministic reproducibility in unit test
        let mut lcg_state: u64 = 0xdeadbeef12345678;
        let mut next_u64 = || {
            lcg_state = lcg_state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            lcg_state
        };

        for _ in 0..trials {
            let mut monitor = FrankenMtpEProcessMonitor::new(config);
            for _ in 0..horizon {
                // Anomaly with probability 0.05 (well within H0)
                let rand_val = (next_u64() % 1000) as f64 / 1000.0;
                let is_anomaly = rand_val < 0.05;
                if let MonitorDecision::Alarm { .. } = monitor.observe(is_anomaly) {
                    false_alarms += 1;
                    break;
                }
            }
        }

        let empirical_false_alarm_rate = false_alarms as f64 / trials as f64;
        println!(
            "AF-3 Ville calibration: trials={trials}, false_alarms={false_alarms}, empirical_rate={empirical_false_alarm_rate:.4}, alpha={}",
            config.alpha
        );
        // Empirical false alarm rate must be bounded by alpha (with margin for small sample size)
        assert!(
            empirical_false_alarm_rate <= config.alpha,
            "empirical false alarm rate {empirical_false_alarm_rate} exceeded alpha {}",
            config.alpha
        );
    }
}
