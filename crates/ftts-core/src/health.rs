//! Runtime health: catch a run going wrong *while it is still running*.
//!
//! [`admission`](crate::admission) answers "will this fit" before anything is allocated. This
//! module answers a different question: the request was affordable and started, so **is it still
//! behaving?** Every detector here exists because the corresponding failure produces output that
//! looks superficially fine — a plausible-length WAV, a completed run, a nonzero byte count — and
//! an agent consuming `ftts` cannot listen to it. Silence, a stuck decoder, and a repetition loop
//! all sound like "success" to a program.
//!
//! Each detector is a small, allocation-free state machine that a hot loop can call per frame, and
//! each violation carries a remedy rather than a bare label. Nothing here samples wall-clock or
//! spawns a thread: the caller supplies `Instant`s, so tests inject time instead of sleeping.
//!
//! # Why the seam policy is configurable
//!
//! A NaN check over every activation of every layer is a real cost in the steady-state decode
//! loop, and the loop is the whole project. So the policy is explicit: [`SeamPolicy::All`] while
//! developing a kernel, [`SeamPolicy::Sampled`] in production where a NaN that appears at all will
//! almost certainly appear again within a few frames, [`SeamPolicy::Off`] only for a measured
//! benchmark. What is *not* offered is a silent default that quietly stops checking.
//!
//! Bead: `frankentts-v-reliability-d65`.

use core::fmt;
use core::time::Duration;

use crate::admission::StopReason;

/// A named point in the pipeline where values can be inspected.
///
/// Deliberately coarse. These are the boundaries where a numeric fault becomes *observable*, not
/// every tensor: a NaN born in the talker shows up at its logits, and one born in the codec shows
/// up in the PCM.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Seam {
    /// Talker logits, before sampling.
    TalkerLogits,
    /// Microdecoder logits for one residual depth.
    MicrodecoderLogits,
    /// Codec decoder output, before PCM quantisation.
    CodecOutput,
    /// Final PCM handed to the sink.
    Pcm,
}

impl Seam {
    /// Stable wire string for robot mode.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::TalkerLogits => "talker_logits",
            Self::MicrodecoderLogits => "microdecoder_logits",
            Self::CodecOutput => "codec_output",
            Self::Pcm => "pcm",
        }
    }
}

/// How often numeric seams are checked.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SeamPolicy {
    /// Never check. Only for a measured benchmark; a run under this policy may not be reported as
    /// numerically clean, because nothing looked.
    Off,
    /// Check every `every`-th call. A NaN that occurs at all recurs within a few frames in
    /// practice, so sampling trades a bounded detection delay for a hot loop that stays hot.
    Sampled { every: u32 },
    /// Check every call. The kernel-development setting.
    All,
}

impl SeamPolicy {
    /// Whether a run under this policy is entitled to claim it was numerically checked.
    ///
    /// `Off` is not. This exists so a report can say "unchecked" instead of implying clean.
    #[must_use]
    pub const fn is_checking(self) -> bool {
        !matches!(self, Self::Off)
    }
}

/// A detected runtime-health problem.
///
/// Every variant is `Copy` and carries only scalars, so it can travel through the observer without
/// allocating on the hot path.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HealthViolation {
    /// A non-finite value reached a seam.
    NonFinite {
        seam: Seam,
        /// Index within the inspected slice, so the fault is locatable.
        index: usize,
        /// Whether it was NaN (`true`) or an infinity.
        is_nan: bool,
    },
    /// No frame boundary was reached within the watchdog timeout.
    NoProgress {
        frames_emitted: u64,
        stalled_millis: u64,
    },
    /// The reported stop reason contradicts the observed counters.
    StopInconsistent {
        claimed: StopReason,
        frames_emitted: u64,
        frame_cap: u64,
    },
    /// One token, or a short cycle, repeated past the runaway threshold.
    RepetitionRunaway { token: u32, repeats: u32 },
    /// Output stayed below the silence floor for longer than allowed.
    OutputSilent { silent_millis: u64 },
    /// An optimised kernel failed its selftest and the certified scalar path took over.
    KernelDemoted { from: KernelTier, to: KernelTier },
    /// Sustained throughput fell materially below the opening window.
    ThermalDegraded {
        /// Percent below baseline, rounded down.
        percent_below_baseline: u32,
    },
    /// FrankenMTP speculative decode misbehavior exceeded the sequential-test e-value threshold;
    /// speculative decode alarmed and demoted to authoritative sequential execution (AF-3).
    SpeculationDemoted {
        /// The e-value observed when the alarm threshold was crossed, scaled by 100 as integer.
        e_value_x100: u64,
        /// Number of speculative proposals evaluated before demotion.
        steps_observed: u64,
    },
}

impl HealthViolation {
    /// Stable wire string for robot mode.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::NonFinite { .. } => "non_finite",
            Self::NoProgress { .. } => "no_progress",
            Self::StopInconsistent { .. } => "stop_inconsistent",
            Self::RepetitionRunaway { .. } => "repetition_runaway",
            Self::OutputSilent { .. } => "output_silent",
            Self::KernelDemoted { .. } => "kernel_demoted",
            Self::ThermalDegraded { .. } => "thermal_degraded",
            Self::SpeculationDemoted { .. } => "speculation_demoted",
        }
    }

    /// Whether this violation means the audio must not be presented as a clean result.
    ///
    /// A demotion and a thermal report are *informational*: the run is still correct, just slower
    /// or on a different kernel tier. The rest mean the output is wrong, stuck, or empty.
    #[must_use]
    pub const fn invalidates_output(self) -> bool {
        !matches!(
            self,
            Self::KernelDemoted { .. }
                | Self::ThermalDegraded { .. }
                | Self::SpeculationDemoted { .. }
        )
    }

    /// What the caller should actually do about it.
    #[must_use]
    pub const fn remedy(self) -> &'static str {
        match self {
            Self::NonFinite { .. } => {
                "a non-finite value reached this seam: rerun with FTTS_MATH_MODE=strict; if it \
                 persists there, the fault is in the kernel rather than a fast-math approximation"
            }
            Self::NoProgress { .. } => {
                "generation stopped advancing: cancel and retry; if reproducible, capture the \
                 prompt — a stalled decode loop is a bug, not a capacity problem"
            }
            Self::StopInconsistent { .. } => {
                "the stop reason disagrees with the frame counters; treat this result as \
                 untrusted and report it — one of the two is lying about whether audio was cut off"
            }
            Self::RepetitionRunaway { .. } => {
                "the model entered a repetition loop: raise the repetition penalty or shorten the \
                 input; the audio to this point is usable, everything after the loop began is not"
            }
            Self::OutputSilent { .. } => {
                "output was silent past the allowed window: check the voice pack and reference \
                 audio; a silent result is a failure even though it produced bytes"
            }
            Self::KernelDemoted { .. } => {
                "an optimised kernel failed its selftest and the certified scalar path took over: \
                 results stay correct and slower; report the ISA and CPU"
            }
            Self::ThermalDegraded { .. } => {
                "sustained throughput fell below the opening window: expected under thermal load; \
                 do not quote this run's rate as a steady-state number"
            }
            Self::SpeculationDemoted { .. } => {
                "the speculative drafter exceeded the sequential-test error threshold (AF-3); \
                 execution automatically demoted to authoritative sequential decode. The output \
                 remains bit-exact; no action is required unless investigating drafter quality"
            }
        }
    }
}

impl fmt::Display for HealthViolation {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonFinite {
                seam,
                index,
                is_nan,
            } => write!(
                formatter,
                "{} at {} index {index}",
                if *is_nan { "NaN" } else { "infinity" },
                seam.as_str()
            ),
            Self::NoProgress {
                frames_emitted,
                stalled_millis,
            } => write!(
                formatter,
                "no frame progress for {stalled_millis} ms after {frames_emitted} frame(s)"
            ),
            Self::StopInconsistent {
                claimed,
                frames_emitted,
                frame_cap,
            } => write!(
                formatter,
                "stop reason {} contradicts {frames_emitted} frame(s) against a cap of {frame_cap}",
                claimed.as_str()
            ),
            Self::RepetitionRunaway { token, repeats } => {
                write!(formatter, "token {token} repeated {repeats} times")
            }
            Self::OutputSilent { silent_millis } => {
                write!(formatter, "output silent for {silent_millis} ms")
            }
            Self::KernelDemoted { from, to } => write!(
                formatter,
                "kernel demoted from {} to {}",
                from.as_str(),
                to.as_str()
            ),
            Self::ThermalDegraded {
                percent_below_baseline,
            } => write!(
                formatter,
                "throughput {percent_below_baseline}% below the opening window"
            ),
            Self::SpeculationDemoted {
                e_value_x100,
                steps_observed,
            } => write!(
                formatter,
                "speculation demoted to sequential at e-value {:.2} after {steps_observed} step(s)",
                *e_value_x100 as f64 / 100.0
            ),
        }
    }
}

// --------------------------------------------------------------------------------------
// NaN / Inf seams
// --------------------------------------------------------------------------------------

/// Checks numeric seams according to a [`SeamPolicy`].
#[derive(Clone, Debug)]
pub struct NumericGuard {
    policy: SeamPolicy,
    calls: u32,
}

impl NumericGuard {
    #[must_use]
    pub const fn new(policy: SeamPolicy) -> Self {
        Self { policy, calls: 0 }
    }

    #[must_use]
    pub const fn policy(&self) -> SeamPolicy {
        self.policy
    }

    /// Inspect one slice at `seam`, honouring the sampling policy.
    ///
    /// Reports the **first** offending index rather than a count: the first NaN is where the fault
    /// entered, and everything after it is downstream contamination.
    pub fn check(&mut self, seam: Seam, values: &[f32]) -> Result<(), HealthViolation> {
        if !self.should_check() {
            return Ok(());
        }
        for (index, value) in values.iter().enumerate() {
            if !value.is_finite() {
                return Err(HealthViolation::NonFinite {
                    seam,
                    index,
                    is_nan: value.is_nan(),
                });
            }
        }
        Ok(())
    }

    fn should_check(&mut self) -> bool {
        match self.policy {
            SeamPolicy::Off => false,
            SeamPolicy::All => true,
            SeamPolicy::Sampled { every } => {
                // `every == 0` would divide by zero; treat it as "every call" rather than
                // silently disabling the guard, because a misconfigured sampler must not be the
                // reason nothing was checked.
                if every <= 1 {
                    return true;
                }
                let due = self.calls.is_multiple_of(every);
                self.calls = self.calls.wrapping_add(1);
                due
            }
        }
    }
}

// --------------------------------------------------------------------------------------
// No-progress watchdog
// --------------------------------------------------------------------------------------

/// Detects a decode loop that stopped advancing.
///
/// Takes the current time from the caller rather than reading the clock, so a test proves the
/// stall behaviour in microseconds instead of sleeping through the timeout.
#[derive(Clone, Debug)]
pub struct ProgressWatchdog<T> {
    timeout: Duration,
    last_progress: T,
    frames_emitted: u64,
}

impl<T: Copy + core::ops::Sub<T, Output = Duration>> ProgressWatchdog<T> {
    #[must_use]
    pub const fn new(timeout: Duration, started: T) -> Self {
        Self {
            timeout,
            last_progress: started,
            frames_emitted: 0,
        }
    }

    /// Record a frame boundary, resetting the stall timer.
    pub fn record_frame(&mut self, now: T) {
        self.frames_emitted += 1;
        self.last_progress = now;
    }

    #[must_use]
    pub const fn frames_emitted(&self) -> u64 {
        self.frames_emitted
    }

    /// Fail if no frame has been recorded within the timeout.
    pub fn check(&self, now: T) -> Result<(), HealthViolation> {
        let stalled = now - self.last_progress;
        if stalled > self.timeout {
            return Err(HealthViolation::NoProgress {
                frames_emitted: self.frames_emitted,
                stalled_millis: u64::try_from(stalled.as_millis()).unwrap_or(u64::MAX),
            });
        }
        Ok(())
    }
}

// --------------------------------------------------------------------------------------
// Stop-reason consistency
// --------------------------------------------------------------------------------------

/// Cross-check a claimed stop reason against the counters that should corroborate it.
///
/// This is a guard against the *reporting* path lying, which matters more here than usual: the
/// whole truncation story rests on `StopReason` being trustworthy, and an agent has no way to
/// verify it by listening. Two contradictions are detectable without model knowledge:
///
/// - `EndOfSpeech` while sitting exactly on the frame cap. Landing on the cap by coincidence is
///   possible but overwhelmingly likely to be a cap-stop mislabelled as a clean finish — which is
///   precisely the counterfeit green this bead exists to prevent.
/// - `FrameCapReached` while short of the cap. The cap demonstrably did not stop it.
pub fn check_stop_consistency(
    claimed: StopReason,
    frames_emitted: u64,
    frame_cap: u64,
) -> Result<(), HealthViolation> {
    let inconsistent = match claimed {
        StopReason::EndOfSpeech => frames_emitted >= frame_cap,
        StopReason::FrameCapReached => frames_emitted < frame_cap,
        // A duration limit is orthogonal to the frame cap, and a cancellation can land anywhere.
        StopReason::DurationLimitReached | StopReason::Cancelled => false,
    };
    if inconsistent {
        return Err(HealthViolation::StopInconsistent {
            claimed,
            frames_emitted,
            frame_cap,
        });
    }
    Ok(())
}

// --------------------------------------------------------------------------------------
// Repetition runaway
// --------------------------------------------------------------------------------------

/// Detects the classic autoregressive failure: the model latching onto a token or a short cycle.
///
/// Tracks consecutive repeats of a single token and, separately, a repeating short cycle — a
/// two-token ping-pong never trips a consecutive-repeat counter but is just as dead.
#[derive(Clone, Debug)]
pub struct RunawayDetector {
    max_consecutive: u32,
    max_cycle_repeats: u32,
    last: Option<u32>,
    consecutive: u32,
    recent: [u32; Self::CYCLE_WINDOW],
    filled: usize,
    cycle_repeats: u32,
}

impl RunawayDetector {
    /// Longest cycle length considered. Longer cycles are the sampler's business, not a hang.
    const CYCLE_WINDOW: usize = 8;

    #[must_use]
    pub const fn new(max_consecutive: u32, max_cycle_repeats: u32) -> Self {
        Self {
            max_consecutive,
            max_cycle_repeats,
            last: None,
            consecutive: 0,
            recent: [u32::MAX; Self::CYCLE_WINDOW],
            filled: 0,
            cycle_repeats: 0,
        }
    }

    /// Observe one sampled token.
    pub fn observe(&mut self, token: u32) -> Result<(), HealthViolation> {
        if self.last == Some(token) {
            self.consecutive += 1;
        } else {
            self.consecutive = 1;
            self.last = Some(token);
        }
        if self.consecutive > self.max_consecutive {
            return Err(HealthViolation::RepetitionRunaway {
                token,
                repeats: self.consecutive,
            });
        }

        // Two-token cycle detection: compare against the token two positions back.
        if self.filled >= 2 && self.recent[(self.filled - 2) % Self::CYCLE_WINDOW] == token {
            self.cycle_repeats += 1;
            if self.cycle_repeats > self.max_cycle_repeats {
                return Err(HealthViolation::RepetitionRunaway {
                    token,
                    repeats: self.cycle_repeats,
                });
            }
        } else {
            self.cycle_repeats = 0;
        }
        self.recent[self.filled % Self::CYCLE_WINDOW] = token;
        self.filled += 1;
        Ok(())
    }
}

// --------------------------------------------------------------------------------------
// Output silence
// --------------------------------------------------------------------------------------

/// Detects an utterance that is producing bytes but no sound.
///
/// A silent result is a *failure* that every byte-count and duration check calls success, which is
/// exactly why it needs its own detector.
#[derive(Clone, Debug)]
pub struct SilenceDetector {
    floor: i16,
    max_silent_samples: u64,
    sample_rate: u32,
    silent_samples: u64,
}

impl SilenceDetector {
    /// `floor` is the absolute amplitude at or below which a sample counts as silent.
    ///
    /// The window is converted to a **sample count** once, here, rather than compared in
    /// milliseconds on every packet. Comparing integer milliseconds truncates: at 24 kHz a
    /// 100 ms window and 2,401 silent samples both round to 100 ms, so the boundary case
    /// silently failed to fire. Samples are the unit the detector actually counts.
    #[must_use]
    pub const fn new(floor: i16, max_silent: Duration, sample_rate: u32) -> Self {
        Self {
            floor,
            max_silent_samples: (max_silent.as_millis() as u64) * (sample_rate as u64) / 1000,
            sample_rate,
            silent_samples: 0,
        }
    }

    /// Observe one PCM packet. Any sample above the floor resets the run.
    pub fn observe(&mut self, samples: &[i16]) -> Result<(), HealthViolation> {
        for sample in samples {
            if sample.saturating_abs() > self.floor {
                self.silent_samples = 0;
            } else {
                self.silent_samples += 1;
            }
        }
        if self.sample_rate == 0 {
            return Ok(());
        }
        if self.silent_samples > self.max_silent_samples {
            return Err(HealthViolation::OutputSilent {
                silent_millis: self.silent_millis(),
            });
        }
        Ok(())
    }

    /// Milliseconds of trailing silence observed so far.
    #[must_use]
    pub const fn silent_millis(&self) -> u64 {
        if self.sample_rate == 0 {
            return 0;
        }
        self.silent_samples * 1000 / self.sample_rate as u64
    }
}

// --------------------------------------------------------------------------------------
// Kernel demotion
// --------------------------------------------------------------------------------------

/// Which kernel tier is executing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KernelTier {
    /// An ISA-specialised path, named by its feature (`i8mm`, `avx512-vnni`, …).
    Optimized(&'static str),
    /// The certified scalar baseline that every target can compile and every tier is proved against.
    Scalar,
}

impl KernelTier {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Optimized(name) => name,
            Self::Scalar => "scalar",
        }
    }
}

/// Selects a kernel tier and demotes to the certified scalar baseline on selftest failure.
///
/// The direction is one-way on purpose. A tier that failed its selftest has produced at least one
/// wrong answer on this machine; re-promoting it later because a subsequent check passed would be
/// trusting the same evidence that already lied. Correctness outranks speed (G1 > G2), so the run
/// finishes slower and correct.
#[derive(Clone, Debug)]
pub struct KernelSelector {
    preferred: KernelTier,
    active: KernelTier,
    demoted: bool,
}

impl KernelSelector {
    #[must_use]
    pub const fn new(preferred: KernelTier) -> Self {
        Self {
            preferred,
            active: preferred,
            demoted: false,
        }
    }

    #[must_use]
    pub const fn active(&self) -> KernelTier {
        self.active
    }

    #[must_use]
    pub const fn demoted(&self) -> bool {
        self.demoted
    }

    /// Record a selftest failure and fall back. Idempotent; already-scalar stays scalar.
    pub fn on_selftest_failure(&mut self) -> Option<HealthViolation> {
        if self.demoted || matches!(self.active, KernelTier::Scalar) {
            self.demoted = true;
            self.active = KernelTier::Scalar;
            return None;
        }
        let from = self.active;
        self.active = KernelTier::Scalar;
        self.demoted = true;
        Some(HealthViolation::KernelDemoted {
            from,
            to: KernelTier::Scalar,
        })
    }

    /// The tier that was preferred before any demotion, for reporting.
    #[must_use]
    pub const fn preferred(&self) -> KernelTier {
        self.preferred
    }
}

// --------------------------------------------------------------------------------------
// Thermal degradation
// --------------------------------------------------------------------------------------

/// Reports sustained-throughput decline against the opening window.
///
/// Not a failure — a *reporting* obligation. A laptop's first thirty seconds are a turbo window,
/// and quoting that rate as steady-state is how misleading performance numbers get published
/// (plan §9.6 sustained-performance gate). This makes the decline visible so the number can be
/// qualified instead of quietly inflated.
#[derive(Clone, Debug)]
pub struct ThermalReporter {
    baseline: Option<f64>,
    latest: Option<f64>,
    report_below_percent: u32,
}

impl ThermalReporter {
    #[must_use]
    pub const fn new(report_below_percent: u32) -> Self {
        Self {
            baseline: None,
            latest: None,
            report_below_percent,
        }
    }

    /// Observe one window's throughput (any consistent unit; real-time factor is the intended one).
    ///
    /// The first non-zero observation becomes the baseline.
    pub fn observe(&mut self, throughput: f64) -> Option<HealthViolation> {
        if !throughput.is_finite() || throughput <= 0.0 {
            return None;
        }
        self.latest = Some(throughput);
        let baseline = *self.baseline.get_or_insert(throughput);
        if throughput >= baseline {
            return None;
        }
        let percent = ((baseline - throughput) / baseline * 100.0).floor();
        let percent = percent.clamp(0.0, f64::from(u32::MAX)) as u32;
        if percent >= self.report_below_percent {
            return Some(HealthViolation::ThermalDegraded {
                percent_below_baseline: percent,
            });
        }
        None
    }

    #[must_use]
    pub const fn baseline(&self) -> Option<f64> {
        self.baseline
    }

    #[must_use]
    pub const fn latest(&self) -> Option<f64> {
        self.latest
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    #[test]
    fn a_nan_is_located_at_its_first_index() {
        let mut guard = NumericGuard::new(SeamPolicy::All);
        let values = [1.0, 2.0, f32::NAN, f32::NAN];
        let violation = guard
            .check(Seam::TalkerLogits, &values)
            .expect_err("NaN must be caught");
        assert_eq!(
            violation,
            HealthViolation::NonFinite {
                seam: Seam::TalkerLogits,
                index: 2,
                is_nan: true,
            }
        );
        assert!(violation.invalidates_output());
    }

    #[test]
    fn an_infinity_is_distinguished_from_a_nan() {
        let mut guard = NumericGuard::new(SeamPolicy::All);
        let violation = guard
            .check(Seam::CodecOutput, &[f32::INFINITY])
            .expect_err("infinity must be caught");
        assert!(matches!(
            violation,
            HealthViolation::NonFinite { is_nan: false, .. }
        ));
    }

    #[test]
    fn policy_off_never_looks_and_says_so() {
        let mut guard = NumericGuard::new(SeamPolicy::Off);
        assert!(guard.check(Seam::Pcm, &[f32::NAN]).is_ok());
        // The point of is_checking: a run under Off may not be reported as numerically clean.
        assert!(!guard.policy().is_checking());
    }

    #[test]
    fn a_zero_sampling_interval_checks_rather_than_disables() {
        // A misconfigured sampler must not be the reason nothing was inspected.
        let mut guard = NumericGuard::new(SeamPolicy::Sampled { every: 0 });
        assert!(guard.check(Seam::Pcm, &[f32::NAN]).is_err());
    }

    #[test]
    fn sampling_checks_periodically() {
        let mut guard = NumericGuard::new(SeamPolicy::Sampled { every: 3 });
        let bad = [f32::NAN];
        assert!(guard.check(Seam::Pcm, &bad).is_err(), "first call checks");
        assert!(guard.check(Seam::Pcm, &bad).is_ok(), "second is skipped");
        assert!(guard.check(Seam::Pcm, &bad).is_ok(), "third is skipped");
        assert!(guard.check(Seam::Pcm, &bad).is_err(), "fourth checks again");
    }

    #[test]
    fn the_watchdog_fires_only_after_the_timeout() {
        let start = Instant::now();
        let watchdog = ProgressWatchdog::new(Duration::from_millis(500), start);
        assert!(watchdog.check(start + Duration::from_millis(499)).is_ok());
        let violation = watchdog
            .check(start + Duration::from_millis(501))
            .expect_err("a stall past the timeout must fire");
        assert!(matches!(
            violation,
            HealthViolation::NoProgress {
                frames_emitted: 0,
                ..
            }
        ));
    }

    #[test]
    fn recording_a_frame_resets_the_stall_timer() {
        let start = Instant::now();
        let mut watchdog = ProgressWatchdog::new(Duration::from_millis(100), start);
        let later = start + Duration::from_millis(90);
        watchdog.record_frame(later);
        assert!(watchdog.check(later + Duration::from_millis(90)).is_ok());
        assert_eq!(watchdog.frames_emitted(), 1);
    }

    #[test]
    fn end_of_speech_on_the_cap_is_reported_as_inconsistent() {
        // The counterfeit-green case: a cap-stop relabelled as a clean finish.
        let violation = check_stop_consistency(StopReason::EndOfSpeech, 2048, 2048)
            .expect_err("EOS exactly on the cap must be challenged");
        assert!(matches!(
            violation,
            HealthViolation::StopInconsistent { .. }
        ));
        assert!(violation.invalidates_output());
    }

    #[test]
    fn a_cap_stop_short_of_the_cap_is_inconsistent() {
        assert!(check_stop_consistency(StopReason::FrameCapReached, 100, 2048).is_err());
    }

    #[test]
    fn consistent_outcomes_pass() {
        assert!(check_stop_consistency(StopReason::EndOfSpeech, 100, 2048).is_ok());
        assert!(check_stop_consistency(StopReason::FrameCapReached, 2048, 2048).is_ok());
        // Cancellation and a duration limit can legitimately land anywhere.
        assert!(check_stop_consistency(StopReason::Cancelled, 7, 2048).is_ok());
        assert!(check_stop_consistency(StopReason::DurationLimitReached, 7, 2048).is_ok());
    }

    #[test]
    fn a_stuck_token_trips_the_runaway_detector() {
        let mut detector = RunawayDetector::new(4, 8);
        for _ in 0..4 {
            detector.observe(42).expect("within threshold");
        }
        let violation = detector
            .observe(42)
            .expect_err("the fifth repeat must trip");
        assert!(matches!(
            violation,
            HealthViolation::RepetitionRunaway {
                token: 42,
                repeats: 5
            }
        ));
    }

    #[test]
    fn a_two_token_cycle_trips_even_though_nothing_repeats_consecutively() {
        // The case a consecutive-repeat counter alone would miss entirely.
        let mut detector = RunawayDetector::new(100, 3);
        let mut result = Ok(());
        for index in 0..12 {
            result = detector.observe(if index % 2 == 0 { 7 } else { 9 });
            if result.is_err() {
                break;
            }
        }
        assert!(result.is_err(), "a ping-pong cycle must be detected");
    }

    #[test]
    fn ordinary_variety_does_not_trip_the_detector() {
        let mut detector = RunawayDetector::new(4, 3);
        for token in 0..64u32 {
            detector.observe(token).expect("varied tokens are healthy");
        }
    }

    #[test]
    fn silence_past_the_window_is_a_violation() {
        let mut detector = SilenceDetector::new(4, Duration::from_millis(100), 24_000);
        // 24 kHz: 2,400 samples is exactly 100 ms, so 2,401 exceeds it.
        let silent = vec![0i16; 2_401];
        let violation = detector
            .observe(&silent)
            .expect_err("silence past the window must fire");
        assert!(matches!(violation, HealthViolation::OutputSilent { .. }));
    }

    #[test]
    fn any_audible_sample_resets_the_silence_run() {
        let mut detector = SilenceDetector::new(4, Duration::from_millis(100), 24_000);
        detector.observe(&vec![0i16; 2_000]).expect("under window");
        detector.observe(&[9_000]).expect("audible sample resets");
        assert_eq!(detector.silent_millis(), 0);
        detector
            .observe(&vec![0i16; 2_000])
            .expect("run restarted, so still under the window");
    }

    #[test]
    fn demotion_is_one_way_and_reported_once() {
        let mut selector = KernelSelector::new(KernelTier::Optimized("i8mm"));
        assert_eq!(selector.active(), KernelTier::Optimized("i8mm"));

        let violation = selector
            .on_selftest_failure()
            .expect("the first demotion is reported");
        assert_eq!(
            violation,
            HealthViolation::KernelDemoted {
                from: KernelTier::Optimized("i8mm"),
                to: KernelTier::Scalar,
            }
        );
        // Informational: the run stays correct, just slower.
        assert!(!violation.invalidates_output());
        assert_eq!(selector.active(), KernelTier::Scalar);

        // A second failure must not re-report, and must never re-promote: the tier already
        // produced a wrong answer on this machine.
        assert!(selector.on_selftest_failure().is_none());
        assert_eq!(selector.active(), KernelTier::Scalar);
        assert!(selector.demoted());
        assert_eq!(selector.preferred(), KernelTier::Optimized("i8mm"));
    }

    #[test]
    fn thermal_decline_is_reported_against_the_opening_window() {
        let mut reporter = ThermalReporter::new(10);
        assert!(
            reporter.observe(20.0).is_none(),
            "the first sample is the baseline"
        );
        assert!(
            reporter.observe(19.0).is_none(),
            "5% is under the threshold"
        );
        let violation = reporter
            .observe(17.0)
            .expect("15% below baseline must be reported");
        assert_eq!(
            violation,
            HealthViolation::ThermalDegraded {
                percent_below_baseline: 15
            }
        );
        // A slower sustained rate is honest, not broken.
        assert!(!violation.invalidates_output());
        assert_eq!(reporter.baseline(), Some(20.0));
        assert_eq!(reporter.latest(), Some(17.0));
    }

    #[test]
    fn a_nonsense_throughput_sample_is_ignored_rather_than_becoming_the_baseline() {
        let mut reporter = ThermalReporter::new(10);
        assert!(reporter.observe(0.0).is_none());
        assert!(reporter.observe(f64::NAN).is_none());
        assert_eq!(reporter.baseline(), None, "no baseline was established");
        assert!(reporter.observe(10.0).is_none());
        assert_eq!(reporter.baseline(), Some(10.0));
    }

    #[test]
    fn every_violation_carries_a_remedy_and_a_wire_name() {
        let violations = [
            HealthViolation::NonFinite {
                seam: Seam::Pcm,
                index: 0,
                is_nan: true,
            },
            HealthViolation::NoProgress {
                frames_emitted: 1,
                stalled_millis: 2,
            },
            HealthViolation::StopInconsistent {
                claimed: StopReason::EndOfSpeech,
                frames_emitted: 1,
                frame_cap: 1,
            },
            HealthViolation::RepetitionRunaway {
                token: 1,
                repeats: 2,
            },
            HealthViolation::OutputSilent { silent_millis: 1 },
            HealthViolation::KernelDemoted {
                from: KernelTier::Optimized("i8mm"),
                to: KernelTier::Scalar,
            },
            HealthViolation::ThermalDegraded {
                percent_below_baseline: 11,
            },
            HealthViolation::SpeculationDemoted {
                e_value_x100: 10000,
                steps_observed: 5,
            },
        ];
        for violation in violations {
            assert!(!violation.as_str().is_empty());
            assert!(
                violation.remedy().len() > 40,
                "{}: a remedy must tell the caller what to do",
                violation.as_str()
            );
            assert!(!violation.to_string().is_empty());
        }
    }
}
