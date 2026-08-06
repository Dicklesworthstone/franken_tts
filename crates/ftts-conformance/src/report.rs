//! Structured test receipts — one JSON object per event, on stdout.
//!
//! Two jobs. First, carry the honest skip signal: `libtest` reports an early `return` as a pass, so
//! `outcome` here is the only place `skipped` and `passed` stay distinguishable, and the ladder
//! runner aggregates these receipts instead of `libtest`'s counts. Second, make a red run
//! self-localizing — provenance, seam, tolerance and its source travel with the verdict, so nobody
//! reruns under a debugger to find out *which* fixture and *which* tolerance were in play.

use std::time::Duration;

use serde_json::{Value, json};

/// The verdict for one contract check. `Skipped` is never folded into `Passed`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Outcome {
    /// The check ran and held.
    Passed,
    /// The check ran and did not hold.
    Failed,
    /// The check did not run. **Not** a pass.
    Skipped,
    /// A known, ledgered divergence that is expected to fail.
    ///
    /// Always `XFAIL`, never `SKIP`: a known-bad result must keep being executed so its
    /// unexpected *success* is also detectable.
    ExpectedFailure,
}

impl Outcome {
    /// The stable wire string used in receipts.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Passed => "passed",
            Self::Failed => "failed",
            Self::Skipped => "skipped",
            Self::ExpectedFailure => "xfail",
        }
    }
}

/// Where the expected values came from.
///
/// A comparison whose fixture provenance is unknown cannot be debugged: "actual != expected" is
/// useless without knowing which oracle dump produced `expected` and whether it is still current.
#[derive(Clone, Debug, Default)]
pub struct FixtureProvenance {
    /// Path or identifier of the oracle dump.
    pub source: String,
    /// SHA-256 of the fixture bytes, when one is known.
    pub sha256: Option<String>,
    /// Upstream revision the fixture was generated against.
    pub revision: Option<String>,
}

impl FixtureProvenance {
    /// Creates provenance naming only the source.
    #[must_use]
    pub fn new(source: impl Into<String>) -> Self {
        Self {
            source: source.into(),
            sha256: None,
            revision: None,
        }
    }

    /// Attaches the fixture content hash.
    #[must_use]
    pub fn with_sha256(mut self, sha256: impl Into<String>) -> Self {
        self.sha256 = Some(sha256.into());
        self
    }

    /// Attaches the upstream revision the fixture came from.
    #[must_use]
    pub fn with_revision(mut self, revision: impl Into<String>) -> Self {
        self.revision = Some(revision.into());
        self
    }

    fn to_json(&self) -> Value {
        json!({
            "source": self.source,
            "sha256": self.sha256,
            "revision": self.revision,
        })
    }
}

/// One receipt for one contract check.
///
/// Build it with the setters, then [`Receipt::emit`]. Every field is optional except the test name
/// and outcome, so a cheap check stays cheap while a ladder rung can carry full provenance.
#[derive(Clone, Debug)]
pub struct Receipt {
    test: String,
    outcome: Outcome,
    contract: Option<String>,
    seam: Option<String>,
    reason: Option<String>,
    provenance: Option<FixtureProvenance>,
    tolerance: Option<f64>,
    tolerance_source: Option<String>,
    seed: Option<u64>,
    elapsed: Option<Duration>,
    detail: Option<Value>,
}

impl Receipt {
    /// Starts a receipt for a named test.
    ///
    /// `test` should encode the contract being tested — `contract_a_l2_talker_layer17_cosine`,
    /// not `test_talker_3`.
    #[must_use]
    pub fn new(test: impl Into<String>, outcome: Outcome) -> Self {
        Self {
            test: test.into(),
            outcome,
            contract: None,
            seam: None,
            reason: None,
            provenance: None,
            tolerance: None,
            tolerance_source: None,
            seed: None,
            elapsed: None,
            detail: None,
        }
    }

    /// Names the conformance contract, e.g. `ConformanceExact/L2`.
    #[must_use]
    pub fn contract(mut self, contract: impl Into<String>) -> Self {
        self.contract = Some(contract.into());
        self
    }

    /// Names the seam under test, e.g. `talker.layer17.attn_out`.
    #[must_use]
    pub fn seam(mut self, seam: impl Into<String>) -> Self {
        self.seam = Some(seam.into());
        self
    }

    /// Explains a skip or a failure. **Required** for [`Outcome::Skipped`].
    #[must_use]
    pub fn reason(mut self, reason: impl Into<String>) -> Self {
        self.reason = Some(reason.into());
        self
    }

    /// Records which oracle dump supplied the expected values.
    #[must_use]
    pub fn provenance(mut self, provenance: FixtureProvenance) -> Self {
        self.provenance = Some(provenance);
        self
    }

    /// Records the tolerance used and, crucially, where that number came from.
    ///
    /// A tolerance without a source is how epsilon creep starts: state the artifact that justifies
    /// it (the oracle's measured nondeterminism floor), not a hand-picked constant.
    #[must_use]
    pub fn tolerance(mut self, tolerance: f64, source: impl Into<String>) -> Self {
        self.tolerance = Some(tolerance);
        self.tolerance_source = Some(source.into());
        self
    }

    /// Records the RNG seed, so a stochastic failure is reproducible.
    #[must_use]
    pub const fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Records wall-clock time for the check.
    #[must_use]
    pub const fn elapsed(mut self, elapsed: Duration) -> Self {
        self.elapsed = Some(elapsed);
        self
    }

    /// Attaches arbitrary structured detail, such as a comparator summary.
    #[must_use]
    pub fn detail(mut self, detail: Value) -> Self {
        self.detail = Some(detail);
        self
    }

    /// Renders the receipt as a single JSON object.
    #[must_use]
    pub fn to_json(&self) -> Value {
        json!({
            "event": "contract_check",
            "test": self.test,
            "outcome": self.outcome.as_str(),
            "contract": self.contract,
            "seam": self.seam,
            "reason": self.reason,
            "fixture": self.provenance.as_ref().map(FixtureProvenance::to_json),
            "tolerance": self.tolerance,
            "tolerance_source": self.tolerance_source,
            "seed": self.seed,
            "elapsed_ms": self.elapsed.map(|d| d.as_secs_f64() * 1000.0),
            "detail": self.detail,
        })
    }

    /// Renders the receipt as one line of NDJSON.
    #[must_use]
    pub fn to_line(&self) -> String {
        self.to_json().to_string()
    }

    /// Prints the receipt to stdout as one NDJSON line.
    ///
    /// Panics when the receipt is a skip without a reason: an unexplained skip is indistinguishable
    /// from a silently disabled test, which is the failure mode this whole module exists to prevent.
    pub fn emit(&self) {
        assert!(
            !(self.outcome == Outcome::Skipped && self.reason.is_none()),
            "skip receipt for `{}` has no reason; an unexplained skip is a disabled test",
            self.test
        );
        println!("{}", self.to_line());
    }
}

/// Emits a stage-transition event for multi-stage runs.
///
/// End-to-end scripts log every transition with wall-clock and key intermediate hashes (token
/// stream, PCM) so a red run names its stage immediately instead of requiring a bisect.
pub fn emit_stage(stage: &str, elapsed: Duration, hashes: &[(&str, &str)]) {
    let hash_map: serde_json::Map<String, Value> = hashes
        .iter()
        .map(|(name, value)| ((*name).to_owned(), Value::String((*value).to_owned())))
        .collect();
    let event = json!({
        "event": "stage",
        "stage": stage,
        "elapsed_ms": elapsed.as_secs_f64() * 1000.0,
        "hashes": Value::Object(hash_map),
    });
    println!("{event}");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn skipped_is_never_reported_as_passed() {
        assert_eq!(Outcome::Skipped.as_str(), "skipped");
        assert_ne!(Outcome::Skipped.as_str(), Outcome::Passed.as_str());
        // XFAIL must remain its own state so an unexpected pass is still visible.
        assert_eq!(Outcome::ExpectedFailure.as_str(), "xfail");
        assert_ne!(Outcome::ExpectedFailure.as_str(), Outcome::Skipped.as_str());
    }

    #[test]
    fn receipt_carries_provenance_tolerance_and_its_source() {
        let receipt = Receipt::new("contract_a_l2_talker_layer17_cosine", Outcome::Passed)
            .contract("ConformanceExact/L2")
            .seam("talker.layer17.attn_out")
            .provenance(
                FixtureProvenance::new("fixtures/talker_l17.npz")
                    .with_sha256("abc123")
                    .with_revision("5d839924"),
            )
            .tolerance(1.5e-3, "docs/truth-pack/nondeterminism-floor.json")
            .seed(42)
            .elapsed(Duration::from_millis(12));

        let value = receipt.to_json();
        assert_eq!(value["outcome"], "passed");
        assert_eq!(value["seam"], "talker.layer17.attn_out");
        assert_eq!(value["fixture"]["sha256"], "abc123");
        assert_eq!(value["fixture"]["revision"], "5d839924");
        assert_eq!(
            value["tolerance_source"],
            "docs/truth-pack/nondeterminism-floor.json"
        );
        assert_eq!(value["seed"], 42);
        // One JSON object per line keeps CI aggregation trivial.
        assert!(!receipt.to_line().contains('\n'));
    }

    #[test]
    #[should_panic(expected = "an unexplained skip is a disabled test")]
    fn a_skip_without_a_reason_is_rejected() {
        Receipt::new("contract_a_l0_tokenizer_ids", Outcome::Skipped).emit();
    }
}
