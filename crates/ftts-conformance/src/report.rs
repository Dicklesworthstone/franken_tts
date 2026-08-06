//! Structured test receipts — one JSON object per event, on stdout.
//!
//! Two jobs. First, carry the honest skip signal: `libtest` reports an early `return` as a pass, so
//! `outcome` here is the only place `skipped` and `passed` stay distinguishable, and the ladder
//! runner aggregates these receipts instead of `libtest`'s counts. Second, make a red run
//! self-localizing — provenance, seam, tolerance and its source travel with the verdict, so nobody
//! reruns under a debugger to find out *which* fixture and *which* tolerance were in play.

use std::{
    env,
    fs::OpenOptions,
    io::Write,
    time::{Duration, Instant},
};

use serde_json::{Value, json};

/// Environment variable naming a file that receives every emitted event as NDJSON.
///
/// `libtest` captures stdout and prints it only for failing tests, so the receipt stream that is
/// supposed to distinguish `skipped` from `passed` is invisible on a green run — precisely when the
/// distinction matters. Pointing this at a file gives CI (and the ladder runner) the full stream
/// regardless of capture. Unset means stdout only.
pub const RECEIPTS_ENV: &str = "FTTS_RECEIPTS";

/// Emits one NDJSON line to stdout and, when [`RECEIPTS_ENV`] is set, appends it to that file.
///
/// # Panics
///
/// Panics when the sink is configured but unwritable. Dropping receipts silently would recreate the
/// exact failure this module exists to prevent: a green-looking run whose evidence never existed.
fn emit_line(line: &str) {
    println!("{line}");
    let Ok(path) = env::var(RECEIPTS_ENV) else {
        return;
    };
    if path.trim().is_empty() {
        return;
    }
    append_event_line(&path, line).unwrap_or_else(|error| {
        panic!(
            "{RECEIPTS_ENV}={path} is set but the receipt sink is unwritable ({error}); \
             receipts would be lost silently, which is how a run with no evidence reads as green"
        )
    });
}

/// Appends one NDJSON line to the receipt sink at `path`, creating it if needed.
///
/// Split out from the environment lookup so the file behavior is directly testable: edition 2024
/// makes `env::set_var` `unsafe`, which this crate forbids, so no test can install the variable
/// in-process.
///
/// The line and its terminator go out in a single `write_all`, so concurrent `libtest` threads
/// appending to the same `O_APPEND` file cannot interleave a partial record.
///
/// # Errors
///
/// Returns the underlying I/O error when the sink cannot be opened or written.
pub fn append_event_line(path: &str, line: &str) -> std::io::Result<()> {
    let mut record = String::with_capacity(line.len() + 1);
    record.push_str(line);
    record.push('\n');
    OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?
        .write_all(record.as_bytes())
}

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
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CaptureProvenance {
    /// Claim tier emitted by the fixture generator, e.g. `native_cuda` or `cpu_fp32_fallback`.
    pub oracle_class: String,
    /// Device on which the oracle actually executed, e.g. `cpu` or `cuda:0`.
    pub device: String,
    /// Floating-point dtype used for the capture, e.g. `float32` or `bfloat16`.
    pub dtype: String,
}

impl CaptureProvenance {
    /// Creates an all-or-nothing capture identity so a receipt cannot omit its precision or device.
    #[must_use]
    pub fn new(
        oracle_class: impl Into<String>,
        device: impl Into<String>,
        dtype: impl Into<String>,
    ) -> Self {
        Self {
            oracle_class: oracle_class.into(),
            device: device.into(),
            dtype: dtype.into(),
        }
    }

    fn to_json(&self) -> Value {
        json!({
            "oracle_class": self.oracle_class,
            "device": self.device,
            "dtype": self.dtype,
        })
    }
}

#[derive(Clone, Debug, Default)]
pub struct FixtureProvenance {
    /// Path or identifier of the oracle dump.
    pub source: String,
    /// SHA-256 of the fixture bytes, when one is known.
    pub sha256: Option<String>,
    /// Upstream revision the fixture was generated against.
    pub revision: Option<String>,
    /// Device/precision claim made by the fixture producer, when supplied by its manifest.
    pub capture: Option<CaptureProvenance>,
}

impl FixtureProvenance {
    /// Creates provenance naming only the source.
    #[must_use]
    pub fn new(source: impl Into<String>) -> Self {
        Self {
            source: source.into(),
            sha256: None,
            revision: None,
            capture: None,
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

    /// Attaches the oracle class, actual device, and capture precision from fixture provenance.
    #[must_use]
    pub fn with_capture_provenance(
        mut self,
        oracle_class: impl Into<String>,
        device: impl Into<String>,
        dtype: impl Into<String>,
    ) -> Self {
        self.capture = Some(CaptureProvenance::new(oracle_class, device, dtype));
        self
    }

    fn to_json(&self) -> Value {
        json!({
            "source": self.source,
            "sha256": self.sha256,
            "revision": self.revision,
            "capture": self.capture.as_ref().map(CaptureProvenance::to_json),
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

    /// Emits the receipt as one NDJSON line — to stdout, and to [`RECEIPTS_ENV`] when it is set.
    ///
    /// Panics when the receipt is a skip without a reason: an unexplained skip is indistinguishable
    /// from a silently disabled test, which is the failure mode this whole module exists to prevent.
    pub fn emit(&self) {
        assert!(
            !(self.outcome == Outcome::Skipped && self.reason.is_none()),
            "skip receipt for `{}` has no reason; an unexplained skip is a disabled test",
            self.test
        );
        emit_line(&self.to_line());
    }
}

/// A running stage, timed from construction.
///
/// Convention (c): an end-to-end run logs every stage transition with wall-clock and the key
/// intermediate hashes, so a red run names its stage immediately instead of needing a bisect.
///
/// ```
/// use ftts_conformance::report::{Stage, token_stream_hash};
///
/// let stage = Stage::start("talker_decode");
/// let tokens = vec![1_u32, 2, 3];
/// stage.finish(&[("token_stream", &token_stream_hash(&tokens))]);
/// ```
#[derive(Debug)]
pub struct Stage {
    name: String,
    started: Instant,
}

impl Stage {
    /// Starts timing a named stage.
    #[must_use]
    pub fn start(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            started: Instant::now(),
        }
    }

    /// Emits the stage event with its elapsed wall-clock and the supplied intermediate hashes.
    pub fn finish(self, hashes: &[(&str, &str)]) {
        emit_stage(&self.name, self.started.elapsed(), hashes);
    }
}

/// Emits a stage-transition event for multi-stage runs.
///
/// Prefer [`Stage`], which cannot report a wall-clock that drifted from the work it measured.
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
    emit_line(&event.to_string());
}

// ── Intermediate hashes ──────────────────────────────────────────────────────────────────────
//
// FNV-1a/64. These identify *which* stage first diverged between two runs; they are deliberately
// NOT cryptographic and must never be used for artifact provenance (`FixtureProvenance::sha256`
// carries a real digest supplied by the fixture generator). The `fnv1a64:` prefix is part of the
// value so a reader can never mistake one for the other.

const FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// Stable non-cryptographic hash of raw bytes, rendered as `fnv1a64:<hex>`.
#[must_use]
pub fn bytes_hash(bytes: &[u8]) -> String {
    let mut hash = FNV_OFFSET_BASIS;
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    format!("fnv1a64:{hash:016x}")
}

/// Stable hash of a codec/talker token stream.
///
/// Hashes the little-endian encoding, so a stream that differs only in element *order* still
/// differs in hash — reordering is a real divergence, not an equivalent encoding.
#[must_use]
pub fn token_stream_hash(tokens: &[u32]) -> String {
    let mut bytes = Vec::with_capacity(tokens.len() * 4);
    for token in tokens {
        bytes.extend_from_slice(&token.to_le_bytes());
    }
    bytes_hash(&bytes)
}

/// Stable hash of a PCM buffer, over raw sample bits.
///
/// Bit-level on purpose: `-0.0`, `+0.0`, and each NaN payload hash differently, because at this
/// seam those are divergences worth seeing rather than values worth normalizing away.
#[must_use]
pub fn pcm_hash(samples: &[f32]) -> String {
    let mut bytes = Vec::with_capacity(samples.len() * 4);
    for sample in samples {
        bytes.extend_from_slice(&sample.to_bits().to_le_bytes());
    }
    bytes_hash(&bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

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
                    .with_revision("5d839924")
                    .with_capture_provenance("cpu_fp32_fallback", "cpu", "float32"),
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
            value["fixture"]["capture"]["oracle_class"],
            "cpu_fp32_fallback"
        );
        assert_eq!(value["fixture"]["capture"]["device"], "cpu");
        assert_eq!(value["fixture"]["capture"]["dtype"], "float32");
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

    #[test]
    fn the_receipt_sink_appends_one_parseable_line_per_event() {
        let path = env::temp_dir().join(format!("ftts-receipts-{}.ndjson", std::process::id()));
        let path = path.to_str().expect("temp path is UTF-8").to_owned();
        let _ = fs::remove_file(&path);

        let first = Receipt::new("contract_a_l0_tokenizer_ids", Outcome::Passed).to_line();
        let second = Receipt::new("contract_a_l4_codec_tokens", Outcome::Skipped)
            .reason("model artifact unavailable")
            .to_line();
        append_event_line(&path, &first).expect("sink is writable");
        append_event_line(&path, &second).expect("sink appends rather than truncating");

        let contents = fs::read_to_string(&path).expect("sink is readable");
        let lines: Vec<&str> = contents.lines().collect();
        assert_eq!(lines.len(), 2, "append must not truncate: {contents}");
        for line in &lines {
            let parsed: Value = serde_json::from_str(line).expect("every line is one JSON object");
            assert_eq!(parsed["event"], "contract_check");
        }
        // The second receipt keeps its honest verdict all the way to the aggregated file.
        let last: Value = serde_json::from_str(lines[1]).expect("parses");
        assert_eq!(last["outcome"], "skipped");

        fs::remove_file(&path).expect("temp sink is removable");
    }

    #[test]
    fn an_unwritable_configured_sink_is_a_loud_failure_not_a_lost_receipt() {
        let error = append_event_line(
            "/nonexistent/ftts-receipts-directory/receipts.ndjson",
            "{\"event\":\"contract_check\"}",
        )
        .expect_err("writing into a nonexistent directory must fail");
        // `emit_line` turns exactly this error into a panic; the point is that it is never Ok(()).
        assert!(!error.to_string().is_empty());
    }

    #[test]
    fn intermediate_hashes_are_stable_labelled_and_order_sensitive() {
        assert_eq!(token_stream_hash(&[1, 2, 3]), token_stream_hash(&[1, 2, 3]));
        assert_ne!(
            token_stream_hash(&[1, 2, 3]),
            token_stream_hash(&[1, 3, 2]),
            "reordering a token stream is a divergence, not an equivalent encoding"
        );
        assert_ne!(
            token_stream_hash(&[1, 2, 3]),
            token_stream_hash(&[1, 2, 3, 0]),
            "a trailing element must change the hash"
        );
        assert_ne!(
            pcm_hash(&[0.0]),
            pcm_hash(&[-0.0]),
            "bit-level hashing so a sign-of-zero divergence stays visible"
        );
        // The algorithm is named in the value so nobody reads it as a content digest.
        assert!(bytes_hash(b"abc").starts_with("fnv1a64:"));
        assert_eq!(bytes_hash(b"").len(), "fnv1a64:".len() + 16);
    }
}
