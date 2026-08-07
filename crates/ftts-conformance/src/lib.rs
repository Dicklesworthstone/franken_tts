#![forbid(unsafe_code)]

//! Development-only home for integration, conformance, and benchmark suites.
//!
//! # THE TEST OBSERVABILITY CONVENTION
//!
//! **This is the single authoritative definition. Every bead's "unit tests" clause inherits it, so
//! no other bead restates it.** The promise behind the project is that after implementation we can
//! be *sure* things work — which requires failures to be **informative**, not merely detected. A
//! test that says `assertion failed: left != right` has detected a problem and told you nothing.
//!
//! ## (a) Every test emits structured, detailed logs
//!
//! A failing test must localize the problem **without a debugger rerun**. Carry: assertion context
//! (expected vs actual, with shapes/dtypes/indices), the seam under test, fixture provenance (which
//! oracle dump, which hash, which revision), RNG seeds, the tolerance used **and its source**, and
//! timing. Use [`report::Receipt`].
//!
//! ## (b) Comparator failures name the first divergent element
//!
//! Never a bare assertion. Reach for [`assert_close!`] (numeric, tolerance-bounded) or
//! [`assert_exact!`] (token streams); both emit their own receipt and, on failure, panic with the
//! self-localizing report rather than `left != right`.
//!
//! [`compare::compare_f32`] plus [`compare::describe_failure`] report the first divergent element
//! with row-major coordinates, then max-abs, max-rel, cosine, and count-over-tolerance — the
//! statistics that separate a SIMD lane bug from a transposed tensor. For exact sequences,
//! [`compare::compare_exact`] plus [`compare::describe_exact_failure`] report the index where the
//! streams parted, the surrounding context, and whether the tail realigns under a one-element
//! [`compare::Shift`] — the difference between a dropped step and a wrong value.
//!
//! ## (c) End-to-end runs log every stage transition
//!
//! Wall-clock plus key intermediate hashes via [`report::Stage`], so a red run names its stage in
//! seconds instead of requiring a bisect. [`report::token_stream_hash`] and [`report::pcm_hash`]
//! supply the hashes, so no e2e author hand-rolls a digest.
//!
//! ## (d) Logs are machine-parseable
//!
//! One JSON object per event, one per line, on stdout, so CI can aggregate failure patterns across
//! runs. [`report::Receipt::to_line`] never emits an embedded newline.
//!
//! `libtest` captures stdout and reveals it only for failing tests, so on a green run the receipt
//! stream is invisible — exactly when `skipped`-vs-`passed` needs auditing. Set
//! [`report::RECEIPTS_ENV`] (`FTTS_RECEIPTS=path.ndjson`) and every event is appended there too:
//!
//! ```console
//! $ FTTS_RECEIPTS=target/receipts.ndjson cargo test -p ftts-conformance
//! $ jq -r 'select(.outcome=="skipped") | .test + "\t" + .reason' target/receipts.ndjson
//! ```
//!
//! `scripts/check.sh` does exactly that on every run and folds any skip it finds into the closing
//! banner, so a suite whose model-gated rungs never executed reports `GREEN WITH SKIPS` rather than
//! green. `scripts/summarize_receipts.py` is the reader; without it this stream would be decoration.
//!
//! ## The `Demo/` contract namespace is reserved
//!
//! Receipts whose `contract` starts with `Demo/` come from the convention demo and the doc examples
//! below — they emit skips *on purpose*, to show the mechanism. The aggregator counts them
//! separately and never treats them as gate signal, because a banner that is permanently yellow for
//! staged reasons teaches readers to ignore it. Every honesty rule still applies to them in full: a
//! `Demo/` skip without a reason fails the gate like any other. **Production ladders never use this
//! namespace** — a real rung that skipped must be visible.
//!
//! ## (e) Test names encode the contract
//!
//! `contract_a_l2_talker_layer17_cosine`, never `test_talker_3`. The name should survive being read
//! alone in a CI summary. The macros below fill the receipt's `test` field from the enclosing
//! function via [`test_name!`], so a receipt cannot drift from the test that emitted it.
//!
//! # THE MODEL GATE
//!
//! Tests needing multi-GB weights must keep CI green without them **without ever counting a skip as
//! a pass**. See [`gate`] for the mechanism and [`report::Outcome`] for why `Skipped` and
//! `ExpectedFailure` are distinct states that never collapse into `Passed`.
//!
//! Three states, all demonstrated in `tests/model_gate_demo.rs`:
//!
//! | State | Condition | Behavior |
//! |---|---|---|
//! | green-with-model | artifact present | run for real; **prove the native path ran** |
//! | skip-without-model | artifact absent | emit `outcome: "skipped"` **with a reason**; assert nothing |
//! | loud-failure | fallback ran under an open gate | fail with the seam named |
//!
//! Known divergences are **XFAIL, never SKIP**: a ledgered known-bad result must keep executing so
//! that its unexpected *success* is also detectable. Use [`xfail`], not an early return.
//!
//! # Usage — the convention is one import away
//!
//! ```
//! use ftts_conformance::{assert_close, prelude::*, require_model};
//!
//! # fn body() {
//! // Skips honestly (with a reason) and returns when the weights are absent.
//! let artifact = require_model!("ConformanceExact/L2");
//!
//! let expected = load_oracle_dump(&artifact);
//! let actual = run_native_talker_layer(&artifact);
//!
//! // Emits its own receipt either way; on failure, panics with a self-localizing report.
//! assert_close!(
//!     seam = "talker.layer17.attn_out",
//!     expected = &expected,
//!     actual = &actual,
//!     tolerance = 1.5e-3,
//!     source = "docs/truth-pack/nondeterminism-floor.json",
//!     shape = &[2, 3],
//! );
//! # }
//! # fn load_oracle_dump(_: &std::path::Path) -> Vec<f32> { vec![0.0; 6] }
//! # fn run_native_talker_layer(_: &std::path::Path) -> Vec<f32> { vec![0.0; 6] }
//! ```

pub mod compare;
pub mod gate;
pub mod ladder;
pub mod npy;
pub mod oracle;
pub mod report;

/// The imports a conformance test almost always wants.
///
/// The macros ([`require_model!`], [`assert_close!`], [`assert_exact!`], [`test_name!`]) are
/// exported at the crate root by `#[macro_export]`, so import them from there.
pub mod prelude {
    pub use crate::compare::{
        Comparison, ExactComparison, Shift, compare_exact, compare_f32, coordinates,
        describe_exact_failure, describe_failure,
    };
    pub use crate::gate::{
        ExecutionPath, ModelGate, NONEXISTENT_FALLBACK, require_native_execution,
    };
    pub use crate::report::{
        CaptureProvenance, FixtureProvenance, OracleTier, Outcome, Receipt, Stage, bytes_hash,
        emit_stage, pcm_hash, token_stream_hash,
    };
    pub use crate::{gated, xfail};
}

/// The path of the enclosing function, as a `&'static str`.
///
/// Lets a receipt name itself after the test that emitted it, so convention (e)'s contract-encoding
/// test name reaches CI without being retyped as a string literal that can silently drift.
///
/// ```
/// use ftts_conformance::test_name;
///
/// fn contract_a_l0_tokenizer_ids() -> &'static str {
///     test_name!()
/// }
/// assert!(contract_a_l0_tokenizer_ids().ends_with("contract_a_l0_tokenizer_ids"));
/// ```
#[macro_export]
macro_rules! test_name {
    () => {{
        fn probe() {}
        fn path_of<T>(_: T) -> &'static str {
            ::core::any::type_name::<T>()
        }
        let path = path_of(probe);
        path.strip_suffix("::probe").unwrap_or(path)
    }};
}

/// Resolves the model gate, or emits an honest skip receipt and returns from the enclosing test.
///
/// Evaluates to the [`std::path::PathBuf`] of the artifact when the gate is open. This is the
/// [`gated`] pattern without the closure, for tests that want `?`-free straight-line bodies.
///
/// ```
/// use ftts_conformance::require_model;
///
/// # fn body() {
/// let artifact = require_model!("ConformanceExact/L4");
/// assert!(artifact.exists());
/// # }
/// ```
#[macro_export]
macro_rules! require_model {
    ($contract:expr $(,)?) => {
        match $crate::gate::ModelGate::resolve() {
            $crate::gate::ModelGate::Present { artifact } => artifact,
            $crate::gate::ModelGate::Absent { reason } => {
                $crate::report::Receipt::new(
                    $crate::test_name!(),
                    $crate::report::Outcome::Skipped,
                )
                .contract($contract)
                .reason(reason)
                .emit();
                return;
            }
        }
    };
}

/// Asserts two `f32` slices agree within tolerance, emitting a receipt and localizing any failure.
///
/// Named arguments. `contract` and `shape` are optional; supply `contract` on any check that is a
/// ladder rung, since that is how the aggregator attributes the receipt.
///
/// ```
/// use ftts_conformance::assert_close;
///
/// let expected = [0.1_f32, 0.2, 0.3, 0.4, 0.5, 0.6];
/// let actual = [0.1_f32, 0.2, 0.3, 0.4, 0.5, 0.6];
/// assert_close!(
///     contract = "Demo/Observability",
///     seam = "codec.upsample.stage0",
///     expected = &expected,
///     actual = &actual,
///     tolerance = 1e-4,
///     source = "docs/truth-pack/nondeterminism-floor.json",
///     shape = &[2, 3],
/// );
/// ```
///
/// The receipt is emitted **before** the panic, so a failing run leaves a `failed` record in the
/// aggregated stream rather than a hole where a receipt should have been.
#[macro_export]
macro_rules! assert_close {
    (
        seam = $seam:expr,
        expected = $expected:expr,
        actual = $actual:expr,
        tolerance = $tolerance:expr,
        source = $source:expr $(,)?
    ) => {
        $crate::assert_close!(
            contract = ::core::option::Option::<&str>::None,
            seam = $seam,
            expected = $expected,
            actual = $actual,
            tolerance = $tolerance,
            source = $source,
        )
    };
    (
        seam = $seam:expr,
        expected = $expected:expr,
        actual = $actual:expr,
        tolerance = $tolerance:expr,
        source = $source:expr,
        shape = $shape:expr $(,)?
    ) => {
        $crate::assert_close!(
            contract = ::core::option::Option::<&str>::None,
            seam = $seam,
            expected = $expected,
            actual = $actual,
            tolerance = $tolerance,
            source = $source,
            shape = $shape,
        )
    };
    (
        contract = $contract:expr,
        seam = $seam:expr,
        expected = $expected:expr,
        actual = $actual:expr,
        tolerance = $tolerance:expr,
        source = $source:expr $(,)?
    ) => {
        $crate::macro_support::assert_close($crate::macro_support::CloseAssertion {
            test: $crate::test_name!(),
            contract: $crate::macro_support::IntoContract::into_contract($contract),
            seam: $seam,
            expected: $expected,
            actual: $actual,
            tolerance: $tolerance,
            tolerance_source: $source,
            shape: ::core::option::Option::None,
        })
    };
    (
        contract = $contract:expr,
        seam = $seam:expr,
        expected = $expected:expr,
        actual = $actual:expr,
        tolerance = $tolerance:expr,
        source = $source:expr,
        shape = $shape:expr $(,)?
    ) => {
        $crate::macro_support::assert_close($crate::macro_support::CloseAssertion {
            test: $crate::test_name!(),
            contract: $crate::macro_support::IntoContract::into_contract($contract),
            seam: $seam,
            expected: $expected,
            actual: $actual,
            tolerance: $tolerance,
            tolerance_source: $source,
            shape: ::core::option::Option::Some($shape),
        })
    };
}

/// Asserts two sequences are exactly equal — the token-stream counterpart of [`assert_close!`].
///
/// ```
/// use ftts_conformance::assert_exact;
///
/// let expected = [11_u32, 22, 33];
/// let actual = [11_u32, 22, 33];
/// assert_exact!(
///     contract = "Demo/Observability",
///     seam = "tokenizer.encode",
///     expected = &expected,
///     actual = &actual,
/// );
/// ```
#[macro_export]
macro_rules! assert_exact {
    (
        seam = $seam:expr,
        expected = $expected:expr,
        actual = $actual:expr $(,)?
    ) => {
        $crate::assert_exact!(
            contract = ::core::option::Option::<&str>::None,
            seam = $seam,
            expected = $expected,
            actual = $actual,
        )
    };
    (
        contract = $contract:expr,
        seam = $seam:expr,
        expected = $expected:expr,
        actual = $actual:expr $(,)?
    ) => {
        $crate::macro_support::assert_exact(
            $crate::test_name!(),
            $crate::macro_support::IntoContract::into_contract($contract),
            $seam,
            $expected,
            $actual,
        )
    };
}

/// Implementation targets for the macros. Not a stable surface — call the macros.
#[doc(hidden)]
pub mod macro_support {
    use crate::{
        compare::{
            Comparison, ExactComparison, compare_exact, compare_f32, describe_exact_failure,
            describe_failure,
        },
        report::{Outcome, Receipt},
    };

    /// Lets the macros' `contract` argument accept either a `&str` or an already-optional value,
    /// so the no-contract arms can forward `None` through the same call shape.
    pub trait IntoContract {
        /// Normalizes to an optional contract name.
        fn into_contract(self) -> Option<&'static str>;
    }

    impl IntoContract for &'static str {
        fn into_contract(self) -> Option<&'static str> {
            Some(self)
        }
    }

    impl IntoContract for Option<&'static str> {
        fn into_contract(self) -> Option<&'static str> {
            self
        }
    }

    /// Applies an optional contract name to a receipt.
    fn with_contract(receipt: Receipt, contract: Option<&str>) -> Receipt {
        match contract {
            Some(name) => receipt.contract(name),
            None => receipt,
        }
    }

    /// Everything [`assert_close`] needs, as named fields.
    ///
    /// A struct rather than a parameter list: the macro fills all eight, and eight positional
    /// arguments of which three are `&str` is a call site where a transposed pair (seam vs
    /// tolerance source) compiles cleanly and lies in the receipt.
    pub struct CloseAssertion<'a> {
        /// The test emitting the receipt, from [`crate::test_name!`].
        pub test: &'a str,
        /// The ladder rung this check belongs to, when it is one.
        pub contract: Option<&'a str>,
        /// The seam under test, e.g. `talker.layer17.attn_out`.
        pub seam: &'a str,
        /// Reference values, from the oracle dump.
        pub expected: &'a [f32],
        /// Values produced by the implementation under test.
        pub actual: &'a [f32],
        /// The absolute tolerance applied elementwise.
        pub tolerance: f64,
        /// The artifact that justifies `tolerance` — never a hand-picked constant.
        pub tolerance_source: &'a str,
        /// Row-major shape, for reporting the first divergence as coordinates.
        pub shape: Option<&'a [usize]>,
    }

    /// Backs [`crate::assert_close!`].
    ///
    /// # Panics
    ///
    /// Panics with the self-localizing report when the comparison does not hold.
    pub fn assert_close(assertion: CloseAssertion<'_>) -> Comparison {
        let CloseAssertion {
            test,
            contract,
            seam,
            expected,
            actual,
            tolerance,
            tolerance_source,
            shape,
        } = assertion;
        let comparison = compare_f32(expected, actual, tolerance);
        let outcome = if comparison.holds() {
            Outcome::Passed
        } else {
            Outcome::Failed
        };
        with_contract(Receipt::new(test, outcome), contract)
            .seam(seam)
            .tolerance(tolerance, tolerance_source)
            .detail(comparison.to_json())
            .emit();
        assert!(
            comparison.holds(),
            "{}",
            describe_failure(seam, &comparison, tolerance, tolerance_source, shape)
        );
        comparison
    }

    /// Backs [`crate::assert_exact!`].
    ///
    /// # Panics
    ///
    /// Panics with the self-localizing report when the sequences are not identical.
    pub fn assert_exact<T>(
        test: &str,
        contract: Option<&str>,
        seam: &str,
        expected: &[T],
        actual: &[T],
    ) -> ExactComparison
    where
        T: PartialEq + std::fmt::Debug,
    {
        let comparison = compare_exact(expected, actual);
        let outcome = if comparison.holds() {
            Outcome::Passed
        } else {
            Outcome::Failed
        };
        with_contract(Receipt::new(test, outcome), contract)
            .seam(seam)
            .detail(comparison.to_json())
            .emit();
        assert!(
            comparison.holds(),
            "{}",
            describe_exact_failure(seam, &comparison)
        );
        comparison
    }
}

/// Runs a known, ledgered divergence as XFAIL — never as a skip.
///
/// `body` runs the check and returns `Ok(())` if it now **holds**; a divergence that still
/// reproduces returns `Err` describing it. Returns `true` when the divergence reproduced (the
/// expected outcome).
///
/// The check keeps executing precisely so its *unexpected success* is detectable: a divergence that
/// silently healed leaves a stale ledger entry, and the day the divergence returns nothing catches
/// it. So an unexpected pass fails the test.
///
/// ```
/// use ftts_conformance::xfail;
///
/// let still_diverging = xfail(
///     "doc_example_tokenizer_regex_class",
///     "Demo/XFAIL",
///     "docs/DISCREPANCIES.md#tokenizer-regex",
///     || Err("upstream splits \\p{N} runs; we split per digit".to_owned()),
/// );
/// assert!(still_diverging);
/// ```
///
/// # Panics
///
/// Panics when `body` returns `Ok(())`, naming the ledger entry that must now be retired.
pub fn xfail<F>(test: &str, contract: &str, ledger: &str, body: F) -> bool
where
    F: FnOnce() -> Result<(), String>,
{
    let started = std::time::Instant::now();
    match body() {
        Err(divergence) => {
            report::Receipt::new(test, report::Outcome::ExpectedFailure)
                .contract(contract)
                .reason(divergence)
                .detail(serde_json::json!({ "ledger": ledger }))
                .elapsed(started.elapsed())
                .emit();
            true
        }
        Ok(()) => {
            report::Receipt::new(test, report::Outcome::Failed)
                .contract(contract)
                .reason("XFAIL unexpectedly passed")
                .detail(serde_json::json!({ "ledger": ledger }))
                .elapsed(started.elapsed())
                .emit();
            panic!(
                "XFAIL `{test}` unexpectedly PASSED: the divergence recorded at `{ledger}` no \
                 longer reproduces. Re-gate it as an ordinary assertion and retire the ledger \
                 entry — a stale XFAIL is a test that will not notice when the divergence returns."
            );
        }
    }
}

/// Runs a model-gated check, or emits an honest skip receipt when weights are absent.
///
/// Wraps the branch every model-gated test would otherwise hand-roll — and hand-roll slightly
/// differently each time, which is how an unexplained skip eventually reads as a pass.
///
/// Returns `true` when the body ran.
///
/// # Examples
///
/// ```
/// use ftts_conformance::{gated, gate::ModelGate};
///
/// // With no model present this records a skip and returns false.
/// let ran = gated("doc_example_codec_token_stream", "Demo/ModelGate", |artifact| {
///     assert!(artifact.exists());
/// });
/// assert_eq!(ran, ModelGate::resolve().is_present());
/// ```
pub fn gated<F>(test: &str, contract: &str, body: F) -> bool
where
    F: FnOnce(&std::path::Path),
{
    let gate = gate::ModelGate::resolve();
    match &gate {
        gate::ModelGate::Present { artifact } => {
            let started = std::time::Instant::now();
            body(artifact);
            report::Receipt::new(test, report::Outcome::Passed)
                .contract(contract)
                .elapsed(started.elapsed())
                .emit();
            true
        }
        gate::ModelGate::Absent { reason } => {
            report::Receipt::new(test, report::Outcome::Skipped)
                .contract(contract)
                .reason(reason)
                .emit();
            false
        }
    }
}
