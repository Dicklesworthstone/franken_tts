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
//! Never a bare assertion. [`compare::compare_f32`] plus [`compare::describe_failure`] report the
//! first divergent element with row-major coordinates, then max-abs, max-rel, cosine, and
//! count-over-tolerance — the statistics that separate a SIMD lane bug from a transposed tensor.
//!
//! ## (c) End-to-end runs log every stage transition
//!
//! Wall-clock plus key intermediate hashes (token-stream hash, PCM hash) via [`report::emit_stage`],
//! so a red run names its stage in seconds instead of requiring a bisect.
//!
//! ## (d) Logs are machine-parseable
//!
//! One JSON object per event, one per line, on stdout, so CI can aggregate failure patterns across
//! runs. [`report::Receipt::to_line`] never emits an embedded newline.
//!
//! ## (e) Test names encode the contract
//!
//! `contract_a_l2_talker_layer17_cosine`, never `test_talker_3`. The name should survive being read
//! alone in a CI summary.
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
//! that its unexpected *success* is also detectable.
//!
//! # Usage
//!
//! ```
//! use ftts_conformance::{
//!     gate::ModelGate,
//!     report::{Outcome, Receipt},
//! };
//!
//! let gate = ModelGate::resolve();
//! let Some(artifact) = gate.artifact() else {
//!     // Honest skip: recorded as `skipped`, never as a pass.
//!     Receipt::new("contract_a_l4_codec_token_stream", Outcome::Skipped)
//!         .contract("ConformanceExact/L4")
//!         .reason("model artifact unavailable")
//!         .emit();
//!     return;
//! };
//! let _ = artifact; // real work goes here
//! ```

pub mod compare;
pub mod gate;
pub mod report;

/// The imports a conformance test almost always wants.
pub mod prelude {
    pub use crate::compare::{Comparison, compare_f32, coordinates, describe_failure};
    pub use crate::gate::{
        ExecutionPath, ModelGate, NONEXISTENT_FALLBACK, require_native_execution,
    };
    pub use crate::report::{FixtureProvenance, Outcome, Receipt, emit_stage};
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
/// let ran = gated("contract_a_l4_codec_token_stream", "ConformanceExact/L4", |artifact| {
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
