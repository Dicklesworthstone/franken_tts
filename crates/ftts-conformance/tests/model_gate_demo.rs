//! Executable demonstration of the model gate and the test observability convention.
//!
//! This file is the worked example every other conformance test copies. It proves all three gate
//! states and shows that a deliberately failed comparison produces a fully self-localizing log.
//!
//! Only the skip-vs-present branch depends on the environment; the other states are exercised
//! deterministically so the demo keeps its meaning on a machine with no weights. That matters:
//! a demo that silently degrades to "nothing ran" would be the exact failure this bead exists to
//! prevent.

use std::{fs, path::PathBuf, process, time::Duration};

use ftts_conformance::{assert_close, assert_exact, gated, prelude::*, test_name};

fn scratch_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("ftts-gate-demo-{tag}-{}", process::id()));
    fs::create_dir_all(&dir).expect("scratch dir is creatable");
    dir
}

/// STATE 1 — green-with-model: the gate opens and the body runs for real.
///
/// Driven through `resolve_from` with a materialized artifact so the green path is proven here even
/// when the developer has no weights. Without this, "green-with-model" would be an untested claim.
#[test]
fn model_gate_state_green_with_model() {
    let dir = scratch_dir("green");
    let artifact = dir.join(ftts_conformance::gate::MODEL_BASENAME);
    fs::write(
        &artifact,
        b"stand-in artifact; the gate checks presence, not contents",
    )
    .expect("artifact is writable");

    let gate = ModelGate::resolve_from(Some(&dir));
    assert!(gate.is_present(), "gate must open when the artifact exists");

    // Under an open gate the native path must be the one that ran.
    require_native_execution("demo.talker.prefill", ExecutionPath::Native)
        .expect("native execution is accepted");

    Receipt::new("demo_model_gate_state_green_with_model", Outcome::Passed)
        .contract("Demo/ModelGate")
        .seam("demo.talker.prefill")
        .provenance(
            FixtureProvenance::new(artifact.display().to_string()).with_revision("5d839924"),
        )
        .elapsed(Duration::from_millis(1))
        .emit();

    fs::remove_file(&artifact).expect("scratch artifact is removable");
}

/// STATE 2 — skip-without-model: honest skip, never a pass.
#[test]
fn model_gate_state_skip_without_model() {
    let gate = ModelGate::resolve_from(None);
    let ModelGate::Absent { reason } = &gate else {
        panic!("gate must be closed when no model directory is supplied");
    };

    let receipt = Receipt::new("demo_model_gate_state_skip_without_model", Outcome::Skipped)
        .contract("Demo/ModelGate")
        .reason(reason);
    receipt.emit();

    // The receipt is the honest signal: libtest will call this test "ok" either way.
    let value = receipt.to_json();
    assert_eq!(
        value["outcome"], "skipped",
        "a skip must never serialize as passed"
    );
    assert!(
        value["reason"].as_str().is_some_and(|r| !r.is_empty()),
        "a skip without a reason is a disabled test"
    );
}

/// STATE 3 — loud failure when a fallback runs under an open gate.
///
/// The silent-fallback bug produces a *green* suite that exercised nothing, so the gate must turn
/// it into a named error rather than a quiet substitution.
#[test]
fn model_gate_state_loud_failure_on_silent_fallback() {
    let error = require_native_execution("demo.codec.decode", ExecutionPath::Fallback)
        .expect_err("a fallback under an open gate must fail loudly");

    assert!(
        error.contains("demo.codec.decode"),
        "names the seam: {error}"
    );
    assert!(
        error.contains("FALLBACK"),
        "names the path that ran: {error}"
    );
    assert!(
        error.contains(NONEXISTENT_FALLBACK),
        "tells the reader how to make this fail at construction: {error}"
    );

    Receipt::new(
        "demo_model_gate_state_loud_failure_on_silent_fallback",
        Outcome::Passed,
    )
    .contract("Demo/ModelGate")
    .seam("demo.codec.decode")
    .detail(serde_json::json!({ "rejected_error": error }))
    .emit();
}

/// The sentinel fallback target must not exist, or state 3 cannot be detected at all.
#[test]
fn nonexistent_fallback_sentinel_is_actually_absent() {
    assert!(
        !std::path::Path::new(NONEXISTENT_FALLBACK).exists(),
        "fallback detection depends on `{NONEXISTENT_FALLBACK}` never existing"
    );
}

/// The convention's core claim: a failed comparison localizes itself.
///
/// Asserts on the *content* of the report rather than on a panic, so the demo proves the log is
/// actionable instead of merely proving that something failed.
#[test]
fn deliberately_failed_comparison_is_fully_self_localizing() {
    // A 2x3 tensor whose element [1, 1] is wrong — one lane, everything else exact.
    let expected = [0.10_f32, 0.20, 0.30, 0.40, 0.50, 0.60];
    let actual = [0.10_f32, 0.20, 0.30, 0.40, 0.75, 0.60];
    let shape = [2_usize, 3];
    let tolerance = 1.0e-4;
    let tolerance_source = "docs/truth-pack/nondeterminism-floor.json";

    let comparison = compare_f32(&expected, &actual, tolerance);
    assert!(!comparison.holds(), "this comparison is meant to fail");

    let report = describe_failure(
        "demo.talker.layer17.attn_out",
        &comparison,
        tolerance,
        tolerance_source,
        Some(&shape),
    );

    // Everything an agent needs at 3am, without rerunning anything.
    for required in [
        "demo.talker.layer17.attn_out", // the seam
        "flat[4]",                      // flat index
        "index[1, 1]",                  // multi-dimensional coordinates
        "expected",
        "actual",
        "max_abs",
        "cosine",
        "shape [2, 3]",
        tolerance_source, // where the tolerance came from, not just its value
        "hint:",          // a diagnosis, not just numbers
    ] {
        assert!(
            report.contains(required),
            "self-localizing report is missing `{required}`:\n{report}"
        );
    }

    assert_eq!(comparison.first_divergence, Some(4));
    assert_eq!(comparison.over_tolerance, 1);

    Receipt::new(
        "demo_comparator_failure_is_self_localizing",
        Outcome::Passed,
    )
    .contract("Demo/Observability")
    .seam("demo.talker.layer17.attn_out")
    .tolerance(tolerance, tolerance_source)
    .detail(comparison.to_json())
    .emit();
}

/// STATE 4 — XFAIL: a known divergence keeps executing, and healing it is itself detectable.
///
/// The bead's rule is "XFAIL never SKIP". A skipped known-bad check stops running, so the day the
/// divergence returns after being fixed, nothing notices.
#[test]
fn known_divergence_is_xfail_and_reports_as_xfail_not_skipped() {
    let still_diverging = xfail(
        "demo_known_divergence_tokenizer_regex_class",
        "Demo/XFAIL",
        "docs/DISCREPANCIES.md#demo-entry",
        || Err("demo divergence: upstream splits digit runs, we split per digit".to_owned()),
    );
    assert!(still_diverging, "the demo divergence must still reproduce");

    // The wire state is its own, distinct from both `skipped` and `passed`.
    assert_eq!(Outcome::ExpectedFailure.as_str(), "xfail");
    assert_ne!(
        Outcome::ExpectedFailure.as_str(),
        Outcome::Skipped.as_str(),
        "an XFAIL must never be recorded as a skip"
    );
}

/// An XFAIL that starts passing fails loudly instead of quietly becoming a no-op.
#[test]
#[should_panic(expected = "unexpectedly PASSED")]
fn an_xfail_that_starts_passing_is_a_loud_failure() {
    xfail(
        "demo_xfail_that_healed",
        "Demo/XFAIL",
        "docs/DISCREPANCIES.md#demo-entry",
        || Ok(()),
    );
}

/// Stage transitions carry wall-clock and intermediate hashes.
#[test]
fn stage_events_carry_timing_and_intermediate_hashes() {
    emit_stage(
        "demo.codec_decode",
        Duration::from_millis(7),
        &[("token_stream", "deadbeef"), ("pcm", "cafebabe")],
    );

    // The timed form cannot report a wall-clock that drifted from the work it measured, and the
    // hash helpers mean an e2e author never hand-rolls a digest for stage localization.
    let tokens: Vec<u32> = (0..16).collect();
    let pcm: Vec<f32> = (0..32_u16).map(|i| f32::from(i) / 32.0).collect();
    let stage = Stage::start("demo.talker_decode");
    let stage_hashes = [
        ("token_stream", token_stream_hash(&tokens)),
        ("pcm", pcm_hash(&pcm)),
    ];
    stage.finish(
        &stage_hashes
            .iter()
            .map(|(name, value)| (*name, value.as_str()))
            .collect::<Vec<_>>(),
    );
}

/// The macros carry the convention with one import — including the receipt's test name.
#[test]
fn convention_macros_emit_receipts_named_after_the_enclosing_test() {
    assert!(
        test_name!().ends_with("convention_macros_emit_receipts_named_after_the_enclosing_test"),
        "a receipt must be able to name itself after the test that emitted it, got `{}`",
        test_name!()
    );

    let expected = [0.10_f32, 0.20, 0.30, 0.40, 0.50, 0.60];
    let actual = [0.10_f32, 0.20, 0.30, 0.40, 0.50, 0.60];
    let comparison = assert_close!(
        contract = "Demo/Observability",
        seam = "demo.codec.upsample_stage0",
        expected = &expected,
        actual = &actual,
        tolerance = 1.0e-4,
        source = "docs/truth-pack/nondeterminism-floor.json",
        shape = &[2, 3],
    );
    assert!(comparison.holds());

    let tokens = [11_u32, 22, 33, 44];
    assert_exact!(
        contract = "Demo/Observability",
        seam = "demo.tokenizer.encode",
        expected = &tokens,
        actual = &tokens,
    );
}

/// A failing `assert_close!` panics with the self-localizing report, not a bare assertion.
///
/// This emits a genuine `failed` receipt on purpose, which is exactly why it carries a `Demo/`
/// contract: the aggregator must not read a staged failure as a red ladder rung.
#[test]
#[should_panic(expected = "demo.talker.layer17.attn_out")]
fn assert_close_panics_with_the_self_localizing_report() {
    let expected = [0.10_f32, 0.20, 0.30, 0.40, 0.50, 0.60];
    let actual = [0.10_f32, 0.20, 0.30, 0.40, 0.75, 0.60];
    assert_close!(
        contract = "Demo/Observability",
        seam = "demo.talker.layer17.attn_out",
        expected = &expected,
        actual = &actual,
        tolerance = 1.0e-4,
        source = "docs/truth-pack/nondeterminism-floor.json",
        shape = &[2, 3],
    );
}

/// A diverging token stream is localized by index, context, and a shift diagnosis.
#[test]
fn exact_token_stream_divergence_is_fully_self_localizing() {
    // `actual` dropped token 33: the tail realigns under a one-element shift.
    let expected = [11_u32, 22, 33, 44, 55, 66];
    let actual = [11_u32, 22, 44, 55, 66];

    let comparison = compare_exact(&expected, &actual);
    assert!(!comparison.holds());
    assert_eq!(comparison.first_divergence, Some(2));
    assert_eq!(comparison.shift, Some(Shift::DroppedFromActual));

    let report = describe_exact_failure("demo.talker.token_stream", &comparison);
    for required in [
        "demo.talker.token_stream", // the seam
        "index 2",                  // where they parted
        "expected_len 6",
        "actual_len 5",
        "context from index",
        "DROPPED", // the diagnosis, not just the numbers
        "hint:",
    ] {
        assert!(
            report.contains(required),
            "exact-divergence report is missing `{required}`:\n{report}"
        );
    }

    Receipt::new(
        "demo_exact_stream_divergence_is_self_localizing",
        Outcome::Passed,
    )
    .contract("Demo/Observability")
    .seam("demo.talker.token_stream")
    .detail(comparison.to_json())
    .emit();
}

/// Receipts must reach CI even though `libtest` swallows stdout on green runs.
///
/// Re-runs this binary's own child test with `FTTS_RECEIPTS` set, then reads the file. This is the
/// only honest proof: edition 2024 makes `env::set_var` `unsafe` (forbidden here), so the
/// environment-driven path cannot be exercised in-process.
#[test]
fn receipts_reach_the_sink_file_despite_libtest_capturing_stdout() {
    let sink = scratch_dir("sink").join("receipts.ndjson");
    let _ = fs::remove_file(&sink);

    let status = process::Command::new(std::env::current_exe().expect("test binary path"))
        .args([
            "--exact",
            "--nocapture",
            "receipt_sink_child_emits_two_events",
        ])
        .env(ftts_conformance::report::RECEIPTS_ENV, &sink)
        .status()
        .expect("the test binary re-runs");
    assert!(status.success(), "child test run failed: {status}");

    let contents = fs::read_to_string(&sink).expect("the sink file exists after the child run");
    let events: Vec<serde_json::Value> = contents
        .lines()
        .map(|line| serde_json::from_str(line).expect("each line is one JSON object"))
        .collect();
    assert_eq!(events.len(), 2, "expected both events:\n{contents}");
    assert_eq!(events[0]["event"], "contract_check");
    assert_eq!(
        events[0]["outcome"], "skipped",
        "the honest skip verdict must survive to the aggregated file"
    );
    assert_eq!(events[1]["event"], "stage");

    fs::remove_file(&sink).expect("scratch sink is removable");
}

/// Child of the sink test above. Emits exactly two events and asserts nothing.
///
/// It runs in the ordinary test pass too, where `FTTS_RECEIPTS` is unset and it is stdout-only.
#[test]
fn receipt_sink_child_emits_two_events() {
    Receipt::new("demo_receipt_sink_child", Outcome::Skipped)
        .contract("Demo/Sink")
        .reason("child process demonstrating the receipt sink")
        .emit();
    emit_stage("demo.sink_child", Duration::from_millis(1), &[]);
}

/// Runs the model-gate child test under an explicit `FTTS_MODEL_DIR`, returning its receipts.
///
/// The gate's real entry point is [`ModelGate::resolve`], which reads the environment — and an
/// environment-driven branch cannot be exercised in-process, because edition 2024 makes
/// `env::set_var` `unsafe` and this crate forbids it. A child process is the only honest way to
/// prove that the variable every conformance test depends on actually opens and closes the gate.
fn child_receipts_with_model_dir(tag: &str, model_dir: &std::path::Path) -> Vec<serde_json::Value> {
    let sink = scratch_dir(&format!("model-dir-{tag}")).join("receipts.ndjson");
    let _ = fs::remove_file(&sink);

    let status = process::Command::new(std::env::current_exe().expect("test binary path"))
        .args([
            "--exact",
            "--nocapture",
            "model_dir_child_reports_the_resolved_gate",
        ])
        .env(ftts_conformance::report::RECEIPTS_ENV, &sink)
        .env(ftts_conformance::gate::MODEL_DIR_ENV, model_dir)
        .status()
        .expect("the test binary re-runs");
    assert!(status.success(), "child test run failed: {status}");

    let contents = fs::read_to_string(&sink).expect("the sink file exists after the child run");
    let events: Vec<serde_json::Value> = contents
        .lines()
        .map(|line| serde_json::from_str(line).expect("each line is one JSON object"))
        .collect();
    fs::remove_file(&sink).expect("scratch sink is removable");
    events
}

/// STATE 1, through the mechanism the whole suite actually uses: `FTTS_MODEL_DIR` opens the gate.
///
/// `resolve_from` proves the logic; only this proves the *wiring*. A typo in the variable name
/// would leave every model-gated test skipping forever on a machine that has the weights, and the
/// suite would stay green while testing nothing.
#[test]
fn ftts_model_dir_opens_the_gate_and_require_model_runs_the_body() {
    let dir = scratch_dir("model-dir-present");
    let artifact = dir.join(ftts_conformance::gate::MODEL_BASENAME);
    fs::write(
        &artifact,
        b"stand-in artifact; the gate checks presence, not contents",
    )
    .expect("artifact is writable");

    let events = child_receipts_with_model_dir("present", &dir);
    fs::remove_file(&artifact).expect("scratch artifact is removable");

    assert_eq!(events.len(), 1, "expected exactly one receipt: {events:?}");
    assert_eq!(
        events[0]["outcome"], "passed",
        "an open gate must run the body, not skip it: {events:?}"
    );
    assert_eq!(
        events[0]["detail"]["artifact"],
        artifact.display().to_string(),
        "the body must receive the artifact the environment named"
    );
}

/// STATE 2, same mechanism: a directory without the artifact closes the gate, honestly.
///
/// The reason must name what was missing — "skipped" alone is indistinguishable from a test
/// someone quietly disabled.
#[test]
fn ftts_model_dir_without_the_artifact_skips_with_a_reason_naming_it() {
    // The directory exists; the artifact does not. That is the everyday CI condition.
    let dir = scratch_dir("model-dir-absent");
    let events = child_receipts_with_model_dir("absent", &dir);

    assert_eq!(events.len(), 1, "expected exactly one receipt: {events:?}");
    assert_eq!(
        events[0]["outcome"], "skipped",
        "absent weights must skip, never pass: {events:?}"
    );
    let reason = events[0]["reason"].as_str().unwrap_or_default();
    assert!(
        reason.contains(ftts_conformance::gate::MODEL_BASENAME),
        "the skip reason must name the missing artifact, got `{reason}`"
    );
}

/// Child of the two `FTTS_MODEL_DIR` tests above. Emits exactly one receipt either way.
///
/// It also runs in the ordinary test pass, where it reports whatever the developer's environment
/// says — a skip on a machine without weights, a real run on one with them.
#[test]
fn model_dir_child_reports_the_resolved_gate() {
    // Closed gate: this emits an honest `skipped` receipt with a reason, and returns.
    let artifact = ftts_conformance::require_model!("Demo/RequireModel");

    // Open gate: everything below runs, and the native path must be the one that ran.
    assert!(
        artifact.is_file(),
        "an open gate must hand the body a real artifact path"
    );
    require_native_execution("demo.require_model", ExecutionPath::Native)
        .expect("native execution is accepted under an open gate");

    Receipt::new(test_name!(), Outcome::Passed)
        .contract("Demo/RequireModel")
        .seam("demo.require_model")
        .detail(serde_json::json!({ "artifact": artifact.display().to_string() }))
        .emit();
}

/// The receipt outcomes and the aggregator that reads them must not drift apart.
///
/// `scripts/summarize_receipts.py` rejects an unknown `outcome` — which is right, but it means a
/// variant added here and not there would fail the gate with "unknown outcome" instead of being
/// counted. The `wire` match below is exhaustive, so a new variant stops compiling until both
/// sides are updated.
#[test]
fn receipt_outcome_wire_strings_match_the_summarizer() {
    const fn wire(outcome: Outcome) -> &'static str {
        match outcome {
            Outcome::Passed => "passed",
            Outcome::Failed => "failed",
            Outcome::Skipped => "skipped",
            Outcome::ExpectedFailure => "xfail",
        }
    }

    let script = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../scripts/summarize_receipts.py");
    let source = fs::read_to_string(&script).expect("the receipt aggregator is present");
    let outcomes_line = source
        .lines()
        .find(|line| line.starts_with("OUTCOMES = {"))
        .expect("the aggregator declares the outcome set it accepts");

    for outcome in [
        Outcome::Passed,
        Outcome::Failed,
        Outcome::Skipped,
        Outcome::ExpectedFailure,
    ] {
        assert_eq!(
            outcome.as_str(),
            wire(outcome),
            "the wire string changed without updating this test"
        );
        assert!(
            outcomes_line.contains(&format!("\"{}\"", outcome.as_str())),
            "`{}` is not in the aggregator's accepted set: {outcomes_line}",
            outcome.as_str()
        );
    }

    // The reserved namespace the crate docs promise, bound to the code that honours it.
    assert!(
        source.contains(r#"DEMO_CONTRACT_PREFIX = "Demo/""#),
        "the aggregator no longer reserves the `Demo/` contract namespace the crate docs describe"
    );
}

/// The `gated` helper returns whether the body ran, and agrees with the resolved gate.
///
/// On a machine without weights this exercises the skip branch; on one with weights, the green
/// branch. Either way the assertion is meaningful — it never silently tests nothing.
#[test]
fn gated_helper_matches_the_resolved_gate() {
    let expected_to_run = ModelGate::resolve().is_present();
    let ran = gated(
        "demo_gated_helper_matches_resolved_gate",
        "Demo/ModelGate",
        |artifact| {
            assert!(
                artifact.exists(),
                "an open gate must hand the body a real artifact path"
            );
        },
    );
    assert_eq!(ran, expected_to_run);
}
