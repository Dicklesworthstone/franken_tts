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

/// Stage transitions carry wall-clock and intermediate hashes.
#[test]
fn stage_events_carry_timing_and_intermediate_hashes() {
    emit_stage(
        "demo.codec_decode",
        Duration::from_millis(7),
        &[("token_stream", "deadbeef"), ("pcm", "cafebabe")],
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
