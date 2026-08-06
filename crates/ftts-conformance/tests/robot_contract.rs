//! The robot NDJSON contract test: emitted events must satisfy the catalogue, and the
//! catalogue must match a frozen fixture.
//!
//! Two distinct properties, and the second is the one that stops silent drift:
//!
//! 1. **Conformance** — every object the binary can emit validates against the catalogue in
//!    `ftts_cli::robot`. This catches a field that changed type or went missing.
//! 2. **Freezing** — the catalogue itself is diffed against `fixtures/robot_schema_v1.json`.
//!    Property 1 alone would happily accept a contract that quietly grew three fields, because
//!    the catalogue and the emitter would have changed together. The fixture is the thing that
//!    does not change on its own, so a schema edit becomes a reviewable diff instead of a
//!    surprise for whoever is parsing our output.
//!
//! Updating the fixture is deliberate and one command; see the failure message.
//!
//! Bead: frankentts-p0-robot-82c.

use std::path::PathBuf;

use ftts_cli::robot::{
    self, DOCUMENTED_ENVIRONMENT, EVENTS, EventType, SCHEMA_VERSION, validate_event,
    validate_ndjson,
};
use serde_json::{Value, json};

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/robot_schema_v1.json")
}

fn current_schema() -> Value {
    robot::schema_document(DOCUMENTED_ENVIRONMENT)
}

/// Build a minimally-populated object for one catalogued event.
fn minimal(name: &str) -> Value {
    let spec = robot::event_spec(name).expect("catalogued event");
    let mut object = serde_json::Map::new();
    object.insert("schema_version".to_owned(), json!(SCHEMA_VERSION));
    object.insert("event".to_owned(), json!(name));
    for field in spec.fields.iter().filter(|field| field.required) {
        let value = if field.ty.ends_with("|null") {
            Value::Null
        } else {
            match field.ty {
                "string" => json!("value"),
                "bool" => json!(false),
                "object" => json!({}),
                "array" => json!([]),
                _ => json!(0),
            }
        };
        object.insert(field.name.to_owned(), value);
    }
    Value::Object(object)
}

#[test]
fn schema_document_matches_the_frozen_fixture() {
    let path = fixture_path();
    let expected: Value = serde_json::from_str(
        &std::fs::read_to_string(&path)
            .unwrap_or_else(|error| panic!("cannot read {}: {error}", path.display())),
    )
    .expect("fixture is valid JSON");
    let actual = current_schema();

    if actual != expected {
        let actual_pretty = serde_json::to_string_pretty(&actual).expect("serialize");
        let scratch = std::env::temp_dir().join("robot_schema_v1.actual.json");
        let _ = std::fs::write(&scratch, format!("{actual_pretty}\n"));
        panic!(
            "the robot NDJSON contract changed.\n\n\
             This is not automatically an error — it is a reviewable one. Agents parse this\n\
             schema, so a change must be seen by a human before it ships.\n\n\
             Emitted schema written to: {}\n\
             If the change is intended, copy it over the fixture and review the diff:\n\
             \n    cp {} {}\n\n\
             If it is not intended, the catalogue in ftts-cli/src/robot.rs drifted.",
            scratch.display(),
            scratch.display(),
            path.display(),
        );
    }
}

#[test]
fn the_schema_document_satisfies_its_own_contract() {
    // `robot schema` is itself an emitted object, so it must validate like any other.
    let problems = validate_event(&current_schema());
    assert!(problems.is_empty(), "{problems:?}");
}

#[test]
fn every_catalogued_event_conforms() {
    for spec in EVENTS {
        let problems = validate_event(&minimal(spec.name));
        assert!(problems.is_empty(), "{}: {problems:?}", spec.name);
    }
}

#[test]
fn the_seven_stream_events_are_all_catalogued() {
    // The bead names these explicitly; a rename would otherwise pass unnoticed because the
    // catalogue would still be self-consistent.
    let names: Vec<&str> = EVENTS.iter().map(|spec| spec.name).collect();
    for required in [
        "run_start",
        "stage",
        "frame",
        "audio_chunk",
        "health",
        "run_complete",
        "run_error",
    ] {
        assert!(names.contains(&required), "catalogue lost {required}");
    }
}

#[test]
fn run_error_carries_an_exit_code_and_reaches_stderr() {
    // Two promises an agent depends on: it can read the process exit code out of the stream
    // without waiting for the process, and it will never find run_error interleaved with PCM.
    let spec = EventType::RunError.spec();
    let exit_code = spec
        .fields
        .iter()
        .find(|field| field.name == "exit_code")
        .expect("run_error carries exit_code");
    assert!(exit_code.required, "exit_code must be required");
    assert_eq!(exit_code.ty, "u8");
    assert_eq!(spec.stream, robot::Stream::Stderr);
}

#[test]
fn a_drifted_event_fails_conformance() {
    // Negative controls: the test would be worthless if it only ever saw valid input.
    let mut missing = minimal("run_complete");
    missing
        .as_object_mut()
        .expect("object")
        .remove("exit_code");
    assert!(!validate_event(&missing).is_empty(), "missing field accepted");

    let mut extra = minimal("health");
    extra
        .as_object_mut()
        .expect("object")
        .insert("surprise".to_owned(), json!(1));
    assert!(!validate_event(&extra).is_empty(), "unknown field accepted");

    let mut wrong_type = minimal("frame");
    wrong_type
        .as_object_mut()
        .expect("object")
        .insert("index".to_owned(), json!("three"));
    assert!(!validate_event(&wrong_type).is_empty(), "wrong type accepted");

    let mut stale = minimal("run_start");
    stale
        .as_object_mut()
        .expect("object")
        .insert("schema_version".to_owned(), json!(SCHEMA_VERSION + 1));
    assert!(!validate_event(&stale).is_empty(), "stale version accepted");
}

#[test]
fn an_ndjson_stream_validates_line_by_line() {
    let stream = format!(
        "{}\n{}\n",
        serde_json::to_string(&minimal("run_start")).expect("json"),
        serde_json::to_string(&minimal("run_complete")).expect("json"),
    );
    assert!(validate_ndjson(&stream).is_empty());

    // A stream is line-oriented by contract: a pretty-printed object spanning several lines is
    // a violation, because `while read line` is how agents consume this.
    let pretty = serde_json::to_string_pretty(&minimal("health")).expect("json");
    assert!(!validate_ndjson(&pretty).is_empty(), "multi-line object accepted");
}

#[test]
fn documented_environment_is_pinned_by_the_fixture() {
    let schema = current_schema();
    let listed: Vec<&str> = schema["environment_variables"]
        .as_array()
        .expect("environment_variables array")
        .iter()
        .map(|value| value.as_str().expect("string"))
        .collect();
    assert_eq!(listed, DOCUMENTED_ENVIRONMENT);
    assert!(
        listed.contains(&"FTTS_MODEL_DIR"),
        "model resolution must stay documented: resolution errors list every searched directory"
    );
}

#[test]
fn exit_codes_are_published_in_the_schema() {
    // Agents branch on these; they are part of the contract, not an implementation detail.
    let schema = current_schema();
    let codes = schema["exit_codes"].as_object().expect("exit_codes object");
    for (code, meaning) in [("0", "success"), ("2", "usage"), ("3", "model not found")] {
        let described = codes
            .get(code)
            .and_then(Value::as_str)
            .unwrap_or_else(|| panic!("exit code {code} is undocumented"));
        assert!(
            described.contains(meaning),
            "exit code {code} describes {described:?}, expected it to mention {meaning:?}"
        );
    }
}
