//! The talk-session NDJSON contract test (schema v2): the same two properties the v1 robot
//! contract pins, applied to the session vocabulary.
//!
//! 1. **Conformance** — events validate against the session catalogue, ops against the op
//!    catalogue, both through the same strict-closed walk as v1.
//! 2. **Freezing** — the self-description emitted by `ftts robot schema session` is diffed
//!    against `fixtures/session_schema_v2.json`, so a change to the session wire shape is a
//!    reviewable fixture diff rather than a surprise for whoever parses our output.
//!
//! The per-utterance seed derivation vectors are pinned in `ftts-cli/src/session_protocol.rs`
//! next to the function they pin; this suite owns the wire shapes.
//!
//! Bead: frankentts-e3zz.

use std::path::PathBuf;

use ftts_cli::robot::DOCUMENTED_ENVIRONMENT;
use ftts_cli::session_protocol::{
    SESSION_EVENTS, SESSION_OPS, SESSION_SCHEMA_VERSION, validate_session_event,
    validate_session_op,
};
use serde_json::{Value, json};

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/session_schema_v2.json")
}

fn current_schema() -> Value {
    ftts_cli::session_protocol::session_schema_document(DOCUMENTED_ENVIRONMENT)
}

/// Build a minimally-populated object for one catalogued event or op.
fn minimal(discriminator: &str, name: &str) -> Value {
    let catalogue = if discriminator == "event" {
        SESSION_EVENTS
    } else {
        SESSION_OPS
    };
    let spec = catalogue
        .iter()
        .find(|spec| spec.name == name)
        .expect("catalogued entry");
    let mut object = serde_json::Map::new();
    if discriminator == "event" {
        object.insert("schema_version".to_owned(), json!(SESSION_SCHEMA_VERSION));
        // The other envelope fields; a real emitter fills session_id/seq via
        // SessionEvent::object, the test only needs shapes the validator accepts.
        object.insert("session_id".to_owned(), json!("s"));
        object.insert("seq".to_owned(), json!(0));
    }
    object.insert(discriminator.to_owned(), json!(name));
    for field in spec.fields.iter().filter(|field| field.required) {
        let value = if field.ty.ends_with("|null") {
            Value::Null
        } else {
            match field.ty {
                "string" => json!("value"),
                "bool" => json!(false),
                "object" => json!({}),
                "array" => json!([]),
                "number" => json!(1.5),
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
        let scratch = std::env::temp_dir().join("session_schema_v2.actual.json");
        let _ = std::fs::write(&scratch, format!("{actual_pretty}\n"));
        panic!(
            "the talk-session NDJSON contract changed.\n\n\
             This is not automatically an error — it is a reviewable one. Agents parse this\n\
             schema, so a change must be seen by a human before it ships.\n\n\
             Emitted schema written to: {}\n\
             If the change is intended, copy it over the fixture and review the diff:\n\
             \n    cp {} {}\n\n\
             If it is not intended, the catalogue in ftts-cli/src/session_protocol.rs drifted.",
            scratch.display(),
            scratch.display(),
            path.display(),
        );
    }
}

#[test]
fn the_schema_document_satisfies_its_own_contract() {
    let problems = validate_session_event(&current_schema());
    assert!(problems.is_empty(), "{problems:?}");
}

#[test]
fn every_catalogued_session_event_conforms() {
    for spec in SESSION_EVENTS {
        let problems = validate_session_event(&minimal("event", spec.name));
        assert!(problems.is_empty(), "{}: {problems:?}", spec.name);
    }
}

#[test]
fn every_catalogued_session_op_conforms() {
    for spec in SESSION_OPS {
        let problems = validate_session_op(&minimal("op", spec.name));
        assert!(problems.is_empty(), "{}: {problems:?}", spec.name);
    }
}

#[test]
fn a_drifted_session_event_fails_conformance() {
    let mut audio = minimal("event", "audio");
    let object = audio.as_object_mut().expect("object");
    object.insert("bytes".to_owned(), json!("many")); // wrong type
    object.insert("extra".to_owned(), json!(true)); // unknown field
    object.remove("byte_offset"); // missing required field
    let problems = validate_session_event(&audio);
    assert_eq!(problems.len(), 3, "{problems:?}");
    assert!(problems.iter().any(|p| p.contains("bytes")), "{problems:?}");
    assert!(
        problems.iter().any(|p| p.contains("byte_offset")),
        "{problems:?}"
    );
    assert!(
        problems.iter().any(|p| p.contains("`extra`")),
        "{problems:?}"
    );
}

#[test]
fn a_v1_stamped_object_is_not_a_session_event() {
    let mut event = minimal("event", "speak_complete");
    event["schema_version"] = json!(1);
    let problems = validate_session_event(&event);
    assert!(
        problems
            .iter()
            .any(|problem| problem.contains("schema_version 1 != contract version 2")),
        "{problems:?}"
    );
}

#[test]
fn a_session_stream_validates_line_by_line() {
    let stream = concat!(
        r#"{"schema_version":2,"event":"session_start","session_id":"s","seq":0,"version":"0.1.8","model":"m","route":"r","pcm":{"format":"s16le","rate":24000,"channels":1},"pid":1}"#,
        "\n",
        r#"{"schema_version":2,"event":"ack","session_id":"s","seq":1,"id":null,"op":"open","context":"c1"}"#,
        "\n",
        r#"{"schema_version":2,"event":"session_error","session_id":"s","seq":2,"kind":"unknown_op","message":"no such op","remediation":"see ftts robot schema session"}"#,
        "\n",
    );
    let problems: Vec<String> = stream
        .lines()
        .enumerate()
        .flat_map(|(index, line)| {
            let value: Value = serde_json::from_str(line).expect("test stream is valid JSON");
            validate_session_event(&value)
                .into_iter()
                .map(move |problem| format!("line {}: {problem}", index + 1))
        })
        .collect();
    assert!(problems.is_empty(), "{problems:?}");
}

#[test]
fn the_session_exit_code_is_published_in_the_schema() {
    let document = current_schema();
    assert_eq!(
        document["exit_codes"]["9"],
        json!("talk-session transport failed"),
        "the session-fatal transport code must be documented in the v2 contract"
    );
}
