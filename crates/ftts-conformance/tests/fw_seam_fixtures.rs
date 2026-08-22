//! Replay-fixture hygiene for the franken_whisper -> ftts seam (bead frankentts-yex8).
//!
//! The four NDJSON streams encode the seam contract agreed with the fw agent; the demo's replay
//! mode (frankentts-4ie8) will branch on them, so they may not rot silently. This suite enforces
//! the CONTRACT-LEVEL invariants — envelope shape, vocabulary, monotone timing, the append-only
//! delta rule — not fw's internals. If fw's real streams ever violate these, the divergence goes
//! to the mail thread, not silently into a widened assertion.

use serde_json::Value;
use std::path::PathBuf;

const EVENT_VOCABULARY: &[&str] = &[
    "vad.start",
    "speech_started",
    "transcript.partial",
    "transcript.delta",
    "transcript.retract",
    "utterance_end",
];

const SCENARIOS: &[&str] = &[
    "normal_turn.ndjson",
    "hesitating_speaker.ndjson",
    "barge_in_mid_agent_speech.ndjson",
    "backchannel_only.ndjson",
];

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/franken_whisper_replay")
}

fn lines(path: &PathBuf) -> Vec<(usize, Value)> {
    let raw = std::fs::read_to_string(path)
        .unwrap_or_else(|error| panic!("cannot read {}: {error}", path.display()));
    raw.lines()
        .enumerate()
        .filter(|(_, line)| !line.trim().is_empty() && !line.trim().starts_with('#'))
        .map(|(index, line)| {
            let value: Value = serde_json::from_str(line).unwrap_or_else(|error| {
                panic!("{}:{}: not JSON: {error}", path.display(), index + 1)
            });
            (index + 1, value)
        })
        .collect()
}

#[test]
fn every_scenario_fixture_exists() {
    for scenario in SCENARIOS {
        let path = fixture_dir().join(scenario);
        assert!(path.is_file(), "missing replay fixture {}", path.display());
    }
}

#[test]
fn every_line_satisfies_the_pinned_envelope_and_vocabulary() {
    for scenario in SCENARIOS {
        let path = fixture_dir().join(scenario);
        let mut previous_seq: Option<u64> = None;
        let mut previous_ts: Option<f64> = None;
        for (line_number, value) in lines(&path) {
            let version = value["schema_version"].as_str().unwrap_or_else(|| {
                panic!("{scenario}:{line_number}: schema_version must be the STRING \"1.1.0\"")
            });
            assert_eq!(version, "1.1.0", "{scenario}:{line_number}");
            let event = value["event"].as_str().expect("event discriminator");
            assert!(
                EVENT_VOCABULARY.contains(&event),
                "{scenario}:{line_number}: {event} is outside the agreed vocabulary"
            );
            assert!(
                value["run_id"].is_string() && value["seq"].is_u64() && value["ts"].is_number(),
                "{scenario}:{line_number}: run_id/seq/ts are required"
            );
            let seq = value["seq"].as_u64().expect("seq");
            let ts = value["ts"].as_f64().expect("ts");
            if let Some(previous) = previous_seq {
                assert!(
                    seq > previous,
                    "{scenario}:{line_number}: seq must increase"
                );
            }
            if let Some(previous) = previous_ts {
                assert!(
                    ts >= previous,
                    "{scenario}:{line_number}: ts must be monotone"
                );
            }
            previous_seq = Some(seq);
            previous_ts = Some(ts);

            match event {
                "transcript.partial" => {
                    let text = value["text"].as_str().expect("partial.text");
                    assert!(!text.is_empty(), "{scenario}:{line_number}");
                    let confidence = value["confidence"].as_f64().expect("partial.confidence");
                    assert!(
                        (0.0..=1.0).contains(&confidence),
                        "{scenario}:{line_number}: confidence out of range"
                    );
                    let start = value["start_sec"].as_f64().expect("start_sec");
                    let end = value["end_sec"].as_f64().expect("end_sec");
                    assert!(start <= end, "{scenario}:{line_number}");
                }
                "transcript.delta" => {
                    assert!(
                        value["utterance_id"].is_string() && value["text"].is_string(),
                        "{scenario}:{line_number}: delta needs utterance_id + text"
                    );
                }
                "utterance_end" => {
                    let reason = value["reason"].as_str().expect("end.reason");
                    assert!(
                        ["endpoint", "timeout", "max_len", "session_end"].contains(&reason),
                        "{scenario}:{line_number}: unknown endpoint reason {reason}"
                    );
                    assert!(
                        value["utterance_id"].is_string()
                            && value["text"].as_str().is_some_and(|t| !t.is_empty()),
                        "{scenario}:{line_number}: end needs utterance_id + finalized text"
                    );
                }
                _ => {}
            }
        }
    }
}

#[test]
fn deltas_are_append_only_within_an_utterance() {
    for scenario in SCENARIOS {
        let path = fixture_dir().join(scenario);
        let mut committed: std::collections::HashMap<String, String> =
            std::collections::HashMap::new();
        let mut last_delta_per_utterance: std::collections::HashMap<String, String> =
            std::collections::HashMap::new();
        for (line_number, value) in lines(&path) {
            match value["event"].as_str().expect("event") {
                "transcript.delta" => {
                    let id = value["utterance_id"].as_str().expect("id");
                    let text = value["text"].as_str().expect("text");
                    if let Some(previous) = committed.get(id) {
                        assert!(
                            text.starts_with(previous.as_str()) && text.len() > previous.len(),
                            "{scenario}:{line_number}: delta is not an extension of the \
                             committed text for {id}"
                        );
                    }
                    committed.insert(id.to_owned(), text.to_owned());
                    last_delta_per_utterance.insert(id.to_owned(), text.to_owned());
                }
                "utterance_end" => {
                    let id = value["utterance_id"].as_str().expect("id");
                    // The finalized text must CONTAIN everything committed so far — the
                    // finalizer may adjust punctuation but never drop committed words.
                    if let Some(last) = last_delta_per_utterance.get(id) {
                        let normalize = |s: &str| {
                            s.trim_end_matches(|c: char| c == '.' || c == '?' || c == '!')
                                .to_lowercase()
                        };
                        assert!(
                            normalize(value["text"].as_str().expect("text"))
                                .contains(&normalize(last)),
                            "{scenario}:{line_number}: finalized text dropped committed words"
                        );
                    }
                    committed.remove(id);
                }
                _ => {}
            }
        }
    }
}

#[test]
fn a_retract_always_follows_a_partial_in_the_same_run() {
    for scenario in SCENARIOS {
        let path = fixture_dir().join(scenario);
        let mut seen_partial = false;
        for (line_number, value) in lines(&path) {
            match value["event"].as_str().expect("event") {
                "transcript.partial" => seen_partial = true,
                "transcript.retract" => {
                    assert!(
                        seen_partial,
                        "{scenario}:{line_number}: retract without a preceding speculative partial"
                    );
                }
                _ => {}
            }
        }
    }
}

#[test]
fn barge_in_scenario_carries_the_trigger_before_any_user_transcript() {
    let path = fixture_dir().join("barge_in_mid_agent_speech.ndjson");
    let stream = lines(&path);
    let speech_started_position = stream
        .iter()
        .position(|(_, v)| v["event"] == "speech_started")
        .expect("barge-in fixture must carry speech_started");
    let first_transcript_position = stream
        .iter()
        .position(|(_, v)| v["event"].as_str().unwrap_or("").starts_with("transcript."))
        .expect("barge-in fixture must carry transcripts");
    assert!(
        speech_started_position < first_transcript_position,
        "the barge-in trigger must precede transcript work: it fires the cancel, the text only \
         explains it"
    );
}
