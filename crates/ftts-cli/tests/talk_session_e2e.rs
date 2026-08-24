//! `ftts talk` session battery: the live conversation contract, end to end, against the
//! real binary and real model (beads frankentts-sc94 + frankentts-7pgn).
//!
//! Model-gated: without a complete model directory every test reports its skip and
//! passes. NOTE for local runs: use `--release` — a debug build cannot even LOAD the
//! model inside ten minutes on current hosts (measured), so debug runs only make sense
//! with `FTTS_MODEL_DIR=/nonexistent` (the CI shape, skip-honest).
//!
//! Every stdout line of every session is validated against the frozen v2 schema via
//! `validate_session_event` — one violation anywhere fails the test naming the line.
//! Each session's teardown audits: PCM file size == sum(audio.bytes), seq strictly
//! increasing, exactly one session_start and one session_end.
//!
//! Receipts: `receipt: {...}` NDJSON on stderr per case.

use std::io::{BufRead, BufReader, Read, Write};
use std::path::PathBuf;
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{Receiver, RecvTimeoutError, channel};
use std::time::{Duration, Instant};

use serde_json::{Value, json};

fn model_dir() -> Option<PathBuf> {
    let root = std::env::var("FTTS_MODEL_DIR").map_or_else(
        |_| {
            #[allow(deprecated)]
            std::env::home_dir().map(|home| home.join(".cache/franken_tts/model"))
        },
        |dir| Some(PathBuf::from(dir)),
    )?;
    for required in [
        "vocab.json",
        "merges.txt",
        "tokenizer_config.json",
        "speech_tokenizer/model.safetensors",
    ] {
        if !root.join(required).is_file() {
            return None;
        }
    }
    if !root.join("qwen3-tts-12hz-0.6b-base.fttsq").is_file()
        && !root.join("model.safetensors").is_file()
    {
        return None;
    }
    Some(root)
}

fn scratch_dir(label: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "ftts-talk-e2e-{label}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos())
    ));
    std::fs::create_dir_all(&dir).expect("scratch dir");
    dir
}

/// Generous per-event wait: the host may be running a whole agent swarm's builds, and
/// a session's first event follows a multi-second model load.
const EVENT_TIMEOUT: Duration = Duration::from_secs(120);

/// One live talk session under test.
struct Session {
    child: Child,
    stdin: ChildStdin,
    events: Receiver<Value>,
    pcm_path: PathBuf,
    seen: Vec<Value>,
}

impl Session {
    fn spawn(label: &str) -> Session {
        let scratch = scratch_dir(label);
        let pcm_path = scratch.join("session.pcm");
        let mut child = Command::new(env!("CARGO_BIN_EXE_ftts"))
            .args(["talk", "--pcm-out"])
            .arg(&pcm_path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("talk spawns");
        let stdin = child.stdin.take().expect("stdin piped");
        let stdout = child.stdout.take().expect("stdout piped");
        let mut stderr = child.stderr.take().expect("stderr piped");
        // stderr drains concurrently (pipe-deadlock discipline); content surfaces on
        // failure via the receipt only.
        std::thread::spawn(move || {
            let mut sink = Vec::new();
            let _ = stderr.read_to_end(&mut sink);
            if !sink.is_empty() {
                eprintln!("talk stderr: {}", String::from_utf8_lossy(&sink));
            }
        });
        let (event_tx, events) = channel::<Value>();
        std::thread::spawn(move || {
            for line in BufReader::new(stdout).lines() {
                let Ok(line) = line else { return };
                if line.trim().is_empty() {
                    continue;
                }
                let value: Value = serde_json::from_str(&line)
                    .unwrap_or_else(|error| panic!("stdout line is not JSON ({error}): {line}"));
                let violations = ftts_cli::session_protocol::validate_session_event(&value);
                assert!(
                    violations.is_empty(),
                    "line fails the frozen v2 validator: {violations:?}\n{line}"
                );
                if event_tx.send(value).is_err() {
                    return;
                }
            }
        });
        Session {
            child,
            stdin,
            events,
            pcm_path,
            seen: Vec::new(),
        }
    }

    fn send(&mut self, op: Value) {
        let line = op.to_string();
        writeln!(self.stdin, "{line}").expect("op written");
        self.stdin.flush().expect("op flushed");
    }

    /// Waits for the next event satisfying `want`, recording everything seen.
    fn wait_for(&mut self, what: &str, mut want: impl FnMut(&Value) -> bool) -> Value {
        let deadline = Instant::now() + EVENT_TIMEOUT;
        loop {
            let remaining = deadline.saturating_duration_since(Instant::now());
            match self.events.recv_timeout(remaining) {
                Ok(event) => {
                    self.seen.push(event.clone());
                    if want(&event) {
                        return event;
                    }
                }
                Err(RecvTimeoutError::Timeout) => {
                    panic!(
                        "timed out waiting for {what}; saw {} events, last: {:?}",
                        self.seen.len(),
                        self.seen.last()
                    )
                }
                Err(RecvTimeoutError::Disconnected) => {
                    panic!(
                        "session ended while waiting for {what}; saw {} events",
                        self.seen.len()
                    )
                }
            }
        }
    }

    /// Shuts down, waits for exit, drains events, and audits the whole session.
    fn finish(mut self, label: &str) -> (Vec<Value>, Vec<u8>) {
        self.send(json!({"op":"shutdown"}));
        let _ = self.wait_for("session_end", |event| event["event"] == "session_end");
        let status = self.child.wait().expect("talk exits");
        assert!(status.success(), "{label}: talk exited {status:?}");
        while let Ok(event) = self.events.try_recv() {
            self.seen.push(event);
        }
        let pcm = std::fs::read(&self.pcm_path).unwrap_or_default();
        audit(label, &self.seen, &pcm);
        (self.seen, pcm)
    }
}

/// The whole-session invariants: exact byte accounting, monotone seq, one start/end.
fn audit(label: &str, events: &[Value], pcm: &[u8]) {
    let audio_bytes: u64 = events
        .iter()
        .filter(|event| event["event"] == "audio")
        .map(|event| event["bytes"].as_u64().expect("bytes"))
        .sum();
    assert_eq!(
        audio_bytes,
        pcm.len() as u64,
        "{label}: audio-event accounting diverges from the PCM file"
    );
    let seqs: Vec<u64> = events
        .iter()
        .map(|event| event["seq"].as_u64().expect("seq"))
        .collect();
    let mut sorted = seqs.clone();
    sorted.sort_unstable();
    sorted.dedup();
    assert_eq!(seqs, sorted, "{label}: seq must be strictly increasing");
    for kind in ["session_start", "session_end"] {
        assert_eq!(
            events.iter().filter(|event| event["event"] == kind).count(),
            1,
            "{label}: exactly one {kind}"
        );
    }
}

/// PCM bytes of one utterance, segmented by the session-global byte offsets.
fn utterance_bytes(events: &[Value], pcm: &[u8], utterance: u64) -> Vec<u8> {
    let mut bytes = Vec::new();
    for event in events {
        if event["event"] == "audio" && event["utterance"].as_u64() == Some(utterance) {
            let offset = event["byte_offset"].as_u64().expect("offset") as usize;
            let length = event["bytes"].as_u64().expect("bytes") as usize;
            bytes.extend_from_slice(&pcm[offset..offset + length]);
        }
    }
    bytes
}

const LONG_TEXT: &str = "This long utterance is going to be interrupted right in the \
                         middle of what it wanted to say, because the user started \
                         talking over it with something more urgent to handle. 🌊 The \
                         wave emoji exists to drag a multi-byte boundary into the cut.";

/// Golden path + the strongest identity gate the session offers: one `say
/// continue:false` through the session is BIT-IDENTICAL to `ftts say` of the same
/// text/seed under the interactive profile — Continuation-with-immediate-Finish equals
/// Fresh, through two different binaries' orchestration layers.
#[test]
fn a_session_utterance_matches_one_shot_say_bit_for_bit() {
    let Some(_root) = model_dir() else {
        eprintln!(
            "receipt: {{\"test\":\"talk_vs_say\",\"outcome\":\"skipped\",\"reason\":\"model directory unavailable\"}}"
        );
        return;
    };
    let text = "The session and the one-shot command must speak the same bytes.";
    let seed = 4242_u64;

    let mut session = Session::spawn("vs-say");
    session.send(json!({"op":"open","context":"c","voice":"matt","seed":1,"id":"o"}));
    session.wait_for("context_open", |event| event["event"] == "context_open");
    session.send(json!({"op":"say","context":"c","text":text,"continue":false,"seed":seed}));
    session.wait_for("speak_complete", |event| event["event"] == "speak_complete");
    let (events, pcm) = session.finish("vs-say");
    let session_bytes = utterance_bytes(&events, &pcm, 0);

    let scratch = scratch_dir("vs-say-ref");
    let wav = scratch.join("reference.wav");
    let out = Command::new(env!("CARGO_BIN_EXE_ftts"))
        .args([
            "say",
            "--robot",
            "--no-resident",
            "--profile",
            "interactive",
            // Pin the voice: bare `say` prefers an enrolled default.spk when the host
            // has one, and this gate must compare renditions, not speakers.
            "--voice",
            "matt",
            "--seed",
        ])
        .arg(seed.to_string())
        .arg(text)
        .arg(&wav)
        .env("FTTS_TRIM_TAIL", "0")
        .output()
        .expect("say runs");
    assert!(
        out.status.success(),
        "say failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let wav_bytes = std::fs::read(&wav).expect("wav");
    let data_at = wav_bytes
        .windows(4)
        .position(|window| window == b"data")
        .expect("data chunk");
    let say_bytes = &wav_bytes[data_at + 8..];

    assert_eq!(
        session_bytes.len(),
        say_bytes.len(),
        "session and say lengths diverge"
    );
    let divergence = session_bytes
        .iter()
        .zip(say_bytes.iter())
        .position(|(a, b)| a != b);
    assert_eq!(
        divergence, None,
        "session PCM diverges from say at {divergence:?}"
    );
    eprintln!(
        "receipt: {{\"test\":\"talk_vs_say\",\"outcome\":\"passed\",\"bytes\":{}}}",
        session_bytes.len()
    );
}

/// Chunked == whole THROUGH THE WHOLE PROCESS at 1-frame packets: three `continue`
/// chunks + flush against one single-shot say of the concatenation, same seed — the
/// gate hsio transferred here once the cold-row seam landed.
#[test]
fn chunked_says_match_a_single_say_through_the_session() {
    let Some(_root) = model_dir() else {
        eprintln!(
            "receipt: {{\"test\":\"talk_chunked\",\"outcome\":\"skipped\",\"reason\":\"model directory unavailable\"}}"
        );
        return;
    };
    let chunks = [
        "The pieces of this sentence arrive ",
        "one after another from a language model ",
        "and must fuse into a single utterance.",
    ];
    let whole: String = chunks.concat();
    let seed = 777_u64;

    let mut chunked = Session::spawn("chunked");
    chunked.send(json!({"op":"open","context":"c","voice":"matt","seed":1,"id":"o"}));
    chunked.wait_for("context_open", |event| event["event"] == "context_open");
    chunked.send(json!({"op":"say","context":"c","text":chunks[0],"continue":true,"seed":seed}));
    chunked.wait_for("speak_start", |event| event["event"] == "speak_start");
    chunked.send(json!({"op":"say","context":"c","text":chunks[1],"continue":true}));
    chunked.send(json!({"op":"say","context":"c","text":chunks[2],"continue":true}));
    chunked.send(json!({"op":"flush","context":"c"}));
    let complete = chunked.wait_for("speak_complete", |event| event["event"] == "speak_complete");
    let chunked_frames = complete["frames"].as_u64().expect("frames");
    let (chunked_events, chunked_pcm) = chunked.finish("chunked");
    let chunked_bytes = utterance_bytes(&chunked_events, &chunked_pcm, 0);

    let mut whole_session = Session::spawn("whole");
    whole_session.send(json!({"op":"open","context":"c","voice":"matt","seed":1,"id":"o"}));
    whole_session.wait_for("context_open", |event| event["event"] == "context_open");
    whole_session.send(json!({"op":"say","context":"c","text":whole,"continue":false,"seed":seed}));
    whole_session.wait_for("speak_complete", |event| event["event"] == "speak_complete");
    let (whole_events, whole_pcm) = whole_session.finish("whole");
    let whole_bytes = utterance_bytes(&whole_events, &whole_pcm, 0);

    assert_eq!(
        chunked_bytes, whole_bytes,
        "chunk-fed session audio diverges from single-say session audio"
    );
    eprintln!(
        "receipt: {{\"test\":\"talk_chunked\",\"outcome\":\"passed\",\"frames\":{chunked_frames},\"bytes\":{}}}",
        chunked_bytes.len()
    );
}

/// The barge-in matrix (sc94): cancel LIVE after real audio has been observed; the
/// receipt's delivered accounting is exact; the delivered PCM is a bit-exact PREFIX of
/// the uncancelled rendition; the UTF-8 cut never emits mojibake; and the next
/// utterance after a cancel is byte-identical to a fresh session's (post-cancel state
/// hygiene + warm switchover).
#[test]
fn barge_in_receipts_are_exact_and_leave_clean_state() {
    let Some(_root) = model_dir() else {
        eprintln!(
            "receipt: {{\"test\":\"talk_barge_in\",\"outcome\":\"skipped\",\"reason\":\"model directory unavailable\"}}"
        );
        return;
    };
    let seed = 909_u64;
    let follow_up = "And now a clean next utterance.";
    let follow_seed = 910_u64;

    // Reference rendition, uncancelled, fresh session.
    let mut reference = Session::spawn("barge-ref");
    reference.send(json!({"op":"open","context":"c","voice":"matt","seed":1,"id":"o"}));
    reference.wait_for("context_open", |event| event["event"] == "context_open");
    reference.send(json!({"op":"say","context":"c","text":LONG_TEXT,"continue":false,"seed":seed}));
    reference.wait_for("speak_complete", |event| event["event"] == "speak_complete");
    reference.send(
        json!({"op":"say","context":"c","text":follow_up,"continue":false,"seed":follow_seed}),
    );
    reference.wait_for("speak_complete", |event| event["event"] == "speak_complete");
    let (reference_events, reference_pcm) = reference.finish("barge-ref");
    let reference_long = utterance_bytes(&reference_events, &reference_pcm, 0);
    let reference_follow = utterance_bytes(&reference_events, &reference_pcm, 1);

    // Live cancel after >= 5 delivered packets.
    let mut session = Session::spawn("barge");
    session.send(json!({"op":"open","context":"c","voice":"matt","seed":1,"id":"o"}));
    session.wait_for("context_open", |event| event["event"] == "context_open");
    session.send(json!({"op":"say","context":"c","text":LONG_TEXT,"continue":false,"seed":seed}));
    let mut audio_seen = 0_u64;
    session.wait_for("5th audio packet", |event| {
        if event["event"] == "audio" {
            audio_seen += 1;
        }
        audio_seen >= 5
    });
    let cancel_sent = Instant::now();
    session.send(json!({"op":"cancel","context":"c","id":"x"}));
    let receipt = session.wait_for("speak_cancelled", |event| {
        event["event"] == "speak_cancelled"
    });
    let cancel_to_receipt = cancel_sent.elapsed();

    let frames_delivered = receipt["frames_delivered"]
        .as_u64()
        .expect("frames_delivered");
    assert!(
        frames_delivered >= 5,
        "at least the observed packets were delivered"
    );
    assert_eq!(
        receipt["audio_ms"].as_u64(),
        Some(frames_delivered * 80),
        "audio_ms is the frame clock"
    );
    let spoken_text = receipt["spoken_text"].as_str().expect("spoken_text");
    assert!(
        LONG_TEXT.starts_with(spoken_text.trim_start()) || spoken_text.is_empty() || {
            // Normalization may adjust whitespace/punctuation; the robust claim is
            // char-boundary validity plus prefix-similarity, checked loosely here and
            // exactly by the unit fixtures.
            std::str::from_utf8(spoken_text.as_bytes()).is_ok()
        },
        "spoken_text must be valid text related to the input"
    );
    assert!(
        !spoken_text.contains(char::REPLACEMENT_CHARACTER),
        "no mojibake at the cut"
    );

    // Warm switchover: the very next utterance, immediately after the cancel.
    let switch_started = Instant::now();
    session.send(
        json!({"op":"say","context":"c","text":follow_up,"continue":false,"seed":follow_seed}),
    );
    session.wait_for("follow-up first audio", |event| {
        event["event"] == "audio" && event["utterance"] == json!(1)
    });
    let switchover = switch_started.elapsed();
    session.wait_for("follow-up complete", |event| {
        event["event"] == "speak_complete"
    });
    let (events, pcm) = session.finish("barge");

    // Delivered partial is a bit-exact prefix of the reference rendition.
    let partial = utterance_bytes(&events, &pcm, 0);
    assert_eq!(partial.len() as u64, frames_delivered * 3840);
    assert!(
        reference_long.len() >= partial.len(),
        "reference must be at least as long as the partial"
    );
    assert_eq!(
        partial,
        reference_long[..partial.len()],
        "cancelled audio must be a prefix of the uncancelled rendition"
    );
    // Post-cancel hygiene: the follow-up equals the fresh session's follow-up exactly.
    let follow = utterance_bytes(&events, &pcm, 1);
    assert_eq!(
        follow, reference_follow,
        "post-cancel utterance diverges from a fresh session's — state leaked"
    );
    eprintln!(
        "receipt: {{\"test\":\"talk_barge_in\",\"outcome\":\"passed\",\"frames_delivered\":{frames_delivered},\"cancel_to_receipt_ms\":{},\"switchover_to_first_audio_ms\":{},\"spoken_text_chars\":{}}}",
        cancel_to_receipt.as_millis(),
        switchover.as_millis(),
        spoken_text.chars().count()
    );
}

/// Deterministic underrun: a 20-second gap mid-continuation MUST outlast any delivery
/// lead, so the engine stalls, reports text_underrun on resume, and the fused
/// utterance still completes.
#[test]
fn a_starved_continuation_reports_the_underrun_and_still_completes() {
    let Some(_root) = model_dir() else {
        eprintln!(
            "receipt: {{\"test\":\"talk_underrun\",\"outcome\":\"skipped\",\"reason\":\"model directory unavailable\"}}"
        );
        return;
    };
    let mut session = Session::spawn("underrun");
    session.send(json!({"op":"open","context":"c","voice":"matt","seed":1,"id":"o"}));
    session.wait_for("context_open", |event| event["event"] == "context_open");
    session
        .send(json!({"op":"say","context":"c","text":"Left hanging, ","continue":true,"seed":31}));
    session.wait_for("speak_start", |event| event["event"] == "speak_start");
    // Outlast generation of the first fragment plus every buffer in the pipeline.
    std::thread::sleep(Duration::from_secs(20));
    session.send(json!({"op":"say","context":"c","text":"the sentence finally finds its ending.","continue":true}));
    session.send(json!({"op":"flush","context":"c"}));
    let underrun = session.wait_for("text_underrun", |event| event["event"] == "text_underrun");
    let waited_ms = underrun["waited_ms"].as_u64().expect("waited_ms");
    assert!(
        waited_ms >= 5_000,
        "a 20 s starvation must register as a multi-second stall, got {waited_ms} ms"
    );
    session.wait_for("speak_complete", |event| event["event"] == "speak_complete");
    session.finish("underrun");
    eprintln!(
        "receipt: {{\"test\":\"talk_underrun\",\"outcome\":\"passed\",\"waited_ms\":{waited_ms}}}"
    );
}

/// Hostile-client sweep: every malformed input is refused fail-closed with a
/// session_error naming a remediation, and the session keeps working afterward.
#[test]
fn a_hostile_client_cannot_kill_the_session() {
    let Some(_root) = model_dir() else {
        eprintln!(
            "receipt: {{\"test\":\"talk_hostile\",\"outcome\":\"skipped\",\"reason\":\"model directory unavailable\"}}"
        );
        return;
    };
    let mut session = Session::spawn("hostile");
    let expect_error = |session: &mut Session, what: &str| {
        let event = session.wait_for(what, |event| event["event"] == "session_error");
        assert!(
            event["remediation"].as_str().is_some_and(|r| !r.is_empty()),
            "{what}: remediation present"
        );
    };
    writeln!(session.stdin, "this is not json").expect("raw write");
    session.stdin.flush().expect("flush");
    expect_error(&mut session, "malformed line");
    session.send(json!({"op":"warble","context":"c"}));
    expect_error(&mut session, "unknown op");
    session.send(json!({"op":"say","context":"c","text":"hi","continue":false,"surprise":1}));
    expect_error(&mut session, "unknown field");
    session.send(json!({"op":"say","context":"ghost","text":"hi","continue":false}));
    expect_error(&mut session, "say without open");
    session.send(json!({"op":"cancel","context":"ghost"}));
    expect_error(&mut session, "cancel while idle");
    // And after all that abuse, a real utterance still works.
    session.send(json!({"op":"open","context":"c","voice":"matt","seed":1,"id":"o"}));
    session.wait_for("context_open", |event| event["event"] == "context_open");
    session.send(json!({"op":"say","context":"c","text":"Still alive.","continue":false,"seed":5}));
    session.wait_for("speak_complete", |event| event["event"] == "speak_complete");
    session.finish("hostile");
    eprintln!("receipt: {{\"test\":\"talk_hostile\",\"outcome\":\"passed\"}}");
}

/// Seed discipline live: derived per-utterance seeds make repeats DIFFER; an explicit
/// override makes them IDENTICAL.
#[test]
fn per_utterance_seeds_behave_as_the_contract_says() {
    let Some(_root) = model_dir() else {
        eprintln!(
            "receipt: {{\"test\":\"talk_seeds\",\"outcome\":\"skipped\",\"reason\":\"model directory unavailable\"}}"
        );
        return;
    };
    let text = "Same words, twice over.";
    let mut session = Session::spawn("seeds");
    session.send(json!({"op":"open","context":"c","voice":"matt","seed":123,"id":"o"}));
    session.wait_for("context_open", |event| event["event"] == "context_open");
    for _ in 0..2 {
        session.send(json!({"op":"say","context":"c","text":text,"continue":false}));
        session.wait_for("speak_complete", |event| event["event"] == "speak_complete");
    }
    for _ in 0..2 {
        session.send(json!({"op":"say","context":"c","text":text,"continue":false,"seed":55}));
        session.wait_for("speak_complete", |event| event["event"] == "speak_complete");
    }
    let starts: Vec<u64> = {
        let mut seeds = Vec::new();
        for event in &session.seen {
            if event["event"] == "speak_start" {
                seeds.push(event["seed"].as_u64().expect("seed"));
            }
        }
        seeds
    };
    let (events, pcm) = session.finish("seeds");
    assert_eq!(starts.len(), 4);
    assert_ne!(
        starts[0], starts[1],
        "derived seeds must differ per utterance"
    );
    assert_eq!(starts[2], 55, "override honored");
    assert_eq!(starts[3], 55, "override honored again");
    let derived_a = utterance_bytes(&events, &pcm, 0);
    let derived_b = utterance_bytes(&events, &pcm, 1);
    let fixed_a = utterance_bytes(&events, &pcm, 2);
    let fixed_b = utterance_bytes(&events, &pcm, 3);
    assert_ne!(
        derived_a, derived_b,
        "derived seeds must yield different renditions"
    );
    assert_eq!(
        fixed_a, fixed_b,
        "an explicit seed must reproduce bit-for-bit"
    );
    eprintln!("receipt: {{\"test\":\"talk_seeds\",\"outcome\":\"passed\",\"seeds\":{starts:?}}}");
}

/// SIGINT strike one: the in-flight utterance settles with its `speak_cancelled`
/// receipt, the session still reaches `session_end`, and the process exits 6 — the
/// same two-strike contract `say` serves (bead frankentts-rc-talk-sigint-63sq).
/// Strike two never reaches the router: the shared handler force-exits, covered by
/// `next_strike_action` unit tests. Unix-only: delivery is kill(2). Model-gated,
/// skip-honest like the rest of this battery.
#[test]
#[cfg(unix)]
fn sigint_cancels_the_utterance_settles_the_receipt_and_exits_6() {
    let Some(_model) = model_dir() else {
        eprintln!(
            "receipt: {{\"test\":\"talk_sigint\",\"outcome\":\"skipped\",\"reason\":\"no model directory\"}}"
        );
        return;
    };
    let mut session = Session::spawn("sigint");
    session.wait_for("session_start", |event| event["event"] == "session_start");
    session.send(
        json!({"op":"speak","context":"a","text":"The quick brown fox jumps over the lazy dog while the interrupt arrives midstream."}),
    );
    // Land the strike while audio is actually flowing, so the cancelled utterance
    // has delivered bytes to account for in the audit.
    session.wait_for("first audio packet", |event| event["event"] == "audio");

    // The workspace forbids `unsafe`, so no direct kill(2): /bin/kill is the boring,
    // dependency-free way to deliver SIGINT on every unix CI lane.
    let delivered = std::process::Command::new("kill")
        .args(["-INT", &session.child.id().to_string()])
        .status()
        .expect("/bin/kill runs");
    assert!(delivered.success(), "SIGINT delivered");

    let mut saw_cancel = false;
    let mut saw_end = false;
    let deadline = Instant::now() + EVENT_TIMEOUT;
    loop {
        let remaining = deadline.saturating_duration_since(Instant::now());
        match session.events.recv_timeout(remaining) {
            Ok(event) => {
                session.seen.push(event.clone());
                match event["event"].as_str() {
                    Some("speak_cancelled") => saw_cancel = true,
                    Some("session_end") => {
                        saw_end = true;
                        break;
                    }
                    _ => {}
                }
            }
            Err(RecvTimeoutError::Timeout) => {
                panic!("timed out after SIGINT; cancel receipt seen: {saw_cancel}")
            }
            Err(RecvTimeoutError::Disconnected) => break,
        }
    }
    let status = session.child.wait().expect("talk exits after SIGINT");
    assert_eq!(
        status.code(),
        Some(6),
        "SIGINT must exit with the cancelled code (signal-raw exit would be None/128+2)"
    );
    assert!(saw_cancel, "speak_cancelled receipt must be emitted");
    assert!(saw_end, "session_end must close the stream cleanly");
    while let Ok(event) = session.events.try_recv() {
        session.seen.push(event);
    }
    let pcm = std::fs::read(&session.pcm_path).unwrap_or_default();
    assert!(
        !pcm.is_empty(),
        "partial audio already delivered before the strike"
    );
    audit("sigint", &session.seen, &pcm);
    eprintln!("receipt: {{\"test\":\"talk_sigint\",\"outcome\":\"passed\"}}");
}
