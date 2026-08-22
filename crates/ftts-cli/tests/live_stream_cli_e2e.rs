//! Live CLI emission contract (bead `frankentts-xqqa`), against the real binary and model.
//!
//! Model-gated: without a complete model directory each test reports the skip and passes.
//! With the model these prove, on this machine:
//!
//! 1. in robot file mode, `audio_chunk` (and throttled `frame`) events are emitted DURING
//!    synthesis — the first `audio_chunk` line precedes `stage{synthesis,end}` in the
//!    event stream, which is proof by ordering on a single stream, no timestamps needed;
//! 2. byte accounting is exact: with tail-trim off, `sum(audio_chunk.bytes)` equals
//!    `run_complete.audio_bytes`;
//! 3. every emitted line survives the strict-closed schema validator, and exactly one
//!    terminal event closes the run — the skill's fail-closed consumer contract;
//! 4. `--stream raw` bypasses the resident (no daemon state file appears) and its stdout
//!    byte stream is byte-identical to the file-mode WAV data section at the same seed;
//! 5. the raw-mode event stream on stderr shows the same during-synthesis ordering.
//!
//! Receipts: each test prints `receipt: {...}` NDJSON to stderr with measured values.

use std::io::Read;
use std::path::PathBuf;
use std::process::{Command, Stdio};

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

const TEXT: &str =
    "The live emission path must stream packets while the model is still speaking, not afterward.";

/// A per-test scratch directory under the system temp root, std-only (no tempfile dep:
/// the workspace lock is pinned). Never deleted — small, uniquely named, disposable.
fn scratch_dir(label: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "ftts-live-e2e-{label}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos())
    ));
    std::fs::create_dir_all(&dir).expect("scratch dir");
    dir
}

struct RunOutput {
    stdout: Vec<u8>,
    stderr: Vec<u8>,
    status: std::process::ExitStatus,
}

/// Spawn `ftts` with both pipes drained concurrently (pipe-deadlock discipline).
fn run_ftts(args: &[&str], envs: &[(&str, &str)]) -> RunOutput {
    let mut command = Command::new(env!("CARGO_BIN_EXE_ftts"));
    command
        .args(args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    for (key, value) in envs {
        command.env(key, value);
    }
    let mut child = command.spawn().expect("ftts spawns");
    let mut stdout_pipe = child.stdout.take().expect("stdout piped");
    let mut stderr_pipe = child.stderr.take().expect("stderr piped");
    let stdout_thread = std::thread::spawn(move || {
        let mut buffer = Vec::new();
        stdout_pipe.read_to_end(&mut buffer).expect("stdout drains");
        buffer
    });
    let mut stderr = Vec::new();
    stderr_pipe.read_to_end(&mut stderr).expect("stderr drains");
    let stdout = stdout_thread.join().expect("stdout thread");
    let status = child.wait().expect("ftts exits");
    RunOutput {
        stdout,
        stderr,
        status,
    }
}

/// Parse an NDJSON byte stream into JSON values, asserting every line is valid JSON and
/// survives the crate's strict-closed event validator.
fn parse_and_validate(stream: &[u8], label: &str) -> Vec<serde_json::Value> {
    let text = String::from_utf8(stream.to_vec())
        .unwrap_or_else(|_| panic!("{label} is not UTF-8 NDJSON"));
    let mut events = Vec::new();
    for (number, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let value: serde_json::Value = serde_json::from_str(line)
            .unwrap_or_else(|error| panic!("{label} line {number} is not JSON ({error}): {line}"));
        let violations = ftts_cli::validate_event(&value);
        assert!(
            violations.is_empty(),
            "{label} line {number} fails the strict-closed validator: {violations:?}\n{line}"
        );
        events.push(value);
    }
    events
}

fn event_name(event: &serde_json::Value) -> &str {
    event.get("event").and_then(|v| v.as_str()).unwrap_or("")
}

fn is_stage(event: &serde_json::Value, name: &str, state: &str) -> bool {
    event_name(event) == "stage"
        && event.get("name").and_then(|v| v.as_str()) == Some(name)
        && event.get("state").and_then(|v| v.as_str()) == Some(state)
}

fn assert_single_terminal(events: &[serde_json::Value], label: &str) {
    let terminals = events
        .iter()
        .filter(|event| matches!(event_name(event), "run_complete" | "run_error"))
        .count();
    assert_eq!(terminals, 1, "{label}: expected exactly one terminal event");
}

#[test]
fn robot_file_mode_emits_audio_chunks_during_synthesis() {
    let Some(_root) = model_dir() else {
        eprintln!(
            "receipt: {{\"test\":\"live_file_mode\",\"outcome\":\"skipped\",\"reason\":\"model directory unavailable\"}}"
        );
        return;
    };
    let scratch = scratch_dir("file");
    let out = scratch.join("live.wav");
    let resident_dir = scratch.join("resident");
    std::fs::create_dir_all(&resident_dir).expect("resident dir");
    let run = run_ftts(
        &[
            "say",
            "--robot",
            "--seed",
            "7",
            "--no-resident",
            TEXT,
            out.to_str().expect("utf-8 path"),
        ],
        &[
            ("FTTS_RESIDENT_DIR", resident_dir.to_str().expect("utf-8")),
            // Exact byte accounting needs the tail trim off: chunk events describe samples
            // handed to the writer, and trimming may withhold up to a quarter second.
            ("FTTS_TRIM_TAIL", "0"),
        ],
    );
    assert!(
        run.status.success(),
        "say failed: {}",
        String::from_utf8_lossy(&run.stderr)
    );
    let events = parse_and_validate(&run.stdout, "robot stdout");
    assert_single_terminal(&events, "file mode");

    let first_chunk = events
        .iter()
        .position(|event| event_name(event) == "audio_chunk")
        .expect("at least one audio_chunk event");
    let synthesis_end = events
        .iter()
        .position(|event| is_stage(event, "synthesis", "end"))
        .expect("a synthesis end stage");
    assert!(
        first_chunk < synthesis_end,
        "first audio_chunk (line {first_chunk}) must precede stage{{synthesis,end}} (line {synthesis_end}) — live emission, not post-hoc"
    );
    let output_begin = events
        .iter()
        .position(|event| is_stage(event, "output", "begin"))
        .expect("an output begin stage");
    assert!(
        output_begin < first_chunk && output_begin < synthesis_end,
        "output begins at the first packet, inside the synthesis window"
    );

    let chunk_bytes: u64 = events
        .iter()
        .filter(|event| event_name(event) == "audio_chunk")
        .map(|event| {
            event
                .get("bytes")
                .and_then(serde_json::Value::as_u64)
                .expect("bytes")
        })
        .sum();
    let complete = events
        .iter()
        .find(|event| event_name(event) == "run_complete")
        .expect("run_complete");
    let audio_bytes = complete
        .get("audio_bytes")
        .and_then(serde_json::Value::as_u64)
        .expect("audio_bytes");
    assert_eq!(
        chunk_bytes, audio_bytes,
        "with tail-trim off, audio_chunk byte accounting must equal the file's bytes"
    );

    let frame_events: Vec<_> = events
        .iter()
        .filter(|event| event_name(event) == "frame")
        .collect();
    assert!(
        !frame_events.is_empty(),
        "a multi-second utterance must produce at least one throttled frame event"
    );
    for event in &frame_events {
        assert!(
            event
                .get("index")
                .and_then(serde_json::Value::as_u64)
                .is_some()
        );
        assert!(
            event
                .get("elapsed_ms")
                .and_then(serde_json::Value::as_u64)
                .is_some()
        );
        assert!(
            event.get("total_estimate").is_some(),
            "total_estimate present (u64 or null)"
        );
    }

    eprintln!(
        "receipt: {{\"test\":\"live_file_mode\",\"outcome\":\"passed\",\"events\":{},\"audio_chunks\":{},\"frame_events\":{},\"first_chunk_line\":{first_chunk},\"synthesis_end_line\":{synthesis_end},\"audio_bytes\":{audio_bytes}}}",
        events.len(),
        events
            .iter()
            .filter(|e| event_name(e) == "audio_chunk")
            .count(),
        frame_events.len()
    );
}

#[test]
fn raw_mode_streams_live_bypasses_resident_and_matches_file_output() {
    let Some(_root) = model_dir() else {
        eprintln!(
            "receipt: {{\"test\":\"live_raw_mode\",\"outcome\":\"skipped\",\"reason\":\"model directory unavailable\"}}"
        );
        return;
    };
    let scratch = scratch_dir("raw");
    let resident_dir = scratch.join("resident");
    std::fs::create_dir_all(&resident_dir).expect("resident dir");

    // Raw run WITHOUT --no-resident: the bypass rule itself is under test.
    let raw = run_ftts(
        &["say", "--robot", "--seed", "7", "--stream", "raw", TEXT],
        &[
            ("FTTS_RESIDENT_DIR", resident_dir.to_str().expect("utf-8")),
            ("FTTS_TRIM_TAIL", "0"),
        ],
    );
    assert!(
        raw.status.success(),
        "raw say failed: {}",
        String::from_utf8_lossy(&raw.stderr)
    );
    assert!(!raw.stdout.is_empty(), "raw stdout carries PCM");
    assert!(
        raw.stdout.len().is_multiple_of(2),
        "raw stream is whole s16le samples"
    );

    // The bypass proof: a raw-streaming run must not have spawned or consulted a resident
    // daemon, so its state directory stays empty.
    let leftovers: Vec<_> = std::fs::read_dir(&resident_dir)
        .expect("resident dir readable")
        .collect();
    assert!(
        leftovers.is_empty(),
        "raw streaming must bypass the resident; found state files: {leftovers:?}"
    );

    // Events live on stderr in raw mode; same during-synthesis ordering proof.
    let events = parse_and_validate(&raw.stderr, "raw-mode stderr");
    assert_single_terminal(&events, "raw mode");
    let first_chunk = events
        .iter()
        .position(|event| event_name(event) == "audio_chunk")
        .expect("audio_chunk events on stderr");
    let synthesis_end = events
        .iter()
        .position(|event| is_stage(event, "synthesis", "end"))
        .expect("synthesis end stage");
    assert!(
        first_chunk < synthesis_end,
        "raw-mode chunks are emitted during synthesis"
    );

    // Byte identity against the file path at the same seed (tail-trim off on both sides).
    let out = scratch.join("reference.wav");
    let file_run = run_ftts(
        &[
            "say",
            "--robot",
            "--seed",
            "7",
            "--no-resident",
            TEXT,
            out.to_str().expect("utf-8 path"),
        ],
        &[
            ("FTTS_RESIDENT_DIR", resident_dir.to_str().expect("utf-8")),
            ("FTTS_TRIM_TAIL", "0"),
        ],
    );
    assert!(file_run.status.success(), "file say failed");
    let wav = std::fs::read(&out).expect("reference wav");
    // Locate the `data` chunk rather than assuming a 44-byte header.
    let data_at = wav
        .windows(4)
        .position(|window| window == b"data")
        .expect("wav data chunk");
    let data = &wav[data_at + 8..];
    assert_eq!(
        data.len(),
        raw.stdout.len(),
        "raw stream length diverges from the file's data section"
    );
    let first_divergence = data.iter().zip(raw.stdout.iter()).position(|(a, b)| a != b);
    assert_eq!(
        first_divergence, None,
        "raw stream bytes diverge from the file's data section at {first_divergence:?}"
    );

    eprintln!(
        "receipt: {{\"test\":\"live_raw_mode\",\"outcome\":\"passed\",\"pcm_bytes\":{},\"events\":{},\"first_chunk_line\":{first_chunk},\"synthesis_end_line\":{synthesis_end}}}",
        raw.stdout.len(),
        events.len()
    );
}

/// `--profile interactive` reaches the codec worker: the first audio_chunk is ONE frame
/// (80 ms, packet_frames "1"), delivered during synthesis. This is the wiring the profile
/// contract promised and the ~240 ms TTFA lever (frankentts-6xcf).
#[test]
fn interactive_profile_delivers_one_frame_packets() {
    let Some(_root) = model_dir() else {
        eprintln!(
            "receipt: {{\"test\":\"interactive_profile\",\"outcome\":\"skipped\",\"reason\":\"model directory unavailable\"}}"
        );
        return;
    };
    let scratch = scratch_dir("interactive");
    let out = scratch.join("interactive.wav");
    let resident_dir = scratch.join("resident");
    std::fs::create_dir_all(&resident_dir).expect("resident dir");
    let run = run_ftts(
        &[
            "say",
            "--robot",
            "--seed",
            "7",
            "--no-resident",
            "--profile",
            "interactive",
            TEXT,
            out.to_str().expect("utf-8 path"),
        ],
        &[
            ("FTTS_RESIDENT_DIR", resident_dir.to_str().expect("utf-8")),
            ("FTTS_TRIM_TAIL", "0"),
        ],
    );
    assert!(
        run.status.success(),
        "interactive say failed: {}",
        String::from_utf8_lossy(&run.stderr)
    );
    let events = parse_and_validate(&run.stdout, "interactive stdout");
    assert_single_terminal(&events, "interactive");
    let chunks: Vec<_> = events
        .iter()
        .filter(|event| event_name(event) == "audio_chunk")
        .collect();
    let first = chunks.first().expect("audio chunks present");
    assert_eq!(
        first.get("duration_ms").and_then(serde_json::Value::as_u64),
        Some(80),
        "interactive first packet must be one 80 ms frame"
    );
    assert_eq!(
        first
            .get("packet_frames")
            .and_then(serde_json::Value::as_str),
        Some("1"),
        "interactive packet_frames metadata"
    );
    // Every non-tail chunk is one frame; ordering proof as in the balanced case.
    for chunk in &chunks[..chunks.len().saturating_sub(1)] {
        assert_eq!(
            chunk.get("duration_ms").and_then(serde_json::Value::as_u64),
            Some(80)
        );
    }
    let first_chunk = events
        .iter()
        .position(|event| event_name(event) == "audio_chunk")
        .expect("chunk");
    let synthesis_end = events
        .iter()
        .position(|event| is_stage(event, "synthesis", "end"))
        .expect("synthesis end");
    assert!(
        first_chunk < synthesis_end,
        "interactive chunks stream during synthesis"
    );
    eprintln!(
        "receipt: {{\"test\":\"interactive_profile\",\"outcome\":\"passed\",\"chunks\":{},\"first_duration_ms\":80}}",
        chunks.len()
    );
}
