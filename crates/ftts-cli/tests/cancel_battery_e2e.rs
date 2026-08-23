//! Cancellation e2e battery (bead `frankentts-9t5v`): prove the cancelled-run contract
//! end to end, with real signals, on the live streaming path.
//!
//! Model-gated: without a complete model directory each test reports the skip and passes.
//! With the model these prove, on this machine:
//!
//! 1. frame-boundary cancel at the library level: nothing lost or duplicated across the
//!    boundary, and the documented `Cancelled` error class comes back;
//! 2. partial-WAV validity under a real SIGINT in file mode — parseable RIFF whose data
//!    size equals the streamed `audio_chunk` accounting, plus the pinned zero-sample
//!    prefill-cancel behavior;
//! 3. raw-mode cancel: stdout bytes == stderr accounting, packet-aligned, schema-valid
//!    events to the last line, exit code 6;
//! 4. SIGINT-to-exit latency asserted against a deliberately generous 2 s bound (fails
//!    only on a genuine hang), measured value in the receipt;
//! 5. compressed-format cancel: conversion skipped, staging WAV kept and named;
//! 6. resident-enabled cancel: client exits 6, daemon survives and serves warm.
//!
//! The library-level storm watchdog lives beside its sibling in `ftts-core`
//! (`cancel_storm_across_many_utterances_without_deadlock`).
//!
//! Receipts: each test prints `receipt: {...}` NDJSON with measured values.

use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

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

/// Long enough that a cancel triggered on the first decoded packet always lands
/// mid-generation (>15 s of speech at the production rate).
const LONG_TEXT: &str = "Please call Stella. Ask her to bring these things with her from the store: \
six spoons of fresh snow peas, five thick slabs of blue cheese, and maybe a snack for her \
brother Bob. We also need a small plastic snake and a big toy frog for the kids. She can scoop \
these things into three red bags, and we will go meet her Wednesday at the train station. \
When the sunlight strikes raindrops in the air, they act as a prism and form a rainbow.";

fn scratch_dir(label: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "ftts-cancel-battery-{label}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos())
    ));
    std::fs::create_dir_all(&dir).expect("scratch dir");
    dir
}

/// Spawn `ftts` with BOTH streams redirected to files: signal cases need to poll the
/// event stream while the child runs, and files make the final assertions byte-exact
/// without drain threads racing the signal.
fn spawn_to_files(
    args: &[&str],
    envs: &[(&str, &str)],
    stdout_path: &Path,
    stderr_path: &Path,
) -> Child {
    let stdout = std::fs::File::create(stdout_path).expect("stdout file");
    let stderr = std::fs::File::create(stderr_path).expect("stderr file");
    let mut command = Command::new(env!("CARGO_BIN_EXE_ftts"));
    command
        .args(args)
        .stdout(Stdio::from(stdout))
        .stderr(Stdio::from(stderr));
    for (key, value) in envs {
        command.env(key, value);
    }
    command.spawn().expect("ftts spawns")
}

/// Poll a growing event file until `needle` appears (or timeout).
fn wait_for_marker(path: &Path, needle: &str, timeout: Duration) -> bool {
    let deadline = Instant::now() + timeout;
    let mut buffer = String::new();
    while Instant::now() < deadline {
        buffer.clear();
        if std::fs::File::open(path)
            .and_then(|mut file| file.read_to_string(&mut buffer))
            .is_ok()
            && buffer.contains(needle)
        {
            return true;
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    false
}

/// Send SIGINT and reap the child, returning wall time from signal to exit.
#[cfg(unix)]
fn sigint_and_wait(mut child: Child) -> (std::process::ExitStatus, Duration) {
    use std::io::Write;
    let sent = Instant::now();
    let pid = child.id();
    // `kill -INT <pid>` — the pinned delivery path; no extra dependency in tests.
    let mut killer = Command::new("kill")
        .args(["-INT", &pid.to_string()])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("kill spawns");
    let _ = killer.wait();
    let _ = std::io::stdout().flush();
    let status = child.wait().expect("ftts exits");
    (status, sent.elapsed())
}

/// Parse an NDJSON file into JSON values, asserting every line is valid JSON and
/// survives the crate's strict-closed event validator.
fn parse_and_validate(path: &Path, label: &str) -> Vec<serde_json::Value> {
    let text =
        std::fs::read_to_string(path).unwrap_or_else(|error| panic!("{label} unreadable: {error}"));
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

fn terminal_event(events: &[serde_json::Value], label: &str) -> serde_json::Value {
    let terminals: Vec<&serde_json::Value> = events
        .iter()
        .filter(|event| matches!(event_name(event), "run_complete" | "run_error"))
        .collect();
    assert_eq!(
        terminals.len(),
        1,
        "{label}: expected exactly one terminal event, got {}",
        terminals.len()
    );
    terminals[0].clone()
}

/// Sum of `audio_chunk.bytes` across the event stream — what consumers use for
/// accounting, and what the artifact must match exactly.
fn chunk_bytes(events: &[serde_json::Value]) -> u64 {
    events
        .iter()
        .filter(|event| event_name(event) == "audio_chunk")
        .filter_map(|event| event.get("bytes").and_then(|v| v.as_u64()))
        .sum()
}

/// Validate a WAV artifact: RIFF/WAVE magic, a `data` chunk whose size matches the
/// remaining file exactly (no trailing garbage). Returns the data size in bytes.
fn validated_wav_data_size(path: &Path, label: &str) -> u64 {
    let bytes = std::fs::read(path).unwrap_or_else(|error| panic!("{label} unreadable: {error}"));
    assert!(bytes.len() >= 44, "{label}: shorter than a WAV header");
    assert_eq!(&bytes[..4], b"RIFF", "{label}: missing RIFF magic");
    assert_eq!(&bytes[8..12], b"WAVE", "{label}: missing WAVE magic");
    let mut offset = 12usize;
    while offset + 8 <= bytes.len() {
        let id = &bytes[offset..offset + 4];
        let size = u32::from_le_bytes(
            bytes[offset + 4..offset + 8]
                .try_into()
                .expect("size field"),
        );
        if id == b"data" {
            let data_start = offset + 8;
            assert_eq!(
                bytes.len(),
                data_start + size as usize,
                "{label}: trailing garbage or short data chunk (file {}, data claims {})",
                bytes.len(),
                size
            );
            return u64::from(size);
        }
        offset += 8 + size as usize + (size as usize & 1);
    }
    panic!("{label}: no data chunk found");
}

// ---------------------------------------------------------------- case 1

/// Frame-boundary cancel, library level: a sink trips the shared token after K packets;
/// every frame the generator produced reached the sink exactly once, and the error is
/// the documented Cancelled class.
#[test]
fn frame_boundary_cancel_loses_nothing_across_the_boundary() {
    use ftts_cli::synth::{self, LoadedModel, ModelBundle, PcmPacketSink};

    struct TripAfterK {
        after_packets: usize,
        seen: std::sync::Arc<std::sync::atomic::AtomicUsize>,
        frames_seen: std::sync::Arc<std::sync::atomic::AtomicUsize>,
        token: ftts_core::CancellationToken,
        tripped: std::sync::Arc<std::sync::atomic::AtomicBool>,
    }
    impl PcmPacketSink for TripAfterK {
        fn deliver(&mut self, _samples: &[f32], frames: usize) -> Result<(), ftts_cli::FttsError> {
            self.seen.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            self.frames_seen
                .fetch_add(frames, std::sync::atomic::Ordering::Relaxed);
            if self.seen.load(std::sync::atomic::Ordering::Relaxed) >= self.after_packets
                && !self
                    .tripped
                    .swap(true, std::sync::atomic::Ordering::Relaxed)
            {
                self.token.cancel();
            }
            Ok(())
        }
    }

    let Some(root) = model_dir() else {
        eprintln!(
            "receipt: {{\"test\":\"frame_boundary\",\"outcome\":\"skipped\",\"reason\":\"model directory unavailable\"}}"
        );
        return;
    };
    let bundle = ModelBundle::resolve(&root).expect("bundle resolves");
    let voice = bundle.root.join("default.spk");
    if !voice.is_file() {
        eprintln!(
            "receipt: {{\"test\":\"frame_boundary\",\"outcome\":\"skipped\",\"reason\":\"no default.spk enrolled\"}}"
        );
        return;
    }
    let loaded = LoadedModel::load(&bundle).expect("model loads");
    let speaker = synth::read_speaker_vector(&voice).expect("speaker vector reads");
    let engine = ftts_core::TtsEngine::from_process_environment().expect("engine starts");

    let token = ftts_core::CancellationToken::new();
    let frames_generated = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let progress = frames_generated.clone();
    let observer = move |event: ftts_core::SynthesisEvent| {
        if matches!(event, ftts_core::SynthesisEvent::FrameProgress { .. }) {
            progress.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
    };
    let packets_seen = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let frames_delivered = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let tripped = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let mut sink = TripAfterK {
        after_packets: 3,
        seen: packets_seen.clone(),
        frames_seen: frames_delivered.clone(),
        token: token.clone(),
        tripped,
    };
    let request = ftts_core::SynthesisRequest::new(LONG_TEXT.to_owned());
    let outcome: Result<ftts_cli::synth::SynthesizedAudio, ftts_cli::FttsError> = synth::synthesize(
        &loaded,
        &engine,
        &request,
        &speaker,
        0,
        &token,
        &observer,
        4,
        None,
        Some(&mut sink),
    );

    let error = match outcome {
        Ok(audio) => panic!(
            "a tripped token must cancel the run; got {} frames instead",
            audio.frames
        ),
        Err(error) => error,
    };
    assert!(
        matches!(error, ftts_cli::FttsError::Cancelled(_)),
        "expected the Cancelled class, got: {error:?}"
    );
    let generated = frames_generated.load(std::sync::atomic::Ordering::Relaxed);
    let delivered_frames = frames_delivered.load(std::sync::atomic::Ordering::Relaxed);
    assert!(
        delivered_frames >= 12,
        "sink tripped after 3 four-frame packets; got {delivered_frames} frames"
    );
    assert_eq!(
        generated, delivered_frames,
        "frames generated must equal frames the codec worker delivered — none lost or duplicated"
    );
    eprintln!(
        "receipt: {{\"test\":\"frame_boundary\",\"outcome\":\"passed\",\"frames\":{delivered_frames},\"packets\":{}}}",
        packets_seen.load(std::sync::atomic::Ordering::Relaxed)
    );
}

// ---------------------------------------------------------------- case 2 + 4

/// Real SIGINT in file mode, mid-generation: exit 6, single cancelled terminal event,
/// and a parseable partial WAV whose data size equals the streamed accounting.
/// Also asserts the 2 s SIGINT-to-exit latency bound and logs the measured value.
// SIGINT semantics are unix-only; the Windows cancel path is bead frankentts-37hv.
#[cfg(unix)]
#[test]
fn file_mode_sigint_leaves_a_valid_partial_wav_and_exits_promptly() {
    if model_dir().is_none() {
        eprintln!(
            "receipt: {{\"test\":\"partial_wav\",\"outcome\":\"skipped\",\"reason\":\"model directory unavailable\"}}"
        );
        return;
    }
    let dir = scratch_dir("file-mode");
    let out = dir.join("partial.wav");
    let stdout_path = dir.join("events.ndjson");
    let stderr_path = dir.join("noise.txt");
    let child = spawn_to_files(
        &[
            "say",
            "--no-resident",
            "--seed",
            "7",
            LONG_TEXT,
            out.to_str().expect("utf8 out"),
        ],
        &[],
        &stdout_path,
        &stderr_path,
    );
    assert!(
        wait_for_marker(
            &stdout_path,
            "\"event\":\"audio_chunk\"",
            Duration::from_secs(300)
        ),
        "first packet never arrived; aborting the cancel point"
    );
    let (status, latency) = sigint_and_wait(child);
    let code = status.code().unwrap_or(-1);
    assert_eq!(code, 6, "cancelled run must exit 6, got {code}");

    // The catalogue puts audio_chunk on stdout but run_error on STDERR, on both
    // stream shapes — the terminal event lives on the error stream.
    let events = parse_and_validate(&stdout_path, "robot events");
    let errors = parse_and_validate(&stderr_path, "robot stderr");
    let terminal = terminal_event(&errors, "file mode stderr");
    assert_eq!(
        event_name(&terminal),
        "run_error",
        "terminal must be run_error"
    );
    assert_eq!(terminal["kind"], "cancelled");
    assert_eq!(terminal["exit_code"], 6);

    let accounted = chunk_bytes(&events);
    let data_size = validated_wav_data_size(&out, "partial WAV");
    assert_eq!(
        data_size, accounted,
        "WAV data size must equal sum(audio_chunk.bytes): artifact {data_size}, events {accounted}"
    );
    // Latency bound: generous by design (roughly 10x the theoretical one-frame +
    // finalization budget); fails only on a genuine hang, never on load.
    assert!(
        latency < Duration::from_secs(2),
        "SIGINT-to-exit took {latency:?}; the contract bounds it at 2 s"
    );
    eprintln!(
        "receipt: {{\"test\":\"partial_wav\",\"outcome\":\"passed\",\"exit_code\":6,\"wav_data_bytes\":{data_size},\"sigint_to_exit_ms\":{}}}",
        latency.as_millis()
    );
}

/// Prefill-phase cancel (signal during model load / before the first frame): pinned
/// behavior is an EMPTY-but-valid WAV, exit 6, kind "cancelled" — never a torn file.
// SIGINT semantics are unix-only; the Windows cancel path is bead frankentts-37hv.
#[cfg(unix)]
#[test]
fn prefill_phase_sigint_pins_the_zero_sample_artifact() {
    if model_dir().is_none() {
        eprintln!(
            "receipt: {{\"test\":\"prefill_cancel\",\"outcome\":\"skipped\",\"reason\":\"model directory unavailable\"}}"
        );
        return;
    }
    let dir = scratch_dir("prefill");
    let out = dir.join("prefill.wav");
    let stdout_path = dir.join("events.ndjson");
    let stderr_path = dir.join("noise.txt");
    let child = spawn_to_files(
        &[
            "say",
            "--no-resident",
            LONG_TEXT,
            out.to_str().expect("utf8 out"),
        ],
        &[],
        &stdout_path,
        &stderr_path,
    );
    // Cancel as soon as synthesis OPENS — before any packet can decode.
    assert!(
        wait_for_marker(
            &stdout_path,
            "\"name\":\"synthesis\"",
            Duration::from_secs(300)
        ),
        "synthesis stage never opened"
    );
    let (status, _latency) = sigint_and_wait(child);
    assert_eq!(status.code().unwrap_or(-1), 6);

    let events = parse_and_validate(&stdout_path, "prefill events");
    let terminal = terminal_event(
        &parse_and_validate(&stderr_path, "prefill stderr"),
        "prefill",
    );
    assert_eq!(terminal["kind"], "cancelled");
    let data_size = validated_wav_data_size(&out, "prefill WAV");
    assert_eq!(
        data_size,
        chunk_bytes(&events),
        "artifact must agree with accounting even at zero packets"
    );
    eprintln!(
        "receipt: {{\"test\":\"prefill_cancel\",\"outcome\":\"passed\",\"wav_data_bytes\":{data_size}}}"
    );
}

// ---------------------------------------------------------------- case 3

/// Raw-mode cancel: PCM on stdout ends at a packet boundary, byte count equals the
/// stderr event accounting, every event survives the validator, exit 6.
// SIGINT semantics are unix-only; the Windows cancel path is bead frankentts-37hv.
#[cfg(unix)]
#[test]
fn raw_mode_sigint_stops_at_a_packet_boundary_with_exact_accounting() {
    if model_dir().is_none() {
        eprintln!(
            "receipt: {{\"test\":\"raw_mode\",\"outcome\":\"skipped\",\"reason\":\"model directory unavailable\"}}"
        );
        return;
    }
    let dir = scratch_dir("raw-mode");
    let pcm_path = dir.join("stream.pcm");
    let events_path = dir.join("events.ndjson");
    let child = spawn_to_files(
        &["say", "--no-resident", "--stream", "raw", LONG_TEXT],
        &[],
        &pcm_path,
        &events_path,
    );
    // Raw-mode events ride STDERR (the catalogue: raw bytes own stdout).
    assert!(
        wait_for_marker(
            &events_path,
            "\"event\":\"audio_chunk\"",
            Duration::from_secs(300)
        ),
        "first raw packet never arrived"
    );
    let (status, _latency) = sigint_and_wait(child);
    assert_eq!(status.code().unwrap_or(-1), 6);

    let events = parse_and_validate(&events_path, "raw events");
    let terminal = terminal_event(&events, "raw mode");
    assert_eq!(terminal["kind"], "cancelled");
    assert!(
        terminal["message"]
            .as_str()
            .is_some_and(|m| m.contains("raw PCM bytes")),
        "raw disposition must report the streamed byte count: {}",
        terminal["message"]
    );

    let pcm = std::fs::read(&pcm_path).expect("raw pcm readable");
    let accounted = chunk_bytes(&events);
    assert_eq!(
        pcm.len() as u64,
        accounted,
        "stdout bytes must equal sum(audio_chunk.bytes)"
    );
    assert_eq!(
        pcm.len() % 3840,
        0,
        "mid-generation cancel must stop at a FULL 4-frame packet boundary (3840 B)"
    );
    eprintln!(
        "receipt: {{\"test\":\"raw_mode\",\"outcome\":\"passed\",\"bytes\":{},\"packets\":{}}}",
        pcm.len(),
        pcm.len() / 3840
    );
}

// ---------------------------------------------------------------- case 5

/// Compressed-format cancel: `.m4a` requested, run cancelled mid-synthesis — no
/// encoder invocation (no `.m4a` ever exists), the staging WAV is kept and named in
/// the terminal event, exit 6.
// SIGINT semantics are unix-only; the Windows cancel path is bead frankentts-37hv.
#[cfg(unix)]
#[test]
fn compressed_target_cancel_skips_the_encoder_and_keeps_the_staging_wav() {
    if model_dir().is_none() {
        eprintln!(
            "receipt: {{\"test\":\"compressed_cancel\",\"outcome\":\"skipped\",\"reason\":\"model directory unavailable\"}}"
        );
        return;
    }
    let dir = scratch_dir("compressed");
    let out = dir.join("cancelled.m4a");
    let stdout_path = dir.join("events.ndjson");
    let stderr_path = dir.join("noise.txt");
    let child = spawn_to_files(
        &[
            "say",
            "--no-resident",
            LONG_TEXT,
            out.to_str().expect("utf8 out"),
        ],
        &[],
        &stdout_path,
        &stderr_path,
    );
    assert!(
        wait_for_marker(
            &stdout_path,
            "\"event\":\"audio_chunk\"",
            Duration::from_secs(300)
        ),
        "first packet never arrived"
    );
    let (status, _) = sigint_and_wait(child);
    assert_eq!(status.code().unwrap_or(-1), 6);

    assert!(
        !out.exists(),
        "the system encoder must never run for a cancelled .m4a request"
    );
    let staging = dir.join("cancelled.m4a.ftts-staging.wav");
    assert!(staging.exists(), "staging WAV must be kept");
    let data_size = validated_wav_data_size(&staging, "staging WAV");

    let events = parse_and_validate(&stdout_path, "compressed events");
    let terminal = terminal_event(
        &parse_and_validate(&stderr_path, "compressed stderr"),
        "compressed",
    );
    // The kept staging WAV must agree with what the client was told was delivered.
    assert_eq!(
        data_size,
        chunk_bytes(&events),
        "staging WAV data must equal sum(audio_chunk.bytes)"
    );
    assert_eq!(terminal["kind"], "cancelled");
    let message = terminal["message"].as_str().expect("message present");
    assert!(
        message.contains("encoding to") && message.contains("skipped"),
        "event must state the skipped encoding: {message}"
    );
    eprintln!(
        "receipt: {{\"test\":\"compressed_cancel\",\"outcome\":\"passed\",\"staging_wav_data_bytes\":{data_size}}}"
    );
}

// ---------------------------------------------------------------- case 6

/// Resident-enabled cancel: the client cancels while the DAEMON synthesizes; the
/// client exits 6 with the documented discard wording, and the daemon survives to
/// serve the next request warm (its serial loop finishing the orphaned request is
/// the pinned v1 behavior — the wire has no cancel op).
// SIGINT semantics are unix-only; the Windows cancel path is bead frankentts-37hv.
#[cfg(unix)]
#[test]
fn resident_client_cancel_discards_and_the_daemon_survives() {
    if model_dir().is_none() {
        eprintln!(
            "receipt: {{\"test\":\"resident_cancel\",\"outcome\":\"skipped\",\"reason\":\"model directory unavailable\"}}"
        );
        return;
    }
    let dir = scratch_dir("resident");
    let resident_env: [(&str, &str); 2] = [
        ("FTTS_RESIDENT_DIR", dir.to_str().expect("utf8 dir")),
        ("FTTS_RESIDENT_IDLE_SECS", "600"),
    ];

    // A: first run warms the daemon (cold spawn + load inside this request).
    let warm_out = dir.join("warm.wav");
    let started = Instant::now();
    let warm = ftts_cli_run(
        &[
            "say",
            "A short warmup sentence for the daemon.",
            warm_out.to_str().expect("utf8"),
        ],
        &resident_env,
    );
    let warm_ms = started.elapsed().as_millis();
    assert_eq!(warm.code().unwrap_or(-1), 0, "warmup run must succeed");
    let daemon_pid = |label: &str| -> u64 {
        std::fs::read_dir(&dir)
            .expect("resident dir exists")
            .filter_map(Result::ok)
            .find(|entry| entry.file_name().to_string_lossy().starts_with("resident-"))
            .and_then(|entry| std::fs::read_to_string(entry.path()).ok())
            .and_then(|raw| serde_json::from_str::<serde_json::Value>(&raw).ok())
            .and_then(|value| value.get("pid").and_then(|v| v.as_u64()))
            .unwrap_or_else(|| panic!("state file with pid missing after {label}"))
    };
    let pid_after_warm = daemon_pid("warmup");

    // B: long run, cancelled WHILE THE DAEMON SYNTHESIZES. On the v1 wire the client
    // sees no events until the whole blob arrives (the post-synthesis packetizer
    // emits them), so an event marker here would fire only after completion — the
    // trigger must be wall-clock (the battery spec allows exactly this for
    // real-signal cases). The daemon is warm from A, so by 25 s in, generation is
    // well underway. SIGINT sets the client's flag; the client stays blocked on the
    // wire until the daemon finishes, then discards and exits 6 — the documented v1
    // boundary this case pins.
    let long_out = dir.join("never.wav");
    let stdout_path = dir.join("b-events.ndjson");
    let stderr_path = dir.join("b-noise.txt");
    let child = spawn_to_files(
        &["say", LONG_TEXT, long_out.to_str().expect("utf8")],
        &resident_env,
        &stdout_path,
        &stderr_path,
    );
    std::thread::sleep(Duration::from_secs(25));
    let (status, latency) = sigint_and_wait(child);
    assert_eq!(status.code().unwrap_or(-1), 6);
    // The sink OPENS the output file before synthesis starts, so a discarded
    // resident reply leaves a HEADER-ONLY WAV (zero samples) rather than no file —
    // pin that exact, parseable shape.
    assert_eq!(
        validated_wav_data_size(&long_out, "discarded reply"),
        0,
        "a discarded resident reply must contain no audio samples"
    );
    // A v1-wire client sees NO audio events before the whole blob arrives; a
    // discarded reply therefore delivered zero packets to stdout.
    let events = parse_and_validate(&stdout_path, "resident cancel events");
    assert!(
        !events
            .iter()
            .any(|event| event_name(event) == "audio_chunk"),
        "a discarded resident run must not emit audio_chunk events"
    );
    let terminal = terminal_event(
        &parse_and_validate(&stderr_path, "resident cancel stderr"),
        "resident cancel",
    );
    assert_eq!(terminal["kind"], "cancelled");
    let message = terminal["message"].as_str().unwrap_or_default();
    assert!(
        message.contains("resident") && message.contains("discarded"),
        "terminal event must carry the discard disposition: {message}"
    );

    // C: the daemon must have survived B and serve warm. Survival is a PID fact,
    // not a speed fact: the daemon that served the warmup must be the one serving
    // after the cancelled request. A wall-clock ratio flaked on a loaded machine
    // (warm measured in a load dip, hot in a spike); timing stays in the receipt as
    // measurement, never as a gate.
    let hot_out = dir.join("hot.wav");
    let started = Instant::now();
    let hot = ftts_cli_run(
        &[
            "say",
            "The daemon survived and serves again.",
            hot_out.to_str().expect("utf8"),
        ],
        &resident_env,
    );
    let hot_ms = started.elapsed().as_millis();
    assert_eq!(hot.code().unwrap_or(-1), 0, "post-cancel run must succeed");
    let pid_after_cancel = daemon_pid("post-cancel serve");
    assert_eq!(
        pid_after_warm, pid_after_cancel,
        "the daemon must survive a client cancel and keep serving"
    );
    eprintln!(
        "receipt: {{\"test\":\"resident_cancel\",\"outcome\":\"passed\",\"cold_ms\":{warm_ms},\"client_cancel_to_exit_ms\":{},\"warm_serve_ms\":{hot_ms}}}",
        latency.as_millis()
    );
}

/// Plain synchronous run (both pipes drained) for resident lifecycle cases.
fn ftts_cli_run(args: &[&str], envs: &[(&str, &str)]) -> std::process::ExitStatus {
    let mut command = Command::new(env!("CARGO_BIN_EXE_ftts"));
    command
        .args(args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    for (key, value) in envs {
        command.env(key, value);
    }
    let mut child = command.spawn().expect("ftts spawns");
    // Drain both pipes concurrently (fail-closed consumer contract).
    let mut stdout_pipe = child.stdout.take().expect("stdout piped");
    let drained = std::thread::spawn(move || {
        let mut sink = Vec::new();
        let _ = stdout_pipe.read_to_end(&mut sink);
    });
    let mut noise = Vec::new();
    let _ = child
        .stderr
        .take()
        .expect("stderr piped")
        .read_to_end(&mut noise);
    let _ = drained.join();
    child.wait().expect("ftts exits")
}
