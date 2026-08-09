//! End-to-end contract of the resident engine, against the real binary and real model.
//!
//! Model-gated: without a complete model directory the test reports the skip and passes,
//! per the repository's model-gated e2e convention. With the model it proves the four
//! promises of the resident path on this machine, whichever OS it is:
//!
//! 1. a first `ftts say` spawns a daemon and completes;
//! 2. a second `ftts say` reuses the same daemon and its synthesis stage is faster,
//!    because the model hydration it skipped happened in run one;
//! 3. `--no-resident` output is byte-identical to resident output (same text, seed,
//!    voice), so the resident path changes nothing but latency;
//! 4. the daemon exits by itself after the idle period and removes its state file.

use std::path::PathBuf;
use std::process::Command;
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

struct SayRun {
    synthesis_ms: u64,
    wav: Vec<u8>,
}

fn run_say(resident_dir: &std::path::Path, out: &std::path::Path, extra: &[&str]) -> SayRun {
    let mut command = Command::new(env!("CARGO_BIN_EXE_ftts"));
    command
        .arg("say")
        .arg("Resident engines should answer twice as politely.")
        .arg("-o")
        .arg(out)
        .args(extra)
        .env("FTTS_RESIDENT_DIR", resident_dir)
        .env("FTTS_RESIDENT_IDLE_SECS", "3");
    let output = command.output().expect("ftts say runs");
    assert!(
        output.status.success(),
        "say failed: {}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
    // Piped stdout gets the NDJSON contract; the synthesis stage duration is the
    // difference between its begin and end events' elapsed_ms.
    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut begin = None;
    let mut end = None;
    for line in stdout.lines() {
        let Ok(event) = serde_json::from_str::<serde_json::Value>(line) else {
            continue;
        };
        if event.get("event").and_then(|v| v.as_str()) == Some("stage")
            && event.get("name").and_then(|v| v.as_str()) == Some("synthesis")
        {
            let elapsed = event.get("elapsed_ms").and_then(serde_json::Value::as_u64);
            match event.get("state").and_then(|v| v.as_str()) {
                Some("begin") => begin = elapsed,
                Some("end") => end = elapsed,
                _ => {}
            }
        }
    }
    let (begin, end) = (
        begin.expect("synthesis begin stage"),
        end.expect("synthesis end stage"),
    );
    SayRun {
        synthesis_ms: end.saturating_sub(begin),
        wav: std::fs::read(out).expect("wav written"),
    }
}

#[test]
fn resident_daemon_reuse_parity_and_idle_exit() {
    let Some(_model) = model_dir() else {
        eprintln!(
            "SKIP-AS-SUCCESS: no complete model directory; resident e2e needs the real model"
        );
        return;
    };
    let scratch = std::env::temp_dir().join(format!("ftts-resident-e2e-{}", std::process::id()));
    std::fs::create_dir_all(&scratch).expect("scratch dir");
    let resident_dir = scratch.join("state");

    // Run 1: spawns the daemon; its synthesis stage includes the daemon-side hydration.
    let first = run_say(&resident_dir, &scratch.join("a.wav"), &[]);
    let state_file = std::fs::read_dir(&resident_dir)
        .expect("state dir exists after a resident run")
        .filter_map(Result::ok)
        .find(|entry| entry.file_name().to_string_lossy().starts_with("resident-"))
        .expect("resident state file present");
    let pid_after_first = std::fs::read_to_string(state_file.path())
        .ok()
        .and_then(|raw| serde_json::from_str::<serde_json::Value>(&raw).ok())
        .and_then(|v| v.get("pid").and_then(serde_json::Value::as_u64))
        .expect("state file carries the daemon pid");

    // Run 2: same daemon, no hydration; the synthesis stage must be visibly faster.
    let second = run_say(&resident_dir, &scratch.join("b.wav"), &[]);
    let pid_after_second = std::fs::read_to_string(state_file.path())
        .ok()
        .and_then(|raw| serde_json::from_str::<serde_json::Value>(&raw).ok())
        .and_then(|v| v.get("pid").and_then(serde_json::Value::as_u64))
        .expect("state file still present after reuse");
    assert_eq!(pid_after_first, pid_after_second, "daemon was reused");
    assert!(
        second.synthesis_ms + 1500 <= first.synthesis_ms,
        "second run should skip hydration: first={}ms second={}ms",
        first.synthesis_ms,
        second.synthesis_ms,
    );

    // Run 3: the resident path must be a pure latency optimization: identical bytes.
    let inline = run_say(&resident_dir, &scratch.join("c.wav"), &["--no-resident"]);
    assert_eq!(
        second.wav, inline.wav,
        "resident and in-process synthesis must produce identical WAV bytes",
    );

    // Idle exit: within the 3 s idle period (plus slack), the daemon removes its state.
    let deadline = Instant::now() + Duration::from_secs(15);
    loop {
        if !state_file.path().exists() {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "daemon did not exit within the idle period",
        );
        std::thread::sleep(Duration::from_millis(250));
    }
}
