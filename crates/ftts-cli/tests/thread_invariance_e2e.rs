//! Thread-count invariance: partitioning the int8 worker team never changes output bits
//! (bead `frankentts-v-metamorphic-0wq`).
//!
//! The team is a process-global (`ftts_kernels::team::armed_native`, a `OnceLock`), so
//! the only honest way to vary it is two processes: run the shipped `ftts` binary twice
//! over identical inputs with `FTTS_INT8_THREADS=1` (serial — no team at all) and `=4`,
//! then compare the rendered WAV byte-for-byte. This exercises exactly the claim the
//! team module documents: "Partitioning never changes output bits".
//!
//! Model-gated; weights absent reports an honest skip.

#![cfg(feature = "ultra-tests")]

use std::path::{Path, PathBuf};
use std::process::Command;

const CONTRACT: &str = "ProductionQuality/metamorphic";
const TEXT: &str = "Hello.";

fn model_dir() -> Option<PathBuf> {
    let root = std::env::var("FTTS_MODEL_DIR").map_or_else(
        |_| {
            #[allow(deprecated)]
            std::env::home_dir().map(|home| home.join(".cache/franken_tts/model"))
        },
        |dir| Some(PathBuf::from(dir)),
    )?;
    Path::new(&root).is_dir().then_some(root)
}

/// Runs one synthesis in its own process and returns the SHA-256 of the WAV bytes.
fn render_with_threads(binary: &Path, work: &Path, threads: &str) -> Result<String, String> {
    let out = work.join(format!("threads-{threads}.wav"));
    let status = Command::new(binary)
        .args(["say", "--voice", "matt", "--seed", "42", TEXT])
        .arg(&out)
        .env("FTTS_INT8_THREADS", threads)
        .env("FTTS_NO_RESIDENT", "1")
        // No CLI flag: the text-proportional cap is an env dial by contract.
        .env("FTTS_MAX_FRAMES", "36")
        .output()
        .map_err(|error| format!("cannot spawn {}: {error}", binary.display()))?;
    if !status.status.success() {
        return Err(format!(
            "ftts say (threads={threads}) exited {}: {}",
            status.status,
            String::from_utf8_lossy(&status.stderr)
                .chars()
                .take(300)
                .collect::<String>()
        ));
    }
    let bytes =
        std::fs::read(&out).map_err(|error| format!("cannot read {}: {error}", out.display()))?;
    Ok(ftts_artifacts::sha256::hex_digest(&bytes))
}
#[test]
fn worker_partition_count_never_changes_the_audio() {
    const TEST: &str = "metamorphic_thread_count_invariance_e2e";
    let emit_skip = |reason: String| {
        println!(
            "{{\"test\":\"{TEST}\",\"contract\":\"{CONTRACT}\",\"outcome\":\"skipped\",\
             \"reason\":\"{}\"}}",
            reason.replace('"', "'")
        );
    };
    if model_dir().is_none() {
        emit_skip("model directory unavailable".to_owned());
        return;
    }
    let binary = PathBuf::from(env!("CARGO_BIN_EXE_ftts"));
    let work = std::env::temp_dir().join(format!("ftts-thread-invariance-{}", std::process::id()));
    std::fs::create_dir_all(&work).expect("work dir");

    let serial = match render_with_threads(&binary, &work, "1") {
        Ok(hash) => hash,
        Err(reason) => {
            emit_skip(reason);
            return;
        }
    };
    let four_way = match render_with_threads(&binary, &work, "4") {
        Ok(hash) => hash,
        Err(reason) => {
            emit_skip(reason);
            return;
        }
    };
    assert_eq!(
        serial, four_way,
        "FTTS_INT8_THREADS=1 vs 4 produced different audio ({serial} vs {four_way}): \
         partitioning changed output bits, contradicting the team contract"
    );
    println!(
        "{{\"test\":\"{TEST}\",\"contract\":\"{CONTRACT}\",\"outcome\":\"passed\",\
         \"reason\":\"threads 1 and 4 → identical wav sha {serial}\"}}"
    );
}
