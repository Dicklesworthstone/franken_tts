//! Cross-target seam taps for the DISC-006 hunt (frankentts-p16p): when enabled, the engine
//! emits stable hash lines at operator boundaries so a native run and a wasm run of the same
//! pinned inputs can be diffed line-by-line to name the first operator whose bits diverge.
//!
//! Hashes are FNV-1a over little-endian bit patterns — target-independent given identical
//! bits. Off by default everywhere: native opts in through `FTTS_DEBUG_TAPS=<path|1>`, a wasm
//! host calls [`set_taps_enabled`] after installing a sink with [`install_tap_sink`]. The sink
//! writes to an append-only FILE rather than stderr: an earlier `eprintln!`-based revision
//! wedged fat-LTO release builds on the stdio ReentrantLock (see the comment in [`tap_emit`]).
//!
//! History note: this module was extracted from [`crate::generate`] when prefill-stage taps
//! needed to live inside `talker.rs`; the public entry points stay re-exported there so the
//! wasm bindings' import paths did not move.

use std::sync::atomic::{AtomicBool, Ordering};

static TAP_ENABLED: AtomicBool = AtomicBool::new(false);
/// A debug-tap sink: receives one line of text per tapped event.
type TapSink = Box<dyn Fn(&str) + Send + Sync>;
static TAP_SINK: std::sync::OnceLock<TapSink> = std::sync::OnceLock::new();
/// Native processes opt in through the environment; read once.
#[cfg(not(target_arch = "wasm32"))]
static NATIVE_TAPS_REQUESTED: std::sync::LazyLock<bool> =
    std::sync::LazyLock::new(|| std::env::var_os("FTTS_DEBUG_TAPS").is_some());
/// The native default sink file, opened on first emission.
#[cfg(not(target_arch = "wasm32"))]
static TAP_FILE: std::sync::Mutex<Option<std::fs::File>> = std::sync::Mutex::new(None);

/// Enables or disables tap emission for this process.
pub fn set_taps_enabled(enabled: bool) {
    TAP_ENABLED.store(enabled, Ordering::Relaxed);
}

/// Installs a custom emission sink (the wasm bindings route lines to `console.error`; the
/// native default appends to a file). Idempotent per process: only the first install wins.
pub fn install_tap_sink(sink: Box<dyn Fn(&str) + Send + Sync>) {
    let _ = TAP_SINK.set(sink);
}

pub(crate) fn taps_active() -> bool {
    if TAP_ENABLED.load(Ordering::Relaxed) {
        return true;
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        *NATIVE_TAPS_REQUESTED
    }
    #[cfg(target_arch = "wasm32")]
    false
}

pub(crate) fn tap_emit(line: &str) {
    if !taps_active() {
        return;
    }
    match TAP_SINK.get() {
        Some(sink) => sink(line),
        None => {
            #[cfg(not(target_arch = "wasm32"))]
            {
                use std::io::Write;
                let mut slot = TAP_FILE.lock().expect("tap file mutex");
                if slot.is_none() {
                    let requested = std::env::var_os("FTTS_DEBUG_TAPS");
                    let one = std::ffi::OsStr::new("1");
                    let path = match requested.as_deref() {
                        Some(value) if !value.is_empty() && value != one => {
                            std::path::PathBuf::from(value)
                        }
                        _ => std::env::temp_dir()
                            .join(format!("ftts-taps-{}.log", std::process::id())),
                    };
                    *slot = Some(
                        std::fs::OpenOptions::new()
                            .create(true)
                            .append(true)
                            .open(path)
                            .expect("open tap sink file"),
                    );
                }
                let file = slot.as_mut().expect("tap file just initialized");
                let _ = file.write_all(line.as_bytes());
                let _ = file.write_all(b"\n");
            }
            #[cfg(target_arch = "wasm32")]
            {
                eprintln!("{line}");
            }
        }
    }
}

/// The same hash over little-endian `u32` values (code groups).
pub(crate) fn tap_hash_u32(values: &[u32]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for value in values {
        for byte in value.to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x100_0000_01b3);
        }
    }
    hash
}
/// FNV-1a over the little-endian bit patterns of a slice — identical bits, identical hash, on
/// every target.
pub(crate) fn tap_hash_f32(values: &[f32]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for value in values {
        for byte in value.to_bits().to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x100_0000_01b3);
        }
    }
    hash
}
