//! C ABI over the native engine, for the iOS app (see `docs/IOS_APP_PLAN.md`).
//!
//! One loading and synthesis implementation exists in this tree — the CLI's `synth`
//! module — and this crate is a thin translation of it into C calling conventions, the
//! same way `ftts-wasm` translates it for the browser. Nothing model-shaped lives here.
//!
//! # Contract
//!
//! - Functions returning `int` use 0 for success; any other value means failure and
//!   [`ftts_last_error_message`] describes it (thread-local, valid until the next
//!   failing call on the same thread).
//! - The engine handle is NOT thread-safe. The caller serializes access (the Swift app
//!   owns it inside an actor).
//! - PCM returned by [`ftts_synthesize`] is owned by the caller and must be released
//!   with [`ftts_pcm_free`], with the exact length that was returned.
//! - Panics do not unwind across the boundary: every entry point catches unwinds and
//!   converts them into error returns. (Release builds are `panic = "abort"` anyway;
//!   the catch keeps debug/test builds sound.)
//!
//! # Unsafe policy
//!
//! The workspace forbids `unsafe_code`; a C ABI cannot exist under that rule, so this
//! crate carries the same posture as `ftts-kernels`: `deny` by default with small,
//! audited `#[allow]` islands, each justifying itself with a SAFETY comment. Every
//! pointer that crosses the boundary is validated for null before use, and lengths are
//! trusted only as far as the header documents them (the caller owns that contract).

use std::cell::RefCell;
use std::ffi::{CStr, CString, c_char};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::Path;
use std::sync::OnceLock;

use ftts_cli::synth::{LoadedModel, ModelBundle, synthesize};
use ftts_core::{CancellationToken, SynthesisRequest, TtsEngine};
use ftts_model_qwen::speaker::{Encoder as SpeakerEncoder, log_mel_from_24khz_pcm};

/// Floats in a speaker x-vector; the fixed width of every `.spk` file and enroll output.
pub const SPEAKER_WIDTH: usize = 1024;

/// The built-in voices: the same preset files every other surface embeds.
const PRESET_VOICES: &[(&str, &str, &[u8])] = &[
    (
        "matt",
        "warm, easy, masculine; the out-of-box default",
        include_bytes!("../../ftts-cli/presets/matt.spk"),
    ),
    (
        "james",
        "natural, conversational, masculine",
        include_bytes!("../../ftts-cli/presets/james.spk"),
    ),
    (
        "leo",
        "relaxed, resonant, masculine",
        include_bytes!("../../ftts-cli/presets/leo.spk"),
    ),
    (
        "robert",
        "steady, measured, masculine",
        include_bytes!("../../ftts-cli/presets/robert.spk"),
    ),
    (
        "judy",
        "bright, articulate, feminine",
        include_bytes!("../../ftts-cli/presets/judy.spk"),
    ),
    (
        "aria",
        "clear, warm, feminine",
        include_bytes!("../../ftts-cli/presets/aria.spk"),
    ),
    (
        "ember",
        "aria's character, a few semitones deeper",
        include_bytes!("../../ftts-cli/presets/ember.spk"),
    ),
];

thread_local! {
    static LAST_ERROR: RefCell<CString> = RefCell::new(CString::default());
}

fn set_error(message: impl Into<String>) {
    let message = message.into();
    let owned = CString::new(message.replace('\0', "?"))
        .unwrap_or_else(|_| CString::new("error message unrepresentable").expect("static"));
    LAST_ERROR.with(|slot| *slot.borrow_mut() = owned);
}

/// The engine handle the header's `FttsEngine` opaques over.
pub struct FttsEngine {
    loaded: LoadedModel,
    engine: TtsEngine,
    bundle: ModelBundle,
    speaker_encoder: Option<SpeakerEncoder>,
}

/// Runs a body, converting any panic into an error return value.
fn guarded<T>(fallback: T, body: impl FnOnce() -> T) -> T {
    match catch_unwind(AssertUnwindSafe(body)) {
        Ok(value) => value,
        Err(payload) => {
            let message = payload
                .downcast_ref::<&str>()
                .map(ToString::to_string)
                .or_else(|| payload.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "panic in ftts engine".to_owned());
            set_error(format!("internal panic: {message}"));
            fallback
        }
    }
}

/// Last failure on this thread, as UTF-8. Never null; empty before any failure.
#[allow(unsafe_code)] // audited export, part of the C ABI surface
#[unsafe(no_mangle)]
pub extern "C" fn ftts_last_error_message() -> *const c_char {
    LAST_ERROR.with(|slot| slot.borrow().as_ptr())
}

/// JSON array of `{name, character}` for the built-in voices. Static lifetime.
#[allow(unsafe_code)] // audited export, part of the C ABI surface
#[unsafe(no_mangle)]
pub extern "C" fn ftts_presets_json() -> *const c_char {
    static JSON: OnceLock<CString> = OnceLock::new();
    JSON.get_or_init(|| {
        let entries: Vec<serde_json::Value> = PRESET_VOICES
            .iter()
            .map(|(name, character, _)| serde_json::json!({"name": name, "character": character}))
            .collect();
        CString::new(serde_json::Value::Array(entries).to_string()).expect("no NUL in JSON")
    })
    .as_ptr()
}

/// Copies the named preset's 1,024-float x-vector into `out`. 0 on success.
#[allow(unsafe_code)] // audited export, part of the C ABI surface
#[unsafe(no_mangle)]
/// # Safety
/// `name` must be NUL-terminated UTF-8; `out` must have room for [`SPEAKER_WIDTH`] floats.
pub unsafe extern "C" fn ftts_preset_vector(name: *const c_char, out: *mut f32) -> i32 {
    guarded(1, || {
        if name.is_null() || out.is_null() {
            set_error("null pointer to ftts_preset_vector");
            return 1;
        }
        // SAFETY: `name` is non-null and the header requires a NUL-terminated UTF-8 string;
        // CStr::from_ptr reads up to that terminator and no further.
        #[allow(unsafe_code)]
        let name = match unsafe { CStr::from_ptr(name) }.to_str() {
            Ok(name) => name,
            Err(_) => {
                set_error("preset name is not UTF-8");
                return 1;
            }
        };
        let Some((_, _, bytes)) = PRESET_VOICES.iter().find(|(n, _, _)| *n == name) else {
            set_error(format!("unknown preset voice `{name}`"));
            return 1;
        };
        // SAFETY: the header requires `out` to have room for SPEAKER_WIDTH floats, and every
        // embedded preset is exactly SPEAKER_WIDTH * 4 bytes (asserted by the test below).
        #[allow(unsafe_code)]
        let target = unsafe { std::slice::from_raw_parts_mut(out, SPEAKER_WIDTH) };
        let (chunks, _remainder) = bytes.as_chunks::<4>();
        for (slot, chunk) in target.iter_mut().zip(chunks) {
            *slot = f32::from_le_bytes(*chunk);
        }
        0
    })
}

/// Opens the engine over a complete model directory. Null on failure.
#[allow(unsafe_code)] // audited export, part of the C ABI surface
#[unsafe(no_mangle)]
/// # Safety
/// `model_dir` must be a NUL-terminated UTF-8 path.
pub unsafe extern "C" fn ftts_engine_open(model_dir: *const c_char) -> *mut FttsEngine {
    guarded(std::ptr::null_mut(), || {
        if model_dir.is_null() {
            set_error("null model_dir");
            return std::ptr::null_mut();
        }
        // SAFETY: non-null, and the header requires a NUL-terminated UTF-8 path.
        #[allow(unsafe_code)]
        let dir = match unsafe { CStr::from_ptr(model_dir) }.to_str() {
            Ok(dir) => dir,
            Err(_) => {
                set_error("model_dir is not UTF-8");
                return std::ptr::null_mut();
            }
        };
        let bundle = match ModelBundle::resolve(Path::new(dir)) {
            Ok(bundle) => bundle,
            Err(error) => {
                set_error(error.to_string());
                return std::ptr::null_mut();
            }
        };
        let loaded = match LoadedModel::load(&bundle) {
            Ok(loaded) => loaded,
            Err(error) => {
                set_error(error.to_string());
                return std::ptr::null_mut();
            }
        };
        let engine = match TtsEngine::from_process_environment() {
            Ok(engine) => engine,
            Err(error) => {
                set_error(format!("cannot start the engine: {error}"));
                return std::ptr::null_mut();
            }
        };
        Box::into_raw(Box::new(FttsEngine {
            loaded,
            engine,
            bundle,
            speaker_encoder: None,
        }))
    })
}

/// Releases an engine returned by [`ftts_engine_open`]. Null is a no-op.
#[allow(unsafe_code)] // audited export, part of the C ABI surface
#[unsafe(no_mangle)]
/// # Safety
/// `engine` must be null or a pointer from [`ftts_engine_open`], passed exactly once.
pub unsafe extern "C" fn ftts_engine_close(engine: *mut FttsEngine) {
    if engine.is_null() {
        return;
    }
    // SAFETY: the header requires this pointer to come from ftts_engine_open exactly once;
    // reconstituting the Box drops the engine and its model.
    #[allow(unsafe_code)]
    drop(unsafe { Box::from_raw(engine) });
}

/// Synthesizes `text` with the given speaker vector. 0 on success, with `out_pcm`/`out_len`
/// set to a mono 24 kHz f32 buffer the caller must release via [`ftts_pcm_free`].
#[allow(unsafe_code)] // audited export, part of the C ABI surface
#[unsafe(no_mangle)]
/// # Safety
/// `engine` from [`ftts_engine_open`] with access serialized by the caller; `text`
/// NUL-terminated UTF-8; `speaker` valid for `speaker_len` floats; `out_pcm`/`out_len`
/// valid for one write each.
pub unsafe extern "C" fn ftts_synthesize(
    engine: *mut FttsEngine,
    text: *const c_char,
    speaker: *const f32,
    speaker_len: usize,
    seed: u64,
    out_pcm: *mut *mut f32,
    out_len: *mut usize,
) -> i32 {
    guarded(1, || {
        if engine.is_null()
            || text.is_null()
            || speaker.is_null()
            || out_pcm.is_null()
            || out_len.is_null()
        {
            set_error("null pointer to ftts_synthesize");
            return 1;
        }
        if speaker_len != SPEAKER_WIDTH {
            set_error(format!(
                "speaker vector must be {SPEAKER_WIDTH} floats, got {speaker_len}"
            ));
            return 1;
        }
        // SAFETY: engine comes from ftts_engine_open and the caller serializes access, so a
        // unique borrow for the duration of this call is sound. text is a NUL-terminated
        // string; speaker points at speaker_len floats — both per the header contract.
        #[allow(unsafe_code)]
        let (engine, text, speaker) = unsafe {
            (
                &mut *engine,
                CStr::from_ptr(text),
                std::slice::from_raw_parts(speaker, speaker_len),
            )
        };
        let Ok(text) = text.to_str() else {
            set_error("text is not UTF-8");
            return 1;
        };
        let request = SynthesisRequest::new(text.to_owned());
        let cancellation = CancellationToken::new();
        let observer = |_event: ftts_core::SynthesisEvent| {};
        match synthesize(
            &engine.loaded,
            &engine.engine,
            &request,
            speaker,
            seed,
            &cancellation,
            &observer,
        ) {
            Ok(audio) => {
                let mut pcm = audio.pcm.into_boxed_slice();
                let len = pcm.len();
                let pointer = pcm.as_mut_ptr();
                std::mem::forget(pcm);
                // SAFETY: out_pcm/out_len are non-null per the check above; each is written once.
                #[allow(unsafe_code)]
                unsafe {
                    *out_pcm = pointer;
                    *out_len = len;
                }
                0
            }
            Err(error) => {
                set_error(error.to_string());
                1
            }
        }
    })
}

/// Releases a PCM buffer from [`ftts_synthesize`]. `len` must be the returned length.
#[allow(unsafe_code)] // audited export, part of the C ABI surface
#[unsafe(no_mangle)]
/// # Safety
/// `(pcm, len)` must be exactly one pair returned by [`ftts_synthesize`].
pub unsafe extern "C" fn ftts_pcm_free(pcm: *mut f32, len: usize) {
    if pcm.is_null() {
        return;
    }
    // SAFETY: the header requires (pcm, len) to be exactly one pair returned by
    // ftts_synthesize, which produced it from Box<[f32]> of that length.
    #[allow(unsafe_code)]
    drop(unsafe { Box::from_raw(std::ptr::slice_from_raw_parts_mut(pcm, len)) });
}

/// Enrolls a voice from mono 24 kHz f32 PCM, writing the x-vector into `out`. 0 on success.
#[allow(unsafe_code)] // audited export, part of the C ABI surface
#[unsafe(no_mangle)]
/// # Safety
/// `engine` from [`ftts_engine_open`] with serialized access; `pcm` valid for `len`
/// floats; `out` must have room for [`SPEAKER_WIDTH`] floats.
pub unsafe extern "C" fn ftts_enroll(
    engine: *mut FttsEngine,
    pcm: *const f32,
    len: usize,
    out: *mut f32,
) -> i32 {
    guarded(1, || {
        if engine.is_null() || pcm.is_null() || out.is_null() {
            set_error("null pointer to ftts_enroll");
            return 1;
        }
        // SAFETY: engine from ftts_engine_open with serialized access; pcm points at len
        // floats; out has room for SPEAKER_WIDTH floats — all per the header contract.
        #[allow(unsafe_code)]
        let (engine, pcm) = unsafe { (&mut *engine, std::slice::from_raw_parts(pcm, len)) };
        if engine.speaker_encoder.is_none() {
            let artifact = engine
                .bundle
                .canonical_main
                .as_deref()
                .unwrap_or(&engine.bundle.main);
            match SpeakerEncoder::load_fttsq(artifact) {
                Ok(encoder) => engine.speaker_encoder = Some(encoder),
                Err(error) => {
                    set_error(format!("speaker encoder hydration failed: {error}"));
                    return 1;
                }
            }
        }
        let mel = match log_mel_from_24khz_pcm(pcm) {
            Ok(mel) => mel,
            Err(error) => {
                set_error(format!("cannot extract speaker features: {error}"));
                return 1;
            }
        };
        let encoder = engine
            .speaker_encoder
            .as_ref()
            .expect("hydrated just above");
        let vector = encoder.encode(&mel.values, mel.frames);
        if vector.len() != SPEAKER_WIDTH || vector.iter().any(|value| !value.is_finite()) {
            set_error("speaker encoder produced an invalid x-vector");
            return 1;
        }
        // SAFETY: out is non-null and the header requires SPEAKER_WIDTH floats of room.
        #[allow(unsafe_code)]
        let target = unsafe { std::slice::from_raw_parts_mut(out, SPEAKER_WIDTH) };
        target.copy_from_slice(&vector);
        0
    })
}

#[cfg(test)]
#[allow(unsafe_code)] // tests exercise the C ABI exactly as a caller would
mod tests {
    use super::*;

    #[test]
    fn every_preset_is_exactly_one_speaker_vector() {
        for (name, _, bytes) in PRESET_VOICES {
            assert_eq!(bytes.len(), SPEAKER_WIDTH * 4, "{name}");
        }
    }

    #[test]
    fn presets_json_is_valid_and_complete() {
        // SAFETY-free: call through the public ABI surface on this thread.
        let raw = ftts_presets_json();
        assert!(!raw.is_null());
        // SAFETY: the pointer is the static CString this crate just built.
        #[allow(unsafe_code)]
        let json = unsafe { CStr::from_ptr(raw) }.to_str().unwrap();
        let parsed: serde_json::Value = serde_json::from_str(json).unwrap();
        assert_eq!(parsed.as_array().unwrap().len(), PRESET_VOICES.len());
    }

    #[test]
    fn preset_vector_round_trips_and_rejects_unknown_names() {
        let name = CString::new("matt").unwrap();
        let mut out = vec![0.0_f32; SPEAKER_WIDTH];
        assert_eq!(
            unsafe { ftts_preset_vector(name.as_ptr(), out.as_mut_ptr()) },
            0
        );
        assert!(
            out.iter().any(|v| *v != 0.0),
            "vector should be non-trivial"
        );

        let missing = CString::new("nobody").unwrap();
        assert_eq!(
            unsafe { ftts_preset_vector(missing.as_ptr(), out.as_mut_ptr()) },
            1
        );
        // SAFETY: reading the thread-local error string this crate owns.
        #[allow(unsafe_code)]
        let message = unsafe { CStr::from_ptr(ftts_last_error_message()) }
            .to_str()
            .unwrap();
        assert!(message.contains("nobody"), "{message}");
    }

    #[test]
    fn open_refuses_a_missing_model_directory_with_a_message() {
        let dir = CString::new("/nonexistent/franken-model").unwrap();
        let engine = unsafe { ftts_engine_open(dir.as_ptr()) };
        assert!(engine.is_null());
        #[allow(unsafe_code)]
        let message = unsafe { CStr::from_ptr(ftts_last_error_message()) }
            .to_str()
            .unwrap();
        assert!(!message.is_empty());
        unsafe { ftts_engine_close(engine) }; // null close is a no-op
    }

    /// Model-gated end-to-end smoke: open, synthesize a word, free, close.
    #[test]
    fn synthesizes_through_the_c_abi_when_the_model_is_present() {
        #[allow(deprecated)]
        let Some(dir) = std::env::home_dir().map(|home| home.join(".cache/franken_tts/model"))
        else {
            eprintln!("SKIP-AS-SUCCESS: no home dir");
            return;
        };
        if !dir.join("vocab.json").is_file() {
            eprintln!("SKIP-AS-SUCCESS: model not present; FFI e2e needs the real model");
            return;
        }
        let c_dir = CString::new(dir.to_str().unwrap()).unwrap();
        let engine = unsafe { ftts_engine_open(c_dir.as_ptr()) };
        assert!(!engine.is_null(), "open failed: {}", last_error());
        let mut speaker = vec![0.0_f32; SPEAKER_WIDTH];
        let matt = CString::new("matt").unwrap();
        assert_eq!(
            unsafe { ftts_preset_vector(matt.as_ptr(), speaker.as_mut_ptr()) },
            0
        );
        let text = CString::new("Hi.").unwrap();
        let mut pcm: *mut f32 = std::ptr::null_mut();
        let mut len = 0_usize;
        let code = unsafe {
            ftts_synthesize(
                engine,
                text.as_ptr(),
                speaker.as_ptr(),
                SPEAKER_WIDTH,
                0,
                &raw mut pcm,
                &raw mut len,
            )
        };
        assert_eq!(code, 0, "synthesize failed: {}", last_error());
        assert!(len > 0 && !pcm.is_null());
        unsafe { ftts_pcm_free(pcm, len) };
        unsafe { ftts_engine_close(engine) };
    }

    fn last_error() -> String {
        #[allow(unsafe_code)]
        unsafe { CStr::from_ptr(ftts_last_error_message()) }
            .to_string_lossy()
            .into_owned()
    }
}
