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
use std::ffi::{CStr, CString, c_char, c_void};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};

use ftts_cli::synth::{
    DENOISE_ARTIFACT_RELPATH, LoadedModel, ModelBundle, ReferenceCleanup, VoiceConditioning,
    denoise_pcm_24k, last_synthesis_profile, speaker_from_reference_pcm, synthesize,
};
use ftts_core::{CancellationToken, SynthesisRequest, TtsEngine};

/// Floats in a speaker x-vector; the fixed width of every `.spk` file and enroll output.
pub const SPEAKER_WIDTH: usize = 1024;

/// Stable ABI version for [`FttsProgressEvent`].
pub const FTTS_PROGRESS_ABI_VERSION: u32 = 1;
pub const FTTS_PROGRESS_FLAG_TOTAL_IS_UPPER_BOUND: u32 = 1;
pub const FTTS_PROGRESS_FLAG_INVALIDATES_OUTPUT: u32 = 2;

pub const FTTS_PROGRESS_KIND_STAGE_STARTED: u32 = 1;
pub const FTTS_PROGRESS_KIND_STAGE_FINISHED: u32 = 2;
pub const FTTS_PROGRESS_KIND_UNIT: u32 = 3;
pub const FTTS_PROGRESS_KIND_ADMISSION: u32 = 4;
pub const FTTS_PROGRESS_KIND_HEALTH: u32 = 5;

pub const FTTS_PROGRESS_STAGE_MODEL_BUNDLE: u32 = 1;
pub const FTTS_PROGRESS_STAGE_MODEL_WEIGHTS: u32 = 2;
pub const FTTS_PROGRESS_STAGE_RUNTIME: u32 = 3;
pub const FTTS_PROGRESS_STAGE_SYNTHESIS: u32 = 4;
pub const FTTS_PROGRESS_STAGE_TEXT: u32 = 5;
pub const FTTS_PROGRESS_STAGE_FRAMES: u32 = 6;
pub const FTTS_PROGRESS_STAGE_CODEC: u32 = 7;
pub const FTTS_PROGRESS_STAGE_RESOURCE_ADMISSION: u32 = 8;
pub const FTTS_PROGRESS_STAGE_HEALTH: u32 = 9;

/// Privacy-safe scalar lifecycle event shared with native Apple clients.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FttsProgressEvent {
    pub abi_version: u32,
    pub kind: u32,
    pub stage: u32,
    pub flags: u32,
    pub current: u64,
    pub total: u64,
    pub detail: u64,
    pub elapsed_ms: f64,
}

impl FttsProgressEvent {
    fn new(kind: u32, stage: u32) -> Self {
        Self {
            abi_version: FTTS_PROGRESS_ABI_VERSION,
            kind,
            stage,
            flags: 0,
            current: 0,
            total: 0,
            detail: 0,
            elapsed_ms: 0.0,
        }
    }
}

/// C callback for progress delivery. A nonzero verdict requests cancellation.
pub type FttsProgressFn =
    unsafe extern "C" fn(ctx: *mut c_void, event: *const FttsProgressEvent) -> i32;

#[derive(Clone)]
struct ProgressEmitter {
    callback: Option<FttsProgressFn>,
    ctx: *mut c_void,
    cancellation: Option<CancellationToken>,
}

// SAFETY: the header explicitly requires the callback/context pair to be callable from
// engine threads. Rust never dereferences `ctx`; it only returns the pair to its owner.
#[allow(unsafe_code)]
unsafe impl Send for ProgressEmitter {}
// SAFETY: same callback contract as the Send implementation. Calls are synchronous;
// any callback-side synchronization is the caller's responsibility.
#[allow(unsafe_code)]
unsafe impl Sync for ProgressEmitter {}

impl ProgressEmitter {
    fn emit(&self, event: FttsProgressEvent) -> bool {
        let Some(callback) = self.callback else {
            return true;
        };
        // SAFETY: the function/context pair follows the contract documented in the C
        // header. `event` remains alive and immutable for the complete synchronous call.
        #[allow(unsafe_code)]
        let verdict = unsafe { callback(self.ctx, &event) };
        if verdict != 0 {
            if let Some(cancellation) = &self.cancellation {
                cancellation.cancel();
            }
            false
        } else {
            true
        }
    }
}

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
    (
        "liam",
        "thoughtful, engaging, masculine",
        include_bytes!("../../ftts-cli/presets/liam.spk"),
    ),
    (
        "anthony",
        "authoritative, articulate, masculine",
        include_bytes!("../../ftts-cli/presets/anthony.spk"),
    ),
    (
        "russell",
        "rich, warm, masculine",
        include_bytes!("../../ftts-cli/presets/russell.spk"),
    ),
    (
        "steve",
        "direct, energetic, masculine",
        include_bytes!("../../ftts-cli/presets/steve.spk"),
    ),
    (
        "daniel",
        "clear, calm, masculine",
        include_bytes!("../../ftts-cli/presets/daniel.spk"),
    ),
    (
        "meryl",
        "expressive, poised, feminine",
        include_bytes!("../../ftts-cli/presets/meryl.spk"),
    ),
    (
        "laurence",
        "deep, measured, masculine",
        include_bytes!("../../ftts-cli/presets/laurence.spk"),
    ),
    (
        "jack",
        "crisp, confident, masculine",
        include_bytes!("../../ftts-cli/presets/jack.spk"),
    ),
    (
        "michael",
        "warm, dynamic, masculine",
        include_bytes!("../../ftts-cli/presets/michael.spk"),
    ),
    (
        "jodie",
        "warm, expressive, feminine",
        include_bytes!("../../ftts-cli/presets/jodie.spk"),
    ),
    (
        "denzel",
        "commanding, charismatic, masculine",
        include_bytes!("../../ftts-cli/presets/denzel.spk"),
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
    last_profile_json: CString,
}

fn synthesis_profile_json() -> CString {
    let profile = last_synthesis_profile();
    let millis = |duration: std::time::Duration| duration.as_secs_f64() * 1_000.0;
    let value = serde_json::json!({
        "total_ms": millis(profile.call),
        "generation_ms": millis(profile.generation),
        "prefill_ms": millis(profile.prefill),
        "microdecoder_ms": millis(profile.microdecoder),
        "feedback_ms": millis(profile.feedback),
        "talker_ms": millis(profile.talker),
        "codec_active_ms": millis(profile.codec_active),
        "frames": profile.frames,
        "team_partitions": profile.team_partitions,
    });
    CString::new(value.to_string()).expect("JSON serialization cannot contain NUL")
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
    // SAFETY: this forwards the identical pointer contract with progress disabled.
    #[allow(unsafe_code)]
    unsafe {
        ftts_engine_open_with_progress(model_dir, None, std::ptr::null_mut())
    }
}

/// Opens the engine while emitting truthful coarse load-stage events.
#[allow(unsafe_code)] // audited export, part of the C ABI surface
#[unsafe(no_mangle)]
/// # Safety
/// `model_dir` must be NUL-terminated UTF-8. When present, `on_progress` and `ctx`
/// follow the callback contract in `ftts_ffi.h` for the duration of this call.
pub unsafe extern "C" fn ftts_engine_open_with_progress(
    model_dir: *const c_char,
    on_progress: Option<FttsProgressFn>,
    ctx: *mut c_void,
) -> *mut FttsEngine {
    guarded(std::ptr::null_mut(), || {
        if model_dir.is_null() {
            set_error("null model_dir");
            return std::ptr::null_mut();
        }
        let progress = ProgressEmitter {
            callback: on_progress,
            ctx,
            cancellation: None,
        };
        // SAFETY: non-null, and the header requires a NUL-terminated UTF-8 path.
        #[allow(unsafe_code)]
        let dir = match unsafe { CStr::from_ptr(model_dir) }.to_str() {
            Ok(dir) => dir,
            Err(_) => {
                set_error("model_dir is not UTF-8");
                return std::ptr::null_mut();
            }
        };
        if !progress.emit(FttsProgressEvent::new(
            FTTS_PROGRESS_KIND_STAGE_STARTED,
            FTTS_PROGRESS_STAGE_MODEL_BUNDLE,
        )) {
            set_error("engine open cancelled");
            return std::ptr::null_mut();
        }
        let bundle = match ModelBundle::resolve(Path::new(dir)) {
            Ok(bundle) => bundle,
            Err(error) => {
                set_error(error.to_string());
                return std::ptr::null_mut();
            }
        };
        if !progress.emit(FttsProgressEvent::new(
            FTTS_PROGRESS_KIND_STAGE_FINISHED,
            FTTS_PROGRESS_STAGE_MODEL_BUNDLE,
        )) {
            set_error("engine open cancelled");
            return std::ptr::null_mut();
        }
        if !progress.emit(FttsProgressEvent::new(
            FTTS_PROGRESS_KIND_STAGE_STARTED,
            FTTS_PROGRESS_STAGE_MODEL_WEIGHTS,
        )) {
            set_error("engine open cancelled");
            return std::ptr::null_mut();
        }
        let loaded = match LoadedModel::load(&bundle) {
            Ok(loaded) => loaded,
            Err(error) => {
                set_error(error.to_string());
                return std::ptr::null_mut();
            }
        };
        if !progress.emit(FttsProgressEvent::new(
            FTTS_PROGRESS_KIND_STAGE_FINISHED,
            FTTS_PROGRESS_STAGE_MODEL_WEIGHTS,
        )) {
            set_error("engine open cancelled");
            return std::ptr::null_mut();
        }
        if !progress.emit(FttsProgressEvent::new(
            FTTS_PROGRESS_KIND_STAGE_STARTED,
            FTTS_PROGRESS_STAGE_RUNTIME,
        )) {
            set_error("engine open cancelled");
            return std::ptr::null_mut();
        }
        let engine = match TtsEngine::from_process_environment() {
            Ok(engine) => engine,
            Err(error) => {
                set_error(format!("cannot start the engine: {error}"));
                return std::ptr::null_mut();
            }
        };
        if !progress.emit(FttsProgressEvent::new(
            FTTS_PROGRESS_KIND_STAGE_FINISHED,
            FTTS_PROGRESS_STAGE_RUNTIME,
        )) {
            set_error("engine open cancelled");
            return std::ptr::null_mut();
        }
        Box::into_raw(Box::new(FttsEngine {
            loaded,
            engine,
            bundle,
            last_profile_json: CString::new("{}").expect("static JSON"),
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
    // SAFETY: forwards the identical pointer contract with progress disabled.
    #[allow(unsafe_code)]
    unsafe {
        ftts_synthesize_with_progress(
            engine,
            text,
            speaker,
            speaker_len,
            seed,
            None,
            std::ptr::null_mut(),
            out_pcm,
            out_len,
        )
    }
}

/// Synthesizes with the same ownership contract as [`ftts_synthesize`] while forwarding
/// the engine's real privacy-safe lifecycle events to `on_progress`.
#[allow(unsafe_code)] // audited export, part of the C ABI surface
#[unsafe(no_mangle)]
/// # Safety
/// As [`ftts_synthesize`]. When present, `on_progress` and `ctx` follow the callback
/// contract in `ftts_ffi.h` for the complete synchronous call.
pub unsafe extern "C" fn ftts_synthesize_with_progress(
    engine: *mut FttsEngine,
    text: *const c_char,
    speaker: *const f32,
    speaker_len: usize,
    seed: u64,
    on_progress: Option<FttsProgressFn>,
    ctx: *mut c_void,
    out_pcm: *mut *mut f32,
    out_len: *mut usize,
) -> i32 {
    guarded(1, || {
        if out_pcm.is_null() || out_len.is_null() {
            set_error("null pointer to ftts_synthesize");
            return 1;
        }
        // A failed call never leaves caller-owned sentinel values looking like a
        // successful allocation. This also makes retry/error paths safe for C hosts
        // that unconditionally inspect the output pair after the status code.
        // SAFETY: both output pointers were checked non-null immediately above.
        #[allow(unsafe_code)]
        unsafe {
            *out_pcm = std::ptr::null_mut();
            *out_len = 0;
        }
        if engine.is_null() || text.is_null() || speaker.is_null() {
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
        let progress = ProgressEmitter {
            callback: on_progress,
            ctx,
            cancellation: Some(cancellation.clone()),
        };
        let predicted_max_frames = Arc::new(AtomicU64::new(0));
        let decoded_frames = Arc::new(AtomicU64::new(0));
        let observer_progress = progress.clone();
        let observer_predicted = Arc::clone(&predicted_max_frames);
        let observer_decoded = Arc::clone(&decoded_frames);
        let observer = move |event: ftts_core::SynthesisEvent| {
            use ftts_core::SynthesisEvent;

            let translated = match event {
                SynthesisEvent::Admission { accepted } => {
                    let mut event = FttsProgressEvent::new(
                        FTTS_PROGRESS_KIND_ADMISSION,
                        FTTS_PROGRESS_STAGE_SYNTHESIS,
                    );
                    event.current = if accepted { 1 } else { 0 };
                    event.total = 1;
                    Some(event)
                }
                SynthesisEvent::ResourceAdmission {
                    admitted,
                    predicted_max_frames,
                    predicted_peak_bytes,
                    budget_bytes,
                } => {
                    observer_predicted.store(predicted_max_frames, Ordering::Release);
                    let mut event = FttsProgressEvent::new(
                        FTTS_PROGRESS_KIND_ADMISSION,
                        FTTS_PROGRESS_STAGE_RESOURCE_ADMISSION,
                    );
                    event.current = predicted_peak_bytes;
                    event.total = budget_bytes;
                    event.detail = predicted_max_frames;
                    if !admitted {
                        event.flags |= FTTS_PROGRESS_FLAG_INVALIDATES_OUTPUT;
                    }
                    Some(event)
                }
                SynthesisEvent::StageStarted { .. } => Some(FttsProgressEvent::new(
                    FTTS_PROGRESS_KIND_STAGE_STARTED,
                    FTTS_PROGRESS_STAGE_SYNTHESIS,
                )),
                SynthesisEvent::StageFinished { elapsed, .. } => {
                    let mut event = FttsProgressEvent::new(
                        FTTS_PROGRESS_KIND_STAGE_FINISHED,
                        FTTS_PROGRESS_STAGE_SYNTHESIS,
                    );
                    event.elapsed_ms = elapsed.as_secs_f64() * 1_000.0;
                    Some(event)
                }
                SynthesisEvent::FrameProgress { frame } => {
                    let mut event =
                        FttsProgressEvent::new(FTTS_PROGRESS_KIND_UNIT, FTTS_PROGRESS_STAGE_FRAMES);
                    event.current = frame.saturating_add(1);
                    event.total = observer_predicted.load(Ordering::Acquire);
                    event.flags |= FTTS_PROGRESS_FLAG_TOTAL_IS_UPPER_BOUND;
                    Some(event)
                }
                SynthesisEvent::TextPrepared { token_count, .. } => {
                    let mut event =
                        FttsProgressEvent::new(FTTS_PROGRESS_KIND_UNIT, FTTS_PROGRESS_STAGE_TEXT);
                    event.current = token_count as u64;
                    Some(event)
                }
                SynthesisEvent::PacketEmitted {
                    frame_count,
                    sample_count,
                } => {
                    let frame_count = u64::try_from(frame_count).unwrap_or(u64::MAX);
                    let current = observer_decoded
                        .fetch_add(frame_count, Ordering::AcqRel)
                        .saturating_add(frame_count);
                    let mut event =
                        FttsProgressEvent::new(FTTS_PROGRESS_KIND_UNIT, FTTS_PROGRESS_STAGE_CODEC);
                    event.current = current;
                    event.total = observer_predicted.load(Ordering::Acquire);
                    event.detail = sample_count as u64;
                    event.flags |= FTTS_PROGRESS_FLAG_TOTAL_IS_UPPER_BOUND;
                    Some(event)
                }
                SynthesisEvent::Health { event: health } => {
                    let mut event = FttsProgressEvent::new(
                        FTTS_PROGRESS_KIND_HEALTH,
                        FTTS_PROGRESS_STAGE_HEALTH,
                    );
                    if health.invalidates_output() {
                        event.flags |= FTTS_PROGRESS_FLAG_INVALIDATES_OUTPUT;
                    }
                    Some(event)
                }
                SynthesisEvent::TextUnderrun { .. }
                | SynthesisEvent::TextStallEnded { .. }
                | SynthesisEvent::TextAppendRejected { .. } => None,
            };
            if let Some(event) = translated {
                observer_progress.emit(event);
            }
        };
        match synthesize(
            &engine.loaded,
            &engine.engine,
            &request,
            // The v1 wire surface carries x-vectors only; ICL packs bypass the FFI
            // fast path by contract (`VoiceConditioning::is_xvector`).
            &VoiceConditioning::XVector(speaker.to_vec()),
            seed,
            &cancellation,
            &observer,
            // Whole-buffer ABI: packet size is unobservable to callers until the
            // streaming FFI (chunk callback) exposes it; 4 is the historical cadence.
            4,
            None,
            None,
        ) {
            Ok(audio) => {
                if cancellation.is_cancelled() {
                    return FTTS_SYNTH_CANCELLED;
                }
                engine.last_profile_json = synthesis_profile_json();
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
            Err(_) if cancellation.is_cancelled() => FTTS_SYNTH_CANCELLED,
            Err(error) => {
                set_error(error.to_string());
                1
            }
        }
    })
}

/// One decoded packet callback, C-shaped. See the header's contract comment.
pub type FttsPacketFn = unsafe extern "C" fn(
    ctx: *mut std::ffi::c_void,
    samples: *const f32,
    len: usize,
    frame_index: u64,
) -> i32;

/// The distinct status for a callback-requested cancellation (mirrors the CLI's
/// exit-code-6 semantics).
pub const FTTS_SYNTH_CANCELLED: i32 = 6;

/// Bridges the engine's packet sink onto the C callback: f32 end to end (the engine's
/// native output and AVAudioEngine's native input — an s16 ABI would force a pointless
/// f32->s16->f32 round trip on the phone; s16 is the CLI raw-stdout pipe contract, a
/// different reader). A nonzero callback return cancels via the token, so the engine
/// stops at its next frame checkpoint and the run reports Cancelled rather than error.
struct CallbackSink {
    on_packet: FttsPacketFn,
    ctx: *mut std::ffi::c_void,
    cancellation: CancellationToken,
    frames_delivered: u64,
    cancelled_by_callback: bool,
}

// SAFETY: the sink crosses onto the engine's decode thread — exactly what the header
// contract promises the caller ("on_packet receives each packet ON THE ENGINE'S DECODE
// THREAD"): `on_packet` and `ctx` must therefore be safe to invoke from that thread,
// which is the caller's obligation, stated in the header. The struct's own fields
// carry no Rust-side aliasing: the pointers are only passed back to C.
// SAFETY: summary of the full note above — caller-owned thread contract, no Rust-side aliasing.
#[allow(unsafe_code)]
unsafe impl Send for CallbackSink {}

impl ftts_cli::synth::PcmPacketSink for CallbackSink {
    fn deliver(&mut self, samples: &[f32], frames: usize) -> Result<(), ftts_cli::FttsError> {
        // SAFETY: `on_packet` and `ctx` are the caller's pair per the header contract;
        // the samples pointer/len describe this packet's slice, valid for the call.
        #[allow(unsafe_code)]
        let verdict = unsafe {
            (self.on_packet)(
                self.ctx,
                samples.as_ptr(),
                samples.len(),
                self.frames_delivered,
            )
        };
        self.frames_delivered += frames as u64;
        if verdict != 0 {
            self.cancelled_by_callback = true;
            self.cancellation.cancel();
            // Not an error: the engine notices the token at the next frame boundary and
            // winds down as a cancellation; erroring here would misreport the outcome.
        }
        Ok(())
    }
}

/// Streaming synthesis: each decoded packet reaches `on_packet` the moment it exists.
/// 0 on success; [`FTTS_SYNTH_CANCELLED`] when the callback cancelled; other nonzero on
/// failure with [`ftts_last_error_message`] set.
#[allow(unsafe_code)] // audited export, part of the C ABI surface
#[unsafe(no_mangle)]
/// # Safety
/// As [`ftts_synthesize`], plus: `on_packet` must be a valid function pointer following
/// the header's contract (prompt return, no unwinding, samples copied out), and `ctx`
/// must be whatever that callback expects.
pub unsafe extern "C" fn ftts_synthesize_streaming(
    engine: *mut FttsEngine,
    text: *const c_char,
    speaker: *const f32,
    speaker_len: usize,
    seed: u64,
    packet_frames: usize,
    on_packet: FttsPacketFn,
    ctx: *mut std::ffi::c_void,
) -> i32 {
    guarded(1, || {
        if engine.is_null() || text.is_null() || speaker.is_null() {
            set_error("null pointer to ftts_synthesize_streaming");
            return 1;
        }
        if speaker_len != SPEAKER_WIDTH {
            set_error(format!(
                "speaker vector must be {SPEAKER_WIDTH} floats, got {speaker_len}"
            ));
            return 1;
        }
        if packet_frames == 0 {
            set_error("packet_frames must be at least 1");
            return 1;
        }
        // SAFETY: as ftts_synthesize — unique engine borrow under the caller's
        // serialization; NUL-terminated text; speaker_len floats behind speaker.
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
        let mut sink = CallbackSink {
            on_packet,
            ctx,
            cancellation: cancellation.clone(),
            frames_delivered: 0,
            cancelled_by_callback: false,
        };
        let result = synthesize(
            &engine.loaded,
            &engine.engine,
            &request,
            &VoiceConditioning::XVector(speaker.to_vec()),
            seed,
            &cancellation,
            &observer,
            packet_frames,
            None,
            Some(&mut sink),
        );
        match result {
            // A callback request owns the outcome even if it arrived on the final
            // packet and the generator had no later checkpoint at which to turn its
            // otherwise-successful return into a cancellation error.
            _ if sink.cancelled_by_callback => FTTS_SYNTH_CANCELLED,
            Ok(_) => {
                engine.last_profile_json = synthesis_profile_json();
                0
            }
            Err(error) => {
                set_error(error.to_string());
                1
            }
        }
    })
}

/// JSON attribution for the most recent successful synthesis on this engine.
///
/// The pointer remains valid until the next successful synthesis or engine close. A null engine
/// returns an empty JSON object so diagnostics never need to manufacture a sentinel allocation.
#[allow(unsafe_code)] // audited export, part of the C ABI surface
#[unsafe(no_mangle)]
/// # Safety
/// `engine` must be null or a live pointer from [`ftts_engine_open`]. The caller serializes access.
pub unsafe extern "C" fn ftts_last_synthesis_profile_json(
    engine: *const FttsEngine,
) -> *const c_char {
    if engine.is_null() {
        return c"{}".as_ptr();
    }
    // SAFETY: the header requires a live engine and serialized access; this immutable borrow lasts
    // only long enough to obtain the CString's stable pointer.
    #[allow(unsafe_code)]
    unsafe {
        (&*engine).last_profile_json.as_ptr()
    }
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

/// 1 when the neural denoiser artifact is present in the engine's model directory,
/// 0 when it is not. This is the same check the enrollment pipeline makes, asked of
/// the engine itself so the host UI can report the truth instead of guessing.
#[allow(unsafe_code)] // audited export, part of the C ABI surface
#[unsafe(no_mangle)]
/// # Safety
/// `engine` from [`ftts_engine_open`] with serialized access.
pub unsafe extern "C" fn ftts_denoise_available(engine: *const FttsEngine) -> i32 {
    guarded(0, || {
        if engine.is_null() {
            return 0;
        }
        // SAFETY: engine from ftts_engine_open with serialized access; shared read only.
        #[allow(unsafe_code)]
        let engine = unsafe { &*engine };
        i32::from(engine.bundle.root.join(DENOISE_ARTIFACT_RELPATH).is_file())
    })
}

/// Denoises mono 24 kHz f32 PCM through the neural denoiser. Writes an owned buffer of
/// the same length to `out_pcm` (release with [`ftts_pcm_free`]). Returns 0 on success;
/// nonzero when the denoiser is absent or fails (the caller keeps its original audio).
#[allow(unsafe_code)] // audited export, part of the C ABI surface
#[unsafe(no_mangle)]
/// # Safety
/// `engine` from [`ftts_engine_open`] with serialized access; `pcm` valid for `len`
/// floats; `out_pcm` non-null.
pub unsafe extern "C" fn ftts_denoise(
    engine: *const FttsEngine,
    pcm: *const f32,
    len: usize,
    out_pcm: *mut *mut f32,
) -> i32 {
    guarded(1, || {
        if out_pcm.is_null() {
            set_error("null pointer to ftts_denoise");
            return 1;
        }
        // SAFETY: the output slot was checked non-null immediately above.
        #[allow(unsafe_code)]
        unsafe {
            *out_pcm = std::ptr::null_mut();
        }
        if engine.is_null() || pcm.is_null() {
            set_error("null pointer to ftts_denoise");
            return 1;
        }
        // SAFETY: per the contract above; the engine is only read.
        #[allow(unsafe_code)]
        let (engine, pcm) = unsafe { (&*engine, std::slice::from_raw_parts(pcm, len)) };
        match denoise_pcm_24k(&engine.bundle, pcm) {
            Ok(Some(cleaned)) => {
                let mut cleaned = cleaned.into_boxed_slice();
                let pointer = cleaned.as_mut_ptr();
                let cleaned_len = cleaned.len();
                if cleaned_len != len {
                    set_error("denoiser changed the sample count");
                    return 1;
                }
                std::mem::forget(cleaned);
                // SAFETY: out_pcm is non-null per the check above; written once.
                #[allow(unsafe_code)]
                unsafe {
                    *out_pcm = pointer;
                }
                0
            }
            Ok(None) => {
                set_error("the neural denoiser is not in the model directory");
                1
            }
            Err(error) => {
                set_error(error.to_string());
                1
            }
        }
    })
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
        // Exactly the CLI's enrollment pipeline, cleanup included. Denoising engages
        // automatically when the FastEnhancer weights are in the model directory - the
        // same default `ftts enroll` applies - and inside the pipeline the neural route
        // is preferred with spectral subtraction as its fallback.
        let mut denoise_report = None;
        let denoise_available = engine.bundle.root.join(DENOISE_ARTIFACT_RELPATH).is_file();
        let cleanup = ReferenceCleanup {
            denoise: denoise_available.then_some(&mut denoise_report),
            dereverb: None,
        };
        let vector = match speaker_from_reference_pcm(&engine.bundle, pcm.to_vec(), cleanup) {
            Ok(vector) => vector,
            Err(error) => {
                set_error(error.to_string());
                return 1;
            }
        };
        if vector.len() != SPEAKER_WIDTH {
            set_error("speaker encoder produced an unexpected vector width");
            return 1;
        }
        // SAFETY: out is non-null and the header requires SPEAKER_WIDTH floats of room.
        #[allow(unsafe_code)]
        let target = unsafe { std::slice::from_raw_parts_mut(out, SPEAKER_WIDTH) };
        target.copy_from_slice(&vector);
        0
    })
}

// ------------------------------------------------------------------------ share video

/// The branded share-video frame renderer, host-encoded (AVAssetWriter on iOS).
pub struct FttsVideoRenderer {
    inner: ftts_video::FrameRenderer,
}

/// Frame width in pixels. Matches the CLI's `ftts make-video` output exactly.
#[allow(unsafe_code)] // audited export, part of the C ABI surface
#[unsafe(no_mangle)]
pub extern "C" fn ftts_video_width() -> u32 {
    ftts_video::WIDTH as u32
}

#[allow(unsafe_code)] // audited export, part of the C ABI surface
#[unsafe(no_mangle)]
pub extern "C" fn ftts_video_height() -> u32 {
    ftts_video::HEIGHT as u32
}

#[allow(unsafe_code)] // audited export, part of the C ABI surface
#[unsafe(no_mangle)]
pub extern "C" fn ftts_video_fps() -> u32 {
    ftts_video::FPS
}

/// Opens a renderer over finished speech PCM. Null on failure (see the error message).
///
/// # Safety
/// `pcm` must be valid for `len` floats; `voice_label` NUL-terminated UTF-8.
#[allow(unsafe_code)] // audited export, part of the C ABI surface
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ftts_video_open(
    pcm: *const f32,
    len: usize,
    sample_rate: u32,
    voice_label: *const c_char,
) -> *mut FttsVideoRenderer {
    guarded(std::ptr::null_mut(), || {
        if pcm.is_null() || voice_label.is_null() {
            set_error("null pointer to ftts_video_open");
            return std::ptr::null_mut();
        }
        // SAFETY: pcm valid for len floats and voice_label NUL-terminated, per the header.
        #[allow(unsafe_code)]
        let (samples, label) = unsafe {
            (
                std::slice::from_raw_parts(pcm, len).to_vec(),
                CStr::from_ptr(voice_label),
            )
        };
        let Ok(label) = label.to_str() else {
            set_error("voice label is not UTF-8");
            return std::ptr::null_mut();
        };
        match ftts_video::FrameRenderer::new(samples, sample_rate, label) {
            Ok(inner) => Box::into_raw(Box::new(FttsVideoRenderer { inner })),
            Err(error) => {
                set_error(error);
                std::ptr::null_mut()
            }
        }
    })
}

/// Total frames the clip renders to.
///
/// # Safety
/// `renderer` must come from [`ftts_video_open`] and not yet be closed.
#[allow(unsafe_code)] // audited export, part of the C ABI surface
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ftts_video_frame_count(renderer: *const FttsVideoRenderer) -> usize {
    if renderer.is_null() {
        return 0;
    }
    // SAFETY: valid per the contract above; shared read only.
    #[allow(unsafe_code)]
    unsafe { &*renderer }.inner.total_frames()
}

/// Renders one frame as RGB24 into `out`, which must hold width*height*3 bytes.
///
/// Concurrent calls over one renderer are sound: rendering reads immutable state
/// (`FrameRenderer` is `Sync` by construction, asserted in `ftts-video`), and the iOS
/// exporter leans on this to render chunks of frames in parallel.
///
/// # Safety
/// `renderer` from [`ftts_video_open`]; `out` valid for `ftts_video_width() *
/// ftts_video_height() * 3` bytes; open/close serialized against in-flight renders.
#[allow(unsafe_code)] // audited export, part of the C ABI surface
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ftts_video_render_frame(
    renderer: *const FttsVideoRenderer,
    frame: usize,
    out: *mut u8,
) -> i32 {
    guarded(1, || {
        if renderer.is_null() || out.is_null() {
            set_error("null pointer to ftts_video_render_frame");
            return 1;
        }
        let bytes = ftts_video::WIDTH * ftts_video::HEIGHT * 3;
        // SAFETY: per the contract above.
        #[allow(unsafe_code)]
        let (renderer, rgb) = unsafe { (&*renderer, std::slice::from_raw_parts_mut(out, bytes)) };
        if frame >= renderer.inner.total_frames() {
            set_error("frame index past the end of the clip");
            return 1;
        }
        renderer.inner.render_into(frame, rgb);
        0
    })
}

/// Renders one frame as BGRA32 with the given row stride (bytes), the layout
/// CoreVideo pixel buffers use. Same concurrency contract as
/// [`ftts_video_render_frame`].
///
/// # Safety
/// `renderer` from [`ftts_video_open`]; `out` valid for `stride * ftts_video_height()`
/// bytes with `stride >= ftts_video_width() * 4`; open/close serialized against
/// in-flight renders.
#[allow(unsafe_code)] // audited export, part of the C ABI surface
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ftts_video_render_frame_bgra(
    renderer: *const FttsVideoRenderer,
    frame: usize,
    out: *mut u8,
    stride: usize,
) -> i32 {
    guarded(1, || {
        if renderer.is_null() || out.is_null() {
            set_error("null pointer to ftts_video_render_frame_bgra");
            return 1;
        }
        if stride < ftts_video::WIDTH * 4 {
            set_error("stride narrower than a BGRA row");
            return 1;
        }
        let Some(buffer_len) = stride.checked_mul(ftts_video::HEIGHT) else {
            set_error("BGRA buffer size overflow");
            return 1;
        };
        // SAFETY: per the contract above.
        #[allow(unsafe_code)]
        let (renderer, bgra) =
            unsafe { (&*renderer, std::slice::from_raw_parts_mut(out, buffer_len)) };
        if frame >= renderer.inner.total_frames() {
            set_error("frame index past the end of the clip");
            return 1;
        }
        renderer.inner.render_into_bgra(frame, bgra, stride);
        0
    })
}

/// Releases a renderer. Null is a no-op.
///
/// # Safety
/// `renderer` must be null or from [`ftts_video_open`], passed exactly once.
#[allow(unsafe_code)] // audited export, part of the C ABI surface
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ftts_video_close(renderer: *mut FttsVideoRenderer) {
    if renderer.is_null() {
        return;
    }
    // SAFETY: exactly-once ownership transfer per the contract above.
    #[allow(unsafe_code)]
    drop(unsafe { Box::from_raw(renderer) });
}

#[cfg(test)]
#[allow(unsafe_code)] // tests exercise the C ABI exactly as a caller would
mod tests {
    use super::*;

    unsafe extern "C" fn record_and_cancel_progress(
        ctx: *mut c_void,
        event: *const FttsProgressEvent,
    ) -> i32 {
        // SAFETY: the test passes a unique pointer to a live Vec and a pointer to a
        // stack event that remains valid for this synchronous callback.
        #[allow(unsafe_code)]
        let (events, event) = unsafe { (&mut *ctx.cast::<Vec<FttsProgressEvent>>(), *event) };
        events.push(event);
        1
    }

    unsafe extern "C" fn record_progress(ctx: *mut c_void, event: *const FttsProgressEvent) -> i32 {
        // SAFETY: the test passes a live Mutex for the complete synchronous ABI call,
        // and the event pointer is valid for this callback invocation. The mutex is
        // required because generation and codec events may arrive from different threads.
        #[allow(unsafe_code)]
        let (events, event) = unsafe {
            (
                &*ctx.cast::<std::sync::Mutex<Vec<FttsProgressEvent>>>(),
                *event,
            )
        };
        events.lock().unwrap().push(event);
        0
    }

    #[test]
    fn progress_event_is_versioned_and_callback_can_cancel() {
        let cancellation = CancellationToken::new();
        let mut events: Vec<FttsProgressEvent> = Vec::new();
        let emitter = ProgressEmitter {
            callback: Some(record_and_cancel_progress),
            ctx: (&raw mut events).cast(),
            cancellation: Some(cancellation.clone()),
        };
        let mut event = FttsProgressEvent::new(FTTS_PROGRESS_KIND_UNIT, FTTS_PROGRESS_STAGE_FRAMES);
        event.current = 7;
        event.total = 12;
        event.flags = FTTS_PROGRESS_FLAG_TOTAL_IS_UPPER_BOUND;

        assert!(!emitter.emit(event));
        assert!(cancellation.is_cancelled());
        assert_eq!(events, vec![event]);
        assert_eq!(events[0].abi_version, FTTS_PROGRESS_ABI_VERSION);
    }

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
        // SAFETY: `name` is a live NUL-terminated CString, and `out` is a live buffer of exactly
        // SPEAKER_WIDTH floats — the width this entry point contracts to fill.
        assert_eq!(
            unsafe { ftts_preset_vector(name.as_ptr(), out.as_mut_ptr()) },
            0
        );
        assert!(
            out.iter().any(|v| *v != 0.0),
            "vector should be non-trivial"
        );

        let missing = CString::new("nobody").unwrap();
        // SAFETY: as above; an unknown name is a value error, not a pointer error, so the same
        // liveness argument holds on the failing path.
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
    fn failing_buffer_abis_clear_their_output_slots() {
        let text = CString::new("hello").unwrap();
        let speaker = [0.0_f32; SPEAKER_WIDTH];
        let input = [0.0_f32; 8];
        let mut pcm = std::ptr::dangling_mut::<f32>();
        let mut len = usize::MAX;

        // SAFETY: the input and output pointers are live for each call. A null engine is
        // deliberately supplied to exercise the value-error path before any model access.
        let synth_code = unsafe {
            ftts_synthesize(
                std::ptr::null_mut(),
                text.as_ptr(),
                speaker.as_ptr(),
                speaker.len(),
                0,
                &raw mut pcm,
                &raw mut len,
            )
        };
        assert_eq!(synth_code, 1);
        assert!(pcm.is_null());
        assert_eq!(len, 0);

        pcm = std::ptr::dangling_mut::<f32>();
        // SAFETY: as above; `input` is a live slice and the output slot is writable.
        let denoise_code =
            unsafe { ftts_denoise(std::ptr::null(), input.as_ptr(), input.len(), &raw mut pcm) };
        assert_eq!(denoise_code, 1);
        assert!(pcm.is_null());
    }

    #[test]
    fn open_refuses_a_missing_model_directory_with_a_message() {
        let dir = CString::new("/nonexistent/franken-model").unwrap();
        // SAFETY: `dir` is a live NUL-terminated CString for the whole call.
        let engine = unsafe { ftts_engine_open(dir.as_ptr()) };
        assert!(engine.is_null());
        // SAFETY: reading the thread-local error string this crate owns and keeps alive.
        #[allow(unsafe_code)]
        let message = unsafe { CStr::from_ptr(ftts_last_error_message()) }
            .to_str()
            .unwrap();
        assert!(!message.is_empty());
        // SAFETY: closing a null handle is explicitly a no-op, which is what is being asserted.
        unsafe { ftts_engine_close(engine) };
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
        // SAFETY: `c_dir` is a live NUL-terminated CString naming a directory that exists.
        let events = std::sync::Mutex::new(Vec::<FttsProgressEvent>::new());
        let context = (&raw const events).cast_mut().cast();
        let engine = unsafe {
            ftts_engine_open_with_progress(c_dir.as_ptr(), Some(record_progress), context)
        };
        assert!(!engine.is_null(), "open failed: {}", last_error());
        {
            let observed = events.lock().unwrap();
            for stage in [
                FTTS_PROGRESS_STAGE_MODEL_BUNDLE,
                FTTS_PROGRESS_STAGE_MODEL_WEIGHTS,
                FTTS_PROGRESS_STAGE_RUNTIME,
            ] {
                assert!(
                    observed.iter().any(|event| {
                        event.kind == FTTS_PROGRESS_KIND_STAGE_FINISHED && event.stage == stage
                    }),
                    "missing completed load stage {stage}: {observed:?}"
                );
            }
        }
        events.lock().unwrap().clear();
        let mut speaker = vec![0.0_f32; SPEAKER_WIDTH];
        let matt = CString::new("matt").unwrap();
        // SAFETY: live CString name, and `speaker` holds exactly SPEAKER_WIDTH floats.
        assert_eq!(
            unsafe { ftts_preset_vector(matt.as_ptr(), speaker.as_mut_ptr()) },
            0
        );
        let text = CString::new("Hi.").unwrap();
        let mut pcm: *mut f32 = std::ptr::null_mut();
        let mut len = 0_usize;
        // SAFETY: `engine` is a non-null handle from a successful open; `text` is a live CString;
        // `speaker` holds SPEAKER_WIDTH floats and its length is passed alongside; `pcm` and `len`
        // are live out-parameters this call writes and the caller frees below.
        let code = unsafe {
            ftts_synthesize_with_progress(
                engine,
                text.as_ptr(),
                speaker.as_ptr(),
                SPEAKER_WIDTH,
                0,
                Some(record_progress),
                context,
                &raw mut pcm,
                &raw mut len,
            )
        };
        assert_eq!(code, 0, "synthesize failed: {}", last_error());
        assert!(len > 0 && !pcm.is_null());
        let observed = events.lock().unwrap();
        assert!(
            observed.iter().any(|event| {
                event.kind == FTTS_PROGRESS_KIND_UNIT
                    && event.stage == FTTS_PROGRESS_STAGE_FRAMES
                    && event.current > 0
            }),
            "missing frame progress: {observed:?}"
        );
        assert!(
            observed.iter().any(|event| {
                event.kind == FTTS_PROGRESS_KIND_UNIT
                    && event.stage == FTTS_PROGRESS_STAGE_CODEC
                    && event.current > 0
                    && event.detail > 0
            }),
            "missing codec packet progress: {observed:?}"
        );
        drop(observed);
        // SAFETY: `pcm`/`len` are exactly the buffer this crate allocated in the call above, freed
        // once; `engine` is the handle from the matching open, closed once.
        unsafe { ftts_pcm_free(pcm, len) };
        // SAFETY: as above — one close for one successful open.
        unsafe { ftts_engine_close(engine) };
    }

    #[test]
    fn video_renderer_round_trips_through_the_abi() {
        let pcm: Vec<f32> = (0..24_000).map(|i| (i as f32 * 0.01).sin() * 0.4).collect();
        let label = CString::new("matt").unwrap();
        // SAFETY: `pcm` is a live Vec whose length is passed alongside it, and `label` is a live
        // NUL-terminated CString; both outlive the call.
        let renderer = unsafe { ftts_video_open(pcm.as_ptr(), pcm.len(), 24_000, label.as_ptr()) };
        assert!(!renderer.is_null(), "{}", last_error());
        // SAFETY: `renderer` is a non-null handle from the open above.
        let frames = unsafe { ftts_video_frame_count(renderer) };
        assert_eq!(frames, 30, "one second at 30 fps");
        // Sized in RGB24, matching `ftts_video_render_frame`'s contract. The BGRA32 entry point is
        // a separate function with its own stride parameter; do not conflate the two here.
        let mut rgb = vec![0u8; (ftts_video_width() * ftts_video_height() * 3) as usize];
        // SAFETY: live renderer, and `rgb` is exactly width * height * 3 bytes as contracted.
        assert_eq!(
            unsafe { ftts_video_render_frame(renderer, 0, rgb.as_mut_ptr()) },
            0
        );
        assert!(rgb.iter().any(|&b| b != 0), "frame should not be black");
        // SAFETY: as above; a past-the-end index is a value error the callee rejects before it
        // touches `out`, so the buffer contract is unchanged.
        assert_eq!(
            unsafe { ftts_video_render_frame(renderer, frames, rgb.as_mut_ptr()) },
            1,
            "past-the-end frame must refuse"
        );
        let mut one_byte = 0_u8;
        // SAFETY: the hostile stride is rejected before the callee constructs a slice or
        // touches the one-byte output. The live renderer/frame satisfy the other preconditions.
        assert_eq!(
            unsafe { ftts_video_render_frame_bgra(renderer, 0, &raw mut one_byte, usize::MAX,) },
            1,
            "overflowing BGRA geometry must be rejected"
        );
        // SAFETY: one close for one successful open, with no render in flight.
        unsafe { ftts_video_close(renderer) };
    }

    fn last_error() -> String {
        // SAFETY: reading the thread-local error string this crate owns; it is always a valid
        // NUL-terminated buffer, empty before any failure, and never freed while borrowed here.
        #[allow(unsafe_code)]
        unsafe { CStr::from_ptr(ftts_last_error_message()) }
            .to_string_lossy()
            .into_owned()
    }
}
