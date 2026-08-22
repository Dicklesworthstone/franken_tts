//! `ftts talk` — one long-lived session per conversation (bead frankentts-edz0).
//!
//! One process holds the warm model for a whole spoken exchange: NDJSON ops arrive on
//! stdin, schema-v2 events leave on stdout (the frozen `session_protocol` vocabulary),
//! and raw mono s16le 24 kHz PCM flows on a SEPARATE channel — `/dev/fd/3` when the
//! spawner inherited fd 3 open (Unix), else `--pcm-out <path|fifo>`. PCM never
//! interleaves with NDJSON anywhere; the audio channel carries one continuous
//! session-global byte stream, silence between utterances is the ABSENCE of writes,
//! and `audio` events' `byte_offset`/`bytes` are the cross-channel sequencing truth.
//!
//! # Thread topology (one parallel owner, bounded channels only)
//!
//! - **stdin reader**: line-oriented; parses and forwards ops to the router.
//! - **router (this thread)**: owns session state (contexts, the single active
//!   utterance) and is the only producer of lifecycle decisions. Ops touching the
//!   active utterance travel through the engine's bounded text feed / the cancel
//!   token; a `say` starting a new utterance becomes a job for the synthesis thread.
//! - **synthesis thread**: the single parallel owner — runs `synth::synthesize` per
//!   utterance over the shared warm [`LoadedModel`] (the int8 route is a process asset
//!   cached inside it), with the engine's continuation feed and a packet sink.
//! - **PCM writer**: consumes decoded packets, writes + flushes the audio channel,
//!   keeps the delivered-byte accounting, and emits `audio` events.
//! - **stdout writer**: the one owner of stdout, so NDJSON lines stay atomic.
//!
//! Deadlock obligations: audio delivery never waits on the event stream (the PCM
//! writer's event sends are lossless but its AUDIO write happens first), and control
//! stays live while audio is parked (cancel trips the engine's per-frame checkpoint
//! and the sink's bounded send). A stalled stdout consumer can pause synthesis —
//! bounded and cancel-aware — which the fail-closed consumer contract (drain both
//! streams concurrently) already forbids; the coalescable event classes (`progress`,
//! `buffer`) are dropped under pressure rather than blocking audio.
//!
//! Statelessness: no transcript, no audio, no history is persisted anywhere; session
//! lifetime is the process. stdin EOF == `shutdown`. The resident daemon is never
//! consulted or spawned — this process IS the warm engine for its conversation.

use std::collections::HashMap;
use std::io::{BufRead, Write};
use std::path::PathBuf;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, SyncSender, TrySendError, sync_channel};
use std::time::Instant;

use serde_json::{Value, json};

use crate::error::{FttsError, FttsExitCode};
use crate::session_protocol::{SessionEvent, utterance_seed, validate_session_op};
use crate::synth::{self, LoadedModel, PcmPacketSink};
use ftts_core::{
    BoundedSender, CancellationToken, SynthesisEvent, SynthesisRequest, TextControl, TextPreparer,
    TtsEngine, text_control_queue,
};

/// Interactive packet size: the session exists for conversation, so first audio must
/// not wait for a 4-frame packet (the profile contract, bead frankentts-6xcf).
const PACKET_FRAMES: usize = 1;
/// Capacity of the engine text feed: far more sentence chunks than any LLM sends
/// between frame boundaries; bounded so a runaway producer parks instead of ballooning.
const TEXT_FEED_CAPACITY: usize = 64;
/// Decoded packets in flight between the codec worker and the PCM writer (one frame
/// each under the interactive packet size — a small, deliberate delivery lead).
const PCM_QUEUE_PACKETS: usize = 32;
/// Event-line queue to the stdout owner.
const EVENT_QUEUE_LINES: usize = 256;

/// Work for the PCM writer: decoded packets, or a drain fence.
enum PcmJob {
    /// One decoded packet on its way to the audio channel.
    Audio {
        context: String,
        utterance: u64,
        samples: Vec<f32>,
        frames: usize,
        /// Set on the first packet of an utterance: milliseconds from synthesis start.
        ttfa_ms: Option<u64>,
    },
    /// A drain barrier: acknowledged only after every prior packet is written and
    /// accounted. The synthesis thread fences before reporting an utterance done, so
    /// the router's receipt (frames_delivered et al.) can never race the last write.
    Fence(SyncSender<()>),
}

/// Everything the router tracks per open context.
struct ContextState {
    speaker: Vec<f32>,
    seed: u64,
    utterances_started: u64,
    /// Withheld tail of the last `continue:true` chunk — never ends mid-word, so BPE
    /// cannot merge across a chunk seam (the producer half of the chunked==whole gate).
    tail: String,
}

/// The one in-flight utterance.
struct ActiveUtterance {
    context: String,
    utterance: u64,
    feed: BoundedSender<TextControl>,
    cancellation: CancellationToken,
    /// Target-token count admitted so far (initial text + accepted appends), for the
    /// truncation receipt.
    target_tokens: u64,
    /// The RAW target ids (pre-wrapper) fed so far: what the listener can actually
    /// hear, and what the spoken-prefix decode must draw from — the assistant wrapper
    /// is prompt scaffolding, never speech.
    target_ids: Vec<u32>,
    /// Whether the text stream was finished (continue:false, flush, or EOF rule).
    text_finished: bool,
}

/// What the synthesis thread reports back to the router when an utterance ends.
enum UtteranceOutcome {
    Complete {
        frames: u64,
        ttfa_ms: Option<u64>,
        elapsed_ms: u64,
    },
    Cancelled,
    Failed(String),
}

/// Router inbox: stdin ops and synthesis-thread completions share one queue.
enum RouterIn {
    Line(String),
    StdinClosed,
    Done {
        context: String,
        utterance: u64,
        outcome: UtteranceOutcome,
    },
}

/// A `say` job handed to the synthesis thread.
struct SpeakJob {
    context: String,
    utterance: u64,
    text: String,
    normalization: ftts_core::NormalizationOptions,
    effective_seed: u64,
    speaker: Vec<f32>,
    feed_rx: ftts_core::BoundedReceiver<TextControl>,
    cancellation: CancellationToken,
}

/// Sink bridging the codec worker to the PCM writer queue.
struct QueueSink {
    queue: SyncSender<PcmJob>,
    context: String,
    utterance: u64,
    started: Instant,
    first_sent: bool,
}

impl PcmPacketSink for QueueSink {
    fn deliver(&mut self, samples: &[f32], frames: usize) -> Result<(), FttsError> {
        let ttfa_ms = if self.first_sent {
            None
        } else {
            self.first_sent = true;
            Some(u64::try_from(self.started.elapsed().as_millis()).unwrap_or(u64::MAX))
        };
        self.queue
            .send(PcmJob::Audio {
                context: self.context.clone(),
                utterance: self.utterance,
                samples: samples.to_vec(),
                frames,
                ttfa_ms,
            })
            .map_err(|_| {
                FttsError::SessionTransport("the PCM writer stopped accepting packets".to_owned())
            })
    }
}

/// Shared counters the PCM writer maintains and the router reads for receipts.
struct DeliveryLedger {
    session_bytes: AtomicU64,
    /// Frames of the CURRENT utterance actually written to the audio channel.
    utterance_frames: AtomicU64,
    /// The CURRENT utterance's delivery TTFA in ms (u64::MAX until the first packet
    /// lands). Same clock as the first `audio` event's `ttfa_ms`, so `speak_complete`
    /// reports the number the orchestrator already saw — one basis, no confusion.
    utterance_ttfa_ms: AtomicU64,
}

impl Default for DeliveryLedger {
    fn default() -> Self {
        Self {
            session_bytes: AtomicU64::new(0),
            utterance_frames: AtomicU64::new(0),
            utterance_ttfa_ms: AtomicU64::new(u64::MAX),
        }
    }
}

/// Emit one event line to the stdout owner. Lifecycle events block (never dropped);
/// pass `coalescable` for progress-class events that may be shed under pressure.
fn emit(
    events: &SyncSender<String>,
    object: serde_json::Map<String, Value>,
    coalescable: bool,
) -> Result<(), FttsError> {
    let line = Value::Object(object).to_string();
    if coalescable {
        match events.try_send(line) {
            Ok(()) | Err(TrySendError::Full(_)) => Ok(()),
            Err(TrySendError::Disconnected(_)) => Err(FttsError::SessionTransport(
                "the event writer stopped".to_owned(),
            )),
        }
    } else {
        events
            .send(line)
            .map_err(|_| FttsError::SessionTransport("the event writer stopped".to_owned()))
    }
}

/// The session id: same shape discipline as robot run ids — no text, no filesystem.
fn session_id() -> String {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.subsec_nanos());
    format!("s{:x}{nanos:08x}", std::process::id())
}

/// Where session PCM goes.
enum AudioChannel {
    File(std::fs::File),
}

impl AudioChannel {
    fn write_all_flush(&mut self, bytes: &[u8]) -> std::io::Result<()> {
        match self {
            Self::File(file) => {
                file.write_all(bytes)?;
                file.flush()
            }
        }
    }
}

/// Resolve the audio channel: `--pcm-out` wins; otherwise fd 3 via `/dev/fd/3` on Unix
/// (a pure-`std` way to adopt an inherited descriptor — no unsafe, per the workspace
/// forbid). On Windows fd inheritance is a unixism: `--pcm-out` is the only route.
fn open_audio_channel(pcm_out: Option<&PathBuf>) -> Result<AudioChannel, FttsError> {
    if let Some(path) = pcm_out {
        let file = std::fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(false)
            .open(path)
            .map_err(|error| {
                FttsError::Usage(format!("cannot open --pcm-out {}: {error}", path.display()))
            })?;
        return Ok(AudioChannel::File(file));
    }
    #[cfg(unix)]
    {
        if let Ok(file) = std::fs::OpenOptions::new().write(true).open("/dev/fd/3") {
            return Ok(AudioChannel::File(file));
        }
        Err(FttsError::Usage(
            "ftts talk needs an audio channel: spawn with fd 3 open, or pass --pcm-out <path|fifo>"
                .to_owned(),
        ))
    }
    #[cfg(not(unix))]
    {
        Err(FttsError::Usage(
            "on this platform pass --pcm-out <path>; fd-3 inheritance is Unix-only".to_owned(),
        ))
    }
}

/// Split `text` so the emitted head never ends mid-word: everything up to and
/// including the last whitespace goes out now, the trailing partial word is withheld
/// for the next chunk (or a flush). `continue:false` sends everything.
fn split_tail(text: &str) -> (&str, &str) {
    match text.rfind(char::is_whitespace) {
        // The whitespace goes WITH the tail: BPE attaches a space to the FOLLOWING
        // word ("Ġword"), so a head ending in space would tokenize the next chunk's
        // first word bare and diverge from the whole-text stream — the exact seam the
        // chunked==whole gate exists to catch (and did, live). Known residual: a run
        // of MULTIPLE whitespace characters at a chunk boundary can still split a
        // multi-space token; LLM sentence streams do not produce those, and the
        // chunked==whole gate is the tripwire if one ever matters.
        Some(at) => text.split_at(at),
        None => ("", text),
    }
}

/// Run one talk session to completion. Returns the process exit code.
/// Owns the process's real stdio: a session is inherently a live transport, its lock
/// types are `Send` (the background reader/writer threads need that), and injected
/// `dyn` handles would not be. The test surface is process-level e2e (the 7pgn
/// driver), same as the resident daemon.
pub fn run_talk(
    bundle: &synth::ModelBundle,
    pcm_out: Option<&PathBuf>,
    voices: &(dyn Fn(&str) -> Result<Vec<f32>, FttsError> + Sync),
    normalization: ftts_core::NormalizationOptions,
    default_context_seed: u64,
) -> Result<(), FttsError> {
    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    let mut audio = open_audio_channel(pcm_out)?;
    // Eager load, announced by session_start: a conversation's first turn must not pay
    // the multi-second hydration silently. Memory-sensitive callers use one-shot `say`.
    let loaded = LoadedModel::load(bundle)?;
    let engine = TtsEngine::from_process_environment()
        .map_err(|error| FttsError::Generic(format!("cannot start the engine: {error}")))?;
    let sid = session_id();

    let (event_tx, event_rx) = sync_channel::<String>(EVENT_QUEUE_LINES);
    let (router_tx, router_rx) = sync_channel::<RouterIn>(EVENT_QUEUE_LINES);
    let (pcm_tx, pcm_rx) = sync_channel::<PcmJob>(PCM_QUEUE_PACKETS);
    let (job_tx, job_rx) = sync_channel::<SpeakJob>(1);
    let ledger = DeliveryLedger::default();
    let seq = AtomicU64::new(0);
    let next_seq = || seq.fetch_add(1, Ordering::Relaxed);

    std::thread::scope(|scope| -> Result<(), FttsError> {
        // --- stdout owner ---------------------------------------------------------
        // The owned `Stdout` handle is Send; its lock is taken INSIDE the thread (the
        // lock guards are deliberately !Send). One thread, one owner, atomic lines.
        let writer = scope.spawn(move || -> Result<(), FttsError> {
            let mut out = stdout.lock();
            while let Ok(line) = event_rx.recv() {
                writeln!(out, "{line}")
                    .and_then(|()| out.flush())
                    .map_err(|error| {
                        FttsError::SessionTransport(format!("cannot write events: {error}"))
                    })?;
            }
            Ok(())
        });

        // --- stdin reader ---------------------------------------------------------
        // DETACHED, not scope-joined: a client that sends `shutdown` while keeping
        // stdin open would otherwise pin the scope (and the process) on a blocked
        // read_line forever. The reader owns the process Stdin ('static), its sends
        // fail harmlessly once the router is gone, and process exit reaps it.
        let stdin_tx = router_tx.clone();
        std::thread::spawn(move || {
            let mut stdin = stdin.lock();
            let mut line = String::new();
            loop {
                line.clear();
                match stdin.read_line(&mut line) {
                    Ok(0) | Err(_) => {
                        let _ = stdin_tx.send(RouterIn::StdinClosed);
                        return;
                    }
                    Ok(_) => {
                        if stdin_tx.send(RouterIn::Line(line.clone())).is_err() {
                            return;
                        }
                    }
                }
            }
        });

        // --- PCM writer -----------------------------------------------------------
        let pcm_events = event_tx.clone();
        let ledger_ref = &ledger;
        let audio_ref = &mut audio;
        let sid_pcm = sid.clone();
        let seq_ref = &seq;
        let pcm_writer = scope.spawn(move || -> Result<(), FttsError> {
            let mut frame_index_in_utterance: HashMap<(String, u64), u64> = HashMap::new();
            while let Ok(job) = pcm_rx.recv() {
                let PcmJob::Audio {
                    context,
                    utterance,
                    samples,
                    frames,
                    ttfa_ms,
                } = job
                else {
                    if let PcmJob::Fence(ack) = job {
                        let _ = ack.send(());
                    }
                    continue;
                };
                let mut bytes = Vec::with_capacity(samples.len() * 2);
                for sample in &samples {
                    bytes
                        .extend_from_slice(&ftts_core::audio::sample_to_i16(*sample).to_le_bytes());
                }
                let offset_before = ledger_ref.session_bytes.load(Ordering::Relaxed);
                // Audio first, accounting second, event last: the event stream can lag
                // audio, never the reverse.
                audio_ref.write_all_flush(&bytes).map_err(|error| {
                    FttsError::SessionTransport(format!("cannot write PCM: {error}"))
                })?;
                ledger_ref
                    .session_bytes
                    .fetch_add(bytes.len() as u64, Ordering::Relaxed);
                ledger_ref
                    .utterance_frames
                    .fetch_add(frames as u64, Ordering::Relaxed);
                if let Some(ttfa_ms) = ttfa_ms {
                    ledger_ref
                        .utterance_ttfa_ms
                        .store(ttfa_ms, Ordering::Relaxed);
                }
                let key = (context.clone(), utterance);
                let frame_index = frame_index_in_utterance.entry(key).or_insert(0);
                let mut event =
                    SessionEvent::Audio.object(&sid_pcm, seq_ref.fetch_add(1, Ordering::Relaxed));
                event.insert("context".to_owned(), json!(context));
                event.insert("utterance".to_owned(), json!(utterance));
                event.insert("byte_offset".to_owned(), json!(offset_before));
                event.insert("bytes".to_owned(), json!(bytes.len() as u64));
                event.insert("frames".to_owned(), json!(frames as u64));
                event.insert("frame_index".to_owned(), json!(*frame_index));
                if let Some(ttfa_ms) = ttfa_ms {
                    event.insert("ttfa_ms".to_owned(), json!(ttfa_ms));
                }
                *frame_index += frames as u64;
                emit(&pcm_events, event, false)?;
            }
            Ok(())
        });

        // --- synthesis thread -------------------------------------------------------
        let synth_events = event_tx.clone();
        let synth_router = router_tx.clone();
        let loaded_ref = &loaded;
        let engine_ref = &engine;
        let pcm_tx_synth = pcm_tx.clone();
        let sid_synth = sid.clone();
        let synthesis = scope.spawn(move || {
            while let Ok(job) = job_rx.recv() {
                let started = Instant::now();
                let mut start = SessionEvent::SpeakStart
                    .object(&sid_synth, seq_ref.fetch_add(1, Ordering::Relaxed));
                start.insert("context".to_owned(), json!(job.context));
                start.insert("utterance".to_owned(), json!(job.utterance));
                start.insert("seed".to_owned(), json!(job.effective_seed));
                if emit(&synth_events, start, false).is_err() {
                    return;
                }
                let mut sink = QueueSink {
                    queue: pcm_tx_synth.clone(),
                    context: job.context.clone(),
                    utterance: job.utterance,
                    started,
                    first_sent: false,
                };
                // Underruns and stall receipts surface live from the engine's observer;
                // progress-class, so shed under pressure rather than blocking frames.
                let observer_state = Mutex::new(synth_events.clone());
                let observer_sid = sid_synth.clone();
                let observer_context = job.context.clone();
                let observer = move |event: SynthesisEvent| {
                    let SynthesisEvent::TextUnderrun { waited } = event else {
                        return;
                    };
                    let Ok(events) = observer_state.lock() else {
                        return;
                    };
                    let mut object = SessionEvent::TextUnderrun
                        .object(&observer_sid, seq_ref.fetch_add(1, Ordering::Relaxed));
                    object.insert("context".to_owned(), json!(observer_context));
                    object.insert(
                        "waited_ms".to_owned(),
                        json!(u64::try_from(waited.as_millis()).unwrap_or(u64::MAX)),
                    );
                    let _ = emit(&events, object, true);
                };
                let request = SynthesisRequest::new(job.text.clone())
                    .with_normalization_options(job.normalization.clone());
                let result = synth::synthesize(
                    loaded_ref,
                    engine_ref,
                    &request,
                    &job.speaker,
                    job.effective_seed,
                    &job.cancellation,
                    &observer,
                    PACKET_FRAMES,
                    Some(&job.feed_rx),
                    Some(&mut sink),
                );
                // Drain fence: every packet this utterance queued is written and
                // accounted before the router hears Done — receipts never race audio.
                let (fence_tx, fence_rx) = sync_channel::<()>(1);
                if pcm_tx_synth.send(PcmJob::Fence(fence_tx)).is_ok() {
                    let _ = fence_rx.recv();
                }
                let outcome = match result {
                    Ok(done) => UtteranceOutcome::Complete {
                        frames: done.frames,
                        ttfa_ms: done
                            .ttfa_audible
                            .or(done.ttfa)
                            .map(|d| u64::try_from(d.as_millis()).unwrap_or(u64::MAX)),
                        elapsed_ms: u64::try_from(started.elapsed().as_millis())
                            .unwrap_or(u64::MAX),
                    },
                    Err(error) if error.exit_code() == FttsExitCode::Cancelled => {
                        UtteranceOutcome::Cancelled
                    }
                    Err(error) => UtteranceOutcome::Failed(error.to_string()),
                };
                if synth_router
                    .send(RouterIn::Done {
                        context: job.context,
                        utterance: job.utterance,
                        outcome,
                    })
                    .is_err()
                {
                    return;
                }
            }
        });

        // --- router (this thread) ---------------------------------------------------
        let result = route(
            &sid,
            &bundle.root.display().to_string(),
            &engine,
            &loaded,
            &normalization,
            voices,
            default_context_seed,
            &router_rx,
            &event_tx,
            &job_tx,
            &pcm_tx,
            &ledger,
            &next_seq,
        );

        // Shutdown order: close the job queue (synthesis exits), then PCM, then events.
        drop(job_tx);
        drop(pcm_tx);
        synthesis.join().expect("synthesis thread");
        pcm_writer.join().expect("pcm writer")?;
        drop(event_tx);
        writer.join().expect("stdout owner")?;
        result
    })
}

/// The router loop: session state and the op vocabulary.
#[allow(clippy::too_many_arguments)]
fn route(
    sid: &str,
    model_label: &str,
    _engine: &TtsEngine,
    loaded: &LoadedModel,
    normalization: &ftts_core::NormalizationOptions,
    voices: &dyn Fn(&str) -> Result<Vec<f32>, FttsError>,
    default_context_seed: u64,
    inbox: &Receiver<RouterIn>,
    events: &SyncSender<String>,
    jobs: &SyncSender<SpeakJob>,
    _pcm: &SyncSender<PcmJob>,
    ledger: &DeliveryLedger,
    next_seq: &dyn Fn() -> u64,
) -> Result<(), FttsError> {
    let mut contexts: HashMap<String, ContextState> = HashMap::new();
    let mut active: Option<ActiveUtterance> = None;

    // The ready handshake.
    let mut start = SessionEvent::SessionStart.object(sid, next_seq());
    start.insert("version".to_owned(), json!(env!("CARGO_PKG_VERSION")));
    start.insert("model".to_owned(), json!(model_label));
    start.insert(
        "route".to_owned(),
        json!(if ftts_kernels::route::optimized_default("FTTS_INT8") {
            "int8"
        } else {
            "f32-reference"
        }),
    );
    start.insert(
        "pcm".to_owned(),
        json!({"format": "s16le", "rate": 24_000, "channels": 1}),
    );
    start.insert("pid".to_owned(), json!(std::process::id()));
    emit(events, start, false)?;

    let error_event = |kind: &str, message: String, remediation: &str, seq: u64| {
        let mut object = SessionEvent::SessionError.object(sid, seq);
        object.insert("kind".to_owned(), json!(kind));
        object.insert("message".to_owned(), json!(message));
        object.insert("remediation".to_owned(), json!(remediation));
        object
    };

    while let Ok(message) = inbox.recv() {
        match message {
            RouterIn::StdinClosed => {
                // EOF == shutdown: cancel anything in flight, EMIT ITS RECEIPT, then end.
                if let Some(state) = active.take() {
                    state.cancellation.cancel();
                    settle_utterance(sid, loaded, &state, inbox, events, ledger, next_seq)?;
                }
                break;
            }
            RouterIn::Done {
                context,
                utterance,
                outcome,
            } => {
                let state = active.take();
                let receipt = finish_utterance(
                    sid,
                    loaded,
                    state.as_ref(),
                    &context,
                    utterance,
                    outcome,
                    ledger,
                    next_seq,
                );
                emit(events, receipt, false)?;
            }
            RouterIn::Line(line) => {
                let trimmed = line.trim();
                if trimmed.is_empty() {
                    continue;
                }
                let value: Value = match serde_json::from_str(trimmed) {
                    Ok(value) => value,
                    Err(error) => {
                        emit(
                            events,
                            error_event(
                                "malformed",
                                format!("stdin line is not JSON: {error}"),
                                "send one JSON object per line; see `ftts robot schema --contract session`",
                                next_seq(),
                            ),
                            false,
                        )?;
                        continue;
                    }
                };
                let violations = validate_session_op(&value);
                if !violations.is_empty() {
                    emit(
                        events,
                        error_event(
                            "invalid-op",
                            format!("op rejected: {}", violations.join("; ")),
                            "fix the op shape; unknown fields and ops are refused fail-closed",
                            next_seq(),
                        ),
                        false,
                    )?;
                    continue;
                }
                let op = value["op"].as_str().unwrap_or_default().to_owned();
                let id = value.get("id").and_then(Value::as_str).map(str::to_owned);
                let context_name = value
                    .get("context")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_owned();

                // The ack is the id-echo mechanism and `ack.id` is REQUIRED by the frozen
                // schema: an op without a client id gets no ack (its effects are its
                // acknowledgement). `maybe_ack` returns None in that case.
                let maybe_ack = |seq: &dyn Fn() -> u64| -> Option<serde_json::Map<String, Value>> {
                    let id = id.as_ref()?;
                    let mut object = SessionEvent::Ack.object(sid, seq());
                    object.insert("id".to_owned(), json!(id));
                    object.insert("op".to_owned(), json!(op));
                    if !context_name.is_empty() {
                        object.insert("context".to_owned(), json!(context_name));
                    }
                    Some(object)
                };

                match op.as_str() {
                    "shutdown" => {
                        if let Some(ack) = maybe_ack(&next_seq) {
                            emit(events, ack, false)?;
                        }
                        if let Some(state) = active.take() {
                            state.cancellation.cancel();
                            settle_utterance(sid, loaded, &state, inbox, events, ledger, next_seq)?;
                        }
                        break;
                    }
                    "open" => {
                        if active
                            .as_ref()
                            .is_some_and(|utterance| utterance.context == context_name)
                        {
                            emit(
                                events,
                                error_event(
                                    "busy",
                                    format!("context {context_name} is speaking"),
                                    "cancel or wait for the receipt before reopening a context",
                                    next_seq(),
                                ),
                                false,
                            )?;
                            continue;
                        }
                        let voice = value
                            .get("voice")
                            .and_then(Value::as_str)
                            .unwrap_or("matt")
                            .to_owned();
                        let seed = value
                            .get("seed")
                            .and_then(Value::as_u64)
                            .unwrap_or(default_context_seed);
                        match voices(&voice) {
                            Ok(speaker) => {
                                contexts.insert(
                                    context_name.clone(),
                                    ContextState {
                                        speaker,
                                        seed,
                                        utterances_started: 0,
                                        tail: String::new(),
                                    },
                                );
                                if let Some(ack) = maybe_ack(&next_seq) {
                                    emit(events, ack, false)?;
                                }
                                let mut object = SessionEvent::ContextOpen.object(sid, next_seq());
                                object.insert("context".to_owned(), json!(context_name));
                                object.insert("voice".to_owned(), json!(voice));
                                object.insert("seed".to_owned(), json!(seed));
                                emit(events, object, false)?;
                            }
                            Err(error) => emit(
                                events,
                                error_event(
                                    "voice",
                                    error.to_string(),
                                    "name a preset, a .spk file, or a voice card image",
                                    next_seq(),
                                ),
                                false,
                            )?,
                        }
                    }
                    "close" => {
                        if active
                            .as_ref()
                            .is_some_and(|utterance| utterance.context == context_name)
                        {
                            emit(
                                events,
                                error_event(
                                    "busy",
                                    format!("context {context_name} is speaking"),
                                    "cancel or flush the utterance before closing the context",
                                    next_seq(),
                                ),
                                false,
                            )?;
                            continue;
                        }
                        contexts.remove(&context_name);
                        if let Some(ack) = maybe_ack(&next_seq) {
                            emit(events, ack, false)?;
                        }
                        let mut object = SessionEvent::ContextClosed.object(sid, next_seq());
                        object.insert("context".to_owned(), json!(context_name));
                        emit(events, object, false)?;
                    }
                    "cancel" => {
                        match &active {
                            Some(utterance) if utterance.context == context_name => {
                                utterance.cancellation.cancel();
                                // The withheld tail belonged to the utterance being
                                // killed; leaking it into the NEXT say would prepend a
                                // fragment of the interrupted turn to the new reply.
                                if let Some(state) = contexts.get_mut(&context_name) {
                                    state.tail.clear();
                                }
                                if let Some(ack) = maybe_ack(&next_seq) {
                                    emit(events, ack, false)?;
                                }
                                // The receipt follows via Done{Cancelled}.
                            }
                            _ => emit(
                                events,
                                error_event(
                                    "idle",
                                    format!("context {context_name} has no active utterance"),
                                    "cancel targets the currently speaking context",
                                    next_seq(),
                                ),
                                false,
                            )?,
                        }
                    }
                    "flush" => match (&mut active, contexts.get_mut(&context_name)) {
                        (Some(utterance), Some(state)) if utterance.context == context_name => {
                            let tail = std::mem::take(&mut state.tail);
                            if !tail.is_empty()
                                && send_append(loaded, normalization, utterance, &tail).is_err()
                            {
                                // The utterance is ending under us (benign race) or the
                                // tail failed to prepare; either way it belonged to the
                                // dying utterance — drop it, the receipt is the truth.
                            }
                            if utterance
                                .feed
                                .send(TextControl::Finish, &utterance.cancellation)
                                .is_ok()
                            {
                                utterance.text_finished = true;
                            }
                            if let Some(ack) = maybe_ack(&next_seq) {
                                emit(events, ack, false)?;
                            }
                        }
                        _ => emit(
                            events,
                            error_event(
                                "idle",
                                format!("context {context_name} has no open utterance to flush"),
                                "flush ends the text stream of the active utterance",
                                next_seq(),
                            ),
                            false,
                        )?,
                    },
                    "say" => {
                        let text = value
                            .get("text")
                            .and_then(Value::as_str)
                            .unwrap_or_default()
                            .to_owned();
                        let keep_open = value
                            .get("continue")
                            .and_then(Value::as_bool)
                            .unwrap_or(false);
                        let Some(state) = contexts.get_mut(&context_name) else {
                            emit(
                                events,
                                error_event(
                                    "no-context",
                                    format!("context {context_name} is not open"),
                                    "send open before say",
                                    next_seq(),
                                ),
                                false,
                            )?;
                            continue;
                        };
                        match &mut active {
                            Some(utterance) if utterance.context == context_name => {
                                if utterance.text_finished {
                                    emit(
                                        events,
                                        error_event(
                                            "busy",
                                            "the utterance is finishing; wait for speak_complete"
                                                .to_owned(),
                                            "start the next say after the receipt",
                                            next_seq(),
                                        ),
                                        false,
                                    )?;
                                    continue;
                                }
                                // Append path: withheld tail + new text, verbatim concat.
                                let combined = format!("{}{}", state.tail, text);
                                state.tail.clear();
                                let (head, tail) = if keep_open {
                                    split_tail(&combined)
                                } else {
                                    (combined.as_str(), "")
                                };
                                if !head.is_empty()
                                    && let Err(error) =
                                        send_append(loaded, normalization, utterance, head)
                                {
                                    // NEVER fatal: a bad chunk (kind "input") or an
                                    // append racing the utterance's end (the cancel-
                                    // aware send trips, or the engine finished and
                                    // dropped the feed) must not kill the session.
                                    let kind = match &error {
                                        FttsError::Input(_) => "input",
                                        _ => "ending",
                                    };
                                    emit(
                                        events,
                                        error_event(
                                            kind,
                                            error.to_string(),
                                            "the utterance's receipt is authoritative; \
                                             say again after it arrives",
                                            next_seq(),
                                        ),
                                        false,
                                    )?;
                                    continue;
                                }
                                state.tail = tail.to_owned();
                                if !keep_open {
                                    // A failed Finish means the utterance is already
                                    // ending on its own; the receipt is the truth.
                                    if utterance
                                        .feed
                                        .send(TextControl::Finish, &utterance.cancellation)
                                        .is_ok()
                                    {
                                        utterance.text_finished = true;
                                    }
                                }
                                if let Some(ack) = maybe_ack(&next_seq) {
                                    emit(events, ack, false)?;
                                }
                            }
                            Some(other) => {
                                emit(
                                    events,
                                    error_event(
                                        "busy",
                                        format!("context {} is speaking", other.context),
                                        "one utterance at a time: cancel it or wait for its receipt",
                                        next_seq(),
                                    ),
                                    false,
                                )?;
                            }
                            None => {
                                // New utterance. The head starts synthesis; a withheld
                                // tail (continue:true) waits for the next chunk.
                                let combined = format!("{}{}", state.tail, text);
                                state.tail.clear();
                                let (head_owned, tail) = if keep_open {
                                    let (head, tail) = split_tail(&combined);
                                    (head.to_owned(), tail.to_owned())
                                } else {
                                    (combined.clone(), String::new())
                                };
                                if head_owned.trim().is_empty() {
                                    // Nothing speakable yet: hold everything.
                                    state.tail = combined;
                                    if let Some(ack) = maybe_ack(&next_seq) {
                                        emit(events, ack, false)?;
                                    }
                                    continue;
                                }
                                state.tail = tail;
                                // Tokenize FIRST: a refused text must not burn an
                                // utterance index or kill the session (kind "input").
                                let target_ids =
                                    match chunk_target_ids(loaded, &head_owned, normalization) {
                                        Ok((raw, _)) => raw,
                                        Err(error) => {
                                            emit(
                                                events,
                                                error_event(
                                                    "input",
                                                    error.to_string(),
                                                    "fix the text; the context is unharmed",
                                                    next_seq(),
                                                ),
                                                false,
                                            )?;
                                            continue;
                                        }
                                    };
                                let utterance_index = state.utterances_started;
                                state.utterances_started += 1;
                                let effective_seed = value
                                    .get("seed")
                                    .and_then(Value::as_u64)
                                    .unwrap_or_else(|| utterance_seed(state.seed, utterance_index));
                                let (feed_tx, feed_rx) = text_control_queue(TEXT_FEED_CAPACITY)
                                    .map_err(|error| {
                                        FttsError::Generic(format!(
                                            "cannot create the text feed: {error}"
                                        ))
                                    })?;
                                let cancellation = CancellationToken::new();
                                let mut utterance = ActiveUtterance {
                                    context: context_name.clone(),
                                    utterance: utterance_index,
                                    feed: feed_tx,
                                    cancellation: cancellation.clone(),
                                    target_tokens: target_ids.len() as u64,
                                    target_ids,
                                    text_finished: false,
                                };
                                if !keep_open {
                                    // Infallible in practice: the receiver is still in
                                    // our hands (feed_rx below) and the token is fresh,
                                    // so a failure here is a wiring bug worth dying on.
                                    utterance
                                        .feed
                                        .send(TextControl::Finish, &cancellation)
                                        .map_err(|error| {
                                            FttsError::SessionTransport(format!(
                                                "cannot finish the text stream: {error}"
                                            ))
                                        })?;
                                    utterance.text_finished = true;
                                }
                                ledger.utterance_frames.store(0, Ordering::Relaxed);
                                ledger.utterance_ttfa_ms.store(u64::MAX, Ordering::Relaxed);
                                jobs.send(SpeakJob {
                                    context: context_name.clone(),
                                    utterance: utterance_index,
                                    text: head_owned,
                                    normalization: normalization.clone(),
                                    effective_seed,
                                    speaker: state.speaker.clone(),
                                    feed_rx,
                                    cancellation,
                                })
                                .map_err(|_| {
                                    FttsError::SessionTransport(
                                        "the synthesis thread stopped".to_owned(),
                                    )
                                })?;
                                active = Some(utterance);
                                if let Some(ack) = maybe_ack(&next_seq) {
                                    emit(events, ack, false)?;
                                }
                            }
                        }
                    }
                    other => {
                        emit(
                            events,
                            error_event(
                                "invalid-op",
                                format!("unknown op {other}"),
                                "see `ftts robot schema --contract session`",
                                next_seq(),
                            ),
                            false,
                        )?;
                    }
                }
            }
        }
    }

    let end = SessionEvent::SessionEnd.object(sid, next_seq());
    emit(events, end, false)?;
    Ok(())
}

/// Tokenize one chunk exactly as synthesis does, returning `(raw ids, wrapped ids)` —
/// wrapped for the engine, raw for the truncation receipt (the wrapper is scaffolding).
fn chunk_target_ids(
    loaded: &LoadedModel,
    text: &str,
    normalization: &ftts_core::NormalizationOptions,
) -> Result<(Vec<u32>, Vec<u32>), FttsError> {
    let prepared = loaded
        .shared_tokenizer()
        .prepare(text, normalization)
        .map_err(|error| FttsError::Input(format!("text preparation failed: {error}")))?;
    let wrapped =
        ftts_model_qwen::checkpoint::TalkerCheckpoint::wrap_target_ids(&prepared.token_ids);
    Ok((prepared.token_ids, wrapped))
}

/// Send one already-hygienic chunk into the active utterance's feed.
fn send_append(
    loaded: &LoadedModel,
    normalization: &ftts_core::NormalizationOptions,
    utterance: &mut ActiveUtterance,
    chunk: &str,
) -> Result<(), FttsError> {
    let (raw, wrapped) = chunk_target_ids(loaded, chunk, normalization)?;
    let prepared = ftts_core::PreparedText::new(
        wrapped,
        ftts_core::NormalizationTrace {
            mode: ftts_core::NormalizationMode::Verbatim,
            unicode_version: String::new(),
            changes: Vec::new(),
        },
    );
    utterance
        .feed
        .send(TextControl::Append(prepared), &utterance.cancellation)
        .map_err(|error| {
            FttsError::SessionTransport(format!("cannot append to the text stream: {error}"))
        })?;
    // Receipt bookkeeping only AFTER the send: a failed append never reached the
    // model, so counting it would inflate the truncation upper bound.
    utterance.target_tokens += raw.len() as u64;
    utterance.target_ids.extend_from_slice(&raw);
    Ok(())
}

/// Build the terminal receipt for an utterance and clear the active slot.
#[allow(clippy::too_many_arguments)]
fn finish_utterance(
    sid: &str,
    loaded: &LoadedModel,
    state: Option<&ActiveUtterance>,
    context: &str,
    utterance: u64,
    outcome: UtteranceOutcome,
    ledger: &DeliveryLedger,
    next_seq: &dyn Fn() -> u64,
) -> serde_json::Map<String, Value> {
    match outcome {
        UtteranceOutcome::Complete {
            frames,
            ttfa_ms,
            elapsed_ms,
        } => {
            let audio_ms = frames * 80;
            let mut object = SessionEvent::SpeakComplete.object(sid, next_seq());
            object.insert("context".to_owned(), json!(context));
            object.insert("utterance".to_owned(), json!(utterance));
            object.insert("frames".to_owned(), json!(frames));
            object.insert("audio_ms".to_owned(), json!(audio_ms));
            // One TTFA basis per session: the delivery clock the first `audio` event
            // used. The synthesize-internal audible mark is the fallback for the
            // degenerate no-delivery case only.
            let delivered_ttfa = ledger.utterance_ttfa_ms.load(Ordering::Relaxed);
            let ttfa_ms = if delivered_ttfa == u64::MAX {
                ttfa_ms
            } else {
                Some(delivered_ttfa)
            };
            if let Some(ttfa_ms) = ttfa_ms {
                object.insert("ttfa_ms".to_owned(), json!(ttfa_ms));
            }
            let rtf = if elapsed_ms > 0 {
                (audio_ms as f64) / (elapsed_ms as f64)
            } else {
                0.0
            };
            object.insert("rtf".to_owned(), json!((rtf * 100.0).round() / 100.0));
            object
        }
        UtteranceOutcome::Cancelled => {
            let frames_delivered = ledger.utterance_frames.load(Ordering::Relaxed);
            let mut object = SessionEvent::SpeakCancelled.object(sid, next_seq());
            object.insert("context".to_owned(), json!(context));
            object.insert("utterance".to_owned(), json!(utterance));
            object.insert("frames_delivered".to_owned(), json!(frames_delivered));
            object.insert("audio_ms".to_owned(), json!(frames_delivered * 80));
            let (spoken_tokens, spoken_text) = state
                .map(|active| spoken_prefix(loaded, active, frames_delivered))
                .unwrap_or((0, String::new()));
            object.insert("text_spoken_tokens".to_owned(), json!(spoken_tokens));
            object.insert("spoken_text".to_owned(), json!(spoken_text));
            object
        }
        UtteranceOutcome::Failed(message) => {
            let mut object = SessionEvent::SessionError.object(sid, next_seq());
            object.insert("kind".to_owned(), json!("synthesis"));
            // session_error carries no utterance field (frozen shape), so the
            // correlation the orchestrator needs rides in the message text.
            object.insert(
                "message".to_owned(),
                json!(format!(
                    "context {context} utterance {utterance}: {message}"
                )),
            );
            object.insert(
                "remediation".to_owned(),
                json!("the session survives; open a fresh utterance or context"),
            );
            object
        }
    }
}

/// The token-accurate spoken prefix: text is consumed one target token per frame, so
/// frames DELIVERED bound the tokens the listener heard. Decoded defensively — a
/// partial multi-byte sequence at the cut is dropped rather than emitted as mojibake.
fn spoken_prefix(
    loaded: &LoadedModel,
    active: &ActiveUtterance,
    frames_delivered: u64,
) -> (u64, String) {
    // `target_ids` are the RAW target ids (wrapper scaffolding excluded at capture).
    // Conservative rule: cap at delivered frames, never exceed the ids actually fed —
    // and per the documented semantics this is a consumed-text UPPER BOUND on speech.
    let tokens = frames_delivered.min(active.target_tokens);
    let take = usize::try_from(tokens).unwrap_or(usize::MAX);
    let prefix_ids: Vec<u32> = active.target_ids.iter().copied().take(take).collect();
    let text = loaded
        .shared_tokenizer()
        .decode(&prefix_ids)
        .unwrap_or_default()
        .trim_end_matches(char::REPLACEMENT_CHARACTER)
        .to_owned();
    (tokens, text)
}

/// Wait out a cancelled utterance during shutdown and emit its terminal receipt — the
/// orchestrator's transcript truncation depends on it even when the session is ending.
fn settle_utterance(
    sid: &str,
    loaded: &LoadedModel,
    state: &ActiveUtterance,
    inbox: &Receiver<RouterIn>,
    events: &SyncSender<String>,
    ledger: &DeliveryLedger,
    next_seq: &dyn Fn() -> u64,
) -> Result<(), FttsError> {
    while let Ok(message) = inbox.recv() {
        if let RouterIn::Done {
            context,
            utterance,
            outcome,
        } = message
            && context == state.context
            && utterance == state.utterance
        {
            let receipt = finish_utterance(
                sid,
                loaded,
                Some(state),
                &context,
                utterance,
                outcome,
                ledger,
                next_seq,
            );
            return emit(events, receipt, false);
        }
    }
    Ok(())
}
