#![forbid(unsafe_code)]

//! Safe, blocking public engine primitives.
//!
//! `TtsEngine` owns the one async runtime used below the synchronous public
//! facade. Model work is intentionally absent in Phase 0, but the admission,
//! cancellation, budget, observer, and bounded-streaming contracts are real so
//! later model stages cannot introduce a second orchestration path.

pub mod admission;
pub mod audio;
pub mod health;

use std::{
    env, fmt,
    ops::Range,
    sync::{
        Arc, OnceLock,
        atomic::{AtomicBool, Ordering},
        mpsc::{self, Receiver, RecvTimeoutError, SyncSender, TrySendError},
    },
    thread,
    time::{Duration, Instant},
};

use asupersync::runtime::{Runtime, RuntimeBuilder};

/// Identifies this crate's scaffold revision.
pub const SCAFFOLD_REVISION: u8 = 2;

const DEFAULT_QUEUE_CAPACITY: usize = 8;
const DEFAULT_SYNTHESIS_BUDGET: Duration = Duration::from_secs(30);
const DEFAULT_ENROLL_BUDGET: Duration = Duration::from_secs(30);
const BACKPRESSURE_POLL: Duration = Duration::from_millis(1);

/// Wall time allowed per generated frame, on top of the stage's startup allowance.
///
/// A flat whole-stage deadline cannot tell "the model is hung" from "the caller asked for more
/// speech", and it answers both with the same refusal. That is what made a twelve-word utterance
/// fail while a two-word one passed: nothing was wrong, the request was simply longer. Bounding the
/// *rate* instead is length-independent, which is the property a budget actually wants.
///
/// The number is measured, not guessed. On the machine this was calibrated on, release synthesis
/// runs ~1.05 s/frame (20 frames in 20.9 s), so 8 s/frame leaves ~7.6x headroom for a colder or
/// busier host while still catching a genuine stall within one frame. This is deliberately loose:
/// the codebase is pre-optimization (the whole project exists to move this number), so a tight
/// budget here would encode today's slowness as tomorrow's contract.
const DEFAULT_SYNTHESIS_FRAME_BUDGET: Duration = Duration::from_secs(8);

/// How much slower an unoptimized build is, applied to both synthesis budgets.
///
/// Measured on the same machine and the same utterance as the frame budget above: a debug build
/// spent 26.8 s loading where release spent 2.2 s (12x), and had not finished the same 20 frames
/// after 20 minutes where release took 20.9 s — so >57x on the decode loop, or >60 s/frame.
///
/// That measurement is a lower bound, not a clean one: the run shared the machine with concurrent
/// cargo builds, so some of the 57x is contention rather than the profile. 32x is chosen to sit
/// above the honest part of that range with room to spare — at 32x the per-frame allowance is 256 s
/// against >60 s observed, roughly 4x headroom, matching the release tier's intent rather than
/// leaving debug on a knife edge. The cost of being too generous is only that a genuinely hung
/// debug run takes a few minutes to be caught; the cost of being too tight is refusing correct
/// work, which is the failure this whole mechanism exists to stop.
///
/// This exists because a developer running `cargo test` or `cargo run` without `--release` is doing
/// something legitimate, and being told their correct request "exceeded its budget" teaches them
/// the engine is broken when it is only slow.
const DEBUG_BUILD_SLOWDOWN: u32 = 32;

/// Process-wide engine defaults read once from `FTTS_STAGE_BUDGET_*_MS`.
///
/// The initial budget names are `FTTS_STAGE_BUDGET_SYNTHESIS_MS` and
/// `FTTS_STAGE_BUDGET_ENROLL_MS`. Invalid or zero values retain their documented
/// defaults; configuration errors should never silently create an unbounded stage.
pub fn process_engine_config() -> EngineConfig {
    static CONFIG: OnceLock<EngineConfig> = OnceLock::new();
    CONFIG.get_or_init(EngineConfig::from_environment).clone()
}

/// Fixed limits for one engine instance.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EngineConfig {
    /// Capacity for each independent PCM and event queue.
    pub stream_queue_capacity: usize,
    /// Wall time allowed for one synthesis stage *before* the per-frame allowance is added.
    ///
    /// This is the startup grace: prefill, cache warmup, and the first frame. It is not the whole
    /// stage's ceiling — see [`Self::synthesis_frame_budget`], which extends the deadline as frames
    /// are actually produced. A generator that never yields its first frame still trips at exactly
    /// this value, so this remains the knob that catches a hang.
    pub synthesis_stage_budget: Duration,
    /// Wall time added to the synthesis deadline for each frame already generated.
    ///
    /// This is what makes the budget scale with the length of the utterance instead of refusing
    /// long ones. Zero is rejected by [`Self::validate`]: it would silently restore the flat
    /// whole-stage deadline this field exists to replace.
    pub synthesis_frame_budget: Duration,
    /// Maximum wall time for one enrollment CPU stage.
    pub enroll_stage_budget: Duration,
    /// Predicted-peak-memory policy applied to every synthesis request.
    pub admission: admission::AdmissionPolicy,
    /// Maximum time a continuation utterance may stall in [`FrameStep::AwaitingText`]
    /// before the engine finishes its text stream for it.
    ///
    /// A stall is a caller that has not yet supplied more text — a slow LLM, not a wedged
    /// generator — so it is excluded from the synthesis budget. This cap bounds the wait:
    /// on expiry the engine calls `finish_text` and lets the utterance end through the
    /// model's own EOS, so a conversation whose text source died degrades to a completed
    /// sentence, never a hung process. Fixed timeout = the deterministic fallback the
    /// Alien-Artifact contract requires of any adaptive-ish behavior.
    pub text_stall_timeout: Duration,
}

impl Default for EngineConfig {
    fn default() -> Self {
        // The synthesis budgets are scaled by build profile; enrollment is not, because it does no
        // per-frame model work and its 30 s is not close to binding.
        let slowdown = build_profile_slowdown();
        Self {
            stream_queue_capacity: DEFAULT_QUEUE_CAPACITY,
            synthesis_stage_budget: DEFAULT_SYNTHESIS_BUDGET * slowdown,
            synthesis_frame_budget: DEFAULT_SYNTHESIS_FRAME_BUDGET * slowdown,
            enroll_stage_budget: DEFAULT_ENROLL_BUDGET,
            admission: admission::AdmissionPolicy::default(),
            text_stall_timeout: DEFAULT_TEXT_STALL_TIMEOUT,
        }
    }
}

/// Default cap on one continuous [`FrameStep::AwaitingText`] stall (see
/// [`EngineConfig::text_stall_timeout`]). Generous on purpose: a conversation can pause
/// while its LLM thinks, and the cost of waiting is silence, not resources — the frame
/// loop is parked, not spinning.
pub const DEFAULT_TEXT_STALL_TIMEOUT: Duration = Duration::from_secs(60);

/// How often a stalled frame loop wakes to check cancellation and the stall cap while
/// waiting for text. 10 ms keeps cancel latency well inside one frame without busy-spin.
const TEXT_STALL_POLL: Duration = Duration::from_millis(10);

/// The multiplier applied to the synthesis budgets for the current build profile.
///
/// `debug_assertions` is the available proxy for "unoptimized". It is not exact — a release build
/// with `debug-assertions = true` is charged the debug multiplier — but erring toward the larger
/// budget only costs a hung run some extra seconds before it is caught, while erring the other way
/// refuses correct work, which is the failure this whole mechanism exists to stop.
const fn build_profile_slowdown() -> u32 {
    if cfg!(debug_assertions) {
        DEBUG_BUILD_SLOWDOWN
    } else {
        1
    }
}

impl EngineConfig {
    fn from_environment() -> Self {
        let mut config = Self::default();
        config.synthesis_stage_budget = stage_budget_from_environment(
            "FTTS_STAGE_BUDGET_SYNTHESIS_MS",
            config.synthesis_stage_budget,
        );
        config.synthesis_frame_budget = stage_budget_from_environment(
            "FTTS_STAGE_BUDGET_FRAME_MS",
            config.synthesis_frame_budget,
        );
        config.enroll_stage_budget = stage_budget_from_environment(
            "FTTS_STAGE_BUDGET_ENROLL_MS",
            config.enroll_stage_budget,
        );
        // An unparseable or zero value keeps the documented default rather than creating an
        // unbounded budget, matching the stage-budget policy above: a configuration mistake must
        // never silently remove a limit.
        config.admission.budget_bytes = positive_u64_from_environment("FTTS_MEMORY_BUDGET_MB")
            .and_then(|megabytes| megabytes.checked_mul(1024 * 1024))
            .unwrap_or(config.admission.budget_bytes);
        if let Some(max_frames) = positive_u64_from_environment("FTTS_MAX_FRAMES") {
            // An explicit cap is obeyed exactly: it both replaces the policy default and
            // disables the text-derived EOS backstop that otherwise bounds a bare `ftts say`.
            config.admission.max_new_tokens = max_frames;
            config.admission.heuristic_eos_backstop = false;
        }
        config
    }

    fn validate(&self) -> Result<(), EngineError> {
        if self.stream_queue_capacity == 0 {
            return Err(EngineError::InvalidConfiguration(
                "stream queue capacity must be greater than zero",
            ));
        }
        if self.synthesis_stage_budget.is_zero()
            || self.synthesis_frame_budget.is_zero()
            || self.enroll_stage_budget.is_zero()
        {
            return Err(EngineError::InvalidConfiguration(
                "stage budgets must be greater than zero",
            ));
        }
        Ok(())
    }
}

/// Reads a strictly positive `u64` from the environment, or `None` when unset or unusable.
fn positive_u64_from_environment(name: &str) -> Option<u64> {
    env::var(name)
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .filter(|value| *value > 0)
}

fn stage_budget_from_environment(name: &str, fallback: Duration) -> Duration {
    env::var(name)
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .filter(|milliseconds| *milliseconds > 0)
        .map(Duration::from_millis)
        .unwrap_or(fallback)
}

/// A caller-owned cancellation signal for one request.
///
/// Cloning this token is cheap and preserves a single cancellation state. The
/// token is passed into CPU-stage closures; those closures must checkpoint at
/// every talker-frame boundary once model execution is connected.
#[derive(Clone, Debug, Default)]
pub struct CancellationToken {
    cancelled: Arc<AtomicBool>,
}

impl CancellationToken {
    /// Creates an active cancellation token.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Requests cooperative cancellation.
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
    }

    /// Returns whether cancellation has been requested.
    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Acquire)
    }

    /// Returns `Cancelled` when a stage must stop cooperatively.
    pub fn checkpoint(&self) -> Result<(), EngineError> {
        if self.is_cancelled() {
            Err(EngineError::Cancelled)
        } else {
            Ok(())
        }
    }
}

/// The kind of a bounded stream queue.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StreamKind {
    /// PCM packets only.
    Pcm,
    /// Structured lifecycle events only.
    Events,
    /// Incremental text controls feeding a continuation utterance.
    Text,
}

/// A PCM packet emitted by the codec path.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PcmPacket {
    /// Number of 80 ms codec frames represented by this packet.
    pub frame_count: u8,
    /// Interleaved signed 16-bit PCM samples.
    pub samples: Vec<i16>,
}

/// A streaming endpoint pair with independent bounded PCM and event queues.
///
/// The queue separation makes an event consumer stall unable to block PCM
/// delivery (and vice versa). Producers park under backpressure and observe
/// request cancellation while waiting; no unbounded buffering is available.
pub struct StreamQueues {
    /// PCM producer endpoint.
    pub pcm: BoundedSender<PcmPacket>,
    /// PCM consumer endpoint.
    pub pcm_receiver: BoundedReceiver<PcmPacket>,
    /// Event producer endpoint.
    pub events: BoundedSender<SynthesisEvent>,
    /// Event consumer endpoint.
    pub event_receiver: BoundedReceiver<SynthesisEvent>,
}

impl StreamQueues {
    /// Creates distinct bounded queues for PCM and lifecycle events.
    pub fn new(capacity: usize) -> Result<Self, EngineError> {
        if capacity == 0 {
            return Err(EngineError::InvalidConfiguration(
                "stream queue capacity must be greater than zero",
            ));
        }
        let (pcm, pcm_receiver) = bounded_queue(capacity, StreamKind::Pcm);
        let (events, event_receiver) = bounded_queue(capacity, StreamKind::Events);
        Ok(Self {
            pcm,
            pcm_receiver,
            events,
            event_receiver,
        })
    }
}

/// A bounded queue producer that cooperates with cancellation while stalled.
#[derive(Clone)]
pub struct BoundedSender<T> {
    kind: StreamKind,
    sender: SyncSender<T>,
}

impl<T> BoundedSender<T> {
    /// Sends one item, parking while the bounded queue is full.
    pub fn send(&self, mut item: T, cancellation: &CancellationToken) -> Result<(), EngineError> {
        loop {
            cancellation.checkpoint()?;
            match self.sender.try_send(item) {
                Ok(()) => return Ok(()),
                Err(TrySendError::Full(returned)) => {
                    item = returned;
                    thread::sleep(BACKPRESSURE_POLL);
                }
                Err(TrySendError::Disconnected(_)) => {
                    return Err(EngineError::StreamDisconnected(self.kind));
                }
            }
        }
    }
}

/// A bounded queue consumer.
pub struct BoundedReceiver<T> {
    kind: StreamKind,
    receiver: Receiver<T>,
}

impl<T> BoundedReceiver<T> {
    /// Takes one item without blocking; `Ok(None)` when the queue is momentarily empty.
    ///
    /// # Errors
    ///
    /// [`EngineError::StreamDisconnected`] once every sender is gone (after the queue
    /// drains) — for a text-control feed the caller treats that as end-of-text, not a
    /// failure.
    pub fn try_next(&self) -> Result<Option<T>, EngineError> {
        match self.receiver.try_recv() {
            Ok(item) => Ok(Some(item)),
            Err(mpsc::TryRecvError::Empty) => Ok(None),
            Err(mpsc::TryRecvError::Disconnected) => {
                Err(EngineError::StreamDisconnected(self.kind))
            }
        }
    }

    /// Receives one item, timing out when no item arrives in `timeout`.
    pub fn recv_timeout(&self, timeout: Duration) -> Result<T, EngineError> {
        match self.receiver.recv_timeout(timeout) {
            Ok(item) => Ok(item),
            Err(RecvTimeoutError::Timeout) => Err(EngineError::QueueTimeout),
            Err(RecvTimeoutError::Disconnected) => Err(EngineError::StreamDisconnected(self.kind)),
        }
    }
}

fn bounded_queue<T>(capacity: usize, kind: StreamKind) -> (BoundedSender<T>, BoundedReceiver<T>) {
    let (sender, receiver) = mpsc::sync_channel(capacity);
    (
        BoundedSender { kind, sender },
        BoundedReceiver { kind, receiver },
    )
}

/// One incremental text instruction for a continuation utterance.
///
/// Produced by a session/caller thread, consumed by the synthesis thread between frames
/// (the engine applies it to the generator, which owns tokenizer-independent projection).
/// Chunk-boundary hygiene (withholding a trailing partial word so BPE cannot merge across
/// a chunk seam) is the PRODUCER's responsibility — the engine applies chunks verbatim.
#[derive(Clone, Debug)]
pub enum TextControl {
    /// Extend the open utterance with an already-prepared text chunk.
    Append(PreparedText),
    /// No more text will arrive; the terminal EOS becomes reachable.
    Finish,
}

/// Creates the bounded feed a caller uses to stream text into a continuation synthesis.
///
/// Dropping the sender is equivalent to sending [`TextControl::Finish`]: the engine
/// treats disconnect as end-of-text, so a session that crashes mid-conversation degrades
/// to a completed sentence rather than a stalled loop.
///
/// # Errors
///
/// If `capacity` is zero.
pub fn text_control_queue(
    capacity: usize,
) -> Result<(BoundedSender<TextControl>, BoundedReceiver<TextControl>), EngineError> {
    if capacity == 0 {
        return Err(EngineError::InvalidConfiguration(
            "text control queue capacity must be greater than zero",
        ));
    }
    Ok(bounded_queue(capacity, StreamKind::Text))
}

/// One step of the autoregressive frame loop.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FrameStep {
    /// A generated 16-code frame.
    Frame(CodeFrame),
    /// The model emitted codec EOS; the utterance is complete.
    Finished,
    /// A continuation utterance caught up with its known text: generating another frame
    /// would substitute the model's "text is over" pad signal and wind the utterance
    /// down early. Side-effect-free by contract — no RNG draw, no KV growth, no position
    /// advance — so a stalled-then-resumed run is bit-identical to a never-stalled one.
    AwaitingText,
}

/// The caller-visible text-normalization policy.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum NormalizationMode {
    /// Pinned upstream semantics: NFC and nothing else.
    #[default]
    Verbatim,
    /// Reserved for unambiguous policies; currently deliberately no-op beyond NFC.
    Conservative,
    /// Apply explicit language-span pronunciation entries after NFC.
    LocaleAware,
}

/// A byte range in normalized text with a caller-supplied language identifier.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LanguageSpan {
    /// The range, expressed over the NFC-normalized input.
    pub range: Range<usize>,
    /// A caller-supplied BCP-47-like language identifier.
    pub language: String,
}

/// An explicit pronunciation expansion.
///
/// Entries are only applied in a matching language span, or globally when
/// `language` is `"und"`. The engine neither persists nor logs this text.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PronunciationEntry {
    /// Language to which the entry applies.
    pub language: String,
    /// Surface text to recognize.
    pub surface: String,
    /// Caller-supplied spoken replacement.
    pub spoken: String,
}

/// Caller-supplied behavior layered over the pinned verbatim path.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct NormalizationOptions {
    /// Requested policy. The default is the ConformanceExact verbatim route.
    pub mode: NormalizationMode,
    /// Explicit language overrides for locale-aware entries.
    pub language_spans: Vec<LanguageSpan>,
    /// Caller-supplied pronunciation entries for locale-aware handling.
    pub pronunciation_lexicon: Vec<PronunciationEntry>,
}

/// One observable normalization change.
///
/// This detailed form is returned only to the caller that owns the input text.
/// Observer events use [`NormalizationTraceSummary`] instead, so trace sinks do
/// not receive sensitive before/after text.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NormalizationChange {
    /// Stable name of the rule that made the change.
    pub rule: &'static str,
    /// Input before the rule was applied.
    pub before: String,
    /// Output after the rule was applied.
    pub after: String,
}

/// A deterministic record of what the normalizer did and why.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NormalizationTrace {
    /// Policy used for the request.
    pub mode: NormalizationMode,
    /// Unicode data version used by the tokenizer implementation.
    pub unicode_version: String,
    /// Detailed caller-owned changes.
    pub changes: Vec<NormalizationChange>,
}

impl NormalizationTrace {
    /// Produces the privacy-safe observer form of this trace.
    #[must_use]
    pub fn summary(&self) -> NormalizationTraceSummary {
        let mut rules = self
            .changes
            .iter()
            .map(|change| change.rule.to_owned())
            .collect::<Vec<_>>();
        rules.sort_unstable();
        rules.dedup();
        NormalizationTraceSummary {
            mode: self.mode,
            unicode_version: self.unicode_version.clone(),
            rules,
            change_count: self.changes.len(),
        }
    }
}

/// The privacy-safe normalization information allowed on an observer event.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NormalizationTraceSummary {
    /// Policy used for the request.
    pub mode: NormalizationMode,
    /// Unicode data version used by the tokenizer implementation.
    pub unicode_version: String,
    /// Applied rule names, sorted and deduplicated.
    pub rules: Vec<String>,
    /// Number of detailed changes made by those rules.
    pub change_count: usize,
}

/// Token ids and a caller-owned trace returned by a model-specific text preparer.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PreparedText {
    /// Token ids the model will consume.
    pub token_ids: Vec<u32>,
    /// Detailed normalization record, retained only in request-local memory.
    pub normalization_trace: NormalizationTrace,
}

impl PreparedText {
    /// Constructs a prepared text payload from model-specific tokenization.
    #[must_use]
    pub fn new(token_ids: Vec<u32>, normalization_trace: NormalizationTrace) -> Self {
        Self {
            token_ids,
            normalization_trace,
        }
    }
}

/// Named failure from a model-specific text preparer.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TextPreparationError {
    message: String,
}

impl TextPreparationError {
    /// Constructs a named preparation failure without exposing model error types to the engine.
    #[must_use]
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for TextPreparationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for TextPreparationError {}

/// Model-specific text preparation used by the blocking engine facade.
///
/// `ftts-core` owns this boundary so it never depends on a particular model
/// crate. Model crates implement it with their tokenizer and retain ownership
/// of the detailed text trace.
pub trait TextPreparer: Send + Sync {
    /// Normalizes and tokenizes one request according to its explicit options.
    fn prepare(
        &self,
        text: &str,
        options: &NormalizationOptions,
    ) -> Result<PreparedText, TextPreparationError>;
}

/// One generated codec frame: the talker's primary code plus the 15 residual codes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CodeFrame {
    /// Group 0 is the primary code; groups 1..16 are the microdecoder residuals, in depth order.
    pub codes: Vec<u32>,
}

/// A model-side failure while generating codec frames.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GenerationError {
    message: String,
}

impl GenerationError {
    /// Wraps a model-specific failure description.
    #[must_use]
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for GenerationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for GenerationError {}

/// How one utterance's text stream relates to future [`FrameGenerator::append_text`] calls.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum UtteranceStart {
    /// The whole text is known now; the terminal EOS marker rides in the initial prompt.
    #[default]
    Fresh,
    /// The text stream stays open: codec EOS becomes reachable only after
    /// [`FrameGenerator::finish_text`], and [`FrameGenerator::append_text`] may extend the
    /// utterance while frames are being generated.
    Continuation,
}

/// Model-specific autoregressive frame generation behind the blocking engine facade.
///
/// Like [`TextPreparer`], `ftts-core` owns only the boundary: the model crate implements the
/// prompt assembly, talker forward, and 15-step microdecoder behind these calls, and the
/// engine owns admission, budgets, cancellation, and observer events around them.
pub trait FrameGenerator {
    /// Prepares per-utterance state (prompt assembly and talker prefill) for one request.
    ///
    /// `Continuation` requires streaming x-vector assembly; ICL and non-streaming prompts
    /// reject it explicitly rather than approximating it.
    fn begin_utterance(
        &mut self,
        prepared: &PreparedText,
        mode: UtteranceStart,
    ) -> Result<(), GenerationError>;

    /// Extends the open utterance's trailing text with an already-prepared chunk.
    ///
    /// Tokenization and normalization belong to the caller (chunk-boundary hygiene included):
    /// this receives `PreparedText`, runs the cold-table gather and text projection between
    /// frames on the synthesis thread, and changes nothing about position numbering — appended
    /// rows are consumed at future frame indices exactly as if they had always been there.
    fn append_text(&mut self, prepared: &PreparedText) -> Result<(), GenerationError>;

    /// Marks the open continuation's text stream complete, making the terminal EOS reachable.
    fn finish_text(&mut self) -> Result<(), GenerationError>;

    /// Advances the frame loop by one step.
    ///
    /// [`FrameStep::AwaitingText`] is legal only for continuation utterances whose text
    /// is exhausted and unfinished, and MUST be side-effect-free: returning it and being
    /// called again after an append must produce exactly the frames a never-stalled run
    /// would. The engine owns all waiting — implementations never block here.
    fn next_frame(&mut self) -> Result<FrameStep, GenerationError>;
}

/// A synchronous synthesis request.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SynthesisRequest {
    /// Text to synthesize. The Phase 0 shell accepts an empty request.
    pub text: String,
    /// Caller-owned policy passed unchanged to the model-specific tokenizer.
    pub normalization_options: NormalizationOptions,
    /// Whether the observer may receive a privacy-safe normalization summary.
    pub trace_normalization: bool,
}

impl SynthesisRequest {
    /// Creates a request from caller-owned text.
    #[must_use]
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            normalization_options: NormalizationOptions::default(),
            trace_normalization: false,
        }
    }

    /// Replaces the default verbatim normalization policy for this request.
    #[must_use]
    pub fn with_normalization_options(
        mut self,
        normalization_options: NormalizationOptions,
    ) -> Self {
        self.normalization_options = normalization_options;
        self
    }

    /// Allows the caller-owned observer to receive a text-free trace summary.
    #[must_use]
    pub const fn with_normalization_trace(mut self, trace_normalization: bool) -> Self {
        self.trace_normalization = trace_normalization;
        self
    }
}

/// A synchronous enrollment request.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EnrollmentRequest {
    /// Caller-provided reference bytes. Decoding is connected in the model stage.
    pub reference_audio: Vec<u8>,
}

/// A completed synthesis result.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SynthesisResult {
    /// Number of generated codec frames.
    pub generated_frames: u64,
    /// Every generated 16-code frame, in emission order, for the codec stage.
    pub code_frames: Vec<CodeFrame>,
    /// Number of token ids produced by the request-local text preparer.
    pub prepared_token_count: usize,
}

/// A completed empty-pipeline enrollment result.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EnrollmentResult {
    /// The engine shell has not yet created a voice pack.
    pub accepted_reference_bytes: usize,
}

/// A stage named in observer events and budget errors.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EngineStage {
    /// The complete synthesis pipeline.
    Synthesis,
    /// The complete enrollment pipeline.
    Enrollment,
}

/// A health signal emitted through the caller-owned observer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HealthEvent {
    /// A request exceeded its configured stage budget.
    BudgetExceeded,
    /// A request observed cooperative cancellation.
    Cancelled,
    /// A runtime-health detector fired ([`health`]).
    ///
    /// Carried through the same observer as every other lifecycle event so a caller learns about
    /// a NaN, a stall, a repetition loop or a silent output *while the run is happening*, rather
    /// than inferring it afterwards from audio it cannot listen to. The violation itself says
    /// whether the output is still usable — see [`health::HealthViolation::invalidates_output`].
    Violation(health::HealthViolation),
}

impl HealthEvent {
    /// Whether this signal means the run's output must not be presented as a clean result.
    #[must_use]
    pub const fn invalidates_output(self) -> bool {
        match self {
            Self::BudgetExceeded | Self::Cancelled => true,
            Self::Violation(violation) => violation.invalidates_output(),
        }
    }

    /// Stable wire string for robot mode.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::BudgetExceeded => "budget_exceeded",
            Self::Cancelled => "cancelled",
            Self::Violation(violation) => violation.as_str(),
        }
    }
}

/// Lifecycle information delivered to a caller-owned observer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SynthesisEvent {
    /// Concurrency admission outcome before model work begins.
    Admission { accepted: bool },
    /// Resource admission outcome: the predicted peak memory for this utterance.
    ///
    /// Distinct from [`SynthesisEvent::Admission`], which is the one-live-synthesis lease. Emitted
    /// for accepted and rejected requests alike, so a capacity problem is visible in the event
    /// stream rather than only in an error string.
    ResourceAdmission {
        /// Whether the request was admitted.
        admitted: bool,
        /// Frames the request may generate.
        predicted_max_frames: u64,
        /// Predicted peak bytes for the utterance.
        predicted_peak_bytes: u64,
        /// The budget it was measured against.
        budget_bytes: u64,
    },
    /// A CPU stage started.
    StageStarted { stage: EngineStage },
    /// A CPU stage completed within its budget.
    StageFinished {
        stage: EngineStage,
        elapsed: Duration,
    },
    /// A talker-frame boundary was reached.
    FrameProgress { frame: u64 },
    /// A continuation stall ended: text arrived after `waited` of [`FrameStep::AwaitingText`].
    ///
    /// Sessions surface this as their `text_underrun` wire event. Generation at >1x
    /// real time normally hides short stalls behind the delivered-audio lead; this event
    /// is how a caller learns the lead ran dry.
    TextUnderrun { waited: Duration },
    /// A continuation stall hit [`EngineConfig::text_stall_timeout`] (or its feed
    /// disconnected / was never supplied): the engine finished the text stream itself so
    /// the utterance ends through the model's own EOS instead of hanging.
    TextStallEnded { waited: Duration },
    /// An append was refused by rolling admission (context ceiling); the in-flight
    /// utterance continues unchanged with the text it already has.
    TextAppendRejected { requested_tokens: u64 },
    /// A caller explicitly requested a privacy-safe normalization trace summary.
    TextPrepared {
        /// Number of token ids that entered the model path.
        token_count: usize,
        /// No raw or rewritten text is included in this observer payload.
        normalization: NormalizationTraceSummary,
    },
    /// A packet entered the PCM stream.
    PacketEmitted {
        frame_count: u8,
        sample_count: usize,
    },
    /// A health event for the current request.
    Health { event: HealthEvent },
}

/// Caller-owned telemetry for synthesis and enrollment.
///
/// CLI trace mode, robot NDJSON, and benchmarking all consume this same hook;
/// the engine keeps neither global telemetry nor persisted synthesis state.
pub trait SynthesisObserver: Send + Sync {
    /// Receives one lifecycle event synchronously on the calling thread.
    fn on_event(&self, event: SynthesisEvent);
}

impl<F> SynthesisObserver for F
where
    F: Fn(SynthesisEvent) + Send + Sync,
{
    fn on_event(&self, event: SynthesisEvent) {
        self(event);
    }
}

/// Errors produced by the synchronous engine facade.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EngineError {
    /// The one-live-synthesis admission limit rejected a concurrent request.
    Busy,
    /// The caller or an expired stage budget requested cancellation.
    Cancelled,
    /// A CPU stage did not complete within its configured budget.
    BudgetExceeded(EngineStage),
    /// A stream consumer disappeared.
    StreamDisconnected(StreamKind),
    /// A queue receive timed out.
    QueueTimeout,
    /// The model-specific text preparer rejected the request.
    TextPreparation(TextPreparationError),
    /// The model-specific frame generator failed mid-utterance.
    Generation(GenerationError),
    /// Predicted peak memory for this utterance exceeded the budget.
    ///
    /// Raised **before** any KV or codec state is allocated, so a rejected request has committed
    /// nothing and the caller can retry with shorter text or a different cap.
    ResourceAdmission(admission::AdmissionRejection),
    /// Engine construction received an invalid setting.
    InvalidConfiguration(&'static str),
    /// The owned runtime could not be constructed.
    Runtime(String),
}

impl fmt::Display for EngineError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Busy => formatter.write_str("another synthesis is already active"),
            Self::Cancelled => formatter.write_str("synthesis cancelled"),
            Self::BudgetExceeded(stage) => write!(formatter, "{stage:?} stage budget exceeded"),
            Self::StreamDisconnected(kind) => write!(formatter, "{kind:?} stream disconnected"),
            Self::QueueTimeout => formatter.write_str("bounded queue receive timed out"),
            Self::TextPreparation(error) => write!(formatter, "text preparation failed: {error}"),
            Self::Generation(error) => write!(formatter, "frame generation failed: {error}"),
            Self::ResourceAdmission(rejection) => {
                write!(
                    formatter,
                    "resource admission refused the request: {rejection}"
                )
            }
            Self::InvalidConfiguration(message) => formatter.write_str(message),
            Self::Runtime(message) => write!(formatter, "runtime initialization failed: {message}"),
        }
    }
}

impl std::error::Error for EngineError {}

/// Blocking public engine facade.
///
/// The runtime is owned below this facade and is never exposed to callers.
/// Admission uses an atomic lease, so no mutex is held across CPU work and one
/// engine never runs more than one synthesis fanout at a time.
pub struct TtsEngine {
    runtime: Runtime,
    config: EngineConfig,
    synthesis_active: AtomicBool,
}

impl TtsEngine {
    /// Creates an engine with explicit, validated limits.
    pub fn new(config: EngineConfig) -> Result<Self, EngineError> {
        config.validate()?;
        let runtime = RuntimeBuilder::current_thread()
            .blocking_threads(1, 1)
            .build()
            .map_err(|error| EngineError::Runtime(error.to_string()))?;
        Ok(Self {
            runtime,
            config,
            synthesis_active: AtomicBool::new(false),
        })
    }

    /// Creates an engine from the process-wide environment defaults.
    pub fn from_process_environment() -> Result<Self, EngineError> {
        Self::new(process_engine_config())
    }

    /// Runs one blocking synthesis: text preparation, admission, then the model decode loop.
    ///
    /// The decode loop runs on the calling thread rather than through [`Self::run_stage`]: frame
    /// generators borrow model weights, so they cannot cross the `'static` spawn boundary, and a
    /// per-frame deadline check is the natural budget seam for an autoregressive loop anyway.
    pub fn synthesize<P: TextPreparer + ?Sized>(
        &self,
        request: SynthesisRequest,
        text_preparer: &P,
        frame_generator: &mut dyn FrameGenerator,
        cancellation: &CancellationToken,
        observer: &dyn SynthesisObserver,
        text_feed: Option<&BoundedReceiver<TextControl>>,
    ) -> Result<SynthesisResult, EngineError> {
        let _admission = self.acquire_synthesis_admission(observer)?;
        cancellation.checkpoint().inspect_err(|_| {
            observer.on_event(SynthesisEvent::Health {
                event: HealthEvent::Cancelled,
            });
        })?;
        let prepared = text_preparer
            .prepare(&request.text, &request.normalization_options)
            .map_err(EngineError::TextPreparation)?;
        if request.trace_normalization {
            observer.on_event(SynthesisEvent::TextPrepared {
                token_count: prepared.token_ids.len(),
                normalization: prepared.normalization_trace.summary(),
            });
        }

        // Resource admission sits exactly here, and the position is the point: after tokenization
        // (the prompt length is not knowable before it) and before any stage runs. A request that
        // cannot fit is refused having allocated nothing — never discovered halfway through a long
        // generation. See `admission` for the OQ-6 rule.
        let prompt_tokens = prepared.token_ids.len() as u64;
        let plan = match self.config.admission.admit(prompt_tokens) {
            Ok(plan) => {
                observer.on_event(SynthesisEvent::ResourceAdmission {
                    admitted: true,
                    predicted_max_frames: plan.predicted_max_frames,
                    predicted_peak_bytes: plan.predicted_peak_bytes,
                    budget_bytes: plan.budget_bytes,
                });
                plan
            }
            Err(rejection) => {
                if let admission::AdmissionRejection::BudgetExceeded { plan } = rejection {
                    observer.on_event(SynthesisEvent::ResourceAdmission {
                        admitted: false,
                        predicted_max_frames: plan.predicted_max_frames,
                        predicted_peak_bytes: plan.predicted_peak_bytes,
                        budget_bytes: plan.budget_bytes,
                    });
                }
                return Err(EngineError::ResourceAdmission(rejection));
            }
        };

        observer.on_event(SynthesisEvent::StageStarted {
            stage: EngineStage::Synthesis,
        });
        let started = Instant::now();
        // The deadline rolls forward as frames are produced: startup grace, plus one frame budget
        // for every frame already in hand. A stalled generator makes no progress, so its deadline
        // stops moving and it is caught within one frame budget of wherever it stopped — while a
        // caller who simply asked for more speech is granted proportionally more time instead of
        // being refused for it. Total work stays bounded by `predicted_max_frames` regardless, so
        // dropping the flat whole-stage ceiling gives up no safety.
        let startup_budget = self.config.synthesis_stage_budget;
        let frame_budget = self.config.synthesis_frame_budget;
        let mut code_frames: Vec<CodeFrame> = Vec::new();
        // A text feed makes this a continuation: the terminal EOS is held back until the
        // feed finishes (explicitly, by disconnect, or by the stall cap).
        let mode = if text_feed.is_some() {
            UtteranceStart::Continuation
        } else {
            UtteranceStart::Fresh
        };
        frame_generator
            .begin_utterance(&prepared, mode)
            .map_err(EngineError::Generation)?;
        let mut plan = plan;
        let mut total_prompt_tokens = prompt_tokens;
        // Stall time is a waiting caller, not a wedged generator: it is excluded from the
        // rolling deadline so a slow LLM cannot trip the wedge detector, while a generator
        // that stops moving OUTSIDE a stall is still caught within one frame budget.
        let mut stalled_total = Duration::ZERO;
        // Set once the text stream is over — by Finish, by feed disconnect, or by the
        // stall cap — so end-of-text is applied exactly once.
        let mut text_done = false;
        let mut feed_closed = text_feed.is_none();
        while (code_frames.len() as u64) < plan.predicted_max_frames {
            cancellation.checkpoint().inspect_err(|_| {
                observer.on_event(SynthesisEvent::Health {
                    event: HealthEvent::Cancelled,
                });
            })?;
            // Apply any text controls that arrived while frames were being generated, so
            // appends land at the earliest frame boundary rather than only during stalls.
            if let Some(feed) = text_feed
                && !feed_closed
                && !text_done
            {
                loop {
                    match feed.try_next() {
                        Ok(Some(control)) => {
                            if self.apply_text_control(
                                control,
                                frame_generator,
                                &mut plan,
                                &mut total_prompt_tokens,
                                &mut text_done,
                                observer,
                            )? {
                                break;
                            }
                        }
                        Ok(None) => break,
                        Err(EngineError::StreamDisconnected(_)) => {
                            // Every sender is gone: nothing more can ever arrive. Treat
                            // it as Finish so the utterance completes instead of stalling
                            // to the cap.
                            self.finish_text_once(
                                frame_generator,
                                &mut text_done,
                                Duration::ZERO,
                                observer,
                            )?;
                            feed_closed = true;
                            break;
                        }
                        Err(other) => return Err(other),
                    }
                }
            }
            // Saturating, because a caller-supplied frame budget times a large frame count can
            // overflow `Duration`; an unreachable deadline is the right answer there, not a panic.
            let deadline = frame_budget
                .checked_mul(u32::try_from(code_frames.len()).unwrap_or(u32::MAX))
                .and_then(|earned| earned.checked_add(startup_budget))
                .unwrap_or(Duration::MAX);
            if started.elapsed().saturating_sub(stalled_total) > deadline {
                observer.on_event(SynthesisEvent::Health {
                    event: HealthEvent::BudgetExceeded,
                });
                return Err(EngineError::BudgetExceeded(EngineStage::Synthesis));
            }
            match frame_generator
                .next_frame()
                .map_err(EngineError::Generation)?
            {
                FrameStep::Frame(frame) => {
                    observer.on_event(SynthesisEvent::FrameProgress {
                        frame: code_frames.len() as u64,
                    });
                    code_frames.push(frame);
                }
                FrameStep::Finished => break,
                FrameStep::AwaitingText => {
                    let stall_started = Instant::now();
                    let resumed = 'stall: {
                        if feed_closed || text_done {
                            // Nothing can arrive (no feed, or it already ended) yet the
                            // generator still awaits text — end the stream for it rather
                            // than spin. `text_done` here means a logic mismatch a test
                            // would catch; `feed_closed` is the documented no-feed case.
                            break 'stall false;
                        }
                        let feed = text_feed.expect("feed_closed is true when text_feed is None");
                        loop {
                            cancellation.checkpoint().inspect_err(|_| {
                                observer.on_event(SynthesisEvent::Health {
                                    event: HealthEvent::Cancelled,
                                });
                            })?;
                            if stall_started.elapsed() >= self.config.text_stall_timeout {
                                break 'stall false;
                            }
                            match feed.recv_timeout(TEXT_STALL_POLL) {
                                Ok(control) => {
                                    let finished = self.apply_text_control(
                                        control,
                                        frame_generator,
                                        &mut plan,
                                        &mut total_prompt_tokens,
                                        &mut text_done,
                                        observer,
                                    )?;
                                    if !finished {
                                        observer.on_event(SynthesisEvent::TextUnderrun {
                                            waited: stall_started.elapsed(),
                                        });
                                    }
                                    break 'stall true;
                                }
                                Err(EngineError::QueueTimeout) => {}
                                Err(EngineError::StreamDisconnected(_)) => {
                                    feed_closed = true;
                                    break 'stall false;
                                }
                                Err(other) => return Err(other),
                            }
                        }
                    };
                    if !resumed {
                        self.finish_text_once(
                            frame_generator,
                            &mut text_done,
                            stall_started.elapsed(),
                            observer,
                        )?;
                    }
                    stalled_total += stall_started.elapsed();
                }
            }
        }
        observer.on_event(SynthesisEvent::StageFinished {
            stage: EngineStage::Synthesis,
            elapsed: started.elapsed(),
        });
        Ok(SynthesisResult {
            generated_frames: code_frames.len() as u64,
            code_frames,
            prepared_token_count: usize::try_from(total_prompt_tokens)
                .unwrap_or(prepared.token_ids.len()),
        })
    }

    /// Applies one [`TextControl`] to the generator on the synthesis thread.
    ///
    /// Returns `true` when the control ended the text stream. An [`TextControl::Append`]
    /// passes ROLLING ADMISSION first — the same policy that admitted the request,
    /// re-evaluated at the grown token count. A refused append leaves the in-flight
    /// utterance untouched (the caller hears about it via
    /// [`SynthesisEvent::TextAppendRejected`] and the utterance speaks the text it has);
    /// an admitted one updates the frame cap so a growing utterance earns proportionally
    /// more frames, with the context ceiling as the hard bound protecting the KV budget.
    fn apply_text_control(
        &self,
        control: TextControl,
        frame_generator: &mut dyn FrameGenerator,
        plan: &mut admission::AdmissionPlan,
        total_prompt_tokens: &mut u64,
        text_done: &mut bool,
        observer: &dyn SynthesisObserver,
    ) -> Result<bool, EngineError> {
        match control {
            TextControl::Append(prepared) => {
                let added = prepared.token_ids.len() as u64;
                let requested = total_prompt_tokens.saturating_add(added);
                match self.config.admission.admit(requested) {
                    Ok(new_plan) => {
                        frame_generator
                            .append_text(&prepared)
                            .map_err(EngineError::Generation)?;
                        observer.on_event(SynthesisEvent::ResourceAdmission {
                            admitted: true,
                            predicted_max_frames: new_plan.predicted_max_frames,
                            predicted_peak_bytes: new_plan.predicted_peak_bytes,
                            budget_bytes: new_plan.budget_bytes,
                        });
                        *plan = new_plan;
                        *total_prompt_tokens = requested;
                        Ok(false)
                    }
                    Err(_) => {
                        observer.on_event(SynthesisEvent::TextAppendRejected {
                            requested_tokens: requested,
                        });
                        Ok(false)
                    }
                }
            }
            TextControl::Finish => {
                if !*text_done {
                    frame_generator
                        .finish_text()
                        .map_err(EngineError::Generation)?;
                    *text_done = true;
                }
                Ok(true)
            }
        }
    }

    /// Ends the text stream on the generator's behalf, exactly once, and says so.
    fn finish_text_once(
        &self,
        frame_generator: &mut dyn FrameGenerator,
        text_done: &mut bool,
        waited: Duration,
        observer: &dyn SynthesisObserver,
    ) -> Result<(), EngineError> {
        if !*text_done {
            frame_generator
                .finish_text()
                .map_err(EngineError::Generation)?;
            *text_done = true;
            observer.on_event(SynthesisEvent::TextStallEnded { waited });
        }
        Ok(())
    }

    /// Runs the Phase 0 enrollment shell through the owned runtime.
    pub fn enroll(
        &self,
        request: EnrollmentRequest,
        cancellation: &CancellationToken,
        observer: &dyn SynthesisObserver,
    ) -> Result<EnrollmentResult, EngineError> {
        observer.on_event(SynthesisEvent::Admission { accepted: true });
        self.run_stage(
            EngineStage::Enrollment,
            self.config.enroll_stage_budget,
            cancellation,
            observer,
            |_| Ok(()),
        )?;
        Ok(EnrollmentResult {
            accepted_reference_bytes: request.reference_audio.len(),
        })
    }

    fn acquire_synthesis_admission(
        &self,
        observer: &dyn SynthesisObserver,
    ) -> Result<SynthesisAdmission<'_>, EngineError> {
        match self.synthesis_active.compare_exchange(
            false,
            true,
            Ordering::AcqRel,
            Ordering::Acquire,
        ) {
            Ok(_) => {
                observer.on_event(SynthesisEvent::Admission { accepted: true });
                Ok(SynthesisAdmission { engine: self })
            }
            Err(_) => {
                observer.on_event(SynthesisEvent::Admission { accepted: false });
                Err(EngineError::Busy)
            }
        }
    }

    fn run_stage<R, F>(
        &self,
        stage: EngineStage,
        budget: Duration,
        cancellation: &CancellationToken,
        observer: &dyn SynthesisObserver,
        work: F,
    ) -> Result<R, EngineError>
    where
        R: Send + 'static,
        F: FnOnce(CancellationToken) -> Result<R, EngineError> + Send + 'static,
    {
        cancellation.checkpoint().inspect_err(|_| {
            observer.on_event(SynthesisEvent::Health {
                event: HealthEvent::Cancelled,
            });
        })?;
        observer.on_event(SynthesisEvent::StageStarted { stage });
        let started = Instant::now();
        let (sender, receiver) = mpsc::sync_channel(1);
        let stage_cancellation = cancellation.clone();
        let task_cancellation = cancellation.clone();
        let task = self
            .runtime
            .spawn_blocking(move || {
                let result = task_cancellation
                    .checkpoint()
                    .and_then(|()| work(task_cancellation));
                let _ignored_if_timed_out = sender.send(result);
            })
            .ok_or_else(|| EngineError::Runtime("blocking pool was not configured".to_owned()))?;

        match receiver.recv_timeout(budget) {
            Ok(result) => {
                let result = result?;
                observer.on_event(SynthesisEvent::StageFinished {
                    stage,
                    elapsed: started.elapsed(),
                });
                Ok(result)
            }
            Err(RecvTimeoutError::Timeout) => {
                stage_cancellation.cancel();
                task.cancel();
                observer.on_event(SynthesisEvent::Health {
                    event: HealthEvent::BudgetExceeded,
                });
                Err(EngineError::BudgetExceeded(stage))
            }
            Err(RecvTimeoutError::Disconnected) => Err(EngineError::Runtime(
                "blocking stage disconnected before producing a result".to_owned(),
            )),
        }
    }
}

struct SynthesisAdmission<'a> {
    engine: &'a TtsEngine,
}

impl Drop for SynthesisAdmission<'_> {
    fn drop(&mut self) {
        self.engine.synthesis_active.store(false, Ordering::Release);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    #[derive(Default)]
    struct RecordingObserver {
        events: Mutex<Vec<SynthesisEvent>>,
    }

    impl RecordingObserver {
        fn events(&self) -> Vec<SynthesisEvent> {
            self.events
                .lock()
                .expect("test observer lock poisoned")
                .clone()
        }
    }

    impl SynthesisObserver for RecordingObserver {
        fn on_event(&self, event: SynthesisEvent) {
            self.events
                .lock()
                .expect("test observer lock poisoned")
                .push(event);
        }
    }

    fn engine_with_budget(budget: Duration) -> TtsEngine {
        TtsEngine::new(EngineConfig {
            synthesis_stage_budget: budget,
            ..EngineConfig::default()
        })
        .expect("test engine builds")
    }

    /// An engine with both synthesis budgets pinned, for exercising the rolling deadline.
    fn engine_with_frame_budget(startup: Duration, per_frame: Duration) -> TtsEngine {
        TtsEngine::new(EngineConfig {
            synthesis_stage_budget: startup,
            synthesis_frame_budget: per_frame,
            ..EngineConfig::default()
        })
        .expect("test engine builds")
    }

    /// Emits `remaining` frames, each costing `per_frame`, then either stops at EOS or hangs.
    ///
    /// Real slowness and a real hang differ only in whether progress continues, which is exactly
    /// what the rolling deadline keys on — so both have to be expressible by one generator.
    /// `stall: None` ends the utterance cleanly; `Some(d)` wedges it, so the deadline is what ends
    /// it. Getting this wrong is easy and silent: a generator that hangs instead of stopping makes
    /// the "slow but legal" case fail as a budget refusal and look like the bug it was testing for.
    struct PacedFrameGenerator {
        remaining: usize,
        per_frame: Duration,
        stall: Option<Duration>,
        began: bool,
    }

    impl FrameGenerator for PacedFrameGenerator {
        fn begin_utterance(
            &mut self,
            _prepared: &PreparedText,
            _mode: UtteranceStart,
        ) -> Result<(), GenerationError> {
            self.began = true;
            Ok(())
        }

        fn append_text(&mut self, _prepared: &PreparedText) -> Result<(), GenerationError> {
            Err(GenerationError::new(
                "the pacing fake does not model text appends",
            ))
        }

        fn finish_text(&mut self) -> Result<(), GenerationError> {
            Err(GenerationError::new(
                "the pacing fake does not model text appends",
            ))
        }

        fn next_frame(&mut self) -> Result<FrameStep, GenerationError> {
            assert!(self.began, "next_frame before begin_utterance");
            if self.remaining == 0 {
                let Some(stall) = self.stall else {
                    return Ok(FrameStep::Finished);
                };
                thread::sleep(stall);
                return Ok(FrameStep::Frame(CodeFrame { codes: vec![0; 16] }));
            }
            self.remaining -= 1;
            thread::sleep(self.per_frame);
            Ok(FrameStep::Frame(CodeFrame { codes: vec![0; 16] }))
        }
    }

    /// The humane case: steady progress that would blow a flat whole-stage deadline still finishes.
    ///
    /// This is the regression that motivated the rolling budget — a twelve-word utterance was
    /// refused for being long while a two-word one passed, with nothing actually wrong. Ten frames
    /// at 20 ms each need ~200 ms, far past the 50 ms startup grace; only the per-frame term makes
    /// the run legal, so a reversion to a flat ceiling fails here.
    #[test]
    fn steady_progress_past_the_startup_grace_is_not_refused_for_being_long() {
        let engine = engine_with_frame_budget(Duration::from_millis(50), Duration::from_millis(30));
        let observer = RecordingObserver::default();
        let mut generator = PacedFrameGenerator {
            remaining: 10,
            per_frame: Duration::from_millis(20),
            stall: None,
            began: false,
        };

        let result = engine
            .synthesize(
                SynthesisRequest::new(""),
                &TestTextPreparer,
                &mut generator,
                &CancellationToken::new(),
                &observer,
                None,
            )
            .expect("a steadily-progressing run must not be refused");

        assert_eq!(
            result.generated_frames, 10,
            "all ten frames must survive; a flat 50 ms ceiling would have cut this at ~2"
        );
    }

    /// The other half: a generator that stops progressing is still caught, and caught promptly.
    ///
    /// Without this, "scale the budget with the work" could be satisfied by removing the budget.
    #[test]
    fn a_generator_that_stops_progressing_is_still_caught_within_its_earned_deadline() {
        let engine = engine_with_frame_budget(Duration::from_millis(50), Duration::from_millis(30));
        let observer = RecordingObserver::default();
        let mut generator = PacedFrameGenerator {
            remaining: 3,
            per_frame: Duration::from_millis(1),
            stall: Some(Duration::from_millis(400)),
            began: false,
        };

        let started = Instant::now();
        let error = engine
            .synthesize(
                SynthesisRequest::new(""),
                &TestTextPreparer,
                &mut generator,
                &CancellationToken::new(),
                &observer,
                None,
            )
            .expect_err("a stalled generator must still be refused");
        let elapsed = started.elapsed();

        assert_eq!(error, EngineError::BudgetExceeded(EngineStage::Synthesis));
        assert!(
            observer.events().contains(&SynthesisEvent::Health {
                event: HealthEvent::BudgetExceeded,
            }),
            "the stall must be reported on the health channel, not only as a return value"
        );
        // Three frames earn 50 + 3*30 = 140 ms. One 400 ms stall crosses it; the run must end on
        // that stall rather than accumulating further deadline it never earned.
        assert!(
            elapsed < Duration::from_millis(2000),
            "stall detection took {elapsed:?}; the deadline is not supposed to keep growing while \
             no frames are produced"
        );
    }

    /// A zero per-frame budget silently restores the flat deadline, so it is a configuration error.
    #[test]
    fn a_zero_frame_budget_is_rejected_rather_than_collapsing_to_a_flat_deadline() {
        let built = TtsEngine::new(EngineConfig {
            synthesis_frame_budget: Duration::ZERO,
            ..EngineConfig::default()
        });
        assert!(
            matches!(built, Err(EngineError::InvalidConfiguration(_))),
            "a zero per-frame budget must be refused, not accepted as a flat deadline"
        );
    }

    /// The unoptimized build gets a larger allowance, because it is slower for reasons that are
    /// not the caller's fault. Asserting the relationship rather than the constant keeps this
    /// honest if the measured multiplier is ever re-calibrated.
    #[test]
    fn an_unoptimized_build_is_granted_a_larger_synthesis_budget() {
        let config = EngineConfig::default();
        let expected = u32::from(cfg!(debug_assertions)) * (DEBUG_BUILD_SLOWDOWN - 1) + 1;
        assert_eq!(
            config.synthesis_frame_budget,
            DEFAULT_SYNTHESIS_FRAME_BUDGET * expected
        );
        assert_eq!(
            config.synthesis_stage_budget,
            DEFAULT_SYNTHESIS_BUDGET * expected
        );
        // Enrollment does no per-frame model work, so it is deliberately not scaled.
        assert_eq!(config.enroll_stage_budget, DEFAULT_ENROLL_BUDGET);
    }

    /// Emits a fixed number of all-zero frames, then EOS. Panics if the loop skips prefill.
    ///
    /// `polls` counts every `next_frame` call, which is what separates "the generator stopped" from
    /// "the loop stopped asking": a ceiling-bound run never polls for the frame past the ceiling,
    /// while an EOS-bound run must poll exactly once more than it received.
    struct ScriptedFrameGenerator {
        remaining: usize,
        began: bool,
        endless: bool,
        polls: usize,
    }

    impl ScriptedFrameGenerator {
        fn emitting(frames: usize) -> Self {
            Self {
                remaining: frames,
                began: false,
                endless: false,
                polls: 0,
            }
        }

        /// Never returns `None`, so only the engine's own ceiling can end the utterance.
        fn endless() -> Self {
            Self {
                remaining: 0,
                began: false,
                endless: true,
                polls: 0,
            }
        }
    }

    impl FrameGenerator for ScriptedFrameGenerator {
        fn begin_utterance(
            &mut self,
            _prepared: &PreparedText,
            _mode: UtteranceStart,
        ) -> Result<(), GenerationError> {
            self.began = true;
            Ok(())
        }

        fn append_text(&mut self, _prepared: &PreparedText) -> Result<(), GenerationError> {
            Err(GenerationError::new(
                "the scripted fake does not model text appends",
            ))
        }

        fn finish_text(&mut self) -> Result<(), GenerationError> {
            Err(GenerationError::new(
                "the scripted fake does not model text appends",
            ))
        }

        fn next_frame(&mut self) -> Result<FrameStep, GenerationError> {
            assert!(self.began, "next_frame before begin_utterance");
            self.polls += 1;
            if self.endless {
                return Ok(FrameStep::Frame(CodeFrame { codes: vec![0; 16] }));
            }
            if self.remaining == 0 {
                return Ok(FrameStep::Finished);
            }
            self.remaining -= 1;
            Ok(FrameStep::Frame(CodeFrame { codes: vec![0; 16] }))
        }
    }

    /// An engine whose admitted ceiling is exactly `max_new_tokens` frames.
    fn engine_with_frame_ceiling(max_new_tokens: u64) -> TtsEngine {
        TtsEngine::new(EngineConfig {
            synthesis_stage_budget: Duration::from_secs(5),
            admission: admission::AdmissionPolicy {
                max_new_tokens,
                ..admission::AdmissionPolicy::default()
            },
            ..EngineConfig::default()
        })
        .expect("test engine builds")
    }

    /// The ceiling the engine admitted this request under, as the observer saw it.
    fn admitted_ceiling(observer: &RecordingObserver) -> u64 {
        observer
            .events()
            .into_iter()
            .find_map(|event| match event {
                SynthesisEvent::ResourceAdmission {
                    admitted: true,
                    predicted_max_frames,
                    ..
                } => Some(predicted_max_frames),
                _ => None,
            })
            .expect("an admitted request reports its frame ceiling")
    }

    struct TestTextPreparer;

    impl TextPreparer for TestTextPreparer {
        fn prepare(
            &self,
            _text: &str,
            options: &NormalizationOptions,
        ) -> Result<PreparedText, TextPreparationError> {
            Ok(PreparedText::new(
                vec![7, 11],
                NormalizationTrace {
                    mode: options.mode,
                    unicode_version: "15.1.0".to_owned(),
                    changes: vec![NormalizationChange {
                        rule: "unicode_nfc",
                        before: "caller-owned secret".to_owned(),
                        after: "caller-owned secret".to_owned(),
                    }],
                },
            ))
        }
    }

    /// The EOS case: the generator decides, and the loop asks exactly once past the last frame.
    ///
    /// Frame count alone cannot make this claim — a ceiling that happened to equal the frame count
    /// would produce the same number. Asserting the ceiling had slack *and* that the loop polled
    /// for the frame after the last one pins the stop to the generator's `None`.
    #[test]
    fn the_decode_loop_stops_on_the_generators_eos_and_polls_exactly_once_past_it() {
        let engine = engine_with_frame_ceiling(64);
        let observer = RecordingObserver::default();
        let mut generator = ScriptedFrameGenerator::emitting(3);

        let result = engine
            .synthesize(
                SynthesisRequest::new(""),
                &TestTextPreparer,
                &mut generator,
                &CancellationToken::new(),
                &observer,
                None,
            )
            .expect("scripted pipeline succeeds");

        let ceiling = admitted_ceiling(&observer);
        assert!(
            ceiling > 3,
            "ceiling {ceiling} must exceed the 3 emitted frames, or the stop is ambiguous"
        );
        assert_eq!(result.generated_frames, 3, "EOS bounds the utterance");
        assert_eq!(result.code_frames.len(), 3);
        assert_eq!(
            generator.polls, 4,
            "the loop must poll once past the last frame to observe EOS, and then stop"
        );
    }

    /// The ceiling case: a generator that never stops is truncated at exactly the admitted ceiling.
    ///
    /// This is where an off-by-one would live, and where nothing else would catch it — a loop that
    /// ran one frame long or short would still look like "it stopped".
    #[test]
    fn a_generator_that_never_stops_is_truncated_exactly_at_the_admitted_ceiling() {
        let engine = engine_with_frame_ceiling(5);
        let observer = RecordingObserver::default();
        let mut generator = ScriptedFrameGenerator::endless();

        let result = engine
            .synthesize(
                SynthesisRequest::new(""),
                &TestTextPreparer,
                &mut generator,
                &CancellationToken::new(),
                &observer,
                None,
            )
            .expect("a ceiling-bound utterance still completes");

        let ceiling = admitted_ceiling(&observer);
        assert_eq!(
            ceiling, 5,
            "the policy's max_new_tokens is the ceiling here"
        );
        assert_eq!(
            result.generated_frames, ceiling,
            "an endless generator must be cut at the ceiling, not one frame either side"
        );
        assert_eq!(result.code_frames.len() as u64, ceiling);
        assert_eq!(
            generator.polls as u64, ceiling,
            "once the ceiling is reached the loop must stop asking, not poll a discarded frame"
        );
    }

    /// The boundary: EOS arriving exactly at the ceiling is still a clean stop, not an overrun.
    #[test]
    fn eos_landing_exactly_on_the_ceiling_yields_the_ceiling_frames() {
        let engine = engine_with_frame_ceiling(4);
        let observer = RecordingObserver::default();
        let mut generator = ScriptedFrameGenerator::emitting(4);

        let result = engine
            .synthesize(
                SynthesisRequest::new(""),
                &TestTextPreparer,
                &mut generator,
                &CancellationToken::new(),
                &observer,
                None,
            )
            .expect("scripted pipeline succeeds");

        assert_eq!(admitted_ceiling(&observer), 4);
        assert_eq!(result.generated_frames, 4);
        assert_eq!(
            generator.polls, 4,
            "the ceiling is reached first, so the generator is never asked for a fifth frame"
        );
    }

    #[test]
    fn the_decode_loop_drives_the_generator_and_reports_every_frame() {
        let engine = engine_with_budget(Duration::from_secs(1));
        let cancellation = CancellationToken::new();
        let observer = RecordingObserver::default();
        let mut generator = ScriptedFrameGenerator::emitting(2);

        let result = engine
            .synthesize(
                SynthesisRequest::new(""),
                &TestTextPreparer,
                &mut generator,
                &cancellation,
                &observer,
                None,
            )
            .expect("scripted pipeline succeeds");

        assert_eq!(result.generated_frames, 2);
        assert_eq!(result.code_frames.len(), 2);
        assert!(
            result
                .code_frames
                .iter()
                .all(|frame| frame.codes.len() == 16)
        );
        assert_eq!(result.prepared_token_count, 2);
        let events = observer.events();
        assert!(
            matches!(
                events.as_slice(),
                [
                    SynthesisEvent::Admission { accepted: true },
                    // Resource admission runs after tokenization and before the first stage.
                    SynthesisEvent::ResourceAdmission { admitted: true, .. },
                    SynthesisEvent::StageStarted {
                        stage: EngineStage::Synthesis,
                    },
                    SynthesisEvent::FrameProgress { frame: 0 },
                    SynthesisEvent::FrameProgress { frame: 1 },
                    SynthesisEvent::StageFinished {
                        stage: EngineStage::Synthesis,
                        ..
                    },
                ]
            ),
            "unexpected event sequence: {events:?}"
        );
    }

    /// The load-bearing promise: an unaffordable request is refused having allocated nothing.
    ///
    /// Proven by the *absence* of any stage event — if a stage had started, work would already have
    /// been committed, which is the "died halfway through" failure admission exists to prevent.
    #[test]
    fn an_unaffordable_request_is_refused_before_any_stage_runs() {
        let mut config = EngineConfig {
            synthesis_stage_budget: Duration::from_secs(1),
            ..EngineConfig::default()
        };
        // A budget far below even the bounded per-utterance state.
        config.admission.budget_bytes = 1;
        let engine = TtsEngine::new(config).expect("engine builds");
        let cancellation = CancellationToken::new();
        let observer = RecordingObserver::default();

        let error = engine
            .synthesize(
                SynthesisRequest::new(""),
                &TestTextPreparer,
                &mut ScriptedFrameGenerator::emitting(0),
                &cancellation,
                &observer,
                None,
            )
            .expect_err("an unaffordable request must be refused");

        assert!(
            matches!(error, EngineError::ResourceAdmission(_)),
            "got {error}"
        );

        let events = observer.events();
        assert!(
            !events.iter().any(|event| matches!(
                event,
                SynthesisEvent::StageStarted { .. }
                    | SynthesisEvent::StageFinished { .. }
                    | SynthesisEvent::FrameProgress { .. }
            )),
            "a refused request must not start any stage; got {events:?}"
        );
        // The rejection is visible in the stream, not only in the error string.
        assert!(
            events.iter().any(|event| matches!(
                event,
                SynthesisEvent::ResourceAdmission {
                    admitted: false,
                    ..
                }
            )),
            "a capacity refusal must appear in the event stream: {events:?}"
        );
    }

    #[test]
    fn the_admission_policy_is_configurable_and_defaults_are_documented() {
        let config = EngineConfig::default();
        assert_eq!(
            config.admission.budget_bytes,
            admission::DEFAULT_BUDGET_BYTES
        );
        assert_eq!(
            config.admission.max_new_tokens,
            admission::DEFAULT_MAX_NEW_TOKENS
        );
        // The default policy applies the text-derived EOS backstop: a 512-token prompt is
        // granted `512 * 4 + 64` frames, not the flat 8,192-frame ceiling — the sampled EOS is a
        // stochastic stop, so an unbounded default would let one unlucky utterance run for
        // minutes. The flat ceiling still binds for explicit caps (see the admission tests).
        let plan = config
            .admission
            .admit(512)
            .expect("the documented default must admit its own worked case");
        assert_eq!(
            plan.predicted_max_frames,
            512 * admission::HEURISTIC_FRAMES_PER_PROMPT_TOKEN
                + admission::HEURISTIC_FRAME_HEADROOM
        );
        assert!(plan.fits());

        // And the worked 8192-frame sizing case still admits when the cap is explicit.
        let mut explicit = config.admission;
        explicit.heuristic_eos_backstop = false;
        let plan = explicit
            .admit(512)
            .expect("the documented explicit-cap case must admit");
        assert_eq!(plan.predicted_max_frames, admission::DEFAULT_MAX_NEW_TOKENS);
        assert!(plan.fits());
    }

    #[test]
    fn cancellation_is_observed_before_the_cpu_stage_starts() {
        let engine = engine_with_budget(Duration::from_secs(1));
        let cancellation = CancellationToken::new();
        cancellation.cancel();
        let observer = RecordingObserver::default();

        let error = engine
            .synthesize(
                SynthesisRequest::new("cancelled"),
                &TestTextPreparer,
                &mut ScriptedFrameGenerator::emitting(0),
                &cancellation,
                &observer,
                None,
            )
            .expect_err("cancelled request must not run");

        assert_eq!(error, EngineError::Cancelled);
        assert_eq!(
            observer.events(),
            vec![
                SynthesisEvent::Admission { accepted: true },
                SynthesisEvent::Health {
                    event: HealthEvent::Cancelled,
                },
            ]
        );
    }

    #[test]
    fn stage_budget_cancels_cooperative_cpu_work() {
        let engine = engine_with_budget(Duration::from_millis(5));
        let cancellation = CancellationToken::new();
        let observer = RecordingObserver::default();

        let error = engine
            .run_stage(
                EngineStage::Synthesis,
                Duration::from_millis(5),
                &cancellation,
                &observer,
                |token| -> Result<(), EngineError> {
                    loop {
                        token.checkpoint()?;
                        thread::sleep(Duration::from_millis(1));
                    }
                },
            )
            .expect_err("long stage must time out");

        assert_eq!(error, EngineError::BudgetExceeded(EngineStage::Synthesis));
        assert!(cancellation.is_cancelled());
        assert!(observer.events().contains(&SynthesisEvent::Health {
            event: HealthEvent::BudgetExceeded,
        }));
    }

    #[test]
    fn pcm_and_events_have_independent_bounded_queues() {
        let queues = StreamQueues::new(1).expect("queue config is valid");
        let cancellation = CancellationToken::new();
        queues
            .events
            .send(SynthesisEvent::Admission { accepted: true }, &cancellation)
            .expect("event queue accepts first event");
        queues
            .pcm
            .send(
                PcmPacket {
                    frame_count: 1,
                    samples: vec![1, -1],
                },
                &cancellation,
            )
            .expect("full event queue cannot block PCM queue");

        assert_eq!(
            queues
                .pcm_receiver
                .recv_timeout(Duration::from_millis(10))
                .expect("PCM arrives"),
            PcmPacket {
                frame_count: 1,
                samples: vec![1, -1],
            }
        );
    }

    #[test]
    fn explicit_normalization_trace_is_text_free() {
        let engine = engine_with_budget(Duration::from_secs(1));
        let observer = RecordingObserver::default();
        let request = SynthesisRequest::new("caller-owned secret")
            .with_normalization_options(NormalizationOptions {
                mode: NormalizationMode::LocaleAware,
                ..NormalizationOptions::default()
            })
            .with_normalization_trace(true);

        engine
            .synthesize(
                request,
                &TestTextPreparer,
                &mut ScriptedFrameGenerator::emitting(0),
                &CancellationToken::new(),
                &observer,
                None,
            )
            .expect("explicit trace request succeeds");

        let trace = observer
            .events()
            .into_iter()
            .find_map(|event| match event {
                SynthesisEvent::TextPrepared {
                    token_count,
                    normalization,
                } => Some((token_count, normalization)),
                _ => None,
            })
            .expect("explicit request emits a trace summary");
        assert_eq!(trace.0, 2);
        assert_eq!(trace.1.mode, NormalizationMode::LocaleAware);
        assert_eq!(trace.1.unicode_version, "15.1.0");
        assert_eq!(trace.1.rules, vec!["unicode_nfc"]);
        assert_eq!(trace.1.change_count, 1);
        assert!(
            !format!("{:?}", trace.1).contains("caller-owned secret"),
            "observer trace summaries must never contain sensitive before/after text"
        );
    }

    #[test]
    fn a_health_violation_reaches_the_caller_through_the_observer() {
        // The wiring the reliability bead requires: a detector firing must be visible to the
        // caller through the SAME hook as every other lifecycle event. A violation that only
        // exists inside the engine is a violation nobody can act on.
        let seen = Arc::new(std::sync::Mutex::new(Vec::new()));
        let sink = Arc::clone(&seen);
        let observer = move |event: SynthesisEvent| {
            if let SynthesisEvent::Health { event } = event {
                sink.lock().expect("observer lock").push(event);
            }
        };

        let violation = health::HealthViolation::OutputSilent {
            silent_millis: 1_500,
        };
        observer(SynthesisEvent::Health {
            event: HealthEvent::Violation(violation),
        });
        let demotion = health::HealthViolation::KernelDemoted {
            from: health::KernelTier::Optimized("i8mm"),
            to: health::KernelTier::Scalar,
        };
        observer(SynthesisEvent::Health {
            event: HealthEvent::Violation(demotion),
        });

        let events = seen.lock().expect("observer lock").clone();
        assert_eq!(events.len(), 2);
        assert_eq!(events[0], HealthEvent::Violation(violation));
        assert_eq!(events[0].as_str(), "output_silent");
        // Silence invalidates the result; a kernel demotion does not — the run is still correct.
        assert!(events[0].invalidates_output());
        assert!(!events[1].invalidates_output());
        assert_eq!(events[1].as_str(), "kernel_demoted");
    }

    #[test]
    fn many_utterances_without_deadlock_watchdog() {
        let (done_sender, done_receiver) = mpsc::sync_channel(1);
        let worker = thread::spawn(move || {
            let engine = engine_with_budget(Duration::from_secs(1));
            let observer = RecordingObserver::default();
            for _ in 0..64 {
                engine
                    .synthesize(
                        SynthesisRequest::new("watchdog"),
                        &TestTextPreparer,
                        &mut ScriptedFrameGenerator::emitting(1),
                        &CancellationToken::new(),
                        &observer,
                        None,
                    )
                    .expect("empty utterance succeeds");
            }
            done_sender
                .send(())
                .expect("watchdog completion receiver lives");
        });

        done_receiver
            .recv_timeout(Duration::from_secs(2))
            .expect("many utterances watchdog expired");
        worker.join().expect("watchdog worker does not panic");
    }

    /// Cancellation-storm variant of the watchdog (bead frankentts-9t5v): many
    /// consecutive utterances on ONE engine, each cancelled at a deterministic
    /// pseudo-random frame boundary. The tripwire for any new channel topology: a
    /// cancel that leaks a slot, wedges a bounded send, or loses the admission guard
    /// hangs this within a few utterances. LCG "randomness" so failures reproduce.
    #[test]
    fn cancel_storm_across_many_utterances_without_deadlock() {
        struct CancelPartway {
            remaining: usize,
            token: CancellationToken,
        }
        impl FrameGenerator for CancelPartway {
            fn begin_utterance(
                &mut self,
                _prepared: &PreparedText,
                _mode: UtteranceStart,
            ) -> Result<(), GenerationError> {
                Ok(())
            }

            fn append_text(&mut self, _prepared: &PreparedText) -> Result<(), GenerationError> {
                Ok(())
            }

            fn finish_text(&mut self) -> Result<(), GenerationError> {
                Ok(())
            }

            fn next_frame(&mut self) -> Result<FrameStep, GenerationError> {
                if self.remaining == 0 {
                    self.token.cancel();
                    return Ok(FrameStep::Finished);
                }
                self.remaining -= 1;
                Ok(FrameStep::Frame(CodeFrame { codes: vec![0; 16] }))
            }
        }

        let (done_sender, done_receiver) = mpsc::sync_channel(1);
        let worker = thread::spawn(move || {
            let engine = engine_with_budget(Duration::from_secs(1));
            let observer = RecordingObserver::default();
            let mut state = 0x2545_F491_4F6C_DD1D_u64;
            let mut next = move || {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (state >> 33) as usize
            };
            for _ in 0..64 {
                let token = CancellationToken::new();
                let frames = 1 + next() % 8;
                let cancel_after = next() % (frames + 1);
                let outcome = engine.synthesize(
                    SynthesisRequest::new("storm"),
                    &TestTextPreparer,
                    &mut CancelPartway {
                        remaining: cancel_after,
                        token: token.clone(),
                    },
                    &token,
                    &observer,
                    None,
                );
                // Both exits are clean: the generator ran out (Ok) or the token
                // tripped mid-loop (Cancelled). Anything else — hang, other error —
                // fails the watchdog below or panics here.
                assert!(
                    outcome.is_ok() || matches!(&outcome, Err(EngineError::Cancelled)),
                    "unexpected storm outcome: {outcome:?}"
                );
            }
            done_sender
                .send(())
                .expect("watchdog completion receiver lives");
        });

        done_receiver
            .recv_timeout(Duration::from_secs(8))
            .expect("cancel-storm watchdog expired");
        worker.join().expect("storm worker does not panic");
    }

    // ------------------------------------------------------------------ continuation stalls

    /// A continuation-aware fake: emits one frame per "supplied" text unit, returns
    /// `AwaitingText` when it catches up on an open continuation, and after `finish_text`
    /// emits `tail` more frames before `Finished` (the model's pad-driven windup).
    struct ContinuationFakeGenerator {
        supplied: usize,
        consumed: usize,
        text_done: bool,
        continuation: bool,
        tail: usize,
        appends: usize,
    }

    impl ContinuationFakeGenerator {
        fn starting_with(supplied: usize, tail: usize) -> Self {
            Self {
                supplied,
                consumed: 0,
                text_done: false,
                continuation: false,
                tail,
                appends: 0,
            }
        }
    }

    impl FrameGenerator for ContinuationFakeGenerator {
        fn begin_utterance(
            &mut self,
            _prepared: &PreparedText,
            mode: UtteranceStart,
        ) -> Result<(), GenerationError> {
            self.continuation = matches!(mode, UtteranceStart::Continuation);
            if !self.continuation {
                self.text_done = true;
            }
            Ok(())
        }

        fn append_text(&mut self, prepared: &PreparedText) -> Result<(), GenerationError> {
            assert!(self.continuation, "append on a fresh utterance");
            assert!(!self.text_done, "append after finish_text");
            self.supplied += prepared.token_ids.len();
            self.appends += 1;
            Ok(())
        }

        fn finish_text(&mut self) -> Result<(), GenerationError> {
            assert!(!self.text_done, "finish_text twice");
            self.text_done = true;
            Ok(())
        }

        fn next_frame(&mut self) -> Result<FrameStep, GenerationError> {
            if self.consumed < self.supplied {
                self.consumed += 1;
                return Ok(FrameStep::Frame(CodeFrame { codes: vec![0; 16] }));
            }
            if !self.text_done {
                return Ok(FrameStep::AwaitingText);
            }
            if self.tail > 0 {
                self.tail -= 1;
                return Ok(FrameStep::Frame(CodeFrame { codes: vec![0; 16] }));
            }
            Ok(FrameStep::Finished)
        }
    }

    fn stall_engine(text_stall_timeout: Duration) -> TtsEngine {
        TtsEngine::new(EngineConfig {
            synthesis_stage_budget: Duration::from_secs(5),
            text_stall_timeout,
            ..EngineConfig::default()
        })
        .expect("engine starts")
    }

    fn appended_text(tokens: usize) -> PreparedText {
        PreparedText::new(
            vec![7; tokens],
            NormalizationTrace {
                mode: NormalizationMode::Verbatim,
                unicode_version: "15.1.0".to_owned(),
                changes: Vec::new(),
            },
        )
    }

    /// Controls queued before the loop starts are drained at frame boundaries: the run
    /// never stalls, no underrun event fires, and appended text extends the utterance.
    #[test]
    fn pre_queued_appends_extend_the_utterance_without_a_stall() {
        let engine = stall_engine(Duration::from_secs(5));
        let observer = RecordingObserver::default();
        let mut generator = ContinuationFakeGenerator::starting_with(2, 1);
        let (feed_tx, feed_rx) = text_control_queue(8).expect("queue");
        let token = CancellationToken::new();
        feed_tx
            .send(TextControl::Append(appended_text(3)), &token)
            .expect("append queued");
        feed_tx
            .send(TextControl::Finish, &token)
            .expect("finish queued");

        let result = engine
            .synthesize(
                SynthesisRequest::new(""),
                &TestTextPreparer,
                &mut generator,
                &CancellationToken::new(),
                &observer,
                Some(&feed_rx),
            )
            .expect("continuation completes");

        // 2 initial + 3 appended + 1 tail frame after finish.
        assert_eq!(result.generated_frames, 6);
        assert_eq!(generator.appends, 1);
        assert!(
            !observer
                .events()
                .iter()
                .any(|event| matches!(event, SynthesisEvent::TextUnderrun { .. })),
            "pre-queued text must not register as an underrun"
        );
    }

    /// A genuine stall: text arrives while the loop is parked in AwaitingText. The run
    /// resumes, completes, and reports the underrun with a non-zero wait.
    #[test]
    fn a_stall_resumes_on_late_text_and_reports_the_underrun() {
        let engine = stall_engine(Duration::from_secs(5));
        let observer = RecordingObserver::default();
        let mut generator = ContinuationFakeGenerator::starting_with(1, 0);
        let (feed_tx, feed_rx) = text_control_queue(8).expect("queue");

        let sender = thread::spawn(move || {
            thread::sleep(Duration::from_millis(60));
            let token = CancellationToken::new();
            feed_tx
                .send(TextControl::Append(appended_text(2)), &token)
                .expect("late append");
            feed_tx.send(TextControl::Finish, &token).expect("finish");
            // Sender drops here; disconnect after Finish is a no-op for the engine.
        });

        let result = engine
            .synthesize(
                SynthesisRequest::new(""),
                &TestTextPreparer,
                &mut generator,
                &CancellationToken::new(),
                &observer,
                Some(&feed_rx),
            )
            .expect("stalled continuation completes");
        sender.join().expect("sender thread");

        assert_eq!(result.generated_frames, 3);
        let underrun_waits: Vec<Duration> = observer
            .events()
            .iter()
            .filter_map(|event| match event {
                SynthesisEvent::TextUnderrun { waited } => Some(*waited),
                _ => None,
            })
            .collect();
        assert!(
            underrun_waits
                .iter()
                .any(|waited| *waited >= Duration::from_millis(40)),
            "the stall must be visible as an underrun with a real wait, got {underrun_waits:?}"
        );
    }

    /// Cancellation reaches a parked stall within the poll interval, not the stall cap.
    #[test]
    fn cancellation_interrupts_a_stall_promptly() {
        let engine = stall_engine(Duration::from_secs(60));
        let observer = RecordingObserver::default();
        let mut generator = ContinuationFakeGenerator::starting_with(1, 0);
        let (_feed_tx, feed_rx) = text_control_queue(8).expect("queue");
        let cancellation = CancellationToken::new();
        let trip = cancellation.clone();
        let canceller = thread::spawn(move || {
            thread::sleep(Duration::from_millis(40));
            trip.cancel();
        });

        let started = Instant::now();
        let error = engine
            .synthesize(
                SynthesisRequest::new(""),
                &TestTextPreparer,
                &mut generator,
                &cancellation,
                &observer,
                Some(&feed_rx),
            )
            .expect_err("cancel during stall must abort");
        canceller.join().expect("canceller");

        assert_eq!(error, EngineError::Cancelled);
        assert!(
            started.elapsed() < Duration::from_secs(5),
            "cancel must not wait out the 60 s stall cap"
        );
    }

    /// The stall cap is a deterministic fallback: with no text ever arriving, the engine
    /// finishes the stream itself and the utterance ends through the generator's windup.
    #[test]
    fn the_stall_cap_finishes_the_text_stream_deterministically() {
        let engine = stall_engine(Duration::from_millis(40));
        let observer = RecordingObserver::default();
        let mut generator = ContinuationFakeGenerator::starting_with(1, 2);
        // Sender stays alive: this is a silent LLM, not a dropped one.
        let (_feed_tx, feed_rx) = text_control_queue(8).expect("queue");

        let result = engine
            .synthesize(
                SynthesisRequest::new(""),
                &TestTextPreparer,
                &mut generator,
                &CancellationToken::new(),
                &observer,
                Some(&feed_rx),
            )
            .expect("capped stall still completes");

        assert_eq!(result.generated_frames, 3, "1 supplied + 2 windup frames");
        assert!(
            observer
                .events()
                .iter()
                .any(|event| matches!(event, SynthesisEvent::TextStallEnded { .. })),
            "the cap must be visible on the record"
        );
    }

    /// Dropping every sender IS finish: the engine treats disconnect as end-of-text.
    #[test]
    fn feed_disconnect_is_end_of_text() {
        let engine = stall_engine(Duration::from_secs(60));
        let observer = RecordingObserver::default();
        let mut generator = ContinuationFakeGenerator::starting_with(2, 1);
        let (feed_tx, feed_rx) = text_control_queue(8).expect("queue");
        drop(feed_tx);

        let started = Instant::now();
        let result = engine
            .synthesize(
                SynthesisRequest::new(""),
                &TestTextPreparer,
                &mut generator,
                &CancellationToken::new(),
                &observer,
                Some(&feed_rx),
            )
            .expect("disconnected feed completes the utterance");

        assert_eq!(result.generated_frames, 3);
        assert!(
            started.elapsed() < Duration::from_secs(5),
            "disconnect must not wait out the stall cap"
        );
    }

    /// Stall time is excluded from the rolling budget: a frame budget that would refuse a
    /// 200 ms delay as a wedge does not refuse the same delay spent waiting for text.
    #[test]
    fn stall_time_is_not_charged_against_the_synthesis_budget() {
        let engine = TtsEngine::new(EngineConfig {
            synthesis_stage_budget: Duration::from_millis(80),
            synthesis_frame_budget: Duration::from_millis(10),
            text_stall_timeout: Duration::from_secs(5),
            ..EngineConfig::default()
        })
        .expect("engine starts");
        let observer = RecordingObserver::default();
        let mut generator = ContinuationFakeGenerator::starting_with(1, 0);
        let (feed_tx, feed_rx) = text_control_queue(8).expect("queue");

        let sender = thread::spawn(move || {
            thread::sleep(Duration::from_millis(200));
            let token = CancellationToken::new();
            feed_tx
                .send(TextControl::Append(appended_text(1)), &token)
                .expect("late append");
            feed_tx.send(TextControl::Finish, &token).expect("finish");
        });

        let result = engine
            .synthesize(
                SynthesisRequest::new(""),
                &TestTextPreparer,
                &mut generator,
                &CancellationToken::new(),
                &observer,
                Some(&feed_rx),
            )
            .expect("a 200 ms text wait is a stall, not a wedge");
        sender.join().expect("sender");
        assert_eq!(result.generated_frames, 2);
    }

    /// An append that would blow the context ceiling is refused, visibly, and the run
    /// keeps speaking the text it already has.
    #[test]
    fn a_ceiling_breaking_append_is_rejected_and_the_run_survives() {
        let engine = stall_engine(Duration::from_millis(40));
        let observer = RecordingObserver::default();
        let mut generator = ContinuationFakeGenerator::starting_with(2, 1);
        let (feed_tx, feed_rx) = text_control_queue(8).expect("queue");
        let token = CancellationToken::new();
        // Far past MAX_CONTEXT_TOKENS in one append.
        feed_tx
            .send(TextControl::Append(appended_text(100_000)), &token)
            .expect("oversized append queued");

        let result = engine
            .synthesize(
                SynthesisRequest::new(""),
                &TestTextPreparer,
                &mut generator,
                &CancellationToken::new(),
                &observer,
                Some(&feed_rx),
            )
            .expect("the run survives a rejected append");

        assert_eq!(
            generator.appends, 0,
            "the rejected append never reached the generator"
        );
        assert_eq!(
            result.generated_frames, 3,
            "2 supplied + 1 windup after the stall cap"
        );
        assert!(
            observer
                .events()
                .iter()
                .any(|event| matches!(event, SynthesisEvent::TextAppendRejected { .. })),
            "the rejection must be on the record"
        );
    }
}
