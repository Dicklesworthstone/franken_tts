#![forbid(unsafe_code)]

//! Safe, blocking public engine primitives.
//!
//! `TtsEngine` owns the one async runtime used below the synchronous public
//! facade. Model work is intentionally absent in Phase 0, but the admission,
//! cancellation, budget, observer, and bounded-streaming contracts are real so
//! later model stages cannot introduce a second orchestration path.

pub mod admission;

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
    /// Maximum wall time for one synthesis CPU stage.
    pub synthesis_stage_budget: Duration,
    /// Maximum wall time for one enrollment CPU stage.
    pub enroll_stage_budget: Duration,
    /// Predicted-peak-memory policy applied to every synthesis request.
    pub admission: admission::AdmissionPolicy,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            stream_queue_capacity: DEFAULT_QUEUE_CAPACITY,
            synthesis_stage_budget: DEFAULT_SYNTHESIS_BUDGET,
            enroll_stage_budget: DEFAULT_ENROLL_BUDGET,
            admission: admission::AdmissionPolicy::default(),
        }
    }
}

impl EngineConfig {
    fn from_environment() -> Self {
        let mut config = Self::default();
        config.synthesis_stage_budget = stage_budget_from_environment(
            "FTTS_STAGE_BUDGET_SYNTHESIS_MS",
            config.synthesis_stage_budget,
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
        config.admission.max_new_tokens = positive_u64_from_environment("FTTS_MAX_FRAMES")
            .unwrap_or(config.admission.max_new_tokens);
        config
    }

    fn validate(&self) -> Result<(), EngineError> {
        if self.stream_queue_capacity == 0 {
            return Err(EngineError::InvalidConfiguration(
                "stream queue capacity must be greater than zero",
            ));
        }
        if self.synthesis_stage_budget.is_zero() || self.enroll_stage_budget.is_zero() {
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

/// A completed empty-pipeline synthesis result.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SynthesisResult {
    /// Number of generated codec frames.
    pub generated_frames: u64,
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

    /// Runs the Phase 0 empty synthesis pipeline through the owned runtime.
    pub fn synthesize<P: TextPreparer + ?Sized>(
        &self,
        request: SynthesisRequest,
        text_preparer: &P,
        cancellation: &CancellationToken,
        observer: &dyn SynthesisObserver,
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
        match self.config.admission.admit(prompt_tokens) {
            Ok(plan) => observer.on_event(SynthesisEvent::ResourceAdmission {
                admitted: true,
                predicted_max_frames: plan.predicted_max_frames,
                predicted_peak_bytes: plan.predicted_peak_bytes,
                budget_bytes: plan.budget_bytes,
            }),
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
        }

        self.run_stage(
            EngineStage::Synthesis,
            self.config.synthesis_stage_budget,
            cancellation,
            observer,
            |_| Ok(()),
        )?;
        observer.on_event(SynthesisEvent::FrameProgress { frame: 0 });
        Ok(SynthesisResult {
            generated_frames: 0,
            prepared_token_count: prepared.token_ids.len(),
        })
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

    #[test]
    fn empty_pipeline_emits_admission_stage_and_frame_events() {
        let engine = engine_with_budget(Duration::from_secs(1));
        let cancellation = CancellationToken::new();
        let observer = RecordingObserver::default();

        let result = engine
            .synthesize(
                SynthesisRequest::new(""),
                &TestTextPreparer,
                &cancellation,
                &observer,
            )
            .expect("empty pipeline succeeds");

        assert_eq!(result.generated_frames, 0);
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
                    SynthesisEvent::StageFinished {
                        stage: EngineStage::Synthesis,
                        ..
                    },
                    SynthesisEvent::FrameProgress { frame: 0 },
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
                &cancellation,
                &observer,
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
        // The default policy must admit the ordinary case it was sized for: an 8192-frame cap at a
        // 512-token prompt is 952 MiB of talker KV, which has to fit inside the 2 GiB default.
        let plan = config
            .admission
            .admit(512)
            .expect("the documented default must admit its own worked case");
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
                &cancellation,
                &observer,
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
                &CancellationToken::new(),
                &observer,
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
                        &CancellationToken::new(),
                        &observer,
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
}
