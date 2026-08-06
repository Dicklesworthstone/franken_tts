#![forbid(unsafe_code)]

//! Safe, blocking public engine primitives.
//!
//! `TtsEngine` owns the one async runtime used below the synchronous public
//! facade. Model work is intentionally absent in Phase 0, but the admission,
//! cancellation, budget, observer, and bounded-streaming contracts are real so
//! later model stages cannot introduce a second orchestration path.

use std::{
    env, fmt,
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
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            stream_queue_capacity: DEFAULT_QUEUE_CAPACITY,
            synthesis_stage_budget: DEFAULT_SYNTHESIS_BUDGET,
            enroll_stage_budget: DEFAULT_ENROLL_BUDGET,
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

/// A synchronous synthesis request.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SynthesisRequest {
    /// Text to synthesize. The Phase 0 shell accepts an empty request.
    pub text: String,
}

impl SynthesisRequest {
    /// Creates a request from caller-owned text.
    #[must_use]
    pub fn new(text: impl Into<String>) -> Self {
        Self { text: text.into() }
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
    /// Admission outcome before model work begins.
    Admission { accepted: bool },
    /// A CPU stage started.
    StageStarted { stage: EngineStage },
    /// A CPU stage completed within its budget.
    StageFinished {
        stage: EngineStage,
        elapsed: Duration,
    },
    /// A talker-frame boundary was reached.
    FrameProgress { frame: u64 },
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
    pub fn synthesize(
        &self,
        _request: SynthesisRequest,
        cancellation: &CancellationToken,
        observer: &dyn SynthesisObserver,
    ) -> Result<SynthesisResult, EngineError> {
        let _admission = self.acquire_synthesis_admission(observer)?;
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
        cancellation.checkpoint().map_err(|error| {
            observer.on_event(SynthesisEvent::Health {
                event: HealthEvent::Cancelled,
            });
            error
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

    #[test]
    fn empty_pipeline_emits_admission_stage_and_frame_events() {
        let engine = engine_with_budget(Duration::from_secs(1));
        let cancellation = CancellationToken::new();
        let observer = RecordingObserver::default();

        let result = engine
            .synthesize(SynthesisRequest::new(""), &cancellation, &observer)
            .expect("empty pipeline succeeds");

        assert_eq!(result.generated_frames, 0);
        let events = observer.events();
        assert!(matches!(
            events.as_slice(),
            [
                SynthesisEvent::Admission { accepted: true },
                SynthesisEvent::StageStarted {
                    stage: EngineStage::Synthesis,
                },
                SynthesisEvent::StageFinished {
                    stage: EngineStage::Synthesis,
                    ..
                },
                SynthesisEvent::FrameProgress { frame: 0 },
            ]
        ));
    }

    #[test]
    fn cancellation_is_observed_before_the_cpu_stage_starts() {
        let engine = engine_with_budget(Duration::from_secs(1));
        let cancellation = CancellationToken::new();
        cancellation.cancel();
        let observer = RecordingObserver::default();

        let error = engine
            .synthesize(SynthesisRequest::new("cancelled"), &cancellation, &observer)
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
    fn many_utterances_without_deadlock_watchdog() {
        let (done_sender, done_receiver) = mpsc::sync_channel(1);
        let worker = thread::spawn(move || {
            let engine = engine_with_budget(Duration::from_secs(1));
            let observer = RecordingObserver::default();
            for _ in 0..64 {
                engine
                    .synthesize(
                        SynthesisRequest::new("watchdog"),
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
