//! KernelTeam v0: the persistent worker pool for int8 GEMV/GEMM output-column partitions.
//!
//! This is the doctrine's "persistent, dispatch-free steady state" in its first shippable form:
//! workers are spawned once per process, parked on a condvar between operations (no busy wait,
//! no work stealing, no task submission), and each dispatch hands every worker a disjoint
//! contiguous range of output columns. Integer accumulation makes the parallel result *exactly*
//! the serial result per element — partitioning never changes a single output bit, so thread
//! count is a pure speed knob.
//!
//! ## The safety argument, in full
//!
//! A [`Job`] carries raw pointers into the caller's slices. Three facts make that sound:
//!
//! 1. **Lifetime**: [`Team::linear_q8`] does not return until every worker has decremented
//!    `pending` to zero, so the pointers outlive every access.
//! 2. **Aliasing**: workers write only `out[row * n + col]` for `col` inside their own disjoint
//!    column range; reads (`x_q`, scales, weights, bias) are shared and immutable for the whole
//!    dispatch because the caller holds the only `&mut` (to `out`) and blocks.
//! 3. **One parallel owner per team**: `dispatch_gate` serializes whole dispatches, so a second
//!    thread cannot overwrite a team's job while its workers are mid-partition, and workers
//!    themselves never dispatch (their compute is a leaf loop).
//!
//! ## Two disjoint teams: engine and codec
//!
//! The codec decodes concurrently with generation on its own worker thread. Off Apple platforms
//! (where Accelerate does not absorb its GEMMs) the codec is compute-bound — about 2.5 GMAC per
//! 80 ms frame — and on small machines one serial codec thread is the whole pipeline's ceiling.
//! So the codec worker may own a second, fully separate team ([`use_codec_team_on_this_thread`]):
//! its own control block, workers, and dispatch gate. The engine-team invariants above hold for
//! each team independently; no worker of either team ever dispatches, so nothing nests; and the
//! default sizes keep `engine + codec <= cores`, so the two never oversubscribe the machine
//! (a preempted worker stalls its whole barrier). Partitioning never changes a bit in either
//! team, so both sizes are pure speed knobs.
//!
//! A stress test drives thousands of mixed-shape dispatches and a watchdog test bounds wall
//! time, per the `many_utterances_without_deadlock` policy.
//!
//! ## Relation to the plan's "sense-reversing barrier"
//!
//! The doctrine text describes the steady-state rendezvous as a sense-reversing atomic
//! barrier. What ships here is a **generation-counter** barrier with a spin-then-park fast path:
//! the dispatcher publishes the job under the mutex, mirrors the generation into an atomic
//! `epoch` (Release), and sets an atomic `pending` count first; workers spin briefly on `epoch`
//! and the dispatcher on `pending` (see `spin_window`), falling back to the condvars, which are
//! notified only when someone is actually parked (`sleepers` for `go`; the last worker always
//! takes the lock before notifying `done`, after the dispatcher's under-lock check, so no wake
//! is lost). Jobs are still read under the lock they were written under. Dispatch overhead did
//! show up on a profile — ~430 dispatches per frame at a measured 26–62 µs park/wake round trip
//! on a 4-vCPU Linux host — which is what promoted the atomic fast path.

use crate::int8::{Int8Tier, QuantizedMatrix, dot_w8a16};
use std::sync::{Condvar, Mutex, OnceLock};

/// Ask Apple to schedule the current worker at the user-initiated QoS class.
///
/// Kernel-team workers and the concurrent codec worker both serve a foreground synthesis
/// request. Keeping the platform call here preserves one audited unsafe boundary instead of
/// duplicating it in higher-level crates. Other platforms deliberately report `false` and leave
/// scheduling unchanged.
#[cfg(target_vendor = "apple")]
pub fn request_user_initiated_qos_for_current_thread() -> bool {
    // SAFETY: this changes only the calling thread's scheduling class. It does not dereference
    // pointers, transfer ownership, or alter any memory-safety contract.
    #[allow(unsafe_code)]
    unsafe {
        libc::pthread_set_qos_class_self_np(libc::qos_class_t::QOS_CLASS_USER_INITIATED, 0) == 0
    }
}

#[cfg(not(target_vendor = "apple"))]
pub fn request_user_initiated_qos_for_current_thread() -> bool {
    false
}

/// One dispatched operation, shared read-only with every worker.
#[derive(Clone, Copy)]
enum Job {
    /// W8A8 linear partitioned over output columns.
    Linear(LinearJob),
    /// W8A16 (weight-only) linear partitioned over output columns.
    W8A16Linear(W8A16LinearJob),
    /// f32 GQA attention partitioned over query heads.
    Attention(AttentionJob),
    /// f32 dense linear (the packed GEMM) partitioned over output columns.
    F32Linear(F32LinearJob),
}

/// The codec's dense route, partitioned over output columns.
///
/// This is the codec's whole arithmetic budget: every convolution (via im2col), every ConvNeXt
/// pointwise pair, and every transformer projection reaches one function, and that function
/// measured 92% of browser frame time while running on a single thread.
///
/// Column partitioning is exact here for the same reason it is for the int8 job: no reduction
/// crosses a column, so a stripe computed in isolation has the bits the whole call would have
/// written (`packed_gemm::column_partitions_are_bit_identical_to_the_whole`).
#[derive(Clone, Copy)]
struct F32LinearJob {
    x: *const f32,
    weight: *const f32,
    /// Null when the projection is bias-free.
    bias: *const f32,
    out: *mut f32,
    m: usize,
    k: usize,
    n: usize,
    partitions: usize,
}

#[derive(Clone, Copy)]
struct LinearJob {
    x_q: *const i8,
    x_scales: *const f32,
    w_data: *const i8,
    w_scales: *const f32,
    /// Null when the projection is bias-free.
    bias: *const f32,
    out: *mut f32,
    m: usize,
    n: usize,
    k: usize,
    tier: Int8Tier,
    /// Total partitions this dispatch, including the caller's partition 0.
    partitions: usize,
}

/// The weight-only quantized route (`FTTS_INT8=w8a16`), partitioned over output columns.
///
/// Column partitioning is exact for the same reason as the W8A8 job: no reduction crosses a
/// column, and every element is one `dot_w8a16` over the same span in the same order the
/// serial loop would run.
#[derive(Clone, Copy)]
struct W8A16LinearJob {
    x: *const f32,
    w_data: *const i8,
    w_scales: *const f32,
    /// Null when the projection is bias-free.
    bias: *const f32,
    out: *mut f32,
    m: usize,
    n: usize,
    k: usize,
    /// Total partitions this dispatch, including the caller's partition 0.
    partitions: usize,
}

/// The default-arithmetic GQA attention, partitioned over query heads.
///
/// Head independence is the whole safety-and-exactness story: no reduction crosses a head, and
/// each head writes only its own `head_dim` span of every output row, so any head partition is
/// bit-identical to the serial full-range call.
#[derive(Clone, Copy)]
struct AttentionJob {
    queries: *const f32,
    keys: *const f32,
    values: *const f32,
    mask: *const f32,
    query_positions: usize,
    key_positions: usize,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    out: *mut f32,
    partitions: usize,
}

// SAFETY: the pointers a Job carries are dereferenced only between dispatch and join (module
// docs, fact 1), reads are shared-immutable and writes disjoint (fact 2). Sending the
// descriptor to parked threads is exactly the mechanism those facts govern.
unsafe impl Send for Job {}
// SAFETY: workers only read the descriptor fields; interior data races are excluded by the
// disjoint-write partition argument above.
unsafe impl Sync for Job {}

struct Control {
    generation: u64,
    job: Option<Job>,
    /// Workers currently parked on `go`; the dispatcher notifies only when this is non-zero.
    sleepers: usize,
    /// Set when any partition panicked during the current dispatch, so the caller can
    /// propagate a loud failure instead of hanging on a worker that will never report done.
    panicked: bool,
}

struct Shared {
    control: Mutex<Control>,
    go: Condvar,
    done: Condvar,
    /// Mirror of `control.generation`, published (Release) after the job is in place, so a
    /// spinning worker can see a new dispatch without taking the lock.
    epoch: std::sync::atomic::AtomicU64,
    /// Worker partitions of the current dispatch that have not reported done. Set before the
    /// epoch is published; the last decrement wakes a parked dispatcher.
    pending: std::sync::atomic::AtomicUsize,
    /// Whether this team's threads may spin before parking. Only a team that leaves the machine
    /// undersubscribed spins: on a fully subscribed machine a spinning worker steals the core of
    /// the very thread it is waiting for (measured: a 4-way team on 4 vCPUs went from ~40 µs to
    /// ~96 µs per dispatch with spinning on).
    spin: bool,
}

impl Shared {
    fn new(spin: bool) -> Self {
        Self {
            control: Mutex::new(Control {
                generation: 0,
                job: None,
                sleepers: 0,
                panicked: false,
            }),
            go: Condvar::new(),
            done: Condvar::new(),
            epoch: std::sync::atomic::AtomicU64::new(0),
            pending: std::sync::atomic::AtomicUsize::new(0),
            spin,
        }
    }
}

/// How long a worker (between dispatches) or the dispatcher (awaiting its workers) spins on an
/// atomic before parking on a condvar.
///
/// The decode loop issues ~430 small dispatches per 80 ms frame (four per layer step: 28 talker
/// layers, 15 × 5 microdecoder steps), separated by short serial stretches (norms, attention,
/// sampling). A park/wake round trip measured ~26–62 µs on a 4-vCPU Linux host — tens of
/// milliseconds per frame of pure handshake — while the work itself is often under 100 µs.
/// Spinning briefly turns most of those into an atomic load. The window is bounded so an idle
/// team (between utterances, or while the generator samples) still parks and costs nothing.
/// Apple platforms keep the measured park-immediately behavior (battery, and their dispatch
/// overhead was never shown to matter); `FTTS_TEAM_SPIN_US` overrides everywhere. Timing only:
/// which thread computes which partition is unchanged, so no output bit can move.
#[cfg(not(target_arch = "wasm32"))]
fn spin_window() -> std::time::Duration {
    static WINDOW: OnceLock<std::time::Duration> = OnceLock::new();
    *WINDOW.get_or_init(|| {
        let default_us = if cfg!(target_vendor = "apple") { 0 } else { 50 };
        let micros = std::env::var("FTTS_TEAM_SPIN_US")
            .ok()
            .and_then(|value| value.trim().parse::<u64>().ok())
            .unwrap_or(default_us)
            .min(10_000);
        std::time::Duration::from_micros(micros)
    })
}

/// Spins until `ready()` or the spin window elapses; returns whether `ready()` became true.
///
/// Never spins for a team built without `spin` (see [`Shared`]). wasm never spins: it has no
/// monotonic clock to bound the window (`Instant::now` traps), and its Workers already park
/// cheaply on `atomic.wait`.
fn spin_until(shared: &Shared, ready: impl Fn() -> bool) -> bool {
    if ready() {
        return true;
    }
    if !shared.spin {
        return false;
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let window = spin_window();
        if window.is_zero() {
            return false;
        }
        let start = std::time::Instant::now();
        loop {
            for _ in 0..32 {
                std::hint::spin_loop();
                if ready() {
                    return true;
                }
            }
            if start.elapsed() >= window {
                return ready();
            }
        }
    }
    #[cfg(target_arch = "wasm32")]
    {
        false
    }
}

/// A kernel team. The engine team is armed by default at min(6, cores) partitions on Apple and
/// min(6, cores - 1) elsewhere (`FTTS_INT8_THREADS` overrides; 1 disarms), and explicitly by the
/// host on wasm; the codec team takes the remaining cores up to four off Apple
/// (`FTTS_CODEC_THREADS` overrides; 1 disarms). See the module docs.
pub struct Team {
    shared: &'static Shared,
    /// Total partitions per dispatch: spawned workers + the calling thread.
    partitions: usize,
    dispatch_gate: Mutex<()>,
}

thread_local! {
    static TEAM_BYPASS: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// Set on the codec worker: [`armed`] resolves to the codec team instead of the engine team.
    static CODEC_TEAM_THREAD: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

struct TeamBypassReset(bool);

impl Drop for TeamBypassReset {
    fn drop(&mut self) {
        TEAM_BYPASS.with(|cell| cell.set(self.0));
    }
}

/// Partitions the armed team runs with, or 1 when execution is serial.
///
/// Exposed so a host with no environment variables — a browser — can report whether threading
/// actually engaged, instead of running serially and looking identical.
#[must_use]
pub fn partitions() -> usize {
    armed().map_or(1, |team| team.partitions)
}

/// Makes THIS thread run its int8 linears serially, never dispatching to the team.
///
/// The codec pipeline worker sets this: its work is meant to overlap with the generator's
/// team dispatches on spare cores, and routing it through the shared team would merely
/// interleave the two through the dispatch gate instead of running them concurrently.
pub fn bypass_team_on_this_thread() {
    TEAM_BYPASS.with(|cell| cell.set(true));
}

/// Makes THIS thread dispatch to the codec team (module docs) instead of the engine team.
///
/// For the codec pipeline worker: it overlaps the generator's engine-team dispatches, so it must
/// never contend for that team's gate — but where spare cores exist it can still fan its own
/// GEMMs out. When the codec team is disarmed (sized to one partition, or on wasm) this thread
/// runs serially, exactly like [`bypass_team_on_this_thread`].
pub fn use_codec_team_on_this_thread() {
    CODEC_TEAM_THREAD.with(|cell| cell.set(true));
}

/// Partitions of the codec team, or 1 while it is disarmed or not yet armed by a codec thread.
///
/// Does not arm the team as a side effect, so reporting it costs no threads.
#[must_use]
pub fn codec_partitions() -> usize {
    #[cfg(not(target_arch = "wasm32"))]
    {
        CODEC_TEAM
            .get()
            .and_then(Option::as_ref)
            .map_or(1, |team| team.partitions)
    }
    #[cfg(target_arch = "wasm32")]
    {
        1
    }
}

/// Runs `body` with team dispatch bypassed on this thread, restoring the previous state after.
///
/// For callers that need the SERIAL kernel for a bounded stretch — the int8 autotuner probes
/// tier cost, and a probe routed through the team would time dispatch overhead plus whatever
/// the workers are doing, not the tier — without permanently opting the thread out the way
/// [`bypass_team_on_this_thread`] (meant for worker threads) does.
pub fn with_team_bypassed<R>(body: impl FnOnce() -> R) -> R {
    let previous = TEAM_BYPASS.with(std::cell::Cell::get);
    TEAM_BYPASS.with(|cell| cell.set(true));
    // Restoration must survive an unwind: the autotuner deliberately contains probe
    // failures, and leaving this thread bypassed after one would silently route every
    // later kernel through the serial path.
    let _reset = TeamBypassReset(previous);
    body()
}

/// Whether the current thread opted out of team dispatch.
#[must_use]
pub fn thread_bypassed() -> bool {
    TEAM_BYPASS.with(std::cell::Cell::get)
}

/// The team this thread dispatches to, if parallel execution is enabled for it: the codec team on
/// a thread that called [`use_codec_team_on_this_thread`], else the engine team.
///
/// `FTTS_INT8_THREADS` / `FTTS_CODEC_THREADS` set each team's total partition count (caller
/// included); `1` means serial (no threads spawned, no team). Values are clamped to the machine's
/// available parallelism. Read once per team.
pub fn armed() -> Option<&'static Team> {
    // wasm32 cannot spawn its own threads: `wasm32-unknown-unknown` has no `std::thread::spawn`,
    // because only the host can create the Workers that share this module's linear memory. So the
    // team is *installed* from JS once its Workers are up (see `install_wasm_team`) instead of
    // being created on first use, and stays `None` until then — which is also the correct answer
    // for any browser without `SharedArrayBuffer`.
    let codec_thread = CODEC_TEAM_THREAD.with(std::cell::Cell::get);
    #[cfg(target_arch = "wasm32")]
    {
        // The browser has one host-installed team; a codec thread keeps its serial behavior.
        if codec_thread {
            None
        } else {
            WASM_TEAM.get().and_then(Option::as_ref)
        }
    }
    #[cfg(not(target_arch = "wasm32"))]
    if codec_thread {
        codec_team_native()
    } else {
        armed_native()
    }
}

/// The team installed by the host, once its Workers exist.
#[cfg(target_arch = "wasm32")]
static WASM_TEAM: OnceLock<Option<Team>> = OnceLock::new();

/// The shared control block Workers park on, published before any of them starts.
#[cfg(target_arch = "wasm32")]
static WASM_SHARED: OnceLock<&'static Shared> = OnceLock::new();

/// Publishes the shared control block Workers park on, without sizing the team.
///
/// Split from [`arm_wasm_team`] deliberately. A Worker must be able to park *before* the team
/// exists, because the team's width has to be the number of Workers that actually started — and
/// that is not known until they report. Publishing first, arming second, is what makes a Worker
/// that fails to boot cost a partition rather than a hang.
///
/// # Why any of this runs in a Worker
///
/// The dispatcher is partition 0 and blocks on a condvar until the others report done.
/// `atomic.wait` **traps on a browser's main thread**, so the owning thread must itself be a
/// Worker — here, the engine Worker that already runs synthesis. Arming from the main thread
/// would not merely be slow, it would abort.
#[cfg(target_arch = "wasm32")]
pub fn publish_wasm_block() {
    let _ = WASM_SHARED.get_or_init(|| Box::leak(Box::new(Shared::new(false))));
}

/// Arms a `partitions`-way team over the already-published control block.
///
/// Call this only after `partitions - 1` Workers have confirmed they are parked in
/// [`wasm_worker_loop`]. Sizing the team before they report would be a deadlock waiting to
/// happen: the dispatcher waits for `pending` to count down from `partitions - 1` and blocks until it
/// reaches zero, so a partition that never started is a partition that never reports done.
///
/// `partitions <= 1` arms nothing, which is the serial fallback a browser without
/// `SharedArrayBuffer` — or one where every Worker failed to start — correctly lands on.
#[cfg(target_arch = "wasm32")]
pub fn arm_wasm_team(partitions: usize) {
    if partitions <= 1 {
        let _ = WASM_TEAM.set(None);
        return;
    }
    publish_wasm_block();
    let shared = *WASM_SHARED.get().expect("just published");
    let _ = WASM_TEAM.set(Some(Team {
        shared,
        partitions,
        dispatch_gate: Mutex::new(()),
    }));
}

/// The body every spawned Worker runs, forever.
///
/// # Panics
///
/// Panics if called before [`install_wasm_team`] published the control block — a Worker that
/// started before its team is a host wiring bug, and parking on a block that does not exist yet
/// would hang instead of saying so.
#[cfg(target_arch = "wasm32")]
pub fn wasm_worker_loop(worker: usize) {
    let shared = *WASM_SHARED
        .get()
        .expect("worker started before install_wasm_team published the control block");
    worker_loop(shared, worker)
}

/// The engine team's default partition count on a machine with `ceiling` hardware threads.
///
/// Six ways is the measured knee on M4 Pro (memory-bound beyond it). Off Apple platforms one
/// hardware thread is left to the codec worker, which decodes concurrently and is compute-bound
/// there: measured on a 4-vCPU x86 host, a 4-way engine team plus the codec thread (five busy
/// threads on four cores) was ~10% slower end to end than a 3-way team, because the barrier waits
/// for whichever worker the scheduler preempted. Apple keeps the measured six — its codec GEMMs go
/// through Accelerate and it has efficiency cores to spare.
#[cfg(any(not(target_arch = "wasm32"), test))]
fn engine_partitions_default(ceiling: usize) -> usize {
    if cfg!(target_vendor = "apple") {
        6.min(ceiling)
    } else {
        6.min(ceiling.saturating_sub(1)).max(1)
    }
}

/// The codec team's default partition count: the hardware threads the engine team leaves free,
/// at most four (the codec's GEMMs are short; wider stripes buy little), and serial on Apple
/// platforms, whose codec GEMMs go through Accelerate rather than this team.
#[cfg(any(not(target_arch = "wasm32"), test))]
fn codec_partitions_default(ceiling: usize, engine: usize) -> usize {
    if cfg!(target_vendor = "apple") {
        1
    } else {
        ceiling.saturating_sub(engine).clamp(1, 4)
    }
}

/// Reads a partition-count override, clamped to `ceiling`.
#[cfg(not(target_arch = "wasm32"))]
fn partitions_from_env(name: &str, default: usize, ceiling: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
        .min(ceiling)
}

#[cfg(not(target_arch = "wasm32"))]
fn hardware_threads() -> usize {
    std::thread::available_parallelism().map_or(1, usize::from)
}

#[cfg(not(target_arch = "wasm32"))]
fn engine_partitions() -> usize {
    let ceiling = hardware_threads();
    // Partitioning never changes output bits, so the default applies everywhere, reference
    // route included.
    partitions_from_env(
        "FTTS_INT8_THREADS",
        engine_partitions_default(ceiling),
        ceiling,
    )
}

/// The codec team's configured width (`1` = serial), whether or not a codec thread armed it.
#[cfg(not(target_arch = "wasm32"))]
fn codec_partitions_configured() -> usize {
    let ceiling = hardware_threads();
    partitions_from_env(
        "FTTS_CODEC_THREADS",
        codec_partitions_default(ceiling, engine_partitions()),
        ceiling,
    )
}

/// Whether both teams together — the codec worker counting as one thread even when serial —
/// fit the machine's hardware threads, the precondition for spinning (see [`Shared`]).
#[cfg(not(target_arch = "wasm32"))]
fn teams_undersubscribe() -> bool {
    engine_partitions() + codec_partitions_configured() <= hardware_threads()
}

#[cfg(not(target_arch = "wasm32"))]
fn armed_native() -> Option<&'static Team> {
    static TEAM: OnceLock<Option<Team>> = OnceLock::new();
    TEAM.get_or_init(|| spawn_native_team(engine_partitions(), "ftts-int8"))
        .as_ref()
}

/// The codec team, armed on first use by a codec thread.
#[cfg(not(target_arch = "wasm32"))]
static CODEC_TEAM: OnceLock<Option<Team>> = OnceLock::new();

#[cfg(not(target_arch = "wasm32"))]
fn codec_team_native() -> Option<&'static Team> {
    CODEC_TEAM
        .get_or_init(|| spawn_native_team(codec_partitions_configured(), "ftts-codec"))
        .as_ref()
}

/// Spawns a `partitions`-way team (`partitions - 1` workers; the dispatcher is partition 0), or
/// `None` for a serial one.
#[cfg(not(target_arch = "wasm32"))]
fn spawn_native_team(partitions: usize, name: &str) -> Option<Team> {
    if partitions <= 1 {
        return None;
    }
    let shared: &'static Shared = Box::leak(Box::new(Shared::new(teams_undersubscribe())));
    // Workers 1..partitions; the caller is partition 0. Threads live for the process and
    // park on the condvar between dispatches, so leaking their handles is deliberate.
    for worker in 1..partitions {
        std::thread::Builder::new()
            .name(format!("{name}-{worker}"))
            // Debug builds inline the kernel dispatch chains deeply enough to
            // overflow the 2 MiB default worker stack non-deterministically (the
            // startup autotuner picks tiers under memory pressure), which surfaced
            // as spontaneous aborts in the metamorphic invariants. Release builds
            // have headroom either way; 16 MiB is cheap for long-lived parked
            // threads and removes the entire failure class.
            .stack_size(16 * 1024 * 1024)
            .spawn(move || {
                // On Apple platforms a thread without an elevated QoS class is fair
                // game for the efficiency cores. The team barrier waits for its
                // slowest member, so one demoted worker sets the pace of every
                // dispatch; ask for the same class the caller's UI work runs at.
                let _ = request_user_initiated_qos_for_current_thread();
                worker_loop(shared, worker)
            })
            .expect("spawn kernel-team worker");
    }
    Some(Team {
        shared,
        partitions,
        dispatch_gate: Mutex::new(()),
    })
}

fn worker_loop(shared: &'static Shared, worker: usize) {
    use std::sync::atomic::Ordering;
    let mut seen = 0_u64;
    loop {
        // Spin briefly on the published epoch before parking (see `spin_window`); either way the
        // job is read under the lock, which the dispatcher wrote it under.
        let _ = spin_until(shared, || shared.epoch.load(Ordering::Acquire) != seen);
        let job = {
            let mut control = lock_control(shared);
            if control.generation == seen {
                control.sleepers += 1;
                while control.generation == seen {
                    control = shared
                        .go
                        .wait(control)
                        .unwrap_or_else(std::sync::PoisonError::into_inner);
                }
                control.sleepers -= 1;
            }
            seen = control.generation;
            control.job.expect("generation bumped without a job")
        };
        // A panicking partition must still report done, or the caller hangs forever waiting
        // for a decrement that will never come. The panic is recorded and re-raised loudly on
        // the caller's thread instead.
        #[cfg(test)]
        let injected = worker > 0 && tests::panic_injected_for(shared);
        #[cfg(not(test))]
        let injected = false;
        let outcome = std::panic::catch_unwind(|| {
            if injected {
                panic!("injected worker panic for the hang-hardening test");
            }
            run_partition(&job, worker)
        });
        if outcome.is_err() {
            lock_control(shared).panicked = true;
        }
        if shared.pending.fetch_sub(1, Ordering::AcqRel) == 1 {
            // Last one out wakes the dispatcher if it parked. Taking the lock orders this notify
            // after the dispatcher's own under-lock check of `pending`, so the wake cannot be lost.
            let _control = lock_control(shared);
            shared.done.notify_all();
        }
    }
}

/// Locks team control, tolerating poison: every dispatch re-establishes the full invariant
/// (job, generation, pending) from scratch, so a lock poisoned by an earlier panic carries
/// no state that could mislead the next dispatch.
fn lock_control(shared: &Shared) -> std::sync::MutexGuard<'_, Control> {
    shared
        .control
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Computes one worker's contiguous column range. Identical arithmetic to the serial
/// weight-stationary loop in [`crate::int8::linear_q8`], restricted to `[start, end)`.
fn run_partition(job: &Job, worker: usize) {
    let _ = worker;
    match job {
        Job::Linear(job) => run_linear_partition(job, worker),
        Job::W8A16Linear(job) => run_w8a16_linear_partition(job, worker),
        Job::Attention(job) => run_attention_partition(job, worker),
        Job::F32Linear(job) => run_f32_linear_partition(job, worker),
    }
}

/// One worker's query-head range of an attention job. Same extracted loop the serial reference
/// runs (`f32ref::gqa_attention_head_range_with_arithmetic` with the default arithmetic).
fn run_attention_partition(job: &AttentionJob, worker: usize) {
    let chunk = job.q_heads.div_ceil(job.partitions);
    let start = (worker * chunk).min(job.q_heads);
    let end = ((worker + 1) * chunk).min(job.q_heads);
    if start >= end {
        return;
    }
    // SAFETY: same three facts as the linear job (module docs) — the caller joins before its
    // slices can die, and reads are shared-immutable for the dispatch. The output stays a raw
    // pointer: every worker turning it into a whole-buffer `&mut` would put several live `&mut`
    // on one allocation, which is undefined behaviour even though the writes are disjoint.
    let (queries, keys, values, mask) = unsafe {
        (
            std::slice::from_raw_parts(
                job.queries,
                job.query_positions * job.q_heads * job.head_dim,
            ),
            std::slice::from_raw_parts(job.keys, job.key_positions * job.kv_heads * job.head_dim),
            std::slice::from_raw_parts(job.values, job.key_positions * job.kv_heads * job.head_dim),
            std::slice::from_raw_parts(job.mask, job.query_positions * job.key_positions),
        )
    };
    // SAFETY: `out` is valid for the full [query_positions, q_heads, head_dim] span for this
    // dispatch, and this worker's `start..end` head range is disjoint from every other
    // partition's, so no two live borrows ever overlap.
    unsafe {
        crate::f32ref::gqa_attention_head_range_into(
            queries,
            keys,
            values,
            mask,
            job.query_positions,
            job.key_positions,
            job.q_heads,
            job.kv_heads,
            job.head_dim,
            crate::f32ref::F32SoftmaxArithmetic::ReciprocalMultiply,
            crate::f32ref::F32LinearAccumulation::Scalar,
            start..end,
            job.out,
        );
    }
}

fn run_linear_partition(job: &LinearJob, worker: usize) {
    let chunk = job.n.div_ceil(job.partitions);
    let start = (worker * chunk).min(job.n);
    let end = ((worker + 1) * chunk).min(job.n);
    if start >= end {
        return;
    }
    // SAFETY: module-docs facts 1-3 — pointers outlive the dispatch, reads are shared-immutable,
    // and this worker writes only columns in its own [start, end) range. The output deliberately
    // stays a raw pointer: a whole-buffer `&mut` per worker would be several live `&mut` on one
    // allocation, which is undefined behaviour regardless of the writes being disjoint, and
    // `rustc` marks `&mut` `noalias` so the optimizer is entitled to act on it.
    let (x_q, x_scales, w_data, w_scales, bias) = unsafe {
        (
            std::slice::from_raw_parts(job.x_q, job.m * job.k),
            std::slice::from_raw_parts(job.x_scales, job.m),
            std::slice::from_raw_parts(job.w_data, job.n * job.k),
            std::slice::from_raw_parts(job.w_scales, job.n),
            (!job.bias.is_null()).then(|| std::slice::from_raw_parts(job.bias, job.n)),
        )
    };
    // SAFETY: `out` spans the full [m, n] output for this dispatch, and `start..end` is this
    // partition's exclusive column range, so no other partition writes these cells. The loop nest
    // is the serial kernel's own (`linear_q8_columns`), which is what keeps the two bit-identical.
    unsafe {
        crate::int8::linear_q8_columns(
            x_q,
            x_scales,
            w_data,
            w_scales,
            bias,
            job.m,
            job.n,
            job.k,
            job.tier,
            start..end,
            job.out,
        );
    }
}

/// One worker's column stripe of a W8A16 job — the same loop `int8::linear_w8a16` runs
/// serially, sharing `dot_w8a16`, so each element's f32 operation order is unchanged.
fn run_w8a16_linear_partition(job: &W8A16LinearJob, worker: usize) {
    let chunk = job.n.div_ceil(job.partitions);
    let start = (worker * chunk).min(job.n);
    let end = ((worker + 1) * chunk).min(job.n);
    if start >= end {
        return;
    }
    // SAFETY: module-docs facts 1-3, the same aliasing story as the W8A8 job — pointers outlive
    // the dispatch, reads are shared-immutable, writes land only in this worker's column range,
    // and the output stays a raw pointer because several live whole-buffer `&mut` would be UB
    // even with disjoint writes.
    let (x, w_data, w_scales, bias) = unsafe {
        (
            std::slice::from_raw_parts(job.x, job.m * job.k),
            std::slice::from_raw_parts(job.w_data, job.n * job.k),
            std::slice::from_raw_parts(job.w_scales, job.n),
            (!job.bias.is_null()).then(|| std::slice::from_raw_parts(job.bias, job.n)),
        )
    };
    for col in start..end {
        let w_row = &w_data[col * job.k..(col + 1) * job.k];
        let w_scale = w_scales[col];
        let bias_term = bias.map(|b| b[col]);
        for row in 0..job.m {
            let x_row = &x[row * job.k..(row + 1) * job.k];
            let acc = dot_w8a16(x_row, w_row);
            let value = acc * w_scale;
            // SAFETY: `col` is inside this partition's exclusive range and `row < m`, so this
            // address is written by no other partition for the duration of the dispatch.
            unsafe {
                *job.out.add(row * job.n + col) = bias_term.map_or(value, |b| value + b);
            }
        }
    }
}

/// Computes this worker's column stripe of an f32 dense linear.
fn run_f32_linear_partition(job: &F32LinearJob, worker: usize) {
    // Stripes are tile-aligned (for every dispatched ISA level) so every partition stays on the
    // packed register-tiled path; a ragged boundary would push one partition onto the scalar
    // remainder loop for no reason.
    let chunk = job
        .n
        .div_ceil(job.partitions)
        .next_multiple_of(crate::packed_gemm::STRIPE_COLUMNS);
    let start = (worker * chunk).min(job.n);
    let end = ((worker + 1) * chunk).min(job.n);
    if start >= end {
        return;
    }
    // SAFETY: module-docs facts 1-3. The pointers outlive the dispatch (the caller blocks until
    // every partition reports done), the reads are shared-immutable for its duration, and this
    // worker writes only columns in its own [start, end) stripe. `out` stays a raw pointer for the
    // same reason as the int8 job: several whole-buffer `&mut` would be UB even with disjoint
    // writes, because `rustc` marks `&mut` `noalias`.
    unsafe {
        let x = std::slice::from_raw_parts(job.x, job.m * job.k);
        let weight = std::slice::from_raw_parts(job.weight, job.n * job.k);
        let bias = (!job.bias.is_null()).then(|| std::slice::from_raw_parts(job.bias, job.n));
        crate::packed_gemm::linear_packed_range(
            x, weight, bias, job.m, job.k, job.n, start, end, job.out,
        );
    }
}

impl Team {
    /// Runs one f32 dense linear across the team, bit-identically to the serial packed kernel.
    ///
    /// # Panics
    ///
    /// Panics on shape mismatches, exactly as the serial kernel does.
    ///
    /// The argument count mirrors the serial kernel's signature exactly, which is the point: a
    /// caller swaps one call for the other with no reshaping, so any divergence would be a
    /// compile error rather than a silent behavioural difference.
    #[allow(clippy::too_many_arguments)]
    pub fn linear_f32(
        &self,
        x: &[f32],
        weight: &[f32],
        bias: Option<&[f32]>,
        m: usize,
        k: usize,
        n: usize,
        out: &mut [f32],
    ) {
        assert_eq!(x.len(), m * k, "x must be [m, k]");
        assert_eq!(weight.len(), n * k, "weight must be [n, k]");
        assert_eq!(out.len(), m * n, "out must be [m, n]");
        if let Some(bias) = bias {
            assert_eq!(bias.len(), n, "bias must be [n]");
        }
        let job = Job::F32Linear(F32LinearJob {
            x: x.as_ptr(),
            weight: weight.as_ptr(),
            bias: bias.map_or(std::ptr::null(), <[f32]>::as_ptr),
            out: out.as_mut_ptr(),
            m,
            k,
            n,
            partitions: self.partitions,
        });
        self.dispatch(job);
    }

    /// Runs one W8A8 linear across the team. Bit-identical to the serial path per element.
    ///
    /// # Panics
    ///
    /// Panics on shape mismatches, exactly as the serial kernel does.
    #[allow(clippy::too_many_arguments)]
    pub fn linear_q8(
        &self,
        x_q: &[i8],
        x_scales: &[f32],
        weight: &QuantizedMatrix,
        bias: Option<&[f32]>,
        m: usize,
        out: &mut [f32],
        tier: Int8Tier,
    ) {
        let (n, k) = (weight.n, weight.k);
        assert_eq!(x_q.len(), m * k, "x_q must be [m, k]");
        assert_eq!(x_scales.len(), m, "x_scales must be [m]");
        assert_eq!(out.len(), m * n, "out must be [m, n]");
        if let Some(bias) = bias {
            assert_eq!(bias.len(), n, "bias must be [n]");
        }

        let job = Job::Linear(LinearJob {
            x_q: x_q.as_ptr(),
            x_scales: x_scales.as_ptr(),
            w_data: weight.data.as_ptr(),
            w_scales: weight.scales.as_ptr(),
            bias: bias.map_or(std::ptr::null(), <[f32]>::as_ptr),
            out: out.as_mut_ptr(),
            m,
            n,
            k,
            tier,
            partitions: self.partitions,
        });

        self.dispatch(job);
    }

    /// Runs one W8A16 (weight-only) linear across the team. Bit-identical to the serial path
    /// per element — every output element is the same `dot_w8a16` reduction.
    ///
    /// # Panics
    ///
    /// Panics on shape mismatches, exactly as the serial kernel does.
    pub fn linear_w8a16(
        &self,
        x: &[f32],
        weight: &QuantizedMatrix,
        bias: Option<&[f32]>,
        m: usize,
        out: &mut [f32],
    ) {
        let (n, k) = (weight.n, weight.k);
        assert_eq!(x.len(), m * k, "x must be [m, k]");
        assert_eq!(out.len(), m * n, "out must be [m, n]");
        if let Some(bias) = bias {
            assert_eq!(bias.len(), n, "bias must be [n]");
        }
        let job = Job::W8A16Linear(W8A16LinearJob {
            x: x.as_ptr(),
            w_data: weight.data.as_ptr(),
            w_scales: weight.scales.as_ptr(),
            bias: bias.map_or(std::ptr::null(), <[f32]>::as_ptr),
            out: out.as_mut_ptr(),
            m,
            n,
            k,
            partitions: self.partitions,
        });
        self.dispatch(job);
    }

    /// Runs the default-arithmetic GQA attention across the team, partitioned over query
    /// heads. Bit-identical to the serial `f32ref::gqa_attention` (same extracted loop).
    ///
    /// # Panics
    ///
    /// Panics on shape mismatches, exactly as the serial reference does.
    #[allow(clippy::too_many_arguments)]
    pub fn gqa_attention(
        &self,
        queries: &[f32],
        keys: &[f32],
        values: &[f32],
        mask: &[f32],
        query_positions: usize,
        key_positions: usize,
        q_heads: usize,
        kv_heads: usize,
        head_dim: usize,
        out: &mut [f32],
    ) {
        assert!(
            kv_heads > 0 && q_heads.is_multiple_of(kv_heads),
            "GQA head geometry"
        );
        assert_eq!(
            queries.len(),
            query_positions * q_heads * head_dim,
            "queries shape"
        );
        assert_eq!(
            keys.len(),
            key_positions * kv_heads * head_dim,
            "keys shape"
        );
        assert_eq!(
            values.len(),
            key_positions * kv_heads * head_dim,
            "values shape"
        );
        assert_eq!(mask.len(), query_positions * key_positions, "mask shape");
        assert_eq!(out.len(), query_positions * q_heads * head_dim, "out shape");
        let job = Job::Attention(AttentionJob {
            queries: queries.as_ptr(),
            keys: keys.as_ptr(),
            values: values.as_ptr(),
            mask: mask.as_ptr(),
            query_positions,
            key_positions,
            q_heads,
            kv_heads,
            head_dim,
            out: out.as_mut_ptr(),
            partitions: self.partitions,
        });
        self.dispatch(job);
    }

    /// The shared dispatch/work/join cycle (module-docs facts 1-3).
    fn dispatch(&self, job: Job) {
        use std::sync::atomic::Ordering;
        // One dispatch at a time, held through the join (module-docs fact 3). Poison
        // tolerance: a prior caller's panic leaves no dispatch state behind — everything is
        // re-established below.
        let _gate = self
            .dispatch_gate
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        {
            let mut control = lock_control(self.shared);
            control.job = Some(job);
            control.panicked = false;
            // `pending` is in place before the epoch publishes the job, so no worker can report
            // done against a stale count.
            self.shared
                .pending
                .store(self.partitions - 1, Ordering::Relaxed);
            control.generation += 1;
            self.shared
                .epoch
                .store(control.generation, Ordering::Release);
            if control.sleepers > 0 {
                self.shared.go.notify_all();
            }
        }

        // The caller is partition 0: it works instead of idling. Its own panic must still
        // wait out the workers (they hold live pointers into the caller's slices), so the
        // join below runs before any unwind continues.
        let caller_outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            run_partition(&job, 0);
        }));

        let workers_done = || self.shared.pending.load(Ordering::Acquire) == 0;
        let mut control = if spin_until(self.shared, workers_done) {
            lock_control(self.shared)
        } else {
            let mut control = lock_control(self.shared);
            while !workers_done() {
                control = self
                    .shared
                    .done
                    .wait(control)
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
            }
            control
        };
        control.job = None;
        let worker_panicked = control.panicked;
        drop(control);

        if let Err(payload) = caller_outcome {
            std::panic::resume_unwind(payload);
        }
        assert!(
            !worker_panicked,
            "a team worker panicked during this dispatch; the output buffer is not fully written"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::int8::{linear_q8, linear_w8a16};

    /// One-shot fuse consumed by a worker of ONE SPECIFIC team.
    ///
    /// Targeted by the `Shared` block's address rather than a bare bool: the test binary
    /// runs concurrently, other tests (and, since the attention wiring, plain f32ref calls)
    /// dispatch on the default-armed GLOBAL team, and an untargeted fuse was consumed by
    /// whichever team's worker happened to run first — failing this test and panicking an
    /// innocent one.
    pub(super) static PANIC_INJECT_TARGET: std::sync::atomic::AtomicUsize =
        std::sync::atomic::AtomicUsize::new(0);

    /// Consumes the fuse iff it targets `shared`'s team.
    pub(super) fn panic_injected_for(shared: &Shared) -> bool {
        let target = std::ptr::from_ref(shared) as usize;
        PANIC_INJECT_TARGET
            .compare_exchange(
                target,
                0,
                std::sync::atomic::Ordering::SeqCst,
                std::sync::atomic::Ordering::SeqCst,
            )
            .is_ok()
    }

    #[test]
    fn a_panicking_worker_fails_the_dispatch_loudly_instead_of_hanging() {
        let team = test_team(3);
        let weight = matrix(64, 32, 5);
        let x_q = vec![1_i8; 32];
        let mut out = vec![0.0_f32; 64];
        PANIC_INJECT_TARGET.store(
            std::ptr::from_ref(team.shared) as usize,
            std::sync::atomic::Ordering::SeqCst,
        );
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            team.linear_q8(&x_q, &[1.0], &weight, None, 1, &mut out, Int8Tier::Scalar);
        }));
        assert!(
            outcome.is_err(),
            "a worker panic must surface at the caller, not hang or pass"
        );
        // And the team must still be usable afterwards.
        team.linear_q8(&x_q, &[1.0], &weight, None, 1, &mut out, Int8Tier::Scalar);
        assert!(out.iter().all(|value| value.is_finite()));
    }

    fn matrix(n: usize, k: usize, seed: u64) -> QuantizedMatrix {
        let mut state = seed;
        let data: Vec<i8> = (0..n * k)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1);
                (((state >> 33) % 255) as i32 - 127) as i8
            })
            .collect();
        let scales: Vec<f32> = (0..n).map(|row| 0.001 + (row % 7) as f32 * 0.01).collect();
        QuantizedMatrix { data, scales, n, k }
    }

    /// A directly constructed team, so the test controls the partition count regardless of the
    /// process environment.
    fn test_team(partitions: usize) -> Team {
        let shared: &'static Shared = Box::leak(Box::new(Shared::new(true)));
        for worker in 1..partitions {
            std::thread::spawn(move || worker_loop(shared, worker));
        }
        Team {
            shared,
            partitions,
            dispatch_gate: Mutex::new(()),
        }
    }

    #[test]
    fn default_team_sizes_never_oversubscribe_the_machine() {
        // Off Apple the engine leaves the codec a hardware thread and the codec takes only what
        // is left (capped at four); a serial team is one thread either way. Apple keeps the
        // measured six-way engine and a serial codec.
        for ceiling in 1..=64 {
            let engine = engine_partitions_default(ceiling);
            let codec = codec_partitions_default(ceiling, engine);
            assert!(engine >= 1 && codec >= 1, "ceiling {ceiling}");
            assert!(engine <= 6 && codec <= 4, "ceiling {ceiling}");
            if cfg!(target_vendor = "apple") {
                assert_eq!(codec, 1);
            } else {
                assert!(
                    engine + codec <= ceiling.max(2),
                    "ceiling {ceiling}: engine {engine} + codec {codec}"
                );
            }
        }
    }

    #[test]
    fn a_codec_thread_never_resolves_to_the_engine_team() {
        let engine = armed().map(std::ptr::from_ref);
        let codec = std::thread::spawn(|| {
            use_codec_team_on_this_thread();
            armed().map(|team| std::ptr::from_ref(team) as usize)
        })
        .join()
        .expect("codec probe thread");
        if let (Some(engine), Some(codec)) = (engine, codec) {
            assert_ne!(engine as usize, codec, "codec thread got the engine team");
        }
        // Reporting agrees with what the codec thread's armed() built: its width when armed,
        // one when the codec team is serial.
        match codec {
            Some(_) => assert!(codec_partitions() > 1),
            None => assert_eq!(codec_partitions(), 1),
        }
    }

    #[test]
    fn two_disjoint_teams_dispatching_concurrently_stay_bit_identical_to_serial() {
        // The engine/codec split's whole safety claim: two teams with their own gates and
        // workers run f32 GEMMs at the same time without disturbing each other's bits.
        let (m, k, n) = (24, 384, 160);
        let x = values_of(m * k, 11);
        let weight = values_of(n * k, 12);
        let bias = values_of(n, 13);
        let mut serial = vec![0.0_f32; m * n];
        crate::packed_gemm::linear_packed(&x, &weight, Some(&bias), m, k, n, &mut serial);
        let teams: Vec<&'static Team> = (0..2)
            .map(|_| &*Box::leak(Box::new(test_team(2))))
            .collect();
        std::thread::scope(|scope| {
            for team in teams {
                let (x, weight, bias, serial) = (&x, &weight, &bias, &serial);
                scope.spawn(move || {
                    for _ in 0..200 {
                        let mut out = vec![0.0_f32; m * n];
                        team.linear_f32(x, weight, Some(bias), m, k, n, &mut out);
                        assert!(
                            out.iter()
                                .zip(serial)
                                .all(|(a, b)| a.to_bits() == b.to_bits()),
                            "a concurrent team's GEMM diverged from serial"
                        );
                    }
                });
            }
        });
    }

    #[test]
    fn scoped_team_bypass_restores_state_after_a_caught_panic() {
        TEAM_BYPASS.with(|cell| cell.set(false));
        let outcome = std::panic::catch_unwind(|| {
            with_team_bypassed(|| {
                assert!(thread_bypassed());
                panic!("contained probe failure");
            });
        });
        assert!(outcome.is_err());
        assert!(!thread_bypassed());
    }

    #[test]
    fn every_partition_count_is_bit_identical_to_serial_at_model_shapes() {
        for &(m, n, k) in &[
            (1_usize, 2048_usize, 1024_usize),
            (1, 1024, 3072),
            (16, 3072, 1024),
            (2, 517, 129), // deliberately ragged: tail partitions and odd K
        ] {
            let weight = matrix(n, k, 42 ^ (n as u64) << 20);
            let x_q: Vec<i8> = (0..m * k).map(|i| ((i * 31 + 7) % 255) as i8).collect();
            let x_scales: Vec<f32> = (0..m).map(|row| 0.02 + row as f32 * 0.005).collect();
            let mut serial = vec![0.0_f32; m * n];
            linear_q8(
                &x_q,
                &x_scales,
                &weight,
                None,
                m,
                &mut serial,
                Int8Tier::Scalar,
            );
            for partitions in [2_usize, 3, 4, 8] {
                let team = test_team(partitions);
                let mut parallel = vec![0.0_f32; m * n];
                team.linear_q8(
                    &x_q,
                    &x_scales,
                    &weight,
                    None,
                    m,
                    &mut parallel,
                    Int8Tier::Scalar,
                );
                for (index, (a, b)) in serial.iter().zip(&parallel).enumerate() {
                    assert_eq!(
                        a.to_bits(),
                        b.to_bits(),
                        "partitions={partitions} m={m} n={n} k={k} element {index}"
                    );
                }
            }
        }
    }

    #[test]
    fn w8a16_partitioning_is_bit_identical_to_serial_at_model_shapes() {
        for &(m, n, k) in &[
            (1_usize, 2048_usize, 1024_usize),
            (1, 1024, 3072),
            (16, 3072, 1024),
            (2, 517, 129), // deliberately ragged: tail partitions and odd K
        ] {
            let weight = matrix(n, k, 97 ^ (n as u64) << 20);
            let x: Vec<f32> = (0..m * k)
                .map(|i| ((i * 37 + 11) % 255) as f32 / 64.0 - 1.5)
                .collect();
            let bias: Vec<f32> = (0..n).map(|col| (col % 13) as f32 * 0.25 - 1.0).collect();
            // A genuinely serial reference: the bypass keeps the entry point off the
            // process-wide team even though these shapes clear its fan-out threshold.
            let mut serial = vec![0.0_f32; m * n];
            with_team_bypassed(|| {
                linear_w8a16(&x, &weight, Some(&bias), m, &mut serial);
            });
            for partitions in [2_usize, 3, 4, 8] {
                let team = test_team(partitions);
                let mut parallel = vec![0.0_f32; m * n];
                team.linear_w8a16(&x, &weight, Some(&bias), m, &mut parallel);
                for (index, (a, b)) in serial.iter().zip(&parallel).enumerate() {
                    assert_eq!(
                        a.to_bits(),
                        b.to_bits(),
                        "w8a16 partitions={partitions} m={m} n={n} k={k} element {index}"
                    );
                }
            }
        }
    }

    #[test]
    fn thousands_of_mixed_dispatches_complete_without_deadlock() {
        // The many_utterances_without_deadlock policy at kernel scale: hammer one team with
        // mixed shapes; a hang here fails by test-harness timeout rather than passing silently.
        let team = test_team(4);
        let weight_a = matrix(256, 512, 7);
        let weight_b = matrix(96, 128, 11);
        let x_a: Vec<i8> = vec![3; 512];
        let x_b: Vec<i8> = vec![-5; 2 * 128];
        let x_c: Vec<f32> = vec![0.75; 512];
        let mut out_a = vec![0.0_f32; 256];
        let mut out_b = vec![0.0_f32; 2 * 96];
        let mut out_c = vec![0.0_f32; 256];
        for _ in 0..2_000 {
            team.linear_q8(
                &x_a,
                &[0.5],
                &weight_a,
                None,
                1,
                &mut out_a,
                Int8Tier::Scalar,
            );
            team.linear_q8(
                &x_b,
                &[0.5, 0.25],
                &weight_b,
                None,
                2,
                &mut out_b,
                Int8Tier::Scalar,
            );
            team.linear_w8a16(&x_c, &weight_a, None, 1, &mut out_c);
        }
        assert!(out_a.iter().all(|value| value.is_finite()));
        assert!(out_b.iter().all(|value| value.is_finite()));
        assert!(out_c.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn attention_partitioning_is_bit_identical_to_serial_at_talker_geometry() {
        // Talker decode shape: 16 query heads / 8 KV heads / head_dim 128, growing KV; plus a
        // prefill-like seq>1 case and a ragged 5-partition split of 16 heads.
        for &(query_positions, key_positions) in &[(1_usize, 37_usize), (4, 24)] {
            let (q_heads, kv_heads, head_dim) = (16_usize, 8_usize, 128_usize);
            let queries = values_of(query_positions * q_heads * head_dim, 21);
            let keys = values_of(key_positions * kv_heads * head_dim, 22);
            let values = values_of(key_positions * kv_heads * head_dim, 23);
            let mut mask = vec![0.0_f32; query_positions * key_positions];
            for (index, slot) in mask.iter_mut().enumerate() {
                if index % 11 == 3 {
                    *slot = f32::NEG_INFINITY;
                }
            }
            let mut serial = vec![0.0_f32; queries.len()];
            crate::f32ref::gqa_attention(
                &queries,
                &keys,
                &values,
                &mask,
                query_positions,
                key_positions,
                q_heads,
                kv_heads,
                head_dim,
                &mut serial,
            );
            for partitions in [2_usize, 5, 8] {
                let team = test_team(partitions);
                let mut parallel = vec![0.0_f32; queries.len()];
                team.gqa_attention(
                    &queries,
                    &keys,
                    &values,
                    &mask,
                    query_positions,
                    key_positions,
                    q_heads,
                    kv_heads,
                    head_dim,
                    &mut parallel,
                );
                assert_eq!(
                    serial.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    parallel.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    "partitions={partitions} qp={query_positions}"
                );
            }
        }
    }

    fn values_of(len: usize, seed: u64) -> Vec<f32> {
        let mut state = seed;
        (0..len)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1);
                ((state >> 33) as f32 / (1u64 << 31) as f32) - 0.5
            })
            .collect()
    }

    #[test]
    fn bias_reaches_every_partition() {
        let (m, n, k) = (2_usize, 130_usize, 64_usize);
        let weight = matrix(n, k, 99);
        let bias: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let x_q: Vec<i8> = vec![1; m * k];
        let x_scales = vec![1.0_f32; m];
        let mut serial = vec![0.0_f32; m * n];
        linear_q8(
            &x_q,
            &x_scales,
            &weight,
            Some(&bias),
            m,
            &mut serial,
            Int8Tier::Scalar,
        );
        let team = test_team(3);
        let mut parallel = vec![0.0_f32; m * n];
        team.linear_q8(
            &x_q,
            &x_scales,
            &weight,
            Some(&bias),
            m,
            &mut parallel,
            Int8Tier::Scalar,
        );
        assert_eq!(
            serial.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            parallel.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
    }
}
