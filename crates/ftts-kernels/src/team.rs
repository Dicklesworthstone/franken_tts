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
//!    `remaining` to zero, so the pointers outlive every access.
//! 2. **Aliasing**: workers write only `out[row * n + col]` for `col` inside their own disjoint
//!    column range; reads (`x_q`, scales, weights, bias) are shared and immutable for the whole
//!    dispatch because the caller holds the only `&mut` (to `out`) and blocks.
//! 3. **One parallel owner**: `dispatch_gate` serializes whole dispatches, so a second engine
//!    thread cannot overwrite the job while workers are mid-partition, and workers themselves
//!    never dispatch (their compute is a leaf loop).
//!
//! A stress test drives thousands of mixed-shape dispatches and a watchdog test bounds wall
//! time, per the `many_utterances_without_deadlock` policy.

use crate::int8::{Int8Tier, QuantizedMatrix, dot_i32};
use std::sync::{Condvar, Mutex, OnceLock};

/// One dispatched operation, shared read-only with every worker.
#[derive(Clone, Copy)]
enum Job {
    /// W8A8 linear partitioned over output columns.
    Linear(LinearJob),
    /// f32 GQA attention partitioned over query heads.
    Attention(AttentionJob),
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
    remaining: usize,
    /// Set when any partition panicked during the current dispatch, so the caller can
    /// propagate a loud failure instead of hanging on a worker that will never report done.
    panicked: bool,
}

struct Shared {
    control: Mutex<Control>,
    go: Condvar,
    done: Condvar,
}

/// The process-wide team. Exists only when `FTTS_INT8_THREADS` requests more than one thread.
pub struct Team {
    shared: &'static Shared,
    /// Total partitions per dispatch: spawned workers + the calling thread.
    partitions: usize,
    dispatch_gate: Mutex<()>,
}

thread_local! {
    static TEAM_BYPASS: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
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

/// Whether the current thread opted out of team dispatch.
#[must_use]
pub fn thread_bypassed() -> bool {
    TEAM_BYPASS.with(std::cell::Cell::get)
}

/// The team for this process, if parallel execution is enabled.
///
/// `FTTS_INT8_THREADS` sets the total partition count (caller included); `1` or unset means
/// serial (no threads spawned, no team). Values are clamped to the machine's available
/// parallelism. Read once.
pub fn armed() -> Option<&'static Team> {
    // wasm32 cannot spawn its own threads: `wasm32-unknown-unknown` has no `std::thread::spawn`,
    // because only the host can create the Workers that share this module's linear memory. So the
    // team is *installed* from JS once its Workers are up (see `install_wasm_team`) instead of
    // being created on first use, and stays `None` until then — which is also the correct answer
    // for any browser without `SharedArrayBuffer`.
    #[cfg(target_arch = "wasm32")]
    {
        WASM_TEAM.get().and_then(Option::as_ref)
    }
    #[cfg(not(target_arch = "wasm32"))]
    armed_native()
}

/// The team installed by the host, once its Workers exist.
#[cfg(target_arch = "wasm32")]
static WASM_TEAM: OnceLock<Option<Team>> = OnceLock::new();

/// The shared control block Workers park on, published before any of them starts.
#[cfg(target_arch = "wasm32")]
static WASM_SHARED: OnceLock<&'static Shared> = OnceLock::new();

/// Publishes the shared control block and arms a `partitions`-way team.
///
/// Call once, from the thread that will own dispatch, before spawning `partitions - 1` Workers
/// that each call [`wasm_worker_loop`]. Returns the number of Workers the caller must start.
///
/// # Why this shape
///
/// The caller is partition 0 and blocks on a condvar until the others report done. `atomic.wait`
/// **traps on a browser's main thread**, so the owning thread must itself be a Worker — in this
/// project, the engine Worker that already runs synthesis. Arming from the main thread would not
/// merely be slow, it would abort.
///
/// Returns 0 (and arms nothing) when `partitions <= 1`, which is the serial fallback.
#[cfg(target_arch = "wasm32")]
pub fn install_wasm_team(partitions: usize) -> usize {
    if partitions <= 1 {
        let _ = WASM_TEAM.set(None);
        return 0;
    }
    let shared: &'static Shared = *WASM_SHARED.get_or_init(|| {
        Box::leak(Box::new(Shared {
            control: Mutex::new(Control {
                generation: 0,
                job: None,
                remaining: 0,
                panicked: false,
            }),
            go: Condvar::new(),
            done: Condvar::new(),
        }))
    });
    let _ = WASM_TEAM.set(Some(Team {
        shared,
        partitions,
        dispatch_gate: Mutex::new(()),
    }));
    partitions - 1
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

#[cfg(not(target_arch = "wasm32"))]
fn armed_native() -> Option<&'static Team> {
    static TEAM: OnceLock<Option<Team>> = OnceLock::new();
    TEAM.get_or_init(|| {
        let ceiling = std::thread::available_parallelism().map_or(1, usize::from);
        // Default six ways: the measured knee on M4 Pro (memory-bound beyond it). Partitioning
        // never changes output bits, so the default applies everywhere, reference route included.
        let requested: usize = std::env::var("FTTS_INT8_THREADS")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(6);
        let partitions = requested.min(ceiling);
        if partitions <= 1 {
            return None;
        }
        let shared: &'static Shared = Box::leak(Box::new(Shared {
            control: Mutex::new(Control {
                generation: 0,
                job: None,
                remaining: 0,
                panicked: false,
            }),
            go: Condvar::new(),
            done: Condvar::new(),
        }));
        // Workers 1..partitions; the caller is partition 0. Threads live for the process and
        // park on the condvar between dispatches, so leaking their handles is deliberate.
        for worker in 1..partitions {
            std::thread::Builder::new()
                .name(format!("ftts-int8-{worker}"))
                .spawn(move || worker_loop(shared, worker))
                .expect("spawn int8 worker");
        }
        Some(Team {
            shared,
            partitions,
            dispatch_gate: Mutex::new(()),
        })
    })
    .as_ref()
}

fn worker_loop(shared: &'static Shared, worker: usize) {
    let mut seen = 0_u64;
    loop {
        let job = {
            let mut control = lock_control(shared);
            while control.generation == seen {
                control = shared
                    .go
                    .wait(control)
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
            }
            seen = control.generation;
            control.job.expect("generation bumped without a job")
        };
        // A panicking partition must still report done, or the caller hangs forever waiting
        // for a decrement that will never come. The panic is recorded and re-raised loudly on
        // the caller's thread instead.
        let outcome = std::panic::catch_unwind(|| run_partition(&job, worker));
        let mut control = lock_control(shared);
        if outcome.is_err() {
            control.panicked = true;
        }
        control.remaining -= 1;
        if control.remaining == 0 {
            shared.done.notify_all();
        }
    }
}

/// Locks team control, tolerating poison: every dispatch re-establishes the full invariant
/// (job, generation, remaining) from scratch, so a lock poisoned by an earlier panic carries
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
    #[cfg(test)]
    if worker > 0 && tests::PANIC_INJECT.swap(false, std::sync::atomic::Ordering::SeqCst) {
        panic!("injected worker panic for the hang-hardening test");
    }
    match job {
        Job::Linear(job) => run_linear_partition(job, worker),
        Job::Attention(job) => run_attention_partition(job, worker),
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
    for col in start..end {
        let w_row = &w_data[col * job.k..(col + 1) * job.k];
        let w_scale = w_scales[col];
        let bias_term = bias.map(|b| b[col]);
        for row in 0..job.m {
            let x_row = &x_q[row * job.k..(row + 1) * job.k];
            let acc = dot_i32(x_row, w_row, job.tier);
            let value = acc as f32 * (x_scales[row] * w_scale);
            // SAFETY: `col` is inside this partition's exclusive range and `row < m`, so this
            // address is written by no other partition for the duration of the dispatch.
            unsafe {
                *job.out.add(row * job.n + col) = bias_term.map_or(value, |b| value + b);
            }
        }
    }
}

impl Team {
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
            control.generation += 1;
            control.remaining = self.partitions - 1;
            control.panicked = false;
            self.shared.go.notify_all();
        }

        // The caller is partition 0: it works instead of idling. Its own panic must still
        // wait out the workers (they hold live pointers into the caller's slices), so the
        // join below runs before any unwind continues.
        let caller_outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            run_partition(&job, 0);
        }));

        let mut control = lock_control(self.shared);
        while control.remaining > 0 {
            control = self
                .shared
                .done
                .wait(control)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
        }
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
    use crate::int8::linear_q8;

    /// One-shot fuse consumed by [`run_partition`] on a worker thread.
    pub(super) static PANIC_INJECT: std::sync::atomic::AtomicBool =
        std::sync::atomic::AtomicBool::new(false);

    #[test]
    fn a_panicking_worker_fails_the_dispatch_loudly_instead_of_hanging() {
        let team = test_team(3);
        let weight = matrix(64, 32, 5);
        let x_q = vec![1_i8; 32];
        let mut out = vec![0.0_f32; 64];
        PANIC_INJECT.store(true, std::sync::atomic::Ordering::SeqCst);
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
        let shared: &'static Shared = Box::leak(Box::new(Shared {
            control: Mutex::new(Control {
                generation: 0,
                job: None,
                remaining: 0,
                panicked: false,
            }),
            go: Condvar::new(),
            done: Condvar::new(),
        }));
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
    fn thousands_of_mixed_dispatches_complete_without_deadlock() {
        // The many_utterances_without_deadlock policy at kernel scale: hammer one team with
        // mixed shapes; a hang here fails by test-harness timeout rather than passing silently.
        let team = test_team(4);
        let weight_a = matrix(256, 512, 7);
        let weight_b = matrix(96, 128, 11);
        let x_a: Vec<i8> = vec![3; 512];
        let x_b: Vec<i8> = vec![-5; 2 * 128];
        let mut out_a = vec![0.0_f32; 256];
        let mut out_b = vec![0.0_f32; 2 * 96];
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
        }
        assert!(out_a.iter().all(|value| value.is_finite()));
        assert!(out_b.iter().all(|value| value.is_finite()));
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
