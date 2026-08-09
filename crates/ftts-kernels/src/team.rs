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
struct Job {
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
    let chunk = job.n.div_ceil(job.partitions);
    let start = (worker * chunk).min(job.n);
    let end = ((worker + 1) * chunk).min(job.n);
    if start >= end {
        return;
    }
    // SAFETY: module-docs facts 1-3 — pointers outlive the dispatch, reads are shared-immutable,
    // and this worker writes only columns in its own [start, end) range.
    let (x_q, x_scales, w_data, w_scales, bias, out) = unsafe {
        (
            std::slice::from_raw_parts(job.x_q, job.m * job.k),
            std::slice::from_raw_parts(job.x_scales, job.m),
            std::slice::from_raw_parts(job.w_data, job.n * job.k),
            std::slice::from_raw_parts(job.w_scales, job.n),
            (!job.bias.is_null()).then(|| std::slice::from_raw_parts(job.bias, job.n)),
            std::slice::from_raw_parts_mut(job.out, job.m * job.n),
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
            out[row * job.n + col] = bias_term.map_or(value, |b| value + b);
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

        let job = Job {
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
        };

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
