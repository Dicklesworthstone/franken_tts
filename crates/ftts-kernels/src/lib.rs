#![deny(unsafe_op_in_unsafe_fn)]

//! The sole crate permitted to contain audited kernel `unsafe` islands.
//!
//! Every future unsafe kernel must be feature-gated, carry a `SAFETY:` comment,
//! and retain a bit-identical safe scalar fallback.
//!
//! ## Frankentorch boundary
//!
//! This Phase-0 scaffold intentionally has no frankentorch dependency: it does
//! not yet expose a kernel facade that consumes one. Add each dependency only
//! alongside a real facade API and its scalar fallback; do not predeclare a
//! machine-local or otherwise unused substrate dependency.
//!
//! # Permanent integer-kernel law
//!
//! A route cannot become dispatchable until [`selftest::run_selftest`] proves its all-extreme
//! reduction at every model-specific census binding row against an independent i64 oracle: the
//! U8S8 envelope (`255 * 127 * K`, the ceiling for a future unsigned-activation VNNI route) on
//! the checked scalar path, and the S8S8 kernel contract (`±127 * 127 * K`) through the real
//! [`int8::dot_i32`] on every available tier with scalar equality. Every native tier, Q4 unpack
//! path, codec convolution, verifier, and batched variant extends that same surface before it
//! may be selected.

pub mod enhance;
pub mod f32ref;
pub mod int4;
pub mod int8;
pub mod mmap;
pub mod packed_gemm;
pub mod route;
pub mod selftest;
pub mod sleef;
pub mod startup_env;
pub mod team;

/// Identifies this crate's scaffold revision.
pub const SCAFFOLD_REVISION: u8 = 1;

/// Per-thread allocation counting for zero-allocation verification tests
/// (frankentts-k-rcd-engine-6e3). Thread-local counters so parallel tests do
/// not pollute each other; a pass-through to [`System`] when inactive.
///
/// This lives in the one crate that hosts audited `unsafe` (the
/// `unsafe impl GlobalAlloc`); the rest of the workspace forbids `unsafe` and
/// consumes it via [`CountingAlloc::with_counting`].
pub mod test_alloc {
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::cell::Cell;

    thread_local! {
        static ACTIVE: Cell<bool> = const { Cell::new(false) };
        static COUNT: Cell<usize> = const { Cell::new(0) };
    }

    /// The counting pass-through allocator.
    pub struct CountingAlloc;

    impl CountingAlloc {
        /// Runs `f` with this thread's allocation counting enabled; returns
        /// the closure's value and the number of allocations it performed.
        pub fn with_counting<T>(f: impl FnOnce() -> T) -> (T, usize) {
            ACTIVE.with(|a| a.set(true));
            COUNT.with(|c| c.set(0));
            let out = f();
            ACTIVE.with(|a| a.set(false));
            let count = COUNT.with(Cell::get);
            (out, count)
        }
    }

    // SAFETY: pure delegation to the system allocator; the only extra work is
    // a thread-local counter increment when the canonical-parity tests have
    // counting enabled.
    unsafe impl GlobalAlloc for CountingAlloc {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            if ACTIVE.with(Cell::get) {
                COUNT.with(|c| c.set(c.get() + 1));
            }
            unsafe { System.alloc(layout) }
        }

        unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
            unsafe { System.dealloc(ptr, layout) }
        }

        unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
            if ACTIVE.with(Cell::get) {
                COUNT.with(|c| c.set(c.get() + 1));
            }
            unsafe { System.realloc(ptr, layout, new_size) }
        }
    }
}
