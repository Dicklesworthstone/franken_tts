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
//! U8S8 reduction at every model-specific census binding row against an independent i64 oracle.
//! The initial scalar route establishes the executable proof surface; every native tier, Q4
//! unpack path, codec convolution, verifier, and batched variant extends that same surface before
//! it may be selected.

pub mod mmap;
pub mod selftest;

/// Identifies this crate's scaffold revision.
pub const SCAFFOLD_REVISION: u8 = 1;
