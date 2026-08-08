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

pub mod f32ref;
pub mod int8;
pub mod mmap;
pub mod selftest;
pub mod sleef;

/// Identifies this crate's scaffold revision.
pub const SCAFFOLD_REVISION: u8 = 1;
