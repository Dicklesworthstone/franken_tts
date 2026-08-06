#![deny(unsafe_op_in_unsafe_fn)]

//! The sole crate permitted to contain audited kernel `unsafe` islands.
//!
//! Every future unsafe kernel must be feature-gated, carry a `SAFETY:` comment,
//! and retain a bit-identical safe scalar fallback.

/// Identifies this crate's scaffold revision.
pub const SCAFFOLD_REVISION: u8 = 1;
