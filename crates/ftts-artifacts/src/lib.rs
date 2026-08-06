#![forbid(unsafe_code)]

//! Safe readers and writers for FrankenTTS artifacts.

pub mod census;
pub mod fttsq;
pub mod safetensors;
pub mod sha256;

/// Identifies this crate's scaffold revision.
pub const SCAFFOLD_REVISION: u8 = 1;
