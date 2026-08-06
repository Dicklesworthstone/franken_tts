#![forbid(unsafe_code)]

//! Safe readers and writers for FrankenTTS artifacts.

pub mod census;
pub mod safetensors;

/// Identifies this crate's scaffold revision.
pub const SCAFFOLD_REVISION: u8 = 1;
