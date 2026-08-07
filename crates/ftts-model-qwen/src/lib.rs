#![forbid(unsafe_code)]

//! Safe Qwen3-TTS model orchestration.

pub mod prompt;
pub mod sampler;
pub mod talker;
pub mod tokenizer;

/// Identifies this crate's scaffold revision.
pub const SCAFFOLD_REVISION: u8 = 1;
