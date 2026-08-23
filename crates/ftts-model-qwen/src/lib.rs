#![forbid(unsafe_code)]
#![feature(float_erf)]

//! Safe Qwen3-TTS model orchestration.

pub mod checkpoint;
pub mod codec;
pub mod generate;
pub mod microdecoder;
pub mod prompt;
pub mod sampler;
pub mod speaker;
pub mod speech_encoder;
pub mod talker;
pub mod tokenizer;

/// Identifies this crate's scaffold revision.
pub const SCAFFOLD_REVISION: u8 = 1;
