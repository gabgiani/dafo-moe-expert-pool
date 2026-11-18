//! The host of a compiled model: everything around the arithmetic that a program needs to hold a conversation with it.
//!
//! The compiler and the runtime know no text. This module adds what Gemma 4 needs: a tokenizer built from the GGUF, the chat
//! template (with tools), the sampler, and an engine thread that keeps several conversations in cache and reuses what they share.

pub mod artifact;
pub mod chat;
#[cfg(any(target_os = "macos", windows, target_os = "linux"))]
pub mod engine;
pub mod sampler;
pub mod tokenizer;
