//! Compiler of language models for Metal: see the README and `docs/RESULTS.md`.

#[cfg(target_os = "macos")]
pub mod backend_metal;
pub mod frontends;
pub mod gguf;
pub mod ir;
#[cfg(target_os = "macos")]
pub mod metal;
pub mod quant;
#[cfg(target_os = "macos")]
pub mod pool;
#[cfg(target_os = "macos")]
pub mod runtime;
/// What a program needs around a compiled model to talk to it: tokenizer, chat template with tools, sampling, an engine thread.
#[cfg(feature = "host")]
pub mod host;
/// A native vision encoder for Gemma 4's projector file (needs the `vision` feature).
#[cfg(all(feature = "vision", target_os = "macos"))]
pub mod vision;
