//! Compiler of language models for Metal: see the README and `docs/RESULTS.md`.

/// The plan builder (named for the first backend): graph -> dispatches, for Metal or CUDA kernels.
pub mod backend_metal;
pub mod frontends;
pub mod gguf;
pub mod ir;
#[cfg(target_os = "macos")]
pub mod metal;
pub mod quant;
#[cfg(any(target_os = "macos", windows, target_os = "linux"))]
pub mod pool;
#[cfg(any(target_os = "macos", windows, target_os = "linux"))]
pub mod runtime;
/// What a program needs around a compiled model to talk to it: tokenizer, chat template with tools, sampling, an engine thread.
#[cfg(feature = "host")]
pub mod host;
/// A native vision encoder for Gemma 4's projector file (needs the `vision` feature).
#[cfg(all(feature = "vision", any(target_os = "macos", windows, target_os = "linux")))]
pub mod vision;
/// The CUDA device layer (NVIDIA GPUs on Windows and Linux); the driver and NVRTC are loaded when the first device is made.
#[cfg(any(windows, target_os = "linux"))]
pub mod cuda;
