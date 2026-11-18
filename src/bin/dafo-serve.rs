//! `dafo-serve`: serves a compiled model over an OpenAI-compatible HTTP API, with a bearer token and a chat page.
//!
//!   dafo-serve MODEL.dafo            a folder made by `dafo-compile compile MODEL.gguf --out MODEL.dafo`
//!   dafo-serve MODEL.gguf            compiles at start, no folder
//!
//! Options: --host 127.0.0.1  --port 8080  --token TOKEN  --context TOKENS  --slots N
//!          --pool N [--profile FILE]   keep N experts per layer in memory out of the model's (the full model on disk), those of FILE
//!
//! GET  /                      the chat page (no token needed to load it; the page asks for the token)
//! GET  /health                200 when the model is loaded
//! GET  /v1/models             the model (needs the token)
//! POST /v1/chat/completions   messages, tools, streaming (needs the token)
//!
//! Without --token (or DAFO_SERVE_TOKEN) a random token is made and printed at start.

#[cfg(not(any(target_os = "macos", windows, target_os = "linux")))]
fn main() {
    eprintln!("dafo-serve needs a Mac (Metal) or an NVIDIA GPU (CUDA)");
    std::process::exit(1);
}

#[cfg(any(target_os = "macos", windows, target_os = "linux"))]
fn main() {
    if let Err(error) = dafo_compiler_llm::host::server::run(std::env::args().skip(1).collect()) {
        eprintln!("dafo-serve: {error:#}");
        std::process::exit(1);
    }
}
