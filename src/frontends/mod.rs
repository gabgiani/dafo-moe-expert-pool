//! One module per model architecture: reads the source file and emits the graph of a token.

pub mod gemma4;

use anyhow::{Result, bail};

use crate::gguf::{Gguf, Scalar};
use crate::ir::Graph;

/// Builds the graph of one token of the model in `gguf`, choosing the frontend by `general.architecture`.
pub fn token_graph(gguf: &Gguf) -> Result<Graph> {
    match gguf.metadata("general.architecture") {
        Some(Scalar::Text(architecture)) if architecture == "gemma4" => gemma4::token_graph(gguf),
        Some(Scalar::Text(architecture)) => bail!("no frontend for the architecture {architecture}"),
        _ => bail!("the file does not declare its architecture"),
    }
}
