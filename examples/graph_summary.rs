//! Builds the graph of one token of a model and prints its shape and the bytes of weights a token reads.
//!
//!   cargo run --release --example graph_summary -- MODEL.gguf

use anyhow::{Context, Result};
use dafo_compiler_llm::{frontends, gguf::Gguf, ir::Op};

fn main() -> Result<()> {
    let path = std::env::args().nth(1).context("usage: graph_summary MODEL.gguf")?;
    let gguf = Gguf::open(&path)?;
    let graph = frontends::token_graph(&gguf)?;
    let mut counts = std::collections::BTreeMap::new();
    for op in &graph.ops {
        let name = format!("{op:?}");
        *counts.entry(name.split([' ', '{']).next().unwrap_or("").to_string()).or_insert(0usize) += 1;
    }
    println!("{} operations, {} values", graph.ops.len(), graph.values());
    for (name, count) in counts {
        println!("  {name:<16} {count}");
    }
    let traffic = graph.bytes_per_token();
    let gib = |bytes: usize| bytes as f64 / (1u64 << 30) as f64;
    println!(
        "bytes read per token: {:.3} GiB (dense {:.3}, experts {:.3}, head {:.3}, embedding row {:.4}, norms and router {:.4})",
        gib(traffic.total()),
        gib(traffic.dense),
        gib(traffic.experts),
        gib(traffic.head),
        gib(traffic.embedding),
        gib(traffic.small)
    );
    let sliding = graph.ops.iter().filter(|op| matches!(op, Op::Attention { kind: dafo_compiler_llm::ir::AttentionKind::Sliding { .. }, .. })).count();
    println!("attention layers: {sliding} sliding, {} global", graph.ops.iter().filter(|op| matches!(op, Op::Attention { .. })).count() - sliding);
    Ok(())
}
