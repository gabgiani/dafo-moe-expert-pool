//! Time of each kernel of the 128-token plan, run alone after a real batch: where a prompt's time goes.
//!
//!   cargo run --release --example profile_batch -- MODEL.gguf

use anyhow::{Context, Result};
use dafo_compiler_llm::{backend_metal, frontends, gguf::Gguf, runtime::Runtime};

fn main() -> Result<()> {
    let mut arguments = std::env::args().skip(1);
    let model = arguments.next().context("usage: profile_batch MODEL.gguf [BATCH]")?;
    let batch: usize = arguments.next().map(|value| value.parse()).transpose()?.unwrap_or(128);
    let gguf = std::sync::Arc::new(Gguf::open(&model)?);
    let graph = frontends::token_graph(&gguf)?;
    let plans = vec![backend_metal::compile(&graph, &gguf, 2048, 1, true)?, backend_metal::compile(&graph, &gguf, 2048, batch, true)?];
    let mut runtime = Runtime::new(&gguf, &plans, 1)?;
    let tokens: Vec<u32> = (0..(3 * batch) as u32).map(|index| 1000 + (index * 7919) % 20000).collect();
    for round in 0..3 {
        runtime.run_batch_for_test(batch, &tokens[round * batch..(round + 1) * batch], round * batch)?;
    }
    let mut rows = runtime.profile_by_kernel(batch, 5)?;
    rows.sort_by(|a, b| b.2.total_cmp(&a.2));
    let total: f64 = rows.iter().map(|row| row.2).sum();
    println!("{:<22} {:>10} {:>12} {:>7}", "kernel", "dispatches", "ms per batch", "share");
    for (name, count, seconds) in &rows {
        println!("{name:<22} {count:>10} {:>12.2} {:>6.0}%", seconds * 1000.0, seconds / total * 100.0);
    }
    println!("sum {:.1} ms for {batch} tokens = {:.0} tokens/s", total * 1000.0, batch as f64 / total);
    Ok(())
}
