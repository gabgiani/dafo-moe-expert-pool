//! Time of each kernel of the plan run alone after a few real tokens: where a token's time goes and how close each kernel is to
//! the memory bandwidth.
//!
//!   cargo run --release --example profile_kernels -- MODEL.gguf [TOKEN_IDS]

use anyhow::{Context, Result};
use dafo_compiler_llm::{backend_metal, frontends, gguf::Gguf, runtime::Runtime};

fn main() -> Result<()> {
    let mut arguments = std::env::args().skip(1);
    let model = arguments.next().context("usage: profile_kernels MODEL.gguf [TOKENS]")?;
    let tokens: Vec<u32> = match arguments.next() {
        Some(list) => list.split(',').map(|value| value.parse()).collect::<Result<_, _>>()?,
        None => vec![2, 105, 2364, 107, 5279, 529, 7001, 563],
    };
    let gguf = std::sync::Arc::new(Gguf::open(&model)?);
    let graph = frontends::token_graph(&gguf)?;
    let plan = backend_metal::compile(&graph, &gguf, 512, 1, true)?;
    let mut runtime = Runtime::new(&gguf, &[plan.clone()], 1)?;
    for (position, &token) in tokens.iter().enumerate() {
        runtime.step(token, position)?;
    }
    // Bytes each kernel reads per token, by the graph.
    let mut bytes = std::collections::HashMap::<&str, usize>::new();
    for step in &plan.steps {
        let _ = step;
    }
    let traffic = graph.bytes_per_token();
    bytes.insert("mv_q8_0", traffic.dense + traffic.head);
    bytes.insert("mv_id_q4_k", graph.ops.iter().map(|op| match op { dafo_compiler_llm::ir::Op::ExpertsLinear { weight, used, total, .. } if weight.encoding == dafo_compiler_llm::gguf::Encoding::Q4K => weight.bytes / total * used, _ => 0 }).sum());
    bytes.insert("mv_id_q5_1", graph.ops.iter().map(|op| match op { dafo_compiler_llm::ir::Op::ExpertsLinear { weight, used, total, .. } if weight.encoding == dafo_compiler_llm::gguf::Encoding::Q5_1 => weight.bytes / total * used, _ => 0 }).sum());
    bytes.insert("mv_id_q8_0", graph.ops.iter().map(|op| match op { dafo_compiler_llm::ir::Op::ExpertsLinear { weight, used, total, .. } if weight.encoding == dafo_compiler_llm::gguf::Encoding::Q8_0 => weight.bytes / total * used, _ => 0 }).sum());
    let mut rows = runtime.profile_by_kernel(1, 9)?;
    rows.sort_by(|a, b| b.2.total_cmp(&a.2));
    let total: f64 = rows.iter().map(|row| row.2).sum();
    println!("{:<16} {:>9} {:>10} {:>7} {:>9}", "kernel", "dispatches", "ms/token", "share", "GB/s");
    for (name, count, seconds) in &rows {
        let rate = bytes.get(name).map(|bytes| format!("{:.0}", *bytes as f64 / seconds / 1e9)).unwrap_or_default();
        println!("{name:<16} {count:>9} {:>10.2} {:>6.0}% {rate:>9}", seconds * 1000.0, seconds / total * 100.0);
    }
    println!("sum of the kernels run alone: {:.1} ms", total * 1000.0);
    let started = std::time::Instant::now();
    let mut gpu = 0.0;
    for position in tokens.len()..tokens.len() + 20 {
        gpu += runtime.step(563, position)?.1.gpu;
    }
    println!("whole token: {:.1} ms wall, {:.1} ms GPU", started.elapsed().as_secs_f64() * 50.0, gpu * 50.0);
    Ok(())
}
