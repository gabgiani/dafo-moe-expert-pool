//! `dafo-compile`: look at a model, compile it for this Mac's GPU and measure the result, without any host application.
//!
//!   dafo-compile list                                    the models you can pull (like `ollama list`)
//!   dafo-compile pull NAME                               download a model and its projector, joining the chunks (like `ollama pull`)
//!   dafo-compile rm NAME                                 delete a pulled model
//!   dafo-compile inspect MODEL.gguf                      what the model is, what a token reads, what the caches cost
//!   dafo-compile compile MODEL.gguf [--context N] [--out DIR [--copy-weights]] [--projector MMPROJ.gguf]
//!                                                        build the plans for this machine and report them; with --out, leave
//!                                                        the compiled model in a folder that `dafo-serve DIR` serves
//!                                                        (needs a build with `--features host`); with --projector the folder also
//!                                                        holds the vision projector and the plan that reads images (`--features vision`
//!                                                        to serve them)
//!   dafo-compile bench   MODEL.gguf [--context N] [--pool N]
//!                                                        generation and prompt-reading speed on this machine; with --pool N the
//!                                                        experts are served from a pool of N per layer (the model on disk may have more)
//!
//! `bench` feeds the model synthetic token ids: it measures speed, not text quality. To talk to a model (tokenizer, chat template,
//! sampling, an HTTP API) use a host such as DAFO Swarm, which embeds this crate; see the README.

use anyhow::{Context, Result, bail};
use dafo_compiler_llm::ir::{AttentionKind, Op};
use dafo_compiler_llm::{frontends, gguf::Gguf};

fn main() -> Result<()> {
    let mut arguments = std::env::args().skip(1);
    let command = arguments.next().unwrap_or_default();
    // The commands about the catalog take a model *name*, not a file.
    #[cfg(feature = "pull")]
    match command.as_str() {
        "list" | "ls" => return dafo_compiler_llm::models::print_list(),
        "pull" => {
            let name = arguments.next().with_context(|| "usage: dafo-compile pull NAME   (see: dafo-compile list)")?;
            return dafo_compiler_llm::models::pull(&name).map(|_| ());
        }
        "rm" => return dafo_compiler_llm::models::remove(&arguments.next().with_context(|| "usage: dafo-compile rm NAME")?),
        _ => {}
    }
    let model = arguments.next().with_context(|| usage())?;
    let rest: Vec<String> = arguments.collect();
    let context = flag(&rest, "--context")?.unwrap_or(8192);
    match command.as_str() {
        "inspect" => inspect(&model, context),
        "compile" => compile(&model, context, flag_text(&rest, "--out")?, rest.iter().any(|argument| argument == "--copy-weights"), flag_text(&rest, "--projector")?),
        "bench" => bench(&model, context, flag(&rest, "--pool")?, flag(&rest, "--vram-experts")?),
        _ => bail!("{}", usage()),
    }
}

fn usage() -> String {
    "usage: dafo-compile inspect|compile|bench MODEL.gguf [--context TOKENS]\n       dafo-compile list | pull NAME | rm NAME      (models of the DAFO catalog; needs a build with --features host)".to_owned()
}

fn flag(arguments: &[String], name: &str) -> Result<Option<usize>> {
    match arguments.iter().position(|argument| argument == name) {
        Some(position) => Ok(Some(arguments.get(position + 1).with_context(|| format!("{name} needs a value"))?.parse()?)),
        None => Ok(None),
    }
}

fn flag_text(arguments: &[String], name: &str) -> Result<Option<String>> {
    match arguments.iter().position(|argument| argument == name) {
        Some(position) => Ok(Some(arguments.get(position + 1).with_context(|| format!("{name} needs a value"))?.clone())),
        None => Ok(None),
    }
}

fn gib(bytes: usize) -> f64 {
    bytes as f64 / (1u64 << 30) as f64
}

fn inspect(model: &str, context: usize) -> Result<()> {
    let gguf = Gguf::open(model)?;
    let tensors = gguf.tensors().count();
    let weight_bytes: usize = gguf.tensors().map(|(_, tensor)| tensor.bytes()).sum();
    println!("{model}");
    println!("  {tensors} tensors, {:.2} GiB of weights", gib(weight_bytes));
    let graph = frontends::token_graph(&gguf)?;
    let mut counts = std::collections::BTreeMap::new();
    for op in &graph.ops {
        let name = format!("{op:?}");
        *counts.entry(name.split([' ', '{']).next().unwrap_or("").to_string()).or_insert(0usize) += 1;
    }
    println!("  the graph of one token: {} operations, {} values", graph.ops.len(), graph.values());
    for (name, count) in counts {
        println!("    {name:<16} {count}");
    }
    let sliding = graph.ops.iter().filter(|op| matches!(op, Op::Attention { kind: AttentionKind::Sliding { .. }, .. })).count();
    let attention = graph.ops.iter().filter(|op| matches!(op, Op::Attention { .. })).count();
    println!("  attention layers: {sliding} sliding-window, {} global", attention - sliding);
    let traffic = graph.bytes_per_token();
    println!(
        "  a token reads {:.2} GiB (dense {:.2}, experts {:.2}, output head {:.2}, embedding row {:.4}, norms and router {:.4})",
        gib(traffic.total()),
        gib(traffic.dense),
        gib(traffic.experts),
        gib(traffic.head),
        gib(traffic.embedding),
        gib(traffic.small)
    );
    println!("  generation is memory-bound: at 100 GB/s that is at most {:.1} tokens/s", 100e9 / traffic.total() as f64);
    {
        let (fixed, per_token) = dafo_compiler_llm::backend_metal::cache_footprint(&graph);
        println!(
            "  keys and values of one conversation: {:.0} MiB fixed (sliding-window rings) + {:.1} KiB per token of context ({:.0} MiB at {context} tokens)",
            fixed as f64 / 1048576.0,
            per_token as f64 / 1024.0,
            (fixed + per_token * context) as f64 / 1048576.0
        );
    }
    Ok(())
}

#[cfg(any(target_os = "macos", windows, target_os = "linux"))]
fn compile(model: &str, context: usize, out: Option<String>, copy_weights: bool, projector: Option<String>) -> Result<()> {
    use dafo_compiler_llm::backend_metal;
    let gguf = Gguf::open(model)?;
    let graph = frontends::token_graph(&gguf)?;
    let started = std::time::Instant::now();
    let plans = backend_metal::compile_plans_for(backend_metal::Target::native(), &graph, &gguf, context, projector.is_some())?;
    println!("compiled in {:.2} s for a context of {context} tokens:", started.elapsed().as_secs_f64());
    for plan in &plans {
        println!("  batch {:>4}: {} dispatches, {} scratch buffers", plan.batch, plan.steps.len(), plan.scratch.len());
    }
    println!("  kernels: {} KiB of Metal source", plans[0].source.len() / 1024);
    let accesses = backend_metal::kernel_buffer_access(&plans[0].source);
    println!("  access analysis: {} kernels, read from their signatures (decides which dispatches may overlap)", accesses.len());
    if let Some(directory) = out {
        #[cfg(feature = "host")]
        {
            dafo_compiler_llm::host::artifact::write(std::path::Path::new(&directory), std::path::Path::new(model), &gguf, &plans, context, copy_weights, projector.as_deref().map(std::path::Path::new))?;
            println!("left the compiled model in {directory}/ (manifest.json, model.gguf, tokenizer.json, kernels.metal); serve it with: dafo-serve {directory}");
        }
        #[cfg(not(feature = "host"))]
        {
            let _ = (directory, copy_weights, &projector);
            bail!("--out needs a build with the host: cargo build --release --features host");
        }
    }
    Ok(())
}

#[cfg(any(target_os = "macos", windows, target_os = "linux"))]
fn bench(model: &str, context: usize, pool: Option<usize>, device: Option<usize>) -> Result<()> {
    use dafo_compiler_llm::{backend_metal, runtime::Runtime};
    use std::time::Instant;
    let gguf = std::sync::Arc::new(Gguf::open(model)?);
    let mut graph = frontends::token_graph(&gguf)?;
    if let Some(size) = pool {
        backend_metal::use_expert_pool(&mut graph, size);
    }
    let plans = backend_metal::compile_plans_for(backend_metal::Target::native(), &graph, &gguf, context, false)?;
    let mut runtime = Runtime::with_expert_tiers(&gguf, &plans, 1, pool.map(|size| match device {
        Some(device) if device < size => (device, size - device),
        _ => (size, 0),
    }))?;
    println!("{} on {}: {:.2} GiB of weights mapped, {} dispatches per token", model, runtime.device_name(), gib(runtime.weight_bytes()), runtime.dispatch_count());
    // Prompt reading: batched, on a prompt of varied ids.
    let prompt: Vec<u32> = (0..512u32).map(|index| 1000 + (index * 7919) % 20000).collect();
    runtime.prefill(&prompt[..128], 0)?; // warm-up: first use of every kernel
    let started = Instant::now();
    let mut next = runtime.prefill(&prompt, 128)?;
    let seconds = started.elapsed().as_secs_f64();
    println!("reading a prompt:  {:.0} tokens/s ({} tokens in {seconds:.2} s)", prompt.len() as f64 / seconds, prompt.len());
    // Generation: each token fed back, as a host would.
    let tokens = 64;
    let mut position = 128 + prompt.len();
    let started = Instant::now();
    for _ in 0..tokens {
        next = runtime.step(next, position)?.0;
        position += 1;
    }
    let seconds = started.elapsed().as_secs_f64();
    println!("generating:        {:.1} tokens/s ({:.1} ms per token)", tokens as f64 / seconds, seconds * 1000.0 / tokens as f64);
    Ok(())
}

#[cfg(not(any(target_os = "macos", windows, target_os = "linux")))]
fn compile(_: &str, _: usize, _: Option<String>, _: bool, _: Option<String>) -> Result<()> {
    bail!("compiling needs a Mac: the first backend targets Metal (inspect works anywhere)")
}

#[cfg(not(any(target_os = "macos", windows, target_os = "linux")))]
fn bench(_: &str, _: usize, _: Option<usize>, _: Option<usize>) -> Result<()> {
    bail!("benchmarking needs a Mac: the first backend targets Metal (inspect works anywhere)")
}
