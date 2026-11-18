//! Runs a fixed prompt greedily and prints the ids it generates, optionally with the experts restricted to a selection
//! (`{"layer": [expert ids]}`, as written by scripts/dev/moe-trace-stats.py). A full 128-expert model restricted to the experts a cut
//! kept must generate exactly what that cut generates:
//!
//!   check_mask CUT.gguf                       the cut, every expert it has
//!   check_mask FULL.gguf --select cal48.json  the full model, only the experts the cut kept
#[cfg(any(target_os = "macos", windows, target_os = "linux"))]
fn main() -> anyhow::Result<()> {
    use dafo_compiler_llm::{backend_metal, frontends, gguf::Gguf, runtime::Runtime};
    let mut arguments = std::env::args().skip(1);
    let model = arguments.next().expect("MODEL.gguf");
    let (mut select, mut pool, mut swap, mut serial, mut vram) = (None, None, None::<String>, false, None::<usize>);
    while let Some(flag) = arguments.next() {
        match flag.as_str() {
            "--select" => select = arguments.next(),
            "--swap" => swap = arguments.next(),
            "--serial" => serial = true,
            "--vram" => vram = arguments.next().map(|n| n.parse::<usize>().expect("vram experts")),
            "--pool" => pool = arguments.next().map(|n| n.parse::<usize>().expect("pool size")),
            other => panic!("unknown option {other}"),
        }
    }
    let gguf = std::sync::Arc::new(Gguf::open(&model)?);
    let mut graph = frontends::token_graph(&gguf)?;
    if let Some(size) = pool {
        backend_metal::use_expert_pool(&mut graph, size);
    }
    let plans = backend_metal::compile_plans_for(backend_metal::Target::native(), &graph, &gguf, 2048, false)?;
    let started = std::time::Instant::now();
    let mut runtime = Runtime::with_expert_tiers(&gguf, &plans, 1, pool.map(|size| match vram {
        Some(device) if device < size => (device, size - device),
        _ => (size, 0),
    }))?;
    println!("runtime ready in {:.1} s ({:.2} GiB of weights)", started.elapsed().as_secs_f64(), runtime.weight_bytes() as f64 / (1u64 << 30) as f64);
    if let Some(file) = select {
        let text = std::fs::read_to_string(&file)?;
        let value: serde_json::Value = serde_json::from_str(&text)?;
        let total = gguf.tensor("blk.0.ffn_gate_inp.weight").expect("router tensor").dims[1] as usize;
        let mut copied = 0;
        for (layer, ids) in value.as_object().expect("an object") {
            let ids: Vec<usize> = ids.as_array().expect("an array").iter().map(|id| id.as_u64().unwrap() as usize).collect();
            copied += runtime.set_allowed_experts(layer.parse()?, &ids, total)?;
        }
        println!("{copied} experts copied into the pool");
        println!("experts restricted by {file} ({total} per layer in the model)");
    }
    let tokenizer = dafo_compiler_llm::host::tokenizer::from_gguf(&gguf)?;
    let text = "Customer invoice 90044 is overdue by 25 days. Write a short, polite reminder email that mentions the amount due and the payment terms.";
    let prompt: Vec<u32> = tokenizer.encode(text, true).map_err(anyhow::Error::msg)?.get_ids().to_vec();
    runtime.select_slot(0);
    let mut next = if serial {
        // One token at a time, no batch plan.
        let mut next = 0;
        for (position, &token) in prompt.iter().enumerate() {
            next = runtime.step(token, position)?.0;
        }
        next
    } else {
        runtime.prefill(&prompt, 0)?
    };
    let mut position = prompt.len();
    let mut out = Vec::new();
    for _ in 0..24 {
        let (token, _) = runtime.step(next, position)?;
        out.push(token);
        if out.len() <= 3 {
            let logits = runtime.logits();
            let best = logits.iter().cloned().fold(f32::MIN, f32::max);
            println!("step {}: token {token}, largest logit {best:.4}, sum {:.2}", out.len(), logits.iter().sum::<f32>());
        }
        next = token;
        position += 1;
    }
    println!("{out:?}");
    if let Some((size, layers)) = runtime.pool_shape() {
        let total = gguf.tensor("blk.0.ffn_gate_inp.weight").expect("router tensor").dims[1] as usize;
        let demand = runtime.take_demand(layers, total);
        let (mut wanted, mut missed) = (0u64, 0u64);
        for (layer, row) in demand.iter().enumerate() {
            let resident: std::collections::HashSet<usize> = runtime.resident_experts(layer).into_iter().collect();
            for (expert, &count) in row.iter().enumerate() {
                wanted += count as u64;
                if !resident.contains(&expert) {
                    missed += count as u64;
                }
            }
        }
        println!("pool of {size}: the router wanted {wanted} (layer, expert) choices, {missed} of them outside the pool ({:.1} %)", 100.0 * missed as f64 / wanted.max(1) as f64);
    }
    if let Some(file) = swap {
        // Change the experts between two conversations and time it.
        let value: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&file)?)?;
        let total = gguf.tensor("blk.0.ffn_gate_inp.weight").expect("router tensor").dims[1] as usize;
        let started = std::time::Instant::now();
        let mut copied = 0;
        for (layer, ids) in value.as_object().expect("an object") {
            let ids: Vec<usize> = ids.as_array().expect("an array").iter().map(|id| id.as_u64().unwrap() as usize).collect();
            copied += runtime.set_allowed_experts(layer.parse()?, &ids, total)?;
        }
        println!("swapped to {file}: {copied} experts ({:.0} MB) copied in {:.2} s", copied as f64 * 3.72, started.elapsed().as_secs_f64());
        let mut next = runtime.prefill(&prompt, 0)?;
        let mut position = prompt.len();
        let mut text = Vec::new();
        for _ in 0..12 {
            let (token, _) = runtime.step(next, position)?;
            text.push(token);
            next = token;
            position += 1;
        }
        println!("after the swap: {text:?}");
    }
    Ok(())
}
#[cfg(not(any(target_os = "macos", windows, target_os = "linux")))]
fn main() {}
