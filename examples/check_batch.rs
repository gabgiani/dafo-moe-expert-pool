//! Compares a batch plan with the one-token plan operation by operation on real weights: runs N tokens in one batch, then the same
//! tokens one at a time, and compares each layer's output at every position that the single runs keep. Locates where a batch
//! drifts from the token-by-token result.
//!
//!   cargo run --release --example check_batch -- MODEL.gguf [N]

use anyhow::{Context, Result};
use dafo_compiler_llm::ir::Op;
use dafo_compiler_llm::{backend_metal, frontends, gguf::Gguf, runtime::Runtime};

fn main() -> Result<()> {
    let mut arguments = std::env::args().skip(1);
    let model = arguments.next().context("usage: check_batch MODEL.gguf [N]")?;
    let n: usize = arguments.next().map(|value| value.parse()).transpose()?.unwrap_or(16);
    let gguf = std::sync::Arc::new(Gguf::open(&model)?);
    let graph = frontends::token_graph(&gguf)?;
    // No buffer reuse in either plan: every value stays readable after the run.
    let single = backend_metal::compile_for(backend_metal::Target::native(), &graph, &gguf, 512, 1, false)?;
    let batched = backend_metal::compile_for(backend_metal::Target::native(), &graph, &gguf, 512, n, false)?;
    let tokens: Vec<u32> = (0..n as u32 + 1).map(|index| [2u32, 105, 2364, 107, 5279, 529, 7001, 563, 9259, 236761, 1018, 1884, 563, 506, 5279, 529][index as usize % 16] + index).collect();
    let mut runtime = Runtime::new(&gguf, &[single, batched], 1)?;

    // Batch of the first n tokens (positions 0..n).
    runtime.run_batch_for_test(n, &tokens[..n], 0)?;
    // The outputs of every layer for every token of the batch.
    let layer_outputs: Vec<usize> = graph
        .ops
        .iter()
        .filter_map(|op| if let Op::ScaleBy { output, .. } = op { Some(*output) } else { None })
        .collect();
    let batched_layers: Vec<Vec<f32>> = layer_outputs.iter().map(|&value| runtime.read_value_of(n, value)).collect();

    // The same tokens one by one; the single plan's value after token t is compared with row t of the batch.
    let width = batched_layers[0].len() / n;
    let mut worst = Vec::new();
    for position in 0..n {
        runtime.step(tokens[position], position)?;
        for (layer, &value) in layer_outputs.iter().enumerate() {
            let one = runtime.read_value(value);
            let row = &batched_layers[layer][position * width..(position + 1) * width];
            let scale = one.iter().fold(0f32, |largest, x| largest.max(x.abs())).max(1e-6);
            let error = one.iter().zip(row).fold(0f32, |largest, (a, b)| largest.max((a - b).abs())) / scale;
            if position == n - 1 || position == 0 {
                worst.push((position, layer, error));
            }
        }
    }
    for (position, layer, error) in worst {
        println!("token {position} layer {layer:2}: relative difference {error:.2e}");
    }
    Ok(())
}
