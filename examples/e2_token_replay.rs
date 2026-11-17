//! Experiment E2, second part (see `docs/RESULTS.md`): the dense reads of one token (every Q8_0 matrix of the
//! layers, in order) as matrix-vector products, encoded in one command buffer against one command buffer per matrix.
//! It measures how much of the memory bandwidth survives when the matrices are small and many.
//!
//!   cargo run --release --example e2_token_replay -- MODEL.gguf

use anyhow::{Context, Result};
use dafo_compiler_llm::gguf::{Encoding, Gguf};
use dafo_compiler_llm::metal::Gpu;

const KERNELS: &str = include_str!("../experiments/e2-layout/kernels.metal");
const ROWS_PER_GROUP: usize = 4;
const GROUPS: usize = 2;
const WARMUP: usize = 3;
const RUNS: usize = 15;

#[repr(C)]
#[derive(Clone, Copy)]
struct Params {
    width: u32,
    rows: u32,
}

struct Matrix {
    offset: usize,
    params: Params,
}

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

fn main() -> Result<()> {
    let model = std::env::args().nth(1).context("usage: e2_token_replay MODEL.gguf")?;
    let gguf = Gguf::open(&model)?;
    let layer = |name: &str| name.split('.').nth(1).and_then(|index| index.parse::<usize>().ok()).unwrap_or(usize::MAX);
    let mut tensors: Vec<_> = gguf
        .tensors()
        .filter(|(name, tensor)| name.starts_with("blk.") && tensor.encoding == Encoding::Q8_0 && tensor.dims.len() == 2)
        .collect();
    tensors.sort_by_key(|(name, _)| (layer(name), name.to_string()));
    let bytes: usize = tensors.iter().map(|(_, tensor)| tensor.bytes()).sum();
    let widest = tensors.iter().map(|(_, tensor)| tensor.dims[0]).max().context("the model has no Q8_0 matrices")?;
    let tallest = tensors.iter().map(|(_, tensor)| tensor.dims[1]).max().unwrap_or(0);

    let gpu = Gpu::new()?;
    // One buffer with every matrix; the padding covers the rows a thread group reads past the last matrix.
    let mut weights = gpu.buffer(bytes + ROWS_PER_GROUP * GROUPS * widest * 2)?;
    let mut matrices = Vec::with_capacity(tensors.len());
    let mut offset = 0;
    for (_, tensor) in &tensors {
        weights.as_mut_slice::<u8>()[offset..offset + tensor.bytes()].copy_from_slice(gguf.data(tensor));
        matrices.push(Matrix { offset, params: Params { width: tensor.dims[0] as u32, rows: tensor.dims[1] as u32 } });
        offset += tensor.bytes();
    }
    let input = gpu.buffer_from(&(0..widest).map(|index| ((index % 17) as f32 - 8.0) / 64.0).collect::<Vec<_>>())?;
    let output = gpu.buffer(tallest * 4)?;
    let library = gpu.library(&format!("#define NR {ROWS_PER_GROUP}\n#define NSG {GROUPS}\n{KERNELS}"))?;
    let kernel = gpu.kernel(&library, "mv_q8_gguf")?;
    println!(
        "{}: {} Q8_0 matrices of the layers, {:.0} MiB (from {:.1} to {:.1} MiB each)",
        gpu.name(),
        matrices.len(),
        bytes as f64 / 1048576.0,
        tensors.iter().map(|(_, tensor)| tensor.bytes()).min().unwrap_or(0) as f64 / 1048576.0,
        tensors.iter().map(|(_, tensor)| tensor.bytes()).max().unwrap_or(0) as f64 / 1048576.0,
    );

    let encode = |encoder: &dafo_compiler_llm::metal::Encoder, matrix: &Matrix| {
        encoder.kernel(&kernel);
        encoder.buffer(0, &weights, matrix.offset);
        encoder.buffer(1, &input, 0);
        encoder.buffer(2, &output, 0);
        encoder.value(3, &matrix.params);
        encoder.dispatch([(matrix.params.rows as usize).div_ceil(ROWS_PER_GROUP * GROUPS), 1, 1], [GROUPS * 32, 1, 1]);
    };
    let together = || -> Result<(f64, f64)> {
        let timing = gpu.run(|encoder| matrices.iter().for_each(|matrix| encode(encoder, matrix)))?;
        Ok((timing.gpu, timing.wall))
    };
    let separately = || -> Result<(f64, f64)> {
        let started = std::time::Instant::now();
        let mut gpu_seconds = 0.0;
        for matrix in &matrices {
            gpu_seconds += gpu.run(|encoder| encode(encoder, matrix))?.gpu;
        }
        Ok((gpu_seconds, started.elapsed().as_secs_f64()))
    };
    let report = |label: &str, measure: &dyn Fn() -> Result<(f64, f64)>| -> Result<()> {
        for _ in 0..WARMUP {
            measure()?;
        }
        let timings = (0..RUNS).map(|_| measure()).collect::<Result<Vec<_>>>()?;
        let gpu_seconds = median(timings.iter().map(|timing| timing.0).collect());
        let wall_seconds = median(timings.iter().map(|timing| timing.1).collect());
        println!(
            "{label:<34} GPU {:>6.2} ms ({:>5.1} GB/s)   wall {:>6.2} ms ({:>5.1} GB/s)",
            gpu_seconds * 1000.0,
            bytes as f64 / gpu_seconds / 1e9,
            wall_seconds * 1000.0,
            bytes as f64 / wall_seconds / 1e9,
        );
        Ok(())
    };
    report("one command buffer for all", &together)?;
    report("one command buffer per matrix", &separately)?;
    Ok(())
}
