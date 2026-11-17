//! Experiment E2 (see `docs/RESULTS.md`): memory bandwidth of a quantized matrix-vector product with the layout
//! of the GGUF against a planar layout, on the real output matrix and a real final state.
//!
//!   cargo run --release --example e2_layout -- MODEL.gguf STATES.bin
//!
//! `STATES.bin` is a dump of `DAFO_SWARM_DUMP_FINAL_HIDDEN`; its last state is the vector.

use anyhow::{Context, Result, ensure};
use dafo_compiler_llm::gguf::{Encoding, Gguf};
use dafo_compiler_llm::metal::{Buffer, Gpu, Timing};

const KERNELS: &str = include_str!("../experiments/e2-layout/kernels.metal");
const BLOCK: usize = 32;
const BLOCK_BYTES: usize = 34;
const WARMUP: usize = 3;
const RUNS: usize = 15;

#[repr(C)]
#[derive(Clone, Copy)]
struct Params {
    width: u32,
    rows: u32,
}

fn half_to_f32(bits: u16) -> f32 {
    let sign = if bits & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exponent = ((bits >> 10) & 0x1f) as i32;
    let mantissa = (bits & 0x3ff) as f32;
    sign * match exponent {
        0 => mantissa * 2f32.powi(-24),
        31 => f32::INFINITY,
        _ => (1.0 + mantissa / 1024.0) * 2f32.powi(exponent - 15),
    }
}

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

fn main() -> Result<()> {
    let mut arguments = std::env::args().skip(1);
    let model = arguments.next().context("usage: e2_layout MODEL.gguf STATES.bin")?;
    let states = arguments.next().context("usage: e2_layout MODEL.gguf STATES.bin")?;

    let gguf = Gguf::open(&model)?;
    let name = if gguf.tensor("output.weight").is_some() { "output.weight" } else { "token_embd.weight" };
    let tensor = gguf.tensor(name).context("the model has no output matrix")?;
    ensure!(tensor.encoding == Encoding::Q8_0, "{name} is not Q8_0");
    let (width, rows) = (tensor.dims[0], tensor.dims[1]);
    let blocks = width / BLOCK;
    let data = gguf.data(tensor);

    let dump = std::fs::read(&states)?;
    ensure!(dump.len() >= width * 4, "the dump holds no state");
    let vector: Vec<f32> =
        dump[dump.len() - width * 4..].chunks_exact(4).map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap())).collect();

    // Reference on the CPU for a sample of rows.
    let sample: Vec<usize> = (0..rows).step_by(4099).collect();
    let reference: Vec<f64> = sample
        .iter()
        .map(|&row| {
            (0..blocks)
                .map(|block| {
                    let bytes = &data[(row * blocks + block) * BLOCK_BYTES..][..BLOCK_BYTES];
                    let scale = half_to_f32(u16::from_le_bytes([bytes[0], bytes[1]])) as f64;
                    let sum: f64 = bytes[2..].iter().zip(&vector[block * BLOCK..]).map(|(&quant, &value)| (quant as i8) as f64 * value as f64).sum();
                    sum * scale
                })
                .sum()
        })
        .collect();

    let gpu = Gpu::new()?;
    println!("{}: {name} {rows} x {width} Q8_0, {:.0} MiB", gpu.name(), data.len() as f64 / 1048576.0);

    let original = gpu.buffer_from(data)?;
    let mut quants = gpu.buffer(rows * width)?;
    let mut scales = gpu.buffer(rows * blocks * 2)?;
    {
        let quants = quants.as_mut_slice::<u8>();
        let scales = scales.as_mut_slice::<u8>();
        for (index, block) in data.chunks_exact(BLOCK_BYTES).enumerate() {
            scales[index * 2..index * 2 + 2].copy_from_slice(&block[..2]);
            quants[index * BLOCK..(index + 1) * BLOCK].copy_from_slice(&block[2..]);
        }
    }
    let input = gpu.buffer_from(&vector)?;
    let output = gpu.buffer(rows * 4)?;
    let params = Params { width: width as u32, rows: rows as u32 };
    let bytes = data.len() as f64;

    println!("{:<16} {:>3} {:>4} {:>10} {:>10} {:>10} {:>12}", "kernel", "NR", "NSG", "ms (GPU)", "GB/s", "ms (wall)", "max rel err");
    let planar = ["mv_q8_planar8", "mv_q8_planar16", "mv_q8_planar32"];
    let mut variants = vec![("mv_q8_gguf", 4, 2)];
    variants.extend(planar.iter().map(|&kernel| (kernel, 4, 2)));
    for (rows_per_group, groups) in [(2, 2), (8, 2), (4, 1), (4, 4), (8, 4), (2, 4), (16, 2)] {
        variants.push(("mv_q8_gguf", rows_per_group, groups));
        variants.push(("mv_q8_planar16", rows_per_group, groups));
    }
    for (kernel_name, rows_per_group, groups) in variants {
        let library = gpu.library(&format!("#define NR {rows_per_group}\n#define NSG {groups}\n{KERNELS}"))?;
        let kernel = gpu.kernel(&library, kernel_name)?;
        let run = |output: &Buffer| -> Result<Timing> {
            gpu.run(|encoder| {
                encoder.kernel(&kernel);
                if kernel_name == "mv_q8_gguf" {
                    encoder.buffer(0, &original, 0);
                    encoder.buffer(1, &input, 0);
                    encoder.buffer(2, output, 0);
                    encoder.value(3, &params);
                } else {
                    encoder.buffer(0, &quants, 0);
                    encoder.buffer(1, &scales, 0);
                    encoder.buffer(2, &input, 0);
                    encoder.buffer(3, output, 0);
                    encoder.value(4, &params);
                }
                encoder.dispatch([rows.div_ceil(rows_per_group * groups), 1, 1], [groups * 32, 1, 1]);
            })
        };
        for _ in 0..WARMUP {
            run(&output)?;
        }
        let timings = (0..RUNS).map(|_| run(&output)).collect::<Result<Vec<_>>>()?;
        let gpu_seconds = median(timings.iter().map(|timing| timing.gpu).collect());
        let wall_seconds = median(timings.iter().map(|timing| timing.wall).collect());
        let result = output.as_slice::<f32>();
        let scale = reference.iter().fold(0f64, |largest, value| largest.max(value.abs()));
        let error = sample.iter().zip(&reference).fold(0f64, |largest, (&row, &expected)| largest.max((result[row] as f64 - expected).abs())) / scale;
        println!(
            "{kernel_name:<16} {rows_per_group:>3} {groups:>4} {:>10.2} {:>10.1} {:>10.2} {error:>12.2e}",
            gpu_seconds * 1000.0,
            bytes / gpu_seconds / 1e9,
            wall_seconds * 1000.0,
        );
    }
    Ok(())
}
