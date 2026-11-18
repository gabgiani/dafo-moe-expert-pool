//! A smoke test of the CUDA layer: opens the device, compiles the CUDA kernels of the compiler with NVRTC, loads every one of them and
//! runs `add` and `rms_norm` against the CPU.
#[cfg(any(windows, target_os = "linux"))]
fn main() -> anyhow::Result<()> {
    use dafo_compiler_llm::cuda::Gpu;
    let gpu = Gpu::new()?;
    let (free, total) = gpu.memory();
    println!("{}: {:.2} GiB free of {:.2} GiB", gpu.name(), free as f64 / 1073741824.0, total as f64 / 1073741824.0);
    let header = "#define NR 4\n#define NSG 2\n#define STATE_VISION 584\n#define ROUTE_MAX_WIDTH 2816\n#define ROUTE_MAX_EXPERTS 128\n#define ROUTE_MAX_USED 8\n";
    let source = format!("{header}{}", include_str!("../src/backend_cuda/kernels.cu"));
    let started = std::time::Instant::now();
    let library = gpu.library(&source)?;
    println!("compiled in {:.2} s", started.elapsed().as_secs_f64());
    let names: [&'static str; 24] = [
        "embed_q8", "zero_u32", "rms_norm", "scale_by", "add", "gelu_mul", "mv_q8_0", "mv_id_q4_k", "mv_id_q5_1", "mv_id_q8_0", "experts_sum", "route",
        "rope", "cache_write", "attention_256", "attention_512", "argmax_token", "mm_q8_0", "kernel_moe_route_rows", "mm_id_q8_0", "mm_id_q5_1", "mm_id_q4_k", "stage_experts", "stage_all",
    ];
    let mut kernels = std::collections::HashMap::new();
    for name in names {
        kernels.insert(name, gpu.kernel(&library, name)?);
    }
    println!("{} kernels loaded", kernels.len());

    let n = 1000usize;
    let a: Vec<f32> = (0..n).map(|i| i as f32 * 0.5).collect();
    let b: Vec<f32> = (0..n).map(|i| 3.0 - i as f32).collect();
    let (ba, bb, out) = (gpu.buffer_from(&a)?, gpu.buffer_from(&b)?, gpu.buffer(n * 4)?);
    gpu.run(|encoder| {
        encoder.kernel(&kernels["add"]);
        encoder.buffer(0, &ba, 0);
        encoder.buffer(1, &bb, 0);
        encoder.buffer(2, &out, 0);
        encoder.value(3, &(n as u32));
        encoder.dispatch([n.div_ceil(64), 1, 1], [64, 1, 1]);
    })?;
    let result: Vec<f32> = out.to_vec();
    let wrong = (0..n).filter(|&i| (result[i] - (a[i] + b[i])).abs() > 1e-6).count();
    println!("add: {wrong} wrong of {n}");

    // rms_norm: 4 groups of 256 values, scaled.
    let (width, groups) = (256usize, 4usize);
    let x: Vec<f32> = (0..width * groups).map(|i| ((i * 37 % 101) as f32 - 50.0) / 25.0).collect();
    let w: Vec<f32> = (0..width).map(|i| 0.5 + (i % 7) as f32 * 0.1).collect();
    let (bx, bw, by) = (gpu.buffer_from(&x)?, gpu.buffer_from(&w)?, gpu.buffer(width * groups * 4)?);
    gpu.run(|encoder| {
        encoder.kernel(&kernels["rms_norm"]);
        encoder.buffer(0, &bx, 0);
        encoder.buffer(1, &bw, 0);
        encoder.buffer(2, &by, 0);
        encoder.value(3, &(width as u32));
        encoder.value(4, &1e-6f32);
        encoder.value(5, &1u32);
        encoder.dispatch([groups, 1, 1], [128, 1, 1]);
    })?;
    let y: Vec<f32> = by.to_vec();
    let mut worst = 0f32;
    for g in 0..groups {
        let row = &x[g * width..(g + 1) * width];
        let inverse = 1.0 / (row.iter().map(|v| v * v).sum::<f32>() / width as f32 + 1e-6).sqrt();
        for i in 0..width {
            worst = worst.max((y[g * width + i] - row[i] * inverse * w[i]).abs());
        }
    }
    println!("rms_norm: largest difference {worst:e}");

    // What the GPU reads from pinned host memory per second, with the staging kernel: 300 MiB, three times.
    let bytes = 300usize << 20;
    let host = gpu.buffer_shared(bytes)?;
    let device = gpu.buffer(bytes)?;
    let stage_all = gpu.kernel(&library, "stage_all")?;
    // The copy engine (cuMemcpyHtoD) from the same pinned memory.
    let mut device_copy = gpu.buffer(bytes)?;
    for round in 0..3 {
        let started = std::time::Instant::now();
        device_copy.write_bytes(0, host.as_slice::<u8>())?;
        let seconds = started.elapsed().as_secs_f64();
        println!("host -> device by cuMemcpyHtoD, round {round}: {:.2} GB/s", bytes as f64 / seconds / 1e9);
    }
    for round in 0..3 {
        let timing = gpu.run(|encoder| {
            encoder.kernel(&stage_all);
            encoder.buffer(0, &host, 0);
            encoder.buffer(1, &device, 0);
            encoder.value(2, &(bytes as u64));
            encoder.dispatch([(bytes / 16).div_ceil(256), 1, 1], [256, 1, 1]);
        })?;
        println!("host -> device by kernel, round {round}: {:.2} GB/s ({:.1} ms)", bytes as f64 / timing.gpu / 1e9, timing.gpu * 1000.0);
    }
    Ok(())
}
#[cfg(not(any(windows, target_os = "linux")))]
fn main() {}
