//! Runs the native vision encoder on an image and prints what it makes (needs `--features vision`): the number of soft tokens, their width,
//! a few statistics, the time, and optionally writes the embeddings as raw little-endian f32 to a file.
//!
//!   encode_image MMPROJ.gguf IMAGE.png [OUT.f32] [MAX_TOKENS]
#[cfg(all(feature = "vision", target_os = "macos"))]
fn main() -> anyhow::Result<()> {
    let mut arguments = std::env::args().skip(1);
    let projector = arguments.next().expect("MMPROJ.gguf");
    let image = std::fs::read(arguments.next().expect("IMAGE"))?;
    let out = arguments.next();
    let max_tokens: usize = arguments.next().map(|value| value.parse()).transpose()?.unwrap_or(560);
    let started = std::time::Instant::now();
    let encoder = dafo_compiler_llm::vision::VisionEncoder::open(std::path::Path::new(&projector), max_tokens)?;
    println!("opened in {:.2} s", started.elapsed().as_secs_f64());
    for run in 0..2 {
        let started = std::time::Instant::now();
        let encoded = encoder.encode(&image)?;
        let n = encoded.embeddings.len() as f64;
        let mean = encoded.embeddings.iter().map(|&value| value as f64).sum::<f64>() / n;
        let variance = encoded.embeddings.iter().map(|&value| (value as f64 - mean).powi(2)).sum::<f64>() / n;
        let bad = encoded.embeddings.iter().filter(|value| !value.is_finite()).count();
        println!("run {run}: {} soft tokens of {} values in {:.0} ms; mean {mean:.5}, std {:.5}, non-finite {bad}", encoded.tokens, encoded.width, started.elapsed().as_secs_f64() * 1000.0, variance.sqrt());
        if run == 1 {
            if let Some(path) = &out {
                let bytes: Vec<u8> = encoded.embeddings.iter().flat_map(|value| value.to_le_bytes()).collect();
                std::fs::write(path, bytes)?;
                println!("wrote {path}");
            }
        }
    }
    Ok(())
}
#[cfg(not(all(feature = "vision", target_os = "macos")))]
fn main() {}
