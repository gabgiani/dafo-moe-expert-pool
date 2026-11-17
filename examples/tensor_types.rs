//! Counts the tensors of a GGUF by storage type (does the compiler have a kernel for each?).
use dafo_compiler_llm::gguf::Gguf;
fn main() -> anyhow::Result<()> {
    let gguf = Gguf::open(&std::env::args().nth(1).expect("MODEL.gguf"))?;
    let mut counts = std::collections::BTreeMap::new();
    for (_, tensor) in gguf.tensors() {
        let entry = counts.entry(format!("{:?}", tensor.encoding)).or_insert((0usize, 0usize));
        entry.0 += 1;
        entry.1 += tensor.bytes();
    }
    for (name, (count, bytes)) in counts {
        println!("{name:<8} {count:>4} tensors {:>8.2} GiB", bytes as f64 / (1u64 << 30) as f64);
    }
    Ok(())
}
