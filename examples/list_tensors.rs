//! Lists the metadata keys (scalars) and the tensors of a GGUF file: name, dimensions, encoding.
fn main() -> anyhow::Result<()> {
    let path = std::env::args().nth(1).expect("FILE.gguf");
    let filter = std::env::args().nth(2).unwrap_or_default();
    let gguf = dafo_compiler_llm::gguf::Gguf::open(&path)?;
    for (name, tensor) in gguf.tensors() {
        if name.contains(&filter) {
            println!("{name:<70} {:?} {:?}", tensor.dims, tensor.encoding);
        }
    }
    Ok(())
}
