//! Prints the token ids and pieces a prompt becomes with the tokenizer built from a GGUF (needs `--features host`).
fn main() -> anyhow::Result<()> {
    #[cfg(feature = "host")]
    {
        let mut arguments = std::env::args().skip(1);
        let model = arguments.next().expect("MODEL.gguf");
        let prompt = arguments.next().expect("prompt").replace("\\n", "\n");
        let gguf = dafo_compiler_llm::gguf::Gguf::open(model)?;
        let tokenizer = dafo_compiler_llm::host::tokenizer::from_gguf(&gguf)?;
        let encoding = tokenizer.encode(prompt.as_str(), true).map_err(anyhow::Error::msg)?;
        println!("{:?}", encoding.get_ids());
        println!("{:?}", encoding.get_tokens());
    }
    Ok(())
}
