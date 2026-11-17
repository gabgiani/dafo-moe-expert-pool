//! The Gemma 4 tokenizer, built from the arrays a GGUF carries (pieces, merges, token types): byte-level BPE with a metaspace and
//! the newline split Gemma uses.

use anyhow::{Context, Result, bail, ensure};
use tokenizers::{
    AddedToken, Tokenizer,
    decoders::{byte_fallback::ByteFallback, sequence::Sequence as DecoderSequence},
    models::bpe::{BPE, Vocab},
    pre_tokenizers::{
        metaspace::{Metaspace, PrependScheme},
        sequence::Sequence as PreTokenizerSequence,
        split::{Split, SplitPattern},
    },
    processors::template::TemplateProcessing,
    tokenizer::SplitDelimiterBehavior,
};

use crate::gguf::{Gguf, Scalar};

pub fn from_gguf(gguf: &Gguf) -> Result<Tokenizer> {
    let model = match gguf.metadata("tokenizer.ggml.model") {
        Some(Scalar::Text(model)) => model.as_str(),
        _ => bail!("the model declares no tokenizer"),
    };
    ensure!(model == "gemma4", "tokenizer `{model}` is not supported: only Gemma 4's");
    ensure!(!matches!(gguf.metadata("tokenizer.ggml.add_space_prefix"), Some(Scalar::Bool(true))), "Gemma 4's tokenizer must not add a space prefix");
    ensure!(matches!(gguf.metadata("tokenizer.ggml.add_bos_token"), Some(Scalar::Bool(true))), "Gemma 4's tokenizer must add its BOS token");
    let tokens = gguf.strings("tokenizer.ggml.tokens")?;
    let types = gguf.integers("tokenizer.ggml.token_type")?;
    ensure!(types.len() == tokens.len(), "{} tokens but {} token types", tokens.len(), types.len());
    let byte_tokens = types.iter().filter(|&&kind| kind == 6).count();
    ensure!(byte_tokens == 256, "the tokenizer has {byte_tokens} byte tokens, expected 256");
    let vocab: Vocab = tokens.iter().enumerate().map(|(id, token)| (token.clone(), id as u32)).collect();
    let merges = gguf
        .strings("tokenizer.ggml.merges")?
        .into_iter()
        .map(|merge| merge.split_once(' ').map(|(left, right)| (left.to_owned(), right.to_owned())).with_context(|| format!("bad merge `{merge}`")))
        .collect::<Result<Vec<_>>>()?;
    let unknown = gguf.unsigned("tokenizer.ggml.unknown_token_id").context("no unknown token id")?;
    let bpe = BPE::builder()
        .vocab_and_merges(vocab, merges)
        .unk_token(tokens.get(unknown).context("unknown token id out of range")?.clone())
        .byte_fallback(true)
        .build()
        .map_err(anyhow::Error::msg)?;
    let mut tokenizer = Tokenizer::new(bpe);
    let newline = Split::new(SplitPattern::Regex("[\n]+".to_owned()), SplitDelimiterBehavior::Isolated, false).map_err(anyhow::Error::msg)?;
    let metaspace = Metaspace::new('▁', PrependScheme::Never, false);
    tokenizer.with_pre_tokenizer(Some(PreTokenizerSequence::new(vec![newline.into(), metaspace.clone().into()])));
    tokenizer.with_decoder(Some(DecoderSequence::new(vec![ByteFallback::new().into(), metaspace.into()])));
    let special: Vec<AddedToken> = types.iter().enumerate().filter(|(_, kind)| (2..=5).contains(*kind)).map(|(id, _)| AddedToken::from(tokens[id].clone(), true)).collect();
    tokenizer.add_special_tokens(special).map_err(anyhow::Error::msg)?;
    let bos_id = gguf.unsigned("tokenizer.ggml.bos_token_id").context("no BOS token id")?;
    let bos = tokens.get(bos_id).context("BOS token id out of range")?.clone();
    let processor = TemplateProcessing::builder()
        .try_single(vec![bos.clone(), "$0".to_owned()])
        .map_err(anyhow::Error::msg)?
        .try_pair(vec![format!("{bos}:0"), "$A:0".to_owned(), format!("{bos}:1"), "$B:1".to_owned()])
        .map_err(anyhow::Error::msg)?
        .special_tokens(vec![(bos, bos_id as u32)])
        .build()
        .map_err(anyhow::Error::msg)?;
    tokenizer.with_post_processor(Some(processor));
    Ok(tokenizer)
}
