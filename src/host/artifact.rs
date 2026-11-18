//! The compiled model as a folder, outside any application:
//!
//! ```text
//! model.dafo/
//!   manifest.json   what it is, the context it was compiled for, the plans, how the host must read and stop
//!   model.gguf      the weights (a hard link or symbolic link to the file it was compiled from, or a copy)
//!   tokenizer.json  the tokenizer, ready to load
//!   kernels.metal   the Metal source the plans use, as generated for this model
//! ```
//!
//! The dispatch plans themselves are rebuilt from the weights and the manifest in milliseconds when the folder is opened; they are
//! not stored. The folder is what a host needs to serve the model, and it can be moved to another Mac.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use serde_json::{Value, json};
use tokenizers::Tokenizer;

use crate::gguf::{Gguf, Scalar};

pub const FORMAT: u64 = 1;

/// What a folder says about itself.
pub struct Opened {
    pub name: String,
    pub weights: PathBuf,
    pub context: usize,
    pub tokenizer: Tokenizer,
    /// The vision projector stored in the folder (`mmproj.gguf`), when it was compiled with one.
    pub projector: Option<PathBuf>,
    /// Most soft tokens one image may use.
    pub visual_tokens: usize,
}

pub fn is_artifact(path: &Path) -> bool {
    path.is_dir() && path.join("manifest.json").is_file()
}

/// Writes the folder `directory` for the model at `weights`, compiled for `context` tokens. `copy` makes a copy of the weights
/// instead of linking them.
pub fn write(directory: &Path, weights: &Path, gguf: &Gguf, plans: &[crate::backend_metal::Plan], context: usize, copy: bool, projector: Option<&Path>) -> Result<()> {
    std::fs::create_dir_all(directory).with_context(|| format!("cannot create {}", directory.display()))?;
    let weights = weights.canonicalize()?;
    let target = directory.join("model.gguf");
    if target.exists() {
        bail!("{} already exists: pick an empty folder", target.display());
    }
    let how = if copy {
        std::fs::copy(&weights, &target)?;
        "copy"
    } else if std::fs::hard_link(&weights, &target).is_ok() {
        "hard link"
    } else {
        link_or_copy(&weights, &target)?
    };
    // The vision projector goes beside the weights, linked or copied the same way.
    let projector_file = match projector {
        Some(path) => {
            let path = path.canonicalize().with_context(|| format!("cannot find the projector {}", path.display()))?;
            let target = directory.join("mmproj.gguf");
            if copy {
                std::fs::copy(&path, &target)?;
            } else if std::fs::hard_link(&path, &target).is_err() {
                link_or_copy(&path, &target)?;
            }
            Some("mmproj.gguf")
        }
        None => None,
    };
    let tokenizer = super::tokenizer::from_gguf(gguf)?;
    tokenizer.save(directory.join("tokenizer.json"), false).map_err(anyhow::Error::msg)?;
    std::fs::write(directory.join(if cfg!(target_os = "macos") { "kernels.metal" } else { "kernels.cu" }), &plans[0].source)?;
    let name = match gguf.metadata("general.name") {
        Some(Scalar::Text(name)) => name.clone(),
        _ => weights.file_stem().map(|stem| stem.to_string_lossy().into_owned()).unwrap_or_default(),
    };
    let manifest = json!({
        "format": FORMAT,
        "compiler": env!("CARGO_PKG_VERSION"),
        "name": name,
        "architecture": match gguf.metadata("general.architecture") { Some(Scalar::Text(text)) => text.as_str(), _ => "" },
        "weights": { "file": "model.gguf", "stored_as": how, "bytes": std::fs::metadata(&weights)?.len() },
        "context": context,
        "plans": plans.iter().map(|plan| json!({ "batch": plan.batch, "dispatches": plan.steps.len(), "scratch_buffers": plan.scratch.len() })).collect::<Vec<_>>(),
        "tokenizer": "tokenizer.json",
        "kernels": "kernels.metal",
        "chat": "gemma4",
        "vision": projector_file.map(|file| json!({ "projector": file, "visual_tokens": 560 })),
        "stop_tokens": ["<turn|>", "<|tool_response>"],
    });
    std::fs::write(directory.join("manifest.json"), serde_json::to_string_pretty(&manifest)? + "\n")?;
    Ok(())
}

/// A symbolic link where the system allows one cheaply (Unix); a copy elsewhere (Windows links need a privilege).
fn link_or_copy(weights: &Path, target: &Path) -> Result<&'static str> {
    #[cfg(unix)]
    {
        std::os::unix::fs::symlink(weights, target)?;
        Ok("symbolic link")
    }
    #[cfg(not(unix))]
    {
        std::fs::copy(weights, target)?;
        Ok("copy")
    }
}

pub fn read(directory: &Path) -> Result<Opened> {
    let manifest: Value = serde_json::from_slice(&std::fs::read(directory.join("manifest.json")).context("no manifest.json: not a compiled model folder")?)?;
    if manifest["format"].as_u64() != Some(FORMAT) {
        bail!("this folder has format {}, this build reads format {FORMAT}", manifest["format"]);
    }
    let weights = directory.join(manifest["weights"]["file"].as_str().context("manifest has no weights")?);
    if !weights.exists() {
        bail!("the weights {} are missing (a link whose target moved?)", weights.display());
    }
    let tokenizer = Tokenizer::from_file(directory.join(manifest["tokenizer"].as_str().unwrap_or("tokenizer.json"))).map_err(anyhow::Error::msg)?;
    Ok(Opened {
        name: manifest["name"].as_str().unwrap_or("model").to_owned(),
        weights,
        context: manifest["context"].as_u64().unwrap_or(16384) as usize,
        tokenizer,
        projector: manifest["vision"]["projector"].as_str().map(|file| directory.join(file)).filter(|path| path.exists()),
        visual_tokens: manifest["vision"]["visual_tokens"].as_u64().unwrap_or(560) as usize,
    })
}
