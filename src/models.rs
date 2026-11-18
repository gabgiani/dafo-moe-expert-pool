//! Models you can pull, like `ollama list` and `ollama pull`: the catalog of DAFO's release page, the Gemma 4 GGUF files in it that the
//! compiler can run, and a downloader that rebuilds a model from the chunks of its mirror (each verified by SHA-256, resumable) or from
//! one address, and puts it, with its vision projector, in the models folder. `dafo-serve NAME` then serves it by name.
//!
//! The models folder is `$DAFO_MODELS` or `~/.dafo/models`; a model is the folder `<id>/` with its GGUF and `mmproj-*.gguf`.

use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail, ensure};
use serde_json::Value;
use sha2::{Digest, Sha256};

/// Where the catalog of models lives (the same one the DAFO Swarm nodes read).
pub const CATALOG_URL: &str = "https://github.com/gabgiani/dafo-swarm-downloads/releases/download/model-catalog/models.json";

/// A file of a model: the GGUF itself or the vision projector.
#[derive(Clone, Debug)]
pub struct RemoteFile {
    pub name: String,
    pub size: u64,
    pub sha256: Option<String>,
    /// A direct address, when the file is not served by the mirror.
    pub url: Option<String>,
}

/// An entry of the catalog the compiler can run.
#[derive(Clone, Debug)]
pub struct Entry {
    pub id: String,
    pub name: String,
    pub quantization: String,
    pub notes: String,
    pub engine: String,
    pub file: RemoteFile,
    /// A `model.mirror-parts.json` describing the file as chunks of a release.
    pub mirror: Option<String>,
    pub extra: Vec<RemoteFile>,
}

pub fn models_dir() -> PathBuf {
    if let Some(path) = std::env::var_os("DAFO_MODELS") {
        return PathBuf::from(path);
    }
    let home = std::env::var_os("HOME").or_else(|| std::env::var_os("USERPROFILE")).map(PathBuf::from).unwrap_or_else(|| PathBuf::from("."));
    home.join(".dafo").join("models")
}

fn agent() -> ureq::Agent {
    ureq::Agent::config_builder().timeout_connect(Some(Duration::from_secs(30))).timeout_recv_response(Some(Duration::from_secs(120))).build().into()
}

fn get_text(url: &str) -> Result<String> {
    agent().get(url).call().with_context(|| format!("cannot fetch {url}"))?.body_mut().read_to_string().with_context(|| format!("cannot read {url}"))
}

fn text(value: &Value, key: &str) -> String {
    value.get(key).and_then(Value::as_str).unwrap_or("").to_owned()
}

fn file_of(value: &Value) -> RemoteFile {
    RemoteFile {
        name: text(value, "file_name"),
        size: value.get("size_bytes").and_then(Value::as_u64).unwrap_or(0),
        sha256: value.get("sha256").and_then(Value::as_str).map(str::to_owned),
        url: value.get("download_url").and_then(Value::as_str).map(str::to_owned),
    }
}

/// The models of the catalog this compiler can run: the Gemma 4 GGUF files marked for the compiled engine.
pub fn catalog() -> Result<Vec<Entry>> {
    let url = std::env::var("DAFO_CATALOG_URL").unwrap_or_else(|_| CATALOG_URL.to_owned());
    let document: Value = serde_json::from_str(&get_text(&url)?).context("the catalog is not JSON")?;
    let models = document.get("models").and_then(Value::as_array).context("the catalog has no `models`")?;
    Ok(models
        .iter()
        .filter(|model| text(model, "architecture") == "gemma4" && text(model, "engine") == "compiled" && text(model, "file_name").ends_with(".gguf"))
        .map(|model| Entry {
            id: text(model, "id"),
            name: text(model, "name"),
            quantization: text(model, "quantization"),
            notes: text(model, "notes"),
            engine: text(model, "engine"),
            file: file_of(model),
            mirror: model.get("mirror_manifest_url").and_then(Value::as_str).map(str::to_owned),
            extra: model.get("extra_files").and_then(Value::as_array).map(|files| files.iter().map(file_of).collect()).unwrap_or_default(),
        })
        .collect())
}

/// The folder of a model by id.
pub fn model_dir(id: &str) -> PathBuf {
    models_dir().join(id)
}

/// The GGUF of a pulled model (the file that is not a projector), if the folder has one.
pub fn gguf_in(directory: &Path) -> Option<PathBuf> {
    let mut found: Vec<PathBuf> = std::fs::read_dir(directory)
        .ok()?
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| path.extension().is_some_and(|extension| extension == "gguf") && !path.file_name().is_some_and(|name| name.to_string_lossy().starts_with("mmproj")))
        .collect();
    found.sort();
    found.into_iter().next()
}

/// A pulled model by name: its id, or the start of it when only one matches.
pub fn resolve(name: &str) -> Option<PathBuf> {
    let direct = model_dir(name);
    if direct.is_dir() {
        return gguf_in(&direct);
    }
    let matches: Vec<PathBuf> = std::fs::read_dir(models_dir())
        .ok()?
        .filter_map(|entry| entry.ok())
        .filter(|entry| entry.file_name().to_string_lossy().starts_with(name))
        .map(|entry| entry.path())
        .collect();
    if matches.len() == 1 { gguf_in(&matches[0]) } else { None }
}

/// The ids of the models in the models folder.
pub fn pulled() -> Vec<String> {
    let mut ids: Vec<String> = std::fs::read_dir(models_dir())
        .map(|entries| entries.filter_map(|entry| entry.ok()).filter(|entry| gguf_in(&entry.path()).is_some()).map(|entry| entry.file_name().to_string_lossy().into_owned()).collect())
        .unwrap_or_default();
    ids.sort();
    ids
}

fn gib(bytes: u64) -> f64 {
    bytes as f64 / (1u64 << 30) as f64
}

/// `ollama list`: the catalog with what is already on this machine marked.
pub fn print_list() -> Result<()> {
    let entries = catalog()?;
    let local = pulled();
    println!("{:<34} {:>9}  {:<8} {}", "NAME", "SIZE", "PULLED", "DESCRIPTION");
    for entry in &entries {
        let size = entry.file.size + entry.extra.iter().map(|file| file.size).sum::<u64>();
        println!("{:<34} {:>7.1} GB  {:<8} {}", entry.id, size as f64 / 1e9, if local.contains(&entry.id) { "yes" } else { "" }, entry.name);
    }
    println!("\nmodels folder: {}   (pull one with: dafo-compiler pull NAME)", models_dir().display());
    Ok(())
}

/// `ollama pull`: downloads the model and its projector into the models folder.
pub fn pull(name: &str) -> Result<PathBuf> {
    let entries = catalog()?;
    let matching: Vec<&Entry> = entries.iter().filter(|entry| entry.id == name).collect();
    let entry = match matching.first() {
        Some(entry) => *entry,
        None => {
            let starts: Vec<&Entry> = entries.iter().filter(|entry| entry.id.starts_with(name)).collect();
            ensure!(starts.len() == 1, "no model named `{name}` in the catalog{} (see: dafo-compile list)", if starts.len() > 1 { " (the name matches several)" } else { "" });
            starts[0]
        }
    };
    let directory = model_dir(&entry.id);
    std::fs::create_dir_all(&directory).with_context(|| format!("cannot create {}", directory.display()))?;
    println!("pulling {} ({})", entry.id, entry.name);
    let target = directory.join(&entry.file.name);
    if file_is_good(&target, &entry.file) {
        println!("  {} is already here", entry.file.name);
    } else {
        match &entry.mirror {
            Some(manifest) => {
                if let Err(error) = pull_from_mirror(manifest, &entry.file, &target) {
                    // The mirror failed: the same file from its original address, if the catalog has one.
                    let Some(url) = entry.file.url.as_deref() else { return Err(error) };
                    println!("\n  the mirror failed ({error:#}); trying {url}");
                    download(url, &target, &entry.file)?;
                }
            }
            None => {
                let url = entry.file.url.as_deref().context("the catalog gives no address for this model")?;
                download(url, &target, &entry.file)?;
            }
        }
    }
    for file in &entry.extra {
        let target = directory.join(&file.name);
        if file_is_good(&target, file) {
            println!("  {} is already here", file.name);
        } else {
            let url = file.url.as_deref().with_context(|| format!("the catalog gives no address for {}", file.name))?;
            download(url, &target, file)?;
        }
    }
    println!("done: {}\nserve it with: dafo-serve {}", directory.display(), entry.id);
    Ok(directory)
}

fn file_is_good(path: &Path, file: &RemoteFile) -> bool {
    std::fs::metadata(path).is_ok_and(|metadata| metadata.len() == file.size && file.size > 0)
}

fn sha256_of(path: &Path, label: &str) -> Result<String> {
    let mut file = File::open(path)?;
    let total = file.metadata()?.len();
    let (mut hasher, mut buffer, mut done) = (Sha256::new(), vec![0u8; 8 << 20], 0u64);
    let mut shown = Instant::now();
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
        done += read as u64;
        if shown.elapsed() > Duration::from_millis(500) {
            print!("\r  checking {label}: {:.0} %   ", 100.0 * done as f64 / total.max(1) as f64);
            let _ = std::io::stdout().flush();
            shown = Instant::now();
        }
    }
    println!("\r  checked {label}                    ");
    Ok(hasher.finalize().iter().map(|byte| format!("{byte:02x}")).collect())
}

/// Downloads `url` to `target`, resuming a partial file (and again after a dropped connection), and checks the size and the SHA-256.
fn download(url: &str, target: &Path, file: &RemoteFile) -> Result<()> {
    let mut attempts = 0;
    loop {
        attempts += 1;
        match download_once(url, target, file) {
            Err(error) if attempts < 6 && error.to_string().contains("stopped") => println!("\n  {error:#}; resuming"),
            other => return other,
        }
    }
}

fn download_once(url: &str, target: &Path, file: &RemoteFile) -> Result<()> {
    let partial = PathBuf::from(format!("{}.part", target.display()));
    let mut offset = std::fs::metadata(&partial).map(|metadata| metadata.len()).unwrap_or(0);
    if file.size > 0 && offset > file.size {
        std::fs::remove_file(&partial)?;
        offset = 0;
    }
    if offset < file.size || file.size == 0 {
        let mut request = agent().get(url);
        if offset > 0 {
            request = request.header("Range", &format!("bytes={offset}-"));
        }
        let mut response = request.call().with_context(|| format!("cannot download {url}"))?;
        let resumed = response.status().as_u16() == 206;
        let mut output = OpenOptions::new().create(true).write(true).open(&partial)?;
        if resumed {
            output.seek(SeekFrom::End(0))?;
        } else {
            output.set_len(0)?;
            offset = 0;
        }
        let mut reader = response.body_mut().as_reader();
        let (mut buffer, mut done, started, mut shown) = (vec![0u8; 1 << 20], offset, Instant::now(), Instant::now());
        loop {
            let read = reader.read(&mut buffer).with_context(|| format!("the download of {} stopped (run the command again to resume)", file.name))?;
            if read == 0 {
                break;
            }
            output.write_all(&buffer[..read])?;
            done += read as u64;
            if shown.elapsed() > Duration::from_millis(500) {
                let speed = (done - offset) as f64 / started.elapsed().as_secs_f64().max(0.001);
                print!("\r  {}: {:.2} of {:.2} GiB  {:.0} MB/s   ", file.name, gib(done), gib(file.size.max(done)), speed / 1e6);
                let _ = std::io::stdout().flush();
                shown = Instant::now();
            }
        }
        println!("\r  {}: {:.2} GiB downloaded                          ", file.name, gib(done));
        output.flush()?;
    }
    let size = std::fs::metadata(&partial)?.len();
    ensure!(file.size == 0 || size == file.size, "{} is {size} bytes, the catalog says {} (run the command again to resume)", file.name, file.size);
    if let Some(expected) = &file.sha256 {
        let actual = sha256_of(&partial, &file.name)?;
        if &actual != expected {
            let _ = std::fs::remove_file(&partial);
            bail!("{} does not match its SHA-256 (got {actual}, expected {expected}): the partial file was removed, run the command again", file.name);
        }
    }
    std::fs::rename(&partial, target)?;
    Ok(())
}

/// Rebuilds a file from the chunks of a mirror: each chunk is downloaded, checked against its own SHA-256 and appended, so a damaged chunk
/// is fetched again and the rest are kept; the finished file is checked against the whole-file SHA-256.
fn pull_from_mirror(manifest_url: &str, file: &RemoteFile, target: &Path) -> Result<()> {
    let manifest: Value = serde_json::from_str(&get_text(manifest_url)?).context("the mirror manifest is not JSON")?;
    let entries = manifest.get("files").and_then(Value::as_array).context("the mirror manifest has no files")?;
    let entry = entries.iter().find(|entry| text(entry, "relative_path") == file.name).with_context(|| format!("the mirror has no {}", file.name))?;
    let mut chunks: Vec<(String, u64, u64, String)> = entry
        .get("chunks")
        .and_then(Value::as_array)
        .context("the mirror lists no chunks")?
        .iter()
        .map(|chunk| (text(chunk, "file_name"), chunk.get("offset").and_then(Value::as_u64).unwrap_or(0), chunk.get("size_bytes").and_then(Value::as_u64).unwrap_or(0), text(chunk, "sha256")))
        .collect();
    chunks.sort_by_key(|chunk| chunk.1);
    let base = manifest_url.rsplit_once('/').map(|(base, _)| base).context("a manifest address without a folder")?;
    let partial = PathBuf::from(format!("{}.part", target.display()));
    // Keep the chunks already joined: the partial file always ends at a chunk boundary after a successful append.
    let mut length = std::fs::metadata(&partial).map(|metadata| metadata.len()).unwrap_or(0);
    let mut next = chunks.iter().position(|chunk| chunk.1 + chunk.2 > length).unwrap_or(chunks.len());
    if let Some(chunk) = chunks.get(next) {
        length = chunk.1;
    }
    let mut output = OpenOptions::new().create(true).write(true).open(&partial)?;
    output.set_len(length)?;
    output.seek(SeekFrom::Start(length))?;
    if next > 0 {
        println!("  resuming after {} of {} chunks", next, chunks.len());
    }
    let started = Instant::now();
    while next < chunks.len() {
        let (name, _, size, sha256) = chunks[next].clone();
        let mut attempts = 0;
        let bytes = loop {
            attempts += 1;
            let url = format!("{base}/{name}");
            let result = (|| -> Result<Vec<u8>> {
                let mut response = agent().get(&url).call().with_context(|| format!("cannot download {url}"))?;
                let mut reader = response.body_mut().as_reader();
                let mut bytes = Vec::with_capacity(size as usize);
                let (mut buffer, mut shown, chunk_started) = (vec![0u8; 1 << 20], Instant::now(), Instant::now());
                loop {
                    let read = reader.read(&mut buffer)?;
                    if read == 0 {
                        break;
                    }
                    bytes.extend_from_slice(&buffer[..read]);
                    if shown.elapsed() > Duration::from_millis(500) {
                        let before: u64 = chunks[..next].iter().map(|chunk| chunk.2).sum();
                        print!("\r  {}: chunk {} of {}  {:.2} of {:.2} GiB  {:.0} MB/s   ", file.name, next + 1, chunks.len(), gib(before + bytes.len() as u64), gib(file.size), bytes.len() as f64 / chunk_started.elapsed().as_secs_f64().max(0.001) / 1e6);
                        let _ = std::io::stdout().flush();
                        shown = Instant::now();
                    }
                }
                ensure!(bytes.len() as u64 == size, "chunk {name} has {} bytes, expected {size}", bytes.len());
                let actual: String = Sha256::digest(&bytes).iter().map(|byte| format!("{byte:02x}")).collect();
                ensure!(actual == sha256, "chunk {name} failed its SHA-256");
                Ok(bytes)
            })();
            match result {
                Ok(bytes) => break bytes,
                Err(error) if attempts < 3 => println!("\n  {error:#}; fetching it again"),
                Err(error) => return Err(error),
            }
        };
        output.write_all(&bytes)?;
        output.flush()?;
        next += 1;
    }
    println!("\r  {}: {} chunks joined                                   ", file.name, chunks.len());
    drop(output);
    let size = std::fs::metadata(&partial)?.len();
    ensure!(file.size == 0 || size == file.size, "the joined file is {size} bytes, the catalog says {}", file.size);
    if let Some(expected) = &file.sha256 {
        let actual = sha256_of(&partial, &file.name)?;
        if &actual != expected {
            let _ = std::fs::remove_file(&partial);
            bail!("the joined {} does not match its SHA-256 (got {actual}, expected {expected}): removed, run the command again", file.name);
        }
    }
    std::fs::rename(&partial, target)?;
    Ok(())
}

/// `ollama rm`.
pub fn remove(name: &str) -> Result<()> {
    let directory = model_dir(name);
    ensure!(directory.is_dir(), "`{name}` is not in {} (see: dafo-compile list)", models_dir().display());
    std::fs::remove_dir_all(&directory)?;
    println!("removed {}", directory.display());
    Ok(())
}
