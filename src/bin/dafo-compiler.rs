//! `dafo-compiler`: pull a model, serve it, chat with it, the way `ollama` does, with nothing to configure. The machine is looked at and
//! the experts in memory, the context, the conversations at once and the images are decided from what it has.
//!
//!   dafo-compiler pull [NAME]     download a model (no name, or `gemma4`: the one that fits this machine)
//!   dafo-compiler list            the models you can pull, and which you have
//!   dafo-compiler serve [NAME]    the API (OpenAI-compatible) and the chat page on http://127.0.0.1:8080
//!   dafo-compiler chat [NAME]     a chat in the terminal (joins a running `serve`, or loads the model itself)
//!   dafo-compiler show [NAME]     what this machine would do with the model
//!   dafo-compiler rm NAME         delete a pulled model
//!
//! Without a NAME the model is the only one you pulled, or the last one you used. Compiling to a folder and benchmarks are in `dafo-compile`.

#[cfg(any(target_os = "macos", windows, target_os = "linux"))]
mod cli {
    use std::io::{BufRead, Write};
    use std::path::PathBuf;
    use std::sync::Arc;
    use std::sync::atomic::AtomicBool;
    use std::sync::mpsc;

    use anyhow::{Context, Result, bail};
    use dafo_compiler_llm::host::engine::{Event, Job, Settings};
    use dafo_compiler_llm::host::sampler::Sampling;
    use dafo_compiler_llm::host::{chat, server};
    use dafo_compiler_llm::{auto, gguf::Gguf, models};
    use serde_json::{Value, json};

    const HELP: &str = "dafo-compiler: pull, serve and chat with a model, like ollama

  dafo-compiler pull [NAME]     download a model (no name, or `gemma4`: the one that fits this machine)
  dafo-compiler list            the models you can pull, and which you have
  dafo-compiler serve [NAME]    the API (OpenAI-compatible) and the chat page on http://127.0.0.1:8080
  dafo-compiler chat [NAME]     a chat in the terminal (joins a running `serve`, or loads the model itself)
  dafo-compiler show [NAME]     what this machine would do with the model
  dafo-compiler rm NAME         delete a pulled model

Without a NAME the model is the only one you pulled, or the last one you used.
Compiling to a folder and benchmarks are in `dafo-compile`; every option of the server is in `dafo-serve`.";

    fn last_used_file() -> PathBuf {
        models::models_dir().parent().map(|folder| folder.join("last")).unwrap_or_else(|| PathBuf::from("dafo-last"))
    }

    fn remember(name: &str) {
        let _ = std::fs::write(last_used_file(), name);
    }

    /// The model to use: the one named, else the only one pulled, else the last one used.
    fn pick_model(name: Option<&str>) -> Result<String> {
        if let Some(name) = name {
            if models::resolve(name).is_none() {
                bail!("`{name}` is not pulled: dafo-compiler pull {name}");
            }
            return Ok(name.to_owned());
        }
        let pulled = models::pulled();
        match pulled.len() {
            0 => bail!("no model pulled yet: dafo-compiler pull gemma4   (see: dafo-compiler list)"),
            1 => Ok(pulled[0].clone()),
            _ => {
                let last = std::fs::read_to_string(last_used_file()).ok().map(|text| text.trim().to_owned());
                match last.filter(|last| pulled.contains(last)) {
                    Some(last) => Ok(last),
                    None => bail!("several models are pulled ({}): say which one: dafo-compiler serve NAME", pulled.join(", ")),
                }
            }
        }
    }

    fn gib(bytes: u64) -> f64 {
        bytes as f64 / (1u64 << 30) as f64
    }

    fn pull(name: Option<&str>) -> Result<()> {
        let entries = models::catalog()?;
        let machine = auto::machine(&models::models_dir());
        let chosen = match name {
            Some(name) if name != "gemma4" => name.to_owned(),
            _ => {
                // The complete model when the machine can hold a useful pool of it and the disk has room; otherwise the smaller cut.
                let complete = entries.iter().find(|entry| entry.notes.contains("all 128 experts") || entry.id.ends_with("-pool"));
                let cut = entries.iter().find(|entry| !entry.id.ends_with("-pool"));
                let total = |entry: &models::Entry| entry.file.size + entry.extra.iter().map(|file| file.size).sum::<u64>();
                let pick = match (complete, cut) {
                    (Some(complete), _) if machine.ram >= 12 * (1 << 30) && machine.disk_free > total(complete) + (2 << 30) => complete,
                    (_, Some(cut)) if machine.disk_free > total(cut) + (2 << 30) => cut,
                    (Some(entry), _) | (_, Some(entry)) => entry,
                    _ => bail!("the catalog lists no model this compiler can run"),
                };
                println!("this machine: {}, {:.0} GB of memory, {:.0} GB of free disk", machine.gpu, gib(machine.ram), gib(machine.disk_free));
                println!("choosing {}: {}", pick.id, if pick.id.ends_with("-pool") { "the complete model, with every expert available" } else { "the smaller cut: the complete model needs more disk than is free" });
                pick.id.clone()
            }
        };
        models::pull(&chosen)?;
        remember(&chosen);
        Ok(())
    }

    fn show(name: Option<&str>) -> Result<()> {
        let name = pick_model(name)?;
        let path = models::resolve(&name).context("model not found")?;
        let gguf = Gguf::open(&path)?;
        let graph = dafo_compiler_llm::frontends::token_graph(&gguf)?;
        let size = auto::ModelSize::of(&gguf, std::fs::metadata(&path)?.len(), dafo_compiler_llm::backend_metal::cache_footprint(&graph));
        let machine = auto::machine(&models::models_dir());
        let projector = path.parent().is_some_and(|folder| std::fs::read_dir(folder).is_ok_and(|entries| entries.filter_map(|entry| entry.ok()).any(|entry| entry.file_name().to_string_lossy().starts_with("mmproj"))));
        let plan = auto::plan(&machine, &size, projector);
        println!("model:   {name}  ({:.1} GiB, {} experts per layer, {} layers)", gib(size.file_bytes), size.experts, size.layers);
        println!("machine: {}", plan.summary);
        for note in &plan.notes {
            println!("  note: {note}");
        }
        println!("files:   {}", path.parent().map(|folder| folder.display().to_string()).unwrap_or_default());
        Ok(())
    }

    fn serve(arguments: &[String]) -> Result<()> {
        let (name, rest) = split_name(arguments);
        let name = pick_model(name.as_deref())?;
        remember(&name);
        let mut args = vec![name];
        args.extend(rest);
        server::run(args)
    }

    /// The first argument, when it is not a flag, is the model name.
    fn split_name(arguments: &[String]) -> (Option<String>, Vec<String>) {
        match arguments.first() {
            Some(first) if !first.starts_with("--") => (Some(first.clone()), arguments[1..].to_vec()),
            _ => (None, arguments.to_vec()),
        }
    }

    fn option(arguments: &[String], name: &str) -> Option<String> {
        arguments.iter().position(|argument| argument == name).and_then(|at| arguments.get(at + 1).cloned())
    }

    fn print_help_for_chat() {
        println!("  /bye           leave            /clear        forget the conversation");
        println!("  /image PATH    attach an image to your next message");
    }

    /// One turn of the terminal chat: reads a line, returns the attachment and the text, or `None` to leave.
    fn read_turn(images: &mut Vec<PathBuf>) -> Result<Option<String>> {
        loop {
            print!(">>> ");
            std::io::stdout().flush()?;
            let mut line = String::new();
            if std::io::stdin().lock().read_line(&mut line)? == 0 {
                return Ok(None);
            }
            let line = line.trim().to_owned();
            match line.as_str() {
                "" => continue,
                "/bye" | "/exit" | "/quit" => return Ok(None),
                "/clear" => return Ok(Some("\u{1}clear".into())),
                "/?" | "/help" => print_help_for_chat(),
                _ if line.starts_with("/image ") => {
                    let path = PathBuf::from(line["/image ".len()..].trim().trim_matches('"'));
                    if path.is_file() {
                        println!("(image attached: {})", path.display());
                        images.push(path);
                    } else {
                        println!("(no such file: {})", path.display());
                    }
                }
                _ => return Ok(Some(line)),
            }
        }
    }

    fn user_message(text: &str, images: &[PathBuf]) -> Result<Value> {
        if images.is_empty() {
            return Ok(json!({ "role": "user", "content": text }));
        }
        let mut parts = Vec::new();
        for path in images {
            let bytes = std::fs::read(path).with_context(|| format!("cannot read {}", path.display()))?;
            let kind = match path.extension().and_then(|extension| extension.to_str()).map(str::to_lowercase).as_deref() {
                Some("jpg" | "jpeg") => "jpeg",
                Some("webp") => "webp",
                _ => "png",
            };
            parts.push(json!({ "type": "image_url", "image_url": { "url": format!("data:image/{kind};base64,{}", chat::base64_encode(&bytes)) } }));
        }
        parts.push(json!({ "type": "text", "text": text }));
        Ok(json!({ "role": "user", "content": parts }))
    }

    fn chat_command(arguments: &[String]) -> Result<()> {
        let (name, rest) = split_name(arguments);
        let port: u16 = option(&rest, "--port").and_then(|port| port.parse().ok()).unwrap_or(8080);
        let health = format!("http://127.0.0.1:{port}/health");
        // A server already running: talk to it (the model is already in memory).
        let running = ureq::get(&health).call().is_ok();
        let (token, _) = server::stored_token();
        let mut messages: Vec<Value> = Vec::new();
        let mut images: Vec<PathBuf> = Vec::new();
        if running {
            println!("connected to the server on port {port}   (/? for help, /bye to leave)");
            loop {
                let Some(text) = read_turn(&mut images)? else { break };
                if text == "\u{1}clear" {
                    messages.clear();
                    println!("(conversation cleared)");
                    continue;
                }
                messages.push(user_message(&text, &images)?);
                images.clear();
                let body = json!({ "messages": messages, "stream": true });
                let response = ureq::post(&format!("http://127.0.0.1:{port}/v1/chat/completions")).header("Authorization", &format!("Bearer {token}")).header("Content-Type", "application/json").send(serde_json::to_string(&body)?.as_bytes());
                let mut answer = String::new();
                match response {
                    Ok(mut response) => {
                        let reader = std::io::BufReader::new(response.body_mut().as_reader());
                        for line in reader.lines() {
                            let line = line?;
                            let Some(data) = line.strip_prefix("data: ") else { continue };
                            if data == "[DONE]" {
                                break;
                            }
                            if let Ok(value) = serde_json::from_str::<Value>(data) {
                                if let Some(piece) = value["choices"][0]["delta"]["content"].as_str() {
                                    print!("{piece}");
                                    std::io::stdout().flush()?;
                                    answer.push_str(piece);
                                }
                            }
                        }
                        println!("\n");
                    }
                    Err(error) => {
                        println!("error: {error}\n");
                        messages.pop();
                        continue;
                    }
                }
                messages.push(json!({ "role": "assistant", "content": answer }));
            }
            return Ok(());
        }
        // No server: load the model here.
        let name = pick_model(name.as_deref())?;
        remember(&name);
        let mut args = vec![name];
        args.extend(rest);
        let started = server::start(&args)?;
        // From here the terminal is the conversation: no progress lines from the engine.
        dafo_compiler_llm::host::engine::QUIET.store(true, std::sync::atomic::Ordering::Relaxed);
        println!("\n{} is ready   (/? for help, /bye to leave)", started.name);
        let engine = started.engine;
        loop {
            let Some(text) = read_turn(&mut images)? else { break };
            if text == "\u{1}clear" {
                messages.clear();
                println!("(conversation cleared)");
                continue;
            }
            messages.push(user_message(&text, &images)?);
            images.clear();
            let prompt = match chat::render(&messages, &[], false) {
                Ok(prompt) => prompt,
                Err(message) => {
                    println!("error: {message}\n");
                    messages.pop();
                    continue;
                }
            };
            let pictures = chat::images_of(&messages).unwrap_or_default();
            let (events, receiver) = mpsc::channel();
            let settings = Settings { sampling: Sampling::default(), max_tokens: 2048, stop: Vec::new() };
            engine.submit(Job { prompt, settings, stream: true, cancel: Arc::new(AtomicBool::new(false)), events, profile: None, images: pictures })?;
            let mut final_text = None;
            for event in receiver {
                match event {
                    Event::Delta(piece) => {
                        print!("{piece}");
                        std::io::stdout().flush()?;
                    }
                    Event::Done { text, .. } => {
                        final_text = Some(text);
                        break;
                    }
                    Event::Failed { message, .. } => {
                        println!("\nerror: {message}");
                        break;
                    }
                }
            }
            println!("\n");
            match final_text {
                Some(text) => messages.push(json!({ "role": "assistant", "content": text })),
                None => {
                    messages.pop();
                }
            }
        }
        Ok(())
    }

    pub fn run() -> Result<()> {
        let arguments: Vec<String> = std::env::args().skip(1).collect();
        let command = arguments.first().map(String::as_str).unwrap_or("help");
        let rest = &arguments[arguments.len().min(1)..];
        match command {
            "pull" => pull(rest.first().map(String::as_str).filter(|name| !name.starts_with("--"))),
            "list" | "ls" => models::print_list(),
            "rm" => models::remove(rest.first().context("usage: dafo-compiler rm NAME")?),
            "show" => show(rest.first().map(String::as_str)),
            "serve" => serve(rest),
            "chat" | "run" => chat_command(rest),
            "help" | "--help" | "-h" => {
                println!("{HELP}");
                Ok(())
            }
            "version" | "--version" => {
                println!("dafo-compiler {}", env!("CARGO_PKG_VERSION"));
                Ok(())
            }
            other => bail!("unknown command `{other}`\n\n{HELP}"),
        }
    }
}

#[cfg(any(target_os = "macos", windows, target_os = "linux"))]
fn main() {
    if let Err(error) = cli::run() {
        eprintln!("dafo-compiler: {error:#}");
        std::process::exit(1);
    }
}

#[cfg(not(any(target_os = "macos", windows, target_os = "linux")))]
fn main() {
    eprintln!("dafo-compiler needs a Mac (Metal) or an NVIDIA GPU (CUDA)");
}
