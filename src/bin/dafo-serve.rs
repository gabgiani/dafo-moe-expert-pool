//! `dafo-serve`: serves a compiled model over an OpenAI-compatible HTTP API, with a bearer token and a chat page.
//!
//!   dafo-serve MODEL.dafo            a folder made by `dafo-compile compile MODEL.gguf --out MODEL.dafo`
//!   dafo-serve MODEL.gguf            compiles at start, no folder
//!
//! Options: --host 127.0.0.1  --port 8080  --token TOKEN  --context TOKENS  --slots N
//!          --pool N [--profile FILE]   keep N experts per layer in memory out of the model's (the full model on disk), those of FILE
//!
//! GET  /                      the chat page (no token needed to load it; the page asks for the token)
//! GET  /health                200 when the model is loaded
//! GET  /v1/models             the model (needs the token)
//! POST /v1/chat/completions   messages, tools, streaming (needs the token)
//!
//! Without --token (or DAFO_SERVE_TOKEN) a random token is made and printed at start.

#[cfg(not(any(target_os = "macos", windows, target_os = "linux")))]
fn main() {
    eprintln!("dafo-serve needs a Mac (Metal) or an NVIDIA GPU (CUDA)");
    std::process::exit(1);
}

#[cfg(any(target_os = "macos", windows, target_os = "linux"))]
fn main() {
    if let Err(error) = server::run() {
        eprintln!("dafo-serve: {error:#}");
        std::process::exit(1);
    }
}

#[cfg(any(target_os = "macos", windows, target_os = "linux"))]
mod server {
    use std::collections::HashSet;
    use std::io::Read;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
    use std::sync::mpsc::{self, Receiver};
    use std::sync::Arc;
    use std::time::{SystemTime, UNIX_EPOCH};

    use anyhow::{Context, Result, bail};
    use dafo_compiler_llm::gguf::Gguf;
    use dafo_compiler_llm::host::engine::{Engine, Event, Job, Settings};
    use dafo_compiler_llm::host::sampler::Sampling;
    use dafo_compiler_llm::host::{artifact, chat, tokenizer};
    use serde_json::{Value, json};
    use tiny_http::{Header, Method, Request, Response, Server, StatusCode};

    const CHAT_PAGE: &str = include_str!("../host/chat.html");
    const MAX_BODY: u64 = 64 << 20;
    const DEFAULT_CONTEXT: usize = 16384;

    struct State {
        engine: Engine,
        name: String,
        token: String,
        counter: AtomicU64,
    }

    fn usage() -> &'static str {
        "usage: dafo-serve MODEL.dafo|MODEL.gguf|NAME [--host H] [--port P] [--token T] [--context TOKENS] [--slots N] [--pool EXPERTS [--vram-experts N] [--profile FILE] [--profiles DIR] [--adaptive] [--debug]] [--projector FILE [--visual-tokens N]] [--open]"
    }

    fn option(arguments: &[String], name: &str) -> Result<Option<String>> {
        match arguments.iter().position(|argument| argument == name) {
            Some(at) => Ok(Some(arguments.get(at + 1).with_context(|| format!("{name} needs a value"))?.clone())),
            None => Ok(None),
        }
    }

    fn random_token() -> String {
        let mut bytes = [0u8; 16];
        if let Ok(mut random) = std::fs::File::open("/dev/urandom") {
            let _ = random.read_exact(&mut bytes);
        }
        format!("dafo_{}", bytes.iter().map(|byte| format!("{byte:02x}")).collect::<String>())
    }

    pub fn run() -> Result<()> {
        let arguments: Vec<String> = std::env::args().skip(1).collect();
        let Some(model) = arguments.first().filter(|first| !first.starts_with("--")) else { bail!("{}", usage()) };
        let model = PathBuf::from(model);
        let host = option(&arguments, "--host")?.unwrap_or_else(|| "127.0.0.1".to_owned());
        let port: u16 = option(&arguments, "--port")?.map(|port| port.parse()).transpose()?.unwrap_or(8080);
        let slots: usize = option(&arguments, "--slots")?.map(|value| value.parse()).transpose()?.unwrap_or(2);
        let context_option: Option<usize> = option(&arguments, "--context")?.map(|value| value.parse()).transpose()?;
        let supplied = option(&arguments, "--token")?.or_else(|| std::env::var("DAFO_SERVE_TOKEN").ok()).filter(|token| !token.is_empty());
        let generated = supplied.is_none();
        let token = supplied.unwrap_or_else(random_token);

        // A name that is not a path is a model pulled with `dafo-compile pull`.
        let model = if model.exists() {
            model
        } else {
            match dafo_compiler_llm::models::resolve(&model.to_string_lossy()) {
                Some(path) => {
                    eprintln!("model {} found in {}", model.display(), dafo_compiler_llm::models::models_dir().display());
                    path
                }
                None => anyhow::bail!("{} is neither a file nor a pulled model (see: dafo-compile list, dafo-compile pull NAME)", model.display()),
            }
        };
        let (weights, tokenizer, name, context, stored_projector, stored_tokens) = if artifact::is_artifact(&model) {
            let opened = artifact::read(&model)?;
            (opened.weights, opened.tokenizer, opened.name, context_option.unwrap_or(opened.context), opened.projector, Some(opened.visual_tokens))
        } else {
            let gguf = Gguf::open(&model)?;
            let name = model.file_stem().map(|stem| stem.to_string_lossy().into_owned()).unwrap_or_else(|| "model".to_owned());
            (model.clone(), tokenizer::from_gguf(&gguf)?, name, context_option.unwrap_or(DEFAULT_CONTEXT), None, None)
        };
        eprintln!("compiling {} for {context} tokens of context, {slots} conversations in cache ...", weights.display());
        let started = std::time::Instant::now();
        let pool = match option(&arguments, "--pool")? {
            Some(size) => Some(dafo_compiler_llm::host::engine::PoolConfig { size: size.parse()?, device_experts: option(&arguments, "--vram-experts")?.map(|value| value.parse()).transpose()?, profile: option(&arguments, "--profile")?.map(Into::into), profiles_dir: option(&arguments, "--profiles")?.map(Into::into), adaptive: arguments.iter().any(|argument| argument == "--adaptive"), debug: arguments.iter().any(|argument| argument == "--debug") }),
            None => None,
        };
        let projector: Option<std::path::PathBuf> = match option(&arguments, "--projector")? {
            Some(path) => Some(path.into()),
            // An `mmproj-*.gguf` beside the model is its vision projector.
            None if stored_projector.is_some() => stored_projector.clone(),
            None if artifact::is_artifact(&model) => None,
            None => weights.parent().and_then(|directory| std::fs::read_dir(directory).ok()).and_then(|entries| {
                entries.filter_map(|entry| entry.ok().map(|entry| entry.path())).find(|path| path.file_name().is_some_and(|name| name.to_string_lossy().starts_with("mmproj") && name.to_string_lossy().ends_with(".gguf")))
            }),
        };
        let projector = if cfg!(feature = "vision") { projector } else { None };
        if let Some(path) = &projector {
            eprintln!("vision projector: {}", path.display());
        }
        let visual_tokens: usize = option(&arguments, "--visual-tokens")?.map(|value| value.parse()).transpose()?.or(stored_tokens).unwrap_or(560);
        let engine = Engine::start_with(&weights, tokenizer, context, slots, dafo_compiler_llm::host::engine::EngineOptions { pool, projector, visual_tokens })?;
        eprintln!("ready in {:.1} s on {} ({:.2} GiB of weights mapped)", started.elapsed().as_secs_f64(), engine.device, engine.weight_bytes as f64 / (1u64 << 30) as f64);

        let server = Server::http(format!("{host}:{port}")).map_err(|error| anyhow::anyhow!("cannot listen on {host}:{port}: {error}"))?;
        let state = Arc::new(State { engine, name, token, counter: AtomicU64::new(0) });
        eprintln!("serving {} on http://{host}:{port}", state.name);
        eprintln!("  chat page:  http://{host}:{port}/#token={}", state.token);
        eprintln!("  API:        http://{host}:{port}/v1/chat/completions");
        if generated {
            eprintln!("  token (made for this run; pass --token or DAFO_SERVE_TOKEN to choose it): {}", state.token);
        } else {
            eprintln!("  token: as given");
        }
        if arguments.iter().any(|argument| argument == "--open") {
            let link = format!("http://{host}:{port}/#token={}", state.token);
            let _ = std::process::Command::new(if cfg!(target_os = "macos") { "open" } else { "xdg-open" }).arg(link).spawn();
        }
        for request in server.incoming_requests() {
            let state = Arc::clone(&state);
            std::thread::spawn(move || handle(&state, request));
        }
        Ok(())
    }

    fn header(name: &str, value: &str) -> Header {
        Header::from_bytes(name.as_bytes(), value.as_bytes()).expect("valid header")
    }

    fn reply(request: Request, status: u16, content_type: &str, body: String) {
        let response = Response::from_string(body)
            .with_status_code(StatusCode(status))
            .with_header(header("Content-Type", content_type))
            .with_header(header("Access-Control-Allow-Origin", "*"));
        let _ = request.respond(response);
    }

    fn error(request: Request, status: u16, code: &str, message: &str) {
        let kind = if status >= 500 { "server_error" } else { "invalid_request_error" };
        reply(request, status, "application/json", json!({ "error": { "message": message, "type": kind, "code": code } }).to_string());
    }

    fn authorised(state: &State, request: &Request) -> bool {
        let given = request
            .headers()
            .iter()
            .find(|header| header.field.equiv("Authorization"))
            .and_then(|header| header.value.as_str().strip_prefix("Bearer "))
            .unwrap_or("");
        // Constant time in the length of the expected token.
        given.len() == state.token.len() && given.bytes().zip(state.token.bytes()).fold(0u8, |difference, (a, b)| difference | (a ^ b)) == 0
    }

    fn handle(state: &State, mut request: Request) {
        let path = request.url().split('?').next().unwrap_or("").to_owned();
        let method = request.method().clone();
        if method == Method::Options {
            let response = Response::empty(StatusCode(204))
                .with_header(header("Access-Control-Allow-Origin", "*"))
                .with_header(header("Access-Control-Allow-Headers", "Authorization, Content-Type"))
                .with_header(header("Access-Control-Allow-Methods", "GET, POST, OPTIONS"));
            let _ = request.respond(response);
            return;
        }
        match (&method, path.as_str()) {
            (Method::Get, "/") => reply(request, 200, "text/html; charset=utf-8", CHAT_PAGE.replace("__MODEL__", &state.name)),
            (Method::Get, "/health") => reply(request, 200, "application/json", json!({ "status": "ok", "model": state.name, "context": state.engine.context }).to_string()),
            (Method::Get, "/v1/models") => {
                if !authorised(state, &request) {
                    return error(request, 401, "invalid_api_key", "missing or wrong bearer token");
                }
                reply(request, 200, "application/json", json!({ "object": "list", "data": [{ "id": state.name, "object": "model", "created": now(), "owned_by": "dafo-compiler-llm", "context_length": state.engine.context }] }).to_string());
            }
            (Method::Post, "/v1/chat/completions") => {
                if !authorised(state, &request) {
                    return error(request, 401, "invalid_api_key", "missing or wrong bearer token");
                }
                let mut body = String::new();
                if request.as_reader().take(MAX_BODY).read_to_string(&mut body).is_err() {
                    return error(request, 400, "invalid_body", "the body is not UTF-8 text");
                }
                match serde_json::from_str::<Value>(&body) {
                    Ok(value) => chat_completion(state, request, value),
                    Err(problem) => error(request, 400, "invalid_json", &format!("the body is not JSON: {problem}")),
                }
            }
            (Method::Get, "/v1/experts") => {
                if !authorised(state, &request) {
                    return error(request, 401, "invalid_api_key", "missing or wrong bearer token");
                }
                reply(request, 200, "application/json", {
                    let status = state.engine.adaptive.lock().unwrap().clone();
                    json!({ "profiles": state.engine.profiles, "current": *state.engine.current_profile.lock().unwrap(), "mode": format!("{:?}", status.mode).to_lowercase(), "last_swaps": status.last_swaps, "last_miss": status.last_miss, "requests": status.requests, "prefill_changed_percent": status.prefill_changed_percent, "changed_percent": status.changed_percent, "per_layer": status.per_layer }).to_string()
                });
            }
            (Method::Post, "/v1/experts") => {
                if !authorised(state, &request) {
                    return error(request, 401, "invalid_api_key", "missing or wrong bearer token");
                }
                let mut body = String::new();
                if request.as_reader().take(MAX_BODY).read_to_string(&mut body).is_err() {
                    return error(request, 400, "invalid_body", "the body is not UTF-8 text");
                }
                let value: Value = match serde_json::from_str(&body) {
                    Ok(value) => value,
                    Err(problem) => return error(request, 400, "invalid_json", &format!("the body is not JSON: {problem}")),
                };
                // {"layers": {"0": [ids], ...}, "name": "mine"} sets the experts directly; {"profile": "name"} switches to a stored one;
                // {"adaptive": true|false} starts or stops the exchange of experts by demand; {"freeze": true, "save_as": "name"} settles the pool.
                let outcome = if let Some(on) = value.get("adaptive").and_then(Value::as_bool) {
                    state.engine.adaptive_command(dafo_compiler_llm::host::engine::AdaptiveCommand::Enable(on))
                } else if value.get("freeze").and_then(Value::as_bool) == Some(true) {
                    state.engine.adaptive_command(dafo_compiler_llm::host::engine::AdaptiveCommand::Freeze { save_as: value.get("save_as").and_then(Value::as_str).map(str::to_owned) })
                } else if let Some(layers) = value.get("layers").and_then(Value::as_object) {
                    let parsed: Option<Vec<(usize, Vec<usize>)>> = layers
                        .iter()
                        .map(|(layer, ids)| Some((layer.parse().ok()?, ids.as_array()?.iter().map(|id| id.as_u64().map(|id| id as usize)).collect::<Option<Vec<_>>>()?)))
                        .collect();
                    match parsed {
                        Some(layers) => state.engine.set_experts(value.get("name").and_then(Value::as_str).unwrap_or("custom").to_owned(), layers),
                        None => return error(request, 400, "invalid_layers", "layers must be {\"layer\": [expert ids]}"),
                    }
                } else if let Some(name) = value.get("profile").and_then(Value::as_str) {
                    match state.engine.profile_layers(name) {
                        Some(layers) => state.engine.set_experts(name.to_owned(), layers),
                        None => return error(request, 404, "unknown_experts_profile", &format!("no experts profile named {name}")),
                    }
                } else {
                    return error(request, 400, "invalid_request", "send {\"profile\": name} or {\"layers\": {...}}");
                };
                match outcome {
                    Ok(note) => reply(request, 200, "application/json", json!({ "ok": true, "note": note }).to_string()),
                    Err(problem) => error(request, 400, "experts_not_set", &format!("{problem:#}")),
                }
            }
            _ => error(request, 404, "not_found", "unknown route"),
        }
    }

    fn now() -> u64 {
        SystemTime::now().duration_since(UNIX_EPOCH).map(|elapsed| elapsed.as_secs()).unwrap_or(0)
    }

    fn string_or_list(value: Option<&Value>) -> Vec<String> {
        match value {
            Some(Value::String(text)) => vec![text.clone()],
            Some(Value::Array(items)) => items.iter().filter_map(|item| item.as_str().map(str::to_owned)).collect(),
            _ => Vec::new(),
        }
    }

    fn chat_completion(state: &State, request: Request, body: Value) {
        let Some(messages) = body.get("messages").and_then(Value::as_array).filter(|messages| !messages.is_empty()) else {
            return error(request, 400, "invalid_messages", "`messages` must be a non-empty list");
        };
        let tools: Vec<Value> = if body.get("tool_choice").and_then(Value::as_str) == Some("none") { Vec::new() } else { body.get("tools").and_then(Value::as_array).cloned().unwrap_or_default() };
        let allowed: HashSet<String> = tools.iter().filter_map(|tool| tool["function"]["name"].as_str().map(str::to_owned)).collect();
        let thinking = body.get("think").and_then(Value::as_bool).unwrap_or(false);
        let prompt = match chat::render(messages, &tools, thinking) {
            Ok(prompt) => prompt,
            Err(message) => return error(request, 400, "invalid_messages", &message),
        };
        let number = |key: &str| body.get(key).and_then(Value::as_f64);
        let defaults = Sampling::default();
        let sampling = Sampling {
            temperature: number("temperature").unwrap_or(defaults.temperature),
            top_k: number("top_k").map(|value| value as usize).unwrap_or(defaults.top_k),
            top_p: number("top_p").unwrap_or(defaults.top_p),
            repeat_penalty: number("repeat_penalty").map(|value| value as f32).unwrap_or(defaults.repeat_penalty),
            repeat_last_n: number("repeat_last_n").map(|value| value as usize).unwrap_or(defaults.repeat_last_n),
            seed: number("seed").map(|value| value as u64).unwrap_or(defaults.seed),
        };
        let max_tokens = number("max_completion_tokens").or_else(|| number("max_tokens")).map(|value| value.max(1.0) as usize).unwrap_or(2048);
        let settings = Settings { sampling, max_tokens, stop: string_or_list(body.get("stop")) };
        let wants_stream = body.get("stream").and_then(Value::as_bool).unwrap_or(false);
        // With tools declared the answer is read whole (a call is only known once it is complete), then sent as one chunk if streaming.
        let stream_now = wants_stream && tools.is_empty();
        let (events, receiver) = mpsc::channel();
        let cancel = Arc::new(AtomicBool::new(false));
        let profile = body.get("dafo_experts").and_then(Value::as_str).map(str::to_owned);
        let images = match chat::images_of(messages) {
            Ok(images) => images,
            Err(message) => return error(request, 400, "invalid_image", &message),
        };
        if !images.is_empty() && !state.engine.vision {
            return error(request, 400, "images_not_supported", "this server has no vision projector: start it with --projector mmproj-....gguf (a build with --features vision)");
        }
        let job = Job { prompt, settings, stream: stream_now, cancel: Arc::clone(&cancel), events, profile, images };
        if state.engine.submit(job).is_err() {
            return error(request, 500, "engine_stopped", "the engine stopped");
        }
        let id = format!("chatcmpl-{:x}{:04x}", now(), state.counter.fetch_add(1, Ordering::Relaxed) & 0xffff);
        if stream_now {
            let reader = SseReader { events: receiver, buffer: Vec::new(), position: 0, id, model: state.name.clone(), created: now(), started: false, finished: false, cancel };
            let response = Response::new(StatusCode(200), vec![header("Content-Type", "text/event-stream"), header("Cache-Control", "no-cache"), header("Access-Control-Allow-Origin", "*")], reader, None, None);
            let _ = request.respond(response);
            return;
        }
        // Wait for the whole answer.
        let outcome = loop {
            match receiver.recv() {
                Ok(Event::Delta(_)) => {}
                Ok(done) => break done,
                Err(_) => break Event::Failed { code: "engine_stopped", message: "the engine stopped".into() },
            }
        };
        match outcome {
            Event::Failed { code, message } => error(request, if code == "context_length_exceeded" { 400 } else { 500 }, code, &message),
            Event::Done { text, finish, prompt_tokens, completion_tokens } => {
                let (content, calls) = if allowed.is_empty() { (chat::clean(&text), Vec::new()) } else { chat::parse_answer(&text, &allowed) };
                let calls_json: Vec<Value> = calls
                    .iter()
                    .enumerate()
                    .map(|(index, call)| json!({ "index": index, "id": format!("call_{}_{index}", &id[9..]), "type": "function", "function": { "name": call.name, "arguments": call.arguments } }))
                    .collect();
                let finish = if calls.is_empty() { finish } else { "tool_calls" };
                let usage = json!({ "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "total_tokens": prompt_tokens + completion_tokens });
                if wants_stream {
                    let chunk = |delta: Value, finish: Value| format!("data: {}\n\n", json!({ "id": id, "object": "chat.completion.chunk", "created": now(), "model": state.name, "choices": [{ "index": 0, "delta": delta, "finish_reason": finish }] }));
                    let mut out = chunk(json!({ "role": "assistant", "content": content }), Value::Null);
                    if !calls_json.is_empty() {
                        out.push_str(&chunk(json!({ "tool_calls": calls_json }), Value::Null));
                    }
                    out.push_str(&chunk(json!({}), json!(finish)));
                    out.push_str(&format!("data: {}\n\ndata: [DONE]\n\n", json!({ "id": id, "object": "chat.completion.chunk", "created": now(), "model": state.name, "choices": [], "usage": usage })));
                    let response = Response::from_string(out).with_header(header("Content-Type", "text/event-stream")).with_header(header("Access-Control-Allow-Origin", "*"));
                    let _ = request.respond(response);
                } else {
                    let mut message = json!({ "role": "assistant", "content": if content.is_empty() && !calls_json.is_empty() { Value::Null } else { json!(content) } });
                    if !calls_json.is_empty() {
                        message["tool_calls"] = json!(calls_json);
                    }
                    reply(request, 200, "application/json", json!({ "id": id, "object": "chat.completion", "created": now(), "model": state.name, "choices": [{ "index": 0, "message": message, "finish_reason": finish }], "usage": usage }).to_string());
                }
            }
            Event::Delta(_) => unreachable!("deltas are skipped above"),
        }
    }

    /// Server-sent events produced as the engine generates; dropping it (the client left) cancels the request.
    struct SseReader {
        events: Receiver<Event>,
        buffer: Vec<u8>,
        position: usize,
        id: String,
        model: String,
        created: u64,
        started: bool,
        finished: bool,
        cancel: Arc<AtomicBool>,
    }

    impl SseReader {
        fn chunk(&self, delta: Value, finish: Value) -> String {
            format!("data: {}\n\n", json!({ "id": self.id, "object": "chat.completion.chunk", "created": self.created, "model": self.model, "choices": [{ "index": 0, "delta": delta, "finish_reason": finish }] }))
        }

        fn refill(&mut self) {
            self.buffer.clear();
            self.position = 0;
            if !self.started {
                self.started = true;
                self.buffer.extend(self.chunk(json!({ "role": "assistant", "content": "" }), Value::Null).into_bytes());
                return;
            }
            if self.finished {
                return;
            }
            let text = match self.events.recv() {
                Ok(Event::Delta(text)) => self.chunk(json!({ "content": text }), Value::Null),
                Ok(Event::Done { text: _, finish, prompt_tokens, completion_tokens }) => {
                    self.finished = true;
                    let usage = json!({ "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "total_tokens": prompt_tokens + completion_tokens });
                    format!(
                        "{}data: {}\n\ndata: [DONE]\n\n",
                        self.chunk(json!({}), json!(finish)),
                        json!({ "id": self.id, "object": "chat.completion.chunk", "created": self.created, "model": self.model, "choices": [], "usage": usage })
                    )
                }
                Ok(Event::Failed { code, message }) => {
                    self.finished = true;
                    format!("data: {}\n\ndata: [DONE]\n\n", json!({ "error": { "message": message, "type": "server_error", "code": code } }))
                }
                Err(_) => {
                    self.finished = true;
                    "data: [DONE]\n\n".to_owned()
                }
            };
            self.buffer.extend(text.into_bytes());
        }
    }

    impl Read for SseReader {
        fn read(&mut self, out: &mut [u8]) -> std::io::Result<usize> {
            if self.position >= self.buffer.len() {
                self.refill();
                if self.buffer.is_empty() {
                    return Ok(0);
                }
            }
            let count = out.len().min(self.buffer.len() - self.position);
            out[..count].copy_from_slice(&self.buffer[self.position..self.position + count]);
            self.position += count;
            Ok(count)
        }
    }

    impl Drop for SseReader {
        fn drop(&mut self) {
            self.cancel.store(true, Ordering::Release);
        }
    }

}
