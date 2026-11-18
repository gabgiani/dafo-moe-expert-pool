//! One thread that owns the compiled model and serves requests one after another, keeping several conversations in cache.
//!
//! A request goes to the cached conversation that shares the longest start with its prompt (a follow-up reads only what is new);
//! otherwise to an empty slot, otherwise to the one used longest ago. Requests are served one at a time: decoding two at once
//! would not be faster on a chip whose generation is limited by memory bandwidth.

use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};

use anyhow::{Context, Result};
use tokenizers::Tokenizer;

use super::sampler::{Sampler, Sampling};
use crate::backend_metal::{self, RING_SLACK};
use crate::frontends;
use crate::gguf::{Gguf, Scalar};
use crate::runtime::Runtime;

/// Set by a program that holds a conversation in the terminal, so the engine's progress lines do not break into it.
pub static QUIET: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

macro_rules! note {
    ($($argument:tt)*) => {
        if !QUIET.load(std::sync::atomic::Ordering::Relaxed) {
            eprintln!($($argument)*);
        }
    };
}

/// Tokens read per turn of the engine loop, so a cancelled request is noticed within a slice.
const READ_SLICE: usize = 512;

#[derive(Clone, Debug)]
pub struct Settings {
    pub sampling: Sampling,
    pub max_tokens: usize,
    pub stop: Vec<String>,
}

pub enum Event {
    /// New text of the answer (only when the request streams).
    Delta(String),
    Done { text: String, finish: &'static str, prompt_tokens: usize, completion_tokens: usize },
    Failed { code: &'static str, message: String },
}

pub struct Job {
    pub prompt: String,
    pub settings: Settings,
    pub stream: bool,
    pub cancel: Arc<AtomicBool>,
    pub events: Sender<Event>,
    /// A profile (by name) the experts must be switched to before this request; `None` keeps what the pool holds.
    pub profile: Option<String>,
    /// The images of the prompt, in order (each stands at an image marker of the text).
    pub images: Vec<Vec<u8>>,
}

/// Optional parts of an engine.
#[derive(Clone, Debug, Default)]
pub struct EngineOptions {
    pub pool: Option<PoolConfig>,
    /// The vision projector (`mmproj-*.gguf`): with it the engine reads images (needs the `vision` feature).
    pub projector: Option<std::path::PathBuf>,
    /// Most soft tokens one image may use (70, 140, 280, 560 or 1120).
    pub visual_tokens: usize,
}

pub struct Engine {
    jobs: Sender<Message>,
    /// Names of the profiles loaded from the profiles folder, and the one the pool holds now.
    pub profiles: Vec<String>,
    stored: std::collections::BTreeMap<String, Profile>,
    pub current_profile: Arc<std::sync::Mutex<String>>,
    pub adaptive: Arc<std::sync::Mutex<AdaptiveStatus>>,
    pub context: usize,
    pub slots: usize,
    pub device: String,
    pub weight_bytes: usize,
    /// Whether the engine reads images.
    pub vision: bool,
}

struct Slot {
    history: Vec<u32>,
    last_used: u64,
    /// A request is running in this conversation.
    busy: bool,
}

enum Stage {
    Reading,
    Generating,
}

/// A request in progress: it takes one turn of the engine at a time (a slice of its prompt, or one token of its answer), so several run at once.
struct Active {
    job: Job,
    slot: usize,
    ids: Vec<u32>,
    runs: Vec<ImageRun>,
    cursor: usize,
    reused: usize,
    max_tokens: usize,
    stage: Stage,
    sampler: Sampler,
    generated: Vec<u32>,
    text: String,
    emitted: usize,
    hold: usize,
    logits: Vec<f32>,
    started: std::time::Instant,
}

/// Serve a model with more experts than fit in memory: the expert tensors live in a pool of `size` experts per layer that is filled from
/// the model file, with the experts of `profile` (a JSON `{"layer": [expert ids]}`, as written by scripts/dev/moe-trace-stats.py) when given.
#[derive(Clone, Debug)]
pub struct PoolConfig {
    /// Experts per layer kept resident in all.
    pub size: usize,
    /// Of those, how many sit in device memory (the rest in pinned host memory, read over PCIe; CUDA only). `None`: all of them.
    pub device_experts: Option<usize>,
    /// The experts to load at start.
    pub profile: Option<std::path::PathBuf>,
    /// The experts to start with when there is no profile file (hottest first).
    pub initial: Option<Profile>,
    /// A folder of `name.json` profiles a request (or `POST /v1/experts`) can switch to on the fly.
    pub profiles_dir: Option<std::path::PathBuf>,
    /// Let the engine move experts in and out of the pool by what the requests ask for, until it settles (see [`Adaptive`]).
    pub adaptive: bool,
    /// Report, after every request, which share of the pool was exchanged (and per layer).
    pub debug: bool,
}

pub use crate::pool::{AdaptiveStatus, Mode, Profile};
use crate::pool::AdaptivePool;

/// A command for the adaptive pool.
#[derive(Clone, Debug)]
pub enum AdaptiveCommand {
    /// Start moving experts again (`true`), or leave the pool as it is (`false`).
    Enable(bool),
    /// Settle now on what the pool holds; saved under `save_as` in the profiles folder when given.
    Freeze { save_as: Option<String> },
}

/// What the engine receives: a chat job, or a change of the experts held in the pool (done between two requests).
pub enum Message {
    Job(Job),
    SetExperts { name: String, layers: Profile, reply: Sender<std::result::Result<String, String>> },
    Adaptive { command: AdaptiveCommand, reply: Sender<std::result::Result<String, String>> },
}

/// Reads a profile: the experts to keep per layer.
pub fn read_profile(path: &Path) -> Result<Vec<(usize, Vec<usize>)>> {
    let value: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(path).with_context(|| format!("cannot read {}", path.display()))?)?;
    let object = value.as_object().context("a profile is a JSON object {layer: [experts]}")?;
    object
        .iter()
        .map(|(layer, ids)| {
            let ids = ids.as_array().context("a layer holds a list of expert ids")?.iter().map(|id| id.as_u64().map(|id| id as usize).context("an expert id")).collect::<Result<Vec<_>>>()?;
            Ok((layer.parse::<usize>().context("a layer number")?, ids))
        })
        .collect()
}

impl Engine {
    /// Compiles `weights` for `context` tokens and starts serving on a thread.
    pub fn start(weights: &Path, tokenizer: Tokenizer, context: usize, slots: usize) -> Result<Self> {
        Self::start_with(weights, tokenizer, context, slots, EngineOptions::default())
    }

    /// Like [`Engine::start`]; with a pool the model's expert tensors are kept in a pool of experts that can be changed (see [`PoolConfig`]).
    pub fn start_with_pool(weights: &Path, tokenizer: Tokenizer, context: usize, slots: usize, pool: Option<PoolConfig>) -> Result<Self> {
        Self::start_with(weights, tokenizer, context, slots, EngineOptions { pool, ..Default::default() })
    }

    /// The general form: a pool, a vision projector.
    pub fn start_with(weights: &Path, tokenizer: Tokenizer, context: usize, slots: usize, options: EngineOptions) -> Result<Self> {
        let EngineOptions { pool, projector, visual_tokens } = options;
        let visual_tokens = if visual_tokens == 0 { 560 } else { visual_tokens };
        anyhow::ensure!(visual_tokens <= backend_metal::VISION_BATCH || projector.is_none(), "the plan that reads images holds {} soft tokens: --visual-tokens {visual_tokens} does not fit (use 70, 140, 280 or 560)", backend_metal::VISION_BATCH);
        let weights = weights.to_path_buf();
        let (jobs, receiver) = mpsc::channel::<Message>();
        let mut profiles: std::collections::BTreeMap<String, Profile> = Default::default();
        if let Some(directory) = pool.as_ref().and_then(|pool| pool.profiles_dir.as_ref()) {
            for entry in std::fs::read_dir(directory).with_context(|| format!("cannot read {}", directory.display()))? {
                let path = entry?.path();
                if path.extension().is_some_and(|extension| extension == "json") {
                    let name = path.file_stem().map(|stem| stem.to_string_lossy().into_owned()).unwrap_or_default();
                    profiles.insert(name, read_profile(&path)?);
                }
            }
        }
        let names: Vec<String> = profiles.keys().cloned().collect();
        let stored = profiles.clone();
        let current = Arc::new(std::sync::Mutex::new(pool.as_ref().and_then(|pool| pool.profile.as_ref()).map(|path| path.file_stem().map(|stem| stem.to_string_lossy().into_owned()).unwrap_or_default()).unwrap_or_default()));
        let shared_current = Arc::clone(&current);
        let mut adaptive_pool = AdaptivePool::new(0, pool.as_ref().is_some_and(|pool| pool.adaptive), pool.as_ref().is_some_and(|pool| pool.debug));
        adaptive_pool.set_logger(|line| note!("{line}"));
        let status = Arc::clone(&adaptive_pool.status);
        let profiles_dir = pool.as_ref().and_then(|pool| pool.profiles_dir.clone());
        let (ready_sender, ready) = mpsc::channel::<Result<(String, usize, bool)>>();
        std::thread::Builder::new().name("dafo-engine".into()).spawn(move || {
            let built = (|| -> Result<(Arc<Gguf>, Runtime, Option<Vision>)> {
                let gguf = Arc::new(Gguf::open(&weights)?);
                let mut graph = frontends::token_graph(&gguf)?;
                if let Some(pool) = &pool {
                    backend_metal::use_expert_pool(&mut graph, pool.size);
                }
                let vision = match &projector {
                    Some(path) => Some(open_vision(path, visual_tokens)?),
                    None => None,
                };
                let plans = backend_metal::compile_plans_for(backend_metal::Target::native(), &graph, &gguf, context, vision.is_some())?;
                let mut runtime = Runtime::with_expert_tiers(&gguf, &plans, slots, pool.as_ref().map(|pool| match pool.device_experts {
                    Some(device) if device < pool.size => (device, pool.size - device),
                    _ => (pool.size, 0),
                }))?;
                if let Some(initial) = pool.as_ref().filter(|pool| pool.profile.is_none()).and_then(|pool| pool.initial.as_ref()) {
                    let total = gguf.tensor("blk.0.ffn_gate_inp.weight").context("a mixture-of-experts model")?.dims[1] as usize;
                    let started = std::time::Instant::now();
                    let mut copied = 0;
                    for (layer, ids) in initial {
                        copied += runtime.set_allowed_experts(*layer, ids, total)?;
                    }
                    note!("starting experts loaded into the pool: {copied} copied in {:.1} s", started.elapsed().as_secs_f64());
                }
                if let Some(profile) = pool.as_ref().and_then(|pool| pool.profile.as_ref()) {
                    let total = gguf.tensor("blk.0.ffn_gate_inp.weight").context("a mixture-of-experts model")?.dims[1] as usize;
                    let started = std::time::Instant::now();
                    let mut copied = 0;
                    for (layer, ids) in read_profile(profile)? {
                        copied += runtime.set_allowed_experts(layer, &ids, total)?;
                    }
                    note!("experts of {} loaded into the pool: {copied} copied in {:.1} s", profile.display(), started.elapsed().as_secs_f64());
                }
                Ok((gguf, runtime, vision))
            })();
            match built {
                Ok((gguf, runtime, vision)) => {
                    let _ = ready_sender.send(Ok((runtime.device_name(), runtime.weight_bytes(), vision.is_some())));
                    let total = gguf.tensor("blk.0.ffn_gate_inp.weight").map(|tensor| tensor.dims[1] as usize).unwrap_or(0);
                    let mut adaptive_pool = adaptive_pool;
                    adaptive_pool.set_total(total);
                    Worker::new(&gguf, runtime, tokenizer, context, slots, profiles, shared_current, adaptive_pool, profiles_dir, vision).run(receiver);
                }
                Err(error) => {
                    let _ = ready_sender.send(Err(error));
                }
            }
        })?;
        let (device, weight_bytes, vision) = ready.recv().context("the engine thread stopped while loading")??;
        Ok(Self { jobs, profiles: names, stored, current_profile: current, adaptive: status, context, slots, device, weight_bytes, vision })
    }

    pub fn submit(&self, job: Job) -> Result<()> {
        self.jobs.send(Message::Job(job)).map_err(|_| anyhow::anyhow!("the engine stopped"))
    }

    pub fn profile_layers(&self, name: &str) -> Option<Profile> {
        self.stored.get(name).cloned()
    }

    /// Starts or stops the exchange of experts, or settles the pool on what it holds now.
    pub fn adaptive_command(&self, command: AdaptiveCommand) -> Result<String> {
        let (reply, answer) = mpsc::channel();
        self.jobs.send(Message::Adaptive { command, reply }).map_err(|_| anyhow::anyhow!("the engine stopped"))?;
        answer.recv().context("the engine stopped")?.map_err(anyhow::Error::msg)
    }

    /// Changes the experts the pool holds (after the request being served, before the next); returns what it did.
    pub fn set_experts(&self, name: String, layers: Profile) -> Result<String> {
        let (reply, answer) = mpsc::channel();
        self.jobs.send(Message::SetExperts { name, layers, reply }).map_err(|_| anyhow::anyhow!("the engine stopped"))?;
        answer.recv().context("the engine stopped")?.map_err(anyhow::Error::msg)
    }
}

/// The vision encoder, when the engine was built with the `vision` feature.
#[cfg(feature = "vision")]
type Vision = crate::vision::VisionEncoder;
#[cfg(not(feature = "vision"))]
enum Vision {}

#[cfg(feature = "vision")]
fn open_vision(path: &Path, visual_tokens: usize) -> Result<Vision> {
    crate::vision::VisionEncoder::open(path, visual_tokens)
}
#[cfg(not(feature = "vision"))]
fn open_vision(_: &Path, _: usize) -> Result<Vision> {
    anyhow::bail!("this build has no vision encoder: build with --features vision")
}

/// Marks a token id that stands for a soft token of an image (the id carries the image's digest, so a conversation that shows the same
/// image again finds it in the cache and a different one does not).
const IMAGE_TOKEN_FLAG: u32 = 0x8000_0000;
/// Images whose embeddings the engine keeps.
const IMAGE_CACHE: usize = 8;

/// The soft tokens of an image at some place of the prompt.
struct ImageRun {
    start: usize,
    len: usize,
    width: usize,
    embeddings: Arc<Vec<f32>>,
}

fn fnv1a(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325u64, |hash, byte| (hash ^ *byte as u64).wrapping_mul(0x0100_0000_01b3))
}

struct Worker {
    runtime: Runtime,
    tokenizer: Tokenizer,
    slots: Vec<Slot>,
    context: usize,
    stop_tokens: Vec<u32>,
    clock: u64,
    profiles: std::collections::BTreeMap<String, Profile>,
    current_profile: Arc<std::sync::Mutex<String>>,
    experts_total: usize,
    pool: AdaptivePool,
    profiles_dir: Option<std::path::PathBuf>,
    vision: Option<Vision>,
    /// Embeddings of the images seen lately, by hash of their bytes (oldest first).
    image_cache: Vec<(u64, Arc<Vec<f32>>, usize)>,
}

impl Worker {
    fn new(gguf: &Gguf, runtime: Runtime, tokenizer: Tokenizer, context: usize, slots: usize, profiles: std::collections::BTreeMap<String, Profile>, current_profile: Arc<std::sync::Mutex<String>>, pool: AdaptivePool, profiles_dir: Option<std::path::PathBuf>, vision: Option<Vision>) -> Self {
        let mut stop_tokens = Vec::new();
        if let Some(Scalar::Unsigned(eos)) = gguf.metadata("tokenizer.ggml.eos_token_id") {
            stop_tokens.push(*eos as u32);
        }
        stop_tokens.extend(["<turn|>", "<|tool_response>"].iter().filter_map(|token| tokenizer.token_to_id(token)));
        Self { runtime, tokenizer, slots: (0..slots).map(|_| Slot { history: Vec::new(), last_used: 0, busy: false }).collect(), context, stop_tokens, clock: 0, profiles, current_profile, experts_total: gguf.tensor("blk.0.ffn_gate_inp.weight").map(|tensor| tensor.dims[1] as usize).unwrap_or(0), pool, profiles_dir, vision, image_cache: Vec::new() }
    }

    /// Takes requests from `messages` and runs them, a turn each, until the channel closes: while one request reads its prompt or writes its
    /// answer the others advance too, each in a conversation (slot) of its own.
    fn run(mut self, messages: Receiver<Message>) {
        let mut waiting: std::collections::VecDeque<Job> = std::collections::VecDeque::new();
        let mut active: Vec<Active> = Vec::new();
        let mut closed = false;
        loop {
            if active.is_empty() && waiting.is_empty() {
                if closed {
                    return;
                }
                match messages.recv() {
                    Ok(message) => self.take(message, &mut waiting),
                    Err(_) => return,
                }
            }
            loop {
                match messages.try_recv() {
                    Ok(message) => self.take(message, &mut waiting),
                    Err(std::sync::mpsc::TryRecvError::Empty) => break,
                    Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                        closed = true;
                        break;
                    }
                }
            }
            // Start the waiting requests a free conversation can take.
            while let Some(job) = waiting.front() {
                if job.cancel.load(Ordering::Acquire) {
                    waiting.pop_front();
                    continue;
                }
                if self.slots.iter().all(|slot| slot.busy) {
                    break;
                }
                let job = waiting.pop_front().expect("a waiting request");
                if let Some(name) = &job.profile {
                    if let Err(message) = self.switch_profile(name) {
                        let _ = job.events.send(Event::Failed { code: "unknown_experts_profile", message });
                        continue;
                    }
                }
                let events = job.events.clone();
                match self.begin(job) {
                    Ok(Some(request)) => active.push(request),
                    Ok(None) => {}
                    Err(error) => {
                        let _ = events.send(Event::Failed { code: "engine_error", message: format!("{error:#}") });
                    }
                }
            }
            // One turn for each running request.
            let mut index = 0;
            while index < active.len() {
                match self.advance(&mut active[index]) {
                    Ok(false) => index += 1,
                    Ok(true) => {
                        let request = active.remove(index);
                        self.slots[request.slot].busy = false;
                        self.finish_request();
                    }
                    Err(error) => {
                        let request = active.remove(index);
                        self.slots[request.slot].busy = false;
                        // The conversation's cache may hold half a request: the next one starts over.
                        self.slots[request.slot].history.clear();
                        let _ = request.job.events.send(Event::Failed { code: "engine_error", message: format!("{error:#}") });
                        self.finish_request();
                    }
                }
            }
        }
    }

    fn take(&mut self, message: Message, waiting: &mut std::collections::VecDeque<Job>) {
        match message {
            Message::Job(job) => waiting.push_back(job),
            Message::Adaptive { command, reply } => {
                let result = self.command(command).map_err(|error| format!("{error:#}"));
                let _ = reply.send(result);
            }
            Message::SetExperts { name, layers, reply } => {
                let result = self.apply(&name, &layers).map_err(|error| format!("{error:#}"));
                let _ = reply.send(result);
            }
        }
    }

    /// What follows a request: the pool sees what the router wanted and may exchange experts or settle.
    fn finish_request(&mut self) {
        if let Some(block) = self.pool.after_request(&mut self.runtime) {
            let _ = self.save_profile("settled", &block);
            self.profiles.insert("settled".to_owned(), block);
            *self.current_profile.lock().unwrap() = "settled".to_owned();
        }
    }

    fn command(&mut self, command: AdaptiveCommand) -> Result<String> {
        let (_, layers) = self.runtime.pool_shape().context("the model is not served from an expert pool (--pool)")?;
        match command {
            AdaptiveCommand::Enable(on) => {
                self.pool.enable(on);
                Ok(format!("experts are {}", if on { "exchanged by what the requests ask for" } else { "left as they are" }))
            }
            AdaptiveCommand::Freeze { save_as } => {
                let block = self.pool.freeze(&self.runtime)?;
                let name = save_as.unwrap_or_else(|| "frozen".to_owned());
                let mut note = format!("the pool stays on its current block of experts ({layers} layers)");
                if let Some(path) = self.save_profile(&name, &block)? {
                    note.push_str(&format!("; saved as {}", path.display()));
                }
                self.profiles.insert(name.clone(), block);
                *self.current_profile.lock().unwrap() = name;
                Ok(note)
            }
        }
    }

    fn save_profile(&self, name: &str, block: &Profile) -> Result<Option<std::path::PathBuf>> {
        let Some(directory) = &self.profiles_dir else { return Ok(None) };
        let object: serde_json::Map<String, serde_json::Value> = block.iter().map(|(layer, ids)| (layer.to_string(), serde_json::Value::from(ids.clone()))).collect();
        let path = directory.join(format!("{name}.json"));
        std::fs::write(&path, serde_json::to_string(&object)?)?;
        Ok(Some(path))
    }

    fn switch_profile(&mut self, name: &str) -> std::result::Result<(), String> {
        if *self.current_profile.lock().unwrap() == name {
            return Ok(());
        }
        let layers = self.profiles.get(name).cloned().ok_or_else(|| format!("no experts profile named {name}"))?;
        self.apply(name, &layers).map(|_| ()).map_err(|error| format!("{error:#}"))
    }

    /// Puts the experts of `layers` in the pool. The conversations in cache stay valid: what they hold are keys and values, and the model
    /// changes only for the tokens that follow.
    fn apply(&mut self, name: &str, layers: &Profile) -> Result<String> {
        let started = std::time::Instant::now();
        let mut copied = 0;
        for (layer, ids) in layers {
            copied += self.runtime.set_allowed_experts(*layer, ids, self.experts_total)?;
        }
        *self.current_profile.lock().unwrap() = name.to_owned();
        let note = format!("experts profile {name}: {copied} experts ({:.0} MB) copied in {:.2} s", copied as f64 * 3.72, started.elapsed().as_secs_f64());
        note!("{note}");
        Ok(note)
    }

    /// The slot for `prompt` and how much of it the slot already holds.
    fn pick_slot(&self, prompt: &[u32]) -> (usize, usize) {
        (0..self.slots.len())
            .filter(|&index| !self.slots[index].busy)
            .map(|index| {
                let history = &self.slots[index].history;
                (index, reusable_prefix(history, prompt, RING_SLACK), history.is_empty(), self.slots[index].last_used)
            })
            .max_by_key(|&(_, reused, empty, last_used)| (reused, empty, std::cmp::Reverse(last_used)))
            .map(|(index, reused, _, _)| (index, reused))
            .expect("a free slot")
    }

    /// The embeddings of an image: encoded once, then kept for a while. Returns them with their number of tokens and width.
    fn encode_image(&mut self, bytes: &[u8]) -> Result<(Arc<Vec<f32>>, usize, usize, u64)> {
        let key = fnv1a(bytes);
        if let Some(index) = self.image_cache.iter().position(|(cached, _, _)| *cached == key) {
            let entry = self.image_cache.remove(index);
            let (embeddings, width) = (entry.1.clone(), entry.2);
            self.image_cache.push(entry);
            return Ok((embeddings.clone(), embeddings.len() / width, width, key));
        }
        #[cfg(feature = "vision")]
        {
            let vision = self.vision.as_ref().context("the engine has no vision projector")?;
            let started = std::time::Instant::now();
            let encoded = vision.encode(bytes)?;
            note!("image: {} soft tokens in {:.0} ms", encoded.tokens, started.elapsed().as_secs_f64() * 1000.0);
            let embeddings = Arc::new(encoded.embeddings);
            if self.image_cache.len() >= IMAGE_CACHE {
                self.image_cache.remove(0);
            }
            self.image_cache.push((key, embeddings.clone(), encoded.width));
            Ok((embeddings, encoded.tokens, encoded.width, key))
        }
        #[cfg(not(feature = "vision"))]
        {
            let _ = bytes;
            anyhow::bail!("this build has no vision encoder")
        }
    }

    /// The prompt as token ids and the images in it: each image marker becomes as many tokens as the image has soft tokens.
    fn tokenize(&mut self, job: &Job) -> Result<(Vec<u32>, Vec<ImageRun>)> {
        let ids = self.tokenizer.encode(job.prompt.as_str(), true).map_err(anyhow::Error::msg)?.get_ids().to_vec();
        if job.images.is_empty() {
            return Ok((ids, Vec::new()));
        }
        anyhow::ensure!(self.vision.is_some(), "this server was started without a vision projector, so it cannot read images (--projector mmproj-....gguf)");
        let image_token = self.tokenizer.token_to_id("<|image|>").context("the tokenizer does not declare <|image|>")?;
        let placeholders = ids.iter().filter(|&&token| token == image_token).count();
        anyhow::ensure!(placeholders == job.images.len(), "the prompt has {placeholders} image markers for {} images", job.images.len());
        let (mut prompt, mut runs, mut next) = (Vec::with_capacity(ids.len() + 560 * placeholders), Vec::new(), 0);
        for token in ids {
            if token != image_token {
                prompt.push(token);
                continue;
            }
            let (embeddings, tokens, width, key) = self.encode_image(&job.images[next])?;
            next += 1;
            let id = IMAGE_TOKEN_FLAG | ((key as u32) & !IMAGE_TOKEN_FLAG);
            runs.push(ImageRun { start: prompt.len(), len: tokens, width, embeddings });
            prompt.extend(std::iter::repeat_n(id, tokens));
        }
        Ok((prompt, runs))
    }

    /// Starts a request: reads its prompt into tokens (and images), picks a free conversation and sets up the turns. `None` when it was
    /// refused (the reason has been sent).
    fn begin(&mut self, job: Job) -> Result<Option<Active>> {
        let (ids, runs) = match self.tokenize(&job) {
            Ok(tokenized) => tokenized,
            Err(error) => {
                let _ = job.events.send(Event::Failed { code: "invalid_images", message: format!("{error:#}") });
                return Ok(None);
            }
        };
        if ids.len() + 1 > self.context {
            let _ = job.events.send(Event::Failed {
                code: "context_length_exceeded",
                message: format!("the conversation is {} tokens and the model was compiled for {}: start a new chat or use a larger context (--context)", ids.len(), self.context),
            });
            return Ok(None);
        }
        let max_tokens = job.settings.max_tokens.min(self.context - ids.len()).max(1);
        let (slot, reused) = self.pick_slot(&ids);
        self.clock += 1;
        self.slots[slot].last_used = self.clock;
        self.slots[slot].busy = true;
        self.slots[slot].history.truncate(reused);
        let cursor = self.slots[slot].history.len();
        let hold = job.settings.stop.iter().map(|stop| stop.len()).max().unwrap_or(0);
        let sampler = Sampler::new(job.settings.sampling.clone());
        Ok(Some(Active { job, slot, ids, runs, cursor, reused, max_tokens, stage: Stage::Reading, sampler, generated: Vec::new(), text: String::new(), emitted: 0, hold, logits: Vec::new(), started: std::time::Instant::now() }))
    }

    /// One turn of a request: a slice of its prompt, or one token of its answer. `Ok(true)` when it is over.
    fn advance(&mut self, request: &mut Active) -> Result<bool> {
        if request.job.cancel.load(Ordering::Acquire) {
            self.slots[request.slot].history.clear();
            return Ok(true);
        }
        self.runtime.select_slot(request.slot);
        match request.stage {
            Stage::Reading => {
                // The next image at or after the cursor: text is read up to it, and the image in one piece.
                let next_image = request.runs.iter().find(|run| run.start + run.len > request.cursor);
                let end = match next_image {
                    Some(run) if run.start == request.cursor => {
                        self.runtime.run_image(&run.embeddings, run.width, request.cursor)?;
                        let end = request.cursor + run.len;
                        self.slots[request.slot].history.extend_from_slice(&request.ids[request.cursor..end]);
                        request.cursor = end;
                        anyhow::ensure!(end < request.ids.len(), "the prompt ends with an image: there is nothing to answer");
                        return Ok(false);
                    }
                    Some(run) => run.start.min(request.cursor + READ_SLICE),
                    None => (request.cursor + READ_SLICE).min(request.ids.len()),
                };
                self.runtime.prefill(&request.ids[request.cursor..end], request.cursor)?;
                self.slots[request.slot].history.extend_from_slice(&request.ids[request.cursor..end]);
                request.cursor = end;
                if end == request.ids.len() {
                    // What the prompt asked of the router tells which experts the answer will want: settle them before decoding.
                    request.logits = self.runtime.logits();
                    self.pool.after_prompt(&mut self.runtime);
                    request.stage = Stage::Generating;
                    note!("request: {} prompt tokens ({} reused from conversation {}) in {:.1} s", request.ids.len(), request.reused, request.slot, request.started.elapsed().as_secs_f64());
                }
                Ok(false)
            }
            Stage::Generating => {
                let history = &self.slots[request.slot].history;
                let window = request.sampler.window();
                let recent = if window == 0 { &[][..] } else { &history[history.len().saturating_sub(window)..] };
                let token = request.sampler.sample(std::mem::take(&mut request.logits), recent);
                let mut finish = None;
                if self.stop_tokens.contains(&token) {
                    finish = Some("stop");
                } else {
                    request.generated.push(token);
                    request.text = self.tokenizer.decode(&request.generated, false).map_err(anyhow::Error::msg)?;
                    if let Some(at) = request.job.settings.stop.iter().filter_map(|stop| request.text.find(stop.as_str())).min() {
                        request.text.truncate(at);
                        finish = Some("stop");
                    } else {
                        if request.job.stream && !request.text.ends_with('\u{fffd}') {
                            let mut safe = request.text.len().saturating_sub(request.hold);
                            while !request.text.is_char_boundary(safe) {
                                safe -= 1;
                            }
                            if safe > request.emitted {
                                let _ = request.job.events.send(Event::Delta(request.text[request.emitted..safe].to_owned()));
                                request.emitted = safe;
                            }
                        }
                        if request.generated.len() >= request.max_tokens {
                            finish = Some("length");
                        }
                    }
                }
                if let Some(finish) = finish {
                    if request.job.stream && request.text.len() > request.emitted && request.text.is_char_boundary(request.emitted) {
                        let _ = request.job.events.send(Event::Delta(request.text[request.emitted..].to_owned()));
                    }
                    let text = std::mem::take(&mut request.text);
                    let _ = request.job.events.send(Event::Done { text, finish, prompt_tokens: request.ids.len(), completion_tokens: request.generated.len() });
                    return Ok(true);
                }
                let position = self.slots[request.slot].history.len();
                self.runtime.step(token, position)?;
                self.slots[request.slot].history.push(token);
                request.logits = self.runtime.logits();
                Ok(false)
            }
        }
    }

}

/// How many leading tokens of `prompt` the caches already hold: the longest start they share with `history`, as long as the caches
/// can go back to it (a sliding layer's ring keeps `slack` positions beyond its window) and at least one token is left to read.
fn reusable_prefix(history: &[u32], prompt: &[u32], slack: usize) -> usize {
    let common = history.iter().zip(prompt).take_while(|(a, b)| a == b).count().min(prompt.len().saturating_sub(1));
    if common > 0 && history.len() - common <= slack { common } else { 0 }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_follow_up_reuses_the_cache_and_a_distant_rewind_does_not() {
        let history: Vec<u32> = (0..1000).collect();
        let mut prompt = history.clone();
        prompt.extend([7, 8, 9]);
        assert_eq!(reusable_prefix(&history, &prompt, 640), 1000);
        assert_eq!(reusable_prefix(&history, &history[..100], 640), 0, "900 positions back is beyond the rings");
        assert_eq!(reusable_prefix(&history, &history, 640), 999, "one token is read again so the logits exist");
    }
}
