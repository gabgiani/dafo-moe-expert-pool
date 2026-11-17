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
}

struct Slot {
    history: Vec<u32>,
    last_used: u64,
}

/// Serve a model with more experts than fit in memory: the expert tensors live in a pool of `size` experts per layer that is filled from
/// the model file, with the experts of `profile` (a JSON `{"layer": [expert ids]}`, as written by scripts/dev/moe-trace-stats.py) when given.
#[derive(Clone, Debug)]
pub struct PoolConfig {
    pub size: usize,
    /// The experts to load at start.
    pub profile: Option<std::path::PathBuf>,
    /// A folder of `name.json` profiles a request (or `POST /v1/experts`) can switch to on the fly.
    pub profiles_dir: Option<std::path::PathBuf>,
    /// Let the engine move experts in and out of the pool by what the requests ask for, until it settles (see [`Adaptive`]).
    pub adaptive: bool,
    /// Report, after every request, which share of the pool was exchanged (and per layer).
    pub debug: bool,
}

/// Experts to keep per layer, by layer.
pub type Profile = Vec<(usize, Vec<usize>)>;

/// How the pool is kept: moving experts by what the router keeps asking for, or settled on a block.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Mode {
    /// The pool does not change by itself.
    Manual,
    /// After every request the experts the router wanted most replace the ones it wanted least, while that still changes things.
    Adapting,
    /// The pool settled: the same block of experts stays resident and nothing is exchanged any more.
    Frozen,
}

/// What the adaptive pool reports.
#[derive(Clone, Debug)]
pub struct AdaptiveStatus {
    pub mode: Mode,
    /// Experts replaced after the last request, and the share of what the router wanted that was outside the pool.
    pub last_swaps: usize,
    pub last_miss: f64,
    pub requests: u64,
    /// Share of the pool's slots (experts per layer × layers) exchanged after reading the last prompt, and after the last answer.
    pub prefill_changed_percent: f64,
    pub changed_percent: f64,
    /// Experts exchanged per layer after the last answer.
    pub per_layer: Vec<usize>,
}

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
        Self::start_with_pool(weights, tokenizer, context, slots, None)
    }

    /// Like [`Engine::start`]; with a pool the model's expert tensors are kept in a pool of experts that can be changed (see [`PoolConfig`]).
    pub fn start_with_pool(weights: &Path, tokenizer: Tokenizer, context: usize, slots: usize, pool: Option<PoolConfig>) -> Result<Self> {
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
        let status = Arc::new(std::sync::Mutex::new(AdaptiveStatus { mode: if pool.as_ref().is_some_and(|pool| pool.adaptive) { Mode::Adapting } else { Mode::Manual }, last_swaps: 0, last_miss: 0.0, requests: 0, prefill_changed_percent: 0.0, changed_percent: 0.0, per_layer: Vec::new() }));
        let shared_status = Arc::clone(&status);
        let profiles_dir = pool.as_ref().and_then(|pool| pool.profiles_dir.clone());
        let debug = pool.as_ref().is_some_and(|pool| pool.debug);
        let (ready_sender, ready) = mpsc::channel::<Result<(String, usize)>>();
        std::thread::Builder::new().name("dafo-engine".into()).spawn(move || {
            let built = (|| -> Result<(Arc<Gguf>, Runtime)> {
                let gguf = Arc::new(Gguf::open(&weights)?);
                let mut graph = frontends::token_graph(&gguf)?;
                if let Some(pool) = &pool {
                    backend_metal::use_expert_pool(&mut graph, pool.size);
                }
                let plans = backend_metal::compile_plans(&graph, &gguf, context, false)?;
                let mut runtime = Runtime::with_expert_pool(&gguf, &plans, slots, pool.as_ref().map(|pool| pool.size))?;
                if let Some(profile) = pool.as_ref().and_then(|pool| pool.profile.as_ref()) {
                    let total = gguf.tensor("blk.0.ffn_gate_inp.weight").context("a mixture-of-experts model")?.dims[1] as usize;
                    let started = std::time::Instant::now();
                    let mut copied = 0;
                    for (layer, ids) in read_profile(profile)? {
                        copied += runtime.set_allowed_experts(layer, &ids, total)?;
                    }
                    eprintln!("experts of {} loaded into the pool: {copied} copied in {:.1} s", profile.display(), started.elapsed().as_secs_f64());
                }
                Ok((gguf, runtime))
            })();
            match built {
                Ok((gguf, runtime)) => {
                    let _ = ready_sender.send(Ok((runtime.device_name(), runtime.weight_bytes())));
                    Worker::new(&gguf, runtime, tokenizer, context, slots, profiles, shared_current, shared_status, profiles_dir, debug).run(receiver);
                }
                Err(error) => {
                    let _ = ready_sender.send(Err(error));
                }
            }
        })?;
        let (device, weight_bytes) = ready.recv().context("the engine thread stopped while loading")??;
        Ok(Self { jobs, profiles: names, stored, current_profile: current, adaptive: status, context, slots, device, weight_bytes })
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
    status: Arc<std::sync::Mutex<AdaptiveStatus>>,
    profiles_dir: Option<std::path::PathBuf>,
    /// Per layer and expert: a running score of how much the router wanted it (each request weighs the same).
    score: Vec<Vec<f64>>,
    calm: usize,
    frozen_miss: f64,
    hot: usize,
    debug: bool,
}

/// Requests in a row that exchange (almost) nothing before the pool settles.
const CALM_REQUESTS: usize = 3;
/// An expert enters the pool only when it was wanted this much more than the one it replaces.
const SWAP_MARGIN: f64 = 1.25;
/// Share of the pool a request may exchange and still count as calm.
const CALM_SHARE: f64 = 0.01;
/// A settled pool resumes exchanging when this many requests in a row miss this much more than when it settled.
const THAW_REQUESTS: usize = 3;

impl Worker {
    fn new(gguf: &Gguf, runtime: Runtime, tokenizer: Tokenizer, context: usize, slots: usize, profiles: std::collections::BTreeMap<String, Profile>, current_profile: Arc<std::sync::Mutex<String>>, status: Arc<std::sync::Mutex<AdaptiveStatus>>, profiles_dir: Option<std::path::PathBuf>, debug: bool) -> Self {
        let mut stop_tokens = Vec::new();
        if let Some(Scalar::Unsigned(eos)) = gguf.metadata("tokenizer.ggml.eos_token_id") {
            stop_tokens.push(*eos as u32);
        }
        stop_tokens.extend(["<turn|>", "<|tool_response>"].iter().filter_map(|token| tokenizer.token_to_id(token)));
        Self { runtime, tokenizer, slots: (0..slots).map(|_| Slot { history: Vec::new(), last_used: 0 }).collect(), context, stop_tokens, clock: 0, profiles, current_profile, experts_total: gguf.tensor("blk.0.ffn_gate_inp.weight").map(|tensor| tensor.dims[1] as usize).unwrap_or(0), status, profiles_dir, score: Vec::new(), calm: 0, frozen_miss: 0.0, hot: 0, debug }
    }

    fn run(mut self, messages: Receiver<Message>) {
        for message in messages {
            match message {
                Message::Job(job) => {
                    if job.cancel.load(Ordering::Acquire) {
                        continue;
                    }
                    let events = job.events.clone();
                    if let Some(name) = &job.profile {
                        if let Err(message) = self.switch_profile(name) {
                            let _ = events.send(Event::Failed { code: "unknown_experts_profile", message });
                            continue;
                        }
                    }
                    if let Err(error) = self.serve(&job) {
                        let _ = events.send(Event::Failed { code: "engine_error", message: format!("{error:#}") });
                    }
                    self.adapt(false);
                }
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
    }

    fn command(&mut self, command: AdaptiveCommand) -> Result<String> {
        let (size, layers) = self.runtime.pool_shape().context("the model is not served from an expert pool (--pool)")?;
        let _ = size;
        match command {
            AdaptiveCommand::Enable(on) => {
                let mut status = self.status.lock().unwrap();
                status.mode = if on { Mode::Adapting } else { Mode::Manual };
                self.calm = 0;
                self.hot = 0;
                Ok(format!("experts are {}", if on { "exchanged by what the requests ask for" } else { "left as they are" }))
            }
            AdaptiveCommand::Freeze { save_as } => {
                self.status.lock().unwrap().mode = Mode::Frozen;
                let block: Profile = (0..layers).map(|layer| (layer, self.runtime.resident_experts(layer))).collect();
                let name = save_as.unwrap_or_else(|| "frozen".to_owned());
                let mut note = format!("the pool stays on its current block of experts ({layers} layers)");
                if let Some(directory) = &self.profiles_dir {
                    let object: serde_json::Map<String, serde_json::Value> = block.iter().map(|(layer, ids)| (layer.to_string(), serde_json::Value::from(ids.clone()))).collect();
                    let path = directory.join(format!("{name}.json"));
                    std::fs::write(&path, serde_json::to_string(&object)?)?;
                    note.push_str(&format!("; saved as {}", path.display()));
                }
                self.profiles.insert(name.clone(), block);
                *self.current_profile.lock().unwrap() = name;
                Ok(note)
            }
        }
    }

    /// After a request: see what the router wanted, exchange experts if that still helps, and settle when it stops helping.
    fn adapt(&mut self, during_prefill: bool) {
        let Some((size, layers)) = self.runtime.pool_shape() else { return };
        if during_prefill && self.status.lock().unwrap().mode != Mode::Adapting {
            return;
        }
        let total = self.experts_total;
        let demand = self.runtime.take_demand(layers, total);
        let mode = self.status.lock().unwrap().mode;
        let (mut wanted, mut missed) = (0u64, 0u64);
        let mut resident_sets: Vec<std::collections::HashSet<usize>> = Vec::with_capacity(layers);
        for layer in 0..layers {
            let resident: std::collections::HashSet<usize> = self.runtime.resident_experts(layer).into_iter().collect();
            for (expert, &count) in demand[layer].iter().enumerate() {
                wanted += count as u64;
                if !resident.contains(&expert) {
                    missed += count as u64;
                }
            }
            resident_sets.push(resident);
        }
        if wanted == 0 {
            return;
        }
        let miss = missed as f64 / wanted as f64;
        let mut swaps = 0;
        let mut per_layer = vec![0usize; layers];
        let slots_total = (size * layers).max(1) as f64;
        match mode {
            Mode::Manual => {}
            Mode::Frozen => {
                if miss > self.frozen_miss * 1.5 + 0.05 {
                    self.hot += 1;
                } else {
                    self.hot = 0;
                }
                if self.hot >= THAW_REQUESTS {
                    eprintln!("experts: the requests miss {:.0} % of what the router wants (was {:.0} % when settled): exchanging again", miss * 100.0, self.frozen_miss * 100.0);
                    self.status.lock().unwrap().mode = Mode::Adapting;
                    self.hot = 0;
                    self.calm = 0;
                }
            }
            Mode::Adapting => {
                if self.score.len() != layers {
                    self.score = vec![vec![0.0; total]; layers];
                }
                for layer in 0..layers {
                    let sum: f64 = demand[layer].iter().map(|&count| count as f64).sum::<f64>().max(1.0);
                    for expert in 0..total {
                        self.score[layer][expert] = 0.6 * self.score[layer][expert] + 0.4 * demand[layer][expert] as f64 / sum;
                    }
                    let mut resident: Vec<usize> = resident_sets[layer].iter().copied().collect();
                    let mut outside: Vec<usize> = (0..total).filter(|expert| !resident_sets[layer].contains(expert)).collect();
                    resident.sort_by(|a, b| self.score[layer][*a].partial_cmp(&self.score[layer][*b]).unwrap());
                    outside.sort_by(|a, b| self.score[layer][*b].partial_cmp(&self.score[layer][*a]).unwrap());
                    let mut next: std::collections::HashSet<usize> = resident_sets[layer].clone();
                    let mut changed = 0;
                    for (weakest, strongest) in resident.iter().zip(outside.iter()) {
                        if self.score[layer][*strongest] > SWAP_MARGIN * self.score[layer][*weakest] + 1e-6 {
                            next.remove(weakest);
                            next.insert(*strongest);
                            changed += 1;
                        } else {
                            break;
                        }
                    }
                    if changed > 0 {
                        let ids: Vec<usize> = next.into_iter().collect();
                        if self.runtime.set_allowed_experts(layer, &ids, total).is_ok() {
                            swaps += changed;
                            per_layer[layer] = changed;
                        }
                    }
                }
                if during_prefill {
                    let percent = 100.0 * swaps as f64 / slots_total;
                    eprintln!("experts: after reading the prompt the router wanted {:.1} % outside the pool; {swaps} experts exchanged before answering ({percent:.1} % of the pool changed)", miss * 100.0);
                    if self.debug {
                        eprintln!("  debug, per layer: {per_layer:?}");
                    }
                    self.status.lock().unwrap().prefill_changed_percent = percent;
                    return;
                }
                if swaps as f64 <= CALM_SHARE * (size * layers) as f64 {
                    self.calm += 1;
                } else {
                    self.calm = 0;
                }
                if self.calm >= CALM_REQUESTS {
                    self.status.lock().unwrap().mode = Mode::Frozen;
                    self.frozen_miss = miss;
                    self.hot = 0;
                    let block: Profile = (0..layers).map(|layer| (layer, self.runtime.resident_experts(layer))).collect();
                    self.profiles.insert("settled".to_owned(), block.clone());
                    *self.current_profile.lock().unwrap() = "settled".to_owned();
                    if let Some(directory) = &self.profiles_dir {
                        let object: serde_json::Map<String, serde_json::Value> = block.iter().map(|(layer, ids)| (layer.to_string(), serde_json::Value::from(ids.clone()))).collect();
                        let _ = std::fs::write(directory.join("settled.json"), serde_json::to_string(&object).unwrap_or_default());
                    }
                    eprintln!("experts: settled after {CALM_REQUESTS} requests without exchanges (the router wanted {:.1} % outside the pool); this block stays resident (saved as profile \"settled\")", miss * 100.0);
                }
            }
        }
        let mut status = self.status.lock().unwrap();
        status.last_swaps = swaps;
        status.last_miss = miss;
        status.requests += 1;
        status.changed_percent = 100.0 * swaps as f64 / slots_total;
        status.per_layer = per_layer.clone();
        if mode == Mode::Adapting {
            eprintln!("experts: the router wanted {:.1} % outside the pool; {swaps} experts exchanged ({:.1} % of the pool changed)", miss * 100.0, status.changed_percent);
            if self.debug {
                eprintln!("  debug, per layer: {per_layer:?}");
            }
        }
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
        eprintln!("{note}");
        Ok(note)
    }

    /// The slot for `prompt` and how much of it the slot already holds.
    fn pick_slot(&self, prompt: &[u32]) -> (usize, usize) {
        (0..self.slots.len())
            .map(|index| {
                let history = &self.slots[index].history;
                (index, reusable_prefix(history, prompt, RING_SLACK), history.is_empty(), self.slots[index].last_used)
            })
            .max_by_key(|&(_, reused, empty, last_used)| (reused, empty, std::cmp::Reverse(last_used)))
            .map(|(index, reused, _, _)| (index, reused))
            .expect("at least one slot")
    }

    fn serve(&mut self, job: &Job) -> Result<()> {
        let ids = self.tokenizer.encode(job.prompt.as_str(), true).map_err(anyhow::Error::msg)?.get_ids().to_vec();
        if ids.len() + 1 > self.context {
            let _ = job.events.send(Event::Failed {
                code: "context_length_exceeded",
                message: format!("the conversation is {} tokens and the model was compiled for {}: start a new chat or compile for a larger context (dafo-serve --context)", ids.len(), self.context),
            });
            return Ok(());
        }
        let max_tokens = job.settings.max_tokens.min(self.context - ids.len()).max(1);
        let (slot, reused) = self.pick_slot(&ids);
        let started = std::time::Instant::now();
        self.clock += 1;
        self.slots[slot].last_used = self.clock;
        self.slots[slot].history.truncate(reused);
        self.runtime.select_slot(slot);
        let outcome = self.generate(job, slot, &ids, max_tokens);
        eprintln!("request: {} prompt tokens ({reused} reused from conversation {slot}) in {:.1} s", ids.len(), started.elapsed().as_secs_f64());
        if outcome.is_err() {
            self.slots[slot].history.clear();
        }
        outcome
    }

    fn generate(&mut self, job: &Job, slot: usize, ids: &[u32], max_tokens: usize) -> Result<()> {
        let mut cursor = self.slots[slot].history.len();
        while cursor < ids.len() {
            if job.cancel.load(Ordering::Acquire) {
                self.slots[slot].history.clear();
                return Ok(());
            }
            let end = (cursor + READ_SLICE).min(ids.len());
            self.runtime.prefill(&ids[cursor..end], cursor)?;
            self.slots[slot].history.extend_from_slice(&ids[cursor..end]);
            cursor = end;
        }
        // What the prompt asked of the router tells which experts the answer will want: settle them before decoding.
        self.adapt(true);
        let mut logits = self.runtime.logits();
        let mut sampler = Sampler::new(job.settings.sampling.clone());
        let mut generated: Vec<u32> = Vec::new();
        let (mut emitted, mut text) = (0usize, String::new());
        let hold = job.settings.stop.iter().map(|stop| stop.len()).max().unwrap_or(0);
        let finish = loop {
            if job.cancel.load(Ordering::Acquire) {
                self.slots[slot].history.clear();
                return Ok(());
            }
            let history = &self.slots[slot].history;
            let window = sampler.window();
            let recent = if window == 0 { &[][..] } else { &history[history.len().saturating_sub(window)..] };
            let token = sampler.sample(std::mem::take(&mut logits), recent);
            if self.stop_tokens.contains(&token) {
                break "stop";
            }
            generated.push(token);
            text = self.tokenizer.decode(&generated, false).map_err(anyhow::Error::msg)?;
            if let Some(at) = job.settings.stop.iter().filter_map(|stop| text.find(stop.as_str())).min() {
                text.truncate(at);
                break "stop";
            }
            if job.stream && !text.ends_with('\u{fffd}') {
                let mut safe = text.len().saturating_sub(hold);
                while !text.is_char_boundary(safe) {
                    safe -= 1;
                }
                if safe > emitted {
                    let _ = job.events.send(Event::Delta(text[emitted..safe].to_owned()));
                    emitted = safe;
                }
            }
            if generated.len() >= max_tokens {
                break "length";
            }
            let position = self.slots[slot].history.len();
            self.runtime.step(token, position)?;
            self.slots[slot].history.push(token);
            logits = self.runtime.logits();
        };
        if job.stream && text.len() > emitted && text.is_char_boundary(emitted) {
            let _ = job.events.send(Event::Delta(text[emitted..].to_owned()));
        }
        let _ = job.events.send(Event::Done { text, finish, prompt_tokens: ids.len(), completion_tokens: generated.len() });
        Ok(())
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
