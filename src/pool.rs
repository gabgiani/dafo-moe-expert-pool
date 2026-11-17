//! The expert pool of a mixture-of-experts model, kept by what the router asks for.
//!
//! A [`Runtime`] built with an expert pool (`Runtime::with_expert_pool`) holds only some of the experts of each layer; this module decides
//! **which**: it reads what the router wanted (`Runtime::take_demand`), exchanges experts, and settles on a block when the exchange stops
//! being needed. It knows nothing of tokenizers or HTTP, so a host of any kind (the `dafo-serve` engine, an application's engine) drives it:
//!
//! * [`AdaptivePool::after_prompt`] after the prompt was read and before the answer is decoded;
//! * [`AdaptivePool::after_request`] when the answer is complete.
//!
//! See `docs/adaptive.md`.

use std::collections::HashSet;
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};

use crate::runtime::Runtime;

/// The experts to keep resident, by layer.
pub type Profile = Vec<(usize, Vec<usize>)>;

/// How the pool is kept.
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

/// Requests in a row that exchange (almost) nothing before the pool settles.
pub const CALM_REQUESTS: usize = 3;
/// An expert enters the pool only when it was wanted this much more than the one it replaces.
pub const SWAP_MARGIN: f64 = 1.25;
/// Share of the pool a request may exchange and still count as calm.
pub const CALM_SHARE: f64 = 0.01;
/// A settled pool resumes exchanging when this many requests in a row miss this much more than when it settled.
pub const THAW_REQUESTS: usize = 3;

pub struct AdaptivePool {
    total: usize,
    debug: bool,
    pub status: Arc<Mutex<AdaptiveStatus>>,
    /// Per layer and expert: a running score of how much the router wanted it (each request weighs the same).
    score: Vec<Vec<f64>>,
    calm: usize,
    frozen_miss: f64,
    hot: usize,
    logger: Box<dyn Fn(&str) + Send>,
}

impl AdaptivePool {
    /// `total` experts per layer in the model; `adaptive` starts exchanging at once; `debug` prints the exchange per layer.
    pub fn new(total: usize, adaptive: bool, debug: bool) -> Self {
        let status = AdaptiveStatus { mode: if adaptive { Mode::Adapting } else { Mode::Manual }, last_swaps: 0, last_miss: 0.0, requests: 0, prefill_changed_percent: 0.0, changed_percent: 0.0, per_layer: Vec::new() };
        Self { total, debug, status: Arc::new(Mutex::new(status)), score: Vec::new(), calm: 0, frozen_miss: 0.0, hot: 0, logger: Box::new(|line| eprintln!("{line}")) }
    }

    /// The number of experts per layer of the model (known once the model is open).
    pub fn set_total(&mut self, total: usize) {
        self.total = total;
    }

    /// Where the pool says what it does (standard error by default).
    pub fn set_logger(&mut self, logger: impl Fn(&str) + Send + 'static) {
        self.logger = Box::new(logger);
    }

    fn say(&self, line: String) {
        (self.logger)(&line);
    }

    pub fn mode(&self) -> Mode {
        self.status.lock().unwrap().mode
    }

    /// Starts (`true`) or stops (`false`) exchanging experts by demand.
    pub fn enable(&mut self, on: bool) {
        self.status.lock().unwrap().mode = if on { Mode::Adapting } else { Mode::Manual };
        self.calm = 0;
        self.hot = 0;
    }

    /// Settles on what the pool holds now and returns it.
    pub fn freeze(&mut self, runtime: &Runtime) -> Result<Profile> {
        let (_, layers) = runtime.pool_shape().context("the runtime has no expert pool")?;
        self.status.lock().unwrap().mode = Mode::Frozen;
        Ok(resident_block(runtime, layers))
    }

    /// The experts the pool holds, by layer.
    pub fn block(runtime: &Runtime) -> Profile {
        runtime.pool_shape().map(|(_, layers)| resident_block(runtime, layers)).unwrap_or_default()
    }

    /// Puts the experts of `layers` in the pool; returns how many were copied.
    pub fn load(&self, runtime: &mut Runtime, layers: &Profile) -> Result<usize> {
        let mut copied = 0;
        for (layer, ids) in layers {
            copied += runtime.set_allowed_experts(*layer, ids, self.total)?;
        }
        Ok(copied)
    }

    /// After the prompt was read, before the answer: what the prompt asked of the router tells which experts the answer will want.
    pub fn after_prompt(&mut self, runtime: &mut Runtime) {
        if self.mode() == Mode::Adapting {
            self.observe(runtime, true);
        }
    }

    /// When an answer is complete. Returns the block when the pool has just settled (a host may save it as a profile).
    pub fn after_request(&mut self, runtime: &mut Runtime) -> Option<Profile> {
        self.observe(runtime, false)
    }

    fn observe(&mut self, runtime: &mut Runtime, during_prefill: bool) -> Option<Profile> {
        let (size, layers) = runtime.pool_shape()?;
        let total = self.total;
        let demand = runtime.take_demand(layers, total);
        let mode = self.mode();
        let (mut wanted, mut missed) = (0u64, 0u64);
        let mut resident_sets: Vec<HashSet<usize>> = Vec::with_capacity(layers);
        for layer in 0..layers {
            let resident: HashSet<usize> = runtime.resident_experts(layer).into_iter().collect();
            for (expert, &count) in demand[layer].iter().enumerate() {
                wanted += count as u64;
                if !resident.contains(&expert) {
                    missed += count as u64;
                }
            }
            resident_sets.push(resident);
        }
        if wanted == 0 {
            return None;
        }
        let miss = missed as f64 / wanted as f64;
        let mut swaps = 0;
        let mut per_layer = vec![0usize; layers];
        let slots_total = (size * layers).max(1) as f64;
        let mut settled = None;
        match mode {
            Mode::Manual => {}
            Mode::Frozen => {
                if during_prefill {
                    return None;
                }
                if miss > self.frozen_miss * 1.5 + 0.05 {
                    self.hot += 1;
                } else {
                    self.hot = 0;
                }
                if self.hot >= THAW_REQUESTS {
                    self.say(format!("experts: the requests miss {:.0} % of what the router wants (was {:.0} % when settled): exchanging again", miss * 100.0, self.frozen_miss * 100.0));
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
                    let mut next: HashSet<usize> = resident_sets[layer].clone();
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
                        if runtime.set_allowed_experts(layer, &ids, total).is_ok() {
                            swaps += changed;
                            per_layer[layer] = changed;
                        }
                    }
                }
                if during_prefill {
                    let percent = 100.0 * swaps as f64 / slots_total;
                    self.say(format!("experts: after reading the prompt the router wanted {:.1} % outside the pool; {swaps} experts exchanged before answering ({percent:.1} % of the pool changed)", miss * 100.0));
                    if self.debug {
                        self.say(format!("  debug, per layer: {per_layer:?}"));
                    }
                    self.status.lock().unwrap().prefill_changed_percent = percent;
                    return None;
                }
                if swaps as f64 <= CALM_SHARE * slots_total {
                    self.calm += 1;
                } else {
                    self.calm = 0;
                }
                if self.calm >= CALM_REQUESTS {
                    self.status.lock().unwrap().mode = Mode::Frozen;
                    self.frozen_miss = miss;
                    self.hot = 0;
                    settled = Some(resident_block(runtime, layers));
                    self.say(format!("experts: settled after {CALM_REQUESTS} requests without exchanges (the router wanted {:.1} % outside the pool); this block stays resident", miss * 100.0));
                }
            }
        }
        if during_prefill {
            return None;
        }
        let mut status = self.status.lock().unwrap();
        status.last_swaps = swaps;
        status.last_miss = miss;
        status.requests += 1;
        status.changed_percent = 100.0 * swaps as f64 / slots_total;
        status.per_layer = per_layer.clone();
        if mode == Mode::Adapting {
            self.say(format!("experts: the router wanted {:.1} % outside the pool; {swaps} experts exchanged ({:.1} % of the pool changed)", miss * 100.0, status.changed_percent));
            if self.debug {
                self.say(format!("  debug, per layer: {per_layer:?}"));
            }
        }
        settled
    }
}

fn resident_block(runtime: &Runtime, layers: usize) -> Profile {
    (0..layers).map(|layer| (layer, runtime.resident_experts(layer))).collect()
}

/// The `size` experts to start with in each layer from a ranking (every layer's expert ids, most useful first), as `{"layer": [ids]}` would give.
pub fn top_of_ranking(ranking: &Profile, size: usize) -> Profile {
    ranking.iter().map(|(layer, ids)| (*layer, ids.iter().copied().take(size).collect())).collect()
}

/// Experts per layer the memory allows: the weights (dense part and the pool) may take about 46 % of the machine's memory, which is
/// what a 16 GiB Mac runs a 7.4 GiB model with. Capped at the model's `experts`; a result equal to `experts` means no pool is needed.
pub fn automatic_pool_size(total_memory: u64, dense_bytes: u64, bytes_per_expert_all_layers: u64, experts: usize) -> usize {
    let budget = (total_memory as f64 * 0.46) as u64;
    let room = budget.saturating_sub(dense_bytes);
    ((room / bytes_per_expert_all_layers.max(1)) as usize).clamp(8, experts)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pool_size_follows_the_memory() {
        let gib = 1u64 << 30;
        let (dense, expert) = (2 * gib + gib / 20, 30 * 3_900_000u64); // Gemma 4 26B-A4B: dense part, one expert of every layer
        assert_eq!(automatic_pool_size(16 * gib, dense, expert, 128), 48);
        assert_eq!(automatic_pool_size(96 * gib, dense, expert, 128), 128);
        assert!(automatic_pool_size(32 * gib, dense, expert, 128) > 48 && automatic_pool_size(32 * gib, dense, expert, 128) < 128);
        assert_eq!(automatic_pool_size(4 * gib, dense, expert, 128), 8);
    }

    #[test]
    fn ranking_gives_the_first_of_each_layer() {
        let ranking: Profile = vec![(0, vec![5, 3, 9, 1]), (1, vec![2, 7, 4, 8])];
        assert_eq!(top_of_ranking(&ranking, 2), vec![(0, vec![5, 3]), (1, vec![2, 7])]);
    }
}
