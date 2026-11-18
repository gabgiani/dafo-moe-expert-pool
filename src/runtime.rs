//! Runtime: executes compiled plans. It knows kernels, buffers and dispatches; nothing about any model.
//!
//! The first plan is the one-token plan (it chooses the next token); the others take batches of tokens and are used to read a
//! prompt. They share the weights, the attention caches and the state buffer, and each has activations of its own.

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Context, Result, anyhow, bail, ensure};

use crate::backend_metal::{self, Arg, MAX_BATCH, Plan, STATE_TOKENS, STATE_VISION, STATE_WORDS, VISION_BATCH};
use crate::gguf::Gguf;
#[cfg(target_os = "macos")]
use crate::metal::{Buffer, Encoder, Gpu, Kernel, Timing};
#[cfg(not(target_os = "macos"))]
use crate::cuda::{Buffer, Encoder, Gpu, Kernel, Timing};

const PAGE: usize = 16384;
/// Weight buffers are made of at most this many bytes of the mapped file.
const MAX_WEIGHT_BUFFER: usize = 3 << 30;

#[derive(Clone, Copy, Debug)]
enum Which {
    Value(usize),
    Weight(usize),
    CacheKey(usize),
    CacheValue(usize),
    Table(usize),
    Scratch(usize),
    /// The rows an image's soft tokens are read from.
    Extra,
    State,
    Demand,
}

#[derive(Clone, Copy, Debug)]
enum Bound {
    Buffer(Which, usize),
    U32(u32),
    I64(i64),
    U64(u64),
    F32(f32),
}

/// Window of dispatches the scheduler looks through to find ones that can overlap.
const SCHEDULE_WINDOW: usize = 24;

/// A resource (buffer) a dispatch touches, as a number; `None` for the ones nothing writes (weights, tables).
fn resource_of(which: Which) -> Option<u64> {
    Some(match which {
        Which::Value(value) => value as u64,
        Which::CacheKey(layer) => (1 << 32) | layer as u64,
        Which::CacheValue(layer) => (2 << 32) | layer as u64,
        Which::Scratch(index) => (3 << 32) | index as u64,
        Which::Extra => 4 << 32,
        Which::State => 5 << 32,
        Which::Demand => 6 << 32,
        Which::Weight(_) | Which::Table(_) => return None,
    })
}

/// The dispatches of a program in groups that can overlap. With `DAFO_SERIAL_DISPATCH` set every dispatch is a group of its own
/// (the order of the plan, for comparing). A kernel the access analysis does not know is taken to write everything it is given.
fn schedule_dispatches(dispatches: &[Dispatch], kernel_names: &[&'static str], access: &HashMap<String, Vec<(usize, bool)>>) -> Vec<Vec<usize>> {
    if std::env::var_os("DAFO_SERIAL_DISPATCH").is_some() {
        return (0..dispatches.len()).map(|index| vec![index]).collect();
    }
    let accesses: Vec<(Vec<u64>, Vec<u64>)> = dispatches
        .iter()
        .map(|dispatch| {
            let known = access.get(kernel_names[dispatch.kernel]);
            let (mut reads, mut writes) = (Vec::new(), Vec::new());
            for (index, arg) in dispatch.args.iter().enumerate() {
                let Bound::Buffer(which, _) = arg else { continue };
                let Some(resource) = resource_of(*which) else { continue };
                let written = known.is_none_or(|entries| entries.iter().find(|(position, _)| *position == index).is_none_or(|(_, written)| *written));
                if written { writes.push(resource) } else { reads.push(resource) }
            }
            (reads, writes)
        })
        .collect();
    backend_metal::schedule(&accesses, SCHEDULE_WINDOW)
}

struct Dispatch {
    kernel: usize,
    args: Vec<Bound>,
    groups: [usize; 3],
    threads: [usize; 3],
    shared_memory: usize,
}

/// A plan with its own activations.
struct Program {
    batch: usize,
    /// The dispatches in groups that can overlap (see `backend_metal::schedule`); a barrier goes between groups.
    groups: Vec<Vec<usize>>,
    /// Reads images (a batch larger than prompts use): its activations are allocated by the first image.
    vision: bool,
    dispatches: Vec<Dispatch>,
    values: Vec<Option<Buffer>>,
    canonical: Vec<usize>,
    tables: Vec<Buffer>,
    scratch: Vec<Buffer>,
    logits: Option<usize>,
    softcap: Option<f32>,
    /// Elements of each value's buffer and bytes of each scratch buffer, for a program that allocates them later.
    lens: Vec<usize>,
    scratch_bytes: Vec<usize>,
    allocated: bool,
}

pub struct Runtime {
    gpu: Gpu,
    kernels: Vec<Kernel>,
    kernel_names: Vec<&'static str>,
    /// `programs[0]` runs one token; the rest are batch plans, largest first.
    programs: Vec<Program>,
    /// Buffers over the mapped weights file.
    weights: Vec<Buffer>,
    /// The key and value caches (half precision) of every slot, by layer. A slot holds one conversation.
    caches: Vec<HashMap<usize, (Buffer, Buffer)>>,
    /// The slot the next run uses.
    slot: usize,
    state: Buffer,
    /// Per layer and expert, how many times the router wanted it (all experts, blocked or not) since the last [`Runtime::take_demand`].
    demand: Buffer,
    /// Soft-token embeddings of the image being read (made by the first image).
    extra: Option<Buffer>,
    weight_bytes: usize,
    /// Experts resident per layer when the runtime was built with an expert pool.
    pool: Option<Pool>,
    /// Expert slots in device memory (the others are in host memory); u32::MAX when all are.
    device_slots: u32,
    /// The mapped file the weight buffers point into; declared last so it is dropped after them.
    _gguf: Arc<Gguf>,
}

/// The experts of each layer that sit in the pool: buffers of `size` experts for the two expert tensors of every layer.
struct Pool {
    /// Experts per layer in all: `device` slots in device memory followed by the rest in pinned host memory.
    size: usize,
    device: usize,
    /// Per layer: where `ffn_gate_up_exps` and `ffn_down_exps` live, and which expert holds each slot.
    layers: Vec<PoolLayer>,
}

/// One expert tensor of a layer in the pool: the weight buffer of its device slots, that of its host slots (if any) and the bytes of an expert.
#[derive(Clone, Copy)]
struct Tier {
    device: usize,
    host: Option<usize>,
    stride: usize,
}

struct PoolLayer {
    gate_up: Tier,
    down: Tier,
    slots: Vec<Option<usize>>,
    total: usize,
}

/// Largest gap between two tensors a weight buffer still wraps in one piece.
const MAX_GAP: usize = 4 << 20;

fn round_down(value: usize) -> usize {
    value / PAGE * PAGE
}

fn round_up(value: usize) -> usize {
    value.div_ceil(PAGE) * PAGE
}

impl Runtime {
    /// Builds the kernels and wraps the weights of `gguf` (mapped, not copied) for the plans. `plans[0]` must be the one-token plan.
    /// `slots` conversations keep caches of their own. The runtime keeps the file mapped.
    pub fn new(gguf: &Arc<Gguf>, plans: &[Plan], slots: usize) -> Result<Self> {
        Self::with_expert_pool(gguf, plans, slots, None)
    }

    /// Like [`Runtime::new`]; with `pool = Some(n)` the expert tensors are not mapped but copied, `n` experts per layer, into buffers the
    /// runtime fills on demand ([`Runtime::set_allowed_experts`]). The plans must come from a graph prepared by
    /// `backend_metal::use_expert_pool(graph, n)`. The memory the experts take is then `n` of the model's experts per layer, whatever
    /// their number in the file: the model is the full one on disk and a cut in memory.
    pub fn with_expert_pool(gguf: &Arc<Gguf>, plans: &[Plan], slots: usize, pool: Option<usize>) -> Result<Self> {
        Self::with_expert_tiers(gguf, plans, slots, pool.map(|size| (size, 0)))
    }

    /// Like [`Runtime::with_expert_pool`] with two places for the experts: `(device, host)` slots per layer, the first in device memory and
    /// the rest in pinned host memory the GPU reads over PCIe (CUDA only; the plans must be compiled for a pool of `device + host`).
    /// The experts of the file that are in neither stay on disk (mapped) until the host asks for them.
    pub fn with_expert_tiers(gguf: &Arc<Gguf>, plans: &[Plan], slots: usize, tiers: Option<(usize, usize)>) -> Result<Self> {
        let pool = tiers.map(|(device, host)| device + host);
        let device_slots = tiers.map_or(u32::MAX, |(device, host)| if host == 0 { u32::MAX } else { device as u32 });
        ensure!(!plans.is_empty() && plans[0].batch == 1, "the first plan must take one token");
        ensure!(slots >= 1, "at least one slot");
        for plan in plans {
            ensure!(plan.source == plans[0].source, "the plans must share their kernels");
        }
        let gpu = Gpu::new()?;
        let library = gpu.library(&plans[0].source)?;

        let mut kernel_index: HashMap<&'static str, usize> = HashMap::new();
        let mut kernels = Vec::new();
        for plan in plans {
            for step in &plan.steps {
                if !kernel_index.contains_key(step.kernel) {
                    kernel_index.insert(step.kernel, kernels.len());
                    kernels.push(gpu.kernel(&library, step.kernel)?);
                }
            }
        }
        let mut kernel_names = vec![""; kernels.len()];
        for (name, &index) in &kernel_index {
            kernel_names[index] = name;
        }

        // Weights: the mapped file in page-aligned pieces that hold whole tensors.
        let weight_names: std::collections::BTreeSet<&String> = plans.iter().flat_map(|plan| plan.weights.iter()).collect();
        let mut spans: Vec<(usize, usize, &str)> = Vec::new();
        let is_expert_tensor = |name: &str| pool.is_some() && (name.ends_with(".ffn_gate_up_exps.weight") || name.ends_with(".ffn_down_exps.weight"));
        let mut pool_names: Vec<&String> = Vec::new();
        for name in weight_names {
            if is_expert_tensor(name) {
                pool_names.push(name);
                continue;
            }
            let tensor = gguf.tensor(name).with_context(|| format!("the plan needs {name}, which the file lacks"))?;
            spans.push((tensor.offset, tensor.offset + tensor.bytes(), name));
        }
        spans.sort();
        let limit = MAX_WEIGHT_BUFFER.min(gpu.max_buffer_bytes());
        let map = gguf.mapped();
        // Metal wraps whole pages (the tail of the last page of the file is readable); a copy to another device must stop at the end of the file.
        let mapped_end = if cfg!(target_os = "macos") { round_up(map.len()) } else { map.len() };
        let mut buffers = Vec::new();
        let mut location: HashMap<&str, (usize, usize)> = HashMap::new();
        let mut host_location: HashMap<&str, usize> = HashMap::new();
        let mut weight_bytes = 0;
        let mut index = 0;
        while index < spans.len() {
            let start = round_down(spans[index].0);
            let mut end = round_up(spans[index].1);
            let first = index;
            // A buffer ends where the file leaves a gap (the experts of a pool are not mapped): wrapping the gap would put it in the
            // device's working set too.
            while index < spans.len() && round_up(spans[index].1) - start <= limit && (index == first || round_down(spans[index].0) <= end + MAX_GAP) {
                end = round_up(spans[index].1);
                index += 1;
            }
            if index == first {
                bail!("{} is larger than a Metal buffer can be", spans[first].2);
            }
            let end = end.min(mapped_end);
            // The mapping covers whole pages, so the tail of the last page of the file is readable.
            let buffer = unsafe { gpu.buffer_no_copy(map.as_ptr().add(start), end - start)? };
            for &(from, to, name) in &spans[first..index] {
                location.insert(name, (buffers.len(), from - start));
                weight_bytes += to - from;
            }
            buffers.push(buffer);
        }

        // The expert pool: a buffer of `size` experts for each expert tensor the plans read, empty until the host chooses experts.
        let mut pool_state = None;
        if let Some(size) = pool {
            let mut layers: Vec<PoolLayer> = Vec::new();
            for name in &pool_names {
                let tensor = gguf.tensor(name).with_context(|| format!("the plan needs {name}, which the file lacks"))?;
                let total = *tensor.dims.last().context("an expert tensor without dimensions")? as usize;
                ensure!(size >= 1 && size <= total, "the pool of {size} experts does not fit a tensor of {total}");
                let stride = tensor.bytes() / total;
                let layer: usize = name.split('.').nth(1).and_then(|part| part.parse().ok()).with_context(|| format!("cannot read the layer of {name}"))?;
                let (device_count, host_count) = tiers.expect("a pool has tiers");
                while layers.len() <= layer {
                    let empty = Tier { device: usize::MAX, host: None, stride: 0 };
                    layers.push(PoolLayer { gate_up: empty, down: empty, slots: vec![None; size], total });
                }
                let device_index = buffers.len();
                buffers.push(gpu.buffer(device_count * stride)?);
                location.insert(name.as_str(), (device_index, 0));
                weight_bytes += device_count * stride;
                let mut host_index = None;
                if host_count > 0 {
                    host_index = Some(buffers.len());
                    buffers.push(gpu.buffer_shared(host_count * stride)?);
                    host_location.insert(name.as_str(), buffers.len() - 1);
                }
                let tier = Tier { device: device_index, host: host_index, stride };
                if name.contains("gate_up") {
                    layers[layer].gate_up = tier;
                } else {
                    layers[layer].down = tier;
                }
            }
            pool_state = Some(Pool { size, device: tiers.map_or(size, |(device, _)| device), layers });
        }

        let mut caches = Vec::new();
        for _ in 0..slots {
            let mut slot = HashMap::new();
            for spec in &plans[0].caches {
                slot.insert(spec.layer, (gpu.buffer(spec.elements * 2)?, gpu.buffer(spec.elements * 2)?));
            }
            caches.push(slot);
        }
        let state = gpu.buffer_shared(STATE_WORDS * 4)?;
        let demand = gpu.buffer_shared(backend_metal::MAX_EXPERT_LAYERS * backend_metal::EXPERT_MASK_STRIDE * 4)?;

        let access = if cfg!(target_os = "macos") { backend_metal::kernel_buffer_access(&plans[0].source) } else { HashMap::new() };
        let mut programs = Vec::new();
        for plan in plans {
            let vision = plan.batch > MAX_BATCH;
            let mut values: Vec<Option<Buffer>> = (0..plan.lens.len()).map(|_| None).collect();
            let mut scratch = Vec::new();
            if !vision {
                for (value, &owner) in plan.canonical.iter().enumerate() {
                    if owner == value {
                        values[value] = Some(gpu.buffer(plan.lens[value] * 4)?);
                    }
                }
                scratch = plan.scratch.iter().map(|&bytes| gpu.buffer(bytes)).collect::<Result<Vec<_>>>()?;
            }
            let tables = plan.tables.iter().map(|table| gpu.buffer_from(table)).collect::<Result<Vec<_>>>()?;
            let mut dispatches = Vec::with_capacity(plan.steps.len());
            for step in &plan.steps {
                let mut args = Vec::with_capacity(step.args.len());
                for arg in &step.args {
                    args.push(match arg {
                        Arg::Value(value) => Bound::Buffer(Which::Value(plan.canonical[*value]), 0),
                        Arg::ValueAt(value, offset) => Bound::Buffer(Which::Value(plan.canonical[*value]), *offset),
                        Arg::Weight(name) => {
                            let &(buffer, offset) = location.get(name.as_str()).ok_or_else(|| anyhow!("weight {name} has no buffer"))?;
                            Bound::Buffer(Which::Weight(buffer), offset)
                        }
                        Arg::WeightHost(name) => {
                            let buffer = match host_location.get(name.as_str()) {
                                Some(&buffer) => buffer,
                                None => location.get(name.as_str()).ok_or_else(|| anyhow!("weight {name} has no buffer"))?.0,
                            };
                            Bound::Buffer(Which::Weight(buffer), 0)
                        }
                        Arg::DeviceSlots => Bound::U32(device_slots),
                        Arg::CacheKey(layer) => Bound::Buffer(Which::CacheKey(*layer), 0),
                        Arg::CacheValue(layer) => Bound::Buffer(Which::CacheValue(*layer), 0),
                        Arg::Table(table) => Bound::Buffer(Which::Table(*table), 0),
                        Arg::Extra => Bound::Buffer(Which::Extra, 0),
                        Arg::Scratch(index) => Bound::Buffer(Which::Scratch(*index), 0),
                        Arg::State => Bound::Buffer(Which::State, 0),
                        Arg::Demand => Bound::Buffer(Which::Demand, 0),
                        Arg::U32(value) => Bound::U32(*value),
                        Arg::I64(value) => Bound::I64(*value),
                        Arg::U64(value) => Bound::U64(*value),
                        Arg::F32(value) => Bound::F32(*value),
                    });
                }
                dispatches.push(Dispatch {
                    kernel: kernel_index[step.kernel],
                    args,
                    groups: step.groups,
                    threads: step.threads,
                    shared_memory: step.shared_memory,
                });
            }
            let groups = schedule_dispatches(&dispatches, &kernel_names, &access);
            programs.push(Program {
                batch: plan.batch,
                groups,
                vision,
                dispatches,
                values,
                canonical: plan.canonical.clone(),
                tables,
                scratch,
                logits: plan.logits,
                softcap: plan.softcap,
                lens: plan.lens.clone(),
                scratch_bytes: plan.scratch.clone(),
                allocated: !vision,
            });
        }
        // Batch plans largest first, after the one-token plan.
        programs[1..].sort_by(|a, b| b.batch.cmp(&a.batch));
        let mut runtime = Self { gpu, kernels, kernel_names, programs, weights: buffers, caches, slot: 0, state, demand, extra: None, weight_bytes, pool: pool_state, device_slots, _gguf: gguf.clone() };
        if let Some(pool) = &runtime.pool {
            // Until the host chooses, each layer holds its first experts.
            let (size, layers) = (pool.size, pool.layers.len());
            for layer in 0..layers {
                let first: Vec<usize> = (0..size).collect();
                runtime.set_allowed_experts(layer, &first, runtime.pool.as_ref().unwrap().layers[layer].total)?;
            }
        }
        Ok(runtime)
    }

    /// Conversations the runtime keeps caches for.
    pub fn slots(&self) -> usize {
        self.caches.len()
    }

    /// Chooses the slot (conversation) the next steps and prefills use.
    pub fn select_slot(&mut self, slot: usize) {
        assert!(slot < self.caches.len(), "no such slot");
        self.slot = slot;
    }

    pub fn device_name(&self) -> String {
        self.gpu.name()
    }

    /// Dispatches of the one-token plan.
    pub fn dispatch_count(&self) -> usize {
        self.programs[0].dispatches.len()
    }

    /// Bytes of weights the plans read (mapped, not copied).
    pub fn weight_bytes(&self) -> usize {
        self.weight_bytes
    }

    /// Lets the router of mixture-of-experts layer `layer` (in graph order) choose only among `allowed` of its `total` experts; the
    /// others are blocked and its weights renormalize among the rest, exactly as if the model had been cut to those experts. Needs at
    /// least as many allowed experts as the model uses per token. Takes effect on the next token (call it between tokens).
    ///
    /// Without an expert pool this only changes what the router may choose. With one (`with_expert_pool`) `allowed` must hold at most
    /// the pool's size: the experts already resident stay where they are, the others are copied from the file into free slots. Returns
    /// the number of experts copied.
    pub fn set_allowed_experts(&mut self, layer: usize, allowed: &[usize], total: usize) -> Result<usize> {
        use crate::backend_metal::{EXPERT_MASK_STRIDE, MAX_EXPERT_LAYERS, STATE_EXPERT_MASK};
        anyhow::ensure!(layer < MAX_EXPERT_LAYERS && total <= EXPERT_MASK_STRIDE, "layer {layer} or {total} experts do not fit the mask");
        anyhow::ensure!(allowed.iter().all(|&expert| expert < total), "an allowed expert is out of range");
        let mut copied = 0;
        let mut words = vec![0xffff_ffffu32; EXPERT_MASK_STRIDE];
        if let Some(pool) = &mut self.pool {
            anyhow::ensure!(allowed.len() <= pool.size, "{} experts do not fit a pool of {}", allowed.len(), pool.size);
            let device = pool.device;
            let entry = pool.layers.get_mut(layer).with_context(|| format!("layer {layer} has no experts in the pool"))?;
            // `allowed` is in priority order: the first `device` experts belong in device memory, the rest in host memory.
            let mut order: Vec<usize> = Vec::with_capacity(allowed.len());
            for &expert in allowed {
                if !order.contains(&expert) {
                    order.push(expert);
                }
            }
            let slot_of: std::collections::HashMap<usize, usize> = entry.slots.iter().enumerate().filter_map(|(slot, expert)| expert.map(|expert| (expert, slot))).collect();
            // Which experts stay where they are, and which have to be placed (loaded from the file, or moved between the tiers).
            let mut to_place: Vec<(usize, bool, Option<usize>)> = Vec::new(); // (expert, wants device, current slot)
            let mut keep = std::collections::HashSet::new();
            for (rank, &expert) in order.iter().enumerate() {
                let wants_device = rank < device;
                match slot_of.get(&expert) {
                    Some(&slot) if (slot < device) == wants_device => {
                        keep.insert(slot);
                    }
                    other => to_place.push((expert, wants_device, other.copied())),
                }
            }
            // The bytes of the experts that move: read them before their old slot is reused.
            let map = self._gguf.mapped();
            let mut moving: Vec<(usize, bool, [Vec<u8>; 2])> = Vec::new();
            for &(expert, wants_device, current) in &to_place {
                let mut data = [Vec::new(), Vec::new()];
                for (index, (tier, tensor_name)) in [(entry.gate_up, "ffn_gate_up_exps"), (entry.down, "ffn_down_exps")].into_iter().enumerate() {
                    data[index] = match current {
                        Some(slot) if slot < device => self.weights[tier.device].read_bytes(slot * tier.stride, tier.stride),
                        Some(slot) => {
                            let host = tier.host.context("a host slot without a host buffer")?;
                            self.weights[host].read_bytes((slot - device) * tier.stride, tier.stride)
                        }
                        None => {
                            let tensor = self._gguf.tensor(&format!("blk.{layer}.{tensor_name}.weight")).context("expert tensor missing")?;
                            let from = tensor.offset + expert * tier.stride;
                            let bytes = map[from..from + tier.stride].to_vec();
                            // The copy is the resident one: let the file's pages go so the page cache does not keep a second copy.
                            self._gguf.advise(from, tier.stride, false);
                            bytes
                        }
                    };
                }
                moving.push((expert, wants_device, data));
            }
            let mut free_device: Vec<usize> = (0..device).filter(|slot| !keep.contains(slot)).collect();
            let mut free_host: Vec<usize> = (device..entry.slots.len()).filter(|slot| !keep.contains(slot)).collect();
            free_device.reverse();
            free_host.reverse();
            let mut slots = vec![None; entry.slots.len()];
            for &slot in &keep {
                slots[slot] = entry.slots[slot];
            }
            for (expert, wants_device, data) in moving {
                let slot = if wants_device { free_device.pop() } else { free_host.pop() }.context("no free slot in the pool")?;
                for (index, tier) in [entry.gate_up, entry.down].into_iter().enumerate() {
                    if slot < device {
                        self.weights[tier.device].write_bytes(slot * tier.stride, &data[index])?;
                    } else {
                        let host = tier.host.context("a host slot without a host buffer")?;
                        self.weights[host].write_bytes((slot - device) * tier.stride, &data[index])?;
                    }
                }
                slots[slot] = Some(expert);
                copied += 1;
            }
            entry.slots = slots;
            for (slot, expert) in entry.slots.iter().enumerate() {
                if let Some(expert) = expert {
                    words[*expert] = slot as u32 + 1;
                }
            }
        } else {
            for &expert in allowed {
                words[expert] = 0;
            }
        }
        self.state.as_mut_slice::<u32>()[STATE_EXPERT_MASK + layer * EXPERT_MASK_STRIDE..][..EXPERT_MASK_STRIDE].copy_from_slice(&words);
        Ok(copied)
    }

    /// What the router wanted since the last call, per layer (in graph order) and expert: how many times the expert was among the ones it
    /// would have chosen with every expert allowed. Resets the counters. Call it when the GPU is idle (between requests).
    pub fn take_demand(&mut self, layers: usize, total: usize) -> Vec<Vec<u32>> {
        use crate::backend_metal::EXPERT_MASK_STRIDE;
        let words = self.demand.as_mut_slice::<u32>();
        let mut out = Vec::with_capacity(layers);
        for layer in 0..layers {
            let row = &mut words[layer * EXPERT_MASK_STRIDE..][..EXPERT_MASK_STRIDE];
            out.push(row[..total].to_vec());
            row.fill(0);
        }
        out
    }

    /// Experts per layer the pool holds, and the layers it covers (`None` without a pool).
    pub fn pool_shape(&self) -> Option<(usize, usize)> {
        self.pool.as_ref().map(|pool| (pool.size, pool.layers.len()))
    }

    /// The experts resident in the pool for `layer`.
    pub fn resident_experts(&self, layer: usize) -> Vec<usize> {
        self.pool.as_ref().and_then(|pool| pool.layers.get(layer)).map(|entry| entry.slots.iter().flatten().copied().collect()).unwrap_or_default()
    }

    /// Lets every expert of every layer be chosen (the state at start).
    pub fn allow_all_experts(&mut self) {
        use crate::backend_metal::{EXPERT_MASK_STRIDE, MAX_EXPERT_LAYERS, STATE_EXPERT_MASK};
        self.state.as_mut_slice::<u32>()[STATE_EXPERT_MASK..STATE_EXPERT_MASK + EXPERT_MASK_STRIDE * MAX_EXPERT_LAYERS].fill(0);
    }

    fn buffer<'a>(&'a self, program: &'a Program, which: Which) -> &'a Buffer {
        match which {
            Which::Value(value) => program.values[value].as_ref().expect("a value without a buffer"),
            Which::Weight(index) => &self.weights[index],
            Which::CacheKey(layer) => &self.caches[self.slot][&layer].0,
            Which::CacheValue(layer) => &self.caches[self.slot][&layer].1,
            Which::Table(index) => &program.tables[index],
            Which::Scratch(index) => &program.scratch[index],
            // Before the first image there are no extra rows; the kernel never reads them then.
            Which::Extra => self.extra.as_ref().unwrap_or(&self.state),
            Which::State => &self.state,
            Which::Demand => &self.demand,
        }
    }

    fn encode_dispatch(&self, program: &Program, dispatch: &Dispatch, encoder: &Encoder) {
        encoder.kernel(&self.kernels[dispatch.kernel]);
        for (index, arg) in dispatch.args.iter().enumerate() {
            match arg {
                Bound::Buffer(which, offset) => encoder.buffer(index, self.buffer(program, *which), *offset),
                Bound::U32(value) => encoder.value(index, value),
                Bound::I64(value) => encoder.value(index, value),
                Bound::U64(value) => encoder.value(index, value),
                Bound::F32(value) => encoder.value(index, value),
            }
        }
        if dispatch.shared_memory > 0 {
            encoder.shared_memory(0, dispatch.shared_memory);
        }
        encoder.dispatch(dispatch.groups, dispatch.threads);
    }

    /// The dispatches one after another, as a serial encoder runs them (used to time kernels alone).
    fn run_dispatches(&self, program: &Program, selected: &[&Dispatch]) -> Result<Timing> {
        self.gpu.run(|encoder| {
            for dispatch in selected {
                self.encode_dispatch(program, dispatch, encoder);
            }
        })
    }

    /// The whole program, the dispatches of a group overlapping and a barrier between groups.
    fn run_scheduled(&self, program: &Program) -> Result<Timing> {
        self.gpu.run_concurrent(|encoder| {
            for (position, group) in program.groups.iter().enumerate() {
                if position > 0 {
                    encoder.barrier();
                }
                for &index in group {
                    self.encode_dispatch(program, &program.dispatches[index], encoder);
                }
            }
        })
    }

    /// Makes the activations of a program that allocates them on first use.
    fn ensure_allocated(&mut self, index: usize) -> Result<()> {
        if self.programs[index].allocated {
            return Ok(());
        }
        let program = &mut self.programs[index];
        for (value, &owner) in program.canonical.iter().enumerate() {
            if owner == value {
                program.values[value] = Some(self.gpu.buffer(program.lens[value] * 4)?);
            }
        }
        program.scratch = program.scratch_bytes.iter().map(|&bytes| self.gpu.buffer(bytes)).collect::<Result<Vec<_>>>()?;
        program.allocated = true;
        Ok(())
    }

    /// Runs a plan on `tokens` starting at `position`. `image_last` is the last position of the image the tokens belong to, if they do.
    fn run_program(&mut self, program: usize, tokens: &[u32], position: usize, image_last: Option<usize>) -> Result<Timing> {
        self.ensure_allocated(program)?;
        {
            let state = self.state.as_mut_slice::<u32>();
            state[0] = position as u32;
            state[3] = tokens.len() as u32;
            state[STATE_TOKENS..STATE_TOKENS + tokens.len()].copy_from_slice(tokens);
            state[STATE_VISION..STATE_VISION + tokens.len()].fill(image_last.unwrap_or(0) as u32);
        }
        let program = &self.programs[program];
        self.run_scheduled(program)
    }

    /// Runs the model on `token` at `position` and returns the token it chooses, with the time of the command buffer.
    pub fn step(&mut self, token: u32, position: usize) -> Result<(u32, Timing)> {
        let timing = self.run_program(0, &[token], position, None)?;
        Ok((self.state.as_slice::<u32>()[2], timing))
    }

    /// Reads `tokens` starting at `position` and returns the token the model chooses after the last one. The tokens go in batches
    /// where a batch plan fits (the weights are then read once per batch), and the last token always goes alone.
    pub fn prefill(&mut self, tokens: &[u32], position: usize) -> Result<u32> {
        ensure!(!tokens.is_empty(), "nothing to read");
        let mut done = 0;
        while tokens.len() - done > 1 {
            let remaining = tokens.len() - done - 1;
            let Some(index) = (1..self.programs.len()).find(|&index| !self.programs[index].vision && self.programs[index].batch <= remaining) else { break };
            let batch = self.programs[index].batch;
            self.run_program(index, &tokens[done..done + batch], position + done, None)?;
            done += batch;
        }
        let mut next = 0;
        for (offset, &token) in tokens[done..].iter().enumerate() {
            next = self.step(token, position + done + offset)?.0;
        }
        Ok(next)
    }

    /// Reads the soft tokens of one image (`embeddings` is `tokens * width` values, used as they are) at `position`: they all go through
    /// the model in one run, and in the sliding layers each sees all the others. The model's logits are not made (the prompt goes on).
    pub fn run_image(&mut self, embeddings: &[f32], width: usize, position: usize) -> Result<()> {
        ensure!(width > 0 && embeddings.len() % width == 0, "the embeddings are not whole rows");
        let tokens = embeddings.len() / width;
        ensure!(tokens > 0 && tokens <= VISION_BATCH, "an image of {tokens} tokens does not fit the plan that reads images ({VISION_BATCH})");
        let index = self.programs.iter().position(|program| program.vision).context("the runtime has no plan for images")?;
        if self.extra.as_ref().is_none_or(|extra| extra.bytes() < VISION_BATCH * width * 4) {
            self.extra = Some(self.gpu.buffer_shared(VISION_BATCH * width * 4)?);
        }
        self.extra.as_mut().expect("made above").as_mut_slice::<f32>()[..embeddings.len()].copy_from_slice(embeddings);
        let ids: Vec<u32> = (0..tokens as u32).map(|row| 0x8000_0000 | row).collect();
        self.run_program(index, &ids, position, Some(position + tokens - 1))?;
        Ok(())
    }

    /// The logits of the last position (after the model's cap), for sampling on the CPU.
    pub fn logits(&self) -> Vec<f32> {
        let program = &self.programs[0];
        let Some(value) = program.logits else { return Vec::new() };
        let mut logits = self.read_value(value);
        if let Some(cap) = program.softcap {
            for logit in &mut logits {
                *logit = cap * (*logit / cap).tanh();
            }
        }
        logits
    }

    /// Time of the dispatches of each kernel of a plan run alone (median of `runs`), in the order the kernels first appear. Run
    /// after real work, so the router's choices and the caches hold real data. Returns `(kernel, dispatches, seconds)`.
    pub fn profile_by_kernel(&self, batch: usize, runs: usize) -> Result<Vec<(&'static str, usize, f64)>> {
        let program = self.programs.iter().find(|program| program.batch == batch).context("no plan of that batch")?;
        let mut result = Vec::new();
        for (index, name) in self.kernel_names.iter().enumerate() {
            let selected: Vec<&Dispatch> = program.dispatches.iter().filter(|dispatch| dispatch.kernel == index).collect();
            if selected.is_empty() {
                continue;
            }
            let mut times = Vec::new();
            for _ in 0..runs {
                times.push(self.run_dispatches(program, &selected)?.gpu);
            }
            times.sort_by(f64::total_cmp);
            result.push((*name, selected.len(), times[times.len() / 2]));
        }
        Ok(result)
    }

    /// Names of the kernels in the order of their indices.
    pub fn kernel_names(&self) -> &[&'static str] {
        &self.kernel_names
    }

    /// The key and value caches of a layer (for tests and comparisons).
    pub fn read_cache(&self, layer: usize) -> Option<(Vec<f32>, Vec<f32>)> {
        let widen = |buffer: &Buffer| buffer.to_vec::<u16>().iter().map(|&bits| crate::quant::half_to_f32(bits)).collect::<Vec<f32>>();
        self.caches[self.slot].get(&layer).map(|(key, value)| (widen(key), widen(value)))
    }

    /// Runs the plan of `batch` tokens on `tokens` (for tests and comparisons).
    pub fn run_batch_for_test(&mut self, batch: usize, tokens: &[u32], position: usize) -> Result<()> {
        let index = self.programs.iter().position(|program| program.batch == batch).context("no plan of that batch")?;
        self.run_program(index, tokens, position, None)?;
        Ok(())
    }

    /// The elements of a value of the plan of `batch` tokens (for tests and comparisons).
    pub fn read_value_of(&self, batch: usize, value: usize) -> Vec<f32> {
        let program = self.programs.iter().find(|program| program.batch == batch).expect("no plan of that batch");
        let canonical = program.canonical[value];
        program.values[canonical].as_ref().map(|buffer| buffer.to_vec::<f32>()).unwrap_or_default()
    }

    /// The elements of a value of the one-token plan after the last step (for tests and comparisons).
    pub fn read_value(&self, value: usize) -> Vec<f32> {
        let program = &self.programs[0];
        let canonical = program.canonical[value];
        program.values[canonical].as_ref().map(|buffer| buffer.to_vec::<f32>()).unwrap_or_default()
    }
}
