//! Metal backend: turns the graph of a token into an execution plan (a list of kernel dispatches over named buffers) and the
//! source of the kernels, with the constants of the model written into its header. The plan names no model: it is what the
//! runtime executes.
//!
//! A plan serves `batch` tokens at once: with one token the products are matrix-vector (reading the weights costs a token's time);
//! with more they are matrix-matrix (the weights are read once for the whole batch), which is how a prompt is read.

use std::collections::{BTreeSet, HashMap};

use anyhow::{Result, bail, ensure};

use crate::gguf::{Encoding, Gguf};
use crate::ir::{AttentionKind, Graph, Op, Value, Weight};

const KERNELS: &str = include_str!("backend_metal/kernels.metal");
const MATMUL: &str = include_str!("backend_metal/matmul.metal");
const CUDA_KERNELS: &str = include_str!("backend_cuda/kernels.cu");
/// Rows per SIMD group and SIMD groups per thread group of the matrix-vector products.
const ROWS_PER_SIMD_GROUP: usize = 4;
const SIMD_GROUPS: usize = 2;
/// The largest batch any plan may have (a sliding layer's ring keeps at least this many positions beyond its window, so a batch
/// writing its keys before attending never overwrites a position an earlier token of the same batch still reads).
pub const MAX_BATCH: usize = 128;
/// Positions beyond its window a sliding layer's ring keeps. A request may continue from a point up to this many positions before
/// the end of what the caches hold (its prompt shares a prefix with the previous one but not the answer that followed it).
pub const RING_SLACK: usize = 640;
/// The plan that reads an image: every token of an image must be in one batch (they see each other), and an image is at most this many
/// tokens (the compiled engine uses 560 per image). Its size must not exceed `RING_SLACK`.
pub const VISION_BATCH: usize = 576;
/// Where the token ids of a batch start in the state buffer, and where the per-token image limits start after them.
pub const STATE_TOKENS: usize = 8;
pub const STATE_VISION: usize = STATE_TOKENS + VISION_BATCH;
/// Where the expert masks start: one row of `EXPERT_MASK_STRIDE` words per mixture-of-experts layer (in the order the layers
/// appear in the graph); a word that is not zero blocks that expert from the router of the layer. All zero (the initial state of the
/// buffer) lets every expert be chosen.
pub const STATE_EXPERT_MASK: usize = STATE_VISION + VISION_BATCH;
pub const EXPERT_MASK_STRIDE: usize = 128;
pub const MAX_EXPERT_LAYERS: usize = 64;
/// Words of the state buffer.
pub const STATE_WORDS: usize = STATE_EXPERT_MASK + EXPERT_MASK_STRIDE * MAX_EXPERT_LAYERS;

/// An argument of a dispatch: it takes the next index of the kernel (a buffer, or a small constant copied by value).
#[derive(Clone, Debug, PartialEq)]
pub enum Arg {
    Value(Value),
    /// A value starting `bytes` after its first element.
    ValueAt(Value, usize),
    Weight(String),
    /// The part of an expert tensor that lives in pinned host memory (CUDA with two tiers of experts; the weight itself otherwise).
    WeightHost(String),
    /// Number of expert slots that live in device memory (the rest are in the host part); larger than any slot when all do.
    DeviceSlots,
    CacheKey(usize),
    CacheValue(usize),
    Table(usize),
    /// The extra embeddings buffer: rows an image's soft tokens are read from.
    Extra,
    /// A scratch buffer of the plan (by index in `Plan::scratch`).
    Scratch(usize),
    State,
    /// Counters of how often each expert of each layer was among the ones the router wanted (see the `route` kernel).
    Demand,
    U32(u32),
    I64(i64),
    U64(u64),
    F32(f32),
}

#[derive(Clone, Debug)]
pub struct Step {
    pub kernel: &'static str,
    pub args: Vec<Arg>,
    pub groups: [usize; 3],
    pub threads: [usize; 3],
    /// Bytes of threadgroup memory the kernel is given at index 0 (0 when it declares none).
    pub shared_memory: usize,
}

#[derive(Clone, Debug)]
pub struct CacheSpec {
    pub layer: usize,
    /// Elements of each of the key and value buffers.
    pub elements: usize,
}

/// Bytes the key and value caches of one conversation take: `(fixed, per_context_token)`. The sliding layers keep a ring of a window
/// plus `RING_SLACK` positions whatever the context; the global layers grow with it. Half-precision keys and values.
pub fn cache_footprint(graph: &crate::ir::Graph) -> (usize, usize) {
    let (mut fixed, mut per_token) = (0usize, 0usize);
    for op in &graph.ops {
        if let Op::CacheWrite { kv_heads, head_dim, window, .. } = op {
            let row = kv_heads * head_dim * 2 /* key and value */ * 2 /* bytes of a half */;
            match window {
                Some(window) => fixed += row * (window + RING_SLACK),
                None => per_token += row,
            }
        }
    }
    (fixed, per_token)
}

/// Makes the expert products of `graph` read from a pool of `slots` experts per layer instead of from all the experts of the model: the
/// host (see `Runtime::with_expert_pool`) decides which experts fill the slots and can change them between tokens. The router still scores
/// every expert; it blocks the ones that are not in a slot and renormalizes among the rest.
pub fn use_expert_pool(graph: &mut crate::ir::Graph, slots: usize) {
    for op in &mut graph.ops {
        if let Op::ExpertsLinear { total, .. } = op {
            *total = slots;
        }
    }
}

/// What the compiler writes for one model: the kernels, the dispatches of a batch and the data they need.
#[derive(Clone, Debug)]
pub struct Plan {
    pub source: String,
    /// Tokens the plan processes in one run.
    pub batch: usize,
    pub steps: Vec<Step>,
    /// Elements of each value's buffer (a value that shares another's buffer is listed under `canonical[value]`).
    pub lens: Vec<usize>,
    /// The value that owns the buffer of each value: aliases and values that reuse a buffer point at one owner.
    pub canonical: Vec<Value>,
    pub caches: Vec<CacheSpec>,
    /// Constant tables computed at compile time (rotary frequencies).
    pub tables: Vec<Vec<f32>>,
    /// Bytes of each scratch buffer.
    pub scratch: Vec<usize>,
    /// Names of the weights the steps read.
    pub weights: Vec<String>,
    /// Positions the global attention caches hold.
    pub max_context: usize,
    /// The value holding the logits of the last position, and the cap the model applies to them (`cap * tanh(x / cap)`).
    pub logits: Option<Value>,
    pub softcap: Option<f32>,
}

struct Builder<'a> {
    target: Target,
    gguf: &'a Gguf,
    plan: Plan,
    /// Elements of each value for one token.
    lens: Vec<usize>,
    batch: usize,
    tables: HashMap<(usize, u32, Option<String>), usize>,
    weights: BTreeSet<String>,
    /// The `ids` value whose experts were last listed by `kernel_moe_route_rows` (the lists stay valid for every product that uses them).
    routed: Option<Value>,
    counts: Option<usize>,
    rows: Option<usize>,
    /// Mixture-of-experts layers seen so far (the router of each reads its own row of the expert mask).
    routes: usize,
}

fn blocks(items: usize, per_group: usize) -> usize {
    items.div_ceil(per_group)
}

fn mm_id_kernel(encoding: Encoding, name: &str) -> Result<&'static str> {
    Ok(match encoding {
        Encoding::Q4K => "mm_id_q4_k",
        Encoding::Q5_1 => "mm_id_q5_1",
        Encoding::Q8_0 => "mm_id_q8_0",
        other => bail!("{name}: no batched expert kernel for {other:?}"),
    })
}

impl Builder<'_> {
    fn weight(&mut self, weight: &Weight) -> Arg {
        self.weights.insert(weight.name.clone());
        Arg::Weight(weight.name.clone())
    }

    fn step(&mut self, kernel: &'static str, args: Vec<Arg>, groups: [usize; 3], threads: [usize; 3]) {
        self.plan.steps.push(Step { kernel, args, groups, threads, shared_memory: 0 });
    }

    fn scratch(&mut self, bytes: usize) -> usize {
        self.plan.scratch.push(bytes);
        self.plan.scratch.len() - 1
    }

    /// The table of inverse rotary frequencies of a head width, made once per (width, base, factors).
    fn rope_table(&mut self, head_dim: usize, base: f32, factors: &Option<Weight>) -> Result<usize> {
        let key = (head_dim, base.to_bits(), factors.as_ref().map(|factors| factors.name.clone()));
        if let Some(&index) = self.tables.get(&key) {
            return Ok(index);
        }
        let mut inverse: Vec<f32> =
            (0..head_dim / 2).map(|index| 1f32 / (base as f64).powf((index * 2) as f64 / head_dim as f64) as f32).collect();
        if let Some(factors) = factors {
            let tensor = self.gguf.tensor(&factors.name).ok_or_else(|| anyhow::anyhow!("no tensor {}", factors.name))?;
            ensure!(tensor.encoding == Encoding::F32 && tensor.values() == head_dim / 2, "unexpected rotary factors");
            for (value, bytes) in inverse.iter_mut().zip(self.gguf.data(tensor).chunks_exact(4)) {
                *value /= f32::from_le_bytes(bytes.try_into()?);
            }
        }
        self.plan.tables.push(inverse);
        let index = self.plan.tables.len() - 1;
        self.tables.insert(key, index);
        Ok(index)
    }

    fn op(&mut self, op: &Op, max_context: usize) -> Result<()> {
        let batch = self.batch;
        match op {
            Op::EmbedRow { table, scale, output } => {
                ensure!(table.encoding == Encoding::Q8_0, "embedding table must be Q8_0");
                let width = table.dims[0];
                let weight = self.weight(table);
                self.step(
                    "embed_q8",
                    vec![weight, Arg::State, Arg::Value(*output), Arg::U32(width as u32), Arg::F32(*scale), Arg::Extra],
                    [blocks(width, 64), batch, 1],
                    [64, 1, 1],
                );
            }
            Op::RmsNorm { scale, input, output, groups, epsilon } => {
                let width = self.lens[*input] / groups;
                let (weight, has_scale) = match scale {
                    Some(scale) => (self.weight(scale), 1),
                    None => (Arg::Value(*input), 0),
                };
                self.step(
                    "rms_norm",
                    vec![Arg::Value(*input), weight, Arg::Value(*output), Arg::U32(width as u32), Arg::F32(*epsilon), Arg::U32(has_scale)],
                    [groups * batch, 1, 1],
                    [256, 1, 1],
                );
            }
            Op::Linear { weight, input, output } => {
                ensure!(weight.encoding == Encoding::Q8_0, "{}: only Q8_0 matrices are supported ({:?})", weight.name, weight.encoding);
                let (width, rows) = (weight.dims[0], weight.dims[1]);
                let weight = self.weight(weight);
                if batch == 1 {
                    self.step(
                        "mv_q8_0",
                        vec![weight, Arg::Value(*input), Arg::Value(*output), Arg::U32(width as u32), Arg::U32(rows as u32)],
                        [blocks(rows, ROWS_PER_SIMD_GROUP * SIMD_GROUPS), 1, 1],
                        [SIMD_GROUPS * 32, 1, 1],
                    );
                } else {
                    ensure!(width % 32 == 0, "the matrix-matrix kernel needs a width multiple of 32");
                    let row_bytes = (width / 32 * 34) as u64;
                    let (k, t) = (width as u64, batch as u64);
                    self.plan.steps.push(Step {
                        kernel: "mm_q8_0",
                        args: vec![
                            weight,
                            Arg::Value(*input),
                            Arg::Value(*output),
                            Arg::I64(width as i64),
                            Arg::I64(1),
                            Arg::U64(row_bytes),
                            Arg::U64(row_bytes * rows as u64),
                            Arg::U64(row_bytes * rows as u64),
                            Arg::I64(1),
                            Arg::U64(4),
                            Arg::U64(k * 4),
                            Arg::U64(k * 4 * t),
                            Arg::U64(k * 4 * t),
                            Arg::I64(rows as i64),
                            Arg::I64(batch as i64),
                            Arg::U32(1),
                            Arg::U32(1),
                        ],
                        groups: [blocks(batch, 32), blocks(rows, 64), 1],
                        threads: [128, 1, 1],
                        shared_memory: 8192,
                    });
                }
            }
            Op::Rope { input, output, heads, head_dim, base, factors } => {
                let table = self.rope_table(*head_dim, *base, factors)?;
                self.step(
                    "rope",
                    vec![Arg::Value(*input), Arg::Value(*output), Arg::Table(table), Arg::State, Arg::U32(*head_dim as u32), Arg::U32(*heads as u32)],
                    [blocks(head_dim / 2, 32), *heads, batch],
                    [32, 1, 1],
                );
            }
            Op::CacheWrite { layer, key, value, kv_heads, head_dim, window } => {
                let capacity = window.map_or(max_context, |window| window + RING_SLACK);
                self.plan.caches.push(CacheSpec { layer: *layer, elements: kv_heads * capacity * head_dim });
                self.step(
                    "cache_write",
                    vec![
                        Arg::Value(*key),
                        Arg::Value(*value),
                        Arg::CacheKey(*layer),
                        Arg::CacheValue(*layer),
                        Arg::State,
                        Arg::U32(*head_dim as u32),
                        Arg::U32(capacity as u32),
                        Arg::U32(*kv_heads as u32),
                    ],
                    [blocks(*head_dim, 32), *kv_heads, batch],
                    [32, 1, 1],
                );
            }
            Op::Attention { layer, kind, heads, kv_heads, head_dim, query, output } => {
                let (capacity, window) = match kind {
                    AttentionKind::Sliding { window } => (*window + RING_SLACK, *window),
                    AttentionKind::Global => (max_context, 0),
                };
                let kernel = match head_dim {
                    256 => "attention_256",
                    512 => "attention_512",
                    other => bail!("no attention kernel for heads of {other} values"),
                };
                self.step(
                    kernel,
                    vec![
                        Arg::Value(*query),
                        Arg::CacheKey(*layer),
                        Arg::CacheValue(*layer),
                        Arg::Value(*output),
                        Arg::State,
                        Arg::U32(*heads as u32),
                        Arg::U32(*kv_heads as u32),
                        Arg::U32(capacity as u32),
                        Arg::U32(window as u32),
                    ],
                    [*heads, batch, 1],
                    [256, 1, 1],
                );
            }
            Op::GeluMul { gate, up, output } => {
                let n = self.lens[*output] * batch;
                self.step(
                    "gelu_mul",
                    vec![Arg::Value(*gate), Arg::Value(*up), Arg::Value(*output), Arg::U32(n as u32), Arg::U32(0), Arg::U32(0), Arg::U32(0)],
                    [blocks(n, 64), 1, 1],
                    [64, 1, 1],
                );
            }
            Op::GeluMulFused { input, output, slots, width } => {
                self.step(
                    "gelu_mul",
                    vec![
                        Arg::Value(*input),
                        Arg::ValueAt(*input, width * 4),
                        Arg::Value(*output),
                        Arg::U32(*width as u32),
                        Arg::U32(2 * *width as u32),
                        Arg::U32(2 * *width as u32),
                        Arg::U32(*width as u32),
                    ],
                    [blocks(*width, 64), slots * batch, 1],
                    [64, 1, 1],
                );
            }
            Op::Add { left, right, output } => {
                let n = self.lens[*output] * batch;
                self.step("add", vec![Arg::Value(*left), Arg::Value(*right), Arg::Value(*output), Arg::U32(n as u32)], [blocks(n, 64), 1, 1], [64, 1, 1]);
            }
            Op::Alias { .. } => {}
            Op::ScaleBy { scale, input, output } => {
                let n = self.lens[*output] * batch;
                let weight = self.weight(scale);
                self.step("scale_by", vec![Arg::Value(*input), weight, Arg::Value(*output), Arg::U32(n as u32)], [blocks(n, 64), 1, 1], [64, 1, 1]);
            }
            Op::Route { router, router_scale, expert_scale, input, used, total, epsilon, ids, weights } => {
                ensure!(router.encoding == Encoding::F32, "the router matrix must be F32");
                let width = router.dims[0];
                let (router_weight, scale_weight, expert_weight) = (self.weight(router), self.weight(router_scale), self.weight(expert_scale));
                ensure!(self.routes < MAX_EXPERT_LAYERS && *total <= EXPERT_MASK_STRIDE, "too many experts or expert layers for the mask");
                let mask_at = STATE_EXPERT_MASK + self.routes * EXPERT_MASK_STRIDE;
                self.routes += 1;
                self.step(
                    "route",
                    vec![
                        Arg::Value(*input),
                        scale_weight,
                        router_weight,
                        expert_weight,
                        Arg::Value(*ids),
                        Arg::Value(*weights),
                        Arg::U32(width as u32),
                        Arg::U32(*total as u32),
                        Arg::U32(*used as u32),
                        Arg::F32(*epsilon),
                        Arg::State,
                        Arg::U32(mask_at as u32),
                        Arg::Demand,
                        Arg::U32((mask_at - STATE_EXPERT_MASK) as u32),
                    ],
                    [batch, 1, 1],
                    [1024, 1, 1],
                );
            }
            Op::ExpertsLinear { weight, input, per_slot_input, ids, used, total, rows, width, output } => {
                if batch == 1 {
                    let kernel = match weight.encoding {
                        Encoding::Q4K => "mv_id_q4_k",
                        Encoding::Q5_1 => "mv_id_q5_1",
                        Encoding::Q8_0 => "mv_id_q8_0",
                        other => bail!("{}: no expert kernel for {other:?}", weight.name),
                    };
                    let host = Arg::WeightHost(weight.name.clone());
                    let weight = self.weight(weight);
                    let mut arguments = vec![
                        weight,
                        Arg::Value(*ids),
                        Arg::Value(*input),
                        Arg::Value(*output),
                        Arg::U32(*width as u32),
                        Arg::U32(*rows as u32),
                        Arg::U32(if *per_slot_input { *width as u32 } else { 0 }),
                    ];
                    if self.target == Target::Cuda {
                        arguments.extend([host, Arg::DeviceSlots]);
                    }
                    self.step(
                        kernel,
                        arguments,
                        [blocks(*rows, ROWS_PER_SIMD_GROUP * SIMD_GROUPS), *used, 1],
                        [SIMD_GROUPS * 32, 1, 1],
                    );
                } else {
                    // The (token, slot) pairs of each expert are listed on the device once per layer; every expert's weights are then
                    // read once for all the tokens that chose it.
                    let kernel = mm_id_kernel(weight.encoding, &weight.name)?;
                    let counts = match self.counts {
                        Some(index) => index,
                        None => {
                            let index = self.scratch(total * 4);
                            self.counts = Some(index);
                            index
                        }
                    };
                    let rows_list = match self.rows {
                        Some(index) => index,
                        None => {
                            let index = self.scratch(total * batch * 4);
                            self.rows = Some(index);
                            index
                        }
                    };
                    if self.routed != Some(*ids) {
                        self.step("zero_u32", vec![Arg::Scratch(counts), Arg::U32(*total as u32)], [blocks(*total, 32), 1, 1], [32, 1, 1]);
                        self.step(
                            "kernel_moe_route_rows",
                            vec![
                                Arg::Value(*ids),
                                Arg::Scratch(counts),
                                Arg::Scratch(rows_list),
                                Arg::I64(*used as i64),
                                Arg::I64(batch as i64),
                                Arg::I64(*total as i64),
                            ],
                            [blocks(batch * used, 64), 1, 1],
                            [64, 1, 1],
                        );
                        self.routed = Some(*ids);
                    }
                    let (per_block, block_bytes) = weight.encoding.block();
                    ensure!(width % 32 == 0 && width % per_block == 0, "{}: width {width} does not fit the batched kernel", weight.name);
                    let row_bytes = (width / per_block * block_bytes) as u64;
                    let input_rows = if *per_slot_input { *used } else { 1 };
                    let host = Arg::WeightHost(weight.name.clone());
                    let weight = self.weight(weight);
                    let shared = (8192 + batch * 4).div_ceil(16) * 16;
                    ensure!(shared <= 32768, "the batch is too large for the batched expert kernel");
                    let cuda = self.target == Target::Cuda;
                    self.plan.steps.push(Step {
                        kernel,
                        args: [vec![
                            weight,
                            Arg::Value(*input),
                            Arg::Value(*output),
                            Arg::Scratch(rows_list),
                            Arg::Scratch(counts),
                            Arg::I64(batch as i64),
                            Arg::I64(*width as i64),
                            Arg::I64(*total as i64),
                            Arg::U64(row_bytes),
                            Arg::U64(row_bytes * *rows as u64),
                            Arg::I64(input_rows as i64),
                            Arg::I64(batch as i64),
                            Arg::U64(4),
                            Arg::U64(*width as u64 * 4),
                            Arg::U64(*width as u64 * 4 * input_rows as u64),
                            Arg::I64(*rows as i64),
                            Arg::I64(*used as i64),
                        ], if cuda { vec![host, Arg::DeviceSlots] } else { Vec::new() }].concat(),
                        groups: [blocks(batch, 32), blocks(*rows, 64), *total],
                        threads: [128, 1, 1],
                        shared_memory: shared,
                    });
                }
            }
            Op::ExpertsSum { input, weights, output, slots, width } => {
                self.step(
                    "experts_sum",
                    vec![Arg::Value(*input), Arg::Value(*weights), Arg::Value(*output), Arg::U32(*width as u32), Arg::U32(*slots as u32)],
                    [blocks(*width, 64), batch, 1],
                    [64, 1, 1],
                );
            }
            Op::Head { weight, input, logits, softcap } => {
                ensure!(weight.encoding == Encoding::Q8_0, "the output matrix must be Q8_0");
                // Only a plan of one token chooses a token: a batch is for reading a prompt and leaves the last token to it.
                if batch > 1 {
                    return Ok(());
                }
                self.plan.logits = Some(*logits);
                self.plan.softcap = *softcap;
                let (width, rows) = (weight.dims[0], weight.dims[1]);
                let weight = self.weight(weight);
                self.step(
                    "mv_q8_0",
                    vec![weight, Arg::Value(*input), Arg::Value(*logits), Arg::U32(width as u32), Arg::U32(rows as u32)],
                    [blocks(rows, ROWS_PER_SIMD_GROUP * SIMD_GROUPS), 1, 1],
                    [SIMD_GROUPS * 32, 1, 1],
                );
                self.step("argmax_token", vec![Arg::Value(*logits), Arg::State, Arg::U32(rows as u32)], [1, 1, 1], [1024, 1, 1]);
            }
        }
        Ok(())
    }
}

/// Every value an operation reads or writes.
fn op_values(op: &Op) -> Vec<Value> {
    match op {
        Op::EmbedRow { output, .. } => vec![*output],
        Op::RmsNorm { input, output, .. } | Op::Linear { input, output, .. } | Op::ScaleBy { input, output, .. } | Op::Rope { input, output, .. } => {
            vec![*input, *output]
        }
        Op::CacheWrite { key, value, .. } => vec![*key, *value],
        Op::Attention { query, output, .. } => vec![*query, *output],
        Op::GeluMul { gate, up, output } => vec![*gate, *up, *output],
        Op::GeluMulFused { input, output, .. } => vec![*input, *output],
        Op::Add { left, right, output } => vec![*left, *right, *output],
        Op::Alias { input, output } => vec![*input, *output],
        Op::Route { input, ids, weights, .. } => vec![*input, *ids, *weights],
        Op::ExpertsLinear { input, ids, output, .. } => vec![*input, *ids, *output],
        Op::ExpertsSum { input, weights, output, .. } => vec![*input, *weights, *output],
        Op::Head { input, logits, .. } => vec![*input, *logits],
    }
}

/// Which buffer each value lives in: aliases share one, and with `reuse` a value takes a buffer another value of the same size no longer needs.
fn assign_buffers(graph: &Graph, reuse: bool) -> Vec<Value> {
    let count = graph.values();
    let mut root: Vec<Value> = (0..count).collect();
    for op in &graph.ops {
        if let Op::Alias { input, output } = op {
            root[*output] = root[*input];
        }
    }
    let mut first = vec![usize::MAX; count];
    let mut last = vec![0usize; count];
    for (index, op) in graph.ops.iter().enumerate() {
        for value in op_values(op) {
            let value = root[value];
            first[value] = first[value].min(index);
            last[value] = last[value].max(index);
        }
    }
    let mut owner: Vec<Value> = (0..count).collect();
    if reuse {
        let mut free: HashMap<usize, Vec<Value>> = HashMap::new();
        let mut ending: HashMap<usize, Vec<Value>> = HashMap::new();
        for value in 0..count {
            if root[value] == value && first[value] != usize::MAX {
                ending.entry(last[value]).or_default().push(value);
            }
        }
        for (index, op) in graph.ops.iter().enumerate() {
            let mut seen = BTreeSet::new();
            for value in op_values(op) {
                let value = root[value];
                if first[value] == index && seen.insert(value) {
                    if let Some(buffer) = free.get_mut(&graph.lens[value]).and_then(Vec::pop) {
                        owner[value] = buffer;
                    }
                }
            }
            for value in ending.remove(&index).unwrap_or_default() {
                free.entry(graph.lens[value]).or_default().push(owner[value]);
            }
        }
    }
    (0..count).map(|value| owner[root[value]]).collect()
}

/// Compiles the graph of a token into a plan for `batch` tokens. `max_context` is the positions the global attention caches
/// hold; with `reuse` the plan's activations share buffers once their values are dead (without it every value keeps its own,
/// which tests read back).
pub fn compile(graph: &Graph, gguf: &Gguf, max_context: usize, batch: usize, reuse: bool) -> Result<Plan> {
    compile_for(Target::Metal, graph, gguf, max_context, batch, reuse)
}

/// The GPU language a plan's kernels are written in. The dispatches are the same: both sets of kernels take the same arguments in the same
/// order and a thread group is a block.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Target {
    /// Metal Shading Language (Apple GPUs).
    Metal,
    /// CUDA C++, compiled at run time by NVRTC (NVIDIA GPUs).
    Cuda,
}

impl Target {
    /// The GPU language of the machine this program runs on: Metal on a Mac, CUDA elsewhere.
    pub fn native() -> Self {
        if cfg!(target_os = "macos") { Target::Metal } else { Target::Cuda }
    }
}

/// [`compile`] for a given [`Target`].
pub fn compile_for(target: Target, graph: &Graph, gguf: &Gguf, max_context: usize, batch: usize, reuse: bool) -> Result<Plan> {
    ensure!((1..=VISION_BATCH).contains(&batch), "a plan takes between 1 and {VISION_BATCH} tokens");
    let canonical = assign_buffers(graph, reuse);
    let mut lens = vec![0; graph.values()];
    for (value, &owner) in canonical.iter().enumerate() {
        lens[owner] = lens[owner].max(graph.lens[value] * batch);
    }
    let mut builder = Builder {
        target,
        gguf,
        plan: Plan {
            source: String::new(),
            batch,
            steps: Vec::new(),
            lens,
            canonical,
            caches: Vec::new(),
            tables: Vec::new(),
            scratch: Vec::new(),
            weights: Vec::new(),
            max_context,
            logits: None,
            softcap: None,
        },
        lens: graph.lens.clone(),
        batch,
        tables: HashMap::new(),
        weights: Default::default(),
        routed: None,
        counts: None,
        rows: None,
        routes: 0,
    };
    for op in &graph.ops {
        builder.op(op, max_context)?;
    }
    let experts = graph.ops.iter().filter_map(|op| if let Op::Route { total, .. } = op { Some(*total) } else { None }).max().unwrap_or(1);
    let used = graph.ops.iter().filter_map(|op| if let Op::Route { used, .. } = op { Some(*used) } else { None }).max().unwrap_or(1);
    let router_width = graph.ops.iter().filter_map(|op| if let Op::Route { router, .. } = op { Some(router.dims[0]) } else { None }).max().unwrap_or(1);
    ensure!(router_width <= 4096, "the router input is wider than the kernel supports");
    ensure!(experts <= 128, "the router kernel supports at most 128 experts");
    let header = format!(
        "// Generated by dafo-compiler-llm.\n#define NR {ROWS_PER_SIMD_GROUP}\n#define NSG {SIMD_GROUPS}\n#define STATE_VISION {STATE_VISION}\n#define ROUTE_MAX_WIDTH {}\n#define ROUTE_MAX_EXPERTS {experts}\n#define ROUTE_MAX_USED {used}\n",
        router_width.max(1)
    );
    let mut plan = builder.plan;
    plan.weights = builder.weights.into_iter().collect();
    plan.source = match target {
        Target::Metal => format!("{header}{KERNELS}\n{MATMUL}"),
        Target::Cuda => format!("{header}{CUDA_KERNELS}"),
    };
    Ok(plan)
}

/// The batch sizes the engine reads a prompt with, largest first (a prompt of any length ends on the one-token plan).
pub const PROMPT_BATCHES: [usize; 2] = [128, 16];

/// The plans a runtime needs: the one-token plan, then one per prompt batch size and, with `vision`, the plan that reads an image.
pub fn compile_plans(graph: &Graph, gguf: &Gguf, max_context: usize, vision: bool) -> Result<Vec<Plan>> {
    compile_plans_for(Target::Metal, graph, gguf, max_context, vision)
}

/// [`compile_plans`] for a given [`Target`].
pub fn compile_plans_for(target: Target, graph: &Graph, gguf: &Gguf, max_context: usize, vision: bool) -> Result<Vec<Plan>> {
    let mut plans = vec![compile_for(target, graph, gguf, max_context, 1, true)?];
    for batch in PROMPT_BATCHES {
        plans.push(compile_for(target, graph, gguf, max_context, batch, true)?);
    }
    if vision {
        plans.push(compile_for(target, graph, gguf, max_context, VISION_BATCH, true)?);
    }
    Ok(plans)
}

/// What a kernel does with each of its buffers, read from the signatures in the Metal source: `(buffer index, written)`. A buffer
/// declared `constant` or `device const` is only read; a plain `device` one may be written. The compiler needs it to know which
/// dispatches can run at the same time (see [`schedule`]); reading it from the source keeps it right when a kernel changes.
pub fn kernel_buffer_access(source: &str) -> HashMap<String, Vec<(usize, bool)>> {
    // The text between the parenthesis that opens at `open` and its match.
    fn balanced(text: &str, open: usize) -> Option<&str> {
        let mut depth = 0usize;
        for (offset, character) in text[open..].char_indices() {
            match character {
                '(' => depth += 1,
                ')' => {
                    depth -= 1;
                    if depth == 0 {
                        return Some(&text[open + 1..open + offset]);
                    }
                }
                _ => {}
            }
        }
        None
    }
    fn split_parameters(list: &str) -> Vec<&str> {
        let (mut parts, mut depth, mut start) = (Vec::new(), 0i32, 0);
        for (index, character) in list.char_indices() {
            match character {
                '(' | '[' | '<' => depth += 1,
                ')' | ']' | '>' => depth -= 1,
                ',' if depth == 0 => {
                    parts.push(&list[start..index]);
                    start = index + 1;
                }
                _ => {}
            }
        }
        parts.push(&list[start..]);
        parts
    }
    // Every `kernel void NAME(...)`: the parameters of each function, templates included.
    let mut functions: HashMap<&str, &str> = HashMap::new();
    let mut from = 0;
    while let Some(found) = source[from..].find("kernel void ") {
        let name_start = from + found + "kernel void ".len();
        let name_end = name_start + source[name_start..].find(|c: char| !(c.is_alphanumeric() || c == '_')).unwrap_or(0);
        if let Some(open) = source[name_end..].find('(').map(|offset| name_end + offset)
            && source[name_end..open].trim().is_empty()
            && let Some(parameters) = balanced(source, open)
        {
            functions.insert(&source[name_start..name_end], parameters);
        }
        from = name_end.max(from + 1);
    }
    // Host names of the instantiations of templates: `[[host_name("mm_q8_0")]] kernel TYPE kernel_mul_mm<...>`.
    let mut hosts: Vec<(String, &str)> = functions.keys().map(|name| (name.to_string(), *name)).collect();
    let mut from = 0;
    while let Some(found) = source[from..].find("[[host_name(\"") {
        let start = from + found + "[[host_name(\"".len();
        let Some(end) = source[start..].find('"').map(|offset| start + offset) else { break };
        let rest = &source[end..];
        if let Some(kernel) = rest.find("kernel ").filter(|&offset| offset < 80) {
            let mut words = rest[kernel + "kernel ".len()..].split_whitespace();
            let _type = words.next();
            if let Some(word) = words.next() {
                let function = word.split('<').next().unwrap_or(word);
                if let Some((&key, _)) = functions.get_key_value(function) {
                    hosts.push((source[start..end].to_owned(), key));
                }
            }
        }
        from = end;
    }
    let mut access = HashMap::new();
    for (host, function) in hosts {
        let Some(parameters) = functions.get(function) else { continue };
        let mut entries: Vec<(Option<usize>, bool)> = Vec::new();
        for parameter in split_parameters(parameters) {
            let parameter = parameter.trim();
            // Built-in values and threadgroup memory are not buffers.
            let explicit = parameter.find("[[buffer(").map(|at| {
                let digits: String = parameter[at + "[[buffer(".len()..].chars().take_while(char::is_ascii_digit).collect();
                digits.parse::<usize>().unwrap_or(0)
            });
            if parameter.contains("[[") && explicit.is_none() {
                continue;
            }
            let declaration = parameter.split("[[").next().unwrap_or(parameter);
            let written = declaration.contains("device") && !declaration.contains("const");
            entries.push((explicit, written));
        }
        // Parameters without an index take the lowest free ones, in order.
        let taken: std::collections::HashSet<usize> = entries.iter().filter_map(|(index, _)| *index).collect();
        let mut free = (0..).filter(|index| !taken.contains(index));
        let resolved = entries.into_iter().map(|(index, written)| (index.unwrap_or_else(|| free.next().expect("an index")), written)).collect();
        access.insert(host, resolved);
    }
    access
}

/// Orders `accesses` (for each dispatch, the resources it reads and the ones it writes) into groups that can run at the same time:
/// a dispatch joins the first group it does not conflict with (read after write, write after read, write after write), looking at
/// most `window` dispatches ahead, and never moves past a dispatch it conflicts with. A barrier goes between groups. The result
/// keeps every dependency of the program order, so it computes the same values.
pub fn schedule(accesses: &[(Vec<u64>, Vec<u64>)], window: usize) -> Vec<Vec<usize>> {
    use std::collections::HashSet;
    let mut remaining: Vec<usize> = (0..accesses.len()).collect();
    let mut groups = Vec::new();
    while !remaining.is_empty() {
        let (mut group_reads, mut group_writes): (HashSet<u64>, HashSet<u64>) = (HashSet::new(), HashSet::new());
        let (mut skipped_reads, mut skipped_writes): (HashSet<u64>, HashSet<u64>) = (HashSet::new(), HashSet::new());
        let (mut taken, mut left) = (Vec::new(), Vec::new());
        for (position, &index) in remaining.iter().enumerate() {
            if position >= window {
                left.push(index);
                continue;
            }
            let (reads, writes) = &accesses[index];
            let conflicts = |other_reads: &HashSet<u64>, other_writes: &HashSet<u64>| {
                reads.iter().any(|resource| other_writes.contains(resource)) || writes.iter().any(|resource| other_writes.contains(resource) || other_reads.contains(resource))
            };
            if !conflicts(&group_reads, &group_writes) && !conflicts(&skipped_reads, &skipped_writes) {
                group_reads.extend(reads.iter().copied());
                group_writes.extend(writes.iter().copied());
                taken.push(index);
            } else {
                skipped_reads.extend(reads.iter().copied());
                skipped_writes.extend(writes.iter().copied());
                left.push(index);
            }
        }
        groups.push(taken);
        remaining = left;
    }
    groups
}

#[cfg(test)]
mod schedule_tests {
    use super::*;

    #[test]
    fn independent_dispatches_share_a_group_and_dependent_ones_do_not() {
        // 0 writes A; 1 and 2 read A and write B and C; 3 reads B and C and writes D; 4 is unrelated to all.
        let accesses = vec![(vec![], vec![1]), (vec![1], vec![2]), (vec![1], vec![3]), (vec![2, 3], vec![4]), (vec![9], vec![8])];
        let groups = schedule(&accesses, 16);
        assert_eq!(groups, vec![vec![0, 4], vec![1, 2], vec![3]]);
    }

    #[test]
    fn a_write_after_a_read_waits_for_the_reader() {
        // 0 reads A; 1 writes A (a reused buffer): they cannot run together, and 1 cannot be moved before 0.
        let groups = schedule(&[(vec![1], vec![2]), (vec![], vec![1])], 16);
        assert_eq!(groups, vec![vec![0], vec![1]]);
    }

    #[test]
    fn a_dispatch_never_moves_past_one_it_depends_on() {
        // 0 writes A, 1 reads A, 2 is independent: 2 may join 0, but 1 stays after 0.
        let groups = schedule(&[(vec![], vec![1]), (vec![1], vec![2]), (vec![], vec![3])], 16);
        assert_eq!(groups, vec![vec![0, 2], vec![1]]);
    }

    #[test]
    fn kernel_signatures_say_which_buffers_are_written() {
        let source = r#"
            kernel void scale(device const float * input [[buffer(0)]], device float * output [[buffer(1)]], constant uint & n [[buffer(2)]], uint gid [[thread_position_in_grid]]) {}
            template<typename T>
            kernel void mat(device const uchar * weights, device float * dst, constant int64_t & k, threadgroup uchar * shared [[threadgroup(0)]], uint3 tg [[threadgroup_position_in_grid]]) {}
            template [[host_name("mat_f")]] kernel mat_t mat<float>;
        "#;
        let access = kernel_buffer_access(source);
        assert_eq!(access["scale"], vec![(0, false), (1, true), (2, false)]);
        assert_eq!(access["mat_f"], vec![(0, false), (1, true), (2, false)]);
    }
}
