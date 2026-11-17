//! Intermediate representation: the graph of one token of a model and its weights. Nothing here names a model.

use crate::gguf::Encoding;

/// A value flowing between operations (an activation vector or the like).
pub type Value = usize;

/// A weight of the model, by its name in the source file.
#[derive(Clone, Debug, PartialEq)]
pub struct Weight {
    pub name: String,
    pub dims: Vec<usize>,
    pub encoding: Encoding,
    /// Bytes of the whole tensor in the source.
    pub bytes: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AttentionKind {
    /// Attends to the last `window` positions.
    Sliding { window: usize },
    /// Attends to every position.
    Global,
}

#[derive(Clone, Debug)]
pub enum Op {
    /// The row of the embedding table of the current token, multiplied by `scale`.
    EmbedRow { table: Weight, scale: f32, output: Value },
    /// Root-mean-square normalization of `groups` independent groups of the input, with or without a learned scale
    /// (one scale of the group's width, shared by the groups).
    RmsNorm { scale: Option<Weight>, input: Value, output: Value, groups: usize, epsilon: f32 },
    /// Quantized matrix times vector; every row of the matrix is read.
    Linear { weight: Weight, input: Value, output: Value },
    /// Rotary embedding (half-split) of `heads` heads of the token's position. The frequencies are
    /// `1 / base^(2i / head_dim)`, divided by `factors` when given.
    Rope { input: Value, output: Value, heads: usize, head_dim: usize, base: f32, factors: Option<Weight> },
    /// Writes the token's key and value into the cache of `layer` (a ring of `window` positions when sliding).
    CacheWrite { layer: usize, key: Value, value: Value, kv_heads: usize, head_dim: usize, window: Option<usize> },
    Attention { layer: usize, kind: AttentionKind, heads: usize, kv_heads: usize, head_dim: usize, query: Value, output: Value },
    /// `gelu(gate) * up`.
    GeluMul { gate: Value, up: Value, output: Value },
    /// `gelu(gate) * up` for `slots` vectors whose gate and up halves are stored one after the other (`width` each).
    GeluMulFused { input: Value, output: Value, slots: usize, width: usize },
    Add { left: Value, right: Value, output: Value },
    /// `output` is the same data as `input` (no copy).
    Alias { input: Value, output: Value },
    /// Multiplies by the one-element `scale`.
    ScaleBy { scale: Weight, input: Value, output: Value },
    /// Router of a mixture of experts: normalizes the input, scales it, picks `used` of `total` experts and gives their
    /// softmax weights times the per-expert scale.
    Route {
        router: Weight,
        router_scale: Weight,
        expert_scale: Weight,
        input: Value,
        used: usize,
        total: usize,
        epsilon: f32,
        ids: Value,
        weights: Value,
    },
    /// A stacked expert matrix (`total` matrices of `rows` rows of `width` values): only the chosen experts are read. The input
    /// is one vector shared by the slots, or one per slot.
    ExpertsLinear {
        weight: Weight,
        input: Value,
        per_slot_input: bool,
        ids: Value,
        used: usize,
        total: usize,
        rows: usize,
        width: usize,
        output: Value,
    },
    /// Weighted sum of the chosen experts' outputs.
    ExpertsSum { input: Value, weights: Value, output: Value, slots: usize, width: usize },
    /// The output matrix applied to the last position; the chosen token is the highest logit.
    Head { weight: Weight, softcap: Option<f32>, input: Value, logits: Value },
}

#[derive(Clone, Debug, Default)]
pub struct Graph {
    pub ops: Vec<Op>,
    /// Elements of each value.
    pub lens: Vec<usize>,
}

impl Graph {
    pub fn value(&mut self, len: usize) -> Value {
        self.lens.push(len);
        self.lens.len() - 1
    }

    pub fn push(&mut self, op: Op) {
        self.ops.push(op);
    }

    pub fn values(&self) -> usize {
        self.lens.len()
    }

    /// Bytes of weights a token reads, by kind of operation. A stacked expert matrix counts only its chosen experts.
    pub fn bytes_per_token(&self) -> Traffic {
        let mut traffic = Traffic::default();
        for op in &self.ops {
            match op {
                Op::EmbedRow { table, .. } => traffic.embedding += table.bytes / table.dims.last().copied().unwrap_or(1),
                Op::RmsNorm { scale: Some(scale), .. } | Op::ScaleBy { scale, .. } => traffic.small += scale.bytes,
                Op::Linear { weight, .. } => traffic.dense += weight.bytes,
                Op::Route { router, router_scale, expert_scale, .. } => traffic.small += router.bytes + router_scale.bytes + expert_scale.bytes,
                Op::ExpertsLinear { weight, used, total, .. } => traffic.experts += weight.bytes / total * used,
                Op::Head { weight, .. } => traffic.head += weight.bytes,
                _ => {}
            }
        }
        traffic
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Traffic {
    pub dense: usize,
    pub experts: usize,
    pub head: usize,
    pub embedding: usize,
    pub small: usize,
}

impl Traffic {
    pub fn total(&self) -> usize {
        self.dense + self.experts + self.head + self.embedding + self.small
    }
}
