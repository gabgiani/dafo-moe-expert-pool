//! Gemma 4 (dense and mixture-of-experts variants): the graph of one token, read from the tensors and metadata of the GGUF.
//!
//! One decoder layer: attention (norm, q/k/v, per-head norms, rotary, cache, attention, output, post norm, residual), the
//! dense MLP and, when the layer has expert tensors, the routed experts (normalized separately and added), then the post
//! norm, the residual and the layer output scale. Finally the output norm and the head.

use anyhow::{Context, Result, ensure};

use crate::gguf::{Gguf, Scalar};
use crate::ir::{AttentionKind, Graph, Op, Value, Weight};

fn weight(gguf: &Gguf, name: &str) -> Result<Weight> {
    let tensor = gguf.tensor(name).with_context(|| format!("the model has no tensor {name}"))?;
    Ok(Weight { name: name.to_string(), dims: tensor.dims.clone(), encoding: tensor.encoding, bytes: tensor.bytes() })
}

fn optional_weight(gguf: &Gguf, name: &str) -> Result<Option<Weight>> {
    gguf.tensor(name).map(|_| weight(gguf, name)).transpose()
}

pub fn token_graph(gguf: &Gguf) -> Result<Graph> {
    let layers = gguf.unsigned("gemma4.block_count").context("gemma4.block_count")?;
    let hidden_size = gguf.unsigned("gemma4.embedding_length").context("gemma4.embedding_length")?;
    let heads = gguf.unsigned("gemma4.attention.head_count").context("gemma4.attention.head_count")?;
    let window = gguf.unsigned("gemma4.attention.sliding_window").context("gemma4.attention.sliding_window")?;
    let global_head_dim = gguf.unsigned("gemma4.attention.key_length").context("gemma4.attention.key_length")?;
    let sliding_head_dim = gguf.unsigned("gemma4.attention.key_length_swa").unwrap_or(global_head_dim);
    let global_base = gguf.float("gemma4.rope.freq_base").context("gemma4.rope.freq_base")? as f32;
    let sliding_base = gguf.float("gemma4.rope.freq_base_swa").unwrap_or(global_base as f64) as f32;
    let epsilon = gguf.float("gemma4.attention.layer_norm_rms_epsilon").unwrap_or(1e-6) as f32;
    let used = gguf.unsigned("gemma4.expert_used_count").unwrap_or(0);
    let total = gguf.unsigned("gemma4.expert_count").unwrap_or(0);
    let softcap = gguf.float("gemma4.final_logit_softcapping").map(|value| value as f32);
    ensure!(gguf.unsigned("gemma4.embedding_length_per_layer_input").unwrap_or(0) == 0, "per-layer inputs are not supported yet");
    ensure!(gguf.unsigned("gemma4.attention.shared_kv_layers").unwrap_or(0) == 0, "shared KV layers are not supported yet");

    let mut graph = Graph::default();
    let mut hidden = graph.value(hidden_size);
    graph.push(Op::EmbedRow { table: weight(gguf, "token_embd.weight")?, scale: (hidden_size as f32).sqrt(), output: hidden });

    for layer in 0..layers {
        let name = |suffix: &str| format!("blk.{layer}.{suffix}");
        let sliding = match gguf.per_layer("gemma4.attention.sliding_window_pattern", layer) {
            Some(Scalar::Bool(value)) => value,
            _ => false,
        };
        let kv_heads = match gguf.per_layer("gemma4.attention.head_count_kv", layer) {
            Some(Scalar::Unsigned(value)) => value as usize,
            Some(Scalar::Signed(value)) if value > 0 => value as usize,
            _ => heads,
        };
        let (head_dim, kind, base, factors) = if sliding {
            (sliding_head_dim, AttentionKind::Sliding { window }, sliding_base, None)
        } else {
            (global_head_dim, AttentionKind::Global, global_base, optional_weight(gguf, "rope_freqs.weight")?)
        };

        // Attention block.
        let normalized = graph.value(hidden_size);
        graph.push(Op::RmsNorm { scale: Some(weight(gguf, &name("attn_norm.weight"))?), input: hidden, output: normalized, groups: 1, epsilon });
        let (query_raw, key_raw, value_raw) = (graph.value(heads * head_dim), graph.value(kv_heads * head_dim), graph.value(kv_heads * head_dim));
        graph.push(Op::Linear { weight: weight(gguf, &name("attn_q.weight"))?, input: normalized, output: query_raw });
        graph.push(Op::Linear { weight: weight(gguf, &name("attn_k.weight"))?, input: normalized, output: key_raw });
        // Global layers have no value matrix: the value is the key projection, before the key norm.
        match optional_weight(gguf, &name("attn_v.weight"))? {
            Some(weight) => graph.push(Op::Linear { weight, input: normalized, output: value_raw }),
            None => graph.push(Op::Alias { input: key_raw, output: value_raw }),
        }
        let (query_normed, key_normed, value_normed) = (graph.value(heads * head_dim), graph.value(kv_heads * head_dim), graph.value(kv_heads * head_dim));
        graph.push(Op::RmsNorm { scale: Some(weight(gguf, &name("attn_q_norm.weight"))?), input: query_raw, output: query_normed, groups: heads, epsilon });
        graph.push(Op::RmsNorm { scale: Some(weight(gguf, &name("attn_k_norm.weight"))?), input: key_raw, output: key_normed, groups: kv_heads, epsilon });
        graph.push(Op::RmsNorm { scale: None, input: value_raw, output: value_normed, groups: kv_heads, epsilon });
        let (query, key) = (graph.value(heads * head_dim), graph.value(kv_heads * head_dim));
        graph.push(Op::Rope { input: query_normed, output: query, heads, head_dim, base, factors: factors.clone() });
        graph.push(Op::Rope { input: key_normed, output: key, heads: kv_heads, head_dim, base, factors });
        graph.push(Op::CacheWrite {
            layer,
            key,
            value: value_normed,
            kv_heads,
            head_dim,
            window: matches!(kind, AttentionKind::Sliding { .. }).then_some(window),
        });
        let context = graph.value(heads * head_dim);
        graph.push(Op::Attention { layer, kind, heads, kv_heads, head_dim, query, output: context });
        let projected = graph.value(hidden_size);
        graph.push(Op::Linear { weight: weight(gguf, &name("attn_output.weight"))?, input: context, output: projected });
        let attention = graph.value(hidden_size);
        graph.push(Op::RmsNorm { scale: Some(weight(gguf, &name("post_attention_norm.weight"))?), input: projected, output: attention, groups: 1, epsilon });
        let residual = graph.value(hidden_size);
        graph.push(Op::Add { left: hidden, right: attention, output: residual });

        // Feed-forward block: dense MLP and, in mixture-of-experts layers, the routed experts.
        let ffn = weight(gguf, &name("ffn_gate.weight"))?;
        let ffn_width = ffn.dims[1];
        let normalized = graph.value(hidden_size);
        graph.push(Op::RmsNorm { scale: Some(weight(gguf, &name("ffn_norm.weight"))?), input: residual, output: normalized, groups: 1, epsilon });
        let (gate, up, activated, dense) = (graph.value(ffn_width), graph.value(ffn_width), graph.value(ffn_width), graph.value(hidden_size));
        graph.push(Op::Linear { weight: ffn, input: normalized, output: gate });
        graph.push(Op::Linear { weight: weight(gguf, &name("ffn_up.weight"))?, input: normalized, output: up });
        graph.push(Op::GeluMul { gate, up, output: activated });
        graph.push(Op::Linear { weight: weight(gguf, &name("ffn_down.weight"))?, input: activated, output: dense });

        let mut mixed: Value = dense;
        if let Some(router) = optional_weight(gguf, &name("ffn_gate_inp.weight"))? {
            let gate_up = weight(gguf, &name("ffn_gate_up_exps.weight"))?;
            let down = weight(gguf, &name("ffn_down_exps.weight"))?;
            let (gate_up_width, gate_up_rows) = (gate_up.dims[0], gate_up.dims[1]);
            let (down_width, down_rows) = (down.dims[0], down.dims[1]);
            let expert_width = gate_up_rows / 2;
            ensure!(down_width == expert_width, "expert matrices do not fit together");
            let shared = graph.value(hidden_size);
            graph.push(Op::RmsNorm { scale: Some(weight(gguf, &name("post_ffw_norm_1.weight"))?), input: dense, output: shared, groups: 1, epsilon });
            let (ids, weights) = (graph.value(used), graph.value(used));
            graph.push(Op::Route {
                router,
                router_scale: weight(gguf, &name("ffn_gate_inp.scale"))?,
                expert_scale: weight(gguf, &name("ffn_down_exps.scale"))?,
                input: residual,
                used,
                total,
                epsilon,
                ids,
                weights,
            });
            let expert_input = graph.value(hidden_size);
            graph.push(Op::RmsNorm { scale: Some(weight(gguf, &name("pre_ffw_norm_2.weight"))?), input: residual, output: expert_input, groups: 1, epsilon });
            let (projected, expert_activated, expert_out, summed) =
                (graph.value(used * gate_up_rows), graph.value(used * expert_width), graph.value(used * down_rows), graph.value(hidden_size));
            graph.push(Op::ExpertsLinear {
                weight: gate_up,
                input: expert_input,
                per_slot_input: false,
                ids,
                used,
                total,
                rows: gate_up_rows,
                width: gate_up_width,
                output: projected,
            });
            graph.push(Op::GeluMulFused { input: projected, output: expert_activated, slots: used, width: expert_width });
            graph.push(Op::ExpertsLinear {
                weight: down,
                input: expert_activated,
                per_slot_input: true,
                ids,
                used,
                total,
                rows: down_rows,
                width: down_width,
                output: expert_out,
            });
            graph.push(Op::ExpertsSum { input: expert_out, weights, output: summed, slots: used, width: down_rows });
            let routed = graph.value(hidden_size);
            graph.push(Op::RmsNorm { scale: Some(weight(gguf, &name("post_ffw_norm_2.weight"))?), input: summed, output: routed, groups: 1, epsilon });
            mixed = graph.value(hidden_size);
            graph.push(Op::Add { left: shared, right: routed, output: mixed });
        }
        let feed_forward = graph.value(hidden_size);
        graph.push(Op::RmsNorm { scale: Some(weight(gguf, &name("post_ffw_norm.weight"))?), input: mixed, output: feed_forward, groups: 1, epsilon });
        let summed = graph.value(hidden_size);
        graph.push(Op::Add { left: residual, right: feed_forward, output: summed });
        hidden = match optional_weight(gguf, &name("layer_output_scale.weight"))? {
            Some(scale) => {
                let scaled = graph.value(hidden_size);
                graph.push(Op::ScaleBy { scale, input: summed, output: scaled });
                scaled
            }
            None => summed,
        };
    }

    let normalized = graph.value(hidden_size);
    graph.push(Op::RmsNorm { scale: Some(weight(gguf, "output_norm.weight")?), input: hidden, output: normalized, groups: 1, epsilon });
    let head = if gguf.tensor("output.weight").is_some() { "output.weight" } else { "token_embd.weight" };
    let head = weight(gguf, head)?;
    let logits = graph.value(head.dims[1]);
    graph.push(Op::Head { weight: head, softcap, input: normalized, logits });
    Ok(graph)
}
