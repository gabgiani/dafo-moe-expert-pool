//! Runs a few tokens on the GPU and checks every operation of the last one against a CPU computation made from the values the
//! GPU read, with the real weights. Prints the operations whose results differ.
//!
//!   cargo run --release --example check_ops -- MODEL.gguf [TOKEN_IDS_COMMA_SEPARATED]

use anyhow::{Context, Result, bail};
use dafo_compiler_llm::backend_metal;
use dafo_compiler_llm::frontends;
use dafo_compiler_llm::gguf::{Encoding, Gguf};
use dafo_compiler_llm::ir::{AttentionKind, Op, Weight};
use dafo_compiler_llm::quant;
use dafo_compiler_llm::runtime::Runtime;

fn rows_of<'a>(gguf: &'a Gguf, weight: &Weight) -> (usize, usize, &'a [u8]) {
    (weight.dims[0], weight.dims[1], gguf.data(gguf.tensor(&weight.name).unwrap()))
}

/// `rows` rows of a (stacked) matrix at byte offset `start` times `x`, in double precision.
fn matvec(encoding: Encoding, data: &[u8], width: usize, rows: usize, x: &[f32]) -> Result<Vec<f32>> {
    let (per_block, block_bytes) = encoding.block();
    let row_bytes = width / per_block * block_bytes;
    let mut values = vec![0f32; width];
    (0..rows)
        .map(|row| {
            quant::dequantize(encoding, &data[row * row_bytes..(row + 1) * row_bytes], &mut values)?;
            Ok(values.iter().zip(x).map(|(&a, &b)| a as f64 * b as f64).sum::<f64>() as f32)
        })
        .collect()
}

fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + (0.797_884_6_f32 * (x + 0.044715 * x * x * x)).tanh())
}

fn f32s(gguf: &Gguf, weight: &Weight) -> Vec<f32> {
    gguf.data(gguf.tensor(&weight.name).unwrap()).chunks_exact(4).map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap())).collect()
}

fn main() -> Result<()> {
    let mut arguments = std::env::args().skip(1);
    let model = arguments.next().context("usage: check_ops MODEL.gguf [TOKENS]")?;
    let tokens: Vec<u32> = match arguments.next() {
        Some(list) => list.split(',').map(|value| value.parse()).collect::<Result<_, _>>()?,
        None => vec![2, 105, 2364, 107],
    };
    let gguf = std::sync::Arc::new(Gguf::open(&model)?);
    let graph = frontends::token_graph(&gguf)?;
    let plan = backend_metal::compile_for(backend_metal::Target::native(), &graph, &gguf, 512, 1, false)?;
    let mut runtime = Runtime::new(&gguf, &[plan], 1)?;
    for (position, &token) in tokens.iter().enumerate() {
        runtime.step(token, position)?;
    }
    let position = tokens.len() - 1;
    let token = tokens[position];
    println!("checked token {token} at position {position}");
    let mut bad = 0;
    let mut checked = 0;
    for (index, op) in graph.ops.iter().enumerate() {
        let read = |value: usize| runtime.read_value(value);
        let (name, got, want): (String, Vec<f32>, Vec<f32>) = match op {
            Op::EmbedRow { table, scale, output } => {
                let (width, _, data) = rows_of(&gguf, table);
                let row_bytes = width / 32 * 34;
                let mut values = vec![0f32; width];
                quant::dequantize(Encoding::Q8_0, &data[token as usize * row_bytes..][..row_bytes], &mut values)?;
                ("EmbedRow".into(), read(*output), values.iter().map(|value| value * scale).collect())
            }
            Op::RmsNorm { scale, input, output, groups, epsilon } => {
                let x = read(*input);
                let width = x.len() / groups;
                let weights = scale.as_ref().map(|scale| f32s(&gguf, scale));
                let mut want = vec![0f32; x.len()];
                for group in 0..*groups {
                    let slice = &x[group * width..(group + 1) * width];
                    let inverse = 1.0 / (slice.iter().map(|value| (*value as f64) * (*value as f64)).sum::<f64>() / width as f64 + *epsilon as f64).sqrt();
                    for i in 0..width {
                        want[group * width + i] = (slice[i] as f64 * inverse * weights.as_ref().map_or(1.0, |weights| weights[i] as f64)) as f32;
                    }
                }
                (format!("RmsNorm({groups})"), read(*output), want)
            }
            Op::Linear { weight, input, output } => {
                let (width, rows, data) = rows_of(&gguf, weight);
                ("Linear".to_string() + &format!(" {}", weight.name), read(*output), matvec(weight.encoding, data, width, rows, &read(*input))?)
            }
            Op::Rope { input, output, heads, head_dim, base, factors } => {
                let x = read(*input);
                let factors = factors.as_ref().map(|factors| f32s(&gguf, factors));
                let mut want = x.clone();
                for head in 0..*heads {
                    for i in 0..head_dim / 2 {
                        let mut inverse = 1f32 / (*base as f64).powf((i * 2) as f64 / *head_dim as f64) as f32;
                        if let Some(factors) = &factors {
                            inverse /= factors[i];
                        }
                        let angle = position as f32 * inverse;
                        let (s, c) = angle.sin_cos();
                        let (x1, x2) = (x[head * head_dim + i], x[head * head_dim + i + head_dim / 2]);
                        want[head * head_dim + i] = x1 * c - x2 * s;
                        want[head * head_dim + i + head_dim / 2] = x2 * c + x1 * s;
                    }
                }
                ("Rope".into(), read(*output), want)
            }
            Op::CacheWrite { layer, key, value, kv_heads, head_dim, window } => {
                let (key_cache, value_cache) = runtime.read_cache(*layer).unwrap();
                let capacity = window.map_or(512, |window| window + backend_metal::RING_SLACK);
                let slot = position % capacity;
                let (mut got, mut want) = (Vec::new(), Vec::new());
                for head in 0..*kv_heads {
                    let start = (head * capacity + slot) * head_dim;
                    got.extend_from_slice(&key_cache[start..start + head_dim]);
                    got.extend_from_slice(&value_cache[start..start + head_dim]);
                    want.extend_from_slice(&read(*key)[head * head_dim..(head + 1) * head_dim]);
                    want.extend_from_slice(&read(*value)[head * head_dim..(head + 1) * head_dim]);
                }
                (format!("CacheWrite layer {layer}"), got, want)
            }
            Op::Attention { layer, kind, heads, kv_heads, head_dim, query, output } => {
                let (key_cache, value_cache) = runtime.read_cache(*layer).unwrap();
                let q = read(*query);
                let (capacity, window) = match kind {
                    AttentionKind::Sliding { window } => (*window + backend_metal::RING_SLACK, *window),
                    AttentionKind::Global => (512, 0),
                };
                let begin = if window > 0 && position + 1 > window { position + 1 - window } else { 0 };
                let mut want = vec![0f32; heads * head_dim];
                for head in 0..*heads {
                    let kv = head / (heads / kv_heads);
                    let at = |p: usize| (kv * capacity + p % capacity) * head_dim;
                    let scores: Vec<f64> = (begin..=position)
                        .map(|p| (0..*head_dim).map(|i| q[head * head_dim + i] as f64 * key_cache[at(p) + i] as f64).sum())
                        .collect();
                    let largest = scores.iter().cloned().fold(f64::MIN, f64::max);
                    let weights: Vec<f64> = scores.iter().map(|score| (score - largest).exp()).collect();
                    let total: f64 = weights.iter().sum();
                    for i in 0..*head_dim {
                        want[head * head_dim + i] = ((begin..=position).zip(&weights).map(|(p, weight)| weight * value_cache[at(p) + i] as f64).sum::<f64>() / total) as f32;
                    }
                }
                (format!("Attention layer {layer} {kind:?}"), read(*output), want)
            }
            Op::GeluMul { gate, up, output } => {
                ("GeluMul".into(), read(*output), read(*gate).iter().zip(read(*up)).map(|(gate, up)| gelu(*gate) * up).collect())
            }
            Op::GeluMulFused { input, output, slots, width } => {
                let x = read(*input);
                let want = (0..*slots).flat_map(|slot| (0..*width).map(move |i| (slot, i))).map(|(slot, i)| gelu(x[slot * 2 * width + i]) * x[slot * 2 * width + width + i]).collect();
                ("GeluMulFused".into(), read(*output), want)
            }
            Op::Add { left, right, output } => ("Add".into(), read(*output), read(*left).iter().zip(read(*right)).map(|(a, b)| a + b).collect()),
            Op::Alias { .. } => continue,
            Op::ScaleBy { scale, input, output } => {
                let factor = f32s(&gguf, scale)[0];
                ("ScaleBy".into(), read(*output), read(*input).iter().map(|value| value * factor).collect())
            }
            Op::Route { router, router_scale, expert_scale, input, used, total, epsilon, ids, weights } => {
                let x = read(*input);
                let inverse = 1.0 / (x.iter().map(|value| (*value as f64).powi(2)).sum::<f64>() / x.len() as f64 + *epsilon as f64).sqrt();
                let scale = f32s(&gguf, router_scale);
                let normalized: Vec<f32> = (0..x.len()).map(|i| (x[i] as f64 * inverse * scale[i] as f64 / (x.len() as f64).sqrt()) as f32).collect();
                let matrix = f32s(&gguf, router);
                let logits: Vec<f32> = (0..*total).map(|row| (0..x.len()).map(|i| normalized[i] as f64 * matrix[row * x.len() + i] as f64).sum::<f64>() as f32).collect();
                let mut order: Vec<usize> = (0..*total).collect();
                order.sort_by(|a, b| logits[*b].total_cmp(&logits[*a]));
                let chosen = &order[..*used];
                let largest = logits[chosen[0]] as f64;
                let exps: Vec<f64> = chosen.iter().map(|&e| (logits[e] as f64 - largest).exp()).collect();
                let sum: f64 = exps.iter().sum();
                let expert = f32s(&gguf, expert_scale);
                let mut want: Vec<f32> = chosen.iter().map(|&e| e as f32).collect();
                want.extend(chosen.iter().zip(&exps).map(|(&e, value)| (value / sum) as f32 * expert[e]));
                // The ids are u32 bit patterns.
                let mut got: Vec<f32> = runtime.read_value(*ids).iter().map(|value| value.to_bits() as f32).collect();
                got.extend(read(*weights));
                ("Route (ids, weights)".into(), got, want)
            }
            Op::ExpertsLinear { weight, input, per_slot_input, ids, used, rows, width, output, .. } => {
                let data = gguf.data(gguf.tensor(&weight.name).unwrap());
                let (per_block, block_bytes) = weight.encoding.block();
                let matrix_bytes = rows * width / per_block * block_bytes;
                let x = read(*input);
                let chosen: Vec<usize> = runtime.read_value(*ids).iter().map(|value| value.to_bits() as usize).collect();
                let mut want = Vec::new();
                for (slot, &expert) in chosen.iter().take(*used).enumerate() {
                    let vector = if *per_slot_input { &x[slot * width..(slot + 1) * width] } else { &x[..*width] };
                    want.extend(matvec(weight.encoding, &data[expert * matrix_bytes..(expert + 1) * matrix_bytes], *width, *rows, vector)?);
                }
                (format!("ExpertsLinear {}", weight.name), read(*output), want)
            }
            Op::ExpertsSum { input, weights, output, slots, width } => {
                let (x, w) = (read(*input), read(*weights));
                let want = (0..*width).map(|i| (0..*slots).map(|s| w[s] * x[s * width + i]).sum()).collect();
                ("ExpertsSum".into(), read(*output), want)
            }
            Op::Head { weight, input, logits, .. } => {
                let (width, rows, data) = rows_of(&gguf, weight);
                let want = matvec(weight.encoding, data, width, rows, &read(*input))?;
                let got = read(*logits);
                let best = |values: &[f32]| values.iter().enumerate().fold((0, f32::MIN), |best, (index, &value)| if value > best.1 { (index, value) } else { best }).0;
                println!("head: GPU chooses {} CPU chooses {}", best(&got), best(&want));
                ("Head logits".into(), got, want)
            }
        };
        if got.len() != want.len() {
            println!("op {index} {name}: sizes differ ({} vs {})", got.len(), want.len());
            bad += 1;
            continue;
        }
        let scale = want.iter().fold(0f32, |largest, value| largest.max(value.abs())).max(1e-6);
        let error = got.iter().zip(&want).fold(0f32, |largest, (a, b)| largest.max((a - b).abs())) / scale;
        checked += 1;
        if !(error < 2e-3) {
            bad += 1;
            if bad <= 12 {
                let worst = got.iter().zip(&want).enumerate().fold((0, 0f32), |worst, (i, (a, b))| if (a - b).abs() > worst.1 { (i, (a - b).abs()) } else { worst }).0;
                println!("op {index} {name}: relative error {error:.3e} (largest value {scale:.3e}); element {worst}: GPU {} CPU {}", got[worst], want[worst]);
            }
        }
    }
    println!("{checked} operations checked, {bad} differ");
    if bad > 0 {
        bail!("some operations differ");
    }
    Ok(())
}
