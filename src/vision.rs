//! A native vision encoder for Gemma 4 on Metal: the projector file (`mmproj-*.gguf`, projector type `gemma4v`) turns an image into the
//! soft-token embeddings the language model reads (see `Runtime::run_image`). No framework: the file is mapped and wrapped as one Metal
//! buffer (F16 weights, not copied) and about twenty dispatches per layer run the patch-based ViT.
//!
//! What the encoder does, in the order it does it (this follows the reference tower of the model):
//!
//! 1. **Preprocess** (CPU): decode, scale to a size that is a multiple of 48 pixels (3 patches of 16) with between 70 and `max_tokens`
//!    cells of 48×48, keeping the aspect ratio; Catmull-Rom resampling; values in [0, 1].
//! 2. **Patch embedding**: every 16×16×3 patch, scaled to [-1, 1], times the patch matrix, plus two learned position embeddings (column, row).
//! 3. **27 layers** of: RMS norm → q, k, v → per-head RMS norms (v without weight) → 2D rotary embedding (half of each head with the column,
//!    half with the row, base 100) → full bidirectional attention (scale 1) → o → post-norm → residual; then RMS norm → GELU(gate) · up →
//!    down → post-norm → residual.
//! 4. **Pooling**: 3×3 cells averaged, times √hidden, then the model's standardization; an RMS norm without weight; the projection to the
//!    language model's width.
//!
//! The matrix products are the compiler's `kernel_mul_mm` instantiated for F16 weights; the attention is a flash-attention loop on
//! 8×8 simdgroup matrices (`vision.metal`).

use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result, bail, ensure};
use image::imageops::FilterType;

use crate::gguf::{Gguf, Scalar};
use crate::metal::{Buffer, Gpu, Kernel};

const MATMUL: &str = include_str!("backend_metal/matmul.metal");
const VISION: &str = include_str!("backend_metal/vision.metal");
/// Fewest cells an image is scaled up to (the reference preprocessor's minimum).
const MIN_SOFT_TOKENS: usize = 70;
const POOL: usize = 3;
const ROPE_THETA: f32 = 100.0;
const PAGE: usize = 16384;

/// The encoder of one projector file.
pub struct VisionEncoder {
    gpu: Gpu,
    gguf: Arc<Gguf>,
    weights: Buffer,
    kernels: std::collections::HashMap<&'static str, Kernel>,
    hidden: usize,
    intermediate: usize,
    layers: usize,
    heads: usize,
    head_dim: usize,
    patch: usize,
    eps: f32,
    output: usize,
    max_tokens: usize,
    standardize: bool,
}

/// What one image became.
pub struct EncodedImage {
    /// `tokens` × `width` values, row by row.
    pub embeddings: Vec<f32>,
    pub tokens: usize,
    pub width: usize,
}

fn unsigned(gguf: &Gguf, key: &str) -> Result<usize> {
    match gguf.metadata(key) {
        Some(Scalar::Unsigned(value)) => Ok(*value as usize),
        Some(Scalar::Signed(value)) => Ok(*value as usize),
        other => bail!("the projector lacks `{key}` ({other:?})"),
    }
}

impl VisionEncoder {
    /// Opens the projector file. `max_tokens` is the most soft tokens one image may use (70, 140, 280, 560 or 1120).
    pub fn open(path: &Path, max_tokens: usize) -> Result<Self> {
        ensure!([70, 140, 280, 560, 1120].contains(&max_tokens), "the visual token budget is 70, 140, 280, 560 or 1120");
        let gguf = Arc::new(Gguf::open(path).with_context(|| format!("cannot open the projector {}", path.display()))?);
        match gguf.metadata("clip.vision.projector_type") {
            Some(Scalar::Text(kind)) if kind == "gemma4v" => {}
            other => bail!("this encoder reads the gemma4v projector, not {other:?}"),
        }
        let hidden = unsigned(&gguf, "clip.vision.embedding_length")?;
        let heads = unsigned(&gguf, "clip.vision.attention.head_count")?;
        let eps = match gguf.metadata("clip.vision.attention.layer_norm_epsilon") {
            Some(Scalar::Float(value)) => *value as f32,
            other => bail!("the projector lacks the norm epsilon ({other:?})"),
        };
        let head_dim = hidden / heads;
        ensure!(head_dim == 72 && heads * head_dim == hidden, "the attention kernel is written for heads of 72 values (this projector has {head_dim})");
        let layers = unsigned(&gguf, "clip.vision.block_count")?;
        for layer in 0..layers {
            for name in ["attn_q", "attn_k", "attn_v", "attn_out", "ffn_gate", "ffn_up", "ffn_down"] {
                ensure!(
                    gguf.tensor(&format!("v.blk.{layer}.{name}.input_min")).is_none(),
                    "this projector clips activations (v.blk.{layer}.{name}.input_min): not supported yet"
                );
            }
        }
        let output = gguf.tensor("mm.input_projection.weight").context("the projector has no mm.input_projection.weight")?.dims[1] as usize;
        let gpu = Gpu::new()?;
        let source = format!("#include <metal_stdlib>\nusing namespace metal;\n{MATMUL}\n{VISION}");
        let library = gpu.library(&source)?;
        let mut kernels = std::collections::HashMap::new();
        for name in ["mm_f16", "v_patchify", "v_add_position", "v_rms_norm", "v_add", "v_gelu_mul", "v_rope", "v_to_half", "v_attention", "v_pool"] {
            kernels.insert(name, gpu.kernel(&library, name)?);
        }
        let map = gguf.mapped();
        let length = map.len().div_ceil(PAGE) * PAGE;
        let weights = unsafe { gpu.buffer_no_copy(map.as_ptr(), length)? };
        Ok(Self {
            gpu,
            weights,
            kernels,
            hidden,
            intermediate: unsigned(&gguf, "clip.vision.feed_forward_length")?,
            layers,
            heads,
            head_dim,
            patch: unsigned(&gguf, "clip.vision.patch_size")?,
            eps,
            output,
            max_tokens,
            standardize: gguf.tensor("v.std_bias").is_some(),
            gguf,
        })
    }

    /// Width of the embeddings (the language model's hidden size).
    pub fn width(&self) -> usize {
        self.output
    }

    /// Side, in pixels, of the square each soft token covers, and the most tokens an image may use.
    pub fn pixels_per_token(&self) -> usize {
        self.patch * POOL
    }

    pub fn max_tokens(&self) -> usize {
        self.max_tokens
    }

    /// Reads an image (JPEG, PNG or WebP) and returns its soft tokens.
    pub fn encode(&self, bytes: &[u8]) -> Result<EncodedImage> {
        let image = image::load_from_memory(bytes).context("cannot decode the image; JPEG, PNG and WebP are supported")?.to_rgb8();
        let (width, height) = image.dimensions();
        ensure!(width > 0 && height > 0, "the image has no pixels");
        let (target_height, target_width) = target_size(height as usize, width as usize, self.pixels_per_token(), MIN_SOFT_TOKENS, self.max_tokens)?;
        let resized = image::imageops::resize(&image, target_width as u32, target_height as u32, FilterType::CatmullRom);
        let mut pixels = vec![0f32; 3 * target_height * target_width];
        for y in 0..target_height {
            for x in 0..target_width {
                let pixel = resized.get_pixel(x as u32, y as u32);
                for channel in 0..3 {
                    pixels[channel * target_height * target_width + y * target_width + x] = pixel.0[channel] as f32 / 255.0;
                }
            }
        }
        self.run(&pixels, target_height, target_width)
    }

    fn tensor_offset(&self, name: &str) -> Result<usize> {
        Ok(self.gguf.tensor(name).with_context(|| format!("the projector lacks {name}"))?.offset)
    }

    fn run(&self, pixels: &[f32], height: usize, width: usize) -> Result<EncodedImage> {
        let (hidden, heads, head_dim, patch) = (self.hidden, self.heads, self.head_dim, self.patch);
        let (grid_w, grid_h) = (width / patch, height / patch);
        let patches = grid_w * grid_h;
        ensure!(grid_w % POOL == 0 && grid_h % POOL == 0, "the patch grid {grid_w}x{grid_h} does not divide into {POOL}x{POOL} cells");
        let padded = patches.div_ceil(8) * 8;
        let cells = patches / (POOL * POOL);
        let patch_values = 3 * patch * patch;
        // The matrix kernel reads the input in steps of 32 columns: the intermediate activations get a zero pad.
        let wide = self.intermediate.div_ceil(32) * 32;
        let gpu = &self.gpu;
        let floats = |n: usize| gpu.buffer(n * 4);
        let pixel_buffer = gpu.buffer_from(pixels)?;
        let patch_buffer = floats(padded * patch_values)?;
        let x = floats(padded * hidden)?;
        let normed = floats(padded * hidden)?;
        let (qf, kf, vf) = (floats(padded * hidden)?, floats(padded * hidden)?, floats(padded * hidden)?);
        let (qh, kh, vh) = (gpu.buffer(padded * hidden * 2)?, gpu.buffer(padded * hidden * 2)?, gpu.buffer(padded * hidden * 2)?);
        let attention = floats(padded * hidden)?;
        let temporary = floats(padded * hidden)?;
        let gate = floats(padded * self.intermediate)?;
        let up = floats(padded * self.intermediate)?;
        let mlp = floats(padded * wide)?;
        let pooled = floats(cells * hidden)?;
        let result = floats(cells * self.output)?;

        let (hidden_u, patches_u, grid_u) = (hidden as u32, patches as u32, grid_w as u32);
        let eps = self.eps;
        let position_entries = self.gguf.tensor("v.position_embd.weight").context("no position table")?.dims[1] as u32;
        let weights = &self.weights;
        let kernels = &self.kernels;

        // One matrix product: `rows` x `k` F16 weights (at `weight` in the file) times `tokens` rows of `input`, rows `input_stride` apart.
        let matmul = |encoder: &crate::metal::Encoder, weight: usize, k: usize, rows: usize, input: &Buffer, input_stride: usize, output: &Buffer, tokens: usize| {
            let row_bytes = (k * 2) as u64;
            let k_steps = k.div_ceil(32) * 32;
            encoder.kernel(&kernels["mm_f16"]);
            encoder.buffer(0, weights, weight);
            encoder.buffer(1, input, 0);
            encoder.buffer(2, output, 0);
            encoder.value(3, &(k_steps as i64));
            encoder.value(4, &1i64);
            encoder.value(5, &row_bytes);
            encoder.value(6, &(row_bytes * rows as u64));
            encoder.value(7, &(row_bytes * rows as u64));
            encoder.value(8, &1i64);
            encoder.value(9, &4u64);
            encoder.value(10, &((input_stride * 4) as u64));
            encoder.value(11, &((input_stride * 4 * tokens) as u64));
            encoder.value(12, &((input_stride * 4 * tokens) as u64));
            encoder.value(13, &(rows as i64));
            encoder.value(14, &(tokens as i64));
            encoder.value(15, &1u32);
            encoder.value(16, &1u32);
            encoder.shared_memory(0, 8192);
            encoder.dispatch([tokens.div_ceil(32), rows.div_ceil(64), 1], [128, 1, 1]);
        };
        let rms = |encoder: &crate::metal::Encoder, input: &Buffer, scale: Option<usize>, output: &Buffer, width: usize, groups: usize| {
            encoder.kernel(&kernels["v_rms_norm"]);
            encoder.buffer(0, input, 0);
            encoder.buffer(1, weights, scale.unwrap_or(0));
            encoder.buffer(2, output, 0);
            encoder.value(3, &(width as u32));
            encoder.value(4, &eps);
            encoder.value(5, &(scale.is_some() as u32));
            encoder.dispatch([groups, 1, 1], [32, 1, 1]);
        };
        let add = |encoder: &crate::metal::Encoder, target: &Buffer, other: &Buffer, n: usize| {
            encoder.kernel(&kernels["v_add"]);
            encoder.buffer(0, target, 0);
            encoder.buffer(1, other, 0);
            encoder.value(2, &(n as u32));
            encoder.dispatch([n.div_ceil(256), 1, 1], [256, 1, 1]);
        };
        let rope = |encoder: &crate::metal::Encoder, input: &Buffer, output: &Buffer| {
            encoder.kernel(&kernels["v_rope"]);
            encoder.buffer(0, input, 0);
            encoder.buffer(1, output, 0);
            encoder.value(2, &(heads as u32));
            encoder.value(3, &(head_dim as u32));
            encoder.value(4, &grid_u);
            encoder.value(5, &ROPE_THETA);
            encoder.dispatch([head_dim / 2, heads, patches], [head_dim / 2, 1, 1]);
        };

        let mut offsets = Vec::new();
        for layer in 0..self.layers {
            let mut layer_offsets = std::collections::HashMap::new();
            for name in ["attn_q", "attn_k", "attn_v", "attn_out", "ffn_gate", "ffn_up", "ffn_down"] {
                layer_offsets.insert(name, self.tensor_offset(&format!("v.blk.{layer}.{name}.weight"))?);
            }
            for name in ["ln1", "ln2", "attn_q_norm", "attn_k_norm", "attn_post_norm", "ffn_post_norm"] {
                layer_offsets.insert(name, self.tensor_offset(&format!("v.blk.{layer}.{name}.weight"))?);
            }
            offsets.push(layer_offsets);
        }
        let patch_weight = self.tensor_offset("v.patch_embd.weight")?;
        let position_table = self.tensor_offset("v.position_embd.weight")?;
        let projection = self.tensor_offset("mm.input_projection.weight")?;
        let (std_bias, std_scale) = if self.standardize { (self.tensor_offset("v.std_bias")?, self.tensor_offset("v.std_scale")?) } else { (0, 0) };

        self.gpu.run(|encoder| {
            // Patches and their embedding.
            encoder.kernel(&kernels["v_patchify"]);
            encoder.buffer(0, &pixel_buffer, 0);
            encoder.buffer(1, &patch_buffer, 0);
            encoder.value(2, &(width as u32));
            encoder.value(3, &(height as u32));
            encoder.value(4, &(patch as u32));
            encoder.value(5, &grid_u);
            encoder.dispatch([patch_values.div_ceil(64), patches, 1], [64, 1, 1]);
            matmul(encoder, patch_weight, patch_values, hidden, &patch_buffer, patch_values, &x, patches);
            encoder.kernel(&kernels["v_add_position"]);
            encoder.buffer(0, &x, 0);
            encoder.buffer(1, weights, position_table);
            encoder.value(2, &hidden_u);
            encoder.value(3, &grid_u);
            encoder.value(4, &position_entries);
            encoder.dispatch([hidden.div_ceil(64), patches, 1], [64, 1, 1]);

            for layer in &offsets {
                rms(encoder, &x, Some(layer["ln1"]), &normed, hidden, patches);
                matmul(encoder, layer["attn_q"], hidden, hidden, &normed, hidden, &qf, patches);
                matmul(encoder, layer["attn_k"], hidden, hidden, &normed, hidden, &kf, patches);
                matmul(encoder, layer["attn_v"], hidden, hidden, &normed, hidden, &vf, patches);
                rms(encoder, &qf, Some(layer["attn_q_norm"]), &qf, head_dim, patches * heads);
                rms(encoder, &kf, Some(layer["attn_k_norm"]), &kf, head_dim, patches * heads);
                rms(encoder, &vf, None, &vf, head_dim, patches * heads);
                rope(encoder, &qf, &qh);
                rope(encoder, &kf, &kh);
                encoder.kernel(&kernels["v_to_half"]);
                encoder.buffer(0, &vf, 0);
                encoder.buffer(1, &vh, 0);
                encoder.value(2, &((patches * hidden) as u32));
                encoder.dispatch([(patches * hidden).div_ceil(256), 1, 1], [256, 1, 1]);
                encoder.kernel(&kernels["v_attention"]);
                encoder.buffer(0, &qh, 0);
                encoder.buffer(1, &kh, 0);
                encoder.buffer(2, &vh, 0);
                encoder.buffer(3, &attention, 0);
                encoder.value(4, &patches_u);
                encoder.value(5, &(heads as u32));
                encoder.dispatch([padded / 8, heads, 1], [32, 1, 1]);
                matmul(encoder, layer["attn_out"], hidden, hidden, &attention, hidden, &temporary, patches);
                rms(encoder, &temporary, Some(layer["attn_post_norm"]), &temporary, hidden, patches);
                add(encoder, &x, &temporary, patches * hidden);

                rms(encoder, &x, Some(layer["ln2"]), &normed, hidden, patches);
                matmul(encoder, layer["ffn_gate"], hidden, self.intermediate, &normed, hidden, &gate, patches);
                matmul(encoder, layer["ffn_up"], hidden, self.intermediate, &normed, hidden, &up, patches);
                encoder.kernel(&kernels["v_gelu_mul"]);
                encoder.buffer(0, &gate, 0);
                encoder.buffer(1, &up, 0);
                encoder.buffer(2, &mlp, 0);
                encoder.value(3, &(self.intermediate as u32));
                encoder.value(4, &(self.intermediate as u32));
                encoder.value(5, &(wide as u32));
                encoder.dispatch([self.intermediate.div_ceil(64), patches, 1], [64, 1, 1]);
                matmul(encoder, layer["ffn_down"], self.intermediate, hidden, &mlp, wide, &temporary, patches);
                rms(encoder, &temporary, Some(layer["ffn_post_norm"]), &temporary, hidden, patches);
                add(encoder, &x, &temporary, patches * hidden);
            }

            // Pooling, standardization, the norm without weight and the projection to the language model.
            encoder.kernel(&kernels["v_pool"]);
            encoder.buffer(0, &x, 0);
            encoder.buffer(1, weights, std_bias);
            encoder.buffer(2, weights, std_scale);
            encoder.buffer(3, &pooled, 0);
            encoder.value(4, &hidden_u);
            encoder.value(5, &grid_u);
            encoder.value(6, &(POOL as u32));
            encoder.value(7, &(self.standardize as u32));
            encoder.dispatch([hidden.div_ceil(64), cells, 1], [64, 1, 1]);
            rms(encoder, &pooled, None, &pooled, hidden, cells);
            matmul(encoder, projection, hidden, self.output, &pooled, hidden, &result, cells);
        })?;
        let embeddings = result.as_slice::<f32>()[..cells * self.output].to_vec();
        Ok(EncodedImage { embeddings, tokens: cells, width: self.output })
    }
}

/// The size an image is scaled to: both sides multiples of `align` (the 48 pixels one soft token covers), the area between `min_tokens` and
/// `max_tokens` cells of `align`², the aspect ratio kept.
pub fn target_size(height: usize, width: usize, align: usize, min_tokens: usize, max_tokens: usize) -> Result<(usize, usize)> {
    ensure!(min_tokens <= max_tokens, "the visual token budget is inverted");
    let a = align as f64;
    let round_by = |value: f64| ((value / a).round() as usize) * align;
    let ceil_by = |value: f64| ((value / a).ceil() as usize) * align;
    let floor_by = |value: f64| ((value / a).floor() as usize) * align;
    let area = align * align;
    let (min_pixels, max_pixels) = (min_tokens * area, max_tokens * area);
    let source_pixels = (height * width) as f64;
    let mut target_height = round_by(height as f64).max(align);
    let mut target_width = round_by(width as f64).max(align);
    if target_height * target_width > max_pixels {
        let beta = (source_pixels / max_pixels as f64).sqrt();
        target_height = floor_by(height as f64 / beta).max(align);
        target_width = floor_by(width as f64 / beta).max(align);
    } else if target_height * target_width < min_pixels {
        let beta = (min_pixels as f64 / source_pixels).sqrt();
        target_height = ceil_by(height as f64 * beta);
        target_width = ceil_by(width as f64 * beta);
    }
    ensure!(target_height * target_width <= max_pixels, "an image of {width}x{height} cannot fit the visual token budget");
    Ok((target_height, target_width))
}

#[cfg(test)]
mod tests {
    use super::target_size;

    #[test]
    fn sizes_are_multiples_of_a_token_and_inside_the_budget() {
        // 1280x1000 at 560 tokens (48 px cells): scaled down to fit.
        let (h, w) = target_size(1000, 1280, 48, 70, 560).unwrap();
        assert!(h % 48 == 0 && w % 48 == 0 && h * w <= 560 * 48 * 48);
        // A tiny image is scaled up to at least 70 cells.
        let (h, w) = target_size(2, 2, 48, 70, 560).unwrap();
        assert!(h * w >= 70 * 48 * 48 && h % 48 == 0 && w % 48 == 0);
        // A small image already inside the budget is only snapped to the grid.
        assert_eq!(target_size(480, 480, 48, 70, 560).unwrap(), (480, 480));
    }
}
