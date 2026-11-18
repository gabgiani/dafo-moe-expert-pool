---
title: The Native Vision Encoder
---

# The Native Vision Encoder

_Reading images with Gemma 4's projector, on Metal and CUDA, with no framework_

Gemma 4 reads images through a projector file (`mmproj-*.gguf`, projector type `gemma4v`): a 27-layer vision transformer (hidden size 1152, 16 heads of 72,
F16 weights, 1.19 GiB) that turns an image into *soft tokens*, vectors the language model reads in place of text tokens. `src/vision.rs` is that
encoder on Metal, **without any framework**: the file is mapped and wrapped as one Metal buffer and about twenty dispatches per layer run it.
It is built with `--features vision` (one extra dependency, the `image` crate, to decode JPEG/PNG/WebP and resample).

## What it computes

1. **Preprocess (CPU).** Decode; scale to a size whose sides are multiples of 48 px (one soft token covers 3×3 patches of 16 px) with between 70 and
   `max_tokens` (default 560) cells, keeping the aspect ratio; Catmull-Rom resampling; values in [0, 1].
2. **Patch embedding.** Each 16×16×3 patch, scaled to [−1, 1], times the patch matrix, plus two learned position embeddings (one per axis, tables of 10 240 entries).
3. **27 layers:** RMS norm → q, k, v → RMS norm per head (v without weight) → 2D rotary embedding (half of each head rotated with the patch's column, half with its row, base 100)
   → full bidirectional attention with scale 1 → o → post-norm → residual; RMS norm → GELU(gate) · up → down → post-norm → residual.
4. **Pooling.** 3×3 patches averaged into one cell, × √hidden, the model's standardization `(x − bias) · scale`, an RMS norm without weight, and the projection to the
   language model's width (2816).

## The kernels (`src/backend_metal/vision.metal`)

* **Matrix products:** the compiler's `kernel_mul_mm` (simdgroup matrices, from llama.cpp's tile code) instantiated for F16 weights (`mm_f16`). The matrix kernel reads its input in
  steps of 32 columns; the one width that is not a multiple of 32 (the 4304 of the feed-forward) gets a zero pad.
* **Attention:** a flash-attention loop on 8×8 simdgroup matrices: one SIMD group takes 8 queries of one head, walks the keys in blocks of 8, keeps an online softmax
  (running max and sum) and rescales its accumulator with a diagonal matrix product. Queries, keys and values are half; scores and accumulators are float. A full image
  is about 4 700 patches, so attention is about a third of the time.
* Small kernels: patchify, position add, RMS norm (one SIMD group per group of values, also used per head), 2D rope, GELU·up, pooling.

## Checked against a reference

The Swarm's encoder (Candle, F32) and this one were run on the same 1280×1000 image: **520 soft tokens of 2816 values, cosine similarity 0.999999 on average and
0.99995 at worst per token**, largest single difference 0.12 on values of standard deviation 1.65 (`native_vision_matches_the_candle_encoder`, which needs the
Candle encoder and so lives in the Swarm's tree). On a Mac with an M4: 4.1–4.9 s for that image (about 1.6 s of it attention, the rest matrix products at roughly 2.6 TFLOPS);
it is not tuned yet.

## Using it

```
dafo-serve MODEL.gguf --projector mmproj.gguf        # an mmproj-*.gguf beside the model is found by itself
dafo-compile compile MODEL.gguf --out MODEL.dafo --projector mmproj.gguf    # the folder holds the projector and the plan that reads images
```

An image goes in an OpenAI message as `{"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}` (`data:` URLs only: nothing is fetched from the web); the chat page
attaches images. In the text each image is `<|image><|image|><image|>`; the engine replaces the middle token by the soft tokens, reads them in one run (every soft token of an image
sees the others in the sliding layers) and keeps the embeddings of the last 8 images, so a conversation that shows the same image again does not encode it again. A test image
of an invoice was read correctly (invoice number, customer, total) with the full model through the expert pool and with a compiled folder.

Limits: projectors that clip activations (`input_min`/`output_max` tensors) are refused; attention is written for heads of 72 values; one image runs at a time.
