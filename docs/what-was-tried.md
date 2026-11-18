# Measured results

Everything here was measured on one machine: **Apple M4, 16 GB**, macOS, with **Gemma 4 26B-A4B cut to the 48 most used of its 128
experts per layer, Q4_K_M** (7.39 GiB, 30 layers: 25 sliding-window and 5 global). Numbers come from `dafo-compile bench`, the
examples in this repository, and `llama-bench` (llama.cpp build 9960) on the same file, with the machine otherwise idle. Speeds move
by a few percent between runs and by much more when memory is under pressure (the model is memory-mapped: if the system evicts its
pages, the next request is slow).

## Generation

A token reads 3.21 GiB of weights (dense 1.63, experts 0.84, output head 0.73). At the ≈ 100 GB/s this chip reaches in practice
that is at most ≈ 29 tokens/s, and every engine is close to it.

| | tokens/s |
|---|---|
| this compiler, short context | 24.9 – 25.6 |
| Candle (same machine, same question) | 24.3 |
| llama.cpp (`llama-bench tg32`) | 26.8 |
| this compiler after a 4,446-token prompt | 19.7 (Candle 20.5) |

llama.cpp is ahead on generation by a few percent here; the difference is not in the kernels that read the weights.

## Reading a prompt

| | tokens/s |
|---|---|
| first version, one token at a time | 21 |
| batched (128 and 16 tokens per plan, matrix-matrix kernels) | 122 – 129 on a 4,446-token prompt; 168 – 205 on 512 – 1,400 tokens |
| Candle | 150 – 200 |
| llama.cpp (`llama-bench pp128 / pp256 / pp512`) | 312 / 355 / 373 |

Where a 128-token batch spends its time (`examples/profile_batch.rs`): expert matrix products with 4-bit weights 40 %, dense
8-bit products 27 %, expert products with 5-bit weights 21 %, attention 7 %, everything else 5 %. Time per token does **not** improve
with larger batches (128 → 256 tokens: same ms per token), while llama.cpp's does (312 → 355 → 373).

### What was tried to close the gap, and what happened

| Change | Result |
|---|---|
| Batching the prompt (the first big step) | 21 → 129 tok/s |
| Activation tile in half precision instead of float (as current llama.cpp) | −3 % kernel time, slightly larger divergence from token-by-token (KL 0.044 against 0.038): dropped |
| Output tile written by all four simdgroups instead of one | no change |
| Metal language version and fast-math flags fixed explicitly | no change |
| Timing with the weight decompression and reads removed (not a valid kernel, a measurement) | the 4-bit expert kernel goes 209 → 178 ms and the 5-bit one 108 → 87 ms: decompression is 15 – 20 % of the time; the rest is the matrix multiply itself |
| Overlapping independent dispatches (the scheduling pass, see README) | prompt reading +2.5 – 5 %, generation +3 %, results **bit-identical** with and without it |
| Bigger batches (256, 512) | no gain, more memory |

The remaining gap (1.5 – 2×) is inside the multiply loop of the expert kernels: with a 128-token batch each expert sees about 21 tokens
for a tile of 32 columns, and padded columns still cost. Candidates not yet tried: a 64-column tile for the experts, and choosing the
tile shape per operation by measuring on the device.

## Correctness

- Each operation of a token against a CPU reference (`examples/check_ops.rs`): 0 differ.
- A batch against the same tokens one at a time, layer by layer (`examples/check_batch.rs`): the next-token distributions differ by
  KL 0.04 and the same token wins, which is what summing in a different order gives.
- Perplexity of a 2,048-token text, second half only (positions 1,024 – 2,047, past the sliding window): **3,015.7** here against
  **3,017.9** with llama.cpp on the same file (0.07 % apart).
- Several conversations at the same time give the same text as each one alone; a conversation read from scratch and the same one
  read after another with the same first tokens give the same text (the cache is reused correctly).
- The scheduling pass changes nothing: 600 positions give identical choices and losses with and without it.

## Notes for hosts

These are not in the compiler, but they cost real time to find:

- **Penalise a repeated token once, not once per occurrence.** A repetition penalty applied for every occurrence in the window
  compounds (a digit seen five times is divided five times) and makes a model unable to copy numbers it has just read; it then
  notices the mismatch and loops. llama.cpp and Candle penalise each distinct token once.
- **Close an empty thought before the answer** for models that open one (`<|channel>thought\n<channel|>` in Gemma 4) if you do not
  want reasoning; and cut generations that repeat a block of tokens (`looping_output_end`-style) rather than waiting for the limit.
- **Size the context from memory.** The caches of the global-attention layers grow with the context (20 KiB per token for this
  model); the sliding-window layers cost a fixed 325 MiB per conversation.
