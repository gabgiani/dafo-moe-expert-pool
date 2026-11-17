---
title: The expert pool
---

# The expert pool: how all 128 experts stay available

## The problem

Gemma 4 26B-A4B has 30 layers, each with 128 experts; the router picks 8 per token. The experts are 13.4 of the 15.8 GiB of the model. On a
16 GB Mac, Metal lets a program use about 11 GiB, so the file does not fit. The usual answer is to **cut** the model to the experts a workload
uses (here: 48 per layer, 7.4 GiB) and throw the rest away. That works, but the cut is fixed: a different kind of conversation finds experts missing.

## The observation that makes a cut equal to a mask

The router of Gemma 4 computes probabilities over **all** experts, takes the 8 largest and **renormalizes the weights among those 8**. So if the
experts that a full model would choose are all present in a cut, that layer's output is *identical* to the full model's; when one is missing, the
next best among the present ones takes its place. Removing an expert is therefore the same as setting its router logit to −∞ before the top-8.
This was checked on llama.cpp (a mask over the full model and a cut file gave the same log-probabilities), and it holds in the compiled engine
(next section).

## The pool

`backend_metal::use_expert_pool(graph, N)` and `Runtime::with_expert_pool(.., Some(N))` change two things:

1. **The expert tensors are not mapped.** For each layer the runtime allocates one buffer of `N` experts per expert tensor
   (`ffn_gate_up_exps`, `ffn_down_exps`) and fills its *slots* by copying each expert's contiguous slice of bytes from the model file.
   A slot is 3.72 MB; `N = 48` over 30 layers is 7.41 GiB, the same as the cut.
2. **The router writes slots, not expert numbers.** The `route` kernel scores all 128 experts, reads a per-layer table in the state buffer
   (one word per expert: `0xFFFFFFFF` = not in the pool, otherwise slot + 1), gives −∞ to the blocked ones, takes the top 8, renormalizes and writes
   the **slot** of each chosen expert. The expert products (`mv_id_*`, `mm_id_*`) read their weights by slot, so they do not know the model has
   128 experts.

`Runtime::set_allowed_experts(layer, experts, total)` changes what the pool holds **between two tokens**: experts already resident stay in their
slot, the missing ones are copied into free slots. The dispatch plans do not change; only the buffers and the table do. Changing 32 experts
(119 MB) took 0.23 s on the test Mac.

### Why not just map the whole file and mask?

It was tried first, and is simpler: map the 15.8 GiB file, block experts in the router, and `madvise(DONTNEED)` the blocked ones. It fails on 16 GB
with `kIOGPUCommandBufferCallbackErrorOutOfMemory`: Metal counts **every buffer bound to a command buffer** against the device's working set,
whether or not its pages are ever touched. (It would work on a Mac with enough memory, such as a 96 GB machine.) The pool keeps only the
experts in buffers, so the working set is the pool.

### Equality with the cut

A full 128-expert file, a pool of 48 and the selection that a 48-expert cut file kept (`examples/check_mask.rs`) give **identical logits** to that
cut file: the same largest logit and the same sum of the logits over the first three steps, and the same generated token ids. At any moment the
model *is* a 48-expert model; what changes is which 48.

## What this is, and what it is not

* It is **not** the full 128-expert model at full quality: at any moment the router can only use the experts in the pool. The claim is that
  **every expert is reachable** — none is discarded for good — and that the pool can be exchanged on the fly.
* How good the answers are depends on how well the 48 resident experts match the conversation. [The adaptive pool](adaptive.md) is the part that
  tries to make them match.
