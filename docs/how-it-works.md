---
title: How the compiler works
---

# How the compiler works

Most engines *interpret* a model: for each token they walk the layers and, for every operation, the host code chooses a kernel, sets its
arguments, allocates temporaries and launches it. This project does that work **once, ahead of time**, and turns a token into a fixed plan of
GPU dispatches.

```
 model.gguf ──► frontend ──► graph ──► passes ──► backend ──► plan ──► runtime
 (weights +     (one module   (15        (buffer     (Metal      (list of   (runs the plan;
  metadata)      per family)   generic    reuse,      kernels     GPU        knows no model,
                               ops)       overlap,    in MSL)     dispatches knows no tokenizer)
                                          batching)               + buffers)
```

* **Frontend** (`src/frontends/`) reads the GGUF metadata and tensor table and emits a graph of generic operations (`Linear`, `RmsNorm`,
  `Attention`, `Route`, `ExpertsLinear`, …). Gemma 4 is the only family so far.
* **Graph** (`src/ir.rs`): 15 operations over values of known size. Because every shape is known, the compiler can say before running anything
  how many bytes a token reads (`dafo-compile inspect`) and what the caches cost.
* **Passes**: activation buffers are laid out by liveness (values never alive together share one buffer); independent dispatches are grouped to
  run at the same time with a memory barrier only where one reads what another wrote (the access of each kernel is read from the signatures of
  its Metal source); several plans are built for different batch sizes (1 token to generate, 128 and 16 to read a prompt).
* **Backend** (`src/backend_metal.rs`, `src/backend_metal/*.metal`): turns the graph into dispatches. The weights are **not copied**: the model
  file is memory-mapped and wrapped as Metal buffers.
* **Runtime** (`src/runtime.rs`): executes a plan. One token is one command buffer. It holds the key/value caches of several conversations.

Generation is memory-bound: a token of this model reads 3.21 GiB (dense layers 1.63, experts 0.84, output head 0.73, …), so at the chip's
bandwidth the ceiling is about 29 tokens/s and every engine sits near it. A compiled plan removes the per-token host overhead; it cannot beat
the bandwidth.

## A compiled model is a folder

`dafo-compile compile MODEL.gguf --out MODEL.dafo` leaves `manifest.json`, the weights (linked, not copied), `tokenizer.json` and
`kernels.metal` in a folder that belongs to no application. The dispatch plans are rebuilt in milliseconds when the folder is opened, so the
folder cannot go stale against the Metal driver. `dafo-serve MODEL.dafo` serves it over an OpenAI-compatible API (tools, streaming, bearer
token) with a chat page. See [Run it](run.md).
