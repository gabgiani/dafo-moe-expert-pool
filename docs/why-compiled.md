---
title: Why a compiler
---

# Why compile a model instead of interpreting it

## What an interpreter does on every token
A typical engine walks the model for each token: for every operation the host code picks a kernel, sets its arguments, allocates temporaries and launches it. That is thousands of small decisions and launches per token (a Gemma 4 token is 933 operations),
repeated identically for every token, in code that has to be general enough for every model.

## What compiling does once instead
The compiler reads the GGUF and builds the graph of one token. Because every shape is known, it can decide everything ahead of time and hand the runtime a **fixed plan of GPU dispatches**:

* **Memory is planned, not allocated.** Activations are laid out by liveness: values that are never alive together share one buffer, so a whole token needs a few megabytes of temporaries instead of one buffer per operation.
* **Dispatches overlap where they can.** Independent dispatches run at the same time, with a barrier only where one reads what another wrote. What each kernel reads and writes is taken from its own signature, so the schedule stays right when a kernel changes.
* **One command buffer per token.** The host does no per-operation work while generating.
* **A plan per situation.** One plan for generating (one token, matrix-vector products) and others for reading prompts (128 or 16 tokens at a time, matrix-matrix products that read each weight once for the whole batch) and for images.
* **Weights are not copied.** On a Mac the model file is memory-mapped and wrapped as GPU buffers; a 7.4 GiB model is ready in 0.1 s.
* **Costs are known before running.** `dafo-compile inspect` says how many bytes a token reads (3.21 GiB for the 48-expert Gemma 4 26B-A4B), so it also says the ceiling: generation is memory-bound, about 29 tokens/s at 100 GB/s.
* **Correctness is checkable.** Every operation of a token is compared against a CPU computation made from the values the GPU read (`check_ops`), and batched plans against single-token ones. That is how a second backend (CUDA) was brought up against the first (Metal): 928 of 928 operations matched on the first full run.
* **One plan, several GPUs.** The plan names dispatches over buffers; the kernels for Metal (Apple) and CUDA (NVIDIA) take the same arguments in the same order. The expert pool, the vision encoder and the server do not know which one runs.

## What this buys you, honestly
**Speed is on par with the best interpreters, not above them.** On an Apple M4 with 16 GB, the 48-expert Gemma 4 26B-A4B generates at 25.6 tokens/s here, 24.3 in Candle and 25–26.8 in llama.cpp: they all sit near the bandwidth ceiling, which no scheduling can beat.
Reading a prompt is where llama.cpp is ahead (312–373 tokens/s against 168–205 here). Compiling does not make a memory-bound model faster. What it makes possible is something else:

1. **Plans that do not change while the model's content does.** Because the dispatch plan only names buffers and tables, the *experts in memory* can be swapped between two tokens without recompiling or rebuilding anything. That is the whole basis of the
   [expert pool](expert-pool.md): the only engine we know that serves the complete 128-expert Gemma 4 26B-A4B on a 16 GB Mac, with every expert still reachable.
2. **One small, auditable runtime with the features included.** A small Rust crate with three dependencies (`anyhow`, `memmap2`, `objc2-metal`; `libloading` on Windows/Linux), no llama.cpp, no Candle: a server, a chat, tools, a vision encoder, a tokenizer built from the GGUF.
3. **Predictable memory.** Weights, caches and temporaries are sized from the graph, so a host can choose the context from the memory it has instead of finding out at the first long prompt.
4. **A portable shape.** Adding a model family is a frontend (the graph of its token); adding a GPU language is a set of kernels. The scheduler, the pool and the server are shared.

## Where it loses
Prompt reading (1.5–2× slower than llama.cpp's tuned kernels), one model family, and the CUDA kernels are a first version (no tensor cores yet). [Measurements](RESULTS.md) and [Limits](limits.md) have the numbers, and
[what was tried](what-was-tried.md) lists the optimizations that did not work.
