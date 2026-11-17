# dafo-compiler-llm

**A compiler for language models.** It reads a model file (GGUF), builds the graph of what one token computes, and turns it into a
fixed plan of GPU dispatches for the chip it runs on. Nothing in the plan is decided while the model runs.

The first backend is Metal (Apple GPUs); the first model is Gemma 4 26B-A4B, a mixture-of-experts model. The compiler is written in
Rust, has two small dependencies (`anyhow`, `memmap2`; `objc2-metal` on macOS) and does not depend on Candle, llama.cpp or any
other inference framework.

> Status: working and measured, early. One model family, one backend. See [Limits](#limits) and [docs/RESULTS.md](RESULTS.md),
> which include what did **not** work.

## What "compiled" means here

Most engines *interpret* a model: for every token they walk the layers, and for every operation the host code picks a kernel, sets
its arguments, allocates temporaries and launches it. This compiler does that work once, ahead of time:

```
 model.gguf ──► frontend ──► graph ──► passes ──► backend ──► plan ──► runtime
 (weights +     (one module   (15        (buffer     (Metal:     (list of   (runs the plan;
  metadata)      per model     generic    reuse,      kernels     GPU        knows no model,
                 family)       ops)       overlap,    written     dispatches knows no tokenizer)
                                          batching)   in MSL)     + buffers)
```

- **Frontend** (`src/frontends/`): one module per architecture. It reads the metadata and tensor table and emits a graph of
  generic operations (`Linear`, `RmsNorm`, `Attention`, `Route`, `ExpertsLinear`, …). Supporting a new model family means writing
  a frontend, not new kernels, when the model uses the same operations.
- **Graph** (`src/ir.rs`): 15 operations over values with known sizes. Because every shape is known, the compiler can say, before
  running anything, how many bytes a token reads (`dafo-compile inspect`) and how much memory the caches need.
- **Passes**: the activation buffers of the whole token are laid out by liveness (values that are never alive together share one
  buffer, so a token needs a few megabytes of temporaries, not one buffer per operation); independent dispatches are grouped to run
  at the same time with a memory barrier only where one reads what another wrote (the access of each kernel is read from the
  signatures in its Metal source, so it stays right when a kernel changes); and several plans are built for different batch sizes
  (1 token for generation, 128 and 16 for reading a prompt, a large one for images).
- **Backend** (`src/backend_metal.rs`, `src/backend_metal/*.metal`): turns the graph into dispatches and the kernels they use.
  Weights are not copied: the model file is memory-mapped and wrapped as Metal buffers.
- **Runtime** (`src/runtime.rs`): executes a plan. One token is one command buffer. It holds the key/value caches of several
  conversations ("slots") and knows nothing about tokenizers or chat formats.

## What is compiled, and what a host adds

The compiler and runtime do the **arithmetic and the memory plan** of a model. They do not read text. A *host* program supplies:

| Host's job | Where it lives for DAFO Swarm |
|---|---|
| Tokenizer and chat template | DAFO Swarm's node (not in this repository) |
| Sampling (temperature, top-k, top-p, repetition penalty, JSON-schema masks) | DAFO Swarm's node |
| Scheduling of several requests, prefix reuse of cached conversations | DAFO Swarm's node |
| An HTTP API (OpenAI-compatible) | DAFO Swarm |

This repository ships a host for Gemma 4 (`src/host/`, built with `--features host`): the tokenizer, the chat template with tools,
the sampler and an engine thread. On top of it, **`dafo-serve`** is a standalone OpenAI-compatible server with a bearer token and a
chat page (see [Serve a compiled model](#serve-a-compiled-model)). [DAFO Swarm](https://github.com/gabgiani/dafo-swarm-downloads)
embeds the crate with a host of its own. You can also link the crate as a library and write your own (tokenize, `prefill`, `step`, sample).

A compiled model is **a folder**: `dafo-compile compile MODEL.gguf --out MODEL.dafo` leaves the manifest, the weights (linked, not
copied), the tokenizer and the Metal kernels in one place that does not belong to any application. The dispatch plans themselves are not
stored: they are rebuilt from the folder in milliseconds when it is opened.

## Use

Requirements: a Mac with Apple silicon, Rust 1.85+. `inspect` also runs elsewhere.

```
cargo build --release
./target/release/dafo-compile inspect MODEL.gguf                 # what the model is, what a token reads, what the caches cost
./target/release/dafo-compile compile MODEL.gguf --context 8192  # build the plans for this machine
./target/release/dafo-compile bench   MODEL.gguf                 # generation and prompt-reading speed here
```

To leave the compiled model in a folder and serve it, build with the host (`--features host`):

```
cargo build --release --features host
./target/release/dafo-compile compile MODEL.gguf --context 16384 --out MODEL.dafo
./target/release/dafo-serve MODEL.dafo --port 8080 --token MY-SECRET
```

`inspect` on the 48-expert cut of Gemma 4 26B-A4B (Q4_K_M, 7.39 GiB):

```
  the graph of one token: 933 operations, 933 values
  attention layers: 25 sliding-window, 5 global
  a token reads 3.21 GiB (dense 1.63, experts 0.84, output head 0.73, ...)
  generation is memory-bound: at 100 GB/s that is at most 29.0 tokens/s
  keys and values of one conversation: 325 MiB fixed (sliding-window rings) + 20.0 KiB per token of context
```

As a library:

```rust
use std::sync::Arc;
use dafo_compiler_llm::{backend_metal, frontends, gguf::Gguf, runtime::Runtime};

let gguf = Arc::new(Gguf::open("model.gguf")?);
let graph = frontends::token_graph(&gguf)?;                         // the model as a graph
let plans = backend_metal::compile_plans(&graph, &gguf, 8192, false)?;   // compiled for this GPU, 8192 tokens of context
let mut runtime = Runtime::new(&gguf, &plans, 2)?;                  // two conversations kept in cache
runtime.select_slot(0);
let mut next = runtime.prefill(&prompt_ids, 0)?;                    // read the prompt (batched)
next = runtime.step(next, prompt_ids.len())?.0;                     // one token; `runtime.logits()` has the distribution to sample
```

The context is a compile-time parameter because the global-attention caches grow with it. `backend_metal::cache_footprint(&graph)`
returns what one conversation costs (a fixed part plus bytes per token of context), so a host can size the context from the memory
it has; DAFO Swarm does that.

## Serve a compiled model

`dafo-serve` takes a compiled folder (or a `.gguf`, compiling at start) and serves:

| Route | |
|---|---|
| `GET /` | a chat page: token field, system prompt, streaming, stop button, text-file attachments |
| `GET /health` | model name and context, no token needed |
| `GET /v1/models` | needs `Authorization: Bearer TOKEN` |
| `POST /v1/chat/completions` | OpenAI format: `messages`, `stream`, `tools` / `tool_calls` / `tool` messages, `temperature`, `top_p`, `top_k`, `repeat_penalty`, `seed`, `max_tokens`, `stop` |

```
curl localhost:8080/v1/chat/completions -H "Authorization: Bearer MY-SECRET" -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hello"}],"stream":true}'
```

Any OpenAI client works by pointing its base URL at `http://HOST:8080/v1` with the token as API key. Without `--token` (or
`DAFO_SERVE_TOKEN`) a random token is made and printed at start; the server listens on `127.0.0.1` unless `--host` says otherwise.
Details, limits and the format of the folder: [docs/SERVER.md](SERVER.md).

## Measured (Apple M4, 16 GB; Gemma 4 26B-A4B cut to 48 experts, Q4_K_M)

| | this compiler | Candle | llama.cpp |
|---|---|---|---|
| Generation, short context | **25.6 tok/s** | 24.3 | ≈ 25 – 26.8 |
| Reading a prompt (batched) | 168 – 205 tok/s | 150 – 200 | **312 – 373** |

Generation is limited by memory bandwidth: a token reads 3.21 GiB, so the ceiling on this chip is about 29 tok/s and every engine
sits near it. Reading a prompt is where llama.cpp is ahead today (1.5 – 2×); see [docs/RESULTS.md](RESULTS.md) for what was
tried and why. Correctness: every operation is checked against a CPU reference (`examples/check_ops.rs`), and over a 2,000-token
stretch past the sliding window the compiled engine's perplexity is 3,015.7 against llama.cpp's 3,017.9 on the same file.

## Limits

- **One model family.** Gemma 4 (dense and mixture-of-experts GGUF with Q4_K, Q5_1 and Q8_0 weights). Other families need a frontend
  and, if they use other quantizations, kernels.
- **One backend.** Metal. The graph is backend-independent; a CUDA backend would take the same graph.
- **Prompt reading** is slower than llama.cpp's tuned kernels.
- **The host is Gemma 4's**: tokenizer, chat template, tools format. The compiler core (`ir`, passes, backend, runtime) knows none of it.
- **`dafo-serve` serves one request at a time** (a queue), keeping several conversations in cache; it does not decode two at once.
  **Images are not supported** by it yet: the vision encoder lives in DAFO Swarm, not in this repository.
- **The folder stores no dispatch plans**: they are rebuilt at open (milliseconds).

## Layout

```
src/gguf.rs           GGUF reader: tensor table and metadata, memory-mapped
src/quant.rs          CPU reference decoding of Q8_0, Q5_1, Q4_K (for the tests)
src/ir.rs             the graph and the bytes a token reads
src/frontends/        one module per model family (gemma4 first)
src/backend_metal.rs  graph -> plan; buffer reuse; overlap scheduling; cache footprint
src/backend_metal/    the Metal kernels (kernels.metal, matmul.metal)
src/metal.rs          device, kernels, buffers, timed command buffers
src/runtime.rs        executes a plan
src/host/             tokenizer, Gemma 4 chat template with tools, sampler, engine thread, the compiled folder, the chat page (--features host)
src/bin/dafo-compile.rs   the command-line tool
src/bin/dafo-serve.rs     the OpenAI-compatible server (--features host)
examples/             correctness checks and profilers (check_ops, check_batch, profile_*, ...)
experiments/          the first experiments (output head in two stages; memory layout)
docs/                 how it works and the measured results
```

## Tests

```
cargo test                                          # unit tests (scheduling, kernel access analysis, GGUF, quantization)
cargo run --release --example check_ops -- MODEL.gguf     # each operation of a token, GPU against CPU
cargo run --release --example check_batch -- MODEL.gguf   # a batch against the same tokens one at a time
```

## License and credits

MIT (see `LICENSE`). The matrix-matrix kernels (`src/backend_metal/matmul.metal`) are adapted from
[llama.cpp](https://github.com/ggml-org/llama.cpp) (MIT) through the copy in `candle-metal-kernels`; see `NOTICE`. The GGUF format
is the one defined by the ggml project. Gemma is a model of Google and is subject to the Gemma Terms of Use.

Built for [DAFO Swarm](https://github.com/gabgiani/dafo-swarm-downloads), where it runs as an alternative engine for Gemma 4 on Macs.
