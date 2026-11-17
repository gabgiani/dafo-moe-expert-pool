# Every expert available, 16 GB of memory

**A mixture-of-experts LLM (Gemma 4 26B-A4B, 128 experts per layer) running on a Mac with 16 GB of RAM at about 24 tokens/s, with all 128
experts of every layer still reachable.** The ones the conversation needs are moved into memory on the fly; when the choice settles they stay.

* It is a **compiler**: it reads the GGUF file, builds the graph of one token and turns it into a fixed plan of Metal dispatches (Rust, no llama.cpp or Candle).
* The 15.8 GiB model does not fit in the ~11 GiB a 16 GB Mac gives to the GPU, so only a **pool of 48 experts per layer (7.4 GiB)** is resident. The
  router is told which experts are in the pool (the rest get a logit of −∞, so the weights renormalize among the others — exactly what a model cut to
  those experts computes) and the pool is **refilled from the file on disk**, between tokens, by what the router keeps asking for.
* When the exchange stops being needed (three requests in a row change at most 1 % of the pool) the block is **frozen** and nothing moves any more:
  a specialist for that workload. If the topic changes, it thaws.
* A **debug mode** shows which share of the pool was exchanged after the prompt and after each answer, per layer.
* `dafo-serve` is an OpenAI-compatible server (bearer token, streaming, tools, text attachments) with a chat page; its link carries the token, so nothing has to be typed.

```
cargo build --release --features host
./target/release/dafo-serve gemma-4-26B-A4B-it-UD-Q4_K_M.gguf --port 8091 --context 4096 --slots 1 \
  --pool 48 --profile profiles/cal48.json --profiles profiles --adaptive --debug --open
```

## Read how it works

| | |
|---|---|
| [The expert pool](docs/expert-pool.md) | the technique: router mask + slot indirection; why a mask equals a cut; why mapping the whole file fails on 16 GB |
| [The adaptive pool](docs/adaptive.md) | exchange by demand, settle, freeze, thaw; the debug mode; an observed run |
| [Measurements](docs/results.md) | speed (23.8 tok/s with the pool, 24.7 for the cut file), equality with the cut, and a simulation of which experts to keep |
| [How the compiler works](docs/how-it-works.md) | GGUF → graph → passes → Metal plan → runtime |
| [Run it](docs/run.md) · [Server](docs/SERVER.md) · [Limits](docs/limits.md) | |

## Honest summary

* Measured (Apple M4, 16 GB): **23.7–23.8 tok/s** generating and about 160 tok/s reading a prompt with the pool of 48; the pool's logits are **identical** to a 48-expert
  cut file's; swapping 32 experts takes 0.23 s.
* At any moment the model is a **48-expert** model: this is not full-model quality. What is new is that no expert is thrown away and the 48 follow the conversation.
* Which experts to keep resident was studied on a trace of the full model (`analysis/`): reading the prompt predicts the experts (+10 points of main-expert
  coverage over a fixed set at 48 per layer, a *simulation*). Settling on a long real workload, answer quality against the full model, and pools of 72+ (they do not fit 16 GB)
  were **not** measured. See [Limits](docs/limits.md).
* One model family, one backend (Metal). The original compiler README is [docs/COMPILER.md](docs/COMPILER.md).

## License and credits

MIT (see `LICENSE`). The matrix-matrix kernels (`src/backend_metal/matmul.metal`) are adapted from [llama.cpp](https://github.com/ggml-org/llama.cpp) (MIT) through the copy in
`candle-metal-kernels`; see `NOTICE`. The GGUF format is the one defined by the ggml project. Gemma is a model of Google and is subject to the Gemma Terms of Use; the
model file is not included. `analysis/llama-moe-trace.patch` is a patch for llama.cpp's server that records which experts the router chooses.
