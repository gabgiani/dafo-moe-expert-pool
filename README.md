# Every Expert Available, 16 GB of Memory

_A compiled language-model engine with an exchangeable, adaptive expert pool — Metal and CUDA._

**A compiled language-model engine (Rust, Metal and CUDA) that serves a mixture-of-experts model larger than your memory with all of its experts still available.** Gemma 4 26B-A4B (128 experts per layer, 15.8 GiB) runs on a Mac with 16 GB of RAM at about
24 tokens/s: 48 experts per layer stay resident, every other expert is one disk read away, and the ones your conversations need are moved into memory by themselves until the choice settles.

```
cargo build --release --features "host vision"
dafo-compile pull gemma-4-26b-a4b-q4-k-m-pool       # download the model and its vision projector, like `ollama pull`
dafo-serve gemma-4-26b-a4b-q4-k-m-pool --pool 48 --profile profiles/cal48.json --profiles profiles --adaptive --debug --open
```

## What You Get From a Compiled Model
* An **OpenAI-compatible API with a bearer token** (streaming, tool calls, `usage`) and a **chat page** with text and image attachments; the link carries the token, so nothing has to be typed. → [Features](docs/features.md)
* **Images** read by a native vision encoder on the same GPU (no second runtime). → [The vision encoder](docs/vision.md)
* **Conversation caching** with prefix reuse, a compiled **folder** you can move between machines, sampling, and a library API.
* **Models by name, like ollama:** `dafo-compile list`, `pull`, `rm`; `dafo-serve NAME`.
* **Two platforms:** macOS on Apple silicon (Metal) and Windows/Linux with an NVIDIA GPU (CUDA, loaded at run time).
* **Control of the expert pool** while it runs: see which share of it was exchanged after each prompt and answer (`--debug`, `GET /v1/experts`, the chat header), switch profiles, freeze it, turn adaptation off.

## The Technique, in Short
1. **Measure** which experts the router uses on real work, by tracing the full model. → [Choosing experts](docs/choosing-experts.md)
2. **A cut is a mask.** The router renormalizes among the experts it picks, so removing an expert is the same as blocking it; a pool of slots in memory, with a table from expert to slot, makes any set of experts a model that costs only the pool's memory. → [The expert pool](docs/expert-pool.md)
3. **Exchange by demand.** The router also reports which experts it wanted but did not find; after each prompt and answer the pool swaps its weakest experts for the most wanted ones, settles when nothing changes, and resumes if the topic does.
4. **Compile.** The dispatch plan only names buffers, so swapping what is *in* the buffers needs no recompilation. → [Why a compiler](docs/why-compiled.md)

## Honest Summary
* Measured on an Apple M4 with 16 GB: 23.7–23.8 tokens/s with the pool (24.7 for a fixed 48-expert cut); the pool's logits are **identical** to the cut's; exchanging 32 experts takes 0.23 s.
* **On a 16 GB Mac only this engine runs the complete model**: llama.cpp and Candle need all 15.8 GiB mapped. Their numbers in the comparison table are for the 48-expert cut.
* At any moment the model uses the experts that are resident: a specialist that follows the work, not the full model at full quality. Reading the prompt predicts the experts (+10 points over a fixed set at 48 per layer, a simulation).
* **On a 6 GB NVIDIA card the complete model runs at about 3 tokens/s, which is not usable;** the CUDA backend is correct and fast when the experts fit in video memory (44 tokens/s with a 12-expert model).
* Settling on a long real workload, answer quality against the complete model, and pools of 72+ were **not** measured. → [Limits](docs/limits.md) · [Measurements](docs/RESULTS.md)

## How to Use It
→ [Use it](docs/use.md): build, compile to a folder, serve, call the API, open the chat, run on Windows.

## License and Credits
MIT (see `LICENSE`). The matrix-matrix kernels (`src/backend_metal/matmul.metal`) are adapted from [llama.cpp](https://github.com/ggml-org/llama.cpp) (MIT) through the copy in `candle-metal-kernels`; see `NOTICE`.
The GGUF format is the one defined by the ggml project. Gemma is a model of Google and is subject to the Gemma Terms of Use; no model file is included. `analysis/` holds the scripts and the llama.cpp patch used to measure the router.
