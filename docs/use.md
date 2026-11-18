---
title: Use it
---

# Compile a model, serve it, talk to it

## Requirements
* A Mac with Apple silicon, or a Windows/Linux machine with an NVIDIA GPU (driver installed; the CUDA toolkit is *not* needed to build or run: NVRTC is loaded at run time).
* Rust 1.85+ to build.
* A Gemma 4 GGUF (for example `gemma-4-26B-A4B-it-UD-Q4_K_M.gguf` from Hugging Face; Gemma is subject to the Gemma Terms of Use) and, for images, its projector `mmproj-*.gguf`.

```
cargo build --release --features "host vision"
```
(`host` is the server and chat; `vision` adds image reading and pulls one dependency, the `image` crate.)

## 1. Look at the model first
```
dafo-compile inspect MODEL.gguf
```
prints what one token reads (dense layers, experts, output head), the speed ceiling that follows from your memory bandwidth, and what the key/value cache of a conversation costs. No GPU needed.

## 2. Serve it (the quick way)
```
dafo-serve MODEL.gguf --port 8080 --token MY-SECRET --context 8192 --open
```
`--open` opens the chat page in your browser with the token already in the link (the page reads it and removes it from the address bar). Without `--token` a random one is made and printed.
An `mmproj-*.gguf` next to the model is found by itself and turns on image reading.

## 3. Or compile once into a folder
```
dafo-compile compile MODEL.gguf --context 16384 --out MODEL.dafo --projector mmproj.gguf
dafo-serve MODEL.dafo --port 8080
```
The folder holds `manifest.json`, the weights (hard-linked, not copied), `tokenizer.json`, the GPU kernels and the projector. It belongs to no application and can be moved to another machine of the same kind.
Opening it takes about 0.1 s: the dispatch plans are rebuilt from the weights in milliseconds, so the folder never goes stale against a driver update.

## 4. A model larger than your memory (mixture of experts)
```
dafo-serve FULL.gguf --pool 48 --profile profiles/cal48.json --profiles profiles --adaptive --debug --open
```
Keeps 48 experts per layer in memory and every other expert reachable on disk; see [The expert pool](expert-pool.md) and [Choosing experts](choosing-experts.md). With an NVIDIA GPU too small for the pool
add `--vram-experts V` to keep only the hottest V experts per layer in video memory and the rest in system RAM.

## Call the API
```
curl localhost:8080/v1/chat/completions -H "Authorization: Bearer MY-SECRET" -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hello"}],"stream":true}'
```
```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8080/v1", api_key="MY-SECRET")
reply = client.chat.completions.create(model="any", messages=[{"role": "user", "content": "Summarise this invoice in one line."}])
print(reply.choices[0].message.content)
```
Images, tools and every field are in [Features](features.md).

## Options of `dafo-serve`
| | |
|---|---|
| `--host`, `--port` | where to listen (default `127.0.0.1:8080`: this machine only) |
| `--token` or `DAFO_SERVE_TOKEN` | the bearer token |
| `--context N` | tokens of context per conversation |
| `--slots N` | conversations kept in cache (default 2) |
| `--projector FILE`, `--visual-tokens N` | vision projector and the most soft tokens one image may use (70, 140, 280 or 560) |
| `--pool N` | experts per layer kept resident; give the **full** GGUF |
| `--vram-experts V` | of those, how many in video memory (CUDA); the rest in pinned system RAM |
| `--profile FILE`, `--profiles DIR` | the experts to start with, and a folder of named profiles to switch between |
| `--adaptive`, `--debug` | exchange experts by what requests ask for; print the share of the pool exchanged |
| `--open` | open the chat page with the token in the link |

## Test it
```
cargo test --features "host vision"
cargo run --release --example check_ops   -- MODEL.gguf     # every operation of a token, GPU against CPU
cargo run --release --example check_batch -- MODEL.gguf     # a batch of tokens against the same tokens one by one
```
