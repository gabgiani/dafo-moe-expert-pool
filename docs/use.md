---
title: Use It
---

# Use It

_Pull a model, compile it, serve it, call the API_

## Quick start: three commands
```
dafo-compiler pull gemma4     # downloads the model that fits this machine, with its vision projector
dafo-compiler serve           # the API and the chat page on http://127.0.0.1:8080 (stays running)
dafo-compiler chat            # a chat in the terminal (joins the running server, or loads the model itself)
```
Nothing to configure. The machine is looked at and these are decided for you:
* **Experts in memory** (for a mixture-of-experts model larger than your memory): how many experts per layer stay resident, from the Mac's memory or from the free video memory and RAM of an NVIDIA machine; the others stay one disk read away.
* **Context length** and **how many conversations run at once** (up to 4), from what memory is left.
* **Images** on or off, depending on whether the vision encoder fits next to the model.
* On an NVIDIA GPU, how many experts go in video memory and how many in pinned RAM.

`serve` and `chat` print one line saying what was chosen and why, and notes when something will be slow; `dafo-compiler show` prints it without starting anything. The API token is made once and kept in `~/.dafo/token`.
In the chat: `/image PATH` attaches an image to your next message, `/clear` forgets the conversation, `/bye` leaves. Without a name, the model is the only one you pulled or the last one you used; `list`, `rm` and `show` do what they say.
The longer commands below are for when you want to choose.

## Requirements
* A Mac with Apple silicon, or a Windows/Linux machine with an NVIDIA GPU (driver installed; the CUDA toolkit is *not* needed to build or run: NVRTC is loaded at run time).
* Rust 1.85+ to build.
* A Gemma 4 GGUF: pull one with `dafo-compile pull` (below), or bring your own (for example `gemma-4-26B-A4B-it-UD-Q4_K_M.gguf` from Hugging Face; Gemma is subject to the Gemma Terms of Use) with its projector `mmproj-*.gguf` for images.

```
cargo build --release --features "host vision"
```
(`host` is the server and chat; `vision` adds image reading and pulls one dependency, the `image` crate.)

## 0. Get a model, the long way
```
dafo-compiler list                                 # the models you can pull, with their size and which are already here
dafo-compiler pull gemma-4-26b-a4b-q4-k-m-pool     # a model by name
dafo-serve gemma-4-26b-a4b-q4-k-m-pool --open      # serve it by name, with every option available
dafo-compiler rm gemma-4-26b-a4b-cut48-q4-k-m      # delete one
```
`list` reads the same catalog the DAFO Swarm nodes read (a release of `gabgiani/dafo-swarm-downloads`) and shows only the models this compiler can run. `pull` rebuilds a model that is published in chunks (the 48-expert cut) by downloading each chunk,
checking it against its own SHA-256, joining them, and checking the finished file; a damaged chunk is fetched again and the rest are kept. A model served from one address (the complete 128-expert file, from Hugging Face) is downloaded with resume.
Both pull the vision projector too, so images work at once. Everything goes to `~/.dafo/models/<name>/` (`DAFO_MODELS` changes it; `DAFO_CATALOG_URL` points to another catalog). If a pull is interrupted, run it again and it resumes.

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
| `NAME` instead of a path | a model pulled with `dafo-compile pull` |
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
