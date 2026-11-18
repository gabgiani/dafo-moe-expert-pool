---
title: DAFO Compiler
---

# DAFO Compiler

_A compiler for language models: serve a model with an OpenAI-compatible API, a chat and images, and run a mixture-of-experts model larger than your memory with all its experts available._

## What a compiled model gives you

You compile a model file (GGUF) once and get a folder that any program can serve: an **OpenAI-compatible API with a bearer token**, a **chat page with image attachments**,
**tool calls**, **conversation caching**, and **image reading through a native vision encoder**. For a mixture-of-experts model that does not fit your memory it also gives
you something no other engine does on a 16 GB Mac: **all the experts of the model stay available**, and the ones your conversations need are moved into memory by themselves.

| You want to… | Read |
|---|---|
| compile a model, serve it, call the API, open the chat | [Use it](use.md) |
| know what the compiled model can do (API fields, tools, chat, images, cache) | [Features](features.md) |
| understand why compiling is better than interpreting, and where it is *not* | [Why a compiler](why-compiled.md) |
| run a model larger than your memory with all its experts available | [The expert pool](expert-pool.md) |
| know how the experts to keep in memory are chosen, and how that changes while you work | [Choosing experts](choosing-experts.md) |
| read images | [The vision encoder](vision.md) |
| see the numbers | [Measurements](RESULTS.md) · [Limits](limits.md) |

Platforms: **macOS on Apple silicon** (Metal) and **Windows or Linux with an NVIDIA GPU** (CUDA). One model family so far: Gemma 4.
