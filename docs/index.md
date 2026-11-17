---
title: A Mixture-of-Experts model with every expert available, on a 16 GB Mac
---

# Every expert available, 16 GB of memory

This repository serves **Gemma 4 26B-A4B** (a mixture-of-experts model: 128 experts per layer, 8 chosen per token) on a Mac with
**16 GB of RAM** at about **24 tokens/s**, while **all 128 experts of every layer stay available**: the ones the conversation needs are
moved into memory on the fly, and when the choice settles they stay.

The model file is 15.8 GiB, so it cannot be loaded whole. Instead of cutting experts away for good (the usual way to fit it), a **pool**
of 48 experts per layer (7.4 GiB) is resident, the router is told which experts are in it, and the pool is refilled from the file on disk.

| Page | What it explains |
|---|---|
| [How the compiler works](how-it-works.md) | GGUF → graph → plans → one command buffer per token; why this is a *compiler* and not an interpreter |
| [The expert pool](expert-pool.md) | the technique: router mask + slot indirection, equal to a cut model at every moment |
| [The adaptive pool](adaptive.md) | exchanging experts by demand, settling on a block, thawing on a topic change; the debug mode |
| [Measurements](results.md) | speed, equality with the cut model, which policy picks experts best (and the limits of those numbers) |
| [Run it](run.md) | build, compile, serve, chat page and API |
| [Limits](limits.md) | what is not done, what was not measured |

Everything here was built and measured on one machine (Apple M4, 16 GB). Where something is a simulation or was not measured, the page says so.
