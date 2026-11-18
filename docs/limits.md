---
title: Limits
---

# Limits

_What is not done and what was not measured_

* **One model family.** Gemma 4 (the mixture-of-experts 26B-A4B for the expert pool; the dense and small variants have the compiled path but not the pool). Another family is a new frontend.
* **Two GPU languages, unequal maturity.** Metal is the tuned one. The CUDA kernels are a first version: no tensor cores yet, matrix-vector products at about a third of video-memory bandwidth, and on Windows/Linux the Swarm's own engine does not use them yet (only `dafo-serve` and `dafo-compile`).
* **A complete model larger than video memory is slow on a small NVIDIA card.** With the experts split between video memory and system RAM, the GPU reads the RAM ones over PCIe (3.5 GB/s measured for a kernel, 5.6 GB/s for a DMA copy on an RTX 3060 laptop):
  about 3 tokens/s generating and 4 tokens/s reading a prompt. llama.cpp with the experts on the CPU does 15–17 on the same machine. On a GPU that holds the experts it does not apply (44 tokens/s with a 12-expert model entirely in video memory).
* **Images and the model compete for 6 GB.** On that card the vision encoder needs 1.2 GB of weights plus activations next to the model; it worked with 8 experts per layer in video memory, one 280-token image (2 s), and no other process on the GPU.
* **Pinned system memory is limited** by the operating system (about 5–6 GB on the Windows test machine), which caps the RAM tier of the expert pool.
* **The pool is a cut that moves.** At any moment the model uses the experts that are resident; the answers are as good as those match the conversation. Only equality with a fixed cut was verified, plus a simulation of which experts to keep; there is no benchmark of the adaptive pool's answers against the complete model.
* **Settling was designed and its parts observed, not shown on a long real workload.** In the runs reported the pool had not settled yet. With the adaptive pool on, the same request repeated can come back worded differently (the pool changed between the two).
* **72 experts and larger pools** need more memory than a 16 GB Mac has; they were not run.
* **Speed:** about 24 tokens/s generating and 160 tokens/s reading a prompt on an M4; llama.cpp reads prompts 1.5–2× faster on the same machine.
* **`dafo-serve` serves one request at a time** (a queue), reads images only as `data:` URLs, and does not do JSON-schema constrained output (the Swarm's own engine does).
* **Exchanging experts happens between requests** (and once after the prompt), never in the middle of an answer. The images' soft-token plan holds 576 tokens, so a budget of 1120 does not fit.
* **The 15.8 GiB file must be on a local disk;** it is read only to refill slots.
