---
title: Limits
---

# Limits

* **One model family, one backend.** Gemma 4 (the MoE 26B-A4B for the pool), Metal only. The graph is backend-independent; a CUDA backend would take the
  same graph (a plan exists, none is written).
* **The pool is a cut that moves.** At every moment the model is a 48-expert model; answers are as good as the resident experts match the conversation.
  No benchmark compares the adaptive pool's answers with the full 128-expert model's; only equality with a fixed cut was verified, and the policy
  numbers are a simulation.
* **Settling was designed and its mechanism observed, but not shown on a long real workload.** In the runs reported the pool had not settled yet.
* **72 experts and larger pools were not run**: they need more memory than a 16 GB Mac has. A machine with enough memory could use them, and could map the whole file
  (no pool needed).
* **Speed:** about 24 tok/s generating, about 160 tok/s reading a prompt; llama.cpp reads prompts 1.5–2× faster on the same machine.
* **`dafo-serve` serves one request at a time** (a queue). Images need `--features vision` and a projector file; the native encoder is verified against a reference on one image (cosine 0.999999) and is not tuned (about 4–5 s for a full image).
* **Exchanging happens between requests** (and once after the prompt), never in the middle of an answer.
* The 15.8 GiB file must be on a local disk; reading it is only needed to refill slots.
