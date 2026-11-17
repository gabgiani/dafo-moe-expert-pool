---
title: Measurements
---

# Measurements

Machine: Apple M4, 16 GB, macOS. Model: `gemma-4-26B-A4B-it-UD-Q4_K_M.gguf` (15.8 GiB, 128 experts per layer). Every number below was measured here
unless it says *simulation*.

## Speed

`dafo-compile bench MODEL.gguf --context 4096 [--pool 48]`, one conversation, 64 generated tokens, 512-token prompt, two runs for the pool:

| Configuration | Weights in memory | Generation | Prompt reading |
|---|---|---|---|
| 48-expert cut file, no pool | 7.39 GiB | 24.7 tok/s (40.5 ms/token) | 167 tok/s |
| **Full file, pool of 48** | 7.41 GiB | **23.7 – 23.8 tok/s** (≈42 ms/token) | 159 – 160 tok/s |

The pool costs about 4 % of generation speed (the router scores 128 experts instead of 48 and does a second, unmasked selection to count demand).
Generation is memory-bound (a token reads 3.21 GiB; about 29 tok/s is the ceiling at 100 GB/s). An earlier measurement of the same cut file, before the
pool existed, gave 25.6 tok/s; the 24.7 above is the same file measured on the day, so part of the gap is the machine (thermal state, other programs).

Other costs: opening the engine on the full file with a profile takes 17 s (7.5 s of it copying 879 experts into the pool). Exchanging experts later:
32 experts (119 MB) in 0.23 s; 512 experts (1.9 GB) in 2.7 s.

## Equality with the cut file

Full file + pool of 48 + the selection a 48-expert cut kept, against the cut file itself, on a real prompt (`examples/check_mask.rs`):
the largest logit and the sum of all logits of the first three generated tokens are **identical**, and so are the 24 generated ids. The operations of the
cut file also still match their CPU reference (`check_ops`: 928 operations, 0 differ).

## Which experts to keep resident: a simulation

Trace of the **full** model on an RTX 3060 laptop with an instrumented llama.cpp server (a patch is in `analysis/`): 10 domains × 12 requests
(triage, mail, notes Q&A, code, numbers, summarising, general knowledge, reasoning, Spanish, translation), the first 6 of each domain for calibration
and the last 6 for validation, generated tokens only. For each validation request a set of *N* experts per layer is chosen by a policy and we count the
share of tokens whose **main expert** (the one with the largest router weight) is in the set. `analysis/moe-policy.py` reproduces it.

| N per layer | global (other domains) | prompt only | prompt + global | the domain's own profile | oracle (the answer itself) |
|---|---|---|---|---|---|
| 32 | 60.0 % | 76.7 % | 78.1 % | 84.8 % | 94.3 % |
| 48 | 74.4 % | 84.6 % | 86.5 % | 92.8 % | 98.8 % |
| 72 | 88.2 % | 91.1 % | 94.7 % | 98.1 % | 99.9 % |

* The out-of-business domains are where a fixed set fails (N = 48: code 53.7 %, general knowledge 54.8 %). Reading the prompt lifts them to 80.9 % and
  83.1 %: **the context predicts the experts** (about +10 points on average at 48), and a profile of the domain adds about +8 more.
* Limits of this simulation: the router of the *full* model (a pool changes later layers' routing); short prompts (tens of tokens; real agent prompts are
  longer and carry more information); 6 requests per domain profile; the oracle overfits. It says where the gain is, not what quality you will get.

Earlier, on a business workload, the 48-expert cut answered 16 of 16 application tasks correctly in English; a capabilities battery did not reach full
marks with any cut and the failures moved between cuts without a clear order; Spanish is lost when a cut is calibrated on English only.

## Memory

The pool is the expert memory: *N* experts × 3.72 MB × 30 layers (48 → 7.41 GiB; 72 → 10.5 GiB, which does **not** leave room on a 16 GB Mac and was not run).
The full file is only read to refill slots; after each copy the file's pages are released so the page cache does not hold a second copy.
