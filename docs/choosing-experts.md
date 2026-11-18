---
title: Choosing which experts to keep in memory
---

# How we decided which experts to load, and how that changes while you work

Gemma 4 26B-A4B has 30 layers of 128 experts; the router picks 8 per token. The experts are 13.4 of the model's 15.8 GiB. A 16 GB Mac cannot hold them all, so something has to decide **which experts are in memory**.
This page is the method we used and what it taught us. The mechanism that makes it possible (a router mask and a pool of slots, with the rest of the file on disk) is in [The expert pool](expert-pool.md).

## 1. First, measure what the router actually does
We cannot guess from the weights which experts a workload uses, so we recorded it. A patch to llama.cpp's server (`analysis/llama-moe-trace.patch`) logs, for every token and every layer, the 8 experts the router chose and their router probabilities.
We ran the **full** model on a calibration set of real work (customer-support style tasks in English: triage of incoming messages, reading email, answering from data, and five multi-step tool tasks), 20,426 tokens per layer of which 3,997 were generated.

What the trace showed:
* **Use is concentrated, but only moderately.** 16 experts per layer cover about half of the router's choices (uniform use would be 12.5 %), 32 cover 72 %, 64 cover 93 %. On average 119 of the 128 experts of a layer were used at least once.
* **There is no "SAP expert" or "closing-a-turn expert".** The control tokens (turn markers, the channel that holds the reasoning) are processed by the same spread of experts as any other token; no expert dominates a token type. The experts do not form families either
  (the average cosine similarity of an expert's router vector with its nearest neighbour is 0.33), and the router weights are flat (the 8 chosen carry only 26 % of the router's probability over the 128).
  This matches what the literature finds for other MoE models: experts specialize by *syntactic* role more than by topic.
* **But a task does concentrate use.** Looking at triage alone, 16 experts hold the main expert of 78–82 % of tokens and 24 hold 90 %; mixing all tasks, 62 % and 74 %. The unit of specialization is the *task*, not the token.

## 2. Choose by the router's own weight, not by how often an expert appears
Counting appearances ignores that the first of the 8 chosen carries most of the weight, and that prompt tokens (long tables of JSON) are not what the answer needs. We score each expert by the **renormalized router weight it receives**, weighting prompt tokens at a quarter of generated ones,
and force in any expert that once carried a weight of 0.7 or more on a generated token ("mandatory" experts). Calibration and validation sets are separate: what picks the experts is never what grades them.

## 3. A cut is a mask (this made everything else cheap to test)
Gemma 4's router takes the 8 largest probabilities and renormalizes among them. So removing an expert is the same as setting its logit to −∞ before the top-8: if the experts the full model would choose are all present, the layer's output is identical;
if one is missing, the next best present takes its place. We checked it: a mask over the full model and a cut file gave the same log-probabilities. That let us try selections without producing a new file each time.

## 4. How many experts are enough
| Experts per layer | What we saw |
|---|---|
| 16 | broke the model: incoherent text, loops in the reasoning channel (discarded) |
| 24–32 | unstable: a copied amount came out wrong (`486372765300364` for `912128.68`), answers left inside the reasoning channel |
| **48** | all 16 application tasks answered correctly; the main expert of 89 % of calibration tokens is resident |
| 64 | the same 16 of 16, 94 % coverage, but too much memory for a 16 GB Mac |

A fixed cut of 48 also **loses what its calibration did not contain**: it was calibrated on English business tasks and loses Spanish and general knowledge (it answers "what is the capital of France" wrongly).
That is the problem a fixed cut cannot solve and the reason for the next part: nothing should be thrown away for good.

## 5. Can the context predict the experts? (a simulation)
We recorded a second trace on the full model: 10 kinds of work (triage, mail, notes Q&A, code, numbers, summarizing, general knowledge, reasoning, Spanish, translation), 12 requests each, half for calibration and half for validation.
For each validation request we chose the experts with a policy and counted how many tokens had their main expert resident:

| Experts per layer | a fixed set (calibrated on other work) | from the prompt alone | prompt + fixed set | the profile of that kind of work | oracle |
|---|---|---|---|---|---|
| 32 | 60.0 % | 76.7 % | 78.1 % | 84.8 % | 94.3 % |
| 48 | 74.4 % | 84.6 % | 86.5 % | 92.8 % | 98.8 % |
| 72 | 88.2 % | 91.1 % | 94.7 % | 98.1 % | 99.9 % |

Reading the prompt predicts the experts (+10 points at 48, and the biggest gains are exactly where a fixed set fails: code 54 → 81 %, general knowledge 55 → 83 %). A profile of the kind of work adds about 8 more. This is a simulation on the router of the full model, with short prompts, so it says where the gain is, not what quality you will get.

## 6. So the pool is dynamic: how it works while you use it
Nothing is discarded; the pool of N experts per layer is exchanged by what the work asks for.

1. **The router reports what it wanted.** Besides the choice it makes with the pool's mask, the router kernel counts which experts it *would have chosen with every expert allowed* (a small counter per layer and expert). The host reads and clears it between requests. The share of those wanted experts that are not resident is the **miss rate**.
2. **After reading the prompt, before answering.** The demand of the prompt's tokens says what the answer will want, so the engine exchanges experts at that point (the layers have already processed the prompt; the key/value caches stay valid) and then decodes.
3. **After each answer.** The demand is folded into a running score per (layer, expert), each request weighing the same (`score = 0.6·old + 0.4·share`). In each layer the weakest resident expert is replaced by the strongest outside one **only if it was wanted at least 1.25× more**: the margin keeps the pool from oscillating.
4. **It settles.** When three consecutive requests exchange at most 1 % of the pool's slots, the pool is declared settled: that block of experts is frozen, saved as a profile, and nothing is exchanged any more. The model is now a fixed specialist for that work.
5. **It thaws.** If three consecutive requests miss clearly more than when it settled (1.5× plus five points), the topic has changed and exchange resumes.
6. **It can be driven by hand.** Profiles are named JSON files (`{"layer": [expert ids, hottest first]}`); you can start from one, switch between them per request or by `POST /v1/experts`, freeze the current block, or turn adaptation off.
7. **With two tiers of memory** (CUDA: video memory and system RAM) the experts are kept in order of their score: the hottest in video memory, the next in pinned RAM, the rest on disk; exchanging an expert is a copy of 3.7 MB.

**What it looks like.** Starting from a pool calibrated on English tasks and sending questions in Spanish one after another, the share of what the router wanted that was outside the pool fell from 49 % after the first prompt to about 10–12 % within three requests,
with the pool exchanging 692 experts at first and 100–150 afterwards. With `--debug` the server prints, after every prompt and answer, which **percentage of the pool was exchanged**, in all and per layer; the same numbers are in `GET /v1/experts` and in the chat header.

**What is not shown yet.** The settle-and-freeze behaviour was designed and its parts observed, but not demonstrated on a long real workload; and the answers of the adaptive pool were not benchmarked against the complete model (only equality with a fixed cut was verified).
See [Limits](limits.md).
