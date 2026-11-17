---
title: The adaptive pool
---

# The adaptive pool: exchange by demand, settle, stay

## What the router says it wants

Inside the `route` kernel, besides the choice made with the mask, a second selection is made **without** the mask — what the router would
have chosen with every expert allowed — and counted per (layer, expert) in a small buffer (`demand`). The host reads and clears it
between requests (`Runtime::take_demand`). The share of those wanted experts that are not resident is the **miss rate**.

## The policy (`--adaptive`)

* **After reading the prompt, before answering** (`adapt(prefill)`): the demand of the prompt's tokens says which experts the answer will want.
  The engine exchanges experts, then decodes. (The layers have already processed the prompt; the key/value caches stay valid.)
* **After each answer**: the demand is folded into a running score per (layer, expert) (each request weighs the same: `score = 0.6·score + 0.4·share`).
  In each layer the weakest resident expert is replaced by the strongest outside one **only if it was wanted at least 1.25× more** (hysteresis, so the
  pool does not oscillate).
* **Settling:** when three consecutive requests exchange at most 1 % of the pool's slots, the pool is declared settled. The block is frozen, saved
  as the profile `settled` (`settled.json`), and **nothing is exchanged any more** — the model is a fixed 48-expert specialist for that workload.
* **Thawing:** if three consecutive requests miss clearly more than when it settled (1.5× + 5 points), exchange resumes: the topic changed.
* `POST /v1/experts` can start/stop the exchange (`{"adaptive": true}`), freeze now (`{"freeze": true, "save_as": "name"}`) or switch to a stored
  profile (`{"profile": "name"}`); a chat request can carry `"dafo_experts": "name"`.

## Debug mode (`--debug`)

After each request the engine prints which **share of the pool was exchanged** (experts replaced ÷ (experts per layer × layers)), after the prompt and after
the answer, plus the count per layer. `GET /v1/experts` returns the same (`prefill_changed_percent`, `changed_percent`, `per_layer`, `last_miss`) and
the chat page shows it in its header. Without `--debug` the percentage is still in the log line; `--debug` adds the per-layer breakdown.

## An observed run

Starting from a pool calibrated on English customer-support tasks (`cal48`) and sending eight unrelated questions in **Spanish**, one after the other:

```
experts: after reading the prompt the router wanted 49.0 % outside the pool; 692 experts exchanged before answering
experts: the router wanted 23.6 % outside the pool; 353 experts exchanged
experts: after reading the prompt the router wanted  9.8 % outside the pool; 173 experts exchanged before answering
experts: the router wanted 11.6 % outside the pool; 149 experts exchanged
   ... 
experts: after reading the prompt the router wanted 11.0 % outside the pool; 142 experts exchanged before answering
```

The miss rate fell from 49 % to about 10–12 % within three requests. In that run it had **not settled** yet after eight requests (the exchanged
count stayed above the 1 % threshold); the earlier run without the prompt-time exchange went from 56.6 % to 9.0 % in ten requests. A longer run with a
real, stable workload is the test that matters, and was not done. What was shown is the mechanism: it adapts, it converges to a plateau and it reacts
to a change of topic (a miss rate of 27.7 % after changing subject).
