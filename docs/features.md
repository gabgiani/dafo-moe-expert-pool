---
title: Features
---

# What a compiled model can do

Everything below is served by one program, `dafo-serve`, over a compiled model (see [Use it](use.md)). It serves one request at a time (a queue) and keeps several conversations in cache.

## An OpenAI-compatible API with a bearer token
| Route | |
|---|---|
| `POST /v1/chat/completions` | chat, streaming (`stream: true`, server-sent events) or whole answers |
| `GET /v1/models` | the model, its context length |
| `GET /health` | model name and context; no token needed |
| `GET /`, `GET /v1/experts`, `POST /v1/experts` | the chat page; the expert pool ([below](#expert-pool-control)) |

Every `/v1` route needs `Authorization: Bearer TOKEN` (401 otherwise); the server listens on `127.0.0.1` unless you pass `--host`. Any OpenAI client works by pointing its base URL at `http://HOST:PORT/v1`.

**Request fields:** `messages` (roles `system`, `user`, `assistant`, `tool`; content as text or a list of parts), `tools`, `tool_choice` (`"none"` hides the tools), `stream`, `temperature` (0.7), `top_p` (0.8), `top_k` (20),
`repeat_penalty` (1.15), `repeat_last_n` (64), `seed`, `max_tokens` / `max_completion_tokens` (2048), `stop`. One extension, `"think": true`, lets the model reason before answering.
**Answers** carry `usage` (prompt and completion tokens), a finish reason (`stop`, `length`, `tool_calls`) and, when streaming, deltas as they are produced.

## Tool calls
Tools are declared in the OpenAI format and rendered the way Gemma 4 expects (a system-turn declaration). The model's answer is read in both of its formats, the JSON one and Gemma's native `<|tool_call>` syntax, and returned as
`tool_calls` with generated ids; `tool` messages go back as tool responses. A call to a tool you did not declare stays text. With tools declared the answer is read whole and sent as one chunk when streaming (a call is only known once it is complete).

## Images
Attach images as `image_url` parts with `data:` URLs (PNG, JPEG, WebP) in any message. They are read by a **native vision encoder** that runs on the same GPU as the model (see [the vision encoder](vision.md)), so no second runtime is needed.
A conversation that shows the same image again does not encode it again, and its cached prefix is reused.

## A chat page
`GET /` serves a page with no external resources: streaming, a stop button, tokens per second, a system prompt, **attachments (text files and images)**, and the token already filled in when you open it from the link `dafo-serve --open` prints.
With an expert pool it shows, in its header, how much of the pool was exchanged after each prompt and each answer.

## Conversation cache and prefix reuse
The server keeps `--slots` conversations in cache. A follow-up reads only the tokens that are new: with 550 of a 551-token prompt cached, the request took 1.8 s instead of 10 s. A request goes to the slot that shares the longest prefix with it
(the sliding-window layers keep a margin of positions so a conversation can go back a little, to the point where its prompt diverged from the previous answer).

## Sampling
Temperature, top-k, top-p, seed, and a repetition penalty applied **once per distinct token** (the way llama.cpp does it; applying it per occurrence made loops worse). The seed fixes the sampling stream.

## The compiled folder
`manifest.json` + weights (linked) + `tokenizer.json` (built from the GGUF: BPE with byte fallback) + GPU kernels + optional projector. Everything a host needs, nothing application-specific.

## Expert pool control
For mixture-of-experts models served with `--pool` ([the expert pool](expert-pool.md), [choosing experts](choosing-experts.md)):
* `GET /v1/experts` → mode (`manual`, `adapting`, `frozen`), the stored profiles, the share of the pool exchanged after the last prompt and after the last answer, and per layer; the share of what the router wanted that was outside the pool.
* `POST /v1/experts` → `{"profile": "name"}` switches to a stored set; `{"layers": {...}, "name": "x"}` sets one; `{"adaptive": true|false}` starts or stops the exchange; `{"freeze": true, "save_as": "mine"}` settles on the current block and saves it.
* A chat request can carry `"dafo_experts": "name"` to use a profile for that request.

## As a library
The crate can be linked instead: tokenize, `prefill`, `step`, sample, with the runtime knowing no model and no tokenizer. The server, chat template, tools and sampler are a separate layer (`--features host`).
