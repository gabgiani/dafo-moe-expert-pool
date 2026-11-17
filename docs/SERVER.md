# dafo-serve: a compiled model behind an OpenAI-compatible API

Build: `cargo build --release --features host` (a Mac with Apple silicon). Two programs come out of it:

```
dafo-compile compile MODEL.gguf --context 16384 --out MODEL.dafo     # leaves the compiled model in a folder
dafo-serve MODEL.dafo --port 8080 --token MY-SECRET                  # serves it
```

`dafo-serve MODEL.gguf` also works: it compiles at start and leaves nothing on disk.

## The compiled folder

```
MODEL.dafo/
  manifest.json   format, compiler version, model name, context it was compiled for, the plans (batch, dispatches), stop tokens
  model.gguf      the weights: a hard link to the file compiled from (no second copy), else a symbolic link; --copy-weights copies
  tokenizer.json  the tokenizer, ready to load
  kernels.metal   the Metal source generated for this model
```

It belongs to no application and can be moved to another Mac (use `--copy-weights` first if the folder must be self-contained).
What is *not* stored are the dispatch plans: they are rebuilt from the weights in milliseconds when the folder is opened (a 7.4 GiB
model is ready in 0.1 s), so the folder cannot go stale against the Metal driver. Storing them is possible later; the format has a
version number for it.

## Options

| | |
|---|---|
| `--host` | address to listen on (default `127.0.0.1`: only this Mac) |
| `--port` | default 8080 |
| `--token` | the bearer token; or `DAFO_SERVE_TOKEN`; without either a random one is made and printed |
| `--context` | tokens of context per conversation; default the folder's, or 16384 for a `.gguf` |
| `--slots` | conversations kept in cache (default 2); a follow-up that continues one reads only its new tokens |

### Serving more experts than fit in memory (`--pool`)

| | |
|---|---|
| `--pool N` | keep `N` experts per layer in memory out of the model's (128 for Gemma 4 26B-A4B); give it the **full** GGUF. [How it works](expert-pool.md) |
| `--profile FILE` | the experts to load at start: JSON `{"layer": [expert ids]}` |
| `--profiles DIR` | a folder of `name.json` profiles a request or `POST /v1/experts` can switch to |
| `--adaptive` | exchange experts by what the requests ask for, until the pool settles ([adaptive pool](adaptive.md)) |
| `--debug` | after every request, print the share of the pool that was exchanged, per layer; the chat header shows it too |
| `--projector FILE` | the vision projector (`mmproj-*.gguf`; one beside the model is found by itself); needs a build with `--features vision`. [How it works](vision.md) |
| `--open` | open the chat page in the browser; the link carries the token (`#token=...`) so nothing has to be typed |

`GET /v1/experts` reports the mode (`manual`, `adapting`, `frozen`), the profiles, how much of the pool changed after the last prompt and after
the last answer (`prefill_changed_percent`, `changed_percent`, `per_layer`) and the share of what the router wanted that was outside the pool
(`last_miss`). `POST /v1/experts` takes `{"profile": "name"}`, `{"layers": {...}, "name": "x"}`, `{"adaptive": true|false}` or
`{"freeze": true, "save_as": "name"}`. A chat request can carry `"dafo_experts": "name"` to use a profile for that request.

The context costs memory: one conversation needs a fixed part plus bytes per token (`dafo-compile inspect MODEL.gguf` prints both;
for the 48-expert Gemma 4 26B-A4B it is 325 MiB + 20 KiB per token).

## API

All routes under `/v1` need `Authorization: Bearer TOKEN` (401 otherwise). `/` and `/health` do not.

`POST /v1/chat/completions` reads `messages` (roles `system`, `user`, `assistant`, `tool`; content as a string or a list of text
parts), `tools`, `tool_choice` (`"none"` hides the tools), `stream`, `temperature` (default 0.7), `top_p` (0.8), `top_k` (20),
`repeat_penalty` (1.15), `repeat_last_n` (64), `seed`, `max_tokens` / `max_completion_tokens` (2048), `stop`. One extension:
`"think": true` lets the model reason before answering; by default its thought is closed empty, which is faster and is what the
cache of a follow-up expects.

Tools follow Gemma 4's format: declared in the system turn, answered as `<tool_call>…</tool_call>` (Gemma's native
`<|tool_call>call:name{…}<tool_call|>` is read too), and returned as OpenAI `tool_calls` with `finish_reason: "tool_calls"`. A call
to a tool that was not declared stays in the text. With tools declared a streaming request is answered whole in one chunk, because a
call is only known once it is complete.

Errors use the OpenAI shape: `{"error": {"message", "type", "code"}}`. A conversation longer than the compiled context is a 400
with code `context_length_exceeded`.

## Chat page

`GET /` serves a page with no external resources: token field (kept in the browser), system prompt, streaming, a stop button, tokens
per second, and attachments. **Attachments are text files** (`.txt`, `.md`, `.csv`, `.json`, code…): the page puts their content in the
message, up to 400 kB each. **Images** (PNG, JPEG, WebP, up to 8 MB) are attached too when the server has a vision projector (`--projector`, built with
`--features vision`): they travel as `image_url` parts with `data:` URLs; without a projector a request with an image gets a 400 that says so. See [vision.md](vision.md).

## Limits

- Gemma 4 only (tokenizer, chat template, tools format), on Metal.
- One request is served at a time; the others wait. Several conversations stay in cache, so alternating between them does not
  re-read them. It does not decode two at once: generation is limited by memory bandwidth, so that would not be faster.
- No image input, no JSON-schema constrained output, no log-probabilities, no `n > 1`.
- The token protects the API; the server speaks plain HTTP. To expose it beyond a trusted network put TLS in front of it.
