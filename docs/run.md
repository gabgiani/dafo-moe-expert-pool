---
title: Run it
---

# Run it

Requirements: a Mac with Apple silicon, Rust 1.85+, the model file (Gemma 4 26B-A4B-it, Q4_K_M GGUF from Hugging Face; Gemma is a model of Google and is
subject to the Gemma Terms of Use). The model file is **not** in this repository.

```
cargo build --release --features host
```

## Look at the model, benchmark it

```
./target/release/dafo-compile inspect MODEL.gguf                    # what a token reads, what the caches cost
./target/release/dafo-compile bench   MODEL.gguf --context 4096 --pool 48
```

## Serve it with the expert pool

```
./target/release/dafo-serve MODEL.gguf \
  --port 8091 --context 4096 --slots 1 \
  --pool 48 --profile profiles/cal48.json --profiles profiles --adaptive --debug --open
```

* `MODEL.gguf` is the **full** file (128 experts per layer); `--pool 48` keeps 48 per layer in memory (7.4 GiB).
* `--profile` is the experts to start with: a JSON `{"layer": [expert ids]}`. `scripts`/`analysis/moe-trace-stats.py` writes one from a trace of the
  full model; without one the pool starts with experts 0–47 of each layer and `--adaptive` moves it.
* `--open` opens the chat page with the token in the link (`http://127.0.0.1:8091/#token=...`; the page reads it and removes it from the address bar).
  Without `--token` a random one is made and printed.
* The log shows, after each prompt and answer, the miss rate and the **share of the pool exchanged**; `--debug` adds the count per layer.
  `GET /v1/experts` gives the same as JSON and the chat header shows it.

```
curl -s localhost:8091/v1/experts -H "Authorization: Bearer TOKEN"                                  # mode, profiles, % changed
curl -s localhost:8091/v1/experts -X POST -H "Authorization: Bearer TOKEN" -d '{"profile":"sap32"}'  # switch profile
curl -s localhost:8091/v1/experts -X POST -H "Authorization: Bearer TOKEN" -d '{"freeze":true,"save_as":"mine"}'
```

Any OpenAI client works against `http://127.0.0.1:8091/v1` with the token as API key. See [SERVER.md](SERVER.md) for the options, the tools format and limits.

## Compile to a folder (without the pool)

```
./target/release/dafo-compile compile MODEL.gguf --context 16384 --out MODEL.dafo
./target/release/dafo-serve MODEL.dafo --port 8080
```

## Tests

```
cargo test --features host
cargo run --release --example check_ops  -- MODEL.gguf               # each operation, GPU against CPU
cargo run --release --features host --example check_mask -- FULL.gguf --select profiles/cal48.json --pool 48   # equals the cut
```
