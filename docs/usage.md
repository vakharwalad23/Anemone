# Using Anemone

Build once with `swift build -c release`; the binary is `.build/release/anemone`. Every command takes `--model`, which is a model folder made by the converter or a Hugging Face repo id (`org/name`) to download into `~/.cache/anemone`. Without `--model`, `$ANEMONE_MODEL` is used, else `vakharwalad23/Qwen3-8B-Anemone`.

The first load of a model compiles it for the Neural Engine (on an M3 Pro: 10 minutes for the 4B, 12 to 14 for an 8B with 4096 context, 19 to 23 for 8192 and 16384, 22 for the 14B). macOS caches the result per executable, so later loads take a few seconds, until the cache goes stale (see below).

## Commands

| command | what it does |
|---|---|
| `anemone run "<prompt>"` | Answers one prompt and streams the reply to stdout. `--stats` prints speed to stderr. |
| `anemone chat` | Multi-turn chat in the terminal. An empty line or Ctrl-D quits. `--system` sets a system prompt. |
| `anemone serve` | OpenAI-compatible HTTP API on `127.0.0.1:8080` (`--host`, `--port`). |
| `anemone bench` | Six fixed prompts, plain and speculative: tokens/s, acceptance, draft and verify time, and a token-for-token lossless check. `--only plain` or `--only speculative` runs one mode; `--json` writes the results with the chip, memory, macOS build and power source. |

Shared options: `--max-tokens` (reply length), `--thinking` (let Qwen3 think first; slower, because the drafter accepts less on thinking text), `--no-draft` on `run` (plain greedy decoding on the ANE).

```sh
.build/release/anemone run --model models/qwen3-8b --stats "Explain how a CPU cache works."
.build/release/anemone chat --model models/qwen3-8b --system "Answer in one paragraph."
```

## The server

```sh
.build/release/anemone serve --model models/qwen3-8b --port 8080
```

| endpoint | notes |
|---|---|
| `GET /v1/models` | the loaded model |
| `POST /v1/chat/completions` | `messages` (string content or a list of text parts), `max_tokens` or `max_completion_tokens`, `stream` |

- Decoding is greedy; `temperature`, `top_p` and other sampling fields are ignored.
- One request runs at a time; others wait their turn.
- Streaming uses server-sent events in the OpenAI chunk format and ends with `data: [DONE]`.

```sh
curl http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Name three primary colors."}], "max_tokens": 64}'
```

## The Swift package

```swift
// Package.swift
.package(url: "https://github.com/vakharwalad23/Anemone.git", branch: "main"),
// target dependencies
.product(name: "Anemone", package: "Anemone"),
```

`ModelFolder.download` fetches any Hugging Face repo that holds a model folder (once, into `~/.cache/anemone`); a local folder works the same without it.

```swift
import Anemone

let folder = try await ModelFolder.download("vakharwalad23/Qwen3-8B-Anemone")
let model = try await LanguageModel.load(from: folder)

let reply = try await model.respond(to: [ChatMessage(.user, "Write a haiku about the sea.")]) { piece in
    print(piece, terminator: "")
}
print("\n\(reply.stats.tokensPerSecond) tok/s, \(reply.stats.acceptedPerCycle) drafts accepted per cycle")
```

| API | notes |
|---|---|
| `LanguageModel.load(from:)` | loads a model folder; the first load compiles for the ANE |
| `respond(to:options:onText:)` | runs one reply, calling `onText` with each new piece of text; returns text, tokens and stats |
| `generate(_:options:)` | the same as an `AsyncThrowingStream` of `.text` and `.finished` events |
| `GenerationOptions` | `maxTokens`, `speculative` (default on), `thinking` (default off) |
| `ModelFolder.download(_:progress:)` | fetches a model folder from Hugging Face |

`LanguageModel` is an actor and runs one request at a time.

## Removing a model

A model downloaded by repo id lives in two places that share the same disk blocks (about 6 GB in total for Qwen3-8B): the model folder in `~/.cache/anemone` and the Hugging Face download cache. The compiled Neural Engine programs sit in the Core AI cache under the executable's name; remove them only to remove every model, since the next load then compiles again. The last two paths are small URL session files.

```sh
rm -rf ~/.cache/anemone/models/vakharwalad23/Qwen3-8B-Anemone
rm -rf ~/.cache/huggingface/hub/models--vakharwalad23--Qwen3-8B-Anemone
rm -rf ~/Library/Caches/coreai-cache/*/anemone
rm -rf ~/Library/Caches/anemone ~/Library/HTTPStorages/anemone
```

## When a model compiles again

If the Neural Engine compiler rejects part of a model, Core AI runs it on another device without an error, and the model is many times slower (a 32768-context export, 2026-10-27). After each load Anemone reads the compiled copy and prints a warning that names the functions running off the ANE.

Sometimes Core AI returns nothing for a cached compiled program instead of compiling it again. Anemone then deletes that program's cache entry (`~/Library/Caches/coreai-cache/<macOS build>/anemone/<main.hash in hex>`) and loads it once more, which compiles it from scratch: 771 s for an 8B and 1309 s for the 14B on an M3 Pro, then 6 to 20 s on later loads.

Keep free disk space: compiling a model takes about 2.5x the size of its `.aimodel` files, because macOS keeps a second copy for the Neural Engine outside this cache. The 4B, 8B and 14B together (15.6 GB of programs) took 39 GB; the 14B alone took 25.6 GB at the peak of its compile and 22.6 GB after (8.6 GB of it in the Core AI cache).

What we saw on 2026-10-25 (M3 Pro 18 GB, macOS 26A434):

- With 6 to 24 GB free all day, compiled models kept going stale: right after the 14B compiled, the 8B compiled just before it no longer loaded. After a crash, every compiled model went stale.
- With 60 GB free, the 4B, an 8B and the 14B compiled one after another, and all three then loaded from the cache in 8 to 15 s.
- Rebuilding Anemone and moving a model folder do not make a model stale.

Tested on 2026-10-27: with the 8B compiled and loading in 10.0 s, filling the disk until a write failed made macOS free 23 GB that no readable file held, the Neural Engine's compiled copies. Core AI's own cache (25.7 GB) was untouched, yet the next load of the 8B compiled again for 753.4 s. So when the disk runs low, macOS purges the Neural Engine's copy of compiled models, and the next load of each pays a full compile (sometimes Core AI returns nothing for the entry left behind instead, the case above). If a load still fails after the automatic retry, Anemone stops with a message; removing the whole cache makes every model compile again on its next load:

```sh
rm -rf ~/Library/Caches/coreai-cache/*/anemone
```
