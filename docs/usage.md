# Using Anemone

Build once with `swift build -c release`; the binary is `.build/release/anemone`. Every command takes `--model`, which is a model folder made by the converter or a Hugging Face repo id (`org/name`) to download into `~/.cache/anemone`. Without `--model`, `$ANEMONE_MODEL` is used.

The first load of a model compiles it for the Neural Engine (about 7.5 minutes on an M3 Pro, 12 minutes for a 4096-context folder). macOS caches the result per executable, so later loads take a few seconds.

## Commands

| command | what it does |
|---|---|
| `anemone run "<prompt>"` | Answers one prompt and streams the reply to stdout. `--stats` prints speed to stderr. |
| `anemone chat` | Multi-turn chat in the terminal. An empty line or Ctrl-D quits. `--system` sets a system prompt. |
| `anemone serve` | OpenAI-compatible HTTP API on `127.0.0.1:8080` (`--host`, `--port`). |
| `anemone bench` | Six fixed prompts, plain and speculative: tokens/s, acceptance, draft and verify time, and a token-for-token lossless check. `--only plain` or `--only speculative` runs one mode; `--json` writes the results. |

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

Prebuilt model folders are not on Hugging Face yet; `ModelFolder.download` works for any repo that holds one, and a local folder works the same without it.

```swift
import Anemone

let folder = try await ModelFolder.download("org/qwen3-8b-anemone")
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
