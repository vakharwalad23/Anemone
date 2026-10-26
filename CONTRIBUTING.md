# Contributing

Anemone is young: three published models (Qwen3-4B, Qwen3-8B, Qwen3-14B; the converter also handles DeepSeek-R1-0528-Qwen3-8B) on one tested Mac (M3 Pro). Results from other Macs are the most useful contribution right now: run `anemone bench` and open an issue with its output, your chip, memory and macOS build.

## Requirements

- Apple Silicon Mac, macOS 27, Xcode 27, Swift 6.
- Python tools (the model converter) run through [uv](https://docs.astral.sh/uv/) only.

## Code

- One responsibility per file and per type. If a file needs a section header to be readable, split it.
- Small functions whose names say what they do.
- No dead code, no commented-out code, no unused parameters, no scaffolding for later.
- No abstraction without a second user: no protocol with one conformer, no factory for one product, no setting for a value that never changes.
- Hardware choices (ANE, GPU, CPU), shapes and quantization are explicit arguments, never hidden globals.
- Swift follows `swift format` defaults. No force unwraps outside tests.
- Python: line length 140, type hints on public functions.
- Model weights, converted models, caches and raw results never go into git.

## Comments

- One line, and only when the reason is not obvious from the code: a hardware quirk, a measured number, a constraint.
- No multi-line comments, no banners, no section dividers, no TODO without an owner and a reason.
- Never restate what the code does. Rename or extract a function instead.

## Commits

[Conventional Commits](https://www.conventionalcommits.org/): `type(scope): subject`.

- Types: `feat`, `fix`, `perf`, `refactor`, `test`, `docs`, `build`, `chore`, `exp`.
- Scopes: `engine`, `target`, `drafter`, `cli`, `sdk`, `bench`, `convert`, `repo`.
- Subject: imperative, lowercase, no trailing period, 50 characters or fewer. Body only when the reason is not obvious, wrapped at 72.
- One logical change per commit. No generated-by or co-author trailers for tools.

```text
perf(target): chain chunks with encode
```

## Docs

- ASCII only. This must print nothing:

```sh
git grep -nP '[^\x00-\x7F]'
```

- Lead with what the reader needs. Plain words, short sentences, no marketing words, no emoji.
- A flow with three or more steps gets a Mermaid diagram.
- A calculation gets its formula in a `math` block, every symbol defined, and one worked example with measured numbers.
- Say what does not work yet.

## Numbers

A performance or quality claim needs a measurement behind it.

- AC power (`pmset -g batt` says AC Power), quiet machine, no video call or camera app (they share the ANE).
- Warm up first. Report the median of 20 warm runs. Compare A and B as interleaved pairs.
- Record chip, memory, macOS build and power source with every result. Never compare numbers across macOS versions.
- Speed changes must keep the output lossless: speculative output equal to plain greedy, token for token.

## Pull requests

- One topic per pull request, small enough to review in one sitting.
- Say what changed, why, and the numbers before and after if it touches speed or quality.
