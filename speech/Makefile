REVISION := b59c13a733fce5cf193230d0fa317ee7f145a108
FILES := config.json generation_config.json processor_config.json tokenizer.json tokenizer_config.json model.safetensors LICENSE-WEIGHTS NOTICE.md
WORK := work/speech
CHECKPOINT := $(WORK)/orukeet
WINDOW_SECONDS := 15.0
DTYPE ?= float16
MEMGUARD ?=

.PHONY: help env download export build transcribe parity

help:
	@echo "Run from the repo root as make -f speech/Makefile <target>."
	@echo "env         create the speech uv env"
	@echo "download    fetch the oruk/orukeet root files at $(REVISION) into $(CHECKPOINT)"
	@echo "export      export a static $(WINDOW_SECONDS) s bundle, DTYPE=float16|float32; MEMGUARD=<script> wraps it"
	@echo "build       build anemone-speech in release"
	@echo "transcribe  BUNDLE= MANIFEST= AUDIO= OUT=: transcribe every clip of a corpus manifest"
	@echo "parity      MANIFEST= AUDIO= HYPS=: tokens vs transformers generate on the first 50 clips"

env:
	uv sync --project speech

download:
	hf download oruk/orukeet $(FILES) --revision $(REVISION) --local-dir $(CHECKPOINT)
	shasum -a 256 $(addprefix $(CHECKPOINT)/,$(FILES))

export:
	$(MEMGUARD) uv run --project speech python speech/convert/export_orukeet.py --checkpoint $(CHECKPOINT) --dtype $(DTYPE) --audio-seconds $(WINDOW_SECONDS) --out $(WORK)/exports

build:
	swift build -c release --product anemone-speech

transcribe:
	.build/release/anemone-speech --bundle $(BUNDLE) --manifest $(MANIFEST) --audio $(AUDIO) --out $(OUT)

parity:
	uv run --project speech python speech/convert/parity.py --checkpoint $(CHECKPOINT) --manifest $(MANIFEST) --audio $(AUDIO) --hyps $(HYPS) --max-seconds $(WINDOW_SECONDS)
