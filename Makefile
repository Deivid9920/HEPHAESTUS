# HEPHAESTUS build orchestration.
#
# Engine CLI contract (fixed here; scripts and tests rely on it):
#
#   heph golden  --manifest M --weights W --tokenizer DIR
#                --prompts FILE --tokens N --dump FILE
#       Dumps fp32 logits of the last position for every prompt, as raw
#       little-endian float32 [n_prompts][vocab], for the golden tests.
#
#   heph generate --manifest M --weights W --tokenizer DIR
#                 [--prompt TEXT | --interactive]
#                 [--greedy] [--max-new-tokens T] [--out FILE]
#       Greedy or sampled generation; streaming token-by-token on stdout.
#       With --out FILE (non-interactive), writes one line of token ids.
#
#   heph tokenize --tokenizer DIR --file IN --out OUT
#       One line of whitespace-separated token ids per input line.
#
#   heph quantize --manifest M --weights W --mode {int8,int4,ternary}
#                 --group G --out DIR
#       Writes DIR/tensors.tsv + DIR/weights.bin (engine-side format
#       documented in src/quant/quantize.h contract).
#
#   heph bench --manifest M --weights W --tokenizer DIR --mode M
#              --prompts N --warmup W --max-new-tokens T --out FILE
#              [--holdout DIR]
#       Writes JSON: {mode, tokens_per_s_decode, ttft_s, peak_rss_mb,
#       per_prompt_ms[]} — real measured values only. --holdout adds
#       ppl + ppl_tokens measured on the frozen PROMETHEUS-NS holdout
#       (the one sanctioned optional extension of the bench CLI).

PYTHON  ?= python3
VENV    := .venv
PY      := $(VENV)/bin/python

MODE    ?= fp32
MANIFEST  := artifacts/model_manifest.txt
TENSORS   := artifacts/tensors.tsv
WEIGHTS   := artifacts/nano_fp32.safetensors
TOKDIR    := artifacts/tokenizer
HEPH      := build/heph

.PHONY: setup export configure build test golden quantize bench bench-all chat lint docker-build clean

setup:
	test -f requirements.txt
	$(PYTHON) -m venv $(VENV)
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt

export:
	$(PY) scripts/export_artifacts.py --config config.yaml

configure:
	cmake -S . -B build -DCMAKE_BUILD_TYPE=Release

build: configure
	cmake --build build -j

test: build
	ctest --test-dir build --output-on-failure
	$(PY) -m pytest tests/golden -v

golden: build
	$(PY) tests/golden/golden_test.py --config config.yaml

quantize: build
	$(HEPH) quantize --manifest $(MANIFEST) --weights $(WEIGHTS) \
		--mode $(MODE) --group 128 --out artifacts/quant/$(MODE)

bench: build
	$(HEPH) bench --manifest $(MANIFEST) --weights $(WEIGHTS) \
		--tokenizer $(TOKDIR) --mode $(MODE) --prompts 50 --warmup 5 \
		--max-new-tokens 128 --out docs/bench_$(MODE).json

bench-all: build
	for m in fp32 int8 int4 ternary; do \
	  $(MAKE) bench MODE=$$m; \
	done
	$(PY) scripts/render_bench.py --docs docs --out docs/benchmark.md

chat: build
	$(HEPH) generate --manifest $(MANIFEST) --weights $(WEIGHTS) \
		--tokenizer $(TOKDIR) --interactive --greedy

lint: build
	cmake --build build 2>&1 | tee /dev/null
	@command -v clang-format >/dev/null 2>&1 && \
	  clang-format --dry-run -Werror src/*/*.cpp src/*/*.h || \
	  echo "clang-format not installed: skipped"

docker-build:
	docker build -t hephaestus .

clean:
	rm -rf build tests/golden/fixtures
