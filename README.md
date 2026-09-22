# HEPHAESTUS

A CPU-only inference engine in C++17 for the PROMETHEUS-NS nano decoder
(RMSNorm, SwiGLU, RoPE, GQA, tied embeddings). It loads the model's
safetensors weights with hand-written SIMD kernels (AVX2/AVX-512, no BLAS,
no Eigen), applies progressive weight quantization (int8 per-channel,
int4 groupwise, ternary 1.58-bit), quantizes the KV-cache, and reports
real benchmarks against the PyTorch baseline.

Correctness is enforced by golden tests: a pure-NumPy reference
implementation (`tests/golden/reference_model.py`) is the specification;
the engine must reproduce its greedy continuations exactly and its
logits within fixed tolerances.

## Architecture

```
flowchart LR
    subgraph Export
        A[PROMETHEUS-NS checkpoint .pt] --> B[export_artifacts.py]
        B --> C[nano_fp32.safetensors]
        B --> D[model_manifest.txt + tensors.tsv]
        B --> E[tokenizer vocab.json + merges.txt]
    end
    subgraph Engine heph
        F[safetensors loader] --> G[forward: GEMV SIMD + RMSNorm + SwiGLU + RoPE + GQA]
        G --> H[KV-cache int8 paged]
        G --> I[quantize: int8 / int4 / ternary]
        I --> G
        G --> J[generate / golden / bench]
    end
    K[reference_model.py oracle] -. golden tolerance .-> J
```

## Repository layout

```
src/
  model_def.h        canonical tensor names + arch contract (anchor)
  core/              tensor type, sha256, minimal json
  loader/            safetensors memory map + manifest/tensors.tsv parsing
  kernels/           GEMV, RMSNorm, SwiGLU, RoPE: scalar reference + SIMD
  model/             forward pass, forward-dump, greedy generation
  kv/                KV-cache: fp32, int8 quantized, 16-token pages
  quant/             weight quantization + quantized GEMV
  tokenizer/         byte-level BPE runtime (vocab.json + merges.txt)
  bench/             benchmark suite
tests/
  golden/            NumPy oracle + cross-engine golden tests
  cpp/               kernel parity tests (scalar vs SIMD)
scripts/             export, mapping, bench rendering
```

## Quickstart

```bash
make setup
# point config.yaml -> paths.checkpoint at the PROMETHEUS-NS nano .pt, then:
make export
python3 scripts/export_tokenizer.py --source <prometheus-ns>/artifacts/tokenizer/tokenizer.json
make test
```

## Usage targets

| Target | Purpose |
|--------|---------|
| `make export` | checkpoint -> safetensors + manifest + tokenizer artifacts |
| `make golden` | engine vs NumPy oracle (logits tolerance + exact greedy match) |
| `make quantize MODE=int4` | write quantized weight artifacts |
| `make bench MODE=fp32` | benchmark one mode |
| `make bench-all` | fp32, int8, int4, ternary -> docs/benchmark.md |
| `make chat` | greedy interactive REPL |

## Measured metrics

Every number below must come from an executed run (`docs/bench_*.json`);
none may be estimated.

| Metric | fp32 | int8 | int4 | ternary |
|--------|------|------|------|---------|
| decode tokens/s (batch 1) | pending | pending | pending | pending |
| perplexity (PROMETHEUS holdout) | pending | pending | pending | pending |
| peak RSS (MB) | pending | pending | pending | pending |
| greedy match vs oracle | pending | pending | pending | pending |

## Limits (honest)

- Batch size 1 only; no continuous batching. CPU only; CUDA is out of scope.
- Single fixed architecture (PROMETHEUS-NS nano); not a generic runtime.
- Ternary 1.58-bit quantization on a model NOT trained for it (no QAT) is
  expected to degrade substantially. That degradation is a primary finding
  of this project and must be reported as measured, never hidden.
- The C++ BPE runtime is a reader, not a trainer; artifacts come from HF.
- POSIX-only build (getrusage, sys/resource.h, GCC flags): compile and run
  on Linux or WSL2, as for PROMETHEUS-NS.
