#!/usr/bin/env python3
"""Render docs/benchmark.md from the docs/bench_*.json sources.

docs/benchmark.md is OWNED by this script: regenerate with
`make bench-all`; hand edits to results are prohibited. The fp32
baseline is mandatory — the renderer refuses to publish without it.
"""

import argparse
import json
import sys
from pathlib import Path

MODES = ["fp32", "int8", "int4", "ternary"]

METHODOLOGY = """## Methodology

All numbers are produced by `heph bench` on the same machine and
committed as JSON files under `docs/`. Nothing here is estimated.

- **decode tokens/s (batch 1)**: greedy decoding, `bench.max_new_tokens`
  new tokens per prompt, after `bench.warmup` discarded warmup prompts,
  median over `bench.prompts` prompts. Prompt prefill is excluded from
  the decode rate (it is reported as TTFT).
- **TTFT (s)**: time from invocation to the first emitted token,
  including prompt prefill. Median over the same prompts.
- **per-prompt latency**: median and p95 of the full generate call
  (prefill + decode) over all measured prompts.
- **peak RSS (MB)**: process peak from
  `getrusage(RUSAGE_SELF).ru_maxrss` (Linux reports KiB; converted).
  Caveat: it is the process lifetime peak, including the weight loader,
  not the steady-state decode footprint.
- **perplexity (optional)**: measured by the engine's bench subcommand
  when `--holdout` is provided (the one sanctioned optional extension
  of the bench CLI; additive flags only, existing flags unchanged).
  Defined as mean NLL over all predicted tokens on non-overlapping
  windows of `max_seq` over the frozen PROMETHEUS-NS holdout, with
  `ppl_tokens` recording the number of predicted tokens. fp32
  perplexity comes from the same engine path (fp32 logits are
  cross-checked against the NumPy oracle by the golden tests). Modes
  without a holdout run render '-'.
- **ppl delta vs fp32 (%)**: computed at render time as
  `(ppl_mode - ppl_fp32) / ppl_fp32 * 100` when both are present.

Honesty rules: re-running `make bench-all` overwrites this file and the
JSON sources; publishing requires committing both. A missing optional
metric renders as '-' and must never be invented.
"""


def load_mode(docs: Path, mode: str) -> dict | None:
    path = docs / f"bench_{mode}.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(value, spec: str = ".2f") -> str:
    if value is None:
        return "-"
    try:
        return format(float(value), spec)
    except (TypeError, ValueError):
        return str(value)


def cell(mode: str, key: str, spec: str = ".2f") -> str:
    raise NotImplementedError  # replaced below


def render(results: dict, fp32: dict | None) -> str:
    def cell(m: str, key: str, spec: str = ".2f") -> str:
        data = results.get(m)
        if data is None:
            return "pending"
        if key.startswith("_"):
            arr = data.get("per_prompt_ms") or []
            if not arr:
                return "-"
            s = sorted(arr)
            if key == "_median_ms":
                mid = len(s) // 2
                value = s[mid] if len(s) % 2 else (s[mid - 1] + s[mid]) / 2.0
            else:  # _p95_ms
                idx = min(len(s) - 1, int(round(0.95 * (len(s) - 1))))
                value = s[idx]
            return format(float(value), spec)
        return _fmt(data.get(key), spec)

    def ppl_delta(mode: str) -> str:
        data = results.get(mode)
        if data is None:
            return "pending"
        if mode == "fp32":
            return "-"
        if fp32 is None or data.get("ppl") is None \
                or fp32.get("ppl") in (None, 0):
            return "-"
        delta = (data["ppl"] - fp32["ppl"]) / fp32["ppl"] * 100.0
        return _fmt(delta, "+.2f")

    rows = []
    rows.append("| Metric | " + " | ".join(MODES) + " |")
    rows.append("|" + "---|" * (len(MODES) + 1))
    rows.append("| decode tokens/s (batch 1) | "
                + " | ".join(cell(m, "tokens_per_s_decode", ".1f")
                             for m in MODES) + " |")
    rows.append("| TTFT (s) | "
                + " | ".join(cell(m, "ttft_s") for m in MODES) + " |")
    rows.append("| per-prompt median (ms) | "
                + " | ".join(cell(m, "_median_ms", ".1f") for m in MODES)
                + " |")
    rows.append("| per-prompt p95 (ms) | "
                + " | ".join(cell(m, "_p95_ms", ".1f") for m in MODES)
                + " |")
    rows.append("| peak RSS (MB) | "
                + " | ".join(cell(m, "peak_rss_mb", ".1f") for m in MODES)
                + " |")
    rows.append("| perplexity | "
                + " | ".join(cell(m, "ppl") for m in MODES) + " |")
    rows.append("| ppl delta vs fp32 (%) | "
                + " | ".join(ppl_delta(m) for m in MODES) + " |")
    rows.append("| ppl tokens (n) | "
                + " | ".join(cell(m, "ppl_tokens", ".0f") for m in MODES)
                + " |")

    header = (
        "<!-- Generated by scripts/render_bench.py from docs/bench_*.json. "
        "Do not edit results by hand: rerun make bench-all. -->\n\n"
        "# HEPHAESTUS benchmark report\n\n"
    )
    return header + "\n".join(rows) + "\n\n" + METHODOLOGY


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", default="docs")
    ap.add_argument("--out", default="docs/benchmark.md")
    args = ap.parse_args()

    docs = Path(args.docs)
    docs.mkdir(parents=True, exist_ok=True)
    results = {mode: load_mode(docs, mode) for mode in MODES}
    fp32 = results["fp32"]

    out = Path(args.out)
    out.write_text(render(results, fp32), encoding="utf-8")
    present = [m for m in MODES if results[m] is not None]
    print(f"wrote {out} (modes present: {present or 'none'})")
    if fp32 is None:
        print("ERROR: fp32 baseline missing: run `make bench MODE=fp32` "
              "before publishing this report", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
