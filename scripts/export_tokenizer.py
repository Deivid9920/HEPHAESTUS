#!/usr/bin/env python3
"""Export the PROMETHEUS-NS tokenizer into engine-side artifacts.

This is the ONLY tokenizer artifact producer. Run it once after
`make setup`; it fails loudly if the source is not a byte-level BPE or
if the rebuilt vocab/merges diverge from the source on the parity
samples. Outputs (into --out, default artifacts/tokenizer):
  vocab.json      token string -> id, in the byte->unicode space
  merges.txt      one merge per line, priority order (rank = line number)
  specials.json   {bos, eos, pad, unk} ids
  tokenizer.json  verbatim copy of the source (HF cross-checks read it)

Byte-level pitfalls the C++ runtime MUST handle (parity tests enforce):
vocab keys are in the byte->unicode space ('G\u0300' = U+0120 for byte
0x20), the regex pre-split MUST run before merges, and encode must NOT
inject special tokens.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers

SAMPLE_STRINGS = [
    "The quick brown fox jumps over the lazy dog.",
    "La velocidad de la luz es constante en el vacío.",
    "In 2024, the model processed 3,141 tokens/second.",
    "el ruido y la señal: una historia sobre datos",
]


def load_source(path: Path) -> tuple[dict, dict, dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    model = data["model"]
    if model.get("type") != "BPE":
        raise SystemExit(f"source model type is {model.get('type')!r}, need BPE")
    if model.get("byte_fallback"):
        raise SystemExit("byte_fallback tokenizers are out of contract")
    pre = data.get("pre_tokenizer") or {}
    if pre.get("type") != "ByteLevel":
        raise SystemExit("source must use the ByteLevel pre-tokenizer")
    return data, model, pre


def parse_merges(model: dict) -> list[tuple[str, str]]:
    merges: list[tuple[str, str]] = []
    for entry in model["merges"]:
        if isinstance(entry, list):
            merges.append((entry[0], entry[1]))
        else:
            left, right = entry.split(" ", 1)
            merges.append((left, right))
    return merges


def extract_specials(data: dict) -> dict[str, int]:
    by_content = {t["content"]: int(t["id"]) for t in data.get("added_tokens", [])}
    lookup = {
        "bos": ("<|bos|>", "<s>"),
        "eos": ("<|eos|>", "</s>"),
        "pad": ("<|pad|>", "<pad>"),
        "unk": ("<|unk|>", "<unk>"),
    }
    specials: dict[str, int] = {}
    missing = []
    for name, candidates in lookup.items():
        for cand in candidates:
            if cand in by_content:
                specials[name] = by_content[cand]
                break
        else:
            missing.append(name)
    if missing:
        raise SystemExit(f"source tokenizer lacks special tokens: {missing}")
    return specials


def rebuild_tokenizer(model: dict, pre_tokenizer: dict,
                      vocab: dict[str, int], merges: list[tuple[str, str]]):
    bpe = models.BPE(
        vocab=vocab,
        merges=merges,
        unk_token=model.get("unk_token"),
        fuse_unk=model.get("fuse_unk", False),
        byte_fallback=model.get("byte_fallback", False),
    )
    tok = Tokenizer(bpe)
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(
        add_prefix_space=pre_tokenizer.get("add_prefix_space", True),
        use_regex=pre_tokenizer.get("use_regex", True),
    )
    tok.decoder = decoders.ByteLevel()
    return tok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True,
                    help="path to PROMETHEUS-NS tokenizer.json")
    ap.add_argument("--out", default="artifacts/tokenizer")
    args = ap.parse_args()

    source_path = Path(args.source)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    data, model, pre_tokenizer = load_source(source_path)
    vocab: dict[str, int] = model["vocab"]
    merges = parse_merges(model)
    specials = extract_specials(data)

    original = Tokenizer.from_file(str(source_path))
    rebuilt = rebuild_tokenizer(model, pre_tokenizer, vocab, merges)

    prompts_file = Path("tests/golden/prompts.txt")
    if prompts_file.is_file():
        for line in prompts_file.read_text(encoding="utf-8").splitlines():
            if line.strip() and not line.startswith("#"):
                SAMPLE_STRINGS.append(line.strip())

    mismatches = 0
    for text in SAMPLE_STRINGS:
        expected = original.encode(text, add_special_tokens=False).ids
        got = rebuilt.encode(text).ids
        if got != expected:
            mismatches += 1
            print(f"  MISMATCH {text!r}:\n    source {expected[:12]}\n"
                  f"    rebuilt {got[:12]}")
    if mismatches:
        raise SystemExit(
            f"vocab.json/merges.txt parity FAILED on {mismatches} sample(s): "
            "the exported files are not consistent with the source; aborting"
        )

    shutil.copyfile(source_path, out_dir / "tokenizer.json")
    with (out_dir / "vocab.json").open("w", encoding="utf-8") as fh:
        json.dump(vocab, fh, ensure_ascii=False, indent=None)
    with (out_dir / "merges.txt").open("w", encoding="utf-8") as fh:
        fh.write("#version: hephaestus-export-1\n")
        for left, right in merges:
            fh.write(f"{left} {right}\n")
    with (out_dir / "specials.json").open("w", encoding="utf-8") as fh:
        json.dump(specials, fh, indent=2)

    print(f"exported tokenizer to {out_dir}: "
          f"vocab {len(vocab)}, merges {len(merges)}, specials {specials}")
    print(f"parity sample: {len(SAMPLE_STRINGS)} strings OK")
    print("byte-level pitfalls for the C++ runtime (documented in "
          "tests/golden/test_tokenize_parity.py): vocab keys are in the "
          "byte->unicode space ('\u0120'=space), the regex pre-split MUST run "
          "before merges, and encode must NOT inject special tokens.")


if __name__ == "__main__":
    main()
