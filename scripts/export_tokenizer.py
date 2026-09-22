#!/usr/bin/env python3
"""Export the PROMETHEUS-NS tokenizer to the HEPHAESTUS runtime format.

This is the ONLY tokenizer artifact producer. It reads the source
tokenizer.json (HF tokenizers format, byte-level BPE), extracts the
vocab, merges and special tokens, verifies that a rebuilt tokenizer
reproduces the source ids on every sample string (including the golden
prompts), and writes:

  artifacts/tokenizer/tokenizer.json   verbatim copy (HF oracle side)
  artifacts/tokenizer/vocab.json       token -> id (unicode space)
  artifacts/tokenizer/merges.txt       ranked merge pairs
  artifacts/tokenizer/specials.json    {bos, eos, pad, unk}

It fails loudly if the source is not byte-level BPE or if vocab/merges
parity with the source breaks on any sample.
"""

import argparse
import json
import shutil
from pathlib import Path

SAMPLE_STRINGS = [
    "The ancient mariner sails at dawn.",
    "Space, the final frontier: 42 light-years away!",
    "E = mc^2 describes mass-energy equivalence.",
    "naive cafe facade resume",
    "line one\nline two\ttabbed",
]

BPE_MAX_ELEMENT = 100_000_000


def load_source(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    model = data.get("model", {})
    if model.get("type") != "BPE":
        raise SystemExit(
            f"source tokenizer is {model.get('type')!r}, expected byte-level BPE: "
            "HEPHAESTUS only reads the PROMETHEUS-NS byte-level BPE"
        )
    pre_tokenizer = data.get("pre_tokenizer", {}) or {}
    if pre_tokenizer.get("type") != "ByteLevel":
        raise SystemExit(
            "source pre_tokenizer is not ByteLevel: the C++ runtime mirrors "
            "the GPT-2 byte-level pipeline only"
        )
    return data, model, pre_tokenizer


def parse_merges(model: dict) -> list[tuple[str, str]]:
    merges = model.get("merges", [])
    parsed = []
    for entry in merges:
        if isinstance(entry, list):
            if len(entry) != 2:
                raise SystemExit(f"unsupported merge entry: {entry!r}")
            left, right = entry
            if isinstance(left, list):  # new-style [pair, id] entries
                left, right = left
        elif isinstance(entry, str):
            parts = entry.split(" ")
            if len(parts) != 2:
                raise SystemExit(f"unsupported merge entry: {entry!r}")
            left, right = parts
        else:
            raise SystemExit(f"unsupported merge entry: {entry!r}")
        parsed.append((left, right))
    return parsed


def extract_specials(data: dict) -> dict:
    specials: dict[str, int] = {}
    added = data.get("added_tokens", []) or []
    for tok in added:
        content = tok.get("content", "")
        for name in ("bos", "eos", "pad", "unk"):
            marker = f"<|{name}|>"
            if content == marker:
                specials[name] = tok["id"]
    vocab = data.get("model", {}).get("vocab", {})
    for name in ("bos", "eos", "pad", "unk"):
        marker = f"<|{name}|>"
        if name not in specials and marker in vocab:
            specials[name] = vocab[marker]
    missing = [n for n in ("bos", "eos", "pad", "unk") if n not in specials]
    if missing:
        raise SystemExit(f"source tokenizer lacks special tokens: {missing}")
    return specials


def rebuild_tokenizer(model: dict, pre_tokenizer: dict, vocab: dict,
                      merges: list[tuple[str, str]]):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers

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

    from tokenizers import Tokenizer

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
          "byte->unicode space ('Ġ'=space), the regex pre-split MUST run "
          "before merges, and encode must NOT inject special tokens.")


if __name__ == "__main__":
    main()
