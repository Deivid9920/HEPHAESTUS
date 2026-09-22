#!/usr/bin/env python3
"""Export a PROMETHEUS-NS nano checkpoint to the HEPHAESTUS engine format.

Reads the checkpoint referenced by config.yaml paths.checkpoint, maps its
state_dict keys to the canonical engine tensor names using
scripts/export_mapping.yaml, verifies the tied-embedding contract and the
canonical name set, and writes:

  artifacts/nano_fp32.safetensors   row-major fp32 weights
  artifacts/tensors.tsv             name/shape/dtype/absolute offset/bytes
  artifacts/model_manifest.txt      arch + tolerances + weights sha256
  artifacts/tokenizer/EXPORT_NOTE.txt  pointer to scripts/export_tokenizer.py

The export fails loudly on unmapped keys, tied-embedding violations and
canonical-set mismatches: the mapping file is the thing to fix, never the
checkpoint.
"""

import argparse
import hashlib
import json
import re
import struct
from pathlib import Path

import torch
import yaml


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_mapping(path: Path) -> tuple[list[str], list[dict]]:
    with path.open(encoding="utf-8") as fh:
        doc = yaml.safe_load(fh)
    return doc.get("ignore_patterns", []), doc.get("rules", [])


def apply_rules(keys: list[str], ignores: list[str], rules: list[dict]) -> tuple[dict, list[str]]:
    ignore_res = [re.compile(p) for p in ignores]
    compiled = [(re.compile(r["pattern"]), r["target"]) for r in rules]
    mapped: dict[str, "torch.Tensor"] = {}
    unmapped: list[str] = []
    for key in keys:
        if any(rx.search(key) for rx in ignore_res):
            continue
        for rx, target in compiled:
            m = rx.match(key)
            if m:
                name = target.format(i=m.group(1)) if m.groups() else target
                if name in mapped:
                    raise SystemExit(f"collision: {key} -> {name} (already mapped)")
                mapped[name] = None  # filled by the caller with the tensor
                break
        else:
            unmapped.append(key)
    return mapped, unmapped


def infer_arch(mapped: dict) -> dict:
    layer_ids = sorted({
        int(m.group(1))
        for name in mapped
        if (m := re.match(r"^layer\.(\d+)\.", name))
    })
    n_layer = len(layer_ids)
    if layer_ids != list(range(n_layer)):
        raise SystemExit(f"layer indices are not contiguous: {layer_ids}")
    vocab, d_model = mapped["embedding.weight"].shape
    d_ff = mapped["layer.0.ffn.w_gate.weight"].shape[0]
    d_head = None
    return {
        "n_layer": n_layer,
        "n_head": 6,          # nano contract (src/model/model_def.h)
        "n_kv_head": 6,       # nano contract: GQA group size 1
        "d_model": d_model,
        "d_ff": d_ff,
        "max_seq": 256,       # nano contract
        "vocab": vocab,
        "d_head": d_head,
    }


def canonical_set(n_layer: int) -> set[str]:
    names = {"embedding.weight", "final_norm.weight"}
    for i in range(n_layer):
        for leaf in ("attn_norm", "attn.wq", "attn.wk", "attn.wv", "attn.wo",
                     "ffn_norm", "ffn.w_gate", "ffn.w_up", "ffn.w_down"):
            names.add(f"layer.{i}.{leaf}.weight")
    return names


def write_safetensors(mapped: dict, path: Path) -> int:
    header = {}
    offset = 0
    tensors = []
    for name in sorted(mapped):
        t = mapped[name].contiguous().to(torch.float32)
        n_bytes = t.numel() * 4
        header[name] = {
            "dtype": "F32",
            "shape": list(t.shape),
            "data_offsets": [offset, offset + n_bytes],
        }
        tensors.append((name, t, n_bytes))
        offset += n_bytes
    header["__metadata__"] = {"format": "pt"}
    blob = json.dumps(header, separators=(",", ":")).encode("utf-8")
    pad = (8 - (len(blob) % 8)) % 8
    blob += b" " * pad
    with path.open("wb") as fh:
        fh.write(struct.pack("<Q", len(blob)))
        fh.write(blob)
        for _, t, n_bytes in tensors:
            fh.write(t.view(torch.uint8).numpy().tobytes())
            del n_bytes
    return offset


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ckpt_path = cfg["paths"]["checkpoint"]
    if not ckpt_path:
        raise SystemExit(
            "config.yaml paths.checkpoint is empty: point it at the "
            "PROMETHEUS-NS nano .pt before running make export"
        )
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state", ckpt)

    repo = Path(args.config).resolve().parent
    ignores, rules = load_mapping(repo / "scripts/export_mapping.yaml")

    mapped_targets, unmapped = apply_rules(sorted(state.keys()), ignores, rules)
    if unmapped:
        raise SystemExit(
            "UNMAPPED KEYS — extend scripts/export_mapping.yaml until "
            "coverage is 100%:\n  " + "\n  ".join(sorted(unmapped))
        )

    # materialize tensors under their canonical names
    mapped = {}
    src_by_target = {}
    for key in state.keys():
        if key in unmapped:
            continue
        target = None
        for rx, tmpl in [(re.compile(r["pattern"]), r["target"]) for r in rules]:
            m = rx.match(key)
            if m:
                target = tmpl.format(i=m.group(1)) if m.groups() else tmpl
                break
        if target is not None:
            src_by_target[target] = key
    for target, key in src_by_target.items():
        mapped[target] = state[key]

    lm_head = mapped.pop("lm_head.weight", None)
    embedding = mapped["embedding.weight"]
    if lm_head is not None:
        assert torch.equal(lm_head, embedding), \
            "lm_head.weight differs from embedding: tied-embeddings violated"

    arch = infer_arch(mapped)
    expected = canonical_set(arch["n_layer"])
    missing = expected - set(mapped)
    extra = set(mapped) - expected
    if missing or extra:
        raise SystemExit(
            f"mapping validation failed.\nmissing: {sorted(missing)}\nextra: {sorted(extra)}"
        )

    out_dir = repo / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)
    weights_path = out_dir / "nano_fp32.safetensors"
    write_safetensors(mapped, weights_path)

    sha = hashlib.sha256(weights_path.read_bytes()).hexdigest()

    # tensors.tsv: absolute file offsets (header + 8-byte length prefix)
    header_len = 0
    with weights_path.open("rb") as fh:
        header_len = struct.unpack("<Q", fh.read(8))[0]
    blob = json.loads(weights_path.read_bytes()[8:8 + header_len])
    with (out_dir / "tensors.tsv").open("w", encoding="utf-8") as fh:
        for name in sorted(blob):
            if name == "__metadata__":
                continue
            info = blob[name]
            start, end = info["data_offsets"]
            shape = "x".join(map(str, info["shape"]))
            fh.write(f"{name}\t{shape}\tF32\t{8 + header_len + start}\t"
                     f"{end - start}\n")

    manifest = {
        "arch": arch,
        "rms_norm_eps": cfg["arch_defaults"]["rms_norm_eps"],
        "rope_theta": cfg["arch_defaults"]["rope_theta"],
        "weights_sha256": sha,
        "weights_file": str(weights_path.name),
        "tensors_file": "tensors.tsv",
        "d_head": arch.get("d_head"),
    }
    with (out_dir / "model_manifest.txt").open("w", encoding="utf-8") as fh:
        for key, value in manifest.items():
            fh.write(f"{key} = {value}\n")

    tok_dir = out_dir / "tokenizer"
    tok_dir.mkdir(exist_ok=True)
    note = (
        "Tokenizer artifacts are NOT produced here. Re-export them from "
        "the PROMETHEUS-NS training tokenizer once: tokenizers.Tokenizer."
        "from_file/pickle -> save(vocab.json, merges.txt) into "
        "artifacts/tokenizer/, plus specials.json {bos,eos,pad,unk}. "
        "The C++ BPE runtime and the NumPy oracle both read these files."
    )
    (tok_dir / "EXPORT_NOTE.txt").write_text(note, encoding="utf-8")

    print(f"exported {len(mapped)} tensors -> {weights_path}")
    print(f"arch: {arch}")
    print(f"sha256: {sha}")
    print(note)


if __name__ == "__main__":
    main()
