#!/usr/bin/env python3
"""Export a PROMETHEUS-NS checkpoint into engine-side artifacts.

Pipeline:
  checkpoint .pt --map--> canonical tensors --write--> nano_fp32.safetensors
                                                       tensors.tsv (offsets)
                                                       model_manifest.txt
                                                       tokenizer EXPORT_NOTE

The mapping rules live in scripts/export_mapping.yaml; the export fails
loudly listing every unmapped key, and the rules (never the checkpoint)
must be extended until coverage is exactly 100%.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
from pathlib import Path

import torch
import yaml

REPO = Path(__file__).resolve().parents[1]


def load_state_dict(path: Path) -> tuple[dict, dict]:
    """Return (state_dict, metadata) tolerating wrapped checkpoints."""
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    meta = {}
    if isinstance(ckpt, dict) and not any(
        hasattr(v, "shape") for v in ckpt.values()
    ):
        # Wrapped checkpoint: {"model_state": ..., "step": ..., ...}
        for key in ("model_state", "model", "state_dict", "model_state_dict"):
            if key in ckpt and isinstance(ckpt[key], dict):
                sd = ckpt[key]
                meta = {
                    k: v
                    for k, v in ckpt.items()
                    if not isinstance(v, dict) or k in ("profile",)
                }
                break
        else:
            raise SystemExit(
                f"checkpoint {path} has no model state dict; keys: "
                f"{sorted(ckpt.keys())}"
            )
    else:
        sd = ckpt
    return sd, meta


def apply_rules(
    sd: dict, ignore_patterns: list[str], rules: list[dict]
) -> tuple[dict, list[str]]:
    """Map source keys to canonical names; first matching rule wins."""
    ignores = [re.compile(p) for p in ignore_patterns]
    compiled = [(re.compile(r["pattern"]), r["target"]) for r in rules]

    mapped: dict[str, torch.Tensor] = {}
    unmapped: list[str] = []
    for key, tensor in sd.items():
        if any(ig.search(key) for ig in ignores):
            continue
        target = None
        for pattern, template in compiled:
            m = pattern.match(key)
            if m:
                target = template.format(i=m.group(1)) if m.groups() else template
                break
        if target is None:
            unmapped.append(key)
            continue
        if target in mapped:
            raise SystemExit(f"collision: {target} mapped twice")
        mapped[target] = tensor.detach().to(torch.float32).contiguous()
    return mapped, unmapped


def infer_arch(mapped: dict, meta: dict) -> dict:
    """Derive the architecture from the checkpoint profile, cross-checked
    against tensor shapes (shapes win on any conflict)."""
    prof = meta.get("profile") or {}
    emb = mapped["embedding.weight"]
    vocab, d_model = int(emb.shape[0]), int(emb.shape[1])

    layer_idx = set()
    for name in mapped:
        m = re.match(r"^layer\.(\d+)\.", name)
        if m:
            layer_idx.add(int(m.group(1)))
    n_layer = len(layer_idx) or int(prof.get("n_layer", 0))
    if sorted(layer_idx) != list(range(n_layer)):
        raise SystemExit(f"layer indices not contiguous: {sorted(layer_idx)}")

    wk = mapped["layer.0.attn.wk.weight"]
    d_head = int(prof.get("d_head") or 0) or d_model // int(prof.get("n_head", 0) or 1)
    n_kv_head = int(wk.shape[0]) // d_head if d_head else int(prof.get("n_kv_head", 0))
    n_head = int(mapped["layer.0.attn.wq.weight"].shape[0]) // d_head if d_head else 0
    d_ff = int(mapped["layer.0.ffn.w_down.weight"].shape[1])

    arch = {
        "d_model": d_model,
        "n_layer": n_layer,
        "n_head": n_head or int(prof.get("n_head", 0)),
        "n_kv_head": n_kv_head or int(prof.get("n_kv_head", 0)),
        "d_head": d_head,
        "d_ff": d_ff,
        "vocab": vocab,
        "max_seq": int(prof.get("max_seq", 256)),
        "tie_embeddings": bool(prof.get("tie_embeddings", True)),
    }
    for field in ("n_head", "n_kv_head"):
        if prof and arch[field] != int(prof.get(field, arch[field])):
            raise SystemExit(
                f"shape-derived {field}={arch[field]} contradicts checkpoint "
                f"profile {prof.get(field)}"
            )
    return arch


def canonical_set(n_layer: int) -> set[str]:
    names = {"embedding.weight", "final_norm.weight"}
    for i in range(n_layer):
        for suffix in (
            "attn_norm.weight",
            "attn.wq.weight",
            "attn.wk.weight",
            "attn.wv.weight",
            "attn.wo.weight",
            "ffn_norm.weight",
            "ffn.w_gate.weight",
            "ffn.w_up.weight",
            "ffn.w_down.weight",
        ):
            names.add(f"layer.{i}.{suffix}")
    return names


def write_safetensors(mapped: dict, path: Path) -> int:
    """Write the safetensors container; returns the data length."""
    header: dict[str, object] = {}
    blobs: list[bytes] = []
    offset = 0
    for name in sorted(mapped):
        arr = mapped[name].numpy()
        nbytes = arr.nbytes
        header[name] = {
            "dtype": "F32",
            "shape": list(arr.shape),
            "data_offsets": [offset, offset + nbytes],
        }
        blobs.append(arr.tobytes())
        offset += nbytes
    header["__metadata__"] = {"format": "pt"}

    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    pad = (8 - len(header_bytes) % 8) % 8  # keep data 8-byte aligned
    header_bytes += b" " * pad
    with path.open("wb") as fh:
        fh.write(struct.pack("<Q", len(header_bytes)))
        fh.write(header_bytes)
        for blob in blobs:
            fh.write(blob)
    return offset


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    cfg = yaml.safe_load((REPO / args.config).read_text(encoding="utf-8"))
    ckpt_path = cfg["paths"].get("checkpoint") or ""
    if not ckpt_path:
        raise SystemExit(
            "set config.yaml -> paths.checkpoint to the PROMETHEUS-NS .pt "
            "before export"
        )
    sd, meta = load_state_dict(REPO / ckpt_path)

    mapping = yaml.safe_load(
        (REPO / "scripts/export_mapping.yaml").read_text(encoding="utf-8")
    )
    mapped, unmapped = apply_rules(
        sd, mapping["ignore_patterns"], mapping["rules"]
    )
    if unmapped:
        raise SystemExit(
            "UNMAPPED KEYS — extend scripts/export_mapping.yaml until "
            "coverage is 100%:\n  " + "\n  ".join(sorted(unmapped))
        )

    lm_head = mapped.pop("lm_head.weight", None)
    embedding = mapped["embedding.weight"]
    if lm_head is not None:
        assert torch.equal(lm_head, embedding), \
            "lm_head.weight differs from embedding: tied-embeddings violated"

    arch = infer_arch(mapped, meta)
    expected = canonical_set(arch["n_layer"])
    missing = expected - set(mapped)
    extra = set(mapped) - expected
    if missing or extra:
        raise SystemExit(
            f"mapping validation failed.\nmissing: {sorted(missing)}\n"
            f"extra: {sorted(extra)}"
        )

    out_dir = Path("artifacts")
    out_dir.mkdir(parents=True, exist_ok=True)
    weights_path = out_dir / "nano_fp32.safetensors"
    data_len = write_safetensors(mapped, weights_path)

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
