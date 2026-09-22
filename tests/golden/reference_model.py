"""Pure-NumPy reference implementation of the PROMETHEUS-NS nano decoder.

This file IS the numerical specification for HEPHAESTUS: the C++ engine
must reproduce its greedy continuations EXACTLY and its logits within
the config tolerances. Never edit the oracle to make the engine pass.

Conventions mirrored from PROMETHEUS-NS (prometheus_ns/model/blocks.py):
  - RMSNorm without mean subtraction: x * rsqrt(mean(x^2) + eps) * weight
  - RoPE half-split: the first half of each head rotates against the
    second half; tables are [max_seq, d_head/2].
  - SwiGLU: w_down(silu(w_gate(x)) * w_up(x))
  - Pre-norm residual blocks; final norm; tied lm_head (embedding.T).

GQA note (handoff warning): with the nano profile n_head == n_kv_head,
so the per-kv-head einsum is direct. When migrating to architectures
with real GQA (n_kv_head < n_head), this oracle and the engine must
update the repeat_kv expansion at the same time — the pairing is
documented in the src/model/transformer.h contract.
"""

from __future__ import annotations

import ast
import json
import math
import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]


def rope_cache(max_seq: int, d_head: int, theta: float
               ) -> tuple[np.ndarray, np.ndarray]:
    inv_freq = 1.0 / (theta ** (np.arange(0, d_head, 2, dtype=np.float32)
                                / np.float32(d_head)))
    positions = np.arange(max_seq, dtype=np.float32)
    angles = np.outer(positions, inv_freq)
    return angles.cos().astype(np.float32), angles.sin().astype(np.float32)


def apply_rope(x: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    """x: [seq, heads, d_head]; cos/sin: [seq, d_head/2] (half-split)."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return np.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos],
                          axis=-1).astype(np.float32)


def rms_norm(x: np.ndarray, weight: np.ndarray, eps: float) -> np.ndarray:
    norm = np.mean(x.astype(np.float32) ** 2, axis=-1, keepdims=True)
    return (x * (1.0 / np.sqrt(norm + eps)) * weight).astype(np.float32)


def silu(x: np.ndarray) -> np.ndarray:
    return (x / (1.0 + np.exp(-x))).astype(np.float32)


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return (e / np.sum(e, axis=-1, keepdims=True)).astype(np.float32)


@dataclass
class ModelMeta:
    d_model: int
    n_layer: int
    n_head: int
    n_kv_head: int
    d_head: int
    d_ff: int
    vocab: int
    max_seq: int
    rms_norm_eps: float
    rope_theta: float


class TensorStore:
    """Read-only view over the exported safetensors file (fp32)."""

    def __init__(self, weights_path: Path) -> None:
        self._path = Path(weights_path)
        raw = self._path.read_bytes()
        (header_len,) = struct.unpack("<Q", raw[:8])
        header = json.loads(raw[8:8 + header_len])
        base = 8 + header_len
        self.tensors: dict[str, np.ndarray] = {}
        for name, info in header.items():
            if name == "__metadata__":
                continue
            start, end = info["data_offsets"]
            shape = tuple(int(s) for s in info["shape"])
            arr = np.frombuffer(raw, dtype="<f4", count=end - start,
                                offset=base + start).reshape(shape)
            self.tensors[name] = arr

    def close(self) -> None:
        self.tensors = {}


@dataclass
class ReferenceModel:
    m: ModelMeta
    w: TensorStore
    _rope: tuple = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self._rope = rope_cache(self.m.max_seq, self.m.d_head,
                                self.m.rope_theta)

    def logits(self,
               tokens: list[int],
               pos_offset: int = 0,
               kv: list[tuple[np.ndarray, np.ndarray]] | None = None
               ) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
        """Full forward for `tokens` starting at absolute position
        pos_offset. kv is the layer cache list updated in place-style
        (returned). Returns logits of the LAST position only, shape
        [vocab]."""
        m = self.m
        x = self.w.tensors["embedding.weight"][tokens].astype(np.float32)
        seq = len(tokens)
        cos, sin = self._rope
        cos, sin = cos[pos_offset:pos_offset + seq], sin[pos_offset:pos_offset + seq]

        # causal mask over [seq, pos_offset + seq]
        mask = np.full((seq, pos_offset + seq), -np.inf, dtype=np.float32)
        for i in range(seq):
            mask[i, :pos_offset + i + 1] = 0.0
            mask[i, pos_offset + i + 1:] = -np.inf

        kv = list(kv) if kv is not None else []
        for layer in range(m.n_layer):
            p = f"layer.{layer}."
            h = rms_norm(x, self.w.tensors[p + "attn_norm.weight"],
                         self.m.rms_norm_eps)
            q = (h @ self.w.tensors[p + "attn.wq.weight"].T).reshape(
                seq, m.n_head, m.d_head)
            k = (h @ self.w.tensors[p + "attn.wk.weight"].T).reshape(
                seq, m.n_kv_head, m.d_head)
            v = (h @ self.w.tensors[p + "attn.wv.weight"].T).reshape(
                seq, m.n_kv_head, m.d_head)
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)

            # this layer's cache: all tokens so far (read + extend)
            if layer < len(kv):
                prev_k, prev_v = kv[layer]
                k_full = np.concatenate([prev_k, k], axis=0).astype(np.float32)
                v_full = np.concatenate([prev_v, v], axis=0).astype(np.float32)
            else:
                k_full, v_full = k, v
            if layer < len(kv):
                kv[layer] = (k_full, v_full)
            else:
                kv.append((k_full, v_full))

            # attention per kv-head; GQA maps n_head//n_kv_head q-heads
            # per kv-head (repeat_kv — see module docstring).
            rep = m.n_head // m.n_kv_head
            k_rep = np.repeat(k_full, rep, axis=1) if rep > 1 else k_full
            v_rep = np.repeat(v_full, rep, axis=1) if rep > 1 else v_full
            scores = np.einsum("qhd,khd->hqk", q, k_rep) / np.sqrt(m.d_head)
            scores = scores + mask[None, :, :]
            probs = softmax(scores)
            attn_out = np.einsum("hqk,khd->qhd", probs, v_rep)
            attn_out = attn_out.reshape(seq, m.n_head * m.d_head)
            x = x + attn_out @ self.w.tensors[p + "attn.wo.weight"].T

            h2 = rms_norm(x, self.w.tensors[p + "ffn_norm.weight"],
                          self.m.rms_norm_eps)
            gate = silu(h2 @ self.w.tensors[p + "ffn.w_gate.weight"].T)
            up = h2 @ self.w.tensors[p + "ffn.w_up.weight"].T
            x = x + (gate * up) @ self.w.tensors[p + "ffn.w_down.weight"].T

        x_last = rms_norm(x[-1:], self.w.tensors["final_norm.weight"],
                          self.m.rms_norm_eps)
        logits = x_last @ self.w.tensors["embedding.weight"].T
        return logits[0], kv

    def greedy(self, prompt_tokens: list[int], n_new: int,
               eos_id: int | None = None) -> list[int]:
        out: list[int] = []
        kv: list[tuple[np.ndarray, np.ndarray]] = []
        tokens = list(prompt_tokens)
        logits, kv = self.logits(tokens, 0, kv)
        for _ in range(n_new):
            nxt = int(np.argmax(logits))
            if eos_id is not None and nxt == eos_id:
                break
            out.append(nxt)
            logits, kv = self.logits([nxt], pos_offset=len(tokens), kv=kv)
            tokens.append(nxt)
        return out


def parse_manifest(path: Path) -> dict:
    manifest: dict = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip() or "=" not in line:
            continue
        key, value = line.split("=", 1)
        manifest[key.strip()] = value.strip()
    return manifest


def oracle_from_cfg(cfg: dict) -> tuple[ReferenceModel, TensorStore]:
    """Build the oracle from config.yaml paths (manifest + weights)."""
    manifest = parse_manifest(REPO / cfg["paths"]["model_manifest"])
    arch = ast.literal_eval(manifest["arch"])
    meta = ModelMeta(
        d_model=int(arch["d_model"]),
        n_layer=int(arch["n_layer"]),
        n_head=int(arch["n_head"]),
        n_kv_head=int(arch["n_kv_head"]),
        d_head=int(manifest.get("d_head", arch.get("d_head"))),
        d_ff=int(arch["d_ff"]),
        vocab=int(arch["vocab"]),
        max_seq=int(arch["max_seq"]),
        rms_norm_eps=float(manifest["rms_norm_eps"]),
        rope_theta=float(manifest["rope_theta"]),
    )
    store = TensorStore(REPO / cfg["paths"]["weights"])
    return ReferenceModel(m=meta, w=store), store
