"""Pure-NumPy reference implementation of the PROMETHEUS-NS nano decoder.

This module IS the numerical specification for the HEPHAESTUS C++
engine: the engine must reproduce its greedy continuations exactly and
its fp32 logits within the tolerances fixed in config.yaml
(golden.max_abs_diff, golden.cosine_min). Never edit this file to make
the engine pass; a divergence means the engine is wrong.

Semantics mirrored from the PROMETHEUS-NS model (prometheus_ns.model):

  - pre-norm blocks: x + attn(rms_norm(x)); x + ffn(rms_norm(x))
  - RMSNorm without mean subtraction: x * rsqrt(mean(x^2) + eps) * w
  - SwiGLU: down(silu(gate(x)) * up(x)), all projections stored as
    row-major [out, in] matrices applied as x @ W.T
  - RoPE half-split convention with tables [max_seq, d_head/2],
    inv_freq = 1 / theta^(2i/d_head)
  - causal attention scaled by 1/sqrt(d_head) over the per-layer cache
    (past + current); head h attends kv-head h // (n_head // n_kv_head)
    (identity for nano, where n_head == n_kv_head)
  - tied embeddings: lm_head.weight == embedding.weight, applied as
    logits = final_norm(x_last) @ embedding.T

GQA contract note: with n_kv_head < n_head the kv expansion used here
must change together with the engine's repeat_kv (documented in
src/model/transformer.h). The nano engine never takes that path.
"""

from __future__ import annotations

import json
import mmap
import re
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

# nano architecture contract (src/model/model_def.h); the manifest must
# agree with every value here or the oracle refuses to load.
NANO = {
    "n_layer": 6,
    "n_head": 6,
    "n_kv_head": 6,
    "d_model": 384,
    "d_ff": 1024,
    "max_seq": 256,
    "vocab": 8000,
}


@dataclass
class ModelConfig:
    n_layer: int
    n_head: int
    n_kv_head: int
    d_model: int
    d_ff: int
    max_seq: int
    vocab: int
    d_head: int
    rms_norm_eps: float
    rope_theta: float


class SafetensorsWeights:
    """Read-only NumPy view over a fp32 safetensors file."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._file = open(path, "rb")
        self._map = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
        (header_len,) = struct.unpack("<Q", self._map[:8])
        header = json.loads(self._map[8:8 + header_len])
        self.tensors: dict[str, np.ndarray] = {}
        for name, info in header.items():
            if name == "__metadata__":
                continue
            assert info["dtype"] == "F32", f"{name}: expected F32"
            start, end = info["data_offsets"]
            base = 8 + header_len
            raw = np.frombuffer(
                self._map, dtype="<f4", count=end - start, offset=base + start
            )
            self.tensors[name] = raw.reshape(info["shape"])

    def close(self) -> None:
        self._map.close()
        self._file.close()


def load_manifest(path: Path) -> dict:
    manifest: dict = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or "=" not in line:
            continue
        key, value = line.split("=", 1)
        manifest[key.strip()] = value.strip()
    arch = manifest["arch"]
    inner = dict(re.findall(r"'(\w+)':\s*([^,}]+)", arch))
    arch_dict = {
        key: (int(value) if key != "d_head" else (None if value == "None" else int(value)))
        for key, value in inner.items()
    }
    arch_dict["d_head"] = arch_dict["d_head"] or arch_dict["d_model"] // arch_dict["n_head"]
    return {
        "arch": arch_dict,
        "rms_norm_eps": float(manifest["rms_norm_eps"]),
        "rope_theta": float(manifest["rope_theta"]),
    }


def oracle_from_cfg(cfg: dict) -> tuple["ReferenceModel", SafetensorsWeights]:
    repo = Path(__file__).resolve().parents[2]
    manifest = load_manifest(repo / cfg["paths"]["model_manifest"])
    weights = SafetensorsWeights(repo / cfg["paths"]["weights"])
    model = ReferenceModel(manifest, weights)
    return model, weights


def rope_cache(max_seq: int, d_head: int, theta: float) -> tuple[np.ndarray, np.ndarray]:
    inv_freq = 1.0 / (theta ** (np.arange(0, d_head, 2, dtype=np.float32) / d_head))
    positions = np.arange(max_seq, dtype=np.float32)
    angles = np.outer(positions, inv_freq)
    return angles.cos(), angles.sin()


def apply_rope(x: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    """Rotate ``x`` of shape [seq, n_head, d_head]; cos/sin are [seq, d_head/2]."""
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return np.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)


def rms_norm(x: np.ndarray, weight: np.ndarray, eps: float) -> np.ndarray:
    norm = np.sqrt((x.astype(np.float32) ** 2).mean(axis=-1, keepdims=True) + eps)
    return x / norm * weight


def silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - x.max(axis=-1, keepdims=True)
    e = np.exp(shifted)
    return e / e.sum(axis=-1, keepdims=True)


class ReferenceModel:
    """NumPy oracle over the exported safetensors weights."""

    def __init__(self, manifest: dict, weights: SafetensorsWeights) -> None:
        arch = manifest["arch"]
        for key, expected in NANO.items():
            actual = arch[key]
            if actual != expected:
                raise SystemExit(
                    f"manifest {key}={actual} disagrees with the nano contract "
                    f"({expected}): the engine only supports the nano architecture"
                )
        self.m = ModelConfig(
            n_layer=arch["n_layer"],
            n_head=arch["n_head"],
            n_kv_head=arch["n_kv_head"],
            d_model=arch["d_model"],
            d_ff=arch["d_ff"],
            max_seq=arch["max_seq"],
            vocab=arch["vocab"],
            d_head=arch["d_head"],
            rms_norm_eps=manifest["rms_norm_eps"],
            rope_theta=manifest["rope_theta"],
        )
        self.w = weights

    def logits(
        self,
        tokens: list[int],
        pos_offset: int = 0,
        kv: list[tuple[np.ndarray, np.ndarray]] | None = None,
    ) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
        """Forward ``tokens`` starting at absolute position ``pos_offset``.

        ``kv`` carries one (k, v) pair per layer holding the cache of all
        tokens so far (past + current chunk, appended by each layer).
        Returns the logits of the LAST position only, shape [vocab].
        """
        m = self.m
        x = self.w.tensors["embedding.weight"][tokens].astype(np.float32)
        seq = len(tokens)
        cos, sin = rope_cache(m.max_seq, m.d_head, m.rope_theta)
        cos = cos[pos_offset:pos_offset + seq]
        sin = sin[pos_offset:pos_offset + seq]
        if kv is None:
            kv = []

        # causal mask over the columns [0, pos_offset + seq)
        mask = np.zeros((seq, pos_offset + seq), dtype=np.float32)
        for i in range(seq):
            mask[i, pos_offset + i + 1:] = -np.inf

        rep = m.n_head // m.n_kv_head
        for layer in range(m.n_layer):
            p = f"layer.{layer}."
            h = rms_norm(x, self.w.tensors[p + "attn_norm.weight"], m.rms_norm_eps)
            q = (h @ self.w.tensors[p + "attn.wq.weight"].T).reshape(seq, m.n_head, m.d_head)
            k = (h @ self.w.tensors[p + "attn.wk.weight"].T).reshape(seq, m.n_kv_head, m.d_head)
            v = (h @ self.w.tensors[p + "attn.wv.weight"].T).reshape(seq, m.n_kv_head, m.d_head)
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)
            kv.append((k, v))

            k_full, v_full = kv[layer]  # this layer's cache: all tokens so far
            # head h attends kv-head h // rep (identity for nano: rep == 1)
            k_tiled = np.repeat(k_full, rep, axis=1).transpose(1, 0, 2)  # [H, kv_seq, d]
            v_tiled = np.repeat(v_full, rep, axis=1).transpose(1, 0, 2)
            scores = np.einsum("qhd,hkd->hqk", q, k_tiled) / np.sqrt(m.d_head)
            scores = scores + mask[None, :, :]
            probs = softmax(scores)
            attn_out = np.einsum("hqk,hkd->qhd", probs, v_tiled)
            attn_out = attn_out.reshape(seq, m.n_head * m.d_head)
            x = x + attn_out @ self.w.tensors[p + "attn.wo.weight"].T

            h2 = rms_norm(x, self.w.tensors[p + "ffn_norm.weight"], m.rms_norm_eps)
            gate = silu(h2 @ self.w.tensors[p + "ffn.w_gate.weight"].T)
            up = h2 @ self.w.tensors[p + "ffn.w_up.weight"].T
            x = x + (gate * up) @ self.w.tensors[p + "ffn.w_down.weight"].T

        x_last = rms_norm(x[-1:], self.w.tensors["final_norm.weight"], m.rms_norm_eps)
        logits = x_last @ self.w.tensors["embedding.weight"].T
        return logits[0], kv

    def greedy(
        self, prompt_tokens: list[int], n_new: int, eos_id: int | None = None
    ) -> list[int]:
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
