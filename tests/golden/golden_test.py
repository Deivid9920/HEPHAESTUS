"""Golden tests: the C++ engine vs the NumPy oracle.

The oracle (reference_model.py) is the numerical specification. The
engine must match its greedy continuations EXACTLY and its last-position
logits within the config tolerances (cosine_min, max_abs_diff) over the
full vocabulary. Never relax a tolerance to make a failing engine pass.

Run via `make test` / `make golden` (the engine binary must be built).
"""

from __future__ import annotations

import hashlib
import json
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reference_model import oracle_from_cfg  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
HEPH = REPO / "build" / "heph"


@pytest.fixture(scope="session")
def cfg() -> dict:
    return yaml.safe_load((REPO / "config.yaml").read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def oracle(cfg: dict):
    model, weights = oracle_from_cfg(cfg)
    yield model
    weights.close()


def run_engine(args: list[str], timeout: int = 600) -> subprocess.CompletedProcess:
    result = subprocess.run(
        [str(HEPH), *args], capture_output=True, text=True, timeout=timeout
    )
    assert result.returncode == 0, (
        f"engine failed ({result.returncode}):\n{result.stderr}"
    )
    return result


def encode_prompts(cfg: dict) -> list[list[int]]:
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(
        str(REPO / cfg["paths"]["tokenizer_dir"] / "tokenizer.json")
    )
    lines = [
        l for l in (REPO / cfg["golden"]["prompts_file"])
        .read_text(encoding="utf-8").splitlines()
        if l.strip() and not l.startswith("#")
    ]
    return [tok.encode(line).ids for line in lines]


def engine_greedy_ids(cfg: dict, prompt: str) -> list[int]:
    result = run_engine([
        "generate",
        "--manifest", str(REPO / cfg["paths"]["model_manifest"]),
        "--weights", str(REPO / cfg["paths"]["weights"]),
        "--tokenizer", str(REPO / cfg["paths"]["tokenizer_dir"]),
        "--prompt", prompt,
        "--max-new-tokens", str(cfg["golden"]["tokens_per_prompt"]),
        "--greedy", "--ids",
    ])
    return [int(x) for x in result.stdout.split()]


def test_engine_binary_built() -> None:
    assert HEPH.is_file(), "build/heph missing: run make build"


def test_greedy_exact_match(cfg: dict, oracle) -> None:
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(
        str(REPO / cfg["paths"]["tokenizer_dir"] / "tokenizer.json")
    )
    prompts = [l for l in (REPO / cfg["golden"]["prompts_file"])
               .read_text(encoding="utf-8").splitlines()
               if l.strip() and not l.startswith("#")]
    for prompt in prompts:
        ids = tok.encode(prompt).ids
        expected = oracle.greedy(ids, cfg["golden"]["tokens_per_prompt"])
        got = engine_greedy_ids(cfg, prompt)
        assert got == expected, (
            f"greedy mismatch on prompt {prompt!r}:\n"
            f"first divergence at index "
            f"{next((i for i, (a, b) in enumerate(zip(expected, got)) if a != b), min(len(expected), len(got)))}"
        )


def test_logits_tolerance(cfg: dict, oracle, tmp_path: Path) -> None:
    prompts = encode_prompts(cfg)
    dump = tmp_path / "logits.bin"
    run_engine([
        "golden", "--manifest", str(REPO / cfg["paths"]["model_manifest"]),
        "--weights", str(REPO / cfg["paths"]["weights"]),
        "--tokenizer", str(REPO / cfg["paths"]["tokenizer_dir"]),
        "--prompts", str(REPO / cfg["golden"]["prompts_file"]),
        "--tokens", str(cfg["golden"]["tokens_per_prompt"]),
        "--dump", str(dump),
    ])
    raw = dump.read_bytes()
    n_prompts = len(prompts)
    vocab = oracle.m.vocab
    engine_logits = np.frombuffer(raw[: n_prompts * vocab * 4], dtype="<f4") \
        .reshape(n_prompts, vocab)

    max_diff, min_cos = 0.0, 1.0
    for i, ids in enumerate(prompts):
        ref_logits, _ = oracle.logits(ids)
        diff = float(np.max(np.abs(engine_logits[i] - ref_logits)))
        cos = float(np.dot(engine_logits[i], ref_logits) /
                    (np.linalg.norm(engine_logits[i]) *
                     np.linalg.norm(ref_logits)))
        max_diff, min_cos = max(max_diff, diff), min(min_cos, cos)
    assert max_diff <= cfg["golden"]["max_abs_diff"], f"max abs diff {max_diff}"
    assert min_cos >= cfg["golden"]["cosine_min"], f"min cosine {min_cos}"


def test_tensor_digests(cfg: dict, tmp_path: Path) -> None:
    """FASE 1 VERIF: every tensor the loader maps must be the exact byte
    range recorded in tensors.tsv — the engine's per-tensor sha256 must
    match an independent digest of the same slice of the weights file."""
    sha_path = tmp_path / "tensor_sha.txt"
    run_engine([
        "golden", "--manifest", str(REPO / cfg["paths"]["model_manifest"]),
        "--weights", str(REPO / cfg["paths"]["weights"]),
        "--tokenizer", str(REPO / cfg["paths"]["tokenizer_dir"]),
        "--prompts", str(REPO / cfg["golden"]["prompts_file"]),
        "--tokens", str(cfg["golden"]["tokens_per_prompt"]),
        "--dump-sha", str(sha_path),
    ])
    engine_sha = {}
    for line in sha_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, name = line.split()[:2]
        engine_sha[name] = digest

    weights_path = REPO / cfg["paths"]["weights"]
    raw = weights_path.read_bytes()
    (header_len,) = struct.unpack("<Q", raw[:8])
    header = json.loads(raw[8:8 + header_len])
    base = 8 + header_len
    assert set(header) - {"__metadata__"} == set(engine_sha), \
        "engine tensor set differs from safetensors header"
    for name, info in header.items():
        if name == "__metadata__":
            continue
        start, end = info["data_offsets"]
        expect = hashlib.sha256(raw[base + start: base + end]).hexdigest()
        assert engine_sha[name] == expect, f"digest mismatch for {name}"
