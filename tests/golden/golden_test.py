"""Golden tests: the C++ engine vs the NumPy oracle.

The engine must reproduce the oracle's greedy continuations EXACTLY
(determinism is the test) and its fp32 last-position logits within the
config tolerances (cosine >= golden.cosine_min, max abs diff <=
golden.max_abs_diff) over the full vocabulary.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from reference_model import ReferenceModel, oracle_from_cfg

REPO = Path(__file__).resolve().parents[2]
HEPH = REPO / "build" / "heph"


@pytest.fixture(scope="session")
def cfg() -> dict:
    with (REPO / "config.yaml").open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@pytest.fixture(scope="session")
def oracle(cfg: dict):
    model, weights = oracle_from_cfg(cfg)
    yield model
    weights.close()


def prompt_lines(cfg: dict) -> list[str]:
    return [
        line
        for line in (REPO / cfg["golden"]["prompts_file"])
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip() and not line.startswith("#")
    ]


def encode_prompts(cfg: dict) -> list[list[int]]:
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(
        str(REPO / cfg["paths"]["tokenizer_dir"] / "tokenizer.json")
    )
    return [tok.encode(line).ids for line in prompt_lines(cfg)]


def run_engine(args: list[str]) -> None:
    result = subprocess.run(
        [str(HEPH), *args], capture_output=True, text=True, timeout=600
    )
    assert result.returncode == 0, (
        f"engine failed ({result.returncode}):\n{result.stderr}"
    )


def engine_greedy_ids(cfg: dict, prompt: str) -> list[int]:
    out = REPO / cfg["golden"]["fixtures_dir"] / "greedy_ids.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    run_engine([
        "generate",
        "--manifest", str(REPO / cfg["paths"]["model_manifest"]),
        "--weights", str(REPO / cfg["paths"]["weights"]),
        "--tokenizer", str(REPO / cfg["paths"]["tokenizer_dir"]),
        "--prompt", prompt,
        "--greedy",
        "--max-new-tokens", str(cfg["golden"]["tokens_per_prompt"]),
        "--out", str(out),
    ])
    return [int(x) for x in out.read_text(encoding="utf-8").split()]


def test_engine_binary_built() -> None:
    assert HEPH.is_file(), "build/heph missing: run make build"


def test_greedy_exact_match(cfg: dict, oracle: ReferenceModel) -> None:
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(
        str(REPO / cfg["paths"]["tokenizer_dir"] / "tokenizer.json")
    )
    prompts = prompt_lines(cfg)
    for prompt in prompts:
        ids = tok.encode(prompt).ids
        expected = oracle.greedy(ids, cfg["golden"]["tokens_per_prompt"])
        got = engine_greedy_ids(cfg, prompt)
        assert got == expected, (
            f"greedy mismatch on prompt {prompt!r}:\n"
            f"first divergence at index "
            f"{next((i for i, (a, b) in enumerate(zip(expected, got)) if a != b), min(len(expected), len(got)))}"
        )


def test_logits_tolerance(cfg: dict, oracle: ReferenceModel,
                          tmp_path: Path) -> None:
    prompts = encode_prompts(cfg)
    dump = tmp_path / "logits.bin"
    run_engine([
        "golden",
        "--manifest", str(REPO / cfg["paths"]["model_manifest"]),
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


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
