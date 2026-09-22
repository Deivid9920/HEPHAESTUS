"""Tokenization parity: the C++ BPE runtime vs Hugging Face tokenizers.

Collected automatically by pytest alongside golden_test.py; SKIPS (not
fails) before artifacts and build exist. Byte-level BPE covers all 256
bytes, so unk must never fire: a fired unk means the byte->unicode map
is broken.
"""

import json
import subprocess
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
HEPH = REPO / "build" / "heph"

EXOTIC = ["éèê", "Ω≈ç√∫", "fiflﬃ ligatures", "𝕌𝕟𝕚𝕔𝕠𝕕𝕖"]


def _config() -> dict:
    with (REPO / "config.yaml").open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _tokenizer_dir(cfg: dict) -> Path:
    return REPO / cfg["paths"]["tokenizer_dir"]


def _engine_tokenize(tok_dir: Path, texts: list[str], tmp_path: Path) -> list[list[int]]:
    inp = tmp_path / "lines.txt"
    out = tmp_path / "ids.txt"
    inp.write_text("\n".join(texts) + "\n", encoding="utf-8")
    result = subprocess.run(
        [str(HEPH), "tokenize", "--tokenizer", str(tok_dir),
         "--file", str(inp), "--out", str(out)],
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr
    lines = out.read_text(encoding="utf-8").rstrip("\n").split("\n")
    return [[int(x) for x in line.split()] for line in lines]


def test_engine_tokenize_matches_hf(tmp_path: Path) -> None:
    cfg = _config()
    tok_dir = _tokenizer_dir(cfg)
    if not (tok_dir / "tokenizer.json").is_file() or not HEPH.is_file():
        pytest.skip("artifacts or engine not ready")
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(tok_dir / "tokenizer.json"))
    texts = ["The ancient mariner sails at dawn.",
             "Space, the final frontier: 42 light-years away!",
             "E = mc^2 describes mass-energy equivalence."]
    ids = _engine_tokenize(tok_dir, texts, tmp_path)
    for got, text in zip(ids, texts):
        expected = tok.encode(text, add_special_tokens=False).ids
        assert got == expected, (
            f"tokenize mismatch on {text!r}\n"
            f"  engine  ({len(got)}): {got[:12]}\n"
            f"  hf      ({len(expected)}): {expected[:12]}"
        )


def test_unknown_bytes_never_map_to_unk(tmp_path: Path) -> None:
    """Byte-level BPE covers all 256 bytes: unk must never fire."""
    cfg = _config()
    tok_dir = _tokenizer_dir(cfg)
    if not (tok_dir / "tokenizer.json").is_file() or not HEPH.is_file():
        pytest.skip("artifacts or engine not ready")
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(tok_dir / "tokenizer.json"))
    unk_id = json.loads((tok_dir / "specials.json").read_text("utf-8"))["unk"]

    inp = tmp_path / "exotic.txt"
    out = tmp_path / "ids.txt"
    inp.write_text("\n".join(EXOTIC) + "\n", encoding="utf-8")
    result = subprocess.run(
        [str(HEPH), "tokenize", "--tokenizer", str(tok_dir),
         "--file", str(inp), "--out", str(out)],
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr
    for line, text in zip(
            out.read_text(encoding="utf-8").rstrip("\n").split("\n"), EXOTIC):
        ids = [int(x) for x in line.split()]
        assert unk_id not in ids, f"unk fired for {text!r}: byte mapping broken"
        assert ids == tok.encode(text, add_special_tokens=False).ids
