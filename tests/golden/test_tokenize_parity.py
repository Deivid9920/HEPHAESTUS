"""Parity tests for the C++ byte-level BPE runtime against HuggingFace.

Pitfalls these tests enforce (also printed by scripts/export_tokenizer.py):
  (1) the GPT-2 byte->unicode alphabet map must run BEFORE the vocab
      lookup ('G\u0300'-style entries; 'Ġ' = U+0120 encodes byte 0x20);
  (2) the ByteLevel regex pre-split MUST run before merges, with the
      same Unicode letter/number classification as HF;
  (3) encode must NOT inject special tokens nor ever emit <|unk|> —
      byte-level BPE covers all 256 bytes, so unk firing means the byte
      map is broken.

Both tests SKIP (not fail) before the tokenizer artifacts or the engine
binary exist, so the skeleton phase stays clean.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
HEPH = REPO / "build" / "heph"


def _config() -> dict:
    import yaml

    return yaml.safe_load((REPO / "config.yaml").read_text(encoding="utf-8"))


def _ready(cfg: dict) -> bool:
    tok_dir = REPO / cfg["paths"]["tokenizer_dir"]
    return (tok_dir / "tokenizer.json").is_file() and HEPH.is_file()


def _engine_ids(tok_dir: Path, texts: list[str], tmp_path: Path) -> list[list[int]]:
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


def test_prompts_parity(tmp_path: Path) -> None:
    cfg = _config()
    if not _ready(cfg):
        pytest.skip("artifacts or engine not ready")
    from tokenizers import Tokenizer

    tok_dir = REPO / cfg["paths"]["tokenizer_dir"]
    tok = Tokenizer.from_file(str(tok_dir / "tokenizer.json"))
    texts = [l for l in (REPO / cfg["golden"]["prompts_file"])
             .read_text(encoding="utf-8").splitlines()
             if l.strip() and not l.startswith("#")]
    ids_all = _engine_ids(tok_dir, texts, tmp_path)
    for text, ids in zip(texts, ids_all):
        expected = tok.encode(text, add_special_tokens=False).ids
        assert ids == expected, (
            f"tokenize mismatch on {text!r}\n"
            f"  engine ({len(ids)}): {ids[:12]}\n"
            f"  hf     ({len(expected)}): {expected[:12]}"
        )


def test_unknown_bytes_never_map_to_unk(tmp_path: Path) -> None:
    """Byte-level BPE covers all 256 bytes: unk must never fire."""
    cfg = _config()
    if not _ready(cfg):
        pytest.skip("artifacts or engine not ready")
    from tokenizers import Tokenizer

    tok_dir = REPO / cfg["paths"]["tokenizer_dir"]
    tok = Tokenizer.from_file(str(tok_dir / "tokenizer.json"))
    unk_id = json.loads((tok_dir / "specials.json").read_text("utf-8"))["unk"]

    exotic = ["\u00e9\u00e8\u00ea", "Ω≈ç√∫", "fiflﬃ ligatures", "𝕌𝕟𝕚𝕔𝕠𝕕𝕖"]
    ids_all = _engine_ids(tok_dir, exotic, tmp_path)
    for text, ids in zip(exotic, ids_all):
        assert unk_id not in ids, f"unk fired for {text!r}: byte mapping broken"
        assert ids == tok.encode(text, add_special_tokens=False).ids, (
            f"tokenize mismatch on {text!r}"
        )
