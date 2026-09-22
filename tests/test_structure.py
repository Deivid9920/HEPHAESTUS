"""Structure contract for HEPHAESTUS.

This file is an immutable anchor from the specification package: it
verifies the repository layout, the configuration contract, the build
files and the canonical tensor names before any engine work starts.
"""

import re
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]

ANCHOR_FILES = [
    "config.yaml",
    "requirements.txt",
    ".gitignore",
    "Dockerfile",
    "README.md",
    "Makefile",
    "CMakeLists.txt",
    "scripts/export_mapping.yaml",
    "scripts/export_artifacts.py",
    "scripts/export_tokenizer.py",
    "scripts/render_bench.py",
    "scripts/scaffold.py",
    "tests/golden/reference_model.py",
    "tests/golden/golden_test.py",
    "tests/golden/test_tokenize_parity.py",
    "tests/golden/prompts.txt",
    "src/model/model_def.h",
    "docs/benchmark.md",
]

PATH_KEYS = [
    "model_manifest",
    "tensors_table",
    "weights",
    "tokenizer_dir",
    "checkpoint",
]

GOLDEN_KEYS = [
    "prompts_file",
    "fixtures_dir",
    "tokens_per_prompt",
    "max_abs_diff",
    "cosine_min",
]

KV_KEYS = ["page_tokens", "quant"]

BENCH_KEYS = ["prompts", "warmup", "max_new_tokens", "modes"]

MAKEFILE_TARGETS = [
    "setup",
    "export",
    "configure",
    "build",
    "test",
    "golden",
    "quantize",
    "bench",
    "bench-all",
    "chat",
    "lint",
    "docker-build",
    "clean",
]

# The canonical engine tensor names for one transformer layer; the test
# expands them over the six nano layers against src/model/model_def.h.
CANONICAL_TENSORS = [
    "layer.{i}.attn_norm.weight",
    "layer.{i}.attn.wq.weight",
    "layer.{i}.attn.wk.weight",
    "layer.{i}.attn.wv.weight",
    "layer.{i}.attn.wo.weight",
    "layer.{i}.ffn_norm.weight",
    "layer.{i}.ffn.w_gate.weight",
    "layer.{i}.ffn.w_up.weight",
    "layer.{i}.ffn.w_down.weight",
]

GLOBAL_TENSORS = ["embedding.weight", "final_norm.weight", "lm_head.weight"]

FORBIDDEN_MARKERS = [
    "as an ai",
    "as a language model",
    "i cannot",
    "i'm sorry",
    "co-authored-by",
    "assistant:",
    "chatgpt",
    "openai",
    "claude",
    "anthropic",
    "glm",
    "z.ai",
    "worklog",
]

SCAN_EXTENSIONS = {".py", ".yaml", ".yml", ".md", ".txt", ".cpp", ".h"}
SCAN_BASENAMES = {"Makefile", "Dockerfile", "CMakeLists.txt"}
# tests are excluded from the marker scan: test sources legitimately quote
# the forbidden patterns as string literals, exactly like the anchor
# reference_model.py documents. Scanned surfaces: src/, scripts/, docs/,
# config files, Makefile, Dockerfile, CMakeLists.
SCAN_SKIP_DIRS = {"build", ".venv", ".git", "__pycache__", ".pytest_cache",
                  "artifacts", "tests"}


@pytest.fixture(scope="session")
def config() -> dict:
    with (REPO_ROOT / "config.yaml").open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@pytest.mark.parametrize("rel", ANCHOR_FILES)
def test_anchor_files_present(rel: str) -> None:
    assert (REPO_ROOT / rel).is_file(), f"missing anchor file: {rel}"


@pytest.mark.parametrize("key", PATH_KEYS)
def test_path_keys(config: dict, key: str) -> None:
    assert key in config["paths"]


@pytest.mark.parametrize("key", GOLDEN_KEYS)
def test_golden_keys(config: dict, key: str) -> None:
    assert key in config["golden"]


def test_arch_defaults(config: dict) -> None:
    arch = config["arch_defaults"]
    assert arch["rms_norm_eps"] == pytest.approx(1.0e-5)
    assert arch["rope_theta"] == pytest.approx(10000.0)
    assert arch["d_head"] is None


def test_quant_invariants(config: dict) -> None:
    quant = config["quant"]
    assert quant["int4_group"] in (32, 64, 128)
    assert quant["ternary_scale"] in ("mean_abs", "absmax")
    assert quant["int8_axis"] in ("out_channel", "in_channel")


@pytest.mark.parametrize("key", KV_KEYS)
def test_kv_keys(config: dict, key: str) -> None:
    assert key in config["kv"]


def test_kv_page_tokens_power_of_two(config: dict) -> None:
    page = config["kv"]["page_tokens"]
    assert page >= 8 and (page & (page - 1)) == 0


@pytest.mark.parametrize("key", BENCH_KEYS)
def test_bench_keys(config: dict, key: str) -> None:
    assert key in config["bench"]


def test_bench_modes(config: dict) -> None:
    modes = config["bench"]["modes"]
    assert set(modes) == {"fp32", "int8", "int4", "ternary"}


def test_cmake_declares_project_and_std17() -> None:
    text = (REPO_ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    assert re.search(r"project\s*\(\s*hephaestus\b", text)
    assert re.search(r"CXX_STANDARD\s+17", text)
    assert re.search(r"enable_testing\s*\(", text)


def test_makefile_targets_declared() -> None:
    text = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")
    declared = set(re.findall(r"^([a-zA-Z_-]+)\s*:", text, flags=re.MULTILINE))
    missing = [t for t in MAKEFILE_TARGETS if t not in declared]
    assert not missing, f"Makefile missing targets: {missing}"


def test_model_def_covers_canonical_names() -> None:
    text = (REPO_ROOT / "src/model/model_def.h").read_text(encoding="utf-8")
    for i in range(6):  # nano has 6 layers
        for template in CANONICAL_TENSORS:
            name = template.format(i=i)
            assert name in text, f"model_def.h missing canonical name: {name}"
    for name in GLOBAL_TENSORS:
        assert name in text, f"model_def.h missing canonical name: {name}"


def _iter_scan_targets():
    for path in sorted(REPO_ROOT.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(REPO_ROOT).as_posix()
        if any(rel.startswith(skip) or f"/{skip}/" in rel
               for skip in SCAN_SKIP_DIRS):
            continue
        if path.suffix in SCAN_EXTENSIONS or path.name in SCAN_BASENAMES:
            yield path


def test_no_ai_markers() -> None:
    offenders = []
    for path in _iter_scan_targets():
        text = path.read_text(encoding="utf-8", errors="replace").lower()
        for marker in FORBIDDEN_MARKERS:
            if marker in text:
                rel = path.relative_to(REPO_ROOT)
                offenders.append(f"{rel}: contains '{marker}'")
    assert not offenders, (
        "prohibited attribution markers found:\n" + "\n".join(offenders)
    )
