#!/usr/bin/env python3
"""Generate the HEPHAESTUS C++ tree with contracted stubs.

Idempotent: existing files are kept (the agent's implementation is never
overwritten); missing files are created from the contracts below. The
headers are the FINAL contracts — implementations evolve inside them,
the signatures do not move.

Anchors (immutable contracts, must exist before this script runs):
Makefile, config.yaml, requirements.txt, README.md, tests/test_structure.py,
scripts/export_artifacts.py, scripts/export_mapping.yaml,
tests/golden/reference_model.py, tests/golden/golden_test.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PLAIN_DIRS = [
    "src/core",
    "src/loader",
    "src/kernels",
    "src/model",
    "src/kv",
    "src/quant",
    "src/tokenizer",
    "src/bench",
    "tests/cpp",
    "tests/golden",
]

CMAKELISTS = r"""
cmake_minimum_required(VERSION 3.16)
project(hephaestus CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# GCC/Clang flags per spec. cl.exe would need /O2 /arch:AVX2 /W4; this
# project builds under Linux (bare, WSL2 or Docker) by contract.
set(HEPH_FLAGS -O3 -march=native -Wall -Wextra)

add_library(heph_core STATIC
  src/core/tensor.cpp
  src/core/json.cpp
  src/loader/safetensors.cpp
  src/kernels/kernels.cpp
  src/model/transformer.cpp
  src/kv/kv_cache.cpp
  src/quant/quantize.cpp
  src/tokenizer/bpe.cpp
  src/bench/bench.cpp
)
target_include_directories(heph_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_compile_options(heph_core PRIVATE ${HEPH_FLAGS})

add_executable(heph src/main.cpp)
target_link_libraries(heph PRIVATE heph_core)
target_compile_options(heph PRIVATE ${HEPH_FLAGS})

enable_testing()
add_executable(test_kernels tests/cpp/test_kernels.cpp)
target_link_libraries(test_kernels PRIVATE heph_core)
target_compile_options(test_kernels PRIVATE ${HEPH_FLAGS})
add_test(NAME kernel_parity COMMAND test_kernels)
""".strip()

MODEL_DEF_H = r"""
#ifndef HEPH_MODEL_DEF_H_
#define HEPH_MODEL_DEF_H_

// Canonical architecture contract for the PROMETHEUS-NS nano decoder.
//
// WHY HARDCODED: HEPHAESTUS supports EXACTLY this architecture — that is
// the premise of the project, not a limitation to apologize for. Freezing
// the shapes here lets every kernel assume compile-time-known widths and
// lets the loader reject any checkpoint that does not match, instead of
// paying for generic dispatch the spec forbids. If the architecture ever
// changes, this file is the single place to update and the golden tests
// fail loudly until engine and oracle agree again.
//
// nano profile: 6 pre-norm decoder layers, RMSNorm (eps 1e-5), SwiGLU
// (d_ff 1024), RoPE theta 10000 with the half-split convention, GQA with
// n_kv_head == n_head == 6 (repeat_kv is a no-op today — see
// src/model/transformer.h), tied embeddings (lm_head == embedding) over
// a vocab of 8000, context 256 tokens, KV-cache pages of 16 tokens.

#define HEPH_N_LAYER 6
#define HEPH_N_HEAD 6
#define HEPH_N_KV_HEAD 6
#define HEPH_D_MODEL 384
#define HEPH_D_FF 1024
#define HEPH_VOCAB 8000
#define HEPH_D_HEAD (HEPH_D_MODEL / HEPH_N_HEAD)
#define HEPH_KV_PAGE_TOKENS 16

namespace heph {

constexpr int kNLayer = HEPH_N_LAYER;
constexpr int kNHead = HEPH_N_HEAD;
constexpr int kNKvHead = HEPH_N_KV_HEAD;
constexpr int kDModel = HEPH_D_MODEL;
constexpr int kDFf = HEPH_D_FF;
constexpr int kVocab = HEPH_VOCAB;
constexpr int kDHead = HEPH_D_HEAD;
constexpr int kKvPageTokens = HEPH_KV_PAGE_TOKENS;

// Canonical tensor names, enumerated explicitly for all six layers so
// the contract is greppable and diffable (tests/test_structure.py reads
// this file for every name). The export mapping must produce exactly
// this set; the loader refuses anything else.
inline const char* const kCanonicalTensors[] = {
    "embedding.weight",
    "layer.0.attn_norm.weight",
    "layer.1.attn_norm.weight",
    "layer.2.attn_norm.weight",
    "layer.3.attn_norm.weight",
    "layer.4.attn_norm.weight",
    "layer.5.attn_norm.weight",
    "layer.0.attn.wq.weight",
    "layer.1.attn.wq.weight",
    "layer.2.attn.wq.weight",
    "layer.3.attn.wq.weight",
    "layer.4.attn.wq.weight",
    "layer.5.attn.wq.weight",
    "layer.0.attn.wk.weight",
    "layer.1.attn.wk.weight",
    "layer.2.attn.wk.weight",
    "layer.3.attn.wk.weight",
    "layer.4.attn.wk.weight",
    "layer.5.attn.wk.weight",
    "layer.0.attn.wv.weight",
    "layer.1.attn.wv.weight",
    "layer.2.attn.wv.weight",
    "layer.3.attn.wv.weight",
    "layer.4.attn.wv.weight",
    "layer.5.attn.wv.weight",
    "layer.0.attn.wo.weight",
    "layer.1.attn.wo.weight",
    "layer.2.attn.wo.weight",
    "layer.3.attn.wo.weight",
    "layer.4.attn.wo.weight",
    "layer.5.attn.wo.weight",
    "layer.0.ffn_norm.weight",
    "layer.1.ffn_norm.weight",
    "layer.2.ffn_norm.weight",
    "layer.3.ffn_norm.weight",
    "layer.4.ffn_norm.weight",
    "layer.5.ffn_norm.weight",
    "layer.0.ffn.w_gate.weight",
    "layer.1.ffn.w_gate.weight",
    "layer.2.ffn.w_gate.weight",
    "layer.3.ffn.w_gate.weight",
    "layer.4.ffn.w_gate.weight",
    "layer.5.ffn.w_gate.weight",
    "layer.0.ffn.w_up.weight",
    "layer.1.ffn.w_up.weight",
    "layer.2.ffn.w_up.weight",
    "layer.3.ffn.w_up.weight",
    "layer.4.ffn.w_up.weight",
    "layer.5.ffn.w_up.weight",
    "layer.0.ffn.w_down.weight",
    "layer.1.ffn.w_down.weight",
    "layer.2.ffn.w_down.weight",
    "layer.3.ffn.w_down.weight",
    "layer.4.ffn.w_down.weight",
    "layer.5.ffn.w_down.weight",
    "final_norm.weight",
    "lm_head.weight",  // tied to embedding: verified at export, never exported
};

inline constexpr int kCanonicalTensorCount =
    sizeof(kCanonicalTensors) / sizeof(kCanonicalTensors[0]);

}  // namespace heph

#endif  // HEPH_MODEL_DEF_H_
""".strip()

PROMPTS = """# Golden prompts: one prompt per line; the engine and the oracle must
# agree on 64 greedy continuations for each (and on last-position logits).
The quick brown fox jumps over the lazy dog near the river bank while the sun sets.
La inteligencia artificial combina datos, modelos y una buena dosis de paciencia.
In the beginning the engine was only a skeleton of stubs that threw exceptions.
El motor de inferencia mide tokens por segundo con una metodologia honesta y fija.
Memory, alignment and accumulation order decide whether SIMD kernels are fast and correct.
"""

TEST_KERNELS_CPP = r"""
// Kernel parity tests: scalar reference vs SIMD dispatch.
//
// Contract (protocolo de operacion): every SIMD kernel ships with a
// scalar reference and a parity test — bit-identical for integer
// kernels, tolerance 1e-6 for float kernels on the inputs used here.
// These tests are RED while the kernels are stubs; that is the
// contracted starting state of the project.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "kernels/kernels.h"
#include "quant/quantize.h"
#include "core/tensor.h"

namespace {

int g_failures = 0;

void report(const std::string& name, bool ok, const std::string& detail = "") {
  std::printf("%-52s %s%s%s\n", name.c_str(), ok ? "[ OK ]" : "[FAIL]",
              detail.empty() ? "" : " — ", detail.c_str());
  if (!ok) g_failures++;
}

std::vector<float> uniform_vector(std::mt19937& rng, size_t n, float scale) {
  std::uniform_real_distribution<float> dist(-scale, scale);
  std::vector<float> v(n);
  for (auto& x : v) x = dist(rng);
  return v;
}

void expect_close(const std::string& name, const std::vector<float>& a,
                  const std::vector<float>& b, float tol) {
  if (a.size() != b.size()) {
    report(name, false, "size mismatch");
    return;
  }
  float worst = 0.0f;
  size_t where = 0;
  for (size_t i = 0; i < a.size(); ++i) {
    float d = std::fabs(a[i] - b[i]);
    if (d > worst) {
      worst = d;
      where = i;
    }
  }
  report(name, worst <= tol,
         worst <= tol ? ""
                      : "max diff " + std::to_string(worst) + " at " +
                            std::to_string(where));
}

}  // namespace

int main() {
  std::mt19937 rng(20260922u);

  // --- GEMV: y = W x, W row-major [rows x cols], scalar vs SIMD -----
  {
    struct Size { int rows, cols; };
    const Size sizes[] = {{384, 384}, {1024, 384}, {384, 1024}, {7, 13}};
    for (const auto& s : sizes) {
      auto W = uniform_vector(rng, s.rows * s.cols, 0.05f);
      auto x = uniform_vector(rng, s.cols, 0.05f);
      std::vector<float> y_ref(s.rows, 0.0f), y_simd(s.rows, 0.0f);
      heph::gemv_scalar(W.data(), x.data(), y_ref.data(), s.rows, s.cols);
      try {
        heph::gemv(W.data(), x.data(), y_simd.data(), s.rows, s.cols);
        std::vector<float> y_ref2 = y_ref;  // keep separate on purpose
        expect_close("gemv parity " + std::to_string(s.rows) + "x" +
                         std::to_string(s.cols),
                     y_ref2, y_simd, 1e-6f);
      } catch (const std::exception& e) {
        report("gemv parity " + std::to_string(s.rows) + "x" +
                   std::to_string(s.cols), false, e.what());
      }
    }
  }

  // --- RMSNorm parity -------------------------------------------------
  {
    auto x = uniform_vector(rng, 384, 1.0f);
    auto w = uniform_vector(rng, 384, 0.5f);
    std::vector<float> a(384), b(384);
    heph::rmsnorm_scalar(x.data(), w.data(), a.data(), 384, 1e-5f);
    try {
      heph::rmsnorm(x.data(), w.data(), b.data(), 384, 1e-5f);
      expect_close("rmsnorm parity", a, b, 1e-6f);
    } catch (const std::exception& e) {
      report("rmsnorm parity", false, e.what());
    }
  }

  // --- SiLU parity ----------------------------------------------------
  {
    auto x = uniform_vector(rng, 1024, 8.0f);
    std::vector<float> a(1024), b(1024);
    heph::silu_scalar(x.data(), a.data(), 1024);
    try {
      heph::silu(x.data(), b.data(), 1024);
      expect_close("silu parity", a, b, 1e-6f);
    } catch (const std::exception& e) {
      report("silu parity", false, e.what());
    }
  }

  // --- softmax parity ---------------------------------------------------
  {
    auto x = uniform_vector(rng, 8000, 12.0f);
    std::vector<float> a(8000), b(8000);
    heph::softmax_scalar(x.data(), a.data(), 8000);
    try {
      heph::softmax(x.data(), b.data(), 8000);
      expect_close("softmax parity", a, b, 1e-6f);
    } catch (const std::exception& e) {
      report("softmax parity", false, e.what());
    }
  }

  // --- RoPE parity (half-split, 6 heads x 64 dims) --------------------
  {
    const int heads = 6, d = 64, seq = 33;
    auto q = uniform_vector(rng, seq * heads * d, 1.0f);
    auto cos = uniform_vector(rng, seq * (d / 2), 1.0f);
    auto sin = uniform_vector(rng, seq * (d / 2), 1.0f);
    std::vector<float> a = q, b = q;
    heph::rope_scalar(a.data(), cos.data(), sin.data(), seq, heads, d);
    try {
      heph::rope(b.data(), cos.data(), sin.data(), seq, heads, d);
      expect_close("rope parity", a, b, 1e-6f);
    } catch (const std::exception& e) {
      report("rope parity", false, e.what());
    }
  }

  // --- runtime dispatch sanity ----------------------------------------
  try {
    bool avx2 = heph::has_avx2();
    bool avx512 = heph::has_avx512();
    report("dispatch flags consistent", avx2 && (!avx512 || avx2),
           avx512 ? "AVX-512 available" : "AVX-2 path");
  } catch (const std::exception& e) {
    report("dispatch flags consistent", false, e.what());
  }

  // --- tensor alignment -------------------------------------------------
  try {
    heph::Tensor t = heph::make_tensor(17, 19);
    bool aligned = (reinterpret_cast<uintptr_t>(t.data) % 64) == 0;
    report("tensor buffer 64B aligned", aligned && t.size() == 17 * 19);
    heph::free_tensor(t);
  } catch (const std::exception& e) {
    report("tensor buffer 64B aligned", false, e.what());
  }

  // --- quantized kernels: int accumulation is BIT EXACT -----------------
  try {
    const int rows = 48, cols = 384;  // cols > 256 on purpose (int16 would overflow)
    auto W = uniform_vector(rng, rows * cols, 0.05f);
    auto x = uniform_vector(rng, cols, 0.05f);
    std::vector<float> y_ref(rows), y_simd(rows);
    heph::gemv_q8_scalar(W.data(), x.data(), y_ref.data(), rows, cols);
    heph::gemv_q8(W.data(), x.data(), y_simd.data(), rows, cols);
    bool exact = std::memcmp(y_ref.data(), y_simd.data(),
                             sizeof(float) * rows) == 0;
    report("gemv_q8 int32 accumulation bit-exact", exact);
  } catch (const std::exception& e) {
    report("gemv_q8 int32 accumulation bit-exact", false, e.what());
  }

  try {
    const int rows = 32, cols = 256;  // multiple of two groups of 128
    auto W = uniform_vector(rng, rows * cols, 0.05f);
    auto x = uniform_vector(rng, cols, 0.05f);
    std::vector<float> y_ref(rows), y_simd(rows);
    heph::gemv_q4_scalar(W.data(), x.data(), y_ref.data(), rows, cols, 128);
    heph::gemv_q4(W.data(), x.data(), y_simd.data(), rows, cols, 128);
    bool exact = std::memcmp(y_ref.data(), y_simd.data(),
                             sizeof(float) * rows) == 0;
    report("gemv_q4 nibble order bit-exact", exact);
  } catch (const std::exception& e) {
    report("gemv_q4 nibble order bit-exact", false, e.what());
  }

  try {
    const int rows = 32, cols = 384;
    auto W = uniform_vector(rng, rows * cols, 0.05f);
    auto x = uniform_vector(rng, cols, 0.05f);
    std::vector<float> y_ref(rows), y_simd(rows);
    heph::gemv_t1_scalar(W.data(), x.data(), y_ref.data(), rows, cols);
    heph::gemv_t1(W.data(), x.data(), y_simd.data(), rows, cols);
    bool exact = std::memcmp(y_ref.data(), y_simd.data(),
                             sizeof(float) * rows) == 0;
    report("gemv_t1 lookup bit-exact", exact);
  } catch (const std::exception& e) {
    report("gemv_t1 lookup bit-exact", false, e.what());
  }

  if (g_failures) {
    std::printf("\nkernel parity: %d FAILURES\n", g_failures);
  } else {
    std::printf("\nkernel parity: all green\n");
  }
  return g_failures ? 1 : 0;
}
""".strip()

STUBS = {}

STUBS["src/core/tensor.h"] = (r"""
// Minimal tensor type: row-major, fp32.
//
// Contract: every activation buffer handed to SIMD kernels comes from
// alloc_f32 (posix_memalign, 64B) so aligned loads stay legal. Kernels
// nevertheless use unaligned load intrinsics (mandate #3: alignment can
// never crash us) — the aligned allocation is hygiene, not a dependency.

#include <cstddef>
#include <cstdint>

namespace heph {

float* alloc_f32(size_t n);
void free_f32(float* p);

struct Tensor {
  float* data = nullptr;
  int64_t rows = 0;
  int64_t cols = 0;
  int64_t size() const { return rows * cols; }
};

Tensor make_tensor(int64_t rows, int64_t cols);
void free_tensor(Tensor& t);

}  // namespace heph
""", r"""
#include "core/tensor.h"

#include <cstddef>
#include <stdexcept>
#include <cstdlib>

namespace heph {

float* alloc_f32(size_t n) {
  throw std::runtime_error("not implemented: src/core/tensor.cpp");
}

void free_f32(float* p) {
  throw std::runtime_error("not implemented: src/core/tensor.cpp");
}

Tensor make_tensor(int64_t rows, int64_t cols) {
  throw std::runtime_error("not implemented: src/core/tensor.cpp");
}

void free_tensor(Tensor& t) {
  throw std::runtime_error("not implemented: src/core/tensor.cpp");
}

}  // namespace heph
""")

STUBS["src/core/json.h"] = (r"""
// Hand-rolled minimal JSON reader (the engine has ZERO third-party
// dependencies: no nlohmann). Supports the safetensors header,
// vocab.json and specials.json: null/bool/number/string/array/object,
// UTF-8 with \uXXXX escapes INCLUDING surrogate pairs (the byte-level
// alphabet contains U+0120 and friends).

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace heph::json {

struct Value {
  enum class Type { Null, Bool, Number, String, Array, Object };
  Type type = Type::Null;
  bool boolean = false;
  double number = 0.0;
  std::string text;
  std::vector<Value> items;
  std::vector<std::pair<std::string, Value>> fields;

  const Value* find(const char* key) const;
  int64_t as_int() const;
  double as_double() const;
  const std::string& as_string() const;
};

Value parse(const char* data, size_t n);
inline Value parse(const std::string& s) { return parse(s.data(), s.size()); }

}  // namespace heph::json
""", r"""
#include "core/json.h"

#include <stdexcept>

namespace heph::json {

const Value* Value::find(const char* key) const {
  throw std::runtime_error("not implemented: src/core/json.cpp");
}

int64_t Value::as_int() const {
  throw std::runtime_error("not implemented: src/core/json.cpp");
}

double Value::as_double() const {
  throw std::runtime_error("not implemented: src/core/json.cpp");
}

const std::string& Value::as_string() const {
  throw std::runtime_error("not implemented: src/core/json.cpp");
}

Value parse(const char* data, size_t n) {
  throw std::runtime_error("not implemented: src/core/json.cpp");
}

}  // namespace heph::json
""")

STUBS["src/loader/safetensors.h"] = (r"""
// Own safetensors loader + artifact parsers.
//
// Format: 8-byte little-endian header length, JSON header
// {name: {dtype, shape, data_offsets:[start,end]}}, then raw data.
// The loader validates dtype == F32 and keeps a read-only view into the
// file bytes (no copies). tensors.tsv offsets written by the exporter
// are absolute: 8 + header_len + start.
//
// FASE 1 VERIF: tensor_sha256() must return the digest of every mapped
// tensor region; the golden tests compare against independent digests
// of the same byte ranges of the weights file.

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace heph {

std::string sha256_hex(const uint8_t* data, size_t n);

struct SafeTensors {
  static SafeTensors load(const std::string& path);
  bool has(const std::string& name) const;
  const float* data(const std::string& name) const;
  const std::vector<int64_t>& shape(const std::string& name) const;
  size_t count() const;
  const std::vector<std::string>& names() const;
  std::vector<std::pair<std::string, std::string>> tensor_sha256() const;

 private:
  std::vector<uint8_t> blob_;
  std::vector<std::string> names_;
  struct Entry {
    size_t byte_offset = 0;
    size_t nbytes = 0;
    std::vector<int64_t> shape;
  };
  std::map<std::string, Entry> entries_;
};

struct Manifest {
  int d_model = 0, n_layer = 0, n_head = 0, n_kv_head = 0;
  int d_head = 0, d_ff = 0, vocab = 0, max_seq = 0;
  float rms_norm_eps = 0.0f, rope_theta = 0.0f;
  std::string weights_file, weights_sha256;
  static Manifest load(const std::string& path);
};

struct TensorsTable {
  struct Row {
    std::string name;
    std::vector<int64_t> shape;
    std::string dtype;
    size_t offset = 0;
    size_t nbytes = 0;
  };
  static std::vector<Row> load(const std::string& path);
};

}  // namespace heph
""", r"""
#include "loader/safetensors.h"

#include <stdexcept>

namespace heph {

std::string sha256_hex(const uint8_t* data, size_t n) {
  throw std::runtime_error("not implemented: src/loader/safetensors.cpp");
}

SafeTensors SafeTensors::load(const std::string& path) {
  throw std::runtime_error("not implemented: src/loader/safetensors.cpp");
}

bool SafeTensors::has(const std::string& name) const {
  throw std::runtime_error("not implemented: src/loader/safetensors.cpp");
}

const float* SafeTensors::data(const std::string& name) const {
  throw std::runtime_error("not implemented: src/loader/safetensors.cpp");
}

const std::vector<int64_t>& SafeTensors::shape(const std::string& name) const {
  throw std::runtime_error("not implemented: src/loader/safetensors.cpp");
}

size_t SafeTensors::count() const {
  throw std::runtime_error("not implemented: src/loader/safetensors.cpp");
}

const std::vector<std::string>& SafeTensors::names() const {
  throw std::runtime_error("not implemented: src/loader/safetensors.cpp");
}

std::vector<std::pair<std::string, std::string>> SafeTensors::tensor_sha256()
    const {
  throw std::runtime_error("not implemented: src/loader/safetensors.cpp");
}

Manifest Manifest::load(const std::string& path) {
  throw std::runtime_error("not implemented: src/loader/safetensors.cpp");
}

std::vector<TensorsTable::Row> TensorsTable::load(const std::string& path) {
  throw std::runtime_error("not implemented: src/loader/safetensors.cpp");
}

}  // namespace heph
""")

STUBS["src/kernels/kernels.h"] = (r"""
// Own kernels: GEMV, RMSNorm, SiLU, softmax, RoPE.
//
// Contract: every kernel has a scalar reference (*_scalar) and a SIMD
// dispatch (*). Parity (tests/cpp/test_kernels.cpp): 1e-6 tolerance for
// float kernels on the tested inputs.
//
// SIMD notes (frontier problems #3):
//   - AVX2 + FMA intrinsics; AVX-512 detected at runtime and used when
//     present (has_avx512()).
//   - Use unaligned loads (loadu) — mandate #3 — and accumulate dots
//     lane-parallel, then horizontal-reduce in a FIXED order so results
//     are reproducible run to run.
//   - RoPE is half-split: x1 = first half of the head, x2 = second;
//     out = [x1*cos - x2*sin, x1*sin + x2*cos] (matches PROMETHEUS-NS
//     and the NumPy oracle). x layout: [seq][heads][d_head]; tables
//     [seq][d_head/2].

namespace heph {

bool has_avx2();
bool has_avx512();

void gemv_scalar(const float* w, const float* x, float* y, int rows,
                 int cols);
void rmsnorm_scalar(const float* x, const float* w, float* out, int n,
                    float eps);
void silu_scalar(const float* x, float* out, int n);
void softmax_scalar(const float* x, float* out, int n);
void rope_scalar(float* x, const float* cos, const float* sin, int seq,
                 int heads, int d_head);
void add_in_place(float* acc, const float* add, int n);

void gemv(const float* w, const float* x, float* y, int rows, int cols);
void rmsnorm(const float* x, const float* w, float* out, int n, float eps);
void silu(const float* x, float* out, int n);
void softmax(const float* x, float* out, int n);
void rope(float* x, const float* cos, const float* sin, int seq, int heads,
          int d_head);

}  // namespace heph
""", r"""
#include "kernels/kernels.h"

#include <stdexcept>

namespace heph {

bool has_avx2() {
  throw std::runtime_error("not implemented: src/kernels/kernels.cpp");
}

bool has_avx512() {
  throw std::runtime_error("not implemented: src/kernels/kernels.cpp");
}

void gemv_scalar(const float* w, const float* x, float* y, int rows,
                 int cols) {
  throw std::runtime_error("not implemented: src/kernels/kernels.cpp");
}

void rmsnorm_scalar(const float* x, const float* w, float* out, int n,
                    float eps) {
  throw std::runtime_error("not implemented: src/kernels/kernels.cpp");
}

void silu_scalar(const float* x, float* out, int n) {
  throw std::runtime_error("not implemented: src/kernels/kernels.cpp");
}

void softmax_scalar(const float* x, float* out, int n) {
  throw std::runtime_error("not implemented: src/kernels/kernels.cpp");
}

void rope_scalar(float* x, const float* cos, const float* sin, int seq,
                 int heads, int d_head) {
  throw std::runtime_error("not implemented: src/kernels/kernels.cpp");
}

void add_in_place(float* acc, const float* add, int n) {
  throw std::runtime_error("not implemented: src/kernels/kernels.cpp");
}

void gemv(const float* w, const float* x, float* y, int rows, int cols) {
  throw std::runtime_error("not implemented: src/kernels/kernels.cpp");
}

void rmsnorm(const float* x, const float* w, float* out, int n, float eps) {
  throw std::runtime_error("not implemented: src/kernels/kernels.cpp");
}

void silu(const float* x, float* out, int n) {
  throw std::runtime_error("not implemented: src/kernels/kernels.cpp");
}

void softmax(const float* x, float* out, int n) {
  throw std::runtime_error("not implemented: src/kernels/kernels.cpp");
}

void rope(float* x, const float* cos, const float* sin, int seq, int heads,
          int d_head) {
  throw std::runtime_error("not implemented: src/kernels/kernels.cpp");
}

}  // namespace heph
""")

STUBS["src/model/transformer.h"] = (r"""
// Forward pass, forward-dump and greedy generation for the frozen nano
// contract (src/model/model_def.h).
//
// repeat_kv contract (package warning #2): nano has n_head == n_kv_head,
// so the KV expansion is identity today. If a future profile uses real
// GQA (n_kv_head < n_head), the expansion must be switched on HERE and
// in tests/golden/reference_model.py (np.repeat) AT THE SAME TIME — the
// golden tests compare engine vs oracle outputs and would catch any
// one-sided change.
//
// Attention numerics: scores = q.k / sqrt(d_head), causal mask, softmax
// in fp32, attention out = probs.v, then wo. Kernels come from
// kernels/; the KV cache from kv/.

#include <vector>

#include "loader/safetensors.h"
#include "model/model_def.h"
#include "quant/quantize.h"

namespace heph {

class KVCache;

struct LayerWeights {
  const float *attn_norm = nullptr, *wq = nullptr, *wk = nullptr,
              *wv = nullptr, *wo = nullptr, *ffn_norm = nullptr,
              *w_gate = nullptr, *w_up = nullptr, *w_down = nullptr;
};

struct Weights {
  const float* embedding = nullptr;
  const float* final_norm = nullptr;
  LayerWeights layers[kNLayer] = {};
  static Weights load(const SafeTensors& st, const Manifest& mf);
};

class Transformer {
 public:
  Transformer(Weights w, const Manifest& m);
  void forward_last_logits(const int* tokens, int seq, float* out) const;
  void forward_with_cache(const int* tokens, int seq, int pos_offset,
                          KVCache& kv, float* out) const;
  std::vector<int> greedy(const std::vector<int>& prompt, int n_new,
                          int eos_id, KVCache& kv) const;
  const Manifest& manifest() const { return m_; }

 private:
  Weights w_;
  Manifest m_;
  std::vector<float> cos_, sin_;  // rope tables [max_seq, d_head/2]
  void build_rope_tables();
};

class QuantTransformer {
 public:
  QuantTransformer(const QuantModel& qm, const Manifest& m);
  void forward_last_logits(const int* tokens, int seq, float* out) const;
  std::vector<int> greedy(const std::vector<int>& prompt, int n_new,
                          int eos_id, KVCache& kv) const;
  const Manifest& manifest() const { return m_; }

 private:
  const QuantModel& qm_;
  Manifest m_;
  std::vector<float> cos_, sin_;
  void build_rope_tables();
};

}  // namespace heph
""", r"""
#include "model/transformer.h"

#include <stdexcept>

namespace heph {

Weights Weights::load(const SafeTensors& st, const Manifest& mf) {
  throw std::runtime_error("not implemented: src/model/transformer.cpp");
}

Transformer::Transformer(Weights w, const Manifest& m)
    : w_(w), m_(m) {}

void Transformer::forward_last_logits(const int* tokens, int seq,
                                      float* out) const {
  throw std::runtime_error("not implemented: src/model/transformer.cpp");
}

void Transformer::forward_with_cache(const int* tokens, int seq,
                                     int pos_offset, KVCache& kv,
                                     float* out) const {
  throw std::runtime_error("not implemented: src/model/transformer.cpp");
}

std::vector<int> Transformer::greedy(const std::vector<int>& prompt,
                                     int n_new, int eos_id,
                                     KVCache& kv) const {
  throw std::runtime_error("not implemented: src/model/transformer.cpp");
}

QuantTransformer::QuantTransformer(const QuantModel& qm, const Manifest& m)
    : qm_(qm), m_(m) {}

void QuantTransformer::forward_last_logits(const int* tokens, int seq,
                                           float* out) const {
  throw std::runtime_error("not implemented: src/model/transformer.cpp");
}

std::vector<int> QuantTransformer::greedy(const std::vector<int>& prompt,
                                          int n_new, int eos_id,
                                          KVCache& kv) const {
  throw std::runtime_error("not implemented: src/model/transformer.cpp");
}

}  // namespace heph
""")

STUBS["src/kv/kv_cache.h"] = (r"""
// Paged KV cache (Fase 3 fp32, Fase 5 int8).
//
// Pages of kKvPageTokens (16) tokens; a block table grows on demand.
//
// PARTIAL-BLOCK MASKING CONTRACT (frontier problem #6): at step t=18 the
// first page holds 16 valid tokens and the second only 2 — the remaining
// 14 slots are uninitialized poison. Attention must multiply scores only
// over tokens() entries; read_layer() returns exactly tokens() rows, so
// callers never see the poison tail.
//
// RoPE STATE CONTRACT (frontier problem #6b): only the INCOMING token is
// rotated, at its absolute position t. Cached K/V keep the rotation they
// were stored with; never re-rotate the accumulated buffer.
//
// KVCacheInt8 stores, per head and per token, q8[d_head] + fp32 scale
// (kv.quant: int8 from config.yaml).

#include <cstdint>
#include <memory>
#include <vector>

#include "model/model_def.h"

namespace heph {

class KVCache {
 public:
  virtual ~KVCache() = default;
  virtual void reset() = 0;
  virtual void append(int layer, const float* k, const float* v,
                      int n_new) = 0;
  virtual int tokens() const = 0;
  // Dequantized contiguous view [tokens() x n_kv_head x d_head].
  virtual void read_layer(int layer, float* k_cont, float* v_cont) const = 0;
};

class KVCacheFp32 final : public KVCache {
 public:
  void reset() override;
  void append(int layer, const float* k, const float* v, int n_new) override;
  int tokens() const override;
  void read_layer(int layer, float* k_cont, float* v_cont) const override;

 private:
  int total_ = 0;
  std::vector<std::vector<std::unique_ptr<float[]>>> k_pages_, v_pages_;
};

class KVCacheInt8 final : public KVCache {
 public:
  void reset() override;
  void append(int layer, const float* k, const float* v, int n_new) override;
  int tokens() const override;
  void read_layer(int layer, float* k_cont, float* v_cont) const override;

 private:
  int total_ = 0;
  std::vector<std::vector<std::unique_ptr<int8_t[]>>> kq_pages_, vq_pages_;
  std::vector<std::vector<std::unique_ptr<float[]>>> ks_pages_, vs_pages_;
};

}  // namespace heph
""", r"""
#include "kv/kv_cache.h"

#include <stdexcept>

namespace heph {

void KVCacheFp32::reset() {
  throw std::runtime_error("not implemented: src/kv/kv_cache.cpp");
}

void KVCacheFp32::append(int layer, const float* k, const float* v,
                         int n_new) {
  throw std::runtime_error("not implemented: src/kv/kv_cache.cpp");
}

int KVCacheFp32::tokens() const {
  throw std::runtime_error("not implemented: src/kv/kv_cache.cpp");
}

void KVCacheFp32::read_layer(int layer, float* k_cont,
                             float* v_cont) const {
  throw std::runtime_error("not implemented: src/kv/kv_cache.cpp");
}

void KVCacheInt8::reset() {
  throw std::runtime_error("not implemented: src/kv/kv_cache.cpp");
}

void KVCacheInt8::append(int layer, const float* k, const float* v,
                         int n_new) {
  throw std::runtime_error("not implemented: src/kv/kv_cache.cpp");
}

int KVCacheInt8::tokens() const {
  throw std::runtime_error("not implemented: src/kv/kv_cache.cpp");
}

void KVCacheInt8::read_layer(int layer, float* k_cont,
                             float* v_cont) const {
  throw std::runtime_error("not implemented: src/kv/kv_cache.cpp");
}

}  // namespace heph
""")

STUBS["src/quant/quantize.h"] = (r"""
// Progressive weight quantization: int8 / int4 groupwise / ternary.
//
// Engine-side artifact format (heph quantize --out DIR):
//   DIR/tensors.tsv   name \t rows x cols \t {Q8,Q4,T1} \t offset \t nbytes
//   DIR/weights.bin   packed payload at the recorded offsets
//
// Numerical contracts (frontier problem #4):
//   - int8: symmetric per OUTPUT channel (row): scale[r] = max|w[r,:]|/127;
//     q = round(w/scale) clamped to [-127, 127].
//   - int4: groupwise along the input dim (group=128), scale per group
//     stored as fp16 bits; packing is LOW NIBBLE FIRST: byte = (q0&0xF)
//     | (q1<<4) for weight pairs (2i, 2i+1) — the SAME order the kernel
//     reads (mandate #2: define once, both sides agree, tests pin it).
//   - ternary: q in {-1,0,1} = round-to-nearest of w/scale with
//     scale = mean(|w|) per tensor; 2-bit codes packed 4-per-byte
//     low-first; GEMV uses a lookup of {-scale, 0, +scale}.
//   - INT ACCUMULATION (mandate #4): every quantized dot accumulates in
//     int32. 16-bit accumulation overflows at d_model=384.
//
// Ternary on a model not trained for it (no QAT) degrades severely —
// that is a primary measured finding, never a bug to hide.

#include <cstdint>
#include <string>
#include <vector>

#include "loader/safetensors.h"

namespace heph {

// Parity kernels: quantize a row on the fly from fp weights (both the
// scalar and the SIMD version share the same quantization step, so the
// int accumulation path is compared bit a bit).
void gemv_q8_scalar(const float* w, const float* x, float* y, int rows,
                    int cols);
void gemv_q8(const float* w, const float* x, float* y, int rows, int cols);
void gemv_q4_scalar(const float* w, const float* x, float* y, int rows,
                    int cols, int group);
void gemv_q4(const float* w, const float* x, float* y, int rows, int cols,
             int group);
void gemv_t1_scalar(const float* w, const float* x, float* y, int rows,
                    int cols);
void gemv_t1(const float* w, const float* x, float* y, int rows, int cols);

struct QuantModel {
  enum class Mode { Int8, Int4, Ternary };

  struct QTensor {
    std::string name;
    int rows = 0, cols = 0;
    Mode mode = Mode::Int8;
    size_t offset = 0, nbytes = 0;
    std::vector<float> scales;  // int8: rows; int4: rows*ceil(cols/group); ternary: 1
  };

  Mode mode = Mode::Int8;
  int group = 128;
  std::vector<uint8_t> blob;
  std::vector<QTensor> tensors;

  const QTensor* tensor(const std::string& name) const;
  // y[rows] = Wq @ x; returns the leading output scale application is
  // already applied (scales folded inside).
  float qgemv(const std::string& name, const float* x, float* y) const;

  static void write_dir(const QuantModel& qm, const std::string& dir);
  static QuantModel read_dir(const std::string& dir);
};

QuantModel build_quantized(const SafeTensors& st, const std::string& mode,
                           int group);

}  // namespace heph
""", r"""
#include "quant/quantize.h"

#include <stdexcept>

namespace heph {

void gemv_q8_scalar(const float* w, const float* x, float* y, int rows,
                    int cols) {
  throw std::runtime_error("not implemented: src/quant/quantize.cpp");
}

void gemv_q8(const float* w, const float* x, float* y, int rows, int cols) {
  throw std::runtime_error("not implemented: src/quant/quantize.cpp");
}

void gemv_q4_scalar(const float* w, const float* x, float* y, int rows,
                    int cols, int group) {
  throw std::runtime_error("not implemented: src/quant/quantize.cpp");
}

void gemv_q4(const float* w, const float* x, float* y, int rows, int cols,
             int group) {
  throw std::runtime_error("not implemented: src/quant/quantize.cpp");
}

void gemv_t1_scalar(const float* w, const float* x, float* y, int rows,
                    int cols) {
  throw std::runtime_error("not implemented: src/quant/quantize.cpp");
}

void gemv_t1(const float* w, const float* x, float* y, int rows, int cols) {
  throw std::runtime_error("not implemented: src/quant/quantize.cpp");
}

QuantModel build_quantized(const SafeTensors& st, const std::string& mode,
                           int group) {
  throw std::runtime_error("not implemented: src/quant/quantize.cpp");
}

void QuantModel::write_dir(const QuantModel& qm, const std::string& dir) {
  throw std::runtime_error("not implemented: src/quant/quantize.cpp");
}

QuantModel QuantModel::read_dir(const std::string& dir) {
  throw std::runtime_error("not implemented: src/quant/quantize.cpp");
}

float QuantModel::qgemv(const std::string& name, const float* x,
                        float* y) const {
  throw std::runtime_error("not implemented: src/quant/quantize.cpp");
}

}  // namespace heph
""")

STUBS["src/tokenizer/bpe.h"] = (r"""
// Byte-level BPE runtime (a reader, not a trainer).
//
// The three pitfalls these contracts encode (tests/golden/
// test_tokenize_parity.py enforces all three):
//   (1) GPT-2 byte->unicode alphabet map BEFORE vocab lookup
//       (byte 0x20 renders as U+0120); decode applies the inverse.
//   (2) The GPT-2 pre-tokenizer
//         's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|
//         \s+(?!\S)|\s+
//       implemented as a HAND-WRITTEN lexer over Unicode letter/number
//       categories: std::regex has no Unicode support and Boost/PCRE are
//       forbidden (frontier problem #2).
//   (3) Merges run by RANK: repeatedly merge the adjacent pair with the
//       lowest merges.txt priority, never simply the first pair found
//       scanning left to right.
// encode() never injects special tokens; unk must never fire (byte
// coverage is total).

#include <map>
#include <string>
#include <utility>
#include <vector>

namespace heph {

struct SpecialIds {
  int pad = 0, bos = 1, eos = 2, unk = 3;
};

class BPETokenizer {
 public:
  static BPETokenizer load(const std::string& dir);
  std::vector<int> encode(const std::string& text) const;
  std::string decode(const std::vector<int>& ids) const;
  int vocab_size() const;
  const SpecialIds& specials() const { return special_ids_; }

 private:
  SpecialIds special_ids_;
  std::vector<std::string> vocab_strings_;  // id -> symbol (byte-unicode space)
  std::map<std::string, int> vocab_ids_;    // symbol -> id
  std::map<std::pair<std::string, std::string>, int> merge_rank_;
};

}  // namespace heph
""", r"""
#include "tokenizer/bpe.h"

#include <stdexcept>

namespace heph {

BPETokenizer BPETokenizer::load(const std::string& dir) {
  throw std::runtime_error("not implemented: src/tokenizer/bpe.cpp");
}

std::vector<int> BPETokenizer::encode(const std::string& text) const {
  throw std::runtime_error("not implemented: src/tokenizer/bpe.cpp");
}

std::string BPETokenizer::decode(const std::vector<int>& ids) const {
  throw std::runtime_error("not implemented: src/tokenizer/bpe.cpp");
}

int BPETokenizer::vocab_size() const {
  throw std::runtime_error("not implemented: src/tokenizer/bpe.cpp");
}

}  // namespace heph
""")

STUBS["src/bench/bench.h"] = (r"""
// Benchmark suite: {fp32, int8, int4, ternary} x {tokens/s decode
// batch=1, TTFT, peak RSS, ppl}.
//
// Methodology (docs/benchmark.md is generated from these outputs):
//   - decode tokens/s: greedy decode of bench.max_new_tokens after
//     bench.warmup discarded warmup prompts; prefill excluded (TTFT).
//   - peak RSS: getrusage(RUSAGE_SELF).ru_maxrss — Linux reports KiB,
//     converted to MB; process lifetime peak including the loader.
//   - ppl (optional --holdout): mean NLL over all predicted tokens on
//     non-overlapping max_seq windows of the frozen PROMETHEUS-NS
//     holdout; fp32 ppl comes from the same engine path.
// Every value is measured; nothing is estimated (REGLA DE HONESTIDAD).

#include <string>
#include <vector>

#include "tokenizer/bpe.h"

namespace heph {

// Engine abstraction the bench loop drives (fp32 or quantized).
class IEngine {
 public:
  virtual ~IEngine() = default;
  virtual int vocab() const = 0;
  virtual std::vector<int> generate_greedy(const std::vector<int>& prompt,
                                           int n_new, int eos_id) = 0;
  virtual double mean_nll(const std::vector<int>& tokens, int window) = 0;
  virtual void reset() = 0;
};

double peak_rss_mb();

struct Holdout {
  std::string text;
  static Holdout load(const std::string& path);
};

struct BenchOptions {
  int prompts = 50;
  int warmup = 5;
  int max_new_tokens = 128;
};

struct BenchOutput {
  std::string mode;
  double tokens_per_s_decode = 0.0;
  double ttft_s = 0.0;
  double peak_rss_mb = 0.0;
  std::vector<double> per_prompt_ms;
  double ppl = 0.0;
  long ppl_tokens = 0;
  std::string to_json() const;
};

BenchOutput run_bench(IEngine& engine, const BPETokenizer& tok,
                      const BenchOptions& opts, const Holdout* holdout,
                      const std::string& mode);

}  // namespace heph
""", r"""
#include "bench/bench.h"

#include <stdexcept>

namespace heph {

double peak_rss_mb() {
  throw std::runtime_error("not implemented: src/bench/bench.cpp");
}

Holdout Holdout::load(const std::string& path) {
  throw std::runtime_error("not implemented: src/bench/bench.cpp");
}

BenchOutput run_bench(IEngine& engine, const BPETokenizer& tok,
                      const BenchOptions& opts, const Holdout* holdout,
                      const std::string& mode) {
  throw std::runtime_error("not implemented: src/bench/bench.cpp");
}

std::string BenchOutput::to_json() const {
  throw std::runtime_error("not implemented: src/bench/bench.cpp");
}

}  // namespace heph
""")

STUBS["src/main.cpp"] = (r"""
// heph — HEPHAESTUS engine CLI.
//
// Subcommands and flags are FIXED by the Makefile comments (anchor):
//   heph golden    --manifest M --weights W --tokenizer D --prompts F
//                  --tokens N --dump FILE [--dump-sha FILE]
//   heph generate  --manifest M --weights W --tokenizer D
//                  [--interactive | --prompt P] [--greedy]
//                  [--temperature T] [--top-p P] [--max-new-tokens N]
//                  [--ids]
//   heph bench     --manifest M --weights W --tokenizer D --mode MODE
//                  --prompts N --warmup W --max-new-tokens T --out FILE
//                  [--holdout FILE]
//   heph quantize  --manifest M --weights W --mode MODE --group G
//                  --out DIR
//   heph tokenize  --tokenizer D --file IN --out OUT
""", r"""
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>

int main(int argc, char** argv) {
  (void)argc;
  (void)argv;
  std::fprintf(stderr,
               "heph — HEPHAESTUS engine (skeleton; kernels are stubs)\n");
  throw std::runtime_error("not implemented: src/main.cpp");
}
""")


def cpp_stub(contract: str, guard: str) -> str:
    """A header is its contract comment wrapped in an include guard."""
    if guard:
        return (f"#ifndef {guard}\n#define {guard}\n\n{contract}\n\n"
                f"#endif  // {guard}\n")
    return contract + "\n"


def write(rel: str, content: str, binary: bool = False) -> None:
    target = ROOT / rel
    if target.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content.encode("utf-8"))


def main() -> int:
    created, kept = [], []
    for rel in PLAIN_DIRS:
        (ROOT / rel).mkdir(parents=True, exist_ok=True)

    def tracked_write(rel: str, content: str) -> None:
        if (ROOT / rel).exists():
            kept.append(rel)
            return
        write(rel, content)
        created.append(rel)

    tracked_write("CMakeLists.txt", CMAKELISTS + "\n")
    tracked_write("src/model/model_def.h", MODEL_DEF_H + "\n")
    tracked_write("tests/cpp/test_kernels.cpp", TEST_KERNELS_CPP + "\n")
    tracked_write("tests/golden/prompts.txt", PROMPTS)

    for rel, (contract, cpp) in STUBS.items():
        if rel.endswith(".h"):
            guard = "_" + rel.upper().replace("/", "_").replace(".", "_") + "_"
            tracked_write(rel, cpp_stub(contract, guard))
        elif rel.endswith(".cpp"):
            tracked_write(rel, cpp)

    print("created:")
    for rel in created:
        print(f"  + {rel}")
    if kept:
        print("kept existing:")
        for rel in kept:
            print(f"  = {rel}")

    anchors = [
        "Makefile", "config.yaml", "requirements.txt", "README.md",
        "tests/test_structure.py", "scripts/export_artifacts.py",
        "scripts/export_mapping.yaml", "tests/golden/reference_model.py",
        "tests/golden/golden_test.py",
    ]
    print("anchors:")
    missing = 0
    for rel in anchors:
        ok = (ROOT / rel).is_file()
        print(f"  {'ok' if ok else '!!'} {rel}: "
              f"{'present' if ok else 'MISSING: copy it from the specification package'}")
        missing += 0 if ok else 1
    print("\nnext: make setup && make configure && make build && make test")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())

