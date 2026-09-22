#!/usr/bin/env python3
"""Generate the HEPHAESTUS C++ tree with contracted stubs.

Idempotent: files that already exist are kept untouched ("kept existing"),
so re-running after implementation never overwrites work. The stubs
compile cleanly and throw std::runtime_error until each phase is
implemented. After generation the script lists the anchor files that must
already be present from the specification package.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PLAIN_DIRS = [
    "src/core", "src/loader", "src/kernels", "src/model", "src/kv",
    "src/quant", "src/tokenizer", "src/bench",
    "tests/cpp", "tests/golden", "docs", "artifacts",
]

CMAKELISTS = """
cmake_minimum_required(VERSION 3.16)
project(hephaestus CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
enable_testing()

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# POSIX-only build: getrusage/sys/resource.h and GCC/Clang flags.
# Run on Linux or WSL2 (see README limits).
add_compile_options(-O3 -march=native -Wall -Wextra)

add_executable(heph
  src/main.cpp
  src/core/tensor.cpp
  src/core/sha256.cpp
  src/core/json.cpp
  src/loader/loader.cpp
  src/kernels/kernels.cpp
  src/model/transformer.cpp
  src/kv/kv_cache.cpp
  src/quant/quantize.cpp
  src/tokenizer/bpe.cpp
  src/bench/bench.cpp
)
target_include_directories(heph PRIVATE src)

add_executable(test_kernels
  tests/cpp/test_kernels.cpp
  src/core/tensor.cpp
  src/core/sha256.cpp
  src/kernels/kernels.cpp
  src/kv/kv_cache.cpp
  src/quant/quantize.cpp
)
target_include_directories(test_kernels PRIVATE src)
add_test(NAME kernel_parity COMMAND test_kernels)
"""

MODEL_DEF_H = """
// Canonical tensor names and the architecture contract for HEPHAESTUS.
//
// The engine supports EXACTLY the PROMETHEUS-NS nano decoder and nothing
// else. Hardcoding the architecture is deliberate (spec constraint): a
// fixed shape removes every generic-dispatch branch from the hot path,
// lets the GEMV kernels use compile-time-friendly strides, and keeps the
// KV-cache layout statically known. Supporting more architectures is out
// of scope by design; the manifest is still validated against these
// constants at load time and the engine refuses any disagreement.

#pragma once

namespace heph {

// nano architecture contract. The export manifest must match every value.
constexpr int kNLayers = 6;
constexpr int kNHead = 6;
constexpr int kNKvHead = 6;       // GQA group size 1 for nano
constexpr int kDModel = 384;
constexpr int kDFf = 1024;
constexpr int kDHead = kDModel / kNHead;  // 64
constexpr int kMaxSeq = 256;
constexpr int kVocab = 8000;

// Canonical tensor names, expanded for the six nano layers. The loader
// keys its weight table by exactly these strings; scripts/export_mapping.yaml
// must produce them from the PROMETHEUS-NS checkpoint state_dict.
static const char* const kCanonicalTensorNames[] = {
    "embedding.weight",
    "layer.0.attn_norm.weight", "layer.0.attn.wq.weight",
    "layer.0.attn.wk.weight",   "layer.0.attn.wv.weight",
    "layer.0.attn.wo.weight",   "layer.0.ffn_norm.weight",
    "layer.0.ffn.w_gate.weight", "layer.0.ffn.w_up.weight",
    "layer.0.ffn.w_down.weight",
    "layer.1.attn_norm.weight", "layer.1.attn.wq.weight",
    "layer.1.attn.wk.weight",   "layer.1.attn.wv.weight",
    "layer.1.attn.wo.weight",   "layer.1.ffn_norm.weight",
    "layer.1.ffn.w_gate.weight", "layer.1.ffn.w_up.weight",
    "layer.1.ffn.w_down.weight",
    "layer.2.attn_norm.weight", "layer.2.attn.wq.weight",
    "layer.2.attn.wk.weight",   "layer.2.attn.wv.weight",
    "layer.2.attn.wo.weight",   "layer.2.ffn_norm.weight",
    "layer.2.ffn.w_gate.weight", "layer.2.ffn.w_up.weight",
    "layer.2.ffn.w_down.weight",
    "layer.3.attn_norm.weight", "layer.3.attn.wq.weight",
    "layer.3.attn.wk.weight",   "layer.3.attn.wv.weight",
    "layer.3.attn.wo.weight",   "layer.3.ffn_norm.weight",
    "layer.3.ffn.w_gate.weight", "layer.3.ffn.w_up.weight",
    "layer.3.ffn.w_down.weight",
    "layer.4.attn_norm.weight", "layer.4.attn.wq.weight",
    "layer.4.attn.wk.weight",   "layer.4.attn.wv.weight",
    "layer.4.attn.wo.weight",   "layer.4.ffn_norm.weight",
    "layer.4.ffn.w_gate.weight", "layer.4.ffn.w_up.weight",
    "layer.4.ffn.w_down.weight",
    "layer.5.attn_norm.weight", "layer.5.attn.wq.weight",
    "layer.5.attn.wk.weight",   "layer.5.attn.wv.weight",
    "layer.5.attn.wo.weight",   "layer.5.ffn_norm.weight",
    "layer.5.ffn.w_gate.weight", "layer.5.ffn.w_up.weight",
    "layer.5.ffn.w_down.weight",
    "final_norm.weight",
    "lm_head.weight",  // tied to embedding.weight; verified equal at export
};
static constexpr int kCanonicalTensorCount =
    static_cast<int>(sizeof(kCanonicalTensorNames) / sizeof(kCanonicalTensorNames[0]));

}  // namespace heph
"""

TEST_KERNELS_CPP = """
// Kernel parity tests: every SIMD implementation must match its scalar
// reference — bit-identical for integer kernels, max-abs-diff <= 1e-6
// for float kernels (the scalar reference mirrors the SIMD lane
// ordering, so ISA differences are isolated from ordering differences).
// Run via ctest (make test).

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "kernels/kernels.h"
#include "model/model_def.h"
#include "quant/quantize.h"

using namespace heph;

namespace {

int g_failures = 0;

void expect_close(const char* name, const std::vector<float>& a,
                  const std::vector<float>& b, float tol) {
  if (a.size() != b.size()) {
    std::printf("FAIL %s: size %zu vs %zu\\n", name, a.size(), b.size());
    ++g_failures;
    return;
  }
  double max_diff = 0.0;
  for (size_t i = 0; i < a.size(); ++i) {
    double d = std::fabs(static_cast<double>(a[i]) - static_cast<double>(b[i]));
    if (d > max_diff) max_diff = d;
  }
  if (max_diff > tol) {
    std::printf("FAIL %s: max diff %.3g > %.3g\\n", name, max_diff, tol);
    ++g_failures;
  } else {
    std::printf("ok   %s (max diff %.3g)\\n", name, max_diff);
  }
}

void expect_true(const char* name, bool ok) {
  if (!ok) {
    std::printf("FAIL %s\\n", name);
    ++g_failures;
  } else {
    std::printf("ok   %s\\n", name);
  }
}

std::vector<float> random_vector(size_t n, unsigned seed, float scale = 0.25f) {
  std::mt19937 gen(seed);
  std::uniform_real_distribution<float> dist(-scale, scale);
  std::vector<float> v(n);
  for (auto& x : v) x = dist(gen);
  return v;
}

}  // namespace

int main() {
  const int n_out = kDFf;   // 1024
  const int n_in = kDModel; // 384

  // fp32 GEMV: scalar reference vs SIMD dispatch (AVX2, and AVX-512F
  // when the runtime reports it) with identical lane ordering.
  auto w = random_vector(static_cast<size_t>(n_out) * n_in, 1);
  auto x = random_vector(n_in, 2);
  std::vector<float> out_ref(n_out), out_simd(n_out);
  gemv_f32_scalar(out_ref.data(), w.data(), x.data(), n_out, n_in);
  gemv_f32(out_simd.data(), w.data(), x.data(), n_out, n_in);
  expect_close("gemv_f32 simd vs scalar", out_simd, out_ref, 1e-6f);

  // RMSNorm: one row, eps inside the square root.
  auto nw = random_vector(n_in, 3, 1.0f);
  std::vector<float> rms_ref(n_in), rms_simd(n_in);
  rms_norm_scalar(x.data(), rms_ref.data(), nw.data(), n_in, 1e-5f);
  rms_norm(x.data(), rms_simd.data(), nw.data(), n_in, 1e-5f);
  expect_close("rms_norm simd vs scalar", rms_simd, rms_ref, 1e-6f);

  // SwiGLU elementwise gate*up.
  auto g = random_vector(n_in, 4);
  auto u = random_vector(n_in, 5);
  std::vector<float> sw_ref(n_in), sw_simd(n_in);
  swiglu_scalar(g.data(), u.data(), sw_ref.data(), n_in);
  swiglu(g.data(), u.data(), sw_simd.data(), n_in);
  expect_close("swiglu simd vs scalar", sw_simd, sw_ref, 1e-6f);

  // RoPE: in-place rotation of a [3, kNHead, kDHead] block at offset 7.
  // Rotation must preserve the per-pair norm (half-split convention).
  auto q = random_vector(3 * kNHead * kDHead, 6);
  std::vector<float> q_ref = q, q_simd = q;
  RopeTables tables = build_rope_tables(kMaxSeq, kDHead, 10000.0f);
  apply_rope_scalar(q_ref.data(), tables, 3, 7);
  apply_rope(q_simd.data(), tables, 3, 7);
  expect_close("rope simd vs scalar", q_simd, q_ref, 1e-6f);
  double max_norm_drift = 0.0;
  for (int i = 0; i < 3 * kNHead; ++i) {
    float before = 0.0f, after = 0.0f;
    for (int j = 0; j < kDHead; ++j) {
      const float b = q[i * kDHead + j], a = q_simd[i * kDHead + j];
      before += b * b;
      after += a * a;
    }
    max_norm_drift = std::max(max_norm_drift,
                              std::fabs(static_cast<double>(std::sqrt(before) -
                                                            std::sqrt(after))));
  }
  expect_true("rope preserves per-head norm", max_norm_drift < 1e-5f);

  // int8 GEMV: integer path must be bit-exact between scalar and SIMD,
  // and track the dequantized fp32 dot within quantization error.
  std::vector<int8_t> qw(static_cast<size_t>(n_out) * n_in);
  std::vector<float> scales(n_out);
  quantize_rows_int8(w.data(), n_out, n_in, qw.data(), scales.data());
  std::vector<float> qi_ref(n_out), qi_simd(n_out);
  gemv_int8_scalar(qi_ref.data(), qw.data(), scales.data(), x.data(),
                   n_out, n_in);
  gemv_int8(qi_simd.data(), qw.data(), scales.data(), x.data(), n_out, n_in);
  expect_true("gemv_int8 bit-exact (scalar vs simd)",
              std::memcmp(qi_ref.data(), qi_simd.data(),
                          sizeof(float) * n_out) == 0);
  std::vector<float> q_deq(static_cast<size_t>(n_out) * n_in);
  for (size_t r = 0; r < static_cast<size_t>(n_out); ++r)
    for (int c = 0; c < n_in; ++c)
      q_deq[r * n_in + c] = static_cast<float>(qw[r * n_in + c]) * scales[r];
  std::vector<float> q_fp32(n_out);
  gemv_f32_scalar(q_fp32.data(), q_deq.data(), x.data(), n_out, n_in);
  expect_close("gemv_int8 vs dequantized fp32", qi_simd, q_fp32, 2e-3f);

  // int4 packing contract: weight 2i lives in the LOW nibble of byte i
  // (low-first), weight 2i+1 in the high nibble. Codes are q + 8.
  {
    std::vector<float> row = {0.3f, -0.3f};
    // scale = max_abs/7 = 0.3/7; codes: round(0.3/scale)+8 = 15,
    // round(-0.3/scale)+8 = 1 -> byte = 0x1F (low nibble 15, high 1).
    uint16_t scale_fp16 = 0;
    uint8_t byte = 0;
    quantize_rows_int4(row.data(), 1, 2, 128, &byte, &scale_fp16);
    expect_true("int4 low-nibble-first packing", byte == 0x1F);
  }
  std::vector<uint8_t> i4p((n_in + 1) / 2);
  std::vector<uint16_t> i4s((n_in + 127) / 128);
  quantize_rows_int4(w.data(), 1, n_in, 128, i4p.data(), i4s.data());
  std::vector<float> i4row(n_in);
  unpack_row_int4(i4p.data(), i4s.data(), n_in, 128, i4row.data());
  float worst_int4 = 0.0f;
  {
    // per-group scale from the fp16 storage
    float max_abs = 0.0f;
    for (int grp = 0; grp < (n_in + 127) / 128; ++grp) {
      max_abs = 0.0f;
      const int lo = grp * 128, hi = std::min(lo + 128, n_in);
      for (int c = lo; c < hi; ++c) max_abs = std::max(max_abs, std::fabs(w[c]));
      const float scale = max_abs / 7.0f;
      for (int c = lo; c < hi; ++c)
        worst_int4 = std::max(worst_int4, std::fabs(i4row[c] - w[c]) - scale * 0.5f);
    }
  }
  expect_true("int4 roundtrip within half-step", worst_int4 <= 1e-3f);

  // ternary: lookup GEMV must equal the scalar dot of the dequantized
  // row bit-exactly (same accumulation order, same lut values).
  std::vector<uint8_t> tpack((n_in + 3) / 4);
  float tscale = 0.0f;
  quantize_row_ternary(w.data(), n_in, tpack.data(), &tscale);
  std::vector<float> tern_row(n_in);
  ternary_dequant_row(tpack.data(), tscale, tern_row.data(), n_in);
  bool tern_values_ok = true;
  for (int c = 0; c < n_in; ++c) {
    const float v = tern_row[c];
    if (!(v == 0.0f || v == tscale || v == -tscale)) tern_values_ok = false;
  }
  expect_true("ternary values in {-s, 0, +s}", tern_values_ok);
  float t_ref = 0.0f, t_simd = 0.0f;
  gemv_f32_scalar(&t_ref, tern_row.data(), x.data(), 1, n_in);
  gemv_ternary_row(&t_simd, tpack.data(), tscale, x.data(), n_in);
  expect_close("gemv_ternary lookup vs scalar", {t_simd}, {t_ref}, 1e-6f);

  // softmax: stable on huge inputs and sums to one.
  {
    std::vector<float> big = {10000.0f, 10001.0f, 9999.0f};
    softmax_inplace(big.data(), static_cast<int>(big.size()));
    const float sum = big[0] + big[1] + big[2];
    expect_true("softmax stable on 1e4 inputs",
                std::fabs(sum - 1.0f) < 1e-6f && big[1] > big[0]);
  }

  if (g_failures == 0) {
    std::printf("all kernel parity tests passed\\n");
    return 0;
  }
  std::printf("%d parity test group(s) failed\\n", g_failures);
  return 1;
}
"""

PROMPTS = """# Golden prompts: one per line, blank lines and #-comments ignored.
# Chosen so the greedy argmax gaps stay far above fp32 accumulation noise
# and the tokenized lengths fall in the 64-256 window used by the logits
# tolerance test.

The ancient philosopher walked through the marble courtyard at dawn, pondering the nature of knowledge and the limits of human understanding, while the city slowly awakened beneath him and merchants prepared their stalls for the morning market.

In the depth of winter, the researchers catalogued every specimen collected during the expedition, comparing the morphological details of each sample against the historical records preserved in the university archives since the previous century.

The signal processing pipeline converts raw measurements into meaningful features: first the data is filtered to remove noise, then normalized to a common scale, and finally projected onto a lower dimensional space for efficient storage and retrieval.

Long before the printing press transformed Europe, scribes in dim monasteries copied manuscripts by candlelight, preserving the works of antiquity through generations of patient labour, and their careful hands guarded knowledge that would otherwise have vanished forever.

Every mathematical proof begins with a set of assumptions that must be stated clearly: from those axioms, the argument proceeds step by step, each deduction following from the ones before it, until the theorem stands established beyond reasonable doubt.
"""

# ---------------------------------------------------------------------------
# Contract headers: real declarations, stub bodies. Implementing a phase
# means filling the .cpp files; these interfaces are stable.
# ---------------------------------------------------------------------------

HEADERS = {}

HEADERS["src/core/tensor.h"] = """
// Minimal tensor type: row-major fp32 with 64-byte aligned storage, so
// SIMD loads never cross allocation boundaries and aligned_hint stays
// true. The engine never owns more than a handful of these.

#pragma once

#include <cstdint>
#include <vector>

namespace heph {

struct Tensor {
    int rows = 0;
    int cols = 0;
    std::vector<float> data;  // size rows*cols, row-major

    float& at(int r, int c) { return data[static_cast<size_t>(r) * cols + c]; }
    const float& at(int r, int c) const {
        return data[static_cast<size_t>(r) * cols + c];
    }
};

// 64-byte aligned buffer of n floats (posix_memalign semantics via
// operator new alignment); used by kernels that want alignment safety.
float* aligned_floats(size_t n);
void aligned_free(float* p);

}  // namespace heph
"""

HEADERS["src/core/sha256.h"] = """
// Self-contained SHA-256 (FIPS 180-4) for artifact integrity: the
// loader verifies the weights file digest against the manifest and can
// expose per-tensor digests. No third-party dependencies.

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace heph {

// Returns the lowercase hex digest of len bytes at data.
std::string sha256_hex(const uint8_t* data, size_t len);

}  // namespace heph
"""

HEADERS["src/core/json.h"] = """
// Minimal JSON reader for the safetensors header, vocab.json and
// specials.json. Supports objects, arrays, strings (with \\\\uXXXX and
// standard escapes), doubles, booleans and null. The engine never
// writes JSON except the flat bench report, which is formatted by hand.

#pragma once

#include <map>
#include <string>
#include <variant>
#include <vector>

namespace heph {

struct JsonValue;

using JsonObj = std::map<std::string, JsonValue>;  // key order preserved by map

struct JsonValue {
    std::variant<std::nullptr_t, bool, double, std::string,
                 std::vector<JsonValue>, JsonObj> v;

    bool is_null() const { return std::holds_alternative<std::nullptr_t>(v); }
    bool is_bool() const { return std::holds_alternative<bool>(v); }
    bool is_number() const { return std::holds_alternative<double>(v); }
    bool is_string() const { return std::holds_alternative<std::string>(v); }
    bool is_array() const { return std::holds_alternative<std::vector<JsonValue>>(v); }
    bool is_object() const { return std::holds_alternative<JsonObj>(v); }

    double number() const { return std::get<double>(v); }
    const std::string& string() const { return std::get<std::string>(v); }
    const std::vector<JsonValue>& array() const {
        return std::get<std::vector<JsonValue>>(v);
    }
    const JsonObj& object() const { return std::get<JsonObj>(v); }
};

// Parses text; throws std::runtime_error with a position hint on error.
JsonValue json_parse(const std::string& text);

}  // namespace heph
"""

HEADERS["src/loader/loader.h"] = """
// Safetensors loader and manifest/tensors.tsv parsing.
//
// The loader memory-maps artifacts/nano_fp32.safetensors, parses the
// JSON header, and exposes fp32 tensor views at the absolute offsets
// recorded in tensors.tsv (header + 8-byte length prefix). Integrity:
// the whole-file SHA-256 must equal manifest.weights_sha256 and every
// tensor range must match the header's data_offsets; per-tensor SHA-256
// digests are available for cross-checks against fixtures.

#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace heph {

struct Manifest {
    int n_layer = 0;
    int n_head = 0;
    int n_kv_head = 0;
    int d_model = 0;
    int d_ff = 0;
    int max_seq = 0;
    int vocab = 0;
    int d_head = 0;
    float rms_norm_eps = 0.0f;
    float rope_theta = 0.0f;
    std::string weights_sha256;
    std::string weights_file;
};

struct TensorEntry {
    std::string name;
    std::string shape;   // "rows x cols" as written in tensors.tsv
    std::string dtype;
    uint64_t offset = 0;  // absolute file offset
    uint64_t bytes = 0;
};

// Parses model_manifest.txt ("key = value" lines, arch as the Python
// dict repr). Throws std::runtime_error on missing keys or any
// disagreement with the nano contract (src/model/model_def.h).
Manifest load_manifest(const std::string& path);

// Parses tensors.tsv (name, shape, dtype, absolute offset, bytes).
std::vector<TensorEntry> load_tensors_table(const std::string& path);

class SafetensorsFile {
  public:
    explicit SafetensorsFile(const std::string& path);
    ~SafetensorsFile();
    SafetensorsFile(const SafetensorsFile&) = delete;
    SafetensorsFile& operator=(const SafetensorsFile&) = delete;

    // Whole-file digest; compared against Manifest::weights_sha256.
    std::string file_sha256() const;
    // Per-tensor digest over the tensor's byte range.
    std::string tensor_sha256(const std::string& name) const;
    // Read-only fp32 view of one tensor's data (row-major).
    const float* data(const std::string& name) const;
    std::vector<uint64_t> shape(const std::string& name) const;

  private:
    struct Impl;
    Impl* impl_;
};

}  // namespace heph
"""

HEADERS["src/kernels/kernels.h"] = """
// Compute kernels: a plain scalar reference plus SIMD paths (AVX2 and,
// when runtime detection reports it, AVX-512F). Parity contract
// (tests/cpp/test_kernels.cpp): the scalar reference and every SIMD
// path accumulate in the SAME lane ordering, so float results agree
// bit-for-bit and integer results are bit-identical; the test tolerance
// is 1e-6 for floats. Projections are row-major [out, in] and applied
// as out[r] = sum_c W[r*n_in+c] * x[c] (x @ W.T semantics, matching
// tests/golden/reference_model.py).

#pragma once

#include <cstdint>
#include <vector>

namespace heph {

// fp32 GEMV, W row-major [n_out, n_in].
void gemv_f32_scalar(float* out, const float* w, const float* x,
                     int n_out, int n_in);
void gemv_f32(float* out, const float* w, const float* x,
              int n_out, int n_in);  // SIMD dispatch: AVX-512F > AVX2 > scalar

// RMSNorm over one row: out = x / sqrt(mean(x^2) + eps) * w.
void rms_norm_scalar(const float* x, float* out, const float* w,
                     int n, float eps);
void rms_norm(const float* x, float* out, const float* w, int n, float eps);

// SwiGLU gate elementwise: out = silu(g) * u with silu(z) = z*sigmoid(z).
void swiglu_scalar(const float* g, const float* u, float* out, int n);
void swiglu(const float* g, const float* u, float* out, int n);

// Precomputed rotary tables for positions [0, max_seq), half-split
// convention: inv_freq = 1/theta^(2i/d_head), tables of shape
// [max_seq, d_head/2] (cos, sin).
struct RopeTables {
    int max_seq = 0;
    int half = 0;
    std::vector<float> cos;
    std::vector<float> sin;
};

RopeTables build_rope_tables(int max_seq, int d_head, float theta);

// In-place rotation of x laid out [seq, n_head, d_head]; position of
// row i is pos_offset + i. Only the NEW tokens are ever rotated by the
// caller (rotating a whole accumulated buffer would over-rotate cached
// positions).
void apply_rope_scalar(float* x, const RopeTables& t, int seq, int pos_offset);
void apply_rope(float* x, const RopeTables& t, int seq, int pos_offset);

// Numerically stable softmax in place over n values.
void softmax_inplace(float* v, int n);

}  // namespace heph
"""

HEADERS["src/kv/kv_cache.h"] = """
// KV caches: fp32 reference cache and the quantized paged cache.
//
// PagedInt8KvCache implements the phase-5 contract: pages of
// kv.page_tokens (16) tokens, a block table mapping logical page index
// to a physical slot, no external fragmentation, and int8 quantization
// per (layer, head, token) with scale = max_abs/127. Attention over a
// partially filled last page MUST ignore the uninitialized slots: the
// cache exposes valid_len() per page and the model masks those columns
// before the softmax (garbage entering the softmax would corrupt the
// distribution). Only the incoming token's k/v are rotated before
// append; cached rows keep their original rotation.

#pragma once

#include <cstdint>
#include <vector>

namespace heph {

class KvCacheFp32 {
  public:
    KvCacheFp32(int n_layers, int n_kv_head, int d_head, int max_seq);

    // Appends one rotated (k, v) row pair for a layer at position pos.
    void append(int layer, int pos, const float* k_row, const float* v_row);
    // Pointers to the cached row of (layer, head) at position pos.
    const float* k_row(int layer, int head, int pos) const;
    const float* v_row(int layer, int head, int pos) const;

  private:
    int n_layers_;
    int n_kv_head_;
    int d_head_;
    int max_seq_;
    // [layer][head][pos][d]
    std::vector<float> k_;
    std::vector<float> v_;
};

class PagedInt8KvCache {
  public:
    static constexpr int kPageTokens = 16;

    PagedInt8KvCache(int n_layers, int n_kv_head, int d_head, int max_pages);

    // Quantizes and appends one row pair; allocates a new page from the
    // pool when pos crosses a page boundary (block table append).
    void append(int layer, int pos, const float* k_row, const float* v_row);

    int valid_len(int page) const;  // tokens filled in a logical page
    int n_pages(int layer) const;

    // Dequantized view of one (layer, head, page) token row into out.
    void k_row(int layer, int head, int page, int slot, float* out) const;
    void v_row(int layer, int head, int page, int slot, float* out) const;

  private:
    int n_layers_;
    int n_kv_head_;
    int d_head_;
    // physical pool: [layer][head][page][slot][d], int8
    std::vector<int8_t> k_pool_;
    std::vector<int8_t> v_pool_;
    // per (layer, head, page, slot) scale
    std::vector<float> k_scale_;
    std::vector<float> v_scale_;
    std::vector<int> valid_;          // tokens filled per (layer, page)
    std::vector<int> block_table_;    // logical page -> physical page
};

}  // namespace heph
"""

HEADERS["src/quant/quantize.h"] = """
// Progressive weight quantization: int8 / int4 / ternary, each mode
// measurable and revertible (the fp32 weights always stay the source).
//
// Formats (writer and reader share this contract):
//   int8 per output channel, symmetric: scale[r] = max_abs(row)/127,
//       q = round(w/scale) clamped to [-127, 127].
//   int4 groupwise (group along the input dim, fp16 scale per group):
//       scale = max_abs(group)/7, q = round(w/scale) clamped [-8, 7],
//       stored as code = q + 8 in [0, 15]. Packing is LOW NIBBLE
//       FIRST: weight 2i -> low nibble of byte i, weight 2i+1 -> high
//       nibble. The unpacking kernel mirrors exactly this order.
//   ternary 1.58-bit per tensor: scale = mean(|w|); code 0 -> 0,
//       1 -> +1, 2 -> -1 (3 reserved); 2-bit codes packed 4 per byte,
//       first weight in the LOWEST 2 bits. GEMV uses a lookup table
//       lut[code] = {0, +scale, -scale, 0}.
//
// Accumulation contract: integer products are ALWAYS widened to 32-bit
// accumulators (a d_model=384 row overflows 16-bit lanes).

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "loader/loader.h"

namespace heph {

enum class QuantMode { Fp32, Int8, Int4, Ternary };

QuantMode parse_quant_mode(const std::string& name);
const char* quant_mode_name(QuantMode mode);

// Row int8 quantization + GEMV (dequantizing dot with per-row scale).
void quantize_rows_int8(const float* w, int n_out, int n_in,
                        int8_t* q, float* scales);
void gemv_int8_scalar(float* out, const int8_t* q, const float* scales,
                      const float* x, int n_out, int n_in);
void gemv_int8(float* out, const int8_t* q, const float* scales,
               const float* x, int n_out, int n_in);

// Groupwise int4 quantization + GEMV (low-nibble-first packing).
void quantize_rows_int4(const float* w, int n_out, int n_in, int group,
                        uint8_t* packed, uint16_t* scales_fp16);
void unpack_row_int4(const uint8_t* packed, const uint16_t* scales_fp16,
                     int n_in, int group, float* out);
void gemv_int4_scalar(float* out, const uint8_t* packed,
                      const uint16_t* scales_fp16, const float* x,
                      int n_out, int n_in, int group);
void gemv_int4(float* out, const uint8_t* packed, const uint16_t* scales_fp16,
               const float* x, int n_out, int n_in, int group);

// Ternary quantization + lookup GEMV (one row at a time).
void quantize_row_ternary(const float* w, int n, uint8_t* packed, float* scale);
void ternary_dequant_row(const uint8_t* packed, float scale, float* out, int n);
void gemv_ternary_row(float* out, const uint8_t* packed, float scale,
                      const float* x, int n_in);

// In-process quantized weight set used by the bench modes: built from
// the fp32 weights, one QuantizedLayer per projection.
struct QuantizedMat {
    std::vector<uint8_t> packed;   // int8 bytes / int4 nibbles / ternary codes
    std::vector<float> scales;     // per row (int8/ternary) or per group (int4)
    std::vector<uint16_t> scales_fp16;
    int n_out = 0;
    int n_in = 0;
    int group = 0;
};

QuantizedMat quantize_matrix(const float* w, int n_out, int n_in,
                             QuantMode mode, int group);

}  // namespace heph
"""

HEADERS["src/model/transformer.h"] = """
// The transformer: fp32 forward matching tests/golden/reference_model.py
// step for step (pre-norm blocks, half-split RoPE applied only to the
// incoming tokens, causal attention scaled 1/sqrt(d_head) over the
// per-layer cache, SwiGLU MLP, final norm, tied-embedding head), greedy
// generation and logits dumping for the golden tests.

#pragma once

#include <string>
#include <vector>

#include "kv/kv_cache.h"
#include "loader/loader.h"
#include "quant/quantize.h"

namespace heph {

enum class KvMode { Fp32, Int8Paged };

class Transformer {
  public:
    Transformer(const Manifest& manifest, const float* embedding,
                const std::vector<const float*>& layer_weights,
                const float* final_norm);

    // Logits of the LAST token of `tokens` placed at `pos_offset`,
    // updating the per-layer caches (past + current chunk).
    void logits_last(const std::vector<int>& tokens, int pos_offset);

    // Greedy continuation: exact argmax (lowest index on ties), no
    // special-token injection, stops before eos_id when not negative.
    std::vector<int> greedy_generate(const std::vector<int>& prompt,
                                     int n_new, int eos_id);

    const float* last_logits() const;
    int vocab() const;

    void reset_cache();
    void set_kv_mode(KvMode mode);

  private:
    struct Impl;
    Impl* impl_;
};

// Sampling helpers used by non-greedy generation (temperature, top-p).
int sample_token(const float* logits, int vocab, float temperature,
                 float top_p, unsigned* rng_state);

}  // namespace heph
"""

HEADERS["src/tokenizer/bpe.h"] = """
// Byte-level BPE runtime (reader, not trainer): loads vocab.json +
// merges.txt + specials.json produced by scripts/export_tokenizer.py.
//
// Encode pipeline (parity-tested against Hugging Face by
// tests/golden/test_tokenize_parity.py):
//   1. GPT-2 byte -> unicode alphabet map BEFORE vocab lookup
//      ('Ġ' = space, byte 0x20 -> U+0120);
//   2. the ByteLevel pre-split BEFORE merges — implemented as a manual
//      scanner (std::regex has no Unicode property support and third
//      party regex libraries are out of scope);
//   3. BPE merges by rank: always merge the pair with the LOWEST rank
//      in merges.txt, never the first pair found left-to-right;
//   4. encode never injects special tokens and never emits <|unk|>:
//      byte-level coverage is total, so unk firing means the byte map
//      is broken.
//
// Multi-byte UTF-8 caution: mapped characters like 'Ġ' occupy two bytes
// in UTF-8 (0xC4 0xA0); all merge/vocab work happens on codepoint
// sequences, never on raw std::string indices.

#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace heph {

class BpeTokenizer {
  public:
    void load(const std::string& dir);  // vocab.json, merges.txt, specials.json

    std::vector<int> encode(const std::string& text) const;
    std::string decode(const std::vector<int>& ids) const;

    int bos_id() const;
    int eos_id() const;
    int unk_id() const;
    int vocab_size() const;

  private:
    std::unordered_map<std::string, int> vocab_;   // unicode-space piece -> id
    std::vector<std::string> id_to_piece_;
    std::unordered_map<uint64_t, int> merge_rank_; // (left, right) packed -> rank
    std::vector<int> byte_token_;                  // 256 mapped base tokens
    int bos_ = -1;
    int eos_ = -1;
    int unk_ = -1;
};

}  // namespace heph
"""

HEADERS["src/bench/bench.h"] = """
// Benchmark suite: {fp32, int8, int4, ternary} x {decode tokens/s
// (batch 1), TTFT, peak RSS, perplexity}. Real measured values only:
// every number written here comes from a stopwatch or getrusage, never
// from a model. Methodology (docs/benchmark.md): warmup prompts are
// discarded, medians are over bench.prompts prompts, prompt prefill is
// excluded from the decode rate and reported as TTFT, peak RSS is the
// process lifetime peak from getrusage(RUSAGE_SELF).ru_maxrss. The
// optional --holdout run adds ppl + ppl_tokens as mean NLL over
// non-overlapping max_seq windows of the frozen PROMETHEUS-NS holdout.

#pragma once

#include <string>
#include <vector>

#include "loader/loader.h"
#include "quant/quantize.h"

namespace heph {

struct BenchResult {
    std::string mode;
    double tokens_per_s_decode = 0.0;
    double ttft_s = 0.0;
    double peak_rss_mb = 0.0;
    std::vector<double> per_prompt_ms;
    double ppl = 0.0;      // only with --holdout
    long ppl_tokens = 0;   // only with --holdout
    bool has_ppl = false;
};

struct PromptSource {
    std::vector<std::string> lines;  // already filtered (non-empty, no #)
};

PromptSource load_bench_prompts(const std::string& path, int n);
BenchResult run_bench(const Manifest& manifest, const std::string& weights_path,
                      const std::string& tokenizer_dir, QuantMode mode,
                      int n_prompts, int warmup, int max_new_tokens,
                      const std::string& holdout_dir);
void write_bench_json(const std::string& path, const BenchResult& r);

}  // namespace heph
"""

STUBS = {
    "src/core/tensor.cpp": "aligned buffer helpers",
    "src/core/sha256.cpp": "SHA-256 implementation",
    "src/core/json.cpp": "minimal JSON parser",
    "src/loader/loader.cpp": "manifest + tensors.tsv + safetensors mmap loader",
    "src/kernels/kernels.cpp": "scalar reference + AVX2/AVX-512 kernels",
    "src/model/transformer.cpp": "forward, greedy generation, logits dump",
    "src/kv/kv_cache.cpp": "fp32 KV cache and paged int8 KV cache",
    "src/quant/quantize.cpp": "int8/int4/ternary quantization and GEMV kernels",
    "src/tokenizer/bpe.cpp": "byte-level BPE runtime",
    "src/bench/bench.cpp": "benchmark suite and JSON report",
    "src/main.cpp": "CLI dispatch: golden|generate|tokenize|quantize|bench",
}

# Throwing definitions for every declared symbol: the skeleton state must
# compile, link and fail at RUNTIME (std::runtime_error) until each phase
# is implemented, exactly as the handoff contract requires.
STUB_CPP = {}

STUB_CPP["src/core/tensor.cpp"] = """
#include "core/tensor.h"
#include <stdexcept>
namespace heph {
float* aligned_floats(size_t n) {
  throw std::runtime_error("heph: core/tensor not implemented (phase 0 stub)");
}
void aligned_free(float* p) {
  throw std::runtime_error("heph: core/tensor not implemented (phase 0 stub)");
}
}  // namespace heph
"""

STUB_CPP["src/core/sha256.cpp"] = """
#include "core/sha256.h"
#include <stdexcept>
namespace heph {
std::string sha256_hex(const uint8_t*, size_t) {
  throw std::runtime_error("heph: core/sha256 not implemented (phase 0 stub)");
}
}  // namespace heph
"""

STUB_CPP["src/core/json.cpp"] = """
#include "core/json.h"
#include <stdexcept>
namespace heph {
JsonValue json_parse(const std::string&) {
  throw std::runtime_error("heph: core/json not implemented (phase 0 stub)");
}
}  // namespace heph
"""

STUB_CPP["src/loader/loader.cpp"] = """
#include "loader/loader.h"
#include <stdexcept>
namespace heph {
Manifest load_manifest(const std::string&) {
  throw std::runtime_error("heph: loader not implemented (phase 1)");
}
std::vector<TensorEntry> load_tensors_table(const std::string&) {
  throw std::runtime_error("heph: loader not implemented (phase 1)");
}
struct SafetensorsFile::Impl {};
SafetensorsFile::SafetensorsFile(const std::string&) {
  throw std::runtime_error("heph: loader not implemented (phase 1)");
}
SafetensorsFile::~SafetensorsFile() = default;
std::string SafetensorsFile::file_sha256() const {
  throw std::runtime_error("heph: loader not implemented (phase 1)");
}
std::string SafetensorsFile::tensor_sha256(const std::string&) const {
  throw std::runtime_error("heph: loader not implemented (phase 1)");
}
const float* SafetensorsFile::data(const std::string&) const {
  throw std::runtime_error("heph: loader not implemented (phase 1)");
}
std::vector<uint64_t> SafetensorsFile::shape(const std::string&) const {
  throw std::runtime_error("heph: loader not implemented (phase 1)");
}
}  // namespace heph
"""

STUB_CPP["src/kernels/kernels.cpp"] = """
#include "kernels/kernels.h"
#include <stdexcept>
namespace heph {
void gemv_f32_scalar(float*, const float*, const float*, int, int) {
  throw std::runtime_error("heph: kernels not implemented (phase 2)");
}
void gemv_f32(float*, const float*, const float*, int, int) {
  throw std::runtime_error("heph: kernels not implemented (phase 2)");
}
void rms_norm_scalar(const float*, float*, const float*, int, float) {
  throw std::runtime_error("heph: kernels not implemented (phase 2)");
}
void rms_norm(const float*, float*, const float*, int, float) {
  throw std::runtime_error("heph: kernels not implemented (phase 2)");
}
void swiglu_scalar(const float*, const float*, float*, int) {
  throw std::runtime_error("heph: kernels not implemented (phase 2)");
}
void swiglu(const float*, const float*, float*, int) {
  throw std::runtime_error("heph: kernels not implemented (phase 2)");
}
RopeTables build_rope_tables(int, int, float) {
  throw std::runtime_error("heph: kernels not implemented (phase 2)");
}
void apply_rope_scalar(float*, const RopeTables&, int, int) {
  throw std::runtime_error("heph: kernels not implemented (phase 2)");
}
void apply_rope(float*, const RopeTables&, int, int) {
  throw std::runtime_error("heph: kernels not implemented (phase 2)");
}
void softmax_inplace(float*, int) {
  throw std::runtime_error("heph: kernels not implemented (phase 2)");
}
}  // namespace heph
"""

STUB_CPP["src/kv/kv_cache.cpp"] = """
#include "kv/kv_cache.h"
#include <stdexcept>
namespace heph {
KvCacheFp32::KvCacheFp32(int, int, int, int) {
  throw std::runtime_error("heph: kv_cache not implemented (phase 3/5)");
}
void KvCacheFp32::append(int, int, const float*, const float*) {
  throw std::runtime_error("heph: kv_cache not implemented (phase 3/5)");
}
const float* KvCacheFp32::k_row(int, int, int) const {
  throw std::runtime_error("heph: kv_cache not implemented (phase 3/5)");
}
const float* KvCacheFp32::v_row(int, int, int) const {
  throw std::runtime_error("heph: kv_cache not implemented (phase 3/5)");
}
PagedInt8KvCache::PagedInt8KvCache(int, int, int, int) {
  throw std::runtime_error("heph: kv_cache not implemented (phase 5)");
}
void PagedInt8KvCache::append(int, int, const float*, const float*) {
  throw std::runtime_error("heph: kv_cache not implemented (phase 5)");
}
int PagedInt8KvCache::valid_len(int) const {
  throw std::runtime_error("heph: kv_cache not implemented (phase 5)");
}
int PagedInt8KvCache::n_pages(int) const {
  throw std::runtime_error("heph: kv_cache not implemented (phase 5)");
}
void PagedInt8KvCache::k_row(int, int, int, int, float*) const {
  throw std::runtime_error("heph: kv_cache not implemented (phase 5)");
}
void PagedInt8KvCache::v_row(int, int, int, int, float*) const {
  throw std::runtime_error("heph: kv_cache not implemented (phase 5)");
}
}  // namespace heph
"""

STUB_CPP["src/quant/quantize.cpp"] = """
#include "quant/quantize.h"
#include <stdexcept>
namespace heph {
QuantMode parse_quant_mode(const std::string&) {
  throw std::runtime_error("heph: quantize not implemented (phase 4)");
}
const char* quant_mode_name(QuantMode) {
  throw std::runtime_error("heph: quantize not implemented (phase 4)");
}
void quantize_rows_int8(const float*, int, int, int8_t*, float*) {
  throw std::runtime_error("heph: quantize not implemented (phase 4)");
}
void gemv_int8_scalar(float*, const int8_t*, const float*, const float*, int, int) {
  throw std::runtime_error("heph: quantize not implemented (phase 4)");
}
void gemv_int8(float*, const int8_t*, const float*, const float*, int, int) {
  throw std::runtime_error("heph: quantize not implemented (phase 4)");
}
void quantize_rows_int4(const float*, int, int, int, uint8_t*, uint16_t*) {
  throw std::runtime_error("heph: quantize not implemented (phase 4)");
}
void unpack_row_int4(const uint8_t*, const uint16_t*, int, int, float*) {
  throw std::runtime_error("heph: quantize not implemented (phase 4)");
}
void gemv_int4_scalar(float*, const uint8_t*, const uint16_t*, const float*, int, int, int) {
  throw std::runtime_error("heph: quantize not implemented (phase 4)");
}
void gemv_int4(float*, const uint8_t*, const uint16_t*, const float*, int, int, int) {
  throw std::runtime_error("heph: quantize not implemented (phase 4)");
}
void quantize_row_ternary(const float*, int, uint8_t*, float*) {
  throw std::runtime_error("heph: quantize not implemented (phase 4)");
}
void ternary_dequant_row(const uint8_t*, float, float*, int) {
  throw std::runtime_error("heph: quantize not implemented (phase 4)");
}
void gemv_ternary_row(float*, const uint8_t*, float, const float*, int) {
  throw std::runtime_error("heph: quantize not implemented (phase 4)");
}
QuantizedMat quantize_matrix(const float*, int, int, QuantMode, int) {
  throw std::runtime_error("heph: quantize not implemented (phase 4)");
}
}  // namespace heph
"""

STUB_CPP["src/model/transformer.cpp"] = """
#include "model/transformer.h"
#include <stdexcept>
namespace heph {
struct Transformer::Impl {};
Transformer::Transformer(const Manifest&, const float*,
                         const std::vector<const float*>&, const float*) {
  throw std::runtime_error("heph: transformer not implemented (phase 2/3)");
}
void Transformer::logits_last(const std::vector<int>&, int) {
  throw std::runtime_error("heph: transformer not implemented (phase 2/3)");
}
std::vector<int> Transformer::greedy_generate(const std::vector<int>&,
                                              int, int) {
  throw std::runtime_error("heph: transformer not implemented (phase 2/3)");
}
const float* Transformer::last_logits() const {
  throw std::runtime_error("heph: transformer not implemented (phase 2/3)");
}
int Transformer::vocab() const {
  throw std::runtime_error("heph: transformer not implemented (phase 2/3)");
}
void Transformer::reset_cache() {
  throw std::runtime_error("heph: transformer not implemented (phase 2/3)");
}
void Transformer::set_kv_mode(KvMode) {
  throw std::runtime_error("heph: transformer not implemented (phase 2/3)");
}
int sample_token(const float*, int, float, float, unsigned*) {
  throw std::runtime_error("heph: transformer not implemented (phase 3)");
}
}  // namespace heph
"""

STUB_CPP["src/tokenizer/bpe.cpp"] = """
#include "tokenizer/bpe.h"
#include <stdexcept>
namespace heph {
void BpeTokenizer::load(const std::string&) {
  throw std::runtime_error("heph: tokenizer not implemented (phase 2)");
}
std::vector<int> BpeTokenizer::encode(const std::string&) const {
  throw std::runtime_error("heph: tokenizer not implemented (phase 2)");
}
std::string BpeTokenizer::decode(const std::vector<int>&) const {
  throw std::runtime_error("heph: tokenizer not implemented (phase 2)");
}
int BpeTokenizer::bos_id() const {
  throw std::runtime_error("heph: tokenizer not implemented (phase 2)");
}
int BpeTokenizer::eos_id() const {
  throw std::runtime_error("heph: tokenizer not implemented (phase 2)");
}
int BpeTokenizer::unk_id() const {
  throw std::runtime_error("heph: tokenizer not implemented (phase 2)");
}
int BpeTokenizer::vocab_size() const {
  throw std::runtime_error("heph: tokenizer not implemented (phase 2)");
}
}  // namespace heph
"""

STUB_CPP["src/bench/bench.cpp"] = """
#include "bench/bench.h"
#include <stdexcept>
namespace heph {
PromptSource load_bench_prompts(const std::string&, int) {
  throw std::runtime_error("heph: bench not implemented (phase 6)");
}
BenchResult run_bench(const Manifest&, const std::string&,
                      const std::string&, QuantMode, int, int, int,
                      const std::string&) {
  throw std::runtime_error("heph: bench not implemented (phase 6)");
}
void write_bench_json(const std::string&, const BenchResult&) {
  throw std::runtime_error("heph: bench not implemented (phase 6)");
}
}  // namespace heph
"""

STUB_CPP["src/main.cpp"] = """
// CLI dispatch: golden|generate|tokenize|quantize|bench (flags fixed in
// the Makefile contract comments).
#include <iostream>
#include <stdexcept>
#include <string>

int main(int argc, char** argv) {
  const std::string cmd = argc > 1 ? argv[1] : "";
  std::cerr << "heph: subcommand '" << cmd
            << "' not implemented (phase 0 stub)\\n";
  return 2;
}
"""


def cpp_stub(contract: str, guard: str) -> str:
    lines = [
        "// Contract stub: compiles and links, throws until implemented.",
        f"// {contract}",
        "",
    ]
    if guard:
        lines += [f"#ifndef {guard}", f"#define {guard}", ""]
    return "\n".join(lines)


def cpp_stub_cpp(rel: str, contract: str) -> str:
    return (
        "// Contract stub: compiles and links, throws until implemented.\n"
        f"// {rel}: {contract}\n"
        "#include <stdexcept>\n"
        "\n"
        "// The real implementation replaces this translation unit in its\n"
        "// phase; the declared interfaces (see the matching .h) are stable.\n"
        "namespace heph {\n"
        "\n"
        "static void ensure_not_implemented_stub(const char* unit) {\n"
        "  (void)unit;\n"
        "}\n"
        "\n"
        "}  // namespace heph\n"
    )


def main() -> None:
    created, kept = [], []
    for rel in PLAIN_DIRS:
        (ROOT / rel).mkdir(parents=True, exist_ok=True)

    def write(rel: str, content: str, binary: bool = False) -> None:
        target = ROOT / rel
        if target.exists():
            kept.append(rel)
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content.encode("utf-8"))
        created.append(rel)

    write("CMakeLists.txt", CMAKELISTS.strip() + "\n")
    write("src/model/model_def.h", MODEL_DEF_H.strip() + "\n")
    write("tests/cpp/test_kernels.cpp", TEST_KERNELS_CPP.strip() + "\n")
    write("tests/golden/prompts.txt", PROMPTS)

    for rel, header in HEADERS.items():
        write(rel, header.strip() + "\n")

    for rel, body in STUB_CPP.items():
        write(rel, body.strip() + "\n")

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
    for rel in anchors:
        ok = (ROOT / rel).is_file()
        print(f"  {'ok' if ok else '!!'} {rel}: "
              f"{'present' if ok else 'MISSING: copy it from the specification package'}")

    print("\nnext: make setup && make configure && make build && make test")


if __name__ == "__main__":
    main()
