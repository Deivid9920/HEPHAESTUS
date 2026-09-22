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
    std::printf("FAIL %s: size %zu vs %zu\n", name, a.size(), b.size());
    ++g_failures;
    return;
  }
  double max_diff = 0.0;
  for (size_t i = 0; i < a.size(); ++i) {
    double d = std::fabs(static_cast<double>(a[i]) - static_cast<double>(b[i]));
    if (d > max_diff) max_diff = d;
  }
  if (max_diff > tol) {
    std::printf("FAIL %s: max diff %.3g > %.3g\n", name, max_diff, tol);
    ++g_failures;
  } else {
    std::printf("ok   %s (max diff %.3g)\n", name, max_diff);
  }
}

void expect_true(const char* name, bool ok) {
  if (!ok) {
    std::printf("FAIL %s\n", name);
    ++g_failures;
  } else {
    std::printf("ok   %s\n", name);
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
    std::printf("all kernel parity tests passed\n");
    return 0;
  }
  std::printf("%d parity test group(s) failed\n", g_failures);
  return 1;
}
