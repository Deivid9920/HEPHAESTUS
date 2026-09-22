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
