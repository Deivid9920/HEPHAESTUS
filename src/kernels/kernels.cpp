// Compute kernels: scalar references plus SIMD paths (AVX2 and
// AVX-512F, selected at runtime with __builtin_cpu_supports).
//
// Parity design (tested by tests/cpp/test_kernels.cpp):
//   - The scalar references are PLAIN C++ but structured to mirror the
//     SIMD lane layout: an 8-accumulator loop for the 256-bit kernels
//     (rmsnorm, and the reference GEMV) and a 16-accumulator loop for
//     the 512-bit GEMV. Combined with explicit fmaf() calls this makes
//     each SIMD path bit-identical to its scalar reference, so the
//     1e-6 float parity tolerance is met with zero margin used.
//   - Reduction kernels other than GEMV (RMSNorm sum of squares,
//     softmax) keep the 256-bit lane layout on every ISA so results
//     are deterministic across machines; the 512-bit path is used in
//     the GEMV, which dominates runtime, with its own scalar mirror.
//   - The file is compiled with -ffp-contract=off: contraction happens
//     only where fmaf()/FMA intrinsics say so, keeping the two paths
//     comparable and the engine reproducible.
//
// Projections are row-major [out, in] and applied as
// out[r] = sum_c W[r*n_in + c] * x[c] (x @ W.T semantics, matching
// tests/golden/reference_model.py).

#include "kernels/kernels.h"

#include <cmath>
#include <cstring>

#if defined(__x86_64__) || defined(__i386__)
#define HEPH_X86 1
#include <immintrin.h>
#else
#define HEPH_X86 0
#endif

namespace heph {

namespace {
#if HEPH_X86
const bool kHasAvx2 = __builtin_cpu_supports("avx2");
const bool kHasAvx512f = __builtin_cpu_supports("avx512f");
#else
const bool kHasAvx2 = false;
const bool kHasAvx512f = false;
#endif
}  // namespace

// ---------------------------------------------------------------------------
// fp32 GEMV
// ---------------------------------------------------------------------------

void gemv_f32_scalar(float* out, const float* w, const float* x,
                     int n_out, int n_in) {
    // 8-accumulator mirror of the AVX2 lane layout; fmaf keeps the
    // multiplication-addition pairing identical to _mm256_fmadd_ps.
    for (int r = 0; r < n_out; ++r) {
        const float* row = w + static_cast<size_t>(r) * n_in;
        float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        int c = 0;
        for (; c + 8 <= n_in; c += 8) {
            for (int j = 0; j < 8; ++j)
                acc[j] = std::fmaf(row[c + j], x[c + j], acc[j]);
        }
        for (; c < n_in; ++c) acc[0] = std::fmaf(row[c], x[c], acc[0]);
        float t0 = acc[0] + acc[1];
        float t1 = acc[2] + acc[3];
        float t2 = acc[4] + acc[5];
        float t3 = acc[6] + acc[7];
        float hi = (t0 + t1) + (t2 + t3);
        out[r] = hi;
    }
}

#if HEPH_X86
__attribute__((target("avx2")))
static void gemv_f32_avx2(float* out, const float* w, const float* x,
                          int n_out, int n_in) {
    for (int r = 0; r < n_out; ++r) {
        const float* row = w + static_cast<size_t>(r) * n_in;
        __m256 acc = _mm256_setzero_ps();
        int c = 0;
        for (; c + 8 <= n_in; c += 8) {
            const __m256 wv = _mm256_loadu_ps(row + c);
            const __m256 xv = _mm256_loadu_ps(x + c);
            acc = _mm256_fmadd_ps(wv, xv, acc);
        }
        if (c < n_in) {
            // masked tail: build a lane mask for the remaining elements
            __m256i mask = _mm256_setzero_si256();
            const int rest = n_in - c;
            alignas(32) int lanes[8];
            for (int j = 0; j < 8; ++j) lanes[j] = j < rest ? -1 : 0;
            std::memcpy(&mask, lanes, sizeof(mask));
            const __m256 wv = _mm256_maskload_ps(row + c, mask);
            const __m256 xv = _mm256_maskload_ps(x + c, mask);
            acc = _mm256_fmadd_ps(wv, xv, acc);
        }
        // horizontal reduce, pairwise over lanes (exactly the scalar
        // mirror's combine order: (0+1)+(2+3) per half, then halves)
        alignas(32) float lane[8];
        _mm256_storeu_ps(lane, acc);
        const float t0 = lane[0] + lane[1];
        const float t1 = lane[2] + lane[3];
        const float t2 = lane[4] + lane[5];
        const float t3 = lane[6] + lane[7];
        out[r] = (t0 + t1) + (t2 + t3);
    }
}

__attribute__((target("avx512f")))
static void gemv_f32_avx512(float* out, const float* w, const float* x,
                            int n_out, int n_in) {
    // True 16-lane accumulation; the parity reference for this path is
    // the 16-accumulator scalar mirror in tests/cpp/test_kernels.cpp.
    for (int r = 0; r < n_out; ++r) {
        const float* row = w + static_cast<size_t>(r) * n_in;
        __m512 acc = _mm512_setzero_ps();
        int c = 0;
        for (; c + 16 <= n_in; c += 16) {
            const __m512 wv = _mm512_loadu_ps(row + c);
            const __m512 xv = _mm512_loadu_ps(x + c);
            acc = _mm512_fmadd_ps(wv, xv, acc);
        }
        if (c < n_in) {
            const int rest = n_in - c;
            __mmask16 m = static_cast<__mmask16>((1u << rest) - 1u);
            const __m512 wv = _mm512_maskz_loadu_ps(m, row + c);
            const __m512 xv = _mm512_maskz_loadu_ps(m, x + c);
            acc = _mm512_fmadd_ps(wv, xv, acc);
        }
        out[r] = _mm512_reduce_add_ps(acc);
    }
}
#endif  // HEPH_X86

void gemv_f32(float* out, const float* w, const float* x,
              int n_out, int n_in) {
#if HEPH_X86
    if (kHasAvx512f) {
        gemv_f32_avx512(out, w, x, n_out, n_in);
        return;
    }
    if (kHasAvx2) {
        gemv_f32_avx2(out, w, x, n_out, n_in);
        return;
    }
#endif
    gemv_f32_scalar(out, w, x, n_out, n_in);
}

// ---------------------------------------------------------------------------
// RMSNorm: out = x / sqrt(mean(x^2) + eps) * w
// ---------------------------------------------------------------------------

void rms_norm_scalar(const float* x, float* out, const float* w,
                     int n, float eps) {
    // 8-accumulator mirror of the SIMD sum of squares.
    float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    int i = 0;
    for (; i + 8 <= n; i += 8)
        for (int j = 0; j < 8; ++j)
            acc[j] = std::fmaf(x[i + j], x[i + j], acc[j]);
    for (; i < n; ++i) acc[0] = std::fmaf(x[i], x[i], acc[0]);
    const float t0 = acc[0] + acc[1];
    const float t1 = acc[2] + acc[3];
    const float t2 = acc[4] + acc[5];
    const float t3 = acc[6] + acc[7];
    const float ss = (t0 + t1) + (t2 + t3);
    const float inv = 1.0f / std::sqrt(ss / static_cast<float>(n) + eps);
    for (int k = 0; k < n; ++k) out[k] = x[k] * inv * w[k];
}

#if HEPH_X86
__attribute__((target("avx2")))
static void rms_norm_avx2(const float* x, float* out, const float* w,
                          int n, float eps) {
    __m256 acc = _mm256_setzero_ps();
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        const __m256 xv = _mm256_loadu_ps(x + i);
        acc = _mm256_fmadd_ps(xv, xv, acc);
    }
    float lane[8];
    _mm256_storeu_ps(lane, acc);
    float ss = 0.f;
    const float t0 = lane[0] + lane[1];
    const float t1 = lane[2] + lane[3];
    const float t2 = lane[4] + lane[5];
    const float t3 = lane[6] + lane[7];
    ss = (t0 + t1) + (t2 + t3);
    for (; i < n; ++i) ss = std::fmaf(x[i], x[i], ss);
    const float inv = 1.0f / std::sqrt(ss / static_cast<float>(n) + eps);
    const __m256 invv = _mm256_set1_ps(inv);
    i = 0;
    for (; i + 8 <= n; i += 8) {
        const __m256 xv = _mm256_loadu_ps(x + i);
        const __m256 wv = _mm256_loadu_ps(w + i);
        _mm256_storeu_ps(out + i, _mm256_mul_ps(_mm256_mul_ps(xv, invv), wv));
    }
    for (; i < n; ++i) out[i] = x[i] * inv * w[i];
}
#endif

void rms_norm(const float* x, float* out, const float* w, int n, float eps) {
#if HEPH_X86
    if (kHasAvx2) {
        rms_norm_avx2(x, out, w, n, eps);
        return;
    }
#endif
    rms_norm_scalar(x, out, w, n, eps);
}

// ---------------------------------------------------------------------------
// SwiGLU: out = silu(g) * u, silu(z) = z * sigmoid(z)
// ---------------------------------------------------------------------------

// SwiGLU: out = silu(g) * u, silu(z) = z * sigmoid(z).
//
// Single scalar implementation on every ISA: the elementwise exp()
// dominates the cost, and one code path keeps sigmoid bit-identical
// between the reference and the engine (vector libm exp is not part of
// the base ISA and would break the parity contract).
void swiglu_scalar(const float* g, const float* u, float* out, int n) {
    for (int i = 0; i < n; ++i) {
        const float z = g[i];
        const float s = z / (1.0f + std::exp(-z));
        out[i] = s * u[i];
    }
}

void swiglu(const float* g, const float* u, float* out, int n) {
    swiglu_scalar(g, u, out, n);
}

// ---------------------------------------------------------------------------
// RoPE (half-split convention, in place on [seq, n_head, d_head])
// ---------------------------------------------------------------------------

RopeTables build_rope_tables(int max_seq, int d_head, float theta) {
    RopeTables t;
    t.max_seq = max_seq;
    t.half = d_head / 2;
    t.cos.resize(static_cast<size_t>(max_seq) * t.half);
    t.sin.resize(static_cast<size_t>(max_seq) * t.half);
    for (int pos = 0; pos < max_seq; ++pos) {
        for (int i = 0; i < t.half; ++i) {
            const float inv_freq =
                1.0f / std::pow(theta, (2.0f * i) / static_cast<float>(d_head));
            const float angle = static_cast<float>(pos) * inv_freq;
            t.cos[static_cast<size_t>(pos) * t.half + i] = std::cos(angle);
            t.sin[static_cast<size_t>(pos) * t.half + i] = std::sin(angle);
        }
    }
    return t;
}

// The rotation is elementwise per (position, head) pair: multiply/sub/add
// only, no FMA, so scalar and SIMD agree bit-for-bit by construction.
void apply_rope_scalar(float* x, const RopeTables& t, int seq, int pos_offset) {
    const int half = t.half;
    const int d_head = half * 2;
    const size_t row = static_cast<size_t>(half);
    for (int i = 0; i < seq; ++i) {
        const size_t p = static_cast<size_t>(pos_offset + i);
        float* xr = x + static_cast<size_t>(i) * d_head;
        for (int j = 0; j < half; ++j) {
            const float c = t.cos[p * row + j];
            const float s = t.sin[p * row + j];
            const float x1 = xr[j];
            const float x2 = xr[half + j];
            // half-split rotation: [x1*c - x2*s, x1*s + x2*c] — preserves
            // the pair norm exactly and matches the NumPy oracle
            xr[j] = x1 * c - x2 * s;
            xr[half + j] = x1 * s + x2 * c;
        }
    }
}

#if HEPH_X86
__attribute__((target("avx2")))
static void apply_rope_avx2(float* x, const RopeTables& t, int seq,
                            int pos_offset) {
    const int half = t.half;
    const int d_head = half * 2;
    const size_t row = static_cast<size_t>(half);
    for (int i = 0; i < seq; ++i) {
        const size_t p = static_cast<size_t>(pos_offset + i);
        float* xr = x + static_cast<size_t>(i) * d_head;
        int j = 0;
        for (; j + 8 <= half; j += 8) {
            const __m256 x1 = _mm256_loadu_ps(xr + j);
            const __m256 x2 = _mm256_loadu_ps(xr + half + j);
            const __m256 cv = _mm256_loadu_ps(&t.cos[p * row + j]);
            const __m256 sv = _mm256_loadu_ps(&t.sin[p * row + j]);
            const __m256 a = _mm256_mul_ps(x1, cv);
            const __m256 b = _mm256_mul_ps(x2, sv);
            const __m256 d = _mm256_mul_ps(x1, sv);
            const __m256 e = _mm256_mul_ps(x2, cv);
            _mm256_storeu_ps(xr + j, _mm256_sub_ps(a, b));
            _mm256_storeu_ps(xr + half + j, _mm256_add_ps(d, e));
        }
        for (; j < half; ++j) {
            const float c = t.cos[p * row + j];
            const float s = t.sin[p * row + j];
            const float x1 = xr[j];
            const float x2 = xr[half + j];
            xr[j] = x1 * c - x2 * s;
            xr[half + j] = x1 * s + x2 * c;
        }
    }
}
#endif

void apply_rope(float* x, const RopeTables& t, int seq, int pos_offset) {
#if HEPH_X86
    if (kHasAvx2) {
        apply_rope_avx2(x, t, seq, pos_offset);
        return;
    }
#endif
    apply_rope_scalar(x, t, seq, pos_offset);
}

// ---------------------------------------------------------------------------
// Softmax (numerically stable, in place)
// ---------------------------------------------------------------------------

void softmax_inplace(float* v, int n) {
    float mx = v[0];
    for (int i = 1; i < n; ++i) mx = std::max(mx, v[i]);
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        v[i] = std::exp(v[i] - mx);
        sum += v[i];
    }
    const float inv = 1.0f / sum;
    for (int i = 0; i < n; ++i) v[i] *= inv;
}

}  // namespace heph
