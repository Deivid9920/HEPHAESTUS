// Progressive weight quantization: int8 / int4 groupwise / ternary.
//
// Numerical contracts (enforced by tests/cpp/test_kernels.cpp):
//   - int8: symmetric per output channel, scale = max_abs(row)/127,
//     q = round(w/scale) clamped to [-127, 127]. GEMV accumulates the
//     int8-as-float products in the SAME 8-lane pattern as
//     gemv_f32_scalar, so scalar and SIMD paths are bit-identical and
//     the quantized result tracks the dequantized fp32 dot within
//     quantization error.
//   - int4: groupwise along the input dim, scale = max_abs(group)/7
//     stored as fp16, code = round(w/scale)+8 in [0, 15], packed LOW
//     NIBBLE FIRST (weight 2i -> low nibble of byte i, weight 2i+1 ->
//     high nibble). The unpacking kernel mirrors exactly this order.
//   - ternary: codes in {0, 1, 2} meaning {0, +s, -s}, 2-bit codes
//     packed 4 per byte with the FIRST weight in the LOWEST bits; the
//     GEMV reads codes through a lookup table lut[code] = {0, +s, -s, 0}.
//     quantize_row_ternary() derives s = mean(|w|) of its row;
//     quantize_matrix() honors the spec's per-tensor scale = mean(|w|)
//     over the whole matrix and stores it on every row.
//
// Accumulation contract: no int8/int4/ternary kernel ever multiplies
// into a 16-bit accumulator; products are widened to 32-bit lanes
// (a d_model=384 row overflows 16 bits immediately) or accumulated in
// the fp32 lane pattern shared with the scalar reference.
//
// Quantized kernels keep the 256-bit lane layout on every ISA: the
// bit-exactness contract between the scalar mirror and the dispatched
// implementation must not depend on which ISA the host reports. The
// 512-bit path is reserved for the fp32 GEMV, which dominates runtime.

#include "quant/quantize.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>

#if defined(__x86_64__) || defined(__i386__)
#define HEPH_X86 1
#include <immintrin.h>
#else
#define HEPH_X86 0
#endif

namespace heph {

namespace {

inline int clamp_int(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

// IEEE 754 binary16 <-> float (round-to-nearest-even on the way down).
uint16_t float_to_half(float f) {
    uint32_t x;
    std::memcpy(&x, &f, sizeof(x));
    const uint32_t sign = (x >> 16) & 0x8000u;
    const int32_t exp = static_cast<int32_t>((x >> 23) & 0xFFu) - 127;
    const uint32_t mant = x & 0x007FFFFFu;
    if (((x >> 23) & 0xFFu) == 0xFFu)  // inf / nan
        return static_cast<uint16_t>(sign | 0x7C00u | (mant ? 0x200u : 0));
    if (exp > 15) return static_cast<uint16_t>(sign | 0x7C00u);  // overflow -> inf
    if (exp >= -14) {
        const uint32_t half =
            static_cast<uint32_t>(exp + 15) << 10 | (mant >> 13);
        // round to nearest even on the 13 dropped bits
        const uint32_t rest = mant & 0x1FFFu;
        uint32_t h = half + (rest > 0x1000u ? 1u : (rest == 0x1000u ? (half & 1u) : 0u));
        return static_cast<uint16_t>(sign | h);
    }
    // subnormal half
    const float scaled = std::ldexp(f, 24);
    const float rounded = std::nearbyint(scaled);
    return static_cast<uint16_t>(sign | static_cast<uint32_t>(static_cast<int>(rounded)));
}

float half_to_float(uint16_t h) {
    const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
    const uint32_t exp = (h >> 10) & 0x1Fu;
    const uint32_t mant = h & 0x3FFu;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            // subnormal: normalize
            int e = -14;
            uint32_t m = mant;
            while ((m & 0x400u) == 0) {
                m <<= 1;
                --e;
            }
            bits = sign | static_cast<uint32_t>(127 + e) << 23 |
                   (m & 0x3FFu) << 13;
        }
    } else if (exp == 31) {
        bits = sign | 0xFFu << 23 | mant << 13;
    } else {
        bits = sign | (exp - 15u + 127u) << 23 | mant << 13;
    }
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

// 8-lane combine shared by every quantized GEMV (the exact tail of the
// gemv_f32_scalar mirror).
inline float combine8(const float acc[8]) {
    const float t0 = acc[0] + acc[1];
    const float t1 = acc[2] + acc[3];
    const float t2 = acc[4] + acc[5];
    const float t3 = acc[6] + acc[7];
    return (t0 + t1) + (t2 + t3);
}

}  // namespace

QuantMode parse_quant_mode(const std::string& name) {
    if (name == "fp32") return QuantMode::Fp32;
    if (name == "int8") return QuantMode::Int8;
    if (name == "int4") return QuantMode::Int4;
    if (name == "ternary") return QuantMode::Ternary;
    throw std::invalid_argument("unknown quant mode: " + name);
}

const char* quant_mode_name(QuantMode mode) {
    switch (mode) {
        case QuantMode::Fp32: return "fp32";
        case QuantMode::Int8: return "int8";
        case QuantMode::Int4: return "int4";
        case QuantMode::Ternary: return "ternary";
    }
    return "fp32";
}

// ---------------------------------------------------------------------------
// int8
// ---------------------------------------------------------------------------

void quantize_rows_int8(const float* w, int n_out, int n_in, int8_t* q,
                        float* scales) {
    for (int r = 0; r < n_out; ++r) {
        const float* row = w + static_cast<size_t>(r) * n_in;
        float max_abs = 0.0f;
        for (int c = 0; c < n_in; ++c)
            max_abs = std::max(max_abs, std::fabs(row[c]));
        const float scale = max_abs > 0.0f ? max_abs / 127.0f : 1.0f;
        scales[r] = scale;
        int8_t* qr = q + static_cast<size_t>(r) * n_in;
        for (int c = 0; c < n_in; ++c)
            qr[c] = static_cast<int8_t>(
                clamp_int(static_cast<int>(std::lrint(row[c] / scale)), -127, 127));
    }
}

void gemv_int8_scalar(float* out, const int8_t* q, const float* scales,
                      const float* x, int n_out, int n_in) {
    // 8-accumulator mirror of the dispatched lane layout (see file
    // header): the int8 weight is widened to fp32 and every product is
    // fused with fmaf exactly like the SIMD path.
    for (int r = 0; r < n_out; ++r) {
        const int8_t* row = q + static_cast<size_t>(r) * n_in;
        float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        int c = 0;
        for (; c + 8 <= n_in; c += 8) {
            for (int j = 0; j < 8; ++j)
                acc[j] = std::fmaf(static_cast<float>(row[c + j]), x[c + j],
                                   acc[j]);
        }
        for (; c < n_in; ++c)
            acc[0] = std::fmaf(static_cast<float>(row[c]), x[c], acc[0]);
        out[r] = combine8(acc) * scales[r];
    }
}

#if HEPH_X86
__attribute__((target("avx2")))
static void gemv_int8_avx2(float* out, const int8_t* q, const float* scales,
                           const float* x, int n_out, int n_in) {
    for (int r = 0; r < n_out; ++r) {
        const int8_t* row = q + static_cast<size_t>(r) * n_in;
        __m256 acc = _mm256_setzero_ps();
        int c = 0;
        for (; c + 8 <= n_in; c += 8) {
            const __m128i i8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(row + c));
            const __m256 wv = _mm256_cvtepi32_ps(
                _mm256_cvtepi8_epi32(i8));
            const __m256 xv = _mm256_loadu_ps(x + c);
            acc = _mm256_fmadd_ps(wv, xv, acc);
        }
        alignas(32) float lane[8];
        _mm256_storeu_ps(lane, acc);
        // scalar tail folds sequentially into lane 0, mirroring the
        // scalar reference's tail loop exactly
        for (; c < n_in; ++c)
            lane[0] = std::fmaf(static_cast<float>(row[c]), x[c], lane[0]);
        out[r] = combine8(lane) * scales[r];
    }
}
#endif

void gemv_int8(float* out, const int8_t* q, const float* scales,
               const float* x, int n_out, int n_in) {
#if HEPH_X86
    gemv_int8_avx2(out, q, scales, x, n_out, n_in);
#else
    gemv_int8_scalar(out, q, scales, x, n_out, n_in);
#endif
}

// ---------------------------------------------------------------------------
// int4 groupwise
// ---------------------------------------------------------------------------

void quantize_rows_int4(const float* w, int n_out, int n_in, int group,
                        uint8_t* packed, uint16_t* scales_fp16) {
    if (group <= 0) throw std::invalid_argument("int4 group must be positive");
    const int n_groups = (n_in + group - 1) / group;
    const int row_bytes = (n_in + 1) / 2;
    for (int r = 0; r < n_out; ++r) {
        const float* row = w + static_cast<size_t>(r) * n_in;
        uint8_t* prow = packed + static_cast<size_t>(r) * row_bytes;
        std::memset(prow, 0, static_cast<size_t>(row_bytes));
        uint16_t* pscale = scales_fp16 + static_cast<size_t>(r) * n_groups;
        for (int g = 0; g < n_groups; ++g) {
            const int lo = g * group;
            const int hi = std::min(lo + group, n_in);
            float max_abs = 0.0f;
            for (int c = lo; c < hi; ++c)
                max_abs = std::max(max_abs, std::fabs(row[c]));
            const float scale = max_abs > 0.0f ? max_abs / 7.0f : 1.0f;
            pscale[g] = float_to_half(scale);
            const float inv = 1.0f / half_to_float(pscale[g]);
            for (int c = lo; c < hi; ++c) {
                const int code =
                    clamp_int(static_cast<int>(std::lrint(row[c] * inv)) + 8, 0, 15);
                uint8_t* byte = &prow[c / 2];
                if ((c & 1) == 0)
                    *byte = static_cast<uint8_t>((*byte & 0xF0u) | code);
                else
                    *byte = static_cast<uint8_t>((*byte & 0x0Fu) | (code << 4));
            }
        }
    }
}

void unpack_row_int4(const uint8_t* packed, const uint16_t* scales_fp16,
                     int n_in, int group, float* out) {
    const int n_groups = (n_in + group - 1) / group;
    std::vector<float> gscale(static_cast<size_t>(n_groups));
    for (int g = 0; g < n_groups; ++g)
        gscale[static_cast<size_t>(g)] = half_to_float(scales_fp16[g]);
    for (int c = 0; c < n_in; ++c) {
        const uint8_t byte = packed[c / 2];
        const int code = (c & 1) == 0 ? (byte & 0x0Fu) : (byte >> 4);
        out[c] = static_cast<float>(code - 8) *
                 gscale[static_cast<size_t>(c / group)];
    }
}

void gemv_int4_scalar(float* out, const uint8_t* packed,
                      const uint16_t* scales_fp16, const float* x, int n_out,
                      int n_in, int group) {
    const int row_bytes = (n_in + 1) / 2;
    const int n_groups = (n_in + group - 1) / group;
    for (int r = 0; r < n_out; ++r) {
        const uint8_t* row = packed + static_cast<size_t>(r) * row_bytes;
        // group scales for this row, decoded once
        std::vector<float> gscale(static_cast<size_t>(n_groups));
        for (int g = 0; g < n_groups; ++g)
            gscale[static_cast<size_t>(g)] =
                half_to_float(scales_fp16[static_cast<size_t>(r) * n_groups + g]);
        float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        int c = 0;
        for (; c + 8 <= n_in; c += 8) {
            for (int j = 0; j < 8; ++j) {
                const uint8_t byte = row[(c + j) / 2];
                const int code = ((c + j) & 1) == 0 ? (byte & 0x0Fu) : (byte >> 4);
                const float v = static_cast<float>(code - 8) *
                                gscale[static_cast<size_t>((c + j) / group)];
                acc[j] = std::fmaf(v, x[c + j], acc[j]);
            }
        }
        for (; c < n_in; ++c) {
            const uint8_t byte = row[c / 2];
            const int code = (c & 1) == 0 ? (byte & 0x0Fu) : (byte >> 4);
            const float v = static_cast<float>(code - 8) *
                            gscale[static_cast<size_t>(c / group)];
            acc[0] = std::fmaf(v, x[c], acc[0]);
        }
        out[r] = combine8(acc);
    }
}

void gemv_int4(float* out, const uint8_t* packed, const uint16_t* scales_fp16,
               const float* x, int n_out, int n_in, int group) {
    // Nibble extraction dominates and is inherently scalar; one code
    // path keeps the bit-exact contract (see file header).
    gemv_int4_scalar(out, packed, scales_fp16, x, n_out, n_in, group);
}

// ---------------------------------------------------------------------------
// ternary 1.58-bit
// ---------------------------------------------------------------------------

namespace {
inline void pack_ternary_row_with_scale(const float* w, int n, uint8_t* packed,
                                        float scale) {
    std::memset(packed, 0, static_cast<size_t>(n + 3) / 4);
    for (int c = 0; c < n; ++c) {
        const int t = clamp_int(static_cast<int>(std::lrint(w[c] / scale)), -1, 1);
        const int code = t > 0 ? 1 : (t < 0 ? 2 : 0);
        // first weight in the LOWEST 2 bits of each byte
        packed[c / 4] |= static_cast<uint8_t>(code << (2 * (c % 4)));
    }
}
}  // namespace

void quantize_row_ternary(const float* w, int n, uint8_t* packed,
                          float* scale) {
    double sum_abs = 0.0;
    for (int c = 0; c < n; ++c) sum_abs += std::fabs(static_cast<double>(w[c]));
    const float mean_abs =
        sum_abs > 0.0 ? static_cast<float>(sum_abs / n) : 1.0f;
    *scale = mean_abs;
    pack_ternary_row_with_scale(w, n, packed, mean_abs);
}

void ternary_dequant_row(const uint8_t* packed, float scale, float* out,
                         int n) {
    static const float lut[4] = {0.0f, 1.0f, -1.0f, 0.0f};
    for (int c = 0; c < n; ++c) {
        const int code = (packed[c / 4] >> (2 * (c % 4))) & 0x3;
        out[c] = lut[code] * scale;
    }
}

void gemv_ternary_row(float* out, const uint8_t* packed, float scale,
                      const float* x, int n_in) {
    static const float lut[4] = {0.0f, 1.0f, -1.0f, 0.0f};
    float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    int c = 0;
    for (; c + 8 <= n_in; c += 8) {
        for (int j = 0; j < 8; ++j) {
            const int code = (packed[(c + j) / 4] >> (2 * ((c + j) % 4))) & 0x3;
            acc[j] = std::fmaf(lut[code] * scale, x[c + j], acc[j]);
        }
    }
    for (; c < n_in; ++c) {
        const int code = (packed[c / 4] >> (2 * (c % 4))) & 0x3;
        acc[0] = std::fmaf(lut[code] * scale, x[c], acc[0]);
    }
    out[0] = combine8(acc);
}

// ---------------------------------------------------------------------------
// QuantizedMat
// ---------------------------------------------------------------------------

QuantizedMat quantize_matrix(const float* w, int n_out, int n_in,
                             QuantMode mode, int group) {
    QuantizedMat qm;
    qm.mode = mode;
    qm.n_out = n_out;
    qm.n_in = n_in;
    qm.group = group;
    const size_t elems = static_cast<size_t>(n_out) * n_in;
    if (mode == QuantMode::Int8) {
        qm.packed.resize(elems);
        qm.scales.resize(n_out);
        quantize_rows_int8(w, n_out, n_in,
                           reinterpret_cast<int8_t*>(qm.packed.data()),
                           qm.scales.data());
    } else if (mode == QuantMode::Int4) {
        const int n_groups = (n_in + group - 1) / group;
        qm.packed.resize(static_cast<size_t>(n_out) * ((n_in + 1) / 2));
        qm.scales_fp16.resize(static_cast<size_t>(n_out) * n_groups);
        quantize_rows_int4(w, n_out, n_in, group, qm.packed.data(),
                           qm.scales_fp16.data());
    } else if (mode == QuantMode::Ternary) {
        // per-tensor scale = mean(|w|) over the WHOLE matrix (spec); the
        // same value is stored on every row so the GEMV loop stays uniform.
        double sum_abs = 0.0;
        for (size_t i = 0; i < elems; ++i)
            sum_abs += std::fabs(static_cast<double>(w[i]));
        const float tensor_scale =
            sum_abs > 0.0 ? static_cast<float>(sum_abs / elems) : 1.0f;
        const int row_bytes = (n_in + 3) / 4;
        qm.packed.resize(static_cast<size_t>(n_out) * row_bytes);
        qm.scales.assign(n_out, tensor_scale);
        for (int r = 0; r < n_out; ++r)
            pack_ternary_row_with_scale(w + static_cast<size_t>(r) * n_in, n_in,
                                        qm.packed.data() +
                                            static_cast<size_t>(r) * row_bytes,
                                        tensor_scale);
    } else {
        throw std::invalid_argument("quantize_matrix: fp32 is not packed");
    }
    return qm;
}

}  // namespace heph
