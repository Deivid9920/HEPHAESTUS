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
