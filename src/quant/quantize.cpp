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
