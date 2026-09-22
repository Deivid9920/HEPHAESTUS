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
