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
