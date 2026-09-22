#include "core/sha256.h"
#include <stdexcept>
namespace heph {
std::string sha256_hex(const uint8_t*, size_t) {
  throw std::runtime_error("heph: core/sha256 not implemented (phase 0 stub)");
}
}  // namespace heph
