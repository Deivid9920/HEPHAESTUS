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
