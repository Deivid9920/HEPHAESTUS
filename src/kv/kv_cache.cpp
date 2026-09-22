#include "kv/kv_cache.h"
#include <stdexcept>
namespace heph {
KvCacheFp32::KvCacheFp32(int, int, int, int) {
  throw std::runtime_error("heph: kv_cache not implemented (phase 3/5)");
}
void KvCacheFp32::append(int, int, const float*, const float*) {
  throw std::runtime_error("heph: kv_cache not implemented (phase 3/5)");
}
const float* KvCacheFp32::k_row(int, int, int) const {
  throw std::runtime_error("heph: kv_cache not implemented (phase 3/5)");
}
const float* KvCacheFp32::v_row(int, int, int) const {
  throw std::runtime_error("heph: kv_cache not implemented (phase 3/5)");
}
PagedInt8KvCache::PagedInt8KvCache(int, int, int, int) {
  throw std::runtime_error("heph: kv_cache not implemented (phase 5)");
}
void PagedInt8KvCache::append(int, int, const float*, const float*) {
  throw std::runtime_error("heph: kv_cache not implemented (phase 5)");
}
int PagedInt8KvCache::valid_len(int) const {
  throw std::runtime_error("heph: kv_cache not implemented (phase 5)");
}
int PagedInt8KvCache::n_pages(int) const {
  throw std::runtime_error("heph: kv_cache not implemented (phase 5)");
}
void PagedInt8KvCache::k_row(int, int, int, int, float*) const {
  throw std::runtime_error("heph: kv_cache not implemented (phase 5)");
}
void PagedInt8KvCache::v_row(int, int, int, int, float*) const {
  throw std::runtime_error("heph: kv_cache not implemented (phase 5)");
}
}  // namespace heph
