// Aligned buffer helpers: the kernels only rely on natural vector
// alignment, but 64-byte storage keeps every aligned load inside one
// cache line and removes the misalignment class of SIGSEGV entirely.
// (Mandate: prefer _mm256_loadu_ps unless alignment is guaranteed; with
// this allocator both forms are safe.)

#include "core/tensor.h"

#include <cstdlib>

namespace heph {

float* aligned_floats(size_t n) {
    void* ptr = nullptr;
    const size_t bytes = ((n * sizeof(float) + 63) / 64) * 64;
    if (posix_memalign(&ptr, 64, bytes) != 0) return nullptr;
    return static_cast<float*>(ptr);
}

void aligned_free(float* p) { std::free(p); }

}  // namespace heph
