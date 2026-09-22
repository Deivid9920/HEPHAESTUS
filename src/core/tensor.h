// Minimal tensor type: row-major fp32 with 64-byte aligned storage, so
// SIMD loads never cross allocation boundaries and aligned_hint stays
// true. The engine never owns more than a handful of these.

#pragma once

#include <cstdint>
#include <vector>

namespace heph {

struct Tensor {
    int rows = 0;
    int cols = 0;
    std::vector<float> data;  // size rows*cols, row-major

    float& at(int r, int c) { return data[static_cast<size_t>(r) * cols + c]; }
    const float& at(int r, int c) const {
        return data[static_cast<size_t>(r) * cols + c];
    }
};

// 64-byte aligned buffer of n floats (posix_memalign semantics via
// operator new alignment); used by kernels that want alignment safety.
float* aligned_floats(size_t n);
void aligned_free(float* p);

}  // namespace heph
