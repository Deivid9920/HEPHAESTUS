// fp32 KV cache and the quantized paged cache (phase-3/5 contracts).

#include "kv/kv_cache.h"

#include <cmath>
#include <cstring>
#include <stdexcept>

namespace heph {

// ---------------------------------------------------------------------------
// KvCacheFp32: [layer][head][pos][d]
// ---------------------------------------------------------------------------

KvCacheFp32::KvCacheFp32(int n_layers, int n_kv_head, int d_head, int max_seq)
    : n_layers_(n_layers),
      n_kv_head_(n_kv_head),
      d_head_(d_head),
      max_seq_(max_seq),
      k_(static_cast<size_t>(n_layers) * n_kv_head * max_seq * d_head, 0.0f),
      v_(static_cast<size_t>(n_layers) * n_kv_head * max_seq * d_head, 0.0f) {}

void KvCacheFp32::append(int layer, int pos, const float* k_row,
                         const float* v_row) {
    // k_row / v_row are [n_kv_head][d_head] contiguous for one token.
    for (int h = 0; h < n_kv_head_; ++h) {
        const size_t dst =
            ((static_cast<size_t>(layer) * n_kv_head_ + h) * max_seq_ + pos) *
            d_head_;
        std::memcpy(&k_[dst], k_row + static_cast<size_t>(h) * d_head_,
                    sizeof(float) * d_head_);
        std::memcpy(&v_[dst], v_row + static_cast<size_t>(h) * d_head_,
                    sizeof(float) * d_head_);
    }
}

const float* KvCacheFp32::k_row(int layer, int head, int pos) const {
    return &k_[((static_cast<size_t>(layer) * n_kv_head_ + head) * max_seq_ +
                pos) * d_head_];
}

const float* KvCacheFp32::v_row(int layer, int head, int pos) const {
    return &v_[((static_cast<size_t>(layer) * n_kv_head_ + head) * max_seq_ +
                pos) * d_head_];
}

// ---------------------------------------------------------------------------
// PagedInt8KvCache: 16-token pages, block table, per (layer, head, token)
// symmetric int8 quantization with scale = max_abs/127.
//
// Attention must only READ slots below valid_len(layer, page): the
// partially filled last page keeps its uninitialized slots out of the
// softmax because the caller's loop bound is pos_offset + p + 1, which
// never exceeds the number of appended tokens.
// ---------------------------------------------------------------------------

PagedInt8KvCache::PagedInt8KvCache(int n_layers, int n_kv_head, int d_head,
                                   int max_pages)
    : n_layers_(n_layers),
      n_kv_head_(n_kv_head),
      d_head_(d_head),
      k_pool_(static_cast<size_t>(n_layers) * n_kv_head * max_pages *
              kPageTokens * d_head),
      v_pool_(static_cast<size_t>(n_layers) * n_kv_head * max_pages *
              kPageTokens * d_head),
      k_scale_(static_cast<size_t>(n_layers) * n_kv_head * max_pages *
               kPageTokens, 0.0f),
      v_scale_(static_cast<size_t>(n_layers) * n_kv_head * max_pages *
               kPageTokens, 0.0f),
      valid_(static_cast<size_t>(n_layers) * 64, 0),  // per (layer, page)
      max_pages_(max_pages) {
    block_table_.reserve(max_pages);
}

void PagedInt8KvCache::append(int layer, int pos, const float* k_row,
                              const float* v_row) {
    const int page = pos / kPageTokens;
    const int slot = pos % kPageTokens;
    if (page >= max_pages_)
        throw std::runtime_error("kv cache: page capacity exceeded");
    // block table grows in allocation order; sequential appends never
    // fragment (freed pages are not reused within one generation).
    while (static_cast<int>(block_table_.size()) <= page)
        block_table_.push_back(static_cast<int>(block_table_.size()));
    const int phys = block_table_[page];
    for (int h = 0; h < n_kv_head_; ++h) {
        const float* ksrc = k_row + static_cast<size_t>(h) * d_head_;
        const float* vsrc = v_row + static_cast<size_t>(h) * d_head_;
        const size_t base =
            ((static_cast<size_t>(layer) * n_kv_head_ + h) * max_pages_ +
             phys) * kPageTokens * d_head_ +
            static_cast<size_t>(slot) * d_head_;
        float kmax = 0.0f, vmax = 0.0f;
        for (int d = 0; d < d_head_; ++d) {
            kmax = std::max(kmax, std::fabs(ksrc[d]));
            vmax = std::max(vmax, std::fabs(vsrc[d]));
        }
        const float ks = kmax / 127.0f;
        const float vs = vmax / 127.0f;
        k_scale_[(static_cast<size_t>(layer) * n_kv_head_ + h) * max_pages_ *
                     kPageTokens +
                 phys * kPageTokens + slot] = ks;
        v_scale_[(static_cast<size_t>(layer) * n_kv_head_ + h) * max_pages_ *
                     kPageTokens +
                 phys * kPageTokens + slot] = vs;
        for (int d = 0; d < d_head_; ++d) {
            k_pool_[base + d] = static_cast<int8_t>(
                std::lround(ksrc[d] / (ks > 0.0f ? ks : 1.0f)));
            v_pool_[base + d] = static_cast<int8_t>(
                std::lround(vsrc[d] / (vs > 0.0f ? vs : 1.0f)));
        }
    }
    const size_t vidx = static_cast<size_t>(layer) * 64 + page;
    valid_[vidx] = std::max(valid_[vidx], slot + 1);
}

int PagedInt8KvCache::valid_len(int layer, int page) const {
    return valid_[static_cast<size_t>(layer) * 64 + page];
}

int PagedInt8KvCache::n_pages(int layer) const {
    int n = 0;
    for (int p = 0; p < max_pages_; ++p)
        if (valid_len(layer, p) > 0) n = p + 1;
    return n;
}

void PagedInt8KvCache::k_row(int layer, int head, int page, int slot,
                             float* out) const {
    const int phys = block_table_[page];
    const size_t base =
        ((static_cast<size_t>(layer) * n_kv_head_ + head) * max_pages_ + phys) *
            kPageTokens * d_head_ +
        static_cast<size_t>(slot) * d_head_;
    const float s =
        k_scale_[(static_cast<size_t>(layer) * n_kv_head_ + head) * max_pages_ *
                     kPageTokens +
                 phys * kPageTokens + slot];
    for (int d = 0; d < d_head_; ++d)
        out[d] = static_cast<float>(k_pool_[base + d]) * s;
}

void PagedInt8KvCache::v_row(int layer, int head, int page, int slot,
                             float* out) const {
    const int phys = block_table_[page];
    const size_t base =
        ((static_cast<size_t>(layer) * n_kv_head_ + head) * max_pages_ + phys) *
            kPageTokens * d_head_ +
        static_cast<size_t>(slot) * d_head_;
    const float s =
        v_scale_[(static_cast<size_t>(layer) * n_kv_head_ + head) * max_pages_ *
                     kPageTokens +
                 phys * kPageTokens + slot];
    for (int d = 0; d < d_head_; ++d)
        out[d] = static_cast<float>(v_pool_[base + d]) * s;
}

}  // namespace heph
