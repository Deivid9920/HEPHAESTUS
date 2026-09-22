// KV caches: fp32 reference cache and the quantized paged cache.
//
// PagedInt8KvCache implements the phase-5 contract: pages of
// kv.page_tokens (16) tokens, a block table mapping logical page index
// to a physical slot, no external fragmentation, and int8 quantization
// per (layer, head, token) with scale = max_abs/127. Attention over a
// partially filled last page MUST ignore the uninitialized slots: the
// cache exposes valid_len() per page and the model masks those columns
// before the softmax (garbage entering the softmax would corrupt the
// distribution). Only the incoming token's k/v are rotated before
// append; cached rows keep their original rotation.

#pragma once

#include <cstdint>
#include <vector>

namespace heph {

class KvCacheFp32 {
  public:
    KvCacheFp32(int n_layers, int n_kv_head, int d_head, int max_seq);

    // Appends one rotated (k, v) row pair for a layer at position pos.
    void append(int layer, int pos, const float* k_row, const float* v_row);
    // Pointers to the cached row of (layer, head) at position pos.
    const float* k_row(int layer, int head, int pos) const;
    const float* v_row(int layer, int head, int pos) const;

  private:
    int n_layers_;
    int n_kv_head_;
    int d_head_;
    int max_seq_;
    // [layer][head][pos][d]
    std::vector<float> k_;
    std::vector<float> v_;
};

class PagedInt8KvCache {
  public:
    static constexpr int kPageTokens = 16;

    PagedInt8KvCache(int n_layers, int n_kv_head, int d_head, int max_pages);

    // Quantizes and appends one row pair; allocates a new page from the
    // pool when pos crosses a page boundary (block table append).
    void append(int layer, int pos, const float* k_row, const float* v_row);

    int valid_len(int page) const;  // tokens filled in a logical page
    int n_pages(int layer) const;

    // Dequantized view of one (layer, head, page) token row into out.
    void k_row(int layer, int head, int page, int slot, float* out) const;
    void v_row(int layer, int head, int page, int slot, float* out) const;

  private:
    int n_layers_;
    int n_kv_head_;
    int d_head_;
    // physical pool: [layer][head][page][slot][d], int8
    std::vector<int8_t> k_pool_;
    std::vector<int8_t> v_pool_;
    // per (layer, head, page, slot) scale
    std::vector<float> k_scale_;
    std::vector<float> v_scale_;
    std::vector<int> valid_;          // tokens filled per (layer, page)
    std::vector<int> block_table_;    // logical page -> physical page
};

}  // namespace heph
