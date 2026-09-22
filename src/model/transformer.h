// The transformer: fp32 forward matching tests/golden/reference_model.py
// step for step (pre-norm blocks, half-split RoPE applied only to the
// incoming tokens, causal attention scaled 1/sqrt(d_head) over the
// per-layer cache, SwiGLU MLP, final norm, tied-embedding head), greedy
// generation and logits dumping for the golden tests.

#pragma once

#include <string>
#include <vector>

#include "kv/kv_cache.h"
#include "loader/loader.h"
#include "quant/quantize.h"

namespace heph {

enum class KvMode { Fp32, Int8Paged };

class Transformer {
  public:
    Transformer(const Manifest& manifest, const float* embedding,
                const std::vector<const float*>& layer_weights,
                const float* final_norm);

    // Logits of the LAST token of `tokens` placed at `pos_offset`,
    // updating the per-layer caches (past + current chunk).
    void logits_last(const std::vector<int>& tokens, int pos_offset);

    // Greedy continuation: exact argmax (lowest index on ties), no
    // special-token injection, stops before eos_id when not negative.
    std::vector<int> greedy_generate(const std::vector<int>& prompt,
                                     int n_new, int eos_id);

    const float* last_logits() const;
    int vocab() const;

    void reset_cache();
    void set_kv_mode(KvMode mode);

  private:
    struct Impl;
    Impl* impl_;
};

// Sampling helpers used by non-greedy generation (temperature, top-p).
int sample_token(const float* logits, int vocab, float temperature,
                 float top_p, unsigned* rng_state);

}  // namespace heph
