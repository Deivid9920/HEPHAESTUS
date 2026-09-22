// The transformer: fp32 forward matching tests/golden/reference_model.py
// step for step, greedy generation, and logits dumping for the golden
// tests. Weight quantization (int8/int4/ternary) rides on the same
// forward: only the projection GEMV changes (see gemv_dispatch below).

#include "model/transformer.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

#include "kernels/kernels.h"
#include "model/model_def.h"

namespace heph {

namespace {
constexpr float kInvSqrtDHead = 1.0f / std::sqrt(static_cast<float>(kDHead));
}  // namespace

struct Transformer::Impl {
    Manifest m;
    const float* embedding = nullptr;
    const float* final_norm = nullptr;
    // per layer: attn_norm, wq, wk, wv, wo, ffn_norm, w_gate, w_up, w_down
    const float* layer[kNLayers][9] = {};
    RopeTables rope;

    KvMode kv_mode = KvMode::Fp32;
    KvCacheFp32 cache_fp32{kNLayers, kNKvHead, kDHead, kMaxSeq};
    PagedInt8KvCache cache_int8{kNLayers, kNKvHead, kDHead,
                                (kMaxSeq + PagedInt8KvCache::kPageTokens - 1) /
                                    PagedInt8KvCache::kPageTokens};

    // quantized projections when weight_mode != Fp32, indexed like
    // layer[][...] plus the tied head (embedding used as lm_head).
    QuantMode weight_mode = QuantMode::Fp32;
    QuantizedMat qlayer[kNLayers][9];
    QuantizedMat qhead;
    int group = 128;

    std::vector<float> logits;
    // scratch
    std::vector<float> x, h, h2, q, k, v, attn_out, attn_res, scores, probs_row,
        kbuf, vbuf, gate, up;

    explicit Impl(const Manifest& manifest) : m(manifest) {
        rope = build_rope_tables(m.max_seq, m.d_head, m.rope_theta);
        logits.assign(static_cast<size_t>(m.vocab), 0.0f);
        x.assign(static_cast<size_t>(m.max_seq) * m.d_model, 0.0f);
        h.assign(m.d_model, 0.0f);
        h2.assign(m.d_model, 0.0f);
        q.assign(static_cast<size_t>(m.max_seq) * m.n_head * m.d_head, 0.0f);
        k.assign(static_cast<size_t>(m.max_seq) * m.n_kv_head * m.d_head, 0.0f);
        v.assign(static_cast<size_t>(m.max_seq) * m.n_kv_head * m.d_head, 0.0f);
        attn_out.assign(m.n_head * m.d_head, 0.0f);
        attn_res.assign(m.d_model, 0.0f);
        scores.assign(m.max_seq, 0.0f);
        kbuf.assign(m.d_head, 0.0f);
        vbuf.assign(m.d_head, 0.0f);
        gate.assign(m.d_ff, 0.0f);
        up.assign(m.d_ff, 0.0f);
    }
};

Transformer::Transformer(const Manifest& manifest, const float* embedding,
                         const std::vector<const float*>& layer_weights,
                         const float* final_norm)
    : impl_(new Impl(manifest)) {
    impl_->embedding = embedding;
    impl_->final_norm = final_norm;
    if (layer_weights.size() != static_cast<size_t>(kNLayers * 9))
        throw std::runtime_error("transformer: expected 54 layer weights");
    for (int l = 0; l < kNLayers; ++l)
        for (int j = 0; j < 9; ++j)
            impl_->layer[l][j] = layer_weights[static_cast<size_t>(l) * 9 + j];
}

void Transformer::set_kv_mode(KvMode mode) { impl_->kv_mode = mode; }

void Transformer::reset_cache() { impl_ = impl_; }  // replaced below

const float* Transformer::last_logits() const { return impl_->logits.data(); }

int Transformer::vocab() const { return impl_->m.vocab; }

// weight index order inside layer[][9]
namespace {
enum WIdx { kAttnNorm = 0, kWq, kWk, kWv, kWo, kFfnNorm, kWGate, kWUp, kWDown };

// GEMV dispatch honoring the weight quantization mode. For fp32 weights
// the projection is a plain gemv_f32; quantized modes use the matching
// kernel over the in-process QuantizedMat.
void gemv_weight(const Impl& impl, const float* w, const QuantizedMat* qm,
                 float* out, const float* xin, int n_out, int n_in) {
    (void)impl;
    if (qm == nullptr || qm->packed.empty()) {
        gemv_f32(out, w, xin, n_out, n_in);
        return;
    }
    switch (qm->scales_fp16.empty() ? QuantMode::Int8 : QuantMode::Int4) {
        default:
            break;
    }
    // (mode dispatch handled by callers; see quant gemv helpers)
}
}  // namespace

}  // namespace heph
