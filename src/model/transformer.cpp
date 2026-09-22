// The transformer: fp32 forward matching tests/golden/reference_model.py
// step for step, greedy generation, and logits dumping for the golden
// tests. Weight quantization (int8/int4/ternary) rides on the same
// forward: only the projection GEMV changes (see gemv_weight below).
//
// Numerical contracts:
//   - RoPE (half-split) is applied ONLY to the incoming tokens at their
//     absolute positions; cached K/V rows keep the rotation they were
//     stored with (never re-rotate the accumulated buffer).
//   - Attention reads exactly positions [0, pos_offset + i] for query i:
//     the paged int8 cache's partially filled last page is never touched
//     beyond its valid slots, so uninitialized poison cannot reach the
//     softmax.
//   - GQA repeat_kv: n_head == n_kv_head for the nano contract, so each
//     query head owns its kv head. If a future profile sets
//     n_kv_head < n_head, the expansion below AND the NumPy oracle
//     (reference_model.py) must be switched at the same time.

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
constexpr int kRep = kNHead / kNKvHead;  // GQA expansion (1 on nano)
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
    int cache_len = 0;

    // quantized projections when weight_mode != Fp32, indexed like
    // layer[][...] plus the tied head (embedding used as lm_head).
    // Norm vectors stay fp32: they are not GEMV weights.
    QuantMode weight_mode = QuantMode::Fp32;
    QuantizedMat qlayer[kNLayers][9];
    QuantizedMat qhead;
    int group = 128;
    bool quantized = false;

    std::vector<float> logits;
    // scratch
    std::vector<float> x, h, h2, q, k, v, attn_out, attn_res, scores,
        gate, up;

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

void Transformer::set_weight_mode(QuantMode mode, int group) {
    impl_->weight_mode = mode;
    impl_->group = group;
    impl_->quantized = mode != QuantMode::Fp32;
    if (!impl_->quantized) return;
    for (int l = 0; l < kNLayers; ++l) {
        impl_->qlayer[l][1] = quantize_matrix(impl_->layer[l][1], kDModel,
                                              kDModel, mode, group);  // wq
        impl_->qlayer[l][2] = quantize_matrix(impl_->layer[l][2], kDModel,
                                              kDModel, mode, group);  // wk
        impl_->qlayer[l][3] = quantize_matrix(impl_->layer[l][3], kDModel,
                                              kDModel, mode, group);  // wv
        impl_->qlayer[l][4] = quantize_matrix(impl_->layer[l][4], kDModel,
                                              kDModel, mode, group);  // wo
        impl_->qlayer[l][6] = quantize_matrix(impl_->layer[l][6], kDFf,
                                              kDModel, mode, group);  // w_gate
        impl_->qlayer[l][7] = quantize_matrix(impl_->layer[l][7], kDFf,
                                              kDModel, mode, group);  // w_up
        impl_->qlayer[l][8] = quantize_matrix(impl_->layer[l][8], kDModel,
                                              kDFf, mode, group);     // w_down
    }
    impl_->qhead = quantize_matrix(impl_->embedding, kVocab, kDModel, mode,
                                   group);
}

void Transformer::reset_cache() {
    impl_->cache_fp32 = KvCacheFp32{kNLayers, kNKvHead, kDHead, kMaxSeq};
    impl_->cache_int8 =
        PagedInt8KvCache{kNLayers, kNKvHead, kDHead,
                         (kMaxSeq + PagedInt8KvCache::kPageTokens - 1) /
                             PagedInt8KvCache::kPageTokens};
    impl_->cache_len = 0;
}

const float* Transformer::last_logits() const { return impl_->logits.data(); }

int Transformer::vocab() const { return impl_->m.vocab; }

// weight index order inside layer[][9]
namespace {
enum WIdx { kAttnNorm = 0, kWq, kWk, kWv, kWo, kFfnNorm, kWGate, kWUp, kWDown };

// Projection GEMV honoring the quantized-weight mode. For fp32 (or when
// a projection has no packed form) it is a plain gemv_f32.
void gemv_weight(const Transformer::Impl& impl, int layer, int widx,
                 float* out, const float* xin, int n_out, int n_in) {
    if (!impl.quantized || widx == kAttnNorm || widx == kFfnNorm) {
        gemv_f32(out, impl.layer[layer][widx], xin, n_out, n_in);
        return;
    }
    const QuantizedMat& qm = impl.qlayer[layer][widx];
    switch (qm.mode) {
        case QuantMode::Int8:
            gemv_int8(out, reinterpret_cast<const int8_t*>(qm.packed.data()),
                      qm.scales.data(), xin, n_out, n_in);
            return;
        case QuantMode::Int4:
            gemv_int4(out, qm.packed.data(), qm.scales_fp16.data(), xin,
                      n_out, n_in, qm.group);
            return;
        case QuantMode::Ternary: {
            const int row_bytes = (n_in + 3) / 4;
            for (int r = 0; r < n_out; ++r)
                gemv_ternary_row(out + r,
                                 qm.packed.data() +
                                     static_cast<size_t>(r) * row_bytes,
                                 qm.scales[r], xin, n_in);
            return;
        }
        default:
            gemv_f32(out, impl.layer[layer][widx], xin, n_out, n_in);
            return;
    }
}

void gemv_head(const Transformer::Impl& impl, float* out, const float* xin) {
    const int n_out = impl.m.vocab;
    const int n_in = impl.m.d_model;
    if (!impl.quantized) {
        gemv_f32(out, impl.embedding, xin, n_out, n_in);
        return;
    }
    const QuantizedMat& qm = impl.qhead;
    switch (qm.mode) {
        case QuantMode::Int8:
            gemv_int8(out, reinterpret_cast<const int8_t*>(qm.packed.data()),
                      qm.scales.data(), xin, n_out, n_in);
            return;
        case QuantMode::Int4:
            gemv_int4(out, qm.packed.data(), qm.scales_fp16.data(), xin,
                      n_out, n_in, qm.group);
            return;
        case QuantMode::Ternary: {
            const int row_bytes = (n_in + 3) / 4;
            for (int r = 0; r < n_out; ++r)
                gemv_ternary_row(out + r,
                                 qm.packed.data() +
                                     static_cast<size_t>(r) * row_bytes,
                                 qm.scales[r], xin, n_in);
            return;
        }
        default:
            gemv_f32(out, impl.embedding, xin, n_out, n_in);
            return;
    }
}
}  // namespace

void Transformer::logits_last(const std::vector<int>& tokens, int pos_offset) {
    Impl& impl = *impl_;
    const int seq = static_cast<int>(tokens.size());
    if (seq <= 0) throw std::runtime_error("logits_last: empty token chunk");
    if (pos_offset < 0 || pos_offset + seq > impl.m.max_seq)
        throw std::runtime_error("logits_last: context overflow");
    if (impl.cache_len != pos_offset)
        throw std::runtime_error(
            "logits_last: pos_offset does not match the cache length");

    const int dm = impl.m.d_model;
    const int nh = impl.m.n_head;
    const int nkv = impl.m.n_kv_head;
    const int dh = impl.m.d_head;
    const int dff = impl.m.d_ff;

    // embeddings
    for (int i = 0; i < seq; ++i) {
        if (tokens[i] < 0 || tokens[i] >= impl.m.vocab)
            throw std::runtime_error("logits_last: token id out of range");
        std::memcpy(impl.x.data() + static_cast<size_t>(i) * dm,
                    impl.embedding + static_cast<size_t>(tokens[i]) * dm,
                    sizeof(float) * dm);
    }

    for (int l = 0; l < kNLayers; ++l) {
        // q, k, v for the chunk (one row per token per projection)
        for (int i = 0; i < seq; ++i) {
            const float* xi = impl.x.data() + static_cast<size_t>(i) * dm;
            rms_norm(xi, impl.h.data(), impl.layer[l][kAttnNorm], dm,
                     impl.m.rms_norm_eps);
            gemv_weight(impl, l, kWq, impl.q.data() + static_cast<size_t>(i) * nh * dh,
                        impl.h.data(), nh * dh, dm);
            gemv_weight(impl, l, kWk, impl.k.data() + static_cast<size_t>(i) * nkv * dh,
                        impl.h.data(), nkv * dh, dm);
            gemv_weight(impl, l, kWv, impl.v.data() + static_cast<size_t>(i) * nkv * dh,
                        impl.h.data(), nkv * dh, dm);
        }
        // RoPE: rotate only the incoming rows, at their absolute
        // positions (apply_rope works on one [pos][d_head] row per call
        // per head — the tables are indexed by the token's position).
        for (int i = 0; i < seq; ++i) {
            for (int h = 0; h < nh; ++h)
                apply_rope(impl.q.data() +
                               (static_cast<size_t>(i) * nh + h) * dh,
                           impl.rope, 1, pos_offset + i);
            for (int h = 0; h < nkv; ++h)
                apply_rope(impl.k.data() +
                               (static_cast<size_t>(i) * nkv + h) * dh,
                           impl.rope, 1, pos_offset + i);
        }
        // append to the selected cache (fp32 or quantized paged)
        for (int i = 0; i < seq; ++i) {
            const float* ki = impl.k.data() + static_cast<size_t>(i) * nkv * dh;
            const float* vi = impl.v.data() + static_cast<size_t>(i) * nkv * dh;
            if (impl.kv_mode == KvMode::Fp32)
                impl.cache_fp32.append(l, pos_offset + i, ki, vi);
            else
                impl.cache_int8.append(l, pos_offset + i, ki, vi);
        }
        // attention + MLP
        std::vector<float> kbuf(static_cast<size_t>(dh));
        std::vector<float> vbuf(static_cast<size_t>(dh));
        for (int i = 0; i < seq; ++i) {
            float* xi = impl.x.data() + static_cast<size_t>(i) * dm;
            const int kv_len = pos_offset + i + 1;  // valid cache slots
            for (int h = 0; h < nh; ++h) {
                const int kvh = h / kRep;  // repeat_kv: identity on nano
                const float* qh = impl.q.data() +
                                  (static_cast<size_t>(i) * nh + h) * dh;
                float* sc = impl.scores.data();
                for (int p = 0; p < kv_len; ++p) {
                    const float* kr;
                    if (impl.kv_mode == KvMode::Fp32) {
                        kr = impl.cache_fp32.k_row(l, kvh, p);
                    } else {
                        impl.cache_int8.k_row(l, kvh, p / PagedInt8KvCache::kPageTokens,
                                              p % PagedInt8KvCache::kPageTokens,
                                              kbuf.data());
                        kr = kbuf.data();
                    }
                    float dot = 0.0f;
                    for (int j = 0; j < dh; ++j)
                        dot = std::fmaf(qh[j], kr[j], dot);
                    sc[p] = dot * kInvSqrtDHead;
                }
                softmax_inplace(sc, kv_len);
                float* oh = impl.attn_out.data() + static_cast<size_t>(h) * dh;
                std::memset(oh, 0, sizeof(float) * dh);
                for (int p = 0; p < kv_len; ++p) {
                    const float w = sc[p];
                    const float* vr;
                    if (impl.kv_mode == KvMode::Fp32) {
                        vr = impl.cache_fp32.v_row(l, kvh, p);
                    } else {
                        impl.cache_int8.v_row(l, kvh, p / PagedInt8KvCache::kPageTokens,
                                              p % PagedInt8KvCache::kPageTokens,
                                              vbuf.data());
                        vr = vbuf.data();
                    }
                    for (int j = 0; j < dh; ++j)
                        oh[j] = std::fmaf(w, vr[j], oh[j]);
                }
            }
            gemv_weight(impl, l, kWo, impl.attn_res.data(), impl.attn_out.data(),
                        dm, nh * dh);
            for (int j = 0; j < dm; ++j) xi[j] += impl.attn_res[j];

            rms_norm(xi, impl.h2.data(), impl.layer[l][kFfnNorm], dm,
                     impl.m.rms_norm_eps);
            gemv_weight(impl, l, kWGate, impl.gate.data(), impl.h2.data(), dff, dm);
            gemv_weight(impl, l, kWUp, impl.up.data(), impl.h2.data(), dff, dm);
            swiglu(impl.gate.data(), impl.up.data(), impl.gate.data(), dff);
            gemv_weight(impl, l, kWDown, impl.attn_res.data(), impl.gate.data(),
                        dm, dff);
            for (int j = 0; j < dm; ++j) xi[j] += impl.attn_res[j];
        }
    }

    // final norm over the LAST position, tied-embedding head
    const float* xlast = impl.x.data() + static_cast<size_t>(seq - 1) * dm;
    rms_norm(xlast, impl.h.data(), impl.final_norm, dm, impl.m.rms_norm_eps);
    gemv_head(impl, impl.logits.data(), impl.h.data());
    impl.cache_len = pos_offset + seq;
}

std::vector<int> Transformer::greedy_generate(const std::vector<int>& prompt,
                                              int n_new, int eos_id) {
    Impl& impl = *impl_;
    std::vector<int> out;
    out.reserve(static_cast<size_t>(std::max(n_new, 0)));
    logits_last(prompt, impl.cache_len);
    for (int k = 0; k < n_new; ++k) {
        // exact argmax, lowest index on ties (max_element keeps first)
        const float* lg = impl.logits.data();
        const int nxt = static_cast<int>(
            std::max_element(lg, lg + impl.m.vocab) - lg);
        if (eos_id >= 0 && nxt == eos_id) break;
        out.push_back(nxt);
        logits_last(std::vector<int>{nxt}, impl.cache_len);
    }
    return out;
}

int Transformer::cache_len() const { return impl_->cache_len; }

// Sampling helpers used by non-greedy generation (temperature, top-p).
int sample_token(const float* logits, int vocab, float temperature,
                 float top_p, unsigned* rng_state) {
    if (temperature <= 0.0f) {
        return static_cast<int>(
            std::max_element(logits, logits + vocab) - logits);
    }
    // softmax over the tempered logits
    std::vector<float> probs(static_cast<size_t>(vocab));
    float mx = logits[0];
    for (int i = 1; i < vocab; ++i) mx = std::max(mx, logits[i] / temperature);
    double sum = 0.0;
    for (int i = 0; i < vocab; ++i) {
        const float e = std::exp(logits[i] / temperature - mx);
        probs[static_cast<size_t>(i)] = e;
        sum += e;
    }
    for (int i = 0; i < vocab; ++i) probs[static_cast<size_t>(i)] /= static_cast<float>(sum);

    // top-p: keep the smallest prefix of descending probs with mass >= top_p
    std::vector<int> order(static_cast<size_t>(vocab));
    for (int i = 0; i < vocab; ++i) order[static_cast<size_t>(i)] = i;
    std::sort(order.begin(), order.end(),
              [&](int a, int b) { return probs[static_cast<size_t>(a)] > probs[static_cast<size_t>(b)]; });
    double cum = 0.0;
    size_t keep = 0;
    for (; keep < order.size(); ++keep) {
        cum += probs[static_cast<size_t>(order[keep])];
        if (cum >= static_cast<double>(top_p)) {
            ++keep;
            break;
        }
    }
    if (keep == 0) keep = 1;
    // renormalize the kept prefix and sample
    double kept = 0.0;
    for (size_t i = 0; i < keep; ++i)
        kept += probs[static_cast<size_t>(order[i])];
    // deterministic LCG from rng_state
    *rng_state = *rng_state * 1664525u + 1013904223u;
    const double u = static_cast<double>(*rng_state >> 8) / 16777216.0;
    double acc = 0.0;
    for (size_t i = 0; i < keep; ++i) {
        acc += probs[static_cast<size_t>(order[i])] / kept;
        if (u <= acc) return order[i];
    }
    return order[keep - 1];
}

}  // namespace heph
