// Canonical tensor names and the architecture contract for HEPHAESTUS.
//
// The engine supports EXACTLY the PROMETHEUS-NS nano decoder and nothing
// else. Hardcoding the architecture is deliberate (spec constraint): a
// fixed shape removes every generic-dispatch branch from the hot path,
// lets the GEMV kernels use compile-time-friendly strides, and keeps the
// KV-cache layout statically known. Supporting more architectures is out
// of scope by design; the manifest is still validated against these
// constants at load time and the engine refuses any disagreement.

#pragma once

namespace heph {

// nano architecture contract. The export manifest must match every value.
constexpr int kNLayers = 6;
constexpr int kNHead = 6;
constexpr int kNKvHead = 6;       // GQA group size 1 for nano
constexpr int kDModel = 384;
constexpr int kDFf = 1024;
constexpr int kDHead = kDModel / kNHead;  // 64
constexpr int kMaxSeq = 256;
constexpr int kVocab = 8000;

// Canonical tensor names, expanded for the six nano layers. The loader
// keys its weight table by exactly these strings; scripts/export_mapping.yaml
// must produce them from the PROMETHEUS-NS checkpoint state_dict.
static const char* const kCanonicalTensorNames[] = {
    "embedding.weight",
    "layer.0.attn_norm.weight", "layer.0.attn.wq.weight",
    "layer.0.attn.wk.weight",   "layer.0.attn.wv.weight",
    "layer.0.attn.wo.weight",   "layer.0.ffn_norm.weight",
    "layer.0.ffn.w_gate.weight", "layer.0.ffn.w_up.weight",
    "layer.0.ffn.w_down.weight",
    "layer.1.attn_norm.weight", "layer.1.attn.wq.weight",
    "layer.1.attn.wk.weight",   "layer.1.attn.wv.weight",
    "layer.1.attn.wo.weight",   "layer.1.ffn_norm.weight",
    "layer.1.ffn.w_gate.weight", "layer.1.ffn.w_up.weight",
    "layer.1.ffn.w_down.weight",
    "layer.2.attn_norm.weight", "layer.2.attn.wq.weight",
    "layer.2.attn.wk.weight",   "layer.2.attn.wv.weight",
    "layer.2.attn.wo.weight",   "layer.2.ffn_norm.weight",
    "layer.2.ffn.w_gate.weight", "layer.2.ffn.w_up.weight",
    "layer.2.ffn.w_down.weight",
    "layer.3.attn_norm.weight", "layer.3.attn.wq.weight",
    "layer.3.attn.wk.weight",   "layer.3.attn.wv.weight",
    "layer.3.attn.wo.weight",   "layer.3.ffn_norm.weight",
    "layer.3.ffn.w_gate.weight", "layer.3.ffn.w_up.weight",
    "layer.3.ffn.w_down.weight",
    "layer.4.attn_norm.weight", "layer.4.attn.wq.weight",
    "layer.4.attn.wk.weight",   "layer.4.attn.wv.weight",
    "layer.4.attn.wo.weight",   "layer.4.ffn_norm.weight",
    "layer.4.ffn.w_gate.weight", "layer.4.ffn.w_up.weight",
    "layer.4.ffn.w_down.weight",
    "layer.5.attn_norm.weight", "layer.5.attn.wq.weight",
    "layer.5.attn.wk.weight",   "layer.5.attn.wv.weight",
    "layer.5.attn.wo.weight",   "layer.5.ffn_norm.weight",
    "layer.5.ffn.w_gate.weight", "layer.5.ffn.w_up.weight",
    "layer.5.ffn.w_down.weight",
    "final_norm.weight",
    "lm_head.weight",  // tied to embedding.weight; verified equal at export
};
static constexpr int kCanonicalTensorCount =
    static_cast<int>(sizeof(kCanonicalTensorNames) / sizeof(kCanonicalTensorNames[0]));

}  // namespace heph
