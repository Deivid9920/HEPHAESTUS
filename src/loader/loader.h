// Safetensors loader and manifest/tensors.tsv parsing.
//
// The loader memory-maps artifacts/nano_fp32.safetensors, parses the
// JSON header, and exposes fp32 tensor views at the absolute offsets
// recorded in tensors.tsv (header + 8-byte length prefix). Integrity:
// the whole-file SHA-256 must equal manifest.weights_sha256 and every
// tensor range must match the header's data_offsets; per-tensor SHA-256
// digests are available for cross-checks against fixtures.

#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace heph {

struct Manifest {
    int n_layer = 0;
    int n_head = 0;
    int n_kv_head = 0;
    int d_model = 0;
    int d_ff = 0;
    int max_seq = 0;
    int vocab = 0;
    int d_head = 0;
    float rms_norm_eps = 0.0f;
    float rope_theta = 0.0f;
    std::string weights_sha256;
    std::string weights_file;
};

struct TensorEntry {
    std::string name;
    std::string shape;   // "rows x cols" as written in tensors.tsv
    std::string dtype;
    uint64_t offset = 0;  // absolute file offset
    uint64_t bytes = 0;
};

// Parses model_manifest.txt ("key = value" lines, arch as the Python
// dict repr). Throws std::runtime_error on missing keys or any
// disagreement with the nano contract (src/model/model_def.h).
Manifest load_manifest(const std::string& path);

// Parses tensors.tsv (name, shape, dtype, absolute offset, bytes).
std::vector<TensorEntry> load_tensors_table(const std::string& path);

class SafetensorsFile {
  public:
    explicit SafetensorsFile(const std::string& path);
    ~SafetensorsFile();
    SafetensorsFile(const SafetensorsFile&) = delete;
    SafetensorsFile& operator=(const SafetensorsFile&) = delete;

    // Whole-file digest; compared against Manifest::weights_sha256.
    std::string file_sha256() const;
    // Per-tensor digest over the tensor's byte range.
    std::string tensor_sha256(const std::string& name) const;
    // Read-only fp32 view of one tensor's data (row-major).
    const float* data(const std::string& name) const;
    std::vector<uint64_t> shape(const std::string& name) const;

  private:
    struct Impl;
    Impl* impl_;
};

}  // namespace heph
