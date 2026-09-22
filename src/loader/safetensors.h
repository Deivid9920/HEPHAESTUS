#ifndef _SRC_LOADER_SAFETENSORS_H_
#define _SRC_LOADER_SAFETENSORS_H_


// Own safetensors loader + artifact parsers.
//
// Format: 8-byte little-endian header length, JSON header
// {name: {dtype, shape, data_offsets:[start,end]}}, then raw data.
// The loader validates dtype == F32 and keeps a read-only view into the
// file bytes (no copies). tensors.tsv offsets written by the exporter
// are absolute: 8 + header_len + start.
//
// FASE 1 VERIF: tensor_sha256() must return the digest of every mapped
// tensor region; the golden tests compare against independent digests
// of the same byte ranges of the weights file.

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace heph {

std::string sha256_hex(const uint8_t* data, size_t n);

struct SafeTensors {
  static SafeTensors load(const std::string& path);
  bool has(const std::string& name) const;
  const float* data(const std::string& name) const;
  const std::vector<int64_t>& shape(const std::string& name) const;
  size_t count() const;
  const std::vector<std::string>& names() const;
  std::vector<std::pair<std::string, std::string>> tensor_sha256() const;

 private:
  std::vector<uint8_t> blob_;
  std::vector<std::string> names_;
  struct Entry {
    size_t byte_offset = 0;
    size_t nbytes = 0;
    std::vector<int64_t> shape;
  };
  std::map<std::string, Entry> entries_;
};

struct Manifest {
  int d_model = 0, n_layer = 0, n_head = 0, n_kv_head = 0;
  int d_head = 0, d_ff = 0, vocab = 0, max_seq = 0;
  float rms_norm_eps = 0.0f, rope_theta = 0.0f;
  std::string weights_file, weights_sha256;
  static Manifest load(const std::string& path);
};

struct TensorsTable {
  struct Row {
    std::string name;
    std::vector<int64_t> shape;
    std::string dtype;
    size_t offset = 0;
    size_t nbytes = 0;
  };
  static std::vector<Row> load(const std::string& path);
};

}  // namespace heph


#endif  // _SRC_LOADER_SAFETENSORS_H_
