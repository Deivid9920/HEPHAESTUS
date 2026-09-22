// Byte-level BPE runtime (reader, not trainer): loads vocab.json +
// merges.txt + specials.json produced by scripts/export_tokenizer.py.
//
// Encode pipeline (parity-tested against Hugging Face by
// tests/golden/test_tokenize_parity.py):
//   1. GPT-2 byte -> unicode alphabet map BEFORE vocab lookup
//      ('Ġ' = space, byte 0x20 -> U+0120);
//   2. the ByteLevel pre-split BEFORE merges — implemented as a manual
//      scanner (std::regex has no Unicode property support and third
//      party regex libraries are out of scope);
//   3. BPE merges by rank: always merge the pair with the LOWEST rank
//      in merges.txt, never the first pair found left-to-right;
//   4. encode never injects special tokens and never emits <|unk|>:
//      byte-level coverage is total, so unk firing means the byte map
//      is broken.
//
// Multi-byte UTF-8 caution: mapped characters like 'Ġ' occupy two bytes
// in UTF-8 (0xC4 0xA0); all merge/vocab work happens on codepoint
// sequences, never on raw std::string indices.

#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace heph {

class BpeTokenizer {
  public:
    void load(const std::string& dir);  // vocab.json, merges.txt, specials.json

    std::vector<int> encode(const std::string& text) const;
    std::string decode(const std::vector<int>& ids) const;

    int bos_id() const;
    int eos_id() const;
    int unk_id() const;
    int vocab_size() const;

  private:
    std::unordered_map<std::string, int> vocab_;   // unicode-space piece -> id
    std::vector<std::string> id_to_piece_;
    std::unordered_map<uint64_t, int> merge_rank_; // (left, right) packed -> rank
    std::vector<int> byte_token_;                  // 256 mapped base tokens
    int bos_ = -1;
    int eos_ = -1;
    int unk_ = -1;
};

}  // namespace heph
