#include "tokenizer/bpe.h"
#include <stdexcept>
namespace heph {
void BpeTokenizer::load(const std::string&) {
  throw std::runtime_error("heph: tokenizer not implemented (phase 2)");
}
std::vector<int> BpeTokenizer::encode(const std::string&) const {
  throw std::runtime_error("heph: tokenizer not implemented (phase 2)");
}
std::string BpeTokenizer::decode(const std::vector<int>&) const {
  throw std::runtime_error("heph: tokenizer not implemented (phase 2)");
}
int BpeTokenizer::bos_id() const {
  throw std::runtime_error("heph: tokenizer not implemented (phase 2)");
}
int BpeTokenizer::eos_id() const {
  throw std::runtime_error("heph: tokenizer not implemented (phase 2)");
}
int BpeTokenizer::unk_id() const {
  throw std::runtime_error("heph: tokenizer not implemented (phase 2)");
}
int BpeTokenizer::vocab_size() const {
  throw std::runtime_error("heph: tokenizer not implemented (phase 2)");
}
}  // namespace heph
