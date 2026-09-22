#include "bench/bench.h"
#include <stdexcept>
namespace heph {
PromptSource load_bench_prompts(const std::string&, int) {
  throw std::runtime_error("heph: bench not implemented (phase 6)");
}
BenchResult run_bench(const Manifest&, const std::string&,
                      const std::string&, QuantMode, int, int, int,
                      const std::string&) {
  throw std::runtime_error("heph: bench not implemented (phase 6)");
}
void write_bench_json(const std::string&, const BenchResult&) {
  throw std::runtime_error("heph: bench not implemented (phase 6)");
}
}  // namespace heph
