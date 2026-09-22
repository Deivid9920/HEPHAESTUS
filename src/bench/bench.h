// Benchmark suite: {fp32, int8, int4, ternary} x {decode tokens/s
// (batch 1), TTFT, peak RSS, perplexity}. Real measured values only:
// every number written here comes from a stopwatch or getrusage, never
// from a model. Methodology (docs/benchmark.md): warmup prompts are
// discarded, medians are over bench.prompts prompts, prompt prefill is
// excluded from the decode rate and reported as TTFT, peak RSS is the
// process lifetime peak from getrusage(RUSAGE_SELF).ru_maxrss. The
// optional --holdout run adds ppl + ppl_tokens as mean NLL over
// non-overlapping max_seq windows of the frozen PROMETHEUS-NS holdout.

#pragma once

#include <string>
#include <vector>

#include "loader/loader.h"
#include "quant/quantize.h"

namespace heph {

struct BenchResult {
    std::string mode;
    double tokens_per_s_decode = 0.0;
    double ttft_s = 0.0;
    double peak_rss_mb = 0.0;
    std::vector<double> per_prompt_ms;
    double ppl = 0.0;      // only with --holdout
    long ppl_tokens = 0;   // only with --holdout
    bool has_ppl = false;
};

struct PromptSource {
    std::vector<std::string> lines;  // already filtered (non-empty, no #)
};

PromptSource load_bench_prompts(const std::string& path, int n);
BenchResult run_bench(const Manifest& manifest, const std::string& weights_path,
                      const std::string& tokenizer_dir, QuantMode mode,
                      const std::string& prompts_file, int n_prompts,
                      int warmup, int max_new_tokens,
                      const std::string& holdout_dir);
void write_bench_json(const std::string& path, const BenchResult& r);

}  // namespace heph
