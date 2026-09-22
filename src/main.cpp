// CLI dispatch: golden|generate|tokenize|quantize|bench (flags fixed in
// the Makefile contract comments).
//
// Exit codes: 0 success, 1 runtime failure (the error message is on
// stderr), 2 usage error.

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "bench/bench.h"
#include "loader/loader.h"
#include "model/model_def.h"
#include "model/transformer.h"
#include "quant/quantize.h"
#include "tokenizer/bpe.h"

namespace {

using namespace heph;

struct Args {
    std::vector<std::pair<std::string, std::string>> kv;
    bool has(const std::string& key) const {
        for (const auto& p : kv)
            if (p.first == key) return true;
        return false;
    }
    std::string str(const std::string& key,
                    const std::string& fallback = "") const {
        for (const auto& p : kv)
            if (p.first == key) return p.second;
        return fallback;
    }
    long num(const std::string& key, long fallback) const {
        for (const auto& p : kv)
            if (p.first == key) return std::stol(p.second);
        return fallback;
    }
    float fnum(const std::string& key, float fallback) const {
        for (const auto& p : kv)
            if (p.first == key) return std::stof(p.second);
        return fallback;
    }
};

Args parse_args(int argc, char** argv, int first) {
    Args a;
    for (int i = first; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg.rfind("--", 0) != 0)
            throw std::runtime_error("unexpected argument: " + arg);
        std::string key = arg.substr(2);
        if (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0) {
            a.kv.emplace_back(key, argv[++i]);
        } else {
            a.kv.emplace_back(key, "");
        }
    }
    return a;
}

std::vector<const float*> build_layer_weights(const SafetensorsFile& f,
                                              int n_layer) {
    static const char* kOrder[9] = {
        "attn_norm.weight", "attn.wq.weight", "attn.wk.weight",
        "attn.wv.weight",   "attn.wo.weight", "ffn_norm.weight",
        "ffn.w_gate.weight", "ffn.w_up.weight", "ffn.w_down.weight",
    };
    std::vector<const float*> lw;
    lw.reserve(static_cast<size_t>(n_layer) * 9);
    for (int l = 0; l < n_layer; ++l)
        for (const char* suffix : kOrder)
            lw.push_back(f.data("layer." + std::to_string(l) + "." + suffix));
    return lw;
}

std::vector<std::string> read_prompt_lines(const std::string& path) {
    std::ifstream fh(path);
    if (!fh) throw std::runtime_error("cannot open prompts file: " + path);
    std::vector<std::string> lines;
    std::string line;
    while (std::getline(fh, line))
        if (!line.empty() && line[0] != '#') lines.push_back(line);
    return lines;
}

Transformer build_transformer(const Manifest& manifest,
                              const std::string& weights_path) {
    SafetensorsFile* f = new SafetensorsFile(weights_path);
    // Leaks intentionally: the process is short-lived and the Transformer
    // keeps raw views into the mapped weights.
    return Transformer(manifest, f->data("embedding.weight"),
                       build_layer_weights(*f, manifest.n_layer),
                       f->data("final_norm.weight"));
}

int cmd_golden(const Args& a) {
    const Manifest m = load_manifest(a.str("manifest"));
    BpeTokenizer tok;
    tok.load(a.str("tokenizer"));
    Transformer tf = build_transformer(m, a.str("weights"));
    const auto prompts = read_prompt_lines(a.str("prompts"));
    const std::string dump = a.str("dump");
    if (dump.empty()) throw std::runtime_error("golden: --dump is required");
    if (a.num("tokens", 64) <= 0)
        throw std::runtime_error("golden: --tokens must be positive");

    std::ofstream out(dump, std::ios::binary);
    if (!out) throw std::runtime_error("cannot write dump: " + dump);
    for (const std::string& line : prompts) {
        const std::vector<int> ids = tok.encode(line);
        tf.reset_cache();
        tf.logits_last(ids, 0);
        out.write(reinterpret_cast<const char*>(tf.last_logits()),
                  sizeof(float) * static_cast<size_t>(m.vocab));
    }
    return 0;
}

int cmd_generate(const Args& a) {
    const Manifest m = load_manifest(a.str("manifest"));
    BpeTokenizer tok;
    tok.load(a.str("tokenizer"));
    Transformer tf = build_transformer(m, a.str("weights"));
    const bool greedy = a.has("greedy");
    const float temperature = a.fnum("temperature", 1.0f);
    const float top_p = a.fnum("top-p", 1.0f);
    const int max_new = static_cast<int>(a.num("max-new-tokens", 64));
    // Non-interactive generation never stops at eos: the golden contract
    // compares against oracle.greedy(ids, n) which has no stop token.
    const int eos = a.has("interactive") ? tok.eos_id() : -1;
    unsigned rng = 20260922u;

    auto emit = [&](const std::vector<int>& ids) {
        // stream text token-by-token on stdout
        std::string text = tok.decode(ids);
        std::cout << text << std::flush;
    };

    if (a.has("interactive")) {
        std::cout << "heph interactive (" << quant_mode_name(QuantMode::Fp32)
                  << ", greedy=" << greedy << ") — empty line exits\n";
        std::string line;
        while (std::cout << "> ", std::getline(std::cin, line)) {
            if (line.empty()) break;
            const std::vector<int> ids = tok.encode(line);
            tf.reset_cache();
            if (greedy) {
                const std::vector<int> gen =
                    tf.greedy_generate(ids, max_new, eos);
                // stream as it decodes: re-emit whole text (the cache is
                // already advanced; decode is cheap)
                emit(gen);
            } else {
                std::vector<int> gen;
                tf.logits_last(ids, 0);
                for (int k = 0; k < max_new; ++k) {
                    const int nxt = sample_token(tf.last_logits(), m.vocab,
                                                 temperature, top_p, &rng);
                    if (nxt == eos) break;
                    gen.push_back(nxt);
                    emit({nxt});
                    tf.logits_last(std::vector<int>{nxt}, tf.cache_len());
                }
                if (gen.empty()) std::cout << std::flush;
            }
            std::cout << "\n";
        }
        return 0;
    }

    const std::string prompt = a.str("prompt");
    if (prompt.empty()) throw std::runtime_error("generate: --prompt required");
    const std::vector<int> ids = tok.encode(prompt);
    tf.reset_cache();
    std::vector<int> gen;
    if (greedy) {
        gen = tf.greedy_generate(ids, max_new, eos);
    } else {
        tf.logits_last(ids, 0);
        for (int k = 0; k < max_new; ++k) {
            const int nxt =
                sample_token(tf.last_logits(), m.vocab, temperature, top_p, &rng);
            if (nxt == eos) break;
            gen.push_back(nxt);
            tf.logits_last(std::vector<int>{nxt}, tf.cache_len());
        }
    }
    if (a.has("out")) {
        std::ofstream out(a.str("out"));
        if (!out) throw std::runtime_error("cannot write ids: " + a.str("out"));
        for (size_t i = 0; i < gen.size(); ++i) {
            if (i) out << ' ';
            out << gen[i];
        }
        out << "\n";
    } else {
        emit(gen);
        std::cout << "\n";
    }
    return 0;
}

int cmd_tokenize(const Args& a) {
    BpeTokenizer tok;
    tok.load(a.str("tokenizer"));
    std::ifstream in(a.str("file"));
    if (!in) throw std::runtime_error("cannot open input: " + a.str("file"));
    std::ofstream out(a.str("out"));
    if (!out) throw std::runtime_error("cannot open output: " + a.str("out"));
    std::string line;
    while (std::getline(in, line)) {
        const std::vector<int> ids = tok.encode(line);
        for (size_t i = 0; i < ids.size(); ++i) {
            if (i) out << ' ';
            out << ids[i];
        }
        out << "\n";
    }
    return 0;
}

int cmd_quantize(const Args& a) {
    const Manifest m = load_manifest(a.str("manifest"));
    const QuantMode mode = parse_quant_mode(a.str("mode"));
    if (mode == QuantMode::Fp32)
        throw std::runtime_error("quantize: mode must be int8, int4 or ternary");
    const int group = static_cast<int>(a.num("group", 128));
    const std::string out_dir = a.str("out");
    SafetensorsFile f(a.str("weights"));

    std::vector<std::string> names;
    names.push_back("embedding.weight");
    for (int l = 0; l < m.n_layer; ++l) {
        const std::string p = "layer." + std::to_string(l) + ".";
        names.push_back(p + "attn_norm.weight");
        names.push_back(p + "attn.wq.weight");
        names.push_back(p + "attn.wk.weight");
        names.push_back(p + "attn.wv.weight");
        names.push_back(p + "attn.wo.weight");
        names.push_back(p + "ffn_norm.weight");
        names.push_back(p + "ffn.w_gate.weight");
        names.push_back(p + "ffn.w_up.weight");
        names.push_back(p + "ffn.w_down.weight");
    }
    names.push_back("final_norm.weight");

    std::ostringstream tsv;
    std::vector<uint8_t> blob;
    for (const std::string& name : names) {
        const auto shape = f.shape(name);  // [rows, cols]
        const long rows = static_cast<long>(shape[0]);
        const long cols = static_cast<long>(shape[1]);
        const bool is_norm = name.find("norm") != std::string::npos;
        std::string dtype;
        const size_t offset = blob.size();
        const float* w = f.data(name);
        const auto put = [&](const uint8_t* src, size_t n) {
            blob.insert(blob.end(), src, src + n);
        };
        if (is_norm) {
            // Norm vectors are not GEMV weights: stored raw fp32 in every
            // mode (documented in the quantize.h contract).
            dtype = "F32";
            put(reinterpret_cast<const uint8_t*>(w),
                sizeof(float) * static_cast<size_t>(rows) * cols);
        } else {
            const QuantizedMat qm = quantize_matrix(w, static_cast<int>(rows),
                                                    static_cast<int>(cols),
                                                    mode, group);
            dtype = qm.mode == QuantMode::Int8   ? "Q8"
                    : qm.mode == QuantMode::Int4 ? "Q4"
                                                 : "T1";
            put(qm.packed.data(), qm.packed.size());
            if (qm.mode == QuantMode::Int4) {
                put(reinterpret_cast<const uint8_t*>(qm.scales_fp16.data()),
                    sizeof(uint16_t) * qm.scales_fp16.size());
            } else {
                put(reinterpret_cast<const uint8_t*>(qm.scales.data()),
                    sizeof(float) * qm.scales.size());
            }
        }
        tsv << name << "\t" << rows << " x " << cols << "\t" << dtype << "\t"
            << offset << "\t" << blob.size() - offset << "\n";
    }

    const std::string cmd = "mkdir -p " + out_dir;
    if (std::system(cmd.c_str()) != 0)
        throw std::runtime_error("cannot create dir: " + out_dir);
    {
        std::ofstream fh(out_dir + "/weights.bin", std::ios::binary);
        if (!fh) throw std::runtime_error("cannot write weights.bin");
        fh.write(reinterpret_cast<const char*>(blob.data()),
                 static_cast<std::streamsize>(blob.size()));
    }
    {
        std::ofstream fh(out_dir + "/tensors.tsv");
        if (!fh) throw std::runtime_error("cannot write tensors.tsv");
        fh << tsv.str();
    }
    std::cout << "quantized " << names.size() << " tensors ("
              << quant_mode_name(mode) << ", group " << group << ") -> "
              << out_dir << " (" << blob.size() << " bytes)\n";
    return 0;
}

int cmd_bench(const Args& a) {
    const Manifest m = load_manifest(a.str("manifest"));
    const QuantMode mode = parse_quant_mode(a.str("mode", "fp32"));
    const BenchResult r = run_bench(
        m, a.str("weights"), a.str("tokenizer"), mode,
        "tests/golden/prompts.txt", static_cast<int>(a.num("prompts", 50)),
        static_cast<int>(a.num("warmup", 5)),
        static_cast<int>(a.num("max-new-tokens", 128)), a.str("holdout"));
    write_bench_json(a.str("out"), r);
    std::cout << "bench " << r.mode << ": " << r.tokens_per_s_decode
              << " tokens/s (decode), ttft " << r.ttft_s << " s, rss "
              << r.peak_rss_mb << " MB";
    if (r.has_ppl) std::cout << ", ppl " << r.ppl << " (n=" << r.ppl_tokens
                             << ")";
    std::cout << "\n";
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr
            << "usage: heph {golden|generate|tokenize|quantize|bench} ...\n"
            << "flags are fixed by the Makefile contract comments\n";
        return 2;
    }
    const std::string cmd = argv[1];
    try {
        if (cmd == "golden") return cmd_golden(parse_args(argc, argv, 2));
        if (cmd == "generate") return cmd_generate(parse_args(argc, argv, 2));
        if (cmd == "tokenize") return cmd_tokenize(parse_args(argc, argv, 2));
        if (cmd == "quantize") return cmd_quantize(parse_args(argc, argv, 2));
        if (cmd == "bench") return cmd_bench(parse_args(argc, argv, 2));
        std::cerr << "heph: unknown subcommand '" << cmd << "'\n";
        return 2;
    } catch (const std::exception& e) {
        std::cerr << "heph " << cmd << ": " << e.what() << "\n";
        return 1;
    }
}
