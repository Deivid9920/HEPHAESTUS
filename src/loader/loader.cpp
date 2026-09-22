// Safetensors loader: manifest + tensors.tsv parsing and the mmap'd
// weights file with digest verification (phase-1 contract).

#include "loader/loader.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cmath>
#include <cstring>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include "core/json.h"
#include "core/sha256.h"

namespace heph {

namespace {

std::string read_file(const std::string& path) {
    std::ifstream fh(path, std::ios::binary);
    if (!fh) throw std::runtime_error("cannot open file: " + path);
    std::ostringstream ss;
    ss << fh.rdbuf();
    return ss.str();
}

int parse_int_strict(const std::string& text, const char* what) {
    try {
        size_t consumed = 0;
        const int v = std::stoi(text, &consumed);
        if (consumed != text.size())
            throw std::invalid_argument("trailing");
        return v;
    } catch (const std::exception&) {
        throw std::runtime_error(std::string("manifest: bad int for ") + what +
                                 ": '" + text + "'");
    }
}

float parse_float_strict(const std::string& text, const char* what) {
    try {
        size_t consumed = 0;
        const float v = std::stof(text, &consumed);
        if (consumed != text.size())
            throw std::invalid_argument("trailing");
        return v;
    } catch (const std::exception&) {
        throw std::runtime_error(std::string("manifest: bad float for ") + what +
                                 ": '" + text + "'");
    }
}

// The manifest writes the arch as the Python dict repr, e.g.
// arch = {'n_layer': 6, 'n_head': 6, ...}. Extract one int field.
int arch_field(const std::string& repr, const char* key) {
    const std::string needle = std::string("'") + key + "':";
    const size_t at = repr.find(needle);
    if (at == std::string::npos)
        throw std::runtime_error(std::string("manifest arch missing ") + key);
    size_t pos = at + needle.size();
    while (pos < repr.size() && repr[pos] == ' ') ++pos;
    size_t end = pos;
    while (end < repr.size() && (isdigit(static_cast<unsigned char>(repr[end]))))
        ++end;
    return parse_int_strict(repr.substr(pos, end - pos), key);
}

void validate_nano(const Manifest& m) {
    if (m.n_layer != 6 || m.n_head != 6 || m.n_kv_head != 6 || m.d_model != 384 ||
        m.d_ff != 1024 || m.max_seq != 256 || m.vocab != 8000 || m.d_head != 64) {
        throw std::runtime_error(
            "manifest disagrees with the nano contract (src/model/model_def.h): "
            "the engine supports exactly the PROMETHEUS-NS nano architecture");
    }
}

}  // namespace

Manifest load_manifest(const std::string& path) {
    const std::string text = read_file(path);
    Manifest m;
    bool have_sha = false;
    std::stringstream ss(text);
    std::string line;
    while (std::getline(ss, line)) {
        if (line.empty() || line[0] == '#') continue;
        const size_t eq = line.find('=');
        if (eq == std::string::npos) continue;
        const std::string key = line.substr(0, eq);
        std::string value = line.substr(eq + 1);
        while (!value.empty() && (value.front() == ' ')) value.erase(value.begin());
        while (!value.empty() && (value.back() == ' ' || value.back() == '\r'))
            value.pop_back();
        if (key == "arch") {
            m.n_layer = arch_field(value, "n_layer");
            m.n_head = arch_field(value, "n_head");
            m.n_kv_head = arch_field(value, "n_kv_head");
            m.d_model = arch_field(value, "d_model");
            m.d_ff = arch_field(value, "d_ff");
            m.max_seq = arch_field(value, "max_seq");
            m.vocab = arch_field(value, "vocab");
            // d_head may be None in the manifest; the contract derives it.
            m.d_head = m.d_model / m.n_head;
        } else if (key == "rms_norm_eps") {
            m.rms_norm_eps = parse_float_strict(value, "rms_norm_eps");
        } else if (key == "rope_theta") {
            m.rope_theta = parse_float_strict(value, "rope_theta");
        } else if (key == "weights_sha256") {
            m.weights_sha256 = value;
            have_sha = true;
        } else if (key == "weights_file") {
            m.weights_file = value;
        }
    }
    if (!have_sha || m.weights_file.empty())
        throw std::runtime_error("manifest missing weights_sha256/weights_file");
    validate_nano(m);
    return m;
}

std::vector<TensorEntry> load_tensors_table(const std::string& path) {
    std::ifstream fh(path);
    if (!fh) throw std::runtime_error("cannot open tensors table: " + path);
    std::vector<TensorEntry> entries;
    std::string line;
    while (std::getline(fh, line)) {
        if (line.empty()) continue;
        std::vector<std::string> parts;
        std::stringstream ls(line);
        std::string part;
        while (std::getline(ls, part, '\t')) parts.push_back(part);
        if (parts.size() != 5)
            throw std::runtime_error("tensors.tsv: expected 5 columns, got " +
                                     std::to_string(parts.size()));
        TensorEntry e;
        e.name = parts[0];
        e.shape = parts[1];
        e.dtype = parts[2];
        e.offset = static_cast<uint64_t>(parse_int_strict(parts[3], "offset"));
        e.bytes = static_cast<uint64_t>(parse_int_strict(parts[4], "bytes"));
        if (e.dtype != "F32")
            throw std::runtime_error("tensors.tsv: unsupported dtype " + e.dtype);
        entries.push_back(std::move(e));
    }
    return entries;
}

struct SafetensorsFile::Impl {
    int fd = -1;
    void* map = nullptr;
    size_t size = 0;
    uint64_t data_start = 0;
    std::map<std::string, std::vector<uint64_t>> shapes;
    std::map<std::string, std::pair<uint64_t, uint64_t>> offsets;
    std::string path;
};

SafetensorsFile::SafetensorsFile(const std::string& path) : impl_(new Impl) {
    impl_->path = path;
    impl_->fd = ::open(path.c_str(), O_RDONLY);
    if (impl_->fd < 0) throw std::runtime_error("cannot open weights: " + path);
    struct stat st;
    if (::fstat(impl_->fd, &st) != 0) {
        ::close(impl_->fd);
        throw std::runtime_error("cannot stat weights: " + path);
    }
    impl_->size = static_cast<size_t>(st.st_size);
    impl_->map = ::mmap(nullptr, impl_->size, PROT_READ, MAP_PRIVATE,
                        impl_->fd, 0);
    if (impl_->map == MAP_FAILED) {
        ::close(impl_->fd);
        throw std::runtime_error("mmap failed for " + path);
    }

    const uint8_t* bytes = static_cast<const uint8_t*>(impl_->map);
    uint64_t header_len = 0;
    std::memcpy(&header_len, bytes, sizeof(header_len));
    if (8 + header_len > impl_->size)
        throw std::runtime_error("safetensors header exceeds file size");
    const std::string header_text(
        reinterpret_cast<const char*>(bytes + 8), header_len);
    const JsonValue header = json_parse(header_text);
    if (!header.is_object())
        throw std::runtime_error("safetensors header is not a JSON object");
    impl_->data_start = 8 + header_len;
    for (const auto& [name, info] : header.object()) {
        if (name == "__metadata__") continue;
        if (!info.is_object()) throw std::runtime_error("bad tensor entry");
        const auto& obj = info.object();
        const auto dtype = obj.at("dtype").string();
        if (dtype != "F32")
            throw std::runtime_error("tensor " + name + ": unsupported dtype " + dtype);
        const auto& shape = obj.at("shape").array();
        std::vector<uint64_t> dims;
        for (const auto& d : shape) dims.push_back(static_cast<uint64_t>(d.number()));
        const auto& off = obj.at("data_offsets").array();
        const uint64_t start = static_cast<uint64_t>(off.at(0).number());
        const uint64_t end = static_cast<uint64_t>(off.at(1).number());
        uint64_t numel = 1;
        for (const uint64_t d : dims) numel *= d;
        if (end - start != numel * 4)
            throw std::runtime_error("tensor " + name + ": size mismatch");
        impl_->shapes[name] = dims;
        impl_->offsets[name] = {start, end};
    }
}

SafetensorsFile::~SafetensorsFile() {
    if (impl_->map) ::munmap(impl_->map, impl_->size);
    if (impl_->fd >= 0) ::close(impl_->fd);
    delete impl_;
}

std::string SafetensorsFile::file_sha256() const {
    return sha256_hex(static_cast<const uint8_t*>(impl_->map), impl_->size);
}

std::string SafetensorsFile::tensor_sha256(const std::string& name) const {
    const auto it = impl_->offsets.find(name);
    if (it == impl_->offsets.end())
        throw std::runtime_error("tensor_sha256: unknown tensor " + name);
    const uint8_t* bytes = static_cast<const uint8_t*>(impl_->map);
    return sha256_hex(bytes + impl_->data_start + it->second.first,
                      it->second.second - it->second.first);
}

const float* SafetensorsFile::data(const std::string& name) const {
    const auto it = impl_->offsets.find(name);
    if (it == impl_->offsets.end())
        throw std::runtime_error("weights: unknown tensor " + name);
    const uint8_t* bytes = static_cast<const uint8_t*>(impl_->map);
    return reinterpret_cast<const float*>(bytes + impl_->data_start +
                                          it->second.first);
}

std::vector<uint64_t> SafetensorsFile::shape(const std::string& name) const {
    const auto it = impl_->shapes.find(name);
    if (it == impl_->shapes.end())
        throw std::runtime_error("weights: unknown tensor " + name);
    return it->second;
}

}  // namespace heph
