// Minimal JSON reader: recursive descent over objects, arrays, strings
// (standard escapes plus \uXXXX with surrogate pairs), doubles, booleans
// and null. Sufficient for the safetensors header, vocab.json and
// specials.json; deliberately tiny because the engine has zero
// third-party dependencies.

#include "core/json.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <stdexcept>

namespace heph {

namespace {

class Parser {
  public:
    explicit Parser(const std::string& text) : s_(text) {}

    JsonValue parse() {
        skip_ws();
        JsonValue v = parse_value();
        skip_ws();
        if (pos_ != s_.size())
            fail("trailing characters after JSON value");
        return v;
    }

  private:
    const std::string& s_;
    size_t pos_ = 0;

    void fail(const char* what) const {
        std::fprintf(stderr, "json error at byte %zu: %s\n", pos_, what);
        throw std::runtime_error(std::string("json parse error: ") + what);
    }

    void skip_ws() {
        while (pos_ < s_.size()) {
            const char c = s_[pos_];
            if (c == ' ' || c == '\t' || c == '\n' || c == '\r') ++pos_;
            else break;
        }
    }

    char peek() const {
        if (pos_ >= s_.size()) fail("unexpected end of input");
        return s_[pos_];
    }

    void expect(char c) {
        if (pos_ >= s_.size() || s_[pos_] != c)
            fail("unexpected character");
        ++pos_;
    }

    bool consume(char c) {
        if (pos_ < s_.size() && s_[pos_] == c) {
            ++pos_;
            return true;
        }
        return false;
    }

    static void append_utf8(std::string& out, uint32_t cp) {
        if (cp < 0x80) {
            out += static_cast<char>(cp);
        } else if (cp < 0x800) {
            out += static_cast<char>(0xC0 | (cp >> 6));
            out += static_cast<char>(0x80 | (cp & 0x3F));
        } else if (cp < 0x10000) {
            out += static_cast<char>(0xE0 | (cp >> 12));
            out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
            out += static_cast<char>(0x80 | (cp & 0x3F));
        } else {
            out += static_cast<char>(0xF0 | (cp >> 18));
            out += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
            out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
            out += static_cast<char>(0x80 | (cp & 0x3F));
        }
    }

    static uint32_t parse_hex4(const std::string& s, size_t at) {
        uint32_t v = 0;
        for (int i = 0; i < 4; ++i) {
            const char c = s[at + static_cast<size_t>(i)];
            v <<= 4;
            if (c >= '0' && c <= '9') v |= static_cast<uint32_t>(c - '0');
            else if (c >= 'a' && c <= 'f') v |= static_cast<uint32_t>(c - 'a' + 10);
            else if (c >= 'A' && c <= 'F') v |= static_cast<uint32_t>(c - 'A' + 10);
            else throw std::runtime_error("json parse error: bad \\u escape");
        }
        return v;
    }

    std::string parse_string() {
        expect('"');
        std::string out;
        while (true) {
            if (pos_ >= s_.size()) fail("unterminated string");
            const char c = s_[pos_++];
            if (c == '"') break;
            if (c != '\\') {
                out += c;
                continue;
            }
            if (pos_ >= s_.size()) fail("bad escape");
            const char e = s_[pos_++];
            switch (e) {
                case '"': out += '"'; break;
                case '\\': out += '\\'; break;
                case '/': out += '/'; break;
                case 'b': out += '\b'; break;
                case 'f': out += '\f'; break;
                case 'n': out += '\n'; break;
                case 'r': out += '\r'; break;
                case 't': out += '\t'; break;
                case 'u': {
                    uint32_t cp = parse_hex4(s_, pos_);
                    pos_ += 4;
                    if (cp >= 0xD800 && cp <= 0xDBFF && pos_ + 1 < s_.size() &&
                        s_[pos_] == '\\' && s_[pos_ + 1] == 'u') {
                        const uint32_t lo = parse_hex4(s_, pos_ + 2);
                        if (lo >= 0xDC00 && lo <= 0xDFFF) {
                            cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                            pos_ += 6;
                        }
                    }
                    append_utf8(out, cp);
                    break;
                }
                default: fail("bad escape character");
            }
        }
        return out;
    }

    double parse_number() {
        const size_t start = pos_;
        if (pos_ < s_.size() && (s_[pos_] == '-' || s_[pos_] == '+')) ++pos_;
        while (pos_ < s_.size() &&
               ((s_[pos_] >= '0' && s_[pos_] <= '9') || s_[pos_] == '.' ||
                s_[pos_] == 'e' || s_[pos_] == 'E' || s_[pos_] == '+' ||
                s_[pos_] == '-'))
            ++pos_;
        try {
            return std::stod(s_.substr(start, pos_ - start));
        } catch (const std::exception&) {
            fail("bad number");
            return 0.0;
        }
    }

    JsonValue parse_value() {
        skip_ws();
        const char c = peek();
        if (c == '{') {
            JsonValue v; JsonObj obj;
            ++pos_;
            skip_ws();
            if (consume('}')) { v.v = std::move(obj); return v; }
            while (true) {
                skip_ws();
                std::string key = parse_string();
                skip_ws();
                expect(':');
                JsonValue item = parse_value();
                obj.emplace(std::move(key), std::move(item));
                skip_ws();
                if (consume(',')) continue;
                expect('}');
                break;
            }
            v.v = std::move(obj);
            return v;
        }
        if (c == '[') {
            JsonValue v; std::vector<JsonValue> arr;
            ++pos_;
            skip_ws();
            if (consume(']')) { v.v = std::move(arr); return v; }
            while (true) {
                arr.push_back(parse_value());
                skip_ws();
                if (consume(',')) continue;
                expect(']');
                break;
            }
            v.v = std::move(arr);
            return v;
        }
        if (c == '"') {
            JsonValue v; v.v = parse_string(); return v;
        }
        if (s_.compare(pos_, 4, "true") == 0) {
            pos_ += 4; JsonValue v; v.v = true; return v;
        }
        if (s_.compare(pos_, 5, "false") == 0) {
            pos_ += 5; JsonValue v; v.v = false; return v;
        }
        if (s_.compare(pos_, 4, "null") == 0) {
            pos_ += 4; return JsonValue{};
        }
        JsonValue v; v.v = parse_number(); return v;
    }
};

}  // namespace

JsonValue json_parse(const std::string& text) {
    Parser p(text);
    return p.parse();
}

}  // namespace heph
