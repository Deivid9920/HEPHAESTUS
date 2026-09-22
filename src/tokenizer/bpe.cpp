// Byte-level BPE runtime (reader, not trainer).
//
// Pipeline (mirrors HF's ByteLevel + BPE exactly; parity is enforced by
// tests/golden/test_tokenize_parity.py):
//   1. GPT-2 byte -> unicode alphabet map BEFORE vocab lookup
//      ('Ġ' = space: byte 0x20 -> U+0120);
//   2. the ByteLevel pre-split BEFORE merges — a manual scanner with a
//      Unicode category table (std::regex cannot do \p{L}/\p{N} and
//      third-party regex libraries are out of scope);
//   3. merges by rank: find the pair with the LOWEST merges.txt rank in
//      the word, merge ALL of its occurrences in one left-to-right
//      pass, repeat (HF semantics — never "first pair found");
//   4. encode never injects special tokens; unk can never fire because
//      byte-level coverage is total (all 256 mapped bytes are tokens).
//
// All merge work happens on codepoint sequences (mapped characters like
// 'Ġ' occupy TWO UTF-8 bytes; treating char as a byte would corrupt the
// merge bookkeeping).

#include "tokenizer/bpe.h"

#include <algorithm>
#include <fstream>
#include <stdexcept>

#include "core/json.h"

namespace heph {

namespace {

// ---- UTF-8 helpers --------------------------------------------------------

void utf8_append(std::string& out, uint32_t cp) {
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

// Decodes one codepoint at *pos; advances pos. Invalid bytes decode as
// Latin-1 (never happens for valid UTF-8 input).
uint32_t utf8_next(const std::string& s, size_t& pos) {
    const unsigned char c = static_cast<unsigned char>(s[pos++]);
    if (c < 0x80) return c;
    if ((c & 0xE0) == 0xC0 && pos < s.size()) {
        const uint32_t cp = ((c & 0x1F) << 6) | (s[pos] & 0x3F);
        pos += 1;
        return cp;
    }
    if ((c & 0xF0) == 0xE0 && pos + 1 < s.size()) {
        const uint32_t cp = ((c & 0x0F) << 12) | ((s[pos] & 0x3F) << 6) |
                            (s[pos + 1] & 0x3F);
        pos += 2;
        return cp;
    }
    if ((c & 0xF8) == 0xF0 && pos + 2 < s.size()) {
        const uint32_t cp = ((c & 0x07) << 18) | ((s[pos] & 0x3F) << 12) |
                            ((s[pos + 1] & 0x3F) << 6) | (s[pos + 2] & 0x3F);
        pos += 3;
        return cp;
    }
    return c;  // Latin-1 fallback for stray bytes
}

// ---- Unicode category tables (compact, standard ranges) -------------------
//
// \p{L} letters and \p{N} numbers as used by the GPT-2 pre-tokenizer
// regex. The table covers the standard letter/number blocks a text
// corpus produces (Latin, Greek, Cyrillic, Hebrew, Arabic, Indic, Thai,
// CJK, Kana, Hangul, ligatures, mathematical alphanumerics); anything
// else falls into the "other" class, which is what the regex does for
// symbols like '=' or '≈'.

bool is_letter(uint32_t cp) {
    if (cp < 128) return (cp >= 'A' && cp <= 'Z') || (cp >= 'a' && cp <= 'z');
    if (cp < 0x250) {  // Latin-1 + Latin Extended-A + IPA extensions
        return (cp >= 0xAA && cp <= 0xAA) || (cp == 0xB5) || (cp == 0xBA) ||
               (cp >= 0xC0 && cp <= 0xD6) || (cp >= 0xD8 && cp <= 0xF6) ||
               (cp >= 0xF8 && cp <= 0x24F);
    }
    if (cp >= 0x250 && cp < 0x2B0) return true;             // IPA ext
    if (cp >= 0x2C6 && cp <= 0x2D1) return true;            // modifier
    if (cp >= 0x2E0 && cp <= 0x2E4) return true;
    if (cp >= 0x370 && cp <= 0x3FF) {                        // Greek
        return !(cp == 0x378 || cp == 0x379 || cp == 0x380 || cp == 0x381 ||
                 cp == 0x382 || cp == 0x383 || cp == 0x387 || cp == 0x38B ||
                 cp == 0x38D || cp == 0x3A2 || cp == 0x3CF || cp == 0x3D8 ||
                 cp == 0x3D9);
    }
    if (cp >= 0x400 && cp <= 0x52F) return true;             // Cyrillic
    if (cp >= 0x531 && cp <= 0x556) return true;             // Armenian
    if (cp >= 0x561 && cp <= 0x587) return true;
    if (cp >= 0x5D0 && cp <= 0x5EA) return true;             // Hebrew
    if (cp >= 0x5F0 && cp <= 0x5F2) return true;
    if (cp >= 0x620 && cp <= 0x64A) return true;             // Arabic
    if (cp >= 0x66E && cp <= 0x66F) return true;
    if (cp >= 0x671 && cp <= 0x6D3) return true;
    if (cp == 0x6D5) return true;
    if (cp >= 0x710 && cp <= 0x72F) return true;             // Syriac
    if (cp >= 0x74D && cp <= 0x7A5) return true;
    if (cp >= 0x904 && cp <= 0x939) return true;             // Devanagari
    if (cp >= 0x958 && cp <= 0x961) return true;
    {
        // remaining Indic letter ranges (Bengali..Malayalam, Gujarati,
        // Oriya, Tamil, Telugu, Kannada, Malayalam, Sinhala)
        if (cp >= 0x985 && cp <= 0x98C) return true;
        if (cp >= 0x98F && cp <= 0x990) return true;
        if (cp >= 0x993 && cp <= 0x9A8) return true;
        if (cp >= 0x9AA && cp <= 0x9B0) return true;
        if (cp == 0x9B2) return true;
        if (cp >= 0x9B6 && cp <= 0x9B9) return true;
        if (cp >= 0x9DC && cp <= 0x9DD) return true;
        if (cp >= 0x9DF && cp <= 0x9E1) return true;
        if (cp >= 0xA05 && cp <= 0xA0A) return true;
        if (cp >= 0xA0F && cp <= 0xA10) return true;
        if (cp >= 0xA13 && cp <= 0xA28) return true;
        if (cp >= 0xA2A && cp <= 0xA30) return true;
        if (cp >= 0xA32 && cp <= 0xA33) return true;
        if (cp >= 0xA35 && cp <= 0xA36) return true;
        if (cp >= 0xA38 && cp <= 0xA39) return true;
        if (cp >= 0xA85 && cp <= 0xA8B) return true;
        if (cp >= 0xA8F && cp <= 0xA91) return true;
        if (cp >= 0xA93 && cp <= 0xAA8) return true;
        if (cp >= 0xAAA && cp <= 0xAB0) return true;
        if (cp >= 0xAB2 && cp <= 0xAB3) return true;
        if (cp >= 0xAB5 && cp <= 0xAB9) return true;
        if (cp >= 0xB05 && cp <= 0xB0C) return true;
        if (cp >= 0xB0F && cp <= 0xB10) return true;
        if (cp >= 0xB13 && cp <= 0xB28) return true;
        if (cp >= 0xB2A && cp <= 0xB39) return true;
        if (cp >= 0xB85 && cp <= 0xBB9) return true;
        if (cp >= 0xC05 && cp <= 0xC0C) return true;
        if (cp >= 0xC0E && cp <= 0xC10) return true;
        if (cp >= 0xC12 && cp <= 0xC28) return true;
        if (cp >= 0xC2A && cp <= 0xC39) return true;
        if (cp >= 0xC85 && cp <= 0xC8C) return true;
        if (cp >= 0xC8E && cp <= 0xC90) return true;
        if (cp >= 0xC92 && cp <= 0xCA8) return true;
        if (cp >= 0xCAA && cp <= 0xCB3) return true;
        if (cp >= 0xCB5 && cp <= 0xCB9) return true;
        if (cp >= 0xD05 && cp <= 0xD0C) return true;
        if (cp >= 0xD0E && cp <= 0xD10) return true;
        if (cp >= 0xD12 && cp <= 0xD28) return true;
        if (cp >= 0xD2A && cp <= 0xD39) return true;
        if (cp >= 0xD85 && cp <= 0xDBB) return true;  // Sinhala
        if (cp == 0xDBD) return true;
        if (cp >= 0xDC0 && cp <= 0xDC6) return true;
    }
    if (cp >= 0xE01 && cp <= 0xE30) return true;             // Thai
    if (cp == 0xE32 || cp == 0xE33) return true;
    if (cp >= 0xE40 && cp <= 0xE46) return true;
    if (cp >= 0x1E00 && cp <= 0x1FFF) return true;           // Latin ext add.
    if (cp >= 0x2160 && cp <= 0x2182) return true;           // Roman numerals
    if (cp >= 0x2C60 && cp <= 0x2CE4) return true;           // Latin ext-C
    if (cp >= 0x2D00 && cp <= 0x2D25) return true;           // Georgian sup.
    if (cp >= 0x3005 && cp <= 0x3007) return true;           // CJK marks
    if (cp >= 0x3021 && cp <= 0x3029) return true;
    if (cp >= 0x3031 && cp <= 0x3035) return true;
    if (cp >= 0x3041 && cp <= 0x3096) return true;           // Hiragana
    if (cp >= 0x309D && cp <= 0x309F) return true;
    if (cp >= 0x30A1 && cp <= 0x30FA) return true;           // Katakana
    if (cp >= 0x30FC && cp <= 0x30FF) return true;
    if (cp >= 0x3105 && cp <= 0x312D) return true;           // Bopomofo
    if (cp >= 0x3131 && cp <= 0x318E) return true;           // Hangul comp.
    if (cp >= 0x31A0 && cp <= 0x31BA) return true;
    if (cp >= 0x31F0 && cp <= 0x31FF) return true;           // Katakana ext
    if (cp >= 0x3400 && cp <= 0x4DBF) return true;           // CJK ext-A
    if (cp >= 0x4E00 && cp <= 0x9FFF) return true;           // CJK unified
    if (cp >= 0xA000 && cp <= 0xA48C) return true;           // Yi
    if (cp >= 0xAC00 && cp <= 0xD7A3) return true;           // Hangul
    if (cp >= 0xF900 && cp <= 0xFA6D) return true;           // CJK compat
    if (cp >= 0xFA70 && cp <= 0xFAD9) return true;
    if (cp >= 0xFB00 && cp <= 0xFB06) return true;           // ligatures
    if (cp >= 0xFB13 && cp <= 0xFB17) return true;
    if (cp >= 0xFB1D && cp <= 0xFB28) return true;
    if (cp >= 0xFB2A && cp <= 0xFBB1) return true;
    if (cp >= 0xFBD3 && cp <= 0xFD3D) return true;
    if (cp >= 0xFD50 && cp <= 0xFDC7) return true;
    if (cp >= 0xFDF0 && cp <= 0xFDFB) return true;
    if (cp >= 0xFE70 && cp <= 0xFEFC) return true;           // Arabic pres.
    if (cp >= 0xFF21 && cp <= 0xFF3A) return true;           // fullwidth
    if (cp >= 0xFF41 && cp <= 0xFF5A) return true;
    if (cp >= 0xFF66 && cp <= 0xFFDC) return true;
    if (cp >= 0x10000 && cp <= 0x1000B) return true;         // Linear B
    if (cp >= 0x10140 && cp <= 0x10174) return true;         // Greek acroph.
    if (cp >= 0x10300 && cp <= 0x1031E) return true;         // Old Italic
    if (cp >= 0x10330 && cp <= 0x1034A) return true;         // Gothic
    if (cp >= 0x10380 && cp <= 0x1039D) return true;         // Ugaritic
    if (cp >= 0x10400 && cp <= 0x1049D) return true;         // Deseret..
    if (cp >= 0x10800 && cp <= 0x10838) return true;
    if (cp >= 0x1D400 && cp <= 0x1D454) return true;         // math alphanum
    if (cp >= 0x1D456 && cp <= 0x1D49C) return true;
    if (cp >= 0x1D49E && cp <= 0x1D49F) return true;
    if (cp == 0x1D4A2) return true;
    if (cp >= 0x1D4A5 && cp <= 0x1D4A6) return true;
    if (cp >= 0x1D4A9 && cp <= 0x1D4AC) return true;
    if (cp >= 0x1D4AE && cp <= 0x1D4B9) return true;
    if (cp == 0x1D4BB) return true;
    if (cp >= 0x1D4BD && cp <= 0x1D4C3) return true;
    if (cp >= 0x1D4C5 && cp <= 0x1D505) return true;
    if (cp >= 0x1D507 && cp <= 0x1D50A) return true;
    if (cp >= 0x1D50D && cp <= 0x1D514) return true;
    if (cp >= 0x1D516 && cp <= 0x1D51C) return true;
    if (cp >= 0x1D51E && cp <= 0x1D539) return true;
    if (cp >= 0x1D53B && cp <= 0x1D53E) return true;
    if (cp >= 0x1D540 && cp <= 0x1D544) return true;
    if (cp == 0x1D546) return true;
    if (cp >= 0x1D54A && cp <= 0x1D550) return true;
    if (cp >= 0x1D552 && cp <= 0x1D6A5) return true;         // incl. 𝕌𝕟𝕚𝕔𝕠𝕕𝕖
    if (cp >= 0x1D6A8 && cp <= 0x1D7C2) return true;         // Greek italic
    if (cp >= 0x1D800 && cp <= 0x1D9FF) return false;
    return false;
}

bool is_number(uint32_t cp) {
    if (cp < 128) return cp >= '0' && cp <= '9';
    if (cp >= 0x660 && cp <= 0x669) return true;             // Arabic-Indic
    if (cp >= 0x6F0 && cp <= 0x6F9) return true;
    if (cp >= 0x7C0 && cp <= 0x7C9) return true;
    if (cp >= 0x966 && cp <= 0x96F) return true;             // Devanagari
    if (cp >= 0x9E6 && cp <= 0x9EF) return true;
    if (cp >= 0xA66 && cp <= 0xA6F) return true;
    if (cp >= 0xAE6 && cp <= 0xAEF) return true;
    if (cp >= 0xB66 && cp <= 0xB6F) return true;
    if (cp >= 0xBE6 && cp <= 0xBEF) return true;
    if (cp >= 0xC66 && cp <= 0xC6F) return true;
    if (cp >= 0xCE6 && cp <= 0xCEF) return true;
    if (cp >= 0xD66 && cp <= 0xD6F) return true;
    if (cp >= 0xE50 && cp <= 0xE59) return true;             // Thai
    if (cp >= 0xED0 && cp <= 0xED9) return true;
    if (cp >= 0xF20 && cp <= 0xF29) return true;
    if (cp >= 0x1040 && cp <= 0x1049) return true;
    if (cp >= 0x1090 && cp <= 0x1099) return true;
    if (cp >= 0x17E0 && cp <= 0x17E9) return true;
    if (cp >= 0x1810 && cp <= 0x1819) return true;
    if (cp >= 0x1B50 && cp <= 0x1B59) return true;
    if (cp >= 0x2070 && cp <= 0x2079) return true;           // superscripts
    if (cp >= 0x2080 && cp <= 0x2089) return true;           // subscripts
    if (cp >= 0x2460 && cp <= 0x249B) return true;           // circled
    if (cp >= 0xFF10 && cp <= 0xFF19) return true;           // fullwidth
    if (cp >= 0x1D7CE && cp <= 0x1D7FF) return true;         // math digits
    return false;
}

bool is_space(uint32_t cp) {
    switch (cp) {
        case 0x09: case 0x0A: case 0x0B: case 0x0C: case 0x0D:
        case 0x20: case 0x85: case 0xA0: case 0x1680:
        case 0x2028: case 0x2029: case 0x202F: case 0x205F: case 0x3000:
            return true;
        default: return cp >= 0x2000 && cp <= 0x200A;
    }
}

// ---- GPT-2 byte -> unicode alphabet ---------------------------------------

// Printable byte ranges map to themselves; the rest are shifted to
// 256+n in the exact GPT-2 order (HF tokenizers byte_level.alphabet).
void build_byte_map(uint32_t byte_to_cp[256]) {
    bool printable[256] = {false};
    for (int b = 33; b <= 126; ++b) printable[b] = true;
    for (int b = 161; b <= 172; ++b) printable[b] = true;
    for (int b = 174; b <= 255; ++b) printable[b] = true;
    int n = 0;
    for (int b = 0; b < 256; ++b) {
        if (printable[b]) byte_to_cp[b] = static_cast<uint32_t>(b);
        else byte_to_cp[b] = static_cast<uint32_t>(256 + n++);
    }
}

bool contraction_at(const std::vector<uint32_t>& cps, size_t i, size_t n,
                    size_t* len) {
    if (i >= n || cps[i] != '\'') return false;
    if (i + 1 < n) {
        const uint32_t c = cps[i + 1];
        if (c == 's' || c == 't' || c == 'm' || c == 'd') {
            *len = 2;
            return true;
        }
        if (i + 2 < n) {
            const uint32_t c2 = cps[i + 2];
            if ((c == 'r' && c2 == 'e') || (c == 'v' && c2 == 'e') ||
                (c == 'l' && c2 == 'l')) {
                *len = 3;
                return true;
            }
        }
    }
    return false;
}

// One alternative of the GPT-2 regex: returns the piece length in
// codepoints, or 0 when it does not match at position i.
size_t match_piece(const std::vector<uint32_t>& cps, size_t i, size_t n) {
    size_t clen = 0;
    if (contraction_at(cps, i, n, &clen)) return clen;
    size_t j = i;
    bool had_space = false;
    if (cps[j] == ' ' && j + 1 < n) {  // ' ?' prefix
        had_space = true;
        ++j;
    }
    const uint32_t c = cps[j];
    if (is_letter(c)) {
        while (j < n && is_letter(cps[j])) ++j;
        return j - i;
    }
    if (is_number(c)) {
        while (j < n && is_number(cps[j])) ++j;
        return j - i;
    }
    if (!is_space(c)) {  // [^\s\p{L}\p{N}]+
        while (j < n && !is_space(cps[j]) && !is_letter(cps[j]) &&
               !is_number(cps[j]))
            ++j;
        return j - i;
    }
    if (had_space) {
        // space followed by whitespace: ' ?' matched nothing useful;
        // fall through to whitespace rules starting at the space.
        j = i;
    }
    // \s+(?!\S): whitespace run whose end is not followed by non-space;
    // greedily this consumes all but the LAST space of a run followed
    // by a word (the last space goes to the next ' ?' alternative).
    if (is_space(cps[j])) {
        size_t k = j;
        while (k < n && is_space(cps[k])) ++k;
        if (k == n) return k - i;              // run at end of text
        if (k - j >= 2) return k - i - 1;      // leave last space
        return 0;                              // single space before word:
                                               // handled by ' ?' of the
                                               // next alternatives
    }
    return 0;
}

}  // namespace

void BpeTokenizer::load(const std::string& dir) {
    const std::string vocab_text = [&] {
        std::ifstream fh(dir + "/vocab.json", std::ios::binary);
        if (!fh) throw std::runtime_error("cannot open vocab.json in " + dir);
        std::ostringstream ss;
        ss << fh.rdbuf();
        return ss.str();
    }();
    const JsonValue jv = json_parse(vocab_text);
    if (!jv.is_object()) throw std::runtime_error("vocab.json is not an object");
    for (const auto& [piece, id] : jv.object()) {
        vocab_[piece] = static_cast<int>(id.number());
        id_to_piece_.push_back(piece);
    }

    std::ifstream mh(dir + "/merges.txt");
    if (!mh) throw std::runtime_error("cannot open merges.txt in " + dir);
    std::string line;
    int rank = 0;
    bool first = true;
    while (std::getline(mh, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (first) {  // #version header
            first = false;
            if (!line.empty() && line[0] == '#') continue;
        }
        if (line.empty() || line[0] == '#') continue;
        const size_t sp = line.find(' ');
        if (sp == std::string::npos)
            throw std::runtime_error("merges.txt: bad line: " + line);
        const std::string left = line.substr(0, sp);
        const std::string right = line.substr(sp + 1);
        merge_rank_[left + '\x01' + right] = rank++;
    }

    std::ifstream sh(dir + "/specials.json");
    if (!sh) throw std::runtime_error("cannot open specials.json in " + dir);
    std::ostringstream ss;
    ss << sh.rdbuf();
    const JsonValue sp = json_parse(ss.str());
    if (!sp.is_object()) throw std::runtime_error("specials.json malformed");
    auto get_special = [&](const char* name) {
        const auto it = sp.object().find(name);
        if (it == sp.object().end())
            throw std::runtime_error(std::string("specials.json missing ") + name);
        return static_cast<int>(it->second.number());
    };
    bos_ = get_special("bos");
    eos_ = get_special("eos");
    unk_ = get_special("unk");

    // The 256 mapped base tokens must all exist (byte-level coverage).
    uint32_t byte_to_cp[256];
    build_byte_map(byte_to_cp);
    byte_token_.resize(256, -1);
    for (int b = 0; b < 256; ++b) {
        std::string piece;
        utf8_append(piece, byte_to_cp[b]);
        const auto it = vocab_.find(piece);
        if (it == vocab_.end())
            throw std::runtime_error("byte-level coverage broken: byte " +
                                     std::to_string(b) + " not in vocab");
        byte_token_[b] = it->second;
    }
    if (static_cast<int>(id_to_piece_.size()) != vocab_size())
        throw std::runtime_error("vocab ids are not contiguous");
}

std::vector<int> BpeTokenizer::encode(const std::string& text) const {
    if (byte_token_.empty()) throw std::runtime_error("tokenizer not loaded");
    // Byte-level alphabet mapping BEFORE any vocab lookup: every input
    // byte becomes one mapped codepoint.
    uint32_t byte_to_cp[256];
    build_byte_map(byte_to_cp);

    // HF ByteLevel(add_prefix_space=True) prepends one space when the
    // input does not start with one (an empty string becomes " ").
    std::string prepared = text;
    if (prepared.empty() || prepared[0] != ' ') prepared = " " + prepared;

    std::vector<uint32_t> cps;
    cps.reserve(prepared.size());
    for (size_t i = 0; i < prepared.size();) cps.push_back(utf8_next(prepared, i));

    std::vector<int> out;
    size_t i = 0;
    std::vector<std::string> syms;
    std::vector<int> sym_ids;
    while (i < cps.size()) {
        size_t len = 1;
        if (i < cps.size()) {
            const size_t m = match_piece(cps, i, cps.size());
            len = m > 0 ? m : 1;
        }
        // bytes of the original piece -> mapped codepoints -> symbols
        syms.clear();
        sym_ids.clear();
        std::string piece_bytes;
        for (size_t k = i; k < i + len; ++k) utf8_append(piece_bytes, cps[k]);
        for (unsigned char b : piece_bytes) {
            std::string sym;
            utf8_append(sym, byte_to_cp[b]);
            syms.push_back(sym);
        }

        // BPE: repeatedly merge ALL occurrences of the lowest-ranked pair
        // (HF semantics), until no adjacent pair has a rank.
        while (syms.size() >= 2) {
            int best = -1;
            for (size_t k = 0; k + 1 < syms.size(); ++k) {
                const auto it = merge_rank_.find(syms[k] + '\x01' + syms[k + 1]);
                if (it != merge_rank_.end() &&
                    (best < 0 || it->second < best))
                    best = it->second;
            }
            if (best < 0) break;
            // one left-to-right pass merging every occurrence of the
            // pair that carries `best`
            std::vector<std::string> merged;
            for (size_t k = 0; k < syms.size();) {
                if (k + 1 < syms.size()) {
                    const auto it = merge_rank_.find(syms[k] + '\x01' + syms[k + 1]);
                    if (it != merge_rank_.end() && it->second == best) {
                        merged.push_back(syms[k] + syms[k + 1]);
                        k += 2;
                        continue;
                    }
                }
                merged.push_back(syms[k]);
                ++k;
            }
            syms.swap(merged);
        }

        for (const auto& sym : syms) {
            const auto it = vocab_.find(sym);
            if (it == vocab_.end())
                throw std::runtime_error(
                    "bpe: piece not in vocab (byte map broken): " + sym);
            out.push_back(it->second);
        }
        i += len;
    }
    return out;
}

std::string BpeTokenizer::decode(const std::vector<int>& ids) const {
    // inverse byte map: codepoint -> byte
    std::string bytes;
    bytes.reserve(ids.size() * 2);
    for (const int id : ids) {
        if (id < 0 || id >= static_cast<int>(id_to_piece_.size())) continue;
        const std::string& piece = id_to_piece_[id];
        for (size_t i = 0; i < piece.size();) {
            const uint32_t cp = utf8_next(piece, i);
            if (cp < 256) {
                bytes += static_cast<char>(cp);
            } else if (cp >= 256 && cp < 512) {
                // invert 256+n: order defined by build_byte_map
                uint32_t byte_to_cp[256];
                build_byte_map(byte_to_cp);
                bool found = false;
                for (int b = 0; b < 256; ++b) {
                    if (byte_to_cp[b] == cp) {
                        bytes += static_cast<char>(b);
                        found = true;
                        break;
                    }
                }
                if (!found) bytes += '?';
            } else {
                bytes += '?';
            }
        }
    }
    return bytes;
}

int BpeTokenizer::bos_id() const { return bos_; }
int BpeTokenizer::eos_id() const { return eos_; }
int BpeTokenizer::unk_id() const { return unk_; }
int BpeTokenizer::vocab_size() const {
    return static_cast<int>(vocab_.size());
}

}  // namespace heph
