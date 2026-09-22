// Minimal JSON reader for the safetensors header, vocab.json and
// specials.json. Supports objects, arrays, strings (with \\uXXXX and
// standard escapes), doubles, booleans and null. The engine never
// writes JSON except the flat bench report, which is formatted by hand.

#pragma once

#include <map>
#include <string>
#include <variant>
#include <vector>

namespace heph {

struct JsonValue;

using JsonObj = std::map<std::string, JsonValue>;  // key order preserved by map

struct JsonValue {
    std::variant<std::nullptr_t, bool, double, std::string,
                 std::vector<JsonValue>, JsonObj> v;

    bool is_null() const { return std::holds_alternative<std::nullptr_t>(v); }
    bool is_bool() const { return std::holds_alternative<bool>(v); }
    bool is_number() const { return std::holds_alternative<double>(v); }
    bool is_string() const { return std::holds_alternative<std::string>(v); }
    bool is_array() const { return std::holds_alternative<std::vector<JsonValue>>(v); }
    bool is_object() const { return std::holds_alternative<JsonObj>(v); }

    double number() const { return std::get<double>(v); }
    const std::string& string() const { return std::get<std::string>(v); }
    const std::vector<JsonValue>& array() const {
        return std::get<std::vector<JsonValue>>(v);
    }
    const JsonObj& object() const { return std::get<JsonObj>(v); }
};

// Parses text; throws std::runtime_error with a position hint on error.
JsonValue json_parse(const std::string& text);

}  // namespace heph
