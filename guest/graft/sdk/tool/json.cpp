// SPDX-License-Identifier: GPL-2.0-or-later OR LGPL-3.0-or-later
#include "json.h"

#include <cstdio>
#include <limits>

namespace graft::tool {

Json Json::makeInteger(int64_t value) {
    Json json;
    json.kind = Kind::Integer;
    json.integer = value;
    return json;
}

Json Json::makeString(std::string value) {
    Json json;
    json.kind = Kind::String;
    json.string = std::move(value);
    return json;
}

Json Json::makeArray(std::vector<Json> items) {
    Json json;
    json.kind = Kind::Array;
    json.array = std::move(items);
    return json;
}

Json Json::makeObject(std::vector<std::pair<std::string, Json>> members) {
    Json json;
    json.kind = Kind::Object;
    json.object = std::move(members);
    return json;
}

const Json *Json::find(const std::string &key) const {
    for (const auto &member : object) {
        if (member.first == key)
            return &member.second;
    }
    return nullptr;
}

namespace {

class Parser {
public:
    Parser(const std::string &text, const std::string &what) : text_(text), what_(what) {}

    Json document() {
        Json value = this->value();
        space();
        if (at_ != text_.size())
            fail("extra data after the value");
        return value;
    }

private:
    const std::string &text_;
    const std::string &what_;
    size_t at_ = 0;
    size_t depth_ = 0;

    [[noreturn]] void fail(const std::string &reason) const {
        size_t line = 1, column = 1;
        for (size_t i = 0; i < at_ && i < text_.size(); ++i) {
            if (text_[i] == '\n') {
                ++line;
                column = 1;
            } else {
                ++column;
            }
        }
        throw Error(what_ + ": " + reason + " at line " + std::to_string(line) + " column " +
                    std::to_string(column));
    }

    void space() {
        while (at_ < text_.size() && (text_[at_] == ' ' || text_[at_] == '\t' ||
                                      text_[at_] == '\n' || text_[at_] == '\r'))
            ++at_;
    }

    bool literal(const char *word) {
        std::string expected(word);
        if (text_.compare(at_, expected.size(), expected) != 0)
            return false;
        at_ += expected.size();
        return true;
    }

    Json value() {
        space();
        if (at_ == text_.size())
            fail("expected a value");
        char c = text_[at_];
        Json json;
        if (c == '{' || c == '[') {
            if (++depth_ > MaxNesting)
                fail("arrays and objects nested deeper than " + std::to_string(MaxNesting) +
                     " levels");
            json = c == '{' ? object() : array();
            --depth_;
            return json;
        }
        if (c == '"')
            return Json::makeString(string());
        if (c == '-' || (c >= '0' && c <= '9'))
            return number();
        if (literal("true")) {
            json.kind = Json::Kind::Boolean;
            json.boolean = true;
        } else if (literal("false")) {
            json.kind = Json::Kind::Boolean;
        } else if (!literal("null")) {
            fail("expected a value");
        }
        return json;
    }

    Json object() {
        ++at_;
        Json json = Json::makeObject({});
        space();
        if (at_ < text_.size() && text_[at_] == '}') {
            ++at_;
            return json;
        }
        for (;;) {
            space();
            if (at_ == text_.size() || text_[at_] != '"')
                fail("expected a member name");
            std::string key = string();
            space();
            if (at_ == text_.size() || text_[at_] != ':')
                fail("expected ':'");
            ++at_;
            Json member = value();
            bool replaced = false;
            for (auto &existing : json.object) {
                if (existing.first == key) {
                    existing.second = std::move(member);
                    replaced = true;
                    break;
                }
            }
            if (!replaced)
                json.object.emplace_back(std::move(key), std::move(member));
            space();
            if (at_ < text_.size() && text_[at_] == ',') {
                ++at_;
                continue;
            }
            if (at_ < text_.size() && text_[at_] == '}') {
                ++at_;
                return json;
            }
            fail("expected ',' or '}'");
        }
    }

    Json array() {
        ++at_;
        Json json = Json::makeArray({});
        space();
        if (at_ < text_.size() && text_[at_] == ']') {
            ++at_;
            return json;
        }
        for (;;) {
            json.array.push_back(value());
            space();
            if (at_ < text_.size() && text_[at_] == ',') {
                ++at_;
                continue;
            }
            if (at_ < text_.size() && text_[at_] == ']') {
                ++at_;
                return json;
            }
            fail("expected ',' or ']'");
        }
    }

    unsigned hexDigits() {
        if (text_.size() - at_ < 4)
            fail("truncated \\u escape");
        unsigned value = 0;
        for (int i = 0; i < 4; ++i) {
            char c = text_[at_++];
            value <<= 4;
            if (c >= '0' && c <= '9')
                value |= static_cast<unsigned>(c - '0');
            else if (c >= 'a' && c <= 'f')
                value |= static_cast<unsigned>(c - 'a' + 10);
            else if (c >= 'A' && c <= 'F')
                value |= static_cast<unsigned>(c - 'A' + 10);
            else
                fail("invalid \\u escape");
        }
        return value;
    }

    std::string string() {
        ++at_;
        std::string out;
        for (;;) {
            if (at_ == text_.size())
                fail("unterminated string");
            char c = text_[at_++];
            if (c == '"')
                break;
            if (static_cast<unsigned char>(c) < 0x20)
                fail("control character in a string");
            if (c != '\\') {
                out += c;
                continue;
            }
            if (at_ == text_.size())
                fail("unterminated string");
            char escape = text_[at_++];
            switch (escape) {
            case '"': out += '"'; break;
            case '\\': out += '\\'; break;
            case '/': out += '/'; break;
            case 'b': out += '\b'; break;
            case 'f': out += '\f'; break;
            case 'n': out += '\n'; break;
            case 'r': out += '\r'; break;
            case 't': out += '\t'; break;
            case 'u': {
                unsigned point = hexDigits();
                // A high surrogate followed by an escaped low one is one code point.
                if (point >= 0xd800 && point < 0xdc00 && text_.compare(at_, 2, "\\u") == 0) {
                    size_t mark = at_;
                    at_ += 2;
                    unsigned low = hexDigits();
                    if (low >= 0xdc00 && low < 0xe000)
                        point = 0x10000 + ((point - 0xd800) << 10) + (low - 0xdc00);
                    else
                        at_ = mark;
                }
                if (point >= 0xd800 && point < 0xe000)
                    fail("unpaired surrogate in a string");
                appendUtf8(out, point);
                break;
            }
            default: fail("invalid escape");
            }
        }
        if (!validUtf8(out))
            fail("string is not UTF-8");
        return out;
    }

    Json number() {
        size_t start = at_;
        if (text_[at_] == '-')
            ++at_;
        auto digits = [&] {
            size_t first = at_;
            while (at_ < text_.size() && text_[at_] >= '0' && text_[at_] <= '9')
                ++at_;
            return at_ - first;
        };
        size_t whole = at_;
        if (!digits())
            fail("expected a value");
        if (text_[whole] == '0' && at_ - whole > 1)
            fail("leading zero in a number");
        bool real = false;
        if (at_ < text_.size() && text_[at_] == '.') {
            ++at_;
            if (!digits())
                fail("expected a digit");
            real = true;
        }
        if (at_ < text_.size() && (text_[at_] == 'e' || text_[at_] == 'E')) {
            ++at_;
            if (at_ < text_.size() && (text_[at_] == '+' || text_[at_] == '-'))
                ++at_;
            if (!digits())
                fail("expected a digit");
            real = true;
        }
        Json json;
        if (real) {
            json.kind = Json::Kind::Real;
            return json;
        }
        bool negative = text_[start] == '-';
        uint64_t magnitude = 0;
        for (size_t i = whole; i < at_; ++i) {
            uint64_t digit = static_cast<uint64_t>(text_[i] - '0');
            if (magnitude > (std::numeric_limits<uint64_t>::max() - digit) / 10)
                fail("integer out of range");
            magnitude = magnitude * 10 + digit;
        }
        uint64_t limit = static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) + negative;
        if (magnitude > limit)
            fail("integer out of range");
        json.kind = Json::Kind::Integer;
        json.integer =
                negative ? static_cast<int64_t>(0 - magnitude) : static_cast<int64_t>(magnitude);
        return json;
    }
};

// A string as Python's json module writes it with ensure_ascii: quoted, with
// every character outside space to tilde escaped.
void dumpString(std::string &out, const std::string &value) {
    out += '"';
    for (size_t at = 0; at < value.size();) {
        uint32_t point = nextCodePoint(value, at);
        switch (point) {
        case '"': out += "\\\""; continue;
        case '\\': out += "\\\\"; continue;
        case '\n': out += "\\n"; continue;
        case '\r': out += "\\r"; continue;
        case '\t': out += "\\t"; continue;
        case '\b': out += "\\b"; continue;
        case '\f': out += "\\f"; continue;
        default: break;
        }
        if (point >= 0x20 && point <= 0x7e) {
            out += static_cast<char>(point);
            continue;
        }
        auto unit = [&](uint32_t value16) {
            char buffer[8];
            std::snprintf(buffer, sizeof buffer, "\\u%04x", static_cast<unsigned>(value16));
            out += buffer;
        };
        if (point >= 0x10000) {
            unit(0xd800 + ((point - 0x10000) >> 10));
            unit(0xdc00 + ((point - 0x10000) & 0x3ff));
        } else {
            unit(point);
        }
    }
    out += '"';
}

void dump(std::string &out, const Json &value, int depth) {
    std::string inner = "\n" + std::string(static_cast<size_t>(2 * (depth + 1)), ' ');
    std::string outer = "\n" + std::string(static_cast<size_t>(2 * depth), ' ');
    switch (value.kind) {
    case Json::Kind::Null: out += "null"; break;
    case Json::Kind::Boolean: out += value.boolean ? "true" : "false"; break;
    case Json::Kind::Integer: out += std::to_string(value.integer); break;
    case Json::Kind::Real: throw Error("cannot write a real number");
    case Json::Kind::String: dumpString(out, value.string); break;
    case Json::Kind::Array:
        if (value.array.empty()) {
            out += "[]";
            break;
        }
        out += '[';
        for (size_t i = 0; i < value.array.size(); ++i) {
            out += i ? "," + inner : inner;
            dump(out, value.array[i], depth + 1);
        }
        out += outer + "]";
        break;
    case Json::Kind::Object:
        if (value.object.empty()) {
            out += "{}";
            break;
        }
        out += '{';
        for (size_t i = 0; i < value.object.size(); ++i) {
            out += i ? "," + inner : inner;
            dumpString(out, value.object[i].first);
            out += ": ";
            dump(out, value.object[i].second, depth + 1);
        }
        out += outer + "}";
        break;
    }
}

} // namespace

Json parseJson(const std::string &text, const std::string &what) {
    return Parser(text, what).document();
}

std::string dumpJson(const Json &value) {
    std::string out;
    dump(out, value, 0);
    return out;
}

} // namespace graft::tool
