// SPDX-License-Identifier: GPL-2.0-or-later OR LGPL-3.0-or-later
#include "plist.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <optional>

namespace graft::tool {

Plist Plist::makeString(std::string value) {
    Plist plist;
    plist.text = std::move(value);
    return plist;
}

Plist Plist::makeBoolean(bool value) {
    Plist plist;
    plist.kind = Kind::Boolean;
    plist.boolean = value;
    return plist;
}

Plist Plist::makeDict() {
    Plist plist;
    plist.kind = Kind::Dict;
    return plist;
}

const Plist *Plist::find(const std::string &key) const {
    for (const auto &entry : dict) {
        if (entry.first == key)
            return &entry.second;
    }
    return nullptr;
}

Plist *Plist::find(const std::string &key) {
    for (auto &entry : dict) {
        if (entry.first == key)
            return &entry.second;
    }
    return nullptr;
}

void Plist::set(const std::string &key, Plist value) {
    if (Plist *existing = find(key))
        *existing = std::move(value);
    else
        dict.emplace_back(key, std::move(value));
}

void Plist::erase(const std::string &key) {
    dict.erase(std::remove_if(dict.begin(), dict.end(),
                              [&](const auto &entry) { return entry.first == key; }),
               dict.end());
}

namespace {

bool startsWith(const std::string &text, size_t at, const char *prefix) {
    return text.compare(at, std::strlen(prefix), prefix) == 0;
}

bool isSpace(char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r';
}

// An integer element's text as plistlib reads it, in canonical decimal: decimal
// digits with an optional sign, or hexadecimal digits right after 0x, within the
// range plistlib writes, -2^63 to 2^64 - 1.
std::optional<std::string> integerText(const std::string &raw) {
    bool hexadecimal = startsWith(raw, 0, "0x") || startsWith(raw, 0, "0X");
    size_t first = hexadecimal ? 2 : 0, last = raw.size();
    while (!hexadecimal && first < last && isSpace(raw[first]))
        ++first;
    while (last > first && isSpace(raw[last - 1]))
        --last;
    bool negative = false;
    if (!hexadecimal && first < last && (raw[first] == '+' || raw[first] == '-')) {
        negative = raw[first] == '-';
        ++first;
    }
    if (first == last)
        return std::nullopt;
    uint64_t magnitude = 0, radix = hexadecimal ? 16 : 10;
    for (size_t i = first; i < last; ++i) {
        char c = raw[i];
        uint64_t digit;
        if (c >= '0' && c <= '9')
            digit = static_cast<uint64_t>(c - '0');
        else if (hexadecimal && c >= 'a' && c <= 'f')
            digit = static_cast<uint64_t>(c - 'a' + 10);
        else if (hexadecimal && c >= 'A' && c <= 'F')
            digit = static_cast<uint64_t>(c - 'A' + 10);
        else
            return std::nullopt;
        if (magnitude > (std::numeric_limits<uint64_t>::max() - digit) / radix)
            return std::nullopt;
        magnitude = magnitude * radix + digit;
    }
    if (negative && magnitude > uint64_t{1} << 63)
        return std::nullopt;
    std::string text = std::to_string(magnitude);
    return negative && magnitude ? "-" + text : text;
}

// Reads an XML property list as plistlib's parser does: each element's text is
// the character data since the last element began, and a value goes into the
// open dictionary under the last key, or the open array, or becomes the root.
class Parser {
public:
    Parser(std::string text, const std::string &what) : text_(std::move(text)), what_(what) {}

    Plist document() {
        while (at_ < text_.size()) {
            if (text_[at_] == '<')
                markup();
            else
                characters();
        }
        if (!elements_.empty() || !root_)
            fail("unterminated document");
        return *root_;
    }

private:
    std::string text_;
    const std::string &what_;
    size_t at_ = 0;
    std::vector<std::string> elements_;
    std::string data_;
    std::vector<Plist *> stack_;
    std::optional<std::string> key_;
    std::optional<Plist> root_;
    bool rootSeen_ = false;

    [[noreturn]] void fail(const std::string &reason) const {
        size_t line =
                1 +
                static_cast<size_t>(std::count(
                        text_.begin(),
                        text_.begin() + static_cast<ptrdiff_t>(std::min(at_, text_.size())), '\n'));
        throw Error(what_ + ": " + reason + " at line " + std::to_string(line));
    }

    size_t find(const char *terminator) {
        size_t end = text_.find(terminator, at_);
        if (end == std::string::npos)
            fail("unterminated markup");
        return end;
    }

    void markup() {
        if (startsWith(text_, at_, "<?")) {
            at_ = find("?>") + 2;
        } else if (startsWith(text_, at_, "<!--")) {
            at_ = find("-->") + 3;
        } else if (startsWith(text_, at_, "<![CDATA[")) {
            if (elements_.empty())
                fail("character data outside the root element");
            size_t start = at_ + 9;
            at_ = find("]]>");
            data_.append(text_, start, at_ - start);
            at_ += 3;
        } else if (startsWith(text_, at_, "<!DOCTYPE")) {
            doctype();
        } else if (startsWith(text_, at_, "</")) {
            at_ += 2;
            std::string element = name();
            while (at_ < text_.size() && isSpace(text_[at_]))
                ++at_;
            if (at_ == text_.size() || text_[at_] != '>')
                fail("malformed end tag");
            ++at_;
            if (elements_.empty() || elements_.back() != element)
                fail("mismatched end tag </" + element + ">");
            elements_.pop_back();
            end(element);
        } else {
            ++at_;
            std::string element = name();
            bool empty = attributes();
            if (elements_.empty() && rootSeen_)
                fail("a second root element");
            rootSeen_ = true;
            elements_.push_back(element);
            begin(element);
            if (empty) {
                elements_.pop_back();
                end(element);
            }
        }
    }

    // Skip a document type declaration. One with an internal subset may declare
    // entities, which a property list has no use for, so it is refused.
    void doctype() {
        char quote = 0;
        for (; at_ < text_.size(); ++at_) {
            char c = text_[at_];
            if (quote) {
                quote = c == quote ? 0 : quote;
            } else if (c == '"' || c == '\'') {
                quote = c;
            } else if (c == '[') {
                fail("a document type with declarations is not supported");
            } else if (c == '>') {
                ++at_;
                return;
            }
        }
        fail("unterminated document type");
    }

    std::string name() {
        size_t start = at_;
        while (at_ < text_.size() && !isSpace(text_[at_]) && text_[at_] != '>' && text_[at_] != '/')
            ++at_;
        if (at_ == start)
            fail("malformed tag");
        return text_.substr(start, at_ - start);
    }

    // Skip a start tag's attributes, returning whether the tag is empty (/>).
    bool attributes() {
        for (;;) {
            while (at_ < text_.size() && isSpace(text_[at_]))
                ++at_;
            if (at_ == text_.size())
                fail("unterminated tag");
            if (text_[at_] == '>') {
                ++at_;
                return false;
            }
            if (startsWith(text_, at_, "/>")) {
                at_ += 2;
                return true;
            }
            size_t equals = text_.find('=', at_);
            if (equals == std::string::npos)
                fail("malformed attribute");
            at_ = equals + 1;
            while (at_ < text_.size() && isSpace(text_[at_]))
                ++at_;
            if (at_ == text_.size() || (text_[at_] != '"' && text_[at_] != '\''))
                fail("malformed attribute");
            char quote = text_[at_++];
            size_t close = text_.find(quote, at_);
            if (close == std::string::npos)
                fail("unterminated attribute");
            at_ = close + 1;
        }
    }

    void characters() {
        size_t end = text_.find('<', at_);
        if (end == std::string::npos)
            end = text_.size();
        if (elements_.empty()) {
            for (; at_ < end; ++at_) {
                if (!isSpace(text_[at_]))
                    fail("text outside the root element");
            }
            return;
        }
        while (at_ < end) {
            if (text_[at_] != '&') {
                data_ += text_[at_++];
                continue;
            }
            size_t semicolon = text_.find(';', at_);
            if (semicolon == std::string::npos || semicolon > end)
                fail("malformed entity reference");
            std::string entity = text_.substr(at_ + 1, semicolon - at_ - 1);
            at_ = semicolon + 1;
            if (entity == "amp")
                data_ += '&';
            else if (entity == "lt")
                data_ += '<';
            else if (entity == "gt")
                data_ += '>';
            else if (entity == "quot")
                data_ += '"';
            else if (entity == "apos")
                data_ += '\'';
            else if (entity.size() > 1 && entity[0] == '#')
                appendUtf8(data_, characterReference(entity));
            else
                fail("undefined entity &" + entity + ";");
        }
    }

    uint32_t characterReference(const std::string &entity) {
        bool hexadecimal = entity[1] == 'x';
        size_t first = hexadecimal ? 2 : 1;
        if (first == entity.size())
            fail("malformed character reference");
        uint32_t point = 0;
        for (size_t i = first; i < entity.size(); ++i) {
            char c = entity[i];
            uint32_t digit;
            if (c >= '0' && c <= '9')
                digit = static_cast<uint32_t>(c - '0');
            else if (hexadecimal && c >= 'a' && c <= 'f')
                digit = static_cast<uint32_t>(c - 'a' + 10);
            else if (hexadecimal && c >= 'A' && c <= 'F')
                digit = static_cast<uint32_t>(c - 'A' + 10);
            else
                fail("malformed character reference");
            point = point * (hexadecimal ? 16 : 10) + digit;
            if (point > 0x10ffff)
                fail("malformed character reference");
        }
        if (!point || (point >= 0xd800 && point < 0xe000))
            fail("malformed character reference");
        return point;
    }

    void begin(const std::string &element) {
        data_.clear();
        if (element == "dict" || element == "array") {
            if (stack_.size() == MaxNesting)
                fail("dictionaries and arrays nested deeper than " + std::to_string(MaxNesting) +
                     " levels");
            Plist container;
            container.kind = element == "dict" ? Plist::Kind::Dict : Plist::Kind::Array;
            stack_.push_back(add(std::move(container)));
        } else if (element != "plist" && element != "key" && element != "string" &&
                   element != "integer" && element != "true" && element != "false") {
            fail("<" + element + "> values are not supported");
        }
    }

    void end(const std::string &element) {
        if (element == "dict") {
            if (key_ && !key_->empty())
                fail("missing value for key '" + *key_ + "'");
            stack_.pop_back();
        } else if (element == "array") {
            stack_.pop_back();
        } else if (element == "key") {
            if ((key_ && !key_->empty()) || stack_.empty() ||
                stack_.back()->kind != Plist::Kind::Dict)
                fail("unexpected key");
            key_ = data();
        } else if (element == "string") {
            add(Plist::makeString(data()));
        } else if (element == "true" || element == "false") {
            add(Plist::makeBoolean(element == "true"));
        } else if (element == "integer") {
            std::optional<std::string> text = integerText(data());
            if (!text)
                fail("invalid integer");
            Plist integer;
            integer.kind = Plist::Kind::Integer;
            integer.text = *text;
            add(std::move(integer));
        }
    }

    std::string data() {
        std::string text = std::move(data_);
        data_.clear();
        return text;
    }

    // Place a value as plistlib's add_object does, returning where it now lives.
    Plist *add(Plist value) {
        if (key_) {
            if (stack_.empty() || stack_.back()->kind != Plist::Kind::Dict)
                fail("unexpected element");
            Plist &dict = *stack_.back();
            dict.set(*key_, std::move(value));
            Plist *placed = dict.find(*key_);
            key_.reset();
            return placed;
        }
        if (stack_.empty()) {
            root_ = std::move(value);
            return &*root_;
        }
        if (stack_.back()->kind != Plist::Kind::Array)
            fail("unexpected element");
        stack_.back()->array.push_back(std::move(value));
        return &stack_.back()->array.back();
    }
};

// A string as plistlib escapes it, refusing the control characters XML cannot hold.
std::string escape(const std::string &text) {
    std::string out;
    for (size_t at = 0; at < text.size(); ++at) {
        char c = text[at];
        unsigned char byte = static_cast<unsigned char>(c);
        if (byte < 0x20 && c != '\t' && c != '\n' && c != '\r')
            throw Error("Info.plist strings can't contain control characters");
        if (c == '\r') {
            out += '\n';
            if (at + 1 < text.size() && text[at + 1] == '\n')
                ++at;
        } else if (c == '&') {
            out += "&amp;";
        } else if (c == '<') {
            out += "&lt;";
        } else if (c == '>') {
            out += "&gt;";
        } else {
            out += c;
        }
    }
    return out;
}

void writeLine(std::string &out, int level, const std::string &line) {
    out.append(static_cast<size_t>(level), '\t');
    out += line;
    out += '\n';
}

void write(std::string &out, const Plist &value, int level) {
    switch (value.kind) {
    case Plist::Kind::String:
        writeLine(out, level, "<string>" + escape(value.text) + "</string>");
        break;
    case Plist::Kind::Integer:
        writeLine(out, level, "<integer>" + value.text + "</integer>");
        break;
    case Plist::Kind::Boolean: writeLine(out, level, value.boolean ? "<true/>" : "<false/>"); break;
    case Plist::Kind::Dict: {
        if (value.dict.empty()) {
            writeLine(out, level, "<dict/>");
            break;
        }
        std::vector<const std::pair<std::string, Plist> *> entries;
        for (const auto &entry : value.dict)
            entries.push_back(&entry);
        std::sort(entries.begin(), entries.end(),
                  [](const auto *a, const auto *b) { return a->first < b->first; });
        writeLine(out, level, "<dict>");
        for (const auto *entry : entries) {
            writeLine(out, level + 1, "<key>" + escape(entry->first) + "</key>");
            write(out, entry->second, level + 1);
        }
        writeLine(out, level, "</dict>");
        break;
    }
    case Plist::Kind::Array:
        if (value.array.empty()) {
            writeLine(out, level, "<array/>");
            break;
        }
        writeLine(out, level, "<array>");
        for (const Plist &item : value.array)
            write(out, item, level + 1);
        writeLine(out, level, "</array>");
        break;
    }
}

} // namespace

Plist parsePlist(const Bytes &data, const std::string &what) {
    std::string text(data.begin(), data.end());
    if (startsWith(text, 0, "bplist"))
        throw Error(what + ": a binary property list is not supported; convert it with "
                           "plutil -convert xml1");
    if (startsWith(text, 0, "\xef\xbb\xbf"))
        text.erase(0, 3);
    if (!validUtf8(text))
        throw Error(what + ": property list is not UTF-8");
    // XML reads every line ending as a line feed.
    std::string normalized;
    for (size_t at = 0; at < text.size(); ++at) {
        if (text[at] != '\r') {
            normalized += text[at];
            continue;
        }
        normalized += '\n';
        if (at + 1 < text.size() && text[at + 1] == '\n')
            ++at;
    }
    return Parser(std::move(normalized), what).document();
}

Bytes dumpPlist(const Plist &value) {
    std::string out = "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
                      "<!DOCTYPE plist PUBLIC \"-//Apple//DTD PLIST 1.0//EN\" "
                      "\"http://www.apple.com/DTDs/PropertyList-1.0.dtd\">\n"
                      "<plist version=\"1.0\">\n";
    write(out, value, 0);
    out += "</plist>\n";
    return Bytes(out.begin(), out.end());
}

} // namespace graft::tool
