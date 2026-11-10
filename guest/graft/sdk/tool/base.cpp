// SPDX-License-Identifier: GPL-2.0-or-later OR LGPL-3.0-or-later
#include "base.h"

#include <cerrno>
#include <cstdio>
#include <cstring>

namespace graft::tool {

uint16_t load16(const uint8_t *at) {
    return static_cast<uint16_t>(at[0] | at[1] << 8);
}

uint32_t load32(const uint8_t *at) {
    return static_cast<uint32_t>(at[0]) | static_cast<uint32_t>(at[1]) << 8 |
           static_cast<uint32_t>(at[2]) << 16 | static_cast<uint32_t>(at[3]) << 24;
}

uint64_t load64(const uint8_t *at) {
    return static_cast<uint64_t>(load32(at)) | static_cast<uint64_t>(load32(at + 4)) << 32;
}

void store16(uint8_t *at, uint16_t value) {
    at[0] = static_cast<uint8_t>(value);
    at[1] = static_cast<uint8_t>(value >> 8);
}

void store32(uint8_t *at, uint32_t value) {
    for (int i = 0; i < 4; ++i)
        at[i] = static_cast<uint8_t>(value >> (8 * i));
}

void store64(uint8_t *at, uint64_t value) {
    for (int i = 0; i < 8; ++i)
        at[i] = static_cast<uint8_t>(value >> (8 * i));
}

namespace {

// The reason the last C library call on path failed, for a refusal.
Error fileError(const char *verb, const std::string &path) {
    return Error("cannot " + std::string(verb) + " " + path + ": " + std::strerror(errno));
}

} // namespace

Bytes readFile(const std::string &path) {
    std::FILE *file = std::fopen(path.c_str(), "rb");
    if (!file)
        throw fileError("read", path);
    Bytes data;
    uint8_t buffer[65536];
    size_t count;
    while ((count = std::fread(buffer, 1, sizeof buffer, file)) > 0)
        data.insert(data.end(), buffer, buffer + count);
    bool failed = std::ferror(file);
    std::fclose(file);
    if (failed)
        throw fileError("read", path);
    return data;
}

std::string readText(const std::string &path) {
    Bytes data = readFile(path);
    return std::string(data.begin(), data.end());
}

void writeFile(const std::string &path, const Bytes &data) {
    std::FILE *file = std::fopen(path.c_str(), "wb");
    if (!file)
        throw fileError("write", path);
    bool failed = std::fwrite(data.data(), 1, data.size(), file) != data.size();
    failed = std::fclose(file) != 0 || failed;
    if (failed)
        throw fileError("write", path);
}

void writeText(const std::string &path, const std::string &text) {
    writeFile(path, Bytes(text.begin(), text.end()));
}

namespace {

// The code point of the sequence at text[at] and its length, or a length of 0
// when the sequence is not well-formed.
std::pair<uint32_t, size_t> decode(const std::string &text, size_t at) {
    auto byte = [&](size_t i) { return static_cast<uint8_t>(text[i]); };
    uint8_t lead = byte(at);
    if (lead < 0x80)
        return {lead, 1};
    size_t length = lead >= 0xf0 ? 4 : lead >= 0xe0 ? 3 : lead >= 0xc0 ? 2 : 0;
    if (!length || lead > 0xf4 || at + length > text.size())
        return {0, 0};
    uint32_t point = lead & (0x7f >> length);
    for (size_t i = 1; i < length; ++i) {
        if ((byte(at + i) & 0xc0) != 0x80)
            return {0, 0};
        point = point << 6 | (byte(at + i) & 0x3f);
    }
    static const uint32_t least[] = {0, 0, 0x80, 0x800, 0x10000};
    if (point < least[length] || point > 0x10ffff || (point >= 0xd800 && point <= 0xdfff))
        return {0, 0};
    return {point, length};
}

} // namespace

bool validUtf8(const std::string &text) {
    for (size_t at = 0; at < text.size();) {
        auto [point, length] = decode(text, at);
        (void)point;
        if (!length)
            return false;
        at += length;
    }
    return true;
}

uint32_t nextCodePoint(const std::string &text, size_t &at) {
    auto [point, length] = decode(text, at);
    at += length ? length : 1;
    return point;
}

void appendUtf8(std::string &out, uint32_t codePoint) {
    if (codePoint < 0x80) {
        out += static_cast<char>(codePoint);
    } else if (codePoint < 0x800) {
        out += static_cast<char>(0xc0 | codePoint >> 6);
        out += static_cast<char>(0x80 | (codePoint & 0x3f));
    } else if (codePoint < 0x10000) {
        out += static_cast<char>(0xe0 | codePoint >> 12);
        out += static_cast<char>(0x80 | (codePoint >> 6 & 0x3f));
        out += static_cast<char>(0x80 | (codePoint & 0x3f));
    } else {
        out += static_cast<char>(0xf0 | codePoint >> 18);
        out += static_cast<char>(0x80 | (codePoint >> 12 & 0x3f));
        out += static_cast<char>(0x80 | (codePoint >> 6 & 0x3f));
        out += static_cast<char>(0x80 | (codePoint & 0x3f));
    }
}

std::string hex(uint64_t value) {
    static const char digits[] = "0123456789abcdef";
    std::string out;
    do {
        out.insert(out.begin(), digits[value & 0xf]);
        value >>= 4;
    } while (value);
    return "0x" + out;
}

} // namespace graft::tool
