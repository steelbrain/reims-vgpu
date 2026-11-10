// SPDX-License-Identifier: GPL-2.0-or-later OR LGPL-3.0-or-later
// What every part of graft-kext shares: the Error a refusal throws, little-endian
// fields, whole-file reads and writes, and UTF-8 checks.
#pragma once
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace graft::tool {

using Bytes = std::vector<uint8_t>;

// An input graft-kext refuses: a file, declaration, layout, Info.plist, Mach-O
// file or collection. The command prints its message after its prefix and exits 1.
class Error : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// Little-endian fields at a byte offset the caller has bounded.
uint16_t load16(const uint8_t *at);
uint32_t load32(const uint8_t *at);
uint64_t load64(const uint8_t *at);
void store16(uint8_t *at, uint16_t value);
void store32(uint8_t *at, uint32_t value);
void store64(uint8_t *at, uint64_t value);

Bytes readFile(const std::string &path);
std::string readText(const std::string &path);
void writeFile(const std::string &path, const Bytes &data);
void writeText(const std::string &path, const std::string &text);

// Whether text is well-formed UTF-8 as Python's strict decoder takes it: no
// overlong forms, no surrogates and nothing above U+10FFFF.
bool validUtf8(const std::string &text);
// The code point of the well-formed UTF-8 sequence at text[at], advancing at past it.
uint32_t nextCodePoint(const std::string &text, size_t &at);
// code point as UTF-8, appended to out.
void appendUtf8(std::string &out, uint32_t codePoint);

// "0x" and value in lowercase hexadecimal, as Python formats {value:#x}.
std::string hex(uint64_t value);

// The deepest the JSON and property-list parsers nest arrays, objects and
// dictionaries. A declaration, layouts.json or Info.plist nests a few levels;
// the limit keeps the parsers and every walk of what they read well inside the
// stack, so a deeper document is refused rather than crashing graft-kext.
constexpr size_t MaxNesting = 64;

} // namespace graft::tool
