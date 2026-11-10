// SPDX-License-Identifier: GPL-2.0-or-later OR LGPL-3.0-or-later
// The JSON graft-kext reads, a consumer's graft.json and the probe's
// layouts.json, and the JSON it writes, layouts.json.
#pragma once
#include "base.h"

#include <string>
#include <utility>
#include <vector>

namespace graft::tool {

struct Json {
    // A number with a fraction or an exponent is Real; the tool reads no real
    // value, so it keeps none, only that the number was not an integer.
    enum class Kind { Null, Boolean, Integer, Real, String, Array, Object };

    Kind kind = Kind::Null;
    bool boolean = false;
    int64_t integer = 0;
    std::string string;
    std::vector<Json> array;
    // An object's members in their order. A key given twice keeps its first
    // place and its last value, as Python's json module reads it.
    std::vector<std::pair<std::string, Json>> object;

    static Json makeInteger(int64_t value);
    static Json makeString(std::string value);
    static Json makeArray(std::vector<Json> items);
    static Json makeObject(std::vector<std::pair<std::string, Json>> members);

    // The member named key of an object, or null.
    const Json *find(const std::string &key) const;
};

// Parse JSON text, throwing Error, prefixed with what, where it is malformed.
Json parseJson(const std::string &text, const std::string &what);
// value as Python's json.dumps(value, indent=2) writes it: ASCII, each member and
// item on its own line, indented two spaces a level.
std::string dumpJson(const Json &value);

} // namespace graft::tool
