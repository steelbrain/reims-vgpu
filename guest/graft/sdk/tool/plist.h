// SPDX-License-Identifier: GPL-2.0-or-later OR LGPL-3.0-or-later
// A kext's Info.plist: the XML property lists graft-kext reads, and the one it
// writes for a staged consumer. It handles the values a kext's Info.plist holds,
// strings, integers, booleans, dictionaries and arrays, and refuses a binary
// property list and any other value.
#pragma once
#include "base.h"

#include <string>
#include <utility>
#include <vector>

namespace graft::tool {

struct Plist {
    enum class Kind { String, Integer, Boolean, Dict, Array };

    Kind kind = Kind::String;
    // A string's text, or an integer's value in decimal.
    std::string text;
    bool boolean = false;
    std::vector<Plist> array;
    // A dictionary's entries. A key given twice keeps its first place and its
    // last value; the writer sorts the keys.
    std::vector<std::pair<std::string, Plist>> dict;

    static Plist makeString(std::string value);
    static Plist makeBoolean(bool value);
    static Plist makeDict();

    // The entry named key of a dictionary, or null.
    const Plist *find(const std::string &key) const;
    Plist *find(const std::string &key);
    // Set the entry named key, adding it at the end when it is new.
    void set(const std::string &key, Plist value);
    // Remove the entry named key, if there is one.
    void erase(const std::string &key);
};

// Parse an XML property list, throwing Error, prefixed with what, for one that
// is malformed or holds an unsupported value.
Plist parsePlist(const Bytes &data, const std::string &what);
// value as Python's plistlib.dumps writes it: an XML property list with the
// dictionaries' keys sorted and one tab of indentation a level.
Bytes dumpPlist(const Plist &value);

} // namespace graft::tool
