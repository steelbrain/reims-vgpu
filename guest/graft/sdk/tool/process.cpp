// SPDX-License-Identifier: GPL-2.0-or-later OR LGPL-3.0-or-later
#include "process.h"

#include <cstdio>
#include <cstdlib>
#include <random>

namespace graft::tool {

namespace {

// arg as one shell word: unchanged when every character is one the shell takes
// literally, otherwise single-quoted.
std::string shellQuote(const std::string &arg) {
    if (arg.empty())
        return "''";
    bool safe = true;
    for (char c : arg) {
        bool word = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') ||
                    c == '_';
        if (!word && std::string("@%+=:,./-").find(c) == std::string::npos)
            safe = false;
    }
    if (safe)
        return arg;
    std::string quoted = "'";
    for (char c : arg)
        quoted += c == '\'' ? std::string("'\"'\"'") : std::string(1, c);
    return quoted + "'";
}

} // namespace

std::string shellJoin(const std::vector<std::string> &command) {
    std::string line;
    for (const std::string &arg : command)
        line += (line.empty() ? "" : " ") + shellQuote(arg);
    return line;
}

bool run(const std::vector<std::string> &command, const std::string &output,
         const std::string &errors) {
    std::string line = shellJoin(command);
    if (!output.empty())
        line += " >" + shellQuote(output);
    if (!errors.empty())
        line += " 2>" + shellQuote(errors);
    // What graft-kext has written so far goes out before the tool writes its own.
    std::fflush(stdout);
    std::fflush(stderr);
    return std::system(line.c_str()) == 0;
}

TemporaryDirectory::TemporaryDirectory(const std::string &prefix) {
    std::error_code missing;
    std::filesystem::path parent = std::filesystem::temp_directory_path(missing);
    if (missing)
        throw Error("cannot use the temporary directory: " + missing.message());
    std::random_device random;
    for (int attempt = 0; attempt < 100; ++attempt) {
        char suffix[16];
        std::snprintf(suffix, sizeof suffix, "%08x", static_cast<unsigned>(random()));
        std::filesystem::path candidate = parent / (prefix + suffix);
        std::error_code error;
        if (std::filesystem::create_directory(candidate, error)) {
            path_ = candidate;
            return;
        }
        if (error)
            throw Error("cannot create a directory in " + parent.string() + ": " + error.message());
    }
    throw Error("cannot create a directory in " + parent.string());
}

TemporaryDirectory::~TemporaryDirectory() {
    std::error_code ignored;
    std::filesystem::remove_all(path_, ignored);
}

} // namespace graft::tool
