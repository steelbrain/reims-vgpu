// SPDX-License-Identifier: GPL-2.0-or-later OR LGPL-3.0-or-later
// Running Xcode's tools, and the scratch directory the layout probe compiles in.
// A command runs through the shell, which std::system offers on every host, with
// each argument quoted, so no argument is read as shell syntax.
#pragma once
#include "base.h"

#include <filesystem>
#include <string>
#include <vector>

namespace graft::tool {

// command as one shell line, each argument quoted only where it needs it, as
// Python's shlex.join writes it.
std::string shellJoin(const std::vector<std::string> &command);

// Run command, returning whether it exited 0. Its standard output, or its
// standard error, goes to the file named, when one is; otherwise to graft-kext's.
bool run(const std::vector<std::string> &command, const std::string &output = "",
         const std::string &errors = "");

// A directory of its own under the system's temporary directory, removed with
// everything in it when the object goes.
class TemporaryDirectory {
public:
    explicit TemporaryDirectory(const std::string &prefix);
    ~TemporaryDirectory();
    TemporaryDirectory(const TemporaryDirectory &) = delete;
    TemporaryDirectory &operator=(const TemporaryDirectory &) = delete;

    const std::filesystem::path &path() const { return path_; }

private:
    std::filesystem::path path_;
};

} // namespace graft::tool
