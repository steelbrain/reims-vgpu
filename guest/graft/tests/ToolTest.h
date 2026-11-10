// SPDX-License-Identifier: GPL-2.0-or-later OR LGPL-3.0-or-later
// What the tests of the SDK's graft-kext share: a consumer as the tool's build
// left it, a check that an operation is refused for its reason, a manifest read
// back as Graft reads it, and where a Mach-O file's headers are, for a test to
// edit them.
#pragma once
#include "Check.h"
#include "consumer.h"
#include "macho.h"
#include "plist.h"
#include <graft/ManifestFormat.h>

#include <stdio.h>
#include <string>
#include <vector>

namespace graft_test {

// A consumer that graft-kext build wrote into directory, as NAME.kext with its
// work directory beside it: the declaration it was built from, the probe's
// layouts, the versioned Info.plist, its first link (census) and its second,
// with the reservation (unbound), and the staged binary.
struct Built {
    std::string work;
    graft::tool::Declaration declaration;
    graft::tool::Layouts layouts;
    graft::tool::Plist info;
    std::string gate;
    graft::tool::Bytes census, unbound, staged;

    Built(const std::string &declaration, const std::string &directory, const std::string &name)
        : work(directory + "/graft"), declaration(graft::tool::readDeclaration(declaration)),
          layouts(graft::tool::readLayouts(work + "/layouts.json")),
          info(graft::tool::parsePlist(
                  graft::tool::readFile(work + "/Unbound.kext/Contents/Info.plist"), "Info.plist")),
          gate(graft::tool::gateKey(info)), census(graft::tool::readFile(work + "/census")),
          unbound(graft::tool::readFile(work + "/Unbound.kext/Contents/MacOS/" + name)),
          staged(graft::tool::readFile(directory + "/" + name + ".kext/Contents/MacOS/" + name)) {}
};

// Whether operation throws the tool's Error with reason in its text. Any other
// outcome is reported on stderr.
template <typename Operation>
bool refuses(Operation &&operation, const std::string &reason) {
    try {
        operation();
    } catch (const graft::tool::Error &error) {
        if (std::string(error.what()).find(reason) != std::string::npos)
            return true;
        fprintf(stderr, "refused for another reason: %s\n", error.what());
        return false;
    }
    fprintf(stderr, "not refused; expected: %s\n", reason.c_str());
    return false;
}

// The manifest in bytes, which must be one Graft accepts. The view reads bytes,
// which must outlive it.
inline graft::manifest::View manifestOf(const graft::tool::Bytes &bytes) {
    graft::manifest::View view;
    CHECK(graft::manifest::open(bytes.data(), bytes.size(), view).error ==
          graft::manifest::Error::None);
    return view;
}

// The file offsets of a Mach-O file's segment commands and section headers, in
// the order graft::tool::Image lists its segments and sections. The file must be
// one Image reads, so that its commands lie inside it.
struct Headers {
    std::vector<uint64_t> segments, sections;
};

inline Headers headers(const graft::tool::Bytes &data) {
    using namespace graft::tool;
    Headers out;
    uint64_t at = MACH_HEADER_SIZE;
    for (uint32_t i = 0; i < load32(&data[16]); ++i, at += load32(&data[at + 4])) {
        if (load32(&data[at]) != LC_SEGMENT_64)
            continue;
        out.segments.push_back(at);
        for (uint32_t section = 0; section < load32(&data[at + 64]); ++section)
            out.sections.push_back(at + SEGMENT_COMMAND_SIZE +
                                   uint64_t{section} * SECTION_HEADER_SIZE);
    }
    return out;
}

// The bytes of image's __DATA,__graft section.
inline graft::tool::Bytes graftSection(const graft::tool::Image &image) {
    const graft::tool::Section &section = image.section("__DATA", "__graft");
    return graft::tool::Bytes(image.data.begin() + section.offset,
                              image.data.begin() + section.offset + section.size);
}

} // namespace graft_test
