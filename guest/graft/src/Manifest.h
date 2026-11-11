// SPDX-License-Identifier: GPL-2.0-or-later OR LGPL-3.0-or-later
// The consumer manifest Graft reads. Its records, constants and validator are
// the SDK's <graft/ManifestFormat.h>, which the consumer's build tool writes
// manifests by; this header adds what only the kext needs.
#pragma once
#include <graft/ManifestFormat.h>

namespace graft {
namespace manifest {

// The class whose registered metaclass verifies provider `library`'s
// collection entry: its anchor, or else the first class the consumer declares
// from it. Null where it has neither: the provider cannot be verified.
inline const char *libraryAnchor(const View &view, uint32_t library) {
    const auto &h = view.header();
    if (!view.valid() || library >= h.libraries.count)
        return nullptr;
    if (auto anchor = view.string(view.row<Library>(h.libraries, library).anchor))
        return anchor;
    for (uint32_t i = 0; i < h.classes.count; ++i) {
        auto cls = view.row<Class>(h.classes, i);
        if (cls.library == library)
            return view.string(cls.name);
    }
    return nullptr;
}

// The version of the manifest at bytes when open() refused it for that alone: a
// sound header whose version this release does not read, as a consumer built
// with another release's SDK carries. 0 for any other failure, among them a
// version this release reads whose tables are another architecture's.
inline uint32_t unsupportedVersion(const void *bytes, Failure failure) {
    if (failure.error != Error::Version || !bytes)
        return 0;
    Header h;
    memcpy(&h, bytes, sizeof(h));
    return h.version != Version && h.version != VersionArm64e ? h.version : 0;
}

} // namespace manifest
} // namespace graft
