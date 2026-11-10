// SPDX-License-Identifier: GPL-2.0-or-later OR LGPL-3.0-or-later
// A test-only kext: prove that a manifest section survives an x86-64 kext link,
// and that src/Manifest.h compiles with kext flags. This fixture is never
// installed or booted.
#include "Manifest.h"
#include <mach/mach_types.h>

namespace {
using namespace graft::manifest;
struct Fixture {
    Header header;
    char strings[24];
};
constexpr Table Empty{sizeof(Fixture), 0};
__attribute__((section("__DATA,__graft"), used))
const Fixture manifest{{Magic,
                        Version,
                        sizeof(Fixture),
                        sizeof(Header),
                        {offsetof(Fixture, strings), sizeof(Fixture::strings)},
                        Empty,
                        Empty,
                        Empty,
                        Empty,
                        Empty,
                        Empty,
                        Empty,
                        Empty,
                        Empty,
                        Empty,
                        Empty,
                        Empty,
                        Empty,
                        1,
                        13},
                       "\0graft-bound\0"
                       "0x52471234"};
} // namespace

extern "C" kern_return_t fixture_start(kmod_info_t *, void *) {
    return KERN_SUCCESS;
}
extern "C" kern_return_t fixture_stop(kmod_info_t *, void *) {
    return KERN_SUCCESS;
}
extern "C" {
KMOD_EXPLICIT_DECL(ai.aneesiqbal.graft.test.KernelFixture, "1.0.0", fixture_start, fixture_stop)
}
