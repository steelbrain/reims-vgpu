// SPDX-License-Identifier: GPL-2.0-or-later OR LGPL-3.0-or-later
// The consumer fixture the SDK tool's tests build and stage. It imports from its
// provider a pointer (parent_global), direct calls (the destructor's calls to
// Parent's), a qualified virtual call (Parent::inherited, which -fapple-kext
// makes through Parent's vtable), and two weak pointers: weak_present, which a
// provider would define, and weak_absent, which none would. fixture_marker
// carries the name of the metaclass pointer staging renames.
#include "Fixture.h"
#include <mach/mach_types.h>
Consumer::~Consumer() {}
int Consumer::changed(int n) {
    return Parent::inherited(n) + 3;
}
int Consumer::appended(int n) {
    return n + 7;
}
extern "C" int fixture_marker __asm__("__ZN8Consumer10superClassE");
int fixture_marker = 7;
extern int parent_global;
int *fixture_global = &parent_global;
extern int weak_present __attribute__((weak_import));
extern int weak_absent __attribute__((weak_import));
int *fixture_weak[] = {&weak_present, &weak_absent};
extern "C" kern_return_t fixture_start(kmod_info_t *, void *) {
    return KERN_SUCCESS;
}
extern "C" kern_return_t fixture_stop(kmod_info_t *, void *) {
    return KERN_SUCCESS;
}
extern "C" {
KMOD_EXPLICIT_DECL(ai.aneesiqbal.graft.test.ConsumerFixture, "1.0.0", fixture_start, fixture_stop)
}
