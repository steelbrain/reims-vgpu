// SPDX-License-Identifier: GPL-2.0-or-later OR LGPL-3.0-or-later
// Test-only owned client for the arm64e route layer (src/arm64e/Route.cpp). It
// routes the three AMFI admission operations to assembly wrappers that count the
// call and tail-branch the authenticated original, so the probe needs no MAC
// argument types and proves two things on a real boot: that Graft resolves AMFI,
// builds the shadow policy and repoints the list entry ("routed"), and that the
// shadow sits in the live MAC dispatch path (every routed op is entered). It then
// routes the Sandbox policy's own file_check_mmap the same way, naming the policy,
// to exercise a second policy's shadow in one boot and hold its original apart
// from AMFI's; that leg reports on its own GraftRouteProbeSandbox* keys. arm64e
// only; the x86-64 owned-client probe is tests/kexts/core-probe/CoreProbe.cpp.
#include <IOKit/IOLib.h>
#include <IOKit/IORegistryEntry.h>
#include <IOKit/IOService.h>
#include <graft/Graft.h>
#include <kern/thread.h>
#include <mach/thread_act.h>

class GraftRouteProbe final : public IOService {
    OSDeclareDefaultStructors(GraftRouteProbe)
};
OSDefineMetaClassAndStructors(GraftRouteProbe, IOService);

extern "C" {
void *graft_route_probe_originals[4]{};
alignas(8) uint64_t graft_route_probe_calls[4]{};
}

// Each wrapper increments its call counter, then tail-branches to the original
// as a plain C function pointer: braaz authenticates it under key IA with a zero
// discriminator, the schema Graft signs the handed-back original with, and keeps
// the dispatcher's full call state. It touches only x9/x10/x16, none of which
// carry arguments or must survive the call. A wrong original (mis-signed by the
// route layer) faults here under FPAC, so the branch is also a check of the
// handed-back pointer.
#define ROUTE_WRAPPER(index)                                                                       \
    extern "C" void graft_route_probe_##index();                                                   \
    __asm__(".text\n.p2align 2\n.globl _graft_route_probe_" #index "\n"                            \
            "_graft_route_probe_" #index ":\n"                                                     \
            "    adrp x16, _graft_route_probe_calls@PAGE\n"                                        \
            "    add  x16, x16, _graft_route_probe_calls@PAGEOFF\n"                                \
            "    add  x16, x16, #(8*" #index ")\n"                                                 \
            "1:  ldxr x9, [x16]\n"                                                                 \
            "    add  x9, x9, #1\n"                                                                \
            "    stxr w10, x9, [x16]\n"                                                            \
            "    cbnz w10, 1b\n"                                                                   \
            "    adrp x16, _graft_route_probe_originals@PAGE\n"                                    \
            "    add  x16, x16, _graft_route_probe_originals@PAGEOFF\n"                            \
            "    ldr  x16, [x16, #(8*" #index ")]\n"                                               \
            "    braaz x16\n");
ROUTE_WRAPPER(0) // file_check_mmap (AMFI)
ROUTE_WRAPPER(1) // file_check_library_validation (AMFI)
ROUTE_WRAPPER(2) // vnode_check_signature (AMFI)
ROUTE_WRAPPER(3) // file_check_mmap (Sandbox)
#undef ROUTE_WRAPPER

namespace {
// This probe's own bundle identifier (also GRAFT_CLIENT_ID), which Graft serves
// because it names a placed kext.
constexpr const char *Client = "ai.aneesiqbal.graft.RouteProbe";
const char *const operations[] = {"file_check_mmap", "file_check_library_validation",
                                  "vnode_check_signature"};

void state(const char *value) {
    IOLog("graft route probe: %s\n", value);
    auto root = IORegistryEntry::getRegistryRoot();
    if (root)
        root->setProperty("GraftRouteProbeState", value);
}

// The Sandbox leg's own state, kept apart from the AMFI legs so its outcome does
// not gate theirs. Set true when the Sandbox route installs, so lateCheck knows
// to look for its call.
bool sandboxRouted;

void sandboxState(const char *value) {
    IOLog("graft route probe (Sandbox): %s\n", value);
    auto root = IORegistryEntry::getRegistryRoot();
    if (root)
        root->setProperty("GraftRouteProbeSandboxState", value);
}

void lateCheck(void *, wait_result_t) {
    // After routing, the admission ops fire as the system loads code: mmap
    // almost at once, signature and library validation on each code load. Wait,
    // then record each operation's count and report whether every one was
    // entered through the shadow.
    IOSleep(20000);
    auto root = IORegistryEntry::getRegistryRoot();
    bool observed = true;
    for (unsigned i = 0; i < 3; ++i) {
        uint64_t count = __atomic_load_n(graft_route_probe_calls + i, __ATOMIC_RELAXED);
        observed = count != 0 && observed;
        char key[96];
        snprintf(key, sizeof(key), "GraftRouteProbeCalls.%s", operations[i]);
        if (root)
            root->setProperty(key, count, 64);
    }
    state(observed ? "passed" : "refused: a routed op was never entered");
    // The Sandbox leg, if it installed: file_check_mmap fires as soon as code is
    // mapped, so its counter is non-zero on a live boot. Reported on its own key.
    if (sandboxRouted) {
        uint64_t count = __atomic_load_n(graft_route_probe_calls + 3, __ATOMIC_RELAXED);
        if (root)
            root->setProperty("GraftRouteProbeCalls.Sandbox.file_check_mmap", count, 64);
        sandboxState(count ? "passed" : "refused: the Sandbox route was never entered");
    }
    thread_terminate(current_thread());
}
} // namespace

void graft_client_initialize() {
    if (!graft_client_ready(Client)) {
        state("refused: Graft unavailable");
        return;
    }
    void (*wrappers[])() = {graft_route_probe_0, graft_route_probe_1, graft_route_probe_2};
    graft_route_request requests[3]{};
    for (unsigned i = 0; i < 3; ++i)
        requests[i] = {operations[i], reinterpret_cast<void *>(wrappers[i]),
                       graft_route_probe_originals + i};
    if (!graft_route_kernel(Client, requests, 3)) {
        state("refused: routes");
        return;
    }
    // Also route the Sandbox policy's own file_check_mmap, naming the policy in a
    // second call. Its original is held apart from AMFI's, in its own slot. The
    // leg reports on its own keys and does not gate the AMFI result.
    graft_route_request sandbox{"Sandbox:file_check_mmap",
                                reinterpret_cast<void *>(graft_route_probe_3),
                                graft_route_probe_originals + 3};
    sandboxRouted = graft_route_kernel(Client, &sandbox, 1);
    sandboxState(sandboxRouted ? "waiting: route-entry check" : "refused: route");
    state("waiting: route-entry check");
    thread_t thread = THREAD_NULL;
    if (kernel_thread_start(lateCheck, nullptr, &thread) != KERN_SUCCESS) {
        state("refused: worker allocation");
        return;
    }
    thread_deallocate(thread);
}
