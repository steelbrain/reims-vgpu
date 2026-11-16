// SPDX-License-Identifier: GPL-2.0-or-later OR LGPL-3.0-or-later
// Test-only owned client for the arm64e route layer (src/arm64e/Route.cpp). It
// routes the three AMFI admission operations to assembly wrappers that count the
// call and tail-branch the authenticated original, so the probe needs no MAC
// argument types and proves two things on a real boot: that Graft resolves AMFI,
// builds the shadow policy and repoints the list entry ("routed"), and that the
// shadow sits in the live MAC dispatch path (every routed op is entered). It then
// routes two of the Sandbox policy's own operations the same way, naming the
// policy: file_check_mmap, an admission hook, and policy_syscall, the policy's
// mac_syscall(2) entry. This exercises a second policy's shadow in one boot,
// holds its originals apart from AMFI's, and covers policy_syscall, whose live
// slot and discriminator differ from the admission hooks'. The wrappers only
// count and tail-branch the original, so the probe never touches policy_syscall's
// user-space argument. That leg reports on its own GraftRouteProbeSandbox* keys.
// arm64e only; the x86-64 owned-client probe is tests/kexts/core-probe/CoreProbe.cpp.
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
void *graft_route_probe_originals[5]{};
alignas(8) uint64_t graft_route_probe_calls[5]{};
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
ROUTE_WRAPPER(4) // policy_syscall (Sandbox)
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
// to look for its calls. The Sandbox operations, routed together in one call,
// with originals and wrappers at indices 3 and 4.
bool sandboxRouted;
const char *const sandboxOperations[] = {"Sandbox:file_check_mmap", "Sandbox:policy_syscall"};
constexpr unsigned SandboxBase = 3;
constexpr unsigned SandboxCount = 2;

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
    // mapped, and policy_syscall on every sandbox mac_syscall, so both counters
    // are non-zero on a live boot. Each is reported on its own key, and the leg
    // passes only when every Sandbox op was entered through its shadow.
    if (sandboxRouted) {
        bool sandboxObserved = true;
        for (unsigned i = 0; i < SandboxCount; ++i) {
            uint64_t count =
                    __atomic_load_n(graft_route_probe_calls + SandboxBase + i, __ATOMIC_RELAXED);
            sandboxObserved = count != 0 && sandboxObserved;
            char key[96];
            snprintf(key, sizeof(key), "GraftRouteProbeCalls.%s", sandboxOperations[i]);
            if (root)
                root->setProperty(key, count, 64);
        }
        sandboxState(sandboxObserved ? "passed" : "refused: a Sandbox route was never entered");
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
    // Also route two of the Sandbox policy's own operations, naming the policy, in
    // a second call: file_check_mmap and policy_syscall. Their originals are held
    // apart from AMFI's, in their own slots. The leg reports on its own keys and
    // does not gate the AMFI result.
    void (*sandboxWrappers[])() = {graft_route_probe_3, graft_route_probe_4};
    graft_route_request sandbox[SandboxCount]{};
    for (unsigned i = 0; i < SandboxCount; ++i)
        sandbox[i] = {sandboxOperations[i], reinterpret_cast<void *>(sandboxWrappers[i]),
                      graft_route_probe_originals + SandboxBase + i};
    sandboxRouted = graft_route_kernel(Client, sandbox, SandboxCount);
    sandboxState(sandboxRouted ? "waiting: route-entry check" : "refused: route");
    state("waiting: route-entry check");
    thread_t thread = THREAD_NULL;
    if (kernel_thread_start(lateCheck, nullptr, &thread) != KERN_SUCCESS) {
        state("refused: worker allocation");
        return;
    }
    thread_deallocate(thread);
}
