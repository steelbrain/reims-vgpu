// SPDX-License-Identifier: GPL-2.0-or-later OR LGPL-3.0-or-later
// Test-only owned client. Its assembly wrappers preserve the dispatchers'
// complete call state, so this probe needs no guessed MAC argument types.
#include <IOKit/IOLib.h>
#include <IOKit/IORegistryEntry.h>
#include <IOKit/IOService.h>
#include <graft/Graft.h>
#include <kern/thread.h>
#include <mach/thread_act.h>

class GraftCoreProbe final : public IOService {
    OSDeclareDefaultStructors(GraftCoreProbe)
};
OSDefineMetaClassAndStructors(GraftCoreProbe, IOService);

extern "C" {
void *graft_probe_originals[6]{};
alignas(8) uint64_t graft_probe_calls[6]{};
}

// Save flags before counting and restore them before tail-calling the
// trampoline. No argument, register, stack argument or return value changes.
#define PROBE_ENTRY(index, offset)                                                                 \
    extern "C" void graft_probe_##index();                                                         \
    __asm__(".text\n.p2align 4\n.globl _graft_probe_" #index "\n"                                  \
            "_graft_probe_" #index ":\n"                                                           \
            "pushfq\nlock incq _graft_probe_calls+" #offset "(%rip)\npopfq\n"                      \
            "jmpq *_graft_probe_originals+" #offset "(%rip)\n");
PROBE_ENTRY(0, 0)
PROBE_ENTRY(1, 8)
PROBE_ENTRY(2, 16)
PROBE_ENTRY(3, 24)
PROBE_ENTRY(4, 32)
PROBE_ENTRY(5, 40)
#undef PROBE_ENTRY

namespace {
constexpr const char *Client = "ai.aneesiqbal.RgfxGate";
const char *const symbols[] = {"_mac_file_check_library_validation",
                               "_mac_file_check_mmap",
                               "_mac_vnode_check_signature",
                               "_mac_vnode_check_create",
                               "_mac_vnode_check_open",
                               "_mac_vnode_check_truncate"};

void state(const char *value) {
    IOLog("graft probe: %s\n", value);
    auto root = IORegistryEntry::getRegistryRoot();
    if (root)
        root->setProperty("GraftProbeState", value);
}

void lateCheck(void *, wait_result_t) {
    bool bsd = false;
    for (unsigned i = 0; i < 6000; ++i) {
        auto resources = IOService::getResourceService();
        if (resources && resources->getProperty("IOBSD")) {
            bsd = true;
            break;
        }
        IOSleep(10);
    }
    if (bsd) {
        // Exercise lazy lookups after userland has begun, independently
        // checking the calibrated address against a normal Boot-KC import.
        IOSleep(20000);
        uint64_t address = 0;
        bool valid = graft_kernel_symbol(Client, "_IOLog", &address) &&
                     address == reinterpret_cast<uint64_t>(&IOLog);
        for (const char *name : symbols)
            valid = graft_kernel_symbol(Client, name, &address) && valid;
        auto root = IORegistryEntry::getRegistryRoot();
        bool observed = false;
        for (unsigned attempt = 0; attempt < 120; ++attempt) {
            observed = true;
            for (unsigned i = 0; i < 6; ++i) {
                uint64_t count = __atomic_load_n(graft_probe_calls + i, __ATOMIC_RELAXED);
                observed = count != 0 && observed;
                char key[128];
                snprintf(key, sizeof(key), "GraftProbeCalls.%s", symbols[i]);
                if (root)
                    root->setProperty(key, count, 64);
            }
            if (observed || !valid)
                break;
            IOSleep(1000);
        }
        state(valid && observed ? "passed" : "refused: late symbols or missing route calls");
    } else {
        state("refused: BSD did not publish");
    }
    thread_terminate(current_thread());
}
} // namespace

void graft_client_initialize() {
    if (!graft_client_ready(Client)) {
        state("refused: Graft unavailable");
        return;
    }
    void (*wrappers[])() = {graft_probe_0, graft_probe_1, graft_probe_2,
                            graft_probe_3, graft_probe_4, graft_probe_5};
    graft_route_request requests[6]{};
    for (unsigned i = 0; i < 6; ++i)
        requests[i] = {symbols[i], reinterpret_cast<void *>(wrappers[i]),
                       graft_probe_originals + i};
    if (!graft_route_kernel(Client, requests, 6)) {
        state("refused: routes");
        return;
    }
    state("waiting: late symbol and route checks");
    thread_t thread = THREAD_NULL;
    if (kernel_thread_start(lateCheck, nullptr, &thread) != KERN_SUCCESS) {
        state("refused: worker allocation");
        return;
    }
    thread_deallocate(thread);
}
