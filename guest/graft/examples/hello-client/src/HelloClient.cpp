// SPDX-License-Identifier: GPL-2.0-or-later OR LGPL-3.0-or-later
// A minimal owned client, to read alongside docs/consumers.md. An owned client
// is an ordinary kext that links against Graft's exported C interface
// (graft/Graft.h) through the start glue sdk/Start.cpp. The glue declares the
// kmod from GRAFT_CLIENT_ID and calls graft_client_initialize() once, from the
// kext's start routine.
//
// Graft serves a call when its identifier names a placed kext, so a client
// passes its own bundle identifier (here ai.aneesiqbal.graft.HelloClient). A
// call whose identifier names no placed kext is ignored.
//
// This example does the smallest useful thing: it confirms Graft is active, asks
// for one kernel symbol by name, logs the address, and publishes its state. It
// installs no routes; routing a kernel function is shown by the test client
// tests/kexts/core-probe.
#include <IOKit/IOService.h>
#include <graft/Graft.h>

// An injected kext is not started for OSBundleRequired Root alone, so the client
// has an IOService subclass and a personality matching IOResources (Info.plist),
// which makes the kernel start the kext and run graft_client_initialize(). The
// class itself does no work.
class GraftHelloClient final : public IOService {
    OSDeclareDefaultStructors(GraftHelloClient)
};
OSDefineMetaClassAndStructors(GraftHelloClient, IOService);

namespace {
// This kext's own bundle identifier, which GRAFT_CLIENT_ID also sets. Graft
// serves the call because this identifier names a placed kext.
constexpr const char *Client = "ai.aneesiqbal.graft.HelloClient";
} // namespace

void graft_client_initialize() {
    // graft_client_ready is true only when Graft is active and the identifier
    // names a placed kext. Graft initializes in its own start routine, before a
    // client runs, so this is the ordinary case; a false here means Graft
    // refused to start and its own GraftState says why.
    if (!graft_client_ready(Client)) {
        graft_client_state(Client, "refused", "Graft not active");
        return;
    }
    // Look one symbol up in the running kernel's own symbol table. _IOLog is
    // exported on every release. graft_kernel_symbol returns false and sets the
    // client's state to "refused" naming the symbol if it is missing, and stores
    // 0 in that case.
    uint64_t address = 0;
    if (!graft_kernel_symbol(Client, "_IOLog", &address))
        return;
    graft_client_log(Client, "_IOLog resolves to 0x%llx", address);
    graft_client_state(Client, "ready", "resolved _IOLog");
}
