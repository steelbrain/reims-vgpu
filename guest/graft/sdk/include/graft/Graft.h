// SPDX-License-Identifier: GPL-2.0-or-later OR LGPL-3.0-or-later
// Graft's interface for its owned client. Each call takes the client's bundle
// identifier, and Graft serves only RgfxGate's, "com.steelbrain.RgfxGate".
// Kernel and Boot-collection imports stay the boot linker's work; these
// services act on the running kernel. docs/design.md, "Owned clients", has
// the details.
#pragma once
#include <stddef.h>
#include <stdint.h>
#ifndef __cplusplus
#include <stdbool.h>
#endif

// One function to route: the kernel symbol to route, the replacement that
// receives its calls, and where Graft stores a callable original.
typedef struct graft_route_request {
    const char *symbol;
    void *replacement;
    void **original;
} graft_route_request;

#ifdef __cplusplus
extern "C" {
#endif

// Whether Graft is active and identifier is the owned client's.
bool graft_client_ready(const char *identifier);

// Looks symbol up by name in the running kernel's own symbol table, such as
// "_csfg_get_cdhash", and stores its address, or 0. On a miss, the client's
// state becomes "refused", naming the symbol.
bool graft_kernel_symbol(const char *identifier, const char *symbol, uint64_t *address);

// Installs count routes, at most six, all together or none, before BSD
// starts. Graft publishes the outcome as the client's state: "routed", or
// "refused" naming the route and the reason. It returns false and records
// nothing when Graft is not active, or count is 0 or more than six.
// Routes live for the boot. Keep replacement code, wrapper state and original
// storage resident even on failure: an unverified rollback can retain a route.
bool graft_route_kernel(const char *identifier, const graft_route_request *requests, size_t count);

// Adds a line to Graft's log, prefixed with identifier.
void graft_client_log(const char *identifier, const char *format, ...)
        __attribute__((format(printf, 2, 3)));

// Sets the client's state and reason in GraftClient.<identifier> on the
// IORegistry root, published as "<state>: <reason>".
void graft_client_state(const char *identifier, const char *state, const char *reason);

#ifdef __cplusplus
}
#endif
