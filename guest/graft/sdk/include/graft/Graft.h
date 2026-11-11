// SPDX-License-Identifier: GPL-2.0-or-later OR LGPL-3.0-or-later
// Graft's interface for a client kext. Each call takes the caller's own bundle
// identifier, and Graft serves it when that identifier names a kext the boot
// linker placed: a fileset entry in the Boot collection on x86-64, a device-tree
// node the firmware wrote on arm64e, so any placed kext can be a client. Kernel
// and Boot-collection imports stay the boot linker's work; these services act on
// the running kernel. docs/design.md, "Owned clients", has the details.
#pragma once
#include <stddef.h>
#include <stdint.h>
#ifndef __cplusplus
#include <stdbool.h>
#endif

// One function to route: what to route, the replacement that receives its calls,
// and where Graft stores a callable original. On x86-64 `symbol` is a function in
// the kernel's own symbol table, and the replacement and original have that
// function's prototype. On arm64e `symbol` is instead a MAC policy operation name
// (for example "vnode_check_signature", not a kernel symbol), and the replacement
// and original have the operation's prototype: the mpo_*_t type from XNU's
// security/mac_policy.h, not the mac_* dispatcher's.
//
// Graft writes the original as a void *. Keep it in a void * and convert it to
// the prototype where you call it, on both architectures. Do not point
// `original` at a typed function-pointer variable: an arm64e kext built with
// -mkernel authenticates a call through a typed variable with a discriminator
// derived from its type, while a function pointer held in a void * carries none,
// so the conversion is what re-signs it.
typedef struct graft_route_request {
    const char *symbol;
    void *replacement;
    void **original;
} graft_route_request;

#ifdef __cplusplus
extern "C" {
#endif

// Whether Graft is active and identifier names a placed kext.
bool graft_client_ready(const char *identifier);

// Looks symbol up by name in the running kernel's own symbol table, such as
// "_csfg_get_cdhash", and stores its address, or 0. On a miss, the client's
// state becomes "refused", naming the symbol.
bool graft_kernel_symbol(const char *identifier, const char *symbol, uint64_t *address);

// Installs count routes, at most sixteen, all together or none. Each request's
// form is per-architecture (see graft_route_request). On x86-64 the routes are
// kernel-text trampolines and must be in place before BSD starts; on arm64 the
// route is one atomic swap of a policy table, safe at any time, so a client that
// matches as an IOKit driver installs after BSD. Each replacement must lie in an
// executable segment of the kext that identifier names, or the request is
// refused; Graft does not check which kext makes the call. Graft publishes the
// outcome as the client's state: "routed", or "refused" naming the route and
// the reason. It returns false and records nothing when Graft is not active, or
// count is 0 or more than sixteen. A kernel function another client already
// routes is refused, so routes across clients never collide.
// Routes live for the boot. Keep replacement code, wrapper state and original
// storage resident even on failure: an unverified rollback can retain a route.
bool graft_route_kernel(const char *identifier, const graft_route_request *requests, size_t count);

// Adds a line to Graft's log, prefixed with identifier. This and
// graft_client_state work whether or not Graft is active, so a client can record
// that Graft refused.
void graft_client_log(const char *identifier, const char *format, ...)
        __attribute__((format(printf, 2, 3)));

// Sets the client's state and reason in GraftClient.<identifier> on the
// IORegistry root, published as "<state>: <reason>".
void graft_client_state(const char *identifier, const char *state, const char *reason);

#ifdef __cplusplus
}
#endif
