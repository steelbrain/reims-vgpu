// Private interface for our guest kexts. Kernel/Boot-KC imports remain the
// boot linker's responsibility; these services operate on the running kernel.
#pragma once
#include <stddef.h>
#include <stdint.h>
#ifndef __cplusplus
#include <stdbool.h>
#endif

typedef struct graft_route_request {
    const char *symbol;
    void *replacement;
    void **original;
} graft_route_request;

#ifdef __cplusplus
extern "C" {
#endif
bool graft_client_ready(const char *identifier);
bool graft_kernel_symbol(const char *identifier, const char *symbol, uint64_t *address);
// Routes live for the boot. Keep replacement code, wrapper state and original
// storage resident even on failure: an unverified rollback can retain a route.
bool graft_route_kernel(const char *identifier, const graft_route_request *requests, size_t count);
void graft_client_log(const char *identifier, const char *format, ...) __attribute__((format(printf, 2, 3)));
void graft_client_state(const char *identifier, const char *state, const char *reason);
#ifdef __cplusplus
}
#endif
