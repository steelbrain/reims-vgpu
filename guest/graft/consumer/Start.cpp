#include <mach/mach_types.h>

#ifndef GRAFT_CLIENT_ID
#error Define GRAFT_CLIENT_ID as the client bundle identifier token
#endif
#ifndef GRAFT_CLIENT_VERSION
#error Define GRAFT_CLIENT_VERSION as a string
#endif

extern void graft_client_initialize();
extern "C" kern_return_t graft_start(kmod_info_t *, void *) {
    graft_client_initialize();
    return KERN_SUCCESS;
}
extern "C" kern_return_t graft_stop(kmod_info_t *, void *) {
    // Client routes and their original trampolines remain in use for boot.
    return KERN_FAILURE;
}
#define GRAFT_DECLARE(id, version) KMOD_EXPLICIT_DECL(id, version, graft_start, graft_stop)
extern "C" {
GRAFT_DECLARE(GRAFT_CLIENT_ID, GRAFT_CLIENT_VERSION)
}
