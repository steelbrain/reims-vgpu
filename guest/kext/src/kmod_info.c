/*
 * The kext's kmod_info, which the kernel reads for its identity, and the
 * start and stop routines it names. The kernel runs the C++ constructors and
 * destructors itself, so they have nothing to do.
 */

#include <mach/mach_types.h>

kern_return_t _start(kmod_info_t *ki, void *data);
kern_return_t _stop(kmod_info_t *ki, void *data);

kern_return_t _start(kmod_info_t *ki, void *data)
{
	return KERN_SUCCESS;
}

kern_return_t _stop(kmod_info_t *ki, void *data)
{
	return KERN_SUCCESS;
}

// The macro quotes its first argument as written, so the ID is spelled here.
KMOD_EXPLICIT_DECL(com.steelbrain.ReimsGraphics, BUNDLE_VERSION, _start, _stop)
