/*
 * RgfxSuperpages: see RgfxSuperpages.hpp. Our own code.
 */

#include "RgfxSuperpages.hpp"

#include <mach/mach.h>
#include <mach/mach_vm.h>

int rgfxTakeSuperpage(uint8_t **bytes)
{
	mach_vm_address_t at = 0;
	const kern_return_t kr =
	    mach_vm_allocate(mach_task_self(), &at, kRgfxSuperpage,
	                     VM_FLAGS_ANYWHERE | VM_FLAGS_SUPERPAGE_SIZE_2MB |
	                         VM_MAKE_TAG(VM_MEMORY_IOACCELERATOR));
	if (kr == KERN_SUCCESS)
		*bytes = reinterpret_cast<uint8_t *>(at);
	return kr;
}
