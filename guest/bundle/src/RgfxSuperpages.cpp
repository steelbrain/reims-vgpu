/*
 * RgfxSuperpages: see RgfxSuperpages.hpp. Our own code.
 */

#include "RgfxSuperpages.hpp"

#include <mach/mach.h>
#include <mach/mach_vm.h>

namespace {
constexpr int kTag = VM_MAKE_TAG(VM_MEMORY_IOACCELERATOR);

int kernelSuperpages(void *, uint64_t *at, uint64_t size, bool fixed)
{
	mach_vm_address_t a = fixed ? *at : 0;
	const int where = fixed ? VM_FLAGS_FIXED : VM_FLAGS_ANYWHERE;
	const kern_return_t kr =
	    mach_vm_allocate(mach_task_self(), &a, size, where | VM_FLAGS_SUPERPAGE_SIZE_2MB | kTag);
	if (kr == KERN_SUCCESS)
		*at = a;
	return kr;
}

int kernelReserve(void *, uint64_t *at, uint64_t size)
{
	// Never touched: the superpages take its place, piece by piece. Not inherited, as superpages
	// are not.
	mach_vm_address_t a = 0;
	const kern_return_t kr =
	    mach_vm_map(mach_task_self(), &a, size, kRgfxSuperpage - 1, VM_FLAGS_ANYWHERE | kTag,
	                MEMORY_OBJECT_NULL, 0, FALSE, VM_PROT_NONE, VM_PROT_DEFAULT, VM_INHERIT_NONE);
	if (kr == KERN_SUCCESS)
		*at = a;
	return kr;
}

void kernelDeallocate(void *, uint64_t at, uint64_t size)
{
	mach_vm_deallocate(mach_task_self(), at, size);
}
} // namespace

RgfxSuperpageVm rgfxKernelSuperpageVm()
{
	RgfxSuperpageVm vm;
	vm.superpages = &kernelSuperpages;
	vm.reserve = &kernelReserve;
	vm.deallocate = &kernelDeallocate;
	return vm;
}

int rgfxTakeSuperpages(const RgfxSuperpageVm &vm, uint64_t size, uint8_t **bytes)
{
	if (size == 0 || size % kRgfxSuperpage != 0)
		return KERN_INVALID_ARGUMENT;
	uint64_t base = 0;
	if (size == kRgfxSuperpage) {
		const int kr = vm.superpages(vm.context, &base, size, false);
		if (kr == KERN_SUCCESS)
			*bytes = reinterpret_cast<uint8_t *>(base);
		return kr;
	}
	int kr = vm.reserve(vm.context, &base, size);
	if (kr != KERN_SUCCESS)
		return kr;
	for (uint64_t offset = 0; offset < size; offset += kRgfxSuperpage) {
		uint64_t at = base + offset;
		vm.deallocate(vm.context, at, kRgfxSuperpage);
		kr = vm.superpages(vm.context, &at, kRgfxSuperpage, true);
		if (kr != KERN_SUCCESS) {
			// The superpages before the hole and the reservation after it; the hole stays as
			// the refusal left it.
			if (offset != 0)
				vm.deallocate(vm.context, base, offset);
			const uint64_t rest = offset + kRgfxSuperpage;
			if (rest < size)
				vm.deallocate(vm.context, base + rest, size - rest);
			return kr;
		}
	}
	*bytes = reinterpret_cast<uint8_t *>(base);
	return KERN_SUCCESS;
}

void rgfxGiveSuperpages(const RgfxSuperpageVm &vm, uint8_t *bytes, uint64_t size)
{
	vm.deallocate(vm.context, reinterpret_cast<uintptr_t>(bytes), size);
}
