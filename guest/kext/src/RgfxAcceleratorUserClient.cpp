/*
 * RgfxAcceleratorUserClient: see RgfxAcceleratorUserClient.hpp. The region map and the
 * doorbell, over our own protocol.
 */

#include "RgfxAcceleratorUserClient.hpp"

#include <IOKit/IOLib.h>

#define super IOUserClient
OSDefineMetaClassAndStructors(RgfxAcceleratorUserClient, IOUserClient);

// MARK: - Lifecycle

bool RgfxAcceleratorUserClient::initWithTask(task_t owningTask, void *securityID, UInt32 type,
                                             OSDictionary *properties)
{
	if (!super::initWithTask(owningTask, securityID, type, properties))
		return false;
	task = owningTask;
	owner = nullptr;
	return true;
}

bool RgfxAcceleratorUserClient::start(IOService *provider)
{
	owner = OSDynamicCast(RgfxAccelerator, provider);
	if (!owner)
		return false;
	return super::start(provider);
}

void RgfxAcceleratorUserClient::stop(IOService *provider)
{
	owner = nullptr;
	super::stop(provider);
}

IOReturn RgfxAcceleratorUserClient::clientClose()
{
	// IOUserClient releases us when the last map is gone; detach from the accelerator
	// so the connection ends.
	if (!isInactive())
		terminate();
	return kIOReturnSuccess;
}

// MARK: - The region mapping

IOReturn RgfxAcceleratorUserClient::clientMemoryForType(UInt32 type, IOOptionBits *options,
                                                        IOMemoryDescriptor **memory)
{
	if (type != kRgfxAccelMemoryRegion)
		return kIOReturnUnsupported;
	if (!owner)
		return kIOReturnNotReady;

	IODeviceMemory *region = owner->copyRegion();
	if (!region)
		return kIOReturnNoMemory;

	// The region is host memory behind a prefetchable BAR; map it write-back, as the
	// framebuffer maps it in the kernel, so the bundle's ring and resource writes are
	// cached. IOUserClient releases this reference when the last map is destroyed.
	*options |= kIOMapCopybackCache;
	*memory = region;
	return kIOReturnSuccess;
}

// MARK: - External methods (the doorbell)

IOReturn RgfxAcceleratorUserClient::doorbellMethod(OSObject *target, void *,
                                                   IOExternalMethodArguments *arguments)
{
	RgfxAcceleratorUserClient *self = OSDynamicCast(RgfxAcceleratorUserClient, target);
	if (!self || !self->owner)
		return kIOReturnNotReady;
	self->owner->ringDoorbell(static_cast<UInt32>(arguments->scalarInput[0]));
	return kIOReturnSuccess;
}

IOReturn RgfxAcceleratorUserClient::externalMethod(uint32_t selector,
                                                   IOExternalMethodArguments *arguments,
                                                   IOExternalMethodDispatch *dispatch,
                                                   OSObject *target, void *reference)
{
	// Indexed by selector; kRgfxAccelMethodDoorbell is 0. One scalar in (the ring
	// id), nothing else.
	static const IOExternalMethodDispatch methods[kRgfxAccelMethodCount] = {
		{ &RgfxAcceleratorUserClient::doorbellMethod, 1, 0, 0, 0 },
	};

	if (selector >= kRgfxAccelMethodCount)
		return kIOReturnBadArgument;
	dispatch = const_cast<IOExternalMethodDispatch *>(&methods[selector]);
	target = this;
	return super::externalMethod(selector, arguments, dispatch, target, reference);
}
