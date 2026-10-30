/*
 * RgfxAcceleratorUserClient: see RgfxAcceleratorUserClient.hpp. The region map and the
 * doorbell, over our own protocol.
 */

#include "RgfxAcceleratorUserClient.hpp"

#include <IOKit/IOLib.h>
#include <IOKit/IOMemoryDescriptor.h>

#include "RgfxTrace.hpp"

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
	ucType = type;
	hasContext = false;
	context = 0;
	slot = 0;
	queueCount = 0;
	queueLock = IOLockAlloc();
	completionLock = IOLockAlloc();
	completionCount = 0;
	completionsClosed = false;
	return queueLock != nullptr && completionLock != nullptr;
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
	releaseContext();
	owner = nullptr;
	super::stop(provider);
}

void RgfxAcceleratorUserClient::free()
{
	if (queueLock)
		IOLockFree(queueLock);
	queueLock = nullptr;
	if (completionLock)
		IOLockFree(completionLock);
	completionLock = nullptr;
	super::free();
}

IOReturn RgfxAcceleratorUserClient::clientDied()
{
	// The process died: what it held of the device goes, as at a close.
	releaseContext();
	return super::clientDied();
}

IOReturn RgfxAcceleratorUserClient::clientClose()
{
	releaseContext();
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
	// RGFX DIAG (Wall-2 trace; remove before merge): which memory type the base maps.
	RgfxCaller caller;
	IOLog("RGFX: clientMemoryForType type=%u on user client type %u from %s[%d] (ours=%u)\n", type,
	      ucType, caller.name, caller.pid, kRgfxAccelMemoryRegion);
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

// The user client Apple's IOGPU base opens to build the Metal device
// (kRgfxIOGPUDeviceUserClientType, from IOGPUDeviceCreateWithOptions inside [super
// initWithAcceleratorPort:]), and the two calls it
// makes there, measured: selector 2, the device's info (536 bytes out), then selector 0, its
// configuration (64 bytes out). Each must succeed at its exact size, or the device is nil.
enum {
	kRgfxIOGPUSelConfig = 0,
	kRgfxIOGPUSelDeviceInfo = 2,
};
static const uint32_t kRgfxIOGPUConfigSize = 64;
static const uint32_t kRgfxIOGPUDeviceInfoSize = 536;
// The configuration's working-set budget (at 0x10). Its shared-memory size (at 0x20) is the
// region's: BAR 2's size, from the device.
static const uint64_t kRgfxWorkingSetBytes = 8ull << 30;

// Copy a reply into the method's struct-output, whether IOKit provided an inline buffer or an
// output memory descriptor, and report the exact length written.
static IOReturn rgfxWriteStructOut(IOExternalMethodArguments *a, const void *buf, uint32_t len)
{
	if (a->structureOutputSize < len)
		return kIOReturnBadArgument;
	if (a->structureOutput) {
		memcpy(a->structureOutput, buf, len);
	} else if (a->structureOutputDescriptor) {
		a->structureOutputDescriptor->writeBytes(0, buf, len);
	} else {
		return kIOReturnBadArgument;
	}
	a->structureOutputSize = len;
	return kIOReturnSuccess;
}

IOReturn RgfxAcceleratorUserClient::deviceMethod(uint32_t selector,
                                                 IOExternalMethodArguments *arguments)
{
	if (selector == kRgfxIOGPUSelConfig) {
		uint8_t config[kRgfxIOGPUConfigSize] = {};
		*reinterpret_cast<uint64_t *>(config + 0x10) = kRgfxWorkingSetBytes;
		*reinterpret_cast<uint64_t *>(config + 0x20) = owner ? owner->getRegionSize() : 0;
		return rgfxWriteStructOut(arguments, config, sizeof(config));
	}
	// The device info, all zero: its u32 at 0x14 counts the u32 entries of a table at 0x18,
	// which the base reads, so the device has none.
	uint8_t info[kRgfxIOGPUDeviceInfoSize] = {};
	return rgfxWriteStructOut(arguments, info, sizeof(info));
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

#if RGFX_FB_TRACE
	// A trace build logs the first calls of each (user-client type, selector): their shape,
	// their first scalars and their caller.
	{
		static char site;
		if (arguments && rgfxFirstCalls(&site, (static_cast<UInt64>(ucType) << 32) | selector, 6)) {
			RgfxCaller caller;
			UInt64 in[5] = {};
			for (UInt32 i = 0; i < arguments->scalarInputCount && i < 5; i++)
				in[i] = arguments->scalarInput[i];
			IOLog("RGFX uc type %u sel 0x%x from %s[%d]: scalars %u/%u struct %u/%u%s%s, in "
			      "0x%llx 0x%llx 0x%llx 0x%llx 0x%llx\n",
			      ucType, selector, caller.name, caller.pid, arguments->scalarInputCount,
			      arguments->scalarOutputCount, arguments->structureInputSize,
			      arguments->structureOutputSize,
			      arguments->structureInputDescriptor ? " inDesc" : "",
			      arguments->structureOutputDescriptor ? " outDesc" : "", in[0], in[1], in[2],
			      in[3], in[4]);
		}
	}
#endif

	// Our own protocol: the process's context (its ids, blocks and queues), on our own user
	// client alone.
	if (ucType == kRgfxAccelUserClientType && arguments && selector >= kRgfxAccelMethodOpenContext &&
	    selector <= kRgfxAccelMethodSurfaceScanout)
		return contextMethod(selector, arguments);

	// Our own protocol: the doorbell. ONLY on our own user client (type 0). Apple's IOGPU
	// construction opens other UC types (5, 4) and also drives selector 0 on them -- type-5
	// selector 0 is Config64, a 64-byte struct-out -- so the doorbell must be gated on our
	// own type and arg shape, or Config64 would misroute into the doorbell dispatch (which
	// requires one scalar in, no output) and fail with kIOReturnBadArgument, NULLing the
	// device-create and leaving the MTLDevice nil.
	if (ucType == kRgfxAccelUserClientType && selector == kRgfxAccelMethodDoorbell && arguments &&
	    arguments->scalarInputCount == 1 && arguments->scalarOutputCount == 0 &&
	    arguments->structureOutputSize == 0) {
		dispatch = const_cast<IOExternalMethodDispatch *>(&methods[kRgfxAccelMethodDoorbell]);
		target = this;
		return super::externalMethod(selector, arguments, dispatch, target, reference);
	}

	// The IOGPU base's device calls, on its own client.
	if (ucType == kRgfxIOGPUDeviceUserClientType && arguments && arguments->scalarInputCount == 0 &&
	    arguments->scalarOutputCount == 0 && arguments->structureInputSize == 0 &&
	    ((selector == kRgfxIOGPUSelConfig &&
	      arguments->structureOutputSize == kRgfxIOGPUConfigSize) ||
	     (selector == kRgfxIOGPUSelDeviceInfo &&
	      arguments->structureOutputSize == kRgfxIOGPUDeviceInfoSize)))
		return deviceMethod(selector, arguments);

	// Nothing else is ours to answer. A caller of a selector we do not implement gets
	// kIOReturnUnsupported and no output, logged once per (user-client type, selector).
	static char site;
	if (rgfxFirstCalls(&site, (static_cast<UInt64>(ucType) << 32) | selector, 1)) {
		RgfxCaller caller;
		IOLog("RGFX: user client type %u: selector 0x%x from %s[%d] is not supported "
		      "(scalars %u/%u, struct %u/%u)\n",
		      ucType, selector, caller.name, caller.pid, arguments ? arguments->scalarInputCount : 0,
		      arguments ? arguments->scalarOutputCount : 0,
		      arguments ? arguments->structureInputSize : 0,
		      arguments ? arguments->structureOutputSize : 0);
	}
	return kIOReturnUnsupported;
}

// RGFX DIAG (Wall-2 trace; remove before merge): the legacy external-method and trap
// dispatch paths. If the IOGPUMetalDevice base uses the old ABI instead of
// externalMethod, these fire and name the index it wants; super returns none.
IOExternalMethod *RgfxAcceleratorUserClient::getTargetAndMethodForIndex(IOService **targetP,
                                                                        UInt32 index)
{
	IOLog("RGFX: getTargetAndMethodForIndex index=%u\n", index);
	return super::getTargetAndMethodForIndex(targetP, index);
}

IOExternalTrap *RgfxAcceleratorUserClient::getTargetAndTrapForIndex(IOService **targetP,
                                                                    UInt32 index)
{
	IOLog("RGFX: getTargetAndTrapForIndex index=%u\n", index);
	return super::getTargetAndTrapForIndex(targetP, index);
}
