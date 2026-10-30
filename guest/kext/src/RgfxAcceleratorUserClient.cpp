/*
 * RgfxAcceleratorUserClient: see RgfxAcceleratorUserClient.hpp. The region map and the
 * doorbell, over our own protocol.
 */

#include "RgfxAcceleratorUserClient.hpp"

#include <IOKit/IOLib.h>
#include <IOKit/IOMemoryDescriptor.h>

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
	ucType = type; // RGFX DIAG (remove before merge): correlate selectors to UC type
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
	// RGFX DIAG (Wall-2 trace; remove before merge): which memory type the base maps.
	IOLog("RGFX: clientMemoryForType type=%u (ours=%u)\n", type, kRgfxAccelMemoryRegion);
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

// RGFX DIAG (IOGPU construction handshake; remove before merge): constants + a struct-output
// helper used by externalMethod to answer Apple's IOGPU device-creation ABI.
enum {
	kRgfxIOGPUDeviceInfoType = 5, // the UC type IOGPUDeviceCreateWithOptions opens for device info
	kRgfxIOGPUSelConfig64    = 0, // its selector 0: the 64-byte "Config64" struct
};
static const uint32_t kRgfxIOGPUStructOutMax = 1024;       // largest construction struct-out is 536
static const uint64_t kRgfxWorkingSetBytes   = 8ull << 30;  // Config64 0x10: working-set budget (8 GiB)
static const uint64_t kRgfxSharedMemBytes    = 256ull << 20; // Config64 0x20: shared-memory size (256 MiB)

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

	// RGFX DIAG (Wall-2 trace; remove before merge): which method the base calls via
	// the modern dispatch, its argument shape, and its scalar inputs (the input values
	// may reveal selector 7's semantics -- a port, a size, an address).
	uint32_t scalarIn = arguments ? arguments->scalarInputCount : 0;
	IOLog("RGFX: externalMethod ucType=%u selector=%u scalarIn=%u structIn=%u scalarOut=%u structOut=%u in0=0x%llx in1=0x%llx\n",
	      ucType, selector, scalarIn, arguments ? arguments->structureInputSize : 0,
	      arguments ? arguments->scalarOutputCount : 0,
	      arguments ? arguments->structureOutputSize : 0,
	      (arguments && scalarIn > 0) ? arguments->scalarInput[0] : 0ull,
	      (arguments && scalarIn > 1) ? arguments->scalarInput[1] : 0ull);

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

	// RGFX DIAG IOGPU construction handshake (remove before merge): answer Apple's IOGPU
	// device-creation ABI so IOGPUDeviceCreateWithOptions (run inside [super
	// initWithAcceleratorPort:]) can build the IOGPUMetalDevice base. During construction the
	// base opens a type-5 UC and calls selector 2 (536-byte device-info) then selector 0
	// (64-byte Config64); each MUST return success at its EXACT requested size, or Create
	// returns NULL and the device inits to nil. In the 536-byte blob, offset 0x14 is a u32
	// entry count the base mallocs (count*8) and loops over reading a u32 table at 0x18 --
	// garbage there faults it -- so we keep it 0 to skip the loop. In Config64, offset 0x10 is
	// the working-set budget and 0x20 the shared-memory size; both must be non-zero for a
	// usable device. Every other selector (the lazy command-queue / resource paths: type-0
	// sel 3/7/9/17 and all of type-4) is NOT reached during construction; a zeroed struct-out
	// plus benign scalar success keeps anything that does reach them moving until they are
	// built out.
	if (arguments) {
		for (uint32_t i = 0; i < arguments->scalarOutputCount; ++i)
			arguments->scalarOutput[i] = (i == 0) ? 1ull : 0ull;

		if (arguments->structureOutputSize > 0) {
			uint8_t buf[kRgfxIOGPUStructOutMax];
			uint32_t len = arguments->structureOutputSize;
			if (len > sizeof(buf))
				len = sizeof(buf);
			bzero(buf, len);
			if ((ucType & 0xffff) == kRgfxIOGPUDeviceInfoType &&
			    selector == kRgfxIOGPUSelConfig64 && len >= 0x28) {
				*reinterpret_cast<uint64_t *>(buf + 0x10) = kRgfxWorkingSetBytes;
				*reinterpret_cast<uint64_t *>(buf + 0x20) = kRgfxSharedMemBytes;
			}
			// selector 2 (536): the 0x14 entry count stays 0 (skip the base's loop).
			return rgfxWriteStructOut(arguments, buf, len);
		}
	}
	return kIOReturnSuccess;
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
