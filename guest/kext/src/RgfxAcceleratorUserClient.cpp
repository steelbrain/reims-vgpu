/*
 * RgfxAcceleratorUserClient: see RgfxAcceleratorUserClient.hpp. The region map and the
 * doorbell, over our own protocol.
 */

#include "RgfxAcceleratorUserClient.hpp"

#include <IOKit/IOLib.h>
#include <IOKit/IOMemoryDescriptor.h>

#include "RgfxIOGPUReplies.hpp"
#include "RgfxTrace.hpp"

#define super IOUserClient
OSDefineMetaClassAndStructors(RgfxAcceleratorUserClient, IOUserClient);

// MARK: - Lifecycle

// Whether `task`, which is opening the client, is WindowServer. The kernel's accessors of a
// process's code-signing identifier and platform-binary status are private KPIs, which a kext
// outside Apple's namespace cannot link. So the test is the process's name and an entitlement of
// WindowServer's that Apple alone can grant: AMFI lets no binary hold a com.apple.private
// entitlement unless Apple signed it, so a process merely named WindowServer fails the test.
// WindowServer holds com.apple.private.graphics-restart-no-kill on macOS 26.7.1 (an Intel Mac)
// and macOS 27 (an Apple-silicon Mac), read from its signature. The client is made on the
// opening thread, so the current process is the task's.
static bool rgfxIsWindowServer(task_t task)
{
	if (task != current_task())
		return false;
	char name[MAXCOMLEN + 1] = {};
	proc_selfname(name, sizeof(name));
	if (strncmp(name, "WindowServer", sizeof(name)) != 0)
		return false;
	OSObject *held =
	    IOUserClient::copyClientEntitlement(task, "com.apple.private.graphics-restart-no-kill");
	const bool ok = held == kOSBooleanTrue;
	OSSafeReleaseNULL(held);
	return ok;
}

bool RgfxAcceleratorUserClient::initWithTask(task_t owningTask, void *securityID, UInt32 type,
                                             OSDictionary *properties)
{
	if (!super::initWithTask(owningTask, securityID, type, properties))
		return false;
	task = owningTask;
	windowServer = rgfxIsWindowServer(owningTask);
	if (windowServer)
		IOLog("RgfxAccelerator: WindowServer's client, whose blocks may take the reserve\n");
	owner = nullptr;
	ucType = type;
	hasContext = false;
	context = 0;
	slot = 0;
	queueCount = 0;
	creatingQueue = 0;
	queueLock = IOLockAlloc();
	listLock = IOLockAlloc();
	completionLock = IOLockAlloc();
	completionCount = 0;
	completionsClosed = false;
	windowLock = IOLockAlloc();
	windowMemory = nullptr;
	windowMap = nullptr;
	windowBase = 0;
	windowSize = 0;
	grants = grantNs = revokes = revokeNs = 0;
	return queueLock != nullptr && listLock != nullptr && completionLock != nullptr &&
	       windowLock != nullptr;
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
	if (listLock)
		IOLockFree(listLock);
	listLock = nullptr;
	if (completionLock)
		IOLockFree(completionLock);
	completionLock = nullptr;
	closeWindow();
	if (windowLock)
		IOLockFree(windowLock);
	windowLock = nullptr;
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

// MARK: - No region mapping

IOReturn RgfxAcceleratorUserClient::clientMemoryForType(UInt32 type, IOOptionBits *options,
                                                        IOMemoryDescriptor **memory)
{
	// The region holds every process's blocks, the rings and the device page: none of it is
	// mapped whole. The process's part of it is in its window (openWindow), and the device info
	// comes from kRgfxAccelMethodDeviceInfo. Logged once per (user-client type, memory type).
	static char site;
	if (rgfxFirstCalls(&site, (static_cast<UInt64>(ucType) << 32) | type, 1)) {
		RgfxCaller caller;
		IOLog("RGFX: user client type %u: memory type %u from %s[%d] is not mapped\n", ucType,
		      type, caller.name, caller.pid);
	}
	return kIOReturnUnsupported;
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
// (kRgfxIOGPUDeviceUserClientType), and the two calls it makes there: RgfxIOGPUReplies.hpp.

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
	uint8_t reply[kRgfxIOGPUDeviceInfoSize];
	static_assert(kRgfxIOGPUConfigSize <= sizeof(reply), "the configuration fits the reply");
	RgfxMemorySize memory;
	const uint64_t dedicated = owner && owner->readMemorySize(&memory) ? memory.dedicatedBytes : 0;
	rgfxIOGPUDeviceReply(selector, reply, dedicated, owner ? owner->getRegionSize() : 0);
	return rgfxWriteStructOut(arguments, reply,
	                          selector == kRgfxIOGPUSelConfig ? kRgfxIOGPUConfigSize
	                                                          : kRgfxIOGPUDeviceInfoSize);
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

	// Our own protocol: the process's context (its ids, blocks and queues), and the features it
	// works by, on our own user client alone.
	if (ucType == kRgfxAccelUserClientType && arguments &&
	    ((selector >= kRgfxAccelMethodContextFirst && selector <= kRgfxAccelMethodContextLast) ||
	     selector == kRgfxAccelMethodFeatures || selector == kRgfxAccelMethodTakeAnswers))
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
	if (ucType == kRgfxIOGPUDeviceUserClientType && arguments &&
	    rgfxIOGPUDeviceReplySize(selector, arguments->scalarInputCount, arguments->scalarOutputCount,
	                             arguments->structureInputSize, arguments->structureOutputSize))
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
