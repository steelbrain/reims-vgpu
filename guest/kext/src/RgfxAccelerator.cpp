/*
 * RgfxAccelerator: see RgfxAccelerator.hpp for the service's role, the Metal
 * discovery keys it carries, and the boot-gated base-class decision.
 */

#include "RgfxAccelerator.hpp"

#include "ReimsGraphics.hpp"
#include "RgfxAcceleratorUserClient.hpp"
#include "RgfxDisplayPipeUserClient.hpp"

#include <IOKit/IOLib.h>

#include "RgfxTrace.hpp"

#define super IOAccelerator
OSDefineMetaClassAndStructors(RgfxAccelerator, IOAccelerator);

// MARK: - Registers

UInt32 RgfxAccelerator::readRegister(UInt32 offset) const
{
	return *reinterpret_cast<const volatile UInt32 *>(registers + offset);
}

void RgfxAccelerator::writeRegister(UInt32 offset, UInt32 value)
{
	*reinterpret_cast<volatile UInt32 *>(registers + offset) = value;
}

// MARK: - IOService

bool RgfxAccelerator::start(IOService *provider)
{
	ReimsGraphics *fb = nullptr;
	pci = OSDynamicCast(IOPCIDevice, provider);
	if (!pci)
		return false;
	doorbellLock = IOSimpleLockAlloc();
	if (!doorbellLock)
		return false;

	// The framebuffer enables these too; doing so again is harmless. The
	// accelerator needs BAR 0 for the doorbell and BAR 2 to hand to the bundle.
	pci->setMemoryEnable(true);
	pci->setBusLeadEnable(true);

	registerMap = pci->mapDeviceMemoryWithRegister(kIOPCIConfigBaseAddress0);
	if (!registerMap) {
		IOLog("RgfxAccelerator: BAR0 not mapped\n");
		goto fail;
	}
	registers = reinterpret_cast<volatile UInt8 *>(registerMap->getVirtualAddress());
	if (readRegister(RGFX_REG_MAGIC) != RGFX_MAGIC || readRegister(RGFX_REG_VERSION) != RGFX_VERSION) {
		IOLog("RgfxAccelerator: not the device: magic 0x%x, version %u\n",
		      readRegister(RGFX_REG_MAGIC), readRegister(RGFX_REG_VERSION));
		goto fail;
	}

	region = pci->getDeviceMemoryWithRegister(kIOPCIConfigBaseAddress2);
	if (!region) {
		IOLog("RgfxAccelerator: no BAR2\n");
		goto fail;
	}
	region->retain();
	regionSize = (static_cast<UInt64>(readRegister(RGFX_REG_REGION_SIZE_HI)) << 32) |
	             readRegister(RGFX_REG_REGION_SIZE_LO);
	if (regionSize > region->getLength())
		regionSize = region->getLength();
	if (regionSize <= RGFX_ALLOCATIONS_OFFSET) {
		IOLog("RgfxAccelerator: the region is too small: 0x%llx bytes\n", regionSize);
		goto fail;
	}

	// The Metal discovery keys (MetalPluginName / MetalPluginClassName) and the PCI
	// match that make Metal load the bundle are in src/Info.plist's accelerator
	// personality, so they are already registry properties on this service. The
	// accelerator does not touch pixels or the control ring (the framebuffer owns
	// those); it only publishes those keys and vends the region-map + doorbell user
	// client. registerService lets Metal's IOServiceMatching("IOAccelerator")
	// notification see it.
	if (!super::start(provider))
		goto fail;
	// Name this accelerator on the framebuffer, if it has started already (else the display
	// pipe's first association does).
	OSSafeReleaseNULL(fb);
	fb = copyFramebuffer();
	registerService();
	IOLog("RgfxAccelerator: ready, region 0x%llx bytes, framebuffer %s\n", regionSize,
	      fb ? "named" : "not yet started");
	OSSafeReleaseNULL(fb);
	return true;
fail:
	teardown();
	return false;
}

void RgfxAccelerator::stop(IOService *provider)
{
	// The framebuffer stops calling queueInterrupt (and lets this accelerator go).
	if (ReimsGraphics *fb = copyFramebuffer()) {
		fb->setQueueInterrupt(nullptr, nullptr);
		fb->release();
	}
	super::stop(provider);
	teardown();
}

void RgfxAccelerator::teardown()
{
	OSSafeReleaseNULL(region);
	OSSafeReleaseNULL(registerMap);
	registers = nullptr;
}

void RgfxAccelerator::free()
{
	if (doorbellLock)
		IOSimpleLockFree(doorbellLock);
	doorbellLock = nullptr;
	super::free();
}

// MARK: - The user client

IOReturn RgfxAccelerator::newUserClient(task_t owningTask, void *securityID, UInt32 type,
                                        OSDictionary *properties, IOUserClient **handler)
{
	// The types callers open, measured: ours (0), the display pipe (4), and the one the
	// IOGPUMetalDevice base builds its device over (5). Any other is refused, logged once per
	// type.
	RgfxCaller caller;
	IOLog("RGFX: newUserClient type=%u from %s[%d]\n", type, caller.name, caller.pid);
	if (type == kRgfxDisplayPipeUserClientType)
		return newDisplayPipe(owningTask, securityID, type, properties, handler);
	if (type != kRgfxAccelUserClientType && type != kRgfxIOGPUDeviceUserClientType) {
		static char site;
		if (rgfxFirstCalls(&site, type, 1))
			IOLog("RGFX: user client type %u from %s[%d] is not supported\n", type, caller.name,
			      caller.pid);
		return kIOReturnUnsupported;
	}

	RgfxAcceleratorUserClient *client = OSTypeAlloc(RgfxAcceleratorUserClient);
	if (!client)
		return kIOReturnNoMemory;
	if (!client->initWithTask(owningTask, securityID, type, properties)) {
		client->release();
		return kIOReturnBadArgument;
	}
	if (!client->attach(this)) {
		client->release();
		return kIOReturnInternalError;
	}
	if (!client->start(this)) {
		client->detach(this);
		client->release();
		return kIOReturnInternalError;
	}
	*handler = client;
	return kIOReturnSuccess;
}

// The display pipe WindowServer presents through (RgfxDisplayPipeUserClient).
IOReturn RgfxAccelerator::newDisplayPipe(task_t owningTask, void *securityID, UInt32 type,
                                         OSDictionary *properties, IOUserClient **handler)
{
	RgfxDisplayPipeUserClient *client = OSTypeAlloc(RgfxDisplayPipeUserClient);
	if (!client)
		return kIOReturnNoMemory;
	if (!client->initWithTask(owningTask, securityID, type, properties)) {
		client->release();
		return kIOReturnBadArgument;
	}
	if (!client->attach(this)) {
		client->release();
		return kIOReturnInternalError;
	}
	if (!client->start(this)) {
		client->detach(this);
		client->release();
		return kIOReturnInternalError;
	}
	*handler = client;
	return kIOReturnSuccess;
}

// MARK: - Reached from the user client

ReimsGraphics *RgfxAccelerator::copyFramebuffer()
{
	if (!pci)
		return nullptr;
	ReimsGraphics *found = nullptr;
	OSIterator *clients = pci->getClientIterator();
	if (clients) {
		while (OSObject *o = clients->getNextObject()) {
			found = OSDynamicCast(ReimsGraphics, o);
			if (found) {
				found->retain();
				break;
			}
		}
		clients->release();
	}
	if (found && !found->getProperty("IOAccelTypes")) {
		char path[512];
		int length = sizeof(path);
		if (getPath(path, &length, gIOServicePlane))
			found->setProperty("IOAccelTypes", path);
	}
	return found;
}

IODeviceMemory *RgfxAccelerator::copyRegion() const
{
	if (region)
		region->retain();
	return region;
}

void RgfxAccelerator::ringDoorbell(UInt32 ringId)
{
	if (!registers)
		return;
	IOInterruptState s = IOSimpleLockLockDisableInterrupt(doorbellLock);
	writeRegister(RGFX_REG_DOORBELL, ringId);
	IOSimpleLockUnlockEnableInterrupt(doorbellLock, s);
}

#if RGFX_FB_TRACE
// MARK: - Trace: the properties processes read

OSObject *RgfxAccelerator::copyProperty(const OSSymbol *aKey) const
{
	OSObject *o = super::copyProperty(aKey);
	rgfxTracePropertyRead("accel", aKey ? aKey->getCStringNoCopy() : nullptr, o != nullptr);
	return o;
}

OSObject *RgfxAccelerator::copyProperty(const OSString *aKey) const
{
	OSObject *o = super::copyProperty(aKey);
	rgfxTracePropertyRead("accel", aKey ? aKey->getCStringNoCopy() : nullptr, o != nullptr);
	return o;
}

OSObject *RgfxAccelerator::copyProperty(const char *aKey) const
{
	OSObject *o = super::copyProperty(aKey);
	rgfxTracePropertyRead("accel", aKey, o != nullptr);
	return o;
}
#endif
