/*
 * RgfxSurfaceUserClient: see RgfxSurfaceUserClient.hpp. An IOAccel surface, over the client's
 * backing or the framebuffer's scanout.
 */

#include "RgfxSurfaceUserClient.hpp"

#include "RgfxAccelerator.hpp"
#include "RgfxSurfacePlace.hpp"
#include "RgfxTrace.hpp"

#include <IOKit/IOLib.h>
#include <IOKit/IOMemoryDescriptor.h>

#define super IOUserClient
OSDefineMetaClassAndStructors(RgfxSurfaceUserClient, IOUserClient);

namespace {
// A region in a method's structure input: its rectangle count, its bounds, then that many
// rectangles (IOAccelDeviceRegion).
bool readRegion(const IOExternalMethodArguments *a, IOAccelBounds *bounds)
{
	if (!a->structureInput || a->structureInputSize < sizeof(IOAccelDeviceRegion))
		return false;
	const IOAccelDeviceRegion *region =
	    static_cast<const IOAccelDeviceRegion *>(a->structureInput);
	if (region->num_rects > (a->structureInputSize - sizeof(IOAccelDeviceRegion)) /
	                            sizeof(IOAccelBounds))
		return false;
	*bounds = region->bounds;
	return bounds->w > 0 && bounds->h > 0;
}

// A lock's answer, into the method's structure output: inline, or through its descriptor, which
// is over the caller's memory, not wired, so prepared for the write and completed after it.
IOReturn writeLockInfo(IOExternalMethodArguments *a, const IOAccelSurfaceInformation &info)
{
	if (a->structureOutput) {
		memcpy(a->structureOutput, &info, sizeof(info));
	} else if (IOMemoryDescriptor *desc = a->structureOutputDescriptor) {
		IOReturn r = desc->prepare(kIODirectionIn);
		if (r != kIOReturnSuccess)
			return r;
		const bool written = desc->writeBytes(0, &info, sizeof(info)) == sizeof(info);
		desc->complete(kIODirectionIn);
		if (!written)
			return kIOReturnBadArgument;
	} else {
		return kIOReturnBadArgument;
	}
	a->structureOutputSize = sizeof(info);
	return kIOReturnSuccess;
}
} // namespace

// MARK: - Lifecycle

bool RgfxSurfaceUserClient::initWithTask(task_t owningTask, void *securityID, UInt32 type,
                                         OSDictionary *properties)
{
	if (!super::initWithTask(owningTask, securityID, type, properties))
		return false;
	task = owningTask;
	owner = nullptr;
	haveId = false;
	surfaceId = 0;
	modeBits = 0;
	haveShape = false;
	framebufferIndex = 0;
	bounds = {};
	backing = 0;
	backingRowBytes = 0;
	backingLength = 0;
	scanoutMap = nullptr;
	scanoutOffset = 0;
	scanoutLength = 0;
	scanoutRowBytes = 0;
	stateLock = IOLockAlloc();
	return stateLock != nullptr;
}

bool RgfxSurfaceUserClient::start(IOService *provider)
{
	owner = OSDynamicCast(RgfxAccelerator, provider);
	if (!owner)
		return false;
	return super::start(provider);
}

void RgfxSurfaceUserClient::stop(IOService *provider)
{
	dropSurface();
	owner = nullptr;
	super::stop(provider);
}

void RgfxSurfaceUserClient::free()
{
	if (stateLock)
		IOLockFree(stateLock);
	stateLock = nullptr;
	super::free();
}

IOReturn RgfxSurfaceUserClient::clientDied()
{
	dropSurface();
	return super::clientDied();
}

IOReturn RgfxSurfaceUserClient::clientClose()
{
	dropSurface();
	if (!isInactive())
		terminate();
	return kIOReturnSuccess;
}

// The surface goes: out of the accelerator's table, and its scanout mapping out of the task.
void RgfxSurfaceUserClient::dropSurface()
{
	if (owner)
		owner->unregisterSurface(this);
	if (!stateLock)
		return;
	IOLockLock(stateLock);
	IOMemoryMap *map = scanoutMap;
	scanoutMap = nullptr;
	haveId = false;
	IOLockUnlock(stateLock);
	OSSafeReleaseNULL(map);
}

bool RgfxSurfaceUserClient::dropScanoutMap()
{
	if (!stateLock)
		return false;
	IOLockLock(stateLock);
	IOMemoryMap *map = scanoutMap;
	scanoutMap = nullptr;
	IOLockUnlock(stateLock);
	bool had = map != nullptr;
	OSSafeReleaseNULL(map);
	return had;
}

// MARK: - The methods

IOReturn RgfxSurfaceUserClient::externalMethod(uint32_t selector,
                                               IOExternalMethodArguments *a,
                                               IOExternalMethodDispatch *dispatch,
                                               OSObject *target, void *reference)
{
	if (a && owner) {
		switch (selector) {
		case kIOAccelSurfaceSetIDMode:
			if (a->scalarInputCount == 2 && a->scalarOutputCount == 0 &&
			    a->structureOutputSize == 0)
				return setIDMode(a);
			break;
		case kIOAccelSurfaceSetShapeBackingAndLength:
			if (a->scalarInputCount == 5 && a->scalarOutputCount == 0 &&
			    a->structureOutputSize == 0)
				return setShape(a, true);
			break;
		case kIOAccelSurfaceSetShape:
			if (a->scalarInputCount == 2 && a->scalarOutputCount == 0 &&
			    a->structureOutputSize == 0)
				return setShape(a, false);
			break;
		case kIOAccelSurfaceReadLockOptions:
		case kIOAccelSurfaceWriteLockOptions:
			if (a->scalarInputCount == 1 && a->scalarOutputCount == 0 &&
			    a->structureInputSize == 0 &&
			    a->structureOutputSize == sizeof(IOAccelSurfaceInformation))
				return lock(a);
			break;
		case kIOAccelSurfaceFlush:
			// The pixels of a surface with no backing are the display's already: nothing to
			// copy. A backing surface's flush is not supported.
			if (a->scalarInputCount == 2 && a->scalarOutputCount == 0 &&
			    a->structureOutputSize == 0 && !hasBacking())
				return kIOReturnSuccess;
			break;
		case kIOAccelSurfaceReadUnlockOptions:
		case kIOAccelSurfaceWriteUnlockOptions:
			// The memory is the client's or the scanout, both coherent with the CPU: nothing
			// to hand back.
			if (a->scalarInputCount == 1 && a->scalarOutputCount == 0 &&
			    a->structureOutputSize == 0)
				return kIOReturnSuccess;
			break;
		default:
			break;
		}
	}
	static char site;
	if (rgfxFirstCalls(&site, selector, 1)) {
		RgfxCaller caller;
		IOLog("RGFX surface: method 0x%x from %s[%d] is not supported (scalars %u/%u, "
		      "struct %u/%u)\n",
		      selector, caller.name, caller.pid, a ? a->scalarInputCount : 0,
		      a ? a->scalarOutputCount : 0, a ? a->structureInputSize : 0,
		      a ? a->structureOutputSize : 0);
	}
	return kIOReturnUnsupported;
}

IOReturn RgfxSurfaceUserClient::setIDMode(IOExternalMethodArguments *a)
{
	UInt32 id = static_cast<UInt32>(a->scalarInput[0]);
	if (id == 0 || a->scalarInput[0] > UINT32_MAX || a->scalarInput[1] > UINT32_MAX)
		return kIOReturnBadArgument;
	IOReturn r = owner->registerSurface(this, id);
	if (r != kIOReturnSuccess)
		return r;
	IOLockLock(stateLock);
	haveId = true;
	surfaceId = id;
	modeBits = static_cast<UInt32>(a->scalarInput[1]);
	IOLockUnlock(stateLock);
	return kIOReturnSuccess;
}

IOReturn RgfxSurfaceUserClient::setShape(IOExternalMethodArguments *a, bool withBacking)
{
	IOAccelBounds shape;
	if (!readRegion(a, &shape) || a->scalarInput[1] > UINT32_MAX)
		return kIOReturnBadArgument;
	mach_vm_address_t newBacking = 0;
	UInt32 rowBytes = 0;
	UInt64 length = 0;
	if (withBacking && a->scalarInput[2]) {
		// The client's own memory: rows of at least the shape's width, as long as its rows.
		newBacking = a->scalarInput[2];
		length = a->scalarInput[4];
		if (a->scalarInput[3] > UINT32_MAX ||
		    a->scalarInput[3] < static_cast<UInt64>(shape.w) * 4 ||
		    length / a->scalarInput[3] < static_cast<UInt64>(shape.h))
			return kIOReturnBadArgument;
		rowBytes = static_cast<UInt32>(a->scalarInput[3]);
	}
	IOLockLock(stateLock);
	haveShape = true;
	framebufferIndex = static_cast<UInt32>(a->scalarInput[1]);
	bounds = shape;
	if (withBacking) {
		backing = newBacking;
		backingRowBytes = rowBytes;
		backingLength = length;
	}
	// The scanout's mapping stays: it maps the whole scanout, under any shape.
	IOLockUnlock(stateLock);
	return kIOReturnSuccess;
}

bool RgfxSurfaceUserClient::hasBacking()
{
	IOLockLock(stateLock);
	bool has = backing != 0;
	IOLockUnlock(stateLock);
	return has;
}

bool RgfxSurfaceUserClient::isScanout(UInt32 width, UInt32 height)
{
	IOLockLock(stateLock);
	bool scanout = haveShape && !backing && framebufferIndex == 0 && bounds.x == 0 &&
	               bounds.y == 0 && static_cast<UInt32>(bounds.w) == width &&
	               static_cast<UInt32>(bounds.h) == height;
	IOLockUnlock(stateLock);
	return scanout;
}

IOReturn RgfxSurfaceUserClient::lock(IOExternalMethodArguments *a)
{
	IOAccelSurfaceInformation info;
	bzero(&info, sizeof(info));

	RgfxAccelerator::Scanout scanout = {};
	bool haveScanout = owner->currentScanout(&scanout);
	// The scanout is its owner's alone (RgfxScanoutOwner.hpp).
	bool ownsScanout = owner->ownsScanout(task);
	IOLockLock(stateLock);
	if (!haveShape) {
		IOLockUnlock(stateLock);
		return kIOReturnNotReady;
	}
	info.width = static_cast<UInt32>(bounds.w);
	info.height = static_cast<UInt32>(bounds.h);
	// The one pixel format the protocol names: the color depth of the surface's mode.
	info.pixelFormat = modeBits & kIOAccelSurfaceModeColorDepthBits;
	if (backing) {
		info.address[0] = backing;
		info.rowBytes = backingRowBytes;
		IOLockUnlock(stateLock);
		return writeLockInfo(a, info);
	}
	if (!ownsScanout) {
		// Any other task's lock is refused, and a mapping of the scanout it holds from before
		// goes.
		IOMemoryMap *map = scanoutMap;
		scanoutMap = nullptr;
		IOLockUnlock(stateLock);
		OSSafeReleaseNULL(map);
		static char site;
		RgfxCaller caller;
		if (rgfxFirstCalls(&site, static_cast<UInt64>(caller.pid), 2))
			IOLog("RGFX surface: a lock of the scanout is refused to %s[%d], which does not own "
			      "it\n",
			      caller.name, caller.pid);
		return kIOReturnNotPermitted;
	}
	// The scanout's pixels under the shape (RgfxSurfacePlace.hpp).
	uint64_t at = 0;
	bool onScanout = haveScanout && framebufferIndex == 0 &&
	                 rgfxSurfaceInScanout(bounds.x, bounds.y, bounds.w, bounds.h, scanout.width,
	                                      scanout.height, scanout.rowBytes, &at);
	bool current = scanoutMap && scanoutOffset == scanout.offset &&
	               scanoutLength == scanout.length && scanoutRowBytes == scanout.rowBytes;
	IOLockUnlock(stateLock);
	if (!onScanout) {
		static char site;
		if (rgfxFirstCalls(&site, 0, 1))
			IOLog("RGFX surface: a lock of %ux%u at %d,%d on framebuffer %u has no memory "
			      "(scanout %ux%u)\n",
			      info.width, info.height, bounds.x, bounds.y, framebufferIndex, scanout.width,
			      scanout.height);
		return kIOReturnNoMemory;
	}
	if (!current) {
		// The scanout as it is now, mapped into the client's task as the bundle and the
		// framebuffer map the region: write-back. Kept across shapes, and made again when a
		// mode moves the scanout.
		IODeviceMemory *region = owner->copyRegion();
		IODeviceMemory *span =
		    region ? IODeviceMemory::withSubRange(region, scanout.offset, scanout.length)
		           : nullptr;
		OSSafeReleaseNULL(region);
		if (!span)
			return kIOReturnNoMemory;
		IOMemoryMap *made =
		    span->createMappingInTask(task, 0, kIOMapAnywhere | kIOMapCopybackCache);
		span->release();
		if (!made)
			return kIOReturnNoMemory;
		IOLockLock(stateLock);
		IOMemoryMap *old = scanoutMap;
		scanoutMap = made;
		scanoutOffset = scanout.offset;
		scanoutLength = scanout.length;
		scanoutRowBytes = scanout.rowBytes;
		IOLockUnlock(stateLock);
		OSSafeReleaseNULL(old);
	}
	IOLockLock(stateLock);
	info.address[0] = scanoutMap ? scanoutMap->getAddress() + at : 0;
	info.rowBytes = scanoutRowBytes;
	IOLockUnlock(stateLock);
	if (!info.address[0])
		return kIOReturnNoMemory;
	return writeLockInfo(a, info);
}
