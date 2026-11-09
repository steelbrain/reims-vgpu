/*
 * RgfxAcceleratorUserClient's window (RgfxAcceleratorUserClient.hpp): the one range of the
 * process's address space where the client's part of the region lies, each byte at the window's
 * base plus its region offset, as the bundle reaches it.
 *
 * The window is the region's size, made at the context's open over memory of the client's own:
 * a pageable buffer, mapped read-only anywhere in the process and nowhere else, which holds no
 * page until the process reads one there and then reads zeros. The buffer is purgeable, and never
 * made volatile, so the process's footprint grows by the window's pages it reads, and not by the
 * region's size at the open (RgfxWindowSpans.hpp, kRgfxWindowMemoryOptions, says why); the
 * region's spans are device memory, which no footprint counts. The process sees the region only
 * where the kext maps it in, over the window at the span's own offset (mapSpan): its blocks, the
 * shared blocks of the surface planes it holds, and the scanout it is granted. The kext maps a
 * span in before the process learns of it, and maps the window's own memory back over it,
 * read-only, before the span can go to anyone else (unmapSpan): a released span reads zeros and
 * a write there faults. So a process never maps a byte of the region that is not its own: no
 * ring, no device page, no other process's block.
 *
 * Placing a mapping at a fixed address takes IOKit's overwrite mapping, which IOKit never takes
 * down itself (an overwrite mapping made no room, so its unmap frees none, read from the public
 * xnu source: IOMemoryDescriptor::doUnmap). So each span's mapping is let go of as soon as it is
 * made, and only another overwrite mapping takes it back. The window's own mapping is an
 * ordinary one: letting it go at the close (closeWindow) frees the whole range, every span in it
 * with it, before the context's blocks are freed.
 *
 * Every failure fails the asking op with an IOReturn: nothing here panics or stops the device.
 *
 * Our own code; it names only SDK-public IOKit calls.
 */

#include "RgfxAcceleratorUserClient.hpp"
#include "RgfxWindowSpans.hpp"

#include <IOKit/IOBufferMemoryDescriptor.h>
#include <IOKit/IOLib.h>
#include <IOKit/IOMemoryDescriptor.h>
#include <kern/clock.h>

static_assert(kRgfxWindowDirectionInOut == kIODirectionInOut, "the window's direction");
static_assert(kRgfxWindowPageable == kIOMemoryPageable, "the window's pageable bit");
static_assert(kRgfxWindowPurgeable == kIOMemoryPurgeable, "the window's purgeable bit");

namespace {
UInt64 nanosecondsSince(UInt64 start)
{
	UInt64 ns = 0;
	absolutetime_to_nanoseconds(mach_absolute_time() - start, &ns);
	return ns;
}
} // namespace

IOReturn RgfxAcceleratorUserClient::openWindow()
{
	if (!owner || !windowLock)
		return kIOReturnNotReady;
	IOLockLock(windowLock);
	if (windowMap) {
		IOLockUnlock(windowLock);
		return kIOReturnSuccess;
	}
	const UInt64 size = owner->getRegionSize() & ~(kRgfxWindowPage - 1);
	// Made here, on the process's thread, so the process owns it and its footprint counts the
	// pages it reads there (RgfxWindowSpans.hpp).
	IOBufferMemoryDescriptor *memory = size ? IOBufferMemoryDescriptor::inTaskWithOptions(
	                                              TASK_NULL, kRgfxWindowMemoryOptions, size,
	                                              kRgfxWindowPage)
	                                        : nullptr;
	IOMemoryMap *map =
	    memory ? memory->createMappingInTask(task, 0, kIOMapAnywhere | kIOMapReadOnly, 0, size)
	           : nullptr;
	if (!map) {
		IOLockUnlock(windowLock);
		OSSafeReleaseNULL(memory);
		IOLog("RgfxAccelerator: no window of 0x%llx bytes for the context\n", size);
		return kIOReturnNoMemory;
	}
	windowMemory = memory;
	windowMap = map;
	windowBase = map->getAddress();
	windowSize = size;
	IOLockUnlock(windowLock);
	return kIOReturnSuccess;
}

void RgfxAcceleratorUserClient::closeWindow()
{
	if (!windowLock)
		return;
	IOLockLock(windowLock);
	IOMemoryMap *map = windowMap;
	IOBufferMemoryDescriptor *memory = windowMemory;
	windowMap = nullptr;
	windowMemory = nullptr;
	windowBase = 0;
	windowSize = 0;
	// The window's range goes, every span mapped in it with it.
	OSSafeReleaseNULL(map);
	OSSafeReleaseNULL(memory);
	const UInt64 n = grants, ns = grantNs, r = revokes, rns = revokeNs;
	grants = grantNs = revokes = revokeNs = 0;
	IOLockUnlock(windowLock);
	if (n || r)
		IOLog("RgfxAccelerator: context %u's window: %llu spans mapped in (%llu us), %llu "
		      "taken back (%llu us)\n",
		      context, n, ns / 1000, r, rns / 1000);
}

IOReturn RgfxAcceleratorUserClient::mapSpan(UInt64 offset, UInt64 length, bool writeCombined)
{
	if (!windowLock)
		return kIOReturnNotReady;
	// A write-combined span from the region's descriptor of its own, and every other span from
	// the region's (RgfxAccelerator::copyWriteCombinedRegion says why); where the first is
	// missing, as on arm64 (RgfxRegionMapping.hpp), the span is mapped write-back.
	IODeviceMemory *region =
	    owner && writeCombined ? owner->copyWriteCombinedRegion() : nullptr;
	const bool combined = region != nullptr;
	if (!region)
		region = owner ? owner->copyRegion() : nullptr;
	if (!region)
		return kIOReturnNotReady;
	IOReturn r = kIOReturnSuccess;
	IOLockLock(windowLock);
	if (!windowMap) {
		r = kIOReturnNotOpen;
	} else if (!rgfxWindowSpan(windowSize, offset, length) || offset + length > region->getLength()) {
		r = kIOReturnBadArgument;
	} else {
		const UInt64 start = mach_absolute_time();
		// The region's bytes over the window's: write-back as asked, which xnu's x86-64 pmap
		// maps uncached for a page it does not manage (PCD, the guest PAT's UC-) and its arm64
		// pmap maps Normal write-back, or write-combined (PCD and PAT, the guest PAT's WC).
		const IOOptionBits cache = combined ? kIOMapWriteCombineCache : kIOMapCopybackCache;
		IOMemoryMap *map = region->createMappingInTask(task, windowBase + offset,
		                                               kIOMapOverwrite | cache, offset, length);
		if (map) {
			// The mapping stays: an overwrite mapping is never taken down by its release.
			map->release();
			grants++;
			grantNs += nanosecondsSince(start);
		} else {
			r = kIOReturnNoMemory;
		}
	}
	IOLockUnlock(windowLock);
	region->release();
	if (r != kIOReturnSuccess && r != kIOReturnNotOpen)
		IOLog("RgfxAccelerator: context %u: 0x%llx bytes at 0x%llx not mapped in: 0x%x\n",
		      context, length, offset, r);
	return r;
}

void RgfxAcceleratorUserClient::unmapSpan(UInt64 offset, UInt64 length)
{
	if (!windowLock)
		return;
	IOLockLock(windowLock);
	if (windowMap && rgfxWindowSpan(windowSize, offset, length)) {
		const UInt64 start = mach_absolute_time();
		// The window's own bytes, read-only, back over the span.
		IOMemoryMap *map = windowMemory->createMappingInTask(
		    task, windowBase + offset, kIOMapOverwrite | kIOMapReadOnly, offset, length);
		if (map) {
			map->release();
			revokes++;
			revokeNs += nanosecondsSince(start);
		} else {
			// The span stays mapped, and the process could reach what goes there next: its
			// window goes instead, and the process reaches nothing of the region from now on.
			IOLog("RgfxAccelerator: context %u: 0x%llx bytes at 0x%llx not taken back: the "
			      "window goes\n",
			      context, length, offset);
			IOMemoryMap *window = windowMap;
			IOBufferMemoryDescriptor *memory = windowMemory;
			windowMap = nullptr;
			windowMemory = nullptr;
			windowBase = 0;
			windowSize = 0;
			OSSafeReleaseNULL(window);
			OSSafeReleaseNULL(memory);
		}
	}
	IOLockUnlock(windowLock);
}
