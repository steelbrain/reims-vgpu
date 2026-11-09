/*
 * RgfxAccelerator's IOAccel surfaces: the surface clients (RgfxSurfaceUserClient) that have an
 * id, by id, and the scanout spans our own user clients may name because a surface of their
 * process is drawn into the scanout.
 *
 * WindowServer draws a display into the IOAccel surface on its framebuffer; with the Metal
 * compositor, through a texture the bundle makes over that surface's memory, asked for by the
 * surface's id (-newTextureWithSurface:buffer:). For a surface over the scanout, that memory is
 * the scanout's span of the region, outside every process's blocks, so the process's messages
 * could not name it: grantSurfaceScanout lets the client of a process that holds the surface name
 * the span (grantedRange, which pinRange asks), until the client's context is released or the
 * surface goes, and tells the device the span is the client's context's (BLOCK_GRANT), as the
 * device checks the spans of the messages on the queues' rings, which bypass the kext.
 *
 * The device keeps that grant until the context's close, or until another context is granted
 * the span: nothing frees the scanout or gives it to a process as a block, so the grant lets the
 * context draw into the display and reach nothing else. The span is in the client's window while
 * the kext's grant lasts: mapped in at the grant, under the lock, and taken back where the grant
 * goes or moves (the surface goes, or a mode change moves the scanout).
 *
 * Only the scanout's owner, the task that holds the framebuffer's server connection, is granted
 * it (RgfxScanoutOwner.hpp); any other task's ask is refused, whatever surface it made. When the
 * owner moves to a new connection's task, every other task's grant goes (scanoutOwnerMoved).
 *
 * Our own code; it names only SDK-public IOKit calls.
 */

#include "RgfxAccelerator.hpp"
#include "RgfxAcceleratorUserClient.hpp"
#include "RgfxSurfaceUserClient.hpp"
#include "ReimsGraphics.hpp"
#include "RgfxScanoutOwner.hpp"
#include "RgfxWindowSpans.hpp"
#include "RgfxTrace.hpp"

#include <IOKit/IOLib.h>

namespace {
// The most surfaces with an id at once, and the most scanout grants.
constexpr UInt32 kMaxSurfaces = 256;
constexpr UInt32 kMaxGrants = 8;

struct Grant {
	RgfxAcceleratorUserClient *client;
	UInt32 surfaceId;
	UInt64 offset;
	UInt64 length;
	// The task the grant was made for, the scanout's owner then.
	const void *task;
};

// Under lock: the surfaces (each retained) and their ids, and the grants. One initialised
// object, made at the first surface's id.
struct SurfaceState {
	IOLock *lock;
	UInt32 count;
	RgfxSurfaceUserClient *surfaces[kMaxSurfaces];
	UInt32 ids[kMaxSurfaces];
	Grant grants[kMaxGrants];
};
SurfaceState surfaceState = {nullptr, 0, {}, {}, {}};

IOLock *surfaceLock()
{
	IOLock *lock = __atomic_load_n(&surfaceState.lock, __ATOMIC_ACQUIRE);
	if (lock)
		return lock;
	IOLock *made = IOLockAlloc();
	if (!made)
		return nullptr;
	IOLock *expected = nullptr;
	if (__atomic_compare_exchange_n(&surfaceState.lock, &expected, made, false,
	                                __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE))
		return made;
	IOLockFree(made);
	return expected;
}

// Drops grant `g`, its span out of its client's window first. Under the lock, which the client's
// release takes before its window goes.
void dropGrant(Grant &g)
{
	g.client->unmapSpan(g.offset, g.length);
	g = {};
}

// Drops the grants made for surface `id`. Under the lock.
void dropGrantsOf(UInt32 id)
{
	for (UInt32 i = 0; i < kMaxGrants; i++) {
		if (surfaceState.grants[i].client && surfaceState.grants[i].surfaceId == id)
			dropGrant(surfaceState.grants[i]);
	}
}
} // namespace

IOReturn RgfxAccelerator::registerSurface(RgfxSurfaceUserClient *surface, UInt32 surfaceId)
{
	IOLock *lock = surfaceLock();
	if (!lock)
		return kIOReturnNoMemory;
	IOLockLock(lock);
	UInt32 mine = kMaxSurfaces;
	for (UInt32 i = 0; i < surfaceState.count; i++) {
		if (surfaceState.surfaces[i] == surface) {
			mine = i;
		} else if (surfaceState.ids[i] == surfaceId) {
			IOLockUnlock(lock);
			return kIOReturnExclusiveAccess;
		}
	}
	if (mine != kMaxSurfaces) {
		if (surfaceState.ids[mine] != surfaceId)
			dropGrantsOf(surfaceState.ids[mine]);
		surfaceState.ids[mine] = surfaceId;
		IOLockUnlock(lock);
		return kIOReturnSuccess;
	}
	if (surfaceState.count == kMaxSurfaces) {
		IOLockUnlock(lock);
		return kIOReturnNoResources;
	}
	surface->retain();
	surfaceState.surfaces[surfaceState.count] = surface;
	surfaceState.ids[surfaceState.count] = surfaceId;
	surfaceState.count++;
	IOLockUnlock(lock);
	return kIOReturnSuccess;
}

void RgfxAccelerator::unregisterSurface(RgfxSurfaceUserClient *surface)
{
	IOLock *lock = __atomic_load_n(&surfaceState.lock, __ATOMIC_ACQUIRE);
	if (!lock)
		return;
	RgfxSurfaceUserClient *gone = nullptr;
	IOLockLock(lock);
	for (UInt32 i = 0; i < surfaceState.count; i++) {
		if (surfaceState.surfaces[i] != surface)
			continue;
		dropGrantsOf(surfaceState.ids[i]);
		gone = surface;
		surfaceState.count--;
		surfaceState.surfaces[i] = surfaceState.surfaces[surfaceState.count];
		surfaceState.ids[i] = surfaceState.ids[surfaceState.count];
		break;
	}
	IOLockUnlock(lock);
	OSSafeReleaseNULL(gone);
}

bool RgfxAccelerator::ownsScanout(task_t task)
{
	ReimsGraphics *fb = copyFramebuffer();
	if (!fb)
		return false;
	bool owns = fb->ownsScanout(task);
	fb->release();
	return owns;
}

bool RgfxAccelerator::currentScanout(Scanout *scanout)
{
	ReimsGraphics *fb = copyFramebuffer();
	if (!fb)
		return false;
	bool ok = fb->currentScanout(&scanout->width, &scanout->height, &scanout->rowBytes,
	                             &scanout->offset);
	fb->release();
	if (!ok)
		return false;
	// The span a window maps of it, whole window pages: 16 KiB on arm64 (RgfxWindowSpans.hpp).
	scanout->length =
	    rgfxWindowSpanLength(static_cast<UInt64>(scanout->rowBytes) * scanout->height);
	return true;
}

IOReturn RgfxAccelerator::grantSurfaceScanout(RgfxAcceleratorUserClient *client, UInt32 context,
                                              task_t task, UInt32 surfaceId, Scanout *scanout)
{
	// The display's pixels are its owner's alone (RgfxScanoutOwner.hpp).
	if (!ownsScanout(task)) {
		static char site;
		RgfxCaller caller;
		if (rgfxFirstCalls(&site, static_cast<UInt64>(caller.pid), 2))
			IOLog("RGFX surface: the scanout is refused to %s[%d], which does not own it\n",
			      caller.name, caller.pid);
		return kIOReturnNotPermitted;
	}
	if (!currentScanout(scanout))
		return kIOReturnNotReady;
	IOLock *lock = __atomic_load_n(&surfaceState.lock, __ATOMIC_ACQUIRE);
	if (!lock)
		return kIOReturnNotFound;
	IOLockLock(lock);
	RgfxSurfaceUserClient *surface = nullptr;
	for (UInt32 i = 0; i < surfaceState.count; i++) {
		if (surfaceState.ids[i] == surfaceId)
			surface = surfaceState.surfaces[i];
	}
	IOReturn r = kIOReturnSuccess;
	if (!surface || surface->owningTask() != task)
		r = kIOReturnNotFound;
	else if (!surface->isScanout(scanout->width, scanout->height))
		r = kIOReturnUnsupported;
	if (r == kIOReturnSuccess) {
		// The client's grant, made again for this surface and the scanout as it is now.
		UInt32 slot = kMaxGrants;
		for (UInt32 i = 0; i < kMaxGrants; i++) {
			if (surfaceState.grants[i].client == client) {
				slot = i;
				break;
			}
			if (!surfaceState.grants[i].client && slot == kMaxGrants)
				slot = i;
		}
		if (slot == kMaxGrants) {
			r = kIOReturnNoResources;
		} else {
			// In the client's window as the scanout is now; where it moved, the old span is
			// taken back.
			Grant &g = surfaceState.grants[slot];
			if (g.client && (g.offset != scanout->offset || g.length != scanout->length))
				dropGrant(g);
			r = client->mapSpan(scanout->offset, scanout->length);
			if (r == kIOReturnSuccess)
				g = {client, surfaceId, scanout->offset, scanout->length, task};
			else if (g.client)
				dropGrant(g);
			else
				client->unmapSpan(scanout->offset, scanout->length);
		}
	}
	IOLockUnlock(lock);
	if (r != kIOReturnSuccess)
		return r;
	// Told at each grant, not only a new one: another context of the process may have been
	// granted the span since, and the device keeps one holder.
	RgfxBlockGrant grant = {};
	grant.context = context;
	grant.offset = scanout->offset;
	grant.size = scanout->length;
	r = postLifecycle(RGFX_CMD_BLOCK_GRANT, &grant, sizeof(grant), 0, nullptr, 0, nullptr,
	                  nullptr, 0);
	if (r != kIOReturnSuccess)
		revokeGrants(client);
	return r;
}

void RgfxAccelerator::revokeGrants(const void *client)
{
	IOLock *lock = __atomic_load_n(&surfaceState.lock, __ATOMIC_ACQUIRE);
	if (!lock)
		return;
	IOLockLock(lock);
	for (UInt32 i = 0; i < kMaxGrants; i++) {
		if (surfaceState.grants[i].client == client)
			dropGrant(surfaceState.grants[i]);
	}
	IOLockUnlock(lock);
}

void RgfxAccelerator::scanoutOwnerMoved(task_t owner)
{
	UInt32 grants = 0, maps = 0;
	if (IOLock *lock = __atomic_load_n(&surfaceState.lock, __ATOMIC_ACQUIRE)) {
		IOLockLock(lock);
		for (UInt32 i = 0; i < kMaxGrants; i++) {
			Grant &g = surfaceState.grants[i];
			if (g.client && !rgfxKeepsScanout(g.task, owner)) {
				dropGrant(g);
				grants++;
			}
		}
		IOLockUnlock(lock);
	}
	// Every surface client, those with no id among them, attaches to the accelerator: the other
	// tasks' are gathered (retained) first, so none is dropped while the clients are walked.
	OSArray *others = OSArray::withCapacity(8);
	OSIterator *clients = others ? getClientIterator() : nullptr;
	if (clients) {
		do {
			clients->reset();
			others->flushCollection();
			while (OSObject *o = clients->getNextObject()) {
				RgfxSurfaceUserClient *surface = OSDynamicCast(RgfxSurfaceUserClient, o);
				if (surface && !rgfxKeepsScanout(surface->owningTask(), owner))
					others->setObject(surface);
			}
		} while (!clients->isValid());
		clients->release();
	}
	for (unsigned i = 0; others && i < others->getCount(); i++) {
		if (static_cast<RgfxSurfaceUserClient *>(others->getObject(i))->dropScanoutMap())
			maps++;
	}
	OSSafeReleaseNULL(others);
	if (grants || maps)
		IOLog("RGFX surface: the scanout's owner moved: %u grants and %u mappings of the scanout "
		      "in other tasks were taken back\n",
		      grants, maps);
}

bool RgfxAccelerator::grantedRange(const void *client, UInt64 offset, UInt64 length)
{
	IOLock *lock = __atomic_load_n(&surfaceState.lock, __ATOMIC_ACQUIRE);
	if (!lock)
		return false;
	bool granted = false;
	IOLockLock(lock);
	for (UInt32 i = 0; i < kMaxGrants && !granted; i++) {
		const Grant &g = surfaceState.grants[i];
		granted = g.client == client && offset >= g.offset && offset - g.offset <= g.length &&
		          length <= g.length - (offset - g.offset);
	}
	IOLockUnlock(lock);
	return granted;
}
