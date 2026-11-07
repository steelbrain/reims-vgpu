/*
 * RgfxGLUserClients: see RgfxGLUserClients.hpp. A GL window's surface and its drawables, with
 * their buffers as IOSurfaces in the surface's task.
 */

#include "RgfxGLUserClients.hpp"

#include "RgfxIOSurfaceKPI.hpp"
#include "RgfxTrace.hpp"

#include <IOKit/IOLib.h>
#include <IOKit/IOMemoryDescriptor.h>
#include <kern/task_ref.h>
#include <libkern/c++/OSDictionary.h>
#include <libkern/c++/OSNumber.h>
#include <libkern/c++/OSString.h>

namespace {
// The most window surfaces of the accelerator at once.
constexpr uint32_t kMaxSurfaces = 1024;

// The GL lock (IOGPU's surface lock) and what it guards: the listed surfaces, and IOSurfaceRoot,
// retained once found.
struct GLState {
	IOLock *lock;
	RgfxGLSurfaceUserClient *surfaces[kMaxSurfaces];
	uint32_t count;
	IOService *root;
};
GLState gl = {nullptr, {}, 0, nullptr};

IOLock *glLock()
{
	IOLock *lock = __atomic_load_n(&gl.lock, __ATOMIC_ACQUIRE);
	if (lock)
		return lock;
	IOLock *made = IOLockAlloc();
	if (!made)
		return nullptr;
	IOLock *expected = nullptr;
	if (__atomic_compare_exchange_n(&gl.lock, &expected, made, false, __ATOMIC_ACQ_REL,
	                                __ATOMIC_ACQUIRE))
		return made;
	IOLockFree(made);
	return expected;
}

// IOSurfaceRoot, found at the first ask. Under the GL lock.
IOService *rootLocked()
{
	if (!gl.root) {
		OSDictionary *matching = IOService::serviceMatching("IOSurfaceRoot");
		gl.root = matching ? IOService::copyMatchingService(matching) : nullptr;
		OSSafeReleaseNULL(matching);
	}
	return gl.root;
}

void setU32(OSDictionary *d, const char *key, uint32_t value)
{
	OSNumber *n = OSNumber::withNumber(value, 32);
	if (n) {
		d->setObject(key, n);
		n->release();
	}
}

// Logs a refusal of a GL client's, once per site and answer.
void logRefusal(const void *site, const char *what, IOReturn r, uint64_t detail)
{
	if (!rgfxFirstCalls(site, static_cast<uint32_t>(r), 1))
		return;
	RgfxCaller caller;
	IOLog("RGFX GL: %s from %s[%d] answers 0x%x (0x%llx)\n", what, caller.name, caller.pid, r,
	      detail);
}
} // namespace

// MARK: - The window surface

#define super IOUserClient
OSDefineMetaClassAndStructors(RgfxGLSurfaceUserClient, IOUserClient);

bool RgfxGLSurfaceUserClient::initWithTask(task_t owningTask, void *securityID, UInt32 type,
                                           OSDictionary *properties)
{
	task = nullptr;
	listed = false;
	dropped = false;
	drawableCount = 0;
	bzero(slots, sizeof slots);
	bzero(drawables, sizeof drawables);
	state = RgfxGLSurface();
	if (!super::initWithTask(owningTask, securityID, type, properties))
		return false;
	IOLock *lock = glLock();
	if (!lock)
		return false;
	// The surface's buffers are made in its task, from a drawable's call too, which may be
	// another task's: the task is held until the surface is freed, as IOGPU holds it.
	task = owningTask;
	task_reference(task);
	IOLockLock(lock);
	if (gl.count < kMaxSurfaces) {
		gl.surfaces[gl.count++] = this;
		listed = true;
	}
	IOLockUnlock(lock);
	if (!listed) {
		static char site;
		logRefusal(&site, "a window surface past the most at once", kIOReturnNoResources,
		           kMaxSurfaces);
	}
	return listed;
}

void RgfxGLSurfaceUserClient::stop(IOService *provider)
{
	dropSurface();
	super::stop(provider);
}

IOReturn RgfxGLSurfaceUserClient::clientClose()
{
	dropSurface();
	if (!isInactive())
		terminate();
	return kIOReturnSuccess;
}

void RgfxGLSurfaceUserClient::free()
{
	dropSurface();
	if (task)
		task_deallocate(task);
	task = nullptr;
	super::free();
}

void RgfxGLSurfaceUserClient::dropSurface()
{
	IOLock *lock = glLock();
	if (!lock)
		return;
	IOLockLock(lock);
	uint32_t detached = 0;
	if (listed) {
		for (uint32_t i = 0; i < gl.count; i++) {
			if (gl.surfaces[i] == this) {
				gl.surfaces[i] = gl.surfaces[--gl.count];
				gl.surfaces[gl.count] = nullptr;
				break;
			}
		}
		listed = false;
	}
	if (!dropped) {
		dropped = true;
		// Each drawable lets go of the surface (IOGPU's remove_surface): its lookups fail from
		// here, and its reference goes once the lock is let go.
		for (uint32_t i = 0; i < drawableCount; i++)
			drawables[i]->surface = nullptr;
		detached = drawableCount;
		drawableCount = 0;
		bzero(drawables, sizeof drawables);
		releaseSlotsLocked((1u << kRgfxGLSlots) - 1);
	}
	IOLockUnlock(lock);
	for (uint32_t i = 0; i < detached; i++)
		release();
}

RgfxGLSurfaceUserClient *RgfxGLSurfaceUserClient::findLocked(uint32_t id)
{
	for (uint32_t i = 0; i < gl.count; i++) {
		RgfxGLSurfaceUserClient *s = gl.surfaces[i];
		if (!s->dropped && s->state.id == id && id != kRgfxGLNoId)
			return s;
	}
	return nullptr;
}

IOReturn RgfxGLSurfaceUserClient::externalMethod(uint32_t selector, IOExternalMethodArguments *a,
                                                 IOExternalMethodDispatch *dispatch,
                                                 OSObject *target, void *reference)
{
	switch (selector) {
	case 7: // SetIDMode(id, mode)
		if (a->scalarInputCount != 2 || a->scalarOutputCount != 0 || a->structureInputSize ||
		    a->structureInputDescriptor || a->structureOutputSize || a->structureOutputDescriptor)
			return kIOReturnBadArgument;
		return setIDMode(a);
	case 9: // SetShape(options, framebuffer index) + the region
		if (a->scalarInputCount != 2 || a->scalarOutputCount != 0 || a->structureOutputSize ||
		    a->structureOutputDescriptor)
			return kIOReturnBadArgument;
		return setShape(a);
	default:
		break;
	}
	static char site;
	if (rgfxFirstCalls(&site, selector, 1)) {
		RgfxCaller caller;
		IOLog("RGFX GL: window surface method %u from %s[%d] is not supported (scalars %u/%u, "
		      "struct %u/%u)\n",
		      selector, caller.name, caller.pid, a->scalarInputCount, a->scalarOutputCount,
		      a->structureInputSize, a->structureOutputSize);
	}
	return kIOReturnUnsupported;
}

IOReturn RgfxGLSurfaceUserClient::setIDMode(IOExternalMethodArguments *a)
{
	const uint32_t id = static_cast<uint32_t>(a->scalarInput[0]);
	const uint32_t mode = static_cast<uint32_t>(a->scalarInput[1]);
	IOLock *lock = glLock();
	IOLockLock(lock);
	IOReturn r = kIOReturnNotAttached;
	if (!dropped) {
		r = rgfxGLSetIDMode(state, id, mode, drawableCount);
		if (r == kIOReturnSuccess) {
			resetRequestsLocked();
			pruneLocked();
			reshapeLocked();
		}
	}
	IOLockUnlock(lock);
	if (r != kIOReturnSuccess) {
		static char site;
		logRefusal(&site, "SetIDMode", r, (static_cast<uint64_t>(id) << 32) | mode);
	}
	return r;
}

IOReturn RgfxGLSurfaceUserClient::setShape(IOExternalMethodArguments *a)
{
	// The region: inline, or, past what a call carries inline, through its descriptor, of which
	// the rule reads the header alone (its count, its bounds) and the length.
	uint8_t header[12] = {0};
	const uint8_t *region = static_cast<const uint8_t *>(a->structureInput);
	uint64_t length = a->structureInputSize;
	if (IOMemoryDescriptor *md = a->structureInputDescriptor) {
		length = md->getLength();
		region = nullptr;
		if (length >= sizeof header && md->prepare(kIODirectionOut) == kIOReturnSuccess) {
			if (md->readBytes(0, header, sizeof header) == sizeof header)
				region = header;
			md->complete(kIODirectionOut);
		}
	}
	const uint32_t options = static_cast<uint32_t>(a->scalarInput[0]);
	const uint32_t framebufferIndex = static_cast<uint32_t>(a->scalarInput[1]);
	IOLock *lock = glLock();
	IOLockLock(lock);
	IOReturn r = dropped ? kIOReturnNotAttached
	                     : rgfxGLSetShape(state, options, framebufferIndex, region, length);
	IOLockUnlock(lock);
	if (r != kIOReturnSuccess) {
		static char site;
		logRefusal(&site, "SetShape", r, (static_cast<uint64_t>(options) << 32) | framebufferIndex);
	}
	return r;
}

void RgfxGLSurfaceUserClient::resetRequestsLocked()
{
	uint32_t requests = 0;
	for (uint32_t i = 0; i < drawableCount; i++)
		requests |= drawables[i]->requested;
	rgfxGLResetRequests(state, requests);
}

void RgfxGLSurfaceUserClient::pruneLocked()
{
	releaseSlotsLocked(rgfxGLPrune(state, existingLocked()));
}

void RgfxGLSurfaceUserClient::reshapeLocked()
{
	const uint32_t remake = rgfxGLReshape(state);
	releaseSlotsLocked(remake);
	for (uint32_t slot = 0; slot < kRgfxGLSlots; slot++)
		if (remake & (1u << slot))
			makeSlotLocked(slot);
}

uint32_t RgfxGLSurfaceUserClient::existingLocked() const
{
	uint32_t mask = 0;
	for (uint32_t slot = 0; slot < kRgfxGLSlots; slot++)
		if (slots[slot])
			mask |= 1u << slot;
	return mask;
}

void RgfxGLSurfaceUserClient::releaseSlotsLocked(uint32_t mask)
{
	for (uint32_t slot = 0; slot < kRgfxGLSlots; slot++) {
		if (!(mask & (1u << slot)) || !slots[slot])
			continue;
		rgfxIOSurfaceRootReleaseAsync(gl.root, slots[slot]);
		slots[slot] = nullptr;
	}
}

void RgfxGLSurfaceUserClient::makeSlotLocked(uint32_t slot)
{
	RgfxGLBuffer b;
	if (!rgfxGLBufferOf(slot, state.mode, state.width, state.height, b))
		return;
	IOService *root = rootLocked();
	OSDictionary *d = root ? OSDictionary::withCapacity(4) : nullptr;
	OSObject *made = nullptr;
	if (d) {
		// IOSurface.kext's kIOSurface* keys, as IOGPU sets them.
		if (b.hasFormat)
			setU32(d, "IOSurfacePixelFormat", b.format);
		setU32(d, "IOSurfaceBytesPerElement", b.bytesPerElement);
		setU32(d, "IOSurfaceWidth", b.width);
		setU32(d, "IOSurfaceHeight", b.height);
		made = rgfxIOSurfaceRootCreate(root, task, d);
		d->release();
	}
	if (made) {
		rgfxIOSurfaceClientRetain(made);
		char name[0x80];
		snprintf(name, sizeof name, "IOGPUSurfaceMTL sid %08x buffer %u", state.id, slot);
		if (OSString *s = OSString::withCString(name)) {
			uint32_t error = 0;
			rgfxIOSurfaceSetValue(made, "IOSurfaceName", s, &error);
			s->release();
		}
	} else {
		static char site;
		logRefusal(&site, "a window surface's buffer", kIOReturnNoMemory,
		           (static_cast<uint64_t>(b.width) << 32) | b.height);
	}
	if (slots[slot])
		rgfxIOSurfaceRootReleaseAsync(gl.root, slots[slot]);
	slots[slot] = made;
}

// MARK: - The drawable

#undef super
#define super IOUserClient
OSDefineMetaClassAndStructors(RgfxGLDrawableUserClient, IOUserClient);

namespace {
// IOGPU's six methods, with its counts and sizes.
const IOExternalMethodDispatch kDrawableMethods[] = {
    {RgfxGLDrawableUserClient::sSetSurface, 0, sizeof(RgfxGLSetSurface), 0, 0},
    {RgfxGLDrawableUserClient::sSetSurfaceGetConfig, 0, sizeof(RgfxGLSetSurface), 0,
     sizeof(RgfxGLConfigStatus)},
    {RgfxGLDrawableUserClient::sSetSwapRect, 4, 0, 0, 0},
    {RgfxGLDrawableUserClient::sSetSwapInterval, 2, 0, 0, 0},
    {RgfxGLDrawableUserClient::sSignalSharedEvent, 3, 0, 0, 0},
    {RgfxGLDrawableUserClient::sLookup, 1, 0, 1, 0},
};
} // namespace

bool RgfxGLDrawableUserClient::initWithTask(task_t owningTask, void *securityID, UInt32 type,
                                            OSDictionary *properties)
{
	task = owningTask;
	surface = nullptr;
	requested = 0;
	face = 0;
	level = 0;
	bzero(swapRect, sizeof swapRect);
	bzero(swapInterval, sizeof swapInterval);
	if (!super::initWithTask(owningTask, securityID, type, properties))
		return false;
	return glLock() != nullptr;
}

IOReturn RgfxGLDrawableUserClient::clientClose()
{
	IOLock *lock = glLock();
	RgfxGLSurfaceUserClient *held = nullptr;
	IOLockLock(lock);
	detachLocked(&held, true);
	IOLockUnlock(lock);
	OSSafeReleaseNULL(held);
	if (!isInactive())
		terminate();
	return kIOReturnSuccess;
}

void RgfxGLDrawableUserClient::free()
{
	if (IOLock *lock = glLock()) {
		RgfxGLSurfaceUserClient *held = nullptr;
		IOLockLock(lock);
		detachLocked(&held, true);
		IOLockUnlock(lock);
		OSSafeReleaseNULL(held);
	}
	super::free();
}

void RgfxGLDrawableUserClient::detachLocked(RgfxGLSurfaceUserClient **held, bool prune)
{
	RgfxGLSurfaceUserClient *s = surface;
	if (!s)
		return;
	for (uint32_t i = 0; i < s->drawableCount; i++) {
		if (s->drawables[i] == this) {
			s->drawables[i] = s->drawables[--s->drawableCount];
			s->drawables[s->drawableCount] = nullptr;
			break;
		}
	}
	s->resetRequestsLocked();
	if (prune)
		s->pruneLocked();
	surface = nullptr;
	*held = s;
}

IOReturn RgfxGLDrawableUserClient::externalMethod(uint32_t selector, IOExternalMethodArguments *a,
                                                  IOExternalMethodDispatch *dispatch,
                                                  OSObject *target, void *reference)
{
	if (selector < sizeof kDrawableMethods / sizeof kDrawableMethods[0]) {
		dispatch = const_cast<IOExternalMethodDispatch *>(&kDrawableMethods[selector]);
		target = this;
		reference = nullptr;
	}
	return super::externalMethod(selector, a, dispatch, target, reference);
}

IOReturn RgfxGLDrawableUserClient::sSetSurface(OSObject *target, void *, IOExternalMethodArguments *a)
{
	RgfxGLSetSurface in;
	memcpy(&in, a->structureInput, sizeof in);
	return static_cast<RgfxGLDrawableUserClient *>(target)->setSurface(in);
}

IOReturn RgfxGLDrawableUserClient::sSetSurfaceGetConfig(OSObject *target, void *,
                                                       IOExternalMethodArguments *a)
{
	RgfxGLSetSurface in;
	memcpy(&in, a->structureInput, sizeof in);
	RgfxGLConfigStatus out;
	bzero(&out, sizeof out);
	IOReturn r = static_cast<RgfxGLDrawableUserClient *>(target)->setSurfaceGetConfig(in, out);
	if (r == kIOReturnSuccess)
		memcpy(a->structureOutput, &out, sizeof out);
	return r;
}

IOReturn RgfxGLDrawableUserClient::sSetSwapRect(OSObject *target, void *, IOExternalMethodArguments *a)
{
	RgfxGLDrawableUserClient *d = static_cast<RgfxGLDrawableUserClient *>(target);
	for (int i = 0; i < 4; i++)
		d->swapRect[i] = static_cast<int16_t>(a->scalarInput[i]);
	return kIOReturnSuccess;
}

IOReturn RgfxGLDrawableUserClient::sSetSwapInterval(OSObject *target, void *,
                                                   IOExternalMethodArguments *a)
{
	RgfxGLDrawableUserClient *d = static_cast<RgfxGLDrawableUserClient *>(target);
	for (int i = 0; i < 2; i++)
		d->swapInterval[i] = static_cast<int16_t>(a->scalarInput[i]);
	return kIOReturnSuccess;
}

IOReturn RgfxGLDrawableUserClient::sSignalSharedEvent(OSObject *, void *, IOExternalMethodArguments *a)
{
	static char site;
	logRefusal(&site, "a drawable's shared event", kIOReturnUnsupported, a->scalarInput[0]);
	return kIOReturnUnsupported;
}

IOReturn RgfxGLDrawableUserClient::sLookup(OSObject *target, void *, IOExternalMethodArguments *a)
{
	RgfxGLDrawableUserClient *d = static_cast<RgfxGLDrawableUserClient *>(target);
	const uint64_t index = a->scalarInput[0];
	IOLock *lock = glLock();
	IOLockLock(lock);
	IOReturn r = kIOReturnBadArgument;
	uint32_t name = 0;
	const int32_t slot = rgfxGLSlotOfIndex(index);
	if (d->surface && slot >= 0 && d->surface->slots[slot])
		r = rgfxIOSurfaceRootPortName(gl.root, d->task, d->surface->slots[slot], &name);
	IOLockUnlock(lock);
	if (r == kIOReturnSuccess) {
		a->scalarOutput[0] = name;
	} else {
		static char site;
		logRefusal(&site, "a drawable's lookup", r, index);
	}
	return r;
}

IOReturn RgfxGLDrawableUserClient::setSurface(const RgfxGLSetSurface &in)
{
	if (!rgfxGLDrawableModeTaken(in.mode))
		return kIOReturnBadArgument;
	IOLock *lock = glLock();
	IOLockLock(lock);
	RgfxGLSurfaceUserClient *found = nullptr;
	IOReturn r = kIOReturnSuccess;
	if (in.surfaceId) {
		found = RgfxGLSurfaceUserClient::findLocked(in.surfaceId);
		if (!found || in.face || in.level ||
		    (found != surface && found->drawableCount == RgfxGLSurfaceUserClient::kMaxDrawables))
			r = kIOReturnBadArgument;
	}
	RgfxGLSurfaceUserClient *held = nullptr;
	if (r == kIOReturnSuccess) {
		// Off the surface it was on: the surface's request bits follow, and its buffers too
		// where it is not the one it attaches to again.
		detachLocked(&held, surface != found);
		face = in.face;
		level = in.level;
		if (found) {
			const uint64_t mode = rgfxGLDrawableModeOn(in.mode, found->state.mode);
			found->resetRequestsLocked();
			found->state.mode = rgfxGLSurfaceModeWith(found->state.mode, mode, found->drawableCount);
			requested = rgfxGLDrawableRequestBits(mode);
			found->drawables[found->drawableCount++] = this;
			found->state.requested |= requested;
			found->retain();
			surface = found;
			found->pruneLocked();
			found->reshapeLocked();
		}
	}
	IOLockUnlock(lock);
	OSSafeReleaseNULL(held);
	if (r != kIOReturnSuccess) {
		static char site;
		logRefusal(&site, "a drawable's set surface", r,
		           (static_cast<uint64_t>(in.surfaceId) << 32) | (in.face << 16) | in.level);
	}
	return r;
}

IOReturn RgfxGLDrawableUserClient::setSurfaceGetConfig(const RgfxGLSetSurface &in,
                                                      RgfxGLConfigStatus &out)
{
	IOLock *lock = glLock();
	// A colour depth named, other than the surface's, is refused before the set surface.
	if (in.surfaceId && in.colorDepthMode != 0xffffffff) {
		IOLockLock(lock);
		RgfxGLSurfaceUserClient *s = RgfxGLSurfaceUserClient::findLocked(in.surfaceId);
		const bool differs = s && (s->state.mode & 0xf) != (in.colorDepthMode & 0xf);
		IOLockUnlock(lock);
		if (differs) {
			static char site;
			logRefusal(&site, "a drawable's colour depth", kIOReturnError, in.colorDepthMode);
			return kIOReturnError;
		}
	}
	IOReturn r = setSurface(in);
	if (r != kIOReturnSuccess)
		return r;
	if (in.fixedSource) {
		static char site;
		logRefusal(&site, "a drawable's scaled source", kIOReturnUnsupported, in.scaleOptions);
		return kIOReturnUnsupported;
	}
	IOLockLock(lock);
	if (surface) {
		// The drawable's size is the surface's: no scaled source, and level 0.
		out.mode = surface->state.mode;
		out.width = out.surfaceWidth = surface->state.width;
		out.height = out.surfaceHeight = surface->state.height;
	} else {
		r = kIOReturnError;
	}
	IOLockUnlock(lock);
	return r;
}
