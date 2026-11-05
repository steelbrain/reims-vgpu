/*
 * RgfxDisplayPipeUserClient: see RgfxDisplayPipeUserClient.hpp. The display pipe's selectors,
 * presenting each transaction's IOSurface through the framebuffer's control ring.
 */

#include "RgfxDisplayPipeUserClient.hpp"

#include "ReimsGraphics.hpp"
#include "RgfxAccelerator.hpp"
#include "RgfxPipeColor.hpp"
#include "RgfxPipeNotifications.hpp"

#include <IOKit/IOLib.h>
#include <IOKit/IOMemoryDescriptor.h>
#include <kern/clock.h>
#include <libkern/c++/OSSerialize.h>

#define super IOUserClient
OSDefineMetaClassAndStructors(RgfxDisplayPipeUserClient, IOUserClient);

namespace {

// The argument structures' sizes.
constexpr UInt32 kNotifySize = 0x18;
constexpr UInt32 kTransactionSize = 0x118;
constexpr UInt32 kWaitSize = 0xc;
constexpr UInt32 kScalerSize = 24;

// Inside a notification registration: its kind.
constexpr UInt32 kNotifyKind = 0x10;

// Inside a transaction: its dirty mask, the time it asks to be shown at (where the mask has
// kDirtyTimestamp; read only for the log), its id and options, plane 0 (0x40 bytes), and the
// cookie its completion carries back.
constexpr UInt32 kTxnDirty = 0x0;
constexpr UInt32 kTxnTimestamp = 0x8;
constexpr UInt64 kDirtyTimestamp = 0x100;
constexpr UInt32 kTxnId = 0x10;
constexpr UInt32 kTxnOptions = 0x14;
constexpr UInt32 kTxnPlane0 = 0x18;
constexpr UInt32 kTxnCookie = 0x100;
// Inside a plane: its type (1, an IOSurface) and the surface's global ID.
constexpr UInt32 kPlaneType = 0x0;
constexpr UInt32 kPlaneSurface = 0x8;
constexpr UInt32 kPlaneIOSurface = 1;

// Inside a notification registration: the callback and refcon a VBL request names.
constexpr UInt32 kNotifyCallback = 0x0;
constexpr UInt32 kNotifyRefcon = 0x8;

// The messages back, a transaction's completion and a VBL: RgfxPipeNotifications.hpp.

// The entitlement an association needs: WindowServer has it.
const char *const kEntitlement = "com.apple.private.hid.client.event-monitor";

// The accelerator id the association returns: our one accelerator.
constexpr UInt64 kAcceleratorId = 1;

// The calls logged in full (sizes, the registrations' and transactions' fields); after that,
// every 600th transaction.
constexpr UInt32 kLogFirst = 48;
constexpr UInt32 kLogEvery = 600;
// The transactions with no cookie logged, per client.
constexpr UInt32 kLogUnnotified = 8;
// The colour calls logged in full (header fields, first and last entries), per client; after
// that, every kLogEvery-th.
constexpr UInt32 kLogColor = 32;

inline UInt32 rd32(const UInt8 *p, UInt32 off)
{
	UInt32 v;
	memcpy(&v, p + off, sizeof(v));
	return v;
}

inline UInt64 rd64(const UInt8 *p, UInt32 off)
{
	UInt64 v;
	memcpy(&v, p + off, sizeof(v));
	return v;
}

// The structure input of `a`, at least `size` bytes, inline or through its descriptor (copied
// into `buf`); nullptr where it is shorter, or its memory cannot be wired. The descriptor is over
// the caller's memory, not wired: it is prepared for the read and completed after it.
const UInt8 *structIn(IOExternalMethodArguments *a, UInt8 *buf, UInt32 size)
{
	if (a->structureInput && a->structureInputSize >= size)
		return static_cast<const UInt8 *>(a->structureInput);
	IOMemoryDescriptor *desc = a->structureInputDescriptor;
	if (!desc || desc->getLength() < size || desc->prepare(kIODirectionOut) != kIOReturnSuccess)
		return nullptr;
	const bool read = desc->readBytes(0, buf, size) == size;
	desc->complete(kIODirectionOut);
	return read ? buf : nullptr;
}

} // namespace

// MARK: - Lifecycle

bool RgfxDisplayPipeUserClient::initWithTask(task_t owningTask, void *securityID, UInt32 type,
                                             OSDictionary *properties)
{
	if (!super::initWithTask(owningTask, securityID, type, properties))
		return false;
	task = owningTask;
	owner = nullptr;
	framebuffer = nullptr;
	associated = false;
	lock = IOLockAlloc();
	submitLock = IOLockAlloc();
	registered = false;
	vblWaiting = 0;
	nextTransaction = 0;
	begun = 0;
	completed = 0;
	logged = 0;
	unnotifiedLogged = 0;
	transactions = Transactions();
	displayColor = false;
	stagedFor = 0;
	for (GammaStage &g : staged)
		g = {};
	colorLogged = 0;
	return lock != nullptr && submitLock != nullptr;
}

bool RgfxDisplayPipeUserClient::start(IOService *provider)
{
	owner = OSDynamicCast(RgfxAccelerator, provider);
	if (!owner)
		return false;
	return super::start(provider);
}

void RgfxDisplayPipeUserClient::stop(IOService *provider)
{
	releaseRegistrations();
	super::stop(provider);
}

IOReturn RgfxDisplayPipeUserClient::clientClose()
{
	releaseRegistrations();
	if (!isInactive())
		terminate();
	return kIOReturnSuccess;
}

void RgfxDisplayPipeUserClient::free()
{
	dropStaged();
	OSSafeReleaseNULL(framebuffer);
	if (lock)
		IOLockFree(lock);
	lock = nullptr;
	if (submitLock)
		IOLockFree(submitLock);
	submitLock = nullptr;
	super::free();
}

// Stops the VBL hook and gives back the wake ports the registrations kept.
void RgfxDisplayPipeUserClient::releaseRegistrations()
{
	if (framebuffer)
		framebuffer->removePipe(this);
	if (!lock)
		return;
	IOLockLock(lock);
	if (registered) {
		releaseAsyncReference64(transactionRef);
		registered = false;
	}
	for (UInt32 i = 0; i < vblWaiting; i++)
		releaseAsyncReference64(vblRequests[i].reference);
	vblWaiting = 0;
	IOLockUnlock(lock);
}

// Lets go of the gamma tables staged, with `lock` not held: free, before the client exists, and
// sBegin, after it moved them out under the lock.
void RgfxDisplayPipeUserClient::dropStaged()
{
	for (GammaStage &g : staged) {
		if (g.entries)
			IOFree(g.entries, static_cast<vm_size_t>(g.count) * kRgfxGammaEntrySize);
		g = {};
	}
	stagedFor = 0;
}

// MARK: - Dispatch

IOReturn RgfxDisplayPipeUserClient::externalMethod(uint32_t selector,
                                                   IOExternalMethodArguments *arguments,
                                                   IOExternalMethodDispatch *dispatch,
                                                   OSObject *target, void *reference)
{
	static const IOExternalMethodDispatch methods[10] = {
		/* 0 */ { &RgfxDisplayPipeUserClient::sAssociate, 1, 0, 1, 0 },
		/* 1 */ { &RgfxDisplayPipeUserClient::sModeScaler, 0, 0, 0, kScalerSize },
		/* 2 */ { &RgfxDisplayPipeUserClient::sCapabilities, 0, 0, 0, kIOUCVariableStructureSize },
		/* 3 */ { &RgfxDisplayPipeUserClient::sRequestNotify, 0, kNotifySize, 0, 0 },
		/* 4 */ { &RgfxDisplayPipeUserClient::sBegin, 0, 0, 1, 0 },
		/* 5 */ { nullptr, 0, 0, 0, 0 },
		/* 6 */ { nullptr, 0, 0, 0, 0 },
		/* 7 */ { &RgfxDisplayPipeUserClient::sSeven, 0, 0, 0, 0 },
		/* 8 */ { &RgfxDisplayPipeUserClient::sSubmit, 0, kTransactionSize, 0, 0 },
		/* 9 */ { &RgfxDisplayPipeUserClient::sWait, 0, kWaitSize, 0, 0 },
	};
	if (logged < kLogFirst) {
		logged++;
		IOLog("RGFX pipe: selector %u scalarIn %u structIn %u scalarOut %u structOut %u "
		      "outDesc %u async %u\n",
		      selector, arguments->scalarInputCount, arguments->structureInputSize,
		      arguments->scalarOutputCount, arguments->structureOutputSize,
		      arguments->structureOutputDescriptor ? arguments->structureOutputDescriptorSize : 0,
		      arguments->asyncWakePort != MACH_PORT_NULL ? 1u : 0u);
	}
	// The gamma selectors, where the host shows the display's colour (RgfxPipeColor.hpp).
	static const IOExternalMethodDispatch gamma = {
		&RgfxDisplayPipeUserClient::sGamma, 0, kIOUCVariableStructureSize, 0, 0
	};
	const bool color = rgfxGammaSlot(selector) >= 0 && displayColor;
	if (!color && (selector >= 10 || !methods[selector].function)) {
		IOLog("RGFX pipe: selector %u is not supported\n", selector);
		return kIOReturnUnsupported;
	}
	dispatch = const_cast<IOExternalMethodDispatch *>(color ? &gamma : &methods[selector]);
	return super::externalMethod(selector, arguments, dispatch, this, reference);
}

// MARK: - The selectors

IOReturn RgfxDisplayPipeUserClient::sAssociate(OSObject *target, void *,
                                               IOExternalMethodArguments *a)
{
	RgfxDisplayPipeUserClient *self = OSDynamicCast(RgfxDisplayPipeUserClient, target);
	if (!self || !self->owner)
		return kIOReturnNotAttached;
	UInt32 index = static_cast<UInt32>(a->scalarInput[0]);
	// Our one display is pipe 0. A framebuffer that IOPresentment found without IOAccelIndex
	// comes as -1: it is the same display, the only one there is.
	if (index != 0 && index != 0xffffffffu) {
		IOLog("RGFX pipe: associate: no pipe %u\n", index);
		return kIOReturnBadArgument;
	}
	OSObject *entitled = copyClientEntitlement(self->task, kEntitlement);
	bool allowed = entitled && entitled != kOSBooleanFalse;
	OSSafeReleaseNULL(entitled);
	if (!allowed) {
		IOLog("RGFX pipe: associate: the client lacks %s\n", kEntitlement);
		return kIOReturnExclusiveAccess;
	}
	ReimsGraphics *fb = self->owner->copyFramebuffer();
	if (!fb) {
		IOLog("RGFX pipe: associate: no framebuffer yet\n");
		return kIOReturnNotReady;
	}
	OSSafeReleaseNULL(self->framebuffer);
	self->framebuffer = fb;
	self->displayColor = fb->displayColor();
	self->associated = true;
	if (!fb->addPipe(self, &RgfxDisplayPipeUserClient::vblankHook,
	                 &RgfxDisplayPipeUserClient::presentHook)) {
		self->associated = false;
		IOLog("RGFX pipe: associate: the framebuffer serves no more pipe clients\n");
		return kIOReturnNoResources;
	}
	a->scalarOutput[0] = kAcceleratorId;
	IOLog("RGFX pipe: associated index %d with accelerator %llu\n", static_cast<int>(index),
	      kAcceleratorId);
	return kIOReturnSuccess;
}

IOReturn RgfxDisplayPipeUserClient::sModeScaler(OSObject *, void *, IOExternalMethodArguments *)
{
	return kIOReturnUnsupported;
}

IOReturn RgfxDisplayPipeUserClient::sCapabilities(OSObject *target, void *,
                                                  IOExternalMethodArguments *a)
{
	RgfxDisplayPipeUserClient *self = OSDynamicCast(RgfxDisplayPipeUserClient, target);
	if (!self || !self->owner)
		return kIOReturnNotAttached;

	// The capabilities the accelerator publishes (AcceleratorProperties'
	// IOAccelDisplayPipeCapabilities), so the pipe and the registry agree.
	OSDictionary *caps = nullptr;
	OSDictionary *props = OSDynamicCast(OSDictionary, self->owner->getProperty("AcceleratorProperties"));
	if (props)
		caps = OSDynamicCast(OSDictionary, props->getObject("IOAccelDisplayPipeCapabilities"));
	OSDictionary *made = nullptr;
	if (!caps) {
		made = OSDictionary::withCapacity(2);
		if (!made)
			return kIOReturnNoMemory;
		made->setObject("DisplayPipeSupported", kOSBooleanTrue);
		made->setObject("TransactionsSupported", kOSBooleanTrue);
		caps = made;
	}
	OSSerialize *s = OSSerialize::withCapacity(1024);
	if (!s || !caps->serialize(s)) {
		OSSafeReleaseNULL(s);
		OSSafeReleaseNULL(made);
		return kIOReturnUnsupported;
	}
	const char *text = s->text();
	UInt32 len = s->getLength(); // the text and its terminating NUL
	IOReturn r = kIOReturnSuccess;
	if (a->structureOutputDescriptor) {
		// The output comes as a descriptor over the caller's memory, not wired: copy through a
		// kernel mapping of it (writeBytes on an unwired descriptor panics).
		if (len > a->structureOutputDescriptorSize) {
			r = kIOReturnNoSpace;
		} else {
			IOMemoryMap *map = a->structureOutputDescriptor->map();
			if (!map) {
				r = kIOReturnVMError;
			} else {
				memcpy(reinterpret_cast<void *>(map->getVirtualAddress()), text, len);
				map->release();
				a->structureOutputDescriptorSize = len;
			}
		}
	} else if (a->structureOutput) {
		if (len > a->structureOutputSize) {
			r = kIOReturnNoSpace;
		} else {
			memcpy(a->structureOutput, text, len);
			a->structureOutputSize = len;
		}
	} else {
		r = kIOReturnBadArgument;
	}
	IOLog("RGFX pipe: capabilities (%u bytes): 0x%x\n", len, r);
	s->release();
	OSSafeReleaseNULL(made);
	return r;
}

IOReturn RgfxDisplayPipeUserClient::sRequestNotify(OSObject *target, void *,
                                                   IOExternalMethodArguments *a)
{
	RgfxDisplayPipeUserClient *self = OSDynamicCast(RgfxDisplayPipeUserClient, target);
	if (!self || !self->associated)
		return kIOReturnNotAttached;
	UInt8 buf[kNotifySize];
	const UInt8 *in = structIn(a, buf, kNotifySize);
	if (!in || a->asyncWakePort == MACH_PORT_NULL || !a->asyncReference ||
	    a->asyncReferenceCount < 3)
		return kIOReturnBadArgument;
	UInt32 kind = rd32(in, kNotifyKind);
	if (kind >= kKinds)
		return kIOReturnBadArgument;
	if (kind == kKindVbl) {
		// A request for the next refresh's notification. The device's VBLANK events go on
		// under the lock, so the last request's answer, which turns them off under it, never
		// passes a new request.
		IOLockLock(self->lock);
		if (self->vblWaiting == kMaxVblRequests) {
			IOLockUnlock(self->lock);
			return kIOReturnNoResources;
		}
		VblRequest &r = self->vblRequests[self->vblWaiting++];
		bcopy(a->asyncReference, r.reference, sizeof(OSAsyncReference64));
		r.callback = rd64(in, kNotifyCallback);
		r.refcon = rd64(in, kNotifyRefcon);
		self->framebuffer->setPipeVblankEnabled(self, true);
		IOLockUnlock(self->lock);
		if (self->logged < kLogFirst) {
			self->logged++;
			IOLog("RGFX pipe: VBL request: callback %016llx refcon %016llx\n", r.callback,
			      r.refcon);
		}
		return kIOReturnSuccess;
	}
	IOLog("RGFX pipe: notify kind %u: %016llx %016llx %016llx\n", kind, rd64(in, 0), rd64(in, 8),
	      rd64(in, 16));
	IOLockLock(self->lock);
	if (self->registered) {
		IOLockUnlock(self->lock);
		return kIOReturnBusy;
	}
	bcopy(a->asyncReference, self->transactionRef, sizeof(OSAsyncReference64));
	self->registered = true;
	IOLockUnlock(self->lock);
	return kIOReturnSuccess;
}

IOReturn RgfxDisplayPipeUserClient::sBegin(OSObject *target, void *, IOExternalMethodArguments *a)
{
	RgfxDisplayPipeUserClient *self = OSDynamicCast(RgfxDisplayPipeUserClient, target);
	if (!self || !self->associated)
		return kIOReturnNotAttached;
	// What the transaction begun before staged, never submitted, goes (RgfxPipeColor.hpp).
	GammaStage dropped[3];
	IOLockLock(self->lock);
	if (++self->nextTransaction == 0)
		self->nextTransaction = 1;
	self->begun = self->nextTransaction;
	a->scalarOutput[0] = self->begun;
	for (UInt32 i = 0; i < 3; i++) {
		dropped[i] = self->staged[i];
		self->staged[i] = {};
	}
	self->stagedFor = 0;
	IOLockUnlock(self->lock);
	for (GammaStage &g : dropped) {
		if (g.entries)
			IOFree(g.entries, static_cast<vm_size_t>(g.count) * kRgfxGammaEntrySize);
	}
	return kIOReturnSuccess;
}

IOReturn RgfxDisplayPipeUserClient::sSeven(OSObject *target, void *, IOExternalMethodArguments *)
{
	RgfxDisplayPipeUserClient *self = OSDynamicCast(RgfxDisplayPipeUserClient, target);
	if (!self || !self->associated)
		return kIOReturnNotAttached;
	return kIOReturnSuccess;
}

// Selectors 5 to 7, where the host took the feature: the gamma table, decoded as macOS's kernel
// decodes it, staged for the begun transaction (RgfxPipeColor.hpp). The structure comes inline
// or as a descriptor, which is mapped read-only (kIOReturnVMError where it cannot be).
IOReturn RgfxDisplayPipeUserClient::sGamma(OSObject *target, void *, IOExternalMethodArguments *a)
{
	RgfxDisplayPipeUserClient *self = OSDynamicCast(RgfxDisplayPipeUserClient, target);
	if (!self || !self->associated)
		return kIOReturnNotAttached;
	const int slot = rgfxGammaSlot(a->selector);
	if (slot < 0)
		return kIOReturnBadArgument;
	IOMemoryMap *map = nullptr;
	const UInt8 *in = nullptr;
	UInt64 size = 0;
	if (a->structureInput) {
		in = static_cast<const UInt8 *>(a->structureInput);
		size = a->structureInputSize;
	} else if (a->structureInputDescriptor) {
		map = a->structureInputDescriptor->map(kIOMapReadOnly);
		if (!map)
			return kIOReturnVMError;
		in = reinterpret_cast<const UInt8 *>(map->getVirtualAddress());
		size = map->getLength();
	}
	RgfxGammaArgs args = {};
	if (!rgfxGammaDecode(in, size, &args)) {
		OSSafeReleaseNULL(map);
		if (self->colorLogged < kLogColor) {
			self->colorLogged++;
			IOLog("RGFX pipe: gamma selector %u: %llu bytes, refused\n", a->selector, size);
		}
		return kIOReturnBadArgument;
	}
	// Copied before the lock: up to 2 MiB, which the work loop never waits behind.
	const UInt8 *from = in + kRgfxGammaHeaderSize;
	const vm_size_t bytes = static_cast<vm_size_t>(args.count) * kRgfxGammaEntrySize;
	UInt8 *entries = nullptr;
	if (args.count) {
		entries = static_cast<UInt8 *>(IOMalloc(bytes));
		if (!entries) {
			OSSafeReleaseNULL(map);
			return kIOReturnNoMemory;
		}
		memcpy(entries, from, bytes);
	}
	OSSafeReleaseNULL(map);

	IOLockLock(self->lock);
	const UInt32 begun = self->begun;
	const RgfxGammaAction action = rgfxGammaAction(a->selector, args, begun);
	GammaStage old = {};
	if (action == RgfxGammaAction::Stage) {
		// The begin before it let go of anything staged for another transaction.
		self->stagedFor = begun;
		old = self->staged[slot];
		self->staged[slot] = {entries, args.count, true};
		entries = nullptr;
	}
	const bool log = self->colorLogged < kLogColor || (begun && begun % kLogEvery == 0);
	if (self->colorLogged < kLogColor)
		self->colorLogged++;
	IOLockUnlock(self->lock);
	if (log) {
		const UInt8 *last = args.count ? from + bytes - kRgfxGammaEntrySize : from;
		IOLog("RGFX pipe: gamma selector %u transaction %u (begun %u) plane %u entries %u "
		      "field_0c %x field_10 %08x field_14 %08x: %s; first %08x %08x %08x last %08x %08x "
		      "%08x\n",
		      a->selector, args.transaction, begun, args.plane, args.count, args.field_0c,
		      args.field_10, args.field_14,
		      action == RgfxGammaAction::Stage    ? "staged"
		      : action == RgfxGammaAction::Ignore ? "not shown"
		                                          : "refused",
		      args.count ? rgfxColorRead32(from, 0) : 0, args.count ? rgfxColorRead32(from, 4) : 0,
		      args.count ? rgfxColorRead32(from, 8) : 0, args.count ? rgfxColorRead32(last, 0) : 0,
		      args.count ? rgfxColorRead32(last, 4) : 0, args.count ? rgfxColorRead32(last, 8) : 0);
	}
	if (entries)
		IOFree(entries, bytes);
	if (old.entries)
		IOFree(old.entries, static_cast<vm_size_t>(old.count) * kRgfxGammaEntrySize);
	return action == RgfxGammaAction::BadArgument ? kIOReturnBadArgument : kIOReturnSuccess;
}

IOReturn RgfxDisplayPipeUserClient::sSubmit(OSObject *target, void *, IOExternalMethodArguments *a)
{
	RgfxDisplayPipeUserClient *self = OSDynamicCast(RgfxDisplayPipeUserClient, target);
	if (!self || !self->associated || !self->framebuffer)
		return kIOReturnNotAttached;
	UInt8 buf[kTransactionSize];
	const UInt8 *t = structIn(a, buf, kTransactionSize);
	if (!t)
		return kIOReturnBadArgument;
	UInt32 id = rd32(t, kTxnId);
	UInt32 planeType = rd32(t, kTxnPlane0 + kPlaneType);
	UInt32 surface = rd32(t, kTxnPlane0 + kPlaneSurface);
	UInt64 cookie = rd64(t, kTxnCookie);
	bool log = id <= kLogFirst / 4 || id % kLogEvery == 0;
	if (log) {
		UInt64 dirty = rd64(t, kTxnDirty);
		// The time asked for, in microseconds from now (0 where it asks for none).
		long long at = (dirty & kDirtyTimestamp)
		                   ? (static_cast<long long>(rd64(t, kTxnTimestamp)) -
		                      static_cast<long long>(mach_absolute_time())) / 1000
		                   : 0;
		IOLog("RGFX pipe: transaction %u (begun %u): dirty %llx at %lld us options %x plane0 "
		      "type %u ids %u %u cookie %llx\n",
		      id, self->begun, dirty, at, rd32(t, kTxnOptions), planeType, surface,
		      rd32(t, kTxnPlane0 + kPlaneSurface + 4), cookie);
	}
	// One submit at a time, in their order, under a lock the work loop never takes.
	IOLockLock(self->submitLock);
	IOReturn r = self->submit(t, id, planeType, surface, cookie);
	IOLockUnlock(self->submitLock);
	return r;
}

// sSubmit's transaction, submitLock held.
IOReturn RgfxDisplayPipeUserClient::submit(const UInt8 *t, UInt32 id, UInt32 planeType,
                                           UInt32 surface, UInt64 cookie)
{
	IOLockLock(lock);
	if (begun == id)
		begun = 0;
	// Room for one more outstanding transaction, waiting up to five seconds.
	uint64_t deadline = 0;
	clock_interval_to_deadline(5000, kMillisecondScale, &deadline);
	while (!transactions.hasRoom()) {
		if (IOLockSleepDeadline(lock, &completed, deadline, THREAD_UNINT) == THREAD_TIMED_OUT &&
		    !transactions.hasRoom()) {
			IOLockUnlock(lock);
			IOLog("RGFX pipe: transaction %u: no room in 5 s, not sent\n", id);
			return kIOReturnTimeout;
		}
	}
	// Notified at its completion where it carries a cookie (RgfxPipeTransactions.hpp).
	bool notified = rgfxPipeNotifies(cookie);
	if (!notified && unnotifiedLogged < kLogUnnotified) {
		unnotifiedLogged++;
		IOLog("RGFX pipe: transaction %u (options 0x%x) carries no cookie: it completes for its "
		      "waiters, without a notification\n",
		      id, rd32(t, kTxnOptions));
	}
	IOLockUnlock(lock);
	// What the transaction posts, its colour then its present, goes with no other pipe client's
	// post between them (ReimsGraphics::lockPresents): the host stages the colour for the next
	// present it takes, and drops a stage whose parts a present cuts short.
	framebuffer->lockPresents();
	const IOReturn r = post(t, id, planeType, surface, cookie, notified);
	framebuffer->unlockPresents();
	return r;
}

// submit's posts, submitLock and the framebuffer's presents lock held, `lock` not; the room for
// the transaction found. No other submit runs meanwhile (submitLock), and presented() only
// completes transactions, so the room stays.
IOReturn RgfxDisplayPipeUserClient::post(const UInt8 *t, UInt32 id, UInt32 planeType,
                                         UInt32 surface, UInt64 cookie, bool notified)
{
	// The colour the transaction sets goes first.
	bool colored = false;
	if (displayColor) {
		const IOReturn c = postColor(t, id, &colored);
		if (c != kIOReturnSuccess) {
			IOLog("RGFX pipe: transaction %u: its colour not sent: 0x%x\n", id, c);
			return c;
		}
	}
	IOLockLock(lock);
	const RgfxPipeSends sends =
	    rgfxPipeSends(planeType == kPlaneIOSurface && surface != 0, colored);
	if (!rgfxPipeWaitsForAnswer(sends)) {
		// Nothing for the host to show: the display keeps its surface, and the transaction is
		// done with the last one before it (at once where none is outstanding).
		Transactions::Completion now = {};
		if (transactions.nothingToShow(id, cookie, notified, &now))
			completeLocked(now, mach_absolute_time());
		IOLockUnlock(lock);
		return kIOReturnSuccess;
	}
	// A transaction that sets only a colour keeps the surface shown (RGFX_PRESENT_KEEP).
	const bool kept = sends == RgfxPipeSends::Keep;
	const UInt32 present = kept ? 0 : surface;
	const UInt32 flags = kept ? RGFX_PRESENT_KEEP : 0;
	// Posted and recorded under the lock, so the host's answer, which the work loop hands to
	// presented() under the same lock, finds it recorded. Where the control ring has no room, the
	// lock is let go while the post waits for it: the work loop takes the lock at every COMPLETE,
	// and every queue's completions wait behind its pass.
	uint64_t ringDeadline = 0;
	clock_interval_to_deadline(5000, kMillisecondScale, &ringDeadline);
	UInt64 seqno = 0;
	IOReturn r = kIOReturnSuccess;
	for (;;) {
		const UInt64 mark = framebuffer->roomMark();
		r = framebuffer->postPresent(present, id, flags, &seqno);
		if (r != kIOReturnNoSpace)
			break;
		if (framebuffer->askForRoom() &&
		    (r = framebuffer->postPresent(present, id, flags, &seqno)) != kIOReturnNoSpace)
			break;
		IOLockUnlock(lock);
		const bool more = framebuffer->waitForRoom(mark, ringDeadline);
		IOLockLock(lock);
		if (!more) {
			r = kIOReturnTimeout;
			break;
		}
	}
	if (r != kIOReturnSuccess) {
		IOLockUnlock(lock);
		IOLog("RGFX pipe: transaction %u: PRESENT_SURFACE not sent: 0x%x\n", id, r);
		return r;
	}
	// Done at the host's answer: the display shows its surface, or its colour, from then.
	transactions.sent(seqno, id, cookie, notified);
	IOLockUnlock(lock);
	return kIOReturnSuccess;
}

// The colour transaction `id` sets (RgfxPipeColor.hpp), submitLock held and `lock` not: the
// gamma tables staged for it, then the matrix where its dirty mask has the matrix's bit, each as
// DISPLAY_COLOR parts; `*sent` says whether any went.
IOReturn RgfxDisplayPipeUserClient::postColor(const UInt8 *t, UInt32 id, bool *sent)
{
	GammaStage taken[3] = {};
	IOLockLock(lock);
	if (stagedFor == id) {
		for (UInt32 i = 0; i < 3; i++) {
			taken[i] = staged[i];
			staged[i] = {};
		}
		stagedFor = 0;
	}
	IOLockUnlock(lock);
	IOReturn r = kIOReturnSuccess;
	*sent = false;
	for (UInt32 i = 0; i < 3 && r == kIOReturnSuccess; i++) {
		if (!taken[i].set)
			continue;
		r = framebuffer->postDisplayColor(id, rgfxGammaSlotStage(static_cast<int>(i)),
		                                  taken[i].count, taken[i].entries);
		*sent = true;
	}
	const UInt64 dirty = rd64(t, kTxnDirty);
	if (r == kIOReturnSuccess && (dirty & kRgfxDirtyMatrix)) {
		r = framebuffer->postDisplayColor(id, RGFX_COLOR_MATRIX, kRgfxMatrixRows,
		                                  t + kRgfxTxnMatrix);
		*sent = true;
		if (colorLogged < kLogColor || id % kLogEvery == 0) {
			colorLogged += colorLogged < kLogColor ? 1 : 0;
			IOLog("RGFX pipe: transaction %u: colour matrix %08x %08x %08x %08x / %08x %08x %08x "
			      "%08x / %08x %08x %08x %08x\n",
			      id, rd32(t, kRgfxTxnMatrix), rd32(t, kRgfxTxnMatrix + 4),
			      rd32(t, kRgfxTxnMatrix + 8), rd32(t, kRgfxTxnMatrix + 12),
			      rd32(t, kRgfxTxnMatrix + 16), rd32(t, kRgfxTxnMatrix + 20),
			      rd32(t, kRgfxTxnMatrix + 24), rd32(t, kRgfxTxnMatrix + 28),
			      rd32(t, kRgfxTxnMatrix + 32), rd32(t, kRgfxTxnMatrix + 36),
			      rd32(t, kRgfxTxnMatrix + 40), rd32(t, kRgfxTxnMatrix + 44));
		}
	}
	for (GammaStage &g : taken) {
		if (g.entries)
			IOFree(g.entries, static_cast<vm_size_t>(g.count) * kRgfxGammaEntrySize);
	}
	return r;
}

IOReturn RgfxDisplayPipeUserClient::sWait(OSObject *target, void *, IOExternalMethodArguments *a)
{
	RgfxDisplayPipeUserClient *self = OSDynamicCast(RgfxDisplayPipeUserClient, target);
	if (!self || !self->associated)
		return kIOReturnNotAttached;
	UInt8 buf[kWaitSize];
	const UInt8 *w = structIn(a, buf, kWaitSize);
	if (!w)
		return kIOReturnBadArgument;
	UInt32 id = rd32(w, 0);
	UInt32 ms = rd32(w, 8);
	uint64_t deadline = 0;
	clock_interval_to_deadline(ms, kMillisecondScale, &deadline);
	IOReturn r = kIOReturnSuccess;
	IOLockLock(self->lock);
	if (id == 0 || id > self->nextTransaction) {
		r = kIOReturnBadArgument;
	} else if (id == self->begun) {
		r = kIOReturnNotReady; // begun, never submitted
	} else {
		while (id > self->completed) {
			if (IOLockSleepDeadline(self->lock, &self->completed, deadline, THREAD_UNINT) == THREAD_TIMED_OUT &&
			    id > self->completed) {
				r = kIOReturnTimeout;
				break;
			}
		}
	}
	IOLockUnlock(self->lock);
	return r;
}

// MARK: - Notifications

// Completes a transaction: waiters woken, and its completion notification where it is notified
// (RgfxPipeTransactions.hpp) and one is registered. Called with the lock held.
void RgfxDisplayPipeUserClient::completeLocked(const Transactions::Completion &c, UInt64 time)
{
	if (c.id > completed)
		completed = c.id;
	if (c.notified && registered) {
		io_user_reference_t words[kRgfxPipeCompletionWords];
		rgfxPipeCompletionWords(words, c.cookie, c.id, time);
		sendAsyncResult64(transactionRef, kIOReturnSuccess, words, kRgfxPipeCompletionWords);
	}
	IOLockWakeup(lock, &completed, false);
}

void RgfxDisplayPipeUserClient::presentHook(OSObject *client, UInt64 seqno, UInt32 status,
                                            UInt64 time)
{
	RgfxDisplayPipeUserClient *self = OSDynamicCast(RgfxDisplayPipeUserClient, client);
	if (self)
		self->presented(seqno, status, time);
}

// The host answered a COMPLETE: where it is one of our PRESENT_SURFACEs, the display shows that
// transaction's surface since the refresh at `time`, so it and every transaction before it are
// done then, oldest first.
void RgfxDisplayPipeUserClient::presented(UInt64 seqno, UInt32 status, UInt64 time)
{
	IOLockLock(lock);
	Transactions::Completion done[kMaxOutstanding];
	UInt32 n = transactions.answered(seqno, done);
	if (n == 0) {
		IOLockUnlock(lock);
		return;
	}
	if (status != 0 && logged < kLogFirst) {
		logged++;
		IOLog("RGFX pipe: transaction %u: the host did not show its IOSurface (status %u)\n",
		      done[n - 1].id, status);
	}
	for (UInt32 i = 0; i < n; i++)
		completeLocked(done[i], time);
	IOLockUnlock(lock);
}

void RgfxDisplayPipeUserClient::vblankHook(OSObject *client, UInt64, UInt64 time, UInt64 next)
{
	RgfxDisplayPipeUserClient *self = OSDynamicCast(RgfxDisplayPipeUserClient, client);
	if (self)
		self->sendVbl(time, next);
}

// A refresh: each waiting VBL request is answered, once, with its callback and refcon, the
// refresh's time and the next one's; with none left waiting, the device's VBLANK events go off.
void RgfxDisplayPipeUserClient::sendVbl(UInt64 time, UInt64 next)
{
	IOLockLock(lock);
	if (vblWaiting == 0) {
		IOLockUnlock(lock);
		return;
	}
	for (UInt32 i = 0; i < vblWaiting; i++) {
		VblRequest &r = vblRequests[i];
		io_user_reference_t words[kRgfxPipeVblWords];
		rgfxPipeVblWords(words, r.callback, r.refcon, time, next);
		sendAsyncResult64(r.reference, kIOReturnSuccess, words, kRgfxPipeVblWords);
		releaseAsyncReference64(r.reference);
	}
	vblWaiting = 0;
	framebuffer->setPipeVblankEnabled(this, false);
	IOLockUnlock(lock);
}
