/*
 * RgfxDisplayPipeUserClient: see RgfxDisplayPipeUserClient.hpp. The display pipe's selectors,
 * presenting each transaction's IOSurface through the framebuffer's control ring.
 */

#include "RgfxDisplayPipeUserClient.hpp"

#include "ReimsGraphics.hpp"
#include "RgfxAccelerator.hpp"
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
// into `buf`); nullptr where it is shorter.
const UInt8 *structIn(IOExternalMethodArguments *a, UInt8 *buf, UInt32 size)
{
	if (a->structureInput && a->structureInputSize >= size)
		return static_cast<const UInt8 *>(a->structureInput);
	if (a->structureInputDescriptor && a->structureInputDescriptor->getLength() >= size &&
	    a->structureInputDescriptor->readBytes(0, buf, size) == size)
		return buf;
	return nullptr;
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
	if (selector >= 10 || !methods[selector].function) {
		IOLog("RGFX pipe: selector %u is not supported\n", selector);
		return kIOReturnUnsupported;
	}
	dispatch = const_cast<IOExternalMethodDispatch *>(&methods[selector]);
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
	IOLockLock(self->lock);
	if (++self->nextTransaction == 0)
		self->nextTransaction = 1;
	self->begun = self->nextTransaction;
	a->scalarOutput[0] = self->begun;
	IOLockUnlock(self->lock);
	return kIOReturnSuccess;
}

IOReturn RgfxDisplayPipeUserClient::sSeven(OSObject *target, void *, IOExternalMethodArguments *)
{
	RgfxDisplayPipeUserClient *self = OSDynamicCast(RgfxDisplayPipeUserClient, target);
	if (!self || !self->associated)
		return kIOReturnNotAttached;
	return kIOReturnSuccess;
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
	if (planeType != kPlaneIOSurface || surface == 0) {
		// Nothing for the host to show: the display keeps its surface, and the transaction is
		// done with the last one before it (at once where none is outstanding).
		Transactions::Completion now = {};
		if (transactions.nothingToShow(id, cookie, notified, &now))
			completeLocked(now, mach_absolute_time());
		IOLockUnlock(lock);
		return kIOReturnSuccess;
	}
	// Posted and recorded under the lock, so the host's answer, which the work loop hands to
	// presented() under the same lock, finds it recorded. Where the control ring has no room, the
	// lock is let go while the post waits for it: the work loop takes the lock at every COMPLETE,
	// and every queue's completions wait behind its pass. No other submit runs meanwhile
	// (submitLock), and presented() only completes transactions, so the room found above stays.
	uint64_t ringDeadline = 0;
	clock_interval_to_deadline(5000, kMillisecondScale, &ringDeadline);
	UInt64 seqno = 0;
	IOReturn r = kIOReturnSuccess;
	for (;;) {
		const UInt64 mark = framebuffer->roomMark();
		r = framebuffer->postPresent(surface, id, &seqno);
		if (r != kIOReturnNoSpace)
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
	transactions.sent(seqno, id, cookie, notified);
	IOLockUnlock(lock);
	return kIOReturnSuccess;
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
