/*
 * RgfxDisplayPipeUserClient: see RgfxDisplayPipeUserClient.hpp. The display pipe's selectors,
 * presenting each transaction's IOSurface through the framebuffer's control ring.
 */

#include "RgfxDisplayPipeUserClient.hpp"

#include "ReimsGraphics.hpp"
#include "RgfxAccelerator.hpp"

#include <IOKit/IOLib.h>
#include <IOKit/IOMemoryDescriptor.h>
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

// Inside a transaction: its id and options, plane 0 (0x40 bytes), and the cookie its
// completion carries back.
constexpr UInt32 kTxnId = 0x10;
constexpr UInt32 kTxnOptions = 0x14;
constexpr UInt32 kTxnPlane0 = 0x18;
constexpr UInt32 kTxnCookie = 0x100;
// Inside a plane: its type (1, an IOSurface) and the surface's global ID.
constexpr UInt32 kPlaneType = 0x0;
constexpr UInt32 kPlaneSurface = 0x8;
constexpr UInt32 kPlaneIOSurface = 1;

// The messages back: a transaction's completion, 11 words (the cookie in word 0, the id in
// word 1's high half, the completion time in word 3, action 1 in word 8), and a VBL, 6 words.
constexpr UInt32 kCompletionWords = 11;
constexpr UInt32 kVblWords = 6;

// The entitlement an association needs: WindowServer has it.
const char *const kEntitlement = "com.apple.private.hid.client.event-monitor";

// The accelerator id the association returns: our one accelerator.
constexpr UInt64 kAcceleratorId = 1;

// The calls logged in full (sizes, the registrations' and transactions' fields); after that,
// every 600th transaction.
constexpr UInt32 kLogFirst = 48;
constexpr UInt32 kLogEvery = 600;

// A frame at 60 Hz in mach_absolute_time units (nanoseconds on x86-64): the VBL's next-tick
// estimate before two ticks have been seen.
constexpr UInt64 kDefaultVblPeriod = 16666667;

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
	for (UInt32 k = 0; k < kKinds; k++)
		registered[k] = false;
	nextTransaction = 0;
	begun = 0;
	completed = 0;
	lastVblTime = 0;
	logged = 0;
	return lock != nullptr;
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
	super::free();
}

// Stops the VBL hook and gives back the wake ports the registrations kept.
void RgfxDisplayPipeUserClient::releaseRegistrations()
{
	if (framebuffer)
		framebuffer->setPipeVblank(nullptr, nullptr);
	if (!lock)
		return;
	IOLockLock(lock);
	for (UInt32 k = 0; k < kKinds; k++) {
		if (registered[k]) {
			releaseAsyncReference64(references[k]);
			registered[k] = false;
		}
	}
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
	IOLog("RGFX pipe: notify kind %u: %016llx %016llx %016llx\n", kind, rd64(in, 0), rd64(in, 8),
	      rd64(in, 16));
	// A VBL registration turns the device's VBLANK events on first, so a failed enable leaves
	// no reference behind.
	if (kind == kKindVbl)
		self->framebuffer->setPipeVblank(self, &RgfxDisplayPipeUserClient::vblankHook);
	IOLockLock(self->lock);
	if (self->registered[kind]) {
		IOLockUnlock(self->lock);
		return kIOReturnBusy;
	}
	bcopy(a->asyncReference, self->references[kind], sizeof(OSAsyncReference64));
	self->cookies[kind] = rd64(in, 0);
	self->registered[kind] = true;
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
	if (log)
		IOLog("RGFX pipe: transaction %u (begun %u): dirty %llx options %x plane0 type %u "
		      "ids %u %u cookie %llx\n",
		      id, self->begun, rd64(t, 0), rd32(t, kTxnOptions), planeType, surface,
		      rd32(t, kTxnPlane0 + kPlaneSurface + 4), cookie);
	if (planeType == kPlaneIOSurface && surface != 0) {
		IOReturn r = self->framebuffer->presentSurface(surface, id);
		if (r != kIOReturnSuccess && log)
			IOLog("RGFX pipe: transaction %u: the host did not show IOSurface %u: 0x%x\n", id,
			      surface, r);
	}
	IOLockLock(self->lock);
	self->completed = id;
	if (self->begun == id)
		self->begun = 0;
	IOLockUnlock(self->lock);
	self->sendCompletion(id, cookie, mach_absolute_time());
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
	IOLockLock(self->lock);
	// Transactions complete inside their submit, so a submitted one is already done.
	bool done = id != 0 && id <= self->completed;
	bool pending = id != 0 && id == self->begun;
	IOLockUnlock(self->lock);
	if (done)
		return kIOReturnSuccess;
	return pending ? kIOReturnNotReady : kIOReturnBadArgument;
}

// MARK: - Notifications

void RgfxDisplayPipeUserClient::sendCompletion(UInt32 transactionId, UInt64 cookie, UInt64 time)
{
	io_user_reference_t words[kCompletionWords] = {};
	words[0] = cookie;
	words[1] = static_cast<io_user_reference_t>(transactionId) << 32;
	words[3] = time;
	words[8] = 1;
	IOLockLock(lock);
	if (registered[kKindTransaction])
		sendAsyncResult64(references[kKindTransaction], kIOReturnSuccess, words, kCompletionWords);
	IOLockUnlock(lock);
}

void RgfxDisplayPipeUserClient::vblankHook(OSObject *client, UInt64 count, UInt64 time)
{
	RgfxDisplayPipeUserClient *self = OSDynamicCast(RgfxDisplayPipeUserClient, client);
	if (self)
		self->sendVbl(count, time);
}

// A VBL: the registration's cookie, the tick's count, and its time and the next tick's.
void RgfxDisplayPipeUserClient::sendVbl(UInt64 count, UInt64 time)
{
	IOLockLock(lock);
	UInt64 period = lastVblTime && time > lastVblTime ? time - lastVblTime : kDefaultVblPeriod;
	lastVblTime = time;
	if (registered[kKindVbl]) {
		io_user_reference_t words[kVblWords] = {};
		words[0] = cookies[kKindVbl];
		words[1] = count << 32;
		words[3] = time;
		words[4] = time + period;
		sendAsyncResult64(references[kKindVbl], kIOReturnSuccess, words, kVblWords);
	}
	IOLockUnlock(lock);
}
