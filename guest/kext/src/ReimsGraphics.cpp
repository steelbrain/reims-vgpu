/*
 * ReimsGraphics: the reims-graphics device's display driver (ReimsGraphics.hpp
 * says what it does and on which threads).
 */

#include "ReimsGraphics.hpp"
#include "RgfxAccelerator.hpp"
#include "RgfxFeatures.hpp"

#include <IOKit/IOLib.h>
#include <IOKit/IOUserClient.h>
#include <kern/clock.h>
#include <kern/task.h>
#include <kern/task_ref.h>
#include <libkern/libkern.h>

#include "RgfxTrace.hpp"

#define super IOFramebuffer
OSDefineMetaClassAndStructors(ReimsGraphics, IOFramebuffer);

namespace {

// The control ring's byte ring is a power of two, so an index wraps with a mask.
constexpr UInt32 kRingBytes = RGFX_CONTROL_RING_DATA_SIZE;
constexpr UInt32 kRingMask = kRingBytes - 1;
static_assert((kRingBytes & kRingMask) == 0, "the ring size is not a power of two");
constexpr UInt32 kHeaderBytes = sizeof(RgfxMsgHeader);

// A message's size is a multiple of this.
constexpr UInt32 kMsgAlign = 8;

// The ring header's fields, by their offset in it (each in a 64-byte line).
constexpr UInt32 kRingProducer = __builtin_offsetof(RgfxRingHeader, producer);
constexpr UInt32 kRingConsumer = __builtin_offsetof(RgfxRingHeader, consumer);
constexpr UInt32 kRingConsumerIdle = __builtin_offsetof(RgfxRingHeader, consumer_idle);
constexpr UInt32 kRingProducerWaiting = __builtin_offsetof(RgfxRingHeader, producer_waiting);
constexpr UInt32 kRingArmed = __builtin_offsetof(RgfxRingHeader, armed);

// The cursor image's slot, at the region's end: 256 x 256 BGRA8, the largest
// image the hardware cursor takes. A larger one is IOFramebuffer's to draw.
constexpr UInt32 kCursorMax = 256;
constexpr UInt64 kCursorSlot = kCursorMax * kCursorMax * 4;

// Scanout rows start on 256-byte boundaries, as the option ROM's do, so the
// boot frame keeps its layout when the driver takes the display over.
constexpr UInt32 kStrideAlign = 256;
constexpr UInt32 kMaxDimension = 16384;

// How long a command waits for ring space, and then for its answer.
constexpr UInt32 kPostTimeoutMs = 5000;
constexpr UInt32 kAnswerTimeoutMs = 5000;
// How often the control ring's room line is logged at most, while posts sleep for room.
constexpr UInt32 kRoomLogMs = 1000;

// Answers carry the device's status; a payload the driver cannot read is
// answered with this one.
constexpr UInt32 kStatusUnreadable = 0xffffffff;

// The ring's header fields: atomic, as the protocol orders them.
inline UInt64 load64(const volatile UInt8 *p)
{
	return __atomic_load_n(reinterpret_cast<const volatile UInt64 *>(p), __ATOMIC_ACQUIRE);
}

inline void store64(volatile UInt8 *p, UInt64 value)
{
	__atomic_store_n(reinterpret_cast<volatile UInt64 *>(p), value, __ATOMIC_RELEASE);
}

inline UInt32 load32(const volatile UInt8 *p)
{
	return __atomic_load_n(reinterpret_cast<const volatile UInt32 *>(p), __ATOMIC_ACQUIRE);
}

inline void store32(volatile UInt8 *p, UInt32 value)
{
	__atomic_store_n(reinterpret_cast<volatile UInt32 *>(p), value, __ATOMIC_RELEASE);
}

inline void fullFence()
{
	__atomic_thread_fence(__ATOMIC_SEQ_CST);
}

// A payload's 32-bit field, little-endian as x86-64 is.
inline UInt32 field32(const volatile UInt8 *payload, UInt32 offset)
{
	return *reinterpret_cast<const volatile UInt32 *>(payload + offset);
}

#if RGFX_FB_TRACE
// A trace line, logged for a site's first `limit` calls.
#define FBTRACE(limit, fmt, ...)                                                             \
	do {                                                                                 \
		static UInt32 traced_;                                                       \
		if (__atomic_fetch_add(&traced_, 1, __ATOMIC_RELAXED) < (limit))           \
			IOLog("RGFX fb " fmt "\n", ##__VA_ARGS__);                          \
	} while (0)

// An attribute's call, logged for the first few calls of each (kind, attribute).
void traceAttribute(const char *kind, IOIndex connect, IOSelect attribute, uintptr_t value,
                    IOReturn r)
{
	struct Seen {
		const char *kind;
		IOSelect attribute;
		UInt32 count;
	};
	static Seen seen[128];
	UInt32 i = 0;
	for (; i < 128 && seen[i].kind; i++) {
		if (seen[i].kind == kind && seen[i].attribute == attribute)
			break;
	}
	if (i == 128)
		return;
	if (!seen[i].kind) {
		seen[i].kind = kind;
		seen[i].attribute = attribute;
	}
	if (seen[i].count++ >= 4)
		return;
	char cc[5];
	for (int b = 0; b < 4; b++) {
		char c = static_cast<char>((attribute >> (24 - 8 * b)) & 0xff);
		cc[b] = c >= 0x20 && c < 0x7f ? c : '.';
	}
	cc[4] = 0;
	IOLog("RGFX fb %s conn %d '%s' (0x%x) value 0x%lx -> 0x%x\n", kind, static_cast<int>(connect),
	      cc, attribute, static_cast<unsigned long>(value), r);
}
#else
#define FBTRACE(...) do {} while (0)
inline void traceAttribute(const char *, IOIndex, IOSelect, uintptr_t, IOReturn) {}
#endif

} // namespace

// MARK: - Registers and the control rings

UInt32 ReimsGraphics::readRegister(UInt32 offset) const
{
	return *reinterpret_cast<const volatile UInt32 *>(registers + offset);
}

void ReimsGraphics::writeRegister(UInt32 offset, UInt32 value)
{
	*reinterpret_cast<volatile UInt32 *>(registers + offset) = value;
}

volatile UInt8 *ReimsGraphics::ring(UInt32 offset) const
{
	return rings + offset;
}

// Puts one command on ring 0 if it fits, without blocking: callable with a
// spin lock held. The doorbell rings only when the host's consumer sleeps.
bool ReimsGraphics::tryPost(UInt32 type, const void *message, UInt32 size, UInt64 *seqno,
                            UInt64 *waiterSeqno)
{
	if (size < kHeaderBytes || size % kMsgAlign || size > kLifecycleMessageMax || !ready)
		return false;
	IOSimpleLockLock(submitLock);
	const bool written = writeLocked(type, message, size, seqno, waiterSeqno);
	if (written)
		publishLocked();
	IOSimpleLockUnlock(submitLock);
	return written;
}

// Writes one command at the producer's place if it fits, submitLock held, without publishing
// it: false where the ring has no room for it.
bool ReimsGraphics::writeLocked(UInt32 type, const void *message, UInt32 size, UInt64 *seqno,
                                UInt64 *waiterSeqno)
{
	volatile UInt8 *header = ring(RGFX_CONTROL_SUBMIT_OFFSET);
	volatile UInt8 *bytes = header + RGFX_RING_HEADER_SIZE;
	UInt64 used = produced - load64(header + kRingConsumer);
	UInt32 offset = static_cast<UInt32>(produced) & kRingMask;
	UInt32 tail = kRingBytes - offset;
	UInt32 pad = tail < size ? tail : 0;
	if (used > kRingBytes || kRingBytes - used < static_cast<UInt64>(pad) + size)
		return false;
	if (pad) {
		// A message never wraps: type 0 pads to the ring's end. When 8
		// bytes are left, they hold its type and size alone.
		volatile UInt32 *p = reinterpret_cast<volatile UInt32 *>(bytes + offset);
		p[0] = RGFX_MSG_PAD;
		p[1] = pad;
		if (pad >= kHeaderBytes)
			*reinterpret_cast<volatile UInt64 *>(bytes + offset + 8) = 0;
		produced += pad;
		offset = 0;
	}
	UInt64 s = nextSeqno++;
	if (waiterSeqno)
		__atomic_store_n(waiterSeqno, s, __ATOMIC_RELEASE);
	// The caller filled the payload fields; the header is stamped here.
	RgfxMsgHeader h = {type, size, s};
	UInt64 words[2];
	memcpy(words, &h, sizeof(words));
	volatile UInt64 *dst = reinterpret_cast<volatile UInt64 *>(bytes + offset);
	dst[0] = words[0];
	dst[1] = words[1];
	const UInt8 *src = static_cast<const UInt8 *>(message);
	for (UInt32 i = kHeaderBytes / 8; i < size / 8; i++) {
		UInt64 w;
		memcpy(&w, src + i * 8, sizeof(w));
		dst[i] = w;
	}
	produced += size;
	if (seqno)
		*seqno = s;
	return true;
}

// Publishes what writeLocked wrote, submitLock held: the doorbell rings only when the host's
// consumer sleeps.
void ReimsGraphics::publishLocked()
{
	volatile UInt8 *header = ring(RGFX_CONTROL_SUBMIT_OFFSET);
	store64(header + kRingProducer, produced);
	fullFence();
	if (load32(header + kRingConsumerIdle) == 1)
		writeRegister(RGFX_REG_DOORBELL, RGFX_CONTROL_RING_ID);
}

// Puts one command on ring 0, waiting for space (waitForRoom): thread context only. A message
// that can never fit is refused at once.
IOReturn ReimsGraphics::post(UInt32 type, const void *message, UInt32 size, UInt64 *seqno,
                             UInt64 *waiterSeqno)
{
	if (size < kHeaderBytes || size % kMsgAlign || size > kLifecycleMessageMax)
		return kIOReturnBadArgument;
	uint64_t deadline = 0;
	clock_interval_to_deadline(kPostTimeoutMs, kMillisecondScale, &deadline);
	for (;;) {
		const UInt64 mark = roomMark();
		if (tryPost(type, message, size, seqno, waiterSeqno))
			return kIOReturnSuccess;
		if (!ready)
			return kIOReturnNotReady;
		if (askForRoom() && tryPost(type, message, size, seqno, waiterSeqno))
			return kIOReturnSuccess;
		if (!waitForRoom(mark, deadline)) {
			IOLog("ReimsGraphics: command %u: the control ring stayed full\n", type);
			return ready ? kIOReturnTimeout : kIOReturnNotReady;
		}
	}
}

UInt64 ReimsGraphics::roomMark() const
{
	return rgfxRingRoomMark(room);
}

bool ReimsGraphics::askForRoom()
{
	if (!submitRoom)
		return false;
	store32(ring(RGFX_CONTROL_SUBMIT_OFFSET) + kRingProducerWaiting, 1);
	fullFence();
	return true;
}

bool ReimsGraphics::waitForRoom(UInt64 mark, uint64_t deadline)
{
	uint64_t backstop = 0;
	nanoseconds_to_absolutetime(static_cast<uint64_t>(rgfxRingRoomBackstopUs(submitRoom)) * 1000,
	                            &backstop);
	rgfxRingRoomWait(room);
	bool log = false;
	RgfxRingRoomStats seen = {};
	IOLockLock(stateLock);
	if (ready && rgfxRingRoomSleeps(room, mark)) {
		const uint64_t from = mach_absolute_time();
		const wait_result_t r = IOLockSleepDeadline(
		    stateLock, &room, rgfxRingRoomSleepUntil(from, backstop, deadline), THREAD_UNINT);
		uint64_t ns = 0, nowNs = 0;
		const uint64_t to = mach_absolute_time();
		absolutetime_to_nanoseconds(to - from, &ns);
		absolutetime_to_nanoseconds(to, &nowNs);
		log = rgfxRingRoomNote(roomStats, ns, r == THREAD_AWAKENED, nowNs,
		                       static_cast<uint64_t>(kRoomLogMs) * 1000000ull);
		seen = roomStats;
	}
	IOLockUnlock(stateLock);
	rgfxRingRoomDone(room);
	if (log)
		IOLog("ReimsGraphics: control ring room: %llu sleeps (%llu woken by a pass), %llu us "
		      "asleep in all, %llu us at most; %llu SUBMIT_ROOM records taken\n",
		      seen.sleeps, seen.woken, seen.sleptNs / 1000, seen.mostNs / 1000,
		      __atomic_load_n(&roomRecords, __ATOMIC_RELAXED));
	return ready && mach_absolute_time() < deadline;
}

// Sends a display-control command and waits for its answer, COMPLETE or DISPLAYS, which the
// work loop hands over: thread context only, never on the work loop.
IOReturn ReimsGraphics::command(UInt32 type, const void *message, UInt32 size)
{
	uint64_t deadline = 0;
	clock_interval_to_deadline(kAnswerTimeoutMs, kMillisecondScale, &deadline);
	// Take a waiter first, so the answer finds this thread by seqno once the message is on the
	// ring; a free one frees up with the shared array as its event.
	ControlWaiter *w = nullptr;
	IOLockLock(stateLock);
	while (!w) {
		for (UInt32 i = 0; i < kControlWaiters && !w; i++) {
			if (!controlWaiters[i].waiting)
				w = &controlWaiters[i];
		}
		if (!w && IOLockSleepDeadline(stateLock, controlWaiters, deadline, THREAD_UNINT) ==
		              THREAD_TIMED_OUT)
			break;
	}
	if (!w) {
		IOLockUnlock(stateLock);
		IOLog("ReimsGraphics: command %u: every control waiter is taken\n", type);
		return kIOReturnBusy;
	}
	w->waiting = true;
	w->seqno = 0;
	w->status = 0;
	w->done = false;
	IOLockUnlock(stateLock);

	UInt64 s = 0;
	IOReturn r = post(type, message, size, &s, &w->seqno);
	IOLockLock(stateLock);
	while (r == kIOReturnSuccess && !w->done) {
		if (IOLockSleepDeadline(stateLock, &w->done, deadline, THREAD_UNINT) ==
		    THREAD_TIMED_OUT && !w->done)
			r = kIOReturnTimeout;
	}
	UInt32 status = w->status;
	w->waiting = false;
	w->seqno = 0;
	// A slot freed: the free event is the shared array, the slot seekers'.
	IOLockWakeup(stateLock, controlWaiters, false);
	IOLockUnlock(stateLock);

	if (r == kIOReturnTimeout) {
		IOLog("ReimsGraphics: command %u (seqno %llu): no answer\n", type, s);
		return kIOReturnTimeout;
	}
	if (r != kIOReturnSuccess)
		return r;
	if (status != 0) {
		IOLog("ReimsGraphics: command %u (seqno %llu): status %u\n", type, s, status);
		return kIOReturnError;
	}
	return kIOReturnSuccess;
}

IOReturn ReimsGraphics::postLifecycle(UInt32 type, const void *message, UInt32 size,
                                      UInt32 replyType, void *reply, UInt32 replyCapacity,
                                      UInt32 *replyLength, UInt64 *seqno, UInt32 timeoutMs)
{
	if (replyLength)
		*replyLength = 0;
	if (!replyType)
		return post(type, message, size, seqno);

	// Take a waiter, so the answer finds this thread once the message is on the ring.
	uint64_t deadline = 0;
	clock_interval_to_deadline(timeoutMs, kMillisecondScale, &deadline);
	LifecycleWaiter *w = nullptr;
	IOLockLock(stateLock);
	while (!w) {
		for (UInt32 i = 0; i < kLifecycleWaiters && !w; i++) {
			if (lifecycleWaiters[i].replyType == 0)
				w = &lifecycleWaiters[i];
		}
		if (!w && IOLockSleepDeadline(stateLock, lifecycleWaiters, deadline, THREAD_UNINT) ==
		              THREAD_TIMED_OUT)
			break;
	}
	if (!w) {
		IOLockUnlock(stateLock);
		IOLog("ReimsGraphics: lifecycle %u: every waiter is taken\n", type);
		return kIOReturnBusy;
	}
	w->seqno = 0;
	w->replyType = replyType;
	w->length = 0;
	w->done = false;
	IOLockUnlock(stateLock);

	UInt64 s = 0;
	IOReturn r = post(type, message, size, &s, &w->seqno);
	IOLockLock(stateLock);
	// Each waiter sleeps on its own `done`, so an answer wakes only the thread it is for, not
	// every waiter (which, with many in flight, would herd on stateLock and starve the work loop
	// that delivers the answers). The free event below stays the shared array, a slot's seekers'.
	while (r == kIOReturnSuccess && !w->done) {
		if (IOLockSleepDeadline(stateLock, &w->done, deadline, THREAD_UNINT) ==
		    THREAD_TIMED_OUT && !w->done)
			r = kIOReturnTimeout;
	}
	if (r == kIOReturnSuccess) {
		UInt32 n = w->length < replyCapacity ? w->length : replyCapacity;
		memcpy(reply, w->reply, n);
		if (replyLength)
			*replyLength = n;
	}
	w->replyType = 0;
	w->seqno = 0;
	IOLockWakeup(stateLock, lifecycleWaiters, false);
	IOLockUnlock(stateLock);
	if (r == kIOReturnTimeout)
		IOLog("ReimsGraphics: lifecycle %u (seqno %llu): no answer\n", type, s);
	if (seqno)
		*seqno = s;
	return r;
}

IOReturn ReimsGraphics::postDeferred(UInt32 context, UInt32 type, const void *message,
                                     UInt32 size, UInt64 *seqno)
{
	IOLockLock(stateLock);
	const int slot = deferredAnswers.reserve(context);
	IOLockUnlock(stateLock);
	if (slot < 0)
		return kIOReturnNoResources;
	UInt64 s = 0;
	// The slot's seqno is stamped before the device can see the create, so its answer finds it.
	IOReturn r = post(type, message, size, &s, deferredAnswers.seqnoOf(slot));
	if (r != kIOReturnSuccess) {
		IOLockLock(stateLock);
		deferredAnswers.cancel(slot);
		IOLockUnlock(stateLock);
	}
	if (seqno)
		*seqno = s;
	return r;
}

IOReturn ReimsGraphics::takeDeferred(UInt32 context, UInt64 waitFor, UInt32 timeoutMs, UInt8 *out,
                                     UInt32 capacity, UInt32 *length, UInt32 *count)
{
	uint64_t deadline = 0;
	clock_interval_to_deadline(timeoutMs, kMillisecondScale, &deadline);
	IOReturn r = kIOReturnSuccess;
	IOLockLock(stateLock);
	while (waitFor != 0 &&
	       deferredAnswers.state(context, waitFor) == DeferredAnswers::kWaiting) {
		if (IOLockSleepDeadline(stateLock, &deferredAnswers, deadline, THREAD_UNINT) ==
		        THREAD_TIMED_OUT &&
		    deferredAnswers.state(context, waitFor) == DeferredAnswers::kWaiting) {
			r = kIOReturnTimeout;
			break;
		}
	}
	*length = deferredAnswers.take(context, out, capacity, count);
	IOLockUnlock(stateLock);
	if (r == kIOReturnTimeout)
		IOLog("ReimsGraphics: context %u: no answer to its deferred create (seqno %llu)\n",
		      context, waitFor);
	return r;
}

void ReimsGraphics::dropDeferred(UInt32 context)
{
	IOLockLock(stateLock);
	deferredAnswers.drop(context);
	IOLockWakeup(stateLock, &deferredAnswers, false);
	IOLockUnlock(stateLock);
}

void ReimsGraphics::forgetDeferred(UInt64 seqno)
{
	IOLockLock(stateLock);
	deferredAnswers.forget(seqno);
	IOLockUnlock(stateLock);
}

void ReimsGraphics::kickWorkLoop()
{
	if (kick)
		kick->interruptOccurred(nullptr, nullptr, 0);
}

// MARK: - The work loop

void ReimsGraphics::serviceAction(OSObject *owner, IOInterruptEventSource *, int)
{
	static_cast<ReimsGraphics *>(owner)->service();
}

// Drains the completion ring, then arms its interrupt and looks again, so an
// event published meanwhile is never left without one. Then sends what
// IOFramebuffer asked of vblank, and a display refresh a DISPLAY_CHANGED
// asked for, neither of which may wait for ring space here: the host may be
// waiting for this loop to free the completion ring.
void ReimsGraphics::service()
{
	if (!ready)
		return;
	volatile UInt8 *events = ring(RGFX_CONTROL_COMPLETE_OFFSET);
	for (;;) {
		drainEvents();
		store32(events + kRingArmed, 1);
		fullFence();
		if (load64(events + kRingProducer) == consumed)
			break;
	}
	// The host has moved on: a post waiting for room on the control ring looks again.
	if (rgfxRingRoomPass(room)) {
		IOLockLock(stateLock);
		IOLockWakeup(stateLock, &room, false);
		IOLockUnlock(stateLock);
	}

	// The queues' completions share the vector: the accelerator looks at its rings.
	IOSimpleLockLock(interruptLock);
	OSObject *queues = queueTarget;
	QueueInterruptHook queuesHook = queueHook;
	if (queues)
		queues->retain();
	IOSimpleLockUnlock(interruptLock);
	if (queues) {
		queuesHook(queues);
		queues->release();
	}

	IOSimpleLockLock(interruptLock);
	Interrupt late = {};
	if (connectChange.armed(connect.enabled && connect.proc))
		late = connect;
	bool want = vblankWanted || pipes.anyVblank();
	IOSimpleLockUnlock(interruptLock);
	if (late.proc)
		late.proc(late.target, late.ref);
	if (want != vblankSent) {
		RgfxSetVblank v = {};
		v.display = displayId;
		v.enabled = want ? 1u : 0u;
		if (tryPost(RGFX_CMD_SET_VBLANK, &v, sizeof(v), nullptr))
			vblankSent = want;
	}
	if (refreshWanted) {
		UInt64 seqno = 0;
		RgfxGetDisplays getDisplays = {};
		if (tryPost(RGFX_CMD_GET_DISPLAYS, &getDisplays, sizeof(getDisplays), &seqno)) {
			refreshWanted = false;
			refreshSeqno = seqno;
		}
	}
}

bool ReimsGraphics::drainEvents()
{
	volatile UInt8 *header = ring(RGFX_CONTROL_COMPLETE_OFFSET);
	volatile UInt8 *bytes = header + RGFX_RING_HEADER_SIZE;
	UInt64 end = load64(header + kRingProducer);
	bool any = consumed != end;
	while (consumed != end) {
		if (end - consumed > kRingBytes) {
			IOLog("ReimsGraphics: the completion ring's producer is out of range; skipping to it\n");
			consumed = end;
			break;
		}
		UInt32 offset = static_cast<UInt32>(consumed % kRingBytes);
		UInt32 tail = kRingBytes - offset;
		if (tail < kHeaderBytes) {
			// Too few bytes for a header: the producer padded to the end.
			consumed += tail;
			continue;
		}
		const volatile UInt8 *m = bytes + offset;
		UInt32 type = field32(m, 0);
		UInt32 size = field32(m, 4);
		UInt64 seqno = *reinterpret_cast<const volatile UInt64 *>(m + 8);
		if (type == RGFX_MSG_PAD) {
			consumed += tail;
			continue;
		}
		if (size < kHeaderBytes || size % kMsgAlign || size > tail || size > end - consumed) {
			IOLog("ReimsGraphics: malformed event (type %u, size %u); skipping the ring's contents\n",
			      type, size);
			consumed = end;
			break;
		}
		handleEvent(type, seqno, m + kHeaderBytes, size - kHeaderBytes);
		consumed += size;
	}
	store64(header + kRingConsumer, consumed);
	return any;
}

void ReimsGraphics::handleEvent(UInt32 type, UInt64 seqno, const volatile UInt8 *payload, UInt32 size)
{
	Interrupt fire = {};
	switch (type) {
	case RGFX_EVT_COMPLETE: {
		if (answerLifecycle(type, seqno, payload, size))
			return;
		UInt32 status = size >= sizeof(uint32_t) ? field32(payload, 0) : kStatusUnreadable;
		// A block release's, which no thread waits for: the accelerator's.
		IOSimpleLockLock(interruptLock);
		OSObject *releases = releaseTarget;
		ReleaseAnswerHook releasesHook = releaseHook;
		if (releases)
			releases->retain();
		IOSimpleLockUnlock(interruptLock);
		if (releases) {
			const bool released = releasesHook(releases, seqno, status);
			releases->release();
			if (released)
				return;
		}
		answer(seqno, status);
		// Every pipe client: the one whose transaction this answers takes it, at the time of
		// the refresh the host answered in.
		UInt64 shown = vblankClock.shownAt(mach_absolute_time());
		OSObject *clients[kMaxPipeClients];
		IOSimpleLockLock(interruptLock);
		UInt32 n = pipes.all(clients);
		PipePresentHook hook = pipePresentHook;
		for (UInt32 i = 0; i < n; i++)
			clients[i]->retain();
		IOSimpleLockUnlock(interruptLock);
		for (UInt32 i = 0; i < n; i++) {
			if (hook)
				hook(clients[i], seqno, status, shown);
			clients[i]->release();
		}
		return;
	}
	case RGFX_EVT_DISPLAYS: {
		answer(seqno, parseDisplays(payload, size));
		if (seqno != refreshSeqno)
			return;
		// The refresh a DISPLAY_CHANGED asked for: now IOFramebuffer
		// senses the connection again, or once it has its connect interrupt
		// registered and on (RgfxDisplayModes.hpp, RgfxConnectChange).
		refreshSeqno = 0;
		IOSimpleLockLock(interruptLock);
		if (connectChange.changed(connect.enabled && connect.proc))
			fire = connect;
		IOSimpleLockUnlock(interruptLock);
		break;
	}
	case RGFX_EVT_DISPLAY_CHANGED:
		refreshWanted = true;
		return;
	case RGFX_EVT_SUBMIT_ROOM:
		// The host took from the control ring after a post set its producer_waiting: the pass
		// that drains this wakes the posts waiting for room (service). Counted for the room line.
		__atomic_fetch_add(&roomRecords, 1, __ATOMIC_RELAXED);
		return;
	case RGFX_EVT_PIPELINE_REFLECTION:
	case RGFX_EVT_SIZE_AND_ALIGN:
	case RGFX_EVT_SAMPLER_STATE:
	case RGFX_EVT_ICB_STATE:
		answerLifecycle(type, seqno, payload, size);
		return;
	case RGFX_EVT_VBLANK: {
		OSObject *clients[kMaxPipeClients];
		IOSimpleLockLock(interruptLock);
		fire = vblank;
		UInt32 n = pipes.wantingVblank(clients);
		PipeVblankHook hook = pipeVblankHook;
		for (UInt32 i = 0; i < n; i++)
			clients[i]->retain();
		IOSimpleLockUnlock(interruptLock);
		// The event's count and the host's stamp are 8 and 16 bytes into its payload
		// (RgfxVblank); the refresh's time is the stamp in the guest's clock.
		UInt64 count = size >= 16 ? *reinterpret_cast<const volatile UInt64 *>(payload + 8) : 0;
		UInt64 now = mach_absolute_time();
		UInt64 time = size >= 24 ? vblankClock.vblank(count,
		                                              *reinterpret_cast<const volatile UInt64 *>(payload + 16),
		                                              now)
		                         : now;
		UInt64 next = size >= 24 ? vblankClock.next() : now + RgfxVblankClock::kDefaultPeriodNs;
		for (UInt32 i = 0; i < n; i++) {
			if (hook)
				hook(clients[i], count, time, next);
			clients[i]->release();
		}
		break;
	}
	default:
		return;
	}
	if (fire.enabled && fire.proc)
		fire.proc(fire.target, fire.ref);
}

// Hands a lifecycle message's answer to its waiting thread, or keeps a deferred create's for its
// process: false where neither waits for an answer of this type to `seqno`.
bool ReimsGraphics::answerLifecycle(UInt32 type, UInt64 seqno, const volatile UInt8 *payload,
                                    UInt32 size)
{
	if (seqno == 0)
		return false;
	bool found = false;
	IOLockLock(stateLock);
	for (UInt32 i = 0; i < kLifecycleWaiters && !found; i++) {
		LifecycleWaiter &w = lifecycleWaiters[i];
		if (w.replyType != type || w.done ||
		    __atomic_load_n(&w.seqno, __ATOMIC_ACQUIRE) != seqno)
			continue;
		found = true;
		RgfxMsgHeader h = {type, size + kHeaderBytes, seqno};
		UInt32 n = size + kHeaderBytes < kLifecycleReplyMax ? size + kHeaderBytes
		                                                    : kLifecycleReplyMax;
		memcpy(w.reply, &h, kHeaderBytes);
		for (UInt32 b = kHeaderBytes; b < n; b++)
			w.reply[b] = payload[b - kHeaderBytes];
		w.length = n;
		w.done = true;
		// Only the thread this answer is for: its own `done` event, one thread.
		IOLockWakeup(stateLock, &w.done, true);
	}
	// Else a deferred create's (postDeferred), kept for its process to take; its waiters, if any,
	// look again.
	if (!found && deferredAnswers.answer(RgfxMsgHeader{type, size + kHeaderBytes, seqno},
	                                     payload, size)) {
		found = true;
		IOLockWakeup(stateLock, &deferredAnswers, false);
	}
	IOLockUnlock(stateLock);
	return found;
}

// Hands a display-control answer (a COMPLETE or a DISPLAYS) to its command() thread, matched by
// seqno. A COMPLETE no command() waits for (a PRESENT_SURFACE's, which the pipe clients take
// instead) matches no waiter and is dropped.
void ReimsGraphics::answer(UInt64 seqno, UInt32 status)
{
	if (seqno == 0)
		return;
	IOLockLock(stateLock);
	for (UInt32 i = 0; i < kControlWaiters; i++) {
		ControlWaiter &w = controlWaiters[i];
		if (w.waiting && !w.done && __atomic_load_n(&w.seqno, __ATOMIC_ACQUIRE) == seqno) {
			w.status = status;
			w.done = true;
			// Only the thread this answer is for: its own `done` event, one thread.
			IOLockWakeup(stateLock, &w.done, true);
			break;
		}
	}
	IOLockUnlock(stateLock);
}

// Takes the display's modes from a DISPLAYS answer. The driver's display is
// the first listed when it starts, and the display of that id after; a list
// without it leaves the display disconnected with its modes. Modes the
// scanout space cannot hold are left out. A list that changes the modes makes
// the next server connection drop the mode configuration the last one left
// (newUserClient). Returns the answer's status.
UInt32 ReimsGraphics::parseDisplays(const volatile UInt8 *payload, UInt32 size)
{
	if (size < sizeof(uint32_t))
		return kStatusUnreadable;
	UInt32 count = field32(payload, 0);
	// The first display record follows count, past the message header.
	UInt32 offset = RGFX_DISPLAYS_ENTRIES_OFFSET - kHeaderBytes;

	IOLockLock(stateLock);
	bool found = false;
	UInt32 id = 0, isConnected = 0, preferred = 0, listed = 0, listAt = 0;
	for (UInt32 i = 0; i < count; i++) {
		if (size - offset < sizeof(RgfxDisplayEntry)) {
			IOLockUnlock(stateLock);
			return kStatusUnreadable;
		}
		UInt32 did = field32(payload, offset);
		UInt32 n = field32(payload, offset + 12);
		UInt32 at = offset + sizeof(RgfxDisplayEntry);
		if (static_cast<UInt64>(n) * sizeof(RgfxModeEntry) > size - at) {
			IOLockUnlock(stateLock);
			return kStatusUnreadable;
		}
		if (!found && (!haveDisplay || did == displayId)) {
			found = true;
			id = did;
			isConnected = field32(payload, offset + 4);
			preferred = field32(payload, offset + 8);
			listed = n;
			listAt = at;
		}
		offset = at + n * sizeof(RgfxModeEntry);
	}
	if (!found) {
		connected = false;
		IOLockUnlock(stateLock);
		return 0;
	}

	Mode fresh[kMaxModes];
	UInt32 freshCount = 0, freshPreferred = kMaxModes;
	UInt64 limit = scanoutLimit();
	for (UInt32 j = 0; j < listed && freshCount < kMaxModes; j++) {
		UInt32 at = listAt + j * sizeof(RgfxModeEntry);
		Mode m = {0, field32(payload, at), field32(payload, at + 4), field32(payload, at + 8)};
		if (!m.width || !m.height || m.width > kMaxDimension || m.height > kMaxDimension ||
		    static_cast<UInt64>(strideFor(m.width)) * m.height > limit) {
			IOLog("ReimsGraphics: display %u: mode %ux%u left out: the region cannot hold it\n",
			      id, m.width, m.height);
			continue;
		}
		if (j == preferred)
			freshPreferred = freshCount;
		fresh[freshCount++] = m;
	}
	haveDisplay = true;
	displayId = id;
	connected = isConnected != 0;
	if (modeSet.take(fresh, freshCount, freshPreferred))
		modeConfig.listChanged();
	IOLockUnlock(stateLock);
	return 0;
}

// MARK: - Modes and scanout

UInt32 ReimsGraphics::refreshMhzOf(const IODetailedTimingInformationV2 &t)
{
	UInt64 total = static_cast<UInt64>(t.horizontalActive + t.horizontalBlanking) *
	               (t.verticalActive + t.verticalBlanking);
	return total ? static_cast<UInt32>((t.pixelClock * 1000 + total / 2) / total) : 0;
}

void ReimsGraphics::imageSize(const Mode &m, UInt16 *width, UInt16 *height) const
{
	// The EDID's size, from the preferred mode.
	Mode native = m;
	for (UInt32 i = 0; i < modeSet.modeCount; i++) {
		if (modeSet.modes[i].id == modeSet.preferredId)
			native = modeSet.modes[i];
	}
	UInt32 ppi = native.width >= 2880 ? 220 : 110;
	*width = static_cast<UInt16>((native.width * 254 + ppi * 5) / (ppi * 10));
	*height = static_cast<UInt16>((native.height * 254 + ppi * 5) / (ppi * 10));
}

UInt32 ReimsGraphics::strideFor(UInt32 width)
{
	return (width * 4 + kStrideAlign - 1) & ~(kStrideAlign - 1);
}

// The scanout buffer's room: RGFX_SCANOUT_SPAN at the guest's allocations, before the
// accelerator's blocks (RgfxScanoutOwner.hpp), so no mode reaches a process's memory.
UInt64 ReimsGraphics::scanoutLimit() const
{
	return rgfxScanoutRoom(regionSize, kCursorSlot);
}

IOReturn ReimsGraphics::setScanout(const Mode &mode)
{
	UInt32 stride = strideFor(mode.width);
	if (static_cast<UInt64>(stride) * mode.height > scanoutLimit())
		return kIOReturnNoSpace;
	RgfxSetScanout s = {};
	s.display = displayId;
	s.width = mode.width;
	s.height = mode.height;
	s.stride = stride;
	s.offset = RGFX_ALLOCATIONS_OFFSET;
	IOReturn r = command(RGFX_CMD_SET_SCANOUT, &s, sizeof(s));
	if (r != kIOReturnSuccess)
		return r;
	IOLockLock(stateLock);
	modeSet.set(mode);
	IOLockUnlock(stateLock);
	IOLog("ReimsGraphics: display %u: scanout %ux%u at %u.%03u Hz, stride %u\n", displayId,
	      mode.width, mode.height, mode.refreshMhz / 1000, mode.refreshMhz % 1000, stride);
	return kIOReturnSuccess;
}

bool ReimsGraphics::ownsScanout(task_t task)
{
	if (!task || !stateLock)
		return false;
	IOLockLock(stateLock);
	const IOUserClient *client = static_cast<const IOUserClient *>(scanoutOwner.client);
	bool owns = scanoutOwner.owns(task, client && !client->isInactive());
	IOLockUnlock(stateLock);
	return owns;
}

bool ReimsGraphics::currentScanout(UInt32 *width, UInt32 *height, UInt32 *rowBytes,
                                   UInt64 *offset)
{
	if (!stateLock)
		return false;
	IOLockLock(stateLock);
	bool have = modeSet.haveCurrent;
	Mode m = modeSet.current;
	IOLockUnlock(stateLock);
	if (!have)
		return false;
	*width = m.width;
	*height = m.height;
	*rowBytes = strideFor(m.width);
	*offset = RGFX_ALLOCATIONS_OFFSET;
	return true;
}

// MARK: - IOService

// IOFramebuffer makes the connection; a server connection's task is the scanout's owner from
// then on, in place of the one before it, and while the open is in flight
// (RgfxScanoutOwner.hpp).
void ReimsGraphics::scanoutOwnerMoved(task_t owner)
{
	RgfxAccelerator *accelerator = copyAccelerator();
	if (!accelerator)
		return;
	accelerator->scanoutOwnerMoved(owner);
	accelerator->release();
}

RgfxAccelerator *ReimsGraphics::copyAccelerator()
{
	if (!pci)
		return nullptr;
	RgfxAccelerator *accelerator = nullptr;
	if (OSIterator *clients = pci->getClientIterator()) {
		while (OSObject *o = clients->getNextObject()) {
			accelerator = OSDynamicCast(RgfxAccelerator, o);
			if (accelerator) {
				accelerator->retain();
				break;
			}
		}
		clients->release();
	}
	return accelerator;
}

IOReturn ReimsGraphics::newUserClient(task_t owningTask, void *securityID, UInt32 type,
                                      IOUserClient **clientH)
{
	// WindowServer's first server connection settles whether the accelerator registers this
	// boot (RgfxAccelerator::settleBundle), before IOFramebuffer makes the connection and with no
	// lock of ours held.
	if (type == kIOFBServerConnectType) {
		if (RgfxAccelerator *accelerator = copyAccelerator()) {
			accelerator->settleBundle();
			accelerator->release();
		}
	}
	if (type != kIOFBServerConnectType || !stateLock) {
		IOReturn r = super::newUserClient(owningTask, securityID, type, clientH);
		FBTRACE(32, "newUserClient type %u -> 0x%x", type, r);
		return r;
	}
	IOLockLock(stateLock);
	const IOUserClient *held = static_cast<const IOUserClient *>(scanoutOwner.client);
	bool marked = scanoutOwner.beginOpen(owningTask, held && !held->isInactive());
	bool stale = modeConfig.serverOpens();
	IOLockUnlock(stateLock);
	if (stale && getProperty(kIOFBConfigKey)) {
		// Made from a list the device has changed since (RgfxDisplayModes.hpp).
		removeProperty(kIOFBConfigKey);
		IOLog("ReimsGraphics: the display's modes changed since the last server connection: "
		      "its mode configuration is dropped\n");
	}
	IOReturn r = super::newUserClient(owningTask, securityID, type, clientH);
	FBTRACE(32, "newUserClient type %u -> 0x%x", type, r);
	IOUserClient *made = r == kIOReturnSuccess && clientH ? *clientH : nullptr;
	if (made) {
		task_reference(owningTask);
		made->retain();
	}
	const void *oldTask = nullptr, *oldClient = nullptr;
	IOLockLock(stateLock);
	scanoutOwner.endOpen(owningTask, made, marked, &oldTask, &oldClient);
	IOLockUnlock(stateLock);
	if (oldTask)
		task_deallocate(static_cast<task_t>(const_cast<void *>(oldTask)));
	if (oldClient)
		static_cast<const IOUserClient *>(oldClient)->release();
	// The owner moved: every other task's hold of the scanout is taken back.
	if (made)
		scanoutOwnerMoved(owningTask);
	RgfxCaller caller;
	if (made)
		IOLog("ReimsGraphics: the scanout's owner is %s[%d], which holds the server connection\n",
		      caller.name, caller.pid);
	else
		IOLog("ReimsGraphics: %s[%d]'s server connection was not made (0x%x): the scanout's "
		      "owner stays\n",
		      caller.name, caller.pid, r);
	return r;
}

bool ReimsGraphics::start(IOService *provider)
{
	pci = OSDynamicCast(IOPCIDevice, provider);
	if (!pci)
		return false;
	submitLock = IOSimpleLockAlloc();
	interruptLock = IOSimpleLockAlloc();
	stateLock = IOLockAlloc();
	if (!submitLock || !interruptLock || !stateLock)
		return false;
	nextSeqno = 1;
	modeSet.reset();
	taken = false;
	submitRoom = false;
	roomStats = {};
	roomRecords = 0;

	pci->setMemoryEnable(true);
	pci->setBusLeadEnable(true);

	int msixIndex = -1;
	IOReturn r = kIOReturnSuccess;
	UInt32 count = 0;
	RgfxGetDisplays getDisplays = {};

	registerMap = pci->mapDeviceMemoryWithRegister(kIOPCIConfigBaseAddress0);
	if (!registerMap) {
		IOLog("ReimsGraphics: BAR0 not mapped\n");
		goto fail;
	}
	registers = reinterpret_cast<volatile UInt8 *>(registerMap->getVirtualAddress());
	if (readRegister(RGFX_REG_MAGIC) != RGFX_MAGIC) {
		IOLog("ReimsGraphics: not the device: magic 0x%x\n", readRegister(RGFX_REG_MAGIC));
		goto fail;
	}
	// The wire's version names the region's layout and the messages (design.md, "The wire's
	// version"): a kext of another version misreads them, so it leaves the device, writing
	// nothing to it.
	if (readRegister(RGFX_REG_VERSION) != RGFX_VERSION) {
		IOLog("ReimsGraphics: the device speaks wire version %u, this kext version %u: not starting. "
		      "The kext and the host must come from one build.\n",
		      readRegister(RGFX_REG_VERSION), RGFX_VERSION);
		goto fail;
	}

	region = pci->getDeviceMemoryWithRegister(kIOPCIConfigBaseAddress2);
	if (!region) {
		IOLog("ReimsGraphics: no BAR2\n");
		goto fail;
	}
	region->retain();
	regionSize = (static_cast<UInt64>(readRegister(RGFX_REG_REGION_SIZE_HI)) << 32) |
	             readRegister(RGFX_REG_REGION_SIZE_LO);
	if (regionSize > region->getLength())
		regionSize = region->getLength();
	if (regionSize <= RGFX_ALLOCATIONS_OFFSET + kCursorSlot) {
		IOLog("ReimsGraphics: the region is too small: 0x%llx bytes\n", regionSize);
		goto fail;
	}
	ringMap = region->createMappingInTask(kernel_task, 0, kIOMapAnywhere | kIOMapCopybackCache, 0,
	                                      RGFX_ALLOCATIONS_OFFSET);
	cursorMap = region->createMappingInTask(kernel_task, 0, kIOMapAnywhere | kIOMapCopybackCache,
	                                        regionSize - kCursorSlot, kCursorSlot);
	if (!ringMap || !cursorMap) {
		IOLog("ReimsGraphics: BAR2 not mapped\n");
		goto fail;
	}
	rings = reinterpret_cast<volatile UInt8 *>(ringMap->getVirtualAddress());
	cursorPixels = reinterpret_cast<UInt8 *>(cursorMap->getVirtualAddress());

	// MSI-X vector 0 is the device's first message-signalled interrupt.
	for (int i = 0;; i++) {
		int type = 0;
		if (pci->getInterruptType(i, &type) != kIOReturnSuccess)
			break;
		if (type & (kIOInterruptTypePCIMessaged | kIOInterruptTypePCIMessagedX)) {
			msixIndex = i;
			break;
		}
	}
	if (msixIndex < 0) {
		IOLog("ReimsGraphics: no MSI-X interrupt\n");
		goto fail;
	}
	workLoop = IOWorkLoop::workLoop();
	if (workLoop) {
		msix = IOInterruptEventSource::interruptEventSource(this, serviceAction, pci, msixIndex);
		kick = IOInterruptEventSource::interruptEventSource(this, serviceAction);
	}
	if (!msix || !kick || workLoop->addEventSource(msix) != kIOReturnSuccess ||
	    workLoop->addEventSource(kick) != kIOReturnSuccess) {
		IOLog("ReimsGraphics: no work loop\n");
		goto fail;
	}

	// The device from scratch: reset, then the features the kext takes of the host's offer
	// (RgfxFeatures.hpp), then take the rings' counters as they stand and say the driver is
	// ready. Every refusal above leaves the device untouched; from here on it is ours, and it is
	// reset again when the kext lets it go.
	taken = true;
	writeRegister(RGFX_REG_STATUS, RGFX_STATUS_RESET);
	{
		const uint64_t offered = (static_cast<uint64_t>(readRegister(RGFX_REG_FEATURES_HI)) << 32) |
		                         readRegister(RGFX_REG_FEATURES_LO);
		const uint64_t wanted = rgfxDriverFeatures(offered);
		writeRegister(RGFX_REG_DRIVER_FEATURES_LO, static_cast<UInt32>(wanted));
		writeRegister(RGFX_REG_DRIVER_FEATURES_HI, static_cast<UInt32>(wanted >> 32));
		const uint64_t took =
		    (static_cast<uint64_t>(readRegister(RGFX_REG_DRIVER_FEATURES_HI)) << 32) |
		    readRegister(RGFX_REG_DRIVER_FEATURES_LO);
		IOLog("ReimsGraphics: features offered 0x%llx, taken 0x%llx of 0x%llx\n", offered, took,
		      wanted);
		// The same three words on the registry, where the guest's tools read them at any time:
		// the kernel's message buffer has let go of the line by the time a login runs.
		setProperty("RgfxFeaturesOffered", offered, 64);
		setProperty("RgfxFeaturesWanted", wanted, 64);
		setProperty("RgfxFeaturesTaken", took, 64);
		submitRoom = (took & RGFX_FEATURE_SUBMIT_ROOM) != 0;
	}
	produced = load64(ring(RGFX_CONTROL_SUBMIT_OFFSET) + kRingProducer);
	consumed = load64(ring(RGFX_CONTROL_COMPLETE_OFFSET) + kRingConsumer);
	writeRegister(RGFX_REG_STATUS, RGFX_STATUS_DRIVER_READY);
	ready = true;
	msix->enable();
	kick->enable();
	kickWorkLoop();

	r = command(RGFX_CMD_GET_DISPLAYS, &getDisplays, sizeof(getDisplays));
	IOLockLock(stateLock);
	count = haveDisplay ? modeSet.modeCount : 0;
	IOLockUnlock(stateLock);
	if (r != kIOReturnSuccess || !count) {
		IOLog("ReimsGraphics: the device lists no usable display\n");
		goto fail;
	}
	IOLog("ReimsGraphics: display %u, %u modes, region 0x%llx bytes\n", displayId, count, regionSize);

	// The display's accelerator and pipe, which the wrangler, CoreDisplay and IOPresentment read
	// off the framebuffer when it is published: our one display is pipe 0 of the accelerator on
	// the same PCI device, named by its service-plane path, which is the PCI device's path and
	// the accelerator's class name whichever of the two starts first.
	{
		char path[512];
		int length = sizeof(path) - sizeof("/RgfxAccelerator");
		if (pci->getPath(path, &length, gIOServicePlane)) {
			strlcat(path, "/RgfxAccelerator", sizeof(path));
			setProperty("IOAccelTypes", path);
		}
	}
	setProperty("IOAccelIndex", 0ull, 32);

	// The framebuffer is its own controller's one dependent (index 0), named by its own registry
	// ID. IOFramebuffer finishes a framebuffer's open (postOpen, in IOGraphics' open source) only
	// for one with a dependent ID: there it publishes kIOFBNeedsRefreshKey, initialises the
	// scanout (initFB) and tests the hardware cursor, which publishes kIOFBCursorInfoKey.
	setProperty(kIOFBDependentIDKey, getRegistryEntryID(), 64);
	setProperty(kIOFBDependentIndexKey, 0ull, 32);

	// The timings the OS may program (validateDetailedTiming takes only the device's modes
	// among them). The scanout has no signal of its own, so any sync and level goes, and
	// nothing optional (interlace, DSC: a version 1 range); no link limits are given, as no
	// link carries the pixels. RGB at 8 bits, in one-pixel steps.
	{
		IODisplayTimingRange range;
		bzero(&range, sizeof(range));
		range.version = kIOTimingRangeV1;
		range.minPixelClock = 1000000;
		range.maxPixelClock = 100000000000ull;
		range.supportedSyncFlags = kIORangeSupportsSeparateSyncs | kIORangeSupportsSyncOnGreen |
		                           kIORangeSupportsCompositeSync | kIORangeSupportsVSyncSerration;
		range.supportedSignalLevels = kIORangeSupportsSignal_0700_0300 | kIORangeSupportsSignal_0714_0286 |
		                              kIORangeSupportsSignal_1000_0400 | kIORangeSupportsSignal_0700_0000;
		range.minFrameRate = 1;
		range.maxFrameRate = 1000;
		range.minLineRate = 1;
		range.maxLineRate = 100000000;
		range.maxHorizontalTotal = 2 * kMaxDimension;
		range.maxVerticalTotal = 2 * kMaxDimension;
		range.charSizeHorizontalActive = 1;
		range.charSizeHorizontalBlanking = 1;
		range.charSizeHorizontalSyncOffset = 1;
		range.charSizeHorizontalSyncPulse = 1;
		range.charSizeVerticalActive = 1;
		range.charSizeVerticalBlanking = 1;
		range.charSizeVerticalSyncOffset = 1;
		range.charSizeVerticalSyncPulse = 1;
		range.charSizeHorizontalBorderLeft = 1;
		range.charSizeHorizontalBorderRight = 1;
		range.charSizeVerticalBorderTop = 1;
		range.charSizeVerticalBorderBottom = 1;
		range.charSizeHorizontalTotal = 1;
		range.charSizeVerticalTotal = 1;
		range.minHorizontalActiveClocks = 1;
		range.maxHorizontalActiveClocks = kMaxDimension;
		range.maxHorizontalBlankingClocks = kMaxDimension;
		range.maxHorizontalSyncOffsetClocks = kMaxDimension;
		range.maxHorizontalPulseWidthClocks = kMaxDimension;
		range.minVerticalActiveClocks = 1;
		range.maxVerticalActiveClocks = kMaxDimension;
		range.maxVerticalBlankingClocks = kMaxDimension;
		range.maxVerticalSyncOffsetClocks = kMaxDimension;
		range.maxVerticalPulseWidthClocks = kMaxDimension;
		range.supportedPixelEncoding = kIORangePixelEncodingRGB444;
		range.supportedBitsPerColorComponent = kIORangeBitsPerColorComponent8;
		range.supportedColorimetryModes = kIORangeColorimetryNativeRGB | kIORangeColorimetrysRGB;
		range.supportedDynamicRangeModes = kIORangeDynamicRangeSDR;
		OSData *data = OSData::withBytes(&range, sizeof(range));
		if (data) {
			setProperty(kIOFBTimingRangeKey, data);
			data->release();
		}
	}

	// The class name user space reads for the framebuffer: IOObjectCopyClass and IOObjectGetClass
	// answer this property where the service carries it. CoreDisplay takes the display's GPU
	// vendor from that name, searching it for the vendor names it knows, and keeps WindowServer's
	// accelerated, transactional display model (kIOWindowServerActiveAttribute) off for a name in
	// which it finds none, as it does for "ReimsGraphics".
	setProperty("IOClassNameOverride", "ReimsGraphicsAMD");

	if (super::start(provider))
		return true;
fail:
	quiesce();
	teardown();
	return false;
}

void ReimsGraphics::stop(IOService *provider)
{
	quiesce();
	super::stop(provider);
	teardown();
}

// No more commands, and no work loop action left running: removing an event
// source waits for its action.
void ReimsGraphics::quiesce()
{
	ready = false;
	if (!workLoop)
		return;
	if (msix) {
		msix->disable();
		workLoop->removeEventSource(msix);
	}
	if (kick) {
		kick->disable();
		workLoop->removeEventSource(kick);
	}
}

// A start refused before it took the device leaves it unreset: a reset ends the firmware's scanout,
// and the host's screen would keep the firmware's last frame under the boot framebuffer driver the
// OS falls back to.
void ReimsGraphics::teardown()
{
	if (registers && taken)
		writeRegister(RGFX_REG_STATUS, RGFX_STATUS_RESET);
	taken = false;
	OSSafeReleaseNULL(msix);
	OSSafeReleaseNULL(kick);
	OSSafeReleaseNULL(workLoop);
	OSSafeReleaseNULL(cursorMap);
	OSSafeReleaseNULL(ringMap);
	OSSafeReleaseNULL(region);
	OSSafeReleaseNULL(registerMap);
	rings = nullptr;
	cursorPixels = nullptr;
	registers = nullptr;
}

void ReimsGraphics::free()
{
	if (submitLock)
		IOSimpleLockFree(submitLock);
	if (interruptLock)
		IOSimpleLockFree(interruptLock);
	if (stateLock)
		IOLockFree(stateLock);
	submitLock = interruptLock = nullptr;
	stateLock = nullptr;
	const void *oldTask = nullptr, *oldClient = nullptr;
	scanoutOwner.clear(&oldTask, &oldClient);
	if (oldTask)
		task_deallocate(static_cast<task_t>(const_cast<void *>(oldTask)));
	if (oldClient)
		static_cast<const IOUserClient *>(oldClient)->release();
	OSObject *clients[kMaxPipeClients];
	UInt32 held = pipes.all(clients);
	for (UInt32 i = 0; i < held; i++) {
		bool showScanout = false;
		pipes.remove(clients[i], &showScanout);
		clients[i]->release();
	}
	OSSafeReleaseNULL(queueTarget);
	OSSafeReleaseNULL(releaseTarget);
	super::free();
}

// MARK: - IOFramebuffer: the device and its modes

IOReturn ReimsGraphics::enableController()
{
	Mode m;
	IOLockLock(stateLock);
	IODisplayModeID startup = modeSet.startupId;
	bool ok = modeSet.modeFor(startup, &m);
	IOLockUnlock(stateLock);
	IOReturn r = ok ? setScanout(m) : kIOReturnNotReady;
	FBTRACE(8, "enableController: startup mode %d -> 0x%x", static_cast<int>(startup), r);
	return r;
}

IODeviceMemory *ReimsGraphics::getApertureRange(IOPixelAperture aperture)
{
	if (aperture != kIOFBSystemAperture || !region)
		return nullptr;
	Mode m;
	IOLockLock(stateLock);
	bool ok = modeSet.haveCurrent ? (m = modeSet.current, true)
	                              : modeSet.modeFor(modeSet.startupId, &m);
	IOLockUnlock(stateLock);
	if (!ok)
		return nullptr;
	UInt64 length = round_page_64(static_cast<UInt64>(strideFor(m.width)) * m.height);
	FBTRACE(8, "getApertureRange: %ux%u, 0x%llx bytes", m.width, m.height, length);
	return IODeviceMemory::withSubRange(region, RGFX_ALLOCATIONS_OFFSET, length);
}

// The scanout's room. IOFramebuffer maps it into the task that asks for it on a user client (its
// server connection, and its shared one for any task of the console's user), calling this on that
// task's thread (IOGraphics' open source, clientMemoryForType). It asks in its own setup too (in
// the kernel, and inside the server connection's open), where it reads the room's size into
// kIOFBMemorySizeKey and the aperture's offset in it. So the room goes to the scanout's owner on
// the connection it holds alone, and IOFramebuffer's setup in the kernel or in the open and every
// other task get none (RgfxScanoutOwner.hpp): the framebuffer publishes no memory size, unless the
// owner sets a new mode, whose setup runs on the owner's thread. The aperture (getApertureRange),
// which the kernel maps for the console, is another range, and stays.
IODeviceMemory *ReimsGraphics::getVRAMRange()
{
	if (!region)
		return nullptr;
	task_t task = current_task();
	bool connected = false;
	if (task != kernel_task && stateLock) {
		IOLockLock(stateLock);
		const IOUserClient *client = static_cast<const IOUserClient *>(scanoutOwner.client);
		connected = scanoutOwner.holdsConnection(task, client && !client->isInactive());
		IOLockUnlock(stateLock);
	}
	bool given = rgfxMayMapVram(task, kernel_task, connected);
	static char site;
	RgfxCaller caller;
	if (caller.pid != 0 && rgfxFirstCalls(&site, static_cast<UInt64>(caller.pid), 4))
		IOLog("ReimsGraphics: the VRAM range %s %s[%d]\n",
		      given ? "goes to the scanout's owner" : "is refused to", caller.name, caller.pid);
	if (!given)
		return nullptr;
	return IODeviceMemory::withSubRange(region, RGFX_ALLOCATIONS_OFFSET, scanoutLimit());
}

const char *ReimsGraphics::getPixelFormats()
{
	static const char formats[] = IO32BitDirectPixels "\0";
	return formats;
}

IOItemCount ReimsGraphics::getDisplayModeCount()
{
	IOLockLock(stateLock);
	reportedCount = modeSet.list(nullptr);
	IOLockUnlock(stateLock);
	FBTRACE(16, "getDisplayModeCount -> %u", reportedCount);
	return reportedCount;
}

// The caller's array holds the count last reported. A DISPLAYS answer may
// change the list in between: then the array still names the mode set, and the
// connect interrupt that follows has IOFramebuffer ask again (Modes::fill).
IOReturn ReimsGraphics::getDisplayModes(IODisplayModeID *allDisplayModes)
{
	IOLockLock(stateLock);
	modeSet.fill(allDisplayModes, reportedCount);
	FBTRACE(16, "getDisplayModes: %u modes, first %d", reportedCount,
	        reportedCount ? static_cast<int>(allDisplayModes[0]) : -1);
	IOLockUnlock(stateLock);
	return kIOReturnSuccess;
}

IOReturn ReimsGraphics::getInformationForDisplayMode(IODisplayModeID displayMode,
                                                     IODisplayModeInformation *info)
{
	Mode m;
	IOLockLock(stateLock);
	bool ok = modeSet.modeFor(displayMode, &m);
	bool preferred = ok && modeSet.isPreferred(m);
	UInt16 mmWide = 0, mmHigh = 0;
	if (ok)
		imageSize(m, &mmWide, &mmHigh);
	IOLockUnlock(stateLock);
	FBTRACE(48, "getInformationForDisplayMode %d: %s %ux%u", static_cast<int>(displayMode),
	        ok ? "ok" : "UNKNOWN", ok ? m.width : 0, ok ? m.height : 0);
	if (!ok || !info)
		return kIOReturnBadArgument;
	bzero(info, sizeof(*info));
	info->nominalWidth = m.width;
	info->nominalHeight = m.height;
	info->refreshRate = static_cast<IOFixed1616>((static_cast<UInt64>(m.refreshMhz) << 16) / 1000);
	info->maxDepthIndex = 0;
	info->flags = kDisplayModeValidFlag | kDisplayModeSafeFlag | (preferred ? kDisplayModeDefaultFlag : 0);
	info->imageWidth = mmWide;
	info->imageHeight = mmHigh;
	return kIOReturnSuccess;
}

UInt64 ReimsGraphics::getPixelFormatsForDisplayMode(IODisplayModeID, IOIndex)
{
	return 0;
}

IOReturn ReimsGraphics::getPixelInformation(IODisplayModeID displayMode, IOIndex depth,
                                            IOPixelAperture aperture, IOPixelInformation *pixelInfo)
{
	FBTRACE(48, "getPixelInformation mode %d depth %d aperture %d", static_cast<int>(displayMode),
	        static_cast<int>(depth), static_cast<int>(aperture));
	if (depth != 0 || aperture != kIOFBSystemAperture || !pixelInfo)
		return kIOReturnUnsupportedMode;
	Mode m;
	IOLockLock(stateLock);
	bool ok = modeSet.modeFor(displayMode, &m);
	IOLockUnlock(stateLock);
	if (!ok)
		return kIOReturnBadArgument;
	// 32-bit BGRA8: little-endian 0x00RRGGBB words.
	bzero(pixelInfo, sizeof(*pixelInfo));
	pixelInfo->bytesPerRow = strideFor(m.width);
	pixelInfo->bitsPerPixel = 32;
	pixelInfo->pixelType = kIORGBDirectPixels;
	pixelInfo->componentCount = 3;
	pixelInfo->bitsPerComponent = 8;
	pixelInfo->componentMasks[0] = 0x00ff0000;
	pixelInfo->componentMasks[1] = 0x0000ff00;
	pixelInfo->componentMasks[2] = 0x000000ff;
	strlcpy(pixelInfo->pixelFormat, IO32BitDirectPixels, sizeof(pixelInfo->pixelFormat));
	pixelInfo->activeWidth = m.width;
	pixelInfo->activeHeight = m.height;
	return kIOReturnSuccess;
}

IOReturn ReimsGraphics::getCurrentDisplayMode(IODisplayModeID *displayMode, IOIndex *depth)
{
	IOLockLock(stateLock);
	IODisplayModeID id = modeSet.haveCurrent ? modeSet.current.id : modeSet.startupId;
	IOLockUnlock(stateLock);
	FBTRACE(32, "getCurrentDisplayMode -> %d", static_cast<int>(id));
	if (displayMode)
		*displayMode = id;
	if (depth)
		*depth = 0;
	return kIOReturnSuccess;
}

IOReturn ReimsGraphics::setDisplayMode(IODisplayModeID displayMode, IOIndex depth)
{
	FBTRACE(16, "setDisplayMode %d depth %d", static_cast<int>(displayMode), static_cast<int>(depth));
	if (depth != 0)
		return kIOReturnUnsupportedMode;
	Mode m;
	IOLockLock(stateLock);
	bool ok = modeSet.modeFor(displayMode, &m);
	IOLockUnlock(stateLock);
	return ok ? setScanout(m) : kIOReturnBadArgument;
}

IOReturn ReimsGraphics::getStartupDisplayMode(IODisplayModeID *displayMode, IOIndex *depth)
{
	IOLockLock(stateLock);
	IODisplayModeID id = modeSet.startupId;
	IOLockUnlock(stateLock);
	FBTRACE(16, "getStartupDisplayMode -> %d", static_cast<int>(id));
	if (displayMode)
		*displayMode = id;
	if (depth)
		*depth = 0;
	return kIOReturnSuccess;
}

IOReturn ReimsGraphics::setStartupDisplayMode(IODisplayModeID displayMode, IOIndex depth)
{
	FBTRACE(16, "setStartupDisplayMode %d depth %d", static_cast<int>(displayMode),
	        static_cast<int>(depth));
	if (depth != 0)
		return kIOReturnUnsupportedMode;
	Mode m;
	IOLockLock(stateLock);
	bool ok = modeSet.modeFor(displayMode, &m);
	if (ok)
		modeSet.startupId = displayMode;
	IOLockUnlock(stateLock);
	return ok ? kIOReturnSuccess : kIOReturnBadArgument;
}

// Each mode is a fixed-rate panel's, with the timing its EDID entry gives.
IOReturn ReimsGraphics::getTimingInfoForDisplayMode(IODisplayModeID displayMode,
                                                    IOTimingInformation *info)
{
	Mode m;
	IODetailedTimingInformationV2 given;
	IOLockLock(stateLock);
	bool ok = modeSet.modeFor(displayMode, &m);
	const Modes::Programmed *p = modeSet.programmedFor(displayMode);
	if (p)
		given = p->timing;
	IOLockUnlock(stateLock);
	FBTRACE(48, "getTimingInfoForDisplayMode %d: %s%s, flags in 0x%x", static_cast<int>(displayMode),
	        ok ? "ok" : "UNKNOWN", p ? " programmed" : "", info ? static_cast<unsigned>(info->flags) : 0);
	if (!ok || !info)
		return kIOReturnBadArgument;
	if (p) {
		// A programmed timing is no standard one: it is given back as the OS gave it.
		bzero(info, sizeof(*info));
		info->appleTimingID = kIOTimingIDInvalid;
		info->flags = kIODetailedTimingValid;
		info->detailedInfo.v2 = given;
		return kIOReturnSuccess;
	}
	Timing timing;
	timingFor(m, &timing);
	bzero(info, sizeof(*info));
	info->appleTimingID = kIOTimingIDApple_FixedRateLCD;
	info->flags = kIODetailedTimingValid;
	IODetailedTimingInformationV2 &t = info->detailedInfo.v2;
	t.pixelClock = timing.pixelClock;
	t.minPixelClock = timing.pixelClock;
	t.maxPixelClock = timing.pixelClock;
	t.horizontalActive = timing.hActive;
	t.horizontalBlanking = timing.hBlank;
	t.horizontalSyncOffset = timing.hFront;
	t.horizontalSyncPulseWidth = timing.hSync;
	t.verticalActive = timing.vActive;
	t.verticalBlanking = timing.vBlank;
	t.verticalSyncOffset = timing.vFront;
	t.verticalSyncPulseWidth = timing.vSync;
	// Reduced blanking's sync: horizontal positive, vertical negative.
	t.horizontalSyncConfig = kIOSyncPositivePolarity;
	t.verticalSyncConfig = 0;
	return kIOReturnSuccess;
}

// A timing the device can show: a listed mode's size and rate, unscaled and progressive. For a
// mode description, the description's mode information is filled in.
IOReturn ReimsGraphics::validateDetailedTiming(void *description, IOByteCount descripSize)
{
	IOFBDisplayModeDescription *desc = nullptr;
	IODetailedTimingInformationV2 *t = nullptr;
	if (description && descripSize == sizeof(IOFBDisplayModeDescription)) {
		desc = static_cast<IOFBDisplayModeDescription *>(description);
		t = &desc->timingInfo.detailedInfo.v2;
	} else if (description && descripSize == sizeof(IODetailedTimingInformationV2)) {
		t = static_cast<IODetailedTimingInformationV2 *>(description);
	} else {
		FBTRACE(8, "validateDetailedTiming: %lu bytes: not a timing",
		        static_cast<unsigned long>(descripSize));
		return kIOReturnUnsupported;
	}
	bool scaled = (t->horizontalScaled && t->horizontalScaled != t->horizontalActive) ||
	              (t->verticalScaled && t->verticalScaled != t->verticalActive) ||
	              t->horizontalScaledInset || t->verticalScaledInset;
	UInt32 refresh = refreshMhzOf(*t);
	Mode host;
	IOLockLock(stateLock);
	bool ok = !scaled && !(t->signalConfig & kIOInterlacedCEATiming) &&
	          modeSet.deviceMode(t->horizontalActive, t->verticalActive, refresh, &host);
	bool preferred = ok && host.id == modeSet.preferredId;
	UInt16 mmWide = 0, mmHigh = 0;
	if (ok)
		imageSize(host, &mmWide, &mmHigh);
	IOLockUnlock(stateLock);
	FBTRACE(32, "validateDetailedTiming %ux%u (scaled %ux%u) at %u mHz, clock %llu: %s",
	        t->horizontalActive, t->verticalActive, t->horizontalScaled, t->verticalScaled, refresh,
	        t->pixelClock, ok ? "ok" : "REFUSED");
	if (!ok)
		return kIOReturnUnsupportedMode;
	if (desc) {
		IODisplayModeInformation &info = desc->info;
		bzero(&info, sizeof(info));
		info.nominalWidth = t->horizontalActive;
		info.nominalHeight = t->verticalActive;
		info.refreshRate = static_cast<IOFixed1616>((static_cast<UInt64>(refresh) << 16) / 1000);
		info.maxDepthIndex = 0;
		info.flags = kDisplayModeValidFlag | kDisplayModeSafeFlag | (preferred ? kDisplayModeDefaultFlag : 0);
		info.imageWidth = mmWide;
		info.imageHeight = mmHigh;
	}
	return kIOReturnSuccess;
}

// Each element is an OSData of an IODetailedTimingInformationV2 whose first reserved word is
// the mode ID the OS gave it, at or past kIODisplayModeIDReservedBase. The mode set takes the ID
// of a programmed timing of its size and rate, as the list now names it.
IOReturn ReimsGraphics::setDetailedTimings(OSArray *array)
{
	UInt32 taken = 0;
	IOLockLock(stateLock);
	modeSet.clearProgrammed();
	for (UInt32 i = 0; array && i < array->getCount(); i++) {
		OSData *data = OSDynamicCast(OSData, array->getObject(i));
		if (!data || data->getLength() < sizeof(IODetailedTimingInformationV2))
			continue;
		IODetailedTimingInformationV2 t;
		memcpy(&t, data->getBytesNoCopy(), sizeof(t));
		IODisplayModeID id = static_cast<IODisplayModeID>(t.__reservedA[0]);
		if (!t.horizontalActive || !t.verticalActive || t.horizontalActive > kMaxDimension ||
		    t.verticalActive > kMaxDimension)
			continue;
		Mode m = {id, t.horizontalActive, t.verticalActive, refreshMhzOf(t)};
		if (modeSet.addProgrammed(id, m, t, kIODisplayModeIDReservedBase) &&
		    modeSet.deviceMode(m.width, m.height, m.refreshMhz, nullptr))
			taken++;
	}
	modeSet.followProgrammed();
	UInt32 count = modeSet.programmedCount;
	IOLockUnlock(stateLock);
	if (array)
		setProperty(kIOFBDetailedTimingsKey, array);
	else
		removeProperty(kIOFBDetailedTimingsKey);
	IOLog("ReimsGraphics: display %u: %u programmed timings, %u of them the device's modes\n",
	      displayId, count, taken);
	return kIOReturnSuccess;
}

// MARK: - IOFramebuffer: attributes and the display's connection

IOReturn ReimsGraphics::getAttribute(IOSelect attribute, uintptr_t *value)
{
	if (attribute == kIOHardwareCursorAttribute) {
		if (value)
			*value = 1;
		traceAttribute("getAttribute", -1, attribute, 1, kIOReturnSuccess);
		return kIOReturnSuccess;
	}
	if (attribute == kIOWindowServerActiveAttribute) {
		// WindowServer's access to the display, as it last set it (setAttribute).
		if (!value)
			return kIOReturnBadArgument;
		*value = windowServerActive;
		traceAttribute("getAttribute", -1, attribute, *value, kIOReturnSuccess);
		return kIOReturnSuccess;
	}
	IOReturn r = super::getAttribute(attribute, value);
	traceAttribute("getAttribute", -1, attribute, value ? *value : 0, r);
	return r;
}

IOReturn ReimsGraphics::setAttribute(IOSelect attribute, uintptr_t value)
{
	// The device has no power states of its own, and the framebuffer registers none (Power).
	if (attribute == kIOPowerAttribute) {
		traceAttribute("setAttribute", -1, attribute, value, kIOReturnSuccess);
		return kIOReturnSuccess;
	}
	if (attribute == kIOWindowServerActiveAttribute) {
		// WindowServer's access to the display (a kIOWSAA_ state: CPU only, GPU only, or moving
		// between them, with or without the transactional bit, which the display pipe's
		// transactions follow): recorded, for getAttribute. The display shows the scanout, or
		// the surface the display pipe last presented, in every state.
		windowServerActive = static_cast<UInt32>(value);
		traceAttribute("setAttribute", -1, attribute, value, kIOReturnSuccess);
		return kIOReturnSuccess;
	}
	IOReturn r = super::setAttribute(attribute, value);
	traceAttribute("setAttribute", -1, attribute, value, r);
	return r;
}

IOItemCount ReimsGraphics::getConnectionCount()
{
	return 1;
}

IOReturn ReimsGraphics::getAttributeForConnection(IOIndex connectIndex, IOSelect attribute,
                                                  uintptr_t *value)
{
#if RGFX_FB_TRACE
	IOReturn r = connectionAttribute(connectIndex, attribute, value);
	traceAttribute("getAttributeForConnection", connectIndex, attribute,
	               value && r == kIOReturnSuccess ? *value : 0, r);
	return r;
}

IOReturn ReimsGraphics::connectionAttribute(IOIndex connectIndex, IOSelect attribute,
                                            uintptr_t *value)
{
#endif
	if (connectIndex != 0)
		return kIOReturnBadArgument;
	switch (attribute) {
	case kConnectionEnable:
	case kConnectionCheckEnable:
		if (!value)
			return kIOReturnBadArgument;
		IOLockLock(stateLock);
		*value = connected ? 1 : 0;
		IOLockUnlock(stateLock);
		return kIOReturnSuccess;
	case kConnectionFlags:
		if (!value)
			return kIOReturnBadArgument;
		*value = 0;
		return kIOReturnSuccess;
	case kConnectionSupportsHLDDCSense:
		// The driver's own EDID, block by block (getDDCBlock); value may be null.
		return kIOReturnSuccess;
	case kConnectionSupportsLLDDCSense:
	case kConnectionSupportsAppleSense:
		// No I2C lines and no sense lines.
		return kIOReturnUnsupported;
	default:
		// IOFramebuffer answers the rest, the display's parameters among them
		// (kConnectionDisplayParameterCount and kConnectionDisplayParameters: its red, green and
		// blue gamma scales and its VBL multiplier), which IODisplay publishes as
		// IODisplayParameters.
		return super::getAttributeForConnection(connectIndex, attribute, value);
	}
}

IOReturn ReimsGraphics::setAttributeForConnection(IOIndex connectIndex, IOSelect attribute,
                                                  uintptr_t value)
{
#if RGFX_FB_TRACE
	IOReturn r = setConnectionAttribute(connectIndex, attribute, value);
	traceAttribute("setAttributeForConnection", connectIndex, attribute, value, r);
	return r;
}

IOReturn ReimsGraphics::setConnectionAttribute(IOIndex connectIndex, IOSelect attribute,
                                               uintptr_t value)
{
#endif
	if (connectIndex != 0)
		return kIOReturnBadArgument;
	switch (attribute) {
	case kConnectionProbe:
	case kConnectionChanged:
	case kConnectionPower:
		// The modes are already the device's latest: the driver asks again
		// on every DISPLAY_CHANGED before it signals the change.
		return kIOReturnSuccess;
	default:
		return super::setAttributeForConnection(connectIndex, attribute, value);
	}
}

// MARK: - The display's timings and EDID

void ReimsGraphics::timingFor(const Mode &mode, Timing *t)
{
	// Reduced blanking: 80 pixels of horizontal blank (8 front porch, 32 sync), and enough
	// lines of vertical blank to last 460 us at the mode's rate, at least the sync's 8 and
	// the back porch's 6 and one front porch line. 460 us is the fraction
	// refreshMhz * 460 / 1e9 of a frame, so the blank is vActive * f / (1 - f) lines.
	t->hActive = mode.width;
	t->hBlank = 80;
	t->hFront = 8;
	t->hSync = 32;
	t->vActive = mode.height;
	t->vSync = 8;
	UInt64 blankPart = static_cast<UInt64>(mode.refreshMhz) * 460;
	UInt64 activePart = 1000000000ull > blankPart ? 1000000000ull - blankPart : 1;
	UInt64 vBlank = (static_cast<UInt64>(mode.height) * blankPart + activePart - 1) / activePart;
	if (vBlank < 8 + 6 + 1)
		vBlank = 8 + 6 + 1;
	t->vBlank = static_cast<UInt32>(vBlank);
	t->vFront = t->vBlank - t->vSync - 6;
	UInt64 total = static_cast<UInt64>(t->hActive + t->hBlank) * (t->vActive + t->vBlank);
	UInt64 clock = total * mode.refreshMhz / 1000;
	t->pixelClock = (clock + 5000) / 10000 * 10000;
}

UInt32 ReimsGraphics::edidExtensionCount() const
{
	UInt32 n = (modeSet.modeCount + kEdidModesPerBlock - 1) / kEdidModesPerBlock;
	return n ? n : 1;
}

namespace {

// The EDID's identity: our own three-letter PNP manufacturer ID "RVG", packed as five
// bits a letter ('A' is 1), and our product code.
constexpr UInt16 kEdidManufacturer = ((('R' - '@') << 10) | (('V' - '@') << 5) | ('G' - '@'));
constexpr UInt16 kEdidProduct = 0x5247;
constexpr char kEdidName[] = "Reims Display";

// The largest pixel clock a base block's detailed timing holds: 16 bits of 10 kHz.
constexpr UInt64 kEdidDetailedClockMax = 0xffffull * 10000;

// The size, in millimetres, the EDID gives a mode's display: at 220 pixels per inch for a
// mode at least 2880 pixels wide, as a high-density panel, and 110 otherwise.
UInt32 edidMillimetres(UInt32 pixels, UInt32 width)
{
	UInt32 ppi = width >= 2880 ? 220 : 110;
	return (pixels * 254 + ppi * 5) / (ppi * 10);
}

void edidChecksum(UInt8 *block)
{
	UInt8 sum = 0;
	for (UInt32 i = 0; i < 127; i++)
		sum = static_cast<UInt8>(sum + block[i]);
	block[127] = static_cast<UInt8>(-sum);
}

// One 10-bit chromaticity coordinate, from thousandths.
constexpr UInt32 chroma(UInt32 thousandths)
{
	return (thousandths * 1024 + 500) / 1000;
}

} // namespace

// Block 0 is the base block: the identity, sRGB colour, the preferred mode's detailed timing
// when its clock fits one (else the largest mode whose does), and the product name. Each
// extension is a DisplayID 1.3 section with a type I timing block of up to five modes, the
// preferred mode first and marked so.
bool ReimsGraphics::makeEdid(UInt32 index, UInt8 *b) const
{
	UInt32 extensions = edidExtensionCount();
	if (index > extensions)
		return false;
	bzero(b, kEdidBlockBytes);

	// The modes, the preferred one first.
	Mode order[kMaxModes];
	UInt32 n = 0;
	for (UInt32 i = 0; i < modeSet.modeCount; i++) {
		if (modeSet.modes[i].id == modeSet.preferredId)
			order[n++] = modeSet.modes[i];
	}
	for (UInt32 i = 0; i < modeSet.modeCount; i++) {
		if (modeSet.modes[i].id != modeSet.preferredId)
			order[n++] = modeSet.modes[i];
	}

	if (index > 0) {
		UInt32 first = (index - 1) * kEdidModesPerBlock;
		UInt32 count = first < n ? n - first : 0;
		if (count > kEdidModesPerBlock)
			count = kEdidModesPerBlock;
		b[0] = 0x70; // DisplayID
		b[1] = 0x13; // version 1.3
		b[3] = 0x00; // an extension section
		b[4] = 0x00; // no further sections
		UInt32 at = 5;
		if (count) {
			b[at++] = 0x03; // type I detailed timings
			b[at++] = 0x00;
			b[at++] = static_cast<UInt8>(count * 20);
			for (UInt32 j = 0; j < count; j++, at += 20) {
				Timing t;
				timingFor(order[first + j], &t);
				UInt32 clock = static_cast<UInt32>(t.pixelClock / 10000) - 1;
				UInt8 *d = b + at;
				d[0] = static_cast<UInt8>(clock);
				d[1] = static_cast<UInt8>(clock >> 8);
				d[2] = static_cast<UInt8>(clock >> 16);
				// Preferred (bit 7) for the first mode of the first block; aspect undefined.
				d[3] = static_cast<UInt8>((index == 1 && j == 0 ? 0x80 : 0x00) | 0x08);
				UInt16 fields[8] = {
				    static_cast<UInt16>(t.hActive - 1), static_cast<UInt16>(t.hBlank - 1),
				    // Bit 15: positive horizontal sync.
				    static_cast<UInt16>((t.hFront - 1) | 0x8000), static_cast<UInt16>(t.hSync - 1),
				    static_cast<UInt16>(t.vActive - 1), static_cast<UInt16>(t.vBlank - 1),
				    static_cast<UInt16>(t.vFront - 1), static_cast<UInt16>(t.vSync - 1),
				};
				for (UInt32 f = 0; f < 8; f++) {
					d[4 + f * 2] = static_cast<UInt8>(fields[f]);
					d[5 + f * 2] = static_cast<UInt8>(fields[f] >> 8);
				}
			}
		}
		b[2] = static_cast<UInt8>(at - 5);
		UInt8 sum = 0;
		for (UInt32 i = 1; i < at; i++)
			sum = static_cast<UInt8>(sum + b[i]);
		b[at] = static_cast<UInt8>(-sum);
		edidChecksum(b);
		return true;
	}

	// The base block's detailed timing: the preferred mode, or the largest that fits.
	const Mode *detailed = nullptr;
	for (UInt32 i = 0; i < n; i++) {
		Timing t;
		timingFor(order[i], &t);
		if (t.pixelClock > kEdidDetailedClockMax || order[i].width > 4095 || order[i].height > 4095)
			continue;
		if (!detailed || i == 0 ||
		    static_cast<UInt64>(order[i].width) * order[i].height >
		        static_cast<UInt64>(detailed->width) * detailed->height)
			detailed = &order[i];
		if (i == 0)
			break;
	}
	const Mode &native = n ? order[0] : modeSet.current;

	static const UInt8 header[8] = {0x00, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x00};
	memcpy(b, header, sizeof(header));
	b[8] = static_cast<UInt8>(kEdidManufacturer >> 8);
	b[9] = static_cast<UInt8>(kEdidManufacturer);
	b[10] = static_cast<UInt8>(kEdidProduct);
	b[11] = static_cast<UInt8>(kEdidProduct >> 8);
	// The serial: the product code over the device's display id, so each display keeps its own.
	UInt32 serial = (static_cast<UInt32>(kEdidProduct) << 16) | ((displayId + 1) & 0xffff);
	b[12] = static_cast<UInt8>(serial);
	b[13] = static_cast<UInt8>(serial >> 8);
	b[14] = static_cast<UInt8>(serial >> 16);
	b[15] = static_cast<UInt8>(serial >> 24);
	b[16] = 0;    // week unspecified
	b[17] = 36;   // 2026
	b[18] = 1;    // EDID 1.4
	b[19] = 4;
	b[20] = 0xa5; // digital, 8 bits a component, DisplayPort
	b[21] = static_cast<UInt8>((edidMillimetres(native.width, native.width) + 5) / 10);
	b[22] = static_cast<UInt8>((edidMillimetres(native.height, native.width) + 5) / 10);
	b[23] = 120;  // gamma 2.2
	// RGB 4:4:4, sRGB the default colour space, and the first timing the preferred one
	// where it is.
	b[24] = static_cast<UInt8>(0x04 | (detailed == &order[0] ? 0x02 : 0x00));

	// sRGB's primaries and D65 white, 10 bits each.
	const UInt32 rx = chroma(640), ry = chroma(330), gx = chroma(300), gy = chroma(600);
	const UInt32 bx = chroma(150), by = chroma(60), wx = chroma(313), wy = chroma(329);
	b[25] = static_cast<UInt8>(((rx & 3) << 6) | ((ry & 3) << 4) | ((gx & 3) << 2) | (gy & 3));
	b[26] = static_cast<UInt8>(((bx & 3) << 6) | ((by & 3) << 4) | ((wx & 3) << 2) | (wy & 3));
	b[27] = static_cast<UInt8>(rx >> 2);
	b[28] = static_cast<UInt8>(ry >> 2);
	b[29] = static_cast<UInt8>(gx >> 2);
	b[30] = static_cast<UInt8>(gy >> 2);
	b[31] = static_cast<UInt8>(bx >> 2);
	b[32] = static_cast<UInt8>(by >> 2);
	b[33] = static_cast<UInt8>(wx >> 2);
	b[34] = static_cast<UInt8>(wy >> 2);
	// No established timings; no standard timings (01 01 each).
	for (UInt32 i = 38; i < 54; i++)
		b[i] = 0x01;

	UInt8 *d = b + 54;
	if (detailed) {
		Timing t;
		timingFor(*detailed, &t);
		UInt32 clock = static_cast<UInt32>(t.pixelClock / 10000);
		UInt32 hmm = edidMillimetres(t.hActive, native.width);
		UInt32 vmm = edidMillimetres(t.vActive, native.width);
		d[0] = static_cast<UInt8>(clock);
		d[1] = static_cast<UInt8>(clock >> 8);
		d[2] = static_cast<UInt8>(t.hActive);
		d[3] = static_cast<UInt8>(t.hBlank);
		d[4] = static_cast<UInt8>(((t.hActive >> 8) << 4) | (t.hBlank >> 8));
		d[5] = static_cast<UInt8>(t.vActive);
		d[6] = static_cast<UInt8>(t.vBlank);
		d[7] = static_cast<UInt8>(((t.vActive >> 8) << 4) | (t.vBlank >> 8));
		d[8] = static_cast<UInt8>(t.hFront);
		d[9] = static_cast<UInt8>(t.hSync);
		d[10] = static_cast<UInt8>(((t.vFront & 0xf) << 4) | (t.vSync & 0xf));
		d[11] = static_cast<UInt8>(((t.hFront >> 8) << 6) | ((t.hSync >> 8) << 4) |
		                           ((t.vFront >> 4) << 2) | (t.vSync >> 4));
		d[12] = static_cast<UInt8>(hmm);
		d[13] = static_cast<UInt8>(vmm);
		d[14] = static_cast<UInt8>(((hmm >> 8) << 4) | ((vmm >> 8) & 0xf));
		// Digital separate sync, vertical negative, horizontal positive.
		d[17] = 0x1a;
	} else {
		d[3] = 0x10; // a dummy descriptor
	}
	d += 18;

	// The product name, padded as the descriptor asks.
	d[3] = 0xfc;
	UInt32 len = sizeof(kEdidName) - 1;
	for (UInt32 i = 0; i < 13; i++)
		d[5 + i] = i < len ? static_cast<UInt8>(kEdidName[i]) : (i == len ? 0x0a : 0x20);
	d += 18;
	d[3] = 0x10;
	d += 18;
	d[3] = 0x10;

	b[126] = static_cast<UInt8>(extensions);
	edidChecksum(b);
	return true;
}

bool ReimsGraphics::hasDDCConnect(IOIndex connectIndex)
{
	IOLockLock(stateLock);
	bool r = connectIndex == 0 && haveDisplay && connected;
	IOLockUnlock(stateLock);
	FBTRACE(8, "hasDDCConnect %d -> %d", static_cast<int>(connectIndex), r);
	return r;
}

// Block 1 is the base block and 2 on its extensions, as IODisplay asks for them.
IOReturn ReimsGraphics::getDDCBlock(IOIndex connectIndex, UInt32 blockNumber, IOSelect blockType,
                                    IOOptionBits options, UInt8 *data, IOByteCount *length)
{
	if (connectIndex != 0 || blockType != kIODDCBlockTypeEDID || !blockNumber || !data || !length ||
	    *length < kEdidBlockBytes)
		return kIOReturnBadArgument;
	IOLockLock(stateLock);
	bool ok = haveDisplay && makeEdid(blockNumber - 1, data);
	IOLockUnlock(stateLock);
	FBTRACE(16, "getDDCBlock %d block %u -> %d", static_cast<int>(connectIndex), blockNumber, ok);
	if (!ok)
		return kIOReturnUnsupported;
	*length = kEdidBlockBytes;
	return kIOReturnSuccess;
}

// MARK: - Power

// The framebuffer stays out of power management, as the stock framebuffer does: it registers
// no power states, so it and its AppleDisplay stay at power state 0 while the display wrangler
// idles and wakes. IOFramebuffer then never turns the display's sleep into one of the
// framebuffer's: WindowServer hears of no display sleep and takes no sleep image, and the
// scanout goes on showing what it shows. With IOFBDependentID published, IOFramebuffer counts
// the framebuffer usable from its start without a power driver.

// MARK: - IOFramebuffer: the hardware cursor

// IOFramebuffer converts the image into the slot as packed little-endian
// 0xAARRGGBB words, premultiplied: BGRA8 in memory, the format SET_CURSOR
// takes. The command's answer means the host has the image, so the slot is
// free for the next one.
IOReturn ReimsGraphics::setCursorImage(void *cursorImage)
{
	if (!ready)
		return kIOReturnNotReady;
	IOHardwareCursorDescriptor desc;
	bzero(&desc, sizeof(desc));
	desc.majorVersion = kHardwareCursorDescriptorMajorVersion;
	desc.minorVersion = kHardwareCursorDescriptorMinorVersion;
	desc.width = kCursorMax;
	desc.height = kCursorMax;
	desc.bitDepth = 32;
	IOHardwareCursorInfo info;
	bzero(&info, sizeof(info));
	info.majorVersion = kHardwareCursorInfoMajorVersion;
	info.minorVersion = kHardwareCursorInfoMinorVersion;
	info.hardwareCursorData = cursorPixels;
	if (!convertCursorImage(cursorImage, &desc, &info) || !info.cursorWidth || !info.cursorHeight)
		return kIOReturnUnsupported;
	RgfxSetCursor c = {};
	c.display = displayId;
	c.width = info.cursorWidth;
	c.height = info.cursorHeight;
	c.hot_x = info.cursorHotSpotX;
	c.hot_y = info.cursorHotSpotY;
	c.offset = regionSize - kCursorSlot;
	IOReturn r = command(RGFX_CMD_SET_CURSOR, &c, sizeof(c));
	if (r == kIOReturnSuccess) {
		hotX = info.cursorHotSpotX;
		hotY = info.cursorHotSpotY;
	}
	return r;
}

// IOFramebuffer gives the image's top-left corner; MOVE_CURSOR takes the
// hotspot's place. Never blocks.
IOReturn ReimsGraphics::setCursorState(SInt32 x, SInt32 y, bool visible)
{
	RgfxMoveCursor m = {};
	m.display = displayId;
	m.x = x + static_cast<SInt32>(hotX);
	m.y = y + static_cast<SInt32>(hotY);
	m.visible = visible ? 1u : 0u;
	return tryPost(RGFX_CMD_MOVE_CURSOR, &m, sizeof(m), nullptr) ? kIOReturnSuccess : kIOReturnBusy;
}

// MARK: - IOFramebuffer: vblank and connect interrupts

// A registration starts enabled. The VBLANK events follow the vblank one:
// the work loop sends SET_VBLANK when IOFramebuffer turns it on or off.
IOReturn ReimsGraphics::registerForInterruptType(IOSelect interruptType, IOFBInterruptProc proc,
                                                 OSObject *target, void *ref, void **interruptRef)
{
	Interrupt *i = interruptType == kIOFBVBLInterruptType       ? &vblank
	               : interruptType == kIOFBConnectInterruptType ? &connect
	                                                            : nullptr;
	FBTRACE(8, "registerForInterruptType '%c%c%c%c'", static_cast<char>(interruptType >> 24),
	        static_cast<char>(interruptType >> 16), static_cast<char>(interruptType >> 8),
	        static_cast<char>(interruptType));
	if (!i)
		return kIOReturnUnsupported;
	IOSimpleLockLock(interruptLock);
	*i = {proc, target, ref, true};
	if (i == &vblank)
		vblankWanted = true;
	IOSimpleLockUnlock(interruptLock);
	if (interruptRef)
		*interruptRef = i;
	kickWorkLoop();
	return kIOReturnSuccess;
}

IOReturn ReimsGraphics::unregisterInterrupt(void *interruptRef)
{
	if (interruptRef != &vblank && interruptRef != &connect)
		return kIOReturnBadArgument;
	Interrupt *i = static_cast<Interrupt *>(interruptRef);
	IOSimpleLockLock(interruptLock);
	*i = {};
	if (i == &vblank)
		vblankWanted = false;
	IOSimpleLockUnlock(interruptLock);
	kickWorkLoop();
	return kIOReturnSuccess;
}

IOReturn ReimsGraphics::setInterruptState(void *interruptRef, UInt32 state)
{
	if (interruptRef != &vblank && interruptRef != &connect)
		return kIOReturnBadArgument;
	Interrupt *i = static_cast<Interrupt *>(interruptRef);
	IOSimpleLockLock(interruptLock);
	i->enabled = state == kEnabledInterruptState;
	if (i == &vblank)
		vblankWanted = i->enabled && i->proc;
	IOSimpleLockUnlock(interruptLock);
	kickWorkLoop();
	return kIOReturnSuccess;
}

// MARK: - The display pipe

IOReturn ReimsGraphics::postPresent(UInt32 surfaceId, UInt32 transactionId, UInt64 *seqno)
{
	// A surface whose GPU copy is the host's device cache carries its CPU generation, so the
	// display shows what the CPU wrote under its read-write locks too.
	struct {
		RgfxPresentSurface present;
		RgfxPresentSurfaceGen gen;
	} p = {};
	static_assert(sizeof(RgfxPresentSurface) == RGFX_PRESENT_SURFACE_GEN_OFFSET,
	              "the generation follows the present");
	p.present.display = displayId;
	p.present.iosurface_id = surfaceId;
	p.present.transaction_id = transactionId;
	const bool cached = rgfxSurfaceCacheGeneration(surfaceId, &p.gen.cpu_gen);
	// Never waits for room: the caller does, with its lock let go (waitForRoom).
	IOReturn r = kIOReturnSuccess;
	if (!tryPost(RGFX_CMD_PRESENT_SURFACE, &p, cached ? sizeof(p) : sizeof(p.present), seqno))
		r = ready ? kIOReturnNoSpace : kIOReturnNotReady;
	if (r == kIOReturnSuccess) {
		IOSimpleLockLock(interruptLock);
		pipes.shows(surfaceId);
		IOSimpleLockUnlock(interruptLock);
	}
	return r;
}

bool ReimsGraphics::addPipe(OSObject *client, PipeVblankHook vblankHook, PipePresentHook presentHook)
{
	if (!client)
		return false;
	// Retained before it is published; let go again where it was held already or has no room.
	client->retain();
	IOSimpleLockLock(interruptLock);
	bool already = pipes.holds(client);
	bool held = already || pipes.add(client);
	if (held) {
		pipeVblankHook = vblankHook;
		pipePresentHook = presentHook;
	}
	IOSimpleLockUnlock(interruptLock);
	if (already || !held)
		client->release();
	return held;
}

void ReimsGraphics::removePipe(OSObject *client)
{
	bool showScanout = false;
	IOSimpleLockLock(interruptLock);
	bool held = pipes.remove(client, &showScanout);
	IOSimpleLockUnlock(interruptLock);
	if (!held)
		return;
	client->release();
	if (showScanout) {
		// The last client went with a surface shown: the scanout again, so whatever draws into
		// it next is seen. Its answer goes to no client.
		RgfxPresentSurface p = {};
		p.display = displayId;
		IOReturn r = post(RGFX_CMD_PRESENT_SURFACE, &p, sizeof(p), nullptr);
		if (r != kIOReturnSuccess)
			IOLog("ReimsGraphics: the last display pipe client went; PRESENT_SURFACE 0 not sent: "
			      "0x%x\n", r);
	}
	kickWorkLoop();
}

void ReimsGraphics::setQueueInterrupt(OSObject *target, QueueInterruptHook hook)
{
	if (!hook)
		target = nullptr;
	if (target)
		target->retain();
	IOSimpleLockLock(interruptLock);
	OSObject *old = queueTarget;
	queueTarget = target;
	queueHook = target ? hook : nullptr;
	IOSimpleLockUnlock(interruptLock);
	if (old)
		old->release();
}

void ReimsGraphics::setReleaseAnswer(OSObject *target, ReleaseAnswerHook hook)
{
	if (!hook)
		target = nullptr;
	if (target)
		target->retain();
	IOSimpleLockLock(interruptLock);
	OSObject *old = releaseTarget;
	releaseTarget = target;
	releaseHook = target ? hook : nullptr;
	IOSimpleLockUnlock(interruptLock);
	if (old)
		old->release();
}

IOReturn ReimsGraphics::postReleases(const RgfxBlockRelease *releases, UInt32 count,
                                     UInt64 *const *waiterSeqnos, UInt32 *posted)
{
	static_assert(sizeof(RgfxBlockRelease) % kMsgAlign == 0 &&
	                  sizeof(RgfxBlockRelease) <= kLifecycleMessageMax,
	              "a release is not a whole message");
	*posted = 0;
	uint64_t deadline = 0;
	clock_interval_to_deadline(kPostTimeoutMs, kMillisecondScale, &deadline);
	// As many as fit, under one hold of the ring's lock and with one doorbell at most: whether
	// all are posted.
	auto look = [&]() {
		UInt32 n = 0;
		IOSimpleLockLock(submitLock);
		while (*posted + n < count &&
		       writeLocked(RGFX_CMD_BLOCK_RELEASE, &releases[*posted + n], sizeof(*releases),
		                   nullptr, waiterSeqnos[*posted + n]))
			n++;
		if (n)
			publishLocked();
		IOSimpleLockUnlock(submitLock);
		*posted += n;
		return *posted == count;
	};
	while (*posted < count) {
		if (!ready)
			return kIOReturnNotReady;
		const UInt64 mark = roomMark();
		if (look() || (askForRoom() && look()))
			break;
		if (!waitForRoom(mark, deadline)) {
			IOLog("ReimsGraphics: command %u: the control ring stayed full\n",
			      RGFX_CMD_BLOCK_RELEASE);
			return ready ? kIOReturnTimeout : kIOReturnNotReady;
		}
	}
	return kIOReturnSuccess;
}

void ReimsGraphics::setPipeVblankEnabled(OSObject *client, bool enabled)
{
	IOSimpleLockLock(interruptLock);
	pipes.setVblank(client, enabled);
	IOSimpleLockUnlock(interruptLock);
	kickWorkLoop();
}

#if RGFX_FB_TRACE
// MARK: - The trace build's forwarding overrides

IOReturn ReimsGraphics::open()
{
	IOReturn r = super::open();
	FBTRACE(8, "open -> 0x%x", r);
	return r;
}

void ReimsGraphics::close()
{
	FBTRACE(8, "close");
	super::close();
}

IOReturn ReimsGraphics::setGammaTable(UInt32 channelCount, UInt32 dataCount, UInt32 dataWidth,
                                      void *data)
{
	IOReturn r = super::setGammaTable(channelCount, dataCount, dataWidth, data);
	FBTRACE(4, "setGammaTable %u x %u x %u bits -> 0x%x", channelCount, dataCount, dataWidth, r);
	return r;
}

IOReturn ReimsGraphics::setPowerState(unsigned long powerStateOrdinal, IOService *device)
{
	IOReturn r = super::setPowerState(powerStateOrdinal, device);
	FBTRACE(16, "setPowerState %lu -> 0x%x", powerStateOrdinal, r);
	return r;
}

IOReturn ReimsGraphics::powerStateWillChangeTo(IOPMPowerFlags flags, unsigned long state,
                                               IOService *device)
{
	IOReturn r = super::powerStateWillChangeTo(flags, state, device);
	FBTRACE(16, "powerStateWillChangeTo flags 0x%lx state %lu -> 0x%x",
	        static_cast<unsigned long>(flags), state, r);
	return r;
}

IOReturn ReimsGraphics::powerStateDidChangeTo(IOPMPowerFlags flags, unsigned long state,
                                              IOService *device)
{
	IOReturn r = super::powerStateDidChangeTo(flags, state, device);
	FBTRACE(16, "powerStateDidChangeTo flags 0x%lx state %lu -> 0x%x",
	        static_cast<unsigned long>(flags), state, r);
	return r;
}

OSObject *ReimsGraphics::copyProperty(const OSSymbol *aKey) const
{
	OSObject *o = super::copyProperty(aKey);
	rgfxTracePropertyRead("fb", aKey ? aKey->getCStringNoCopy() : nullptr, o != nullptr);
	return o;
}

OSObject *ReimsGraphics::copyProperty(const OSString *aKey) const
{
	OSObject *o = super::copyProperty(aKey);
	rgfxTracePropertyRead("fb", aKey ? aKey->getCStringNoCopy() : nullptr, o != nullptr);
	return o;
}

OSObject *ReimsGraphics::copyProperty(const char *aKey) const
{
	OSObject *o = super::copyProperty(aKey);
	rgfxTracePropertyRead("fb", aKey, o != nullptr);
	return o;
}

IOReturn ReimsGraphics::requestProbe(IOOptionBits options)
{
	IOReturn r = super::requestProbe(options);
	FBTRACE(8, "requestProbe 0x%x -> 0x%x", static_cast<unsigned>(options), r);
	return r;
}

IOReturn ReimsGraphics::message(UInt32 type, IOService *provider, void *argument)
{
	IOReturn r = super::message(type, provider, argument);
	FBTRACE(16, "message 0x%x -> 0x%x", type, r);
	return r;
}

IOReturn ReimsGraphics::setupForCurrentConfig()
{
	IOReturn r = super::setupForCurrentConfig();
	FBTRACE(8, "setupForCurrentConfig -> 0x%x", r);
	return r;
}

bool ReimsGraphics::isConsoleDevice()
{
	bool r = super::isConsoleDevice();
	FBTRACE(8, "isConsoleDevice -> %d", r);
	return r;
}

IOReturn ReimsGraphics::setCLUTWithEntries(IOColorEntry *colors, UInt32 index, UInt32 numEntries,
                                           IOOptionBits options)
{
	IOReturn r = super::setCLUTWithEntries(colors, index, numEntries, options);
	FBTRACE(4, "setCLUTWithEntries %u+%u -> 0x%x", index, numEntries, r);
	return r;
}

IOReturn ReimsGraphics::connectFlags(IOIndex connectIndex, IODisplayModeID displayMode,
                                     IOOptionBits *flags)
{
	IOReturn r = super::connectFlags(connectIndex, displayMode, flags);
	FBTRACE(48, "connectFlags conn %d mode %d -> 0x%x flags 0x%x", static_cast<int>(connectIndex),
	        static_cast<int>(displayMode), r, flags ? static_cast<unsigned>(*flags) : 0);
	return r;
}

#endif
