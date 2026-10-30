/*
 * ReimsGraphics: the reims-graphics device's display driver (ReimsGraphics.hpp
 * says what it does and on which threads).
 */

#include "ReimsGraphics.hpp"

#include <IOKit/IOLib.h>
#include <kern/clock.h>
#include <libkern/libkern.h>

#define super IOFramebuffer
OSDefineMetaClassAndStructors(ReimsGraphics, IOFramebuffer);

namespace {

constexpr UInt32 kRingBytes = RG_CONTROL_RING_BYTES;
constexpr UInt32 kHeaderBytes = sizeof(struct rg_msg_header);

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
bool ReimsGraphics::tryPost(UInt32 type, const void *payload, UInt32 payloadSize, UInt64 *seqno)
{
	UInt64 words[8] = {};
	UInt32 size = RG_MSG_SIZE(payloadSize);
	if (size > sizeof(words) || !ready)
		return false;
	volatile UInt8 *header = ring(RG_CONTROL_SUBMIT_OFFSET);
	volatile UInt8 *bytes = header + RG_RING_HEADER_SIZE;

	IOSimpleLockLock(submitLock);
	UInt64 used = produced - load64(header + RG_RING_CONSUMER);
	UInt32 offset = static_cast<UInt32>(produced % kRingBytes);
	UInt32 tail = kRingBytes - offset;
	UInt32 pad = tail < size ? tail : 0;
	if (used > kRingBytes || kRingBytes - used < static_cast<UInt64>(pad) + size) {
		IOSimpleLockUnlock(submitLock);
		return false;
	}
	if (pad) {
		// A message never wraps: type 0 pads to the ring's end. When 8
		// bytes are left, they hold its type and size alone.
		volatile UInt32 *p = reinterpret_cast<volatile UInt32 *>(bytes + offset);
		p[0] = RG_MSG_PAD;
		p[1] = pad;
		if (pad >= kHeaderBytes)
			*reinterpret_cast<volatile UInt64 *>(bytes + offset + 8) = 0;
		produced += pad;
		offset = 0;
	}
	UInt64 s = nextSeqno++;
	struct rg_msg_header h = {type, size, s};
	memcpy(words, &h, sizeof(h));
	if (payloadSize)
		memcpy(reinterpret_cast<UInt8 *>(words) + sizeof(h), payload, payloadSize);
	volatile UInt64 *dst = reinterpret_cast<volatile UInt64 *>(bytes + offset);
	for (UInt32 i = 0; i < size / 8; i++)
		dst[i] = words[i];
	produced += size;
	store64(header + RG_RING_PRODUCER, produced);
	fullFence();
	if (load32(header + RG_RING_CONSUMER_IDLE) == 1)
		writeRegister(RG_REG_DOORBELL, RG_CONTROL_RING_ID);
	IOSimpleLockUnlock(submitLock);

	if (seqno)
		*seqno = s;
	return true;
}

// Puts one command on ring 0, waiting for space: thread context only.
IOReturn ReimsGraphics::post(UInt32 type, const void *payload, UInt32 payloadSize, UInt64 *seqno)
{
	for (UInt32 waited = 0;; waited++) {
		if (tryPost(type, payload, payloadSize, seqno))
			return kIOReturnSuccess;
		if (!ready)
			return kIOReturnNotReady;
		if (waited >= kPostTimeoutMs) {
			IOLog("ReimsGraphics: command %u: the control ring stayed full\n", type);
			return kIOReturnTimeout;
		}
		IOSleep(1);
	}
}

// Sends a command and waits for its answer, COMPLETE or DISPLAYS, which the
// work loop hands over: thread context only, never on the work loop.
IOReturn ReimsGraphics::command(UInt32 type, const void *payload, UInt32 payloadSize)
{
	UInt64 seqno = 0;
	IOReturn r = post(type, payload, payloadSize, &seqno);
	if (r != kIOReturnSuccess)
		return r;

	uint64_t deadline = 0;
	clock_interval_to_deadline(kAnswerTimeoutMs, kMillisecondScale, &deadline);
	UInt32 status = 0;
	bool found = false;
	IOLockLock(stateLock);
	while (!found) {
		for (UInt32 i = 0; i < kAnswers && !found; i++) {
			if (answers[i].seqno == seqno) {
				status = answers[i].status;
				found = true;
			}
		}
		if (!found && IOLockSleepDeadline(stateLock, answers, deadline, THREAD_UNINT) == THREAD_TIMED_OUT)
			break;
	}
	IOLockUnlock(stateLock);

	if (!found) {
		IOLog("ReimsGraphics: command %u (seqno %llu): no answer\n", type, seqno);
		return kIOReturnTimeout;
	}
	if (status != 0) {
		IOLog("ReimsGraphics: command %u (seqno %llu): status %u\n", type, seqno, status);
		return kIOReturnError;
	}
	return kIOReturnSuccess;
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
	volatile UInt8 *events = ring(RG_CONTROL_EVENT_OFFSET);
	for (;;) {
		drainEvents();
		store32(events + RG_RING_ARMED, 1);
		fullFence();
		if (load64(events + RG_RING_PRODUCER) == consumed)
			break;
	}

	IOSimpleLockLock(interruptLock);
	bool want = vblankWanted;
	IOSimpleLockUnlock(interruptLock);
	if (want != vblankSent) {
		struct rg_set_vblank v = {displayId, want ? 1u : 0u};
		if (tryPost(RG_CMD_SET_VBLANK, &v, sizeof(v), nullptr))
			vblankSent = want;
	}
	if (refreshWanted) {
		UInt64 seqno = 0;
		if (tryPost(RG_CMD_GET_DISPLAYS, nullptr, 0, &seqno)) {
			refreshWanted = false;
			refreshSeqno = seqno;
		}
	}
}

bool ReimsGraphics::drainEvents()
{
	volatile UInt8 *header = ring(RG_CONTROL_EVENT_OFFSET);
	volatile UInt8 *bytes = header + RG_RING_HEADER_SIZE;
	UInt64 end = load64(header + RG_RING_PRODUCER);
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
		if (type == RG_MSG_PAD) {
			consumed += tail;
			continue;
		}
		if (size < kHeaderBytes || size % RG_MSG_ALIGN || size > tail || size > end - consumed) {
			IOLog("ReimsGraphics: malformed event (type %u, size %u); skipping the ring's contents\n",
			      type, size);
			consumed = end;
			break;
		}
		handleEvent(type, seqno, m + kHeaderBytes, size - kHeaderBytes);
		consumed += size;
	}
	store64(header + RG_RING_CONSUMER, consumed);
	return any;
}

void ReimsGraphics::handleEvent(UInt32 type, UInt64 seqno, const volatile UInt8 *payload, UInt32 size)
{
	Interrupt fire = {};
	switch (type) {
	case RG_EVT_COMPLETE:
		answer(seqno, size >= sizeof(struct rg_complete) ? field32(payload, 0) : kStatusUnreadable);
		return;
	case RG_EVT_DISPLAYS: {
		answer(seqno, parseDisplays(payload, size));
		if (seqno != refreshSeqno)
			return;
		// The refresh a DISPLAY_CHANGED asked for: now IOFramebuffer
		// senses the connection again.
		refreshSeqno = 0;
		IOSimpleLockLock(interruptLock);
		fire = connect;
		IOSimpleLockUnlock(interruptLock);
		break;
	}
	case RG_EVT_DISPLAY_CHANGED:
		refreshWanted = true;
		return;
	case RG_EVT_VBLANK:
		IOSimpleLockLock(interruptLock);
		fire = vblank;
		IOSimpleLockUnlock(interruptLock);
		break;
	default:
		return;
	}
	if (fire.enabled && fire.proc)
		fire.proc(fire.target, fire.ref);
}

void ReimsGraphics::answer(UInt64 seqno, UInt32 status)
{
	IOLockLock(stateLock);
	answers[nextAnswer % kAnswers] = {seqno, status};
	nextAnswer++;
	IOLockWakeup(stateLock, answers, false);
	IOLockUnlock(stateLock);
}

// Takes the display's modes from a DISPLAYS answer. The driver's display is
// the first listed when it starts, and the display of that id after; a list
// without it leaves the display disconnected with its modes. Modes the
// scanout space cannot hold are left out. Returns the answer's status.
UInt32 ReimsGraphics::parseDisplays(const volatile UInt8 *payload, UInt32 size)
{
	if (size < sizeof(struct rg_displays))
		return kStatusUnreadable;
	UInt32 count = field32(payload, 0);
	UInt32 offset = sizeof(struct rg_displays);

	IOLockLock(stateLock);
	bool found = false;
	UInt32 id = 0, isConnected = 0, preferred = 0, listed = 0, listAt = 0;
	for (UInt32 i = 0; i < count; i++) {
		if (size - offset < sizeof(struct rg_display)) {
			IOLockUnlock(stateLock);
			return kStatusUnreadable;
		}
		UInt32 did = field32(payload, offset);
		UInt32 n = field32(payload, offset + 12);
		UInt32 at = offset + sizeof(struct rg_display);
		if (static_cast<UInt64>(n) * sizeof(struct rg_mode) > size - at) {
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
		offset = at + n * sizeof(struct rg_mode);
	}
	if (!found) {
		connected = false;
		IOLockUnlock(stateLock);
		return 0;
	}

	Mode fresh[kMaxModes];
	UInt32 freshCount = 0;
	IODisplayModeID freshPreferred = 0;
	UInt64 limit = scanoutLimit();
	for (UInt32 j = 0; j < listed && freshCount < kMaxModes; j++) {
		UInt32 at = listAt + j * sizeof(struct rg_mode);
		Mode m = {0, field32(payload, at), field32(payload, at + 4), field32(payload, at + 8)};
		if (!m.width || !m.height || m.width > kMaxDimension || m.height > kMaxDimension ||
		    static_cast<UInt64>(strideFor(m.width)) * m.height > limit) {
			IOLog("ReimsGraphics: display %u: mode %ux%u left out: the region cannot hold it\n",
			      id, m.width, m.height);
			continue;
		}
		for (UInt32 k = 0; k < modeCount && !m.id; k++) {
			if (modes[k].width == m.width && modes[k].height == m.height &&
			    modes[k].refreshMhz == m.refreshMhz)
				m.id = modes[k].id;
		}
		if (!m.id && haveCurrent && current.width == m.width && current.height == m.height &&
		    current.refreshMhz == m.refreshMhz)
			m.id = current.id;
		if (!m.id)
			m.id = nextModeId++;
		fresh[freshCount++] = m;
		if (j == preferred)
			freshPreferred = m.id;
	}
	haveDisplay = true;
	displayId = id;
	connected = isConnected != 0;
	memcpy(modes, fresh, freshCount * sizeof(Mode));
	modeCount = freshCount;
	preferredId = freshPreferred ? freshPreferred : (freshCount ? fresh[0].id : 0);
	IOLockUnlock(stateLock);
	return 0;
}

// MARK: - Modes and scanout

// The mode of an ID: one listed, or the mode set, which the list may have
// dropped. Call with stateLock held.
bool ReimsGraphics::modeFor(IODisplayModeID id, Mode *mode) const
{
	for (UInt32 i = 0; i < modeCount; i++) {
		if (modes[i].id == id) {
			*mode = modes[i];
			return true;
		}
	}
	if (haveCurrent && current.id == id) {
		*mode = current;
		return true;
	}
	return false;
}

UInt32 ReimsGraphics::strideFor(UInt32 width)
{
	return (width * 4 + kStrideAlign - 1) & ~(kStrideAlign - 1);
}

// The scanout buffer's room: from the guest's allocations up to the cursor.
UInt64 ReimsGraphics::scanoutLimit() const
{
	return regionSize - RG_GUEST_ALLOC_OFFSET - kCursorSlot;
}

IOReturn ReimsGraphics::setScanout(const Mode &mode)
{
	UInt32 stride = strideFor(mode.width);
	if (static_cast<UInt64>(stride) * mode.height > scanoutLimit())
		return kIOReturnNoSpace;
	struct rg_set_scanout s = {displayId, mode.width, mode.height, stride, RG_GUEST_ALLOC_OFFSET};
	IOReturn r = command(RG_CMD_SET_SCANOUT, &s, sizeof(s));
	if (r != kIOReturnSuccess)
		return r;
	IOLockLock(stateLock);
	current = mode;
	haveCurrent = true;
	IOLockUnlock(stateLock);
	IOLog("ReimsGraphics: display %u: scanout %ux%u at %u.%03u Hz, stride %u\n", displayId,
	      mode.width, mode.height, mode.refreshMhz / 1000, mode.refreshMhz % 1000, stride);
	return kIOReturnSuccess;
}

// MARK: - IOService

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
	nextModeId = 1;

	pci->setMemoryEnable(true);
	pci->setBusLeadEnable(true);

	int msixIndex = -1;
	IOReturn r = kIOReturnSuccess;
	UInt32 count = 0;

	registerMap = pci->mapDeviceMemoryWithRegister(kIOPCIConfigBaseAddress0);
	if (!registerMap) {
		IOLog("ReimsGraphics: BAR0 not mapped\n");
		goto fail;
	}
	registers = reinterpret_cast<volatile UInt8 *>(registerMap->getVirtualAddress());
	if (readRegister(RG_REG_MAGIC) != RG_MAGIC || readRegister(RG_REG_VERSION) != RG_VERSION) {
		IOLog("ReimsGraphics: not the device: magic 0x%x, version %u\n",
		      readRegister(RG_REG_MAGIC), readRegister(RG_REG_VERSION));
		goto fail;
	}

	region = pci->getDeviceMemoryWithRegister(kIOPCIConfigBaseAddress2);
	if (!region) {
		IOLog("ReimsGraphics: no BAR2\n");
		goto fail;
	}
	region->retain();
	regionSize = (static_cast<UInt64>(readRegister(RG_REG_REGION_SIZE_HI)) << 32) |
	             readRegister(RG_REG_REGION_SIZE_LO);
	if (regionSize > region->getLength())
		regionSize = region->getLength();
	if (regionSize <= RG_GUEST_ALLOC_OFFSET + kCursorSlot) {
		IOLog("ReimsGraphics: the region is too small: 0x%llx bytes\n", regionSize);
		goto fail;
	}
	ringMap = region->createMappingInTask(kernel_task, 0, kIOMapAnywhere | kIOMapCopybackCache, 0,
	                                      RG_GUEST_ALLOC_OFFSET);
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

	// The device from scratch: reset, then take the rings' counters as
	// they stand and say the driver is ready.
	writeRegister(RG_REG_STATUS, RG_STATUS_RESET);
	produced = load64(ring(RG_CONTROL_SUBMIT_OFFSET) + RG_RING_PRODUCER);
	consumed = load64(ring(RG_CONTROL_EVENT_OFFSET) + RG_RING_CONSUMER);
	writeRegister(RG_REG_STATUS, RG_STATUS_DRIVER_READY);
	ready = true;
	msix->enable();
	kick->enable();
	kickWorkLoop();

	r = command(RG_CMD_GET_DISPLAYS, nullptr, 0);
	IOLockLock(stateLock);
	startupId = preferredId;
	count = haveDisplay ? modeCount : 0;
	IOLockUnlock(stateLock);
	if (r != kIOReturnSuccess || !count) {
		IOLog("ReimsGraphics: the device lists no usable display\n");
		goto fail;
	}
	IOLog("ReimsGraphics: display %u, %u modes, region 0x%llx bytes\n", displayId, count, regionSize);

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

void ReimsGraphics::teardown()
{
	if (registers)
		writeRegister(RG_REG_STATUS, RG_STATUS_RESET);
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
	super::free();
}

// MARK: - IOFramebuffer: the device and its modes

IOReturn ReimsGraphics::enableController()
{
	Mode m;
	IOLockLock(stateLock);
	bool ok = modeFor(startupId, &m);
	IOLockUnlock(stateLock);
	return ok ? setScanout(m) : kIOReturnNotReady;
}

IODeviceMemory *ReimsGraphics::getApertureRange(IOPixelAperture aperture)
{
	if (aperture != kIOFBSystemAperture || !region)
		return nullptr;
	Mode m;
	IOLockLock(stateLock);
	bool ok = haveCurrent ? (m = current, true) : modeFor(startupId, &m);
	IOLockUnlock(stateLock);
	if (!ok)
		return nullptr;
	UInt64 length = round_page_64(static_cast<UInt64>(strideFor(m.width)) * m.height);
	return IODeviceMemory::withSubRange(region, RG_GUEST_ALLOC_OFFSET, length);
}

IODeviceMemory *ReimsGraphics::getVRAMRange()
{
	if (!region)
		return nullptr;
	return IODeviceMemory::withSubRange(region, RG_GUEST_ALLOC_OFFSET, scanoutLimit());
}

const char *ReimsGraphics::getPixelFormats()
{
	static const char formats[] = IO32BitDirectPixels "\0";
	return formats;
}

IOItemCount ReimsGraphics::getDisplayModeCount()
{
	IOLockLock(stateLock);
	reportedCount = modeCount;
	IOLockUnlock(stateLock);
	return reportedCount;
}

// The caller's array holds the count last reported. A DISPLAYS answer may
// change the list in between: then the array is filled with the preferred
// mode, and the connect interrupt that follows has IOFramebuffer ask again.
IOReturn ReimsGraphics::getDisplayModes(IODisplayModeID *allDisplayModes)
{
	IOLockLock(stateLock);
	for (UInt32 i = 0; i < reportedCount; i++)
		allDisplayModes[i] = i < modeCount ? modes[i].id : preferredId;
	IOLockUnlock(stateLock);
	return kIOReturnSuccess;
}

IOReturn ReimsGraphics::getInformationForDisplayMode(IODisplayModeID displayMode,
                                                     IODisplayModeInformation *info)
{
	Mode m;
	IOLockLock(stateLock);
	bool ok = modeFor(displayMode, &m);
	bool preferred = displayMode == preferredId;
	IOLockUnlock(stateLock);
	if (!ok || !info)
		return kIOReturnBadArgument;
	bzero(info, sizeof(*info));
	info->nominalWidth = m.width;
	info->nominalHeight = m.height;
	info->refreshRate = static_cast<IOFixed1616>((static_cast<UInt64>(m.refreshMhz) << 16) / 1000);
	info->maxDepthIndex = 0;
	info->flags = kDisplayModeValidFlag | kDisplayModeSafeFlag | (preferred ? kDisplayModeDefaultFlag : 0);
	return kIOReturnSuccess;
}

UInt64 ReimsGraphics::getPixelFormatsForDisplayMode(IODisplayModeID, IOIndex)
{
	return 0;
}

IOReturn ReimsGraphics::getPixelInformation(IODisplayModeID displayMode, IOIndex depth,
                                            IOPixelAperture aperture, IOPixelInformation *pixelInfo)
{
	if (depth != 0 || aperture != kIOFBSystemAperture || !pixelInfo)
		return kIOReturnUnsupportedMode;
	Mode m;
	IOLockLock(stateLock);
	bool ok = modeFor(displayMode, &m);
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
	IODisplayModeID id = haveCurrent ? current.id : startupId;
	IOLockUnlock(stateLock);
	if (displayMode)
		*displayMode = id;
	if (depth)
		*depth = 0;
	return kIOReturnSuccess;
}

IOReturn ReimsGraphics::setDisplayMode(IODisplayModeID displayMode, IOIndex depth)
{
	if (depth != 0)
		return kIOReturnUnsupportedMode;
	Mode m;
	IOLockLock(stateLock);
	bool ok = modeFor(displayMode, &m);
	IOLockUnlock(stateLock);
	return ok ? setScanout(m) : kIOReturnBadArgument;
}

IOReturn ReimsGraphics::getStartupDisplayMode(IODisplayModeID *displayMode, IOIndex *depth)
{
	IOLockLock(stateLock);
	IODisplayModeID id = startupId;
	IOLockUnlock(stateLock);
	if (displayMode)
		*displayMode = id;
	if (depth)
		*depth = 0;
	return kIOReturnSuccess;
}

IOReturn ReimsGraphics::setStartupDisplayMode(IODisplayModeID displayMode, IOIndex depth)
{
	if (depth != 0)
		return kIOReturnUnsupportedMode;
	Mode m;
	IOLockLock(stateLock);
	bool ok = modeFor(displayMode, &m);
	if (ok)
		startupId = displayMode;
	IOLockUnlock(stateLock);
	return ok ? kIOReturnSuccess : kIOReturnBadArgument;
}

// The display has no EDID: each mode is a fixed-rate panel's, with no
// blanking, so the pixel clock gives back the mode's rate.
IOReturn ReimsGraphics::getTimingInfoForDisplayMode(IODisplayModeID displayMode,
                                                    IOTimingInformation *info)
{
	Mode m;
	IOLockLock(stateLock);
	bool ok = modeFor(displayMode, &m);
	IOLockUnlock(stateLock);
	if (!ok || !info)
		return kIOReturnBadArgument;
	bzero(info, sizeof(*info));
	info->appleTimingID = kIOTimingIDApple_FixedRateLCD;
	info->flags = kIODetailedTimingValid;
	IODetailedTimingInformationV2 &t = info->detailedInfo.v2;
	t.pixelClock = static_cast<UInt64>(m.width) * m.height * m.refreshMhz / 1000;
	t.minPixelClock = t.pixelClock;
	t.maxPixelClock = t.pixelClock;
	t.horizontalActive = m.width;
	t.verticalActive = m.height;
	return kIOReturnSuccess;
}

// MARK: - IOFramebuffer: attributes and the display's connection

IOReturn ReimsGraphics::getAttribute(IOSelect attribute, uintptr_t *value)
{
	if (attribute == kIOHardwareCursorAttribute) {
		if (value)
			*value = 1;
		return kIOReturnSuccess;
	}
	return super::getAttribute(attribute, value);
}

IOReturn ReimsGraphics::setAttribute(IOSelect attribute, uintptr_t value)
{
	// The device has no power states of its own.
	if (attribute == kIOPowerAttribute)
		return kIOReturnSuccess;
	return super::setAttribute(attribute, value);
}

IOItemCount ReimsGraphics::getConnectionCount()
{
	return 1;
}

IOReturn ReimsGraphics::getAttributeForConnection(IOIndex connectIndex, IOSelect attribute,
                                                  uintptr_t *value)
{
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
	case kConnectionDisplayParameterCount:
		if (!value)
			return kIOReturnBadArgument;
		*value = 0;
		return kIOReturnSuccess;
	case kConnectionSupportsHLDDCSense:
	case kConnectionSupportsLLDDCSense:
	case kConnectionSupportsAppleSense:
		// No EDID and no sense lines.
		return kIOReturnUnsupported;
	default:
		return super::getAttributeForConnection(connectIndex, attribute, value);
	}
}

IOReturn ReimsGraphics::setAttributeForConnection(IOIndex connectIndex, IOSelect attribute,
                                                  uintptr_t value)
{
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
	struct rg_set_cursor c = {displayId, info.cursorWidth, info.cursorHeight, info.cursorHotSpotX,
	                          info.cursorHotSpotY, 0, regionSize - kCursorSlot};
	IOReturn r = command(RG_CMD_SET_CURSOR, &c, sizeof(c));
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
	struct rg_move_cursor m = {displayId, x + static_cast<SInt32>(hotX), y + static_cast<SInt32>(hotY),
	                           visible ? 1u : 0u};
	return tryPost(RG_CMD_MOVE_CURSOR, &m, sizeof(m), nullptr) ? kIOReturnSuccess : kIOReturnBusy;
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
