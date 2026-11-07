/*
 * ReimsGraphics: the reims-graphics device's display driver for x86-64 macOS
 * guests, an IOFramebuffer on the PCI device (docs/graphics-kext/design.md,
 * Decided 12 and "Stage 2 wire protocol").
 *
 * It drives one display, the first the device lists. WindowServer composites
 * in software into the scanout buffer, which sits in BAR2 at the start of the
 * guest's allocations; the device presents it, so the driver itself never
 * touches pixels. Everything else is the control ring: modes, scanout,
 * cursor, vblank and display changes.
 *
 * Threads: IOFramebuffer calls in on its own threads. Commands with an answer
 * wait for it, and commands without one (MOVE_CURSOR, SET_VBLANK) never
 * block, since IOFramebuffer may move the cursor with a spin lock held. The
 * driver's work loop is the control completion ring's only consumer: it
 * drains it on MSI-X vector 0, hands answers to the waiting threads, and
 * turns VBLANK and DISPLAY_CHANGED into IOFramebuffer's interrupts.
 *
 * The control ring is the guest's one ring to the device besides the command
 * queues' own, so the accelerator posts the Metal processes' object lifecycle
 * on it too (postLifecycle), under the same lock; the work loop hands each of
 * those its answer (a COMPLETE, a pipeline's reflection, a heap texture
 * size, a sampler's resource id) by seqno.
 */

#ifndef REIMS_GRAPHICS_HPP_
#define REIMS_GRAPHICS_HPP_

#include <IOKit/IOInterruptEventSource.h>
#include <IOKit/IOLocks.h>
#include <IOKit/IOWorkLoop.h>
#include <IOKit/graphics/IOFramebuffer.h>
#include <IOKit/pci/IOPCIDevice.h>

#include "RgfxBootConsole.hpp"
#include "RgfxDeferredAnswers.hpp"
#include "RgfxDisplayModes.hpp"
#include "RgfxPipeClients.hpp"
#include "RgfxRingRoom.hpp"
#include "RgfxScanoutOwner.hpp"
#include "RgfxVblankClock.hpp"
#include "reims_graphics_wire.h"

class RgfxAccelerator;

class ReimsGraphics : public IOFramebuffer {
	OSDeclareDefaultStructors(ReimsGraphics);

public:
	// IOService
	bool start(IOService *provider) override;
	void stop(IOService *provider) override;
	void free() override;

	// IOFramebuffer: the device and its modes
	IOReturn enableController() override;
	IODeviceMemory *getApertureRange(IOPixelAperture aperture) override;
	IODeviceMemory *getVRAMRange() override;
	const char *getPixelFormats() override;
	IOItemCount getDisplayModeCount() override;
	IOReturn getDisplayModes(IODisplayModeID *allDisplayModes) override;
	IOReturn getInformationForDisplayMode(IODisplayModeID displayMode,
	                                      IODisplayModeInformation *info) override;
	UInt64 getPixelFormatsForDisplayMode(IODisplayModeID displayMode, IOIndex depth) override;
	IOReturn getPixelInformation(IODisplayModeID displayMode, IOIndex depth,
	                             IOPixelAperture aperture, IOPixelInformation *pixelInfo) override;
	IOReturn getCurrentDisplayMode(IODisplayModeID *displayMode, IOIndex *depth) override;
	IOReturn setDisplayMode(IODisplayModeID displayMode, IOIndex depth) override;
	IOReturn getStartupDisplayMode(IODisplayModeID *displayMode, IOIndex *depth) override;
	IOReturn setStartupDisplayMode(IODisplayModeID displayMode, IOIndex depth) override;
	IOReturn getTimingInfoForDisplayMode(IODisplayModeID displayMode,
	                                     IOTimingInformation *info) override;
	// IODisplay: a mode's valid, safe and default flags, which IOFramebuffer gives in place of
	// those getInformationForDisplayMode answers (modeFlags).
	IOReturn connectFlags(IOIndex connectIndex, IODisplayModeID displayMode,
	                      IOOptionBits *flags) override;

	// IOFramebuffer: the timings the OS programs, under the framebuffer's
	// kIOFBTimingRangeKey. The driver takes a timing whose size and rate are a
	// mode the device lists, and lists the programmed ones in place of its own.
	IOReturn validateDetailedTiming(void *description, IOByteCount descripSize) override;
	IOReturn setDetailedTimings(OSArray *array) override;

	// IOFramebuffer: attributes and the display's connection
	IOReturn getAttribute(IOSelect attribute, uintptr_t *value) override;
	IOReturn setAttribute(IOSelect attribute, uintptr_t value) override;
	IOItemCount getConnectionCount() override;
	IOReturn getAttributeForConnection(IOIndex connectIndex, IOSelect attribute,
	                                   uintptr_t *value) override;
	IOReturn setAttributeForConnection(IOIndex connectIndex, IOSelect attribute,
	                                   uintptr_t value) override;

	// IOFramebuffer: the display's EDID, made by the driver from the display's native mode
	// (makeEdid), which IODisplay reads for the display's identity and size.
	bool hasDDCConnect(IOIndex connectIndex) override;
	IOReturn getDDCBlock(IOIndex connectIndex, UInt32 blockNumber, IOSelect blockType,
	                     IOOptionBits options, UInt8 *data, IOByteCount *length) override;

	// IOFramebuffer: the hardware cursor
	IOReturn setCursorImage(void *cursorImage) override;
	IOReturn setCursorState(SInt32 x, SInt32 y, bool visible) override;

	// IOFramebuffer: vblank and connect interrupts
	IOReturn registerForInterruptType(IOSelect interruptType, IOFBInterruptProc proc,
	                                  OSObject *target, void *ref,
	                                  void **interruptRef) override;
	IOReturn unregisterInterrupt(void *interruptRef) override;
	IOReturn setInterruptState(void *interruptRef, UInt32 state) override;

	// IOService: the user clients IOFramebuffer makes. The task a server connection
	// (kIOFBServerConnectType) is made for owns the scanout from then on (ownsScanout).
	IOReturn newUserClient(task_t owningTask, void *securityID, UInt32 type,
	                       IOUserClient **clientH) override;
	// IOFramebuffer's setup for the current mode, which gets no VRAM range (getVRAMRange).
	IOReturn setupForCurrentConfig() override;

#if RGFX_FB_TRACE
	// A trace build (make TRACE=1) logs what IOFramebuffer and its clients call on the
	// driver, forwarding each of these to IOFramebuffer.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Woverloaded-virtual"
	IOReturn open() override;
	void close() override;
	IOReturn setGammaTable(UInt32 channelCount, UInt32 dataCount, UInt32 dataWidth,
	                       void *data) override;
#pragma clang diagnostic pop
	IOReturn setPowerState(unsigned long powerStateOrdinal, IOService *device) override;
	IOReturn powerStateWillChangeTo(IOPMPowerFlags flags, unsigned long state,
	                                IOService *device) override;
	IOReturn powerStateDidChangeTo(IOPMPowerFlags flags, unsigned long state,
	                               IOService *device) override;
	IOReturn requestProbe(IOOptionBits options) override;
	IOReturn message(UInt32 type, IOService *provider, void *argument = 0) override;
	bool isConsoleDevice() override;
	IOReturn setCLUTWithEntries(IOColorEntry *colors, UInt32 index, UInt32 numEntries,
	                            IOOptionBits options) override;
	// And the properties processes read of it.
	using IOFramebuffer::copyProperty;
	OSObject *copyProperty(const OSSymbol *aKey) const override;
	OSObject *copyProperty(const OSString *aKey) const override;
	OSObject *copyProperty(const char *aKey) const override;
#endif

	// The display pipe (RgfxDisplayPipeUserClient, on the accelerator) presents through the
	// framebuffer, which owns the control ring. postPresent puts PRESENT_SURFACE for the
	// IOSurface of global ID `surfaceId` on the ring without waiting (thread context only),
	// answering kIOReturnNoSpace where the ring has no room: the caller waits for room with no
	// lock of the work loop's held (roomMark, waitForRoom), as the work loop hands the present's
	// answer over under the caller's lock. The host answers it once the display shows the
	// surface. addPipe adds an associated pipe client
	// (retained; RgfxPipeClients.hpp: IOPresentment associates several), and the hooks, both
	// called on the work loop for each client: `present(client, seqno, status, time)` for every
	// COMPLETE (each pipe client picks its own seqnos), `time` that of the refresh whose answer
	// it is, and, while the client's setPipeVblankEnabled is on, `vblank(client, count, time,
	// next)` for each VBLANK, `time` the refresh's and `next` the next one's. Times are the
	// guest's mach_absolute_time, the host's stamps mapped into it (RgfxVblankClock.hpp).
	// removePipe lets one go; the last one's going shows the scanout again where the pipe
	// showed a surface (thread context only).
	// The scanout as it is now (setScanout): its shape, bytes per row and offset in the region.
	// False before a mode is set.
	bool currentScanout(UInt32 *width, UInt32 *height, UInt32 *bytesPerRow, UInt64 *offset);
	// Whether `task` owns the scanout: it holds the framebuffer's server connection, or its open
	// of one is in flight (RgfxScanoutOwner.hpp). Only the owner reaches the display's pixels.
	bool ownsScanout(task_t task);
	// The scanout's owner moved to `owner`: the accelerator takes every other task's hold of the
	// scanout back (RgfxAccelerator::scanoutOwnerMoved). With no lock held.
	void scanoutOwnerMoved(task_t owner);
	// The accelerator on our PCI device, retained for the caller, or nullptr before it started.
	RgfxAccelerator *copyAccelerator();

	typedef void (*PipeVblankHook)(OSObject *client, UInt64 count, UInt64 time, UInt64 next);
	typedef void (*PipePresentHook)(OSObject *client, UInt64 seqno, UInt32 status, UInt64 time);
	// `flags` RGFX_PRESENT_KEEP, with `surfaceId` 0, keeps what the display shows: the present
	// shows only the colour posted before it.
	IOReturn postPresent(UInt32 surfaceId, UInt32 transactionId, UInt32 flags, UInt64 *seqno);
	// The display pipe's colour (RgfxPipeColor.hpp): displayColor is whether the host took
	// RGFX_FEATURE_DISPLAY_COLOR; postDisplayColor puts stage `stage`'s `count` entries (16
	// bytes each at `entries`) for transaction `transactionId` on the control ring as
	// DISPLAY_COLOR parts, in order, each waiting for room as post does: thread context only,
	// with no lock of the work loop's held. The host stages them for the next PRESENT_SURFACE.
	bool displayColor() const { return displayColorTaken; }
	IOReturn postDisplayColor(UInt32 transactionId, UInt32 stage, UInt32 count,
	                          const UInt8 *entries);
	// A pipe client's transaction posts its colour, then its present, holding the presents lock
	// across both, so no other client's post comes between them: the host applies the colour
	// parts to the next present it takes, and drops a stage whose parts a present cuts short.
	// Thread context only; the work loop never takes it.
	void lockPresents() { IOLockLock(presentsLock); }
	void unlockPresents() { IOLockUnlock(presentsLock); }
	// Room on the control ring for a post that found it full (RgfxRingRoom.hpp): roomMark, read
	// before the post's look, is the work loop's passes so far; waitForRoom sleeps until a pass
	// after `mark` (each device interrupt makes one: the host has moved on), its backstop or
	// `deadline` (mach_absolute_time), and answers false once `deadline` has passed or the
	// device is not ready. Thread context only, never on the work loop.
	UInt64 roomMark() const;
	bool waitForRoom(UInt64 mark, uint64_t deadline);
	// Where the host took SUBMIT_ROOM: tells it that a post waits for room on the control ring
	// (producer_waiting 1, then a full fence), so its next take posts RGFX_EVT_SUBMIT_ROOM, whose
	// pass wakes the post. The caller then looks once more before waitForRoom: either that look
	// finds what the host took, or the host's take finds the flag. False, nothing done, where the
	// host did not take it.
	bool askForRoom();
	bool addPipe(OSObject *client, PipeVblankHook vblank, PipePresentHook present);
	void removePipe(OSObject *client);
	void setPipeVblankEnabled(OSObject *client, bool enabled);

	// The device's one MSI-X vector also carries the command queues' completion interrupts
	// (each QUEUE_CREATE names it). setQueueInterrupt names the accelerator (retained) and its
	// hook, which the work loop calls after each pass over the control ring, so the
	// accelerator looks at its queues' completion rings once per interrupt. nullptr clears it.
	typedef void (*QueueInterruptHook)(OSObject *target);
	void setQueueInterrupt(OSObject *target, QueueInterruptHook hook);

	// The accelerator's block releases, which no thread waits for. postReleases puts `count`
	// BLOCK_RELEASEs on the control ring in order, as many at a time as fit, under one hold of
	// its lock and with one doorbell at most, waiting for room as post does; it writes each one's
	// seqno into *waiterSeqnos[i] before the message is published, and counts those posted in
	// *posted, which is `count` on success. setReleaseAnswer names the accelerator (retained) and
	// its hook, which the work loop calls with each COMPLETE no lifecycle waiter takes, before the
	// display's; the hook answers whether the seqno was a release of its own. nullptr clears it.
	typedef bool (*ReleaseAnswerHook)(OSObject *target, UInt64 seqno, UInt32 status);
	void setReleaseAnswer(OSObject *target, ReleaseAnswerHook hook);
	IOReturn postReleases(const RgfxBlockRelease *releases, UInt32 count,
	                      UInt64 *const *waiterSeqnos, UInt32 *posted);
	// A lifecycle message (a CONTEXT_CLOSE) no thread waits for here, its COMPLETE handed to the
	// setReleaseAnswer hook: as postLifecycle with `replyType` 0, but it writes the seqno into
	// *waiterSeqno before the message is published, so the answer, however soon it comes, finds
	// the caller's entry. Thread context only.
	IOReturn postAnswered(UInt32 type, const void *message, UInt32 size, UInt64 *waiterSeqno);

	// The accelerator's object lifecycle (a create or destroy, QUEUE_CREATE, CONTEXT_CLOSE) on
	// the control ring: `message` is the whole message in kernel memory, at most
	// kLifecycleMessageMax bytes, whose header is stamped here with `type`, its size and the
	// seqno (*seqno, when not null). With `replyType` 0 it returns once the message is on the
	// ring; else it waits up to `timeoutMs` for the device's answer of that type to this
	// seqno and copies it, header and all, into `reply` (*replyLength its bytes, at most
	// `replyCapacity`). Thread context only, never on the work loop.
	static constexpr UInt32 kLifecycleMessageMax = RGFX_CONTROL_RING_DATA_SIZE / 2;
	static constexpr UInt32 kLifecycleReplyMax = 128;
	IOReturn postLifecycle(UInt32 type, const void *message, UInt32 size, UInt32 replyType,
	                       void *reply, UInt32 replyCapacity, UInt32 *replyLength, UInt64 *seqno,
	                       UInt32 timeoutMs);

	// A pipeline create of context `context` posted without waiting for its answer, which the
	// device's answer then waits for in a slot of the deferred answers (RgfxDeferredAnswers.hpp)
	// until takeDeferred takes it: as postLifecycle with `replyType` 0, but for its slot.
	// kIOReturnNoResources, posting nothing, where no slot is free: the caller posts the create
	// as postLifecycle does, and waits. Thread context only.
	IOReturn postDeferred(UInt32 context, UInt32 type, const void *message, UInt32 size,
	                      UInt64 *seqno);
	// Copies context `context`'s answers to its deferred creates into the `capacity` bytes at
	// `out`, each whole, in their seqnos' order, while they fit, and frees their slots: the bytes
	// in `*length`, the answers in `*count`. Where `waitFor` is the seqno of one of its creates
	// still waiting, waits up to `timeoutMs` for its answer first. Thread context only.
	IOReturn takeDeferred(UInt32 context, UInt64 waitFor, UInt32 timeoutMs, UInt8 *out,
	                      UInt32 capacity, UInt32 *length, UInt32 *count);
	// Frees context `context`'s deferred slots, at its close, and lets its waiters go.
	void dropDeferred(UInt32 context);
	// Frees the deferred slot of the create at `seqno`, posted after its context closed.
	void forgetDeferred(UInt64 seqno);

private:
#if RGFX_FB_TRACE
	IOReturn connectionAttribute(IOIndex connectIndex, IOSelect attribute, uintptr_t *value);
	IOReturn setConnectionAttribute(IOIndex connectIndex, IOSelect attribute, uintptr_t value);
#endif

	// The display's modes. IDs stay with a mode (its size and rate) for the
	// driver's life, as IOFramebuffer asks, and are never reused.
	// (RgfxDisplayModes.hpp: the list IOFramebuffer sees, the mode set and the startup mode.)
	using Mode = RgfxMode;
	static constexpr UInt32 kMaxModes = 64;
	using Modes = RgfxDisplayModes<IODetailedTimingInformationV2, kMaxModes>;
	// The most display pipe clients served at once (RgfxPipeClients.hpp).
	static constexpr UInt32 kMaxPipeClients = 8;

	// A mode's timing: the active size with reduced blanking (VESA CVT's second reduced
	// blanking: a fixed 80-pixel horizontal blank and at least 460 us of vertical blank),
	// its pixel clock in 10 kHz steps, as an EDID carries it. getTimingInfoForDisplayMode
	// gives it.
	struct Timing {
		UInt64 pixelClock;
		UInt32 hActive, hBlank, hFront, hSync;
		UInt32 vActive, vBlank, vFront, vSync;
	};
	static void timingFor(const Mode &mode, Timing *timing);

	// The display's EDID: one base block, with no timings. makeEdid fills block `index` (0,
	// the base) and returns false past it. Call with stateLock held.
	static constexpr UInt32 kEdidBlockBytes = 128;
	bool makeEdid(UInt32 index, UInt8 *block) const;

	static UInt32 refreshMhzOf(const IODetailedTimingInformationV2 &timing);
	// The device's preferred mode, the display's native one; `mode` where the device lists
	// none. Call with stateLock held.
	Mode nativeMode(const Mode &mode) const;
	// The physical size of a mode's display, in millimetres, as the EDID gives it.
	void imageSize(const Mode &mode, UInt16 *width, UInt16 *height) const;
	// The IODisplayModeInformation flags of a mode, the preferred one or not.
	static UInt32 modeFlags(const Mode &mode, bool preferred);

	// One of IOFramebuffer's interrupt registrations.
	struct Interrupt {
		IOFBInterruptProc proc;
		OSObject *target;
		void *ref;
		bool enabled;
	};

	// A thread waiting in command() for a display-control message's answer (a COMPLETE or a
	// DISPLAYS), keyed by seqno as the lifecycle waiters are. A burst of PRESENT_SURFACE
	// completions, which also answer on the control ring, then matches no waiter and is dropped,
	// rather than evicting a control answer from a fixed ring before its thread reads it. Free
	// while `waiting` is false; `seqno` is 0 until the message is on the ring, then the answer's
	// thread fills `status`, sets `done`, and wakes this waiter alone.
	struct ControlWaiter {
		UInt64 seqno;
		UInt32 status;
		bool waiting;
		bool done;
	};
	static constexpr UInt32 kControlWaiters = 8;

	// A thread waiting for a lifecycle message's answer: free while replyType is 0; its seqno
	// is 0 until the message is on the ring.
	struct LifecycleWaiter {
		UInt64 seqno;
		UInt32 replyType;
		UInt32 length;
		bool done;
		UInt8 reply[kLifecycleReplyMax];
	};
	static constexpr UInt32 kLifecycleWaiters = 32;

	// The deferred creates' answers (postDeferred): 256 slots, at most 64 a context.
	static constexpr UInt32 kDeferredAnswers = 256;
	static constexpr UInt32 kDeferredPerContext = 64;
	typedef RgfxDeferredAnswers<kDeferredAnswers, kDeferredPerContext, kLifecycleReplyMax>
	    DeferredAnswers;

	// Registers and the control rings.
	UInt32 readRegister(UInt32 offset) const;
	void writeRegister(UInt32 offset, UInt32 value);
	volatile UInt8 *ring(UInt32 offset) const;
	// A message is a full struct (RgfxSetScanout and the like) whose header
	// tryPost stamps with the type, size and seqno; the caller fills the rest.
	// `waiterSeqno`, when not null, gets the seqno before the message is
	// published, so its answer finds its waiter.
	bool tryPost(UInt32 type, const void *message, UInt32 size, UInt64 *seqno,
	             UInt64 *waiterSeqno = nullptr);
	IOReturn post(UInt32 type, const void *message, UInt32 size, UInt64 *seqno,
	              UInt64 *waiterSeqno = nullptr);
	// tryPost's halves, submitLock held: writeLocked writes one message where it fits, and
	// publishLocked publishes what was written.
	bool writeLocked(UInt32 type, const void *message, UInt32 size, UInt64 *seqno,
	                 UInt64 *waiterSeqno);
	void publishLocked();
	IOReturn command(UInt32 type, const void *message, UInt32 size);
	void kickWorkLoop();

	// The work loop: the completion ring and what follows from it.
	static void serviceAction(OSObject *owner, IOInterruptEventSource *sender, int count);
	void service();
	bool drainEvents();
	void handleEvent(UInt32 type, UInt64 seqno, const volatile UInt8 *payload, UInt32 size);
	UInt32 parseDisplays(const volatile UInt8 *payload, UInt32 size);
	void answer(UInt64 seqno, UInt32 status);
	bool answerLifecycle(UInt32 type, UInt64 seqno, const volatile UInt8 *payload, UInt32 size);

	// Ending: no more work loop actions, then the device and mappings go.
	void quiesce();
	void teardown();

	// The boot framebuffer the OS had at the start, published (RgfxBootConsole.hpp).
	void publishBootConsole();

	// Modes and scanout.
	static UInt32 strideFor(UInt32 width);
	UInt64 scanoutLimit() const;
	IOReturn setScanout(const Mode &mode);
	// The boot screen in the scanout, black, once the display first shows a surface (handleEvent).
	void clearBootScreen();

	IOPCIDevice *pci;
	IOMemoryMap *registerMap;
	volatile UInt8 *registers;
	IODeviceMemory *region;
	UInt64 regionSize;
	IOMemoryMap *ringMap;
	volatile UInt8 *rings;
	IOMemoryMap *cursorMap;
	UInt8 *cursorPixels;
	IOWorkLoop *workLoop;
	IOInterruptEventSource *msix;
	IOInterruptEventSource *kick;

	// The submission ring, under submitLock: the bytes produced and the
	// next seqno.
	IOSimpleLock *submitLock;
	UInt64 produced;
	UInt64 nextSeqno;

	// The cursor image's hotspot, which MOVE_CURSOR adds: set once
	// SET_CURSOR is answered.
	UInt32 hotX;
	UInt32 hotY;

	// The completion ring's bytes consumed: the work loop's alone.
	UInt64 consumed;

	// The control ring's room (roomMark, waitForRoom): the work loop's passes, which it moves
	// at each, and the posts waiting for room, which a pass wakes under stateLock. Atomic.
	RgfxRingRoom room;
	// The host took SUBMIT_ROOM (askForRoom), set before `ready`.
	bool submitRoom;
	// The host took DISPLAY_COLOR (displayColor), set before `ready`.
	bool displayColorTaken;
	// The host took HOST_TIME: each VBLANK reads the host's clock for the refreshes' times
	// (vblankClock), set before `ready`.
	bool hostTime;
	// The sleeps for room (waitForRoom), under stateLock, and the SUBMIT_ROOM records the work
	// loop took (atomic), for the log line waitForRoom writes at most every kRoomLogMs.
	RgfxRingRoomStats roomStats;
	UInt64 roomRecords;

	// The display pipe's presents (lockPresents), taken before stateLock where both are.
	IOLock *presentsLock;
	// Under stateLock: answers, the lifecycle's waiters, the display and its modes, the mode
	// set, and the scanout's owner (its task referenced, its connection retained).
	IOLock *stateLock;
	RgfxScanoutOwner scanoutOwner;
	ControlWaiter controlWaiters[kControlWaiters];
	LifecycleWaiter lifecycleWaiters[kLifecycleWaiters];
	// The deferred creates' answers; its waiters (takeDeferred) sleep on it, and each answer or
	// drop wakes them all.
	DeferredAnswers deferredAnswers;
	bool haveDisplay;
	UInt32 displayId;
	bool connected;
	Modes modeSet;
	RgfxModeConfig modeConfig;
	UInt32 reportedCount;

	// kIOWindowServerActiveAttribute's value as WindowServer last set it: 0
	// (kIOWSAA_Unaccelerated) before any set. IOFramebuffer gets and sets it under its gate.
	UInt32 windowServerActive;

	// Under interruptLock: IOFramebuffer's registrations, whether it wants
	// VBLANK events, and a display change waiting for its connect interrupt.
	IOSimpleLock *interruptLock;
	Interrupt vblank;
	Interrupt connect;
	RgfxConnectChange displayChange;
	bool vblankWanted;
	// The display pipe's clients (each retained), which want VBLANK, and the surface shown; the
	// hooks every client takes.
	RgfxPipeClients<OSObject *, kMaxPipeClients> pipes;
	PipeVblankHook pipeVblankHook;
	PipePresentHook pipePresentHook;
	// The accelerator (retained) and its queue-interrupt hook; and, the same, its release hook.
	OSObject *queueTarget;
	QueueInterruptHook queueHook;
	OSObject *releaseTarget;
	ReleaseAnswerHook releaseHook;

	// The work loop's alone: what SET_VBLANK last asked, the refreshes' times, and a display
	// refresh after DISPLAY_CHANGED, waiting to be sent or answered.
	bool vblankSent;
	RgfxVblankClock vblankClock;
	bool refreshWanted;
	UInt64 refreshSeqno;

	// The boot screen the scanout holds until clearBootScreen: the boot framebuffer the OS had at
	// the start (publishBootConsole); the largest span a mode's scanout has had (setScanout, under
	// stateLock); the seqno of the latest present of a surface before the clear, which tryPost
	// writes before the host sees the present; and whether it was cleared (the work loop's).
	RgfxBootConsole bootConsole;
	UInt64 largestScanout;
	UInt64 bootScreenSeqno;
	bool bootScreenCleared;

	bool ready;
	// start's and stop's alone: whether start reset the device to take it. Only a device taken is
	// reset when the kext lets it go; one refused before stays as the firmware left it.
	bool taken;
};

#endif // REIMS_GRAPHICS_HPP_
