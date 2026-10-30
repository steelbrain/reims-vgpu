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
 * those its answer (a COMPLETE, a pipeline's reflection) by seqno.
 */

#ifndef REIMS_GRAPHICS_HPP_
#define REIMS_GRAPHICS_HPP_

#include <IOKit/IOInterruptEventSource.h>
#include <IOKit/IOLocks.h>
#include <IOKit/IOWorkLoop.h>
#include <IOKit/graphics/IOFramebuffer.h>
#include <IOKit/pci/IOPCIDevice.h>

#include "reims_graphics_wire.h"

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

	// IOFramebuffer: attributes and the display's connection
	IOReturn getAttribute(IOSelect attribute, uintptr_t *value) override;
	IOReturn setAttribute(IOSelect attribute, uintptr_t value) override;
	IOItemCount getConnectionCount() override;
	IOReturn getAttributeForConnection(IOIndex connectIndex, IOSelect attribute,
	                                   uintptr_t *value) override;
	IOReturn setAttributeForConnection(IOIndex connectIndex, IOSelect attribute,
	                                   uintptr_t value) override;

	// IOFramebuffer: the display's EDID, made by the driver from the display's modes
	// (makeEdid), which IODisplay reads for the display's identity.
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
	IOReturn newUserClient(task_t owningTask, void *securityID, UInt32 type,
	                       IOUserClient **clientH) override;
	IOReturn requestProbe(IOOptionBits options) override;
	IOReturn message(UInt32 type, IOService *provider, void *argument = 0) override;
	IOReturn setupForCurrentConfig() override;
	bool isConsoleDevice() override;
	IOReturn setCLUTWithEntries(IOColorEntry *colors, UInt32 index, UInt32 numEntries,
	                            IOOptionBits options) override;
	IOReturn validateDetailedTiming(void *description, IOByteCount descripSize) override;
	IOReturn setDetailedTimings(OSArray *array) override;
	IOReturn connectFlags(IOIndex connectIndex, IODisplayModeID displayMode,
	                      IOOptionBits *flags) override;
#endif

	// The display pipe (RgfxDisplayPipeUserClient, on the accelerator) presents through the
	// framebuffer, which owns the control ring. postPresent puts PRESENT_SURFACE for the
	// IOSurface of global ID `surfaceId` on the ring without waiting (thread context only); the
	// host answers it once the display shows the surface. setPipe names the pipe's client
	// (retained) and its hooks, both called on the work loop: `present(client, seqno, status)`
	// for every COMPLETE (the pipe picks its own seqnos), and, while setPipeVblankEnabled is on,
	// `vblank(client, count, time)` for each VBLANK, `time` the guest's mach_absolute_time at the
	// event. nullptr clears them.
	typedef void (*PipeVblankHook)(OSObject *client, UInt64 count, UInt64 time);
	typedef void (*PipePresentHook)(OSObject *client, UInt64 seqno, UInt32 status);
	IOReturn postPresent(UInt32 surfaceId, UInt32 transactionId, UInt64 *seqno);
	void setPipe(OSObject *client, PipeVblankHook vblank, PipePresentHook present);
	void setPipeVblankEnabled(bool enabled);

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

private:
#if RGFX_FB_TRACE
	IOReturn connectionAttribute(IOIndex connectIndex, IOSelect attribute, uintptr_t *value);
	IOReturn setConnectionAttribute(IOIndex connectIndex, IOSelect attribute, uintptr_t value);
#endif

	// The display's modes. IDs stay with a mode (its size and rate) for the
	// driver's life, as IOFramebuffer asks, and are never reused.
	struct Mode {
		IODisplayModeID id;
		UInt32 width;
		UInt32 height;
		UInt32 refreshMhz;
	};
	static constexpr UInt32 kMaxModes = 64;

	// A mode's timing: the active size with reduced blanking (VESA CVT's second reduced
	// blanking: a fixed 80-pixel horizontal blank and at least 460 us of vertical blank),
	// its pixel clock in 10 kHz steps, as an EDID carries it. The EDID and
	// getTimingInfoForDisplayMode both give it, so IODisplay and WindowServer find the
	// driver's modes among the display's.
	struct Timing {
		UInt64 pixelClock;
		UInt32 hActive, hBlank, hFront, hSync;
		UInt32 vActive, vBlank, vFront, vSync;
	};
	static void timingFor(const Mode &mode, Timing *timing);

	// The display's EDID: a base block, then DisplayID extension blocks listing every mode,
	// since a mode's pixel clock can be past the base block's limit. makeEdid fills block
	// `index` (0 the base) and returns false past the last. Call with stateLock held.
	static constexpr UInt32 kEdidBlockBytes = 128;
	static constexpr UInt32 kEdidModesPerBlock = 5;
	static constexpr UInt32 kEdidMaxExtensions = (kMaxModes + kEdidModesPerBlock - 1) / kEdidModesPerBlock;
	UInt32 edidExtensionCount() const;
	bool makeEdid(UInt32 index, UInt8 *block) const;

	// Power: IOFramebuffer sets kIOPowerAttribute to one of the driver's three power
	// states (off, doze, on); the driver tells IOFramebuffer as it leaves and reaches on.
	static constexpr UInt32 kPowerStateOff = 0;
	static constexpr UInt32 kPowerStateDoze = 1;
	static constexpr UInt32 kPowerStateOn = 2;
	static constexpr UInt32 kPowerStateCount = 3;
	void initPower();
	IOReturn setPower(UInt32 state);

	// One of IOFramebuffer's interrupt registrations.
	struct Interrupt {
		IOFBInterruptProc proc;
		OSObject *target;
		void *ref;
		bool enabled;
	};

	// Recent answers (COMPLETE and DISPLAYS) by seqno, for the waiting
	// threads. A waiter may look after its answer has come.
	struct Answer {
		UInt64 seqno;
		UInt32 status;
	};
	static constexpr UInt32 kAnswers = 16;

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

	// Modes and scanout.
	bool modeFor(IODisplayModeID id, Mode *mode) const;
	static UInt32 strideFor(UInt32 width);
	UInt64 scanoutLimit() const;
	IOReturn setScanout(const Mode &mode);

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

	// Under stateLock: answers, the lifecycle's waiters, the display and its modes, the mode
	// set.
	IOLock *stateLock;
	Answer answers[kAnswers];
	UInt32 nextAnswer;
	LifecycleWaiter lifecycleWaiters[kLifecycleWaiters];
	bool haveDisplay;
	UInt32 displayId;
	bool connected;
	Mode modes[kMaxModes];
	UInt32 modeCount;
	UInt32 reportedCount;
	IODisplayModeID preferredId;
	IODisplayModeID nextModeId;
	bool haveCurrent;
	Mode current;
	IODisplayModeID startupId;

	// Under interruptLock: IOFramebuffer's registrations, and whether it
	// wants VBLANK events.
	IOSimpleLock *interruptLock;
	Interrupt vblank;
	Interrupt connect;
	bool vblankWanted;
	// The display pipe's client (retained), its hooks, and whether it wants VBLANK.
	OSObject *pipeClient;
	PipeVblankHook pipeVblankHook;
	PipePresentHook pipePresentHook;
	bool pipeVblankEnabled;

	// The work loop's alone: what SET_VBLANK last asked, and a display
	// refresh after DISPLAY_CHANGED, waiting to be sent or answered.
	bool vblankSent;
	bool refreshWanted;
	UInt64 refreshSeqno;

	// IOFramebuffer's thread alone: the power state and whether it is registered.
	bool powerRegistered;
	UInt32 powerState;

	bool ready;
};

#endif // REIMS_GRAPHICS_HPP_
