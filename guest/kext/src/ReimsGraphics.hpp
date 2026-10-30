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

	// IOFramebuffer: the hardware cursor
	IOReturn setCursorImage(void *cursorImage) override;
	IOReturn setCursorState(SInt32 x, SInt32 y, bool visible) override;

	// IOFramebuffer: vblank and connect interrupts
	IOReturn registerForInterruptType(IOSelect interruptType, IOFBInterruptProc proc,
	                                  OSObject *target, void *ref,
	                                  void **interruptRef) override;
	IOReturn unregisterInterrupt(void *interruptRef) override;
	IOReturn setInterruptState(void *interruptRef, UInt32 state) override;

private:
	// The display's modes. IDs stay with a mode (its size and rate) for the
	// driver's life, as IOFramebuffer asks, and are never reused.
	struct Mode {
		IODisplayModeID id;
		UInt32 width;
		UInt32 height;
		UInt32 refreshMhz;
	};
	static constexpr UInt32 kMaxModes = 64;

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

	// Registers and the control rings.
	UInt32 readRegister(UInt32 offset) const;
	void writeRegister(UInt32 offset, UInt32 value);
	volatile UInt8 *ring(UInt32 offset) const;
	bool tryPost(UInt32 type, const void *payload, UInt32 payloadSize, UInt64 *seqno);
	IOReturn post(UInt32 type, const void *payload, UInt32 payloadSize, UInt64 *seqno);
	IOReturn command(UInt32 type, const void *payload, UInt32 payloadSize);
	void kickWorkLoop();

	// The work loop: the completion ring and what follows from it.
	static void serviceAction(OSObject *owner, IOInterruptEventSource *sender, int count);
	void service();
	bool drainEvents();
	void handleEvent(UInt32 type, UInt64 seqno, const volatile UInt8 *payload, UInt32 size);
	UInt32 parseDisplays(const volatile UInt8 *payload, UInt32 size);
	void answer(UInt64 seqno, UInt32 status);

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

	// Under stateLock: answers, the display and its modes, the mode set.
	IOLock *stateLock;
	Answer answers[kAnswers];
	UInt32 nextAnswer;
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

	// The work loop's alone: what SET_VBLANK last asked, and a display
	// refresh after DISPLAY_CHANGED, waiting to be sent or answered.
	bool vblankSent;
	bool refreshWanted;
	UInt64 refreshSeqno;

	bool ready;
};

#endif // REIMS_GRAPHICS_HPP_
