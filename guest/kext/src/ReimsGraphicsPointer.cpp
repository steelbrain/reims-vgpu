/*
 * ReimsGraphicsPointer: the x86-64 framebuffer's move of the cursor onto the display the host's
 * pointer is over, on the device's POINTER_OVER (design.md, "Multiple displays", "The pointer
 * across displays"; RgfxPointerOver.hpp for the event and the point).
 *
 * macOS scales the VM's one USB tablet over the display its cursor is on, so the host's pointer
 * reaches display i once the cursor is there. The move is IOHIDSystem's own: a mouse-moved, or the
 * drag of the button held, posted with kIOHIDSetCursorPosition at the display's point of the
 * global space, as IOHIDPostEvent posts one for a process, which WindowServer follows. Measured
 * on x86-64 macOS 11.7.11, 15.8.2 and 26.7.2 with two displays (lane ptrcross, potluck jobs
 * m3-max-34583 to 34585): that post puts the cursor at its point of display 1, and the tablet's
 * next position lands on display 1; IOHIDSystem::extSetMouseLocation answers success and leaves
 * the cursor where it was; and a process's IOHIDSetMouseLocation and IOHIDPostEvent are refused
 * (kIOReturnNotPermitted), its CGEventPost dropped. IOHIDSystem lives in IOHIDFamily on those
 * releases, the library Info.plist names.
 *
 * The post runs on a thread call of display 0's framebuffer, off the device's work loop, as it
 * waits for IOHIDSystem's command gate. Each POINTER_OVER replaces the one before it that has not
 * moved the cursor yet: the latest is where the host's pointer is.
 *
 * The other way, each display's rectangle in the global space goes to the host in DISPLAY_BOUNDS
 * whenever it changed, posted after a move of the cursor (RgfxDisplayBounds.hpp), by which the
 * host carries its captured pointer past one display's edge onto the display there.
 */

#include "ReimsGraphics.hpp"
#include "RgfxTrace.hpp"

#include <IOKit/IOLib.h>
#include <IOKit/hidsystem/IOHIDSystem.h>
#include <IOKit/hidsystem/IOLLEvent.h>

namespace {

// IOHIDPostEvent's options (IOHIDLib.h, user space): the event's location is where the cursor
// goes.
constexpr int kSetCursorPosition = 0x2;

// IOHIDSystem's post of an event (IOHIDSystem.h), called by its symbol in IOHIDFamily:
// -fapple-kext would call it, a virtual, through the SDK's vtable slot, which need not be the
// guest's. Its first argument is the evioLLEvent; it answers kIOReturnNotOpen before WindowServer
// opened IOHIDSystem's events.
extern "C" IOReturn rgfxHidPostEvent(IOHIDSystem *self, void *event, void *, void *, void *,
                                     void *, void *)
    asm("__ZN11IOHIDSystem12extPostEventEPvS0_S0_S0_S0_S0_");

int eventTypeOf(RgfxPointerMove move)
{
	switch (move) {
	case RgfxPointerMove::leftDragged:
		return NX_LMOUSEDRAGGED;
	case RgfxPointerMove::rightDragged:
		return NX_RMOUSEDRAGGED;
	case RgfxPointerMove::otherDragged:
		return NX_OMOUSEDRAGGED;
	case RgfxPointerMove::moved:
		break;
	}
	return NX_MOUSEMOVED;
}

} // namespace

bool ReimsGraphics::startPointer()
{
	pointerAt = {};
	pointerCall = thread_call_allocate(movePointer, this);
	if (!pointerCall) {
		IOLog("ReimsGraphics: no thread call for the pointer's moves\n");
		return false;
	}
	return true;
}

void ReimsGraphics::freePointer()
{
	thread_call_t call = pointerCall;
	pointerCall = nullptr;
	if (!call)
		return;
	thread_call_cancel_wait(call);
	thread_call_free(call);
}

// handleEvent's POINTER_OVER, on the device's work loop: the display, point and buttons kept for
// the thread call, which moves the cursor.
void ReimsGraphics::pointerOver(const volatile UInt8 *payload, UInt32 size)
{
	RgfxPointerAt at = {};
	if (!pointerCall || !rgfxPointerOverRead(payload, size, &at)) {
		static char unread;
		if (rgfxFirstCalls(&unread, size, 1))
			IOLog("ReimsGraphics: a POINTER_OVER of %u bytes moves nothing\n", size);
		return;
	}
	IOSimpleLockLock(interruptLock);
	pointerAt = at;
	IOSimpleLockUnlock(interruptLock);
	thread_call_enter(pointerCall);
}

void ReimsGraphics::movePointer(thread_call_param_t self, thread_call_param_t)
{
	ReimsGraphics *me = static_cast<ReimsGraphics *>(self);
	IOSimpleLockLock(me->interruptLock);
	const RgfxPointerAt at = me->pointerAt;
	IOSimpleLockUnlock(me->interruptLock);
	if (!me->hasStarted())
		return;
	// The display's mode and its rectangle in the global space.
	ReimsGraphics *fb = me->copyDisplay(me->drivenIndex(at.display));
	if (!fb)
		return;
	UInt32 width = 0, height = 0, bytesPerRow = 0;
	UInt64 offset = 0;
	const bool moded = fb->currentScanout(&width, &height, &bytesPerRow, &offset);
	IOGBounds *bounds = nullptr;
	fb->getBoundingRect(&bounds);
	RgfxGlobalRect rect = {};
	if (bounds)
		rect = {bounds->minx, bounds->miny, bounds->maxx, bounds->maxy};
	fb->release();
	int32_t globalX = 0, globalY = 0;
	if (!moded || !bounds ||
	    !rgfxPointerGlobal(rect, width, height, at.x, at.y, &globalX, &globalY)) {
		static char unplaced;
		if (rgfxFirstCalls(&unplaced, at.display, 1))
			IOLog("ReimsGraphics: the pointer is over display %u, which has no mode or place on "
			      "the desktop yet; the cursor stays\n",
			      at.display);
		return;
	}
	IOHIDSystem *hid = IOHIDSystem::instance();
	if (!hid) {
		static char noHid;
		if (rgfxFirstCalls(&noHid, 0, 1))
			IOLog("ReimsGraphics: no IOHIDSystem to move the cursor with\n");
		return;
	}
	const RgfxPointerMove move = rgfxPointerMoveOf(at.buttons);
	evioLLEvent event;
	bzero(&event, sizeof(event));
	event.type = eventTypeOf(move);
	event.setCursor = kSetCursorPosition;
	event.location.x = rgfxPointerLocation(globalX);
	event.location.y = rgfxPointerLocation(globalY);
	const IOReturn posted =
	    rgfxHidPostEvent(hid, &event, nullptr, nullptr, nullptr, nullptr, nullptr);
	// The first two of each display's, and of its refused ones.
	static char moved;
	const UInt64 key = (static_cast<UInt64>(at.display) << 32) | (posted != kIOReturnSuccess);
	if (rgfxFirstCalls(&moved, key, 2))
		IOLog("ReimsGraphics: the host's pointer is over display %u at %d,%d: the cursor %s to "
		      "%d,%d of the desktop (0x%x)\n",
		      at.display, at.x, at.y,
		      move == RgfxPointerMove::moved ? "moves" : "is dragged", globalX, globalY, posted);
}

// Display 0's, from any display's setCursorState: each driven display's rectangle that differs
// from the one the host was last told, posted, and counted told once it is on the ring. A
// display whose framebuffer holds no rectangle yet is told once it has one.
void ReimsGraphics::tellBounds()
{
	if (!displayBoundsTaken || !interruptLock)
		return;
	ReimsGraphics *fbs[RGFX_DISPLAYS_MAX];
	const UInt32 n = copyDisplays(fbs);
	for (UInt32 i = 0; i < n; i++) {
		const UInt32 display = fbs[i]->displayId;
		IOGBounds *bounds = nullptr;
		fbs[i]->getBoundingRect(&bounds);
		if (!bounds)
			continue;
		const RgfxGlobalRect rect = {bounds->minx, bounds->miny, bounds->maxx, bounds->maxy};
		IOSimpleLockLock(interruptLock);
		const bool tell = rgfxBoundsToTell(boundsTold, display, rect);
		IOSimpleLockUnlock(interruptLock);
		if (!tell)
			continue;
		const RgfxDisplayBounds command = rgfxBoundsCommand(display, rect);
		if (!tryPost(RGFX_CMD_DISPLAY_BOUNDS, &command, sizeof(command), nullptr))
			continue;
		IOSimpleLockLock(interruptLock);
		rgfxBoundsWereTold(&boundsTold, display, rect);
		IOSimpleLockUnlock(interruptLock);
		// The first two of each display's.
		static char told;
		if (rgfxFirstCalls(&told, display, 2))
			IOLog("ReimsGraphics: display %u lies at %d,%d to %d,%d of the desktop\n", display,
			      rect.minx, rect.miny, rect.maxx, rect.maxy);
	}
	releaseDisplays(fbs, n);
}
