/*
 * ReimsGraphicsPointer: the framebuffer's move of the cursor onto the display the host's pointer is
 * over, on the device's POINTER_OVER, and each display's rectangle in the global space told the
 * host (design.md, "Multiple displays", "The pointer across displays"; RgfxPointerOver.hpp for the
 * event and the point, RgfxDisplayBounds.hpp for the rectangles told).
 *
 * macOS scales the VM's one USB tablet over the display its cursor is on, so the host's pointer
 * reaches display i once the cursor is there. The move is IOHIDSystem's own: a mouse-moved, or the
 * drag of the button held, posted with kIOHIDSetCursorPosition at the display's point of the
 * global space, as IOHIDPostEvent posts one for a process, which WindowServer follows. Measured
 * on x86-64 macOS 11.7.11, 15.8.2 and 26.7.2 with two displays (lane ptrcross, potluck jobs
 * m3-max-34583 to 34585): that post puts the cursor at its point of display 1, and the tablet's
 * next position lands on display 1; IOHIDSystem::extSetMouseLocation answers success and leaves
 * the cursor where it was; and a process's IOHIDSetMouseLocation and IOHIDPostEvent are refused
 * (kIOReturnNotPermitted), its CGEventPost dropped. Measured on arm64 macOS 14.6.1 and 27.0.1
 * too (lane a64ptr, potluck jobs m3-max-35287 and 35345): the post answers success in 15 to 23 us
 * and puts the cursor at its point, on display 1 at two displays in each of three arrangements, a
 * drag's as a move's. IOHIDSystem lives in IOHIDFamily on those releases, the library each
 * Info.plist names.
 *
 * The post runs on a thread call of display 0's framebuffer, off the device's work loop, as it
 * waits for IOHIDSystem's command gate. Each POINTER_OVER replaces the one before it that has not
 * moved the cursor yet: the latest is where the host's pointer is.
 *
 * Where the point lies, and the rectangles the host is told:
 *  - on x86-64, each framebuffer holds its display's rectangle as WindowServer placed it
 *    (getBoundingRect), and a move of the cursor on any display tells the host each display's
 *    that changed (tellBounds);
 *  - on arm64, WindowServer tells the displays nothing of where it placed them, so the kext
 *    learns it from its own posts: a crossing posts at the display's pixel of the rectangle held
 *    for it, and again beside the other displays where the cursor did not reach it, at most
 *    kRgfxCrossingPosts, each landing (the MOVE_CURSOR that shows the cursor on the display a
 *    post aimed at) checking that display's rectangle; a rectangle a landing confirmed is told
 *    the host, as x86-64's are (RgfxDisplayPlaces.hpp).
 */

#include "ReimsGraphics.hpp"
#include "RgfxTrace.hpp"

#include <IOKit/IOLib.h>
#include <IOKit/hidsystem/IOHIDSystem.h>
#include <IOKit/hidsystem/IOLLEvent.h>
#include <kern/clock.h>

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

// IOHIDSystem's move of the cursor to (x, y) of the global space, of the kind `move`;
// kIOReturnNoDevice where there is no IOHIDSystem.
IOReturn postMove(RgfxPointerMove move, int32_t x, int32_t y)
{
	IOHIDSystem *hid = IOHIDSystem::instance();
	if (!hid) {
		static char noHid;
		if (rgfxFirstCalls(&noHid, 0, 1))
			IOLog("ReimsGraphics: no IOHIDSystem to move the cursor with\n");
		return kIOReturnNoDevice;
	}
	evioLLEvent event;
	bzero(&event, sizeof(event));
	event.type = eventTypeOf(move);
	event.setCursor = kSetCursorPosition;
	event.location.x = rgfxPointerLocation(x);
	event.location.y = rgfxPointerLocation(y);
	return rgfxHidPostEvent(hid, &event, nullptr, nullptr, nullptr, nullptr, nullptr);
}

#if defined(__arm64e__)
uint64_t nowNs()
{
	uint64_t ns = 0;
	absolutetime_to_nanoseconds(mach_absolute_time(), &ns);
	return ns;
}

// `call` made at `ns` of nowNs's clock, or at once where that is past.
void callAt(thread_call_t call, uint64_t ns)
{
	uint64_t deadline = 0;
	nanoseconds_to_absolutetime(ns, &deadline);
	thread_call_enter_delayed(call, deadline);
}
#endif

} // namespace

bool ReimsGraphics::startPointer()
{
	pointerAt = {};
	pointerCall = thread_call_allocate(movePointer, this);
	if (!pointerCall) {
		IOLog("ReimsGraphics: no thread call for the pointer's moves\n");
		return false;
	}
#if defined(__arm64e__)
	places = {};
	crossing = {};
	pointerTold = 0;
	pointerToldAt = 0;
	pointerCrossed = 0;
	crossingsMissed = 0;
	landingCall = thread_call_allocate(landingWaited, this);
	if (!landingCall) {
		IOLog("ReimsGraphics: no thread call for the cursor's landings\n");
		return false;
	}
#endif
	return true;
}

// Once the work loop can ask for no move: neither call is made again, and each is waited for.
// On arm64 each can make the other (a move waits for its landing, a landing waited for moves
// again), so the move is waited for again after the landing's wait.
void ReimsGraphics::freePointer()
{
	thread_call_t pointer = __atomic_exchange_n(&pointerCall, nullptr, __ATOMIC_ACQ_REL);
#if defined(__arm64e__)
	thread_call_t landing = __atomic_exchange_n(&landingCall, nullptr, __ATOMIC_ACQ_REL);
	if (pointer)
		thread_call_cancel_wait(pointer);
	if (landing) {
		thread_call_cancel_wait(landing);
		thread_call_free(landing);
	}
#endif
	if (pointer) {
		thread_call_cancel_wait(pointer);
		thread_call_free(pointer);
	}
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
#if defined(__arm64e__)
	pointerTold++;
	pointerToldAt = nowNs();
#endif
	IOSimpleLockUnlock(interruptLock);
	thread_call_enter(pointerCall);
}

#if defined(__x86_64__)
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
	const UInt32 display = fb->displayId;
	fb->release();
	RgfxGlobalRect rects[RGFX_DISPLAYS_MAX] = {};
	bool placed[RGFX_DISPLAYS_MAX] = {};
	me->displayPlaces(rects, placed);
	int32_t globalX = 0, globalY = 0;
	if (!moded || display >= RGFX_DISPLAYS_MAX || !placed[display] ||
	    !rgfxPointerGlobal(rects[display], width, height, at.x, at.y, &globalX, &globalY)) {
		static char unplaced;
		if (rgfxFirstCalls(&unplaced, at.display, 1))
			IOLog("ReimsGraphics: the pointer is over display %u, which has no mode or place on "
			      "the desktop yet; the cursor stays\n",
			      at.display);
		return;
	}
	const RgfxPointerMove move = rgfxPointerMoveOf(at.buttons);
	const IOReturn posted = postMove(move, globalX, globalY);
	if (posted == kIOReturnNoDevice)
		return;
	// The first two of each display's, and of its refused ones.
	static char moved;
	const UInt64 key = (static_cast<UInt64>(at.display) << 32) | (posted != kIOReturnSuccess);
	if (rgfxFirstCalls(&moved, key, 2))
		IOLog("ReimsGraphics: the host's pointer is over display %u at %d,%d: the cursor %s to "
		      "%d,%d of the desktop (0x%x)\n",
		      at.display, at.x, at.y,
		      move == RgfxPointerMove::moved ? "moves" : "is dragged", globalX, globalY, posted);
}

// Each framebuffer holds its display's rectangle as WindowServer placed it (getBoundingRect).
void ReimsGraphics::displayPlaces(RgfxGlobalRect *rects, bool *placed)
{
	ReimsGraphics *fbs[RGFX_DISPLAYS_MAX];
	const UInt32 n = copyDisplays(fbs);
	for (UInt32 i = 0; i < n; i++) {
		const UInt32 display = fbs[i]->displayId;
		IOGBounds *bounds = nullptr;
		fbs[i]->getBoundingRect(&bounds);
		if (!bounds || display >= RGFX_DISPLAYS_MAX)
			continue;
		rects[display] = {bounds->minx, bounds->miny, bounds->maxx, bounds->maxy};
		placed[display] = true;
	}
	releaseDisplays(fbs, n);
}
#endif

#if defined(__arm64e__)
// A crossing's next post: a POINTER_OVER the thread call has not taken begins one onto its display
// at its pixel; the places are brought to the displays' modes; and the crossing's next point, if
// it has one, is posted, its landing waited for until its deadline (landingWaited).
void ReimsGraphics::movePointer(thread_call_param_t self, thread_call_param_t)
{
	ReimsGraphics *me = static_cast<ReimsGraphics *>(self);
	if (!me->hasStarted())
		return;
	RgfxPlaceMode modes[RGFX_DISPLAYS_MAX] = {};
	ReimsGraphics *fbs[RGFX_DISPLAYS_MAX];
	const UInt32 n = me->copyDisplays(fbs);
	for (UInt32 i = 0; i < n; i++) {
		UInt32 width = 0, height = 0, bytesPerRow = 0;
		UInt64 offset = 0;
		const UInt32 display = fbs[i]->displayId;
		if (display < RGFX_DISPLAYS_MAX &&
		    fbs[i]->currentScanout(&width, &height, &bytesPerRow, &offset))
			modes[display] = {width, height, fbs[i]->displayScale};
	}
	releaseDisplays(fbs, n);
	const uint64_t now = nowNs();
	int32_t gx = 0, gy = 0;
	IOSimpleLockLock(me->interruptLock);
	const RgfxPointerAt at = me->pointerAt;
	if (me->pointerCrossed != me->pointerTold) {
		me->pointerCrossed = me->pointerTold;
		rgfxCrossingStart(&me->crossing, at.display, at.x, at.y, me->pointerToldAt);
	}
	rgfxPlacesFit(&me->places, modes);
	const bool going = me->crossing.active;
	const bool posting = rgfxCrossingNext(me->places, &me->crossing, now, &gx, &gy);
	if (posting)
		me->crossing.postSeq = __atomic_load_n(&me->cursorMoves, __ATOMIC_ACQUIRE);
	const bool missed = going && !me->crossing.active;
	const UInt32 target = me->crossing.display;
	const UInt32 posts = me->crossing.posts;
	const RgfxPlaceState state =
	    target < RGFX_DISPLAYS_MAX ? me->places.at[target].state : RgfxPlaceState::none;
	const uint64_t deadline = rgfxCrossingDeadline(me->crossing);
	IOSimpleLockUnlock(me->interruptLock);
	if (missed)
		me->crossingMissed(target, posts);
	if (!posting)
		return;
	const RgfxPointerMove move = rgfxPointerMoveOf(at.buttons);
	const IOReturn posted = postMove(move, gx, gy);
	if (posted != kIOReturnSuccess) {
		// Nothing moved: the crossing is over.
		IOSimpleLockLock(me->interruptLock);
		me->crossing.waiting = false;
		me->crossing.active = false;
		IOSimpleLockUnlock(me->interruptLock);
	} else if (thread_call_t landing = __atomic_load_n(&me->landingCall, __ATOMIC_ACQUIRE)) {
		callAt(landing, deadline);
	}
	// The first four of each display's, and of its refused ones.
	static char moved;
	const UInt64 key = (static_cast<UInt64>(target) << 32) | (posted != kIOReturnSuccess);
	if (rgfxFirstCalls(&moved, key, 4))
		IOLog("ReimsGraphics: the host's pointer is over display %u at %d,%d: post %u of its "
		      "crossing, the cursor %s to %d,%d of the desktop, where display %u is %s (0x%x)\n",
		      target, at.x, at.y, posts,
		      move == RgfxPointerMove::moved ? "moves" : "is dragged", gx, gy, target,
		      state == RgfxPlaceState::confirmed ? "confirmed"
		      : state == RgfxPlaceState::learned ? "learned"
		                                         : "guessed",
		      posted);
}

// A post's landing not seen by its deadline: the crossing is over where its display last showed
// the cursor at the pixel the post aimed at; otherwise the cursor is not there, and the crossing,
// where one goes on, posts again, learning no place where no display showed the cursor where the
// post's point lies or is held at (RgfxDisplayPlaces.hpp). Called early, it waits on. Each
// display's last MOVE_CURSOR is read outside its gate: a swap racing it leaves it a frame old.
void ReimsGraphics::landingWaited(thread_call_param_t self, thread_call_param_t)
{
	ReimsGraphics *me = static_cast<ReimsGraphics *>(self);
	RgfxCursorsShown shown = {};
	ReimsGraphics *fbs[RGFX_DISPLAYS_MAX];
	const UInt32 n = me->copyDisplays(fbs);
	for (UInt32 i = 0; i < n; i++) {
		const RgfxMobileCursor &cursor = fbs[i]->mobileCursor;
		if (fbs[i]->displayId < RGFX_DISPLAYS_MAX)
			shown.at[fbs[i]->displayId] = {cursor.moved && cursor.visible, cursor.x, cursor.y,
			                               cursor.seq};
	}
	releaseDisplays(fbs, n);
	IOSimpleLockLock(me->interruptLock);
	const RgfxCrossingStep step = rgfxCrossingWaited(me->places, &me->crossing, nowNs(), shown);
	const bool waiting = me->crossing.waiting;
	const uint64_t deadline = rgfxCrossingDeadline(me->crossing);
	IOSimpleLockUnlock(me->interruptLock);
	if (step == RgfxCrossingStep::again) {
		if (thread_call_t call = __atomic_load_n(&me->pointerCall, __ATOMIC_ACQUIRE))
			thread_call_enter(call);
	} else if (waiting) {
		if (thread_call_t call = __atomic_load_n(&me->landingCall, __ATOMIC_ACQUIRE))
			callAt(call, deadline);
	}
}

// A crossing over with the cursor not on its display, which stays where it is: the first logged,
// each counted (RgfxCrossingsMissed).
void ReimsGraphics::crossingMissed(UInt32 display, UInt32 posts)
{
	const UInt32 missed = __atomic_add_fetch(&crossingsMissed, 1, __ATOMIC_RELAXED);
	setProperty("RgfxCrossingsMissed", static_cast<unsigned long long>(missed), 32);
	if (missed == 1)
		IOLog("ReimsGraphics: the cursor did not reach display %u in %u posts; it stays where it "
		      "is (the first of this boot; each is counted in RgfxCrossingsMissed)\n",
		      display, posts);
}

// Display `display`'s MOVE_CURSOR with the cursor shown at pixel `x`, `y`, from its swap: where a
// post is out, maybe its landing (rgfxCrossingLanded), which checks the place of the display it
// aimed at; the crossing posts again where the cursor is on another display than its own, or
// waits a little longer for its landing.
void ReimsGraphics::cursorLanded(UInt32 display, int32_t x, int32_t y)
{
	if (!interruptLock || !__atomic_load_n(&pointerCall, __ATOMIC_ACQUIRE))
		return;
	bool confirmed = false;
	IOSimpleLockLock(interruptLock);
	const int32_t gx = crossing.gx, gy = crossing.gy;
	const RgfxCrossingStep step =
	    rgfxCrossingLanded(&places, &crossing, display, x, y, nowNs(), &confirmed);
	const RgfxGlobalRect rect =
	    display < RGFX_DISPLAYS_MAX ? places.at[display].rect : RgfxGlobalRect{};
	const RgfxPlaceState state =
	    display < RGFX_DISPLAYS_MAX ? places.at[display].state : RgfxPlaceState::none;
	const uint64_t after = crossing.landedAfter;
	const uint64_t deadline = rgfxCrossingDeadline(crossing);
	IOSimpleLockUnlock(interruptLock);
	if (step == RgfxCrossingStep::again) {
		if (thread_call_t call = __atomic_load_n(&pointerCall, __ATOMIC_ACQUIRE))
			thread_call_enter(call);
	} else if (step == RgfxCrossingStep::held) {
		if (thread_call_t call = __atomic_load_n(&landingCall, __ATOMIC_ACQUIRE))
			callAt(call, deadline);
	}
	// The first four landings on each display.
	static char landed;
	if (step != RgfxCrossingStep::none && rgfxFirstCalls(&landed, display, 4))
		IOLog("ReimsGraphics: the post at %d,%d showed the cursor on display %u at %d,%d%s: "
		      "its place %d,%d to %d,%d, %s%s\n",
		      gx, gy, display, x, y,
		      step == RgfxCrossingStep::arrived ? ", its landing" : "", rect.minx, rect.miny,
		      rect.maxx, rect.maxy,
		      state == RgfxPlaceState::confirmed ? "confirmed"
		      : state == RgfxPlaceState::learned ? "learned"
		                                         : "guessed",
		      step == RgfxCrossingStep::again  ? "; posting again"
		      : step == RgfxCrossingStep::held ? "; waiting for its landing"
		                                       : "");
	static char took;
	if (step == RgfxCrossingStep::arrived && rgfxFirstCalls(&took, 0, 4))
		IOLog("ReimsGraphics: a crossing's landing came %llu us after its post\n", after / 1000);
}

// Each display's place a landing confirmed.
void ReimsGraphics::displayPlaces(RgfxGlobalRect *rects, bool *placed)
{
	IOSimpleLockLock(interruptLock);
	rgfxPlacesConfirmed(places, rects, placed);
	IOSimpleLockUnlock(interruptLock);
}
#endif

// Display 0's, from any display's move of the cursor: each driven display's place that changed.
void ReimsGraphics::tellBounds()
{
	if (!displayBoundsTaken || !interruptLock)
		return;
	RgfxGlobalRect rects[RGFX_DISPLAYS_MAX] = {};
	bool placed[RGFX_DISPLAYS_MAX] = {};
	displayPlaces(rects, placed);
	tellPlaces(rects, placed);
}

// Each driven display's rectangle that differs from the one the host was last told, posted, and
// counted told once it is on the ring. A display with no place yet is told once it has one.
void ReimsGraphics::tellPlaces(const RgfxGlobalRect *rects, const bool *placed)
{
	for (UInt32 display = 0; display < RGFX_DISPLAYS_MAX; display++) {
		if (!placed[display])
			continue;
		const RgfxGlobalRect &rect = rects[display];
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
}
