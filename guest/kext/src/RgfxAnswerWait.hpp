// RgfxAnswerWait: how a thread of the kext waits for the host's answer to an ask of it
// (ReimsGraphics::postLifecycle and takeDeferred), and which of the kext's waiters an ask takes.
//
// An ask waits in one of three modes, its caller's `timeoutMs`:
//  - kRgfxWaitUntilAnswered: no deadline, and the sleep THREAD_INTERRUPTIBLE. The wait ends at the
//    answer, at the device's stop (quiesce wakes every waiter, and a waiter that finds the device
//    stopped ends unanswered), or at the thread's abort: a SIGKILL, a signal whose default action
//    ends the process, or the process's exit (thread_abort). A caught signal (thread_abort_safely)
//    breaks only a THREAD_ABORTSAFE sleep, so a SIGPROF or SIGCHLD never fails the ask. The ask is
//    the process's: a create, a query, a shared event's name or import, a queue's create, a
//    surface's pages or its cache's copy into them. The host answers each once the asking
//    context's lane takes it, behind what the context posted before it, a pipeline's compile
//    among them, which takes seconds cold and longer on a slow host: Metal's own calls wait for
//    the GPU's driver as long, with no deadline of their own. An aborted ask's thread is ending:
//    it waits interruptibly for nothing more in the call (the abort stays set until it returns
//    to the user), and a create it posted gets its destroy behind it (RgfxSubmitGate.hpp).
//  - kRgfxWaitNoDeadline: no deadline, THREAD_UNINT, ending at the answer or the device's stop:
//    a POOL_TAKE's, whose context's close waits for the takes in flight (RgfxPoolBlocks.hpp).
//  - any other value, a deadline in milliseconds, THREAD_UNINT: the ask ends at it unanswered,
//    or at the device's stop. Kept only for asks whose answers never wait behind a compile, each
//    caller saying why: the display's commands, a context's close, the device's kept pipeline
//    answers at the start, and an arm64 guest's video asks.
//
// The waiters are split by kind (rgfxWaiterKind), each kind taking only its own, so the asks
// that wait with no deadline never take every waiter: the creates and queries a process submits,
// and its shared events' names and imports, 32; a surface's pages and its cache's copy into them,
// any process's, WindowServer's for its own surfaces and at its CPU locks among them, 16; and
// the device's own asks of the host, a queue's create, a context's close, a pool's take, the kept
// pipeline answers and an arm64 guest's video asks, 16. A thread finding its kind's waiters
// taken waits for one in the ask's mode. The display's commands (ReimsGraphics::command) have 8
// waiters of their own.
//
// Header-only and free of the kernel's headers, so its native test (tests/answer_wait_test.cpp)
// builds it as the kext does.
//
// Our own code; it depends only on the committed wire header.
#ifndef RGFX_ANSWER_WAIT_HPP_
#define RGFX_ANSWER_WAIT_HPP_

#include <stdint.h>

#include "reims_graphics_wire.h"

// The modes with no deadline (above).
constexpr uint32_t kRgfxWaitNoDeadline = UINT32_MAX;
constexpr uint32_t kRgfxWaitUntilAnswered = UINT32_MAX - 1;

// Whether a wait of `timeoutMs` has a deadline.
constexpr bool rgfxWaitHasDeadline(uint32_t timeoutMs)
{
	return timeoutMs != kRgfxWaitNoDeadline && timeoutMs != kRgfxWaitUntilAnswered;
}

// Whether a wait of `timeoutMs` sleeps THREAD_INTERRUPTIBLE, ended by the thread's abort; the
// others sleep THREAD_UNINT.
constexpr bool rgfxWaitInterruptible(uint32_t timeoutMs)
{
	return timeoutMs == kRgfxWaitUntilAnswered;
}

// How one sleep of a wait ended.
enum RgfxSleepEnd : uint32_t {
	// Woken (an answer, a freed waiter, the device's stop), or a spurious wake: look again.
	kRgfxSleepWoken = 0,
	// Its deadline passed.
	kRgfxSleepTimedOut,
	// The thread was aborted (THREAD_INTERRUPTED).
	kRgfxSleepInterrupted,
};

// How a wait ended.
enum RgfxWaitEnd : uint32_t {
	kRgfxWaitAnswered = 0,
	kRgfxWaitTimedOut,
	kRgfxWaitAborted,
	// The device stopped first: no answer comes.
	kRgfxWaitStopped,
};

// Waits until `done()`: false while `up()` holds, it sleeps (`sleep()`, under the caller's lock,
// which `done` and `up` read under too) and looks again at each wake. A wait whose sleep timed out
// or was interrupted ends there, answered all the same where the answer came with it; one that
// finds the device stopped ends unanswered.
template <typename Done, typename Up, typename Sleep>
RgfxWaitEnd rgfxAwait(Done done, Up up, Sleep sleep)
{
	for (;;) {
		if (done())
			return kRgfxWaitAnswered;
		if (!up())
			return kRgfxWaitStopped;
		const RgfxSleepEnd slept = sleep();
		if (slept == kRgfxSleepWoken)
			continue;
		if (done())
			return kRgfxWaitAnswered;
		return slept == kRgfxSleepTimedOut ? kRgfxWaitTimedOut : kRgfxWaitAborted;
	}
}

// The kinds of the kext's lifecycle waiters (above), and how many each has.
enum RgfxWaiterKind : uint32_t {
	kRgfxWaitersCreate = 0,
	kRgfxWaitersSurface,
	kRgfxWaitersDevice,
	kRgfxWaiterKinds,
};
constexpr uint32_t kRgfxCreateWaiters = 32;
constexpr uint32_t kRgfxSurfaceWaiters = 16;
constexpr uint32_t kRgfxDeviceWaiters = 16;
constexpr uint32_t kRgfxLifecycleWaiters =
    kRgfxCreateWaiters + kRgfxSurfaceWaiters + kRgfxDeviceWaiters;

// The kind of waiter an ask of command `type` takes.
constexpr RgfxWaiterKind rgfxWaiterKind(uint32_t type)
{
	switch (type) {
	case RGFX_CMD_SURFACE_PAGES:
	case RGFX_CMD_SURFACE_DOWNLOAD:
		return kRgfxWaitersSurface;
	case RGFX_CMD_QUEUE_CREATE:
	case RGFX_CMD_CONTEXT_CLOSE:
	case RGFX_CMD_POOL_TAKE:
	case RGFX_CMD_PRIVATE_TAKE:
	case RGFX_CMD_PIPELINE_ANSWERS:
	case RGFX_CMD_VT_MESSAGE:
		return kRgfxWaitersDevice;
	default:
		return kRgfxWaitersCreate;
	}
}

// The first of kind `kind`'s waiters, and how many it has (`*count`), in the array of
// kRgfxLifecycleWaiters.
constexpr uint32_t rgfxWaitersFirst(RgfxWaiterKind kind)
{
	return kind == kRgfxWaitersCreate    ? 0
	       : kind == kRgfxWaitersSurface ? kRgfxCreateWaiters
	                                     : kRgfxCreateWaiters + kRgfxSurfaceWaiters;
}

constexpr uint32_t rgfxWaitersCount(RgfxWaiterKind kind)
{
	return kind == kRgfxWaitersCreate    ? kRgfxCreateWaiters
	       : kind == kRgfxWaitersSurface ? kRgfxSurfaceWaiters
	                                     : kRgfxDeviceWaiters;
}

// The index of a free waiter of kind `kind` (`isFree(i)`, `i` an index into the array), or -1
// where every one of its kind is taken: the other kinds' are never its.
template <typename IsFree>
int32_t rgfxFreeWaiter(RgfxWaiterKind kind, IsFree isFree)
{
	const uint32_t first = rgfxWaitersFirst(kind), end = first + rgfxWaitersCount(kind);
	for (uint32_t i = first; i < end; i++) {
		if (isFree(i))
			return static_cast<int32_t>(i);
	}
	return -1;
}

#endif // RGFX_ANSWER_WAIT_HPP_
