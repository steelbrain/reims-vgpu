// The native test of RgfxAnswerWait.hpp: a wait for the host's answer ends at the answer, at its
// deadline where it has one, at the thread's abort, and at the device's stop, whose wake ends it
// unanswered; and creates that wait with no deadline, every one of their waiters taken, leave the
// surface asks and the device's own theirs.
#include "RgfxAnswerWait.hpp"

#include <cstdio>
#include <cstdlib>

namespace {
int checks = 0;

void check(bool ok, const char *what)
{
	checks++;
	if (!ok) {
		std::fprintf(stderr, "answer_wait_test: FAILED: %s\n", what);
		std::exit(1);
	}
}

// A wait's world: what each sleep does, in turn, and the device's state. A sleep may land the
// answer, stop the device (quiesce's wake), time out or be interrupted.
struct World {
	enum Step { kWake, kAnswer, kStop, kTimeOut, kAbort, kAbortAnswered };
	const Step *steps = nullptr;
	int count = 0;
	int slept = 0;
	bool answered = false;
	bool up = true;

	RgfxWaitEnd wait()
	{
		return rgfxAwait([this] { return answered; }, [this] { return up; },
		                 [this] {
			                 const Step s = slept < count ? steps[slept] : kTimeOut;
			                 slept++;
			                 switch (s) {
			                 case kAnswer:
				                 answered = true;
				                 return kRgfxSleepWoken;
			                 case kStop:
				                 up = false;
				                 return kRgfxSleepWoken;
			                 case kTimeOut:
				                 return kRgfxSleepTimedOut;
			                 case kAbort:
				                 return kRgfxSleepInterrupted;
			                 case kAbortAnswered:
				                 answered = true;
				                 return kRgfxSleepInterrupted;
			                 default:
				                 return kRgfxSleepWoken;
			                 }
		                 });
	}
};
} // namespace

int main()
{
	// The modes.
	check(!rgfxWaitHasDeadline(kRgfxWaitUntilAnswered) && !rgfxWaitHasDeadline(kRgfxWaitNoDeadline),
	      "the modes with no deadline have none");
	check(rgfxWaitHasDeadline(5000) && rgfxWaitHasDeadline(10000) && rgfxWaitHasDeadline(0),
	      "any other value is a deadline");
	check(rgfxWaitInterruptible(kRgfxWaitUntilAnswered), "a wait until answered is interruptible");
	check(!rgfxWaitInterruptible(kRgfxWaitNoDeadline) && !rgfxWaitInterruptible(5000),
	      "a pool take's, and one with a deadline, are not");

	// The answer.
	{
		World w;
		w.answered = true;
		check(w.wait() == kRgfxWaitAnswered && w.slept == 0, "answered before it sleeps");
	}
	{
		// A compile that takes as long as it takes: woken by other answers, the wait looks
		// again each time and sleeps on, with no deadline of its own.
		World::Step steps[1000];
		for (World::Step &s : steps)
			s = World::kWake;
		steps[999] = World::kAnswer;
		World w;
		w.steps = steps;
		w.count = 1000;
		check(w.wait() == kRgfxWaitAnswered && w.slept == 1000,
		      "a wait until answered sleeps through every other wake until its answer");
	}

	// Its deadline, where it has one.
	{
		const World::Step steps[] = {World::kWake, World::kTimeOut};
		World w;
		w.steps = steps;
		w.count = 2;
		check(w.wait() == kRgfxWaitTimedOut && w.slept == 2, "a deadline ends it unanswered");
	}

	// The abort: the thread's process is ending (thread_abort).
	{
		const World::Step steps[] = {World::kWake, World::kAbort};
		World w;
		w.steps = steps;
		w.count = 2;
		check(w.wait() == kRgfxWaitAborted && w.slept == 2 && !w.answered,
		      "an abort ends it unanswered, and it sleeps no more");
	}
	{
		const World::Step steps[] = {World::kAbortAnswered};
		World w;
		w.steps = steps;
		w.count = 1;
		check(w.wait() == kRgfxWaitAnswered, "an answer that came with the abort is taken");
	}

	// A wake at stop: quiesce wakes every waiter, and one that finds the device stopped ends.
	{
		const World::Step steps[] = {World::kWake, World::kStop, World::kAnswer};
		World w;
		w.steps = steps;
		w.count = 3;
		check(w.wait() == kRgfxWaitStopped && w.slept == 2 && !w.answered,
		      "the stop's wake ends a wait until answered, which no answer would end");
	}
	{
		World w;
		w.up = false;
		check(w.wait() == kRgfxWaitStopped && w.slept == 0,
		      "a device stopped before the wait: it never sleeps");
	}
	{
		World w;
		w.up = false;
		w.answered = true;
		check(w.wait() == kRgfxWaitAnswered, "an answer that came before the stop is taken");
	}

	// The waiters' kinds.
	check(rgfxWaiterKind(RGFX_CMD_SURFACE_PAGES) == kRgfxWaitersSurface &&
	          rgfxWaiterKind(RGFX_CMD_SURFACE_DOWNLOAD) == kRgfxWaitersSurface,
	      "a surface's pages and its cache's copy take surface waiters");
	check(rgfxWaiterKind(RGFX_CMD_QUEUE_CREATE) == kRgfxWaitersDevice &&
	          rgfxWaiterKind(RGFX_CMD_CONTEXT_CLOSE) == kRgfxWaitersDevice &&
	          rgfxWaiterKind(RGFX_CMD_POOL_TAKE) == kRgfxWaitersDevice &&
	          rgfxWaiterKind(RGFX_CMD_PRIVATE_TAKE) == kRgfxWaitersDevice &&
	          rgfxWaiterKind(RGFX_CMD_PIPELINE_ANSWERS) == kRgfxWaitersDevice &&
	          rgfxWaiterKind(RGFX_CMD_VT_MESSAGE) == kRgfxWaitersDevice,
	      "a queue's create, a close, a pool's take, the kept answers and the video take the "
	      "device's");
	check(rgfxWaiterKind(RGFX_CMD_RENDER_PIPELINE_CREATE) == kRgfxWaitersCreate &&
	          rgfxWaiterKind(RGFX_CMD_LIBRARY_CREATE) == kRgfxWaitersCreate &&
	          rgfxWaiterKind(RGFX_CMD_LIBRARY_FROM_REGION) == kRgfxWaitersCreate &&
	          rgfxWaiterKind(RGFX_CMD_SHARED_EVENT_IMPORT) == kRgfxWaitersCreate,
	      "a process's creates and shared events take create waiters");
	check(kRgfxCreateWaiters == 32 && kRgfxSurfaceWaiters == 16 && kRgfxDeviceWaiters == 16 &&
	          kRgfxLifecycleWaiters == 64,
	      "32, 16 and 16 of 64");
	{
		bool seen[kRgfxLifecycleWaiters] = {};
		for (uint32_t k = 0; k < kRgfxWaiterKinds; k++) {
			const RgfxWaiterKind kind = static_cast<RgfxWaiterKind>(k);
			for (uint32_t i = 0; i < rgfxWaitersCount(kind); i++) {
				const uint32_t at = rgfxWaitersFirst(kind) + i;
				check(at < kRgfxLifecycleWaiters && !seen[at], "the kinds' waiters do not overlap");
				seen[at] = true;
			}
		}
		for (bool s : seen)
			check(s, "and every waiter is one kind's");
	}
	{
		// Every create waiter taken, each create waiting for a compile: a surface's pages, a
		// queue's create and a close still find theirs, and a 33rd create waits for one of its own.
		bool taken[kRgfxLifecycleWaiters] = {};
		auto isFree = [&taken](uint32_t i) { return !taken[i]; };
		for (uint32_t n = 0; n < kRgfxCreateWaiters; n++) {
			const int32_t at = rgfxFreeWaiter(kRgfxWaitersCreate, isFree);
			check(at >= 0, "each create takes a waiter");
			taken[at] = true;
		}
		check(rgfxFreeWaiter(kRgfxWaitersCreate, isFree) < 0,
		      "the 33rd create waits for one of the creates' waiters");
		const int32_t surface = rgfxFreeWaiter(rgfxWaiterKind(RGFX_CMD_SURFACE_PAGES), isFree);
		check(surface >= static_cast<int32_t>(kRgfxCreateWaiters),
		      "WindowServer's surface pages find a waiter behind the creates");
		taken[surface] = true;
		check(rgfxFreeWaiter(rgfxWaiterKind(RGFX_CMD_QUEUE_CREATE), isFree) >= 0 &&
		          rgfxFreeWaiter(rgfxWaiterKind(RGFX_CMD_CONTEXT_CLOSE), isFree) >= 0,
		      "and its queue's create and a close find theirs");
		// Every surface waiter taken too, each lock waiting for a copy behind the compile: the
		// device's own asks still find theirs.
		for (uint32_t n = 1; n < kRgfxSurfaceWaiters; n++)
			taken[rgfxFreeWaiter(kRgfxWaitersSurface, isFree)] = true;
		check(rgfxFreeWaiter(kRgfxWaitersSurface, isFree) < 0 &&
		          rgfxFreeWaiter(kRgfxWaitersDevice, isFree) >= 0,
		      "the device's asks are never the surfaces' either");
		taken[0] = false;
		check(rgfxFreeWaiter(kRgfxWaitersCreate, isFree) == 0, "a create's freed waiter is taken");
	}

	std::printf("answer_wait_test: OK (%d checks)\n", checks);
	return 0;
}
