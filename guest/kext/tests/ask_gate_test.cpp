// The native test of RgfxAskGate.hpp: a client's asks run beside each other, and its close waits
// for those in flight and lets none begin after it. Raced on threads as the kext's selectors and
// releaseContext use it: no ask's work lands after the close has let go of what the context
// holds, and the close never waits for an ask that began after it.
#include "RgfxAskGate.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <thread>
#include <vector>

namespace {
int checks = 0;

void check(bool ok, const char *what)
{
	checks++;
	if (!ok) {
		std::fprintf(stderr, "ask_gate_test: FAILED: %s\n", what);
		std::exit(1);
	}
}

void rules()
{
	RgfxAskGate g = {};
	check(!rgfxAskBegin(g, false) && g.inFlight == 0, "no ask begins for a context not open");
	check(rgfxAskBegin(g, true) && g.inFlight == 1, "an ask begins");
	check(rgfxAskBegin(g, true) && g.inFlight == 2 && g.most == 2,
	      "a second begins beside it while the first waits for the device");
	check(!rgfxAskEnd(g) && g.inFlight == 1, "one ends, no close waiting");
	check(rgfxAskClose(g) == 1 && g.closing && !rgfxAskDrained(g),
	      "the close begins with one ask in flight, which it waits for");
	check(!rgfxAskBegin(g, true) && g.inFlight == 1, "no ask begins once the close has");
	check(rgfxAskEnd(g) && rgfxAskDrained(g), "the last one ends, waking the close");
	check(!rgfxAskEnd(g) && g.inFlight == 0, "an end with none in flight counts nothing");

	RgfxAskGate idle = {};
	check(rgfxAskClose(idle) == 0 && rgfxAskDrained(idle),
	      "a close with no ask in flight goes on at once");
}

// The kext's use, raced: each ask makes a hold once its device wait ends, between its begin and
// its end, as acquireSurfaceBlock does; the close, once the asks drain, lets go of every hold the
// context has, as dropSurfaceBlocks does. A hold made after that let-go would outlive the context.
void race()
{
	constexpr int kRounds = 300;
	constexpr int kThreads = 4;
	int madeAfter = 0, overlapped = 0;
	for (int round = 0; round < kRounds; round++) {
		std::mutex listLock, holdLock;
		std::condition_variable drained;
		RgfxAskGate g = {};
		std::vector<int> holds;
		bool letGo = false;
		std::atomic<int> begun{0};
		auto asker = [&](int id) {
			for (;;) {
				{
					std::lock_guard<std::mutex> l(listLock);
					if (!rgfxAskBegin(g, true))
						return;
				}
				begun++;
				// The device's answer.
				std::this_thread::sleep_for(std::chrono::microseconds(20));
				{
					std::lock_guard<std::mutex> l(holdLock);
					madeAfter += letGo ? 1 : 0;
					holds.push_back(id);
				}
				std::lock_guard<std::mutex> l(listLock);
				if (rgfxAskEnd(g))
					drained.notify_all();
			}
		};
		std::vector<std::thread> threads;
		for (int i = 0; i < kThreads; i++)
			threads.emplace_back(asker, i);
		while (begun.load() < 1 + round % kThreads)
			std::this_thread::yield();
		// The close: no ask begins from here, those in flight end, then the let-go.
		{
			std::unique_lock<std::mutex> l(listLock);
			rgfxAskClose(g);
			drained.wait(l, [&] { return rgfxAskDrained(g); });
			overlapped += g.most > 1 ? 1 : 0;
		}
		{
			std::lock_guard<std::mutex> l(holdLock);
			letGo = true;
			holds.clear();
		}
		for (std::thread &t : threads)
			t.join();
		std::lock_guard<std::mutex> l(holdLock);
		madeAfter += static_cast<int>(holds.size());
	}
	check(madeAfter == 0, "no ask's hold is made after the close let go of the context's");
	check(overlapped > 0, "and the asks ran beside each other meanwhile");
}
} // namespace

int main()
{
	rules();
	race();
	std::printf("ask_gate_test: OK (%d checks)\n", checks);
	return 0;
}
