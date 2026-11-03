// The native test of RgfxQueueLimits.hpp: a process makes as many queues as the device watches,
// and WindowServer can make one however many the others hold.
#include "RgfxQueueLimits.hpp"

#include <cstdio>
#include <cstdlib>

namespace {
int checks = 0;

void check(bool ok, const char *what)
{
	checks++;
	if (!ok) {
		std::fprintf(stderr, "queue_limits_test: FAILED: %s\n", what);
		std::exit(1);
	}
}
} // namespace

int main()
{
	// One process: its 2048th queue is made, its 2049th refused.
	check(rgfxQueueAdmitted(0), "a process's first queue");
	check(rgfxQueueAdmitted(256), "past the old limit of 256");
	check(rgfxQueueAdmitted(2047), "its 2048th queue");
	check(!rgfxQueueAdmitted(2048), "not its 2049th");

	// The device watches one process at its limit beside others holding as many again, and
	// WindowServer beside them.
	check(rgfxWatchAdmitted(2047, false), "the device watches a process's 2048th queue");
	check(rgfxWatchAdmitted(2047 + 2048, false),
	      "and its 2048th beside 2048 of other processes'");
	check(!rgfxWatchAdmitted(4096, false), "but no other process's queue past 4096 in all");
	check(rgfxWatchAdmitted(4096, true), "WindowServer's queue past them");
	check(rgfxWatchAdmitted(4096 + 63, true), "to the reserve's last");
	check(!rgfxWatchAdmitted(4096 + 64, true), "and not past it");
	check(kRgfxMaxWatchedQueues == 2 * kRgfxMaxQueuesPerClient + kRgfxWindowServerQueues,
	      "the watch table holds a process at its limit, as many others and WindowServer's "
	      "reserve");

	// A process reaches its own limit before the device's, alone or beside others' 2048.
	const uint32_t othersHeld[] = {0, 5, 2048};
	for (uint32_t others : othersHeld) {
		uint32_t made = 0;
		while (rgfxQueueAdmitted(made) && rgfxWatchAdmitted(others + made, false))
			made++;
		check(made == 2048, "a process makes 2048 queues beside the others'");
	}

	std::printf("queue_limits_test: OK (%d checks)\n", checks);
	return 0;
}
