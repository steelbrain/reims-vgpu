// RgfxQueueLimits: how many command queues the kext takes, of one process and of every process.
//
// A process may make as many queues as the device watches: Metal on a Mac has no limit at this
// scale (an M3 Max made 1536 idle queues in one process, measured). The device's tables are sized
// for one process at that limit beside other processes holding as many again, and WindowServer:
// the last kRgfxWindowServerQueues of the watched rings go to WindowServer's queues alone, so no
// other process, however many queues it makes, leaves the compositor unable to make one (as
// RgfxBlockTable's reserve does for blocks). Sized for one process alone, a process's queues past
// what the others held were refused: with 5 queues of other processes watched, a process's 2044th
// was (measured).
//
// Header-only and free of the kernel's headers, so its native test (tests/queue_limits_test.cpp)
// builds it as the kext does.
//
// Our own code.
#ifndef RGFX_QUEUE_LIMITS_HPP_
#define RGFX_QUEUE_LIMITS_HPP_

#include <stdint.h>

// The most queues one process (one user client) holds at once.
static constexpr uint32_t kRgfxMaxQueuesPerClient = 2048;
// The watched rings only WindowServer's queues take.
static constexpr uint32_t kRgfxWindowServerQueues = 64;
// The most queues watched at once, over every process: one process at its limit, the others'
// as many again, and WindowServer's reserve.
static constexpr uint32_t kRgfxMaxWatchedQueues =
    2 * kRgfxMaxQueuesPerClient + kRgfxWindowServerQueues;

// Whether a process that holds `held` queues may make one more.
inline bool rgfxQueueAdmitted(uint32_t held)
{
	return held < kRgfxMaxQueuesPerClient;
}

// Whether one more queue may be watched where `watched` are, for WindowServer's
// (`windowServer`) or another process's.
inline bool rgfxWatchAdmitted(uint32_t watched, bool windowServer)
{
	return watched < (windowServer ? kRgfxMaxWatchedQueues
	                               : kRgfxMaxWatchedQueues - kRgfxWindowServerQueues);
}

#endif // RGFX_QUEUE_LIMITS_HPP_
