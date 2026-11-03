/*
 * RgfxRingRoomEvents: a queue's submission ring's room records (RGFX_FEATURE_SUBMIT_ROOM), for a
 * commit that waits for room on the ring (RgfxRingWriter::post).
 *
 * A post that finds the ring full stores 1 in the ring's producer_waiting, fences and looks once
 * more; still full, it waits here. After its next take of the ring, the host reads the flag, clears
 * it and posts an RGFX_EVT_SUBMIT_ROOM on the queue's completion ring, which the queue's reader
 * hands over (arrived). The writer and the reader's handler share one, which may outlive the
 * writer: a record can come as the queue goes.
 *
 * Header-only, so the queue tests build it as the bundle does. Our own code.
 */
#ifndef RGFX_RING_ROOM_EVENTS_HPP_
#define RGFX_RING_ROOM_EVENTS_HPP_

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <mutex>

class RgfxRingRoomEvents {
public:
	/// The host posted a room record: the posts waiting for room look again.
	void arrived()
	{
		{
			std::lock_guard<std::mutex> hold(m_lock);
			m_count++;
		}
		m_arrived.notify_all();
	}

	/// The records so far, read before a post looks at the ring.
	uint64_t seen()
	{
		std::lock_guard<std::mutex> hold(m_lock);
		return m_count;
	}

	/// Waits until a record after `seen` comes, or for `most` at most: whether one came.
	bool waitPast(uint64_t seen, std::chrono::microseconds most)
	{
		std::unique_lock<std::mutex> hold(m_lock);
		return m_arrived.wait_for(hold, most, [&] { return m_count != seen; });
	}

private:
	std::mutex m_lock;
	std::condition_variable m_arrived;
	uint64_t m_count = 0;
};

/// What a writer's sleeps for room came to, for its log line (RgfxRingWriter::post): the sleeps,
/// those a room record ended, and their time, in all and at most. The caller serialises.
struct RgfxRingRoomSleeps {
	uint64_t sleeps = 0;
	uint64_t woken = 0;
	uint64_t sleptNs = 0;
	uint64_t mostNs = 0;
	// When the last line was logged, in nanoseconds; 0 before the first.
	uint64_t loggedNs = 0;

	/// Counts one sleep of `ns`, `woken` where a room record ended it: whether the counts are
	/// logged now, at the first sleep and then at most every `intervalNs` (`nowNs` on the same
	/// clock as loggedNs).
	bool note(uint64_t ns, bool byRecord, uint64_t nowNs, uint64_t intervalNs)
	{
		sleeps++;
		woken += byRecord ? 1 : 0;
		sleptNs += ns;
		if (ns > mostNs)
			mostNs = ns;
		if (loggedNs != 0 && nowNs - loggedNs < intervalNs)
			return false;
		loggedNs = nowNs != 0 ? nowNs : 1;
		return true;
	}
};

#endif // RGFX_RING_ROOM_EVENTS_HPP_
