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

#endif // RGFX_RING_ROOM_EVENTS_HPP_
