/*
 * RgfxRetirement: holds back an object's release on the host until every command buffer the
 * process had in flight when the guest let the object go has completed.
 *
 * A texture's OBJECT_DESTROY rides ring 0 through the kext, while a command buffer rides its
 * queue's own ring, and the host decodes a buffer once ring 0's messages published before it took
 * the buffer are carried out. A destroy posted while a buffer that names the texture still waits
 * on its ring would reach the host first, and the buffer would name an object the host has
 * dropped. Metal keeps an in-flight buffer's resources alive (retained references), so the
 * release waits for the buffers: each committed buffer that will complete holds a ticket from
 * begin() to end(), and retire() runs its release once every ticket issued before it has ended,
 * at once where none is live. Releases run in the order retired, on the thread that ends the last
 * ticket they wait for (the completion thread) or the caller's.
 *
 * Our own C++; tested off the guest.
 */

#ifndef RGFX_RETIREMENT_HPP_
#define RGFX_RETIREMENT_HPP_

#include <cstdint>
#include <deque>
#include <functional>
#include <mutex>
#include <set>
#include <utility>

class RgfxRetirement {
public:
	/// The process's: every queue's command buffers and every object's release.
	static RgfxRetirement &process();

	/// A command buffer that will complete is committed: its ticket, for end().
	uint64_t begin();

	/// The command buffer holding `ticket` completed (or never will): the releases waiting
	/// only for it and earlier tickets run.
	void end(uint64_t ticket);

	/// Runs `release` once every ticket begun before now has ended: at once, on this thread,
	/// where none is live.
	void retire(std::function<void()> release);

	/// How many releases wait, and how many tickets are live (tests).
	size_t waiting();
	size_t live();

private:
	std::mutex m_lock;
	uint64_t m_next = 1;
	std::set<uint64_t> m_live;
	// In retire() order, so in the order of the last ticket each waits for.
	std::deque<std::pair<uint64_t, std::function<void()>>> m_waiting;
};

#endif // RGFX_RETIREMENT_HPP_
