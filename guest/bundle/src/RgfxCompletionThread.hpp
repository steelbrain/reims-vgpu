/*
 * RgfxCompletionThread: the Metal process's completion thread, which runs every command
 * buffer's completion as the host posts it, whether or not anything waits for the buffer.
 *
 * The host posts each command buffer's RGFX_EVT_CB_COMPLETE on its queue's own completion
 * ring and raises the ring's vector when the guest armed the ring. The accelerator kext
 * watches the process's rings: on the vector's interrupt it wakes the process's waiter for
 * every ring that holds records the process has not taken (kRgfxAccelMethodWaitCompletions).
 * This thread is that waiter. Each pass it drains every queue's reader (RgfxReplyReader::
 * pump, which hands each buffer's records in ring order to its queue's completion queue,
 * where its handlers run), arms each ring and looks at it again, and only then sleeps in the
 * kext with the wake count it last saw, so a completion posted during the pass returns the
 * sleep at once and none is lost. It runs no handler, so a handler that waits holds back no
 * other queue's completions, and the host never waits on it for room on a ring.
 *
 * While the thread drives a reader, it is the reader's only consumer: a command buffer's
 * waitUntilCompleted blocks until the buffer's record has run on its queue's completion
 * queue, and pumps the ring itself only once the thread has ended (the process's context is
 * gone).
 *
 * Our own C++; the kext call is the caller's (WaitFn), so the thread is tested off the guest.
 */

#ifndef RGFX_COMPLETION_THREAD_HPP_
#define RGFX_COMPLETION_THREAD_HPP_

#include <cstdint>
#include <memory>

class RgfxReplyReader;

class RgfxCompletionThread {
public:
	/// What a wait answered: woken, with the kext's wake count; interrupted (a signal),
	/// to wait again; or the process's context is gone, and the thread ends.
	enum WaitResult { kWoke, kRetry, kStop };

	/// The thread's wait: given the wake count it last saw (0 at first), it blocks until the
	/// count differs and stores the new count in `*now` (kWoke). Called with `context`.
	typedef WaitResult (*WaitFn)(void *context, uint64_t last, uint64_t *now);

	/// A thread that waits with `wait`; it runs once start() is called.
	RgfxCompletionThread(WaitFn wait, void *context);

	/// Lets the readers go undriven, so their waiters pump them; the thread ends at its
	/// next wake.
	~RgfxCompletionThread();

	RgfxCompletionThread(const RgfxCompletionThread &) = delete;
	RgfxCompletionThread &operator=(const RgfxCompletionThread &) = delete;

	/// Starts the thread. False where it could not be made: its readers stay undriven.
	bool start();

	/// Drives `reader` from now on, while the thread runs and the reader lives: arms its
	/// ring, and marks it driven.
	void add(const std::shared_ptr<RgfxReplyReader> &reader);

	/// Whether the calling thread is a completion thread (a handler runs on it).
	static bool onCurrentThread();

	/// Whether the thread runs: started and not ended.
	bool running() const;

private:
	struct Shared;
	static void run(std::shared_ptr<Shared> shared);
	std::shared_ptr<Shared> m_shared;
};

#endif // RGFX_COMPLETION_THREAD_HPP_
