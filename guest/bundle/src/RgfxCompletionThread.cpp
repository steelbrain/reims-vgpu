/*
 * RgfxCompletionThread: the process's completion thread (RgfxCompletionThread.hpp).
 */

#include "RgfxCompletionThread.hpp"

#include <atomic>
#include <mutex>
#include <thread>
#include <vector>

#include "RgfxReplyReader.hpp"

namespace {
thread_local bool t_onCompletionThread = false;
} // namespace

// What the thread shares with its owner: it outlives the owner while the thread sleeps in
// the kext.
struct RgfxCompletionThread::Shared {
	WaitFn wait;
	void *context;
	std::mutex lock; // the readers
	std::vector<std::weak_ptr<RgfxReplyReader>> readers;
	std::atomic<bool> running{false};
	std::atomic<bool> stopping{false};

	// The live readers, the gone ones dropped; marks each driven or not.
	std::vector<std::shared_ptr<RgfxReplyReader>> live(bool driven)
	{
		std::vector<std::shared_ptr<RgfxReplyReader>> out;
		std::lock_guard<std::mutex> hold(lock);
		size_t kept = 0;
		for (auto &weak : readers) {
			if (std::shared_ptr<RgfxReplyReader> reader = weak.lock()) {
				reader->setDriven(driven);
				out.push_back(std::move(reader));
				readers[kept++] = weak;
			}
		}
		readers.resize(kept);
		return out;
	}
};

RgfxCompletionThread::RgfxCompletionThread(WaitFn wait, void *context)
    : m_shared(std::make_shared<Shared>())
{
	m_shared->wait = wait;
	m_shared->context = context;
}

RgfxCompletionThread::~RgfxCompletionThread()
{
	m_shared->stopping.store(true, std::memory_order_release);
	m_shared->live(false);
}

bool RgfxCompletionThread::start()
{
	if (m_shared->wait == nullptr || m_shared->running.load(std::memory_order_acquire))
		return false;
	m_shared->running.store(true, std::memory_order_release);
	m_shared->live(true);
	try {
		std::thread(&RgfxCompletionThread::run, m_shared).detach();
	} catch (...) {
		m_shared->running.store(false, std::memory_order_release);
		m_shared->live(false);
		return false;
	}
	return true;
}

void RgfxCompletionThread::add(const std::shared_ptr<RgfxReplyReader> &reader)
{
	if (reader == nullptr)
		return;
	{
		std::lock_guard<std::mutex> hold(m_shared->lock);
		m_shared->readers.push_back(reader);
	}
	// A new ring holds nothing yet: armed, the host's first completion on it raises the
	// vector, and the kext wakes this thread for it.
	reader->armAndCheck();
	reader->setDriven(m_shared->running.load(std::memory_order_acquire) &&
	                  !m_shared->stopping.load(std::memory_order_acquire));
}

bool RgfxCompletionThread::onCurrentThread()
{
	return t_onCompletionThread;
}

bool RgfxCompletionThread::running() const
{
	return m_shared->running.load(std::memory_order_acquire);
}

void RgfxCompletionThread::run(std::shared_ptr<Shared> shared)
{
	t_onCompletionThread = true;
	uint64_t seen = 0;
	for (;;) {
		// Every ring drained, then armed and looked at again: a record published before the
		// arm is taken here, and one after it raises the vector.
		for (const std::shared_ptr<RgfxReplyReader> &reader : shared->live(true)) {
			do {
				reader->pump();
			} while (reader->armAndCheck());
		}
		if (shared->stopping.load(std::memory_order_acquire))
			break;
		uint64_t now = seen;
		WaitResult r = shared->wait(shared->context, seen, &now);
		if (r == kStop || shared->stopping.load(std::memory_order_acquire))
			break;
		if (r == kWoke)
			seen = now;
	}
	shared->running.store(false, std::memory_order_release);
	// Undriven from now on: a waiter pumps its ring itself.
	shared->live(false);
}
