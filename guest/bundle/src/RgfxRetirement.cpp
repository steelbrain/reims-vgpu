/*
 * RgfxRetirement: see RgfxRetirement.hpp.
 */

#include "RgfxRetirement.hpp"

#include <vector>

RgfxRetirement &RgfxRetirement::process()
{
	static RgfxRetirement *shared = new RgfxRetirement(); // lives as long as the process
	return *shared;
}

uint64_t RgfxRetirement::begin()
{
	std::lock_guard<std::mutex> hold(m_lock);
	uint64_t ticket = m_next++;
	m_live.insert(ticket);
	return ticket;
}

void RgfxRetirement::end(uint64_t ticket)
{
	std::unique_lock<std::mutex> held(m_lock);
	m_live.erase(ticket);
	// Another thread running releases takes the ones this made due too.
	if (!m_draining)
		drain(held);
}

void RgfxRetirement::retire(std::function<void()> release)
{
	std::unique_lock<std::mutex> held(m_lock);
	// Behind the releases already waiting, so they run in the order retired: with a ticket
	// live, or while a thread runs the ones due (its drain takes this one after them).
	m_waiting.emplace_back(m_next - 1, std::move(release));
	if (!m_draining)
		drain(held);
}

void RgfxRetirement::drain(std::unique_lock<std::mutex> &held)
{
	m_draining = true;
	for (;;) {
		uint64_t oldest = m_live.empty() ? UINT64_MAX : *m_live.begin();
		if (m_waiting.empty() || m_waiting.front().first >= oldest)
			break;
		std::vector<std::function<void()>> due;
		while (!m_waiting.empty() && m_waiting.front().first < oldest) {
			due.push_back(std::move(m_waiting.front().second));
			m_waiting.pop_front();
		}
		held.unlock();
		for (auto &release : due)
			release();
		due.clear(); // what the releases captured goes before the lock is taken again
		held.lock();
	}
	m_draining = false;
}

size_t RgfxRetirement::live()
{
	std::lock_guard<std::mutex> hold(m_lock);
	return m_live.size();
}

size_t RgfxRetirement::waiting()
{
	std::lock_guard<std::mutex> hold(m_lock);
	return m_waiting.size();
}
