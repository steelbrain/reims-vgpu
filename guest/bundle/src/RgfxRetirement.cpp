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
	std::vector<std::function<void()>> due;
	{
		std::lock_guard<std::mutex> hold(m_lock);
		m_live.erase(ticket);
		uint64_t oldest = m_live.empty() ? UINT64_MAX : *m_live.begin();
		while (!m_waiting.empty() && m_waiting.front().first < oldest) {
			due.push_back(std::move(m_waiting.front().second));
			m_waiting.pop_front();
		}
	}
	for (auto &release : due)
		release();
}

void RgfxRetirement::retire(std::function<void()> release)
{
	{
		std::lock_guard<std::mutex> hold(m_lock);
		// Behind the releases already waiting, so they run in the order retired; with no
		// ticket live, none waits (end() ran them all).
		if (!m_live.empty()) {
			m_waiting.emplace_back(m_next - 1, std::move(release));
			return;
		}
	}
	release();
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
