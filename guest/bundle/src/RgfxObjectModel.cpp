/*
 * RgfxIdSpace: the device context's object-id, resource-id and GPU-address
 * number lines (RgfxObjectModel.hpp).
 */

#include "RgfxObjectModel.hpp"

#include <algorithm>
#include <iterator>

#include <os/log.h>

namespace {
uint64_t alignUp(uint64_t value, uint64_t align)
{
	return (value + (align - 1)) & ~(align - 1);
}
} // namespace

RgfxIdSpace::RgfxIdSpace(uint64_t address_base, uint64_t address_span) : m_nextObjId(1)
{
	resetAddresses(address_base, address_span);
}

RgfxObjId RgfxIdSpace::allocObjId()
{
	// Relaxed is enough: the only invariant is uniqueness, and the id reaches
	// the host inside a ring message whose publish carries its own ordering.
	return m_nextObjId.fetch_add(1, std::memory_order_relaxed);
}

uint64_t RgfxIdSpace::allocResourceId()
{
	std::lock_guard<std::mutex> hold(m_idsLock);
	for (uint32_t k : m_chunks) {
		if (m_freeIds[k] != 0) {
			const unsigned bit = static_cast<unsigned>(__builtin_ctzll(m_freeIds[k]));
			m_freeIds[k] &= m_freeIds[k] - 1;
			return 1 + uint64_t{k} * RGFX_RESOURCE_ID_CHUNK + bit;
		}
	}
	// Every id the process holds is in use: one more chunk. The source is called
	// with the lock held, so two threads that run out at once take one chunk.
	uint64_t first = 0;
	if (m_chunkSource != nullptr) {
		if (!m_chunkSource(m_chunkSourceContext, &first))
			return 0;
	} else {
		if (m_nextLocalChunk == RGFX_RESOURCE_ID_CHUNKS)
			return 0;
		first = 1 + uint64_t{m_nextLocalChunk++} * RGFX_RESOURCE_ID_CHUNK;
	}
	// A chunk the pool does not hold, or one held already, is no chunk.
	if (first == 0 || (first - 1) % RGFX_RESOURCE_ID_CHUNK != 0 ||
	    (first - 1) / RGFX_RESOURCE_ID_CHUNK >= RGFX_RESOURCE_ID_CHUNKS)
		return 0;
	const uint32_t k = static_cast<uint32_t>((first - 1) / RGFX_RESOURCE_ID_CHUNK);
	if (m_heldChunks[k])
		return 0;
	m_heldChunks[k] = true;
	m_chunks.push_back(k);
	// Every id but the first, which is the caller's.
	m_freeIds[k] = ~uint64_t{0} << 1;
	return first;
}

void RgfxIdSpace::freeResourceId(uint64_t id)
{
	if (id == 0)
		return;
	const uint64_t k = (id - 1) / RGFX_RESOURCE_ID_CHUNK;
	std::lock_guard<std::mutex> hold(m_idsLock);
	if (k < RGFX_RESOURCE_ID_CHUNKS && m_heldChunks[k])
		m_freeIds[k] |= uint64_t{1} << ((id - 1) % RGFX_RESOURCE_ID_CHUNK);
}

void RgfxIdSpace::setResourceIdSource(ResourceIdChunkFn source, void *context)
{
	std::lock_guard<std::mutex> hold(m_idsLock);
	m_chunkSource = source;
	m_chunkSourceContext = context;
}

uint64_t RgfxIdSpace::addressSpan(uint64_t size)
{
	const uint64_t pages = (std::max<uint64_t>(size, 1) + kAddressPage - 1) / kAddressPage;
	return alignUp((pages + 1) * kAddressPage, kAddressGranule);
}

void RgfxIdSpace::resetAddresses(uint64_t base, uint64_t span)
{
	m_freeRanges.clear();
	m_livePicks.clear();
	const uint64_t start = alignUp(base, kAddressGranule);
	const uint64_t end = base + span;
	if (start < end)
		m_freeRanges[start] = end - start;
	m_addressCursor = start;
}

uint64_t RgfxIdSpace::allocAddress(uint64_t size, uint64_t align)
{
	align = std::max(align, kAddressGranule);
	const uint64_t span = addressSpan(size);
	std::lock_guard<std::mutex> hold(m_addressLock);
	// Where `span` bytes from `from` on fit the free range at `at` of `length` bytes.
	auto fit = [&](uint64_t at, uint64_t length, uint64_t from, uint64_t *start) {
		const uint64_t s = alignUp(std::max(at, from), align);
		if (s < at || s - at >= length || length - (s - at) < span)
			return false;
		*start = s;
		return true;
	};
	// Next fit: the free range the cursor lies in, from the cursor, and those after it; then
	// round from the range's first.
	auto it = m_freeRanges.upper_bound(m_addressCursor);
	if (it != m_freeRanges.begin()) {
		auto before = std::prev(it);
		if (before->first + before->second > m_addressCursor)
			it = before;
	}
	for (int pass = 0; pass < 2; pass++) {
		for (; it != m_freeRanges.end(); ++it) {
			uint64_t start = 0;
			if (!fit(it->first, it->second, pass == 0 ? m_addressCursor : 0, &start))
				continue;
			const uint64_t at = it->first, end = it->first + it->second;
			m_freeRanges.erase(it);
			if (start > at)
				m_freeRanges[at] = start - at;
			if (start + span < end)
				m_freeRanges[start + span] = end - (start + span);
			m_livePicks[start] = span;
			m_addressCursor = start + span;
			return start;
		}
		it = m_freeRanges.begin();
	}
	if (!m_addressesFull) {
		m_addressesFull = true;
		uint64_t live = 0;
		for (const auto &pick : m_livePicks)
			live += pick.second;
		os_log_error(OS_LOG_DEFAULT,
		             "RGFX: no GPU address left for %llu bytes: %llu MiB of the context's range "
		             "in %llu picks",
		             (unsigned long long)size, (unsigned long long)(live >> 20),
		             (unsigned long long)m_livePicks.size());
	}
	return 0;
}

void RgfxIdSpace::freeAddress(uint64_t address)
{
	std::lock_guard<std::mutex> hold(m_addressLock);
	auto live = m_livePicks.find(address);
	if (live == m_livePicks.end())
		return;
	uint64_t start = live->first, length = live->second;
	m_livePicks.erase(live);
	// Joined with the free ranges either side of it.
	auto after = m_freeRanges.lower_bound(start);
	if (after != m_freeRanges.begin()) {
		auto before = std::prev(after);
		if (before->first + before->second == start) {
			start = before->first;
			length += before->second;
			m_freeRanges.erase(before);
		}
	}
	if (after != m_freeRanges.end() && after->first == start + length) {
		length += after->second;
		m_freeRanges.erase(after);
	}
	m_freeRanges[start] = length;
}

void RgfxIdSpace::setBases(uint64_t obj_id_base, uint64_t address_base, uint64_t address_span)
{
	// 0 names no object, so a context's first id is its base + 1.
	m_nextObjId.store(obj_id_base + 1, std::memory_order_relaxed);
	std::lock_guard<std::mutex> hold(m_addressLock);
	resetAddresses(address_base, address_span);
}
