/*
 * RgfxIdSpace: the device context's object-id, resource-id and GPU-address
 * number lines (RgfxObjectModel.hpp).
 */

#include "RgfxObjectModel.hpp"

RgfxIdSpace::RgfxIdSpace(uint64_t address_base) : m_nextObjId(1), m_nextAddress(address_base)
{
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

uint64_t RgfxIdSpace::allocAddress(uint64_t size, uint64_t align)
{
	if (align < 1)
		align = 1;
	const uint64_t mask = align - 1;
	// A compare-and-swap loop so the aligned base and the bump stay consistent
	// across concurrent callers; each caller gets a disjoint range.
	uint64_t cur = m_nextAddress.load(std::memory_order_relaxed);
	for (;;) {
		uint64_t base = (cur + mask) & ~mask;
		uint64_t next = base + size;
		if (m_nextAddress.compare_exchange_weak(
		        cur, next, std::memory_order_relaxed, std::memory_order_relaxed))
			return base;
	}
}

void RgfxIdSpace::setBases(uint64_t obj_id_base, uint64_t address_base)
{
	// 0 names no object, so a context's first id is its base + 1.
	m_nextObjId.store(obj_id_base + 1, std::memory_order_relaxed);
	m_nextAddress.store(address_base, std::memory_order_relaxed);
}
