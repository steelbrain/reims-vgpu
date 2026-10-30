/*
 * RgfxIdSpace: the device context's object-id, resource-id and GPU-address
 * number lines (RgfxObjectModel.hpp).
 */

#include "RgfxObjectModel.hpp"

RgfxIdSpace::RgfxIdSpace(uint64_t address_base)
    : m_nextObjId(1), m_nextResourceId(1), m_nextAddress(address_base)
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
	return m_nextResourceId.fetch_add(1, std::memory_order_relaxed);
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

void RgfxIdSpace::setBases(uint64_t obj_id_base, uint64_t address_base, uint64_t resource_id_base)
{
	// 0 names no object, so a context's first id is its base + 1.
	m_nextObjId.store(obj_id_base + 1, std::memory_order_relaxed);
	m_nextAddress.store(address_base, std::memory_order_relaxed);
	m_nextResourceId.store(resource_id_base, std::memory_order_relaxed);
}
