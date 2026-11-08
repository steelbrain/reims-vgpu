/*
 * RgfxRegion: the BAR-region bump allocator (RgfxRegion.hpp). The policy and the
 * reserved head are documented in the header.
 */

#include "RgfxRegion.hpp"

#include <cstring>

RgfxRegion::RgfxRegion(volatile uint8_t *base, uint64_t size)
    : m_base(base), m_size(size), m_floor(RGFX_ALLOCATIONS_OFFSET), m_cursor(RGFX_ALLOCATIONS_OFFSET)
{
}

uint64_t RgfxRegion::alignUp(uint64_t value, uint64_t align)
{
	return (value + (align - 1)) & ~(align - 1);
}

uint64_t RgfxRegion::used() const
{
	// m_cursor only grows and never drops below the floor, so the subtraction is safe
	// without the lock (a racing allocate may make this a slight under-count; it is a
	// diagnostic).
	return m_cursor - m_floor;
}

bool RgfxRegion::freeBytes(uint64_t *bytes) const
{
	if (m_alloc != nullptr) {
		return false;
	}
	*bytes = m_size > m_cursor ? m_size - m_cursor : 0;
	return true;
}

void RgfxRegion::setBlockSource(AllocFn alloc, ReleaseFn release, void *context)
{
	std::lock_guard<std::mutex> guard(m_lock);
	m_alloc = alloc;
	m_release = release;
	m_sourceContext = context;
}

void RgfxRegion::setWriteCombinedSource(WriteCombinedAllocFn alloc)
{
	std::lock_guard<std::mutex> guard(m_lock);
	m_writeCombinedAlloc = alloc;
}

void RgfxRegion::setReclaim(ReclaimFn reclaim, void *context)
{
	std::lock_guard<std::mutex> guard(m_lock);
	m_reclaim = reclaim;
	m_reclaimContext = context;
}

bool RgfxRegion::reclaimed()
{
	return m_reclaim != nullptr && m_reclaim(m_reclaimContext);
}

void RgfxRegion::setSurfaceSource(SurfaceAcquireFn acquire, ReleaseFn release, void *context)
{
	std::lock_guard<std::mutex> guard(m_lock);
	m_surfaceAcquire = acquire;
	m_surfaceRelease = release;
	m_surfaceContext = context;
}

RgfxRegion::SurfaceBlock RgfxRegion::acquireSurface(const void *plane, uint64_t length,
                                                    uint64_t size)
{
	SurfaceBlock out = {};
	uint64_t offset = 0, serial = 0;
	bool fresh = false;
	if (m_surfaceAcquire == nullptr ||
	    !m_surfaceAcquire(m_surfaceContext, plane, length, size, &offset, &serial, &fresh)) {
		return out; // block.ok == false
	}
	if (offset < m_floor || offset > m_size || size > m_size - offset) {
		releaseSurface(offset);
		return out;
	}
	out.block = {offset, m_base + offset, size, true};
	out.serial = serial;
	out.fresh = fresh;
	return out;
}

void RgfxRegion::setSurfacePagesSource(SurfacePagesFn acquire, uint64_t alignment, void *context)
{
	std::lock_guard<std::mutex> guard(m_lock);
	m_pagesAcquire = acquire;
	m_pagesAlignment = alignment;
	m_pagesContext = context;
}

RgfxRegion::SurfaceBlock RgfxRegion::acquireSurfacePages(const void *base, uint64_t length,
                                                         uint64_t size, uint32_t surfaceId,
                                                         bool cacheable,
                                                         const RgfxSurfacePagesLayout *layout,
                                                         PagesAnswer *answer)
{
	SurfaceBlock out = {};
	uint64_t offset = 0, serial = 0;
	bool fresh = false, cached = false;
	auto ask = [&] {
		return m_pagesAcquire(m_pagesContext, base, length, size, surfaceId, cacheable, layout,
		                      &offset, &serial, &fresh, &cached);
	};
	*answer = m_pagesAcquire == nullptr ? kPagesUnsupported : ask();
	if (*answer == kPagesNoRoom && reclaimed()) {
		*answer = ask();
	}
	if (*answer != kPagesGiven) {
		return out; // block.ok == false
	}
	if (offset < m_floor || offset > m_size || size > m_size - offset) {
		releaseSurface(offset);
		*answer = kPagesRefused;
		return out;
	}
	out.block = {offset, m_base + offset, size, true};
	out.serial = serial;
	out.fresh = fresh;
	out.cached = cached;
	return out;
}

void RgfxRegion::setSurfaceWriteSource(SurfaceWritesFn note, SurfaceRetireFn retire,
                                       void *context)
{
	std::lock_guard<std::mutex> guard(m_lock);
	m_writesNote = note;
	m_writesRetire = retire;
	m_writesContext = context;
}

void RgfxRegion::noteSurfaceWrites(uint64_t queue, uint64_t seqno, const uint64_t *writes,
                                   uint32_t count)
{
	if (m_writesNote != nullptr && count != 0) {
		m_writesNote(m_writesContext, queue, seqno, writes, count);
	}
}

void RgfxRegion::retireSurfaceWrites(uint64_t queue, uint64_t seqno)
{
	if (m_writesRetire != nullptr) {
		m_writesRetire(m_writesContext, queue, seqno);
	}
}

void RgfxRegion::setPagesBlockSource(PagesBlockFn acquire, void *context)
{
	std::lock_guard<std::mutex> guard(m_lock);
	m_pagesBlock = acquire;
	m_pagesBlockContext = context;
}

RgfxRegion::Block RgfxRegion::acquirePages(const void *address, uint64_t length, bool writable)
{
	Block out = {};
	uint64_t offset = 0, size = 0;
	if (m_pagesBlock == nullptr ||
	    !m_pagesBlock(m_pagesBlockContext, address, length, writable, &offset, &size))
		return out; // ok == false
	if (offset < m_floor || offset > m_size || size > m_size - offset) {
		release(offset);
		return out;
	}
	out.offset = offset;
	out.size = size;
	out.ok = true;
	return out;
}

void RgfxRegion::releaseSurface(uint64_t offset)
{
	if (m_surfaceRelease != nullptr) {
		m_surfaceRelease(m_surfaceContext, offset);
	}
}

void RgfxRegion::release(uint64_t offset)
{
	if (m_release != nullptr) {
		m_release(m_sourceContext, offset);
	}
}

RgfxRegion::Block RgfxRegion::allocate(uint64_t size, uint64_t align)
{
	return allocate(size, align, UINT64_MAX);
}

RgfxRegion::Block RgfxRegion::allocate(uint64_t size, uint64_t align, uint64_t writeCombinedFrom)
{
	Block out = {};
	if (!isPowerOfTwo(align)) {
		return out; // ok == false
	}

	if (m_alloc != nullptr) {
		uint64_t offset = 0;
		const bool combined = m_writeCombinedAlloc != nullptr && writeCombinedFrom < size;
		auto ask = [&] {
			return combined ? m_writeCombinedAlloc(m_sourceContext, size, align,
			                                       writeCombinedFrom, &offset)
			                : m_alloc(m_sourceContext, size, align, &offset);
		};
		const bool placed = ask() || (reclaimed() && ask());
		if (!placed || offset < m_floor || offset > m_size || size > m_size - offset) {
			return out; // no room; ok == false
		}
		out.offset = offset;
		out.ptr = m_base + offset;
		out.size = size;
		out.ok = true;
		return out;
	}

	std::lock_guard<std::mutex> guard(m_lock);

	uint64_t offset = alignUp(m_cursor, align);
	// Overflow-safe room check: the end must not pass the region's size.
	if (offset < m_cursor || offset > m_size || size > m_size - offset) {
		return out; // no room; ok == false
	}
	m_cursor = offset + size;

	out.offset = offset;
	out.ptr = m_base + offset;
	out.size = size;
	out.ok = true;
	return out;
}

RgfxRegion::RingPair RgfxRegion::allocateRingPair(uint32_t submit_data_size,
                                                  uint32_t complete_data_size)
{
	RingPair out = {};
	out.submit_data_size = submit_data_size;
	out.complete_data_size = complete_data_size;

	// Each ring's byte ring is a power of two (the host reads it with a mask); the
	// completion ring is optional (0 == none in the single-ring model).
	if (!isPowerOfTwo(submit_data_size) ||
	    (complete_data_size != 0 && !isPowerOfTwo(complete_data_size))) {
		return out; // ok == false
	}

	// Where the kext maps part of a block write-combined, the submission ring's data takes it,
	// from its block's second page, and the ring's header, which the host writes too and the
	// writer reads, ends the first page, write-back (WRITE-COMBINED PARTS).
	const bool combined = writeCombines();
	const uint64_t head = combined ? kDefaultBlockAlign - RGFX_RING_HEADER_SIZE : 0;
	Block submit = combined
	                   ? allocate(kDefaultBlockAlign + submit_data_size, kDefaultBlockAlign,
	                              kDefaultBlockAlign)
	                   : allocate(RGFX_RING_HEADER_SIZE + submit_data_size, kRingAlign);
	if (!submit.ok) {
		return out;
	}
	Block complete = {};
	if (complete_data_size != 0) {
		complete = allocate(RGFX_RING_HEADER_SIZE + complete_data_size, kRingAlign);
		if (!complete.ok) {
			release(submit.offset);
			return out;
		}
	}

	// Zero each ring's header + byte ring so a fresh RgfxRingWriter (producer 0,
	// consumer 0, consumer_idle 0, armed 0) starts clean over BAR memory whose
	// contents are otherwise undefined.
	std::memset(const_cast<uint8_t *>(submit.ptr + head), 0, submit.size - head);
	if (complete_data_size != 0) {
		std::memset(const_cast<uint8_t *>(complete.ptr), 0, complete.size);
	}

	out.submit_offset = submit.offset + head;
	out.complete_offset = complete_data_size != 0 ? complete.offset : 0;
	out.submit_block = submit.offset;
	out.complete_block = out.complete_offset;
	out.ok = true;
	return out;
}

bool RgfxRegion::contains(const void *p, uint64_t len) const
{
	if (p == nullptr) {
		return false;
	}
	const uint8_t *q = static_cast<const uint8_t *>(p);
	const uint8_t *base = const_cast<const uint8_t *>(m_base);
	if (q < base) {
		return false;
	}
	uint64_t offset = static_cast<uint64_t>(q - base);
	// Inside the guest-allocable region, and the whole span fits before its end.
	return offset >= m_floor && offset <= m_size && len <= m_size - offset;
}

uint64_t RgfxRegion::offsetOf(const void *p) const
{
	if (!contains(p, 0)) {
		return 0;
	}
	return static_cast<uint64_t>(static_cast<const uint8_t *>(p) -
	                             const_cast<const uint8_t *>(m_base));
}
