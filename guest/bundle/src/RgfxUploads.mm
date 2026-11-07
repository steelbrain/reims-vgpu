/*
 * RgfxUploads: see RgfxUploads.h. Our own ObjC++; no Apple private SPI.
 */

#import "RgfxUploads.h"

#import "RgfxBuffer.h"

#include <cstring>

#include "RgfxRegion.hpp"
#include "RgfxSerializer.hpp"

namespace {

uint64_t alignUp(uint64_t v, uint64_t a)
{
	return (v + a - 1) & ~(a - 1);
}

} // namespace

RgfxUploadPool::RgfxUploadPool(RgfxRegion *region, RgfxSerializer *serializer,
                               id<MTLDevice> device, RgfxSlabs *slabs)
    : m_region(region), m_serializer(serializer), m_slabs(slabs), m_device(device)
{
}

bool RgfxUploadPool::take(uint64_t size, RgfxUploadChunk *chunk)
{
	{
		std::lock_guard<std::mutex> hold(m_lock);
		for (size_t i = 0; i < m_free.size(); i++) {
			if (m_free[i].size >= size) {
				*chunk = m_free[i];
				m_free.erase(m_free.begin() + static_cast<std::ptrdiff_t>(i));
				return true;
			}
		}
	}
	if (m_serializer == nullptr) {
		return false;
	}
	uint64_t bytes = alignUp(size > RgfxUploadArena::kUploadChunkSize
	                             ? size
	                             : RgfxUploadArena::kUploadChunkSize,
	                         RgfxRegion::kDefaultBlockAlign);
	// A Shared buffer of the slabs, in the process's own pages where the device maps them.
	if (m_slabs != nullptr) {
		RgfxBuffer *buffer = [RgfxBuffer bufferForDeviceWithLength:static_cast<NSUInteger>(bytes)
		                                                   options:MTLResourceStorageModeShared
		                                                     bytes:nullptr
		                                                 maxLength:0
		                                                     slabs:m_slabs
		                                                    device:m_device
		                                                serializer:m_serializer];
		if (buffer != nil) {
			{
				std::lock_guard<std::mutex> hold(m_lock);
				m_made++;
			}
			chunk->buffer = buffer;
			chunk->bytes = static_cast<uint8_t *>([buffer contents]);
			chunk->size = bytes;
			return true;
		}
	}
	if (m_region == nullptr) {
		return false;
	}
	RgfxRegion::Block block = m_region->allocate(bytes);
	if (!block.ok) {
		return false;
	}
	uint8_t *ptr = const_cast<uint8_t *>(block.ptr);
	// A Shared buffer over the block: the host reads the bytes in place.
	RgfxBuffer *buffer =
	    [RgfxBuffer bufferOverHostMemoryWithContents:ptr
	                                    regionOffset:block.offset
	                                          length:static_cast<NSUInteger>(bytes)
	                                         options:MTLResourceStorageModeShared
	                                     deallocator:nil
	                                          device:m_device
	                                      serializer:m_serializer];
	if (buffer == nil) {
		m_region->release(block.offset);
		return false;
	}
	{
		std::lock_guard<std::mutex> hold(m_lock);
		m_made++;
	}
	chunk->buffer = buffer;
	chunk->bytes = ptr;
	chunk->size = bytes;
	return true;
}

void RgfxUploadPool::giveBack(std::vector<RgfxUploadChunk> &chunks)
{
	std::lock_guard<std::mutex> hold(m_lock);
	for (RgfxUploadChunk &c : chunks) {
		m_free.push_back(c);
	}
	chunks.clear();
}

uint64_t RgfxUploadPool::made() const
{
	std::lock_guard<std::mutex> hold(m_lock);
	return m_made;
}

RgfxBodyPool::RgfxBodyPool(RgfxRegion *region) : m_region(region)
{
}

RgfxBodyPool::~RgfxBodyPool()
{
	if (m_region != nullptr) {
		for (const RgfxBodyBlock &b : m_free) {
			m_region->release(b.offset);
		}
	}
}

bool RgfxBodyPool::take(uint64_t size, RgfxBodyBlock *block)
{
	uint64_t bytes = kGranule;
	while (bytes < size && bytes <= (UINT64_MAX >> 1)) {
		bytes <<= 1;
	}
	if (bytes < size) {
		return false;
	}
	{
		std::lock_guard<std::mutex> hold(m_lock);
		size_t best = m_free.size();
		for (size_t i = 0; i < m_free.size(); i++) {
			const bool fits = m_free[i].size >= bytes;
			if (fits && (best == m_free.size() || m_free[i].size < m_free[best].size))
				best = i;
		}
		if (best != m_free.size()) {
			*block = m_free[best];
			m_free.erase(m_free.begin() + static_cast<std::ptrdiff_t>(best));
			return true;
		}
	}
	if (m_region == nullptr) {
		return false;
	}
	RgfxRegion::Block made = m_region->allocate(bytes);
	if (!made.ok) {
		return false;
	}
	{
		std::lock_guard<std::mutex> hold(m_lock);
		m_made++;
	}
	*block = {made.offset, made.ptr, bytes};
	return true;
}

void RgfxBodyPool::giveBack(const RgfxBodyBlock &block)
{
	if (block.ptr == nullptr)
		return; // none was had: a buffer whose message found no block
	std::lock_guard<std::mutex> hold(m_lock);
	m_free.push_back(block);
}

uint64_t RgfxBodyPool::made() const
{
	std::lock_guard<std::mutex> hold(m_lock);
	return m_made;
}

RgfxUploadArena::RgfxUploadArena(std::shared_ptr<RgfxUploadPool> pool) : m_pool(std::move(pool))
{
}

RgfxUploadArena::~RgfxUploadArena()
{
	giveBack();
}

bool RgfxUploadArena::put(const void *bytes, uint64_t length,
                          __unsafe_unretained id<MTLBuffer> *buffer, uint64_t *offset)
{
	if (m_pool == nullptr) {
		m_failed.store(true, std::memory_order_release);
		return false;
	}
	os_unfair_lock_lock(&m_lock);
	uint64_t at = alignUp(m_cursor, kUploadAlign);
	if (m_chunks.empty() || at + length > m_chunks.back().size) {
		RgfxUploadChunk chunk = {};
		// Rare: a chunk a 64 KiB of bytes, from the pool, which may make one.
		if (!m_pool->take(length == 0 ? 1 : length, &chunk)) {
			os_unfair_lock_unlock(&m_lock);
			m_failed.store(true, std::memory_order_release);
			return false;
		}
		m_chunks.push_back(chunk);
		at = 0;
	}
	RgfxUploadChunk &c = m_chunks.back();
	if (length > 0 && bytes != nullptr) {
		std::memcpy(c.bytes + at, bytes, static_cast<size_t>(length));
	}
	m_cursor = at + length;
	*buffer = c.buffer;
	*offset = at;
	os_unfair_lock_unlock(&m_lock);
	return true;
}

void RgfxUploadArena::giveBack()
{
	std::vector<RgfxUploadChunk> chunks;
	os_unfair_lock_lock(&m_lock);
	chunks.swap(m_chunks);
	m_cursor = 0;
	os_unfair_lock_unlock(&m_lock);
	if (m_pool != nullptr && !chunks.empty()) {
		m_pool->giveBack(chunks);
	}
}
