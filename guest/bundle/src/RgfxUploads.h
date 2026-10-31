/*
 * RgfxUploads: where the bytes an encoder's set*Bytes:length:atIndex: hands in go, so the host
 * reads them as a buffer bound at an offset (the wire carries no inline constant data).
 *
 * A queue holds one RgfxUploadPool: chunks of the device region, each a Shared buffer over its
 * block (one BUFFER_OVER_HOST_MEMORY on ring 0 when the chunk is made, none after), which the
 * host sees in place, so the bytes are copied once, into the chunk, and never again. A command
 * buffer holds one RgfxUploadArena: it takes chunks from its queue's pool as its encoders need
 * room, packs each call's bytes at the next 256-byte boundary (the largest constant-buffer
 * offset alignment of the host GPUs, Metal's included), and gives the chunks back to the pool
 * once the host has completed it, so the next command buffer reuses them. A chunk is
 * kUploadChunkSize bytes; a call larger than that takes a chunk of its own size, rounded up to a
 * page.
 *
 * Thread safety: the arena and the pool are each locked: a parallel render encoder's
 * sub-encoders may fill one arena from several threads, and completions give chunks back on the
 * completion thread. The arena holds the pool, so the pool outlives the queue while its command
 * buffers are in flight.
 *
 * Our own ObjC++; no Apple private SPI.
 */

#ifndef RGFX_UPLOADS_H_
#define RGFX_UPLOADS_H_

#import <Metal/Metal.h>

#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

class RgfxRegion;
class RgfxSerializer;

/// One chunk: its buffer (a Shared RgfxBuffer over a region block), CPU pointer and size.
struct RgfxUploadChunk {
	id<MTLBuffer> buffer;
	uint8_t *bytes;
	uint64_t size;
};

class RgfxUploadPool {
public:
	/// A pool over `region`, its chunks' buffers created through `serializer` for `device`
	/// (weak; nil in a unit test). Both pointers are borrowed and outlive the pool's queue.
	RgfxUploadPool(RgfxRegion *region, RgfxSerializer *serializer, id<MTLDevice> device);

	/// A free chunk of at least `size` bytes, made where none is free. `ok` false (no chunk)
	/// where the region has no room or the pool has no region or route.
	bool take(uint64_t size, RgfxUploadChunk *chunk);

	/// Takes back chunks a command buffer is done with.
	void giveBack(std::vector<RgfxUploadChunk> &chunks);

	/// How many chunks the pool has made (diagnostics and tests).
	uint64_t made() const;

private:
	RgfxRegion *m_region;
	RgfxSerializer *m_serializer;
	__weak id<MTLDevice> m_device;
	mutable std::mutex m_lock;
	std::vector<RgfxUploadChunk> m_free;
	uint64_t m_made = 0;
};

class RgfxUploadArena {
public:
	explicit RgfxUploadArena(std::shared_ptr<RgfxUploadPool> pool);
	~RgfxUploadArena();
	RgfxUploadArena(const RgfxUploadArena &) = delete;
	RgfxUploadArena &operator=(const RgfxUploadArena &) = delete;

	/// Copies `length` bytes at `bytes` into a chunk at a 256-byte boundary, and answers the
	/// chunk's buffer and the offset there. False (nothing copied) where no chunk was had.
	bool put(const void *bytes, uint64_t length, id<MTLBuffer> __strong *buffer,
	         uint64_t *offset);

	/// Gives every chunk taken back to the pool: the host is done reading them.
	void giveBack();

	/// The offset alignment of each call's bytes in a chunk.
	static const uint64_t kUploadAlign = 256;
	/// A chunk's size.
	static const uint64_t kUploadChunkSize = 0x10000;

private:
	std::shared_ptr<RgfxUploadPool> m_pool;
	std::mutex m_lock;
	std::vector<RgfxUploadChunk> m_chunks; // taken; the last is the one being filled
	uint64_t m_cursor = 0;                 // the next free byte of the last chunk
};

#endif // RGFX_UPLOADS_H_
