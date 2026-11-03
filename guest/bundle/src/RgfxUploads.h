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

#include <atomic>
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

/// A block a command buffer too large for its queue's ring puts its message in.
struct RgfxBodyBlock {
	uint64_t offset;
	volatile uint8_t *ptr;
	uint64_t size;
};

/// A queue's blocks for the messages of its command buffers too large for its ring: one taken
/// at the commit, given back at the completion (the host read the message out of it before it
/// ran the buffer), or as a buffer no completion comes for goes, and reused by the next such
/// buffer, so the block stays the process's, and
/// in its window, with no kext call per buffer. A block is a power of two of at least
/// kGranule bytes; the pool keeps kKeep free ones and gives the rest back to the region, and
/// gives back all it holds when it goes. Locked: commits and completions come on any thread.
class RgfxBodyPool {
public:
	static constexpr uint64_t kGranule = 64 * 1024;
	static constexpr size_t kKeep = 8;

	/// A pool over `region` (borrowed; it outlives the pool, as the upload pool's does).
	explicit RgfxBodyPool(RgfxRegion *region);
	~RgfxBodyPool();
	RgfxBodyPool(const RgfxBodyPool &) = delete;
	RgfxBodyPool &operator=(const RgfxBodyPool &) = delete;

	/// A free block of at least `size` bytes, made where none of its size is free. False where
	/// the region has no room.
	bool take(uint64_t size, RgfxBodyBlock *block);
	/// Takes back a block a command buffer is done with.
	void giveBack(const RgfxBodyBlock &block);
	/// How many blocks the pool has made (diagnostics and tests).
	uint64_t made() const;

private:
	RgfxRegion *m_region;
	mutable std::mutex m_lock;
	std::vector<RgfxBodyBlock> m_free;
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

	/// Whether a put() found no chunk: the command buffer's inline bytes are not all bound, so
	/// it fails (MTLCommandBufferErrorOutOfMemory, as Metal's does when memory runs out).
	bool failed() const { return m_failed.load(std::memory_order_acquire); }

	/// The offset alignment of each call's bytes in a chunk.
	static const uint64_t kUploadAlign = 256;
	/// A chunk's size.
	static const uint64_t kUploadChunkSize = 0x10000;

private:
	std::shared_ptr<RgfxUploadPool> m_pool;
	std::mutex m_lock;
	std::vector<RgfxUploadChunk> m_chunks; // taken; the last is the one being filled
	uint64_t m_cursor = 0;                 // the next free byte of the last chunk
	std::atomic<bool> m_failed{false};     // a put() found no chunk
};

#endif // RGFX_UPLOADS_H_
