/*
 * RgfxUploads: where the bytes an encoder's set*Bytes:length:atIndex: hands in go, so the host
 * reads them as a buffer bound at an offset (the wire carries no inline constant data).
 *
 * A queue holds one RgfxUploadPool: chunks, each a Shared buffer of the device's slabs
 * (RgfxSlabs.hpp), which lie in the process's own pages where the device maps them, or, where
 * the pool has no slabs, a Shared buffer over a block of the device region (one
 * BUFFER_OVER_HOST_MEMORY on ring 0 when the chunk is made, none after). The host sees either in
 * place, so the bytes are copied once, into the chunk, and never again. The pages first: the
 * region is mapped uncached in the guest, which an AMD host applies (RgfxRegion.hpp). A command
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
#include <os/lock.h>
#include <vector>

#include "RgfxEncoder.hpp"

class RgfxRegion;
class RgfxSerializer;
class RgfxSlabs;

/// One chunk: its buffer (a Shared RgfxBuffer of the slabs, or over a region block), CPU pointer
/// and size.
struct RgfxUploadChunk {
	id<MTLBuffer> buffer;
	uint8_t *bytes;
	uint64_t size;
};

class RgfxUploadPool {
public:
	/// A pool over `slabs` where given, else over `region`, its chunks' buffers created through
	/// `serializer` for `device` (weak; nil in a unit test). The pointers are borrowed and outlive
	/// the pool's queue.
	RgfxUploadPool(RgfxRegion *region, RgfxSerializer *serializer, id<MTLDevice> device,
	               RgfxSlabs *slabs = nullptr);

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
	RgfxSlabs *m_slabs;
	__weak id<MTLDevice> m_device;
	mutable std::mutex m_lock;
	std::vector<RgfxUploadChunk> m_free;
	uint64_t m_made = 0;
};

/// A block a command buffer too large for its queue's ring puts its message in.
using RgfxBodyBlock = RgfxMessageBlock;

/// A queue's blocks for the messages of its command buffers too large for its ring: one taken
/// as the message outgrows the ring (RgfxEncoder::setMessageBlocks), or at the commit, given
/// back at the completion (the host read the message out of it before it ran the buffer), or as
/// a buffer no completion comes for goes, and reused by a later such buffer, so the block stays
/// the process's, and in its window, with no kext call per buffer. A block is a power of two of
/// at least kGranule bytes. The pool keeps every block given back, as the upload pool keeps its
/// chunks, and a take reuses the smallest free one that holds the message: so of each size it
/// makes blocks only up to the most of its buffers in flight at once (the queue's
/// maxCommandBufferCount at most), and a stream of large buffers calls the kext for none past
/// them, where a block given back to the region beyond a few kept cost a later buffer a block's
/// allocation and release in the kext, a mapping into the process and the device's grant and
/// release on the control ring. Measured on an RTX 5080 host with ringfill, either way an
/// out-of-line buffer cost what an inline one does, about 2.1 to 2.5 us a dispatch, up to 2000
/// in flight. It gives back all it holds when it goes. Locked: commits and completions come on
/// any thread.
class RgfxBodyPool final : public RgfxMessageBlocks {
public:
	static constexpr uint64_t kGranule = 64 * 1024;

	/// A pool over `region` (borrowed; it outlives the pool, as the upload pool's does).
	explicit RgfxBodyPool(RgfxRegion *region);
	~RgfxBodyPool();
	RgfxBodyPool(const RgfxBodyPool &) = delete;
	RgfxBodyPool &operator=(const RgfxBodyPool &) = delete;

	/// The smallest free block of at least `size` bytes, made where none is free. False where
	/// the region has no room.
	bool take(uint64_t size, RgfxBodyBlock *block) override;
	/// Takes back a block a command buffer is done with.
	void giveBack(const RgfxBodyBlock &block) override;
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
	/// chunk's buffer (unretained: the arena holds it until giveBack) and the offset there.
	/// False (nothing copied) where no chunk was had.
	bool put(const void *bytes, uint64_t length, __unsafe_unretained id<MTLBuffer> *buffer,
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
	os_unfair_lock m_lock = OS_UNFAIR_LOCK_INIT; // a put is a few stores: no sleeping lock
	std::vector<RgfxUploadChunk> m_chunks; // taken; the last is the one being filled
	uint64_t m_cursor = 0;                 // the next free byte of the last chunk
	std::atomic<bool> m_failed{false};     // a put() found no chunk
};

#endif // RGFX_UPLOADS_H_
