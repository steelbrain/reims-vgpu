/*
 * RgfxUploads: where the bytes an encoder's set*Bytes:length:atIndex: hands in go, so the host
 * reads them as a buffer bound at an offset (the wire carries no inline constant data).
 *
 * A queue holds one RgfxUploadPool: chunks, each a buffer of the device's slabs
 * (RgfxSlabs.hpp), which lie in the process's own pages where the device maps them, or, where
 * the pool has no slabs, a buffer over a block of the device region (one
 * BUFFER_OVER_HOST_MEMORY on ring 0 when the chunk is made, none after); Shared, or Managed on a
 * GPU of its own memory (below). The host sees a Shared chunk in place, so the bytes are copied
 * once, into the chunk, and never again. The pages first: the
 * region is mapped uncached in the guest under xnu's PAT, which an AMD host applies, until the
 * x86-64 kext sets the PAT's entry 2 to write-back (RgfxRegion.hpp). A command
 * buffer holds one RgfxUploadArena: it takes chunks from its queue's pool as its encoders need
 * room, packs each call's bytes at the next 256-byte boundary (the largest constant-buffer
 * offset alignment of the host GPUs, Metal's included), and gives the chunks back to the pool
 * once the host has completed it, so the next command buffer reuses them. A chunk is
 * kUploadChunkSize bytes; a call larger than that takes a chunk of its own size, rounded up to a
 * page.
 *
 * Each encoder fills a run of a chunk of its own (RgfxUploadCursor), where its puts take no lock,
 * and hands what is left of it back as it ends, so the command buffer's next encoder fills on
 * from there: the bytes pack as they would from one cursor of the whole buffer's, and a
 * parallel render encoder's sub-encoders, each with a run of its own, fill one arena from
 * several threads. With a lock taken for every put, a setVertexBytes of 16 bytes cost about
 * 100 ns in a macOS 26.7.2 x86 guest on a Radeon 780M host and on an RTX 3090 host, against 12 ns
 * for Metal's on an AMD 5300M and an Intel 630, and `sample` put 89% of it in
 * os_unfair_lock_unlock (measured with tools/encode_probe.m, 2026-11-12).
 *
 * Where the device keeps a GPU copy of a Managed buffer (it answers hasUnifiedMemory NO: a GPU
 * of its own memory), the chunks are Managed buffers, and the commit tells the host the bytes
 * its encoders wrote in each (tellWritten: a didModifyRange: of each chunk's written extent), so
 * the GPU reads them from its own memory, as Metal's driver copies set*Bytes data into its own
 * memory on the AMD 5300M (1.00x of a Private buffer's reads, where a Shared buffer's over the
 * bus read 1.05x at 4 KiB and up to 1.5x at 64 KiB on a dependent chain; measured with a
 * constant-read probe under macOS 26.7.1, 2026-11-16). Where the device's word holds
 * RGFX_FEATURE_UPLOADS_WRITTEN, the extents ride the command buffer's own stream
 * (RGFX_OP_UPLOADS_WRITTEN), which the host carries out as it begins the buffer; else a range
 * message of each goes ahead of the command buffer on the control ring, as an app's does. Either
 * way the host copies the range into the GPU's copy before the buffer's commands. The stream's
 * record waits for nothing on ring 0, where a range message costs the commit a held message's
 * timer and kext call, and the buffer a wait at the host's gate for the lane that carries the
 * message out (traced on an RX 7900 XTX host, 2026-11-16: medians of 12 and 11 us more before
 * the host took a one-dispatch buffer, and 18 to 26 us more at the gate). A chunk goes back to its pool only once the host has completed the command buffer,
 * so no command buffer in flight reads a chunk a later one writes, and none committed before the
 * buffer uses a chunk it names.
 * Where the GPU shares the CPU's memory the chunks are Shared, and the host reads them in place.
 *
 * Thread safety: one thread puts through a cursor at a time, as Metal's encoders are used; the
 * arena's chunks, its spare run and the pool are each locked: runs are taken and handed back on
 * the encoders' threads, and completions give chunks back on the completion thread. The arena
 * holds the pool, so the pool outlives the queue while its command buffers are in flight.
 *
 * Our own ObjC++; no Apple private SPI.
 */

#ifndef RGFX_UPLOADS_H_
#define RGFX_UPLOADS_H_

#import <Metal/Metal.h>

#include <atomic>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <os/lock.h>
#include <vector>

#include "RgfxEncoder.hpp"

class RgfxRegion;
class RgfxSerializer;
class RgfxSlabs;

/// One chunk: its buffer (an RgfxBuffer of the slabs, or over a region block, Shared or
/// Managed as its pool's), CPU pointer and size, and, while a command buffer holds it, the bytes
/// from its start the command buffer's puts wrote (RgfxUploadArena::tellWritten).
struct RgfxUploadChunk {
	id<MTLBuffer> buffer;
	uint8_t *bytes;
	uint64_t size;
	uint64_t written = 0;
};

class RgfxUploadPool {
public:
	/// A pool over `slabs` where given, else over `region`, its chunks' buffers created through
	/// `serializer` for `device` (weak; nil in a unit test), Managed where `managed` (the
	/// device keeps a GPU copy of a Managed buffer), else Shared. The pointers are borrowed and
	/// outlive the pool's queue.
	RgfxUploadPool(RgfxRegion *region, RgfxSerializer *serializer, id<MTLDevice> device,
	               RgfxSlabs *slabs = nullptr, bool managed = false);

	/// A free chunk of at least `size` bytes, made where none is free. `ok` false (no chunk)
	/// where the region has no room or the pool has no region or route.
	bool take(uint64_t size, RgfxUploadChunk *chunk);

	/// Takes back chunks a command buffer is done with.
	void giveBack(std::vector<RgfxUploadChunk> &chunks);

	/// How many chunks the pool has made (diagnostics and tests).
	uint64_t made() const;

	/// Whether its chunks are Managed buffers, whose written bytes a commit tells the host.
	bool managed() const { return m_managed; }

private:
	RgfxRegion *m_region;
	RgfxSerializer *m_serializer;
	RgfxSlabs *m_slabs;
	const bool m_managed;
	__weak id<MTLDevice> m_device;
	mutable std::mutex m_lock;
	std::vector<RgfxUploadChunk> m_free;
	uint64_t m_made = 0;
};

/// A block a command buffer too large for its queue's ring puts its message in.
using RgfxBodyBlock = RgfxMessageBlock;

/// A queue's blocks for the messages of its command buffers too large for its ring: one taken
/// at the commit, the message copied into it (RgfxEncoder::finish), given back at the completion
/// (the host read the message out of it before it ran the buffer), or as a buffer no completion
/// comes for goes, and reused by a later such buffer, so the block stays
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

/// The run of an arena's chunk one encoder fills (RgfxUploadArena::put): the chunk's buffer and
/// bytes, its size, the next free byte, the arena's generation when it was taken, and which of
/// the arena's chunks it is. Empty (no bytes) until the encoder's first put, and once handed
/// back (RgfxUploadArena::release).
struct RgfxUploadCursor {
	__unsafe_unretained id<MTLBuffer> buffer = nil;
	uint8_t *bytes = nullptr;
	uint64_t size = 0;
	uint64_t at = 0;
	uint64_t generation = 0;
	size_t chunk = 0;
};

class RgfxUploadArena {
public:
	explicit RgfxUploadArena(std::shared_ptr<RgfxUploadPool> pool);
	~RgfxUploadArena();
	RgfxUploadArena(const RgfxUploadArena &) = delete;
	RgfxUploadArena &operator=(const RgfxUploadArena &) = delete;

	/// Copies `length` bytes at `bytes` into `cursor`'s run at its next 256-byte boundary, and
	/// answers the chunk's buffer (unretained: the arena holds it until giveBack) and the offset
	/// there. Where the run has no room, or the cursor none yet, it takes one first (refill):
	/// the spare run an ended encoder handed back, else a chunk of the pool's. False (nothing
	/// copied) where no chunk was had. A put that fits its run takes no lock.
	bool put(RgfxUploadCursor &cursor, const void *bytes, uint64_t length,
	         __unsafe_unretained id<MTLBuffer> *buffer, uint64_t *offset)
	{
		const uint64_t at = alignUp(cursor.at);
		if (__builtin_expect(cursor.bytes != nullptr && at <= cursor.size &&
		                         length <= cursor.size - at &&
		                         cursor.generation == m_generation.load(std::memory_order_relaxed),
		                     1)) {
			if (length > 0 && bytes != nullptr)
				std::memcpy(cursor.bytes + at, bytes, static_cast<size_t>(length));
			cursor.at = at + length;
			*buffer = cursor.buffer;
			*offset = at;
			return true;
		}
		return refill(cursor, bytes, length, buffer, offset);
	}

	/// Hands back what is left of `cursor`'s run as its encoder ends, and empties it: the next
	/// encoder's first put fills on from there (the spare run, the roomier where several are
	/// handed back).
	void release(RgfxUploadCursor &cursor);

	/// At the command buffer's commit, once its encoders have ended: where the pool's chunks are
	/// Managed, tells the host each chunk's bytes the puts wrote, from its start to the end of the
	/// last run's bytes, so the host's copy of them in the GPU's memory holds them before the
	/// buffer runs: as records of the buffer's RGFX_OP_UPLOADS_WRITTEN appended to `stream`, where
	/// given (the device's word holds RGFX_FEATURE_UPLOADS_WRITTEN), else, and for a chunk whose
	/// buffer names no range itself (RgfxBuffer's rgfxRangeObjId), by its didModifyRange:.
	/// Nothing where they are Shared, or nothing was put.
	void tellWritten(std::vector<RgfxUploadWritten> *stream = nullptr);

	/// Gives every chunk taken back to the pool: the host is done reading them. A cursor's run
	/// taken before it is the pool's again, and the cursor's next put takes another.
	void giveBack();

	/// Whether a put() found no chunk: the command buffer's inline bytes are not all bound, so
	/// it fails (MTLCommandBufferErrorOutOfMemory, as Metal's does when memory runs out).
	bool failed() const { return m_failed.load(std::memory_order_acquire); }

	/// The offset alignment of each call's bytes in a chunk.
	static const uint64_t kUploadAlign = 256;
	/// A chunk's size.
	static const uint64_t kUploadChunkSize = 0x10000;

private:
	static uint64_t alignUp(uint64_t v) { return (v + kUploadAlign - 1) & ~(kUploadAlign - 1); }
	/// put() where `cursor`'s run has no room for `length` bytes: takes a run that has, under
	/// the lock, then copies there.
	bool refill(RgfxUploadCursor &cursor, const void *bytes, uint64_t length,
	            __unsafe_unretained id<MTLBuffer> *buffer, uint64_t *offset);
	/// Notes the bytes `cursor`'s run wrote in its chunk, where it holds a run of this
	/// generation's, as it leaves the run. Under the lock.
	void noteWritten(const RgfxUploadCursor &cursor);

	std::shared_ptr<RgfxUploadPool> m_pool;
	os_unfair_lock m_lock = OS_UNFAIR_LOCK_INIT; // a run's take is a few stores: no sleeping lock
	std::vector<RgfxUploadChunk> m_chunks;       // taken, until giveBack
	RgfxUploadCursor m_spare;                    // the run an ended encoder handed back, if any
	// Moved by each giveBack, under the lock: a cursor of an older one holds no run.
	std::atomic<uint64_t> m_generation{1};
	std::atomic<bool> m_failed{false}; // a put() found no chunk
};

#endif // RGFX_UPLOADS_H_
