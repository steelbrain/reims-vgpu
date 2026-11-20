/*
 * RgfxMemoryShadow: the GPU's copy of app memory a no-copy buffer wraps
 * (`newBufferWithBytesNoCopy:length:options:deallocator:` over memory outside the device
 * region that the host does not use in place: read-only Managed memory, a Managed buffer's where
 * the device makes no Managed buffer over a Shared one, or memory the kext or the host makes no
 * pages block of, read-only memory among it where the kext makes no read-only blocks). The buffer lies over a range of a slab of its own
 * (RgfxSlabs.hpp, RgfxBuffer.h), the block here, and `contents` stays the app's pointer, as
 * Metal's does.
 *
 * The block and the app's memory sync at two points, as the IOSurface copies do
 * (RgfxIOSurface.h):
 *   - before a commit of a command buffer that uses the buffer, from the memory: syncIn();
 *   - when a command buffer that may write it completes, as its completion is taken off the
 *     ring, on its queue's write-back queue, not behind the queue's earlier handlers, and before
 *     it reports completion, back to the memory: writeBack(). A write is a shader binding, a
 *     blit's destination, a fill, a synchronize, a render attachment of a texture over it, or a
 *     use with Write usage.
 * Each side's bytes that moved since the two last agreed go to the other, byte by byte, so a
 * CPU write and a GPU write to different bytes both stand, whatever the order of the syncs:
 * the bytes both agreed on last are kept (`m_agreed`). Where both moved the same byte, the
 * side that syncs last wins, as a data race on Metal's shared memory has no answer either.
 *
 * Memory the process maps read-only is never written back, and keeps no agreed copy: each
 * commit that uses its block writes the whole of the memory into it, with no compare, so the
 * GPU's writes to the block are dropped at the next such commit, as Metal's GPU drops writes to
 * read-only memory.
 *
 * Left as Metal's differs: a GPU write shows in the memory once the completion of its command
 * buffer is taken, not as it runs (a CPU waiting on a shared event the buffer signals sees the
 * bytes as of the last completion taken); a GPU write to read-only memory shows to the work of
 * its command buffer and of those in flight with it until the next commit that uses the block.
 *
 * Counted (RgfxMemoryShadow.cpp): the commits that sync in a shadow of each kind, writable and
 * read-only, the bytes the syncs went through and their time, and the completions that write
 * back, logged at the info level as the process's totals within a minute of a change.
 *
 * Free of Metal's headers, so the serializer test builds it as the bundle does. Thread-safe. Our
 * own code.
 */
#ifndef RGFX_MEMORY_SHADOW_HPP_
#define RGFX_MEMORY_SHADOW_HPP_

#include <stddef.h>
#include <stdint.h>
#include <time.h>

#include <algorithm>
#include <atomic>
#include <cstring>
#include <memory>
#include <mutex>
#include <new>
#include <vector>

/// What the process's shadows' syncs did (RgfxMemoryShadow.cpp), by kind: kWritable, the
/// shadows of memory the process maps writable, which merge both ways, and kReadOnly, those of
/// read-only memory, which a commit writes whole.
struct RgfxShadowCounts {
	static const unsigned kWritable = 0, kReadOnly = 1, kKinds = 2;
	// Commits that synced in a shadow of the kind.
	uint64_t commits[kKinds];
	// The bytes the syncs went through: a writable shadow's compared at its commits and
	// completions, a read-only one's written whole at its commits.
	uint64_t bytes[kKinds];
	// The time those syncs took.
	uint64_t ns[kKinds];
	// Completions that wrote back a writable shadow.
	uint64_t writeBacks;
};

/// Adds `delta` to the process's counts; logs the totals within a minute.
void rgfxShadowsNote(const RgfxShadowCounts &delta);

/// The process's counts so far.
RgfxShadowCounts rgfxShadowCounts();

class RgfxMemoryShadow {
public:
	/// The block `block` (`length` bytes) as the GPU's copy of the app's `memory`, seeded from
	/// it. `writable`: the GPU's writes go back to the memory (false for memory the process
	/// maps read-only, which keeps no agreed copy). ok() is false where the agreed copy could not
	/// be allocated.
	RgfxMemoryShadow(const void *memory, void *block, size_t length, bool writable)
	    : m_memory(static_cast<uint8_t *>(const_cast<void *>(memory))),
	      m_block(static_cast<uint8_t *>(block)), m_length(length), m_writable(writable),
	      m_agreed(writable ? new (std::nothrow) uint8_t[length == 0 ? 1 : length] : nullptr)
	{
		if (m_agreed != nullptr)
			std::memcpy(m_agreed.get(), m_memory, m_length);
		if (ok())
			std::memcpy(m_block, m_writable ? m_agreed.get() : m_memory, m_length);
		live().fetch_add(1, std::memory_order_relaxed);
	}
	~RgfxMemoryShadow() { live().fetch_sub(1, std::memory_order_relaxed); }

	RgfxMemoryShadow(const RgfxMemoryShadow &) = delete;
	RgfxMemoryShadow &operator=(const RgfxMemoryShadow &) = delete;

	bool ok() const { return !m_writable || m_agreed != nullptr; }
	size_t length() const { return m_length; }
	bool writable() const { return m_writable; }

	/// Memory -> block: the bytes the CPU wrote since the last sync; the whole of read-only
	/// memory, which puts back the GPU's writes to the block.
	void syncIn()
	{
		std::lock_guard<std::mutex> g(m_lock);
		if (!m_writable)
			std::memcpy(m_block, m_memory, m_length);
		else if (m_agreed != nullptr)
			merge(m_memory, m_block, m_agreed.get(), m_length);
	}

	/// Block -> memory: the bytes the GPU wrote since the last sync. Nothing for read-only
	/// memory.
	void writeBack()
	{
		std::lock_guard<std::mutex> g(m_lock);
		if (m_agreed != nullptr && m_writable)
			merge(m_block, m_memory, m_agreed.get(), m_length);
	}

	/// Whether any shadow lives in the process: the encoders look for one only then.
	static bool anyLive() { return live().load(std::memory_order_relaxed) != 0; }

private:
	static std::atomic<int64_t> &live()
	{
		static std::atomic<int64_t> count{0};
		return count;
	}

	/// Copies to `to` (and `agreed`) each byte of `from` that differs from `agreed`, storing
	/// only those, so the bytes the other side moved stay as they are.
	static void merge(const uint8_t *from, uint8_t *to, uint8_t *agreed, size_t length)
	{
		constexpr size_t kChunk = 4096;
		for (size_t base = 0; base < length; base += kChunk) {
			const size_t n = std::min(kChunk, length - base);
			if (std::memcmp(from + base, agreed + base, n) == 0)
				continue;
			size_t i = base;
			const size_t end = base + n;
			for (; i + 8 <= end; i += 8) {
				uint64_t f, a;
				std::memcpy(&f, from + i, 8);
				std::memcpy(&a, agreed + i, 8);
				if (f == a)
					continue;
				if (allBytesDiffer(f ^ a)) {
					std::memcpy(to + i, &f, 8);
					std::memcpy(agreed + i, &f, 8);
					continue;
				}
				for (size_t k = 0; k < 8; k++)
					copyByte(from, to, agreed, i + k);
			}
			for (; i < end; i++)
				copyByte(from, to, agreed, i);
		}
	}

	static void copyByte(const uint8_t *from, uint8_t *to, uint8_t *agreed, size_t i)
	{
		const uint8_t b = from[i];
		if (b != agreed[i]) {
			to[i] = b;
			agreed[i] = b;
		}
	}

	static bool allBytesDiffer(uint64_t x)
	{
		// A byte of x is 0 where the two agree: no zero byte (the classic haszero test).
		return ((x - 0x0101010101010101ull) & ~x & 0x8080808080808080ull) == 0;
	}

	std::mutex m_lock;
	uint8_t *m_memory;
	uint8_t *m_block;
	size_t m_length;
	bool m_writable;
	std::unique_ptr<uint8_t[]> m_agreed;
};

/// A command buffer's use of a shadow, as its encoders note it (RgfxEncoder::noteShadowUse).
struct RgfxShadowUse {
	std::shared_ptr<RgfxMemoryShadow> shadow;
	bool writes;
};

/// At a commit: syncs in each shadow `uses` names, once each, and answers those the command
/// buffer may write, which its completion writes back (rgfxShadowsWriteBack); counted.
inline std::vector<std::shared_ptr<RgfxMemoryShadow>> rgfxShadowsSyncIn(
    std::vector<RgfxShadowUse> uses)
{
	std::vector<std::shared_ptr<RgfxMemoryShadow>> writes;
	if (uses.empty())
		return writes;
	std::sort(uses.begin(), uses.end(), [](const RgfxShadowUse &a, const RgfxShadowUse &b) {
		return a.shadow.get() < b.shadow.get();
	});
	RgfxShadowCounts delta = {};
	for (size_t i = 0; i < uses.size();) {
		size_t j = i;
		bool w = false;
		for (; j < uses.size() && uses[j].shadow == uses[i].shadow; j++)
			w = w || uses[j].writes;
		const uint64_t t0 = clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
		uses[i].shadow->syncIn();
		const unsigned kind = uses[i].shadow->writable() ? RgfxShadowCounts::kWritable
		                                                 : RgfxShadowCounts::kReadOnly;
		delta.commits[kind] = 1;
		delta.bytes[kind] += uses[i].shadow->length();
		delta.ns[kind] += clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - t0;
		if (w)
			writes.push_back(uses[i].shadow);
		i = j;
	}
	rgfxShadowsNote(delta);
	return writes;
}

/// At a completion: writes back each of `writes`, those rgfxShadowsSyncIn answered (nothing for a
/// read-only one), and lets go of them; counted.
inline void rgfxShadowsWriteBack(std::vector<std::shared_ptr<RgfxMemoryShadow>> &writes)
{
	if (writes.empty())
		return;
	RgfxShadowCounts delta = {};
	for (const std::shared_ptr<RgfxMemoryShadow> &shadow : writes) {
		if (!shadow->writable())
			continue;
		const uint64_t t0 = clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
		shadow->writeBack();
		delta.writeBacks = 1;
		delta.bytes[RgfxShadowCounts::kWritable] += shadow->length();
		delta.ns[RgfxShadowCounts::kWritable] += clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - t0;
	}
	writes.clear();
	if (delta.writeBacks != 0)
		rgfxShadowsNote(delta);
}

#endif // RGFX_MEMORY_SHADOW_HPP_
