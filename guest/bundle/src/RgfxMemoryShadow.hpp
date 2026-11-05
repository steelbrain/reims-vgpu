/*
 * RgfxMemoryShadow: the GPU's copy of app memory a no-copy buffer wraps
 * (`newBufferWithBytesNoCopy:length:options:deallocator:` over memory outside the device
 * region that the host does not use in place: read-only memory, or memory the kext or the host
 * makes no pages block of). The buffer lies over a range of a slab of its own
 * (RgfxSlabs.hpp, RgfxBuffer.h), the block here, and `contents` stays the app's pointer, as
 * Metal's does.
 *
 * The block and the app's memory sync at two points, as the IOSurface copies do
 * (RgfxIOSurface.h):
 *   - before a commit of a command buffer that uses the buffer, from the memory: syncIn();
 *   - when a command buffer that may write it completes, before it reports completion, back to
 *     the memory: writeBack(). A write is a shader binding, a blit's destination, a fill, a
 *     synchronize, a render attachment of a texture over it, or a use with Write usage.
 * Each side's bytes that moved since the two last agreed go to the other, byte by byte, so a
 * CPU write and a GPU write to different bytes both stand, whatever the order of the syncs:
 * the bytes both agreed on last are kept (`m_agreed`). Where both moved the same byte, the
 * side that syncs last wins, as a data race on Metal's shared memory has no answer either.
 *
 * Left as Metal's differs: a GPU write shows in the memory at the completion of its command
 * buffer, not as it runs (a CPU waiting on a shared event the buffer signals sees the bytes
 * as of the last completion); memory the process maps read-only is never written back.
 *
 * Header-only and free of Metal's headers, so the serializer test builds it as the bundle
 * does. Thread-safe. Our own code.
 */
#ifndef RGFX_MEMORY_SHADOW_HPP_
#define RGFX_MEMORY_SHADOW_HPP_

#include <stddef.h>
#include <stdint.h>

#include <algorithm>
#include <atomic>
#include <cstring>
#include <memory>
#include <mutex>
#include <new>
#include <vector>

class RgfxMemoryShadow {
public:
	/// The block `block` (`length` bytes) as the GPU's copy of the app's `memory`, seeded from
	/// it. `writable`: the GPU's writes go back to the memory (false for memory the process
	/// maps read-only). ok() is false where the agreed copy could not be allocated.
	RgfxMemoryShadow(const void *memory, void *block, size_t length, bool writable)
	    : m_memory(static_cast<uint8_t *>(const_cast<void *>(memory))),
	      m_block(static_cast<uint8_t *>(block)), m_length(length), m_writable(writable),
	      m_agreed(new (std::nothrow) uint8_t[length == 0 ? 1 : length])
	{
		if (m_agreed != nullptr) {
			std::memcpy(m_agreed.get(), m_memory, m_length);
			std::memcpy(m_block, m_agreed.get(), m_length);
		}
		live().fetch_add(1, std::memory_order_relaxed);
	}
	~RgfxMemoryShadow() { live().fetch_sub(1, std::memory_order_relaxed); }

	RgfxMemoryShadow(const RgfxMemoryShadow &) = delete;
	RgfxMemoryShadow &operator=(const RgfxMemoryShadow &) = delete;

	bool ok() const { return m_agreed != nullptr; }
	size_t length() const { return m_length; }
	bool writable() const { return m_writable; }

	/// Memory -> block: the bytes the CPU wrote since the last sync.
	void syncIn()
	{
		std::lock_guard<std::mutex> g(m_lock);
		if (m_agreed != nullptr)
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
/// buffer may write, which its completion writes back.
inline std::vector<std::shared_ptr<RgfxMemoryShadow>> rgfxShadowsSyncIn(
    std::vector<RgfxShadowUse> uses)
{
	std::vector<std::shared_ptr<RgfxMemoryShadow>> writes;
	if (uses.empty())
		return writes;
	std::sort(uses.begin(), uses.end(), [](const RgfxShadowUse &a, const RgfxShadowUse &b) {
		return a.shadow.get() < b.shadow.get();
	});
	for (size_t i = 0; i < uses.size();) {
		size_t j = i;
		bool w = false;
		for (; j < uses.size() && uses[j].shadow == uses[i].shadow; j++)
			w = w || uses[j].writes;
		uses[i].shadow->syncIn();
		if (w)
			writes.push_back(uses[i].shadow);
		i = j;
	}
	return writes;
}

#endif // RGFX_MEMORY_SHADOW_HPP_
