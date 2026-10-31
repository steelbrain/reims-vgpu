/*
 * RgfxHeapRanges: where an Automatic heap places the resources made from it, as Metal's does.
 * The host makes the heap a Placement heap, and each resource goes to the offset picked here.
 *
 * Metal's Automatic heap, as measured (M3 Max, macOS 27, buffers placed by their gpuAddress):
 * - a resource takes its size exactly: a buffer its length, a texture its
 *   heapTextureSizeAndAlignWithDescriptor: size;
 * - it lies at the lowest offset, at its alignment, where that many bytes are free (first fit,
 *   not best fit): with holes of 5000 and 1000 bytes in that order, 800 bytes go in the first;
 * - the size goes back when the resource goes or is made aliasable, whichever comes first, and
 *   it joins the free bytes beside it;
 * - usedSize is the sum of the sizes taken; maxAvailableSizeWithAlignment: is the longest run of
 *   free bytes from an offset at that alignment (0 and 1 count as 1; Metal asserts for one that
 *   is not a power of two, and this answers 0);
 * - a size of 0 and a size no free run holds make nothing.
 *
 * Not thread-safe: RgfxHeap holds a lock around it. Header-only and free of Metal's headers, so
 * the tests build it as the bundle does. Our own code.
 */
#ifndef RGFX_HEAP_RANGES_HPP_
#define RGFX_HEAP_RANGES_HPP_

#include <stdint.h>

#include <iterator>
#include <map>

class RgfxHeapRanges {
public:
	/// A heap of `size` bytes, every one free.
	explicit RgfxHeapRanges(uint64_t size)
	{
		if (size != 0)
			m_free[0] = size;
	}

	/// Takes `size` bytes at the lowest offset aligned to `align` (0 and 1 are any offset)
	/// where all of them are free, and answers it in `*offset`. False, taking nothing, for a
	/// size of 0, an alignment that is not a power of two, or where no free run holds them.
	bool take(uint64_t size, uint64_t align, uint64_t *offset)
	{
		if (size == 0 || !powerOfTwo(align))
			return false;
		for (auto it = m_free.begin(); it != m_free.end(); ++it) {
			const uint64_t start = it->first, end = it->first + it->second;
			uint64_t at = 0;
			if (!alignUp(start, align, &at) || at >= end || end - at < size)
				continue;
			// The run splits into what lies before the taken bytes and what lies after them.
			m_free.erase(it);
			if (at > start)
				m_free[start] = at - start;
			if (at + size < end)
				m_free[at + size] = end - (at + size);
			m_used += size;
			*offset = at;
			return true;
		}
		return false;
	}

	/// Gives back the `size` bytes at `offset` that take() took, joined to the free bytes on
	/// either side.
	void give(uint64_t offset, uint64_t size)
	{
		if (size == 0)
			return;
		m_used -= size < m_used ? size : m_used;
		uint64_t start = offset, length = size;
		auto after = m_free.lower_bound(offset);
		if (after != m_free.begin()) {
			auto before = std::prev(after);
			if (before->first + before->second == start) {
				start = before->first;
				length += before->second;
				m_free.erase(before);
			}
		}
		if (after != m_free.end() && after->first == offset + size) {
			length += after->second;
			m_free.erase(after);
		}
		m_free[start] = length;
	}

	/// The bytes taken and not given back.
	uint64_t used() const { return m_used; }

	/// The longest run of free bytes from an offset aligned to `align` (0 and 1 are any
	/// offset); 0 for an alignment that is not a power of two.
	uint64_t maxAvailable(uint64_t align) const
	{
		if (!powerOfTwo(align))
			return 0;
		uint64_t best = 0;
		for (const auto &run : m_free) {
			const uint64_t end = run.first + run.second;
			uint64_t at = 0;
			if (alignUp(run.first, align, &at) && at < end && end - at > best)
				best = end - at;
		}
		return best;
	}

private:
	static bool powerOfTwo(uint64_t align) { return (align & (align - 1)) == 0; }

	static bool alignUp(uint64_t v, uint64_t align, uint64_t *out)
	{
		if (align <= 1) {
			*out = v;
			return true;
		}
		if (__builtin_add_overflow(v, align - 1, out))
			return false;
		*out &= ~(align - 1);
		return true;
	}

	std::map<uint64_t, uint64_t> m_free; // the free runs: start -> length, none touching
	uint64_t m_used = 0;
};

#endif // RGFX_HEAP_RANGES_HPP_
