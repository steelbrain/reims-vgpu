/*
 * RgfxSlabs: the slab allocator of Shared and Managed buffers (RgfxSlabs.hpp), and its source
 * over the device region. Our own C++.
 */

#include "RgfxSlabs.hpp"

#include <mach/kern_return.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <iterator>

#if defined(__APPLE__)
#include <pthread.h>
#endif

#include "RgfxRegion.hpp"
#include "RgfxSerializer.hpp"

namespace {
uint64_t roundUp(uint64_t value, uint64_t unit)
{
	return (value + unit - 1) & ~(unit - 1);
}

uint64_t powerOfTwoAtLeast(uint64_t value)
{
	uint64_t p = 1;
	while (p < value)
		p <<= 1;
	return p;
}

// MTLResourceOptions' storage mode, and its Shared and Managed values.
constexpr uint32_t kStorageShift = 4;
constexpr uint32_t kStorageMask = 0xf0;
constexpr uint32_t kStorageShared = 0;
constexpr uint32_t kStorageManaged = 1;
// MTLHeapTypePlacement: the guest places every buffer in a slab's heap itself.
constexpr int32_t kHeapTypePlacement = 1;
} // namespace

// --- the region's slabs --------------------------------------------------------------------

RgfxRegionSlabSource::RgfxRegionSlabSource(RgfxRegion &region, RgfxSerializer &serializer,
                                           bool managedAtOffset, bool managedOverCopy)
    : m_region(region), m_serializer(serializer), m_managedAtOffset(managedAtOffset),
      m_managedOverCopy(managedOverCopy)
{
}

uint32_t RgfxRegionSlabSource::classOptions(uint32_t options) const
{
	return overCopy(options) ? (options & ~kStorageMask) : options;
}

bool RgfxRegionSlabSource::overCopy(uint32_t options) const
{
	return m_managedOverCopy && ((options & kStorageMask) >> kStorageShift) == kStorageManaged;
}

bool RgfxRegionSlabSource::makeOverCopy(const Made &copy, uint64_t length, uint32_t options,
                                        Made *out)
{
	RgfxCreated created = m_serializer.createManagedBuffer(copy.objId, length, options);
	if (created.obj_id == 0 || m_serializer.refused(created))
		return false; // no address left, or the device never got it
	out->objId = created.obj_id;
	out->gpuAddress = created.gpu_address;
	out->ownsAddress = true;
	return true;
}

void RgfxRegionSlabSource::dropCopy(const Made &copy, std::function<void()> then)
{
	// Its address, where its create picked one, goes back once the destroy is on the ring, then
	// its range (RgfxSerializer::destroy).
	RgfxSerializer &serializer = m_serializer;
	const uint64_t address = copy.ownsAddress ? copy.gpuAddress : 0;
	m_serializer.destroy(copy.objId, [&serializer, address, then = std::move(then)] {
		if (address != 0)
			serializer.releaseAddress(address);
		then();
	});
}

uint64_t RgfxRegionSlabSource::capacity() const
{
	return m_region.size() > m_region.floor() ? m_region.size() - m_region.floor() : 0;
}

bool RgfxRegionSlabSource::heaped(uint32_t options) const
{
	const uint32_t storage = (options & kStorageMask) >> kStorageShift;
	return storage == kStorageShared || (storage == kStorageManaged && m_managedAtOffset);
}

bool RgfxRegionSlabSource::take(uint64_t size, Memory *out)
{
	RgfxRegion::Block block = m_region.allocate(size, kUnit);
	if (!block.ok)
		return false;
	out->bytes = const_cast<uint8_t *>(block.ptr);
	out->size = size;
	out->regionOffset = block.offset;
	out->zeroed = false; // whatever the block's last holder left in it
	return true;
}

void RgfxRegionSlabSource::give(const Memory &memory)
{
	m_region.release(memory.regionOffset);
}

bool RgfxRegionSlabSource::makeHeap(const Memory &memory, uint32_t options, Heap *out)
{
	// A Managed class's heap holds its buffers' CPU copies: Shared, of the class's cache mode
	// and hazard tracking.
	RgfxCreated created = m_serializer.createHeapOverHostMemory(
	    memory.size, options & ~kStorageMask, kHeapTypePlacement, memory.regionOffset);
	if (created.obj_id == 0 || m_serializer.refused(created))
		return false; // no address left, or the device never got it
	out->objId = created.obj_id;
	out->gpuAddress = created.gpu_address;
	return true;
}

void RgfxRegionSlabSource::dropHeap(const Heap &heap)
{
	// Its address goes back once the destroy is on the ring (RgfxSerializer::destroy).
	RgfxSerializer &serializer = m_serializer;
	const uint64_t address = heap.gpuAddress;
	m_serializer.destroy(heap.objId, [&serializer, address] { serializer.releaseAddress(address); });
}

bool RgfxRegionSlabSource::makeBuffer(const Memory &memory, const Heap &heap, uint64_t offset,
                                      uint64_t length, uint32_t options, Made *out)
{
	if (heap.objId != 0 && ((options & kStorageMask) >> kStorageShift) == kStorageManaged) {
		// At an address of its own; its CPU copy is placed in the heap at the offset.
		RgfxCreated created = m_serializer.createBufferManagedAtOffset(heap.objId, offset, length,
		                                                               options);
		if (created.obj_id == 0 || m_serializer.refused(created))
			return false;
		out->objId = created.obj_id;
		out->gpuAddress = created.gpu_address;
		out->ownsAddress = true;
		return true;
	}
	if (heap.objId != 0) {
		// At the heap's address plus the offset, where the host places it in the heap it pinned.
		RgfxCreated created = m_serializer.createBufferAtOffset(heap.objId, offset, length, options);
		if (m_serializer.refused(created))
			return false;
		out->objId = created.obj_id;
		out->gpuAddress = heap.gpuAddress + offset;
		out->ownsAddress = false;
		return true;
	}
	// Over its own range of the block: a Managed buffer's CPU copy.
	RgfxCreated created =
	    m_serializer.createBufferOverHostMemory(length, options, memory.regionOffset + offset);
	if (created.obj_id == 0 || m_serializer.refused(created))
		return false;
	out->objId = created.obj_id;
	out->gpuAddress = created.gpu_address;
	out->ownsAddress = true;
	return true;
}

// --- the process's own pages -----------------------------------------------------------------

RgfxPagesSlabSource::RgfxPagesSlabSource(RgfxRegion &region, RgfxSerializer &serializer,
                                         const RgfxPagesMemory &memory, uint64_t capacity,
                                         bool managedAtOffset, bool managedOverCopy)
    : RgfxRegionSlabSource(region, serializer, managedAtOffset, managedOverCopy),
      m_memory(memory), m_capacity(capacity)
{
}

bool RgfxPagesSlabSource::serves(uint32_t options, uint64_t size) const
{
	// A 2 MiB slab for any heaped class; a larger one only for a Shared buffer of its own, which
	// BUFFER_OVER_PAGES makes over the view.
	const bool shared = ((options & kStorageMask) >> kStorageShift) == kStorageShared;
	return heaped(options) && (size == kUnit || shared);
}

bool RgfxPagesSlabSource::heapedSlab(const Memory &memory, uint32_t options) const
{
	return memory.size == kUnit && heaped(options);
}

bool RgfxPagesSlabSource::take(uint64_t size, Memory *out)
{
	if (size == 0 || size % kUnit != 0)
		return false;
	// The kext's chunks, the block's own memory; where it gives none, plain pages, which the kext
	// wires as it makes the block; where the kernel has none to give or the kext no pages block,
	// the region's slab.
	RgfxRegion::Block chunks = m_region.acquireChunkedPages(size);
	if (chunks.ok) {
		out->bytes = const_cast<uint8_t *>(chunks.ptr);
		out->size = size;
		out->regionOffset = chunks.offset;
		out->blockSize = chunks.size;
		out->zeroed = true; // the kext zero-fills its chunks
		out->chunked = true;
		return true;
	}
	uint8_t *bytes = nullptr;
	if (m_memory.pages == nullptr ||
	    m_memory.pages(m_memory.context, size, &bytes) != KERN_SUCCESS)
		return false;
	RgfxRegion::Block block = m_region.acquirePages(bytes, size, true);
	if (!block.ok) {
		m_memory.unmap(m_memory.context, bytes, size);
		return false;
	}
	out->bytes = bytes;
	out->size = size;
	out->regionOffset = block.offset;
	out->blockSize = block.size;
	out->zeroed = true; // fresh pages of the kernel's, zero-filled
	return true;
}

void RgfxPagesSlabSource::give(const Memory &memory)
{
	// The block's release first: the kext answers once the device has let go of it, and unwires
	// the pages, or unmaps and frees its chunks.
	m_region.release(memory.regionOffset);
	if (!memory.chunked)
		m_memory.unmap(m_memory.context, memory.bytes, memory.size);
}

bool RgfxPagesSlabSource::makeHeap(const Memory &memory, uint32_t options, Heap *out)
{
	// As a region slab's: Shared, of the class's cache mode and hazard tracking.
	RgfxCreated created = m_serializer.createHeapOverPages(
	    memory.size, options & ~kStorageMask, kHeapTypePlacement, memory.regionOffset,
	    memory.blockSize);
	if (created.obj_id == 0 || m_serializer.refused(created))
		return false; // no address left, or the device never got it
	out->objId = created.obj_id;
	out->gpuAddress = created.gpu_address;
	return true;
}

bool RgfxPagesSlabSource::makeBuffer(const Memory &memory, const Heap &heap, uint64_t offset,
                                     uint64_t length, uint32_t options, Made *out)
{
	if (heap.objId != 0)
		return RgfxRegionSlabSource::makeBuffer(memory, heap, offset, length, options, out);
	// A dedicated slab larger than 2 MiB (offset 0): a Shared buffer over the block's view, at an
	// address picked for it.
	RgfxCreated created = m_serializer.createBufferOverPages(length, options, memory.regionOffset,
	                                                         memory.blockSize, offset);
	if (created.obj_id == 0 || m_serializer.refused(created))
		return false;
	out->objId = created.obj_id;
	out->gpuAddress = created.gpu_address + offset;
	out->ownsAddress = true;
	return true;
}

// --- runs ----------------------------------------------------------------------------------

void RgfxSlabs::Runs::add(uint64_t at, uint64_t length)
{
	if (length == 0)
		return;
	bytes += length;
	uint64_t start = at, end = at + length;
	auto after = runs.lower_bound(at);
	if (after != runs.begin()) {
		auto before = std::prev(after);
		if (before->first + before->second == start) {
			start = before->first;
			runs.erase(before);
		}
	}
	if (after != runs.end() && after->first == end) {
		end += after->second;
		runs.erase(after);
	}
	runs[start] = end - start;
}

bool RgfxSlabs::Runs::takeFirstFit(uint64_t length, uint64_t *at)
{
	if (length > bytes)
		return false;
	for (auto it = runs.begin(); it != runs.end(); ++it) {
		if (it->second < length)
			continue;
		const uint64_t start = it->first, rest = it->second - length;
		runs.erase(it);
		if (rest != 0)
			runs[start + length] = rest;
		bytes -= length;
		*at = start;
		return true;
	}
	return false;
}

bool RgfxSlabs::Runs::takeAt(uint64_t at, uint64_t length)
{
	auto it = runs.upper_bound(at);
	if (it == runs.begin())
		return false;
	--it;
	const uint64_t start = it->first, end = it->first + it->second;
	if (at + length > end)
		return false;
	runs.erase(it);
	if (start < at)
		runs[start] = at - start;
	if (at + length < end)
		runs[at + length] = end - (at + length);
	bytes -= length;
	return true;
}

void RgfxSlabs::Runs::cut(uint64_t at, uint64_t length,
                          std::vector<std::pair<uint64_t, uint64_t>> *out)
{
	const uint64_t end = at + length;
	auto it = runs.upper_bound(at);
	if (it != runs.begin() && std::prev(it)->first + std::prev(it)->second > at)
		--it;
	while (it != runs.end() && it->first < end) {
		const uint64_t start = it->first, stop = it->first + it->second;
		it = runs.erase(it);
		if (start < at)
			runs[start] = at - start;
		if (stop > end)
			runs[end] = stop - end;
		const uint64_t from = std::max(start, at), to = std::min(stop, end);
		bytes -= to - from;
		if (out != nullptr)
			out->emplace_back(from, to - from);
	}
}

// --- the allocator ---------------------------------------------------------------------------

RgfxSlabs::RgfxSlabs(RgfxSlabSource &source) : m_source(source)
{
}

void RgfxSlabs::setFirst(RgfxSlabSource *first)
{
	std::lock_guard<std::mutex> hold(m_lock);
	m_first = first;
}

RgfxSlabs::~RgfxSlabs()
{
	{
		std::lock_guard<std::mutex> hold(m_lock);
		m_stop = true;
	}
	m_wake.notify_all();
	if (m_thread.joinable())
		m_thread.join();
	for (Slab *slab : m_all)
		delete slab;
}

bool RgfxSlabs::start()
{
	std::lock_guard<std::mutex> hold(m_lock);
	if (m_running)
		return true;
	try {
		m_thread = std::thread(&RgfxSlabs::run, this);
	} catch (...) {
		return false;
	}
	m_running = true;
	return true;
}

void RgfxSlabs::setClock(Clock now, void *context)
{
	std::lock_guard<std::mutex> hold(m_lock);
	m_clock = now;
	m_clockContext = context;
}

uint64_t RgfxSlabs::nowMs() const
{
	if (m_clock != nullptr)
		return m_clock(m_clockContext);
	return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
	                                 std::chrono::steady_clock::now().time_since_epoch())
	                                 .count());
}

RgfxSlabs::Class &RgfxSlabs::classOf(uint32_t options)
{
	return m_classes[options];
}

RgfxSlabs::Slab *RgfxSlabs::makeSlab(RgfxSlabSource &source, uint64_t size, uint32_t options,
                                     bool dedicated, bool *refused)
{
	RgfxSlabSource::Memory memory;
	if (!source.take(size, &memory))
		return nullptr;
	RgfxSlabSource::Heap heap;
	if (source.heapedSlab(memory, options) && !source.makeHeap(memory, options, &heap)) {
		source.give(memory);
		*refused = true; // the device never got the heap: no smaller slab gets one either
		return nullptr;
	}
	Slab *slab = new Slab;
	slab->source = &source;
	slab->memory = memory;
	slab->heap = heap;
	slab->options = options;
	slab->dedicated = dedicated;
	slab->free.add(0, memory.size);
	if (!memory.zeroed)
		slab->dirty.add(0, memory.size);
	return slab;
}

bool RgfxSlabs::takeIn(Slab *slab, uint64_t size, uint64_t keep,
                       std::vector<std::pair<uint64_t, uint64_t>> *zero, uint64_t *at)
{
	if (slab->dedicated) {
		if (!slab->free.takeAt(0, size))
			return false;
		*at = 0;
	} else if (!slab->free.takeFirstFit(size, at)) {
		return false;
	}
	slab->used += size;
	// The caller writes its first `keep` bytes; the rest of what was dirty reads zero first.
	std::vector<std::pair<uint64_t, uint64_t>> dirty;
	slab->dirty.cut(*at, size, &dirty);
	for (const auto &piece : dirty) {
		const uint64_t from = std::max(piece.first, *at + keep);
		const uint64_t to = piece.first + piece.second;
		if (from < to)
			zero->emplace_back(from, to - from);
	}
	if (slab->dirty.bytes == 0)
		m_dirtySlabs.erase(slab);
	return true;
}

bool RgfxSlabs::take(uint64_t length, uint32_t options, uint64_t keep, Range *out)
{
	// A Managed buffer over a copy lies in its Shared class (MANAGED OVER A COPY).
	options = m_source.classOptions(options);
	RgfxSlabSource *first = nullptr;
	{
		std::lock_guard<std::mutex> hold(m_lock);
		first = m_first;
	}
	const uint64_t widest = std::max(m_source.unit(), first != nullptr ? first->unit() : 0);
	if (length == 0 || length > UINT64_MAX - widest)
		return false;
	const uint64_t size = roundUp(length, kGranule);
	keep = std::min(keep, size);
	const bool dedicated = size > kSmallMax;
	const uint64_t unit = m_source.unit();
	std::vector<std::pair<uint64_t, uint64_t>> zero;
	Slab *in = nullptr;
	uint64_t at = 0;
	uint64_t want = 0;
	bool wake = false; // dirty bytes left for the background
	{
		std::lock_guard<std::mutex> hold(m_lock);
		if (dedicated) {
			// The smallest cached slab of the class that holds it, and no more than twice it.
			Slab *best = nullptr;
			for (Slab *slab : m_cached) {
				const uint64_t need = roundUp(size, slab->source->unit());
				if (slab->options == options && slab->zeroing == 0 && slab->memory.size >= need &&
				    slab->memory.size / 2 <= need &&
				    (best == nullptr || slab->memory.size < best->memory.size))
					best = slab;
			}
			if (best != nullptr && takeIn(best, size, keep, &zero, &at)) {
				uncache(best);
				in = best;
				wake = best->dirty.bytes != 0;
			}
		} else {
			Class &cls = classOf(options);
			for (Slab *slab : cls.open) {
				const bool wasEmpty = slab->used == 0;
				if (slab->free.bytes >= size && takeIn(slab, size, keep, &zero, &at)) {
					if (wasEmpty)
						cls.empty--;
					if (slab->free.bytes == 0)
						cls.open.erase(slab);
					in = slab;
					break;
				}
			}
			if (in == nullptr) {
				want = std::max(cls.nextSize, powerOfTwoAtLeast(2 * size));
				want = std::max(std::min(want, kSlabMax), roundUp(size, unit));
				want = roundUp(want, unit);
			}
		}
	}
	if (in == nullptr) {
		// A new slab, from the first source at its unit, else the region's; once both refused
		// one, the process's own room given back and the slab asked again; then, for a small
		// buffer, region slabs half as large down to the buffer's size.
		const uint64_t least = roundUp(size, unit);
		uint64_t ask = dedicated ? least : want;
		bool refused = false;
		auto fromEither = [&]() -> Slab * {
			const uint64_t firstSize =
			    first != nullptr ? roundUp(dedicated ? size : ask, first->unit()) : 0;
			if (first != nullptr && first->serves(options, firstSize)) {
				bool firstRefused = false; // the region's slab is asked all the same
				Slab *slab = makeSlab(*first, firstSize, options, dedicated, &firstRefused);
				if (slab != nullptr)
					return slab;
			}
			return makeSlab(m_source, ask, options, dedicated, &refused);
		};
		Slab *made = fromEither();
		if (made == nullptr && !refused) {
			dropDoomed(true);
			made = fromEither();
		}
		while (made == nullptr && !refused && !dedicated && ask > least) {
			ask = std::max(roundUp(ask / 2, unit), least);
			made = makeSlab(m_source, ask, options, dedicated, &refused);
		}
		if (made == nullptr)
			return false; // the source has no room for it
		std::lock_guard<std::mutex> hold(m_lock);
		made->serial = m_nextSerial++;
		m_all.insert(made);
		takeIn(made, size, keep, &zero, &at); // a new slab holds it from its first byte
		if (made->dirty.bytes != 0) {
			m_dirtySlabs.insert(made);
			wake = true;
		}
		if (!dedicated) {
			Class &cls = classOf(options);
			cls.slabs.push_back(made);
			if (made->free.bytes != 0)
				cls.open.insert(made);
			if (ask == want)
				cls.nextSize = std::min(want * 2, kSlabMax);
		}
		in = made;
	}
	if (wake)
		m_wake.notify_one();
	for (const auto &piece : zero)
		std::memset(in->memory.bytes + piece.first, 0, piece.second);
	out->slab = in;
	out->offset = at;
	out->size = size;
	out->contents = in->memory.bytes + at;
	return true;
}

bool RgfxSlabs::makeBuffer(const Range &range, uint64_t length, uint32_t options,
                           RgfxSlabSource::Made *out)
{
	Slab *slab = const_cast<Slab *>(static_cast<const Slab *>(range.slab));
	RgfxSlabSource &source = *slab->source;
	if (!source.overCopy(options))
		return source.makeBuffer(slab->memory, slab->heap, range.offset, length, options, out);
	// The Shared copy first, as a buffer of the slab's class, then the Managed buffer over it.
	RgfxSlabSource::Made copy;
	if (!source.makeBuffer(slab->memory, slab->heap, range.offset, length, slab->options, &copy))
		return false;
	// Where the Managed buffer is not made, the caller's give() destroys the copy all the same.
	const bool made = source.makeOverCopy(copy, length, options, out);
	std::lock_guard<std::mutex> hold(m_lock);
	slab->copies[range.offset] = copy;
	return made;
}

void RgfxSlabs::uncache(Slab *slab)
{
	auto it = std::find(m_cached.begin(), m_cached.end(), slab);
	if (it == m_cached.end())
		return;
	m_cached.erase(it);
	m_cachedBytes -= slab->memory.size;
}

void RgfxSlabs::doom(Slab *slab)
{
	if (slab->doomed)
		return;
	slab->doomed = true;
	if (slab->dedicated) {
		uncache(slab);
	} else {
		Class &cls = classOf(slab->options);
		cls.open.erase(slab);
		auto it = std::find(cls.slabs.begin(), cls.slabs.end(), slab);
		if (it != cls.slabs.end())
			cls.slabs.erase(it);
		if (slab->used == 0 && cls.empty > 0)
			cls.empty--;
	}
	m_dirtySlabs.erase(slab);
	m_doomed.push_back(slab);
}

void RgfxSlabs::give(const Range &range)
{
	Slab *slab = const_cast<Slab *>(static_cast<const Slab *>(range.slab));
	RgfxSlabSource::Made copy;
	{
		std::lock_guard<std::mutex> hold(m_lock);
		auto it = slab->copies.find(range.offset);
		if (it != slab->copies.end()) {
			copy = it->second;
			slab->copies.erase(it);
		}
	}
	if (copy.objId == 0) {
		giveBack(range);
		return;
	}
	// The Managed buffer's destroy is posted: its copy's follows it, then the range goes back.
	slab->source->dropCopy(copy, [this, range] { giveBack(range); });
}

void RgfxSlabs::giveBack(const Range &range)
{
	Slab *slab = const_cast<Slab *>(static_cast<const Slab *>(range.slab));
	bool dropNow = false;
	{
		std::lock_guard<std::mutex> hold(m_lock);
		slab->free.add(range.offset, range.size);
		slab->dirty.add(range.offset, range.size);
		slab->used -= range.size;
		m_dirtySlabs.insert(slab);
		if (slab->dedicated) {
			// Cached for a later buffer, the oldest of its source's let go past a sixteenth of the
			// source.
			slab->idleSince = nowMs();
			m_cached.push_back(slab);
			m_cachedBytes += slab->memory.size;
			const RgfxSlabSource *source = slab->source;
			const uint64_t cap = source->capacity() / 16;
			while (cachedBytesOf(source) > cap)
				doom(*std::find_if(m_cached.begin(), m_cached.end(),
				                   [source](const Slab *s) { return s->source == source; }));
		} else {
			Class &cls = classOf(slab->options);
			cls.open.insert(slab);
			if (slab->used == 0 && ++cls.empty > kEmptyKept) {
				doom(slab);
			} else if (slab->used == 0) {
				// Kept for kCacheIdleMs, then let go where nothing took from it meanwhile.
				slab->idleSince = nowMs();
				const uint64_t due = slab->idleSince + kCacheIdleMs;
				if (m_emptyDue == 0 || due < m_emptyDue)
					m_emptyDue = due;
			}
		}
		dropNow = !m_running && !m_doomed.empty();
	}
	m_wake.notify_one();
	if (dropNow)
		dropDoomed(false);
}

void RgfxSlabs::drop(Slab *slab)
{
	if (slab->heap.objId != 0)
		slab->source->dropHeap(slab->heap);
	slab->source->give(slab->memory);
	delete slab;
}

uint64_t RgfxSlabs::cachedBytesOf(const RgfxSlabSource *source) const
{
	uint64_t bytes = 0;
	for (const Slab *slab : m_cached) {
		if (slab->source == source)
			bytes += slab->memory.size;
	}
	return bytes;
}

bool RgfxSlabs::reclaim()
{
	return dropDoomed(true) != 0;
}

size_t RgfxSlabs::dropDoomed(bool evict)
{
	std::vector<Slab *> drops;
	{
		std::lock_guard<std::mutex> hold(m_lock);
		if (evict) {
			while (!m_cached.empty())
				doom(m_cached.front());
			for (auto &entry : m_classes) {
				std::vector<Slab *> empty;
				for (Slab *slab : entry.second.slabs) {
					if (slab->used == 0)
						empty.push_back(slab);
				}
				for (Slab *slab : empty)
					doom(slab);
			}
		}
		// A slab whose piece the background is zeroing goes once the piece is done.
		for (auto it = m_doomed.begin(); it != m_doomed.end();) {
			if ((*it)->zeroing == 0) {
				drops.push_back(*it);
				m_all.erase(*it);
				it = m_doomed.erase(it);
			} else {
				++it;
			}
		}
	}
	for (Slab *slab : drops)
		drop(slab);
	return drops.size();
}

bool RgfxSlabs::step()
{
	Slab *drops = nullptr;
	Slab *zeroIn = nullptr;
	uint64_t at = 0, length = 0;
	{
		std::lock_guard<std::mutex> hold(m_lock);
		// Cached slabs unused too long go with the doomed ones.
		const uint64_t now = nowMs();
		while (!m_cached.empty() && now - m_cached.front()->idleSince >= kCacheIdleMs)
			doom(m_cached.front());
		if (m_emptyDue != 0 && now >= m_emptyDue)
			decayEmpty(now);
		for (auto it = m_doomed.begin(); it != m_doomed.end(); ++it) {
			if ((*it)->zeroing == 0) {
				drops = *it;
				m_all.erase(drops);
				m_doomed.erase(it);
				break;
			}
		}
		if (drops == nullptr) {
			while (!m_dirtySlabs.empty()) {
				Slab *slab = *m_dirtySlabs.begin();
				if (slab->dirty.bytes == 0) {
					m_dirtySlabs.erase(slab);
					continue;
				}
				// Every dirty byte is a free one: the piece leaves the free runs while zeroed.
				auto first = slab->dirty.runs.begin();
				at = first->first;
				length = std::min(first->second, kZeroPiece);
				slab->dirty.cut(at, length, nullptr);
				if (slab->dirty.bytes == 0)
					m_dirtySlabs.erase(slab);
				if (!slab->free.takeAt(at, length))
					continue; // not free: a buffer's, which its create zeroed
				slab->zeroing++;
				zeroIn = slab;
				break;
			}
		}
	}
	if (drops != nullptr) {
		drop(drops);
		return true;
	}
	if (zeroIn == nullptr)
		return false;
	std::memset(zeroIn->memory.bytes + at, 0, length);
	std::lock_guard<std::mutex> hold(m_lock);
	zeroIn->free.add(at, length);
	zeroIn->zeroing--;
	if (!zeroIn->dedicated && !zeroIn->doomed)
		classOf(zeroIn->options).open.insert(zeroIn);
	return true;
}

void RgfxSlabs::run()
{
#if defined(__APPLE__)
	// Behind the app's work: a create zeroes what the thread has not reached.
	pthread_set_qos_class_self_np(QOS_CLASS_UTILITY, 0);
	pthread_setname_np("com.reims.rgfx.slabs");
#endif
	for (;;) {
		while (step()) {
			std::lock_guard<std::mutex> hold(m_lock);
			if (m_stop)
				return;
		}
		std::unique_lock<std::mutex> hold(m_lock);
		if (m_stop)
			return;
		const bool work = !m_doomed.empty() || !m_dirtySlabs.empty();
		if (work)
			continue;
		const int64_t left = idleWait(nowMs());
		if (left < 0)
			m_wake.wait(hold);
		else
			m_wake.wait_for(hold, std::chrono::milliseconds(left + 1));
		if (m_stop)
			return;
	}
}

void RgfxSlabs::decayEmpty(uint64_t now)
{
	std::vector<Slab *> idle;
	uint64_t due = 0;
	for (auto &entry : m_classes) {
		for (Slab *slab : entry.second.slabs) {
			if (slab->used != 0 || slab->doomed)
				continue;
			const uint64_t at = slab->idleSince + kCacheIdleMs;
			if (now >= at)
				idle.push_back(slab);
			else if (due == 0 || at < due)
				due = at;
		}
	}
	for (Slab *slab : idle)
		doom(slab);
	m_emptyDue = due;
}

int64_t RgfxSlabs::idleWait(uint64_t now) const
{
	uint64_t due = m_emptyDue;
	if (!m_cached.empty()) {
		const uint64_t at = m_cached.front()->idleSince + kCacheIdleMs;
		if (due == 0 || at < due)
			due = at;
	}
	if (due == 0)
		return -1;
	return due > now ? static_cast<int64_t>(due - now) : 0;
}

RgfxSlabs::Stats RgfxSlabs::stats()
{
	std::lock_guard<std::mutex> hold(m_lock);
	Stats out;
	for (const Slab *slab : m_all) {
		out.slabs++;
		out.slabBytes += slab->memory.size;
		out.dirtyBytes += slab->dirty.bytes;
		out.usedBytes += slab->used;
		if (!slab->dedicated && slab->used == 0 && !slab->doomed)
			out.emptySlabs++;
		if (m_first != nullptr && slab->source == m_first)
			out.firstSlabs++;
	}
	out.cachedSlabs = m_cached.size();
	out.cachedBytes = m_cachedBytes;
	out.dooming = m_doomed.size();
	return out;
}
