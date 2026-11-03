/*
 * RgfxSlabs: the slab allocator of Shared and Managed buffers (RgfxSlabs.hpp), and its source
 * over the device region. Our own C++.
 */

#include "RgfxSlabs.hpp"

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
                                           bool managedAtOffset)
    : m_region(region), m_serializer(serializer), m_managedAtOffset(managedAtOffset)
{
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
	m_serializer.destroy(heap.objId);
	m_serializer.releaseAddress(heap.gpuAddress);
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

uint64_t RgfxSlabs::nowMs() const
{
	return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
	                                 std::chrono::steady_clock::now().time_since_epoch())
	                                 .count());
}

RgfxSlabs::Class &RgfxSlabs::classOf(uint32_t options)
{
	return m_classes[options];
}

RgfxSlabs::Slab *RgfxSlabs::makeSlab(uint64_t size, uint32_t options, bool dedicated,
                                     bool *refused)
{
	RgfxSlabSource::Memory memory;
	if (!m_source.take(size, &memory))
		return nullptr;
	RgfxSlabSource::Heap heap;
	if (m_source.heaped(options) && !m_source.makeHeap(memory, options, &heap)) {
		m_source.give(memory);
		*refused = true; // the device never got the heap: no smaller slab gets one either
		return nullptr;
	}
	Slab *slab = new Slab;
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
	if (length == 0 || length > UINT64_MAX - m_source.unit())
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
			const uint64_t need = roundUp(size, unit);
			Slab *best = nullptr;
			for (Slab *slab : m_cached) {
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
		// A new slab; once the source refused one, the process's own room given back and the slab
		// asked again; then, for a small buffer, slabs half as large down to the buffer's size.
		const uint64_t least = roundUp(size, unit);
		uint64_t ask = dedicated ? least : want;
		bool refused = false;
		Slab *made = makeSlab(ask, options, dedicated, &refused);
		if (made == nullptr && !refused) {
			dropDoomed(true);
			made = makeSlab(ask, options, dedicated, &refused);
		}
		while (made == nullptr && !refused && !dedicated && ask > least) {
			ask = std::max(roundUp(ask / 2, unit), least);
			made = makeSlab(ask, options, dedicated, &refused);
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
	const Slab *slab = static_cast<const Slab *>(range.slab);
	return m_source.makeBuffer(slab->memory, slab->heap, range.offset, length, options, out);
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
	bool dropNow = false;
	{
		std::lock_guard<std::mutex> hold(m_lock);
		slab->free.add(range.offset, range.size);
		slab->dirty.add(range.offset, range.size);
		slab->used -= range.size;
		m_dirtySlabs.insert(slab);
		if (slab->dedicated) {
			// Cached for a later buffer, the oldest let go past a sixteenth of the source.
			slab->idleSince = nowMs();
			m_cached.push_back(slab);
			m_cachedBytes += slab->memory.size;
			const uint64_t cap = m_source.capacity() / 16;
			while (m_cachedBytes > cap && !m_cached.empty())
				doom(m_cached.front());
		} else {
			Class &cls = classOf(slab->options);
			cls.open.insert(slab);
			if (slab->used == 0 && ++cls.empty > kEmptyKept)
				doom(slab);
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
		m_source.dropHeap(slab->heap);
	m_source.give(slab->memory);
	delete slab;
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
		if (m_cached.empty()) {
			m_wake.wait(hold);
		} else {
			const uint64_t idle = nowMs() - m_cached.front()->idleSince;
			const uint64_t left = idle >= kCacheIdleMs ? 0 : kCacheIdleMs - idle;
			m_wake.wait_for(hold, std::chrono::milliseconds(left + 1));
		}
		if (m_stop)
			return;
	}
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
	}
	out.cachedSlabs = m_cached.size();
	out.cachedBytes = m_cachedBytes;
	out.dooming = m_doomed.size();
	return out;
}
