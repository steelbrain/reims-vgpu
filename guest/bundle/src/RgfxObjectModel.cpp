/*
 * RgfxIdSpace: the device context's object-id, resource-id and GPU-address
 * number lines (RgfxObjectModel.hpp).
 */

#include "RgfxObjectModel.hpp"

#include <algorithm>
#include <iterator>

#include "RgfxLog.hpp"

namespace {
uint64_t alignUp(uint64_t value, uint64_t align)
{
	return (value + (align - 1)) & ~(align - 1);
}

// A pick of `span` bytes at a multiple of `align` from `origin(at)`, the address a free range at
// `at` is aligned from (at or below it), from the free ranges `free` (by start), next fit: the
// range the cursor lies in, from the cursor, and those after it; then round from the first. Its
// start, its span taken out of `free` and the cursor moved past it; 0 where none fits.
template <typename Origin>
uint64_t pickFrom(std::map<uint64_t, uint64_t> &free, uint64_t &cursor, uint64_t span,
                  uint64_t align, Origin origin)
{
	// Where `span` bytes from `from` on fit the free range at `at` of `length` bytes.
	auto fit = [&](uint64_t at, uint64_t length, uint64_t from, uint64_t *start) {
		const uint64_t base = origin(at);
		const uint64_t s = base + alignUp(std::max(at, from) - base, align);
		if (s < at || s - at >= length || length - (s - at) < span)
			return false;
		*start = s;
		return true;
	};
	auto it = free.upper_bound(cursor);
	if (it != free.begin()) {
		auto before = std::prev(it);
		if (before->first + before->second > cursor)
			it = before;
	}
	for (int pass = 0; pass < 2; pass++) {
		for (; it != free.end(); ++it) {
			uint64_t start = 0;
			if (!fit(it->first, it->second, pass == 0 ? cursor : 0, &start))
				continue;
			const uint64_t at = it->first, end = it->first + it->second;
			free.erase(it);
			if (start > at)
				free[at] = start - at;
			if (start + span < end)
				free[start + span] = end - (start + span);
			cursor = start + span;
			return start;
		}
		it = free.begin();
	}
	return 0;
}

// Gives the `length` bytes at `start` back to the free ranges `free`, joined with those either
// side of it that lie in [lo, hi).
void giveTo(std::map<uint64_t, uint64_t> &free, uint64_t start, uint64_t length, uint64_t lo,
            uint64_t hi)
{
	auto after = free.lower_bound(start);
	if (after != free.begin()) {
		auto before = std::prev(after);
		if (before->first >= lo && before->first + before->second == start) {
			start = before->first;
			length += before->second;
			free.erase(before);
		}
	}
	if (after != free.end() && after->first == start + length && after->first < hi) {
		length += after->second;
		free.erase(after);
	}
	free[start] = length;
}
} // namespace

RgfxIdSpace::RgfxIdSpace(uint64_t address_base, uint64_t address_span) : m_nextObjId(1)
{
	resetAddresses(address_base, address_span);
}

RgfxObjId RgfxIdSpace::allocObjId()
{
	// Relaxed is enough: the only invariant is uniqueness, and the id reaches
	// the host inside a ring message whose publish carries its own ordering.
	return m_nextObjId.fetch_add(1, std::memory_order_relaxed);
}

// A small chunk's free ids fit its 16-bit mask.
static_assert(RGFX_RESOURCE_ID_SMALL_CHUNK == 16, "a small chunk is 16 ids");
static_assert(RGFX_RESOURCE_ID_CHUNK % RGFX_RESOURCE_ID_SMALL_CHUNK == 0,
              "a chunk is whole small chunks");

uint64_t RgfxIdSpace::allocResourceId()
{
	std::lock_guard<std::mutex> hold(m_idsLock);
	for (uint32_t k : m_chunks) {
		if (m_freeIds[k] != 0) {
			const unsigned bit = static_cast<unsigned>(__builtin_ctz(m_freeIds[k]));
			m_freeIds[k] = static_cast<uint16_t>(m_freeIds[k] & (m_freeIds[k] - 1));
			return 1 + uint64_t{k} * RGFX_RESOURCE_ID_SMALL_CHUNK + bit;
		}
	}
	// Every id the process holds is in use: one more chunk. The source is called
	// with the lock held, so two threads that run out at once take one chunk.
	const uint64_t pool = uint64_t{RGFX_RESOURCE_ID_SMALL_CHUNKS} * RGFX_RESOURCE_ID_SMALL_CHUNK;
	uint64_t size = m_chunkIds;
	uint64_t first = 0;
	if (m_chunkSource != nullptr) {
		uint64_t granted = 0;
		if (!m_chunkSource(m_chunkSourceContext, &first, &granted))
			return 0;
		// The chunk holds the ids its grant says, whatever size this process took the
		// device's features to give: a process whose features query a sandbox refused took
		// chunks of 64 where the kext granted 16, and named ids no chunk of its held
		// (measured: Safari's WebContent on macOS 12.7.6, BadResourceId at the host).
		if (granted != 0) {
			if (granted % RGFX_RESOURCE_ID_SMALL_CHUNK != 0 || granted > RGFX_RESOURCE_ID_CHUNK) {
				os_log_error(rgfxLog(),
				             "RGFX: a resource-id chunk of %llu ids from the kext is no chunk; "
				             "no texture",
				             (unsigned long long)granted);
				return 0;
			}
			size = granted;
		}
	} else {
		if ((uint64_t{m_nextLocalChunk} + 1) * size > pool)
			return 0;
		first = 1 + uint64_t{m_nextLocalChunk++} * size;
	}
	// A chunk the pool does not hold, or one with an id held already, is no chunk.
	if (first == 0 || (first - 1) % size != 0 || first - 1 > pool - size)
		return 0;
	const uint32_t k0 = static_cast<uint32_t>((first - 1) / RGFX_RESOURCE_ID_SMALL_CHUNK);
	const uint32_t n = static_cast<uint32_t>(size / RGFX_RESOURCE_ID_SMALL_CHUNK);
	for (uint32_t i = 0; i < n; i++) {
		if (m_heldChunks[k0 + i])
			return 0;
	}
	for (uint32_t i = 0; i < n; i++) {
		m_heldChunks[k0 + i] = true;
		m_chunks.push_back(k0 + i);
		m_freeIds[k0 + i] = 0xffff;
	}
	// Every id but the first, which is the caller's.
	m_freeIds[k0] = 0xfffe;
	return first;
}

void RgfxIdSpace::freeResourceId(uint64_t id)
{
	if (id == 0)
		return;
	const uint64_t k = (id - 1) / RGFX_RESOURCE_ID_SMALL_CHUNK;
	std::lock_guard<std::mutex> hold(m_idsLock);
	if (k < RGFX_RESOURCE_ID_SMALL_CHUNKS && m_heldChunks[k])
		m_freeIds[k] = static_cast<uint16_t>(
		    m_freeIds[k] | (1u << ((id - 1) % RGFX_RESOURCE_ID_SMALL_CHUNK)));
}

void RgfxIdSpace::setResourceIdSource(ResourceIdChunkFn source, void *context, uint64_t chunk_ids)
{
	std::lock_guard<std::mutex> hold(m_idsLock);
	m_chunkSource = source;
	m_chunkSourceContext = context;
	// A size that is not whole small chunks, or more than a chunk, is a chunk's.
	const bool whole = chunk_ids != 0 && chunk_ids % RGFX_RESOURCE_ID_SMALL_CHUNK == 0 &&
	                   chunk_ids <= RGFX_RESOURCE_ID_CHUNK;
	m_chunkIds = whole ? chunk_ids : RGFX_RESOURCE_ID_CHUNK;
}

uint64_t RgfxIdSpace::addressSpan(uint64_t size)
{
	const uint64_t pages = (std::max<uint64_t>(size, 1) + kAddressPage - 1) / kAddressPage;
	return alignUp((pages + 1) * kAddressPage, kAddressGranule);
}

void RgfxIdSpace::resetAddresses(uint64_t base, uint64_t span)
{
	m_freeRanges.clear();
	m_livePicks.clear();
	const uint64_t start = alignUp(base, kAddressGranule);
	const uint64_t end = base + span;
	if (start < end)
		m_freeRanges[start] = end - start;
	m_addressCursor = start;
}

uint64_t RgfxIdSpace::allocAddress(uint64_t size, uint64_t align)
{
	align = std::max(align, kAddressGranule);
	const uint64_t span = addressSpan(size);
	std::lock_guard<std::mutex> hold(m_addressLock);
	auto absolute = [](uint64_t) -> uint64_t { return 0; };
	if (const uint64_t start = pickFrom(m_freeRanges, m_addressCursor, span, align, absolute)) {
		m_livePicks[start] = span;
		return start;
	}
	if (!m_addressesFull) {
		m_addressesFull = true;
		uint64_t live = 0;
		for (const auto &pick : m_livePicks)
			live += pick.second;
		os_log_error(rgfxLog(),
		             "RGFX: no GPU address left for %llu bytes: %llu MiB of the context's range "
		             "in %llu picks",
		             (unsigned long long)size, (unsigned long long)(live >> 20),
		             (unsigned long long)m_livePicks.size());
	}
	return 0;
}

void RgfxIdSpace::freeAddress(uint64_t address)
{
	uint64_t giveAt = 0, giveLength = 0;
	{
		std::lock_guard<std::mutex> hold(m_addressLock);
		auto live = m_livePicks.find(address);
		if (live != m_livePicks.end()) {
			const uint64_t length = live->second;
			m_livePicks.erase(live);
			giveTo(m_freeRanges, address, length, 0, UINT64_MAX);
			return;
		}
		auto pick = m_privatePicks.find(address);
		if (pick == m_privatePicks.end())
			return;
		const uint64_t length = pick->second;
		m_privatePicks.erase(pick);
		auto block = privateBlockOf(address);
		if (block == m_privateBlocks.end())
			return;
		const uint64_t lo = block->first, hi = block->first + block->second.length;
		if (block->second.own) {
			// One pick's own block, which holds no other pick and no free range: given back
			// with its pick.
			m_privateBlocks.erase(block);
			giveAt = lo;
			giveLength = hi - lo;
		} else {
			giveTo(m_privateFree, address, length, lo, hi);
			if (--block->second.picks != 0)
				return;
			// Empty: kept where it is the only empty block, for the next picks; else given
			// back, its one free range taken out first so no pick lands in it.
			for (const auto &other : m_privateBlocks)
				if (other.first != lo && other.second.picks == 0) {
					m_privateFree.erase(lo);
					m_privateBlocks.erase(block);
					giveAt = lo;
					giveLength = hi - lo;
					break;
				}
		}
	}
	// With no lock held: a call into the kext, which first posts the destroys the process holds,
	// the last object's in the block among them.
	if (giveAt != 0)
		m_privateGive(m_privateContext, giveAt, giveLength);
}

std::map<uint64_t, RgfxIdSpace::PrivateBlock>::iterator RgfxIdSpace::privateBlockOf(
    uint64_t address)
{
	auto it = m_privateBlocks.upper_bound(address);
	if (it == m_privateBlocks.begin())
		return m_privateBlocks.end();
	--it;
	return address - it->first < it->second.length ? it : m_privateBlocks.end();
}

void RgfxIdSpace::setPrivateBlockSource(PrivateTakeFn take, PrivateGiveFn give, void *context)
{
	std::lock_guard<std::mutex> hold(m_addressLock);
	m_privateTake = take;
	m_privateGive = give;
	m_privateContext = context;
}

uint64_t RgfxIdSpace::allocPrivateAddress(uint64_t size, uint64_t asked, uint64_t blockAlign)
{
	if (m_privateTake == nullptr)
		return allocAddress(size, asked);
	const uint64_t align = std::max({asked, blockAlign, kAddressGranule});
	const uint64_t span = addressSpan(size);
	// The address a free range is aligned from: 0, or with a `blockAlign` its block's first
	// byte. Caller holds m_addressLock.
	auto origin = [&](uint64_t at) -> uint64_t {
		if (blockAlign == 0)
			return 0;
		const auto block = privateBlockOf(at);
		return block == m_privateBlocks.end() ? 0 : block->first;
	};
	auto pick = [&]() -> uint64_t {
		const uint64_t start = pickFrom(m_privateFree, m_privateCursor, span, align, origin);
		if (start != 0) {
			m_privatePicks[start] = span;
			privateBlockOf(start)->second.picks++;
		}
		return start;
	};
	{
		std::lock_guard<std::mutex> hold(m_addressLock);
		if (const uint64_t start = pick())
			return start;
	}
	// No block has room: a new one, taken with no lock held (a call into the kext, which waits for
	// the host's answer), with room for the span at its alignment wherever the block starts:
	// kPrivateBlockBytes, or where the span and its alignment are more, the pick's own block.
	uint64_t at = 0, length = 0;
	const bool sized = span <= RGFX_PRIVATE_TAKE_MAX && align <= RGFX_PRIVATE_TAKE_MAX - span;
	const uint64_t want =
	    sized ? std::max(kPrivateBlockBytes, alignUp(span + align, RGFX_PRIVATE_BLOCK_ALIGN)) : 0;
	if (!sized || want > RGFX_PRIVATE_TAKE_MAX ||
	    !m_privateTake(m_privateContext, want, &at, &length) || at == 0 || length < want ||
	    at > UINT64_MAX - length) {
		if (!m_privateRefused.exchange(true))
			os_log_error(rgfxLog(),
			             "RGFX: no Private block of %llu bytes for a pick of %llu bytes: "
			             "Private objects the context's blocks have no room for lie in its own "
			             "range",
			             (unsigned long long)want, (unsigned long long)size);
		return allocAddress(size, asked);
	}
	{
		std::lock_guard<std::mutex> hold(m_addressLock);
		if (want > kPrivateBlockBytes) {
			// The pick's own block: the pick at its first aligned byte, which the block's length
			// leaves room for wherever it starts, and no free range for other picks.
			const uint64_t base = blockAlign == 0 ? 0 : at;
			const uint64_t start = base + alignUp(at - base, align);
			m_privateBlocks[at] = PrivateBlock{length, 1, true};
			m_privatePicks[start] = span;
			return start;
		}
		m_privateBlocks[at] = PrivateBlock{length, 0, false};
		m_privateFree[at] = length;
		if (const uint64_t start = pick())
			return start;
	}
	// Other threads' picks took the room first: the context's range, as for a refusal.
	return allocAddress(size, asked);
}

size_t RgfxIdSpace::privateBlocksHeld() const
{
	std::lock_guard<std::mutex> hold(m_addressLock);
	return m_privateBlocks.size();
}

size_t RgfxIdSpace::privatePicksLive() const
{
	std::lock_guard<std::mutex> hold(m_addressLock);
	return m_privatePicks.size();
}

void RgfxIdSpace::setBases(uint64_t obj_id_base, uint64_t address_base, uint64_t address_span)
{
	// 0 names no object, so a context's first id is its base + 1.
	m_nextObjId.store(obj_id_base + 1, std::memory_order_relaxed);
	std::lock_guard<std::mutex> hold(m_addressLock);
	resetAddresses(address_base, address_span);
}
