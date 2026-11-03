/*
 * RgfxIcbSlotTable: see RgfxIcbSlotTable.hpp. Our own code.
 */

#include "RgfxIcbSlotTable.hpp"

#include <algorithm>

#include "RgfxSerializer.hpp"

// The setters write both kinds of slot through the render slot's offsets: the two share their
// first 32 bytes, and a compute slot's dispatch lies where a render slot's draw does.
static_assert(offsetof(RgfxIcbRenderSlot, mask_b) == offsetof(RgfxIcbComputeSlot, mask_b), "");
static_assert(offsetof(RgfxIcbRenderSlot, pipeline) == offsetof(RgfxIcbComputeSlot, pipeline), "");
static_assert(offsetof(RgfxIcbRenderSlot, op) == offsetof(RgfxIcbComputeSlot, op), "");
static_assert(offsetof(RgfxIcbRenderSlot, barrier) == offsetof(RgfxIcbComputeSlot, barrier), "");
static_assert(offsetof(RgfxIcbRenderSlot, strided) == offsetof(RgfxIcbComputeSlot, strided), "");
static_assert(offsetof(RgfxIcbRenderSlot, action) == offsetof(RgfxIcbComputeSlot, dispatch), "");
static_assert(sizeof(RgfxIcbRenderSlot::action) == 72, "the largest draw's body");
static_assert(sizeof(RgfxIcbDrawIndexedPatches) == 72, "the largest draw's body");

uint32_t RgfxIcbSlotTable::strideFor(uint16_t kind, uint32_t a, uint32_t b)
{
	if (a > RGFX_ICB_SLOT_BINDINGS || b > RGFX_ICB_SLOT_BINDINGS)
		return 0;
	const uint32_t buffer = sizeof(RgfxIcbSlotBuffer);
	switch (kind) {
	case RGFX_ICB_SLOTS_RENDER:
		return sizeof(RgfxIcbRenderSlot) + buffer * (a + b);
	case RGFX_ICB_SLOTS_COMPUTE:
		return sizeof(RgfxIcbComputeSlot) + buffer * a + (4 * b + 7) / 8 * 8;
	default:
		return 0;
	}
}

RgfxIcbSlotTable::RgfxIcbSlotTable(RgfxSerializer *serializer, uint64_t icb, uint16_t kind, uint32_t a,
                           uint32_t b, uint32_t count, uint32_t stride, volatile uint8_t *base,
                           uint64_t offset)
    : m_serializer(serializer), m_icb(icb), m_kind(kind), m_a(a), m_b(b), m_count(count),
      m_stride(stride),
      m_fixed(kind == RGFX_ICB_SLOTS_RENDER ? sizeof(RgfxIcbRenderSlot) : sizeof(RgfxIcbComputeSlot)),
      m_base(base), m_offset(offset)
{
	const uint32_t chunks = (count + kChunk - 1) / kChunk;
	m_dirtyWords = (chunks + 63) / 64;
	m_dirty.reset(new std::atomic<uint64_t>[m_dirtyWords]);
	for (uint32_t w = 0; w < m_dirtyWords; w++)
		m_dirty[w].store(0, std::memory_order_relaxed);
}

void RgfxIcbSlotTable::resetRange(uint32_t first, uint32_t n)
{
	if (n == 0)
		return;
	// The stores need no order among themselves: the fence orders them all before the chunks'
	// bits, as each setter's OR does its own.
	for (uint32_t i = first; i < first + n; i++) {
		uint8_t *s = slot(i);
		__atomic_store_n(reinterpret_cast<uint64_t *>(s + 8), uint64_t(0), __ATOMIC_RELAXED);
		__atomic_store_n(reinterpret_cast<uint64_t *>(s), RGFX_ICB_SLOT_RESET_RANGE,
		                 __ATOMIC_RELAXED);
	}
	std::atomic_thread_fence(std::memory_order_seq_cst);
	const uint32_t lastChunk = (first + n - 1) / kChunk;
	for (uint32_t chunk = first / kChunk; chunk <= lastChunk;) {
		const uint32_t word = chunk / 64, bit = chunk % 64;
		const uint32_t take = std::min<uint32_t>(64 - bit, lastChunk - chunk + 1);
		const uint64_t bits = (take == 64 ? ~0ull : ((1ull << take) - 1)) << bit;
		m_dirty[word].fetch_or(bits, std::memory_order_seq_cst);
		chunk += take;
	}
	if (!pending.load(std::memory_order_seq_cst))
		enqueue();
}

bool RgfxIcbSlotTable::toRecords()
{
	if (m_active.exchange(false, std::memory_order_seq_cst))
		m_serializer->flushIcbSlots(this);
	return false;
}

void RgfxIcbSlotTable::enqueue()
{
	m_serializer->icbSlotsPending(this);
}

void RgfxIcbSlotTable::takeRuns(std::vector<RgfxIcbSlotRun> &runs)
{
	RgfxIcbSlotRun run = {};
	run.icb_obj_id = m_icb;
	run.slot_bytes = m_stride;
	run.kind = m_kind;
	run.bindings_a = static_cast<uint8_t>(m_a);
	run.bindings_b = static_cast<uint8_t>(m_b);
	bool open = false;
	// A run of chunks: closed at the first clear bit after a set one.
	auto close = [&](uint32_t endChunk) {
		const uint32_t end = std::min<uint64_t>(uint64_t(endChunk) * kChunk, m_count);
		run.command_count = end - run.first_command;
		runs.push_back(run);
		open = false;
	};
	for (uint32_t w = 0; w < m_dirtyWords; w++) {
		// Swapped whether or not it reads set: a load first could miss a setter's bit that a
		// cleared `pending` does not put back on the list.
		const uint64_t bits = m_dirty[w].exchange(0, std::memory_order_seq_cst);
		for (uint32_t bit = 0; bit < 64; bit++) {
			const uint32_t chunk = w * 64 + bit;
			const bool set = (bits >> bit) & 1;
			if (set && !open) {
				run.first_command = chunk * kChunk;
				run.region_offset = m_offset + uint64_t(run.first_command) * m_stride;
				open = true;
			} else if (!set && open) {
				close(chunk);
			}
			if (bits >> bit == 0 && !open)
				break;
		}
	}
	if (open)
		close(m_dirtyWords * 64);
}
