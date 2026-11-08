/*
 * RgfxIcbSlotTable: an indirect command buffer's CPU encodes as slots in a block of the process's
 * region, where the host took ICB_SLOTS (crates/reims-vgpu-graphics/src/icb.rs, "Slots"). Each command
 * has a slot of its own, `stride` bytes from the next; a setter writes its field there and sets
 * the field's bit in the slot's mask with an atomic OR, as a GPU's driver writes a command's
 * record into memory the GPU reads, and posts nothing. The flush (RgfxSerializer::flushIcbWrites)
 * names the commands written since the last one, as runs of 64-command chunks, in one ICB_SLOTS;
 * the host swaps each mask for 0 as it takes a slot, and carries out the bits it took.
 *
 * What is written since a flush is kept as one bit a chunk of 64 commands (`m_dirty`), set with an
 * atomic OR on a chunk's first write, and a `pending` flag that puts the table on the serialiser's
 * list once a flush cycle (RgfxSerializer::icbSlotsPending). The order makes a write and a flush
 * on another thread meet: a setter ORs the mask, then the chunk's bit, then reads `pending`; a
 * flush clears `pending`, then swaps the chunks' words for 0. A chunk bit set after the swap finds
 * `pending` clear, so the table goes on the list again, and the next flush names it.
 *
 * A buffer takes slots only where its commands are of one kind, its bindings fit a slot (at most
 * RGFX_ICB_SLOT_BINDINGS of each kind), and its block fits the caps (kMaxBytes for a buffer,
 * RgfxSerializer's for the process). A call a slot cannot hold (a binding past the slot's count)
 * turns the buffer to records for good (toRecords): its written slots are posted first, so the
 * host carries them out before the records that follow.
 *
 * Our own code; it depends only on the committed wire header.
 */

#ifndef RGFX_ICB_SLOT_TABLE_HPP_
#define RGFX_ICB_SLOT_TABLE_HPP_

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

#include "reims_graphics_wire.h"

class RgfxSerializer;

class RgfxIcbSlotTable {
public:
	/// The most bytes one buffer's slots take: a few such buffers leave the process's Shared
	/// buffers their room in the region (an app has about 400 MiB of it). 16384 commands of two
	/// vertex buffers and two fragment buffers take 3.2 MiB.
	static constexpr uint64_t kMaxBytes = 8ull << 20;

	/// Commands a dirty bit covers.
	static constexpr uint32_t kChunk = 64;

	/// The bytes from one slot to the next for a buffer of `kind` (RGFX_ICB_SLOTS_RENDER or
	/// RGFX_ICB_SLOTS_COMPUTE) with `a` bindings of the first kind and `b` of the second, as the
	/// host's slot_bytes reads them; 0 where none fits.
	static uint32_t strideFor(uint16_t kind, uint32_t a, uint32_t b);

	/// The slots of indirect command buffer `icb`, of `count` commands of `kind` with bindings
	/// (`a`, `b`), at `base`, region offset `offset`, `stride` apart, zeroed by the caller.
	/// Flushes go through `serializer`.
	RgfxIcbSlotTable(RgfxSerializer *serializer, uint64_t icb, uint16_t kind, uint32_t a, uint32_t b,
	             uint32_t count, uint32_t stride, volatile uint8_t *base, uint64_t offset);

	uint64_t icb() const { return m_icb; }
	uint16_t kind() const { return m_kind; }
	uint32_t count() const { return m_count; }
	uint32_t stride() const { return m_stride; }
	uint64_t offset() const { return m_offset; }
	uint64_t bytes() const { return uint64_t(m_count) * m_stride; }

	/// Whether setters write slots: false once the buffer turned to records.
	bool active() const { return m_active.load(std::memory_order_relaxed); }

	/// Command `i`'s slot.
	uint8_t *slot(uint32_t i) const
	{
		return const_cast<uint8_t *>(m_base) + uint64_t(i) * m_stride;
	}

	// --- setters: each writes its field, sets its bit and marks the command written --------

	/// `setRenderPipelineState:` or `setComputePipelineState:`.
	void setPipeline(uint32_t i, uint64_t pipeline)
	{
		uint8_t *s = slot(i);
		store64(s + offsetof(RgfxIcbRenderSlot, pipeline), pipeline);
		set(s, i, RGFX_ICB_SLOT_PIPELINE);
	}

	/// A binding of the first kind (a vertex or kernel buffer) `index`, its attribute stride where
	/// `strided`. False, the buffer turned to records, where the slot holds no such binding.
	bool setBuffer(uint32_t i, uint64_t index, uint64_t buffer, uint64_t offset, uint64_t stride,
	               bool strided)
	{
		if (index >= m_a)
			return toRecords();
		uint8_t *s = slot(i);
		uint8_t *b = s + m_fixed + index * sizeof(RgfxIcbSlotBuffer);
		store64(b, buffer);
		store64(b + 8, offset);
		store64(b + 16, stride);
		uint32_t *bits = reinterpret_cast<uint32_t *>(s + offsetof(RgfxIcbRenderSlot, strided));
		const uint32_t bit = 1u << index, was = __atomic_load_n(bits, __ATOMIC_RELAXED);
		__atomic_store_n(bits, strided ? (was | bit) : (was & ~bit), __ATOMIC_RELAXED);
		set(s, i, RGFX_ICB_SLOT_BUFFER_0 << index);
		return true;
	}

	/// A render command's fragment buffer `index`. False, as setBuffer.
	bool setFragmentBuffer(uint32_t i, uint64_t index, uint64_t buffer, uint64_t offset)
	{
		if (index >= m_b)
			return toRecords();
		uint8_t *s = slot(i);
		uint8_t *b = s + m_fixed + (m_a + index) * sizeof(RgfxIcbSlotBuffer);
		store64(b, buffer);
		store64(b + 8, offset);
		setB(s, i, 1ull << index);
		return true;
	}

	/// A compute command's threadgroup memory length `index`. False, as setBuffer.
	bool setThreadgroupMemory(uint32_t i, uint64_t index, uint32_t length)
	{
		if (index >= m_b)
			return toRecords();
		uint8_t *s = slot(i);
		uint8_t *l = s + m_fixed + m_a * sizeof(RgfxIcbSlotBuffer) + index * 4;
		__atomic_store_n(reinterpret_cast<uint32_t *>(l), length, __ATOMIC_RELAXED);
		setB(s, i, 1ull << index);
		return true;
	}

	/// The draw or the dispatch: `op` and its record's body, `size` bytes (zeros past it, in a
	/// render slot).
	void setAction(uint32_t i, uint16_t op, const void *body, size_t size)
	{
		uint8_t *s = slot(i);
		__atomic_store_n(reinterpret_cast<uint16_t *>(s + offsetof(RgfxIcbRenderSlot, op)), op,
		                 __ATOMIC_RELAXED);
		uint8_t *at = s + offsetof(RgfxIcbRenderSlot, action);
		const size_t room = m_kind == RGFX_ICB_SLOTS_RENDER ? 72 : sizeof(RgfxIcbDispatch);
		std::memcpy(at, body, size < room ? size : room);
		if (size < room)
			std::memset(at + size, 0, room - size);
		set(s, i, RGFX_ICB_SLOT_ACTION);
	}

	/// `setBarrier` (`on`) or `clearBarrier`.
	void setBarrier(uint32_t i, bool on)
	{
		uint8_t *s = slot(i);
		__atomic_store_n(reinterpret_cast<uint16_t *>(s + offsetof(RgfxIcbRenderSlot, barrier)),
		                 uint16_t(on ? 1 : 0), __ATOMIC_RELAXED);
		set(s, i, RGFX_ICB_SLOT_BARRIER);
	}

	/// A compute command's `setStageInRegion:`.
	void setStageIn(uint32_t i, const RgfxIcbRegion &region)
	{
		uint8_t *s = slot(i);
		std::memcpy(s + offsetof(RgfxIcbComputeSlot, stage_in), &region, sizeof(region));
		set(s, i, RGFX_ICB_SLOT_STAGE_IN);
	}

	/// A compute command's `setImageblockWidth:height:`.
	void setImageblock(uint32_t i, const RgfxIcbImageblock &imageblock)
	{
		uint8_t *s = slot(i);
		std::memcpy(s + offsetof(RgfxIcbComputeSlot, imageblock), &imageblock, sizeof(imageblock));
		set(s, i, RGFX_ICB_SLOT_IMAGEBLOCK);
	}

	/// The command's `reset`: what was set before it goes, and the host resets the command
	/// before it carries out what is set after.
	void reset(uint32_t i)
	{
		uint8_t *s = slot(i);
		__atomic_store_n(reinterpret_cast<uint64_t *>(s + 8), uint64_t(0), __ATOMIC_RELAXED);
		__atomic_store_n(reinterpret_cast<uint64_t *>(s), RGFX_ICB_SLOT_RESET, __ATOMIC_SEQ_CST);
		touch(i);
	}

	/// The buffer's `resetWithRange:` of commands `first..first + n` (inside the size): each slot
	/// holds the range reset alone, which the host carries out as one call for a run of them.
	void resetRange(uint32_t first, uint32_t n);

	/// Turns the buffer to records for good, posting its written slots first. False always, for
	/// the setters' callers, which then send the record.
	bool toRecords();

	// --- the flush --------------------------------------------------------------------------

	/// Whether the table is on the serialiser's list (icbSlotsPending): cleared by the flush that
	/// takes it, before it swaps the chunks' bits.
	std::atomic<bool> pending{false};

	/// The runs of the chunks written since the last take, each the commands of consecutive
	/// chunks, appended to `runs`; the chunks' bits are swapped for 0.
	void takeRuns(std::vector<RgfxIcbSlotRun> &runs);

private:
	static void store64(uint8_t *at, uint64_t v)
	{
		__atomic_store_n(reinterpret_cast<uint64_t *>(at), v, __ATOMIC_RELAXED);
	}

	// The field's bit `bit` of mask word `word` (0 or 1) of command `i`'s slot `s`, set after
	// the field, then the command marked written.
	void set(uint8_t *s, uint32_t i, uint64_t bit)
	{
		__atomic_fetch_or(reinterpret_cast<uint64_t *>(s), bit, __ATOMIC_SEQ_CST);
		touch(i);
	}
	void setB(uint8_t *s, uint32_t i, uint64_t bit)
	{
		__atomic_fetch_or(reinterpret_cast<uint64_t *>(s + 8), bit, __ATOMIC_SEQ_CST);
		touch(i);
	}

	// Command `i` was written: its chunk's bit set where clear, and the table put on the list
	// where it is not.
	void touch(uint32_t i)
	{
		const uint32_t chunk = i / kChunk;
		std::atomic<uint64_t> &word = m_dirty[chunk / 64];
		const uint64_t bit = 1ull << (chunk % 64);
		if ((word.load(std::memory_order_seq_cst) & bit) == 0)
			word.fetch_or(bit, std::memory_order_seq_cst);
		if (!pending.load(std::memory_order_seq_cst))
			enqueue();
	}
	void enqueue();

	RgfxSerializer *m_serializer;
	uint64_t m_icb;
	uint16_t m_kind;
	uint32_t m_a, m_b;
	uint32_t m_count;
	uint32_t m_stride;
	uint32_t m_fixed; // the fixed part's bytes, where the bindings start
	volatile uint8_t *m_base;
	uint64_t m_offset;
	std::atomic<bool> m_active{true};
	std::unique_ptr<std::atomic<uint64_t>[]> m_dirty; // a bit a chunk, 64 chunks a word
	uint32_t m_dirtyWords;
};

#endif // RGFX_ICB_SLOT_TABLE_HPP_
