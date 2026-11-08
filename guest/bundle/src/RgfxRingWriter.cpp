/*
 * RgfxRingWriter: the guest producer's reserve-write-publish-doorbell step
 * (RgfxRingWriter.hpp). Mirrors the host consumer in
 * crates/reims-vgpu-graphics/src/ring.rs.
 */

#include "RgfxRingWriter.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <memory>
#include <os/log.h>
#include <thread>

namespace {

// The ring header fields, by their offsets in RgfxRingHeader.
constexpr uint32_t kProducer = __builtin_offsetof(RgfxRingHeader, producer);
constexpr uint32_t kConsumer = __builtin_offsetof(RgfxRingHeader, consumer);
constexpr uint32_t kConsumerIdle = __builtin_offsetof(RgfxRingHeader, consumer_idle);
constexpr uint32_t kProducerWaiting = __builtin_offsetof(RgfxRingHeader, producer_waiting);

// How long a post waits for the host's room record at most where a completion thread takes it,
// before it looks again: a backstop only.
constexpr uint32_t kRoomBackstopUs = 20000;

// How often a writer's room line is logged at most, while its posts sleep for room.
constexpr uint64_t kSleepsLogNs = 1000000000;

// A message's header is 16 bytes; sizes are a multiple of 8.
constexpr uint32_t kHeaderBytes = sizeof(RgfxMsgHeader);
constexpr uint32_t kMsgAlign = 8;

uint64_t load64(const volatile uint8_t *p)
{
	return __atomic_load_n(reinterpret_cast<const volatile uint64_t *>(p), __ATOMIC_ACQUIRE);
}

void store64(volatile uint8_t *p, uint64_t value)
{
	__atomic_store_n(reinterpret_cast<volatile uint64_t *>(p), value, __ATOMIC_RELEASE);
}

uint32_t load32(const volatile uint8_t *p)
{
	return __atomic_load_n(reinterpret_cast<const volatile uint32_t *>(p), __ATOMIC_ACQUIRE);
}

void store32(volatile uint8_t *p, uint32_t value)
{
	__atomic_store_n(reinterpret_cast<volatile uint32_t *>(p), value, __ATOMIC_SEQ_CST);
}

void fullFence()
{
	__atomic_thread_fence(__ATOMIC_SEQ_CST);
}

// Orders every store before it, write-combined ones among them, before every store after it:
// the ring's data and the bodies and chunks a message names may lie in write-combined parts of
// their blocks (RgfxRegion.hpp, WRITE-COMBINED PARTS), whose stores x86 does not keep in order
// with others, as a release store does for write-back memory.
void storeFence()
{
#if defined(__x86_64__) || defined(__i386__)
	__builtin_ia32_sfence();
#else
	__atomic_thread_fence(__ATOMIC_RELEASE);
#endif
}

// Copies `len` bytes to the ring at byte offset `at` (already within the data
// area, no wrap). A plain copy: storeFence orders it before the release store of
// `producer`, so the host reads it whole.
void writeBytes(volatile uint8_t *data, uint32_t at, const void *src, uint32_t len)
{
	std::memcpy(const_cast<uint8_t *>(data + at), src, len);
}

} // namespace

RgfxRingWriter::RgfxRingWriter(volatile uint8_t *ring_base, uint32_t data_size, uint32_t ring_id,
                               RgfxDoorbell doorbell, void *doorbell_context)
    : m_header(ring_base), m_data(ring_base + RGFX_RING_HEADER_SIZE), m_dataSize(data_size),
      m_mask(data_size - 1), m_ringId(ring_id), m_doorbell(doorbell),
      m_doorbellContext(doorbell_context), m_forward(nullptr), m_forwardContext(nullptr),
      m_produced(0), m_nextSeqno(1)
{
	// The rings the bundle writes itself, its command queues' submission rings, are its own:
	// each is fresh, in a block the kext handed the process, with the process its single
	// producer, so the writer starts from 0. The lifecycle goes through the kext instead (the
	// forwarding writer below), which posts it on the control ring.
}

RgfxRingWriter::RgfxRingWriter(RgfxForward forward, void *forward_context, uint32_t max_size)
    : m_header(nullptr), m_data(nullptr), m_dataSize(max_size), m_mask(0), m_ringId(0),
      m_doorbell(nullptr), m_doorbellContext(nullptr), m_forward(forward),
      m_forwardContext(forward_context), m_produced(0), m_nextSeqno(1)
{
}

RgfxRingWriter::~RgfxRingWriter()
{
	RgfxRingRoomSleeps seen;
	{
		std::lock_guard<std::mutex> hold(m_sleepsLock);
		if (!m_sleeps.unlogged())
			return;
		seen = m_sleeps;
	}
	logSleeps(seen);
}

void RgfxRingWriter::setDeferred(RgfxForwardDeferred post, RgfxAwaitAnswer await, void *context)
{
	m_deferredPost = post;
	m_await = await;
	m_deferredContext = context;
}

bool RgfxRingWriter::tryPostKnown(uint32_t type, const void *message, uint32_t size,
                                  uint64_t *out_seqno, bool *out_deferred)
{
	*out_deferred = false;
	if (!defers())
		return tryPost(type, message, size, out_seqno);
	if (size < kHeaderBytes || size % kMsgAlign != 0 || size > m_dataSize)
		return false;
	std::unique_ptr<uint8_t[]> buf(new uint8_t[size]);
	std::memcpy(buf.get(), message, size);
	RgfxMsgHeader header = { type, size, 0 };
	std::memcpy(buf.get(), &header, sizeof(header));
	storeFence();
	uint64_t seqno = 0;
	if (!m_deferredPost(m_deferredContext, buf.get(), size, &seqno, out_deferred))
		return false;
	if (out_seqno)
		*out_seqno = seqno;
	return true;
}

bool RgfxRingWriter::awaitAnswer(uint64_t seqno)
{
	return m_await != nullptr && m_await(m_deferredContext, seqno);
}

bool RgfxRingWriter::fits(uint32_t size) const
{
	if (size < kHeaderBytes || size % kMsgAlign != 0)
		return false;
	return m_forward ? size <= m_dataSize : size <= m_dataSize / 2;
}

bool RgfxRingWriter::post(uint32_t type, const void *message, uint32_t size, uint64_t *out_seqno,
                          const std::function<void()> &whileWaiting,
                          const std::function<void(uint64_t)> &beforePublish)
{
	if (!fits(size))
		return false;
	uint32_t sleepUs = 0;
	for (;;) {
		// The room records so far, read before the look: one that comes after it ends the wait.
		const uint64_t seen = m_room ? m_room->seen() : 0;
		if (tryPost(type, message, size, out_seqno, nullptr, beforePublish))
			return true;
		// Where the host posts room records, it is told a post waits, then the ring is looked
		// at once more: either this look finds what the host took, or the host's next take finds
		// the flag and posts a record.
		if (m_room) {
			store32(m_header + kProducerWaiting, 1);
			fullFence();
			if (tryPost(type, message, size, out_seqno, nullptr, beforePublish))
				return true;
		}
		// The host takes the ring on a thread of its own: make sure it is awake, then let it.
		if (m_doorbell && (sleepUs == 0 || load32(m_header + kConsumerIdle) == 1))
			m_doorbell(m_doorbellContext, m_ringId);
		if (whileWaiting)
			whileWaiting();
		sleepUs = sleepUs == 0 ? 20 : std::min<uint32_t>(sleepUs * 2, 1000);
		const auto from = std::chrono::steady_clock::now();
		bool byRecord = false;
		if (m_room)
			byRecord = m_room->waitPast(
			    seen, std::chrono::microseconds(whileWaiting ? sleepUs : kRoomBackstopUs));
		else
			std::this_thread::sleep_for(std::chrono::microseconds(sleepUs));
		noteSleep(from, byRecord);
	}
}

void RgfxRingWriter::noteSleep(std::chrono::steady_clock::time_point from, bool byRecord)
{
	const auto to = std::chrono::steady_clock::now();
	const auto ns = [](std::chrono::steady_clock::duration d) {
		return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(d).count());
	};
	RgfxRingRoomSleeps seen;
	{
		std::lock_guard<std::mutex> hold(m_sleepsLock);
		if (!m_sleeps.note(ns(to - from), byRecord, ns(to.time_since_epoch()), kSleepsLogNs))
			return;
		seen = m_sleeps;
	}
	logSleeps(seen);
}

void RgfxRingWriter::logSleeps(const RgfxRingRoomSleeps &seen) const
{
	os_log(OS_LOG_DEFAULT,
	       "RGFX: ring %u room: %llu sleeps (%llu ended by a SUBMIT_ROOM record), %llu us asleep "
	       "in all, %llu us at most",
	       m_ringId, seen.sleeps, seen.woken, seen.sleptNs / 1000, seen.mostNs / 1000);
}

bool RgfxRingWriter::tryPost(uint32_t type, const void *message, uint32_t size, uint64_t *out_seqno,
                             uint32_t *out_status,
                             const std::function<void(uint64_t)> &beforePublish)
{
	if (size < kHeaderBytes || size % kMsgAlign != 0 || size > m_dataSize)
		return false;
	if (out_status)
		*out_status = kRgfxNoAnswer;

	if (m_forward) {
		// The kext stamps the seqno; the type and size are the message's own.
		uint8_t stack[256];
		std::unique_ptr<uint8_t[]> heap;
		uint8_t *buf = stack;
		if (size > sizeof(stack)) {
			heap.reset(new uint8_t[size]);
			buf = heap.get();
		}
		std::memcpy(buf, message, size);
		RgfxMsgHeader header = { type, size, 0 };
		std::memcpy(buf, &header, sizeof(header));
		// What the message names (a library's staging chunk) is written before the kext posts it.
		storeFence();
		uint64_t seqno = 0;
		if (!m_forward(m_forwardContext, buf, size, &seqno, out_status))
			return false;
		if (beforePublish)
			beforePublish(seqno);
		if (out_seqno)
			*out_seqno = seqno;
		return true;
	}

	std::lock_guard<std::mutex> guard(m_lock);

	uint64_t used = m_produced - load64(m_header + kConsumer);
	uint32_t offset = static_cast<uint32_t>(m_produced) & m_mask;
	uint32_t tail = m_dataSize - offset;
	// A message never wraps: pad the bytes left before the ring's end when the
	// message would not fit there.
	uint32_t pad = tail < size ? tail : 0;
	if (used > m_dataSize || static_cast<uint64_t>(m_dataSize) - used < static_cast<uint64_t>(pad) + size)
		return false;

	if (pad) {
		// A pad is a type-0 message covering the bytes left (8 or more); its
		// seqno is not written.
		uint32_t padHeader[2] = { RGFX_MSG_PAD, pad };
		writeBytes(m_data, offset, padHeader, sizeof(padHeader));
		m_produced += pad;
		offset = 0;
	}

	uint64_t seqno = m_nextSeqno++;
	// Copy the caller's payload, then stamp the header over its first 16 bytes.
	writeBytes(m_data, offset, message, size);
	RgfxMsgHeader header = { type, size, seqno };
	writeBytes(m_data, offset, &header, sizeof(header));
	m_produced += size;
	if (beforePublish)
		beforePublish(seqno);

	storeFence();
	store64(m_header + kProducer, m_produced);
	fullFence();
	if (m_doorbell && load32(m_header + kConsumerIdle) == 1)
		m_doorbell(m_doorbellContext, m_ringId);

	if (out_seqno)
		*out_seqno = seqno;
	return true;
}
