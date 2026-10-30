/*
 * RgfxRingWriter: the guest producer's reserve-write-publish-doorbell step
 * (RgfxRingWriter.hpp). Mirrors the host consumer in
 * crates/reims-graphics/src/ring.rs.
 */

#include "RgfxRingWriter.hpp"

#include <cstdio>
#include <cstring>
#include <memory>
#include <fcntl.h>
#include <unistd.h>

namespace {

// RGFX DIAG (remove before merge): append one post's trace to /tmp/rgfx-diag.log, to see the
// control vs per-queue submission handshake (ring id, counts, idle, wake) from the guest side.
void rgfxRingDiag(uint32_t ring_id, uint32_t type, uint32_t size, uint64_t seqno, uint64_t producer,
                  uint64_t consumer, uint32_t idle, bool rang)
{
	char line[200];
	int n = snprintf(line, sizeof(line),
	                 "RGFX: post ring=0x%x type=0x%x size=%u seqno=%llu producer=%llu "
	                 "consumer=%llu idle=%u rang=%d\n",
	                 ring_id, type, size, (unsigned long long)seqno, (unsigned long long)producer,
	                 (unsigned long long)consumer, idle, rang ? 1 : 0);
	if (n <= 0)
		return;
	int fd = open("/tmp/rgfx-diag.log", O_WRONLY | O_CREAT | O_APPEND, 0666);
	if (fd < 0)
		return;
	(void)!write(fd, line, static_cast<size_t>(n < (int)sizeof(line) ? n : (int)sizeof(line)));
	close(fd);
}

// The ring header fields, by their offsets in RgfxRingHeader.
constexpr uint32_t kProducer = __builtin_offsetof(RgfxRingHeader, producer);
constexpr uint32_t kConsumer = __builtin_offsetof(RgfxRingHeader, consumer);
constexpr uint32_t kConsumerIdle = __builtin_offsetof(RgfxRingHeader, consumer_idle);

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

void fullFence()
{
	__atomic_thread_fence(__ATOMIC_SEQ_CST);
}

// Copies `len` bytes to the ring at byte offset `at` (already within the data
// area, no wrap). The region is write-back memory; a plain copy is right, and
// the release store of `producer` orders it before the host reads it.
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
	// The bundle's rings (the device ring and the per-queue rings) are its own: fresh, and the
	// host resets each at device init and at reset, so the writer starts from 0. (The interim
	// "join at the current producer" was for sharing the framebuffer kext's control ring; the
	// device ring replaced that, so each ring has a single producer and starts clean.)
}

RgfxRingWriter::RgfxRingWriter(RgfxForward forward, void *forward_context, uint32_t max_size)
    : m_header(nullptr), m_data(nullptr), m_dataSize(max_size), m_mask(0), m_ringId(0),
      m_doorbell(nullptr), m_doorbellContext(nullptr), m_forward(forward),
      m_forwardContext(forward_context), m_produced(0), m_nextSeqno(1)
{
}

bool RgfxRingWriter::tryPost(uint32_t type, const void *message, uint32_t size, uint64_t *out_seqno)
{
	if (size < kHeaderBytes || size % kMsgAlign != 0 || size > m_dataSize)
		return false;

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
		uint64_t seqno = 0;
		if (!m_forward(m_forwardContext, buf, size, &seqno))
			return false;
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

	store64(m_header + kProducer, m_produced);
	fullFence();
	uint32_t idle = load32(m_header + kConsumerIdle);
	bool rang = m_doorbell && idle == 1;
	if (rang)
		m_doorbell(m_doorbellContext, m_ringId);

	// RGFX DIAG (remove before merge): trace this post's ring state and wake.
	rgfxRingDiag(m_ringId, type, size, seqno, m_produced, load64(m_header + kConsumer), idle, rang);

	if (out_seqno)
		*out_seqno = seqno;
	return true;
}
