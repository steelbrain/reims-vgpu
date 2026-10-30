/*
 * RgfxRingWriter: the guest's end of a submission ring (guest -> host) in the
 * device's BAR 2 region (docs/graphics-kext/design.md, "Stage 2 wire protocol"
 * and "Stage 3 wire protocol"; crates/reims-graphics/src/ring.rs is the host's
 * mirror, whose test `Guest` is this write path).
 *
 * A ring is an RgfxRingHeader followed by a byte ring of `data_size` bytes (a
 * power of two). The writer keeps the free-running `producer` count and the next
 * per-ring seqno. Posting a message: pad to the ring's end if it would not fit
 * before it (a message never wraps), copy the message, publish `producer` with a
 * release store, issue a full fence, and ring the doorbell only if the host
 * marked the ring idle (`consumer_idle` == 1). The message's 16-byte header has
 * its `type`, `size` and `seqno` stamped by the writer; the caller fills the
 * payload.
 *
 * The region is ordinary write-back memory the guest maps; the header fields are
 * touched with C11 atomics, as the protocol orders them. One lock serialises the
 * whole reserve-write-publish step, since several of the app's threads may post
 * to one queue's ring.
 *
 * This is our own code and depends only on the committed wire header. For the
 * host unit test the ring memory is a plain buffer and the doorbell is a counter;
 * in the bundle it is BAR-mapped memory and an MMIO register write.
 */

#ifndef RGFX_RING_WRITER_HPP_
#define RGFX_RING_WRITER_HPP_

#include <cstdint>
#include <mutex>

#include "reims_graphics_wire.h"

/// Rings the doorbell for ring `ring_id`: in the bundle, writes `ring_id` to the
/// device's `DOORBELL` register; in a test, records the wake.
typedef void (*RgfxDoorbell)(void *context, uint32_t ring_id);

class RgfxRingWriter {
public:
	/// A writer over the ring whose header is at `ring_base`, with a byte ring
	/// of `data_size` bytes (a power of two) right after the header. `ring_id`
	/// is what the doorbell takes. `doorbell` may be null (no wake), e.g. before
	/// the device is marked ready.
	RgfxRingWriter(volatile uint8_t *ring_base, uint32_t data_size, uint32_t ring_id,
	               RgfxDoorbell doorbell, void *doorbell_context);

	/// Writes a message of `size` bytes (its payload in `message`, a multiple of
	/// 8, at least the header and at most `data_size`) and publishes it, stamping
	/// the header's `type`, `size` and `seqno`. Returns false, writing nothing,
	/// when the ring has no room until the host consumes more, or when the
	/// arguments are out of range. On success `*out_seqno` (when not null) holds
	/// the stamped seqno.
	bool tryPost(uint32_t type, const void *message, uint32_t size, uint64_t *out_seqno);

	/// The byte ring's size.
	uint32_t dataSize() const { return m_dataSize; }

private:
	volatile uint8_t *m_header;
	volatile uint8_t *m_data;
	uint32_t m_dataSize;
	uint32_t m_mask;
	uint32_t m_ringId;
	RgfxDoorbell m_doorbell;
	void *m_doorbellContext;

	std::mutex m_lock;
	uint64_t m_produced;
	uint64_t m_nextSeqno;
};

#endif // RGFX_RING_WRITER_HPP_
