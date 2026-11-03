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
#include <functional>
#include <memory>
#include <mutex>

#include "RgfxRingRoomEvents.hpp"
#include "reims_graphics_wire.h"

/// Rings the doorbell for ring `ring_id`: in the bundle, writes `ring_id` to the
/// device's `DOORBELL` register; in a test, records the wake.
typedef void (*RgfxDoorbell)(void *context, uint32_t ring_id);

/// What tryPost() gives as a message's answer where none came back: the message is not one the
/// device answers, or the ring is one the process writes itself, whose answers come later.
constexpr uint32_t kRgfxNoAnswer = 0xffff'ffffu;

/// Hands one whole message (its header's `type` and `size` stamped) to a ring the process
/// does not write itself: in the bundle, the accelerator kext, which posts it on the control
/// ring and stamps its seqno; true once it is posted, with `*out_seqno` the seqno. Where the
/// device answers the message with RGFX_EVT_COMPLETE, the kext waits for the answer and
/// `*out_status` (when not null) is its status; kRgfxNoAnswer otherwise.
typedef bool (*RgfxForward)(void *context, const void *message, uint32_t size,
                            uint64_t *out_seqno, uint32_t *out_status);

/// Hands a pipeline create equal to one the device accepted before to the kext as RgfxForward
/// does, asking it not to wait for the answer (the accelerator's deferred submit). True once
/// posted, `*out_seqno` its seqno. `*out_deferred` says whether its answer comes later
/// (RgfxAwaitAnswer); where the kext kept no room for it, it was answered as any create, the answer
/// handed on as RgfxForward hands its answers, and `*out_deferred` is false.
typedef bool (*RgfxForwardDeferred)(void *context, const void *message, uint32_t size,
                                    uint64_t *out_seqno, bool *out_deferred);

/// Takes the answers the kext keeps for the process's deferred creates, waiting first for the one
/// at `seqno`, and hands each on (to the process's lifecycle reader): false where none came.
typedef bool (*RgfxAwaitAnswer)(void *context, uint64_t seqno);

class RgfxRingWriter {
public:
	/// A writer over the ring whose header is at `ring_base`, with a byte ring
	/// of `data_size` bytes (a power of two) right after the header. `ring_id`
	/// is what the doorbell takes. `doorbell` may be null (no wake), e.g. before
	/// the device is marked ready.
	RgfxRingWriter(volatile uint8_t *ring_base, uint32_t data_size, uint32_t ring_id,
	               RgfxDoorbell doorbell, void *doorbell_context);

	/// A writer that hands each message to `forward` instead of writing a ring: the
	/// process's object lifecycle, which the kext posts on the control ring. `max_size` is
	/// the largest message it takes (dataSize()).
	RgfxRingWriter(RgfxForward forward, void *forward_context, uint32_t max_size);

	/// Logs the room line once more where its posts slept since the last one (noteSleep).
	~RgfxRingWriter();

	/// Writes a message of `size` bytes (its payload in `message`, a multiple of
	/// 8, at least the header and at most `data_size`) and publishes it, stamping
	/// the header's `type`, `size` and `seqno`. Returns false, writing nothing,
	/// when the ring has no room until the host consumes more, or when the
	/// arguments are out of range. On success `*out_seqno` (when not null) holds
	/// the stamped seqno, and `*out_status` (when not null) the COMPLETE status the
	/// forward handed back, or kRgfxNoAnswer. `beforePublish` (when set) is called with
	/// the seqno before the host can see the message: under the ring's lock, before
	/// `producer` moves (for a forwarding writer, once the kext has posted it).
	bool tryPost(uint32_t type, const void *message, uint32_t size, uint64_t *out_seqno,
	             uint32_t *out_status = nullptr,
	             const std::function<void(uint64_t)> &beforePublish = {});

	/// Where a forwarding writer posts a known pipeline create (RgfxForwardDeferred) and takes
	/// its answer (RgfxAwaitAnswer), both called with `context`: the kext's deferred submit,
	/// where it has one.
	void setDeferred(RgfxForwardDeferred post, RgfxAwaitAnswer await, void *context);

	/// Posts a pipeline create equal to one the device accepted before, as tryPost does, without
	/// waiting for its answer where the route keeps answers for later: `*out_deferred` whether
	/// its answer comes later (RgfxAwaitAnswer). With no deferred post (setDeferred), or where
	/// the kext kept no room for the answer, it is posted and answered as tryPost's.
	bool tryPostKnown(uint32_t type, const void *message, uint32_t size, uint64_t *out_seqno,
	                  bool *out_deferred);

	/// Takes the answers the route keeps for the process's deferred creates, waiting for the one
	/// at `seqno` (RgfxAwaitAnswer): false where the route keeps none.
	bool awaitAnswer(uint64_t seqno);

	/// Whether the route keeps answers for later (setDeferred), so a known pipeline create need
	/// not wait.
	bool defers() const { return m_forward != nullptr && m_deferredPost != nullptr; }

	/// Whether post() places a message of `size` bytes: a valid size of at most half the
	/// byte ring, so the pad to the ring's end before it never keeps it out of an empty ring.
	/// A larger message goes out of line (COMMAND_BUFFER_IN_REGION). A forwarding writer
	/// takes up to its `max_size`.
	bool fits(uint32_t size) const;

	/// tryPost(), waiting while the ring has no room: the doorbell rung, so a host that went
	/// idle takes the ring again, then a sleep of 20 µs doubling to 1 ms between looks, and
	/// `whileWaiting` (when set) called before each, as the host consumes on its own thread.
	/// With room events (setRoomEvents), a post that finds no room stores 1 in the ring's
	/// producer_waiting, fences and looks once more, then waits for the host's room record
	/// instead of the sleep: up to 20 ms where no `whileWaiting` is given (a completion thread
	/// takes the record), else up to the sleep's step, as `whileWaiting` may be what takes it.
	/// False, writing nothing, only for a message that does not fit().
	bool post(uint32_t type, const void *message, uint32_t size, uint64_t *out_seqno,
	          const std::function<void()> &whileWaiting = {},
	          const std::function<void(uint64_t)> &beforePublish = {});

	/// The byte ring's size.
	uint32_t dataSize() const { return m_dataSize; }

	/// The ring's room records, where the device's word holds RGFX_FEATURE_SUBMIT_ROOM: set once,
	/// before the first post (the queue's reader hands them over). A forwarding writer takes
	/// none.
	void setRoomEvents(std::shared_ptr<RgfxRingRoomEvents> room)
	{
		if (m_header != nullptr)
			m_room = std::move(room);
	}

private:
	volatile uint8_t *m_header;
	volatile uint8_t *m_data;
	uint32_t m_dataSize;
	uint32_t m_mask;
	uint32_t m_ringId;
	RgfxDoorbell m_doorbell;
	void *m_doorbellContext;
	RgfxForward m_forward;
	void *m_forwardContext;
	RgfxForwardDeferred m_deferredPost = nullptr;
	RgfxAwaitAnswer m_await = nullptr;
	void *m_deferredContext = nullptr;

	// Counts one sleep of post's from `from` (byRecord: a room record ended it), and logs the
	// counts at the first sleep and then at most every second.
	void noteSleep(std::chrono::steady_clock::time_point from, bool byRecord);
	// The room line, with the counts `seen`.
	void logSleeps(const RgfxRingRoomSleeps &seen) const;

	std::shared_ptr<RgfxRingRoomEvents> m_room; // setRoomEvents'
	// The posts' sleeps for room, under m_sleepsLock, logged at the first, at most every second
	// after it (post), and once more as the writer goes where sleeps came since the last line.
	std::mutex m_sleepsLock;
	RgfxRingRoomSleeps m_sleeps;

	std::mutex m_lock;
	uint64_t m_produced;
	uint64_t m_nextSeqno;
};

#endif // RGFX_RING_WRITER_HPP_
