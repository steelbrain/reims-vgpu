/*
 * RgfxRingWriter: the guest's end of a submission ring (guest -> host) in the
 * device's BAR 2 region (docs/graphics-kext/design.md, "Stage 2 wire protocol"
 * and "Stage 3 wire protocol"; crates/reims-vgpu-graphics/src/ring.rs is the host's
 * mirror, whose test `Guest` is this write path).
 *
 * A ring is an RgfxRingHeader followed by a byte ring of `data_size` bytes (a
 * power of two). The writer keeps the free-running `producer` count and the next
 * per-ring seqno. Posting a message: pad to the ring's end if it would not fit
 * before it (a message never wraps), copy the message, publish `producer` with a
 * release store, issue a full fence, and ring the doorbell only if the host
 * marked the ring asleep (`consumer_idle` RGFX_CONSUMER_ASLEEP), or walking
 * (RGFX_CONSUMER_WALKING) for a command buffer it may answer at the doorbell (tryPost).
 * The message's 16-byte header has
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
#include <array>
#include <functional>
#include <memory>
#include <mutex>
#include <time.h>
#include <vector>

#include "RgfxRingRoomEvents.hpp"
#include "reims_graphics_wire.h"

/// Rings the doorbell for ring `ring_id`: in the bundle, writes `ring_id` to the
/// device's `DOORBELL` register; in a test, records the wake.
typedef void (*RgfxDoorbell)(void *context, uint32_t ring_id);

/// The guest's clock in nanoseconds, the kext's: mach_absolute_time in nanoseconds, which the
/// command buffers' times are stamped with where the device's word holds RGFX_FEATURE_GUEST_TIMES
/// (RgfxEncoder::stampGuestTimes, RgfxRingWriter::tryPost's `stampAt`).
inline uint64_t rgfxGuestClockNs()
{
	return clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
}

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

/// Hands a pipeline create of pipeline key `key` (32 bytes, rgfxSharedPipelineKey) to the kext as
/// RgfxForward does, asking it not to wait for the answer where it keeps the answer to an equal
/// create of any process (the accelerator's shared submit). True once posted, `*out_seqno` its
/// seqno. Where it kept one, `*out_deferred` is true and `*out_reflection` that answer, stamped
/// for this create, and the create's own answer comes later (RgfxAwaitAnswer); else it was
/// answered as any create, the answer handed on as RgfxForward hands its answers, and
/// `*out_deferred` is false.
typedef bool (*RgfxForwardShared)(void *context, const void *message, uint32_t size,
                                  const uint8_t *key, uint64_t *out_seqno, bool *out_deferred,
                                  RgfxPipelineReflection *out_reflection);

/// Takes the answers the kext keeps for the process's deferred creates, waiting first for the one
/// at `seqno`, and hands each on (to the process's lifecycle reader): false where none came.
typedef bool (*RgfxAwaitAnswer)(void *context, uint64_t seqno);

/// Hands `bytes` bytes of whole messages back to back (each header's `type` and `size` stamped),
/// which no one waits for, to the kext in one call (the accelerator's SubmitBatch, which takes
/// the types the kext's rgfxSubmitBatchSize names): how many of them it posted, from the first,
/// in their order. Fewer than all
/// only where the kext refused them, as it does where its control ring stays full.
typedef uint32_t (*RgfxForwardBatch)(void *context, const void *messages, uint32_t bytes);

/// A command queue's publish the kext makes after the held messages it posts (the accelerator's
/// SubmitBatch with its publish): the queue, its submission ring's new producer, whether its
/// command buffer is one the host answers at the doorbell (rung for a walking ring too), and
/// whether that doorbell names RGFX_DOORBELL_ASKS.
struct RgfxQueuePublish {
	uint64_t queue;
	uint64_t producer;
	bool answered;
	bool asks;
};

/// Hands `bytes` bytes of held messages to the kext as RgfxForwardBatch does, and with them
/// `publish`, which the kext makes after them once it has posted every one, as the queue's writer
/// would (RgfxRingWriter::tryPost): how many it posted, from the first, and `*out_published`
/// whether it published.
typedef uint32_t (*RgfxForwardBatchPublish)(void *context, const void *messages, uint32_t bytes,
                                            const RgfxQueuePublish &publish, bool *out_published);

class RgfxRingWriter;

/// A command buffer's post that hands its publish to the control ring's held messages, so a commit
/// behind them makes one kext call (RgfxRingWriter::tryPost): the control ring's writer and the
/// buffer's queue. `refused` is set where the post left the buffer off its ring because a held
/// message other than a destroy was refused, which the buffer may name (as
/// RgfxSerializer::postHeldCreates fails it).
struct RgfxPublishVia {
	RgfxRingWriter *control = nullptr;
	uint64_t queue = 0;
	bool refused = false;
};

/// How long a message RgfxRingWriter::postLater() holds waits at most for others to go with it:
/// a run of destroys an app lets go back to back takes tens of microseconds.
constexpr uint64_t kRgfxLaterNs = 1000000;

struct RgfxRingLater;

/// While one lives on a thread, the postLater() calls the thread makes arm no timer: the timers
/// they would have armed are armed as it ends, each then posting what is held setBatch's time
/// later at most. A command buffer's delivery holds one from the release of what the buffer kept
/// to its waiters' release, so the arm, a wake of the dispatch manager's thread (about 15 us of
/// the delivery's on a 780M host, 11-14), comes after the waiters are let go, not before.
class RgfxRingLaterArms {
public:
	RgfxRingLaterArms();
	~RgfxRingLaterArms();
	RgfxRingLaterArms(const RgfxRingLaterArms &) = delete;
	RgfxRingLaterArms &operator=(const RgfxRingLaterArms &) = delete;

private:
	friend class RgfxRingWriter;
	RgfxRingLaterArms *m_outer;
	std::vector<std::shared_ptr<RgfxRingLater>> m_arms;
};

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

	/// Logs the room line once more where its posts slept since the last one (noteSleep), and
	/// posts what postLater() holds.
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
	/// The doorbell rings where the host marked the ring asleep (RGFX_CONSUMER_ASLEEP), and, with
	/// `answered` (a command buffer that asks for its CB_SUBMITTED, which the host may answer
	/// inside the doorbell's write), where it marked it walking (RGFX_CONSUMER_WALKING): a
	/// walking host takes the ring again once its walk is done, with no doorbell.
	/// A `stampAt` other than 0 is where in the message its posted time lies (a command buffer's
	/// RGFX_OP_GUEST_TIMES, RGFX_GUEST_TIMES_POSTED_OFFSET): the guest's clock is written there in
	/// the ring's copy, under the lock, after the copy and `beforePublish` and before `producer`
	/// moves, as the seqno is stamped. A forwarding writer writes no stamp.
	/// With `via`, the publish goes to `via->control`'s held messages (flushPublishing), which
	/// the kext posts first and then publishes this ring as this writer would; where none is held
	/// or the kext did not publish, this writer publishes as it does without `via`, but where a
	/// held message other than a destroy is still held then: the message is taken off the ring
	/// again, unpublished, its seqno the next post's, `via->refused` set and false returned.
	bool tryPost(uint32_t type, const void *message, uint32_t size, uint64_t *out_seqno,
	             uint32_t *out_status = nullptr,
	             const std::function<void(uint64_t)> &beforePublish = {}, bool answered = false,
	             uint32_t stampAt = 0, RgfxPublishVia *via = nullptr);

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

	/// Where a forwarding writer posts what postLater() holds (RgfxForwardBatch), called with the
	/// forward's context, at most `max_bytes` a call, `later_ns` after the first held at most
	/// (kRgfxLaterNs; a test's own). Set once, before the first post.
	void setBatch(RgfxForwardBatch post, uint32_t max_bytes, uint64_t later_ns = kRgfxLaterNs);

	/// Whether postLater() holds messages to go together (setBatch).
	bool batches() const { return m_later != nullptr; }

	/// Posts a message no one waits for (a destroy, a create that names no block) together with
	/// others where the writer batches (setBatch): held, to go in the order held before anything
	/// the writer posts after it (each tryPost, tryPostKnown and tryPostShared posts the held first),
	/// at flushLater(), once the held reach the batch's bytes, or setBatch's time after the first
	/// held (after the end of the RgfxRingLaterArms the thread that held it was in, where it was).
	/// Else, or for a message larger than a batch, posted as tryPost() posts it. True once
	/// held or posted; false, holding nothing more, where the held had to go first to make room
	/// and the route refused them. A held message the route refuses stays held, first, and the
	/// writer posts nothing else until it goes.
	bool postLater(uint32_t type, const void *message, uint32_t size);

	/// Posts what postLater() holds, in its order: whether none is held now. One atomic load
	/// where none is.
	bool flushLater();

	/// Whether postLater() holds a message of a type other than `type`.
	bool holdsBut(uint32_t type);

	/// Where a batching writer's route takes a queue's publish with the held messages
	/// (RgfxForwardBatchPublish), called with the forward's context. Set once, after setBatch,
	/// before the first post.
	void setBatchPublish(RgfxForwardBatchPublish post);

	/// Whether the route takes a queue's publish with the held messages (setBatchPublish).
	bool publishesQueues() const;

	/// Posts what postLater() holds, as flushLater() does, with `publish` in its last call: true
	/// where the route posted every held message and published the queue's ring after them. False
	/// where none was held (nothing posted), where the route refused a held message (the rest stay
	/// held, first) or published nothing: the queue's writer then publishes, or not.
	bool flushPublishing(const RgfxQueuePublish &publish);

	/// Whether the route keeps answers for later (setDeferred), so a known pipeline create need
	/// not wait.
	bool defers() const { return m_forward != nullptr && m_deferredPost != nullptr; }

	/// Where a forwarding writer that defers posts a pipeline create against the answers the kext
	/// keeps for every process (RgfxForwardShared), called with setDeferred's context.
	void setShared(RgfxForwardShared post) { m_sharedPost = post; }

	/// Whether the route answers a pipeline create from the kext's answers of every process
	/// (setShared).
	bool shares() const { return defers() && m_sharedPost != nullptr; }

	/// Posts a pipeline create of pipeline key `key` (32 bytes), as tryPost does, without waiting
	/// for its answer where the kext keeps an equal create's: `*out_deferred` whether its answer
	/// comes later (RgfxAwaitAnswer), `*out_reflection` then the kept answer. Without a shared
	/// post (setShared), it is posted and answered as tryPost's.
	bool tryPostShared(uint32_t type, const void *message, uint32_t size,
	                   const std::array<uint8_t, 32> &key, uint64_t *out_seqno, bool *out_deferred,
	                   RgfxPipelineReflection *out_reflection);

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
	/// False, writing nothing, only for a message that does not fit(), or where tryPost set
	/// `via->refused`. `stampAt` and `via` as tryPost's: the stamp is the post's that finds room.
	bool post(uint32_t type, const void *message, uint32_t size, uint64_t *out_seqno,
	          const std::function<void()> &whileWaiting = {},
	          const std::function<void(uint64_t)> &beforePublish = {}, bool answered = false,
	          uint32_t stampAt = 0, RgfxPublishVia *via = nullptr);

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

	/// Where the device's word holds RGFX_FEATURE_DOORBELL_ASKS: a post that rings for a command
	/// buffer it may answer at the doorbell (`answered`) writes the ring's id with
	/// RGFX_DOORBELL_ASKS, which the host answers inside the write, and every other doorbell the
	/// id alone, which the host may take with no exit to its process. Set once, before the first
	/// post.
	void setDoorbellAsks(bool asks) { m_doorbellAsks = asks; }

private:
	volatile uint8_t *m_header;
	volatile uint8_t *m_data;
	uint32_t m_dataSize;
	uint32_t m_mask;
	uint32_t m_ringId;
	RgfxDoorbell m_doorbell;
	void *m_doorbellContext;
	bool m_doorbellAsks = false; // setDoorbellAsks'
	RgfxForward m_forward;
	void *m_forwardContext;
	RgfxForwardDeferred m_deferredPost = nullptr;
	RgfxForwardShared m_sharedPost = nullptr;
	RgfxAwaitAnswer m_await = nullptr;
	void *m_deferredContext = nullptr;

	// Counts one sleep of post's from `from` (byRecord: a room record ended it), and logs the
	// counts at the first sleep and then at most every second.
	void noteSleep(std::chrono::steady_clock::time_point from, bool byRecord);
	// The room line, with the counts `seen`.
	void logSleeps(const RgfxRingRoomSleeps &seen) const;

	// What postLater() holds, shared with its timer (setBatch's; null where the writer does not
	// batch).
	std::shared_ptr<RgfxRingLater> m_later;

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
