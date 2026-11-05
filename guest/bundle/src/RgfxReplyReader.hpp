/*
 * RgfxReplyReader: the guest's end of a host-to-guest completion ring — the consumer
 * mirror of the host producer in crates/reims-graphics/src/ring.rs (Producer) and
 * crates/reims-graphics/src/device.rs (ControlCompletionPoster). It reads the host's
 * reply messages off the ring and dispatches each to the wrapper that is waiting on
 * it:
 *   - RGFX_EVT_CB_SUBMITTED (RgfxCbSubmitted) -> the command buffer whose submission
 *     seqno it carries, which asked for it: it is submitted, and commitAndWaitUntilSubmitted
 *     returns;
 *   - RGFX_EVT_CB_SCHEDULED (RgfxCbScheduled) -> the command buffer whose submission
 *     seqno it carries: it is scheduled, its scheduled handlers fire, a waiter wakes;
 *   - RGFX_EVT_CB_COMPLETE (RgfxCbComplete) -> the command buffer whose submission
 *     seqno it carries: its completed handlers fire, its status flips, a waiter wakes;
 *   - RGFX_EVT_PIPELINE_REFLECTION (RgfxPipelineReflection) -> the pipeline-state
 *     wrapper whose create seqno it carries: the wrapper's reflection fills;
 *   - RGFX_EVT_EVENT_SIGNALLED (RgfxEventSignalled) -> the shared-event wrapper named
 *     by its obj_id: its value mirror updates and its listeners fire. The host posts it
 *     on the completion ring of the queue whose command buffer signalled the event, ahead
 *     of that buffer's CB_COMPLETE; a queue's reader hands it to the device's
 *     (setEventsTo), where the process's shared events listen.
 * Every other message type (the control-command answers RGFX_EVT_COMPLETE / DISPLAYS /
 * VBLANK, which the control path owns) is skipped. The three reply structs' byte
 * layout is the committed wire header's; the host builders this mirrors are
 * crates/reims-vgpu/src/graphics/completion.rs (cb_complete / cb_failed) and
 * decode/mod.rs (render_reflection / compute_reflection / emit_pipeline_refused).
 *
 * ONE READER PER QUEUE. Each command queue has a completion ring of its own, which the
 * host posts its command buffers' completions on, and a reader over it; the device's
 * ringless reader takes the replies the kext hands back from the control ring (deliver).
 *
 * DRIVEN BY THE COMPLETION THREAD. The read is `pump()` (alias `drainReplies()`). The
 * process's completion thread (RgfxCompletionThread) drives each queue's reader: it
 * pumps, arms the ring (armAndCheck: `armed` set, so the host raises the vector for its
 * next completion) and sleeps in the kext until the vector shows a record for it. While
 * a reader is driven() the thread is its only consumer; an undriven one (no thread, or
 * its thread has ended) is pumped by the waiters in waitUntilCompleted. A command buffer's
 * handlers registered here hand its records on to its queue's completion queue
 * (RgfxCommandObjects.h, COMPLETION), so what runs on the pumping thread is short. The
 * reader is testable off the guest by feeding ring bytes and calling pump().
 *
 * THREADING. Registration (the expect / addEventListener calls) runs on the app's threads while
 * the completion thread pumps; one lock guards the registries and the consume cursor. A
 * matched handler is moved out and invoked after the lock is released, so a handler
 * may register again or release the wrapper without re-entering the lock. One consumer
 * at a time runs the handlers, so they run in ring order.
 *
 * Our own C++; depends only on the committed wire header. The handlers are supplied
 * by the ObjC++ wrappers (which capture their Metal object), so this file names no
 * Metal SPI.
 */

#ifndef RGFX_REPLY_READER_HPP_
#define RGFX_REPLY_READER_HPP_

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "RgfxRingRoomEvents.hpp"
#include "reims_graphics_wire.h"

class RgfxReplyReader {
public:
	/// A command buffer's completion handler, invoked once with the whole message.
	typedef std::function<void(const RgfxCbComplete &)> CompletionHandler;
	/// A command buffer's scheduled handler, invoked once with the whole message.
	typedef std::function<void(const RgfxCbScheduled &)> ScheduledHandler;
	/// A command buffer's submitted handler, invoked once with the whole message.
	typedef std::function<void(const RgfxCbSubmitted &)> SubmittedHandler;
	/// A pipeline create's reflection handler, invoked once with the whole message.
	typedef std::function<void(const RgfxPipelineReflection &)> ReflectionHandler;
	/// A shared event's listener, invoked on every EVENT_SIGNALLED for its obj_id.
	typedef std::function<void(const RgfxEventSignalled &)> EventHandler;

	/// A reader over the completion ring whose header is at `ring_base`, with a byte
	/// ring of `data_size` bytes (a power of two) right after the header. The ring is
	/// host-to-guest: the host produces, this consumes.
	RgfxReplyReader(volatile uint8_t *ring_base, uint32_t data_size);

	/// A reader with no ring: the replies the kext hands back from the control ring
	/// (deliver), for the process's object lifecycle. pump() reads nothing.
	RgfxReplyReader();

	/// Dispatches one reply the kext handed back (a whole message): a pipeline create's
	/// RGFX_EVT_PIPELINE_REFLECTION, which arrives before its wrapper registers, so one
	/// with no handler yet is kept and handed to the expectReflection that registers for
	/// its seqno. Other types are ignored.
	void deliver(const void *message, uint32_t size);

	/// Registers `handler` for the command buffer submitted at `seqno`; fired once,
	/// when its RGFX_EVT_CB_COMPLETE arrives, then removed. A second registration for
	/// the same seqno replaces the first.
	void expectCompletion(uint64_t seqno, CompletionHandler handler);

	/// Registers `handler` for the command buffer submitted at `seqno`; fired once, when its
	/// RGFX_EVT_CB_SCHEDULED arrives, then removed. Its RGFX_EVT_CB_COMPLETE removes it unfired
	/// (the host posts no CB_SCHEDULED for a buffer it could not carry out): the completion
	/// stands for both.
	void expectScheduled(uint64_t seqno, ScheduledHandler handler);

	/// Registers `handler` for the command buffer submitted at `seqno`, which asked for its
	/// submission (RGFX_OP_SUBMITTED); fired once, when its RGFX_EVT_CB_SUBMITTED arrives, then
	/// removed. Its RGFX_EVT_CB_SCHEDULED or RGFX_EVT_CB_COMPLETE removes it unfired: the host
	/// posts CB_SUBMITTED ahead of both, and none for a buffer it could not take.
	void expectSubmitted(uint64_t seqno, SubmittedHandler handler);

	/// Registers `handler` for the pipeline created at `seqno`; fired once, when its
	/// RGFX_EVT_PIPELINE_REFLECTION arrives, then removed.
	void expectReflection(uint64_t seqno, ReflectionHandler handler);

	/// Registers `handler` for the shared event `obj_id`; fired on every
	/// RGFX_EVT_EVENT_SIGNALLED for it until removed. Returns a token for removal.
	uint64_t addEventListener(uint64_t obj_id, EventHandler handler);

	/// Removes the event listener `token` (addEventListener's return). A token that
	/// is not registered is ignored.
	void removeEventListener(uint64_t token);

	/// Hands every RGFX_EVT_EVENT_SIGNALLED this reader reads to `events` (signalled)
	/// instead of this reader's own listeners: a queue's reader, to the device's.
	/// `events` outlives this reader, as the process's device does its queues.
	void setEventsTo(RgfxReplyReader *events);

	/// Fires this reader's listeners for `msg`'s shared event, in registration order.
	void signalled(const RgfxEventSignalled &msg);

	/// Hands each RGFX_EVT_SUBMIT_ROOM this reader reads to `room` (arrived): a queue's reader,
	/// to its submission ring's writer (RgfxRingWriter::setRoomEvents).
	void setRoomEvents(std::shared_ptr<RgfxRingRoomEvents> room);

	/// Reads every message the host has published and not yet consumed, dispatches
	/// each to its handler, and publishes the consumer count so the host sees the
	/// room freed. Returns how many messages it dispatched (pads and unmatched or
	/// skipped types do not count). Stops early, publishing what it took, on a
	/// malformed message size (a broken ring).
	uint32_t pump();

	/// Alias for pump() (the device's boot-thread name).
	uint32_t drainReplies() { return pump(); }

	/// Arms the ring (its `armed` set to 1, so the host raises the vector with its next
	/// completion), then, after a full fence, whether the host published records this has
	/// not taken: true means pump again before sleeping. False for a ringless reader.
	bool armAndCheck();

	/// Whether the ring holds, among the records not yet taken, the host's word that the command
	/// buffer submitted at `seqno` is submitted: its RGFX_EVT_CB_SUBMITTED, or the CB_SCHEDULED
	/// or CB_COMPLETE that stands for one. Reads at most kLookRecords records past the consumer
	/// count and takes none, so the ring's consumer still takes each in ring order and runs its
	/// handler. A host that answers a buffer on its doorbell's thread has posted the record
	/// before the doorbell write returns, so the thread that posted the buffer finds it here
	/// with no wake of the completion thread (RgfxCmdBuffer's rgfxPost).
	bool holdsSubmitted(uint64_t seqno);

	/// The most records holdsSubmitted reads.
	static constexpr uint32_t kLookRecords = 64;

	/// Stops reading the ring for good: pump() and armAndCheck() touch it no more, so its
	/// block can go back to the region (the queue's end) though the completion thread still
	/// holds the reader.
	void close();

	/// Whether the completion thread drives this reader (it is then the only consumer).
	void setDriven(bool driven) { m_driven.store(driven, std::memory_order_release); }
	bool driven() const { return m_driven.load(std::memory_order_acquire); }
	/// Whether the reader reads a ring (the device's lifecycle reader reads none).
	bool hasRing() const { return m_header != nullptr; }

	/// Whether a completion / scheduling / submission / reflection is still awaited for
	/// `seqno` (tests).
	bool awaitingCompletion(uint64_t seqno);
	bool awaitingScheduled(uint64_t seqno);
	bool awaitingSubmitted(uint64_t seqno);
	bool awaitingReflection(uint64_t seqno);

private:
	/// One event listener: its obj_id and its handler, kept by token.
	struct EventListener {
		uint64_t obj_id;
		EventHandler handler;
	};

	/// The handlers listening for `obj_id`, in registration order. Called with the lock held.
	std::vector<EventHandler> listenersOf(uint64_t obj_id);

	volatile uint8_t *m_header;
	volatile uint8_t *m_data;
	uint32_t m_dataSize;
	uint32_t m_mask;

	std::mutex m_lock;
	uint64_t m_consumed; // the guest's count: what it has taken
	bool m_closed = false; // close(): the ring is not read again
	std::atomic<bool> m_driven{false};

	std::unordered_map<uint64_t, CompletionHandler> m_completions;
	std::unordered_map<uint64_t, ScheduledHandler> m_scheduled;
	std::unordered_map<uint64_t, SubmittedHandler> m_submitted;
	std::unordered_map<uint64_t, ReflectionHandler> m_reflections;
	std::unordered_map<uint64_t, RgfxPipelineReflection> m_earlyReflections;
	std::unordered_map<uint64_t, EventListener> m_eventListeners;
	uint64_t m_nextToken;
	RgfxReplyReader *m_eventsTo = nullptr; // setEventsTo: borrowed
	std::shared_ptr<RgfxRingRoomEvents> m_room; // setRoomEvents'

};

#endif // RGFX_REPLY_READER_HPP_
