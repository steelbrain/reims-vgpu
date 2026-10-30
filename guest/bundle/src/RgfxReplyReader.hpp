/*
 * RgfxReplyReader: the guest's end of a host-to-guest completion ring — the consumer
 * mirror of the host producer in crates/reims-graphics/src/ring.rs (Producer) and
 * crates/reims-graphics/src/device.rs (ControlCompletionPoster). It reads the host's
 * reply messages off the ring and dispatches each to the wrapper that is waiting on
 * it:
 *   - RGFX_EVT_CB_COMPLETE (RgfxCbComplete) -> the command buffer whose submission
 *     seqno it carries: its completed handlers fire, its status flips, a waiter wakes;
 *   - RGFX_EVT_PIPELINE_REFLECTION (RgfxPipelineReflection) -> the pipeline-state
 *     wrapper whose create seqno it carries: the wrapper's reflection fills;
 *   - RGFX_EVT_EVENT_SIGNALLED (RgfxEventSignalled) -> the shared-event wrapper named
 *     by its obj_id: its value mirror updates and its listeners fire.
 * Every other message type (the control-command answers RGFX_EVT_COMPLETE / DISPLAYS /
 * VBLANK, which the control path owns) is skipped. The three reply structs' byte
 * layout is the committed wire header's; the host builders this mirrors are
 * crates/reims-vgpu/src/graphics/completion.rs (cb_complete / cb_failed) and
 * decode/mod.rs (render_reflection / compute_reflection / emit_pipeline_refused).
 *
 * THE SINGLE-RING MODEL. The host posts all three replies on the one control
 * completion ring (control.rs: "replies … come back on the completion ring through
 * the device's poster"), so for now the reader reads that ring. When the host grows
 * per-queue completion rings, a reader per queue reads each; the dispatch is the same.
 *
 * DRIVEN BY POLLING. The read is driven by `pump()` (alias `drainReplies()`), which
 * the device calls from a boot thread (or a waiter from waitUntilCompleted). It is
 * not interrupt-driven yet: it does not set the ring's `armed` flag, so the host
 * raises no vector; a boot thread that sleeps on the completion vector, arming the
 * ring, is a follow-up. The reader is testable off the guest by feeding ring bytes
 * and calling pump().
 *
 * THREADING. Registration (the expect / addEventListener calls) runs on the app's threads while
 * a boot thread pumps; one lock guards the registries and the consume cursor. A
 * matched handler is moved out and invoked after the lock is released, so a handler
 * may register again or release the wrapper without re-entering the lock.
 *
 * Our own C++; depends only on the committed wire header. The handlers are supplied
 * by the ObjC++ wrappers (which capture their Metal object), so this file names no
 * Metal SPI.
 */

#ifndef RGFX_REPLY_READER_HPP_
#define RGFX_REPLY_READER_HPP_

#include <cstdint>
#include <functional>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "reims_graphics_wire.h"

class RgfxReplyReader {
public:
	/// A command buffer's completion handler, invoked once with the whole message.
	typedef std::function<void(const RgfxCbComplete &)> CompletionHandler;
	/// A pipeline create's reflection handler, invoked once with the whole message.
	typedef std::function<void(const RgfxPipelineReflection &)> ReflectionHandler;
	/// A shared event's listener, invoked on every EVENT_SIGNALLED for its obj_id.
	typedef std::function<void(const RgfxEventSignalled &)> EventHandler;

	/// A reader over the completion ring whose header is at `ring_base`, with a byte
	/// ring of `data_size` bytes (a power of two) right after the header. The ring is
	/// host-to-guest: the host produces, this consumes.
	RgfxReplyReader(volatile uint8_t *ring_base, uint32_t data_size);

	/// Registers `handler` for the command buffer submitted at `seqno`; fired once,
	/// when its RGFX_EVT_CB_COMPLETE arrives, then removed. A second registration for
	/// the same seqno replaces the first.
	void expectCompletion(uint64_t seqno, CompletionHandler handler);

	/// Registers `handler` for the pipeline created at `seqno`; fired once, when its
	/// RGFX_EVT_PIPELINE_REFLECTION arrives, then removed.
	void expectReflection(uint64_t seqno, ReflectionHandler handler);

	/// Registers `handler` for the shared event `obj_id`; fired on every
	/// RGFX_EVT_EVENT_SIGNALLED for it until removed. Returns a token for removal.
	uint64_t addEventListener(uint64_t obj_id, EventHandler handler);

	/// Removes the event listener `token` (addEventListener's return). A token that
	/// is not registered is ignored.
	void removeEventListener(uint64_t token);

	/// Reads every message the host has published and not yet consumed, dispatches
	/// each to its handler, and publishes the consumer count so the host sees the
	/// room freed. Returns how many messages it dispatched (pads and unmatched or
	/// skipped types do not count). Stops early, publishing what it took, on a
	/// malformed message size (a broken ring).
	uint32_t pump();

	/// Alias for pump() (the device's boot-thread name).
	uint32_t drainReplies() { return pump(); }

	/// Whether a completion / reflection is still awaited for `seqno` (tests).
	bool awaitingCompletion(uint64_t seqno);
	bool awaitingReflection(uint64_t seqno);

private:
	/// One event listener: its obj_id and its handler, kept by token.
	struct EventListener {
		uint64_t obj_id;
		EventHandler handler;
	};

	volatile uint8_t *m_header;
	volatile uint8_t *m_data;
	uint32_t m_dataSize;
	uint32_t m_mask;

	std::mutex m_lock;
	uint64_t m_consumed; // the guest's count: what it has taken

	std::unordered_map<uint64_t, CompletionHandler> m_completions;
	std::unordered_map<uint64_t, ReflectionHandler> m_reflections;
	std::unordered_map<uint64_t, EventListener> m_eventListeners;
	uint64_t m_nextToken;
};

#endif // RGFX_REPLY_READER_HPP_
