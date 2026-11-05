/*
 * RgfxReplyReader: the guest completion-ring consumer (RgfxReplyReader.hpp). Mirrors
 * the host producer in crates/reims-graphics/src/ring.rs: the host writes a message,
 * publishes `producer` with a release store and pads to the ring's end so a message
 * never wraps; this reads `producer` with an acquire load, walks whole messages from
 * its own `consumer` count skipping pads, and publishes `consumer` with a release
 * store so the host sees the room freed.
 */

#include "RgfxReplyReader.hpp"

#include <algorithm>

#include <cstring>

namespace {

// The ring header fields, by their offsets in RgfxRingHeader (as ring.rs names them).
constexpr uint32_t kProducer = __builtin_offsetof(RgfxRingHeader, producer);
constexpr uint32_t kConsumer = __builtin_offsetof(RgfxRingHeader, consumer);
constexpr uint32_t kArmed = __builtin_offsetof(RgfxRingHeader, armed);

// A non-pad message has at least its 16-byte header; a pad has an 8-byte one. Sizes
// are a multiple of 8.
constexpr uint32_t kHeaderBytes = sizeof(RgfxMsgHeader);
constexpr uint32_t kPadBytes = 8;
constexpr uint32_t kMsgAlign = 8;

uint64_t load64(const volatile uint8_t *p)
{
	return __atomic_load_n(reinterpret_cast<const volatile uint64_t *>(p), __ATOMIC_ACQUIRE);
}

void store64(volatile uint8_t *p, uint64_t value)
{
	__atomic_store_n(reinterpret_cast<volatile uint64_t *>(p), value, __ATOMIC_RELEASE);
}

// Copies `len` bytes out of the ring at byte offset `at` (already within the data
// area, no wrap). The region is write-back memory; the acquire load of `producer`
// ordered this read after the host's write of the bytes.
void readBytes(const volatile uint8_t *data, uint32_t at, void *dst, uint32_t len)
{
	std::memcpy(dst, const_cast<const uint8_t *>(data + at), len);
}

} // namespace

RgfxReplyReader::RgfxReplyReader(volatile uint8_t *ring_base, uint32_t data_size)
    : m_header(ring_base), m_data(ring_base + RGFX_RING_HEADER_SIZE), m_dataSize(data_size),
      m_mask(data_size - 1), m_consumed(0), m_nextToken(1)
{
}

void RgfxReplyReader::expectCompletion(uint64_t seqno, CompletionHandler handler)
{
	std::lock_guard<std::mutex> guard(m_lock);
	m_completions[seqno] = std::move(handler);
}

void RgfxReplyReader::expectScheduled(uint64_t seqno, ScheduledHandler handler)
{
	std::lock_guard<std::mutex> guard(m_lock);
	m_scheduled[seqno] = std::move(handler);
}

void RgfxReplyReader::expectSubmitted(uint64_t seqno, SubmittedHandler handler)
{
	std::lock_guard<std::mutex> guard(m_lock);
	m_submitted[seqno] = std::move(handler);
}

RgfxReplyReader::RgfxReplyReader()
    : m_header(nullptr), m_data(nullptr), m_dataSize(0), m_mask(0), m_consumed(0), m_nextToken(1)
{
}

void RgfxReplyReader::expectReflection(uint64_t seqno, ReflectionHandler handler)
{
	RgfxPipelineReflection early = {};
	{
		std::lock_guard<std::mutex> guard(m_lock);
		auto it = m_earlyReflections.find(seqno);
		if (it == m_earlyReflections.end()) {
			m_reflections[seqno] = std::move(handler);
			return;
		}
		early = it->second;
		m_earlyReflections.erase(it);
	}
	handler(early);
}

void RgfxReplyReader::deliver(const void *message, uint32_t size)
{
	RgfxPipelineReflection msg = {};
	if (size < sizeof(msg))
		return;
	std::memcpy(&msg, message, sizeof(msg));
	if (msg.header.type != RGFX_EVT_PIPELINE_REFLECTION)
		return;
	ReflectionHandler handler;
	{
		std::lock_guard<std::mutex> guard(m_lock);
		auto it = m_reflections.find(msg.header.seqno);
		if (it == m_reflections.end()) {
			m_earlyReflections[msg.header.seqno] = msg;
			return;
		}
		handler = std::move(it->second);
		m_reflections.erase(it);
	}
	handler(msg);
}

uint64_t RgfxReplyReader::addEventListener(uint64_t obj_id, EventHandler handler)
{
	std::lock_guard<std::mutex> guard(m_lock);
	uint64_t token = m_nextToken++;
	m_eventListeners[token] = EventListener{ obj_id, std::move(handler) };
	return token;
}

void RgfxReplyReader::removeEventListener(uint64_t token)
{
	std::lock_guard<std::mutex> guard(m_lock);
	m_eventListeners.erase(token);
}

void RgfxReplyReader::setEventsTo(RgfxReplyReader *events)
{
	std::lock_guard<std::mutex> guard(m_lock);
	m_eventsTo = events;
}

void RgfxReplyReader::setRoomEvents(std::shared_ptr<RgfxRingRoomEvents> room)
{
	std::lock_guard<std::mutex> guard(m_lock);
	m_room = std::move(room);
}

std::vector<RgfxReplyReader::EventHandler> RgfxReplyReader::listenersOf(uint64_t obj_id)
{
	// In registration order: the tokens rise with each registration.
	std::vector<std::pair<uint64_t, EventHandler>> found;
	for (const auto &entry : m_eventListeners) {
		if (entry.second.obj_id == obj_id)
			found.emplace_back(entry.first, entry.second.handler);
	}
	std::sort(found.begin(), found.end(),
	          [](const auto &a, const auto &b) { return a.first < b.first; });
	std::vector<EventHandler> handlers;
	for (auto &f : found)
		handlers.push_back(std::move(f.second));
	return handlers;
}

void RgfxReplyReader::signalled(const RgfxEventSignalled &msg)
{
	std::vector<EventHandler> handlers;
	{
		std::lock_guard<std::mutex> guard(m_lock);
		handlers = listenersOf(msg.obj_id);
	}
	for (EventHandler &handler : handlers)
		handler(msg);
}

bool RgfxReplyReader::armAndCheck()
{
	if (m_header == nullptr)
		return false;
	std::lock_guard<std::mutex> guard(m_lock);
	if (m_closed)
		return false;
	// The host's post publishes `producer`, fences, then reads and clears `armed`; this
	// sets `armed`, fences, then reads `producer`. One of the two sees the other's write:
	// either the host raises the vector, or this sees the record and pumps again.
	__atomic_store_n(reinterpret_cast<volatile uint32_t *>(m_header + kArmed), 1u,
	                 __ATOMIC_SEQ_CST);
	__atomic_thread_fence(__ATOMIC_SEQ_CST);
	uint64_t producer = load64(m_header + kProducer);
	return producer != m_consumed;
}

bool RgfxReplyReader::holdsSubmitted(uint64_t seqno)
{
	if (m_header == nullptr)
		return false;
	std::lock_guard<std::mutex> guard(m_lock);
	if (m_closed)
		return false;
	// The records from the consumer count to `producer` are the guest's until it takes them,
	// so the host writes none of their bytes while this reads them; the count moves only under
	// the lock, which this holds.
	const uint64_t producer = load64(m_header + kProducer);
	uint64_t at = m_consumed;
	for (uint32_t looked = 0; at != producer && looked < kLookRecords; looked++) {
		const uint32_t offset = static_cast<uint32_t>(at) & m_mask;
		uint8_t head[8];
		readBytes(m_data, offset, head, sizeof(head));
		uint32_t type = 0;
		uint32_t size = 0;
		std::memcpy(&type, head, 4);
		std::memcpy(&size, head + 4, 4);
		const uint32_t least = type == RGFX_MSG_PAD ? kPadBytes : kHeaderBytes;
		if (size % kMsgAlign != 0 || size < least || size > m_dataSize)
			return false;
		if (type == RGFX_EVT_CB_SUBMITTED || type == RGFX_EVT_CB_SCHEDULED ||
		    type == RGFX_EVT_CB_COMPLETE) {
			RgfxMsgHeader header = {};
			readBytes(m_data, offset, &header, sizeof(header));
			if (header.seqno == seqno)
				return true;
		}
		at += size;
	}
	return false;
}

void RgfxReplyReader::close()
{
	std::lock_guard<std::mutex> guard(m_lock);
	m_closed = true;
}

bool RgfxReplyReader::awaitingCompletion(uint64_t seqno)
{
	std::lock_guard<std::mutex> guard(m_lock);
	return m_completions.find(seqno) != m_completions.end();
}

bool RgfxReplyReader::awaitingScheduled(uint64_t seqno)
{
	std::lock_guard<std::mutex> guard(m_lock);
	return m_scheduled.find(seqno) != m_scheduled.end();
}

bool RgfxReplyReader::awaitingSubmitted(uint64_t seqno)
{
	std::lock_guard<std::mutex> guard(m_lock);
	return m_submitted.find(seqno) != m_submitted.end();
}

bool RgfxReplyReader::awaitingReflection(uint64_t seqno)
{
	std::lock_guard<std::mutex> guard(m_lock);
	return m_reflections.find(seqno) != m_reflections.end();
}

uint32_t RgfxReplyReader::pump()
{
	// Collect the matched handlers and their messages under the lock, then invoke them
	// after releasing it (a handler may register again or release its wrapper without
	// re-entering the reader's lock).
	std::vector<std::function<void()>> calls;
	if (m_header == nullptr)
		return 0;
	{
		std::lock_guard<std::mutex> guard(m_lock);
		if (m_closed)
			return 0;
		uint64_t producer = load64(m_header + kProducer);
		while (m_consumed != producer) {
			uint32_t offset = static_cast<uint32_t>(m_consumed) & m_mask;
			// Read `type` and `size` first (they fit before the ring's end, as the
			// host pads so a message never wraps).
			uint32_t type = 0;
			uint32_t size = 0;
			{
				uint8_t head[8];
				readBytes(m_data, offset, head, sizeof(head));
				std::memcpy(&type, head, 4);
				std::memcpy(&size, head + 4, 4);
			}
			uint32_t least = type == RGFX_MSG_PAD ? kPadBytes : kHeaderBytes;
			// A broken size would spin or read out of the ring; stop and publish what
			// was taken (the host decoder treats such a ring as broken too).
			if (size % kMsgAlign != 0 || size < least || size > m_dataSize) {
				break;
			}
			if (type == RGFX_MSG_PAD) {
				m_consumed += size;
				continue;
			}

			switch (type) {
			case RGFX_EVT_CB_SUBMITTED: {
				RgfxCbSubmitted msg = {};
				if (size >= sizeof(msg)) {
					readBytes(m_data, offset, &msg, sizeof(msg));
					auto it = m_submitted.find(msg.header.seqno);
					if (it != m_submitted.end()) {
						SubmittedHandler handler = std::move(it->second);
						m_submitted.erase(it);
						calls.emplace_back([handler, msg]() { handler(msg); });
					}
				}
				break;
			}
			case RGFX_EVT_CB_SCHEDULED: {
				RgfxCbScheduled msg = {};
				if (size >= sizeof(msg)) {
					readBytes(m_data, offset, &msg, sizeof(msg));
					// The scheduling stands for a CB_SUBMITTED the host did not post.
					m_submitted.erase(msg.header.seqno);
					auto it = m_scheduled.find(msg.header.seqno);
					if (it != m_scheduled.end()) {
						ScheduledHandler handler = std::move(it->second);
						m_scheduled.erase(it);
						calls.emplace_back([handler, msg]() { handler(msg); });
					}
				}
				break;
			}
			case RGFX_EVT_CB_COMPLETE: {
				RgfxCbComplete msg = {};
				if (size >= sizeof(msg)) {
					readBytes(m_data, offset, &msg, sizeof(msg));
					// The completion stands for a CB_SUBMITTED and a CB_SCHEDULED the host
					// did not post.
					m_submitted.erase(msg.header.seqno);
					m_scheduled.erase(msg.header.seqno);
					auto it = m_completions.find(msg.header.seqno);
					if (it != m_completions.end()) {
						CompletionHandler handler = std::move(it->second);
						m_completions.erase(it);
						calls.emplace_back([handler, msg]() { handler(msg); });
					}
				}
				break;
			}
			case RGFX_EVT_PIPELINE_REFLECTION: {
				RgfxPipelineReflection msg = {};
				if (size >= sizeof(msg)) {
					readBytes(m_data, offset, &msg, sizeof(msg));
					auto it = m_reflections.find(msg.header.seqno);
					if (it != m_reflections.end()) {
						ReflectionHandler handler = std::move(it->second);
						m_reflections.erase(it);
						calls.emplace_back([handler, msg]() { handler(msg); });
					}
				}
				break;
			}
			case RGFX_EVT_SUBMIT_ROOM:
				// The host took from the queue's submission ring, whose producer waits for room.
				if (m_room != nullptr) {
					std::shared_ptr<RgfxRingRoomEvents> room = m_room;
					calls.emplace_back([room]() { room->arrived(); });
				}
				break;
			case RGFX_EVT_EVENT_SIGNALLED: {
				RgfxEventSignalled msg = {};
				if (size >= sizeof(msg)) {
					readBytes(m_data, offset, &msg, sizeof(msg));
					if (m_eventsTo != nullptr) {
						RgfxReplyReader *to = m_eventsTo;
						calls.emplace_back([to, msg]() { to->signalled(msg); });
					} else {
						for (EventHandler &handler : listenersOf(msg.obj_id))
							calls.emplace_back([handler, msg]() { handler(msg); });
					}
				}
				break;
			}
			default:
				// A control-command answer (COMPLETE / DISPLAYS / VBLANK) or a type
				// this reader does not own: skip its bytes. (Follow-up: a reply type
				// the host posts that is not yet dispatched here lands in this arm.)
				break;
			}
			m_consumed += size;
		}
		// Publish the consumer count so the host's producer sees the room freed.
		store64(m_header + kConsumer, m_consumed);
	}

	for (auto &call : calls) {
		call();
	}
	return static_cast<uint32_t>(calls.size());
}
