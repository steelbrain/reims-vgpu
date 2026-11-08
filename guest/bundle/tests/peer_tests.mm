/*
 * peer_tests: the wire round trip of the bundle's own ring code against the host's library,
 * through its C API as QEMU drives it (`make peer-test`). The bundle's half is its real code: the
 * serialiser, a command queue made as the device makes one (its rings placed by RgfxRegion, its
 * submission ring's RgfxRingWriter, its completion ring's RgfxReplyReader), the completion thread
 * (RgfxCompletionThread), and the command buffers' commit and wait. The kext's half is a stand-in
 * here (Kext, below), which does what the accelerator kext does for a process: the device's start,
 * the object lifecycle and the queues' creates on the control ring, each block's grant and release,
 * the doorbells, and the wake of the completion thread when the device's vector shows a record on
 * one of its queues' rings. The host is the library itself, on this Mac's GPU, with no window.
 *
 * It records what crosses the wire as lines of a trace, as crates/reims-graphics's FakeGuest does
 * (src/device/tests/peer.rs):
 *   - every register read and write, doorbells among them;
 *   - every record the guest publishes on the control ring and on a queue's submission ring;
 *   - every record the host publishes on the control ring, as the kext takes it, and on a
 *     queue's completion ring, in ring order where the traffic has settled, with a command
 *     buffer's GPU times as `T` where the host gave them;
 *   - each ring header where the traffic has settled, every nonzero word of it.
 * Interrupts and the completion thread's wakes are counted, not traced: the host posts a command
 * buffer's CB_SCHEDULED and its CB_COMPLETE close together, and whether the two raise the vector
 * once or twice depends on when the GPU finishes. The count of each run is printed. The Rust
 * FakeGuest traces each interrupt exactly, against a host that answers at once.
 * The script's trace is kept in tests/peer_wire2.trace. The run fails where the traffic differs
 * from it; RGFX_PEER_RECORD=1 records it again from this build, and its diff is reviewed with the
 * change. The script runs three times on three devices, and each run's trace must be the same.
 *
 * The script fixes every choice the wire leaves to timing:
 *   - each step waits for the traffic to settle before the next: every submission ring taken and
 *     idle (so each post rings its doorbell), every completion ring taken and armed, every
 *     interrupt serviced, and the completion thread asleep in its wait;
 *   - the control ring's answers are taken by the stand-in's service thread, as the kext's
 *     interrupt handler takes them, only on an interrupt.
 *
 * Usage: peer_tests TRACE (`make peer-test`). RGFX_PEER_DUMP=1 also prints the first run's trace
 * to stderr. Our own code; the library's C API and the committed wire header.
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <condition_variable>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <map>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#import "RgfxCommandObjects.h"
#include "RgfxCompletionThread.hpp"
#include "RgfxObjectModel.hpp"
#include "RgfxRegion.hpp"
#include "RgfxReplyReader.hpp"
#include "RgfxRingWriter.hpp"
#include "RgfxSerializer.hpp"
#include "reims_graphics_wire.h"
#include "reims_vgpu.h"

namespace {

// The process's context number, as the kext gives the first process.
constexpr uint32_t kContext = 1;
// The queues' ring sizes, as the device's (RgfxMetalDevice+CommandBuffer.mm).
constexpr uint32_t kQueueSubmitDataSize = 32768;
constexpr uint32_t kQueueCompleteDataSize = 32768;
// The largest lifecycle message the kext takes (RgfxMetalDevice.mm).
constexpr uint32_t kLifecycleMessageMax = RGFX_CONTROL_RING_DATA_SIZE / 2;
// A block's granularity in the kext (RgfxBlockTable.hpp).
constexpr uint64_t kBlockPage = 0x1000;

[[noreturn]] void fail(const char *fmt, ...)
{
	va_list ap;
	va_start(ap, fmt);
	std::fprintf(stderr, "peer_tests: ");
	std::vfprintf(stderr, fmt, ap);
	std::fprintf(stderr, "\n");
	va_end(ap);
	std::exit(1);
}

std::string format(const char *fmt, ...)
{
	char buf[1024];
	va_list ap;
	va_start(ap, fmt);
	std::vsnprintf(buf, sizeof buf, fmt, ap);
	va_end(ap);
	return buf;
}

// Waits about 5 s for `done`, failing with `what` where it never holds.
void waitFor(const char *what, const std::function<bool()> &done)
{
	for (int i = 0; i < 5000; i++) {
		if (done())
			return;
		std::this_thread::sleep_for(std::chrono::milliseconds(1));
	}
	if (!done())
		fail("%s", what);
}

uint64_t load64(volatile uint8_t *p)
{
	return __atomic_load_n(reinterpret_cast<volatile uint64_t *>(p), __ATOMIC_SEQ_CST);
}
uint32_t load32(volatile uint8_t *p)
{
	return __atomic_load_n(reinterpret_cast<volatile uint32_t *>(p), __ATOMIC_SEQ_CST);
}
void store64(volatile uint8_t *p, uint64_t v)
{
	__atomic_store_n(reinterpret_cast<volatile uint64_t *>(p), v, __ATOMIC_SEQ_CST);
}
void store32(volatile uint8_t *p, uint32_t v)
{
	__atomic_store_n(reinterpret_cast<volatile uint32_t *>(p), v, __ATOMIC_SEQ_CST);
}

constexpr uint64_t kProducer = offsetof(RgfxRingHeader, producer);
constexpr uint64_t kConsumer = offsetof(RgfxRingHeader, consumer);
constexpr uint64_t kConsumerIdle = offsetof(RgfxRingHeader, consumer_idle);
constexpr uint64_t kArmed = offsetof(RgfxRingHeader, armed);

std::string registerName(uint32_t offset)
{
	switch (offset) {
	case RGFX_REG_MAGIC: return "MAGIC";
	case RGFX_REG_VERSION: return "VERSION";
	case RGFX_REG_FEATURES_LO: return "FEATURES_LO";
	case RGFX_REG_FEATURES_HI: return "FEATURES_HI";
	case RGFX_REG_STATUS: return "STATUS";
	case RGFX_REG_REGION_SIZE_LO: return "REGION_SIZE_LO";
	case RGFX_REG_REGION_SIZE_HI: return "REGION_SIZE_HI";
	case RGFX_REG_DOORBELL: return "DOORBELL";
	default: return format("0x%03x", offset);
	}
}

std::string typeName(uint32_t type)
{
	switch (type) {
	case RGFX_MSG_PAD: return "PAD";
	case RGFX_CMD_QUEUE_CREATE: return "QUEUE_CREATE";
	case RGFX_CMD_FENCE_CREATE: return "FENCE_CREATE";
	case RGFX_CMD_OBJECT_DESTROY: return "OBJECT_DESTROY";
	case RGFX_CMD_CONTEXT_CLOSE: return "CONTEXT_CLOSE";
	case RGFX_CMD_BLOCK_GRANT: return "BLOCK_GRANT";
	case RGFX_CMD_BLOCK_RELEASE: return "BLOCK_RELEASE";
	case RGFX_CMD_COMMAND_BUFFER: return "COMMAND_BUFFER";
	case RGFX_CMD_COMMAND_BUFFER_IN_REGION: return "COMMAND_BUFFER_IN_REGION";
	case RGFX_EVT_COMPLETE: return "COMPLETE";
	case RGFX_EVT_CB_COMPLETE: return "CB_COMPLETE";
	case RGFX_EVT_CB_SUBMITTED: return "CB_SUBMITTED";
	case RGFX_EVT_CB_SCHEDULED: return "CB_SCHEDULED";
	default: return format("0x%x", type);
	}
}

// The 8-byte words of a record past its 16-byte header, little-endian, in hex; a command
// buffer's GPU times as `T` where the host gave them (they differ from run to run).
std::string bodyWords(const std::vector<uint8_t> &bytes)
{
	uint32_t type = 0;
	std::memcpy(&type, bytes.data(), 4);
	std::string out;
	for (size_t at = sizeof(RgfxMsgHeader); at < bytes.size(); at += 8) {
		uint64_t word = 0;
		std::memcpy(&word, bytes.data() + at, std::min<size_t>(8, bytes.size() - at));
		bool time = type == RGFX_EVT_CB_COMPLETE && (at == offsetof(RgfxCbComplete, gpu_start_ns) ||
		                                              at == offsetof(RgfxCbComplete, gpu_end_ns));
		if (!out.empty())
			out += ' ';
		out += time && word != 0 ? std::string("T") : format("0x%llx", (unsigned long long)word);
	}
	return out;
}

std::string recordLine(const char *verb, const std::string &ring, const std::vector<uint8_t> &bytes)
{
	RgfxMsgHeader h = {};
	std::memcpy(&h, bytes.data(), sizeof(h));
	return format("%s %s %s seqno=%llu size=%u %s", verb, ring.c_str(), typeName(h.type).c_str(),
	              (unsigned long long)h.seqno, h.size, bodyWords(bytes).c_str());
}

// The trace, appended to from the test's thread, the stand-in's service thread and the
// library's threads (the interrupts).
class Trace {
public:
	void note(std::string line)
	{
		while (!line.empty() && line.back() == ' ')
			line.pop_back();
		std::lock_guard<std::mutex> hold(m_lock);
		m_lines.push_back(std::move(line));
	}

	std::string text()
	{
		std::lock_guard<std::mutex> hold(m_lock);
		std::string out;
		for (const std::string &l : m_lines)
			out += l + "\n";
		return out;
	}

private:
	std::mutex m_lock;
	std::vector<std::string> m_lines;
};

// One of the process's queues as the kext watches it.
struct WatchedQueue {
	std::string name;
	RgfxQueueCreate create;
	uint64_t submitLogged = 0;   // the submission ring's count the trace has shown
	uint64_t lastProducer = 0;   // the completion ring's producer at the kext's last look
	uint64_t completeLogged = 0; // the completion ring's count the trace has shown
};

// The stand-in for the accelerator kext and the framebuffer driver: the device, for one process.
class Kext {
public:
	explicit Kext(Trace &trace) : m_trace(trace)
	{
		RvgGraphicsCallbacks callbacks = {};
		callbacks.struct_size = sizeof(callbacks);
		callbacks.ctx = this;
		callbacks.raise_interrupt = &Kext::raise;
		RvgGraphicsDeviceConfig config = {};
		config.struct_size = sizeof(config);
		config.callbacks = &callbacks;
		if (rvg_graphics_device_create(&config, &m_device) != RVG_OK || m_device == nullptr)
			fail("no graphics device");
		m_region = static_cast<volatile uint8_t *>(rvg_graphics_device_region(m_device, &m_size));
		m_service = std::thread([this] { serviceLoop(); });
	}

	~Kext()
	{
		{
			// The process's completion thread let out of its wait, and out of this, first.
			std::unique_lock<std::mutex> hold(m_lock);
			m_stopping = true;
			m_closed = true;
			m_changed.notify_all();
			m_changed.wait(hold, [&] { return !m_waiting; });
		}
		m_service.join();
		rvg_graphics_device_release(m_device);
	}

	volatile uint8_t *region() const { return m_region; }
	uint64_t regionSize() const { return m_size; }

	uint32_t read(uint32_t offset)
	{
		uint32_t value = rvg_graphics_device_mmio_read(m_device, offset);
		m_trace.note(format("read %s = 0x%x", registerName(offset).c_str(), value));
		return value;
	}

	// A write as the vCPU makes it (QEMU's write path).
	void write(uint32_t offset, uint32_t value)
	{
		m_trace.note(format("write %s = 0x%x", registerName(offset).c_str(), value));
		rvg_graphics_device_mmio_write_taking(m_device, offset, value);
	}

	// The framebuffer driver's start (ReimsGraphics::start): the device checked, reset, the
	// control rings' counts taken as they stand, then the driver ready.
	void start()
	{
		if (read(RGFX_REG_MAGIC) != RGFX_MAGIC || read(RGFX_REG_VERSION) != RGFX_VERSION)
			fail("not the device");
		uint64_t size = (static_cast<uint64_t>(read(RGFX_REG_REGION_SIZE_HI)) << 32) |
		                read(RGFX_REG_REGION_SIZE_LO);
		if (size != m_size)
			fail("the region is 0x%llx bytes, the register says 0x%llx",
			     (unsigned long long)m_size, (unsigned long long)size);
		read(RGFX_REG_FEATURES_LO);
		read(RGFX_REG_FEATURES_HI);
		write(RGFX_REG_STATUS, RGFX_STATUS_RESET);
		m_produced = load64(ring0Submit() + kProducer);
		m_consumed = load64(ring0Complete() + kConsumer);
		write(RGFX_REG_STATUS, RGFX_STATUS_DRIVER_READY);
		read(RGFX_REG_STATUS);
		// The control ring armed, as the service leaves it.
		armControl();
	}

	// The device reset the kext's stop writes.
	void reset()
	{
		write(RGFX_REG_STATUS, RGFX_STATUS_RESET);
		read(RGFX_REG_STATUS);
	}

	// Posts `message` (its header's type set) on the control ring as the kext's postLifecycle
	// does, stamping its size and seqno, and waits for its answer of `replyType` where that is
	// not 0. Waits first for the host to take what came before and sleep, so the doorbell is
	// rung. The message's seqno.
	uint64_t post(std::vector<uint8_t> message, uint32_t replyType, std::vector<uint8_t> *reply)
	{
		waitFor("the host never took the control ring", [&] { return controlIdle(); });
		uint64_t seqno;
		{
			std::lock_guard<std::mutex> posting(m_postLock);
			const uint64_t data = RGFX_CONTROL_RING_DATA_SIZE;
			RgfxMsgHeader h = {};
			std::memcpy(&h, message.data(), sizeof(h));
			h.size = static_cast<uint32_t>(message.size());
			h.seqno = seqno = ++m_seqno;
			std::memcpy(message.data(), &h, sizeof(h));
			uint64_t left = data - m_produced % data;
			if (left < message.size()) {
				uint32_t pad[2] = {RGFX_MSG_PAD, static_cast<uint32_t>(left)};
				std::memcpy(const_cast<uint8_t *>(ring0Submit() + RGFX_RING_HEADER_SIZE +
				                                  m_produced % data),
				            pad, sizeof pad);
				m_produced += left;
				m_trace.note(format("post ring0 PAD size=%llu", (unsigned long long)left));
			}
			std::memcpy(const_cast<uint8_t *>(ring0Submit() + RGFX_RING_HEADER_SIZE +
			                                  m_produced % data),
			            message.data(), message.size());
			m_produced += message.size();
			if (replyType != 0) {
				std::lock_guard<std::mutex> hold(m_lock);
				m_waiters[seqno] = Waiter{replyType, {}, false};
			}
			m_trace.note(recordLine("post", "ring0", message));
			store64(ring0Submit() + kProducer, m_produced);
			__atomic_thread_fence(__ATOMIC_SEQ_CST);
			if (load32(ring0Submit() + kConsumerIdle) == 1)
				write(RGFX_REG_DOORBELL, RGFX_CONTROL_RING_ID);
		}
		if (replyType == 0)
			return seqno;
		std::unique_lock<std::mutex> hold(m_lock);
		if (!m_changed.wait_for(hold, std::chrono::seconds(5),
		                        [&] { return m_waiters[seqno].done; }))
			fail("no answer to the control ring's seqno %llu", (unsigned long long)seqno);
		if (reply)
			*reply = std::move(m_waiters[seqno].reply);
		m_waiters.erase(seqno);
		return seqno;
	}

	// The process's context's close (the kext's, at the process's end), answered.
	uint32_t closeContext()
	{
		RgfxContextClose close = {};
		close.header.type = RGFX_CMD_CONTEXT_CLOSE;
		close.context = kContext;
		std::vector<uint8_t> reply;
		post(bytesOf(close), RGFX_EVT_COMPLETE, &reply);
		return statusOf(reply);
	}

	// --- the process's calls, as RgfxRegion's, RgfxSerializer's, RgfxRingWriter's and
	// --- RgfxCompletionThread's hooks take them

	// The kext's alloc: a block placed past the last, a page at least, its grant on the control
	// ring before the process learns of it.
	static bool alloc(void *context, uint64_t size, uint64_t align, uint64_t *offset)
	{
		Kext *k = static_cast<Kext *>(context);
		uint64_t a = std::max(align, kBlockPage);
		uint64_t at = (k->m_nextBlock + a - 1) & ~(a - 1);
		uint64_t bytes = (size + kBlockPage - 1) & ~(kBlockPage - 1);
		if (at + bytes > k->m_size)
			return false;
		k->m_nextBlock = at + bytes;
		k->m_blocks[at] = bytes;
		RgfxBlockGrant grant = {};
		grant.header.type = RGFX_CMD_BLOCK_GRANT;
		grant.context = kContext;
		grant.offset = at;
		grant.size = bytes;
		k->post(bytesOf(grant), 0, nullptr);
		*offset = at;
		return true;
	}

	// The kext's free: the block's release on the control ring, answered.
	static void release(void *context, uint64_t offset)
	{
		Kext *k = static_cast<Kext *>(context);
		auto it = k->m_blocks.find(offset);
		if (it == k->m_blocks.end())
			fail("a release of no block, at 0x%llx", (unsigned long long)offset);
		RgfxBlockRelease release = {};
		release.header.type = RGFX_CMD_BLOCK_RELEASE;
		release.context = kContext;
		release.offset = offset;
		release.size = it->second;
		k->m_blocks.erase(it);
		std::vector<uint8_t> reply;
		k->post(bytesOf(release), RGFX_EVT_COMPLETE, &reply);
		if (statusOf(reply) != RGFX_STATUS_OK)
			fail("the release of 0x%llx was refused", (unsigned long long)offset);
	}

	// The kext's submit: one lifecycle message on the control ring, waiting where the device
	// answers it (RgfxAcceleratorUserClient::submitMethod's table), its COMPLETE's status handed
	// back and any other answer to the process's lifecycle reader.
	static bool forward(void *context, const void *message, uint32_t size, uint64_t *out_seqno,
	                    uint32_t *out_status)
	{
		Kext *k = static_cast<Kext *>(context);
		RgfxMsgHeader h = {};
		std::memcpy(&h, message, sizeof(h));
		uint32_t replyType = 0;
		switch (h.type) {
		case RGFX_CMD_RENDER_PIPELINE_CREATE:
		case RGFX_CMD_COMPUTE_PIPELINE_CREATE: replyType = RGFX_EVT_PIPELINE_REFLECTION; break;
		case RGFX_CMD_HEAP_TEXTURE_SIZE_AND_ALIGN: replyType = RGFX_EVT_SIZE_AND_ALIGN; break;
		case RGFX_CMD_SAMPLER_CREATE: replyType = RGFX_EVT_SAMPLER_STATE; break;
		case RGFX_CMD_LIBRARY_CREATE:
		case RGFX_CMD_LIBRARY_LOOKUP:
		case RGFX_CMD_ARGUMENT_LAYOUT: replyType = RGFX_EVT_COMPLETE; break;
		case RGFX_CMD_LIBRARY_FROM_REGION: fail("the script streams no library");
		default: break;
		}
		const uint8_t *bytes = static_cast<const uint8_t *>(message);
		std::vector<uint8_t> reply;
		uint64_t seqno = k->post(std::vector<uint8_t>(bytes, bytes + size), replyType, &reply);
		if (out_seqno)
			*out_seqno = seqno;
		if (out_status)
			*out_status = replyType == RGFX_EVT_COMPLETE ? statusOf(reply) : kRgfxNoAnswer;
		if (replyType != 0 && replyType != RGFX_EVT_COMPLETE && k->m_lifecycle)
			k->m_lifecycle->deliver(reply.data(), static_cast<uint32_t>(reply.size()));
		return true;
	}

	// The kext's QueueCreate: the queue's completion ring watched, then its create on the
	// control ring with the device's one vector, answered.
	static bool openQueue(void *context, const RgfxQueueCreate &q)
	{
		Kext *k = static_cast<Kext *>(context);
		RgfxQueueCreate open = q;
		open.header.type = RGFX_CMD_QUEUE_CREATE;
		open.vector = RGFX_CONTROL_VECTOR;
		{
			std::lock_guard<std::mutex> hold(k->m_lock);
			WatchedQueue w;
			w.name = format("queue%zu", ++k->m_queuesMade);
			w.create = open;
			k->m_queues.push_back(w);
		}
		std::vector<uint8_t> reply;
		k->post(bytesOf(open), RGFX_EVT_COMPLETE, &reply);
		if (statusOf(reply) == RGFX_STATUS_OK)
			return true;
		k->unwatch(open.obj_id);
		return false;
	}

	// The kext's QueueDestroy: the queue's ring no longer watched, then its destroy.
	static void closeQueue(void *context, RgfxObjId objId)
	{
		Kext *k = static_cast<Kext *>(context);
		k->unwatch(objId);
		RgfxObjectDestroy destroy = {};
		destroy.header.type = RGFX_CMD_OBJECT_DESTROY;
		destroy.obj_id = objId;
		k->post(bytesOf(destroy), 0, nullptr);
	}

	// The kext's doorbell method, for a queue's submission ring: the records published since
	// the trace's last look, then the write.
	static void doorbell(void *context, uint32_t ringId)
	{
		Kext *k = static_cast<Kext *>(context);
		k->logSubmissions();
		k->write(RGFX_REG_DOORBELL, ringId);
	}

	// The kext's WaitCompletions: returns once the wake count is not `last`.
	static RgfxCompletionThread::WaitResult waitCompletions(void *context, uint64_t last,
	                                                        uint64_t *now)
	{
		Kext *k = static_cast<Kext *>(context);
		std::unique_lock<std::mutex> hold(k->m_lock);
		k->m_waiting = true;
		k->m_waitingFor = last;
		k->m_changed.notify_all();
		k->m_changed.wait(hold, [&] { return k->m_wakes != last || k->m_closed; });
		k->m_waiting = false;
		k->m_changed.notify_all();
		*now = k->m_wakes;
		return k->m_closed ? RgfxCompletionThread::kStop : RgfxCompletionThread::kWoke;
	}

	void setLifecycleReader(RgfxReplyReader *reader) { m_lifecycle = reader; }

	// The interrupts the host raised and the completion thread's wakes, so far.
	std::pair<uint64_t, uint64_t> counts()
	{
		std::lock_guard<std::mutex> hold(m_lock);
		return {m_raised, m_wakes};
	}

	// A process's completion thread runs from now on: the traffic settles only with it asleep in
	// its wait.
	void expectWaiter()
	{
		std::lock_guard<std::mutex> hold(m_lock);
		m_expectWaiter = true;
	}

	// Waits until the traffic settled, then shows the rings' headers: the control ring and each
	// watched queue's submission ring taken and idle, their completion rings taken and armed,
	// every interrupt serviced, and the completion thread asleep in its wait for the count.
	void settle()
	{
		waitFor("the traffic never settled", [&] { return settled(); });
		logSubmissions();
		logCompletions();
		std::vector<std::pair<std::string, uint64_t>> rings = {
		    {"ring0.submit", RGFX_CONTROL_SUBMIT_OFFSET},
		    {"ring0.complete", RGFX_CONTROL_COMPLETE_OFFSET},
		};
		{
			std::lock_guard<std::mutex> hold(m_lock);
			for (const WatchedQueue &q : m_queues) {
				rings.push_back({q.name + ".submit", q.create.submit_offset});
				rings.push_back({q.name + ".complete", q.create.complete_offset});
			}
		}
		for (const auto &[name, at] : rings) {
			std::string words;
			for (uint64_t w = 0; w < RGFX_RING_HEADER_SIZE; w += 8) {
				uint64_t word = load64(m_region + at + w);
				if (word != 0)
					words += format(" +0x%02llx=0x%llx", (unsigned long long)w,
					                (unsigned long long)word);
			}
			m_trace.note("header " + name + words);
		}
	}

private:
	struct Waiter {
		uint32_t type;
		std::vector<uint8_t> reply;
		bool done;
	};

	template <typename T> static std::vector<uint8_t> bytesOf(const T &message)
	{
		const uint8_t *p = reinterpret_cast<const uint8_t *>(&message);
		return std::vector<uint8_t>(p, p + sizeof(T));
	}

	static uint32_t statusOf(const std::vector<uint8_t> &reply)
	{
		RgfxComplete complete = {};
		if (reply.size() < sizeof(complete))
			return kRgfxNoAnswer;
		std::memcpy(&complete, reply.data(), sizeof(complete));
		return complete.status;
	}

	volatile uint8_t *ring0Submit() const { return m_region + RGFX_CONTROL_SUBMIT_OFFSET; }
	volatile uint8_t *ring0Complete() const { return m_region + RGFX_CONTROL_COMPLETE_OFFSET; }

	bool controlIdle() const
	{
		return load64(ring0Submit() + kConsumer) == m_produced &&
		       load32(ring0Submit() + kConsumerIdle) == 1;
	}

	bool settled()
	{
		if (!controlIdle() || load64(ring0Complete() + kProducer) != m_consumed ||
		    load32(ring0Complete() + kArmed) != 1)
			return false;
		std::lock_guard<std::mutex> hold(m_lock);
		if (m_raised != m_serviced || (m_expectWaiter && (!m_waiting || m_waitingFor != m_wakes)))
			return false;
		for (const WatchedQueue &q : m_queues) {
			volatile uint8_t *submit = m_region + q.create.submit_offset;
			volatile uint8_t *complete = m_region + q.create.complete_offset;
			if (load64(submit + kConsumer) != load64(submit + kProducer) ||
			    load32(submit + kConsumerIdle) != 1 ||
			    load64(complete + kConsumer) != load64(complete + kProducer) ||
			    load32(complete + kArmed) != 1)
				return false;
		}
		return true;
	}

	void unwatch(RgfxObjId objId)
	{
		std::lock_guard<std::mutex> hold(m_lock);
		for (auto it = m_queues.begin(); it != m_queues.end(); ++it) {
			if (it->create.obj_id == objId) {
				m_queues.erase(it);
				return;
			}
		}
	}

	// Each record of `ring` (its header at `at`, `data` bytes) from count `from` to `to`, in
	// the trace as `verb`, pads included.
	void logRecords(const char *verb, const std::string &name, uint64_t at, uint64_t data,
	                uint64_t from, uint64_t to)
	{
		while (from < to) {
			volatile uint8_t *rec = m_region + at + RGFX_RING_HEADER_SIZE + from % data;
			RgfxMsgHeader h = {};
			std::memcpy(&h, const_cast<uint8_t *>(rec), sizeof(h));
			if (h.size < 8 || h.size % 8)
				fail("a broken record on %s", name.c_str());
			if (h.type == RGFX_MSG_PAD) {
				m_trace.note(format("%s %s PAD size=%u", verb, name.c_str(), h.size));
			} else {
				std::vector<uint8_t> bytes(h.size);
				std::memcpy(bytes.data(), const_cast<uint8_t *>(rec), h.size);
				m_trace.note(recordLine(verb, name, bytes));
			}
			from += h.size;
		}
	}

	// The records the host published on the process's queues' completion rings since the
	// trace's last look.
	void logCompletions()
	{
		std::lock_guard<std::mutex> hold(m_lock);
		for (WatchedQueue &q : m_queues) {
			uint64_t produced = load64(m_region + q.create.complete_offset + kProducer);
			logRecords("take", q.name, q.create.complete_offset, q.create.complete_data_size,
			           q.completeLogged, produced);
			q.completeLogged = produced;
		}
	}

	// The records the process published on its queues' submission rings since the trace's
	// last look.
	void logSubmissions()
	{
		std::lock_guard<std::mutex> hold(m_lock);
		for (WatchedQueue &q : m_queues) {
			uint64_t produced = load64(m_region + q.create.submit_offset + kProducer);
			logRecords("post", q.name, q.create.submit_offset, q.create.submit_data_size,
			           q.submitLogged, produced);
			q.submitLogged = produced;
		}
	}

	static void raise(void *ctx, uint32_t vector)
	{
		Kext *k = static_cast<Kext *>(ctx);
		if (vector != RGFX_CONTROL_VECTOR)
			fail("an interrupt on vector %u", vector);
		std::lock_guard<std::mutex> hold(k->m_lock);
		k->m_raised++;
		k->m_changed.notify_all();
	}

	void armControl()
	{
		store32(ring0Complete() + kArmed, 1);
		__atomic_thread_fence(__ATOMIC_SEQ_CST);
	}

	// The framebuffer driver's service, on each interrupt: the control ring's answers taken and
	// handed to their waiters, the ring armed and looked at again; then the accelerator's look
	// at the queues' rings (RgfxAccelerator::queueInterrupt), which wakes the completion thread
	// for a ring whose producer moved and that holds records the process has not taken.
	void serviceLoop()
	{
		std::unique_lock<std::mutex> hold(m_lock);
		for (;;) {
			m_changed.wait(hold, [&] { return m_raised != m_serviced || m_stopping; });
			if (m_stopping)
				return;
			uint64_t target = m_raised;
			hold.unlock();
			const uint64_t data = RGFX_CONTROL_RING_DATA_SIZE;
			for (;;) {
				uint64_t produced = load64(ring0Complete() + kProducer);
				while (m_consumed != produced) {
					volatile uint8_t *rec =
					    ring0Complete() + RGFX_RING_HEADER_SIZE + m_consumed % data;
					RgfxMsgHeader h = {};
					std::memcpy(&h, const_cast<uint8_t *>(rec), sizeof(h));
					if (h.size < 8 || h.size % 8)
						fail("a broken record on the control completion ring");
					std::vector<uint8_t> bytes(h.size);
					std::memcpy(bytes.data(), const_cast<uint8_t *>(rec), h.size);
					m_consumed += h.size;
					store64(ring0Complete() + kConsumer, m_consumed);
					if (h.type == RGFX_MSG_PAD) {
						m_trace.note(format("take ring0 PAD size=%u", h.size));
						continue;
					}
					m_trace.note(recordLine("take", "ring0", bytes));
					std::lock_guard<std::mutex> waiters(m_lock);
					auto it = m_waiters.find(h.seqno);
					if (it != m_waiters.end() && it->second.type == h.type) {
						it->second.reply = std::move(bytes);
						it->second.done = true;
						m_changed.notify_all();
					}
				}
				armControl();
				if (load64(ring0Complete() + kProducer) == m_consumed)
					break;
			}
			hold.lock();
			for (WatchedQueue &q : m_queues) {
				volatile uint8_t *complete = m_region + q.create.complete_offset;
				uint64_t producer = load64(complete + kProducer);
				if (producer == q.lastProducer)
					continue;
				q.lastProducer = producer;
				if (producer != load64(complete + kConsumer))
					m_wakes++;
			}
			m_serviced = target;
			m_changed.notify_all();
		}
	}

	Trace &m_trace;
	RvgGraphicsDevice *m_device = nullptr;
	volatile uint8_t *m_region = nullptr;
	uint64_t m_size = 0;
	RgfxReplyReader *m_lifecycle = nullptr;

	std::mutex m_postLock; // the control ring's producer side
	uint64_t m_produced = 0;
	uint64_t m_seqno = 0;
	uint64_t m_consumed = 0; // the control completion ring's count (the service thread's)

	uint64_t m_nextBlock = RGFX_CONTEXT_ALLOCATIONS_OFFSET;
	std::map<uint64_t, uint64_t> m_blocks;

	std::mutex m_lock; // all below
	std::condition_variable m_changed;
	std::map<uint64_t, Waiter> m_waiters;
	std::vector<WatchedQueue> m_queues;
	size_t m_queuesMade = 0;
	uint64_t m_raised = 0;
	uint64_t m_serviced = 0;
	uint64_t m_wakes = 0;
	bool m_expectWaiter = false;
	bool m_waiting = false;
	uint64_t m_waitingFor = 0;
	bool m_closed = false;
	bool m_stopping = false;
	std::thread m_service;
};

// The process's device half: its ids, region, lifecycle route, serialiser and completion thread,
// as RgfxMetalDevice's initWithAcceleratorPort: sets them up over the kext.
struct Process {
	Kext &kext;
	RgfxIdSpace ids;
	RgfxRegion region;
	RgfxReplyReader lifecycle;
	RgfxRingWriter control;
	RgfxSerializer serializer;
	RgfxCompletionThread completions;
	bool asksSubmitted;

	explicit Process(Kext &k)
	    : kext(k), region(k.region(), k.regionSize()),
	      control(&Kext::forward, &k, kLifecycleMessageMax), serializer(control, ids),
	      completions(&Kext::waitCompletions, &k)
	{
		// The context's range as the kext hands it: a unit of GPU addresses.
		ids.setBases(static_cast<uint64_t>(kContext) << RGFX_CONTEXT_OBJ_ID_SHIFT,
		             RGFX_CONTEXT_VA_BASE + kContext * RGFX_CONTEXT_VA_UNIT,
		             RGFX_CONTEXT_VA_UNIT);
		region.setBlockSource(&Kext::alloc, &Kext::release, &k);
		k.setLifecycleReader(&lifecycle);
		serializer.setQueueOpener(&Kext::openQueue, &k);
		serializer.setQueueCloser(&Kext::closeQueue, &k);
		// The caps blob the host wrote, as the kext reads it for the process.
		RgfxDeviceInfo info = {};
		std::memcpy(&info, const_cast<uint8_t *>(k.region() + RGFX_DEVICE_INFO_OFFSET), sizeof info);
		asksSubmitted = (info.features_lo & RGFX_FEATURE_CB_SUBMITTED) != 0;
		if (!completions.start())
			fail("no completion thread");
		k.expectWaiter();
	}

	// A command queue, made as RgfxMetalDevice's newCommandQueueWithMaxCommandBufferCount: does.
	RgfxCommandQueue *newQueue()
	{
		RgfxRegion::RingPair rings =
		    region.allocateRingPair(kQueueSubmitDataSize, kQueueCompleteDataSize);
		if (!rings.ok)
			fail("no ring pair");
		auto *submit = new RgfxRingWriter(region.base() + rings.submit_offset, kQueueSubmitDataSize,
		                                  static_cast<uint32_t>(rings.submit_offset),
		                                  &Kext::doorbell, &kext);
		RgfxCommandQueue *queue = [RgfxCommandQueue queueWithSubmitRing:submit
		                                                   submitOffset:rings.submit_offset
		                                                 completeOffset:rings.complete_offset
		                                               completeDataSize:rings.complete_data_size
		                                                            ids:&ids
		                                                       maxCount:0
		                                                         device:nil
		                                                     serializer:&serializer];
		if (queue == nil)
			fail("no queue");
		RgfxRegion *r = &region;
		[queue rgfxSetRelease:[submit, r, rings] {
			delete submit;
			r->release(rings.submit_offset);
			r->release(rings.complete_offset);
		}];
		auto *reader =
		    new RgfxReplyReader(region.base() + rings.complete_offset, rings.complete_data_size);
		reader->setEventsTo(&lifecycle);
		[queue adoptReplyReader:reader];
		queue.region = &region;
		queue.asksSubmitted = asksSubmitted;
		completions.add([queue sharedReplyReader]);
		return queue;
	}
};

// Commits `cb` and waits for it; it must complete.
void commitAndWait(id<MTLCommandBuffer> cb)
{
	[cb commit];
	[cb waitUntilCompleted];
	if (cb.status != MTLCommandBufferStatusCompleted)
		fail("a command buffer ended with status %ld (%s)", (long)cb.status,
		     cb.error.description.UTF8String ?: "no error");
}

// The wire-2 bundle's script: what today's kext and bundle do for a process that makes a queue,
// commits two command buffers on it and goes.
// - the device's start;
// - the process's queue: its ring pair's blocks granted, its create, its barrier fences;
// - an empty command buffer, then one with a blit encoder and a compute encoder, each waited for;
// - the queue's end: its destroy, its blocks released;
// - the context's close, and the kext's reset.
void wire2Script(Kext &kext)
{
	kext.start();
	kext.settle();
	Process process(kext);
	kext.settle();
	@autoreleasepool {
		RgfxCommandQueue *queue = process.newQueue();
		kext.settle();
		commitAndWait([queue commandBuffer]);
		kext.settle();
		id<MTLCommandBuffer> cb = [queue commandBuffer];
		[[cb blitCommandEncoder] endEncoding];
		[[cb computeCommandEncoder] endEncoding];
		commitAndWait(cb);
		kext.settle();
		cb = nil;
		queue = nil;
	}
	kext.settle();
	if (kext.closeContext() != RGFX_STATUS_OK)
		fail("the context's close was refused");
	kext.settle();
	kext.reset();
}

std::string traceOf(void (*script)(Kext &), int run)
{
	Trace trace;
	{
		Kext kext(trace);
		script(kext);
		auto [interrupts, wakes] = kext.counts();
		std::printf("peer_tests: run %d: %llu interrupts, %llu wakes of the completion thread\n", run,
		            (unsigned long long)interrupts, (unsigned long long)wakes);
	}
	return trace.text();
}

// The line where `a` and `b` first differ, 1-based.
size_t firstDifference(const std::string &a, const std::string &b)
{
	std::istringstream sa(a), sb(b);
	std::string la, lb;
	for (size_t n = 1;; n++) {
		bool ga = static_cast<bool>(std::getline(sa, la));
		bool gb = static_cast<bool>(std::getline(sb, lb));
		if (!ga || !gb || la != lb)
			return n;
	}
}

std::string lineOf(const std::string &text, size_t n)
{
	std::istringstream s(text);
	std::string line;
	for (size_t i = 0; i < n && std::getline(s, line); i++) {
	}
	return line;
}

} // namespace

int main(int argc, char **argv)
{
	if (argc != 2) {
		std::fprintf(stderr, "usage: peer_tests TRACE\n");
		return 2;
	}
	const char *path = argv[1];
	std::string first = traceOf(&wire2Script, 1);
	if (std::getenv("RGFX_PEER_DUMP"))
		std::fputs(first.c_str(), stderr);
	for (int run = 2; run <= 3; run++) {
		std::string again = traceOf(&wire2Script, run);
		if (again != first) {
			size_t n = firstDifference(first, again);
			fail("run %d's trace differs from run 1's at line %zu:\n  run 1: %s\n  run %d: %s", run,
			     n, lineOf(first, n).c_str(), run, lineOf(again, n).c_str());
		}
	}
	const char *record = std::getenv("RGFX_PEER_RECORD");
	if (record != nullptr && std::strcmp(record, "1") == 0) {
		std::ofstream out(path);
		out << first;
		if (!out)
			fail("could not write %s", path);
		std::printf("peer_tests: recorded %s\n", path);
		return 0;
	}
	std::ifstream in(path);
	std::stringstream recorded;
	recorded << in.rdbuf();
	if (recorded.str() != first) {
		size_t n = firstDifference(recorded.str(), first);
		fail("the wire traffic differs from %s at line %zu:\n  recorded: %s\n  now:      %s\n"
		     "(RGFX_PEER_RECORD=1 records it again from this build)",
		     path, n, lineOf(recorded.str(), n).c_str(), lineOf(first, n).c_str());
	}
	std::printf("peer_tests: the wire traffic is %s's, in 3 runs\n", path);
	return 0;
}
