/*
 * rgfx_test_ring: the self-contained host-side consumer the bundle's user-space
 * checks read their wire bytes back through. It builds a ring in a plain buffer
 * (no BAR, no MMIO), counts doorbells, and walks messages back off the ring the
 * way crates/reims-vgpu-graphics/src/ring.rs does, so a test can post through the real
 * RgfxRingWriter/RgfxSerializer and assert the exact bytes that landed.
 *
 * Shared by the serialiser check (tests/serializer_tests.cpp, the 387 wire-byte
 * checks) and the Metal shell's routing check (tests/shell_tests.mm): a wrapper
 * routing test posts through a serialiser onto one of these rings and reads the
 * message back with expectMessage<T>, exactly as the serialiser check does. Each
 * check is its own binary with its own main, so the `inline` definitions here are
 * one per binary.
 *
 * Our own code; depends only on the committed wire header.
 */

#ifndef RGFX_TEST_RING_HPP_
#define RGFX_TEST_RING_HPP_

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <crt_externs.h>
#include <pthread.h>
#include <spawn.h>
#include <sys/wait.h>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "reims_graphics_wire.h"

/// Counts the checks a binary has run; printed in its one-line summary. Atomic, as tests check
/// from several threads at once.
inline std::atomic<int> g_checks{0};

#define CHECK(cond)                                                                                \
	do {                                                                                           \
		g_checks++;                                                                                \
		if (!(cond)) {                                                                             \
			std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);                  \
			std::abort();                                                                          \
		}                                                                                          \
	} while (0)

/// Runs this binary, `self`, again as `self mode name`, with its mutexes on libpthread's fair-share
/// policy (PTHREAD_MUTEX_DEFAULT_POLICY=1), and checks that it exits 0. On that policy a mutex
/// unlocked with a waiter goes to the waiter; on the default one the unlocking thread can take it
/// again first. A test of what another thread does between two holds of one mutex runs so: its
/// thread waits on the mutex through the first hold and takes it before the second. Measured on
/// the M3 Max, macOS 27.0.1: serializer_tests' test_held_since_the_post_is_not_counted, against
/// the two-hold code it guards, passed on the default policy and failed on fair share.
inline void rgfxRunFairShare(const char *self, const char *mode, const char *name)
{
	static const char kPolicy[] = "PTHREAD_MUTEX_DEFAULT_POLICY=";
	std::vector<char *> env;
	for (char **e = *_NSGetEnviron(); *e != nullptr; e++)
		if (std::strncmp(*e, kPolicy, sizeof(kPolicy) - 1) != 0)
			env.push_back(*e);
	std::string fair = std::string(kPolicy) + "1";
	env.push_back(fair.data());
	env.push_back(nullptr);
	char *args[] = {const_cast<char *>(self), const_cast<char *>(mode), const_cast<char *>(name),
	                nullptr};
	pid_t pid = 0;
	CHECK(posix_spawn(&pid, self, nullptr, nullptr, args, env.data()) == 0);
	int status = 0;
	CHECK(waitpid(pid, &status, 0) == pid);
	if (!WIFEXITED(status) || WEXITSTATUS(status) != 0)
		std::fprintf(stderr, "%s %s ended with status %d\n", mode, name, status);
	CHECK(WIFEXITED(status) && WEXITSTATUS(status) == 0);
}

/// Whether this process's mutexes take libpthread's fair-share policy (rgfxRunFairShare).
inline bool rgfxMutexesFairShare()
{
	pthread_mutexattr_t attr;
	pthread_mutexattr_init(&attr);
	int policy = 0;
	pthread_mutexattr_getpolicy_np(&attr, &policy);
	pthread_mutexattr_destroy(&attr);
	return policy == PTHREAD_MUTEX_POLICY_FAIRSHARE_NP;
}

// A ring's region: its 256-byte header then a power-of-two byte ring, in a plain
// buffer, with a doorbell counter. Mirrors a BAR-mapped ring for the test.
struct TestRing {
	std::vector<uint8_t> mem;
	uint32_t dataSize;
	uint64_t consumed = 0; // the host's count
	static inline std::atomic<int> doorbells{0};

	explicit TestRing(uint32_t data_size)
	    : mem(RGFX_RING_HEADER_SIZE + data_size, 0), dataSize(data_size)
	{
	}

	volatile uint8_t *base() { return mem.data(); }
	volatile uint8_t *data() { return mem.data() + RGFX_RING_HEADER_SIZE; }

	uint64_t producer()
	{
		uint64_t v;
		std::memcpy(&v, mem.data() + __builtin_offsetof(RgfxRingHeader, producer), 8);
		return v;
	}
	void setConsumer(uint64_t v)
	{
		std::memcpy(mem.data() + __builtin_offsetof(RgfxRingHeader, consumer), &v, 8);
	}
	void setConsumerIdle(uint32_t v)
	{
		std::memcpy(mem.data() + __builtin_offsetof(RgfxRingHeader, consumer_idle), &v, 4);
	}

	static void doorbell(void *, uint32_t) { doorbells.fetch_add(1); }

	// --- the host-producer side (host -> guest rings, for RgfxReplyReader) --------
	// A completion ring the reader drains: the test plays the host here, writing a
	// reply message and publishing `producer`, mirroring crates/reims-vgpu-graphics/src/
	// ring.rs Producer::post (pad to the end so a message never wraps). `msg` already
	// holds its header (type, size, seqno). Returns false with nothing written when the
	// ring has no room.
	uint64_t produced = 0; // the host's producer count, for host -> guest rings

	// Atomic, as the host's are: a reader on another thread (the completion thread)
	// publishes `consumer` and reads `producer` meanwhile.
	uint64_t consumerCount()
	{
		return __atomic_load_n(reinterpret_cast<uint64_t *>(
		                           mem.data() + __builtin_offsetof(RgfxRingHeader, consumer)),
		                       __ATOMIC_ACQUIRE);
	}

	void setProducer(uint64_t v)
	{
		__atomic_store_n(reinterpret_cast<uint64_t *>(mem.data() +
		                                              __builtin_offsetof(RgfxRingHeader, producer)),
		                 v, __ATOMIC_RELEASE);
	}

	bool postFromHost(const void *msg, uint32_t size)
	{
		uint64_t guest = consumerCount();
		uint64_t used = produced - guest;
		if (used > dataSize)
			return false;
		uint32_t off = static_cast<uint32_t>(produced) & (dataSize - 1);
		uint32_t tail = dataSize - off;
		uint32_t pad = tail < size ? tail : 0;
		if (static_cast<uint64_t>(pad) + size > dataSize - used)
			return false;
		if (pad) {
			uint32_t padHeader[2] = { RGFX_MSG_PAD, pad };
			std::memcpy(const_cast<uint8_t *>(data()) + off, padHeader, sizeof(padHeader));
			produced += pad;
			off = 0;
		}
		std::memcpy(const_cast<uint8_t *>(data()) + off, msg, size);
		produced += size;
		std::atomic_thread_fence(std::memory_order_release);
		setProducer(produced);
		return true;
	}

	// The next message from the host's count, pads skipped; false when none more.
	bool next(RgfxMsgHeader *header, std::vector<uint8_t> *bytes)
	{
		for (;;) {
			if (consumed >= producer())
				return false;
			uint32_t off = static_cast<uint32_t>(consumed) & (dataSize - 1);
			RgfxMsgHeader h;
			std::memcpy(&h, const_cast<uint8_t *>(data()) + off, sizeof(h));
			if (h.type == RGFX_MSG_PAD) {
				consumed += h.size;
				continue;
			}
			*header = h;
			bytes->resize(h.size);
			std::memcpy(bytes->data(), const_cast<uint8_t *>(data()) + off, h.size);
			consumed += h.size;
			setConsumer(consumed);
			return true;
		}
	}
};

// Reads a message back into a wire struct T and returns it, checking the type
// and that the on-wire size equals sizeof(T).
template <typename T> T expectMessage(TestRing &ring, uint32_t type)
{
	RgfxMsgHeader h = {};
	std::vector<uint8_t> bytes;
	CHECK(ring.next(&h, &bytes));
	if (h.type != type)
		std::fprintf(stderr, "expectMessage: type 0x%x, expected 0x%x\n", h.type, type);
	CHECK(h.type == type);
	CHECK(h.size == sizeof(T));
	CHECK(bytes.size() == sizeof(T));
	T msg;
	std::memcpy(&msg, bytes.data(), sizeof(T));
	return msg;
}

/// A queue's QUEUE_CREATE, then the FENCE_CREATEs of its barrier fences, which it makes with
/// itself (RgfxCommandObjects.mm, QUEUE BARRIERS).
inline RgfxQueueCreate expectQueueCreate(TestRing &ring)
{
	RgfxQueueCreate q = expectMessage<RgfxQueueCreate>(ring, RGFX_CMD_QUEUE_CREATE);
	for (int i = 0; i < 8; i++)
		(void)expectMessage<RgfxObjectCreate>(ring, RGFX_CMD_FENCE_CREATE);
	return q;
}

#endif // RGFX_TEST_RING_HPP_
