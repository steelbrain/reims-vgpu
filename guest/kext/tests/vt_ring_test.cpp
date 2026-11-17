// The native test of RgfxVtRing.hpp: the take of the VideoToolbox ring through a VT_RING's count,
// over records written as the host writes them (its pads to the ring's end included), and the
// counts and records it refuses.
#include "RgfxVtRing.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {
int checks = 0;

void check(bool ok, const char *what)
{
	checks++;
	if (!ok) {
		std::fprintf(stderr, "vt_ring_test: FAILED: %s\n", what);
		std::exit(1);
	}
}

constexpr uint64_t kSize = RGFX_VT_RING_DATA_SIZE;

// The host's end of the ring: it writes each record at its count, a pad to the ring's end first
// where the bytes left there are fewer than the record.
struct Host {
	std::vector<uint8_t> bytes = std::vector<uint8_t>(kSize);
	uint64_t count = 0;

	void put32(uint64_t at, uint32_t v) { std::memcpy(&bytes[at & (kSize - 1)], &v, 4); }

	// A record of `type` carrying `text` as a VT_MESSAGE carries its bytes; its count after.
	uint64_t write(uint32_t type, const std::string &text)
	{
		const uint32_t length = static_cast<uint32_t>(text.size());
		const uint32_t size = (RGFX_VT_HOST_MESSAGE_BYTES_OFFSET + length + 7) & ~7u;
		const uint64_t tail = kSize - (count & (kSize - 1));
		if (tail < size) {
			put32(count, RGFX_MSG_PAD);
			put32(count + 4, static_cast<uint32_t>(tail));
			count += tail;
		}
		const uint64_t at = count & (kSize - 1);
		std::memset(&bytes[at], 0, size);
		put32(count, type);
		put32(count + 4, size);
		put32(count + 16, length);
		std::memcpy(&bytes[at + RGFX_VT_HOST_MESSAGE_BYTES_OFFSET], text.data(), length);
		count += size;
		return count;
	}
};

// What the take handed on: each message's bytes, read as ReimsGraphics::vtMessage reads them.
struct Taken {
	std::vector<std::string> messages;

	RgfxVtRingTaken take(const Host &host, uint64_t consumed, uint64_t produced)
	{
		return rgfxTakeVtRing(host.bytes.data(), consumed, produced,
		                      [this](const volatile uint8_t *payload, uint32_t size) {
			                      constexpr uint32_t kBytesAt =
			                          RGFX_VT_HOST_MESSAGE_BYTES_OFFSET - sizeof(RgfxMsgHeader);
			                      uint32_t length = 0;
			                      std::memcpy(&length, const_cast<const uint8_t *>(payload), 4);
			                      check(size >= kBytesAt && length <= size - kBytesAt,
			                            "a record holds its message");
			                      messages.emplace_back(
			                          reinterpret_cast<const char *>(
			                              const_cast<const uint8_t *>(payload) + kBytesAt),
			                          length);
		                      });
	}
};
} // namespace

int main()
{
	static_assert(RGFX_FEATURE_VT_RING == (1ull << 54), "VT_RING is bit 54");
	static_assert(RGFX_EVT_VT_RING == 0x40f && sizeof(RgfxVtRing) == 24, "VT_RING is 0x40f");
	static_assert(RGFX_VT_RING_OFFSET >= RGFX_CONTROL_COMPLETE_OFFSET + RGFX_CONTROL_RING_SPAN,
	              "the ring follows the control completion ring's span");
	static_assert(RGFX_VT_RING_OFFSET + RGFX_RING_HEADER_SIZE + RGFX_VT_RING_DATA_SIZE <=
	                  RGFX_ALLOCATIONS_OFFSET,
	              "the ring lies before the guest's allocations, in the control rings' mapping");

	// Two messages, taken in order through the count; a count taken already takes nothing.
	{
		Host host;
		Taken taken;
		const uint64_t first = host.write(RGFX_EVT_VT_MESSAGE, "vtpv-reply");
		const uint64_t second = host.write(RGFX_EVT_VT_MESSAGE, "again");
		check(first == 40 && second == 72, "records are padded to 8");
		RgfxVtRingTaken t = taken.take(host, 0, first);
		check(t.consumed == first && t.fault == RgfxVtRingFault::None, "a take reaches the count");
		check(taken.messages.size() == 1 && taken.messages[0] == "vtpv-reply",
		      "a take hands on the records through the count alone");
		t = taken.take(host, first, second);
		check(t.consumed == second && taken.messages.size() == 2 && taken.messages[1] == "again",
		      "the next take goes on from the count");
		t = taken.take(host, second, first);
		check(t.consumed == second && t.fault == RgfxVtRingFault::None && taken.messages.size() == 2,
		      "a count taken already takes nothing");
		t = taken.take(host, second, second);
		check(t.consumed == second && taken.messages.size() == 2, "the same count takes nothing");
	}

	// A record the bytes left at the ring's end do not hold follows a pad, at the ring's start;
	// a record of another type is passed.
	{
		Host host;
		Taken taken;
		const std::string longest(RGFX_VT_HOST_MESSAGE_MAX, 'x');
		uint64_t produced = 0;
		for (int i = 0; i < 127; i++)
			produced = host.write(RGFX_EVT_VT_MESSAGE, longest);
		RgfxVtRingTaken t = taken.take(host, 0, produced);
		check(t.consumed == produced && taken.messages.size() == 127, "127 of the longest fit");
		const uint64_t passed = host.write(RGFX_EVT_VT_STATUS, "not a message");
		const uint64_t wrapped = host.write(RGFX_EVT_VT_MESSAGE, longest);
		check(passed < kSize && wrapped > kSize, "the record after the pad starts the ring again");
		t = taken.take(host, produced, wrapped);
		check(t.consumed == wrapped && t.fault == RgfxVtRingFault::None, "a pad is passed");
		check(taken.messages.size() == 128 && taken.messages.back() == longest,
		      "the record after the pad is taken whole, and the other type is not");
	}

	// A count out of the ring's reach, and records of sizes the protocol does not allow, take
	// nothing more: the count moves to the one given.
	{
		Host host;
		Taken taken;
		RgfxVtRingTaken t = taken.take(host, 0, kSize + 8);
		check(t.consumed == kSize + 8 && t.fault == RgfxVtRingFault::Count &&
		          taken.messages.empty(),
		      "a count ahead by more than the ring holds");
		const uint64_t produced = host.write(RGFX_EVT_VT_MESSAGE, "short");
		host.put32(4, 20);
		t = taken.take(host, 0, produced);
		check(t.consumed == produced && t.fault == RgfxVtRingFault::Size && taken.messages.empty(),
		      "a size not a multiple of 8");
		host.put32(4, 8);
		t = taken.take(host, 0, produced);
		check(t.fault == RgfxVtRingFault::Size && taken.messages.empty(), "a size under a header");
		host.put32(4, 64);
		t = taken.take(host, 0, produced);
		check(t.fault == RgfxVtRingFault::Size && taken.messages.empty(),
		      "a size past the count given");
		host.put32(4, 32);
		t = taken.take(host, 0, produced);
		check(t.fault == RgfxVtRingFault::None && taken.messages.size() == 1 &&
		          taken.messages[0] == "short",
		      "the record as written");
	}

	std::printf("vt_ring_test: %d checks passed\n", checks);
	return 0;
}
