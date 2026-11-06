// The display's gamma capability (RgfxGammaCapability.hpp), built natively: the entry
// RgfxDeviceControl answers to request 2 of attribute 0x711, read back as macOS 26.7.1's
// IOPresentment and CoreDisplay read it, gives plane 0 a gamma table of 8-bit type and 256
// entries, which the display pipe's gamma selectors take and the host gets in one part.
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <vector>

#include "RgfxGammaCapability.hpp"

static int g_checks = 0, g_failed = 0;
#define CHECK(e)                                                                   \
	do {                                                                           \
		g_checks++;                                                                \
		if (!(e)) {                                                                \
			g_failed++;                                                            \
			std::fprintf(stderr, "%s:%d: CHECK(%s) failed\n", __FILE__, __LINE__, #e); \
		}                                                                          \
	} while (0)

static uint32_t bits(float f)
{
	uint32_t v;
	std::memcpy(&v, &f, sizeof(v));
	return v;
}

static uint32_t read32(const std::vector<uint8_t> &b, uint32_t off)
{
	uint32_t v;
	std::memcpy(&v, b.data() + off, sizeof(v));
	return v;
}

static uint64_t read64(const std::vector<uint8_t> &b, uint32_t off)
{
	uint64_t v;
	std::memcpy(&v, b.data() + off, sizeof(v));
	return v;
}

// The bits a channel CoreDisplay gives a table type (GammaChannelTableEntryCountAndType); 0 for
// a type that is no table.
static uint32_t channelBits(uint32_t type)
{
	if (type & 0x2200)
		return 16;
	if (type & 0x1100)
		return 12;
	if (type & 0x880)
		return 10;
	if (type & 0x458)
		return 8;
	return 0;
}

// The answer to request 2 as RgfxDeviceControl builds it: 0x196c bytes, the count 1 at 0x08, and
// the entry at 0x0c.
static std::vector<uint8_t> answer()
{
	std::vector<uint8_t> a(0x196c);
	const uint32_t count = 1;
	std::memcpy(a.data() + kRgfxCapabilityCount, &count, sizeof(count));
	const RgfxGammaCapability g = rgfxGammaCapability();
	std::memcpy(a.data() + kRgfxCapabilityEntries, &g, sizeof(g));
	return a;
}

int main()
{
	// The fields sit where IOPresentment reads them.
	CHECK(sizeof(RgfxGammaCapability) == 0x30);
	CHECK(offsetof(RgfxGammaCapability, field_04) == 0x04);
	CHECK(offsetof(RgfxGammaCapability, field_0c) == 0x0c);
	CHECK(offsetof(RgfxGammaCapability, field_10) == 0x10);
	CHECK(offsetof(RgfxGammaCapability, field_14) == 0x14);
	CHECK(offsetof(RgfxGammaCapability, field_18) == 0x18);
	CHECK(offsetof(RgfxGammaCapability, field_20) == 0x20);
	CHECK(offsetof(RgfxGammaCapability, field_24) == 0x24);
	CHECK(offsetof(RgfxGammaCapability, field_28) == 0x28);
	CHECK(sizeof(RgfxGammaCapability) <= kRgfxCapabilityStride);
	CHECK(kRgfxCapabilityEntries + kRgfxCapabilityMaxEntries * kRgfxCapabilityStride <= 0x196c);

	// Read back as IOPresentment reads it: one entry, which names plane 0 and no other.
	const std::vector<uint8_t> a = answer();
	const uint32_t count = read32(a, kRgfxCapabilityCount);
	CHECK(count == 1 && count <= kRgfxCapabilityMaxEntries);
	const uint32_t e = kRgfxCapabilityEntries;
	const uint64_t planes = read64(a, e + 0x04);
	CHECK(planes & (1ull << 0));
	CHECK(!(planes & (1ull << 1)));

	// The AMD's entry 0, value for value.
	CHECK(read32(a, e + 0x00) == 0x88005);
	CHECK(planes == 1);
	CHECK(read32(a, e + 0x0c) == bits(0.0f));
	CHECK(read32(a, e + 0x10) == 0x10);
	CHECK(read32(a, e + 0x14) == 0);
	CHECK(read64(a, e + 0x18) == 0);
	CHECK(read32(a, e + 0x20) == bits(0.0f));
	CHECK(read32(a, e + 0x24) == bits(1.0f));
	CHECK(read64(a, e + 0x28) == 256);

	// Read as CoreDisplay reads it: a table, not a configuration (type 1), of 8 bits a channel and
	// 256 entries, so plane 0 has gamma.
	const uint32_t type = read32(a, e + 0x10);
	CHECK((type & (type - 1)) == 0);
	CHECK(type != 1);
	CHECK(channelBits(type) == 8);
	CHECK(read64(a, e + 0x28) != 0);

	// Past the count: zeros, so IOPresentment finds no second entry.
	bool rest = true;
	for (size_t i = kRgfxCapabilityEntries + sizeof(RgfxGammaCapability); i < a.size(); i++)
		rest = rest && a[i] == 0;
	CHECK(rest);

	// A table of the most entries the entry declares is one the gamma selectors take, and the host
	// gets it in one DISPLAY_COLOR part.
	const uint32_t n = static_cast<uint32_t>(read64(a, e + 0x28));
	std::vector<uint8_t> table(kRgfxGammaHeaderSize + n * kRgfxGammaEntrySize);
	const uint32_t txn = 7;
	std::memcpy(table.data() + 0x00, &txn, sizeof(txn));
	std::memcpy(table.data() + 0x08, &n, sizeof(n));
	RgfxGammaArgs args;
	CHECK(rgfxGammaDecode(table.data(), table.size(), &args) && args.count == n);
	CHECK(rgfxGammaAction(kRgfxPipePostGamma, args, txn) == RgfxGammaAction::Stage);
	CHECK(rgfxGammaAction(kRgfxPipePlaneGamma, args, txn) == RgfxGammaAction::Stage);
	CHECK(rgfxColorPartCount(n) == 1);

	if (g_failed) {
		std::fprintf(stderr, "gamma_capability_test: %d of %d checks failed\n", g_failed, g_checks);
		return 1;
	}
	std::printf("gamma_capability_test: %d checks passed\n", g_checks);
	return 0;
}
