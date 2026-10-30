/*
 * RgfxLibraryKey: see RgfxLibraryKey.hpp. Our own C++.
 */
#include "RgfxLibraryKey.hpp"

#include <CommonCrypto/CommonDigest.h>

#include <cstring>

namespace {

// A .metallib header's length, and where its bitcode section's offset and size lie.
constexpr size_t kHeaderLength = 0x58;
constexpr size_t kBitcodeOffset = 0x48;
constexpr size_t kBitcodeSize = 0x50;

uint64_t rdLe64(const uint8_t *p)
{
	uint64_t v;
	std::memcpy(&v, p, sizeof(v));
	return v;
}

// CC_SHA256_Update over a range that may pass CC_LONG's 32 bits.
void update(CC_SHA256_CTX &ctx, const uint8_t *p, size_t n)
{
	while (n > 0) {
		const CC_LONG take = n > 0x4000'0000 ? 0x4000'0000 : static_cast<CC_LONG>(n);
		CC_SHA256_Update(&ctx, p, take);
		p += take;
		n -= take;
	}
}

} // namespace

bool rgfxLibraryKey(const uint8_t *bytes, size_t length, uint8_t key[32])
{
	if (bytes == nullptr || length < kHeaderLength || std::memcmp(bytes, "MTLB", 4) != 0)
		return false;
	const uint64_t at = rdLe64(bytes + kBitcodeOffset);
	const uint64_t size = rdLe64(bytes + kBitcodeSize);
	if (at < kHeaderLength || at > length || size > length - at)
		return false;
	CC_SHA256_CTX ctx;
	CC_SHA256_Init(&ctx);
	update(ctx, reinterpret_cast<const uint8_t *>(kRgfxLibraryKeyPrefix),
	       sizeof(kRgfxLibraryKeyPrefix) - 1);
	update(ctx, bytes, static_cast<size_t>(at));
	update(ctx, bytes + at + size, static_cast<size_t>(length - at - size));
	CC_SHA256_Final(key, &ctx);
	return true;
}
