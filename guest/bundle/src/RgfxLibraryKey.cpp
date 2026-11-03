/*
 * RgfxLibraryKey: see RgfxLibraryKey.hpp. Our own C++.
 */
#include "RgfxLibraryKey.hpp"

#include <CommonCrypto/CommonDigest.h>

#include <array>
#include <cstring>
#include <map>
#include <mutex>
#include <tuple>

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

namespace {

std::tuple<uint64_t, uint64_t, uint64_t, int64_t, int64_t, int64_t, int64_t>
fields(const RgfxFileIdentity &f)
{
	return std::make_tuple(f.device, f.inode, f.size, f.modified_s, f.modified_ns, f.changed_s,
	                       f.changed_ns);
}

} // namespace

bool RgfxFileIdentity::operator<(const RgfxFileIdentity &o) const
{
	return fields(*this) < fields(o);
}

bool RgfxFileIdentity::operator==(const RgfxFileIdentity &o) const
{
	return fields(*this) == fields(o);
}

bool rgfxLibraryKeyOfFile(const RgfxFileIdentity &file, const uint8_t *bytes, size_t length,
                          uint8_t key[32])
{
	typedef std::array<uint8_t, 32> Key;
	static std::mutex lock;
	// Never destroyed: a library may be loaded on any thread while the process exits.
	static auto *kept = new std::map<RgfxFileIdentity, Key>();
	{
		std::lock_guard<std::mutex> hold(lock);
		auto found = kept->find(file);
		if (found != kept->end()) {
			std::memcpy(key, found->second.data(), 32);
			return true;
		}
	}
	if (!rgfxLibraryKey(bytes, length, key))
		return false;
	std::lock_guard<std::mutex> hold(lock);
	if (kept->size() < kRgfxKeptFileKeys) {
		Key k;
		std::memcpy(k.data(), key, k.size());
		kept->emplace(file, k);
	}
	return true;
}
