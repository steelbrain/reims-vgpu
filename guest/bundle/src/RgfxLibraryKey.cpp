/*
 * RgfxLibraryKey: see RgfxLibraryKey.hpp. Our own C++.
 */
#include "RgfxLibraryKey.hpp"

#include <CommonCrypto/CommonDigest.h>
#include <dispatch/dispatch.h>

#include <array>
#include <cstring>
#include <map>
#include <mutex>
#include <tuple>
#include <vector>

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

// Bytes a key names, as runs back to back.
struct Run {
	const uint8_t *bytes;
	size_t length;
};

struct Pieces {
	const Run *runs;
	size_t runCount;
	size_t total;
	uint8_t *digests; // 32 bytes a piece
};

// The SHA-256 of piece `i` of what `context` (Pieces) names, into its digests.
void hashPiece(void *context, size_t i)
{
	const Pieces &p = *static_cast<const Pieces *>(context);
	size_t from = i * kRgfxLibraryKeyPiece;
	size_t left = p.total - from < kRgfxLibraryKeyPiece ? p.total - from : kRgfxLibraryKeyPiece;
	CC_SHA256_CTX ctx;
	CC_SHA256_Init(&ctx);
	size_t base = 0;
	for (size_t r = 0; r < p.runCount && left > 0; r++) {
		const Run &run = p.runs[r];
		if (from < base + run.length) {
			const size_t at = from - base;
			const size_t take = run.length - at < left ? run.length - at : left;
			update(ctx, run.bytes + at, take);
			from += take;
			left -= take;
		}
		base += run.length;
	}
	CC_SHA256_Final(p.digests + i * 32, &ctx);
}

// The SHA-256 of `prefix`, then of each piece's SHA-256 of the runs' bytes, in order: the pieces
// hashed on the process's threads at once where there are several (dispatch_apply_f).
void keyOfPieces(const char *prefix, size_t prefixLength, const Run *runs, size_t runCount,
                 uint8_t key[32])
{
	size_t total = 0;
	for (size_t r = 0; r < runCount; r++)
		total += runs[r].length;
	const size_t count = (total + kRgfxLibraryKeyPiece - 1) / kRgfxLibraryKeyPiece;
	std::vector<uint8_t> digests(count * 32);
	Pieces pieces = {runs, runCount, total, digests.data()};
	if (count == 1)
		hashPiece(&pieces, 0);
	else if (count > 1)
		dispatch_apply_f(count, DISPATCH_APPLY_AUTO, &pieces, &hashPiece);
	CC_SHA256_CTX ctx;
	CC_SHA256_Init(&ctx);
	update(ctx, reinterpret_cast<const uint8_t *>(prefix), prefixLength);
	update(ctx, digests.data(), digests.size());
	CC_SHA256_Final(key, &ctx);
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
	const Run runs[] = {{bytes, static_cast<size_t>(at)},
	                    {bytes + at + size, static_cast<size_t>(length - at - size)}};
	keyOfPieces(kRgfxLibraryKeyPrefix, sizeof(kRgfxLibraryKeyPrefix) - 1, runs, 2, key);
	return true;
}

bool rgfxLibraryWholeKey(const uint8_t *bytes, size_t length, uint8_t key[32])
{
	if (bytes == nullptr || length < kHeaderLength || std::memcmp(bytes, "MTLB", 4) != 0)
		return false;
	const Run runs[] = {{bytes, length}};
	keyOfPieces(kRgfxLibraryWholeKeyPrefix, sizeof(kRgfxLibraryWholeKeyPrefix) - 1, runs, 1,
	            key);
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

namespace {

typedef std::array<uint8_t, 32> Key;

// `make`'s key of the bytes read from `file`, kept in `kept` (under `lock`) for the first
// kRgfxKeptFileKeys identities, for the process's life.
bool keyOfFile(std::mutex &lock, std::map<RgfxFileIdentity, Key> &kept,
               bool (*make)(const uint8_t *, size_t, uint8_t *), const RgfxFileIdentity &file,
               const uint8_t *bytes, size_t length, uint8_t key[32])
{
	{
		std::lock_guard<std::mutex> hold(lock);
		auto found = kept.find(file);
		if (found != kept.end()) {
			std::memcpy(key, found->second.data(), 32);
			return true;
		}
	}
	if (!make(bytes, length, key))
		return false;
	std::lock_guard<std::mutex> hold(lock);
	if (kept.size() < kRgfxKeptFileKeys) {
		Key k;
		std::memcpy(k.data(), key, k.size());
		kept.emplace(file, k);
	}
	return true;
}

} // namespace

bool rgfxLibraryKeyOfFile(const RgfxFileIdentity &file, const uint8_t *bytes, size_t length,
                          uint8_t key[32])
{
	static std::mutex lock;
	// Never destroyed: a library may be loaded on any thread while the process exits.
	static auto *kept = new std::map<RgfxFileIdentity, Key>();
	return keyOfFile(lock, *kept, &rgfxLibraryKey, file, bytes, length, key);
}

bool rgfxLibraryWholeKeyOfFile(const RgfxFileIdentity &file, const uint8_t *bytes, size_t length,
                               uint8_t key[32])
{
	static std::mutex lock;
	static auto *kept = new std::map<RgfxFileIdentity, Key>();
	return keyOfFile(lock, *kept, &rgfxLibraryWholeKey, file, bytes, length, key);
}
