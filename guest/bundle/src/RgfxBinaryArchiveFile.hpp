/*
 * RgfxBinaryArchiveFile: what an MTLBinaryArchive of the device holds, its keys, and the file
 * `serializeToURL:` writes.
 *
 * The device compiles every pipeline on the host from its functions and state, so its archive
 * holds no compiled code: it holds the keys of what was added, per architecture, so that a
 * pipeline create with MTLPipelineOptionFailOnBinaryArchiveMiss answers a hit or a miss as Metal
 * does. A key is the SHA-256 of what Metal keys that function's compile by (RgfxArchiveKey, and
 * RgfxBinaryArchive.mm for the fields of each stage).
 *
 * The file, every integer little-endian:
 *   "RGFXBAR1"                               8 bytes
 *   u32 section count
 *   per section:
 *     u32 architecture name length, then its UTF-8 bytes
 *     u32 entry count, then each entry: u8 kind, 32 key bytes
 * A section holds the entries of one architecture (`MTLDevice.architecture.name`): a device
 * looks up only its own, and keeps the others to write back, as a Metal archive keeps the
 * slices of other GPUs.
 *
 * Our own C++; depends only on CommonCrypto.
 */
#ifndef RGFX_BINARY_ARCHIVE_FILE_HPP_
#define RGFX_BINARY_ARCHIVE_FILE_HPP_

#include <CommonCrypto/CommonDigest.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <set>
#include <string>
#include <vector>

/// What an archive entry keys: one function's compile, by its stage, or a function added alone.
enum RgfxArchiveKind : uint8_t {
	kRgfxArchiveCompute = 1,
	kRgfxArchiveVertex = 2,
	kRgfxArchiveFragment = 3,
	kRgfxArchiveTile = 4,
	kRgfxArchiveObject = 5,
	kRgfxArchiveMesh = 6,
	/// `addFunctionWithDescriptor:library:error:`: a function, which keys no pipeline.
	kRgfxArchiveFunction = 7,
	/// A stitched library (`addLibraryWithDescriptor:error:`), which the archive keeps in memory
	/// only: Metal's archive lists none in its keys and writes none to a file (measured).
	kRgfxArchiveStitched = 8,
};

/// One entry: its kind and the 32-byte key.
struct RgfxArchiveEntry {
	uint8_t kind = 0;
	uint8_t key[32] = {};

	bool operator<(const RgfxArchiveEntry &o) const
	{
		if (kind != o.kind)
			return kind < o.kind;
		return std::memcmp(key, o.key, sizeof(key)) < 0;
	}
	bool operator==(const RgfxArchiveEntry &o) const
	{
		return kind == o.kind && std::memcmp(key, o.key, sizeof(key)) == 0;
	}
};

/// The entries of one architecture.
struct RgfxArchiveSection {
	std::string architecture;
	std::set<RgfxArchiveEntry> entries;
};

/// What a file's bytes are.
enum class RgfxArchiveFileKind {
	/// This device's format: its sections were read.
	Ours,
	/// A Metal archive (a fat file of GPU slices, magic 0xcbfebabe or 0xcafebabe, at least one
	/// slice and each inside the file), or a whole .metallib, which Metal makes an archive of:
	/// none of its slices is this device's.
	Metal,
	/// Neither, or one cut short: Metal's "has an invalid format".
	Invalid,
};

/// Reads the `length` bytes at `bytes` as an archive file, into `sections` where they are ours.
RgfxArchiveFileKind rgfxReadArchiveFile(const uint8_t *bytes, size_t length,
                                        std::vector<RgfxArchiveSection> &sections);

/// The file of `sections`, in their order, each section's entries sorted.
std::vector<uint8_t> rgfxWriteArchiveFile(const std::vector<RgfxArchiveSection> &sections);

/// An entry's key: the SHA-256 of "rgfx-archive-v1", the kind, then each field added, each an
/// integer of a fixed width or a length and its bytes, so that no two field lists hash alike.
class RgfxArchiveKey {
public:
	explicit RgfxArchiveKey(uint8_t kind) : m_kind(kind)
	{
		static const char kPrefix[] = "rgfx-archive-v1";
		CC_SHA256_Init(&m_ctx);
		CC_SHA256_Update(&m_ctx, kPrefix, sizeof(kPrefix) - 1);
		CC_SHA256_Update(&m_ctx, &kind, 1);
	}
	void u32(uint32_t v) { CC_SHA256_Update(&m_ctx, &v, sizeof(v)); }
	void u64(uint64_t v) { CC_SHA256_Update(&m_ctx, &v, sizeof(v)); }
	void bytes(const void *p, size_t n)
	{
		u64(n);
		if (n != 0)
			CC_SHA256_Update(&m_ctx, p, static_cast<CC_LONG>(n));
	}
	void text(const char *s) { bytes(s, s == nullptr ? 0 : std::strlen(s)); }
	RgfxArchiveEntry finish()
	{
		RgfxArchiveEntry e;
		e.kind = m_kind;
		CC_SHA256_Final(e.key, &m_ctx);
		return e;
	}

private:
	uint8_t m_kind;
	CC_SHA256_CTX m_ctx;
};

#endif // RGFX_BINARY_ARCHIVE_FILE_HPP_
