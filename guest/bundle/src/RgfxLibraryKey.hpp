/*
 * RgfxLibraryKey: a .metallib's content key, by which the device keeps the libraries it made
 * for every process of the guest (RgfxLibraryLookup in the wire header).
 *
 * Our own C++; depends only on the committed wire header and CommonCrypto.
 */
#ifndef RGFX_LIBRARY_KEY_HPP_
#define RGFX_LIBRARY_KEY_HPP_

#include <cstddef>
#include <cstdint>

/// The bytes a library's key is the SHA-256 of first (RgfxLibraryLookup).
constexpr char kRgfxLibraryKeyPrefix[] = "rgfx-library-v2";

/// The pieces a key hashes the bytes it names in, each on its own (RgfxLibraryLookup): 256 KiB,
/// the last the rest, so the pieces of a framework's library, megabytes of them, are hashed
/// across the guest's cores. CommonCrypto's SHA-256 takes 2.1 ms a MiB on a guest core (567 MB/s
/// on the RTX 3090 host's guest, with or without the CPU's SHA extensions, measured).
constexpr size_t kRgfxLibraryKeyPiece = 256u << 10;

/// Writes the content key of the `length` bytes at `bytes` to `key`: the SHA-256 of
/// kRgfxLibraryKeyPrefix, then of the SHA-256 of each kRgfxLibraryKeyPiece piece, in order, of
/// the bytes before the .metallib's bitcode section followed by the bytes after it (the section
/// is the u64 offset and size at 0x48 and 0x50 of its header). The functions' HASH tags, which
/// name their bitcode, lie outside the section and are not checked here; the host checks them
/// before it keeps a library under a key. False, writing nothing, for bytes that are no .metallib
/// (no "MTLB" magic) or whose section does not lie inside them.
bool rgfxLibraryKey(const uint8_t *bytes, size_t length, uint8_t key[32]);

/// The bytes the key of a library's whole bytes is the SHA-256 of first (rgfxLibraryWholeKey).
constexpr char kRgfxLibraryWholeKeyPrefix[] = "rgfx-library-whole-v2";

/// The largest library rgfxLibraryWholeKey's key is asked for (RgfxMetalDevice+Pipelines.mm):
/// 4 MiB, 16 pieces.
constexpr size_t kRgfxWholeKeyMost = 4u << 20;

/// Writes the key of the whole `length` bytes at `bytes`, the bitcode included, to `key`: the
/// SHA-256 of kRgfxLibraryWholeKeyPrefix, then of the SHA-256 of each kRgfxLibraryKeyPiece piece
/// of the bytes, in order. The host keeps a library whose functions' HASH tags it cannot vouch
/// for (Core Image's libraries: their tags are no SHA-256 of their bitcode) under this key
/// instead of rgfxLibraryKey's, as it names the bytes with nothing left to check. False, writing
/// nothing, for bytes that are no .metallib.
bool rgfxLibraryWholeKey(const uint8_t *bytes, size_t length, uint8_t key[32]);

/// A file's identity, as stat(2) gives it: its device and inode, its size, and its modification
/// and status-change times to the nanosecond, each of which a rewrite of the file changes.
struct RgfxFileIdentity {
	uint64_t device;
	uint64_t inode;
	uint64_t size;
	int64_t modified_s;
	int64_t modified_ns;
	int64_t changed_s;
	int64_t changed_ns;

	bool operator<(const RgfxFileIdentity &o) const;
	bool operator==(const RgfxFileIdentity &o) const;
};

/// rgfxLibraryKey of the `length` bytes at `bytes`, read from the file whose identity was `file`
/// both before and after the read: the key the process found before for that identity, with no
/// byte read, else rgfxLibraryKey's, kept for it (the first kRgfxKeptFileKeys identities, for the
/// process's life). So a library a process loads again from a file (newLibraryWithURL:) is not
/// hashed again, and a file rewritten since is.
bool rgfxLibraryKeyOfFile(const RgfxFileIdentity &file, const uint8_t *bytes, size_t length,
                          uint8_t key[32]);

/// rgfxLibraryWholeKey of bytes read from the file `file`, kept for it as rgfxLibraryKeyOfFile
/// keeps its key.
bool rgfxLibraryWholeKeyOfFile(const RgfxFileIdentity &file, const uint8_t *bytes, size_t length,
                               uint8_t key[32]);

/// The most file identities rgfxLibraryKeyOfFile keeps.
constexpr size_t kRgfxKeptFileKeys = 1024;

#endif // RGFX_LIBRARY_KEY_HPP_
