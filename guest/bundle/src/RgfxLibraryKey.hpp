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
constexpr char kRgfxLibraryKeyPrefix[] = "rgfx-library-v1";

/// Writes the content key of the `length` bytes at `bytes` to `key`: the SHA-256 of
/// kRgfxLibraryKeyPrefix, the bytes before the .metallib's bitcode section, then the bytes after
/// it (the section is the u64 offset and size at 0x48 and 0x50 of its header). The functions'
/// HASH tags, which name their bitcode, lie outside the section and are not checked here; the
/// host checks them before it keeps a library under a key. False, writing nothing, for bytes
/// that are no .metallib (no "MTLB" magic) or whose section does not lie inside them.
bool rgfxLibraryKey(const uint8_t *bytes, size_t length, uint8_t key[32]);

#endif // RGFX_LIBRARY_KEY_HPP_
