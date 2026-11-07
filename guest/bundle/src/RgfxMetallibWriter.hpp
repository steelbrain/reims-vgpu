/*
 * RgfxMetallibWriter: a .metallib of one function, written from its AIR bitcode, as the
 * device's GL functions make their libraries (RgfxGLFunction.h): the bitcode the GL code
 * generator lowers a shader to goes to the host as any other library does.
 *
 * The layout is the one `metal -mmacosx-version-min=11.0` writes for one function (read from
 * such a file; tools/gld_lower.m's `wrap` checks it against Metal's own loader):
 *   - an 88-byte header: "MTLB", the u16s 0x8001, 2 and 5, six zero bytes, the u64 file size,
 *     then the u64 offset and size of the function list, the public and the private metadata and
 *     the bitcode section;
 *   - the function list: a u32 count of 1, then the function's tag group, a u32 size that counts
 *     itself, then NAME (the name, NUL-terminated), TYPE (the stage), HASH (the SHA-256 of the
 *     bitcode section), OFFT (three u64 offsets, all 0), VERS (the u16 AIR and language major and
 *     minor versions) and MDSZ (the u64 size of the bitcode section), each a 4-byte name, a u16
 *     size and the data, and ENDT;
 *   - the header's extension: UUID (16 bytes) and ENDT;
 *   - the public and private metadata, a u32 size of 8 and ENDT each;
 *   - the bitcode section: a wrapper (the u32s 0x0B17C0DE, 0, 20, the bitcode's size and
 *     0xffffffff), the bitcode, and zeros to a multiple of 16.
 *
 * Our own C++; depends only on CommonCrypto.
 */
#ifndef RGFX_METALLIB_WRITER_HPP_
#define RGFX_METALLIB_WRITER_HPP_

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

/// A function's stage, as a .metallib's TYPE tag holds it.
enum class RgfxMetallibStage : uint8_t { Vertex = 0, Fragment = 1, Kernel = 2 };

/// The one function a .metallib of RgfxWriteMetallib holds.
struct RgfxMetallibFunction {
	std::string name;
	RgfxMetallibStage stage = RgfxMetallibStage::Vertex;
	uint16_t airMajor = 2, airMinor = 3;           // the module's AIR version
	uint16_t languageMajor = 2, languageMinor = 3; // the language version VERS names
	const uint8_t *bitcode = nullptr;              // the bitcode, unwrapped
	size_t bitcodeLength = 0;
};

/// The .metallib of `f`, in the layout above. Its UUID is the first 16 bytes of the bitcode
/// section's hash, so the same function always gives the same bytes (and the same library key).
std::vector<uint8_t> RgfxWriteMetallib(const RgfxMetallibFunction &f);

/// The bitcode inside a bitcode wrapper (0x0B17C0DE, then the u32 version, offset, size and CPU
/// type), as LLVM writes a module of an Apple target: the `*length` bytes at `bitcode` where they
/// are not wrapped, else the bytes the wrapper names, `*length` set to their size. A .metallib's
/// section is wrapped once, and Metal's compiler refuses a section whose bitcode is wrapped again
/// (measured: macOS 26.7.1, its service aborts reading the module).
const uint8_t *RgfxUnwrapBitcode(const uint8_t *bitcode, size_t *length);

#endif // RGFX_METALLIB_WRITER_HPP_
