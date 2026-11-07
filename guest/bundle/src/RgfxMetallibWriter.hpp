/*
 * RgfxMetallibWriter: a .metallib of functions, written from their AIR bitcode, as the device's
 * GL functions make their libraries (RgfxGLFunction.h): the bitcode the GL code generator lowers a
 * shader to goes to the host as any other library does. It also reads the kernels out of the bytes
 * Metal's compiler answers for OpenCL C source, which are no .metallib (RgfxReadClLibrary).
 *
 * The layout is the one `metal -mmacosx-version-min=11.0` writes for one function (read from
 * such a file; tools/gld_lower.m's `wrap` checks it against Metal's own loader), with each
 * function's entries placed as `metal` places those of a library of several (read from such a
 * file, tests/suite/fixtures' and the Vulkan platform's metallibs):
 *   - an 88-byte header: "MTLB", the u16s 0x8001, 2 and 5, six zero bytes, the u64 file size,
 *     then the u64 offset and size of the function list, the public and the private metadata and
 *     the bitcode section;
 *   - the function list: a u32 count, then each function's tag group, a u32 size that counts
 *     itself, then NAME (the name, NUL-terminated), TYPE (the stage), HASH (the SHA-256 of the
 *     function's bitcode), OFFT (the u64 offsets of its public and private metadata and of its
 *     bitcode, each in its section), VERS (the u16 AIR and language major and minor versions) and
 *     MDSZ (the u64 size of its bitcode), each a 4-byte name, a u16 size and the data, and ENDT;
 *     the header's list size counts the groups, not the count;
 *   - the header's extension: UUID (16 bytes) and ENDT;
 *   - the public and private metadata, for each function a u32 size of 8 and ENDT;
 *   - the bitcode section: each function's in turn, a wrapper (the u32s 0x0B17C0DE, 0, 20, the
 *     bitcode's size and 0xffffffff), the bitcode, and zeros to a multiple of 16.
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

/// A function a .metallib of RgfxWriteMetallib holds.
struct RgfxMetallibFunction {
	std::string name;
	RgfxMetallibStage stage = RgfxMetallibStage::Vertex;
	uint16_t airMajor = 2, airMinor = 3;           // the module's AIR version
	uint16_t languageMajor = 2, languageMinor = 3; // the language version VERS names
	const uint8_t *bitcode = nullptr;              // the bitcode, unwrapped
	size_t bitcodeLength = 0;
};

/// The .metallib of `functions`, in their order, in the layout above. Its UUID is the first 16
/// bytes of the bitcode section's hash, so the same functions always give the same bytes (and the
/// same library key).
std::vector<uint8_t> RgfxWriteMetallib(const std::vector<RgfxMetallibFunction> &functions);

/// The .metallib of `f` alone.
std::vector<uint8_t> RgfxWriteMetallib(const RgfxMetallibFunction &f);

/// The .metallib of the kernels in `bytes`, in `out`, where `bytes` are the container Metal's
/// compiler answers (libraryDataContents) for a library of OpenCL C source (MTLCompileOptions'
/// sourceLanguage 1, as the GL renderer compiles a compute program): each kernel's module under
/// its name, in the container's order, and the writer's VERS (the container states no versions;
/// the host compiles from each module's own metadata, read from metal2vulkan's code at the pin).
/// False, with `out` untouched, where `bytes` are not that container or do not read whole. The
/// layout (measured: macOS 26.7.1, the Intel UHD 630's and the AMD Radeon Pro 5300M's devices, 15
/// programs of our own of one or two kernels; macOS 27, the M3 Max's, two of them):
///   - the u32 1, then 12 zero bytes;
///   - from 0x10, ten pairs of a u32 offset and a u32 size, of which the first places the
///     function table and the fourth the modules (the others are not read: the second holds 8
///     zero bytes a function, the sixth "default.metallib", the seventh a u32 count and the names);
///   - the function table: a u16 (8 on macOS 26.7.1, 9 on macOS 27; not read), the u16s 2, 1
///     and 1, a u32 count, then each function's seven u32s
///     and its NUL-terminated name. The first u32 is the stage (2, a kernel, the only one seen),
///     the third the function's offset in the second section, the fifth its module's offset from
///     the fourth's; the second, fourth and sixth were 0 and the seventh 0xfc (field_04, field_0c,
///     field_14, field_18: not read);
///   - each module: its u32 size, then the module in a bitcode wrapper.
bool RgfxMetallibOfClLibrary(const uint8_t *bytes, size_t length, std::vector<uint8_t> &out);

/// The bitcode inside a bitcode wrapper (0x0B17C0DE, then the u32 version, offset, size and CPU
/// type), as LLVM writes a module of an Apple target: the `*length` bytes at `bitcode` where they
/// are not wrapped, else the bytes the wrapper names, `*length` set to their size. A .metallib's
/// section is wrapped once, and Metal's compiler refuses a section whose bitcode is wrapped again
/// (measured: macOS 26.7.1, its service aborts reading the module).
const uint8_t *RgfxUnwrapBitcode(const uint8_t *bitcode, size_t *length);

#endif // RGFX_METALLIB_WRITER_HPP_
