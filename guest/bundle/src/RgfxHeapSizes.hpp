/*
 * RgfxHeapSizes: what the device answers heapBufferSizeAndAlignWithLength:options: and
 * heapTextureSizeAndAlignWithDescriptor:, the sizes an app places resources in a heap by.
 *
 * - A buffer takes its length, at the alignment of the host's answer for one page of Private
 *   storage (the device-info blob's `heap_buffer_page_private`). Metal answers the length at a
 *   multiple of 256 for every length and options (measured: M3 Max, macOS 27, lengths 1 to 3 MiB,
 *   every storage and cache mode), and so does metal2vulkan (read: its runtime's heap.rs).
 * - A texture's answer depends on the host GPU's layout, and no query on the wire carries it yet
 *   (protocol-gaps-wire.md), so the device answers a bound that holds Metal's answer for every
 *   shape measured (M3 Max, macOS 27: 806 descriptors of 1D, 2D, arrays, cubes, 3D, multisample,
 *   color, depth, stencil, BC, ASTC, ETC2/EAC and PVRTC formats; the bound is 1.06 to 640 times
 *   Metal's size, about 1.1 times for a texture of a few MiB and more):
 *     - each level's blocks rounded up to the 16 KiB tile of its block size (128x128 blocks of
 *       1 byte, 128x64 of 2, 64x64 of 4, 64x32 of 8, 32x32 of 16), with every depth slice of a
 *       3D texture at each level (Metal's 3D mip chains keep more than half each level); a 1D
 *       level's row at a multiple of 128 bytes;
 *     - times the samples, plus an eighth for compression metadata, rounded up to 16 KiB, for
 *       each slice (array element, cube face);
 *     - at an alignment of 16 KiB, the coarsest Metal answers (128 B for a lone 2D texture,
 *       16 KiB for arrays, cubes and 3D).
 *   A heap sized by it holds the textures, with room to spare; it over-estimates small
 *   textures most (a 1x1 texture takes 16 KiB). The host's own answer replaces it once the wire
 *   carries the query.
 *
 * It also answers the size an IOSurface takes for a texture over it (rgfxSurfaceTextureAllocSize,
 * copyIOSurfaceSharedTextureProperties:'s IOSurfaceAllocSize).
 *
 * Header-only and free of Metal's headers, so the tests build it as the bundle does. Our own
 * code.
 */
#ifndef RGFX_HEAP_SIZES_HPP_
#define RGFX_HEAP_SIZES_HPP_

#include <stdint.h>

#include "RgfxManagedLayout.hpp"

/// A heap placement's size and alignment, as MTLSizeAndAlign holds them.
struct RgfxHeapSizeAlign {
	uint64_t size;
	uint64_t align;
};

/// Metal's buffer alignment in a heap (measured above), where the host gave no answer.
constexpr uint64_t kRgfxHeapBufferAlign = 256;
/// The alignment and the slice granule of the texture bound.
constexpr uint64_t kRgfxHeapTextureGranule = 16384;

/// heapBufferSizeAndAlignWithLength:options: of `length`, from the host's alignment for a page
/// (0 where the host had no answer, which takes Metal's 256).
inline RgfxHeapSizeAlign rgfxHeapBufferSizeAndAlign(uint64_t length, uint64_t hostAlign)
{
	return {length, hostAlign != 0 ? hostAlign : kRgfxHeapBufferAlign};
}

/// The block of `format` for the bound: the CPU copy's block (RgfxManagedLayout.hpp), then the
/// depth, stencil and PVRTC formats, else 16 bytes a texel, the largest texel there is.
inline RgfxFormatBlock rgfxHeapFormatBlock(uint32_t format)
{
	RgfxFormatBlock block = {};
	if (rgfxPixelFormatBlock(format, &block))
		return block;
	switch (format) {
	case 253: // Stencil8
		return {1, 1, 1, false};
	case 250: // Depth16Unorm
		return {2, 1, 1, false};
	case 252: // Depth32Float
	case 255: // Depth24Unorm_Stencil8
	case 262: // X24_Stencil8
		return {4, 1, 1, false};
	case 260: // Depth32Float_Stencil8
	case 261: // X32_Stencil8
		return {8, 1, 1, false};
	case 160: case 161: case 164: case 165: // PVRTC 2 bits a texel
		return {8, 8, 4, false};
	case 162: case 163: case 166: case 167: // PVRTC 4 bits a texel
		return {8, 4, 4, false};
	default:
		return {16, 1, 1, false};
	}
}

namespace rgfx_heap_detail {

inline bool alignUp(uint64_t v, uint64_t a, uint64_t *out)
{
	if (__builtin_add_overflow(v, a - 1, out))
		return false;
	*out -= *out % a;
	return true;
}

} // namespace rgfx_heap_detail

/// heapTextureSizeAndAlignWithDescriptor:'s bound (see above) for a texture of `format` and
/// `MTLTextureType` `type`, `width` x `height` x `depth` texels, `levels` mipmap levels,
/// `arrayLength` elements and `samples` samples. False for a descriptor no texture has (a 0
/// dimension or count) or one past 2^64 bytes, which Metal answers 0 bytes.
inline bool rgfxHeapTextureBound(uint32_t format, uint32_t type, uint64_t width, uint64_t height,
                                 uint64_t depth, uint32_t levels, uint64_t arrayLength,
                                 uint64_t samples, RgfxHeapSizeAlign *out)
{
	using rgfx_heap_detail::alignUp;
	*out = {0, kRgfxHeapTextureGranule};
	if (width == 0 || height == 0 || depth == 0 || levels == 0 || levels > 32 ||
	    arrayLength == 0 || samples == 0)
		return false;
	const RgfxFormatBlock block = rgfxHeapFormatBlock(format);
	uint64_t tileW = 32, tileH = 32;
	switch (block.bytes) {
	case 1: tileW = 128; tileH = 128; break;
	case 2: tileW = 128; tileH = 64; break;
	case 4: tileW = 64; tileH = 64; break;
	case 8: tileW = 64; tileH = 32; break;
	default: break;
	}
	const bool oneD = type == 0 || type == 1 || type == 9; // 1D, 1DArray, TextureBuffer
	const bool threeD = type == 7;
	const bool cube = type == 5 || type == 6;
	uint64_t slice = 0;
	for (uint32_t l = 0; l < levels; l++) {
		const uint64_t w = (width >> l) ? (width >> l) : 1;
		const uint64_t h = (height >> l) ? (height >> l) : 1;
		const uint64_t bw = (w + block.width - 1) / block.width;
		const uint64_t bh = (h + block.height - 1) / block.height;
		uint64_t level = 0, rows = 0;
		if (oneD) {
			if (__builtin_mul_overflow(bw, block.bytes, &level) || !alignUp(level, 128, &level))
				return false;
		} else if (!alignUp(bw, tileW, &level) || !alignUp(bh, tileH, &rows) ||
		           __builtin_mul_overflow(level, rows, &level) ||
		           __builtin_mul_overflow(level, block.bytes, &level) ||
		           __builtin_mul_overflow(level, threeD ? depth : 1, &level)) {
			return false;
		}
		if (__builtin_add_overflow(slice, level, &slice))
			return false;
	}
	uint64_t slices = 0, total = 0;
	if (__builtin_mul_overflow(slice, samples, &slice) ||
	    __builtin_add_overflow(slice, slice / 8, &slice) ||
	    !alignUp(slice, kRgfxHeapTextureGranule, &slice) ||
	    __builtin_mul_overflow(arrayLength, cube ? 6 : 1, &slices) ||
	    __builtin_mul_overflow(slice, slices, &total))
		return false;
	out->size = total;
	return true;
}

/// The bytes an IOSurface holds for a texture of `format` and `MTLTextureType` `type`, `width` x
/// `height` x `depth` texels, `levels` mipmap levels, `arrayLength` elements and `samples`
/// samples, laid out as the device's textures over a surface read one, linear: every level in
/// order, each its slices (array elements, six faces a cube) and samples, each slice its images
/// (the level's depth for a 3D texture, else one), each image its rows of blocks (the block of
/// rgfxHeapFormatBlock), each row's bytes at a multiple of `rowAlign` (the host's linear texture
/// alignment, which a surface's rows meet for the device to use its pages in place). False for a descriptor no texture has (a 0 dimension or count), or past 2^64.
inline bool rgfxSurfaceTextureAllocSize(uint32_t format, uint32_t type, uint64_t width,
                                        uint64_t height, uint64_t depth, uint32_t levels,
                                        uint64_t arrayLength, uint64_t samples, uint64_t rowAlign,
                                        uint64_t *out)
{
	using rgfx_heap_detail::alignUp;
	*out = 0;
	if (width == 0 || height == 0 || depth == 0 || levels == 0 || levels > 32 ||
	    arrayLength == 0 || samples == 0 || rowAlign == 0)
		return false;
	const RgfxFormatBlock block = rgfxHeapFormatBlock(format);
	const bool threeD = type == 7;
	const bool cube = type == 5 || type == 6;
	uint64_t slices = 0, total = 0;
	if (__builtin_mul_overflow(arrayLength, cube ? 6 : 1, &slices) ||
	    __builtin_mul_overflow(slices, samples, &slices))
		return false;
	for (uint32_t l = 0; l < levels; l++) {
		const uint64_t w = (width >> l) ? (width >> l) : 1;
		const uint64_t h = (height >> l) ? (height >> l) : 1;
		const uint64_t d = threeD ? ((depth >> l) ? (depth >> l) : 1) : 1;
		uint64_t row = 0, level = 0;
		if (__builtin_mul_overflow((w + block.width - 1) / block.width, block.bytes, &row) ||
		    !alignUp(row, rowAlign, &row) ||
		    __builtin_mul_overflow(row, (h + block.height - 1) / block.height, &level) ||
		    __builtin_mul_overflow(level, d, &level) ||
		    __builtin_mul_overflow(level, slices, &level) ||
		    __builtin_add_overflow(total, level, &total))
			return false;
	}
	*out = total;
	return true;
}

#endif // RGFX_HEAP_SIZES_HPP_
