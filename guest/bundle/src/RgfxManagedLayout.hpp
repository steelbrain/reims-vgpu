/*
 * RgfxManagedLayout: a Managed texture's CPU copy, the region block the guest maps beside the
 * GPU's copy (docs/graphics-kext/design.md, Decided 4).
 *
 * - rgfxPixelFormatBlock(): the bytes and the size in texels of one block of texels of an
 *   `MTLPixelFormat`, for the formats a CPU copy holds: the uncompressed color formats whose
 *   texels are a power of two of bytes, and the BC, EAC/ETC2 and ASTC block formats. Depth,
 *   stencil, PVRTC and the three-component 8/16/32-bit formats have none: their textures keep
 *   no CPU copy.
 * - RgfxManagedLayout: where each level and slice lies in the copy. The host computes the same
 *   layout from the create's fields (crates/reims-graphics/src/managed.rs, whose test holds the
 *   cases tests/shell_tests.mm does): every level in order; each level its slices (the array's
 *   elements, six faces a cube); each slice its images (the level's depth for a 3D texture,
 *   else one); each image its rows of blocks, each row's bytes rounded up to
 *   RGFX_MANAGED_ROW_ALIGN.
 *
 * Header-only and free of Metal's headers, so the tests build it as the bundle does. Our own
 * code.
 */
#ifndef RGFX_MANAGED_LAYOUT_HPP_
#define RGFX_MANAGED_LAYOUT_HPP_

#include <stdint.h>

#include "reims_graphics_wire.h"

/// One block of texels of a pixel format.
struct RgfxFormatBlock {
	uint32_t bytes;
	uint32_t width;
	uint32_t height;
	/// A texture over a buffer can have the format: one texel a block, neither depth nor
	/// stencil (RGFX_MANAGED_TEXTURE_LINEAR).
	bool linear;
};

/// The block of `MTLPixelFormat` `format`, where a CPU copy holds the format.
inline bool rgfxPixelFormatBlock(uint32_t format, RgfxFormatBlock *out)
{
	auto texel = [&](uint32_t bytes) {
		*out = {bytes, 1, 1, true};
		return true;
	};
	auto block = [&](uint32_t bytes, uint32_t w, uint32_t h) {
		*out = {bytes, w, h, false};
		return true;
	};
	switch (format) {
	case 1:                                // A8Unorm
	case 10: case 11: case 12: case 13: case 14: // R8
		return texel(1);
	case 20: case 22: case 23: case 24: case 25: // R16
	case 30: case 31: case 32: case 33: case 34: // RG8
	case 40: case 41: case 42: case 43:          // B5G6R5, A1BGR5, ABGR4, BGR5A1
		return texel(2);
	case 53: case 54: case 55:                   // R32
	case 60: case 62: case 63: case 64: case 65: // RG16
	case 70: case 71: case 72: case 73: case 74: // RGBA8
	case 80: case 81:                            // BGRA8
	case 90: case 91: case 92: case 93: case 94: // RGB10A2, RG11B10, RGB9E5, BGR10A2
	case 554: case 555:                          // BGR10_XR
		return texel(4);
	case 103: case 104: case 105:                     // RG32
	case 110: case 112: case 113: case 114: case 115: // RGBA16
	case 552: case 553:                               // BGRA10_XR
		return texel(8);
	case 123: case 124: case 125: // RGBA32
		return texel(16);
	case 240: case 241: // GBGR422, BGRG422: two texels a block
		return block(4, 2, 1);
	case 130: case 131: // BC1
	case 140: case 141: // BC4
	case 170: case 172: // EAC_R11
	case 180: case 181: // ETC2_RGB8
	case 182: case 183: // ETC2_RGB8A1
		return block(8, 4, 4);
	case 132: case 133: case 134: case 135: // BC2, BC3
	case 142: case 143:                     // BC5
	case 150: case 151: case 152: case 153: // BC6H, BC7
	case 174: case 176:                     // EAC_RG11
	case 178: case 179:                     // EAC_RGBA8
		return block(16, 4, 4);
	default:
		break;
	}
	// ASTC: sRGB from 186, LDR from 204, HDR from 222, each in the same order of block sizes
	// (191, 209 and 227 are none).
	static const uint8_t kAstc[][2] = {{4, 4},  {5, 4},   {5, 5},   {6, 5},   {6, 6},
	                                   {0, 0},  {8, 5},   {8, 6},   {8, 8},   {10, 5},
	                                   {10, 6}, {10, 8},  {10, 10}, {12, 10}, {12, 12}};
	static const uint32_t kAstcBases[] = {186, 204, 222};
	for (uint32_t base : kAstcBases) {
		if (format >= base && format < base + 15 && kAstc[format - base][0] != 0)
			return block(16, kAstc[format - base][0], kAstc[format - base][1]);
	}
	return false;
}

/// Whether the host lays a Managed texture of `desc`, of a format of block `block`, over its CPU
/// copy on a GPU that shares the CPU's memory: a one-level, one-sample 2D texture of a format a
/// texture over a buffer can have, as the host's in_place_shape asks
/// (crates/reims-vgpu/src/graphics/decode/managed.rs). Any other shape is two copies there too.
inline bool rgfxManagedLiesOverCopy(const RgfxTextureDescriptor &desc,
                                    const RgfxFormatBlock &block)
{
	return block.linear && desc.texture_type == 2 /* MTLTextureType2D */ &&
	       desc.mipmap_level_count == 1 && desc.sample_count <= 1 && desc.array_length <= 1;
}

/// Where one level of a CPU copy lies.
struct RgfxManagedLevel {
	uint64_t offset;   // of its first slice
	uint64_t size[3];  // texels: width, height, depth
	uint64_t bytesPerRow;
	uint64_t bytesPerImage;
	uint64_t bytesPerSlice;
};

/// A Managed texture's CPU copy layout.
class RgfxManagedLayout {
public:
	static constexpr uint32_t kMaxLevels = 32;

	/// The layout of texture `desc` of blocks `block`; false for a shape no CPU copy has (a
	/// dimension or a count of 0, more than one sample, more levels than a side halves to, a
	/// size past 2^64), as the host refuses it.
	bool init(const RgfxTextureDescriptor &desc, const RgfxFormatBlock &block)
	{
		const bool threeD = desc.texture_type == 7; // MTLTextureType3D
		const uint64_t w = desc.width, h = desc.height, d = threeD ? desc.depth : 1;
		uint64_t largest = w > h ? w : h;
		largest = largest > d ? largest : d;
		if (block.bytes == 0 || block.width == 0 || block.height == 0 || w == 0 || h == 0 ||
		    d == 0 || desc.array_length == 0 || desc.sample_count > 1 ||
		    desc.mipmap_level_count == 0 || desc.mipmap_level_count > kMaxLevels ||
		    (largest >> (desc.mipmap_level_count - 1)) == 0)
			return false;
		const bool cube = desc.texture_type == 5 || desc.texture_type == 6;
		m_slices = static_cast<uint64_t>(desc.array_length) * (cube ? 6 : 1);
		m_block = block;
		m_levels = desc.mipmap_level_count;
		uint64_t offset = 0;
		for (uint32_t l = 0; l < m_levels; l++) {
			RgfxManagedLevel &lv = m_level[l];
			lv.offset = offset;
			lv.size[0] = (w >> l) ? (w >> l) : 1;
			lv.size[1] = (h >> l) ? (h >> l) : 1;
			lv.size[2] = (d >> l) ? (d >> l) : 1;
			uint64_t row, image, slice, level, end;
			if (__builtin_mul_overflow(divCeil(lv.size[0], block.width), block.bytes, &row) ||
			    __builtin_add_overflow(row, RGFX_MANAGED_ROW_ALIGN - 1, &row))
				return false;
			row &= ~(RGFX_MANAGED_ROW_ALIGN - 1);
			if (__builtin_mul_overflow(row, divCeil(lv.size[1], block.height), &image) ||
			    __builtin_mul_overflow(image, lv.size[2], &slice) ||
			    __builtin_mul_overflow(slice, m_slices, &level) ||
			    __builtin_add_overflow(offset, level, &end))
				return false;
			lv.bytesPerRow = row;
			lv.bytesPerImage = image;
			lv.bytesPerSlice = slice;
			offset = end;
		}
		m_size = offset;
		return true;
	}

	uint32_t levels() const { return m_levels; }
	uint64_t slices() const { return m_slices; }
	uint64_t size() const { return m_size; }
	const RgfxFormatBlock &block() const { return m_block; }
	const RgfxManagedLevel &level(uint32_t l) const { return m_level[l]; }

	/// Where the texels of `size` at `origin` of `level` and `slice` lie: the offset of the
	/// first block; false for an empty region, one past the level or the slices, or one whose
	/// origin is not at a block's start.
	bool region(uint32_t level, uint64_t slice, const uint64_t origin[3], const uint64_t size[3],
	            uint64_t *offset) const
	{
		if (level >= m_levels || slice >= m_slices)
			return false;
		const RgfxManagedLevel &lv = m_level[level];
		for (int i = 0; i < 3; i++) {
			if (size[i] == 0 || origin[i] > lv.size[i] || size[i] > lv.size[i] - origin[i])
				return false;
		}
		if (origin[0] % m_block.width || origin[1] % m_block.height)
			return false;
		*offset = lv.offset + slice * lv.bytesPerSlice + origin[2] * lv.bytesPerImage +
		          origin[1] / m_block.height * lv.bytesPerRow +
		          origin[0] / m_block.width * m_block.bytes;
		return true;
	}

	/// The bytes of a row of `width` texels of blocks, and the rows of blocks of `height`.
	uint64_t rowBytes(uint64_t width) const { return divCeil(width, m_block.width) * m_block.bytes; }
	uint64_t rows(uint64_t height) const { return divCeil(height, m_block.height); }

private:
	static uint64_t divCeil(uint64_t a, uint64_t b) { return (a + b - 1) / b; }

	RgfxManagedLevel m_level[kMaxLevels] = {};
	uint32_t m_levels = 0;
	uint64_t m_slices = 0;
	uint64_t m_size = 0;
	RgfxFormatBlock m_block = {};
};

#endif // RGFX_MANAGED_LAYOUT_HPP_
