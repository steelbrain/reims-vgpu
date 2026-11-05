/*
 * RgfxTexture: see RgfxTexture.h. Our own ObjC++; no Apple private SPI.
 */

#import "RgfxTexture.h"

#import "RgfxBuffer.h"
#import "RgfxHeap.h"
#import "RgfxMetalError.h"
#import "RgfxMetalPrivate.h"
#import "RgfxUnsupported.h"

#include <cstring>
#include <memory>

#include "RgfxManagedLayout.hpp"
#include "RgfxRegion.hpp"
#include "RgfxSerializer.hpp" // RgfxSerializer, RgfxTextureDescriptor (wire)

#import <objc/runtime.h>

#include <algorithm>
#include <atomic>
#include <cstring>

/// Maps an MTLTextureDescriptor into the wire's RgfxTextureDescriptor. Each Metal
/// enum is carried as its raw value (the wire's convention); the four swizzle
/// channels are packed one byte each, red lowest, as RgfxTextureDescriptor.swizzle
/// asks. `allowGPUOptimizedContents` and `compressionType` have no wire field yet
/// (a follow-up); every other descriptor field is carried.
RgfxTextureDescriptor rgfxTextureDescriptorFromMTL(MTLTextureDescriptor *d)
{
	RgfxTextureDescriptor out = {};
	out.pixel_format = static_cast<uint32_t>(d.pixelFormat);
	out.texture_type = static_cast<uint32_t>(d.textureType);
	out.width = static_cast<uint32_t>(d.width);
	out.height = static_cast<uint32_t>(d.height);
	out.depth = static_cast<uint32_t>(d.depth);
	out.mipmap_level_count = static_cast<uint32_t>(d.mipmapLevelCount);
	out.array_length = static_cast<uint32_t>(d.arrayLength);
	out.sample_count = static_cast<uint32_t>(d.sampleCount);
	out.usage = static_cast<uint32_t>(d.usage);
	out.options = static_cast<uint32_t>(d.resourceOptions);
	MTLTextureSwizzleChannels sw = d.swizzle;
	out.swizzle = static_cast<uint32_t>(sw.red) | (static_cast<uint32_t>(sw.green) << 8) |
	              (static_cast<uint32_t>(sw.blue) << 16) | (static_cast<uint32_t>(sw.alpha) << 24);
	return out;
}

namespace {

// A block of a pixel format: its bytes and its width and height in pixels (1 x 1 where the
// format is not block-compressed). 0 bytes for a format this does not size.
struct RgfxBlockShape {
	uint32_t bytes, width, height;
};

RgfxBlockShape rgfxBlockShapeOf(MTLPixelFormat f)
{
	switch (f) {
	case MTLPixelFormatA8Unorm:
	case MTLPixelFormatR8Unorm:
	case MTLPixelFormatR8Unorm_sRGB:
	case MTLPixelFormatR8Snorm:
	case MTLPixelFormatR8Uint:
	case MTLPixelFormatR8Sint:
	case MTLPixelFormatStencil8:
		return {1, 1, 1};
	case MTLPixelFormatR16Unorm:
	case MTLPixelFormatR16Snorm:
	case MTLPixelFormatR16Uint:
	case MTLPixelFormatR16Sint:
	case MTLPixelFormatR16Float:
	case MTLPixelFormatRG8Unorm:
	case MTLPixelFormatRG8Unorm_sRGB:
	case MTLPixelFormatRG8Snorm:
	case MTLPixelFormatRG8Uint:
	case MTLPixelFormatRG8Sint:
	case MTLPixelFormatB5G6R5Unorm:
	case MTLPixelFormatA1BGR5Unorm:
	case MTLPixelFormatABGR4Unorm:
	case MTLPixelFormatBGR5A1Unorm:
	case MTLPixelFormatDepth16Unorm:
		return {2, 1, 1};
	case MTLPixelFormatGBGR422:
	case MTLPixelFormatBGRG422:
		return {4, 2, 1};
	case MTLPixelFormatR32Uint:
	case MTLPixelFormatR32Sint:
	case MTLPixelFormatR32Float:
	case MTLPixelFormatRG16Unorm:
	case MTLPixelFormatRG16Snorm:
	case MTLPixelFormatRG16Uint:
	case MTLPixelFormatRG16Sint:
	case MTLPixelFormatRG16Float:
	case MTLPixelFormatRGBA8Unorm:
	case MTLPixelFormatRGBA8Unorm_sRGB:
	case MTLPixelFormatRGBA8Snorm:
	case MTLPixelFormatRGBA8Uint:
	case MTLPixelFormatRGBA8Sint:
	case MTLPixelFormatBGRA8Unorm:
	case MTLPixelFormatBGRA8Unorm_sRGB:
	case MTLPixelFormatRGB10A2Unorm:
	case MTLPixelFormatRGB10A2Uint:
	case MTLPixelFormatRG11B10Float:
	case MTLPixelFormatRGB9E5Float:
	case MTLPixelFormatBGR10A2Unorm:
	case MTLPixelFormatBGR10_XR:
	case MTLPixelFormatBGR10_XR_sRGB:
	case MTLPixelFormatDepth32Float:
	case MTLPixelFormatDepth24Unorm_Stencil8:
	case MTLPixelFormatX24_Stencil8:
		return {4, 1, 1};
	case MTLPixelFormatRG32Uint:
	case MTLPixelFormatRG32Sint:
	case MTLPixelFormatRG32Float:
	case MTLPixelFormatRGBA16Unorm:
	case MTLPixelFormatRGBA16Snorm:
	case MTLPixelFormatRGBA16Uint:
	case MTLPixelFormatRGBA16Sint:
	case MTLPixelFormatRGBA16Float:
	case MTLPixelFormatBGRA10_XR:
	case MTLPixelFormatBGRA10_XR_sRGB:
	case MTLPixelFormatDepth32Float_Stencil8:
	case MTLPixelFormatX32_Stencil8:
		return {8, 1, 1};
	case MTLPixelFormatRGBA32Uint:
	case MTLPixelFormatRGBA32Sint:
	case MTLPixelFormatRGBA32Float:
		return {16, 1, 1};
	case MTLPixelFormatBC1_RGBA:
	case MTLPixelFormatBC1_RGBA_sRGB:
	case MTLPixelFormatBC4_RUnorm:
	case MTLPixelFormatBC4_RSnorm:
	case MTLPixelFormatEAC_R11Unorm:
	case MTLPixelFormatEAC_R11Snorm:
	case MTLPixelFormatETC2_RGB8:
	case MTLPixelFormatETC2_RGB8_sRGB:
	case MTLPixelFormatETC2_RGB8A1:
	case MTLPixelFormatETC2_RGB8A1_sRGB:
		return {8, 4, 4};
	case MTLPixelFormatBC2_RGBA:
	case MTLPixelFormatBC2_RGBA_sRGB:
	case MTLPixelFormatBC3_RGBA:
	case MTLPixelFormatBC3_RGBA_sRGB:
	case MTLPixelFormatBC5_RGUnorm:
	case MTLPixelFormatBC5_RGSnorm:
	case MTLPixelFormatBC6H_RGBFloat:
	case MTLPixelFormatBC6H_RGBUfloat:
	case MTLPixelFormatBC7_RGBAUnorm:
	case MTLPixelFormatBC7_RGBAUnorm_sRGB:
	case MTLPixelFormatEAC_RG11Unorm:
	case MTLPixelFormatEAC_RG11Snorm:
	case MTLPixelFormatEAC_RGBA8:
	case MTLPixelFormatEAC_RGBA8_sRGB:
		return {16, 4, 4};
	case MTLPixelFormatASTC_4x4_sRGB:
	case MTLPixelFormatASTC_4x4_LDR:
	case MTLPixelFormatASTC_4x4_HDR:
		return {16, 4, 4};
	case MTLPixelFormatASTC_5x4_sRGB:
	case MTLPixelFormatASTC_5x4_LDR:
	case MTLPixelFormatASTC_5x4_HDR:
		return {16, 5, 4};
	case MTLPixelFormatASTC_5x5_sRGB:
	case MTLPixelFormatASTC_5x5_LDR:
	case MTLPixelFormatASTC_5x5_HDR:
		return {16, 5, 5};
	case MTLPixelFormatASTC_6x5_sRGB:
	case MTLPixelFormatASTC_6x5_LDR:
	case MTLPixelFormatASTC_6x5_HDR:
		return {16, 6, 5};
	case MTLPixelFormatASTC_6x6_sRGB:
	case MTLPixelFormatASTC_6x6_LDR:
	case MTLPixelFormatASTC_6x6_HDR:
		return {16, 6, 6};
	case MTLPixelFormatASTC_8x5_sRGB:
	case MTLPixelFormatASTC_8x5_LDR:
	case MTLPixelFormatASTC_8x5_HDR:
		return {16, 8, 5};
	case MTLPixelFormatASTC_8x6_sRGB:
	case MTLPixelFormatASTC_8x6_LDR:
	case MTLPixelFormatASTC_8x6_HDR:
		return {16, 8, 6};
	case MTLPixelFormatASTC_8x8_sRGB:
	case MTLPixelFormatASTC_8x8_LDR:
	case MTLPixelFormatASTC_8x8_HDR:
		return {16, 8, 8};
	case MTLPixelFormatASTC_10x5_sRGB:
	case MTLPixelFormatASTC_10x5_LDR:
	case MTLPixelFormatASTC_10x5_HDR:
		return {16, 10, 5};
	case MTLPixelFormatASTC_10x6_sRGB:
	case MTLPixelFormatASTC_10x6_LDR:
	case MTLPixelFormatASTC_10x6_HDR:
		return {16, 10, 6};
	case MTLPixelFormatASTC_10x8_sRGB:
	case MTLPixelFormatASTC_10x8_LDR:
	case MTLPixelFormatASTC_10x8_HDR:
		return {16, 10, 8};
	case MTLPixelFormatASTC_10x10_sRGB:
	case MTLPixelFormatASTC_10x10_LDR:
	case MTLPixelFormatASTC_10x10_HDR:
		return {16, 10, 10};
	case MTLPixelFormatASTC_12x10_sRGB:
	case MTLPixelFormatASTC_12x10_LDR:
	case MTLPixelFormatASTC_12x10_HDR:
		return {16, 12, 10};
	case MTLPixelFormatASTC_12x12_sRGB:
	case MTLPixelFormatASTC_12x12_LDR:
	case MTLPixelFormatASTC_12x12_HDR:
		return {16, 12, 12};
	default:
		return {0, 1, 1};
	}
}

// Whether `f`'s texels are stored in blocks of more than one (BC, ETC, EAC, ASTC, PVRTC) or
// share their chroma between two (the 4:2:2 formats).
bool rgfxIsBlockOrSubsampled(MTLPixelFormat f)
{
	const RgfxBlockShape b = rgfxBlockShapeOf(f);
	return b.width > 1 || b.height > 1 ||
	       (f >= MTLPixelFormatPVRTC_RGB_2BPP && f <= MTLPixelFormatPVRTC_RGBA_4BPP_sRGB);
}

// A CPU copy's shape: the region's rows of blocks, each row's bytes, and its images (depth).
struct RgfxCopyShape {
	uint64_t rowBytes, rows, images;
};

// The shape of a CPU copy of `region` of a texture in `format`; false where the format is not
// sized here or the region is empty.
bool rgfxCopyShapeOf(MTLPixelFormat format, MTLRegion region, RgfxCopyShape *out)
{
	RgfxBlockShape b = rgfxBlockShapeOf(format);
	if (b.bytes == 0 || region.size.width == 0 || region.size.height == 0 ||
	    region.size.depth == 0) {
		return false;
	}
	out->rowBytes = (region.size.width + b.width - 1) / b.width * b.bytes;
	out->rows = (region.size.height + b.height - 1) / b.height;
	out->images = region.size.depth;
	return true;
}

// Logs, once per kind, a CPU copy the texture does not make.
void rgfxNoteCopyRefused(std::atomic<bool> &logged, NSString *what, RgfxTexture *t)
{
	if (!logged.exchange(true)) {
		NSLog(@"RGFX: texture 0x%llx (format %lu, storage mode %lu): %@; nothing is copied",
		      (unsigned long long)[t rgfxObjId], (unsigned long)t.pixelFormat,
		      (unsigned long)t.storageMode, what);
	}
}

// The device's queue for the CPU's copies into and out of its textures, made at the first one
// and kept with the device.
id<MTLCommandQueue> rgfxCopyQueue(id<MTLDevice> device)
{
	static char key;
	if (device == nil) {
		return nil;
	}
	@synchronized(device) {
		id<MTLCommandQueue> queue = objc_getAssociatedObject(device, &key);
		if (queue == nil) {
			queue = [device newCommandQueue];
			if (queue != nil) {
				objc_setAssociatedObject(device, &key, queue, OBJC_ASSOCIATION_RETAIN_NONATOMIC);
			}
		}
		return queue;
	}
}

// --- views ---------------------------------------------------------------------

// MTLTextureSwizzleChannelsDefault, one byte a channel, red lowest.
constexpr uint32_t kRgfxDefaultSwizzle = 0x05040302;

// The usage a view of a swizzle other than the default keeps of its texture's: ShaderRead,
// PixelFormatView and the two bits Metal sets of its own, 0x4000 and 0x10000 (measured on the
// M3 Max under macOS 27.0.1: a texture of usage 0x17 answers a swizzled view of 0x11, and of a
// default swizzle one of 0x17).
constexpr uint64_t kRgfxSwizzledViewUsage =
    MTLTextureUsageShaderRead | MTLTextureUsagePixelFormatView | 0x4000 | 0x10000;

// What a texture made with MTLTextureUsageUnknown answers for its usage: ShaderRead, ShaderWrite,
// RenderTarget and PixelFormatView, 0x17, private, shared or over an IOSurface, its views and a
// CAMetalLayer drawable of a layer with framebufferOnly NO too (measured on the M3 Max under
// macOS 27.0.1, and the AMD 5300M and the Intel UHD 630 under 26.7.1). The wire carries the
// usage the descriptor gave, 0 among them, which the host's platform takes as Metal does.
constexpr uint64_t kRgfxUnknownUsageAnswer = MTLTextureUsageShaderRead |
                                             MTLTextureUsageShaderWrite |
                                             MTLTextureUsageRenderTarget |
                                             MTLTextureUsagePixelFormatView;

uint64_t rgfxAnsweredUsage(uint64_t usage)
{
	return usage == MTLTextureUsageUnknown ? kRgfxUnknownUsageAnswer : usage;
}

uint32_t rgfxPackSwizzle(MTLTextureSwizzleChannels sw)
{
	return static_cast<uint32_t>(sw.red) | (static_cast<uint32_t>(sw.green) << 8) |
	       (static_cast<uint32_t>(sw.blue) << 16) | (static_cast<uint32_t>(sw.alpha) << 24);
}

// The slices of a texture of `type` with `arrayLength` elements: an array's elements, six
// faces each for a cube; one for any other type.
uint64_t rgfxSlicesOf(MTLTextureType type, uint64_t arrayLength)
{
	switch (type) {
	case MTLTextureType1DArray:
	case MTLTextureType2DArray:
	case MTLTextureType2DMultisampleArray:
		return arrayLength;
	case MTLTextureTypeCube:
		return 6;
	case MTLTextureTypeCubeArray:
		return 6 * arrayLength;
	default:
		return 1;
	}
}

// The array length a view of `type` over `slices` slices answers.
uint64_t rgfxArrayLengthOf(MTLTextureType type, uint64_t slices)
{
	switch (type) {
	case MTLTextureType1DArray:
	case MTLTextureType2DArray:
	case MTLTextureType2DMultisampleArray:
		return slices;
	case MTLTextureTypeCubeArray:
		return slices / 6;
	default:
		return 1;
	}
}

// The family of texture types a view may change among, as Metal's do: the 1D types; the 2D, 2D
// array and cube types; the multisample types; 3D; texture buffers (measured on the M3 Max
// under macOS 27.0.1: a 3D view of a 2D array fails Metal's assertion, a 2D and a 2D array view
// of a cube do not). 0 for a value that is no type.
int rgfxViewFamily(MTLTextureType type)
{
	switch (type) {
	case MTLTextureType1D:
	case MTLTextureType1DArray:
		return 1;
	case MTLTextureType2D:
	case MTLTextureType2DArray:
	case MTLTextureTypeCube:
	case MTLTextureTypeCubeArray:
		return 2;
	case MTLTextureType2DMultisample:
	case MTLTextureType2DMultisampleArray:
		return 3;
	case MTLTextureType3D:
		return 4;
	case MTLTextureTypeTextureBuffer:
		return 5;
	default:
		return 0;
	}
}

// Whether a view of `type` may take `count` slices, as Metal's assertion asks (measured on the
// M3 Max under macOS 27.0.1: a 2D view of two slices and a cube view of five fail it): one for a
// type with no slices of its own, a multiple of six for a cube, at least one for an array.
bool rgfxViewTakesSlices(MTLTextureType type, uint64_t count)
{
	switch (type) {
	case MTLTextureType1DArray:
	case MTLTextureType2DArray:
	case MTLTextureType2DMultisampleArray:
		return count > 0;
	case MTLTextureTypeCube:
	case MTLTextureTypeCubeArray:
		return count > 0 && count % 6 == 0;
	default:
		return count == 1;
	}
}

// Whether Metal may make a view in `to` of a texture in `from`, as far as the formats' blocks
// tell (RgfxManagedLayout.hpp): not where the blocks differ in bytes or in size, or where one
// is block-compressed and the other is not (measured on the M3 Max under macOS 27.0.1:
// RGBA16Float in RGBA8Unorm, BC1 in RG32Uint and ASTC 4x4 in RGBA32Uint fail Metal's
// assertion; RGBA8Unorm in R32Float, RG16Unorm in RGBA8Unorm and RGB10A2Unorm in R32Uint make
// a view). Every other pair, a format with no block there (depth, stencil, PVRTC) or a 4:2:2
// one among them, is the host's to refuse.
bool rgfxViewFormatsMayMatch(MTLPixelFormat from, MTLPixelFormat to)
{
	RgfxFormatBlock a = {}, b = {};
	auto subsampled = [](MTLPixelFormat f) {
		return f == MTLPixelFormatGBGR422 || f == MTLPixelFormatBGRG422;
	};
	if (subsampled(from) || subsampled(to) || !rgfxPixelFormatBlock(from, &a) ||
	    !rgfxPixelFormatBlock(to, &b)) {
		return true;
	}
	return a.bytes == b.bytes && a.width == b.width && a.height == b.height &&
	       a.linear == b.linear;
}

// Why the bundle answers a view nil without asking the host, each logged once.
enum RgfxViewRefusal {
	kNoFormat,
	kFormatClass,
	kLevels,
	kSlices,
	kTypeFamily,
	kSliceCount,
	kBufferSlices,
	kViews
};

// Metal's name of each format no multisample texture has, as its validation names it: the
// compressed formats (BC, PVRTC, EAC and ETC2, ASTC), the 4:2:2 ones, and Unspecialized.
struct RgfxNamedFormat {
	uint32_t format;
	const char *name;
};
const RgfxNamedFormat kNoMultisample[] = {
    {130, "MTLPixelFormatBC1_RGBA"},       {131, "MTLPixelFormatBC1_RGBA_sRGB"},
    {132, "MTLPixelFormatBC2_RGBA"},       {133, "MTLPixelFormatBC2_RGBA_sRGB"},
    {134, "MTLPixelFormatBC3_RGBA"},       {135, "MTLPixelFormatBC3_RGBA_sRGB"},
    {140, "MTLPixelFormatBC4_RUnorm"},     {141, "MTLPixelFormatBC4_RSnorm"},
    {142, "MTLPixelFormatBC5_RGUnorm"},    {143, "MTLPixelFormatBC5_RGSnorm"},
    {150, "MTLPixelFormatBC6H_RGBFloat"},  {151, "MTLPixelFormatBC6H_RGBUfloat"},
    {152, "MTLPixelFormatBC7_RGBAUnorm"},  {153, "MTLPixelFormatBC7_RGBAUnorm_sRGB"},
    {160, "MTLPixelFormatPVRTC_RGB_2BPP"}, {161, "MTLPixelFormatPVRTC_RGB_2BPP_sRGB"},
    {162, "MTLPixelFormatPVRTC_RGB_4BPP"}, {163, "MTLPixelFormatPVRTC_RGB_4BPP_sRGB"},
    {164, "MTLPixelFormatPVRTC_RGBA_2BPP"}, {165, "MTLPixelFormatPVRTC_RGBA_2BPP_sRGB"},
    {166, "MTLPixelFormatPVRTC_RGBA_4BPP"}, {167, "MTLPixelFormatPVRTC_RGBA_4BPP_sRGB"},
    {170, "MTLPixelFormatEAC_R11Unorm"},   {172, "MTLPixelFormatEAC_R11Snorm"},
    {174, "MTLPixelFormatEAC_RG11Unorm"},  {176, "MTLPixelFormatEAC_RG11Snorm"},
    {178, "MTLPixelFormatEAC_RGBA8"},      {179, "MTLPixelFormatEAC_RGBA8_sRGB"},
    {180, "MTLPixelFormatETC2_RGB8"},      {181, "MTLPixelFormatETC2_RGB8_sRGB"},
    {182, "MTLPixelFormatETC2_RGB8A1"},    {183, "MTLPixelFormatETC2_RGB8A1_sRGB"},
    {186, "MTLPixelFormatASTC_4x4_sRGB"},  {187, "MTLPixelFormatASTC_5x4_sRGB"},
    {188, "MTLPixelFormatASTC_5x5_sRGB"},  {189, "MTLPixelFormatASTC_6x5_sRGB"},
    {190, "MTLPixelFormatASTC_6x6_sRGB"},  {192, "MTLPixelFormatASTC_8x5_sRGB"},
    {193, "MTLPixelFormatASTC_8x6_sRGB"},  {194, "MTLPixelFormatASTC_8x8_sRGB"},
    {195, "MTLPixelFormatASTC_10x5_sRGB"}, {196, "MTLPixelFormatASTC_10x6_sRGB"},
    {197, "MTLPixelFormatASTC_10x8_sRGB"}, {198, "MTLPixelFormatASTC_10x10_sRGB"},
    {199, "MTLPixelFormatASTC_12x10_sRGB"}, {200, "MTLPixelFormatASTC_12x12_sRGB"},
    {204, "MTLPixelFormatASTC_4x4_LDR"},   {205, "MTLPixelFormatASTC_5x4_LDR"},
    {206, "MTLPixelFormatASTC_5x5_LDR"},   {207, "MTLPixelFormatASTC_6x5_LDR"},
    {208, "MTLPixelFormatASTC_6x6_LDR"},   {210, "MTLPixelFormatASTC_8x5_LDR"},
    {211, "MTLPixelFormatASTC_8x6_LDR"},   {212, "MTLPixelFormatASTC_8x8_LDR"},
    {213, "MTLPixelFormatASTC_10x5_LDR"},  {214, "MTLPixelFormatASTC_10x6_LDR"},
    {215, "MTLPixelFormatASTC_10x8_LDR"},  {216, "MTLPixelFormatASTC_10x10_LDR"},
    {217, "MTLPixelFormatASTC_12x10_LDR"}, {218, "MTLPixelFormatASTC_12x12_LDR"},
    {222, "MTLPixelFormatASTC_4x4_HDR"},   {223, "MTLPixelFormatASTC_5x4_HDR"},
    {224, "MTLPixelFormatASTC_5x5_HDR"},   {225, "MTLPixelFormatASTC_6x5_HDR"},
    {226, "MTLPixelFormatASTC_6x6_HDR"},   {228, "MTLPixelFormatASTC_8x5_HDR"},
    {229, "MTLPixelFormatASTC_8x6_HDR"},   {230, "MTLPixelFormatASTC_8x8_HDR"},
    {231, "MTLPixelFormatASTC_10x5_HDR"},  {232, "MTLPixelFormatASTC_10x6_HDR"},
    {233, "MTLPixelFormatASTC_10x8_HDR"},  {234, "MTLPixelFormatASTC_10x10_HDR"},
    {235, "MTLPixelFormatASTC_12x10_HDR"}, {236, "MTLPixelFormatASTC_12x12_HDR"},
    {240, "MTLPixelFormatGBGR422"},        {241, "MTLPixelFormatBGRG422"},
    {263, "MTLPixelFormatUnspecialized"},
};

} // namespace

// Metal's validation lists each rule a descriptor breaks, a line each, in this order (measured:
// a 4x and a 2x 2DMultisample texture of BC1, GBGR422 and Unspecialized, a 1x one of BC1, and 2D
// ones of 45 and 100; the x86 GPUs name a format they lack, ASTC among them, as an invalid one).
void rgfxEndAtInvalidTextureDescriptor(MTLTextureDescriptor *descriptor)
{
	if (descriptor == nil) {
		return;
	}
	const uint64_t format = descriptor.pixelFormat;
	const MTLTextureType type = descriptor.textureType;
	const bool multisample =
	    type == MTLTextureType2DMultisample || type == MTLTextureType2DMultisampleArray;
	const bool threeChannel = (format >= 45 && format <= 48) ||
	                          (format >= 95 && format <= 100) || (format >= 120 && format <= 122);
	const char *name = nullptr;
	for (const RgfxNamedFormat &f : kNoMultisample) {
		if (f.format == format) {
			name = f.name;
		}
	}
	if (!threeChannel && !(multisample && name != nullptr)) {
		return;
	}
	NSMutableString *words = [NSMutableString stringWithString:@"Texture Descriptor Validation\n"];
	if (threeChannel) {
		[words appendFormat:@"MTLTextureDescriptor has invalid pixelFormat (%llu).\n",
		                    (unsigned long long)format];
		if (multisample) {
			[words appendString:@"MTLTextureDescriptor pixelFormat (MTLPixelFormatInvalid) does "
			                    @"not support multisample textures.\n"];
		}
	} else {
		[words appendFormat:@"MTLTextureDescriptor pixelFormat (%s) does not support multisample "
		                    @"textures.\n",
		                    name];
		const bool subsampled = format == 240 || format == 241;
		const bool blocks = format != 263; // compressed, or 4:2:2
		if (descriptor.sampleCount > 1 && blocks) {
			[words appendString:@"MTLTextureDescriptor has a compressed pixel format set with a "
			                    @"sample count greater than one.\n"];
		} else if (descriptor.sampleCount <= 1) {
			[words appendString:@"MTLTextureDescriptor sampleCount must be > 1 for multisample "
			                    @"textures.\n"];
		}
		if (blocks) {
			[words appendString:@"MTLTextureDescriptor: compressed pixel formats require texture "
			                    @"type of MTLTextureType2D[Array], MTLTextureType3D or "
			                    @"MTLTextureTypeCube[Array].\n"];
		}
		if (subsampled) {
			[words appendString:@"MTLTextureDescriptor: MTLPixelFormat422 requires "
			                    @"MTLTextureType2D.\n"];
		}
	}
	RgfxMetalAssertionFailed("-[MTLTextureDescriptorInternal validateWithDevice:]", words);
}

@interface RgfxTexture () <RgfxMetalTextureInternals>
@end

@implementation RgfxTexture {
	RgfxTextureDescriptor _desc; // the mapped shape the getters answer from
	uint64_t _gpuAddress;        // the picked GPU address (0 when heap-placed)
	uint64_t _resourceId;        // the picked texture resource id
	// A Managed texture's CPU copy: the Shared buffer over its block, the block's bytes, and
	// their layout. Nil and null for any other texture.
	RgfxBuffer *_managedCpu;
	uint8_t *_managedBytes;
	std::unique_ptr<RgfxManagedLayout> _managedLayout;
	// A Managed texture the host keeps in two copies on a GPU that shares the CPU's memory: its
	// getBytes: reads the GPU's copy on the device (rgfxManagedGetBytes:).
	BOOL _managedReadsGpuCopy;
	// The buffer a texture made over one lies in (kept), its offset and bytes per row.
	id<MTLBuffer> _buffer;
	uint64_t _bufferOffset;
	uint64_t _bufferBytesPerRow;
	// A view's texture (kept), and the level and slice of it the view starts at; nil and 0 for
	// a texture that is no view.
	RgfxTexture *_parent;
	NSUInteger _parentLevel;
	NSUInteger _parentSlice;
	// The descriptor's members the wire's texture descriptor does not carry, answered as made
	// (rgfxTakeDescriptor:): Metal's defaults until then.
	BOOL _allowGPUOptimizedContents;
	MTLTextureCompressionType _compressionType;
	BOOL _framebufferOnly; // the descriptor's private framebufferOnly (RgfxMetalPrivate.h)
	uint64_t _number;      // uniqueIdentifier
}

namespace {
// The last uniqueIdentifier a texture took: every texture, a view among them, takes the next as
// it is made.
std::atomic<uint64_t> g_lastTextureNumber{0};
} // namespace

+ (instancetype)managedTextureWithDescriptor:(MTLTextureDescriptor *)descriptor
                                      region:(RgfxRegion *)region
                                       slabs:(RgfxSlabs *)slabs
                                      device:(id<MTLDevice>)device
                                  serializer:(RgfxSerializer *)serializer
{
	if (descriptor == nil || region == nullptr || serializer == nullptr) {
		return nil;
	}
	RgfxTextureDescriptor desc = rgfxTextureDescriptorFromMTL(descriptor);
	RgfxFormatBlock block = {};
	auto layout = std::make_unique<RgfxManagedLayout>();
	if (!rgfxPixelFormatBlock(desc.pixel_format, &block) || !layout->init(desc, block)) {
		return nil;
	}
	// A new Managed texture reads zero, as Metal's do. Its copy's memory goes back once the host
	// has dropped the copy's buffer, which goes after the texture that holds it (its destroy
	// first).
	RgfxBuffer *cpu = nil;
	uint8_t *bytes = nullptr;
	if (slabs != nullptr) {
		// A Shared buffer of the slabs, zeroed as every new one is.
		cpu = [RgfxBuffer bufferForDeviceWithLength:layout->size()
		                                    options:MTLResourceStorageModeShared
		                                      bytes:nullptr
		                                  maxLength:0
		                                      slabs:slabs
		                                     device:device
		                                 serializer:serializer];
		if (cpu == nil)
			return nil;
		bytes = static_cast<uint8_t *>([cpu contents]);
	} else {
		RgfxRegion::Block copy = region->allocate(layout->size());
		if (!copy.ok) {
			return nil;
		}
		bytes = const_cast<uint8_t *>(copy.ptr);
		std::memset(bytes, 0, layout->size());
		const uint64_t offset = copy.offset;
		cpu = [RgfxBuffer bufferOverHostMemoryWithContents:bytes
		                                      regionOffset:offset
		                                            length:layout->size()
		                                           options:MTLResourceStorageModeShared
		                                       deallocator:^{ region->release(offset); }
		                                            device:device
		                                        serializer:serializer];
		if (cpu == nil) {
			region->release(offset); // no buffer lies over the block
			return nil;
		}
	}
	RgfxCreated created = serializer->createManagedTexture(
	    desc, [cpu rgfxObjId], block.bytes, block.width, block.height,
	    block.linear ? RGFX_MANAGED_TEXTURE_LINEAR : 0);
	if (created.resource_id == 0) {
		return nil;
	}
	if (serializer->refused(created))
		return nil; // the device never got it; the copy's buffer goes with `cpu`
	RgfxTexture *t = [[RgfxTexture alloc] initWithObjId:created.obj_id
	                                         gpuAddress:created.gpu_address
	                                         resourceId:created.resource_id
	                                         descriptor:desc
	                                             device:device
	                                         serializer:serializer];
	if (t != nil) {
		t->_managedCpu = cpu;
		t->_managedBytes = bytes;
		t->_managedLayout = std::move(layout);
		t->_managedReadsGpuCopy =
		    [device hasUnifiedMemory] && !rgfxManagedLiesOverCopy(desc, block);
		[t rgfxTakeDescriptor:descriptor];
	}
	return t;
}

+ (instancetype)textureWithDescriptor:(MTLTextureDescriptor *)descriptor
                               buffer:(RgfxResource<MTLBuffer> *)buffer
                               offset:(uint64_t)offset
                          bytesPerRow:(uint64_t)bytesPerRow
                               device:(id<MTLDevice>)device
                           serializer:(RgfxSerializer *)serializer
{
	return [self rgfxTextureWithDescriptor:descriptor
	                                buffer:buffer
	                                offset:offset
	                           bytesPerRow:bytesPerRow
	                                linear:NO
	                         bytesPerImage:0
	                                device:device
	                            serializer:serializer];
}

+ (instancetype)linearTextureWithDescriptor:(MTLTextureDescriptor *)descriptor
                                     buffer:(RgfxResource<MTLBuffer> *)buffer
                                     offset:(uint64_t)offset
                                bytesPerRow:(uint64_t)bytesPerRow
                              bytesPerImage:(uint64_t)bytesPerImage
                                     device:(id<MTLDevice>)device
                                 serializer:(RgfxSerializer *)serializer
{
	return [self rgfxTextureWithDescriptor:descriptor
	                                buffer:buffer
	                                offset:offset
	                           bytesPerRow:bytesPerRow
	                                linear:YES
	                         bytesPerImage:bytesPerImage
	                                device:device
	                            serializer:serializer];
}

// A texture over `buffer`, made by the linear form's message where `linear` says so.
+ (instancetype)rgfxTextureWithDescriptor:(MTLTextureDescriptor *)descriptor
                                   buffer:(RgfxResource<MTLBuffer> *)buffer
                                   offset:(uint64_t)offset
                              bytesPerRow:(uint64_t)bytesPerRow
                                   linear:(BOOL)linear
                            bytesPerImage:(uint64_t)bytesPerImage
                                   device:(id<MTLDevice>)device
                               serializer:(RgfxSerializer *)serializer
{
	if (serializer == nullptr || descriptor == nil || buffer == nil) {
		return nil;
	}
	RgfxTextureDescriptor desc = rgfxTextureDescriptorFromMTL(descriptor);
	RgfxCreated created =
	    linear ? serializer->createLinearTextureFromBuffer([buffer rgfxObjId], offset, bytesPerRow,
	                                                       bytesPerImage, desc)
	           : serializer->createTextureFromBuffer([buffer rgfxObjId], offset, bytesPerRow, desc);
	if (created.resource_id == 0) {
		return nil;
	}
	if (serializer->refused(created))
		return nil; // the device never got it
	RgfxTexture *t = [[RgfxTexture alloc] initWithObjId:created.obj_id
	                                         gpuAddress:0
	                                         resourceId:created.resource_id
	                                         descriptor:desc
	                                             device:device
	                                         serializer:serializer];
	if (t != nil) {
		t->_buffer = buffer;
		t->_bufferOffset = offset;
		t->_bufferBytesPerRow = bytesPerRow;
		[t rgfxTakeDescriptor:descriptor];
	}
	return t;
}

- (id<MTLBuffer>)buffer
{
	return _parent != nil ? _parent.buffer : _buffer;
}

// The bytes the texture takes: its buffer's, for one made over a buffer (measured on the M3 Max
// under macOS 27.0.1: a texture of four rows over buffers of 5000, 65536 and 200000 bytes
// answers 5000, 65536 and 212992, each buffer's allocatedSize); the size its Automatic heap
// placed it by, for one made there (measured on the M3 Max under macOS 27: a texture's
// heapTextureSizeAndAlignWithDescriptor: size); else every level of every slice and sample, in
// its format's blocks (0 for a format not sized here).
- (NSUInteger)allocatedSize
{
	if (_parent != nil) {
		return _parent.allocatedSize; // a view takes its texture's memory
	}
	if (_buffer != nil) {
		return _buffer.allocatedSize;
	}
	if (self.rgfxHeapRangeSize != 0) {
		return static_cast<NSUInteger>(self.rgfxHeapRangeSize); // its Automatic heap's size of it
	}
	RgfxBlockShape b = rgfxBlockShapeOf(static_cast<MTLPixelFormat>(_desc.pixel_format));
	uint64_t slices = _desc.array_length ? _desc.array_length : 1;
	if (_desc.texture_type == MTLTextureTypeCube || _desc.texture_type == MTLTextureTypeCubeArray) {
		slices *= 6;
	}
	uint64_t samples = _desc.sample_count ? _desc.sample_count : 1;
	uint64_t total = 0;
	uint32_t levels = _desc.mipmap_level_count ? _desc.mipmap_level_count : 1;
	for (uint32_t level = 0; level < levels; level++) {
		uint64_t w = std::max<uint64_t>(1, _desc.width >> level);
		uint64_t h = std::max<uint64_t>(1, _desc.height >> level);
		uint64_t d = std::max<uint64_t>(1, _desc.depth >> level);
		total += ((w + b.width - 1) / b.width) * ((h + b.height - 1) / b.height) * d * b.bytes;
	}
	return static_cast<NSUInteger>(total * slices * samples);
}

- (NSUInteger)bufferOffset
{
	return _parent != nil ? _parent.bufferOffset : static_cast<NSUInteger>(_bufferOffset);
}

- (NSUInteger)bufferBytesPerRow
{
	return _parent != nil ? _parent.bufferBytesPerRow
	                      : static_cast<NSUInteger>(_bufferBytesPerRow);
}

+ (instancetype)textureWithDescriptor:(MTLTextureDescriptor *)descriptor
                                 heap:(RgfxObjId)heap
                           heapOffset:(uint64_t)heapOffset
                               device:(id<MTLDevice>)device
                           serializer:(RgfxSerializer *)serializer
{
	if (serializer == nullptr || descriptor == nil) {
		return nil; // a texture must have a route and a shape
	}
	RgfxTextureDescriptor desc = rgfxTextureDescriptorFromMTL(descriptor);
	RgfxCreated created = serializer->createTexture(desc, heap, heapOffset);
	if (created.resource_id == 0) {
		return nil; // no resource id left: no texture, as Metal answers one it cannot make
	}
	if (serializer->refused(created))
		return nil; // the device never got it
	RgfxTexture *t = [[RgfxTexture alloc] initWithObjId:created.obj_id
	                                        gpuAddress:created.gpu_address
	                                        resourceId:created.resource_id
	                                        descriptor:desc
	                                            device:device
	                                        serializer:serializer];
	[t rgfxTakeDescriptor:descriptor];
	return t;
}

- (instancetype)initWithObjId:(uint64_t)objId
                   gpuAddress:(uint64_t)gpuAddress
                   resourceId:(uint64_t)resourceId
                   descriptor:(const RgfxTextureDescriptor &)desc
                       device:(id<MTLDevice>)device
                   serializer:(RgfxSerializer *)serializer
{
	self = [super initWithObjId:objId device:device serializer:serializer];
	if (self) {
		_desc = desc;
		_gpuAddress = gpuAddress;
		_resourceId = resourceId;
		_allowGPUOptimizedContents = YES;
		_compressionType = MTLTextureCompressionTypeLossless;
		_number = g_lastTextureNumber.fetch_add(1) + 1;
	}
	return self;
}

- (void)rgfxTakeDescriptor:(MTLTextureDescriptor *)descriptor
{
	if (descriptor == nil) {
		return;
	}
	_allowGPUOptimizedContents = descriptor.allowGPUOptimizedContents;
	_compressionType = descriptor.compressionType;
	_framebufferOnly = [descriptor respondsToSelector:@selector(framebufferOnly)] &&
	                   [(id<RgfxMetalTextureDescriptor>)descriptor framebufferOnly];
}

- (void)dealloc
{
	// The host drops the texture as the app lets it go, after any command buffer that names it
	// (each keeps it while it runs), then its id goes back to the process's chunks, so a
	// texture made with the id later is created on ring 0 after this one's destroy.
	// A texture over a buffer keeps the buffer until then: the host's texture lies in it.
	RgfxSerializer *serializer = _serializer;
	uint64_t resourceId = _resourceId;
	uint64_t address = _gpuAddress; // a device-placed texture's, picked by its create
	// A view keeps its texture until then too, so the host drops the view first.
	id<MTLBuffer> buffer = _buffer;
	RgfxTexture *parent = _parent;
	[self rgfxRetireThen:[serializer, resourceId, address, buffer, parent]() mutable {
		buffer = nil;
		parent = nil;
		if (serializer == nullptr)
			return;
		serializer->releaseResourceId(resourceId);
		serializer->releaseAddress(address);
	}];
}

// --- MTLTexture shape (from the stored descriptor) ---------------------------

- (MTLTextureType)textureType
{
	return static_cast<MTLTextureType>(_desc.texture_type);
}

- (MTLPixelFormat)pixelFormat
{
	return static_cast<MTLPixelFormat>(_desc.pixel_format);
}

- (NSUInteger)width
{
	return _desc.width;
}

- (NSUInteger)height
{
	return _desc.height;
}

- (NSUInteger)depth
{
	return _desc.depth;
}

- (NSUInteger)mipmapLevelCount
{
	return _desc.mipmap_level_count;
}

- (NSUInteger)arrayLength
{
	return _desc.array_length;
}

- (NSUInteger)sampleCount
{
	return _desc.sample_count;
}

- (MTLTextureUsage)usage
{
	return static_cast<MTLTextureUsage>(rgfxAnsweredUsage(_desc.usage));
}

- (MTLTextureSwizzleChannels)swizzle
{
	return MTLTextureSwizzleChannelsMake(
	    static_cast<MTLTextureSwizzle>(_desc.swizzle & 0xff),
	    static_cast<MTLTextureSwizzle>((_desc.swizzle >> 8) & 0xff),
	    static_cast<MTLTextureSwizzle>((_desc.swizzle >> 16) & 0xff),
	    static_cast<MTLTextureSwizzle>((_desc.swizzle >> 24) & 0xff));
}

- (MTLResourceID)gpuResourceID
{
	MTLResourceID rid;
	rid._impl = _resourceId;
	return rid;
}

// --- MTLResource (decoded from the descriptor's resource options) -------------

// The texture's options as Metal answers them (measured on the M3 Max under macOS 27): those its
// descriptor asked, a Default hazard tracking mode resolved to Tracked (0x220 for Private, 0x210
// for Managed); a heap's texture answers the heap's (0x120 in a Private heap), one over a buffer
// the buffer's, whatever its descriptor asked, and a view its texture's.
- (MTLResourceOptions)resourceOptions
{
	if (_parent != nil) {
		return _parent.resourceOptions;
	}
	if (_buffer != nil) {
		return _buffer.resourceOptions;
	}
	id<MTLHeap> heap = [super heap];
	return heap != nil ? heap.resourceOptions
	                   : RgfxResolveHazardTracking(static_cast<MTLResourceOptions>(_desc.options),
	                                               MTLResourceHazardTrackingModeTracked);
}

// The options it was made with: its descriptor's as asked (0x20 for Private, 0x100 for Shared
// Untracked); its heap's for a heap's texture, as resourceOptions; its buffer's for one over a
// buffer, and its texture's for a view.
- (MTLResourceOptions)unfilteredResourceOptions
{
	if (_parent != nil) {
		return [_parent unfilteredResourceOptions];
	}
	if (_buffer != nil) {
		return [(id<RgfxMetalBufferInternals>)_buffer unfilteredResourceOptions];
	}
	id<MTLHeap> heap = [super heap];
	return heap != nil ? heap.resourceOptions : static_cast<MTLResourceOptions>(_desc.options);
}

- (MTLStorageMode)storageMode
{
	return static_cast<MTLStorageMode>((_desc.options & MTLResourceStorageModeMask) >>
	                                   MTLResourceStorageModeShift);
}

- (MTLCPUCacheMode)cpuCacheMode
{
	return static_cast<MTLCPUCacheMode>((_desc.options & MTLResourceCPUCacheModeMask) >>
	                                    MTLResourceCPUCacheModeShift);
}

- (MTLHazardTrackingMode)hazardTrackingMode
{
	return static_cast<MTLHazardTrackingMode>((self.resourceOptions &
	                                           MTLResourceHazardTrackingModeMask) >>
	                                          MTLResourceHazardTrackingModeShift);
}

// --- views ---------------------------------------------------------------------
// A view is a texture of its own id and resource id over its texture's texels
// (RGFX_CMD_TEXTURE_VIEW), which it keeps. It answers its own format, type, levels, slices
// and swizzle, and its texture's memory: the texture's buffer, IOSurface, heap, offset,
// aliasing and size, as Metal's views do (measured on the M3 Max under macOS 27.0.1). Its
// CPU copies are its texture's, at the view's levels and slices of it.
//
// Metal ends the process at a view it does not make (a failed assertion: levels or slices
// outside the texture, a type of another family, a format of another class), and answers nil
// for one in no format. The bundle answers nil, sending nothing, for each of these it can tell
// from the texture (rgfxView...), and the host refuses any other its platform does not make.
// So it answers a view of more than one slice of a 2D array over a buffer, which Metal asserts
// at on the M3 Max and an M4 (`sliceRange.length == 1`: read from metal2vulkan's REFUSALS.md,
// measured there) and the host's platform ends the host's process at.

- (nullable id<MTLTexture>)rgfxViewInFormat:(MTLPixelFormat)format
                                       type:(MTLTextureType)type
                                     levels:(NSRange)levels
                                     slices:(NSRange)slices
                                    swizzle:(MTLTextureSwizzleChannels)swizzle
                                      flags:(uint32_t)flags
{
	if (_serializer == nullptr) {
		return nil;
	}
	const bool subrange = (flags & RGFX_TEXTURE_VIEW_SUBRANGE) != 0;
	const bool swizzled = (flags & RGFX_TEXTURE_VIEW_SWIZZLE) != 0;
	const MTLTextureType parentType = self.textureType;
	const uint64_t parentLevels = self.mipmapLevelCount;
	const uint64_t parentSlices = rgfxSlicesOf(parentType, self.arrayLength);
	RgfxViewRefusal refusal = kViews;
	if (format == MTLPixelFormatInvalid) {
		refusal = kNoFormat;
	} else if (!rgfxViewFormatsMayMatch(self.pixelFormat, format)) {
		refusal = kFormatClass;
	} else if (subrange && (levels.length == 0 || levels.location >= parentLevels ||
	                        levels.length > parentLevels - levels.location)) {
		refusal = kLevels;
	} else if (subrange && (slices.length == 0 || slices.location >= parentSlices ||
	                        slices.length > parentSlices - slices.location)) {
		refusal = kSlices;
	} else if (subrange &&
	           (rgfxViewFamily(type) == 0 || rgfxViewFamily(type) != rgfxViewFamily(parentType))) {
		refusal = kTypeFamily;
	} else if (subrange && !rgfxViewTakesSlices(type, slices.length)) {
		refusal = kSliceCount;
	} else if ((subrange ? slices.length : parentSlices) > 1 && self.buffer != nil) {
		refusal = kBufferSlices;
	}
	if (refusal != kViews) {
		static std::atomic<bool> logged[kViews];
		static NSString *const what[kViews] = {
		    @"in no pixel format", @"in a pixel format of another class",
		    @"of levels outside it", @"of slices outside it", @"of a type of another family",
		    @"of a slice count its type does not take",
		    @"of more than one slice of an array over a buffer"};
		if (!logged[refusal].exchange(true)) {
			NSLog(@"RGFX: a view of texture 0x%llx (format %lu, type %lu) %@ (format %lu, type "
			      @"%lu): answered nil",
			      (unsigned long long)[self rgfxObjId], (unsigned long)self.pixelFormat,
			      (unsigned long)parentType, what[refusal], (unsigned long)format,
			      (unsigned long)type);
		}
		return nil;
	}

	// The view's shape: its texture's, from the view's first level, of the view's type, levels
	// and slices where it names them, and of its swizzle where it names one.
	const uint64_t base = subrange ? levels.location : 0;
	RgfxTextureDescriptor desc = {};
	desc.pixel_format = static_cast<uint32_t>(format);
	desc.texture_type = static_cast<uint32_t>(subrange ? type : parentType);
	desc.width = static_cast<uint32_t>(std::max<uint64_t>(1, self.width >> base));
	desc.height = static_cast<uint32_t>(std::max<uint64_t>(1, self.height >> base));
	desc.depth = static_cast<uint32_t>(std::max<uint64_t>(1, self.depth >> base));
	desc.mipmap_level_count = static_cast<uint32_t>(subrange ? levels.length : parentLevels);
	desc.array_length =
	    static_cast<uint32_t>(subrange ? rgfxArrayLengthOf(type, slices.length) : self.arrayLength);
	desc.sample_count = static_cast<uint32_t>(self.sampleCount);
	// The texture's usage as its descriptor gave it, 0 included, so the view's wire usage is
	// its texture's; a swizzled view keeps part of what the texture answers.
	uint64_t usage = _desc.usage;
	uint32_t packed = rgfxPackSwizzle(self.swizzle);
	if (swizzled) {
		packed = rgfxPackSwizzle(swizzle);
		if (packed != kRgfxDefaultSwizzle) {
			usage = rgfxAnsweredUsage(usage) & kRgfxSwizzledViewUsage;
		}
	}
	desc.usage = static_cast<uint32_t>(usage);
	desc.options = static_cast<uint32_t>(self.resourceOptions);
	desc.swizzle = packed;

	RgfxTextureView msg = {};
	msg.parent_obj_id = [self rgfxObjId];
	msg.pixel_format = static_cast<uint32_t>(format);
	msg.flags = flags;
	if (subrange) {
		msg.texture_type = static_cast<uint32_t>(type);
		msg.level_start = static_cast<uint32_t>(levels.location);
		msg.level_count = static_cast<uint32_t>(levels.length);
		msg.slice_start = static_cast<uint32_t>(slices.location);
		msg.slice_count = static_cast<uint32_t>(slices.length);
	}
	if (swizzled) {
		msg.swizzle = packed;
	}
	RgfxCreated created = _serializer->createTextureView(msg);
	if (created.resource_id == 0) {
		return nil; // no resource id left, as for any texture
	}
	if (_serializer->refused(created))
		return nil; // the device never got it
	RgfxTexture *view = [[RgfxTexture alloc] initWithObjId:created.obj_id
	                                            gpuAddress:0
	                                            resourceId:created.resource_id
	                                            descriptor:desc
	                                                device:self.device
	                                            serializer:_serializer];
	if (view != nil) {
		view->_parent = self;
		// A view of a texture over an IOSurface writes and reads the surface's texels.
		view->_rgfxSurfaceRoot = _rgfxSurfaceRoot;
		view->_parentLevel = static_cast<NSUInteger>(base);
		view->_parentSlice = subrange ? slices.location : 0;
		// The view's storage is its texture's, and so are its layout's choices.
		view->_allowGPUOptimizedContents = _allowGPUOptimizedContents;
		view->_compressionType = _compressionType;
		view->_framebufferOnly = _framebufferOnly;
	}
	return view;
}

- (id<MTLTexture>)newTextureViewWithPixelFormat:(MTLPixelFormat)pixelFormat
{
	return [self rgfxViewInFormat:pixelFormat
	                         type:self.textureType
	                       levels:NSMakeRange(0, 0)
	                       slices:NSMakeRange(0, 0)
	                      swizzle:MTLTextureSwizzleChannelsDefault
	                        flags:0];
}

- (id<MTLTexture>)newTextureViewWithPixelFormat:(MTLPixelFormat)pixelFormat
                                    textureType:(MTLTextureType)textureType
                                         levels:(NSRange)levelRange
                                         slices:(NSRange)sliceRange
{
	return [self rgfxViewInFormat:pixelFormat
	                         type:textureType
	                       levels:levelRange
	                       slices:sliceRange
	                      swizzle:MTLTextureSwizzleChannelsDefault
	                        flags:RGFX_TEXTURE_VIEW_SUBRANGE];
}

- (id<MTLTexture>)newTextureViewWithPixelFormat:(MTLPixelFormat)pixelFormat
                                    textureType:(MTLTextureType)textureType
                                         levels:(NSRange)levelRange
                                         slices:(NSRange)sliceRange
                                        swizzle:(MTLTextureSwizzleChannels)swizzle
{
	return [self rgfxViewInFormat:pixelFormat
	                         type:textureType
	                       levels:levelRange
	                       slices:sliceRange
	                      swizzle:swizzle
	                        flags:RGFX_TEXTURE_VIEW_SUBRANGE | RGFX_TEXTURE_VIEW_SWIZZLE];
}

// The descriptor names every field of the longest form, so it is made as that form. The wire
// carries no lowest level of detail, which macOS 27's descriptors add (minLOD): a view of
// another than the default 0 is answered nil, logged once, rather than made without it.
- (id<MTLTexture>)newTextureViewWithDescriptor:(MTLTextureViewDescriptor *)descriptor
{
	if (descriptor == nil) {
		return nil;
	}
	if ([descriptor respondsToSelector:@selector(minLOD)] && descriptor.minLOD != 0.0f) {
		static std::atomic<bool> logged{false};
		if (!logged.exchange(true)) {
			NSLog(@"RGFX: a view of texture 0x%llx with a lowest level of detail of %f: answered "
			      @"nil (the device carries none)",
			      (unsigned long long)[self rgfxObjId], descriptor.minLOD);
		}
		return nil;
	}
	return [self rgfxViewInFormat:descriptor.pixelFormat
	                         type:descriptor.textureType
	                       levels:descriptor.levelRange
	                       slices:descriptor.sliceRange
	                      swizzle:descriptor.swizzle
	                        flags:RGFX_TEXTURE_VIEW_SUBRANGE | RGFX_TEXTURE_VIEW_SWIZZLE];
}

// Every texture, a view among them, keeps the default lowest level of detail
// (newTextureViewWithDescriptor:).
- (float)minLOD
{
	return 0.0f;
}

- (NSUInteger)parentRelativeLevel
{
	return _parentLevel;
}

- (NSUInteger)parentRelativeSlice
{
	return _parentSlice;
}

// A view's memory is its texture's: the heap it lies in, at its offset, aliased with it. Making
// a view aliasable changes nothing, and its purgeable state stays NonVolatile and leaves its
// texture's as it was (measured on the M3 Max under macOS 27.0.1). A texture over a buffer lies
// in the buffer's memory alike: the buffer's heap at the buffer's offset, aliasable as the buffer
// is, and its purgeable state stays NonVolatile, whatever the buffer's (measured on the M3 Max
// under macOS 27).
- (id<MTLHeap>)heap
{
	if (_parent != nil) {
		return _parent.heap;
	}
	return _buffer != nil ? _buffer.heap : [super heap];
}

- (NSUInteger)heapOffset
{
	if (_parent != nil) {
		return _parent.heapOffset;
	}
	return _buffer != nil ? _buffer.heapOffset : [super heapOffset];
}

- (BOOL)isAliasable
{
	if (_parent != nil) {
		return _parent.isAliasable;
	}
	return _buffer != nil ? _buffer.isAliasable : [super isAliasable];
}

- (void)makeAliasable
{
	if (_parent == nil) {
		[super makeAliasable];
	}
}

- (MTLPurgeableState)setPurgeableState:(MTLPurgeableState)state
{
	if (_parent != nil || _buffer != nil) {
		return MTLPurgeableStateNonVolatile;
	}
	return [super setPurgeableState:state];
}

// The allocation a view lies in is its texture's, and one over a buffer the buffer's (measured on
// the M3 Max under macOS 27), and so are the bytes it meets another resource over.
- (uint64_t)rgfxAllocationId
{
	if (_parent != nil) {
		return _parent.rgfxAllocationId;
	}
	return _buffer != nil ? static_cast<RgfxResource *>(_buffer).rgfxAllocationId
	                      : [super rgfxAllocationId];
}

- (id<MTLHeap>)rgfxSpanOffset:(uint64_t *)offset size:(uint64_t *)size
{
	if (_parent != nil) {
		return [_parent rgfxSpanOffset:offset size:size];
	}
	return _buffer != nil ? [static_cast<RgfxResource *>(_buffer) rgfxSpanOffset:offset size:size]
	                      : [super rgfxSpanOffset:offset size:size];
}

// Whether `level` and `slice` are the view's, for a CPU copy through it.
- (BOOL)rgfxViewHoldsLevel:(NSUInteger)level slice:(NSUInteger)slice
{
	return level < _desc.mipmap_level_count &&
	       slice < rgfxSlicesOf(static_cast<MTLTextureType>(_desc.texture_type), _desc.array_length);
}

// --- CPU access ----------------------------------------------------------------
//
// Each form dispatches on the texture's storage: a Managed texture with a CPU copy reads and
// writes the copy (rgfxManaged...), with no round trip, but for the reads of one the host keeps
// in two copies on a GPU that shares the CPU's memory; any other is copied on the device
// (rgfxCopyRegion:, below), or refused where the CPU cannot reach it.

/// The bytes of the texels of `region` of `level` and `slice` in the CPU copy, copied to or from
/// `pixels` (each row `bytesPerRow`, each image `bytesPerImage` apart), as `toCopy` says. False
/// for a texture with no CPU copy; a region the layout does not hold copies nothing.
- (BOOL)rgfxManagedCopy:(MTLRegion)region
                  level:(NSUInteger)level
                  slice:(NSUInteger)slice
                 pixels:(uint8_t *)pixels
            bytesPerRow:(NSUInteger)bytesPerRow
          bytesPerImage:(NSUInteger)bytesPerImage
                 toCopy:(BOOL)toCopy
                 offset:(uint64_t *)offset
{
	if (_managedLayout == nullptr) {
		return NO;
	}
	const RgfxManagedLayout &layout = *_managedLayout;
	const uint64_t origin[3] = {region.origin.x, region.origin.y, region.origin.z};
	const uint64_t size[3] = {region.size.width, region.size.height, region.size.depth};
	*offset = UINT64_MAX;
	uint64_t at = 0;
	if (pixels == nullptr || level > UINT32_MAX ||
	    !layout.region(static_cast<uint32_t>(level), slice, origin, size, &at)) {
		return YES;
	}
	const RgfxManagedLevel &lv = layout.level(static_cast<uint32_t>(level));
	const uint64_t rowBytes = layout.rowBytes(size[0]);
	const uint64_t rows = layout.rows(size[1]);
	for (uint64_t z = 0; z < size[2]; z++) {
		for (uint64_t r = 0; r < rows; r++) {
			uint8_t *copy = _managedBytes + at + z * lv.bytesPerImage + r * lv.bytesPerRow;
			uint8_t *app = pixels + z * bytesPerImage + r * bytesPerRow;
			if (toCopy) {
				std::memcpy(copy, app, rowBytes);
			} else {
				std::memcpy(app, copy, rowBytes);
			}
		}
	}
	*offset = at;
	return YES;
}

/// replaceRegion: of a Managed texture: the CPU copy written, and the host told the region. False
/// for a texture with no CPU copy.
- (BOOL)rgfxManagedReplaceRegion:(MTLRegion)region
                           level:(NSUInteger)level
                           slice:(NSUInteger)slice
                           bytes:(const void *)bytes
                     bytesPerRow:(NSUInteger)bytesPerRow
                   bytesPerImage:(NSUInteger)bytesPerImage
{
	uint64_t at = 0;
	if (![self rgfxManagedCopy:region
	                     level:level
	                     slice:slice
	                    pixels:static_cast<uint8_t *>(const_cast<void *>(bytes))
	               bytesPerRow:bytesPerRow
	             bytesPerImage:bytesPerImage
	                    toCopy:YES
	                    offset:&at]) {
		return NO;
	}
	if (at != UINT64_MAX && _serializer != nullptr) {
		const uint32_t origin[3] = {static_cast<uint32_t>(region.origin.x),
		                            static_cast<uint32_t>(region.origin.y),
		                            static_cast<uint32_t>(region.origin.z)};
		const uint32_t size[3] = {static_cast<uint32_t>(region.size.width),
		                          static_cast<uint32_t>(region.size.height),
		                          static_cast<uint32_t>(region.size.depth)};
		_serializer->textureReplaceRegion([self rgfxObjId], static_cast<uint32_t>(level),
		                                  static_cast<uint32_t>(slice), origin, size);
	}
	return YES;
}

/// getBytes: of a Managed texture: the CPU copy read, as the GPU left it at the last
/// synchronize the process waited for. False for a texture with no CPU copy, and for one the host
/// keeps in two copies on a GPU that shares the CPU's memory (_managedReadsGpuCopy), whose read
/// goes to the GPU's copy on the device: there Metal's CPU reads of a Managed texture see the
/// GPU's writes with no synchronize (the Intel 630 and the M3, docs/graphics-kext/dgpu-oracle.md
/// Q2), and its replaceRegion: reaches that copy before any later command buffer.
- (BOOL)rgfxManagedGetBytes:(void *)bytes
                bytesPerRow:(NSUInteger)bytesPerRow
              bytesPerImage:(NSUInteger)bytesPerImage
                 fromRegion:(MTLRegion)region
                      level:(NSUInteger)level
                      slice:(NSUInteger)slice
{
	if (_managedReadsGpuCopy) {
		return NO;
	}
	uint64_t at = 0;
	return [self rgfxManagedCopy:region
	                       level:level
	                       slice:slice
	                      pixels:static_cast<uint8_t *>(bytes)
	                 bytesPerRow:bytesPerRow
	               bytesPerImage:bytesPerImage
	                      toCopy:NO
	                      offset:&at];
}

- (void)replaceRegion:(MTLRegion)region
          mipmapLevel:(NSUInteger)level
                slice:(NSUInteger)slice
            withBytes:(const void *)pixelBytes
          bytesPerRow:(NSUInteger)bytesPerRow
        bytesPerImage:(NSUInteger)bytesPerImage
{
	if (_parent != nil) {
		if (![self rgfxViewHoldsLevel:level slice:slice]) {
			static std::atomic<bool> logged{false};
			rgfxNoteCopyRefused(logged, @"replaceRegion: of a level or slice the view lacks", self);
			return;
		}
		[_parent replaceRegion:region
		           mipmapLevel:level + _parentLevel
		                 slice:slice + _parentSlice
		             withBytes:pixelBytes
		           bytesPerRow:bytesPerRow
		         bytesPerImage:bytesPerImage];
		return;
	}
	if ([self rgfxManagedReplaceRegion:region
	                             level:level
	                             slice:slice
	                             bytes:pixelBytes
	                       bytesPerRow:bytesPerRow
	                     bytesPerImage:bytesPerImage]) {
		return;
	}
	switch (self.storageMode) {
	case MTLStorageModePrivate:
	case MTLStorageModeMemoryless: {
		static std::atomic<bool> logged{false};
		rgfxNoteCopyRefused(logged, @"replaceRegion: of a texture the CPU cannot reach", self);
		return;
	}
	default:
		[self rgfxCopyRegion:region
		         mipmapLevel:level
		               slice:slice
		               bytes:const_cast<void *>(pixelBytes)
		         bytesPerRow:bytesPerRow
		       bytesPerImage:bytesPerImage
		             toBytes:NO];
		return;
	}
}

- (void)replaceRegion:(MTLRegion)region
          mipmapLevel:(NSUInteger)level
            withBytes:(const void *)pixelBytes
          bytesPerRow:(NSUInteger)bytesPerRow
{
	[self replaceRegion:region
	        mipmapLevel:level
	              slice:0
	          withBytes:pixelBytes
	        bytesPerRow:bytesPerRow
	      bytesPerImage:0];
}

- (void)getBytes:(void *)pixelBytes
      bytesPerRow:(NSUInteger)bytesPerRow
    bytesPerImage:(NSUInteger)bytesPerImage
       fromRegion:(MTLRegion)region
      mipmapLevel:(NSUInteger)level
            slice:(NSUInteger)slice
{
	if (_parent != nil) {
		if (![self rgfxViewHoldsLevel:level slice:slice]) {
			static std::atomic<bool> logged{false};
			rgfxNoteCopyRefused(logged, @"getBytes: of a level or slice the view lacks", self);
			return;
		}
		[_parent getBytes:pixelBytes
		      bytesPerRow:bytesPerRow
		    bytesPerImage:bytesPerImage
		       fromRegion:region
		      mipmapLevel:level + _parentLevel
		            slice:slice + _parentSlice];
		return;
	}
	if ([self rgfxManagedGetBytes:pixelBytes
	                  bytesPerRow:bytesPerRow
	                bytesPerImage:bytesPerImage
	                   fromRegion:region
	                        level:level
	                        slice:slice]) {
		return;
	}
	switch (self.storageMode) {
	case MTLStorageModePrivate:
	case MTLStorageModeMemoryless: {
		static std::atomic<bool> logged{false};
		rgfxNoteCopyRefused(logged, @"getBytes: of a texture the CPU cannot reach", self);
		return;
	}
	default:
		[self rgfxCopyRegion:region
		         mipmapLevel:level
		               slice:slice
		               bytes:pixelBytes
		         bytesPerRow:bytesPerRow
		       bytesPerImage:bytesPerImage
		             toBytes:YES];
		return;
	}
}

- (void)getBytes:(void *)pixelBytes
     bytesPerRow:(NSUInteger)bytesPerRow
      fromRegion:(MTLRegion)region
     mipmapLevel:(NSUInteger)level
{
	[self getBytes:pixelBytes
	      bytesPerRow:bytesPerRow
	    bytesPerImage:0
	       fromRegion:region
	      mipmapLevel:level
	            slice:0];
}

- (id<MTLTexture>)parentTexture
{
	return _parent;
}

- (IOSurfaceRef)iosurface
{
	return _parent != nil ? _parent.iosurface : nullptr;
}

- (NSUInteger)iosurfacePlane
{
	return _parent != nil ? _parent.iosurfacePlane : 0;
}

// A CAMetalLayer's drawable answers the layer's framebufferOnly, which the layer sets on the
// texture's descriptor; any other texture NO.
- (BOOL)isFramebufferOnly
{
	return _framebufferOnly;
}

// --- the CPU's copies of a texture with no CPU copy of its own ----------------------
// Private and Memoryless textures give the CPU no access. One made over a buffer is copied in
// the buffer's bytes, where its rows lie; any other (a Shared texture, or a Managed one with no
// CPU copy) is copied through a staging buffer of the region's rows, by a blit the call waits
// for, so the GPU's later work sees what was written and the bytes read are the GPU's.
//
// Over a Managed buffer, Metal's replaceRegion: reaches the GPU's next read of the texture with
// no didModifyRange: of the app's (measured on the AMD 5300M, whose Managed buffers are two
// copies, under macOS 26.7.1: `oracle_storage AMD lintex managed`), so a write there tells the
// host the rows' span, as the buffer's didModifyRange: would. The AMD uploads the whole buffer
// at a replaceRegion: of a corner, bytes the CPU wrote with no call among them; the span alone
// is uploaded here, as a didModifyRange: uploads its range alone (docs/graphics-kext/
// dgpu-oracle.md, Q2 and Q3).

// The bytes from one slice of a texture over a buffer to the next in the buffer: its rows'
// bytes, rounded up to 128 bytes. The host's linear form lays out a 2D array's slices so, whatever
// the bytesPerImage asked, as Metal's does (measured on the M3 Max and an M4 under macOS 27 by
// metal2vulkan, whose runtime lays them out the same: read from its linear_plan.rs). Every
// other texture over a buffer has one slice.
- (uint64_t)rgfxBufferSlicePitch
{
	const RgfxBlockShape b = rgfxBlockShapeOf(self.pixelFormat);
	const uint64_t rows = (std::max<uint64_t>(1, _desc.height) + b.height - 1) / b.height;
	return (_bufferBytesPerRow * rows + 127) / 128 * 128;
}

// Copies `region` of `level` and `slice` between the texture and the caller's `bytes`, rows
// `bytesPerRow` apart and images `bytesPerImage` apart: out of the texture into `bytes` when
// `toBytes`, else into the texture.
- (void)rgfxCopyRegion:(MTLRegion)region
           mipmapLevel:(NSUInteger)level
                 slice:(NSUInteger)slice
                 bytes:(void *)bytes
           bytesPerRow:(NSUInteger)bytesPerRow
         bytesPerImage:(NSUInteger)bytesPerImage
               toBytes:(BOOL)toBytes
{
	RgfxCopyShape shape = {};
	if (bytes == nullptr || !rgfxCopyShapeOf(self.pixelFormat, region, &shape)) {
		static std::atomic<bool> logged{false};
		rgfxNoteCopyRefused(logged, @"a CPU copy of an empty region or an unsized format",
		                    self);
		return;
	}
	uint64_t width = std::max<uint64_t>(1, _desc.width >> level);
	uint64_t height = std::max<uint64_t>(1, _desc.height >> level);
	uint64_t depth = std::max<uint64_t>(1, _desc.depth >> level);
	uint64_t imageBytes = shape.images > 1 ? bytesPerImage : bytesPerRow * shape.rows;
	if (level >= std::max<uint32_t>(1, _desc.mipmap_level_count) ||
	    region.origin.x + region.size.width > width ||
	    region.origin.y + region.size.height > height ||
	    region.origin.z + region.size.depth > depth || bytesPerRow < shape.rowBytes ||
	    (shape.images > 1 && imageBytes < bytesPerRow * shape.rows)) {
		static std::atomic<bool> logged{false};
		rgfxNoteCopyRefused(logged, @"a CPU copy past the texture or with rows too short", self);
		return;
	}
	auto copyRows = [&](uint8_t *dst, uint64_t dstRow, uint64_t dstImage, const uint8_t *src,
	                    uint64_t srcRow, uint64_t srcImage) {
		for (uint64_t z = 0; z < shape.images; z++) {
			for (uint64_t r = 0; r < shape.rows; r++) {
				std::memcpy(dst + z * dstImage + r * dstRow, src + z * srcImage + r * srcRow,
				            shape.rowBytes);
			}
		}
	};
	uint8_t *caller = static_cast<uint8_t *>(bytes);

	if (_buffer != nil) {
		// Its one level lies in the buffer, rows _bufferBytesPerRow apart, and each slice of a 2D
		// array (the linear form's) rgfxBufferSlicePitch after the one before.
		uint8_t *contents = static_cast<uint8_t *>([_buffer contents]);
		RgfxBlockShape b = rgfxBlockShapeOf(self.pixelFormat);
		const uint64_t slices =
		    rgfxSlicesOf(static_cast<MTLTextureType>(_desc.texture_type), _desc.array_length);
		if (contents == nullptr || level != 0 || slice >= slices || shape.images != 1) {
			static std::atomic<bool> logged{false};
			rgfxNoteCopyRefused(logged, @"a CPU copy of a texture over a buffer it cannot reach",
			                    self);
			return;
		}
		uint8_t *at = contents + _bufferOffset + slice * [self rgfxBufferSlicePitch] +
		              region.origin.y / b.height * _bufferBytesPerRow +
		              region.origin.x / b.width * b.bytes;
		if (toBytes) {
			copyRows(caller, bytesPerRow, imageBytes, at, _bufferBytesPerRow, 0);
		} else {
			copyRows(at, _bufferBytesPerRow, 0, caller, bytesPerRow, imageBytes);
			// The rows' span, from the first row's first byte written to the last row's last,
			// told as the buffer's didModifyRange: tells it: a Managed buffer's GPU copy takes
			// it (BUFFER_DID_MODIFY_RANGE, a no-copy one's block first taking the app's bytes),
			// and a Shared buffer's posts nothing.
			const uint64_t first = static_cast<uint64_t>(at - contents);
			const uint64_t span = (shape.rows - 1) * _bufferBytesPerRow + shape.rowBytes;
			[_buffer didModifyRange:NSMakeRange(first, span)];
		}
		return;
	}

	// Through a staging buffer of the region's rows, packed.
	id<MTLDevice> device = self.device;
	id<MTLCommandQueue> queue = rgfxCopyQueue(device);
	uint64_t stageImage = shape.rowBytes * shape.rows;
	id<MTLBuffer> stage = [device newBufferWithLength:stageImage * shape.images
	                                          options:MTLResourceStorageModeShared];
	uint8_t *staged = static_cast<uint8_t *>([stage contents]);
	id<MTLCommandBuffer> cb = [queue commandBuffer];
	id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
	if (staged == nullptr || blit == nil) {
		static std::atomic<bool> logged{false};
		rgfxNoteCopyRefused(logged, @"no staging buffer or command buffer for a CPU copy", self);
		[blit endEncoding];
		return;
	}
	if (toBytes) {
		[blit copyFromTexture:self
		                 sourceSlice:slice
		                 sourceLevel:level
		                sourceOrigin:region.origin
		                  sourceSize:region.size
		                    toBuffer:stage
		           destinationOffset:0
		      destinationBytesPerRow:shape.rowBytes
		    destinationBytesPerImage:stageImage];
	} else {
		copyRows(staged, shape.rowBytes, stageImage, caller, bytesPerRow, imageBytes);
		[blit copyFromBuffer:stage
		           sourceOffset:0
		      sourceBytesPerRow:shape.rowBytes
		    sourceBytesPerImage:stageImage
		             sourceSize:region.size
		              toTexture:self
		       destinationSlice:slice
		       destinationLevel:level
		      destinationOrigin:region.origin];
	}
	[blit endEncoding];
	[cb commit];
	[cb waitUntilCompleted];
	if (cb.status != MTLCommandBufferStatusCompleted) {
		static std::atomic<bool> logged{false};
		rgfxNoteCopyRefused(logged, @"the device failed a CPU copy's blit", self);
		return;
	}
	if (toBytes) {
		copyRows(caller, bytesPerRow, imageBytes, staged, shape.rowBytes, stageImage);
	}
}

// --- the descriptor's members the wire does not carry -------------------------------

- (BOOL)allowGPUOptimizedContents
{
	return _allowGPUOptimizedContents;
}

- (MTLTextureCompressionType)compressionType
{
	return _compressionType;
}

// --- what Metal's texture answers past its protocol (RgfxMetalPrivate.h) ----------------
// Each as Metal's (measured on the M3 Max under macOS 27), for a texture of every kind the
// device makes.

// The GPU address of the texture's memory, nonzero for every kind: its texture's for a view, at
// any level and slice, its buffer's plus its offset for one over a buffer, its heap's plus its
// offset there for one in a heap (RgfxIOSurfaceTexture answers its own). Else it is the address
// its create reserved, which no other resource takes, where Metal's is that of the texture's
// memory: the host does not place a device-placed texture's memory at it.
- (uint64_t)gpuAddress
{
	if (_parent != nil) {
		return [_parent gpuAddress];
	}
	if (_buffer != nil) {
		return _buffer.gpuAddress + _bufferOffset;
	}
	id<MTLHeap> heap = [super heap];
	if (heap != nil) {
		return [(id<RgfxMetalHeapInternals>)heap gpuAddress] + self.rgfxHeapPlacement;
	}
	return _gpuAddress;
}

// The size of the allocation the texture lies in: its texture's for a view, its buffer's for one
// over a buffer (the heap's, for a buffer in a heap), its heap's for one in a heap, else its own
// allocatedSize.
- (uint64_t)resourceSize
{
	if (_parent != nil) {
		return [_parent resourceSize];
	}
	if (_buffer != nil) {
		return [(id<RgfxMetalBufferInternals>)_buffer resourceSize];
	}
	id<MTLHeap> heap = [super heap];
	return heap != nil ? heap.size : self.allocatedSize;
}

RGFX_RESOURCE_SPI

// The CPU's address of the texture's memory: its texture's for a view, its buffer's plus its
// offset for one over a buffer (the offset itself, over a Private buffer), its heap's plus its
// offset there for a heap's texture (the offset itself, in a Private heap), else NULL, as Metal's
// answers for a Private or IOSurface texture. Metal's Shared and Managed textures in no heap have
// a CPU address, laid out as its GPU lays them out; the device lays out no such texture's memory
// where the CPU reaches it, so NULL.
- (void *)virtualAddress
{
	if (_parent != nil) {
		return [_parent virtualAddress];
	}
	uintptr_t base = 0, offset = 0;
	if (_buffer != nil) {
		base = reinterpret_cast<uintptr_t>([(id<RgfxMetalBufferInternals>)_buffer virtualAddress]);
		offset = static_cast<uintptr_t>(_bufferOffset);
	} else if (id<MTLHeap> heap = [super heap]) {
		if ([heap isKindOfClass:[RgfxHeap class]]) {
			base = reinterpret_cast<uintptr_t>(static_cast<RgfxHeap *>(heap).rgfxContents);
		}
		offset = static_cast<uintptr_t>(self.rgfxHeapPlacement);
	} else {
		return nullptr;
	}
	return reinterpret_cast<void *>(base + offset);
}

// The texture's number in the process: 1, 2, ... in the order textures are made, views among them,
// as Metal's.
- (uint64_t)uniqueIdentifier
{
	return _number;
}

// The device's textures are no drawable's, and are shown unrotated: Metal answers NO and 0 for
// every texture an app makes (private, shared, IOSurface and views).
- (BOOL)isDrawable
{
	return NO;
}

- (NSUInteger)rotation
{
	return 0;
}

// The CPU wrote the texture's memory. Over a Managed buffer, Metal's uploads the whole buffer to
// the GPU's copy (measured on the AMD 5300M under macOS 26.7.1, whose Managed buffers are two
// copies: `oracle_storage AMD didmodify length` and `nocopy`; the Intel 630's are one), as the
// buffer's didModifyRange: of all of it tells the host (BUFFER_DID_MODIFY_RANGE, a no-copy one's
// block first taking the app's bytes); a view's is its texture's. Any other texture has nothing to
// tell: one over a Shared buffer is the buffer's bytes, which the host reads in place, one over an
// IOSurface takes the surface's bytes at a commit once a CPU has written them under a lock
// (RgfxIOSurface.h), and no other texture's memory is the CPU's to write.
- (void)didModifyData
{
	if (_parent != nil) {
		[_parent didModifyData];
		return;
	}
	id<MTLBuffer> buffer = _buffer;
	if (buffer != nil)
		[buffer didModifyRange:NSMakeRange(0, buffer.length)];
}

// Whether the GPU keeps the texture's contents losslessly compressed, by the rule Metal's follows
// (measured on the M3 Max under macOS 27): NO for one over a buffer or an IOSurface, a Memoryless
// one, one that does not allow GPU-optimized contents, one of no usage or of any of ShaderWrite,
// PixelFormatView and ShaderAtomic, one under 16 texels wide or high (1D textures and texture
// buffers among them), and one of a block-compressed (BC, ETC, EAC, ASTC, PVRTC) or 4:2:2 format;
// YES for any other, of any storage mode, sample count, level count or type, depth, stencil and
// heap textures among them, and lossy ones. A view answers its texture's.
- (BOOL)isCompressed
{
	if (_parent != nil) {
		return [_parent isCompressed];
	}
	if (_buffer != nil || self.iosurface != nullptr ||
	    self.storageMode == MTLStorageModeMemoryless || !_allowGPUOptimizedContents) {
		return NO;
	}
	const MTLTextureUsage usage = self.usage;
	const MTLTextureUsage uncompressed =
	    MTLTextureUsageShaderWrite | MTLTextureUsagePixelFormatView | MTLTextureUsageShaderAtomic;
	if (usage == MTLTextureUsageUnknown || (usage & uncompressed) != 0) {
		return NO;
	}
	if (_desc.width < 16 || _desc.height < 16) {
		return NO;
	}
	return !rgfxIsBlockOrSubsampled(self.pixelFormat);
}

// --- what a texture that is not sparse, shared or remote answers ----------------------
// The device makes none of those (their members below answer nil), so these are exact.

// The deprecated owner of the texture's storage: its parent texture, else its buffer.
- (id<MTLResource>)rootResource
{
	id<MTLTexture> parent = self.parentTexture;
	return parent != nil ? (id<MTLResource>)parent : (id<MTLResource>)self.buffer;
}

- (BOOL)isShareable
{
	return NO;
}

- (MTLTextureSparseTier)sparseTextureTier
{
	return MTLTextureSparseTierNone;
}

- (BOOL)isSparse
{
	return NO;
}

- (NSUInteger)firstMipmapInTail
{
	return 0;
}

- (NSUInteger)tailSizeInBytes
{
	return 0;
}

- (id<MTLTexture>)remoteStorageTexture
{
	return nil;
}

// --- unsupported --------------------------------------------------------------------

// One device: no other device to view the texture from.
- (id<MTLTexture>)newRemoteTextureViewForDevice:(id<MTLDevice>)device
{
	RGFX_UNSUPPORTED("remote texture views");
	return nil;
}

- (MTLSharedTextureHandle *)newSharedTextureHandle
{
	RGFX_UNSUPPORTED("shared texture handles");
	return nil;
}

@end
