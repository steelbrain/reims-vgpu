/*
 * RgfxTexture: see RgfxTexture.h. Our own ObjC++; no Apple private SPI.
 */

#import "RgfxTexture.h"

#import "RgfxBuffer.h"

#include <cstring>
#include <memory>

#include "RgfxManagedLayout.hpp"
#include "RgfxRegion.hpp"
#include "RgfxRetirement.hpp"
#include "RgfxSerializer.hpp" // RgfxSerializer, RgfxTextureDescriptor (wire)

#include <algorithm>

namespace {

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

} // namespace

@implementation RgfxTexture {
	RgfxTextureDescriptor _desc; // the mapped shape the getters answer from
	uint64_t _gpuAddress;        // the picked GPU address (0 when heap-placed)
	uint64_t _resourceId;        // the picked texture resource id
	// A Managed texture's CPU copy: the Shared buffer over its block, the block's bytes, and
	// their layout. Nil and null for any other texture.
	RgfxBuffer *_managedCpu;
	uint8_t *_managedBytes;
	std::unique_ptr<RgfxManagedLayout> _managedLayout;
	// The buffer a texture made over one lies in (kept), its offset and bytes per row.
	id<MTLBuffer> _buffer;
	uint64_t _bufferOffset;
	uint64_t _bufferBytesPerRow;
}

+ (instancetype)managedTextureWithDescriptor:(MTLTextureDescriptor *)descriptor
                                      region:(RgfxRegion *)region
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
	RgfxRegion::Block copy = region->allocate(layout->size());
	if (!copy.ok) {
		return nil;
	}
	// A new Managed texture reads zero, as Metal's do.
	uint8_t *bytes = const_cast<uint8_t *>(copy.ptr);
	std::memset(bytes, 0, layout->size());
	// The block goes back once the host has dropped the copy's buffer, which goes after the
	// texture that holds it (its destroy first).
	const uint64_t offset = copy.offset;
	RgfxBuffer *cpu = [RgfxBuffer bufferOverHostMemoryWithContents:bytes
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
	RgfxCreated created = serializer->createManagedTexture(
	    desc, [cpu rgfxObjId], block.bytes, block.width, block.height,
	    block.linear ? RGFX_MANAGED_TEXTURE_LINEAR : 0);
	if (created.resource_id == 0) {
		return nil;
	}
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
	if (serializer == nullptr || descriptor == nil || buffer == nil) {
		return nil;
	}
	RgfxTextureDescriptor desc = rgfxTextureDescriptorFromMTL(descriptor);
	RgfxCreated created =
	    serializer->createTextureFromBuffer([buffer rgfxObjId], offset, bytesPerRow, desc);
	if (created.resource_id == 0) {
		return nil;
	}
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
	}
	return t;
}

- (id<MTLBuffer>)buffer
{
	return _buffer;
}

// The bytes the texture takes: its rows in its buffer, for one made over a buffer; else every
// level of every slice and sample, in its format's blocks (0 for a format not sized here).
- (NSUInteger)allocatedSize
{
	if (_buffer != nil) {
		return static_cast<NSUInteger>(_bufferBytesPerRow * _desc.height);
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
	return static_cast<NSUInteger>(_bufferOffset);
}

- (NSUInteger)bufferBytesPerRow
{
	return static_cast<NSUInteger>(_bufferBytesPerRow);
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
	return [[RgfxTexture alloc] initWithObjId:created.obj_id
	                              gpuAddress:created.gpu_address
	                              resourceId:created.resource_id
	                              descriptor:desc
	                                  device:device
	                              serializer:serializer];
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
	}
	return self;
}

- (void)dealloc
{
	// The host drops the texture once every command buffer in flight now has completed (one
	// may still name it), then its id goes back to the process's chunks, so a texture made
	// with the id later is created on ring 0 after this one's destroy.
	// A texture over a buffer keeps the buffer until then: the host's texture lies in it.
	RgfxSerializer *serializer = _serializer;
	uint64_t resourceId = _resourceId;
	uint64_t address = _gpuAddress; // a device-placed texture's, picked by its create
	id<MTLBuffer> buffer = _buffer;
	[self rgfxRetireThen:[serializer, resourceId, address, buffer]() mutable {
		buffer = nil;
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
	return static_cast<MTLTextureUsage>(_desc.usage);
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

- (MTLResourceOptions)resourceOptions
{
	return static_cast<MTLResourceOptions>(_desc.options);
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
	return static_cast<MTLHazardTrackingMode>((_desc.options & MTLResourceHazardTrackingModeMask) >>
	                                          MTLResourceHazardTrackingModeShift);
}

// --- views / backings ---------------------------------------------------------
// A plain device- or heap-placed texture has no parent or IOSurface; one made over a
// buffer answers its buffer (above). Texture views are a later piece.

// --- CPU access ----------------------------------------------------------------
//
// Each form dispatches on the texture's storage: a Managed texture with a CPU copy reads and
// writes the copy (rgfxManaged...), with no round trip.

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
/// synchronize the process waited for. False for a texture with no CPU copy.
- (BOOL)rgfxManagedGetBytes:(void *)bytes
                bytesPerRow:(NSUInteger)bytesPerRow
              bytesPerImage:(NSUInteger)bytesPerImage
                 fromRegion:(MTLRegion)region
                      level:(NSUInteger)level
                      slice:(NSUInteger)slice
{
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
	if ([self rgfxManagedReplaceRegion:region
	                             level:level
	                             slice:slice
	                             bytes:pixelBytes
	                       bytesPerRow:bytesPerRow
	                     bytesPerImage:bytesPerImage]) {
		return;
	}
	// Other storage modes: not handled here.
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
	if ([self rgfxManagedGetBytes:pixelBytes
	                  bytesPerRow:bytesPerRow
	                bytesPerImage:bytesPerImage
	                   fromRegion:region
	                        level:level
	                        slice:slice]) {
		return;
	}
	// Other storage modes: not handled here.
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
	return nil;
}

- (IOSurfaceRef)iosurface
{
	return nullptr;
}

- (BOOL)isFramebufferOnly
{
	return NO;
}

@end
