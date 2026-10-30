/*
 * RgfxTexture: see RgfxTexture.h. Our own ObjC++; no Apple private SPI.
 */

#import "RgfxTexture.h"

#include "RgfxSerializer.hpp" // RgfxSerializer, RgfxTextureDescriptor (wire)

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

} // namespace

@implementation RgfxTexture {
	RgfxTextureDescriptor _desc; // the mapped shape the getters answer from
	uint64_t _gpuAddress;        // the picked GPU address (0 when heap-placed)
	uint64_t _resourceId;        // the picked texture resource id
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
	// The id goes back to the process's chunks, for its next texture.
	if (_serializer != nullptr) {
		_serializer->releaseResourceId(_resourceId);
	}
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

// --- views / backings (none in this slice) -----------------------------------
// A plain device- or heap-placed texture has no parent, backing buffer or
// IOSurface; those paths (texture views, buffer-backed textures, IOSurface
// import) are later resource/transport pieces.

- (id<MTLTexture>)parentTexture
{
	return nil;
}

- (id<MTLBuffer>)buffer
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
