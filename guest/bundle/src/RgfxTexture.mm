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
	RgfxSerializer *serializer = _serializer;
	uint64_t resourceId = _resourceId;
	uint64_t address = _gpuAddress; // a device-placed texture's, picked by its create
	[self rgfxRetireThen:[serializer, resourceId, address] {
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
