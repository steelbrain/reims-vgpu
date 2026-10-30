/*
 * RgfxHeap: see RgfxHeap.h. Our own ObjC++; no Apple private SPI.
 */

#import "RgfxHeap.h"

#import "RgfxBuffer.h"
#import "RgfxTexture.h"

#include "RgfxSerializer.hpp" // RgfxSerializer

@implementation RgfxHeap {
	uint64_t _size;
	MTLResourceOptions _options;
	MTLHeapType _type;
}

+ (instancetype)heapWithDescriptor:(MTLHeapDescriptor *)descriptor
                            device:(id<MTLDevice>)device
                        serializer:(RgfxSerializer *)serializer
{
	if (serializer == nullptr || descriptor == nil) {
		return nil; // a heap must have a route and a shape
	}
	RgfxCreated created =
	    serializer->createHeap(static_cast<uint64_t>(descriptor.size),
	                           static_cast<uint32_t>(descriptor.resourceOptions),
	                           static_cast<int32_t>(descriptor.type));
	RgfxHeap *heap = [[RgfxHeap alloc] initWithObjId:created.obj_id
	                                          device:device
	                                      serializer:serializer];
	if (heap) {
		heap->_size = static_cast<uint64_t>(descriptor.size);
		heap->_options = descriptor.resourceOptions;
		heap->_type = descriptor.type;
	}
	return heap;
}

+ (instancetype)heapOverHostMemoryWithDescriptor:(MTLHeapDescriptor *)descriptor
                                    regionOffset:(uint64_t)regionOffset
                                          device:(id<MTLDevice>)device
                                      serializer:(RgfxSerializer *)serializer
{
	if (serializer == nullptr || descriptor == nil) {
		return nil; // a heap must have a route and a shape
	}
	RgfxCreated created = serializer->createHeapOverHostMemory(
	    static_cast<uint64_t>(descriptor.size),
	    static_cast<uint32_t>(descriptor.resourceOptions),
	    static_cast<int32_t>(descriptor.type), regionOffset);
	RgfxHeap *heap = [[RgfxHeap alloc] initWithObjId:created.obj_id
	                                          device:device
	                                      serializer:serializer];
	if (heap) {
		heap->_size = static_cast<uint64_t>(descriptor.size);
		heap->_options = descriptor.resourceOptions;
		heap->_type = descriptor.type;
	}
	return heap;
}

// --- MTLHeap shape (from the stored descriptor) ------------------------------

- (NSUInteger)size
{
	return static_cast<NSUInteger>(_size);
}

- (MTLHeapType)type
{
	return _type;
}

- (MTLResourceOptions)resourceOptions
{
	return _options;
}

- (MTLStorageMode)storageMode
{
	return static_cast<MTLStorageMode>((_options & MTLResourceStorageModeMask) >>
	                                   MTLResourceStorageModeShift);
}

- (MTLCPUCacheMode)cpuCacheMode
{
	return static_cast<MTLCPUCacheMode>((_options & MTLResourceCPUCacheModeMask) >>
	                                    MTLResourceCPUCacheModeShift);
}

- (MTLHazardTrackingMode)hazardTrackingMode
{
	return static_cast<MTLHazardTrackingMode>((_options & MTLResourceHazardTrackingModeMask) >>
	                                          MTLResourceHazardTrackingModeShift);
}

- (NSUInteger)usedSize
{
	// This slice does not track placement occupancy (the region/block allocator,
	// a later transport piece, will). Report nothing used.
	return 0;
}

- (NSUInteger)currentAllocatedSize
{
	return static_cast<NSUInteger>(_size);
}

- (NSUInteger)maxAvailableSizeWithAlignment:(NSUInteger)alignment
{
	(void)alignment;
	return static_cast<NSUInteger>(_size);
}

- (MTLPurgeableState)setPurgeableState:(MTLPurgeableState)state
{
	// No host purgeability wire yet; the heap is always resident.
	(void)state;
	return MTLPurgeableStateNonVolatile;
}

// --- sub-allocation (Placement: the guest names the offset) -------------------

- (id<MTLBuffer>)newBufferWithLength:(NSUInteger)length
                             options:(MTLResourceOptions)options
                              offset:(NSUInteger)offset
{
	RgfxSerializer *ser = [self serializer];
	if (ser == nullptr) {
		return nil;
	}
	RgfxCreated created = ser->createBufferAtOffset([self rgfxObjId], offset, length,
	                                                static_cast<uint32_t>(options));
	// A heap-placed buffer's address is derived host-side from the heap (0 here).
	return [[RgfxBuffer alloc] initWithObjId:created.obj_id
	                              gpuAddress:created.gpu_address
	                                  length:length
	                                 options:options
	                                  device:self.device
	                              serializer:ser];
}

- (id<MTLTexture>)newTextureWithDescriptor:(MTLTextureDescriptor *)descriptor
                                    offset:(NSUInteger)offset
{
	return [RgfxTexture textureWithDescriptor:descriptor
	                                     heap:[self rgfxObjId]
	                               heapOffset:offset
	                                   device:self.device
	                               serializer:[self serializer]];
}

// --- sub-allocation (Automatic: Metal picks the offset) ----------------------
// These need the host's per-resource size and alignment to place the resource;
// that query arrives with the region/block transport. Documented nil stubs until
// then (the shell returns nil, as elsewhere, rather than invent a placement).

- (id<MTLBuffer>)newBufferWithLength:(NSUInteger)length options:(MTLResourceOptions)options
{
	(void)length;
	(void)options;
	return nil;
}

- (id<MTLTexture>)newTextureWithDescriptor:(MTLTextureDescriptor *)descriptor
{
	(void)descriptor;
	return nil;
}

@end
