/*
 * RgfxHeap: see RgfxHeap.h. Our own ObjC++; no Apple private SPI.
 */

#import "RgfxHeap.h"

#import "RgfxBuffer.h"
#import "RgfxTexture.h"
#import "RgfxUnsupported.h"

#include "RgfxSerializer.hpp" // RgfxSerializer

@implementation RgfxHeap {
	uint64_t _size;
	MTLResourceOptions _options;
	MTLHeapType _type;
	void (^_deallocator)(void); // gives a Shared heap's region block back
	uint64_t _gpuAddress;       // the GPU address its create picked, given back after its destroy
}

// The host drops the heap once every command buffer in flight now has completed, and only
// then is a Shared heap's block given back. The resources placed in the heap keep it alive
// (RgfxResource's `heap`), so they went first.
- (void)dealloc
{
	void (^deallocator)(void) = _deallocator;
	RgfxSerializer *serializer = _serializer;
	uint64_t address = _gpuAddress;
	[self rgfxRetireThen:[deallocator, serializer, address] {
		if (deallocator != nil)
			deallocator();
		if (serializer != nullptr)
			serializer->releaseAddress(address);
	}];
}

+ (instancetype)heapWithDescriptor:(MTLHeapDescriptor *)descriptor
                            device:(id<MTLDevice>)device
                        serializer:(RgfxSerializer *)serializer
{
	if (serializer == nullptr || descriptor == nil) {
		return nil; // a heap must have a route and a shape
	}
	if (descriptor.size == 0) {
		return nil; // as Metal answers a heap of no bytes
	}
	RgfxCreated created =
	    serializer->createHeap(static_cast<uint64_t>(descriptor.size),
	                           static_cast<uint32_t>(descriptor.resourceOptions),
	                           static_cast<int32_t>(descriptor.type));
	if (created.obj_id == 0) {
		return nil; // no GPU address left for it
	}
	RgfxHeap *heap = [[RgfxHeap alloc] initWithObjId:created.obj_id
	                                          device:device
	                                      serializer:serializer];
	if (heap) {
		heap->_size = static_cast<uint64_t>(descriptor.size);
		heap->_options = descriptor.resourceOptions;
		heap->_type = descriptor.type;
		heap->_gpuAddress = created.gpu_address;
	}
	return heap;
}

+ (instancetype)heapOverHostMemoryWithDescriptor:(MTLHeapDescriptor *)descriptor
                                    regionOffset:(uint64_t)regionOffset
                                     deallocator:(void (^)(void))deallocator
                                          device:(id<MTLDevice>)device
                                      serializer:(RgfxSerializer *)serializer
{
	if (serializer == nullptr || descriptor == nil) {
		return nil; // a heap must have a route and a shape
	}
	if (descriptor.size == 0) {
		return nil; // as Metal answers a heap of no bytes
	}
	RgfxCreated created = serializer->createHeapOverHostMemory(
	    static_cast<uint64_t>(descriptor.size),
	    static_cast<uint32_t>(descriptor.resourceOptions),
	    static_cast<int32_t>(descriptor.type), regionOffset);
	if (created.obj_id == 0) {
		return nil; // no GPU address left for it
	}
	RgfxHeap *heap = [[RgfxHeap alloc] initWithObjId:created.obj_id
	                                          device:device
	                                      serializer:serializer];
	if (heap) {
		heap->_size = static_cast<uint64_t>(descriptor.size);
		heap->_options = descriptor.resourceOptions;
		heap->_type = descriptor.type;
		heap->_gpuAddress = created.gpu_address;
		heap->_deallocator = [deallocator copy];
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
// Only a Placement heap places at an offset: of a heap of another type, Metal's validation
// fails the call, and these answer nil, as the host refuses such a placement.

- (id<MTLBuffer>)newBufferWithLength:(NSUInteger)length
                             options:(MTLResourceOptions)options
                              offset:(NSUInteger)offset
{
	RgfxSerializer *ser = [self serializer];
	if (ser == nullptr || _type != MTLHeapTypePlacement) {
		return nil;
	}
	RgfxCreated created = ser->createBufferAtOffset([self rgfxObjId], offset, length,
	                                                static_cast<uint32_t>(options));
	// A heap-placed buffer's address is derived host-side from the heap (0 here). It keeps
	// the heap alive.
	RgfxBuffer *buffer = [[RgfxBuffer alloc] initWithObjId:created.obj_id
	                                            gpuAddress:created.gpu_address
	                                                length:length
	                                               options:options
	                                                device:self.device
	                                            serializer:ser];
	[buffer rgfxSetHeap:self];
	[buffer rgfxSetHeapOffset:offset];
	return buffer;
}

- (id<MTLTexture>)newTextureWithDescriptor:(MTLTextureDescriptor *)descriptor
                                    offset:(NSUInteger)offset
{
	if (_type != MTLHeapTypePlacement) {
		return nil;
	}
	RgfxTexture *texture = [RgfxTexture textureWithDescriptor:descriptor
	                                                     heap:[self rgfxObjId]
	                                               heapOffset:offset
	                                                   device:self.device
	                                               serializer:[self serializer]];
	[texture rgfxSetHeap:self]; // the texture keeps the heap alive
	[texture rgfxSetHeapOffset:offset];
	return texture;
}

// --- sub-allocation (Automatic: Metal picks the offset) ----------------------
// These need the host's per-resource size and alignment to place the resource;
// that query arrives with the region/block transport. Documented nil stubs until
// then (the shell returns nil, as elsewhere, rather than invent a placement).

- (id<MTLBuffer>)newBufferWithLength:(NSUInteger)length options:(MTLResourceOptions)options
{
	RGFX_UNSUPPORTED("automatic heap placement (no host size and alignment on the wire)");
	return nil;
}

- (id<MTLTexture>)newTextureWithDescriptor:(MTLTextureDescriptor *)descriptor
{
	RGFX_UNSUPPORTED("automatic heap placement (no host size and alignment on the wire)");
	return nil;
}

// --- MTLAllocation -------------------------------------------------------------------

// The bytes the heap takes: its size, all of it reserved at its create.
- (NSUInteger)allocatedSize
{
	return static_cast<NSUInteger>(_size);
}

// --- unsupported: the device makes no acceleration structure ------------------------

- (id<MTLAccelerationStructure>)newAccelerationStructureWithSize:(NSUInteger)size
{
	RGFX_UNSUPPORTED("ray tracing");
	return nil;
}

- (id<MTLAccelerationStructure>)newAccelerationStructureWithDescriptor:
    (MTLAccelerationStructureDescriptor *)descriptor
{
	RGFX_UNSUPPORTED("ray tracing");
	return nil;
}

- (id<MTLAccelerationStructure>)newAccelerationStructureWithSize:(NSUInteger)size
                                                          offset:(NSUInteger)offset
{
	RGFX_UNSUPPORTED("ray tracing");
	return nil;
}

- (id<MTLAccelerationStructure>)newAccelerationStructureWithDescriptor:
                                    (MTLAccelerationStructureDescriptor *)descriptor
                                                                offset:(NSUInteger)offset
{
	RGFX_UNSUPPORTED("ray tracing");
	return nil;
}

@end
