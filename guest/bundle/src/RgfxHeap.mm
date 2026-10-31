/*
 * RgfxHeap: see RgfxHeap.h. Our own ObjC++; no Apple private SPI.
 */

#import "RgfxHeap.h"

#import "RgfxBuffer.h"
#import "RgfxTexture.h"
#import "RgfxUnsupported.h"

#include <memory>
#include <mutex>

#include "RgfxHeapRanges.hpp"
#include "RgfxHeapSizes.hpp"
#include "RgfxSerializer.hpp" // RgfxSerializer

/// Metal's page, which a heap's size is rounded up to.
static constexpr uint64_t kHeapPage = 0x4000;

MTLSizeAndAlign rgfxHeapTextureSizeAndAlign(RgfxSerializer *serializer,
                                            MTLTextureDescriptor *descriptor, Class cls, SEL cmd)
{
	if (descriptor == nil) {
		return {0, 0};
	}
	uint64_t size = 0, align = 0;
	if (serializer != nullptr &&
	    serializer->heapTextureSizeAndAlign(rgfxTextureDescriptorFromMTL(descriptor), &size,
	                                        &align)) {
		return {static_cast<NSUInteger>(size), static_cast<NSUInteger>(align)};
	}
	RgfxLogUnsupported(cls, cmd, "the host's heap texture size (no answer came back): a bound "
	                             "of Metal's is answered");
	RgfxHeapSizeAlign sa = {};
	rgfxHeapTextureBound(static_cast<uint32_t>(descriptor.pixelFormat),
	                     static_cast<uint32_t>(descriptor.textureType), descriptor.width,
	                     descriptor.height, descriptor.depth,
	                     static_cast<uint32_t>(descriptor.mipmapLevelCount),
	                     descriptor.arrayLength, descriptor.sampleCount, &sa);
	return {static_cast<NSUInteger>(sa.size), static_cast<NSUInteger>(sa.align)};
}

@interface RgfxHeap ()
// Returned retained, as the new… members that return it are, so no autorelease pool holds it.
- (nullable RgfxBuffer *)rgfxBufferAt:(uint64_t)offset
                               length:(NSUInteger)length
                              options:(MTLResourceOptions)options NS_RETURNS_RETAINED;
@end

@implementation RgfxHeap {
	uint64_t _size; // the descriptor's, rounded up to kHeapPage
	MTLResourceOptions _options;
	MTLHeapType _type;
	void (^_deallocator)(void); // gives a Shared heap's region block back
	uint64_t _gpuAddress;       // the GPU address its create picked, given back after its destroy
	uint8_t *_contents;         // a Shared heap's bytes, its region block's; null for any other
	// An Automatic heap's placements, under _rangesLock; null for a heap of another type.
	std::unique_ptr<RgfxHeapRanges> _ranges;
	std::mutex _rangesLock;
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
	[heap rgfxTakeDescriptor:descriptor gpuAddress:created.gpu_address];
	return heap;
}

+ (instancetype)heapOverHostMemoryWithDescriptor:(MTLHeapDescriptor *)descriptor
                                    regionOffset:(uint64_t)regionOffset
                                        contents:(void *)contents
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
		[heap rgfxTakeDescriptor:descriptor gpuAddress:created.gpu_address];
		heap->_contents = static_cast<uint8_t *>(contents);
		heap->_deallocator = [deallocator copy];
	}
	return heap;
}

// The shape a factory's create made the heap with. The size, which the create checked takes an
// address, is rounded up to Metal's page, as the host's platform rounds it.
- (void)rgfxTakeDescriptor:(MTLHeapDescriptor *)descriptor gpuAddress:(uint64_t)gpuAddress
{
	const uint64_t size = static_cast<uint64_t>(descriptor.size);
	_size = size + (kHeapPage - size % kHeapPage) % kHeapPage;
	_options = descriptor.resourceOptions;
	_type = descriptor.type;
	_gpuAddress = gpuAddress;
	if (_type == MTLHeapTypeAutomatic) {
		_ranges = std::make_unique<RgfxHeapRanges>(_size);
	}
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

// An Automatic heap's placements answer what is used and what is free (RgfxHeapRanges.hpp); a
// Placement heap, whose resources the app places, answers 0 for both, as Metal's does (measured:
// M3 Max, macOS 27, with a buffer placed in it). Every heap holds its whole size.
- (NSUInteger)usedSize
{
	if (_ranges == nullptr) {
		return 0;
	}
	std::lock_guard<std::mutex> hold(_rangesLock);
	return static_cast<NSUInteger>(_ranges->used());
}

- (NSUInteger)currentAllocatedSize
{
	return static_cast<NSUInteger>(_size);
}

- (NSUInteger)maxAvailableSizeWithAlignment:(NSUInteger)alignment
{
	if (_ranges == nullptr) {
		return 0;
	}
	std::lock_guard<std::mutex> hold(_rangesLock);
	return static_cast<NSUInteger>(_ranges->maxAvailable(alignment));
}

- (void)rgfxGiveRangeAt:(uint64_t)offset size:(uint64_t)size
{
	if (_ranges == nullptr) {
		return;
	}
	std::lock_guard<std::mutex> hold(_rangesLock);
	_ranges->give(offset, size);
}

// `size` bytes of an Automatic heap at `align`: the offset picked, or false where no free run
// holds them.
- (BOOL)rgfxTakeRange:(uint64_t)size align:(uint64_t)align offset:(uint64_t *)offset
{
	if (_ranges == nullptr) {
		return NO;
	}
	std::lock_guard<std::mutex> hold(_rangesLock);
	return _ranges->take(size, align, offset);
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
	if (_type != MTLHeapTypePlacement) {
		return nil;
	}
	RgfxBuffer *buffer = [self rgfxBufferAt:offset length:length options:options];
	[buffer rgfxSetHeapOffset:offset];
	return buffer;
}

// A buffer of `length` bytes at `offset` in the heap, kept by it, routed as an
// RGFX_CMD_BUFFER_AT_OFFSET. Nil with no route, or past a Shared heap's bytes.
- (RgfxBuffer *)rgfxBufferAt:(uint64_t)offset
                      length:(NSUInteger)length
                     options:(MTLResourceOptions)options
{
	RgfxSerializer *ser = [self serializer];
	if (ser == nullptr) {
		return nil;
	}
	// A Shared heap's buffer is its bytes at the offset, which must lie in the heap.
	if (_contents != nullptr && (offset > _size || length > _size - offset)) {
		return nil;
	}
	RgfxCreated created = ser->createBufferAtOffset([self rgfxObjId], offset, length,
	                                                static_cast<uint32_t>(options));
	// At the heap's address plus the offset, where the host places it in the heap it pinned.
	// A Shared heap's buffer is the heap's bytes there, which the host's is too; it keeps the
	// heap, and so the bytes, alive.
	const uint64_t address = _gpuAddress + offset;
	RgfxBuffer *buffer =
	    _contents != nullptr
	        ? [[RgfxBuffer alloc] initWithObjId:created.obj_id
	                                 gpuAddress:address
	                                     length:length
	                                    options:options
	                                   contents:_contents + offset
	                                deallocator:nil
	                                     device:self.device
	                                 serializer:ser]
	        : [[RgfxBuffer alloc] initWithObjId:created.obj_id
	                                 gpuAddress:address
	                                     length:length
	                                    options:options
	                                     device:self.device
	                                 serializer:ser];
	[buffer rgfxSetHeap:self];
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

// --- sub-allocation (Automatic: the heap picks the offset) --------------------
// The heap places each at the lowest free offset at its alignment (RgfxHeapRanges.hpp), as
// Metal's does, and nil where no free run holds it, or for a heap of another type. Its heapOffset
// answers 0, as Metal's does for a resource of an Automatic heap (measured: M3 Max, macOS 27).

- (id<MTLBuffer>)newBufferWithLength:(NSUInteger)length options:(MTLResourceOptions)options
{
	// The device's heap buffer alignment (the host's, in its device info), else Metal's.
	id<MTLDevice> device = self.device;
	const uint64_t hostAlign =
	    device != nil ? [device heapBufferSizeAndAlignWithLength:length options:options].align : 0;
	const uint64_t align = hostAlign != 0 ? hostAlign : kRgfxHeapBufferAlign;
	uint64_t offset = 0;
	if ([self serializer] == nullptr || ![self rgfxTakeRange:length align:align offset:&offset]) {
		return nil;
	}
	RgfxBuffer *buffer = [self rgfxBufferAt:offset length:length options:options];
	if (buffer == nil) {
		[self rgfxGiveRangeAt:offset size:length];
		return nil;
	}
	[buffer rgfxHoldHeapRange:offset size:length];
	return buffer;
}

- (id<MTLTexture>)newTextureWithDescriptor:(MTLTextureDescriptor *)descriptor
{
	RgfxSerializer *ser = [self serializer];
	if (ser == nullptr || descriptor == nil || _ranges == nullptr) {
		return nil;
	}
	const MTLSizeAndAlign sa = rgfxHeapTextureSizeAndAlign(ser, descriptor, [self class], _cmd);
	uint64_t offset = 0;
	if (![self rgfxTakeRange:sa.size align:sa.align offset:&offset]) {
		return nil;
	}
	RgfxTexture *texture = [RgfxTexture textureWithDescriptor:descriptor
	                                                     heap:[self rgfxObjId]
	                                               heapOffset:offset
	                                                   device:self.device
	                                               serializer:ser];
	if (texture == nil) {
		[self rgfxGiveRangeAt:offset size:sa.size];
		return nil;
	}
	[texture rgfxSetHeap:self]; // the texture keeps the heap alive
	[texture rgfxHoldHeapRange:offset size:sa.size];
	return texture;
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
