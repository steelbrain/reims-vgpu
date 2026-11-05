/*
 * RgfxHeap: see RgfxHeap.h. Our own ObjC++; no Apple private SPI.
 */

#import "RgfxHeap.h"

#import "RgfxBuffer.h"
#import "RgfxInterop.h"
#import "RgfxMetalPrivate.h"
#import "RgfxTexture.h"
#import "RgfxUnsupported.h"

#include <algorithm>
#include <cstring>
#include <memory>
#include <mutex>

#include "RgfxHeapRanges.hpp"
#include "RgfxHeapSizes.hpp"
#include "RgfxRegion.hpp"
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

@interface RgfxHeap () <RgfxMetalHeapInternals>
// Returned retained, as the new… members that return them are, so no autorelease pool holds them.
- (nullable RgfxBuffer *)rgfxBufferAt:(uint64_t)offset
                               length:(NSUInteger)length
                              options:(MTLResourceOptions)options NS_RETURNS_RETAINED;
- (nullable RgfxBuffer *)rgfxPlaceBufferOfLength:(NSUInteger)length
                                         options:(MTLResourceOptions)options
                                       alignment:(uint64_t)alignment NS_RETURNS_RETAINED;
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
	uint64_t _traceId; // globalTraceObjectID
}

// Its trace id (RgfxInterop.h), taken as it was made.
- (uint64_t)globalTraceObjectID
{
	return _traceId;
}

// The host drops the heap as the app lets it go, after any command buffer that names it (each
// keeps it while it runs), and only then is a Shared heap's block given back. The resources placed in the heap keep it alive
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
	if (serializer->refused(created))
		return nil; // the device never got it
	RgfxHeap *heap = [[RgfxHeap alloc] initWithObjId:created.obj_id
	                                          device:device
	                                      serializer:serializer];
	if (heap)
		heap->_traceId = RgfxNextTraceObjectId();
	[heap rgfxTakeDescriptor:descriptor gpuAddress:created.gpu_address];
	return heap;
}

+ (instancetype)heapForDeviceWithDescriptor:(MTLHeapDescriptor *)descriptor
                                  maxLength:(NSUInteger)maxLength
                                     region:(RgfxRegion *)region
                                     device:(id<MTLDevice>)device
                                 serializer:(RgfxSerializer *)serializer
{
	// The x86 Macs' heap descriptor validation ends the process at a Shared heap (oracle_bundle
	// heapbuffers, AMD 5300M and Intel 630, macOS 26.7.1; the M3 Max makes one), sending nothing.
	if (descriptor != nil && descriptor.storageMode == MTLStorageModeShared)
		RgfxMetalAssertionFailed("-[MTLHeapDescriptorInternal validateWithDevice:]",
		                         @"Heap Descriptor Validation\nShared storage mode disallowed\n");
	if (descriptor == nil || descriptor.size == 0 ||
	    (maxLength != 0 && descriptor.size > maxLength))
		return nil;
	(void)region; // no heap the device makes lies in the region: a Shared one ends the process
	return [RgfxHeap heapWithDescriptor:descriptor device:device serializer:serializer];
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
	if (serializer->refused(created))
		return nil; // the device never got it
	RgfxHeap *heap = [[RgfxHeap alloc] initWithObjId:created.obj_id
	                                          device:device
	                                      serializer:serializer];
	if (heap) {
		heap->_traceId = RgfxNextTraceObjectId();
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

// The heap's options, its hazard tracking resolved as Metal's are: a Default mode is Untracked, as
// Metal documents and answers (measured on the M3 Max under macOS 27: 0x120 for a Private heap of
// either type, 0x100 for a Shared one, 0x200 for a Shared one made Tracked). Its unfiltered options
// are the same.
- (MTLResourceOptions)resourceOptions
{
	return RgfxResolveHazardTracking(_options, MTLResourceHazardTrackingModeUntracked);
}

- (MTLResourceOptions)unfilteredResourceOptions
{
	return self.resourceOptions;
}

// The heap's base: a resource placed at offset 0 lies there, as in Metal's (measured on the M3 Max
// under macOS 27, a Placement and an Automatic heap alike).
- (uint64_t)gpuAddress
{
	return _gpuAddress;
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
	return static_cast<MTLHazardTrackingMode>((self.resourceOptions &
	                                           MTLResourceHazardTrackingModeMask) >>
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

- (void *)rgfxContents
{
	return _contents;
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
	if (ser->refused(created))
		return nil; // the device never got it
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
	// As Metal's validation, sending nothing.
	rgfxEndAtInvalidTextureDescriptor(descriptor, self.device);
	rgfxEndAtHeapStorageMismatch(descriptor, self.storageMode);
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
	return [self rgfxPlaceBufferOfLength:length options:options alignment:1];
}

// A buffer of the descriptor's length, placed as newBufferWithLength:options: places one, at the
// descriptor's alignment where it is larger than the device's, with the heap's options whatever
// the descriptor's, as Metal's (measured on the M3 Max under macOS 27: an Automatic heap places a
// buffer of a 64 KiB alignment at the next 64 KiB and one of the default alignment of 1 at the
// device's 256 bytes, and answers a Private heap's options for one of Shared options). A Placement
// heap, which places nothing itself, answers nil, and so does a length of 0 or no descriptor. One
// over memory of its own (contents, an IOSurface, no-copy, address ranges, a pinned or parent
// address, sparse pages) the device does not make: nil, logged.
- (id<MTLBuffer>)newBufferWithDescriptor:(id)descriptor
{
	id<RgfxMetalBufferDescriptor> d = descriptor;
	if (d == nil || _ranges == nullptr || ![d respondsToSelector:@selector(length)] ||
	    d.length == 0) {
		return nil;
	}
	const bool ownMemory =
	    ([d respondsToSelector:@selector(contents)] && d.contents != nullptr) ||
	    ([d respondsToSelector:@selector(iosurface)] && d.iosurface != nullptr) ||
	    ([d respondsToSelector:@selector(noCopy)] && d.noCopy) ||
	    ([d respondsToSelector:@selector(addressRanges)] && d.addressRanges != nil) ||
	    ([d respondsToSelector:@selector(pinnedGPUAddress)] && d.pinnedGPUAddress != 0) ||
	    ([d respondsToSelector:@selector(parentGPUAddress)] && d.parentGPUAddress != 0) ||
	    ([d respondsToSelector:@selector(placementSparsePageSize)] &&
	     d.placementSparsePageSize != 0);
	if (ownMemory) {
		RGFX_UNSUPPORTED("buffers over memory of their own");
		return nil;
	}
	const uint64_t alignment = [d respondsToSelector:@selector(alignment)] ? d.alignment : 1;
	return [self rgfxPlaceBufferOfLength:d.length options:_options alignment:alignment];
}

// An Automatic heap's buffer of `length` bytes and `options`, at the device's heap buffer
// alignment or `alignment`, whichever is larger; nil where no free run holds it.
- (RgfxBuffer *)rgfxPlaceBufferOfLength:(NSUInteger)length
                                options:(MTLResourceOptions)options
                              alignment:(uint64_t)alignment
{
	// The device's heap buffer alignment (the host's, in its device info), else Metal's.
	id<MTLDevice> device = self.device;
	const uint64_t hostAlign =
	    device != nil ? [device heapBufferSizeAndAlignWithLength:length options:options].align : 0;
	const uint64_t align =
	    std::max<uint64_t>(hostAlign != 0 ? hostAlign : kRgfxHeapBufferAlign, alignment);
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
	// As Metal's validation, before the host is asked the texture's size.
	rgfxEndAtInvalidTextureDescriptor(descriptor, self.device);
	rgfxEndAtHeapStorageMismatch(descriptor, self.storageMode);
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

- (id<MTLAccelerationStructure>)newAccelerationStructureWithSize:(NSUInteger)size
                                                   resourceIndex:(NSUInteger)index
{
	RGFX_UNSUPPORTED("ray tracing");
	return nil;
}

@end
