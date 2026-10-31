/*
 * RgfxBuffer: see RgfxBuffer.h. Our own ObjC++; no Apple private SPI.
 */

#import "RgfxBuffer.h"
#import "RgfxUnsupported.h"

#include <cstdlib>

#include "RgfxSerializer.hpp"

@implementation RgfxBuffer {
	uint64_t _gpuAddress;
	NSUInteger _length;
	MTLResourceOptions _options;
	// The buffer's CPU-visible backing `contents` vends. For a device buffer
	// (createBuffer) this is a plain page-aligned allocation the wrapper owns; for a
	// region-backed buffer (createBufferOverHostMemory) it is a pointer into the device
	// region (or the caller's no-copy pages), which the wrapper does NOT own.
	void *_contents;
	// YES for the device-buffer path: free(_contents) on dealloc. NO for a region-backed
	// buffer: the region (or the caller) owns the pages; `_hostDeallocator` is called
	// instead (no-copy's deallocator), when set.
	BOOL _ownsContents;
	void (^_hostDeallocator)(void);
	// YES where the buffer's create picked its GPU address (a device buffer, or one over host
	// memory), which goes back after its destroy; NO for one placed in a heap.
	BOOL _ownsAddress;
}

+ (instancetype)bufferWithLength:(NSUInteger)length
                         options:(MTLResourceOptions)options
                          device:(id<MTLDevice>)device
                      serializer:(RgfxSerializer *)serializer
{
	if (serializer == nullptr) {
		return nil; // a buffer must have a route to the host
	}
	RgfxCreated created = serializer->createBuffer(length, static_cast<uint32_t>(options));
	if (created.obj_id == 0) {
		return nil; // no GPU address left for it
	}
	RgfxBuffer *buffer = [[RgfxBuffer alloc] initWithObjId:created.obj_id
	                                            gpuAddress:created.gpu_address
	                                                length:length
	                                               options:options
	                                                device:device
	                                            serializer:serializer];
	// One whose contents could not be made keeps its address from use: its destroy may wait
	// behind command buffers in flight, and nothing here follows it.
	if (buffer != nil)
		buffer->_ownsAddress = YES;
	return buffer;
}

+ (instancetype)bufferOverHostMemoryWithContents:(void *)contents
                                    regionOffset:(uint64_t)regionOffset
                                          length:(NSUInteger)length
                                         options:(MTLResourceOptions)options
                                     deallocator:(void (^)(void))deallocator
                                          device:(id<MTLDevice>)device
                                      serializer:(RgfxSerializer *)serializer
{
	if (serializer == nullptr) {
		return nil; // a buffer must have a route to the host
	}
	RgfxCreated created = serializer->createBufferOverHostMemory(
	    length, static_cast<uint32_t>(options), regionOffset);
	if (created.obj_id == 0) {
		return nil; // no GPU address left for it
	}
	RgfxBuffer *buffer = [[RgfxBuffer alloc] initWithObjId:created.obj_id
	                                            gpuAddress:created.gpu_address
	                                                length:length
	                                               options:options
	                                              contents:contents
	                                           deallocator:deallocator
	                                                device:device
	                                            serializer:serializer];
	if (buffer != nil)
		buffer->_ownsAddress = YES;
	return buffer;
}

- (instancetype)initWithObjId:(uint64_t)objId
                   gpuAddress:(uint64_t)gpuAddress
                       length:(NSUInteger)length
                      options:(MTLResourceOptions)options
                       device:(id<MTLDevice>)device
                   serializer:(RgfxSerializer *)serializer
{
	self = [super initWithObjId:objId device:device serializer:serializer];
	if (self) {
		_gpuAddress = gpuAddress;
		_length = length;
		_options = options;
		_contents = nullptr;
		_ownsContents = YES;
		if (length > 0) {
			// Page-aligned so it matches a region mapping's alignment.
			if (posix_memalign(&_contents, 0x1000, length) != 0 || _contents == nullptr) {
				return nil;
			}
			std::memset(_contents, 0, length);
		}
	}
	return self;
}

- (instancetype)initWithObjId:(uint64_t)objId
                   gpuAddress:(uint64_t)gpuAddress
                       length:(NSUInteger)length
                      options:(MTLResourceOptions)options
                     contents:(void *)contents
                  deallocator:(void (^)(void))deallocator
                       device:(id<MTLDevice>)device
                   serializer:(RgfxSerializer *)serializer
{
	self = [super initWithObjId:objId device:device serializer:serializer];
	if (self) {
		_gpuAddress = gpuAddress;
		_length = length;
		_options = options;
		_contents = contents; // into the region, or the caller's no-copy pages
		_ownsContents = NO;
		_hostDeallocator = [deallocator copy];
	}
	return self;
}

- (void)dealloc
{
	// The device buffer's own pages are the guest's alone. The host drops the buffer once
	// every command buffer in flight now has completed, and only then is the memory under a
	// region-backed one given back (its block, or the no-copy caller's pages), so nothing
	// reuses it while the host may still read it.
	if (_ownsContents)
		std::free(_contents);
	void (^deallocator)(void) = _hostDeallocator;
	RgfxSerializer *serializer = _serializer;
	uint64_t address = _ownsAddress ? _gpuAddress : 0;
	[self rgfxRetireThen:[deallocator, serializer, address] {
		if (deallocator != nil)
			deallocator();
		if (serializer != nullptr && address != 0)
			serializer->releaseAddress(address);
	}];
}

// --- MTLBuffer ---------------------------------------------------------------

- (void *)contents
{
	return _contents;
}

- (NSUInteger)length
{
	return _length;
}

// The bytes the buffer takes: its length.
- (NSUInteger)allocatedSize
{
	return _length;
}

- (uint64_t)gpuAddress
{
	return _gpuAddress;
}

- (MTLResourceOptions)resourceOptions
{
	return _options;
}

// The modes `resourceOptions` holds: an app reads `storageMode` to tell a Managed buffer, whose
// writes it names with didModifyRange:.
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

- (void)didModifyRange:(NSRange)range
{
	// A Managed buffer's contents are its CPU copy, a region block: the host copies the range
	// to the GPU's copy before any command buffer committed after this, on any queue
	// (RGFX_CMD_BUFFER_DID_MODIFY_RANGE). A Shared or Private buffer has nothing to tell, as
	// Metal does nothing for one.
	if ([self storageMode] != MTLStorageModeManaged || _ownsContents || _serializer == nullptr ||
	    range.length == 0) {
		return;
	}
	_serializer->bufferDidModifyRange([self rgfxObjId], range.location, range.length);
}

// --- what a buffer that is not sparse or remote answers ------------------------------

- (MTLBufferSparseTier)sparseBufferTier
{
	return MTLBufferSparseTierNone; // the device makes no placement-sparse buffer
}

- (id<MTLBuffer>)remoteStorageBuffer
{
	return nil; // not a remote view (the device makes none)
}

// Debug markers name ranges for capture tools only: no capture tool sees this device's work,
// so they change nothing, as Metal's do with no capture under way.
- (void)addDebugMarker:(NSString *)marker range:(NSRange)range
{
}

- (void)removeAllDebugMarkers
{
}

// --- unsupported --------------------------------------------------------------------

// One device: no other device to view the buffer from.
- (id<MTLBuffer>)newRemoteBufferViewForDevice:(id<MTLDevice>)device
{
	RGFX_UNSUPPORTED("remote buffer views");
	return nil;
}

- (id<MTLTensor>)newTensorWithDescriptor:(MTLTensorDescriptor *)descriptor
                                  offset:(NSUInteger)offset
                                   error:(__autoreleasing NSError **)error
{
	RGFX_UNSUPPORTED_ERROR(error, "tensors");
	return nil;
}

@end
