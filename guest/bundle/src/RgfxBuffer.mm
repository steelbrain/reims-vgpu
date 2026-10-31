/*
 * RgfxBuffer: see RgfxBuffer.h. Our own ObjC++; no Apple private SPI.
 */

#import "RgfxBuffer.h"

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
	return [[RgfxBuffer alloc] initWithObjId:created.obj_id
	                              gpuAddress:created.gpu_address
	                                  length:length
	                                 options:options
	                                  device:device
	                              serializer:serializer];
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
	return [[RgfxBuffer alloc] initWithObjId:created.obj_id
	                              gpuAddress:created.gpu_address
	                                  length:length
	                                 options:options
	                                contents:contents
	                             deallocator:deallocator
	                                  device:device
	                              serializer:serializer];
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
	if (_ownsContents) {
		std::free(_contents);
	} else if (_hostDeallocator != nil) {
		_hostDeallocator();
	}
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

- (uint64_t)gpuAddress
{
	return _gpuAddress;
}

- (MTLResourceOptions)resourceOptions
{
	return _options;
}

- (void)didModifyRange:(NSRange)range
{
	// A Managed buffer's contents are its CPU copy, a region block: the host copies the range
	// to the GPU's copy before any command buffer committed after this, on any queue
	// (RGFX_CMD_BUFFER_DID_MODIFY_RANGE). A Shared or Private buffer has nothing to tell, as
	// Metal does nothing for one.
	const MTLStorageMode storage = static_cast<MTLStorageMode>(
	    (_options & MTLResourceStorageModeMask) >> MTLResourceStorageModeShift);
	if (storage != MTLStorageModeManaged || _ownsContents || _serializer == nullptr ||
	    range.length == 0) {
		return;
	}
	_serializer->bufferDidModifyRange([self rgfxObjId], range.location, range.length);
}

@end
