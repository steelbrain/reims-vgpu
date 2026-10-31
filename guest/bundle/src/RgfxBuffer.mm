/*
 * RgfxBuffer: see RgfxBuffer.h. Our own ObjC++; no Apple private SPI.
 */

#import "RgfxBuffer.h"
#import "RgfxTexture.h"
#import "RgfxUnsupported.h"

#include <mach/mach.h>
#include <mach/mach_vm.h>

#include <cstdlib>
#include <cstring>

#include "RgfxMemoryShadow.hpp"
#include "RgfxRegion.hpp"
#include "RgfxSerializer.hpp"

namespace {
/// Whether this process maps every byte of [p, p + length) readable, and in `*writable` whether
/// it maps them all writable too.
bool rgfxMappedReadable(const void *p, size_t length, bool *writable)
{
	*writable = true;
	mach_vm_address_t at = reinterpret_cast<uintptr_t>(p);
	const mach_vm_address_t end = at + length;
	if (end < at)
		return false;
	while (at < end) {
		mach_vm_address_t base = at;
		mach_vm_size_t size = 0;
		vm_region_basic_info_data_64_t info = {};
		mach_msg_type_number_t count = VM_REGION_BASIC_INFO_COUNT_64;
		mach_port_t object = MACH_PORT_NULL;
		kern_return_t kr = mach_vm_region(mach_task_self(), &base, &size, VM_REGION_BASIC_INFO_64,
		                                  reinterpret_cast<vm_region_info_t>(&info), &count,
		                                  &object);
		if (object != MACH_PORT_NULL)
			mach_port_deallocate(mach_task_self(), object);
		// The region found is the one holding `at`, or the next one past a hole.
		if (kr != KERN_SUCCESS || base > at || (info.protection & VM_PROT_READ) == 0)
			return false;
		if ((info.protection & VM_PROT_WRITE) == 0)
			*writable = false;
		at = base + size;
	}
	return true;
}

/// Whether `options` names Shared or Managed storage, the modes Metal makes a no-copy buffer in.
bool rgfxNoCopyStorage(MTLResourceOptions options)
{
	const NSUInteger mode = (options & MTLResourceStorageModeMask) >> MTLResourceStorageModeShift;
	return mode == MTLStorageModeShared || mode == MTLStorageModeManaged;
}
} // namespace

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
	// A no-copy buffer over app memory outside the region: the block it lies over, the GPU's
	// copy of `_contents` (RgfxMemoryShadow.hpp).
	std::shared_ptr<RgfxMemoryShadow> _shadow;
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

+ (instancetype)bufferWithBytesNoCopy:(void *)pointer
                               length:(NSUInteger)length
                              options:(MTLResourceOptions)options
                          deallocator:(void (^)(void *, NSUInteger))deallocator
                               region:(RgfxRegion *)region
                               device:(id<MTLDevice>)device
                           serializer:(RgfxSerializer *)serializer
{
	if (length == 0 || !rgfxNoCopyStorage(options) || region == nullptr || serializer == nullptr)
		return nil;
	void (^appDeallocator)(void *, NSUInteger) = [deallocator copy];
	// Bytes already in a block of the process's: the host sees them in place.
	if (pointer != nullptr && region->contains(pointer, length)) {
		void (^done)(void) = nil;
		if (appDeallocator != nil)
			done = ^{ appDeallocator(pointer, length); };
		return [RgfxBuffer bufferOverHostMemoryWithContents:pointer
		                                      regionOffset:region->offsetOf(pointer)
		                                            length:length
		                                           options:options
		                                       deallocator:done
		                                            device:device
		                                        serializer:serializer];
	}
	bool writable = true;
	if (pointer != nullptr && !rgfxMappedReadable(pointer, length, &writable))
		return nil;
	RgfxRegion::Block block = region->allocate(length);
	if (!block.ok)
		return nil;
	uint8_t *bytes = const_cast<uint8_t *>(block.ptr);
	std::shared_ptr<RgfxMemoryShadow> shadow;
	void *contents = pointer;
	if (pointer != nullptr) {
		shadow = std::make_shared<RgfxMemoryShadow>(pointer, bytes, length, writable);
		if (!shadow->ok()) {
			region->release(block.offset);
			return nil;
		}
	} else {
		// No memory of the caller's: the block is the buffer's contents, as Metal's buffer
		// has memory of its own.
		std::memset(bytes, 0, length);
		contents = bytes;
	}
	// The block goes back once the host has dropped the buffer, then the caller's memory.
	const uint64_t offset = block.offset;
	RgfxBuffer *buffer = [RgfxBuffer bufferOverHostMemoryWithContents:contents
	                                                     regionOffset:offset
	                                                           length:length
	                                                          options:options
	                                                      deallocator:^{
		                                                      region->release(offset);
		                                                      if (appDeallocator != nil)
			                                                      appDeallocator(pointer, length);
	                                                      }
	                                                           device:device
	                                                       serializer:serializer];
	if (buffer == nil) {
		region->release(offset); // no buffer lies over the block
		return nil;
	}
	buffer->_shadow = std::move(shadow);
	return buffer;
}

- (std::shared_ptr<RgfxMemoryShadow>)rgfxShadow
{
	return _shadow;
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
	// A no-copy buffer's CPU copy is the app's memory: the block takes its bytes first.
	if (_shadow != nullptr)
		_shadow->syncIn();
	_serializer->bufferDidModifyRange([self rgfxObjId], range.location, range.length);
}

// --- what a buffer that is not sparse or remote answers ------------------------------

// A texture over the buffer's bytes from `offset`, rows `bytesPerRow` apart
// (TEXTURE_FROM_BUFFER); it keeps the buffer, which its `buffer` answers.
- (id<MTLTexture>)newTextureWithDescriptor:(MTLTextureDescriptor *)descriptor
                                    offset:(NSUInteger)offset
                               bytesPerRow:(NSUInteger)bytesPerRow
{
	return [RgfxTexture textureWithDescriptor:descriptor
	                                   buffer:self
	                                   offset:offset
	                              bytesPerRow:bytesPerRow
	                                   device:self.device
	                               serializer:[self serializer]];
}

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

std::shared_ptr<RgfxMemoryShadow> rgfxShadowOf(id resource)
{
	// A texture's bytes are its buffer's, through every view between (RgfxTexture's -buffer).
	if ([resource isKindOfClass:[RgfxTexture class]])
		resource = [(RgfxTexture *)resource buffer];
	if ([resource isKindOfClass:[RgfxBuffer class]])
		return [(RgfxBuffer *)resource rgfxShadow];
	return nullptr;
}
