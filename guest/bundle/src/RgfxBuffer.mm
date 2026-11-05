/*
 * RgfxBuffer: see RgfxBuffer.h. Our own ObjC++; no Apple private SPI.
 */

#import "RgfxBuffer.h"
#import "RgfxMetalError.h"
#import "RgfxMetalPrivate.h"
#import "RgfxTexture.h"
#import "RgfxUnsupported.h"

#include <mach/mach.h>
#include <mach/mach_vm.h>
#include <os/log.h>

#include <atomic>
#include <cstdlib>
#include <cstring>

#include "RgfxManagedLayout.hpp"
#include "RgfxMemoryShadow.hpp"
#include "RgfxRegion.hpp"
#include "RgfxSerializer.hpp"
#include "RgfxSlabs.hpp"

namespace {
/// The page the kext rounds a pages block's range out to (its RgfxPagesBlock.hpp), the guest's:
/// the device's view of a block starts at the page of the buffer's first byte.
constexpr uint64_t kRgfxGuestPage = 0x1000;

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

/// Whether `options` names Shared or Managed storage: the modes whose buffer has contents the CPU
/// reaches, and the ones Metal makes a no-copy buffer in.
bool rgfxCpuStorage(MTLResourceOptions options)
{
	const NSUInteger mode = (options & MTLResourceStorageModeMask) >> MTLResourceStorageModeShift;
	return mode == MTLStorageModeShared || mode == MTLStorageModeManaged;
}
} // namespace

@interface RgfxBuffer () <RgfxMetalBufferInternals>
@end

@implementation RgfxBuffer {
	uint64_t _gpuAddress;
	NSUInteger _length;
	MTLResourceOptions _options;
	// The buffer's CPU-visible backing `contents` vends. For a Shared or Managed device buffer
	// (createBuffer) this is a plain page-aligned allocation the wrapper owns, and a Private
	// one has none; for a
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
	// The bytes of its slab a device buffer takes, whole pages (RgfxSlabs); 0 for any other,
	// which takes its length.
	NSUInteger _allocatedSize;
	// The GPU address the create picked, which goes back after the destroy where `_ownsAddress`:
	// a buffer over pages lies past it by its pointer's offset in its page; 0 for any other,
	// whose pick is `_gpuAddress`.
	uint64_t _addressPick;
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
	if (serializer->refused(created))
		return nil; // the device never got it
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
	if (serializer->refused(created))
		return nil; // the device never got it
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

+ (instancetype)bufferOverSurfacePagesWithContents:(void *)contents
                                      blockOffset:(uint64_t)blockOffset
                                        blockSize:(uint64_t)blockSize
                                           length:(NSUInteger)length
                                          options:(MTLResourceOptions)options
                                      deallocator:(void (^)(void))deallocator
                                           device:(id<MTLDevice>)device
                                       serializer:(RgfxSerializer *)serializer
{
	if (serializer == nullptr) {
		return nil; // a buffer must have a route to the host
	}
	RgfxCreated created = serializer->createBufferOverSurfacePages(
	    length, static_cast<uint32_t>(options), blockOffset, blockSize);
	if (serializer->refused(created))
		return nil; // the device never got it
	return [[RgfxBuffer alloc] initWithObjId:created.obj_id
	                              gpuAddress:0
	                                  length:length
	                                 options:options
	                                contents:contents
	                             deallocator:deallocator
	                                  device:device
	                              serializer:serializer];
}

+ (instancetype)bufferForDeviceWithLength:(NSUInteger)length
                                  options:(MTLResourceOptions)options
                                    bytes:(const void *)bytes
                                maxLength:(NSUInteger)maxLength
                                    slabs:(RgfxSlabs *)slabs
                                   device:(id<MTLDevice>)device
                               serializer:(RgfxSerializer *)serializer
{
	if (length == 0 || (maxLength != 0 && length > maxLength))
		return nil;
	if (!rgfxCpuStorage(options)) {
		// The GPU's alone: no bytes reach it from here.
		if (bytes != nullptr)
			return nil;
		return [RgfxBuffer bufferWithLength:length options:options device:device serializer:serializer];
	}
	if (slabs == nullptr || serializer == nullptr)
		return nil;
	// Buffers share a slab where the host's would share a heap: of one storage, CPU cache and
	// hazard tracking mode, a Default one resolved as Metal resolves a buffer's.
	const uint32_t resolved = static_cast<uint32_t>(
	    RgfxResolveHazardTracking(options, MTLResourceHazardTrackingModeTracked));
	RgfxSlabs::Range range;
	if (!slabs->take(length, resolved, bytes != nullptr ? length : 0, &range))
		return nil;
	if (bytes != nullptr)
		std::memcpy(range.contents, bytes, length);
	RgfxSlabSource::Made made;
	if (!slabs->makeBuffer(range, length, static_cast<uint32_t>(options), &made)) {
		slabs->give(range); // no buffer lies over the range
		return nil;
	}
	// The range goes back to its slab once the host has dropped the buffer.
	RgfxBuffer *buffer = [[RgfxBuffer alloc] initWithObjId:made.objId
	                                            gpuAddress:made.gpuAddress
	                                                length:length
	                                               options:options
	                                              contents:range.contents
	                                           deallocator:^{ slabs->give(range); }
	                                                device:device
	                                            serializer:serializer];
	if (buffer != nil) {
		buffer->_ownsAddress = made.ownsAddress;
		buffer->_allocatedSize = range.size;
	}
	return buffer;
}

+ (instancetype)bufferWithBytesNoCopy:(void *)pointer
                               length:(NSUInteger)length
                              options:(MTLResourceOptions)options
                          deallocator:(void (^)(void *, NSUInteger))deallocator
                               region:(RgfxRegion *)region
                                slabs:(RgfxSlabs *)slabs
                               device:(id<MTLDevice>)device
                           serializer:(RgfxSerializer *)serializer
{
	if (length == 0 || !rgfxCpuStorage(options) || region == nullptr || slabs == nullptr ||
	    serializer == nullptr)
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
	// The app's own pages in place, where the kext wires them for the buffer and the host maps
	// them: nothing copied or synced. Read-only and Managed memory, and memory the kext or the
	// host makes no pages block of, takes a shadow in a slab below.
	if (pointer != nullptr && writable && region->hasPagesBlocks() &&
	    (options & MTLResourceStorageModeMask) == MTLResourceStorageModeShared) {
		RgfxBuffer *inPlace = [RgfxBuffer bufferOverPagesOf:pointer
		                                             length:length
		                                            options:options
		                                        deallocator:appDeallocator
		                                             region:region
		                                             device:device
		                                         serializer:serializer];
		if (inPlace != nil)
			return inPlace;
	}
	// The GPU's copy is a range of a slab, as a device buffer of these options takes
	// (RgfxSlabs.hpp): the process's own pages where the device takes them, so the region holds
	// no app-sized copy. The shadow writes all of it from the memory, so only a null pointer's is
	// zeroed.
	const uint32_t resolved = static_cast<uint32_t>(
	    RgfxResolveHazardTracking(options, MTLResourceHazardTrackingModeTracked));
	RgfxSlabs::Range range;
	if (!slabs->take(length, resolved, pointer != nullptr ? length : 0, &range))
		return nil;
	uint8_t *bytes = range.contents;
	std::shared_ptr<RgfxMemoryShadow> shadow;
	void *contents = pointer;
	if (pointer != nullptr) {
		shadow = std::make_shared<RgfxMemoryShadow>(pointer, bytes, length, writable);
		if (!shadow->ok()) {
			slabs->give(range);
			return nil;
		}
	} else {
		// No memory of the caller's: the range is the buffer's contents, as Metal's buffer has
		// memory of its own.
		contents = bytes;
	}
	RgfxSlabSource::Made made;
	if (!slabs->makeBuffer(range, length, static_cast<uint32_t>(options), &made)) {
		slabs->give(range); // no buffer lies over the range
		return nil;
	}
	// The range goes back to its slab once the host has dropped the buffer, then the caller's
	// memory.
	RgfxBuffer *buffer = [[RgfxBuffer alloc] initWithObjId:made.objId
	                                            gpuAddress:made.gpuAddress
	                                                length:length
	                                               options:options
	                                              contents:contents
	                                           deallocator:^{
		                                           slabs->give(range);
		                                           if (appDeallocator != nil)
			                                           appDeallocator(pointer, length);
	                                           }
	                                                device:device
	                                            serializer:serializer];
	if (buffer == nil)
		return nil;
	buffer->_ownsAddress = made.ownsAddress;
	buffer->_shadow = std::move(shadow);
	return buffer;
}

// A no-copy buffer over the app's own pages (bufferWithBytesNoCopy): a pages block of them, and a
// buffer over its view at the pointer's offset in its page, which keeps the block until the host
// has dropped the buffer. Nil where the kext gives no block, or the device never got the create.
+ (instancetype)bufferOverPagesOf:(void *)pointer
                           length:(NSUInteger)length
                          options:(MTLResourceOptions)options
                      deallocator:(void (^)(void *, NSUInteger))appDeallocator
                           region:(RgfxRegion *)region
                           device:(id<MTLDevice>)device
                       serializer:(RgfxSerializer *)serializer
{
	RgfxRegion::Block block = region->acquirePages(pointer, length, true);
	if (!block.ok)
		return nil;
	const uint64_t offset = reinterpret_cast<uintptr_t>(pointer) & (kRgfxGuestPage - 1);
	RgfxCreated created = serializer->createBufferOverPages(length, static_cast<uint32_t>(options),
	                                                        block.offset, block.size, offset);
	if (created.obj_id == 0 || serializer->refused(created)) {
		region->release(block.offset); // no buffer lies over the block
		return nil;
	}
	// The block goes back once the host has dropped the buffer, then the caller's memory: the
	// kext unwires its pages once the device has answered the block's release.
	const uint64_t at = block.offset;
	RgfxBuffer *buffer = [[RgfxBuffer alloc] initWithObjId:created.obj_id
	                                            gpuAddress:created.gpu_address + offset
	                                                length:length
	                                               options:options
	                                              contents:pointer
	                                           deallocator:^{
		                                           region->release(at);
		                                           if (appDeallocator != nil)
			                                           appDeallocator(pointer, length);
	                                           }
	                                                device:device
	                                            serializer:serializer];
	if (buffer != nil) {
		buffer->_ownsAddress = YES;
		buffer->_addressPick = created.gpu_address;
	}
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
		// A Private buffer's bytes are the GPU's alone: `contents` answers NULL, as Metal
		// documents for the mode, and nothing of the guest's memory stands for them.
		if (length > 0 && rgfxCpuStorage(options)) {
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
	// The device buffer's own pages are the guest's alone. The host drops the buffer as the app
	// lets it go, after any command buffer that names it (each keeps it while it runs), and only
	// then is the memory under a region-backed one given back (its block, or the no-copy
	// caller's pages), so nothing reuses it while the host may still read it.
	if (_ownsContents)
		std::free(_contents);
	void (^deallocator)(void) = _hostDeallocator;
	RgfxSerializer *serializer = _serializer;
	uint64_t address = !_ownsAddress ? 0 : _addressPick != 0 ? _addressPick : _gpuAddress;
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

// The bytes the buffer takes: a device buffer's range of its slab, whole 4 KiB pages, as both
// x86-64 Macs' GPUs answer (4096 for a 256-byte Shared buffer, measured: AMD 5300M, Intel 630;
// the M3 Max packs small buffers and answers 256); any other its length.
- (NSUInteger)allocatedSize
{
	return _allocatedSize != 0 ? _allocatedSize : _length;
}

- (uint64_t)gpuAddress
{
	return _gpuAddress;
}

// The buffer's options as Metal answers them (measured on the M3 Max under macOS 27): those it was
// made with, a Default hazard tracking mode resolved to Tracked (0x200 for options of 0, 0x220 for
// Private, 0x201 for WriteCombined, and Untracked kept); a heap's buffer answers the heap's,
// whatever it was asked (0x120 in a Private heap, for a buffer asked to be Tracked too).
- (MTLResourceOptions)resourceOptions
{
	id<MTLHeap> heap = self.heap;
	return heap != nil ? heap.resourceOptions
	                   : RgfxResolveHazardTracking(_options, MTLResourceHazardTrackingModeTracked);
}

// The options it was made with, as asked; a heap's buffer answers the heap's, as resourceOptions.
- (MTLResourceOptions)unfilteredResourceOptions
{
	id<MTLHeap> heap = self.heap;
	return heap != nil ? heap.resourceOptions : _options;
}

// The modes `resourceOptions` holds: an app reads `storageMode` to tell a Managed buffer, whose
// writes it names with didModifyRange:.
- (MTLStorageMode)storageMode
{
	return static_cast<MTLStorageMode>((self.resourceOptions & MTLResourceStorageModeMask) >>
	                                   MTLResourceStorageModeShift);
}

- (MTLCPUCacheMode)cpuCacheMode
{
	return static_cast<MTLCPUCacheMode>((self.resourceOptions & MTLResourceCPUCacheModeMask) >>
	                                    MTLResourceCPUCacheModeShift);
}

- (MTLHazardTrackingMode)hazardTrackingMode
{
	return static_cast<MTLHazardTrackingMode>((self.resourceOptions &
	                                           MTLResourceHazardTrackingModeMask) >>
	                                          MTLResourceHazardTrackingModeShift);
}

// --- the resource SPI (RgfxMetalPrivate.h) ---------------------------------------------

RGFX_RESOURCE_SPI

// The CPU's address of the buffer's memory, its contents, as Metal's (measured on the M3 Max under
// macOS 27): NULL for a Private buffer. A heap's buffer lies at the heap's CPU address plus its
// offset there, which for a Private heap, at NULL, is the offset itself.
- (void *)virtualAddress
{
	if (_contents == nullptr && self.heap != nil) {
		return reinterpret_cast<void *>(static_cast<uintptr_t>(self.rgfxHeapPlacement));
	}
	return _contents;
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

// Whether a texture of `descriptor` over the buffer from `offset`, rows `bytesPerRow` apart, lies
// at the device's linear texture alignment for its format
// (minimumLinearTextureAlignmentForPixelFormat:): Metal's rule for both forms of a 2D texture over
// a buffer of either storage, the public one and the linear SPI. Off it the AMD 5300M answers nil and the Intel 630 ends the process at
// its validation's failed assertion (measured under macOS 26.7.1: `oracle_storage AMD linalign`, a
// 468 x 36 BGRA8 texture at offsets 0, 0x40 and 0x100 and pitches 1872, 1920 and 2048, made at
// offsets 0 and 0x100 with the pitch 2048 alone); the bundle answers nil, as the AMD, logged at
// debug. A texture buffer takes an alignment of its own, and a device that answers none (a unit
// test's nil device) asks none.
- (BOOL)rgfxAtLinearAlignment:(MTLTextureDescriptor *)descriptor
                       offset:(NSUInteger)offset
                  bytesPerRow:(NSUInteger)bytesPerRow
{
	if (descriptor == nil || descriptor.textureType == MTLTextureTypeTextureBuffer)
		return YES;
	const NSUInteger alignment =
	    [self.device minimumLinearTextureAlignmentForPixelFormat:descriptor.pixelFormat];
	if (alignment <= 1 || (offset % alignment == 0 && bytesPerRow % alignment == 0))
		return YES;
	os_log_debug(OS_LOG_DEFAULT,
	             "RGFX: a texture over buffer 0x%llx at offset %lu, %lu bytes a row (format %lu): "
	             "not at the device's linear texture alignment, %lu: answered nil",
	             (unsigned long long)[self rgfxObjId], (unsigned long)offset,
	             (unsigned long)bytesPerRow, (unsigned long)descriptor.pixelFormat,
	             (unsigned long)alignment);
	return NO;
}

// A texture over the buffer's bytes from `offset`, rows `bytesPerRow` apart
// (TEXTURE_FROM_BUFFER); it keeps the buffer, which its `buffer` answers. Metal ends the process
// at a descriptor of more than one level, with these words, on the AMD, the Intel and the M3 GPU
// alike (docs/graphics-kext/dgpu-oracle.md, Q11), and so does the bundle, sending nothing. One off
// the device's linear texture alignment is nil (rgfxAtLinearAlignment:).
- (id<MTLTexture>)newTextureWithDescriptor:(MTLTextureDescriptor *)descriptor
                                    offset:(NSUInteger)offset
                               bytesPerRow:(NSUInteger)bytesPerRow
{
	if (descriptor != nil && descriptor.mipmapLevelCount > 1) {
		RgfxMetalAssertionFailed(
		    "_mtlValidateStrideTextureParameters",
		    [NSString stringWithFormat:@"Texture Descriptor Validation\nLinear texture: cannot be "
		                               @"mipmapped, mipmapLevelCount (%lu) disallowed\n",
		                               (unsigned long)descriptor.mipmapLevelCount]);
	}
	if (![self rgfxAtLinearAlignment:descriptor offset:offset bytesPerRow:bytesPerRow])
		return nil;
	return [RgfxTexture textureWithDescriptor:descriptor
	                                   buffer:self
	                                   offset:offset
	                              bytesPerRow:bytesPerRow
	                                   device:self.device
	                               serializer:[self serializer]];
}

// A texture over the buffer's bytes from `offset`, rows `bytesPerRow` apart, as
// newTextureWithDescriptor:offset:bytesPerRow: makes one (TEXTURE_FROM_BUFFER), with the buffer's
// storage. WindowServer's MetalTiledBacking makes one for each tile of a window's backing: on the
// x86-64 guest (macOS 26) the display's backing is a Shared buffer of 3840 x 2400 BGRA8 pixels
// (36864000 bytes), and the tile a 2D BGRA8Unorm descriptor of 1024 x 1024, usage ShaderRead |
// RenderTarget, storage Managed, at offset 0 with the backing's pitch, 15360 bytes a row.
//
// Measured on the M3 Max under macOS 27.0.1: the buffer of Metal's validation layer
// (MTLDebugBuffer) answers the selector, checks the call and passes it to the driver's buffer,
// which on the M3 (AGXG15XFamilyBuffer, of every storage mode) does not answer it, so the M3 makes
// no such texture. The validation asserts that the descriptor's storage is Managed, that the
// buffer's is Shared, and that offset + bytesPerRow x (height - 1) + the row's bytes fit in the
// buffer (a 1024 x 1024 RGBA32Float tile at a pitch of 61440 needs 62869504 bytes); any offset
// and pitch past that pass (an offset of 4, a pitch of 4100). WindowServer's arguments pass. Not
// measured: what the driver of a GPU with Managed storage answers, as no such GPU is on hand, nor
// how Metal refuses what its validation asserts at: the bundle answers nil, logged once for each
// rule, as it does for the linear form's. A descriptor of more than one level the AMD and the
// Intel GPU make, level 0 over the buffer's bytes and levels 1 and up in the GPU's own memory
// (docs/graphics-kext/dgpu-oracle.md, Q11), which the host does not carry: the bundle answers nil,
// logged once, and sends nothing.
- (id<MTLTexture>)newTiledTextureWithDescriptor:(MTLTextureDescriptor *)descriptor
                                         offset:(NSUInteger)offset
                                    bytesPerRow:(NSUInteger)bytesPerRow
{
	if (descriptor == nil) {
		return nil;
	}
	if (descriptor.storageMode != MTLStorageModeManaged) {
		static std::atomic<bool> logged{false};
		if (!logged.exchange(true)) {
			NSLog(@"RGFX: a tiled texture over buffer 0x%llx with descriptor storage %lu, not "
			      @"Managed, which Metal's validation asserts at: answered nil",
			      (unsigned long long)[self rgfxObjId], (unsigned long)descriptor.storageMode);
		}
		return nil;
	}
	if ([self storageMode] != MTLStorageModeShared) {
		static std::atomic<bool> logged{false};
		if (!logged.exchange(true)) {
			NSLog(@"RGFX: a tiled texture over buffer 0x%llx of storage %lu, not Shared, which "
			      @"Metal's validation asserts at: answered nil",
			      (unsigned long long)[self rgfxObjId], (unsigned long)[self storageMode]);
		}
		return nil;
	}
	if (descriptor.mipmapLevelCount > 1) {
		static std::atomic<bool> logged{false};
		if (!logged.exchange(true)) {
			NSLog(@"RGFX: a tiled texture over buffer 0x%llx of %lu mipmap levels, whose levels "
			      @"1 and up the host does not carry: answered nil",
			      (unsigned long long)[self rgfxObjId],
			      (unsigned long)descriptor.mipmapLevelCount);
		}
		return nil;
	}
	RgfxFormatBlock block = {};
	if (rgfxPixelFormatBlock(static_cast<uint32_t>(descriptor.pixelFormat), &block) &&
	    block.linear && descriptor.height > 0) {
		const unsigned __int128 needs =
		    static_cast<unsigned __int128>(offset) +
		    static_cast<unsigned __int128>(bytesPerRow) * (descriptor.height - 1) +
		    static_cast<unsigned __int128>(descriptor.width) * block.bytes;
		if (needs > [self length]) {
			static std::atomic<bool> logged{false};
			if (!logged.exchange(true)) {
				NSLog(@"RGFX: a tiled texture past buffer 0x%llx's %lu bytes (offset %lu, "
				      @"%lu bytes a row, %lu x %lu), which Metal's validation asserts at: "
				      @"answered nil",
				      (unsigned long long)[self rgfxObjId], (unsigned long)[self length],
				      (unsigned long)offset, (unsigned long)bytesPerRow,
				      (unsigned long)descriptor.width, (unsigned long)descriptor.height);
			}
			return nil;
		}
	}
	// As newTextureWithDescriptor:offset:bytesPerRow: makes one, at any offset and pitch: the AMD's
	// tiled form takes those its public form refuses (`oracle_storage AMD linalign`).
	MTLTextureDescriptor *shared = [descriptor copy];
	shared.storageMode = [self storageMode];
	return [RgfxTexture textureWithDescriptor:shared
	                                   buffer:self
	                                   offset:offset
	                              bytesPerRow:bytesPerRow
	                                   device:self.device
	                               serializer:[self serializer]];
}

// A texture over the buffer's bytes as Metal's linear form makes one, which the host's platform
// makes by the same form (LINEAR_TEXTURE_FROM_BUFFER), carrying `bytesPerImage` as asked. Metal's
// (measured on the M3 Max under macOS 27) is a texture of the buffer's bytes from the offset, rows
// `bytesPerRow` apart, as newTextureWithDescriptor:offset:bytesPerRow: makes one, and takes a 2D
// array too, its slices `bytesPerRow` times its height apart whatever `bytesPerImage`; it ends the
// process, at its validation's failed assertion, for a texture of another type than 2D, 2D array
// and texture buffer, or of more than one level, which the bundle answers nil, logged, as it
// answers a view Metal ends the process at. One past the buffer's bytes Metal answers nil, and the
// host refuses; one off the device's linear texture alignment is nil (rgfxAtLinearAlignment:).
- (id<MTLTexture>)newLinearTextureWithDescriptor:(MTLTextureDescriptor *)descriptor
                                          offset:(NSUInteger)offset
                                     bytesPerRow:(NSUInteger)bytesPerRow
                                   bytesPerImage:(NSUInteger)bytesPerImage
{
	if (descriptor == nil) {
		return nil;
	}
	const MTLTextureType type = descriptor.textureType;
	if ((type != MTLTextureType2D && type != MTLTextureType2DArray &&
	     type != MTLTextureTypeTextureBuffer) ||
	    descriptor.mipmapLevelCount != 1) {
		static std::atomic<bool> logged{false};
		if (!logged.exchange(true)) {
			NSLog(@"RGFX: a linear texture over buffer 0x%llx of type %lu and %lu levels, which "
			      @"Metal's validation ends the process at: answered nil",
			      (unsigned long long)[self rgfxObjId], (unsigned long)type,
			      (unsigned long)descriptor.mipmapLevelCount);
		}
		return nil;
	}
	if (![self rgfxAtLinearAlignment:descriptor offset:offset bytesPerRow:bytesPerRow])
		return nil;
	return [RgfxTexture linearTextureWithDescriptor:descriptor
	                                         buffer:self
	                                         offset:offset
	                                    bytesPerRow:bytesPerRow
	                                  bytesPerImage:bytesPerImage
	                                         device:self.device
	                                     serializer:[self serializer]];
}

// The size of the allocation the buffer lies in: its heap's, or its own, as Metal's (measured on
// the M3 Max under macOS 27: a 1 MiB buffer answers its size, one in a heap the heap's; Metal
// suballocates small buffers from 128 KiB pools, which answer the pool's size). The slabs the
// device's buffers lie in are the host's objects and none of the app's, so a buffer answers its
// own pages.
- (uint64_t)resourceSize
{
	id<MTLHeap> heap = self.heap;
	return heap != nil ? heap.size : self.allocatedSize;
}

// The device makes no buffer over an IOSurface, so the buffer lies in none, as a buffer made from
// a length or bytes answers none (measured on the M3 Max under macOS 27).
- (IOSurfaceRef)iosurface
{
	return nullptr;
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
