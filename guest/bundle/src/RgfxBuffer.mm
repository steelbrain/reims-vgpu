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

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>

#include "RgfxLog.hpp"
#include "RgfxManagedLayout.hpp"
#include "RgfxMemoryShadow.hpp"
#include "RgfxRegion.hpp"
#include "RgfxSerializer.hpp"
#include "RgfxSlabs.hpp"

namespace {
/// The page the kext rounds a pages block's range out to (its RgfxPagesBlock.hpp), the guest's:
/// the device's view of a block starts at the page of the buffer's first byte.
constexpr uint64_t kRgfxGuestPage = 0x1000;

/// MTLResourceOptions bit 24, Metal's private bit, which a create carries to the host
/// (kRgfxHostResourceOptions) and the host's runtime takes as read-only memory.
constexpr MTLResourceOptions kBit24 = 1u << 24;

/// The joins of no-copy buffers over the app's own pages (RgfxRegion::acquirePages): one of
/// buffers with bit 24, and one of the rest.
constexpr uint32_t kPagesJoin = 1;
constexpr uint32_t kPagesJoinBit24 = 2;

// The whole 4 KiB pages `length` bytes at `pointer` lie in, which both x86-64 Macs' GPUs answer a
// buffer's allocatedSize with (oracle_bundle buffers and nocopy.*, AMD 5300M and Intel 630, macOS
// 26.7.1): a buffer made from a length or bytes in any storage mode answers its length rounded up
// to a page (4096 for 1 byte, 8192 for 4097), and one made over the app's bytes the pages its
// bytes meet (8192 for 4096 bytes 16 bytes into a page).
NSUInteger rgfxPagesSpanned(const void *pointer, NSUInteger length)
{
	const uint64_t start = reinterpret_cast<uintptr_t>(pointer) & (kRgfxGuestPage - 1);
	return static_cast<NSUInteger>((start + length + kRgfxGuestPage - 1) & ~(kRgfxGuestPage - 1));
}

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

/// The name Metal's assertions give a texture type, of those its linear-texture assertion names
/// (`oracle_storage DEVICE lintypes`): nil for another.
NSString *rgfxLinearTypeName(MTLTextureType type)
{
	switch (type) {
	case MTLTextureType1D:
		return @"MTLTextureType1D";
	case MTLTextureType1DArray:
		return @"MTLTextureType1DArray";
	case MTLTextureType2DArray:
		return @"MTLTextureType2DArray";
	case MTLTextureType2DMultisample:
		return @"MTLTextureType2DMultisample";
	case MTLTextureTypeCube:
		return @"MTLTextureTypeCube";
	case MTLTextureTypeCubeArray:
		return @"MTLTextureTypeCubeArray";
	case MTLTextureType3D:
		return @"MTLTextureType3D";
	case MTLTextureType2DMultisampleArray:
		return @"MTLTextureType2DMultisampleArray";
	default:
		return nil;
	}
}

/// Ends the process as Metal's validation does at a texture over a buffer of a type it does not
/// make there, `allowed` naming the types it makes ("MTLTextureType2D", or the linear SPI's
/// "MTLTextureType2D or linear MTLTextureType2DArray"): a line for a depth other than 1, then the
/// type's (measured on the M3 Max under macOS 27.0.1 and the AMD 5300M under macOS 26.7.1, for a
/// 3D texture 2 deep; the Intel 630 asserts "Buffer texture: cannot support depth > 1." there).
[[noreturn]] void rgfxLinearTypeDies(MTLTextureDescriptor *descriptor, NSString *type,
                                     NSString *allowed)
{
	NSMutableString *words = [NSMutableString stringWithString:@"Texture Descriptor Validation\n"];
	if (descriptor.depth != 1) {
		[words appendFormat:@"Linear texture: must have depth == 1, depth (%lu) disallowed\n",
		                    (unsigned long)descriptor.depth];
	}
	[words appendFormat:@"Linear texture: must be of type %@, textureType (%@) disallowed\n",
	                    allowed, type];
	RgfxMetalAssertionFailed("_mtlValidateStrideTextureParameters", words);
}

/// The granule an Apple GPU's buffer allocation is rounded up to, by which it answers a texture
/// over a buffer of at least that many bytes whose texels end past its allocation.
constexpr uint64_t kRgfxAllocationGranule = 16384;

/// The pool an Apple GPU suballocates a buffer of fewer bytes than kRgfxAllocationGranule from,
/// newBufferWithLength:'s and newBufferWithBytes:'s of every storage mode: its resourceSize (SPI),
/// 131072 on the M3 Max under macOS 27.0.1 (`oracle_pastlen Apple limits`).
constexpr uint64_t kRgfxApplePool = 131072;

/// Whether a texture over a buffer of `length` bytes, from `offset` with rows `bytesPerRow` apart,
/// of `height` rows of `rowBytes` bytes of texels in each of `slices` slices (a 2D array's,
/// `array`, rounded up to 128 bytes apart), is nil. Its end, counted from the buffer's start, is a
/// 2D texture's or a texture buffer's last texel, whatever the bytes per row past it, and a 2D
/// array's slices' end, each slice its rows' whole pitch, as the M3 Max counts it under macOS
/// 27.0.1 (`oracle_pastlen Apple limits`, dgpu-oracle.md Q11). On every device, where the host's
/// platform answers it nil (metal2vulkan's linear_plan.rs "Length"): its texels end past the
/// length, the buffer is at least 16 KiB, and the end lies past `hostAllocation`
/// (rgfxHostAllocation); the AMD 5300M and the Intel 630 make it past the allocation too (`tbpa`,
/// macOS 26.7.1), but the host's platform refuses a create the bundle posts for one. And where
/// `appleAllocation` is not 0, an Apple GPU's allocation of the buffer (rgfxAppleAllocation),
/// where the end lies past it, as the M3 Max answers it nil, a 2D array's with its texels inside
/// the length too (two slices of 64 x 16 R32Uint, rows 512 bytes apart, from 256 over 16384
/// bytes: the last texel at the length, the slices' end 256 bytes past it). Short of both the
/// texture is made over bytes past the length however far.
bool rgfxTexelsPastAllocation(uint64_t length, uint64_t offset, uint64_t bytesPerRow,
                              uint64_t height, uint64_t rowBytes, uint64_t slices, bool array,
                              uint64_t hostAllocation, uint64_t appleAllocation)
{
	using wide = unsigned __int128;
	const wide rows = std::max<uint64_t>(1, height);
	const wide count = std::max<uint64_t>(1, slices);
	const wide image = static_cast<wide>(bytesPerRow) * rows;
	const wide pitch = array ? (image + 127) / 128 * 128 : image;
	const wide texelsEnd = offset + pitch * (count - 1) + bytesPerRow * (rows - 1) + rowBytes;
	const wide end = array ? offset + pitch * count : texelsEnd;
	if (appleAllocation != 0 && end > appleAllocation)
		return true;
	if (texelsEnd <= length || length < kRgfxAllocationGranule)
		return false;
	return end > hostAllocation;
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
	// The bytes allocatedSize answers: the whole pages a device buffer's range of its slab takes
	// (RgfxSlabs), a Private one's length takes, or a no-copy one's bytes meet; 0 for any other,
	// which answers its length.
	NSUInteger _allocatedSize;
	// The GPU address the create picked, which goes back after the destroy where `_ownsAddress`:
	// a buffer over pages lies past it by its pointer's offset in its page; 0 for any other,
	// whose pick is `_gpuAddress`.
	uint64_t _addressPick;
	// A no-copy buffer over the app's own pages (bufferOverPagesOf): the region whose pages block
	// it lies over, which the buffer leaves joinable no longer as it goes (RgfxRegion::leavePages),
	// and the block's offset; null for any other.
	RgfxRegion *_pagesRegion;
	uint64_t _pagesBlock;
	// YES for the buffer newBufferWithBytesNoCopy: makes (bufferWithBytesNoCopy), over the
	// caller's bytes, which stays NonVolatile (setPurgeableState:).
	BOOL _noCopy;
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
	if (length == 0 || (maxLength != 0 && length > maxLength) || RgfxMakesNoResourceOf(options))
		return nil;
	if (!rgfxCpuStorage(options)) {
		// The GPU's alone: no bytes reach it from here.
		if (bytes != nullptr)
			return nil;
		RgfxBuffer *buffer =
		    [RgfxBuffer bufferWithLength:length options:options device:device serializer:serializer];
		if (buffer != nil)
			buffer->_allocatedSize = rgfxPagesSpanned(nullptr, length);
		return buffer;
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
		slabs->give(range); // no buffer lies over the range, but a copy made for it goes first
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
	RgfxBuffer *buffer = [RgfxBuffer bufferOverBytes:pointer
	                                          length:length
	                                         options:options
	                                     deallocator:deallocator
	                                          region:region
	                                           slabs:slabs
	                                          device:device
	                                      serializer:serializer];
	if (buffer != nil) {
		buffer->_allocatedSize = rgfxPagesSpanned(buffer->_contents, length);
		buffer->_noCopy = YES;
	}
	return buffer;
}

// bufferWithBytesNoCopy's buffer, before it answers its allocatedSize.
+ (instancetype)bufferOverBytes:(void *)pointer
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
	// them: a Shared buffer's memory, and a Managed buffer's CPU copy, where the device makes a
	// Managed buffer over a Shared one (bufferOverPagesOf). Shared memory the process maps
	// read-only lies there too, where the kext makes a block of it (RGFX_FEATURE_READ_ONLY_PAGES),
	// its buffer made with Metal's private bit 24, which the host's runtime takes as memory whose
	// GPU writes it drops: Metal makes a buffer over read-only memory whose GPU reads see the
	// memory and whose blit fills and kernel stores complete with their writes dropped, the
	// memory and a file mapped under it unchanged (the AMD 5300M and the Intel UHD 630 under
	// macOS 26.7.1, oracle_bundle nocopy.ro; the M3, docs/accuracy-ledger.md
	// `read-only-pages-written`, whose row the host's drop follows). The kext marks the block
	// read-only, and the host lays nothing over it that could write its pages. Read-only Managed
	// memory, whose copies the host would write back to the pages, and memory the kext or the
	// host makes no pages block of (read-only memory among it where the kext makes none), takes
	// a shadow in a slab below, which never writes back to read-only memory.
	const bool managed = (options & MTLResourceStorageModeMask) == MTLResourceStorageModeManaged;
	const bool overPages =
	    writable ? !managed || slabs->overCopy(static_cast<uint32_t>(options)) : !managed;
	if (pointer != nullptr && overPages && region->hasPagesBlocks()) {
		RgfxBuffer *inPlace = [RgfxBuffer bufferOverPagesOf:pointer
		                                             length:length
		                                            options:options
		                                           writable:writable
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
		slabs->give(range); // no buffer lies over the range, but a copy made for it goes first
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
// has dropped the buffer. Nil where the kext gives no block, or the device never got a create.
// A Managed one is a Managed buffer whose CPU copy is a Shared one over the view
// (MANAGED_BUFFER_CREATE), which the host makes two copies where its GPU has memory of its own,
// as Metal keeps the AMD 5300M's no-copy Managed buffer (its GPU sees a CPU write only after a
// didModifyRange:, and its CPU a GPU write only after a synchronize; docs/graphics-kext/
// dgpu-oracle.md, Q3), and one copy, over the pages, where its GPU shares the CPU's memory, as the
// Intel 630's is. Its didModifyRange: and a blit's synchronize of it reach the host, which Apple's
// OpenCL makes around its maps of a buffer's memory: a synchronizeResource: before a map, a
// didModifyRange: after a map for writing (measured: a macOS 26.6.1 guest on the RTX 5080 host,
// 2026-11-11). The Managed buffer's address keeps the pointer's offset in its page too.
//
// Buffers over the same bytes share their pages block (RgfxRegion::acquirePages) while those bytes
// are the same memory, so the host makes one view of the pages, and one copy where it copies them
// rather than importing them, as on a Mac they lie in the same pages; over other memory the app
// mapped at the same address since, a buffer takes a block of its own. Apple's OpenCL makes such buffers: a Shared no-copy one over
// its staging pages at each clEnqueueWriteBuffer, the same pages each time, while the one before
// still lives (measured: a macOS 26.6.1 guest on the RTX 5080 host, 2026-11-11). One with Metal's
// private bit 24, which the host's runtime takes as read-only memory, joins only its own kind,
// never a writable one's view.
//
// Memory the process maps read-only (not `writable`, Shared alone) takes a block the kext wires
// for reading and marks read-only, which only buffers of bit 24 lie over: the buffer is made with
// the bit, sent to the host alone, so its GPU writes are dropped as Metal drops them, and its
// options answer the app's, as Metal's leave the bit out. Its blocks join only read-only ones
// (RgfxRegion keys them by writability).
+ (instancetype)bufferOverPagesOf:(void *)pointer
                           length:(NSUInteger)length
                          options:(MTLResourceOptions)options
                         writable:(bool)writable
                      deallocator:(void (^)(void *, NSUInteger))appDeallocator
                           region:(RgfxRegion *)region
                           device:(id<MTLDevice>)device
                       serializer:(RgfxSerializer *)serializer
{
	const MTLResourceOptions hostOptions = writable ? options : options | kBit24;
	const uint32_t join = (hostOptions & kBit24) != 0 ? kPagesJoinBit24 : kPagesJoin;
	RgfxRegion::Block block = region->acquirePages(pointer, length, writable, join);
	if (!block.ok)
		return nil;
	// The block's view starts at the page of the pointer's first byte, of the kext's page.
#if defined(__arm64__)
	const uint64_t page = region->pagesPage();
#else
	const uint64_t page = kRgfxGuestPage;
#endif
	const uint64_t offset = reinterpret_cast<uintptr_t>(pointer) & (page - 1);
	const bool managed = (options & MTLResourceStorageModeMask) == MTLResourceStorageModeManaged;
	const MTLResourceOptions overPages =
	    managed ? (hostOptions & ~MTLResourceStorageModeMask) | MTLResourceStorageModeShared
	            : hostOptions;
	RgfxCreated created = serializer->createBufferOverPages(
	    length, static_cast<uint32_t>(overPages), block.offset, block.size, offset);
	const uint64_t at = block.offset;
	if (created.obj_id == 0 || serializer->refused(created)) {
		region->leavePages(at);
		region->release(at); // no buffer lies over the block
		return nil;
	}
	// A Managed one's CPU copy: the Shared buffer over the pages, destroyed once the Managed
	// buffer's destroy is posted, its address then given back, and then the block.
	const RgfxObjId copy = managed ? created.obj_id : 0;
	const uint64_t copyPick = created.gpu_address;
	if (managed) {
		// The app's own bytes, which its program wrote: nothing is said of them.
		created = serializer->createManagedBuffer(copy, length, static_cast<uint32_t>(options),
		                                          length, RgfxSerializer::kDefaultAddressAlign,
		                                          offset);
		if (created.obj_id == 0 || serializer->refused(created)) {
			region->leavePages(at);
			serializer->destroy(copy, [serializer, copyPick, region, at] {
				serializer->releaseAddress(copyPick);
				region->release(at);
			});
			return nil;
		}
	}
	// The block goes back once the host has dropped the buffer, then the caller's memory: the
	// kext unwires its pages once the device has answered the block's release. It is joinable no
	// longer once the buffer has gone for the app (dealloc).
	void (^released)(void) = ^{
		region->release(at);
		if (appDeallocator != nil)
			appDeallocator(pointer, length);
	};
	RgfxBuffer *buffer = [[RgfxBuffer alloc] initWithObjId:created.obj_id
	                                            gpuAddress:created.gpu_address + offset
	                                                length:length
	                                               options:options
	                                              contents:pointer
	                                           deallocator:^{
		                                           if (copy == 0) {
			                                           released();
			                                           return;
		                                           }
		                                           serializer->destroy(copy, [serializer,
		                                                                      copyPick,
		                                                                      released] {
			                                           serializer->releaseAddress(copyPick);
			                                           released();
		                                           });
	                                           }
	                                                device:device
	                                            serializer:serializer];
	if (buffer == nil) {
		region->leavePages(at);
		return nil;
	}
	buffer->_ownsAddress = YES;
	buffer->_addressPick = created.gpu_address;
	buffer->_pagesRegion = region;
	buffer->_pagesBlock = at;
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
	// Gone for the app, which may free the memory under it once its deallocator runs: a buffer
	// made since over the same bytes takes a block of its own.
	if (_pagesRegion != nullptr)
		_pagesRegion->leavePages(_pagesBlock);
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

// The bytes the buffer takes, whole 4 KiB pages, as both x86-64 Macs' GPUs answer
// (rgfxPagesSpanned; the M3 Max packs small buffers and answers 256 for a 256-byte one): a device
// buffer's range of its slab, a Private one's length rounded up, a no-copy one's pages; any other
// its length.
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
// whatever it was asked (0x120 in a Private heap, for a buffer asked to be Tracked too). The three
// modes alone: the AMD 5300M and the Intel 630 answer a buffer made with any one of bits 10 to 31
// beside a storage mode without that bit, of a length or no-copy (0x200, 0x210 and 0x220 for
// Shared, Managed and Private; measured: macOS 26.7.1), the GL renderer's bit 20 (0x100000, as
// it makes its no-copy buffers) and Metal's private bit 24 among them.
// A no-copy buffer stays NonVolatile whatever it is told: its bytes are the caller's, which no
// purgeable allocation of the host backs. Metal's do so on the M3 Max, the AMD Radeon Pro 5300M
// and the Intel UHD Graphics 630 alike (NonVolatile to Volatile, KeepCurrent, Empty, KeepCurrent
// and NonVolatile; measured: macOS 27.0.1 and 26.7.1, oracle_bundle purgeable.kinds). Any other
// keeps the state it is told (RgfxResource).
- (MTLPurgeableState)setPurgeableState:(MTLPurgeableState)state
{
	if (_noCopy) {
		return MTLPurgeableStateNonVolatile;
	}
	return [super setPurgeableState:state];
}

- (MTLResourceOptions)resourceOptions
{
	id<MTLHeap> heap = self.heap;
	const MTLResourceOptions modes =
	    MTLResourceCPUCacheModeMask | MTLResourceStorageModeMask | MTLResourceHazardTrackingModeMask;
	return heap != nil ? heap.resourceOptions
	                   : RgfxResolveHazardTracking(_options & modes, MTLResourceHazardTrackingModeTracked);
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
	// A Managed buffer's contents are its CPU copy, in the process's memory: the host copies the
	// range to the GPU's copy before any command buffer committed after this, on any queue
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

- (BOOL)rgfxOverCallerBytes
{
	return _noCopy;
}

// The bytes from the buffer's start the host's platform makes a texture over it end within, for a
// buffer of 16 KiB or more (metal2vulkan's linear_plan.rs "Length", as the M3 Max counts them):
// a heap's buffer's heap's size, and any other's length rounded up to 16 KiB. The platform counts
// a buffer over pages, a no-copy buffer's or one over the device's region, from the host
// pointer's offset in its 16 KiB page, which the guest does not know and which only adds to the
// bound: the length rounded up never reaches past it.
- (uint64_t)rgfxHostAllocation
{
	id<MTLHeap> heap = self.heap;
	if (heap != nil)
		return heap.size;
	const uint64_t g = kRgfxAllocationGranule;
	return (_length + g - 1) / g * g;
}

// The bytes from the buffer's start an Apple GPU makes a texture over it end within: its
// allocation, the resourceSize (SPI) Metal answers for its buffer on the M3 Max under macOS
// 27.0.1, wherever the buffer lies in it (`oracle_pastlen Apple limits`, dgpu-oracle.md Q11). A
// heap's buffer's is the heap's size; a no-copy buffer's the 16 KiB pages its bytes meet; any
// other's its length rounded up to 16 KiB, but one of fewer bytes, which Metal suballocates from
// a pool, the pool's 128 KiB.
- (uint64_t)rgfxAppleAllocation
{
	id<MTLHeap> heap = self.heap;
	if (heap != nil)
		return heap.size;
	const uint64_t g = kRgfxAllocationGranule;
	if (_noCopy) {
		const uint64_t start = reinterpret_cast<uintptr_t>(_contents) & (g - 1);
		return (start + _length + g - 1) / g * g;
	}
	return _length < g ? kRgfxApplePool : (_length + g - 1) / g * g;
}

// Whether a texture of `descriptor` over the buffer from `offset`, rows `bytesPerRow` apart, is
// nil for its end past the host's platform's allocation of the buffer (rgfxHostAllocation), or an
// Apple GPU's (rgfxAppleAllocation) under its rules (rgfxTexelsPastAllocation), logged once: a 2D texture, a 2D array, its slices where the linear form lays them, or a texture
// buffer, of a format whose texture a buffer backs (another is answered on its own). The M3 Max
// answers a 2D texture's nil there as a texture buffer's (64 x 65 RGBA8Unorm over 16384 bytes, 64
// x 129 over 20000; `tbpa`, dgpu-oracle.md Q11), where the x86 Macs make both, and so it does past
// the 128 KiB pool of a buffer of fewer bytes than 16 KiB, past a no-copy buffer's pages and past
// a heap's size, which the host's platform carries a tail past the length for: the bundle answers
// those under an Apple GPU's rules alone (rgfxAppleGPU), as the x86 Macs make them.
- (BOOL)rgfxPastAllocation:(MTLTextureDescriptor *)descriptor
                    offset:(NSUInteger)offset
               bytesPerRow:(NSUInteger)bytesPerRow
{
	RgfxFormatBlock block = {};
	if (descriptor == nil ||
	    !rgfxPixelFormatBlock(static_cast<uint32_t>(descriptor.pixelFormat), &block) ||
	    !block.linear)
		return NO;
	const bool array = descriptor.textureType == MTLTextureType2DArray;
	const uint64_t rowBytes = static_cast<uint64_t>(descriptor.width) * block.bytes;
	const uint64_t apple = rgfxAppleGPU(self.device) ? [self rgfxAppleAllocation] : 0;
	if (!rgfxTexelsPastAllocation([self length], offset, bytesPerRow, descriptor.height, rowBytes,
	                              array ? descriptor.arrayLength : 1, array,
	                              [self rgfxHostAllocation], apple))
		return NO;
	static std::atomic<bool> logged{false};
	if (!logged.exchange(true)) {
		rgfxLogLine(@"RGFX: a texture of type %lu over buffer 0x%llx of %lu bytes past its allocation "
		            @"(offset %lu, %lu bytes a row, %lu x %lu%s): answered nil",
		            (unsigned long)descriptor.textureType, (unsigned long long)[self rgfxObjId],
		            (unsigned long)[self length], (unsigned long)offset, (unsigned long)bytesPerRow,
		            (unsigned long)descriptor.width, (unsigned long)descriptor.height,
		            apple != 0 ? ", an Apple GPU's rules" : "");
	}
	return YES;
}

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
	id<MTLDevice> device = self.device;
	const NSUInteger alignment =
	    [device respondsToSelector:@selector(rgfxLinearTextureAlignmentForPixelFormat:)]
	        ? [(id<RgfxLinearAlignmentSource>)device
	              rgfxLinearTextureAlignmentForPixelFormat:descriptor.pixelFormat]
	        : [device minimumLinearTextureAlignmentForPixelFormat:descriptor.pixelFormat];
	if (alignment <= 1 || (offset % alignment == 0 && bytesPerRow % alignment == 0))
		return YES;
	os_log_debug(rgfxLog(),
	             "RGFX: a texture over buffer 0x%llx at offset %lu, %lu bytes a row (format %lu): "
	             "not at the device's linear texture alignment, %lu: answered nil",
	             (unsigned long long)[self rgfxObjId], (unsigned long)offset,
	             (unsigned long)bytesPerRow, (unsigned long)descriptor.pixelFormat,
	             (unsigned long)alignment);
	return NO;
}

// A texture over the buffer's bytes from `offset`, rows `bytesPerRow` apart
// (TEXTURE_FROM_BUFFER); it keeps the buffer, which its `buffer` answers. Metal ends the process
// at a descriptor of more than one level, and at a 1D, 1D array, 2D array, cube or cube array
// type, with these words, on the AMD, the Intel and the M3 GPU (docs/graphics-kext/dgpu-oracle.md,
// Q11), and so does the bundle, sending nothing: the host's platform would end the host's process
// at the type. So it does at a 3D or multisample type, with the M3's words (rgfxLinearTypeDies);
// the x86 GPUs end the process there too, the Intel GPU with other words at the 3D one, and both
// at a multisample one in Shared storage before they reach its type. A texture buffer is made, as
// the AMD and the Intel GPU make one over a Shared or Managed buffer at any offset (4, 100 and
// 256 among them) with rows of at least its texels' bytes, and their kernels read the buffer's
// bytes from the offset (measured under macOS 26.7.1, RGBA32Float and RGBA8Unorm: a scratch
// probe, dgpu-oracle.md Q11), which the host's platform makes by this form too. Rows of 0 bytes
// end the process with the words both say; fewer than its texels' bytes the AMD answers nil, and
// so does the bundle, where the Intel GPU ends the process. One whose texels lie past the
// buffer's bytes is made, as both make it and the M3 Max does (4 RGBA32Float texels at offset
// 4064 of a 4096-byte buffer; dgpu-oracle.md Q11), and so does the host's platform, its texels
// past the buffer's bytes in a tail of the texture's own. A block-compressed format answers nil
// (below). A texture buffer or 2D texture past an Apple GPU's allocation of the buffer is nil, as
// the M3's and the host's (rgfxPastAllocation:). One off the device's linear texture alignment is
// nil (rgfxAtLinearAlignment:).
- (id<MTLTexture>)newTextureWithDescriptor:(MTLTextureDescriptor *)descriptor
                                    offset:(NSUInteger)offset
                               bytesPerRow:(NSUInteger)bytesPerRow
{
	// Where Metal's descriptor validation ends the process, so does the bundle: a depth texture
	// over the buffer's bytes, as the x86 Macs end it (rgfxEndAtInvalidTextureDescriptor), and as
	// an Apple GPU ends it after (rgfxEndAtLinearDepthStencil).
	rgfxEndAtInvalidTextureDescriptor(descriptor, self.device);
	rgfxEndAtLinearDepthStencil(descriptor, self.device);
	if (descriptor != nil && descriptor.mipmapLevelCount > 1) {
		RgfxMetalAssertionFailed(
		    "_mtlValidateStrideTextureParameters",
		    [NSString stringWithFormat:@"Texture Descriptor Validation\nLinear texture: cannot be "
		                               @"mipmapped, mipmapLevelCount (%lu) disallowed\n",
		                               (unsigned long)descriptor.mipmapLevelCount]);
	}
	if (descriptor != nil && descriptor.textureType == MTLTextureTypeTextureBuffer) {
		RgfxFormatBlock block = {};
		const uint64_t rowBytes =
		    rgfxPixelFormatBlock(static_cast<uint32_t>(descriptor.pixelFormat), &block) && block.linear
		        ? static_cast<uint64_t>(descriptor.width) * block.bytes
		        : 0;
		if (bytesPerRow == 0 && rowBytes != 0) {
			RgfxMetalAssertionFailed(
			    "_mtlValidateStrideTextureParameters",
			    [NSString stringWithFormat:@"Texture Descriptor Validation\nLinear texture: bytesPerRow (0) "
			                               @"must be greater or equal to (%llu) bytes\n",
			                               (unsigned long long)rowBytes]);
		}
		if (bytesPerRow < rowBytes) {
			static std::atomic<bool> logged{false};
			if (!logged.exchange(true)) {
				rgfxLogLine(@"RGFX: a texture buffer over buffer 0x%llx of %lu bytes a row, fewer than its "
				            @"%llu texels' bytes: answered nil",
				            (unsigned long long)[self rgfxObjId], (unsigned long)bytesPerRow,
				            (unsigned long long)rowBytes);
			}
			return nil;
		}
	} else if (descriptor != nil && descriptor.textureType != MTLTextureType2D) {
		NSString *type = rgfxLinearTypeName(descriptor.textureType);
		if (type != nil)
			rgfxLinearTypeDies(descriptor, type, @"MTLTextureType2D");
		static std::atomic<bool> logged{false};
		if (!logged.exchange(true)) {
			rgfxLogLine(@"RGFX: a texture over buffer 0x%llx of type %lu, not 2D, which the host does "
			            @"not carry: answered nil",
			            (unsigned long long)[self rgfxObjId], (unsigned long)descriptor.textureType);
		}
		return nil;
	}
	// A block-compressed format (BC1 to BC7) answers nil, sending nothing, as the AMD 5300M answers
	// it over a Shared buffer, 64 x 64 and 62 x 64, where the Intel 630 makes it (oracle_texdesc
	// buffer bc1.*): the host's platform makes no texture of one over a buffer's bytes, and one it
	// refused would drop every draw that named it. Both make one of a single block, 2 x 2, which
	// the bundle answers nil for too. On an Apple GPU so does every compressed format it has (PVRTC,
	// EAC, ETC2 and ASTC too), a two-plane format and Unspecialized, as the M3 Max under macOS
	// 27.0.1 answers each (oracle_texdesc Apple buffer, 11-09).
	if (descriptor != nil) {
		const uint64_t f = descriptor.pixelFormat;
		if (rgfxBlockCompressedFormat(f, self.device) ||
		    rgfxDeviceMakesNoTextureOf(descriptor, self.device, /*overBuffer=*/true)) {
			static std::atomic<bool> logged{false};
			if (!logged.exchange(true)) {
				rgfxLogLine(@"RGFX: a texture over buffer 0x%llx of format %llu, which "
				            @"the host does not carry: answered nil",
				            (unsigned long long)[self rgfxObjId], (unsigned long long)f);
			}
			return nil;
		}
	}
	// Texels past the buffer's bytes are made, but where they end past an Apple GPU's allocation
	// of the buffer, whose nil the host's platform answers on every device: the bundle answers
	// nil, logged once, and sends nothing.
	if ([self rgfxPastAllocation:descriptor offset:offset bytesPerRow:bytesPerRow])
		return nil;
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
// Intel GPU make as a copy of the buffer's bytes with every level made from them, taken at its
// first use and again at the first use after its didModifyData (RgfxTexture.mm), which no other
// CPU write reaches and from which no GPU write reaches the buffer
// (docs/graphics-kext/dgpu-oracle.md, Q11), as WindowServer asks under their feature profile (54 x
// 54 and 64 x 60 BGRA8Unorm of 6 and 7 levels, over a 16384-byte Shared buffer at 256 bytes a
// row) and Safari's CoreGraphics for its pictures, and so does the device: TEXTURE_FROM_BUFFER
// carries the levels, and the host's platform makes the same copy by its tiled form (logged once
// a process, with the levels). A type other than 2D the bundle answers nil,
// logged once, and sends nothing, which those GPUs make of every type their descriptor validation
// passes (`oracle_storage DEVICE lintypes tiled:TYPE`, Q11) and whose refusal by the host's
// platform ends the host's process (resource.rs, new_texture).
- (id<MTLTexture>)newTiledTextureWithDescriptor:(MTLTextureDescriptor *)descriptor
                                         offset:(NSUInteger)offset
                                    bytesPerRow:(NSUInteger)bytesPerRow
{
	if (descriptor == nil) {
		return nil;
	}
	// The descriptor's validation first, as by every form (Q11).
	rgfxEndAtInvalidTextureDescriptor(descriptor, self.device);
	if (descriptor.storageMode != MTLStorageModeManaged) {
		static std::atomic<bool> logged{false};
		if (!logged.exchange(true)) {
			rgfxLogLine(@"RGFX: a tiled texture over buffer 0x%llx with descriptor storage %lu, not "
			            @"Managed, which Metal's validation asserts at: answered nil",
			            (unsigned long long)[self rgfxObjId], (unsigned long)descriptor.storageMode);
		}
		return nil;
	}
	if ([self storageMode] != MTLStorageModeShared) {
		static std::atomic<bool> logged{false};
		if (!logged.exchange(true)) {
			rgfxLogLine(@"RGFX: a tiled texture over buffer 0x%llx of storage %lu, not Shared, which "
			            @"Metal's validation asserts at: answered nil",
			            (unsigned long long)[self rgfxObjId], (unsigned long)[self storageMode]);
		}
		return nil;
	}
	if (descriptor.textureType != MTLTextureType2D) {
		static std::atomic<bool> logged{false};
		if (!logged.exchange(true)) {
			rgfxLogLine(@"RGFX: a tiled texture over buffer 0x%llx of type %lu, not 2D, which the host "
			            @"does not carry: answered nil",
			            (unsigned long long)[self rgfxObjId], (unsigned long)descriptor.textureType);
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
				rgfxLogLine(@"RGFX: a tiled texture past buffer 0x%llx's %lu bytes (offset %lu, "
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
	RgfxTexture *texture = [RgfxTexture textureWithDescriptor:shared
	                                                   buffer:self
	                                                   offset:offset
	                                              bytesPerRow:bytesPerRow
	                                                   device:self.device
	                                               serializer:[self serializer]];
	if (descriptor.mipmapLevelCount > 1) {
		static std::atomic<bool> logged{false};
		if (!logged.exchange(true)) {
			rgfxLogLine(@"RGFX: a tiled texture over buffer 0x%llx of %lu mipmap levels (%lu x %lu, "
			            @"format %lu, offset %lu, %lu bytes a row): %@ (logged once a process)",
			            (unsigned long long)[self rgfxObjId],
			            (unsigned long)descriptor.mipmapLevelCount, (unsigned long)descriptor.width,
			            (unsigned long)descriptor.height, (unsigned long)descriptor.pixelFormat,
			            (unsigned long)offset, (unsigned long)bytesPerRow,
			            texture != nil ? @"made" : @"answered nil");
		}
	}
	return texture;
}

// A texture over the buffer's bytes as Metal's linear form makes one, which the host's platform
// makes by the same form (LINEAR_TEXTURE_FROM_BUFFER), carrying `bytesPerImage` as asked. Metal's
// (measured on the M3 Max under macOS 27) is a texture of the buffer's bytes from the offset, rows
// `bytesPerRow` apart, as newTextureWithDescriptor:offset:bytesPerRow: makes one, and takes a 2D
// array too, its slices its rows' bytes rounded up to 128 apart whatever `bytesPerImage`
// (RgfxTexture.mm, rgfxBufferSlicePitch), where the AMD and the Intel GPU end the process at the
// array ("2D linear texture array is not supported.", dgpu-oracle.md, Q11). It ends the process,
// at its validation's failed assertion, for a texture of another type than 2D, 2D array and
// texture buffer: with the same words on the three GPUs for a 1D, 1D array, cube or cube array
// type, which the bundle ends it with, and with the M3's for a 3D or multisample one
// (rgfxLinearTypeDies, Q11). It ends it too at more than one level, which the bundle answers nil,
// logged, as it answers a view Metal ends the process at. One whose texels run past the buffer's
// bytes is made, as by the public form, and one whose texels end past an Apple GPU's allocation of
// the buffer is nil (rgfxPastAllocation:); one off the device's linear texture alignment is nil
// (rgfxAtLinearAlignment:).
- (id<MTLTexture>)newLinearTextureWithDescriptor:(MTLTextureDescriptor *)descriptor
                                          offset:(NSUInteger)offset
                                     bytesPerRow:(NSUInteger)bytesPerRow
                                   bytesPerImage:(NSUInteger)bytesPerImage
{
	if (descriptor == nil) {
		return nil;
	}
	// The descriptor's validation first, as by every form (Q11), and an Apple GPU's end at a depth
	// or stencil format (rgfxEndAtLinearDepthStencil).
	rgfxEndAtInvalidTextureDescriptor(descriptor, self.device);
	rgfxEndAtLinearDepthStencil(descriptor, self.device);
	const MTLTextureType type = descriptor.textureType;
	const bool made = type == MTLTextureType2D || type == MTLTextureType2DArray ||
	                  type == MTLTextureTypeTextureBuffer;
	if (!made && rgfxLinearTypeName(type) != nil) {
		rgfxLinearTypeDies(descriptor, rgfxLinearTypeName(type),
		                   @"MTLTextureType2D or linear MTLTextureType2DArray");
	}
	if (!made || descriptor.mipmapLevelCount != 1) {
		static std::atomic<bool> logged{false};
		if (!logged.exchange(true)) {
			rgfxLogLine(@"RGFX: a linear texture over buffer 0x%llx of type %lu and %lu levels, which "
			            @"Metal's validation ends the process at: answered nil",
			            (unsigned long long)[self rgfxObjId], (unsigned long)type,
			            (unsigned long)descriptor.mipmapLevelCount);
		}
		return nil;
	}
	if ([self rgfxPastAllocation:descriptor offset:offset bytesPerRow:bytesPerRow])
		return nil;
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

// The device makes no buffer over an IOSurface, as neither x86 Mac does (newBufferWithIOSurface:,
// RgfxMetalDevice), so the buffer lies in none, as a buffer made from a length or bytes answers
// none (measured on the M3 Max under macOS 27).
- (IOSurfaceRef)iosurface
{
	return nullptr;
}

// No surface for the Neural Engine: null, as Metal's IOAccel buffer answers, which neither x86
// Mac's buffer overrides (read from code; measured on the AMD 5300M and the Intel 630,
// %spi-answers).
- (IOSurfaceRef)_aneIOSurface
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
