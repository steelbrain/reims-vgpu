/*
 * RgfxBuffer: the reims-graphics bundle's MTLBuffer wrapper, and the worked
 * example of the wrapper convention (RgfxResource.h). The resource slice adds
 * RgfxTexture/RgfxSampler/RgfxHeap/… the same way; this is the one concrete
 * wrapper slice 0 ships so the pattern is pinned and unit-tested.
 *
 * A buffer is created by `+bufferWithLength:options:device:serializer:`, which
 * routes an `RGFX_CMD_BUFFER_CREATE` through the serialiser (picking the objId and
 * GPU address) and wraps the result. The device category `newBufferWithLength:
 * options:` (resource slice) is a thin forwarder to this factory. The getters
 * answer from local state: `contents` hands back the buffer's CPU-visible backing,
 * `length`/`gpuAddress`/`resourceOptions` the stored values.
 *
 * Our own ObjC++: it conforms to the real `MTLBuffer` protocol but subclasses only
 * NSObject (via RgfxResource), so it instantiates and is unit-tested on the Mac.
 */

#ifndef RGFX_BUFFER_H_
#define RGFX_BUFFER_H_

#import <Metal/Metal.h>

#import "RgfxResource.h"

#ifdef __cplusplus
#include <memory>

class RgfxMemoryShadow;
class RgfxRegion;
class RgfxSerializer;
class RgfxSlabs;
#endif

NS_ASSUME_NONNULL_BEGIN

@interface RgfxBuffer : RgfxResource <MTLBuffer>

/// Creates a device buffer of `length` bytes with `options`, serialising an
/// `RGFX_CMD_BUFFER_CREATE` through `serializer` (which picks the objId and GPU
/// address). `device` is the owning shell (nil in a unit test). A Shared or Managed
/// one has contents of its own, zeroed; a Private one has none (`contents` answers
/// NULL). Returns nil if `serializer` is null (a buffer must have a route) or the
/// backing cannot be allocated.
+ (nullable instancetype)bufferWithLength:(NSUInteger)length
                                  options:(MTLResourceOptions)options
                                   device:(nullable id<MTLDevice>)device
                               serializer:(nullable RgfxSerializer *)serializer;

/// Creates a buffer backed by a block of the device region at `regionOffset`,
/// serialising an `RGFX_CMD_BUFFER_OVER_HOST_MEMORY` through `serializer` (which picks
/// the objId and GPU address). `contents` is the CPU pointer the buffer vends (the
/// region block, or the caller's pages for a no-copy buffer); it is NOT owned here —
/// `deallocator` (may be nil) is called on dealloc instead of freeing it. The host
/// sees the bytes in place, so no copy is made. Returns nil if `serializer` is null.
+ (nullable instancetype)bufferOverHostMemoryWithContents:(void *)contents
                                             regionOffset:(uint64_t)regionOffset
                                                   length:(NSUInteger)length
                                                  options:(MTLResourceOptions)options
                                              deallocator:(nullable void (^)(void))deallocator
                                                   device:(nullable id<MTLDevice>)device
                                               serializer:(nullable RgfxSerializer *)serializer;

/// Creates a Shared buffer of `length` bytes over an IOSurface's own pages, those the shared
/// block at `blockOffset` (its holders' part `blockSize` bytes) has the host use in place,
/// serialising an `RGFX_CMD_BUFFER_OVER_SURFACE_PAGES` through `serializer`. `contents` is the
/// surface's base in this process; it is NOT owned here, and `deallocator` (may be nil) is
/// called on dealloc. The buffer has no GPU address: the create names none. Returns nil if
/// `serializer` is null.
+ (nullable instancetype)bufferOverSurfacePagesWithContents:(void *)contents
                                               blockOffset:(uint64_t)blockOffset
                                                 blockSize:(uint64_t)blockSize
                                                    length:(NSUInteger)length
                                                   options:(MTLResourceOptions)options
                                               deallocator:(nullable void (^)(void))deallocator
                                                    device:(nullable id<MTLDevice>)device
                                                serializer:(nullable RgfxSerializer *)serializer;

#ifdef __cplusplus
/// The buffer `newBufferWithLength:options:` (`bytes` null) and
/// `newBufferWithBytes:length:options:` make: a Shared one in a range of a slab of `slabs`, which
/// the host sees in place, a Managed one with its CPU copy there (RgfxSlabs.hpp), and a Private
/// one the GPU's alone, its contents NULL. The range is whole 4 KiB pages, which allocatedSize
/// answers, as both x86-64 Macs' GPUs do; its bytes are `length` bytes from `bytes`, then zeros.
///
/// Nil where Metal makes none (measured on the M3 Max, macOS 27, in every storage mode, with
/// bytes and without): for a length of 0 and for one past `maxLength` (the device's
/// maxBufferLength; 0 holds none back). Nil too where `slabs` has no room for a Shared or
/// Managed one, as Metal answers a buffer it cannot place, never memory the GPU does not see;
/// and for bytes with a mode whose contents the CPU never reaches (Private), which Metal's
/// validation refuses.
+ (nullable instancetype)bufferForDeviceWithLength:(NSUInteger)length
                                           options:(MTLResourceOptions)options
                                             bytes:(nullable const void *)bytes
                                         maxLength:(NSUInteger)maxLength
                                             slabs:(nullable RgfxSlabs *)slabs
                                            device:(nullable id<MTLDevice>)device
                                        serializer:(nullable RgfxSerializer *)serializer;

/// The buffer `newBufferWithBytesNoCopy:length:options:deallocator:` makes over the caller's
/// `length` bytes at `pointer`, whose `contents` is `pointer`, with `deallocator` (may be nil)
/// called with them once the buffer is gone from the host. Bytes in `region` (a block of the
/// process's) the host sees in place. Shared bytes the process may write, where `region` has
/// pages blocks (RgfxRegion::acquirePages), the host sees in place too: the kext wires their
/// pages for the buffer's life, and the buffer's GPU address keeps the pointer's offset in its
/// page. Any others, and those the kext or the host makes no pages block of, get a block of
/// their own, the GPU's copy, synced with them at the command buffers that use the buffer
/// (RgfxMemoryShadow.hpp). A null `pointer` is a buffer with contents of its own, zeroed, whose
/// deallocator gets null.
///
/// What Metal makes, as measured (M3 Max, macOS 27, 16 KiB and, under Rosetta, 4 KiB pages):
/// any pointer and length, at any alignment, over memory the process maps readable: malloc'd,
/// vm_allocate'd, read-only, or a shared or private file mapping, and two buffers over the same
/// or overlapping bytes. Nil for a length of 0 and for a range not wholly mapped; Private
/// storage fails an assertion, where this answers nil. Nil also where `region` has no room.
+ (nullable instancetype)bufferWithBytesNoCopy:(nullable void *)pointer
                                        length:(NSUInteger)length
                                       options:(MTLResourceOptions)options
                                   deallocator:(nullable void (^)(void *_Nullable, NSUInteger))deallocator
                                        region:(nullable RgfxRegion *)region
                                        device:(nullable id<MTLDevice>)device
                                    serializer:(nullable RgfxSerializer *)serializer;

/// The GPU's copy of app memory the buffer lies over (a no-copy buffer's outside the device
/// region), null for any other.
- (std::shared_ptr<RgfxMemoryShadow>)rgfxShadow;
#endif

/// Designated initialiser: wraps an already-created host buffer. `objId` and
/// `gpuAddress` are the guest-picked values the create returned; `length` and
/// `options` are the buffer's. Allocates the CPU-visible backing `contents` vends,
/// which this owns and frees, for a Shared or Managed buffer; a Private one has none.
- (nullable instancetype)initWithObjId:(uint64_t)objId
                            gpuAddress:(uint64_t)gpuAddress
                                length:(NSUInteger)length
                               options:(MTLResourceOptions)options
                                device:(nullable id<MTLDevice>)device
                            serializer:(nullable RgfxSerializer *)serializer
    NS_DESIGNATED_INITIALIZER;

/// Designated initialiser for a region-backed buffer: `contents` is the CPU pointer
/// the buffer vends, not owned here; `deallocator` (may be nil) is called on dealloc.
- (nullable instancetype)initWithObjId:(uint64_t)objId
                            gpuAddress:(uint64_t)gpuAddress
                                length:(NSUInteger)length
                               options:(MTLResourceOptions)options
                              contents:(nullable void *)contents
                           deallocator:(nullable void (^)(void))deallocator
                                device:(nullable id<MTLDevice>)device
                            serializer:(nullable RgfxSerializer *)serializer
    NS_DESIGNATED_INITIALIZER;

- (instancetype)initWithObjId:(uint64_t)objId
                       device:(nullable id<MTLDevice>)device
                   serializer:(nullable RgfxSerializer *)serializer NS_UNAVAILABLE;

@end

NS_ASSUME_NONNULL_END

#ifdef __cplusplus
/// The copy of app memory `resource` reaches: a no-copy buffer's, or that of the buffer under
/// a texture, through its views; null for any other (or nil).
std::shared_ptr<RgfxMemoryShadow> rgfxShadowOf(id _Nullable resource);
#endif

#endif // RGFX_BUFFER_H_
