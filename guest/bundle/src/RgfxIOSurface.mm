/*
 * RgfxIOSurface: see RgfxIOSurface.h. Our own ObjC++.
 */

#import "RgfxIOSurface.h"

#import "RgfxBuffer.h"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <fcntl.h>
#include <map>
#include <mutex>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <utility>

#include "RgfxEncoder.hpp"
#include "RgfxRegion.hpp"
#include "RgfxSerializer.hpp"

namespace {

/// The row pitch of a copy: a multiple of every linear-texture row alignment a host
/// GPU asks for.
const uint64_t kShadowRowAlign = 256;

uint64_t alignUp(uint64_t v, uint64_t a)
{
	return (v + a - 1) & ~(a - 1);
}

/// The shape of one plane of a surface, as its CPU mapping lays it out.
struct PlaneShape {
	uint64_t rowBytes;        // bytes from one row of elements to the next
	uint64_t rows;            // rows of elements
	uint64_t bytesPerElement; // bytes of one element
	uint64_t elementWidth;    // pixels across one element
	uint64_t elementHeight;   // pixels down one element
};

/// The shape of plane `plane` of `s`; a surface of no planes has the one plane 0.
/// False when the surface has no such plane.
bool planeShape(IOSurfaceRef s, size_t plane, PlaneShape *out)
{
	size_t planes = IOSurfaceGetPlaneCount(s);
	size_t height;
	if (planes == 0) {
		if (plane != 0) {
			return false;
		}
		out->rowBytes = IOSurfaceGetBytesPerRow(s);
		height = IOSurfaceGetHeight(s);
		out->bytesPerElement = IOSurfaceGetBytesPerElement(s);
		out->elementWidth = IOSurfaceGetElementWidth(s);
		out->elementHeight = IOSurfaceGetElementHeight(s);
	} else {
		if (plane >= planes) {
			return false;
		}
		out->rowBytes = IOSurfaceGetBytesPerRowOfPlane(s, plane);
		height = IOSurfaceGetHeightOfPlane(s, plane);
		out->bytesPerElement = IOSurfaceGetBytesPerElementOfPlane(s, plane);
		out->elementWidth = IOSurfaceGetElementWidthOfPlane(s, plane);
		out->elementHeight = IOSurfaceGetElementHeightOfPlane(s, plane);
	}
	uint64_t eh = std::max<uint64_t>(out->elementHeight, 1);
	out->rows = (height + eh - 1) / eh;
	return out->rowBytes != 0 && out->rows != 0;
}

/// The CPU address of plane `plane` of `s` in this process (null when unmapped).
uint8_t *planeBase(IOSurfaceRef s, size_t plane)
{
	void *p = IOSurfaceGetPlaneCount(s) == 0 ? IOSurfaceGetBaseAddress(s)
	                                         : IOSurfaceGetBaseAddressOfPlane(s, plane);
	return static_cast<uint8_t *>(p);
}

/// The process's copies between surfaces' pages and the GPU's view of them, by direction: a
/// diagnostic, a line in kCopyLog at most every kCopyReportNs while copies run.
struct CopyCount {
	std::atomic<uint64_t> count{0};
	std::atomic<uint64_t> bytes{0};
	std::atomic<uint64_t> ns{0};
};

CopyCount gUploads;    // pages -> the GPU's view
CopyCount gWriteBacks; // the GPU's view -> pages

const uint64_t kCopyReportNs = 10ull * 1000 * 1000 * 1000;
const char *const kCopyLog = "/tmp/rgfx-surface-copies.log";

uint64_t nowNs()
{
	return clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
}

/// Logs the copies since the last report, once kCopyReportNs has passed since it.
void reportCopies(uint64_t now)
{
	static std::mutex lock;
	static uint64_t last = 0, upCount = 0, upBytes = 0, upNs = 0, wbCount = 0, wbBytes = 0,
	                wbNs = 0;
	std::unique_lock<std::mutex> g(lock, std::try_to_lock);
	if (!g.owns_lock()) {
		return;
	}
	if (last == 0) {
		last = now;
		return;
	}
	if (now - last < kCopyReportNs) {
		return;
	}
	uint64_t uc = gUploads.count, ub = gUploads.bytes, un = gUploads.ns;
	uint64_t wc = gWriteBacks.count, wb = gWriteBacks.bytes, wn = gWriteBacks.ns;
	char line[256];
	int n = snprintf(line, sizeof(line),
	                 "RGFX: surface copies, pid %d %s, %.1f s: %llu uploads %.1f MiB %.2f ms, "
	                 "%llu write-backs %.1f MiB %.2f ms\n",
	                 getpid(), getprogname(), (now - last) / 1e9, uc - upCount,
	                 (ub - upBytes) / 1048576.0, (un - upNs) / 1e6, wc - wbCount,
	                 (wb - wbBytes) / 1048576.0, (wn - wbNs) / 1e6);
	int fd = open(kCopyLog, O_WRONLY | O_CREAT | O_APPEND, 0666);
	if (fd >= 0 && n > 0) {
		(void)write(fd, line, std::min(static_cast<size_t>(n), sizeof(line) - 1));
	}
	if (fd >= 0) {
		close(fd);
	}
	last = now;
	upCount = uc, upBytes = ub, upNs = un, wbCount = wc, wbBytes = wb, wbNs = wn;
}

/// Counts a copy of `bytes` that started at `start`.
void countCopy(CopyCount &c, uint64_t bytes, uint64_t start)
{
	uint64_t now = nowNs();
	c.count.fetch_add(1, std::memory_order_relaxed);
	c.bytes.fetch_add(bytes, std::memory_order_relaxed);
	c.ns.fetch_add(now - start, std::memory_order_relaxed);
	reportCopies(now);
}

} // namespace

/// This process's copy of one surface plane, in a block of the device region.
struct RgfxSurfaceShadow {
	IOSurfaceRef surface = nullptr; // retained
	uint32_t surfaceId = 0;
	uint32_t plane = 0;
	PlaneShape shape = {};
	uint64_t pitch = 0; // the copy's row pitch (kShadowRowAlign multiple)
	RgfxRegion::Block block = {};
	RgfxRegion *region = nullptr; // where the block goes back at the copy's end
	id<MTLBuffer> buffer = nil; // the Shared buffer over the block
	std::mutex lock;            // one sync at a time
	uint32_t seed = 0;          // the surface's seed when the copy last matched it

	~RgfxSurfaceShadow();

	uint8_t *copy() const { return const_cast<uint8_t *>(block.ptr); }

	/// Pages -> copy, under a read lock; records the seed. Caller holds `lock`.
	void upload()
	{
		uint32_t s = 0;
		uint64_t start = nowNs();
		if (IOSurfaceLock(surface, kIOSurfaceLockReadOnly, &s) != kIOReturnSuccess) {
			return;
		}
		const uint8_t *src = planeBase(surface, plane);
		if (src != nullptr) {
			copyRows(copy(), pitch, src, shape.rowBytes);
		}
		IOSurfaceUnlock(surface, kIOSurfaceLockReadOnly, &s);
		seed = s;
		countCopy(gUploads, shape.rowBytes * shape.rows, start);
	}

	/// Copy -> pages, under a write lock; records the seed the unlock leaves, so the
	/// next commit does not read our own write back. Caller holds `lock`.
	void writeBack()
	{
		uint32_t s = 0;
		uint64_t start = nowNs();
		if (IOSurfaceLock(surface, 0, &s) != kIOReturnSuccess) {
			return;
		}
		uint8_t *dst = planeBase(surface, plane);
		if (dst != nullptr) {
			copyRows(dst, shape.rowBytes, copy(), pitch);
		}
		IOSurfaceUnlock(surface, 0, &s);
		seed = s;
		countCopy(gWriteBacks, shape.rowBytes * shape.rows, start);
	}

	/// Uploads when the surface's seed moved since the copy last matched it.
	void uploadIfChanged()
	{
		std::lock_guard<std::mutex> g(lock);
		if (IOSurfaceGetSeed(surface) != seed) {
			upload();
		}
	}

	void copyRows(uint8_t *dst, uint64_t dstPitch, const uint8_t *src, uint64_t srcPitch) const
	{
		if (dstPitch == srcPitch) {
			std::memcpy(dst, src, shape.rowBytes * shape.rows);
			return;
		}
		for (uint64_t r = 0; r < shape.rows; r++) {
			std::memcpy(dst + r * dstPitch, src + r * srcPitch, shape.rowBytes);
		}
	}
};

namespace {

/// The process's copies by (IOSurfaceID, plane). A dead copy's block goes back to the
/// region (RgfxRegion::release), which hands it to the accelerator kext to give out again.
struct Registry {
	std::mutex lock;
	std::map<std::pair<uint32_t, uint32_t>, std::weak_ptr<RgfxSurfaceShadow>> shadows;
};

Registry &registry()
{
	static Registry *r = new Registry(); // never destroyed: copies die at any time
	return *r;
}

} // namespace

RgfxSurfaceShadow::~RgfxSurfaceShadow()
{
	// The buffer gives the block back once the host has dropped it (its deallocator); one
	// never made leaves the block to give back here.
	if (buffer != nil)
		buffer = nil;
	else if (block.ok && region != nullptr)
		region->release(block.offset);
	if (surface != nullptr) {
		CFRelease(surface);
	}
}

namespace {

/// The process's copy of `surface`'s plane `plane`, made (and synced) on first use.
std::shared_ptr<RgfxSurfaceShadow> shadowFor(IOSurfaceRef surface, uint32_t plane,
                                             RgfxRegion *region, id<MTLDevice> device,
                                             RgfxSerializer *serializer)
{
	PlaneShape shape;
	if (!planeShape(surface, plane, &shape) || planeBase(surface, plane) == nullptr) {
		return nullptr;
	}
	uint32_t surfaceId = IOSurfaceGetID(surface);
	Registry &r = registry();
	std::shared_ptr<RgfxSurfaceShadow> shadow;
	{
		std::lock_guard<std::mutex> g(r.lock);
		auto it = r.shadows.find({surfaceId, plane});
		if (it != r.shadows.end()) {
			shadow = it->second.lock();
			// A live copy holds its surface, so its id still names that surface.
			if (shadow != nullptr) {
				return shadow;
			}
		}
		shadow = std::make_shared<RgfxSurfaceShadow>();
		shadow->surface = (IOSurfaceRef)CFRetain(surface);
		shadow->surfaceId = surfaceId;
		shadow->plane = plane;
		shadow->shape = shape;
		shadow->pitch = alignUp(shape.rowBytes, kShadowRowAlign);
		uint64_t size = alignUp(shadow->pitch * shape.rows, RgfxRegion::kDefaultBlockAlign);
		shadow->block = region->allocate(size);
		shadow->region = region;
		if (!shadow->block.ok) {
			return nullptr;
		}
		r.shadows[{surfaceId, plane}] = shadow;
	}
	uint64_t offset = shadow->block.offset;
	shadow->buffer = [RgfxBuffer bufferOverHostMemoryWithContents:shadow->copy()
	                                                regionOffset:offset
	                                                      length:shadow->block.size
	                                                     options:MTLResourceStorageModeShared
	                                                 deallocator:^{ region->release(offset); }
	                                                      device:device
	                                                  serializer:serializer];
	if (shadow->buffer == nil) {
		return nullptr;
	}
	{
		std::lock_guard<std::mutex> g(shadow->lock);
		shadow->upload();
	}
	return shadow;
}

/// The wire shape of `d`, as RgfxTexture maps a descriptor.
RgfxTextureDescriptor wireDescriptor(MTLTextureDescriptor *d)
{
	RgfxTextureDescriptor out = {};
	out.pixel_format = static_cast<uint32_t>(d.pixelFormat);
	out.texture_type = static_cast<uint32_t>(d.textureType);
	out.width = static_cast<uint32_t>(d.width);
	out.height = static_cast<uint32_t>(d.height);
	out.depth = static_cast<uint32_t>(d.depth);
	out.mipmap_level_count = static_cast<uint32_t>(d.mipmapLevelCount);
	out.array_length = static_cast<uint32_t>(d.arrayLength);
	out.sample_count = static_cast<uint32_t>(d.sampleCount);
	out.usage = static_cast<uint32_t>(d.usage);
	out.options = static_cast<uint32_t>(d.resourceOptions);
	MTLTextureSwizzleChannels sw = d.swizzle;
	out.swizzle = static_cast<uint32_t>(sw.red) | (static_cast<uint32_t>(sw.green) << 8) |
	              (static_cast<uint32_t>(sw.blue) << 16) | (static_cast<uint32_t>(sw.alpha) << 24);
	return out;
}

} // namespace

@interface RgfxIOSurfaceTexture ()
- (std::shared_ptr<RgfxSurfaceShadow>)rgfxShadow;
@end

@implementation RgfxIOSurfaceTexture {
	std::shared_ptr<RgfxSurfaceShadow> _shadow;
}

+ (instancetype)textureWithDescriptor:(MTLTextureDescriptor *)descriptor
                            iosurface:(IOSurfaceRef)surface
                                plane:(NSUInteger)plane
                               region:(RgfxRegion *)region
                               device:(id<MTLDevice>)device
                           serializer:(RgfxSerializer *)serializer
{
	if (descriptor == nil || surface == nullptr || region == nullptr || serializer == nullptr) {
		return nil;
	}
	std::shared_ptr<RgfxSurfaceShadow> shadow =
	    shadowFor(surface, static_cast<uint32_t>(plane), region, device, serializer);
	if (shadow == nullptr) {
		return nil;
	}
	// The guest sees the shape it asked for; the host's texture lies over the Shared
	// buffer of the copy, so it is Shared.
	RgfxTextureDescriptor desc = wireDescriptor(descriptor);
	RgfxTextureDescriptor hostDesc = desc;
	hostDesc.options = (hostDesc.options & ~static_cast<uint32_t>(MTLResourceStorageModeMask)) |
	                   static_cast<uint32_t>(MTLResourceStorageModeShared);
	RgfxCreated created = serializer->createTextureFromIOSurface(
	    [(RgfxResource *)shadow->buffer rgfxObjId], 0, shadow->pitch, shadow->surfaceId,
	    shadow->plane, hostDesc);
	if (created.resource_id == 0) {
		return nil; // no resource id left: no texture, as Metal answers one it cannot make
	}
	RgfxIOSurfaceTexture *t = [[RgfxIOSurfaceTexture alloc] initWithObjId:created.obj_id
	                                                           gpuAddress:0
	                                                           resourceId:created.resource_id
	                                                           descriptor:desc
	                                                               device:device
	                                                           serializer:serializer];
	if (t != nil) {
		t->_shadow = std::move(shadow);
	}
	return t;
}

- (std::shared_ptr<RgfxSurfaceShadow>)rgfxShadow
{
	return _shadow;
}

- (IOSurfaceRef)iosurface
{
	return _shadow->surface;
}

- (NSUInteger)iosurfacePlane
{
	return _shadow->plane;
}

// The bytes the texture takes: its copy of the plane in the region (SecurityAgent's RenderBox
// asks as it accounts a surface's texture).
- (NSUInteger)allocatedSize
{
	return static_cast<NSUInteger>(_shadow->block.size);
}

// The texture's own CPU access reads and writes the copy, for formats of one element
// per pixel at level 0 / slice 0; a write also goes to the surface's pages.

- (BOOL)rgfxCpuRegion:(MTLRegion)region level:(NSUInteger)level slice:(NSUInteger)slice
{
	const PlaneShape &s = _shadow->shape;
	return level == 0 && slice == 0 && s.elementWidth == 1 && s.elementHeight == 1 &&
	       region.origin.y + region.size.height <= s.rows &&
	       (region.origin.x + region.size.width) * s.bytesPerElement <= s.rowBytes;
}

- (void)getBytes:(void *)pixelBytes
      bytesPerRow:(NSUInteger)bytesPerRow
    bytesPerImage:(NSUInteger)bytesPerImage
       fromRegion:(MTLRegion)region
      mipmapLevel:(NSUInteger)level
            slice:(NSUInteger)slice
{
	if (![self rgfxCpuRegion:region level:level slice:slice]) {
		return;
	}
	_shadow->uploadIfChanged();
	std::lock_guard<std::mutex> g(_shadow->lock);
	uint64_t bpe = _shadow->shape.bytesPerElement;
	for (NSUInteger r = 0; r < region.size.height; r++) {
		std::memcpy(static_cast<uint8_t *>(pixelBytes) + r * bytesPerRow,
		            _shadow->copy() + (region.origin.y + r) * _shadow->pitch + region.origin.x * bpe,
		            region.size.width * bpe);
	}
}

- (void)getBytes:(void *)pixelBytes
     bytesPerRow:(NSUInteger)bytesPerRow
      fromRegion:(MTLRegion)region
     mipmapLevel:(NSUInteger)level
{
	[self getBytes:pixelBytes
	      bytesPerRow:bytesPerRow
	    bytesPerImage:0
	       fromRegion:region
	      mipmapLevel:level
	            slice:0];
}

- (void)replaceRegion:(MTLRegion)region
          mipmapLevel:(NSUInteger)level
                slice:(NSUInteger)slice
            withBytes:(const void *)pixelBytes
          bytesPerRow:(NSUInteger)bytesPerRow
        bytesPerImage:(NSUInteger)bytesPerImage
{
	if (![self rgfxCpuRegion:region level:level slice:slice]) {
		return;
	}
	_shadow->uploadIfChanged();
	std::lock_guard<std::mutex> g(_shadow->lock);
	uint64_t bpe = _shadow->shape.bytesPerElement;
	for (NSUInteger r = 0; r < region.size.height; r++) {
		std::memcpy(_shadow->copy() + (region.origin.y + r) * _shadow->pitch + region.origin.x * bpe,
		            static_cast<const uint8_t *>(pixelBytes) + r * bytesPerRow,
		            region.size.width * bpe);
	}
	_shadow->writeBack();
}

- (void)replaceRegion:(MTLRegion)region
          mipmapLevel:(NSUInteger)level
            withBytes:(const void *)pixelBytes
          bytesPerRow:(NSUInteger)bytesPerRow
{
	[self replaceRegion:region
	        mipmapLevel:level
	              slice:0
	          withBytes:pixelBytes
	        bytesPerRow:bytesPerRow
	      bytesPerImage:0];
}

@end

void rgfxIOSurfaceSyncBeforeCommit(void)
{
	std::vector<std::shared_ptr<RgfxSurfaceShadow>> live;
	{
		Registry &r = registry();
		std::lock_guard<std::mutex> g(r.lock);
		for (auto it = r.shadows.begin(); it != r.shadows.end();) {
			std::shared_ptr<RgfxSurfaceShadow> s = it->second.lock();
			if (s == nullptr) {
				it = r.shadows.erase(it);
				continue;
			}
			live.push_back(std::move(s));
			++it;
		}
	}
	for (const auto &s : live) {
		s->uploadIfChanged();
	}
}

void rgfxIOSurfaceNoteWrite(RgfxEncoder *enc, id texture)
{
	// A view writes the texels of the texture it views, through every view between.
	id root = texture;
	while ([root isKindOfClass:[RgfxTexture class]] && [(RgfxTexture *)root parentTexture] != nil) {
		root = [(RgfxTexture *)root parentTexture];
	}
	if (enc != nullptr && [root isKindOfClass:[RgfxIOSurfaceTexture class]]) {
		enc->noteSurfaceWrite([(RgfxIOSurfaceTexture *)root rgfxShadow]);
	}
}

void rgfxIOSurfaceNoteShaderWrite(RgfxEncoder *enc, id texture)
{
	if ([texture isKindOfClass:[RgfxTexture class]] &&
	    ([(id<MTLTexture>)texture usage] & MTLTextureUsageShaderWrite) != 0) {
		rgfxIOSurfaceNoteWrite(enc, texture);
	}
}

void rgfxIOSurfaceWriteBack(const std::vector<std::shared_ptr<void>> &writes)
{
	std::vector<RgfxSurfaceShadow *> done;
	for (const auto &w : writes) {
		auto *s = static_cast<RgfxSurfaceShadow *>(w.get());
		if (std::find(done.begin(), done.end(), s) != done.end()) {
			continue;
		}
		done.push_back(s);
		std::lock_guard<std::mutex> g(s->lock);
		s->writeBack();
	}
}
