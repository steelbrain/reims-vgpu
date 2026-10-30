/*
 * RgfxIOSurface: see RgfxIOSurface.h. Our own ObjC++.
 */

#import "RgfxIOSurface.h"

#import "RgfxBuffer.h"

#include <algorithm>
#include <cstring>
#include <map>
#include <mutex>
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

} // namespace

/// This process's copy of one surface plane, in a block of the device region.
struct RgfxSurfaceShadow {
	IOSurfaceRef surface = nullptr; // retained
	uint32_t surfaceId = 0;
	uint32_t plane = 0;
	PlaneShape shape = {};
	uint64_t pitch = 0; // the copy's row pitch (kShadowRowAlign multiple)
	RgfxRegion::Block block = {};
	id<MTLBuffer> buffer = nil; // the Shared buffer over the block
	std::mutex lock;            // one sync at a time
	uint32_t seed = 0;          // the surface's seed when the copy last matched it

	~RgfxSurfaceShadow();

	uint8_t *copy() const { return const_cast<uint8_t *>(block.ptr); }

	/// Pages -> copy, under a read lock; records the seed. Caller holds `lock`.
	void upload()
	{
		uint32_t s = 0;
		if (IOSurfaceLock(surface, kIOSurfaceLockReadOnly, &s) != kIOReturnSuccess) {
			return;
		}
		const uint8_t *src = planeBase(surface, plane);
		if (src != nullptr) {
			copyRows(copy(), pitch, src, shape.rowBytes);
		}
		IOSurfaceUnlock(surface, kIOSurfaceLockReadOnly, &s);
		seed = s;
	}

	/// Copy -> pages, under a write lock; records the seed the unlock leaves, so the
	/// next commit does not read our own write back. Caller holds `lock`.
	void writeBack()
	{
		uint32_t s = 0;
		if (IOSurfaceLock(surface, 0, &s) != kIOReturnSuccess) {
			return;
		}
		uint8_t *dst = planeBase(surface, plane);
		if (dst != nullptr) {
			copyRows(dst, shape.rowBytes, copy(), pitch);
		}
		IOSurfaceUnlock(surface, 0, &s);
		seed = s;
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

/// The process's copies by (IOSurfaceID, plane), and the blocks of dead copies by
/// size, which the next copy of that size takes (the region frees nothing).
struct Registry {
	std::mutex lock;
	std::map<std::pair<uint32_t, uint32_t>, std::weak_ptr<RgfxSurfaceShadow>> shadows;
	std::multimap<uint64_t, RgfxRegion::Block> freeBlocks;
};

Registry &registry()
{
	static Registry *r = new Registry(); // never destroyed: copies die at any time
	return *r;
}

} // namespace

RgfxSurfaceShadow::~RgfxSurfaceShadow()
{
	buffer = nil; // the host drops the buffer before the block is reused
	if (block.ok) {
		Registry &r = registry();
		std::lock_guard<std::mutex> g(r.lock);
		r.freeBlocks.emplace(block.size, block);
	}
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
		auto fit = r.freeBlocks.find(size);
		if (fit != r.freeBlocks.end()) {
			shadow->block = fit->second;
			r.freeBlocks.erase(fit);
		} else {
			shadow->block = region->allocate(size);
		}
		if (!shadow->block.ok) {
			return nullptr;
		}
		r.shadows[{surfaceId, plane}] = shadow;
	}
	shadow->buffer = [RgfxBuffer bufferOverHostMemoryWithContents:shadow->copy()
	                                                regionOffset:shadow->block.offset
	                                                      length:shadow->block.size
	                                                     options:MTLResourceStorageModeShared
	                                                 deallocator:nil
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
	if (enc != nullptr && [texture isKindOfClass:[RgfxIOSurfaceTexture class]]) {
		enc->noteSurfaceWrite([(RgfxIOSurfaceTexture *)texture rgfxShadow]);
	}
}

void rgfxIOSurfaceNoteShaderWrite(RgfxEncoder *enc, id texture)
{
	if ([texture isKindOfClass:[RgfxIOSurfaceTexture class]] &&
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
