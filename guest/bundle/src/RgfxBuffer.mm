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
	// The buffer's CPU-visible backing `contents` vends. In slice 0 this is a
	// plain page-aligned allocation; when the device-region block allocator lands
	// it becomes a mapping into the device region (a Shared buffer's backing the
	// host sees directly), and this malloc goes away. Owned by the wrapper.
	void *_contents;
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
		if (length > 0) {
			// Page-aligned so it can later be replaced by a region mapping.
			if (posix_memalign(&_contents, 0x1000, length) != 0 || _contents == nullptr) {
				return nil;
			}
			std::memset(_contents, 0, length);
		}
	}
	return self;
}

- (void)dealloc
{
	std::free(_contents);
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
	// Managed-buffer flush: push the CPU-written bytes in `range` to the GPU. A
	// Shared buffer over coherent region memory (what slice 0 uses) needs no
	// flush — the host already sees the bytes — so this is a no-op here. The sync
	// slice wires it to a host-visible flush op if/when a Managed buffer over
	// non-coherent memory lands, routing through the serialiser like every other
	// mutating call.
	(void)range;
}

@end
