/*
 * RgfxResource: see RgfxResource.h for the wrapper convention this base
 * establishes. Our own ObjC++; no Apple private SPI.
 */

#import "RgfxResource.h"

#import <Metal/Metal.h>

#include "RgfxRetirement.hpp"
#include "RgfxSerializer.hpp"

@implementation RgfxResource {
	__weak id<MTLDevice> _device;
	id<MTLHeap> _heap; // the heap the resource lies in, kept alive
}

- (instancetype)initWithObjId:(uint64_t)objId
                       device:(id<MTLDevice>)device
                   serializer:(RgfxSerializer *)serializer
{
	self = [super init];
	if (self) {
		_objId = objId;
		_device = device;
		_serializer = serializer;
	}
	return self;
}

- (uint64_t)rgfxObjId
{
	return _objId;
}

- (id<MTLDevice>)device
{
	return _device;
}

- (RgfxSerializer *)serializer
{
	return _serializer;
}

- (id<MTLHeap>)heap
{
	return _heap;
}

- (void)rgfxSetHeap:(id<MTLHeap>)heap
{
	_heap = heap;
}

- (void)rgfxRetireThen:(std::function<void()>)after
{
	RgfxSerializer *serializer = _serializer;
	if (serializer == nullptr) {
		if (after)
			after(); // no host object to wait for
		return;
	}
	uint64_t objId = _objId;
	RgfxRetirement::process().retire([serializer, objId, after = std::move(after)] {
		serializer->destroy(objId);
		if (after)
			after();
	});
}

@end
