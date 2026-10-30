/*
 * RgfxFence: see RgfxFence.h. Our own ObjC++; no Apple private SPI.
 */

#import "RgfxFence.h"

#include "RgfxSerializer.hpp"

@implementation RgfxFence

+ (instancetype)fenceWithDevice:(id<MTLDevice>)device serializer:(RgfxSerializer *)serializer
{
	if (serializer == nullptr) {
		return nil; // a fence must have a route to the host
	}
	RgfxCreated created = serializer->createFence();
	return [[RgfxFence alloc] initWithObjId:created.obj_id device:device serializer:serializer];
}

// MTLFence adds no state or methods beyond `device` and `label`, which
// RgfxResource already carries. The fence is otherwise an opaque handle named on
// the wire by `rgfxObjId` when an encoder updates or waits on it.

@end
