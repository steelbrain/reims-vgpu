/*
 * RgfxEvent: see RgfxEvent.h. Our own ObjC++; no Apple private SPI.
 */

#import "RgfxEvent.h"

#include "RgfxSerializer.hpp"

@implementation RgfxEvent

// The host drops the event as the app lets it go, after any command buffer that names it (each
// keeps it while it runs).
- (void)dealloc
{
	[self rgfxRetireThen:nullptr];
}

+ (instancetype)eventWithDevice:(id<MTLDevice>)device serializer:(RgfxSerializer *)serializer
{
	if (serializer == nullptr) {
		return nil; // an event must have a route to the host
	}
	RgfxCreated created = serializer->createEvent();
	if (serializer->refused(created))
		return nil; // the device never got it
	return [[RgfxEvent alloc] initWithObjId:created.obj_id device:device serializer:serializer];
}

// MTLEvent adds no state or methods beyond `device` and `label`, which
// RgfxResource already carries. The event is otherwise an opaque handle named on
// the wire by `rgfxObjId` when an encoder signals or waits on it.

@end
