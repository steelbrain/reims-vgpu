/*
 * RgfxMetalDevice (Sync): see RgfxMetalDevice+Sync.h. Slice 4 — the device's
 * sync-object factories. Our own ObjC++; the only Apple-private name is
 * RgfxMetalDevice's base (RgfxMetalPrivate.h).
 */

#import "RgfxMetalDevice+Sync.h"

#import "RgfxEvent.h"
#import "RgfxFence.h"
#import "RgfxSharedEvent.h"

#include "RgfxSerializer.hpp"

@implementation RgfxMetalDevice (Sync)

- (id<MTLEvent>)newEvent
{
	return [RgfxEvent eventWithDevice:self serializer:[self serializer]];
}

- (id<MTLFence>)newFence
{
	return [RgfxFence fenceWithDevice:self serializer:[self serializer]];
}

- (id<MTLSharedEvent>)newSharedEvent
{
	return [RgfxSharedEvent sharedEventWithInitialValue:0 device:self serializer:[self serializer]];
}

- (id<MTLSharedEvent>)newSharedEventWithHandle:(MTLSharedEventHandle *)sharedEventHandle
{
	// Importing a shared event from another process's handle needs the host
	// transport that resolves the handle to a host event; that is a later transport
	// piece (paired with RgfxSharedEvent's -newSharedEventHandle stub). Until it
	// lands this declines the import. FOLLOW-UP: resolve the handle through the host
	// and wrap the event it names.
	(void)sharedEventHandle;
	return nil;
}

@end
