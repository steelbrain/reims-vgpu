/*
 * RgfxMetalDevice (Sync): see RgfxMetalDevice+Sync.h. Slice 4 — the device's
 * sync-object factories. Our own ObjC++; the only Apple-private name is
 * RgfxMetalDevice's base (RgfxMetalPrivate.h).
 */

#import "RgfxMetalDevice+Sync.h"

#import "RgfxEvent.h"
#import "RgfxFence.h"
#import "RgfxSharedEvent.h"
#import "RgfxUnsupported.h"

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
	RgfxSharedEvent *event = [RgfxSharedEvent sharedEventWithInitialValue:0
	                                                              device:self
	                                                          serializer:[self serializer]];
	// Register for the host's RGFX_EVT_EVENT_SIGNALLED so the value mirror tracks the
	// host counter and notifyListener blocks fire.
	[event rgfxRegisterWithReplyReader:[self replyReader]];
	return event;
}

- (id<MTLSharedEvent>)newSharedEventWithHandle:(MTLSharedEventHandle *)sharedEventHandle
{
	// A handle names another process's event, which the wire has no way to share (no event
	// one context makes and another uses; RgfxSharedEvent's newSharedEventHandle mints none),
	// so the import is declined, and logged.
	(void)sharedEventHandle;
	RGFX_UNSUPPORTED("shared event handles (an event shared across processes)");
	return nil;
}

// The creators with options (measured: _MTLDevice's on macOS 27's x86-64 slice, `@24@0:8q16`
// each). What the options ask for no header states; the AMD 5300M and the Intel 630 answer an
// event of the form without options for every options value, and a shared event for options 0
// and 1, nil for any other (RgfxEvent.h, RgfxSharedEvent.h).
- (id<MTLEvent>)newEventWithOptions:(NSInteger)options
{
	return [RgfxEvent eventWithOptions:options device:self serializer:[self serializer]];
}

- (id<MTLSharedEvent>)newSharedEventWithOptions:(NSInteger)options
{
	RgfxSharedEvent *event = [RgfxSharedEvent sharedEventWithOptions:options
	                                                          device:self
	                                                      serializer:[self serializer]];
	[event rgfxRegisterWithReplyReader:[self replyReader]];
	return event;
}

// A shared event over a Mach port (measured: _MTLDevice's, `@20@0:8I16`): as one from another
// process's handle, above, the device has no host event to name by it.
- (id<MTLSharedEvent>)newSharedEventWithMachPort:(mach_port_t)port
{
	(void)port;
	RGFX_UNSUPPORTED("shared events from another process");
	return nil;
}

@end
