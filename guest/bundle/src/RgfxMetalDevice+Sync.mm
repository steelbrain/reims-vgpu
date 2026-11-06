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

// An event made from a handle, which another process's event (or this one's) made: of the port it
// carries, labelled as the handle is, as Metal's (measured: the AMD 5300M and the Intel 630,
// oracle_events refs and label). nil for a nil handle, as Metal's.
- (id<MTLSharedEvent>)newSharedEventWithHandle:(MTLSharedEventHandle *)sharedEventHandle
{
	if (sharedEventHandle == nil)
		return nil;
	return [self rgfxSharedEventFromPort:[sharedEventHandle eventPort]
	                               label:sharedEventHandle.label];
}

// The event of a port, from a handle or not (RgfxSharedEvent's sharedEventFromPort:), registered
// for the values the host posts it where it is a new one.
- (id<MTLSharedEvent>)rgfxSharedEventFromPort:(mach_port_t)port label:(NSString *)label
{
	RgfxSerializer *serializer = [self serializer];
	if (serializer == nullptr || !serializer->sharesEvents()) {
		RGFX_UNSUPPORTED("shared events from another process where the kext or the host does not "
		                 "carry them");
		return nil;
	}
	RgfxSharedEvent *event = [RgfxSharedEvent sharedEventFromPort:port
	                                                        label:label
	                                                       device:self
	                                                   serializer:serializer];
	if (event != nil && !event.rgfxIsAlias)
		[event rgfxRegisterWithReplyReader:[self replyReader]];
	return event;
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

// A shared event over a Mach port (measured: _MTLDevice's, `@20@0:8I16`), another event's
// eventPort: unlabelled, as Metal's (oracle_events label).
- (id<MTLSharedEvent>)newSharedEventWithMachPort:(mach_port_t)port
{
	return [self rgfxSharedEventFromPort:port label:nil];
}

@end
