/*
 * RgfxSharedEvent: see RgfxSharedEvent.h. Our own ObjC++; no Apple private SPI.
 */

#import "RgfxSharedEvent.h"

#include "RgfxSerializer.hpp"

@implementation RgfxSharedEvent {
	// Local mirror of the host-side counter. Seeded from the create's initial
	// value; the host->guest sync that updates it from RGFX_EVT_EVENT_SIGNALLED is
	// a later transport piece (see the header).
	uint64_t _signaledValue;
}

+ (instancetype)sharedEventWithInitialValue:(uint64_t)initialValue
                                     device:(id<MTLDevice>)device
                                 serializer:(RgfxSerializer *)serializer
{
	if (serializer == nullptr) {
		return nil; // a shared event must have a route to the host
	}
	RgfxCreated created = serializer->createSharedEvent(initialValue);
	RgfxSharedEvent *ev = [[RgfxSharedEvent alloc] initWithObjId:created.obj_id
	                                                     device:device
	                                                 serializer:serializer];
	if (ev != nil) {
		ev->_signaledValue = initialValue;
	}
	return ev;
}

// --- MTLSharedEvent ----------------------------------------------------------

- (uint64_t)signaledValue
{
	// The host holds the authoritative counter; this returns the local mirror
	// until the EVENT_SIGNALLED reader lands (see the header).
	return _signaledValue;
}

- (void)setSignaledValue:(uint64_t)signaledValue
{
	// A CPU-side set of the counter has no wire op of its own in this slice: the
	// lifecycle serialiser carries SHARED_EVENT_CREATE only, and the GPU-timeline
	// signal is an encoder op (the command-buffer/encoders slice), not a device
	// call. FOLLOW-UP: when the host grows a shared-event SET op (and/or the reply
	// ring reports the host's value back), route the set through the serialiser and
	// reconcile the mirror. For now this updates only the local mirror.
	_signaledValue = signaledValue;
}

- (void)notifyListener:(MTLSharedEventListener *)listener
               atValue:(uint64_t)value
                 block:(MTLSharedEventNotificationBlock)block
{
	// Registering a CPU completion callback needs the host->guest value stream
	// (RGFX_EVT_EVENT_SIGNALLED) to know when `value` is reached. That reader is a
	// later transport piece, so this is a documented stub: no listener is recorded
	// and no block is scheduled yet. FOLLOW-UP: keep the (listener, value, block)
	// tuples and fire them from the EVENT_SIGNALLED handler.
	(void)listener;
	(void)value;
	(void)block;
}

- (MTLSharedEventHandle *)newSharedEventHandle
{
	// A shared-event handle is the cross-process token used to rebuild the event in
	// another process; minting one needs the host transport that backs it. Until
	// that lands this returns nil (a documented stub, a later transport piece).
	return nil;
}

@end
