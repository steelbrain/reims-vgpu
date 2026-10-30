/*
 * RgfxSharedEvent: see RgfxSharedEvent.h. Our own ObjC++; no Apple private SPI.
 */

#import "RgfxSharedEvent.h"

#include "RgfxReplyReader.hpp"
#include "RgfxSerializer.hpp"

// One pending notifyListener registration: the value at which the block fires.
@interface RgfxSharedEventPending : NSObject
@property(nonatomic) uint64_t value;
@property(nonatomic, copy) MTLSharedEventNotificationBlock block;
@end
@implementation RgfxSharedEventPending
@end

@interface RgfxSharedEvent ()
- (void)rgfxEventSignalled:(const RgfxEventSignalled *)msg;
@end

@implementation RgfxSharedEvent {
	// Local mirror of the host-side counter. Seeded from the create's initial value
	// and updated from RGFX_EVT_EVENT_SIGNALLED once registered with the reply reader.
	uint64_t _signaledValue;
	// Pending notifyListener registrations, fired when the value reaches theirs.
	NSMutableArray<RgfxSharedEventPending *> *_pending;
	// Borrowed: the reader this event is registered with, and the listener token to
	// remove on dealloc.
	RgfxReplyReader *_reader;
	uint64_t _eventToken;
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
		ev->_pending = [NSMutableArray array];
	}
	return ev;
}

- (void)dealloc
{
	if (_reader != nullptr && _eventToken != 0) {
		_reader->removeEventListener(_eventToken);
	}
}

- (void)rgfxRegisterWithReplyReader:(RgfxReplyReader *)reader
{
	if (reader == nullptr) {
		return;
	}
	_reader = reader;
	__weak RgfxSharedEvent *weakSelf = self;
	_eventToken = reader->addEventListener([self rgfxObjId], [weakSelf](const RgfxEventSignalled &m) {
		RgfxSharedEvent *strongSelf = weakSelf;
		if (strongSelf != nil) {
			[strongSelf rgfxEventSignalled:&m];
		}
	});
}

// The host signalled this event to `msg->value`: update the mirror and fire every
// pending notifyListener block whose value has been reached, in registration order.
- (void)rgfxEventSignalled:(const RgfxEventSignalled *)msg
{
	_signaledValue = msg->value;
	NSMutableArray<RgfxSharedEventPending *> *fired = [NSMutableArray array];
	for (RgfxSharedEventPending *p in _pending) {
		if (msg->value >= p.value) {
			[fired addObject:p];
		}
	}
	for (RgfxSharedEventPending *p in fired) {
		[_pending removeObjectIdenticalTo:p];
		p.block(self, msg->value);
	}
}

// --- MTLSharedEvent ----------------------------------------------------------

- (uint64_t)signaledValue
{
	// The host holds the authoritative counter; this returns the local mirror, which
	// the reply reader updates from RGFX_EVT_EVENT_SIGNALLED once registered.
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
	// A CPU completion callback fires when the event's value reaches `value`. If the
	// mirror has already reached it, fire now; otherwise keep the (value, block) until
	// an RGFX_EVT_EVENT_SIGNALLED through the reply reader reaches it (rgfxEventSignalled:).
	// `listener` names the dispatch queue Metal would fire the block on; we fire
	// synchronously (a documented simplification until the listener's queue is wired).
	(void)listener;
	if (block == nil) {
		return;
	}
	if (_signaledValue >= value) {
		block(self, _signaledValue);
		return;
	}
	RgfxSharedEventPending *p = [[RgfxSharedEventPending alloc] init];
	p.value = value;
	p.block = block;
	[_pending addObject:p];
}

- (MTLSharedEventHandle *)newSharedEventHandle
{
	// A shared-event handle is the cross-process token used to rebuild the event in
	// another process; minting one needs the host transport that backs it. Until
	// that lands this returns nil (a documented stub, a later transport piece).
	return nil;
}

@end
