/*
 * RgfxSharedEvent: see RgfxSharedEvent.h. Our own ObjC++; no Apple private SPI.
 */

#import "RgfxSharedEvent.h"
#import "RgfxUnsupported.h"
#import "RgfxInterop.h"
#import "RgfxMetalPrivate.h"

#include "RgfxReplyReader.hpp"
#include "RgfxSerializer.hpp"

// One pending notifyListener registration: the value at which the block fires, on the
// listener's queue.
@interface RgfxSharedEventPending : NSObject
@property(nonatomic) uint64_t value;
@property(nonatomic, copy) MTLSharedEventNotificationBlock block;
@property(nonatomic, strong) dispatch_queue_t queue;
@end
@implementation RgfxSharedEventPending
RGFX_LOG_UNRECOGNIZED
@end

@interface RgfxSharedEvent () <RgfxMetalSharedEventInternals>
- (void)rgfxRaiseTo:(uint64_t)value;
@end

@implementation RgfxSharedEvent {
	// The value as this process knows it: the create's initial value, raised by the CPU's sets
	// and by the host's RGFX_EVT_EVENT_SIGNALLED once registered with the reply reader. It only
	// rises, as Metal's does (measured on the M3 Max: a set or a GPU signal to a lower value
	// leaves it). Guarded by `_valueChanged`, with `_pending`.
	uint64_t _signaledValue;
	// Pending notifyListener registrations, fired when the value reaches theirs.
	NSMutableArray<RgfxSharedEventPending *> *_pending;
	// Borrowed: the reader this event is registered with, and the listener token to
	// remove on dealloc.
	RgfxReplyReader *_reader;
	uint64_t _eventToken;
	// Signalled each time the value moves, for waitUntilSignaledValue:timeoutMS:.
	NSCondition *_valueChanged;
	uint64_t _traceId; // globalTraceObjectID
}

+ (instancetype)sharedEventWithOptions:(NSInteger)options
                               device:(id<MTLDevice>)device
                           serializer:(RgfxSerializer *)serializer
{
	if (options != 0 && options != 1)
		return nil;
	return [self sharedEventWithInitialValue:0 device:device serializer:serializer];
}

+ (instancetype)sharedEventWithInitialValue:(uint64_t)initialValue
                                     device:(id<MTLDevice>)device
                                 serializer:(RgfxSerializer *)serializer
{
	if (serializer == nullptr) {
		return nil; // a shared event must have a route to the host
	}
	RgfxCreated created = serializer->createSharedEvent(initialValue);
	if (serializer->refused(created))
		return nil; // the device never got it
	RgfxSharedEvent *ev = [[RgfxSharedEvent alloc] initWithObjId:created.obj_id
	                                                     device:device
	                                                 serializer:serializer];
	if (ev != nil) {
		ev->_traceId = RgfxNextTraceObjectId();
		ev->_signaledValue = initialValue;
		ev->_pending = [NSMutableArray array];
		ev->_valueChanged = [[NSCondition alloc] init];
	}
	return ev;
}

- (void)dealloc
{
	if (_reader != nullptr && _eventToken != 0) {
		_reader->removeEventListener(_eventToken);
	}
	// The host drops the event as the app lets it go, after any command buffer that names it
	// (each keeps it while it runs).
	[self rgfxRetireThen:nullptr];
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
			[strongSelf rgfxRaiseTo:m.value];
		}
	});
}

// The value reached `value` (a command buffer's signal the host posted, or the CPU's set): it
// rises to it where it is higher, and each pending notifyListener block whose value it reached
// fires on its listener's queue, in registration order, with the value it was registered at, as
// Metal's do (measured).
- (void)rgfxRaiseTo:(uint64_t)value
{
	NSMutableArray<RgfxSharedEventPending *> *fired = [NSMutableArray array];
	[_valueChanged lock];
	if (value > _signaledValue)
		_signaledValue = value;
	const uint64_t now = _signaledValue;
	NSMutableArray<RgfxSharedEventPending *> *left = [NSMutableArray array];
	for (RgfxSharedEventPending *p in _pending)
		[(now >= p.value ? fired : left) addObject:p];
	_pending = left;
	[_valueChanged broadcast];
	[_valueChanged unlock];
	for (RgfxSharedEventPending *p in fired) {
		MTLSharedEventNotificationBlock block = p.block;
		const uint64_t at = p.value;
		dispatch_async(p.queue, ^{
			block(self, at);
		});
	}
}

// --- MTLSharedEvent ----------------------------------------------------------

// Its trace id (RgfxInterop.h), taken as it was made.
- (uint64_t)globalTraceObjectID
{
	return _traceId;
}

// The event as an IOSurface shared event, which Core Image asks for: the event itself, as Metal's
// (measured on the M3 Max under macOS 27: the same object, of its own class; a plain MTLEvent
// answers no such member).
- (id)IOSurfaceSharedEvent
{
	return self;
}

- (uint64_t)signaledValue
{
	[_valueChanged lock];
	const uint64_t value = _signaledValue;
	[_valueChanged unlock];
	return value;
}

- (void)setSignaledValue:(uint64_t)signaledValue
{
	// To the host first, whose GPU waits it may release, then here.
	if (_serializer != nullptr)
		_serializer->setSharedEventValue([self rgfxObjId], signaledValue);
	[self rgfxRaiseTo:signaledValue];
}

// Blocks until the value reaches `value` (YES) or `milliseconds` pass (NO), as Metal's does. The
// value moves on the host's EVENT_SIGNALLED and on the CPU's set, and the wait sleeps until one
// moves it. Where the event is registered with a reader over a ring that no thread drives (a
// unit test), the wait takes the ring meanwhile; the device's reader reads none, its signals
// coming from the queues' readers, which the completion thread drives.
- (BOOL)waitUntilSignaledValue:(uint64_t)value timeoutMS:(uint64_t)milliseconds
{
	NSDate *deadline =
	    [NSDate dateWithTimeIntervalSinceNow:static_cast<double>(milliseconds) / 1000.0];
	for (;;) {
		[_valueChanged lock];
		BOOL reached = _signaledValue >= value;
		BOOL pumps = _reader != nullptr && _reader->hasRing() && !_reader->driven();
		if (!reached && !pumps) {
			[_valueChanged waitUntilDate:deadline];
		}
		reached = _signaledValue >= value;
		[_valueChanged unlock];
		if (reached) {
			return YES;
		}
		if ([deadline timeIntervalSinceNow] <= 0) {
			return NO;
		}
		if (pumps && _reader->pump() == 0) {
			[NSThread sleepForTimeInterval:0.001];
		}
	}
}

- (void)notifyListener:(MTLSharedEventListener *)listener
               atValue:(uint64_t)value
                 block:(MTLSharedEventNotificationBlock)block
{
	// `block` runs on the listener's queue once the value reaches `value`, with `value`: at
	// once where it already has, else when a signal or a set reaches it (rgfxRaiseTo:).
	if (block == nil) {
		return;
	}
	dispatch_queue_t queue = listener.dispatchQueue;
	if (queue == nil)
		queue = dispatch_get_global_queue(QOS_CLASS_DEFAULT, 0);
	MTLSharedEventNotificationBlock copied = [block copy];
	[_valueChanged lock];
	const BOOL reached = _signaledValue >= value;
	if (!reached) {
		RgfxSharedEventPending *p = [[RgfxSharedEventPending alloc] init];
		p.value = value;
		p.block = copied;
		p.queue = queue;
		[_pending addObject:p];
	}
	[_valueChanged unlock];
	if (reached) {
		dispatch_async(queue, ^{
			copied(self, value);
		});
	}
}

- (MTLSharedEventHandle *)newSharedEventHandle
{
	// A shared-event handle is the cross-process token another process rebuilds the event from
	// (newSharedEventWithHandle:). The wire has no event one context makes and another uses, so
	// none is minted, and the ask is logged.
	RGFX_UNSUPPORTED("shared event handles (an event shared across processes)");
	return nil;
}

@end
