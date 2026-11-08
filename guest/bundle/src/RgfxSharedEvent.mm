/*
 * RgfxSharedEvent: see RgfxSharedEvent.h. Our own ObjC++; no Apple private SPI.
 */

#import "RgfxSharedEvent.h"
#import "RgfxUnsupported.h"
#import "RgfxInterop.h"
#import "RgfxMetalPrivate.h"

#include "RgfxReplyReader.hpp"
#include "RgfxSerializer.hpp"

#include <mach/mach.h>

#include <mutex>

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

namespace {
// This process's events by their ports, so an event made from one of them here is its alias:
// weak, each removed as its event goes.
std::mutex gPortsLock;
NSMapTable<NSNumber *, RgfxSharedEvent *> *gPorts;

void rgfxNamePort(mach_port_t port, RgfxSharedEvent *event)
{
	std::lock_guard<std::mutex> hold(gPortsLock);
	if (gPorts == nil)
		gPorts = [NSMapTable strongToWeakObjectsMapTable];
	[gPorts setObject:event forKey:@(port)];
}

RgfxSharedEvent *rgfxEventOfPort(mach_port_t port)
{
	std::lock_guard<std::mutex> hold(gPortsLock);
	return [gPorts objectForKey:@(port)];
}

void rgfxForgetPort(mach_port_t port, RgfxSharedEvent *event)
{
	std::lock_guard<std::mutex> hold(gPortsLock);
	RgfxSharedEvent *named = [gPorts objectForKey:@(port)];
	if (named == nil || named == event)
		[gPorts removeObjectForKey:@(port)];
}

// Whether `port` is a send right of this task's and no more: a port this task receives, a dead
// name and MACH_PORT_NULL name no other process's event.
bool rgfxSendRightOnly(mach_port_t port)
{
	if (!MACH_PORT_VALID(port))
		return false;
	mach_port_type_t type = 0;
	if (mach_port_type(mach_task_self(), port, &type) != KERN_SUCCESS)
		return false;
	return (type & MACH_PORT_TYPE_SEND) != 0 && (type & MACH_PORT_TYPE_RECEIVE) == 0;
}
} // namespace

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
	// Under _portLock: its port (eventPort), MACH_PORT_NULL until asked, the export's or the one
	// it was made from, and whether it was asked (an export is asked once). It holds one send
	// reference of it where `_holdsPortRef`, let go at its dealloc.
	std::mutex _portLock;
	mach_port_t _eventPort;
	BOOL _portAsked;
	BOOL _holdsPortRef;
	// The event this one is this process's alias of: every call is its.
	RgfxSharedEvent *_original;
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

+ (instancetype)sharedEventFromPort:(mach_port_t)port
                              label:(NSString *)label
                             device:(id<MTLDevice>)device
                         serializer:(RgfxSerializer *)serializer
{
	if (!rgfxSendRightOnly(port))
		return nil;
	// One of this process's own: its alias.
	if (RgfxSharedEvent *original = rgfxEventOfPort(port)) {
		RgfxSharedEvent *alias = [[RgfxSharedEvent alloc] initWithObjId:[original rgfxObjId]
		                                                         device:device
		                                                     serializer:serializer];
		if (alias == nil)
			return nil;
		alias->_original = original->_original ? original->_original : original;
		alias->_traceId = RgfxNextTraceObjectId();
		alias->_eventPort = port;
		alias->_portAsked = YES;
		alias->_holdsPortRef =
		    mach_port_mod_refs(mach_task_self(), port, MACH_PORT_RIGHT_SEND, 1) == KERN_SUCCESS;
		alias.label = label;
		return alias;
	}
	if (serializer == nullptr || !serializer->sharesEvents())
		return nil;
	uint64_t value = 0;
	RgfxCreated created = serializer->takeSharedEvent(port, &value);
	if (!created.posted)
		return nil;
	RgfxSharedEvent *ev = [[RgfxSharedEvent alloc] initWithObjId:created.obj_id
	                                                     device:device
	                                                 serializer:serializer];
	if (ev == nil)
		return nil;
	ev->_traceId = RgfxNextTraceObjectId();
	ev->_signaledValue = value;
	ev->_pending = [NSMutableArray array];
	ev->_valueChanged = [[NSCondition alloc] init];
	ev->_eventPort = port;
	ev->_portAsked = YES;
	ev->_holdsPortRef =
	    mach_port_mod_refs(mach_task_self(), port, MACH_PORT_RIGHT_SEND, 1) == KERN_SUCCESS;
	ev.label = label;
	rgfxNamePort(port, ev);
	return ev;
}

- (BOOL)rgfxIsAlias
{
	return _original != nil;
}

- (void)dealloc
{
	{
		std::lock_guard<std::mutex> hold(_portLock);
		if (_eventPort != MACH_PORT_NULL) {
			if (_original == nil)
				rgfxForgetPort(_eventPort, self);
			// Its send reference: an export's last, where no handle or other event holds the
			// port, ends the port, and the kext lets go of the name.
			if (_holdsPortRef)
				mach_port_deallocate(mach_task_self(), _eventPort);
		}
	}
	// An alias names its original's host event, which goes with the original.
	if (_original != nil)
		return;
	if (_reader != nullptr && _eventToken != 0) {
		_reader->removeEventListener(_eventToken);
	}
	// The host drops the event as the app lets it go, after any command buffer that names it
	// (each keeps it while it runs).
	[self rgfxRetireThen:nullptr];
}

- (void)rgfxRegisterWithReplyReader:(RgfxReplyReader *)reader
{
	if (reader == nullptr || _original != nil) {
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
	if (_original != nil)
		return [_original signaledValue];
	[_valueChanged lock];
	const uint64_t value = _signaledValue;
	[_valueChanged unlock];
	return value;
}

- (void)setSignaledValue:(uint64_t)signaledValue
{
	if (_original != nil) {
		[_original setSignaledValue:signaledValue];
		return;
	}
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
	if (_original != nil)
		return [_original waitUntilSignaledValue:value timeoutMS:milliseconds];
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
	if (_original != nil) {
		// The original's value, the event passed to the block this one, as it was asked of.
		__weak RgfxSharedEvent *weakSelf = self;
		MTLSharedEventNotificationBlock asked = [block copy];
		[_original notifyListener:listener
		                  atValue:value
		                    block:^(id<MTLSharedEvent> event, uint64_t at) {
			                    RgfxSharedEvent *strongSelf = weakSelf;
			                    asked(strongSelf ? strongSelf : event, at);
		                    }];
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

// The event's port: a port of the kext's that names its host event, given at the first ask and
// the same name at each (measured: Metal's, oracle_events refs), which the event holds a send
// reference of; the one it was made from for an event made from a port or a handle.
// MACH_PORT_NULL where the kext gave none, or the device carries no event across processes.
- (mach_port_t)eventPort
{
	std::lock_guard<std::mutex> hold(_portLock);
	if (!_portAsked && _serializer != nullptr && _serializer->sharesEvents()) {
		_portAsked = YES;
		_eventPort = _serializer->shareEvent([self rgfxObjId]);
		_holdsPortRef = _eventPort != MACH_PORT_NULL;
		if (_eventPort != MACH_PORT_NULL)
			rgfxNamePort(_eventPort, self);
	}
	return _eventPort;
}

// The trace id of the label, which a handle made of the event asks for and keeps: none, the
// device keeping no trace (measured: a handle made of an event whose labelTraceID answers 0
// answers 0, oracle_events fake).
- (uint64_t)labelTraceID
{
	return 0;
}

// The token another process makes the event from (newSharedEventWithHandle:): Metal's own
// handle over the event's port, which takes a send reference of it of its own and keeps the
// label (measured, oracle_events refs, label). nil where the event has no port.
- (MTLSharedEventHandle *)newSharedEventHandle
{
	if ([self eventPort] == MACH_PORT_NULL) {
		RGFX_UNSUPPORTED("shared event handles where the kext or the host does not carry events "
		                 "across processes");
		return nil;
	}
	return [[MTLSharedEventHandle alloc] initWithSharedEvent:self];
}

// NO: the event's value is never rolled back, as Metal's base shared event answers, which
// neither x86 Mac's overrides (read from code; measured on the AMD 5300M and the Intel 630,
// %spi-answers).
- (BOOL)supportsRollback
{
	return NO;
}

// --- NSSecureCoding: through an NSXPCCoder alone, as Metal's (measured, oracle_events coding:
// a keyed archiver's encode and decode each raise, and the convenience archiver answers
// NSCocoaErrorDomain 4866) ------------------------------------------------------------------

+ (BOOL)supportsSecureCoding
{
	return YES;
}

// Its handle, which carries its port; the far side makes an event of it on its own device.
- (void)encodeWithCoder:(NSCoder *)coder
{
	if (![coder isKindOfClass:[NSXPCCoder class]])
		@throw [NSException exceptionWithName:NSInvalidArgumentException
		                               reason:@"This object may only be encoded by an NSXPCCoder."
		                             userInfo:nil];
	[coder encodeObject:[self newSharedEventHandle] forKey:@"handle"];
}

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wobjc-designated-initializers"
- (instancetype)initWithCoder:(NSCoder *)coder
{
	if (![coder isKindOfClass:[NSXPCCoder class]])
		@throw [NSException exceptionWithName:NSInvalidArgumentException
		                               reason:@"This object may only be decoded by an NSXPCCoder."
		                             userInfo:nil];
	MTLSharedEventHandle *handle = [coder decodeObjectOfClass:[MTLSharedEventHandle class]
	                                                   forKey:@"handle"];
	id<MTLDevice> device = MTLCreateSystemDefaultDevice();
	return handle != nil ? (id)[device newSharedEventWithHandle:handle] : nil;
}
#pragma clang diagnostic pop

@end
