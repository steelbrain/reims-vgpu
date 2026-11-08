/*
 * RgfxEvent: see RgfxEvent.h. Our own ObjC++; no Apple private SPI.
 */

#import "RgfxEvent.h"

#include <atomic>

#include "RgfxSerializer.hpp"

@implementation RgfxEvent {
	std::atomic<uint64_t> _signalled; // rgfxSignalledValue; only rises
	std::atomic<bool> _enableBarrier;  // enableBarrier; YES from its init
}

// The host drops the event as the app lets it go, after any command buffer that names it (each
// keeps it while it runs).
- (void)dealloc
{
	[self rgfxRetireThen:nullptr];
}

+ (instancetype)eventWithOptions:(NSInteger)options
                         device:(id<MTLDevice>)device
                     serializer:(RgfxSerializer *)serializer
{
	(void)options;
	return [self eventWithDevice:device serializer:serializer];
}

+ (instancetype)eventWithDevice:(id<MTLDevice>)device serializer:(RgfxSerializer *)serializer
{
	if (serializer == nullptr) {
		return nil; // an event must have a route to the host
	}
	RgfxCreated created = serializer->createEvent();
	if (serializer->refused(created))
		return nil; // the device never got it
	RgfxEvent *event = [[RgfxEvent alloc] initWithObjId:created.obj_id device:device serializer:serializer];
	if (event != nil) {
		event->_enableBarrier.store(true, std::memory_order_relaxed);
	}
	return event;
}

- (uint64_t)rgfxSignalledValue
{
	return _signalled.load(std::memory_order_acquire);
}

- (void)rgfxNoteSignalled:(uint64_t)value
{
	uint64_t now = _signalled.load(std::memory_order_relaxed);
	while (now < value &&
	       !_signalled.compare_exchange_weak(now, value, std::memory_order_acq_rel))
		;
}

// --- MTLEventSPI ----------------------------------------------------------------------

// Whether a wait on the event is a barrier: stored and answered, YES from the event's init, as
// Metal's IOAccel event, which both x86 Macs' events are, keeps it (read from code; a fresh one
// answers YES on the AMD 5300M and the Intel 630, %spi-answers). The host's wait orders the work
// after it whatever it holds.
- (BOOL)enableBarrier
{
	return _enableBarrier.load(std::memory_order_relaxed);
}

- (void)setEnableBarrier:(BOOL)enableBarrier
{
	_enableBarrier.store(enableBarrier, std::memory_order_relaxed);
}

// NO: the event's value is never rolled back, as both x86 Macs' events answer (%spi-answers).
- (BOOL)supportsRollback
{
	return NO;
}

// MTLEvent adds no other state or methods beyond `device` and `label`, which RgfxResource
// already carries. The event is otherwise an opaque handle named on the wire by `rgfxObjId` when
// an encoder signals or waits on it.

@end
