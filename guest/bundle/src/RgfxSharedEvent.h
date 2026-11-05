/*
 * RgfxSharedEvent: the reims-graphics bundle's MTLSharedEvent wrapper (sync slice).
 * A shared event is a CPU/GPU-shared counter: encoders signal and wait on it on the
 * GPU timeline, and the CPU reads/sets its `signaledValue` and registers listeners
 * that fire when it reaches a value. Like RgfxEvent it carries its `rgfxObjId` (the
 * encoders slice names it on the wire) plus the base `label`.
 *
 * Created by `+sharedEventWithInitialValue:device:serializer:`, which routes an
 * RGFX_CMD_SHARED_EVENT_CREATE (carrying the initial value) through the serialiser
 * and wraps the result. The device category `newSharedEvent` (RgfxMetalDevice+Sync)
 * forwards to it with an initial value of 0.
 *
 * THE VALUE LIVES HOST-SIDE. The host's event is the one command buffers wait on and
 * signal (RGFX_OP_WAIT_EVENT / RGFX_OP_SIGNAL_EVENT). This process keeps the value as it
 * knows it: a CPU set goes to the host (RGFX_CMD_SHARED_EVENT_SET_VALUE) and raises it
 * here, and a command buffer's signal raises it once the host posts RGFX_EVT_EVENT_SIGNALLED:
 * as the host's event reaches the value where the buffer has work after the signal, else when
 * the buffer finishes, and ahead of its completion. It only rises, as Metal's does. The
 * notifyListener blocks fire on their listener's queue with the value they were registered
 * at, as Metal's do.
 *
 * Our own ObjC++: it conforms to the real MTLSharedEvent protocol but subclasses
 * only NSObject (via RgfxResource), so it instantiates and is unit-tested on the
 * Mac.
 */

#ifndef RGFX_SHARED_EVENT_H_
#define RGFX_SHARED_EVENT_H_

#import <Metal/Metal.h>

#import "RgfxResource.h"

#ifdef __cplusplus
class RgfxSerializer;
class RgfxReplyReader;
#endif

NS_ASSUME_NONNULL_BEGIN

@interface RgfxSharedEvent : RgfxResource <MTLSharedEvent>

/// Creates a shared event starting at `initialValue`, serialising an
/// RGFX_CMD_SHARED_EVENT_CREATE through `serializer` (which picks the objId) and
/// seeding the value. `device` is the owning shell (nil in a unit
/// test). Returns nil if `serializer` is null (a shared event must have a route).
+ (nullable instancetype)sharedEventWithInitialValue:(uint64_t)initialValue
                                             device:(nullable id<MTLDevice>)device
                                         serializer:(nullable RgfxSerializer *)serializer;

/// The shared event the device's `newSharedEventWithOptions:` (SPI) answers: one of the form
/// without options for options 0 and 1, and nil for any other, as the AMD 5300M and the Intel 630
/// answer (measured under macOS 26.7.1, oracle_bundle events: options 2, 3, 4, 8 and 16 nil).
/// What the options mean no header states.
+ (nullable instancetype)sharedEventWithOptions:(NSInteger)options
                                         device:(nullable id<MTLDevice>)device
                                     serializer:(nullable RgfxSerializer *)serializer;

/// Registers with `reader` for the host's RGFX_EVT_EVENT_SIGNALLED for this event's
/// obj_id: each one raises the value and fires the notifyListener blocks it reached.
/// The listener is removed on dealloc. The Sync device slice calls this after the create; a unit
/// test calls it directly before feeding the signal.
- (void)rgfxRegisterWithReplyReader:(RgfxReplyReader *)reader;

@end

NS_ASSUME_NONNULL_END

#endif // RGFX_SHARED_EVENT_H_
