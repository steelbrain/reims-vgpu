/*
 * RgfxEvent: the reims-graphics bundle's MTLEvent wrapper (sync slice). An event
 * is an opaque GPU-timeline handle Metal hands to command encoders, which name it
 * on the wire by its `rgfxObjId` (the command-buffer/encoders slice extracts it);
 * the guest side carries nothing but that id and the base `label`.
 *
 * Created by `+eventWithDevice:serializer:`, which routes an RGFX_CMD_EVENT_CREATE
 * through the serialiser (picking the objId) and wraps the result, exactly as
 * RgfxBuffer does for buffers. The device category `newEvent` (RgfxMetalDevice+
 * Sync) is a thin forwarder to this factory.
 *
 * Our own ObjC++: it conforms to the real MTLEvent protocol but subclasses only
 * NSObject (via RgfxResource), so it instantiates and is unit-tested on the Mac.
 */

#ifndef RGFX_EVENT_H_
#define RGFX_EVENT_H_

#import <Metal/Metal.h>

#import "RgfxResource.h"

#ifdef __cplusplus
class RgfxSerializer;
#endif

NS_ASSUME_NONNULL_BEGIN

@interface RgfxEvent : RgfxResource <MTLEvent>

/// Creates an event, serialising an RGFX_CMD_EVENT_CREATE through `serializer`
/// (which picks the objId). `device` is the owning shell (nil in a unit test).
/// Returns nil if `serializer` is null (an event must have a route to the host).
+ (nullable instancetype)eventWithDevice:(nullable id<MTLDevice>)device
                              serializer:(nullable RgfxSerializer *)serializer;

/// The highest value a command buffer of this process signalled the event to that the host has
/// finished, 0 for none: only command buffers signal a plain event, and only its process's. The
/// post of a buffer that waits on the event reads it (RgfxCmdBuffer's rgfxPost).
- (uint64_t)rgfxSignalledValue;

/// The host finished a command buffer that signalled the event to `value`.
- (void)rgfxNoteSignalled:(uint64_t)value;

/// The event the device's `newEventWithOptions:` (SPI) answers: one of the form without options,
/// whatever `options` is, as the AMD 5300M and the Intel 630 answer for options 0 to 16 (measured
/// under macOS 26.7.1, oracle_bundle events: the class newEvent answers, every time).
+ (nullable instancetype)eventWithOptions:(NSInteger)options
                                   device:(nullable id<MTLDevice>)device
                               serializer:(nullable RgfxSerializer *)serializer;

@end

NS_ASSUME_NONNULL_END

#endif // RGFX_EVENT_H_
