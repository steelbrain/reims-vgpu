/*
 * RgfxMetalDevice (CommandBuffer): slice 3's device category — the device's
 * `newCommandQueue*` entry points, which stand up an RgfxCommandQueue (routing an
 * RGFX_CMD_QUEUE_CREATE through the serialiser) over a submission ring placed in the
 * device region.
 *
 * This category subclasses nothing new, but a category's methods are registered
 * against the RgfxMetalDevice class symbol, which subclasses the Apple-private device
 * base — so, like RgfxMetalDevice.mm and the other device categories, it is COMPILED
 * into the bundle but excluded from the user-space unit test. The command queue,
 * command buffer and encoders it vends (RgfxCommandObjects.h) subclass only NSObject
 * and are unit-tested there directly.
 *
 * Our own ObjC++; the only Apple-private names are the forward-declarations in
 * RgfxMetalPrivate.h.
 */

#ifndef RGFX_METAL_DEVICE_COMMAND_BUFFER_H_
#define RGFX_METAL_DEVICE_COMMAND_BUFFER_H_

#import "RgfxMetalDevice.h"

NS_ASSUME_NONNULL_BEGIN

@interface RgfxMetalDevice (CommandBuffer)

- (nullable id<MTLCommandQueue>)newCommandQueue;
- (nullable id<MTLCommandQueue>)newCommandQueueWithDescriptor:(MTLCommandQueueDescriptor *)descriptor;
- (nullable id<MTLCommandQueue>)newCommandQueueWithMaxCommandBufferCount:
    (NSUInteger)maxCommandBufferCount;

@end

NS_ASSUME_NONNULL_END

#endif // RGFX_METAL_DEVICE_COMMAND_BUFFER_H_
