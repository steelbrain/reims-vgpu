/*
 * RgfxMetalDevice (Sync): slice 4 of the device shell's category split (see
 * RgfxMetalDevice.h). It implements the MTLDevice sync-object factories —
 * newEvent, newFence, newSharedEvent, newSharedEventWithHandle: — each a thin
 * forwarder that routes a create through the device's serialiser (reached via the
 * `-serializer` accessor) and wraps the result in the matching RgfxResource
 * subclass (RgfxEvent / RgfxFence / RgfxSharedEvent).
 *
 * Like the rest of the device shell this category is COMPILED into the x86_64
 * bundle but NOT linked into the user-space unit test (it is a category on
 * RgfxMetalDevice, which subclasses an Apple-private base that resolves only in the
 * guest). The wrappers it vends are NSObject-based and are what the unit test
 * exercises, driving their factories directly with a test serialiser.
 *
 * The four methods below are already declared by the MTLDevice protocol the device
 * conforms to; re-declaring them here documents that this category owns them. Our
 * own ObjC++; no Apple-private names beyond RgfxMetalDevice's base.
 */

#ifndef RGFX_METAL_DEVICE_SYNC_H_
#define RGFX_METAL_DEVICE_SYNC_H_

#import "RgfxMetalDevice.h"

NS_ASSUME_NONNULL_BEGIN

@interface RgfxMetalDevice (Sync)

- (nullable id<MTLEvent>)newEvent;
- (nullable id<MTLFence>)newFence;
- (nullable id<MTLSharedEvent>)newSharedEvent;
- (nullable id<MTLSharedEvent>)newSharedEventWithHandle:(MTLSharedEventHandle *)sharedEventHandle;
- (nullable id<MTLSharedEvent>)newSharedEventWithMachPort:(mach_port_t)port;

@end

NS_ASSUME_NONNULL_END

#endif // RGFX_METAL_DEVICE_SYNC_H_
