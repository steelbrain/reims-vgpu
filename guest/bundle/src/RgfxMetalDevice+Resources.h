/*
 * RgfxMetalDevice (Resources): the resource-creation slice of the guest bundle's
 * MTLDevice (the category split is described in RgfxMetalDevice.h). It implements
 * the device's MTLDevice resource creators — buffers, textures, samplers,
 * depth/stencil states and heaps — each a thin forwarder to the matching wrapper
 * factory (RgfxBuffer/RgfxTexture/RgfxHeap), routing through the device's
 * `-serializer`, or, for samplers and depth/stencil states, to the device's
 * RgfxStateCache, which answers equal descriptors with one object.
 *
 * Like the device class itself, this category is COMPILED into the bundle but not
 * linked into the user-space unit test: it names RgfxMetalDevice, which subclasses
 * an Apple-private base. The wrapper factories it forwards to are what the unit
 * test exercises directly.
 *
 * Our own ObjC++; no Apple private SPI beyond RgfxMetalDevice's own.
 */

#ifndef RGFX_METAL_DEVICE_RESOURCES_H_
#define RGFX_METAL_DEVICE_RESOURCES_H_

#import "RgfxMetalDevice.h"

NS_ASSUME_NONNULL_BEGIN

@interface RgfxMetalDevice (Resources)
@end

NS_ASSUME_NONNULL_END

#endif // RGFX_METAL_DEVICE_RESOURCES_H_
