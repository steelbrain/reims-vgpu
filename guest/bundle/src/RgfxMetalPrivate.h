/*
 * RgfxMetalPrivate: minimal forward-declarations of the Apple-private Metal
 * interface the reims-graphics guest bundle plugs into. Metal loads our bundle as
 * the guest's MTLDevice (via MetalPluginName/MetalPluginClassName on the
 * accelerator's IOKit personality) and expects a device that subclasses Apple's
 * private IOAccel-backed Metal base class and conforms to <MTLDevice, MTLDeviceSPI>
 * (docs/graphics-kext/design.md, Stage 3; and the Stage-3 Apple-interface notes).
 *
 * This header declares ONLY the private names our shell names: the base class we
 * subclass, the one base initialiser we call `super` on, and the private
 * protocols our classes adopt, each with ONLY the private selectors a slice
 * actually overrides. It is authored by us from a measured enumeration of the
 * macOS 27 Metal.framework runtime; it is NOT a copy of any Apple header, and no
 * Apple header or SPI dump is committed to this repository. Every private
 * declaration is tagged `measured` per docs/testing.md's tag convention.
 *
 * The device class chain the runtime shows is
 *   AGX<gpu>Device -> AGX<family>Device -> IOGPUMetalDevice -> _MTLDevice -> NSObject
 * We subclass IOGPUMetalDevice (the generic IOGPU-backed Metal device base, which
 * supplies the limit tables and the IOAccel resource-pool plumbing) rather than a
 * silicon-specific AGX class, and answer identity/limits/features ourselves from
 * the host's caps blob.
 *
 * Because these base classes and protocols live in Metal.framework and the IOGPU
 * stack, the symbols resolve only at load time in the guest: the bundle links with
 * `-undefined dynamic_lookup`, and the device class is excluded from the user-space
 * unit test (which needs the base at runtime). The NSObject-based wrappers
 * (RgfxResource and its subclasses) do not depend on anything here, so they
 * instantiate and are unit-tested on the Mac.
 */

#ifndef RGFX_METAL_PRIVATE_H_
#define RGFX_METAL_PRIVATE_H_

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <mach/mach_port.h>

// measured: macOS 27 Metal.framework — the root of the private Metal device class
// chain (below NSObject). We never override its methods; we only need the symbol
// so IOGPUMetalDevice can name it as its superclass.
@interface _MTLDevice : NSObject
@end

// measured: macOS 27 Metal.framework — the generic IOGPU-backed Metal device base
// our device subclasses. We declare only `-initWithAcceleratorPort:`, the entry
// point Metal calls and the one base method our `-init` chains to with `super`.
// (Selector `-initWithAcceleratorPort:` type-encoding `@20@0:8I16`: returns id,
// takes a mach_port_t.)
@interface IOGPUMetalDevice : _MTLDevice
- (nullable instancetype)initWithAcceleratorPort:(mach_port_t)port;
@end

// measured: macOS 27 Metal.framework — the private device SPI our MTLDevice must
// conform to. Metal's CoreImage/GL/OpenCL/WindowServer layers call ~439 selectors
// on it; this forward-declaration lists ONLY the ones a slice overrides, answered
// from the host caps blob. A later slice that overrides more SPI adds its
// selectors here, each tagged measured. (All four below are NSUInteger getters,
// type-encoding `Q16@0:8`; `-supportsPrimitiveType:` is `B24@0:8Q16`.)
@protocol MTLDeviceSPI <MTLDevice>
- (NSUInteger)maxVertexSamplers;
- (NSUInteger)maxFragmentTextures;
- (NSUInteger)maxComputeTextures;
- (NSUInteger)maxVertexAmplificationCount;
- (BOOL)supportsPrimitiveType:(NSUInteger)primitiveType;
@end

// measured: macOS 27 Metal.framework — the private command-queue SPI our queue
// wrapper will conform to (the command-buffer/encoders slice). The shell overrides
// none of its selectors yet, so this is a bare forward-declaration establishing
// the adoption; that slice adds the selectors it overrides, each tagged measured.
@protocol MTLCommandQueueSPI <MTLCommandQueue>
@end

// measured: macOS 27 Metal.framework — the private command-buffer SPI our command
// buffer wrapper will conform to (the command-buffer/encoders slice). As above, a
// bare forward-declaration until that slice overrides selectors here.
@protocol MTLCommandBufferSPI <MTLCommandBuffer>
@end

#endif // RGFX_METAL_PRIVATE_H_
