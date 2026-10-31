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
// measured: macOS 26 Metal.framework, required, `@32@0:8Q16Q24`: a texture over a legacy
// IOAccel surface, by its id and buffer (CoreDisplay's display surfaces).
- (nullable id<MTLTexture>)newTextureWithSurface:(NSUInteger)surfaceId buffer:(NSUInteger)buffer;
@end

// measured: macOS 27 Metal.framework — the private command-queue SPI our queue
// wrapper conforms to, with the selectors it overrides.
@protocol MTLCommandQueueSPI <MTLCommandQueue>
// measured: macOS 26 Metal.framework, required: `c24@0:8Q16`, `c28@0:8Q16S24` and `Q16@0:8`.
- (BOOL)setGPUPriority:(NSUInteger)priority;
- (BOOL)setGPUPriority:(NSUInteger)priority offset:(unsigned short)offset;
- (NSUInteger)getGPUPriority;
- (BOOL)setBackgroundGPUPriority:(NSUInteger)priority;
- (BOOL)setBackgroundGPUPriority:(NSUInteger)priority offset:(unsigned short)offset;
- (NSUInteger)getBackgroundGPUPriority;
@end

// measured: macOS 27 Metal.framework — the private command-buffer SPI our command
// buffer wrapper will conform to (the command-buffer/encoders slice), with the selectors it
// overrides.
@protocol MTLCommandBufferSPI <MTLCommandBuffer>
// measured: macOS 26 Metal.framework, optional, `v24@0:8Q16` and `Q16@0:8`.
- (void)setProtectionOptions:(uint64_t)options;
- (uint64_t)protectionOptions;
// measured: macOS 26 Metal.framework, optional, `c16@0:8`.
- (BOOL)commitAndWaitUntilSubmitted;
@end

// measured: macOS 26 Metal.framework — the private resource SPI our resources answer, with the
// selectors they override (RgfxResource).
@protocol MTLResourceSPI <MTLResource>
// measured: required, `v20@0:8i16` and `i16@0:8`.
- (void)setResponsibleProcess:(int)pid;
- (int)responsibleProcess;
// measured: required, `Q16@0:8` (MTLHeapSPI has it too, optional); no setter is declared.
- (uint64_t)protectionOptions;
@end

#endif // RGFX_METAL_PRIVATE_H_
