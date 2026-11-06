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

// measured: macOS 27 and 26.6.1 (the x86-64 guest, on our device) Metal.framework — the base
// device class's own selectors the Core Image link route calls on `super`, implemented only by
// _MTLDevice. `-newLibraryWithData:error:` (`@32@0:8@16^@24`) makes Metal's own library over
// the bytes. The two links (`@40@0:8@16r^{?=BQ^{?}}24^@32`) link Core Image's extern functions on
// the CPU into a library of one kernel; their second argument is Core Image's, passed through
// unread, so it is declared opaque.
NS_ASSUME_NONNULL_BEGIN
@interface IOGPUMetalDevice (RgfxImageFilterLink)
- (nullable id<MTLLibrary>)newLibraryWithData:(dispatch_data_t)data
                                        error:(NSError *_Nullable *_Nullable)error;
- (nullable id<MTLLibrary>)newLibraryWithImageFilterFunctionsSPI:(NSArray<id<MTLFunction>> *)functions
                                         imageFilterFunctionInfo:(const void *)info
                                                           error:(NSError *_Nullable *_Nullable)error;
- (nullable id<MTLLibrary>)newLibraryWithCIFilters:(NSArray<id<MTLFunction>> *)functions
                           imageFilterFunctionInfo:(const void *)info
                                             error:(NSError *_Nullable *_Nullable)error;
// measured: macOS 27 and 26.6.1, `@32@0:8@16^@24`, implemented only by _MTLDevice; on macOS 27 it
// stitches on the CPU, reading each function's name and bitcode hash.
- (nullable id<MTLLibrary>)newLibraryWithStitchedDescriptor:(MTLStitchedLibraryDescriptor *)descriptor
                                                      error:(NSError *_Nullable *_Nullable)error;
// measured: macOS 27 and 26.6.1, `@40@0:8@16@24^@32`, _MTLDevice's: a stitched library of a DAG's
// text (Metal's parser reads it) over `functions`, which may not be empty (Metal asserts).
- (nullable id<MTLLibrary>)newLibraryWithDAG:(id)dag
                                   functions:(NSArray<id<MTLFunction>> *)functions
                                       error:(NSError *_Nullable *_Nullable)error;
// measured: macOS 27 and 26.7.1, `@40@0:8@16@24^@32` and `@48@0:8@16@24@32^@40`, _MTLDevice's: a
// library of source compiled by Metal's frontend; the first calls the second on the device with
// no task, which reaches Metal's compiler service through _MTLDevice alone.
- (nullable id<MTLLibrary>)newLibraryWithSource:(NSString *)source
                                        options:(nullable MTLCompileOptions *)options
                                          error:(NSError *_Nullable *_Nullable)error;
- (nullable id<MTLLibrary>)newLibraryWithSource:(NSString *)source
                                        options:(nullable MTLCompileOptions *)options
                                   compilerTask:(nullable id)compilerTask
                                          error:(NSError *_Nullable *_Nullable)error;
// measured: macOS 27 and 26.7.1, `@32@0:8@16^@24`, `@32@0:8@16^@24` and `@40@0:8@16@24^@32`,
// _MTLDevice's: a library of the visible and stitchable functions given (Metal asserts at an empty
// array), of a Metal package's URL, and of an MPS graph package's.
- (nullable id<MTLLibrary>)newLibraryWithFunctionArray:(NSArray<id<MTLFunction>> *)functions
                                                 error:(NSError *_Nullable *_Nullable)error;
- (nullable id<MTLLibrary>)newLibraryWithMetalPackageURL:(NSURL *)url
                                                   error:(NSError *_Nullable *_Nullable)error;
- (nullable id<MTLLibrary>)newLibraryWithMPSGraphPackageURL:(NSURL *)url
                                                       name:(nullable NSString *)name
                                                      error:(NSError *_Nullable *_Nullable)error;
@end

// measured: macOS 27 and 26.6.1 Metal.framework — the private selectors of Metal's own library
// (_MTLLibrary) the link route reads: its extern functions' names (`@16@0:8`), one of them by
// name (`@24@0:8@16`, nil for a name it does not hold), and the bytes it was made from or linked
// into (`libraryDataContents`, `@16@0:8`, an NSData of a .metallib).
@protocol RgfxMetalLibraryInternals
- (NSArray<NSString *> *)externFunctionNames;
- (nullable id<MTLFunction>)newExternFunctionWithName:(NSString *)name;
- (nullable NSData *)libraryDataContents;
@end

// measured: macOS 27 (M3 Max) and macOS 26.7.1 (AMD Radeon Pro 5300M) Metal.framework — the
// initialisers of Metal's reflection classes (the runtime's method lists, with their type
// encodings) that RgfxReflection.mm makes a stitchable function's reflection with:
//   MTLTypeInternal                  initWithDataType:                         @24@0:8Q16
//   MTLStructTypeInternal            initWithMembers:count:typeName:           @40@0:8^@16Q24@32
//   MTLArrayTypeInternal             initWithArrayLength:elementType:stride:pixelFormat:aluType:
//                                    details:                                  @64@0:8Q16Q24Q32Q40Q48@56
//   MTLPointerTypeInternal           initWithElementType:elementTypeDescription:access:alignment:
//                                    dataSize:elementIsIndirectArgumentBuffer:isConstantBuffer:
//                                                                              @64@0:8Q16@24Q32Q40Q48B56B60
//   MTLTextureReferenceTypeInternal  initWithDataType:textureType:access:isDepthTexture:
//                                                                              @44@0:8Q16Q24Q32B40
//   MTLStructMemberInternal          initWithName:offset:dataType:pixelFormat:aluType:
//                                    indirectArgumentIndex:render_target:raster_order_group:details:
//                                                                              @88@0:8@16Q24Q32Q40Q48Q56Q64Q72@80
//   MTLBindingInternal               initWithName:type:access:index:active:arrayLength:
//                                    typeDescription:                          @68@0:8@16Q24Q32Q40B48Q52@60
//   MTLTextureBindingInternal        initWithName:access:isActive:locationIndex:arraySize:dataType:
//                                    textureType:isDepthTexture:               @72@0:8@16Q24B32Q36Q44Q52Q60B68
// What each parameter holds is read from the objects Metal's own compiler makes, getter by getter
// (tools/reflection_check.mm); `details` is the nested type object.
@protocol RgfxMetalReflectionInitialisers
- (nullable instancetype)initWithDataType:(NSUInteger)dataType;
- (nullable instancetype)initWithMembers:(__unsafe_unretained id _Nonnull const *_Nullable)members
                                   count:(NSUInteger)count
                                typeName:(nullable NSString *)typeName;
- (nullable instancetype)initWithArrayLength:(NSUInteger)arrayLength
                                 elementType:(NSUInteger)elementType
                                      stride:(NSUInteger)stride
                                 pixelFormat:(NSUInteger)pixelFormat
                                     aluType:(NSUInteger)aluType
                                     details:(nullable id)details;
- (nullable instancetype)initWithElementType:(NSUInteger)elementType
                      elementTypeDescription:(nullable id)elementTypeDescription
                                      access:(NSUInteger)access
                                   alignment:(NSUInteger)alignment
                                    dataSize:(NSUInteger)dataSize
             elementIsIndirectArgumentBuffer:(BOOL)elementIsIndirectArgumentBuffer
                            isConstantBuffer:(BOOL)isConstantBuffer;
- (nullable instancetype)initWithDataType:(NSUInteger)dataType
                              textureType:(NSUInteger)textureType
                                   access:(NSUInteger)access
                           isDepthTexture:(BOOL)isDepthTexture;
- (nullable instancetype)initWithName:(NSString *)name
                               offset:(NSUInteger)offset
                             dataType:(NSUInteger)dataType
                          pixelFormat:(NSUInteger)pixelFormat
                              aluType:(NSUInteger)aluType
                indirectArgumentIndex:(NSUInteger)indirectArgumentIndex
                        render_target:(NSUInteger)renderTarget
                   raster_order_group:(NSUInteger)rasterOrderGroup
                              details:(nullable id)details;
- (nullable instancetype)initWithName:(NSString *)name
                                 type:(NSUInteger)type
                               access:(NSUInteger)access
                                index:(NSUInteger)index
                               active:(BOOL)active
                          arrayLength:(NSUInteger)arrayLength
                      typeDescription:(nullable id)typeDescription;
- (nullable instancetype)initWithName:(NSString *)name
                               access:(NSUInteger)access
                             isActive:(BOOL)isActive
                        locationIndex:(NSUInteger)locationIndex
                            arraySize:(NSUInteger)arraySize
                             dataType:(NSUInteger)dataType
                          textureType:(NSUInteger)textureType
                       isDepthTexture:(BOOL)isDepthTexture;
@end
NS_ASSUME_NONNULL_END

// measured: macOS 27 Metal.framework — the private device SPI our MTLDevice must
// conform to. Metal's CoreImage/GL/OpenCL/WindowServer layers call ~439 selectors
// on it; this forward-declaration lists ONLY the ones a slice overrides, answered
// from the host caps blob. A later slice that overrides more SPI adds its
// selectors here, each tagged measured. (All four below are NSUInteger getters,
// type-encoding `Q16@0:8`; `-supportsPrimitiveType:` is `B24@0:8Q16`.) The base
// classes' creators the device answers that nothing in the bundle calls or types
// are tagged where they are implemented instead (RgfxMetalDevice+*.mm), and
// tests/inherited_tests.mm checks them against the base classes at run time.
@protocol MTLDeviceSPI <MTLDevice>
- (NSUInteger)maxVertexSamplers;
- (NSUInteger)maxFragmentTextures;
- (NSUInteger)maxComputeTextures;
- (NSUInteger)maxVertexAmplificationCount;
- (BOOL)supportsPrimitiveType:(NSUInteger)primitiveType;
// measured: macOS 27 Metal.framework's x86-64 slice, _MTLDevice's, `Q16@0:8` and `c16@0:8`:
// the linear texture alignments, and the framebuffer read, tile shader and imageblock answers
// the host's device-info blob carries.
- (NSUInteger)linearTextureAlignmentBytes;
- (NSUInteger)deviceLinearTextureAlignmentBytes;
- (BOOL)isFramebufferReadSupported;
- (BOOL)supportsTileShaders;
- (BOOL)supportsNonSquareTileShaders;
- (BOOL)supportsImageBlocks;
// measured: macOS 26 Metal.framework, required, `@32@0:8Q16Q24`: a texture over a legacy
// IOAccel surface, by its id and buffer (CoreDisplay's display surfaces).
- (nullable id<MTLTexture>)newTextureWithSurface:(NSUInteger)surfaceId buffer:(NSUInteger)buffer;
// measured: macOS 27 Metal.framework's x86-64 slice, _MTLDevice's, `@32@0:8@16^@24` and
// `v32@0:8@16@?24`; WindowServer on the macOS 26 guest sends the first.
- (nullable id<MTLComputePipelineState>)
    newComputePipelineStateWithDescriptor:(MTLComputePipelineDescriptor *_Nonnull)descriptor
                                    error:(NSError *_Nullable *_Nullable)error;
- (void)newComputePipelineStateWithDescriptor:(MTLComputePipelineDescriptor *_Nonnull)descriptor
                            completionHandler:
                                (MTLNewComputePipelineStateCompletionHandler _Nonnull)completionHandler;
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

// measured: macOS 27 Metal.framework — the initialisers of Metal's own IO command queue
// (`_MTLIOCommandQueue`, `initWithDevice:descriptor:`, `@32@0:8@16@24`) and of its raw and
// compressed file handles (`_MTLIOHandleRaw`, `initWithDevice:path:error:uncached:`,
// `@44@0:8@16r*24^@32B40`; `_MTLIOHandleCompressed`,
// `initWithDevice:path:compressionType:error:uncached:`, `@52@0:8@16r*24q32^@40B48`). Each works
// over the device it is given, through that device's public command queue, shared event and
// buffer members: Metal's queue over Metal's device loads buffers, process memory and textures
// as the device's own queue does (measured on the M3 Max). The device's IO creators make them
// (RgfxMetalDevice+Unsupported.mm); the classes are looked up at run time.
NS_ASSUME_NONNULL_BEGIN
@interface NSObject (RgfxMetalIO)
- (nullable instancetype)initWithDevice:(id<MTLDevice>)device
                             descriptor:(MTLIOCommandQueueDescriptor *)descriptor;
- (nullable instancetype)initWithDevice:(id<MTLDevice>)device
                                   path:(const char *)path
                                  error:(NSError *_Nullable *_Nullable)error
                               uncached:(BOOL)uncached;
- (nullable instancetype)initWithDevice:(id<MTLDevice>)device
                                   path:(const char *)path
                        compressionType:(NSInteger)compressionType
                                  error:(NSError *_Nullable *_Nullable)error
                               uncached:(BOOL)uncached;
@end
NS_ASSUME_NONNULL_END

// measured: macOS 26 Metal.framework — the private resource SPI our resources answer, with the
// selectors they override (RgfxResource).
@protocol MTLResourceSPI <MTLResource>
// measured: required, `v20@0:8i16` and `i16@0:8`.
- (void)setResponsibleProcess:(int)pid;
- (int)responsibleProcess;
// measured: required, `Q16@0:8` (MTLHeapSPI has it too, optional); no setter is declared.
- (uint64_t)protectionOptions;
@end

// measured: macOS 27 Metal.framework, on the M3 Max: members Metal's own heap, buffer, texture
// and sampler classes answer, which MPS, Core Image, QuartzCore and SkyLight call and no protocol
// of the SDK declares for that kind, with their type encodings. Each wrapper adopts its protocol
// in its own file.
NS_ASSUME_NONNULL_BEGIN
// The resource SPI's members (MTLResourceSPI) and two more, which Metal's buffer and texture
// classes answer alike.
@protocol RgfxMetalResourceInternals
// `Q16@0:8`: the id of the allocation the resource lies in.
- (uint64_t)allocationID;
// `B24@0:8@16`, and `B32@0:8r^@16Q24` for the two lists: whether the resource's memory meets
// another's.
- (BOOL)doesAliasResource:(nullable id<MTLResource>)resource;
- (BOOL)doesAliasAllResources:(const id<MTLResource> _Nonnull[_Nullable])resources
                        count:(NSUInteger)count;
- (BOOL)doesAliasAnyResources:(const id<MTLResource> _Nonnull[_Nullable])resources
                        count:(NSUInteger)count;
// `B16@0:8` each, and `v16@0:8` for the wait.
- (BOOL)isComplete;
- (BOOL)isWriteComplete;
- (BOOL)isPurgeable;
- (void)waitUntilComplete;
// `Q16@0:8`: the options the resource was made with.
- (MTLResourceOptions)unfilteredResourceOptions;
// `Q16@0:8`: the size of the allocation the resource lies in.
- (uint64_t)resourceSize;
// `^v16@0:8`: the CPU's address of the resource's memory.
- (nullable void *)virtualAddress;
@end

@protocol RgfxMetalBufferInternals <RgfxMetalResourceInternals>
// `^{__IOSurface=}16@0:8`: the surface a buffer made over one lies in.
- (nullable IOSurfaceRef)iosurface;
// `@48@0:8@16Q24Q32Q40`: a texture over the buffer's bytes, as newTextureWithDescriptor:offset:
// bytesPerRow: makes one, which takes a 2D array too, its slices `bytesPerImage` apart.
- (nullable id<MTLTexture>)newLinearTextureWithDescriptor:(MTLTextureDescriptor *)descriptor
                                                  offset:(NSUInteger)offset
                                             bytesPerRow:(NSUInteger)bytesPerRow
                                           bytesPerImage:(NSUInteger)bytesPerImage;
// `@40@0:8@16Q24Q32` (Metal's validation layer's MTLDebugBuffer): a texture over a Shared
// buffer's bytes from a Managed descriptor, rows `bytesPerRow` apart (RgfxBuffer.mm).
- (nullable id<MTLTexture>)newTiledTextureWithDescriptor:(MTLTextureDescriptor *)descriptor
                                                 offset:(NSUInteger)offset
                                            bytesPerRow:(NSUInteger)bytesPerRow;
@end

@protocol RgfxMetalTextureInternals <RgfxMetalResourceInternals>
// `Q16@0:8`: the GPU address of the texture's memory.
- (uint64_t)gpuAddress;
// `B16@0:8`: whether the GPU keeps the texture's contents losslessly compressed.
- (BOOL)isCompressed;
// `B16@0:8`: whether the texture is a drawable's.
- (BOOL)isDrawable;
// `Q16@0:8`: the rotation the texture's contents are shown at.
- (NSUInteger)rotation;
// `Q16@0:8`: the texture's number among the process's textures.
- (uint64_t)uniqueIdentifier;
// `v16@0:8`: the CPU wrote the texture's memory.
- (void)didModifyData;
@end

@protocol RgfxMetalSamplerInternals
// `Q16@0:8` each: the sampler's GPU address, its pixel format and its identifier.
- (uint64_t)gpuAddress;
- (MTLPixelFormat)pixelFormat;
- (uint64_t)uniqueIdentifier;
@end

@protocol RgfxMetalSharedEventInternals
// `@16@0:8`: the event as an IOSurface shared event.
- (id)IOSurfaceSharedEvent;
// `I16@0:8`: the event's Mach port (IOSurfaceSharedEvent's), the same name at each ask; the
// event holds one send reference of it, let go at its dealloc (measured, oracle_events refs: the
// AMD 5300M and the Intel 630 under macOS 26.7.1).
- (mach_port_t)eventPort;
// `Q16@0:8`: the trace id of the event's label, which a handle made of the event keeps.
- (uint64_t)labelTraceID;
@end

// measured: MTLSharedEventHandle on macOS 26.7.1 (oracle_events intro and fake, the AMD 5300M and
// the Intel 630): its maker from an event, `@24@0:8@16`, which asks the event its eventPort, its
// label and its labelTraceID, and takes a send reference of the port of its own, let go at its
// dealloc; and its port, `I16@0:8`, which its NSXPC coding carries to another process.
@interface MTLSharedEventHandle (RgfxPrivate)
- (instancetype)initWithSharedEvent:(id<MTLSharedEvent>)event;
- (mach_port_t)eventPort;
@end

@protocol RgfxMetalHeapInternals
// `Q16@0:8`: the heap's base address, where a resource at offset 0 lies.
- (uint64_t)gpuAddress;
// `Q16@0:8`: the options the heap was made with, as resourceOptions answers them.
- (MTLResourceOptions)unfilteredResourceOptions;
// `@24@0:8@16`: a buffer the heap places, of a buffer descriptor's length (MTLBufferDescriptor,
// below).
- (nullable id<MTLBuffer>)newBufferWithDescriptor:(nullable id)descriptor;
// `@32@0:8Q16Q24`: an acceleration structure of `size` bytes in the heap.
- (nullable id<MTLAccelerationStructure>)newAccelerationStructureWithSize:(NSUInteger)size
                                                            resourceIndex:(NSUInteger)index;
@end

// measured: macOS 27 Metal.framework: the members of MTLBufferDescriptor (an NSObject) that
// newBufferWithDescriptor: reads here, with their type encodings: `length`, `alignment`,
// `pinnedGPUAddress` and `parentGPUAddress` `Q16@0:8`, `placementSparsePageSize` `q16@0:8`,
// `contents` `^v16@0:8`, `noCopy` `B16@0:8`, `iosurface` `^{__IOSurface=}16@0:8` and
// `addressRanges` `@16@0:8`. A new descriptor's alignment is 1, and every other member is 0, NO
// or nil.
@protocol RgfxMetalBufferDescriptor <NSObject>
- (NSUInteger)length;
- (NSUInteger)alignment;
- (uint64_t)pinnedGPUAddress;
- (uint64_t)parentGPUAddress;
- (NSInteger)placementSparsePageSize;
- (nullable void *)contents;
- (BOOL)noCopy;
- (nullable IOSurfaceRef)iosurface;
- (nullable id)addressRanges;
@end
// measured: macOS 27 and 26.7.1 Metal.framework: MTLTextureDescriptor's private
// `framebufferOnly` (`B16@0:8` on arm64, `c16@0:8` on x86-64; NO in a new descriptor), which a
// CAMetalLayer sets on its drawables' descriptors from its own framebufferOnly. Metal's textures
// answer it as isFramebufferOnly, and so do their views (M3 Max and AMD 5300M).
@protocol RgfxMetalTextureDescriptor <NSObject>
- (BOOL)framebufferOnly;
@end
NS_ASSUME_NONNULL_END

#endif // RGFX_METAL_PRIVATE_H_
