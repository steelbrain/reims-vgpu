/*
 * RgfxMetalDevice (Unsupported): see the header. Our own ObjC++.
 */

#import "RgfxMetalDevice+Unsupported.h"

#import "RgfxBlockCall.h"
#import "RgfxUnsupported.h"

namespace {
constexpr const char *kRayTracing = "ray tracing";
constexpr const char *kFunctionPointers = "function pointers";
constexpr const char *kMetal4 = "the Metal 4 API";
constexpr const char *kOpenGL = "OpenGL";
constexpr const char *kArgumentLayouts = "argument buffers of a layout object";
} // namespace

@implementation RgfxMetalDevice (Unsupported)

// --- ray tracing ---------------------------------------------------------------------

- (id<MTLAccelerationStructure>)newAccelerationStructureWithSize:(NSUInteger)size
{
	RGFX_UNSUPPORTED(kRayTracing);
	return nil;
}

- (id<MTLAccelerationStructure>)newAccelerationStructureWithDescriptor:
    (MTLAccelerationStructureDescriptor *)descriptor
{
	RGFX_UNSUPPORTED(kRayTracing);
	return nil;
}

// --- function pointers and dynamic libraries ------------------------------------------

- (id<MTLFunctionHandle>)functionHandleWithFunction:(id<MTLFunction>)function
{
	RGFX_UNSUPPORTED(kFunctionPointers);
	return nil;
}

- (id<MTLFunctionHandle>)functionHandleWithBinaryFunction:(id<MTL4BinaryFunction>)function
{
	RGFX_UNSUPPORTED(kFunctionPointers);
	return nil;
}

- (id<MTLDynamicLibrary>)newDynamicLibrary:(id<MTLLibrary>)library error:(NSError **)error
{
	RGFX_UNSUPPORTED_ERROR_IN(error, "dynamic libraries", MTLDynamicLibraryDomain,
	                          MTLDynamicLibraryErrorUnsupported);
	return nil;
}

- (id<MTLDynamicLibrary>)newDynamicLibraryWithURL:(NSURL *)url error:(NSError **)error
{
	RGFX_UNSUPPORTED_ERROR_IN(error, "dynamic libraries", MTLDynamicLibraryDomain,
	                          MTLDynamicLibraryErrorUnsupported);
	return nil;
}

// --- argument encoders --------------------------------------------------------------

// No argument-buffer layout on the wire: no encoder, which apps and Metal's own frameworks check,
// rather than one that lays out nothing.
- (id<MTLArgumentEncoder>)newArgumentEncoderWithBufferBinding:(id<MTLBufferBinding>)bufferBinding
{
	RGFX_UNSUPPORTED("argument encoders (no argument-buffer layout on the wire)");
	return nil;
}

// --- pipelines the wire does not create -----------------------------------------------

- (id<MTLRenderPipelineState>)
    newRenderPipelineStateWithTileDescriptor:(MTLTileRenderPipelineDescriptor *)descriptor
                                     options:(MTLPipelineOption)options
                                  reflection:(MTLAutoreleasedRenderPipelineReflection *)reflection
                                       error:(NSError **)error
{
	if (reflection)
		*reflection = nil;
	RGFX_UNSUPPORTED_ERROR(error, "tile pipelines (no tile pipeline create on the wire)");
	return nil;
}

- (void)newRenderPipelineStateWithTileDescriptor:(MTLTileRenderPipelineDescriptor *)descriptor
                                         options:(MTLPipelineOption)options
                               completionHandler:
                                   (MTLNewRenderPipelineStateWithReflectionCompletionHandler)
                                       completionHandler
{
	NSError *error = nil;
	RGFX_UNSUPPORTED_ERROR(&error, "tile pipelines (no tile pipeline create on the wire)");
	completionHandler = [completionHandler copy];
	RgfxRunCreation(^{
		if (completionHandler)
			completionHandler(nil, nil, error);
	});
}

- (id<MTLRenderPipelineState>)
    newRenderPipelineStateWithMeshDescriptor:(MTLMeshRenderPipelineDescriptor *)descriptor
                                     options:(MTLPipelineOption)options
                                  reflection:(MTLAutoreleasedRenderPipelineReflection *)reflection
                                       error:(NSError **)error
{
	if (reflection)
		*reflection = nil;
	RGFX_UNSUPPORTED_ERROR(error, "mesh pipelines (no mesh pipeline create on the wire)");
	return nil;
}

- (void)newRenderPipelineStateWithMeshDescriptor:(MTLMeshRenderPipelineDescriptor *)descriptor
                                         options:(MTLPipelineOption)options
                               completionHandler:
                                   (MTLNewRenderPipelineStateWithReflectionCompletionHandler)
                                       completionHandler
{
	NSError *error = nil;
	RGFX_UNSUPPORTED_ERROR(&error, "mesh pipelines (no mesh pipeline create on the wire)");
	completionHandler = [completionHandler copy];
	RgfxRunCreation(^{
		if (completionHandler)
			completionHandler(nil, nil, error);
	});
}

// --- resources the device does not make -----------------------------------------------

- (id<MTLBuffer>)newBufferWithLength:(NSUInteger)length
                             options:(MTLResourceOptions)options
             placementSparsePageSize:(MTLSparsePageSize)placementSparsePageSize
{
	RGFX_UNSUPPORTED("placement sparse buffers");
	return nil;
}

// Textures shared with another process by handle: the wire names no texture across processes
// (IOSurface-backed textures are the shared ones it carries).
- (id<MTLTexture>)newSharedTextureWithDescriptor:(MTLTextureDescriptor *)descriptor
{
	RGFX_UNSUPPORTED("shared textures (no cross-process texture handle on the wire)");
	return nil;
}

- (id<MTLTexture>)newSharedTextureWithHandle:(MTLSharedTextureHandle *)sharedHandle
{
	RGFX_UNSUPPORTED("shared textures (no cross-process texture handle on the wire)");
	return nil;
}

- (id<MTLTensor>)newTensorWithDescriptor:(MTLTensorDescriptor *)descriptor
                                   error:(NSError **)error
{
	RGFX_UNSUPPORTED_ERROR_IN(error, "tensors", MTLTensorDomain, MTLTensorErrorInternalError);
	return nil;
}

- (id<MTLTensor>)newTensorWithDescriptor:(MTLTensorDescriptor *)descriptor
                             attachments:(MTLTensorBufferAttachments *)attachments
                                   error:(NSError **)error
{
	RGFX_UNSUPPORTED_ERROR_IN(error, "tensors", MTLTensorDomain, MTLTensorErrorInternalError);
	return nil;
}

- (id<MTLRasterizationRateMap>)newRasterizationRateMapWithDescriptor:
    (MTLRasterizationRateMapDescriptor *)descriptor
{
	RGFX_UNSUPPORTED("rasterization rate maps");
	return nil;
}

- (id<MTLResidencySet>)newResidencySetWithDescriptor:(MTLResidencySetDescriptor *)desc
                                               error:(NSError **)error
{
	RGFX_UNSUPPORTED_ERROR(error, "residency sets (none on the wire)");
	return nil;
}

// --- counters and shader logging ------------------------------------------------------

- (id<MTLCounterSampleBuffer>)newCounterSampleBufferWithDescriptor:
                                  (MTLCounterSampleBufferDescriptor *)descriptor
                                                             error:(NSError **)error
{
	RGFX_UNSUPPORTED_ERROR_IN(error, "counter sampling", MTLCounterErrorDomain,
	                          MTLCounterSampleBufferErrorInvalid);
	return nil;
}

- (id<MTLLogState>)newLogStateWithDescriptor:(MTLLogStateDescriptor *)descriptor
                                       error:(NSError **)error
{
	RGFX_UNSUPPORTED_ERROR_IN(error, "shader log states (none on the wire)",
	                          MTLLogStateErrorDomain, MTLLogStateErrorInvalid);
	return nil;
}

// --- the Metal 4 API ---------------------------------------------------------------------

- (id<MTL4CommandQueue>)newMTL4CommandQueue
{
	RGFX_UNSUPPORTED(kMetal4);
	return nil;
}

- (id<MTL4CommandQueue>)newMTL4CommandQueueWithDescriptor:(MTL4CommandQueueDescriptor *)descriptor
                                                    error:(NSError **)error
{
	RGFX_UNSUPPORTED_ERROR(error, kMetal4);
	return nil;
}

- (id<MTL4CommandBuffer>)newCommandBuffer
{
	RGFX_UNSUPPORTED(kMetal4);
	return nil;
}

- (id<MTL4CommandAllocator>)newCommandAllocator
{
	RGFX_UNSUPPORTED(kMetal4);
	return nil;
}

- (id<MTL4CommandAllocator>)newCommandAllocatorWithDescriptor:
                                (MTL4CommandAllocatorDescriptor *)descriptor
                                                        error:(NSError **)error
{
	RGFX_UNSUPPORTED_ERROR(error, kMetal4);
	return nil;
}

- (id<MTL4Compiler>)newCompilerWithDescriptor:(MTL4CompilerDescriptor *)descriptor
                                        error:(NSError **)error
{
	RGFX_UNSUPPORTED_ERROR(error, kMetal4);
	return nil;
}

- (id<MTL4ArgumentTable>)newArgumentTableWithDescriptor:(MTL4ArgumentTableDescriptor *)descriptor
                                                  error:(NSError **)error
{
	RGFX_UNSUPPORTED_ERROR(error, kMetal4);
	return nil;
}

- (id<MTL4Archive>)newArchiveWithURL:(NSURL *)url error:(NSError **)error
{
	RGFX_UNSUPPORTED_ERROR(error, kMetal4);
	return nil;
}

- (id<MTL4CounterHeap>)newCounterHeapWithDescriptor:(MTL4CounterHeapDescriptor *)descriptor
                                              error:(NSError **)error
{
	RGFX_UNSUPPORTED_ERROR(error, kMetal4);
	return nil;
}

- (id<MTL4PipelineDataSetSerializer>)newPipelineDataSetSerializerWithDescriptor:
    (MTL4PipelineDataSetSerializerDescriptor *)descriptor
{
	RGFX_UNSUPPORTED(kMetal4);
	return nil;
}

- (id<MTLTextureViewPool>)newTextureViewPoolWithDescriptor:(MTLResourceViewPoolDescriptor *)descriptor
                                                     error:(NSError **)error
{
	RGFX_UNSUPPORTED_ERROR(error, kMetal4);
	return nil;
}

// --- the base classes' creators no header declares --------------------------------------
//
// Creators IOGPUMetalDevice and _MTLDevice implement, measured with their type encodings by
// enumerating both classes on macOS 27's x86-64 slice and on the guest's macOS 26 (each of the
// guest's is on macOS 27's list). Inherited, they build on the base device's compiler, IOGPU
// objects or OpenGL state, none of which the device has, and answer nil without an error; so
// each answers as the bundle's other unsupported members do. tests/inherited_tests.mm keeps every
// base-class creator either answered by the device or in its list of those left to the base.

// Mesh pipelines without options (`@32@0:8@16^@24`, `v32@0:8@16@?24`), refused as the forms with
// options are, above.
- (id<MTLRenderPipelineState>)newRenderPipelineStateWithMeshDescriptor:
                                  (MTLMeshRenderPipelineDescriptor *)descriptor
                                                                 error:(NSError **)error
{
	RGFX_UNSUPPORTED_ERROR(error, "mesh pipelines (no mesh pipeline create on the wire)");
	return nil;
}

- (void)newRenderPipelineStateWithMeshDescriptor:(MTLMeshRenderPipelineDescriptor *)descriptor
                               completionHandler:(id)completionHandler
{
	NSError *error = nil;
	RGFX_UNSUPPORTED_ERROR(&error, "mesh pipelines (no mesh pipeline create on the wire)");
	completionHandler = [completionHandler copy];
	RgfxRunCreation(^{
		RgfxCallCreationHandler(completionHandler, nil, error, _cmd);
	});
}

// Ray tracing: acceleration structures over a buffer (`@32@0:8@16Q24`, `@40@0:8@16Q24Q32`), of a
// size at a resource index (`@32@0:8Q16Q24`) or with a descriptor (`@32@0:8Q16@24`), and
// intersection function tables (`@24@0:8@16`).
- (id<MTLAccelerationStructure>)newAccelerationStructureWithBuffer:(id<MTLBuffer>)buffer
                                                            offset:(NSUInteger)offset
{
	RGFX_UNSUPPORTED(kRayTracing);
	return nil;
}

- (id<MTLAccelerationStructure>)newAccelerationStructureWithBuffer:(id<MTLBuffer>)buffer
                                                            offset:(NSUInteger)offset
                                                     resourceIndex:(NSUInteger)resourceIndex
{
	RGFX_UNSUPPORTED(kRayTracing);
	return nil;
}

- (id<MTLAccelerationStructure>)newAccelerationStructureWithSize:(NSUInteger)size
                                                   resourceIndex:(NSUInteger)resourceIndex
{
	RGFX_UNSUPPORTED(kRayTracing);
	return nil;
}

- (id<MTLAccelerationStructure>)newAccelerationStructureWithSize:(NSUInteger)size
                                                  withDescriptor:(id)descriptor
{
	RGFX_UNSUPPORTED(kRayTracing);
	return nil;
}

- (id<MTLIntersectionFunctionTable>)newIntersectionFunctionTableWithDescriptor:(id)descriptor
{
	RGFX_UNSUPPORTED(kRayTracing);
	return nil;
}

// Function pointers: a visible function's compile (`v40@0:8@16@24^@32`,
// `v48@0:8@16@24@32^@40`, `v40@0:8@16@24@?32`) and a function's handle at a resource index
// (`@32@0:8@16Q24`).
- (void)compileVisibleFunction:(id<MTLFunction>)function
                withDescriptor:(id)descriptor
                         error:(NSError **)error
{
	RGFX_UNSUPPORTED_ERROR(error, kFunctionPointers);
}

- (void)compileVisibleFunction:(id<MTLFunction>)function
                withDescriptor:(id)descriptor
      destinationBinaryArchive:(id<MTLBinaryArchive>)archive
                         error:(NSError **)error
{
	RGFX_UNSUPPORTED_ERROR(error, kFunctionPointers);
}

- (void)compileVisibleFunction:(id<MTLFunction>)function
                withDescriptor:(id)descriptor
             completionHandler:(id)completionHandler
{
	NSError *error = nil;
	RGFX_UNSUPPORTED_ERROR(&error, kFunctionPointers);
	completionHandler = [completionHandler copy];
	RgfxRunCreation(^{
		RgfxCallCreationHandler(completionHandler, nil, error, _cmd);
	});
}

- (id<MTLFunctionHandle>)functionHandleWithFunction:(id<MTLFunction>)function
                                      resourceIndex:(NSUInteger)resourceIndex
{
	RGFX_UNSUPPORTED(kFunctionPointers);
	return nil;
}

// Dynamic libraries (`@40@0:8@16@24^@32`, `@32@0:8@16^@24` twice, `@40@0:8@16Q24^@32`) and binary
// libraries (`@40@0:8Q16@24^@32`).
- (id<MTLDynamicLibrary>)newDynamicLibrary:(id<MTLLibrary>)library
                         computeDescriptor:(id)descriptor
                                     error:(NSError **)error
{
	RGFX_UNSUPPORTED_ERROR_IN(error, "dynamic libraries", MTLDynamicLibraryDomain,
	                          MTLDynamicLibraryErrorUnsupported);
	return nil;
}

- (id<MTLDynamicLibrary>)newDynamicLibraryWithDescriptor:(id)descriptor error:(NSError **)error
{
	RGFX_UNSUPPORTED_ERROR_IN(error, "dynamic libraries", MTLDynamicLibraryDomain,
	                          MTLDynamicLibraryErrorUnsupported);
	return nil;
}

- (id<MTLDynamicLibrary>)newDynamicLibraryFromURL:(NSURL *)url error:(NSError **)error
{
	RGFX_UNSUPPORTED_ERROR_IN(error, "dynamic libraries", MTLDynamicLibraryDomain,
	                          MTLDynamicLibraryErrorUnsupported);
	return nil;
}

- (id<MTLDynamicLibrary>)newDynamicLibraryWithURL:(NSURL *)url
                                          options:(NSUInteger)options
                                            error:(NSError **)error
{
	RGFX_UNSUPPORTED_ERROR_IN(error, "dynamic libraries", MTLDynamicLibraryDomain,
	                          MTLDynamicLibraryErrorUnsupported);
	return nil;
}

- (id)newBinaryLibraryWithOptions:(NSUInteger)options url:(NSURL *)url error:(NSError **)error
{
	RGFX_UNSUPPORTED_ERROR(error, "binary libraries");
	return nil;
}

// OpenGL's functions of its IR (`@32@0:8^v16Q24` and `@40@0:8^v16@24Q32` for each of GL, GL Core
// and GL ES) and drawables (`@16@0:8`): the device runs no OpenGL.
- (id<MTLFunction>)newFunctionWithGLIR:(void *)ir functionType:(NSUInteger)functionType
{
	RGFX_UNSUPPORTED(kOpenGL);
	return nil;
}

- (id<MTLFunction>)newFunctionWithGLIR:(void *)ir
                     inputsDescription:(id)inputsDescription
                          functionType:(NSUInteger)functionType
{
	RGFX_UNSUPPORTED(kOpenGL);
	return nil;
}

- (id<MTLFunction>)newFunctionWithGLCoreIR:(void *)ir functionType:(NSUInteger)functionType
{
	RGFX_UNSUPPORTED(kOpenGL);
	return nil;
}

- (id<MTLFunction>)newFunctionWithGLCoreIR:(void *)ir
                         inputsDescription:(id)inputsDescription
                              functionType:(NSUInteger)functionType
{
	RGFX_UNSUPPORTED(kOpenGL);
	return nil;
}

- (id<MTLFunction>)newFunctionWithGLESIR:(void *)ir functionType:(NSUInteger)functionType
{
	RGFX_UNSUPPORTED(kOpenGL);
	return nil;
}

- (id<MTLFunction>)newFunctionWithGLESIR:(void *)ir
                       inputsDescription:(id)inputsDescription
                            functionType:(NSUInteger)functionType
{
	RGFX_UNSUPPORTED(kOpenGL);
	return nil;
}

- (id)newGLDrawable
{
	RGFX_UNSUPPORTED(kOpenGL);
	return nil;
}

// Argument buffers of a layout (`@24@0:8@16` each): the host lays arguments out
// (RgfxArgumentEncoder.h), from their descriptors, never from a layout object of the base's.
- (id<MTLArgumentEncoder>)newArgumentEncoderWithLayout:(id)layout
{
	RGFX_UNSUPPORTED(kArgumentLayouts);
	return nil;
}

- (id<MTLArgumentEncoder>)newIndirectArgumentEncoderWithLayout:(id)layout
{
	RGFX_UNSUPPORTED(kArgumentLayouts);
	return nil;
}

- (id)newIndirectArgumentBufferLayoutWithStructType:(id)structType
{
	RGFX_UNSUPPORTED(kArgumentLayouts);
	return nil;
}

// Queues, allocators, events and assertions the wire has nothing for (`@16@0:8` each but the
// assertion's `@32@0:8q16^@24`).
- (id)newCommandQueue4
{
	RGFX_UNSUPPORTED("newCommandQueue4's queue (none on the wire)");
	return nil;
}

- (id)newDevicePoolAliasedCommandAllocator
{
	RGFX_UNSUPPORTED("aliased command allocators (none on the wire)");
	return nil;
}

- (id)newLateEvalEvent
{
	RGFX_UNSUPPORTED("late-evaluated events (none on the wire)");
	return nil;
}

- (id)newPerformanceStateAssertion:(NSInteger)state error:(NSError **)error
{
	RGFX_UNSUPPORTED_ERROR(error, "performance state assertions (none on the wire)");
	return nil;
}

// Pipeline libraries of a file (`@32@0:8@16^@24`) and profiles (`@24@0:8Q16`).
- (id)newPipelineLibraryWithFilePath:(NSString *)filePath error:(NSError **)error
{
	RGFX_UNSUPPORTED_ERROR(error, "pipeline libraries");
	return nil;
}

- (id)newProfileWithExecutionSize:(NSUInteger)executionSize
{
	RGFX_UNSUPPORTED("profiles");
	return nil;
}

// Residency sets (`@16@0:8`), counters' resolve (`@40@0:8@16{_NSRange=QQ}24`), rasterization
// rate maps of a screen size (`@56@0:8{?=QQQ}16Q40r^@48`) and textures shared by handle at a
// resource index (`@32@0:8@16Q24`), as their public forms above.
- (id)allocResidencySet
{
	RGFX_UNSUPPORTED("residency sets (none on the wire)");
	return nil;
}

- (NSData *)resolveCounters:(id<MTLCounterSampleBuffer>)buffer withRange:(NSRange)range
{
	RGFX_UNSUPPORTED("counter sampling");
	return nil;
}

- (id<MTLRasterizationRateMap>)
    newRasterizationRateMapWithScreenSize:(MTLSize)screenSize
                               layerCount:(NSUInteger)layerCount
                                   layers:(const __unsafe_unretained id *)layers
{
	RGFX_UNSUPPORTED("rasterization rate maps");
	return nil;
}

- (id<MTLTexture>)newSharedTextureWithHandle:(MTLSharedTextureHandle *)sharedHandle
                           withResourceIndex:(NSUInteger)resourceIndex
{
	RGFX_UNSUPPORTED("shared textures (no cross-process texture handle on the wire)");
	return nil;
}

@end
