/*
 * RgfxMetalDevice (Unsupported): see the header. Our own ObjC++.
 */

#import "RgfxMetalDevice+Unsupported.h"

#import "RgfxUnsupported.h"

namespace {
constexpr const char *kRayTracing = "ray tracing";
constexpr const char *kFunctionPointers = "function pointers";
constexpr const char *kMetal4 = "the Metal 4 API";
constexpr const char *kIO = "Metal IO command queues";
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

// --- argument encoders and indirect command buffers -----------------------------------

// No argument-buffer layout on the wire: no encoder, which apps and Metal's own frameworks check,
// rather than one that lays out nothing.
- (id<MTLArgumentEncoder>)newArgumentEncoderWithBufferBinding:(id<MTLBufferBinding>)bufferBinding
{
	RGFX_UNSUPPORTED("argument encoders (no argument-buffer layout on the wire)");
	return nil;
}

- (id<MTLIndirectCommandBuffer>)newIndirectCommandBufferWithDescriptor:
                                    (MTLIndirectCommandBufferDescriptor *)descriptor
                                                       maxCommandCount:(NSUInteger)maxCount
                                                               options:(MTLResourceOptions)options
{
	RGFX_UNSUPPORTED("indirect command buffers (none on the wire)");
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
	if (completionHandler)
		completionHandler(nil, nil, error);
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
	if (completionHandler)
		completionHandler(nil, nil, error);
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

// --- Metal IO --------------------------------------------------------------------------

- (id<MTLIOCommandQueue>)newIOCommandQueueWithDescriptor:(MTLIOCommandQueueDescriptor *)descriptor
                                                   error:(NSError **)error
{
	RGFX_UNSUPPORTED_ERROR_IN(error, kIO, MTLIOErrorDomain, MTLIOErrorInternal);
	return nil;
}

- (id<MTLIOFileHandle>)newIOFileHandleWithURL:(NSURL *)url error:(NSError **)error
{
	RGFX_UNSUPPORTED_ERROR_IN(error, kIO, MTLIOErrorDomain, MTLIOErrorInternal);
	return nil;
}

- (id<MTLIOFileHandle>)newIOFileHandleWithURL:(NSURL *)url
                            compressionMethod:(MTLIOCompressionMethod)compressionMethod
                                        error:(NSError **)error
{
	RGFX_UNSUPPORTED_ERROR_IN(error, kIO, MTLIOErrorDomain, MTLIOErrorInternal);
	return nil;
}

- (id<MTLIOFileHandle>)newIOHandleWithURL:(NSURL *)url error:(NSError **)error
{
	RGFX_UNSUPPORTED_ERROR_IN(error, kIO, MTLIOErrorDomain, MTLIOErrorInternal);
	return nil;
}

- (id<MTLIOFileHandle>)newIOHandleWithURL:(NSURL *)url
                        compressionMethod:(MTLIOCompressionMethod)compressionMethod
                                    error:(NSError **)error
{
	RGFX_UNSUPPORTED_ERROR_IN(error, kIO, MTLIOErrorDomain, MTLIOErrorInternal);
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

@end
