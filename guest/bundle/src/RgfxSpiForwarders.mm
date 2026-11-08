// The members of Metal's SPI protocols (MTLLibrarySPI, MTLFunctionSPI, ...) that are another
// member the class answers (a public one, or reflectionWithOptions:) with one more argument,
// answered as that member answers. Metal's own objects adopt the SPI protocols, and the guest's
// frameworks send these without asking respondsToSelector: first: macOS 13's Maps (VectorKit's
// MetalResourceManager::newShaderFunction) sends newFunctionWithName:constantValues:
// pipelineLibrary:error: to the libraries it makes, and ended on an unrecognized selector before
// this. The AMD 5300M's and the Intel 630's objects both answer each member here (measured with
// respondsToSelector:, macOS 26.7.1).
//
// What each drops:
//   - pipelineLibrary:, functionCache: and binaryArchives: name where an answer made before may
//     be looked up. The device makes no pipeline library (newPipelineLibraryWithFilePath:error: is
//     unsupported), so the answer is made as the other member makes it.
//   - options: of an archive's adds and of serializeToURL:options:error:, which the public
//     members, taking none, fill and write the archive without.
//   - resourceIndex: and resourceIndices: ask for a resource's place in a table of resources the
//     device does not keep: the view or pipeline is made as the public member makes it.
//   - timeout: bounds the wait, which the public member's does not.
//   - options: of presentDrawable:options:, which presentDrawable: presents without.
//   - completionHandler: takes the answer on the creation queue instead of as the return value.
// The arguments the forwarders drop are taken as integers, so an object passed there is neither
// retained nor released, whatever its type.
#import "RgfxBinaryArchive.h"
#import "RgfxBlockCall.h"
#import "RgfxCommandObjects.h"
#import "RgfxComputePipelineState.h"
#import "RgfxFunction.h"
#import "RgfxGLFunction.h"
#import "RgfxHeap.h"
#import "RgfxLibrary.h"
#import "RgfxRenderPipelineState.h"
#import "RgfxTexture.h"
#import "RgfxUnsupported.h"

// The MTLFunctionSPI member RgfxFunction.mm answers.
@interface RgfxFunction (RgfxSpiReflection)
- (id)reflectionWithOptions:(NSUInteger)options;
@end

// --- MTLLibrarySPI ---------------------------------------------------------------

@implementation RgfxLibrary (RgfxSpiForwarders)

- (id<MTLFunction>)newFunctionWithName:(NSString *)name
                        constantValues:(MTLFunctionConstantValues *)constantValues
                       pipelineLibrary:(uintptr_t)pipelineLibrary
                                 error:(__autoreleasing NSError **)error
{
	(void)pipelineLibrary;
	return [self newFunctionWithName:name constantValues:constantValues error:error];
}

- (void)newFunctionWithName:(NSString *)name
             constantValues:(MTLFunctionConstantValues *)constantValues
            pipelineLibrary:(uintptr_t)pipelineLibrary
          completionHandler:(void (^)(id<MTLFunction> _Nullable, NSError *_Nullable))completionHandler
{
	(void)pipelineLibrary;
	[self newFunctionWithName:name constantValues:constantValues completionHandler:completionHandler];
}

- (id<MTLFunction>)newFunctionWithName:(NSString *)name
                        constantValues:(MTLFunctionConstantValues *)constantValues
                         functionCache:(uintptr_t)functionCache
                                 error:(__autoreleasing NSError **)error
{
	(void)functionCache;
	return [self newFunctionWithName:name constantValues:constantValues error:error];
}

// The function, as newFunctionWithDescriptor:error: makes it, added to `destinationArchive` where
// that is one of the device's, as addFunctionWithDescriptor:library:error: adds it.
- (id<MTLFunction>)newFunctionWithDescriptor:(MTLFunctionDescriptor *)descriptor
                          destinationArchive:(id<MTLBinaryArchive>)destinationArchive
                                       error:(__autoreleasing NSError **)error
{
	id<MTLFunction> fn = [self newFunctionWithDescriptor:descriptor error:error];
	if (fn != nil && [(id)destinationArchive isKindOfClass:[RgfxBinaryArchive class]])
		[destinationArchive addFunctionWithDescriptor:descriptor library:self error:nil];
	return fn;
}

@end

@implementation RgfxFunctionArrayLibrary (RgfxSpiForwarders)

- (id<MTLFunction>)newFunctionWithName:(NSString *)name
                        constantValues:(MTLFunctionConstantValues *)constantValues
                       pipelineLibrary:(uintptr_t)pipelineLibrary
                                 error:(__autoreleasing NSError **)error
{
	(void)pipelineLibrary;
	return [self newFunctionWithName:name constantValues:constantValues error:error];
}

- (void)newFunctionWithName:(NSString *)name
             constantValues:(MTLFunctionConstantValues *)constantValues
            pipelineLibrary:(uintptr_t)pipelineLibrary
          completionHandler:(void (^)(id<MTLFunction> _Nullable, NSError *_Nullable))completionHandler
{
	(void)pipelineLibrary;
	[self newFunctionWithName:name constantValues:constantValues completionHandler:completionHandler];
}

- (id<MTLFunction>)newFunctionWithName:(NSString *)name
                        constantValues:(MTLFunctionConstantValues *)constantValues
                         functionCache:(uintptr_t)functionCache
                                 error:(__autoreleasing NSError **)error
{
	(void)functionCache;
	return [self newFunctionWithName:name constantValues:constantValues error:error];
}

- (id<MTLFunction>)newFunctionWithDescriptor:(MTLFunctionDescriptor *)descriptor
                          destinationArchive:(id<MTLBinaryArchive>)destinationArchive
                                       error:(__autoreleasing NSError **)error
{
	id<MTLFunction> fn = [self newFunctionWithDescriptor:descriptor error:error];
	if (fn != nil && [(id)destinationArchive isKindOfClass:[RgfxBinaryArchive class]])
		[destinationArchive addFunctionWithDescriptor:descriptor library:self error:nil];
	return fn;
}

@end

// --- MTLFunctionSPI --------------------------------------------------------------

@implementation RgfxFunction (RgfxSpiForwarders)

- (id<MTLArgumentEncoder>)newArgumentEncoderWithBufferIndex:(NSUInteger)bufferIndex
                                            pipelineLibrary:(uintptr_t)pipelineLibrary
{
	(void)pipelineLibrary;
	return [self newArgumentEncoderWithBufferIndex:bufferIndex];
}

- (id<MTLArgumentEncoder>)newArgumentEncoderWithBufferIndex:(NSUInteger)bufferIndex
                                                 reflection:(MTLAutoreleasedArgument *)reflection
                                            pipelineLibrary:(uintptr_t)pipelineLibrary
{
	(void)pipelineLibrary;
	return [self newArgumentEncoderWithBufferIndex:bufferIndex reflection:reflection];
}

- (id<MTLArgumentEncoder>)newArgumentEncoderWithBufferIndex:(NSUInteger)bufferIndex
                                                 reflection:(MTLAutoreleasedArgument *)reflection
                                             binaryArchives:(uintptr_t)binaryArchives
{
	(void)binaryArchives;
	return [self newArgumentEncoderWithBufferIndex:bufferIndex reflection:reflection];
}

- (id)reflectionWithOptions:(NSUInteger)options pipelineLibrary:(uintptr_t)pipelineLibrary
{
	(void)pipelineLibrary;
	return [self reflectionWithOptions:options];
}

- (id)reflectionWithOptions:(NSUInteger)options binaryArchives:(uintptr_t)binaryArchives
{
	(void)binaryArchives;
	return [self reflectionWithOptions:options];
}

// reflectionWithOptions:'s answer, handed to `completionHandler` on the creation queue, as the
// library's asynchronous makers hand theirs (RgfxRunCreation).
- (void)reflectionWithOptions:(NSUInteger)options completionHandler:(void (^)(id _Nullable))completionHandler
{
	completionHandler = [completionHandler copy];
	RgfxRunCreation(^{
		id reflection = [self reflectionWithOptions:options];
		if (completionHandler)
			completionHandler(reflection);
	});
}

@end

@implementation RgfxGLFunction (RgfxSpiForwarders)

- (id<MTLArgumentEncoder>)newArgumentEncoderWithBufferIndex:(NSUInteger)bufferIndex
                                            pipelineLibrary:(uintptr_t)pipelineLibrary
{
	(void)pipelineLibrary;
	return [self newArgumentEncoderWithBufferIndex:bufferIndex];
}

- (id<MTLArgumentEncoder>)newArgumentEncoderWithBufferIndex:(NSUInteger)bufferIndex
                                                 reflection:(MTLAutoreleasedArgument *)reflection
                                            pipelineLibrary:(uintptr_t)pipelineLibrary
{
	(void)pipelineLibrary;
	return [self newArgumentEncoderWithBufferIndex:bufferIndex reflection:reflection];
}

- (id<MTLArgumentEncoder>)newArgumentEncoderWithBufferIndex:(NSUInteger)bufferIndex
                                                 reflection:(MTLAutoreleasedArgument *)reflection
                                             binaryArchives:(uintptr_t)binaryArchives
{
	(void)binaryArchives;
	return [self newArgumentEncoderWithBufferIndex:bufferIndex reflection:reflection];
}

@end

// --- MTLBinaryArchiveSPI ---------------------------------------------------------

@implementation RgfxBinaryArchive (RgfxSpiForwarders)

- (BOOL)addComputePipelineFunctionsWithDescriptor:(MTLComputePipelineDescriptor *)descriptor
                                          options:(NSUInteger)options
                                            error:(__autoreleasing NSError **)error
{
	(void)options;
	return [self addComputePipelineFunctionsWithDescriptor:descriptor error:error];
}

- (BOOL)addRenderPipelineFunctionsWithDescriptor:(MTLRenderPipelineDescriptor *)descriptor
                                         options:(NSUInteger)options
                                           error:(__autoreleasing NSError **)error
{
	(void)options;
	return [self addRenderPipelineFunctionsWithDescriptor:descriptor error:error];
}

- (BOOL)addTileRenderPipelineFunctionsWithDescriptor:(MTLTileRenderPipelineDescriptor *)descriptor
                                             options:(NSUInteger)options
                                               error:(__autoreleasing NSError **)error
{
	(void)options;
	return [self addTileRenderPipelineFunctionsWithDescriptor:descriptor error:error];
}

- (BOOL)addMeshRenderPipelineFunctionsWithDescriptor:(MTLMeshRenderPipelineDescriptor *)descriptor
                                             options:(NSUInteger)options
                                               error:(__autoreleasing NSError **)error
{
	(void)options;
	return [self addMeshRenderPipelineFunctionsWithDescriptor:descriptor error:error];
}

- (BOOL)serializeToURL:(NSURL *)url options:(NSUInteger)options error:(__autoreleasing NSError **)error
{
	(void)options;
	return [self serializeToURL:url error:error];
}

@end

// --- MTLCommandBufferSPI ---------------------------------------------------------

@implementation RgfxCmdBuffer (RgfxSpiForwarders)

- (void)encodeWaitForEvent:(id<MTLEvent>)event value:(uint64_t)value timeout:(uint64_t)timeout
{
	(void)timeout;
	[self encodeWaitForEvent:event value:value];
}

- (void)presentDrawable:(id<MTLDrawable>)drawable options:(uintptr_t)options
{
	(void)options;
	[self presentDrawable:drawable];
}

@end

// --- MTLTextureSPI ---------------------------------------------------------------

@implementation RgfxTexture (RgfxSpiForwarders)

- (id<MTLTexture>)newTextureViewWithPixelFormat:(MTLPixelFormat)pixelFormat
                                  resourceIndex:(NSUInteger)resourceIndex
{
	(void)resourceIndex;
	return [self newTextureViewWithPixelFormat:pixelFormat];
}

- (id<MTLTexture>)newTextureViewWithPixelFormat:(MTLPixelFormat)pixelFormat
                                    textureType:(MTLTextureType)textureType
                                         levels:(NSRange)levelRange
                                         slices:(NSRange)sliceRange
                                  resourceIndex:(NSUInteger)resourceIndex
{
	(void)resourceIndex;
	return [self newTextureViewWithPixelFormat:pixelFormat
	                               textureType:textureType
	                                    levels:levelRange
	                                    slices:sliceRange];
}

- (id<MTLTexture>)newTextureViewWithPixelFormat:(MTLPixelFormat)pixelFormat
                                    textureType:(MTLTextureType)textureType
                                         levels:(NSRange)levelRange
                                         slices:(NSRange)sliceRange
                                        swizzle:(MTLTextureSwizzleChannels)swizzle
                                  resourceIndex:(NSUInteger)resourceIndex
{
	(void)resourceIndex;
	return [self newTextureViewWithPixelFormat:pixelFormat
	                               textureType:textureType
	                                    levels:levelRange
	                                    slices:sliceRange
	                                   swizzle:swizzle];
}

@end

// --- the members whose public counterparts the device does not support -----------

@implementation RgfxHeap (RgfxSpiForwarders)

- (id<MTLAccelerationStructure>)newAccelerationStructureWithSize:(NSUInteger)size
                                                          offset:(NSUInteger)offset
                                                   resourceIndex:(NSUInteger)resourceIndex
{
	RGFX_UNSUPPORTED("ray tracing");
	return nil;
}

@end

@implementation RgfxComputePipelineState (RgfxSpiForwarders)

- (id<MTLComputePipelineState>)
    newComputePipelineStateWithAdditionalBinaryFunctions:(NSArray<id<MTLFunction>> *)functions
                                         resourceIndices:(uintptr_t)resourceIndices
                                                   error:(__autoreleasing NSError **)error
{
	RGFX_UNSUPPORTED_ERROR(error, "binary functions");
	return nil;
}

@end

@implementation RgfxRenderPipelineState (RgfxSpiForwarders)

- (id<MTLRenderPipelineState>)
    newRenderPipelineStateWithAdditionalBinaryFunctions:(NSArray<id<MTLFunction>> *)vertexFunctions
                      fragmentAdditionalBinaryFunctions:(NSArray<id<MTLFunction>> *)fragmentFunctions
                                                  error:(__autoreleasing NSError **)error
{
	RGFX_UNSUPPORTED_ERROR(error, "binary functions");
	return nil;
}

@end
