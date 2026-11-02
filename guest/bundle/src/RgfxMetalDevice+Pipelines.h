/*
 * RgfxMetalDevice (Pipelines): slice 2 of the device shell's category split
 * (RgfxMetalDevice.h). It owns the library, function, pipeline and argument-encoder
 * creation selectors of <MTLDevice>, each a thin forwarder that routes through
 * `[self serializer]` to a wrapper factory (RgfxLibrary, RgfxRenderPipelineState,
 * RgfxComputePipelineState, RgfxArgumentEncoder) — the RgfxBuffer pattern.
 *
 * The descriptor→params mapping lives in the wrapper factories (so it is unit-tested
 * off the device); this category only reads bytes (data/URL), builds the completion
 * closures, and compiles source at run time through Metal's own frontend over the device
 * (RgfxCompileOverMetal).
 *
 * Reflection: a pipeline create is answered by the host with an RgfxPipelineReflection
 * on the completion ring, read by a later transport piece; the reflection out-params
 * of the `…reflection:` selectors are therefore set to nil (documented), and the
 * pipeline state is returned as usual, holding objId + seqno.
 *
 * Our own ObjC++; the only Apple-private names are in RgfxMetalPrivate.h.
 */

#ifndef RGFX_METAL_DEVICE_PIPELINES_H_
#define RGFX_METAL_DEVICE_PIPELINES_H_

#import "RgfxBinaryArchive.h"
#import "RgfxMetalDevice.h"

NS_ASSUME_NONNULL_BEGIN

@interface RgfxMetalDevice (Pipelines) <RgfxStitchingDevice>

// Libraries.
- (nullable id<MTLLibrary>)newLibraryWithData:(dispatch_data_t)data
                                        error:(__autoreleasing NSError *_Nullable *_Nullable)error;
- (nullable id<MTLLibrary>)newLibraryWithURL:(NSURL *)url
                                       error:(__autoreleasing NSError *_Nullable *_Nullable)error;
- (nullable id<MTLLibrary>)newDefaultLibrary;
- (nullable id<MTLLibrary>)newDefaultLibraryWithBundle:(NSBundle *)bundle
                                                 error:(__autoreleasing NSError *_Nullable *_Nullable)
                                                           error;
- (nullable id<MTLLibrary>)newLibraryWithSource:(NSString *)source
                                        options:(nullable MTLCompileOptions *)options
                                          error:(__autoreleasing NSError *_Nullable *_Nullable)error;
- (void)newLibraryWithSource:(NSString *)source
                     options:(nullable MTLCompileOptions *)options
           completionHandler:(MTLNewLibraryCompletionHandler)completionHandler;

// Render pipelines.
- (nullable id<MTLRenderPipelineState>)
    newRenderPipelineStateWithDescriptor:(MTLRenderPipelineDescriptor *)descriptor
                                   error:(__autoreleasing NSError *_Nullable *_Nullable)error;
- (nullable id<MTLRenderPipelineState>)
    newRenderPipelineStateWithDescriptor:(MTLRenderPipelineDescriptor *)descriptor
                                 options:(MTLPipelineOption)options
                              reflection:(MTLAutoreleasedRenderPipelineReflection *_Nullable)reflection
                                   error:(__autoreleasing NSError *_Nullable *_Nullable)error;
- (void)newRenderPipelineStateWithDescriptor:(MTLRenderPipelineDescriptor *)descriptor
                           completionHandler:
                               (MTLNewRenderPipelineStateCompletionHandler)completionHandler;
- (void)newRenderPipelineStateWithDescriptor:(MTLRenderPipelineDescriptor *)descriptor
                                     options:(MTLPipelineOption)options
                           completionHandler:
                               (MTLNewRenderPipelineStateWithReflectionCompletionHandler)
                                   completionHandler;

// Compute pipelines.
- (nullable id<MTLComputePipelineState>)
    newComputePipelineStateWithFunction:(id<MTLFunction>)computeFunction
                                  error:(__autoreleasing NSError *_Nullable *_Nullable)error;
- (nullable id<MTLComputePipelineState>)
    newComputePipelineStateWithFunction:(id<MTLFunction>)computeFunction
                                options:(MTLPipelineOption)options
                             reflection:(MTLAutoreleasedComputePipelineReflection *_Nullable)reflection
                                  error:(__autoreleasing NSError *_Nullable *_Nullable)error;
- (void)newComputePipelineStateWithFunction:(id<MTLFunction>)computeFunction
                          completionHandler:
                              (MTLNewComputePipelineStateCompletionHandler)completionHandler;
- (void)newComputePipelineStateWithFunction:(id<MTLFunction>)computeFunction
                                    options:(MTLPipelineOption)options
                          completionHandler:
                              (MTLNewComputePipelineStateWithReflectionCompletionHandler)
                                  completionHandler;
- (nullable id<MTLComputePipelineState>)
    newComputePipelineStateWithDescriptor:(MTLComputePipelineDescriptor *)descriptor
                                  options:(MTLPipelineOption)options
                               reflection:
                                   (MTLAutoreleasedComputePipelineReflection *_Nullable)reflection
                                    error:(__autoreleasing NSError *_Nullable *_Nullable)error;
- (void)newComputePipelineStateWithDescriptor:(MTLComputePipelineDescriptor *)descriptor
                                      options:(MTLPipelineOption)options
                            completionHandler:
                                (MTLNewComputePipelineStateWithReflectionCompletionHandler)
                                    completionHandler;
- (nullable id<MTLComputePipelineState>)
    newComputePipelineStateWithDescriptor:(MTLComputePipelineDescriptor *)descriptor
                                    error:(__autoreleasing NSError *_Nullable *_Nullable)error;
- (void)newComputePipelineStateWithDescriptor:(MTLComputePipelineDescriptor *)descriptor
                            completionHandler:
                                (MTLNewComputePipelineStateCompletionHandler)completionHandler;

// Argument encoders.
- (nullable id<MTLArgumentEncoder>)
    newArgumentEncoderWithArguments:(NSArray<MTLArgumentDescriptor *> *)arguments;

@end

NS_ASSUME_NONNULL_END

#endif // RGFX_METAL_DEVICE_PIPELINES_H_
