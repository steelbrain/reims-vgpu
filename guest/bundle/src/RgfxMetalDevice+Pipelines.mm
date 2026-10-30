/*
 * RgfxMetalDevice (Pipelines): see RgfxMetalDevice+Pipelines.h. Slice 2 of the device
 * shell. Our own ObjC++; the only Apple-private names are in RgfxMetalPrivate.h.
 */

#import "RgfxMetalDevice+Pipelines.h"

#import "RgfxArgumentEncoder.h"
#import "RgfxComputePipelineState.h"
#import "RgfxLibrary.h"
#import "RgfxMetalError.h"
#import "RgfxRenderPipelineState.h"

#include "RgfxSerializer.hpp"

@implementation RgfxMetalDevice (Pipelines)

// --- libraries ---------------------------------------------------------------

- (id<MTLLibrary>)newLibraryWithData:(dispatch_data_t)data
                               error:(__autoreleasing NSError **)error
{
	if (data == nil) {
		if (error)
			*error = RgfxMakeError(RgfxMetalErrorInvalidDescriptor, @"newLibraryWithData: nil data");
		return nil;
	}
	// Map the (possibly non-contiguous) dispatch_data_t to one byte range, then
	// forward it to the host unchanged.
	const void *bytes = nullptr;
	size_t size = 0;
	dispatch_data_t map = dispatch_data_create_map(data, &bytes, &size);
	id<MTLLibrary> lib = [RgfxLibrary libraryWithData:bytes
	                                           length:size
	                                           device:self
	                                       serializer:[self serializer]];
	(void)map; // released by ARC; `bytes` stayed valid for the create above
	if (lib == nil && error)
		*error = RgfxMakeError([self serializer] ? RgfxMetalErrorRingFull : RgfxMetalErrorNoRoute,
		                       @"newLibraryWithData: create failed");
	return lib;
}

- (id<MTLLibrary>)newLibraryWithURL:(NSURL *)url error:(__autoreleasing NSError **)error
{
	NSError *readError = nil;
	NSData *data = [NSData dataWithContentsOfURL:url options:0 error:&readError];
	if (data == nil) {
		if (error)
			*error = readError ?: RgfxMakeError(RgfxMetalErrorInvalidDescriptor,
			                                    @"newLibraryWithURL: could not read URL");
		return nil;
	}
	id<MTLLibrary> lib = [RgfxLibrary libraryWithData:data.bytes
	                                           length:data.length
	                                           device:self
	                                       serializer:[self serializer]];
	if (lib == nil && error)
		*error = RgfxMakeError([self serializer] ? RgfxMetalErrorRingFull : RgfxMetalErrorNoRoute,
		                       @"newLibraryWithURL: create failed");
	return lib;
}

- (id<MTLLibrary>)newDefaultLibrary
{
	// DOCUMENTED STUB. There is no default library baked into the bundle; the guest
	// provides libraries by data/URL. nil is the documented answer, as Metal allows
	// when a process has no default.metallib.
	return nil;
}

- (id<MTLLibrary>)newDefaultLibraryWithBundle:(NSBundle *)bundle
                                        error:(__autoreleasing NSError **)error
{
	(void)bundle;
	if (error)
		*error = RgfxMakeError(RgfxMetalErrorUnsupported,
		                       @"newDefaultLibraryWithBundle: no default library in the bundle");
	return nil;
}

- (id<MTLLibrary>)newLibraryWithSource:(NSString *)source
                               options:(MTLCompileOptions *)options
                                 error:(__autoreleasing NSError **)error
{
	// DOCUMENTED UNSUPPORTED. Source→AIR needs the Metal compiler, which runs on the
	// host; the guest shell carries no compiler. We refuse rather than fabricate an
	// empty library — the way the host device-info answers an unsupported query. A
	// host-side compile path would land its own wire message.
	(void)source;
	(void)options;
	if (error)
		*error = RgfxMakeError(RgfxMetalErrorUnsupported,
		                       @"newLibraryWithSource: runtime source compilation is not supported; "
		                       @"provide a prebuilt .metallib via newLibraryWithData:/WithURL:");
	return nil;
}

- (void)newLibraryWithSource:(NSString *)source
                     options:(MTLCompileOptions *)options
           completionHandler:(MTLNewLibraryCompletionHandler)completionHandler
{
	NSError *err = nil;
	id<MTLLibrary> lib = [self newLibraryWithSource:source options:options error:&err];
	if (completionHandler)
		completionHandler(lib, err);
}

// --- render pipelines --------------------------------------------------------

- (id<MTLRenderPipelineState>)
    newRenderPipelineStateWithDescriptor:(MTLRenderPipelineDescriptor *)descriptor
                                   error:(__autoreleasing NSError **)error
{
	return [RgfxRenderPipelineState renderPipelineStateWithDescriptor:descriptor
	                                                          device:self
	                                                      serializer:[self serializer]
	                                                           error:error];
}

- (id<MTLRenderPipelineState>)
    newRenderPipelineStateWithDescriptor:(MTLRenderPipelineDescriptor *)descriptor
                                 options:(MTLPipelineOption)options
                              reflection:(MTLAutoreleasedRenderPipelineReflection *)reflection
                                   error:(__autoreleasing NSError **)error
{
	(void)options;
	// The reflection is answered by the host over the completion ring (reader is a
	// later transport piece), so there is none to hand back synchronously.
	if (reflection)
		*reflection = nil;
	return [RgfxRenderPipelineState renderPipelineStateWithDescriptor:descriptor
	                                                          device:self
	                                                      serializer:[self serializer]
	                                                           error:error];
}

- (void)newRenderPipelineStateWithDescriptor:(MTLRenderPipelineDescriptor *)descriptor
                           completionHandler:
                               (MTLNewRenderPipelineStateCompletionHandler)completionHandler
{
	NSError *err = nil;
	id<MTLRenderPipelineState> state = [self newRenderPipelineStateWithDescriptor:descriptor
	                                                                       error:&err];
	if (completionHandler)
		completionHandler(state, err);
}

- (void)newRenderPipelineStateWithDescriptor:(MTLRenderPipelineDescriptor *)descriptor
                                     options:(MTLPipelineOption)options
                           completionHandler:
                               (MTLNewRenderPipelineStateWithReflectionCompletionHandler)
                                   completionHandler
{
	(void)options;
	NSError *err = nil;
	id<MTLRenderPipelineState> state = [self newRenderPipelineStateWithDescriptor:descriptor
	                                                                       error:&err];
	if (completionHandler)
		completionHandler(state, nil, err); // reflection: see the error: variant above
}

// --- compute pipelines -------------------------------------------------------

- (id<MTLComputePipelineState>)newComputePipelineStateWithFunction:(id<MTLFunction>)computeFunction
                                                            error:(__autoreleasing NSError **)error
{
	return [RgfxComputePipelineState computePipelineStateWithFunction:computeFunction
	                                                          device:self
	                                                      serializer:[self serializer]
	                                                           error:error];
}

- (id<MTLComputePipelineState>)
    newComputePipelineStateWithFunction:(id<MTLFunction>)computeFunction
                                options:(MTLPipelineOption)options
                             reflection:(MTLAutoreleasedComputePipelineReflection *)reflection
                                  error:(__autoreleasing NSError **)error
{
	(void)options;
	if (reflection)
		*reflection = nil;
	return [RgfxComputePipelineState computePipelineStateWithFunction:computeFunction
	                                                          device:self
	                                                      serializer:[self serializer]
	                                                           error:error];
}

- (void)newComputePipelineStateWithFunction:(id<MTLFunction>)computeFunction
                          completionHandler:
                              (MTLNewComputePipelineStateCompletionHandler)completionHandler
{
	NSError *err = nil;
	id<MTLComputePipelineState> state = [self newComputePipelineStateWithFunction:computeFunction
	                                                                       error:&err];
	if (completionHandler)
		completionHandler(state, err);
}

- (void)newComputePipelineStateWithFunction:(id<MTLFunction>)computeFunction
                                    options:(MTLPipelineOption)options
                          completionHandler:
                              (MTLNewComputePipelineStateWithReflectionCompletionHandler)
                                  completionHandler
{
	(void)options;
	NSError *err = nil;
	id<MTLComputePipelineState> state = [self newComputePipelineStateWithFunction:computeFunction
	                                                                       error:&err];
	if (completionHandler)
		completionHandler(state, nil, err);
}

- (id<MTLComputePipelineState>)
    newComputePipelineStateWithDescriptor:(MTLComputePipelineDescriptor *)descriptor
                                  options:(MTLPipelineOption)options
                               reflection:(MTLAutoreleasedComputePipelineReflection *)reflection
                                    error:(__autoreleasing NSError **)error
{
	(void)options;
	if (reflection)
		*reflection = nil;
	return [RgfxComputePipelineState computePipelineStateWithDescriptor:descriptor
	                                                            device:self
	                                                        serializer:[self serializer]
	                                                             error:error];
}

- (void)newComputePipelineStateWithDescriptor:(MTLComputePipelineDescriptor *)descriptor
                                      options:(MTLPipelineOption)options
                            completionHandler:
                                (MTLNewComputePipelineStateWithReflectionCompletionHandler)
                                    completionHandler
{
	(void)options;
	NSError *err = nil;
	id<MTLComputePipelineState> state =
	    [RgfxComputePipelineState computePipelineStateWithDescriptor:descriptor
	                                                         device:self
	                                                     serializer:[self serializer]
	                                                          error:&err];
	if (completionHandler)
		completionHandler(state, nil, err);
}

// --- argument encoders -------------------------------------------------------

- (id<MTLArgumentEncoder>)
    newArgumentEncoderWithArguments:(NSArray<MTLArgumentDescriptor *> *)arguments
{
	return [RgfxArgumentEncoder argumentEncoderWithArguments:arguments
	                                                 device:self
	                                             serializer:[self serializer]];
}

@end
