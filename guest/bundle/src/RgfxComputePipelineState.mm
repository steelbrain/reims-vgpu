/*
 * RgfxComputePipelineState: see RgfxComputePipelineState.h. Our own ObjC++; no Apple
 * private SPI.
 */

#import "RgfxComputePipelineState.h"

#include <string>

#import "RgfxFunction.h"
#import "RgfxMetalError.h"

#include "RgfxSerializer.hpp"

namespace {

/// Builds an RgfxComputePipelineParams from the compute function and the descriptor
/// knobs. `fn` must be an RgfxFunction (what the bundle vends); nil is a refusal the
/// caller reports. `label` out-param keeps the entry-point C string alive.
bool fillComputeParams(RgfxComputePipelineParams &p, id<MTLFunction> fn, uint32_t flags,
                       uint32_t maxTotalThreads, std::string &nameStore)
{
	if (fn == nil) {
		return false;
	}
	RgfxFunction *rf = (RgfxFunction *)fn;
	p = RgfxComputePipelineParams{};
	nameStore = [rf rgfxEntryPointUTF8] ? [rf rgfxEntryPointUTF8] : "";
	p.function.library = rf.rgfxLibraryObjId;
	p.function.entry_point = nameStore.c_str();
	p.function.constants = [rf rgfxConstants];
	p.function.constant_count = [rf rgfxConstantCount];
	p.flags = flags;
	p.max_total_threads_per_threadgroup = maxTotalThreads;
	p.max_call_stack_depth = 0;
	return true;
}

} // namespace

// The private create both public factories funnel through.
@interface RgfxComputePipelineState ()
+ (nullable instancetype)createWithFunction:(id<MTLFunction>)function
                                      flags:(uint32_t)flags
                            maxTotalThreads:(uint32_t)maxTotalThreads
                                      label:(nullable NSString *)label
                                     device:(nullable id<MTLDevice>)device
                                 serializer:(nullable RgfxSerializer *)serializer
                                      error:(__autoreleasing NSError *_Nullable *_Nullable)error;
@end

@implementation RgfxComputePipelineState {
	uint64_t _seqno;
}

+ (instancetype)computePipelineStateWithFunction:(id<MTLFunction>)function
                                          device:(id<MTLDevice>)device
                                      serializer:(RgfxSerializer *)serializer
                                           error:(__autoreleasing NSError **)error
{
	return [self createWithFunction:function
	                          flags:0
	                maxTotalThreads:0
	                          label:nil
	                         device:device
	                     serializer:serializer
	                          error:error];
}

+ (instancetype)computePipelineStateWithDescriptor:(MTLComputePipelineDescriptor *)descriptor
                                            device:(id<MTLDevice>)device
                                        serializer:(RgfxSerializer *)serializer
                                             error:(__autoreleasing NSError **)error
{
	uint32_t flags = 0;
	if (descriptor.threadGroupSizeIsMultipleOfThreadExecutionWidth)
		flags |= RGFX_COMPUTE_PIPELINE_TG_MULTIPLE_OF_EXEC_WIDTH;
	if (descriptor.supportIndirectCommandBuffers)
		flags |= RGFX_COMPUTE_PIPELINE_SUPPORT_ICB;
	return [self createWithFunction:descriptor.computeFunction
	                          flags:flags
	                maxTotalThreads:static_cast<uint32_t>(descriptor.maxTotalThreadsPerThreadgroup)
	                          label:descriptor.label
	                         device:device
	                     serializer:serializer
	                          error:error];
}

+ (instancetype)createWithFunction:(id<MTLFunction>)function
                             flags:(uint32_t)flags
                   maxTotalThreads:(uint32_t)maxTotalThreads
                             label:(NSString *)label
                            device:(id<MTLDevice>)device
                        serializer:(RgfxSerializer *)serializer
                             error:(__autoreleasing NSError **)error
{
	if (serializer == nullptr) {
		if (error)
			*error = RgfxMakeError(RgfxMetalErrorNoRoute, @"compute pipeline: no serialiser route");
		return nil;
	}
	RgfxComputePipelineParams p = {};
	std::string nameStore;
	if (!fillComputeParams(p, function, flags, maxTotalThreads, nameStore)) {
		if (error)
			*error = RgfxMakeError(RgfxMetalErrorInvalidDescriptor,
			                       @"compute pipeline: no compute function");
		return nil;
	}
	RgfxCreated created = serializer->createComputePipeline(p);
	if (!created.posted) {
		if (error)
			*error = RgfxMakeError(RgfxMetalErrorRingFull,
			                       @"compute pipeline: create did not fit the ring");
		return nil;
	}
	if (error)
		*error = nil;
	return [[RgfxComputePipelineState alloc] initWithObjId:created.obj_id
	                                                 seqno:created.seqno
	                                                 label:label
	                                                device:device
	                                            serializer:serializer];
}

- (instancetype)initWithObjId:(uint64_t)objId
                        seqno:(uint64_t)seqno
                        label:(NSString *)label
                       device:(id<MTLDevice>)device
                   serializer:(RgfxSerializer *)serializer
{
	self = [super initWithObjId:objId device:device serializer:serializer];
	if (self) {
		_seqno = seqno;
		self.label = label;
	}
	return self;
}

- (uint64_t)rgfxSeqno
{
	return _seqno;
}

// --- MTLComputePipelineState: reflection (DOCUMENTED PLACEHOLDERS) ------------
// As with the render pipeline, the host's RgfxPipelineReflection reply is read by a
// later transport piece; until then these answer documented placeholders rather than
// fabricating the host's values.

- (NSUInteger)maxTotalThreadsPerThreadgroup
{
	return 1024; // Metal's common threadgroup floor until reflection is read
}

- (NSUInteger)threadExecutionWidth
{
	return 32; // the common Apple-GPU SIMD width until reflection is read
}

- (NSUInteger)staticThreadgroupMemoryLength
{
	return 0;
}

- (MTLResourceID)gpuResourceID
{
	return (MTLResourceID){0}; // 0 until the reflection reply carries the real id
}

@end
