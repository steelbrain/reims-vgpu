/*
 * RgfxComputePipelineState: see RgfxComputePipelineState.h. Our own ObjC++; no Apple
 * private SPI.
 */

#import "RgfxComputePipelineState.h"

#include <string>

#import "RgfxFunction.h"
#import "RgfxMetalError.h"
#import "RgfxUnsupported.h"

#include "RgfxReplyReader.hpp"
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
	if (![(id)fn isKindOfClass:[RgfxFunction class]]) {
		// Another device's library made it: it names no library the host holds.
		NSLog(@"RGFX: compute pipeline: function %@ (%@) is not from a library of this device",
		      fn.name, NSStringFromClass([(id)fn class]));
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
// Defined in the main implementation; declared here so the lambda in
// rgfxRegisterWithReplyReader: sees its signature.
- (void)rgfxFillReflection:(const RgfxPipelineReflection *)r;
// Keeps what the descriptor asked that the pipeline answers (rgfx*Desc ivars).
- (void)rgfxTakeDescriptor:(MTLComputePipelineDescriptor *)descriptor flags:(uint32_t)flags;
@end

@implementation RgfxComputePipelineState {
	uint64_t _seqno;
	// What the create asked that the pipeline answers: its RGFX_COMPUTE_PIPELINE_* flags, the
	// required threadgroup size and the forward-progress use (Metal's defaults for a pipeline
	// made from a function alone).
	uint32_t _createFlags;
	MTLSize _requiredThreadsPerThreadgroup;
	NSInteger _forwardProgressUsage;
	// The host's reflection, filled once RGFX_EVT_PIPELINE_REFLECTION for `_seqno`
	// arrives; until then `_haveReflection` is NO and the getters answer placeholders.
	RgfxPipelineReflection _reflection;
	BOOL _haveReflection;
}

// The host drops the pipeline once every command buffer in flight now has completed, as one may
// still name it.
- (void)dealloc
{
	[self rgfxRetireThen:nullptr];
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
	RgfxComputePipelineState *state =
	    [self createWithFunction:descriptor.computeFunction
	                       flags:flags
	             maxTotalThreads:static_cast<uint32_t>(descriptor.maxTotalThreadsPerThreadgroup)
	                       label:descriptor.label
	                      device:device
	                  serializer:serializer
	                       error:error];
	[state rgfxTakeDescriptor:descriptor flags:flags];
	return state;
}

- (void)rgfxTakeDescriptor:(MTLComputePipelineDescriptor *)descriptor flags:(uint32_t)flags
{
	_createFlags = flags;
	// Both are newer than some guests' Metal (macOS 26 and 27): absent, Metal's defaults stand.
	if ([descriptor respondsToSelector:@selector(requiredThreadsPerThreadgroup)])
		_requiredThreadsPerThreadgroup = descriptor.requiredThreadsPerThreadgroup;
	if ([descriptor respondsToSelector:@selector(forwardProgressUsage)])
		_forwardProgressUsage = static_cast<NSInteger>(descriptor.forwardProgressUsage);
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
	if (NSError *unspecialised = [RgfxFunction rgfxUnspecialisedError:function role:@"function"]) {
		if (error)
			*error = unspecialised;
		return nil;
	}
	RgfxComputePipelineParams p = {};
	std::string nameStore;
	if (!fillComputeParams(p, function, flags, maxTotalThreads, nameStore)) {
		if (error)
			*error = RgfxMakeError(RgfxMetalErrorInvalidDescriptor,
			                       @"compute pipeline: no compute function of this device");
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

- (void)rgfxRegisterWithReplyReader:(RgfxReplyReader *)reader
{
	if (reader == nullptr) {
		return;
	}
	__weak RgfxComputePipelineState *weakSelf = self;
	reader->expectReflection(_seqno, [weakSelf](const RgfxPipelineReflection &r) {
		RgfxComputePipelineState *strongSelf = weakSelf;
		if (strongSelf != nil) {
			[strongSelf rgfxFillReflection:&r];
		}
	});
}

- (void)rgfxFillReflection:(const RgfxPipelineReflection *)r
{
	_reflection = *r;
	_haveReflection = YES;
}

- (BOOL)rgfxRefused
{
	return _haveReflection && _reflection.status == RGFX_PIPELINE_REFUSED;
}

// --- MTLComputePipelineState: reflection --------------------------------------
// The host's RgfxPipelineReflection reply is read by the reply reader. Before it
// arrives these answer documented placeholders; after it they report the host's
// values.

- (NSUInteger)maxTotalThreadsPerThreadgroup
{
	if (_haveReflection) {
		return static_cast<NSUInteger>(_reflection.max_total_threads_per_threadgroup);
	}
	return 1024; // Metal's common threadgroup floor until reflection is read
}

- (NSUInteger)threadExecutionWidth
{
	if (_haveReflection) {
		return _reflection.thread_execution_width;
	}
	return 32; // the common Apple-GPU SIMD width until reflection is read
}

- (NSUInteger)staticThreadgroupMemoryLength
{
	if (_haveReflection) {
		return static_cast<NSUInteger>(_reflection.static_threadgroup_memory_length);
	}
	return 0;
}

- (MTLResourceID)gpuResourceID
{
	if (_haveReflection) {
		return (MTLResourceID){ _reflection.gpu_resource_id };
	}
	return (MTLResourceID){0}; // 0 until the reflection reply carries the real id
}

// --- what the create asked ----------------------------------------------------------

- (BOOL)supportIndirectCommandBuffers
{
	if (_haveReflection)
		return (_reflection.flags & RGFX_PIPELINE_REFLECT_SUPPORT_ICB) != 0;
	return (_createFlags & RGFX_COMPUTE_PIPELINE_SUPPORT_ICB) != 0;
}

- (MTLSize)requiredThreadsPerThreadgroup
{
	return _requiredThreadsPerThreadgroup;
}

- (MTLForwardProgressUsage)forwardProgressUsage
{
	return static_cast<MTLForwardProgressUsage>(_forwardProgressUsage);
}

// The device validates no shader, so no pipeline of it runs with validation.
- (MTLShaderValidation)shaderValidation
{
	return MTLShaderValidationDisabled;
}

// Metal's answer for every pipeline a device (not a Metal 4 compiler) makes.
- (MTLComputePipelineReflection *)reflection
{
	return nil;
}

// --- not on the wire ------------------------------------------------------------------
// The host answers no pipeline's memory, imageblock or persistent-threadgroup queries yet
// (RgfxPipelineReflection carries none): 0, logged.

- (NSUInteger)allocatedSize
{
	RGFX_UNSUPPORTED("a pipeline's allocated size (not in the host's reflection)");
	return 0;
}

- (NSUInteger)imageblockMemoryLengthForDimensions:(MTLSize)imageblockDimensions
{
	RGFX_UNSUPPORTED("imageblock memory queries (not in the host's reflection)");
	return 0;
}

- (NSUInteger)recommendedPersistentThreadgroupsPerGridForThreadsPerThreadgroup:
    (MTLSize)threadsPerThreadgroup
{
	RGFX_UNSUPPORTED("persistent threadgroup queries (not in the host's reflection)");
	return 0;
}

// --- unsupported: function pointers, binary functions and function tables ---------------

- (id<MTLFunctionHandle>)functionHandleWithFunction:(id<MTLFunction>)function
{
	RGFX_UNSUPPORTED("function pointers");
	return nil;
}

- (id<MTLFunctionHandle>)functionHandleWithName:(NSString *)name
{
	RGFX_UNSUPPORTED("function pointers");
	return nil;
}

- (id<MTLFunctionHandle>)functionHandleWithBinaryFunction:(id<MTL4BinaryFunction>)function
{
	RGFX_UNSUPPORTED("function pointers");
	return nil;
}

- (id<MTLComputePipelineState>)
    newComputePipelineStateWithAdditionalBinaryFunctions:(NSArray<id<MTLFunction>> *)functions
                                                   error:(__autoreleasing NSError **)error
{
	RGFX_UNSUPPORTED_ERROR(error, "binary functions");
	return nil;
}

- (id<MTLComputePipelineState>)
    newComputePipelineStateWithBinaryFunctions:(NSArray<id<MTL4BinaryFunction>> *)additionalBinaryFunctions
                                         error:(NSError **)error
{
	RGFX_UNSUPPORTED_ERROR(error, "binary functions");
	return nil;
}

- (id<MTLVisibleFunctionTable>)newVisibleFunctionTableWithDescriptor:
    (MTLVisibleFunctionTableDescriptor *)descriptor
{
	RGFX_UNSUPPORTED("function pointers");
	return nil;
}

- (id<MTLIntersectionFunctionTable>)newIntersectionFunctionTableWithDescriptor:
    (MTLIntersectionFunctionTableDescriptor *)descriptor
{
	RGFX_UNSUPPORTED("ray tracing");
	return nil;
}

@end
