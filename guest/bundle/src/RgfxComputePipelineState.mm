/*
 * RgfxComputePipelineState: see RgfxComputePipelineState.h. Our own ObjC++; no Apple
 * private SPI.
 */

#import "RgfxComputePipelineState.h"

#include <string>
#include <vector>

#import "RgfxFunction.h"
#import "RgfxMetalError.h"
#import "RgfxUnsupported.h"

#include "RgfxKnownPipelines.hpp"
#include "RgfxReplyReader.hpp"
#include "RgfxSerializer.hpp"

namespace {

// The stage-input descriptor's layout/attribute arrays expose no count, so the mapping scans
// the indices MTLStageInputOutputDescriptor addresses (0..30) and keeps the entries in use: a
// layout whose stride is non-zero, an attribute whose format is not MTLAttributeFormatInvalid.
constexpr NSUInteger kMaxStageInputBuffers = 31;
constexpr NSUInteger kMaxStageInputAttributes = 31;

/// The in-use layouts and attributes of `sd` (nil: none).
void readStageInput(MTLStageInputOutputDescriptor *sd, std::vector<RgfxVertexBufferLayout> &layouts,
                    std::vector<RgfxVertexAttribute> &attributes)
{
	if (sd == nil)
		return;
	for (NSUInteger i = 0; i < kMaxStageInputBuffers; i++) {
		MTLBufferLayoutDescriptor *l = sd.layouts[i];
		if (l.stride == 0)
			continue;
		RgfxVertexBufferLayout rl = {};
		rl.buffer_index = static_cast<uint32_t>(i);
		rl.stride = static_cast<uint32_t>(l.stride);
		rl.step_function = static_cast<uint32_t>(l.stepFunction);
		rl.step_rate = static_cast<uint32_t>(l.stepRate);
		layouts.push_back(rl);
	}
	for (NSUInteger i = 0; i < kMaxStageInputAttributes; i++) {
		MTLAttributeDescriptor *a = sd.attributes[i];
		if (a.format == MTLAttributeFormatInvalid)
			continue;
		RgfxVertexAttribute ra = {};
		ra.attribute_index = static_cast<uint32_t>(i);
		ra.format = static_cast<uint32_t>(a.format);
		ra.offset = static_cast<uint32_t>(a.offset);
		ra.buffer_index = static_cast<uint32_t>(a.bufferIndex);
		attributes.push_back(ra);
	}
}

/// The refusal of a pipeline that names `fn`, a function another device's library made: it
/// names no library the host holds.
NSError *foreignFunctionError(NSString *what, id<MTLFunction> fn)
{
	NSString *msg = [NSString stringWithFormat:@"compute pipeline: %@ %@ (%@) is not from a library of this device",
	                                           what, fn.name, NSStringFromClass([(id)fn class])];
	NSLog(@"RGFX: %@", msg);
	return RgfxMakeError(RgfxMetalErrorInvalidDescriptor, msg);
}

} // namespace

// Defined in the main implementation; declared here so the lambda in
// rgfxRegisterWithReplyReader: sees its signature.
@interface RgfxComputePipelineState ()
- (void)rgfxFillReflection:(const RgfxPipelineReflection *)r;
// A create posted without waiting takes `known`, the equal create's reflection, at once, and
// later its own answer.
- (void)rgfxTakeKnown:(const RgfxPipelineReflection &)known;
- (void)rgfxTakeAnswer:(const RgfxPipelineReflection *)r;
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
	std::shared_ptr<const RgfxAirFunction> _air; // its function's arguments; null where unknown
	// A keyed create's key, and the known pipelines its answer is kept in once accepted, until
	// it registers; a create posted without waiting has its own answer to wait for.
	std::shared_ptr<RgfxKnownPipelines> _known;
	RgfxPipelineKey _key;
	BOOL _deferred;
	RgfxPipelineAnswer _answer;
}

// The host drops the pipeline as the app lets it go, after any command buffer that names it (each
// keeps it while it runs).
- (void)dealloc
{
	[self rgfxRetireThen:nullptr];
}

// A pipeline of the function alone is one of a descriptor of it, every other field Metal's
// default. Metal's descriptor takes no nil function, so a nil one is refused here.
+ (instancetype)computePipelineStateWithFunction:(id<MTLFunction>)function
                                          device:(id<MTLDevice>)device
                                      serializer:(RgfxSerializer *)serializer
                                           error:(__autoreleasing NSError **)error
{
	return [self computePipelineStateWithFunction:function
	                                       device:device
	                                   serializer:serializer
	                                        known:nullptr
	                                        error:error];
}

+ (instancetype)computePipelineStateWithFunction:(id<MTLFunction>)function
                                          device:(id<MTLDevice>)device
                                      serializer:(RgfxSerializer *)serializer
                                           known:(std::shared_ptr<RgfxKnownPipelines>)known
                                           error:(__autoreleasing NSError **)error
{
	if (function == nil) {
		if (error)
			*error = RgfxMakeError(serializer == nullptr ? RgfxMetalErrorNoRoute
			                                             : RgfxMetalErrorInvalidDescriptor,
			                       serializer == nullptr
			                           ? @"compute pipeline: no serialiser route"
			                           : @"compute pipeline: no compute function of this device");
		return nil;
	}
	MTLComputePipelineDescriptor *descriptor = [[MTLComputePipelineDescriptor alloc] init];
	descriptor.computeFunction = function;
	return [self computePipelineStateWithDescriptor:descriptor
	                                         device:device
	                                     serializer:serializer
	                                          known:std::move(known)
	                                          error:error];
}

+ (instancetype)computePipelineStateWithDescriptor:(MTLComputePipelineDescriptor *)descriptor
                                            device:(id<MTLDevice>)device
                                        serializer:(RgfxSerializer *)serializer
                                             error:(__autoreleasing NSError **)error
{
	return [self computePipelineStateWithDescriptor:descriptor
	                                         device:device
	                                     serializer:serializer
	                                          known:nullptr
	                                          error:error];
}

+ (instancetype)computePipelineStateWithDescriptor:(MTLComputePipelineDescriptor *)descriptor
                                            device:(id<MTLDevice>)device
                                        serializer:(RgfxSerializer *)serializer
                                             known:(std::shared_ptr<RgfxKnownPipelines>)known
                                             error:(__autoreleasing NSError **)error
{
	if (serializer == nullptr) {
		if (error)
			*error = RgfxMakeError(RgfxMetalErrorNoRoute, @"compute pipeline: no serialiser route");
		return nil;
	}
	id<MTLFunction> function = descriptor.computeFunction;
	if (NSError *unspecialised = [RgfxFunction rgfxUnspecialisedError:function role:@"function"]) {
		if (error)
			*error = unspecialised;
		return nil;
	}
	if (function == nil) {
		if (error)
			*error = RgfxMakeError(RgfxMetalErrorInvalidDescriptor,
			                       @"compute pipeline: no compute function of this device");
		return nil;
	}
	if (![(id)function isKindOfClass:[RgfxFunction class]]) {
		if (error)
			*error = foreignFunctionError(@"function", function);
		return nil;
	}
	RgfxLinkedFunctionList linked;
	if (id<MTLFunction> foreign = RgfxCollectLinkedFunctions(descriptor.linkedFunctions, linked)) {
		if (error)
			*error = foreignFunctionError(@"linked function", foreign);
		return nil;
	}

	RgfxFunction *rf = (RgfxFunction *)function;
	RgfxComputePipelineParams p = {};
	p.function.library = rf.rgfxLibraryObjId;
	p.function.entry_point = [rf rgfxEntryPointUTF8] ? [rf rgfxEntryPointUTF8] : "";
	p.function.constants = [rf rgfxConstants];
	p.function.constant_count = [rf rgfxConstantCount];
	if (descriptor.threadGroupSizeIsMultipleOfThreadExecutionWidth)
		p.flags |= RGFX_COMPUTE_PIPELINE_TG_MULTIPLE_OF_EXEC_WIDTH;
	if (descriptor.supportIndirectCommandBuffers)
		p.flags |= RGFX_COMPUTE_PIPELINE_SUPPORT_ICB;
	// A kernel's function of OpenCL mode (newFunctionWithPluginData:bitcodeType:, made only where
	// the host makes such pipelines): the host links it for OpenCL mode, and answers the maximum
	// threads per threadgroup of one of reqd_work_group_size.
	if ([rf rgfxClBindings] != nullptr)
		p.flags |= RGFX_COMPUTE_PIPELINE_OPENCL_MODE;
	p.max_total_threads_per_threadgroup =
	    static_cast<uint32_t>(descriptor.maxTotalThreadsPerThreadgroup);
	p.max_call_stack_depth = static_cast<uint32_t>(descriptor.maxCallStackDepth);

	std::vector<RgfxVertexBufferLayout> layouts;
	std::vector<RgfxVertexAttribute> attributes;
	MTLStageInputOutputDescriptor *stageInput = descriptor.stageInputDescriptor;
	readStageInput(stageInput, layouts, attributes);
	p.stage_input_layouts = layouts.empty() ? nullptr : layouts.data();
	p.stage_input_layout_count = static_cast<uint32_t>(layouts.size());
	p.stage_input_attributes = attributes.empty() ? nullptr : attributes.data();
	p.stage_input_attribute_count = static_cast<uint32_t>(attributes.size());
	if (stageInput != nil) {
		p.stage_input_index_type = static_cast<uint32_t>(stageInput.indexType);
		p.stage_input_index_buffer_index = static_cast<uint32_t>(stageInput.indexBufferIndex);
	}
	p.linked = linked.refs.empty() ? nullptr : linked.refs.data();
	p.linked_count = static_cast<uint32_t>(linked.refs.size());

	RgfxPipelinePost post;
	post.known = known.get();
	RgfxCreated created = serializer->createComputePipeline(p, &post);
	if (!created.posted) {
		if (error)
			*error = RgfxMakeError(RgfxMetalErrorRingFull,
			                       @"compute pipeline: create did not fit the ring");
		return nil;
	}
	if (error)
		*error = nil;
	RgfxComputePipelineState *state =
	    [[RgfxComputePipelineState alloc] initWithObjId:created.obj_id
	                                              seqno:created.seqno
	                                              label:descriptor.label
	                                             device:device
	                                         serializer:serializer];
	[state rgfxTakeDescriptor:descriptor flags:p.flags];
	if (state != nil) {
		state->_air = [rf rgfxAirFunction];
		if (post.deferred) {
			[state rgfxTakeKnown:post.reflection];
		} else if (post.keyed) {
			state->_known = known;
			state->_key = post.key;
		}
	}
	return state;
}

- (void)rgfxTakeKnown:(const RgfxPipelineReflection &)known
{
	_reflection = known;
	_reflection.header.seqno = _seqno;
	_reflection.obj_id = self.rgfxObjId;
	_reflection.gpu_resource_id = 0; // its own, which gpuResourceID waits for
	_haveReflection = YES;
	_deferred = YES;
}

- (BOOL)rgfxDeferred
{
	return _deferred;
}

- (std::shared_ptr<const RgfxAirFunction>)rgfxAirFunction
{
	return _air;
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
	if (_deferred) {
		// Its own answer: the resource ID gpuResourceID waits for.
		reader->expectReflection(_seqno, [weakSelf](const RgfxPipelineReflection &r) {
			RgfxComputePipelineState *strongSelf = weakSelf;
			if (strongSelf != nil)
				[strongSelf rgfxTakeAnswer:&r];
		});
		return;
	}
	// A keyed create's answer is kept once the device accepted it, so equal creates need not wait.
	std::shared_ptr<RgfxKnownPipelines> known = std::move(_known);
	const RgfxPipelineKey key = _key;
	reader->expectReflection(_seqno, [weakSelf, known, key](const RgfxPipelineReflection &r) {
		if (known)
			known->learn(key, r);
		RgfxComputePipelineState *strongSelf = weakSelf;
		if (strongSelf != nil) {
			[strongSelf rgfxFillReflection:&r];
		}
	});
}

// A create posted without waiting: its own answer, whose resource ID gpuResourceID answers (as
// RgfxRenderPipelineState's).
- (void)rgfxTakeAnswer:(const RgfxPipelineReflection *)r
{
	_answer.take(*r);
	if (r->status != RGFX_PIPELINE_OK)
		NSLog(@"RGFX: the device refused compute pipeline %llu, made as an equal one it accepted",
		      static_cast<unsigned long long>(self.rgfxObjId));
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
	if (_deferred) {
		RgfxSerializer *serializer = [self serializer];
		const uint64_t seqno = _seqno;
		return (MTLResourceID){ _answer.resourceId([serializer, seqno] {
			if (serializer != nullptr)
				serializer->awaitAnswer(seqno);
		}) };
	}
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

// --- Metal's private members, as its pipelines answer them (measured: macOS 27, M3 Max) ---

// 0 for every pipeline, one that supports indirect command buffers too.
- (uint64_t)gpuAddress
{
	return 0;
}

// Unique and stable for the object while it lives: Metal's is an address inside its pipeline
// object, ours the object's address.
- (uint64_t)uniqueIdentifier
{
	return static_cast<uint64_t>(reinterpret_cast<uintptr_t>((__bridge void *)self));
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
