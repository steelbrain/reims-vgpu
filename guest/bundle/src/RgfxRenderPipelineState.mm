/*
 * RgfxRenderPipelineState: see RgfxRenderPipelineState.h. Our own ObjC++. Of Apple's private
 * SPI it reads only the descriptor's private render state, by key, where the host takes it.
 */

#import "RgfxRenderPipelineState.h"
#include "RgfxLogOnce.hpp"
#import "RgfxUnsupported.h"

#import <objc/message.h>

#include <cstring>
#include <vector>

#import "RgfxFunction.h"
#import "RgfxMetalError.h"

#include "RgfxKnownPipelines.hpp"
#include "RgfxReplyReader.hpp"
#include "RgfxSerializer.hpp"

// The reflection-fill entry the reader's handler calls (defined below); declared here
// so the lambda in rgfxRegisterWithReplyReader: sees its signature.
@interface RgfxRenderPipelineState ()
- (void)rgfxFillReflection:(const RgfxPipelineReply *)r;
// A create posted without waiting takes `known`, the equal create's reflection, at once, and
// later its own answer.
- (void)rgfxTakeKnown:(const RgfxPipelineReflection &)known;
- (void)rgfxTakeAnswer:(const RgfxPipelineReflection *)r;
@end

namespace {

// The vertex descriptor's layout/attribute arrays expose no count (a known
// pipeline-descriptor fidelity gap), so the mapping scans a fixed range of indices
// and keeps the entries that differ from the default. MTLVertexDescriptor addresses
// buffer and attribute indices 0..30. A layout is in use when its stride is
// non-zero; an attribute when its format is not MTLVertexFormatInvalid.
constexpr NSUInteger kMaxVertexBuffers = 31;
constexpr NSUInteger kMaxVertexAttributes = 31;

/// Fills `fr` from an id<MTLFunction>, which must be an RgfxFunction (what the bundle
/// vends). A nil function leaves `fr` naming no function (library 0, null name).
/// Answers false for a function another device's library made: it names no library
/// the host holds, so the create is refused rather than sent.
bool fillFunctionRef(RgfxFunctionRef &fr, id<MTLFunction> fn)
{
	fr = RgfxFunctionRef{};
	if (fn == nil) {
		return true;
	}
	if (![(id)fn isKindOfClass:[RgfxFunction class]]) {
		return false;
	}
	RgfxFunction *rf = (RgfxFunction *)fn;
	fr.library = rf.rgfxLibraryObjId;
	fr.entry_point = [rf rgfxEntryPointUTF8];
	fr.constants = [rf rgfxConstants];
	fr.constant_count = [rf rgfxConstantCount];
	return true;
}

/// The arguments of a stage's function `fn`, an RgfxFunction or nil: none for no function, which
/// binds nothing; null where they are unknown.
std::shared_ptr<const RgfxAirFunction> stageArguments(id<MTLFunction> fn)
{
	if (fn == nil)
		return std::make_shared<const RgfxAirFunction>();
	return [(RgfxFunction *)fn rgfxAirFunction];
}

/// Fills `colors` from a descriptor's colour attachments, whose array has no count either: one is
/// in use when its pixel format is not Invalid. Metal supports up to 8 (the wire's
/// RgfxColorAttachmentDesc colors[8]). Answers the count, the highest in-use index + 1, so a sparse
/// set keeps its slots.
uint32_t fillColors(MTLRenderPipelineColorAttachmentDescriptorArray *attachments,
                    RgfxColorAttachmentDesc (&colors)[8])
{
	uint32_t ncolor = 0;
	for (uint32_t i = 0; i < 8; i++) {
		MTLRenderPipelineColorAttachmentDescriptor *ca = attachments[i];
		if (ca.pixelFormat == MTLPixelFormatInvalid) {
			continue;
		}
		RgfxColorAttachmentDesc &d = colors[i];
		d.pixel_format = static_cast<uint32_t>(ca.pixelFormat);
		d.blending_enabled = ca.isBlendingEnabled ? 1 : 0;
		d.source_rgb_blend_factor = static_cast<uint32_t>(ca.sourceRGBBlendFactor);
		d.destination_rgb_blend_factor = static_cast<uint32_t>(ca.destinationRGBBlendFactor);
		d.rgb_blend_operation = static_cast<uint32_t>(ca.rgbBlendOperation);
		d.source_alpha_blend_factor = static_cast<uint32_t>(ca.sourceAlphaBlendFactor);
		d.destination_alpha_blend_factor = static_cast<uint32_t>(ca.destinationAlphaBlendFactor);
		d.alpha_blend_operation = static_cast<uint32_t>(ca.alphaBlendOperation);
		d.write_mask = static_cast<uint32_t>(ca.writeMask);
		ncolor = i + 1;
	}
	return ncolor;
}

// --- mesh render pipelines ------------------------------------------------------------
// Metal's refusals of a mesh render pipeline's descriptor are its compiler's, in its domain
// (measured: guest/bundle/tools/oracle/oracle_mesh on the AMD Radeon Pro 5300M and the Intel UHD
// Graphics 630, macOS 26.7.1, the x86 Macs the device imitates).
NSErrorDomain const kCompilerErrorDomain = @"CompilerError";

// The largest payload an object function may pass its mesh threadgroups, in bytes: every Mac's
// (measured: oracle_mesh payload-big, refused at 16385 on the 5300M, the 630 and the M3 Max).
constexpr NSUInteger kMaxMeshPayload = 16384;

NSError *compilerError(NSInteger code, NSString *words)
{
	return [NSError errorWithDomain:kCompilerErrorDomain
	                           code:code
	                       userInfo:@{NSLocalizedDescriptionKey : words}];
}

/// Whether the library told the type of `fn`, one of the device's functions or nil, and if so
/// that type in `type`.
bool knownType(id<MTLFunction> fn, MTLFunctionType &type)
{
	if (fn == nil || ![(id)fn isKindOfClass:[RgfxFunction class]] ||
	    ![(RgfxFunction *)fn rgfxFunctionTypeKnown])
		return false;
	type = fn.functionType;
	return true;
}

/// Metal's refusal of a stage function of another stage's type, nil where each function has its
/// stage's type, or one its library did not tell (oracle_mesh vertex-as-mesh, vertex-as-mesh-norast,
/// object-as-mesh, mesh-as-fragment, fragment-as-object and mesh-as-object). The x86 Macs refuse a
/// vertex function as the mesh function with code 2 and the words below, and another as the mesh
/// function or another than a fragment function as the fragment function with code 1 and the words
/// of a compiler that crashed ("Compilation failed due to an interrupted connection:
/// XPC_ERROR_CONNECTION_INTERRUPTED. This error occurred after multiple retries."), which here are
/// those of Metal's validation layer instead. As the object function, a fragment function is
/// refused so on the 630 and ends the process with SIGSEGV on the 5300M, and a mesh function is
/// made on both and refused so on the M3 Max: refused, as the host refuses each of these before its
/// GPU's Metal sees them (the M3 Max's ends the process with SIGSEGV at a fragment function as the
/// object function, a mesh function as the fragment function, and a vertex function as the mesh
/// function).
NSError *wrongStageError(MTLMeshRenderPipelineDescriptor *descriptor)
{
	MTLFunctionType t = MTLFunctionTypeVertex;
	if (knownType(descriptor.meshFunction, t) && t != MTLFunctionTypeMesh) {
		if (t == MTLFunctionTypeVertex)
			return compilerError(2, @"RasterizationEnabled is false but the vertex shader's return "
			                        @"type is not void");
		return compilerError(1, @"meshFunction functionType is not MTLFunctionTypeMesh.");
	}
	if (knownType(descriptor.objectFunction, t) && t != MTLFunctionTypeObject)
		return compilerError(1, @"objectFunction functionType is not MTLFunctionTypeObject.");
	if (knownType(descriptor.fragmentFunction, t) && t != MTLFunctionTypeFragment)
		return compilerError(1, @"fragmentFunction functionType is not MTLFunctionTypeFragment.");
	return nil;
}

/// Metal's refusal of a limit past the device's, nil where none is: a payload past 16 KiB, in the
/// x86 Macs' words, and a stage's maximum of threads in a threadgroup past `maxTotalThreads`, in
/// the 5300M's (oracle_mesh payload-big, mesh-2048 and object-2048; the 630 makes the latter and
/// answers 1024, the M3 Max refuses it).
NSError *limitError(MTLMeshRenderPipelineDescriptor *descriptor, NSUInteger maxTotalThreads)
{
	if (descriptor.payloadMemoryLength > kMaxMeshPayload)
		return compilerError(
		    2, [NSString stringWithFormat:@"payloadMemoryLength property of "
		                                  @"MTLMeshRenderPipelineDescriptor (%lu bytes) exceeds "
		                                  @"maximum supported by this device (%lu bytes)",
		                                  static_cast<unsigned long>(descriptor.payloadMemoryLength),
		                                  static_cast<unsigned long>(kMaxMeshPayload)]);
	for (NSUInteger most : {descriptor.maxTotalThreadsPerObjectThreadgroup,
	                        descriptor.maxTotalThreadsPerMeshThreadgroup})
		if (most > maxTotalThreads)
			return compilerError(
			    2, [NSString stringWithFormat:@"maxThreadsPerThreadgroup pipeline descriptor value "
			                                  @"(%lu) exceeds maximum supported (%lu)\n",
			                                  static_cast<unsigned long>(most),
			                                  static_cast<unsigned long>(maxTotalThreads)]);
	return nil;
}

uint32_t wire32(NSUInteger v)
{
	return v > UINT32_MAX ? UINT32_MAX : static_cast<uint32_t>(v);
}

void fillSize(uint32_t (&out)[3], MTLSize size)
{
	out[0] = wire32(size.width);
	out[1] = wire32(size.height);
	out[2] = wire32(size.depth);
}

// The descriptor's value of its private property `key` (Apple SPI), read by key-value coding
// where the descriptor answers its getter (`key` or `is<Key>`); nil where it answers neither.
NSNumber *privateValue(MTLRenderPipelineDescriptor *d, NSString *key)
{
	NSString *is = [NSString stringWithFormat:@"is%@%@", [key substringToIndex:1].uppercaseString,
	                                          [key substringFromIndex:1]];
	if (![d respondsToSelector:NSSelectorFromString(key)] &&
	    ![d respondsToSelector:NSSelectorFromString(is)])
		return nil;
	id v = [d valueForKey:key];
	return [v isKindOfClass:[NSNumber class]] ? v : nil;
}

// A pipeline stage's linked functions, where the descriptor has the getter. MTLLinkedFunctions
// reached the render and mesh descriptors after macOS 11 (vertexLinkedFunctions and its siblings),
// and sending one of those getters to a descriptor that lacks it raises an unrecognized-selector
// exception that ends the process; CoreDisplay creates a render pipeline for its display on macOS
// 11 (CreateMetalDevice), which read it there. Answer nil where the getter is absent, so the
// create goes on with no linked functions, as that descriptor carries none.
MTLLinkedFunctions *stageLinkedFunctions(id descriptor, SEL getter)
{
	if (![descriptor respondsToSelector:getter])
		return nil;
	return [descriptor valueForKey:NSStringFromSelector(getter)];
}

// The private state of `d` that Apple's OpenGL renderer sets (RgfxRenderPipelinePrivateState),
// each property Metal's default where the descriptor does not answer it: a sample mask of all
// ones, a coverage of 1, logic operation 0 and the flags off.
RgfxRenderPipelinePrivateState privateStateOf(MTLRenderPipelineDescriptor *d)
{
	RgfxRenderPipelinePrivateState s = {};
	NSNumber *v = privateValue(d, @"sampleMask");
	s.sample_mask = v != nil ? v.unsignedLongLongValue : UINT64_MAX;
	const float coverage = (v = privateValue(d, @"sampleCoverage")) != nil ? v.floatValue : 1.0f;
	std::memcpy(&s.sample_coverage, &coverage, sizeof(coverage));
	s.logic_operation = (v = privateValue(d, @"logicOperation")) != nil ? wire32(v.unsignedLongValue) : 0;
	const struct {
		NSString *key;
		uint32_t flag;
	} flags[] = {
	    {@"openGLModeEnabled", RGFX_PRIVATE_RENDER_OPENGL_MODE},
	    {@"sampleCoverageInvert", RGFX_PRIVATE_RENDER_SAMPLE_COVERAGE_INVERT},
	    {@"logicOperationEnabled", RGFX_PRIVATE_RENDER_LOGIC_OPERATION},
	    {@"depthStencilWriteDisabled", RGFX_PRIVATE_RENDER_DEPTH_STENCIL_WRITE_DISABLED},
	};
	for (const auto &f : flags)
		if (privateValue(d, f.key).boolValue)
			s.flags |= f.flag;
	return s;
}

} // namespace

// The device's makers of function tables of no stage (MTLDeviceSPI), which Metal's base render
// pipeline asks for each stage's (RgfxMetalDevice.mm, RgfxMetalDevice+Unsupported.mm).
@protocol RgfxDeviceFunctionTables
- (id<MTLVisibleFunctionTable>)newVisibleFunctionTableWithDescriptor:
    (MTLVisibleFunctionTableDescriptor *)descriptor;
- (id<MTLIntersectionFunctionTable>)newIntersectionFunctionTableWithDescriptor:
    (MTLIntersectionFunctionTableDescriptor *)descriptor;
@end

@implementation RgfxRenderPipelineState {
	uint64_t _seqno;
	// What the descriptor asked that the pipeline answers.
	BOOL _supportIndirectCommandBuffers;
	// The host's reflection, filled once RGFX_EVT_PIPELINE_REFLECTION for `_seqno`
	// arrives through the reply reader; until then `_haveReflection` is NO and the
	// getters answer documented placeholders.
	RgfxPipelineReflection _reflection;
	BOOL _haveReflection;
	// Its functions' arguments; null where unknown.
	std::shared_ptr<const RgfxAirFunction> _vertexAir;
	std::shared_ptr<const RgfxAirFunction> _fragmentAir;
	// A mesh render pipeline's: its object and mesh functions' arguments (none for a vertex and
	// fragment pipeline), whether it has an object function, the sizes its descriptor required,
	// and the stages' limits its answer carried (`_haveMeshLimits`).
	BOOL _mesh;
	std::shared_ptr<const RgfxAirFunction> _objectAir;
	std::shared_ptr<const RgfxAirFunction> _meshAir;
	MTLSize _requiredObject;
	MTLSize _requiredMesh;
	RgfxMeshPipelineReflection _meshLimits;
	BOOL _haveMeshLimits;
	// A keyed create's key, and the known pipelines its answer is kept in once accepted, until
	// it registers; a create posted without waiting has its own answer to wait for.
	std::shared_ptr<RgfxKnownPipelines> _known;
	RgfxPipelineKey _key;
	BOOL _deferred;
	RgfxPipelineAnswer _answer;
	NSInteger _textureWriteRoundingMode; // the descriptor's (MTLRenderPipelineDescriptorSPI)
}

// The host drops the pipeline as the app lets it go, after any command buffer that names it (each
// keeps it while it runs).
- (void)dealloc
{
	[self rgfxRetireThen:nullptr];
}

+ (instancetype)renderPipelineStateWithDescriptor:(MTLRenderPipelineDescriptor *)descriptor
                                           device:(id<MTLDevice>)device
                                       serializer:(RgfxSerializer *)serializer
                                            error:(__autoreleasing NSError **)error
{
	return [self renderPipelineStateWithDescriptor:descriptor
	                                        device:device
	                                    serializer:serializer
	                                         known:nullptr
	                                         error:error];
}

+ (instancetype)renderPipelineStateWithDescriptor:(MTLRenderPipelineDescriptor *)descriptor
                                           device:(id<MTLDevice>)device
                                       serializer:(RgfxSerializer *)serializer
                                            known:(std::shared_ptr<RgfxKnownPipelines>)known
                                            error:(__autoreleasing NSError **)error
{
	if (serializer == nullptr) {
		if (error)
			*error = RgfxMakeError(RgfxMetalErrorNoRoute, @"render pipeline: no serialiser route");
		return nil;
	}

	NSError *unspecialised =
	    [RgfxFunction rgfxUnspecialisedError:descriptor.vertexFunction role:@"vertexFunction"]
	        ?: [RgfxFunction rgfxUnspecialisedError:descriptor.fragmentFunction
	                                           role:@"fragmentFunction"];
	if (unspecialised != nil) {
		if (error)
			*error = unspecialised;
		return nil;
	}

	RgfxRenderPipelineParams p = {};
	if (!fillFunctionRef(p.vertex, descriptor.vertexFunction) ||
	    !fillFunctionRef(p.fragment, descriptor.fragmentFunction)) {
		id<MTLFunction> foreign = descriptor.vertexFunction;
		if ([(id)foreign isKindOfClass:[RgfxFunction class]])
			foreign = descriptor.fragmentFunction;
		NSString *msg = [NSString
		    stringWithFormat:@"render pipeline: function %@ (%@) is not from a library of this device",
		                     foreign.name, NSStringFromClass([(id)foreign class])];
		static char site;
		if (rgfxLogFirst(&site))
			NSLog(@"RGFX: %@", msg);
		if (error)
			*error = RgfxMakeError(RgfxMetalErrorInvalidDescriptor, msg);
		return nil;
	}

	// Each stage's linked functions; one another device's library made names no library the
	// host holds, so the create is refused rather than sent.
	RgfxLinkedFunctionList vertexLinked, fragmentLinked;
	id<MTLFunction> foreignLinked = RgfxCollectLinkedFunctions(
	    stageLinkedFunctions(descriptor, @selector(vertexLinkedFunctions)), vertexLinked);
	if (foreignLinked == nil)
		foreignLinked = RgfxCollectLinkedFunctions(
		    stageLinkedFunctions(descriptor, @selector(fragmentLinkedFunctions)), fragmentLinked);
	if (foreignLinked != nil) {
		NSString *msg = [NSString
		    stringWithFormat:@"render pipeline: linked function %@ (%@) is not from a library of this device",
		                     foreignLinked.name, NSStringFromClass([(id)foreignLinked class])];
		static char site;
		if (rgfxLogFirst(&site))
			NSLog(@"RGFX: %@", msg);
		if (error)
			*error = RgfxMakeError(RgfxMetalErrorInvalidDescriptor, msg);
		return nil;
	}
	p.vertex_linked = vertexLinked.refs.empty() ? nullptr : vertexLinked.refs.data();
	p.vertex_linked_count = static_cast<uint32_t>(vertexLinked.refs.size());
	p.fragment_linked = fragmentLinked.refs.empty() ? nullptr : fragmentLinked.refs.data();
	p.fragment_linked_count = static_cast<uint32_t>(fragmentLinked.refs.size());

	std::string label;
	if (descriptor.label != nil && descriptor.label.UTF8String != nullptr) {
		label = descriptor.label.UTF8String;
		p.label = label.c_str();
	}

	p.raster_sample_count = static_cast<uint32_t>(descriptor.rasterSampleCount);

	p.color_attachment_count = fillColors(descriptor.colorAttachments, p.colors);

	p.depth_pixel_format = static_cast<uint32_t>(descriptor.depthAttachmentPixelFormat);
	p.stencil_pixel_format = static_cast<uint32_t>(descriptor.stencilAttachmentPixelFormat);

	uint32_t flags = 0;
	if (descriptor.isAlphaToCoverageEnabled)
		flags |= RGFX_RENDER_PIPELINE_ALPHA_TO_COVERAGE;
	if (descriptor.isAlphaToOneEnabled)
		flags |= RGFX_RENDER_PIPELINE_ALPHA_TO_ONE;
	if (descriptor.isRasterizationEnabled)
		flags |= RGFX_RENDER_PIPELINE_RASTERIZATION_ENABLED;
	if (descriptor.supportIndirectCommandBuffers)
		flags |= RGFX_RENDER_PIPELINE_SUPPORT_ICB;
	p.flags = flags;
	p.input_primitive_topology = static_cast<uint32_t>(descriptor.inputPrimitiveTopology);
	p.max_vertex_amplification_count =
	    static_cast<uint32_t>(descriptor.maxVertexAmplificationCount);
	RgfxRenderPipelinePrivateState privateState = {};
	if (serializer->carriesPrivateRenderState()) {
		privateState = privateStateOf(descriptor);
		p.private_state = &privateState;
	}

	// Vertex descriptor (optional): scan for the in-use layouts and attributes.
	std::vector<RgfxVertexBufferLayout> layouts;
	std::vector<RgfxVertexAttribute> attributes;
	MTLVertexDescriptor *vd = descriptor.vertexDescriptor;
	if (vd != nil) {
		for (NSUInteger i = 0; i < kMaxVertexBuffers; i++) {
			MTLVertexBufferLayoutDescriptor *l = vd.layouts[i];
			if (l.stride == 0) {
				continue;
			}
			RgfxVertexBufferLayout rl = {};
			rl.buffer_index = static_cast<uint32_t>(i);
			rl.stride = static_cast<uint32_t>(l.stride);
			rl.step_function = static_cast<uint32_t>(l.stepFunction);
			rl.step_rate = static_cast<uint32_t>(l.stepRate);
			layouts.push_back(rl);
		}
		for (NSUInteger i = 0; i < kMaxVertexAttributes; i++) {
			MTLVertexAttributeDescriptor *a = vd.attributes[i];
			if (a.format == MTLVertexFormatInvalid) {
				continue;
			}
			RgfxVertexAttribute ra = {};
			ra.attribute_index = static_cast<uint32_t>(i);
			ra.format = static_cast<uint32_t>(a.format);
			ra.offset = static_cast<uint32_t>(a.offset);
			ra.buffer_index = static_cast<uint32_t>(a.bufferIndex);
			attributes.push_back(ra);
		}
	}
	p.vertex_layouts = layouts.empty() ? nullptr : layouts.data();
	p.vertex_layout_count = static_cast<uint32_t>(layouts.size());
	p.vertex_attributes = attributes.empty() ? nullptr : attributes.data();
	p.vertex_attribute_count = static_cast<uint32_t>(attributes.size());

	RgfxPipelinePost post;
	post.known = known.get();
	RgfxCreated created = serializer->createRenderPipeline(p, &post);
	if (!created.posted) {
		if (error)
			*error = RgfxMakeError(RgfxMetalErrorRingFull,
			                       @"render pipeline: create did not fit the ring");
		return nil;
	}
	if (error)
		*error = nil;
	RgfxRenderPipelineState *state = [[RgfxRenderPipelineState alloc] initWithObjId:created.obj_id
	                                                                          seqno:created.seqno
	                                                                          label:descriptor.label
	                                                                         device:device
	                                                                     serializer:serializer];
	if (state != nil) {
		state->_supportIndirectCommandBuffers = descriptor.supportIndirectCommandBuffers;
		[state rgfxTakeRoundingOf:descriptor];
		state->_vertexAir = stageArguments(descriptor.vertexFunction);
		state->_fragmentAir = stageArguments(descriptor.fragmentFunction);
		if (post.deferred) {
			[state rgfxTakeKnown:post.reflection];
		} else if (post.keyed) {
			state->_known = known;
			state->_key = post.key;
		}
	}
	return state;
}

// Ends the process where Metal's validation of a mesh render pipeline's descriptor does, with its
// words (measured: guest/bundle/tools/oracle/oracle_mesh, plain Metal, no validation layer, on the
// AMD Radeon Pro 5300M and the Intel UHD Graphics 630 under macOS 26.7.1 (25G241), and the M3 Max
// under macOS 27.0.1 (26A434)):
// - no mesh function, on all three: "validateWithDevice:1884: failed assertion `Mesh Render
//   Pipeline Descriptor Validation\nmeshFunction must not be nil.\n'" (1956 on the M3 Max);
// - no fragment function while rasterization is enabled, on all three: "...\nfragmentFunction
//   must not be nil.\n'";
// - supportIndirectCommandBuffers YES, on the 5300M and the 630: "validateWithDevice:1884: failed
//   assertion `Mesh Render Pipeline Descriptor Validation\nUse of mesh shader pipelines in
//   indirect command buffers is not supported by this device.\n'". The M3 Max makes that
//   pipeline; the device imitates the x86 Macs, whose families claim no mesh draws in indirect
//   command buffers either (supportsFamily: answers NO for Apple7 to Apple9 on them, and on the
//   device).
// Metal's message lists each failed check; its order where several fail is not measured.
+ (void)rgfxEndAtInvalidMeshDescriptor:(MTLMeshRenderPipelineDescriptor *)descriptor
{
	NSMutableString *failed = [NSMutableString string];
	if (descriptor.meshFunction == nil)
		[failed appendString:@"meshFunction must not be nil.\n"];
	if (descriptor.fragmentFunction == nil && descriptor.isRasterizationEnabled)
		[failed appendString:@"fragmentFunction must not be nil.\n"];
	// supportIndirectCommandBuffers reached the mesh descriptor in macOS 14; before it a mesh
	// pipeline cannot ask for indirect command buffers, so there is nothing to refuse.
	BOOL meshSupportsIcb =
	    [descriptor respondsToSelector:@selector(supportIndirectCommandBuffers)] &&
	    descriptor.supportIndirectCommandBuffers;
	if (meshSupportsIcb)
		[failed appendString:@"Use of mesh shader pipelines in indirect command buffers is not "
		                     @"supported by this device.\n"];
	if (failed.length != 0)
		RgfxMetalAssertionFailed(
		    "validateWithDevice:1884",
		    [@"Mesh Render Pipeline Descriptor Validation\n" stringByAppendingString:failed]);
}

+ (instancetype)meshRenderPipelineStateWithDescriptor:(MTLMeshRenderPipelineDescriptor *)descriptor
                                               device:(id<MTLDevice>)device
                                           serializer:(RgfxSerializer *)serializer
                                      maxTotalThreads:(NSUInteger)maxTotalThreads
                                                error:(__autoreleasing NSError **)error
{
	[self rgfxEndAtInvalidMeshDescriptor:descriptor];
	if (serializer == nullptr) {
		if (error)
			*error = RgfxMakeError(RgfxMetalErrorNoRoute, @"render pipeline: no serialiser route");
		return nil;
	}

	NSError *unspecialised =
	    [RgfxFunction rgfxUnspecialisedError:descriptor.objectFunction role:@"objectFunction"]
	        ?: [RgfxFunction rgfxUnspecialisedError:descriptor.meshFunction role:@"meshFunction"]
	        ?: [RgfxFunction rgfxUnspecialisedError:descriptor.fragmentFunction
	                                           role:@"fragmentFunction"];
	if (unspecialised != nil) {
		if (error)
			*error = unspecialised;
		return nil;
	}

	RgfxMeshRenderPipelineParams p = {};
	id<MTLFunction> foreign = nil;
	if (!fillFunctionRef(p.object, descriptor.objectFunction))
		foreign = descriptor.objectFunction;
	else if (!fillFunctionRef(p.mesh, descriptor.meshFunction))
		foreign = descriptor.meshFunction;
	else if (!fillFunctionRef(p.fragment, descriptor.fragmentFunction))
		foreign = descriptor.fragmentFunction;
	if (foreign != nil) {
		NSString *msg = [NSString
		    stringWithFormat:@"render pipeline: function %@ (%@) is not from a library of this device",
		                     foreign.name, NSStringFromClass([(id)foreign class])];
		static char site;
		if (rgfxLogFirst(&site))
			NSLog(@"RGFX: %@", msg);
		if (error)
			*error = RgfxMakeError(RgfxMetalErrorInvalidDescriptor, msg);
		return nil;
	}

	NSError *refusal = wrongStageError(descriptor) ?: limitError(descriptor, maxTotalThreads);
	if (refusal != nil) {
		if (error)
			*error = refusal;
		return nil;
	}

	RgfxLinkedFunctionList objectLinked, meshLinked, fragmentLinked;
	id<MTLFunction> foreignLinked = RgfxCollectLinkedFunctions(
	    stageLinkedFunctions(descriptor, @selector(objectLinkedFunctions)), objectLinked);
	if (foreignLinked == nil)
		foreignLinked = RgfxCollectLinkedFunctions(
		    stageLinkedFunctions(descriptor, @selector(meshLinkedFunctions)), meshLinked);
	if (foreignLinked == nil)
		foreignLinked = RgfxCollectLinkedFunctions(
		    stageLinkedFunctions(descriptor, @selector(fragmentLinkedFunctions)), fragmentLinked);
	if (foreignLinked != nil) {
		NSString *msg = [NSString
		    stringWithFormat:@"render pipeline: linked function %@ (%@) is not from a library of this device",
		                     foreignLinked.name, NSStringFromClass([(id)foreignLinked class])];
		static char site;
		if (rgfxLogFirst(&site))
			NSLog(@"RGFX: %@", msg);
		if (error)
			*error = RgfxMakeError(RgfxMetalErrorInvalidDescriptor, msg);
		return nil;
	}
	p.object_linked = objectLinked.refs.empty() ? nullptr : objectLinked.refs.data();
	p.object_linked_count = static_cast<uint32_t>(objectLinked.refs.size());
	p.mesh_linked = meshLinked.refs.empty() ? nullptr : meshLinked.refs.data();
	p.mesh_linked_count = static_cast<uint32_t>(meshLinked.refs.size());
	p.fragment_linked = fragmentLinked.refs.empty() ? nullptr : fragmentLinked.refs.data();
	p.fragment_linked_count = static_cast<uint32_t>(fragmentLinked.refs.size());

	std::string label;
	if (descriptor.label != nil && descriptor.label.UTF8String != nullptr) {
		label = descriptor.label.UTF8String;
		p.label = label.c_str();
	}

	p.raster_sample_count = static_cast<uint32_t>(descriptor.rasterSampleCount);
	p.color_attachment_count = fillColors(descriptor.colorAttachments, p.colors);
	p.depth_pixel_format = static_cast<uint32_t>(descriptor.depthAttachmentPixelFormat);
	p.stencil_pixel_format = static_cast<uint32_t>(descriptor.stencilAttachmentPixelFormat);

	uint32_t flags = 0;
	if (descriptor.isAlphaToCoverageEnabled)
		flags |= RGFX_RENDER_PIPELINE_ALPHA_TO_COVERAGE;
	if (descriptor.isAlphaToOneEnabled)
		flags |= RGFX_RENDER_PIPELINE_ALPHA_TO_ONE;
	if (descriptor.isRasterizationEnabled)
		flags |= RGFX_RENDER_PIPELINE_RASTERIZATION_ENABLED;
	// The mesh descriptor gained supportIndirectCommandBuffers in macOS 14; off before it.
	if ([descriptor respondsToSelector:@selector(supportIndirectCommandBuffers)] &&
	    descriptor.supportIndirectCommandBuffers)
		flags |= RGFX_RENDER_PIPELINE_SUPPORT_ICB;
	if (descriptor.objectThreadgroupSizeIsMultipleOfThreadExecutionWidth)
		flags |= RGFX_MESH_PIPELINE_OBJECT_TG_MULTIPLE_OF_EXEC_WIDTH;
	if (descriptor.meshThreadgroupSizeIsMultipleOfThreadExecutionWidth)
		flags |= RGFX_MESH_PIPELINE_MESH_TG_MULTIPLE_OF_EXEC_WIDTH;
	p.flags = flags;
	p.max_total_threads_per_object_threadgroup =
	    wire32(descriptor.maxTotalThreadsPerObjectThreadgroup);
	p.max_total_threads_per_mesh_threadgroup = wire32(descriptor.maxTotalThreadsPerMeshThreadgroup);
	p.max_total_threadgroups_per_mesh_grid = wire32(descriptor.maxTotalThreadgroupsPerMeshGrid);
	p.payload_memory_length = wire32(descriptor.payloadMemoryLength);
	p.max_vertex_amplification_count = wire32(descriptor.maxVertexAmplificationCount);
	// requiredThreads* reached the mesh descriptor in macOS 26; before it they are unset (0,0,0).
	if ([descriptor respondsToSelector:@selector(requiredThreadsPerObjectThreadgroup)]) {
		fillSize(p.required_threads_per_object_threadgroup,
		         descriptor.requiredThreadsPerObjectThreadgroup);
		fillSize(p.required_threads_per_mesh_threadgroup,
		         descriptor.requiredThreadsPerMeshThreadgroup);
	}

	RgfxCreated created = serializer->createMeshRenderPipeline(p);
	if (!created.posted) {
		if (error)
			*error = RgfxMakeError(RgfxMetalErrorRingFull,
			                       @"render pipeline: create did not fit the ring");
		return nil;
	}
	if (error)
		*error = nil;
	RgfxRenderPipelineState *state = [[RgfxRenderPipelineState alloc] initWithObjId:created.obj_id
	                                                                          seqno:created.seqno
	                                                                          label:descriptor.label
	                                                                         device:device
	                                                                     serializer:serializer];
	if (state != nil) {
		if ([descriptor respondsToSelector:@selector(supportIndirectCommandBuffers)])
			state->_supportIndirectCommandBuffers = descriptor.supportIndirectCommandBuffers;
		state->_mesh = YES;
		[state rgfxTakeRoundingOf:descriptor];
		state->_objectAir = stageArguments(descriptor.objectFunction);
		state->_meshAir = stageArguments(descriptor.meshFunction);
		state->_fragmentAir = stageArguments(descriptor.fragmentFunction);
		state->_vertexAir = stageArguments(nil);
		if ([descriptor respondsToSelector:@selector(requiredThreadsPerObjectThreadgroup)]) {
			state->_requiredObject = descriptor.requiredThreadsPerObjectThreadgroup;
			state->_requiredMesh = descriptor.requiredThreadsPerMeshThreadgroup;
		}
	}
	return state;
}

// How the pipeline's texture writes round floats, which Metal's render and mesh descriptors keep
// (0, their default, where the guest's Metal has no such member) and its pipeline answers. The
// host rounds as its default.
- (void)rgfxTakeRoundingOf:(id)descriptor
{
	SEL rounding = sel_registerName("textureWriteRoundingMode");
	if ([descriptor respondsToSelector:rounding])
		_textureWriteRoundingMode = ((NSInteger (*)(id, SEL))objc_msgSend)(descriptor, rounding);
	if (_textureWriteRoundingMode != 0)
		RgfxLogUnsupported([self class], rounding,
		                   "a texture write rounding mode other than the default (none on the wire)");
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

- (std::shared_ptr<const RgfxAirFunction>)rgfxVertexAirFunction
{
	return _vertexAir;
}

- (std::shared_ptr<const RgfxAirFunction>)rgfxFragmentAirFunction
{
	return _fragmentAir;
}

- (std::shared_ptr<const RgfxAirFunction>)rgfxObjectAirFunction
{
	return _mesh ? _objectAir : stageArguments(nil);
}

- (std::shared_ptr<const RgfxAirFunction>)rgfxMeshAirFunction
{
	return _mesh ? _meshAir : stageArguments(nil);
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
	// Weak capture: if the caller releases this state before its reflection arrives,
	// the one-shot handler is a no-op (the reflection is then moot).
	__weak RgfxRenderPipelineState *weakSelf = self;
	if (_deferred) {
		// Its own answer: the resource ID gpuResourceID waits for.
		reader->expectReflection(_seqno, [weakSelf](const RgfxPipelineReflection &r) {
			RgfxRenderPipelineState *strongSelf = weakSelf;
			if (strongSelf != nil)
				[strongSelf rgfxTakeAnswer:&r];
		});
		return;
	}
	// A keyed create's answer is kept once the device accepted it, so equal creates need not wait.
	std::shared_ptr<RgfxKnownPipelines> known = std::move(_known);
	const RgfxPipelineKey key = _key;
	reader->expectReflection(_seqno, [weakSelf, known, key](const RgfxPipelineReply &r) {
		if (known)
			known->learn(key, r);
		RgfxRenderPipelineState *strongSelf = weakSelf;
		if (strongSelf != nil) {
			[strongSelf rgfxFillReflection:&r];
		}
	});
}

// A create posted without waiting: its own answer, whose resource ID gpuResourceID answers. A
// refusal (the platform could not make the pipeline, its descriptor though equal to an accepted
// one's) shows only there, as 0, and in the command buffers that name it.
- (void)rgfxTakeAnswer:(const RgfxPipelineReflection *)r
{
	_answer.take(*r);
	static char site;
	if (r->status != RGFX_PIPELINE_OK && rgfxLogFirst(&site, r->status))
		NSLog(@"RGFX: the device refused render pipeline %llu, made as an equal one it accepted",
		      static_cast<unsigned long long>(self.rgfxObjId));
}

// Stores the host's reflection, and a mesh render pipeline's limits after it; the getters below
// then report them.
- (void)rgfxFillReflection:(const RgfxPipelineReply *)r
{
	_reflection = *r;
	_haveReflection = YES;
	if (r->hasMesh) {
		_meshLimits = r->mesh;
		_haveMeshLimits = YES;
	}
}

- (BOOL)rgfxRefused
{
	return _haveReflection && _reflection.status == RGFX_PIPELINE_REFUSED;
}

// --- MTLRenderPipelineState: reflection --------------------------------------
// The host answers the create with an RgfxPipelineReflection over the completion
// ring, matched by `rgfxSeqno` and read by the reply reader. Before it arrives these
// answer documented placeholders (the caps-placeholder pattern of slice 0); after it
// they report the host's values.

- (NSUInteger)maxTotalThreadsPerThreadgroup
{
	if (_haveReflection) {
		return static_cast<NSUInteger>(_reflection.max_total_threads_per_threadgroup);
	}
	return 1024; // Metal's common threadgroup floor until reflection is read
}

- (BOOL)threadgroupSizeMatchesTileSize
{
	if (_haveReflection) {
		return (_reflection.flags & RGFX_PIPELINE_REFLECT_THREADGROUP_MATCHES_TILE) != 0;
	}
	return NO;
}

- (NSUInteger)imageblockSampleLength
{
	if (_haveReflection) {
		return _reflection.imageblock_sample_length;
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
	return _supportIndirectCommandBuffers;
}

// The device validates no shader, so no pipeline of it runs with validation.
- (MTLShaderValidation)shaderValidation
{
	return MTLShaderValidationDisabled;
}

// Metal's answer for every pipeline a device (not a Metal 4 compiler) makes.
- (MTLRenderPipelineReflection *)reflection
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

// Metal answers {0, 0, 0} and 0 for a vertex and fragment pipeline, and for a mesh render pipeline
// (measured: oracle_mesh on the 5300M, the 630 and the M3 Max, `tg`).
- (MTLSize)requiredThreadsPerThreadgroup
{
	return MTLSizeMake(0, 0, 0);
}

- (NSUInteger)staticThreadgroupMemoryLength
{
	return 0;
}

// --- the mesh and object stages ------------------------------------------------------------
// A mesh render pipeline answers its stages' limits as the host's platform answered them, which its
// create's answer carries (the x86 Macs answer `maxTotalThreadgroupsPerMeshGrid` NSUIntegerMax, the
// wire's highest value; measured: oracle_mesh), and the threadgroup sizes its descriptor required,
// as Metal's do. Metal answers 0 for a pipeline without the stage, as a vertex and fragment
// pipeline is. The device makes no tile pipeline (newRenderPipelineStateWithTileDescriptor:
// answers nil).

- (NSUInteger)maxTotalThreadsPerObjectThreadgroup
{
	return _haveMeshLimits ? _meshLimits.max_total_threads_per_object_threadgroup : 0;
}

- (NSUInteger)maxTotalThreadsPerMeshThreadgroup
{
	return _haveMeshLimits ? _meshLimits.max_total_threads_per_mesh_threadgroup : 0;
}

- (NSUInteger)objectThreadExecutionWidth
{
	return _haveMeshLimits ? _meshLimits.object_thread_execution_width : 0;
}

- (NSUInteger)meshThreadExecutionWidth
{
	return _haveMeshLimits ? _meshLimits.mesh_thread_execution_width : 0;
}

- (NSUInteger)maxTotalThreadgroupsPerMeshGrid
{
	if (!_haveMeshLimits)
		return 0;
	if (_meshLimits.max_total_threadgroups_per_mesh_grid == UINT32_MAX)
		return NSUIntegerMax;
	return _meshLimits.max_total_threadgroups_per_mesh_grid;
}

- (MTLSize)requiredThreadsPerTileThreadgroup
{
	return MTLSizeMake(0, 0, 0);
}

- (MTLSize)requiredThreadsPerObjectThreadgroup
{
	return _mesh ? _requiredObject : MTLSizeMake(0, 0, 0);
}

- (MTLSize)requiredThreadsPerMeshThreadgroup
{
	return _mesh ? _requiredMesh : MTLSizeMake(0, 0, 0);
}

// --- MTLRenderPipelineStateSPI -------------------------------------------------------------
// The members of Metal's render-pipeline SPI that the AMD 5300M's and the Intel 630's pipelines
// both answer (measured with respondsToSelector:, macOS 26.7.1, tools/oracle/oracle_spi_members.m),
// each as Metal's base pipeline, _MTLRenderPipelineState, answers it (read from code, macOS
// 27.0.1), and as a fresh pipeline of both x86 Macs answers those that take no argument (measured:
// macOS 26.7.1, tools/oracle/oracle_spi_members.m answers).

// The debug info of the pipeline's vertex and fragment shaders: the device keeps none, so they
// answer nil.
- (nullable id)newVertexShaderDebugInfo
{
	RGFX_UNSUPPORTED("a shader's debug info");
	return nil;
}

- (nullable id)newFragmentShaderDebugInfo
{
	RGFX_UNSUPPORTED("a shader's debug info");
	return nil;
}

// Metal's base pipeline answers gpuResourceID with gpuHandle, and gpuHandle with resourceIndex: so
// these answer gpuResourceID's value, the one the host numbered the pipeline by.
- (uint64_t)resourceIndex
{
	return [self gpuResourceID]._impl;
}

- (uint64_t)gpuHandle
{
	return [self resourceIndex];
}

// The low 32 bits of uniqueIdentifier, as the base and the AMD 5300M's pipeline answer (the
// Intel 630's answers 0).
- (uint32_t)getVertexShaderTelemetryID
{
	return static_cast<uint32_t>([self uniqueIdentifier]);
}

- (uint32_t)getFragmentShaderTelemetryID
{
	return static_cast<uint32_t>([self uniqueIdentifier]);
}

// No debug instrumentation or global constants for any stage, no binaries and no explicit
// visibility group: the device instruments no shader.

- (id)vertexDebugInstrumentationData
{
	return nil;
}

- (uint64_t)vertexGlobalConstantsBase
{
	return 0;
}

- (uint64_t)vertexGlobalConstantsSize
{
	return 0;
}

- (id)fragmentDebugInstrumentationData
{
	return nil;
}

- (uint64_t)fragmentGlobalConstantsBase
{
	return 0;
}

- (uint64_t)fragmentGlobalConstantsSize
{
	return 0;
}

- (id)tileDebugInstrumentationData
{
	return nil;
}

- (uint64_t)tileGlobalConstantsBase
{
	return 0;
}

- (uint64_t)tileGlobalConstantsSize
{
	return 0;
}

- (id)meshDebugInstrumentationData
{
	return nil;
}

- (uint64_t)meshGlobalConstantsBase
{
	return 0;
}

- (uint64_t)meshGlobalConstantsSize
{
	return 0;
}

- (id)objectDebugInstrumentationData
{
	return nil;
}

- (uint64_t)objectGlobalConstantsBase
{
	return 0;
}

- (uint64_t)objectGlobalConstantsSize
{
	return 0;
}

- (NSDictionary *)pipelineBinaries
{
	return @{};
}

- (uint32_t)explicitVisibilityGroupID
{
	return 0;
}

// MTLShaderValidationDisabled, as shaderValidation answers.
- (NSInteger)shaderValidationState
{
	return MTLShaderValidationDisabled;
}

// The descriptor's rounding of floats on texture writes, as the base keeps it.
- (NSInteger)textureWriteRoundingMode
{
	return _textureWriteRoundingMode;
}

// The base raises an unrecognized selector for these (it sends itself doesNotRecognizeSelector:),
// and so does this.
- (id)functionReflectionWithFunctionDescriptor:(id)descriptor stage:(NSUInteger)stage
{
	(void)descriptor;
	(void)stage;
	[self doesNotRecognizeSelector:_cmd];
	return nil;
}

- (id)reflectionForFunctionDescriptor:(id)descriptor
{
	(void)descriptor;
	[self doesNotRecognizeSelector:_cmd];
	return nil;
}

// A stage's function handle: function pointers, as functionHandleWithFunction:stage:.

- (id)vertexFunctionHandleWithFunction:(id<MTLFunction>)function
{
	(void)function;
	RGFX_UNSUPPORTED("function pointers");
	return nil;
}

- (id)fragmentFunctionHandleWithFunction:(id<MTLFunction>)function
{
	(void)function;
	RGFX_UNSUPPORTED("function pointers");
	return nil;
}

- (id)tileFunctionHandleWithFunction:(id<MTLFunction>)function
{
	(void)function;
	RGFX_UNSUPPORTED("function pointers");
	return nil;
}

- (id)meshFunctionHandleWithFunction:(id<MTLFunction>)function
{
	(void)function;
	RGFX_UNSUPPORTED("function pointers");
	return nil;
}

- (id)objectFunctionHandleWithFunction:(id<MTLFunction>)function
{
	(void)function;
	RGFX_UNSUPPORTED("function pointers");
	return nil;
}

// A stage's function tables, which the base asks its device for, as these do
// (RgfxDeviceFunctionTables).
- (id<MTLVisibleFunctionTable>)newVisibleFunctionTableFromVertexStageWithDescriptor:
    (MTLVisibleFunctionTableDescriptor *)descriptor
{
	id<RgfxDeviceFunctionTables> device = (id<RgfxDeviceFunctionTables>)self.device;
	return [device newVisibleFunctionTableWithDescriptor:descriptor];
}

- (id<MTLIntersectionFunctionTable>)newVertexIntersectionFunctionTableWithDescriptor:
    (MTLIntersectionFunctionTableDescriptor *)descriptor
{
	id<RgfxDeviceFunctionTables> device = (id<RgfxDeviceFunctionTables>)self.device;
	return [device newIntersectionFunctionTableWithDescriptor:descriptor];
}

- (id<MTLVisibleFunctionTable>)newVisibleFunctionTableFromFragmentStageWithDescriptor:
    (MTLVisibleFunctionTableDescriptor *)descriptor
{
	id<RgfxDeviceFunctionTables> device = (id<RgfxDeviceFunctionTables>)self.device;
	return [device newVisibleFunctionTableWithDescriptor:descriptor];
}

- (id<MTLIntersectionFunctionTable>)newFragmentIntersectionFunctionTableWithDescriptor:
    (MTLIntersectionFunctionTableDescriptor *)descriptor
{
	id<RgfxDeviceFunctionTables> device = (id<RgfxDeviceFunctionTables>)self.device;
	return [device newIntersectionFunctionTableWithDescriptor:descriptor];
}

- (id<MTLVisibleFunctionTable>)newVisibleFunctionTableFromTileStageWithDescriptor:
    (MTLVisibleFunctionTableDescriptor *)descriptor
{
	id<RgfxDeviceFunctionTables> device = (id<RgfxDeviceFunctionTables>)self.device;
	return [device newVisibleFunctionTableWithDescriptor:descriptor];
}

- (id<MTLIntersectionFunctionTable>)newTileIntersectionFunctionTableWithDescriptor:
    (MTLIntersectionFunctionTableDescriptor *)descriptor
{
	id<RgfxDeviceFunctionTables> device = (id<RgfxDeviceFunctionTables>)self.device;
	return [device newIntersectionFunctionTableWithDescriptor:descriptor];
}

- (id<MTLVisibleFunctionTable>)newVisibleFunctionTableFromMeshStageWithDescriptor:
    (MTLVisibleFunctionTableDescriptor *)descriptor
{
	id<RgfxDeviceFunctionTables> device = (id<RgfxDeviceFunctionTables>)self.device;
	return [device newVisibleFunctionTableWithDescriptor:descriptor];
}

- (id<MTLIntersectionFunctionTable>)newMeshIntersectionFunctionTableWithDescriptor:
    (MTLIntersectionFunctionTableDescriptor *)descriptor
{
	id<RgfxDeviceFunctionTables> device = (id<RgfxDeviceFunctionTables>)self.device;
	return [device newIntersectionFunctionTableWithDescriptor:descriptor];
}

- (id<MTLVisibleFunctionTable>)newVisibleFunctionTableFromObjectStageWithDescriptor:
    (MTLVisibleFunctionTableDescriptor *)descriptor
{
	id<RgfxDeviceFunctionTables> device = (id<RgfxDeviceFunctionTables>)self.device;
	return [device newVisibleFunctionTableWithDescriptor:descriptor];
}

- (id<MTLIntersectionFunctionTable>)newObjectIntersectionFunctionTableWithDescriptor:
    (MTLIntersectionFunctionTableDescriptor *)descriptor
{
	id<RgfxDeviceFunctionTables> device = (id<RgfxDeviceFunctionTables>)self.device;
	return [device newIntersectionFunctionTableWithDescriptor:descriptor];
}

// A tile pipeline with more binary functions: binary functions, as
// newRenderPipelineStateWithAdditionalBinaryFunctions:error:.
- (id)newTileRenderPipelineStateWithAdditionalBinaryFunctions:(id)functions
                                                        error:(__autoreleasing NSError **)error
{
	(void)functions;
	RGFX_UNSUPPORTED_ERROR(error, "binary functions");
	return nil;
}

// Metal's emulation of mesh shaders on a GPU without them, which its own device sets up for the
// pipelines it makes so: the device carries mesh draws to the host, which draws them, so no
// pipeline of it emulates, and each answers as one that does not, 0, nil and NO (the base
// answers from emulation properties that only its setters make). The flag
// setUsesMeshShaderEmulation: sets is one the base's usesMeshShaderEmulation does not read.
- (BOOL)usesMeshShaderEmulation
{
	return NO;
}

- (void)setUsesMeshShaderEmulation:(BOOL)uses
{
	(void)uses;
}

- (id)emulationMeshShaderPSO
{
	return nil;
}

- (id)emulationObjectShaderPSO
{
	return nil;
}

- (id)emulationMeshPrefixSum1PSO
{
	return nil;
}

- (id)emulationMeshPrefixSum2PSO
{
	return nil;
}

- (id)emulationMeshPrefixSum3PSO
{
	return nil;
}

- (id)emulationNoObjectPrefixSumPSO
{
	return nil;
}

- (id)emulationObjectPrefixSumPSO
{
	return nil;
}

- (NSUInteger)emulationMeshSize
{
	return 0;
}

- (NSUInteger)emulationMeshMaxVertexCount
{
	return 0;
}

- (NSUInteger)emulationMeshMaxPrimitiveCount
{
	return 0;
}

- (NSUInteger)emulationObjectShaderIntermediateBufferSlot
{
	return 0;
}

- (NSUInteger)emulationMeshShaderIntermediateBufferSlot
{
	return 0;
}

- (NSUInteger)emulationVertexShaderIntermediateBufferSlot
{
	return 0;
}

- (NSUInteger)emulationPrimitiveTopology
{
	return 0;
}

- (NSUInteger)emulationPayloadMemoryLength
{
	return 0;
}

// Setting up an emulation, which the device's pipelines never are.

- (void)setEmulationMeshShaderPSO:(id)pipeline
{
	(void)pipeline;
	RGFX_UNSUPPORTED("Metal's mesh shader emulation (the host draws mesh pipelines)");
}

- (void)setEmulationObjectShaderPSO:(id)pipeline
{
	(void)pipeline;
	RGFX_UNSUPPORTED("Metal's mesh shader emulation (the host draws mesh pipelines)");
}

- (void)setEmulationMeshSize:(NSUInteger)value
{
	(void)value;
	RGFX_UNSUPPORTED("Metal's mesh shader emulation (the host draws mesh pipelines)");
}

- (void)setEmulationMeshMaxVertexCount:(NSUInteger)value
{
	(void)value;
	RGFX_UNSUPPORTED("Metal's mesh shader emulation (the host draws mesh pipelines)");
}

- (void)setEmulationMeshMaxPrimitiveCount:(NSUInteger)value
{
	(void)value;
	RGFX_UNSUPPORTED("Metal's mesh shader emulation (the host draws mesh pipelines)");
}

- (void)setEmulationObjectShaderIntermediateBufferSlot:(NSUInteger)value
{
	(void)value;
	RGFX_UNSUPPORTED("Metal's mesh shader emulation (the host draws mesh pipelines)");
}

- (void)setEmulationMeshShaderIntermediateBufferSlot:(NSUInteger)value
{
	(void)value;
	RGFX_UNSUPPORTED("Metal's mesh shader emulation (the host draws mesh pipelines)");
}

- (void)setEmulationVertexShaderIntermediateBufferSlot:(NSUInteger)value
{
	(void)value;
	RGFX_UNSUPPORTED("Metal's mesh shader emulation (the host draws mesh pipelines)");
}

- (void)setEmulationPrimitiveTopology:(NSUInteger)value
{
	(void)value;
	RGFX_UNSUPPORTED("Metal's mesh shader emulation (the host draws mesh pipelines)");
}

- (void)setEmulationPayloadMemoryLength:(NSUInteger)value
{
	(void)value;
	RGFX_UNSUPPORTED("Metal's mesh shader emulation (the host draws mesh pipelines)");
}

// --- not on the wire ------------------------------------------------------------------
// The host answers no pipeline's memory or imageblock queries yet (RgfxPipelineReflection
// carries none): 0, logged.

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

// --- unsupported: function pointers, binary functions, function tables, Metal 4 ----------

- (id<MTLFunctionHandle>)functionHandleWithFunction:(id<MTLFunction>)function
                                              stage:(MTLRenderStages)stage
{
	RGFX_UNSUPPORTED("function pointers");
	return nil;
}

- (id<MTLFunctionHandle>)functionHandleWithName:(NSString *)name stage:(MTLRenderStages)stage
{
	RGFX_UNSUPPORTED("function pointers");
	return nil;
}

- (id<MTLFunctionHandle>)functionHandleWithBinaryFunction:(id<MTL4BinaryFunction>)function
                                                    stage:(MTLRenderStages)stage
{
	RGFX_UNSUPPORTED("function pointers");
	return nil;
}

- (id<MTLRenderPipelineState>)
    newRenderPipelineStateWithAdditionalBinaryFunctions:
        (MTLRenderPipelineFunctionsDescriptor *)additionalBinaryFunctions
                                                  error:(__autoreleasing NSError **)error
{
	RGFX_UNSUPPORTED_ERROR(error, "binary functions");
	return nil;
}

- (id<MTLRenderPipelineState>)
    newRenderPipelineStateWithBinaryFunctions:
        (MTL4RenderPipelineBinaryFunctionsDescriptor *)binaryFunctionsDescriptor
                                        error:(NSError **)error
{
	RGFX_UNSUPPORTED_ERROR(error, "binary functions");
	return nil;
}

- (MTL4PipelineDescriptor *)newRenderPipelineDescriptorForSpecialization
{
	RGFX_UNSUPPORTED("Metal 4 pipeline specialization");
	return nil;
}

- (id<MTLVisibleFunctionTable>)newVisibleFunctionTableWithDescriptor:
                                   (MTLVisibleFunctionTableDescriptor *)descriptor
                                                               stage:(MTLRenderStages)stage
{
	RGFX_UNSUPPORTED("function pointers");
	return nil;
}

- (id<MTLIntersectionFunctionTable>)
    newIntersectionFunctionTableWithDescriptor:(MTLIntersectionFunctionTableDescriptor *)descriptor
                                         stage:(MTLRenderStages)stage
{
	RGFX_UNSUPPORTED("ray tracing");
	return nil;
}

@end
