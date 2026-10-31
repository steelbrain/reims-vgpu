/*
 * RgfxRenderPipelineState: see RgfxRenderPipelineState.h. Our own ObjC++; no Apple
 * private SPI.
 */

#import "RgfxRenderPipelineState.h"

#include <vector>

#import "RgfxFunction.h"
#import "RgfxMetalError.h"

#include "RgfxReplyReader.hpp"
#include "RgfxSerializer.hpp"

// The reflection-fill entry the reader's handler calls (defined below); declared here
// so the lambda in rgfxRegisterWithReplyReader: sees its signature.
@interface RgfxRenderPipelineState ()
- (void)rgfxFillReflection:(const RgfxPipelineReflection *)r;
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

} // namespace

@implementation RgfxRenderPipelineState {
	uint64_t _seqno;
	// The host's reflection, filled once RGFX_EVT_PIPELINE_REFLECTION for `_seqno`
	// arrives through the reply reader; until then `_haveReflection` is NO and the
	// getters answer documented placeholders.
	RgfxPipelineReflection _reflection;
	BOOL _haveReflection;
}

// The host drops the pipeline once every command buffer in flight now has completed, as one may
// still name it.
- (void)dealloc
{
	[self rgfxRetireThen:nullptr];
}

+ (instancetype)renderPipelineStateWithDescriptor:(MTLRenderPipelineDescriptor *)descriptor
                                           device:(id<MTLDevice>)device
                                       serializer:(RgfxSerializer *)serializer
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
		NSLog(@"RGFX: %@", msg);
		if (error)
			*error = RgfxMakeError(RgfxMetalErrorInvalidDescriptor, msg);
		return nil;
	}

	std::string label;
	if (descriptor.label != nil && descriptor.label.UTF8String != nullptr) {
		label = descriptor.label.UTF8String;
		p.label = label.c_str();
	}

	p.raster_sample_count = static_cast<uint32_t>(descriptor.rasterSampleCount);

	// Colour attachments: the descriptor's array has no count either; a colour
	// attachment is in use when its pixel format is not Invalid. Metal supports up
	// to 8 (the wire's RgfxColorAttachmentDesc colors[8]).
	uint32_t ncolor = 0;
	for (uint32_t i = 0; i < 8; i++) {
		MTLRenderPipelineColorAttachmentDescriptor *ca = descriptor.colorAttachments[i];
		if (ca.pixelFormat == MTLPixelFormatInvalid) {
			continue;
		}
		RgfxColorAttachmentDesc &d = p.colors[i];
		d.pixel_format = static_cast<uint32_t>(ca.pixelFormat);
		d.blending_enabled = ca.isBlendingEnabled ? 1 : 0;
		d.source_rgb_blend_factor = static_cast<uint32_t>(ca.sourceRGBBlendFactor);
		d.destination_rgb_blend_factor = static_cast<uint32_t>(ca.destinationRGBBlendFactor);
		d.rgb_blend_operation = static_cast<uint32_t>(ca.rgbBlendOperation);
		d.source_alpha_blend_factor = static_cast<uint32_t>(ca.sourceAlphaBlendFactor);
		d.destination_alpha_blend_factor = static_cast<uint32_t>(ca.destinationAlphaBlendFactor);
		d.alpha_blend_operation = static_cast<uint32_t>(ca.alphaBlendOperation);
		d.write_mask = static_cast<uint32_t>(ca.writeMask);
		// The count is the highest in-use index + 1, so a sparse set keeps its slots.
		ncolor = i + 1;
	}
	p.color_attachment_count = ncolor;

	p.depth_pixel_format = static_cast<uint32_t>(descriptor.depthAttachmentPixelFormat);
	p.stencil_pixel_format = static_cast<uint32_t>(descriptor.stencilAttachmentPixelFormat);

	uint32_t flags = 0;
	if (descriptor.isAlphaToCoverageEnabled)
		flags |= RGFX_RENDER_PIPELINE_ALPHA_TO_COVERAGE;
	if (descriptor.isAlphaToOneEnabled)
		flags |= RGFX_RENDER_PIPELINE_ALPHA_TO_ONE;
	if (descriptor.isRasterizationEnabled)
		flags |= RGFX_RENDER_PIPELINE_RASTERIZATION_ENABLED;
	p.flags = flags;
	p.input_primitive_topology = static_cast<uint32_t>(descriptor.inputPrimitiveTopology);
	p.max_vertex_amplification_count =
	    static_cast<uint32_t>(descriptor.maxVertexAmplificationCount);

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

	RgfxCreated created = serializer->createRenderPipeline(p);
	if (!created.posted) {
		if (error)
			*error = RgfxMakeError(RgfxMetalErrorRingFull,
			                       @"render pipeline: create did not fit the ring");
		return nil;
	}
	if (error)
		*error = nil;
	return [[RgfxRenderPipelineState alloc] initWithObjId:created.obj_id
	                                                seqno:created.seqno
	                                                label:descriptor.label
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
	// Weak capture: if the caller releases this state before its reflection arrives,
	// the one-shot handler is a no-op (the reflection is then moot).
	__weak RgfxRenderPipelineState *weakSelf = self;
	reader->expectReflection(_seqno, [weakSelf](const RgfxPipelineReflection &r) {
		RgfxRenderPipelineState *strongSelf = weakSelf;
		if (strongSelf != nil) {
			[strongSelf rgfxFillReflection:&r];
		}
	});
}

// Stores the host's reflection; the getters below then report it.
- (void)rgfxFillReflection:(const RgfxPipelineReflection *)r
{
	_reflection = *r;
	_haveReflection = YES;
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
	if (_haveReflection) {
		return (MTLResourceID){ _reflection.gpu_resource_id };
	}
	return (MTLResourceID){0}; // 0 until the reflection reply carries the real id
}

@end
