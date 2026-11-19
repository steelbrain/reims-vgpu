// Records Metal's answers to gpu/encode_render.rs's tessellation_states_tessellate_as_metals on
// this Mac's own Metal, in the lines of its file of answers beside this source
// (encode_render.tessellation_states_tessellate_as_metals.txt, the forms core's
// testkit::metal_answers reads). It draws what the test draws, with no part of this library in
// between: for each case, in the test's order, a render pipeline of render_tess_bump.metal's
// post-tessellation vertex function and its fragment function render_tess_facing into
// BGRA8Unorm, whose tessellation state is Metal's defaults but for the case's field; a pass over a
// Private 32 by 16 BGRA8Unorm target cleared to opaque black that draws the two patches, three
// control points each, with the factor buffer the test binds (two entries of
// MTLTriangleTessellationFactorsHalf, every factor 2.5 in the first and 1 in the second, at offset
// 0, instance stride 0); and a blit of the target into a Shared buffer. The case of the
// control-point index type draws render_tess_bump_cp, which reads its control points through
// [[stage_in]] (attribute 0, a float2 of buffer 0, stepping per patch control point, stride 8),
// by drawIndexedPatches over 16-bit indices into the points stored in reverse order, with the
// first patch's control points 0 and 1 swapped. Each case prints its line,
// `<key>\tbytes 2048 <FNV-1a 128> <first 32 bytes in hex>`, and to stderr its target, a row a
// line: '.' black, 'F' blue (a triangle that faces front), 'B' red (one that faces back).
//
// Provenance: run on the M3 Max under macOS 27.0.1 (26A434), 2026-11-18, with
// tests/shaders/render_tess_bump.metallib as committed; its lines are the file's.
//
// Build and run on a Mac, from the repository's root, with Xcode's tools:
//   T=$(mktemp -d)
//   xcrun -sdk macosx clang -O1 -framework Foundation -framework Metal \
//       crates/reims-vgpu-vulkan/src/gpu/metal_answers/encode_render.tessellation_states_tessellate_as_metals.m \
//       -o "$T/record"
//   "$T/record" tests/shaders/render_tess_bump.metallib
// Built without ARC: a run is short, and what it makes lives until it exits.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

// The target's width and height.
enum { W = 32, H = 16 };

// The two patches' control points, float2 each: the left patch's edge from control point 0 to 1
// runs left to right, the right patch's, its mirror, right to left.
static const float POINTS[12] = {
    -0.96f, 0.07f, -0.04f, 0.07f, -0.48f, 0.95f, 0.96f, 0.07f, 0.04f, 0.07f, 0.48f, 0.95f,
};

// The 16-bit control-point indices of the indexed draw, into POINTS stored in reverse order
// (element i is point 5 - i): points 1, 0, 2, then 3, 4, 5.
static const uint16_t INDICES[6] = {4, 5, 3, 2, 1, 0};

// The factor buffer: every factor of the first patch's entry 2.5, of the second's 1 (halves).
static const uint16_t FACTORS[8] = {0x4100, 0x4100, 0x4100, 0x4100, 0x3c00, 0x3c00, 0x3c00, 0x3c00};

// A case: its key, and the tessellation state's fields; `scale` is set on the encoder where it is
// not 0.
typedef struct {
	const char *key;
	MTLTessellationPartitionMode partition;
	NSUInteger max_factor;
	BOOL scale_enabled;
	MTLTessellationFactorStepFunction step;
	MTLWinding winding;
	MTLTessellationControlPointIndexType indices;
	float scale;
} Case;

static const Case CASES[] = {
    {"Metal's defaults", MTLTessellationPartitionModePow2, 16, NO,
     MTLTessellationFactorStepFunctionConstant, MTLWindingClockwise,
     MTLTessellationControlPointIndexTypeNone, 0},
    {"partition Integer", MTLTessellationPartitionModeInteger, 16, NO,
     MTLTessellationFactorStepFunctionConstant, MTLWindingClockwise,
     MTLTessellationControlPointIndexTypeNone, 0},
    {"partition FractionalOdd", MTLTessellationPartitionModeFractionalOdd, 16, NO,
     MTLTessellationFactorStepFunctionConstant, MTLWindingClockwise,
     MTLTessellationControlPointIndexTypeNone, 0},
    {"partition FractionalEven", MTLTessellationPartitionModeFractionalEven, 16, NO,
     MTLTessellationFactorStepFunctionConstant, MTLWindingClockwise,
     MTLTessellationControlPointIndexTypeNone, 0},
    {"max factor 2", MTLTessellationPartitionModePow2, 2, NO,
     MTLTessellationFactorStepFunctionConstant, MTLWindingClockwise,
     MTLTessellationControlPointIndexTypeNone, 0},
    {"winding CounterClockwise", MTLTessellationPartitionModePow2, 16, NO,
     MTLTessellationFactorStepFunctionConstant, MTLWindingCounterClockwise,
     MTLTessellationControlPointIndexTypeNone, 0},
    {"step PerPatch", MTLTessellationPartitionModePow2, 16, NO,
     MTLTessellationFactorStepFunctionPerPatch, MTLWindingClockwise,
     MTLTessellationControlPointIndexTypeNone, 0},
    {"factor scale 2", MTLTessellationPartitionModePow2, 16, YES,
     MTLTessellationFactorStepFunctionConstant, MTLWindingClockwise,
     MTLTessellationControlPointIndexTypeNone, 2.0f},
    {"control point indices UInt16", MTLTessellationPartitionModePow2, 16, NO,
     MTLTessellationFactorStepFunctionConstant, MTLWindingClockwise,
     MTLTessellationControlPointIndexTypeUInt16, 0},
};

// FNV-1a over `bytes`, 128 bits, as 32 hex digits.
static void fnv128(const uint8_t *bytes, size_t n, char out[33]) {
	unsigned __int128 h = ((unsigned __int128)0x6c62272e07bb0142ULL << 64) | 0x62b821756295c58dULL;
	const unsigned __int128 prime = ((unsigned __int128)0x0000000001000000ULL << 64) | 0x13bULL;
	for (size_t i = 0; i < n; i++) {
		h ^= bytes[i];
		h *= prime;
	}
	snprintf(out, 33, "%016llx%016llx", (unsigned long long)(h >> 64), (unsigned long long)h);
}

int main(int argc, char **argv) {
	if (argc != 2) {
		fprintf(stderr, "usage: %s render_tess_bump.metallib\n", argv[0]);
		return 2;
	}
	id<MTLDevice> device = MTLCreateSystemDefaultDevice();
	NSError *error = nil;
	id<MTLLibrary> library = [device newLibraryWithURL:[NSURL fileURLWithPath:@(argv[1])] error:&error];
	if (!library) {
		fprintf(stderr, "library: %s\n", error.description.UTF8String);
		return 1;
	}
	id<MTLCommandQueue> queue = [device newCommandQueue];
	float reversed[12];
	for (int i = 0; i < 6; i++) {
		reversed[2 * i] = POINTS[2 * (5 - i)];
		reversed[2 * i + 1] = POINTS[2 * (5 - i) + 1];
	}
	id<MTLBuffer> points = [device newBufferWithBytes:POINTS length:sizeof POINTS options:0];
	id<MTLBuffer> staged = [device newBufferWithBytes:reversed length:sizeof reversed options:0];
	id<MTLBuffer> indices = [device newBufferWithBytes:INDICES length:sizeof INDICES options:0];
	id<MTLBuffer> factors = [device newBufferWithBytes:FACTORS length:sizeof FACTORS options:0];
	id<MTLBuffer> out = [device newBufferWithLength:W * H * 4 options:0];
	MTLTextureDescriptor *td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
	                                                                              width:W
	                                                                             height:H
	                                                                          mipmapped:NO];
	td.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
	td.storageMode = MTLStorageModePrivate;
	for (size_t n = 0; n < sizeof CASES / sizeof CASES[0]; n++) {
		const Case *c = &CASES[n];
		BOOL by_index = c->indices != MTLTessellationControlPointIndexTypeNone;
		MTLRenderPipelineDescriptor *pd = [MTLRenderPipelineDescriptor new];
		pd.vertexFunction = [library newFunctionWithName:by_index ? @"render_tess_bump_cp" : @"render_tess_bump"];
		pd.fragmentFunction = [library newFunctionWithName:@"render_tess_facing"];
		pd.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
		pd.tessellationPartitionMode = c->partition;
		pd.maxTessellationFactor = c->max_factor;
		pd.tessellationFactorScaleEnabled = c->scale_enabled;
		pd.tessellationFactorFormat = MTLTessellationFactorFormatHalf;
		pd.tessellationControlPointIndexType = c->indices;
		pd.tessellationFactorStepFunction = c->step;
		pd.tessellationOutputWindingOrder = c->winding;
		if (by_index) {
			MTLVertexDescriptor *vd = [MTLVertexDescriptor vertexDescriptor];
			vd.attributes[0].format = MTLVertexFormatFloat2;
			vd.attributes[0].offset = 0;
			vd.attributes[0].bufferIndex = 0;
			vd.layouts[0].stride = 8;
			vd.layouts[0].stepFunction = MTLVertexStepFunctionPerPatchControlPoint;
			vd.layouts[0].stepRate = 1;
			pd.vertexDescriptor = vd;
		}
		id<MTLRenderPipelineState> ps = [device newRenderPipelineStateWithDescriptor:pd error:&error];
		if (!ps) {
			fprintf(stderr, "%s: pipeline: %s\n", c->key, error.description.UTF8String);
			return 1;
		}
		id<MTLTexture> target = [device newTextureWithDescriptor:td];
		MTLRenderPassDescriptor *rp = [MTLRenderPassDescriptor renderPassDescriptor];
		rp.colorAttachments[0].texture = target;
		rp.colorAttachments[0].loadAction = MTLLoadActionClear;
		rp.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 1);
		rp.colorAttachments[0].storeAction = MTLStoreActionStore;
		id<MTLCommandBuffer> cb = [queue commandBuffer];
		id<MTLRenderCommandEncoder> e = [cb renderCommandEncoderWithDescriptor:rp];
		[e setRenderPipelineState:ps];
		[e setVertexBuffer:by_index ? staged : points offset:0 atIndex:0];
		[e setTessellationFactorBuffer:factors offset:0 instanceStride:0];
		if (c->scale != 0) {
			[e setTessellationFactorScale:c->scale];
		}
		if (by_index) {
			[e drawIndexedPatches:3
			                       patchStart:0
			                       patchCount:2
			                 patchIndexBuffer:nil
			           patchIndexBufferOffset:0
			          controlPointIndexBuffer:indices
			    controlPointIndexBufferOffset:0
			                    instanceCount:1
			                     baseInstance:0];
		} else {
			[e drawPatches:3
			                patchStart:0
			                patchCount:2
			          patchIndexBuffer:nil
			    patchIndexBufferOffset:0
			             instanceCount:1
			              baseInstance:0];
		}
		[e endEncoding];
		id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
		[blit copyFromTexture:target
		                 sourceSlice:0
		                 sourceLevel:0
		                sourceOrigin:MTLOriginMake(0, 0, 0)
		                  sourceSize:MTLSizeMake(W, H, 1)
		                    toBuffer:out
		           destinationOffset:0
		      destinationBytesPerRow:W * 4
		    destinationBytesPerImage:W * H * 4];
		[blit endEncoding];
		[cb commit];
		[cb waitUntilCompleted];
		if (cb.error) {
			fprintf(stderr, "%s: %s\n", c->key, cb.error.description.UTF8String);
			return 1;
		}
		const uint8_t *texels = out.contents;
		char digest[33];
		fnv128(texels, W * H * 4, digest);
		printf("%s\tbytes %d %s ", c->key, W * H * 4, digest);
		for (int i = 0; i < 32; i++) {
			printf("%02x", texels[i]);
		}
		printf("\n");
		fprintf(stderr, "%s\n", c->key);
		for (int y = 0; y < H; y++) {
			char row[W + 1];
			for (int x = 0; x < W; x++) {
				const uint8_t *t = texels + 4 * (y * W + x);
				uint32_t v = (uint32_t)t[0] | (uint32_t)t[1] << 8 | (uint32_t)t[2] << 16 | (uint32_t)t[3] << 24;
				row[x] = v == 0xff000000 ? '.' : v == 0xff0000ff ? 'F' : v == 0xffff0000 ? 'B' : '?';
			}
			row[W] = 0;
			fprintf(stderr, "  %s\n", row);
		}
	}
	return 0;
}
