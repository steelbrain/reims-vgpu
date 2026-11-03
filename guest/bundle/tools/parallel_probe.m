// parallel_probe: a parallel render pass on the Metal device, against what Metal does. Each case is
// one pass into a 64 x 64 RGBA8Unorm target cleared to black, read back through a Shared buffer.
//   order     two sub-encoders, the second made encoded first: the second runs last (Metal runs
//             them in the order they were made), so the target is the second's green.
//   defaults  the first sub-encoder sets cull mode Back, a 1 x 1 viewport and a 1 x 1 scissor
//             rectangle and draws a red quad, which they cull; the second sets nothing and draws a
//             green quad over the target: each sub-encoder starts from the default state, so the
//             whole target is green.
//   threads   two sub-encoders encoded at once on two threads, 2,048 draws each, every draw a
//             one-column quad of its own with its rectangle and colour in set*Bytes: the left half
//             red, the right half green.
// One line a case, "parallel_probe: NAME OK" or "FAIL" with the first wrong pixel; exits 1 where
// a case failed. Shaders compile from source on the device.
// Build: xcrun clang -arch x86_64 -fobjc-arc -framework Metal -framework Foundation
//        parallel_probe.m -o parallel_probe        (-arch arm64 for an Apple-silicon Mac)
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <pthread.h>
#include <stdio.h>
#include <string.h>

enum { kSide = 64 };

static id<MTLDevice> gDevice;
static id<MTLCommandQueue> gQueue;
static id<MTLRenderPipelineState> gPipeline;
static id<MTLTexture> gTarget;
static int gFailed;

static NSString *const kSource =
    @"#include <metal_stdlib>\n"
    @"using namespace metal;\n"
    @"struct V { float4 pos [[position]]; };\n"
    @"vertex V vs(uint vid [[vertex_id]], constant float4 *rect [[buffer(0)]]) {\n"
    @"  float4 r = rect[0];\n"
    @"  float2 p[6] = {float2(r.x, r.y), float2(r.z, r.y), float2(r.x, r.w),\n"
    @"                 float2(r.z, r.y), float2(r.z, r.w), float2(r.x, r.w)};\n"
    @"  V o; o.pos = float4(p[vid], 0, 1); return o;\n"
    @"}\n"
    @"fragment float4 fs(constant float4 &c [[buffer(0)]]) { return c; }\n";

static const float kRed[4] = {1, 0, 0, 1};
static const float kGreen[4] = {0, 1, 0, 1};
static const float kWhole[4] = {-1, -1, 1, 1};

// A quad over `rect` (NDC x0, y0, x1, y1), in `colour`.
static void quad(id<MTLRenderCommandEncoder> e, const float rect[4], const float colour[4])
{
	[e setVertexBytes:rect length:16 atIndex:0];
	[e setFragmentBytes:colour length:16 atIndex:0];
	[e drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:6];
}

static id<MTLParallelRenderCommandEncoder> beginPass(id<MTLCommandBuffer> cb)
{
	MTLRenderPassDescriptor *rp = [MTLRenderPassDescriptor renderPassDescriptor];
	rp.colorAttachments[0].texture = gTarget;
	rp.colorAttachments[0].loadAction = MTLLoadActionClear;
	rp.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 1);
	rp.colorAttachments[0].storeAction = MTLStoreActionStore;
	return [cb parallelRenderCommandEncoderWithDescriptor:rp];
}

// Ends the pass's command buffer, reads the target back and checks each pixel against `want`
// (x -> RGBA8 as 0xAABBGGRR).
static void finish(const char *name, id<MTLCommandBuffer> cb, uint32_t (^want)(int x, int y))
{
	id<MTLBuffer> out = [gDevice newBufferWithLength:kSide * kSide * 4 options:MTLResourceStorageModeShared];
	id<MTLBlitCommandEncoder> b = [cb blitCommandEncoder];
	[b copyFromTexture:gTarget
	             sourceSlice:0
	             sourceLevel:0
	            sourceOrigin:MTLOriginMake(0, 0, 0)
	              sourceSize:MTLSizeMake(kSide, kSide, 1)
	                toBuffer:out
	       destinationOffset:0
	  destinationBytesPerRow:kSide * 4
	destinationBytesPerImage:kSide * kSide * 4];
	[b endEncoding];
	[cb commit];
	[cb waitUntilCompleted];
	if (cb.status != MTLCommandBufferStatusCompleted) {
		printf("parallel_probe: %s FAIL command buffer status %ld error %s\n", name, (long)cb.status,
		       cb.error.description.UTF8String);
		gFailed++;
		return;
	}
	const uint32_t *px = (const uint32_t *)out.contents;
	for (int y = 0; y < kSide; y++) {
		for (int x = 0; x < kSide; x++) {
			const uint32_t w = want(x, y), got = px[y * kSide + x];
			if (got != w) {
				printf("parallel_probe: %s FAIL pixel (%d, %d) is 0x%08x, not 0x%08x\n", name, x, y,
				       got, w);
				gFailed++;
				return;
			}
		}
	}
	printf("parallel_probe: %s OK\n", name);
}

static const uint32_t kRedPx = 0xff0000ff, kGreenPx = 0xff00ff00;

static void caseOrder(void)
{
	id<MTLCommandBuffer> cb = [gQueue commandBuffer];
	id<MTLParallelRenderCommandEncoder> pass = beginPass(cb);
	id<MTLRenderCommandEncoder> first = [pass renderCommandEncoder];
	id<MTLRenderCommandEncoder> second = [pass renderCommandEncoder];
	[second setRenderPipelineState:gPipeline];
	quad(second, kWhole, kGreen);
	[second endEncoding];
	[first setRenderPipelineState:gPipeline];
	quad(first, kWhole, kRed);
	[first endEncoding];
	[pass endEncoding];
	finish("order", cb, ^uint32_t(int x, int y) {
		(void)x, (void)y;
		return kGreenPx;
	});
}

static void caseDefaults(void)
{
	id<MTLCommandBuffer> cb = [gQueue commandBuffer];
	id<MTLParallelRenderCommandEncoder> pass = beginPass(cb);
	id<MTLRenderCommandEncoder> first = [pass renderCommandEncoder];
	id<MTLRenderCommandEncoder> second = [pass renderCommandEncoder];
	[first setRenderPipelineState:gPipeline];
	// The quad winds counter-clockwise, back-facing under the default Clockwise front face.
	[first setCullMode:MTLCullModeBack];
	[first setViewport:(MTLViewport){0, 0, 1, 1, 0, 1}];
	[first setScissorRect:(MTLScissorRect){0, 0, 1, 1}];
	quad(first, kWhole, kRed);
	[first endEncoding];
	[second setRenderPipelineState:gPipeline];
	quad(second, kWhole, kGreen);
	[second endEncoding];
	[pass endEncoding];
	finish("defaults", cb, ^uint32_t(int x, int y) {
		(void)x, (void)y;
		return kGreenPx;
	});
}

struct Columns {
	id<MTLRenderCommandEncoder> __unsafe_unretained encoder;
	int from;           // the first column
	const float *colour;
};

static void *drawColumns(void *arg)
{
	struct Columns *c = arg;
	id<MTLRenderCommandEncoder> e = c->encoder;
	[e setRenderPipelineState:gPipeline];
	for (int i = 0; i < 2048; i++) {
		const int x = c->from + (i % (kSide / 2));
		const float rect[4] = {-1 + 2.0f * x / kSide, -1, -1 + 2.0f * (x + 1) / kSide, 1};
		quad(e, rect, c->colour);
	}
	[e endEncoding];
	return NULL;
}

static void caseThreads(void)
{
	id<MTLCommandBuffer> cb = [gQueue commandBuffer];
	id<MTLParallelRenderCommandEncoder> pass = beginPass(cb);
	id<MTLRenderCommandEncoder> left = [pass renderCommandEncoder];
	id<MTLRenderCommandEncoder> right = [pass renderCommandEncoder];
	struct Columns l = {left, 0, kRed}, r = {right, kSide / 2, kGreen};
	pthread_t tl, tr;
	pthread_create(&tl, NULL, drawColumns, &l);
	pthread_create(&tr, NULL, drawColumns, &r);
	pthread_join(tl, NULL);
	pthread_join(tr, NULL);
	[pass endEncoding];
	finish("threads", cb, ^uint32_t(int x, int y) {
		(void)y;
		return x < kSide / 2 ? kRedPx : kGreenPx;
	});
}

int main(void)
{
	@autoreleasepool {
		gDevice = MTLCreateSystemDefaultDevice();
		if (gDevice == nil) {
			printf("parallel_probe: no Metal device\n");
			return 2;
		}
		printf("parallel_probe: device \"%s\"\n", gDevice.name.UTF8String);
		gQueue = [gDevice newCommandQueue];
		NSError *err = nil;
		id<MTLLibrary> lib = [gDevice newLibraryWithSource:kSource options:nil error:&err];
		MTLRenderPipelineDescriptor *pd = [MTLRenderPipelineDescriptor new];
		pd.vertexFunction = [lib newFunctionWithName:@"vs"];
		pd.fragmentFunction = [lib newFunctionWithName:@"fs"];
		pd.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;
		gPipeline = lib != nil ? [gDevice newRenderPipelineStateWithDescriptor:pd error:&err] : nil;
		if (gPipeline == nil) {
			printf("parallel_probe: no pipeline: %s\n", err.description.UTF8String);
			return 2;
		}
		MTLTextureDescriptor *td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
		                                                                              width:kSide
		                                                                             height:kSide
		                                                                          mipmapped:NO];
		td.usage = MTLTextureUsageRenderTarget;
		td.storageMode = MTLStorageModePrivate;
		gTarget = [gDevice newTextureWithDescriptor:td];
		caseOrder();
		caseDefaults();
		caseThreads();
	}
	printf("parallel_probe: %d failed\n", gFailed);
	return gFailed != 0;
}
