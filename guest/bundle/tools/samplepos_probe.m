// samplepos_probe: programmable sample positions in render passes, compared with what Metal gives
// on an M3 Max (macOS 27).
//   samplepos_probe [DIR]  loads DIR/samplepos_probe.metallib (default: the executable's
//                          directory), prints the device's answers (programmable positions, the
//                          default positions of 1, 2 and 4 samples, the sample counts it makes),
//                          then draws one triangle whose edge halves pixel column 1 of a 4 x 4,
//                          4-sample target in four passes, resolves each, and prints row 0 of
//                          each resolve:
//                          - left: four positions left of the pixel's centre (column 1 covered);
//                          - right: four right of it (column 1 uncovered);
//                          - default: none set (Metal's positions, two of four left);
//                          - depth: left's positions, with a 4-sample depth attachment whose
//                            store action the pass leaves Unknown, then sets on the encoder with
//                            setDepthStoreActionOptions: CustomSamplePositions.
// Prints "SAMPLEPOS OK" when every answer and value is Metal's, else "SAMPLEPOS FAIL: why" and
// exits 1. A completion 20 s late prints HANG and exits 3.
// Build: xcrun clang -arch x86_64 -fobjc-arc -framework Metal -framework Foundation
//        samplepos_probe.m -o samplepos_probe
//        xcrun -sdk macosx metal -c samplepos_probe.metal -o samplepos_probe.air
//        xcrun -sdk macosx metallib samplepos_probe.air -o samplepos_probe.metallib
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fail(const char *why)
{
	printf("SAMPLEPOS FAIL: %s\n", why);
	return 1;
}

// The float an IEEE half's bits hold.
static float halfToFloat(uint16_t h)
{
	const uint32_t sign = (h >> 15) & 1, exp = (h >> 10) & 0x1f, mant = h & 0x3ff;
	float v;
	if (exp == 0)
		v = ldexpf((float)mant, -24);
	else if (exp == 31)
		v = mant ? NAN : INFINITY;
	else
		v = ldexpf((float)(mant | 0x400), (int)exp - 25);
	return sign ? -v : v;
}

static id<MTLTexture> target(id<MTLDevice> d, MTLPixelFormat format, NSUInteger samples,
                             MTLTextureUsage usage)
{
	MTLTextureDescriptor *t = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format
	                                                                             width:4
	                                                                            height:4
	                                                                         mipmapped:NO];
	t.textureType = samples > 1 ? MTLTextureType2DMultisample : MTLTextureType2D;
	t.sampleCount = samples;
	t.storageMode = MTLStorageModePrivate;
	t.usage = usage;
	return [d newTextureWithDescriptor:t];
}

int main(int argc, char **argv)
{
	@autoreleasepool {
		id<MTLDevice> d = MTLCreateSystemDefaultDevice();
		if (!d)
			return fail("no device");
		int bad = 0;

		const BOOL programmable = d.programmableSamplePositionsSupported;
		printf("programmable %d\n", programmable);
		if (!programmable)
			bad |= 1;
		// Metal's defaults on the M3 Max: 1 (0.5, 0.5); 2 (0.75, 0.75) (0.25, 0.25); 4 (0.375,
		// 0.125) (0.875, 0.375) (0.125, 0.625) (0.625, 0.875).
		const float want1[2] = {0.5f, 0.5f};
		const float want2[4] = {0.75f, 0.75f, 0.25f, 0.25f};
		const float want4[8] = {0.375f, 0.125f, 0.875f, 0.375f, 0.125f, 0.625f, 0.625f, 0.875f};
		const float *wants[3] = {want1, want2, want4};
		const NSUInteger counts[3] = {1, 2, 4};
		for (int k = 0; k < 3; k++) {
			MTLSamplePosition p[4] = {};
			[d getDefaultSamplePositions:p count:counts[k]];
			printf("default %lu:", (unsigned long)counts[k]);
			for (NSUInteger i = 0; i < counts[k]; i++) {
				printf(" (%g, %g)", p[i].x, p[i].y);
				if (p[i].x != wants[k][2 * i] || p[i].y != wants[k][2 * i + 1])
					bad |= 2;
			}
			printf("\n");
		}
		printf("sample counts:");
		for (NSUInteger n = 1; n <= 8; n *= 2)
			printf(" %lu=%d", (unsigned long)n, [d supportsTextureSampleCount:n]);
		printf("\n");

		NSString *dir = argc > 1 ? @(argv[1])
		                         : [@(argv[0]) stringByDeletingLastPathComponent];
		NSURL *url = [NSURL fileURLWithPath:[dir stringByAppendingPathComponent:
		                                             @"samplepos_probe.metallib"]];
		NSError *e = nil;
		id<MTLLibrary> lib = [d newLibraryWithURL:url error:&e];
		if (!lib)
			return fail("no library");
		MTLRenderPipelineDescriptor *rd = [MTLRenderPipelineDescriptor new];
		rd.vertexFunction = [lib newFunctionWithName:@"samplepos_vertex"];
		rd.fragmentFunction = [lib newFunctionWithName:@"samplepos_fragment"];
		rd.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA16Float;
		rd.rasterSampleCount = 4;
		id<MTLRenderPipelineState> plain = [d newRenderPipelineStateWithDescriptor:rd error:&e];
		rd.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float;
		id<MTLRenderPipelineState> depthed = [d newRenderPipelineStateWithDescriptor:rd error:&e];
		if (!plain || !depthed)
			return fail("no pipeline");
		MTLDepthStencilDescriptor *dsd = [MTLDepthStencilDescriptor new];
		dsd.depthCompareFunction = MTLCompareFunctionLess;
		dsd.depthWriteEnabled = YES;
		id<MTLDepthStencilState> ds = [d newDepthStencilStateWithDescriptor:dsd];

		const MTLSamplePosition left[4] = {
		    MTLSamplePositionMake(0.125f, 0.125f), MTLSamplePositionMake(0.375f, 0.375f),
		    MTLSamplePositionMake(0.125f, 0.625f), MTLSamplePositionMake(0.375f, 0.875f)};
		const MTLSamplePosition right[4] = {
		    MTLSamplePositionMake(0.625f, 0.125f), MTLSamplePositionMake(0.875f, 0.375f),
		    MTLSamplePositionMake(0.625f, 0.625f), MTLSamplePositionMake(0.875f, 0.875f)};
		const MTLSamplePosition *sets[4] = {left, right, NULL, left};
		const char *names[4] = {"left", "right", "default", "depth"};
		// Row 0's red after the resolve: column 0 covered, 2 and 3 not, and column 1 the share
		// of its samples left of x = 1.5.
		const float wantRow[4][4] = {
		    {1, 1, 0, 0}, {1, 0, 0, 0}, {1, 0.5f, 0, 0}, {1, 1, 0, 0}};

		id<MTLCommandQueue> q = [d newCommandQueue];
		id<MTLCommandBuffer> cb = [q commandBuffer];
		id<MTLBuffer> out = [d newBufferWithLength:4 * 128 options:MTLResourceStorageModeShared];
		id<MTLTexture> resolves[4];
		for (int k = 0; k < 4; k++) {
			MTLRenderPassDescriptor *rp = [MTLRenderPassDescriptor renderPassDescriptor];
			rp.colorAttachments[0].texture =
			    target(d, MTLPixelFormatRGBA16Float, 4, MTLTextureUsageRenderTarget);
			resolves[k] = target(d, MTLPixelFormatRGBA16Float, 1,
			                     MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead);
			rp.colorAttachments[0].resolveTexture = resolves[k];
			rp.colorAttachments[0].loadAction = MTLLoadActionClear;
			rp.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 0);
			rp.colorAttachments[0].storeAction = MTLStoreActionMultisampleResolve;
			const BOOL depth = k == 3;
			if (depth) {
				rp.depthAttachment.texture =
				    target(d, MTLPixelFormatDepth32Float, 4, MTLTextureUsageRenderTarget);
				rp.depthAttachment.loadAction = MTLLoadActionClear;
				rp.depthAttachment.clearDepth = 1.0;
				rp.depthAttachment.storeAction = MTLStoreActionUnknown;
			}
			if (sets[k] != NULL) {
				[rp setSamplePositions:sets[k] count:4];
				MTLSamplePosition back[4];
				if ([rp getSamplePositions:NULL count:0] != 4)
					bad |= 4;
				[rp getSamplePositions:back count:4];
				if (memcmp(back, sets[k], sizeof(back)) != 0)
					bad |= 4;
			}
			id<MTLRenderCommandEncoder> enc = [cb renderCommandEncoderWithDescriptor:rp];
			if (!enc)
				return fail("no render encoder");
			[enc setRenderPipelineState:depth ? depthed : plain];
			if (depth) {
				[enc setDepthStencilState:ds];
				[enc setDepthStoreAction:MTLStoreActionStore];
				// Deprecated in the macOS 27 SDK (no effect on Apple silicon); a macOS 26
				// guest's apps still send it.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
				[enc setDepthStoreActionOptions:MTLStoreActionOptionCustomSamplePositions];
#pragma clang diagnostic pop
			}
			[enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
			[enc endEncoding];
		}
		id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
		for (int k = 0; k < 4; k++) {
			[blit copyFromTexture:resolves[k]
			                 sourceSlice:0
			                 sourceLevel:0
			                sourceOrigin:MTLOriginMake(0, 0, 0)
			                  sourceSize:MTLSizeMake(4, 4, 1)
			                    toBuffer:out
			           destinationOffset:(NSUInteger)k * 128
			      destinationBytesPerRow:32
			    destinationBytesPerImage:128];
		}
		[blit endEncoding];
		dispatch_semaphore_t done = dispatch_semaphore_create(0);
		[cb addCompletedHandler:^(id<MTLCommandBuffer> b) {
			dispatch_semaphore_signal(done);
		}];
		[cb commit];
		if (dispatch_semaphore_wait(done, dispatch_time(DISPATCH_TIME_NOW, 20 * NSEC_PER_SEC))) {
			printf("HANG\n");
			return 3;
		}
		printf("status %ld error %s\n", (long)cb.status,
		       cb.error ? cb.error.localizedDescription.UTF8String : "none");
		if (cb.status != MTLCommandBufferStatusCompleted)
			return fail("the command buffer did not complete");

		const uint16_t *px = (const uint16_t *)out.contents;
		for (int k = 0; k < 4; k++) {
			printf("%-8s", names[k]);
			for (int x = 0; x < 4; x++) {
				const uint16_t *p = px + k * 64 + x * 4;
				const float r = halfToFloat(p[0]), a = halfToFloat(p[3]);
				printf(" (%g, %g)", r, a);
				if (fabsf(r - wantRow[k][x]) > 1.0f / 512 || fabsf(a - wantRow[k][x]) > 1.0f / 512)
					bad |= 8;
			}
			printf("\n");
		}
		if (bad & 1)
			return fail("no programmable sample positions");
		if (bad & 2)
			return fail("default positions are not Metal's");
		if (bad & 4)
			return fail("the descriptor does not hold its positions");
		if (bad & 8)
			return fail("coverage is not Metal's");
		printf("SAMPLEPOS OK\n");
		return 0;
	}
}
