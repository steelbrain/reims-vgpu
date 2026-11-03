// bench: a GPU-heavy benchmark of the Metal device, to compare a guest's GPU lane with another
// lane on the same host, and with real Macs. Every case warms up first, then runs REPS
// repetitions and prints the median of each number (and the spread of the first).
//   bench [-d NAME] [-r REPS] [-m SECONDS] [-i ITERS] [-t SECONDS] [-j THREADS] [-v] [-l] [CASE...]
//     -d NAME     the first of MTLCopyAllDevices whose name holds NAME; the default is
//                 MTLCreateSystemDefaultDevice (which, on a dual-GPU MacBook, switches the
//                 display to the discrete GPU: pass -d there).
//     -r REPS     repetitions per case (default 5).
//     -m SECONDS  more repetitions, until a case has run this long (default 0): long enough for
//                 a sampler of the host's CPU and GPU to see each case.
//     -i ITERS    the heavy fragment shader's loop count (default 256).
//     -t SECONDS  how long `onscreen` animates (default 20).
//     -j THREADS  how many threads encode and commit `draws.10k`'s command buffers at once, onto
//                 its one queue (default 1).
//     -v          a "# rep" line for each repetition of the frame cases (frag.*): when it
//                 started, its frames (one command buffer each) and their rate.
//     -l          lists the devices and exits.
//     CASE        runs only the cases whose name holds one of these; with none, every case
//                 but `onscreen`, which needs the logged-in user's window server session.
// The cases:
//   frag.alu.1080p, frag.alu.2160p  a full-screen triangle with a heavy fragment shader (an
//               ITERS-long FMA loop), offscreen into a Private RGBA8 target, one command buffer
//               a frame, back to back: fps from wall time, gpu_ms from each command buffer's
//               GPUStartTime..GPUEndTime, mpix_s = pixels shaded a second.
//   frag.fill.2160p  16 full-screen layers of a flat colour, blended, a frame: gpix_s.
//   compute.fma      2^20 threads of 8 independent float4 FMA chains: gflops.
//   blit.256m        a 256 MiB Private-to-Private blit copy: gbps = bytes copied a second (the
//               memory moves twice that, read and write).
//   sample.tex       a compute kernel taking 16 bilinear samples a thread from a 4096 x 4096
//               Private RGBA8 texture: gsamples_s.
//   draws.10k        10,000 small draws a command buffer, each with its own setVertexBytes, into
//               a 512 x 512 target, 3 command buffers back to back on each of THREADS threads,
//               one queue: draws_s from wall time, and the CPU's encode time and the GPU's time of
//               one command buffer.
//   upload.64m       the CPU writes a 64 MiB Shared buffer, the GPU blits it into a Private one:
//               mbps end to end (write start to blit done), and the CPU write's own rate.
//   readback.64m     the GPU writes a 64 MiB Shared buffer (a kernel), the CPU copies it out and
//               checks it: mbps end to end (commit to copy done), and the CPU read's own rate.
//   mps.sgemm.2048, mps.sgemm.4096  MPSMatrixMultiplication, FP32, square: gflops.
//   onscreen         a full-screen borderless window whose CAMetalLayer the heavy shader animates
//               at display rate for SECONDS: presented fps (from the drawables' presentedTime),
//               submitted fps, frames the system dropped, gpu_ms a frame.
// One line a case: "bench: NAME OK key=value ...", or "bench: NAME FAIL why" / "SKIP why";
// "# start NAME" before it, so a case that hangs or kills the process is known by the last start
// line. Shaders compile from source at run time on the device (newLibraryWithSource:).
// Build: make bench (build/bench-x86_64, a guest's or an Intel Mac's); make bench ARCH=arm64 for
// an Apple-silicon Mac.
#import <AppKit/AppKit.h>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <QuartzCore/QuartzCore.h>

#include <mach/mach_time.h>
#include <stdatomic.h>
#include <sys/sysctl.h>

#pragma clang diagnostic ignored "-Wunguarded-availability-new"

static id<MTLDevice> gDevice;
static id<MTLCommandQueue> gQueue;
static id<MTLLibrary> gLib;
static NSArray<NSString *> *gPatterns;
static int gReps = 5;
static uint32_t gIters = 256;
static double gSeconds = 20;
// -j: the threads that encode draws.10k at once.
static int gThreads = 1;
// The wall time, commit to completion, of the last command buffer a case's run block waited for.
static double gWall;
// -m: the least time a case's repetitions take; when the case started.
static double gMinSeconds, gCaseStart;
// -v: a line a repetition of the frame cases; the case running.
static BOOL gVerbose;
static NSString *gCaseName;

static NSString *const kShaders = @"#include <metal_stdlib>\n"
"using namespace metal;\n"
"struct VOut { float4 pos [[position]]; };\n"
// One triangle that covers the target: (-1,-1), (3,-1), (-1,3).
"vertex VOut v_full(uint vid [[vertex_id]]) {\n"
"  float2 p = float2((vid << 1) & 2, vid & 2);\n"
"  VOut o; o.pos = float4(p * 2.0 - 1.0, 0.0, 1.0); return o;\n"
"}\n"
"struct Heavy { float t; uint iters; };\n"
"fragment float4 f_heavy(VOut in [[stage_in]], constant Heavy &a [[buffer(0)]]) {\n"
"  float4 x = float4(in.pos.xy * 0.00025, a.t, 1.0);\n"
"  float4 y = x.yzwx + 0.5;\n"
"  const float4 m = float4(0.9995, 0.9993, 0.9991, 0.9997);\n"
"  const float4 c = float4(0.0003, 0.0005, 0.0007, 0.0002);\n"
"  for (uint i = 0; i < a.iters; i++) {\n"
"    x = fma(x, m, y * c);\n"
"    y = fma(y, m, x * c);\n"
"  }\n"
"  return float4(fract(x.xyz + y.yzw), 1.0);\n"
"}\n"
"fragment half4 f_flat(constant float4 &c [[buffer(0)]]) { return half4(c); }\n"
// A small triangle at d.xy, d.zw in size, in clip space.
"vertex VOut v_small(uint vid [[vertex_id]], constant float4 &d [[buffer(0)]]) {\n"
"  float2 p = float2((vid << 1) & 2, vid & 2) * 0.5;\n"
"  VOut o; o.pos = float4(d.xy + p * d.zw, 0.0, 1.0); return o;\n"
"}\n"
"kernel void k_fma(device float4 *out [[buffer(0)]], constant float4 *mc [[buffer(1)]],\n"
"                  constant uint &iters [[buffer(2)]], uint gid [[thread_position_in_grid]]) {\n"
"  const float4 m = mc[0], c = mc[1];\n"
"  float4 a0 = float4(float(gid) * 1e-7), a1 = a0 + 0.1, a2 = a0 + 0.2, a3 = a0 + 0.3;\n"
"  float4 a4 = a0 + 0.4, a5 = a0 + 0.5, a6 = a0 + 0.6, a7 = a0 + 0.7;\n"
"  for (uint i = 0; i < iters; i++) {\n"
"    a0 = fma(a0, m, c); a1 = fma(a1, m, c); a2 = fma(a2, m, c); a3 = fma(a3, m, c);\n"
"    a4 = fma(a4, m, c); a5 = fma(a5, m, c); a6 = fma(a6, m, c); a7 = fma(a7, m, c);\n"
"  }\n"
"  out[gid] = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;\n"
"}\n"
"kernel void k_tex_init(texture2d<float, access::write> t [[texture(0)]],\n"
"                       uint2 gid [[thread_position_in_grid]]) {\n"
"  t.write(float4(float(gid.x & 255) / 255.0, float(gid.y & 255) / 255.0,\n"
"                 float((gid.x ^ gid.y) & 255) / 255.0, 1.0), gid);\n"
"}\n"
"kernel void k_sample(texture2d<float> t [[texture(0)]], device float *out [[buffer(0)]],\n"
"                     uint2 gid [[thread_position_in_grid]], uint2 grid [[threads_per_grid]]) {\n"
"  constexpr sampler s(filter::linear, address::repeat, coord::normalized);\n"
"  const float2 texel = 1.0 / float2(t.get_width(), t.get_height());\n"
"  const float2 base = (float2(gid) * 2.0 + 0.5) * texel;\n"
"  float4 acc = 0.0;\n"
"  for (int k = 0; k < 16; k++) {\n"
"    float2 o = float2(k & 3, k >> 2) * 1.37 - 2.0;\n"
"    acc += t.sample(s, base + o * texel);\n"
"  }\n"
"  out[gid.y * grid.x + gid.x] = acc.x + acc.y + acc.z + acc.w;\n"
"}\n"
"kernel void k_pattern(device uint *out [[buffer(0)]], constant uint &seed [[buffer(1)]],\n"
"                      uint gid [[thread_position_in_grid]]) { out[gid] = gid ^ seed; }\n";

// --- helpers ------------------------------------------------------------------------------------

static double now(void)
{
	static mach_timebase_info_data_t tb;
	if (tb.denom == 0)
		mach_timebase_info(&tb);
	return (double)mach_absolute_time() * tb.numer / tb.denom / 1e9;
}

// Whether a case runs its repetition `r`: the first REPS, then more until it has run -m seconds
// (at most 10,000).
static BOOL more(int r)
{
	return r < gReps || (r < 10000 && now() - gCaseStart < gMinSeconds);
}

static double median(NSArray<NSNumber *> *xs)
{
	if (xs.count == 0)
		return NAN;
	NSArray *s = [xs sortedArrayUsingSelector:@selector(compare:)];
	const NSUInteger n = s.count;
	return n % 2 ? [s[n / 2] doubleValue] : ([s[n / 2 - 1] doubleValue] + [s[n / 2] doubleValue]) / 2;
}

static double minOf(NSArray<NSNumber *> *xs)
{
	return [[xs valueForKeyPath:@"@min.self"] doubleValue];
}

static double maxOf(NSArray<NSNumber *> *xs)
{
	return [[xs valueForKeyPath:@"@max.self"] doubleValue];
}

// A command buffer's GPU time in ms, NaN where the device gives none.
static double gpuMs(id<MTLCommandBuffer> cb)
{
	const double s = cb.GPUStartTime, e = cb.GPUEndTime;
	return (s > 0 && e >= s) ? (e - s) * 1e3 : NAN;
}

// Waits for `cb`, and throws where it failed.
static void finish(id<MTLCommandBuffer> cb)
{
	[cb waitUntilCompleted];
	if (cb.status != MTLCommandBufferStatusCompleted)
		[NSException raise:@"bench" format:@"command buffer status %ld: %@", (long)cb.status,
		                                   cb.error.localizedDescription ?: @"no error"];
}

static id<MTLComputePipelineState> computePipeline(NSString *name)
{
	NSError *err = nil;
	id<MTLFunction> f = [gLib newFunctionWithName:name];
	id<MTLComputePipelineState> p = f ? [gDevice newComputePipelineStateWithFunction:f error:&err] : nil;
	if (p == nil)
		[NSException raise:@"bench" format:@"compute pipeline %@: %@", name, err.localizedDescription ?: @"no function"];
	return p;
}

static id<MTLRenderPipelineState> renderPipeline(NSString *vertex, NSString *fragment, MTLPixelFormat format, BOOL blend)
{
	MTLRenderPipelineDescriptor *d = [MTLRenderPipelineDescriptor new];
	d.vertexFunction = [gLib newFunctionWithName:vertex];
	d.fragmentFunction = [gLib newFunctionWithName:fragment];
	d.colorAttachments[0].pixelFormat = format;
	if (blend) {
		d.colorAttachments[0].blendingEnabled = YES;
		d.colorAttachments[0].sourceRGBBlendFactor = MTLBlendFactorSourceAlpha;
		d.colorAttachments[0].destinationRGBBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
		d.colorAttachments[0].sourceAlphaBlendFactor = MTLBlendFactorOne;
		d.colorAttachments[0].destinationAlphaBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
	}
	NSError *err = nil;
	id<MTLRenderPipelineState> p = [gDevice newRenderPipelineStateWithDescriptor:d error:&err];
	if (p == nil)
		[NSException raise:@"bench" format:@"render pipeline %@/%@: %@", vertex, fragment, err.localizedDescription];
	return p;
}

static id<MTLTexture> target(NSUInteger w, NSUInteger h)
{
	MTLTextureDescriptor *d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
	                                                                             width:w
	                                                                            height:h
	                                                                         mipmapped:NO];
	d.storageMode = MTLStorageModePrivate;
	d.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
	return [gDevice newTextureWithDescriptor:d];
}

static MTLRenderPassDescriptor *pass(id<MTLTexture> t, MTLLoadAction load)
{
	MTLRenderPassDescriptor *p = [MTLRenderPassDescriptor renderPassDescriptor];
	p.colorAttachments[0].texture = t;
	p.colorAttachments[0].loadAction = load;
	p.colorAttachments[0].storeAction = MTLStoreActionStore;
	p.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 1);
	return p;
}

static BOOL wanted(NSString *name, BOOL byDefault)
{
	if (gPatterns.count == 0)
		return byDefault;
	for (NSString *p in gPatterns)
		if ([name containsString:p])
			return YES;
	return NO;
}

// Runs the case `name` where the command line wants it: its block returns the line's values
// ("key=value ..."), or throws to fail; a line starting "SKIP " skips.
static void runCase(NSString *name, BOOL byDefault, NSString * (^body)(void))
{
	if (!wanted(name, byDefault))
		return;
	printf("# start %s\n", name.UTF8String);
	fflush(stdout);
	const double t0 = now();
	gCaseStart = t0;
	gCaseName = name;
	NSString *line = nil;
	@try {
		@autoreleasepool {
			line = body();
		}
	} @catch (NSException *e) {
		line = [@"FAIL " stringByAppendingString:e.reason ?: e.name];
	}
	if (![line hasPrefix:@"FAIL "] && ![line hasPrefix:@"SKIP "])
		line = [@"OK " stringByAppendingString:line];
	printf("bench: %s %s case_s=%.1f\n", name.UTF8String, line.UTF8String, now() - t0);
	fflush(stdout);
}

// The spread of a case's first number: "spread=MIN..MAX".
static NSString *spread(NSArray<NSNumber *> *xs, const char *fmt)
{
	NSString *f = [NSString stringWithFormat:@"spread=%s..%s", fmt, fmt];
	return [NSString stringWithFormat:f, minOf(xs), maxOf(xs)];
}

// --- the cases ----------------------------------------------------------------------------------

typedef struct {
	float t;
	uint32_t iters;
} Heavy;

static void encodeHeavy(id<MTLCommandBuffer> cb, id<MTLRenderPipelineState> p, id<MTLTexture> t, float time)
{
	id<MTLRenderCommandEncoder> e = [cb renderCommandEncoderWithDescriptor:pass(t, MTLLoadActionDontCare)];
	[e setRenderPipelineState:p];
	Heavy h = {time, gIters};
	[e setFragmentBytes:&h length:sizeof h atIndex:0];
	[e drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
	[e endEncoding];
}

// The GPU time a batch of command buffers spans, first start to last end, in ms: they may run
// at once, so a buffer's own GPUStartTime..GPUEndTime counts its neighbours' work too.
static double gpuSpanMs(NSArray<id<MTLCommandBuffer>> *cbs)
{
	double s = INFINITY, e = 0;
	for (id<MTLCommandBuffer> cb in cbs) {
		if (cb.GPUStartTime <= 0 || cb.GPUEndTime < cb.GPUStartTime)
			return NAN;
		s = fmin(s, cb.GPUStartTime);
		e = fmax(e, cb.GPUEndTime);
	}
	return (e - s) * 1e3;
}

// Waits for every buffer of `cbs`: a later one can complete before an earlier one.
static void finishAll(NSArray<id<MTLCommandBuffer>> *cbs)
{
	for (id<MTLCommandBuffer> cb in cbs)
		finish(cb);
}

// Frames of `encode` back to back, one command buffer each: frames a second by wall time (first
// commit to the last completion), and the GPU time a frame by the batch's span (gpu) and by
// frames run alone, committed and waited one at a time (iso). Frames a repetition: about 0.4 s
// of them, by the warm-up frame's time.
static void frames(void (^encode)(id<MTLCommandBuffer>, int), NSMutableArray *fps, NSMutableArray *gpu,
                   NSMutableArray *wall, NSMutableArray *iso)
{
	double one = 0;
	for (int i = 0; i < 4; i++) {
		id<MTLCommandBuffer> w = [gQueue commandBuffer];
		encode(w, 0);
		const double t0 = now();
		[w commit];
		finish(w);
		one = now() - t0;
		if (i > 0)
			[iso addObject:@(gpuMs(w))];
	}
	const int n = (int)fmax(3, fmin(240, 0.4 / fmax(one, 1e-4)));
	for (int r = 0; more(r); r++) {
		NSMutableArray *cbs = [NSMutableArray array];
		const double s = now();
		for (int i = 0; i < n; i++) {
			id<MTLCommandBuffer> cb = [gQueue commandBuffer];
			encode(cb, i + 1);
			[cb commit];
			[cbs addObject:cb];
		}
		finishAll(cbs);
		const double el = now() - s;
		[fps addObject:@(n / el)];
		if (gVerbose) {
			printf("# rep %s %d at %.3f s: %d frames (command buffers) in %.1f ms, %.1f a second, gpu span %.3f ms a frame\n",
			       gCaseName.UTF8String, r, s - gCaseStart, n, el * 1e3, n / el, gpuSpanMs(cbs) / n);
			fflush(stdout);
		}
		[wall addObject:@(el * 1e3 / n)];
		[gpu addObject:@(gpuSpanMs(cbs) / n)];
	}
}

static NSString *fragAlu(NSUInteger w, NSUInteger h)
{
	id<MTLRenderPipelineState> p = renderPipeline(@"v_full", @"f_heavy", MTLPixelFormatRGBA8Unorm, NO);
	id<MTLTexture> t = target(w, h);
	NSMutableArray *fps = [NSMutableArray array], *gpu = [NSMutableArray array], *wall = [NSMutableArray array],
	               *iso = [NSMutableArray array];
	frames(^(id<MTLCommandBuffer> cb, int i) { encodeHeavy(cb, p, t, i * 0.001f); }, fps, gpu, wall, iso);
	const double f = median(fps);
	return [NSString stringWithFormat:@"fps=%.1f gpu_ms=%.3f gpu_iso_ms=%.3f wall_ms=%.3f mpix_s=%.0f iters=%u %@", f,
	                                  median(gpu), median(iso), median(wall), f * w * h / 1e6, gIters,
	                                  spread(fps, "%.1f")];
}

static NSString *fragFill(NSUInteger w, NSUInteger h, int layers)
{
	id<MTLRenderPipelineState> p = renderPipeline(@"v_full", @"f_flat", MTLPixelFormatRGBA8Unorm, YES);
	id<MTLTexture> t = target(w, h);
	NSMutableArray *fps = [NSMutableArray array], *gpu = [NSMutableArray array], *wall = [NSMutableArray array],
	               *iso = [NSMutableArray array];
	frames(
	    ^(id<MTLCommandBuffer> cb, int i) {
		    id<MTLRenderCommandEncoder> e = [cb renderCommandEncoderWithDescriptor:pass(t, MTLLoadActionClear)];
		    [e setRenderPipelineState:p];
		    for (int l = 0; l < layers; l++) {
			    const float c[4] = {(l & 1) * 0.8f, (l & 2) * 0.4f, (l & 4) * 0.2f, 0.5f};
			    [e setFragmentBytes:c length:sizeof c atIndex:0];
			    [e drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
		    }
		    [e endEncoding];
	    },
	    fps, gpu, wall, iso);
	const double f = median(fps), g = median(gpu);
	return [NSString
	    stringWithFormat:@"gpix_s=%.2f fps=%.1f gpu_ms=%.3f gpu_iso_ms=%.3f wall_ms=%.3f gpix_s_gpu=%.2f layers=%d %@",
	                     f * w * h * layers / 1e9, f, g, median(iso), median(wall), w * h * layers / (g * 1e6), layers,
	                     spread(fps, "%.1f")];
}

static NSString *computeFma(void)
{
	id<MTLComputePipelineState> p = computePipeline(@"k_fma");
	const NSUInteger threads = 1u << 20, tg = 256;
	id<MTLBuffer> out = [gDevice newBufferWithLength:threads * 16 options:MTLResourceStorageModePrivate];
	static const float mc[8] = {0.9999f, 0.9999f, 0.9999f, 0.9999f, 1e-5f, 1e-5f, 1e-5f, 1e-5f};
	__block uint32_t iters = 64;
	id<MTLCommandBuffer> (^run)(void) = ^{
		id<MTLCommandBuffer> cb = [gQueue commandBuffer];
		id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
		[e setComputePipelineState:p];
		[e setBuffer:out offset:0 atIndex:0];
		[e setBytes:mc length:sizeof mc atIndex:1];
		[e setBytes:&iters length:sizeof iters atIndex:2];
		[e dispatchThreadgroups:MTLSizeMake(threads / tg, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
		[e endEncoding];
		const double s = now();
		[cb commit];
		finish(cb);
		gWall = now() - s;
		return cb;
	};
	// Calibrate to about 100 ms a dispatch.
	for (int k = 0; k < 8; k++) {
		id<MTLCommandBuffer> cb = run();
		const double ms = isnan(gpuMs(cb)) ? gWall * 1e3 : gpuMs(cb);
		if (ms > 50 || iters >= 65536)
			break;
		iters = (uint32_t)fmin(65536, iters * fmax(2, fmin(16, 100 / fmax(ms, 0.1))));
	}
	NSMutableArray *gf = [NSMutableArray array], *gfw = [NSMutableArray array], *gpu = [NSMutableArray array];
	const double flops = (double)threads * iters * 8 * 4 * 2;
	for (int r = 0; more(r); r++) {
		id<MTLCommandBuffer> cb = run();
		const double g = gpuMs(cb), w = gWall * 1e3;
		[gf addObject:@(flops / (g * 1e6))];
		[gfw addObject:@(flops / (w * 1e6))];
		[gpu addObject:@(g)];
	}
	return [NSString stringWithFormat:@"gflops=%.0f gflops_wall=%.0f gpu_ms=%.2f iters=%u %@", median(gf),
	                                  median(gfw), median(gpu), iters, spread(gf, "%.0f")];
}

static NSString *blitCopy(NSUInteger bytes)
{
	id<MTLBuffer> a = [gDevice newBufferWithLength:bytes options:MTLResourceStorageModePrivate];
	id<MTLBuffer> b = [gDevice newBufferWithLength:bytes options:MTLResourceStorageModePrivate];
	if (a == nil || b == nil)
		return @"FAIL no 256 MiB Private buffers";
	id<MTLCommandBuffer> (^run)(int) = ^(int copies) {
		id<MTLCommandBuffer> cb = [gQueue commandBuffer];
		id<MTLBlitCommandEncoder> e = [cb blitCommandEncoder];
		for (int i = 0; i < copies; i++)
			[e copyFromBuffer:(i & 1) ? b : a sourceOffset:0 toBuffer:(i & 1) ? a : b destinationOffset:0 size:bytes];
		[e endEncoding];
		const double s = now();
		[cb commit];
		finish(cb);
		gWall = now() - s;
		return cb;
	};
	{
		id<MTLCommandBuffer> cb = [gQueue commandBuffer];
		id<MTLBlitCommandEncoder> e = [cb blitCommandEncoder];
		[e fillBuffer:a range:NSMakeRange(0, bytes) value:0x5a];
		[e endEncoding];
		[cb commit];
		finish(cb);
	}
	run(1);
	NSMutableArray *gb = [NSMutableArray array], *gbw = [NSMutableArray array], *gpu = [NSMutableArray array];
	const int copies = 4;
	for (int r = 0; more(r); r++) {
		id<MTLCommandBuffer> cb = run(copies);
		const double g = gpuMs(cb), w = gWall * 1e3;
		[gb addObject:@((double)bytes * copies / (g * 1e6))];
		[gbw addObject:@((double)bytes * copies / (w * 1e6))];
		[gpu addObject:@(g / copies)];
	}
	return [NSString stringWithFormat:@"gbps=%.1f gbps_wall=%.1f gpu_ms=%.2f copies=%d %@", median(gb), median(gbw),
	                                  median(gpu), copies, spread(gb, "%.1f")];
}

static NSString *sampleTex(void)
{
	const NSUInteger n = 4096, grid = 2048, tg = 16;
	MTLTextureDescriptor *d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
	                                                                             width:n
	                                                                            height:n
	                                                                         mipmapped:NO];
	d.storageMode = MTLStorageModePrivate;
	d.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
	id<MTLTexture> t = [gDevice newTextureWithDescriptor:d];
	id<MTLBuffer> out = [gDevice newBufferWithLength:grid * grid * 4 options:MTLResourceStorageModePrivate];
	id<MTLComputePipelineState> init = computePipeline(@"k_tex_init"), sample = computePipeline(@"k_sample");
	{
		id<MTLCommandBuffer> cb = [gQueue commandBuffer];
		id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
		[e setComputePipelineState:init];
		[e setTexture:t atIndex:0];
		[e dispatchThreadgroups:MTLSizeMake(n / tg, n / tg, 1) threadsPerThreadgroup:MTLSizeMake(tg, tg, 1)];
		[e endEncoding];
		[cb commit];
		finish(cb);
	}
	const int passes = 32;
	id<MTLCommandBuffer> (^run)(void) = ^{
		id<MTLCommandBuffer> cb = [gQueue commandBuffer];
		id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
		[e setComputePipelineState:sample];
		[e setTexture:t atIndex:0];
		[e setBuffer:out offset:0 atIndex:0];
		for (int i = 0; i < passes; i++)
			[e dispatchThreadgroups:MTLSizeMake(grid / tg, grid / tg, 1) threadsPerThreadgroup:MTLSizeMake(tg, tg, 1)];
		[e endEncoding];
		const double s = now();
		[cb commit];
		finish(cb);
		gWall = now() - s;
		return cb;
	};
	run();
	NSMutableArray *gs = [NSMutableArray array], *gsw = [NSMutableArray array], *gpu = [NSMutableArray array];
	const double samples = (double)grid * grid * 16 * passes;
	for (int r = 0; more(r); r++) {
		id<MTLCommandBuffer> cb = run();
		const double g = gpuMs(cb), w = gWall * 1e3;
		[gs addObject:@(samples / (g * 1e6))];
		[gsw addObject:@(samples / (w * 1e6))];
		[gpu addObject:@(g / passes)];
	}
	return [NSString stringWithFormat:@"gsamples_s=%.1f gsamples_s_wall=%.1f gpu_ms=%.3f %@", median(gs), median(gsw),
	                                  median(gpu), spread(gs, "%.1f")];
}

static NSString *drawCalls(int draws)
{
	id<MTLRenderPipelineState> p = renderPipeline(@"v_small", @"f_flat", MTLPixelFormatRGBA8Unorm, NO);
	id<MTLTexture> t = target(512, 512);
	const int cbsPerRep = 3, threads = gThreads;
	// A command buffer encoded, and the CPU time its encode took added to `encodeS`.
	id<MTLCommandBuffer> (^encode)(double *) = ^(double *encodeS) {
		const double s = now();
		id<MTLCommandBuffer> cb = [gQueue commandBuffer];
		id<MTLRenderCommandEncoder> e = [cb renderCommandEncoderWithDescriptor:pass(t, MTLLoadActionClear)];
		[e setRenderPipelineState:p];
		const float c[4] = {0.2f, 0.6f, 0.9f, 1};
		[e setFragmentBytes:c length:sizeof c atIndex:0];
		for (int i = 0; i < draws; i++) {
			const float d[4] = {-1 + (i % 100) * 0.02f, -1 + (i / 100 % 100) * 0.02f, 0.015f, 0.015f};
			[e setVertexBytes:d length:sizeof d atIndex:0];
			[e drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
		}
		[e endEncoding];
		*encodeS += now() - s;
		return cb;
	};
	{
		double unused = 0;
		id<MTLCommandBuffer> cb = encode(&unused);
		[cb commit];
		finish(cb);
	}
	NSMutableArray *ds = [NSMutableArray array], *enc = [NSMutableArray array], *gpu = [NSMutableArray array],
	               *wall = [NSMutableArray array];
	double *encodeS = calloc((size_t)threads, sizeof(double));
	for (int r = 0; more(r); r++) {
		NSMutableArray *cbs = [NSMutableArray array];
		NSLock *lock = [NSLock new];
		const double s = now();
		// Each thread encodes and commits its buffers back to back, onto the one queue.
		dispatch_apply((size_t)threads, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t j) {
			encodeS[j] = 0;
			for (int i = 0; i < cbsPerRep; i++) {
				id<MTLCommandBuffer> cb = encode(&encodeS[j]);
				[cb commit];
				[lock lock];
				[cbs addObject:cb];
				[lock unlock];
			}
		});
		finishAll(cbs);
		const double el = now() - s;
		double encoded = 0;
		for (int j = 0; j < threads; j++)
			encoded += encodeS[j];
		const int count = cbsPerRep * threads;
		[ds addObject:@((double)draws * count / el)];
		[enc addObject:@(encoded * 1e3 / count)];
		[gpu addObject:@(gpuSpanMs(cbs) / count)];
		[wall addObject:@(el * 1e3 / count)];
	}
	free(encodeS);
	return [NSString stringWithFormat:@"draws_s=%.0f encode_ms=%.2f gpu_ms=%.2f wall_ms=%.2f draws_cb=%d threads=%d %@",
	                                  median(ds), median(enc), median(gpu), median(wall), draws, threads,
	                                  spread(ds, "%.0f")];
}

static NSString *upload(NSUInteger bytes)
{
	id<MTLBuffer> up = [gDevice newBufferWithLength:bytes options:MTLResourceStorageModeShared];
	id<MTLBuffer> dst = [gDevice newBufferWithLength:bytes options:MTLResourceStorageModePrivate];
	uint32_t *src = malloc(bytes);
	for (NSUInteger i = 0; i < bytes / 4; i++)
		src[i] = (uint32_t)(i * 2654435761u);
	NSMutableArray *e2e = [NSMutableArray array], *cpu = [NSMutableArray array], *gpu = [NSMutableArray array];
	for (int r = -1; more(r); r++) {
		src[0] = (uint32_t)r;
		const double s = now();
		memcpy(up.contents, src, bytes);
		const double c = now();
		id<MTLCommandBuffer> cb = [gQueue commandBuffer];
		id<MTLBlitCommandEncoder> e = [cb blitCommandEncoder];
		[e copyFromBuffer:up sourceOffset:0 toBuffer:dst destinationOffset:0 size:bytes];
		[e endEncoding];
		[cb commit];
		finish(cb);
		const double done = now();
		if (r < 0)
			continue;
		[e2e addObject:@(bytes / ((done - s) * 1e6))];
		[cpu addObject:@(bytes / ((c - s) * 1e6))];
		[gpu addObject:@(gpuMs(cb))];
	}
	free(src);
	return [NSString stringWithFormat:@"mbps=%.0f cpu_write_mbps=%.0f gpu_ms=%.2f %@", median(e2e), median(cpu),
	                                  median(gpu), spread(e2e, "%.0f")];
}

static NSString *readback(NSUInteger bytes)
{
	id<MTLComputePipelineState> p = computePipeline(@"k_pattern");
	id<MTLBuffer> rb = [gDevice newBufferWithLength:bytes options:MTLResourceStorageModeShared];
	uint32_t *dst = malloc(bytes);
	memset(dst, 0, bytes);
	const NSUInteger words = bytes / 4, tg = 256;
	NSMutableArray *e2e = [NSMutableArray array], *cpu = [NSMutableArray array], *gpu = [NSMutableArray array];
	for (int r = -1; more(r); r++) {
		const uint32_t seed = 0x9e3779b9u * (uint32_t)(r + 2);
		const double s = now();
		id<MTLCommandBuffer> cb = [gQueue commandBuffer];
		id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
		[e setComputePipelineState:p];
		[e setBuffer:rb offset:0 atIndex:0];
		[e setBytes:&seed length:sizeof seed atIndex:1];
		[e dispatchThreadgroups:MTLSizeMake(words / tg, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
		[e endEncoding];
		[cb commit];
		finish(cb);
		const double g = now();
		memcpy(dst, rb.contents, bytes);
		const double done = now();
		for (NSUInteger i = 0; i < words; i += 4093)
			if (dst[i] != ((uint32_t)i ^ seed))
				return [NSString stringWithFormat:@"FAIL word %lu reads 0x%08x, the GPU wrote 0x%08x", (unsigned long)i,
				                                  dst[i], (uint32_t)i ^ seed];
		if (r < 0)
			continue;
		[e2e addObject:@(bytes / ((done - s) * 1e6))];
		[cpu addObject:@(bytes / ((done - g) * 1e6))];
		[gpu addObject:@(gpuMs(cb))];
	}
	free(dst);
	return [NSString stringWithFormat:@"mbps=%.0f cpu_read_mbps=%.0f gpu_ms=%.2f %@", median(e2e), median(cpu),
	                                  median(gpu), spread(e2e, "%.0f")];
}

static NSString *sgemm(NSUInteger n)
{
	if (!MPSSupportsMTLDevice(gDevice))
		return @"SKIP MPS does not support the device";
	const NSUInteger rowBytes = [MPSMatrixDescriptor rowBytesFromColumns:n dataType:MPSDataTypeFloat32];
	MPSMatrixDescriptor *md = [MPSMatrixDescriptor matrixDescriptorWithRows:n
	                                                                columns:n
	                                                               rowBytes:rowBytes
	                                                               dataType:MPSDataTypeFloat32];
	id<MTLBuffer> ba = [gDevice newBufferWithLength:n * rowBytes options:MTLResourceStorageModePrivate];
	id<MTLBuffer> bb = [gDevice newBufferWithLength:n * rowBytes options:MTLResourceStorageModePrivate];
	id<MTLBuffer> bc = [gDevice newBufferWithLength:n * rowBytes options:MTLResourceStorageModePrivate];
	{
		// 0x3c3c3c3c is about 0.0115 as a float: normal, and the sums stay small.
		id<MTLCommandBuffer> cb = [gQueue commandBuffer];
		id<MTLBlitCommandEncoder> e = [cb blitCommandEncoder];
		[e fillBuffer:ba range:NSMakeRange(0, n * rowBytes) value:0x3c];
		[e fillBuffer:bb range:NSMakeRange(0, n * rowBytes) value:0x3c];
		[e endEncoding];
		[cb commit];
		finish(cb);
	}
	MPSMatrix *a = [[MPSMatrix alloc] initWithBuffer:ba descriptor:md];
	MPSMatrix *b = [[MPSMatrix alloc] initWithBuffer:bb descriptor:md];
	MPSMatrix *c = [[MPSMatrix alloc] initWithBuffer:bc descriptor:md];
	MPSMatrixMultiplication *mm = [[MPSMatrixMultiplication alloc] initWithDevice:gDevice
	                                                                transposeLeft:NO
	                                                               transposeRight:NO
	                                                                   resultRows:n
	                                                                resultColumns:n
	                                                              interiorColumns:n
	                                                                        alpha:1
	                                                                         beta:0];
	const int per = n <= 2048 ? 4 : 1;
	id<MTLCommandBuffer> (^run)(void) = ^{
		id<MTLCommandBuffer> cb = [gQueue commandBuffer];
		for (int i = 0; i < per; i++)
			[mm encodeToCommandBuffer:cb leftMatrix:a rightMatrix:b resultMatrix:c];
		const double s = now();
		[cb commit];
		finish(cb);
		gWall = now() - s;
		return cb;
	};
	run();
	NSMutableArray *gf = [NSMutableArray array], *gfw = [NSMutableArray array], *gpu = [NSMutableArray array];
	const double flops = 2.0 * n * n * n * per;
	for (int r = 0; more(r); r++) {
		id<MTLCommandBuffer> cb = run();
		const double g = gpuMs(cb), w = gWall * 1e3;
		[gf addObject:@(flops / (g * 1e6))];
		[gfw addObject:@(flops / (w * 1e6))];
		[gpu addObject:@(g / per)];
	}
	return [NSString stringWithFormat:@"gflops=%.0f gflops_wall=%.0f gpu_ms=%.2f per_cb=%d %@", median(gf), median(gfw),
	                                  median(gpu), per, spread(gf, "%.0f")];
}

// --- onscreen -----------------------------------------------------------------------------------

// Animates the heavy shader in a full-screen borderless window for gSeconds after a second of
// warm-up, on a thread of its own; the main thread runs AppKit. The window never takes key focus.
static NSString *onscreen(void)
{
	[NSApplication sharedApplication];
	[NSApp setActivationPolicy:NSApplicationActivationPolicyAccessory];
	NSScreen *screen = NSScreen.mainScreen;
	if (screen == nil)
		return @"SKIP no screen (run it in the logged-in user's session)";
	const NSRect frame = screen.frame;
	const CGFloat scale = screen.backingScaleFactor;
	NSWindow *win = [[NSWindow alloc] initWithContentRect:frame
	                                            styleMask:NSWindowStyleMaskBorderless
	                                              backing:NSBackingStoreBuffered
	                                                defer:NO
	                                               screen:screen];
	win.level = NSStatusWindowLevel;
	win.opaque = YES;
	CAMetalLayer *layer = [CAMetalLayer layer];
	layer.device = gDevice;
	layer.pixelFormat = MTLPixelFormatBGRA8Unorm;
	layer.framebufferOnly = YES;
	layer.opaque = YES;
	layer.contentsScale = scale;
	layer.drawableSize = CGSizeMake(frame.size.width * scale, frame.size.height * scale);
	NSView *view = [[NSView alloc] initWithFrame:NSMakeRect(0, 0, frame.size.width, frame.size.height)];
	view.layer = layer;
	view.wantsLayer = YES;
	win.contentView = view;
	[win orderFrontRegardless];

	id<MTLRenderPipelineState> p = renderPipeline(@"v_full", @"f_heavy", MTLPixelFormatBGRA8Unorm, NO);
	const NSInteger maxFps = screen.maximumFramesPerSecond;
	__block NSString *result = nil;
	NSThread *render = [[NSThread alloc] initWithBlock:^{
		NSMutableArray *presented = [NSMutableArray array], *gpu = [NSMutableArray array];
		NSLock *lock = [NSLock new];
		__block atomic_int dropped = 0, completed = 0;
		int submitted = 0;
		const double start = now(), measure = start + 1, end = measure + gSeconds;
		id<MTLCommandBuffer> last = nil;
		while (now() < end) {
			@autoreleasepool {
				id<CAMetalDrawable> d = [layer nextDrawable];
				if (d == nil)
					continue;
				const BOOL counted = now() >= measure;
				id<MTLCommandBuffer> cb = [gQueue commandBuffer];
				encodeHeavy(cb, p, d.texture, (float)(now() - start));
				if (counted) {
					submitted++;
					[d addPresentedHandler:^(id<MTLDrawable> dr) {
						if (dr.presentedTime > 0) {
							[lock lock];
							[presented addObject:@(dr.presentedTime)];
							[lock unlock];
						} else {
							atomic_fetch_add(&dropped, 1);
						}
					}];
					[cb addCompletedHandler:^(id<MTLCommandBuffer> c) {
						atomic_fetch_add(&completed, 1);
						[lock lock];
						[gpu addObject:@(gpuMs(c))];
						[lock unlock];
					}];
				}
				[cb presentDrawable:d];
				[cb commit];
				last = cb;
			}
		}
		[last waitUntilCompleted];
		[NSThread sleepForTimeInterval:0.3];
		[lock lock];
		NSArray *ts = [presented sortedArrayUsingSelector:@selector(compare:)];
		const double span = ts.count > 1 ? [ts.lastObject doubleValue] - [ts.firstObject doubleValue] : 0;
		const double fps = span > 0 ? (ts.count - 1) / span : 0;
		NSMutableArray *gaps = [NSMutableArray array];
		for (NSUInteger i = 1; i < ts.count; i++)
			[gaps addObject:@(([ts[i] doubleValue] - [ts[i - 1] doubleValue]) * 1e3)];
		result = [NSString
		    stringWithFormat:@"presented_fps=%.1f submitted_fps=%.1f presented=%lu dropped=%d completed=%d "
		                     @"gpu_ms=%.3f frame_ms_p50=%.2f frame_ms_max=%.2f drawable=%.0fx%.0f screen_max_fps=%ld "
		                     @"iters=%u seconds=%.0f",
		                     fps, submitted / gSeconds, (unsigned long)ts.count, atomic_load(&dropped),
		                     atomic_load(&completed), median(gpu), median(gaps), gaps.count ? maxOf(gaps) : 0,
		                     layer.drawableSize.width, layer.drawableSize.height, (long)maxFps, gIters, gSeconds];
		[lock unlock];
		dispatch_async(dispatch_get_main_queue(), ^{
			[NSApp stop:nil];
			[NSApp postEvent:[NSEvent otherEventWithType:NSEventTypeApplicationDefined
			                                    location:NSZeroPoint
			                               modifierFlags:0
			                                   timestamp:0
			                                windowNumber:0
			                                     context:nil
			                                     subtype:0
			                                       data1:0
			                                       data2:0]
			         atStart:YES];
		});
	}];
	[render start];
	[NSApp run];
	[win orderOut:nil];
	return result ?: @"FAIL the render thread gave no result";
}

// --- main ---------------------------------------------------------------------------------------

int main(int argc, char **argv)
{
	@autoreleasepool {
		NSMutableArray *patterns = [NSMutableArray array];
		const char *want = NULL;
		BOOL list = NO;
		for (int i = 1; i < argc; i++) {
			if (strcmp(argv[i], "-d") == 0 && i + 1 < argc)
				want = argv[++i];
			else if (strcmp(argv[i], "-r") == 0 && i + 1 < argc)
				gReps = atoi(argv[++i]);
			else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc)
				gIters = (uint32_t)atoi(argv[++i]);
			else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc)
				gSeconds = atof(argv[++i]);
			else if (strcmp(argv[i], "-m") == 0 && i + 1 < argc)
				gMinSeconds = atof(argv[++i]);
			else if (strcmp(argv[i], "-j") == 0 && i + 1 < argc)
				gThreads = atoi(argv[++i]);
			else if (strcmp(argv[i], "-l") == 0)
				list = YES;
			else if (strcmp(argv[i], "-v") == 0)
				gVerbose = YES;
			else
				[patterns addObject:@(argv[i])];
		}
		gPatterns = patterns;
		if (gReps < 1)
			gReps = 1;
		if (gThreads < 1)
			gThreads = 1;
		if (list) {
			for (id<MTLDevice> d in MTLCopyAllDevices())
				printf("bench: device \"%s\" lowPower=%d removable=%d unified=%d\n", d.name.UTF8String,
				       d.lowPower, d.removable, d.hasUnifiedMemory);
			return 0;
		}
		if (want != NULL) {
			for (id<MTLDevice> d in MTLCopyAllDevices())
				if (gDevice == nil && strcasestr(d.name.UTF8String, want))
					gDevice = d;
		} else {
			gDevice = MTLCreateSystemDefaultDevice();
		}
		if (gDevice == nil) {
			printf("bench: no Metal device%s%s\n", want ? " matching " : "", want ? want : "");
			return 2;
		}
		char model[64] = "?", build[32] = "?";
		int translated = 0;
		size_t len = sizeof(model);
		sysctlbyname("hw.model", model, &len, NULL, 0);
		len = sizeof(build);
		sysctlbyname("kern.osversion", build, &len, NULL, 0);
		len = sizeof(translated);
		sysctlbyname("sysctl.proc_translated", &translated, &len, NULL, 0);
		NSOperatingSystemVersion v = NSProcessInfo.processInfo.operatingSystemVersion;
		printf("bench: device \"%s\", %s, macOS %ld.%ld.%ld (%s), %s%s, reps %d, min %.0f s\n", gDevice.name.UTF8String, model,
		       (long)v.majorVersion, (long)v.minorVersion, (long)v.patchVersion, build,
#if defined(__x86_64__)
		       "x86_64",
#else
		       "arm64",
#endif
		       translated ? " (Rosetta)" : "", gReps, gMinSeconds);
		fflush(stdout);
		gQueue = [gDevice newCommandQueue];
		NSError *err = nil;
		const double c0 = now();
		gLib = [gDevice newLibraryWithSource:kShaders options:nil error:&err];
		if (gLib == nil) {
			printf("bench: shaders FAIL %s\n", err.localizedDescription.UTF8String);
			return 1;
		}
		printf("bench: shaders compiled in %.0f ms\n", (now() - c0) * 1e3);
		fflush(stdout);

		runCase(@"frag.alu.1080p", YES, ^{ return fragAlu(1920, 1080); });
		runCase(@"frag.alu.2160p", YES, ^{ return fragAlu(3840, 2160); });
		runCase(@"frag.fill.2160p", YES, ^{ return fragFill(3840, 2160, 16); });
		runCase(@"compute.fma", YES, ^{ return computeFma(); });
		runCase(@"blit.256m", YES, ^{ return blitCopy(256u << 20); });
		runCase(@"sample.tex", YES, ^{ return sampleTex(); });
		runCase(@"draws.10k", YES, ^{ return drawCalls(10000); });
		runCase(@"upload.64m", YES, ^{ return upload(64u << 20); });
		runCase(@"readback.64m", YES, ^{ return readback(64u << 20); });
		runCase(@"mps.sgemm.2048", YES, ^{ return sgemm(2048); });
		runCase(@"mps.sgemm.4096", YES, ^{ return sgemm(4096); });
		runCase(@"onscreen", NO, ^{ return onscreen(); });
	}
	return 0;
}
