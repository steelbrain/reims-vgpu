// encode_probe: what the hot encoder calls cost the calling thread on a Metal device, in
// nanoseconds a call, to compare a guest's Metal bundle with real Macs' Metal. No GPU work is
// timed: each repetition times a run of one call in one encoder, then ends the encoder, commits
// its command buffer and waits for it, untimed but for the end and the commit (end_us,
// commit_us). The draws draw a triangle whose three vertices are the origin, into a 64 x 64
// target, so the GPU rasterises nothing.
//   encode_probe [-d NAME] [-n CALLS] [-r REPS] [-m SECONDS] [-l] [CASE...]
//     -d NAME     the first of MTLCopyAllDevices whose name holds NAME; the default is
//                 MTLCreateSystemDefaultDevice (on a dual-GPU MacBook pass -d, which keeps the
//                 display where it is).
//     -n CALLS    the calls a repetition times (default 10000); the 4 KiB setVertexBytes case
//                 takes a fifth of them, the encoder cases a tenth and cb.commit a twentieth.
//     -r REPS     repetitions a case, after two untimed ones (default 9).
//     -m SECONDS  more repetitions, until a case has run this long (default 0): long enough for
//                 a sampler (sample, spindump) to see it.
//     -l          lists the devices and exits.
//     CASE        runs only the cases whose name holds one of these; with none, every case.
// The cases, each in a command buffer from [queue commandBuffer] (retained references):
//   render.setVertexBuffer       setVertexBuffer:offset:atIndex: of 8 Shared buffers in turn, at
//                                64 offsets 256 bytes apart, index 0.
//   render.setVertexBytes.16     setVertexBytes:length:atIndex: of 16 bytes, index 1.
//   render.setVertexBytes.4k     the same of 4096 bytes.
//   render.setFragmentTexture    setFragmentTexture:atIndex: of 8 Private 64 x 64 RGBA8 textures
//                                in turn, index 0.
//   render.setFragmentSamplerState  setFragmentSamplerState:atIndex: of 4 samplers in turn.
//   render.setRenderPipelineState   setRenderPipelineState: of 2 pipelines in turn (the second
//                                blends).
//   render.drawPrimitives        drawPrimitives:vertexStart:vertexCount: of 3 vertices, all state
//                                set once before.
//   render.drawIndexedPrimitives drawIndexedPrimitives:indexCount:indexType:indexBuffer:
//                                indexBufferOffset: of 3 16-bit indices.
//   render.draw.mix              a draw as a renderer sets it: setVertexBuffer (8 buffers in
//                                turn), setVertexBytes of 64 bytes, setFragmentTexture (8 in
//                                turn), setFragmentSamplerState (4 in turn) and the indexed draw;
//                                ns_call is a draw's five calls.
//   compute.setBuffer            setBuffer:offset:atIndex: as render.setVertexBuffer.
//   compute.setBytes.16          setBytes:length:atIndex: of 16 bytes, index 1.
//   compute.dispatchThreadgroups dispatchThreadgroups:threadsPerThreadgroup: of one group of one
//                                thread, all state set once before.
//   encoder.compute              computeCommandEncoder then endEncoding, CALLS/10 of them in one
//                                command buffer: ns_call is the pair, make_ns and end_ns each.
//   encoder.render               renderCommandEncoderWithDescriptor: (load and store the target)
//                                then endEncoding, as encoder.compute.
//   cb.commit                    CALLS/20 command buffers back to back, each commandBuffer, a
//                                compute encoder with one dispatch, endEncoding and commit,
//                                waiting only for the last: ns_call is a buffer's calls, cb_ns,
//                                encode_ns (encoder, pipeline, dispatch, end) and commit_ns each.
// One line a case: "encode_probe: NAME OK ns_call=P50 min=.. max=.. cpu_ns_call=P50 ...", from
// the repetitions' wall time (and the thread's CPU time: wall above CPU is time off the CPU),
// after a "# start NAME" line; "FAIL why" where a command buffer fails.
// Build: make encode-probe (build/encode_probe-x86_64, a guest's or an Intel Mac's); ARCH=arm64
// for an Apple-silicon Mac.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <mach/mach_time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/sysctl.h>
#include <time.h>
#include <unistd.h>

static id<MTLDevice> gDevice;
static id<MTLCommandQueue> gQueue;
static NSArray<NSString *> *gPatterns;
static int gCalls = 10000;
static int gReps = 9;
static double gMinSeconds;
static double gNsPerTick;

static NSString *const kSource =
    @"#include <metal_stdlib>\n"
     "using namespace metal;\n"
     "struct VOut { float4 pos [[position]]; float2 uv; };\n"
     "vertex VOut vs(uint vid [[vertex_id]], device const float4 *v [[buffer(0)]],\n"
     "               constant float4 *u [[buffer(1)]]) {\n"
     "  VOut o; o.pos = v[vid] + u[0]; o.uv = o.pos.xy; return o;\n"
     "}\n"
     "fragment float4 fs(VOut in [[stage_in]], texture2d<float> t [[texture(0)]],\n"
     "                   sampler s [[sampler(0)]]) {\n"
     "  return t.sample(s, in.uv);\n"
     "}\n"
     "kernel void k(device uint *o [[buffer(0)]], constant uint4 *c [[buffer(1)]],\n"
     "              uint tid [[thread_position_in_grid]]) {\n"
     "  if (tid == 0) o[0] = c[0].x;\n"
     "}\n";

// What the cases bind. The strong arrays keep them; the cases read the unretained copies, so no
// retain or release of an argument is timed.
static NSArray *gKeep;
static __unsafe_unretained id<MTLBuffer> gBuffers[8];
static __unsafe_unretained id<MTLTexture> gTextures[8];
static __unsafe_unretained id<MTLSamplerState> gSamplers[4];
static __unsafe_unretained id<MTLRenderPipelineState> gPipelines[2];
static id<MTLComputePipelineState> gKernel;
static id<MTLBuffer> gIndices;
static id<MTLTexture> gTarget;
static uint8_t gBytes[4096];

static uint64_t ticks(void)
{
	return mach_absolute_time();
}

static double ns(uint64_t t)
{
	return (double)t * gNsPerTick;
}

static uint64_t threadCpuNs(void)
{
	return clock_gettime_nsec_np(CLOCK_THREAD_CPUTIME_ID);
}

static BOOL wanted(const char *name)
{
	if (gPatterns.count == 0)
		return YES;
	for (NSString *p in gPatterns)
		if (strstr(name, p.UTF8String) != NULL)
			return YES;
	return NO;
}

static int cmpDouble(const void *a, const void *b)
{
	double x = *(const double *)a, y = *(const double *)b;
	return x < y ? -1 : x > y;
}

typedef struct {
	double *v;
	size_t n, cap;
} Samples;

static void sAdd(Samples *s, double value)
{
	if (s->n == s->cap) {
		s->cap = s->cap ? s->cap * 2 : 64;
		s->v = realloc(s->v, s->cap * sizeof(double));
	}
	s->v[s->n++] = value;
}

static double sMedian(Samples *s)
{
	if (s->n == 0)
		return 0;
	qsort(s->v, s->n, sizeof(double), cmpDouble);
	return s->v[s->n / 2];
}

static void sFree(Samples *s)
{
	free(s->v);
	memset(s, 0, sizeof(*s));
}

static MTLRenderPassDescriptor *passDescriptor(MTLLoadAction load)
{
	MTLRenderPassDescriptor *d = [MTLRenderPassDescriptor renderPassDescriptor];
	d.colorAttachments[0].texture = gTarget;
	d.colorAttachments[0].loadAction = load;
	d.colorAttachments[0].storeAction = MTLStoreActionStore;
	d.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 1);
	return d;
}

// The state every draw reads, set once at a render encoder's start.
static void renderPrelude(id<MTLRenderCommandEncoder> e)
{
	[e setRenderPipelineState:gPipelines[0]];
	[e setVertexBuffer:gBuffers[0] offset:0 atIndex:0];
	[e setVertexBytes:gBytes length:16 atIndex:1];
	[e setFragmentTexture:gTextures[0] atIndex:0];
	[e setFragmentSamplerState:gSamplers[0] atIndex:0];
}

static void computePrelude(id<MTLComputeCommandEncoder> e)
{
	[e setComputePipelineState:gKernel];
	[e setBuffer:gBuffers[0] offset:0 atIndex:0];
	[e setBytes:gBytes length:16 atIndex:1];
}

// One repetition's numbers: the timed calls' wall and CPU time, and the parts some cases time.
typedef struct {
	double wallNs, cpuNs;
	double partNs[3];
	double endNs, commitNs;
	BOOL failed;
} Rep;

typedef enum {
	kRenderSetVertexBuffer,
	kRenderSetVertexBytes16,
	kRenderSetVertexBytes4k,
	kRenderSetFragmentTexture,
	kRenderSetFragmentSampler,
	kRenderSetPipeline,
	kRenderDraw,
	kRenderDrawIndexed,
	kRenderDrawMix,
	kComputeSetBuffer,
	kComputeSetBytes16,
	kComputeDispatch,
	kEncoderCompute,
	kEncoderRender,
	kCbCommit,
} Case;

static int callsOf(Case c)
{
	switch (c) {
	case kRenderSetVertexBytes4k:
		return gCalls / 5;
	case kEncoderCompute:
	case kEncoderRender:
		return gCalls / 10;
	case kCbCommit:
		return gCalls / 20;
	default:
		return gCalls;
	}
}

static BOOL finish(id<MTLCommandBuffer> cb)
{
	[cb waitUntilCompleted];
	return cb.status == MTLCommandBufferStatusCompleted;
}

static Rep renderRep(Case c, int n)
{
	Rep r = {0};
	@autoreleasepool {
		id<MTLCommandBuffer> cb = [gQueue commandBuffer];
		id<MTLRenderCommandEncoder> e =
		    [cb renderCommandEncoderWithDescriptor:passDescriptor(MTLLoadActionClear)];
		renderPrelude(e);
		id<MTLBuffer> indices = gIndices;
		uint64_t c0 = threadCpuNs(), t0 = ticks();
		switch (c) {
		case kRenderSetVertexBuffer:
			for (int i = 0; i < n; i++)
				[e setVertexBuffer:gBuffers[i & 7] offset:(NSUInteger)(i & 63) * 256 atIndex:0];
			break;
		case kRenderSetVertexBytes16:
			for (int i = 0; i < n; i++)
				[e setVertexBytes:gBytes length:16 atIndex:1];
			break;
		case kRenderSetVertexBytes4k:
			for (int i = 0; i < n; i++)
				[e setVertexBytes:gBytes length:4096 atIndex:1];
			break;
		case kRenderSetFragmentTexture:
			for (int i = 0; i < n; i++)
				[e setFragmentTexture:gTextures[i & 7] atIndex:0];
			break;
		case kRenderSetFragmentSampler:
			for (int i = 0; i < n; i++)
				[e setFragmentSamplerState:gSamplers[i & 3] atIndex:0];
			break;
		case kRenderSetPipeline:
			for (int i = 0; i < n; i++)
				[e setRenderPipelineState:gPipelines[i & 1]];
			break;
		case kRenderDraw:
			for (int i = 0; i < n; i++)
				[e drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
			break;
		case kRenderDrawIndexed:
			for (int i = 0; i < n; i++)
				[e drawIndexedPrimitives:MTLPrimitiveTypeTriangle
				              indexCount:3
				               indexType:MTLIndexTypeUInt16
				             indexBuffer:indices
				       indexBufferOffset:0];
			break;
		case kRenderDrawMix:
			for (int i = 0; i < n; i++) {
				[e setVertexBuffer:gBuffers[i & 7] offset:0 atIndex:0];
				[e setVertexBytes:gBytes length:64 atIndex:1];
				[e setFragmentTexture:gTextures[i & 7] atIndex:0];
				[e setFragmentSamplerState:gSamplers[i & 3] atIndex:0];
				[e drawIndexedPrimitives:MTLPrimitiveTypeTriangle
				              indexCount:3
				               indexType:MTLIndexTypeUInt16
				             indexBuffer:indices
				       indexBufferOffset:0];
			}
			break;
		default:
			break;
		}
		uint64_t t1 = ticks(), c1 = threadCpuNs();
		[e endEncoding];
		uint64_t t2 = ticks();
		[cb commit];
		uint64_t t3 = ticks();
		r.wallNs = ns(t1 - t0);
		r.cpuNs = (double)(c1 - c0);
		r.endNs = ns(t2 - t1);
		r.commitNs = ns(t3 - t2);
		r.failed = !finish(cb);
	}
	return r;
}

static Rep computeRep(Case c, int n)
{
	Rep r = {0};
	@autoreleasepool {
		id<MTLCommandBuffer> cb = [gQueue commandBuffer];
		id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
		computePrelude(e);
		uint64_t c0 = threadCpuNs(), t0 = ticks();
		switch (c) {
		case kComputeSetBuffer:
			for (int i = 0; i < n; i++)
				[e setBuffer:gBuffers[i & 7] offset:(NSUInteger)(i & 63) * 256 atIndex:0];
			break;
		case kComputeSetBytes16:
			for (int i = 0; i < n; i++)
				[e setBytes:gBytes length:16 atIndex:1];
			break;
		case kComputeDispatch:
			for (int i = 0; i < n; i++)
				[e dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
			break;
		default:
			break;
		}
		uint64_t t1 = ticks(), c1 = threadCpuNs();
		[e endEncoding];
		uint64_t t2 = ticks();
		[cb commit];
		uint64_t t3 = ticks();
		r.wallNs = ns(t1 - t0);
		r.cpuNs = (double)(c1 - c0);
		r.endNs = ns(t2 - t1);
		r.commitNs = ns(t3 - t2);
		r.failed = !finish(cb);
	}
	return r;
}

// encoder.compute and encoder.render: partNs[0] the makes, partNs[1] the ends.
static Rep encoderRep(Case c, int n)
{
	Rep r = {0};
	@autoreleasepool {
		id<MTLCommandBuffer> cb = [gQueue commandBuffer];
		MTLRenderPassDescriptor *pass = passDescriptor(MTLLoadActionLoad);
		uint64_t make = 0, end = 0;
		uint64_t c0 = threadCpuNs(), t0 = ticks();
		for (int i = 0; i < n; i++) {
			uint64_t a = ticks();
			id<MTLCommandEncoder> e = c == kEncoderCompute
			                              ? (id<MTLCommandEncoder>)[cb computeCommandEncoder]
			                              : (id<MTLCommandEncoder>)[cb renderCommandEncoderWithDescriptor:pass];
			uint64_t b = ticks();
			[e endEncoding];
			uint64_t d = ticks();
			make += b - a;
			end += d - b;
		}
		uint64_t t1 = ticks(), c1 = threadCpuNs();
		[cb commit];
		uint64_t t2 = ticks();
		r.wallNs = ns(t1 - t0);
		r.cpuNs = (double)(c1 - c0);
		r.partNs[0] = ns(make);
		r.partNs[1] = ns(end);
		r.commitNs = ns(t2 - t1);
		r.failed = !finish(cb);
	}
	return r;
}

// cb.commit: partNs[0] the commandBuffer calls, [1] the encoding, [2] the commits.
static Rep commitRep(int n)
{
	Rep r = {0};
	@autoreleasepool {
		uint64_t make = 0, encode = 0, commit = 0;
		id<MTLCommandBuffer> last = nil;
		uint64_t c0 = threadCpuNs(), t0 = ticks();
		for (int i = 0; i < n; i++) {
			uint64_t a = ticks();
			id<MTLCommandBuffer> cb = [gQueue commandBuffer];
			uint64_t b = ticks();
			id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
			[e setComputePipelineState:gKernel];
			[e setBuffer:gBuffers[i & 7] offset:0 atIndex:0];
			[e dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
			[e endEncoding];
			uint64_t d = ticks();
			[cb commit];
			uint64_t f = ticks();
			make += b - a;
			encode += d - b;
			commit += f - d;
			last = cb;
		}
		uint64_t t1 = ticks(), c1 = threadCpuNs();
		r.wallNs = ns(t1 - t0);
		r.cpuNs = (double)(c1 - c0);
		r.partNs[0] = ns(make);
		r.partNs[1] = ns(encode);
		r.partNs[2] = ns(commit);
		r.failed = !finish(last);
	}
	return r;
}

static Rep runRep(Case c, int n)
{
	switch (c) {
	case kComputeSetBuffer:
	case kComputeSetBytes16:
	case kComputeDispatch:
		return computeRep(c, n);
	case kEncoderCompute:
	case kEncoderRender:
		return encoderRep(c, n);
	case kCbCommit:
		return commitRep(n);
	default:
		return renderRep(c, n);
	}
}

static void runCase(const char *name, Case c)
{
	if (!wanted(name))
		return;
	printf("# start %s\n", name);
	const int n = callsOf(c);
	for (int i = 0; i < 2; i++) {
		if (runRep(c, n).failed) {
			printf("encode_probe: %s FAIL a command buffer did not complete\n", name);
			return;
		}
	}
	Samples wall = {0}, cpu = {0}, part[3] = {{0}}, end = {0}, commit = {0};
	double started = ns(ticks());
	int reps = 0;
	while (reps < gReps || (ns(ticks()) - started) / 1e9 < gMinSeconds) {
		Rep r = runRep(c, n);
		if (r.failed) {
			printf("encode_probe: %s FAIL a command buffer did not complete\n", name);
			return;
		}
		sAdd(&wall, r.wallNs / n);
		sAdd(&cpu, r.cpuNs / n);
		for (int k = 0; k < 3; k++)
			sAdd(&part[k], r.partNs[k] / n);
		sAdd(&end, r.endNs / 1e3);
		sAdd(&commit, r.commitNs / 1e3);
		reps++;
	}
	double med = sMedian(&wall); // sorts: min and max are the ends
	NSMutableString *line = [NSMutableString
	    stringWithFormat:@"encode_probe: %s OK ns_call=%.1f min=%.1f max=%.1f cpu_ns_call=%.1f", name,
	                     med, wall.v[0], wall.v[wall.n - 1], sMedian(&cpu)];
	if (c == kEncoderCompute || c == kEncoderRender)
		[line appendFormat:@" make_ns=%.1f end_ns=%.1f", sMedian(&part[0]), sMedian(&part[1])];
	else if (c == kCbCommit)
		[line appendFormat:@" cb_ns=%.1f encode_ns=%.1f commit_ns=%.1f", sMedian(&part[0]),
		                   sMedian(&part[1]), sMedian(&part[2])];
	else
		[line appendFormat:@" end_us=%.1f commit_us=%.1f", sMedian(&end), sMedian(&commit)];
	[line appendFormat:@" calls=%d reps=%d", n, reps];
	printf("%s\n", line.UTF8String);
	sFree(&wall);
	sFree(&cpu);
	for (int k = 0; k < 3; k++)
		sFree(&part[k]);
	sFree(&end);
	sFree(&commit);
}

static BOOL setUp(void)
{
	NSError *err = nil;
	id<MTLLibrary> lib = [gDevice newLibraryWithSource:kSource options:nil error:&err];
	if (lib == nil) {
		printf("encode_probe: FAIL the library: %s\n", err.localizedDescription.UTF8String);
		return NO;
	}
	NSMutableArray *keep = [NSMutableArray array];
	MTLRenderPipelineDescriptor *rp = [MTLRenderPipelineDescriptor new];
	rp.vertexFunction = [lib newFunctionWithName:@"vs"];
	rp.fragmentFunction = [lib newFunctionWithName:@"fs"];
	rp.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;
	for (int i = 0; i < 2; i++) {
		rp.colorAttachments[0].blendingEnabled = i == 1;
		id<MTLRenderPipelineState> p = [gDevice newRenderPipelineStateWithDescriptor:rp error:&err];
		if (p == nil) {
			printf("encode_probe: FAIL a render pipeline: %s\n", err.localizedDescription.UTF8String);
			return NO;
		}
		[keep addObject:p];
		gPipelines[i] = p;
	}
	gKernel = [gDevice newComputePipelineStateWithFunction:[lib newFunctionWithName:@"k"] error:&err];
	if (gKernel == nil) {
		printf("encode_probe: FAIL the compute pipeline: %s\n", err.localizedDescription.UTF8String);
		return NO;
	}
	for (int i = 0; i < 8; i++) {
		// Zeros: every vertex at the origin.
		id<MTLBuffer> b = [gDevice newBufferWithLength:64 * 1024 options:MTLResourceStorageModeShared];
		memset(b.contents, 0, b.length);
		[keep addObject:b];
		gBuffers[i] = b;
	}
	MTLTextureDescriptor *td =
	    [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
	                                                       width:64
	                                                      height:64
	                                                   mipmapped:NO];
	td.storageMode = MTLStorageModePrivate;
	td.usage = MTLTextureUsageShaderRead;
	for (int i = 0; i < 8; i++) {
		id<MTLTexture> t = [gDevice newTextureWithDescriptor:td];
		[keep addObject:t];
		gTextures[i] = t;
	}
	td.usage = MTLTextureUsageRenderTarget;
	gTarget = [gDevice newTextureWithDescriptor:td];
	for (int i = 0; i < 4; i++) {
		MTLSamplerDescriptor *sd = [MTLSamplerDescriptor new];
		sd.minFilter = (i & 1) ? MTLSamplerMinMagFilterLinear : MTLSamplerMinMagFilterNearest;
		sd.magFilter = sd.minFilter;
		sd.sAddressMode = (i & 2) ? MTLSamplerAddressModeRepeat : MTLSamplerAddressModeClampToEdge;
		id<MTLSamplerState> s = [gDevice newSamplerStateWithDescriptor:sd];
		[keep addObject:s];
		gSamplers[i] = s;
	}
	const uint16_t idx[4] = {0, 1, 2, 0};
	gIndices = [gDevice newBufferWithBytes:idx length:sizeof(idx) options:MTLResourceStorageModeShared];
	gKeep = keep;
	// Room for every buffer cb.commit has in flight, so its commandBuffer calls never wait.
	gQueue = [gDevice newCommandQueueWithMaxCommandBufferCount:(NSUInteger)(gCalls / 20 + 64)];
	return gQueue != nil && gTarget != nil && gIndices != nil;
}

int main(int argc, char **argv)
{
	setvbuf(stdout, NULL, _IOLBF, 0);
	@autoreleasepool {
		mach_timebase_info_data_t tb;
		mach_timebase_info(&tb);
		gNsPerTick = (double)tb.numer / tb.denom;
		NSMutableArray *patterns = [NSMutableArray array];
		const char *want = NULL;
		BOOL list = NO;
		for (int i = 1; i < argc; i++) {
			if (strcmp(argv[i], "-d") == 0 && i + 1 < argc)
				want = argv[++i];
			else if (strcmp(argv[i], "-n") == 0 && i + 1 < argc)
				gCalls = atoi(argv[++i]);
			else if (strcmp(argv[i], "-r") == 0 && i + 1 < argc)
				gReps = atoi(argv[++i]);
			else if (strcmp(argv[i], "-m") == 0 && i + 1 < argc)
				gMinSeconds = atof(argv[++i]);
			else if (strcmp(argv[i], "-l") == 0)
				list = YES;
			else
				[patterns addObject:@(argv[i])];
		}
		if (gCalls < 20)
			gCalls = 20;
		if (gReps < 1)
			gReps = 1;
		gPatterns = patterns;
		if (list) {
			for (id<MTLDevice> d in MTLCopyAllDevices())
				printf("encode_probe: device \"%s\" lowPower=%d removable=%d unified=%d\n",
				       d.name.UTF8String, d.lowPower, d.removable, d.hasUnifiedMemory);
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
			printf("encode_probe: no Metal device%s%s\n", want ? " matching " : "", want ? want : "");
			return 2;
		}
		char model[64] = "?", build[32] = "?", cpu[128] = "?";
		int cpus = 0;
		size_t len = sizeof(model);
		sysctlbyname("hw.model", model, &len, NULL, 0);
		len = sizeof(build);
		sysctlbyname("kern.osversion", build, &len, NULL, 0);
		len = sizeof(cpu);
		sysctlbyname("machdep.cpu.brand_string", cpu, &len, NULL, 0);
		len = sizeof(cpus);
		sysctlbyname("hw.logicalcpu", &cpus, &len, NULL, 0);
		NSOperatingSystemVersion v = NSProcessInfo.processInfo.operatingSystemVersion;
		printf("encode_probe: device \"%s\" (%s), %s, %s, %d CPUs, macOS %ld.%ld.%ld (%s), pid %d, "
		       "%d calls, %d reps\n",
		       gDevice.name.UTF8String, NSStringFromClass([gDevice class]).UTF8String, model, cpu, cpus,
		       (long)v.majorVersion, (long)v.minorVersion, (long)v.patchVersion, build, getpid(),
		       gCalls, gReps);
		if (!setUp())
			return 1;
		runCase("render.setVertexBuffer", kRenderSetVertexBuffer);
		runCase("render.setVertexBytes.16", kRenderSetVertexBytes16);
		runCase("render.setVertexBytes.4k", kRenderSetVertexBytes4k);
		runCase("render.setFragmentTexture", kRenderSetFragmentTexture);
		runCase("render.setFragmentSamplerState", kRenderSetFragmentSampler);
		runCase("render.setRenderPipelineState", kRenderSetPipeline);
		runCase("render.drawPrimitives", kRenderDraw);
		runCase("render.drawIndexedPrimitives", kRenderDrawIndexed);
		runCase("render.draw.mix", kRenderDrawMix);
		runCase("compute.setBuffer", kComputeSetBuffer);
		runCase("compute.setBytes.16", kComputeSetBytes16);
		runCase("compute.dispatchThreadgroups", kComputeDispatch);
		runCase("encoder.compute", kEncoderCompute);
		runCase("encoder.render", kEncoderRender);
		runCase("cb.commit", kCbCommit);
	}
	return 0;
}
