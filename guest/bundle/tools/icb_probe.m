// icb_probe: what the CPU's encodes into indirect command buffers cost on a device, a command at a
// time, and what the CPU's writes into an IOSurface a texture lies over cost until the GPU reads
// them. To compare a guest's GPU lane with real Macs, case by case: where the device keeps an
// encode or a surface's bytes in memory the guest maps uncached, its stores show here.
//   icb_probe [-d NAME] [-n COMMANDS] [-r REPS] [-l] [CASE...]
//     -d NAME      the first of MTLCopyAllDevices whose name holds NAME; the default is
//                  MTLCreateSystemDefaultDevice.
//     -n COMMANDS  commands an indirect command buffer has (default 16384).
//     -r REPS      timed repetitions of each case, after one untimed (default 5).
//     -l           lists the devices and exits.
//     CASE         runs only the cases whose name holds one of these; -CASE skips the cases whose
//                  name holds CASE. With none, every case.
// Each repetition encodes every command of the same buffer again, as a renderer encodes its
// buffer each frame; each case prints the median and the least of its repetitions:
//   render      an ICB of draws, each command its render pipeline state, a vertex buffer, a
//               fragment buffer and a draw (4 calls): encode_ns (a command, its lookup
//               indirectRenderCommandAtIndex: included), call_ns (a call), mcmd_s (millions of
//               commands a second), then commit_us, an empty command buffer committed and waited
//               after the encode, against base_us, one with no encode before it; then the buffer
//               executed into an 8x1 target: command i draws pixel i % 8 in colour i % 15, so pixel
//               k holds the last such draw's colour, checked.
//   wide        as render, of an ICB that may bind 31 vertex and 31 fragment buffers: where the
//               device keeps a command's encodes in a slot of its bindings, a buffer of this many
//               commands passes a slot table's cap, and its encodes take the device's other route.
//   inherit     an ICB of draws whose pipeline and buffers are inherited, each command a draw alone
//               (1 call); executed under the encoder's pipeline and buffers, checked.
//   compute     an ICB of dispatches, each command its compute pipeline state, two kernel buffers
//               and a dispatch of one thread (4 calls): command i writes word i of its output the
//               value at (i % 16) of its input; executed, every word checked.
//   reset       resetWithRange: of every command of an encoded ICB of draws: reset_us.
//   surface.WxH a BGRA8 IOSurface of W x H, its rows W * 4 bytes rounded up to 64, and a Managed
//               texture over it; each repetition the CPU writes every byte of it (locked, written, unlocked:
//               write_ms), then a command buffer blits the texture into a Shared buffer, committed
//               and waited (e2e_ms); base_ms, that command buffer with no write before it;
//               mb_s, the surface's bytes over (e2e_ms - base_ms); checked. W 1024 lays rows of
//               4096 bytes, which every host's linear textures take in place; W 1040, rows of 4160,
//               64 times an odd number, which a host whose linear textures align rows to 128 or
//               more does not, so its device copies the surface (RgfxIOSurface.mm).
// One line a case: "icb: NAME OK key=value ...", or "icb: NAME FAIL why" / "SKIP why"; "# start
// NAME" before it with the time since the process started and the Unix time. Build: make
// icb-probe (build/icb_probe-x86_64); make icb-probe ARCH=arm64 for an Apple-silicon Mac.
#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <Metal/Metal.h>

#include <mach/mach_time.h>
#include <sys/time.h>
#include <unistd.h>

#pragma clang diagnostic ignored "-Wunguarded-availability-new"

static id<MTLDevice> gDevice;
static id<MTLCommandQueue> gQueue;
static NSArray<NSString *> *gPatterns;
static NSUInteger gCommands = 16384;
static int gReps = 5;
static double gStart;

// The draws and dispatches the cases encode. A draw covers pixel k of an 8x1 target, k the value
// its vertex buffer holds, in the colour its fragment buffer holds; an inherited draw covers pixel
// [[base_instance]] % 8. A dispatch's one thread copies the word of its second buffer into its
// first.
static NSString *const kSource =
    @"#include <metal_stdlib>\n"
     "using namespace metal;\n"
     "struct V { float4 pos [[position]]; };\n"
     "vertex V icb_vs(uint vid [[vertex_id]], uint inst [[base_instance]],\n"
     "                device const float *cell [[buffer(0)]]) {\n"
     "  float k = cell[0] + float(inst % 8);\n"
     "  float2 c = float2(float(vid & 1), float(vid >> 1));\n"
     "  V o; o.pos = float4(-1.0 + (k + c.x) * 0.25, -1.0 + c.y * 2.0, 0.0, 1.0); return o;\n"
     "}\n"
     "fragment float4 icb_fs(V in [[stage_in]], device const float4 *colour [[buffer(0)]]) {\n"
     "  return colour[0];\n"
     "}\n"
     "kernel void icb_k(device uint *out [[buffer(0)]], device const uint *in [[buffer(1)]],\n"
     "                  uint tid [[thread_position_in_grid]]) {\n"
     "  if (tid == 0) out[0] = in[0];\n"
     "}\n";

// The buffers' records are 256 bytes apart: the largest constant buffer offset alignment of the
// GPUs these run on.
static const NSUInteger kStride = 256;

static id<MTLRenderPipelineState> gDraw;
static id<MTLComputePipelineState> gCopy;
static id<MTLBuffer> gCells, gColours, gInputs, gZeroCell;
static id<MTLTexture> gTarget;

// --- helpers ------------------------------------------------------------------------------------

static double now(void)
{
	static mach_timebase_info_data_t tb;
	if (tb.denom == 0)
		mach_timebase_info(&tb);
	return (double)mach_absolute_time() * tb.numer / tb.denom / 1e9;
}

static double unixNow(void)
{
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return tv.tv_sec + tv.tv_usec / 1e6;
}

static int cmpDouble(const void *a, const void *b)
{
	const double x = *(const double *)a, y = *(const double *)b;
	return x < y ? -1 : x > y;
}

// The median and the least of `n` samples (sorted in place).
static double median(double *v, int n)
{
	qsort(v, (size_t)n, sizeof(double), cmpDouble);
	return v[n / 2];
}

static void finish(id<MTLCommandBuffer> cb)
{
	[cb waitUntilCompleted];
	if (cb.status != MTLCommandBufferStatusCompleted)
		[NSException raise:@"icb" format:@"command buffer status %ld: %@", (long)cb.status,
		                                 cb.error.localizedDescription ?: @"no error"];
}

// The wall time of committing an empty command buffer and waiting for it, in seconds.
static double emptyCommit(void)
{
	id<MTLCommandBuffer> cb = [gQueue commandBuffer];
	const double t0 = now();
	[cb commit];
	finish(cb);
	return now() - t0;
}

static BOOL wanted(NSString *name)
{
	BOOL any = NO, picked = NO;
	for (NSString *p in gPatterns) {
		if ([p hasPrefix:@"-"]) {
			if ([name containsString:[p substringFromIndex:1]])
				return NO;
			continue;
		}
		any = YES;
		picked = picked || [name containsString:p];
	}
	return any ? picked : YES;
}

static void runCase(NSString *name, NSString * (^body)(void))
{
	if (!wanted(name))
		return;
	printf("# start %s t=%.3f unix=%.3f\n", name.UTF8String, now() - gStart, unixNow());
	fflush(stdout);
	NSString *line = nil;
	@try {
		@autoreleasepool {
			line = body();
		}
	} @catch (NSException *e) {
		line = [@"FAIL " stringByAppendingString:e.reason ?: e.name];
	}
	if (![line hasPrefix:@"FAIL"] && ![line hasPrefix:@"SKIP"])
		line = [@"OK " stringByAppendingString:line];
	printf("icb: %s %s\n", name.UTF8String, line.UTF8String);
	fflush(stdout);
}

static id<MTLBuffer> sharedBuffer(NSUInteger length)
{
	id<MTLBuffer> b = [gDevice newBufferWithLength:length options:MTLResourceStorageModeShared];
	if (b == nil)
		[NSException raise:@"icb" format:@"no Shared buffer of %lu bytes", (unsigned long)length];
	memset(b.contents, 0, length);
	return b;
}

static void setup(void)
{
	NSError *err = nil;
	id<MTLLibrary> lib = [gDevice newLibraryWithSource:kSource options:nil error:&err];
	if (lib == nil)
		[NSException raise:@"icb" format:@"no library: %@", err.localizedDescription];
	MTLRenderPipelineDescriptor *rd = [MTLRenderPipelineDescriptor new];
	rd.vertexFunction = [lib newFunctionWithName:@"icb_vs"];
	rd.fragmentFunction = [lib newFunctionWithName:@"icb_fs"];
	rd.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;
	rd.supportIndirectCommandBuffers = YES;
	gDraw = [gDevice newRenderPipelineStateWithDescriptor:rd error:&err];
	if (gDraw == nil)
		[NSException raise:@"icb" format:@"no render pipeline: %@", err.localizedDescription];
	MTLComputePipelineDescriptor *cd = [MTLComputePipelineDescriptor new];
	cd.computeFunction = [lib newFunctionWithName:@"icb_k"];
	cd.supportIndirectCommandBuffers = YES;
	gCopy = [gDevice newComputePipelineStateWithDescriptor:cd
	                                               options:MTLPipelineOptionNone
	                                            reflection:nil
	                                                 error:&err];
	if (gCopy == nil)
		[NSException raise:@"icb" format:@"no compute pipeline: %@", err.localizedDescription];
	// Record k of the cells is pixel k; record c of the colours is colour c, red c * 17 of 255;
	// record k of the inputs is 1000 + k.
	gCells = sharedBuffer(8 * kStride);
	gColours = sharedBuffer(16 * kStride);
	gInputs = sharedBuffer(16 * kStride);
	gZeroCell = sharedBuffer(kStride);
	for (NSUInteger k = 0; k < 16; k++) {
		if (k < 8)
			*(float *)((uint8_t *)gCells.contents + k * kStride) = (float)k;
		float *c = (float *)((uint8_t *)gColours.contents + k * kStride);
		c[0] = (float)(k * 17) / 255.0f, c[1] = 0, c[2] = 0, c[3] = 1;
		*(uint32_t *)((uint8_t *)gInputs.contents + k * kStride) = (uint32_t)(1000 + k);
	}
	MTLTextureDescriptor *td =
	    [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
	                                                       width:8
	                                                      height:1
	                                                   mipmapped:NO];
	td.usage = MTLTextureUsageRenderTarget;
	td.storageMode = MTLStorageModePrivate;
	gTarget = [gDevice newTextureWithDescriptor:td];
}

static id<MTLIndirectCommandBuffer> drawBuffer(BOOL inherit, NSUInteger count, NSUInteger binds)
{
	MTLIndirectCommandBufferDescriptor *d = [MTLIndirectCommandBufferDescriptor new];
	d.commandTypes = MTLIndirectCommandTypeDraw;
	d.inheritPipelineState = inherit;
	d.inheritBuffers = inherit;
	d.maxVertexBufferBindCount = inherit ? 0 : binds;
	d.maxFragmentBufferBindCount = inherit ? 0 : binds;
	id<MTLIndirectCommandBuffer> b =
	    [gDevice newIndirectCommandBufferWithDescriptor:d
	                                    maxCommandCount:count
	                                            options:MTLResourceStorageModePrivate];
	if (b == nil)
		[NSException raise:@"icb" format:@"no ICB of %lu draws", (unsigned long)count];
	return b;
}

// Command i: pixel i % 8, colour i % 15 (4 calls).
static void encodeDraws(id<MTLIndirectCommandBuffer> b, NSUInteger count)
{
	for (NSUInteger i = 0; i < count; i++) {
		id<MTLIndirectRenderCommand> c = [b indirectRenderCommandAtIndex:i];
		[c setRenderPipelineState:gDraw];
		[c setVertexBuffer:gCells offset:(i % 8) * kStride atIndex:0];
		[c setFragmentBuffer:gColours offset:(i % 15) * kStride atIndex:0];
		[c drawPrimitives:MTLPrimitiveTypeTriangleStrip
		      vertexStart:0
		      vertexCount:4
		    instanceCount:1
		     baseInstance:0];
	}
}

// Command i: a draw alone of pixel i % 8 (1 call).
static void encodeInherited(id<MTLIndirectCommandBuffer> b, NSUInteger count)
{
	for (NSUInteger i = 0; i < count; i++)
		[[b indirectRenderCommandAtIndex:i] drawPrimitives:MTLPrimitiveTypeTriangleStrip
		                                       vertexStart:0
		                                       vertexCount:4
		                                     instanceCount:1
		                                      baseInstance:i % 8];
}

// Executes `b`'s `count` commands into the target, cleared to black, under the encoder's own
// pipeline and buffers (cell 0, colour 7) where `inherit`, and answers the 8 pixels' red bytes as
// colour indices (red / 17), '?' where a red is no colour's.
static NSString *executeDraws(id<MTLIndirectCommandBuffer> b, NSUInteger count, BOOL inherit)
{
	MTLRenderPassDescriptor *p = [MTLRenderPassDescriptor renderPassDescriptor];
	p.colorAttachments[0].texture = gTarget;
	p.colorAttachments[0].loadAction = MTLLoadActionClear;
	p.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 1);
	p.colorAttachments[0].storeAction = MTLStoreActionStore;
	id<MTLBuffer> out = sharedBuffer(kStride);
	id<MTLCommandBuffer> cb = [gQueue commandBuffer];
	id<MTLRenderCommandEncoder> e = [cb renderCommandEncoderWithDescriptor:p];
	[e useResource:gCells usage:MTLResourceUsageRead stages:MTLRenderStageVertex];
	[e useResource:gColours usage:MTLResourceUsageRead stages:MTLRenderStageFragment];
	if (inherit) {
		[e setRenderPipelineState:gDraw];
		[e setVertexBuffer:gZeroCell offset:0 atIndex:0];
		[e setFragmentBuffer:gColours offset:7 * kStride atIndex:0];
	}
	[e executeCommandsInBuffer:b withRange:NSMakeRange(0, count)];
	[e endEncoding];
	id<MTLBlitCommandEncoder> bl = [cb blitCommandEncoder];
	[bl copyFromTexture:gTarget
	             sourceSlice:0
	             sourceLevel:0
	            sourceOrigin:MTLOriginMake(0, 0, 0)
	              sourceSize:MTLSizeMake(8, 1, 1)
	                toBuffer:out
	       destinationOffset:0
	  destinationBytesPerRow:32
	destinationBytesPerImage:32];
	[bl endEncoding];
	[cb commit];
	finish(cb);
	const uint8_t *px = out.contents;
	char cells[9] = {0};
	for (int k = 0; k < 8; k++) {
		const unsigned red = px[k * 4];
		cells[k] = red % 17 == 0 && red / 17 < 16 ? "0123456789abcdef"[red / 17] : '?';
	}
	return @(cells);
}

// What `count` draws of encodeDraws leave in the target.
static NSString *drawsExpected(NSUInteger count)
{
	char cells[9] = {0};
	for (NSUInteger k = 0; k < 8; k++) {
		NSUInteger last = NSNotFound;
		for (NSUInteger i = count < 16 ? 0 : count - 16; i < count; i++)
			if (i % 8 == k)
				last = i;
		cells[k] = last == NSNotFound ? '0' : "0123456789abcdef"[last % 15];
	}
	return @(cells);
}

// Times `encode` over the case's repetitions, each followed by an empty command buffer's commit:
// "encode_ns=... call_ns=... mcmd_s=... commit_us=... base_us=...".
static NSString *timeEncodes(NSUInteger calls, void (^encode)(void))
{
	@autoreleasepool {
		encode();
	}
	emptyCommit();
	double enc[64], commit[64], base[64];
	const int reps = gReps < 64 ? gReps : 64;
	for (int r = 0; r < reps; r++) {
		base[r] = emptyCommit();
		const double t0 = now();
		@autoreleasepool {
			encode();
		}
		enc[r] = now() - t0;
		commit[r] = emptyCommit();
	}
	double least = enc[0];
	for (int r = 1; r < reps; r++)
		least = enc[r] < least ? enc[r] : least;
	const double e = median(enc, reps);
	const double n = (double)gCommands;
	return [NSString stringWithFormat:@"commands=%lu encode_ns=%.0f encode_min_ns=%.0f call_ns=%.0f "
	                                  @"mcmd_s=%.3f commit_us=%.0f base_us=%.0f",
	                                  (unsigned long)gCommands, e * 1e9 / n, least * 1e9 / n,
	                                  e * 1e9 / (n * calls), n / e / 1e6,
	                                  median(commit, reps) * 1e6, median(base, reps) * 1e6];
}

// --- cases --------------------------------------------------------------------------------------

static NSString *caseRender(NSUInteger binds)
{
	id<MTLIndirectCommandBuffer> b = drawBuffer(NO, gCommands, binds);
	NSString *t = timeEncodes(4, ^{
	  encodeDraws(b, gCommands);
	});
	NSString *got = executeDraws(b, gCommands, NO), *want = drawsExpected(gCommands);
	if (![got isEqualToString:want])
		return [NSString stringWithFormat:@"FAIL %@ drew %@, not %@", t, got, want];
	return [NSString stringWithFormat:@"%@ drew=%@", t, got];
}

static NSString *caseInherit(void)
{
	id<MTLIndirectCommandBuffer> b = drawBuffer(YES, gCommands, 0);
	NSString *t = timeEncodes(1, ^{
	  encodeInherited(b, gCommands);
	});
	// Pixel k drawn by every command of base instance k, in the encoder's colour 7.
	NSString *got = executeDraws(b, gCommands, YES);
	if (![got isEqualToString:@"77777777"])
		return [NSString stringWithFormat:@"FAIL %@ drew %@, not 77777777", t, got];
	return [NSString stringWithFormat:@"%@ drew=%@", t, got];
}

static NSString *caseCompute(void)
{
	MTLIndirectCommandBufferDescriptor *d = [MTLIndirectCommandBufferDescriptor new];
	d.commandTypes = MTLIndirectCommandTypeConcurrentDispatch;
	d.inheritPipelineState = NO;
	d.inheritBuffers = NO;
	d.maxKernelBufferBindCount = 2;
	id<MTLIndirectCommandBuffer> b =
	    [gDevice newIndirectCommandBufferWithDescriptor:d
	                                    maxCommandCount:gCommands
	                                            options:MTLResourceStorageModePrivate];
	if (b == nil)
		return [NSString stringWithFormat:@"FAIL no ICB of %lu dispatches", (unsigned long)gCommands];
	id<MTLBuffer> out = sharedBuffer(gCommands * 4);
	const NSUInteger count = gCommands;
	NSString *t = timeEncodes(4, ^{
	  for (NSUInteger i = 0; i < count; i++) {
		  id<MTLIndirectComputeCommand> c = [b indirectComputeCommandAtIndex:i];
		  [c setComputePipelineState:gCopy];
		  [c setKernelBuffer:out offset:i * 4 atIndex:0];
		  [c setKernelBuffer:gInputs offset:(i % 16) * kStride atIndex:1];
		  [c concurrentDispatchThreadgroups:MTLSizeMake(1, 1, 1)
		              threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
	  }
	});
	id<MTLCommandBuffer> cb = [gQueue commandBuffer];
	id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
	[e useResource:out usage:MTLResourceUsageWrite];
	[e useResource:gInputs usage:MTLResourceUsageRead];
	[e executeCommandsInBuffer:b withRange:NSMakeRange(0, count)];
	[e endEncoding];
	[cb commit];
	finish(cb);
	const uint32_t *w = out.contents;
	NSUInteger bad = 0, first = NSNotFound;
	for (NSUInteger i = 0; i < count; i++) {
		if (w[i] != 1000 + i % 16) {
			bad++;
			first = first == NSNotFound ? i : first;
		}
	}
	if (bad != 0)
		return [NSString stringWithFormat:@"FAIL %@ %lu of %lu words wrong, the first %lu: %u", t,
		                                  (unsigned long)bad, (unsigned long)count,
		                                  (unsigned long)first, w[first]];
	return [NSString stringWithFormat:@"%@ words_ok=%lu", t, (unsigned long)count];
}

static NSString *caseReset(void)
{
	id<MTLIndirectCommandBuffer> b = drawBuffer(NO, gCommands, 1);
	double v[64];
	const int reps = gReps < 64 ? gReps : 64;
	for (int r = 0; r < reps; r++) {
		@autoreleasepool {
			encodeDraws(b, gCommands);
		}
		const double t0 = now();
		[b resetWithRange:NSMakeRange(0, gCommands)];
		v[r] = now() - t0;
	}
	emptyCommit();
	// Every command reset: the target stays clear.
	NSString *got = executeDraws(b, gCommands, NO);
	if (![got isEqualToString:@"00000000"])
		return [NSString stringWithFormat:@"FAIL reset ICB drew %@", got];
	return [NSString stringWithFormat:@"commands=%lu reset_us=%.0f", (unsigned long)gCommands,
	                                  median(v, reps) * 1e6];
}

static NSString *caseSurface(NSUInteger w, NSUInteger h)
{
	NSDictionary *props = @{
		(id)kIOSurfaceWidth : @(w),
		(id)kIOSurfaceHeight : @(h),
		(id)kIOSurfaceBytesPerElement : @4,
		(id)kIOSurfaceBytesPerRow : @((w * 4 + 63) / 64 * 64),
		(id)kIOSurfacePixelFormat : @((uint32_t)'BGRA'),
	};
	IOSurfaceRef s = IOSurfaceCreate((__bridge CFDictionaryRef)props);
	if (s == NULL)
		return @"FAIL no IOSurface";
	const size_t rowBytes = IOSurfaceGetBytesPerRow(s), bytes = rowBytes * h;
	MTLTextureDescriptor *td =
	    [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
	                                                       width:w
	                                                      height:h
	                                                   mipmapped:NO];
	td.usage = MTLTextureUsageShaderRead;
	td.storageMode = MTLStorageModeManaged;
	id<MTLTexture> tex = [gDevice newTextureWithDescriptor:td iosurface:s plane:0];
	if (tex == nil) {
		CFRelease(s);
		return @"FAIL no texture over the IOSurface";
	}
	id<MTLBuffer> out = sharedBuffer(w * 4 * h);
	id<MTLCommandBuffer> (^blit)(void) = ^{
	  id<MTLCommandBuffer> cb = [gQueue commandBuffer];
	  id<MTLBlitCommandEncoder> bl = [cb blitCommandEncoder];
	  [bl copyFromTexture:tex
	               sourceSlice:0
	               sourceLevel:0
	              sourceOrigin:MTLOriginMake(0, 0, 0)
	                sourceSize:MTLSizeMake(w, h, 1)
	                  toBuffer:out
	         destinationOffset:0
	    destinationBytesPerRow:w * 4
	  destinationBytesPerImage:w * 4 * h];
	  [bl endEncoding];
	  return cb;
	};
	double wr[64], e2e[64], base[64];
	const int reps = gReps < 64 ? gReps : 64;
	uint32_t value = 0;
	for (int r = -1; r < reps; r++) {
		id<MTLCommandBuffer> cb = blit();
		double t0 = now();
		[cb commit];
		finish(cb);
		const double b0 = now() - t0;
		value = 0x01020304u * (uint32_t)(r + 2);
		t0 = now();
		IOSurfaceLock(s, 0, NULL);
		uint8_t *p = IOSurfaceGetBaseAddress(s);
		for (size_t y = 0; y < h; y++) {
			uint32_t *row = (uint32_t *)(p + y * rowBytes);
			for (size_t x = 0; x < w; x++)
				row[x] = value;
		}
		IOSurfaceUnlock(s, 0, NULL);
		const double t1 = now();
		cb = blit();
		[cb commit];
		finish(cb);
		const double t2 = now();
		if (r >= 0)
			wr[r] = t1 - t0, e2e[r] = t2 - t0, base[r] = b0;
	}
	CFRelease(s);
	const uint32_t *o = out.contents;
	NSUInteger bad = 0;
	for (size_t i = 0; i < w * h; i++)
		bad += o[i] != value;
	const double wm = median(wr, reps), em = median(e2e, reps), bm = median(base, reps);
	NSString *line = [NSString
	    stringWithFormat:@"row_bytes=%zu mib=%.2f write_ms=%.2f e2e_ms=%.2f base_ms=%.2f mb_s=%.1f",
	                     rowBytes, bytes / 1048576.0, wm * 1e3, em * 1e3, bm * 1e3,
	                     em > bm ? bytes / (em - bm) / 1e6 : 0.0];
	if (bad != 0)
		return [NSString stringWithFormat:@"FAIL %@ %lu of %zu texels not the last write", line,
		                                  (unsigned long)bad, w * h];
	return line;
}

// --- main ---------------------------------------------------------------------------------------

int main(int argc, char **argv)
{
	@autoreleasepool {
		gStart = now();
		const char *want = NULL;
		BOOL list = NO;
		int opt;
		while ((opt = getopt(argc, argv, "d:n:r:l")) != -1) {
			switch (opt) {
			case 'd': want = optarg; break;
			case 'n': gCommands = (NSUInteger)strtoul(optarg, NULL, 10); break;
			case 'r': gReps = atoi(optarg); break;
			case 'l': list = YES; break;
			default:
				fprintf(stderr, "usage: icb_probe [-d NAME] [-n COMMANDS] [-r REPS] [-l] [CASE...]\n");
				return 2;
			}
		}
		if (gCommands < 16 || gReps < 1) {
			fprintf(stderr, "icb_probe: -n 16 at least, -r 1 at least\n");
			return 2;
		}
		NSMutableArray<NSString *> *patterns = [NSMutableArray array];
		for (int i = optind; i < argc; i++)
			[patterns addObject:@(argv[i])];
		gPatterns = patterns;
		NSArray<id<MTLDevice>> *all = MTLCopyAllDevices();
		if (list) {
			for (id<MTLDevice> d in all)
				printf("%s\n", d.name.UTF8String);
			return 0;
		}
		if (want != NULL) {
			for (id<MTLDevice> d in all)
				if ([d.name containsString:@(want)]) {
					gDevice = d;
					break;
				}
		} else {
			gDevice = MTLCreateSystemDefaultDevice();
		}
		if (gDevice == nil) {
			fprintf(stderr, "icb_probe: no device\n");
			return 1;
		}
		gQueue = [gDevice newCommandQueue];
		printf("# device %s, %lu commands, %d repetitions\n", gDevice.name.UTF8String,
		       (unsigned long)gCommands, gReps);
		@try {
			setup();
		} @catch (NSException *e) {
			printf("icb: setup FAIL %s\n", (e.reason ?: e.name).UTF8String);
			return 1;
		}
		runCase(@"render", ^{
		  return caseRender(1);
		});
		runCase(@"wide", ^{
		  return caseRender(31);
		});
		runCase(@"inherit", ^{
		  return caseInherit();
		});
		runCase(@"compute", ^{
		  return caseCompute();
		});
		runCase(@"reset", ^{
		  return caseReset();
		});
		runCase(@"surface.1024x1024", ^{
		  return caseSurface(1024, 1024);
		});
		runCase(@"surface.1040x1024", ^{
		  return caseSurface(1040, 1024);
		});
	}
	return 0;
}
