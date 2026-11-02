// lock_probe: whether a CPU lock of an IOSurface waits for the GPU's use of it as Metal's does.
//   lock_probe [MS [GPU]]  runs every case; MS is how long each case's command buffer runs on the
//                     GPU (300 by default; the work is calibrated on the device first), on the
//                     device whose name contains GPU, else MTLCreateSystemDefaultDevice's.
// Each case makes a 2048 x 2048 BGRA8 surface the CPU filled with 0x11, then commits one command
// buffer that writes it (a compute kernel, or a render pass) or only reads it, and locks it from
// another process (this binary again, "child") or from this one. What Metal answers, measured on
// a MacBook Pro's AMD Radeon Pro 5300M and Intel UHD 630 under macOS 26.7, the answers the x86-64
// guest follows:
//   - another process's read-only or read-write lock of a surface a pending buffer writes returns
//     once the buffer completes, and every pixel is the GPU's;
//   - so does one made after commitAndWaitUntilSubmitted, the way WindowServer hands a capture's
//     surface to its reader;
//   - so does the committing process's own lock once the buffer is scheduled; made right after
//     commit, before Metal has submitted the buffer, it may return at once (Metal submits on a
//     thread of its own, so either answer passes);
//   - a pending buffer that only reads the surface holds no lock, read-only or read-write (the
//     M3 Max under macOS 27 differs: its read-write lock waits for the read);
//   - a write moves the surface's seed by one, by the buffer's submission, and an AvoidSync lock
//     answers kIOReturnCannotLock after it until a plain lock has synced the surface.
// And, as information, what another process reads with no lock once a writing buffer completes:
// the CPU's old pixels on those two GPUs, whose locks sync the surface, the GPU's on the M3 Max.
// Prints a line per case, PASS or FAIL against those answers (INFO for the last), then "LOCK OK"
// when every case passed, else "LOCK FAIL".
// Its shaders are lock_probe.metal, loaded from lock_probe.metallib beside the executable.
// Build: xcrun clang -arch x86_64 -fobjc-arc -framework Metal -framework IOSurface
//        -framework Foundation lock_probe.m -o lock_probe
//        xcrun -sdk macosx metal -c -std=metal3.1 -mmacosx-version-min=14.0 lock_probe.metal
//        -o lock_probe.air
//        xcrun -sdk macosx metallib lock_probe.air -o lock_probe.metallib
#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <Metal/Metal.h>
#include <mach/mach_time.h>
#include <spawn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

// The macOS 27 SDK deprecates global surfaces, which the child looks this one up as.
#pragma clang diagnostic ignored "-Wdeprecated-declarations"

extern char **environ;

// Metal's SPI WindowServer ends each composite with (measured on the M3 Max: it returns once the
// buffer is submitted, before its scheduled handlers run).
@protocol LockProbeSubmitSPI
- (void)commitAndWaitUntilSubmitted;
@end

static const int kW = 2048, kH = 2048;
// The GPU's colour (64, 128, 192, 255) / 255 as BGRA8 in memory, and the CPU's fill. A GPU's
// conversion may round a channel one either way.
static const uint32_t kGpuPixel = 0xff4080c0;
static const uint32_t kCpuPixel = 0x11111111;

static double ms(uint64_t a, uint64_t b)
{
	static mach_timebase_info_data_t tb;
	if (tb.denom == 0)
		mach_timebase_info(&tb);
	return ((double)b - (double)a) * tb.numer / tb.denom / 1e6;
}

// The child: looks surface ID up, locks it (mode 0 read-only, 1 read-write, 2 AvoidSync
// read-only, 3 no lock at all), checks every pixel against EXPECT where it is not 0, unlocks, and
// prints its lock's start and end (mach_absolute_time), its answer and the first pixel that
// differed.
// Whether `got` is `want` within one in each channel.
static bool samePixel(uint32_t got, uint32_t want)
{
	for (int c = 0; c < 32; c += 8) {
		int d = (int)((got >> c) & 0xff) - (int)((want >> c) & 0xff);
		if (d < -1 || d > 1)
			return false;
	}
	return true;
}

// The first pixel of `s` that is not `expect`, with its value in `value`, or -1. The caller holds
// the lock it reads under, if any.
static long firstBad(IOSurfaceRef s, uint32_t expect, uint32_t *value)
{
	const uint32_t *px = (const uint32_t *)IOSurfaceGetBaseAddress(s);
	size_t stride = IOSurfaceGetBytesPerRow(s) / 4;
	for (long y = 0; y < kH; y++) {
		for (long x = 0; x < kW; x++) {
			if (!samePixel(px[y * stride + x], expect)) {
				*value = px[y * stride + x];
				return y * kW + x;
			}
		}
	}
	return -1;
}

static int child(char **argv)
{
	uint32_t sid = (uint32_t)strtoul(argv[2], NULL, 0);
	int mode = atoi(argv[3]);
	uint32_t expect = (uint32_t)strtoul(argv[4], NULL, 0);
	IOSurfaceRef s = IOSurfaceLookup(sid);
	if (!s) {
		printf("nosurface\n");
		return 1;
	}
	uint32_t options = mode == 0 ? kIOSurfaceLockReadOnly
	                 : mode == 1 ? 0
	                             : kIOSurfaceLockReadOnly | kIOSurfaceLockAvoidSync;
	uint64_t a = mach_absolute_time();
	kern_return_t r = mode == 3 ? KERN_SUCCESS : IOSurfaceLock(s, options, NULL);
	uint64_t b = mach_absolute_time();
	long bad = -1;
	uint32_t badValue = 0;
	if (r == KERN_SUCCESS) {
		if (expect != 0)
			bad = firstBad(s, expect, &badValue);
		if (mode != 3)
			IOSurfaceUnlock(s, options, NULL);
	}
	printf("%llu %llu 0x%x %ld 0x%08x\n", a, b, r, bad, badValue);
	CFRelease(s);
	return 0;
}

typedef struct {
	uint64_t start, end;
	uint32_t answer;
	long bad;
	uint32_t badValue;
	bool ok;
} ChildResult;

// Runs the child on surface `sid`, and waits for it.
static ChildResult runChild(const char *self, uint32_t sid, int mode, uint32_t expect)
{
	ChildResult res = {0};
	int fds[2];
	if (pipe(fds) != 0)
		return res;
	char id[32], m[8], e[32];
	snprintf(id, sizeof id, "%u", sid);
	snprintf(m, sizeof m, "%d", mode);
	snprintf(e, sizeof e, "0x%x", expect);
	char *args[] = {(char *)self, (char *)"child", id, m, e, NULL};
	posix_spawn_file_actions_t fa;
	posix_spawn_file_actions_init(&fa);
	posix_spawn_file_actions_adddup2(&fa, fds[1], 1);
	posix_spawn_file_actions_addclose(&fa, fds[0]);
	pid_t pid;
	int sr = posix_spawn(&pid, self, &fa, NULL, args, environ);
	posix_spawn_file_actions_destroy(&fa);
	close(fds[1]);
	char line[256] = {0};
	FILE *f = fdopen(fds[0], "r");
	if (f && fgets(line, sizeof line, f)) {
		unsigned long long a = 0, b = 0;
		unsigned int r = 0, bv = 0;
		long bad = 0;
		if (sscanf(line, "%llu %llu %x %ld %x", &a, &b, &r, &bad, &bv) == 5)
			res = (ChildResult){a, b, r, bad, bv, true};
	}
	if (f)
		fclose(f);
	if (sr == 0) {
		int st;
		waitpid(pid, &st, 0);
	}
	return res;
}

typedef struct {
	id<MTLDevice> device;
	id<MTLCommandQueue> queue;
	id<MTLComputePipelineState> writer, reader;
	id<MTLRenderPipelineState> render;
	id<MTLBuffer> sink;
	uint32_t n;
	const char *self;
	int failures;
} Probe;

typedef enum { kCompute, kRender, kRead } Work;

static IOSurfaceRef newSurface(void)
{
	NSDictionary *props = @{
		(id)kIOSurfaceWidth : @(kW),
		(id)kIOSurfaceHeight : @(kH),
		(id)kIOSurfaceBytesPerElement : @4,
		(id)kIOSurfacePixelFormat : @((uint32_t)'BGRA'),
		(id)kIOSurfaceIsGlobal : @YES,
	};
	IOSurfaceRef s = IOSurfaceCreate((__bridge CFDictionaryRef)props);
	IOSurfaceLock(s, 0, NULL);
	memset(IOSurfaceGetBaseAddress(s), 0x11, IOSurfaceGetAllocSize(s));
	IOSurfaceUnlock(s, 0, NULL);
	return s;
}

// A command buffer that does `work` to a texture over `s`, not committed.
static id<MTLCommandBuffer> encode(Probe *p, IOSurfaceRef s, Work work)
{
	MTLTextureDescriptor *td =
	    [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
	                                                       width:kW
	                                                      height:kH
	                                                   mipmapped:NO];
	td.usage = MTLTextureUsageShaderWrite | MTLTextureUsageShaderRead |
	           MTLTextureUsageRenderTarget;
	// A discrete GPU makes an IOSurface's texture Managed, one that shares the CPU's memory
	// Shared.
	td.storageMode = p->device.hasUnifiedMemory ? MTLStorageModeShared : MTLStorageModeManaged;
	id<MTLTexture> tex = [p->device newTextureWithDescriptor:td iosurface:s plane:0];
	id<MTLCommandBuffer> cb = [p->queue commandBuffer];
	float k = 0;
	if (work == kRender) {
		MTLRenderPassDescriptor *rp = [MTLRenderPassDescriptor renderPassDescriptor];
		rp.colorAttachments[0].texture = tex;
		rp.colorAttachments[0].loadAction = MTLLoadActionClear;
		rp.colorAttachments[0].storeAction = MTLStoreActionStore;
		id<MTLRenderCommandEncoder> e = [cb renderCommandEncoderWithDescriptor:rp];
		[e setRenderPipelineState:p->render];
		// A fragment's work is a fraction of the kernel's: the same loop over far fewer
		// threads at once on most GPUs, so it runs a quarter of it.
		uint32_t n = p->n / 4;
		[e setFragmentBytes:&n length:4 atIndex:0];
		[e setFragmentBytes:&k length:4 atIndex:1];
		[e drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
		[e endEncoding];
	} else {
		id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
		[e setComputePipelineState:work == kRead ? p->reader : p->writer];
		[e setTexture:tex atIndex:0];
		[e setBytes:&p->n length:4 atIndex:0];
		[e setBytes:&k length:4 atIndex:1];
		[e setBuffer:p->sink offset:0 atIndex:2];
		[e dispatchThreads:MTLSizeMake(kW, kH, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
		[e endEncoding];
	}
	return cb;
}

static const char *kModeNames[] = {"read-only", "read-write", "AvoidSync"};

// How a case commits its buffer: plainly, with commitAndWaitUntilSubmitted, or plainly and then
// waits until it is scheduled.
typedef enum { kCommit, kSubmitted, kScheduled } Submit;

// What Metal's lock does: waits for the buffer, returns at once, or either.
typedef enum { kWaits, kAtOnce, kEither } Answer;

// One case: `work` committed as `submit` says, then a lock of `mode` by another process, or by
// this one where `own`. `answer` says what Metal's lock does.
static void lockCase(Probe *p, const char *name, Work work, Submit submit, bool own, int mode,
                     Answer answer)
{
	bool waits = answer == kWaits;
	IOSurfaceRef s = newSurface();
	id<MTLCommandBuffer> cb = encode(p, s, work);
	__block uint64_t done = 0;
	[cb addCompletedHandler:^(id<MTLCommandBuffer> c) {
	  done = mach_absolute_time();
	}];
	uint64_t t0 = mach_absolute_time();
	if (submit == kSubmitted && [cb respondsToSelector:@selector(commitAndWaitUntilSubmitted)])
		[(id<LockProbeSubmitSPI>)cb commitAndWaitUntilSubmitted];
	else
		[cb commit];
	if (submit == kScheduled)
		[cb waitUntilScheduled];
	uint32_t expect = work == kRead ? kCpuPixel : kGpuPixel;
	ChildResult r = {0};
	if (own) {
		uint32_t options = mode == 0 ? kIOSurfaceLockReadOnly : 0;
		r.start = mach_absolute_time();
		r.answer = IOSurfaceLock(s, options, NULL);
		r.end = mach_absolute_time();
		r.bad = waits && r.answer == KERN_SUCCESS ? firstBad(s, expect, &r.badValue) : -1;
		IOSurfaceUnlock(s, options, NULL);
		r.ok = true;
	} else {
		r = runChild(p->self, IOSurfaceGetID(s), mode, waits ? expect : 0);
	}
	[cb waitUntilCompleted];
	double gpu = ms(t0, done);
	bool pass = r.ok && r.answer == KERN_SUCCESS && cb.status == MTLCommandBufferStatusCompleted;
	const char *why = !r.ok                  ? "no answer from the child"
	                  : r.answer != 0        ? "the lock failed"
	                  : cb.status != MTLCommandBufferStatusCompleted ? "the buffer failed"
	                                                                 : "";
	// Waited: returned no earlier than a tenth of the buffer before its completion (the
	// completed handler runs after what the lock waits for). At once: within a fifth of it.
	double before = ms(r.end, done);
	if (pass && waits && before > gpu / 10) {
		pass = false;
		why = "the lock returned before the buffer completed";
	}
	if (pass && answer == kAtOnce && ms(r.start, r.end) > gpu / 5) {
		pass = false;
		why = "the lock waited";
	}
	if (pass && waits && r.bad >= 0) {
		pass = false;
		why = "a pixel is not the GPU's";
	}
	printf("%s %-50s lock %s: took %6.1f ms, returned %6.1f ms after commit; buffer completed "
	       "%6.1f ms after commit",
	       pass ? "PASS" : "FAIL", name, kModeNames[mode], ms(r.start, r.end), ms(t0, r.end), gpu);
	if (r.bad >= 0)
		printf("; pixel %ld is 0x%08x", r.bad, r.badValue);
	printf("%s%s\n", *why ? ": " : "", why);
	if (!pass)
		p->failures++;
	CFRelease(s);
}

// A writing buffer completes, then another process reads the surface with no lock: information,
// counted neither way.
static void unlockedCase(Probe *p)
{
	IOSurfaceRef s = newSurface();
	id<MTLCommandBuffer> cb = encode(p, s, kCompute);
	[cb commit];
	[cb waitUntilCompleted];
	ChildResult r = runChild(p->self, IOSurfaceGetID(s), 3, kGpuPixel);
	printf("INFO %-50s ", "completed write, another process, no lock");
	if (!r.ok)
		printf("no answer from the child\n");
	else if (r.bad < 0)
		printf("every pixel is the GPU's\n");
	else
		printf("pixel %ld is 0x%08x%s\n", r.bad, r.badValue,
		       samePixel(r.badValue, kCpuPixel) ? ", the CPU's" : "");
	CFRelease(s);
}

// The seed and AvoidSync after a write: the seed moves by one, by the buffer's completion, an
// AvoidSync lock from another process answers kIOReturnCannotLock, and once this process's plain
// lock has synced the surface it succeeds.
static void seedCase(Probe *p)
{
	IOSurfaceRef s = newSurface();
	uint32_t before = IOSurfaceGetSeed(s);
	id<MTLCommandBuffer> cb = encode(p, s, kCompute);
	[cb commit];
	[cb waitUntilCompleted];
	uint32_t after = IOSurfaceGetSeed(s);
	ChildResult avoid = runChild(p->self, IOSurfaceGetID(s), 2, 0);
	IOSurfaceLock(s, kIOSurfaceLockReadOnly, NULL);
	IOSurfaceUnlock(s, kIOSurfaceLockReadOnly, NULL);
	ChildResult synced = runChild(p->self, IOSurfaceGetID(s), 2, 0);
	bool seedOk = after == before + 1;
	bool avoidOk = avoid.ok && avoid.answer == kIOReturnCannotLock;
	bool syncedOk = synced.ok && synced.answer == KERN_SUCCESS;
	printf("%s %-50s seed %u -> %u\n", seedOk ? "PASS" : "FAIL", "a write moves the seed by one",
	       before, after);
	printf("%s %-50s other process: 0x%x before a plain lock, 0x%x after\n",
	       avoidOk && syncedOk ? "PASS" : "FAIL", "AvoidSync after a write", avoid.answer,
	       synced.answer);
	p->failures += !seedOk + !(avoidOk && syncedOk);
	CFRelease(s);
}

int main(int argc, char **argv)
{
	setvbuf(stdout, NULL, _IONBF, 0);
	if (argc == 5 && strcmp(argv[1], "child") == 0)
		return child(argv);
	double target = argc > 1 ? atof(argv[1]) : 300;
	@autoreleasepool {
		Probe probe = {0};
		Probe *p = &probe;
		p->self = argv[0];
		if (argc > 2) {
			for (id<MTLDevice> d in MTLCopyAllDevices()) {
				if ([d.name rangeOfString:@(argv[2])].location != NSNotFound) {
					p->device = d;
					break;
				}
			}
		} else {
			p->device = MTLCreateSystemDefaultDevice();
		}
		if (!p->device) {
			printf("LOCK FAIL: no Metal device\n");
			return 1;
		}
		printf("device %s, unified memory %d\n", p->device.name.UTF8String,
		       p->device.hasUnifiedMemory);
		p->queue = [p->device newCommandQueue];
		NSError *err = nil;
		// The shaders come compiled (lock_probe.metal), as the guest's device compiles no source.
		NSString *dir = [@(argv[0]) stringByDeletingLastPathComponent];
		NSURL *url = [NSURL fileURLWithPath:[dir stringByAppendingPathComponent:@"lock_probe.metallib"]];
		id<MTLLibrary> lib = [p->device newLibraryWithURL:url error:&err];
		if (!lib) {
			printf("LOCK FAIL: library %s: %s\n", url.path.UTF8String, err.description.UTF8String);
			return 1;
		}
		p->writer = [p->device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"spin"]
		                                                   error:&err];
		p->reader = [p->device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"reader"]
		                                                   error:&err];
		MTLRenderPipelineDescriptor *rpd = [MTLRenderPipelineDescriptor new];
		rpd.vertexFunction = [lib newFunctionWithName:@"vs"];
		rpd.fragmentFunction = [lib newFunctionWithName:@"fs"];
		rpd.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
		p->render = [p->device newRenderPipelineStateWithDescriptor:rpd error:&err];
		p->sink = [p->device newBufferWithLength:16 options:MTLResourceStorageModeShared];
		if (!p->writer || !p->reader || !p->render) {
			printf("LOCK FAIL: pipelines: %s\n", err.description.UTF8String);
			return 1;
		}
		// Calibrate the loop for `target` ms of GPU work.
		p->n = 2000;
		for (int i = 0; i < 4; i++) {
			IOSurfaceRef s = newSurface();
			id<MTLCommandBuffer> cb = encode(p, s, kCompute);
			uint64_t t0 = mach_absolute_time();
			[cb commit];
			[cb waitUntilCompleted];
			double took = ms(t0, mach_absolute_time());
			CFRelease(s);
			if (took > target * 0.7 && took < target * 1.5)
				break;
			double scale = target / (took > 0.5 ? took : 0.5);
			p->n = (uint32_t)fmin(fmax(p->n * fmin(scale, 50.0), 1.0), 4e9);
		}
		printf("loop of %u iterations a pixel\n", p->n);
		lockCase(p, "write, another process", kCompute, kCommit, false, 0, kWaits);
		lockCase(p, "write, another process", kCompute, kCommit, false, 1, kWaits);
		lockCase(p, "render pass, another process", kRender, kCommit, false, 0, kWaits);
		lockCase(p, "write, waited until submitted, another process", kCompute, kSubmitted, false,
		         0, kWaits);
		lockCase(p, "write, the committing process, at commit", kCompute, kCommit, true, 0,
		         kEither);
		lockCase(p, "write, the committing process, at commit", kCompute, kCommit, true, 1,
		         kEither);
		lockCase(p, "write, the committing process, once scheduled", kCompute, kScheduled, true,
		         0, kWaits);
		lockCase(p, "write, the committing process, once scheduled", kCompute, kScheduled, true,
		         1, kWaits);
		lockCase(p, "read, another process", kRead, kCommit, false, 0, kAtOnce);
		lockCase(p, "read, another process", kRead, kCommit, false, 1, kAtOnce);
		unlockedCase(p);
		seedCase(p);
		printf("%s\n", p->failures == 0 ? "LOCK OK" : "LOCK FAIL");
		return p->failures == 0 ? 0 : 1;
	}
}
