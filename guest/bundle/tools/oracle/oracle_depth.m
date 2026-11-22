// oracle_depth: how many command buffers one queue holds in flight in front of a busy GPU: where
// -commandBuffer blocks (the queue's maxCommandBufferCount), whether commit or
// commitAndWaitUntilSubmitted blocks before that (a limit of the driver or the kernel), and how
// far ahead of the GPU the device answers scheduled.
//   oracle_depth DEVICE [TARGET_MS [COUNT]]
// The spin is calibrated as oracle_backlog's: 2000 steps run three times, the last two's mean GPU
// time scaled to TARGET_MS (default 10); the header line prints the spin and one buffer's GPU time.
// Each arm makes a new queue, then makes and commits buffers of that spin back to back from one
// thread, COUNT of them (default 48; 80 on a queue of the default count), and waits for the last.
//   arm q32-cwus:   newCommandQueueWithMaxCommandBufferCount:32, each by commitAndWaitUntilSubmitted
//   arm q32-commit: the same queue size, each by commit
//   arm qdef-cwus:  newCommandQueue, each by commitAndWaitUntilSubmitted
//   arm qdef-commit: newCommandQueue, each by commit
//   arm q32-cwus-short: as q32-cwus with a spin of TARGET_MS / 5
// Per arm, one line (times in ms, from the thread's own clock):
//   max: the queue's maxCommandBufferCount where the queue answers it (SPI), else "?";
//   make-block: how many -commandBuffer calls took over 1 ms, the first one's index and the
//     buffers then made and not completed ("uncompleted");
//   commit-block: the same for the commit call;
//   uncompleted max: the most buffers made and not completed at any buffer's creation;
//   sched-ahead max: the most buffers scheduled and not completed at any buffer's scheduled time;
//   commit-to-scheduled: p50 and max over all buffers, and the first 4's and the last 4's p50;
//   call: the commit call's p50 and max; GPU: the buffers' mean GPU time; span: first commit to
//   last completion, against COUNT x the mean GPU time.
// With ORACLE_TRACE=1, each buffer's line follows: index, make, commit call, scheduled and
// completed (from the arm's start), its GPU start and end (GPU clock, from the first's start), and
// the uncompleted count at its creation.
// Q49 ran it on each GPU, off screen, as:
//   caffeinate -s ./oracle_depth DEVICE 10 48   (DEVICE 5300M, 630 or Apple)
// dgpu-oracle.md Q49 holds what it found. Build: see README.md.
#import "oracle_common.h"
#include <objc/message.h>

enum { kSide = 1024, kMax = 256 };

static NSString *const kSource = @"#include <metal_stdlib>\n"
	"using namespace metal;\n"
	"kernel void spin(texture2d<uint, access::write> t [[texture(0)]],\n"
	"  constant uint &n [[buffer(0)]], uint2 g [[thread_position_in_grid]]) {\n"
	"  uint x = g.x * 2654435761u + g.y;\n"
	"  for (uint i = 0; i < n; i++) x = x * 1664525u + 1013904223u;\n"
	"  t.write(uint4(x == 7u ? 1u : 0u), g); }\n";

static id<MTLComputePipelineState> gSpin;
static id<MTLTexture> gTex;

static id<MTLCommandBuffer> busy(id<MTLCommandBuffer> cb, uint32_t n)
{
	id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
	[e setComputePipelineState:gSpin];
	[e setTexture:gTex atIndex:0];
	[e setBytes:&n length:4 atIndex:0];
	[e dispatchThreads:MTLSizeMake(kSide, kSide, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
	[e endEncoding];
	return cb;
}

static void cwus(id<MTLCommandBuffer> cb)
{
	SEL sel = NSSelectorFromString(@"commitAndWaitUntilSubmitted");
	if (![(id)cb respondsToSelector:sel]) {
		printf("  ! no commitAndWaitUntilSubmitted: plain commit\n");
		[cb commit];
		return;
	}
	((void (*)(id, SEL))objc_msgSend)(cb, sel);
}

static int cmpd(const void *a, const void *b)
{
	double x = *(const double *)a, y = *(const double *)b;
	return x < y ? -1 : x > y;
}

static double pct(const double *v, int n, double p)
{
	if (n <= 0)
		return 0;
	double *s = malloc(sizeof(double) * n);
	memcpy(s, v, sizeof(double) * n);
	qsort(s, n, sizeof(double), cmpd);
	int i = (int)(p * (n - 1) + 0.5);
	double r = s[i];
	free(s);
	return r;
}

static void arm(id<MTLDevice> dev, const char *name, int qmax, bool useCwus, uint32_t spinN, int count, bool trace)
{
	if (count > kMax)
		count = kMax;
	id<MTLCommandQueue> q = qmax ? [dev newCommandQueueWithMaxCommandBufferCount:qmax] : [dev newCommandQueue];
	long maxAns = -1;
	if ([(id)q respondsToSelector:NSSelectorFromString(@"maxCommandBufferCount")])
		maxAns = ((NSUInteger (*)(id, SEL))objc_msgSend)(q, NSSelectorFromString(@"maxCommandBufferCount"));
	__block double *sched = calloc(kMax, sizeof(double)), *done = calloc(kMax, sizeof(double));
	double mk0[kMax], mk1[kMax], cm1[kMax], gs[kMax], ge[kMax];
	int unc[kMax];
	NSMutableArray *cbs = [NSMutableArray array];
	const double t0 = oracle_now_ms();
	for (int i = 0; i < count; i++) {
		mk0[i] = oracle_now_ms() - t0;
		id<MTLCommandBuffer> cb = [q commandBuffer];
		mk1[i] = oracle_now_ms() - t0;
		int u = 0;
		for (int k = 0; k < i; k++)
			u += !(done[k] > 0 && done[k] <= mk0[i]);
		unc[i] = u;
		busy(cb, spinN);
		[cb addScheduledHandler:^(id<MTLCommandBuffer> x) { sched[i] = oracle_now_ms() - t0; }];
		[cb addCompletedHandler:^(id<MTLCommandBuffer> x) { done[i] = oracle_now_ms() - t0; }];
		const double c0 = oracle_now_ms() - t0;
		if (useCwus)
			cwus(cb);
		else
			[cb commit];
		cm1[i] = oracle_now_ms() - t0;
		mk0[i] = mk1[i] - mk0[i];
		mk1[i] = c0;
		[cbs addObject:cb];
	}
	[(id<MTLCommandBuffer>)cbs.lastObject waitUntilCompleted];
	for (id<MTLCommandBuffer> cb in cbs)
		[cb waitUntilCompleted];
	usleep(5000);
	int makeBlocks = 0, makeFirst = -1, commitBlocks = 0, commitFirst = -1, uncMax = 0, schedAhead = 0;
	double callT[kMax], c2s[kMax], gpu = 0;
	for (int i = 0; i < count; i++) {
		id<MTLCommandBuffer> cb = cbs[i];
		gs[i] = cb.GPUStartTime;
		ge[i] = cb.GPUEndTime;
		gpu += (ge[i] - gs[i]) * 1e3 / count;
		callT[i] = cm1[i] - mk1[i];
		c2s[i] = sched[i] - mk1[i];
		if (mk0[i] > 1.0) {
			makeBlocks++;
			if (makeFirst < 0)
				makeFirst = i;
		}
		if (callT[i] > 1.0) {
			commitBlocks++;
			if (commitFirst < 0)
				commitFirst = i;
		}
		if (unc[i] > uncMax)
			uncMax = unc[i];
		int a = 0;
		for (int k = 0; k < count; k++)
			a += sched[k] <= sched[i] && done[k] > sched[i];
		if (a > schedAhead)
			schedAhead = a;
	}
	double maxDone = 0;
	for (int i = 0; i < count; i++)
		if (done[i] > maxDone)
			maxDone = done[i];
	char mb[64] = "none", cb1[64] = "none";
	if (makeFirst >= 0)
		snprintf(mb, sizeof mb, "first at #%d (uncompleted %d, %.2f ms)", makeFirst, unc[makeFirst], mk0[makeFirst]);
	if (commitFirst >= 0)
		snprintf(cb1, sizeof cb1, "first at #%d (uncompleted %d, %.2f ms)", commitFirst, unc[commitFirst],
		         callT[commitFirst]);
	printf("arm %s: max %s%ld, %d buffers; make-block %d, %s; commit-block %d, %s; uncompleted max %d; "
	       "sched-ahead max %d; commit-to-scheduled p50 %.3f max %.3f (first 4 p50 %.3f, last 4 p50 %.3f); "
	       "call p50 %.3f max %.3f; GPU mean %.2f; span %.1f (count x GPU %.1f)\n",
	       name, maxAns < 0 ? "?" : "", maxAns < 0 ? 0 : maxAns, count, makeBlocks, mb, commitBlocks, cb1, uncMax,
	       schedAhead, pct(c2s, count, 0.5), pct(c2s, count, 1.0), pct(c2s, count < 4 ? count : 4, 0.5),
	       pct(c2s + (count > 4 ? count - 4 : 0), count < 4 ? count : 4, 0.5), pct(callT, count, 0.5),
	       pct(callT, count, 1.0), gpu, maxDone - mk1[0], count * gpu);
	if (trace) {
		for (int i = 0; i < count; i++)
			printf("  #%d make %.3f commit %.3f..%.3f sched %.3f done %.3f gpu %.3f..%.3f unc %d\n", i, mk0[i],
			       mk1[i], cm1[i], sched[i], done[i], (gs[i] - gs[0]) * 1e3, (ge[i] - gs[0]) * 1e3, unc[i]);
	}
	free(sched);
	free(done);
}

int main(int argc, char **argv)
{
	setvbuf(stdout, NULL, _IOLBF, 0);
	@autoreleasepool {
		if (argc < 2) {
			fprintf(stderr, "usage: %s DEVICE [TARGET_MS [COUNT]]\n", argv[0]);
			return 2;
		}
		const double target = argc > 2 ? atof(argv[2]) : 10.0;
		const int count = argc > 3 ? atoi(argv[3]) : 48;
		const bool trace = getenv("ORACLE_TRACE") && strcmp(getenv("ORACLE_TRACE"), "1") == 0;
		id<MTLDevice> dev = oracle_device(argv[1]);
		oracle_header("oracle_depth", dev);
		NSError *err = nil;
		id<MTLLibrary> lib = [dev newLibraryWithSource:kSource options:nil error:&err];
		gSpin = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"spin"] error:&err];
		if (!gSpin) {
			printf("  ! %s\n", err.description.UTF8String);
			return 2;
		}
		MTLTextureDescriptor *d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR32Uint
		                                                                            width:kSide height:kSide mipmapped:NO];
		d.usage = MTLTextureUsageShaderWrite;
		d.storageMode = MTLStorageModePrivate;
		gTex = [dev newTextureWithDescriptor:d];
		id<MTLCommandQueue> cq = [dev newCommandQueue];
		double per = 0;
		for (int i = 0; i < 3; i++) {
			id<MTLCommandBuffer> cb = busy([cq commandBuffer], 2000);
			[cb commit];
			[cb waitUntilCompleted];
			if (i > 0)
				per += (cb.GPUEndTime - cb.GPUStartTime) * 1e3 / 2;
		}
		uint32_t spinN = (uint32_t)(2000.0 * target / per), shortN = spinN / 5 ? spinN / 5 : 1;
		{
			id<MTLCommandBuffer> cb = busy([cq commandBuffer], spinN);
			[cb commit];
			[cb waitUntilCompleted];
			printf("# spin %u: %.2f ms GPU (target %.1f); count %d\n", spinN, (cb.GPUEndTime - cb.GPUStartTime) * 1e3,
			       target, count);
		}
		arm(dev, "q32-cwus", 32, true, spinN, count, trace);
		usleep(300000);
		arm(dev, "q32-commit", 32, false, spinN, count, trace);
		usleep(300000);
		arm(dev, "qdef-cwus", 0, true, spinN, count < 80 ? 80 : count, trace);
		usleep(300000);
		arm(dev, "qdef-commit", 0, false, spinN, count < 80 ? 80 : count, trace);
		usleep(300000);
		arm(dev, "q32-cwus-short", 32, true, shortN, count, trace);
	}
	return 0;
}
