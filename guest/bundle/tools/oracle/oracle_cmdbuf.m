// oracle_cmdbuf: the order the device keeps between encoders and command buffers on its own
// (hazard tracking), and when a command buffer is scheduled and completed, on which threads its
// handlers run, its GPU times and the threads of the asynchronous pipeline creators.
//   oracle_cmdbuf DEVICE [CASE]   every case (or the one named), each in a child process
// Cases:
//   hazards   a slow kernel writes 7 to every word of a buffer (each thread spins first, then
//             writes), and a second dispatch copies the buffer to another, under each ordering
//             Metal offers or leaves out: tracked and untracked buffers, fences, one serial
//             encoder, one concurrent encoder with and without a barrier, a blit after the
//             kernel, a second command buffer, a buffer from an untracked heap, and a second
//             command buffer on a second queue, tracked and untracked, committed once the first's
//             commitAndWaitUntilSubmitted returns (else its commit), and tracked, committed
//             right after the first's commit. Prints the copied words that are not 7 (the copy
//             ran before the write), over 5 runs each.
//   timing    a ~200 ms command buffer and a tiny one behind it: when each is scheduled and
//             completed, from commit; waitUntilScheduled and commitAndWaitUntilSubmitted
//             (where the buffer responds to it) of the tiny one; the handlers' threads and
//             order; GPUStartTime/GPUEndTime and kernelStartTime/kernelEndTime against the
//             host clock; a buffer waiting on an unsignalled shared event; the asynchronous
//             pipeline creators' handler threads; tiny buffers' commit-to-scheduled and
//             commit-to-completed latency.
//   submitted where the buffer responds to commitAndWaitUntilSubmitted, a tiny buffer's call,
//             20 times each: behind a ~100 ms buffer on the same queue, on an idle queue, for a
//             buffer that first waits on an unsignalled shared event, which a second thread
//             signals 100 ms after the call, and behind a buffer of its queue that waits on such
//             an event. For each: when the call returns (ms from the first commit), the buffer's
//             status then (and the long buffer's), whether its scheduled handler had run by
//             then, when that handler runs, and how long waitUntilScheduled blocks right after
//             the call.
//   events    when the CPU sees a shared event's signal encoded between two ~100 ms kernels of
//             one command buffer, 10 runs: a listener's block (notifyListener:atValue:block:), a
//             thread polling signaledValue and a thread in waitUntilSignaledValue:timeoutMS:,
//             against the buffer's GPUStartTime, GPUEndTime and completed handler; a buffer that
//             signals 1 after a ~50 ms kernel and then waits for 2 before another, where the CPU
//             waits for 1 and only then sets 2 (from the waiting thread, and from a listener's
//             block), 10 runs each: whether the wait for 1 returned, when, and when the buffer
//             completed; and a signal at the very end of a buffer, 20 runs: its listener's block
//             against the buffer's completed handler. Times are ms from the commit.
//   gputimes  what GPUStartTime and GPUEndTime answer for one long command buffer: K encoders
//             (K = 1, 8, 32) of ~0.5 ms of GPU work each in one buffer ("whole"), against the
//             same encoders committed back to back one buffer each ("apart"), compute
//             (slow_write) and render (K passes over a 1920 x 1080 BGRA8 target, each clearing
//             it and blending 6 full-target quads), 7 runs each, and a whole compute buffer
//             whose CPU sleeps 1 ms between its encoders as it encodes them. For each, medians
//             of: whole, the span (GPUEndTime less GPUStartTime), GPUStartTime from the commit,
//             and the completed handler from GPUEndTime; apart, the buffers' spans summed, the
//             span from the first's start to the last's end, and the first's start from the
//             first commit. Times are ms.
//   schedsig  the command-buffer SPI encodeSignalEventScheduled:value: (where the buffer responds to
//             it): which class answers it; when a shared event it signals reads the value, against
//             the buffer's scheduled handler, GPUStartTime and GPUEndTime, encoded before and after a
//             ~100 ms kernel on an idle queue and behind a ~200 ms buffer of its queue, where a
//             buffer of a second queue waits for the value; the value it leaves on an event at 5
//             (3, 7, two signals in either order) against a GPU signal and a CPU set of 3; the
//             order against GPU signals of the buffer; against a wait of the buffer on another
//             event, encoded before and after it, and behind a kernel, and of an earlier buffer of
//             its queue; against a wait on the same event; and the queue of a listener's block. Times are ms from the commit.
//   schedsig_open, schedsig_late, schedsig_nil, schedsig_nil_unretained, schedsig_plain,
//   schedsig_plain_unretained
//             the same call with a blit encoder open, after the commit, with a nil event and with
//             a plain event (newEvent), the last two on a buffer that retains its references and
//             on one that does not: a line after the call, the buffer's status and error, the
//             event's value.
// Build: see README.md.
#import "oracle_common.h"
#include <dispatch/dispatch.h>
#include <objc/message.h>
#include <objc/runtime.h>
#include <pthread.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

extern char **environ;

static id<MTLDevice> D;
static id<MTLCommandQueue> Q;
static id<MTLLibrary> L;
static const char *gArgv0;
static id<MTLComputePipelineState> gSlow, gCopy;

static id<MTLComputePipelineState> cpipe(const char *name)
{
	NSError *err = nil;
	id<MTLComputePipelineState> p =
	    [D newComputePipelineStateWithFunction:[L newFunctionWithName:@(name)] error:&err];
	if (!p)
		printf("  ! pipeline %s: %s\n", name, err.description.UTF8String);
	return p;
}

enum { N = 1 << 20 };
static uint32_t gIters = 4000;

static void enc_slow(id<MTLComputeCommandEncoder> e, id<MTLBuffer> b, uint32_t iters)
{
	uint32_t value = 7, mask = 0;
	[e setComputePipelineState:gSlow];
	[e setBuffer:b offset:0 atIndex:0];
	[e setBytes:&iters length:4 atIndex:1];
	[e setBytes:&value length:4 atIndex:2];
	[e setBytes:&mask length:4 atIndex:3];
	[e dispatchThreads:MTLSizeMake(N, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

static void enc_copy(id<MTLComputeCommandEncoder> e, id<MTLBuffer> in, id<MTLBuffer> out)
{
	[e setComputePipelineState:gCopy];
	[e setBuffer:in offset:0 atIndex:0];
	[e setBuffer:out offset:0 atIndex:1];
	[e dispatchThreads:MTLSizeMake(N, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

static size_t not_seven(id<MTLBuffer> c)
{
	const uint32_t *p = c.contents;
	size_t bad = 0;
	for (size_t i = 0; i < N; i++)
		bad += p[i] != 7;
	return bad;
}

enum Order {
	kTracked2Enc,
	kUntracked2Enc,
	kUntracked2EncFence,
	kUntrackedSerial,
	kUntrackedConcurrent,
	kUntrackedConcurrentBarrier,
	kUntrackedBlitAfter,
	kUntracked2Cb,
	kHeap2Enc,
	kTracked2Queue,
	kUntracked2Queue,
	kTracked2QueueAtCommit,
	kOrderCount
};

static const char *const kOrderName[] = {
    "tracked, two compute encoders",
    "untracked, two compute encoders, no fence",
    "untracked, two compute encoders, a fence",
    "untracked, one serial encoder, two dispatches",
    "untracked, one concurrent encoder, no barrier",
    "untracked, one concurrent encoder, memoryBarrier(buffers)",
    "untracked, compute encoder then a blit copy, no fence",
    "untracked, two command buffers, one queue",
    "from an untracked heap, two compute encoders, no fence",
    "tracked, two command buffers, two queues, the second committed once the first is submitted",
    "untracked, two command buffers, two queues, the second committed once the first is submitted",
    "tracked, two command buffers, two queues, the second committed right after the first's commit",
};

static id<MTLCommandQueue> gQ2;

static size_t run_order(enum Order o, id<MTLHeap> heap, double *gpu_ms)
{
	int tracked = o == kTracked2Enc || o == kTracked2Queue || o == kTracked2QueueAtCommit;
	MTLResourceOptions opt = MTLResourceStorageModeShared |
	                         (tracked ? MTLResourceHazardTrackingModeTracked
	                                  : MTLResourceHazardTrackingModeUntracked);
	id<MTLBuffer> b = o == kHeap2Enc
	                      ? [heap newBufferWithLength:N * 4 options:MTLResourceStorageModePrivate]
	                      : [D newBufferWithLength:N * 4 options:opt];
	id<MTLBuffer> c = [D newBufferWithLength:N * 4 options:MTLResourceStorageModeShared];
	memset(c.contents, 0xee, N * 4);
	id<MTLCommandBuffer> cb0 = [Q commandBuffer];
	id<MTLBlitCommandEncoder> z = [cb0 blitCommandEncoder];
	[z fillBuffer:b range:NSMakeRange(0, N * 4) value:0];
	[z endEncoding];
	oracle_commit_wait(cb0, NULL, NULL);
	id<MTLCommandBuffer> cb = [Q commandBuffer];
	id<MTLCommandBuffer> cb2 = nil;
	switch (o) {
	case kTracked2Enc:
	case kUntracked2Enc:
	case kHeap2Enc: {
		id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
		enc_slow(e, b, gIters);
		[e endEncoding];
		e = [cb computeCommandEncoder];
		enc_copy(e, b, c);
		[e endEncoding];
		break;
	}
	case kUntracked2EncFence: {
		id<MTLFence> f = [D newFence];
		id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
		enc_slow(e, b, gIters);
		[e updateFence:f];
		[e endEncoding];
		e = [cb computeCommandEncoder];
		[e waitForFence:f];
		enc_copy(e, b, c);
		[e endEncoding];
		break;
	}
	case kUntrackedSerial:
	case kUntrackedConcurrent:
	case kUntrackedConcurrentBarrier: {
		id<MTLComputeCommandEncoder> e =
		    [cb computeCommandEncoderWithDispatchType:o == kUntrackedSerial
		                                                  ? MTLDispatchTypeSerial
		                                                  : MTLDispatchTypeConcurrent];
		enc_slow(e, b, gIters);
		if (o == kUntrackedConcurrentBarrier)
			[e memoryBarrierWithScope:MTLBarrierScopeBuffers];
		enc_copy(e, b, c);
		[e endEncoding];
		break;
	}
	case kUntrackedBlitAfter: {
		id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
		enc_slow(e, b, gIters);
		[e endEncoding];
		id<MTLBlitCommandEncoder> bl = [cb blitCommandEncoder];
		[bl copyFromBuffer:b sourceOffset:0 toBuffer:c destinationOffset:0 size:N * 4];
		[bl endEncoding];
		break;
	}
	case kUntracked2Cb:
	case kTracked2Queue:
	case kUntracked2Queue:
	case kTracked2QueueAtCommit: {
		id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
		enc_slow(e, b, gIters);
		[e endEncoding];
		cb2 = [o == kUntracked2Cb ? Q : gQ2 commandBuffer];
		e = [cb2 computeCommandEncoder];
		enc_copy(e, b, c);
		[e endEncoding];
		break;
	}
	default:
		break;
	}
	SEL caws = sel_registerName("commitAndWaitUntilSubmitted");
	if ((o == kTracked2Queue || o == kUntracked2Queue) && [(id)cb respondsToSelector:caws])
		((void (*)(id, SEL))objc_msgSend)((id)cb, caws);
	else
		[cb commit];
	if (cb2)
		[cb2 commit];
	[cb waitUntilCompleted];
	[cb2 waitUntilCompleted];
	*gpu_ms = (cb.GPUEndTime - cb.GPUStartTime) * 1e3;
	if (cb.status != MTLCommandBufferStatusCompleted || (cb2 && cb2.status != MTLCommandBufferStatusCompleted))
		printf("  ! status %ld / %ld\n", (long)cb.status, (long)cb2.status);
	return not_seven(c);
}

static void case_hazards(void)
{
	gSlow = cpipe("slow_write");
	gCopy = cpipe("copy_u32");
	if (!gSlow || !gCopy)
		return;
	// Calibrate: the slow kernel alone near 30 ms.
	for (int k = 0; k < 6; k++) {
		id<MTLBuffer> b = [D newBufferWithLength:N * 4 options:MTLResourceStorageModeShared];
		id<MTLCommandBuffer> cb = [Q commandBuffer];
		id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
		enc_slow(e, b, gIters);
		[e endEncoding];
		double gpu;
		oracle_commit_wait(cb, NULL, &gpu);
		if (gpu > 20 || k == 5) {
			printf("case hazards: slow kernel %u iterations x %d threads = %.1f ms GPU\n", gIters, N,
			       gpu);
			break;
		}
		gIters = (uint32_t)(gIters * (30.0 / (gpu > 0.5 ? gpu : 0.5)));
	}
	MTLHeapDescriptor *hd = [MTLHeapDescriptor new];
	hd.size = 64 << 20;
	hd.storageMode = MTLStorageModePrivate;
	hd.hazardTrackingMode = MTLHazardTrackingModeUntracked;
	id<MTLHeap> heap = [D newHeapWithDescriptor:hd];
	gQ2 = [D newCommandQueue];
	for (int o = 0; o < kOrderCount; o++) {
		size_t bad = 0;
		double gpu = 0, worst = 0;
		int racy = 0;
		for (int r = 0; r < 5; r++) {
			@autoreleasepool {
				size_t n = run_order((enum Order)o, heap, &gpu);
				bad += n;
				racy += n != 0;
				if (n > worst)
					worst = n;
			}
		}
		printf("  %-58s | %d of 5 runs raced, %zu words stale (max %.0f of %d) | last %.1f ms\n",
		       kOrderName[o], racy, bad, worst, N, gpu);
	}
}

// ---- timing

typedef struct {
	char what[48];
	double at_ms;
	int main_thread;
	uint64_t tid;
	char queue[64];
} Event;

static Event gEvents[64];
static _Atomic int gEventCount;
static double gT0;

static void note(const char *what)
{
	int i = gEventCount++;
	if (i >= 64)
		return;
	Event *e = &gEvents[i];
	snprintf(e->what, sizeof e->what, "%s", what);
	e->at_ms = oracle_now_ms() - gT0;
	e->main_thread = pthread_main_np();
	pthread_threadid_np(NULL, &e->tid);
	const char *q = dispatch_queue_get_label(DISPATCH_CURRENT_QUEUE_LABEL);
	snprintf(e->queue, sizeof e->queue, "%s", q ? q : "");
}

static void print_events(uint64_t main_tid)
{
	int n = gEventCount < 64 ? gEventCount : 64;
	for (int i = 0; i < n; i++)
		printf("    %8.2f ms  %-34s thread %s%llx queue '%s'\n", gEvents[i].at_ms, gEvents[i].what,
		       gEvents[i].main_thread ? "MAIN " : "", gEvents[i].tid == main_tid ? 0ULL : gEvents[i].tid,
		       gEvents[i].queue);
	gEventCount = 0;
}

static id<MTLCommandBuffer> tiny_cb(id<MTLCommandQueue> q, id<MTLBuffer> b)
{
	id<MTLCommandBuffer> cb = [q commandBuffer];
	id<MTLBlitCommandEncoder> e = [cb blitCommandEncoder];
	[e fillBuffer:b range:NSMakeRange(0, 256) value:1];
	[e endEncoding];
	return cb;
}

static id<MTLCommandBuffer> long_cb(id<MTLCommandQueue> q, id<MTLBuffer> b, uint32_t iters)
{
	id<MTLCommandBuffer> cb = [q commandBuffer];
	id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
	enc_slow(e, b, iters);
	[e endEncoding];
	return cb;
}

static void case_timing(void)
{
	gSlow = cpipe("slow_write");
	gCopy = cpipe("copy_u32");
	if (!gSlow)
		return;
	uint64_t main_tid;
	pthread_threadid_np(NULL, &main_tid);
	id<MTLBuffer> big = [D newBufferWithLength:N * 4 options:MTLResourceStorageModeShared];
	id<MTLBuffer> small = [D newBufferWithLength:4096 options:MTLResourceStorageModeShared];
	// Calibrate the long buffer near 200 ms.
	uint32_t iters = 4000;
	for (int k = 0; k < 6; k++) {
		id<MTLCommandBuffer> cb = long_cb(Q, big, iters);
		double gpu;
		oracle_commit_wait(cb, NULL, &gpu);
		if ((gpu > 150 && gpu < 300) || k == 5) {
			printf("case timing: long buffer %u iterations = %.1f ms GPU\n", iters, gpu);
			break;
		}
		iters = (uint32_t)(iters * (200.0 / (gpu > 0.5 ? gpu : 0.5)));
	}

	printf("  1. a long buffer, then a tiny one behind it on the same queue (ms from the long one's commit)\n");
	{
		id<MTLCommandBuffer> a = long_cb(Q, big, iters), b = tiny_cb(Q, small);
		[a addScheduledHandler:^(id<MTLCommandBuffer> _) { note("long scheduled handler"); }];
		[a addCompletedHandler:^(id<MTLCommandBuffer> _) { note("long completed handler"); }];
		[b addScheduledHandler:^(id<MTLCommandBuffer> _) { note("tiny scheduled handler"); }];
		[b addCompletedHandler:^(id<MTLCommandBuffer> _) { note("tiny completed handler 1"); }];
		[b addCompletedHandler:^(id<MTLCommandBuffer> _) { note("tiny completed handler 2"); }];
		gT0 = oracle_now_ms();
		[a commit];
		note("long committed");
		[b commit];
		note("tiny committed");
		[b waitUntilScheduled];
		note("tiny waitUntilScheduled returned");
		[b waitUntilCompleted];
		note("tiny waitUntilCompleted returned");
		[a waitUntilCompleted];
		usleep(20000);
		print_events(main_tid);
		double c0 = gT0 / 1e3;
		printf("    long: kernelStart %+.2f kernelEnd %+.2f GPUStart %+.2f GPUEnd %+.2f ms\n",
		       (a.kernelStartTime - c0) * 1e3, (a.kernelEndTime - c0) * 1e3,
		       (a.GPUStartTime - c0) * 1e3, (a.GPUEndTime - c0) * 1e3);
		printf("    tiny: kernelStart %+.2f kernelEnd %+.2f GPUStart %+.2f GPUEnd %+.2f ms\n",
		       (b.kernelStartTime - c0) * 1e3, (b.kernelEndTime - c0) * 1e3,
		       (b.GPUStartTime - c0) * 1e3, (b.GPUEndTime - c0) * 1e3);
	}

	SEL caws = sel_registerName("commitAndWaitUntilSubmitted");
	printf("  2. commitAndWaitUntilSubmitted of a tiny buffer behind a long one\n");
	{
		id<MTLCommandBuffer> a = long_cb(Q, big, iters), b = tiny_cb(Q, small);
		if (![(id)b respondsToSelector:caws]) {
			printf("    n/r\n");
		} else {
			gT0 = oracle_now_ms();
			[a commit];
			((void (*)(id, SEL))objc_msgSend)((id)b, caws);
			note("commitAndWaitUntilSubmitted returned");
			[b waitUntilCompleted];
			note("tiny completed");
			[a waitUntilCompleted];
			print_events(main_tid);
		}
	}

	printf("  3. a tiny buffer on a second queue while a long one runs on the first\n");
	{
		id<MTLCommandQueue> q2 = [D newCommandQueue];
		id<MTLCommandBuffer> a = long_cb(Q, big, iters), b = tiny_cb(q2, small);
		[b addScheduledHandler:^(id<MTLCommandBuffer> _) { note("q2 tiny scheduled handler"); }];
		[b addCompletedHandler:^(id<MTLCommandBuffer> _) { note("q2 tiny completed handler"); }];
		[a addCompletedHandler:^(id<MTLCommandBuffer> _) { note("q1 long completed handler"); }];
		gT0 = oracle_now_ms();
		[a commit];
		[b commit];
		[b waitUntilCompleted];
		note("q2 tiny waitUntilCompleted returned");
		[a waitUntilCompleted];
		usleep(20000);
		print_events(main_tid);
	}

	printf("  4. a tiny buffer that first waits on an unsignalled shared event, signalled by the CPU at 300 ms\n");
	{
		id<MTLSharedEvent> ev = [D newSharedEvent];
		id<MTLCommandBuffer> b = [Q commandBuffer];
		[b encodeWaitForEvent:ev value:1];
		id<MTLBlitCommandEncoder> e = [b blitCommandEncoder];
		[e fillBuffer:small range:NSMakeRange(0, 256) value:2];
		[e endEncoding];
		[b addScheduledHandler:^(id<MTLCommandBuffer> _) { note("scheduled handler"); }];
		[b addCompletedHandler:^(id<MTLCommandBuffer> _) { note("completed handler"); }];
		gT0 = oracle_now_ms();
		[b commit];
		usleep(300000);
		note("CPU signals the event");
		ev.signaledValue = 1;
		[b waitUntilCompleted];
		usleep(20000);
		print_events(main_tid);
	}

	printf("  5. asynchronous pipeline creators: does the handler run before the call returns, on which thread\n");
	{
		id<MTLFunction> f = [L newFunctionWithName:@"add_one"];
		gT0 = oracle_now_ms();
		__block int done = 0;
		[D newComputePipelineStateWithFunction:f
		                     completionHandler:^(id<MTLComputePipelineState> p, NSError *err) {
			                     note(p ? "compute handler (made)" : "compute handler (nil)");
			                     done = 1;
		                     }];
		note(done ? "compute call returned (handler already ran)" : "compute call returned (handler pending)");
		MTLRenderPipelineDescriptor *pd = [MTLRenderPipelineDescriptor new];
		pd.vertexFunction = [L newFunctionWithName:@"quad_vs"];
		pd.fragmentFunction = [L newFunctionWithName:@"solid_fs"];
		pd.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm_sRGB;
		__block int rdone = 0;
		[D newRenderPipelineStateWithDescriptor:pd
		                      completionHandler:^(id<MTLRenderPipelineState> p, NSError *err) {
			                      note(p ? "render handler (made)" : "render handler (nil)");
			                      rdone = 1;
		                      }];
		note(rdone ? "render call returned (handler already ran)" : "render call returned (handler pending)");
		for (int k = 0; k < 200 && !(done && rdone); k++)
			usleep(10000);
		print_events(main_tid);
	}

	printf("  6. tiny buffers on an idle GPU, 30 each: commit -> scheduled handler, -> completed handler, -> waitUntilCompleted\n");
	{
		double sch[30], cmp[30], wait[30];
		for (int k = 0; k < 30; k++) {
			__block double ts = 0, tc = 0;
			id<MTLCommandBuffer> b = tiny_cb(Q, small);
			[b addScheduledHandler:^(id<MTLCommandBuffer> _) { ts = oracle_now_ms(); }];
			[b addCompletedHandler:^(id<MTLCommandBuffer> _) { tc = oracle_now_ms(); }];
			double t0 = oracle_now_ms();
			[b commit];
			[b waitUntilCompleted];
			double tw = oracle_now_ms();
			usleep(2000);
			sch[k] = ts - t0;
			cmp[k] = tc - t0;
			wait[k] = tw - t0;
			usleep(5000);
		}
		double *arrs[3] = {sch, cmp, wait};
		const char *names[3] = {"scheduled", "completed", "waitUntilCompleted"};
		for (int a = 0; a < 3; a++) {
			double *v = arrs[a];
			for (int i = 0; i < 30; i++)
				for (int j = i + 1; j < 30; j++)
					if (v[j] < v[i]) {
						double t = v[i];
						v[i] = v[j];
						v[j] = t;
					}
			printf("    %-20s median %.3f ms, min %.3f, p90 %.3f\n", names[a], v[15], v[0], v[27]);
		}
	}
}

static const char *status_name(MTLCommandBufferStatus s)
{
	switch (s) {
	case MTLCommandBufferStatusNotEnqueued:
		return "NotEnqueued";
	case MTLCommandBufferStatusEnqueued:
		return "Enqueued";
	case MTLCommandBufferStatusCommitted:
		return "Committed";
	case MTLCommandBufferStatusScheduled:
		return "Scheduled";
	case MTLCommandBufferStatusCompleted:
		return "Completed";
	case MTLCommandBufferStatusError:
		return "Error";
	}
	return "?";
}

static void sort_ms(double *v, int n)
{
	for (int i = 0; i < n; i++)
		for (int j = i + 1; j < n; j++)
			if (v[j] < v[i]) {
				double t = v[i];
				v[i] = v[j];
				v[j] = t;
			}
}

static void print_ms(const char *what, double *v, int n)
{
	sort_ms(v, n);
	printf("    %-46s median %8.3f ms, min %8.3f, max %8.3f\n", what, v[n / 2], v[0], v[n - 1]);
}

enum { SUBMITTED_RUNS = 20 };

// One run of each `submitted` scenario: 0 behind a long buffer, 1 on an idle queue, 2 waiting on
// an unsignalled shared event, 3 behind a buffer of its queue that waits on one. Times are ms from the first commit (the long buffer's, else the
// call's).
typedef struct {
	double returned, scheduled_handler, wait_scheduled;
	MTLCommandBufferStatus status, long_status;
	int handler_before_return;
} Submitted;

static Submitted submitted_run(int scenario, SEL caws, id<MTLBuffer> big, id<MTLBuffer> small, uint32_t iters)
{
	Submitted r = {0};
	id<MTLCommandBuffer> a = scenario == 0 ? long_cb(Q, big, iters) : nil;
	id<MTLSharedEvent> ev = scenario >= 2 ? [D newSharedEvent] : nil;
	id<MTLCommandBuffer> h = nil;
	if (scenario == 3) {
		h = [Q commandBuffer];
		[h encodeWaitForEvent:ev value:1];
	}
	id<MTLCommandBuffer> b = [Q commandBuffer];
	if (scenario == 2)
		[b encodeWaitForEvent:ev value:1];
	id<MTLBlitCommandEncoder> e = [b blitCommandEncoder];
	[e fillBuffer:small range:NSMakeRange(0, 256) value:1];
	[e endEncoding];
	__block volatile double ts = 0;
	[b addScheduledHandler:^(id<MTLCommandBuffer> _) { ts = oracle_now_ms(); }];
	double t0 = oracle_now_ms();
	if (ev)
		dispatch_after(dispatch_time(DISPATCH_TIME_NOW, 100 * NSEC_PER_MSEC),
		               dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{ ev.signaledValue = 1; });
	if (a)
		[a commit];
	if (h)
		[h commit];
	((void (*)(id, SEL))objc_msgSend)((id)b, caws);
	double t1 = oracle_now_ms();
	r.status = b.status;
	r.long_status = a ? a.status : MTLCommandBufferStatusNotEnqueued;
	r.handler_before_return = ts != 0;
	[b waitUntilScheduled];
	double t2 = oracle_now_ms();
	[b waitUntilCompleted];
	[a waitUntilCompleted];
	usleep(2000);
	r.returned = t1 - t0;
	r.wait_scheduled = t2 - t1;
	r.scheduled_handler = ts - t0;
	return r;
}

static void case_submitted(void)
{
	gSlow = cpipe("slow_write");
	if (!gSlow)
		return;
	id<MTLBuffer> big = [D newBufferWithLength:N * 4 options:MTLResourceStorageModeShared];
	id<MTLBuffer> small = [D newBufferWithLength:4096 options:MTLResourceStorageModeShared];
	SEL caws = sel_registerName("commitAndWaitUntilSubmitted");
	if (![(id)[Q commandBuffer] respondsToSelector:caws]) {
		printf("case submitted: n/r\n");
		return;
	}
	// Calibrate the long buffer near 100 ms.
	uint32_t iters = 2000;
	double gpu = 0;
	for (int k = 0; k < 6; k++) {
		oracle_commit_wait(long_cb(Q, big, iters), NULL, &gpu);
		if ((gpu > 75 && gpu < 150) || k == 5)
			break;
		iters = (uint32_t)(iters * (100.0 / (gpu > 0.5 ? gpu : 0.5)));
	}
	printf("case submitted: long buffer %u iterations = %.1f ms GPU\n", iters, gpu);
	static const char *const names[] = {
	    "1. behind a long buffer on the same queue",
	    "2. on an idle queue",
	    "3. waiting on a shared event a second thread signals at 100 ms",
	    "4. behind a buffer of its queue that waits on a shared event a second thread signals at 100 ms",
	};
	for (int s = 0; s < 4; s++) {
		printf("  %s, %d runs (ms from the first commit)\n", names[s], SUBMITTED_RUNS);
		double ret[SUBMITTED_RUNS], sch[SUBMITTED_RUNS], wsch[SUBMITTED_RUNS];
		int status[6] = {0}, long_status[6] = {0}, before = 0;
		for (int k = 0; k < SUBMITTED_RUNS; k++) {
			Submitted r = submitted_run(s, caws, big, small, iters);
			ret[k] = r.returned;
			sch[k] = r.scheduled_handler;
			wsch[k] = r.wait_scheduled;
			status[r.status < 6 ? r.status : 5]++;
			long_status[r.long_status < 6 ? r.long_status : 5]++;
			before += r.handler_before_return;
		}
		print_ms("commitAndWaitUntilSubmitted returns", ret, SUBMITTED_RUNS);
		print_ms("its scheduled handler runs", sch, SUBMITTED_RUNS);
		print_ms("waitUntilScheduled right after blocks for", wsch, SUBMITTED_RUNS);
		printf("    scheduled handler had run by the return: %d of %d\n", before, SUBMITTED_RUNS);
		printf("    status at the return:");
		for (int i = 0; i < 6; i++)
			if (status[i])
				printf(" %s %d", status_name(i), status[i]);
		printf("\n");
		if (s == 0) {
			printf("    the long buffer's status then:");
			for (int i = 0; i < 6; i++)
				if (long_status[i])
					printf(" %s %d", status_name(i), long_status[i]);
			printf("\n");
		}
	}
}

enum { EVENT_RUNS = 10, END_SIGNAL_RUNS = 20 };

// A command buffer of `parts` ~`iters` kernels, with `between` encoded after each but the last.
static id<MTLCommandBuffer> parts_cb(id<MTLBuffer> big, uint32_t iters, int parts,
                                     void (^between)(id<MTLCommandBuffer>, int))
{
	id<MTLCommandBuffer> cb = [Q commandBuffer];
	for (int k = 0; k < parts; k++) {
		id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
		enc_slow(e, big, iters);
		[e endEncoding];
		if (k + 1 < parts && between)
			between(cb, k);
	}
	return cb;
}

static void print_status_counts(int *status)
{
	printf("    status at the end:");
	for (int i = 0; i < 6; i++)
		if (status[i])
			printf(" %s %d", status_name(i), status[i]);
	printf("\n");
}

static void case_events(void)
{
	gSlow = cpipe("slow_write");
	if (!gSlow)
		return;
	id<MTLBuffer> big = [D newBufferWithLength:N * 4 options:MTLResourceStorageModeShared];
	// Calibrate one kernel near 100 ms.
	uint32_t iters = 2000;
	double gpu = 0;
	for (int k = 0; k < 6; k++) {
		oracle_commit_wait(long_cb(Q, big, iters), NULL, &gpu);
		if ((gpu > 75 && gpu < 150) || k == 5)
			break;
		iters = (uint32_t)(iters * (100.0 / (gpu > 0.5 ? gpu : 0.5)));
	}
	printf("case events: one kernel %u iterations = %.1f ms GPU\n", iters, gpu);
	MTLSharedEventListener *listener = [[MTLSharedEventListener alloc] init];
	dispatch_queue_t gq = dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0);

	printf("  1. kernel, signal 1, kernel in one buffer, %d runs (ms from the commit)\n", EVENT_RUNS);
	{
		double start[EVENT_RUNS], end[EVENT_RUNS], heard[EVENT_RUNS], polled[EVENT_RUNS];
		double waited[EVENT_RUNS], done[EVENT_RUNS], heard_gpu[EVENT_RUNS], left[EVENT_RUNS];
		int missed = 0;
		for (int k = 0; k < EVENT_RUNS; k++) {
			id<MTLSharedEvent> ev = [D newSharedEvent];
			id<MTLCommandBuffer> cb = parts_cb(big, iters, 2, ^(id<MTLCommandBuffer> c, int _) {
				[c encodeSignalEvent:ev value:1];
			});
			__block volatile double tl = 0, tc = 0, tp = 0, tw = 0;
			[ev notifyListener:listener
			           atValue:1
			             block:^(id<MTLSharedEvent> _e, uint64_t _v) { tl = oracle_now_ms(); }];
			[cb addCompletedHandler:^(id<MTLCommandBuffer> _) { tc = oracle_now_ms(); }];
			dispatch_group_t g = dispatch_group_create();
			double t0 = oracle_now_ms();
			[cb commit];
			dispatch_group_async(g, gq, ^{
				while (ev.signaledValue < 1 && oracle_now_ms() - t0 < 5000)
					;
				tp = oracle_now_ms();
			});
			dispatch_group_async(g, gq, ^{
				if ([ev waitUntilSignaledValue:1 timeoutMS:5000])
					tw = oracle_now_ms();
			});
			[cb waitUntilCompleted];
			dispatch_group_wait(g, dispatch_time(DISPATCH_TIME_NOW, 6 * NSEC_PER_SEC));
			usleep(5000);
			if (tl == 0 || tw == 0)
				missed++;
			double c0 = t0 / 1e3;
			start[k] = (cb.GPUStartTime - c0) * 1e3;
			end[k] = (cb.GPUEndTime - c0) * 1e3;
			heard[k] = tl - t0;
			polled[k] = tp - t0;
			waited[k] = tw - t0;
			done[k] = tc - t0;
			heard_gpu[k] = heard[k] - start[k];
			left[k] = done[k] - heard[k];
			usleep(5000);
		}
		print_ms("GPUStartTime", start, EVENT_RUNS);
		print_ms("signaledValue first reads 1 (polling)", polled, EVENT_RUNS);
		print_ms("waitUntilSignaledValue:1 returns", waited, EVENT_RUNS);
		print_ms("the listener's block runs", heard, EVENT_RUNS);
		print_ms("  the same, from GPUStartTime", heard_gpu, EVENT_RUNS);
		print_ms("GPUEndTime", end, EVENT_RUNS);
		print_ms("the completed handler runs", done, EVENT_RUNS);
		print_ms("  completed handler after the listener's block", left, EVENT_RUNS);
		printf("    runs where the listener or the wait missed the signal: %d of %d\n", missed,
		       EVENT_RUNS);
	}

	static const char *const pattern_names[] = {
	    "2. kernel, signal 1, wait for 2, kernel; the CPU waits for 1 (waitUntilSignaledValue), then sets 2",
	    "3. the same; a listener's block for 1 sets 2",
	};
	for (int s = 0; s < 2; s++) {
		printf("  %s, %d runs (ms from the commit)\n", pattern_names[s], EVENT_RUNS);
		double reached_at[EVENT_RUNS], set_at[EVENT_RUNS], done[EVENT_RUNS], end[EVENT_RUNS];
		int reached = 0, completed = 0, status[6] = {0};
		for (int k = 0; k < EVENT_RUNS; k++) {
			id<MTLSharedEvent> ev = [D newSharedEvent];
			id<MTLCommandBuffer> cb = parts_cb(big, iters / 2, 2, ^(id<MTLCommandBuffer> c, int _) {
				[c encodeSignalEvent:ev value:1];
				[c encodeWaitForEvent:ev value:2];
			});
			dispatch_semaphore_t fin = dispatch_semaphore_create(0);
			__block volatile double tc = 0, tr = 0, ts = 0;
			[cb addCompletedHandler:^(id<MTLCommandBuffer> _) {
				tc = oracle_now_ms();
				dispatch_semaphore_signal(fin);
			}];
			double t0 = oracle_now_ms();
			if (s == 1)
				[ev notifyListener:listener
				           atValue:1
				             block:^(id<MTLSharedEvent> e, uint64_t _v) {
					             tr = oracle_now_ms();
					             e.signaledValue = 2;
					             ts = oracle_now_ms();
				             }];
			[cb commit];
			if (s == 0) {
				BOOL got = [ev waitUntilSignaledValue:1 timeoutMS:5000];
				tr = got ? oracle_now_ms() : 0;
				ev.signaledValue = 2; // set whatever the wait answered, so the buffer ends
				ts = oracle_now_ms();
			} else {
				for (int w = 0; w < 500 && ts == 0; w++)
					usleep(10000);
				if (ts == 0) {
					ev.signaledValue = 2;
					ts = oracle_now_ms();
				}
			}
			if (dispatch_semaphore_wait(fin, dispatch_time(DISPATCH_TIME_NOW, 10 * NSEC_PER_SEC)) == 0)
				completed++;
			reached += tr != 0;
			status[cb.status < 6 ? cb.status : 5]++;
			reached_at[k] = tr ? tr - t0 : -1;
			set_at[k] = ts - t0;
			done[k] = tc ? tc - t0 : -1;
			end[k] = (cb.GPUEndTime - t0 / 1e3) * 1e3;
			usleep(5000);
		}
		print_ms("the CPU sees 1", reached_at, EVENT_RUNS);
		print_ms("the CPU has set 2", set_at, EVENT_RUNS);
		print_ms("GPUEndTime", end, EVENT_RUNS);
		print_ms("the completed handler runs", done, EVENT_RUNS);
		printf("    the CPU saw 1 before setting 2: %d of %d; completed within 10 s: %d of %d\n",
		       reached, EVENT_RUNS, completed, EVENT_RUNS);
		print_status_counts(status);
	}

	printf("  4. kernel, then signal 1 as the buffer's last command, %d runs (ms from the commit)\n",
	       END_SIGNAL_RUNS);
	{
		double heard[END_SIGNAL_RUNS], done[END_SIGNAL_RUNS], gap[END_SIGNAL_RUNS];
		double end[END_SIGNAL_RUNS];
		int before = 0;
		for (int k = 0; k < END_SIGNAL_RUNS; k++) {
			id<MTLSharedEvent> ev = [D newSharedEvent];
			id<MTLCommandBuffer> cb = parts_cb(big, iters / 2, 1, nil);
			[cb encodeSignalEvent:ev value:1];
			__block volatile double tl = 0, tc = 0;
			[ev notifyListener:listener
			           atValue:1
			             block:^(id<MTLSharedEvent> _e, uint64_t _v) { tl = oracle_now_ms(); }];
			[cb addCompletedHandler:^(id<MTLCommandBuffer> _) { tc = oracle_now_ms(); }];
			double t0 = oracle_now_ms();
			[cb commit];
			[cb waitUntilCompleted];
			for (int w = 0; w < 100 && (tl == 0 || tc == 0); w++)
				usleep(1000);
			before += tl != 0 && tl <= tc;
			heard[k] = tl - t0;
			done[k] = tc - t0;
			gap[k] = tc - tl;
			end[k] = (cb.GPUEndTime - t0 / 1e3) * 1e3;
			usleep(5000);
		}
		print_ms("GPUEndTime", end, END_SIGNAL_RUNS);
		print_ms("the listener's block runs", heard, END_SIGNAL_RUNS);
		print_ms("the completed handler runs", done, END_SIGNAL_RUNS);
		print_ms("  completed handler after the listener's block", gap, END_SIGNAL_RUNS);
		printf("    the listener's block ran first: %d of %d\n", before, END_SIGNAL_RUNS);
	}
}

// ---- schedsig

enum { SCHED_RUNS = 5 };
static SEL gSchedSel;

static void sched_signal(id<MTLCommandBuffer> cb, id<MTLEvent> ev, uint64_t value)
{
	((void (*)(id, SEL, id, uint64_t))objc_msgSend)((id)cb, gSchedSel, ev, value);
}

// The class in `obj`'s chain whose own methods hold `sel`, or "none".
static const char *defining_class(id obj, SEL sel)
{
	for (Class c = object_getClass(obj); c; c = class_getSuperclass(c)) {
		unsigned n = 0;
		Method *ms = class_copyMethodList(c, &n);
		BOOL has = NO;
		for (unsigned i = 0; i < n && !has; i++)
			has = method_getName(ms[i]) == sel;
		free(ms);
		if (has)
			return class_getName(c);
	}
	return "none";
}

static void enc_tiny(id<MTLCommandBuffer> cb, id<MTLBuffer> b)
{
	id<MTLBlitCommandEncoder> e = [cb blitCommandEncoder];
	[e fillBuffer:b range:NSMakeRange(0, 256) value:1];
	[e endEncoding];
}

static void enc_kernel(id<MTLCommandBuffer> cb, id<MTLBuffer> big, uint32_t iters)
{
	id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
	enc_slow(e, big, iters);
	[e endEncoding];
}

// Polls `ev` on a global queue from `*t0` (once set) until it reads at least `value` or 3 s pass:
// `*seen` gets the time it first did, 0 where it never did.
static void poll_event(dispatch_group_t g, id<MTLSharedEvent> ev, uint64_t value,
                       volatile double *t0, volatile double *seen)
{
	dispatch_group_async(g, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
		while (*t0 == 0)
			;
		while (ev.signaledValue < value && oracle_now_ms() - *t0 < 3000)
			;
		*seen = ev.signaledValue >= value ? oracle_now_ms() : 0;
	});
}

static void print_status(const char *what, id<MTLCommandBuffer> cb)
{
	printf("    %s: status %s, error %s\n", what, status_name(cb.status),
	       cb.error ? [NSString stringWithFormat:@"%@ %ld %@", cb.error.domain, (long)cb.error.code,
	                                             cb.error.localizedDescription]
	                      .UTF8String
	                : "(none)");
}

static void case_schedsig(void)
{
	gSchedSel = sel_registerName("encodeSignalEventScheduled:value:");
	gSlow = cpipe("slow_write");
	if (!gSlow)
		return;
	id<MTLBuffer> big = [D newBufferWithLength:N * 4 options:MTLResourceStorageModeShared];
	id<MTLBuffer> small = [D newBufferWithLength:4096 options:MTLResourceStorageModeShared];
	MTLSharedEventListener *listener = [[MTLSharedEventListener alloc] init];

	printf("case schedsig: 1. who answers encodeSignalEventScheduled:value:\n");
	{
		id<MTLCommandBuffer> cb = [Q commandBuffer];
		id<MTLCommandBuffer> ucb = [Q commandBufferWithUnretainedReferences];
		printf("    a command buffer (%s): responds %d, from %s; unretained (%s): responds %d\n",
		       object_getClassName(cb), (int)[(id)cb respondsToSelector:gSchedSel],
		       defining_class(cb, gSchedSel), object_getClassName(ucb),
		       (int)[(id)ucb respondsToSelector:gSchedSel]);
		SEL k = sel_registerName("_encodeIOGPUKernelSignalEventScheduledCommandArgs:value:");
		id<MTLEvent> pe = [D newEvent];
		id<MTLSharedEvent> se = [D newSharedEvent];
		printf("    a plain event (%s): kernel args from %s; a shared event (%s): from %s\n",
		       object_getClassName(pe), defining_class(pe, k), object_getClassName(se),
		       defining_class(se, k));
		if (![(id)cb respondsToSelector:gSchedSel])
			return;
	}

	// Calibrate one kernel near 100 ms.
	uint32_t iters = 2000;
	double gpu = 0;
	for (int k = 0; k < 6; k++) {
		oracle_commit_wait(long_cb(Q, big, iters), NULL, &gpu);
		if ((gpu > 75 && gpu < 150) || k == 5)
			break;
		iters = (uint32_t)(iters * (100.0 / (gpu > 0.5 ? gpu : 0.5)));
	}
	printf("  one kernel %u iterations = %.1f ms GPU\n", iters, gpu);

	static const char *const idle_names[] = {
	    "2. idle queue: scheduled signal 1, then a ~100 ms kernel",
	    "3. idle queue: a ~100 ms kernel, then scheduled signal 1",
	};
	for (int p = 0; p < 2; p++) {
		printf("  %s, %d runs (ms from the commit)\n", idle_names[p], SCHED_RUNS);
		double polled[SCHED_RUNS], heard[SCHED_RUNS], sched[SCHED_RUNS], start[SCHED_RUNS];
		double end[SCHED_RUNS], done[SCHED_RUNS];
		int in_handler = 0, before_start = 0, listener_first = 0;
		for (int k = 0; k < SCHED_RUNS; k++) {
			id<MTLSharedEvent> ev = [D newSharedEvent];
			id<MTLCommandBuffer> cb = [Q commandBuffer];
			if (p == 0)
				sched_signal(cb, ev, 1);
			enc_kernel(cb, big, iters);
			if (p == 1)
				sched_signal(cb, ev, 1);
			__block volatile double tl = 0, ts = 0, tc = 0, tp = 0, t0 = 0;
			__block volatile uint64_t at_sched = 0;
			[ev notifyListener:listener
			           atValue:1
			             block:^(id<MTLSharedEvent> _e, uint64_t _v) { tl = oracle_now_ms(); }];
			[cb addScheduledHandler:^(id<MTLCommandBuffer> _) {
				ts = oracle_now_ms();
				at_sched = ev.signaledValue;
			}];
			[cb addCompletedHandler:^(id<MTLCommandBuffer> _) { tc = oracle_now_ms(); }];
			dispatch_group_t g = dispatch_group_create();
			poll_event(g, ev, 1, &t0, &tp);
			double c = oracle_now_ms();
			t0 = c;
			[cb commit];
			[cb waitUntilCompleted];
			dispatch_group_wait(g, dispatch_time(DISPATCH_TIME_NOW, 4 * NSEC_PER_SEC));
			usleep(5000);
			polled[k] = tp ? tp - c : -1;
			heard[k] = tl ? tl - c : -1;
			sched[k] = ts - c;
			start[k] = (cb.GPUStartTime - c / 1e3) * 1e3;
			end[k] = (cb.GPUEndTime - c / 1e3) * 1e3;
			done[k] = tc - c;
			in_handler += at_sched >= 1;
			before_start += tp != 0 && polled[k] < start[k];
			listener_first += tl != 0 && tl <= ts;
			usleep(5000);
		}
		print_ms("signaledValue first reads 1 (polling)", polled, SCHED_RUNS);
		print_ms("the listener's block for 1 runs", heard, SCHED_RUNS);
		print_ms("the scheduled handler runs", sched, SCHED_RUNS);
		print_ms("GPUStartTime", start, SCHED_RUNS);
		print_ms("GPUEndTime", end, SCHED_RUNS);
		print_ms("the completed handler runs", done, SCHED_RUNS);
		printf("    the scheduled handler read 1: %d of %d; polling read 1 before GPUStartTime: %d of %d; "
		       "listener's block before the scheduled handler: %d of %d\n",
		       in_handler, SCHED_RUNS, before_start, SCHED_RUNS, listener_first, SCHED_RUNS);
	}

	printf("  4. a ~200 ms buffer A, then buffer B on the same queue: scheduled signal 1 (shared event), a "
	       "tiny blit; buffer C on a second queue waits for 1, a tiny blit; %d runs (ms from B's commit)\n",
	       SCHED_RUNS);
	id<MTLCommandQueue> q2 = [D newCommandQueue];
	{
		double polled[SCHED_RUNS], heard[SCHED_RUNS], sched[SCHED_RUNS], a_end[SCHED_RUNS];
		double b_start[SCHED_RUNS], b_done[SCHED_RUNS], c_done[SCHED_RUNS];
		int c_before_a = 0;
		for (int k = 0; k < SCHED_RUNS; k++) {
			id<MTLSharedEvent> ev = [D newSharedEvent];
			id<MTLCommandBuffer> a = long_cb(Q, big, iters * 2);
			id<MTLCommandBuffer> b = [Q commandBuffer];
			sched_signal(b, ev, 1);
			enc_tiny(b, small);
			id<MTLCommandBuffer> cq = [q2 commandBuffer];
			[cq encodeWaitForEvent:ev value:1];
			enc_tiny(cq, small);
			__block volatile double tl = 0, ts = 0, tb = 0, tcq = 0, tp = 0, t0 = 0;
			[ev notifyListener:listener
			           atValue:1
			             block:^(id<MTLSharedEvent> _e, uint64_t _v) { tl = oracle_now_ms(); }];
			[b addScheduledHandler:^(id<MTLCommandBuffer> _) { ts = oracle_now_ms(); }];
			[b addCompletedHandler:^(id<MTLCommandBuffer> _) { tb = oracle_now_ms(); }];
			[cq addCompletedHandler:^(id<MTLCommandBuffer> _) { tcq = oracle_now_ms(); }];
			dispatch_group_t g = dispatch_group_create();
			[a commit];
			poll_event(g, ev, 1, &t0, &tp);
			double c = oracle_now_ms();
			t0 = c;
			[b commit];
			[cq commit];
			[b waitUntilCompleted];
			[cq waitUntilCompleted];
			dispatch_group_wait(g, dispatch_time(DISPATCH_TIME_NOW, 4 * NSEC_PER_SEC));
			usleep(5000);
			polled[k] = tp ? tp - c : -1;
			heard[k] = tl ? tl - c : -1;
			sched[k] = ts - c;
			a_end[k] = (a.GPUEndTime - c / 1e3) * 1e3;
			b_start[k] = (b.GPUStartTime - c / 1e3) * 1e3;
			b_done[k] = tb - c;
			c_done[k] = tcq - c;
			c_before_a += cq.GPUEndTime < a.GPUEndTime;
			usleep(5000);
		}
		print_ms("signaledValue first reads 1 (polling)", polled, SCHED_RUNS);
		print_ms("the listener's block for 1 runs", heard, SCHED_RUNS);
		print_ms("B's scheduled handler runs", sched, SCHED_RUNS);
		print_ms("A's GPUEndTime", a_end, SCHED_RUNS);
		print_ms("B's GPUStartTime", b_start, SCHED_RUNS);
		print_ms("B's completed handler runs", b_done, SCHED_RUNS);
		print_ms("C's completed handler runs", c_done, SCHED_RUNS);
		printf("    C's GPU work ended before A's: %d of %d\n", c_before_a, SCHED_RUNS);
	}

	printf("  5. value: a shared event at 5, one buffer each (the value once it completed)\n");
	{
		static const char *const names[] = {
		    "scheduled signal 3", "GPU signal 3 (encodeSignalEvent)", "CPU set 3 (setSignaledValue)",
		    "scheduled signal 7", "scheduled signals 6 then 8", "scheduled signals 8 then 6",
		};
		for (int s = 0; s < 6; s++) {
			id<MTLSharedEvent> ev = [D newSharedEvent];
			ev.signaledValue = 5;
			if (s == 2) {
				ev.signaledValue = 3;
				printf("    %-40s -> %llu\n", names[s], (unsigned long long)ev.signaledValue);
				continue;
			}
			id<MTLCommandBuffer> cb = [Q commandBuffer];
			if (s == 0)
				sched_signal(cb, ev, 3);
			else if (s == 1)
				[cb encodeSignalEvent:ev value:3];
			else if (s == 3)
				sched_signal(cb, ev, 7);
			else if (s == 4) {
				sched_signal(cb, ev, 6);
				sched_signal(cb, ev, 8);
			} else {
				sched_signal(cb, ev, 8);
				sched_signal(cb, ev, 6);
			}
			enc_tiny(cb, small);
			[cb commit];
			[cb waitUntilCompleted];
			usleep(2000);
			printf("    %-40s -> %llu\n", names[s], (unsigned long long)ev.signaledValue);
		}
	}

	printf("  6. order against the buffer's GPU signals, %d runs each (ms from the commit)\n", SCHED_RUNS);
	{
		static const char *const names[] = {
		    "a. ~50 ms kernel, GPU signal 2, scheduled signal 1",
		    "b. scheduled signal 2, ~50 ms kernel, GPU signal 1",
		};
		for (int s = 0; s < 2; s++) {
			double h1[SCHED_RUNS], h2[SCHED_RUNS], sched[SCHED_RUNS], end[SCHED_RUNS];
			uint64_t final[SCHED_RUNS], at_sched[SCHED_RUNS];
			for (int k = 0; k < SCHED_RUNS; k++) {
				id<MTLSharedEvent> ev = [D newSharedEvent];
				id<MTLCommandBuffer> cb = [Q commandBuffer];
				if (s == 0) {
					enc_kernel(cb, big, iters / 2);
					[cb encodeSignalEvent:ev value:2];
					sched_signal(cb, ev, 1);
				} else {
					sched_signal(cb, ev, 2);
					enc_kernel(cb, big, iters / 2);
					[cb encodeSignalEvent:ev value:1];
				}
				__block volatile double t1 = 0, t2 = 0, ts = 0;
				__block volatile uint64_t vs = 0;
				[ev notifyListener:listener
				           atValue:1
				             block:^(id<MTLSharedEvent> _e, uint64_t _v) { t1 = oracle_now_ms(); }];
				[ev notifyListener:listener
				           atValue:2
				             block:^(id<MTLSharedEvent> _e, uint64_t _v) { t2 = oracle_now_ms(); }];
				[cb addScheduledHandler:^(id<MTLCommandBuffer> _) {
					ts = oracle_now_ms();
					vs = ev.signaledValue;
				}];
				double c = oracle_now_ms();
				[cb commit];
				[cb waitUntilCompleted];
				usleep(10000);
				h1[k] = t1 ? t1 - c : -1;
				h2[k] = t2 ? t2 - c : -1;
				sched[k] = ts - c;
				end[k] = (cb.GPUEndTime - c / 1e3) * 1e3;
				final[k] = ev.signaledValue;
				at_sched[k] = vs;
				usleep(5000);
			}
			printf("    %s:\n", names[s]);
			print_ms("the listener's block for 1 runs", h1, SCHED_RUNS);
			print_ms("the listener's block for 2 runs", h2, SCHED_RUNS);
			print_ms("the scheduled handler runs", sched, SCHED_RUNS);
			print_ms("GPUEndTime", end, SCHED_RUNS);
			printf("    the value in the scheduled handler:");
			for (int k = 0; k < SCHED_RUNS; k++)
				printf(" %llu", (unsigned long long)at_sched[k]);
			printf("; at the end:");
			for (int k = 0; k < SCHED_RUNS; k++)
				printf(" %llu", (unsigned long long)final[k]);
			printf("\n");
		}
	}

	printf("  7. against the buffer's waits on another shared event W, which the CPU sets to 1 at 200 ms; "
	       "3 runs each (ms from the commit)\n");
	{
		static const char *const names[] = {
		    "a. wait W 1, scheduled signal 1, tiny blit",
		    "b. scheduled signal 1, wait W 1, tiny blit",
		    "c. ~50 ms kernel, wait W 1, scheduled signal 1, tiny blit",
		    "d. scheduled signal 1, ~50 ms kernel, wait W 1, tiny blit",
		};
		for (int s = 0; s < 4; s++) {
			printf("    %s:", names[s]);
			for (int k = 0; k < 3; k++) {
				id<MTLSharedEvent> ev = [D newSharedEvent], w = [D newSharedEvent];
				id<MTLCommandBuffer> cb = [Q commandBuffer];
				if (s == 0) {
					[cb encodeWaitForEvent:w value:1];
					sched_signal(cb, ev, 1);
				} else if (s == 1) {
					sched_signal(cb, ev, 1);
					[cb encodeWaitForEvent:w value:1];
				} else if (s == 2) {
					enc_kernel(cb, big, iters / 2);
					[cb encodeWaitForEvent:w value:1];
					sched_signal(cb, ev, 1);
				} else {
					sched_signal(cb, ev, 1);
					enc_kernel(cb, big, iters / 2);
					[cb encodeWaitForEvent:w value:1];
				}
				enc_tiny(cb, small);
				__block volatile double tp = 0, t0 = 0, ts = 0;
				[cb addScheduledHandler:^(id<MTLCommandBuffer> _) { ts = oracle_now_ms(); }];
				dispatch_group_t g = dispatch_group_create();
				poll_event(g, ev, 1, &t0, &tp);
				double c = oracle_now_ms();
				t0 = c;
				[cb commit];
				usleep(200000);
				MTLCommandBufferStatus st = cb.status;
				double set = oracle_now_ms() - c;
				w.signaledValue = 1;
				[cb waitUntilCompleted];
				dispatch_group_wait(g, dispatch_time(DISPATCH_TIME_NOW, 4 * NSEC_PER_SEC));
				printf(" [status at 200 ms %s; set at %.1f; 1 read at %.2f; scheduled handler %.2f; end %.2f; %s]",
				       status_name(st), set, tp ? tp - c : -1, ts ? ts - c : -1,
				       (cb.GPUEndTime - c / 1e3) * 1e3, status_name(cb.status));
				usleep(5000);
			}
			printf("\n");
		}
	}

	printf("    e. buffer A waits W 1, tiny blit; buffer B behind it on the queue: scheduled signal 1, tiny blit "
	       "(ms from B's commit):");
	for (int k = 0; k < 3; k++) {
		id<MTLSharedEvent> ev = [D newSharedEvent], w = [D newSharedEvent];
		id<MTLCommandBuffer> a = [Q commandBuffer];
		[a encodeWaitForEvent:w value:1];
		enc_tiny(a, small);
		id<MTLCommandBuffer> b = [Q commandBuffer];
		sched_signal(b, ev, 1);
		enc_tiny(b, small);
		__block volatile double tp = 0, t0 = 0;
		dispatch_group_t g = dispatch_group_create();
		poll_event(g, ev, 1, &t0, &tp);
		[a commit];
		double c = oracle_now_ms();
		t0 = c;
		[b commit];
		usleep(200000);
		double set = oracle_now_ms() - c;
		w.signaledValue = 1;
		[b waitUntilCompleted];
		dispatch_group_wait(g, dispatch_time(DISPATCH_TIME_NOW, 4 * NSEC_PER_SEC));
		printf(" [set at %.1f; 1 read at %.2f; %s]", set, tp ? tp - c : -1, status_name(b.status));
		usleep(5000);
	}
	printf("\n");

	printf("  8. the same event: a scheduled signal and a wait on it in one buffer, no CPU set (up to 8 s)\n");
	{
		static const char *const names[] = {
		    "a. wait 1, scheduled signal 1, tiny blit",
		    "b. scheduled signal 1, wait 1, tiny blit",
		};
		for (int s = 0; s < 2; s++) {
			id<MTLSharedEvent> ev = [D newSharedEvent];
			id<MTLCommandBuffer> cb = [Q commandBuffer];
			if (s == 0) {
				[cb encodeWaitForEvent:ev value:1];
				sched_signal(cb, ev, 1);
			} else {
				sched_signal(cb, ev, 1);
				[cb encodeWaitForEvent:ev value:1];
			}
			enc_tiny(cb, small);
			dispatch_semaphore_t fin = dispatch_semaphore_create(0);
			__block volatile double tc = 0, ts = 0;
			[cb addScheduledHandler:^(id<MTLCommandBuffer> _) { ts = oracle_now_ms(); }];
			[cb addCompletedHandler:^(id<MTLCommandBuffer> _) {
				tc = oracle_now_ms();
				dispatch_semaphore_signal(fin);
			}];
			double c = oracle_now_ms();
			[cb commit];
			long timed_out =
			    dispatch_semaphore_wait(fin, dispatch_time(DISPATCH_TIME_NOW, 8 * NSEC_PER_SEC));
			printf("    %s: value %llu; scheduled handler %.2f ms; completed handler %.2f ms%s\n",
			       names[s], (unsigned long long)ev.signaledValue, ts ? ts - c : -1, tc ? tc - c : -1,
			       timed_out ? " (not within 8 s)" : "");
			print_status("  the buffer", cb);
			if (timed_out) {
				ev.signaledValue = 1;
				[cb waitUntilCompleted];
				print_status("  after a CPU set of 1", cb);
			}
		}
	}

	printf("  9. a listener's queue: the block's queue label for a listener of its own queue, and the "
	       "default listener's\n");
	{
		dispatch_queue_t own = dispatch_queue_create("oracle.listener", DISPATCH_QUEUE_SERIAL);
		MTLSharedEventListener *mine = [[MTLSharedEventListener alloc] initWithDispatchQueue:own];
		for (int s = 0; s < 2; s++) {
			id<MTLSharedEvent> ev = [D newSharedEvent];
			id<MTLCommandBuffer> cb = [Q commandBuffer];
			sched_signal(cb, ev, 1);
			enc_tiny(cb, small);
			static char label[64];
			label[0] = 0;
			dispatch_semaphore_t got = dispatch_semaphore_create(0);
			[ev notifyListener:s ? mine : listener
			           atValue:1
			             block:^(id<MTLSharedEvent> _e, uint64_t _v) {
				             const char *q = dispatch_queue_get_label(DISPATCH_CURRENT_QUEUE_LABEL);
				             snprintf(label, sizeof label, "%s", q ? q : "");
				             dispatch_semaphore_signal(got);
			             }];
			[cb commit];
			[cb waitUntilCompleted];
			dispatch_semaphore_wait(got, dispatch_time(DISPATCH_TIME_NOW, 2 * NSEC_PER_SEC));
			printf("    %s listener: '%s'\n", s ? "its own queue's" : "the default", label);
		}
	}
}

// Misuse, each in a child process of its own: the call with an encoder open, after the commit,
// with a nil event (on a buffer that retains its references and on one that does not) and with a
// plain event (newEvent). Prints a line after the call, so a line missing says it ended the
// process.
static void case_schedsig_misuse(const char *which)
{
	gSchedSel = sel_registerName("encodeSignalEventScheduled:value:");
	id<MTLBuffer> small = [D newBufferWithLength:4096 options:MTLResourceStorageModeShared];
	id<MTLSharedEvent> ev = [D newSharedEvent];
	id<MTLCommandBuffer> cb = !strcmp(which, "schedsig_nil_unretained") ||
	                                  !strcmp(which, "schedsig_plain_unretained")
	                              ? [Q commandBufferWithUnretainedReferences]
	                              : [Q commandBuffer];
	if (![(id)cb respondsToSelector:gSchedSel]) {
		printf("case %s: n/r\n", which);
		return;
	}
	if (!strcmp(which, "schedsig_open")) {
		id<MTLBlitCommandEncoder> e = [cb blitCommandEncoder];
		[e fillBuffer:small range:NSMakeRange(0, 256) value:1];
		sched_signal(cb, ev, 1);
		printf("case %s: the call returned with a blit encoder open\n", which);
		[cb commit];
		[cb waitUntilCompleted];
	} else if (!strcmp(which, "schedsig_late")) {
		enc_tiny(cb, small);
		[cb commit];
		sched_signal(cb, ev, 1);
		printf("case %s: the call returned after the commit\n", which);
		[cb waitUntilCompleted];
	} else if (!strncmp(which, "schedsig_plain", 14)) {
		id<MTLEvent> plain = [D newEvent];
		sched_signal(cb, plain, 1);
		printf("case %s: the call returned for a plain event\n", which);
		enc_tiny(cb, small);
		[cb commit];
		[cb waitUntilCompleted];
	} else {
		sched_signal(cb, nil, 1);
		printf("case %s: the call returned for a nil event\n", which);
		enc_tiny(cb, small);
		[cb commit];
		[cb waitUntilCompleted];
	}
	usleep(10000);
	print_status("the buffer", cb);
	printf("    the event's value: %llu\n", (unsigned long long)ev.signaledValue);
}

enum { GPUTIMES_RUNS = 7, GPUTIMES_W = 1920, GPUTIMES_H = 1080 };

static id<MTLRenderPipelineState> gBlend;
static id<MTLTexture> gTarget;

// One render pass over gTarget: a clear, then 6 blended full-target quads.
static void enc_pass(id<MTLCommandBuffer> cb)
{
	MTLRenderPassDescriptor *rp = [MTLRenderPassDescriptor renderPassDescriptor];
	rp.colorAttachments[0].texture = gTarget;
	rp.colorAttachments[0].loadAction = MTLLoadActionClear;
	rp.colorAttachments[0].storeAction = MTLStoreActionStore;
	rp.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 1);
	id<MTLRenderCommandEncoder> e = [cb renderCommandEncoderWithDescriptor:rp];
	[e setRenderPipelineState:gBlend];
	float rect[4] = {-1, -1, 1, 1}, z = 0, color[4] = {0.2f, 0.4f, 0.6f, 0.3f};
	[e setVertexBytes:rect length:sizeof rect atIndex:0];
	[e setVertexBytes:&z length:sizeof z atIndex:1];
	[e setFragmentBytes:color length:sizeof color atIndex:0];
	for (int i = 0; i < 6; i++)
		[e drawPrimitives:MTLPrimitiveTypeTriangleStrip vertexStart:0 vertexCount:4];
	[e endEncoding];
}

// K encoders of one kind in `cb`: compute (slow_write of `iters`) or a render pass; the CPU
// sleeps `sleep_us` between them.
static void enc_k(id<MTLCommandBuffer> cb, int render, int k, id<MTLBuffer> b, uint32_t iters,
                  useconds_t sleep_us)
{
	for (int i = 0; i < k; i++) {
		if (i && sleep_us)
			usleep(sleep_us);
		if (render) {
			enc_pass(cb);
		} else {
			id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
			enc_slow(e, b, iters);
			[e endEncoding];
		}
	}
}

static void gputimes_row(const char *what, int render, int k, id<MTLBuffer> b, uint32_t iters,
                         useconds_t sleep_us)
{
	double span[GPUTIMES_RUNS], start[GPUTIMES_RUNS], handler[GPUTIMES_RUNS];
	double sum[GPUTIMES_RUNS], aspan[GPUTIMES_RUNS], astart[GPUTIMES_RUNS];
	for (int r = 0; r < GPUTIMES_RUNS; r++) {
		id<MTLCommandBuffer> cb = [Q commandBuffer];
		enc_k(cb, render, k, b, iters, sleep_us);
		__block volatile double done = 0;
		[cb addCompletedHandler:^(id<MTLCommandBuffer> _) { done = oracle_now_ms(); }];
		double c0 = oracle_now_ms();
		[cb commit];
		[cb waitUntilCompleted];
		while (done == 0)
			usleep(100);
		span[r] = (cb.GPUEndTime - cb.GPUStartTime) * 1e3;
		start[r] = cb.GPUStartTime * 1e3 - c0;
		handler[r] = done - cb.GPUEndTime * 1e3;
		if (cb.status != MTLCommandBufferStatusCompleted)
			printf("  ! whole status %ld\n", (long)cb.status);
		if (sleep_us) {
			sum[r] = aspan[r] = astart[r] = 0;
			continue;
		}
		id<MTLCommandBuffer> cbs[32];
		for (int i = 0; i < k; i++) {
			cbs[i] = [Q commandBuffer];
			enc_k(cbs[i], render, 1, b, iters, 0);
		}
		double a0 = oracle_now_ms();
		for (int i = 0; i < k; i++)
			[cbs[i] commit];
		[cbs[k - 1] waitUntilCompleted];
		sum[r] = 0;
		for (int i = 0; i < k; i++) {
			[cbs[i] waitUntilCompleted];
			sum[r] += (cbs[i].GPUEndTime - cbs[i].GPUStartTime) * 1e3;
		}
		aspan[r] = (cbs[k - 1].GPUEndTime - cbs[0].GPUStartTime) * 1e3;
		astart[r] = cbs[0].GPUStartTime * 1e3 - a0;
	}
	sort_ms(span, GPUTIMES_RUNS);
	sort_ms(start, GPUTIMES_RUNS);
	sort_ms(handler, GPUTIMES_RUNS);
	sort_ms(sum, GPUTIMES_RUNS);
	sort_ms(aspan, GPUTIMES_RUNS);
	sort_ms(astart, GPUTIMES_RUNS);
	int m = GPUTIMES_RUNS / 2;
	printf("  %-34s K %2d  whole: span %8.3f (min %8.3f max %8.3f) start %+7.3f handler %+7.3f  "
	       "apart: summed %8.3f span %8.3f start %+7.3f\n",
	       what, k, span[m], span[0], span[GPUTIMES_RUNS - 1], start[m], handler[m], sum[m], aspan[m],
	       astart[m]);
}

static void case_gputimes(void)
{
	gSlow = cpipe("slow_write");
	if (!gSlow)
		return;
	MTLRenderPipelineDescriptor *pd = [MTLRenderPipelineDescriptor new];
	pd.vertexFunction = [L newFunctionWithName:@"quad_vs"];
	pd.fragmentFunction = [L newFunctionWithName:@"solid_fs"];
	pd.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
	pd.colorAttachments[0].blendingEnabled = YES;
	pd.colorAttachments[0].sourceRGBBlendFactor = MTLBlendFactorSourceAlpha;
	pd.colorAttachments[0].destinationRGBBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
	NSError *err = nil;
	gBlend = [D newRenderPipelineStateWithDescriptor:pd error:&err];
	if (!gBlend) {
		printf("  ! render pipeline: %s\n", err.description.UTF8String);
		return;
	}
	MTLTextureDescriptor *td = [MTLTextureDescriptor
	    texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
	                                 width:GPUTIMES_W
	                                height:GPUTIMES_H
	                             mipmapped:NO];
	td.usage = MTLTextureUsageRenderTarget;
	td.storageMode = MTLStorageModePrivate;
	gTarget = [D newTextureWithDescriptor:td];
	id<MTLBuffer> b = [D newBufferWithLength:N * 4 options:MTLResourceStorageModePrivate];
	// Calibrate one encoder near 0.5 ms
	uint32_t iters = 40;
	double gpu = 0;
	for (int k = 0; k < 8; k++) {
		id<MTLCommandBuffer> cb = [Q commandBuffer];
		enc_k(cb, 0, 1, b, iters, 0);
		oracle_commit_wait(cb, NULL, &gpu);
		if ((gpu > 0.4 && gpu < 0.65) || k == 7)
			break;
		iters = (uint32_t)(iters * (0.5 / (gpu > 0.01 ? gpu : 0.01)));
		if (iters < 1)
			iters = 1;
	}
	id<MTLCommandBuffer> one = [Q commandBuffer];
	enc_k(one, 1, 1, b, iters, 0);
	double pass = 0;
	oracle_commit_wait(one, NULL, &pass);
	printf("case gputimes: compute encoder %u iterations = %.3f ms GPU; render pass = %.3f ms GPU\n",
	       iters, gpu, pass);
	int ks[] = {1, 8, 32};
	for (int i = 0; i < 3; i++)
		gputimes_row("compute", 0, ks[i], b, iters, 0);
	for (int i = 0; i < 3; i++)
		gputimes_row("render", 1, ks[i], b, iters, 0);
	gputimes_row("compute, CPU sleeps 1 ms between", 0, 8, b, iters, 1000);
}

static int run_case(int argc, char **argv)
{
	L = oracle_library(D, gArgv0, "oracle");
	if (!L)
		return 1;
	if (!strcmp(argv[0], "hazards"))
		case_hazards();
	else if (!strcmp(argv[0], "timing"))
		case_timing();
	else if (!strcmp(argv[0], "submitted"))
		case_submitted();
	else if (!strcmp(argv[0], "events"))
		case_events();
	else if (!strcmp(argv[0], "gputimes"))
		case_gputimes();
	else if (!strcmp(argv[0], "schedsig"))
		case_schedsig();
	else if (!strncmp(argv[0], "schedsig_", 9))
		case_schedsig_misuse(argv[0]);
	else
		return 2;
	return 0;
}

int main(int argc, char **argv)
{
	setvbuf(stdout, NULL, _IOLBF, 0);
	gArgv0 = argv[0];
	if (argc < 2) {
		fprintf(stderr, "usage: %s DEVICE [CASE]\n", argv[0]);
		return 2;
	}
	D = oracle_device(argv[1]);
	if (argc > 2 && !strcmp(argv[2], "--case")) {
		Q = [D newCommandQueue];
		return run_case(argc - 3, argv + 3);
	}
	oracle_header("oracle_cmdbuf", D);
	D = nil;
	static const char *const all[] = {"hazards",  "timing",        "submitted",     "events",
	                                  "gputimes", "schedsig",      "schedsig_open", "schedsig_late",
	                                  "schedsig_nil", "schedsig_nil_unretained", "schedsig_plain",
	                                  "schedsig_plain_unretained"};
	for (size_t i = 0; i < sizeof all / sizeof all[0]; i++) {
		if (argc > 2 && strcmp(argv[2], all[i]))
			continue;
		char *args[] = {argv[0], argv[1], "--case", (char *)all[i], NULL};
		pid_t pid;
		if (posix_spawn(&pid, argv[0], NULL, NULL, args, environ))
			return 1;
		int st;
		waitpid(pid, &st, 0);
		if (WIFSIGNALED(st))
			printf("  => case %s died on signal %d\n", all[i], WTERMSIG(st));
		else if (WEXITSTATUS(st))
			printf("  => case %s exited %d\n", all[i], WEXITSTATUS(st));
	}
	return 0;
}
