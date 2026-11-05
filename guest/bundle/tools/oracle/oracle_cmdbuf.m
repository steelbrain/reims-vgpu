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
// Build: see README.md.
#import "oracle_common.h"
#include <dispatch/dispatch.h>
#include <objc/message.h>
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
	static const char *const all[] = {"hazards", "timing", "submitted", "events"};
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
