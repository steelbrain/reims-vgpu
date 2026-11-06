// submit_probe: the fixed cost and the latency of a command buffer on a Metal device, and of a
// shared event's signal, to compare a guest's GPU lane with real Macs. No shaders: a "tiny"
// command buffer is one blit filling 16 bytes of a Shared buffer, an "empty" one has no encoder.
//   submit_probe [-d NAME] [-t SECONDS] [-g MICROSECONDS] [-n THREADS] [-q QUEUES] [-s SECONDS] [-v] [-l]
//                [CASE...]
//     -d NAME   the first of MTLCopyAllDevices whose name holds NAME; the default is
//               MTLCreateSystemDefaultDevice (on a dual-GPU MacBook pass -d, which keeps the
//               display where it is).
//     -t SEC    how long each case runs after its warm-up (default 3).
//     -g US     the idle gap before each command buffer of the .idle cases and before each
//               signal of the event cases (default 2000).
//     -n MAX    the most threads of the mt cases (default 8; they run 2, 4 and 8, up to MAX).
//     -q N      makes N more queues first, each with one command buffer run through it, and keeps
//               them idle for the whole run: what a device with many quiet queues costs the cases.
//               It prints how many it made; newCommandQueue answering nil ends the making there.
//     -s SEC    after making the -q queues, holds them SEC seconds before the cases run, so several
//               processes can hold quiet queues on one device while another measures.
//     -v        two more lines a case: "# top" with the 8 largest samples of its first
//               distribution (each with when it was taken, seconds into the case, and the
//               wall-clock time), and "# hist" with its log2 histogram in microseconds.
//     -l        lists the devices and exits.
//     CASE      runs only the cases whose name holds one of these; with none, every case.
// The cases (one thread and one queue unless named):
//   rate.commit.empty, rate.commit.tiny   commit back to back, waiting only for the last: cb_s,
//              and the committing thread's time a buffer (make, encode, commit), commit_us.
//   rate.wait.empty, rate.wait.tiny       commit, then waitUntilCompleted, one at a time: cb_s,
//              and ret_us, commit to waitUntilCompleted's return.
//   rate.handler.tiny  commit, then wait on a semaphore its completed handler signals:
//              handler_us, commit to the handler, and wake_us, commit to the waiter's wake.
//   lat.tiny.busy, lat.tiny.idle  as rate.wait.tiny with a scheduled and a completed handler,
//              back to back or after an idle gap (-g): commit_us (the commit call), sched_us
//              and done_us (commit to each handler), ret_us, and from the buffer's own times
//              gpu_start_us (commit to GPUStartTime), gpu_us (GPUStartTime to GPUEndTime) and
//              deliver_us (GPUEndTime to the completed handler).
//   submitted.empty.idle, submitted.tiny.idle  commitAndWaitUntilSubmitted (the SPI WindowServer's
//              display pipe calls) of a lone buffer after an idle gap (-g): ret_us (the call) and
//              sched_us (its start to the buffer's scheduled handler).
//   submitted.tiny.behind  the same call of a tiny buffer committed behind a buffer of a few
//              milliseconds (four fills of 64 MiB of Private memory) on its queue: ret_us; and,
//              from the busy buffer's own times, busy_start_us (its commit to its GPUStartTime)
//              and busy_gpu_us (GPUStartTime to GPUEndTime), and round_us, a round's whole time.
//   mt.commit.qN       N threads, each with its own queue, committing back to back: cb_s.
//   mt.wait.qN         N threads, each with its own queue, each committing and waiting: cb_s,
//              ret_us over all of them.
//   mt.wait.sharedN    the same on one queue the N threads share.
//   event.cpu_cpu      another thread blocks in waitUntilSignaledValue:timeoutMS:, and this one
//              sets signaledValue after the gap: wake_us, the set to the waiter's wake.
//   event.gpu_cpu      a tiny buffer signals the event (encodeSignalEvent:value:) while another
//              thread waits for the value on the CPU: wake_us (commit to the waiter's wake)
//              and done_us (commit to the buffer's completed handler).
//   event.listener     the same, with a notifyListener: block in place of the waiting thread:
//              wake_us, commit to the block.
//   event.cpu_gpu      a tiny buffer that waits for the event (encodeWaitForEvent:value:) is
//              committed; after the gap the CPU sets the value: done_us, the set to the
//              buffer's completed handler.
//   event.gpu_gpu      a buffer on queue 2 waits for the value a tiny buffer on queue 1 signals,
//              committed after the gap: done_us, queue 1's commit to queue 2's buffer's
//              completed handler, and own_us, to queue 1's own.
// Every case also gives cpu_us and sys_us: the process's CPU time (user plus system, and the
// system part), over the case, per command buffer (per signal for event.cpu_cpu).
// A distribution prints as name_us=p50:..,p90:..,p99:..,p999:..,max:..,mean:..,n:.., in
// microseconds. One line a case: "submit_probe: NAME OK ...", or "NAME FAIL why" / "SKIP why",
// after a "# start NAME" line; a completion 10 s late prints HANG and exits 3.
// Build: make submit-probe (build/submit_probe-x86_64, a guest's or an Intel Mac's); ARCH=arm64
// for an Apple-silicon Mac.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <mach/mach_time.h>
#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/sysctl.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

#pragma clang diagnostic ignored "-Wunguarded-availability-new"

static id<MTLDevice> gDevice;
static NSArray<NSString *> *gPatterns;
static double gSeconds = 3;
static double gGapUs = 2000;
static int gMaxThreads = 8;
static int gIdleQueues;
static double gHoldSeconds;
static BOOL gVerbose;
static double gNsPerTick;

// The watchdog's view: completions so far, and whether a case runs.
static _Atomic long gProgress;
static _Atomic int gBusy;
static const char *_Atomic gCaseName;

static double nowUs(void)
{
	return (double)mach_absolute_time() * gNsPerTick / 1e3;
}

static void progress(void)
{
	atomic_fetch_add(&gProgress, 1);
}

static void *watchdog(void *arg)
{
	long last = -1;
	int still = 0;
	for (;;) {
		sleep(1);
		long p = atomic_load(&gProgress);
		if (atomic_load(&gBusy) && p == last) {
			if (++still >= 10) {
				printf("HANG: %s made no progress for 10 s (%ld completions before)\n",
				       atomic_load(&gCaseName), p);
				fflush(stdout);
				_exit(3);
			}
		} else {
			still = 0;
		}
		last = p;
	}
	return NULL;
}

// MARK: - Samples

typedef struct {
	double *v;  // microseconds
	double *at; // when, microseconds since the case began
	size_t n, cap;
} Samples;

static void sAdd(Samples *s, double value, double at)
{
	if (s->n == s->cap) {
		s->cap = s->cap ? s->cap * 2 : 1024;
		s->v = realloc(s->v, s->cap * sizeof(double));
		s->at = realloc(s->at, s->cap * sizeof(double));
	}
	s->v[s->n] = value;
	s->at[s->n] = at;
	s->n++;
}

static void sMerge(Samples *into, const Samples *from)
{
	for (size_t i = 0; i < from->n; i++)
		sAdd(into, from->v[i], from->at[i]);
}

static void sFree(Samples *s)
{
	free(s->v);
	free(s->at);
	memset(s, 0, sizeof(*s));
}

static int cmpDouble(const void *a, const void *b)
{
	double x = *(const double *)a, y = *(const double *)b;
	return x < y ? -1 : x > y;
}

static double quantile(const double *sorted, size_t n, double q)
{
	if (n == 0)
		return 0;
	size_t i = (size_t)ceil(q * (double)n);
	if (i > 0)
		i--;
	return sorted[i < n ? i : n - 1];
}

// "name_us=p50:..,p90:..,p99:..,p999:..,max:..,mean:..,n:..".
static NSString *sLine(const char *name, const Samples *s)
{
	if (s->n == 0)
		return [NSString stringWithFormat:@"%s_us=n:0", name];
	double *sorted = malloc(s->n * sizeof(double));
	memcpy(sorted, s->v, s->n * sizeof(double));
	qsort(sorted, s->n, sizeof(double), cmpDouble);
	double sum = 0;
	for (size_t i = 0; i < s->n; i++)
		sum += sorted[i];
	NSString *line = [NSString
	    stringWithFormat:@"%s_us=p50:%.1f,p90:%.1f,p99:%.1f,p999:%.1f,max:%.1f,mean:%.1f,n:%zu",
	                     name, quantile(sorted, s->n, 0.5), quantile(sorted, s->n, 0.9),
	                     quantile(sorted, s->n, 0.99), quantile(sorted, s->n, 0.999),
	                     sorted[s->n - 1], sum / (double)s->n, s->n];
	free(sorted);
	return line;
}

static double gCaseWall; // the wall clock at the case's start, seconds since the epoch

// -v: the largest samples, with when each was taken; and the log2 histogram.
static void sDetail(const char *name, const Samples *s)
{
	if (!gVerbose || s->n == 0)
		return;
	size_t *idx = malloc(s->n * sizeof(size_t));
	for (size_t i = 0; i < s->n; i++)
		idx[i] = i;
	size_t top = s->n < 8 ? s->n : 8;
	for (size_t k = 0; k < top; k++) {
		size_t best = k;
		for (size_t i = k + 1; i < s->n; i++)
			if (s->v[idx[i]] > s->v[idx[best]])
				best = i;
		size_t t = idx[k];
		idx[k] = idx[best];
		idx[best] = t;
	}
	printf("# top %s:", name);
	for (size_t k = 0; k < top; k++) {
		double at = s->at[idx[k]] / 1e6;
		time_t wall = (time_t)(gCaseWall + at);
		struct tm tm;
		localtime_r(&wall, &tm);
		double frac = fmod(gCaseWall + at, 1.0);
		printf(" %.0f@%.3fs(%02d:%02d:%02d.%03d)", s->v[idx[k]], at, tm.tm_hour, tm.tm_min,
		       tm.tm_sec, (int)(frac * 1000));
	}
	printf("\n# hist %s:", name);
	size_t buckets[24] = {0};
	for (size_t i = 0; i < s->n; i++) {
		int b = s->v[i] < 1 ? 0 : (int)floor(log2(s->v[i])) + 1;
		buckets[b < 23 ? b : 23]++;
	}
	for (int b = 0; b < 24; b++)
		if (buckets[b])
			printf(" <%.0f:%zu", pow(2, b), buckets[b]);
	printf("\n");
	free(idx);
}

// MARK: - The process's CPU time

typedef struct {
	double user, sys;
} Cpu;

static Cpu cpuNow(void)
{
	struct rusage r;
	getrusage(RUSAGE_SELF, &r);
	Cpu c = {r.ru_utime.tv_sec * 1e6 + r.ru_utime.tv_usec, r.ru_stime.tv_sec * 1e6 + r.ru_stime.tv_usec};
	return c;
}

static NSString *cpuLine(Cpu before, long count)
{
	Cpu after = cpuNow();
	double n = count > 0 ? (double)count : 1;
	return [NSString stringWithFormat:@"cpu_us=%.1f sys_us=%.1f",
	                                  (after.user - before.user + after.sys - before.sys) / n,
	                                  (after.sys - before.sys) / n];
}

// MARK: - Command buffers

static id<MTLCommandBuffer> makeBuffer(id<MTLCommandQueue> q, id<MTLBuffer> buf, BOOL tiny,
                                       long i)
{
	id<MTLCommandBuffer> cb = [q commandBuffer];
	if (tiny) {
		id<MTLBlitCommandEncoder> b = [cb blitCommandEncoder];
		[b fillBuffer:buf range:NSMakeRange(0, 16) value:(uint8_t)(i & 0xff)];
		[b endEncoding];
	}
	return cb;
}

static BOOL failed(id<MTLCommandBuffer> cb, NSString **why)
{
	if (cb.status == MTLCommandBufferStatusCompleted)
		return NO;
	*why = [NSString stringWithFormat:@"FAIL status %ld error %@", (long)cb.status,
	                                  cb.error.localizedDescription];
	return YES;
}

static id<MTLBuffer> newShared(void)
{
	return [gDevice newBufferWithLength:4096 options:MTLResourceStorageModeShared];
}

// A few buffers through the queue first: its rings, threads and the blit's path made.
static void warm(id<MTLCommandQueue> q, id<MTLBuffer> buf)
{
	for (int i = 0; i < 32; i++) {
		@autoreleasepool {
			id<MTLCommandBuffer> cb = makeBuffer(q, buf, YES, i);
			[cb commit];
			[cb waitUntilCompleted];
			progress();
		}
	}
}

// Sleeps `us` microseconds, the last of them spinning, so the gap is what was asked.
static void gap(double us)
{
	if (us <= 0)
		return;
	double end = nowUs() + us;
	if (us > 200)
		usleep((useconds_t)(us - 150));
	while (nowUs() < end) {
	}
}

// MARK: - The cases

static NSString *commitRate(BOOL tiny)
{
	id<MTLCommandQueue> q = [gDevice newCommandQueue];
	id<MTLBuffer> buf = newShared();
	warm(q, buf);
	Cpu cpu = cpuNow();
	double t0 = nowUs(), end = t0 + gSeconds * 1e6, loopEnd;
	long n = 0;
	id<MTLCommandBuffer> last = nil;
	for (;;) {
		@autoreleasepool {
			id<MTLCommandBuffer> cb = makeBuffer(q, buf, tiny, n);
			[cb commit];
			progress();
			n++;
			if (nowUs() >= end) {
				last = cb;
				break;
			}
		}
	}
	loopEnd = nowUs();
	[last waitUntilCompleted];
	double t1 = nowUs();
	NSString *why = nil;
	if (failed(last, &why))
		return why;
	return [NSString stringWithFormat:@"OK n=%ld cb_s=%.0f commit_us=%.2f %@", n,
	                                  n / ((t1 - t0) / 1e6), (loopEnd - t0) / n, cpuLine(cpu, n)];
}

static NSString *waitRate(BOOL tiny)
{
	id<MTLCommandQueue> q = [gDevice newCommandQueue];
	id<MTLBuffer> buf = newShared();
	warm(q, buf);
	Samples ret = {0};
	Cpu cpu = cpuNow();
	double t0 = nowUs(), end = t0 + gSeconds * 1e6;
	long n = 0;
	NSString *why = nil;
	while (nowUs() < end) {
		@autoreleasepool {
			id<MTLCommandBuffer> cb = makeBuffer(q, buf, tiny, n);
			double c = nowUs();
			[cb commit];
			[cb waitUntilCompleted];
			double r = nowUs();
			progress();
			if (failed(cb, &why))
				break;
			sAdd(&ret, r - c, c - t0);
			n++;
		}
	}
	double t1 = nowUs();
	if (why)
		return why;
	NSString *line = [NSString stringWithFormat:@"OK n=%ld cb_s=%.0f %@ %@", n, n / ((t1 - t0) / 1e6),
	                                            sLine("ret", &ret), cpuLine(cpu, n)];
	sDetail("ret", &ret);
	sFree(&ret);
	return line;
}

static NSString *handlerRate(void)
{
	id<MTLCommandQueue> q = [gDevice newCommandQueue];
	id<MTLBuffer> buf = newShared();
	warm(q, buf);
	Samples handler = {0}, wake = {0};
	dispatch_semaphore_t sem = dispatch_semaphore_create(0);
	Cpu cpu = cpuNow();
	double t0 = nowUs(), end = t0 + gSeconds * 1e6;
	long n = 0;
	NSString *why = nil;
	__block _Atomic double handled = 0;
	while (nowUs() < end) {
		@autoreleasepool {
			id<MTLCommandBuffer> cb = makeBuffer(q, buf, YES, n);
			[cb addCompletedHandler:^(id<MTLCommandBuffer> b) {
				atomic_store(&handled, nowUs());
				progress();
				dispatch_semaphore_signal(sem);
			}];
			double c = nowUs();
			[cb commit];
			dispatch_semaphore_wait(sem, DISPATCH_TIME_FOREVER);
			double w = nowUs();
			if (failed(cb, &why))
				break;
			sAdd(&handler, atomic_load(&handled) - c, c - t0);
			sAdd(&wake, w - c, c - t0);
			n++;
		}
	}
	double t1 = nowUs();
	if (why)
		return why;
	NSString *line = [NSString stringWithFormat:@"OK n=%ld cb_s=%.0f %@ %@ %@", n,
	                                            n / ((t1 - t0) / 1e6), sLine("handler", &handler),
	                                            sLine("wake", &wake), cpuLine(cpu, n)];
	sDetail("handler", &handler);
	sFree(&handler);
	sFree(&wake);
	return line;
}

typedef struct {
	_Atomic double scheduled, done;
} Stamps;

static NSString *latency(double gapUs)
{
	id<MTLCommandQueue> q = [gDevice newCommandQueue];
	id<MTLBuffer> buf = newShared();
	warm(q, buf);
	Samples commit = {0}, sched = {0}, done = {0}, ret = {0}, gpuStart = {0}, gpu = {0},
	        deliver = {0};
	Cpu cpu = cpuNow();
	double t0 = nowUs(), end = t0 + gSeconds * 1e6;
	long n = 0;
	NSString *why = nil;
	Stamps *st = calloc(1, sizeof(Stamps));
	while (nowUs() < end) {
		@autoreleasepool {
			gap(gapUs);
			atomic_store(&st->scheduled, 0);
			atomic_store(&st->done, 0);
			id<MTLCommandBuffer> cb = makeBuffer(q, buf, YES, n);
			[cb addScheduledHandler:^(id<MTLCommandBuffer> b) {
				atomic_store(&st->scheduled, nowUs());
			}];
			[cb addCompletedHandler:^(id<MTLCommandBuffer> b) {
				atomic_store(&st->done, nowUs());
				progress();
			}];
			double c = nowUs();
			[cb commit];
			double c1 = nowUs();
			[cb waitUntilCompleted];
			double r = nowUs();
			// Metal does not order the completed handler before waitUntilCompleted's return.
			while (atomic_load(&st->done) == 0 && nowUs() - r < 1e6) {
			}
			if (failed(cb, &why))
				break;
			double d = atomic_load(&st->done), s = atomic_load(&st->scheduled);
			double at = c - t0;
			sAdd(&commit, c1 - c, at);
			if (s > 0)
				sAdd(&sched, s - c, at);
			sAdd(&done, d - c, at);
			sAdd(&ret, r - c, at);
			if (cb.GPUStartTime > 0 && cb.GPUEndTime >= cb.GPUStartTime) {
				sAdd(&gpuStart, cb.GPUStartTime * 1e6 - c, at);
				sAdd(&gpu, (cb.GPUEndTime - cb.GPUStartTime) * 1e6, at);
				sAdd(&deliver, d - cb.GPUEndTime * 1e6, at);
			}
			n++;
		}
	}
	free(st);
	if (why)
		return why;
	NSString *line = [NSString
	    stringWithFormat:@"OK n=%ld %@ %@ %@ %@ %@ %@ %@ %@", n, sLine("done", &done),
	                     sLine("ret", &ret), sLine("sched", &sched), sLine("commit", &commit),
	                     sLine("gpu_start", &gpuStart), sLine("gpu", &gpu),
	                     sLine("deliver", &deliver), cpuLine(cpu, n)];
	sDetail("done", &done);
	Samples *all[] = {&commit, &sched, &done, &ret, &gpuStart, &gpu, &deliver};
	for (size_t i = 0; i < sizeof(all) / sizeof(all[0]); i++)
		sFree(all[i]);
	return line;
}

// The command-buffer SPI WindowServer's display pipe calls: commits, then returns once the buffer
// is submitted, not run.
@interface NSObject (SubmitProbeSubmitted)
- (BOOL)commitAndWaitUntilSubmitted;
@end

// commitAndWaitUntilSubmitted of a lone buffer (`tiny`, else empty), after the gap (-g) on an idle
// queue, or (`behind`) of a tiny one committed behind a buffer of a few milliseconds on its queue:
// ret_us, the call; sched_us, the call's start to the buffer's scheduled handler (lone only).
static NSString *submitted(BOOL tiny, BOOL behind)
{
	id<MTLCommandQueue> q = [gDevice newCommandQueue];
	id<MTLBuffer> buf = newShared();
	id<MTLBuffer> big = nil;
	if (behind)
		big = [gDevice newBufferWithLength:64u << 20 options:MTLResourceStorageModePrivate];
	warm(q, buf);
	Samples ret = {0}, sched = {0}, busyStart = {0}, busyGpu = {0}, round = {0};
	Cpu cpu = cpuNow();
	double t0 = nowUs(), end = t0 + gSeconds * 1e6;
	long n = 0;
	NSString *why = nil;
	Stamps *st = calloc(1, sizeof(Stamps));
	while (nowUs() < end) {
		@autoreleasepool {
			id<MTLCommandBuffer> busy = nil;
			double r0 = nowUs(), bc = 0;
			if (behind) {
				busy = [q commandBuffer];
				id<MTLBlitCommandEncoder> b = [busy blitCommandEncoder];
				for (int i = 0; i < 4; i++)
					[b fillBuffer:big range:NSMakeRange(0, big.length) value:(uint8_t)i];
				[b endEncoding];
				bc = nowUs();
				[busy commit];
			} else {
				gap(gGapUs);
			}
			atomic_store(&st->scheduled, 0);
			id<MTLCommandBuffer> cb = makeBuffer(q, buf, tiny || behind, n);
			if (!behind) {
				[cb addScheduledHandler:^(id<MTLCommandBuffer> b) {
					atomic_store(&st->scheduled, nowUs());
				}];
			}
			double c = nowUs();
			BOOL ok = [(id)cb commitAndWaitUntilSubmitted];
			double r = nowUs();
			[cb waitUntilCompleted];
			[busy waitUntilCompleted];
			progress();
			if (!ok) {
				why = @"FAIL commitAndWaitUntilSubmitted answered NO";
				break;
			}
			if (failed(cb, &why) || (busy != nil && failed(busy, &why)))
				break;
			sAdd(&ret, r - c, c - t0);
			double s = atomic_load(&st->scheduled);
			if (s > 0)
				sAdd(&sched, s - c, c - t0);
			if (busy != nil && busy.GPUStartTime > 0 && busy.GPUEndTime >= busy.GPUStartTime) {
				sAdd(&busyStart, busy.GPUStartTime * 1e6 - bc, c - t0);
				sAdd(&busyGpu, (busy.GPUEndTime - busy.GPUStartTime) * 1e6, c - t0);
			}
			sAdd(&round, nowUs() - r0, c - t0);
			n++;
		}
	}
	free(st);
	if (why)
		return why;
	NSString *line = behind ? [NSString stringWithFormat:@"OK n=%ld %@ %@ %@ %@ %@", n,
	                                                     sLine("ret", &ret),
	                                                     sLine("busy_start", &busyStart),
	                                                     sLine("busy_gpu", &busyGpu),
	                                                     sLine("round", &round), cpuLine(cpu, n)]
	                        : [NSString stringWithFormat:@"OK n=%ld %@ %@ %@", n, sLine("ret", &ret),
	                                                     sLine("sched", &sched), cpuLine(cpu, n)];
	sDetail("ret", &ret);
	sFree(&ret);
	sFree(&sched);
	sFree(&busyStart);
	sFree(&busyGpu);
	sFree(&round);
	return line;
}

// N threads, on their own queues or one they share, committing back to back (`wait` NO) or
// each waiting for its buffer (`wait` YES).
static NSString *threads(int count, BOOL shared, BOOL wait)
{
	id<MTLCommandQueue> one = [gDevice newCommandQueue];
	NSMutableArray<id<MTLCommandQueue>> *queues = [NSMutableArray array];
	NSMutableArray<id<MTLBuffer>> *bufs = [NSMutableArray array];
	for (int i = 0; i < count; i++) {
		id<MTLCommandQueue> q = shared ? one : [gDevice newCommandQueue];
		[queues addObject:q];
		[bufs addObject:newShared()];
		warm(q, bufs[i]);
	}
	Samples *per = calloc((size_t)count, sizeof(Samples));
	long *counts = calloc((size_t)count, sizeof(long));
	__block _Atomic int failures = 0;
	dispatch_group_t group = dispatch_group_create();
	Cpu cpu = cpuNow();
	const double t0 = nowUs() + 1000, end = t0 + gSeconds * 1e6;
	for (int i = 0; i < count; i++) {
		dispatch_group_enter(group);
		id<MTLCommandQueue> q = queues[i];
		id<MTLBuffer> buf = bufs[i];
		Samples *mine = &per[i];
		long *n = &counts[i];
		[NSThread detachNewThreadWithBlock:^{
			while (nowUs() < t0) {
			}
			id<MTLCommandBuffer> last = nil;
			while (nowUs() < end) {
				@autoreleasepool {
					id<MTLCommandBuffer> cb = makeBuffer(q, buf, YES, *n);
					double c = nowUs();
					[cb commit];
					if (!wait)
						progress();
					if (wait) {
						[cb waitUntilCompleted];
						sAdd(mine, nowUs() - c, c - t0);
						progress();
						if (cb.status != MTLCommandBufferStatusCompleted)
							atomic_fetch_add(&failures, 1);
					}
					last = cb;
					(*n)++;
				}
			}
			[last waitUntilCompleted];
			if (last.status != MTLCommandBufferStatusCompleted)
				atomic_fetch_add(&failures, 1);
			dispatch_group_leave(group);
		}];
	}
	dispatch_group_wait(group, DISPATCH_TIME_FOREVER);
	double t1 = nowUs();
	long total = 0;
	Samples ret = {0};
	for (int i = 0; i < count; i++) {
		total += counts[i];
		sMerge(&ret, &per[i]);
		sFree(&per[i]);
	}
	free(per);
	free(counts);
	if (atomic_load(&failures))
		return [NSString stringWithFormat:@"FAIL %d command buffers did not complete",
		                                  atomic_load(&failures)];
	NSString *line = [NSString stringWithFormat:@"OK threads=%d n=%ld cb_s=%.0f %@ %@", count, total,
	                                            total / ((t1 - t0) / 1e6),
	                                            wait ? sLine("ret", &ret) : @"", cpuLine(cpu, total)];
	sDetail("ret", &ret);
	sFree(&ret);
	return line;
}

static BOOL waits(id<MTLSharedEvent> ev)
{
	return [(id)ev respondsToSelector:@selector(waitUntilSignaledValue:timeoutMS:)];
}

// The CPU sets the value; another thread waits for it on the CPU.
static NSString *eventCpuCpu(void)
{
	id<MTLSharedEvent> ev = [gDevice newSharedEvent];
	if (!waits(ev))
		return @"SKIP no waitUntilSignaledValue:timeoutMS:";
	Samples wake = {0};
	dispatch_semaphore_t ready = dispatch_semaphore_create(0), woke = dispatch_semaphore_create(0);
	__block _Atomic double wokeAt = 0;
	__block _Atomic uint64_t want = 0;
	__block _Atomic int timeouts = 0;
	__block _Atomic int stop = 0;
	[NSThread detachNewThreadWithBlock:^{
		for (;;) {
			dispatch_semaphore_wait(ready, DISPATCH_TIME_FOREVER);
			if (atomic_load(&stop))
				break;
			if (![ev waitUntilSignaledValue:atomic_load(&want) timeoutMS:5000])
				atomic_fetch_add(&timeouts, 1);
			atomic_store(&wokeAt, nowUs());
			dispatch_semaphore_signal(woke);
		}
	}];
	Cpu cpu = cpuNow();
	double t0 = nowUs(), end = t0 + gSeconds * 1e6;
	long n = 0;
	for (uint64_t v = 1; nowUs() < end; v++) {
		atomic_store(&want, v);
		dispatch_semaphore_signal(ready);
		gap(gGapUs); // the waiter is asleep in the wait by now
		double s = nowUs();
		ev.signaledValue = v;
		dispatch_semaphore_wait(woke, DISPATCH_TIME_FOREVER);
		progress();
		sAdd(&wake, atomic_load(&wokeAt) - s, s - t0);
		n++;
	}
	atomic_store(&stop, 1);
	dispatch_semaphore_signal(ready);
	if (atomic_load(&timeouts))
		return [NSString stringWithFormat:@"FAIL %d waits timed out", atomic_load(&timeouts)];
	NSString *line = [NSString stringWithFormat:@"OK n=%ld %@ %@", n, sLine("wake", &wake), cpuLine(cpu, n)];
	sDetail("wake", &wake);
	sFree(&wake);
	return line;
}

// A tiny buffer signals the value; another thread waits for it on the CPU (`listener` NO), or a
// listener's block runs at it (`listener` YES).
static NSString *eventGpuCpu(BOOL listener)
{
	id<MTLSharedEvent> ev = [gDevice newSharedEvent];
	if (!listener && !waits(ev))
		return @"SKIP no waitUntilSignaledValue:timeoutMS:";
	id<MTLCommandQueue> q = [gDevice newCommandQueue];
	id<MTLBuffer> buf = newShared();
	warm(q, buf);
	MTLSharedEventListener *l = nil;
	if (listener)
		l = [[MTLSharedEventListener alloc]
		    initWithDispatchQueue:dispatch_queue_create("submit_probe.listener", DISPATCH_QUEUE_SERIAL)];
	Samples wake = {0}, done = {0};
	dispatch_semaphore_t ready = dispatch_semaphore_create(0), woke = dispatch_semaphore_create(0);
	__block _Atomic double wokeAt = 0;
	__block _Atomic uint64_t want = 0;
	__block _Atomic int timeouts = 0;
	__block _Atomic int stop = 0;
	if (!listener) {
		[NSThread detachNewThreadWithBlock:^{
			for (;;) {
				dispatch_semaphore_wait(ready, DISPATCH_TIME_FOREVER);
				if (atomic_load(&stop))
					break;
				if (![ev waitUntilSignaledValue:atomic_load(&want) timeoutMS:5000])
					atomic_fetch_add(&timeouts, 1);
				atomic_store(&wokeAt, nowUs());
				dispatch_semaphore_signal(woke);
			}
		}];
	}
	Stamps *st = calloc(1, sizeof(Stamps));
	Cpu cpu = cpuNow();
	double t0 = nowUs(), end = t0 + gSeconds * 1e6;
	long n = 0;
	NSString *why = nil;
	for (uint64_t v = 1; nowUs() < end; v++) {
		@autoreleasepool {
			atomic_store(&want, v);
			atomic_store(&st->done, 0);
			if (listener) {
				[ev notifyListener:l
				           atValue:v
				             block:^(id<MTLSharedEvent> e, uint64_t value) {
					             atomic_store(&wokeAt, nowUs());
					             dispatch_semaphore_signal(woke);
				             }];
			} else {
				dispatch_semaphore_signal(ready);
			}
			id<MTLCommandBuffer> cb = makeBuffer(q, buf, YES, (long)v);
			[cb encodeSignalEvent:ev value:v];
			[cb addCompletedHandler:^(id<MTLCommandBuffer> b) {
				atomic_store(&st->done, nowUs());
			}];
			gap(gGapUs);
			double c = nowUs();
			[cb commit];
			if (dispatch_semaphore_wait(woke, dispatch_time(DISPATCH_TIME_NOW, 5 * NSEC_PER_SEC))) {
				why = @"FAIL the signal never reached the CPU";
				break;
			}
			[cb waitUntilCompleted];
			while (atomic_load(&st->done) == 0) {
			}
			progress();
			if (failed(cb, &why))
				break;
			sAdd(&wake, atomic_load(&wokeAt) - c, c - t0);
			sAdd(&done, atomic_load(&st->done) - c, c - t0);
			n++;
		}
	}
	atomic_store(&stop, 1);
	dispatch_semaphore_signal(ready);
	free(st);
	if (why)
		return why;
	if (atomic_load(&timeouts))
		return [NSString stringWithFormat:@"FAIL %d waits timed out", atomic_load(&timeouts)];
	NSString *line = [NSString stringWithFormat:@"OK n=%ld %@ %@ %@", n, sLine("wake", &wake),
	                                            sLine("done", &done), cpuLine(cpu, n)];
	sDetail("wake", &wake);
	sFree(&wake);
	sFree(&done);
	return line;
}

// A tiny buffer waits for the value the CPU sets after the gap.
static NSString *eventCpuGpu(void)
{
	id<MTLSharedEvent> ev = [gDevice newSharedEvent];
	id<MTLCommandQueue> q = [gDevice newCommandQueue];
	id<MTLBuffer> buf = newShared();
	warm(q, buf);
	Samples done = {0}, ret = {0};
	Stamps *st = calloc(1, sizeof(Stamps));
	Cpu cpu = cpuNow();
	double t0 = nowUs(), end = t0 + gSeconds * 1e6;
	long n = 0;
	NSString *why = nil;
	for (uint64_t v = 1; nowUs() < end; v++) {
		@autoreleasepool {
			atomic_store(&st->done, 0);
			id<MTLCommandBuffer> cb = [q commandBuffer];
			[cb encodeWaitForEvent:ev value:v];
			id<MTLBlitCommandEncoder> b = [cb blitCommandEncoder];
			[b fillBuffer:buf range:NSMakeRange(0, 16) value:(uint8_t)v];
			[b endEncoding];
			[cb addCompletedHandler:^(id<MTLCommandBuffer> cbuf) {
				atomic_store(&st->done, nowUs());
			}];
			[cb commit];
			gap(gGapUs);
			double s = nowUs();
			ev.signaledValue = v;
			[cb waitUntilCompleted];
			double r = nowUs();
			while (atomic_load(&st->done) == 0) {
			}
			progress();
			if (failed(cb, &why))
				break;
			sAdd(&done, atomic_load(&st->done) - s, s - t0);
			sAdd(&ret, r - s, s - t0);
			n++;
		}
	}
	free(st);
	if (why)
		return why;
	NSString *line = [NSString stringWithFormat:@"OK n=%ld %@ %@ %@", n, sLine("done", &done),
	                                            sLine("ret", &ret), cpuLine(cpu, n)];
	sDetail("done", &done);
	sFree(&done);
	sFree(&ret);
	return line;
}

// Queue 2's buffer waits for the value queue 1's tiny buffer signals.
static NSString *eventGpuGpu(void)
{
	id<MTLSharedEvent> ev = [gDevice newSharedEvent];
	id<MTLCommandQueue> q1 = [gDevice newCommandQueue], q2 = [gDevice newCommandQueue];
	id<MTLBuffer> buf1 = newShared(), buf2 = newShared();
	warm(q1, buf1);
	warm(q2, buf2);
	Samples done = {0}, own = {0};
	Stamps *st1 = calloc(1, sizeof(Stamps)), *st2 = calloc(1, sizeof(Stamps));
	Cpu cpu = cpuNow();
	double t0 = nowUs(), end = t0 + gSeconds * 1e6;
	long n = 0;
	NSString *why = nil;
	for (uint64_t v = 1; nowUs() < end; v++) {
		@autoreleasepool {
			atomic_store(&st1->done, 0);
			atomic_store(&st2->done, 0);
			id<MTLCommandBuffer> waiter = [q2 commandBuffer];
			[waiter encodeWaitForEvent:ev value:v];
			id<MTLBlitCommandEncoder> b = [waiter blitCommandEncoder];
			[b fillBuffer:buf2 range:NSMakeRange(0, 16) value:(uint8_t)v];
			[b endEncoding];
			[waiter addCompletedHandler:^(id<MTLCommandBuffer> cbuf) {
				atomic_store(&st2->done, nowUs());
			}];
			[waiter commit];
			id<MTLCommandBuffer> signaller = makeBuffer(q1, buf1, YES, (long)v);
			[signaller encodeSignalEvent:ev value:v];
			[signaller addCompletedHandler:^(id<MTLCommandBuffer> cbuf) {
				atomic_store(&st1->done, nowUs());
			}];
			gap(gGapUs);
			double c = nowUs();
			[signaller commit];
			[waiter waitUntilCompleted];
			[signaller waitUntilCompleted];
			while (atomic_load(&st1->done) == 0 || atomic_load(&st2->done) == 0) {
			}
			progress();
			if (failed(waiter, &why) || failed(signaller, &why))
				break;
			sAdd(&done, atomic_load(&st2->done) - c, c - t0);
			sAdd(&own, atomic_load(&st1->done) - c, c - t0);
			n++;
		}
	}
	free(st1);
	free(st2);
	if (why)
		return why;
	NSString *line = [NSString stringWithFormat:@"OK n=%ld %@ %@ %@", n, sLine("done", &done),
	                                            sLine("own", &own), cpuLine(cpu, n)];
	sDetail("done", &done);
	sFree(&done);
	sFree(&own);
	return line;
}

// MARK: - The runner

static BOOL wanted(NSString *name)
{
	if (gPatterns.count == 0)
		return YES;
	for (NSString *p in gPatterns)
		if ([name containsString:p])
			return YES;
	return NO;
}

static void runCase(NSString *name, NSString * (^body)(void))
{
	if (!wanted(name))
		return;
	printf("# start %s\n", name.UTF8String);
	fflush(stdout);
	atomic_store(&gCaseName, strdup(name.UTF8String));
	struct timeval tv;
	gettimeofday(&tv, NULL);
	gCaseWall = tv.tv_sec + tv.tv_usec / 1e6;
	atomic_store(&gBusy, 1);
	NSString *result;
	@autoreleasepool {
		result = body();
	}
	atomic_store(&gBusy, 0);
	printf("submit_probe: %s %s\n", name.UTF8String, result.UTF8String);
	fflush(stdout);
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
			else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc)
				gSeconds = atof(argv[++i]);
			else if (strcmp(argv[i], "-g") == 0 && i + 1 < argc)
				gGapUs = atof(argv[++i]);
			else if (strcmp(argv[i], "-n") == 0 && i + 1 < argc)
				gMaxThreads = atoi(argv[++i]);
			else if (strcmp(argv[i], "-q") == 0 && i + 1 < argc)
				gIdleQueues = atoi(argv[++i]);
			else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc)
				gHoldSeconds = atof(argv[++i]);
			else if (strcmp(argv[i], "-l") == 0)
				list = YES;
			else if (strcmp(argv[i], "-v") == 0)
				gVerbose = YES;
			else
				[patterns addObject:@(argv[i])];
		}
		gPatterns = patterns;
		if (list) {
			for (id<MTLDevice> d in MTLCopyAllDevices())
				printf("submit_probe: device \"%s\" lowPower=%d removable=%d unified=%d\n",
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
			printf("submit_probe: no Metal device%s%s\n", want ? " matching " : "", want ? want : "");
			return 2;
		}
		char model[64] = "?", build[32] = "?";
		int cpus = 0;
		size_t len = sizeof(model);
		sysctlbyname("hw.model", model, &len, NULL, 0);
		len = sizeof(build);
		sysctlbyname("kern.osversion", build, &len, NULL, 0);
		len = sizeof(cpus);
		sysctlbyname("hw.logicalcpu", &cpus, &len, NULL, 0);
		NSOperatingSystemVersion v = NSProcessInfo.processInfo.operatingSystemVersion;
		printf("submit_probe: device \"%s\" (%s), %s, %d CPUs, macOS %ld.%ld.%ld (%s), %s, %.1f s a "
		       "case, gap %.0f us\n",
		       gDevice.name.UTF8String, NSStringFromClass([gDevice class]).UTF8String, model, cpus,
		       (long)v.majorVersion, (long)v.minorVersion, (long)v.patchVersion, build,
#if defined(__x86_64__)
		       "x86_64",
#else
		       "arm64",
#endif
		       gSeconds, gGapUs);
		pthread_t t;
		pthread_create(&t, NULL, watchdog, NULL);
		NSMutableArray<id<MTLCommandQueue>> *idle = [NSMutableArray array];
		if (gIdleQueues > 0) {
			id<MTLBuffer> buf = newShared();
			double q0 = nowUs();
			// Under the watchdog: a queue whose command buffer never completes is a hang.
			atomic_store(&gCaseName, "making idle queues");
			atomic_store(&gBusy, 1);
			for (int i = 0; i < gIdleQueues; i++) {
				@autoreleasepool {
					id<MTLCommandQueue> q = [gDevice newCommandQueue];
					if (q == nil) {
						printf("submit_probe: newCommandQueue answered nil after %d idle queues\n", i);
						break;
					}
					[idle addObject:q];
					id<MTLCommandBuffer> cb = makeBuffer(q, buf, YES, i);
					[cb commit];
					[cb waitUntilCompleted];
					progress();
				}
			}
			atomic_store(&gBusy, 0);
			printf("submit_probe: %lu idle queues made in %.0f ms\n", (unsigned long)idle.count,
			       (nowUs() - q0) / 1e3);
			fflush(stdout);
			if (gHoldSeconds > 0)
				usleep((useconds_t)(gHoldSeconds * 1e6));
		}

		runCase(@"rate.commit.empty", ^{ return commitRate(NO); });
		runCase(@"rate.commit.tiny", ^{ return commitRate(YES); });
		runCase(@"rate.wait.empty", ^{ return waitRate(NO); });
		runCase(@"rate.wait.tiny", ^{ return waitRate(YES); });
		runCase(@"rate.handler.tiny", ^{ return handlerRate(); });
		runCase(@"lat.tiny.busy", ^{ return latency(0); });
		runCase(@"lat.tiny.idle", ^{ return latency(gGapUs); });
		runCase(@"submitted.empty.idle", ^{ return submitted(NO, NO); });
		runCase(@"submitted.tiny.idle", ^{ return submitted(YES, NO); });
		runCase(@"submitted.tiny.behind", ^{ return submitted(YES, YES); });
		for (int n = 2; n <= gMaxThreads; n *= 2) {
			runCase([NSString stringWithFormat:@"mt.commit.q%d", n], ^{ return threads(n, NO, NO); });
			runCase([NSString stringWithFormat:@"mt.wait.q%d", n], ^{ return threads(n, NO, YES); });
			runCase([NSString stringWithFormat:@"mt.wait.shared%d", n], ^{ return threads(n, YES, YES); });
		}
		runCase(@"event.cpu_cpu", ^{ return eventCpuCpu(); });
		runCase(@"event.gpu_cpu", ^{ return eventGpuCpu(NO); });
		runCase(@"event.listener", ^{ return eventGpuCpu(YES); });
		runCase(@"event.cpu_gpu", ^{ return eventCpuGpu(); });
		runCase(@"event.gpu_gpu", ^{ return eventGpuGpu(); });
	}
	return 0;
}
