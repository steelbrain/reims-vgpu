// oracle_prio: what the device does with the queue priority marks of the command-queue SPI,
// `setGPUPriority:` and `setBackgroundGPUPriority:`, that a guest's background service sets on its
// queues: whether a short compute buffer of a second process starts sooner beside the marked
// process's long buffers. dgpu-oracle.md Q50 holds what it found.
//   oracle_prio DEVICE ARM [ROUNDS [TARGET_MS [AT_MS]]]
// First, on queues of DEVICE made for it: whether a queue answers `setGPUPriority:`,
// `setGPUPriority:offset:`, `setBackgroundGPUPriority:` and `setBackgroundGPUPriority:offset:`
// (`respondsToSelector:`) and each one's type encoding; every method of the queue's classes whose
// name holds "priority"; and what each integer getter among them reads on a fresh queue, after
// `setBackgroundGPUPriority: 2` and then `setGPUPriority: 2` on it, after `setGPUPriority: 0` on a
// second, and after each offset form with 2, offset 1, on a third and a fourth, with what each
// setter returned.
// Process A (this one) calibrates a busy kernel (a spin over a 1024 x 1024 Private R32Uint texture
// of its own) to TARGET_MS (default 60) of GPU time a buffer, on an unmarked queue of its own: warm-up
// buffers for 500 ms, 2 measured and scaled, 2 more measured and scaled again, then 2 measured with
// the spin used. It then makes the queue the rounds run on, marks it as the arm says before any use,
// and runs one buffer on it alone. Each round it commits 4 busy buffers (L0 to L3) back to back by
// `commit`, and AT_MS (default 20) after L0's commit tells process B (a second process, the same
// binary, told over a pipe), which at once commits one short compute buffer (4096 threads writing a
// buffer of its own) by `commit` on a queue of its own: it shares nothing with A. The GPU idles 300
// ms between rounds.
//   ARM 1: no marks (the control).
//   ARM 2: A's queue marked as the guest's background service marks its heavy queues:
//          `setBackgroundGPUPriority: 2`, then `setGPUPriority: 2`. B's unmarked.
//   ARM 3: A's as in arm 2, and B's marked `setGPUPriority: 0`, as the guest's WindowServer marks its
//          own.
//   ARM 4: A unmarked, and B marked `setGPUPriority: 0`: WindowServer's mark alone.
// A selector an arm needs that the queue does not answer ends the run (exit 3), saying so.
// Times are ms on one host clock for both processes: mach absolute time, which the command
// buffers' GPUStartTime and GPUEndTime are on (the calibration line checks it: commit to GPU start
// and GPU end to completed handler, each small and positive). Each round prints, from B's commit:
// - B's GPU start, GPU end, scheduled and completed handlers, and its GPU run;
// - the buffer of A whose GPU run spans B's commit, how far into that run B committed, and its GPU
//   end; B's GPU start against that end (negative: B started before it ended);
// - A's buffers committed before B and not yet started at B's commit (pending), how many of them
//   started after B's GPU start (B ahead of them), and B's start against the first one's start;
// - each of A's 4 buffers' GPU run, and the statuses;
// - the round's reading: "beside" (B started before the running buffer ended), "between" (after
//   it, ahead of every pending buffer), "among" (ahead of some) or "behind" (after them all).
// A last line counts the readings and gives B's commit-to-GPU-start minimum, median and maximum.
// Q50 ran each arm on each GPU, off screen, as:
//   caffeinate -s ./oracle_prio DEVICE ARM 8 60 20      (DEVICE 630 or 5300M; arms 1, 2, 3, 4, 1)
// Build: see README.md.
#import "oracle_common.h"
#include <objc/message.h>
#include <objc/runtime.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

extern char **environ;
enum { kSide = 1024, kLong = 4, kTick = 4096, kTickSpin = 256, kMaxGetters = 16 };

static NSString *const kSource = @"#include <metal_stdlib>\n"
	"using namespace metal;\n"
	"static uint spinFor(uint x, uint spin) {\n"
	"  for (uint i = 0; i < spin; i++) x = x * 1664525u + 1013904223u; return x; }\n"
	"kernel void busy(texture2d<uint, access::write> t [[texture(0)]], constant uint &spin [[buffer(0)]],\n"
	"  uint2 g [[thread_position_in_grid]]) { t.write(uint4(spinFor(g.x * 2654435761u + g.y, spin)), g); }\n"
	"kernel void tick(device uint *b [[buffer(0)]], constant uint &spin [[buffer(1)]],\n"
	"  uint g [[thread_position_in_grid]]) { b[g] = spinFor(g * 2654435761u, spin); }\n";

static const char *const kSetters[] = {"setGPUPriority:", "setGPUPriority:offset:", "setBackgroundGPUPriority:",
                                       "setBackgroundGPUPriority:offset:"};

// The integer getters of no argument among the queue's priority methods.
static struct {
	char name[96];
	char type;
} gGetters[kMaxGetters];
static int gNGetters;

// A raw integer return, cut to its type's width and sign-extended where the type is signed.
static long long cut(unsigned long long raw, char type)
{
	switch (type) {
	case 'c': return (signed char)raw;
	case 'C':
	case 'B': return (unsigned char)raw;
	case 's': return (short)raw;
	case 'S': return (unsigned short)raw;
	case 'i': return (int)raw;
	case 'I': return (unsigned)raw;
	default: return (long long)raw;
	}
}

// Walks the queue's classes below NSObject for methods whose name holds "priority", each name once,
// the most derived class's first; keeps the integer getters of no argument, and prints the methods
// when `print`.
static void priorityMethods(id q, bool print)
{
	NSMutableSet *seen = [NSMutableSet set];
	gNGetters = 0;
	if (print)
		printf("priority methods:");
	for (Class c = object_getClass(q); c && c != [NSObject class]; c = class_getSuperclass(c)) {
		unsigned n = 0;
		Method *ms = class_copyMethodList(c, &n);
		for (unsigned i = 0; i < n; i++) {
			const char *name = sel_getName(method_getName(ms[i]));
			if (!strcasestr(name, "priority") || [seen containsObject:@(name)])
				continue;
			[seen addObject:@(name)];
			if (print)
				printf(" %s %s [%s];", name, method_getTypeEncoding(ms[i]), class_getName(c));
			char rt[16] = "";
			method_getReturnType(ms[i], rt, sizeof rt);
			if (!strchr(name, ':') && strlen(rt) == 1 && strchr("QqLlIiSsCcB", rt[0]) && gNGetters < kMaxGetters) {
				snprintf(gGetters[gNGetters].name, sizeof gGetters[0].name, "%s", name);
				gGetters[gNGetters++].type = rt[0];
			}
		}
		free(ms);
	}
	if (print)
		printf("%s\n", seen.count ? "" : " none");
}

// "{getter=value ...}" for the queue.
static const char *readGetters(id q, char *buf, size_t len)
{
	size_t at = snprintf(buf, len, "{");
	for (int i = 0; i < gNGetters && at < len; i++) {
		SEL s = sel_registerName(gGetters[i].name);
		unsigned long long raw = ((unsigned long long (*)(id, SEL))objc_msgSend)(q, s);
		at += snprintf(buf + at, len - at, "%s%s=%lld", i ? " " : "", gGetters[i].name, cut(raw, gGetters[i].type));
	}
	if (at < len)
		snprintf(buf + at, len - at, "}");
	return buf;
}

// Calls the setter `name` on q with `value` (and `offset` for an offset form); says what it returned.
static const char *callSet(id q, const char *name, NSUInteger value, unsigned short offset, char *buf, size_t len)
{
	SEL s = sel_registerName(name);
	if (![q respondsToSelector:s]) {
		snprintf(buf, len, "not answered");
		return buf;
	}
	Method m = class_getInstanceMethod(object_getClass(q), s);
	char rt[16] = "v";
	if (m)
		method_getReturnType(m, rt, sizeof rt);
	unsigned long long raw;
	if (strchr(name, ':') != strrchr(name, ':'))
		raw = ((unsigned long long (*)(id, SEL, NSUInteger, unsigned short))objc_msgSend)(q, s, value, offset);
	else
		raw = ((unsigned long long (*)(id, SEL, NSUInteger))objc_msgSend)(q, s, value);
	if (rt[0] == 'v')
		snprintf(buf, len, "returned void");
	else
		snprintf(buf, len, "returned %lld", cut(raw, rt[0]));
	return buf;
}

// The first line: which setters a queue answers, its priority methods, and what the getters read.
static void answerSelectors(id<MTLDevice> dev)
{
	id q = [dev newCommandQueue];
	printf("selectors (queue class");
	for (Class c = object_getClass(q); c && c != [NSObject class]; c = class_getSuperclass(c))
		printf(" %s", class_getName(c));
	printf("):");
	for (size_t i = 0; i < sizeof kSetters / sizeof kSetters[0]; i++) {
		SEL s = sel_registerName(kSetters[i]);
		Method m = class_getInstanceMethod(object_getClass(q), s);
		printf(" %s %s%s%s;", kSetters[i], [q respondsToSelector:s] ? "yes" : "NO", m ? " " : "",
		       m ? method_getTypeEncoding(m) : "");
	}
	printf("\n");
	priorityMethods(q, true);
	fflush(stdout);
	char g[1024], r[64];
	printf("readback: a fresh queue %s", readGetters(q, g, sizeof g));
	callSet(q, "setBackgroundGPUPriority:", 2, 0, r, sizeof r);
	printf("; setBackgroundGPUPriority: 2 %s, %s", r, readGetters(q, g, sizeof g));
	callSet(q, "setGPUPriority:", 2, 0, r, sizeof r);
	printf("; then setGPUPriority: 2 %s, %s", r, readGetters(q, g, sizeof g));
	id q0 = [dev newCommandQueue];
	callSet(q0, "setGPUPriority:", 0, 0, r, sizeof r);
	printf("; a second queue: setGPUPriority: 0 %s, %s", r, readGetters(q0, g, sizeof g));
	id q1 = [dev newCommandQueue];
	callSet(q1, "setGPUPriority:offset:", 2, 1, r, sizeof r);
	printf("; a third: setGPUPriority: 2 offset: 1 %s, %s", r, readGetters(q1, g, sizeof g));
	id q2 = [dev newCommandQueue];
	callSet(q2, "setBackgroundGPUPriority:offset:", 2, 1, r, sizeof r);
	printf("; a fourth: setBackgroundGPUPriority: 2 offset: 1 %s, %s\n", r, readGetters(q2, g, sizeof g));
	fflush(stdout);
}

static id<MTLComputePipelineState> pipeline(id<MTLDevice> dev, NSString *name)
{
	NSError *err = nil;
	id<MTLLibrary> lib = [dev newLibraryWithSource:kSource options:nil error:&err];
	id<MTLComputePipelineState> p = lib ? [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:name]
	                                                                         error:&err]
	                                    : nil;
	if (!p) {
		printf("  ! %s\n", err.description.UTF8String);
		exit(2);
	}
	return p;
}

// B: marks its queue in arm 3, runs two warm-up buffers, writes "ready ..." back; then per round, on
// "g", commits one short buffer and writes back
// "commit_ms gpu_start_ms gpu_end_ms scheduled_ms completed_ms status\n".
static int sideB(const char *device, int arm, int in, int out, int rounds)
{
	id<MTLDevice> dev = oracle_device(device);
	id<MTLComputePipelineState> tick = pipeline(dev, @"tick");
	id<MTLCommandQueue> q = [dev newCommandQueue];
	id<MTLBuffer> buf = [dev newBufferWithLength:kTick * 4 options:MTLResourceStorageModePrivate];
	char line[1400], g[1024], r[64];
	int n;
	priorityMethods(q, false);
	if (arm == 3 || arm == 4) {
		if (![(id)q respondsToSelector:sel_registerName("setGPUPriority:")]) {
			n = snprintf(line, sizeof line, "ready NO: B's queue does not answer setGPUPriority:\n");
			write(out, line, n);
			return 3;
		}
		callSet(q, "setGPUPriority:", 0, 0, r, sizeof r);
		n = snprintf(line, sizeof line, "ready, B's queue marked setGPUPriority: 0 %s, %s", r, readGetters(q, g, sizeof g));
	} else {
		n = snprintf(line, sizeof line, "ready, B's queue unmarked %s", readGetters(q, g, sizeof g));
	}
	uint32_t spin = kTickSpin;
	id<MTLCommandBuffer> (^one)(void) = ^id<MTLCommandBuffer>(void) {
		id<MTLCommandBuffer> cb = [q commandBuffer];
		id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
		[e setComputePipelineState:tick];
		[e setBuffer:buf offset:0 atIndex:0];
		[e setBytes:&spin length:4 atIndex:1];
		[e dispatchThreads:MTLSizeMake(kTick, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
		[e endEncoding];
		return cb;
	};
	for (int i = 0; i < 2; i++) {
		id<MTLCommandBuffer> cb = one();
		[cb commit];
		[cb waitUntilCompleted];
		n += snprintf(line + n, sizeof line - n, "%s %.3f", i ? "," : "; warm-up GPU runs", (cb.GPUEndTime - cb.GPUStartTime) * 1e3);
	}
	n += snprintf(line + n, sizeof line - n, " ms\n");
	write(out, line, n);
	for (int k = 0; k < rounds; k++) {
		char c;
		if (read(in, &c, 1) != 1)
			break;
		@autoreleasepool {
			id<MTLCommandBuffer> cb = one();
			__block double sched = 0, done = 0;
			[cb addScheduledHandler:^(id<MTLCommandBuffer> x) { sched = oracle_now_ms(); }];
			[cb addCompletedHandler:^(id<MTLCommandBuffer> x) { done = oracle_now_ms(); }];
			const double t0 = oracle_now_ms();
			[cb commit];
			[cb waitUntilCompleted];
			usleep(2000);
			n = snprintf(line, sizeof line, "%.4f %.4f %.4f %.4f %.4f %ld\n", t0, cb.GPUStartTime * 1e3,
			             cb.GPUEndTime * 1e3, sched, done, (long)cb.status);
			write(out, line, n);
		}
	}
	return 0;
}

static int cmpd(const void *a, const void *b)
{
	double x = *(const double *)a, y = *(const double *)b;
	return x < y ? -1 : x > y;
}

int main(int argc, char **argv)
{
	setvbuf(stdout, NULL, _IOLBF, 0);
	@autoreleasepool {
		if (argc == 7 && strcmp(argv[2], "B") == 0)
			return sideB(argv[1], atoi(argv[3]), atoi(argv[4]), atoi(argv[5]), atoi(argv[6]));
		if (argc < 3 || atoi(argv[2]) < 1 || atoi(argv[2]) > 4) {
			fprintf(stderr, "usage: %s DEVICE 1|2|3|4 [ROUNDS [TARGET_MS [AT_MS]]]\n", argv[0]);
			return 2;
		}
		const int arm = atoi(argv[2]);
		const int rounds = argc > 3 ? atoi(argv[3]) : 8;
		const double target = argc > 4 ? atof(argv[4]) : 60.0;
		const double at = argc > 5 ? atof(argv[5]) : 20.0;
		id<MTLDevice> dev = oracle_device(argv[1]);
		oracle_header("oracle_prio", dev);
		answerSelectors(dev);
		if (arm == 2 || arm == 3) {
			id probe = [dev newCommandQueue];
			for (int i = 0; i < 2; i++) {
				const char *need = i ? "setGPUPriority:" : "setBackgroundGPUPriority:";
				if (![probe respondsToSelector:sel_registerName(need)]) {
					printf("! the queue does not answer %s: arm %d is not run on this GPU\n", need, arm);
					return 3;
				}
			}
		}
		id<MTLComputePipelineState> busyState = pipeline(dev, @"busy");
		MTLTextureDescriptor *td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR32Uint
		                                                                             width:kSide height:kSide mipmapped:NO];
		td.usage = MTLTextureUsageShaderWrite;
		td.storageMode = MTLStorageModePrivate;
		id<MTLTexture> tex = [dev newTextureWithDescriptor:td];
		id<MTLCommandBuffer> (^busy)(id<MTLCommandQueue>, uint32_t) = ^id<MTLCommandBuffer>(id<MTLCommandQueue> q, uint32_t spin) {
			id<MTLCommandBuffer> cb = [q commandBuffer];
			id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
			[e setComputePipelineState:busyState];
			[e setTexture:tex atIndex:0];
			[e setBytes:&spin length:4 atIndex:0];
			[e dispatchThreads:MTLSizeMake(kSide, kSide, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
			[e endEncoding];
			return cb;
		};
		// Calibrate on an unmarked queue: warm-up buffers for 500 ms (a GPU just woken runs them
		// short of the target), then two rounds of 2 measured and scaled.
		id<MTLCommandQueue> cal = [dev newCommandQueue];
		double (^measure)(uint32_t, int) = ^double(uint32_t spin, int count) {
			double sum = 0;
			for (int i = 0; i < count; i++) {
				id<MTLCommandBuffer> cb = busy(cal, spin);
				[cb commit];
				[cb waitUntilCompleted];
				sum += (cb.GPUEndTime - cb.GPUStartTime) * 1e3;
			}
			return sum / count;
		};
		for (const double w0 = oracle_now_ms(); oracle_now_ms() < w0 + 500;)
			measure(2000, 1);
		uint32_t spin = (uint32_t)(2000.0 * target / measure(2000, 2));
		spin = (uint32_t)(spin * target / measure(spin, 2));
		double runs[2], c0 = 0;
		__block double calDone = 0;
		for (int i = 0; i < 2; i++) {
			id<MTLCommandBuffer> cb = busy(cal, spin);
			[cb addCompletedHandler:^(id<MTLCommandBuffer> x) { calDone = oracle_now_ms(); }];
			c0 = oracle_now_ms();
			[cb commit];
			[cb waitUntilCompleted];
			usleep(2000);
			runs[i] = (cb.GPUEndTime - cb.GPUStartTime) * 1e3;
			if (i == 1)
				printf("# arm %d, %d rounds, B told %.1f ms after L0's commit; busy spin %u: %.2f and %.2f ms GPU on an "
				       "unmarked queue (target %.1f); clock check: commit to GPU start %.3f ms, GPU end to completed "
				       "handler %.3f ms\n",
				       arm, rounds, at, spin, runs[0], runs[1], target, cb.GPUStartTime * 1e3 - c0,
				       calDone - cb.GPUEndTime * 1e3);
		}
		// The rounds' queue, marked as the arm says before any use, as the guest marks its queues.
		id<MTLCommandQueue> q = [dev newCommandQueue];
		char g[1024], r1[64], r2[64];
		priorityMethods(q, false);
		if (arm == 2 || arm == 3) {
			callSet(q, "setBackgroundGPUPriority:", 2, 0, r1, sizeof r1);
			callSet(q, "setGPUPriority:", 2, 0, r2, sizeof r2);
			printf("A's queue: setBackgroundGPUPriority: 2 %s, then setGPUPriority: 2 %s, %s", r1, r2,
			       readGetters(q, g, sizeof g));
		} else {
			printf("A's queue: unmarked, %s", readGetters(q, g, sizeof g));
		}
		{
			id<MTLCommandBuffer> cb = busy(q, spin);
			[cb commit];
			[cb waitUntilCompleted];
			printf("; one busy buffer on it alone: %.2f ms GPU\n", (cb.GPUEndTime - cb.GPUStartTime) * 1e3);
		}
		int toB[2], fromB[2];
		pipe(toB);
		pipe(fromB);
		char armS[8], rfd[16], wfd[16], nr[16];
		snprintf(armS, sizeof armS, "%d", arm);
		snprintf(rfd, sizeof rfd, "%d", toB[0]);
		snprintf(wfd, sizeof wfd, "%d", fromB[1]);
		snprintf(nr, sizeof nr, "%d", rounds);
		char *args[] = {argv[0], argv[1], "B", armS, rfd, wfd, nr, NULL};
		pid_t pid = 0;
		if (posix_spawn(&pid, argv[0], NULL, NULL, args, environ) != 0) {
			printf("! posix_spawn of B failed\n");
			return 2;
		}
		close(toB[0]);
		close(fromB[1]);
		FILE *fb = fdopen(fromB[0], "r");
		char ready[1400];
		if (!fgets(ready, sizeof ready, fb)) {
			printf("! no ready line from B\n");
			return 2;
		}
		printf("B: %s", ready);
		if (strncmp(ready, "ready NO", 8) == 0) {
			printf("! arm %d is not run on this GPU\n", arm);
			return 3;
		}
		usleep(300000);
		double bStarts[64];
		int nStarts = 0, beside = 0, between = 0, among = 0, behind = 0, none = 0;
		double aRuns[64 * kLong];
		int nRuns = 0;
		for (int k = 0; k < rounds; k++) {
			@autoreleasepool {
				id<MTLCommandBuffer> ls[kLong];
				double commits[kLong];
				for (int i = 0; i < kLong; i++) {
					ls[i] = busy(q, spin);
					commits[i] = oracle_now_ms();
					[ls[i] commit];
				}
				while (oracle_now_ms() < commits[0] + at)
					usleep(200);
				write(toB[1], "g", 1);
				double tB = 0, bS = 0, bE = 0, bSched = 0, bDone = 0;
				long bst = 0;
				if (fscanf(fb, "%lf %lf %lf %lf %lf %ld", &tB, &bS, &bE, &bSched, &bDone, &bst) != 6) {
					printf("  ! no answer from B\n");
					break;
				}
				for (int i = 0; i < kLong; i++)
					[ls[i] waitUntilCompleted];
				usleep(2000);
				double s[kLong], e[kLong];
				int run = -1;
				for (int i = 0; i < kLong; i++) {
					s[i] = ls[i].GPUStartTime * 1e3 - tB;
					e[i] = ls[i].GPUEndTime * 1e3 - tB;
					if (s[i] <= 0 && e[i] > 0)
						run = i;
					if (nRuns < 64 * kLong)
						aRuns[nRuns++] = e[i] - s[i];
				}
				bS -= tB;
				bE -= tB;
				bSched -= tB;
				bDone -= tB;
				if (nStarts < 64)
					bStarts[nStarts++] = bS;
				int pend = 0, ahead = 0, first = -1;
				for (int i = 0; i < kLong; i++) {
					if (commits[i] < tB && s[i] > 0) {
						pend++;
						if (first < 0)
							first = i;
						if (s[i] > bS)
							ahead++;
					}
				}
				const char *reading;
				if (run >= 0 && bS < e[run]) {
					reading = "beside";
					beside++;
				} else if (pend && ahead == pend) {
					reading = "between";
					between++;
				} else if (ahead) {
					reading = "among";
					among++;
				} else if (pend) {
					reading = "behind";
					behind++;
				} else {
					reading = "no pending buffer";
					none++;
				}
				printf("round %d arm %d: B GPU start %+.3f, end %+.3f, scheduled %+.3f, completed %+.3f ms (GPU run "
				       "%.3f); ",
				       k, arm, bS, bE, bSched, bDone, bE - bS);
				if (run >= 0)
					printf("running at B's commit: L%d, %.2f ms into its run, its GPU end %+.2f: B's start %+.2f ms "
					       "from that end; ",
					       run, -s[run], e[run], bS - e[run]);
				else
					printf("no buffer of A running at B's commit (L0 GPU start %+.2f, L%d end %+.2f); ", s[0],
					       kLong - 1, e[kLong - 1]);
				if (first >= 0)
					printf("pending L%d-L%d (%d): B ahead of %d, B's start %+.2f ms from L%d's start; ", first,
					       kLong - 1, pend, ahead, bS - s[first], first);
				else
					printf("pending none; ");
				printf("A's GPU runs");
				for (int i = 0; i < kLong; i++)
					printf(" %.2f", e[i] - s[i]);
				printf(" ms; status A");
				for (int i = 0; i < kLong; i++)
					printf(" %ld", (long)ls[i].status);
				printf(" B %ld; %s\n", bst, reading);
				usleep(300000);
			}
		}
		close(toB[1]);
		int st;
		waitpid(pid, &st, 0);
		if (nStarts) {
			qsort(bStarts, nStarts, sizeof(double), cmpd);
			qsort(aRuns, nRuns, sizeof(double), cmpd);
			printf("arm %d: B's commit to GPU start min %.3f, median %.3f, max %.3f ms; beside %d, between %d, among %d, "
			       "behind %d, no pending %d, of %d rounds; A's GPU run median %.2f ms (min %.2f, max %.2f)\n",
			       arm, bStarts[0], bStarts[nStarts / 2], bStarts[nStarts - 1], beside, between, among, behind, none,
			       nStarts, aRuns[nRuns / 2], aRuns[0], aRuns[nRuns - 1]);
		}
	}
	return 0;
}
