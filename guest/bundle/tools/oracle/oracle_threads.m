// oracle_threads: what a compute pipeline's descriptor fields of its threadgroups make, so the
// bundle refuses, ends the process at, and answers the same: requiredThreadsPerThreadgroup,
// maxTotalThreadsPerThreadgroup against the kernel's [[max_total_threads_per_threadgroup(n)]] and
// the device's limit, threadGroupSizeIsMultipleOfThreadExecutionWidth beside a required size, and
// macOS 27's forwardProgressUsage, contentionRelief and optimizeForPersistentKernel.
//   oracle_threads DEVICE [STEP]   every step (or the one named), each in a child of its own
// Each step compiles its kernels from source salted with its own number, so neither Metal's
// pipelines of this process nor its shader cache on disk hold them: Metal answers a first compile
// (a state of a kernel compiled before can answer otherwise). The kernels:
//   plain   no attribute
//   max64   [[max_total_threads_per_threadgroup(64)]]
//   max256  [[max_total_threads_per_threadgroup(256)]]
// A step is `KERNEL[-fn][-reqWxHxD][-mtN][-mult][-fpN][-crN][-persist][-warm][-async][-reflect]`:
//   -fn         newComputePipelineStateWithFunction:error: (else a descriptor's)
//   -reqWxHxD   requiredThreadsPerThreadgroup W by H by D
//   -mtN        maxTotalThreadsPerThreadgroup N
//   -mult       threadGroupSizeIsMultipleOfThreadExecutionWidth YES
//   -fpN, -crN  forwardProgressUsage N, contentionRelief N (where the descriptor has them)
//   -persist    optimizeForPersistentKernel YES (where the descriptor has it)
//   -warm       first makes the kernel's pipeline of a plain descriptor, then the step's
//   -async      newComputePipelineStateWithDescriptor:options:completionHandler: (options 0)
//   -reflect    newComputePipelineStateWithDescriptor:options:reflection:error: (options 3)
// and prints
//   STEP: state=made|nil error=DOMAIN/CODE "TEXT"
//     state: tg=maxTotalThreadsPerThreadgroup w=threadExecutionWidth req=WxHxD fp=N
// A step whose child died says how; Metal's words are then on stderr. `device` prints the device's
// limits. Build: see README.md.
#import "oracle_common.h"
#import <objc/message.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

extern char **environ;

static id<MTLDevice> D;

static NSString *source(unsigned salt)
{
	return [NSString
	    stringWithFormat:
	        @"#include <metal_stdlib>\n"
	        @"using namespace metal;\n"
	        @"kernel void plain(device uint *o [[buffer(0)]], uint g [[thread_position_in_grid]]) {\n"
	        @"  o[g] = %uu + g; }\n"
	        @"[[max_total_threads_per_threadgroup(64)]]\n"
	        @"kernel void max64(device uint *o [[buffer(0)]], uint g [[thread_position_in_grid]]) {\n"
	        @"  o[g] = %uu + 2 * g; }\n"
	        @"[[max_total_threads_per_threadgroup(256)]]\n"
	        @"kernel void max256(device uint *o [[buffer(0)]], uint g [[thread_position_in_grid]]) {\n"
	        @"  o[g] = %uu + 3 * g; }\n",
	        salt, salt, salt];
}

static void print_error(NSError *e)
{
	if (!e) {
		printf(" error=nil\n");
		return;
	}
	NSString *text = [e.localizedDescription stringByReplacingOccurrencesOfString:@"\n"
	                                                                    withString:@"\\n"];
	printf(" error=%s/%ld \"%s\"\n", e.domain.UTF8String, (long)e.code, text.UTF8String);
}

static BOOL has(id o, const char *name)
{
	return [o respondsToSelector:sel_registerName(name)];
}

static void print_state(id<MTLComputePipelineState> p)
{
	printf("    state: tg=%lu w=%lu", (unsigned long)p.maxTotalThreadsPerThreadgroup,
	       (unsigned long)p.threadExecutionWidth);
	if (has(p, "requiredThreadsPerThreadgroup")) {
		MTLSize s = p.requiredThreadsPerThreadgroup;
		printf(" req=%lux%lux%lu", (unsigned long)s.width, (unsigned long)s.height,
		       (unsigned long)s.depth);
	} else {
		printf(" req=absent");
	}
	if (has(p, "forwardProgressUsage"))
		printf(" fp=%ld", ((NSInteger(*)(id, SEL))objc_msgSend)(p, sel_registerName("forwardProgressUsage")));
	printf("\n");
}

// The value after `key` in `step` (`-req8x8x1`: "8x8x1"), or NULL.
static const char *option(const char *step, const char *key)
{
	const char *at = strstr(step, key);
	return at ? at + strlen(key) : NULL;
}

static void step_make(const char *step, unsigned salt)
{
	MTLCompileOptions *o = [MTLCompileOptions new];
	o.languageVersion = MTLLanguageVersion3_0;
	NSError *e = nil;
	id<MTLLibrary> lib = [D newLibraryWithSource:source(salt) options:o error:&e];
	if (!lib) {
		printf("%s: library", step);
		print_error(e);
		return;
	}
	char kernel[16] = {0};
	size_t n = strcspn(step, "-");
	memcpy(kernel, step, n < sizeof kernel ? n : sizeof kernel - 1);
	id<MTLFunction> f = [lib newFunctionWithName:@(kernel)];
	if (!f) {
		printf("%s: no function %s\n", step, kernel);
		return;
	}
	if (option(step, "-warm")) {
		MTLComputePipelineDescriptor *w = [MTLComputePipelineDescriptor new];
		w.computeFunction = f;
		NSError *we = nil;
		id<MTLComputePipelineState> first = [D newComputePipelineStateWithDescriptor:w
		                                                                      options:0
		                                                                   reflection:nil
		                                                                        error:&we];
		printf("%s: warm state=%s", step, first ? "made" : "nil");
		print_error(we);
		if (first)
			print_state(first);
	}
	id<MTLComputePipelineState> state = nil;
	e = nil;
	if (option(step, "-fn")) {
		state = [D newComputePipelineStateWithFunction:f error:&e];
	} else {
		MTLComputePipelineDescriptor *d = [MTLComputePipelineDescriptor new];
		d.computeFunction = f;
		const char *v;
		if ((v = option(step, "-req"))) {
			unsigned long w = 0, h = 0, z = 0;
			sscanf(v, "%lux%lux%lu", &w, &h, &z);
			if (!has(d, "requiredThreadsPerThreadgroup")) {
				printf("%s: no requiredThreadsPerThreadgroup\n", step);
				return;
			}
			d.requiredThreadsPerThreadgroup = MTLSizeMake(w, h, z);
		}
		if ((v = option(step, "-mt")))
			d.maxTotalThreadsPerThreadgroup = strtoul(v, NULL, 10);
		if (option(step, "-mult"))
			d.threadGroupSizeIsMultipleOfThreadExecutionWidth = YES;
		const struct {
			const char *key, *setter;
		} kInts[] = {{"-fp", "setForwardProgressUsage:"}, {"-cr", "setContentionRelief:"}};
		for (size_t i = 0; i < sizeof kInts / sizeof kInts[0]; i++) {
			if (!(v = option(step, kInts[i].key)))
				continue;
			SEL s = sel_registerName(kInts[i].setter);
			if (![d respondsToSelector:s]) {
				printf("%s: no %s\n", step, kInts[i].setter);
				return;
			}
			((void (*)(id, SEL, NSInteger))objc_msgSend)(d, s, strtol(v, NULL, 10));
		}
		if (option(step, "-persist")) {
			SEL s = sel_registerName("setOptimizeForPersistentKernel:");
			if (![d respondsToSelector:s]) {
				printf("%s: no setOptimizeForPersistentKernel:\n", step);
				return;
			}
			((void (*)(id, SEL, BOOL))objc_msgSend)(d, s, YES);
		}
		if (option(step, "-async")) {
			dispatch_semaphore_t done = dispatch_semaphore_create(0);
			__block id<MTLComputePipelineState> got = nil;
			__block NSError *gotError = nil;
			[D newComputePipelineStateWithDescriptor:d
			                                 options:0
			                       completionHandler:^(id<MTLComputePipelineState> s,
			                                           MTLComputePipelineReflection *r, NSError *err) {
				                       (void)r;
				                       got = s;
				                       gotError = err;
				                       dispatch_semaphore_signal(done);
			                       }];
			dispatch_semaphore_wait(done, DISPATCH_TIME_FOREVER);
			state = got;
			e = gotError;
		} else if (option(step, "-reflect")) {
			MTLAutoreleasedComputePipelineReflection r = nil;
			state = [D newComputePipelineStateWithDescriptor:d options:3 reflection:&r error:&e];
			printf("%s: reflection=%s\n", step, r ? "made" : "nil");
		} else {
			state = [D newComputePipelineStateWithDescriptor:d options:0 reflection:nil error:&e];
		}
	}
	printf("%s: state=%s", step, state ? "made" : "nil");
	print_error(e);
	if (state)
		print_state(state);
}

static void step_device(void)
{
	MTLSize s = D.maxThreadsPerThreadgroup;
	printf("device: maxThreadsPerThreadgroup=%lux%lux%lu maxThreadgroupMemoryLength=%lu\n",
	       (unsigned long)s.width, (unsigned long)s.height, (unsigned long)s.depth,
	       (unsigned long)D.maxThreadgroupMemoryLength);
}

static void run_child(const char *argv0, const char *dev, const char *step, unsigned salt)
{
	char saltText[16];
	snprintf(saltText, sizeof saltText, "%u", salt);
	char *args[] = {(char *)argv0, (char *)dev, "--step", (char *)step, saltText, NULL};
	pid_t pid;
	if (posix_spawn(&pid, argv0, NULL, NULL, args, environ)) {
		printf("  ! spawn failed\n");
		return;
	}
	int st;
	waitpid(pid, &st, 0);
	if (WIFSIGNALED(st))
		printf("%s: died on signal %d\n", step, WTERMSIG(st));
	else if (WEXITSTATUS(st))
		printf("%s: exited %d\n", step, WEXITSTATUS(st));
}

int main(int argc, char **argv)
{
	setvbuf(stdout, NULL, _IOLBF, 0);
	if (argc < 2) {
		fprintf(stderr, "usage: %s DEVICE [STEP]\n", argv[0]);
		return 2;
	}
	D = oracle_device(argv[1]);
	if (argc > 4 && !strcmp(argv[2], "--step")) {
		@autoreleasepool {
			if (!strcmp(argv[3], "device"))
				step_device();
			else
				step_make(argv[3], (unsigned)strtoul(argv[4], NULL, 10));
		}
		return 0;
	}
	oracle_header("oracle_threads", D);
	printf("# MTL_DEBUG_LAYER=%s\n", getenv("MTL_DEBUG_LAYER") ? getenv("MTL_DEBUG_LAYER") : "unset");
	D = nil;
	// A salt of this run: the time and the process, so no earlier run's cache holds a step's kernels.
	unsigned salt = (unsigned)(time(NULL) * 1000u + (unsigned)getpid() % 1000u);
	if (argc > 2) {
		run_child(argv[0], argv[1], argv[2], salt);
		return 0;
	}
	const char *steps[] = {
		"device",
		// What a kernel's pipeline answers, from the function and from a plain descriptor.
		"plain-fn", "max64-fn", "plain", "max64", "max256",
		// maxTotalThreadsPerThreadgroup against the device's limit and the kernel's attribute.
		"plain-mt32", "plain-mt96", "plain-mt1024", "plain-mt1025", "plain-mt2048",
		"max64-mt32", "max64-mt64", "max64-mt128", "max64-mt1024", "max64-mt2048",
		// requiredThreadsPerThreadgroup alone.
		"plain-req1x1x1", "plain-req7x3x1", "plain-req8x8x2", "plain-req1x1x64", "plain-req32x32x1",
		"plain-req33x32x1", "plain-req64x32x1", "plain-req0x1x1", "plain-req8x0x1", "plain-req0x0x0",
		// ... against the kernel's attribute.
		"max64-req32x1x1", "max64-req64x1x1", "max64-req8x8x1", "max64-req65x1x1", "max64-req128x1x1",
		"max64-req16x8x1", "max64-req1024x1x1", "max256-req16x16x1", "max256-req512x1x1",
		// ... beside maxTotalThreadsPerThreadgroup.
		"plain-req64x1x1-mt32", "plain-req64x1x1-mt64", "plain-req64x1x1-mt128",
		"max64-req64x1x1-mt64", "max64-req32x1x1-mt64", "max64-req32x1x1-mt32", "max64-req128x1x1-mt128",
		// ... beside threadGroupSizeIsMultipleOfThreadExecutionWidth.
		"plain-req3x1x1-mult", "plain-req64x1x1-mult",
		// A kernel compiled before, and the other creators.
		"max64-req128x1x1-warm", "plain-req8x8x2-warm", "max64-req128x1x1-async", "max64-req128x1x1-reflect",
		"plain-req8x8x2-async", "plain-req0x1x1-async",
		// macOS 27's fields.
		"plain-fp0", "plain-fp1", "plain-fp2", "plain-fp3", "plain-fp-1", "plain-cr0", "plain-cr1",
		"plain-cr2", "plain-cr3", "plain-persist", "plain-req64x1x1-persist",
	};
	for (size_t i = 0; i < sizeof steps / sizeof steps[0]; i++)
		run_child(argv[0], argv[1], steps[i], salt + (unsigned)i * 7919u);
	return 0;
}
