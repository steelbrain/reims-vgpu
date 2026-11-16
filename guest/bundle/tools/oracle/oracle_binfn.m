// oracle_binfn: what a device answers for binary functions, a visible function compiled for the
// device (MTLFunctionOptionCompileToBinary): the device's private compile of a visible function in
// its three forms (compileVisibleFunction:withDescriptor:error:, ...destinationBinaryArchive:error:
// and ...completionHandler:), which Metal's own link of Core Image's kernels sends on macOS 12 and
// 13, what Metal's library sends it when it makes a function with that option, a compute pipeline
// linking binary functions, and newComputePipelineStateWithAdditionalBinaryFunctions:error:. Each
// call of the device's compile is counted and described as Metal makes it (the device class's own
// methods, wrapped). No Metal validation layer is asked for: what a guest's process meets. It runs
// the same on the guest's device.
//   oracle_binfn DEVICE
// Steps and their lines:
//   supports functionPointers => YES|NO
//   responds SEL => YES|NO                  the device's three compile forms
//   compile call FORM => function CLASS NAME type T options O, descriptor CLASS name N options O
//                                           [archive CLASS]   each call the device's compile met
//   binary NAME => CLASS type T options O | nil ERROR; compiles N
//                                           newFunctionWithDescriptor: with CompileToBinary
//   direct NAME OPTS => returned, error ERROR | raises X; compiles N
//                                           the device's compile sent by the probe: of a visible
//                                           function (descriptor options 0 or CompileToBinary), a
//                                           kernel, a nil descriptor
//   direct archive NAME => returned, error ERROR, archive keys K | raises X
//   direct async NAME => handler (A, B) | no handler in 5 s | raises X
//   pipeline LIST => made CLASS | nil ERROR  fp_kernel linking LIST as binaryFunctions or functions
//   handle NAME => CLASS | nil              that pipeline's handle of NAME
//   dispatch NAME => v0 v1 ...              out[i] = table[which[i]](i + 10)
//   additional LIST => made CLASS same|other | nil ERROR | raises X
//                                           newComputePipelineStateWithAdditionalBinaryFunctions:
//   device handle NAME => CLASS name N | nil | raises X | not answered
//                                           the device's functionHandleWithFunction: (macOS 26) of
//                                           a binary function made with and without
//                                           MTLFunctionOptionPipelineIndependent, and
//                                           functionHandleWithFunction:resourceIndex: (SPI)
//   device table N => CLASS | nil | raises X | not answered
//                                           the device's newVisibleFunctionTableWithDescriptor: (SPI)
// Build: build.sh; it loads oracle_fnptr.metallib from beside the executable.
#import "oracle_common.h"
#include <objc/message.h>
#include <malloc/malloc.h>
#include <objc/runtime.h>

static const char *cls(id o)
{
	return o ? class_getName(object_getClass(o)) : "nil";
}

static const char *err_text(NSError *e)
{
	if (!e)
		return "(none)";
	static char buf[512];
	snprintf(buf, sizeof(buf), "%s %ld: %s", e.domain.UTF8String, (long)e.code,
	         e.localizedDescription.UTF8String);
	for (char *p = buf; *p; p++)
		if (*p == '\n')
			*p = '|';
	return buf;
}

static int compiles; // calls of the device's compile seen since the last step reset it

static void describe_call(const char *form, id fn, id desc, id archive)
{
	compiles++;
	printf("compile call %s => function %s %s type %ld options %lu, descriptor %s", form, cls(fn),
	       [fn respondsToSelector:@selector(name)] ? [[fn name] UTF8String] : "?",
	       [fn respondsToSelector:@selector(functionType)] ? (long)[fn functionType] : -1L,
	       [fn respondsToSelector:@selector(options)] ? (unsigned long)[(id<MTLFunction>)fn options] : 0UL,
	       cls(desc));
	if ([desc respondsToSelector:@selector(name)])
		printf(" name %s", [[desc name] UTF8String] ?: "(nil)");
	if ([desc isKindOfClass:[MTLFunctionDescriptor class]])
		printf(" options %lu", (unsigned long)((MTLFunctionDescriptor *)desc).options);
	if (archive)
		printf(" archive %s", cls(archive));
	printf("\n");
}

static IMP orig_plain, orig_archive, orig_async;

// Wraps the device class's compile forms, where it has them, so each call is described.
static void wrap_compiles(id<MTLDevice> d)
{
	Class c = object_getClass(d);
	Method m = class_getInstanceMethod(c, sel_registerName("compileVisibleFunction:withDescriptor:error:"));
	if (m) {
		orig_plain = method_getImplementation(m);
		method_setImplementation(m, imp_implementationWithBlock(^(id self, id fn, id desc, NSError **e) {
			describe_call("plain", fn, desc, nil);
			((void (*)(id, SEL, id, id, NSError **))orig_plain)(
			    self, sel_registerName("compileVisibleFunction:withDescriptor:error:"), fn, desc, e);
		}));
	}
	m = class_getInstanceMethod(
	    c, sel_registerName("compileVisibleFunction:withDescriptor:destinationBinaryArchive:error:"));
	if (m) {
		orig_archive = method_getImplementation(m);
		method_setImplementation(
		    m, imp_implementationWithBlock(^(id self, id fn, id desc, id archive, NSError **e) {
			    describe_call("archive", fn, desc, archive);
			    ((void (*)(id, SEL, id, id, id, NSError **))orig_archive)(
			        self,
			        sel_registerName(
			            "compileVisibleFunction:withDescriptor:destinationBinaryArchive:error:"),
			        fn, desc, archive, e);
		    }));
	}
	m = class_getInstanceMethod(c, sel_registerName("compileVisibleFunction:withDescriptor:completionHandler:"));
	if (m) {
		orig_async = method_getImplementation(m);
		method_setImplementation(m, imp_implementationWithBlock(^(id self, id fn, id desc, id handler) {
			describe_call("async", fn, desc, nil);
			((void (*)(id, SEL, id, id, id))orig_async)(
			    self, sel_registerName("compileVisibleFunction:withDescriptor:completionHandler:"), fn,
			    desc, handler);
		}));
	}
}

static MTLFunctionDescriptor *descriptor(NSString *name, MTLFunctionOptions options)
{
	MTLFunctionDescriptor *fd = [MTLFunctionDescriptor functionDescriptor];
	fd.name = name;
	fd.options = options;
	return fd;
}

static id<MTLFunction> binary(id<MTLLibrary> lib, NSString *name)
{
	compiles = 0;
	NSError *e = nil;
	id<MTLFunction> fn = [lib newFunctionWithDescriptor:descriptor(name, MTLFunctionOptionCompileToBinary)
	                                              error:&e];
	if (fn)
		printf("binary %s => %s type %ld options %lu; compiles %d\n", name.UTF8String, cls(fn),
		       (long)fn.functionType, (unsigned long)fn.options, compiles);
	else
		printf("binary %s => nil %s; compiles %d\n", name.UTF8String, err_text(e), compiles);
	return fn;
}

static void direct(id<MTLDevice> d, const char *what, id<MTLFunction> fn, id desc)
{
	SEL s = sel_registerName("compileVisibleFunction:withDescriptor:error:");
	if (![(id)d respondsToSelector:s]) {
		printf("direct %s => not answered\n", what);
		return;
	}
	compiles = 0;
	NSError *e = nil;
	@try {
		((void (*)(id, SEL, id, id, NSError **))objc_msgSend)(d, s, fn, desc, &e);
		printf("direct %s => returned, error %s; compiles %d\n", what, err_text(e), compiles);
	} @catch (NSException *x) {
		printf("direct %s => raises %s: %s\n", what, x.name.UTF8String, x.reason.UTF8String);
	}
}

static void direct_archive(id<MTLDevice> d, id<MTLFunction> fn, id desc)
{
	SEL s = sel_registerName("compileVisibleFunction:withDescriptor:destinationBinaryArchive:error:");
	if (![(id)d respondsToSelector:s]) {
		printf("direct archive => not answered\n");
		return;
	}
	NSError *e = nil;
	id<MTLBinaryArchive> archive =
	    [d newBinaryArchiveWithDescriptor:[MTLBinaryArchiveDescriptor new] error:&e];
	printf("archive made => %s %s\n", cls(archive), err_text(e));
	e = nil;
	@try {
		((void (*)(id, SEL, id, id, id, NSError **))objc_msgSend)(d, s, fn, desc, archive, &e);
		NSString *keys = [archive respondsToSelector:sel_registerName("keys")]
		                     ? [[(id)archive valueForKey:@"keys"] description]
		                     : @"(not answered)";
		printf("direct archive %s => returned, error %s, archive keys %s\n", [fn name].UTF8String,
		       err_text(e), [[keys stringByReplacingOccurrencesOfString:@"\n" withString:@" "] UTF8String]);
	} @catch (NSException *x) {
		printf("direct archive => raises %s: %s\n", x.name.UTF8String, x.reason.UTF8String);
	}
}

static void direct_async(id<MTLDevice> d, id<MTLFunction> fn, id desc)
{
	SEL s = sel_registerName("compileVisibleFunction:withDescriptor:completionHandler:");
	if (![(id)d respondsToSelector:s]) {
		printf("direct async => not answered\n");
		return;
	}
	dispatch_semaphore_t done = dispatch_semaphore_create(0);
	__block NSString *seen = nil;
	// The handler's arguments are read as raw words: the first is described where it is a heap
	// object (malloc_size), the second only printed, as a one-argument handler leaves it unset.
	void (^handler)(void *, void *) = ^(void *a, void *b) {
		NSString *as = a == NULL ? @"nil"
		               : malloc_size(a) > 0
		                   ? [NSString stringWithFormat:@"%s %s", cls((__bridge id)a),
		                                                [(__bridge id)a isKindOfClass:[NSError class]]
		                                                    ? err_text((__bridge NSError *)a)
		                                                    : ""]
		                   : @"not a heap object";
		seen = [NSString stringWithFormat:@"handler (first %@, second %p) on the main thread %d", as, b,
		                                  (int)[NSThread isMainThread]];
		dispatch_semaphore_signal(done);
	};
	@try {
		((void (*)(id, SEL, id, id, id))objc_msgSend)(d, s, fn, desc, handler);
		if (dispatch_semaphore_wait(done, dispatch_time(DISPATCH_TIME_NOW, 5 * NSEC_PER_SEC)) == 0)
			printf("direct async %s => %s\n", [fn name].UTF8String, seen.UTF8String);
		else
			printf("direct async %s => no handler in 5 s\n", [fn name].UTF8String);
	} @catch (NSException *x) {
		printf("direct async => raises %s: %s\n", x.name.UTF8String, x.reason.UTF8String);
	}
}

static id<MTLComputePipelineState> pipeline(id<MTLDevice> d, id<MTLLibrary> lib, const char *what,
                                            NSArray *binaries, NSArray *functions)
{
	MTLComputePipelineDescriptor *cd = [MTLComputePipelineDescriptor new];
	cd.computeFunction = [lib newFunctionWithName:@"fp_kernel"];
	MTLLinkedFunctions *lf = [MTLLinkedFunctions linkedFunctions];
	lf.binaryFunctions = binaries;
	lf.functions = functions;
	cd.linkedFunctions = lf;
	compiles = 0;
	NSError *e = nil;
	id<MTLComputePipelineState> p = nil;
	@try {
		p = [d newComputePipelineStateWithDescriptor:cd options:MTLPipelineOptionNone reflection:nil error:&e];
		printf("pipeline %s => %s %s; compiles %d\n", what, p ? "made" : "nil", p ? cls(p) : err_text(e),
		       compiles);
	} @catch (NSException *x) {
		printf("pipeline %s => raises %s: %s\n", what, x.name.UTF8String, x.reason.UTF8String);
	}
	return p;
}

static void handle(id<MTLComputePipelineState> p, const char *what, id<MTLFunction> fn)
{
	if (!p || !fn)
		return;
	id<MTLFunctionHandle> h = nil;
	@try {
		h = [p functionHandleWithFunction:fn];
		printf("handle %s => %s%s%s\n", what, cls(h), h ? " name " : "", h ? h.name.UTF8String : "");
	} @catch (NSException *x) {
		printf("handle %s => raises %s\n", what, x.name.UTF8String);
	}
}

// out[i] = table[which[i]](i + 10), the table holding `fns`' handles in order.
static void run(id<MTLDevice> d, id<MTLCommandQueue> q, id<MTLComputePipelineState> p, const char *what,
                NSArray<id<MTLFunction>> *fns)
{
	if (!p)
		return;
	MTLVisibleFunctionTableDescriptor *td = [MTLVisibleFunctionTableDescriptor new];
	td.functionCount = fns.count;
	id<MTLVisibleFunctionTable> t = [p newVisibleFunctionTableWithDescriptor:td];
	for (NSUInteger i = 0; i < fns.count; i++) {
		id<MTLFunctionHandle> h = [p functionHandleWithFunction:fns[i]];
		if (!h) {
			printf("dispatch %s => no handle of %s\n", what, fns[i].name.UTF8String);
			return;
		}
		[t setFunction:h atIndex:i];
	}
	enum { n = 4 };
	uint32_t which[n];
	for (uint32_t i = 0; i < n; i++)
		which[i] = i % (uint32_t)fns.count;
	id<MTLBuffer> w = [d newBufferWithBytes:which length:sizeof(which) options:MTLResourceStorageModeShared];
	id<MTLBuffer> out = [d newBufferWithLength:n * 4 options:MTLResourceStorageModeShared];
	memset(out.contents, 0xa5, n * 4);
	id<MTLCommandBuffer> cb = [q commandBuffer];
	id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
	[e setComputePipelineState:p];
	[e setVisibleFunctionTable:t atBufferIndex:0];
	[e setBuffer:w offset:0 atIndex:1];
	[e setBuffer:out offset:0 atIndex:2];
	[e dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(n, 1, 1)];
	[e endEncoding];
	oracle_commit_wait(cb, NULL, NULL);
	printf("dispatch %s =>", what);
	const int32_t *o = out.contents;
	for (int i = 0; i < n; i++)
		printf(" %d", o[i]);
	printf("\n");
}

static id<MTLComputePipelineState> additional(id<MTLComputePipelineState> p, const char *what,
                                              NSArray *fns)
{
	if (!p)
		return nil;
	compiles = 0;
	NSError *e = nil;
	id<MTLComputePipelineState> more = nil;
	@try {
		more = [p newComputePipelineStateWithAdditionalBinaryFunctions:fns error:&e];
		printf("additional %s => %s %s%s; compiles %d\n", what, more ? "made" : "nil",
		       more ? cls(more) : err_text(e), more ? (more == p ? " same" : " other") : "", compiles);
	} @catch (NSException *x) {
		printf("additional %s => raises %s: %s\n", what, x.name.UTF8String, x.reason.UTF8String);
	}
	return more;
}

static void device_handle(id<MTLDevice> d, const char *what, id<MTLFunction> fn)
{
	SEL s = sel_registerName("functionHandleWithFunction:");
	if (![(id)d respondsToSelector:s] || !fn) {
		printf("device handle %s => not answered\n", what);
		return;
	}
	@try {
		id h = ((id (*)(id, SEL, id))objc_msgSend)(d, s, fn);
		printf("device handle %s => %s%s%s\n", what, cls(h), h ? " name " : "",
		       h ? [[h name] UTF8String] : "");
	} @catch (NSException *x) {
		printf("device handle %s => raises %s: %s\n", what, x.name.UTF8String, x.reason.UTF8String);
	}
}

static void device_handle_index(id<MTLDevice> d, const char *what, id<MTLFunction> fn)
{
	SEL s = sel_registerName("functionHandleWithFunction:resourceIndex:");
	if (![(id)d respondsToSelector:s] || !fn) {
		printf("device handle %s => not answered\n", what);
		return;
	}
	@try {
		id h = ((id (*)(id, SEL, id, NSUInteger))objc_msgSend)(d, s, fn, 0);
		printf("device handle %s => %s%s%s\n", what, cls(h), h ? " name " : "",
		       h ? [[h name] UTF8String] : "");
	} @catch (NSException *x) {
		printf("device handle %s => raises %s: %s\n", what, x.name.UTF8String, x.reason.UTF8String);
	}
}

static void device_table(id<MTLDevice> d, NSUInteger n)
{
	SEL s = sel_registerName("newVisibleFunctionTableWithDescriptor:");
	if (![(id)d respondsToSelector:s]) {
		printf("device table %lu => not answered\n", (unsigned long)n);
		return;
	}
	MTLVisibleFunctionTableDescriptor *td = [MTLVisibleFunctionTableDescriptor new];
	td.functionCount = n;
	@try {
		id t = ((id (*)(id, SEL, id))objc_msgSend)(d, s, td);
		printf("device table %lu => %s\n", (unsigned long)n, cls(t));
	} @catch (NSException *x) {
		printf("device table %lu => raises %s: %s\n", (unsigned long)n, x.name.UTF8String,
		       x.reason.UTF8String);
	}
}

int main(int argc, char **argv)
{
	setvbuf(stdout, NULL, _IOLBF, 0);
	@autoreleasepool {
		if (argc < 2) {
			fprintf(stderr, "usage: %s DEVICE\n", argv[0]);
			return 2;
		}
		id<MTLDevice> d = oracle_device(argv[1]);
		oracle_header("oracle_binfn", d);
		printf("supports functionPointers => %s\n", d.supportsFunctionPointers ? "YES" : "NO");
		const char *sels[] = {"compileVisibleFunction:withDescriptor:error:",
		                      "compileVisibleFunction:withDescriptor:destinationBinaryArchive:error:",
		                      "compileVisibleFunction:withDescriptor:completionHandler:"};
		for (int i = 0; i < 3; i++)
			printf("responds %s => %s\n", sels[i],
			       [(id)d respondsToSelector:sel_registerName(sels[i])] ? "YES" : "NO");
		wrap_compiles(d);
		id<MTLCommandQueue> q = [d newCommandQueue];
		id<MTLLibrary> lib = oracle_library(d, argv[0], "oracle_fnptr");
		if (!lib)
			return 1;
		// Metal's library making binary functions.
		id<MTLFunction> add1b = binary(lib, @"fp_add1");
		id<MTLFunction> mul2b = binary(lib, @"fp_mul2");
		id<MTLFunction> negb = binary(lib, @"fp_neg");
		binary(lib, @"fp_kernel");
		// The device's compile, sent as Metal's own link sends it.
		id<MTLFunction> mul2 = [lib newFunctionWithName:@"fp_mul2"];
		id<MTLFunction> neg = [lib newFunctionWithName:@"fp_neg"];
		direct(d, "fp_mul2 options 0", mul2, descriptor(@"fp_mul2", 0));
		direct(d, "fp_mul2 CompileToBinary", mul2, descriptor(@"fp_mul2", MTLFunctionOptionCompileToBinary));
		direct(d, "fp_mul2 nil descriptor", mul2, nil);
		direct(d, "fp_kernel CompileToBinary", [lib newFunctionWithName:@"fp_kernel"],
		       descriptor(@"fp_kernel", MTLFunctionOptionCompileToBinary));
		direct(d, "fp_add1 binary again", add1b, descriptor(@"fp_add1", MTLFunctionOptionCompileToBinary));
		direct_archive(d, neg, descriptor(@"fp_neg", MTLFunctionOptionCompileToBinary));
		direct_async(d, neg, descriptor(@"fp_neg", MTLFunctionOptionCompileToBinary));
		// Pipelines over binary functions.
		id<MTLComputePipelineState> pb = add1b && mul2b ? pipeline(d, lib, "binary add1 mul2", @[ add1b, mul2b ], @[])
		                                                : nil;
		handle(pb, "add1 binary", add1b);
		handle(pb, "neg binary", negb);
		run(d, q, pb, "binary add1 mul2", @[ add1b, mul2b ]);
		id<MTLComputePipelineState> pp = pipeline(d, lib, "plain-listed binary", @[ mul2 ], @[]);
		run(d, q, pp, "plain-listed binary", @[ mul2 ]);
		id<MTLComputePipelineState> pf = pipeline(d, lib, "functions add1", @[], @[ [lib newFunctionWithName:@"fp_add1"] ]);
		// Pipelines with more binary functions.
		id<MTLComputePipelineState> more = negb ? additional(pb, "neg binary", @[ negb ]) : nil;
		handle(more, "neg binary on the new", negb);
		handle(pb, "neg binary on the old", negb);
		if (more && negb)
			run(d, q, more, "additional neg", @[ add1b, mul2b, negb ]);
		additional(pb, "neg plain", @[ neg ]);
		additional(pb, "empty", @[]);
		additional(pf, "neg binary over functions", negb ? @[ negb ] : @[]);
		// The device's own handles and tables, last: a device may end the process at one.
		id<MTLFunction> indep = nil;
		{
			NSError *e = nil;
			indep = [lib newFunctionWithDescriptor:descriptor(@"fp_add1", MTLFunctionOptionCompileToBinary | (1 << 3))
			                                 error:&e];
			printf("binary fp_add1 pipeline-independent => %s %s\n", cls(indep), err_text(e));
		}
		device_handle(d, "fp_add1 pipeline-independent", indep);
		device_handle(d, "fp_add1 binary", add1b);
		device_handle_index(d, "fp_add1 binary at 0", add1b);
		device_table(d, 3);
	}
	return 0;
}
