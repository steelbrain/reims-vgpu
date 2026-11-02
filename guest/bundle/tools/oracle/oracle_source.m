// oracle_source: what a device's runtime source compile makes, and whether Metal's own frontend
// makes it on the device too: the library newLibraryWithSource:options:error: answers, its class,
// the bytes it holds (libraryDataContents, Metal's own library's: a .metallib) and their kind; the
// same through the implementation of the nearest superclass that is Metal's own (`_MTLDevice` or
// `IOGPUMetalDevice`), called on the device itself; and through that class's compiler-task form,
// newLibraryWithSource:options:compilerTask:error: with no task, which the base form calls on the
// device (measured: macOS 26.7.1 and 27, AMD, Intel and the M3, each reaching the compiler service
// through _MTLDevice's getCompilerConnectionManager:, never the device's -compiler). On a real Mac
// the three are one route; on the reims-graphics device the public selectors are the bundle's, and
// the compiler-task form called on the base class is Metal's frontend over the device with none of
// the bundle's library methods in its way. Also the device's `compiler` (SPI).
//   oracle_source DEVICE [OUT]   DEVICE a substring of the device's name ("" for the first);
//                                OUT, where given, a directory each library's bytes are written
//                                to (`public.metallib`, `base.metallib`, `task.metallib`), to
//                                compare across devices
// Lines are `key=value`. A selector the object does not respond to prints n/r and is not sent.
// Build: see README.md.
#import "oracle_common.h"
#include <objc/message.h>
#include <objc/runtime.h>

// A small kernel and a vertex and fragment pair, as ANGLE's and WebKit's translators emit
// such functions at run time.
static NSString *const kSource =
    @"#include <metal_stdlib>\n"
    @"using namespace metal;\n"
    @"struct V { float4 p [[position]]; float2 uv; };\n"
    @"vertex V vmain(uint i [[vertex_id]], constant float2 *xy [[buffer(0)]]) {\n"
    @"  V v; v.p = float4(xy[i], 0, 1); v.uv = xy[i] * 0.5 + 0.5; return v; }\n"
    @"fragment float4 fmain(V v [[stage_in]], texture2d<float> t [[texture(0)]],\n"
    @"                      sampler s [[sampler(0)]]) { return t.sample(s, v.uv); }\n"
    @"kernel void kmain(device float *o [[buffer(0)]], uint i [[thread_position_in_grid]]) {\n"
    @"  o[i] = float(i) * 2.0f; }\n";

// The bytes `lib` holds, Metal's own library's (nil where it answers none).
static NSData *library_bytes(id lib)
{
	SEL sel = sel_registerName("libraryDataContents");
	if (lib == nil || ![lib respondsToSelector:sel])
		return nil;
	return ((id(*)(id, SEL))objc_msgSend)(lib, sel);
}

// What `bytes` are: a .metallib's magic and size, and whether LLVM bitcode (AIR) is in it.
static void describe(const char *tag, id lib, NSError *error, NSData *bytes, const char *out,
                     const char *file)
{
	printf("%s.library=%s\n", tag, lib ? class_getName(object_getClass(lib)) : "nil");
	if (error)
		printf("%s.error=%s %ld %s\n", tag, error.domain.UTF8String, (long)error.code,
		       [error.localizedDescription stringByReplacingOccurrencesOfString:@"\n"
		                                                            withString:@" | "]
		           .UTF8String);
	if (lib) {
		NSArray<NSString *> *names = [[(id<MTLLibrary>)lib functionNames]
		    sortedArrayUsingSelector:@selector(compare:)];
		printf("%s.functions=%s\n", tag, [names componentsJoinedByString:@","].UTF8String);
	}
	if (lib && ![lib respondsToSelector:sel_registerName("libraryDataContents")]) {
		printf("%s.bytes=n/r\n", tag);
		return;
	}
	if (bytes == nil) {
		if (lib)
			printf("%s.bytes=nil\n", tag);
		return;
	}
	const uint8_t *b = bytes.bytes;
	char magic[5] = {0};
	if (bytes.length >= 4)
		memcpy(magic, b, 4);
	// LLVM bitcode, raw (BC C0 DE) or in its wrapper (DE C0 17 0B).
	BOOL bitcode = NO;
	for (NSUInteger i = 0; i + 4 <= bytes.length && !bitcode; i++)
		bitcode = (b[i] == 0x42 && b[i + 1] == 0x43 && b[i + 2] == 0xc0 && b[i + 3] == 0xde) ||
		          (b[i] == 0xde && b[i + 1] == 0xc0 && b[i + 2] == 0x17 && b[i + 3] == 0x0b);
	printf("%s.bytes=%lu magic=%s bitcode=%d\n", tag, (unsigned long)bytes.length, magic, bitcode);
	if (out != NULL) {
		NSString *path = [NSString stringWithFormat:@"%s/%s", out, file];
		printf("%s.written=%d %s\n", tag, [bytes writeToFile:path atomically:YES], path.UTF8String);
	}
}

// The nearest superclass of `cls` whose name starts with an underscore or `IOGPU`: Metal's own
// device class the device is made of.
static Class metal_base(Class cls)
{
	for (Class c = cls; c != Nil; c = class_getSuperclass(c)) {
		const char *n = class_getName(c);
		if (!strcmp(n, "_MTLDevice") || !strncmp(n, "IOGPU", 5))
			return c;
	}
	return Nil;
}

int main(int argc, char **argv)
{
	setvbuf(stdout, NULL, _IOLBF, 0);
	if (argc < 2) {
		fprintf(stderr, "usage: %s DEVICE [OUT]\n", argv[0]);
		return 2;
	}
	const char *out = argc > 2 ? argv[2] : NULL;
	@autoreleasepool {
		id<MTLDevice> d = oracle_device(argv[1]);
		oracle_header("oracle_source", d);
		for (Class c = object_getClass(d); c != Nil; c = class_getSuperclass(c))
			printf("class=%s\n", class_getName(c));
		SEL compilerSel = sel_registerName("compiler");
		if ([(id)d respondsToSelector:compilerSel]) {
			id compiler = ((id(*)(id, SEL))objc_msgSend)(d, compilerSel);
			printf("compiler=%s\n", compiler ? class_getName(object_getClass(compiler)) : "nil");
		} else {
			printf("compiler=n/r\n");
		}

		MTLCompileOptions *options = [MTLCompileOptions new];
		SEL sel = @selector(newLibraryWithSource:options:error:);
		NSError *error = nil;
		double t0 = oracle_now_ms();
		id<MTLLibrary> lib = [d newLibraryWithSource:kSource options:options error:&error];
		printf("public.ms=%.1f\n", oracle_now_ms() - t0);
		describe("public", lib, error, library_bytes(lib), out, "public.metallib");

		Class base = metal_base(object_getClass(d));
		printf("base=%s\n", base ? class_getName(base) : "none");
		if (base != Nil && class_getInstanceMethod(base, sel) != NULL) {
			IMP imp = class_getMethodImplementation(base, sel);
			BOOL own = imp == class_getMethodImplementation(object_getClass(d), sel);
			printf("base.same_as_public=%d\n", own);
			error = nil;
			t0 = oracle_now_ms();
			id<MTLLibrary> blib = ((id(*)(id, SEL, NSString *, MTLCompileOptions *, NSError **))imp)(
			    d, sel, kSource, options, &error);
			printf("base.ms=%.1f\n", oracle_now_ms() - t0);
			describe("base", blib, error, library_bytes(blib), out, "base.metallib");
		} else {
			printf("base.selector=n/r\n");
		}

		SEL tsel = sel_registerName("newLibraryWithSource:options:compilerTask:error:");
		if (base != Nil && class_getInstanceMethod(base, tsel) != NULL) {
			IMP imp = class_getMethodImplementation(base, tsel);
			printf("task.device_overrides=%d\n",
			       imp != class_getMethodImplementation(object_getClass(d), tsel));
			error = nil;
			t0 = oracle_now_ms();
			id<MTLLibrary> tlib = ((id(*)(id, SEL, NSString *, MTLCompileOptions *, id, NSError **))imp)(
			    d, tsel, kSource, options, nil, &error);
			printf("task.ms=%.1f\n", oracle_now_ms() - t0);
			describe("task", tlib, error, library_bytes(tlib), out, "task.metallib");
		} else {
			printf("task.selector=n/r\n");
		}
	}
	return 0;
}
