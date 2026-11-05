// oracle_names: the words Metal gives where a function's constants or a binary archive's add are
// refused, so the bundle can give the same.
//   oracle_names DEVICE [CASE]   every case (or the one named), each step in a child of its own
// Cases:
//   constants   a kernel with one int function constant at index 0, made with a value of each
//               MTLDataType at that index: whether it is made, and the error. Then a kernel
//               with a bfloat constant (Metal 3.1), its constant's reflected type, made with a
//               BFloat value and with a Float one.
//   archive     a binary archive's addMeshRenderPipelineFunctionsWithDescriptor: with no mesh
//               function: with no function at all, with a fragment function, and with an object
//               function. Run it with MTL_DEBUG_LAYER=1 too: where Metal's validation asserts,
//               its words are on stderr, which the line after the step's says it died.
// Each step prints `case.step: made=0|1 error=DOMAIN/CODE "description"`; a step whose child
// died says how. Build: see README.md.
#import "oracle_common.h"
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

extern char **environ;

static id<MTLDevice> D;

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

// MTLDataType's values (Metal's MTLDataType.h, macOS 27 SDK).
static const unsigned kTypes[] = {
    0,   1,   2,   3,   4,   5,   6,   7,   8,   9,   10,  11,  12,  13,  14,  15,  16,  17,
    18,  19,  20,  21,  22,  23,  24,  25,  26,  27,  28,  29,  30,  31,  32,  33,  34,  35,
    36,  37,  38,  39,  40,  41,  42,  43,  44,  45,  46,  47,  48,  49,  50,  51,  52,  53,
    54,  55,  56,  58,  59,  60,  62,  63,  64,  65,  66,  67,  68,  69,  70,  71,  72,  73,
    74,  75,  76,  77,  78,  79,  80,  81,  82,  83,  84,  85,  86,  87,  88,  115, 116, 117,
    118, 121, 122, 123, 124,
};

static NSString *const kIntSource = @"#include <metal_stdlib>\n"
                                    @"using namespace metal;\n"
                                    @"constant int k [[function_constant(0)]];\n"
                                    @"kernel void f(device int *o [[buffer(0)]]) { o[0] = k; }\n";

static NSString *const kBFloatSource = @"#include <metal_stdlib>\n"
                                       @"using namespace metal;\n"
                                       @"constant bfloat k [[function_constant(0)]];\n"
                                       @"kernel void f(device float *o [[buffer(0)]]) "
                                       @"{ o[0] = float(k); }\n";

static id<MTLLibrary> library(NSString *source, MTLLanguageVersion version)
{
	MTLCompileOptions *o = [MTLCompileOptions new];
	o.languageVersion = version;
	NSError *e = nil;
	id<MTLLibrary> lib = [D newLibraryWithSource:source options:o error:&e];
	if (!lib) {
		printf("  ! library:");
		print_error(e);
	}
	return lib;
}

// Function `f` of `lib` made with `bytes` (zeros) as a value of `type` at index 0.
static void make_with(const char *step, id<MTLLibrary> lib, unsigned type)
{
	static uint8_t zeros[256];
	MTLFunctionConstantValues *v = [MTLFunctionConstantValues new];
	NSError *e = nil;
	id<MTLFunction> fn = nil;
	@try {
		[v setConstantValue:zeros type:(MTLDataType)type atIndex:0];
		fn = [lib newFunctionWithName:@"f" constantValues:v error:&e];
	} @catch (NSException *x) {
		printf("%s type %u: threw %s \"%s\"\n", step, type, x.name.UTF8String,
		       x.reason.UTF8String);
		return;
	}
	printf("%s type %u: made=%d", step, type, fn != nil);
	print_error(e);
}

static void step_constants(unsigned type)
{
	id<MTLLibrary> lib = library(kIntSource, MTLLanguageVersion3_0);
	if (lib)
		make_with("constants.int", lib, type);
}

static void step_bfloat(void)
{
	id<MTLLibrary> lib = library(kBFloatSource, MTLLanguageVersion3_1);
	if (!lib)
		return;
	id<MTLFunction> f = [lib newFunctionWithName:@"f"];
	for (NSString *name in f.functionConstantsDictionary) {
		MTLFunctionConstant *c = f.functionConstantsDictionary[name];
		printf("constants.bfloat reflected %s: type %lu index %lu required %d\n",
		       name.UTF8String, (unsigned long)c.type, (unsigned long)c.index, (int)c.required);
	}
	make_with("constants.bfloat", lib, MTLDataTypeBFloat);
	make_with("constants.bfloat", lib, MTLDataTypeFloat);
}

static void step_archive(const char *variant)
{
	MTLBinaryArchiveDescriptor *ad = [MTLBinaryArchiveDescriptor new];
	NSError *e = nil;
	id<MTLBinaryArchive> a = [D newBinaryArchiveWithDescriptor:ad error:&e];
	if (!a) {
		printf("archive.%s: no archive:", variant);
		print_error(e);
		return;
	}
	MTLMeshRenderPipelineDescriptor *m = [MTLMeshRenderPipelineDescriptor new];
	m.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
	if (strcmp(variant, "nothing") != 0) {
		NSString *src = @"#include <metal_stdlib>\n"
		                @"using namespace metal;\n"
		                @"struct P { float4 p [[position]]; };\n"
		                @"fragment float4 frag(P in [[stage_in]]) { return in.p; }\n"
		                @"[[object]] void obj(mesh_grid_properties g) { g.set_threadgroups_per_grid(uint3(1)); }\n";
		id<MTLLibrary> lib = library(src, MTLLanguageVersion3_0);
		if (!lib)
			return;
		if (!strcmp(variant, "fragment"))
			m.fragmentFunction = [lib newFunctionWithName:@"frag"];
		else
			m.objectFunction = [lib newFunctionWithName:@"obj"];
	}
	e = nil;
	@try {
		BOOL ok = [a addMeshRenderPipelineFunctionsWithDescriptor:m error:&e];
		printf("archive.%s: made=%d", variant, (int)ok);
		print_error(e);
	} @catch (NSException *x) {
		printf("archive.%s: threw %s \"%s\"\n", variant, x.name.UTF8String,
		       x.reason.UTF8String);
	}
}

static void run_child(const char *argv0, const char *dev, const char *c, const char *arg)
{
	char *args[] = {(char *)argv0, (char *)dev, "--step", (char *)c, (char *)arg, NULL};
	pid_t pid;
	if (posix_spawn(&pid, argv0, NULL, NULL, args, environ)) {
		printf("  ! spawn failed\n");
		return;
	}
	int st;
	waitpid(pid, &st, 0);
	if (WIFSIGNALED(st))
		printf("  => step %s %s died on signal %d\n", c, arg ? arg : "", WTERMSIG(st));
	else if (WEXITSTATUS(st))
		printf("  => step %s %s exited %d\n", c, arg ? arg : "", WEXITSTATUS(st));
}

int main(int argc, char **argv)
{
	setvbuf(stdout, NULL, _IOLBF, 0);
	if (argc < 2) {
		fprintf(stderr, "usage: %s DEVICE [constants|archive]\n", argv[0]);
		return 2;
	}
	D = oracle_device(argv[1]);
	if (argc > 3 && !strcmp(argv[2], "--step")) {
		const char *s = argv[3];
		const char *arg = argc > 4 ? argv[4] : "";
		if (!strcmp(s, "constants"))
			step_constants((unsigned)atoi(arg));
		else if (!strcmp(s, "bfloat"))
			step_bfloat();
		else if (!strcmp(s, "archive"))
			step_archive(arg);
		else
			return 2;
		return 0;
	}
	oracle_header("oracle_names", D);
	printf("# MTL_DEBUG_LAYER=%s\n", getenv("MTL_DEBUG_LAYER") ? getenv("MTL_DEBUG_LAYER") : "unset");
	D = nil;
	const char *only = argc > 2 ? argv[2] : NULL;
	if (!only || !strcmp(only, "constants")) {
		char n[16];
		for (size_t i = 0; i < sizeof kTypes / sizeof kTypes[0]; i++) {
			snprintf(n, sizeof n, "%u", kTypes[i]);
			run_child(argv[0], argv[1], "constants", n);
		}
		run_child(argv[0], argv[1], "bfloat", NULL);
	}
	if (!only || !strcmp(only, "archive")) {
		run_child(argv[0], argv[1], "archive", "nothing");
		run_child(argv[0], argv[1], "archive", "fragment");
		run_child(argv[0], argv[1], "archive", "object");
	}
	return 0;
}
