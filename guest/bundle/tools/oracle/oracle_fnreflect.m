// oracle_fnreflect: what a library's functions answer of their own reflection, and what pipeline
// states answer of themselves, so the bundle can answer the same.
//   oracle_fnreflect DEVICE [LIB...]
// DEVICE is a substring of the device's name (AMD, Intel, Apple; "" takes the first device). The
// libraries, each made from its URL next to the probe: oracle_fnreflect.metallib and
// oracle_fnreflect_11.metallib (oracle_fnreflect.metal built for macOS 26 and for macOS 11) and
// oracle_reflection.metallib; then oracle_fnreflect.metallib again from its bytes
// (`oracle_fnreflect.data`), a library compiled from source at run time (`source`), and each
// LIB.metallib named, next to the probe. For each,
//   L LIB functions=N
// then for each of its functions, in name order:
//   F LIB/FUNCTION type=T patch=P/N options=O
//   LIB/FUNCTION.vertexAttributes = VALUE        and the walk of what it gives (tools/oracle/
//   LIB/FUNCTION.stageInputAttributes = VALUE      oracle_walk.h: every getter of no argument,
//   LIB/FUNCTION.functionConstantsDictionary = dictionary of N   recursively; each constant
//                                                  by name, as `...[NAME]`)
//   LIB/FUNCTION.reflection = VALUE              the library's reflectionForFunctionWithName:
//                                                  ("not answered" where it does not respond)
//   LIB/FUNCTION.functionReflection = VALUE      the function's SPI reflectionWithOptions: of 3,
//                                                  "same as .reflection" where its walk is the
//                                                  library's
// and for a name the library does not hold, `LIB/nosuch.reflection = VALUE`. Then functions
// specialised by constant values, the same getters under `S NAME`: v_gated of kOffset true and
// kColor false (`v_gated.offset`), of both true (`v_gated.both`) and of both false
// (`v_gated.none`), v_deadread of kColor false and true (`.off`, `.on`), and oracle_reflection's
// k_constants of kOn true, kOff false and kIndex 5 and v_specialised of kMode 1, 0 and 2.
// Then pipeline states, `P CASE made|nil error=...`, then each property of the state's protocol
// and of the protocols it adopts (allocatedSize is MTLAllocation's) that gives an object or a
// scalar, as `P CASE.GETTER = VALUE`, and `P CASE.imageblockMemoryLengthForDimensions:8x8x1 =
// VALUE` ("asserts" where Metal ends the process there):
//   c_tg       k_tg
//   c_stage    k_stage, through a descriptor of a stage input descriptor of its attributes
//   r_plain    v_plain and f_in, one BGRA8 colour attachment
//   r_mixed    v_mixed and f_in, a vertex descriptor of its attributes
//   r_vertex   oracle_reflection's v_only, rasterization disabled
//   r_mesh     oracle_reflection's o_payload, m_payload and f_mesh
// Last, for every class the walks met and the attribute, constant and function reflection classes
// Metal has, its instance variables and methods, under `class NAME : SUPER`. Build: see README.md.
#import "oracle_common.h"
#import "oracle_walk.h"

static id<MTLDevice> D;

static void print_error(NSError *e)
{
	if (!e) {
		printf(" error=nil");
		return;
	}
	NSString *t = [e.localizedDescription stringByReplacingOccurrencesOfString:@"\n" withString:@"\\n"];
	printf(" error=%s/%ld \"%s\"", e.domain.UTF8String, (long)e.code, t.UTF8String);
}

static void print_lines(NSMutableArray<NSString *> *lines)
{
	if (Ivars)
		resolve_ivars(lines);
	for (NSString *l in lines)
		printf("%s\n", l.UTF8String);
}

// `path = VALUE` for the getter `name` of `o`, then the walk of what it gives; a dictionary's
// entries by key.
static void getter(id o, const char *name, NSString *path)
{
	SEL s = sel_registerName(name);
	if (![o respondsToSelector:s]) {
		printf("%s = not answered\n", path.UTF8String);
		return;
	}
	NSString *v = safely(o, s, "@");
	if ([v isEqualToString:@"asserts"]) {
		printf("%s = asserts\n", path.UTF8String);
		return;
	}
	id obj = ((id(*)(id, SEL))objc_msgSend)(o, s);
	if ([obj isKindOfClass:NSDictionary.class]) {
		NSDictionary *d = obj;
		printf("%s = dictionary of %lu\n", path.UTF8String, (unsigned long)d.count);
		for (id key in [d.allKeys sortedArrayUsingSelector:@selector(compare:)]) {
			NSString *at = [NSString stringWithFormat:@"%@[%@]", path, key];
			printf("%s class %s\n", at.UTF8String, class_getName([d[key] class]));
			NSMutableArray<NSString *> *lines = [NSMutableArray array];
			walk(d[key], at, 1, lines);
			print_lines(lines);
		}
		return;
	}
	printf("%s = %s\n", path.UTF8String, v.UTF8String);
	NSMutableArray<NSString *> *lines = [NSMutableArray array];
	walk(obj, path, 1, lines);
	print_lines(lines);
}

// The library's reflectionForFunctionWithName: of `name`, and its walk; the walk's lines, each
// after `path`, in `*walked`.
static void library_reflection(id<MTLLibrary> lib, NSString *name, NSString *path, NSArray<NSString *> **walked)
{
	SEL s = sel_registerName("reflectionForFunctionWithName:");
	if (![(id)lib respondsToSelector:s]) {
		printf("%s = not answered\n", path.UTF8String);
		return;
	}
	id r = ((id(*)(id, SEL, id))objc_msgSend)(lib, s, name);
	printf("%s = %s\n", path.UTF8String, r ? [NSString stringWithFormat:@"<%s>", class_getName([r class])].UTF8String : "nil");
	NSMutableArray<NSString *> *lines = [NSMutableArray array];
	walk(r, path, 1, lines);
	print_lines(lines);
	if (walked != NULL) {
		NSMutableArray<NSString *> *tails = [NSMutableArray array];
		for (NSString *l in lines)
			[tails addObject:[l substringFromIndex:path.length]];
		*walked = tails;
	}
}

// The function's own reflection, the SPI reflectionWithOptions: of 3, against the library's
// (`libraryLines`, its lines after their path): "same as .reflection" where the walks agree.
static void function_reflection(id<MTLFunction> f, NSString *path, NSArray<NSString *> *libraryLines)
{
	SEL s = sel_registerName("reflectionWithOptions:");
	if (![(id)f respondsToSelector:s]) {
		printf("%s = not answered\n", path.UTF8String);
		return;
	}
	id r = ((id(*)(id, SEL, NSUInteger))objc_msgSend)(f, s, 3);
	NSMutableArray<NSString *> *lines = [NSMutableArray array];
	walk(r, path, 1, lines);
	NSMutableArray<NSString *> *tails = [NSMutableArray array];
	for (NSString *l in lines)
		[tails addObject:[l substringFromIndex:path.length]];
	if (r != nil && libraryLines != nil && [tails isEqualToArray:libraryLines]) {
		printf("%s = same as .reflection\n", path.UTF8String);
		return;
	}
	printf("%s = %s\n", path.UTF8String, r ? [NSString stringWithFormat:@"<%s>", class_getName([r class])].UTF8String : "nil");
	print_lines(lines);
}

static void function_answers(id<MTLLibrary> lib, id<MTLFunction> f, NSString *at)
{
	printf("F %s type=%lu patch=%lu/%ld options=%lu\n", at.UTF8String, (unsigned long)f.functionType,
	       (unsigned long)f.patchType, (long)f.patchControlPointCount, (unsigned long)f.options);
	getter(f, "vertexAttributes", [at stringByAppendingString:@".vertexAttributes"]);
	getter(f, "stageInputAttributes", [at stringByAppendingString:@".stageInputAttributes"]);
	getter(f, "functionConstantsDictionary", [at stringByAppendingString:@".functionConstantsDictionary"]);
	NSArray<NSString *> *libraryLines = nil;
	if (lib != nil)
		library_reflection(lib, f.name, [at stringByAppendingString:@".reflection"], &libraryLines);
	function_reflection(f, [at stringByAppendingString:@".functionReflection"], libraryLines);
}

static void library_answers(NSString *tag, id<MTLLibrary> lib)
{
	if (lib == nil) {
		printf("L %s none\n", tag.UTF8String);
		return;
	}
	NSArray<NSString *> *names = [lib.functionNames sortedArrayUsingSelector:@selector(compare:)];
	printf("L %s functions=%lu class=%s\n", tag.UTF8String, (unsigned long)names.count, class_getName([(id)lib class]));
	for (NSString *name in names) {
		id<MTLFunction> f = [lib newFunctionWithName:name];
		NSString *at = [NSString stringWithFormat:@"%@/%@", tag, name];
		if (f == nil) {
			printf("F %s none\n", at.UTF8String);
			continue;
		}
		function_answers(lib, f, at);
	}
	library_reflection(lib, @"nosuch", [tag stringByAppendingString:@"/nosuch.reflection"], NULL);
}

static id<MTLLibrary> url_library(const char *argv0, const char *name)
{
	id<MTLLibrary> lib = oracle_library(D, argv0, name);
	return lib;
}

static NSString *beside(const char *argv0, NSString *file)
{
	NSString *exe = [NSString stringWithUTF8String:argv0];
	if (![exe containsString:@"/"])
		exe = [@"./" stringByAppendingString:exe];
	return [[[exe stringByResolvingSymlinksInPath] stringByDeletingLastPathComponent] stringByAppendingPathComponent:file];
}

static id<MTLLibrary> data_library(const char *argv0, NSString *file)
{
	NSData *bytes = [NSData dataWithContentsOfFile:beside(argv0, file)];
	if (bytes == nil) {
		printf("  ! no %s\n", file.UTF8String);
		return nil;
	}
	dispatch_data_t data = dispatch_data_create(bytes.bytes, bytes.length, dispatch_get_main_queue(),
	                                            DISPATCH_DATA_DESTRUCTOR_DEFAULT);
	NSError *error = nil;
	id<MTLLibrary> lib = [D newLibraryWithData:data error:&error];
	if (lib == nil) {
		printf("  ! library from the bytes of %s:", file.UTF8String);
		print_error(error);
		printf("\n");
	}
	return lib;
}

static NSString *const kSource = @"#include <metal_stdlib>\n"
                                  "using namespace metal;\n"
                                  "constant bool kTint [[function_constant(0)]];\n"
                                  "struct In { float4 position [[attribute(0)]]; half2 uv [[attribute(5)]];\n"
                                  "  float4 tint [[attribute(6), function_constant(kTint)]]; };\n"
                                  "struct Out { float4 position [[position]]; float2 uv; };\n"
                                  "vertex Out s_vertex(In in [[stage_in]], constant float4x4 &m [[buffer(1)]]) {\n"
                                  "  Out o; o.position = m * in.position; if (kTint) o.position += in.tint;\n"
                                  "  o.uv = float2(in.uv); return o; }\n"
                                  "struct KIn { float2 p [[attribute(1)]]; };\n"
                                  "kernel void s_kernel(KIn in [[stage_in]], device float *out [[buffer(0)]],\n"
                                  "  uint gid [[thread_position_in_grid]]) { out[gid] = in.p.x + in.p.y; }\n";

static id<MTLLibrary> source_library(void)
{
	NSError *error = nil;
	id<MTLLibrary> lib = [D newLibraryWithSource:kSource options:nil error:&error];
	if (lib == nil) {
		printf("  ! library from source:");
		print_error(error);
		printf("\n");
	}
	return lib;
}

static void specialised(id<MTLLibrary> lib, NSString *tag, NSString *function, MTLFunctionConstantValues *values)
{
	if (lib == nil)
		return;
	NSError *error = nil;
	id<MTLFunction> f = [lib newFunctionWithName:function constantValues:values error:&error];
	if (f == nil) {
		printf("S %s none", tag.UTF8String);
		print_error(error);
		printf("\n");
		return;
	}
	printf("S %s\n", tag.UTF8String);
	function_answers(lib, f, tag);
}

static void specialisations(id<MTLLibrary> fnreflect, id<MTLLibrary> reflection)
{
	bool yes = true, no = false;
	MTLFunctionConstantValues *v = [MTLFunctionConstantValues new];
	[v setConstantValue:&yes type:MTLDataTypeBool withName:@"kOffset"];
	[v setConstantValue:&no type:MTLDataTypeBool withName:@"kColor"];
	specialised(fnreflect, @"v_gated.offset", @"v_gated", v);
	v = [MTLFunctionConstantValues new];
	[v setConstantValue:&yes type:MTLDataTypeBool withName:@"kOffset"];
	[v setConstantValue:&yes type:MTLDataTypeBool withName:@"kColor"];
	specialised(fnreflect, @"v_gated.both", @"v_gated", v);
	v = [MTLFunctionConstantValues new];
	[v setConstantValue:&no type:MTLDataTypeBool withName:@"kOffset"];
	[v setConstantValue:&no type:MTLDataTypeBool withName:@"kColor"];
	specialised(fnreflect, @"v_gated.none", @"v_gated", v);
	v = [MTLFunctionConstantValues new];
	[v setConstantValue:&no type:MTLDataTypeBool withName:@"kColor"];
	specialised(fnreflect, @"v_deadread.off", @"v_deadread", v);
	v = [MTLFunctionConstantValues new];
	[v setConstantValue:&yes type:MTLDataTypeBool withName:@"kColor"];
	specialised(fnreflect, @"v_deadread.on", @"v_deadread", v);
	v = [MTLFunctionConstantValues new];
	int index = 5;
	[v setConstantValue:&yes type:MTLDataTypeBool withName:@"kOn"];
	[v setConstantValue:&no type:MTLDataTypeBool withName:@"kOff"];
	[v setConstantValue:&index type:MTLDataTypeInt withName:@"kIndex"];
	specialised(reflection, @"k_constants.on", @"k_constants", v);
	v = [MTLFunctionConstantValues new];
	int mode = 1;
	[v setConstantValue:&mode type:MTLDataTypeInt withName:@"kMode"];
	specialised(reflection, @"v_specialised.one", @"v_specialised", v);
	for (mode = 0; mode <= 2; mode += 2) {
		v = [MTLFunctionConstantValues new];
		[v setConstantValue:&mode type:MTLDataTypeInt withName:@"kMode"];
		specialised(reflection, mode ? @"v_specialised.two" : @"v_specialised.zero", @"v_specialised", v);
	}
}

// --- pipeline states -------------------------------------------------------------------------

static void protocol_properties(id o, Protocol *p, NSMutableDictionary<NSString *, NSString *> *out,
                                NSMutableSet<NSString *> *seen)
{
	if (p == nil)
		return;
	NSString *pn = @(protocol_getName(p));
	if ([seen containsObject:pn] || [pn isEqualToString:@"NSObject"])
		return;
	[seen addObject:pn];
	for (int required = 1; required >= 0; required--) {
		unsigned n = 0;
		objc_property_t *ps = protocol_copyPropertyList2(p, &n, required, YES);
		for (unsigned i = 0; i < n; i++) {
			NSString *attrs = @(property_getAttributes(ps[i]));
			NSString *getterName = @(property_getName(ps[i]));
			NSString *type = @"";
			for (NSString *a in [attrs componentsSeparatedByString:@","]) {
				if ([a hasPrefix:@"T"])
					type = [a substringFromIndex:1];
				else if ([a hasPrefix:@"G"])
					getterName = [a substringFromIndex:1];
			}
			if (type.length == 0 || strchr("@QqIiBcCLlSsfd", [type characterAtIndex:0]) == NULL)
				continue;
			SEL s = NSSelectorFromString(getterName);
			out[getterName] = [o respondsToSelector:s] ? safely(o, s, type.UTF8String) : @"not answered";
		}
		free(ps);
	}
	unsigned n = 0;
	Protocol *__unsafe_unretained *adopted = protocol_copyProtocolList(p, &n);
	for (unsigned i = 0; i < n; i++)
		protocol_properties(o, adopted[i], out, seen);
	free(adopted);
}

// imageblockMemoryLengthForDimensions: of 8x8x1, caught as safely() catches a getter.
static NSString *imageblock_length(id o)
{
	SEL s = sel_registerName("imageblockMemoryLengthForDimensions:");
	if (![o respondsToSelector:s])
		return @"not answered";
	fflush(stdout);
	const int saved = dup(2);
	const int null = open("/dev/null", O_WRONLY);
	if (null >= 0)
		dup2(null, 2);
	NSString *out = nil;
	if (sigsetjmp(Jump, 1) == 0) {
		Catching = 1;
		NSUInteger v = ((NSUInteger(*)(id, SEL, MTLSize))objc_msgSend)(o, s, MTLSizeMake(8, 8, 1));
		Catching = 0;
		out = [NSString stringWithFormat:@"%lu", (unsigned long)v];
	} else {
		Catching = 0;
		out = @"asserts";
	}
	dup2(saved, 2);
	close(saved);
	if (null >= 0)
		close(null);
	return out;
}

static void state_answers(const char *name, id state, Protocol *p, NSError *error)
{
	printf("P %s %s", name, state ? "made" : "nil");
	print_error(error);
	printf(" class=%s\n", state ? class_getName([state class]) : "-");
	if (state == nil)
		return;
	NSMutableDictionary<NSString *, NSString *> *answers = [NSMutableDictionary dictionary];
	protocol_properties(state, p, answers, [NSMutableSet set]);
	for (NSString *k in [answers.allKeys sortedArrayUsingSelector:@selector(compare:)])
		printf("P %s.%s = %s\n", name, k.UTF8String, answers[k].UTF8String);
	printf("P %s.imageblockMemoryLengthForDimensions:8x8x1 = %s\n", name, imageblock_length(state).UTF8String);
}

static id<MTLFunction> fn(id<MTLLibrary> lib, NSString *name)
{
	id<MTLFunction> f = [lib newFunctionWithName:name];
	if (f == nil)
		printf("  ! no function %s\n", name.UTF8String);
	return f;
}

static void pipelines(id<MTLLibrary> lib, id<MTLLibrary> reflection)
{
	if (lib == nil)
		return;
	NSError *error = nil;
	id state = [D newComputePipelineStateWithFunction:fn(lib, @"k_tg") error:&error];
	state_answers("c_tg", state, @protocol(MTLComputePipelineState), error);

	MTLComputePipelineDescriptor *cd = [MTLComputePipelineDescriptor new];
	cd.computeFunction = fn(lib, @"k_stage");
	MTLStageInputOutputDescriptor *s = [MTLStageInputOutputDescriptor stageInputOutputDescriptor];
	const struct {
		NSUInteger index;
		MTLAttributeFormat format;
		NSUInteger offset;
	} kIn[] = {{0, MTLAttributeFormatInt2, 0},
	           {4, MTLAttributeFormatUChar4, 8},
	           {9, MTLAttributeFormatFloat, 12},
	           {12, MTLAttributeFormatHalf4, 16}};
	for (unsigned i = 0; i < 4; i++) {
		s.attributes[kIn[i].index].format = kIn[i].format;
		s.attributes[kIn[i].index].offset = kIn[i].offset;
		s.attributes[kIn[i].index].bufferIndex = 1;
	}
	s.layouts[1].stride = 24;
	s.layouts[1].stepFunction = MTLStepFunctionThreadPositionInGridX;
	cd.stageInputDescriptor = s;
	error = nil;
	state = [D newComputePipelineStateWithDescriptor:cd options:0 reflection:nil error:&error];
	state_answers("c_stage", state, @protocol(MTLComputePipelineState), error);

	MTLRenderPipelineDescriptor *rd = [MTLRenderPipelineDescriptor new];
	rd.vertexFunction = fn(lib, @"v_plain");
	rd.fragmentFunction = fn(lib, @"f_in");
	rd.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
	error = nil;
	state = [D newRenderPipelineStateWithDescriptor:rd error:&error];
	state_answers("r_plain", state, @protocol(MTLRenderPipelineState), error);

	rd = [MTLRenderPipelineDescriptor new];
	rd.vertexFunction = fn(lib, @"v_mixed");
	rd.fragmentFunction = fn(lib, @"f_in");
	rd.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
	MTLVertexDescriptor *vd = [MTLVertexDescriptor vertexDescriptor];
	vd.attributes[0].format = MTLVertexFormatFloat3;
	vd.attributes[0].offset = 0;
	vd.attributes[0].bufferIndex = 0;
	vd.attributes[3].format = MTLVertexFormatUChar4;
	vd.attributes[3].offset = 12;
	vd.attributes[3].bufferIndex = 0;
	vd.layouts[0].stride = 16;
	rd.vertexDescriptor = vd;
	error = nil;
	state = [D newRenderPipelineStateWithDescriptor:rd error:&error];
	state_answers("r_mixed", state, @protocol(MTLRenderPipelineState), error);

	rd = [MTLRenderPipelineDescriptor new];
	rd.vertexFunction = reflection != nil ? fn(reflection, @"v_only") : nil;
	rd.rasterizationEnabled = NO;
	error = nil;
	state = [D newRenderPipelineStateWithDescriptor:rd error:&error];
	state_answers("r_vertex", state, @protocol(MTLRenderPipelineState), error);

	Class mesh = NSClassFromString(@"MTLMeshRenderPipelineDescriptor");
	if (mesh == Nil || reflection == nil) {
		printf("P r_mesh: no MTLMeshRenderPipelineDescriptor or no library\n");
		return;
	}
	MTLMeshRenderPipelineDescriptor *md = [mesh new];
	md.objectFunction = fn(reflection, @"o_payload");
	md.meshFunction = fn(reflection, @"m_payload");
	md.fragmentFunction = fn(reflection, @"f_mesh");
	md.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
	error = nil;
	state = [D newRenderPipelineStateWithMeshDescriptor:md options:0 reflection:nil error:&error];
	state_answers("r_mesh", state, @protocol(MTLRenderPipelineState), error);
}

static void dump_class(Class c)
{
	printf("class %s : %s\n", class_getName(c), class_getSuperclass(c) ? class_getName(class_getSuperclass(c)) : "-");
	unsigned n = 0;
	Ivar *vs = class_copyIvarList(c, &n);
	for (unsigned i = 0; i < n; i++)
		printf("  ivar %s %s @%td\n", ivar_getName(vs[i]), ivar_getTypeEncoding(vs[i]) ?: "?", ivar_getOffset(vs[i]));
	free(vs);
	Method *ms = class_copyMethodList(c, &n);
	for (unsigned i = 0; i < n; i++)
		printf("  - %s %s\n", sel_getName(method_getName(ms[i])), method_getTypeEncoding(ms[i]) ?: "?");
	free(ms);
}

// The classes the walks met, and the attribute, constant and function reflection classes Metal has,
// each with its superclasses up to NSObject: their instance variables and instance methods.
static void dump_classes(void)
{
	NSArray *named = @[
		@"MTLVertexAttribute", @"MTLVertexAttributeInternal", @"MTLAttribute", @"MTLAttributeInternal",
		@"MTLFunctionConstant", @"MTLFunctionConstantInternal", @"MTLFunctionReflection",
		@"MTLFunctionReflectionInternal"
	];
	NSMutableOrderedSet<NSString *> *all = [NSMutableOrderedSet orderedSetWithArray:named];
	[all unionOrderedSet:Seen];
	NSMutableSet *done = [NSMutableSet set];
	for (NSString *name in all) {
		if (NSClassFromString(name) == Nil)
			printf("class %s: none\n", name.UTF8String);
		for (Class c = NSClassFromString(name); c != Nil && c != NSObject.class; c = class_getSuperclass(c)) {
			NSString *cn = NSStringFromClass(c);
			if ([done containsObject:cn])
				break;
			[done addObject:cn];
			dump_class(c);
		}
	}
}

int main(int argc, char **argv)
{
	@autoreleasepool {
		if (argc < 2) {
			fprintf(stderr, "usage: oracle_fnreflect DEVICE [LIB...]\n");
			return 2;
		}
		setvbuf(stdout, NULL, _IOLBF, 0);
		D = oracle_device(argv[1]);
		oracle_header("oracle_fnreflect", D);
		walk_init();
		id<MTLLibrary> fnreflect = url_library(argv[0], "oracle_fnreflect");
		id<MTLLibrary> reflection = url_library(argv[0], "oracle_reflection");
		library_answers(@"oracle_fnreflect", fnreflect);
		library_answers(@"oracle_fnreflect_11", url_library(argv[0], "oracle_fnreflect_11"));
		library_answers(@"oracle_reflection", reflection);
		library_answers(@"oracle_fnreflect.data", data_library(argv[0], @"oracle_fnreflect.metallib"));
		library_answers(@"source", source_library());
		for (int i = 2; i < argc; i++)
			library_answers(@(argv[i]), url_library(argv[0], argv[i]));
		specialisations(fnreflect, reflection);
		pipelines(fnreflect, reflection);
		dump_classes();
		printf("# oracle_fnreflect done\n");
	}
	return 0;
}
