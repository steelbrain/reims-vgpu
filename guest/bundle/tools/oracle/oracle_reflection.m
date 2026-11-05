// oracle_reflection: the reflection Metal answers for render and compute pipelines (the
// `reflection:` out-parameter of the pipeline creators, and the completion handlers' argument), so
// the bundle can answer the same.
//   oracle_reflection DEVICE [CASE]   every case (or the one named)
// Cases, over oracle_reflection.metallib (oracle_reflection.metal):
//   c_basic, c_unused, c_types, c_more, c_builtins_only, c_nothing, c_argument_buffer
//               compute pipelines of the kernel of that name
//               (k_*), made with newComputePipelineStateWithFunction:options:reflection:error:
//   c_constants  k_constants specialised with kOn true, kOff false and kIndex 5: its buffer 3 is
//               declared under no constant and read only where kOff is true
//   d_basic      k_basic, through newComputePipelineStateWithDescriptor:options:reflection:error:
//   d_stage_in   k_stage_in through a descriptor whose stage input descriptor has attributes 0
//               (float4, buffer 0, offset 0), 1 (float2, buffer 1), 2 (float4, buffer 0, offset 16)
//               and 5 (float, buffer 2, which the kernel does not declare), layouts of stride 32,
//               8 and 4
//   r_buffers    v_buffers and f_textures, one BGRA8 colour attachment
//   r_stage_in   v_stage_in and f_plain, a vertex descriptor as d_stage_in's
//   r_vertex     v_only, rasterization disabled
//   r_targets    v_buffers and f_targets, two colour attachments (RGBA32Float, RGBA16Float)
//   r_mesh       o_payload, m_payload and f_plain through newRenderPipelineStateWithMeshDescriptor:
//   async        c_basic and r_buffers through the completion-handler forms, options 3
// Each case is made with options 0, 1 (BindingInfo), 2 (BufferTypeInfo) and 3, and prints
//   CASE opts=N: state=made|nil error=... reflection=nil|CLASS
// then every getter of no argument of the reflection and of the objects and arrays it gives,
// recursively (`CASE/N.PATH = VALUE`; a getter that ends the process where it does not apply is
// "asserts"), or `same as opts=M` where the walk is the one an
// earlier option printed. Then, for every class the walks met and the reflection classes Metal
// has, its instance methods with their type encodings and its instance variables, under
// `class NAME : SUPER`. With ORACLE_IVARS set, each object's instance variables are printed beside
// its getters. Build: see README.md.
#import "oracle_common.h"
#import "oracle_walk.h"

static id<MTLDevice> D;
static id<MTLLibrary> L;

static void print_error(NSError *e)
{
	if (!e) {
		printf(" error=nil");
		return;
	}
	NSString *t = [e.localizedDescription stringByReplacingOccurrencesOfString:@"\n" withString:@"\\n"];
	printf(" error=%s/%ld \"%s\"", e.domain.UTF8String, (long)e.code, t.UTF8String);
}

// Whether the arrays `a` and `b` (bindings and arguments of one stage) hold the same objects.
static void same_objects(NSString *what, NSArray *a, NSArray *b)
{
	if (a == nil || b == nil) {
		printf("  %s: %s\n", what.UTF8String, a == b ? "both nil" : "one nil");
		return;
	}
	BOOL same = a.count == b.count;
	for (NSUInteger i = 0; same && i < a.count; i++)
		same = a[i] == b[i];
	printf("  %s: %s (%lu, %lu)%s\n", what.UTF8String, same ? "same objects" : "differ", (unsigned long)a.count,
	       (unsigned long)b.count, a == b ? ", same array" : "");
}

static id getter(id o, const char *name)
{
	SEL s = sel_registerName(name);
	if (![o respondsToSelector:s])
		return nil;
	return ((id(*)(id, SEL))objc_msgSend)(o, s);
}

static NSMutableDictionary<NSString *, NSArray *> *Printed;

// Prints the case's line and its reflection's walk.
static void report(const char *name, unsigned opts, id state, id reflection, NSError *error)
{
	printf("%s opts=%u: state=%s", name, opts, state ? "made" : "nil");
	print_error(error);
	printf(" reflection=%s\n", reflection ? class_getName([reflection class]) : "nil");
	if (reflection == nil)
		return;
	if ([reflection respondsToSelector:@selector(vertexBindings)]) {
		same_objects(@"vertexBindings/vertexArguments", getter(reflection, "vertexBindings"),
		             getter(reflection, "vertexArguments"));
		same_objects(@"fragmentBindings/fragmentArguments", getter(reflection, "fragmentBindings"),
		             getter(reflection, "fragmentArguments"));
	} else {
		same_objects(@"bindings/arguments", getter(reflection, "bindings"), getter(reflection, "arguments"));
	}
	NSMutableArray<NSString *> *lines = [NSMutableArray array];
	walk(reflection, @"", 0, lines);
	if (Ivars)
		resolve_ivars(lines);
	NSString *key = [NSString stringWithFormat:@"%s", name];
	NSArray *before = Printed[key];
	if (before != nil) {
		for (NSUInteger i = 0; i < before.count; i += 2) {
			if ([before[i + 1] isEqual:lines]) {
				printf("  same as opts=%s\n", [before[i] UTF8String]);
				return;
			}
		}
	}
	NSMutableArray *list = before ? [before mutableCopy] : [NSMutableArray array];
	[list addObject:[NSString stringWithFormat:@"%u", opts]];
	[list addObject:lines];
	Printed[key] = list;
	for (NSString *l in lines)
		printf("%s/%u%s\n", name, opts, l.UTF8String);
}

static id<MTLFunction> fn(NSString *name)
{
	id<MTLFunction> f = [L newFunctionWithName:name];
	if (f == nil)
		printf("  ! no function %s\n", name.UTF8String);
	return f;
}

static void compute_function(const char *name, NSString *kernel)
{
	id<MTLFunction> f = fn(kernel);
	for (unsigned opts = 0; opts < 4; opts++) {
		MTLAutoreleasedComputePipelineReflection reflection = nil;
		NSError *error = nil;
		id state = [D newComputePipelineStateWithFunction:f options:opts reflection:&reflection error:&error];
		report(name, opts, state, reflection, error);
	}
}

static void compute_constants(void)
{
	MTLFunctionConstantValues *values = [MTLFunctionConstantValues new];
	bool on = true, off = false;
	int index = 5;
	[values setConstantValue:&on type:MTLDataTypeBool withName:@"kOn"];
	[values setConstantValue:&off type:MTLDataTypeBool withName:@"kOff"];
	[values setConstantValue:&index type:MTLDataTypeInt withName:@"kIndex"];
	NSError *error = nil;
	id<MTLFunction> f = [L newFunctionWithName:@"k_constants" constantValues:values error:&error];
	if (f == nil) {
		printf("c_constants: no function");
		print_error(error);
		printf("\n");
		return;
	}
	for (unsigned opts = 0; opts < 4; opts++) {
		MTLAutoreleasedComputePipelineReflection reflection = nil;
		error = nil;
		id state = [D newComputePipelineStateWithFunction:f options:opts reflection:&reflection error:&error];
		report("c_constants", opts, state, reflection, error);
	}
}

static void stage_attribute(MTLAttributeDescriptorArray *attributes, NSUInteger i, MTLAttributeFormat format,
                            NSUInteger offset, NSUInteger buffer)
{
	MTLAttributeDescriptor *a = attributes[i];
	a.format = format;
	a.offset = offset;
	a.bufferIndex = buffer;
}

static void vertex_attribute(MTLVertexAttributeDescriptorArray *attributes, NSUInteger i, MTLVertexFormat format,
                             NSUInteger offset, NSUInteger buffer)
{
	MTLVertexAttributeDescriptor *a = attributes[i];
	a.format = format;
	a.offset = offset;
	a.bufferIndex = buffer;
}

static MTLStageInputOutputDescriptor *stage_input(void)
{
	MTLStageInputOutputDescriptor *s = [MTLStageInputOutputDescriptor stageInputOutputDescriptor];
	stage_attribute(s.attributes, 0, MTLAttributeFormatFloat4, 0, 0);
	stage_attribute(s.attributes, 1, MTLAttributeFormatFloat2, 0, 1);
	stage_attribute(s.attributes, 2, MTLAttributeFormatFloat4, 16, 0);
	stage_attribute(s.attributes, 5, MTLAttributeFormatFloat, 0, 2);
	s.layouts[0].stride = 32;
	s.layouts[1].stride = 8;
	s.layouts[2].stride = 4;
	for (NSUInteger i = 0; i < 3; i++)
		s.layouts[i].stepFunction = MTLStepFunctionThreadPositionInGridX;
	return s;
}

static MTLVertexDescriptor *vertex_descriptor(void)
{
	MTLVertexDescriptor *v = [MTLVertexDescriptor vertexDescriptor];
	vertex_attribute(v.attributes, 0, MTLVertexFormatFloat4, 0, 0);
	vertex_attribute(v.attributes, 1, MTLVertexFormatFloat2, 0, 1);
	vertex_attribute(v.attributes, 2, MTLVertexFormatFloat4, 16, 0);
	vertex_attribute(v.attributes, 5, MTLVertexFormatFloat, 0, 2);
	v.layouts[0].stride = 32;
	v.layouts[1].stride = 8;
	v.layouts[2].stride = 4;
	return v;
}

static void compute_descriptor(const char *name, NSString *kernel, BOOL stageIn)
{
	MTLComputePipelineDescriptor *d = [MTLComputePipelineDescriptor new];
	d.computeFunction = fn(kernel);
	if (stageIn)
		d.stageInputDescriptor = stage_input();
	for (unsigned opts = 0; opts < 4; opts++) {
		MTLAutoreleasedComputePipelineReflection reflection = nil;
		NSError *error = nil;
		id state = [D newComputePipelineStateWithDescriptor:d options:opts reflection:&reflection error:&error];
		report(name, opts, state, reflection, error);
	}
}

static MTLRenderPipelineDescriptor *render_descriptor(NSString *vertex, NSString *fragment)
{
	MTLRenderPipelineDescriptor *d = [MTLRenderPipelineDescriptor new];
	d.vertexFunction = fn(vertex);
	if (fragment != nil) {
		d.fragmentFunction = fn(fragment);
		d.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
	} else {
		d.rasterizationEnabled = NO;
	}
	return d;
}

static void render(const char *name, MTLRenderPipelineDescriptor *d)
{
	for (unsigned opts = 0; opts < 4; opts++) {
		MTLAutoreleasedRenderPipelineReflection reflection = nil;
		NSError *error = nil;
		id state = [D newRenderPipelineStateWithDescriptor:d options:opts reflection:&reflection error:&error];
		report(name, opts, state, reflection, error);
	}
}

static void mesh(void)
{
	Class cls = NSClassFromString(@"MTLMeshRenderPipelineDescriptor");
	if (cls == Nil) {
		printf("r_mesh: no MTLMeshRenderPipelineDescriptor\n");
		return;
	}
	MTLMeshRenderPipelineDescriptor *d = [cls new];
	d.objectFunction = fn(@"o_payload");
	d.meshFunction = fn(@"m_payload");
	d.fragmentFunction = fn(@"f_mesh");
	d.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
	for (unsigned opts = 0; opts < 4; opts++) {
		MTLAutoreleasedRenderPipelineReflection reflection = nil;
		NSError *error = nil;
		id state = [D newRenderPipelineStateWithMeshDescriptor:d options:opts reflection:&reflection error:&error];
		report("r_mesh", opts, state, reflection, error);
	}
}

static void async_forms(void)
{
	dispatch_semaphore_t done = dispatch_semaphore_create(0);
	[D newComputePipelineStateWithFunction:fn(@"k_basic")
	                               options:3
	                     completionHandler:^(id<MTLComputePipelineState> state,
	                                         MTLComputePipelineReflection *reflection, NSError *error) {
		                     report("async_c_basic", 3, state, reflection, error);
		                     dispatch_semaphore_signal(done);
	                     }];
	dispatch_semaphore_wait(done, DISPATCH_TIME_FOREVER);
	[D newRenderPipelineStateWithDescriptor:render_descriptor(@"v_buffers", @"f_textures")
	                                options:3
	                      completionHandler:^(id<MTLRenderPipelineState> state,
	                                          MTLRenderPipelineReflection *reflection, NSError *error) {
		                      report("async_r_buffers", 3, state, reflection, error);
		                      dispatch_semaphore_signal(done);
	                      }];
	dispatch_semaphore_wait(done, DISPATCH_TIME_FOREVER);
}

static void dump_class(Class c)
{
	printf("class %s : %s\n", class_getName(c), class_getSuperclass(c) ? class_getName(class_getSuperclass(c)) : "-");
	unsigned n = 0;
	Protocol *__unsafe_unretained *ps = class_copyProtocolList(c, &n);
	for (unsigned i = 0; i < n; i++)
		printf("  protocol %s\n", protocol_getName(ps[i]));
	free(ps);
	Ivar *vs = class_copyIvarList(c, &n);
	for (unsigned i = 0; i < n; i++)
		printf("  ivar %s %s @%td\n", ivar_getName(vs[i]), ivar_getTypeEncoding(vs[i]) ?: "?",
		       ivar_getOffset(vs[i]));
	free(vs);
	Method *ms = class_copyMethodList(c, &n);
	for (unsigned i = 0; i < n; i++)
		printf("  - %s %s\n", sel_getName(method_getName(ms[i])), method_getTypeEncoding(ms[i]) ?: "?");
	free(ms);
	ms = class_copyMethodList(object_getClass((id)c), &n);
	for (unsigned i = 0; i < n; i++)
		printf("  + %s %s\n", sel_getName(method_getName(ms[i])), method_getTypeEncoding(ms[i]) ?: "?");
	free(ms);
}

static void dump_classes(void)
{
	NSArray *named = @[
		@"MTLComputePipelineReflection", @"MTLComputePipelineReflectionInternal", @"MTLRenderPipelineReflection",
		@"MTLRenderPipelineReflectionInternal", @"MTLArgument", @"MTLArgumentInternal", @"MTLBindingInternal",
		@"MTLBufferBindingInternal", @"MTLTextureBindingInternal", @"MTLThreadgroupBindingInternal",
		@"MTLType", @"MTLTypeInternal", @"MTLStructType", @"MTLStructTypeInternal", @"MTLStructMember",
		@"MTLStructMemberInternal", @"MTLArrayType", @"MTLArrayTypeInternal", @"MTLPointerType",
		@"MTLPointerTypeInternal", @"MTLTextureReferenceType", @"MTLTextureReferenceTypeInternal"
	];
	NSMutableOrderedSet<NSString *> *all = [NSMutableOrderedSet orderedSetWithArray:named];
	[all unionOrderedSet:Seen];
	NSMutableSet *done = [NSMutableSet set];
	for (NSString *name in all) {
		for (Class c = NSClassFromString(name); c != Nil && c != NSObject.class; c = class_getSuperclass(c)) {
			NSString *cn = NSStringFromClass(c);
			if ([done containsObject:cn])
				break;
			[done addObject:cn];
			dump_class(c);
		}
		if (NSClassFromString(name) == Nil)
			printf("class %s: none\n", name.UTF8String);
	}
}

int main(int argc, char **argv)
{
	@autoreleasepool {
		if (argc < 2) {
			fprintf(stderr, "usage: oracle_reflection DEVICE [CASE]\n");
			return 2;
		}
		setvbuf(stdout, NULL, _IOLBF, 0);
		D = oracle_device(argv[1]);
		oracle_header("oracle_reflection", D);
		L = oracle_library(D, argv[0], "oracle_reflection");
		if (L == nil)
			return 1;
		walk_init();
		Printed = [NSMutableDictionary dictionary];
		const char *only = argc > 2 ? argv[2] : NULL;
#define CASE(n) (only == NULL || strcmp(only, n) == 0)
		if (CASE("c_basic"))
			compute_function("c_basic", @"k_basic");
		if (CASE("c_unused"))
			compute_function("c_unused", @"k_unused");
		if (CASE("c_types"))
			compute_function("c_types", @"k_types");
		if (CASE("c_more"))
			compute_function("c_more", @"k_more");
		if (CASE("c_builtins_only"))
			compute_function("c_builtins_only", @"k_builtins_only");
		if (CASE("c_nothing"))
			compute_function("c_nothing", @"k_nothing");
		if (CASE("c_argument_buffer"))
			compute_function("c_argument_buffer", @"k_argument_buffer");
		if (CASE("c_constants"))
			compute_constants();
		if (CASE("d_basic"))
			compute_descriptor("d_basic", @"k_basic", NO);
		if (CASE("d_stage_in"))
			compute_descriptor("d_stage_in", @"k_stage_in", YES);
		if (CASE("r_buffers"))
			render("r_buffers", render_descriptor(@"v_buffers", @"f_textures"));
		if (CASE("r_stage_in")) {
			MTLRenderPipelineDescriptor *d = render_descriptor(@"v_stage_in", @"f_plain");
			d.vertexDescriptor = vertex_descriptor();
			render("r_stage_in", d);
		}
		if (CASE("r_vertex"))
			render("r_vertex", render_descriptor(@"v_only", nil));
		if (CASE("r_targets")) {
			MTLRenderPipelineDescriptor *d = render_descriptor(@"v_buffers", @"f_targets");
			d.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA32Float;
			d.colorAttachments[1].pixelFormat = MTLPixelFormatRGBA16Float;
			render("r_targets", d);
		}
		if (CASE("r_mesh"))
			mesh();
		if (CASE("async"))
			async_forms();
		dump_classes();
	}
	return 0;
}
