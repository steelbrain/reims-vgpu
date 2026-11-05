// oracle_mesh: what a device answers for mesh render pipelines and draws them, so the bundle and
// the host give the same.
//   oracle_mesh DEVICE [CASE]   every case (or the one named), each step in a child of its own
// Cases:
//   make    newRenderPipelineStateWithMeshDescriptor: of one descriptor a step: whether it is
//           made, its error, and the state's answers (`state:` line): maxTotalThreadsPerThreadgroup
//           (tg), maxTotalThreadsPerObjectThreadgroup (objTg), maxTotalThreadsPerMeshThreadgroup
//           (meshTg), objectThreadExecutionWidth (objW), meshThreadExecutionWidth (meshW),
//           maxTotalThreadgroupsPerMeshGrid (grid), imageblockSampleLength (ibs),
//           threadgroupSizeMatchesTileSize (tile), supportIndirectCommandBuffers (icb),
//           gpuResourceID non-zero (rid), requiredThreadsPerObjectThreadgroup and
//           ...MeshThreadgroup (reqObj, reqMesh), and with reflection asked for, whether one
//           came and its object, mesh and fragment bindings' counts.
//   draw    draws into a 16x4 BGRA8 target cleared to 0, one step a draw form, and prints each
//           of its 8 columns' centre pixel as BBGGRRAA. A mesh threadgroup covers column x of
//           the grid's width (its threadgroup_position_in_grid.x of threadgroups_per_grid.x) in
//           the colour its step names, so the columns say which threadgroups ran of what grid.
//   icb     whether an indirect command buffer of the mesh command types is made, and a mesh
//           draw encoded into one by the CPU and executed, by its pixels.
//   device  the device's family answers and limits that bear on mesh shading.
// A step whose child died says how; Metal's validation's words are then on stderr. Build: see
// README.md.
#import "oracle_common.h"
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

extern char **environ;

static id<MTLDevice> D;

static NSString *const kSource =
    @"#include <metal_stdlib>\n"
    @"using namespace metal;\n"
    @"struct V { float4 p [[position]]; float4 c; };\n"
    @"using M = metal::mesh<V, void, 4, 2, topology::triangle>;\n"
    @"struct Payload { float4 c; uint n; };\n"
    // Column tg.x of tgs.x columns, in colour c.
    @"static void column(M m, uint t, uint x, uint n, float4 c) {\n"
    @"  float x0 = -1.0 + 2.0 * float(x) / float(n), x1 = -1.0 + 2.0 * float(x + 1) / float(n);\n"
    @"  if (t == 0) { m.set_primitive_count(2);\n"
    @"    m.set_index(0, 0); m.set_index(1, 1); m.set_index(2, 2);\n"
    @"    m.set_index(3, 2); m.set_index(4, 1); m.set_index(5, 3); }\n"
    @"  if (t < 4) { V v; v.p = float4((t & 1) ? x1 : x0, (t & 2) ? 1.0 : -1.0, 0, 1); v.c = c;\n"
    @"    m.set_vertex(t, v); }\n"
    @"}\n"
    @"[[mesh]] void m_plain(M m, uint t [[thread_index_in_threadgroup]],\n"
    @"    uint3 tg [[threadgroup_position_in_grid]], uint3 tgs [[threadgroups_per_grid]],\n"
    @"    constant float4 &c [[buffer(0)]]) { column(m, t, tg.x, tgs.x, c); }\n"
    // The colour from a buffer at an offset, and the grid's threads in its alpha.
    @"[[mesh]] void m_grid(M m, uint t [[thread_index_in_threadgroup]],\n"
    @"    uint3 tg [[threadgroup_position_in_grid]], uint3 tgs [[threadgroups_per_grid]],\n"
    @"    uint3 g [[threads_per_grid]], device const float4 *c [[buffer(1)]]) {\n"
    @"  float4 k = c[0]; k.a = float(g.x) / 255.0; column(m, t, tg.x, tgs.x, k); }\n"
    @"[[mesh]] void m_tex(M m, uint t [[thread_index_in_threadgroup]],\n"
    @"    uint3 tg [[threadgroup_position_in_grid]], uint3 tgs [[threadgroups_per_grid]],\n"
    @"    texture2d<float> tex [[texture(0)]], sampler s [[sampler(0)]]) {\n"
    @"  column(m, t, tg.x, tgs.x, tex.sample(s, float2(0.5), level(0))); }\n"
    @"[[mesh]] void m_payload(M m, const object_data Payload &p [[payload]],\n"
    @"    uint t [[thread_index_in_threadgroup]], uint3 tg [[threadgroup_position_in_grid]],\n"
    @"    uint3 tgs [[threadgroups_per_grid]]) {\n"
    @"  float4 k = p.c; k.a = float(p.n) / 255.0; column(m, t, tg.x, tgs.x, k); }\n"
    // An object threadgroup writes its colour (through threadgroup memory) and its threads into
    // the payload, and launches two mesh threadgroups per object threadgroup.
    @"[[object]] void o_fn(object_data Payload &p [[payload]], mesh_grid_properties g,\n"
    @"    uint t [[thread_index_in_threadgroup]], uint n [[threads_per_threadgroup]],\n"
    @"    uint ot [[threadgroup_position_in_grid]], constant float4 *c [[buffer(0)]],\n"
    @"    threadgroup float4 *s [[threadgroup(0)]]) {\n"
    @"  if (t == 0) s[0] = c[ot];\n"
    @"  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    @"  if (t == 0) { p.c = s[0]; p.n = n; g.set_threadgroups_per_grid(uint3(2, 1, 1)); } }\n"
    @"fragment float4 f_col(V in [[stage_in]]) { return in.c; }\n"
    @"vertex V v_plain(uint vid [[vertex_id]]) { V v; v.p = float4(0, 0, 0, 1); v.c = 0; return v; }\n";

static id<MTLLibrary> gLib;

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

static id<MTLLibrary> library(void)
{
	if (gLib)
		return gLib;
	MTLCompileOptions *o = [MTLCompileOptions new];
	o.languageVersion = MTLLanguageVersion3_0;
	NSError *e = nil;
	gLib = [D newLibraryWithSource:kSource options:o error:&e];
	if (!gLib) {
		printf("  ! library:");
		print_error(e);
		exit(3);
	}
	return gLib;
}

static id<MTLFunction> fn(NSString *name)
{
	id<MTLFunction> f = [library() newFunctionWithName:name];
	if (!f) {
		printf("  ! no function %s\n", name.UTF8String);
		exit(3);
	}
	return f;
}

static MTLMeshRenderPipelineDescriptor *base(NSString *mesh)
{
	MTLMeshRenderPipelineDescriptor *m = [MTLMeshRenderPipelineDescriptor new];
	m.meshFunction = fn(mesh);
	m.fragmentFunction = fn(@"f_col");
	m.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
	return m;
}

static void print_size(const char *name, MTLSize s)
{
	printf(" %s=%lux%lux%lu", name, (unsigned long)s.width, (unsigned long)s.height,
	       (unsigned long)s.depth);
}

static void print_state(id<MTLRenderPipelineState> p)
{
	printf("  state: tg=%lu objTg=%lu meshTg=%lu objW=%lu meshW=%lu grid=%lu ibs=%lu tile=%d icb=%d "
	       "rid=%d",
	       (unsigned long)p.maxTotalThreadsPerThreadgroup,
	       (unsigned long)p.maxTotalThreadsPerObjectThreadgroup,
	       (unsigned long)p.maxTotalThreadsPerMeshThreadgroup,
	       (unsigned long)p.objectThreadExecutionWidth, (unsigned long)p.meshThreadExecutionWidth,
	       (unsigned long)p.maxTotalThreadgroupsPerMeshGrid,
	       (unsigned long)p.imageblockSampleLength, (int)p.threadgroupSizeMatchesTileSize,
	       (int)p.supportIndirectCommandBuffers, p.gpuResourceID._impl != 0);
	if ([(id)p respondsToSelector:@selector(requiredThreadsPerObjectThreadgroup)]) {
		print_size("reqObj", p.requiredThreadsPerObjectThreadgroup);
		print_size("reqMesh", p.requiredThreadsPerMeshThreadgroup);
	}
	printf("\n");
}

// One descriptor, made with reflection asked for.
static void make(const char *step, MTLMeshRenderPipelineDescriptor *m, MTLPipelineOption options)
{
	NSError *e = nil;
	MTLAutoreleasedRenderPipelineReflection r = nil;
	id<MTLRenderPipelineState> p = nil;
	@try {
		p = [D newRenderPipelineStateWithMeshDescriptor:m options:options reflection:&r error:&e];
	} @catch (NSException *x) {
		printf("make.%s: threw %s \"%s\"\n", step, x.name.UTF8String, x.reason.UTF8String);
		return;
	}
	printf("make.%s: made=%d", step, p != nil);
	print_error(e);
	if (p)
		print_state(p);
	if (options != MTLPipelineOptionNone) {
		printf("  reflection: %s", r ? "yes" : "nil");
		if (r)
			printf(" object %lu mesh %lu fragment %lu vertex %lu",
			       (unsigned long)r.objectBindings.count, (unsigned long)r.meshBindings.count,
			       (unsigned long)r.fragmentBindings.count, (unsigned long)r.vertexBindings.count);
		printf("\n");
	}
}

static void step_make(const char *s)
{
	MTLPipelineOption reflect = MTLPipelineOptionBindingInfo | MTLPipelineOptionBufferTypeInfo;
	MTLMeshRenderPipelineDescriptor *m = base(@"m_plain");
	if (!strcmp(s, "plain"))
		make(s, m, reflect);
	else if (!strcmp(s, "noreflect"))
		make(s, m, MTLPipelineOptionNone);
	else if (!strcmp(s, "object")) {
		m = base(@"m_payload");
		m.objectFunction = fn(@"o_fn");
		make(s, m, reflect);
	} else if (!strcmp(s, "maxima")) {
		m = base(@"m_payload");
		m.objectFunction = fn(@"o_fn");
		m.maxTotalThreadsPerObjectThreadgroup = 64;
		m.maxTotalThreadsPerMeshThreadgroup = 32;
		m.maxTotalThreadgroupsPerMeshGrid = 8;
		m.payloadMemoryLength = 256;
		make(s, m, MTLPipelineOptionNone);
	} else if (!strcmp(s, "maxima-odd")) {
		m.maxTotalThreadsPerMeshThreadgroup = 100;
		make(s, m, MTLPipelineOptionNone);
	} else if (!strcmp(s, "mesh-2048")) {
		m.maxTotalThreadsPerMeshThreadgroup = 2048;
		make(s, m, MTLPipelineOptionNone);
	} else if (!strcmp(s, "object-2048")) {
		m = base(@"m_payload");
		m.objectFunction = fn(@"o_fn");
		m.maxTotalThreadsPerObjectThreadgroup = 2048;
		make(s, m, MTLPipelineOptionNone);
	} else if (!strcmp(s, "grid-big")) {
		m.maxTotalThreadgroupsPerMeshGrid = 1u << 20;
		make(s, m, MTLPipelineOptionNone);
	} else if (!strcmp(s, "payload-big")) {
		m = base(@"m_payload");
		m.objectFunction = fn(@"o_fn");
		m.payloadMemoryLength = 16385;
		make(s, m, MTLPipelineOptionNone);
	} else if (!strcmp(s, "multiple")) {
		m = base(@"m_payload");
		m.objectFunction = fn(@"o_fn");
		m.objectThreadgroupSizeIsMultipleOfThreadExecutionWidth = YES;
		m.meshThreadgroupSizeIsMultipleOfThreadExecutionWidth = YES;
		make(s, m, MTLPipelineOptionNone);
	} else if (!strcmp(s, "required")) {
		m = base(@"m_payload");
		m.objectFunction = fn(@"o_fn");
		m.requiredThreadsPerObjectThreadgroup = MTLSizeMake(2, 1, 1);
		m.requiredThreadsPerMeshThreadgroup = MTLSizeMake(4, 1, 1);
		make(s, m, MTLPipelineOptionNone);
	} else if (!strcmp(s, "icb")) {
		m.supportIndirectCommandBuffers = YES;
		make(s, m, MTLPipelineOptionNone);
	} else if (!strcmp(s, "norast")) {
		m.fragmentFunction = nil;
		m.rasterizationEnabled = NO;
		make(s, m, MTLPipelineOptionNone);
	} else if (!strcmp(s, "nofragment")) {
		m.fragmentFunction = nil;
		make(s, m, MTLPipelineOptionNone);
	} else if (!strcmp(s, "nomesh")) {
		m.meshFunction = nil;
		make(s, m, MTLPipelineOptionNone);
	} else if (!strcmp(s, "vertex-as-mesh")) {
		m.meshFunction = fn(@"v_plain");
		make(s, m, MTLPipelineOptionNone);
	} else if (!strcmp(s, "msaa4")) {
		m.rasterSampleCount = 4;
		make(s, m, MTLPipelineOptionNone);
	} else if (!strcmp(s, "depth")) {
		m.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float;
		make(s, m, MTLPipelineOptionNone);
	} else if (!strcmp(s, "archive-miss")) {
		NSError *e = nil;
		id<MTLBinaryArchive> a =
		    [D newBinaryArchiveWithDescriptor:[MTLBinaryArchiveDescriptor new] error:&e];
		m.binaryArchives = a ? @[ a ] : nil;
		make(s, m, MTLPipelineOptionFailOnBinaryArchiveMiss);
	} else if (!strcmp(s, "archive-miss-norast") || !strcmp(s, "archive-miss-object")) {
		NSError *e = nil;
		id<MTLBinaryArchive> a =
		    [D newBinaryArchiveWithDescriptor:[MTLBinaryArchiveDescriptor new] error:&e];
		if (!strcmp(s, "archive-miss-norast")) {
			m.fragmentFunction = nil;
			m.rasterizationEnabled = NO;
		} else {
			m = base(@"m_payload");
			m.objectFunction = fn(@"o_fn");
		}
		m.binaryArchives = a ? @[ a ] : nil;
		make(s, m, MTLPipelineOptionFailOnBinaryArchiveMiss);
	} else if (!strcmp(s, "archive-miss-object-norast")) {
		NSError *e = nil;
		id<MTLBinaryArchive> a =
		    [D newBinaryArchiveWithDescriptor:[MTLBinaryArchiveDescriptor new] error:&e];
		m = base(@"m_payload");
		m.objectFunction = fn(@"o_fn");
		m.fragmentFunction = nil;
		m.rasterizationEnabled = NO;
		m.binaryArchives = a ? @[ a ] : nil;
		make(s, m, MTLPipelineOptionFailOnBinaryArchiveMiss);
	} else if (!strcmp(s, "fragment-as-object")) {
		m = base(@"m_payload");
		m.objectFunction = fn(@"f_col");
		make(s, m, MTLPipelineOptionNone);
	} else if (!strcmp(s, "mesh-as-object")) {
		m = base(@"m_payload");
		m.objectFunction = fn(@"m_plain");
		make(s, m, MTLPipelineOptionNone);
	} else if (!strcmp(s, "object-as-mesh")) {
		m.meshFunction = fn(@"o_fn");
		make(s, m, MTLPipelineOptionNone);
	} else if (!strcmp(s, "mesh-as-fragment")) {
		m.fragmentFunction = fn(@"m_plain");
		make(s, m, MTLPipelineOptionNone);
	} else if (!strcmp(s, "vertex-as-mesh-norast")) {
		m.meshFunction = fn(@"v_plain");
		m.fragmentFunction = nil;
		m.rasterizationEnabled = NO;
		make(s, m, MTLPipelineOptionNone);
	} else if (!strcmp(s, "archive-none")) {
		// FailOnBinaryArchiveMiss with no archive listed.
		make(s, m, MTLPipelineOptionFailOnBinaryArchiveMiss);
	} else if (!strcmp(s, "label")) {
		m.label = @"mesh label";
		NSError *e = nil;
		id<MTLRenderPipelineState> p = [D newRenderPipelineStateWithMeshDescriptor:m
		                                                                   options:MTLPipelineOptionNone
		                                                                reflection:nil
		                                                                     error:&e];
		printf("make.label: made=%d label=%s", p != nil, p.label ? p.label.UTF8String : "(nil)");
		print_error(e);
	} else if (!strcmp(s, "async")) {
		dispatch_semaphore_t done = dispatch_semaphore_create(0);
		[D newRenderPipelineStateWithMeshDescriptor:m
		                                    options:MTLPipelineOptionNone
		                          completionHandler:^(id<MTLRenderPipelineState> p,
		                                              MTLRenderPipelineReflection *r, NSError *e) {
			  printf("make.async: made=%d reflection=%s main=%d", p != nil, r ? "yes" : "nil",
			         [NSThread isMainThread]);
			  print_error(e);
			  dispatch_semaphore_signal(done);
		  }];
		dispatch_semaphore_wait(done, DISPATCH_TIME_FOREVER);
	} else if (!strcmp(s, "render-api")) {
		// A mesh function given to a vertex/fragment descriptor.
		MTLRenderPipelineDescriptor *r = [MTLRenderPipelineDescriptor new];
		r.vertexFunction = fn(@"m_plain");
		r.fragmentFunction = fn(@"f_col");
		r.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
		NSError *e = nil;
		id<MTLRenderPipelineState> p = [D newRenderPipelineStateWithDescriptor:r error:&e];
		printf("make.render-api: made=%d", p != nil);
		print_error(e);
	} else {
		exit(2);
	}
}

// ---- draws ----------------------------------------------------------------------------------

enum { kW = 16, kH = 4 };

static id<MTLTexture> target(void)
{
	MTLTextureDescriptor *t = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
	                                                                             width:kW
	                                                                            height:kH
	                                                                         mipmapped:NO];
	t.usage = MTLTextureUsageRenderTarget;
	t.storageMode = MTLStorageModeManaged;
	return [D newTextureWithDescriptor:t];
}

static void print_columns(const char *step, id<MTLCommandQueue> q, id<MTLTexture> t,
                          id<MTLCommandBuffer> cb)
{
	id<MTLBlitCommandEncoder> b = [cb blitCommandEncoder];
	[b synchronizeResource:t];
	[b endEncoding];
	oracle_commit_wait(cb, NULL, NULL);
	uint32_t px[kW * kH];
	[t getBytes:px bytesPerRow:kW * 4 fromRegion:MTLRegionMake2D(0, 0, kW, kH) mipmapLevel:0];
	printf("draw.%s:", step);
	for (NSUInteger c = 0; c < 8; c++)
		printf(" %08x", px[kW * (kH / 2) + c * 2 + 1]);
	printf("\n");
}

static id<MTLRenderPipelineState> state(MTLMeshRenderPipelineDescriptor *m)
{
	NSError *e = nil;
	id<MTLRenderPipelineState> p = [D newRenderPipelineStateWithMeshDescriptor:m
	                                                                   options:MTLPipelineOptionNone
	                                                                reflection:nil
	                                                                     error:&e];
	if (!p) {
		printf("  ! no state:");
		print_error(e);
		exit(3);
	}
	return p;
}

static id<MTLRenderCommandEncoder> pass(id<MTLCommandBuffer> cb, id<MTLTexture> t)
{
	MTLRenderPassDescriptor *rp = [MTLRenderPassDescriptor renderPassDescriptor];
	rp.colorAttachments[0].texture = t;
	rp.colorAttachments[0].loadAction = MTLLoadActionClear;
	rp.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 0);
	rp.colorAttachments[0].storeAction = MTLStoreActionStore;
	return [cb renderCommandEncoderWithDescriptor:rp];
}

static void step_draw(const char *s)
{
	id<MTLCommandQueue> q = [D newCommandQueue];
	id<MTLTexture> t = target();
	id<MTLCommandBuffer> cb = [q commandBuffer];
	id<MTLRenderCommandEncoder> e = pass(cb, t);
	const float red[4] = {1, 0, 0, 1}, green[8] = {0, 1, 0, 1, 0, 0, 1, 1};
	if (!strcmp(s, "threadgroups")) {
		[e setRenderPipelineState:state(base(@"m_plain"))];
		[e setMeshBytes:red length:16 atIndex:0];
		[e drawMeshThreadgroups:MTLSizeMake(4, 1, 1)
		    threadsPerObjectThreadgroup:MTLSizeMake(1, 1, 1)
		      threadsPerMeshThreadgroup:MTLSizeMake(4, 1, 1)];
	} else if (!strcmp(s, "threads")) {
		// 10 threads in mesh threadgroups of 4, with no object function.
		id<MTLBuffer> b = [D newBufferWithLength:64 options:MTLResourceStorageModeShared];
		memcpy((char *)b.contents + 32, green, 16);
		[e setRenderPipelineState:state(base(@"m_grid"))];
		[e setMeshBuffer:b offset:0 atIndex:1];
		[e setMeshBufferOffset:32 atIndex:1];
		[e drawMeshThreads:MTLSizeMake(10, 1, 1)
		    threadsPerObjectThreadgroup:MTLSizeMake(1, 1, 1)
		      threadsPerMeshThreadgroup:MTLSizeMake(4, 1, 1)];
	} else if (!strcmp(s, "threads-plain") || !strcmp(s, "threads-exact")) {
		// drawMeshThreads with a mesh function that reads no grid builtin: 14 threads (or 16)
		// in mesh threadgroups of 4.
		[e setRenderPipelineState:state(base(@"m_plain"))];
		[e setMeshBytes:red length:16 atIndex:0];
		[e drawMeshThreads:MTLSizeMake(!strcmp(s, "threads-plain") ? 14 : 16, 1, 1)
		    threadsPerObjectThreadgroup:MTLSizeMake(1, 1, 1)
		      threadsPerMeshThreadgroup:MTLSizeMake(4, 1, 1)];
	} else if (!strcmp(s, "threadgroups-grid")) {
		id<MTLBuffer> b = [D newBufferWithLength:64 options:MTLResourceStorageModeShared];
		memcpy((char *)b.contents + 32, green, 16);
		[e setRenderPipelineState:state(base(@"m_grid"))];
		[e setMeshBuffer:b offset:32 atIndex:1];
		[e drawMeshThreadgroups:MTLSizeMake(2, 1, 1)
		    threadsPerObjectThreadgroup:MTLSizeMake(1, 1, 1)
		      threadsPerMeshThreadgroup:MTLSizeMake(4, 1, 1)];
	} else if (!strcmp(s, "indirect0")) {
		id<MTLBuffer> args = [D newBufferWithLength:64 options:MTLResourceStorageModeShared];
		const uint32_t a[3] = {3, 1, 1};
		memcpy(args.contents, a, sizeof a);
		[e setRenderPipelineState:state(base(@"m_plain"))];
		[e setMeshBytes:red length:16 atIndex:0];
		[e drawMeshThreadgroupsWithIndirectBuffer:args
		                     indirectBufferOffset:0
		              threadsPerObjectThreadgroup:MTLSizeMake(1, 1, 1)
		                threadsPerMeshThreadgroup:MTLSizeMake(4, 1, 1)];
	} else if (!strcmp(s, "indirect")) {
		id<MTLBuffer> args = [D newBufferWithLength:64 options:MTLResourceStorageModeShared];
		const uint32_t a[3] = {3, 1, 1};
		memcpy((char *)args.contents + 16, a, sizeof a);
		[e setRenderPipelineState:state(base(@"m_plain"))];
		[e setMeshBytes:red length:16 atIndex:0];
		[e drawMeshThreadgroupsWithIndirectBuffer:args
		                     indirectBufferOffset:16
		              threadsPerObjectThreadgroup:MTLSizeMake(1, 1, 1)
		                threadsPerMeshThreadgroup:MTLSizeMake(4, 1, 1)];
	} else if (!strcmp(s, "object") || !strcmp(s, "object-threads")) {
		MTLMeshRenderPipelineDescriptor *m = base(@"m_payload");
		m.objectFunction = fn(@"o_fn");
		[e setRenderPipelineState:state(m)];
		[e setObjectBytes:green length:32 atIndex:0];
		[e setObjectThreadgroupMemoryLength:16 atIndex:0];
		if (!strcmp(s, "object"))
			[e drawMeshThreadgroups:MTLSizeMake(2, 1, 1)
			    threadsPerObjectThreadgroup:MTLSizeMake(3, 1, 1)
			      threadsPerMeshThreadgroup:MTLSizeMake(4, 1, 1)];
		else // 6 threads in object threadgroups of 4: one of 4 and one of 2
			[e drawMeshThreads:MTLSizeMake(6, 1, 1)
			    threadsPerObjectThreadgroup:MTLSizeMake(4, 1, 1)
			      threadsPerMeshThreadgroup:MTLSizeMake(4, 1, 1)];
	} else if (!strcmp(s, "texture")) {
		MTLTextureDescriptor *td =
		    [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
		                                                       width:1
		                                                      height:1
		                                                   mipmapped:NO];
		id<MTLTexture> src = [D newTextureWithDescriptor:td];
		const uint8_t blue[4] = {0, 0, 255, 255};
		[src replaceRegion:MTLRegionMake2D(0, 0, 1, 1) mipmapLevel:0 withBytes:blue bytesPerRow:4];
		id<MTLSamplerState> smp = [D newSamplerStateWithDescriptor:[MTLSamplerDescriptor new]];
		[e setRenderPipelineState:state(base(@"m_tex"))];
		[e setMeshTexture:src atIndex:0];
		[e setMeshSamplerState:smp atIndex:0];
		[e drawMeshThreadgroups:MTLSizeMake(2, 1, 1)
		    threadsPerObjectThreadgroup:MTLSizeMake(1, 1, 1)
		      threadsPerMeshThreadgroup:MTLSizeMake(4, 1, 1)];
	} else if (!strcmp(s, "zero")) {
		[e setRenderPipelineState:state(base(@"m_plain"))];
		[e setMeshBytes:red length:16 atIndex:0];
		[e drawMeshThreadgroups:MTLSizeMake(0, 1, 1)
		    threadsPerObjectThreadgroup:MTLSizeMake(1, 1, 1)
		      threadsPerMeshThreadgroup:MTLSizeMake(4, 1, 1)];
	} else {
		exit(2);
	}
	[e endEncoding];
	print_columns(s, q, t, cb);
}

// ---- indirect command buffers ---------------------------------------------------------------

static void step_icb(const char *s)
{
	MTLIndirectCommandBufferDescriptor *d = [MTLIndirectCommandBufferDescriptor new];
	d.commandTypes = !strcmp(s, "threads") ? MTLIndirectCommandTypeDrawMeshThreads
	                                       : MTLIndirectCommandTypeDrawMeshThreadgroups;
	d.inheritPipelineState = NO;
	d.inheritBuffers = NO;
	d.maxObjectBufferBindCount = 1;
	d.maxMeshBufferBindCount = 1;
	d.maxFragmentBufferBindCount = 1;
	d.maxObjectThreadgroupMemoryBindCount = 1;
	id<MTLIndirectCommandBuffer> icb = [D newIndirectCommandBufferWithDescriptor:d
	                                                             maxCommandCount:2
	                                                                     options:0];
	printf("icb.%s: made=%d", s, icb != nil);
	if (icb)
		printf(" size=%lu allocatedSize=%lu", (unsigned long)icb.size,
		       (unsigned long)icb.allocatedSize);
	printf("\n");
	if (!icb)
		return;
	MTLMeshRenderPipelineDescriptor *m = base(@"m_grid");
	m.supportIndirectCommandBuffers = YES;
	id<MTLRenderPipelineState> p = state(m);
	id<MTLBuffer> b = [D newBufferWithLength:64 options:MTLResourceStorageModeShared];
	const float green[4] = {0, 1, 0, 1};
	memcpy(b.contents, green, 16);
	id<MTLIndirectRenderCommand> c = [icb indirectRenderCommandAtIndex:0];
	@try {
		[c setRenderPipelineState:p];
		[c setMeshBuffer:b offset:0 atIndex:1];
		if (!strcmp(s, "threads"))
			[c drawMeshThreads:MTLSizeMake(10, 1, 1)
			    threadsPerObjectThreadgroup:MTLSizeMake(1, 1, 1)
			      threadsPerMeshThreadgroup:MTLSizeMake(4, 1, 1)];
		else
			[c drawMeshThreadgroups:MTLSizeMake(3, 1, 1)
			    threadsPerObjectThreadgroup:MTLSizeMake(1, 1, 1)
			      threadsPerMeshThreadgroup:MTLSizeMake(4, 1, 1)];
	} @catch (NSException *x) {
		printf("icb.%s: encode threw %s \"%s\"\n", s, x.name.UTF8String, x.reason.UTF8String);
		return;
	}
	id<MTLCommandQueue> q = [D newCommandQueue];
	id<MTLTexture> t = target();
	id<MTLCommandBuffer> cb = [q commandBuffer];
	id<MTLRenderCommandEncoder> e = pass(cb, t);
	[e useResource:b usage:MTLResourceUsageRead stages:MTLRenderStageMesh];
	[e executeCommandsInBuffer:icb withRange:NSMakeRange(0, 1)];
	[e endEncoding];
	char step[32];
	snprintf(step, sizeof step, "icb-%s", s);
	print_columns(step, q, t, cb);
}

// ---- the device -----------------------------------------------------------------------------

static void step_device(void)
{
	printf("device: Metal3=%d Mac2=%d Apple7=%d Apple8=%d Apple9=%d maxThreadsPerThreadgroup=",
	       [D supportsFamily:MTLGPUFamilyMetal3], [D supportsFamily:MTLGPUFamilyMac2],
	       [D supportsFamily:MTLGPUFamilyApple7], [D supportsFamily:MTLGPUFamilyApple8],
	       [D supportsFamily:MTLGPUFamilyApple9]);
	MTLSize s = D.maxThreadsPerThreadgroup;
	printf("%lux%lux%lu maxThreadgroupMemoryLength=%lu\n", (unsigned long)s.width,
	       (unsigned long)s.height, (unsigned long)s.depth,
	       (unsigned long)D.maxThreadgroupMemoryLength);
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
		fprintf(stderr, "usage: %s DEVICE [make|draw|icb|device]\n", argv[0]);
		return 2;
	}
	D = oracle_device(argv[1]);
	if (argc > 3 && !strcmp(argv[2], "--step")) {
		const char *s = argv[3];
		const char *arg = argc > 4 ? argv[4] : "";
		if (!strcmp(s, "make"))
			step_make(arg);
		else if (!strcmp(s, "draw"))
			step_draw(arg);
		else if (!strcmp(s, "icb"))
			step_icb(arg);
		else if (!strcmp(s, "device"))
			step_device();
		else
			return 2;
		return 0;
	}
	oracle_header("oracle_mesh", D);
	printf("# MTL_DEBUG_LAYER=%s\n",
	       getenv("MTL_DEBUG_LAYER") ? getenv("MTL_DEBUG_LAYER") : "unset");
	D = nil;
	const char *only = argc > 2 ? argv[2] : NULL;
	if (!only || !strcmp(only, "device"))
		run_child(argv[0], argv[1], "device", NULL);
	if (!only || !strcmp(only, "make")) {
		const char *steps[] = {"plain",      "noreflect",   "object",     "maxima",
		                       "maxima-odd", "mesh-2048",   "object-2048", "grid-big",
		                       "payload-big", "multiple",   "required",   "icb",
		                       "norast",     "nofragment",  "nomesh",     "vertex-as-mesh",
		                       "msaa4",      "depth",       "archive-miss", "archive-miss-norast",
		                       "archive-miss-object", "archive-miss-object-norast",
		                       "archive-none", "fragment-as-object", "mesh-as-object",
		                       "object-as-mesh", "mesh-as-fragment", "vertex-as-mesh-norast",
		                       "label", "async", "render-api"};
		for (size_t i = 0; i < sizeof steps / sizeof steps[0]; i++)
			run_child(argv[0], argv[1], "make", steps[i]);
	}
	if (!only || !strcmp(only, "draw")) {
		const char *steps[] = {"threadgroups", "threads", "threads-plain", "threads-exact",
		                       "threadgroups-grid", "indirect", "indirect0", "object",
		                       "object-threads", "texture", "zero"};
		for (size_t i = 0; i < sizeof steps / sizeof steps[0]; i++)
			run_child(argv[0], argv[1], "draw", steps[i]);
	}
	if (!only || !strcmp(only, "icb")) {
		run_child(argv[0], argv[1], "icb", "threadgroups");
		run_child(argv[0], argv[1], "icb", "threads");
	}
	return 0;
}
