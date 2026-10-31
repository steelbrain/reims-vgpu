// archive_probe: the default device's MTLBinaryArchive answers, one per line, so a guest's
// answers can be compared line by line with a host's (the same source builds for both).
//   archive_probe FIXTURES DIR        DIR: an empty directory the probe writes its archives in
//   archive_probe FIXTURES DIR MODE   one add Metal may assert on: nil_compute, nil_vertex,
//                                     nil_fragment, nil_tile
// FIXTURES is tests/suite/fixtures/shaders, whose metallibs (two, fc, vert, frag, render_vdesc,
// tile) give the functions, so a guest that compiles no source runs it too.
// Each line is `key=value`. An NSError prints as `error=DOMAIN/CODE "description"`, a call that
// raises as `threw NAME`. Paths print relative to DIR.
// Build: xcrun clang -arch x86_64 -mmacosx-version-min=26.0 -fobjc-arc -framework Metal \
//        -framework Foundation archive_probe.m -o archive_probe
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>
#include <string.h>

#pragma clang diagnostic ignored "-Wunguarded-availability-new"

static NSString *dir;

static const char *errText(NSError *e)
{
	if (e == nil)
		return "nil";
	NSArray *keys = [e.userInfo.allKeys sortedArrayUsingSelector:@selector(compare:)];
	NSString *s = [NSString stringWithFormat:@"%@/%ld \"%@\" keys=%@", e.domain, (long)e.code,
	                                         e.localizedDescription,
	                                         [keys componentsJoinedByString:@","]];
	s = [s stringByReplacingOccurrencesOfString:dir withString:@"DIR"];
	return s.UTF8String;
}

static NSURL *at(NSString *name)
{
	return [NSURL fileURLWithPath:[dir stringByAppendingPathComponent:name]];
}

static void line(const char *key, BOOL ok, NSError *e)
{
	printf("%s=%d error=%s\n", key, (int)ok, errText(e));
}

static id<MTLBinaryArchive> archive(id<MTLDevice> d, const char *key, NSURL *url)
{
	MTLBinaryArchiveDescriptor *desc = [MTLBinaryArchiveDescriptor new];
	desc.url = url;
	NSError *e = nil;
	id<MTLBinaryArchive> a = nil;
	@try {
		a = [d newBinaryArchiveWithDescriptor:desc error:&e];
		line(key, a != nil, e);
	} @catch (NSException *x) {
		printf("%s=threw %s\n", key, x.name.UTF8String);
	}
	return a;
}

static void fileInfo(const char *key, NSString *name)
{
	NSData *data = [NSData dataWithContentsOfURL:at(name)];
	printf("%s.exists=%d\n", key, data != nil);
	if (data == nil)
		return;
	const uint8_t *b = data.bytes;
	printf("%s.size_nonzero=%d head=", key, data.length != 0);
	for (NSUInteger i = 0; i < MIN(data.length, (NSUInteger)16); i++)
		printf("%02x", b[i]);
	printf("\n");
}

static id<MTLFunction> fn(id<MTLLibrary> l, NSString *name)
{
	id<MTLFunction> f = [l newFunctionWithName:name];
	if (f == nil)
		printf("function %s=nil\n", name.UTF8String);
	return f;
}

static id<MTLFunction> g_vs, g_fs, g_vs2;

static MTLRenderPipelineDescriptor *render(id<MTLLibrary> l)
{
	(void)l;
	MTLRenderPipelineDescriptor *r = [MTLRenderPipelineDescriptor new];
	r.vertexFunction = g_vs;
	r.fragmentFunction = g_fs;
	r.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
	return r;
}

static MTLComputePipelineDescriptor *compute(id<MTLFunction> f)
{
	MTLComputePipelineDescriptor *c = [MTLComputePipelineDescriptor new];
	// Metal's setter asserts on nil, so a descriptor of no function is one never set.
	if (f != nil)
		c.computeFunction = f;
	return c;
}

static void addCompute(id<MTLBinaryArchive> a, const char *key, MTLComputePipelineDescriptor *c)
{
	NSError *e = nil;
	@try {
		BOOL ok = [a addComputePipelineFunctionsWithDescriptor:c error:&e];
		line(key, ok, e);
	} @catch (NSException *x) {
		printf("%s=threw %s\n", key, x.name.UTF8String);
	}
}

static void addRender(id<MTLBinaryArchive> a, const char *key, MTLRenderPipelineDescriptor *r)
{
	NSError *e = nil;
	@try {
		BOOL ok = [a addRenderPipelineFunctionsWithDescriptor:r error:&e];
		line(key, ok, e);
	} @catch (NSException *x) {
		printf("%s=threw %s\n", key, x.name.UTF8String);
	}
}

// A compute pipeline of `c` over `archives`, failing on a miss where `fail`.
static void makeCompute(id<MTLDevice> d, const char *key, MTLComputePipelineDescriptor *c,
                        NSArray *archives, BOOL fail)
{
	c.binaryArchives = archives;
	NSError *e = nil;
	@try {
		id p = [d newComputePipelineStateWithDescriptor:c
		                                        options:fail ? MTLPipelineOptionFailOnBinaryArchiveMiss
		                                                     : MTLPipelineOptionNone
		                                     reflection:nil
		                                          error:&e];
		line(key, p != nil, e);
	} @catch (NSException *x) {
		printf("%s=threw %s\n", key, x.name.UTF8String);
	}
}

static void makeRender(id<MTLDevice> d, const char *key, MTLRenderPipelineDescriptor *r,
                       NSArray *archives, BOOL fail)
{
	r.binaryArchives = archives;
	NSError *e = nil;
	@try {
		id p = [d newRenderPipelineStateWithDescriptor:r
		                                       options:fail ? MTLPipelineOptionFailOnBinaryArchiveMiss
		                                                    : MTLPipelineOptionNone
		                                    reflection:nil
		                                         error:&e];
		line(key, p != nil, e);
	} @catch (NSException *x) {
		printf("%s=threw %s\n", key, x.name.UTF8String);
	}
}

static id<MTLLibrary> load(id<MTLDevice> d, NSString *fixtures, NSString *name)
{
	NSError *e = nil;
	NSURL *url = [NSURL fileURLWithPath:[fixtures stringByAppendingPathComponent:
	                                                  [name stringByAppendingString:@".metallib"]]];
	id<MTLLibrary> l = [d newLibraryWithURL:url error:&e];
	if (l == nil)
		printf("library %s error=%s\n", name.UTF8String, errText(e));
	return l;
}

// The vertex descriptor render_vdesc reads: a float2 and a float4 from buffer 0.
static MTLVertexDescriptor *vdesc(void)
{
	MTLVertexDescriptor *vd = [MTLVertexDescriptor vertexDescriptor];
	vd.attributes[0].format = MTLVertexFormatFloat2;
	vd.attributes[1].format = MTLVertexFormatFloat4;
	vd.attributes[1].offset = 8;
	vd.layouts[0].stride = 24;
	return vd;
}

int main(int argc, char **argv)
{
	@autoreleasepool {
		if (argc != 3 && argc != 4) {
			fprintf(stderr, "usage: archive_probe FIXTURES DIR [nil_compute|nil_vertex|nil_fragment|nil_tile]\n");
			return 2;
		}
		NSString *fixtures = [NSString stringWithUTF8String:argv[1]];
		dir = [NSString stringWithUTF8String:argv[2]];
		id<MTLDevice> d = MTLCreateSystemDefaultDevice();
		printf("device=%s\n", d.name.UTF8String);
		NSError *e = nil;
		id<MTLLibrary> two = load(d, fixtures, @"two"), fc = load(d, fixtures, @"fc"),
		               vert = load(d, fixtures, @"vert"), frag = load(d, fixtures, @"frag"),
		               vdl = load(d, fixtures, @"render_vdesc"), tilel = load(d, fixtures, @"tile");
		printf("library=%d\n", two && fc && vert && frag && vdl && tilel);
		if (!(two && fc && vert && frag && vdl && tilel))
			return 1;
		id<MTLLibrary> lib = two;
		id<MTLFunction> k = fn(two, @"a"), k2 = fn(two, @"b");
		g_vs = fn(vert, @"tri_vert");
		g_fs = fn(frag, @"tri_frag");
		g_vs2 = fn(vdl, @"render_vdesc");
		MTLFunctionConstantValues *v1 = [MTLFunctionConstantValues new], *v2 = [MTLFunctionConstantValues new];
		uint32_t one = 1, two_ = 2;
		[v1 setConstantValue:&one type:MTLDataTypeUInt atIndex:0];
		[v2 setConstantValue:&two_ type:MTLDataTypeUInt atIndex:0];
		id<MTLFunction> kc1 = [fc newFunctionWithName:@"fc_kernel" constantValues:v1 error:&e];
		id<MTLFunction> kc2 = [fc newFunctionWithName:@"fc_kernel" constantValues:v2 error:&e];
		printf("constants=%d\n", kc1 != nil && kc2 != nil);

		// The adds Metal may assert on, one a run.
		if (argc == 4) {
			id<MTLBinaryArchive> a = archive(d, "new.for_nil", nil);
			const char *mode = argv[3];
			if (strcmp(mode, "nil_tile") == 0) {
				MTLTileRenderPipelineDescriptor *t = [MTLTileRenderPipelineDescriptor new];
				t.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
				NSError *te = nil;
				@try {
					BOOL ok = [a addTileRenderPipelineFunctionsWithDescriptor:t error:&te];
					line("add.tile_nil_function", ok, te);
				} @catch (NSException *x) {
					printf("add.tile_nil_function=threw %s\n", x.name.UTF8String);
				}
			}
			if (strcmp(mode, "nil_compute") == 0)
				addCompute(a, "add.compute_nil_function", compute(nil));
			MTLRenderPipelineDescriptor *r = render(lib);
			if (strcmp(mode, "nil_vertex") == 0) {
				r.vertexFunction = nil;
				addRender(a, "add.render_nil_vertex", r);
			}
			if (strcmp(mode, "nil_fragment") == 0) {
				r.fragmentFunction = nil;
				addRender(a, "add.render_nil_fragment", r);
			}
			return 0;
		}

		// Making one.
		id<MTLBinaryArchive> empty = archive(d, "new.no_url", nil);
		printf("new.no_url.label=%s device_is_default=%d\n",
		       empty.label ? empty.label.UTF8String : "nil", empty.device == d);
		empty.label = @"probe";
		printf("new.no_url.label_set=%s\n", empty.label.UTF8String);
		archive(d, "new.missing_file", at(@"missing.bin"));
		[[NSData data] writeToURL:at(@"empty.bin") atomically:NO];
		archive(d, "new.empty_file", at(@"empty.bin"));
		[[@"not an archive at all" dataUsingEncoding:NSUTF8StringEncoding] writeToURL:at(@"garbage.bin")
		                                                                    atomically:NO];
		archive(d, "new.garbage_file", at(@"garbage.bin"));
		[[NSFileManager defaultManager] createDirectoryAtURL:at(@"subdir")
		                         withIntermediateDirectories:YES
		                                          attributes:nil
		                                               error:nil];
		archive(d, "new.directory", at(@"subdir"));
		archive(d, "new.http_url", [NSURL URLWithString:@"https://example.invalid/a.bin"]);

		// Serializing one with nothing in it.
		e = nil;
		BOOL ok = [empty serializeToURL:at(@"empty-archive.bin") error:&e];
		line("serialize.empty", ok, e);
		fileInfo("serialize.empty", @"empty-archive.bin");
		archive(d, "load.empty_archive", at(@"empty-archive.bin"));
		e = nil;
		ok = [empty serializeToURL:at(@"no/such/dir/a.bin") error:&e];
		line("serialize.missing_dir", ok, e);
		e = nil;
		@try {
			ok = [empty serializeToURL:[NSURL URLWithString:@"https://example.invalid/a.bin"]
			                     error:&e];
			line("serialize.http_url", ok, e);
		} @catch (NSException *x) {
			printf("serialize.http_url=threw %s\n", x.name.UTF8String);
		}

		// Adding to one.
		id<MTLBinaryArchive> a = archive(d, "new.for_adds", nil);
		addCompute(a, "add.compute", compute(k));
		addCompute(a, "add.compute_again", compute(k));
		addCompute(a, "add.compute_kc1", compute(kc1));
		addRender(a, "add.render", render(lib));
		MTLTileRenderPipelineDescriptor *tile = [MTLTileRenderPipelineDescriptor new];
		tile.tileFunction = fn(tilel, @"tile_k");
		tile.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
		e = nil;
		@try {
			ok = [a addTileRenderPipelineFunctionsWithDescriptor:tile error:&e];
			line("add.tile", ok, e);
		} @catch (NSException *x) {
			printf("add.tile=threw %s\n", x.name.UTF8String);
		}
		MTLFunctionDescriptor *fd = [MTLFunctionDescriptor new];
		fd.name = @"b";
		e = nil;
		ok = [a addFunctionWithDescriptor:fd library:lib error:&e];
		line("add.function", ok, e);
		fd.name = @"no_such_function";
		e = nil;
		@try {
			ok = [a addFunctionWithDescriptor:fd library:lib error:&e];
			line("add.function_missing", ok, e);
		} @catch (NSException *x) {
			printf("add.function_missing=threw %s\n", x.name.UTF8String);
		}

		// Creating pipelines over the archive it was added to.
		NSArray *in = @[ a ];
		makeCompute(d, "hit.compute", compute(k), in, YES);
		makeCompute(d, "miss.compute_k2", compute(k2), in, YES);
		makeCompute(d, "hit.compute_kc1", compute(kc1), in, YES);
		makeCompute(d, "miss.compute_kc2", compute(kc2), in, YES);
		makeCompute(d, "nofail.compute_k2", compute(k2), in, NO);
		makeCompute(d, "miss.compute_no_archives", compute(k), nil, YES);
		makeCompute(d, "miss.compute_empty_archives", compute(k), @[], YES);
		makeCompute(d, "miss.compute_empty_archive", compute(k), @[ empty ], YES);
		makeCompute(d, "hit.compute_second_archive", compute(k), @[ empty, a ], YES);
		MTLComputePipelineDescriptor *c = compute(k);
		c.label = @"another label";
		makeCompute(d, "variant.compute_label", c, in, YES);
		c = compute(k);
		c.maxTotalThreadsPerThreadgroup = 64;
		makeCompute(d, "variant.compute_max_threads", c, in, YES);
		c = compute(k);
		c.threadGroupSizeIsMultipleOfThreadExecutionWidth = YES;
		makeCompute(d, "variant.compute_multiple_of_width", c, in, YES);
		c = compute(k);
		c.buffers[0].mutability = MTLMutabilityImmutable;
		makeCompute(d, "variant.compute_buffer_mutability", c, in, YES);
		e = nil;
		@try {
			id p = [d newComputePipelineStateWithFunction:k
			                                      options:MTLPipelineOptionFailOnBinaryArchiveMiss
			                                   reflection:nil
			                                        error:&e];
			line("miss.compute_function_only", p != nil, e);
		} @catch (NSException *x) {
			printf("miss.compute_function_only=threw %s\n", x.name.UTF8String);
		}

		makeRender(d, "hit.render", render(lib), in, YES);
		MTLRenderPipelineDescriptor *r = render(lib);
		r.label = @"another label";
		makeRender(d, "variant.render_label", r, in, YES);
		r = render(lib);
		r.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;
		makeRender(d, "variant.render_pixel_format", r, in, YES);
		r = render(lib);
		r.colorAttachments[0].blendingEnabled = YES;
		makeRender(d, "variant.render_blending", r, in, YES);
		r = render(lib);
		r.colorAttachments[0].writeMask = MTLColorWriteMaskRed;
		makeRender(d, "variant.render_write_mask", r, in, YES);
		r = render(lib);
		r.rasterSampleCount = 4;
		makeRender(d, "variant.render_sample_count", r, in, YES);
		r = render(lib);
		r.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float;
		makeRender(d, "variant.render_depth_format", r, in, YES);
		r = render(lib);
		r.alphaToCoverageEnabled = YES;
		makeRender(d, "variant.render_alpha_to_coverage", r, in, YES);
		r = render(lib);
		r.inputPrimitiveTopology = MTLPrimitiveTopologyClassTriangle;
		makeRender(d, "variant.render_topology", r, in, YES);
		r = render(lib);
		r.vertexBuffers[0].mutability = MTLMutabilityImmutable;
		makeRender(d, "variant.render_buffer_mutability", r, in, YES);
		r = render(lib);
		r.stencilAttachmentPixelFormat = MTLPixelFormatStencil8;
		makeRender(d, "variant.render_stencil_format", r, in, YES);
		r = render(lib);
		r.alphaToOneEnabled = YES;
		makeRender(d, "variant.render_alpha_to_one", r, in, YES);
		r = render(lib);
		r.rasterizationEnabled = NO;
		makeRender(d, "variant.render_rasterization_off", r, in, YES);
		r = render(lib);
		r.maxVertexAmplificationCount = 2;
		makeRender(d, "variant.render_amplification", r, in, YES);
		r = render(lib);
		r.colorAttachments[1].pixelFormat = MTLPixelFormatRGBA8Unorm;
		makeRender(d, "variant.render_second_color", r, in, YES);
		r = render(lib);
		r.colorAttachments[0].sourceRGBBlendFactor = MTLBlendFactorZero;
		makeRender(d, "variant.render_blend_factor_unused", r, in, YES);
		r = render(lib);
		r.shaderValidation = MTLShaderValidationDisabled;
		makeRender(d, "variant.render_shader_validation", r, in, YES);
		r = render(lib);
		r.supportIndirectCommandBuffers = YES;
		makeRender(d, "variant.render_icb", r, in, YES);
		r = render(lib);
		r.fragmentFunction = nil;
		makeRender(d, "miss.render_no_fragment", r, in, YES);
		addRender(a, "add.render_no_fragment", r);
		makeRender(d, "hit.render_no_fragment", r, in, YES);
		r = render(lib);
		r.vertexFunction = g_vs2;
		r.vertexDescriptor = vdesc();
		makeRender(d, "miss.render_vs2", r, in, YES);
		addRender(a, "add.render_vs2", r);
		makeRender(d, "hit.render_vs2", r, in, YES);
		r = render(lib);
		r.vertexFunction = g_vs2;
		r.vertexDescriptor = vdesc();
		r.vertexDescriptor.layouts[0].stride = 32;
		makeRender(d, "variant.render_vertex_stride", r, in, YES);
		r = render(lib);
		r.vertexFunction = g_vs2;
		r.vertexDescriptor = vdesc();
		r.vertexDescriptor.attributes[0].format = MTLVertexFormatHalf2;
		makeRender(d, "variant.render_vertex_format", r, in, YES);
		makeRender(d, "nofail.render_pixel_format_miss", r, in, NO);
		MTLRenderPipelineDescriptor *(^vs2)(void) = ^{
			MTLRenderPipelineDescriptor *r = render(lib);
			r.vertexFunction = g_vs2;
			r.vertexDescriptor = vdesc();
			return r;
		};
		r = vs2();
		r.vertexDescriptor.attributes[1].offset = 12;
		r.vertexDescriptor.layouts[0].stride = 28;
		makeRender(d, "variant.render_vertex_offset", r, in, YES);
		r = vs2();
		r.vertexDescriptor.attributes[0].bufferIndex = 1;
		r.vertexDescriptor.attributes[1].bufferIndex = 1;
		r.vertexDescriptor.layouts[1].stride = 24;
		r.vertexDescriptor.layouts[0].stride = 0;
		makeRender(d, "variant.render_vertex_buffer_index", r, in, YES);
		r = vs2();
		r.vertexDescriptor.layouts[0].stepFunction = MTLVertexStepFunctionPerInstance;
		makeRender(d, "variant.render_vertex_step_function", r, in, YES);
		r = vs2();
		r.vertexDescriptor.layouts[0].stepFunction = MTLVertexStepFunctionPerInstance;
		r.vertexDescriptor.layouts[0].stepRate = 2;
		makeRender(d, "variant.render_vertex_step_rate", r, in, YES);
		r = render(lib);
		r.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;
		r.inputPrimitiveTopology = MTLPrimitiveTopologyClassTriangle;
		makeRender(d, "variant.render_format_and_topology", r, in, YES);
		r = render(lib);
		r.colorAttachments[0].blendingEnabled = YES;
		addRender(a, "add.render_blending", r);
		r = render(lib);
		r.colorAttachments[0].blendingEnabled = YES;
		r.colorAttachments[0].sourceRGBBlendFactor = MTLBlendFactorZero;
		makeRender(d, "variant.render_blend_factor", r, in, YES);
		r = render(lib);
		r.colorAttachments[0].blendingEnabled = YES;
		r.colorAttachments[0].rgbBlendOperation = MTLBlendOperationMax;
		makeRender(d, "variant.render_blend_operation", r, in, YES);
		c = compute(k);
		c.supportIndirectCommandBuffers = YES;
		makeCompute(d, "variant.compute_icb", c, in, YES);
		c = compute(k);
		c.maxCallStackDepth = 4;
		makeCompute(d, "variant.compute_call_stack_depth", c, in, YES);
		c = compute(k);
		c.shaderValidation = MTLShaderValidationDisabled;
		makeCompute(d, "variant.compute_shader_validation", c, in, YES);

		// Serializing and loading it back.
		e = nil;
		ok = [a serializeToURL:at(@"full.bin") error:&e];
		line("serialize.full", ok, e);
		fileInfo("serialize.full", @"full.bin");
		e = nil;
		ok = [a serializeToURL:at(@"full.bin") error:&e];
		line("serialize.full_overwrite", ok, e);
		id<MTLBinaryArchive> loaded = archive(d, "load.full", at(@"full.bin"));
		NSArray *lin = loaded ? @[ loaded ] : @[];
		makeCompute(d, "loaded.hit.compute", compute(k), lin, YES);
		makeCompute(d, "loaded.hit.compute_kc1", compute(kc1), lin, YES);
		makeCompute(d, "loaded.hit.function_k2", compute(k2), lin, YES);
		makeRender(d, "loaded.hit.render", render(lib), lin, YES);
		makeCompute(d, "loaded.miss.compute_kc2", compute(kc2), lin, YES);
		addCompute(loaded, "loaded.add.compute_kc2", compute(kc2));
		e = nil;
		ok = [loaded serializeToURL:at(@"full2.bin") error:&e];
		line("loaded.serialize", ok, e);

		// The archive's file, each slice's architecture changed: a fat header (big-endian), then
		// nfat_arch entries of cputype, cpusubtype, offset, size and align.
		NSMutableData *bytes = [NSMutableData dataWithContentsOfURL:at(@"full.bin")];
		if (bytes.length >= 8) {
			uint32_t *w = (uint32_t *)bytes.mutableBytes;
			uint32_t nfat = CFSwapInt32BigToHost(w[1]);
			printf("full.magic=%08x nfat=%u\n", CFSwapInt32BigToHost(w[0]), nfat);
			for (uint32_t i = 0; i < nfat && 2 + 5 * i + 4 < bytes.length / 4; i++)
				printf("full.slice%u cputype=%08x cpusubtype=%08x\n", i,
				       CFSwapInt32BigToHost(w[2 + 5 * i]), CFSwapInt32BigToHost(w[3 + 5 * i]));
			NSMutableData *sub = [bytes mutableCopy];
			uint32_t *sw = (uint32_t *)sub.mutableBytes;
			for (uint32_t i = 0; i < nfat && 2 + 5 * i + 4 < sub.length / 4; i++)
				sw[3 + 5 * i] = CFSwapInt32HostToBig(CFSwapInt32BigToHost(sw[3 + 5 * i]) + 1);
			[sub writeToURL:at(@"subtype.bin") atomically:NO];
			id<MTLBinaryArchive> other = archive(d, "load.other_subtype", at(@"subtype.bin"));
			if (other != nil) {
				makeCompute(d, "other_subtype.compute", compute(k), @[ other ], YES);
				makeRender(d, "other_subtype.render", render(lib), @[ other ], YES);
				addCompute(other, "other_subtype.add", compute(k2));
			}
			NSMutableData *type = [bytes mutableCopy];
			uint32_t *tw = (uint32_t *)type.mutableBytes;
			for (uint32_t i = 0; i < nfat && 2 + 5 * i + 4 < type.length / 4; i++)
				tw[2 + 5 * i] = CFSwapInt32HostToBig(CFSwapInt32BigToHost(tw[2 + 5 * i]) ^ 0x4);
			[type writeToURL:at(@"cputype.bin") atomically:NO];
			id<MTLBinaryArchive> otherType = archive(d, "load.other_cputype", at(@"cputype.bin"));
			if (otherType != nil)
				makeCompute(d, "other_cputype.compute", compute(k), @[ otherType ], YES);
			NSMutableData *cut = [[bytes subdataWithRange:NSMakeRange(0, bytes.length / 2)] mutableCopy];
			[cut writeToURL:at(@"cut.bin") atomically:NO];
			archive(d, "load.truncated", at(@"cut.bin"));
		}
	}
	return 0;
}
