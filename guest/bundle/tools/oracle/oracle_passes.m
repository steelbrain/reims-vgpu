// oracle_passes: what a render pass's load and store actions leave in its attachments, and how
// the device answers memoryless textures, framebuffer fetch, raster order groups and tile
// pipelines.
//   oracle_passes DEVICE [CASE]   every case (or the one named), each in a child process
// Cases:
//   actions KIND   a 256 x 256 RGBA8 color attachment of KIND (private, managed, iosurface): each
//                  row clears it to A and stores, then runs one pass of the row's actions, with or
//                  without a draw of B over its left half; prints what each half holds after
//   depth          a Depth32Float attachment cleared to 0.25 and stored or not (DontCare)
//   stencil        a Depth32Float_Stencil8 attachment's stencil cleared to 0x25 and stored or not
//   msaa           a 4x RGBA8 attachment cleared to A and resolved, then loaded again by a pass
//                  that only resolves: whether the samples survived a DontCare store
//   features       a framebuffer-fetch pipeline, a raster-order-group pipeline, a tile pipeline
//                  and (last, as Metal may end the process there) a memoryless texture: made or
//                  refused, with Metal's error
// A half prints as a letter when every texel is that byte, else as a histogram.
// Build: see README.md.
#import "oracle_common.h"
#import <IOSurface/IOSurface.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

extern char **environ;

static id<MTLDevice> D;
static id<MTLCommandQueue> Q;
static id<MTLLibrary> L;
static const char *gArgv0;
enum { W = 256, H = 256 };

static void half(const uint8_t *p, size_t stride, int x0, int x1, char *out, size_t cap)
{
	uint8_t vals[4];
	size_t counts[4] = {0}, other = 0, total = 0;
	int nv = 0;
	for (int y = 0; y < H; y++)
		for (int x = x0 * 4; x < x1 * 4; x++) {
			uint8_t v = p[(size_t)y * stride + x];
			total++;
			int k;
			for (k = 0; k < nv && vals[k] != v; k++)
				;
			if (k < nv)
				counts[k]++;
			else if (nv < 4) {
				vals[nv] = v;
				counts[nv++] = 1;
			} else
				other++;
		}
	size_t o = 0;
	out[0] = 0;
	for (int k = 0; k < nv; k++) {
		char l[8];
		if (vals[k] >= 'A' && vals[k] <= 'Z')
			snprintf(l, sizeof l, "%c", vals[k]);
		else
			snprintf(l, sizeof l, "%02x", vals[k]);
		if (nv == 1)
			o += snprintf(out + o, cap - o, "%s", l);
		else
			o += snprintf(out + o, cap - o, "%s%s:%.0f%%", k ? "," : "", l,
			              100.0 * counts[k] / total);
	}
	if (other)
		snprintf(out + o, cap - o, ",other:%.0f%%", 100.0 * other / total);
}

static void read_color(id<MTLTexture> t, char *left, char *right, size_t cap)
{
	id<MTLBuffer> rb = [D newBufferWithLength:W * H * 4 options:MTLResourceStorageModeShared];
	id<MTLCommandBuffer> cb = [Q commandBuffer];
	id<MTLBlitCommandEncoder> e = [cb blitCommandEncoder];
	[e copyFromTexture:t
	         sourceSlice:0
	         sourceLevel:0
	        sourceOrigin:MTLOriginMake(0, 0, 0)
	          sourceSize:MTLSizeMake(W, H, 1)
	            toBuffer:rb
	   destinationOffset:0
	destinationBytesPerRow:W * 4
	destinationBytesPerImage:W * H * 4];
	[e endEncoding];
	oracle_commit_wait(cb, NULL, NULL);
	half(rb.contents, W * 4, 0, W / 2, left, cap);
	half(rb.contents, W * 4, W / 2, W, right, cap);
}

static id<MTLRenderPipelineState> solid_pipeline(MTLPixelFormat color, MTLPixelFormat depth,
                                                 NSUInteger samples)
{
	MTLRenderPipelineDescriptor *pd = [MTLRenderPipelineDescriptor new];
	pd.vertexFunction = [L newFunctionWithName:@"quad_vs"];
	pd.fragmentFunction = [L newFunctionWithName:@"solid_fs"];
	pd.colorAttachments[0].pixelFormat = color;
	pd.depthAttachmentPixelFormat = depth;
	pd.rasterSampleCount = samples;
	NSError *err = nil;
	id<MTLRenderPipelineState> p = [D newRenderPipelineStateWithDescriptor:pd error:&err];
	if (!p)
		printf("  ! solid pipeline: %s\n", err.description.UTF8String);
	return p;
}

static void draw_left(id<MTLRenderCommandEncoder> e, id<MTLRenderPipelineState> p, uint8_t v)
{
	float rect[4] = {-1, -1, 0, 1};
	float z = 0.5f;
	float c = v / 255.0f;
	float color[4] = {c, c, c, c};
	[e setRenderPipelineState:p];
	[e setVertexBytes:rect length:sizeof rect atIndex:0];
	[e setVertexBytes:&z length:sizeof z atIndex:1];
	[e setFragmentBytes:color length:sizeof color atIndex:0];
	[e drawPrimitives:MTLPrimitiveTypeTriangleStrip vertexStart:0 vertexCount:4];
}

static void color_pass(id<MTLTexture> t, MTLLoadAction load, MTLStoreAction store, uint8_t clear,
                       id<MTLRenderPipelineState> p, uint8_t drawn)
{
	double f = clear / 255.0;
	MTLRenderPassDescriptor *rp = [MTLRenderPassDescriptor renderPassDescriptor];
	rp.colorAttachments[0].texture = t;
	rp.colorAttachments[0].loadAction = load;
	rp.colorAttachments[0].storeAction = store;
	rp.colorAttachments[0].clearColor = MTLClearColorMake(f, f, f, f);
	id<MTLCommandBuffer> cb = [Q commandBuffer];
	id<MTLRenderCommandEncoder> e = [cb renderCommandEncoderWithDescriptor:rp];
	if (p)
		draw_left(e, p, drawn);
	[e endEncoding];
	oracle_commit_wait(cb, NULL, NULL);
}

static const char *load_name(MTLLoadAction a)
{
	return a == MTLLoadActionLoad ? "Load" : a == MTLLoadActionClear ? "Clear" : "DontCare";
}

static const char *store_name(MTLStoreAction a)
{
	return a == MTLStoreActionStore ? "Store" : "DontCare";
}

static id<MTLTexture> color_target(const char *kind)
{
	MTLTextureDescriptor *td =
	    [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
	                                                       width:W
	                                                      height:H
	                                                   mipmapped:NO];
	td.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
	if (!strcmp(kind, "iosurface")) {
		td.storageMode = MTLStorageModeManaged;
		NSDictionary *props = @{
			(id)kIOSurfaceWidth : @(W),
			(id)kIOSurfaceHeight : @(H),
			(id)kIOSurfaceBytesPerElement : @4,
			(id)kIOSurfacePixelFormat : @((uint32_t)'RGBA'),
		};
		IOSurfaceRef s = IOSurfaceCreate((__bridge CFDictionaryRef)props);
		id<MTLTexture> t = [D newTextureWithDescriptor:td iosurface:s plane:0];
		CFRelease(s);
		return t;
	}
	td.storageMode = !strcmp(kind, "managed") ? MTLStorageModeManaged : MTLStorageModePrivate;
	return [D newTextureWithDescriptor:td];
}

static void case_actions(const char *kind)
{
	printf("case actions %s: 256x256 RGBA8, cleared to A and stored first\n", kind);
	id<MTLTexture> t = color_target(kind);
	if (!t) {
		printf("  ! no %s texture\n", kind);
		return;
	}
	id<MTLRenderPipelineState> p = solid_pipeline(MTLPixelFormatRGBA8Unorm, MTLPixelFormatInvalid, 1);
	struct {
		MTLLoadAction load;
		MTLStoreAction store;
		BOOL draw;
	} rows[] = {
	    {MTLLoadActionLoad, MTLStoreActionStore, YES},
	    {MTLLoadActionDontCare, MTLStoreActionStore, YES},
	    {MTLLoadActionDontCare, MTLStoreActionStore, NO},
	    {MTLLoadActionLoad, MTLStoreActionDontCare, YES},
	    {MTLLoadActionLoad, MTLStoreActionDontCare, NO},
	    {MTLLoadActionDontCare, MTLStoreActionDontCare, YES},
	    {MTLLoadActionClear, MTLStoreActionStore, YES},
	    {MTLLoadActionClear, MTLStoreActionDontCare, YES},
	    {MTLLoadActionClear, MTLStoreActionDontCare, NO},
	};
	printf("  %-9s %-9s %-5s | %-24s | %-24s\n", "load", "store", "draw", "left (drawn B)",
	       "right (undrawn)");
	for (size_t i = 0; i < sizeof rows / sizeof rows[0]; i++) {
		char l[64], r[64];
		color_pass(t, MTLLoadActionClear, MTLStoreActionStore, 'A', nil, 0);
		color_pass(t, rows[i].load, rows[i].store, 'C', rows[i].draw ? p : nil, 'B');
		read_color(t, l, r, sizeof l);
		printf("  %-9s %-9s %-5s | %-24s | %-24s\n", load_name(rows[i].load),
		       store_name(rows[i].store), rows[i].draw ? "B" : "-", l, r);
	}
	printf("  (Clear rows clear to C)\n");
}

static void case_depth(void)
{
	printf("case depth: Depth32Float 256x256, cleared to 0.25\n");
	MTLTextureDescriptor *td =
	    [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatDepth32Float
	                                                       width:W
	                                                      height:H
	                                                   mipmapped:NO];
	td.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
	td.storageMode = MTLStorageModePrivate;
	id<MTLTexture> t = [D newTextureWithDescriptor:td];
	id<MTLBuffer> rb = [D newBufferWithLength:W * H * 4 options:MTLResourceStorageModeShared];
	double depths[3] = {0.25, 0.75, 0.5};
	MTLStoreAction stores[3] = {MTLStoreActionStore, MTLStoreActionDontCare, MTLStoreActionDontCare};
	MTLLoadAction loads[3] = {MTLLoadActionClear, MTLLoadActionClear, MTLLoadActionDontCare};
	for (int k = 0; k < 3; k++) {
		MTLRenderPassDescriptor *rp = [MTLRenderPassDescriptor renderPassDescriptor];
		rp.depthAttachment.texture = t;
		rp.depthAttachment.loadAction = loads[k];
		rp.depthAttachment.storeAction = stores[k];
		rp.depthAttachment.clearDepth = depths[k];
		id<MTLCommandBuffer> cb = [Q commandBuffer];
		[[cb renderCommandEncoderWithDescriptor:rp] endEncoding];
		id<MTLBlitCommandEncoder> e = [cb blitCommandEncoder];
		[e copyFromTexture:t
		         sourceSlice:0
		         sourceLevel:0
		        sourceOrigin:MTLOriginMake(0, 0, 0)
		          sourceSize:MTLSizeMake(W, H, 1)
		            toBuffer:rb
		   destinationOffset:0
		destinationBytesPerRow:W * 4
		destinationBytesPerImage:W * H * 4];
		[e endEncoding];
		oracle_commit_wait(cb, NULL, NULL);
		float *d = rb.contents;
		size_t same = 0;
		for (size_t i = 0; i < W * H; i++)
			same += d[i] == d[0];
		printf("  load %-8s clear %.2f store %-8s -> depth %g (%zu of %d texels equal it)\n",
		       load_name(loads[k]), depths[k], store_name(stores[k]), d[0], same, W * H);
	}
}

static void case_stencil(void)
{
	printf("case stencil: Depth32Float_Stencil8 256x256, stencil cleared to 0x25 and stored\n");
	MTLTextureDescriptor *td =
	    [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatDepth32Float_Stencil8
	                                                       width:W
	                                                      height:H
	                                                   mipmapped:NO];
	td.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
	td.storageMode = MTLStorageModePrivate;
	id<MTLTexture> t = [D newTextureWithDescriptor:td];
	id<MTLBuffer> rb = [D newBufferWithLength:W * H options:MTLResourceStorageModeShared];
	uint32_t clears[3] = {0x25, 0x5a, 0x77};
	MTLStoreAction stores[3] = {MTLStoreActionStore, MTLStoreActionDontCare, MTLStoreActionDontCare};
	MTLLoadAction loads[3] = {MTLLoadActionClear, MTLLoadActionClear, MTLLoadActionDontCare};
	for (int k = 0; k < 3; k++) {
		MTLRenderPassDescriptor *rp = [MTLRenderPassDescriptor renderPassDescriptor];
		rp.depthAttachment.texture = t;
		rp.depthAttachment.loadAction = MTLLoadActionClear;
		rp.depthAttachment.storeAction = MTLStoreActionStore;
		rp.depthAttachment.clearDepth = 0.5;
		rp.stencilAttachment.texture = t;
		rp.stencilAttachment.loadAction = loads[k];
		rp.stencilAttachment.storeAction = stores[k];
		rp.stencilAttachment.clearStencil = clears[k];
		id<MTLCommandBuffer> cb = [Q commandBuffer];
		[[cb renderCommandEncoderWithDescriptor:rp] endEncoding];
		id<MTLBlitCommandEncoder> e = [cb blitCommandEncoder];
		[e copyFromTexture:t
		         sourceSlice:0
		         sourceLevel:0
		        sourceOrigin:MTLOriginMake(0, 0, 0)
		          sourceSize:MTLSizeMake(W, H, 1)
		            toBuffer:rb
		   destinationOffset:0
		destinationBytesPerRow:W
		destinationBytesPerImage:W * H
		             options:MTLBlitOptionStencilFromDepthStencil];
		[e endEncoding];
		oracle_commit_wait(cb, NULL, NULL);
		const uint8_t *s = rb.contents;
		size_t same = 0;
		for (size_t i = 0; i < W * H; i++)
			same += s[i] == s[0];
		printf("  load %-8s clear 0x%02x store %-8s -> stencil 0x%02x (%zu of %d texels equal it)\n",
		       load_name(loads[k]), clears[k], store_name(stores[k]), s[0], same, W * H);
	}
}

static void case_msaa(void)
{
	printf("case msaa: 4x RGBA8 256x256 cleared to A\n");
	if (![D supportsTextureSampleCount:4]) {
		printf("  4x not supported\n");
		return;
	}
	MTLTextureDescriptor *md =
	    [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
	                                                       width:W
	                                                      height:H
	                                                   mipmapped:NO];
	md.textureType = MTLTextureType2DMultisample;
	md.sampleCount = 4;
	md.usage = MTLTextureUsageRenderTarget;
	md.storageMode = MTLStorageModePrivate;
	id<MTLTexture> ms = [D newTextureWithDescriptor:md];
	id<MTLTexture> r = color_target("private");
	MTLStoreAction firsts[2] = {MTLStoreActionStoreAndMultisampleResolve,
	                            MTLStoreActionMultisampleResolve};
	for (int k = 0; k < 2; k++) {
		MTLRenderPassDescriptor *rp = [MTLRenderPassDescriptor renderPassDescriptor];
		rp.colorAttachments[0].texture = ms;
		rp.colorAttachments[0].resolveTexture = r;
		rp.colorAttachments[0].loadAction = MTLLoadActionClear;
		rp.colorAttachments[0].storeAction = firsts[k];
		double a = 'A' / 255.0;
		rp.colorAttachments[0].clearColor = MTLClearColorMake(a, a, a, a);
		id<MTLCommandBuffer> cb = [Q commandBuffer];
		[[cb renderCommandEncoderWithDescriptor:rp] endEncoding];
		oracle_commit_wait(cb, NULL, NULL);
		color_pass(r, MTLLoadActionClear, MTLStoreActionStore, 'Z', nil, 0);
		rp.colorAttachments[0].loadAction = MTLLoadActionLoad;
		rp.colorAttachments[0].storeAction = MTLStoreActionMultisampleResolve;
		cb = [Q commandBuffer];
		[[cb renderCommandEncoderWithDescriptor:rp] endEncoding];
		oracle_commit_wait(cb, NULL, NULL);
		char l[64], rr[64];
		read_color(r, l, rr, sizeof l);
		printf("  first pass store %-26s; a later Load + resolve gives %s\n",
		       k ? "MultisampleResolve (samples DontCare)" : "StoreAndMultisampleResolve", l);
	}
}

static void case_features(void)
{
	printf("case features\n");
	NSError *err = nil;
	MTLRenderPipelineDescriptor *pd = [MTLRenderPipelineDescriptor new];
	pd.vertexFunction = [L newFunctionWithName:@"quad_vs"];
	pd.fragmentFunction = [L newFunctionWithName:@"fetch_fs"];
	pd.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;
	printf("  framebuffer fetch pipeline: ");
	fflush(stdout);
	id<MTLRenderPipelineState> p = [D newRenderPipelineStateWithDescriptor:pd error:&err];
	printf("%s %s\n", p ? "made" : "nil", err ? err.localizedDescription.UTF8String : "");
	if (p) {
		// Clear to A (0x41), then draw dst * 0.5 + 0 over the left half: 0x20 or 0x21 where the
		// fetch read the attachment.
		id<MTLTexture> t = color_target("private");
		color_pass(t, MTLLoadActionClear, MTLStoreActionStore, 'A', nil, 0);
		color_pass(t, MTLLoadActionLoad, MTLStoreActionStore, 0, p, 0);
		char l[64], r[64];
		read_color(t, l, r, sizeof l);
		printf("  framebuffer fetch draw over A (0x41) at half weight: left %s, right %s\n", l, r);
	}
	err = nil;
	pd.fragmentFunction = [L newFunctionWithName:@"rog_fs"];
	printf("  raster order group pipeline (areRasterOrderGroupsSupported %d): ",
	       (int)D.areRasterOrderGroupsSupported);
	fflush(stdout);
	p = [D newRenderPipelineStateWithDescriptor:pd error:&err];
	printf("%s %s\n", p ? "made" : "nil", err ? err.localizedDescription.UTF8String : "");
	err = nil;
	MTLTileRenderPipelineDescriptor *tp = [MTLTileRenderPipelineDescriptor new];
	tp.tileFunction = [L newFunctionWithName:@"tile_k"];
	tp.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;
	tp.threadgroupSizeMatchesTileSize = YES;
	printf("  tile pipeline: ");
	fflush(stdout);
	p = [D newRenderPipelineStateWithTileDescriptor:tp
	                                        options:MTLPipelineOptionNone
	                                     reflection:nil
	                                          error:&err];
	printf("%s, error %s\n", p ? "made" : "nil", err ? err.description.UTF8String : "nil");
	// Last: Metal may end the process here.
	MTLTextureDescriptor *td =
	    [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
	                                                       width:W
	                                                      height:H
	                                                   mipmapped:NO];
	td.usage = MTLTextureUsageRenderTarget;
	td.storageMode = MTLStorageModeMemoryless;
	printf("  memoryless texture: ");
	fflush(stdout);
	id<MTLTexture> m = [D newTextureWithDescriptor:td];
	printf("%s\n", m ? "made" : "nil");
}

static int run_case(int argc, char **argv)
{
	L = oracle_library(D, gArgv0, "oracle");
	if (!L)
		return 1;
	if (!strcmp(argv[0], "actions"))
		case_actions(argc > 1 ? argv[1] : "private");
	else if (!strcmp(argv[0], "depth"))
		case_depth();
	else if (!strcmp(argv[0], "stencil"))
		case_stencil();
	else if (!strcmp(argv[0], "msaa"))
		case_msaa();
	else if (!strcmp(argv[0], "features"))
		case_features();
	else
		return 2;
	return 0;
}

int main(int argc, char **argv)
{
	setvbuf(stdout, NULL, _IOLBF, 0);
	gArgv0 = argv[0];
	if (argc < 2) {
		fprintf(stderr, "usage: %s DEVICE [CASE [ARG]]\n", argv[0]);
		return 2;
	}
	D = oracle_device(argv[1]);
	if (argc > 2 && !strcmp(argv[2], "--case")) {
		Q = [D newCommandQueue];
		return run_case(argc - 3, argv + 3);
	}
	oracle_header("oracle_passes", D);
	D = nil;
	static const char *const all[][2] = {
	    {"actions", "private"}, {"actions", "managed"}, {"actions", "iosurface"},
	    {"depth", NULL},        {"stencil", NULL},      {"msaa", NULL},
	    {"features", NULL},
	};
	for (size_t i = 0; i < sizeof all / sizeof all[0]; i++) {
		if (argc > 2 && strcmp(argv[2], all[i][0]))
			continue;
		const char *arg = argc > 3 ? argv[3] : all[i][1];
		char *args[] = {argv[0], argv[1], "--case", (char *)all[i][0], (char *)arg, NULL};
		pid_t pid;
		if (posix_spawn(&pid, argv[0], NULL, NULL, args, environ))
			return 1;
		int st;
		waitpid(pid, &st, 0);
		if (WIFSIGNALED(st))
			printf("  => case %s died on signal %d\n", all[i][0], WTERMSIG(st));
		else if (WEXITSTATUS(st))
			printf("  => case %s exited %d\n", all[i][0], WEXITSTATUS(st));
		if (argc > 3)
			break;
	}
	return 0;
}
