// gameload: a game-shaped Metal load, offscreen, for a fixed number of frames, to rank what a
// guest's frames cost against a real Mac's. Each frame holds the shapes AAA Metal ports make, each
// behind its own switch (--off NAME,...), so a run with one off measures its share:
//   heap     MTLHeap streaming: a placement heap whose texture slots (512 x 512, full mip chains,
//            RGBA8Unorm_sRGB, and BC7 sRGB on every other slot where the device has BC) are made
//            again at their offsets each frame, two a frame, with every mip uploaded, and a 1 MiB
//            vertex buffer placed and filled each frame; per frame in flight, an automatic heap
//            whose transient textures (the bloom chain) are made each frame and made aliasable as
//            soon as the chain is done with each; both heaps untracked, ordered by fences; the
//            materials' argument buffer encoded again each frame over the new slots; useHeap.
//            Off: the same passes over textures and buffers the device made once.
//   rtheap   the render targets, shadow map and tone map output placed once in a placement heap
//            of their own (tracked), as a port of placed resources makes them. Off: the device's.
//   targets  render targets: the scene into a 4x MSAA RGBA16Float color and Depth32Float depth,
//            resolved into a single-sample RGBA16Float, and a UI pass of 64 quads.
//            Off: the scene into a single-sample RGBA16Float with no depth, and no UI pass.
//   views    the UI pass into a BGRA8Unorm view of an RGBA8Unorm texture made with PixelFormatView
//            usage, and the composite into an RGBA8Unorm_sRGB view of an RGBA8Unorm texture made
//            without it. Off, or with targets off: both into the RGBA8Unorm textures themselves.
//   icb      a GPU-driven pass: a compute kernel culls ICB_OBJECTS objects and encodes each one's
//            indexed draw into an indirect command buffer (or resets its command), which the
//            scene pass executes; the commands inherit the pipeline, and the encoder's buffers
//            unless -b, and sample only through the material's argument buffer (icb_fs).
//   cicb     a GPU-driven compute pass: a kernel encodes CICB_COMMANDS concurrent dispatches into
//            a compute indirect command buffer (every fifth reset), and a later compute encoder
//            sets the pipeline and its buffers at non-zero offsets and executes it; the commands
//            inherit the pipeline, and the encoder's buffers unless -b. The dispatches sum their
//            input into a word, which the CPU checks against its own sum as the frame slot comes
//            back: a wrong sum is counted, and the run exits 1 (see cicb_bad).
//   draws    DRAWS CPU-encoded indexed draws in the scene pass (8 pipeline changes over 4
//            function-constant variants, each draw its own object-buffer offset and material
//            argument-buffer offset) and DRAWS/4 in a 2048 x 2048 Depth32Float shadow pass.
//   post     compute post-processing: a luminance histogram (threadgroup and device atomics), a
//            4-level bloom down chain and 3 up, and a tone map into an RGBA8Unorm (shader write),
//            with texture barriers between dependent dispatches.
//            Off: the composite samples the scene's resolve directly.
//   blit     one blit stream a frame: a 512 x 512 RGBA8 texture's level 0 and a 256 KiB buffer
//            uploaded from a Shared staging buffer, the texture's mipmaps generated, and a
//            256 x 256 read-back of the final image into a Shared buffer, which the CPU reads.
// A frame is two command buffers (uploads, shadow, the ICB's encode and the scene; then post,
// UI, composite and read-back), FRAMES_IN_FLIGHT at most in flight.
//
// The buffers the GPU reads and the CPU writes are Shared by default, the frame's constants and
// the materials' argument buffers each frame and the ICBs' argument buffers once, so a discrete
// GPU reads them in system memory. --private NAME,... puts each named one in a Private buffer
// instead, which the GPU reads in its own memory:
//   frame      the frame's constants, copied from their Shared buffer by a blit at the start of
//              the frame's first command buffer (in the upload blit where there is one).
//   materials  the materials' argument buffers, copied the same way where the CPU encodes them
//              each frame (heap on), else once before the frames.
//   args       the ICB's and the compute ICB's argument buffers, copied once before the frames.
//
//   gameload [-d NAME] [-f FRAMES] [-w WARMUP] [-s WxH] [-n DRAWS] [-i ICB_OBJECTS]
//            [-c CICB_COMMANDS] [-k FRAMES_IN_FLIGHT] [-b] [--off NAME,...] [--private NAME,...]
//            [-q] [-l]
//     -d NAME   the first of MTLCopyAllDevices whose name holds NAME (default: the first);
//               never MTLCreateSystemDefaultDevice, which switches a dual-GPU MacBook's display.
//     -f        measured frames (default 300), after -w warm-up frames (default 30).
//     -s        the scene's size (default 1920x1080).
//     -n        CPU draws in the scene pass (default 4000).
//     -i        the ICB pass's objects (default 8192).
//     -c        the compute ICB's commands (default 512).
//     -k        frames in flight (default 2).
//     -b        the ICBs inherit no buffers: their kernels set each drawn command's vertex and
//               fragment buffers (icb_encode_bound) and each dispatch's kernel buffers
//               (cicb_encode_bound); by default they inherit the encoder's.
//     --off     the shapes to leave out (above); --private, the buffers to make Private (above).
//               Both may be given more than once.
//     -q        no per-frame lines.
//     -l        lists the devices and exits.
//
// Output, all on stdout:
//   "GAMELOAD config ..."            the device, switches and sizes.
//   "GAMELOAD window start EPOCH"    the wall clock (s since 1970, to the us) as the measured
//   "GAMELOAD window end EPOCH"      frames start and end, for a host's logs of the same span.
//   "F N cpu C wait W wall L gpu G a A b B"  one line a measured frame, in ms: the CPU's encode
//            and commit time (C), the wait for a frame slot before it (W), the time from the
//            frame before's start to this one's (L), the GPU's span from the first command
//            buffer's GPUStartTime to the second's GPUEndTime (G), and each buffer's own span.
//   "GAMELOAD shape NAME cpu_ms P50 MEAN"   each shape's own CPU encode time a frame.
//   "GAMELOAD summary ..."          per-frame p50/p90/mean of each number, fps, the work a frame
//            holds (draws, dispatches, passes, heap objects made), command-buffer errors, and
//            the compute ICB's sums checked (cicb_checks) and wrong (cicb_bad).
//   "GAMELOAD cicb wrong sum ..."   the first wrong sum, when there is one.
// Shaders compile from source at run time (newLibraryWithSource:), from gameload.metal, which
// the build embeds (make -C guest/bundle/tools/gameload).
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <dispatch/dispatch.h>
#include <mach/mach_time.h>
#include <simd/simd.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/sysctl.h>
#include <sys/time.h>

#include "gameload_source.h"

#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#pragma clang diagnostic ignored "-Wunguarded-availability-new"

enum { S_HEAP, S_RTHEAP, S_TARGETS, S_VIEWS, S_ICB, S_CICB, S_DRAWS, S_POST, S_BLIT, S_COUNT };
static const char *const kShape[S_COUNT] = {"heap", "rtheap", "targets", "views", "icb", "cicb", "draws", "post", "blit"};
static bool on[S_COUNT] = {true, true, true, true, true, true, true, true, true};
// The buffers --private makes Private (the header).
enum { P_FRAME, P_MATERIALS, P_ARGS, P_COUNT };
static const char *const kPrivate[P_COUNT] = {"frame", "materials", "args"};
static bool priv[P_COUNT];

typedef struct {
	simd_float4x4 view_proj;
	simd_float4x4 light_view_proj;
	simd_float4 light_dir;
	float time;
	uint32_t objects;
	uint32_t icb_objects;
	uint32_t cicb_commands;
} GlFrame;

typedef struct {
	simd_float4 pos_scale;
	simd_float4 color;
	simd_float4 rot;
	uint32_t material, pad0, pad1, pad2;
} GlObject;

typedef struct {
	float pos[3], nrm[3], uv[2];
} GlVertex;

enum {
	SLOTS = 16,         // stream slots in the placement heap
	SLOT_SIZE = 512,    // their width and height
	STREAMED = 2,       // slots made again each frame
	MATERIALS = 64,     // materials in the argument buffer
	STATIC_TEX = 8,     // textures made once, for materials and details
	VARIANTS = 4,       // scene pipeline variants
	STREAM_BUF = 1 << 20,
	BLIT_TEX = 512,
	BLIT_BUF = 256 << 10,
	READBACK = 256,
	BLOOM = 4,
	MAX_FRAMES = 100000,
	CICB_LEAD = 64,     // words before the compute ICB's input (its binding's offset, 256 bytes)
	CICB_SLOT = 256,    // bytes a frame in flight's sum takes; the sum lies 64 bytes into it
};

static double now_ms(void)
{
	static mach_timebase_info_data_t tb;
	if (tb.denom == 0)
		mach_timebase_info(&tb);
	return (double)mach_absolute_time() * tb.numer / tb.denom / 1e6;
}

static double epoch_s(void)
{
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return tv.tv_sec + tv.tv_usec / 1e6;
}

static void die(const char *what, NSError *err)
{
	fprintf(stderr, "gameload: %s%s%s\n", what, err ? ": " : "", err ? err.description.UTF8String : "");
	exit(1);
}

static int cmp_double(const void *a, const void *b)
{
	double x = *(const double *)a, y = *(const double *)b;
	return x < y ? -1 : x > y;
}

// p50, p90 and the mean of n values (sorted in a copy).
static void stats(const double *v, int n, double *p50, double *p90, double *mean)
{
	double *s = malloc(sizeof(double) * (n ? n : 1));
	double sum = 0;
	for (int i = 0; i < n; i++) {
		s[i] = v[i];
		sum += v[i];
	}
	qsort(s, n, sizeof(double), cmp_double);
	*p50 = n ? s[n / 2] : 0;
	*p90 = n ? s[(int)(n * 0.9)] : 0;
	*mean = n ? sum / n : 0;
	free(s);
}

static simd_float4x4 perspective(float fovy, float aspect, float near, float far)
{
	float y = 1 / tanf(fovy * 0.5f), x = y / aspect, z = far / (near - far);
	return (simd_float4x4){{{x, 0, 0, 0}, {0, y, 0, 0}, {0, 0, z, -1}, {0, 0, z * near, 0}}};
}

static simd_float4x4 ortho(float w, float h, float near, float far)
{
	return (simd_float4x4){{{2 / w, 0, 0, 0}, {0, 2 / h, 0, 0}, {0, 0, 1 / (near - far), 0},
	                        {0, 0, near / (near - far), 1}}};
}

static simd_float4x4 look_at(simd_float3 eye, simd_float3 at, simd_float3 up)
{
	simd_float3 z = simd_normalize(eye - at), x = simd_normalize(simd_cross(up, z)), y = simd_cross(z, x);
	return (simd_float4x4){{{x.x, y.x, z.x, 0}, {x.y, y.y, z.y, 0}, {x.z, y.z, z.z, 0},
	                        {-simd_dot(x, eye), -simd_dot(y, eye), -simd_dot(z, eye), 1}}};
}

static uint32_t rng_state = 0x12345678;
static uint32_t rnd(void)
{
	rng_state ^= rng_state << 13;
	rng_state ^= rng_state >> 17;
	rng_state ^= rng_state << 5;
	return rng_state;
}
static float frnd(void) { return (rnd() & 0xffffff) / (float)0x1000000; }

static id<MTLDevice> D;
static id<MTLCommandQueue> Q;
// The textures the materials' argument buffer names: the stream slots, made again as they stream,
// and the textures made once.
static id<MTLTexture> slot_tex[SLOTS], statics[STATIC_TEX];
static atomic_int cb_errors;
static char first_error[512];
// The compute ICB's sums checked and wrong, and the first wrong one.
static uint64_t cicb_checks, cicb_bad;
static char cicb_first_bad[160];

// A Private buffer holding `len` bytes of `bytes`, uploaded through a Shared one.
static id<MTLBuffer> private_buffer(const void *bytes, size_t len)
{
	id<MTLBuffer> src = [D newBufferWithBytes:bytes length:len options:MTLResourceStorageModeShared];
	id<MTLBuffer> dst = [D newBufferWithLength:len options:MTLResourceStorageModePrivate];
	id<MTLCommandBuffer> cb = [Q commandBuffer];
	id<MTLBlitCommandEncoder> b = [cb blitCommandEncoder];
	[b copyFromBuffer:src sourceOffset:0 toBuffer:dst destinationOffset:0 size:len];
	[b endEncoding];
	[cb commit];
	[cb waitUntilCompleted];
	return dst;
}

// A Private buffer of `src`'s length, its bytes copied from `src` by a blit (--private).
static id<MTLBuffer> private_copy(id<MTLBuffer> src)
{
	id<MTLBuffer> dst = [D newBufferWithLength:src.length options:MTLResourceStorageModePrivate];
	id<MTLCommandBuffer> cb = [Q commandBuffer];
	id<MTLBlitCommandEncoder> b = [cb blitCommandEncoder];
	[b copyFromBuffer:src sourceOffset:0 toBuffer:dst destinationOffset:0 size:src.length];
	[b endEncoding];
	[cb commit];
	[cb waitUntilCompleted];
	return dst;
}

static MTLTextureDescriptor *tex_desc(MTLPixelFormat fmt, NSUInteger w, NSUInteger h, BOOL mips,
                                      MTLTextureUsage usage)
{
	MTLTextureDescriptor *d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:fmt width:w
	                                                                             height:h
	                                                                          mipmapped:mips];
	d.storageMode = MTLStorageModePrivate;
	d.usage = usage;
	return d;
}

// Bytes a level of a texture takes in a tightly packed upload, and its rows' pitch.
static size_t level_bytes(MTLPixelFormat fmt, NSUInteger w, NSUInteger h, NSUInteger *pitch)
{
	if (fmt == MTLPixelFormatBC7_RGBAUnorm_sRGB) {
		NSUInteger bw = (w + 3) / 4, bh = (h + 3) / 4;
		*pitch = bw * 16;
		return bw * 16 * bh;
	}
	*pitch = w * 4;
	return (size_t)w * 4 * h;
}

static size_t chain_bytes(MTLPixelFormat fmt, NSUInteger size, NSUInteger levels)
{
	size_t total = 0;
	NSUInteger pitch;
	for (NSUInteger l = 0; l < levels; l++) {
		NSUInteger s = MAX(size >> l, 1);
		total += (level_bytes(fmt, s, s, &pitch) + 255) & ~(size_t)255;
	}
	return total;
}

// Every level of `t` from `src` at `off` on, each level's rows tightly packed and its start
// 256-byte aligned (a block-compressed level's copy takes a multiple of its block); returns the
// bytes taken (chain_bytes's).
static size_t upload_chain(id<MTLBlitCommandEncoder> b, id<MTLBuffer> src, size_t off, id<MTLTexture> t)
{
	size_t start = off;
	for (NSUInteger l = 0; l < t.mipmapLevelCount; l++) {
		NSUInteger w = MAX(t.width >> l, 1), h = MAX(t.height >> l, 1), pitch;
		size_t bytes = level_bytes(t.pixelFormat, w, h, &pitch);
		[b copyFromBuffer:src sourceOffset:off sourceBytesPerRow:pitch sourceBytesPerImage:bytes
		       sourceSize:MTLSizeMake(w, h, 1) toTexture:t destinationSlice:0 destinationLevel:l
		destinationOrigin:MTLOriginMake(0, 0, 0)];
		off += (bytes + 255) & ~(size_t)255;
	}
	return off - start;
}

static void list_devices(void)
{
	for (id<MTLDevice> d in MTLCopyAllDevices())
		printf("%s (unified memory %d, low power %d, removable %d)\n", d.name.UTF8String,
		       (int)d.hasUnifiedMemory, (int)d.isLowPower, (int)d.isRemovable);
}

int main(int argc, char **argv)
{
	@autoreleasepool {
		const char *want = "";
		int frames = 300, warmup = 30, draws = 4000, icb_objects = 8192, inflight = 2, quiet = 0, icb_bound = 0;
		int cicb_commands = 512;
		int W = 1920, H = 1080;
		for (int i = 1; i < argc; i++) {
			const char *a = argv[i], *v = i + 1 < argc ? argv[i + 1] : NULL;
			if (!strcmp(a, "-l")) {
				list_devices();
				return 0;
			} else if (!strcmp(a, "-q")) {
				quiet = 1;
			} else if (!strcmp(a, "-b")) {
				icb_bound = 1;
			} else if (v && !strcmp(a, "-d")) {
				want = v, i++;
			} else if (v && !strcmp(a, "-f")) {
				frames = atoi(v), i++;
			} else if (v && !strcmp(a, "-w")) {
				warmup = atoi(v), i++;
			} else if (v && !strcmp(a, "-n")) {
				draws = atoi(v), i++;
			} else if (v && !strcmp(a, "-i")) {
				icb_objects = atoi(v), i++;
			} else if (v && !strcmp(a, "-c")) {
				cicb_commands = atoi(v), i++;
			} else if (v && !strcmp(a, "-k")) {
				inflight = atoi(v), i++;
			} else if (v && !strcmp(a, "-s")) {
				if (sscanf(v, "%dx%d", &W, &H) != 2)
					die("-s takes WxH", nil);
				i++;
			} else if (v && !strcmp(a, "--off")) {
				char *list = strdup(v), *save = NULL;
				for (char *t = strtok_r(list, ",", &save); t; t = strtok_r(NULL, ",", &save)) {
					int k = 0;
					while (k < S_COUNT && strcmp(t, kShape[k]))
						k++;
					if (k == S_COUNT) {
						fprintf(stderr, "gameload: no shape %s (heap rtheap targets views icb cicb draws post blit)\n", t);
						return 2;
					}
					on[k] = false;
				}
				free(list);
				i++;
			} else if (v && !strcmp(a, "--private")) {
				char *list = strdup(v), *save = NULL;
				for (char *t = strtok_r(list, ",", &save); t; t = strtok_r(NULL, ",", &save)) {
					int k = 0;
					while (k < P_COUNT && strcmp(t, kPrivate[k]))
						k++;
					if (k == P_COUNT) {
						fprintf(stderr, "gameload: no buffer %s to make Private (frame materials args)\n", t);
						return 2;
					}
					priv[k] = true;
				}
				free(list);
				i++;
			} else {
				fprintf(stderr, "gameload: unknown argument %s (the source's header lists them)\n", a);
				return 2;
			}
		}
		if (frames < 1 || warmup < 0 || frames + warmup > MAX_FRAMES || inflight < 1 || inflight > 8 ||
		    draws < 0 || icb_objects < 1 || cicb_commands < 1 || cicb_commands > 65536 || W < 64 || H < 64)
			die("an argument is out of range", nil);
		for (id<MTLDevice> d in MTLCopyAllDevices()) {
			if (!*want || strcasestr(d.name.UTF8String, want)) {
				D = d;
				break;
			}
		}
		if (!D) {
			fprintf(stderr, "gameload: no Metal device matching '%s'; there are:\n", want);
			list_devices();
			return 2;
		}
		Q = [D newCommandQueue];
		BOOL bc = NO;
		if ([D respondsToSelector:@selector(supportsBCTextureCompression)])
			bc = D.supportsBCTextureCompression;

		// --- the library and pipelines --------------------------------------------------------
		NSError *err = nil;
		double t_lib = now_ms();
		NSString *src = [[NSString alloc] initWithBytes:gameload_metal length:gameload_metal_len
		                                       encoding:NSUTF8StringEncoding];
		id<MTLLibrary> lib = [D newLibraryWithSource:src options:[MTLCompileOptions new] error:&err];
		if (!lib)
			die("the library", err);
		t_lib = now_ms() - t_lib;

		MTLVertexDescriptor *vd = [MTLVertexDescriptor vertexDescriptor];
		vd.attributes[0].format = MTLVertexFormatFloat3;
		vd.attributes[0].offset = 0;
		vd.attributes[1].format = MTLVertexFormatFloat3;
		vd.attributes[1].offset = 12;
		vd.attributes[2].format = MTLVertexFormatFloat2;
		vd.attributes[2].offset = 24;
		vd.layouts[0].stride = sizeof(GlVertex);

		const NSUInteger samples = on[S_TARGETS] ? 4 : 1;
		const MTLPixelFormat depth_fmt = on[S_TARGETS] ? MTLPixelFormatDepth32Float : MTLPixelFormatInvalid;
		id<MTLFunction> scene_vs = [lib newFunctionWithName:@"scene_vs"];
		id<MTLRenderPipelineState> scene_pso[VARIANTS], icb_pso;
		id<MTLFunction> scene_fs0 = nil;
		for (int v = 0; v < VARIANTS; v++) {
			MTLFunctionConstantValues *cv = [MTLFunctionConstantValues new];
			[cv setConstantValue:&v type:MTLDataTypeInt atIndex:0];
			id<MTLFunction> fs = [lib newFunctionWithName:@"scene_fs" constantValues:cv error:&err];
			if (!fs)
				die("scene_fs", err);
			if (v == 0)
				scene_fs0 = fs;
			MTLRenderPipelineDescriptor *p = [MTLRenderPipelineDescriptor new];
			p.vertexFunction = scene_vs;
			p.fragmentFunction = fs;
			p.vertexDescriptor = vd;
			p.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA16Float;
			p.depthAttachmentPixelFormat = depth_fmt;
			p.rasterSampleCount = samples;
			scene_pso[v] = [D newRenderPipelineStateWithDescriptor:p error:&err];
			if (!scene_pso[v])
				die("a scene pipeline", err);
			if (v == 0) {
				p.supportIndirectCommandBuffers = YES;
				p.fragmentFunction = [lib newFunctionWithName:@"icb_fs"];
				icb_pso = [D newRenderPipelineStateWithDescriptor:p error:&err];
				if (!icb_pso)
					die("the ICB pass's pipeline", err);
			}
		}
		MTLRenderPipelineDescriptor *sp = [MTLRenderPipelineDescriptor new];
		sp.vertexFunction = [lib newFunctionWithName:@"shadow_vs"];
		sp.vertexDescriptor = vd;
		sp.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float;
		id<MTLRenderPipelineState> shadow_pso = [D newRenderPipelineStateWithDescriptor:sp error:&err];
		if (!shadow_pso)
			die("the shadow pipeline", err);
		MTLRenderPipelineDescriptor *up = [MTLRenderPipelineDescriptor new];
		up.vertexFunction = [lib newFunctionWithName:@"ui_vs"];
		up.fragmentFunction = [lib newFunctionWithName:@"ui_fs"];
		const bool views = on[S_TARGETS] && on[S_VIEWS];
		up.colorAttachments[0].pixelFormat = views ? MTLPixelFormatBGRA8Unorm : MTLPixelFormatRGBA8Unorm;
		up.colorAttachments[0].blendingEnabled = YES;
		up.colorAttachments[0].sourceRGBBlendFactor = MTLBlendFactorSourceAlpha;
		up.colorAttachments[0].destinationRGBBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
		up.colorAttachments[0].sourceAlphaBlendFactor = MTLBlendFactorOne;
		up.colorAttachments[0].destinationAlphaBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
		id<MTLRenderPipelineState> ui_pso = [D newRenderPipelineStateWithDescriptor:up error:&err];
		if (!ui_pso)
			die("the UI pipeline", err);
		MTLRenderPipelineDescriptor *cp = [MTLRenderPipelineDescriptor new];
		cp.vertexFunction = [lib newFunctionWithName:@"fullscreen_vs"];
		cp.fragmentFunction = [lib newFunctionWithName:@"composite_fs"];
		cp.colorAttachments[0].pixelFormat = views ? MTLPixelFormatRGBA8Unorm_sRGB : MTLPixelFormatRGBA8Unorm;
		id<MTLRenderPipelineState> composite_pso = [D newRenderPipelineStateWithDescriptor:cp error:&err];
		if (!composite_pso)
			die("the composite pipeline", err);
		id<MTLComputePipelineState> cps[6];
		const char *kernels[6] = {icb_bound ? "icb_encode_bound" : "icb_encode", "hist_clear", "luminance", "bloom_down",
		                          "bloom_up", "tonemap"};
		id<MTLFunction> icb_fn = nil;
		for (int k = 0; k < 6; k++) {
			id<MTLFunction> f = [lib newFunctionWithName:@(kernels[k])];
			if (k == 0)
				icb_fn = f;
			cps[k] = f ? [D newComputePipelineStateWithFunction:f error:&err] : nil;
			if (!cps[k])
				die(kernels[k], err);
		}
		id<MTLComputePipelineState> icb_encode = cps[0], hist_clear = cps[1], luminance = cps[2],
		                            bloom_down = cps[3], bloom_up = cps[4], tonemap = cps[5];
		// The histogram's threadgroups: 16 x 16, or 8 x 8 where the kernel takes fewer (its 64 bins
		// need 64 threads).
		const NSUInteger lum_tg = luminance.maxTotalThreadsPerThreadgroup >= 256 ? 16 : 8;
		// The compute ICB's encoder and the pipeline its dispatches inherit.
		id<MTLFunction> cicb_fn = [lib newFunctionWithName:icb_bound ? @"cicb_encode_bound" : @"cicb_encode"];
		id<MTLComputePipelineState> cicb_encode = cicb_fn ? [D newComputePipelineStateWithFunction:cicb_fn error:&err] : nil;
		if (!cicb_encode)
			die("the compute ICB's encode kernel", err);
		MTLComputePipelineDescriptor *cwd = [MTLComputePipelineDescriptor new];
		cwd.computeFunction = [lib newFunctionWithName:@"cicb_work"];
		cwd.supportIndirectCommandBuffers = YES;
		id<MTLComputePipelineState> cicb_work = [D newComputePipelineStateWithDescriptor:cwd options:0 reflection:nil
		                                                                           error:&err];
		if (!cicb_work)
			die("the compute ICB's pipeline", err);

		MTLDepthStencilDescriptor *dsd = [MTLDepthStencilDescriptor new];
		dsd.depthCompareFunction = MTLCompareFunctionLess;
		dsd.depthWriteEnabled = YES;
		id<MTLDepthStencilState> depth_less = [D newDepthStencilStateWithDescriptor:dsd];
		MTLSamplerDescriptor *sd = [MTLSamplerDescriptor new];
		sd.minFilter = sd.magFilter = MTLSamplerMinMagFilterLinear;
		sd.mipFilter = MTLSamplerMipFilterLinear;
		sd.sAddressMode = sd.tAddressMode = MTLSamplerAddressModeRepeat;
		id<MTLSamplerState> smp = [D newSamplerStateWithDescriptor:sd];
		sd.sAddressMode = sd.tAddressMode = MTLSamplerAddressModeClampToEdge;
		sd.mipFilter = MTLSamplerMipFilterNotMipmapped;
		id<MTLSamplerState> clamp = [D newSamplerStateWithDescriptor:sd];
		sd.compareFunction = MTLCompareFunctionLessEqual;
		id<MTLSamplerState> cmp = [D newSamplerStateWithDescriptor:sd];

		// --- the meshes, objects and materials -------------------------------------------------
		GlVertex verts[24];
		uint16_t idx[36];
		for (int f = 0; f < 6; f++) {
			int ax = f / 2;
			float s = (f & 1) ? 1 : -1;
			for (int c = 0; c < 4; c++) {
				float u = (c & 1) ? 1 : -1, v = (c & 2) ? 1 : -1, p[3];
				p[ax] = s;
				p[(ax + 1) % 3] = u * s;
				p[(ax + 2) % 3] = v;
				GlVertex *vx = &verts[f * 4 + c];
				memcpy(vx->pos, p, sizeof(p));
				vx->nrm[0] = vx->nrm[1] = vx->nrm[2] = 0;
				vx->nrm[ax] = s;
				vx->uv[0] = (c & 1);
				vx->uv[1] = (c >> 1) & 1;
			}
			uint16_t b = f * 4, q[6] = {0, 1, 2, 2, 1, 3};
			for (int k = 0; k < 6; k++)
				idx[f * 6 + k] = b + q[k];
		}
		id<MTLBuffer> mesh = private_buffer(verts, sizeof(verts));
		id<MTLBuffer> indices = private_buffer(idx, sizeof(idx));
		const int n_objects = MAX(MAX(draws, 1), icb_objects);
		GlObject *objs = calloc(n_objects, sizeof(GlObject));
		for (int i = 0; i < n_objects; i++) {
			objs[i].pos_scale = (simd_float4){frnd() * 120 - 60, frnd() * 16 - 4, -(frnd() * 140 + 4), 0.3f + frnd() * 0.7f};
			objs[i].color = (simd_float4){0.4f + frnd() * 0.6f, 0.4f + frnd() * 0.6f, 0.4f + frnd() * 0.6f, 1};
			objs[i].rot = (simd_float4){frnd() - 0.5f, frnd() - 0.5f, frnd() - 0.5f, frnd() * 2};
			objs[i].material = rnd() % MATERIALS;
		}
		id<MTLBuffer> objects = private_buffer(objs, sizeof(GlObject) * n_objects);
		simd_float4 params[MATERIALS];
		for (int m = 0; m < MATERIALS; m++)
			params[m] = (simd_float4){0.6f + frnd() * 0.4f, 0.6f + frnd() * 0.4f, 0.6f + frnd() * 0.4f, frnd()};
		id<MTLBuffer> mat_params = private_buffer(params, sizeof(params));

		// The source of every upload: random bytes, which a BC7 texture takes as blocks too.
		const size_t src_len = 8 << 20;
		uint8_t *src_bytes = malloc(src_len);
		for (size_t i = 0; i < src_len; i += 4) {
			uint32_t r = rnd() | 0xff000000u;
			memcpy(src_bytes + i, &r, 4);
		}

		// Textures made once: the materials' when the heap is off, and the details always.
		{
			MTLTextureDescriptor *d = tex_desc(MTLPixelFormatRGBA8Unorm_sRGB, 256, 256, YES, MTLTextureUsageShaderRead);
			id<MTLBuffer> s = [D newBufferWithBytes:src_bytes length:chain_bytes(d.pixelFormat, 256, d.mipmapLevelCount)
			                                options:MTLResourceStorageModeShared];
			id<MTLCommandBuffer> cb = [Q commandBuffer];
			id<MTLBlitCommandEncoder> b = [cb blitCommandEncoder];
			for (int t = 0; t < STATIC_TEX; t++) {
				statics[t] = [D newTextureWithDescriptor:d];
				upload_chain(b, s, 0, statics[t]);
			}
			[b endEncoding];
			[cb commit];
			[cb waitUntilCompleted];
		}

		// --- the heaps ------------------------------------------------------------------------
		MTLTextureDescriptor *slot_desc[SLOTS];
		NSUInteger slot_off[SLOTS], stream_off[8];
		id<MTLHeap> place = nil, transient[8] = {nil};
		size_t place_size = 0, transient_size = 0;
		MTLTextureDescriptor *bloom_desc[2 * BLOOM - 1];
		{
			// The bloom chain: BLOOM levels down from half the scene's size, then BLOOM - 1 up.
			for (int l = 0; l < BLOOM; l++) {
				bloom_desc[l] = tex_desc(MTLPixelFormatRGBA16Float, MAX(W >> (l + 1), 1), MAX(H >> (l + 1), 1), NO,
				                         MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite);
				if (l < BLOOM - 1)
					bloom_desc[BLOOM + l] = bloom_desc[l];
			}
		}
		id<MTLTexture> bloom_fixed[2 * BLOOM - 1];
		if (on[S_HEAP]) {
			NSUInteger off = 0;
			for (int s = 0; s < SLOTS; s++) {
				MTLPixelFormat f = (bc && (s & 1)) ? MTLPixelFormatBC7_RGBAUnorm_sRGB : MTLPixelFormatRGBA8Unorm_sRGB;
				slot_desc[s] = tex_desc(f, SLOT_SIZE, SLOT_SIZE, YES, MTLTextureUsageShaderRead);
				MTLSizeAndAlign sa = [D heapTextureSizeAndAlignWithDescriptor:slot_desc[s]];
				off = (off + sa.align - 1) / sa.align * sa.align;
				slot_off[s] = off;
				off += sa.size;
			}
			MTLSizeAndAlign ba = [D heapBufferSizeAndAlignWithLength:STREAM_BUF options:MTLResourceStorageModePrivate];
			for (int k = 0; k < inflight; k++) {
				off = (off + ba.align - 1) / ba.align * ba.align;
				stream_off[k] = off;
				off += ba.size;
			}
			MTLHeapDescriptor *hd = [MTLHeapDescriptor new];
			hd.type = MTLHeapTypePlacement;
			hd.storageMode = MTLStorageModePrivate;
			hd.hazardTrackingMode = MTLHazardTrackingModeUntracked;
			hd.size = off;
			place = [D newHeapWithDescriptor:hd];
			if (!place)
				die("the placement heap", nil);
			place_size = off;
			for (int s = 0; s < SLOTS; s++) {
				slot_tex[s] = [place newTextureWithDescriptor:slot_desc[s] offset:slot_off[s]];
				if (!slot_tex[s])
					die("a stream slot", nil);
			}
			// Every slot's first contents, so a slot not yet streamed samples data.
			id<MTLBuffer> s0 = [D newBufferWithBytes:src_bytes length:chain_bytes(MTLPixelFormatRGBA8Unorm_sRGB, SLOT_SIZE, slot_tex[0].mipmapLevelCount)
			                                 options:MTLResourceStorageModeShared];
			id<MTLCommandBuffer> cb = [Q commandBuffer];
			id<MTLBlitCommandEncoder> b = [cb blitCommandEncoder];
			for (int s = 0; s < SLOTS; s++)
				upload_chain(b, s0, 0, slot_tex[s]);
			[b endEncoding];
			[cb commit];
			[cb waitUntilCompleted];

			NSUInteger tsize = 0;
			for (int l = 0; l < 2 * BLOOM - 1; l++) {
				MTLSizeAndAlign sa = [D heapTextureSizeAndAlignWithDescriptor:bloom_desc[l]];
				tsize = (tsize + sa.align - 1) / sa.align * sa.align + sa.size;
			}
			hd = [MTLHeapDescriptor new];
			hd.type = MTLHeapTypeAutomatic;
			hd.storageMode = MTLStorageModePrivate;
			hd.hazardTrackingMode = MTLHazardTrackingModeUntracked;
			hd.size = tsize + (tsize >> 2);
			for (int k = 0; k < inflight; k++) {
				transient[k] = [D newHeapWithDescriptor:hd];
				if (!transient[k])
					die("a transient heap", nil);
			}
			transient_size = hd.size;
		} else {
			for (int l = 0; l < 2 * BLOOM - 1; l++)
				bloom_fixed[l] = [D newTextureWithDescriptor:bloom_desc[l]];
		}
		id<MTLFence> upload_fence = [D newFence], scene_fence = [D newFence];

		// --- targets and the rest ---------------------------------------------------------------
		MTLTextureUsage rt = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
		// The render targets, the shadow map and the tone map's output: with rtheap on, placed in a
		// placement heap of their own (tracked, so the passes need no fences between them), as a
		// port of placed resources makes them; with it off, made by the device.
		enum { T_MSAA_COLOR, T_MSAA_DEPTH, T_HDR, T_FINAL, T_UI, T_POST, T_SHADOW, T_COUNT };
		MTLTextureDescriptor *td[T_COUNT];
		td[T_MSAA_COLOR] = tex_desc(MTLPixelFormatRGBA16Float, W, H, NO, MTLTextureUsageRenderTarget);
		td[T_MSAA_COLOR].textureType = MTLTextureType2DMultisample;
		td[T_MSAA_COLOR].sampleCount = 4;
		td[T_MSAA_DEPTH] = [td[T_MSAA_COLOR] copy];
		td[T_MSAA_DEPTH].pixelFormat = MTLPixelFormatDepth32Float;
		td[T_HDR] = tex_desc(MTLPixelFormatRGBA16Float, W, H, NO, rt);
		td[T_FINAL] = tex_desc(MTLPixelFormatRGBA8Unorm, W, H, NO, rt);
		td[T_UI] = tex_desc(MTLPixelFormatRGBA8Unorm, W, H, NO, rt | MTLTextureUsagePixelFormatView);
		td[T_POST] = tex_desc(MTLPixelFormatRGBA8Unorm, W, H, NO, MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite);
		td[T_SHADOW] = tex_desc(MTLPixelFormatDepth32Float, 2048, 2048, NO, rt);
		id<MTLTexture> tx[T_COUNT];
		size_t targets_heap_size = 0;
		if (on[S_RTHEAP]) {
			NSUInteger off = 0, offs[T_COUNT];
			for (int i = 0; i < T_COUNT; i++) {
				MTLSizeAndAlign sa = [D heapTextureSizeAndAlignWithDescriptor:td[i]];
				off = (off + sa.align - 1) / sa.align * sa.align;
				offs[i] = off;
				off += sa.size;
			}
			MTLHeapDescriptor *hd = [MTLHeapDescriptor new];
			hd.type = MTLHeapTypePlacement;
			hd.storageMode = MTLStorageModePrivate;
			hd.hazardTrackingMode = MTLHazardTrackingModeTracked;
			hd.size = off;
			id<MTLHeap> th = [D newHeapWithDescriptor:hd];
			if (!th)
				die("the targets' heap", nil);
			targets_heap_size = off;
			for (int i = 0; i < T_COUNT; i++) {
				td[i].hazardTrackingMode = MTLHazardTrackingModeTracked;
				tx[i] = [th newTextureWithDescriptor:td[i] offset:offs[i]];
				if (!tx[i])
					die("a target in the targets' heap", nil);
			}
		} else {
			for (int i = 0; i < T_COUNT; i++)
				tx[i] = [D newTextureWithDescriptor:td[i]];
		}
		id<MTLTexture> msaa_color = on[S_TARGETS] ? tx[T_MSAA_COLOR] : nil, msaa_depth = on[S_TARGETS] ? tx[T_MSAA_DEPTH] : nil;
		id<MTLTexture> hdr = tx[T_HDR], final = tx[T_FINAL], ui = tx[T_UI], post_out = tx[T_POST], shadow_map = tx[T_SHADOW];
		id<MTLTexture> final_view = views ? [final newTextureViewWithPixelFormat:MTLPixelFormatRGBA8Unorm_sRGB] : final;
		id<MTLTexture> ui_view = views ? [ui newTextureViewWithPixelFormat:MTLPixelFormatBGRA8Unorm] : ui;
		id<MTLTexture> dyn = [D newTextureWithDescriptor:tex_desc(MTLPixelFormatRGBA8Unorm, BLIT_TEX, BLIT_TEX, YES, MTLTextureUsageShaderRead)];
		id<MTLBuffer> blit_dst = [D newBufferWithLength:BLIT_BUF options:MTLResourceStorageModePrivate];
		id<MTLBuffer> hist = [D newBufferWithLength:64 * 4 options:MTLResourceStorageModePrivate];
		{
			// The textures a disabled shape would have written: cleared once (ui to 0, so the
			// composite shows the scene), the shadow map to far.
			id<MTLCommandBuffer> cb = [Q commandBuffer];
			MTLRenderPassDescriptor *rp = [MTLRenderPassDescriptor renderPassDescriptor];
			rp.colorAttachments[0].texture = ui;
			rp.colorAttachments[0].loadAction = MTLLoadActionClear;
			rp.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 0);
			rp.colorAttachments[0].storeAction = MTLStoreActionStore;
			[[cb renderCommandEncoderWithDescriptor:rp] endEncoding];
			rp = [MTLRenderPassDescriptor renderPassDescriptor];
			rp.depthAttachment.texture = shadow_map;
			rp.depthAttachment.loadAction = MTLLoadActionClear;
			rp.depthAttachment.clearDepth = 1;
			rp.depthAttachment.storeAction = MTLStoreActionStore;
			[[cb renderCommandEncoderWithDescriptor:rp] endEncoding];
			id<MTLBuffer> s = [D newBufferWithBytes:src_bytes length:chain_bytes(MTLPixelFormatRGBA8Unorm, BLIT_TEX, dyn.mipmapLevelCount)
			                                options:MTLResourceStorageModeShared];
			id<MTLBlitCommandEncoder> b = [cb blitCommandEncoder];
			upload_chain(b, s, 0, dyn);
			[b endEncoding];
			[cb commit];
			[cb waitUntilCompleted];
		}

		// Per frame in flight: constants, materials, staging and read-back.
		const NSUInteger frame_stride = (sizeof(GlFrame) + 255) & ~(NSUInteger)255;
		id<MTLBuffer> frame_buf = [D newBufferWithLength:frame_stride * inflight options:MTLResourceStorageModeShared];
		id<MTLArgumentEncoder> mat_enc = [scene_fs0 newArgumentEncoderWithBufferIndex:0];
		const NSUInteger mat_stride = (mat_enc.encodedLength + mat_enc.alignment - 1) / mat_enc.alignment * mat_enc.alignment;
		id<MTLBuffer> mat_buf[8];
		size_t stage_len = 0;
		if (on[S_HEAP])
			stage_len += STREAMED * chain_bytes(MTLPixelFormatRGBA8Unorm_sRGB, SLOT_SIZE, slot_tex[0].mipmapLevelCount) + STREAM_BUF;
		if (on[S_BLIT])
			stage_len += (size_t)BLIT_TEX * BLIT_TEX * 4 + BLIT_BUF;
		stage_len = MAX(stage_len, 4096);
		id<MTLBuffer> staging[8], readback[8];
		for (int k = 0; k < inflight; k++) {
			mat_buf[k] = [D newBufferWithLength:mat_stride * MATERIALS options:MTLResourceStorageModeShared];
			staging[k] = [D newBufferWithLength:stage_len options:MTLResourceStorageModeShared];
			readback[k] = [D newBufferWithLength:READBACK * READBACK * 4 options:MTLResourceStorageModeShared];
		}
		void (^encode_materials)(id<MTLBuffer>) = ^(id<MTLBuffer> buf) {
			const bool streaming = on[S_HEAP];
			for (int m = 0; m < MATERIALS; m++) {
				[mat_enc setArgumentBuffer:buf offset:m * mat_stride];
				[mat_enc setTexture:streaming ? slot_tex[m % SLOTS] : statics[(m * 3) % STATIC_TEX] atIndex:0];
				[mat_enc setTexture:statics[m % STATIC_TEX] atIndex:1];
				[mat_enc setBuffer:mat_params offset:m * sizeof(simd_float4) atIndex:2];
			}
		};
		if (!on[S_HEAP])
			for (int k = 0; k < inflight; k++)
				encode_materials(mat_buf[k]);
		id<MTLIndirectCommandBuffer> icb = nil;
		id<MTLBuffer> icb_args = nil;
		if (on[S_ICB]) {
			MTLIndirectCommandBufferDescriptor *d = [MTLIndirectCommandBufferDescriptor new];
			d.commandTypes = MTLIndirectCommandTypeDrawIndexed;
			d.inheritBuffers = !icb_bound;
			d.inheritPipelineState = YES;
			if (icb_bound) {
				d.maxVertexBufferBindCount = 3;
				d.maxFragmentBufferBindCount = 2;
			}
			icb = [D newIndirectCommandBufferWithDescriptor:d maxCommandCount:icb_objects options:0];
			if (!icb)
				die("the indirect command buffer", nil);
			id<MTLArgumentEncoder> ae = [icb_fn newArgumentEncoderWithBufferIndex:0];
			icb_args = [D newBufferWithLength:ae.encodedLength options:MTLResourceStorageModeShared];
			[ae setArgumentBuffer:icb_args offset:0];
			[ae setIndirectCommandBuffer:icb atIndex:0];
		}
		// The compute ICB, its input (random bytes, 0 to 255 a word), its multiplier and the sums, one
		// a frame in flight, and the sum its dispatches make, worked out here.
		id<MTLIndirectCommandBuffer> cicb = nil;
		id<MTLBuffer> cicb_args = nil, cicb_in = nil, cicb_mul = nil, cicb_sums = nil;
		uint32_t cicb_want = 0;
		if (on[S_CICB]) {
			MTLIndirectCommandBufferDescriptor *d = [MTLIndirectCommandBufferDescriptor new];
			d.commandTypes = MTLIndirectCommandTypeConcurrentDispatch;
			d.inheritBuffers = !icb_bound;
			d.inheritPipelineState = YES;
			if (icb_bound)
				d.maxKernelBufferBindCount = 3;
			cicb = [D newIndirectCommandBufferWithDescriptor:d maxCommandCount:cicb_commands options:0];
			if (!cicb)
				die("the compute indirect command buffer", nil);
			id<MTLArgumentEncoder> ae = [cicb_fn newArgumentEncoderWithBufferIndex:0];
			cicb_args = [D newBufferWithLength:ae.encodedLength options:MTLResourceStorageModeShared];
			[ae setArgumentBuffer:cicb_args offset:0];
			[ae setIndirectCommandBuffer:cicb atIndex:0];
			const size_t words = CICB_LEAD + (size_t)cicb_commands * 64 + 4 * 64;
			uint32_t *in = malloc(words * 4), mul[CICB_LEAD + 1] = {0};
			for (size_t w = 0; w < words; w++)
				in[w] = rnd() & 0xff;
			mul[CICB_LEAD] = 3;
			for (int c = 0; c < cicb_commands; c++) {
				if (c % 5 == 4)
					continue;
				const uint32_t *base = in + CICB_LEAD + (icb_bound ? (size_t)c * 64 : 0);
				for (int g = 0; g < 64 * (1 + c % 4); g++)
					cicb_want += base[g] * mul[CICB_LEAD];
			}
			cicb_in = private_buffer(in, words * 4);
			cicb_mul = private_buffer(mul, sizeof(mul));
			free(in);
			cicb_sums = [D newBufferWithLength:(NSUInteger)CICB_SLOT * inflight options:MTLResourceStorageModeShared];
			memset(cicb_sums.contents, 0, (size_t)CICB_SLOT * inflight);
		}
		// The buffers the GPU reads: the Shared ones the CPU writes, or their Private copies
		// (--private). A copy made each frame is written by the frame's first blit (below).
		id<MTLBuffer> frame_gpu = priv[P_FRAME] ? [D newBufferWithLength:frame_buf.length options:MTLResourceStorageModePrivate]
		                                        : frame_buf;
		id<MTLBuffer> mat_gpu[8];
		for (int k = 0; k < inflight; k++)
			mat_gpu[k] = !priv[P_MATERIALS] ? mat_buf[k]
			             : on[S_HEAP]       ? [D newBufferWithLength:mat_buf[k].length options:MTLResourceStorageModePrivate]
			                                : private_copy(mat_buf[k]);
		id<MTLBuffer> icb_args_gpu = icb_args && priv[P_ARGS] ? private_copy(icb_args) : icb_args;
		id<MTLBuffer> cicb_args_gpu = cicb_args && priv[P_ARGS] ? private_copy(cicb_args) : cicb_args;
		const bool frame_copies = priv[P_FRAME] || (priv[P_MATERIALS] && on[S_HEAP]);
		// The sum of the frame that last had slot k, checked, then cleared for the next.
		void (^cicb_check)(int, int) = ^(int k, int frame) {
			uint32_t *sum = (uint32_t *)((uint8_t *)cicb_sums.contents + (size_t)k * CICB_SLOT + 64);
			if (*sum != cicb_want && cicb_bad++ == 0)
				snprintf(cicb_first_bad, sizeof(cicb_first_bad), "frame %d got %u want %u", frame, *sum, cicb_want);
			cicb_checks++;
			*sum = 0;
		};
		__unsafe_unretained id<MTLResource> statics_list[STATIC_TEX];
		for (int i = 0; i < STATIC_TEX; i++)
			statics_list[i] = statics[i];
		simd_float4 ui_rects[64];
		for (int r = 0; r < 64; r++) {
			float x = -0.95f + (r % 8) * 0.12f, y = 0.9f - (r / 8) * 0.06f;
			ui_rects[r] = (simd_float4){x, y, x + 0.1f, y - 0.04f};
		}

		// --- the frames -------------------------------------------------------------------------
		const int total = warmup + frames;
		const int shadow_draws = draws / 4;
		double *f_cpu = calloc(total, sizeof(double)), *f_wait = calloc(total, sizeof(double)),
		       *f_wall = calloc(total, sizeof(double)), *f_start = calloc(total, sizeof(double));
		double *f_gpu = calloc(total, sizeof(double)), *f_a = calloc(total, sizeof(double)),
		       *f_b = calloc(total, sizeof(double)), *a_start = calloc(total, sizeof(double));
		double *shape_cpu[S_COUNT];
		for (int s = 0; s < S_COUNT; s++)
			shape_cpu[s] = calloc(total, sizeof(double));
		dispatch_semaphore_t slots = dispatch_semaphore_create(inflight);
		uint64_t heap_objects = 0, heap_failures = 0, readback_sum = 0;
		const NSUInteger slot_chain = slot_tex[0] ? slot_tex[0].mipmapLevelCount : 0;
		int passes = 0, dispatches = 0;
		const float aspect = (float)W / H;
		const simd_float4x4 proj = perspective(1.0f, aspect, 0.5f, 200.0f);
		const simd_float4x4 light = simd_mul(ortho(140, 140, 1, 300),
		                                     look_at((simd_float3){60, 120, -10}, (simd_float3){0, 0, -70}, (simd_float3){0, 1, 0}));
		const simd_float3 light_dir = simd_normalize((simd_float3){-60, -120, -60});
		double window_start = 0, window_end = 0;
		id<MTLCommandBuffer> last = nil;

		printf("GAMELOAD config device \"%s\" unified %d bc %d shapes", D.name.UTF8String, (int)D.hasUnifiedMemory, (int)bc);
		for (int s = 0; s < S_COUNT; s++)
			printf(" %s=%d", kShape[s], (int)on[s]);
		printf(" icb_bound %d private", icb_bound);
		for (int p = 0; p < P_COUNT; p++)
			printf(" %s=%d", kPrivate[p], (int)priv[p]);
		printf(" size %dx%d draws %d shadow_draws %d icb_objects %d cicb_commands %d frames %d warmup %d"
		       " inflight %d placement_heap %zu transient_heap %zu targets_heap %zu library_ms %.1f\n",
		       W, H, on[S_DRAWS] ? draws : 0, on[S_DRAWS] ? shadow_draws : 0, on[S_ICB] ? icb_objects : 0,
		       on[S_CICB] ? cicb_commands : 0, frames, warmup, inflight, place_size, transient_size, targets_heap_size, t_lib);
		fflush(stdout);

		for (int f = 0; f < total; f++) {
			@autoreleasepool {
				if (f == warmup) {
					window_start = epoch_s();
					printf("GAMELOAD window start %.6f\n", window_start);
					fflush(stdout);
				}
				double t0 = now_ms();
				dispatch_semaphore_wait(slots, DISPATCH_TIME_FOREVER);
				double t1 = now_ms(), ts;
				f_wait[f] = t1 - t0;
				f_start[f] = t1;
				const int k = f % inflight;
				int fp = 0, fd = 0;
				if (on[S_CICB] && f >= inflight)
					cicb_check(k, f - inflight);

				// The frame's readback from FRAMES_IN_FLIGHT frames ago, which the CPU reads.
				if (on[S_BLIT] && f >= inflight) {
					const uint32_t *px = readback[k].contents;
					for (int i = 0; i < READBACK * READBACK; i += 257)
						readback_sum += px[i];
				}
				GlFrame *fc = (GlFrame *)((uint8_t *)frame_buf.contents + k * frame_stride);
				float t = f / 60.0f;
				simd_float3 eye = {sinf(t * 0.2f) * 10, 6, 6};
				fc->view_proj = simd_mul(proj, look_at(eye, (simd_float3){0, 0, -60}, (simd_float3){0, 1, 0}));
				fc->light_view_proj = light;
				fc->light_dir = (simd_float4){light_dir.x, light_dir.y, light_dir.z, 0};
				fc->time = t;
				fc->objects = n_objects;
				fc->icb_objects = icb_objects;
				fc->cicb_commands = cicb_commands;
				const NSUInteger fo = k * frame_stride;

				// The uploads: the CPU writes this frame's data into its staging buffer.
				ts = now_ms();
				uint8_t *stage = staging[k].contents;
				size_t so = 0, src_at = (size_t)f * 7919 * 4096 % (src_len / 2);
				size_t heap_stage = 0, blit_stage = 0;
				if (on[S_HEAP]) {
					heap_stage = STREAMED * chain_bytes(MTLPixelFormatRGBA8Unorm_sRGB, SLOT_SIZE, slot_chain);
					memcpy(stage, src_bytes + src_at, heap_stage);
					memcpy(stage + heap_stage, verts, sizeof(verts));
					memcpy(stage + heap_stage + sizeof(verts), src_bytes + src_at, STREAM_BUF - sizeof(verts));
					heap_stage += STREAM_BUF;
					shape_cpu[S_HEAP][f] += now_ms() - ts;
				}
				ts = now_ms();
				if (on[S_BLIT]) {
					blit_stage = (size_t)BLIT_TEX * BLIT_TEX * 4 + BLIT_BUF;
					memcpy(stage + heap_stage, src_bytes + src_at + 4096, blit_stage);
					shape_cpu[S_BLIT][f] += now_ms() - ts;
				}

				id<MTLCommandBuffer> a = [Q commandBuffer], b;
				double *ga = &f_a[f], *gas = &a_start[f];
				[a addCompletedHandler:^(id<MTLCommandBuffer> c) {
					*ga = (c.GPUEndTime - c.GPUStartTime) * 1e3;
					*gas = c.GPUStartTime * 1e3;
					if (c.status == MTLCommandBufferStatusError && atomic_fetch_add(&cb_errors, 1) == 0)
						snprintf(first_error, sizeof(first_error), "%s", c.error.description.UTF8String);
				}];

				// 1. The upload blit: the stream slots made again at their offsets, their levels and
				// the vertex buffer uploaded; the blit stream's texture, buffer and mipmaps; and the
				// frame's copies into its Private buffers (--private).
				id<MTLBuffer> stream_buf = nil;
				if (on[S_HEAP] || on[S_BLIT] || frame_copies) {
					ts = now_ms();
					id<MTLBlitCommandEncoder> be = [a blitCommandEncoder];
					if (on[S_HEAP]) {
						[be waitForFence:scene_fence];
						for (int i = 0; i < STREAMED; i++) {
							int s = (f * STREAMED + i) % SLOTS;
							slot_tex[s] = [place newTextureWithDescriptor:slot_desc[s] offset:slot_off[s]];
							if (!slot_tex[s])
								die("a stream slot made again", nil);
							so += upload_chain(be, staging[k], so, slot_tex[s]);
							heap_objects++;
						}
						so = heap_stage - STREAM_BUF;
						stream_buf = [place newBufferWithLength:STREAM_BUF options:MTLResourceStorageModePrivate offset:stream_off[k]];
						if (!stream_buf)
							die("the streamed vertex buffer", nil);
						heap_objects++;
						[be copyFromBuffer:staging[k] sourceOffset:so toBuffer:stream_buf destinationOffset:0 size:STREAM_BUF];
						so += STREAM_BUF;
						shape_cpu[S_HEAP][f] += now_ms() - ts;
					}
					ts = now_ms();
					if (on[S_BLIT]) {
						[be copyFromBuffer:staging[k] sourceOffset:so sourceBytesPerRow:BLIT_TEX * 4
						    sourceBytesPerImage:BLIT_TEX * BLIT_TEX * 4 sourceSize:MTLSizeMake(BLIT_TEX, BLIT_TEX, 1)
						              toTexture:dyn destinationSlice:0 destinationLevel:0 destinationOrigin:MTLOriginMake(0, 0, 0)];
						so += (size_t)BLIT_TEX * BLIT_TEX * 4;
						[be copyFromBuffer:staging[k] sourceOffset:so toBuffer:blit_dst destinationOffset:0 size:BLIT_BUF];
						[be generateMipmapsForTexture:dyn];
					}
					// The CPU writes this frame's constants and materials before the commit, below.
					if (priv[P_FRAME])
						[be copyFromBuffer:frame_buf sourceOffset:fo toBuffer:frame_gpu destinationOffset:fo size:frame_stride];
					if (priv[P_MATERIALS] && on[S_HEAP])
						[be copyFromBuffer:mat_buf[k] sourceOffset:0 toBuffer:mat_gpu[k] destinationOffset:0
						              size:mat_buf[k].length];
					if (on[S_HEAP])
						[be updateFence:upload_fence];
					[be endEncoding];
					fp++;
					shape_cpu[on[S_BLIT] ? S_BLIT : S_HEAP][f] += now_ms() - ts;
				}
				if (on[S_HEAP]) {
					ts = now_ms();
					encode_materials(mat_buf[k]);
					shape_cpu[S_HEAP][f] += now_ms() - ts;
				}

				// 2. The shadow pass.
				ts = now_ms();
				id<MTLBuffer> mesh_buf = on[S_HEAP] ? stream_buf : mesh;
				if (on[S_DRAWS]) {
					MTLRenderPassDescriptor *rp = [MTLRenderPassDescriptor renderPassDescriptor];
					rp.depthAttachment.texture = shadow_map;
					rp.depthAttachment.loadAction = MTLLoadActionClear;
					rp.depthAttachment.clearDepth = 1;
					rp.depthAttachment.storeAction = MTLStoreActionStore;
					id<MTLRenderCommandEncoder> re = [a renderCommandEncoderWithDescriptor:rp];
					if (on[S_HEAP])
						[re waitForFence:upload_fence beforeStages:MTLRenderStageVertex];
					[re setRenderPipelineState:shadow_pso];
					[re setDepthStencilState:depth_less];
					[re setCullMode:MTLCullModeBack];
					[re setVertexBuffer:mesh_buf offset:0 atIndex:0];
					[re setVertexBuffer:frame_gpu offset:fo atIndex:1];
					[re setVertexBuffer:objects offset:0 atIndex:2];
					for (int i = 0; i < shadow_draws; i++) {
						[re setVertexBufferOffset:(NSUInteger)i * sizeof(GlObject) atIndex:2];
						[re drawIndexedPrimitives:MTLPrimitiveTypeTriangle indexCount:36 indexType:MTLIndexTypeUInt16
						              indexBuffer:indices indexBufferOffset:0];
					}
					[re endEncoding];
					fp++;
				}
				shape_cpu[S_DRAWS][f] += now_ms() - ts;

				// 3. The GPU-driven pass's encode.
				if (on[S_ICB]) {
					ts = now_ms();
					id<MTLComputeCommandEncoder> ce = [a computeCommandEncoder];
					[ce setComputePipelineState:icb_encode];
					[ce setBuffer:icb_args_gpu offset:0 atIndex:0];
					[ce setBuffer:objects offset:0 atIndex:1];
					[ce setBuffer:frame_gpu offset:fo atIndex:2];
					[ce setBuffer:indices offset:0 atIndex:3];
					if (icb_bound) {
						[ce setBuffer:on[S_HEAP] ? stream_buf : mesh offset:0 atIndex:4];
						[ce setBuffer:mat_gpu[k] offset:0 atIndex:5];
					}
					[ce useResource:icb usage:MTLResourceUsageWrite];
					NSUInteger tg = MIN(icb_encode.maxTotalThreadsPerThreadgroup, 64);
					[ce dispatchThreadgroups:MTLSizeMake((icb_objects + tg - 1) / tg, 1, 1)
					   threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
					[ce endEncoding];
					fd++;
					shape_cpu[S_ICB][f] += now_ms() - ts;
				}

				// 3b. The GPU-driven compute pass: the compute ICB's encode, then its execution in a
				// compute encoder of its own, its buffers bound at non-zero offsets.
				if (on[S_CICB]) {
					ts = now_ms();
					const NSUInteger sum_off = (NSUInteger)k * CICB_SLOT + 64;
					id<MTLComputeCommandEncoder> ce = [a computeCommandEncoder];
					[ce setComputePipelineState:cicb_encode];
					[ce setBuffer:cicb_args_gpu offset:0 atIndex:0];
					[ce setBuffer:frame_gpu offset:fo atIndex:1];
					if (icb_bound) {
						[ce setBuffer:cicb_in offset:CICB_LEAD * 4 atIndex:2];
						[ce setBuffer:cicb_sums offset:sum_off atIndex:3];
						[ce setBuffer:cicb_mul offset:CICB_LEAD * 4 atIndex:4];
					}
					[ce useResource:cicb usage:MTLResourceUsageWrite];
					NSUInteger tg = MIN(cicb_encode.maxTotalThreadsPerThreadgroup, 64);
					[ce dispatchThreadgroups:MTLSizeMake((cicb_commands + tg - 1) / tg, 1, 1)
					   threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
					[ce endEncoding];
					ce = [a computeCommandEncoder];
					[ce setComputePipelineState:cicb_work];
					if (icb_bound) {
						[ce useResource:cicb_in usage:MTLResourceUsageRead];
						[ce useResource:cicb_mul usage:MTLResourceUsageRead];
						[ce useResource:cicb_sums usage:MTLResourceUsageRead | MTLResourceUsageWrite];
					} else {
						[ce setBuffer:cicb_in offset:CICB_LEAD * 4 atIndex:0];
						[ce setBuffer:cicb_sums offset:sum_off atIndex:1];
						[ce setBuffer:cicb_mul offset:CICB_LEAD * 4 atIndex:2];
					}
					[ce useResource:cicb usage:MTLResourceUsageRead];
					[ce executeCommandsInBuffer:cicb withRange:NSMakeRange(0, cicb_commands)];
					[ce endEncoding];
					fd += 2;
					shape_cpu[S_CICB][f] += now_ms() - ts;
				}

				// 4. The scene pass: CPU draws, then the ICB's.
				ts = now_ms();
				MTLRenderPassDescriptor *rp = [MTLRenderPassDescriptor renderPassDescriptor];
				if (on[S_TARGETS]) {
					rp.colorAttachments[0].texture = msaa_color;
					rp.colorAttachments[0].resolveTexture = hdr;
					rp.colorAttachments[0].storeAction = MTLStoreActionMultisampleResolve;
					rp.depthAttachment.texture = msaa_depth;
					rp.depthAttachment.loadAction = MTLLoadActionClear;
					rp.depthAttachment.clearDepth = 1;
					rp.depthAttachment.storeAction = MTLStoreActionDontCare;
				} else {
					rp.colorAttachments[0].texture = hdr;
					rp.colorAttachments[0].storeAction = MTLStoreActionStore;
				}
				rp.colorAttachments[0].loadAction = MTLLoadActionClear;
				rp.colorAttachments[0].clearColor = MTLClearColorMake(0.05, 0.07, 0.1, 1);
				id<MTLRenderCommandEncoder> re = [a renderCommandEncoderWithDescriptor:rp];
				if (on[S_HEAP]) {
					[re waitForFence:upload_fence beforeStages:MTLRenderStageVertex];
					[re useHeap:place stages:MTLRenderStageVertex | MTLRenderStageFragment];
				}
				[re useResources:statics_list count:STATIC_TEX usage:MTLResourceUsageRead stages:MTLRenderStageFragment];
				[re useResource:mat_params usage:MTLResourceUsageRead stages:MTLRenderStageFragment];
				if (on[S_TARGETS])
					[re setDepthStencilState:depth_less];
				[re setCullMode:MTLCullModeBack];
				[re setVertexBuffer:mesh_buf offset:0 atIndex:0];
				[re setVertexBuffer:frame_gpu offset:fo atIndex:1];
				[re setVertexBuffer:objects offset:0 atIndex:2];
				[re setFragmentBuffer:mat_gpu[k] offset:0 atIndex:0];
				[re setFragmentBuffer:frame_gpu offset:fo atIndex:1];
				[re setFragmentTexture:shadow_map atIndex:0];
				[re setFragmentSamplerState:smp atIndex:0];
				[re setFragmentSamplerState:cmp atIndex:1];
				double t_scene = now_ms() - ts;
				if (on[S_DRAWS]) {
					ts = now_ms();
					const int per_state = MAX(draws / 8, 1);
					for (int i = 0; i < draws; i++) {
						if (i % per_state == 0)
							[re setRenderPipelineState:scene_pso[(i / per_state) % VARIANTS]];
						[re setVertexBufferOffset:(NSUInteger)i * sizeof(GlObject) atIndex:2];
						[re setFragmentBufferOffset:(objs[i].material) * mat_stride atIndex:0];
						[re drawIndexedPrimitives:MTLPrimitiveTypeTriangle indexCount:36 indexType:MTLIndexTypeUInt16
						              indexBuffer:indices indexBufferOffset:0];
					}
					shape_cpu[S_DRAWS][f] += now_ms() - ts;
				}
				if (on[S_ICB]) {
					ts = now_ms();
					[re setRenderPipelineState:icb_pso];
					[re setVertexBufferOffset:0 atIndex:2];
					[re setFragmentBufferOffset:0 atIndex:0];
					[re useResource:indices usage:MTLResourceUsageRead stages:MTLRenderStageVertex];
					[re useResource:objects usage:MTLResourceUsageRead stages:MTLRenderStageVertex];
					if (icb_bound) {
						[re useResource:mesh_buf usage:MTLResourceUsageRead stages:MTLRenderStageVertex];
						[re useResource:frame_gpu usage:MTLResourceUsageRead stages:MTLRenderStageVertex | MTLRenderStageFragment];
						[re useResource:mat_gpu[k] usage:MTLResourceUsageRead stages:MTLRenderStageFragment];
					}
					[re executeCommandsInBuffer:icb withRange:NSMakeRange(0, icb_objects)];
					shape_cpu[S_ICB][f] += now_ms() - ts;
				}
				ts = now_ms();
				if (on[S_HEAP])
					[re updateFence:scene_fence afterStages:MTLRenderStageFragment];
				[re endEncoding];
				fp++;
				[a commit];
				t_scene += now_ms() - ts;
				// The scene pass's own setup and end count with the targets.
				shape_cpu[S_TARGETS][f] += t_scene;

				// 5. Post-processing.
				b = [Q commandBuffer];
				id<MTLTexture> composite_src = hdr;
				if (on[S_POST]) {
					ts = now_ms();
					id<MTLHeap> th = on[S_HEAP] ? transient[k] : nil;
					id<MTLTexture> bl[2 * BLOOM - 1];
					// Down chain first; up chain textures are made as the down levels they replace go.
					for (int l = 0; l < BLOOM; l++) {
						bl[l] = th ? [th newTextureWithDescriptor:bloom_desc[l]] : bloom_fixed[l];
						if (!bl[l]) {
							heap_failures++;
							bl[l] = [D newTextureWithDescriptor:bloom_desc[l]];
						} else if (th)
							heap_objects++;
					}
					id<MTLComputeCommandEncoder> ce = [b computeCommandEncoder];
					[ce setComputePipelineState:hist_clear];
					[ce setBuffer:hist offset:0 atIndex:0];
					[ce dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
					[ce memoryBarrierWithScope:MTLBarrierScopeBuffers];
					[ce setComputePipelineState:luminance];
					[ce setTexture:hdr atIndex:0];
					[ce dispatchThreadgroups:MTLSizeMake((W + lum_tg - 1) / lum_tg, (H + lum_tg - 1) / lum_tg, 1)
					   threadsPerThreadgroup:MTLSizeMake(lum_tg, lum_tg, 1)];
					MTLSize g16 = MTLSizeMake(8, 8, 1);
					float threshold = 1.0f;
					[ce setComputePipelineState:bloom_down];
					[ce setSamplerState:clamp atIndex:0];
					[ce setBytes:&threshold length:sizeof(threshold) atIndex:0];
					fd += 2;
					for (int l = 0; l < BLOOM; l++) {
						[ce setTexture:l ? bl[l - 1] : hdr atIndex:0];
						[ce setTexture:bl[l] atIndex:1];
						[ce dispatchThreadgroups:MTLSizeMake((bl[l].width + 7) / 8, (bl[l].height + 7) / 8, 1)
						   threadsPerThreadgroup:g16];
						[ce memoryBarrierWithScope:MTLBarrierScopeTextures];
						fd++;
						if (l == 0) {
							float zero = 0;
							[ce setBytes:&zero length:sizeof(zero) atIndex:0];
						}
					}
					[ce setComputePipelineState:bloom_up];
					id<MTLTexture> lo = bl[BLOOM - 1];
					for (int l = BLOOM - 2; l >= 0; l--) {
						// The level below and the one about to be read become aliasable once read.
						id<MTLTexture> out = th ? [th newTextureWithDescriptor:bloom_desc[BLOOM + l]] : bloom_fixed[BLOOM + l];
						if (!out) {
							heap_failures++;
							out = [D newTextureWithDescriptor:bloom_desc[BLOOM + l]];
						} else if (th)
							heap_objects++;
						[ce setTexture:lo atIndex:0];
						[ce setTexture:bl[l] atIndex:1];
						[ce setTexture:out atIndex:2];
						[ce dispatchThreadgroups:MTLSizeMake((out.width + 7) / 8, (out.height + 7) / 8, 1)
						   threadsPerThreadgroup:g16];
						[ce memoryBarrierWithScope:MTLBarrierScopeTextures];
						fd++;
						if (th) {
							[lo makeAliasable];
							[bl[l] makeAliasable];
						}
						lo = out;
					}
					[ce setComputePipelineState:tonemap];
					[ce setTexture:hdr atIndex:0];
					[ce setTexture:lo atIndex:1];
					[ce setTexture:post_out atIndex:2];
					[ce setBuffer:hist offset:0 atIndex:0];
					[ce dispatchThreadgroups:MTLSizeMake((W + 7) / 8, (H + 7) / 8, 1) threadsPerThreadgroup:g16];
					fd++;
					[ce endEncoding];
					if (th)
						[lo makeAliasable];
					composite_src = post_out;
					shape_cpu[S_POST][f] += now_ms() - ts;
				}

				// 6. The UI pass, into the BGRA view with views on.
				ts = now_ms();
				if (on[S_TARGETS]) {
					MTLRenderPassDescriptor *up = [MTLRenderPassDescriptor renderPassDescriptor];
					up.colorAttachments[0].texture = ui_view;
					up.colorAttachments[0].loadAction = MTLLoadActionClear;
					up.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 0);
					up.colorAttachments[0].storeAction = MTLStoreActionStore;
					id<MTLRenderCommandEncoder> ue = [b renderCommandEncoderWithDescriptor:up];
					[ue setRenderPipelineState:ui_pso];
					[ue setVertexBytes:ui_rects length:sizeof(ui_rects) atIndex:0];
					[ue drawPrimitives:MTLPrimitiveTypeTriangleStrip vertexStart:0 vertexCount:4 instanceCount:64];
					[ue endEncoding];
					fp++;
				}

				// 7. The composite, into the sRGB view with views on.
				MTLRenderPassDescriptor *fpd = [MTLRenderPassDescriptor renderPassDescriptor];
				fpd.colorAttachments[0].texture = final_view;
				fpd.colorAttachments[0].loadAction = MTLLoadActionDontCare;
				fpd.colorAttachments[0].storeAction = MTLStoreActionStore;
				id<MTLRenderCommandEncoder> fe = [b renderCommandEncoderWithDescriptor:fpd];
				[fe setRenderPipelineState:composite_pso];
				[fe setFragmentTexture:composite_src atIndex:0];
				[fe setFragmentTexture:ui atIndex:1];
				[fe setFragmentTexture:dyn atIndex:2];
				[fe setFragmentSamplerState:smp atIndex:0];
				[fe drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
				[fe endEncoding];
				fp++;
				shape_cpu[S_TARGETS][f] += now_ms() - ts;

				// 8. The read-back.
				if (on[S_BLIT]) {
					ts = now_ms();
					id<MTLBlitCommandEncoder> be = [b blitCommandEncoder];
					[be copyFromTexture:final sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(W / 2 - READBACK / 2, H / 2 - READBACK / 2, 0)
					         sourceSize:MTLSizeMake(READBACK, READBACK, 1) toBuffer:readback[k] destinationOffset:0
					destinationBytesPerRow:READBACK * 4 destinationBytesPerImage:READBACK * READBACK * 4];
					[be endEncoding];
					fp++;
					shape_cpu[S_BLIT][f] += now_ms() - ts;
				}
				double *gg = &f_gpu[f], *gb = &f_b[f];
				[b addCompletedHandler:^(id<MTLCommandBuffer> c) {
					*gb = (c.GPUEndTime - c.GPUStartTime) * 1e3;
					*gg = c.GPUEndTime * 1e3;
					if (c.status == MTLCommandBufferStatusError && atomic_fetch_add(&cb_errors, 1) == 0)
						snprintf(first_error, sizeof(first_error), "%s", c.error.description.UTF8String);
					dispatch_semaphore_signal(slots);
				}];
				[b commit];
				last = b;
				double t2 = now_ms();
				f_cpu[f] = t2 - t1;
				passes = fp;
				dispatches = fd;
			}
		}
		[last waitUntilCompleted];
		// Every frame's completion handler has run once all the slots are back.
		for (int k = 0; k < inflight; k++)
			dispatch_semaphore_wait(slots, DISPATCH_TIME_FOREVER);
		// Back to its first value: libdispatch traps on a semaphore freed below it.
		for (int k = 0; k < inflight; k++)
			dispatch_semaphore_signal(slots);
		// The last frames' sums, whose slots no frame took back.
		if (on[S_CICB])
			for (int f = MAX(total - inflight, 0); f < total; f++)
				cicb_check(f % inflight, f);
		window_end = epoch_s();
		printf("GAMELOAD window end %.6f\n", window_end);
		fflush(stdout);

		// --- the numbers ------------------------------------------------------------------------
		for (int f = 0; f < total; f++) {
			f_wall[f] = f ? f_start[f] - f_start[f - 1] : 0;
			f_gpu[f] = f_gpu[f] - a_start[f];
		}
		const double *m_cpu = f_cpu + warmup, *m_wait = f_wait + warmup, *m_wall = f_wall + warmup,
		             *m_gpu = f_gpu + warmup, *m_a = f_a + warmup, *m_b = f_b + warmup;
		if (!quiet)
			for (int f = 0; f < frames; f++)
				printf("F %d cpu %.3f wait %.3f wall %.3f gpu %.3f a %.3f b %.3f\n", f, m_cpu[f], m_wait[f], m_wall[f],
				       m_gpu[f], m_a[f], m_b[f]);
		for (int s = 0; s < S_COUNT; s++) {
			double p50, p90, mean;
			stats(shape_cpu[s] + warmup, frames, &p50, &p90, &mean);
			printf("GAMELOAD shape %s on %d cpu_ms p50 %.3f mean %.3f\n", kShape[s], (int)on[s], p50, mean);
		}
		double c50, c90, cm, w50, w90, wm, l50, l90, lm, g50, g90, gm, a50, a90, am, b50, b90, bm;
		stats(m_cpu, frames, &c50, &c90, &cm);
		stats(m_wait, frames, &w50, &w90, &wm);
		stats(m_wall, frames, &l50, &l90, &lm);
		stats(m_gpu, frames, &g50, &g90, &gm);
		stats(m_a, frames, &a50, &a90, &am);
		stats(m_b, frames, &b50, &b90, &bm);
		const int frame_draws = (on[S_DRAWS] ? draws + shadow_draws : 0) + on[S_TARGETS] + 1;
		printf("GAMELOAD summary frames %d window_s %.3f fps %.2f"
		       " cpu_ms p50 %.3f p90 %.3f mean %.3f wait_ms p50 %.3f p90 %.3f mean %.3f"
		       " wall_ms p50 %.3f p90 %.3f mean %.3f gpu_ms p50 %.3f p90 %.3f mean %.3f"
		       " gpu_a_ms p50 %.3f mean %.3f gpu_b_ms p50 %.3f mean %.3f"
		       " cpu_draws %d icb_objects %d passes %d dispatches %d command_buffers 2"
		       " heap_objects_per_frame %.2f heap_failures %llu cb_errors %d readback %llu"
		       " cicb_commands %d cicb_checks %llu cicb_bad %llu\n",
		       frames, window_end - window_start, frames / (window_end - window_start), c50, c90, cm, w50, w90, wm,
		       l50, l90, lm, g50, g90, gm, a50, am, b50, bm, frame_draws, on[S_ICB] ? icb_objects : 0, passes,
		       dispatches, (double)heap_objects / total, (unsigned long long)heap_failures, atomic_load(&cb_errors),
		       (unsigned long long)readback_sum, on[S_CICB] ? cicb_commands : 0, (unsigned long long)cicb_checks,
		       (unsigned long long)cicb_bad);
		if (atomic_load(&cb_errors))
			printf("GAMELOAD first command-buffer error: %s\n", first_error);
		if (cicb_bad)
			printf("GAMELOAD cicb wrong sum: %llu of %llu frames, the first %s\n", (unsigned long long)cicb_bad,
			       (unsigned long long)cicb_checks, cicb_first_bad);
		fflush(stdout);
		return atomic_load(&cb_errors) || cicb_bad ? 1 : 0;
	}
}
