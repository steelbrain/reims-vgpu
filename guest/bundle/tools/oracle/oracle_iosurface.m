// oracle_iosurface: when a texture over an IOSurface sees the CPU's writes to the surface, and
// when the surface's pages see the GPU's writes to the texture, with and without
// IOSurfaceLock/Unlock, in one process and across two.
//   oracle_iosurface DEVICE [SCENARIO [W H [USAGE [STORAGE]]]]
//     DEVICE    a substring of the Metal device's name (AMD, Intel, Apple)
//     SCENARIO  single (default), cross, timing, or a script file of steps (below)
//     W H       the surface's size, 1024 x 1024 by default
//     USAGE     rt (ShaderRead|ShaderWrite|RenderTarget, default) or ro (ShaderRead)
//     STORAGE   managed (default), shared or private: the texture descriptor's storage mode
// The surface is BGRA8 ('BGRA'). Each value is a letter L standing for the word
// 0x01010101 * L, written to every pixel (or the top half); a read prints the letter every pixel
// holds, "TOP|BOTTOM" when the halves differ, or a histogram. Each step prints one line:
//   who step arg | result | seed after | ms
// Steps (who is A, this process, or B, a second process of this binary, started by the
// cross scenario):
//   cw L / cwh L   CPU write without a lock (whole surface / top half)
//   lw L / lwh L   lock read-write, CPU write, unlock
//   lrw / lro      lock read-write / read-only and unlock, touching nothing
//   cr             CPU read without a lock
//   lr / lrwr      lock read-only / read-write, CPU read, unlock
//   gr             GPU read of the process's texture (a blit into a Shared buffer)
//   grn            GPU read through a texture made for the read and let go after
//   gw L / gwh L   GPU write through the process's texture (a blit from a Shared buffer), waited
//   gc L           GPU write by a render pass that clears the texture to L, waited
//   nt             make the process's texture again
//   cb             a command buffer that does not name the texture (a fill of another buffer)
//   sl MS          sleep
// Build: see README.md.
#import "oracle_common.h"
#import <IOSurface/IOSurface.h>
#include <spawn.h>
#include <unistd.h>

extern char **environ;

typedef struct {
	id<MTLDevice> dev;
	id<MTLCommandQueue> q;
	IOSurfaceRef s;
	id<MTLTexture> t;
	id<MTLBuffer> rb;    // readback, Shared
	id<MTLBuffer> src;   // GPU-write source, Shared
	id<MTLBuffer> other; // a buffer no step reads
	int w, h;
	size_t bpr;
	MTLTextureUsage usage;
	MTLStorageMode storage;
} Ctx;

static uint32_t word(char l)
{
	return l == '0' ? 0 : 0x01010101u * (uint8_t)l;
}

static void label(uint32_t v, char *out, size_t n)
{
	uint8_t b = v & 0xff;
	if (v == 0)
		snprintf(out, n, "0");
	else if (v == 0x01010101u * b && b >= 'A' && b <= 'Z')
		snprintf(out, n, "%c", b);
	else
		snprintf(out, n, "%08x", v);
}

// What rows [y0, y1) hold, `stride` bytes apart, row length w: one label, or a histogram.
static void summarize_rows(const uint8_t *base, size_t stride, int w, int y0, int y1, char *out,
                           size_t n)
{
	uint32_t vals[4] = {0};
	size_t counts[4] = {0}, other = 0, total = 0;
	int nv = 0;
	for (int y = y0; y < y1; y++) {
		const uint32_t *row = (const uint32_t *)(base + (size_t)y * stride);
		for (int x = 0; x < w; x++) {
			uint32_t v = row[x];
			total++;
			int k;
			for (k = 0; k < nv; k++)
				if (vals[k] == v)
					break;
			if (k < nv)
				counts[k]++;
			else if (nv < 4) {
				vals[nv] = v;
				counts[nv++] = 1;
			} else
				other++;
		}
	}
	if (nv == 1) {
		label(vals[0], out, n);
		return;
	}
	size_t o = 0;
	out[0] = 0;
	for (int k = 0; k < nv; k++) {
		char l[16];
		label(vals[k], l, sizeof l);
		o += snprintf(out + o, n - o, "%s%s:%.1f%%", k ? "," : "", l, 100.0 * counts[k] / total);
	}
	if (other)
		snprintf(out + o, n - o, ",other:%.1f%%", 100.0 * other / total);
}

static void summarize(const uint8_t *base, size_t stride, int w, int h, char *out, size_t n)
{
	char top[96], bottom[96];
	summarize_rows(base, stride, w, 0, h / 2, top, sizeof top);
	summarize_rows(base, stride, w, h / 2, h, bottom, sizeof bottom);
	if (!strcmp(top, bottom))
		snprintf(out, n, "%s", top);
	else
		snprintf(out, n, "%s|%s", top, bottom);
}

static void fill_rows(uint8_t *base, size_t stride, int w, int y0, int y1, uint32_t v)
{
	for (int y = y0; y < y1; y++) {
		uint32_t *row = (uint32_t *)(base + (size_t)y * stride);
		for (int x = 0; x < w; x++)
			row[x] = v;
	}
}

static id<MTLTexture> make_texture(Ctx *c)
{
	MTLTextureDescriptor *td =
	    [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
	                                                       width:c->w
	                                                      height:c->h
	                                                   mipmapped:NO];
	td.usage = c->usage;
	td.storageMode = c->storage;
	return [c->dev newTextureWithDescriptor:td iosurface:c->s plane:0];
}

static void gpu_read(Ctx *c, id<MTLTexture> t, char *out, size_t n, double *ms)
{
	id<MTLCommandBuffer> cb = [c->q commandBuffer];
	id<MTLBlitCommandEncoder> b = [cb blitCommandEncoder];
	[b copyFromTexture:t
	         sourceSlice:0
	         sourceLevel:0
	        sourceOrigin:MTLOriginMake(0, 0, 0)
	          sourceSize:MTLSizeMake(c->w, c->h, 1)
	            toBuffer:c->rb
	   destinationOffset:0
	destinationBytesPerRow:(NSUInteger)c->w * 4
	destinationBytesPerImage:(NSUInteger)c->w * c->h * 4];
	[b endEncoding];
	oracle_commit_wait(cb, ms, NULL);
	summarize(c->rb.contents, (size_t)c->w * 4, c->w, c->h, out, n);
}

static void gpu_write(Ctx *c, char l, int rows, double *ms)
{
	fill_rows(c->src.contents, (size_t)c->w * 4, c->w, 0, rows, word(l));
	id<MTLCommandBuffer> cb = [c->q commandBuffer];
	id<MTLBlitCommandEncoder> b = [cb blitCommandEncoder];
	[b copyFromBuffer:c->src
	           sourceOffset:0
	      sourceBytesPerRow:(NSUInteger)c->w * 4
	    sourceBytesPerImage:(NSUInteger)c->w * c->h * 4
	             sourceSize:MTLSizeMake(c->w, rows, 1)
	              toTexture:c->t
	       destinationSlice:0
	       destinationLevel:0
	      destinationOrigin:MTLOriginMake(0, 0, 0)];
	[b endEncoding];
	oracle_commit_wait(cb, ms, NULL);
}

static void gpu_clear(Ctx *c, char l, double *ms)
{
	double v = (uint8_t)l / 255.0;
	MTLRenderPassDescriptor *rp = [MTLRenderPassDescriptor renderPassDescriptor];
	rp.colorAttachments[0].texture = c->t;
	rp.colorAttachments[0].loadAction = MTLLoadActionClear;
	rp.colorAttachments[0].storeAction = MTLStoreActionStore;
	rp.colorAttachments[0].clearColor = MTLClearColorMake(v, v, v, v);
	id<MTLCommandBuffer> cb = [c->q commandBuffer];
	id<MTLRenderCommandEncoder> e = [cb renderCommandEncoderWithDescriptor:rp];
	[e endEncoding];
	oracle_commit_wait(cb, ms, NULL);
}

// One step; its result in `out`.
static void step(Ctx *c, const char *cmd, const char *arg, char *out, size_t n, double *ms)
{
	char l = arg && arg[0] ? arg[0] : 0;
	snprintf(out, n, "-");
	*ms = 0;
	double t0 = oracle_now_ms();
	IOSurfaceRef s = c->s;
	uint8_t *base = IOSurfaceGetBaseAddress(s);
	if (!strcmp(cmd, "cw") || !strcmp(cmd, "cwh")) {
		fill_rows(base, c->bpr, c->w, 0, cmd[2] ? c->h / 2 : c->h, word(l));
	} else if (!strcmp(cmd, "lw") || !strcmp(cmd, "lwh")) {
		kern_return_t kr = IOSurfaceLock(s, 0, NULL);
		double tl = oracle_now_ms() - t0;
		fill_rows(base, c->bpr, c->w, 0, cmd[2] ? c->h / 2 : c->h, word(l));
		IOSurfaceUnlock(s, 0, NULL);
		snprintf(out, n, "lock rc %#x took %.2f ms", kr, tl);
	} else if (!strcmp(cmd, "lrw") || !strcmp(cmd, "lro")) {
		uint32_t opt = cmd[2] == 'o' ? kIOSurfaceLockReadOnly : 0;
		kern_return_t kr = IOSurfaceLock(s, opt, NULL);
		double tl = oracle_now_ms() - t0;
		IOSurfaceUnlock(s, opt, NULL);
		snprintf(out, n, "lock rc %#x took %.2f ms", kr, tl);
	} else if (!strcmp(cmd, "cr")) {
		summarize(base, c->bpr, c->w, c->h, out, n);
	} else if (!strcmp(cmd, "lr") || !strcmp(cmd, "lrwr")) {
		uint32_t opt = cmd[2] ? 0 : kIOSurfaceLockReadOnly;
		kern_return_t kr = IOSurfaceLock(s, opt, NULL);
		double tl = oracle_now_ms() - t0;
		char sum[200];
		summarize(base, c->bpr, c->w, c->h, sum, sizeof sum);
		IOSurfaceUnlock(s, opt, NULL);
		snprintf(out, n, "%s (lock rc %#x took %.2f ms)", sum, kr, tl);
	} else if (!strcmp(cmd, "gr")) {
		gpu_read(c, c->t, out, n, ms);
		return;
	} else if (!strcmp(cmd, "grn")) {
		id<MTLTexture> t = make_texture(c);
		if (!t) {
			snprintf(out, n, "texture nil");
			return;
		}
		gpu_read(c, t, out, n, ms);
		return;
	} else if (!strcmp(cmd, "gw") || !strcmp(cmd, "gwh")) {
		gpu_write(c, l, cmd[2] ? c->h / 2 : c->h, ms);
		return;
	} else if (!strcmp(cmd, "gc")) {
		if (!(c->usage & MTLTextureUsageRenderTarget)) {
			snprintf(out, n, "n/a (not a render target)");
			return;
		}
		gpu_clear(c, l, ms);
		return;
	} else if (!strcmp(cmd, "nt")) {
		c->t = nil;
		c->t = make_texture(c);
		snprintf(out, n, c->t ? "storage %lu" : "texture nil", (unsigned long)c->t.storageMode);
	} else if (!strcmp(cmd, "cb")) {
		id<MTLCommandBuffer> cb = [c->q commandBuffer];
		id<MTLBlitCommandEncoder> b = [cb blitCommandEncoder];
		[b fillBuffer:c->other range:NSMakeRange(0, c->other.length) value:0x5a];
		[b endEncoding];
		oracle_commit_wait(cb, ms, NULL);
		return;
	} else if (!strcmp(cmd, "sl")) {
		usleep((useconds_t)atoi(arg) * 1000);
	} else {
		snprintf(out, n, "unknown step");
	}
	*ms = oracle_now_ms() - t0;
}

static void ctx_init(Ctx *c, const char *dev, int w, int h, const char *usage, const char *storage,
                     IOSurfaceID sid)
{
	c->dev = oracle_device(dev);
	c->q = [c->dev newCommandQueue];
	c->w = w;
	c->h = h;
	c->usage = !strcmp(usage, "ro") ? MTLTextureUsageShaderRead
	                                 : MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite |
	                                       MTLTextureUsageRenderTarget;
	c->storage = !strcmp(storage, "shared")    ? MTLStorageModeShared
	             : !strcmp(storage, "private") ? MTLStorageModePrivate
	                                           : MTLStorageModeManaged;
	if (sid) {
		c->s = IOSurfaceLookup(sid);
	} else {
		NSDictionary *p = @{
			(id)kIOSurfaceWidth : @(w),
			(id)kIOSurfaceHeight : @(h),
			(id)kIOSurfaceBytesPerElement : @4,
			(id)kIOSurfacePixelFormat : @((uint32_t)'BGRA'),
			(id)kIOSurfaceIsGlobal : @YES,
		};
		c->s = IOSurfaceCreate((__bridge CFDictionaryRef)p);
	}
	if (!c->s) {
		printf("! no surface\n");
		exit(1);
	}
	c->bpr = IOSurfaceGetBytesPerRow(c->s);
	size_t len = (size_t)w * h * 4;
	c->rb = [c->dev newBufferWithLength:len options:MTLResourceStorageModeShared];
	c->src = [c->dev newBufferWithLength:len options:MTLResourceStorageModeShared];
	c->other = [c->dev newBufferWithLength:65536 options:MTLResourceStorageModeShared];
	c->t = make_texture(c);
}

// The peer: reads "STEP ARG" lines, answers "RESULT\tSEED\tMS".
static int peer(int argc, char **argv)
{
	if (argc < 8)
		return 2;
	Ctx c = {0};
	ctx_init(&c, argv[2], atoi(argv[4]), atoi(argv[5]), argv[6], argv[7],
	         (IOSurfaceID)strtoul(argv[3], NULL, 0));
	char line[256];
	setvbuf(stdout, NULL, _IOLBF, 0);
	while (fgets(line, sizeof line, stdin)) {
		char cmd[16] = {0}, arg[32] = {0};
		if (sscanf(line, "%15s %31s", cmd, arg) < 1)
			continue;
		if (!strcmp(cmd, "q"))
			break;
		char out[256];
		double ms;
		@autoreleasepool {
			step(&c, cmd, arg, out, sizeof out, &ms);
		}
		printf("%s\t%u\t%.2f\n", out, IOSurfaceGetSeed(c.s), ms);
	}
	return 0;
}

typedef struct {
	pid_t pid;
	FILE *to, *from;
} Peer;

static void peer_start(Peer *p, const char *self, const char *dev, IOSurfaceID id, int w, int h,
                       const char *usage, const char *storage)
{
	int in[2], out[2];
	pipe(in);
	pipe(out);
	posix_spawn_file_actions_t fa;
	posix_spawn_file_actions_init(&fa);
	posix_spawn_file_actions_adddup2(&fa, in[0], 0);
	posix_spawn_file_actions_adddup2(&fa, out[1], 1);
	posix_spawn_file_actions_addclose(&fa, in[1]);
	posix_spawn_file_actions_addclose(&fa, out[0]);
	char ids[16], ws[16], hs[16];
	snprintf(ids, sizeof ids, "%u", id);
	snprintf(ws, sizeof ws, "%d", w);
	snprintf(hs, sizeof hs, "%d", h);
	char *args[] = {(char *)self, "--peer", (char *)dev, ids, ws, hs,
	                (char *)usage, (char *)storage, NULL};
	if (posix_spawn(&p->pid, self, &fa, NULL, args, environ)) {
		printf("! peer spawn failed\n");
		exit(1);
	}
	close(in[0]);
	close(out[1]);
	p->to = fdopen(in[1], "w");
	p->from = fdopen(out[0], "r");
	setvbuf(p->to, NULL, _IOLBF, 0);
}

static const char *const kSingle[] = {
    "# CPU -> GPU: writes without a lock",
    "A lw A", "A nt", "A gr",
    "A cw B", "A gr", "A cb", "A gr", "A sl 200", "A gr",
    "A grn", "A gr",
    "A lro", "A gr",
    "A lrw", "A gr",
    "# CPU -> GPU: a seed move uploads what",
    "A lw C", "A gr",
    "A cwh D", "A lrw", "A gr",
    "A lwh E", "A gr",
    "A cw F", "A nt", "A gr", "A grn",
    "# GPU -> CPU in the committing process",
    "A lw G", "A gr",
    "A gw H", "A cr", "A sl 200", "A cr", "A lr", "A cr",
    "A gc I", "A cr", "A lrwr", "A cr",
    "A gw J", "A lro", "A cr",
    "A gw K", "A lrw", "A cr",
    "# GPU write, then CPU write without a lock, then a lock",
    "A lw L", "A gr",
    "A gw M", "A cwh N", "A lrwr", "A cr", "A gr",
    "# GPU write, then a locked CPU write of half",
    "A gw O", "A lwh P", "A cr", "A gr",
    "# GPU half write, then a lock",
    "A lw Q", "A gr", "A gwh R", "A lr", "A gr",
    NULL,
};

static const char *const kCross[] = {
    "# A makes the surface; B is another process with its own texture",
    "A lw A", "A nt", "B nt", "A gr", "B gr",
    "# CPU -> GPU across processes",
    "B cw B", "A gr", "B gr", "A cb", "A gr",
    "B lro", "A gr",
    "B lrw", "A gr", "B gr",
    "B lw C", "A gr", "B gr",
    "B cw D", "A lrw", "A gr", "B gr",
    "# GPU -> CPU across processes",
    "A gw E", "A cr", "B cr", "B lr", "A cr", "B cr",
    "A gw F", "B gr", "B cr", "A cr", "A lr", "B cr",
    "# GPU -> GPU across processes",
    "B gw G", "A gr", "A gc H", "B gr",
    NULL,
};

static const char *const kTiming[] = {
    "# the cost of the syncs on a big surface: each pair repeated",
    "A lw A", "A nt", "A gr", "A gr", "A gr",
    "A lrw", "A gr", "A lrw", "A gr", "A lrw", "A gr",
    "A lro", "A gr", "A lro", "A gr",
    "A cw B", "A gr", "A cw C", "A gr",
    "A gw D", "A lr", "A lr", "A gw E", "A lr", "A lr", "A gw F", "A lr",
    "A gw G", "A lrw", "A gw H", "A lrw",
    "A gc I", "A lr", "A gc J", "A lr",
    "A cr", "A cr",
    NULL,
};

int main(int argc, char **argv)
{
	if (argc > 1 && !strcmp(argv[1], "--peer"))
		return peer(argc, argv);
	if (argc < 2) {
		fprintf(stderr, "usage: %s DEVICE [single|cross|timing|FILE [W H [rt|ro [managed|shared|private]]]]\n",
		        argv[0]);
		return 2;
	}
	setvbuf(stdout, NULL, _IOLBF, 0);
	const char *scenario = argc > 2 ? argv[2] : "single";
	int w = argc > 4 ? atoi(argv[3]) : 1024, h = argc > 4 ? atoi(argv[4]) : 1024;
	const char *usage = argc > 5 ? argv[5] : "rt";
	const char *storage = argc > 6 ? argv[6] : "managed";
	Ctx c = {0};
	ctx_init(&c, argv[1], w, h, usage, storage, 0);
	oracle_header("oracle_iosurface", c.dev);
	printf("# scenario %s, surface %dx%d BGRA8 bytesPerRow %zu alloc %zu id %u, usage %s (%#lx), "
	       "storage %s, texture %s\n",
	       scenario, w, h, c.bpr, IOSurfaceGetAllocSize(c.s), IOSurfaceGetID(c.s), usage,
	       (unsigned long)c.usage, storage, c.t ? "made" : "NIL");
	if (!c.t) {
		printf("RESULT texture over the surface is nil\n");
		return 1;
	}
	printf("# texture storageMode %lu cpuCacheMode %lu iosurface %s\n",
	       (unsigned long)c.t.storageMode, (unsigned long)c.t.cpuCacheMode,
	       c.t.iosurface ? "set" : "nil");
	const char *const *script = NULL;
	NSMutableArray<NSString *> *lines = nil;
	if (!strcmp(scenario, "single"))
		script = kSingle;
	else if (!strcmp(scenario, "cross"))
		script = kCross;
	else if (!strcmp(scenario, "timing"))
		script = kTiming;
	else {
		NSString *text = [NSString stringWithContentsOfFile:@(scenario)
		                                           encoding:NSUTF8StringEncoding
		                                              error:nil];
		lines = [[text componentsSeparatedByString:@"\n"] mutableCopy];
	}
	Peer p = {0};
	size_t count = script ? 0 : lines.count;
	if (script)
		while (script[count])
			count++;
	for (size_t i = 0; i < count; i++) {
		const char *line = script ? script[i] : lines[i].UTF8String;
		if (!line[0])
			continue;
		if (line[0] == '#') {
			printf("%s\n", line);
			continue;
		}
		char who[4] = {0}, cmd[16] = {0}, arg[32] = {0};
		sscanf(line, "%3s %15s %31s", who, cmd, arg);
		char out[256];
		double ms;
		uint32_t seed;
		if (who[0] == 'B') {
			if (!p.pid)
				peer_start(&p, argv[0], argv[1], IOSurfaceGetID(c.s), w, h, usage, storage);
			fprintf(p.to, "%s %s\n", cmd, arg[0] ? arg : "-");
			char reply[400];
			if (!fgets(reply, sizeof reply, p.from)) {
				printf("! peer died\n");
				return 1;
			}
			char *tab1 = strchr(reply, '\t');
			*tab1 = 0;
			snprintf(out, sizeof out, "%s", reply);
			seed = (uint32_t)strtoul(tab1 + 1, &tab1, 10);
			ms = strtod(tab1 + 1, NULL);
		} else {
			@autoreleasepool {
				step(&c, cmd, arg, out, sizeof out, &ms);
			}
			seed = IOSurfaceGetSeed(c.s);
		}
		printf("%s %-4s %-2s | %-40s | seed %3u | %8.2f ms\n", who, cmd, arg, out, seed, ms);
	}
	if (p.pid) {
		fprintf(p.to, "q\n");
		int st;
		waitpid(p.pid, &st, 0);
	}
	return 0;
}
