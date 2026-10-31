// surface_probe: one IOSurface shared by two processes on the reims-graphics Metal device.
//   surface_probe check [W H]        correctness across processes, three ways:
//       1. this process writes the surface on the CPU (a lock), the other samples it on the GPU;
//       2. the other renders into it on the GPU, this process reads it on the CPU (a lock);
//       3. this process renders into it on the GPU, the other samples it on the GPU;
//       4. both let their textures go and make them again, then 1 and 3 again.
//     Prints "SURFACE OK" when every pixel of every step matches, else "SURFACE FAIL: why".
//   surface_probe bench [W H SECONDS]  the desktop's hot path: each frame this process renders
//     the surface (a clear) and the other samples it (a blit into a Shared buffer), one after
//     the other, for SECONDS (12 by default, past the bundle's 10 s copy report); prints the
//     milliseconds a frame each side's command buffer took, from commit to completion.
//   W x H is 1920 x 1080 by default. The second process is this binary again, started with
//   "peer"; the two talk through a file in /tmp they both map.
// Build: xcrun clang -arch x86_64 -fobjc-arc -framework Metal -framework IOSurface
//        -framework Foundation surface_probe.m -o surface_probe
#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <Metal/Metal.h>
#include <fcntl.h>
#include <mach/mach_time.h>
#include <spawn.h>
#include <stdatomic.h>
// The macOS 27 SDK deprecates Managed storage and global surfaces, which the x86-64 guests this
// probes still have.
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>

extern char **environ;

// The mailbox: the parent posts an op with `req`, the peer answers it with `ack`.
struct Mailbox {
	_Atomic uint32_t req;
	_Atomic uint32_t ack;
	uint32_t op;
	uint32_t arg;
	uint32_t all; // kOpSample: check every pixel, else the first and the last
	uint32_t result;
	double ms; // the peer's command buffer, commit to completion
};

enum { kOpSample = 1, kOpRender = 2, kOpQuit = 3, kOpRemake = 4 };

static int W = 1920, H = 1080;
static id<MTLDevice> gDevice;
static id<MTLCommandQueue> gQueue;
static id<MTLTexture> gTexture;
static id<MTLBuffer> gReadback;

static double nowMs(void)
{
	static mach_timebase_info_data_t tb;
	if (tb.denom == 0)
		mach_timebase_info(&tb);
	return (double)mach_absolute_time() * tb.numer / tb.denom / 1e6;
}

// The pixel a clear to red `level` (of 255) leaves, as BGRA8's bytes read as a word.
static uint32_t redPixel(uint32_t level)
{
	return 0xff000000u | ((level & 0xff) << 16);
}

static id<MTLDevice> ourDevice(void)
{
	for (id<MTLDevice> d in MTLCopyAllDevices()) {
		if ([NSStringFromClass([d class]) isEqualToString:@"RgfxMetalDevice"])
			return d;
	}
	return nil;
}

// Lets the texture over the surface go, waits for the bundle to let its block go, and makes it
// again.
static int remake(IOSurfaceRef s)
{
	@autoreleasepool {
		gTexture = nil;
	}
	usleep(300 * 1000);
	MTLTextureDescriptor *td =
	    [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
	                                                       width:W
	                                                      height:H
	                                                   mipmapped:NO];
	td.usage = MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget;
	td.storageMode = MTLStorageModeManaged;
	gTexture = [gDevice newTextureWithDescriptor:td iosurface:s plane:0];
	return gTexture == nil;
}

static int setUp(IOSurfaceRef s)
{
	gDevice = ourDevice();
	if (gDevice == nil) {
		printf("SURFACE FAIL: no RgfxMetalDevice\n");
		return 1;
	}
	MTLTextureDescriptor *td =
	    [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
	                                                       width:W
	                                                      height:H
	                                                   mipmapped:NO];
	td.usage = MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget;
	td.storageMode = MTLStorageModeManaged;
	gTexture = [gDevice newTextureWithDescriptor:td iosurface:s plane:0];
	gQueue = [gDevice newCommandQueue];
	gReadback = [gDevice newBufferWithLength:(NSUInteger)W * H * 4
	                                 options:MTLResourceStorageModeShared];
	if (gTexture == nil || gQueue == nil || gReadback == nil) {
		printf("SURFACE FAIL: no texture over the surface, queue or buffer\n");
		return 1;
	}
	return 0;
}

// Commits `cb`, waits, and answers how long it took from commit to completion (-1 if failed).
static double run(id<MTLCommandBuffer> cb)
{
	double t0 = nowMs();
	[cb commit];
	[cb waitUntilCompleted];
	double ms = nowMs() - t0;
	return cb.status == MTLCommandBufferStatusCompleted ? ms : -1;
}

// Renders the texture red `level` (a clear).
static double render(uint32_t level)
{
	MTLRenderPassDescriptor *rp = [MTLRenderPassDescriptor renderPassDescriptor];
	rp.colorAttachments[0].texture = gTexture;
	rp.colorAttachments[0].loadAction = MTLLoadActionClear;
	rp.colorAttachments[0].storeAction = MTLStoreActionStore;
	rp.colorAttachments[0].clearColor = MTLClearColorMake((level & 0xff) / 255.0, 0, 0, 1);
	id<MTLCommandBuffer> cb = [gQueue commandBuffer];
	[[cb renderCommandEncoderWithDescriptor:rp] endEncoding];
	return run(cb);
}

// Samples the texture into the readback buffer (a blit); how many pixels are not `want`, every
// pixel when `all`, else the first and the last.
static uint32_t sample(uint32_t want, bool all, double *ms)
{
	id<MTLCommandBuffer> cb = [gQueue commandBuffer];
	id<MTLBlitCommandEncoder> b = [cb blitCommandEncoder];
	[b copyFromTexture:gTexture
	                 sourceSlice:0
	                 sourceLevel:0
	                sourceOrigin:MTLOriginMake(0, 0, 0)
	                  sourceSize:MTLSizeMake(W, H, 1)
	                    toBuffer:gReadback
	           destinationOffset:0
	      destinationBytesPerRow:(NSUInteger)W * 4
	    destinationBytesPerImage:(NSUInteger)W * H * 4];
	[b endEncoding];
	*ms = run(cb);
	if (*ms < 0)
		return (uint32_t)W * H;
	const uint32_t *p = gReadback.contents;
	uint32_t bad = 0;
	size_t n = (size_t)W * H;
	if (all) {
		for (size_t i = 0; i < n; i++)
			bad += p[i] != want;
	} else {
		bad = (p[0] != want) + (p[n - 1] != want);
	}
	return bad;
}

// The CPU's view: writes every pixel `v` under a lock, or counts the pixels that are not `v`.
static void cpuWrite(IOSurfaceRef s, uint32_t v)
{
	IOSurfaceLock(s, 0, NULL);
	uint8_t *base = IOSurfaceGetBaseAddress(s);
	size_t bpr = IOSurfaceGetBytesPerRow(s);
	for (int y = 0; y < H; y++) {
		uint32_t *row = (uint32_t *)(base + y * bpr);
		for (int x = 0; x < W; x++)
			row[x] = v;
	}
	IOSurfaceUnlock(s, 0, NULL);
}

static uint32_t cpuCheck(IOSurfaceRef s, uint32_t v)
{
	IOSurfaceLock(s, kIOSurfaceLockReadOnly, NULL);
	const uint8_t *base = IOSurfaceGetBaseAddress(s);
	size_t bpr = IOSurfaceGetBytesPerRow(s);
	uint32_t bad = 0;
	for (int y = 0; y < H; y++) {
		const uint32_t *row = (const uint32_t *)(base + y * bpr);
		for (int x = 0; x < W; x++)
			bad += row[x] != v;
	}
	IOSurfaceUnlock(s, kIOSurfaceLockReadOnly, NULL);
	return bad;
}

static struct Mailbox *mapMailbox(const char *path, bool create)
{
	int fd = open(path, O_RDWR | (create ? O_CREAT | O_TRUNC : 0), 0600);
	if (fd < 0 || (create && ftruncate(fd, sizeof(struct Mailbox)) != 0))
		return NULL;
	void *p = mmap(NULL, sizeof(struct Mailbox), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
	close(fd);
	return p == MAP_FAILED ? NULL : p;
}

// The peer: answers the parent's ops until it posts kOpQuit.
static int peer(IOSurfaceID sid, const char *path)
{
	struct Mailbox *mb = mapMailbox(path, false);
	IOSurfaceRef s = IOSurfaceLookup(sid);
	if (mb == NULL || s == NULL) {
		printf("SURFACE FAIL: the peer found no mailbox or no surface %u\n", sid);
		return 1;
	}
	if (setUp(s) != 0)
		return 1;
	uint32_t seen = 0;
	for (;;) {
		uint32_t req;
		while ((req = atomic_load(&mb->req)) == seen)
			usleep(20);
		seen = req;
		@autoreleasepool {
		switch (mb->op) {
		case kOpSample:
			mb->result = sample(mb->arg, mb->all != 0, &mb->ms);
			break;
		case kOpRender:
			mb->ms = render(mb->arg);
			mb->result = mb->ms < 0;
			break;
		case kOpRemake:
			mb->result = remake(s);
			break;
		case kOpQuit:
			atomic_store(&mb->ack, req);
			return 0;
		}
		}
		atomic_store(&mb->ack, req);
	}
}

// Posts `op` to the peer and waits for its answer.
static uint32_t ask(struct Mailbox *mb, uint32_t op, uint32_t arg, bool all)
{
	mb->op = op;
	mb->arg = arg;
	mb->all = all;
	uint32_t req = atomic_load(&mb->req) + 1;
	atomic_store(&mb->req, req);
	while (atomic_load(&mb->ack) != req)
		usleep(20);
	return mb->result;
}

static int cmpDouble(const void *a, const void *b)
{
	double x = *(const double *)a, y = *(const double *)b;
	return x < y ? -1 : x > y;
}

int main(int argc, char **argv)
{
	setvbuf(stdout, NULL, _IONBF, 0);
	@autoreleasepool {
		if (argc >= 6 && strcmp(argv[1], "peer") == 0) {
			W = atoi(argv[3]);
			H = atoi(argv[4]);
			return peer((IOSurfaceID)strtoul(argv[2], NULL, 0), argv[5]);
		}
		bool bench = argc >= 2 && strcmp(argv[1], "bench") == 0;
		if (argc < 2 || (!bench && strcmp(argv[1], "check") != 0)) {
			printf("usage: surface_probe check [W H] | bench [W H SECONDS]\n");
			return 2;
		}
		if (argc >= 4) {
			W = atoi(argv[2]);
			H = atoi(argv[3]);
		}
		double seconds = argc >= 5 ? atof(argv[4]) : 12;
		NSDictionary *props = @{
			(id)kIOSurfaceWidth : @(W),
			(id)kIOSurfaceHeight : @(H),
			(id)kIOSurfaceBytesPerElement : @4,
			(id)kIOSurfacePixelFormat : @((uint32_t)'BGRA'),
			(id)kIOSurfaceIsGlobal : @YES,
		};
		IOSurfaceRef s = IOSurfaceCreate((__bridge CFDictionaryRef)props);
		if (s == NULL) {
			printf("SURFACE FAIL: IOSurfaceCreate\n");
			return 1;
		}
		if (setUp(s) != 0)
			return 1;
		char path[64];
		snprintf(path, sizeof(path), "/tmp/surface_probe.%d", getpid());
		struct Mailbox *mb = mapMailbox(path, true);
		if (mb == NULL) {
			printf("SURFACE FAIL: no mailbox\n");
			return 1;
		}
		char sid[16], ws[16], hs[16];
		snprintf(sid, sizeof(sid), "%u", IOSurfaceGetID(s));
		snprintf(ws, sizeof(ws), "%d", W);
		snprintf(hs, sizeof(hs), "%d", H);
		char *pargv[] = {argv[0], "peer", sid, ws, hs, path, NULL};
		pid_t pid;
		if (posix_spawn(&pid, argv[0], NULL, NULL, pargv, environ) != 0) {
			printf("SURFACE FAIL: no peer process\n");
			return 1;
		}
		printf("surface %s %dx%d bpr %zu, peer pid %d\n", sid, W, H, IOSurfaceGetBytesPerRow(s),
		       pid);
		int rc = 0;
		if (!bench) {
			uint32_t bad;
			cpuWrite(s, 0x11223344);
			if ((bad = ask(mb, kOpSample, 0x11223344, true)) != 0) {
				printf("SURFACE FAIL: 1 the peer's GPU saw %u pixels not the CPU's\n", bad);
				rc = 1;
			} else {
				printf("1 CPU write here, GPU sample in the peer: every pixel\n");
			}
			if (rc == 0 && ask(mb, kOpRender, 0x66, false) != 0) {
				printf("SURFACE FAIL: 2 the peer's render failed\n");
				rc = 1;
			} else if (rc == 0 && (bad = cpuCheck(s, redPixel(0x66))) != 0) {
				printf("SURFACE FAIL: 2 the CPU saw %u pixels not the peer's render\n", bad);
				rc = 1;
			} else if (rc == 0) {
				printf("2 GPU render in the peer, CPU read here: every pixel\n");
			}
			if (rc == 0 && render(0x99) < 0) {
				printf("SURFACE FAIL: 3 the render failed\n");
				rc = 1;
			} else if (rc == 0 && (bad = ask(mb, kOpSample, redPixel(0x99), true)) != 0) {
				printf("SURFACE FAIL: 3 the peer's GPU saw %u pixels not the render\n", bad);
				rc = 1;
			} else if (rc == 0) {
				printf("3 GPU render here, GPU sample in the peer: every pixel\n");
			}
			// 4. Both let their textures go, so the plane's block goes, and make them again.
			if (rc == 0 && (ask(mb, kOpRemake, 0, false) != 0 || remake(s) != 0)) {
				printf("SURFACE FAIL: 4 a texture was not made again\n");
				rc = 1;
			} else if (rc == 0 && (cpuWrite(s, 0x22334455), 0) == 0 &&
			           (bad = ask(mb, kOpSample, 0x22334455, true)) != 0) {
				printf("SURFACE FAIL: 4 the peer's GPU saw %u pixels not the CPU's\n", bad);
				rc = 1;
			} else if (rc == 0 && render(0x44) < 0) {
				printf("SURFACE FAIL: 4 the render failed\n");
				rc = 1;
			} else if (rc == 0 && (bad = ask(mb, kOpSample, redPixel(0x44), true)) != 0) {
				printf("SURFACE FAIL: 4 the peer's GPU saw %u pixels not the render\n", bad);
				rc = 1;
			} else if (rc == 0) {
				printf("4 textures made again: CPU write and GPU render here, GPU sample in the "
				       "peer: every pixel\n");
			}
			if (rc == 0)
				printf("SURFACE OK\n");
		} else {
			size_t cap = 1 << 16, n = 0;
			double *mine = calloc(cap, sizeof(double)), *theirs = calloc(cap, sizeof(double));
			double end = nowMs() + seconds * 1000, start = nowMs();
			uint32_t bad = 0;
			while (nowMs() < end && n < cap) {
				@autoreleasepool {
					uint32_t level = (uint32_t)(n % 255) + 1;
					mine[n] = render(level);
					// The first and last pixel only, so the check costs the CPU nothing.
					bad += ask(mb, kOpSample, redPixel(level), false) != 0;
					theirs[n] = mb->ms;
					n++;
					if (getenv("SURFACE_PROBE_TRACE") != NULL && n % 100 == 0)
						fprintf(stderr, "frame %zu: %.3f ms here, %.3f ms in the peer\n", n,
						        mine[n - 1], theirs[n - 1]);
				}
			}
			double wall = nowMs() - start;
			qsort(mine, n, sizeof(double), cmpDouble);
			qsort(theirs, n, sizeof(double), cmpDouble);
			double sm = 0, st = 0;
			for (size_t i = 0; i < n; i++)
				sm += mine[i], st += theirs[i];
			printf("bench %dx%d: %zu frames in %.1f s (%.2f ms a frame), %u wrong\n", W, H, n,
			       wall / 1000, wall / n, bad);
			printf("  render here:   median %.3f ms, mean %.3f ms, p90 %.3f ms\n", mine[n / 2],
			       sm / n, mine[n * 9 / 10]);
			printf("  sample (peer): median %.3f ms, mean %.3f ms, p90 %.3f ms\n", theirs[n / 2],
			       st / n, theirs[n * 9 / 10]);
			rc = bad != 0;
		}
		ask(mb, kOpQuit, 0, false);
		int status = 0;
		waitpid(pid, &status, 0);
		unlink(path);
		return rc;
	}
}
