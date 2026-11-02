// dormant_stress: drives the kext's dormant surface blocks past each of their caps, in waves,
// and checks the surfaces' pixels on the way (design.md, "The device cache": dormancy).
//   dormant_stress [GPU]   on the device whose name contains GPU, else
//                          MTLCreateSystemDefaultDevice's.
// Every surface is BGRA8, its rows 256-byte aligned and of one plane, so the bundle asks a
// cached block in place for it; each texture over one is cleared by a render pass to the
// surface's own colour, so every block holds a GPU write its let-go copies into the pages. A
// texture released while its surface lives leaves the block dormant. The waves, sized for a
// 16 GiB guest (1024 dormant blocks, 1536 table entries, 2048 MiB of dormant surfaces):
//   W1  count:  1100 surfaces of 256x256, each textured once: past 1024 dormant blocks the
//               oldest go;
//   W2  table:  600 more with their textures held, 1624 entries on 1536: dormant blocks give
//               theirs; then the textures go;
//   W3  op 7, op 5: of W2's surfaces 200 made volatile, 100 empty, 50 volatile then nonvolatile;
//               then every small surface is released;
//   W4  bytes:  44 surfaces of 4096x4096 (64 MiB), 8 of the first 16 made volatile once
//               dormant: past 2048 MiB the volatile ones go first, then the oldest;
//   W5  CPU:    16 of W4's dormant surfaces locked read-write and written by the CPU, then
//               read through a texture by the GPU; 8 more read under a read-only lock;
//   W6  room:   the region filled with Shared buffers until none is made, then a texture over a
//               new surface: dormant blocks give the region room; the buffers go at once;
//   W7  end:    every surface released.
// Between waves it waits 12 s and makes one texture, so the kext's line of the dormant blocks
// (dmesg, at most one each 10 s) falls between them. Each wave prints its start and end on the
// clock dmesg stamps its lines with (seconds since boot), a line per check, PASS or FAIL, and at
// the end "STRESS OK" where every check passed, else "STRESS FAIL".
// Build: xcrun clang -arch x86_64 -fobjc-arc -framework Metal -framework IOSurface
//        -framework Foundation dormant_stress.m -o dormant_stress
#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <Metal/Metal.h>
#include <mach/mach_time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static id<MTLDevice> gDevice;
static id<MTLCommandQueue> gQueue;
static int gFailures;

static double now(void)
{
	static mach_timebase_info_data_t tb;
	if (tb.denom == 0)
		mach_timebase_info(&tb);
	return (double)mach_absolute_time() * tb.numer / tb.denom / 1e9;
}

static void check(bool ok, NSString *what)
{
	if (!ok)
		gFailures++;
	printf("%s %s\n", ok ? "PASS" : "FAIL", what.UTF8String);
	fflush(stdout);
}

// Surface `i` of wave `w`'s colour, as BGRA8 in memory: blue the index's low byte, green its
// high bits, red the wave.
static uint32_t colourOf(int w, int i)
{
	return 0xff000000u | (uint32_t)(0x20 + 0x18 * w) << 16 | (uint32_t)(0x40 | (i >> 8 & 0x3f)) << 8 |
	       (uint32_t)(i & 0xff);
}

// The CPU's pattern of W5, distinct from every colour: alpha below opaque.
static uint32_t patternOf(int i)
{
	return 0x80000000u | (uint32_t)(i * 7 & 0xff) << 16 | 0x00a500u | (uint32_t)(255 - i);
}

// Whether `got` is `want` within one in each channel, as a GPU's conversion may round.
static bool samePixel(uint32_t got, uint32_t want)
{
	for (int c = 0; c < 32; c += 8) {
		int d = (int)(got >> c & 0xff) - (int)(want >> c & 0xff);
		if (d < -1 || d > 1)
			return false;
	}
	return true;
}

static IOSurfaceRef newSurface(int w, int h)
{
	NSDictionary *props = @{
		(id)kIOSurfaceWidth : @(w),
		(id)kIOSurfaceHeight : @(h),
		(id)kIOSurfaceBytesPerElement : @4,
		(id)kIOSurfaceBytesPerRow : @(w * 4),
		(id)kIOSurfacePixelFormat : @((uint32_t)'BGRA'),
	};
	return IOSurfaceCreate((__bridge CFDictionaryRef)props);
}

static id<MTLTexture> textureOver(IOSurfaceRef s)
{
	MTLTextureDescriptor *d =
	    [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
	                                                       width:IOSurfaceGetWidth(s)
	                                                      height:IOSurfaceGetHeight(s)
	                                                   mipmapped:NO];
	d.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
	d.storageMode = MTLStorageModeShared;
	return [gDevice newTextureWithDescriptor:d iosurface:s plane:0];
}

// Clears `t` to `bgra` with a render pass, and waits for it.
static bool clearTo(id<MTLTexture> t, uint32_t bgra)
{
	MTLRenderPassDescriptor *p = [MTLRenderPassDescriptor renderPassDescriptor];
	p.colorAttachments[0].texture = t;
	p.colorAttachments[0].loadAction = MTLLoadActionClear;
	p.colorAttachments[0].storeAction = MTLStoreActionStore;
	p.colorAttachments[0].clearColor =
	    MTLClearColorMake((bgra >> 16 & 0xff) / 255.0, (bgra >> 8 & 0xff) / 255.0,
	                      (bgra & 0xff) / 255.0, (bgra >> 24 & 0xff) / 255.0);
	id<MTLCommandBuffer> cb = [gQueue commandBuffer];
	id<MTLRenderCommandEncoder> e = [cb renderCommandEncoderWithDescriptor:p];
	[e endEncoding];
	[cb commit];
	[cb waitUntilCompleted];
	return cb.status == MTLCommandBufferStatusCompleted;
}

// Makes a texture over `s`, clears it to `bgra`, and lets the texture go: its block sleeps.
static bool paint(IOSurfaceRef s, uint32_t bgra)
{
	@autoreleasepool {
		id<MTLTexture> t = textureOver(s);
		return t != nil && clearTo(t, bgra);
	}
}

// The pixels at the corners and the middle of `s`, read under a CPU lock of `options`, all
// `want`: else the first that is not, in `*bad`.
static bool cpuReads(IOSurfaceRef s, uint32_t options, uint32_t want, uint32_t *bad)
{
	if (IOSurfaceLock(s, options, NULL) != kIOReturnSuccess) {
		*bad = 0xdeadbeef;
		return false;
	}
	const uint32_t *px = (const uint32_t *)IOSurfaceGetBaseAddress(s);
	size_t stride = IOSurfaceGetBytesPerRow(s) / 4, w = IOSurfaceGetWidth(s),
	       h = IOSurfaceGetHeight(s);
	size_t at[5][2] = {{0, 0}, {w - 1, 0}, {0, h - 1}, {w - 1, h - 1}, {w / 2, h / 2}};
	bool ok = true;
	for (int k = 0; k < 5 && ok; k++) {
		uint32_t v = px[at[k][1] * stride + at[k][0]];
		if (!samePixel(v, want)) {
			*bad = v;
			ok = false;
		}
	}
	IOSurfaceUnlock(s, options, NULL);
	return ok;
}

// The same pixels, read by the GPU through a new texture over `s` (a blit into a buffer).
static bool gpuReads(IOSurfaceRef s, uint32_t want, uint32_t *bad)
{
	@autoreleasepool {
		id<MTLTexture> t = textureOver(s);
		id<MTLBuffer> out = [gDevice newBufferWithLength:5 * 256 options:MTLResourceStorageModeShared];
		if (t == nil || out == nil) {
			*bad = 0xdeadbeef;
			return false;
		}
		size_t w = t.width, h = t.height;
		size_t at[5][2] = {{0, 0}, {w - 1, 0}, {0, h - 1}, {w - 1, h - 1}, {w / 2, h / 2}};
		id<MTLCommandBuffer> cb = [gQueue commandBuffer];
		id<MTLBlitCommandEncoder> b = [cb blitCommandEncoder];
		for (int k = 0; k < 5; k++)
			[b copyFromTexture:t
			         sourceSlice:0
			         sourceLevel:0
			        sourceOrigin:MTLOriginMake(at[k][0], at[k][1], 0)
			          sourceSize:MTLSizeMake(1, 1, 1)
			            toBuffer:out
			   destinationOffset:256 * k
			destinationBytesPerRow:4
			destinationBytesPerImage:4];
		[b endEncoding];
		[cb commit];
		[cb waitUntilCompleted];
		const uint8_t *bytes = (const uint8_t *)out.contents;
		for (int k = 0; k < 5; k++) {
			uint32_t v;
			memcpy(&v, bytes + 256 * k, 4);
			if (!samePixel(v, want)) {
				*bad = v;
				return false;
			}
		}
		return true;
	}
}

static void wave(const char *name, const char *what)
{
	printf("== %s at %.3f: %s\n", name, now(), what);
	fflush(stdout);
}

// Waits out the kext's 10 s between lines of the dormant blocks, then gives it a change to log.
static void pause12(void)
{
	sleep(12);
	IOSurfaceRef s = newSurface(256, 256);
	paint(s, colourOf(0, 0));
	CFRelease(s);
	usleep(300000);
	printf("   (pause and poke at %.3f)\n", now());
	fflush(stdout);
}

// Checks surfaces `idx` of `all` (wave `w`) under a read-only CPU lock.
static void sample(const char *what, IOSurfaceRef *all, const int *idx, int n, int w)
{
	int bad = 0;
	uint32_t v = 0, first = 0;
	int firstAt = -1;
	for (int k = 0; k < n; k++) {
		if (!cpuReads(all[idx[k]], kIOSurfaceLockReadOnly, colourOf(w, idx[k]), &v)) {
			if (firstAt < 0) {
				firstAt = idx[k];
				first = v;
			}
			bad++;
		}
	}
	check(bad == 0, [NSString stringWithFormat:@"%s: %d of %d surfaces read their colour%@", what,
	                                           n - bad, n,
	                                           firstAt < 0 ? @""
	                                                       : [NSString stringWithFormat:@" (first bad: %d read 0x%08x, want 0x%08x)",
	                                                                                    firstAt, first, colourOf(w, firstAt)]]);
}

int main(int argc, char **argv)
{
	@autoreleasepool {
		gDevice = MTLCreateSystemDefaultDevice();
		if (argc > 1) {
			for (id<MTLDevice> d in MTLCopyAllDevices())
				if ([d.name containsString:@(argv[1])])
					gDevice = d;
		}
		gQueue = [gDevice newCommandQueue];
		printf("device %s, pid %d\n", gDevice.name.UTF8String, getpid());
		if (gQueue == nil) {
			printf("STRESS FAIL: no queue\n");
			return 1;
		}

		// W1: past the count.
		enum { N1 = 1100, N2 = 600, N4 = 44 };
		static IOSurfaceRef s1[N1], s2[N2], s4[N4];
		wave("W1", "1100 surfaces of 256x256, each textured once and let go");
		int made = 0;
		for (int i = 0; i < N1; i++) {
			s1[i] = newSurface(256, 256);
			made += s1[i] != NULL && paint(s1[i], colourOf(1, i));
		}
		check(made == N1, [NSString stringWithFormat:@"W1: %d of %d surfaces painted", made, N1]);
		usleep(300000);
		int idx1[32];
		for (int k = 0; k < 8; k++)
			idx1[k] = k * 9; // among the oldest, let go past the count
		for (int k = 8; k < 32; k++)
			idx1[k] = 80 + (k - 8) * 42;
		sample("W1", s1, idx1, 32, 1);
		wave("W1 end", "");
		pause12();

		// W2: the table full.
		wave("W2", "600 surfaces of 256x256 with their textures held, then let go");
		NSMutableArray<id<MTLTexture>> *held = [NSMutableArray array];
		made = 0;
		for (int i = 0; i < N2; i++) {
			s2[i] = newSurface(256, 256);
			@autoreleasepool {
				id<MTLTexture> t = s2[i] ? textureOver(s2[i]) : nil;
				if (t != nil && clearTo(t, colourOf(2, i))) {
					[held addObject:t];
					made++;
				}
			}
		}
		check(made == N2, [NSString stringWithFormat:@"W2: %d of %d textures made and held", made, N2]);
		int idx2[24];
		for (int k = 0; k < 24; k++)
			idx2[k] = k * 25;
		sample("W2 held", s2, idx2, 24, 2);
		int idx1b[16];
		for (int k = 0; k < 16; k++)
			idx1b[k] = 76 + k * 64;
		sample("W2 W1's", s1, idx1b, 16, 1);
		[held removeAllObjects];
		usleep(300000);
		sample("W2 let go", s2, idx2, 24, 2);
		wave("W2 end", "");
		pause12();

		// W3: op 7, op 6, then op 5.
		wave("W3", "of W2's surfaces 200 volatile, 100 empty, 50 volatile then nonvolatile; then "
		           "every small surface released");
		uint32_t old = 0;
		int set = 0;
		for (int i = 0; i < 200; i++)
			set += IOSurfaceSetPurgeable(s2[i], kIOSurfacePurgeableVolatile, &old) == kIOReturnSuccess;
		for (int i = 200; i < 300; i++)
			set += IOSurfaceSetPurgeable(s2[i], kIOSurfacePurgeableEmpty, &old) == kIOReturnSuccess;
		for (int i = 300; i < 350; i++) {
			set += IOSurfaceSetPurgeable(s2[i], kIOSurfacePurgeableVolatile, &old) == kIOReturnSuccess;
			IOSurfaceSetPurgeable(s2[i], kIOSurfacePurgeableNonVolatile, &old);
		}
		check(set == 350, [NSString stringWithFormat:@"W3: %d of 350 purgeable states set", set]);
		usleep(300000);
		int idx3[8];
		for (int k = 0; k < 8; k++)
			idx3[k] = 350 + k * 30;
		sample("W3 nonvolatile", s2, idx3, 8, 2);
		for (int i = 0; i < N1; i++)
			if (s1[i])
				CFRelease(s1[i]);
		for (int i = 0; i < N2; i++)
			if (s2[i])
				CFRelease(s2[i]);
		usleep(300000);
		wave("W3 end", "");
		pause12();

		// W4: past the bytes, the volatile first.
		wave("W4", "44 surfaces of 4096x4096 (64 MiB), 8 of the first 16 volatile once dormant");
		made = 0;
		for (int i = 0; i < N4; i++) {
			s4[i] = newSurface(4096, 4096);
			made += s4[i] != NULL && paint(s4[i], colourOf(4, i));
			if (i == 15) {
				usleep(100000);
				for (int k = 0; k < 8; k++)
					IOSurfaceSetPurgeable(s4[k], kIOSurfacePurgeableVolatile, &old);
			}
		}
		check(made == N4, [NSString stringWithFormat:@"W4: %d of %d surfaces painted", made, N4]);
		usleep(300000);
		int idx4[8] = {8, 9, 10, 11, 40, 41, 42, 43};
		sample("W4 nonvolatile", s4, idx4, 8, 4);
		wave("W4 end", "");
		pause12();

		// W5: CPU locks of dormant blocks.
		wave("W5", "16 dormant surfaces written by the CPU, read by the GPU; 8 read read-only");
		int good = 0;
		uint32_t v = 0;
		for (int i = 12; i < 28; i++) {
			if (IOSurfaceLock(s4[i], 0, NULL) != kIOReturnSuccess)
				continue;
			uint32_t *px = (uint32_t *)IOSurfaceGetBaseAddress(s4[i]);
			size_t n = IOSurfaceGetAllocSize(s4[i]) / 4;
			for (size_t k = 0; k < n; k++)
				px[k] = patternOf(i);
			IOSurfaceUnlock(s4[i], 0, NULL);
			if (gpuReads(s4[i], patternOf(i), &v))
				good++;
			else
				printf("   surface %d: the GPU read 0x%08x, want 0x%08x\n", i, v, patternOf(i));
		}
		check(good == 16, [NSString stringWithFormat:@"W5: %d of 16 CPU writes read by the GPU", good]);
		int idx5[8];
		for (int k = 0; k < 8; k++)
			idx5[k] = 28 + k;
		sample("W5 read-only", s4, idx5, 8, 4);
		wave("W5 end", "");
		pause12();

		// W6: the region's room.
		wave("W6", "the region filled with Shared buffers, then a texture over a new surface");
		NSMutableArray<id<MTLBuffer>> *fill = [NSMutableArray array];
		const NSUInteger sizes[] = {32u << 20, 1u << 20, 64u << 10, 4u << 10};
		uint64_t filled = 0;
		double f0 = now();
		for (int k = 0; k < 4; k++) {
			for (;;) {
				id<MTLBuffer> b = [gDevice newBufferWithLength:sizes[k]
				                                       options:MTLResourceStorageModeShared];
				if (b == nil)
					break;
				[fill addObject:b];
				filled += sizes[k];
				if (filled > (8ull << 30))
					break;
			}
		}
		double f1 = now();
		IOSurfaceRef r = newSurface(256, 256);
		id<MTLTexture> rt = textureOver(r);
		double f2 = now();
		[fill removeAllObjects];
		double f3 = now();
		printf("   %llu MiB of buffers in %.0f ms; the texture %s in %.0f ms; the buffers gone in "
		       "%.0f ms\n",
		       filled >> 20, (f1 - f0) * 1e3, rt ? "made" : "NOT made", (f2 - f1) * 1e3,
		       (f3 - f2) * 1e3);
		check(rt != nil && filled < (8ull << 30),
		      [NSString stringWithFormat:@"W6: a texture made over a new surface in a full region"]);
		bool painted = rt != nil && clearTo(rt, colourOf(6, 0));
		rt = nil;
		usleep(100000);
		check(painted && cpuReads(r, kIOSurfaceLockReadOnly, colourOf(6, 0), &v),
		      [NSString stringWithFormat:@"W6: the new surface reads its colour (0x%08x)", v]);
		CFRelease(r);
		wave("W6 end", "");
		pause12();

		// W7: the end.
		wave("W7", "every surface released");
		for (int i = 0; i < N4; i++)
			if (s4[i])
				CFRelease(s4[i]);
		usleep(500000);
		wave("W7 end", "");
		pause12();
		printf("%s (%d failures)\n", gFailures ? "STRESS FAIL" : "STRESS OK", gFailures);
		return gFailures ? 1 : 0;
	}
}
