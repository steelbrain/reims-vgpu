// upload_probe: what the CPU's uploads into textures and Managed buffers cost on a device, at the
// shapes WindowServer uploads: replaceRegion: of Managed and Shared textures, didModifyRange: of
// Managed buffers, and replaceRegion: and didModifyData (SPI) of linear textures over Managed
// buffers. To compare a guest's GPU lane with real Macs, case by case. Each case warms up, then
// times OPS operations and prints the median and the 99th percentile of each step in microseconds
// (us).
//   upload_probe [-d NAME] [-n OPS] [-k BURST] [-l] [CASE...]
//     -d NAME   the first of MTLCopyAllDevices whose name holds NAME; the default is
//               MTLCreateSystemDefaultDevice (on a dual-GPU MacBook that switches the display to
//               the discrete GPU: pass -d there).
//     -n OPS    operations a sample (default 200).
//     -k BURST  the calls of a burst case (default 64).
//     -l        lists the devices and exits.
//     CASE      runs only the cases whose name holds one of these; -CASE skips the cases whose
//               name holds CASE. With none, every case.
// SHAPE is FORMAT.WxH, FORMAT bgra8, a8 or rg8: bgra8.128x32, bgra8.336x64, bgra8.480x36, a8.64x64,
// rg8.64x64 (the widths of the linear textures WindowServer made at pitches 512, 1344, 1920 and 64
// bytes) and bgra8.1024x1024. SIZE is a byte count: 256, 4k, 64k.
//   rr.SHAPE        replaceRegion: of a whole Managed texture: call_us, the call alone, calls back to
//               back, each on the next texture of a pool of 16; spaced_us, the call alone, calls 2 ms
//               apart, the device idle between; e2e_us, one call, then a command buffer that blits
//               the texture into a Shared buffer, committed and waited; base_us, that command buffer
//               with no call before it; ok, the copies that read the bytes replaced.
//   srr.SHAPE       as rr.SHAPE, of a Shared texture.
//   mk.SHAPE        a new Managed texture, as WindowServer makes one in place of a linear texture a
//               device refused: make_us (newTextureWithDescriptor:), free_us (its release), and
//               first_us, a new texture, a replaceRegion: of all of it and a command buffer that
//               blits it into a Shared buffer, committed and waited, against base_us, that command
//               buffer over a texture made before; checked.
//   burst.SHAPE     BURST replaceRegion: calls, each on a Managed texture of a pool of BURST, then
//               one command buffer that blits every one into a Shared buffer, waited: calls_us (the
//               calls), burst_us (the calls, the commit and the wait), base_us (that command buffer
//               alone, no call before it), per_us ((burst_us - base_us) / BURST); checked. Not of
//               bgra8.1024x1024.
//   dmr.SIZE        didModifyRange: of SIZE bytes of a 4 MiB Managed buffer whose CPU copy was
//               written first: call_us, spaced_us, e2e_us (one write and call, then a blit of the
//               range into a Shared buffer, waited), base_us; checked.
//   dmrburst.SIZE   BURST didModifyRange: calls of SIZE bytes, each its own range, then one blit of
//               every range, waited: calls_us, burst_us, base_us, per_us, as burst.
//   linrr.SHAPE     a linear texture over a Managed buffer (the linear SPI,
//               newLinearTextureWithDescriptor:offset:bytesPerRow:bytesPerImage:, else
//               newTextureWithDescriptor:offset:bytesPerRow:) at offset 0, its rows at the device's
//               minimumLinearTextureAlignmentForPixelFormat:, of a buffer just its size:
//               replaceRegion: of the whole texture: call_us, spaced_us, e2e_us, base_us; checked.
//   dmd.SHAPE       that linear texture: the CPU writes the buffer whole, then didModifyData (SPI):
//               call_us (didModifyData alone), spaced_us, e2e_us (write, call, a blit of the texture
//               into a Shared buffer, waited), base_us; checked. SKIP where the texture does not
//               answer didModifyData.
// One line a case: "upload: NAME OK key=value ...", or "upload: NAME FAIL why" / "SKIP why"; "# start
// NAME" before it with the time since the process started and the Unix time, so a host's samples
// line up with the cases. Build: make upload-probe (build/upload_probe-x86_64); make upload-probe
// ARCH=arm64 for an Apple-silicon Mac.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <mach/mach_time.h>
#include <sys/time.h>
#include <unistd.h>

#pragma clang diagnostic ignored "-Wunguarded-availability-new"

// The SPI the cases call where a device answers them.
@protocol UploadProbeLinearTexture
- (id<MTLTexture>)newLinearTextureWithDescriptor:(MTLTextureDescriptor *)descriptor
                                          offset:(NSUInteger)offset
                                     bytesPerRow:(NSUInteger)bytesPerRow
                                   bytesPerImage:(NSUInteger)bytesPerImage;
@end

@protocol UploadProbeDidModifyData
- (void)didModifyData;
@end

static id<MTLDevice> gDevice;
static id<MTLCommandQueue> gQueue;
static NSArray<NSString *> *gPatterns;
static int gOps = 200;
static int gBurst = 64;
static double gStart;

// --- helpers ------------------------------------------------------------------------------------

static double now(void)
{
	static mach_timebase_info_data_t tb;
	if (tb.denom == 0)
		mach_timebase_info(&tb);
	return (double)mach_absolute_time() * tb.numer / tb.denom / 1e9;
}

static double unixNow(void)
{
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return tv.tv_sec + tv.tv_usec / 1e6;
}

// Samples of one step, in microseconds.
typedef struct {
	double *v;
	size_t n, cap;
} Samples;

static void add(Samples *s, double seconds)
{
	if (s->n == s->cap) {
		s->cap = s->cap ? s->cap * 2 : 256;
		s->v = realloc(s->v, s->cap * sizeof(double));
	}
	s->v[s->n++] = seconds * 1e6;
}

static int cmpDouble(const void *a, const void *b)
{
	const double x = *(const double *)a, y = *(const double *)b;
	return x < y ? -1 : x > y;
}

static double pct(const Samples *s, double p)
{
	if (s->n == 0)
		return NAN;
	double *c = malloc(s->n * sizeof(double));
	memcpy(c, s->v, s->n * sizeof(double));
	qsort(c, s->n, sizeof(double), cmpDouble);
	size_t i = (size_t)(p * (s->n - 1) + 0.5);
	const double r = c[i];
	free(c);
	return r;
}

// "KEY_us=P50 KEY_p99=P99" of the samples.
static NSString *st(const char *key, const Samples *s)
{
	return [NSString stringWithFormat:@"%s_us=%.1f %s_p99=%.1f", key, pct(s, 0.5), key, pct(s, 0.99)];
}

static void drop(Samples *s)
{
	free(s->v);
	*s = (Samples){0};
}

static void finish(id<MTLCommandBuffer> cb)
{
	[cb waitUntilCompleted];
	if (cb.status != MTLCommandBufferStatusCompleted)
		[NSException raise:@"upload" format:@"command buffer status %ld: %@", (long)cb.status,
		                                    cb.error.localizedDescription ?: @"no error"];
}

static BOOL wanted(NSString *name)
{
	BOOL any = NO, picked = NO;
	for (NSString *p in gPatterns) {
		if ([p hasPrefix:@"-"]) {
			if ([name containsString:[p substringFromIndex:1]])
				return NO;
			continue;
		}
		any = YES;
		picked = picked || [name containsString:p];
	}
	return any ? picked : YES;
}

static void runCase(NSString *name, NSString * (^body)(void))
{
	if (!wanted(name))
		return;
	printf("# start %s t=%.3f unix=%.3f\n", name.UTF8String, now() - gStart, unixNow());
	fflush(stdout);
	NSString *line = nil;
	@try {
		@autoreleasepool {
			line = body();
		}
	} @catch (NSException *e) {
		line = [@"FAIL " stringByAppendingString:e.reason ?: e.name];
	}
	if (![line hasPrefix:@"FAIL"] && ![line hasPrefix:@"SKIP"])
		line = [@"OK " stringByAppendingString:line];
	printf("upload: %s %s\n", name.UTF8String, line.UTF8String);
	fflush(stdout);
}

// A texture shape: its format, the bytes of a texel, and its size.
typedef struct {
	const char *name;
	MTLPixelFormat format;
	NSUInteger bpp, w, h;
} Shape;

static const Shape kShapes[] = {
    {"bgra8.128x32", MTLPixelFormatBGRA8Unorm, 4, 128, 32},
    {"bgra8.336x64", MTLPixelFormatBGRA8Unorm, 4, 336, 64},
    {"bgra8.480x36", MTLPixelFormatBGRA8Unorm, 4, 480, 36},
    {"a8.64x64", MTLPixelFormatA8Unorm, 1, 64, 64},
    {"rg8.64x64", MTLPixelFormatRG8Unorm, 2, 64, 64},
    {"bgra8.1024x1024", MTLPixelFormatBGRA8Unorm, 4, 1024, 1024},
};

static MTLTextureDescriptor *textureDesc(const Shape *s, MTLStorageMode storage)
{
	MTLTextureDescriptor *d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:s->format
	                                                                             width:s->w
	                                                                            height:s->h
	                                                                         mipmapped:NO];
	d.storageMode = storage;
	d.usage = MTLTextureUsageShaderRead;
	return d;
}

static MTLTextureDescriptor *managedDesc(const Shape *s)
{
	return textureDesc(s, MTLStorageModeManaged);
}

static id<MTLTexture> storedTexture(const Shape *s, MTLStorageMode storage)
{
	id<MTLTexture> t = [gDevice newTextureWithDescriptor:textureDesc(s, storage)];
	if (t == nil)
		[NSException raise:@"upload" format:@"no texture of %s, storage %lu", s->name, (unsigned long)storage];
	return t;
}

static id<MTLTexture> managedTexture(const Shape *s)
{
	return storedTexture(s, MTLStorageModeManaged);
}

// Encodes a blit of all of `t` into `out` at `at`, rows packed.
static void blitOut(id<MTLBlitCommandEncoder> e, id<MTLTexture> t, const Shape *s, id<MTLBuffer> out,
                    NSUInteger at)
{
	[e copyFromTexture:t
	             sourceSlice:0
	             sourceLevel:0
	            sourceOrigin:MTLOriginMake(0, 0, 0)
	              sourceSize:MTLSizeMake(s->w, s->h, 1)
	                toBuffer:out
	       destinationOffset:at
	  destinationBytesPerRow:s->w * s->bpp
	destinationBytesPerImage:s->w * s->bpp * s->h];
}

// Whether the packed copy at `p` of a shape's texels holds `v` at its first, middle and last byte.
static BOOL holds(const uint8_t *p, NSUInteger bytes, uint8_t v)
{
	return p[0] == v && p[bytes / 2] == v && p[bytes - 1] == v;
}

// A command buffer that blits each of `ts` into `out`, packed, committed and waited: its time.
static double blitAll(NSArray<id<MTLTexture>> *ts, const Shape *s, id<MTLBuffer> out)
{
	const NSUInteger bytes = s->w * s->bpp * s->h;
	const double t0 = now();
	id<MTLCommandBuffer> cb = [gQueue commandBuffer];
	id<MTLBlitCommandEncoder> e = [cb blitCommandEncoder];
	for (NSUInteger i = 0; i < ts.count; i++)
		blitOut(e, ts[i], s, out, i * bytes);
	[e endEncoding];
	[cb commit];
	finish(cb);
	return now() - t0;
}

// --- the cases ----------------------------------------------------------------------------------

// replaceRegion: of a whole texture of `storage`: Managed (rr.SHAPE) or Shared (srr.SHAPE).
static NSString *replaceCase(const Shape *s, MTLStorageMode storage)
{
	enum { kPool = 16 };
	const NSUInteger row = s->w * s->bpp, bytes = row * s->h;
	id<MTLTexture> pool[kPool];
	for (int i = 0; i < kPool; i++)
		pool[i] = storedTexture(s, storage);
	id<MTLBuffer> out = [gDevice newBufferWithLength:bytes options:MTLResourceStorageModeShared];
	uint8_t *src = malloc(bytes);
	const MTLRegion all = MTLRegionMake2D(0, 0, s->w, s->h);
	Samples call = {0}, spaced = {0}, e2e = {0}, base = {0};
	int bad = 0, checked = 0;
	// Back to back.
	for (int i = -8; i < gOps; i++) {
		memset(src, (uint8_t)i, bytes);
		const double t0 = now();
		[pool[(i + 8) % kPool] replaceRegion:all mipmapLevel:0 withBytes:src bytesPerRow:row];
		if (i >= 0)
			add(&call, now() - t0);
	}
	// Spaced: the device idle between.
	const int nSpaced = gOps < 100 ? gOps : 100;
	for (int i = -2; i < nSpaced; i++) {
		usleep(2000);
		memset(src, (uint8_t)(i + 7), bytes);
		const double t0 = now();
		[pool[(i + 2) % kPool] replaceRegion:all mipmapLevel:0 withBytes:src bytesPerRow:row];
		if (i >= 0)
			add(&spaced, now() - t0);
	}
	// End to end, against the command buffer alone.
	const int nE2e = gOps < 100 ? gOps : 100;
	for (int i = -2; i < nE2e; i++) {
		id<MTLTexture> t = pool[(i + 2) % kPool];
		const uint8_t v = (uint8_t)(i * 13 + 5);
		memset(src, v, bytes);
		const double t0 = now();
		[t replaceRegion:all mipmapLevel:0 withBytes:src bytesPerRow:row];
		blitAll(@[ t ], s, out);
		const double t1 = now();
		checked++;
		bad += !holds(out.contents, bytes, v);
		const double b = blitAll(@[ t ], s, out);
		if (i >= 0) {
			add(&e2e, t1 - t0);
			add(&base, b);
		}
	}
	free(src);
	NSString *r = [NSString stringWithFormat:@"bytes=%lu %@ %@ %@ %@ ok=%d/%d", (unsigned long)bytes,
	                                         st("call", &call), st("spaced", &spaced), st("e2e", &e2e),
	                                         st("base", &base), checked - bad, checked];
	drop(&call), drop(&spaced), drop(&e2e), drop(&base);
	return bad ? [@"FAIL " stringByAppendingString:r] : r;
}

// A new Managed texture: made, freed, and made, filled and read (mk.SHAPE).
static NSString *makeCase(const Shape *s)
{
	const NSUInteger row = s->w * s->bpp, bytes = row * s->h;
	MTLTextureDescriptor *d = managedDesc(s);
	id<MTLBuffer> out = [gDevice newBufferWithLength:bytes options:MTLResourceStorageModeShared];
	uint8_t *src = malloc(bytes);
	const MTLRegion all = MTLRegionMake2D(0, 0, s->w, s->h);
	Samples make = {0}, freed = {0}, first = {0}, base = {0};
	int bad = 0, checked = 0;
	for (int i = -4; i < gOps; i++) {
		double t0 = 0, t1 = 0;
		@autoreleasepool {
			t0 = now();
			id<MTLTexture> t = [gDevice newTextureWithDescriptor:d];
			t1 = now();
			if (t == nil)
				[NSException raise:@"upload" format:@"no Managed texture of %s", s->name];
			t0 = t1 - t0;
			t1 = now();
			t = nil;
		}
		const double t2 = now();
		if (i >= 0) {
			add(&make, t0);
			add(&freed, t2 - t1);
		}
	}
	const int n = gOps < 100 ? gOps : 100;
	id<MTLTexture> held = managedTexture(s);
	for (int i = -2; i < n; i++) {
		const uint8_t v = (uint8_t)(i * 17 + 3);
		memset(src, v, bytes);
		double e = 0;
		@autoreleasepool {
			const double t0 = now();
			id<MTLTexture> t = [gDevice newTextureWithDescriptor:d];
			[t replaceRegion:all mipmapLevel:0 withBytes:src bytesPerRow:row];
			blitAll(@[ t ], s, out);
			e = now() - t0;
		}
		checked++;
		bad += !holds(out.contents, bytes, v);
		const double b = blitAll(@[ held ], s, out);
		if (i >= 0) {
			add(&first, e);
			add(&base, b);
		}
	}
	free(src);
	NSString *r = [NSString stringWithFormat:@"bytes=%lu %@ %@ %@ %@ ok=%d/%d", (unsigned long)bytes,
	                                         st("make", &make), st("free", &freed), st("first", &first),
	                                         st("base", &base), checked - bad, checked];
	drop(&make), drop(&freed), drop(&first), drop(&base);
	return bad ? [@"FAIL " stringByAppendingString:r] : r;
}

// BURST replaceRegion: calls, then one command buffer of their blits (burst.SHAPE).
static NSString *burstCase(const Shape *s)
{
	const int k = gBurst;
	const NSUInteger row = s->w * s->bpp, bytes = row * s->h;
	NSMutableArray<id<MTLTexture>> *ts = [NSMutableArray arrayWithCapacity:k];
	for (int i = 0; i < k; i++)
		[ts addObject:managedTexture(s)];
	id<MTLBuffer> out = [gDevice newBufferWithLength:bytes * k options:MTLResourceStorageModeShared];
	uint8_t *src = malloc(bytes);
	const MTLRegion all = MTLRegionMake2D(0, 0, s->w, s->h);
	Samples calls = {0}, burst = {0}, base = {0};
	int bad = 0, checked = 0;
	const int rounds = gOps / 10 < 5 ? 5 : gOps / 10;
	for (int r = -1; r < rounds; r++) {
		const double t0 = now();
		for (int i = 0; i < k; i++) {
			memset(src, (uint8_t)(r * 7 + i), bytes);
			[ts[i] replaceRegion:all mipmapLevel:0 withBytes:src bytesPerRow:row];
		}
		const double t1 = now();
		blitAll(ts, s, out);
		const double t2 = now();
		for (int i = 0; i < k; i++) {
			checked++;
			bad += !holds((const uint8_t *)out.contents + (NSUInteger)i * bytes, bytes,
			              (uint8_t)(r * 7 + i));
		}
		const double b = blitAll(ts, s, out);
		if (r >= 0) {
			add(&calls, t1 - t0);
			add(&burst, t2 - t0);
			add(&base, b);
		}
	}
	free(src);
	NSString *line = [NSString stringWithFormat:@"burst=%d bytes=%lu %@ %@ %@ per_us=%.1f ok=%d/%d", k,
	                                            (unsigned long)bytes, st("calls", &calls), st("burst", &burst),
	                                            st("base", &base), (pct(&burst, 0.5) - pct(&base, 0.5)) / k,
	                                            checked - bad, checked];
	drop(&calls), drop(&burst), drop(&base);
	return bad ? [@"FAIL " stringByAppendingString:line] : line;
}

// A blit of `n` ranges of `size` bytes, from `first` in `src` and `stride` apart, into `out`,
// packed, committed and waited: its time.
static double copyRanges(id<MTLBuffer> src, NSUInteger first, id<MTLBuffer> out, int n, NSUInteger size,
                         NSUInteger stride)
{
	const double t0 = now();
	id<MTLCommandBuffer> cb = [gQueue commandBuffer];
	id<MTLBlitCommandEncoder> e = [cb blitCommandEncoder];
	for (int i = 0; i < n; i++)
		[e copyFromBuffer:src
		         sourceOffset:first + (NSUInteger)i * stride
		             toBuffer:out
		    destinationOffset:(NSUInteger)i * size
		                 size:size];
	[e endEncoding];
	[cb commit];
	finish(cb);
	return now() - t0;
}

// didModifyRange: of `size` bytes of a Managed buffer (dmr.SIZE).
static NSString *dmrCase(NSUInteger size)
{
	const NSUInteger length = 4u << 20, stride = size < 4096 ? 4096 : size;
	const int slots = (int)(length / stride);
	id<MTLBuffer> m = [gDevice newBufferWithLength:length options:MTLResourceStorageModeManaged];
	id<MTLBuffer> out = [gDevice newBufferWithLength:size options:MTLResourceStorageModeShared];
	uint8_t *p = m.contents;
	Samples call = {0}, spaced = {0}, e2e = {0}, base = {0};
	int bad = 0, checked = 0;
	for (int i = -8; i < gOps; i++) {
		const NSUInteger at = (NSUInteger)((i + 8) % slots) * stride;
		memset(p + at, (uint8_t)i, size);
		const double t0 = now();
		[m didModifyRange:NSMakeRange(at, size)];
		if (i >= 0)
			add(&call, now() - t0);
	}
	const int n = gOps < 100 ? gOps : 100;
	for (int i = -2; i < n; i++) {
		const NSUInteger at = (NSUInteger)((i + 2) % slots) * stride;
		usleep(2000);
		memset(p + at, (uint8_t)(i + 3), size);
		const double t0 = now();
		[m didModifyRange:NSMakeRange(at, size)];
		if (i >= 0)
			add(&spaced, now() - t0);
	}
	for (int i = -2; i < n; i++) {
		const NSUInteger at = (NSUInteger)((i + 2) % slots) * stride;
		const uint8_t v = (uint8_t)(i * 11 + 9);
		memset(p + at, v, size);
		const double t0 = now();
		[m didModifyRange:NSMakeRange(at, size)];
		copyRanges(m, at, out, 1, size, 0);
		const double t1 = now();
		checked++;
		bad += !holds(out.contents, size, v);
		// The same blit of the same range, nothing written.
		const double b = copyRanges(m, at, out, 1, size, 0);
		if (i >= 0) {
			add(&e2e, t1 - t0);
			add(&base, b);
		}
	}
	NSString *r = [NSString stringWithFormat:@"bytes=%lu %@ %@ %@ %@ ok=%d/%d", (unsigned long)size,
	                                         st("call", &call), st("spaced", &spaced), st("e2e", &e2e),
	                                         st("base", &base), checked - bad, checked];
	drop(&call), drop(&spaced), drop(&e2e), drop(&base);
	return bad ? [@"FAIL " stringByAppendingString:r] : r;
}

// BURST didModifyRange: calls of `size` bytes, then one blit of every range (dmrburst.SIZE).
static NSString *dmrBurstCase(NSUInteger size)
{
	const int k = gBurst;
	const NSUInteger stride = size < 4096 ? 4096 : size, length = stride * (NSUInteger)k;
	id<MTLBuffer> m = [gDevice newBufferWithLength:length options:MTLResourceStorageModeManaged];
	id<MTLBuffer> out = [gDevice newBufferWithLength:size * (NSUInteger)k options:MTLResourceStorageModeShared];
	uint8_t *p = m.contents;
	Samples calls = {0}, burst = {0}, base = {0};
	int bad = 0, checked = 0;
	const int rounds = gOps / 10 < 5 ? 5 : gOps / 10;
	for (int r = -1; r < rounds; r++) {
		for (int i = 0; i < k; i++)
			memset(p + (NSUInteger)i * stride, (uint8_t)(r * 5 + i), size);
		const double t0 = now();
		for (int i = 0; i < k; i++)
			[m didModifyRange:NSMakeRange((NSUInteger)i * stride, size)];
		const double t1 = now();
		copyRanges(m, 0, out, k, size, stride);
		const double t2 = now();
		for (int i = 0; i < k; i++) {
			checked++;
			bad += !holds((const uint8_t *)out.contents + (NSUInteger)i * size, size, (uint8_t)(r * 5 + i));
		}
		const double b = copyRanges(m, 0, out, k, size, stride);
		if (r >= 0) {
			add(&calls, t1 - t0);
			add(&burst, t2 - t0);
			add(&base, b);
		}
	}
	NSString *line = [NSString stringWithFormat:@"burst=%d bytes=%lu %@ %@ %@ per_us=%.1f ok=%d/%d", k,
	                                            (unsigned long)size, st("calls", &calls), st("burst", &burst),
	                                            st("base", &base), (pct(&burst, 0.5) - pct(&base, 0.5)) / k,
	                                            checked - bad, checked];
	drop(&calls), drop(&burst), drop(&base);
	return bad ? [@"FAIL " stringByAppendingString:line] : line;
}

// A linear texture of `s` over a new Managed buffer just its size, at offset 0, rows `pitch` apart
// (the device's linear alignment): by the linear SPI where the buffer answers it, else the public
// form. Nil where the device makes none.
static id<MTLTexture> linearTexture(const Shape *s, NSUInteger pitch)
{
	id<MTLBuffer> b = [gDevice newBufferWithLength:pitch * s->h options:MTLResourceStorageModeManaged];
	MTLTextureDescriptor *d = managedDesc(s);
	id<MTLTexture> t = nil;
	if ([(id)b respondsToSelector:@selector(newLinearTextureWithDescriptor:offset:bytesPerRow:bytesPerImage:)])
		t = [(id<UploadProbeLinearTexture>)b newLinearTextureWithDescriptor:d
		                                                             offset:0
		                                                        bytesPerRow:pitch
		                                                      bytesPerImage:pitch * s->h];
	else
		t = [b newTextureWithDescriptor:d offset:0 bytesPerRow:pitch];
	return t;
}

// replaceRegion: (didModifyData NO) or a CPU write and didModifyData (YES) of a linear texture over
// a Managed buffer (linrr.SHAPE, dmd.SHAPE).
static NSString *linearCase(const Shape *s, BOOL dmd)
{
	enum { kPool = 16 };
	const NSUInteger row = s->w * s->bpp, bytes = row * s->h;
	NSUInteger align = [gDevice minimumLinearTextureAlignmentForPixelFormat:s->format];
	if (align == 0)
		align = 1;
	const NSUInteger pitch = (row + align - 1) / align * align;
	NSMutableArray<id<MTLTexture>> *pool = [NSMutableArray arrayWithCapacity:kPool];
	for (int i = 0; i < kPool; i++) {
		id<MTLTexture> t = linearTexture(s, pitch);
		if (t == nil)
			return [NSString stringWithFormat:@"SKIP no linear texture over a Managed buffer (pitch %lu)",
			                                  (unsigned long)pitch];
		[pool addObject:t];
	}
	if (dmd && ![(id)pool[0] respondsToSelector:@selector(didModifyData)])
		return @"SKIP the texture does not answer didModifyData";
	id<MTLBuffer> out = [gDevice newBufferWithLength:bytes options:MTLResourceStorageModeShared];
	uint8_t *src = malloc(bytes);
	const MTLRegion all = MTLRegionMake2D(0, 0, s->w, s->h);
	// One upload of `v` into texture `i`: the step timed (a replaceRegion:, or didModifyData after
	// the CPU's write of the buffer), its time.
	double (^upload)(int, uint8_t) = ^double(int i, uint8_t v) {
		if (dmd) {
			memset(pool[i].buffer.contents, v, pitch * s->h);
			const double t0 = now();
			[(id<UploadProbeDidModifyData>)pool[i] didModifyData];
			return now() - t0;
		}
		memset(src, v, bytes);
		const double t0 = now();
		[pool[i] replaceRegion:all mipmapLevel:0 withBytes:src bytesPerRow:row];
		return now() - t0;
	};
	Samples call = {0}, spaced = {0}, e2e = {0}, base = {0};
	int bad = 0, checked = 0;
	for (int i = -8; i < gOps; i++) {
		const double t = upload((i + 8) % kPool, (uint8_t)i);
		if (i >= 0)
			add(&call, t);
	}
	const int n = gOps < 100 ? gOps : 100;
	for (int i = -2; i < n; i++) {
		usleep(2000);
		const double t = upload((i + 2) % kPool, (uint8_t)(i + 1));
		if (i >= 0)
			add(&spaced, t);
	}
	for (int i = -2; i < n; i++) {
		const int at = (i + 2) % kPool;
		const uint8_t v = (uint8_t)(i * 13 + 7);
		const double t0 = now();
		upload(at, v);
		blitAll(@[ pool[at] ], s, out);
		const double t1 = now();
		checked++;
		bad += !holds(out.contents, bytes, v);
		const double b = blitAll(@[ pool[at] ], s, out);
		if (i >= 0) {
			add(&e2e, t1 - t0);
			add(&base, b);
		}
	}
	free(src);
	NSString *r = [NSString stringWithFormat:@"bytes=%lu pitch=%lu %@ %@ %@ %@ ok=%d/%d", (unsigned long)bytes,
	                                         (unsigned long)pitch, st("call", &call), st("spaced", &spaced),
	                                         st("e2e", &e2e), st("base", &base), checked - bad, checked];
	drop(&call), drop(&spaced), drop(&e2e), drop(&base);
	return bad ? [@"FAIL " stringByAppendingString:r] : r;
}

// --- main -----------------------------------------------------------------------------------------

int main(int argc, char **argv)
{
	@autoreleasepool {
		gStart = now();
		const char *pick = NULL;
		BOOL list = NO;
		int c;
		while ((c = getopt(argc, argv, "d:n:k:l")) != -1) {
			switch (c) {
			case 'd':
				pick = optarg;
				break;
			case 'n':
				gOps = atoi(optarg);
				break;
			case 'k':
				gBurst = atoi(optarg);
				break;
			case 'l':
				list = YES;
				break;
			default:
				fprintf(stderr, "usage: upload_probe [-d NAME] [-n OPS] [-k BURST] [-l] [CASE...]\n");
				return 2;
			}
		}
		if (gOps < 1 || gBurst < 1) {
			fprintf(stderr, "upload_probe: OPS and BURST are at least 1\n");
			return 2;
		}
		NSMutableArray<NSString *> *patterns = [NSMutableArray array];
		for (int i = optind; i < argc; i++)
			[patterns addObject:@(argv[i])];
		gPatterns = patterns;
		NSArray<id<MTLDevice>> *all = MTLCopyAllDevices();
		if (list) {
			for (id<MTLDevice> d in all)
				printf("%s\n", d.name.UTF8String);
			return 0;
		}
		if (pick) {
			for (id<MTLDevice> d in all) {
				if ([d.name containsString:@(pick)]) {
					gDevice = d;
					break;
				}
			}
		} else {
			gDevice = MTLCreateSystemDefaultDevice();
		}
		if (gDevice == nil) {
			fprintf(stderr, "upload_probe: no device%s%s\n", pick ? " named " : "", pick ? pick : "");
			return 1;
		}
		gQueue = [gDevice newCommandQueue];
		printf("# device %s unified=%d ops=%d burst=%d\n", gDevice.name.UTF8String,
		       (int)gDevice.hasUnifiedMemory, gOps, gBurst);
		const size_t nShapes = sizeof kShapes / sizeof kShapes[0];
		for (size_t i = 0; i < nShapes; i++) {
			const Shape *s = &kShapes[i];
			runCase([NSString stringWithFormat:@"rr.%s", s->name], ^{
				return replaceCase(s, MTLStorageModeManaged);
			});
		}
		for (size_t i = 0; i < nShapes; i++) {
			const Shape *s = &kShapes[i];
			runCase([NSString stringWithFormat:@"mk.%s", s->name], ^{
				return makeCase(s);
			});
		}
		for (size_t i = 0; i < nShapes; i++) {
			const Shape *s = &kShapes[i];
			if (s->w * s->h > 512 * 512)
				continue;
			runCase([NSString stringWithFormat:@"burst.%s", s->name], ^{
				return burstCase(s);
			});
		}
		for (size_t i = 0; i < nShapes; i++) {
			const Shape *s = &kShapes[i];
			runCase([NSString stringWithFormat:@"srr.%s", s->name], ^{
				return replaceCase(s, MTLStorageModeShared);
			});
		}
		const NSUInteger sizes[] = {256, 4096, 65536};
		const char *sizeNames[] = {"256", "4k", "64k"};
		for (size_t i = 0; i < 3; i++) {
			const NSUInteger z = sizes[i];
			runCase([NSString stringWithFormat:@"dmr.%s", sizeNames[i]], ^{
				return dmrCase(z);
			});
		}
		for (size_t i = 0; i < 3; i++) {
			const NSUInteger z = sizes[i];
			runCase([NSString stringWithFormat:@"dmrburst.%s", sizeNames[i]], ^{
				return dmrBurstCase(z);
			});
		}
		for (size_t i = 0; i < nShapes; i++) {
			const Shape *s = &kShapes[i];
			runCase([NSString stringWithFormat:@"linrr.%s", s->name], ^{
				return linearCase(s, NO);
			});
			runCase([NSString stringWithFormat:@"dmd.%s", s->name], ^{
				return linearCase(s, YES);
			});
		}
	}
	return 0;
}
