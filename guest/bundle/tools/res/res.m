// res: what Metal resources cost on a device: making and freeing them, the CPU's access to their
// memory, and the process's memory over many make-and-free rounds. To compare a guest's GPU lane
// with real Macs, case by case. Each case warms up, then times N operations one at a time and
// prints the median and the 99th percentile of each step in microseconds (us), and rates.
//   res [-d NAME] [-n OPS] [-R ROUNDS] [-l] [CASE...]
//     -d NAME   the first of MTLCopyAllDevices whose name holds NAME; the default is
//               MTLCreateSystemDefaultDevice (on a dual-GPU MacBook that switches the display to
//               the discrete GPU: pass -d there).
//     -n OPS    operations a sample (default 200; the large sizes take fewer, as each case says).
//     -R ROUNDS the make-and-free rounds of mem.cycle (default 20).
//     -l        lists the devices and exits.
//     CASE      runs only the cases whose name holds one of these; with none, every case but
//               region.fill, which takes most of a lane's region from every other process, and,
//               on x86-64, heap.shared.16m, where Metal asserts at a Shared heap (measured on the
//               AMD Radeon Pro 5300M and the Intel UHD 630 under macOS 26.7.1). -CASE skips the
//               cases whose name holds CASE.
// The cases (sizes in bytes: 4k = 4096, 1m = 2^20):
//   buf.MODE.SIZE         newBufferWithLength:options: and its release, MODE private, shared or
//               managed: make_us, free_us, pairs_s (make and free a second), and use_us, a make
//               then a command buffer that fills 4 KiB of it, committed and waited, against cb_us,
//               the same command buffer over a buffer made before.
//   buf.bytes.SIZE        newBufferWithBytes:length:options: (Shared) from malloc'd bytes.
//   buf.nocopy.SIZE       newBufferWithBytesNoCopy: (Shared) over vm_allocate'd pages, no
//               deallocator.
//   buf.hold.shared.4k    4,000 Shared buffers made and held: make_us of the first and the last
//               500, then each freed (free_us), then 4,000 made again at once (nil = how many
//               came back nil).
//   buf.recycle.shared.64m  30 rounds of 8 Shared 64 MiB buffers made, one byte a page written,
//               and all freed: round_ms, and nil, the buffers that came back nil.
//   tex.*                 newTextureWithDescriptor: and its release: private.{256,1024,4096}
//               RGBA8, rt.4096 (RGBA8 render target, use_us = make, clear, wait), managed.1024,
//               depth.2048 (Depth32Float target), mips.2048 (BGRA8, every level), linear.1024
//               (over a Shared buffer, newTextureWithDescriptor:offset:bytesPerRow:), view.1024
//               (newTextureViewWithPixelFormat:).
//   ios.create.SIZE       IOSurfaceCreate (BGRA, SIZE x SIZE) and CFRelease, no Metal.
//   ios.tex.SIZE          a texture over a held IOSurface, made and freed.
//   ios.fresh.1024        a new IOSurface, a texture over it, a clear of it, waited, both freed.
//   ios.lock.1024         IOSurfaceLock/Unlock with no GPU work between: lock_us, rlock_us
//               (read-only), unlock_us; use_us, IOSurfaceIncrement/DecrementUseCount.
//   ios.lock.gpu.1024     the GPU clears the surface's texture, waited; then a read-only lock:
//               lock_us, and ok, the locks whose first pixel read the clear's colour.
//   ios.upload.1024       lock, write every byte, unlock, then the GPU copies the texture into a
//               Shared buffer, waited: e2e_us and mbps, and ok as above.
//   ios.inuse.1024        IOSurfaceIsInUse right after a commit that writes the surface, and once
//               it completed: busy, idle (how many of the tries answered true).
//   heap.MODE.SIZE        newHeapWithDescriptor: and its release.
//   heap.place.buf.64k, heap.place.tex.1024  resources made in a held 64 MiB Private heap.
//   heap.query            heapTextureSizeAndAlignWithDescriptor: and heapBufferSizeAndAlignWithLength:.
//   sampler.same, sampler.distinct, sampler.argbuf  newSamplerStateWithDescriptor: of one
//               descriptor (same_obj: how often Metal answered the last call's object, still held), of a new
//               descriptor each call, and of new ones that support argument buffers.
//   depthstencil          newDepthStencilStateWithDescriptor:.
//   argbuf.encoder, argbuf.set, argbuf.resid  an argument encoder of 32 textures and 32 buffers
//               made and freed; its setTexture:/setBuffer: calls a second; gpuResourceID reads.
//   cpu.shared.64m        a new Shared buffer's first write (memset), a second write, a read:
//               GB/s each.
//   cpu.managed.SIZE      a Managed buffer written whole, didModifyRange: (dmr_us), then a blit
//               of it to a Private buffer, waited (e2e_us).
//   cpu.sync.SIZE         the GPU fills a Managed buffer, synchronizeResource:, waited, then the
//               CPU reads it whole and checks it (e2e_us).
//   tex.replace.SIZE      replaceRegion: of a whole Managed RGBA8 texture (us, gbps), then a blit
//               of it into a buffer, waited, and checked (e2e_us).
//   tex.getbytes.SIZE     the GPU writes a whole Managed RGBA8 texture, synchronizeTexture:,
//               waited, then getBytes: (us, gbps), checked.
//   mem.cycle             ROUNDS rounds of a mix (Shared, Private and Managed buffers, textures,
//               heaps with placed resources, samplers, IOSurfaces with textures), each used by a
//               command buffer, waited, then all freed: the process's footprint and resident
//               size after each round ("# mem" lines) and the growth a round.
//   region.fill           1 MiB Shared buffers until nil or 1.5 GiB, every other one freed, then
//               one of 64 and one of 256 MiB; then all freed and one of 512 MiB.
// One line a case: "res: NAME OK key=value ...", or "res: NAME FAIL why" / "SKIP why"; "# start
// NAME" before it with the time since the process started and the Unix time, so a host's samples
// line up with the cases. Build: make res (build/res-x86_64); make res ARCH=arm64 for an
// Apple-silicon Mac.
#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <Metal/Metal.h>

#include <mach/mach.h>
#include <mach/mach_time.h>
#include <sys/sysctl.h>
#include <sys/time.h>

#pragma clang diagnostic ignored "-Wunguarded-availability-new"

static id<MTLDevice> gDevice;
static id<MTLCommandQueue> gQueue;
static NSArray<NSString *> *gPatterns;
static int gOps = 200;
static int gRounds = 20;
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

static double sum(const Samples *s)
{
	double t = 0;
	for (size_t i = 0; i < s->n; i++)
		t += s->v[i];
	return t;
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
		[NSException raise:@"res" format:@"command buffer status %ld: %@", (long)cb.status,
		                                 cb.error.localizedDescription ?: @"no error"];
}

static BOOL wanted(NSString *name, BOOL byDefault)
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
	return any ? picked : byDefault;
}

static void runCase(NSString *name, BOOL byDefault, NSString * (^body)(void))
{
	if (!wanted(name, byDefault))
		return;
	const double t0 = now();
	printf("# start %s t=%.3f unix=%.3f\n", name.UTF8String, t0 - gStart, unixNow());
	fflush(stdout);
	NSString *line = nil;
	@try {
		@autoreleasepool {
			line = body();
		}
	} @catch (NSException *e) {
		line = [@"FAIL " stringByAppendingString:e.reason ?: e.name];
	}
	const BOOL tagged = [line hasPrefix:@"FAIL"] || [line hasPrefix:@"SKIP"];
	printf("res: %s %s%s (%.1f s)\n", name.UTF8String, tagged ? "" : "OK ", line.UTF8String, now() - t0);
	fflush(stdout);
}

static uint64_t parseSize(NSString *s)
{
	if ([s hasSuffix:@"k"])
		return (uint64_t)[s longLongValue] << 10;
	if ([s hasSuffix:@"m"])
		return (uint64_t)[s longLongValue] << 20;
	return (uint64_t)[s longLongValue];
}

// Fewer operations for large sizes, so a sample moves at most about 2 GiB.
static int opsFor(uint64_t bytes)
{
	const uint64_t cap = (2ull << 30) / (bytes ? bytes : 1);
	int n = gOps;
	if ((uint64_t)n > cap)
		n = (int)cap;
	return n < 8 ? 8 : n;
}

static double footprintMiB(double *residentMiB)
{
	task_vm_info_data_t info = {0};
	mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
	if (task_info(mach_task_self(), TASK_VM_INFO, (task_info_t)&info, &count) != KERN_SUCCESS)
		return NAN;
	if (residentMiB)
		*residentMiB = info.resident_size / 1048576.0;
	return info.phys_footprint / 1048576.0;
}

// A command buffer that fills the first 4 KiB (or the length) of `b`, committed and waited.
static void fillWait(id<MTLBuffer> b)
{
	id<MTLCommandBuffer> cb = [gQueue commandBuffer];
	id<MTLBlitCommandEncoder> e = [cb blitCommandEncoder];
	[e fillBuffer:b range:NSMakeRange(0, b.length < 4096 ? b.length : 4096) value:0x5a];
	[e endEncoding];
	[cb commit];
	finish(cb);
}

static void clearWait(id<MTLTexture> t, MTLClearColor c)
{
	MTLRenderPassDescriptor *p = [MTLRenderPassDescriptor renderPassDescriptor];
	if (t.pixelFormat == MTLPixelFormatDepth32Float) {
		p.depthAttachment.texture = t;
		p.depthAttachment.loadAction = MTLLoadActionClear;
		p.depthAttachment.storeAction = MTLStoreActionStore;
		p.depthAttachment.clearDepth = 0.5;
	} else {
		p.colorAttachments[0].texture = t;
		p.colorAttachments[0].loadAction = MTLLoadActionClear;
		p.colorAttachments[0].storeAction = MTLStoreActionStore;
		p.colorAttachments[0].clearColor = c;
	}
	id<MTLCommandBuffer> cb = [gQueue commandBuffer];
	[[cb renderCommandEncoderWithDescriptor:p] endEncoding];
	[cb commit];
	finish(cb);
}

// --- buffers ------------------------------------------------------------------------------------

typedef enum { kLength, kBytes, kNoCopy } BufKind;

static NSString *bufCase(MTLResourceOptions options, uint64_t length, BufKind kind)
{
	const int n = opsFor(length);
	const size_t page = (size_t)getpagesize();
	const size_t mapped = (length + page - 1) & ~(page - 1);
	void *src = NULL;
	if (kind == kBytes) {
		src = malloc(length);
		memset(src, 0x33, length);
	}
	Samples make = {0}, freeing = {0}, use = {0}, base = {0};
	int nils = 0;
	for (int i = -2; i < n; i++) {
		void *pages = NULL;
		if (kind == kNoCopy) {
			vm_address_t a = 0;
			if (vm_allocate(mach_task_self(), &a, mapped, VM_FLAGS_ANYWHERE) != KERN_SUCCESS)
				[NSException raise:@"res" format:@"vm_allocate %zu", mapped];
			pages = (void *)a;
		}
		@autoreleasepool {
			const double t0 = now();
			id<MTLBuffer> b = kind == kLength ? [gDevice newBufferWithLength:length options:options]
			                  : kind == kBytes ? [gDevice newBufferWithBytes:src length:length options:options]
			                                   : [gDevice newBufferWithBytesNoCopy:pages
			                                                               length:mapped
			                                                              options:options
			                                                          deallocator:nil];
			const double t1 = now();
			if (b == nil)
				nils++;
			const double t2 = now();
			b = nil;
			const double t3 = now();
			if (i >= 0) {
				add(&make, t1 - t0);
				add(&freeing, t3 - t2);
			}
		}
		if (pages)
			vm_deallocate(mach_task_self(), (vm_address_t)pages, mapped);
	}
	if (nils)
		return [NSString stringWithFormat:@"FAIL %d of %d buffers came back nil", nils, n + 2];
	// A make then its first use, against the same use of a buffer made before.
	const int m = n < 40 ? n : 40;
	@autoreleasepool {
		id<MTLBuffer> held = kind == kNoCopy ? nil : [gDevice newBufferWithLength:length options:options];
		for (int i = -1; i < m && held != nil; i++) {
			const double t0 = now();
			fillWait(held);
			if (i >= 0)
				add(&base, now() - t0);
		}
		for (int i = -1; i < m && kind == kLength; i++) {
			@autoreleasepool {
				const double t0 = now();
				id<MTLBuffer> b = [gDevice newBufferWithLength:length options:options];
				fillWait(b);
				const double t1 = now();
				b = nil;
				if (i >= 0)
					add(&use, t1 - t0);
			}
		}
	}
	free(src);
	NSString *r = [NSString stringWithFormat:@"%@ %@ pairs_s=%.0f", st("make", &make), st("free", &freeing),
	                                         n / ((sum(&make) + sum(&freeing)) / 1e6)];
	if (use.n)
		r = [r stringByAppendingFormat:@" %@ cb_us=%.1f", st("use", &use), pct(&base, 0.5)];
	drop(&make), drop(&freeing), drop(&use), drop(&base);
	return r;
}

static NSString *bufHold(void)
{
	const int n = 4000;
	NSMutableArray *held = [NSMutableArray arrayWithCapacity:n];
	Samples first = {0}, last = {0}, freeing = {0};
	for (int i = 0; i < n; i++) {
		const double t0 = now();
		id<MTLBuffer> b = [gDevice newBufferWithLength:4096 options:MTLResourceStorageModeShared];
		const double t1 = now();
		if (b == nil)
			return [NSString stringWithFormat:@"FAIL buffer %d of %d came back nil", i, n];
		[held addObject:b];
		if (i < 500)
			add(&first, t1 - t0);
		else if (i >= n - 500)
			add(&last, t1 - t0);
	}
	const double r0 = now();
	while (held.count) {
		@autoreleasepool {
			const double t0 = now();
			[held removeLastObject];
			add(&freeing, now() - t0);
		}
	}
	const double releaseAll = now() - r0;
	int nils = 0;
	const double a0 = now();
	for (int i = 0; i < n; i++) {
		id<MTLBuffer> b = [gDevice newBufferWithLength:4096 options:MTLResourceStorageModeShared];
		if (b == nil)
			nils++;
		else
			[held addObject:b];
	}
	const double again = now() - a0;
	[held removeAllObjects];
	NSString *r = [NSString stringWithFormat:@"%@ %@ %@ free_all_ms=%.1f again_ms=%.1f nil=%d", st("make_first", &first),
	                                         st("make_last", &last), st("free", &freeing), releaseAll * 1e3,
	                                         again * 1e3, nils];
	drop(&first), drop(&last), drop(&freeing);
	return r;
}

static NSString *bufRecycle(void)
{
	const int rounds = 30, per = 8;
	const uint64_t len = 64ull << 20;
	const size_t page = (size_t)getpagesize();
	Samples round = {0};
	int nils = 0, firstNils = 0;
	for (int r = 0; r < rounds; r++) {
		@autoreleasepool {
			const double t0 = now();
			NSMutableArray *held = [NSMutableArray array];
			for (int i = 0; i < per; i++) {
				id<MTLBuffer> b = [gDevice newBufferWithLength:len options:MTLResourceStorageModeShared];
				if (b == nil) {
					nils++;
					if (r == 0)
						firstNils++;
					continue;
				}
				volatile uint8_t *p = b.contents;
				for (uint64_t o = 0; o < len; o += page)
					p[o] = (uint8_t)o;
				[held addObject:b];
			}
			[held removeAllObjects];
			add(&round, now() - t0);
		}
	}
	NSString *s = [NSString stringWithFormat:@"round_ms=%.1f round_max_ms=%.1f nil=%d first_round_nil=%d of %d",
	                                         pct(&round, 0.5) / 1e3, pct(&round, 1.0) / 1e3, nils, firstNils,
	                                         rounds * per];
	drop(&round);
	return s;
}

// --- textures -----------------------------------------------------------------------------------

static MTLTextureDescriptor *tex2d(MTLPixelFormat f, NSUInteger w, MTLStorageMode mode, MTLTextureUsage usage,
                                   BOOL mips)
{
	MTLTextureDescriptor *d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:f width:w height:w mipmapped:mips];
	d.storageMode = mode;
	d.usage = usage;
	return d;
}

static NSString *texCase(MTLTextureDescriptor *d, BOOL useClear)
{
	const uint64_t bytes = (uint64_t)d.width * d.height * 4;
	const int n = opsFor(bytes);
	Samples make = {0}, freeing = {0}, use = {0};
	for (int i = -2; i < n; i++) {
		@autoreleasepool {
			const double t0 = now();
			id<MTLTexture> t = [gDevice newTextureWithDescriptor:d];
			const double t1 = now();
			if (t == nil)
				return @"FAIL a texture came back nil";
			const double t2 = now();
			t = nil;
			const double t3 = now();
			if (i >= 0) {
				add(&make, t1 - t0);
				add(&freeing, t3 - t2);
			}
		}
	}
	for (int i = -1; i < 30 && useClear; i++) {
		@autoreleasepool {
			const double t0 = now();
			id<MTLTexture> t = [gDevice newTextureWithDescriptor:d];
			clearWait(t, MTLClearColorMake(0.25, 0.5, 0.75, 1));
			const double t1 = now();
			t = nil;
			if (i >= 0)
				add(&use, t1 - t0);
		}
	}
	NSString *r = [NSString stringWithFormat:@"%@ %@ pairs_s=%.0f", st("make", &make), st("free", &freeing),
	                                         n / ((sum(&make) + sum(&freeing)) / 1e6)];
	if (use.n)
		r = [r stringByAppendingFormat:@" %@", st("use", &use)];
	drop(&make), drop(&freeing), drop(&use);
	return r;
}

static NSString *texLinear(NSUInteger w)
{
	const NSUInteger align = [gDevice minimumLinearTextureAlignmentForPixelFormat:MTLPixelFormatRGBA8Unorm];
	const NSUInteger row = (w * 4 + align - 1) / align * align;
	id<MTLBuffer> b = [gDevice newBufferWithLength:row * w options:MTLResourceStorageModeShared];
	MTLTextureDescriptor *d = tex2d(MTLPixelFormatRGBA8Unorm, w, MTLStorageModeShared, MTLTextureUsageShaderRead, NO);
	d.resourceOptions = MTLResourceStorageModeShared;
	Samples make = {0}, freeing = {0};
	for (int i = -2; i < gOps; i++) {
		@autoreleasepool {
			const double t0 = now();
			id<MTLTexture> t = [b newTextureWithDescriptor:d offset:0 bytesPerRow:row];
			const double t1 = now();
			if (t == nil)
				return @"FAIL a linear texture came back nil";
			const double t2 = now();
			t = nil;
			const double t3 = now();
			if (i >= 0) {
				add(&make, t1 - t0);
				add(&freeing, t3 - t2);
			}
		}
	}
	NSString *r = [NSString stringWithFormat:@"%@ %@", st("make", &make), st("free", &freeing)];
	drop(&make), drop(&freeing);
	return r;
}

static NSString *texView(NSUInteger w)
{
	MTLTextureDescriptor *d = tex2d(MTLPixelFormatRGBA8Unorm, w, MTLStorageModePrivate,
	                                MTLTextureUsageShaderRead | MTLTextureUsagePixelFormatView, NO);
	id<MTLTexture> base = [gDevice newTextureWithDescriptor:d];
	Samples make = {0}, freeing = {0};
	for (int i = -2; i < gOps; i++) {
		@autoreleasepool {
			const double t0 = now();
			id<MTLTexture> t = [base newTextureViewWithPixelFormat:MTLPixelFormatRGBA8Unorm_sRGB];
			const double t1 = now();
			if (t == nil)
				return @"FAIL a view came back nil";
			const double t2 = now();
			t = nil;
			const double t3 = now();
			if (i >= 0) {
				add(&make, t1 - t0);
				add(&freeing, t3 - t2);
			}
		}
	}
	NSString *r = [NSString stringWithFormat:@"%@ %@", st("make", &make), st("free", &freeing)];
	drop(&make), drop(&freeing);
	return r;
}

// --- IOSurfaces ---------------------------------------------------------------------------------

static IOSurfaceRef makeSurface(NSUInteger w)
{
	const size_t row = IOSurfaceAlignProperty(kIOSurfaceBytesPerRow, w * 4);
	NSDictionary *p = @{
		(id)kIOSurfaceWidth : @(w),
		(id)kIOSurfaceHeight : @(w),
		(id)kIOSurfaceBytesPerElement : @4,
		(id)kIOSurfaceBytesPerRow : @(row),
		(id)kIOSurfacePixelFormat : @((uint32_t)'BGRA'),
	};
	return IOSurfaceCreate((__bridge CFDictionaryRef)p);
}

static MTLTextureDescriptor *surfaceDesc(NSUInteger w)
{
	MTLTextureDescriptor *d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
	                                                                             width:w
	                                                                            height:w
	                                                                         mipmapped:NO];
	d.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite | MTLTextureUsageRenderTarget;
	d.storageMode = gDevice.hasUnifiedMemory ? MTLStorageModeShared : MTLStorageModeManaged;
	return d;
}

static NSString *iosCreate(NSUInteger w)
{
	Samples make = {0}, freeing = {0};
	for (int i = -2; i < gOps; i++) {
		const double t0 = now();
		IOSurfaceRef s = makeSurface(w);
		const double t1 = now();
		if (s == NULL)
			return @"FAIL IOSurfaceCreate";
		const double t2 = now();
		CFRelease(s);
		const double t3 = now();
		if (i >= 0) {
			add(&make, t1 - t0);
			add(&freeing, t3 - t2);
		}
	}
	NSString *r = [NSString stringWithFormat:@"%@ %@", st("make", &make), st("free", &freeing)];
	drop(&make), drop(&freeing);
	return r;
}

static NSString *iosTex(NSUInteger w)
{
	IOSurfaceRef s = makeSurface(w);
	MTLTextureDescriptor *d = surfaceDesc(w);
	Samples make = {0}, freeing = {0};
	const int n = opsFor((uint64_t)w * w * 4);
	for (int i = -2; i < n; i++) {
		@autoreleasepool {
			const double t0 = now();
			id<MTLTexture> t = [gDevice newTextureWithDescriptor:d iosurface:s plane:0];
			const double t1 = now();
			if (t == nil) {
				CFRelease(s);
				return @"FAIL a texture over the surface came back nil";
			}
			const double t2 = now();
			t = nil;
			const double t3 = now();
			if (i >= 0) {
				add(&make, t1 - t0);
				add(&freeing, t3 - t2);
			}
		}
	}
	CFRelease(s);
	NSString *r = [NSString stringWithFormat:@"%@ %@", st("make", &make), st("free", &freeing)];
	drop(&make), drop(&freeing);
	return r;
}

static NSString *iosFresh(NSUInteger w)
{
	MTLTextureDescriptor *d = surfaceDesc(w);
	Samples e2e = {0};
	for (int i = -2; i < 60; i++) {
		@autoreleasepool {
			const double t0 = now();
			IOSurfaceRef s = makeSurface(w);
			id<MTLTexture> t = [gDevice newTextureWithDescriptor:d iosurface:s plane:0];
			if (t == nil) {
				CFRelease(s);
				return @"FAIL a texture over the surface came back nil";
			}
			clearWait(t, MTLClearColorMake(0, 0, 1, 1));
			t = nil;
			CFRelease(s);
			if (i >= 0)
				add(&e2e, now() - t0);
		}
	}
	NSString *r = st("e2e", &e2e);
	drop(&e2e);
	return r;
}

static NSString *iosLock(NSUInteger w)
{
	IOSurfaceRef s = makeSurface(w);
	Samples lock = {0}, rlock = {0}, unlock = {0}, use = {0};
	for (int i = -2; i < gOps; i++) {
		double t0 = now();
		IOSurfaceLock(s, 0, NULL);
		double t1 = now();
		((volatile uint8_t *)IOSurfaceGetBaseAddress(s))[0] = (uint8_t)i;
		double t2 = now();
		IOSurfaceUnlock(s, 0, NULL);
		double t3 = now();
		if (i >= 0) {
			add(&lock, t1 - t0);
			add(&unlock, t3 - t2);
		}
		t0 = now();
		IOSurfaceLock(s, kIOSurfaceLockReadOnly, NULL);
		t1 = now();
		IOSurfaceUnlock(s, kIOSurfaceLockReadOnly, NULL);
		t2 = now();
		IOSurfaceIncrementUseCount(s);
		IOSurfaceDecrementUseCount(s);
		t3 = now();
		if (i >= 0) {
			add(&rlock, t1 - t0);
			add(&use, t3 - t2);
		}
	}
	CFRelease(s);
	NSString *r = [NSString stringWithFormat:@"%@ %@ %@ %@", st("lock", &lock), st("rlock", &rlock), st("unlock", &unlock),
	                                         st("use", &use)];
	drop(&lock), drop(&rlock), drop(&unlock), drop(&use);
	return r;
}

static NSString *iosLockGpu(NSUInteger w)
{
	IOSurfaceRef s = makeSurface(w);
	id<MTLTexture> t = [gDevice newTextureWithDescriptor:surfaceDesc(w) iosurface:s plane:0];
	if (t == nil) {
		CFRelease(s);
		return @"FAIL a texture over the surface came back nil";
	}
	Samples lock = {0}, gpu = {0};
	int ok = 0, n = 0;
	for (int i = -2; i < 100; i++) {
		const double c = (i & 1) ? 1.0 : 0.0; // blue then black: BGRA's first byte
		double t0 = now();
		clearWait(t, MTLClearColorMake(0, 0, c, 1));
		double t1 = now();
		IOSurfaceLock(s, kIOSurfaceLockReadOnly, NULL);
		double t2 = now();
		const uint8_t b = ((volatile uint8_t *)IOSurfaceGetBaseAddress(s))[0];
		IOSurfaceUnlock(s, kIOSurfaceLockReadOnly, NULL);
		if (i >= 0) {
			add(&gpu, t1 - t0);
			add(&lock, t2 - t1);
			n++;
			ok += b == (c > 0 ? 255 : 0);
		}
	}
	t = nil;
	CFRelease(s);
	NSString *r = [NSString stringWithFormat:@"%@ %@ ok=%d of %d", st("lock", &lock), st("clear", &gpu), ok, n];
	drop(&lock), drop(&gpu);
	return r;
}

static NSString *iosUpload(NSUInteger w)
{
	IOSurfaceRef s = makeSurface(w);
	id<MTLTexture> t = [gDevice newTextureWithDescriptor:surfaceDesc(w) iosurface:s plane:0];
	const size_t row = IOSurfaceGetBytesPerRow(s), bytes = row * w;
	id<MTLBuffer> out = [gDevice newBufferWithLength:w * w * 4 options:MTLResourceStorageModeShared];
	if (t == nil || out == nil) {
		CFRelease(s);
		return @"FAIL no texture over the surface, or no buffer";
	}
	Samples e2e = {0}, cpu = {0};
	int ok = 0, n = 0;
	for (int i = -2; i < 60; i++) {
		const uint8_t v = (uint8_t)(i * 37 + 1);
		const double t0 = now();
		IOSurfaceLock(s, 0, NULL);
		memset(IOSurfaceGetBaseAddress(s), v, bytes);
		IOSurfaceUnlock(s, 0, NULL);
		const double t1 = now();
		id<MTLCommandBuffer> cb = [gQueue commandBuffer];
		id<MTLBlitCommandEncoder> e = [cb blitCommandEncoder];
		[e copyFromTexture:t
		         sourceSlice:0
		         sourceLevel:0
		        sourceOrigin:MTLOriginMake(0, 0, 0)
		          sourceSize:MTLSizeMake(w, w, 1)
		            toBuffer:out
		   destinationOffset:0
		  destinationBytesPerRow:w * 4
		destinationBytesPerImage:w * w * 4];
		[e endEncoding];
		[cb commit];
		finish(cb);
		const double t2 = now();
		const uint8_t *o = out.contents;
		if (i >= 0) {
			add(&e2e, t2 - t0);
			add(&cpu, t1 - t0);
			n++;
			ok += o[0] == v && o[w * w * 4 - 1] == v;
		}
	}
	t = nil;
	CFRelease(s);
	NSString *r = [NSString stringWithFormat:@"%@ %@ mbps=%.0f ok=%d of %d", st("e2e", &e2e), st("cpu", &cpu),
	                                         bytes / pct(&e2e, 0.5), ok, n];
	drop(&e2e), drop(&cpu);
	return r;
}

static NSString *iosInUse(NSUInteger w)
{
	IOSurfaceRef s = makeSurface(w);
	id<MTLTexture> t = [gDevice newTextureWithDescriptor:surfaceDesc(w) iosurface:s plane:0];
	if (t == nil) {
		CFRelease(s);
		return @"FAIL a texture over the surface came back nil";
	}
	int busy = 0, idle = 0, inflight = 0;
	const int n = 40;
	for (int i = 0; i < n; i++) {
		MTLRenderPassDescriptor *p = [MTLRenderPassDescriptor renderPassDescriptor];
		p.colorAttachments[0].texture = t;
		p.colorAttachments[0].loadAction = MTLLoadActionClear;
		p.colorAttachments[0].storeAction = MTLStoreActionStore;
		id<MTLCommandBuffer> cb = [gQueue commandBuffer];
		// Several passes, so the buffer is still running when the check comes.
		for (int k = 0; k < 16; k++)
			[[cb renderCommandEncoderWithDescriptor:p] endEncoding];
		[cb commit];
		const BOOL during = IOSurfaceIsInUse(s);
		inflight += cb.status < MTLCommandBufferStatusCompleted;
		finish(cb);
		busy += during;
		idle += IOSurfaceIsInUse(s);
	}
	t = nil;
	CFRelease(s);
	return [NSString stringWithFormat:@"busy=%d idle=%d inflight=%d of %d", busy, idle, inflight, n];
}

// --- heaps --------------------------------------------------------------------------------------

static NSString *heapCase(MTLStorageMode mode, uint64_t size)
{
	MTLHeapDescriptor *d = [MTLHeapDescriptor new];
	d.size = size;
	d.storageMode = mode;
	const int n = opsFor(size);
	Samples make = {0}, freeing = {0};
	for (int i = -2; i < n; i++) {
		@autoreleasepool {
			const double t0 = now();
			id<MTLHeap> h = [gDevice newHeapWithDescriptor:d];
			const double t1 = now();
			if (h == nil)
				return i < 0 ? @"SKIP the device makes no such heap" : @"FAIL a heap came back nil";
			const double t2 = now();
			h = nil;
			const double t3 = now();
			if (i >= 0) {
				add(&make, t1 - t0);
				add(&freeing, t3 - t2);
			}
		}
	}
	NSString *r = [NSString stringWithFormat:@"%@ %@", st("make", &make), st("free", &freeing)];
	drop(&make), drop(&freeing);
	return r;
}

static NSString *heapPlace(BOOL textures)
{
	MTLHeapDescriptor *d = [MTLHeapDescriptor new];
	d.size = 64u << 20;
	d.storageMode = MTLStorageModePrivate;
	id<MTLHeap> h = [gDevice newHeapWithDescriptor:d];
	if (h == nil)
		return @"FAIL no heap";
	MTLTextureDescriptor *td = tex2d(MTLPixelFormatRGBA8Unorm, 1024, MTLStorageModePrivate,
	                                 MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget, NO);
	Samples make = {0}, freeing = {0};
	for (int i = -2; i < gOps; i++) {
		@autoreleasepool {
			const double t0 = now();
			id<MTLResource> r = textures ? (id<MTLResource>)[h newTextureWithDescriptor:td]
			                             : (id<MTLResource>)[h newBufferWithLength:65536 options:MTLResourceStorageModePrivate];
			const double t1 = now();
			if (r == nil)
				return @"FAIL a resource of the heap came back nil";
			const double t2 = now();
			r = nil;
			const double t3 = now();
			if (i >= 0) {
				add(&make, t1 - t0);
				add(&freeing, t3 - t2);
			}
		}
	}
	NSString *r = [NSString stringWithFormat:@"%@ %@ used_after=%lu", st("make", &make), st("free", &freeing),
	                                         (unsigned long)h.usedSize];
	drop(&make), drop(&freeing);
	return r;
}

static NSString *heapQuery(void)
{
	MTLTextureDescriptor *td = tex2d(MTLPixelFormatRGBA8Unorm, 1024, MTLStorageModePrivate, MTLTextureUsageShaderRead, NO);
	Samples tex = {0}, buf = {0};
	for (int i = -2; i < gOps; i++) {
		td.width = 1024 + (NSUInteger)(i & 7);
		double t0 = now();
		MTLSizeAndAlign a = [gDevice heapTextureSizeAndAlignWithDescriptor:td];
		double t1 = now();
		MTLSizeAndAlign b = [gDevice heapBufferSizeAndAlignWithLength:65536 + (NSUInteger)i options:MTLResourceStorageModePrivate];
		double t2 = now();
		if (a.size == 0 || b.size == 0)
			return @"FAIL a size of 0";
		if (i >= 0) {
			add(&tex, t1 - t0);
			add(&buf, t2 - t1);
		}
	}
	NSString *r = [NSString stringWithFormat:@"%@ %@", st("tex", &tex), st("buf", &buf)];
	drop(&tex), drop(&buf);
	return r;
}

// --- state objects ------------------------------------------------------------------------------

static NSString *samplerCase(int kind)
{
	Samples make = {0}, freeing = {0};
	int same = 0;
	// The last call's sampler stays alive through the next call, so an equal answer is Metal's
	// own sharing, not a freed object's address taken again.
	id<MTLSamplerState> last = nil;
	for (int i = -2; i < gOps; i++) {
		@autoreleasepool {
			MTLSamplerDescriptor *d = [MTLSamplerDescriptor new];
			d.minFilter = MTLSamplerMinMagFilterLinear;
			d.magFilter = MTLSamplerMinMagFilterLinear;
			if (kind > 0)
				d.lodMaxClamp = 1000.0f - (float)(i + 2) * 0.25f;
			if (kind == 2)
				d.supportArgumentBuffers = YES;
			const double t0 = now();
			id<MTLSamplerState> s = [gDevice newSamplerStateWithDescriptor:d];
			const double t1 = now();
			if (s == nil)
				return @"FAIL a sampler came back nil";
			same += s == last;
			const double t2 = now();
			last = nil;
			const double t3 = now();
			last = s;
			s = nil;
			if (i >= 0) {
				add(&make, t1 - t0);
				add(&freeing, t3 - t2);
			}
		}
	}
	last = nil;
	NSString *r = [NSString stringWithFormat:@"%@ %@ same_obj=%d", st("make", &make), st("free", &freeing), same];
	drop(&make), drop(&freeing);
	return r;
}

static NSString *depthStencil(void)
{
	Samples make = {0}, freeing = {0};
	for (int i = -2; i < gOps; i++) {
		@autoreleasepool {
			MTLDepthStencilDescriptor *d = [MTLDepthStencilDescriptor new];
			d.depthCompareFunction = (MTLCompareFunction)(1 + ((i + 2) & 7) % 7);
			d.depthWriteEnabled = (i & 8) != 0;
			const double t0 = now();
			id<MTLDepthStencilState> s = [gDevice newDepthStencilStateWithDescriptor:d];
			const double t1 = now();
			if (s == nil)
				return @"FAIL a depth-stencil state came back nil";
			const double t2 = now();
			s = nil;
			const double t3 = now();
			if (i >= 0) {
				add(&make, t1 - t0);
				add(&freeing, t3 - t2);
			}
		}
	}
	NSString *r = [NSString stringWithFormat:@"%@ %@", st("make", &make), st("free", &freeing)];
	drop(&make), drop(&freeing);
	return r;
}

static NSArray<MTLArgumentDescriptor *> *argDescriptors(void)
{
	NSMutableArray *a = [NSMutableArray array];
	for (NSUInteger i = 0; i < 64; i++) {
		MTLArgumentDescriptor *d = [MTLArgumentDescriptor argumentDescriptor];
		d.index = i;
		d.dataType = i < 32 ? MTLDataTypeTexture : MTLDataTypePointer;
		if (i < 32)
			d.textureType = MTLTextureType2D;
		d.access = MTLBindingAccessReadOnly;
		[a addObject:d];
	}
	return a;
}

static NSString *argEncoder(void)
{
	NSArray *args = argDescriptors();
	Samples make = {0}, freeing = {0};
	for (int i = -2; i < gOps; i++) {
		@autoreleasepool {
			const double t0 = now();
			id<MTLArgumentEncoder> e = [gDevice newArgumentEncoderWithArguments:args];
			const double t1 = now();
			if (e == nil)
				return @"FAIL no argument encoder";
			const double t2 = now();
			e = nil;
			const double t3 = now();
			if (i >= 0) {
				add(&make, t1 - t0);
				add(&freeing, t3 - t2);
			}
		}
	}
	NSString *r = [NSString stringWithFormat:@"%@ %@", st("make", &make), st("free", &freeing)];
	drop(&make), drop(&freeing);
	return r;
}

static NSString *argSet(void)
{
	id<MTLArgumentEncoder> e = [gDevice newArgumentEncoderWithArguments:argDescriptors()];
	id<MTLBuffer> ab = [gDevice newBufferWithLength:e.encodedLength options:MTLResourceStorageModeShared];
	MTLTextureDescriptor *td = tex2d(MTLPixelFormatRGBA8Unorm, 64, MTLStorageModePrivate, MTLTextureUsageShaderRead, NO);
	NSMutableArray *texes = [NSMutableArray array], *bufs = [NSMutableArray array];
	for (int i = 0; i < 32; i++) {
		[texes addObject:[gDevice newTextureWithDescriptor:td]];
		[bufs addObject:[gDevice newBufferWithLength:256 options:MTLResourceStorageModePrivate]];
	}
	[e setArgumentBuffer:ab offset:0];
	const int passes = 2000;
	const double t0 = now();
	for (int p = 0; p < passes; p++) {
		for (NSUInteger i = 0; i < 32; i++) {
			[e setTexture:texes[(i + p) & 31] atIndex:i];
			[e setBuffer:bufs[(i + p) & 31] offset:0 atIndex:32 + i];
		}
	}
	const double el = now() - t0;
	return [NSString stringWithFormat:@"set_ns=%.1f sets_s=%.0f encoded_length=%lu", el * 1e9 / (passes * 64),
	                                  passes * 64 / el, (unsigned long)e.encodedLength];
}

static NSString *argResId(void)
{
	MTLTextureDescriptor *td = tex2d(MTLPixelFormatRGBA8Unorm, 64, MTLStorageModePrivate, MTLTextureUsageShaderRead, NO);
	id<MTLTexture> t = [gDevice newTextureWithDescriptor:td];
	const int n = 200000;
	uint64_t acc = 0;
	const double t0 = now();
	for (int i = 0; i < n; i++)
		acc += t.gpuResourceID._impl;
	const double el = now() - t0;
	return [NSString stringWithFormat:@"read_ns=%.1f nonzero=%d", el * 1e9 / n, acc != 0];
}

// --- CPU access ---------------------------------------------------------------------------------

static NSString *cpuShared(uint64_t len)
{
	Samples first = {0}, second = {0}, rd = {0};
	uint64_t total = 0;
	for (int i = -1; i < 10; i++) {
		@autoreleasepool {
			id<MTLBuffer> b = [gDevice newBufferWithLength:len options:MTLResourceStorageModeShared];
			if (b == nil)
				return @"FAIL no buffer";
			double t0 = now();
			memset(b.contents, i & 0xff, len);
			double t1 = now();
			memset(b.contents, (i + 1) & 0xff, len);
			double t2 = now();
			const uint64_t *p = b.contents;
			uint64_t s = 0;
			for (uint64_t k = 0; k < len / 8; k++)
				s += p[k];
			double t3 = now();
			total += s;
			if (i >= 0) {
				add(&first, t1 - t0);
				add(&second, t2 - t1);
				add(&rd, t3 - t2);
			}
		}
	}
	NSString *r = [NSString stringWithFormat:@"first_gbps=%.2f write_gbps=%.2f read_gbps=%.2f first_us=%.0f check=%llu",
	                                         len / pct(&first, 0.5) / 1e3, len / pct(&second, 0.5) / 1e3,
	                                         len / pct(&rd, 0.5) / 1e3, pct(&first, 0.5),
	                                         (unsigned long long)(total & 1)];
	drop(&first), drop(&second), drop(&rd);
	return r;
}

static NSString *cpuManaged(uint64_t len)
{
	id<MTLBuffer> m = [gDevice newBufferWithLength:len options:MTLResourceStorageModeManaged];
	id<MTLBuffer> p = [gDevice newBufferWithLength:len options:MTLResourceStorageModePrivate];
	if (m == nil || p == nil)
		return @"FAIL no buffer";
	Samples dmr = {0}, e2e = {0}, write = {0};
	const int n = opsFor(len) < 100 ? opsFor(len) : 100;
	for (int i = -2; i < n; i++) {
		const double t0 = now();
		memset(m.contents, i & 0xff, len);
		const double t1 = now();
		[m didModifyRange:NSMakeRange(0, len)];
		const double t2 = now();
		id<MTLCommandBuffer> cb = [gQueue commandBuffer];
		id<MTLBlitCommandEncoder> e = [cb blitCommandEncoder];
		[e copyFromBuffer:m sourceOffset:0 toBuffer:p destinationOffset:0 size:len];
		[e endEncoding];
		[cb commit];
		finish(cb);
		const double t3 = now();
		if (i >= 0) {
			add(&write, t1 - t0);
			add(&dmr, t2 - t1);
			add(&e2e, t3 - t0);
		}
	}
	NSString *r = [NSString stringWithFormat:@"%@ %@ write_us=%.1f mbps=%.0f", st("dmr", &dmr), st("e2e", &e2e),
	                                         pct(&write, 0.5), len / pct(&e2e, 0.5)];
	drop(&dmr), drop(&e2e), drop(&write);
	return r;
}

static NSString *cpuSync(uint64_t len)
{
	id<MTLBuffer> m = [gDevice newBufferWithLength:len options:MTLResourceStorageModeManaged];
	if (m == nil)
		return @"FAIL no buffer";
	Samples e2e = {0}, rd = {0};
	const int n = opsFor(len) < 100 ? opsFor(len) : 100;
	int bad = 0;
	for (int i = -2; i < n; i++) {
		const uint8_t v = (uint8_t)(i * 29 + 7);
		const double t0 = now();
		id<MTLCommandBuffer> cb = [gQueue commandBuffer];
		id<MTLBlitCommandEncoder> e = [cb blitCommandEncoder];
		[e fillBuffer:m range:NSMakeRange(0, len) value:v];
		[e synchronizeResource:m];
		[e endEncoding];
		[cb commit];
		finish(cb);
		const double t1 = now();
		const uint64_t *p = m.contents;
		const uint64_t want = 0x0101010101010101ull * v;
		uint64_t diff = 0;
		for (uint64_t k = 0; k < len / 8; k++)
			diff |= p[k] ^ want;
		const double t2 = now();
		bad += diff != 0;
		if (i >= 0) {
			add(&e2e, t2 - t0);
			add(&rd, t2 - t1);
		}
	}
	if (bad)
		return [NSString stringWithFormat:@"FAIL %d of %d reads differ from the GPU's fill", bad, n + 2];
	NSString *r = [NSString stringWithFormat:@"%@ read_us=%.1f mbps=%.0f", st("e2e", &e2e), pct(&rd, 0.5),
	                                         len / pct(&e2e, 0.5)];
	drop(&e2e), drop(&rd);
	return r;
}

static NSString *texReplace(NSUInteger w)
{
	id<MTLTexture> t = [gDevice newTextureWithDescriptor:tex2d(MTLPixelFormatRGBA8Unorm, w, MTLStorageModeManaged,
	                                                           MTLTextureUsageShaderRead, NO)];
	const NSUInteger bytes = w * w * 4;
	id<MTLBuffer> out = [gDevice newBufferWithLength:bytes options:MTLResourceStorageModeShared];
	uint8_t *src = malloc(bytes);
	Samples rep = {0}, e2e = {0};
	int bad = 0;
	const int n = opsFor(bytes) < 60 ? opsFor(bytes) : 60;
	for (int i = -2; i < n; i++) {
		const uint8_t v = (uint8_t)(i * 13 + 3);
		memset(src, v, bytes);
		const double t0 = now();
		[t replaceRegion:MTLRegionMake2D(0, 0, w, w) mipmapLevel:0 withBytes:src bytesPerRow:w * 4];
		const double t1 = now();
		id<MTLCommandBuffer> cb = [gQueue commandBuffer];
		id<MTLBlitCommandEncoder> e = [cb blitCommandEncoder];
		[e copyFromTexture:t
		         sourceSlice:0
		         sourceLevel:0
		        sourceOrigin:MTLOriginMake(0, 0, 0)
		          sourceSize:MTLSizeMake(w, w, 1)
		            toBuffer:out
		   destinationOffset:0
		  destinationBytesPerRow:w * 4
		destinationBytesPerImage:bytes];
		[e endEncoding];
		[cb commit];
		finish(cb);
		const double t2 = now();
		const uint8_t *o = out.contents;
		bad += o[0] != v || o[bytes - 1] != v || o[bytes / 2] != v;
		if (i >= 0) {
			add(&rep, t1 - t0);
			add(&e2e, t2 - t0);
		}
	}
	free(src);
	if (bad)
		return [NSString stringWithFormat:@"FAIL %d of %d copies differ from the bytes replaced", bad, n + 2];
	NSString *r = [NSString stringWithFormat:@"%@ gbps=%.2f %@", st("replace", &rep), bytes / pct(&rep, 0.5) / 1e3,
	                                         st("e2e", &e2e)];
	drop(&rep), drop(&e2e);
	return r;
}

static NSString *texGetBytes(NSUInteger w)
{
	id<MTLTexture> t = [gDevice newTextureWithDescriptor:tex2d(MTLPixelFormatRGBA8Unorm, w, MTLStorageModeManaged,
	                                                           MTLTextureUsageShaderRead, NO)];
	const NSUInteger bytes = w * w * 4;
	id<MTLBuffer> in = [gDevice newBufferWithLength:bytes options:MTLResourceStorageModeShared];
	uint8_t *dst = malloc(bytes);
	Samples get = {0}, e2e = {0};
	int bad = 0;
	const int n = opsFor(bytes) < 60 ? opsFor(bytes) : 60;
	for (int i = -2; i < n; i++) {
		const uint8_t v = (uint8_t)(i * 17 + 5);
		memset(in.contents, v, bytes);
		const double t0 = now();
		id<MTLCommandBuffer> cb = [gQueue commandBuffer];
		id<MTLBlitCommandEncoder> e = [cb blitCommandEncoder];
		[e copyFromBuffer:in
		           sourceOffset:0
		      sourceBytesPerRow:w * 4
		    sourceBytesPerImage:bytes
		             sourceSize:MTLSizeMake(w, w, 1)
		              toTexture:t
		       destinationSlice:0
		       destinationLevel:0
		      destinationOrigin:MTLOriginMake(0, 0, 0)];
		[e synchronizeResource:t];
		[e endEncoding];
		[cb commit];
		finish(cb);
		const double t1 = now();
		[t getBytes:dst bytesPerRow:w * 4 fromRegion:MTLRegionMake2D(0, 0, w, w) mipmapLevel:0];
		const double t2 = now();
		bad += dst[0] != v || dst[bytes - 1] != v || dst[bytes / 2] != v;
		if (i >= 0) {
			add(&get, t2 - t1);
			add(&e2e, t2 - t0);
		}
	}
	free(dst);
	if (bad)
		return [NSString stringWithFormat:@"FAIL %d of %d reads differ from the GPU's copy", bad, n + 2];
	NSString *r = [NSString stringWithFormat:@"%@ gbps=%.2f %@", st("get", &get), bytes / pct(&get, 0.5) / 1e3,
	                                         st("e2e", &e2e)];
	drop(&get), drop(&e2e);
	return r;
}

// --- memory over time ---------------------------------------------------------------------------

static NSString *memCycle(void)
{
	MTLTextureDescriptor *td = tex2d(MTLPixelFormatRGBA8Unorm, 512, MTLStorageModePrivate,
	                                 MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget, NO);
	MTLTextureDescriptor *md = tex2d(MTLPixelFormatRGBA8Unorm, 256, MTLStorageModeManaged, MTLTextureUsageShaderRead, NO);
	MTLHeapDescriptor *hd = [MTLHeapDescriptor new];
	hd.size = 16u << 20;
	hd.storageMode = MTLStorageModePrivate;
	double res0 = 0, resN = 0;
	double f0 = NAN, f1 = NAN, fN = NAN;
	Samples round = {0};
	for (int r = 0; r <= gRounds; r++) {
		const double t0 = now();
		@autoreleasepool {
			NSMutableArray *keep = [NSMutableArray array];
			id<MTLCommandBuffer> cb = [gQueue commandBuffer];
			id<MTLBlitCommandEncoder> e = [cb blitCommandEncoder];
			for (int i = 0; i < 100; i++) {
				id<MTLBuffer> b = [gDevice newBufferWithLength:65536 options:MTLResourceStorageModeShared];
				memset(b.contents, i, 65536);
				[e fillBuffer:b range:NSMakeRange(0, 256) value:1];
				[keep addObject:b];
			}
			for (int i = 0; i < 20; i++) {
				id<MTLBuffer> b = [gDevice newBufferWithLength:1u << 20 options:MTLResourceStorageModePrivate];
				[e fillBuffer:b range:NSMakeRange(0, 4096) value:2];
				[keep addObject:b];
				id<MTLBuffer> m = [gDevice newBufferWithLength:65536 options:MTLResourceStorageModeManaged];
				memset(m.contents, i, 65536);
				[m didModifyRange:NSMakeRange(0, 65536)];
				[e copyFromBuffer:m sourceOffset:0 toBuffer:b destinationOffset:0 size:65536];
				[keep addObject:m];
				[keep addObject:[gDevice newTextureWithDescriptor:td]];
				id<MTLTexture> mt = [gDevice newTextureWithDescriptor:md];
				[e synchronizeResource:mt];
				[keep addObject:mt];
				MTLSamplerDescriptor *sd = [MTLSamplerDescriptor new];
				sd.lodMaxClamp = 100.0f + i;
				[keep addObject:[gDevice newSamplerStateWithDescriptor:sd]];
			}
			for (int i = 0; i < 4; i++) {
				id<MTLHeap> h = [gDevice newHeapWithDescriptor:hd];
				[keep addObject:h];
				for (int k = 0; k < 8; k++) {
					id<MTLBuffer> b = [h newBufferWithLength:65536 options:MTLResourceStorageModePrivate];
					[e fillBuffer:b range:NSMakeRange(0, 4096) value:3];
					[keep addObject:b];
				}
			}
			for (int i = 0; i < 10; i++) {
				IOSurfaceRef s = makeSurface(256);
				id<MTLTexture> t = [gDevice newTextureWithDescriptor:surfaceDesc(256) iosurface:s plane:0];
				CFRelease(s);
				if (t != nil)
					[keep addObject:t];
			}
			[e endEncoding];
			[cb commit];
			finish(cb);
			// The surfaces' textures are written by a pass of their own, then read by the CPU.
			id<MTLCommandBuffer> cb2 = [gQueue commandBuffer];
			for (id o in keep) {
				if ([o conformsToProtocol:@protocol(MTLTexture)] && [o iosurface] != NULL) {
					MTLRenderPassDescriptor *p = [MTLRenderPassDescriptor renderPassDescriptor];
					p.colorAttachments[0].texture = o;
					p.colorAttachments[0].loadAction = MTLLoadActionClear;
					p.colorAttachments[0].storeAction = MTLStoreActionStore;
					[[cb2 renderCommandEncoderWithDescriptor:p] endEncoding];
				}
			}
			[cb2 commit];
			finish(cb2);
			for (id o in keep) {
				if ([o conformsToProtocol:@protocol(MTLTexture)] && [o iosurface] != NULL) {
					IOSurfaceRef s = [o iosurface];
					IOSurfaceLock(s, kIOSurfaceLockReadOnly, NULL);
					(void)((volatile uint8_t *)IOSurfaceGetBaseAddress(s))[0];
					IOSurfaceUnlock(s, kIOSurfaceLockReadOnly, NULL);
				}
			}
			[keep removeAllObjects];
		}
		const double el = now() - t0;
		double res = 0;
		const double f = footprintMiB(&res);
		printf("# mem round %d t=%.3f unix=%.3f footprint_mib=%.2f resident_mib=%.2f round_ms=%.1f\n", r, now() - gStart,
		       unixNow(), f, res, el * 1e3);
		fflush(stdout);
		if (r == 0) {
			f0 = f;
			res0 = res;
		} else {
			add(&round, el);
		}
		if (r == 1)
			f1 = f;
		fN = f;
		resN = res;
	}
	NSString *s = [NSString stringWithFormat:@"footprint_first_mib=%.2f footprint_after1_mib=%.2f footprint_last_mib=%.2f "
	                                         @"growth_kib_round=%.1f resident_first_mib=%.1f resident_last_mib=%.1f round_ms=%.1f",
	                                         f0, f1, fN, (fN - f1) * 1024 / (gRounds > 1 ? gRounds - 1 : 1), res0, resN,
	                                         pct(&round, 0.5) / 1e3];
	drop(&round);
	return s;
}

static NSString *regionFill(void)
{
	NSMutableArray *held = [NSMutableArray array];
	const uint64_t cap = 1536;
	while (held.count < cap) {
		id<MTLBuffer> b = [gDevice newBufferWithLength:1u << 20 options:MTLResourceStorageModeShared];
		if (b == nil)
			break;
		[held addObject:b];
	}
	const NSUInteger got = held.count;
	for (NSInteger i = (NSInteger)held.count - 1; i >= 0; i -= 2)
		[held removeObjectAtIndex:(NSUInteger)i];
	usleep(200000);
	id<MTLBuffer> b64 = [gDevice newBufferWithLength:64u << 20 options:MTLResourceStorageModeShared];
	id<MTLBuffer> b256 = [gDevice newBufferWithLength:256u << 20 options:MTLResourceStorageModeShared];
	const BOOL got64 = b64 != nil, got256 = b256 != nil;
	b64 = nil;
	b256 = nil;
	[held removeAllObjects];
	usleep(200000);
	id<MTLBuffer> b512 = [gDevice newBufferWithLength:512u << 20 options:MTLResourceStorageModeShared];
	const BOOL got512 = b512 != nil;
	b512 = nil;
	return [NSString stringWithFormat:@"mib_1m=%lu holes_64m=%d holes_256m=%d after_512m=%d", (unsigned long)got, got64,
	                                  got256, got512];
}

// --- the run ------------------------------------------------------------------------------------

int main(int argc, char **argv)
{
	@autoreleasepool {
		gStart = now();
		NSMutableArray *patterns = [NSMutableArray array];
		const char *want = NULL;
		BOOL list = NO;
		for (int i = 1; i < argc; i++) {
			if (strcmp(argv[i], "-d") == 0 && i + 1 < argc)
				want = argv[++i];
			else if (strcmp(argv[i], "-n") == 0 && i + 1 < argc)
				gOps = atoi(argv[++i]);
			else if (strcmp(argv[i], "-R") == 0 && i + 1 < argc)
				gRounds = atoi(argv[++i]);
			else if (strcmp(argv[i], "-l") == 0)
				list = YES;
			else
				[patterns addObject:@(argv[i])];
		}
		gPatterns = patterns;
		if (gOps < 8)
			gOps = 8;
		if (gRounds < 2)
			gRounds = 2;
		if (list) {
			for (id<MTLDevice> d in MTLCopyAllDevices())
				printf("res: device \"%s\" lowPower=%d removable=%d unified=%d\n", d.name.UTF8String, d.lowPower,
				       d.removable, d.hasUnifiedMemory);
			return 0;
		}
		if (want != NULL) {
			for (id<MTLDevice> d in MTLCopyAllDevices())
				if (gDevice == nil && strcasestr(d.name.UTF8String, want))
					gDevice = d;
		} else {
			gDevice = MTLCreateSystemDefaultDevice();
		}
		if (gDevice == nil) {
			printf("res: no Metal device%s%s\n", want ? " matching " : "", want ? want : "");
			return 2;
		}
		char model[64] = "?", build[32] = "?";
		size_t len = sizeof(model);
		sysctlbyname("hw.model", model, &len, NULL, 0);
		len = sizeof(build);
		sysctlbyname("kern.osversion", build, &len, NULL, 0);
		NSOperatingSystemVersion v = NSProcessInfo.processInfo.operatingSystemVersion;
		printf("res: device \"%s\", %s, macOS %ld.%ld.%ld (%s), %s, unified=%d, ops %d, unix=%.3f\n",
		       gDevice.name.UTF8String, model, (long)v.majorVersion, (long)v.minorVersion, (long)v.patchVersion, build,
#if defined(__x86_64__)
		       "x86_64",
#else
		       "arm64",
#endif
		       gDevice.hasUnifiedMemory, gOps, unixNow());
		fflush(stdout);
		gQueue = [gDevice newCommandQueue];

		NSArray *sizes = @[ @"4k", @"64k", @"1m", @"16m", @"256m" ];
		NSDictionary *modes = @{
			@"private" : @(MTLResourceStorageModePrivate),
			@"shared" : @(MTLResourceStorageModeShared),
			@"managed" : @(MTLResourceStorageModeManaged),
		};
		for (NSString *mode in @[ @"private", @"shared", @"managed" ])
			for (NSString *size in sizes)
				runCase([NSString stringWithFormat:@"buf.%@.%@", mode, size], YES, ^{
					return bufCase([modes[mode] unsignedIntegerValue], parseSize(size), kLength);
				});
		for (NSString *size in @[ @"4k", @"1m", @"16m" ]) {
			runCase([@"buf.bytes." stringByAppendingString:size], YES,
			        ^{ return bufCase(MTLResourceStorageModeShared, parseSize(size), kBytes); });
			runCase([@"buf.nocopy." stringByAppendingString:size], YES,
			        ^{ return bufCase(MTLResourceStorageModeShared, parseSize(size), kNoCopy); });
		}
		runCase(@"buf.hold.shared.4k", YES, ^{ return bufHold(); });
		runCase(@"buf.recycle.shared.64m", YES, ^{ return bufRecycle(); });

		const MTLTextureUsage sr = MTLTextureUsageShaderRead;
		const MTLTextureUsage rt = MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget;
		for (NSNumber *w in @[ @256, @1024, @4096 ])
			runCase([NSString stringWithFormat:@"tex.private.%@", w], YES, ^{
				return texCase(tex2d(MTLPixelFormatRGBA8Unorm, w.unsignedIntegerValue, MTLStorageModePrivate, sr, NO), NO);
			});
		runCase(@"tex.rt.4096", YES,
		        ^{ return texCase(tex2d(MTLPixelFormatRGBA8Unorm, 4096, MTLStorageModePrivate, rt, NO), YES); });
		runCase(@"tex.managed.1024", YES,
		        ^{ return texCase(tex2d(MTLPixelFormatRGBA8Unorm, 1024, MTLStorageModeManaged, sr, NO), NO); });
		runCase(@"tex.depth.2048", YES, ^{
			return texCase(tex2d(MTLPixelFormatDepth32Float, 2048, MTLStorageModePrivate, MTLTextureUsageRenderTarget, NO),
			               YES);
		});
		runCase(@"tex.mips.2048", YES,
		        ^{ return texCase(tex2d(MTLPixelFormatBGRA8Unorm, 2048, MTLStorageModePrivate, sr, YES), NO); });
		runCase(@"tex.linear.1024", YES, ^{ return texLinear(1024); });
		runCase(@"tex.view.1024", YES, ^{ return texView(1024); });

		for (NSNumber *w in @[ @256, @1024, @4096 ])
			runCase([NSString stringWithFormat:@"ios.create.%@", w], YES, ^{ return iosCreate(w.unsignedIntegerValue); });
		for (NSNumber *w in @[ @256, @1024, @4096 ])
			runCase([NSString stringWithFormat:@"ios.tex.%@", w], YES, ^{ return iosTex(w.unsignedIntegerValue); });
		runCase(@"ios.fresh.1024", YES, ^{ return iosFresh(1024); });
		runCase(@"ios.lock.1024", YES, ^{ return iosLock(1024); });
		runCase(@"ios.lock.gpu.1024", YES, ^{ return iosLockGpu(1024); });
		runCase(@"ios.upload.1024", YES, ^{ return iosUpload(1024); });
		runCase(@"ios.inuse.1024", YES, ^{ return iosInUse(1024); });

		runCase(@"heap.private.16m", YES, ^{ return heapCase(MTLStorageModePrivate, 16u << 20); });
		runCase(@"heap.private.256m", YES, ^{ return heapCase(MTLStorageModePrivate, 256u << 20); });
#if defined(__x86_64__)
		const BOOL sharedHeaps = NO;
#else
		const BOOL sharedHeaps = YES;
#endif
		runCase(@"heap.shared.16m", sharedHeaps, ^{ return heapCase(MTLStorageModeShared, 16u << 20); });
		runCase(@"heap.place.buf.64k", YES, ^{ return heapPlace(NO); });
		runCase(@"heap.place.tex.1024", YES, ^{ return heapPlace(YES); });
		runCase(@"heap.query", YES, ^{ return heapQuery(); });

		runCase(@"sampler.same", YES, ^{ return samplerCase(0); });
		runCase(@"sampler.distinct", YES, ^{ return samplerCase(1); });
		runCase(@"sampler.argbuf", YES, ^{ return samplerCase(2); });
		runCase(@"depthstencil", YES, ^{ return depthStencil(); });
		runCase(@"argbuf.encoder", YES, ^{ return argEncoder(); });
		runCase(@"argbuf.set", YES, ^{ return argSet(); });
		runCase(@"argbuf.resid", YES, ^{ return argResId(); });

		runCase(@"cpu.shared.64m", YES, ^{ return cpuShared(64u << 20); });
		for (NSString *size in @[ @"4k", @"1m", @"64m" ]) {
			runCase([@"cpu.managed." stringByAppendingString:size], YES, ^{ return cpuManaged(parseSize(size)); });
			runCase([@"cpu.sync." stringByAppendingString:size], YES, ^{ return cpuSync(parseSize(size)); });
		}
		for (NSNumber *w in @[ @256, @1024, @4096 ]) {
			runCase([NSString stringWithFormat:@"tex.replace.%@", w], YES, ^{ return texReplace(w.unsignedIntegerValue); });
			runCase([NSString stringWithFormat:@"tex.getbytes.%@", w], YES, ^{ return texGetBytes(w.unsignedIntegerValue); });
		}

		runCase(@"mem.cycle", YES, ^{ return memCycle(); });
		runCase(@"region.fill", NO, ^{ return regionFill(); });
	}
	return 0;
}
