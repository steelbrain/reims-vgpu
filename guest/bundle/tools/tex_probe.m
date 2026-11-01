// tex_probe: textures on the reims-graphics Metal device, with their resource ids, in one process.
//   tex_probe copy      a 64 x 64 BGRA8 Private texture: one blit copies a Shared buffer's pattern
//                       into it and back out into a second buffer, which is checked byte for byte
//   tex_probe hold N    N 4 x 4 textures made and held at once, then the copy above: prints how
//                       many were made, and their lowest and highest gpuResourceID, which must be
//                       distinct, non-zero and below 16381 (the host device's limit)
//   tex_probe handler N the copy above, waited for only by its completed handler (no
//                       waitUntilCompleted), then N command buffers each filling a Shared buffer,
//                       committed at once and waited for only by their handlers, which must run
//                       off the main thread, once each, in commit order, the buffer filled by the
//                       last; then 2 s idle, in which the process must use under 20 ms of CPU (the
//                       completion thread sleeps in the kext)
//   tex_probe churn N   N times: a new 64 x 64 Private texture, one command buffer copying a
//                       pattern of its own through it, committed, and the texture let go before
//                       the buffer completes; each buffer must complete with its own bytes back
//                       (its texture outlives it on the host), and N past the device's 16320
//                       resource ids needs each texture's id back once it is gone
//   tex_probe big N     one command buffer of N fills, each 4 bytes of a Shared buffer with its
//                       own value, far larger than its queue's ring for large N, waited for by its
//                       handler alone: every word must hold its fill
//   tex_probe kinds N   N rounds, each of one object of every kind made and let go while the
//                       round's command buffer is in flight: a new queue, a 4 MiB Shared buffer
//                       (a region block), a 1 MiB Shared heap with a buffer and a texture placed
//                       in it, a 256 MiB Private heap with a texture placed in it, a Private
//                       buffer of a page and a byte, a 64 x 64 IOSurface texture (its copy is a
//                       region block), a Private texture, a sampler, a depth-stencil state, an
//                       event, a fence and a shared event. The buffer fills the 4 MiB buffer and copies a
//                       pattern through the Private texture into it, the copy out waiting on
//                       the fence the copy in updates; each must complete with those bytes in
//                       the Shared buffer, so N past 256 needs each queue to end, and N past
//                       256 needs the blocks back (the region is 1 GiB), and N past 250 needs
//                       the Private heaps' GPU addresses back (64 GiB a process)
//   tex_probe sharedheap N  N rounds of a 1 MiB Shared Placement heap with buffers placed in
//                       it: A of 64 KiB at 0, B of 64 KiB at 64 KiB and C of 4 KiB at 0, over
//                       A's first bytes. Each placed buffer must be the heap's bytes at its
//                       offset (C's contents A's, B's 64 KiB past them) at the heap's GPU address
//                       plus the offset; the CPU writes a pattern through A, which C reads, and
//                       one command buffer copies A to B and fills C's second half: B must hold
//                       the CPU's pattern and A the GPU's fill, each seen through the other's
//                       contents, as Metal's placed resources alias their heap's memory
//   tex_probe autoheap N  N rounds of a 1000000-byte Automatic heap, Private, then Shared, placed in
//                       as Metal places (M3 Max, macOS 27): its size the 16 KiB pages that hold
//                       it; buffers of 100, 100 and 70000 bytes at its first byte, 256 and 512
//                       bytes past it (their gpuAddress), heapOffset 0, then a 256 x 256 RGBA8
//                       texture of the device's heapTextureSizeAndAlignWithDescriptor: size,
//                       usedSize their sizes; one command buffer fills each buffer with its own
//                       byte and copies a pattern through the texture, and every byte must come
//                       back; the first buffer let go and the second made aliasable give their
//                       bytes back (usedSize, within a second of the command buffer's end), and a
//                       buffer of 50 bytes goes at the first byte
//   tex_probe nocopy N  N rounds of newBufferWithBytesNoCopy: over the app's own memory, CA's two
//                       shapes and one Metal takes too: a page-based 0x14000-byte image, a
//                       0x11000-byte one under a 468 x 36 BGRA8 texture at offset 0x100 with
//                       1920-byte rows, and 4000 malloc'd bytes 3 past a 16-byte boundary. Each
//                       buffer's contents must be the app's pointer; one command buffer copies
//                       the first and the texture out, the CPU's bytes, and fills part of the
//                       first and all of the third, which the app's memory must hold once it
//                       completes, with every other byte as the CPU left it; a second copies the
//                       first out again after the CPU rewrites it. The deallocators run once the
//                       buffers go, and a length of 0 or a range not wholly mapped makes none
//   tex_probe events N  N rounds of command buffers ordered by events across two queues, as
//                       Metal's are (M3 Max, macOS 27): a buffer on the second queue waits for a
//                       new shared event to reach 1, then copies a buffer the first queue's next
//                       buffer fills and then signals 1, committed after it, and must copy the
//                       fill; the value is 1 once the signalling buffer's waitUntilCompleted
//                       returns. A listener at 2 runs on its own queue, with 2, once a buffer
//                       signals 2. A buffer waiting for 3 is not completed 2 ms after its
//                       commit, and runs once the CPU sets 3; a set to 1 leaves the value at 3.
//                       A plain MTLEvent orders a copy across the queues as the shared one does
// Prints "TEX OK" when every check passes, else "TEX FAIL: why" and exits 1. A completion 20 s
// late prints HANG and exits 3. The copy waits with waitUntilCompleted, or with its completed
// handler alone (`handler`).
// Build: xcrun clang -arch x86_64 -fobjc-arc -framework Metal -framework Foundation
//        -framework IOSurface tex_probe.m -o tex_probe
#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <Metal/Metal.h>
#include <mach/mach.h>
#include <pthread.h>
#include <sys/resource.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <unistd.h>

static const unsigned long kLimit = 16381;

// One phase's watchdog: it ends the process with HANG if `done` is not set 20 s after it starts.
struct watch {
	atomic_int done;
	const char *what;
};

static void *watchdog(void *arg)
{
	struct watch *w = arg;
	sleep(20);
	if (!atomic_load(&w->done)) {
		printf("HANG: %s never completed\n", w->what);
		fflush(stdout);
		_exit(3);
	}
	return NULL;
}

// Starts a watchdog over `what`; set the returned watch's `done` when it completes.
static struct watch *watch_start(const char *what)
{
	struct watch *w = calloc(1, sizeof(*w)); // the watchdog may outlive the phase: never freed
	atomic_init(&w->done, 0);
	w->what = what;
	pthread_t dog;
	pthread_create(&dog, NULL, watchdog, w);
	pthread_detach(dog);
	return w;
}

static double now_ms(void)
{
	struct timespec t;
	clock_gettime(CLOCK_MONOTONIC, &t);
	return t.tv_sec * 1e3 + t.tv_nsec / 1e6;
}

static int fail(const char *why)
{
	printf("TEX FAIL: %s\n", why);
	return 1;
}

// Waits for the semaphore a command buffer's completed handler signals, and nothing else.
static void wait_by_handler(dispatch_semaphore_t done)
{
	dispatch_semaphore_wait(done, DISPATCH_TIME_FOREVER);
}

// The copy through a Private texture, checked byte for byte; its texture's resource id printed.
// `byHandler` waits for it with a completed handler alone.
static int copy_through_texture(id<MTLDevice> device, id<MTLCommandQueue> queue, int byHandler)
{
	const NSUInteger side = 64, row = side * 4, size = row * side;
	id<MTLBuffer> src = [device newBufferWithLength:size options:MTLResourceStorageModeShared];
	id<MTLBuffer> dst = [device newBufferWithLength:size options:MTLResourceStorageModeShared];
	if (src == nil || dst == nil)
		return fail("no Shared buffer");
	uint8_t *s = (uint8_t *)[src contents];
	for (NSUInteger i = 0; i < size; i++)
		s[i] = (uint8_t)(i * 7 + 3);
	memset([dst contents], 0, size);
	MTLTextureDescriptor *d =
	    [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
	                                                       width:side
	                                                      height:side
	                                                   mipmapped:NO];
	d.storageMode = MTLStorageModePrivate;
	d.usage = MTLTextureUsageShaderRead;
	id<MTLTexture> tex = [device newTextureWithDescriptor:d];
	if (tex == nil)
		return fail("no texture");
	printf("texture gpuResourceID=%llu\n", (unsigned long long)tex.gpuResourceID._impl);
	if (tex.gpuResourceID._impl == 0 || tex.gpuResourceID._impl >= kLimit)
		return fail("texture resource id 0 or past the limit");
	id<MTLCommandBuffer> cb = [queue commandBuffer];
	id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
	[blit copyFromBuffer:src
	           sourceOffset:0
	      sourceBytesPerRow:row
	    sourceBytesPerImage:size
	             sourceSize:MTLSizeMake(side, side, 1)
	              toTexture:tex
	       destinationSlice:0
	       destinationLevel:0
	      destinationOrigin:MTLOriginMake(0, 0, 0)];
	[blit copyFromTexture:tex
	                 sourceSlice:0
	                 sourceLevel:0
	                sourceOrigin:MTLOriginMake(0, 0, 0)
	                  sourceSize:MTLSizeMake(side, side, 1)
	                    toBuffer:dst
	           destinationOffset:0
	      destinationBytesPerRow:row
	    destinationBytesPerImage:size];
	[blit endEncoding];
	dispatch_semaphore_t done = dispatch_semaphore_create(0);
	__block int onMain = 0;
	if (byHandler) {
		[cb addCompletedHandler:^(id<MTLCommandBuffer> b) {
			(void)b;
			onMain = pthread_main_np();
			dispatch_semaphore_signal(done);
		}];
	}
	struct watch *w = watch_start("the copy's command buffer");
	[cb commit];
	if (byHandler)
		wait_by_handler(done);
	else
		[cb waitUntilCompleted];
	atomic_store(&w->done, 1);
	if (byHandler && onMain)
		return fail("the completed handler ran on the main thread");
	printf("status=%ld (4=completed,5=error)\n", (long)cb.status);
	if (cb.status != MTLCommandBufferStatusCompleted)
		return fail(cb.error ? [[cb.error description] UTF8String] : "not completed");
	if (memcmp([src contents], [dst contents], size) != 0) {
		const uint8_t *o = (const uint8_t *)[dst contents];
		printf("dst[0]=0x%02x expected 0x%02x\n", o[0], s[0]);
		return fail("the bytes back from the texture differ");
	}
	return 0;
}

// `n` command buffers, each filling a Shared buffer with its own value, committed back to back
// and waited for only by their completed handlers: each runs once, off the main thread, in
// commit order, and the buffer holds the last one's value.
static int handlers_in_order(id<MTLDevice> device, id<MTLCommandQueue> queue, long n)
{
	id<MTLBuffer> buf = [device newBufferWithLength:256 options:MTLResourceStorageModeShared];
	if (buf == nil)
		return fail("no Shared buffer");
	memset([buf contents], 0, 256);
	long *order = calloc((size_t)n, sizeof(long));
	__block atomic_long ran;
	atomic_init(&ran, 0);
	__block atomic_int offMain;
	atomic_init(&offMain, 1);
	dispatch_semaphore_t all = dispatch_semaphore_create(0);
	struct watch *w = watch_start("the fills' command buffers");
	double start = now_ms();
	for (long i = 0; i < n; i++) {
		id<MTLCommandBuffer> cb = [queue commandBuffer];
		id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
		[blit fillBuffer:buf range:NSMakeRange(0, 256) value:(uint8_t)(i + 1)];
		[blit endEncoding];
		[cb addCompletedHandler:^(id<MTLCommandBuffer> b) {
			long at = atomic_fetch_add(&ran, 1);
			if (at < n)
				order[at] = i;
			if (pthread_main_np() || b.status != MTLCommandBufferStatusCompleted)
				atomic_store(&offMain, 0);
			if (at + 1 == n)
				dispatch_semaphore_signal(all);
		}];
		[cb commit];
	}
	dispatch_semaphore_wait(all, DISPATCH_TIME_FOREVER);
	atomic_store(&w->done, 1);
	double took = now_ms() - start;
	usleep(100000); // a handler run twice would land here
	long total = atomic_load(&ran);
	long misplaced = 0;
	for (long i = 0; i < n; i++)
		misplaced += order[i] != i;
	free(order);
	uint8_t last = ((const uint8_t *)[buf contents])[255];
	printf("handlers ran=%ld of %ld in %.1f ms, out of order=%ld, off main and completed=%d, "
	       "buf=0x%02x\n",
	       total, n, took, misplaced, atomic_load(&offMain), last);
	if (total != n)
		return fail("a completed handler ran more than once");
	if (misplaced)
		return fail("the completed handlers ran out of commit order");
	if (!atomic_load(&offMain))
		return fail("a handler ran on the main thread or saw a buffer not completed");
	if (last != (uint8_t)n)
		return fail("the buffer does not hold the last fill");
	return 0;
}

// `n` textures, each let go right after the command buffer that copies through it is committed.
static int churn(id<MTLDevice> device, id<MTLCommandQueue> queue, long n)
{
	const NSUInteger side = 64, row = side * 4, size = row * side;
	id<MTLBuffer> src = [device newBufferWithLength:size options:MTLResourceStorageModeShared];
	id<MTLBuffer> dst = [device newBufferWithLength:size options:MTLResourceStorageModeShared];
	if (src == nil || dst == nil)
		return fail("no Shared buffer");
	MTLTextureDescriptor *d =
	    [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
	                                                       width:side
	                                                      height:side
	                                                   mipmapped:NO];
	d.storageMode = MTLStorageModePrivate;
	d.usage = MTLTextureUsageShaderRead;
	double start = now_ms();
	long bad = 0, failed = 0, noTexture = 0;
	for (long i = 0; i < n; i++) {
		struct watch *w = watch_start("a churn command buffer");
		memset([src contents], (int)(i * 13 + 5), size);
		memset([dst contents], 0, size);
		dispatch_semaphore_t done = dispatch_semaphore_create(0);
		__block MTLCommandBufferStatus status = MTLCommandBufferStatusNotEnqueued;
		@autoreleasepool {
			id<MTLTexture> tex = [device newTextureWithDescriptor:d];
			if (tex == nil) {
				noTexture++;
				atomic_store(&w->done, 1);
				continue;
			}
			id<MTLCommandBuffer> cb = [queue commandBuffer];
			id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
			[blit copyFromBuffer:src
			           sourceOffset:0
			      sourceBytesPerRow:row
			    sourceBytesPerImage:size
			             sourceSize:MTLSizeMake(side, side, 1)
			              toTexture:tex
			       destinationSlice:0
			       destinationLevel:0
			      destinationOrigin:MTLOriginMake(0, 0, 0)];
			[blit copyFromTexture:tex
			                 sourceSlice:0
			                 sourceLevel:0
			                sourceOrigin:MTLOriginMake(0, 0, 0)
			                  sourceSize:MTLSizeMake(side, side, 1)
			                    toBuffer:dst
			           destinationOffset:0
			      destinationBytesPerRow:row
			    destinationBytesPerImage:size];
			[blit endEncoding];
			[cb addCompletedHandler:^(id<MTLCommandBuffer> b) {
				status = b.status;
				dispatch_semaphore_signal(done);
			}];
			[cb commit];
		} // the texture goes now, while its buffer is in flight
		dispatch_semaphore_wait(done, DISPATCH_TIME_FOREVER);
		atomic_store(&w->done, 1);
		if (status != MTLCommandBufferStatusCompleted)
			failed++;
		else if (memcmp([src contents], [dst contents], size) != 0)
			bad++;
	}
	printf("churn: %ld textures in %.0f ms, failed=%ld, wrong bytes=%ld, no texture=%ld\n", n,
	       now_ms() - start, failed, bad, noTexture);
	if (noTexture)
		return fail("a texture could not be made: an id never came back");
	if (failed)
		return fail("a command buffer failed: its texture went before it ran");
	if (bad)
		return fail("a copy came back wrong");
	return 0;
}

// `n` rounds of one object of every kind, each let go while the round's command buffer runs.
static int kinds(id<MTLDevice> device, long n)
{
	const NSUInteger side = 64, row = side * 4, size = row * side, big = 4u << 20;
	id<MTLBuffer> src = [device newBufferWithLength:size options:MTLResourceStorageModeShared];
	if (src == nil)
		return fail("no Shared buffer");
	MTLTextureDescriptor *d =
	    [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
	                                                       width:side
	                                                      height:side
	                                                   mipmapped:NO];
	d.storageMode = MTLStorageModePrivate;
	d.usage = MTLTextureUsageShaderRead;
	MTLTextureDescriptor *sd =
	    [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
	                                                       width:side
	                                                      height:side
	                                                   mipmapped:NO];
	sd.usage = MTLTextureUsageShaderRead;
	NSDictionary *props = @{
		(id)kIOSurfaceWidth : @(side),
		(id)kIOSurfaceHeight : @(side),
		(id)kIOSurfaceBytesPerElement : @4,
		(id)kIOSurfacePixelFormat : @((uint32_t)'BGRA'),
	};
	double start = now_ms();
	long failed = 0, bad = 0, missing = 0;
	for (long i = 0; i < n; i++) {
		struct watch *w = watch_start("a kinds command buffer");
		uint8_t fill = (uint8_t)(i * 7 + 3), pattern = (uint8_t)(i * 13 + 5);
		memset([src contents], pattern, size);
		dispatch_semaphore_t done = dispatch_semaphore_create(0);
		__block MTLCommandBufferStatus status = MTLCommandBufferStatusNotEnqueued;
		id<MTLBuffer> out = nil; // checked once the round's objects are gone
		@autoreleasepool {
			id<MTLCommandQueue> queue = [device newCommandQueue];
			id<MTLBuffer> shared = [device newBufferWithLength:big
			                                          options:MTLResourceStorageModeShared];
			MTLHeapDescriptor *hd = [MTLHeapDescriptor new];
			hd.size = 1u << 20;
			hd.storageMode = MTLStorageModeShared;
			hd.type = MTLHeapTypePlacement; // resources at offsets the probe names
			id<MTLHeap> heap = [device newHeapWithDescriptor:hd];
			id<MTLBuffer> placed = [heap newBufferWithLength:4096
			                                         options:MTLResourceStorageModeShared
			                                          offset:0];
			// A texture placed in a heap takes the heap's storage mode, as Metal requires.
			MTLTextureDescriptor *placedDesc = [sd copy];
			placedDesc.storageMode = MTLStorageModeShared;
			id<MTLTexture> placedTex = [heap newTextureWithDescriptor:placedDesc offset:0x10000];
			// A 256 MiB Private heap a round: 1000 rounds go round the process's 64 GiB of GPU
			// addresses four times, so its picks must come back as the heaps go.
			MTLHeapDescriptor *pd = [MTLHeapDescriptor new];
			pd.size = 256u << 20;
			pd.storageMode = MTLStorageModePrivate;
			pd.type = MTLHeapTypePlacement;
			id<MTLHeap> privateHeap = [device newHeapWithDescriptor:pd];
			id<MTLTexture> privateTex = [privateHeap newTextureWithDescriptor:d offset:0];
			id<MTLBuffer> deviceBuffer = [device newBufferWithLength:0x4001
			                                                  options:MTLResourceStorageModePrivate];
			IOSurfaceRef surface = IOSurfaceCreate((__bridge CFDictionaryRef)props);
			id<MTLTexture> surfaceTex =
			    surface ? [device newTextureWithDescriptor:sd iosurface:surface plane:0] : nil;
			if (surface)
				CFRelease(surface);
			id<MTLTexture> tex = [device newTextureWithDescriptor:d];
			id<MTLSamplerState> sampler = [device newSamplerStateWithDescriptor:[MTLSamplerDescriptor new]];
			id<MTLDepthStencilState> depth =
			    [device newDepthStencilStateWithDescriptor:[MTLDepthStencilDescriptor new]];
			id<MTLEvent> event = [device newEvent];
			id<MTLFence> fence = [device newFence];
			id<MTLSharedEvent> sharedEvent = [device newSharedEvent];
			if (queue == nil || shared == nil || heap == nil || placed == nil || placedTex == nil ||
			    privateHeap == nil || privateTex == nil || deviceBuffer == nil ||
			    surfaceTex == nil || tex == nil || sampler == nil || depth == nil || event == nil ||
			    fence == nil || sharedEvent == nil) {
				printf("round %ld: queue=%d shared=%d heap=%d placed=%d placedTex=%d "
				       "privateHeap=%d privateTex=%d deviceBuffer=%d surfaceTex=%d "
				       "tex=%d sampler=%d depth=%d event=%d fence=%d sharedEvent=%d\n",
				       i, queue != nil, shared != nil, heap != nil, placed != nil, placedTex != nil,
				       privateHeap != nil, privateTex != nil, deviceBuffer != nil,
				       surfaceTex != nil, tex != nil, sampler != nil, depth != nil, event != nil,
				       fence != nil, sharedEvent != nil);
				missing++;
				atomic_store(&w->done, 1);
				break;
			}
			id<MTLCommandBuffer> cb = [queue commandBuffer];
			id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
			[blit fillBuffer:shared range:NSMakeRange(0, big) value:fill];
			[blit fillBuffer:deviceBuffer range:NSMakeRange(0, 0x4001) value:fill];
			[blit copyFromBuffer:src
			           sourceOffset:0
			      sourceBytesPerRow:row
			    sourceBytesPerImage:size
			             sourceSize:MTLSizeMake(side, side, 1)
			              toTexture:privateTex
			       destinationSlice:0
			       destinationLevel:0
			      destinationOrigin:MTLOriginMake(0, 0, 0)];
			[blit copyFromBuffer:src
			           sourceOffset:0
			      sourceBytesPerRow:row
			    sourceBytesPerImage:size
			             sourceSize:MTLSizeMake(side, side, 1)
			              toTexture:tex
			       destinationSlice:0
			       destinationLevel:0
			      destinationOrigin:MTLOriginMake(0, 0, 0)];
			[blit updateFence:fence];
			[blit endEncoding];
			blit = [cb blitCommandEncoder];
			[blit waitForFence:fence];
			[blit copyFromTexture:tex
			                 sourceSlice:0
			                 sourceLevel:0
			                sourceOrigin:MTLOriginMake(0, 0, 0)
			                  sourceSize:MTLSizeMake(side, side, 1)
			                    toBuffer:shared
			           destinationOffset:0
			      destinationBytesPerRow:row
			    destinationBytesPerImage:size];
			[blit endEncoding];
			[cb addCompletedHandler:^(id<MTLCommandBuffer> b) {
				status = b.status;
				dispatch_semaphore_signal(done);
			}];
			[cb commit];
			out = shared; // kept to check its bytes; every other object goes now
		}
		dispatch_semaphore_wait(done, DISPATCH_TIME_FOREVER);
		atomic_store(&w->done, 1);
		if (missing)
			break;
		@autoreleasepool {
			const uint8_t *b = (const uint8_t *)[out contents];
			long wrong = 0;
			for (NSUInteger k = 0; k < big; k++)
				wrong += b[k] != (k < size ? pattern : fill);
			if (status != MTLCommandBufferStatusCompleted)
				failed++;
			else if (wrong)
				bad++;
			out = nil; // its block goes back for the next round
		}
	}
	printf("kinds: %ld rounds in %.0f ms (%.1f GiB of 4 MiB Shared buffers), failed=%ld, wrong=%ld, "
	       "missing=%ld\n",
	       n, now_ms() - start, (double)n * big / (1u << 30), failed, bad, missing);
	if (missing)
		return fail("an object could not be made: a queue or a block never came back");
	if (failed)
		return fail("a command buffer failed: an object went before it ran");
	if (bad)
		return fail("a Shared buffer came back wrong");
	return 0;
}

// One command buffer of `n` 4-byte fills, each of its own value, waited for by its handler.
static int one_big_buffer(id<MTLDevice> device, id<MTLCommandQueue> queue, long n)
{
	id<MTLBuffer> buf = [device newBufferWithLength:(NSUInteger)(4 * n)
	                                        options:MTLResourceStorageModeShared];
	if (buf == nil)
		return fail("no Shared buffer");
	memset([buf contents], 0, (size_t)(4 * n));
	id<MTLCommandBuffer> cb = [queue commandBuffer];
	id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
	for (long i = 0; i < n; i++)
		[blit fillBuffer:buf range:NSMakeRange((NSUInteger)(4 * i), 4) value:(uint8_t)(i * 7 + 1)];
	[blit endEncoding];
	dispatch_semaphore_t done = dispatch_semaphore_create(0);
	[cb addCompletedHandler:^(id<MTLCommandBuffer> b) {
		(void)b;
		dispatch_semaphore_signal(done);
	}];
	struct watch *w = watch_start("the big command buffer");
	double start = now_ms();
	[cb commit];
	dispatch_semaphore_wait(done, DISPATCH_TIME_FOREVER);
	atomic_store(&w->done, 1);
	double took = now_ms() - start;
	const uint8_t *b = (const uint8_t *)[buf contents];
	long wrong = 0;
	for (long i = 0; i < 4 * n; i++)
		wrong += b[i] != (uint8_t)((i / 4) * 7 + 1);
	printf("big: %ld fills in %.1f ms, status=%ld, wrong bytes=%ld\n", n, took, (long)cb.status,
	       wrong);
	if (cb.status != MTLCommandBufferStatusCompleted)
		return fail(cb.error ? [[cb.error description] UTF8String] : "not completed");
	if (wrong)
		return fail("a fill of the big command buffer is missing");
	return 0;
}

// `n` rounds of a Shared heap whose placed buffers alias its bytes, the CPU's and the GPU's
// writes each seen through the other buffers over them.
static int shared_heap(id<MTLDevice> device, id<MTLCommandQueue> queue, long n)
{
	const NSUInteger half = 0x10000, small = 0x1000;
	const MTLResourceOptions shared = MTLResourceStorageModeShared;
	double start = now_ms();
	for (long i = 0; i < n; i++) {
		@autoreleasepool {
			MTLHeapDescriptor *hd = [MTLHeapDescriptor new];
			hd.size = 1u << 20;
			hd.storageMode = MTLStorageModeShared;
			hd.type = MTLHeapTypePlacement;
			id<MTLHeap> heap = [device newHeapWithDescriptor:hd];
			id<MTLBuffer> a = [heap newBufferWithLength:half options:shared offset:0];
			id<MTLBuffer> b = [heap newBufferWithLength:half options:shared offset:half];
			id<MTLBuffer> c = [heap newBufferWithLength:small options:shared offset:0];
			if (heap == nil || a == nil || b == nil || c == nil) {
				printf("round %ld: heap=%d a=%d b=%d c=%d\n", i, heap != nil, a != nil, b != nil,
				       c != nil);
				return fail("no Shared heap or placed buffer");
			}
			uint8_t *pa = (uint8_t *)[a contents], *pb = (uint8_t *)[b contents];
			uint8_t *pc = (uint8_t *)[c contents];
			if (pa == NULL || pc != pa || pb != pa + half) {
				printf("round %ld: contents a=%p b=%p c=%p\n", i, pa, pb, pc);
				return fail("placed buffers are not the heap's bytes at their offsets");
			}
			if (b.gpuAddress != a.gpuAddress + half || c.gpuAddress != a.gpuAddress ||
			    a.gpuAddress == 0) {
				printf("round %ld: gpuAddress a=0x%llx b=0x%llx c=0x%llx\n", i, a.gpuAddress,
				       b.gpuAddress, c.gpuAddress);
				return fail("placed buffers are not at the heap's address plus their offsets");
			}
			const uint8_t pattern = (uint8_t)(i * 13 + 5), fill = (uint8_t)(i * 7 + 3);
			memset(pa, pattern, half);
			if (pc[small - 1] != pattern)
				return fail("the CPU's write through one placed buffer is not another's");
			struct watch *w = watch_start("a shared heap command buffer");
			id<MTLCommandBuffer> cb = [queue commandBuffer];
			id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
			[blit copyFromBuffer:a sourceOffset:0 toBuffer:b destinationOffset:0 size:half];
			[blit fillBuffer:c range:NSMakeRange(small / 2, small / 2) value:fill];
			[blit endEncoding];
			[cb commit];
			[cb waitUntilCompleted];
			atomic_store(&w->done, 1);
			if (cb.status != MTLCommandBufferStatusCompleted)
				return fail("a shared heap command buffer failed");
			long wrong = 0;
			for (NSUInteger k = 0; k < half; k++)
				wrong += pb[k] != pattern;
			for (NSUInteger k = 0; k < small; k++)
				wrong += pa[k] != (k < small / 2 ? pattern : fill);
			if (wrong) {
				printf("round %ld: %ld wrong bytes (b[0]=0x%02x a[0x800]=0x%02x)\n", i, wrong,
				       pb[0], pa[small / 2]);
				return fail("the GPU's reads and writes of the heap's bytes are not the CPU's");
			}
		}
	}
	printf("sharedheap: %ld rounds in %.0f ms\n", n, now_ms() - start);
	return 0;
}

// One round of `autoheap` in a heap of `mode`.
static int auto_heap_round(id<MTLDevice> device, id<MTLCommandQueue> queue, MTLStorageMode mode,
                           long i)
{
	const MTLResourceOptions o = (MTLResourceOptions)mode << MTLResourceStorageModeShift;
	const MTLResourceOptions shared = MTLResourceStorageModeShared;
	MTLHeapDescriptor *hd = [MTLHeapDescriptor new];
	hd.size = 1000000;
	hd.storageMode = mode;
	hd.type = MTLHeapTypeAutomatic;
	id<MTLHeap> heap = [device newHeapWithDescriptor:hd];
	if (heap == nil)
		return fail("no Automatic heap");
	if (heap.size != 1015808 || heap.currentAllocatedSize != 1015808 || heap.usedSize != 0 ||
	    [heap maxAvailableSizeWithAlignment:256] != 1015808) {
		printf("round %ld: size %lu allocated %lu used %lu available %lu\n", i,
		       (unsigned long)heap.size, (unsigned long)heap.currentAllocatedSize,
		       (unsigned long)heap.usedSize,
		       (unsigned long)[heap maxAvailableSizeWithAlignment:256]);
		return fail("a fresh Automatic heap's sizes are not Metal's");
	}
	id<MTLBuffer> b1 = [heap newBufferWithLength:100 options:o];
	id<MTLBuffer> b2 = [heap newBufferWithLength:100 options:o];
	id<MTLBuffer> b3 = [heap newBufferWithLength:70000 options:o];
	if (b1 == nil || b2 == nil || b3 == nil)
		return fail("no buffer of an Automatic heap");
	const uint64_t base = b1.gpuAddress;
	if (b2.gpuAddress != base + 256 || b3.gpuAddress != base + 512 || b1.heapOffset != 0 ||
	    b3.heapOffset != 0 || b3.heap != heap) {
		printf("round %ld: at +0x%llx +0x%llx, heapOffset %lu\n", i, b2.gpuAddress - base,
		       b3.gpuAddress - base, (unsigned long)b3.heapOffset);
		return fail("an Automatic heap's buffers are not where Metal places them");
	}
	// In a pool of its own: Metal's contents keeps its buffer in the pool (measured: M3 Max,
	// macOS 27), which would hold the first buffer's bytes past its letting go below.
	ptrdiff_t apart = 512;
	@autoreleasepool {
		if (mode == MTLStorageModeShared)
			apart = (uint8_t *)[b3 contents] - (uint8_t *)[b1 contents];
	}
	if (apart != 512)
		return fail("an Automatic Shared heap's buffers are not its bytes at their offsets");
	if (heap.usedSize != 70200 || [heap maxAvailableSizeWithAlignment:256] != 945152 ||
	    [heap maxAvailableSizeWithAlignment:65536] != 884736) {
		printf("round %ld: used %lu available %lu / %lu\n", i, (unsigned long)heap.usedSize,
		       (unsigned long)[heap maxAvailableSizeWithAlignment:256],
		       (unsigned long)[heap maxAvailableSizeWithAlignment:65536]);
		return fail("an Automatic heap's used and available bytes are not Metal's");
	}
	MTLTextureDescriptor *td =
	    [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
	                                                       width:256
	                                                      height:256
	                                                   mipmapped:NO];
	td.storageMode = mode;
	td.usage = MTLTextureUsageShaderRead;
	const MTLSizeAndAlign sa = [device heapTextureSizeAndAlignWithDescriptor:td];
	id<MTLTexture> t = [heap newTextureWithDescriptor:td];
	if (i == 0)
		printf("autoheap: a 256 x 256 RGBA8 texture takes %lu bytes at %lu\n",
		       (unsigned long)sa.size, (unsigned long)sa.align);
	if (t == nil || sa.size < 256 * 256 * 4 || t.allocatedSize != sa.size ||
	    heap.usedSize != 70200 + sa.size || t.heapOffset != 0) {
		printf("round %ld: texture %d, size %lu, allocated %lu, used %lu\n", i, t != nil,
		       (unsigned long)sa.size, (unsigned long)t.allocatedSize,
		       (unsigned long)heap.usedSize);
		return fail("an Automatic heap's texture does not take its size");
	}
	const NSUInteger row = 256 * 4, bytes = row * 256;
	id<MTLBuffer> pattern = [device newBufferWithLength:bytes options:shared];
	id<MTLBuffer> back = [device newBufferWithLength:bytes + 70200 options:shared];
	uint8_t *pp = (uint8_t *)[pattern contents], *pb = (uint8_t *)[back contents];
	for (NSUInteger k = 0; k < bytes; k++)
		pp[k] = (uint8_t)(k * 31 + i);
	const uint8_t f1 = (uint8_t)(i + 1), f2 = (uint8_t)(i + 2), f3 = (uint8_t)(i + 3);
	struct watch *w = watch_start("an automatic heap command buffer");
	id<MTLCommandBuffer> cb = [queue commandBuffer];
	id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
	[blit fillBuffer:b1 range:NSMakeRange(0, 100) value:f1];
	[blit fillBuffer:b2 range:NSMakeRange(0, 100) value:f2];
	[blit fillBuffer:b3 range:NSMakeRange(0, 70000) value:f3];
	[blit copyFromBuffer:pattern
	           sourceOffset:0
	      sourceBytesPerRow:row
	    sourceBytesPerImage:bytes
	             sourceSize:MTLSizeMake(256, 256, 1)
	              toTexture:t
	       destinationSlice:0
	       destinationLevel:0
	      destinationOrigin:MTLOriginMake(0, 0, 0)];
	[blit copyFromTexture:t
	                 sourceSlice:0
	                 sourceLevel:0
	                sourceOrigin:MTLOriginMake(0, 0, 0)
	                  sourceSize:MTLSizeMake(256, 256, 1)
	                    toBuffer:back
	           destinationOffset:0
	      destinationBytesPerRow:row
	    destinationBytesPerImage:bytes];
	[blit copyFromBuffer:b1 sourceOffset:0 toBuffer:back destinationOffset:bytes size:100];
	[blit copyFromBuffer:b2 sourceOffset:0 toBuffer:back destinationOffset:bytes + 100 size:100];
	[blit copyFromBuffer:b3 sourceOffset:0 toBuffer:back destinationOffset:bytes + 200 size:70000];
	[blit endEncoding];
	[cb commit];
	[cb waitUntilCompleted];
	atomic_store(&w->done, 1);
	if (cb.status != MTLCommandBufferStatusCompleted)
		return fail("an automatic heap command buffer failed");
	long wrong = 0;
	for (NSUInteger k = 0; k < bytes; k++)
		wrong += pb[k] != pp[k];
	for (NSUInteger k = 0; k < 70200; k++)
		wrong += pb[bytes + k] != (k < 100 ? f1 : k < 200 ? f2 : f3);
	if (wrong) {
		printf("round %ld: %ld wrong bytes\n", i, wrong);
		return fail("an Automatic heap's resources do not keep their own bytes");
	}
	cb = nil;
	b1 = nil;
	[b2 makeAliasable];
	// The command buffer lets its resources go as it completes, which may come just after the
	// wait returns (Metal's does so on its completion thread): give it a second.
	for (int tries = 0; tries < 100 && heap.usedSize != 70000 + sa.size; tries++)
		usleep(10000);
	if (heap.usedSize != 70000 + sa.size || !b2.isAliasable) {
		printf("round %ld: used %lu\n", i, (unsigned long)heap.usedSize);
		return fail("a buffer let go or made aliasable keeps its bytes of its heap");
	}
	id<MTLBuffer> b4 = [heap newBufferWithLength:50 options:o];
	if (b4 == nil || b4.gpuAddress != base)
		return fail("a buffer after the first is let go is not at the heap's first byte");
	return 0;
}

static int auto_heap(id<MTLDevice> device, id<MTLCommandQueue> queue, long n)
{
	double start = now_ms();
	for (long i = 0; i < n; i++) {
		@autoreleasepool {
			int r = auto_heap_round(device, queue, i % 2 ? MTLStorageModeShared
			                                             : MTLStorageModePrivate,
			                        i);
			if (r != 0)
				return r;
		}
	}
	printf("autoheap: %ld rounds in %.0f ms\n", n, now_ms() - start);
	return 0;
}

// Waits for `cb` under a watchdog named `what`.
static void wait_watched(id<MTLCommandBuffer> cb, const char *what)
{
	struct watch *w = watch_start(what);
	[cb waitUntilCompleted];
	atomic_store(&w->done, 1);
}

static int no_copy_round(id<MTLDevice> device, id<MTLCommandQueue> queue, long i)
{
	const NSUInteger len1 = 0x14000, len2 = 0x11000, len3 = 4000, rows = 36, bpr = 1920;
	vm_address_t a1 = 0, a2 = 0;
	if (vm_allocate(mach_task_self(), &a1, len1, VM_FLAGS_ANYWHERE) != KERN_SUCCESS ||
	    vm_allocate(mach_task_self(), &a2, len2, VM_FLAGS_ANYWHERE) != KERN_SUCCESS)
		return fail("no app memory");
	uint8_t *p1 = (uint8_t *)a1, *p2 = (uint8_t *)a2;
	uint8_t *q = malloc(len3 + 64);
	uint8_t *p3 = (uint8_t *)(((uintptr_t)q + 15) & ~(uintptr_t)15) + 3;
	for (NSUInteger k = 0; k < len1; k++)
		p1[k] = (uint8_t)(k * 13 + i);
	for (NSUInteger k = 0; k < len2; k++)
		p2[k] = (uint8_t)(k * 7 + i);
	memset(q, 0x33, len3 + 64);
	__block atomic_int freed;
	atomic_init(&freed, 0);
	@autoreleasepool {
		id<MTLBuffer> b1 = [device newBufferWithBytesNoCopy:p1
		                                             length:len1
		                                            options:MTLResourceStorageModeShared
		                                        deallocator:^(void *m, NSUInteger l) {
			                                        vm_deallocate(mach_task_self(), (vm_address_t)m, l);
			                                        atomic_fetch_add(&freed, 1);
		                                        }];
		id<MTLBuffer> b2 = [device newBufferWithBytesNoCopy:p2
		                                             length:len2
		                                            options:MTLResourceStorageModeShared
		                                        deallocator:^(void *m, NSUInteger l) {
			                                        vm_deallocate(mach_task_self(), (vm_address_t)m, l);
			                                        atomic_fetch_add(&freed, 1);
		                                        }];
		id<MTLBuffer> b3 = [device newBufferWithBytesNoCopy:p3
		                                             length:len3
		                                            options:MTLResourceStorageModeShared
		                                        deallocator:^(void *m, NSUInteger l) {
			                                        atomic_fetch_add(&freed, 1);
		                                        }];
		if (b1 == nil || b2 == nil || b3 == nil)
			return fail("nil buffer from memory");
		if (b1.contents != p1 || b2.contents != p2 || b3.contents != p3 || b1.length != len1)
			return fail("a no-copy buffer's contents are not the app's bytes");
		MTLTextureDescriptor *td =
		    [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
		                                                       width:468
		                                                      height:rows
		                                                   mipmapped:NO];
		td.storageMode = MTLStorageModeShared;
		td.usage = MTLTextureUsageShaderRead;
		id<MTLTexture> t2 = [b2 newTextureWithDescriptor:td offset:0x100 bytesPerRow:bpr];
		if (t2 == nil)
			return fail("no texture over a no-copy buffer");
		id<MTLBuffer> out1 = [device newBufferWithLength:len1 options:MTLResourceStorageModeShared];
		id<MTLBuffer> out2 = [device newBufferWithLength:468 * 4 * rows
		                                         options:MTLResourceStorageModeShared];
		id<MTLCommandBuffer> cb = [queue commandBuffer];
		id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
		[blit copyFromBuffer:b1 sourceOffset:0 toBuffer:out1 destinationOffset:0 size:len1];
		[blit copyFromTexture:t2
		                 sourceSlice:0
		                 sourceLevel:0
		                sourceOrigin:MTLOriginMake(0, 0, 0)
		                  sourceSize:MTLSizeMake(468, rows, 1)
		                    toBuffer:out2
		           destinationOffset:0
		      destinationBytesPerRow:468 * 4
		    destinationBytesPerImage:468 * 4 * rows];
		[blit fillBuffer:b1 range:NSMakeRange(0x8000, 0x100) value:0xee];
		[blit fillBuffer:b3 range:NSMakeRange(0, len3) value:0x5a];
		[blit endEncoding];
		[cb commit];
		wait_watched(cb, "the no-copy copies");
		if (cb.status != MTLCommandBufferStatusCompleted)
			return fail("the no-copy command buffer failed");
		const uint8_t *o1 = out1.contents, *o2 = out2.contents;
		long wrong = 0;
		for (NSUInteger k = 0; k < len1; k++)
			wrong += o1[k] != (uint8_t)(k * 13 + i);
		for (NSUInteger y = 0; y < rows; y++)
			for (NSUInteger x = 0; x < 468 * 4; x++)
				wrong += o2[y * 468 * 4 + x] != (uint8_t)((0x100 + y * bpr + x) * 7 + i);
		if (wrong) {
			printf("round %ld: %ld wrong bytes read\n", i, wrong);
			return fail("the GPU does not read the app's bytes under a no-copy buffer");
		}
		for (NSUInteger k = 0; k < len1; k++)
			wrong += p1[k] != (k >= 0x8000 && k < 0x8100 ? 0xee : (uint8_t)(k * 13 + i));
		for (NSUInteger k = 0; k < len3 + 64; k++) {
			const uint8_t *at = q + k;
			wrong += *at != (at >= p3 && at < p3 + len3 ? 0x5a : 0x33);
		}
		if (wrong) {
			printf("round %ld: %ld wrong bytes written\n", i, wrong);
			return fail("the app's memory does not hold the GPU's writes alone");
		}
		for (NSUInteger k = 0; k < len1; k++)
			p1[k] = (uint8_t)(k * 5 + i + 1);
		cb = [queue commandBuffer];
		blit = [cb blitCommandEncoder];
		[blit copyFromBuffer:b1 sourceOffset:0 toBuffer:out1 destinationOffset:0 size:len1];
		[blit endEncoding];
		[cb commit];
		wait_watched(cb, "the second no-copy copy");
		for (NSUInteger k = 0; k < len1; k++)
			wrong += o1[k] != (uint8_t)(k * 5 + i + 1);
		if (wrong)
			return fail("a second commit does not take the CPU's new bytes");
		if (atomic_load(&freed) != 0)
			return fail("a deallocator ran while its buffer lived");
		cb = nil;
	}
	for (int tries = 0; tries < 100 && atomic_load(&freed) != 3; tries++)
		usleep(10000);
	if (atomic_load(&freed) != 3) {
		printf("round %ld: %d deallocators ran\n", i, atomic_load(&freed));
		return fail("a no-copy buffer's deallocator never ran");
	}
	free(q);
	vm_address_t holed = 0;
	vm_allocate(mach_task_self(), &holed, 0x8000, VM_FLAGS_ANYWHERE);
	vm_deallocate(mach_task_self(), holed + 0x4000, 0x4000);
	@autoreleasepool {
		if ([device newBufferWithBytesNoCopy:(void *)holed length:0x8000 options:0 deallocator:nil] !=
		        nil ||
		    [device newBufferWithBytesNoCopy:(void *)holed length:0 options:0 deallocator:nil] != nil)
			return fail("a no-copy buffer of no bytes or of unmapped ones");
	}
	vm_deallocate(mach_task_self(), holed, 0x4000);
	return 0;
}

static int no_copy(id<MTLDevice> device, id<MTLCommandQueue> queue, long n)
{
	double start = now_ms();
	for (long i = 0; i < n; i++) {
		@autoreleasepool {
			int r = no_copy_round(device, queue, i);
			if (r != 0)
				return r;
		}
	}
	printf("nocopy: %ld rounds in %.0f ms\n", n, now_ms() - start);
	return 0;
}

// One round of `events`: see the header.
static int events_round(id<MTLDevice> device, id<MTLCommandQueue> first, id<MTLCommandQueue> second,
                        dispatch_queue_t listenerQueue, long i)
{
	const NSUInteger len = 4096;
	const uint8_t fill = (uint8_t)(0x10 + (i % 200));
	id<MTLBuffer> src = [device newBufferWithLength:len options:MTLResourceStorageModeShared];
	id<MTLBuffer> dst = [device newBufferWithLength:len options:MTLResourceStorageModeShared];
	id<MTLBuffer> late = [device newBufferWithLength:len options:MTLResourceStorageModeShared];
	id<MTLSharedEvent> e = [device newSharedEvent];
	id<MTLEvent> plain = [device newEvent];
	if (src == nil || dst == nil || late == nil || e == nil || plain == nil)
		return fail("events: no buffers or events");
	memset(src.contents, 0, len);
	memset(dst.contents, 0xee, len);
	memset(late.contents, 0, len);

	// The second queue's copy waits for the first queue's fill, committed after it.
	struct watch *w = watch_start("events: the copy that waits for a shared event");
	id<MTLCommandBuffer> copy = [second commandBuffer];
	[copy encodeWaitForEvent:e value:1];
	id<MTLBlitCommandEncoder> b = [copy blitCommandEncoder];
	[b copyFromBuffer:src sourceOffset:0 toBuffer:dst destinationOffset:0 size:len];
	[b endEncoding];
	[copy commit];
	usleep(1000);
	id<MTLCommandBuffer> fillCb = [first commandBuffer];
	b = [fillCb blitCommandEncoder];
	[b fillBuffer:src range:NSMakeRange(0, len) value:fill];
	[b endEncoding];
	[fillCb encodeSignalEvent:e value:1];
	[fillCb commit];
	[fillCb waitUntilCompleted];
	if (e.signaledValue != 1) {
		printf("events: value %llu once the signalling buffer completed\n", e.signaledValue);
		return fail("events: a shared event's signal is not seen by its buffer's completion");
	}
	[copy waitUntilCompleted];
	atomic_store(&w->done, 1);
	for (NSUInteger k = 0; k < len; k++) {
		if (((uint8_t *)dst.contents)[k] != fill) {
			printf("events: byte %lu is %#x, not %#x\n", (unsigned long)k,
			       ((uint8_t *)dst.contents)[k], fill);
			return fail("events: a copy ran before the signal it waited for");
		}
	}

	// A listener at 2, on its own queue, with 2.
	MTLSharedEventListener *listener =
	    [[MTLSharedEventListener alloc] initWithDispatchQueue:listenerQueue];
	dispatch_semaphore_t heard = dispatch_semaphore_create(0);
	__block uint64_t heardValue = 0;
	__block int onQueue = 0;
	static char key;
	[e notifyListener:listener
	          atValue:2
	            block:^(id<MTLSharedEvent> ev, uint64_t v) {
		            (void)ev;
		            heardValue = v;
		            onQueue = dispatch_get_specific(&key) == &key;
		            dispatch_semaphore_signal(heard);
	            }];
	dispatch_queue_set_specific(listenerQueue, &key, &key, NULL);
	id<MTLCommandBuffer> two = [first commandBuffer];
	[two encodeSignalEvent:e value:2];
	[two commit];
	if (dispatch_semaphore_wait(heard, dispatch_time(DISPATCH_TIME_NOW, 20 * NSEC_PER_SEC)) != 0)
		return fail("events: a listener never ran");
	if (heardValue != 2 || !onQueue) {
		printf("events: listener value %llu, on its queue %d\n", heardValue, onQueue);
		return fail("events: a listener ran with another value or off its queue");
	}

	// A buffer waiting for 3 runs once the CPU sets 3.
	w = watch_start("events: the buffer the CPU's set releases");
	id<MTLCommandBuffer> held = [second commandBuffer];
	[held encodeWaitForEvent:e value:3];
	b = [held blitCommandEncoder];
	[b fillBuffer:late range:NSMakeRange(0, len) value:0x5a];
	[b endEncoding];
	[held commit];
	usleep(2000);
	if (held.status >= MTLCommandBufferStatusCompleted)
		return fail("events: a buffer ran before the value it waited for");
	e.signaledValue = 3;
	[held waitUntilCompleted];
	atomic_store(&w->done, 1);
	if (((uint8_t *)late.contents)[0] != 0x5a || ((uint8_t *)late.contents)[len - 1] != 0x5a)
		return fail("events: the released buffer did not run");
	e.signaledValue = 1;
	if (e.signaledValue != 3)
		return fail("events: a set to a lower value moved the value");

	// A plain event orders a copy across the queues too.
	memset(dst.contents, 0xee, len);
	w = watch_start("events: the copy that waits for a plain event");
	copy = [second commandBuffer];
	[copy encodeWaitForEvent:plain value:1];
	b = [copy blitCommandEncoder];
	[b copyFromBuffer:src sourceOffset:0 toBuffer:dst destinationOffset:0 size:len];
	[b endEncoding];
	[copy commit];
	usleep(1000);
	fillCb = [first commandBuffer];
	b = [fillCb blitCommandEncoder];
	[b fillBuffer:src range:NSMakeRange(0, len) value:(uint8_t)(fill + 1)];
	[b endEncoding];
	[fillCb encodeSignalEvent:plain value:1];
	[fillCb commit];
	[copy waitUntilCompleted];
	atomic_store(&w->done, 1);
	if (((uint8_t *)dst.contents)[0] != (uint8_t)(fill + 1) ||
	    ((uint8_t *)dst.contents)[len - 1] != (uint8_t)(fill + 1))
		return fail("events: a copy ran before the plain event it waited for");
	return 0;
}

static int events(id<MTLDevice> device, id<MTLCommandQueue> queue, long n)
{
	id<MTLCommandQueue> second = [device newCommandQueue];
	if (second == nil)
		return fail("events: no second queue");
	dispatch_queue_t listenerQueue = dispatch_queue_create("tex_probe.listener", DISPATCH_QUEUE_SERIAL);
	double start = now_ms();
	for (long i = 0; i < n; i++) {
		@autoreleasepool {
			int r = events_round(device, queue, second, listenerQueue, i);
			if (r != 0)
				return r;
		}
	}
	printf("events: %ld rounds in %.0f ms\n", n, now_ms() - start);
	return 0;
}

int main(int argc, char **argv)
{
	@autoreleasepool {
		int handler = argc >= 2 && strcmp(argv[1], "handler") == 0;
		int big = argc >= 2 && strcmp(argv[1], "big") == 0;
		int churning = argc >= 2 && strcmp(argv[1], "churn") == 0;
		int everyKind = argc >= 2 && strcmp(argv[1], "kinds") == 0;
		int sharedHeap = argc >= 2 && strcmp(argv[1], "sharedheap") == 0;
		int autoHeap = argc >= 2 && strcmp(argv[1], "autoheap") == 0;
		int noCopy = argc >= 2 && strcmp(argv[1], "nocopy") == 0;
		int withEvents = argc >= 2 && strcmp(argv[1], "events") == 0;
		if (argc < 2 ||
		    (strcmp(argv[1], "copy") != 0 && strcmp(argv[1], "hold") != 0 && !handler && !big &&
		     !churning && !everyKind && !sharedHeap && !autoHeap && !noCopy && !withEvents) ||
		    (strcmp(argv[1], "copy") != 0 && argc < 3)) {
			fprintf(stderr,
			        "usage: tex_probe copy | hold N | handler N | big N | churn N | kinds N | "
			        "sharedheap N | autoheap N | nocopy N | events N\n");
			return 2;
		}
		id<MTLDevice> device = MTLCreateSystemDefaultDevice();
		if (device == nil)
			return fail("no Metal device");
		printf("device=%s\n", [[device name] UTF8String]);
		id<MTLCommandQueue> queue = [device newCommandQueue];
		if (queue == nil)
			return fail("no queue");
		NSMutableArray<id<MTLTexture>> *held = [NSMutableArray array];
		if (strcmp(argv[1], "hold") == 0) {
			long n = strtol(argv[2], NULL, 10);
			MTLTextureDescriptor *d =
			    [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
			                                                       width:4
			                                                      height:4
			                                                   mipmapped:NO];
			d.storageMode = MTLStorageModePrivate;
			d.usage = MTLTextureUsageShaderRead;
			unsigned long lo = ~0ul, hi = 0;
			uint8_t *seen = calloc(kLimit, 1);
			int distinct = 1;
			for (long i = 0; i < n; i++) {
				id<MTLTexture> t = [device newTextureWithDescriptor:d];
				if (t == nil)
					break;
				unsigned long id = (unsigned long)t.gpuResourceID._impl;
				if (id == 0 || id >= kLimit) {
					free(seen);
					printf("texture %ld: gpuResourceID=%lu\n", i, id);
					return fail("a resource id is 0 or past the limit");
				}
				if (seen[id])
					distinct = 0;
				seen[id] = 1;
				lo = id < lo ? id : lo;
				hi = id > hi ? id : hi;
				[held addObject:t];
			}
			free(seen);
			printf("held=%lu of %ld, gpuResourceID %lu to %lu, distinct=%d\n",
			       (unsigned long)held.count, n, held.count ? lo : 0ul, hi, distinct);
			if (!distinct)
				return fail("two live textures answer one resource id");
			// Every id held: give one back so the copy's texture has one, as an app would.
			if ((long)held.count < n && held.count > 0)
				[held removeLastObject];
		}
		if (big || churning || everyKind || sharedHeap || autoHeap || noCopy || withEvents) {
			long n = strtol(argv[2], NULL, 10);
			int r = big          ? one_big_buffer(device, queue, n)
			        : churning   ? churn(device, queue, n)
			        : sharedHeap ? shared_heap(device, queue, n)
			        : autoHeap   ? auto_heap(device, queue, n)
			        : noCopy     ? no_copy(device, queue, n)
			        : withEvents ? events(device, queue, n)
			                     : kinds(device, n);
			if (r != 0)
				return r;
			printf("TEX OK\n");
			return 0;
		}
		int r = copy_through_texture(device, queue, handler);
		if (r != 0)
			return r;
		if (handler) {
			r = handlers_in_order(device, queue, strtol(argv[2], NULL, 10));
			if (r != 0)
				return r;
			struct rusage before, after;
			getrusage(RUSAGE_SELF, &before);
			sleep(2);
			getrusage(RUSAGE_SELF, &after);
			long us = (after.ru_utime.tv_sec - before.ru_utime.tv_sec) * 1000000L +
			          (after.ru_utime.tv_usec - before.ru_utime.tv_usec) +
			          (after.ru_stime.tv_sec - before.ru_stime.tv_sec) * 1000000L +
			          (after.ru_stime.tv_usec - before.ru_stime.tv_usec);
			printf("idle cpu=%ld us over 2 s\n", us);
			if (us > 20000)
				return fail("the process uses CPU while idle");
		}
		printf("TEX OK\n");
	}
	return 0;
}
