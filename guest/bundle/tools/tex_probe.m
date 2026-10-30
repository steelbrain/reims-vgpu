// tex_probe: textures on the reims-graphics Metal device, with their resource ids, in one process.
//   tex_probe copy      a 64 x 64 BGRA8 Private texture: one blit copies a Shared buffer's pattern
//                       into it and back out into a second buffer, which is checked byte for byte
//   tex_probe hold N    N 4 x 4 textures made and held at once, then the copy above: prints how
//                       many were made, and their lowest and highest gpuResourceID, which must be
//                       distinct, non-zero and below 16381 (the host device's limit)
// Prints "TEX OK" when every check passes, else "TEX FAIL: why" and exits 1. A completion 20 s
// late prints HANG and exits 3.
// Build: xcrun clang -arch x86_64 -fobjc-arc -framework Metal -framework Foundation
//        tex_probe.m -o tex_probe
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static const unsigned long kLimit = 16381;

static int fail(const char *why)
{
	printf("TEX FAIL: %s\n", why);
	return 1;
}

// The copy through a Private texture, checked byte for byte; its texture's resource id printed.
static int copy_through_texture(id<MTLDevice> device, id<MTLCommandQueue> queue)
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
	[cb addCompletedHandler:^(id<MTLCommandBuffer> b) {
	  (void)b;
	  dispatch_semaphore_signal(done);
	}];
	[cb commit];
	if (dispatch_semaphore_wait(done, dispatch_time(DISPATCH_TIME_NOW, 20 * NSEC_PER_SEC)) != 0) {
		printf("HANG: the copy's command buffer never completed\n");
		fflush(stdout);
		_exit(3);
	}
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

int main(int argc, char **argv)
{
	@autoreleasepool {
		if (argc < 2 || (strcmp(argv[1], "copy") != 0 && strcmp(argv[1], "hold") != 0) ||
		    (strcmp(argv[1], "hold") == 0 && argc < 3)) {
			fprintf(stderr, "usage: tex_probe copy | tex_probe hold N\n");
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
		int r = copy_through_texture(device, queue);
		if (r != 0)
			return r;
		printf("TEX OK\n");
	}
	return 0;
}
