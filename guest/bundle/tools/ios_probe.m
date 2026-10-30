// ios_probe: IOSurface-backed textures on the reims-graphics Metal device.
//   ios_probe [HOLD_SECONDS]   make a global 64x64 BGRA surface, CPU-fill it, wrap it in a
//                              texture on our device, check (A) the GPU sees the CPU's
//                              pixels, (B) a blit into the texture reaches the surface's
//                              pages, (C) a render-pass clear reaches them; print the id,
//                              then keep the surface alive HOLD_SECONDS for a reader.
//   (IOS_W/IOS_H set the surface size, 64x64 by default)
//   ios_probe read ID HEX      another process: look the surface up by id, lock it, check
//                              every pixel is HEX.
// Build: xcrun clang -arch x86_64 -fobjc-arc -framework Metal -framework IOSurface
//        -framework Foundation ios_probe.m -o ios_probe
#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <Metal/Metal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int W = 64, H = 64; // IOS_W / IOS_H override

static int checkSurface(IOSurfaceRef s, uint32_t want, const char *what)
{
	uint32_t seed = 0;
	if (IOSurfaceLock(s, kIOSurfaceLockReadOnly, &seed) != kIOReturnSuccess) {
		printf("%s: lock failed\n", what);
		return 1;
	}
	const uint8_t *base = IOSurfaceGetBaseAddress(s);
	size_t bpr = IOSurfaceGetBytesPerRow(s);
	int bad = 0;
	uint32_t first = 0;
	for (int y = 0; y < H; y++) {
		for (int x = 0; x < W; x++) {
			uint32_t v = ((const uint32_t *)(base + y * bpr))[x];
			if (v != want) {
				if (bad == 0) first = v;
				bad++;
			}
		}
	}
	IOSurfaceUnlock(s, kIOSurfaceLockReadOnly, &seed);
	printf("%s: %s (want 0x%08x, %d bad, first bad 0x%08x, seed %u)\n", what,
	       bad ? "FAIL" : "PASS", want, bad, first, seed);
	return bad ? 1 : 0;
}

static int checkBuffer(id<MTLBuffer> b, uint32_t want, const char *what)
{
	const uint32_t *p = b.contents;
	int bad = 0;
	uint32_t first = 0;
	for (int i = 0; i < W * H; i++) {
		if (p[i] != want) {
			if (bad == 0) first = p[i];
			bad++;
		}
	}
	printf("%s: %s (want 0x%08x, %d bad, first bad 0x%08x)\n", what, bad ? "FAIL" : "PASS",
	       want, bad, first);
	return bad ? 1 : 0;
}

static int waitCB(id<MTLCommandBuffer> cb, const char *what)
{
	[cb commit];
	[cb waitUntilCompleted];
	if (cb.status != MTLCommandBufferStatusCompleted) {
		printf("%s: command buffer status %ld error %s\n", what, (long)cb.status,
		       cb.error.description.UTF8String);
		return 1;
	}
	return 0;
}

int main(int argc, char **argv)
{
	setvbuf(stdout, NULL, _IONBF, 0);
	@autoreleasepool {
		if (argc >= 4 && strcmp(argv[1], "read") == 0) {
			IOSurfaceID sid = (IOSurfaceID)strtoul(argv[2], NULL, 0);
			uint32_t want = (uint32_t)strtoul(argv[3], NULL, 16);
			if (getenv("IOS_W")) W = atoi(getenv("IOS_W"));
			if (getenv("IOS_H")) H = atoi(getenv("IOS_H"));
			IOSurfaceRef s = IOSurfaceLookup(sid);
			if (s == NULL) {
				printf("reader: IOSurfaceLookup(%u) = NULL\n", sid);
				return 2;
			}
			int rc = checkSurface(s, want, "reader (second process, by id)");
			CFRelease(s);
			return rc;
		}
		int hold = argc >= 2 ? atoi(argv[1]) : 0;
		if (getenv("IOS_W")) W = atoi(getenv("IOS_W"));
		if (getenv("IOS_H")) H = atoi(getenv("IOS_H"));

		id<MTLDevice> dev = nil;
		for (id<MTLDevice> d in MTLCopyAllDevices()) {
			printf("device %s class %s\n", d.name.UTF8String, NSStringFromClass([d class]).UTF8String);
			if ([NSStringFromClass([d class]) isEqualToString:@"RgfxMetalDevice"]) dev = d;
		}
		if (dev == nil) {
			printf("no RgfxMetalDevice\n");
			return 2;
		}

		NSDictionary *props = @{
			(id)kIOSurfaceWidth : @(W),
			(id)kIOSurfaceHeight : @(H),
			(id)kIOSurfaceBytesPerElement : @4,
			(id)kIOSurfacePixelFormat : @((uint32_t)'BGRA'),
			(id)kIOSurfaceIsGlobal : @YES,
		};
		IOSurfaceRef s = IOSurfaceCreate((__bridge CFDictionaryRef)props);
		if (s == NULL) {
			printf("IOSurfaceCreate failed\n");
			return 2;
		}
		printf("surface id %u bpr %zu\n", IOSurfaceGetID(s), IOSurfaceGetBytesPerRow(s));

		// The CPU writes 0x11223344 everywhere.
		uint32_t seed = 0;
		IOSurfaceLock(s, 0, &seed);
		uint8_t *base = IOSurfaceGetBaseAddress(s);
		for (int y = 0; y < H; y++)
			for (int x = 0; x < W; x++)
				((uint32_t *)(base + y * IOSurfaceGetBytesPerRow(s)))[x] = 0x11223344;
		IOSurfaceUnlock(s, 0, &seed);

		MTLTextureDescriptor *td =
		    [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
		                                                       width:W
		                                                      height:H
		                                                   mipmapped:NO];
		td.usage = MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget;
		td.storageMode = MTLStorageModeManaged;
		id<MTLTexture> tex = [dev newTextureWithDescriptor:td iosurface:s plane:0];
		if (tex == nil) {
			printf("newTextureWithDescriptor:iosurface:plane: = nil\n");
			return 2;
		}
		printf("texture %s %lux%lu iosurface-match=%d plane=%lu\n",
		       NSStringFromClass([tex class]).UTF8String, (unsigned long)tex.width,
		       (unsigned long)tex.height, tex.iosurface == s, (unsigned long)tex.iosurfacePlane);

		id<MTLCommandQueue> q = [dev newCommandQueue];
		id<MTLBuffer> rb = [dev newBufferWithLength:W * H * 4 options:MTLResourceStorageModeShared];
		id<MTLBuffer> src = [dev newBufferWithLength:W * H * 4 options:MTLResourceStorageModeShared];
		int fails = 0;

		// (A) the GPU reads what the CPU wrote.
		{
			id<MTLCommandBuffer> cb = [q commandBuffer];
			id<MTLBlitCommandEncoder> b = [cb blitCommandEncoder];
			[b copyFromTexture:tex sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0, 0, 0)
			            sourceSize:MTLSizeMake(W, H, 1) toBuffer:rb destinationOffset:0
			destinationBytesPerRow:W * 4 destinationBytesPerImage:W * H * 4];
			[b endEncoding];
			fails += waitCB(cb, "A");
			fails += checkBuffer(rb, 0x11223344, "A (GPU reads the CPU's pixels)");
		}
		// (B) a blit into the texture reaches the surface's pages.
		{
			for (int i = 0; i < W * H; i++) ((uint32_t *)src.contents)[i] = 0xff3366cc;
			id<MTLCommandBuffer> cb = [q commandBuffer];
			id<MTLBlitCommandEncoder> b = [cb blitCommandEncoder];
			[b copyFromBuffer:src sourceOffset:0 sourceBytesPerRow:W * 4 sourceBytesPerImage:W * H * 4
			           sourceSize:MTLSizeMake(W, H, 1) toTexture:tex destinationSlice:0
			     destinationLevel:0 destinationOrigin:MTLOriginMake(0, 0, 0)];
			[b endEncoding];
			fails += waitCB(cb, "B");
			fails += checkSurface(s, 0xff3366cc, "B (blit result in the surface's pages)");
		}
		// (C) a render-pass clear reaches the surface's pages (green, BGRA bytes).
		{
			MTLRenderPassDescriptor *rp = [MTLRenderPassDescriptor renderPassDescriptor];
			rp.colorAttachments[0].texture = tex;
			rp.colorAttachments[0].loadAction = MTLLoadActionClear;
			rp.colorAttachments[0].storeAction = MTLStoreActionStore;
			rp.colorAttachments[0].clearColor = MTLClearColorMake(0, 1, 0, 1);
			id<MTLCommandBuffer> cb = [q commandBuffer];
			id<MTLRenderCommandEncoder> e = [cb renderCommandEncoderWithDescriptor:rp];
			[e endEncoding];
			fails += waitCB(cb, "C");
			fails += checkSurface(s, 0xff00ff00, "C (render clear in the surface's pages)");
		}
		// (D) a CPU write after GPU use is seen by the next GPU read.
		{
			IOSurfaceLock(s, 0, &seed);
			for (int y = 0; y < H; y++)
				for (int x = 0; x < W; x++)
					((uint32_t *)(base + y * IOSurfaceGetBytesPerRow(s)))[x] = 0x55667788;
			IOSurfaceUnlock(s, 0, &seed);
			id<MTLCommandBuffer> cb = [q commandBuffer];
			id<MTLBlitCommandEncoder> b = [cb blitCommandEncoder];
			[b copyFromTexture:tex sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0, 0, 0)
			            sourceSize:MTLSizeMake(W, H, 1) toBuffer:rb destinationOffset:0
			destinationBytesPerRow:W * 4 destinationBytesPerImage:W * H * 4];
			[b endEncoding];
			fails += waitCB(cb, "D");
			fails += checkBuffer(rb, 0x55667788, "D (GPU reads a later CPU write)");
		}
		// Leave the clear colour for the reader.
		{
			MTLRenderPassDescriptor *rp = [MTLRenderPassDescriptor renderPassDescriptor];
			rp.colorAttachments[0].texture = tex;
			rp.colorAttachments[0].loadAction = MTLLoadActionClear;
			rp.colorAttachments[0].storeAction = MTLStoreActionStore;
			rp.colorAttachments[0].clearColor = MTLClearColorMake(1, 0, 0, 1);
			id<MTLCommandBuffer> cb = [q commandBuffer];
			[[cb renderCommandEncoderWithDescriptor:rp] endEncoding];
			fails += waitCB(cb, "E");
			fails += checkSurface(s, 0xffff0000, "E (red clear, left for the reader)");
		}
		// IOS_BENCH=N: N full-surface render clears, each committed and waited for alone,
		// timed from commit to completion (the GPU's clear plus the write-back to the
		// surface's pages), as a compositor's frame pays it.
		// The same clears of a Private texture of the surface's size (no surface sync)
		// follow, for comparison.
		int bench = getenv("IOS_BENCH") ? atoi(getenv("IOS_BENCH")) : 0;
		MTLTextureDescriptor *pd =
		    [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
		                                                       width:W
		                                                      height:H
		                                                   mipmapped:NO];
		pd.usage = MTLTextureUsageRenderTarget;
		pd.storageMode = MTLStorageModePrivate;
		id<MTLTexture> priv = bench > 0 ? [dev newTextureWithDescriptor:pd] : nil;
		for (int pass = 0; pass < (bench > 0 ? 2 : 0); pass++) {
		double sum = 0, lo = 1e9, hi = 0;
		for (int i = 0; i < bench; i++) {
			MTLRenderPassDescriptor *rp = [MTLRenderPassDescriptor renderPassDescriptor];
			rp.colorAttachments[0].texture = pass == 0 ? tex : priv;
			rp.colorAttachments[0].loadAction = MTLLoadActionClear;
			rp.colorAttachments[0].storeAction = MTLStoreActionStore;
			rp.colorAttachments[0].clearColor = MTLClearColorMake(1, 0, 0, 1);
			id<MTLCommandBuffer> cb = [q commandBuffer];
			[[cb renderCommandEncoderWithDescriptor:rp] endEncoding];
			NSTimeInterval t0 = [NSDate timeIntervalSinceReferenceDate];
			[cb commit];
			[cb waitUntilCompleted];
			double ms = ([NSDate timeIntervalSinceReferenceDate] - t0) * 1e3;
			sum += ms;
			lo = ms < lo ? ms : lo;
			hi = ms > hi ? ms : hi;
		}
		printf("BENCH %s %dx%d bpr %zu: %d clears, commit->complete ms min %.3f avg %.3f max %.3f\n",
		       pass == 0 ? "iosurface" : "private", W, H, IOSurfaceGetBytesPerRow(s), bench, lo,
		       sum / bench, hi);
		if (pass == 0) fails += checkSurface(s, 0xffff0000, "BENCH (last clear in the pages)");
		}
		printf("RESULT %s (%d failures) SURFACE_ID=%u\n", fails ? "FAIL" : "PASS", fails,
		       IOSurfaceGetID(s));
		fflush(stdout);
		if (hold > 0) sleep(hold);
		CFRelease(s);
		return fails ? 1 : 0;
	}
}
