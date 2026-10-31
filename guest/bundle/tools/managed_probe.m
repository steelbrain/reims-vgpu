// managed_probe: Managed buffers and textures on the reims-graphics Metal device, in one process.
// Each step writes through the CPU copy or the GPU, and reads the other side through a GPU copy
// into a Shared buffer, or through the CPU copy after a synchronize:
//   1. a 4 KiB Managed buffer written by the CPU, didModifyRange: over all of it, then a blit
//      copies it into a Shared buffer, which must hold the bytes;
//   2. bytes 100..199 and 300..399 written again, didModifyRange: over the first span alone, then
//      copied again: the first span must be the new bytes; the second is reported (a GPU with
//      its own copy keeps the old bytes, one that shares the CPU's memory sees the new ones);
//   3. a blit fills bytes 0..255 of the buffer and synchronizes it: its contents must hold the
//      fill once the command buffer completes;
//   4. a 64 x 64 RGBA8 Managed texture of two levels: replaceRegion: of all of level 0 and of a
//      4 x 4 region of level 1, then a blit copies both levels into Shared buffers, which must
//      hold the bytes; getBytes: must read back the CPU's own writes;
//   5. a blit copies a Shared buffer's pattern into level 0 and synchronizes that level:
//      getBytes: must read the pattern once the command buffer completes.
// Prints "MANAGED OK" when every check passes, else "MANAGED FAIL: why" and exits 1. A command
// buffer 20 s late prints HANG and exits 3.
// Build: xcrun clang -arch x86_64 -fobjc-arc -framework Metal -framework Foundation
//        managed_probe.m -o managed_probe
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>
// The macOS 27 SDK deprecates Managed storage, which the x86-64 guests this probes still have.
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static void fail(const char *why)
{
	printf("MANAGED FAIL: %s\n", why);
	fflush(stdout);
	exit(1);
}

// Commits `cb` and waits for it, 20 s at most.
static void run(id<MTLCommandBuffer> cb, const char *what)
{
	dispatch_semaphore_t done = dispatch_semaphore_create(0);
	[cb addCompletedHandler:^(id<MTLCommandBuffer> b) {
	  (void)b;
	  dispatch_semaphore_signal(done);
	}];
	[cb commit];
	if (dispatch_semaphore_wait(done, dispatch_time(DISPATCH_TIME_NOW, 20 * NSEC_PER_SEC)) != 0) {
		printf("HANG: %s never completed\n", what);
		fflush(stdout);
		_exit(3);
	}
	if (cb.status != MTLCommandBufferStatusCompleted) {
		printf("MANAGED FAIL: %s ended with status %ld, error %s\n", what, (long)cb.status,
		       cb.error.localizedDescription.UTF8String);
		exit(1);
	}
}

// The first byte where `a` and `b` differ, or -1.
static long differs(const uint8_t *a, const uint8_t *b, size_t n)
{
	for (size_t i = 0; i < n; i++) {
		if (a[i] != b[i])
			return (long)i;
	}
	return -1;
}

int main(void)
{
	@autoreleasepool {
		id<MTLDevice> device = MTLCreateSystemDefaultDevice();
		if (device == nil)
			fail("no Metal device");
		printf("device %s, hasUnifiedMemory %d\n", device.name.UTF8String,
		       (int)device.hasUnifiedMemory);
		id<MTLCommandQueue> queue = [device newCommandQueue];
		if (queue == nil)
			fail("no queue");

		// 1. CPU writes, didModifyRange:, GPU copy out.
		const NSUInteger n = 4096;
		id<MTLBuffer> managed = [device newBufferWithLength:n options:MTLResourceStorageModeManaged];
		id<MTLBuffer> out = [device newBufferWithLength:n options:MTLResourceStorageModeShared];
		if (managed == nil || out == nil || managed.contents == NULL)
			fail("no Managed buffer");
		if (managed.storageMode != MTLStorageModeManaged)
			fail("the buffer is not Managed");
		uint8_t *cpu = managed.contents;
		uint8_t want[4096];
		for (NSUInteger i = 0; i < n; i++)
			want[i] = (uint8_t)(i * 7 + 3);
		memcpy(cpu, want, n);
		[managed didModifyRange:NSMakeRange(0, n)];
		id<MTLCommandBuffer> cb = [queue commandBuffer];
		id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
		[blit copyFromBuffer:managed sourceOffset:0 toBuffer:out destinationOffset:0 size:n];
		[blit endEncoding];
		run(cb, "the buffer's copy");
		long at = differs(out.contents, want, n);
		if (at >= 0) {
			printf("MANAGED FAIL: the GPU copy differs at byte %ld (0x%02x, not 0x%02x)\n", at,
			       ((uint8_t *)out.contents)[at], want[at]);
			return 1;
		}
		printf("1 buffer: the GPU sees the CPU's bytes after didModifyRange:\n");

		// 2. Two spans written, one named.
		memset(cpu + 100, 0xee, 100);
		memset(cpu + 300, 0xdd, 100);
		[managed didModifyRange:NSMakeRange(100, 100)];
		memset(want + 100, 0xee, 100);
		cb = [queue commandBuffer];
		blit = [cb blitCommandEncoder];
		[blit copyFromBuffer:managed sourceOffset:0 toBuffer:out destinationOffset:0 size:n];
		[blit endEncoding];
		run(cb, "the second copy");
		if (differs(out.contents, want, 300) >= 0)
			fail("the named span did not reach the GPU");
		const uint8_t *seen = (const uint8_t *)out.contents + 300;
		printf("2 buffer: the named span reached the GPU; the unnamed one %s\n",
		       seen[0] == 0xdd ? "did too (one copy)" : "did not (two copies)");

		// 3. GPU fill, synchronizeResource:, CPU read.
		cb = [queue commandBuffer];
		blit = [cb blitCommandEncoder];
		[blit fillBuffer:managed range:NSMakeRange(0, 256) value:0x5a];
		[blit synchronizeResource:managed];
		[blit endEncoding];
		run(cb, "the fill");
		for (int i = 0; i < 256; i++) {
			if (cpu[i] != 0x5a) {
				printf("MANAGED FAIL: contents byte %d is 0x%02x after the synchronize, not "
				       "0x5a\n",
				       i, cpu[i]);
				return 1;
			}
		}
		printf("3 buffer: the CPU sees the GPU's fill after synchronizeResource:\n");

		// 4. replaceRegion: of two levels, GPU copies out, getBytes: of the CPU's writes.
		MTLTextureDescriptor *desc =
		    [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
		                                                       width:64
		                                                      height:64
		                                                   mipmapped:YES];
		desc.mipmapLevelCount = 2;
		desc.storageMode = MTLStorageModeManaged;
		desc.usage = MTLTextureUsageShaderRead;
		id<MTLTexture> tex = [device newTextureWithDescriptor:desc];
		if (tex == nil || tex.storageMode != MTLStorageModeManaged)
			fail("no Managed texture");
		static uint8_t level0[64 * 64 * 4], quad[4 * 4 * 4];
		for (size_t i = 0; i < sizeof(level0); i++)
			level0[i] = (uint8_t)(i * 13 + 1);
		for (size_t i = 0; i < sizeof(quad); i++)
			quad[i] = (uint8_t)(0x80 | i);
		[tex replaceRegion:MTLRegionMake2D(0, 0, 64, 64)
		       mipmapLevel:0
		         withBytes:level0
		       bytesPerRow:64 * 4];
		[tex replaceRegion:MTLRegionMake2D(8, 8, 4, 4) mipmapLevel:1 withBytes:quad bytesPerRow:16];
		id<MTLBuffer> out0 = [device newBufferWithLength:sizeof(level0)
		                                         options:MTLResourceStorageModeShared];
		id<MTLBuffer> out1 = [device newBufferWithLength:32 * 32 * 4
		                                         options:MTLResourceStorageModeShared];
		cb = [queue commandBuffer];
		blit = [cb blitCommandEncoder];
		[blit copyFromTexture:tex
		                 sourceSlice:0
		                 sourceLevel:0
		                sourceOrigin:MTLOriginMake(0, 0, 0)
		                  sourceSize:MTLSizeMake(64, 64, 1)
		                    toBuffer:out0
		           destinationOffset:0
		      destinationBytesPerRow:64 * 4
		    destinationBytesPerImage:sizeof(level0)];
		[blit copyFromTexture:tex
		                 sourceSlice:0
		                 sourceLevel:1
		                sourceOrigin:MTLOriginMake(0, 0, 0)
		                  sourceSize:MTLSizeMake(32, 32, 1)
		                    toBuffer:out1
		           destinationOffset:0
		      destinationBytesPerRow:32 * 4
		    destinationBytesPerImage:32 * 32 * 4];
		[blit endEncoding];
		run(cb, "the texture's copy");
		at = differs(out0.contents, level0, sizeof(level0));
		if (at >= 0) {
			printf("MANAGED FAIL: level 0's GPU copy differs at byte %ld\n", at);
			return 1;
		}
		for (int r = 0; r < 4; r++) {
			const uint8_t *row = (const uint8_t *)out1.contents + (8 + r) * 32 * 4 + 8 * 4;
			if (differs(row, quad + r * 16, 16) >= 0) {
				printf("MANAGED FAIL: level 1's GPU copy differs in row %d of the region\n", r);
				return 1;
			}
		}
		static uint8_t back[64 * 64 * 4];
		[tex getBytes:back bytesPerRow:64 * 4 fromRegion:MTLRegionMake2D(0, 0, 64, 64) mipmapLevel:0];
		if (differs(back, level0, sizeof(back)) >= 0)
			fail("getBytes: did not read back the CPU's writes");
		printf("4 texture: the GPU sees replaceRegion: of two levels; getBytes: reads them\n");

		// 5. GPU write, synchronizeTexture:, getBytes:.
		id<MTLBuffer> src = [device newBufferWithLength:sizeof(level0)
		                                        options:MTLResourceStorageModeShared];
		uint8_t *pattern = src.contents;
		for (size_t i = 0; i < sizeof(level0); i++)
			pattern[i] = (uint8_t)(i * 5 + 11);
		cb = [queue commandBuffer];
		blit = [cb blitCommandEncoder];
		[blit copyFromBuffer:src
		           sourceOffset:0
		      sourceBytesPerRow:64 * 4
		    sourceBytesPerImage:sizeof(level0)
		             sourceSize:MTLSizeMake(64, 64, 1)
		              toTexture:tex
		       destinationSlice:0
		       destinationLevel:0
		      destinationOrigin:MTLOriginMake(0, 0, 0)];
		[blit synchronizeTexture:tex slice:0 level:0];
		[blit endEncoding];
		run(cb, "the texture's write");
		[tex getBytes:back bytesPerRow:64 * 4 fromRegion:MTLRegionMake2D(0, 0, 64, 64) mipmapLevel:0];
		at = differs(back, pattern, sizeof(back));
		if (at >= 0) {
			printf("MANAGED FAIL: getBytes: after synchronizeTexture: differs at byte %ld\n", at);
			return 1;
		}
		printf("5 texture: getBytes: sees the GPU's write after synchronizeTexture:\n");
		printf("MANAGED OK\n");
	}
	return 0;
}
