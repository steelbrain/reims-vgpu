// io_probe: Metal IO (fast resource loading) in one process, compared with what Metal gives on an
// M3 Max (macOS 27).
//   io_probe [DIR]  writes its files into DIR (default: the executable's directory): one raw file of
//                   a known 256 KiB pattern, and the same bytes through MTLIOCreateCompressionContext
//                   in each compression method. It then:
//                   - opens each with the device's file handles, and answers what a missing file
//                     and a bad method give;
//                   - loads from each, in one IO command buffer per file, into a Shared buffer
//                     (loadBuffer), process memory (loadBytes) and a 64 x 64 RGBA8 texture
//                     (loadTexture, read back through a blit), with a shared event signalled at the
//                     end and the status copied into a buffer;
//                   - loads past a file's end, cancels a buffer before its commit, and waits on an
//                     event signalled from the CPU.
//                   It prints each answer and the first mismatching byte of each load.
// Prints "IO OK" when every answer and byte is Metal's, else "IO FAIL: why" and exits 1. A
// completion 20 s late prints HANG and exits 3.
// Build: xcrun clang -arch x86_64 -fobjc-arc -framework Metal -framework Foundation
//        io_probe.m -o io_probe
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { kFileBytes = 256 * 1024, kSide = 64, kTexBytes = kSide * kSide * 4 };

static int bad = 0;

static void check(BOOL ok, const char *what)
{
	if (!ok) {
		printf("  MISMATCH: %s\n", what);
		bad = 1;
	}
}

// The probe's file bytes: a pattern no compression method stores flat, and no run repeats.
static void pattern(uint8_t *p, size_t n)
{
	uint32_t x = 0x12345678u;
	for (size_t i = 0; i < n; i++) {
		x = x * 1664525u + 1013904223u;
		p[i] = (uint8_t)((x >> 24) ^ (i & 0xff ? 0 : 0x5a)) & (i % 3 ? 0xff : 0x0f);
	}
}

// The first byte where `a` and `b` differ, or -1.
static long firstDiff(const uint8_t *a, const uint8_t *b, size_t n)
{
	for (size_t i = 0; i < n; i++)
		if (a[i] != b[i])
			return (long)i;
	return -1;
}

static BOOL waitFor(id<MTLIOCommandBuffer> cb)
{
	dispatch_semaphore_t done = dispatch_semaphore_create(0);
	[cb addCompletedHandler:^(id<MTLIOCommandBuffer> b) {
		dispatch_semaphore_signal(done);
	}];
	[cb commit];
	return dispatch_semaphore_wait(done, dispatch_time(DISPATCH_TIME_NOW, 20 * NSEC_PER_SEC)) == 0;
}

static void printError(const char *what, NSError *e)
{
	printf("  %s: %s %ld\n", what, e ? e.domain.UTF8String : "no error", e ? (long)e.code : 0L);
}

int main(int argc, char **argv)
{
	@autoreleasepool {
		id<MTLDevice> d = MTLCreateSystemDefaultDevice();
		if (!d)
			return printf("IO FAIL: no device\n"), 1;
		NSString *dir = argc > 1 ? @(argv[1])
		                         : [@(argv[0]) stringByDeletingLastPathComponent];
		uint8_t *bytes = malloc(kFileBytes);
		pattern(bytes, kFileBytes);
		NSString *rawPath = [dir stringByAppendingPathComponent:@"io_probe.raw"];
		if (![[NSData dataWithBytesNoCopy:bytes length:kFileBytes freeWhenDone:NO]
		        writeToFile:rawPath
		         atomically:YES])
			return printf("IO FAIL: cannot write %s\n", rawPath.UTF8String), 1;

		// The queue.
		MTLIOCommandQueueDescriptor *qd = [MTLIOCommandQueueDescriptor new];
		printf("descriptor: maxCommandBufferCount %lu priority %ld type %ld maxCommandsInFlight "
		       "%lu allocator %s\n",
		       (unsigned long)qd.maxCommandBufferCount, (long)qd.priority, (long)qd.type,
		       (unsigned long)qd.maxCommandsInFlight, qd.scratchBufferAllocator ? "set" : "nil");
		check(qd.maxCommandBufferCount == 64 && qd.priority == MTLIOPriorityNormal &&
		          qd.type == MTLIOCommandQueueTypeConcurrent && qd.maxCommandsInFlight == 0 &&
		          qd.scratchBufferAllocator == nil,
		      "descriptor defaults");
		NSError *e = nil;
		id<MTLIOCommandQueue> q = [d newIOCommandQueueWithDescriptor:qd error:&e];
		printf("queue: %s, label %s\n", q ? "made" : "nil",
		       q.label ? q.label.UTF8String : "nil");
		if (!q) {
			printError("queue error", e);
			return printf("IO FAIL: no IO queue\n"), 1;
		}
		qd.type = MTLIOCommandQueueTypeSerial;
		qd.priority = MTLIOPriorityHigh;
		id<MTLIOCommandQueue> serial = [d newIOCommandQueueWithDescriptor:qd error:&e];
		check(serial != nil, "a serial, high-priority queue");

		// The handles: a missing file, and a raw file opened with a compression method.
		e = nil;
		NSURL *missing = [NSURL fileURLWithPath:[dir stringByAppendingPathComponent:@"io_probe.none"]];
		id<MTLIOFileHandle> none = [d newIOFileHandleWithURL:missing error:&e];
		printf("missing file: %s\n", none ? "a handle" : "nil");
		printError("missing file error", e);
		// Metal (M3 Max): nil, and Cocoa's file-not-found (NSFileReadNoSuchFileError, 260).
		check(none == nil && [e.domain isEqualToString:NSCocoaErrorDomain] &&
		          e.code == NSFileReadNoSuchFileError,
		      "a missing file answers nil and NSFileReadNoSuchFileError");
		e = nil;
		id<MTLIOFileHandle> rawAsLz4 = [d newIOFileHandleWithURL:[NSURL fileURLWithPath:rawPath]
		                                       compressionMethod:MTLIOCompressionMethodLZ4
		                                                   error:&e];
		printf("raw file as LZ4: %s\n", rawAsLz4 ? "a handle" : "nil");
		printError("raw file as LZ4 error", e);
		check(rawAsLz4 == nil && [e.domain isEqualToString:MTLIOErrorDomain] &&
		          e.code == MTLIOErrorInternal,
		      "a raw file opened as LZ4 answers nil and MTLIOErrorInternal");

		NSMutableArray<id<MTLIOFileHandle>> *handles = [NSMutableArray array];
		NSMutableArray<NSString *> *names = [NSMutableArray array];
		e = nil;
		id<MTLIOFileHandle> raw = [d newIOFileHandleWithURL:[NSURL fileURLWithPath:rawPath]
		                                              error:&e];
		check(raw != nil, "a raw handle");
		if (raw) {
			raw.label = @"raw";
			check([raw.label isEqualToString:@"raw"], "a handle keeps its label");
			[handles addObject:raw];
			[names addObject:@"raw"];
		}
		const char *methods[5] = {"zlib", "lzfse", "lz4", "lzma", "lzbitmap"};
		for (int m = 0; m < 5; m++) {
			NSString *path = [dir stringByAppendingPathComponent:
			                          [NSString stringWithFormat:@"io_probe.%s", methods[m]]];
			MTLIOCompressionContext c = MTLIOCreateCompressionContext(
			    path.fileSystemRepresentation, (MTLIOCompressionMethod)m,
			    MTLIOCompressionContextDefaultChunkSize());
			if (!c) {
				printf("  no compression context for %s\n", methods[m]);
				bad = 1;
				continue;
			}
			MTLIOCompressionContextAppendData(c, bytes, kFileBytes / 2);
			MTLIOCompressionContextAppendData(c, bytes + kFileBytes / 2, kFileBytes / 2);
			if (MTLIOFlushAndDestroyCompressionContext(c) != MTLIOCompressionStatusComplete) {
				printf("  %s: the compression did not complete\n", methods[m]);
				bad = 1;
				continue;
			}
			e = nil;
			id<MTLIOFileHandle> h = [d newIOFileHandleWithURL:[NSURL fileURLWithPath:path]
			                                compressionMethod:(MTLIOCompressionMethod)m
			                                            error:&e];
			check(h != nil, methods[m]);
			if (h) {
				[handles addObject:h];
				[names addObject:@(methods[m])];
			}
		}
		printf("default chunk size %zu\n", MTLIOCompressionContextDefaultChunkSize());

		// Loads from each handle: a buffer, process memory and a texture, an event at the end, and
		// the status copied out.
		id<MTLSharedEvent> ev = [d newSharedEvent];
		id<MTLCommandQueue> gq = [d newCommandQueue];
		MTLTextureDescriptor *td =
		    [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
		                                                       width:kSide
		                                                      height:kSide
		                                                   mipmapped:NO];
		td.storageMode = MTLStorageModePrivate;
		const size_t bufOffset = 4096, bufBytes = 100000, srcOffset = 12345;
		const size_t memBytes = 777, memSrc = 200001;
		const size_t texSrc = 65536;
		for (NSUInteger k = 0; k < handles.count; k++) {
			id<MTLIOFileHandle> h = handles[k];
			id<MTLBuffer> buf = [d newBufferWithLength:bufOffset + bufBytes + 64
			                                   options:MTLResourceStorageModeShared];
			memset(buf.contents, 0xee, buf.length);
			id<MTLBuffer> status = [d newBufferWithLength:16 options:MTLResourceStorageModeShared];
			memset(status.contents, 0xcc, 16);
			uint8_t mem[1024];
			memset(mem, 0xdd, sizeof(mem));
			id<MTLTexture> tex = [d newTextureWithDescriptor:td];
			id<MTLIOCommandBuffer> cb = [q commandBuffer];
			cb.label = @"probe";
			[cb loadBuffer:buf offset:bufOffset size:bufBytes sourceHandle:h sourceHandleOffset:srcOffset];
			[cb loadBytes:mem + 8 size:memBytes sourceHandle:h sourceHandleOffset:memSrc];
			[cb loadTexture:tex
			                  slice:0
			                  level:0
			                   size:MTLSizeMake(kSide, kSide, 1)
			      sourceBytesPerRow:kSide * 4
			    sourceBytesPerImage:kTexBytes
			      destinationOrigin:MTLOriginMake(0, 0, 0)
			           sourceHandle:h
			     sourceHandleOffset:texSrc];
			[cb addBarrier];
			[cb signalEvent:ev value:k + 1];
			[cb copyStatusToBuffer:status offset:8];
			if (!waitFor(cb))
				return printf("HANG\n"), 3;
			printf("%-8s status %ld error %s, event %llu, copied status %lld, label %s\n",
			       names[k].UTF8String, (long)cb.status,
			       cb.error ? cb.error.localizedDescription.UTF8String : "none",
			       ev.signaledValue, ((long long *)status.contents)[1],
			       cb.label.UTF8String);
			check(cb.status == MTLIOStatusComplete && cb.error == nil, "a load completes");
			check(ev.signaledValue == k + 1, "the event is signalled");
			check(((long long *)status.contents)[1] == MTLIOStatusComplete &&
			          ((uint8_t *)status.contents)[0] == 0xcc,
			      "the status is copied, 8 bytes at its offset");
			const uint8_t *b = buf.contents;
			const long db = firstDiff(b + bufOffset, bytes + srcOffset, bufBytes);
			check(db < 0, "loadBuffer's bytes");
			check(b[bufOffset - 1] == 0xee && b[bufOffset + bufBytes] == 0xee,
			      "loadBuffer writes only its range");
			const long dm = firstDiff(mem + 8, bytes + memSrc, memBytes);
			check(dm < 0 && mem[7] == 0xdd && mem[8 + memBytes] == 0xdd, "loadBytes's bytes");
			id<MTLBuffer> back = [d newBufferWithLength:kTexBytes options:MTLResourceStorageModeShared];
			id<MTLCommandBuffer> gcb = [gq commandBuffer];
			id<MTLBlitCommandEncoder> blit = [gcb blitCommandEncoder];
			[blit copyFromTexture:tex
			                 sourceSlice:0
			                 sourceLevel:0
			                sourceOrigin:MTLOriginMake(0, 0, 0)
			                  sourceSize:MTLSizeMake(kSide, kSide, 1)
			                    toBuffer:back
			           destinationOffset:0
			      destinationBytesPerRow:kSide * 4
			    destinationBytesPerImage:kTexBytes];
			[blit endEncoding];
			[gcb commit];
			[gcb waitUntilCompleted];
			const long dt = firstDiff(back.contents, bytes + texSrc, kTexBytes);
			check(dt < 0, "loadTexture's texels");
			printf("  first differences: buffer %ld, bytes %ld, texture %ld\n", db, dm, dt);
		}

		// A load past the end of the raw file; a buffer cancelled before its commit; a wait on an
		// event the CPU signals.
		if (raw) {
			id<MTLBuffer> buf = [d newBufferWithLength:4096 options:MTLResourceStorageModeShared];
			memset(buf.contents, 0xee, 4096);
			id<MTLIOCommandBuffer> cb = [q commandBuffer];
			[cb loadBuffer:buf offset:0 size:4096 sourceHandle:raw sourceHandleOffset:kFileBytes - 100];
			if (!waitFor(cb))
				return printf("HANG\n"), 3;
			const uint8_t *b = buf.contents;
			size_t tail = 0;
			while (tail < 4096 && b[100 + tail] == b[100])
				tail++;
			printf("past the end: status %ld, the file's last 100 bytes %s, then %zu bytes of "
			       "0x%02x\n",
			       (long)cb.status, firstDiff(b, bytes + kFileBytes - 100, 100) < 0 ? "match" : "differ",
			       tail > 3996 ? 3996 : tail, b[100]);
			printError("past the end error", cb.error);
			// Metal (M3 Max): it completes with no error.
			check(cb.status == MTLIOStatusComplete && cb.error == nil, "a load past the end completes");
			check(firstDiff(b, bytes + kFileBytes - 100, 100) < 0, "the file's tail is loaded");
			check(tail >= 3996 && b[100] == 0xee, "nothing is written past the file's end");

			id<MTLIOCommandBuffer> cancelled = [q commandBuffer];
			[cancelled loadBuffer:buf offset:0 size:16 sourceHandle:raw sourceHandleOffset:0];
			[cancelled tryCancel];
			if (!waitFor(cancelled))
				return printf("HANG\n"), 3;
			printf("cancelled before its commit: status %ld\n", (long)cancelled.status);
			printError("cancelled error", cancelled.error);
			// Metal (M3 Max): cancelled, with no error.
			check(cancelled.status == MTLIOStatusCancelled && cancelled.error == nil,
			      "a buffer cancelled before its commit is cancelled");

			id<MTLSharedEvent> gate = [d newSharedEvent];
			id<MTLIOCommandBuffer> waits = [q commandBuffer];
			[waits waitForEvent:gate value:5];
			[waits loadBuffer:buf offset:0 size:16 sourceHandle:raw sourceHandleOffset:0];
			dispatch_semaphore_t done = dispatch_semaphore_create(0);
			[waits addCompletedHandler:^(id<MTLIOCommandBuffer> b) {
				dispatch_semaphore_signal(done);
			}];
			[waits commit];
			const BOOL early =
			    dispatch_semaphore_wait(done, dispatch_time(DISPATCH_TIME_NOW, NSEC_PER_SEC / 2)) == 0;
			gate.signaledValue = 5;
			if (dispatch_semaphore_wait(done, dispatch_time(DISPATCH_TIME_NOW, 20 * NSEC_PER_SEC)))
				return printf("HANG\n"), 3;
			printf("waits for the event: %s, status %ld\n", early ? "no" : "yes", (long)waits.status);
			check(!early && waits.status == MTLIOStatusComplete, "a wait holds the buffer");
		}

		[[NSFileManager defaultManager] removeItemAtPath:rawPath error:nil];
		for (int m = 0; m < 5; m++)
			[[NSFileManager defaultManager]
			    removeItemAtPath:[dir stringByAppendingPathComponent:
			                              [NSString stringWithFormat:@"io_probe.%s", methods[m]]]
			               error:nil];
		free(bytes);
		if (bad)
			return printf("IO FAIL: an answer or a byte is not Metal's\n"), 1;
		printf("IO OK\n");
		return 0;
	}
}
