// mps_probe: Metal Performance Shaders image filters on the default Metal device, as a window
// server's effects use them: a Gaussian blur, a Lanczos scale and an MPSImage read back, on
// BGRA8Unorm and RGBA16Float textures in Private, Shared and Managed storage. Each texture holds
// one color, and a blur or a scale of one color is that color, so every pixel of each result is
// checked against it.
//   mps_probe             for each format and storage mode, runs each step in a command buffer of
//                         its own and checks its pixels:
//                           blur      MPSImageGaussianBlur (sigma 3), source to destination;
//                           inplace   the same blur in place, with a copy allocator;
//                           lanczos   MPSImageLanczosScale to half size;
//                           readback  an MPSImage on the source, read with readBytes:;
//                           temporary a blur into an MPSTemporaryImage, then one out of it.
//                         A step that raises prints the exception's name and reason, which names
//                         the selector and the receiver's class of an unrecognized selector.
//                         Exits 0 when every step ran and matched.
//   mps_probe --discover  as above, but after an unrecognized selector it adds to the receiver's
//                         class a method that answers 0 (nil) and runs the step again, so one run
//                         names every selector a step sends after the first. PROBE ONLY: the
//                         stubs' answers are not Metal's, so a step that passes with stubs shows
//                         which selectors it sends, not that it works. Each stub is printed.
// Run it with OS_ACTIVITY_DT_MODE=YES to put the device's own NSLog lines (its "RGFX:
// unrecognized" lines among them) on stderr.
//
// Build: xcrun clang -arch x86_64 -fobjc-arc -framework Metal -framework MetalPerformanceShaders
//        -framework Foundation tools/mps_probe.m -o mps_probe

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include <math.h>
#include <objc/runtime.h>

// The guest is an x86-64 Mac, where Managed storage is a mode of its own.
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { kSide = 64, kMaxStubs = 32 };

static BOOL g_discover;
static int g_stubCount;
static char g_stubs[kMaxStubs][160];

typedef BOOL (^Step)(id<MTLCommandBuffer> cb);

// The fill color, as each format stores it: BGRA8 bytes, RGBA16Float halves (0.25, 0.5, 0.75, 1).
static const uint8_t kBgra[4] = {40, 120, 200, 255};
static const uint16_t kHalf[4] = {0x3400, 0x3800, 0x3A00, 0x3C00};

static float halfToFloat(uint16_t h)
{
	int e = (h >> 10) & 0x1f, m = h & 0x3ff;
	float v = e == 0 ? ldexpf((float)m, -24) : e == 31 ? INFINITY : ldexpf((float)(m | 0x400), e - 25);
	return (h & 0x8000) ? -v : v;
}

static id stubAnswer(id self, SEL _cmd)
{
	return nil;
}

// Adds the probe-only stub for the selector an "unrecognized selector" reason names. Answers
// whether the step can run again.
static BOOL addStub(NSString *reason)
{
	NSRegularExpression *re = [NSRegularExpression
	        regularExpressionWithPattern:@"^([-+])\\[(\\S+) (\\S+)\\]: unrecognized selector"
	                             options:0
	                               error:nil];
	NSTextCheckingResult *m = [re firstMatchInString:reason options:0 range:NSMakeRange(0, reason.length)];
	if (m == nil || g_stubCount == kMaxStubs)
		return NO;
	NSString *kind = [reason substringWithRange:[m rangeAtIndex:1]];
	NSString *cls = [reason substringWithRange:[m rangeAtIndex:2]];
	NSString *sel = [reason substringWithRange:[m rangeAtIndex:3]];
	Class c = objc_getClass(cls.UTF8String);
	if (c == Nil)
		return NO;
	if ([kind isEqualToString:@"+"])
		c = object_getClass(c);
	if (!class_addMethod(c, sel_registerName(sel.UTF8String), (IMP)stubAnswer, "@@:"))
		return NO;
	snprintf(g_stubs[g_stubCount++], sizeof(g_stubs[0]), "%s[%s %s]", kind.UTF8String, cls.UTF8String,
	         sel.UTF8String);
	printf("    STUB (probe only, answers 0): %s[%s %s]\n", kind.UTF8String, cls.UTF8String, sel.UTF8String);
	return YES;
}

// Runs `step` in a command buffer of its own, commits it and waits. Prints the exception or the
// command buffer's error; with --discover, stubs each unrecognized selector and runs it again.
static BOOL runStep(id<MTLCommandQueue> queue, const char *name, Step step)
{
	for (int attempt = 0; attempt <= kMaxStubs; attempt++) {
		@try {
			id<MTLCommandBuffer> cb = [queue commandBuffer];
			BOOL encoded = step(cb);
			[cb commit];
			[cb waitUntilCompleted];
			if (cb.status != MTLCommandBufferStatusCompleted || cb.error != nil) {
				printf("  %-9s FAIL: command buffer status %ld, error %s\n", name, (long)cb.status,
				       cb.error.description.UTF8String ?: "none");
				return NO;
			}
			return encoded;
		} @catch (NSException *e) {
			printf("  %-9s EXCEPTION %s: %s\n", name, e.name.UTF8String, e.reason.UTF8String ?: "(no reason)");
			if (!g_discover || !addStub(e.reason ?: @""))
				return NO;
		}
	}
	return NO;
}

// Checks the `side` x `side` pixels of `bytes` against the fill color, as stored (`swapped`: red
// and blue exchanged, as a BGRA8 image's feature channels may read), all but those within `border`
// of an edge. Prints the first pixel checked.
static BOOL pixelsMatch(MTLPixelFormat format, const void *bytes, size_t side, size_t border, BOOL swapped,
                        const char *what)
{
	size_t bad = 0, count = 0, first = border * side + border;
	if (format == MTLPixelFormatBGRA8Unorm) {
		const uint8_t *p = bytes;
		uint8_t want[4] = {kBgra[0], kBgra[1], kBgra[2], kBgra[3]};
		if (swapped) {
			want[0] = kBgra[2];
			want[2] = kBgra[0];
		}
		for (size_t y = border; y < side - border; y++)
			for (size_t x = border; x < side - border; x++, count++)
				for (int c = 0; c < 4; c++)
					if (abs((int)p[(y * side + x) * 4 + c] - (int)want[c]) > 2) {
						bad++;
						break;
					}
		p += first * 4;
		printf("    %s: first pixel %u %u %u %u, %zu of %zu differ\n", what, p[0], p[1], p[2], p[3], bad, count);
	} else {
		const uint16_t *p = bytes;
		for (size_t y = border; y < side - border; y++)
			for (size_t x = border; x < side - border; x++, count++)
				for (int c = 0; c < 4; c++)
					if (fabsf(halfToFloat(p[(y * side + x) * 4 + c]) - halfToFloat(kHalf[c])) > 0.01f) {
						bad++;
						break;
					}
		p += first * 4;
		printf("    %s: first pixel %.3f %.3f %.3f %.3f, %zu of %zu differ\n", what, halfToFloat(p[0]),
		       halfToFloat(p[1]), halfToFloat(p[2]), halfToFloat(p[3]), bad, count);
	}
	return bad == 0;
}

static NSUInteger pixelBytes(MTLPixelFormat format)
{
	return format == MTLPixelFormatBGRA8Unorm ? 4 : 8;
}

// Encodes a copy of `texture` into `buffer` (and, for Managed, the synchronize Metal needs).
static void encodeReadback(id<MTLCommandBuffer> cb, id<MTLTexture> texture, id<MTLBuffer> buffer)
{
	id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
	NSUInteger row = texture.width * pixelBytes(texture.pixelFormat);
	[blit copyFromTexture:texture
	                 sourceSlice:0
	                 sourceLevel:0
	                sourceOrigin:MTLOriginMake(0, 0, 0)
	                  sourceSize:MTLSizeMake(texture.width, texture.height, 1)
	                    toBuffer:buffer
	           destinationOffset:0
	      destinationBytesPerRow:row
	    destinationBytesPerImage:row * texture.height];
	if (texture.storageMode == MTLStorageModeManaged)
		[blit synchronizeResource:texture];
	[blit endEncoding];
}

// Reads `texture` back through a blit and checks its pixels, all but those within `border` of an
// edge.
static BOOL textureMatches(id<MTLCommandQueue> queue, id<MTLTexture> texture, size_t border, const char *what)
{
	id<MTLBuffer> buffer = [queue.device newBufferWithLength:texture.width * texture.height *
	                                                         pixelBytes(texture.pixelFormat)
	                                                 options:MTLResourceStorageModeShared];
	__block BOOL ok = NO;
	char name[64];
	snprintf(name, sizeof(name), "(%s)", what);
	if (!runStep(queue, name, ^BOOL(id<MTLCommandBuffer> cb) {
		    encodeReadback(cb, texture, buffer);
		    return YES;
	    }))
		return NO;
	ok = pixelsMatch(texture.pixelFormat, buffer.contents, texture.width, border, NO, what);
	return ok;
}

static id<MTLTexture> makeTexture(id<MTLDevice> device, MTLPixelFormat format, MTLStorageMode storage,
                                  NSUInteger side)
{
	MTLTextureDescriptor *d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format
	                                                                             width:side
	                                                                            height:side
	                                                                         mipmapped:NO];
	d.storageMode = storage;
	d.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite | MTLTextureUsageRenderTarget;
	return [device newTextureWithDescriptor:d];
}

// Fills `texture` with the fill color through a blit from a buffer.
static BOOL fillTexture(id<MTLCommandQueue> queue, id<MTLTexture> texture)
{
	NSUInteger bpp = pixelBytes(texture.pixelFormat), n = texture.width * texture.height;
	id<MTLBuffer> buffer = [queue.device newBufferWithLength:n * bpp options:MTLResourceStorageModeShared];
	for (NSUInteger i = 0; i < n; i++)
		memcpy((uint8_t *)buffer.contents + i * bpp, bpp == 4 ? (const void *)kBgra : (const void *)kHalf, bpp);
	return runStep(queue, "(fill)", ^BOOL(id<MTLCommandBuffer> cb) {
		id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
		[blit copyFromBuffer:buffer
		               sourceOffset:0
		          sourceBytesPerRow:texture.width * bpp
		        sourceBytesPerImage:n * bpp
		                 sourceSize:MTLSizeMake(texture.width, texture.height, 1)
		                  toTexture:texture
		           destinationSlice:0
		           destinationLevel:0
		          destinationOrigin:MTLOriginMake(0, 0, 0)];
		[blit endEncoding];
		return YES;
	});
}

// An exception no step catches (one raised on a completion thread) still prints its reason.
static void uncaught(NSException *e)
{
	printf("UNCAUGHT EXCEPTION %s: %s\n", e.name.UTF8String, e.reason.UTF8String ?: "(no reason)");
}

static int runCase(id<MTLDevice> device, id<MTLCommandQueue> queue, MTLPixelFormat format, MTLStorageMode storage)
{
	static const char *storageNames[] = {"Shared", "Managed", "Private", "Memoryless"};
	printf("== %s %s\n", format == MTLPixelFormatBGRA8Unorm ? "BGRA8Unorm" : "RGBA16Float",
	       storageNames[storage]);
	id<MTLTexture> src = makeTexture(device, format, storage, kSide);
	id<MTLTexture> dst = makeTexture(device, format, storage, kSide);
	id<MTLTexture> half = makeTexture(device, format, storage, kSide / 2);
	if (src == nil || dst == nil || half == nil) {
		printf("  the device makes no %s texture of this format: skipped\n", storageNames[storage]);
		return 0;
	}
	int failures = 0;
	if (!fillTexture(queue, src) || !textureMatches(queue, src, 0, "source"))
		return 1;

	MPSImageGaussianBlur *blur = [[MPSImageGaussianBlur alloc] initWithDevice:device sigma:3.0f];
	blur.edgeMode = MPSImageEdgeModeClamp;
	BOOL ok = runStep(queue, "blur", ^BOOL(id<MTLCommandBuffer> cb) {
		[blur encodeToCommandBuffer:cb sourceTexture:src destinationTexture:dst];
		return YES;
	});
	ok = ok && textureMatches(queue, dst, 0, "blur");
	printf("  blur      %s\n", ok ? "ok" : "FAIL");
	failures += !ok;

	id<MTLTexture> inPlace = makeTexture(device, format, storage, kSide);
	__block id<MTLTexture> inPlaceResult = inPlace;
	ok = fillTexture(queue, inPlace) && runStep(queue, "inplace", ^BOOL(id<MTLCommandBuffer> cb) {
		     __block id<MTLTexture> t = inPlaceResult;
		     MPSCopyAllocator allocator = ^id<MTLTexture> NS_RETURNS_RETAINED(
		             MPSKernel *kernel, id<MTLCommandBuffer> buffer, id<MTLTexture> source) {
			     return makeTexture(device, source.pixelFormat, source.storageMode, source.width);
		     };
		     BOOL done = [blur encodeToCommandBuffer:cb inPlaceTexture:&t fallbackCopyAllocator:allocator];
		     inPlaceResult = t;
		     return done;
	     });
	ok = ok && textureMatches(queue, inPlaceResult, 0, "inplace");
	printf("  inplace   %s%s\n", ok ? "ok" : "FAIL", inPlaceResult != inPlace ? " (through the allocator's copy)" : "");
	failures += !ok;

	MPSImageLanczosScale *lanczos = [[MPSImageLanczosScale alloc] initWithDevice:device];
	ok = runStep(queue, "lanczos", ^BOOL(id<MTLCommandBuffer> cb) {
		[lanczos encodeToCommandBuffer:cb sourceTexture:src destinationTexture:half];
		return YES;
	});
	// Lanczos takes no samples outside the image, so a one-color image darkens within its kernel's
	// reach of an edge (on the M3 Max, 232 of the 1024 pixels at half size, none 3 or more from an
	// edge).
	ok = ok && textureMatches(queue, half, 3, "lanczos");
	printf("  lanczos   %s\n", ok ? "ok" : "FAIL");
	failures += !ok;

	ok = NO;
	if (storage == MTLStorageModePrivate) {
		// readBytes: reads the texture's bytes on the CPU, which a Private texture does not have:
		// the M3 Max reads zeros for BGRA8Unorm and crashes for RGBA16Float.
		printf("  readback  skipped: a Private texture has no bytes the CPU reads\n");
		ok = YES;
	} else @try {
		MPSImage *image = [[MPSImage alloc] initWithTexture:src featureChannels:4];
		NSMutableData *bytes = [NSMutableData dataWithLength:kSide * kSide * pixelBytes(format)];
		[image readBytes:bytes.mutableBytes dataLayout:MPSDataLayoutHeightxWidthxFeatureChannels imageIndex:0];
		ok = pixelsMatch(format, bytes.bytes, kSide, 0, NO, "readback");
		if (!ok && format == MTLPixelFormatBGRA8Unorm)
			ok = pixelsMatch(format, bytes.bytes, kSide, 0, YES, "readback, red and blue swapped");
	} @catch (NSException *e) {
		printf("  readback  EXCEPTION %s: %s\n", e.name.UTF8String, e.reason.UTF8String ?: "(no reason)");
	}
	if (storage != MTLStorageModePrivate)
		printf("  readback  %s\n", ok ? "ok" : "FAIL");
	failures += !ok;

	id<MTLTexture> dst2 = makeTexture(device, format, storage, kSide);
	ok = runStep(queue, "temporary", ^BOOL(id<MTLCommandBuffer> cb) {
		MPSImage *in = [[MPSImage alloc] initWithTexture:src featureChannels:4];
		MPSImage *out = [[MPSImage alloc] initWithTexture:dst2 featureChannels:4];
		MPSImageDescriptor *d = [MPSImageDescriptor
		        imageDescriptorWithChannelFormat:format == MTLPixelFormatBGRA8Unorm ? MPSImageFeatureChannelFormatUnorm8
		                                                                         : MPSImageFeatureChannelFormatFloat16
		                                   width:kSide
		                                  height:kSide
		                         featureChannels:4];
		MPSTemporaryImage *mid = [MPSTemporaryImage temporaryImageWithCommandBuffer:cb imageDescriptor:d];
		[blur encodeToCommandBuffer:cb sourceImage:in destinationImage:mid];
		[blur encodeToCommandBuffer:cb sourceImage:mid destinationImage:out];
		return YES;
	});
	ok = ok && textureMatches(queue, dst2, 0, "temporary");
	printf("  temporary %s\n", ok ? "ok" : "FAIL");
	failures += !ok;
	return failures;
}

int main(int argc, char **argv)
{
	setvbuf(stdout, NULL, _IONBF, 0);
	g_discover = argc > 1 && strcmp(argv[1], "--discover") == 0;
	NSSetUncaughtExceptionHandler(uncaught);
	@autoreleasepool {
		id<MTLDevice> device = MTLCreateSystemDefaultDevice();
		if (device == nil) {
			printf("no default Metal device\n");
			return 2;
		}
		printf("device %s (%s); MPSSupportsMTLDevice %s%s\n", device.name.UTF8String,
		       class_getName([device class]), MPSSupportsMTLDevice(device) ? "YES" : "NO",
		       g_discover ? "; --discover: PROBE-ONLY stubs answer 0" : "");
		id<MTLCommandQueue> queue = [device newCommandQueue];
		int failures = 0;
		const MTLPixelFormat formats[] = {MTLPixelFormatBGRA8Unorm, MTLPixelFormatRGBA16Float};
		const MTLStorageMode storages[] = {MTLStorageModePrivate, MTLStorageModeShared, MTLStorageModeManaged};
		for (int f = 0; f < 2; f++)
			for (int s = 0; s < 3; s++)
				failures += runCase(device, queue, formats[f], storages[s]);
		if (g_stubCount) {
			printf("== %d probe-only stubs (selectors the steps sent that the device's objects lack):\n",
			       g_stubCount);
			for (int i = 0; i < g_stubCount; i++)
				printf("  %s\n", g_stubs[i]);
		}
		printf("mps_probe: %s (%d failures)\n", failures ? "FAIL" : "OK", failures);
		return failures ? 1 : 0;
	}
}
