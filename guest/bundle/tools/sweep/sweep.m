// sweep: a correctness sweep of the system frameworks that run on the Metal device, each case
// comparing the device's output with a reference that does not touch the device: the same work
// on the CPU (formulas, vImage, Accelerate, CoreGraphics), or a known answer made on
// the CPU (a pattern encoded and decoded again, a frame whose number it carries). Every result is
// read back on the CPU, so a GPU write that does not reach memory the CPU reads fails.
//   sweep [-d NAME] [PATTERN...]
//     -d NAME   the cases that choose a device (Core Image's GPU context, MPS, CARenderer) take
//               the first of MTLCopyAllDevices whose name holds NAME; the default is
//               MTLCreateSystemDefaultDevice. ImageIO, AVFoundation and VideoToolbox choose their
//               own.
//     PATTERN   runs only the cases whose name holds one of the patterns.
//   sweep --pattern-png PATH   writes the 128 x 128 structured pattern as a PNG (sweep_webp.h's
//               source) and exits.
// The cases (each file's header says what it compares, and each tolerance where it came from):
//   imageio.*  sweep_imageio.m: encode and decode round trips, WebP decode, desktop pictures;
//   ci.*       sweep_ci.m: Core Image filters on the device against their formulas on the CPU;
//   video.*    sweep_video.m: AVFoundation clips made here, read back; VTPixelTransferSession
//              against vImage;
//   mps.*      sweep_mps.m: MPS image kernels and a matrix multiply against CPU math;
//   ca.*       sweep_ca.m: CARenderer against -renderInContext: into a bitmap.
// Tolerances: each case's is the larger of its starting value (the comment at the case says why)
// and 1.5 times the worst of three reference runs: the stock guest driver on apple-gfx (an RTX 5080
// laptop host on the Vulkan platform, macOS 26.6.1), and a Mac's discrete AMD Radeon Pro 5300M and
// integrated Intel UHD 630 (macOS 26.7.1). A result a reference gets wrong, where the others and
// the M3 Max agree on the formula, sets nothing. Each file's header says which run set which case.
// One line a case: "sweep: NAME OK|FAIL|SKIP", its largest and mean difference and tolerance (or
// why it did not run, or why it failed), and its time. Then a line of the counts. Exits 1 where a
// case failed, else 0. A case that cannot run here (no encoder for a codec, MPS unsupported on the
// device) is SKIP with its reason, never OK.
// Every input is made here on the CPU, but for one WebP (sweep_webp.h, our own gradient) and the
// system's desktop pictures, read at run time.
// Build: make sweep (build/sweep-x86_64); make sweep ARCH=arm64 (build/sweep-arm64) for an
// Apple-silicon Mac.
#import "sweep.h"

#import <ImageIO/ImageIO.h>

#include <mach/mach_time.h>
#include <sys/sysctl.h>
#include <unistd.h>

static NSArray<NSString *> *gPatterns;
static id<MTLDevice> gDevice;
static int gOk, gFail, gSkip;

@implementation SweepResult
+ (instancetype)max:(double)max mean:(double)mean
{
	SweepResult *r = [self new];
	r.max = max;
	r.mean = mean;
	return r;
}
+ (instancetype)skip:(NSString *)why
{
	SweepResult *r = [self new];
	r.skip = why;
	return r;
}
+ (instancetype)fail:(NSString *)why
{
	SweepResult *r = [self new];
	r.fail = why;
	return r;
}
@end

@implementation SweepPixels
+ (instancetype)width:(size_t)width height:(size_t)height
{
	SweepPixels *p = [self new];
	p->_width = width;
	p->_height = height;
	p->_data = [NSMutableData dataWithLength:width * height * 4];
	return p;
}
+ (instancetype)ofImage:(CGImageRef)image
{
	if (image == NULL)
		return nil;
	SweepPixels *p = [self width:CGImageGetWidth(image) height:CGImageGetHeight(image)];
	CGContextRef ctx = CGBitmapContextCreate(p.bytes, p.width, p.height, 8, p.width * 4, sweepSRGB(),
	                                         kCGImageAlphaPremultipliedLast | kCGBitmapByteOrder32Big);
	CGContextSetBlendMode(ctx, kCGBlendModeCopy);
	CGContextDrawImage(ctx, CGRectMake(0, 0, p.width, p.height), image);
	CGContextRelease(ctx);
	return p;
}
- (uint8_t *)bytes
{
	return (uint8_t *)_data.mutableBytes;
}
- (uint8_t *)at:(size_t)x y:(size_t)y
{
	return self.bytes + (y * _width + x) * 4;
}
- (CGImageRef)newImage
{
	bool opaque = true;
	const uint8_t *b = self.bytes;
	for (size_t i = 0; i < _width * _height && opaque; i++)
		opaque = b[i * 4 + 3] == 255;
	CGDataProviderRef dp = CGDataProviderCreateWithCFData((__bridge CFDataRef)[_data copy]);
	CGImageRef image =
	    CGImageCreate(_width, _height, 8, 32, _width * 4, sweepSRGB(),
	                  (opaque ? kCGImageAlphaNoneSkipLast : kCGImageAlphaPremultipliedLast) |
	                      kCGBitmapByteOrder32Big,
	                  dp, NULL, false, kCGRenderingIntentDefault);
	CGDataProviderRelease(dp);
	return image;
}
@end

CGColorSpaceRef sweepSRGB(void)
{
	static CGColorSpaceRef cs;
	if (cs == NULL)
		cs = CGColorSpaceCreateWithName(kCGColorSpaceSRGB);
	return cs;
}

id<MTLDevice> sweepDevice(void)
{
	return gDevice;
}

NSString *sweepTempPath(NSString *suffix)
{
	static int n;
	return [NSString stringWithFormat:@"%@/sweep-%d-%d%@", NSTemporaryDirectory(), getpid(), n++, suffix];
}

SweepPixels *sweepPattern(size_t w, size_t h, SweepPatternKind kind)
{
	SweepPixels *p = [SweepPixels width:w height:h];
	for (size_t y = 0; y < h; y++) {
		for (size_t x = 0; x < w; x++) {
			uint8_t *q = [p at:x y:y];
			int c[3] = {(int)(x * 255 / (w - 1)), (int)(y * 255 / (h - 1)),
			            (int)((x + y) * 255 / (w + h - 2))};
			if (kind != SweepSmooth) {
				const long dx = (long)x - (long)(w * 7 / 10), dy = (long)y - (long)(h * 13 / 20);
				const long r = (long)h / 6;
				if (y >= h / 8 && y < h / 4) {
					const bool on = ((x / 8) + (y / 8)) % 2 == 0;
					c[0] = on ? 240 : 16, c[1] = on ? 240 : 32, c[2] = on ? 32 : 224;
				} else if (dx * dx + dy * dy <= r * r) {
					c[0] = 32, c[1] = 200, c[2] = 96;
				} else if (((x + 1024 - y) % 32) < 2) {
					c[0] = 255, c[1] = 64, c[2] = 0;
				} else if ((x >= 12 && x < 16) || (y >= 12 && y < 16) || (x + 16 >= w && x + 12 < w) ||
				           (y + 16 >= h && y + 12 < h)) {
					c[0] = c[1] = c[2] = 200;
				}
			}
			int a = 255;
			if (kind == SweepAlpha)
				a = 255 - (int)(x * 255 / (w - 1));
			for (int k = 0; k < 3; k++)
				q[k] = (uint8_t)((c[k] * a + 127) / 255);
			q[3] = (uint8_t)a;
		}
	}
	return p;
}

static CGRect clipRect(CGRect rect, size_t w, size_t h)
{
	CGRect all = CGRectMake(0, 0, w, h);
	return CGRectIsNull(rect) ? all : CGRectIntersection(CGRectIntegral(rect), all);
}

SweepResult *sweepCompare(SweepPixels *got, SweepPixels *want, CGRect rect, int channels)
{
	if (got == nil || want == nil)
		return [SweepResult fail:got == nil ? @"no result" : @"no reference"];
	if (got.width != want.width || got.height != want.height)
		return [SweepResult fail:[NSString stringWithFormat:@"size %zu x %zu, want %zu x %zu", got.width,
		                                                    got.height, want.width, want.height]];
	const CGRect r = clipRect(rect, got.width, got.height);
	double sum = 0;
	int worst = 0;
	size_t wx = 0, wy = 0, n = 0;
	for (size_t y = (size_t)r.origin.y; y < (size_t)CGRectGetMaxY(r); y++) {
		for (size_t x = (size_t)r.origin.x; x < (size_t)CGRectGetMaxX(r); x++) {
			const uint8_t *a = [got at:x y:y], *b = [want at:x y:y];
			for (int k = 0; k < channels; k++) {
				const int d = abs((int)a[k] - (int)b[k]);
				sum += d;
				n++;
				if (d > worst)
					worst = d, wx = x, wy = y;
			}
		}
	}
	SweepResult *res = [SweepResult max:worst mean:n ? sum / n : 0];
	if (worst > 0) {
		const uint8_t *a = [got at:wx y:wy], *b = [want at:wx y:wy];
		res.note = [NSString stringWithFormat:@"worst at (%zu,%zu) %u %u %u %u for %u %u %u %u", wx, wy, a[0],
		                                      a[1], a[2], a[3], b[0], b[1], b[2], b[3]];
	}
	return res;
}

SweepResult *sweepCompareFlat(SweepPixels *got, SweepPixels *want, int channels)
{
	SweepResult *all = sweepCompare(got, want, CGRectNull, channels);
	if (all.fail != nil)
		return all;
	int worst = 0;
	size_t wx = 0, wy = 0;
	for (size_t y = 1; y + 1 < want.height; y++) {
		for (size_t x = 1; x + 1 < want.width; x++) {
			const uint8_t *c = [want at:x y:y];
			bool flat = true;
			for (int dy = -1; dy <= 1 && flat; dy++)
				for (int dx = -1; dx <= 1 && flat; dx++)
					flat = memcmp([want at:x + dx y:y + dy], c, 4) == 0;
			if (!flat)
				continue;
			const uint8_t *a = [got at:x y:y];
			for (int k = 0; k < channels; k++) {
				const int d = abs((int)a[k] - (int)c[k]);
				if (d > worst)
					worst = d, wx = x, wy = y;
			}
		}
	}
	SweepResult *res = [SweepResult max:worst mean:all.mean];
	if (worst > 0) {
		const uint8_t *a = [got at:wx y:wy], *b = [want at:wx y:wy];
		res.note = [NSString stringWithFormat:@"worst flat at (%zu,%zu) %u %u %u %u for %u %u %u %u", wx, wy,
		                                      a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3]];
	}
	return res;
}

BOOL sweepOneColour(SweepPixels *p, CGRect rect)
{
	const CGRect r = clipRect(rect, p.width, p.height);
	const uint8_t *first = [p at:(size_t)r.origin.x y:(size_t)r.origin.y];
	for (size_t y = (size_t)r.origin.y; y < (size_t)CGRectGetMaxY(r); y++)
		for (size_t x = (size_t)r.origin.x; x < (size_t)CGRectGetMaxX(r); x++)
			if (memcmp([p at:x y:y], first, 4) != 0)
				return NO;
	return YES;
}

static double nowMs(void)
{
	static mach_timebase_info_data_t tb;
	if (tb.denom == 0)
		mach_timebase_info(&tb);
	return (double)mach_absolute_time() * tb.numer / tb.denom / 1e6;
}

void sweepCase(NSString *name, SweepTol tol, SweepResult * (^run)(void))
{
	if (gPatterns.count != 0) {
		bool match = false;
		for (NSString *p in gPatterns)
			match = match || [name containsString:p];
		if (!match)
			return;
	}
	const double t0 = nowMs();
	SweepResult *r = nil;
	@autoreleasepool {
		@try {
			r = run();
		} @catch (NSException *e) {
			r = [SweepResult fail:[NSString stringWithFormat:@"exception %@: %@", e.name, e.reason]];
		}
	}
	const double ms = nowMs() - t0;
	if (r.skip != nil) {
		gSkip++;
		printf("sweep: %-30s SKIP  %s (%.0f ms)\n", name.UTF8String, r.skip.UTF8String, ms);
	} else if (r.fail != nil) {
		gFail++;
		printf("sweep: %-30s FAIL  %s (%.0f ms)\n", name.UTF8String, r.fail.UTF8String, ms);
	} else {
		const bool ok = r.max <= tol.max && r.mean <= tol.mean;
		ok ? gOk++ : gFail++;
		printf("sweep: %-30s %-4s  max %5.1f mean %6.3f  tol %g/%g%s%s (%.0f ms)\n", name.UTF8String,
		       ok ? "OK" : "FAIL", r.max, r.mean, tol.max, tol.mean, r.note ? "  " : "",
		       r.note ? r.note.UTF8String : "", ms);
	}
	fflush(stdout);
}

int main(int argc, char **argv)
{
	@autoreleasepool {
		// The source of sweep_webp.h: the 128 x 128 structured pattern as a PNG.
		if (argc == 3 && strcmp(argv[1], "--pattern-png") == 0) {
			CGImageRef image = [sweepPattern(128, 128, SweepStructured) newImage];
			CGImageDestinationRef dst = CGImageDestinationCreateWithURL(
			    (__bridge CFURLRef)[NSURL fileURLWithPath:@(argv[2])], CFSTR("public.png"), 1, NULL);
			CGImageDestinationAddImage(dst, image, NULL);
			const bool ok = CGImageDestinationFinalize(dst);
			CFRelease(dst);
			CGImageRelease(image);
			return ok ? 0 : 1;
		}
		NSMutableArray *patterns = [NSMutableArray array];
		const char *want = NULL;
		for (int i = 1; i < argc; i++) {
			if (strcmp(argv[i], "-d") == 0 && i + 1 < argc)
				want = argv[++i];
			else
				[patterns addObject:@(argv[i])];
		}
		gPatterns = patterns;
		if (want != NULL) {
			for (id<MTLDevice> d in MTLCopyAllDevices())
				if (gDevice == nil && strcasestr(d.name.UTF8String, want))
					gDevice = d;
		} else {
			gDevice = MTLCreateSystemDefaultDevice();
		}
		if (gDevice == nil) {
			printf("sweep: no Metal device%s%s\n", want ? " matching " : "", want ? want : "");
			return 2;
		}
		char model[64] = "?", build[32] = "?";
		size_t len = sizeof(model);
		sysctlbyname("hw.model", model, &len, NULL, 0);
		len = sizeof(build);
		sysctlbyname("kern.osversion", build, &len, NULL, 0);
		NSOperatingSystemVersion v = NSProcessInfo.processInfo.operatingSystemVersion;
		printf("sweep: device \"%s\", %s, macOS %ld.%ld.%ld (%s), %s\n", gDevice.name.UTF8String, model,
		       (long)v.majorVersion, (long)v.minorVersion, (long)v.patchVersion, build,
#if defined(__x86_64__)
		       "x86_64"
#else
		       "arm64"
#endif
		);
		sweepImageIO();
		sweepCoreImage();
		sweepVideo();
		sweepMPS();
		sweepCoreAnimation();
		printf("sweep: %d OK, %d FAIL, %d SKIP\n", gOk, gFail, gSkip);
	}
	return gFail != 0 ? 1 : 0;
}
