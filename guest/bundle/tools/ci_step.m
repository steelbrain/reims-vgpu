// ci_step: a Core Image chain over a large picture on the default Metal device, timed, with its
// pixels checked for a lost result.
//   ci_step [IMAGE]   IMAGE defaults to /System/Library/Desktop Pictures/Mac Blue.heic.
// The chain: the picture, CIGaussianBlur (radius 10, over its clamped edges, cropped back to the
// picture), CIColorControls (saturation 1.2, brightness 0.02, contrast 1.1), CISepiaTone (0.4),
// CILanczosScaleTransform (0.5). A CIContext on the device makes a CGImage of its extent
// ("CGImage made", timed), drawn into an RGBA8 sRGB bitmap ("bitmap", timed). Two checks:
// - the middle 512 x 512 of the result not all one colour: Core Image writes a surface on the
//   device and reads it back on the CPU, and with a Metal device ImageIO converts a HEIC's planes
//   on that device, so a decode or a chain whose GPU writes were lost reads as one colour;
// - a 512 x 512 gradient made on the CPU, encoded to HEIC and decoded back, within
//   kRoundTripError per channel on average (lossy; a lost decode is 100 and more).
// The chain is not compared with a software CIContext's: kCIContextUseSoftwareRenderer gives a
// Metal context on the default device (measured: macOS 27, M3 Max), the device under test, and
// no formula of ours is known for this chain's filters. The sweep's Core Image cases
// (tools/sweep/sweep_ci.m) check filters against formulas computed on the CPU.
// Prints the times, the first pixel, each check's numbers, then "ci_step: OK" (exit 0) or
// "ci_step: FAIL" (exit 1).
// Build: xcrun clang -arch x86_64 -fobjc-arc -framework Metal -framework Foundation
//        -framework CoreImage -framework CoreGraphics -framework ImageIO tools/ci_step.m -o ci_step
#import <CoreGraphics/CoreGraphics.h>
#import <CoreImage/CoreImage.h>
#import <Foundation/Foundation.h>
#import <ImageIO/ImageIO.h>
#import <Metal/Metal.h>
#include <mach/mach_time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { kSide = 512, kRoundTripError = 8 };

static double nowMs(void)
{
	static mach_timebase_info_data_t tb;
	if (tb.denom == 0)
		mach_timebase_info(&tb);
	return (double)mach_absolute_time() * tb.numer / tb.denom / 1e6;
}

// `image` drawn into new RGBA8 sRGB bytes, rows top first; the caller frees them.
static uint8_t *bitmapOf(CGImageRef image)
{
	size_t w = CGImageGetWidth(image), h = CGImageGetHeight(image);
	uint8_t *bytes = calloc(w * h, 4);
	CGColorSpaceRef cs = CGColorSpaceCreateWithName(kCGColorSpaceSRGB);
	CGContextRef ctx = CGBitmapContextCreate(bytes, w, h, 8, w * 4, cs,
	                                         kCGImageAlphaPremultipliedLast | kCGBitmapByteOrder32Big);
	CGContextDrawImage(ctx, CGRectMake(0, 0, w, h), image);
	CGContextRelease(ctx);
	CGColorSpaceRelease(cs);
	return bytes;
}

static CIImage *chain(CIImage *picture)
{
	// Blurred over its clamped edges and cropped back, so every pixel of the result is opaque.
	CIImage *blurred = [[picture.imageByClampingToExtent imageByApplyingFilter:@"CIGaussianBlur"
	                                                       withInputParameters:@{kCIInputRadiusKey : @10.0}]
	    imageByCroppingToRect:picture.extent];
	CIImage *colored = [blurred imageByApplyingFilter:@"CIColorControls"
	                              withInputParameters:@{
		                              kCIInputSaturationKey : @1.2,
		                              kCIInputBrightnessKey : @0.02,
		                              kCIInputContrastKey : @1.1,
	                              }];
	CIImage *sepia = [colored imageByApplyingFilter:@"CISepiaTone"
	                            withInputParameters:@{kCIInputIntensityKey : @0.4}];
	return [sepia imageByApplyingFilter:@"CILanczosScaleTransform"
	                withInputParameters:@{kCIInputScaleKey : @0.5, kCIInputAspectRatioKey : @1.0}];
}

// The mean difference per channel between a CPU-made gradient and its HEIC round trip through
// ImageIO; negative where it does not encode or decode.
static double heicRoundTripError(void)
{
	const size_t side = 512;
	uint8_t *src = malloc(side * side * 4);
	for (size_t y = 0; y < side; y++) {
		for (size_t x = 0; x < side; x++) {
			uint8_t *p = src + (y * side + x) * 4;
			p[0] = (uint8_t)(x / 2);
			p[1] = (uint8_t)(y / 2);
			p[2] = (uint8_t)((x + y) / 4);
			p[3] = 255;
		}
	}
	CGColorSpaceRef cs = CGColorSpaceCreateWithName(kCGColorSpaceSRGB);
	CGContextRef ctx = CGBitmapContextCreate(src, side, side, 8, side * 4, cs,
	                                         kCGImageAlphaNoneSkipLast | kCGBitmapByteOrder32Big);
	CGImageRef image = CGBitmapContextCreateImage(ctx);
	NSMutableData *heic = [NSMutableData data];
	CGImageDestinationRef dst = CGImageDestinationCreateWithData((__bridge CFMutableDataRef)heic,
	                                                             CFSTR("public.heic"), 1, NULL);
	bool encoded = false;
	if (dst) {
		CGImageDestinationAddImage(dst, image, NULL);
		encoded = CGImageDestinationFinalize(dst);
		CFRelease(dst);
	}
	CGImageRelease(image);
	CGContextRelease(ctx);
	CGColorSpaceRelease(cs);
	double error = -1;
	CGImageSourceRef in = encoded ? CGImageSourceCreateWithData((__bridge CFDataRef)heic, NULL) : NULL;
	CGImageRef back = in ? CGImageSourceCreateImageAtIndex(in, 0, NULL) : NULL;
	if (back && CGImageGetWidth(back) == side && CGImageGetHeight(back) == side) {
		uint8_t *got = bitmapOf(back);
		double sum = 0;
		for (size_t i = 0; i < side * side; i++)
			for (int c = 0; c < 3; c++)
				sum += abs((int)got[i * 4 + c] - (int)src[i * 4 + c]);
		error = sum / (3.0 * side * side);
		free(got);
	}
	if (back)
		CGImageRelease(back);
	if (in)
		CFRelease(in);
	free(src);
	return error;
}

int main(int argc, char **argv)
{
	@autoreleasepool {
		const char *path = argc > 1 ? argv[1] : "/System/Library/Desktop Pictures/Mac Blue.heic";
		id<MTLDevice> device = MTLCreateSystemDefaultDevice();
		printf("ci_step: device %s\n", device ? device.name.UTF8String : "(none)");
		CIImage *picture = [CIImage imageWithContentsOfURL:[NSURL fileURLWithPath:@(path)]];
		if (device == nil || picture == nil) {
			printf("ci_step: no %s\nci_step: FAIL\n", device == nil ? "device" : "image");
			return 1;
		}
		printf("ci_step: image %s, %.0f x %.0f\n", path, picture.extent.size.width,
		       picture.extent.size.height);
		CIImage *result = chain(picture);
		CGRect extent = CGRectIntegral(result.extent);

		CIContext *gpu = [CIContext contextWithMTLDevice:device];
		double t0 = nowMs();
		CGImageRef made = [gpu createCGImage:result fromRect:extent];
		double t1 = nowMs();
		if (made == NULL) {
			printf("ci_step: no CGImage\nci_step: FAIL\n");
			return 1;
		}
		size_t w = CGImageGetWidth(made), h = CGImageGetHeight(made);
		printf("ci_step: CGImage made %zu x %zu in %.1f ms\n", w, h, t1 - t0);
		uint8_t *got = bitmapOf(made);
		printf("ci_step: bitmap in %.1f ms, first pixel %u %u %u %u\n", nowMs() - t1, got[0], got[1],
		       got[2], got[3]);
		CGImageRelease(made);

		// The middle kSide x kSide of the result.
		const size_t mx = (w - kSide) / 2, my = (h - kSide) / 2;
		bool oneColour = true;
		for (size_t y = my; y < my + kSide && oneColour; y++)
			for (size_t x = mx; x < mx + kSide && oneColour; x++)
				oneColour = memcmp(got + (y * w + x) * 4, got + (my * w + mx) * 4, 4) == 0;
		printf("ci_step: middle %d x %d %s\n", kSide, kSide, oneColour ? "all one colour" : "of more than one colour");
		free(got);
		const double roundTrip = heicRoundTripError();
		printf("ci_step: HEIC round trip of a %d x %d gradient: mean error %.1f per channel\n", kSide,
		       kSide, roundTrip);
		const bool ok = !oneColour && roundTrip >= 0 && roundTrip <= kRoundTripError;
		printf("ci_step: %s\n", ok ? "OK" : "FAIL");
		return ok ? 0 : 1;
	}
}
