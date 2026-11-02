// sweep_imageio: ImageIO's encoders and decoders, which may run their colour conversions on the
// Metal device (a HEIC decode converts its planes there where the process has one).
//   imageio.png, .tiff, .bmp (opaque) and .png.alpha, .tiff.alpha: a 256 x 256 pattern encoded
//     and decoded again, lossless: exact.
//   imageio.jpeg, .heic, .avif, .jpeg2000: a 512 x 512 smooth gradient at quality 0.9, decoded
//     again, against the gradient: lossy, so a mean and a loose max.
//   imageio.heic.alpha: the alpha pattern (sharp edges, an alpha ramp) through HEIC: colour and
//     alpha.
//   imageio.webp: a lossless WebP of the 128 x 128 structured pattern (sweep_webp.h), decoded:
//     exact. ImageIO decodes WebP but does not encode it.
//   imageio.desktop.N: the first three HEIC desktop pictures, decoded whole and box-filtered on
//     the CPU to ImageIO's 512-pixel thumbnail of the same file: a weak check, as both decodes
//     may run on the device; the whole decode must not be one colour.
// A type ImageIO does not encode here is SKIP.
// Tolerances (sweep.m): the starting values hold but for imageio.heic.alpha's mean, 4.8, set by
// all three reference runs alike (3.19; its max, 230 everywhere, leaves 255). The worst of the
// others: the lossy codecs 4/0.51, the desktop pictures 166/2.13, the lossless ones 0.
#import "sweep.h"
#import "sweep_webp.h"

#import <ImageIO/ImageIO.h>

// A decode of `data`, done now (not when first drawn).
static CGImageRef decode(NSData *data) CF_RETURNS_RETAINED
{
	CGImageSourceRef src = CGImageSourceCreateWithData((__bridge CFDataRef)data, NULL);
	if (src == NULL)
		return NULL;
	CGImageRef image = CGImageSourceCreateImageAtIndex(
	    src, 0, (__bridge CFDictionaryRef) @{(id)kCGImageSourceShouldCacheImmediately : @YES});
	CFRelease(src);
	return image;
}

// `pixels` encoded as `type` with `props` and decoded again; nil with `*why` where it does not.
static SweepPixels *roundTrip(NSString *type, SweepPixels *pixels, NSDictionary *props, NSString **why)
{
	NSArray *types = CFBridgingRelease(CGImageDestinationCopyTypeIdentifiers());
	if (![types containsObject:type]) {
		*why = [NSString stringWithFormat:@"ImageIO encodes no %@ here", type];
		return nil;
	}
	CGImageRef image = [pixels newImage];
	NSMutableData *encoded = [NSMutableData data];
	CGImageDestinationRef dst =
	    CGImageDestinationCreateWithData((__bridge CFMutableDataRef)encoded, (__bridge CFStringRef)type, 1, NULL);
	bool ok = false;
	if (dst != NULL) {
		CGImageDestinationAddImage(dst, image, (__bridge CFDictionaryRef)props);
		ok = CGImageDestinationFinalize(dst);
		CFRelease(dst);
	}
	CGImageRelease(image);
	if (!ok) {
		*why = [NSString stringWithFormat:@"%@ encode failed", type];
		return nil;
	}
	CGImageRef back = decode(encoded);
	SweepPixels *out = [SweepPixels ofImage:back];
	if (back != NULL)
		CGImageRelease(back);
	if (out == nil)
		*why = [NSString stringWithFormat:@"%@ decode failed (%lu bytes)", type, (unsigned long)encoded.length];
	return out;
}

static void lossless(NSString *name, NSString *type, SweepPatternKind kind)
{
	// Exact: a lossless codec in and out of sRGB RGBA8 changes no value.
	sweepCase(name, (SweepTol){0, 0}, ^SweepResult * {
		SweepPixels *want = sweepPattern(256, 256, kind);
		NSString *why = nil;
		SweepPixels *got = roundTrip(type, want, @{}, &why);
		if (got == nil)
			return [why hasPrefix:@"ImageIO encodes no"] ? [SweepResult skip:why] : [SweepResult fail:why];
		return sweepCompare(got, want, CGRectNull, kind == SweepAlpha ? 4 : 3);
	});
}

static void lossy(NSString *name, NSString *type, SweepPatternKind kind, int channels, SweepTol tol)
{
	sweepCase(name, tol, ^SweepResult * {
		SweepPixels *want = sweepPattern(512, 512, kind);
		NSString *why = nil;
		SweepPixels *got =
		    roundTrip(type, want, @{(id)kCGImageDestinationLossyCompressionQuality : @0.9}, &why);
		if (got == nil)
			return [why hasPrefix:@"ImageIO encodes no"] ? [SweepResult skip:why] : [SweepResult fail:why];
		if (sweepOneColour(got, CGRectNull))
			return [SweepResult fail:@"the decode is of one colour"];
		return sweepCompare(got, want, CGRectNull, channels);
	});
}

// `big` box-filtered to `w` x `h`: each pixel the mean of the source pixels whose centres fall in
// its footprint.
static SweepPixels *boxDown(SweepPixels *big, size_t w, size_t h)
{
	SweepPixels *out = [SweepPixels width:w height:h];
	for (size_t y = 0; y < h; y++) {
		const size_t y0 = y * big.height / h, y1 = (y + 1) * big.height / h;
		for (size_t x = 0; x < w; x++) {
			const size_t x0 = x * big.width / w, x1 = (x + 1) * big.width / w;
			double s[4] = {0, 0, 0, 0};
			for (size_t yy = y0; yy < y1; yy++)
				for (size_t xx = x0; xx < x1; xx++)
					for (int k = 0; k < 4; k++)
						s[k] += [big at:xx y:yy][k];
			const double n = (double)((y1 - y0) * (x1 - x0));
			for (int k = 0; k < 4; k++)
				[out at:x y:y][k] = (uint8_t)(s[k] / n + 0.5);
		}
	}
	return out;
}

void sweepImageIO(void)
{
	lossless(@"imageio.png", @"public.png", SweepStructured);
	lossless(@"imageio.png.alpha", @"public.png", SweepAlpha);
	lossless(@"imageio.tiff", @"public.tiff", SweepStructured);
	lossless(@"imageio.tiff.alpha", @"public.tiff", SweepAlpha);
	lossless(@"imageio.bmp", @"com.microsoft.bmp", SweepStructured);

	// Starting values: a lossy codec at 0.9 keeps a smooth gradient within a few levels on
	// average; a lost decode is 100 and more.
	lossy(@"imageio.jpeg", @"public.jpeg", SweepSmooth, 3, (SweepTol){48, 4});
	lossy(@"imageio.heic", @"public.heic", SweepSmooth, 3, (SweepTol){48, 4});
	lossy(@"imageio.avif", @"public.avif", SweepSmooth, 3, (SweepTol){48, 4});
	lossy(@"imageio.jpeg2000", @"public.jpeg-2000", SweepSmooth, 3, (SweepTol){48, 4});
	// Sharp edges and an alpha ramp: a mean only (edges ring in a lossy codec). The mean is the
	// reference runs' (3.19 on each) times 1.5.
	lossy(@"imageio.heic.alpha", @"public.heic", SweepAlpha, 4, (SweepTol){255, 4.8});

	sweepCase(@"imageio.webp", (SweepTol){0, 0}, ^SweepResult * {
		CGImageRef image = decode([NSData dataWithBytes:kSweepWebP length:sizeof(kSweepWebP)]);
		SweepPixels *got = [SweepPixels ofImage:image];
		if (image != NULL)
			CGImageRelease(image);
		if (got == nil)
			return [SweepResult fail:@"the WebP does not decode"];
		return sweepCompare(got, sweepPattern(128, 128, SweepStructured), CGRectNull, 4);
	});

	NSString *dir = @"/System/Library/Desktop Pictures";
	NSMutableArray<NSString *> *heics = [NSMutableArray array];
	for (NSString *n in [[[NSFileManager defaultManager] contentsOfDirectoryAtPath:dir error:nil]
	         sortedArrayUsingSelector:@selector(compare:)])
		if ([n.pathExtension isEqualToString:@"heic"] && heics.count < 3)
			[heics addObject:[dir stringByAppendingPathComponent:n]];
	for (NSUInteger i = 0; i < 3; i++) {
		NSString *name = [NSString stringWithFormat:@"imageio.desktop.%lu", (unsigned long)i + 1];
		// Starting values: two resamplers of one picture differ by a few levels on average.
		sweepCase(name, (SweepTol){255, 6}, ^SweepResult * {
			if (i >= heics.count)
				return [SweepResult skip:@"no such HEIC desktop picture here"];
			CGImageSourceRef src =
			    CGImageSourceCreateWithURL((__bridge CFURLRef)[NSURL fileURLWithPath:heics[i]], NULL);
			if (src == NULL)
				return [SweepResult fail:@"the picture does not open"];
			CGImageRef whole = CGImageSourceCreateImageAtIndex(
			    src, 0, (__bridge CFDictionaryRef) @{(id)kCGImageSourceShouldCacheImmediately : @YES});
			CGImageRef thumb = CGImageSourceCreateThumbnailAtIndex(
			    src, 0,
			    (__bridge CFDictionaryRef) @{
				    (id)kCGImageSourceCreateThumbnailFromImageAlways : @YES,
				    (id)kCGImageSourceThumbnailMaxPixelSize : @512,
				    (id)kCGImageSourceShouldCacheImmediately : @YES,
			    });
			CFRelease(src);
			SweepPixels *big = [SweepPixels ofImage:whole], *small = [SweepPixels ofImage:thumb];
			if (whole != NULL)
				CGImageRelease(whole);
			if (thumb != NULL)
				CGImageRelease(thumb);
			if (big == nil || small == nil)
				return [SweepResult fail:big == nil ? @"no whole decode" : @"no thumbnail"];
			if (sweepOneColour(big, CGRectNull))
				return [SweepResult fail:@"the whole decode is of one colour"];
			SweepResult *r = sweepCompare(boxDown(big, small.width, small.height), small, CGRectNull, 3);
			r.note = [NSString stringWithFormat:@"%zu x %zu to %zu x %zu", big.width, big.height,
			                                    small.width, small.height];
			return r;
		});
	}
}
