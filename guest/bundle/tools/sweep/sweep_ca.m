// sweep_ca: Core Animation's renderer on the device against its CPU rendering. Each case builds a
// 256 x 256 layer tree, renders it with a CARenderer into a Managed BGRA8Unorm texture of the
// device on a queue of the device (kCARendererMetalCommandQueue), read back after a
// synchronizeResource, and with -renderInContext: into a CGBitmapContext; both in sRGB. The
// texture's first row is the layer's bottom one (a layer's origin is its bottom left).
//   ca.solid, ca.corners, ca.opacity, ca.rotation, ca.shape, ca.text
//       flat colours: the max over the pixels whose 3 x 3 neighbourhood in the CPU rendering is
//       of one colour (anti-aliased edges differ between the two), the mean over every pixel;
//   ca.gradient, ca.contents
//       a CAGradientLayer, every pixel compared; a layer whose contents are the smooth pattern
//       scaled to 192 x 192, compared inside its edges;
//   ca.heic.grid, ca.heic.grid.cropped
//       the structured pattern as an HEIC of a grid of tiles, 2048 x 1536 (whole tiles) and
//       2000 x 1400 (tiles past the right and bottom edges), drawn as a layer's contents at its
//       own size into a target of that size, against ImageIO's decode into a bitmap, compared as
//       the flat cases are.
// Tolerances (sweep.m): the starting values hold for every case. The worst of the reference
// runs: the flat cases 0/0.10, ca.gradient 15/8.51 (all three alike, the starting value's own
// measure), ca.contents 1/0.33 (stock), ca.heic.grid 1/5.28 and ca.heic.grid.cropped 1/5.33 (the
// AMD and the Intel GPU alike).
#import "sweep.h"

#import <ImageIO/ImageIO.h>
#import <QuartzCore/QuartzCore.h>

enum { kSide = 256 };

static CGColorRef colour(int r, int g, int b, double a)
{
	return (CGColorRef)CFAutorelease(CGColorCreateSRGB(r / 255.0, g / 255.0, b / 255.0, a));
}

static CALayer *rootLayer(void)
{
	CALayer *root = [CALayer layer];
	root.frame = CGRectMake(0, 0, kSide, kSide);
	root.backgroundColor = colour(40, 60, 90, 1);
	root.opaque = YES;
	return root;
}

// `root` rendered on the device into a `w` x `h` target; nil with `*why` where it is not.
static SweepPixels *onDevice(CALayer *root, size_t w, size_t h, NSString **why)
{
	id<MTLDevice> dev = sweepDevice();
	MTLTextureDescriptor *d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
	                                                                             width:w
	                                                                            height:h
	                                                                         mipmapped:NO];
	d.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
	d.storageMode = MTLStorageModeManaged;
	id<MTLTexture> tex = [dev newTextureWithDescriptor:d];
	id<MTLCommandQueue> q = [dev newCommandQueue];
	CARenderer *r = [CARenderer rendererWithMTLTexture:tex
	                                           options:@{
		                                           kCARendererColorSpace : (__bridge id)sweepSRGB(),
		                                           kCARendererMetalCommandQueue : q,
	                                           }];
	if (r == nil) {
		*why = @"no CARenderer over the texture";
		return nil;
	}
	// The renderer draws the committed layer tree: with no run loop here, the transaction is
	// flushed by hand, or the render draws nothing.
	[CATransaction begin];
	[CATransaction setDisableActions:YES];
	r.layer = root;
	r.bounds = CGRectMake(0, 0, w, h);
	[CATransaction commit];
	[CATransaction flush];
	[r beginFrameAtTime:CACurrentMediaTime() timeStamp:NULL];
	[r addUpdateRect:r.bounds];
	[r render];
	[r endFrame];
	id<MTLCommandBuffer> cb = [q commandBuffer];
	id<MTLBlitCommandEncoder> b = [cb blitCommandEncoder];
	[b synchronizeResource:tex];
	[b endEncoding];
	[cb commit];
	[cb waitUntilCompleted];
	if (cb.status != MTLCommandBufferStatusCompleted) {
		*why = [NSString stringWithFormat:@"command buffer %@", cb.error];
		return nil;
	}
	// A layer's origin is its bottom left, the texture's first row the layer's bottom one: the
	// rows are turned over, top first as the bitmap's, and BGRA made RGBA.
	SweepPixels *raw = [SweepPixels width:w height:h], *p = [SweepPixels width:w height:h];
	[tex getBytes:raw.bytes bytesPerRow:w * 4 fromRegion:MTLRegionMake2D(0, 0, w, h) mipmapLevel:0];
	for (size_t y = 0; y < h; y++)
		for (size_t x = 0; x < w; x++) {
			const uint8_t *s = [raw at:x y:h - 1 - y];
			uint8_t *d = [p at:x y:y];
			d[0] = s[2], d[1] = s[1], d[2] = s[0], d[3] = s[3];
		}
	return p;
}

static SweepPixels *onCPU(CALayer *root)
{
	SweepPixels *p = [SweepPixels width:kSide height:kSide];
	CGContextRef ctx = CGBitmapContextCreate(p.bytes, kSide, kSide, 8, kSide * 4, sweepSRGB(),
	                                         kCGImageAlphaPremultipliedLast | kCGBitmapByteOrder32Big);
	[root renderInContext:ctx];
	CGContextRelease(ctx);
	return p;
}

// `build` adds layers to a root of one colour; `flat` picks sweepCompareFlat over sweepCompare,
// over `rect` (top-left origin; CGRectNull: all).
static void caCase(NSString *name, SweepTol tol, bool flat, CGRect rect, void (^build)(CALayer *root))
{
	sweepCase(name, tol, ^SweepResult * {
		[CATransaction begin];
		[CATransaction setDisableActions:YES];
		CALayer *root = rootLayer();
		build(root);
		[CATransaction commit];
		[CATransaction flush];
		NSString *why = nil;
		SweepPixels *got = onDevice(root, kSide, kSide, &why);
		if (got == nil)
			return [SweepResult fail:why];
		SweepPixels *want = onCPU(root);
		if (sweepOneColour(got, CGRectNull) && !sweepOneColour(want, CGRectNull))
			return [SweepResult fail:@"the device's rendering is of one colour"];
		return flat ? sweepCompareFlat(got, want, 3) : sweepCompare(got, want, rect, 3);
	});
}

// The items of type `type` an HEIF file's item info lists: each 'infe' box (version 2 or 3) whose
// item_type is `type`. A crude walk, for the cases' own files: it looks for the box's name and
// reads the type where version 2 (16-bit item ID) or 3 (32-bit) puts it.
static int heifItems(NSData *file, const char *type)
{
	const uint8_t *b = file.bytes;
	const size_t n = file.length;
	int count = 0;
	for (size_t i = 4; i + 20 <= n; i++) {
		if (memcmp(b + i, "infe", 4) != 0)
			continue;
		const uint8_t version = b[i + 4];
		const size_t at = i + 4 + 4 + (version == 3 ? 4 : 2) + 2;
		if ((version == 2 || version == 3) && at + 4 <= n && memcmp(b + at, type, 4) == 0)
			count++;
	}
	return count;
}

// The image in `file` as ImageIO decodes it: at once, into a bitmap, or, `lazy`, where it is first
// drawn.
static CGImageRef heifImage(NSData *file, bool lazy) CF_RETURNS_RETAINED
{
	CGImageSourceRef src = CGImageSourceCreateWithData((__bridge CFDataRef)file, NULL);
	if (src == NULL)
		return NULL;
	NSDictionary *now = @{(id)kCGImageSourceShouldCacheImmediately : @YES};
	CGImageRef image = CGImageSourceCreateImageAtIndex(src, 0, lazy ? NULL : (__bridge CFDictionaryRef)now);
	CFRelease(src);
	return image;
}

// A case of a still HEIF `w` x `h`, which ImageIO encodes as a grid of tiles, as a layer's
// contents at its own size: Core Animation has it decoded into an IOSurface, its tiles laid into
// one canvas by VideoToolbox's tile decoder where the hardware decoder offers it, and draws the
// surface's YCbCr. Compared with the same file decoded by ImageIO on the CPU into a bitmap, as
// the flat cases are, the max away from the pattern's edges, where the two upsample chroma
// differently. Each tile in its place: a tile drawn elsewhere, or none (YCbCr zero, green), moves
// the mean by tens of levels. The max's starting value is the lossy codecs' (sweep_imageio.m).
static void heicGridCase(NSString *name, SweepTol tol, size_t w, size_t h)
{
	sweepCase(name, tol, ^SweepResult * {
		CGImageRef pattern = [sweepPattern(w, h, SweepStructured) newImage];
		NSMutableData *file = [NSMutableData data];
		CGImageDestinationRef dst =
		    CGImageDestinationCreateWithData((__bridge CFMutableDataRef)file, CFSTR("public.heic"), 1, NULL);
		bool encoded = false;
		if (dst != NULL) {
			CGImageDestinationAddImage(dst, pattern,
			                           (__bridge CFDictionaryRef) @{(id)kCGImageDestinationLossyCompressionQuality : @0.9});
			encoded = CGImageDestinationFinalize(dst);
			CFRelease(dst);
		}
		CGImageRelease(pattern);
		if (!encoded)
			return [SweepResult skip:@"ImageIO encodes no HEIC here"];
		const int grids = heifItems(file, "grid"), tiles = heifItems(file, "hvc1");
		if (grids != 1 || tiles < 2)
			return [SweepResult fail:[NSString stringWithFormat:@"the HEIC is no grid: %d grid items, %d tiles", grids, tiles]];
		CGImageRef cpu = heifImage(file, false), lazy = heifImage(file, true);
		SweepPixels *want = [SweepPixels ofImage:cpu];
		if (cpu != NULL)
			CGImageRelease(cpu);
		if (want == nil || lazy == NULL) {
			if (lazy != NULL)
				CGImageRelease(lazy);
			return [SweepResult fail:@"ImageIO decoded no image of the HEIC"];
		}
		[CATransaction begin];
		[CATransaction setDisableActions:YES];
		CALayer *root = [CALayer layer];
		root.frame = CGRectMake(0, 0, w, h);
		root.backgroundColor = colour(40, 60, 90, 1);
		root.opaque = YES;
		CALayer *l = [CALayer layer];
		l.frame = root.bounds;
		l.contents = (__bridge id)lazy;
		CGImageRelease(lazy);
		l.contentsGravity = kCAGravityResize;
		[root addSublayer:l];
		[CATransaction commit];
		[CATransaction flush];
		NSString *why = nil;
		SweepPixels *got = onDevice(root, w, h, &why);
		if (got == nil)
			return [SweepResult fail:why];
		SweepResult *r = sweepCompareFlat(got, want, 3);
		r.note = [NSString stringWithFormat:@"%d tiles", tiles];
		return r;
	});
}

static CALayer *box(CGRect frame, CGColorRef c)
{
	CALayer *l = [CALayer layer];
	l.frame = frame;
	l.backgroundColor = c;
	return l;
}

void sweepCoreAnimation(void)
{
	// Starting values: a flat colour composited on the device is the CPU's within a level or two;
	// edges, anti-aliased differently, only count toward the mean.
	const SweepTol flatTol = {3, 1.5};
	caCase(@"ca.solid", flatTol, true, CGRectNull, ^(CALayer *root) {
		[root addSublayer:box(CGRectMake(32, 32, 128, 96), colour(200, 120, 40, 1))];
	});
	caCase(@"ca.corners", flatTol, true, CGRectNull, ^(CALayer *root) {
		CALayer *l = box(CGRectMake(40, 40, 176, 150), colour(30, 160, 90, 1));
		l.cornerRadius = 24;
		l.borderWidth = 6;
		l.borderColor = colour(240, 240, 240, 1);
		l.masksToBounds = YES;
		[root addSublayer:l];
	});
	caCase(@"ca.opacity", flatTol, true, CGRectNull, ^(CALayer *root) {
		CALayer *a = box(CGRectMake(30, 30, 140, 140), colour(220, 40, 40, 1));
		CALayer *b = box(CGRectMake(90, 90, 140, 140), colour(40, 80, 220, 1));
		a.opacity = 0.5, b.opacity = 0.5;
		[root addSublayer:a];
		[root addSublayer:b];
	});
	caCase(@"ca.rotation", flatTol, true, CGRectNull, ^(CALayer *root) {
		CALayer *l = box(CGRectMake(64, 80, 128, 96), colour(230, 200, 60, 1));
		l.affineTransform = CGAffineTransformMakeRotation(30 * M_PI / 180);
		[root addSublayer:l];
	});
	caCase(@"ca.shape", flatTol, true, CGRectNull, ^(CALayer *root) {
		CAShapeLayer *s = [CAShapeLayer layer];
		s.frame = root.bounds;
		CGMutablePathRef path = CGPathCreateMutable();
		CGPathAddEllipseInRect(path, NULL, CGRectMake(20, 20, 120, 90));
		for (int i = 0; i < 10; i++) {
			const double a = M_PI / 2 + i * M_PI / 5, r = i % 2 ? 30 : 75;
			const CGPoint p = CGPointMake(170 + r * cos(a), 150 + r * sin(a));
			i == 0 ? CGPathMoveToPoint(path, NULL, p.x, p.y) : CGPathAddLineToPoint(path, NULL, p.x, p.y);
		}
		CGPathCloseSubpath(path);
		s.path = path;
		CGPathRelease(path);
		s.fillColor = colour(220, 200, 40, 1);
		s.strokeColor = colour(20, 20, 20, 1);
		s.lineWidth = 4;
		[root addSublayer:s];
	});
	caCase(@"ca.text", flatTol, true, CGRectNull, ^(CALayer *root) {
		CATextLayer *t = [CATextLayer layer];
		t.frame = CGRectMake(16, 100, 224, 60);
		t.string = @"Sweep 0123";
		t.fontSize = 32;
		t.foregroundColor = colour(250, 250, 250, 1);
		t.contentsScale = 1;
		[root addSublayer:t];
	});
	// Starting values: the device interpolates the gradient otherwise than CoreGraphics (measured
	// on the M3 Max: max 15, mean 8.5), 1.5 times that; a gradient lost to the root's colour is 80
	// and more.
	caCase(@"ca.gradient", (SweepTol){23, 13}, false, CGRectNull, ^(CALayer *root) {
		CAGradientLayer *g = [CAGradientLayer layer];
		g.frame = root.bounds;
		g.colors = @[ (__bridge id)colour(230, 40, 30, 1), (__bridge id)colour(30, 60, 230, 1) ];
		g.startPoint = CGPointMake(0, 0);
		g.endPoint = CGPointMake(1, 1);
		[root addSublayer:g];
	});
	// Starting values: the device's texture filter against CoreGraphics' interpolation, inside
	// the layer's edges, over the smooth pattern (sharp edges scaled differ by tens of levels).
	caCase(@"ca.contents", (SweepTol){6, 2}, false, CGRectMake(34, 34, 188, 188), ^(CALayer *root) {
		CALayer *l = [CALayer layer];
		l.frame = CGRectMake(32, 32, 192, 192);
		CGImageRef image = [sweepPattern(kSide, kSide, SweepSmooth) newImage];
		l.contents = (__bridge id)image;
		CGImageRelease(image);
		l.contentsGravity = kCAGravityResize;
		l.minificationFilter = kCAFilterLinear;
		[root addSublayer:l];
	});
	// A still HEIF of a grid of tiles, as Photos' and an asset catalog's large images are, as a
	// layer's contents at its own size (heicGridCase): of whole tiles, and of tiles that reach
	// past the canvas's right and bottom edges.
	heicGridCase(@"ca.heic.grid", (SweepTol){48, 8}, 2048, 1536);
	heicGridCase(@"ca.heic.grid.cropped", (SweepTol){48, 8}, 2000, 1400);
}
