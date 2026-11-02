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
//       scaled to 192 x 192, compared inside its edges.
// Tolerances (sweep.m): the starting values hold for every case. The worst of the reference
// runs: the flat cases 0/0.10, ca.gradient 15/8.51 (all three alike, the starting value's own
// measure), ca.contents 1/0.33 (stock).
#import "sweep.h"

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

static SweepPixels *onDevice(CALayer *root, NSString **why)
{
	id<MTLDevice> dev = sweepDevice();
	MTLTextureDescriptor *d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
	                                                                             width:kSide
	                                                                            height:kSide
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
	r.bounds = CGRectMake(0, 0, kSide, kSide);
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
	SweepPixels *raw = [SweepPixels width:kSide height:kSide], *p = [SweepPixels width:kSide height:kSide];
	[tex getBytes:raw.bytes bytesPerRow:kSide * 4 fromRegion:MTLRegionMake2D(0, 0, kSide, kSide) mipmapLevel:0];
	for (size_t y = 0; y < kSide; y++)
		for (size_t x = 0; x < kSide; x++) {
			const uint8_t *s = [raw at:x y:kSide - 1 - y];
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
		SweepPixels *got = onDevice(root, &why);
		if (got == nil)
			return [SweepResult fail:why];
		SweepPixels *want = onCPU(root);
		if (sweepOneColour(got, CGRectNull) && !sweepOneColour(want, CGRectNull))
			return [SweepResult fail:@"the device's rendering is of one colour"];
		return flat ? sweepCompareFlat(got, want, 3) : sweepCompare(got, want, rect, 3);
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
}
