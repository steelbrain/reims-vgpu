// sweep_ci: Core Image filters on the device against the same math done here on the CPU. A
// software context is no reference: on macOS 27 kCIContextUseSoftwareRenderer gives a Metal
// context on the default device (measured: M3 Max, its description "(metal 1)").
// The GPU context works with no colour management (kCIContextWorkingColorSpace NSNull) in RGBAf,
// over a 256 x 256 pattern made on the CPU (premultiplied, its values taken as they are), and
// renders RGBAf (render:toBitmap:...), so each filter's math is a formula on the pattern's values.
// The CPU's reference computes it in double; both are compared in 8-bit levels (value x 255).
// Spatial filters take the pattern clamped to its extent; Core Image's coordinates run up from
// the bottom left, the pattern's rows down from the top.
//   ci.colormatrix, ci.colorinvert, ci.exposure, ci.gamma, ci.colorcube, ci.kernel.color,
//   ci.sourceover, ci.blendwithmask
//       one pixel in, one out: exact but for float rounding (the cube: its lookup's precision);
//   ci.convolution3, ci.convolution5, ci.morphology.max, ci.morphology.min, ci.gaussian,
//   ci.translate, ci.rotate, ci.pixellate, ci.kernel.warp
//       spatial: kernel sums, order statistics and bilinear samples, compared with both sides
//       clamped to 0..1;
//   ci.areaaverage, ci.areaminmax, ci.histogram
//       reductions; the histogram's bins normalised to sum 1 per channel, its max and mean the L1
//       distance in percent (worst channel, and over the channels);
//   ci.cgimage  the pattern through a colour-managed GPU context (sRGB in and out) and
//       createCGImage:, the path apps and ci_step take: the pattern again.
// Tolerances (sweep.m): the starting values hold for every case. The worst of the reference
// runs: ci.rotate 1.2/0.016 (stock), ci.gaussian 1.1/0.16 (AMD, Intel), ci.colorcube 0.5/0.15,
// ci.kernel.warp 0.5/0.008, ci.histogram 0/0.010 (Intel), the others 0. Stock's ci.convolution3
// (255/13.3) and ci.convolution5 (113/5.5) are a wrong result of that host's, which both other
// runs and the M3 Max give exactly: they set nothing.
#import "sweep.h"

#import <CoreImage/CoreImage.h>

enum { kSide = 256 };

static CIContext *gpu(void)
{
	static CIContext *c;
	if (c == nil)
		c = [CIContext contextWithMTLDevice:sweepDevice()
		                            options:@{
			                            kCIContextWorkingColorSpace : [NSNull null],
			                            kCIContextWorkingFormat : @(kCIFormatRGBAf),
			                            kCIContextCacheIntermediates : @NO,
		                            }];
	return c;
}

// A pattern's values in double, rows top first.
typedef struct {
	size_t w, h;
	double *v; // w * h * 4
} Plane;

static Plane planeOf(SweepPixels *p)
{
	Plane q = {p.width, p.height, calloc(p.width * p.height * 4, sizeof(double))};
	for (size_t i = 0; i < p.width * p.height * 4; i++)
		q.v[i] = p.bytes[i] / 255.0;
	return q;
}

static Plane blank(size_t w, size_t h)
{
	return (Plane){w, h, calloc(w * h * 4, sizeof(double))};
}

// The value of channel `c` at column `x`, row `y` (top first), clamped to the edges.
static double at(Plane p, long x, long y, int c)
{
	x = x < 0 ? 0 : x >= (long)p.w ? (long)p.w - 1 : x;
	y = y < 0 ? 0 : y >= (long)p.h ? (long)p.h - 1 : y;
	return p.v[((size_t)y * p.w + (size_t)x) * 4 + c];
}

// A bilinear sample at Core Image's point (cx, cy), clamped to the edges.
static double sample(Plane p, double cx, double cy, int c)
{
	// Core Image's pixel (i, j) has its centre at (i + 0.5, j + 0.5); row j from the bottom is the
	// pattern's row h - 1 - j.
	const double fx = cx - 0.5, fy = cy - 0.5;
	const long x0 = (long)floor(fx), y0 = (long)floor(fy);
	const double ax = fx - x0, ay = fy - y0;
	const long r0 = (long)p.h - 1 - y0, r1 = r0 - 1;
	return (1 - ay) * ((1 - ax) * at(p, x0, r0, c) + ax * at(p, x0 + 1, r0, c)) +
	       ay * ((1 - ax) * at(p, x0, r1, c) + ax * at(p, x0 + 1, r1, c));
}

static CIImage *imageOf(SweepPixels *p)
{
	return [CIImage imageWithBitmapData:p.data
	                        bytesPerRow:p.width * 4
	                               size:CGSizeMake(p.width, p.height)
	                             format:kCIFormatRGBA8
	                         colorSpace:nil];
}

// `image` rendered over `rect` against `want` (its rows top first), in 8-bit levels on the first
// `channels` channels; with `clamp`, both sides clamped to 0..1 first.
static SweepResult *compareRender(CIImage *image, CGRect rect, Plane want, int channels, bool clamp)
{
	const size_t w = (size_t)rect.size.width, h = (size_t)rect.size.height;
	if (image == nil)
		return [SweepResult fail:@"the filter made no image"];
	if (w != want.w || h != want.h)
		return [SweepResult fail:@"the reference is of another size"];
	NSMutableData *buf = [NSMutableData dataWithLength:w * h * 16];
	float *got = buf.mutableBytes;
	for (size_t i = 0; i < w * h * 4; i++)
		got[i] = NAN;
	[gpu() render:image toBitmap:got rowBytes:w * 16 bounds:rect format:kCIFormatRGBAf colorSpace:nil];
	double worst = 0, sum = 0;
	size_t wi = 0, n = 0;
	for (size_t i = 0; i < w * h; i++)
		for (int c = 0; c < channels; c++) {
			double g = got[i * 4 + c], r = want.v[i * 4 + c];
			if (clamp)
				g = fmin(fmax(g, 0), 1), r = fmin(fmax(r, 0), 1);
			const double d = isnan(g) ? 255 : fabs(g - r) * 255;
			sum += d, n++;
			if (d > worst)
				worst = d, wi = i;
		}
	SweepResult *res = [SweepResult max:worst mean:sum / n];
	if (worst > 0)
		res.note = [NSString stringWithFormat:@"worst at (%zu,%zu) %.4f %.4f %.4f for %.4f %.4f %.4f", wi % w,
		                                      wi / w, got[wi * 4], got[wi * 4 + 1], got[wi * 4 + 2], want.v[wi * 4],
		                                      want.v[wi * 4 + 1], want.v[wi * 4 + 2]];
	free(want.v);
	return res;
}

static const CGRect kAll = {{0, 0}, {kSide, kSide}};

// Starting values: float math on both sides, a fraction of a level apart.
static const SweepTol kExact = {1, 0.1};
// Starting values: kernel weights, sample positions and rounding of the device's own.
static const SweepTol kSpatial = {2, 0.25};

// A per-pixel filter: `f` maps a pixel's channels to the result's.
static void pointCaseTol(NSString *name, SweepPatternKind kind, SweepTol tol, CIImage * (^make)(CIImage *in),
                         void (^f)(const double in[4], double out[4]))
{
	sweepCase(name, tol, ^SweepResult * {
		SweepPixels *p = sweepPattern(kSide, kSide, kind);
		Plane in = planeOf(p), want = blank(kSide, kSide);
		for (size_t i = 0; i < kSide * kSide; i++)
			f(in.v + i * 4, want.v + i * 4);
		free(in.v);
		return compareRender(make(imageOf(p)), kAll, want, 4, false);
	});
}

static void pointCase(NSString *name, SweepPatternKind kind, CIImage * (^make)(CIImage *in),
                      void (^f)(const double in[4], double out[4]))
{
	pointCaseTol(name, kind, kExact, make, f);
}

// A spatial filter over the clamped pattern: `f` gives channel `c` of the result's pixel at
// column x, row y (top first).
static void spatialCase(NSString *name, SweepTol tol, CIImage * (^make)(CIImage *clamped),
                        double (^f)(Plane in, long x, long y, int c))
{
	sweepCase(name, tol, ^SweepResult * {
		SweepPixels *p = sweepPattern(kSide, kSide, SweepStructured);
		Plane in = planeOf(p), want = blank(kSide, kSide);
		for (long y = 0; y < kSide; y++)
			for (long x = 0; x < kSide; x++)
				for (int c = 0; c < 4; c++)
					want.v[((size_t)y * kSide + (size_t)x) * 4 + c] = f(in, x, y, c);
		free(in.v);
		CIImage *out = [make(imageOf(p).imageByClampingToExtent) imageByCroppingToRect:kAll];
		return compareRender(out, kAll, want, 4, true);
	});
}

static double unpremul(const double in[4], int c)
{
	return in[3] > 0 ? in[c] / in[3] : 0;
}

void sweepCoreImage(void)
{
	static const double kM[4][4] = {{0.6, 0.3, 0.1, 0}, {0.2, 0.7, 0.1, 0}, {0.1, 0.2, 0.65, 0}, {0, 0, 0, 1}};
	static const double kBias[4] = {0, 0.02, 0.05, 0};
	pointCase(
	    @"ci.colormatrix", SweepAlpha,
	    ^CIImage *(CIImage *in) {
		    return [in imageByApplyingFilter:@"CIColorMatrix"
		                 withInputParameters:@{
			                 @"inputRVector" : [CIVector vectorWithX:kM[0][0] Y:kM[0][1] Z:kM[0][2] W:kM[0][3]],
			                 @"inputGVector" : [CIVector vectorWithX:kM[1][0] Y:kM[1][1] Z:kM[1][2] W:kM[1][3]],
			                 @"inputBVector" : [CIVector vectorWithX:kM[2][0] Y:kM[2][1] Z:kM[2][2] W:kM[2][3]],
			                 @"inputAVector" : [CIVector vectorWithX:kM[3][0] Y:kM[3][1] Z:kM[3][2] W:kM[3][3]],
			                 @"inputBiasVector" : [CIVector vectorWithX:kBias[0] Y:kBias[1] Z:kBias[2] W:kBias[3]],
		                 }];
	    },
	    ^(const double in[4], double out[4]) {
		    // On unpremultiplied colour, premultiplied again.
		    double u[4] = {unpremul(in, 0), unpremul(in, 1), unpremul(in, 2), in[3]};
		    for (int r = 0; r < 4; r++) {
			    out[r] = kBias[r];
			    for (int k = 0; k < 4; k++)
				    out[r] += kM[r][k] * u[k];
		    }
		    for (int r = 0; r < 3; r++)
			    out[r] *= out[3];
	    });
	pointCase(
	    @"ci.colorinvert", SweepAlpha,
	    ^CIImage *(CIImage *in) {
		    return [in imageByApplyingFilter:@"CIColorInvert"];
	    },
	    ^(const double in[4], double out[4]) {
		    for (int c = 0; c < 3; c++)
			    out[c] = (1 - unpremul(in, c)) * in[3];
		    out[3] = in[3];
	    });
	pointCase(
	    @"ci.exposure", SweepStructured,
	    ^CIImage *(CIImage *in) {
		    return [in imageByApplyingFilter:@"CIExposureAdjust" withInputParameters:@{kCIInputEVKey : @0.5}];
	    },
	    ^(const double in[4], double out[4]) {
		    for (int c = 0; c < 3; c++)
			    out[c] = in[c] * pow(2, 0.5);
		    out[3] = in[3];
	    });
	pointCase(
	    @"ci.gamma", SweepStructured,
	    ^CIImage *(CIImage *in) {
		    return [in imageByApplyingFilter:@"CIGammaAdjust" withInputParameters:@{@"inputPower" : @0.6}];
	    },
	    ^(const double in[4], double out[4]) {
		    for (int c = 0; c < 3; c++)
			    out[c] = pow(in[c], 0.6);
		    out[3] = in[3];
	    });
	{
		// A cube of our own: each colour's channels rotated, squeezed to 0.1..0.9.
		enum { kN = 16 };
		NSMutableData *cube = [NSMutableData dataWithLength:kN * kN * kN * 4 * sizeof(float)];
		float *cv = cube.mutableBytes;
		for (int b = 0; b < kN; b++)
			for (int g = 0; g < kN; g++)
				for (int r = 0; r < kN; r++) {
					float *e = cv + ((b * kN + g) * kN + r) * 4;
					e[0] = 0.1f + 0.8f * g / (kN - 1);
					e[1] = 0.1f + 0.8f * b / (kN - 1);
					e[2] = 0.1f + 0.8f * r / (kN - 1);
					e[3] = 1;
				}
		// The cube is looked up as a texture: its entries' precision is the device's own.
		pointCaseTol(
		    @"ci.colorcube", SweepStructured, kSpatial,
		    ^CIImage *(CIImage *in) {
			    return [in imageByApplyingFilter:@"CIColorCube"
			                 withInputParameters:@{@"inputCubeDimension" : @(kN), @"inputCubeData" : cube}];
		    },
		    ^(const double in[4], double out[4]) {
			    // Trilinear, as the cube is linear in each channel: the rotation squeezed.
			    out[0] = 0.1 + 0.8 * in[1];
			    out[1] = 0.1 + 0.8 * in[2];
			    out[2] = 0.1 + 0.8 * in[0];
			    out[3] = in[3];
		    });
	}
	pointCase(
	    @"ci.kernel.color", SweepAlpha,
	    ^CIImage *(CIImage *in) {
		    CIColorKernel *k = [CIColorKernel
		        kernelWithString:@"kernel vec4 sweepSwizzle(__sample s) { return vec4(s.g, s.b, s.r, s.a) * 0.9 + "
		                         @"vec4(0.05, 0.05, 0.05, 0.0) * s.a; }"];
		    return [k applyWithExtent:in.extent arguments:@[ in ]];
	    },
	    ^(const double in[4], double out[4]) {
		    // `__sample` is premultiplied: the kernel's math on the values as they are.
		    out[0] = in[1] * 0.9 + 0.05 * in[3];
		    out[1] = in[2] * 0.9 + 0.05 * in[3];
		    out[2] = in[0] * 0.9 + 0.05 * in[3];
		    out[3] = in[3] * 0.9;
	    });
	sweepCase(@"ci.sourceover", kExact, ^SweepResult * {
		SweepPixels *fg = sweepPattern(kSide, kSide, SweepAlpha), *bg = sweepPattern(kSide, kSide, SweepSmooth);
		Plane f = planeOf(fg), b = planeOf(bg), want = blank(kSide, kSide);
		for (size_t i = 0; i < kSide * kSide; i++)
			for (int c = 0; c < 4; c++)
				want.v[i * 4 + c] = f.v[i * 4 + c] + b.v[i * 4 + c] * (1 - f.v[i * 4 + 3]);
		free(f.v), free(b.v);
		return compareRender([imageOf(fg) imageByCompositingOverImage:imageOf(bg)], kAll, want, 4, false);
	});
	sweepCase(@"ci.blendwithmask", kExact, ^SweepResult * {
		SweepPixels *fg = sweepPattern(kSide, kSide, SweepStructured), *bg = sweepPattern(kSide, kSide, SweepSmooth);
		SweepPixels *mask = [SweepPixels width:kSide height:kSide];
		for (size_t y = 0; y < kSide; y++)
			for (size_t x = 0; x < kSide; x++) {
				uint8_t *q = [mask at:x y:y];
				q[0] = q[1] = q[2] = (uint8_t)((x + y) * 255 / (2 * kSide - 2));
				q[3] = 255;
			}
		Plane f = planeOf(fg), b = planeOf(bg), m = planeOf(mask), want = blank(kSide, kSide);
		for (size_t i = 0; i < kSide * kSide; i++)
			for (int c = 0; c < 4; c++)
				want.v[i * 4 + c] = f.v[i * 4 + c] * m.v[i * 4] + b.v[i * 4 + c] * (1 - m.v[i * 4]);
		free(f.v), free(b.v), free(m.v);
		CIImage *out = [imageOf(fg) imageByApplyingFilter:@"CIBlendWithMask"
		                              withInputParameters:@{
			                              kCIInputBackgroundImageKey : imageOf(bg),
			                              kCIInputMaskImageKey : imageOf(mask),
		                              }];
		return compareRender(out, kAll, want, 4, false);
	});

	{
		static const double k3[9] = {0, -1, 0, -1, 5, -1, 0, -1, 0};
		spatialCase(
		    @"ci.convolution3", kSpatial,
		    ^CIImage *(CIImage *in) {
			    return [in imageByApplyingFilter:@"CIConvolution3X3"
			                 withInputParameters:@{kCIInputWeightsKey : [CIVector vectorWithValues:k3 count:9]}];
		    },
		    ^double(Plane in, long x, long y, int c) {
			    // Weights in Core Image's order: the first row is the top one.
			    double s = 0;
			    for (int j = 0; j < 3; j++)
				    for (int i = 0; i < 3; i++)
					    s += k3[j * 3 + i] * at(in, x + i - 1, y + j - 1, c);
			    return s;
		    });
		static double k5[25];
		for (int i = 0; i < 25; i++)
			k5[i] = (i % 5 + 2 * (i / 5) + 1) / 175.0;
		spatialCase(
		    @"ci.convolution5", kSpatial,
		    ^CIImage *(CIImage *in) {
			    return [in imageByApplyingFilter:@"CIConvolution5X5"
			                 withInputParameters:@{kCIInputWeightsKey : [CIVector vectorWithValues:k5 count:25]}];
		    },
		    ^double(Plane in, long x, long y, int c) {
			    double s = 0;
			    for (int j = 0; j < 5; j++)
				    for (int i = 0; i < 5; i++)
					    s += k5[j * 5 + i] * at(in, x + i - 2, y + j - 2, c);
			    return s;
		    });
	}
	for (int isMax = 1; isMax >= 0; isMax--) {
		spatialCase(
		    isMax ? @"ci.morphology.max" : @"ci.morphology.min", kSpatial,
		    ^CIImage *(CIImage *in) {
			    return [in imageByApplyingFilter:isMax ? @"CIMorphologyRectangleMaximum" : @"CIMorphologyRectangleMinimum"
			                 withInputParameters:@{@"inputWidth" : @7, @"inputHeight" : @5}];
		    },
		    ^double(Plane in, long x, long y, int c) {
			    double v = isMax ? -1 : 2;
			    for (long j = -2; j <= 2; j++)
				    for (long i = -3; i <= 3; i++)
					    v = isMax ? fmax(v, at(in, x + i, y + j, c)) : fmin(v, at(in, x + i, y + j, c));
			    return v;
		    });
	}
	spatialCase(
	    @"ci.gaussian", kSpatial,
	    ^CIImage *(CIImage *in) {
		    return [in imageByApplyingFilter:@"CIGaussianBlur" withInputParameters:@{kCIInputRadiusKey : @3}];
	    },
	    ^double(Plane in, long x, long y, int c) {
		    // A Gaussian of standard deviation 3, to three of them.
		    double s = 0, n = 0;
		    for (long j = -9; j <= 9; j++)
			    for (long i = -9; i <= 9; i++) {
				    const double w = exp(-(double)(i * i + j * j) / 18);
				    s += w * at(in, x + i, y + j, c), n += w;
			    }
		    return s / n;
	    });
	spatialCase(
	    @"ci.translate", kExact,
	    ^CIImage *(CIImage *in) {
		    return [in imageByApplyingTransform:CGAffineTransformMakeTranslation(17, -9)];
	    },
	    ^double(Plane in, long x, long y, int c) {
		    // Up is Core Image's +y: a move down of 9 rows is the row 9 above on the pattern.
		    return at(in, x - 17, y - 9, c);
	    });
	spatialCase(
	    @"ci.rotate", kSpatial,
	    ^CIImage *(CIImage *in) {
		    CGAffineTransform t = CGAffineTransformMakeTranslation(kSide / 2.0, kSide / 2.0);
		    t = CGAffineTransformRotate(t, 20 * M_PI / 180);
		    t = CGAffineTransformTranslate(t, -kSide / 2.0, -kSide / 2.0);
		    return [in imageByApplyingTransform:t highQualityDownsample:NO];
	    },
	    ^double(Plane in, long x, long y, int c) {
		    // The destination's centre in Core Image's coordinates, rotated back.
		    const double cx = x + 0.5 - kSide / 2.0, cy = (kSide - 1 - y) + 0.5 - kSide / 2.0;
		    const double a = -20 * M_PI / 180;
		    return sample(in, cos(a) * cx - sin(a) * cy + kSide / 2.0, sin(a) * cx + cos(a) * cy + kSide / 2.0, c);
	    });
	spatialCase(
	    @"ci.pixellate", kSpatial,
	    ^CIImage *(CIImage *in) {
		    return [in imageByApplyingFilter:@"CIPixellate"
		                 withInputParameters:@{kCIInputScaleKey : @8, kCIInputCenterKey : [CIVector vectorWithX:0 Y:0]}];
	    },
	    ^double(Plane in, long x, long y, int c) {
		    // Each 8 x 8 cell takes the sample at its centre, between four pixels.
		    const long cy = kSide - 1 - y;
		    return sample(in, (x / 8) * 8 + 4.0, (cy / 8) * 8 + 4.0, c);
	    });
	spatialCase(
	    @"ci.kernel.warp", kSpatial,
	    ^CIImage *(CIImage *in) {
		    CIWarpKernel *k = [CIWarpKernel kernelWithString:@"kernel vec2 sweepWave() { vec2 d = destCoord(); "
		                                                     @"return vec2(d.x + 6.0 * sin(d.y / 12.0), d.y); }"];
		    return [k applyWithExtent:kAll
		                  roiCallback:^CGRect(int index, CGRect r) {
			                  return CGRectInset(r, -8, -1);
		                  }
		                   inputImage:in
		                    arguments:@[]];
	    },
	    ^double(Plane in, long x, long y, int c) {
		    const double cx = x + 0.5, cy = (kSide - 1 - y) + 0.5;
		    return sample(in, cx + 6 * sin(cy / 12), cy, c);
	    });

	// Reductions, against sums and order statistics of the pattern.
	sweepCase(@"ci.areaaverage", kExact, ^SweepResult * {
		SweepPixels *p = sweepPattern(kSide, kSide, SweepStructured);
		Plane in = planeOf(p), want = blank(1, 1);
		for (size_t i = 0; i < kSide * kSide; i++)
			for (int c = 0; c < 4; c++)
				want.v[c] += in.v[i * 4 + c] / (kSide * kSide);
		free(in.v);
		CIImage *out = [imageOf(p) imageByApplyingFilter:@"CIAreaAverage"
		                             withInputParameters:@{kCIInputExtentKey : [CIVector vectorWithCGRect:kAll]}];
		return compareRender(out, CGRectMake(out.extent.origin.x, out.extent.origin.y, 1, 1), want, 4, false);
	});
	sweepCase(@"ci.areaminmax", kExact, ^SweepResult * {
		SweepPixels *p = sweepPattern(kSide, kSide, SweepStructured);
		Plane in = planeOf(p), want = blank(2, 1);
		for (int c = 0; c < 4; c++)
			want.v[c] = 2, want.v[4 + c] = -1;
		for (size_t i = 0; i < kSide * kSide; i++)
			for (int c = 0; c < 4; c++)
				want.v[c] = fmin(want.v[c], in.v[i * 4 + c]), want.v[4 + c] = fmax(want.v[4 + c], in.v[i * 4 + c]);
		free(in.v);
		CIImage *out = [imageOf(p) imageByApplyingFilter:@"CIAreaMinMax"
		                             withInputParameters:@{kCIInputExtentKey : [CIVector vectorWithCGRect:kAll]}];
		return compareRender(out, CGRectMake(out.extent.origin.x, out.extent.origin.y, 2, 1), want, 4, false);
	});
	// Starting values: a value on a bin's boundary may fall either side.
	sweepCase(@"ci.histogram", (SweepTol){2, 1}, ^SweepResult * {
		enum { kBins = 64 };
		SweepPixels *p = sweepPattern(kSide, kSide, SweepStructured);
		CIImage *out = [imageOf(p) imageByApplyingFilter:@"CIAreaHistogram"
		                             withInputParameters:@{
			                             kCIInputExtentKey : [CIVector vectorWithCGRect:kAll],
			                             @"inputCount" : @(kBins),
			                             kCIInputScaleKey : @1,
		                             }];
		float got[kBins * 4];
		[gpu() render:out
		     toBitmap:got
		     rowBytes:sizeof(got)
		       bounds:CGRectMake(out.extent.origin.x, out.extent.origin.y, kBins, 1)
		       format:kCIFormatRGBAf
		   colorSpace:nil];
		double want[3][kBins] = {{0}};
		for (size_t i = 0; i < kSide * kSide; i++)
			for (int c = 0; c < 3; c++) {
				const int bin = (int)fmin(floor(p.bytes[i * 4 + c] / 255.0 * kBins), kBins - 1);
				want[c][bin] += 1.0 / (kSide * kSide);
			}
		double worst = 0, total = 0;
		for (int c = 0; c < 3; c++) {
			double sum = 0, l1 = 0;
			for (int i = 0; i < kBins; i++)
				sum += got[i * 4 + c];
			if (!(sum > 0))
				return [SweepResult fail:@"an empty histogram"];
			for (int i = 0; i < kBins; i++)
				l1 += fabs(got[i * 4 + c] / sum - want[c][i]);
			worst = fmax(worst, l1 * 100), total += l1 * 100;
		}
		SweepResult *r = [SweepResult max:worst mean:total / 3];
		r.note = @"L1 distance in percent";
		return r;
	});

	// Starting values: sRGB to linear half floats and back, a level at most.
	sweepCase(@"ci.cgimage", (SweepTol){1, 0.1}, ^SweepResult * {
		static CIContext *managed;
		if (managed == nil)
			managed = [CIContext contextWithMTLDevice:sweepDevice() options:@{kCIContextCacheIntermediates : @NO}];
		SweepPixels *p = sweepPattern(kSide, kSide, SweepAlpha);
		CIImage *in = [CIImage imageWithBitmapData:p.data
		                               bytesPerRow:kSide * 4
		                                      size:CGSizeMake(kSide, kSide)
		                                    format:kCIFormatRGBA8
		                                colorSpace:sweepSRGB()];
		CGImageRef out = [managed createCGImage:in fromRect:kAll format:kCIFormatRGBA8 colorSpace:sweepSRGB()];
		SweepPixels *got = [SweepPixels ofImage:out];
		if (out != NULL)
			CGImageRelease(out);
		return sweepCompare(got, p, CGRectNull, 4);
	});
}
