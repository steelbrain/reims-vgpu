// sweep_mps: Metal Performance Shaders kernels on the device against the same math on the CPU,
// over a 256 x 256 pattern made on the CPU in a Managed RGBA8Unorm texture (read back after a
// synchronizeResource). Where MPSSupportsMTLDevice says no, every case is SKIP.
//   mps.gaussian    MPSImageGaussianBlur, sigma 3, edges clamped, against a separable Gaussian of
//                   radius 9 on the CPU;
//   mps.sobel       MPSImageSobel to an R32Float texture against the Sobel magnitude of MPS's
//                   default luminance (0.299, 0.587, 0.114), edges clamped; in 8-bit levels;
//   mps.bilinear    MPSImageBilinearScale to 128 x 128: each pixel the mean of its 2 x 2 source;
//   mps.histogram   MPSImageHistogram of 256 bins over R, G and B against counts on the CPU:
//                   max and mean are differences of counts;
//   mps.matmul      MPSMatrixMultiplication of two 256 x 256 float matrices against
//                   cblas_sgemm: max and mean are the differences relative to the largest
//                   magnitude of the result, in parts per million.
// Tolerances (sweep.m): the starting values hold for every case. The worst of the reference
// runs: mps.gaussian 1/0.17, mps.bilinear 1/0.33, mps.matmul 0.6/0.049 ppm (Intel), mps.sobel
// and mps.histogram 0.
#import "sweep.h"

#import <Accelerate/Accelerate.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

enum { kSide = 256 };

static id<MTLTexture> texture(id<MTLDevice> dev, size_t w, size_t h, MTLPixelFormat format)
{
	MTLTextureDescriptor *d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format
	                                                                             width:w
	                                                                            height:h
	                                                                         mipmapped:NO];
	d.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
	d.storageMode = MTLStorageModeManaged;
	return [dev newTextureWithDescriptor:d];
}

static id<MTLTexture> source(id<MTLDevice> dev, SweepPixels *p)
{
	id<MTLTexture> t = texture(dev, p.width, p.height, MTLPixelFormatRGBA8Unorm);
	[t replaceRegion:MTLRegionMake2D(0, 0, p.width, p.height) mipmapLevel:0 withBytes:p.bytes bytesPerRow:p.width * 4];
	return t;
}

// Encodes `work`, then a synchronize of `out`, and waits.
static NSString *run(id<MTLDevice> dev, id<MTLResource> out, void (^work)(id<MTLCommandBuffer>))
{
	id<MTLCommandQueue> q = [dev newCommandQueue];
	id<MTLCommandBuffer> cb = [q commandBuffer];
	work(cb);
	if (out != nil && out.storageMode == MTLStorageModeManaged) {
		id<MTLBlitCommandEncoder> b = [cb blitCommandEncoder];
		[b synchronizeResource:out];
		[b endEncoding];
	}
	[cb commit];
	[cb waitUntilCompleted];
	return cb.status == MTLCommandBufferStatusCompleted ? nil
	                                                    : [NSString stringWithFormat:@"command buffer %@", cb.error];
}

static SweepPixels *readRGBA(id<MTLTexture> t)
{
	SweepPixels *p = [SweepPixels width:t.width height:t.height];
	[t getBytes:p.bytes bytesPerRow:t.width * 4 fromRegion:MTLRegionMake2D(0, 0, t.width, t.height) mipmapLevel:0];
	return p;
}

static void mpsCase(NSString *name, SweepTol tol, SweepResult * (^body)(id<MTLDevice> dev))
{
	sweepCase(name, tol, ^SweepResult * {
		id<MTLDevice> dev = sweepDevice();
		if (!MPSSupportsMTLDevice(dev))
			return [SweepResult skip:[NSString stringWithFormat:@"MPS does not support \"%@\"", dev.name]];
		return body(dev);
	});
}

static int clampi(int v, int lo, int hi)
{
	return v < lo ? lo : v > hi ? hi : v;
}

void sweepMPS(void)
{
	SweepPixels *pattern = sweepPattern(kSide, kSide, SweepStructured);

	// Starting values: one Gaussian, its weights and rounding a level apart.
	mpsCase(@"mps.gaussian", (SweepTol){2, 0.5}, ^SweepResult *(id<MTLDevice> dev) {
		id<MTLTexture> in = source(dev, pattern), out = texture(dev, kSide, kSide, MTLPixelFormatRGBA8Unorm);
		MPSImageGaussianBlur *k = [[MPSImageGaussianBlur alloc] initWithDevice:dev sigma:3];
		k.edgeMode = MPSImageEdgeModeClamp;
		NSString *err = run(dev, out, ^(id<MTLCommandBuffer> cb) {
			[k encodeToCommandBuffer:cb sourceTexture:in destinationTexture:out];
		});
		if (err != nil)
			return [SweepResult fail:err];
		enum { kR = 9 };
		double w[2 * kR + 1], sum = 0;
		for (int i = -kR; i <= kR; i++)
			sum += w[i + kR] = exp(-(double)(i * i) / (2 * 3.0 * 3.0));
		NSMutableData *tmp = [NSMutableData dataWithLength:kSide * kSide * 4 * sizeof(double)];
		double *t = tmp.mutableBytes;
		for (int y = 0; y < kSide; y++)
			for (int x = 0; x < kSide; x++)
				for (int c = 0; c < 4; c++) {
					double s = 0;
					for (int i = -kR; i <= kR; i++)
						s += w[i + kR] * [pattern at:clampi(x + i, 0, kSide - 1) y:y][c];
					t[(y * kSide + x) * 4 + c] = s / sum;
				}
		SweepPixels *want = [SweepPixels width:kSide height:kSide];
		for (int y = 0; y < kSide; y++)
			for (int x = 0; x < kSide; x++)
				for (int c = 0; c < 4; c++) {
					double s = 0;
					for (int i = -kR; i <= kR; i++)
						s += w[i + kR] * t[(clampi(y + i, 0, kSide - 1) * kSide + x) * 4 + c];
					[want at:x y:y][c] = (uint8_t)lround(s / sum);
				}
		return sweepCompare(readRGBA(out), want, CGRectNull, 4);
	});

	mpsCase(@"mps.sobel", (SweepTol){2, 0.5}, ^SweepResult *(id<MTLDevice> dev) {
		id<MTLTexture> in = source(dev, pattern), out = texture(dev, kSide, kSide, MTLPixelFormatR32Float);
		MPSImageSobel *k = [[MPSImageSobel alloc] initWithDevice:dev];
		k.edgeMode = MPSImageEdgeModeClamp;
		NSString *err = run(dev, out, ^(id<MTLCommandBuffer> cb) {
			[k encodeToCommandBuffer:cb sourceTexture:in destinationTexture:out];
		});
		if (err != nil)
			return [SweepResult fail:err];
		NSMutableData *got = [NSMutableData dataWithLength:kSide * kSide * sizeof(float)];
		[out getBytes:got.mutableBytes
		    bytesPerRow:kSide * sizeof(float)
		     fromRegion:MTLRegionMake2D(0, 0, kSide, kSide)
		    mipmapLevel:0];
		const float *g = got.bytes;
		double worst = 0, total = 0;
		for (int y = 0; y < kSide; y++)
			for (int x = 0; x < kSide; x++) {
				double l[3][3];
				for (int j = -1; j <= 1; j++)
					for (int i = -1; i <= 1; i++) {
						const uint8_t *q = [pattern at:clampi(x + i, 0, kSide - 1) y:clampi(y + j, 0, kSide - 1)];
						l[j + 1][i + 1] = (0.299 * q[0] + 0.587 * q[1] + 0.114 * q[2]) / 255;
					}
				const double gx = (l[0][2] + 2 * l[1][2] + l[2][2]) - (l[0][0] + 2 * l[1][0] + l[2][0]);
				const double gy = (l[2][0] + 2 * l[2][1] + l[2][2]) - (l[0][0] + 2 * l[0][1] + l[0][2]);
				const double d = fabs(g[y * kSide + x] - sqrt(gx * gx + gy * gy)) * 255;
				worst = fmax(worst, d), total += d;
			}
		SweepResult *r = [SweepResult max:worst mean:total / (kSide * kSide)];
		r.note = @"in 8-bit levels of the magnitude";
		return r;
	});

	mpsCase(@"mps.bilinear", (SweepTol){2, 0.5}, ^SweepResult *(id<MTLDevice> dev) {
		id<MTLTexture> in = source(dev, pattern), out = texture(dev, kSide / 2, kSide / 2, MTLPixelFormatRGBA8Unorm);
		MPSImageBilinearScale *k = [[MPSImageBilinearScale alloc] initWithDevice:dev];
		NSString *err = run(dev, out, ^(id<MTLCommandBuffer> cb) {
			[k encodeToCommandBuffer:cb sourceTexture:in destinationTexture:out];
		});
		if (err != nil)
			return [SweepResult fail:err];
		SweepPixels *want = [SweepPixels width:kSide / 2 height:kSide / 2];
		for (size_t y = 0; y < kSide / 2; y++)
			for (size_t x = 0; x < kSide / 2; x++)
				for (int c = 0; c < 4; c++)
					[want at:x y:y][c] = (uint8_t)(([pattern at:2 * x y:2 * y][c] + [pattern at:2 * x + 1 y:2 * y][c] +
					                                [pattern at:2 * x y:2 * y + 1][c] +
					                                [pattern at:2 * x + 1 y:2 * y + 1][c] + 2) /
					                               4);
		return sweepCompare(readRGBA(out), want, CGRectNull, 4);
	});

	// Exact: counting needs no arithmetic.
	mpsCase(@"mps.histogram", (SweepTol){0, 0}, ^SweepResult *(id<MTLDevice> dev) {
		id<MTLTexture> in = source(dev, pattern);
		MPSImageHistogramInfo info = {256, NO, {0, 0, 0, 0}, {1, 1, 1, 1}};
		MPSImageHistogram *k = [[MPSImageHistogram alloc] initWithDevice:dev histogramInfo:&info];
		const size_t bytes = [k histogramSizeForSourceFormat:in.pixelFormat];
		id<MTLBuffer> hist = [dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
		memset(hist.contents, 0xff, bytes);
		NSString *err = run(dev, nil, ^(id<MTLCommandBuffer> cb) {
			[k encodeToCommandBuffer:cb sourceTexture:in histogram:hist histogramOffset:0];
		});
		if (err != nil)
			return [SweepResult fail:err];
		if (bytes < 3 * 256 * sizeof(uint32_t))
			return [SweepResult fail:[NSString stringWithFormat:@"a histogram of %zu bytes", bytes]];
		uint32_t want[3][256] = {{0}};
		for (size_t y = 0; y < kSide; y++)
			for (size_t x = 0; x < kSide; x++)
				for (int c = 0; c < 3; c++)
					want[c][[pattern at:x y:y][c]]++;
		const uint32_t *got = hist.contents;
		double worst = 0, total = 0;
		for (int c = 0; c < 3; c++)
			for (int i = 0; i < 256; i++) {
				const double d = fabs((double)got[c * 256 + i] - want[c][i]);
				worst = fmax(worst, d), total += d;
			}
		SweepResult *r = [SweepResult max:worst mean:total / (3 * 256)];
		r.note = @"in counts";
		return r;
	});

	// Starting values: float32 sums of 256 products in another order, 1e-4 of the largest.
	mpsCase(@"mps.matmul", (SweepTol){100, 20}, ^SweepResult *(id<MTLDevice> dev) {
		enum { kN = 256 };
		const size_t bytes = kN * kN * sizeof(float);
		id<MTLBuffer> a = [dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
		id<MTLBuffer> b = [dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
		id<MTLBuffer> c = [dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
		uint32_t seed = 12345;
		float *pa = a.contents, *pb = b.contents;
		for (size_t i = 0; i < kN * kN; i++) {
			seed = seed * 1664525u + 1013904223u;
			pa[i] = (float)((seed >> 8) / 16777216.0 * 2 - 1);
			seed = seed * 1664525u + 1013904223u;
			pb[i] = (float)((seed >> 8) / 16777216.0 * 2 - 1);
		}
		memset(c.contents, 0, bytes);
		MPSMatrixDescriptor *d = [MPSMatrixDescriptor matrixDescriptorWithRows:kN
		                                                               columns:kN
		                                                              rowBytes:kN * sizeof(float)
		                                                              dataType:MPSDataTypeFloat32];
		MPSMatrix *ma = [[MPSMatrix alloc] initWithBuffer:a descriptor:d];
		MPSMatrix *mb = [[MPSMatrix alloc] initWithBuffer:b descriptor:d];
		MPSMatrix *mc = [[MPSMatrix alloc] initWithBuffer:c descriptor:d];
		MPSMatrixMultiplication *k = [[MPSMatrixMultiplication alloc] initWithDevice:dev
		                                                                 resultRows:kN
		                                                              resultColumns:kN
		                                                            interiorColumns:kN];
		NSString *err = run(dev, nil, ^(id<MTLCommandBuffer> cb) {
			[k encodeToCommandBuffer:cb leftMatrix:ma rightMatrix:mb resultMatrix:mc];
		});
		if (err != nil)
			return [SweepResult fail:err];
		NSMutableData *ref = [NSMutableData dataWithLength:bytes];
		float *pr = ref.mutableBytes;
		cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, kN, kN, kN, 1, pa, kN, pb, kN, 0, pr, kN);
		const float *pc = c.contents;
		double big = 0, worst = 0, total = 0;
		for (size_t i = 0; i < kN * kN; i++)
			big = fmax(big, fabs(pr[i]));
		for (size_t i = 0; i < kN * kN; i++) {
			const double d = fabs(pc[i] - pr[i]) / big * 1e6;
			worst = fmax(worst, d), total += d;
		}
		SweepResult *r = [SweepResult max:worst mean:total / (kN * kN)];
		r.note = @"in ppm of the largest magnitude";
		return r;
	});
}
