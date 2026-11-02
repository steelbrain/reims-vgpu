// oracle_ca_ycbcr: which Y'CbCr matrix the frameworks convert a two-plane 4:2:0 surface by, on a
// device: CoreAnimation (a CARenderer drawing a layer whose contents are the surface, into a Metal
// texture of the device, on a queue of the device) and VideoToolbox (VTPixelTransferSession to a
// 32BGRA pixel buffer; it picks its own device, so its lines are the same whatever DEVICE is).
//   oracle_ca_ycbcr DEVICE
// The surfaces: 64 x 64 CVPixelBuffers over IOSurfaces, of OSType x420 (10-bit, video range) or
// 420v (8-bit, video range), the matrix attachment (kCVImageBufferYCbCrMatrixKey, propagated to the
// surface as CoreVideo does) ITU_R_709_2, ITU_R_601_4 or none. Each quadrant holds one patch of
// known codes (the 10-bit codes are the 8-bit ones times 4, in the high bits of each 16-bit word).
// Two renders of each surface: "plain", no colour space on the surface or the renderer, and
// "managed", the surface's primaries and transfer function ITU_R_709_2 and the renderer's colour
// space sRGB.
// Each case prints one line a patch: the colour read at the patch's middle, BT.601's and BT.709's
// conversion of the codes, and which of the two the read is within two levels (2/255) of on every
// channel. A managed render is colour-matched, so its reads are compared across matrices, not with
// the conversions: the line ends "tag-709 vs tag-601 differ by D".
// Build: see README.md.
#import "oracle_common.h"
#import <CoreVideo/CoreVideo.h>
#import <IOSurface/IOSurface.h>
#import <QuartzCore/QuartzCore.h>
#import <VideoToolbox/VideoToolbox.h>

enum { kSize = 64 };

static const uint8_t kPatch[4][3] = {
    {100, 198, 198},
    {154, 60, 60},
    {130, 174, 186},
    {172, 108, 60},
};

static const OSType kTypes[] = {'x420', '420v'};
static CFStringRef matrixTag(int m)
{
	return m == 0 ? kCVImageBufferYCbCrMatrix_ITU_R_709_2
	       : m == 1 ? kCVImageBufferYCbCrMatrix_ITU_R_601_4
	                : NULL;
}
static const char *const kTagName[3] = {"709", "601", "none"};

static void convert(const uint8_t c[3], double kr, double kb, double rgb[3])
{
	double y = (c[0] - 16) / 219.0, pb = (c[1] - 128) / 224.0, pr = (c[2] - 128) / 224.0;
	double r = y + 2 * (1 - kr) * pr;
	double b = y + 2 * (1 - kb) * pb;
	rgb[0] = r;
	rgb[1] = (y - kr * r - kb * b) / (1 - kr - kb);
	rgb[2] = b;
}

static CVPixelBufferRef makeSurface(OSType type, int tag, bool managed)
{
	NSDictionary *attrs = @{(id)kCVPixelBufferIOSurfacePropertiesKey : @{}};
	CVPixelBufferRef pb = NULL;
	if (CVPixelBufferCreate(kCFAllocatorDefault, kSize, kSize, type, (__bridge CFDictionaryRef)attrs,
	                        &pb) != kCVReturnSuccess)
		return NULL;
	const bool ten = type == 'x420';
	CVPixelBufferLockBaseAddress(pb, 0);
	uint8_t *luma = CVPixelBufferGetBaseAddressOfPlane(pb, 0);
	size_t lumaRow = CVPixelBufferGetBytesPerRowOfPlane(pb, 0);
	uint8_t *chroma = CVPixelBufferGetBaseAddressOfPlane(pb, 1);
	size_t chromaRow = CVPixelBufferGetBytesPerRowOfPlane(pb, 1);
	for (int y = 0; y < kSize; y++)
		for (int x = 0; x < kSize; x++) {
			uint8_t v = kPatch[(y >= kSize / 2) * 2 + (x >= kSize / 2)][0];
			if (ten)
				((uint16_t *)(luma + y * lumaRow))[x] = (uint16_t)((v * 4) << 6);
			else
				luma[y * lumaRow + x] = v;
		}
	for (int y = 0; y < kSize / 2; y++)
		for (int x = 0; x < kSize / 2; x++) {
			const uint8_t *p = kPatch[(y >= kSize / 4) * 2 + (x >= kSize / 4)];
			if (ten) {
				uint16_t *row = (uint16_t *)(chroma + y * chromaRow);
				row[2 * x] = (uint16_t)((p[1] * 4) << 6);
				row[2 * x + 1] = (uint16_t)((p[2] * 4) << 6);
			} else {
				chroma[y * chromaRow + 2 * x] = p[1];
				chroma[y * chromaRow + 2 * x + 1] = p[2];
			}
		}
	CVPixelBufferUnlockBaseAddress(pb, 0);
	if (matrixTag(tag))
		CVBufferSetAttachment(pb, kCVImageBufferYCbCrMatrixKey, matrixTag(tag),
		                      kCVAttachmentMode_ShouldPropagate);
	if (managed) {
		CVBufferSetAttachment(pb, kCVImageBufferColorPrimariesKey,
		                      kCVImageBufferColorPrimaries_ITU_R_709_2,
		                      kCVAttachmentMode_ShouldPropagate);
		CVBufferSetAttachment(pb, kCVImageBufferTransferFunctionKey,
		                      kCVImageBufferTransferFunction_ITU_R_709_2,
		                      kCVAttachmentMode_ShouldPropagate);
	}
	return pb;
}

// The four patches' colours as CoreAnimation draws the surface into a BGRA8 texture of `d`, which a
// clear has made magenta first, so a render that writes nothing shows as magenta.
static bool caRender(id<MTLDevice> d, id<MTLCommandQueue> q, CVPixelBufferRef pb, bool managed,
                     float out[4][4])
{
	MTLTextureDescriptor *td =
	    [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
	                                                       width:kSize
	                                                      height:kSize
	                                                   mipmapped:NO];
	td.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
	td.storageMode = MTLStorageModePrivate;
	id<MTLTexture> t = [d newTextureWithDescriptor:td];
	MTLRenderPassDescriptor *rp = [MTLRenderPassDescriptor renderPassDescriptor];
	rp.colorAttachments[0].texture = t;
	rp.colorAttachments[0].loadAction = MTLLoadActionClear;
	rp.colorAttachments[0].storeAction = MTLStoreActionStore;
	rp.colorAttachments[0].clearColor = MTLClearColorMake(1, 0, 1, 1);
	id<MTLCommandBuffer> clear = [q commandBuffer];
	[[clear renderCommandEncoderWithDescriptor:rp] endEncoding];
	oracle_commit_wait(clear, NULL, NULL);
	NSMutableDictionary *opts = [NSMutableDictionary dictionary];
	opts[kCARendererMetalCommandQueue] = q;
	CGColorSpaceRef srgb = NULL;
	if (managed) {
		srgb = CGColorSpaceCreateWithName(kCGColorSpaceSRGB);
		opts[kCARendererColorSpace] = (__bridge id)srgb;
	}
	CARenderer *r = [CARenderer rendererWithMTLTexture:t options:opts];
	if (srgb)
		CGColorSpaceRelease(srgb);
	if (!r)
		return false;
	// The renderer draws the committed layer tree: a process with no run loop flushes its implicit
	// transaction itself, or the render draws nothing.
	[CATransaction begin];
	[CATransaction setDisableActions:YES];
	CALayer *layer = [CALayer layer];
	layer.frame = CGRectMake(0, 0, kSize, kSize);
	layer.contents = (__bridge id)CVPixelBufferGetIOSurface(pb);
	layer.contentsGravity = kCAGravityResize;
	r.layer = layer;
	r.bounds = CGRectMake(0, 0, kSize, kSize);
	[CATransaction commit];
	[CATransaction flush];
	[r beginFrameAtTime:CACurrentMediaTime() timeStamp:NULL];
	[r addUpdateRect:r.bounds];
	[r render];
	[r endFrame];
	id<MTLBuffer> buf = [d newBufferWithLength:kSize * kSize * 4 options:MTLResourceStorageModeShared];
	id<MTLCommandBuffer> cb = [q commandBuffer];
	id<MTLBlitCommandEncoder> b = [cb blitCommandEncoder];
	[b copyFromTexture:t
	             sourceSlice:0
	             sourceLevel:0
	            sourceOrigin:MTLOriginMake(0, 0, 0)
	              sourceSize:MTLSizeMake(kSize, kSize, 1)
	                toBuffer:buf
	       destinationOffset:0
	  destinationBytesPerRow:kSize * 4
	destinationBytesPerImage:kSize * kSize * 4];
	[b endEncoding];
	oracle_commit_wait(cb, NULL, NULL);
	// A layer's origin is its bottom left: the surface's top row is the texture's bottom one.
	const uint8_t *px = buf.contents;
	for (int i = 0; i < 4; i++) {
		int x = (i & 1) ? 48 : 16, y = (i & 2) ? 16 : 48;
		const uint8_t *p = px + (y * kSize + x) * 4;
		out[i][0] = p[2] / 255.0f;
		out[i][1] = p[1] / 255.0f;
		out[i][2] = p[0] / 255.0f;
		out[i][3] = p[3] / 255.0f;
	}
	return true;
}

// The four patches' colours as VideoToolbox transfers the surface to a 32BGRA pixel buffer.
static bool vtTransfer(CVPixelBufferRef pb, float out[4][4])
{
	VTPixelTransferSessionRef s = NULL;
	if (VTPixelTransferSessionCreate(kCFAllocatorDefault, &s) != noErr)
		return false;
	NSDictionary *attrs = @{(id)kCVPixelBufferIOSurfacePropertiesKey : @{}};
	CVPixelBufferRef dst = NULL;
	CVPixelBufferCreate(kCFAllocatorDefault, kSize, kSize, kCVPixelFormatType_32BGRA,
	                    (__bridge CFDictionaryRef)attrs, &dst);
	OSStatus st = dst ? VTPixelTransferSessionTransferImage(s, pb, dst) : -1;
	VTPixelTransferSessionInvalidate(s);
	CFRelease(s);
	if (st != noErr) {
		printf("  ! VTPixelTransferSessionTransferImage: %d\n", (int)st);
		if (dst)
			CVPixelBufferRelease(dst);
		return false;
	}
	CVPixelBufferLockBaseAddress(dst, kCVPixelBufferLock_ReadOnly);
	const uint8_t *base = CVPixelBufferGetBaseAddress(dst);
	size_t row = CVPixelBufferGetBytesPerRow(dst);
	for (int i = 0; i < 4; i++) {
		int x = (i & 1) ? 48 : 16, y = (i & 2) ? 48 : 16;
		const uint8_t *p = base + y * row + x * 4;
		out[i][0] = p[2] / 255.0f;
		out[i][1] = p[1] / 255.0f;
		out[i][2] = p[0] / 255.0f;
		out[i][3] = p[3] / 255.0f;
	}
	CVPixelBufferUnlockBaseAddress(dst, kCVPixelBufferLock_ReadOnly);
	CVPixelBufferRelease(dst);
	return true;
}

static double offBy(const float got[4], const double want[3])
{
	double m = 0;
	for (int i = 0; i < 3; i++)
		m = fmax(m, fabs(got[i] - want[i]));
	return m;
}

static void printReads(const char *path, OSType type, int tag, bool managed, float got[4][4])
{
	char ost[5] = {(char)(type >> 24), (char)(type >> 16), (char)(type >> 8), (char)type, 0};
	for (int i = 0; i < 4; i++) {
		double r601[3], r709[3];
		convert(kPatch[i], 0.299, 0.114, r601);
		convert(kPatch[i], 0.2126, 0.0722, r709);
		double d601 = offBy(got[i], r601), d709 = offBy(got[i], r709);
		const double tol = 2 / 255.0;
		const char *which = managed           ? "(managed)"
		                    : d601 <= tol && d709 <= tol ? "both"
		                    : d601 <= tol               ? "BT.601"
		                    : d709 <= tol               ? "BT.709"
		                                                : "neither";
		printf("%s %s tag-%s %s patch %d: read %.4f %.4f %.4f %.4f; off BT.601 %.4f, BT.709 %.4f -> %s\n",
		       path, ost, kTagName[tag], managed ? "managed" : "plain", i, got[i][0], got[i][1],
		       got[i][2], got[i][3], d601, d709, which);
	}
}

int main(int argc, char **argv)
{
	setvbuf(stdout, NULL, _IOLBF, 0);
	if (argc < 2) {
		fprintf(stderr, "usage: %s DEVICE\n", argv[0]);
		return 2;
	}
	id<MTLDevice> d = oracle_device(argv[1]);
	oracle_header("oracle_ca_ycbcr", d);
	id<MTLCommandQueue> q = [d newCommandQueue];
	for (int managed = 0; managed < 2; managed++) {
		for (size_t ti = 0; ti < sizeof kTypes / sizeof kTypes[0]; ti++) {
			float byTag[3][4][4];
			for (int tag = 0; tag < 3; tag++) {
				@autoreleasepool {
					CVPixelBufferRef pb = makeSurface(kTypes[ti], tag, managed);
					if (!pb) {
						printf("  ! no pixel buffer\n");
						continue;
					}
					char value[48] = "none";
					CFTypeRef v = IOSurfaceCopyValue(CVPixelBufferGetIOSurface(pb),
					                                 CFSTR("IOSurfaceYCbCrMatrix"));
					if (v) {
						if (CFGetTypeID(v) == CFStringGetTypeID())
							CFStringGetCString(v, value, sizeof value, kCFStringEncodingUTF8);
						CFRelease(v);
					}
					printf("# surface value %s\n", value);
					if (caRender(d, q, pb, managed, byTag[tag]))
						printReads("CA", kTypes[ti], tag, managed, byTag[tag]);
					else
						printf("CA: no renderer\n");
					float vt[4][4];
					if (vtTransfer(pb, vt))
						printReads("VT", kTypes[ti], tag, managed, vt);
					CVPixelBufferRelease(pb);
				}
			}
			double diff = 0;
			for (int i = 0; i < 4; i++)
				for (int c = 0; c < 3; c++)
					diff = fmax(diff, fabs(byTag[0][i][c] - byTag[1][i][c]));
			printf("CA %c%c%c%c %s: tag-709 vs tag-601 differ by %.4f\n", (char)(kTypes[ti] >> 24),
			       (char)(kTypes[ti] >> 16), (char)(kTypes[ti] >> 8), (char)kTypes[ti],
			       managed ? "managed" : "plain", diff);
		}
	}
	return 0;
}
