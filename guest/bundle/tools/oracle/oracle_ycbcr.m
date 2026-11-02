// oracle_ycbcr: what a device's Metal reads through a two-plane Y'CbCr texture
// (MTLPixelFormat 500, YCBCR8_420_2P) over plane 0 of a 4:2:0 IOSurface, by the surface's range
// and Y'CbCr matrix.
//   oracle_ycbcr DEVICE              every case, each in a child process
//   oracle_ycbcr DEVICE --case N     case N alone (what the children run)
// The cases: a 64 x 64 CVPixelBuffer over an IOSurface, of OSType 420v (video range) or 420f
// (full range), its matrix set as CoreVideo sets it (kCVImageBufferYCbCrMatrixKey, propagated:
// "cv") or on the surface itself (IOSurfaceSetValue of "IOSurfaceYCbCrMatrix": "surface"), to
// ITU_R_709_2 or ITU_R_601_4, or left unset ("none"). Each quadrant of the surface holds one
// patch of known codes (Y, Cb, Cr), luma and chroma alike.
// Each case prints one line:
//   case N OSTYPE HOW MATRIX: surface value V; texture made (format F) | nil | died(signal)
// and, for a texture made, a line a patch: the codes, the texel `read()` reads at the patch's
// middle, BT.601's and BT.709's conversion of the codes by the OSType's range, and which of the
// two the read is within one level (1/255) of on every channel.
// Build: see README.md.
#import "oracle_common.h"
#import <CoreVideo/CoreVideo.h>
#import <IOSurface/IOSurface.h>
#include <spawn.h>
#include <sys/wait.h>

extern char **environ;

enum { kSize = 64, kYCbCr8_420_2P = 500 };

struct Case {
	OSType ostype;
	const char *how;    // "cv", "surface" or "none"
	CFStringRef matrix; // the attachment's value, or NULL
};

static const struct Case kCases[] = {
    {'420v', "cv", CFSTR("ITU_R_709_2")},      {'420v', "cv", CFSTR("ITU_R_601_4")},
    {'420v', "surface", CFSTR("ITU_R_709_2")}, {'420v', "surface", CFSTR("ITU_R_601_4")},
    {'420v', "none", NULL},                    {'420f', "cv", CFSTR("ITU_R_709_2")},
    {'420f', "cv", CFSTR("ITU_R_601_4")},      {'420f', "surface", CFSTR("ITU_R_709_2")},
    {'420f', "surface", CFSTR("ITU_R_601_4")}, {'420f', "none", NULL},
};
enum { kCaseCount = sizeof kCases / sizeof kCases[0] };

// The patches, one a quadrant (top left, top right, bottom left, bottom right): codes whose
// colour is inside 0.03 to 0.97 on every channel by either matrix and either range, so nothing
// clamps, and which BT.601 and BT.709 read at least 0.08 apart.
static const uint8_t kPatch[4][3] = {
    {100, 198, 198},
    {154, 60, 60},
    {130, 174, 186},
    {172, 108, 60},
};

static void ostypeText(OSType t, char out[5])
{
	out[0] = (char)(t >> 24);
	out[1] = (char)(t >> 16);
	out[2] = (char)(t >> 8);
	out[3] = (char)t;
	out[4] = 0;
}

// The codes' R'G'B' by a matrix of luma weights kr and kb, over the OSType's range.
static void convert(const uint8_t c[3], bool full, double kr, double kb, double rgb[3])
{
	double y = full ? c[0] / 255.0 : (c[0] - 16) / 219.0;
	double pb = full ? (c[1] - 128) / 255.0 : (c[1] - 128) / 224.0;
	double pr = full ? (c[2] - 128) / 255.0 : (c[2] - 128) / 224.0;
	double r = y + 2 * (1 - kr) * pr;
	double b = y + 2 * (1 - kb) * pb;
	rgb[0] = r;
	rgb[1] = (y - kr * r - kb * b) / (1 - kr - kb);
	rgb[2] = b;
}

static double maxDiff(const float got[3], const double want[3])
{
	double m = 0;
	for (int i = 0; i < 3; i++)
		m = fmax(m, fabs(got[i] - want[i]));
	return m;
}

static int runCase(id<MTLDevice> d, const char *argv0, int n)
{
	const struct Case *c = &kCases[n];
	char ost[5];
	ostypeText(c->ostype, ost);
	char matrixText[32] = "-";
	if (c->matrix)
		CFStringGetCString(c->matrix, matrixText, sizeof matrixText, kCFStringEncodingUTF8);

	NSDictionary *attrs = @{(id)kCVPixelBufferIOSurfacePropertiesKey : @{}};
	CVPixelBufferRef pb = NULL;
	CVReturn cr = CVPixelBufferCreate(kCFAllocatorDefault, kSize, kSize, c->ostype,
	                                  (__bridge CFDictionaryRef)attrs, &pb);
	if (cr != kCVReturnSuccess || !pb) {
		printf("case %d %s %s %s: no pixel buffer (%d)\n", n, ost, c->how, matrixText, cr);
		return 1;
	}
	CVPixelBufferLockBaseAddress(pb, 0);
	uint8_t *luma = CVPixelBufferGetBaseAddressOfPlane(pb, 0);
	size_t lumaRow = CVPixelBufferGetBytesPerRowOfPlane(pb, 0);
	uint8_t *chroma = CVPixelBufferGetBaseAddressOfPlane(pb, 1);
	size_t chromaRow = CVPixelBufferGetBytesPerRowOfPlane(pb, 1);
	for (int y = 0; y < kSize; y++)
		for (int x = 0; x < kSize; x++)
			luma[y * lumaRow + x] = kPatch[(y >= kSize / 2) * 2 + (x >= kSize / 2)][0];
	for (int y = 0; y < kSize / 2; y++)
		for (int x = 0; x < kSize / 2; x++) {
			const uint8_t *p = kPatch[(y >= kSize / 4) * 2 + (x >= kSize / 4)];
			chroma[y * chromaRow + 2 * x] = p[1];
			chroma[y * chromaRow + 2 * x + 1] = p[2];
		}
	CVPixelBufferUnlockBaseAddress(pb, 0);

	IOSurfaceRef s = CVPixelBufferGetIOSurface(pb);
	if (!s) {
		printf("case %d %s %s %s: no IOSurface under the pixel buffer\n", n, ost, c->how,
		       matrixText);
		return 1;
	}
	if (c->matrix && !strcmp(c->how, "cv"))
		CVBufferSetAttachment(pb, kCVImageBufferYCbCrMatrixKey, c->matrix,
		                      kCVAttachmentMode_ShouldPropagate);
	else if (c->matrix)
		IOSurfaceSetValue(s, CFSTR("IOSurfaceYCbCrMatrix"), c->matrix);
	char value[48] = "none";
	CFTypeRef v = IOSurfaceCopyValue(s, CFSTR("IOSurfaceYCbCrMatrix"));
	if (v) {
		if (CFGetTypeID(v) != CFStringGetTypeID() ||
		    !CFStringGetCString(v, value, sizeof value, kCFStringEncodingUTF8))
			snprintf(value, sizeof value, "(not a string)");
		CFRelease(v);
	}

	printf("case %d %s %s %s: surface value %s; ", n, ost, c->how, matrixText, value);
	fflush(stdout);
	MTLTextureDescriptor *td = [MTLTextureDescriptor
	    texture2DDescriptorWithPixelFormat:(MTLPixelFormat)kYCbCr8_420_2P
	                                 width:kSize
	                                height:kSize
	                             mipmapped:NO];
	td.usage = MTLTextureUsageShaderRead;
	td.storageMode = d.hasUnifiedMemory ? MTLStorageModeShared : MTLStorageModeManaged;
	id<MTLTexture> t = [d newTextureWithDescriptor:td iosurface:s plane:0];
	if (!t) {
		printf("texture nil\n");
		CVPixelBufferRelease(pb);
		return 0;
	}
	printf("texture made (format %lu, storage %lu)\n", (unsigned long)t.pixelFormat,
	       (unsigned long)t.storageMode);

	id<MTLLibrary> lib = oracle_library(d, argv0, "oracle");
	id<MTLFunction> f = [lib newFunctionWithName:@"ycbcr_read"];
	NSError *err = nil;
	id<MTLComputePipelineState> ps = f ? [d newComputePipelineStateWithFunction:f error:&err] : nil;
	if (!ps) {
		printf("  ! no pipeline: %s\n", err ? err.description.UTF8String : "no function");
		CVPixelBufferRelease(pb);
		return 1;
	}
	id<MTLBuffer> out = [d newBufferWithLength:4 * sizeof(float) * 4
	                                   options:MTLResourceStorageModeShared];
	id<MTLCommandQueue> q = [d newCommandQueue];
	id<MTLCommandBuffer> cb = [q commandBuffer];
	id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
	[e setComputePipelineState:ps];
	[e setTexture:t atIndex:0];
	[e setBuffer:out offset:0 atIndex:0];
	[e dispatchThreads:MTLSizeMake(4, 1, 1) threadsPerThreadgroup:MTLSizeMake(4, 1, 1)];
	[e endEncoding];
	oracle_commit_wait(cb, NULL, NULL);

	const bool full = c->ostype == '420f';
	const float *got = out.contents;
	for (int i = 0; i < 4; i++) {
		double r601[3], r709[3];
		convert(kPatch[i], full, 0.299, 0.114, r601);
		convert(kPatch[i], full, 0.2126, 0.0722, r709);
		const float *g = got + 4 * i;
		double d601 = maxDiff(g, r601), d709 = maxDiff(g, r709);
		const char *which = d601 <= 1 / 255.0 && d709 <= 1 / 255.0 ? "both"
		                    : d601 <= 1 / 255.0                     ? "BT.601"
		                    : d709 <= 1 / 255.0                     ? "BT.709"
		                                                            : "neither";
		printf("  patch Y %3u Cb %3u Cr %3u: read %.4f %.4f %.4f %.4f; BT.601 %.4f %.4f %.4f "
		       "(off %.4f); BT.709 %.4f %.4f %.4f (off %.4f) -> %s\n",
		       kPatch[i][0], kPatch[i][1], kPatch[i][2], g[0], g[1], g[2], g[3], r601[0], r601[1],
		       r601[2], d601, r709[0], r709[1], r709[2], d709, which);
	}
	CVPixelBufferRelease(pb);
	return 0;
}

int main(int argc, char **argv)
{
	setvbuf(stdout, NULL, _IOLBF, 0);
	if (argc < 2) {
		fprintf(stderr, "usage: %s DEVICE [--case N]\n", argv[0]);
		return 2;
	}
	id<MTLDevice> d = oracle_device(argv[1]);
	if (argc > 3 && !strcmp(argv[2], "--case")) {
		int n = atoi(argv[3]);
		return n >= 0 && n < kCaseCount ? runCase(d, argv[0], n) : 2;
	}
	oracle_header("oracle_ycbcr", d);
	d = nil;
	for (int n = 0; n < kCaseCount; n++) {
		char num[8];
		snprintf(num, sizeof num, "%d", n);
		char *args[] = {argv[0], argv[1], "--case", num, NULL};
		fflush(stdout);
		pid_t pid;
		if (posix_spawn(&pid, argv[0], NULL, NULL, args, environ))
			return 1;
		int st;
		waitpid(pid, &st, 0);
		if (WIFSIGNALED(st))
			printf("  => case %d died(%d)\n", n, WTERMSIG(st));
		else if (WEXITSTATUS(st))
			printf("  => case %d exited %d\n", n, WEXITSTATUS(st));
	}
	return 0;
}
