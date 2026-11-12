// oracle_rows: the bytes per row and plane offsets IOSurface works out for a surface made with
// none named (IOSurfaceCreate, as an app or a framework calls it with no kIOSurfaceBytesPerRow),
// the alignments IOSurface answers for them, CoreVideo's rows for an IOSurface-backed pixel
// buffer, and the device's own linear alignments, to tell whether a Mac's surface rows follow its
// GPU or IOSurface's own default. Nothing here draws.
//   oracle_rows DEVICE
// Lines:
//   align KEY A                   IOSurfaceGetPropertyAlignment(KEY)
//   alignrow N => M               IOSurfaceAlignProperty(kIOSurfaceBytesPerRow, N)
//   surface 'OSTY' WxH bpe B => rows R size S
//                                 IOSurfaceCreate with width, height, bytes an element and
//                                 OSType alone
//   planes 'OSTY' WxH => plane P WxH bpe B rows R offset O ...; size S
//                                 IOSurfaceCreate with kIOSurfacePlaneInfo, each plane's width,
//                                 height and bytes an element alone
//   cvpixelbuffer 'OSTY' WxH => plane P rows R offset O ...
//                                 CVPixelBufferCreate with an empty kCVPixelBufferIOSurfacePropertiesKey
//   metal FMT minimumLinearTextureAlignment N
//   metal KEY N                   the device's SPI getter KEY, where it answers it
// Build: see README.md.
#import "oracle_common.h"
#import <CoreVideo/CoreVideo.h>
#import <IOSurface/IOSurface.h>
#import <objc/message.h>

static void align(const char *name, CFStringRef key)
{
	printf("align %s %zu\n", name, IOSurfaceGetPropertyAlignment(key));
}

static void plain(uint32_t ostype, size_t w, size_t h, size_t bpe)
{
	NSDictionary *props = @{
		(id)kIOSurfaceWidth : @(w),
		(id)kIOSurfaceHeight : @(h),
		(id)kIOSurfaceBytesPerElement : @(bpe),
		(id)kIOSurfacePixelFormat : @(ostype),
	};
	IOSurfaceRef s = IOSurfaceCreate((__bridge CFDictionaryRef)props);
	printf("surface '%c%c%c%c' %zux%zu bpe %zu => ", (char)(ostype >> 24), (char)(ostype >> 16),
	       (char)(ostype >> 8), (char)ostype, w, h, bpe);
	if (!s) {
		printf("no surface\n");
		return;
	}
	printf("rows %zu size %zu\n", IOSurfaceGetBytesPerRow(s), IOSurfaceGetAllocSize(s));
	CFRelease(s);
}

static void planes(uint32_t ostype, size_t w, size_t h, size_t n, const size_t *pw, const size_t *ph,
                   const size_t *pbpe)
{
	NSMutableArray *info = [NSMutableArray array];
	for (size_t p = 0; p < n; p++) {
		[info addObject:@{
			(id)kIOSurfacePlaneWidth : @(pw[p]),
			(id)kIOSurfacePlaneHeight : @(ph[p]),
			(id)kIOSurfacePlaneBytesPerElement : @(pbpe[p]),
		}];
	}
	NSDictionary *props = @{
		(id)kIOSurfaceWidth : @(w),
		(id)kIOSurfaceHeight : @(h),
		(id)kIOSurfacePixelFormat : @(ostype),
		(id)kIOSurfacePlaneInfo : info,
	};
	IOSurfaceRef s = IOSurfaceCreate((__bridge CFDictionaryRef)props);
	printf("planes '%c%c%c%c' %zux%zu =>", (char)(ostype >> 24), (char)(ostype >> 16),
	       (char)(ostype >> 8), (char)ostype, w, h);
	if (!s) {
		printf(" no surface\n");
		return;
	}
	for (size_t p = 0; p < IOSurfaceGetPlaneCount(s); p++)
		printf(" plane %zu %zux%zu bpe %zu rows %zu offset %zu;", p, IOSurfaceGetWidthOfPlane(s, p),
		       IOSurfaceGetHeightOfPlane(s, p), IOSurfaceGetBytesPerElementOfPlane(s, p),
		       IOSurfaceGetBytesPerRowOfPlane(s, p),
		       (size_t)((uint8_t *)IOSurfaceGetBaseAddressOfPlane(s, p) -
		                (uint8_t *)IOSurfaceGetBaseAddress(s)));
	printf(" size %zu\n", IOSurfaceGetAllocSize(s));
	CFRelease(s);
}

static void pixelbuffer(OSType ostype, size_t w, size_t h)
{
	NSDictionary *attrs = @{(id)kCVPixelBufferIOSurfacePropertiesKey : @{}};
	CVPixelBufferRef pb = NULL;
	CVReturn r = CVPixelBufferCreate(kCFAllocatorDefault, w, h, ostype, (__bridge CFDictionaryRef)attrs,
	                                 &pb);
	printf("cvpixelbuffer '%c%c%c%c' %zux%zu =>", (char)(ostype >> 24), (char)(ostype >> 16),
	       (char)(ostype >> 8), (char)ostype, w, h);
	if (r != kCVReturnSuccess || !pb) {
		printf(" none (%d)\n", r);
		return;
	}
	IOSurfaceRef s = CVPixelBufferGetIOSurface(pb);
	if (!s) {
		printf(" no surface\n");
		CVPixelBufferRelease(pb);
		return;
	}
	size_t n = IOSurfaceGetPlaneCount(s);
	if (n == 0)
		printf(" rows %zu", IOSurfaceGetBytesPerRow(s));
	for (size_t p = 0; p < n; p++)
		printf(" plane %zu rows %zu offset %zu;", p, IOSurfaceGetBytesPerRowOfPlane(s, p),
		       (size_t)((uint8_t *)IOSurfaceGetBaseAddressOfPlane(s, p) -
		                (uint8_t *)IOSurfaceGetBaseAddress(s)));
	printf(" size %zu\n", IOSurfaceGetAllocSize(s));
	CVPixelBufferRelease(pb);
}

static void spi(id<MTLDevice> d, const char *name)
{
	SEL sel = sel_registerName(name);
	if (![d respondsToSelector:sel]) {
		printf("metal %s -\n", name);
		return;
	}
	NSUInteger v = ((NSUInteger(*)(id, SEL))objc_msgSend)(d, sel);
	printf("metal %s %lu\n", name, (unsigned long)v);
}

int main(int argc, char **argv)
{
	@autoreleasepool {
		if (argc < 2) {
			fprintf(stderr, "usage: oracle_rows DEVICE\n");
			return 2;
		}
		id<MTLDevice> d = oracle_device(argv[1]);
		oracle_header("oracle_rows", d);
		align("kIOSurfaceBytesPerRow", kIOSurfaceBytesPerRow);
		align("kIOSurfacePlaneBytesPerRow", kIOSurfacePlaneBytesPerRow);
		align("kIOSurfaceOffset", kIOSurfaceOffset);
		align("kIOSurfacePlaneOffset", kIOSurfacePlaneOffset);
		align("kIOSurfaceAllocSize", kIOSurfaceAllocSize);
		align("kIOSurfacePlaneSize", kIOSurfacePlaneSize);
		static const size_t rows[] = {1, 8, 64, 144, 192, 384, 1336, 3840, 4004};
		for (size_t i = 0; i < sizeof rows / sizeof rows[0]; i++)
			printf("alignrow %zu => %zu\n", rows[i], IOSurfaceAlignProperty(kIOSurfaceBytesPerRow, rows[i]));
		// RGBA16F ('RGhA', 8 bytes): the widths as3-xtx's shadows had, 18 the brief's.
		static const size_t rgha[][2] = {{1, 1}, {18, 28}, {48, 48}, {85, 162}, {167, 161}, {170, 134}};
		for (size_t i = 0; i < sizeof rgha / sizeof rgha[0]; i++)
			plain('RGhA', rgha[i][0], rgha[i][1], 8);
		static const size_t bgra[][2] = {{96, 96}, {463, 40}, {1001, 600}, {3662, 2000}};
		for (size_t i = 0; i < sizeof bgra / sizeof bgra[0]; i++)
			plain('BGRA', bgra[i][0], bgra[i][1], 4);
		plain('L008', 1920, 1080, 1);
		plain('L008', 1001, 1, 1);
		// A two-plane 420v: luma at 1 byte, chroma pairs at 2 at half size.
		static const size_t yuv[][2] = {{1920, 1080}, {18, 28}, {1001, 563}};
		for (size_t i = 0; i < sizeof yuv / sizeof yuv[0]; i++) {
			size_t w = yuv[i][0], h = yuv[i][1];
			size_t pw[] = {w, (w + 1) / 2}, ph[] = {h, (h + 1) / 2}, pbpe[] = {1, 2};
			planes('420v', w, h, 2, pw, ph, pbpe);
		}
		pixelbuffer(kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange, 1920, 1080);
		pixelbuffer(kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange, 18, 28);
		pixelbuffer(kCVPixelFormatType_32BGRA, 96, 96);
		pixelbuffer(kCVPixelFormatType_64RGBAHalf, 18, 28);
		static const struct {
			const char *name;
			MTLPixelFormat format;
		} formats[] = {{"RGBA16Float", MTLPixelFormatRGBA16Float},
		               {"BGRA8Unorm", MTLPixelFormatBGRA8Unorm},
		               {"R8Unorm", MTLPixelFormatR8Unorm}};
		for (size_t i = 0; i < sizeof formats / sizeof formats[0]; i++)
			printf("metal %s minimumLinearTextureAlignment %lu\n", formats[i].name,
			       (unsigned long)[d minimumLinearTextureAlignmentForPixelFormat:formats[i].format]);
		spi(d, "iosurfaceTextureAlignmentBytes");
		spi(d, "iosurfaceReadOnlyTextureAlignmentBytes");
		spi(d, "linearTextureAlignmentBytes");
		spi(d, "deviceLinearTextureAlignmentBytes");
	}
	return 0;
}
