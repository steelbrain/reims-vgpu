// oracle_texdesc: which texture descriptors the device's newTextureWithDescriptor:, a Private
// heap's newTextureWithDescriptor: and a Shared buffer's newTextureWithDescriptor:offset:
// bytesPerRow: make, answer nil for, or end the process at, by the descriptor alone: its type,
// size, depth, array length, levels, samples, pixel format, usage and storage. No Metal
// validation layer is asked for: what a guest's process meets.
//   oracle_texdesc DEVICE CREATOR            every case, each in a child process of its own
//   oracle_texdesc DEVICE CREATOR --case N   case N alone (what the children run)
//   oracle_texdesc DEVICE CREATOR --list     the cases, one a line, without running them
// CREATOR is `device`, `heap` (a Private heap of 256 MiB, placed by Metal) or `buffer` (a Shared
// buffer of 64 MiB, at offset 0, rows the texture's row of 4-byte texels rounded up to 256).
// The descriptor of a case starts as a 64 x 64 2D BGRA8Unorm texture, one level, one sample,
// depth and array length 1, usage ShaderRead, Private storage (Shared over a buffer), and changes
// what the case's name says. The header gives the device's recommendedMaxWorkingSetSize and
// maxBufferLength. Each case prints:
//   case N NAME | CREATOR | asked: fmt F type T WxHxD a A m M s S usage U storage S'
//   case N NAME => made: fmt F type T WxHxD a A m M s S usage U storage S | nil
// before which a made texture's line gives its allocatedSize, the device's currentAllocatedSize
// and the process's footprint, and a size case's at the device's creator the size and alignment
// heapTextureSizeAndAlignWithDescriptor: answers;
// and, where the case ends the process, Metal's words (its stderr is the child's stdout), then
//   => case N died on signal X | exited X
// Build: see README.md.
#import "oracle_common.h"
#include <mach/mach.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

extern char **environ;

enum { kDefault = -1 };

struct Case {
	const char *name;
	MTLPixelFormat format;
	MTLTextureType type;
	long width, height, depth, array, mips, samples;
	long usage;   // kDefault: ShaderRead
	int storage;  // kDefault: Private (Shared over a buffer)
	int big;      // 1: more bytes than the heap holds; not run over a heap
};

#define T2D MTLTextureType2D
#define BGRA8 MTLPixelFormatBGRA8Unorm
#define MS MTLTextureType2DMultisample
// A case of FORMAT, TYPE, W x H x DEPTH, ARRAY, MIPS and SAMPLES, all else the start's.
#define C(NAME, FORMAT, TYPE, W, H, DEPTH, ARRAY, MIPS, SAMPLES)                                   \
	{NAME, FORMAT, TYPE, W, H, DEPTH, ARRAY, MIPS, SAMPLES, kDefault, kDefault, 0}
// A 2D 64 x 64 case of FORMAT.
#define F(NAME, FORMAT) C(NAME, FORMAT, T2D, 64, 64, 1, 1, 1, 1)
// The same of USAGE and STORAGE.
#define CU(NAME, FORMAT, TYPE, SAMPLES, USAGE, STORAGE)                                            \
	{NAME, FORMAT, TYPE, 64, 64, 1, 1, 1, SAMPLES, USAGE, STORAGE, 0}
// A case of more bytes than the heap holds.
#define CB(NAME, FORMAT, TYPE, W, H, DEPTH, ARRAY)                                                 \
	{NAME, FORMAT, TYPE, W, H, DEPTH, ARRAY, 1, 1, kDefault, kDefault, 1}

static const struct Case kCases[] = {
    F("base", BGRA8),
    // Size.
    C("w0", BGRA8, T2D, 0, 64, 1, 1, 1, 1),
    C("h0", BGRA8, T2D, 64, 0, 1, 1, 1, 1),
    C("d0", BGRA8, T2D, 64, 64, 0, 1, 1, 1),
    C("a0", BGRA8, T2D, 64, 64, 1, 0, 1, 1),
    C("m0", BGRA8, T2D, 64, 64, 1, 1, 0, 1),
    C("s0", BGRA8, T2D, 64, 64, 1, 1, 1, 0),
    CB("w16384", BGRA8, T2D, 16384, 1, 1, 1),
    CB("w16385", BGRA8, T2D, 16385, 1, 1, 1),
    CB("h16385", BGRA8, T2D, 1, 16385, 1, 1),
    CB("w32768.h32768", BGRA8, T2D, 32768, 32768, 1, 1),
    CB("1d.w16385", BGRA8, MTLTextureType1D, 16385, 1, 1, 1),
    C("1d.h2", BGRA8, MTLTextureType1D, 64, 2, 1, 1, 1, 1),
    CB("3d.w2048", BGRA8, MTLTextureType3D, 2048, 1, 1, 1),
    CB("3d.w2049", BGRA8, MTLTextureType3D, 2049, 1, 1, 1),
    CB("3d.d2049", BGRA8, MTLTextureType3D, 1, 1, 2049, 1),
    CB("2darray.a2048", BGRA8, MTLTextureType2DArray, 1, 1, 1, 2048),
    CB("2darray.a2049", BGRA8, MTLTextureType2DArray, 1, 1, 1, 2049),
    C("cube.64x32", BGRA8, MTLTextureTypeCube, 64, 32, 1, 1, 1, 1),
    C("cube.a2", BGRA8, MTLTextureTypeCube, 64, 64, 1, 2, 1, 1),
    CB("cubearray.a342", BGRA8, MTLTextureTypeCubeArray, 1, 1, 1, 342),
    CB("cubearray.a341", BGRA8, MTLTextureTypeCubeArray, 1, 1, 1, 341),
    // Depth, array length, levels and samples the type takes or not.
    C("2d.d2", BGRA8, T2D, 64, 64, 2, 1, 1, 1),
    C("2d.a2", BGRA8, T2D, 64, 64, 1, 2, 1, 1),
    C("2d.s4", BGRA8, T2D, 64, 64, 1, 1, 1, 4),
    C("2d.m7", BGRA8, T2D, 64, 64, 1, 1, 7, 1),
    C("2d.m8", BGRA8, T2D, 64, 64, 1, 1, 8, 1),
    C("2d.m2.w1", BGRA8, T2D, 1, 1, 1, 1, 2, 1),
    C("3d.m8", BGRA8, MTLTextureType3D, 64, 64, 64, 1, 8, 1),
    C("1d.m2", BGRA8, MTLTextureType1D, 64, 1, 1, 1, 2, 1),
    C("2darray.d2", BGRA8, MTLTextureType2DArray, 64, 64, 2, 2, 1, 1),
    C("cube.d2", BGRA8, MTLTextureTypeCube, 64, 64, 2, 1, 1, 1),
    CU("ms.s1", BGRA8, MS, 1, MTLTextureUsageRenderTarget, kDefault),
    CU("ms.s2", BGRA8, MS, 2, MTLTextureUsageRenderTarget, kDefault),
    CU("ms.s3", BGRA8, MS, 3, MTLTextureUsageRenderTarget, kDefault),
    CU("ms.s4", BGRA8, MS, 4, MTLTextureUsageRenderTarget, kDefault),
    CU("ms.s8", BGRA8, MS, 8, MTLTextureUsageRenderTarget, kDefault),
    CU("ms.s16", BGRA8, MS, 16, MTLTextureUsageRenderTarget, kDefault),
    {"ms.m2", BGRA8, MS, 64, 64, 1, 1, 2, 4, MTLTextureUsageRenderTarget, kDefault, 0},
    {"ms.d2", BGRA8, MS, 64, 64, 2, 1, 1, 4, MTLTextureUsageRenderTarget, kDefault, 0},
    {"ms.a2", BGRA8, MS, 64, 64, 1, 2, 1, 4, MTLTextureUsageRenderTarget, kDefault, 0},
    CU("ms.shared", BGRA8, MS, 4, MTLTextureUsageRenderTarget, MTLStorageModeShared),
    CU("ms.managed", BGRA8, MS, 4, MTLTextureUsageRenderTarget, MTLStorageModeManaged),
    {"msarray.a2", BGRA8, MTLTextureType2DMultisampleArray, 64, 64, 1, 2, 1, 4,
     MTLTextureUsageRenderTarget, kDefault, 0},
    CU("ms.bgr5a1", MTLPixelFormatBGR5A1Unorm, MS, 4, MTLTextureUsageRenderTarget, kDefault),
    CU("ms.rgb9e5", MTLPixelFormatRGB9E5Float, MS, 4, MTLTextureUsageShaderRead, kDefault),
    CU("ms.rgba32f", MTLPixelFormatRGBA32Float, MS, 4, MTLTextureUsageRenderTarget, kDefault),
    CU("ms.r32u", MTLPixelFormatR32Uint, MS, 4, MTLTextureUsageRenderTarget, kDefault),
    CU("ms.depth32f", MTLPixelFormatDepth32Float, MS, 4, MTLTextureUsageRenderTarget, kDefault),
    // Storage.
    CU("shared", BGRA8, T2D, 1, kDefault, MTLStorageModeShared),
    CU("managed", BGRA8, T2D, 1, kDefault, MTLStorageModeManaged),
    CU("memoryless", BGRA8, T2D, 1, MTLTextureUsageRenderTarget, 3),
    CU("storage5", BGRA8, T2D, 1, kDefault, 5),
    CU("depth32f.shared", MTLPixelFormatDepth32Float, T2D, 1, MTLTextureUsageRenderTarget,
       MTLStorageModeShared),
    // Usage.
    CU("usage0", BGRA8, T2D, 1, 0, kDefault),
    CU("usage.write", BGRA8, T2D, 1, MTLTextureUsageShaderWrite, kDefault),
    CU("usage.bit7", BGRA8, T2D, 1, 0x80 | MTLTextureUsageShaderRead, kDefault),
    CU("srgb.write", MTLPixelFormatBGRA8Unorm_sRGB, T2D, 1, MTLTextureUsageShaderWrite, kDefault),
    CU("bc1.rt", MTLPixelFormatBC1_RGBA, T2D, 1, MTLTextureUsageRenderTarget, kDefault),
    CU("bc1.write", MTLPixelFormatBC1_RGBA, T2D, 1, MTLTextureUsageShaderWrite, kDefault),
    CU("depth32f.write", MTLPixelFormatDepth32Float, T2D, 1, MTLTextureUsageShaderWrite,
       kDefault),
    CU("rgb9e5.rt", MTLPixelFormatRGB9E5Float, T2D, 1, MTLTextureUsageRenderTarget, kDefault),
    CU("gbgr422.rt", MTLPixelFormatGBGR422, T2D, 1, MTLTextureUsageRenderTarget, kDefault),
    // Formats.
    F("fmt.invalid", MTLPixelFormatInvalid),
    F("fmt.b5g6r5", MTLPixelFormatB5G6R5Unorm),
    F("fmt.a1bgr5", MTLPixelFormatA1BGR5Unorm),
    F("fmt.abgr4", MTLPixelFormatABGR4Unorm),
    F("fmt.bgr5a1", MTLPixelFormatBGR5A1Unorm),
    F("fmt.r8srgb", MTLPixelFormatR8Unorm_sRGB),
    F("fmt.rg8srgb", MTLPixelFormatRG8Unorm_sRGB),
    F("fmt.bgra10xr", MTLPixelFormatBGRA10_XR),
    F("fmt.bgr10xr", MTLPixelFormatBGR10_XR),
    F("fmt.astc4x4", MTLPixelFormatASTC_4x4_LDR),
    F("fmt.etc2", MTLPixelFormatETC2_RGB8),
    F("fmt.eac", MTLPixelFormatEAC_R11Unorm),
    F("fmt.pvrtc", MTLPixelFormatPVRTC_RGBA_4BPP),
    F("fmt.ycbcr500", (MTLPixelFormat)500),
    F("fmt.d24s8", MTLPixelFormatDepth24Unorm_Stencil8),
    F("fmt.x24s8", MTLPixelFormatX24_Stencil8),
    F("fmt.unspecialized", (MTLPixelFormat)263),
    F("fmt.9999", (MTLPixelFormat)9999),
    F("fmt.gbgr422", MTLPixelFormatGBGR422),
    C("gbgr422.63", MTLPixelFormatGBGR422, T2D, 63, 64, 1, 1, 1, 1),
    C("gbgr422.h63", MTLPixelFormatGBGR422, T2D, 64, 63, 1, 1, 1, 1),
    C("gbgr422.2darray", MTLPixelFormatGBGR422, MTLTextureType2DArray, 64, 64, 1, 2, 1, 1),
    C("gbgr422.m2", MTLPixelFormatGBGR422, T2D, 64, 64, 1, 1, 2, 1),
    // Compressed and depth formats by type and size.
    C("bc1.62x64", MTLPixelFormatBC1_RGBA, T2D, 62, 64, 1, 1, 1, 1),
    C("bc1.2x2", MTLPixelFormatBC1_RGBA, T2D, 2, 2, 1, 1, 1, 1),
    C("bc1.1d", MTLPixelFormatBC1_RGBA, MTLTextureType1D, 64, 1, 1, 1, 1, 1),
    C("bc1.3d", MTLPixelFormatBC1_RGBA, MTLTextureType3D, 64, 64, 4, 1, 1, 1),
    C("bc1.cube", MTLPixelFormatBC1_RGBA, MTLTextureTypeCube, 64, 64, 1, 1, 1, 1),
    C("bc1.m7", MTLPixelFormatBC1_RGBA, T2D, 64, 64, 1, 1, 7, 1),
    C("depth32f.3d", MTLPixelFormatDepth32Float, MTLTextureType3D, 64, 64, 4, 1, 1, 1),
    C("depth32f.1d", MTLPixelFormatDepth32Float, MTLTextureType1D, 64, 1, 1, 1, 1, 1),
    C("depth32f.cube", MTLPixelFormatDepth32Float, MTLTextureTypeCube, 64, 64, 1, 1, 1, 1),
    C("depth32f.2darray", MTLPixelFormatDepth32Float, MTLTextureType2DArray, 64, 64, 1, 2, 1, 1),
    C("depth32f.m7", MTLPixelFormatDepth32Float, T2D, 64, 64, 1, 1, 7, 1),
    C("stencil8.3d", MTLPixelFormatStencil8, MTLTextureType3D, 64, 64, 4, 1, 1, 1),
    // A texture buffer at a creator that takes no buffer.
    C("texbuf", BGRA8, MTLTextureTypeTextureBuffer, 64, 1, 1, 1, 1, 1),
    C("texbuf.h4", BGRA8, MTLTextureTypeTextureBuffer, 64, 4, 1, 1, 1, 1),
    C("1darray.h2", BGRA8, MTLTextureType1DArray, 64, 2, 1, 2, 1, 1),
    CB("w16384.h16384", BGRA8, T2D, 16384, 16384, 1, 1),
    C("type10", BGRA8, (MTLTextureType)10, 64, 64, 1, 1, 1, 1),
    // Two rules at once, to see their order in one assertion.
    {"w0.ms.shared.s1", BGRA8, MS, 0, 64, 1, 1, 1, 1, MTLTextureUsageRenderTarget,
     MTLStorageModeShared, 0},
    {"bc1.ms.shared", MTLPixelFormatBC1_RGBA, MS, 64, 64, 1, 1, 1, 4, kDefault,
     MTLStorageModeShared, 0},
    {"b5g6r5.w0.d2", MTLPixelFormatB5G6R5Unorm, T2D, 0, 64, 2, 1, 1, 1, kDefault, kDefault, 0},
    {"w16385.d2", BGRA8, T2D, 16385, 1, 2, 1, 1, 1, kDefault, kDefault, 1},
    {"depth32f.shared.ms", MTLPixelFormatDepth32Float, MS, 64, 64, 1, 1, 1, 4,
     MTLTextureUsageRenderTarget, MTLStorageModeShared, 0},
    // RGBA32Float textures within every rule, of 2 GiB to 8 TiB, around and past the device's
    // memory, at which the device's creator answers nil for the bytes alone.
    CB("size.2g", MTLPixelFormatRGBA32Float, T2D, 16384, 8192, 1, 1),
    CB("size.2g+row", MTLPixelFormatRGBA32Float, T2D, 16384, 8193, 1, 1),
    CB("size.3g", MTLPixelFormatRGBA32Float, T2D, 16384, 12288, 1, 1),
    CB("size.4g", MTLPixelFormatRGBA32Float, T2D, 16384, 16384, 1, 1),
    CB("size.8g", MTLPixelFormatRGBA32Float, MTLTextureType2DArray, 16384, 16384, 1, 2),
    CB("size.16g", MTLPixelFormatRGBA32Float, MTLTextureType2DArray, 16384, 16384, 1, 4),
    CB("size.20g", MTLPixelFormatRGBA32Float, MTLTextureType2DArray, 16384, 16384, 1, 5),
    CB("size.24g", MTLPixelFormatRGBA32Float, MTLTextureType2DArray, 16384, 16384, 1, 6),
    CB("size.28g", MTLPixelFormatRGBA32Float, MTLTextureType2DArray, 16384, 16384, 1, 7),
    CB("size.32g", MTLPixelFormatRGBA32Float, MTLTextureType2DArray, 16384, 16384, 1, 8),
    CB("size.64g", MTLPixelFormatRGBA32Float, MTLTextureType2DArray, 16384, 16384, 1, 16),
    CB("size.3d.128g", MTLPixelFormatRGBA32Float, MTLTextureType3D, 2048, 2048, 2048, 1),
    CB("size.2darray.8t", MTLPixelFormatRGBA32Float, MTLTextureType2DArray, 16384, 16384, 1,
       2048),
};
enum { kCaseCount = sizeof kCases / sizeof kCases[0] };

enum Creator { kDevice, kHeap, kBuffer };

static int creatorOf(const char *s)
{
	if (!strcmp(s, "device"))
		return kDevice;
	if (!strcmp(s, "heap"))
		return kHeap;
	if (!strcmp(s, "buffer"))
		return kBuffer;
	return -1;
}

static int runCase(id<MTLDevice> d, int creator, const char *creatorName, int n)
{
	const struct Case *c = &kCases[n];
	if (creator == kHeap && c->big) {
		printf("case %d %s | %s => not run (more than the heap holds)\n", n, c->name,
		       creatorName);
		return 0;
	}
	MTLTextureDescriptor *td = [MTLTextureDescriptor new];
	td.textureType = c->type;
	td.pixelFormat = c->format;
	td.width = (NSUInteger)c->width;
	td.height = (NSUInteger)c->height;
	td.depth = (NSUInteger)c->depth;
	td.arrayLength = (NSUInteger)c->array;
	td.mipmapLevelCount = (NSUInteger)c->mips;
	td.sampleCount = (NSUInteger)c->samples;
	td.usage = c->usage == kDefault ? MTLTextureUsageShaderRead : (MTLTextureUsage)c->usage;
	td.storageMode = c->storage != kDefault  ? (MTLStorageMode)c->storage
	                 : creator == kBuffer ? MTLStorageModeShared
	                                      : MTLStorageModePrivate;
	printf("case %d %s | %s | asked: fmt %lu type %lu %lux%lux%lu a %lu m %lu s %lu usage 0x%lx "
	       "storage %lu\n",
	       n, c->name, creatorName, (unsigned long)td.pixelFormat, (unsigned long)td.textureType,
	       (unsigned long)td.width, (unsigned long)td.height, (unsigned long)td.depth,
	       (unsigned long)td.arrayLength, (unsigned long)td.mipmapLevelCount,
	       (unsigned long)td.sampleCount, (unsigned long)td.usage,
	       (unsigned long)td.storageMode);
	// The size cases' bytes as Metal places them, which makes nothing.
	if (creator == kDevice && !strncmp(c->name, "size.", 5)) {
		const MTLSizeAndAlign sa = [d heapTextureSizeAndAlignWithDescriptor:td];
		printf("case %d %s heapTextureSizeAndAlign: size %llu align %llu\n", n, c->name,
		       (unsigned long long)sa.size, (unsigned long long)sa.align);
	}
	fflush(stdout);
	id<MTLTexture> t = nil;
	id<MTLHeap> heap = nil;
	id<MTLBuffer> buffer = nil;
	switch (creator) {
	case kDevice:
		t = [d newTextureWithDescriptor:td];
		break;
	case kHeap: {
		MTLHeapDescriptor *hd = [MTLHeapDescriptor new];
		hd.size = 256u << 20;
		hd.storageMode = MTLStorageModePrivate;
		heap = [d newHeapWithDescriptor:hd];
		if (heap == nil) {
			printf("case %d %s => no heap\n", n, c->name);
			return 1;
		}
		t = [heap newTextureWithDescriptor:td];
		break;
	}
	case kBuffer: {
		buffer = [d newBufferWithLength:64u << 20 options:MTLResourceStorageModeShared];
		if (buffer == nil) {
			printf("case %d %s => no buffer\n", n, c->name);
			return 1;
		}
		const NSUInteger row = ((NSUInteger)c->width * 4 + 255) / 256 * 256;
		t = [buffer newTextureWithDescriptor:td offset:0 bytesPerRow:row == 0 ? 256 : row];
		break;
	}
	}
	if (t == nil) {
		printf("case %d %s => nil\n", n, c->name);
	} else {
		task_vm_info_data_t vm;
		mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
		const unsigned long long footprint =
		    task_info(mach_task_self(), TASK_VM_INFO, (task_info_t)&vm, &count) == KERN_SUCCESS
		        ? vm.phys_footprint
		        : 0;
		printf("case %d %s allocatedSize %llu, device's allocated size then %llu, footprint "
		       "%llu\n",
		       n, c->name, (unsigned long long)t.allocatedSize,
		       (unsigned long long)d.currentAllocatedSize, footprint);
		printf("case %d %s => made: fmt %lu type %lu %lux%lux%lu a %lu m %lu s %lu usage 0x%lx "
		       "storage %lu\n",
		       n, c->name, (unsigned long)t.pixelFormat, (unsigned long)t.textureType,
		       (unsigned long)t.width, (unsigned long)t.height, (unsigned long)t.depth,
		       (unsigned long)t.arrayLength, (unsigned long)t.mipmapLevelCount,
		       (unsigned long)t.sampleCount, (unsigned long)t.usage,
		       (unsigned long)t.storageMode);
	}
	fflush(stdout);
	t = nil;
	heap = nil;
	buffer = nil;
	return 0;
}

int main(int argc, char **argv)
{
	setvbuf(stdout, NULL, _IOLBF, 0);
	if (argc < 3 || creatorOf(argv[2]) < 0) {
		fprintf(stderr, "usage: %s DEVICE device|heap|buffer [--case N | --list]\n", argv[0]);
		return 2;
	}
	const int creator = creatorOf(argv[2]);
	if (argc > 3 && !strcmp(argv[3], "--list")) {
		for (int n = 0; n < kCaseCount; n++)
			printf("%d %s\n", n, kCases[n].name);
		return 0;
	}
	if (argc > 4 && !strcmp(argv[3], "--case")) {
		// Metal's words where it ends the process go to stderr: put them in the case's lines.
		dup2(STDOUT_FILENO, STDERR_FILENO);
		int n = atoi(argv[4]);
		if (n < 0 || n >= kCaseCount)
			return 2;
		return runCase(oracle_device(argv[1]), creator, argv[2], n);
	}
	id<MTLDevice> device = oracle_device(argv[1]);
	oracle_header("oracle_texdesc", device);
	printf("# recommendedMaxWorkingSetSize %llu, maxBufferLength %llu, currentAllocatedSize %llu\n",
	       (unsigned long long)device.recommendedMaxWorkingSetSize,
	       (unsigned long long)device.maxBufferLength,
	       (unsigned long long)device.currentAllocatedSize);
	fflush(stdout);
	for (int n = 0; n < kCaseCount; n++) {
		char index[16];
		snprintf(index, sizeof index, "%d", n);
		char *args[] = {argv[0], argv[1], argv[2], "--case", index, NULL};
		pid_t pid;
		if (posix_spawn(&pid, argv[0], NULL, NULL, args, environ))
			return 1;
		int st;
		waitpid(pid, &st, 0);
		if (WIFSIGNALED(st))
			printf("=> case %d died on signal %d\n", n, WTERMSIG(st));
		else if (WEXITSTATUS(st))
			printf("=> case %d exited %d\n", n, WEXITSTATUS(st));
		fflush(stdout);
	}
	return 0;
}
