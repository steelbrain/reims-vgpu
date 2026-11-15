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
//
// Texture buffers (MTLTextureTypeTextureBuffer, 11-14), each step a child process of its own:
//   oracle_texdesc DEVICE CREATOR --texbuf [fixed|max]   fixed widths, then bisection (or one)
//   oracle_texdesc DEVICE CREATOR --tb F W [LEN]    one texture buffer (what the children run)
//   oracle_texdesc DEVICE CREATOR --tbread F W      one, filled and read back by kernels
// The descriptor is MTLTextureDescriptor's textureBufferDescriptorWithPixelFormat:width:
// resourceOptions:usage: of format F, width W, usage ShaderRead (and ShaderWrite for --tbread),
// Shared over a buffer and Private otherwise, as MoltenVK makes a texel buffer view's texture.
// CREATOR `device` is the device's newTextureWithDescriptor:; `heap` a Private heap's, the heap
// of the texture's bytes rounded up to 64 KiB plus 64 KiB, at least 1 MiB and at most
// maxBufferLength; `buffer` a Shared buffer's newTextureWithDescriptor:offset:bytesPerRow: at
// offset 0, bytesPerRow the texture's bytes rounded up to 256, over a buffer of LEN bytes (default
// those bytes, at most maxBufferLength rounded down to 4096). --texbuf runs the formats R8Sint (1
// byte), RG8Unorm (2), RGBA8Unorm (4), RG32Float (8) and RGBA32Float (16) at the widths kTbWidths
// lists (over a buffer, also over one of 4096 bytes), then, for each format, the largest width
// made and the largest width that does not end the process, by bisection from 1 to 2^36. Lines:
//   tb fmt F (B bytes) CREATOR width W | asked: ... [| buffer L bytes, bpr R | heap H bytes]
//   tb fmt F (B bytes) CREATOR width W => made: type T width W ... | nil | no heap | no buffer
// (a child's own), and the parent's
//   tbstep fmt F CREATOR width W [buffer L]: made | nil | no heap | no buffer | died: WORDS
//   tbmax fmt F (B bytes) CREATOR made|alive W (W*B bytes); at W+1: OUTCOME
// where WORDS are what Metal wrote before the child ended, its lines joined by " / ".
// --tbread takes a signed integer format, R8Sint, R16Sint, R32Sint, RG32Sint or RGBA32Sint: over a
// buffer the CPU writes element i's channels as (i * 37 + 11) % 251 - 125, elsewhere a kernel
// writes them; then a kernel reads elements 0, 1, W / 2, W - 2 and W - 1 (texture_buffer read), and
// the line says what each held against what was written:
//   tbread fmt F CREATOR width W: element E read V want V' ... => ok | WRONG
// Build: see README.md.
#import "oracle_common.h"
#include <mach/mach.h>
#include <objc/message.h>
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

// --- texture buffers ------------------------------------------------------------------------

struct TbFormat {
	MTLPixelFormat format;
	unsigned bytes;
};
static const struct TbFormat kTbFormats[] = {
    {MTLPixelFormatR8Sint, 1},     {MTLPixelFormatRG8Unorm, 2},     {MTLPixelFormatRGBA8Unorm, 4},
    {MTLPixelFormatRG32Float, 8},  {MTLPixelFormatRGBA32Float, 16},
};
static const struct TbFormat kTbReadFormats[] = {
    {MTLPixelFormatR8Sint, 1},  {MTLPixelFormatR16Sint, 2},     {MTLPixelFormatR32Sint, 4},
    {MTLPixelFormatRG32Sint, 8}, {MTLPixelFormatRGBA32Sint, 16},
};
static const unsigned long long kTbWidths[] = {
    16384,           16385,           46626,           1ull << 20,      1ull << 24,
    1ull << 26,      (1ull << 26) + 1, 1ull << 27,     (1ull << 27) + 1, 1ull << 28,
    (1ull << 28) + 1, 1ull << 29,     1ull << 30,      1ull << 31,      1ull << 32,
    (1ull << 32) + 46626, 1ull << 33,
};
enum { kTbWidthCount = sizeof kTbWidths / sizeof kTbWidths[0] };
static const unsigned long long kTbBisectTop = 1ull << 36;

static unsigned tbBytes(MTLPixelFormat f)
{
	for (size_t i = 0; i < sizeof kTbFormats / sizeof kTbFormats[0]; i++)
		if (kTbFormats[i].format == f)
			return kTbFormats[i].bytes;
	for (size_t i = 0; i < sizeof kTbReadFormats / sizeof kTbReadFormats[0]; i++)
		if (kTbReadFormats[i].format == f)
			return kTbReadFormats[i].bytes;
	return 0;
}

static unsigned long long tbRound(unsigned long long n, unsigned long long to)
{
	return (n + to - 1) / to * to;
}

// The texture buffer of FORMAT and WIDTH at CREATOR (the buffer LEN bytes, 0 for the default),
// or nil with the line saying why. `write` adds ShaderWrite. `*bufferOut` keeps the buffer.
static id<MTLTexture> tbMake(id<MTLDevice> d, int creator, const char *creatorName,
                             MTLPixelFormat format, unsigned long long width,
                             unsigned long long len, bool write, id<MTLBuffer> *bufferOut)
{
	const unsigned bytes = tbBytes(format);
	const unsigned long long texBytes = width * bytes;
	MTLTextureUsage usage = MTLTextureUsageShaderRead | (write ? MTLTextureUsageShaderWrite : 0);
	MTLResourceOptions options =
	    creator == kBuffer ? MTLResourceStorageModeShared : MTLResourceStorageModePrivate;
	MTLTextureDescriptor *td =
	    [MTLTextureDescriptor textureBufferDescriptorWithPixelFormat:format
	                                                           width:(NSUInteger)width
	                                                 resourceOptions:options
	                                                           usage:usage];
	printf("tb fmt %lu (%u bytes) %s width %llu | asked: type %lu %lux%lux%lu a %lu m %lu s %lu "
	       "usage 0x%lx storage %lu",
	       (unsigned long)format, bytes, creatorName, width, (unsigned long)td.textureType,
	       (unsigned long)td.width, (unsigned long)td.height, (unsigned long)td.depth,
	       (unsigned long)td.arrayLength, (unsigned long)td.mipmapLevelCount,
	       (unsigned long)td.sampleCount, (unsigned long)td.usage, (unsigned long)td.storageMode);
	id<MTLTexture> t = nil;
	const unsigned long long maxLength = d.maxBufferLength;
	switch (creator) {
	case kDevice:
		printf("\n");
		fflush(stdout);
		t = [d newTextureWithDescriptor:td];
		break;
	case kHeap: {
		unsigned long long size = tbRound(texBytes, 65536) + 65536;
		if (size < (1u << 20))
			size = 1u << 20;
		if (size > maxLength)
			size = maxLength / 65536 * 65536;
		printf(" | heap %llu bytes\n", size);
		fflush(stdout);
		MTLHeapDescriptor *hd = [MTLHeapDescriptor new];
		hd.size = (NSUInteger)size;
		hd.storageMode = MTLStorageModePrivate;
		id<MTLHeap> heap = [d newHeapWithDescriptor:hd];
		if (heap == nil) {
			printf("tb fmt %lu %s width %llu => no heap\n", (unsigned long)format, creatorName,
			       width);
			return nil;
		}
		t = [heap newTextureWithDescriptor:td];
		break;
	}
	case kBuffer: {
		const unsigned long long bpr = tbRound(texBytes, 256);
		if (len == 0)
			len = bpr < maxLength ? bpr : maxLength / 4096 * 4096;
		printf(" | buffer %llu bytes, bpr %llu\n", len, bpr);
		fflush(stdout);
		id<MTLBuffer> buffer = [d newBufferWithLength:(NSUInteger)len
		                                      options:MTLResourceStorageModeShared];
		if (buffer == nil) {
			printf("tb fmt %lu %s width %llu => no buffer\n", (unsigned long)format,
			       creatorName, width);
			return nil;
		}
		*bufferOut = buffer;
		t = [buffer newTextureWithDescriptor:td offset:0 bytesPerRow:(NSUInteger)bpr];
		break;
	}
	}
	if (t == nil) {
		printf("tb fmt %lu %s width %llu => nil\n", (unsigned long)format, creatorName, width);
		return nil;
	}
	task_vm_info_data_t vm;
	mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
	const unsigned long long footprint =
	    task_info(mach_task_self(), TASK_VM_INFO, (task_info_t)&vm, &count) == KERN_SUCCESS
	        ? vm.phys_footprint
	        : 0;
	printf("tb fmt %lu %s width %llu => made: type %lu width %lu height %lu storage %lu usage "
	       "0x%lx allocatedSize %llu footprint %llu\n",
	       (unsigned long)format, creatorName, width, (unsigned long)t.textureType,
	       (unsigned long)t.width, (unsigned long)t.height, (unsigned long)t.storageMode,
	       (unsigned long)t.usage, (unsigned long long)t.allocatedSize, footprint);
	fflush(stdout);
	return t;
}

static int tbOne(id<MTLDevice> d, int creator, const char *creatorName, MTLPixelFormat format,
                 unsigned long long width, unsigned long long len)
{
	if (tbBytes(format) == 0) {
		printf("tb fmt %lu: not a format of this probe\n", (unsigned long)format);
		return 2;
	}
	id<MTLBuffer> buffer = nil;
	id<MTLTexture> t = tbMake(d, creator, creatorName, format, width, len, false, &buffer);
	(void)t;
	return 0;
}

static int tbValue(unsigned long long i)
{
	return (int)((i * 37 + 11) % 251) - 125;
}

static NSString *const kTbKernels =
    @"#include <metal_stdlib>\n"
     "using namespace metal;\n"
     "kernel void tb_fill(texture_buffer<int, access::write> t [[texture(0)]],\n"
     "                    uint i [[thread_position_in_grid]]) {\n"
     "  if (i >= t.get_width()) return;\n"
     "  int v = int((ulong(i) * 37 + 11) % 251) - 125;\n"
     "  t.write(int4(v), i);\n"
     "}\n"
     "kernel void tb_read(texture_buffer<int, access::read> t [[texture(0)]],\n"
     "                    device const uint *at [[buffer(0)]], device int *out [[buffer(1)]],\n"
     "                    device uint *width [[buffer(2)]], uint i [[thread_position_in_grid]]) {\n"
     "  out[i] = t.read(at[i]).r;\n"
     "  if (i == 0) width[0] = t.get_width();\n"
     "}\n";

// Makes the texture buffer, fills it (the CPU over a buffer, a kernel elsewhere), and reads five
// elements back by a kernel.
static int tbRead(id<MTLDevice> d, int creator, const char *creatorName, MTLPixelFormat format,
                  unsigned long long width)
{
	unsigned bytes = 0;
	for (size_t i = 0; i < sizeof kTbReadFormats / sizeof kTbReadFormats[0]; i++)
		if (kTbReadFormats[i].format == format)
			bytes = kTbReadFormats[i].bytes;
	if (bytes == 0 || width < 2) {
		printf("tbread fmt %lu: not a signed integer format of this probe, or under 2 wide\n",
		       (unsigned long)format);
		return 2;
	}
	NSError *err = nil;
	id<MTLLibrary> lib = [d newLibraryWithSource:kTbKernels options:nil error:&err];
	if (lib == nil) {
		printf("tbread: library: %s\n", err ? err.description.UTF8String : "nil");
		return 1;
	}
	id<MTLComputePipelineState> fill =
	    [d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"tb_fill"] error:&err];
	id<MTLComputePipelineState> read =
	    [d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"tb_read"] error:&err];
	if (fill == nil || read == nil) {
		printf("tbread: pipeline: %s\n", err ? err.description.UTF8String : "nil");
		return 1;
	}
	id<MTLBuffer> buffer = nil;
	id<MTLTexture> t = tbMake(d, creator, creatorName, format, width, 0, true, &buffer);
	if (t == nil)
		return 0;
	const unsigned channels = format == MTLPixelFormatRG32Sint ? 2 : format == MTLPixelFormatRGBA32Sint ? 4 : 1;
	const unsigned channelBytes = bytes / channels;
	id<MTLCommandQueue> q = [d newCommandQueue];
	if (creator == kBuffer) {
		uint8_t *p = (uint8_t *)buffer.contents;
		const unsigned long long fit = buffer.length / bytes < width ? buffer.length / bytes : width;
		for (unsigned long long i = 0; i < fit; i++) {
			const int v = tbValue(i);
			for (unsigned c = 0; c < channels; c++) {
				uint8_t *e = p + i * bytes + c * channelBytes;
				if (channelBytes == 1)
					*(int8_t *)e = (int8_t)v;
				else if (channelBytes == 2)
					*(int16_t *)e = (int16_t)v;
				else
					*(int32_t *)e = v;
			}
		}
	} else {
		id<MTLCommandBuffer> cb = [q commandBuffer];
		id<MTLComputeCommandEncoder> ce = [cb computeCommandEncoder];
		[ce setComputePipelineState:fill];
		[ce setTexture:t atIndex:0];
		const NSUInteger tg = fill.maxTotalThreadsPerThreadgroup < 256 ? fill.maxTotalThreadsPerThreadgroup : 256;
		[ce dispatchThreadgroups:MTLSizeMake((NSUInteger)((width + tg - 1) / tg), 1, 1)
		   threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
		[ce endEncoding];
		oracle_commit_wait(cb, NULL, NULL);
	}
	const uint32_t at[5] = {0, 1, (uint32_t)(width / 2), (uint32_t)(width - 2), (uint32_t)(width - 1)};
	id<MTLBuffer> atBuf = [d newBufferWithBytes:at length:sizeof at options:MTLResourceStorageModeShared];
	id<MTLBuffer> out = [d newBufferWithLength:5 * sizeof(int32_t) options:MTLResourceStorageModeShared];
	id<MTLBuffer> w = [d newBufferWithLength:sizeof(uint32_t) options:MTLResourceStorageModeShared];
	memset(out.contents, 0x7f, 5 * sizeof(int32_t));
	id<MTLCommandBuffer> cb = [q commandBuffer];
	id<MTLComputeCommandEncoder> ce = [cb computeCommandEncoder];
	[ce setComputePipelineState:read];
	[ce setTexture:t atIndex:0];
	[ce setBuffer:atBuf offset:0 atIndex:0];
	[ce setBuffer:out offset:0 atIndex:1];
	[ce setBuffer:w offset:0 atIndex:2];
	[ce dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(5, 1, 1)];
	[ce endEncoding];
	oracle_commit_wait(cb, NULL, NULL);
	const int32_t *o = (const int32_t *)out.contents;
	bool ok = true;
	printf("tbread fmt %lu %s width %llu: get_width %u;", (unsigned long)format, creatorName,
	       width, *(const uint32_t *)w.contents);
	for (int i = 0; i < 5; i++) {
		const int want = tbValue(at[i]);
		printf(" element %u read %d want %d;", at[i], o[i], want);
		if (o[i] != want)
			ok = false;
	}
	printf(" => %s\n", ok ? "ok" : "WRONG");
	return 0;
}

enum TbOutcome { kTbMade, kTbNil, kTbNoResource, kTbDied, kTbOther };

// Runs one --tb step in a child, its output (stdout and stderr) into `text`; returns its class,
// and in `words` what the child wrote that is not a `tb ` line of its own, joined by " / ".
static enum TbOutcome tbStep(const char *argv0, const char *device, const char *creatorName,
                             MTLPixelFormat format, unsigned long long width,
                             unsigned long long len, char *words, size_t wordsSize, bool echo)
{
	char f[24], w[32], l[32];
	snprintf(f, sizeof f, "%lu", (unsigned long)format);
	snprintf(w, sizeof w, "%llu", width);
	snprintf(l, sizeof l, "%llu", len);
	char *args[] = {(char *)argv0, (char *)device, (char *)creatorName, "--tb", f, w, l, NULL};
	int fds[2];
	if (pipe(fds) != 0)
		return kTbOther;
	posix_spawn_file_actions_t fa;
	posix_spawn_file_actions_init(&fa);
	posix_spawn_file_actions_adddup2(&fa, fds[1], STDOUT_FILENO);
	posix_spawn_file_actions_adddup2(&fa, fds[1], STDERR_FILENO);
	posix_spawn_file_actions_addclose(&fa, fds[0]);
	pid_t pid;
	const int sp = posix_spawn(&pid, argv0, &fa, NULL, args, environ);
	posix_spawn_file_actions_destroy(&fa);
	close(fds[1]);
	if (sp != 0) {
		close(fds[0]);
		return kTbOther;
	}
	NSMutableData *all = [NSMutableData data];
	char buf[4096];
	ssize_t n;
	while ((n = read(fds[0], buf, sizeof buf)) > 0)
		[all appendBytes:buf length:(NSUInteger)n];
	close(fds[0]);
	int st = 0;
	waitpid(pid, &st, 0);
	NSString *text = [[NSString alloc] initWithData:all encoding:NSUTF8StringEncoding] ?: @"";
	if (echo) {
		for (NSString *line in [text componentsSeparatedByString:@"\n"])
			if (line.length > 0)
				printf("  %s\n", line.UTF8String);
	}
	NSMutableArray<NSString *> *other = [NSMutableArray array];
	for (NSString *line in [text componentsSeparatedByString:@"\n"])
		if (line.length > 0 && ![line hasPrefix:@"tb "])
			[other addObject:line];
	snprintf(words, wordsSize, "%s", [other componentsJoinedByString:@" / "].UTF8String);
	if (WIFSIGNALED(st) || WEXITSTATUS(st) != 0) {
		const size_t at = strlen(words);
		snprintf(words + at, wordsSize - at, "%s(%s %d)", at ? " " : "",
		         WIFSIGNALED(st) ? "signal" : "exit",
		         WIFSIGNALED(st) ? WTERMSIG(st) : WEXITSTATUS(st));
		return kTbDied;
	}
	if ([text containsString:@"=> made:"])
		return kTbMade;
	if ([text containsString:@"=> nil"])
		return kTbNil;
	if ([text containsString:@"=> no heap"] || [text containsString:@"=> no buffer"])
		return kTbNoResource;
	return kTbOther;
}

static const char *tbName(enum TbOutcome o)
{
	switch (o) {
	case kTbMade: return "made";
	case kTbNil: return "nil";
	case kTbNoResource: return "no heap or buffer";
	case kTbDied: return "died";
	default: return "other";
	}
}

static void tbStepLine(MTLPixelFormat format, const char *creatorName, unsigned long long width,
                       enum TbOutcome o, const char *words)
{
	printf("tbstep fmt %lu %s width %llu: %s%s%.600s\n", (unsigned long)format, creatorName,
	       width, tbName(o), *words ? ": " : "", words);
}

// The largest width in [lo, hi) for which `want` (made, or not died) holds, given that it holds
// at lo and not at hi.
static unsigned long long tbBisect(const char *argv0, const char *device, const char *creatorName,
                                   MTLPixelFormat format, unsigned long long lo,
                                   unsigned long long hi, bool made)
{
	static char words[8192];
	while (hi - lo > 1) {
		const unsigned long long mid = lo + (hi - lo) / 2;
		const enum TbOutcome o =
		    tbStep(argv0, device, creatorName, format, mid, 0, words, sizeof words, false);
		tbStepLine(format, creatorName, mid, o, words);
		if (made ? o == kTbMade : o != kTbDied)
			lo = mid;
		else
			hi = mid;
	}
	return lo;
}

static int tbSweep(const char *argv0, const char *device, id<MTLDevice> d, int creator,
                   const char *creatorName, const char *part)
{
	const bool fixed = part == NULL || !strcmp(part, "fixed");
	const bool bisect = part == NULL || !strcmp(part, "max");
	oracle_header("oracle_texdesc --texbuf", d);
	printf("# maxBufferLength %llu, recommendedMaxWorkingSetSize %llu", (unsigned long long)d.maxBufferLength,
	       (unsigned long long)d.recommendedMaxWorkingSetSize);
	SEL mtbw = NSSelectorFromString(@"maxTextureBufferWidth");
	if ([d respondsToSelector:mtbw])
		printf(", maxTextureBufferWidth (SPI) %llu",
		       (unsigned long long)((NSUInteger(*)(id, SEL))objc_msgSend)(d, mtbw));
	else
		printf(", maxTextureBufferWidth (SPI) not answered");
	printf("\n# minimumTextureBufferAlignmentForPixelFormat:");
	for (size_t i = 0; i < sizeof kTbFormats / sizeof kTbFormats[0]; i++)
		printf(" %lu=%lu", (unsigned long)kTbFormats[i].format,
		       (unsigned long)[d minimumTextureBufferAlignmentForPixelFormat:kTbFormats[i].format]);
	printf("\n");
	fflush(stdout);
	static char words[8192];
	for (size_t fi = 0; fixed && fi < sizeof kTbFormats / sizeof kTbFormats[0]; fi++) {
		const MTLPixelFormat format = kTbFormats[fi].format;
		for (int wi = 0; wi < kTbWidthCount; wi++) {
			enum TbOutcome o = tbStep(argv0, device, creatorName, format, kTbWidths[wi], 0, words,
			                          sizeof words, true);
			tbStepLine(format, creatorName, kTbWidths[wi], o, words);
			if (creator == kBuffer) {
				o = tbStep(argv0, device, creatorName, format, kTbWidths[wi], 4096, words,
				           sizeof words, true);
				printf("tbstep fmt %lu %s width %llu buffer 4096: %s%s%.600s\n",
				       (unsigned long)format, creatorName, kTbWidths[wi], tbName(o),
				       *words ? ": " : "", words);
			}
			fflush(stdout);
		}
	}
	for (size_t fi = 0; bisect && fi < sizeof kTbFormats / sizeof kTbFormats[0]; fi++) {
		const MTLPixelFormat format = kTbFormats[fi].format;
		const unsigned bytes = kTbFormats[fi].bytes;
		for (int made = 1; made >= 0; made--) {
			const enum TbOutcome atTop =
			    tbStep(argv0, device, creatorName, format, kTbBisectTop, 0, words, sizeof words, false);
			if (made ? atTop == kTbMade : atTop != kTbDied) {
				printf("tbmax fmt %lu (%u bytes) %s %s: past %llu\n", (unsigned long)format,
				       bytes, creatorName, made ? "made" : "alive", kTbBisectTop);
				continue;
			}
			const unsigned long long w =
			    tbBisect(argv0, device, creatorName, format, 1, kTbBisectTop, made);
			const enum TbOutcome next =
			    tbStep(argv0, device, creatorName, format, w + 1, 0, words, sizeof words, false);
			printf("tbmax fmt %lu (%u bytes) %s %s %llu (%llu bytes); at %llu: %s%s%.1200s\n",
			       (unsigned long)format, bytes, creatorName, made ? "made" : "alive", w,
			       w * bytes, w + 1, tbName(next), *words ? ": " : "", words);
			fflush(stdout);
		}
	}
	return 0;
}

int main(int argc, char **argv)
{
	setvbuf(stdout, NULL, _IOLBF, 0);
	if (argc < 3 || creatorOf(argv[2]) < 0) {
		fprintf(stderr, "usage: %s DEVICE device|heap|buffer [--case N | --list | --texbuf [fixed|max] | --tb "
		                "FORMAT WIDTH [LEN] | --tbread FORMAT WIDTH]\n",
		        argv[0]);
		return 2;
	}
	const int creator = creatorOf(argv[2]);
	if (argc > 3 && !strcmp(argv[3], "--texbuf"))
		return tbSweep(argv[0], argv[1], oracle_device(argv[1]), creator, argv[2],
		               argc > 4 ? argv[4] : NULL);
	if (argc > 5 && !strcmp(argv[3], "--tb")) {
		alarm(120);
		return tbOne(oracle_device(argv[1]), creator, argv[2], (MTLPixelFormat)strtoul(argv[4], NULL, 10),
		             strtoull(argv[5], NULL, 10), argc > 6 ? strtoull(argv[6], NULL, 10) : 0);
	}
	if (argc > 5 && !strcmp(argv[3], "--tbread")) {
		oracle_header("oracle_texdesc --tbread", oracle_device(argv[1]));
		return tbRead(oracle_device(argv[1]), creator, argv[2], (MTLPixelFormat)strtoul(argv[4], NULL, 10),
		              strtoull(argv[5], NULL, 10));
	}
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
