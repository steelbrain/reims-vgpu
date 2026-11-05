// oracle_surftex: which textures newTextureWithDescriptor:iosurface:plane: makes over a plane of
// an IOSurface, which it answers nil for, and at which Metal ends the process, by the
// descriptor's type, size, depth, array length, levels, samples, pixel format, usage, storage,
// CPU cache and hazard tracking modes and swizzle, and by the surface's OSType, planes and the
// plane asked for. No Metal validation layer is asked for: what a guest's process meets.
//   oracle_surftex DEVICE              every case, each in a child process of its own
//   oracle_surftex DEVICE --case N     case N alone (what the children run)
//   oracle_surftex DEVICE --list       the cases, one a line, without running them
// The surfaces, each 64 x 64:
//   bgra   'BGRA', 4 bytes an element, made without planes (IOSurfaceCreate, as WindowServer's)
//   bgra1  'BGRA', 4 bytes an element, of one plane (kIOSurfacePlaneInfo)
//   rgha   'RGhA', 8 bytes an element, without planes
//   l008   'L008', 1 byte an element, without planes
//   420v   '420v', a CVPixelBuffer's: plane 0 of 1-byte luma, plane 1 of 2-byte chroma pairs
//          at 32 x 32
//   bgrapad 'BGRA', 4 bytes an element, without planes, 512 bytes a row (kIOSurfaceBytesPerRow)
// The descriptor of a case starts as a 2D texture of the case's format and size, one level, one
// sample, depth and array length 1, usage ShaderRead, the descriptor's own storage, CPU cache
// and hazard tracking modes and the identity swizzle, and changes what the case's name says.
// Each case prints:
//   case N NAME | SURFACE plane P | asked: fmt F type T WxHxD a A m M s S usage U storage S'
//   case N NAME => made: fmt F type T WxHxD a A m M s S usage U storage S plane P | nil
// and, where the case ends the process, Metal's words (its stderr is the child's stdout), then
//   => case N died on signal X | exited X
// Build: see README.md.
#import "oracle_common.h"
#import <CoreVideo/CoreVideo.h>
#import <IOSurface/IOSurface.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

extern char **environ;

enum Surface { kBGRA, kBGRA1, kRGhA, kL008, k420v, kBGRAPad };
static const char *const kSurfaceNames[] = {"bgra", "bgra1", "rgha", "l008", "420v", "bgrapad"};

enum { kSize = 64, kDefault = -1 };

struct Case {
	const char *name;
	enum Surface surface;
	int plane;
	MTLPixelFormat format;
	MTLTextureType type;
	int width, height, depth, array, mips, samples;
	long usage;   // kDefault: ShaderRead
	int storage;  // kDefault: the descriptor's own
	int cache;    // kDefault: the descriptor's own
	int hazard;   // kDefault: the descriptor's own
	int swizzled; // 1: red and blue swapped
};

#define T2D MTLTextureType2D
// A case over SURFACE's plane PLANE of FORMAT at W x H, all else the start's.
#define C(NAME, SURFACE, PLANE, FORMAT, W, H)                                                     \
	{NAME, SURFACE, PLANE, FORMAT, T2D, W, H, 1, 1, 1, 1, kDefault, kDefault, kDefault, kDefault, 0}
// The same of TYPE, DEPTH, ARRAY, MIPS and SAMPLES.
#define CT(NAME, SURFACE, FORMAT, TYPE, W, H, DEPTH, ARRAY, MIPS, SAMPLES)                         \
	{NAME, SURFACE, 0, FORMAT, TYPE, W, H, DEPTH, ARRAY, MIPS, SAMPLES, kDefault, kDefault,         \
	 kDefault, kDefault, 0}
// The same of USAGE, STORAGE, CACHE, HAZARD and SWIZZLED.
#define CU(NAME, SURFACE, PLANE, FORMAT, W, H, USAGE, STORAGE, CACHE, HAZARD, SWIZZLED)            \
	{NAME, SURFACE, PLANE, FORMAT, T2D, W, H, 1, 1, 1, 1, USAGE, STORAGE, CACHE, HAZARD, SWIZZLED}
// A 64 x 64 4x multisample render target in Private storage.
#define CMS(NAME, SURFACE, FORMAT)                                                                 \
	{NAME, SURFACE, 0, FORMAT, MTLTextureType2DMultisample, 64, 64, 1, 1, 1, 4,                    \
	 MTLTextureUsageRenderTarget, MTLStorageModePrivate, kDefault, kDefault, 0}

#define BGRA8 MTLPixelFormatBGRA8Unorm
static const struct Case kCases[] = {
    // The surface's own shape: what every other case changes one thing of.
    C("bgra.base", kBGRA, 0, BGRA8, 64, 64),
    C("bgra1.base", kBGRA1, 0, BGRA8, 64, 64),
    C("rgha.base", kRGhA, 0, MTLPixelFormatRGBA16Float, 64, 64),
    C("l008.base", kL008, 0, MTLPixelFormatR8Unorm, 64, 64),
    C("420v.p0.base", k420v, 0, MTLPixelFormatR8Unorm, 64, 64),
    C("420v.p1.base", k420v, 1, MTLPixelFormatRG8Unorm, 32, 32),
    C("420v.p0.ycbcr", k420v, 0, (MTLPixelFormat)500, 64, 64),
    // Planes.
    C("bgra.plane1", kBGRA, 1, BGRA8, 64, 64),
    C("bgra1.plane1", kBGRA1, 1, BGRA8, 64, 64),
    C("420v.plane2", k420v, 2, MTLPixelFormatR8Unorm, 64, 64),
    // Size, against the plane's.
    C("bgra.32x32", kBGRA, 0, BGRA8, 32, 32),
    C("bgra.64x1", kBGRA, 0, BGRA8, 64, 1),
    C("bgra.65x64", kBGRA, 0, BGRA8, 65, 64),
    C("bgra.64x65", kBGRA, 0, BGRA8, 64, 65),
    C("bgra.128x128", kBGRA, 0, BGRA8, 128, 128),
    C("bgra.0x64", kBGRA, 0, BGRA8, 0, 64),
    C("bgra.64x0", kBGRA, 0, BGRA8, 64, 0),
    C("bgra1.65x64", kBGRA1, 0, BGRA8, 65, 64),
    C("bgra1.64x65", kBGRA1, 0, BGRA8, 64, 65),
    C("420v.p0.65x64", k420v, 0, MTLPixelFormatR8Unorm, 65, 64),
    C("420v.p0.64x65", k420v, 0, MTLPixelFormatR8Unorm, 64, 65),
    C("420v.p1.33x32", k420v, 1, MTLPixelFormatRG8Unorm, 33, 32),
    C("420v.p1.32x33", k420v, 1, MTLPixelFormatRG8Unorm, 32, 33),
    C("420v.p1.64x64", k420v, 1, MTLPixelFormatRG8Unorm, 64, 64),
    C("420v.p0.ycbcr.66x64", k420v, 0, (MTLPixelFormat)500, 66, 64),
    // Type, depth, array length, levels and samples.
    CT("bgra.type1d", kBGRA, BGRA8, MTLTextureType1D, 64, 1, 1, 1, 1, 1),
    CT("bgra.type1darray", kBGRA, BGRA8, MTLTextureType1DArray, 64, 1, 1, 1, 1, 1),
    CT("bgra.type2darray", kBGRA, BGRA8, MTLTextureType2DArray, 64, 64, 1, 1, 1, 1),
    CT("bgra.type2darray.a2", kBGRA, BGRA8, MTLTextureType2DArray, 64, 32, 1, 2, 1, 1),
    CT("bgra.type2dms", kBGRA, BGRA8, MTLTextureType2DMultisample, 64, 64, 1, 1, 1, 4),
    CT("bgra.type2dms.s1", kBGRA, BGRA8, MTLTextureType2DMultisample, 64, 64, 1, 1, 1, 1),
    CT("bgra.typecube", kBGRA, BGRA8, MTLTextureTypeCube, 64, 64, 1, 1, 1, 1),
    CT("bgra.typecubearray", kBGRA, BGRA8, MTLTextureTypeCubeArray, 64, 64, 1, 1, 1, 1),
    CT("bgra.type3d", kBGRA, BGRA8, MTLTextureType3D, 64, 64, 1, 1, 1, 1),
    CT("bgra.type3d.d2", kBGRA, BGRA8, MTLTextureType3D, 64, 32, 2, 1, 1, 1),
    CT("bgra.typebuffer", kBGRA, BGRA8, MTLTextureTypeTextureBuffer, 64, 1, 1, 1, 1, 1),
    CT("bgra.depth2", kBGRA, BGRA8, T2D, 64, 32, 2, 1, 1, 1),
    CT("bgra.array2", kBGRA, BGRA8, T2D, 64, 32, 1, 2, 1, 1),
    CT("bgra.mips2", kBGRA, BGRA8, T2D, 64, 64, 1, 1, 2, 1),
    CT("bgra.mips7", kBGRA, BGRA8, T2D, 64, 64, 1, 1, 7, 1),
    CT("bgra.samples4", kBGRA, BGRA8, T2D, 64, 64, 1, 1, 1, 4),
    CT("bgra.samples2", kBGRA, BGRA8, T2D, 64, 64, 1, 1, 1, 2),
    CT("420v.p0.mips2", k420v, MTLPixelFormatR8Unorm, T2D, 64, 64, 1, 1, 2, 1),
    CT("420v.p0.ycbcr.mips2", k420v, (MTLPixelFormat)500, T2D, 64, 64, 1, 1, 2, 1),
    // Formats of the bytes of the plane's elements, and of other sizes.
    C("bgra.fmt.rgba8", kBGRA, 0, MTLPixelFormatRGBA8Unorm, 64, 64),
    C("bgra.fmt.bgra8srgb", kBGRA, 0, MTLPixelFormatBGRA8Unorm_sRGB, 64, 64),
    C("bgra.fmt.r32f", kBGRA, 0, MTLPixelFormatR32Float, 64, 64),
    C("bgra.fmt.r32u", kBGRA, 0, MTLPixelFormatR32Uint, 64, 64),
    C("bgra.fmt.rg16f", kBGRA, 0, MTLPixelFormatRG16Float, 64, 64),
    C("bgra.fmt.rgb10a2", kBGRA, 0, MTLPixelFormatRGB10A2Unorm, 64, 64),
    C("bgra.fmt.bgr10a2", kBGRA, 0, MTLPixelFormatBGR10A2Unorm, 64, 64),
    C("bgra.fmt.rg11b10f", kBGRA, 0, MTLPixelFormatRG11B10Float, 64, 64),
    C("bgra.fmt.rgb9e5", kBGRA, 0, MTLPixelFormatRGB9E5Float, 64, 64),
    C("bgra.fmt.bgr10xr", kBGRA, 0, MTLPixelFormatBGR10_XR, 64, 64),
    C("bgra.fmt.bgra10xr", kBGRA, 0, MTLPixelFormatBGRA10_XR, 64, 64),
    C("bgra.fmt.rgba16f", kBGRA, 0, MTLPixelFormatRGBA16Float, 64, 64),
    C("bgra.fmt.rgba16f.32x64", kBGRA, 0, MTLPixelFormatRGBA16Float, 32, 64),
    C("bgra.fmt.rgba32f", kBGRA, 0, MTLPixelFormatRGBA32Float, 64, 64),
    C("bgra.fmt.rgba32f.16x64", kBGRA, 0, MTLPixelFormatRGBA32Float, 16, 64),
    C("bgra.fmt.r8", kBGRA, 0, MTLPixelFormatR8Unorm, 64, 64),
    C("bgra.fmt.r8.256x64", kBGRA, 0, MTLPixelFormatR8Unorm, 256, 64),
    C("bgra.fmt.a8", kBGRA, 0, MTLPixelFormatA8Unorm, 64, 64),
    C("bgra.fmt.rg8", kBGRA, 0, MTLPixelFormatRG8Unorm, 64, 64),
    C("bgra.fmt.rg8.128x64", kBGRA, 0, MTLPixelFormatRG8Unorm, 128, 64),
    C("bgra.fmt.r16f", kBGRA, 0, MTLPixelFormatR16Float, 64, 64),
    C("bgra.fmt.b5g6r5", kBGRA, 0, MTLPixelFormatB5G6R5Unorm, 64, 64),
    C("bgra.fmt.bc1", kBGRA, 0, MTLPixelFormatBC1_RGBA, 64, 64),
    C("bgra.fmt.bc7", kBGRA, 0, MTLPixelFormatBC7_RGBAUnorm, 64, 64),
    C("bgra.fmt.invalid", kBGRA, 0, MTLPixelFormatInvalid, 64, 64),
    C("bgra.fmt.gbgr422", kBGRA, 0, MTLPixelFormatGBGR422, 64, 64),
    C("bgra.fmt.gbgr422.128x64", kBGRA, 0, MTLPixelFormatGBGR422, 128, 64),
    C("bgra.fmt.ycbcr", kBGRA, 0, (MTLPixelFormat)500, 64, 64),
    C("bgra.fmt.depth32f", kBGRA, 0, MTLPixelFormatDepth32Float, 64, 64),
    C("bgra.fmt.depth16", kBGRA, 0, MTLPixelFormatDepth16Unorm, 64, 64),
    C("bgra.fmt.stencil8", kBGRA, 0, MTLPixelFormatStencil8, 64, 64),
    C("bgra.fmt.d32s8", kBGRA, 0, MTLPixelFormatDepth32Float_Stencil8, 64, 64),
    C("bgra.fmt.d24s8", kBGRA, 0, MTLPixelFormatDepth24Unorm_Stencil8, 64, 64),
    C("bgra.fmt.x32s8", kBGRA, 0, MTLPixelFormatX32_Stencil8, 64, 64),
    C("rgha.fmt.bgra8", kRGhA, 0, BGRA8, 64, 64),
    C("rgha.fmt.bgra8.128x64", kRGhA, 0, BGRA8, 128, 64),
    C("rgha.fmt.rg32f", kRGhA, 0, MTLPixelFormatRG32Float, 64, 64),
    C("rgha.fmt.rgba16u", kRGhA, 0, MTLPixelFormatRGBA16Uint, 64, 64),
    C("rgha.fmt.rgba32f", kRGhA, 0, MTLPixelFormatRGBA32Float, 64, 64),
    C("l008.fmt.a8", kL008, 0, MTLPixelFormatA8Unorm, 64, 64),
    C("l008.fmt.rg8", kL008, 0, MTLPixelFormatRG8Unorm, 64, 64),
    C("l008.fmt.rg8.32x64", kL008, 0, MTLPixelFormatRG8Unorm, 32, 64),
    C("l008.fmt.bgra8.16x64", kL008, 0, BGRA8, 16, 64),
    C("420v.p0.fmt.rg8", k420v, 0, MTLPixelFormatRG8Unorm, 64, 64),
    C("420v.p0.fmt.rg8.32x64", k420v, 0, MTLPixelFormatRG8Unorm, 32, 64),
    C("420v.p0.fmt.bgra8.16x64", k420v, 0, BGRA8, 16, 64),
    C("420v.p0.fmt.bgra8", k420v, 0, BGRA8, 64, 64),
    C("420v.p1.fmt.r8", k420v, 1, MTLPixelFormatR8Unorm, 32, 32),
    C("420v.p1.fmt.r8.64x32", k420v, 1, MTLPixelFormatR8Unorm, 64, 32),
    C("420v.p1.fmt.r16", k420v, 1, MTLPixelFormatR16Unorm, 32, 32),
    C("420v.p1.fmt.bgra8.16x32", k420v, 1, BGRA8, 16, 32),
    C("420v.p1.ycbcr", k420v, 1, (MTLPixelFormat)500, 32, 32),
    C("420v.p0.ycbcr.63x63", k420v, 0, (MTLPixelFormat)500, 63, 63),
    // Usage, storage, CPU cache and hazard tracking modes, swizzle.
    CU("bgra.usage0", kBGRA, 0, BGRA8, 64, 64, 0, kDefault, kDefault, kDefault, 0),
    CU("bgra.usage.write", kBGRA, 0, BGRA8, 64, 64, MTLTextureUsageShaderWrite, kDefault,
       kDefault, kDefault, 0),
    CU("bgra.usage.rt", kBGRA, 0, BGRA8, 64, 64, MTLTextureUsageRenderTarget, kDefault, kDefault,
       kDefault, 0),
    CU("bgra.usage.all", kBGRA, 0, BGRA8, 64, 64,
       MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite | MTLTextureUsageRenderTarget |
           MTLTextureUsagePixelFormatView,
       kDefault, kDefault, kDefault, 0),
    CU("bgra.usage.atomic", kBGRA, 0, MTLPixelFormatR32Uint, 64, 64,
       MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite | 0x20, kDefault, kDefault,
       kDefault, 0),
    CU("bgra.usage.view", kBGRA, 0, BGRA8, 64, 64,
       MTLTextureUsageShaderRead | MTLTextureUsagePixelFormatView, kDefault, kDefault, kDefault,
       0),
    CU("420v.p0.ycbcr.rt", k420v, 0, (MTLPixelFormat)500, 64, 64,
       MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget, kDefault, kDefault, kDefault, 0),
    CU("420v.p0.ycbcr.63x63.rt", k420v, 0, (MTLPixelFormat)500, 63, 63,
       MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget, kDefault, kDefault, kDefault, 0),
    CU("420v.p0.ycbcr.write", k420v, 0, (MTLPixelFormat)500, 64, 64,
       MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite, kDefault, kDefault, kDefault, 0),
    CU("bgra.shared", kBGRA, 0, BGRA8, 64, 64, kDefault, MTLStorageModeShared, kDefault, kDefault,
       0),
    CU("bgra.managed", kBGRA, 0, BGRA8, 64, 64, kDefault, MTLStorageModeManaged, kDefault,
       kDefault, 0),
    CU("bgra.private", kBGRA, 0, BGRA8, 64, 64, kDefault, MTLStorageModePrivate, kDefault,
       kDefault, 0),
    CU("bgra.memoryless", kBGRA, 0, BGRA8, 64, 64, kDefault, 3, kDefault, kDefault, 0),
    CU("bgra.private.rt", kBGRA, 0, BGRA8, 64, 64, MTLTextureUsageRenderTarget,
       MTLStorageModePrivate, kDefault, kDefault, 0),
    CU("bgra.writecombined", kBGRA, 0, BGRA8, 64, 64, kDefault, kDefault,
       MTLCPUCacheModeWriteCombined, kDefault, 0),
    CU("bgra.untracked", kBGRA, 0, BGRA8, 64, 64, kDefault, kDefault, kDefault,
       MTLHazardTrackingModeUntracked, 0),
    CU("bgra.tracked", kBGRA, 0, BGRA8, 64, 64, kDefault, kDefault, kDefault,
       MTLHazardTrackingModeTracked, 0),
    CU("bgra.swizzle", kBGRA, 0, BGRA8, 64, 64, kDefault, kDefault, kDefault, kDefault, 1),
    CU("bgra.fmt.depth32f.private", kBGRA, 0, MTLPixelFormatDepth32Float, 64, 64,
       MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget, MTLStorageModePrivate, kDefault,
       kDefault, 0),
    // The second round: types, levels and formats over a plane of a planar surface; compressed
    // formats at the flat surface's rows; a surface whose rows are wider than its pixels.
    CT("bgra1.type1d", kBGRA1, BGRA8, MTLTextureType1D, 64, 1, 1, 1, 1, 1),
    CT("bgra1.type2darray", kBGRA1, BGRA8, MTLTextureType2DArray, 64, 64, 1, 1, 1, 1),
    CT("bgra1.type2darray.a2", kBGRA1, BGRA8, MTLTextureType2DArray, 64, 32, 1, 2, 1, 1),
    CT("bgra1.typecube", kBGRA1, BGRA8, MTLTextureTypeCube, 64, 64, 1, 1, 1, 1),
    CT("bgra1.type3d.d2", kBGRA1, BGRA8, MTLTextureType3D, 64, 32, 2, 1, 1, 1),
    CT("bgra1.typebuffer", kBGRA1, BGRA8, MTLTextureTypeTextureBuffer, 64, 1, 1, 1, 1, 1),
    CMS("bgra1.type2dms.private", kBGRA1, BGRA8),
    CT("bgra1.mips2", kBGRA1, BGRA8, T2D, 64, 64, 1, 1, 2, 1),
    CU("bgra1.fmt.depth32f.private", kBGRA1, 0, MTLPixelFormatDepth32Float, 64, 64,
       MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget, MTLStorageModePrivate, kDefault,
       kDefault, 0),
    C("bgra1.fmt.bc1", kBGRA1, 0, MTLPixelFormatBC1_RGBA, 64, 64),
    CT("bgra1.depth2", kBGRA1, BGRA8, T2D, 64, 32, 2, 1, 1, 1),
    CT("420v.p0.type2darray", k420v, MTLPixelFormatR8Unorm, MTLTextureType2DArray, 64, 64, 1, 1,
       1, 1),
    CT("420v.p0.type3d.d2", k420v, MTLPixelFormatR8Unorm, MTLTextureType3D, 64, 32, 2, 1, 1, 1),
    CT("420v.p0.typebuffer", k420v, MTLPixelFormatR8Unorm, MTLTextureTypeTextureBuffer, 64, 1, 1,
       1, 1, 1),
    CMS("bgra.type2dms.private", kBGRA, BGRA8),
    C("bgra.fmt.bc1.128x64", kBGRA, 0, MTLPixelFormatBC1_RGBA, 128, 64),
    C("bgra.fmt.bc1.132x64", kBGRA, 0, MTLPixelFormatBC1_RGBA, 132, 64),
    C("bgra.fmt.bc7.68x64", kBGRA, 0, MTLPixelFormatBC7_RGBAUnorm, 68, 64),
    C("bgra.fmt.bc1.64x256", kBGRA, 0, MTLPixelFormatBC1_RGBA, 64, 256),
    C("bgra.fmt.bc1.64x68", kBGRA, 0, MTLPixelFormatBC1_RGBA, 64, 68),
    CT("bgra.typebuffer.65", kBGRA, BGRA8, MTLTextureTypeTextureBuffer, 65, 1, 1, 1, 1, 1),
    CT("bgra.type2darray.s4", kBGRA, BGRA8, MTLTextureType2DArray, 64, 64, 1, 1, 1, 4),
    CT("bgra.type2darray.d2", kBGRA, BGRA8, MTLTextureType2DArray, 64, 32, 2, 1, 1, 1),
    C("bgrapad.base", kBGRAPad, 0, BGRA8, 64, 64),
    C("bgrapad.128x64", kBGRAPad, 0, BGRA8, 128, 64),
    C("bgrapad.129x64", kBGRAPad, 0, BGRA8, 129, 64),
    C("bgrapad.64x65", kBGRAPad, 0, BGRA8, 64, 65),
    C("bgra.plane2", kBGRA, 2, BGRA8, 64, 64),
    C("bgra.fmt.gbgr422.63x64", kBGRA, 0, MTLPixelFormatGBGR422, 63, 64),
    C("bgra.fmt.gbgr422.64x63", kBGRA, 0, MTLPixelFormatGBGR422, 64, 63),
    CU("bgra.fmt.gbgr422.64x63.rt", kBGRA, 0, MTLPixelFormatGBGR422, 64, 63,
       MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget, kDefault, kDefault, kDefault, 0),
    C("bgra.fmt.bc1.62x64", kBGRA, 0, MTLPixelFormatBC1_RGBA, 62, 64),
    CU("bgra.fmt.bc1.private", kBGRA, 0, MTLPixelFormatBC1_RGBA, 64, 64, kDefault,
       MTLStorageModePrivate, kDefault, kDefault, 0),
    CU("420v.p0.ycbcr.63x64.rt", k420v, 0, (MTLPixelFormat)500, 63, 64,
       MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget, kDefault, kDefault, kDefault, 0),
    CU("420v.p0.ycbcr.64x63.rt", k420v, 0, (MTLPixelFormat)500, 64, 63,
       MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget, kDefault, kDefault, kDefault, 0),
};
enum { kCaseCount = sizeof kCases / sizeof kCases[0] };

// A surface without planes; `bytesPerRow` 0 leaves its rows to IOSurface.
static IOSurfaceRef flatSurface(OSType format, int bytesPerElement, int bytesPerRow)
{
	NSMutableDictionary *props = [@{
		(id)kIOSurfaceWidth : @(kSize),
		(id)kIOSurfaceHeight : @(kSize),
		(id)kIOSurfaceBytesPerElement : @(bytesPerElement),
		(id)kIOSurfacePixelFormat : @(format),
	} mutableCopy];
	if (bytesPerRow != 0)
		props[(id)kIOSurfaceBytesPerRow] = @(bytesPerRow);
	return IOSurfaceCreate((__bridge CFDictionaryRef)props);
}

// A surface of `surface` (one the caller releases), or NULL with a line saying why.
static IOSurfaceRef makeSurface(enum Surface surface, CVPixelBufferRef *keep)
{
	switch (surface) {
	case kBGRA:
		return flatSurface('BGRA', 4, 0);
	case kBGRAPad:
		return flatSurface('BGRA', 4, 512);
	case kRGhA:
		return flatSurface('RGhA', 8, 0);
	case kL008:
		return flatSurface('L008', 1, 0);
	case kBGRA1: {
		const size_t row = kSize * 4;
		NSDictionary *props = @{
			(id)kIOSurfaceWidth : @(kSize),
			(id)kIOSurfaceHeight : @(kSize),
			(id)kIOSurfacePixelFormat : @((uint32_t)'BGRA'),
			(id)kIOSurfaceAllocSize : @(row * kSize),
			(id)kIOSurfacePlaneInfo : @[ @{
				(id)kIOSurfacePlaneWidth : @(kSize),
				(id)kIOSurfacePlaneHeight : @(kSize),
				(id)kIOSurfacePlaneBytesPerElement : @4,
				(id)kIOSurfacePlaneBytesPerRow : @(row),
				(id)kIOSurfacePlaneOffset : @0,
				(id)kIOSurfacePlaneSize : @(row * kSize),
			} ],
		};
		return IOSurfaceCreate((__bridge CFDictionaryRef)props);
	}
	case k420v: {
		NSDictionary *attrs = @{(id)kCVPixelBufferIOSurfacePropertiesKey : @{}};
		CVReturn cr = CVPixelBufferCreate(kCFAllocatorDefault, kSize, kSize, '420v',
		                                  (__bridge CFDictionaryRef)attrs, keep);
		if (cr != kCVReturnSuccess || *keep == NULL) {
			printf("  no 420v pixel buffer (%d)\n", cr);
			return NULL;
		}
		IOSurfaceRef s = CVPixelBufferGetIOSurface(*keep);
		if (s != NULL)
			CFRetain(s);
		return s;
	}
	}
	return NULL;
}

static void surfaceText(IOSurfaceRef s, char *out, size_t n)
{
	uint32_t f = IOSurfaceGetPixelFormat(s);
	int k = snprintf(out, n, "'%c%c%c%c' %zux%zu, %zu planes", (char)(f >> 24), (char)(f >> 16),
	                 (char)(f >> 8), (char)f, IOSurfaceGetWidth(s), IOSurfaceGetHeight(s),
	                 IOSurfaceGetPlaneCount(s));
	size_t planes = IOSurfaceGetPlaneCount(s);
	for (size_t p = 0; p < planes && k > 0 && (size_t)k < n; p++)
		k += snprintf(out + k, n - k, "; plane %zu %zux%zu, %zu bytes an element, %zu a row", p,
		              IOSurfaceGetWidthOfPlane(s, p), IOSurfaceGetHeightOfPlane(s, p),
		              IOSurfaceGetBytesPerElementOfPlane(s, p),
		              IOSurfaceGetBytesPerRowOfPlane(s, p));
	if (planes == 0 && k > 0 && (size_t)k < n)
		snprintf(out + k, n - k, "; %zu bytes an element, %zu a row",
		         IOSurfaceGetBytesPerElement(s), IOSurfaceGetBytesPerRow(s));
}

static int runCase(id<MTLDevice> d, int n)
{
	const struct Case *c = &kCases[n];
	CVPixelBufferRef keep = NULL;
	IOSurfaceRef s = makeSurface(c->surface, &keep);
	if (s == NULL) {
		printf("case %d %s => no surface\n", n, c->name);
		return 1;
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
	if (c->storage != kDefault)
		td.storageMode = (MTLStorageMode)c->storage;
	if (c->cache != kDefault)
		td.cpuCacheMode = (MTLCPUCacheMode)c->cache;
	if (c->hazard != kDefault)
		td.hazardTrackingMode = (MTLHazardTrackingMode)c->hazard;
	if (c->swizzled)
		td.swizzle = MTLTextureSwizzleChannelsMake(MTLTextureSwizzleBlue, MTLTextureSwizzleGreen,
		                                           MTLTextureSwizzleRed, MTLTextureSwizzleAlpha);
	char surface[400];
	surfaceText(s, surface, sizeof surface);
	printf("case %d %s | %s %s plane %d | asked: fmt %lu type %lu %lux%lux%lu a %lu m %lu s %lu "
	       "usage 0x%lx storage %lu cache %lu hazard %lu\n",
	       n, c->name, kSurfaceNames[c->surface], surface, c->plane, (unsigned long)td.pixelFormat,
	       (unsigned long)td.textureType, (unsigned long)td.width, (unsigned long)td.height,
	       (unsigned long)td.depth, (unsigned long)td.arrayLength,
	       (unsigned long)td.mipmapLevelCount, (unsigned long)td.sampleCount,
	       (unsigned long)td.usage, (unsigned long)td.storageMode,
	       (unsigned long)td.cpuCacheMode, (unsigned long)td.hazardTrackingMode);
	fflush(stdout);
	id<MTLTexture> t = [d newTextureWithDescriptor:td iosurface:s plane:(NSUInteger)c->plane];
	if (t == nil) {
		printf("case %d %s => nil\n", n, c->name);
	} else {
		printf("case %d %s => made: fmt %lu type %lu %lux%lux%lu a %lu m %lu s %lu usage 0x%lx "
		       "storage %lu cache %lu hazard %lu plane %lu\n",
		       n, c->name, (unsigned long)t.pixelFormat, (unsigned long)t.textureType,
		       (unsigned long)t.width, (unsigned long)t.height, (unsigned long)t.depth,
		       (unsigned long)t.arrayLength, (unsigned long)t.mipmapLevelCount,
		       (unsigned long)t.sampleCount, (unsigned long)t.usage,
		       (unsigned long)t.storageMode, (unsigned long)t.cpuCacheMode,
		       (unsigned long)t.hazardTrackingMode, (unsigned long)t.iosurfacePlane);
	}
	fflush(stdout);
	t = nil;
	CFRelease(s);
	if (keep != NULL)
		CVPixelBufferRelease(keep);
	return 0;
}

int main(int argc, char **argv)
{
	setvbuf(stdout, NULL, _IOLBF, 0);
	if (argc < 2) {
		fprintf(stderr, "usage: %s DEVICE [--case N | --list]\n", argv[0]);
		return 2;
	}
	if (argc > 2 && !strcmp(argv[2], "--list")) {
		for (int n = 0; n < kCaseCount; n++)
			printf("%d %s\n", n, kCases[n].name);
		return 0;
	}
	if (argc > 3 && !strcmp(argv[2], "--case")) {
		// Metal's words where it ends the process go to stderr: put them in the case's lines.
		dup2(STDOUT_FILENO, STDERR_FILENO);
		int n = atoi(argv[3]);
		if (n < 0 || n >= kCaseCount)
			return 2;
		return runCase(oracle_device(argv[1]), n);
	}
	oracle_header("oracle_surftex", oracle_device(argv[1]));
	fflush(stdout);
	for (int n = 0; n < kCaseCount; n++) {
		char index[16];
		snprintf(index, sizeof index, "%d", n);
		char *args[] = {argv[0], argv[1], "--case", index, NULL};
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
