// oracle_surftex: which textures newTextureWithDescriptor:iosurface:plane: makes over a plane of
// an IOSurface, which it answers nil for, and at which Metal ends the process, by the
// descriptor's type, size, depth, array length, levels, samples, pixel format, usage, storage,
// CPU cache and hazard tracking modes and swizzle, and by the surface's OSType, planes and the
// plane asked for, and every two-plane format. No Metal validation layer is asked for: what a
// guest's process meets.
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
// and hazard tracking modes and the identity swizzle, and changes what the case's name says. A
// `nosurface.` case makes its texture by newTextureWithDescriptor: alone, for the answers to set
// beside the surface's.
// Each case prints:
//   case N NAME | SURFACE plane P | asked: fmt F type T WxHxD a A m M s S usage U storage S'
//   case N NAME => made: fmt F type T WxHxD a A m M s S usage U storage S plane P | nil
// with, for a case that also makes a view of its texture, `case N NAME => view: fmt F usage U`,
// and, where the case ends the process, Metal's words (its stderr is the child's stdout), then
//   => case N died on signal X | exited X
// Then the aliasing cases (the seventh round, 11-13), each in a child process of its own
// (`--alias N`): one command buffer writes texels of an 'L008' 64 x 64 surface of 128-byte rows
// through one texture over its plane in its first encoder, and reads them through another in its
// second. `wide` is RG8Unorm 64 x 64, its rows the surface's whole 128 bytes (the plane's texels
// and the bytes past them); `narrow` is R8Unorm 64 x 64, the plane's own texels. The writer is a
// blit from a buffer (`blit`) or a render pass that clears and stores (`clear`); the reader is a
// blit into a Shared buffer. Both textures keep the device's hazard tracking (`tracked`), or are
// Untracked with the writer updating a fence the reader waits for (`fence`). Each case runs
// kAliasRuns command buffers, the CPU setting every byte of the surface to 0x11 under a lock
// before each, and prints, of the bytes the writer wrote that the reader's texels hold, how many
// runs read all of them, none, or some; and, of the surface after the command buffer completed
// (read under a read-only lock), in how many runs the plane's texels held every byte written
// there, the bytes past them every byte the wide writer wrote there, and the bytes past them
// still held the CPU's 0x11:
//   alias N NAME | surface ... | WRITER through W, read through R, HAZARD
//   alias N NAME => runs R: all A, none Z, some S; surface after: texels all P, bytes past them
//   written Q, still 0x11 O
//   oracle_surftex DEVICE --alias N    aliasing case N alone (what the children run)
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
	int mode;     // kOverSurface, kNoSurface or kViewOf
};

// What a case makes: the texture over the surface's plane; the same descriptor's texture by
// newTextureWithDescriptor:, over no surface; or the texture over the plane and then a view of it
// in the same format, both printed.
enum { kOverSurface = 0, kNoSurface = 1, kViewOf = 2 };

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
// The same of USAGE and STORAGE, made as MODE says.
#define CM(NAME, SURFACE, PLANE, FORMAT, W, H, USAGE, STORAGE, MODE)                               \
	{NAME, SURFACE, PLANE, FORMAT, T2D, W, H, 1, 1, 1, 1, USAGE, STORAGE, kDefault, kDefault, 0,    \
	 MODE}
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
    // The third round: every two-plane format over the luma plane, and 500 over the chroma one.
    C("420v.p0.fmt502", k420v, 0, (MTLPixelFormat)502, 64, 64),
    C("420v.p0.fmt503", k420v, 0, (MTLPixelFormat)503, 64, 64),
    C("420v.p0.fmt505", k420v, 0, (MTLPixelFormat)505, 64, 64),
    C("420v.p0.fmt506", k420v, 0, (MTLPixelFormat)506, 64, 64),
    C("420v.p0.fmt507", k420v, 0, (MTLPixelFormat)507, 64, 64),
    C("420v.p0.fmt508", k420v, 0, (MTLPixelFormat)508, 64, 64),
    C("420v.p0.fmt509", k420v, 0, (MTLPixelFormat)509, 64, 64),
    C("420v.p0.fmt510", k420v, 0, (MTLPixelFormat)510, 64, 64),
    C("420v.p0.fmt520", k420v, 0, (MTLPixelFormat)520, 64, 64),
    C("420v.p0.fmt522", k420v, 0, (MTLPixelFormat)522, 64, 64),
    C("420v.p0.fmt523", k420v, 0, (MTLPixelFormat)523, 64, 64),
    C("420v.p0.fmt525", k420v, 0, (MTLPixelFormat)525, 64, 64),
    C("420v.p0.fmt526", k420v, 0, (MTLPixelFormat)526, 64, 64),
    C("420v.p0.fmt527", k420v, 0, (MTLPixelFormat)527, 64, 64),
    C("420v.p0.fmt528", k420v, 0, (MTLPixelFormat)528, 64, 64),
    C("420v.p0.fmt529", k420v, 0, (MTLPixelFormat)529, 64, 64),
    C("420v.p0.fmt530", k420v, 0, (MTLPixelFormat)530, 64, 64),
    C("420v.p0.fmt540", k420v, 0, (MTLPixelFormat)540, 64, 64),
    C("420v.p0.fmt541", k420v, 0, (MTLPixelFormat)541, 64, 64),
    C("420v.p0.fmt542", k420v, 0, (MTLPixelFormat)542, 64, 64),
    C("420v.p0.fmt543", k420v, 0, (MTLPixelFormat)543, 64, 64),
    C("420v.p0.fmt544", k420v, 0, (MTLPixelFormat)544, 64, 64),
    C("420v.p0.fmt545", k420v, 0, (MTLPixelFormat)545, 64, 64),
    C("420v.p0.fmt550", k420v, 0, (MTLPixelFormat)550, 64, 64),
    C("420v.p0.fmt551", k420v, 0, (MTLPixelFormat)551, 64, 64),
    C("420v.p1.fmt500", k420v, 1, (MTLPixelFormat)500, 32, 32),
    C("bgra1.fmt500", kBGRA1, 0, (MTLPixelFormat)500, 64, 64),
    // The fourth round (11-13), where an Apple GPU answers otherwise than the x86 Macs: the
    // compressed formats only an Apple GPU has, over a surface without planes and over a plane;
    // the cube types over a plane; and a two-plane texture of an odd size, as a render target and
    // not, over each kind of plane.
    C("bgra.fmt.astc4x4", kBGRA, 0, MTLPixelFormatASTC_4x4_LDR, 64, 64),
    C("bgra.fmt.astc4x4.68x64", kBGRA, 0, MTLPixelFormatASTC_4x4_LDR, 68, 64),
    C("bgra.fmt.astc4x4.64x68", kBGRA, 0, MTLPixelFormatASTC_4x4_LDR, 64, 68),
    C("bgra.fmt.etc2", kBGRA, 0, MTLPixelFormatETC2_RGB8, 64, 64),
    C("bgra.fmt.eac", kBGRA, 0, MTLPixelFormatEAC_R11Unorm, 64, 64),
    C("bgra.fmt.pvrtc", kBGRA, 0, (MTLPixelFormat)166, 64, 64),
    C("bgra1.fmt.astc4x4", kBGRA1, 0, MTLPixelFormatASTC_4x4_LDR, 64, 64),
    C("bgra1.fmt.etc2", kBGRA1, 0, MTLPixelFormatETC2_RGB8, 64, 64),
    C("bgra1.fmt.bc7", kBGRA1, 0, MTLPixelFormatBC7_RGBAUnorm, 64, 64),
    C("420v.p0.fmt.bc1", k420v, 0, MTLPixelFormatBC1_RGBA, 64, 64),
    CT("bgra1.typecubearray", kBGRA1, BGRA8, MTLTextureTypeCubeArray, 64, 64, 1, 1, 1, 1),
    CT("420v.p0.typecube", k420v, MTLPixelFormatR8Unorm, MTLTextureTypeCube, 64, 64, 1, 1, 1, 1),
    CT("420v.p0.typecubearray", k420v, MTLPixelFormatR8Unorm, MTLTextureTypeCubeArray, 64, 64,
       1, 1, 1, 1),
    CT("bgra1.typecube.32", kBGRA1, BGRA8, MTLTextureTypeCube, 32, 32, 1, 1, 1, 1),
    CU("420v.p0.ycbcr.63x63.write", k420v, 0, (MTLPixelFormat)500, 63, 63,
       MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite, kDefault, kDefault, kDefault, 0),
    CU("420v.p0.ycbcr.62x62.rt", k420v, 0, (MTLPixelFormat)500, 62, 62,
       MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget, kDefault, kDefault, kDefault, 0),
    CU("420v.p1.fmt500.31x31.rt", k420v, 1, (MTLPixelFormat)500, 31, 31,
       MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget, kDefault, kDefault, kDefault, 0),
    CU("bgra1.fmt500.63x63.rt", kBGRA1, 0, (MTLPixelFormat)500, 63, 63,
       MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget, kDefault, kDefault, kDefault, 0),
    CU("bgra.fmt.ycbcr.63x63.rt", kBGRA, 0, (MTLPixelFormat)500, 63, 63,
       MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget, kDefault, kDefault, kDefault, 0),
    C("bgra.fmt.ycbcr.63x63", kBGRA, 0, (MTLPixelFormat)500, 63, 63),
    CU("420v.p0.fmt502.63x63.rt", k420v, 0, (MTLPixelFormat)502, 63, 63,
       MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget, kDefault, kDefault, kDefault, 0),
    CU("420v.p0.fmt503.63x63.rt", k420v, 0, (MTLPixelFormat)503, 63, 63,
       MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget, kDefault, kDefault, kDefault, 0),
    CU("420v.p0.fmt505.63x63.rt", k420v, 0, (MTLPixelFormat)505, 63, 63,
       MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget, kDefault, kDefault, kDefault, 0),
    CU("420v.p0.fmt506.63x63.rt", k420v, 0, (MTLPixelFormat)506, 63, 63,
       MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget, kDefault, kDefault, kDefault, 0),
    CU("420v.p0.fmt507.63x63.rt", k420v, 0, (MTLPixelFormat)507, 63, 63,
       MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget, kDefault, kDefault, kDefault, 0),
    CU("420v.p0.fmt508.63x63.rt", k420v, 0, (MTLPixelFormat)508, 63, 63,
       MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget, kDefault, kDefault, kDefault, 0),
    CU("420v.p0.fmt509.63x63.rt", k420v, 0, (MTLPixelFormat)509, 63, 63,
       MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget, kDefault, kDefault, kDefault, 0),
    CU("420v.p0.fmt510.63x63.rt", k420v, 0, (MTLPixelFormat)510, 63, 63,
       MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget, kDefault, kDefault, kDefault, 0),
    CU("420v.p0.fmt520.63x63.rt", k420v, 0, (MTLPixelFormat)520, 63, 63,
       MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget, kDefault, kDefault, kDefault, 0),
    CU("420v.p0.fmt522.63x63.rt", k420v, 0, (MTLPixelFormat)522, 63, 63,
       MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget, kDefault, kDefault, kDefault, 0),
    CU("420v.p0.fmt523.63x63.rt", k420v, 0, (MTLPixelFormat)523, 63, 63,
       MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget, kDefault, kDefault, kDefault, 0),
    CU("420v.p0.fmt525.63x63.rt", k420v, 0, (MTLPixelFormat)525, 63, 63,
       MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget, kDefault, kDefault, kDefault, 0),
    CU("420v.p0.fmt526.63x63.rt", k420v, 0, (MTLPixelFormat)526, 63, 63,
       MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget, kDefault, kDefault, kDefault, 0),
    CU("420v.p0.fmt527.63x63.rt", k420v, 0, (MTLPixelFormat)527, 63, 63,
       MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget, kDefault, kDefault, kDefault, 0),
    CU("420v.p0.fmt528.63x63.rt", k420v, 0, (MTLPixelFormat)528, 63, 63,
       MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget, kDefault, kDefault, kDefault, 0),
    CU("420v.p0.fmt529.63x63.rt", k420v, 0, (MTLPixelFormat)529, 63, 63,
       MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget, kDefault, kDefault, kDefault, 0),
    CU("420v.p0.fmt530.63x63.rt", k420v, 0, (MTLPixelFormat)530, 63, 63,
       MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget, kDefault, kDefault, kDefault, 0),
    CU("420v.p0.fmt540.63x63.rt", k420v, 0, (MTLPixelFormat)540, 63, 63,
       MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget, kDefault, kDefault, kDefault, 0),
    CU("420v.p0.fmt541.63x63.rt", k420v, 0, (MTLPixelFormat)541, 63, 63,
       MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget, kDefault, kDefault, kDefault, 0),
    CU("420v.p0.fmt542.63x63.rt", k420v, 0, (MTLPixelFormat)542, 63, 63,
       MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget, kDefault, kDefault, kDefault, 0),
    CU("420v.p0.fmt543.63x63.rt", k420v, 0, (MTLPixelFormat)543, 63, 63,
       MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget, kDefault, kDefault, kDefault, 0),
    CU("420v.p0.fmt544.63x63.rt", k420v, 0, (MTLPixelFormat)544, 63, 63,
       MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget, kDefault, kDefault, kDefault, 0),
    CU("420v.p0.fmt545.63x63.rt", k420v, 0, (MTLPixelFormat)545, 63, 63,
       MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget, kDefault, kDefault, kDefault, 0),
    CU("420v.p0.fmt550.63x63.rt", k420v, 0, (MTLPixelFormat)550, 63, 63,
       MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget, kDefault, kDefault, kDefault, 0),
    CU("420v.p0.fmt551.63x63.rt", k420v, 0, (MTLPixelFormat)551, 63, 63,
       MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget, kDefault, kDefault, kDefault, 0),
    // The fifth round (11-13): which odd sizes and usages an Apple GPU answers nil for, by format.
    CU("420v.p0.fmt500.64x63.rt", k420v, 0, (MTLPixelFormat)500, 64, 63,
       MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget, kDefault, kDefault, kDefault, 0),
    CU("420v.p0.fmt500.63x64.rt", k420v, 0, (MTLPixelFormat)500, 63, 64,
       MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget, kDefault, kDefault, kDefault, 0),
    CU("420v.p0.fmt500.63x63.write", k420v, 0, (MTLPixelFormat)500, 63, 63,
       MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite, kDefault, kDefault, kDefault, 0),
    CU("420v.p0.fmt500.63x63.read", k420v, 0, (MTLPixelFormat)500, 63, 63,
       MTLTextureUsageShaderRead, kDefault, kDefault, kDefault, 0),
    CU("420v.p0.fmt502.64x63.rt", k420v, 0, (MTLPixelFormat)502, 64, 63,
       MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget, kDefault, kDefault, kDefault, 0),
    CU("420v.p0.fmt502.63x64.rt", k420v, 0, (MTLPixelFormat)502, 63, 64,
       MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget, kDefault, kDefault, kDefault, 0),
    CU("420v.p0.fmt502.63x63.write", k420v, 0, (MTLPixelFormat)502, 63, 63,
       MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite, kDefault, kDefault, kDefault, 0),
    CU("420v.p0.fmt502.63x63.read", k420v, 0, (MTLPixelFormat)502, 63, 63,
       MTLTextureUsageShaderRead, kDefault, kDefault, kDefault, 0),
    CU("420v.p0.fmt505.64x63.rt", k420v, 0, (MTLPixelFormat)505, 64, 63,
       MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget, kDefault, kDefault, kDefault, 0),
    CU("420v.p0.fmt505.63x64.rt", k420v, 0, (MTLPixelFormat)505, 63, 64,
       MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget, kDefault, kDefault, kDefault, 0),
    CU("420v.p0.fmt505.63x63.write", k420v, 0, (MTLPixelFormat)505, 63, 63,
       MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite, kDefault, kDefault, kDefault, 0),
    CU("420v.p0.fmt505.63x63.read", k420v, 0, (MTLPixelFormat)505, 63, 63,
       MTLTextureUsageShaderRead, kDefault, kDefault, kDefault, 0),
    CU("420v.p0.fmt506.64x63.rt", k420v, 0, (MTLPixelFormat)506, 64, 63,
       MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget, kDefault, kDefault, kDefault, 0),
    CU("420v.p0.fmt506.63x64.rt", k420v, 0, (MTLPixelFormat)506, 63, 64,
       MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget, kDefault, kDefault, kDefault, 0),
    CU("420v.p0.fmt506.63x63.write", k420v, 0, (MTLPixelFormat)506, 63, 63,
       MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite, kDefault, kDefault, kDefault, 0),
    CU("420v.p0.fmt506.63x63.read", k420v, 0, (MTLPixelFormat)506, 63, 63,
       MTLTextureUsageShaderRead, kDefault, kDefault, kDefault, 0),
    CU("420v.p0.fmt508.64x63.rt", k420v, 0, (MTLPixelFormat)508, 64, 63,
       MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget, kDefault, kDefault, kDefault, 0),
    CU("420v.p0.fmt508.63x64.rt", k420v, 0, (MTLPixelFormat)508, 63, 64,
       MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget, kDefault, kDefault, kDefault, 0),
    CU("420v.p0.fmt508.63x63.write", k420v, 0, (MTLPixelFormat)508, 63, 63,
       MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite, kDefault, kDefault, kDefault, 0),
    CU("420v.p0.fmt508.63x63.read", k420v, 0, (MTLPixelFormat)508, 63, 63,
       MTLTextureUsageShaderRead, kDefault, kDefault, kDefault, 0),
    CU("420v.p0.fmt509.64x63.rt", k420v, 0, (MTLPixelFormat)509, 64, 63,
       MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget, kDefault, kDefault, kDefault, 0),
    CU("420v.p0.fmt509.63x64.rt", k420v, 0, (MTLPixelFormat)509, 63, 64,
       MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget, kDefault, kDefault, kDefault, 0),
    CU("420v.p0.fmt509.63x63.write", k420v, 0, (MTLPixelFormat)509, 63, 63,
       MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite, kDefault, kDefault, kDefault, 0),
    CU("420v.p0.fmt509.63x63.read", k420v, 0, (MTLPixelFormat)509, 63, 63,
       MTLTextureUsageShaderRead, kDefault, kDefault, kDefault, 0),
    CU("420v.p0.fmt520.64x63.rt", k420v, 0, (MTLPixelFormat)520, 64, 63,
       MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget, kDefault, kDefault, kDefault, 0),
    CU("420v.p0.fmt520.63x64.rt", k420v, 0, (MTLPixelFormat)520, 63, 64,
       MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget, kDefault, kDefault, kDefault, 0),
    CU("420v.p0.fmt520.63x63.write", k420v, 0, (MTLPixelFormat)520, 63, 63,
       MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite, kDefault, kDefault, kDefault, 0),
    CU("420v.p0.fmt520.63x63.read", k420v, 0, (MTLPixelFormat)520, 63, 63,
       MTLTextureUsageShaderRead, kDefault, kDefault, kDefault, 0),
    CU("420v.p0.fmt522.64x63.rt", k420v, 0, (MTLPixelFormat)522, 64, 63,
       MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget, kDefault, kDefault, kDefault, 0),
    CU("420v.p0.fmt522.63x64.rt", k420v, 0, (MTLPixelFormat)522, 63, 64,
       MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget, kDefault, kDefault, kDefault, 0),
    CU("420v.p0.fmt522.63x63.write", k420v, 0, (MTLPixelFormat)522, 63, 63,
       MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite, kDefault, kDefault, kDefault, 0),
    CU("420v.p0.fmt522.63x63.read", k420v, 0, (MTLPixelFormat)522, 63, 63,
       MTLTextureUsageShaderRead, kDefault, kDefault, kDefault, 0),
    CU("420v.p0.fmt525.64x63.rt", k420v, 0, (MTLPixelFormat)525, 64, 63,
       MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget, kDefault, kDefault, kDefault, 0),
    CU("420v.p0.fmt525.63x64.rt", k420v, 0, (MTLPixelFormat)525, 63, 64,
       MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget, kDefault, kDefault, kDefault, 0),
    CU("420v.p0.fmt525.63x63.write", k420v, 0, (MTLPixelFormat)525, 63, 63,
       MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite, kDefault, kDefault, kDefault, 0),
    CU("420v.p0.fmt525.63x63.read", k420v, 0, (MTLPixelFormat)525, 63, 63,
       MTLTextureUsageShaderRead, kDefault, kDefault, kDefault, 0),
    CU("420v.p0.fmt526.64x63.rt", k420v, 0, (MTLPixelFormat)526, 64, 63,
       MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget, kDefault, kDefault, kDefault, 0),
    CU("420v.p0.fmt526.63x64.rt", k420v, 0, (MTLPixelFormat)526, 63, 64,
       MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget, kDefault, kDefault, kDefault, 0),
    CU("420v.p0.fmt526.63x63.write", k420v, 0, (MTLPixelFormat)526, 63, 63,
       MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite, kDefault, kDefault, kDefault, 0),
    CU("420v.p0.fmt526.63x63.read", k420v, 0, (MTLPixelFormat)526, 63, 63,
       MTLTextureUsageShaderRead, kDefault, kDefault, kDefault, 0),
    CU("420v.p0.fmt528.64x63.rt", k420v, 0, (MTLPixelFormat)528, 64, 63,
       MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget, kDefault, kDefault, kDefault, 0),
    CU("420v.p0.fmt528.63x64.rt", k420v, 0, (MTLPixelFormat)528, 63, 64,
       MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget, kDefault, kDefault, kDefault, 0),
    CU("420v.p0.fmt528.63x63.write", k420v, 0, (MTLPixelFormat)528, 63, 63,
       MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite, kDefault, kDefault, kDefault, 0),
    CU("420v.p0.fmt528.63x63.read", k420v, 0, (MTLPixelFormat)528, 63, 63,
       MTLTextureUsageShaderRead, kDefault, kDefault, kDefault, 0),
    CU("420v.p0.fmt529.64x63.rt", k420v, 0, (MTLPixelFormat)529, 64, 63,
       MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget, kDefault, kDefault, kDefault, 0),
    CU("420v.p0.fmt529.63x64.rt", k420v, 0, (MTLPixelFormat)529, 63, 64,
       MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget, kDefault, kDefault, kDefault, 0),
    CU("420v.p0.fmt529.63x63.write", k420v, 0, (MTLPixelFormat)529, 63, 63,
       MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite, kDefault, kDefault, kDefault, 0),
    CU("420v.p0.fmt529.63x63.read", k420v, 0, (MTLPixelFormat)529, 63, 63,
       MTLTextureUsageShaderRead, kDefault, kDefault, kDefault, 0),
    CU("420v.p0.fmt540.64x63.rt", k420v, 0, (MTLPixelFormat)540, 64, 63,
       MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget, kDefault, kDefault, kDefault, 0),
    CU("420v.p0.fmt540.63x64.rt", k420v, 0, (MTLPixelFormat)540, 63, 64,
       MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget, kDefault, kDefault, kDefault, 0),
    CU("420v.p0.fmt540.63x63.write", k420v, 0, (MTLPixelFormat)540, 63, 63,
       MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite, kDefault, kDefault, kDefault, 0),
    CU("420v.p0.fmt540.63x63.read", k420v, 0, (MTLPixelFormat)540, 63, 63,
       MTLTextureUsageShaderRead, kDefault, kDefault, kDefault, 0),
    CU("420v.p0.fmt541.64x63.rt", k420v, 0, (MTLPixelFormat)541, 64, 63,
       MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget, kDefault, kDefault, kDefault, 0),
    CU("420v.p0.fmt541.63x64.rt", k420v, 0, (MTLPixelFormat)541, 63, 64,
       MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget, kDefault, kDefault, kDefault, 0),
    CU("420v.p0.fmt541.63x63.write", k420v, 0, (MTLPixelFormat)541, 63, 63,
       MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite, kDefault, kDefault, kDefault, 0),
    CU("420v.p0.fmt541.63x63.read", k420v, 0, (MTLPixelFormat)541, 63, 63,
       MTLTextureUsageShaderRead, kDefault, kDefault, kDefault, 0),
    CU("420v.p0.fmt543.64x63.rt", k420v, 0, (MTLPixelFormat)543, 64, 63,
       MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget, kDefault, kDefault, kDefault, 0),
    CU("420v.p0.fmt543.63x64.rt", k420v, 0, (MTLPixelFormat)543, 63, 64,
       MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget, kDefault, kDefault, kDefault, 0),
    CU("420v.p0.fmt543.63x63.write", k420v, 0, (MTLPixelFormat)543, 63, 63,
       MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite, kDefault, kDefault, kDefault, 0),
    CU("420v.p0.fmt543.63x63.read", k420v, 0, (MTLPixelFormat)543, 63, 63,
       MTLTextureUsageShaderRead, kDefault, kDefault, kDefault, 0),
    CU("420v.p0.fmt544.64x63.rt", k420v, 0, (MTLPixelFormat)544, 64, 63,
       MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget, kDefault, kDefault, kDefault, 0),
    CU("420v.p0.fmt544.63x64.rt", k420v, 0, (MTLPixelFormat)544, 63, 64,
       MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget, kDefault, kDefault, kDefault, 0),
    CU("420v.p0.fmt544.63x63.write", k420v, 0, (MTLPixelFormat)544, 63, 63,
       MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite, kDefault, kDefault, kDefault, 0),
    CU("420v.p0.fmt544.63x63.read", k420v, 0, (MTLPixelFormat)544, 63, 63,
       MTLTextureUsageShaderRead, kDefault, kDefault, kDefault, 0),
    CU("420v.p0.fmt503.63x63.write", k420v, 0, (MTLPixelFormat)503, 63, 63,
       MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite, kDefault, kDefault, kDefault, 0),
    CU("420v.p0.fmt507.63x63.write", k420v, 0, (MTLPixelFormat)507, 63, 63,
       MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite, kDefault, kDefault, kDefault, 0),
    CU("420v.p0.fmt510.63x63.write", k420v, 0, (MTLPixelFormat)510, 63, 63,
       MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite, kDefault, kDefault, kDefault, 0),
    CU("420v.p0.fmt523.63x63.write", k420v, 0, (MTLPixelFormat)523, 63, 63,
       MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite, kDefault, kDefault, kDefault, 0),
    CU("420v.p0.fmt527.63x63.write", k420v, 0, (MTLPixelFormat)527, 63, 63,
       MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite, kDefault, kDefault, kDefault, 0),
    CU("420v.p0.fmt530.63x63.write", k420v, 0, (MTLPixelFormat)530, 63, 63,
       MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite, kDefault, kDefault, kDefault, 0),
    CU("420v.p0.fmt542.63x63.write", k420v, 0, (MTLPixelFormat)542, 63, 63,
       MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite, kDefault, kDefault, kDefault, 0),
    CU("420v.p0.fmt545.63x63.write", k420v, 0, (MTLPixelFormat)545, 63, 63,
       MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite, kDefault, kDefault, kDefault, 0),
    CU("420v.p0.fmt550.63x63.write", k420v, 0, (MTLPixelFormat)550, 63, 63,
       MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite, kDefault, kDefault, kDefault, 0),
    CU("420v.p0.fmt551.63x63.write", k420v, 0, (MTLPixelFormat)551, 63, 63,
       MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite, kDefault, kDefault, kDefault, 0),
    // The sixth round (11-13): which usage bits a texture over a surface keeps, by format, by the
    // pixel format view bit and by the surface, beside the same descriptor's texture over no
    // surface and a view of the texture over the surface.
    CM("bgra.atomic.r32s", kBGRA, 0, MTLPixelFormatR32Sint, 64, 64,
       MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite | 0x20, kDefault, kOverSurface),
    CM("bgra.atomic.only", kBGRA, 0, MTLPixelFormatR32Uint, 64, 64, 0x20, kDefault, kOverSurface),
    CM("bgra.atomic.read", kBGRA, 0, MTLPixelFormatR32Uint, 64, 64,
       MTLTextureUsageShaderRead | 0x20, kDefault, kOverSurface),
    CM("bgra.atomic.view", kBGRA, 0, MTLPixelFormatR32Uint, 64, 64,
       MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite | MTLTextureUsagePixelFormatView |
           0x20, kDefault, kOverSurface),
    CM("bgra.atomic.rt", kBGRA, 0, MTLPixelFormatR32Uint, 64, 64,
       MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite | MTLTextureUsageRenderTarget | 0x20,
           kDefault, kOverSurface),
    CM("bgra.atomic.all", kBGRA, 0, MTLPixelFormatR32Uint, 64, 64,
       MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite | MTLTextureUsageRenderTarget |
           MTLTextureUsagePixelFormatView | 0x20, kDefault, kOverSurface),
    CM("bgra.atomic.bgra8", kBGRA, 0, BGRA8, 64, 64,
       MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite | 0x20, kDefault, kOverSurface),
    CM("bgra.atomic.bgra8.view", kBGRA, 0, BGRA8, 64, 64,
       MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite | MTLTextureUsagePixelFormatView |
           0x20, kDefault, kOverSurface),
    CM("bgra.atomic.r32f", kBGRA, 0, MTLPixelFormatR32Float, 64, 64,
       MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite | 0x20, kDefault, kOverSurface),
    CM("bgra.atomic.rgba8u", kBGRA, 0, MTLPixelFormatRGBA8Uint, 64, 64,
       MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite | 0x20, kDefault, kOverSurface),
    CM("bgra.atomic.shared", kBGRA, 0, MTLPixelFormatR32Uint, 64, 64,
       MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite | 0x20, MTLStorageModeShared,
           kOverSurface),
    CM("bgra.atomic.private", kBGRA, 0, MTLPixelFormatR32Uint, 64, 64,
       MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite | 0x20, MTLStorageModePrivate,
           kOverSurface),
    CM("bgra1.atomic", kBGRA1, 0, MTLPixelFormatR32Uint, 64, 64,
       MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite | 0x20, kDefault, kOverSurface),
    CM("bgrapad.atomic", kBGRAPad, 0, MTLPixelFormatR32Uint, 64, 64,
       MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite | 0x20, kDefault, kOverSurface),
    CM("rgha.atomic.rg32u", kRGhA, 0, MTLPixelFormatRG32Uint, 64, 64,
       MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite | 0x20, kDefault, kOverSurface),
    CM("rgha.atomic.r32u.128x64", kRGhA, 0, MTLPixelFormatR32Uint, 128, 64,
       MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite | 0x20, kDefault, kOverSurface),
    CM("l008.atomic.r8u", kL008, 0, MTLPixelFormatR8Uint, 64, 64,
       MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite | 0x20, kDefault, kOverSurface),
    CM("420v.p0.atomic.r32u", k420v, 0, MTLPixelFormatR32Uint, 16, 64,
       MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite | 0x20, kDefault, kOverSurface),
    CM("420v.p1.atomic.r32u", k420v, 1, MTLPixelFormatR32Uint, 16, 32,
       MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite | 0x20, kDefault, kOverSurface),
    CM("420v.p0.ycbcr.atomic", k420v, 0, (MTLPixelFormat)500, 64, 64,
       MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite | 0x20, kDefault, kOverSurface),
    CM("bgra.view.r32u", kBGRA, 0, MTLPixelFormatR32Uint, 64, 64,
       MTLTextureUsageShaderRead | MTLTextureUsagePixelFormatView, kDefault, kOverSurface),
    CM("bgra.write.r32u", kBGRA, 0, MTLPixelFormatR32Uint, 64, 64,
       MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite, kDefault, kOverSurface),
    CM("bgra.view.only", kBGRA, 0, BGRA8, 64, 64,
       MTLTextureUsagePixelFormatView, kDefault, kOverSurface),
    CM("420v.p0.ycbcr.view", k420v, 0, (MTLPixelFormat)500, 64, 64,
       MTLTextureUsageShaderRead | MTLTextureUsagePixelFormatView, kDefault, kOverSurface),
    CM("420v.p0.view", k420v, 0, MTLPixelFormatR8Unorm, 64, 64,
       MTLTextureUsageShaderRead | MTLTextureUsagePixelFormatView, kDefault, kOverSurface),
    CM("bgra.usage.0x8", kBGRA, 0, BGRA8, 64, 64,
       MTLTextureUsageShaderRead | 0x8, kDefault, kOverSurface),
    CM("bgra.usage.0x40", kBGRA, 0, BGRA8, 64, 64,
       MTLTextureUsageShaderRead | 0x40, kDefault, kOverSurface),
    CM("bgra.usage.0x100", kBGRA, 0, BGRA8, 64, 64,
       MTLTextureUsageShaderRead | 0x100, kDefault, kOverSurface),
    CM("bgra.usage.0x4000", kBGRA, 0, BGRA8, 64, 64,
       MTLTextureUsageShaderRead | 0x4000, kDefault, kOverSurface),
    CM("bgra.usage.0x10000", kBGRA, 0, BGRA8, 64, 64,
       MTLTextureUsageShaderRead | 0x10000, kDefault, kOverSurface),
    CM("nosurface.atomic.r32u", kBGRA, 0, MTLPixelFormatR32Uint, 64, 64,
       MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite | 0x20, kDefault, kNoSurface),
    CM("nosurface.atomic.r32u.private", kBGRA, 0, MTLPixelFormatR32Uint, 64, 64,
       MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite | 0x20, MTLStorageModePrivate,
           kNoSurface),
    CM("nosurface.atomic.bgra8", kBGRA, 0, BGRA8, 64, 64,
       MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite | 0x20, kDefault, kNoSurface),
    CM("nosurface.atomic.rg32u", kBGRA, 0, MTLPixelFormatRG32Uint, 64, 64,
       MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite | 0x20, kDefault, kNoSurface),
    CM("nosurface.usage.0x40", kBGRA, 0, BGRA8, 64, 64,
       MTLTextureUsageShaderRead | 0x40, kDefault, kNoSurface),
    CM("viewof.atomic.r32u", kBGRA, 0, MTLPixelFormatR32Uint, 64, 64,
       MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite | MTLTextureUsagePixelFormatView |
           0x20, kDefault, kViewOf),
    CM("viewof.atomic.r32u.noview", kBGRA, 0, MTLPixelFormatR32Uint, 64, 64,
       MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite | 0x20, kDefault, kViewOf),
    CM("viewof.all.bgra8", kBGRA, 0, BGRA8, 64, 64,
       MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite | MTLTextureUsageRenderTarget |
           MTLTextureUsagePixelFormatView, kDefault, kViewOf),
    // The seventh round (11-13): every other usage bit Metal names none for, one at a time and
    // three at once, and the atomic bit with more formats over no surface.
    CM("bgra.usage.0x8.only", kBGRA, 0, BGRA8, 64, 64, 0x8, kDefault, kOverSurface),
    CM("bgra.usage.0x28.r32u", kBGRA, 0, MTLPixelFormatR32Uint, 64, 64,
       MTLTextureUsageShaderRead | 0x28, kDefault, kOverSurface),
    CM("bgra.usage.0x200", kBGRA, 0, BGRA8, 64, 64,
       MTLTextureUsageShaderRead | 0x200, kDefault, kOverSurface),
    CM("bgra.usage.0x400", kBGRA, 0, BGRA8, 64, 64,
       MTLTextureUsageShaderRead | 0x400, kDefault, kOverSurface),
    CM("bgra.usage.0x800", kBGRA, 0, BGRA8, 64, 64,
       MTLTextureUsageShaderRead | 0x800, kDefault, kOverSurface),
    CM("bgra.usage.0x1000", kBGRA, 0, BGRA8, 64, 64,
       MTLTextureUsageShaderRead | 0x1000, kDefault, kOverSurface),
    CM("bgra.usage.0x2000", kBGRA, 0, BGRA8, 64, 64,
       MTLTextureUsageShaderRead | 0x2000, kDefault, kOverSurface),
    CM("bgra.usage.0x8000", kBGRA, 0, BGRA8, 64, 64,
       MTLTextureUsageShaderRead | 0x8000, kDefault, kOverSurface),
    CM("bgra.usage.0x20000", kBGRA, 0, BGRA8, 64, 64,
       MTLTextureUsageShaderRead | 0x20000, kDefault, kOverSurface),
    CM("bgra.usage.0x40000", kBGRA, 0, BGRA8, 64, 64,
       MTLTextureUsageShaderRead | 0x40000, kDefault, kOverSurface),
    CM("bgra.usage.0x80000", kBGRA, 0, BGRA8, 64, 64,
       MTLTextureUsageShaderRead | 0x80000, kDefault, kOverSurface),
    CM("bgra.usage.0x100000", kBGRA, 0, BGRA8, 64, 64,
       MTLTextureUsageShaderRead | 0x100000, kDefault, kOverSurface),
    CM("bgra.usage.0x1000000", kBGRA, 0, BGRA8, 64, 64,
       MTLTextureUsageShaderRead | 0x1000000, kDefault, kOverSurface),
    CM("bgra.usage.0x80000000", kBGRA, 0, BGRA8, 64, 64,
       MTLTextureUsageShaderRead | 0x80000000, kDefault, kOverSurface),
    CM("bgra.usage.0x1c0", kBGRA, 0, BGRA8, 64, 64,
       MTLTextureUsageShaderRead | 0x1c0, kDefault, kOverSurface),
    CM("bgra.usage.0x1c0.atomic.bgra8", kBGRA, 0, BGRA8, 64, 64,
       MTLTextureUsageShaderRead | 0x1c0 | 0x20, kDefault, kOverSurface),
    CM("nosurface.atomic.rg32s", kBGRA, 0, MTLPixelFormatRG32Sint, 64, 64,
       MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite | 0x20, kDefault, kNoSurface),
    CM("nosurface.atomic.r16u", kBGRA, 0, MTLPixelFormatR16Uint, 64, 64,
       MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite | 0x20, kDefault, kNoSurface),
    CM("nosurface.atomic.rgba32u", kBGRA, 0, MTLPixelFormatRGBA32Uint, 64, 64,
       MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite | 0x20, kDefault, kNoSurface),
    CM("nosurface.atomic.r32s", kBGRA, 0, MTLPixelFormatR32Sint, 64, 64,
       MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite | 0x20, kDefault, kNoSurface),
    CM("nosurface.usage.0x8", kBGRA, 0, BGRA8, 64, 64,
       MTLTextureUsageShaderRead | 0x8, kDefault, kNoSurface),
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
	id<MTLTexture> t = c->mode == kNoSurface
	                       ? [d newTextureWithDescriptor:td]
	                       : [d newTextureWithDescriptor:td iosurface:s plane:(NSUInteger)c->plane];
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
		if (c->mode == kViewOf) {
			id<MTLTexture> v = [t newTextureViewWithPixelFormat:t.pixelFormat];
			if (v == nil)
				printf("case %d %s => view nil\n", n, c->name);
			else
				printf("case %d %s => view: fmt %lu usage 0x%lx\n", n, c->name,
				       (unsigned long)v.pixelFormat, (unsigned long)v.usage);
		}
	}
	fflush(stdout);
	t = nil;
	CFRelease(s);
	if (keep != NULL)
		CVPixelBufferRelease(keep);
	return 0;
}

// The aliasing cases (the header).
enum AliasWay { kWideThenNarrow, kNarrowThenWide };
enum AliasWriter { kBlitWrite, kClearWrite };
struct Alias {
	const char *name;
	enum AliasWay way;
	enum AliasWriter writer;
	int fenced; // 1: Untracked textures, the writer updates a fence the reader waits for
};
static const struct Alias kAliases[] = {
    {"alias.wide-blit.narrow.tracked", kWideThenNarrow, kBlitWrite, 0},
    {"alias.wide-blit.narrow.fence", kWideThenNarrow, kBlitWrite, 1},
    {"alias.wide-clear.narrow.tracked", kWideThenNarrow, kClearWrite, 0},
    {"alias.wide-clear.narrow.fence", kWideThenNarrow, kClearWrite, 1},
    {"alias.narrow-blit.wide.tracked", kNarrowThenWide, kBlitWrite, 0},
    {"alias.narrow-blit.wide.fence", kNarrowThenWide, kBlitWrite, 1},
    {"alias.narrow-clear.wide.tracked", kNarrowThenWide, kClearWrite, 0},
    {"alias.narrow-clear.wide.fence", kNarrowThenWide, kClearWrite, 1},
};
enum { kAliasCount = sizeof kAliases / sizeof kAliases[0], kAliasRuns = 16, kAliasRow = 128 };

// The byte a blit writer writes at byte `x` of row `y` of the texture it writes.
static uint8_t aliasPattern(int x, int y)
{
	return (uint8_t)(0x80 | ((x * 3 + y * 5) & 0x7f));
}

// The byte the writer of `a` leaves at byte `x` of row `y` of the surface, where it writes it;
// -1 where it writes none (past the narrow texture's 64 bytes of a row).
static int aliasWritten(const struct Alias *a, int x, int y)
{
	int wide = a->way == kWideThenNarrow;
	if (!wide && x >= kSize)
		return -1;
	if (a->writer == kBlitWrite)
		return aliasPattern(x, y);
	// A clear to red 128/255 and green 64/255: RG8's two bytes, R8's one.
	return wide && (x & 1) ? 0x40 : 0x80;
}

static id<MTLTexture> aliasTexture(id<MTLDevice> d, IOSurfaceRef s, MTLPixelFormat format,
                                   int fenced)
{
	MTLTextureDescriptor *td = [MTLTextureDescriptor new];
	td.textureType = MTLTextureType2D;
	td.pixelFormat = format;
	td.width = kSize;
	td.height = kSize;
	td.usage = MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget;
	if (fenced)
		td.hazardTrackingMode = MTLHazardTrackingModeUntracked;
	return [d newTextureWithDescriptor:td iosurface:s plane:0];
}

static int runAlias(id<MTLDevice> d, int n)
{
	const struct Alias *a = &kAliases[n];
	IOSurfaceRef s = flatSurface('L008', 1, kAliasRow);
	if (s == NULL) {
		printf("alias %d %s => no surface\n", n, a->name);
		return 1;
	}
	char surface[400];
	surfaceText(s, surface, sizeof surface);
	int wideFirst = a->way == kWideThenNarrow;
	printf("alias %d %s | %s | %s through %s, read through %s, %s\n", n, a->name, surface,
	       a->writer == kBlitWrite ? "a blit" : "a clear", wideFirst ? "wide" : "narrow",
	       wideFirst ? "narrow" : "wide", a->fenced ? "Untracked with a fence" : "tracked");
	fflush(stdout);
	if (IOSurfaceGetBytesPerRow(s) != kAliasRow) {
		printf("alias %d %s => the surface's rows are not %d bytes\n", n, a->name, kAliasRow);
		CFRelease(s);
		return 1;
	}
	id<MTLTexture> wide = aliasTexture(d, s, MTLPixelFormatRG8Unorm, a->fenced);
	id<MTLTexture> narrow = aliasTexture(d, s, MTLPixelFormatR8Unorm, a->fenced);
	if (wide == nil || narrow == nil) {
		printf("alias %d %s => nil (wide %d, narrow %d)\n", n, a->name, wide != nil,
		       narrow != nil);
		CFRelease(s);
		return 1;
	}
	id<MTLTexture> writer = wideFirst ? wide : narrow;
	id<MTLTexture> reader = wideFirst ? narrow : wide;
	NSUInteger writeRow = wideFirst ? kAliasRow : kSize;
	NSUInteger readRow = wideFirst ? kSize : kAliasRow;
	id<MTLBuffer> source = [d newBufferWithLength:writeRow * kSize
	                                      options:MTLResourceStorageModeShared];
	uint8_t *src = source.contents;
	for (int y = 0; y < kSize; y++)
		for (NSUInteger x = 0; x < writeRow; x++)
			src[y * writeRow + x] = aliasPattern((int)x, y);
	id<MTLBuffer> out = [d newBufferWithLength:readRow * kSize options:MTLResourceStorageModeShared];
	id<MTLCommandQueue> q = [d newCommandQueue];
	id<MTLFence> fence = a->fenced ? [d newFence] : nil;
	int all = 0, none = 0, some = 0, after = 0, afterBeside = 0, afterBesideOld = 0;
	for (int run = 0; run < kAliasRuns; run++) {
		IOSurfaceLock(s, 0, NULL);
		memset(IOSurfaceGetBaseAddress(s), 0x11, IOSurfaceGetAllocSize(s));
		IOSurfaceUnlock(s, 0, NULL);
		memset(out.contents, 0, out.length);
		id<MTLCommandBuffer> cb = [q commandBuffer];
		if (a->writer == kBlitWrite) {
			id<MTLBlitCommandEncoder> b = [cb blitCommandEncoder];
			[b copyFromBuffer:source
			           sourceOffset:0
			      sourceBytesPerRow:writeRow
			    sourceBytesPerImage:writeRow * kSize
			             sourceSize:MTLSizeMake(kSize, kSize, 1)
			              toTexture:writer
			       destinationSlice:0
			       destinationLevel:0
			      destinationOrigin:MTLOriginMake(0, 0, 0)];
			if (fence != nil)
				[b updateFence:fence];
			[b endEncoding];
		} else {
			MTLRenderPassDescriptor *rp = [MTLRenderPassDescriptor renderPassDescriptor];
			rp.colorAttachments[0].texture = writer;
			rp.colorAttachments[0].loadAction = MTLLoadActionClear;
			rp.colorAttachments[0].storeAction = MTLStoreActionStore;
			rp.colorAttachments[0].clearColor = MTLClearColorMake(128.0 / 255, 64.0 / 255, 0, 1);
			id<MTLRenderCommandEncoder> r = [cb renderCommandEncoderWithDescriptor:rp];
			if (fence != nil)
				[r updateFence:fence afterStages:MTLRenderStageFragment];
			[r endEncoding];
		}
		id<MTLBlitCommandEncoder> b = [cb blitCommandEncoder];
		if (fence != nil)
			[b waitForFence:fence];
		[b copyFromTexture:reader
		                 sourceSlice:0
		                 sourceLevel:0
		                sourceOrigin:MTLOriginMake(0, 0, 0)
		                  sourceSize:MTLSizeMake(kSize, kSize, 1)
		                    toBuffer:out
		           destinationOffset:0
		      destinationBytesPerRow:readRow
		    destinationBytesPerImage:readRow * kSize];
		[b endEncoding];
		[cb commit];
		[cb waitUntilCompleted];
		if (cb.status != MTLCommandBufferStatusCompleted) {
			printf("alias %d %s => run %d: status %ld, error %s\n", n, a->name, run,
			       (long)cb.status, cb.error.description.UTF8String);
			continue;
		}
		// Of the bytes the writer wrote that the reader's texels hold: how many read written.
		const uint8_t *got = out.contents;
		int want = 0, seen = 0;
		for (int y = 0; y < kSize; y++)
			for (NSUInteger x = 0; x < readRow; x++) {
				int w = aliasWritten(a, (int)x, y);
				if (w < 0)
					continue;
				want++;
				seen += got[y * readRow + x] == w;
			}
		all += seen == want;
		none += seen == 0;
		some += seen != want && seen != 0;
		// The surface after: the plane's texels (the first 64 bytes of each row) and the bytes
		// past them apart, each counted where it holds every byte the writer wrote there.
		IOSurfaceLock(s, kIOSurfaceLockReadOnly, NULL);
		const uint8_t *pages = IOSurfaceGetBaseAddress(s);
		int texels = 1, beside = 1, besideOld = 1;
		for (int y = 0; y < kSize; y++)
			for (int x = 0; x < kAliasRow; x++) {
				int w = aliasWritten(a, x, y);
				uint8_t b = pages[y * kAliasRow + x];
				if (x >= kSize)
					besideOld &= b == 0x11;
				if (w < 0)
					continue;
				if (x < kSize)
					texels &= b == w;
				else
					beside &= b == w;
			}
		IOSurfaceUnlock(s, kIOSurfaceLockReadOnly, NULL);
		after += texels;
		afterBeside += wideFirst && beside;
		afterBesideOld += besideOld;
	}
	printf("alias %d %s => runs %d: all %d, none %d, some %d; surface after: texels all %d, "
	       "bytes past them written %d, still 0x11 %d\n",
	       n, a->name, kAliasRuns, all, none, some, after, afterBeside, afterBesideOld);
	fflush(stdout);
	CFRelease(s);
	return 0;
}

int main(int argc, char **argv)
{
	setvbuf(stdout, NULL, _IOLBF, 0);
	if (argc < 2) {
		fprintf(stderr, "usage: %s DEVICE [--case N | --alias N | --list]\n", argv[0]);
		return 2;
	}
	if (argc > 2 && !strcmp(argv[2], "--list")) {
		for (int n = 0; n < kCaseCount; n++)
			printf("%d %s\n", n, kCases[n].name);
		return 0;
	}
	if (argc > 3 && !strcmp(argv[2], "--alias")) {
		dup2(STDOUT_FILENO, STDERR_FILENO);
		int n = atoi(argv[3]);
		if (n < 0 || n >= kAliasCount)
			return 2;
		return runAlias(oracle_device(argv[1]), n);
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
	for (int n = 0; n < kAliasCount; n++) {
		char index[16];
		snprintf(index, sizeof index, "%d", n);
		char *args[] = {argv[0], argv[1], "--alias", index, NULL};
		pid_t pid;
		if (posix_spawn(&pid, argv[0], NULL, NULL, args, environ))
			return 1;
		int st;
		waitpid(pid, &st, 0);
		if (WIFSIGNALED(st))
			printf("=> alias %d died on signal %d\n", n, WTERMSIG(st));
		else if (WEXITSTATUS(st))
			printf("=> alias %d exited %d\n", n, WEXITSTATUS(st));
		fflush(stdout);
	}
	return 0;
}
