// RgfxGLSurfaces: the rules of a GL window's surface (user client type 0x20) and of the GL
// drawables over it (type 7), which an app's OpenGL contexts drive on our accelerator.
// Header-only and free of the kernel's headers, so its native test (tests/gl_surfaces_test.cpp)
// builds it as the kext does; RgfxGLSurfaceUserClient.cpp and RgfxGLDrawableUserClient.cpp hold
// the surfaces and run the rules.
//
// The protocol is IOGPU's, which macOS's GL renderer over Metal (AppleMetalOpenGLRenderer) and
// SkyLight's client speak to an IOGPU accelerator. Measured on a macOS 26.6.1 guest (the bundle's
// newGLDrawable through Metal's IOGPU base; the kext logging every call on types 7 and 0x20), for
// window-backed and layer-backed views alike:
//   - the app opens type 7, the drawable (IOGPUGLDrawableCreate), and sets its swap rect (selector
//     2: four zero scalars) and swap interval (selector 3: 1, 0);
//   - the app's SLSBindSurface opens type 0x20 for the window's surface (IOAccelCreateSurface:
//     mode bit 0x4000 selects type 0x20), and calls SetIDMode (selector 7) with the surface's id
//     (0x100b) and mode 0x4024, then SetShape (selector 9) with options 2, framebuffer index 0 and
//     a region of no rectangles, its bounds the view's size;
//   - the drawable's set surface (selector 1) names that id, in mode 0xc00 and colour depth 4,
//     and its lookups (selector 5) of indexes 0 and 1 hand the app the two IOSurfaces it draws
//     into, after which its frames make no call of the kext's: WindowServer composites the window
//     from them (screen dumps of both views showed the frames drawn), and opens no user client
//     of the accelerator for it;
//   - a resize sets the shape again, and then the surface on the drawable.
// The rules below are IOGPU's, read from IOGPUFamily's x86_64 code in macOS 26.7.1's KDK
// (IOGPUSurfaceMTL and IOGPUGLDrawableUserClient), with what the kext serves of them:
//   - a surface takes its id once (another id answers kIOReturnBadArgument), and a mode with bits
//     of 0xff8033c0 (after 0x4000) is refused as a bad argument, one with both bits of 0xc00 or
//     without 0x20 (windowed) as an error, and so is a mode that changes bits 0x7f8c3f while a
//     drawable is attached. Modes of bits 0x700000 (IOGPU's buffers shared between slots) are not
//     served: kIOReturnUnsupported;
//   - its shape is the region's bounds, at most the device's largest surface each way; a size of
//     0 is 1. Shape options of 0x3000, a region under 12 bytes or of a length its rectangle count
//     does not give, and a framebuffer index other than 0 are refused;
//   - its buffers are eleven slots, each an IOSurface made in the surface's task as the slots its
//     drawables ask for need it, named "IOGPUSurfaceMTL sid %08x buffer %d": colour slots 0-3 of
//     the shape's size, their bytes per element and pixel format the colour depth's (mode & 0xf,
//     rgfxGLDepthOf), depth and stencil slots 4-6 of 4 bytes, and slots 7-9 of 1 byte, these
//     without a pixel format and each side the next power of two of the shape's. Slot 10 is
//     never made. A slot is made again when the shape's size or the colour depth changes;
//   - a drawable's set surface finds the surface by id among every task's (the first that holds
//     it), attaches to it, and asks for the slots its mode gives (rgfxGLDrawableRequestBits: 3,
//     colour slots 0 and 1, for mode 0xc00). Id 0 detaches it. A drawable of any other face or
//     level than 0, or of a colour depth other than the surface's, is refused. Several drawables
//     may share a surface; the surface makes what any of them asks for;
//   - a lookup's index names a slot (rgfxGLSlotOfIndex), and answers the slot's IOSurface as a
//     port name in the drawable's task.
// Not served, each answering kIOReturnUnsupported: the drawable's scaled sources (a set surface
// of fixedSource) and shared events (selector 4), and every surface method but SetIDMode and
// SetShape (locks, scale, flush, surface_control, which WindowServer's own surfaces of a GL
// window would use).
//
// Our own code.
#ifndef RGFX_GL_SURFACES_HPP_
#define RGFX_GL_SURFACES_HPP_

#include <stdint.h>

// The user-client types, IOGPU's numbers.
constexpr uint32_t kRgfxGLDrawableUserClientType = 7;
constexpr uint32_t kRgfxGLSurfaceUserClientType = 0x20;

// What the rules answer, IOKit's IOReturn numbers (IOKit/IOReturn.h).
constexpr int32_t kRgfxGLSuccess = 0;
constexpr int32_t kRgfxGLError = static_cast<int32_t>(0xe00002bc);       // kIOReturnError
constexpr int32_t kRgfxGLBadArgument = static_cast<int32_t>(0xe00002c2); // kIOReturnBadArgument
constexpr int32_t kRgfxGLUnsupported = static_cast<int32_t>(0xe00002c7); // kIOReturnUnsupported

// A surface's slots, and the masks of each kind.
constexpr uint32_t kRgfxGLSlots = 11;
constexpr uint32_t kRgfxGLColourSlots = 0xf;
constexpr uint32_t kRgfxGLDepthSlots = 0x70;
constexpr uint32_t kRgfxGLByteSlots = 0x380;

// A surface's id before SetIDMode.
constexpr uint32_t kRgfxGLNoId = 0xffffffe0;

// The surface's mode bits the rules read.
constexpr uint32_t kRgfxGLModeSelectsSurface = 0x4000; // IOAccelCreateSurface's type 0x20
constexpr uint32_t kRgfxGLModeRefused = 0xff8033c0;
constexpr uint32_t kRgfxGLModeBuffers = 0xc00;
constexpr uint32_t kRgfxGLModeWindowed = 0x20;
constexpr uint32_t kRgfxGLModeShared = 0x700000;
constexpr uint64_t kRgfxGLModeKept = 0x7f8c3f;
// A drawable's mode bits refused at its set surface, and those its surface's mode never takes
// (IOGPUGLDrawableUserClient::contextModeBits).
constexpr uint64_t kRgfxGLDrawableModeRefused = 0x7f803f;
constexpr uint64_t kRgfxGLContextModeBits = 0x300807fc0ull;

// The largest surface each way: Metal's largest 2D texture on our device.
constexpr uint16_t kRgfxGLMaxSurfaceSize = 16384;

// A colour depth's buffers: bytes per element and IOSurface pixel format; 0 where IOGPU has none.
struct RgfxGLDepth {
	uint32_t bytes;
	uint32_t format;
};

inline RgfxGLDepth rgfxGLDepthOf(uint64_t mode)
{
	switch (mode & 0xf) {
	case 3:
		return {2, 0x35353531}; // '5551'
	case 4:
		return {4, 0x42475241}; // 'BGRA'
	case 11:
		return {8, 0x6c363472}; // 'l64r'
	case 12:
		return {8, 0x52476841}; // 'RGhA'
	case 13:
		return {16, 0x52476641}; // 'RGfA'
	case 15:
		return {4, 0x6c313072}; // 'l10r'
	default:
		return {0, 0};
	}
}

// A window surface's state, under the accelerator's GL lock.
struct RgfxGLSurface {
	uint32_t id = kRgfxGLNoId;
	uint64_t mode = 0;
	// SetShape has given the shape.
	bool shaped = false;
	uint16_t width = 0;
	uint16_t height = 0;
	// The slots its drawables ask for, the colour slots kept since the last reset, and the slots
	// made at the last reshape with the mode and size they were made for.
	uint32_t requested = 0;
	uint32_t kept = 0;
	uint32_t made = 0;
	uint64_t madeMode = 0;
	uint16_t madeWidth = 0;
	uint16_t madeHeight = 0;
};

// SetIDMode(id, mode) of surface `s`, which `drawables` drawables have attached. On success the
// caller resets the request bits, prunes and reshapes (rgfxGLResetRequests, rgfxGLPrune,
// rgfxGLReshape), as IOGPU does.
inline int32_t rgfxGLSetIDMode(RgfxGLSurface &s, uint32_t id, uint32_t mode, uint32_t drawables)
{
	uint64_t m = mode & ~kRgfxGLModeSelectsSurface;
	if (m & kRgfxGLModeRefused)
		return kRgfxGLBadArgument;
	if ((m & kRgfxGLModeBuffers) == kRgfxGLModeBuffers)
		return kRgfxGLError;
	if (m & kRgfxGLModeShared)
		return kRgfxGLUnsupported;
	if (drawables) {
		if ((s.mode & kRgfxGLModeKept) != m)
			return kRgfxGLError;
		m |= s.mode & 0xff8073c0;
	}
	if (!(m & kRgfxGLModeWindowed))
		return kRgfxGLError;
	if (s.id != kRgfxGLNoId && s.id != id)
		return kRgfxGLBadArgument;
	s.mode = m;
	s.id = id;
	s.kept = 0;
	return kRgfxGLSuccess;
}

// SetShape(options, framebuffer index) of surface `s` with the region of `length` bytes at
// `region` (IOAccelDeviceRegion: a u32 rectangle count, the bounds, then the rectangles, each
// four i16s: x, y, w, h).
inline int32_t rgfxGLSetShape(RgfxGLSurface &s, uint32_t options, uint32_t framebufferIndex,
                              const uint8_t *region, uint64_t length)
{
	if (!region || length < 12)
		return kRgfxGLBadArgument;
	if (options & 0x3000)
		return kRgfxGLUnsupported;
	if (framebufferIndex != 0)
		return kRgfxGLError;
	const uint32_t rects = static_cast<uint32_t>(region[0]) | static_cast<uint32_t>(region[1]) << 8 |
	                       static_cast<uint32_t>(region[2]) << 16 | static_cast<uint32_t>(region[3]) << 24;
	if (length != 12 + 8 * static_cast<uint64_t>(rects))
		return kRgfxGLError;
	int16_t w = static_cast<int16_t>(region[8] | region[9] << 8);
	int16_t h = static_cast<int16_t>(region[10] | region[11] << 8);
	if (w < 0 || h < 0 || w > kRgfxGLMaxSurfaceSize || h > kRgfxGLMaxSurfaceSize)
		return kRgfxGLError;
	if (w == 0 || h == 0)
		w = h = 1;
	s.shaped = true;
	s.width = static_cast<uint16_t>(w);
	s.height = static_cast<uint16_t>(h);
	return kRgfxGLSuccess;
}

// The request bits start again from the drawables' (IOGPU's reset_req_bits): `drawableRequests`,
// the OR of the attached drawables' request bits.
inline void rgfxGLResetRequests(RgfxGLSurface &s, uint32_t drawableRequests)
{
	s.kept = 0;
	s.requested = (s.mode & 0x400) ? 3u : (s.requested & 3u);
	s.requested |= drawableRequests;
}

// The slots no longer asked for go (IOGPU's prune_buffers): of `existing`, the slots that hold a
// buffer, the mask the caller releases.
inline uint32_t rgfxGLPrune(RgfxGLSurface &s, uint32_t existing)
{
	uint32_t req = s.requested;
	if (s.mode & kRgfxGLModeWindowed) {
		const uint32_t kept = s.kept;
		s.kept = (req & 3) | kept;
		req |= kept;
		if (s.mode & 0x8000)
			req |= 1;
	} else {
		req |= 1;
	}
	s.requested = req;
	const uint32_t gone = ~req & existing & ((1u << kRgfxGLSlots) - 1);
	s.made &= ~gone;
	return gone;
}

// The slots to make again for the shape and mode (IOGPU's update_shape): the mask the caller
// releases and makes anew, by rgfxGLBufferOf; 0 before the shape is set.
inline uint32_t rgfxGLReshape(RgfxGLSurface &s)
{
	if (!s.shaped)
		return 0;
	const uint64_t diff = s.mode ^ s.madeMode;
	uint32_t remake = s.requested & ~s.made;
	if (diff & 0x7f000f)
		remake |= 3;
	if (s.width != s.madeWidth || s.height != s.madeHeight || (diff & 0x7f0000))
		remake |= s.requested;
	remake &= s.requested;
	s.madeMode = s.mode;
	s.made = s.requested;
	s.madeWidth = s.width;
	s.madeHeight = s.height;
	return remake;
}

// What slot `slot` of a surface of `mode` and `width` x `height` is made as (IOGPU's
// shapeSurface): false where the slot is never made.
struct RgfxGLBuffer {
	uint32_t width;
	uint32_t height;
	uint32_t bytesPerElement;
	bool hasFormat;
	uint32_t format;
};

inline uint32_t rgfxGLPowerOfTwo(uint32_t v)
{
	if (v < 2)
		return 1;
	return 1u << (32 - __builtin_clz(v - 1));
}

inline bool rgfxGLBufferOf(uint32_t slot, uint64_t mode, uint16_t width, uint16_t height,
                           RgfxGLBuffer &out)
{
	if (slot >= kRgfxGLSlots)
		return false;
	const uint32_t bit = 1u << slot;
	if (bit & kRgfxGLColourSlots) {
		const RgfxGLDepth depth = rgfxGLDepthOf(mode);
		out = {width, height, depth.bytes, true, depth.format};
		return true;
	}
	if (bit & (kRgfxGLDepthSlots | kRgfxGLByteSlots)) {
		out = {rgfxGLPowerOfTwo(width), rgfxGLPowerOfTwo(height),
		       (bit & kRgfxGLDepthSlots) ? 4u : 1u, false, 0};
		return true;
	}
	return false;
}

// A drawable's set surface: its input, 0x30 bytes (IOGPUGLDrawableClientSetSurfaceData).
struct RgfxGLSetSurface {
	uint64_t mode;
	uint32_t surfaceId;
	uint32_t colorDepthMode;
	uint32_t face;
	uint32_t level;
	uint32_t isVolatile;
	uint32_t fixedSource;
	uint32_t scaleOptions;
	uint32_t scaledWidth;
	uint32_t scaledHeight;
	uint32_t field_2c;
};
static_assert(sizeof(RgfxGLSetSurface) == 0x30, "the set surface input");

// Its answer, 0x28 bytes (IOGPUGLDrawableClientGetConfigStatus): the surface's mode, the
// drawable's size and the surface's, and two of IOGPU's own (its accelerator's, 0 here).
struct RgfxGLConfigStatus {
	uint64_t mode;
	uint32_t width;
	uint32_t height;
	uint32_t surfaceWidth;
	uint32_t surfaceHeight;
	uint16_t field_18;
	uint8_t field_1a[6];
	uint64_t field_20;
};
static_assert(sizeof(RgfxGLConfigStatus) == 0x28, "the set surface answer");

// Whether a drawable's mode may set a surface at all (bits of kRgfxGLDrawableModeRefused are
// refused as a bad argument, before the surface is looked up).
inline bool rgfxGLDrawableModeTaken(uint64_t mode)
{
	return (mode & kRgfxGLDrawableModeRefused) == 0;
}

// The mode a drawable of mode `drawableMode` asks for on a surface of mode `surfaceMode`: the
// surface's buffering bits (0xc00) win over the drawable's.
inline uint64_t rgfxGLDrawableModeOn(uint64_t drawableMode, uint64_t surfaceMode)
{
	if (!(surfaceMode & kRgfxGLModeBuffers))
		return drawableMode;
	return (drawableMode & ~kRgfxGLModeKept) | ((surfaceMode & 0x800) ? 0x800 : 0x400);
}

// The surface's mode once a drawable of `drawableMode` (rgfxGLDrawableModeOn's) attaches, the only
// one or with `others` attached besides: alone, its mode's bits outside the context's join the
// surface's kept bits; with others, the surface keeps its mode.
inline uint64_t rgfxGLSurfaceModeWith(uint64_t surfaceMode, uint64_t drawableMode, uint32_t others)
{
	if (others)
		return surfaceMode;
	return (surfaceMode & kRgfxGLModeKept) | (drawableMode & ~kRgfxGLContextModeBits);
}

// The slots a drawable of `mode` (rgfxGLDrawableModeOn's) asks for (IOGPU's set_surface).
inline uint32_t rgfxGLDrawableRequestBits(uint64_t mode)
{
	const uint32_t r = static_cast<uint32_t>(mode);
	uint32_t bits = 0;
	const uint32_t base = ((r >> 9) & 2) | ((r >> 11) & 1) | ((r >> 2) & 0x10) | (r & 0x80) |
	                      ((r >> 2) & 0x400);
	if (r & 0x1cc0)
		bits = base;
	const uint32_t depth = r & 0x300;
	if (depth) {
		bits = ((r >> 8) & 0x20) | base | 4;
		if (r & 0x4000)
			bits |= 0x100;
		if (depth > 0x1ff)
			bits |= ((r >> 7) & 0x40) | ((r & 0x4000) ? 0x208 : 0x8);
	}
	return bits;
}

// The slot a lookup's index names (IOGPU's convertGLDrawableIndexToBufferIndex), or -1.
inline int32_t rgfxGLSlotOfIndex(uint64_t index)
{
	switch (index) {
	case 0:
		return 1;
	case 1:
		return 0;
	case 3:
		return 2;
	case 4:
		return 3;
	case 6:
		return 5;
	case 7:
		return 6;
	case 12:
		return 10;
	case 13:
		return 4;
	case 14:
		return 7;
	case 0x7fffffff:
		return 0;
	default:
		return -1;
	}
}

#endif // RGFX_GL_SURFACES_HPP_
