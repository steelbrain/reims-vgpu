/*
 * RgfxVideoCodecs.hpp — what the accelerator publishes of the host's hardware video for AppleGVA
 * (RgfxAccelerator.cpp, publishVideoCodecs), from the video caps the host wrote after the device
 * info (RgfxVideoCaps, RGFX_VIDEO_CAPS_OFFSET).
 *
 * AppleGVA takes a GPU's accelerator for hardware video where it names a video driver
 * (IODVDBundleName, IOGVACodec, IOVARendererID): VideoToolbox then answers that the GPU decodes
 * H.264 (VTIsHardwareDecodeSupported) and lists AppleGVA's H.264 encoder as hardware
 * (VTCopyVideoEncoderList), whatever the driver's own capability block holds. The accelerator
 * names one only where the host's caps list a decoder (rgfxDecodesVideo). Elsewhere, a host with
 * no video arm, it names none, and VideoToolbox answers no hardware decoder or encoder for any
 * codec, as on a Mac without one (measured on a macOS 26.7.2 guest on a host with no video arm,
 * 2026-11-16: with the three keys, VTIsHardwareDecodeSupported(avc1) answered true while a session
 * requiring a hardware decoder failed -12913; without them it answered false, that session failed
 * -12906, kVTCouldNotFindVideoDecoderErr, and the encoder list held no hardware entry).
 *
 * AppleGVA's HEVC decoder serves a GPU whose accelerator carries IOGVAHEVCDecode and, in
 * IOGVAHEVCDecodeCapabilities, the profiles it decodes (VTSupportedProfileArray, VideoToolbox's
 * numbering: 1 Main, 2 Main 10, 3 Main Still Picture) and each one's highest level
 * (VTPerProfileDetails, VTMaxDecodeLevel, as general_level_idc counts it). The accelerator
 * publishes them where the host decodes HEVC, its profiles and level the host's: Main Still
 * Picture where Main is, whose decoder takes its streams. Elsewhere it publishes none, and HEVC
 * decodes in software, as on a GPU without it.
 *
 * AppleGVA's H.264 encoder ranks a GPU by its accelerator's IOGVAH264EncodeCapabilities: the
 * ratings VideoToolbox weighs encoders by (VTRating, VTQualityRating). The accelerator publishes
 * them where the host encodes H.264, as an AMD Mac's accelerator does (350 and 50 on a Radeon Pro
 * 5300M's, macOS 26.7.1), and none elsewhere.
 *
 * AppleGVA's vendor-driver route, which our bundle's video driver serves, is x86-64's. On an arm64
 * guest the keys bring no session to it: with them, on a host that decodes H.264 and HEVC,
 * VideoToolbox answered VTIsHardwareDecodeSupported true for HEVC, yet a session requiring a
 * hardware HEVC decoder failed -17692 and one requiring H.264 -12906, no encoder was hardware
 * and one requiring a hardware encoder failed -12903, and every session that allowed hardware
 * decoded in software (measured on an arm64 macOS 14.6.1 guest on a VA-API host, 2026-11-16). So
 * the arm64e kext (RGFX_GUEST_ARM64) publishes none of them, whatever the host's caps list
 * (kRgfxGuestVideoDriver), and VideoToolbox answers no hardware decoder or encoder for any codec,
 * as on a host with no video arm.
 *
 * Header-only and free of kernel types, so tests/video_codecs_test.cpp checks it on the Mac, as
 * each kext builds it (video_codecs_test, and video_codecs_arm64_test with RGFX_GUEST_ARM64 set).
 */

#ifndef RGFX_VIDEO_CODECS_HPP_
#define RGFX_VIDEO_CODECS_HPP_

#include <stdint.h>

#include "reims_graphics_wire.h"

#ifndef RGFX_GUEST_ARM64
#define RGFX_GUEST_ARM64 0
#endif

// Whether the guest's AppleGVA takes our bundle as a GPU's video driver: the x86-64 guest's.
// The arm64e kext publishes no video key.
constexpr bool kRgfxGuestVideoDriver = !RGFX_GUEST_ARM64;

// What the accelerator publishes for HEVC decode.
struct RgfxHevcDecode {
	// VideoToolbox's profile numbers, `count` of them, in ascending order.
	uint32_t profiles[3];
	uint32_t count;
	// The highest level of every profile.
	uint32_t maxLevel;
};

// Whether `caps` lists a decoder: a decode entry of H.264 or HEVC with a profile, for its
// accelerator to name AppleGVA its video driver. False for an older host's caps, for a host
// with no video arm, which writes none, and on an arm64 guest.
static inline bool rgfxDecodesVideo(const RgfxVideoCaps &caps)
{
	if (!kRgfxGuestVideoDriver || caps.version < RGFX_VIDEO_CAPS_VERSION)
		return false;
	const uint32_t n = caps.codec_count < RGFX_VIDEO_CAPS_CODECS ? caps.codec_count
	                                                             : RGFX_VIDEO_CAPS_CODECS;
	for (uint32_t i = 0; i < n; i++) {
		const RgfxVideoCodecCaps &c = caps.codecs[i];
		if (c.direction == RGFX_VIDEO_DECODE && c.profiles != 0 &&
		    (c.codec == RGFX_VIDEO_CODEC_H264 || c.codec == RGFX_VIDEO_CODEC_HEVC))
			return true;
	}
	return false;
}

// The HEVC decode `caps` answers, for its accelerator to publish: false where the host's caps
// list none (an older host's, or no hardware for it), or list no profile of it, and on an arm64
// guest.
static inline bool rgfxHevcDecode(const RgfxVideoCaps &caps, RgfxHevcDecode *out)
{
	if (!kRgfxGuestVideoDriver || caps.version < RGFX_VIDEO_CAPS_VERSION)
		return false;
	const uint32_t n = caps.codec_count < RGFX_VIDEO_CAPS_CODECS ? caps.codec_count
	                                                             : RGFX_VIDEO_CAPS_CODECS;
	for (uint32_t i = 0; i < n; i++) {
		const RgfxVideoCodecCaps &c = caps.codecs[i];
		if (c.codec != RGFX_VIDEO_CODEC_HEVC || c.direction != RGFX_VIDEO_DECODE)
			continue;
		RgfxHevcDecode d = {};
		if (c.profiles & RGFX_VIDEO_HEVC_MAIN)
			d.profiles[d.count++] = 1;
		if (c.profiles & RGFX_VIDEO_HEVC_MAIN_10)
			d.profiles[d.count++] = 2;
		if (c.profiles & RGFX_VIDEO_HEVC_MAIN)
			d.profiles[d.count++] = 3;
		if (d.count == 0)
			return false;
		d.maxLevel = c.max_level;
		*out = d;
		return true;
	}
	return false;
}

// The ratings the accelerator publishes for H.264 encode: an AMD Mac's (IOGVAH264EncodeCapabilities).
constexpr uint32_t kRgfxH264EncodeRating = 350;
constexpr uint32_t kRgfxH264EncodeQualityRating = 50;

// Whether `caps` lists an H.264 encoder, for its accelerator to publish its ratings: an encode
// entry with a profile and a rate control (RgfxVideoEncodeCaps, whose rate controls 0 say none).
// False for an older host's caps, for a host with no video arm or no encoder, and on an arm64
// guest.
static inline bool rgfxH264Encode(const RgfxVideoCaps &caps)
{
	if (!kRgfxGuestVideoDriver || caps.version < RGFX_VIDEO_CAPS_VERSION)
		return false;
	const uint32_t n = caps.codec_count < RGFX_VIDEO_CAPS_CODECS ? caps.codec_count
	                                                             : RGFX_VIDEO_CAPS_CODECS;
	for (uint32_t i = 0; i < n; i++) {
		const RgfxVideoCodecCaps &c = caps.codecs[i];
		if (c.codec == RGFX_VIDEO_CODEC_H264 && c.direction == RGFX_VIDEO_ENCODE)
			return c.profiles != 0 && caps.encode[i].rate_controls != 0;
	}
	return false;
}

// The ratings the accelerator's HEVC encoder publishes (IOGVAHEVCEncodeCapabilities): as for
// H.264 encode, an AMD Mac's.
constexpr uint32_t kRgfxHevcEncodeRating = 350;
constexpr uint32_t kRgfxHevcEncodeQualityRating = 50;

// What the host's HEVC encoder offers the guest: Main and Main 10, up to its level.
struct RgfxHevcEncode {
	uint32_t maxLevel;
};

// Whether `caps` lists an HEVC encoder the guest's HEVC encoder plug-in can drive, and its level
// in `out`: an encode entry of Main from NV12 and Main 10 from P010, with a constant QP among its
// rate controls, since the plug-in's rate control picks each frame's QP and the driver codes at
// it (guest/bundle/src/RgfxVideoDriver.cpp), and that codes a frame in place of the last one
// (RGFX_VIDEO_ENCODE_CAP_IN_PLACE), since the plug-in codes a frame again at a higher QP where it
// came out over its data rate limit. Both profiles, as the plug-in's session may be of either
// and the driver learns which only from its first frame's source, past its create. False for an
// older host's caps, for a host with no video arm or no such encoder, and on an arm64 guest.
static inline bool rgfxHevcEncode(const RgfxVideoCaps &caps, RgfxHevcEncode *out)
{
	if (!kRgfxGuestVideoDriver || caps.version < RGFX_VIDEO_CAPS_VERSION)
		return false;
	const uint32_t n = caps.codec_count < RGFX_VIDEO_CAPS_CODECS ? caps.codec_count
	                                                             : RGFX_VIDEO_CAPS_CODECS;
	for (uint32_t i = 0; i < n; i++) {
		const RgfxVideoCodecCaps &c = caps.codecs[i];
		if (c.codec != RGFX_VIDEO_CODEC_HEVC || c.direction != RGFX_VIDEO_ENCODE)
			continue;
		const uint32_t profiles = RGFX_VIDEO_HEVC_MAIN | RGFX_VIDEO_HEVC_MAIN_10;
		const uint32_t formats = RGFX_VIDEO_FORMAT_NV12 | RGFX_VIDEO_FORMAT_P010;
		if ((c.profiles & profiles) != profiles || (c.formats & formats) != formats ||
		    !(caps.encode[i].rate_controls & (1u << RGFX_VIDEO_RATE_CQP)) ||
		    !(caps.encode[i].flags & RGFX_VIDEO_ENCODE_CAP_IN_PLACE))
			return false;
		out->maxLevel = c.max_level;
		return true;
	}
	return false;
}

#endif
