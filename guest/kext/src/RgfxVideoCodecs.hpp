/*
 * RgfxVideoCodecs.hpp — what the accelerator publishes of the host's hardware video for AppleGVA
 * (RgfxAccelerator.cpp, publishVideoCodecs), from the video caps the host wrote after the device
 * info (RgfxVideoCaps, RGFX_VIDEO_CAPS_OFFSET).
 *
 * AppleGVA's HEVC decoder serves a GPU whose accelerator carries IOGVAHEVCDecode and, in
 * IOGVAHEVCDecodeCapabilities, the profiles it decodes (VTSupportedProfileArray, VideoToolbox's
 * numbering: 1 Main, 2 Main 10, 3 Main Still Picture) and each one's highest level
 * (VTPerProfileDetails, VTMaxDecodeLevel, as general_level_idc counts it). The accelerator
 * publishes them where the host decodes HEVC, its profiles and level the host's: Main Still
 * Picture where Main is, whose decoder takes its streams. Elsewhere it publishes none, and HEVC
 * decodes in software, as on a GPU without it.
 *
 * Header-only and free of kernel types, so tests/video_codecs_test.cpp checks it on the Mac.
 */

#ifndef RGFX_VIDEO_CODECS_HPP_
#define RGFX_VIDEO_CODECS_HPP_

#include <stdint.h>

#include "reims_graphics_wire.h"

// What the accelerator publishes for HEVC decode.
struct RgfxHevcDecode {
	// VideoToolbox's profile numbers, `count` of them, in ascending order.
	uint32_t profiles[3];
	uint32_t count;
	// The highest level of every profile.
	uint32_t maxLevel;
};

// The HEVC decode `caps` answers, for its accelerator to publish: false where the host's caps
// list none (an older host's, or no hardware for it), or list no profile of it.
static inline bool rgfxHevcDecode(const RgfxVideoCaps &caps, RgfxHevcDecode *out)
{
	if (caps.version < RGFX_VIDEO_CAPS_VERSION)
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

#endif
