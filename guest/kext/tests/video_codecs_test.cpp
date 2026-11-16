// video_codecs_test: what the accelerator publishes of the host's video: its video driver, HEVC
// decode and H.264 encode (RgfxVideoCodecs.hpp). Built with each architecture's rules (the
// Makefile's video_codecs_test, and video_codecs_arm64_test with RGFX_GUEST_ARM64=1): the x86-64
// kext publishes what the host's caps list, the arm64e kext nothing.
#include <cstdio>
#include <cstdlib>

#include "RgfxVideoCodecs.hpp"

static int checks;
#define CHECK(c)                                                                                   \
	do {                                                                                       \
		checks++;                                                                          \
		if (!(c)) {                                                                        \
			std::fprintf(stderr, "%s:%d: CHECK(%s) failed\n", __FILE__, __LINE__, #c); \
			std::abort();                                                              \
		}                                                                                  \
	} while (0)

// Caps of H.264 decode, then HEVC decode of `profiles` up to `level`.
static RgfxVideoCaps caps(uint32_t profiles, uint32_t level)
{
	RgfxVideoCaps c = {};
	c.version = RGFX_VIDEO_CAPS_VERSION;
	c.codec_count = 2;
	c.codecs[0].codec = RGFX_VIDEO_CODEC_H264;
	c.codecs[0].direction = RGFX_VIDEO_DECODE;
	c.codecs[0].profiles = RGFX_VIDEO_H264_HIGH;
	c.codecs[0].max_level = 52;
	c.codecs[1].codec = RGFX_VIDEO_CODEC_HEVC;
	c.codecs[1].direction = RGFX_VIDEO_DECODE;
	c.codecs[1].profiles = profiles;
	c.codecs[1].max_level = level;
	return c;
}

// The x86-64 kext's rules: what the caps list.
static void x86Rules()
{
	CHECK(kRgfxGuestVideoDriver);
	RgfxHevcDecode d;
	// Main and Main 10: Main Still Picture rides with Main.
	CHECK(rgfxHevcDecode(caps(RGFX_VIDEO_HEVC_MAIN | RGFX_VIDEO_HEVC_MAIN_10, 186), &d));
	CHECK(d.count == 3 && d.profiles[0] == 1 && d.profiles[1] == 2 && d.profiles[2] == 3);
	CHECK(d.maxLevel == 186);
	// Main alone.
	CHECK(rgfxHevcDecode(caps(RGFX_VIDEO_HEVC_MAIN, 153), &d));
	CHECK(d.count == 2 && d.profiles[0] == 1 && d.profiles[1] == 3 && d.maxLevel == 153);
	// Main 10 alone.
	CHECK(rgfxHevcDecode(caps(RGFX_VIDEO_HEVC_MAIN_10, 153), &d));
	CHECK(d.count == 1 && d.profiles[0] == 2);
	// No profile, an encode only, H.264 alone, an older host's caps: nothing.
	CHECK(!rgfxHevcDecode(caps(0, 153), &d));
	RgfxVideoCaps encode = caps(RGFX_VIDEO_HEVC_MAIN, 153);
	encode.codecs[1].direction = RGFX_VIDEO_ENCODE;
	CHECK(!rgfxHevcDecode(encode, &d));
	RgfxVideoCaps h264 = caps(RGFX_VIDEO_HEVC_MAIN, 153);
	h264.codec_count = 1;
	CHECK(!rgfxHevcDecode(h264, &d));
	RgfxVideoCaps none = {};
	CHECK(!rgfxHevcDecode(none, &d));
	// A count past the array is read no further than it.
	RgfxVideoCaps past = caps(RGFX_VIDEO_HEVC_MAIN, 153);
	past.codec_count = 1000;
	CHECK(rgfxHevcDecode(past, &d));
	// The video driver is named where the caps list a decoder: H.264 and HEVC, H.264 alone,
	// HEVC alone.
	CHECK(rgfxDecodesVideo(caps(RGFX_VIDEO_HEVC_MAIN, 153)));
	CHECK(rgfxDecodesVideo(h264));
	RgfxVideoCaps hevcOnly = caps(RGFX_VIDEO_HEVC_MAIN, 153);
	hevcOnly.codecs[0] = hevcOnly.codecs[1];
	hevcOnly.codec_count = 1;
	CHECK(rgfxDecodesVideo(hevcOnly));
	// Not where they list none: a host with no video arm writes no caps, an older host's are
	// of no version, an entry of no profile decodes nothing, and an encoder is no decoder.
	CHECK(!rgfxDecodesVideo(none));
	RgfxVideoCaps old = caps(RGFX_VIDEO_HEVC_MAIN, 153);
	old.version = 0;
	CHECK(!rgfxDecodesVideo(old));
	RgfxVideoCaps noProfile = caps(0, 153);
	noProfile.codecs[0].profiles = 0;
	CHECK(!rgfxDecodesVideo(noProfile));
	RgfxVideoCaps encoders = caps(RGFX_VIDEO_HEVC_MAIN, 153);
	encoders.codecs[0].direction = RGFX_VIDEO_ENCODE;
	encoders.codecs[1].direction = RGFX_VIDEO_ENCODE;
	CHECK(!rgfxDecodesVideo(encoders));
	// An entry past the count is not read.
	RgfxVideoCaps counted = caps(RGFX_VIDEO_HEVC_MAIN, 153);
	counted.codec_count = 0;
	CHECK(!rgfxDecodesVideo(counted));
	// H.264 encode's ratings are published where the caps list an H.264 encoder with a profile
	// and a rate control: after the decoders, or alone.
	RgfxVideoCaps encoder = caps(RGFX_VIDEO_HEVC_MAIN, 153);
	encoder.codec_count = 3;
	encoder.codecs[2] = encoder.codecs[0];
	encoder.codecs[2].direction = RGFX_VIDEO_ENCODE;
	encoder.encode[2].rate_controls = 1u << RGFX_VIDEO_RATE_CQP;
	CHECK(rgfxH264Encode(encoder));
	RgfxVideoCaps alone = {};
	alone.version = RGFX_VIDEO_CAPS_VERSION;
	alone.codec_count = 1;
	alone.codecs[0] = encoder.codecs[2];
	alone.encode[0] = encoder.encode[2];
	CHECK(rgfxH264Encode(alone));
	// Not where they list none: decoders alone, an encoder of no rate control (the host lists the
	// codec but encodes nothing) or of no profile, an HEVC encoder, an older host's caps, an entry
	// past the count.
	CHECK(!rgfxH264Encode(caps(RGFX_VIDEO_HEVC_MAIN, 153)));
	RgfxVideoCaps noRate = alone;
	noRate.encode[0].rate_controls = 0;
	CHECK(!rgfxH264Encode(noRate));
	RgfxVideoCaps noEncodeProfile = alone;
	noEncodeProfile.codecs[0].profiles = 0;
	CHECK(!rgfxH264Encode(noEncodeProfile));
	RgfxVideoCaps hevcEncoder = alone;
	hevcEncoder.codecs[0].codec = RGFX_VIDEO_CODEC_HEVC;
	CHECK(!rgfxH264Encode(hevcEncoder));
	RgfxVideoCaps oldEncoder = alone;
	oldEncoder.version = 0;
	CHECK(!rgfxH264Encode(oldEncoder));
	RgfxVideoCaps uncounted = alone;
	uncounted.codec_count = 0;
	CHECK(!rgfxH264Encode(uncounted));
}

// The arm64e kext's rules: no video driver, no HEVC decode and no H.264 encode, whatever the caps
// list.
static void arm64Rules()
{
	CHECK(!kRgfxGuestVideoDriver);
	RgfxHevcDecode d = {};
	// H.264 and HEVC Main and Main 10, H.264 alone, HEVC alone: nothing.
	const RgfxVideoCaps both = caps(RGFX_VIDEO_HEVC_MAIN | RGFX_VIDEO_HEVC_MAIN_10, 186);
	CHECK(!rgfxDecodesVideo(both));
	CHECK(!rgfxHevcDecode(both, &d));
	RgfxVideoCaps h264 = both;
	h264.codec_count = 1;
	CHECK(!rgfxDecodesVideo(h264));
	CHECK(!rgfxHevcDecode(h264, &d));
	RgfxVideoCaps hevcOnly = both;
	hevcOnly.codecs[0] = hevcOnly.codecs[1];
	hevcOnly.codec_count = 1;
	CHECK(!rgfxDecodesVideo(hevcOnly));
	CHECK(!rgfxHevcDecode(hevcOnly, &d));
	// An H.264 encoder with a profile and a rate control, after the decoders or alone: no ratings.
	RgfxVideoCaps encoder = both;
	encoder.codec_count = 3;
	encoder.codecs[2] = encoder.codecs[0];
	encoder.codecs[2].direction = RGFX_VIDEO_ENCODE;
	encoder.encode[2].rate_controls = 1u << RGFX_VIDEO_RATE_CQP;
	CHECK(!rgfxH264Encode(encoder));
	RgfxVideoCaps alone = {};
	alone.version = RGFX_VIDEO_CAPS_VERSION;
	alone.codec_count = 1;
	alone.codecs[0] = encoder.codecs[2];
	alone.encode[0] = encoder.encode[2];
	CHECK(!rgfxH264Encode(alone));
	// Nothing written to the answer.
	CHECK(d.count == 0 && d.maxLevel == 0);
}

int main()
{
	if (RGFX_GUEST_ARM64)
		arm64Rules();
	else
		x86Rules();
	std::printf("video_codecs_test (%s): OK (%d checks)\n", RGFX_GUEST_ARM64 ? "arm64" : "x86-64",
	            checks);
	return 0;
}
