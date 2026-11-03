// video_codecs_test: what the accelerator publishes of the host's HEVC decode
// (RgfxVideoCodecs.hpp).
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

int main()
{
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
	std::printf("video_codecs_test: OK (%d checks)\n", checks);
	return 0;
}
