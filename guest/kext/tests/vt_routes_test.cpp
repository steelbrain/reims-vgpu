// The native test of RgfxVtRoutes.hpp: which clients a host message goes to, the `-uui` the kext
// sends as a client lets go of a UUID, the runs a message of a client's buffer lies in, and
// whether the caps list a decoder for the VideoToolbox service to serve.
#include "RgfxVtRoutes.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {
int checks = 0;

void check(bool ok, const char *what)
{
	checks++;
	if (!ok) {
		std::fprintf(stderr, "vt_routes_test: FAILED: %s\n", what);
		std::exit(1);
	}
}

// A `vtpv` header of `type` for `uuid`.
void header(uint8_t out[kRgfxVtHeaderBytes], const char (&type)[5], uint8_t uuid)
{
	std::memset(out, 0, kRgfxVtHeaderBytes);
	const uint8_t head[8] = {0, 0, 0, 48, 'v', 't', 'p', 'v'};
	std::memcpy(out, head, sizeof(head));
	std::memcpy(out + 8, type, 4);
	std::memset(out + 16, uuid, kRgfxVtUuidBytes);
}
} // namespace

int main()
{
	static_assert(RGFX_FEATURE_VT_MESSAGES == (1ull << 52), "VT_MESSAGES is bit 52");
	static_assert(RGFX_CMD_VT_MESSAGE == 0x2fa, "VT_MESSAGE is 0x2fa");
	static_assert(RGFX_EVT_VT_STATUS == 0x40d && RGFX_EVT_VT_MESSAGE == 0x40e,
	              "VT_STATUS and VT_MESSAGE are 0x40d and 0x40e");
	static_assert(sizeof(RgfxVtMessage) == RGFX_VT_MESSAGE_RUNS_OFFSET, "runs follow the struct");
	static_assert(sizeof(RgfxVtHostMessage) == RGFX_VT_HOST_MESSAGE_BYTES_OFFSET,
	              "bytes follow the struct");

	// What a host message is: for its UUID, a `Bye!` for all, a `rels` box, or unreadable.
	{
		uint8_t m[kRgfxVtHeaderBytes];
		uint8_t uuid[kRgfxVtUuidBytes] = {};
		header(m, "decs", 0x5a);
		check(rgfxVtHostMessage(m, sizeof(m), uuid) == kRgfxVtForUuid && uuid[0] == 0x5a &&
		          uuid[15] == 0x5a,
		      "a vtpv message goes to its UUID's clients");
		header(m, "Bye!", 0);
		check(rgfxVtHostMessage(m, sizeof(m), uuid) == kRgfxVtByeAll, "a Bye! goes to all");
		check(rgfxVtHostMessage(m, 47, uuid) == kRgfxVtUnreadable, "47 bytes are no header");
		m[3] = 40;
		check(rgfxVtHostMessage(m, sizeof(m), uuid) == kRgfxVtUnreadable,
		      "a header atom under 48 bytes is unreadable");
		const uint8_t rels[16] = {0, 0, 0, 16, 'r', 'e', 'l', 's'};
		check(rgfxVtHostMessage(rels, sizeof(rels), uuid) == kRgfxVtReleases, "a rels box");
	}

	// The `-uui`: Apple's kext's 48 bytes.
	{
		uint8_t uuid[kRgfxVtUuidBytes];
		std::memset(uuid, 0x11, sizeof(uuid));
		uint8_t m[kRgfxVtHeaderBytes];
		std::memset(m, 0xee, sizeof(m));
		rgfxVtForgetMessage(m, uuid);
		uint8_t expected[kRgfxVtHeaderBytes] = {0, 0, 0, 48, 'v', 't', 'p', 'v', '-', 'u', 'u', 'i'};
		std::memset(expected + 16, 0x11, kRgfxVtUuidBytes);
		check(!std::memcmp(m, expected, sizeof(m)), "the -uui is a bare header of the UUID");
	}

	// Routes: each UUID's clients once each, a deregistration of a held UUID says so, and a
	// client's close lets go of each UUID it held, each once.
	{
		static RgfxVtRoutes routes;
		routes.count = 0;
		int a = 0, b = 0;
		uint8_t u1[kRgfxVtUuidBytes], u2[kRgfxVtUuidBytes];
		std::memset(u1, 1, sizeof(u1));
		std::memset(u2, 2, sizeof(u2));
		check(routes.add(u1, &a) && routes.add(u1, &a) && routes.add(u1, &b) && routes.add(u2, &a),
		      "registrations are taken, a repeat too");
		const void *clients[4];
		check(routes.clientsOf(u1, clients, 4) == 2 && clients[0] == &a && clients[1] == &b,
		      "a UUID's clients, each once, in order");
		check(routes.clientsOf(u2, clients, 4) == 1 && clients[0] == &a, "the other UUID's");
		check(routes.remove(u1, &b) && !routes.remove(u1, &b),
		      "a held UUID's deregistration says so, an unheld one's does not");
		check(routes.clientsOf(u1, clients, 4) == 1, "the UUID's other client stays");
		uint8_t held[4][kRgfxVtUuidBytes];
		check(routes.removeClient(&a, held, 4) == 2 && held[0][0] == 1 && held[1][0] == 2,
		      "a close lets go of each UUID once, in order");
		check(routes.count == 0, "no route is left");
		for (uint32_t i = 0; i < kRgfxVtRoutesMax; i++)
			routes.add(u1, &a);
		check(!routes.add(u1, &b), "a full table takes no more");
	}

	// The runs of a message: the first `length` bytes of the buffer's runs.
	{
		const RgfxPageRun buffer[2] = {{0x10000, 0x4000}, {0x40000, 0x8000}};
		RgfxPageRun out[2];
		check(rgfxVtMessageRuns(buffer, 2, 48, out, 2) == 1 && out[0].address == 0x10000 &&
		          out[0].length == 48,
		      "a short message in the first run");
		check(rgfxVtMessageRuns(buffer, 2, 0x5000, out, 2) == 2 && out[1].address == 0x40000 &&
		          out[1].length == 0x1000,
		      "a longer one into the second");
		check(rgfxVtMessageRuns(buffer, 2, 0xc001, out, 2) == 0, "more than the runs hold");
		check(rgfxVtMessageRuns(buffer, 2, 0x5000, out, 1) == 0, "more runs than fit");
	}

	// A host `rels` names the lends it lets go of, each big-endian.
	{
		const uint8_t rels[24] = {0, 0, 0, 24, 'r', 'e', 'l', 's', 0, 0, 0, 0, 0, 0, 0, 7,
		                          0, 0, 0, 1, 0, 0, 0, 2};
		uint64_t out[2] = {};
		check(rgfxVtReleasedLends(rels, sizeof(rels), out, 2) == 2 && out[0] == 7 &&
		          out[1] == 0x100000002ull,
		      "a rels of two lends");
		check(rgfxVtReleasedLends(rels, sizeof(rels), out, 1) == 1 && out[0] == 7,
		      "at most what fits");
		check(rgfxVtReleasedLends(rels, 20, out, 2) == 0, "a box its size does not span");
		check(rgfxVtReleasedLends(rels, 8, out, 2) == 0, "a box of no lend");
	}

	// A VT_MESSAGE that lends surfaces: its struct, its runs, then each surface's record and runs.
	{
		const RgfxPageRun runs[1] = {{0x10000, 0x200}};
		const RgfxPageRun first[2] = {{0x80000, 0x1000}, {0x90000, 0x2000}};
		const RgfxPageRun second[1] = {{0xa0000, 0x3000}};
		RgfxVtLent lent[2] = {};
		lent[0].record.lend = 3;
		lent[0].record.run_count = 2;
		lent[0].runs = first;
		lent[1].record.lend = 4;
		lent[1].record.run_count = 1;
		lent[1].runs = second;
		uint8_t out[512];
		const uint32_t size = rgfxVtPackMessage(0x200, runs, 1, lent, 2, out, sizeof(out));
		const uint32_t expected = RGFX_VT_MESSAGE_RUNS_OFFSET + 16 + 2 * sizeof(RgfxVtSurface) +
		                          3 * 16;
		RgfxVtMessage head;
		std::memcpy(&head, out, sizeof(head));
		check(size == expected && head.length == 0x200 && head.run_count == 1 &&
		          head.surface_count == 2 && head.surfaces[0] == 3 && head.surfaces[1] == 4 &&
		          head.surfaces[2] == 0,
		      "the struct names the lends");
		RgfxVtSurface record;
		RgfxPageRun run;
		const uint32_t second_at = RGFX_VT_MESSAGE_RUNS_OFFSET + 16 + sizeof(RgfxVtSurface) + 32;
		std::memcpy(&record, out + second_at, sizeof(record));
		std::memcpy(&run, out + second_at + sizeof(record), sizeof(run));
		check(record.lend == 4 && run.address == 0xa0000 && run.length == 0x3000,
		      "the second surface's record and run follow the first's");
		check(rgfxVtPackMessage(0x200, runs, 1, lent, 2, out, expected - 1) == 0,
		      "a message that does not fit");
		RgfxVtLent six[6] = {};
		check(rgfxVtPackMessage(0x200, runs, 1, six, 6, out, sizeof(out)) == 0,
		      "more surfaces than a message names");
	}

	// The service serves a host whose caps list a decode entry of H.264 or HEVC with a profile.
	{
		RgfxVideoCaps caps = {};
		check(!rgfxVtServesDecoders(caps), "an older host's caps serve nothing");
		caps.version = RGFX_VIDEO_CAPS_VERSION;
		caps.codec_count = 1;
		caps.codecs[0].codec = RGFX_VIDEO_CODEC_H264;
		caps.codecs[0].direction = RGFX_VIDEO_ENCODE;
		caps.codecs[0].profiles = RGFX_VIDEO_H264_HIGH;
		check(!rgfxVtServesDecoders(caps), "an encoder alone serves nothing");
		caps.codecs[0].direction = RGFX_VIDEO_DECODE;
		check(rgfxVtServesDecoders(caps), "an H.264 decoder is served");
		caps.codecs[0].profiles = 0;
		check(!rgfxVtServesDecoders(caps), "a decoder of no profile is not");
		caps.codecs[1] = caps.codecs[0];
		caps.codecs[1].codec = RGFX_VIDEO_CODEC_HEVC;
		caps.codecs[1].profiles = RGFX_VIDEO_HEVC_MAIN;
		check(!rgfxVtServesDecoders(caps), "an entry past the count is not read");
		caps.codec_count = 2;
		check(rgfxVtServesDecoders(caps), "an HEVC decoder is served");
	}

	// The service serves macOS 14 and later.
	check(!rgfxVtServesRelease(22), "none on macOS 13");
	check(rgfxVtServesRelease(23) && rgfxVtServesRelease(24) && rgfxVtServesRelease(25) &&
	          rgfxVtServesRelease(26),
	      "macOS 14, 15, 26 and 27 are served");

	std::printf("vt_routes_test: %d checks passed\n", checks);
	return 0;
}
