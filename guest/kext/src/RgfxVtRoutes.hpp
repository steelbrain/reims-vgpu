/*
 * RgfxVtRoutes.hpp — the rules of the arm64e kext's VideoToolbox service (RgfxVtService.hpp,
 * design.md, "Decided", 17): which of its clients each message of the host goes to, the message
 * the kext sends the host as a client lets go of a session UUID, the runs of a client's message
 * buffer a message lies in, and whether the host's caps list a decoder for the service to serve.
 *
 * Apple's kext of the service, AppleVideoToolboxParavirtualization.kext 15.0.0 (read from the
 * macOS 27.2.1 arm64e VM kernel collection), routes as this does: a host message goes to every
 * client that registered the UUID its `vtpv` header names, once each, and a `Bye!` to every
 * client once for each UUID it registered, that UUID written in; each deregistration of a UUID the
 * client registered, and each UUID a closing client holds, sends the host a `-uui` of it.
 *
 * A message lends the host IOSurfaces by record (RgfxVtSurface, rgfxVtPackMessage), each named by
 * a lend of the kext's, which the host's `rels` names as it lets go of the surface
 * (rgfxVtReleasedLends).
 *
 * Header-only and free of kernel types, so tests/vt_routes_test.cpp checks it on the Mac.
 */

#ifndef RGFX_VT_ROUTES_HPP_
#define RGFX_VT_ROUTES_HPP_

#include <stdint.h>
#include <string.h>

#include "reims_graphics_wire.h"

// A session UUID's bytes, and a `vtpv` header's.
constexpr uint32_t kRgfxVtUuidBytes = 16;
constexpr uint32_t kRgfxVtHeaderBytes = 48;
// The capacity of a client's host-message queue, an IOSharedDataQueue: Apple's kext's.
constexpr uint32_t kRgfxVtQueueBytes = 0x8014;
// The (UUID, client) pairs the service routes at once, the host messages a client keeps while
// its queue is full, and the surfaces the service holds lent: ours, past anything a process's
// sessions hold (a decoder holds a frame's buffer, lent, until the host emits it).
constexpr uint32_t kRgfxVtRoutesMax = 256;
constexpr uint32_t kRgfxVtPendingMax = 256;
constexpr uint32_t kRgfxVtLendsMax = 64;

// Where a `vtpv` header keeps its message type and its UUID.
constexpr uint32_t kRgfxVtTypeAt = 8;
constexpr uint32_t kRgfxVtUuidAt = 16;

static inline uint32_t rgfxVtBe32(const uint8_t *p)
{
	return static_cast<uint32_t>(p[0]) << 24 | static_cast<uint32_t>(p[1]) << 16 |
	       static_cast<uint32_t>(p[2]) << 8 | p[3];
}

// The four bytes of a FourCC, as the wire spells it, as one big-endian u32.
constexpr uint32_t rgfxVtFourCc(const char (&s)[5])
{
	return static_cast<uint32_t>(static_cast<uint8_t>(s[0])) << 24 |
	       static_cast<uint32_t>(static_cast<uint8_t>(s[1])) << 16 |
	       static_cast<uint32_t>(static_cast<uint8_t>(s[2])) << 8 |
	       static_cast<uint32_t>(static_cast<uint8_t>(s[3]));
}

// What a host message is, for its routing.
enum RgfxVtHostKind : uint8_t {
	// A `vtpv` message for the clients of its UUID.
	kRgfxVtForUuid,
	// A `Bye!`: for every client, once for each UUID it registered.
	kRgfxVtByeAll,
	// A `rels` box: the host lets go of surfaces the kext lent it (rgfxVtReleasedLends).
	kRgfxVtReleases,
	// Neither: too short, or not a `vtpv` atom.
	kRgfxVtUnreadable,
};

// What the `length` bytes of the host's `message` are, and its UUID, in `uuid`, for one for a
// UUID.
static inline RgfxVtHostKind rgfxVtHostMessage(const uint8_t *message, uint32_t length,
                                               uint8_t uuid[kRgfxVtUuidBytes])
{
	if (length >= 8 && rgfxVtBe32(message + 4) == rgfxVtFourCc("rels"))
		return kRgfxVtReleases;
	if (length < kRgfxVtHeaderBytes || rgfxVtBe32(message) < kRgfxVtHeaderBytes ||
	    rgfxVtBe32(message + 4) != rgfxVtFourCc("vtpv"))
		return kRgfxVtUnreadable;
	if (rgfxVtBe32(message + kRgfxVtTypeAt) == rgfxVtFourCc("Bye!"))
		return kRgfxVtByeAll;
	memcpy(uuid, message + kRgfxVtUuidAt, kRgfxVtUuidBytes);
	return kRgfxVtForUuid;
}

// The `-uui` the kext sends the host as a client lets go of `uuid`, as Apple's kext writes it:
// a bare 48-byte `vtpv` header with no flags, the UUID, and no reply.
static inline void rgfxVtForgetMessage(uint8_t out[kRgfxVtHeaderBytes],
                                       const uint8_t uuid[kRgfxVtUuidBytes])
{
	static const uint8_t head[16] = {0, 0, 0, 48, 'v', 't', 'p', 'v', '-', 'u', 'u', 'i'};
	memset(out, 0, kRgfxVtHeaderBytes);
	memcpy(out, head, sizeof(head));
	memcpy(out + kRgfxVtUuidAt, uuid, kRgfxVtUuidBytes);
}

// The lends a host `rels` box of `length` bytes names, into `out`, at most `max`: how many; 0 for
// bytes that are no such box. A box is its big-endian size, 8 and 8 a lend, `rels`, then each
// lend big-endian, and spans the message.
static inline uint32_t rgfxVtReleasedLends(const uint8_t *message, uint32_t length, uint64_t *out,
                                           uint32_t max)
{
	if (length < 16 || rgfxVtBe32(message + 4) != rgfxVtFourCc("rels") ||
	    rgfxVtBe32(message) != length || (length - 8) % 8)
		return 0;
	const uint32_t count = (length - 8) / 8;
	uint32_t n = 0;
	for (uint32_t i = 0; i < count && n < max; i++) {
		const uint8_t *at = message + 8 + 8 * i;
		out[n++] = static_cast<uint64_t>(rgfxVtBe32(at)) << 32 | rgfxVtBe32(at + 4);
	}
	return n;
}

// One surface a message lends, as the kext read it of its IOSurface, and its runs.
struct RgfxVtLent {
	RgfxVtSurface record;
	const RgfxPageRun *runs;
};

// The bytes of a VT_MESSAGE of a message of `length` bytes in `runs` (`count` of them), lending
// `lentCount` surfaces, into `out` of `max` bytes: its struct, its runs, then each surface's record
// and runs, `surfaces` naming each surface's lend. Its size, or 0 where it does not fit.
static inline uint32_t rgfxVtPackMessage(uint64_t length, const RgfxPageRun *runs, uint32_t count,
                                         const RgfxVtLent *lent, uint32_t lentCount, uint8_t *out,
                                         uint32_t max)
{
	if (lentCount > RGFX_VT_MESSAGE_SURFACES)
		return 0;
	uint64_t size = RGFX_VT_MESSAGE_RUNS_OFFSET + uint64_t(count) * sizeof(RgfxPageRun);
	for (uint32_t i = 0; i < lentCount; i++)
		size += sizeof(RgfxVtSurface) + uint64_t(lent[i].record.run_count) * sizeof(RgfxPageRun);
	if (size > max)
		return 0;
	RgfxVtMessage head = {};
	head.length = length;
	head.run_count = count;
	head.surface_count = lentCount;
	for (uint32_t i = 0; i < lentCount; i++)
		head.surfaces[i] = lent[i].record.lend;
	uint8_t *at = out;
	memcpy(at, &head, sizeof(head));
	at += sizeof(head);
	memcpy(at, runs, count * sizeof(RgfxPageRun));
	at += count * sizeof(RgfxPageRun);
	for (uint32_t i = 0; i < lentCount; i++) {
		memcpy(at, &lent[i].record, sizeof(RgfxVtSurface));
		at += sizeof(RgfxVtSurface);
		memcpy(at, lent[i].runs, lent[i].record.run_count * sizeof(RgfxPageRun));
		at += lent[i].record.run_count * sizeof(RgfxPageRun);
	}
	return static_cast<uint32_t>(size);
}

// The service's routes: each UUID a client registered, with the client, in the order registered.
// A client is any pointer; the table never follows one.
struct RgfxVtRoutes {
	struct Route {
		uint8_t uuid[kRgfxVtUuidBytes];
		const void *client;
	};
	Route routes[kRgfxVtRoutesMax];
	uint32_t count;

	// Registers `uuid` for `client`, again where it holds it already, as Apple's kext does; false
	// where the table is full.
	bool add(const uint8_t uuid[kRgfxVtUuidBytes], const void *client)
	{
		if (count == kRgfxVtRoutesMax)
			return false;
		memcpy(routes[count].uuid, uuid, kRgfxVtUuidBytes);
		routes[count].client = client;
		count++;
		return true;
	}

	// Deregisters every registration of `uuid` for `client`: whether it held one, for the kext to
	// send the host `-uui`.
	bool remove(const uint8_t uuid[kRgfxVtUuidBytes], const void *client)
	{
		uint32_t kept = 0;
		bool removed = false;
		for (uint32_t i = 0; i < count; i++) {
			if (routes[i].client == client && !memcmp(routes[i].uuid, uuid, kRgfxVtUuidBytes)) {
				removed = true;
				continue;
			}
			routes[kept++] = routes[i];
		}
		count = kept;
		return removed;
	}

	// Deregisters each UUID `client` registered, and writes them, each once, in the order
	// registered, into `out`, at most `max`: how many it wrote.
	uint32_t removeClient(const void *client, uint8_t (*out)[kRgfxVtUuidBytes], uint32_t max)
	{
		uint32_t kept = 0, written = 0;
		for (uint32_t i = 0; i < count; i++) {
			if (routes[i].client != client) {
				routes[kept++] = routes[i];
				continue;
			}
			bool seen = false;
			for (uint32_t w = 0; w < written && !seen; w++)
				seen = !memcmp(out[w], routes[i].uuid, kRgfxVtUuidBytes);
			if (!seen && written < max)
				memcpy(out[written++], routes[i].uuid, kRgfxVtUuidBytes);
		}
		count = kept;
		return written;
	}

	// The clients that registered `uuid`, each once, into `out`, at most `max`: how many.
	uint32_t clientsOf(const uint8_t uuid[kRgfxVtUuidBytes], const void **out, uint32_t max) const
	{
		uint32_t n = 0;
		for (uint32_t i = 0; i < count && n < max; i++) {
			if (memcmp(routes[i].uuid, uuid, kRgfxVtUuidBytes))
				continue;
			bool seen = false;
			for (uint32_t c = 0; c < n && !seen; c++)
				seen = out[c] == routes[i].client;
			if (!seen)
				out[n++] = routes[i].client;
		}
		return n;
	}
};

// The runs of the first `length` bytes of a buffer whose runs are `runs` (`count` of them), into
// `out`, at most `max`: how many, 0 where they do not hold `length` bytes or do not fit.
static inline uint32_t rgfxVtMessageRuns(const RgfxPageRun *runs, uint32_t count, uint64_t length,
                                         RgfxPageRun *out, uint32_t max)
{
	uint32_t n = 0;
	for (uint32_t i = 0; i < count && length; i++) {
		if (n == max)
			return 0;
		const uint64_t take = runs[i].length < length ? runs[i].length : length;
		out[n++] = {runs[i].address, take};
		length -= take;
	}
	return length ? 0 : n;
}

// Whether `caps` lists a decoder for the service to serve: a decode entry of H.264 or HEVC with a
// profile, in caps of the version the kext reads. False for an older host's caps and for a host
// with no video arm, which writes none.
static inline bool rgfxVtServesDecoders(const RgfxVideoCaps &caps)
{
	if (caps.version < RGFX_VIDEO_CAPS_VERSION)
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

#endif // RGFX_VT_ROUTES_HPP_
