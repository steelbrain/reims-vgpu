/*
 * RgfxVideoStatus: what the video driver answers AppleGVA for a decode it could not complete
 * (RgfxVideoDriver.cpp: ScheduleCommands, QueryCommandStatus). Kept apart from the driver so its
 * decisions are unit tested without the Metal device the driver needs
 * (tests/video_h265_tests.cpp).
 *
 * AppleGVA's decode flow reports a decode error to VideoToolbox only where ScheduleCommands
 * fails: for HEVC the frame's callback then carries kVTVideoDecoderMalfunctionErr (-12911), and
 * so does every frame of the session after it. It takes every query of a decode for its
 * completion: it reads neither the command's status nor its error word, and delivers the
 * surface as a decoded frame whatever the query answers (read from AppleGVA's code on macOS
 * 26.7.1, 2026-11-08). A result in the query's fields, a failed call, status 3 and status 0 each
 * deliver the frame (measured on the guest with the RTX 5080 host, 2026-11-03). So the schedule
 * of an HEVC decode waits for the host's verdict on its picture (waitsForHost), and fails where
 * the host failed it: VideoToolbox reports the error from that picture on, and no frame of a
 * picture the host failed reaches the guest. A query of a decode the host has not finished
 * answers it running, as the driver's answer is only what is so.
 *
 * AppleGVA's H.264 decoder reports no failure at all: a failed schedule, answered 5, 1 or 7, once
 * or for every schedule after; the query's status 0, 3 or 4; its field_08 or field_14 set (to 1,
 * 5, 0xe00002bc or 0xffffffff); a failed query call: every frame reaches VideoToolbox as decoded,
 * and a picture whose schedule failed is never decoded (measured on the guest with the RTX 5080
 * host, 2026-11-07, every picture failed on the host). Its pipelines schedule their commands one
 * at a time and keep no decode's result (read from AppleGVA's code on macOS 26.7.1). So an H.264
 * schedule does not wait for the host, as waiting would only cost: its frames reach the guest as
 * decoded, as AppleGVA leaves no way to say otherwise. What the driver can tell before the first
 * picture, a coded size the host does not decode (sizeTaken), it refuses at the context's create,
 * which fails the client's session create, so VideoToolbox decodes the stream in software.
 */

#ifndef RGFX_VIDEO_STATUS_HPP_
#define RGFX_VIDEO_STATUS_HPP_

#include <cstdint>

#include "reims_graphics_wire.h"

namespace rgfx::video {

// A query's status: a command is still decoding, or it is done.
constexpr uint32_t kStatusRunning = 2, kStatusDone = 1;

// ScheduleCommands' answer for a list holding a command the driver did not send, or one the host
// failed.
constexpr uint32_t kNotSent = 5;

// The status a query answers for a command whose work on the host is `hostStatus` (0 still
// running, 1 done, -1 a host failure): running, or done, a failure included.
inline uint32_t queryStatus(int hostStatus)
{
	return hostStatus == 0 ? kStatusRunning : kStatusDone;
}

// Whether a decode's schedule waits for the host's verdict on its picture: where AppleGVA reports
// a failed schedule (HEVC's sessions, `hevc` true), not H.264's (the header comment).
inline bool waitsForHost(bool hevc)
{
	return hevc;
}

// Whether ScheduleCommands fails for a list of decode commands: one holds a command the driver
// did not send (`sent` false), or one the host failed (`hostFailed`, known where its schedule
// waited for the host).
inline bool scheduleFails(bool sent, bool hostFailed)
{
	return !sent || hostFailed;
}

// Whether the host decodes pictures of `width` x `height` coded samples, by its caps (`caps`, null
// for none): within its least and largest coded size, where it names them.
inline bool sizeTaken(const RgfxVideoCodecCaps *caps, uint32_t width, uint32_t height)
{
	if (!caps)
		return false;
	const uint32_t size[2] = {width, height};
	for (int i = 0; i < 2; i++)
		if ((caps->min_size[i] && size[i] < caps->min_size[i]) ||
		    (caps->max_size[i] && size[i] > caps->max_size[i]))
			return false;
	return true;
}

} // namespace rgfx::video

#endif
