/*
 * RgfxVideoStatus: what the video driver answers AppleGVA for a decode it could not complete
 * (RgfxVideoDriver.cpp: ScheduleCommands, QueryCommandStatus). Kept apart from the driver so its
 * decisions are unit tested without the Metal device the driver needs
 * (tests/video_h265_tests.cpp).
 *
 * AppleGVA reports a decode error to VideoToolbox only where ScheduleCommands fails: the frame's
 * callback then carries kVTVideoDecoderMalfunctionErr (-12911). It ignores every answer of
 * QueryCommandStatus tried for a done command: a result in the query's fields, a failed call,
 * status 3 and status 0 each deliver the untouched surface as a decoded frame (measured on the
 * guest with the RTX 5080 host, 2026-11-03). So a command the driver cannot send fails its
 * schedule, and a decode that fails on the host after its schedule fails the session's next
 * schedule: one error a failure, a frame late, never a silent frame.
 */

#ifndef RGFX_VIDEO_STATUS_HPP_
#define RGFX_VIDEO_STATUS_HPP_

#include <cstdint>

namespace rgfx::video {

// A query's status: a command is still decoding, or it is done.
constexpr uint32_t kStatusRunning = 2, kStatusDone = 1;

// ScheduleCommands' answer for a list holding a command the driver did not send.
constexpr uint32_t kNotSent = 5;

// What the driver reports for a command whose work on the host is `hostStatus` (0 still running,
// 1 done, -1 a host failure): its status, and whether its session's next schedule is to fail
// for it.
struct QueryReport {
	uint32_t status;
	bool hostFailed;
};

inline QueryReport queryReport(int hostStatus)
{
	if (hostStatus == 0)
		return {kStatusRunning, false};
	return {kStatusDone, hostStatus == -1};
}

// Whether ScheduleCommands fails for a decode command: one the driver did not send (`sent`
// false), or one on a session whose earlier decode failed on the host since its last failed
// schedule (`hostFailed`).
inline bool scheduleFails(bool sent, bool hostFailed)
{
	return !sent || hostFailed;
}

} // namespace rgfx::video

#endif
