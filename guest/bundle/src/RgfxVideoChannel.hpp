/*
 * RgfxVideoChannel: the video driver's way to the host (RgfxVideoDriver.cpp), through the
 * process's Metal device, the bundle's own (RgfxMetalDevice+Video.mm).
 *
 * The driver runs in VideoToolbox's decoder and encoder services, where Metal's default device
 * is ours. A session is a video session on the host and a command queue of its own; each
 * picture is a command buffer of one video encoder on that queue, so it completes, names the
 * surface it writes and holds that surface's CPU locks as any command buffer does. Each access
 * unit is copied into the next of the session's own bitstream buffers, Shared buffers of the
 * device that the host reads in place: AppleGVA writes the next access unit into the same
 * buffer of its pool as soon as a picture is scheduled, without waiting for the decode that
 * reads it. The output surfaces are reached through textures over their planes, made once per
 * surface.
 *
 * Plain C++, so the driver's AppleGVA side stays C++; every handle is opaque.
 */

#ifndef RGFX_VIDEO_CHANNEL_HPP_
#define RGFX_VIDEO_CHANNEL_HPP_

#include <IOSurface/IOSurfaceRef.h>
#include <cstddef>
#include <cstdint>

#include "reims_graphics_wire.h"

namespace rgfx::video {

/// The host's answers for `codec` (RGFX_VIDEO_CODEC_*) in `direction` (RGFX_VIDEO_DECODE or
/// RGFX_VIDEO_ENCODE), from the process's device: null where the process has no device of ours
/// or the host's hardware lacks the codec.
const RgfxVideoCodecCaps *hostCodec(uint32_t codec, uint32_t direction);

/// A session on the host, its queue, and the textures it made over output surfaces.
struct Session;

/// A session of `codec` in `direction`; null where the host has no such codec.
Session *openSession(uint32_t codec, uint32_t direction);

/// Ends `session` (null: nothing): its textures go, and the host drops it once its command
/// buffers are done.
void closeSession(Session *session);

/// Lets go of the textures `session` made over output surfaces, so the surfaces can go.
void forgetOutputs(Session *session);

/// One H.264 access unit to decode: its slices' NAL units in Annex-B form, `size` bytes at
/// `bytes`, after the SPS and PPS NAL units at `parameterSets` where they changed
/// (`parameterSetsSize` bytes, 0 for none), into the two-plane surface `output`. The call
/// copies the bytes: the caller may write them again once it returns.
struct DecodeH264 {
	const uint8_t *bytes;
	size_t size;
	const uint8_t *parameterSets;
	uint32_t parameterSetsSize;
	IOSurfaceRef output;
};

/// Commits the decode on `session`'s queue: a token for its completion, null where it was not
/// encoded (a surface no texture lies over, or no command buffer).
void *decodeH264(Session *session, const DecodeH264 &decode);

/// Where the work `token` names is: 0 running, 1 done, -1 failed.
int tokenStatus(void *token);

/// Lets go of `token` (null: nothing).
void releaseToken(void *token);

} // namespace rgfx::video

#endif
