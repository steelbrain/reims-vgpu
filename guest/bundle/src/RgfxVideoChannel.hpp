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
 * surface and place a picture lands at in it.
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

/// The least alignment, in bytes, of the first byte and the row pitch of each plane of an output
/// surface of samples of `sampleBytes` bytes (1: NV12, 2: P010): the larger of the alignment the
/// process's device takes for a linear texture of each plane's format, as the textures the decode
/// lands through are, and the 4 bytes of the host's copy into the planes. 0 where the process has
/// no device of ours, or for another sample size.
uint32_t outputPlaneAlignment(uint32_t sampleBytes);

/// How the host lands a decode of `codec` in place, in an output surface's planes
/// (RgfxVideoCaps::layouts): the granularity it writes whole coded rows and columns in, and the
/// alignment of the planes it writes so. All zero where the host lands the codec's pictures by a
/// copy of their crop alone, the layout is past what a surface takes, or the process has no device
/// of ours.
RgfxVideoOutputLayout outputLayout(uint32_t codec);

/// A session on the host, its queue, and the textures it made over output surfaces.
struct Session;

/// A session of `codec` in `direction`; null where the host has no such codec.
Session *openSession(uint32_t codec, uint32_t direction);

/// Ends `session` (null: nothing): its textures go, and the host drops it once its command
/// buffers are done.
void closeSession(Session *session);

/// Lets go of the textures `session` made over output surfaces, so the surfaces can go.
void forgetOutputs(Session *session);

/// One access unit to decode, of its session's codec: its slices' NAL units in Annex-B form,
/// `size` bytes at `bytes`, after the parameter sets at `parameterSets` where they changed
/// (H.264's SPS and PPS, HEVC's SPS and PPS; `parameterSetsSize` bytes, 0 for none), into the
/// two-plane surface `output`: NV12's ('420v', '420f'), or, for HEVC of 10 bits, P010's ('x420',
/// 'xf20'). The call copies the bytes: the caller may write them again once it returns.
///
/// The picture lands at `x`, `y` in `output` (luma samples from its top left; chroma at half of
/// each), `width` by `height` of it, its crop: at (0, 0) for a frame of its own, through
/// textures over the whole planes; at its place for an HEIF grid's tile in its canvas, through
/// textures of its crop there, which lies inside the planes. A place off (0, 0) is of even
/// samples, and each plane's first byte there is at the alignment outputPlaneAlignment answers.
/// For a frame of its own, the op names the rows the surface's planes hold (output_rows), which
/// the host may decode the whole coded picture into in place (outputLayout).
struct Decode {
	const uint8_t *bytes;
	size_t size;
	const uint8_t *parameterSets;
	uint32_t parameterSetsSize;
	IOSurfaceRef output;
	uint32_t x, y, width, height;
};

/// Commits the decode on `session`'s queue: a token for its completion, null where it was not
/// encoded (a surface of no format the codec lands in, a surface no texture lies over, or no
/// command buffer).
void *decode(Session *session, const Decode &decode);

/// Where the work `token` names is: 0 running, 1 done, -1 failed.
int tokenStatus(void *token);

/// Lets go of `token` (null: nothing).
void releaseToken(void *token);

} // namespace rgfx::video

#endif
