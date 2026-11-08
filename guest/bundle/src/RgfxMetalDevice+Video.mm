/*
 * RgfxMetalDevice+Video: the video driver's way to the host, through the process's device
 * (RgfxVideoChannel.hpp). Our own ObjC++.
 */

#import "RgfxCommandObjects.h"
#import "RgfxIOSurface.h"
#import "RgfxMetalDevice.h"

#include <algorithm>
#include <mutex>
#include <unordered_map>

#include "RgfxSerializer.hpp"
#include "RgfxVideoChannel.hpp"

namespace {

// The process's device, where Metal's default device is ours; nil elsewhere.
RgfxMetalDevice *processDevice()
{
	static RgfxMetalDevice *device;
	static dispatch_once_t once;
	dispatch_once(&once, ^{
		id<MTLDevice> d = MTLCreateSystemDefaultDevice();
		if ([d isKindOfClass:[RgfxMetalDevice class]])
			device = (RgfxMetalDevice *)d;
	});
	return device;
}

// The output surfaces a session keeps textures over (a pair for each place a picture lands at in
// one: an HEIF grid's canvas has one for each of its tiles but the first), past which the least
// recently used goes: more than a decoder's pool holds, so a session's own surfaces stay.
constexpr size_t kOutputsKept = 64;
// The largest granularity and alignment a host's layout in place may name
// (RgfxVideoOutputLayout): a CTU, and a page.
constexpr uint32_t kMaxGranularity = 64;
constexpr uint32_t kMaxLayoutAlignment = 4096;

// The bitstream buffers a session uses in turn: a picture waits for the command buffer that
// last read its buffer only where this many are in flight.
constexpr size_t kBitstreams = 8;

// The least a bitstream buffer holds; one grows to the next power of two that holds an
// access unit.
constexpr size_t kBitstreamMin = 256 * 1024;

} // namespace

struct rgfx::video::Session {
	RgfxMetalDevice *device;
	id<MTLCommandQueue> queue;
	uint64_t objId;
	// RGFX_VIDEO_CODEC_*.
	uint32_t codec;
	std::mutex lock;
	// The textures over each output surface's two planes, by IOSurfaceID and the place in it
	// the picture lands at (outputKey), and when each was last used.
	struct Output {
		id<MTLTexture> luma;
		id<MTLTexture> chroma;
		uint64_t lastUse;
		// The luma rows the surface holds for a frame of its own (the op's output_rows), 0 for
		// a tile or a surface whose view is a copy.
		uint32_t rows = 0;
	};
	std::unordered_map<uint64_t, Output> outputs;
	uint64_t uses = 0;
	// The last command buffer committed, which the session's end waits for before the host
	// drops the session its video encoder names.
	id<MTLCommandBuffer> last;
	// Held across a decode, so pictures take bitstream buffers in their queue's order.
	std::mutex submit;
	// The bitstream buffers, each with the command buffer that last read it, and the next to
	// use.
	struct Bitstream {
		id<MTLBuffer> buffer;
		id<MTLCommandBuffer> reader;
	};
	Bitstream bitstreams[kBitstreams];
	size_t nextBitstream = 0;
};

const RgfxVideoCodecCaps *rgfx::video::hostCodec(uint32_t codec, uint32_t direction)
{
	RgfxMetalDevice *device = processDevice();
	if (device == nil)
		return nullptr;
	const RgfxVideoCaps *caps = [device videoCaps];
	for (uint32_t i = 0; i < caps->codec_count; i++) {
		if (caps->codecs[i].codec == codec && caps->codecs[i].direction == direction)
			return &caps->codecs[i];
	}
	return nullptr;
}

RgfxVideoDecodeCaps rgfx::video::decodeCaps(uint32_t codec)
{
	RgfxMetalDevice *device = processDevice();
	if (device == nil)
		return {};
	const RgfxVideoCaps *caps = [device videoCaps];
	for (uint32_t i = 0; i < caps->codec_count && i < RGFX_VIDEO_CAPS_CODECS; i++)
		if (caps->codecs[i].codec == codec && caps->codecs[i].direction == RGFX_VIDEO_DECODE)
			return caps->decode[i];
	return {};
}

RgfxVideoOutputLayout rgfx::video::outputLayout(uint32_t codec)
{
	RgfxMetalDevice *device = processDevice();
	if (device == nil)
		return {};
	const RgfxVideoCaps *caps = [device videoCaps];
	for (uint32_t i = 0; i < caps->codec_count && i < RGFX_VIDEO_CAPS_CODECS; i++) {
		if (caps->codecs[i].codec != codec || caps->codecs[i].direction != RGFX_VIDEO_DECODE)
			continue;
		const RgfxVideoOutputLayout layout = caps->layouts[i];
		// Units of a few macroblocks, and an alignment of a power of two up to a page.
		const bool fits = layout.granularity[0] != 0 && layout.granularity[0] <= kMaxGranularity &&
		                  layout.granularity[1] != 0 && layout.granularity[1] <= kMaxGranularity &&
		                  layout.alignment != 0 && layout.alignment <= kMaxLayoutAlignment &&
		                  (layout.alignment & (layout.alignment - 1)) == 0;
		return fits ? layout : RgfxVideoOutputLayout{};
	}
	return {};
}

uint32_t rgfx::video::outputPlaneAlignment(uint32_t sampleBytes)
{
	RgfxMetalDevice *device = processDevice();
	if (device == nil || (sampleBytes != 1 && sampleBytes != 2))
		return 0;
	const MTLPixelFormat luma = sampleBytes == 1 ? MTLPixelFormatR8Unorm : MTLPixelFormatR16Unorm;
	const MTLPixelFormat chroma = sampleBytes == 1 ? MTLPixelFormatRG8Unorm : MTLPixelFormatRG16Unorm;
	// The host's copy into the planes writes from offsets of whole 4 bytes, as Vulkan's copies on
	// a queue that neither draws nor computes take them (BufferPlanes in the video crate).
	NSUInteger alignment = 4;
	for (MTLPixelFormat format : {luma, chroma}) {
		const NSUInteger a = [device rgfxLinearTextureAlignmentForPixelFormat:format];
		if (a > alignment)
			alignment = a;
	}
	return alignment <= UINT32_MAX ? (uint32_t)alignment : 0;
}

rgfx::video::Session *rgfx::video::openSession(uint32_t codec, uint32_t direction)
{
	if (hostCodec(codec, direction) == nullptr)
		return nullptr;
	RgfxMetalDevice *device = processDevice();
	RgfxSerializer *serializer = [device serializer];
	if (serializer == nullptr)
		return nullptr;
	id<MTLCommandQueue> queue = [device newCommandQueue];
	if (queue == nil)
		return nullptr;
	RgfxCreated created = serializer->createVideoSession(codec, direction);
	if (!created.posted)
		return nullptr;
	Session *session = new Session;
	session->device = device;
	session->queue = queue;
	session->objId = created.obj_id;
	session->codec = codec;
	return session;
}

void rgfx::video::closeSession(Session *session)
{
	if (session == nullptr)
		return;
	id<MTLCommandBuffer> last;
	{
		std::lock_guard<std::mutex> hold(session->lock);
		last = session->last;
		session->outputs.clear();
	}
	[last waitUntilCompleted];
	if (RgfxSerializer *serializer = [session->device serializer])
		serializer->destroy(session->objId);
	delete session;
}

void rgfx::video::forgetOutputs(Session *session)
{
	if (session == nullptr)
		return;
	std::lock_guard<std::mutex> hold(session->lock);
	session->outputs.clear();
}

namespace {

// The key of the textures over `surface` a picture landing at (`x`, `y`) in it uses.
uint64_t outputKey(IOSurfaceRef surface, uint32_t x, uint32_t y)
{
	return (uint64_t)IOSurfaceGetID(surface) << 32 | (uint64_t)(y & 0xffff) << 16 | (x & 0xffff);
}

// A texture of `format` over `surface`'s plane `plane`: of the plane's size, or, `width` by
// `height` from `origin` in it (a tile's place in its canvas, in the plane's elements), where
// `width` is not 0; over the surface's device cache where the host lands pictures there, and over
// its pages where it does not and the texture makes the surface's block.
id<MTLTexture> planeTexture(RgfxMetalDevice *device, IOSurfaceRef surface, size_t plane,
                            MTLPixelFormat format, MTLOrigin origin, size_t width, size_t height)
{
	const size_t w = width ? width : IOSurfaceGetWidthOfPlane(surface, plane);
	const size_t h = width ? height : IOSurfaceGetHeightOfPlane(surface, plane);
	MTLTextureDescriptor *desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format
	                                                                                width:w
	                                                                               height:h
	                                                                            mipmapped:NO];
	desc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
	auto make = [&]() -> id<MTLTexture> {
		if (width == 0)
			return [device newTextureWithDescriptor:desc iosurface:surface plane:plane];
		return [RgfxIOSurfaceTexture textureWithDescriptor:desc
		                                         iosurface:surface
		                                             plane:plane
		                                            origin:origin
		                                            region:[device region]
		                                            device:device
		                                        serializer:[device serializer]];
	};
	const RgfxVideoCaps *caps = [device videoCaps];
	if ((caps->flags & RGFX_VIDEO_CAPS_OUTPUT_CACHED) != 0)
		return make();
	RgfxAskUncachedScope ask;
	return make();
}

// What the host reads an access unit of `codec` in place at
// (RgfxVideoCodecCaps::bitstream_alignment): from a buffer's start, its length a multiple of it.
size_t bitstreamAlignment(uint32_t codec)
{
	const RgfxVideoCodecCaps *caps = rgfx::video::hostCodec(codec, RGFX_VIDEO_DECODE);
	const size_t alignment = caps ? caps->bitstream_alignment : 1;
	return alignment != 0 && (alignment & (alignment - 1)) == 0 ? alignment : 1;
}

// The formats of the planes of an output surface of `format` a decode of `codec` lands in:
// false for a format it lands in none of.
bool planeFormats(uint32_t codec, OSType format, MTLPixelFormat *luma, MTLPixelFormat *chroma)
{
	switch (format) {
	case '420v':
	case '420f':
		*luma = MTLPixelFormatR8Unorm;
		*chroma = MTLPixelFormatRG8Unorm;
		return true;
	case 'x420':
	case 'xf20':
		*luma = MTLPixelFormatR16Unorm;
		*chroma = MTLPixelFormatRG16Unorm;
		return codec == RGFX_VIDEO_CODEC_HEVC;
	default:
		return false;
	}
}

// `decode`'s access unit copied into the next of `session`'s bitstream buffers, from its first
// start code and padded with zeros to the host's alignment: the buffer, and the bytes the host
// reads; nil where the device made no buffer. The caller holds the session's submit lock.
id<MTLBuffer> copyAccessUnit(rgfx::video::Session *session, const rgfx::video::Decode &decode,
                             size_t *size)
{
	// Zero bytes before the first 3-byte start code are the byte stream's, no NAL unit's.
	size_t lead = 0;
	while (lead + 3 < decode.size && decode.bytes[lead] == 0 && decode.bytes[lead + 1] == 0 &&
	       decode.bytes[lead + 2] == 0)
		lead++;
	const size_t bytes = decode.size - lead;
	const size_t alignment = bitstreamAlignment(session->codec);
	const size_t padded = (bytes + alignment - 1) & ~(alignment - 1);
	auto &slot = session->bitstreams[session->nextBitstream++ % kBitstreams];
	[slot.reader waitUntilCompleted];
	slot.reader = nil;
	if (slot.buffer == nil || slot.buffer.length < padded) {
		size_t length = kBitstreamMin;
		while (length < padded)
			length *= 2;
		slot.buffer = [session->device newBufferWithLength:length
		                                           options:MTLResourceStorageModeShared];
		if (slot.buffer == nil || slot.buffer.contents == nullptr) {
			slot.buffer = nil;
			return nil;
		}
	}
	uint8_t *to = (uint8_t *)slot.buffer.contents;
	memcpy(to, decode.bytes + lead, bytes);
	memset(to + bytes, 0, padded - bytes);
	*size = padded;
	return slot.buffer;
}

} // namespace

void *rgfx::video::decode(Session *session, const Decode &decode)
{
	if (session == nullptr || decode.bytes == nullptr || decode.size == 0 ||
	    decode.output == nullptr)
		return nullptr;
	MTLPixelFormat lumaFormat, chromaFormat;
	if (!planeFormats(session->codec, IOSurfaceGetPixelFormat(decode.output), &lumaFormat,
	                  &chromaFormat) ||
	    IOSurfaceGetPlaneCount(decode.output) != 2)
		return nullptr;
	// A picture off the surface's origin (a tile in its canvas) lands through textures of its
	// crop there, each plane's first byte at the host's alignment; at the origin, through the
	// textures over the whole planes.
	const bool inside = decode.x != 0 || decode.y != 0;
	if (inside) {
		const uint32_t sampleBytes = lumaFormat == MTLPixelFormatR16Unorm ? 2 : 1;
		const uint32_t alignment = outputPlaneAlignment(sampleBytes);
		if (decode.x % 2 != 0 || decode.y % 2 != 0 || decode.width == 0 || decode.height == 0 ||
		    alignment == 0 || (decode.x * sampleBytes) % alignment != 0)
			return nullptr;
	}
	id<MTLTexture> luma, chroma;
	uint32_t outputRows = 0;
	{
		std::lock_guard<std::mutex> hold(session->lock);
		const uint64_t id = outputKey(decode.output, decode.x, decode.y);
		auto it = session->outputs.find(id);
		// Another crop at the same place (a grid of other tiles in a canvas surface used again)
		// takes textures of its own.
		if (it != session->outputs.end() && inside &&
		    (it->second.luma.width != decode.width || it->second.luma.height != decode.height)) {
			session->outputs.erase(it);
			it = session->outputs.end();
		}
		if (it == session->outputs.end()) {
			if (session->outputs.size() >= kOutputsKept) {
				auto oldest = session->outputs.begin();
				for (auto o = session->outputs.begin(); o != session->outputs.end(); ++o) {
					if (o->second.lastUse < oldest->second.lastUse)
						oldest = o;
				}
				session->outputs.erase(oldest);
			}
			Session::Output output;
			const size_t w = inside ? decode.width : 0, h = inside ? decode.height : 0;
			output.luma = planeTexture(session->device, decode.output, 0, lumaFormat,
			                           MTLOriginMake(decode.x, decode.y, 0), w, h);
			output.chroma = planeTexture(session->device, decode.output, 1, chromaFormat,
			                             MTLOriginMake(decode.x / 2, decode.y / 2, 0), (w + 1) / 2,
			                             (h + 1) / 2);
			if (output.luma == nil || output.chroma == nil)
				return nullptr;
			// The rows the surface holds past the planes' own, where its view is its pages:
			// the luma rows, and twice the chroma rows, the lesser; 0 where they are no more.
			if (!inside) {
				const uint64_t rows = std::min(rgfxIOSurfaceRowsHeld(output.luma),
				                               2 * rgfxIOSurfaceRowsHeld(output.chroma));
				if (rows > output.luma.height && rows <= UINT32_MAX)
					output.rows = (uint32_t)rows;
			}
			it = session->outputs.emplace(id, output).first;
		}
		it->second.lastUse = ++session->uses;
		luma = it->second.luma;
		chroma = it->second.chroma;
		outputRows = inside ? 0 : it->second.rows;
	}
	std::lock_guard<std::mutex> submit(session->submit);
	id<MTLCommandBuffer> cb = [session->queue commandBuffer];
	if (![cb isKindOfClass:[RgfxCmdBuffer class]])
		return nullptr;
	size_t size = 0;
	id<MTLBuffer> bitstream = copyAccessUnit(session, decode, &size);
	if (bitstream == nil)
		return nullptr;
	[(RgfxCmdBuffer *)cb rgfxEncodeVideoSession:session->objId
	                                      codec:session->codec
	                              parameterSets:decode.parameterSets
	                                       size:decode.parameterSetsSize
	                                  bitstream:bitstream
	                                    ofBytes:size
	                                       luma:luma
	                                     chroma:chroma
	                                 outputRows:outputRows];
	[cb commit];
	session->bitstreams[(session->nextBitstream - 1) % kBitstreams].reader = cb;
	{
		std::lock_guard<std::mutex> hold(session->lock);
		session->last = cb;
	}
	return (void *)CFBridgingRetain(cb);
}

int rgfx::video::tokenStatus(void *token)
{
	if (token == nullptr)
		return -1;
	switch (((__bridge id<MTLCommandBuffer>)token).status) {
	case MTLCommandBufferStatusCompleted:
		return 1;
	case MTLCommandBufferStatusError:
		return -1;
	default:
		return 0;
	}
}

int rgfx::video::waitToken(void *token)
{
	if (token == nullptr)
		return -1;
	[(__bridge id<MTLCommandBuffer>)token waitUntilCompleted];
	return tokenStatus(token) == 1 ? 1 : -1;
}

void *rgfx::video::retainToken(void *token)
{
	if (token != nullptr)
		CFRetain(token);
	return token;
}

void rgfx::video::releaseToken(void *token)
{
	if (token != nullptr)
		CFRelease(token);
}
