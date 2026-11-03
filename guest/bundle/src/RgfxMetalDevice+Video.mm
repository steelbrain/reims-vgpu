/*
 * RgfxMetalDevice+Video: the video driver's way to the host, through the process's device
 * (RgfxVideoChannel.hpp). Our own ObjC++.
 */

#import "RgfxCommandObjects.h"
#import "RgfxIOSurface.h"
#import "RgfxMetalDevice.h"

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

// The output surfaces a session keeps textures over, past which the least recently used goes:
// more than a decoder's pool holds, so a session's own surfaces stay.
constexpr size_t kOutputsKept = 64;

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
	// The textures over each output surface's two planes, by IOSurfaceID, and when each was
	// last used.
	struct Output {
		id<MTLTexture> luma;
		id<MTLTexture> chroma;
		uint64_t lastUse;
	};
	std::unordered_map<uint32_t, Output> outputs;
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
		const NSUInteger a = [device minimumLinearTextureAlignmentForPixelFormat:format];
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

// A texture of `format` over `surface`'s plane `plane`, of the plane's size; over the surface's
// device cache where the host lands pictures there and the texture makes the surface's block.
id<MTLTexture> planeTexture(RgfxMetalDevice *device, IOSurfaceRef surface, size_t plane,
                            MTLPixelFormat format)
{
	MTLTextureDescriptor *desc =
	    [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format
	                                                       width:IOSurfaceGetWidthOfPlane(surface, plane)
	                                                      height:IOSurfaceGetHeightOfPlane(surface, plane)
	                                                   mipmapped:NO];
	desc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
	const RgfxVideoCaps *caps = [device videoCaps];
	if ((caps->flags & RGFX_VIDEO_CAPS_OUTPUT_CACHED) == 0)
		return [device newTextureWithDescriptor:desc iosurface:surface plane:plane];
	RgfxAskCachedScope ask;
	return [device newTextureWithDescriptor:desc iosurface:surface plane:plane];
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
	id<MTLTexture> luma, chroma;
	{
		std::lock_guard<std::mutex> hold(session->lock);
		const uint32_t id = IOSurfaceGetID(decode.output);
		auto it = session->outputs.find(id);
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
			output.luma = planeTexture(session->device, decode.output, 0, lumaFormat);
			output.chroma = planeTexture(session->device, decode.output, 1, chromaFormat);
			if (output.luma == nil || output.chroma == nil)
				return nullptr;
			it = session->outputs.emplace(id, output).first;
		}
		it->second.lastUse = ++session->uses;
		luma = it->second.luma;
		chroma = it->second.chroma;
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
	                                     chroma:chroma];
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

void rgfx::video::releaseToken(void *token)
{
	if (token != nullptr)
		CFRelease(token);
}
