/*
 * RgfxMetalDevice (CommandBuffer): see the header. Vends the command queue from the
 * device; the queue, command buffer and encoders are in RgfxCommandObjects.mm.
 *
 * Our own ObjC++; the only Apple-private names are the forward-declarations in
 * RgfxMetalPrivate.h.
 */

#import "RgfxMetalDevice+CommandBuffer.h"

#import "RgfxCommandObjects.h"

#include "RgfxRegion.hpp"
#include "RgfxRingWriter.hpp"
#include "RgfxSerializer.hpp"

namespace {
// A queue's submission and completion ring byte-ring sizes (each a power of two). The
// completion ring is allocated for the pair but unused by the host yet: replies come
// back on the control completion ring (the single-ring model, RgfxReplyReader); when
// the host grows per-queue completion rings, the reader reads this one.
constexpr uint32_t kQueueSubmitDataSize = 32768;
constexpr uint32_t kQueueCompleteDataSize = 32768;
} // namespace

@implementation RgfxMetalDevice (CommandBuffer)

- (nullable id<MTLCommandQueue>)newCommandQueue
{
	return [self newCommandQueueWithMaxCommandBufferCount:0];
}

- (nullable id<MTLCommandQueue>)newCommandQueueWithMaxCommandBufferCount:
    (NSUInteger)maxCommandBufferCount
{
	RgfxSerializer *ser = [self serializer];
	RgfxRegion *region = [self region];
	if (ser == nullptr || region == nullptr) {
		// No region mapped yet (the wire handshake is a later piece), so there is no
		// ring to place and the queue cannot be stood up from the device. The unit
		// test drives the factory directly with a fixture region (RgfxCommandObjects.h).
		return nil;
	}

	// Place the queue's submission + completion ring pair in the device region, then
	// construct the submission ring's writer (owned by the device) and route an
	// RGFX_CMD_QUEUE_CREATE carrying the allocated offsets. The device-wide reader of
	// the control completion ring is handed to the queue so its command buffers'
	// completions, reflections and event signals are dispatched.
	RgfxRegion::RingPair rings =
	    region->allocateRingPair(kQueueSubmitDataSize, kQueueCompleteDataSize);
	if (!rings.ok) {
		return nil; // the region had no room for the ring pair
	}
	RgfxRingWriter *submit = [self addQueueSubmitRingAtOffset:rings.submit_offset
	                                                dataSize:kQueueSubmitDataSize];
	if (submit == nullptr) {
		return nil;
	}
	RgfxCommandQueue *queue =
	    [RgfxCommandQueue queueWithSubmitRing:submit
	                             submitOffset:rings.submit_offset
	                           completeOffset:rings.complete_offset
	                         completeDataSize:rings.complete_data_size
	                                      ids:[self ids]
	                                 maxCount:maxCommandBufferCount
	                                   device:self
	                               serializer:ser];
	queue.replyReader = [self replyReader];
	return queue;
}

@end
