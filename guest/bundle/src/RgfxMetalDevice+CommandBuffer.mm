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
#include "RgfxCompletionThread.hpp"
#include "RgfxReplyReader.hpp"
#include "RgfxRingWriter.hpp"
#include "RgfxSerializer.hpp"

namespace {
// A queue's submission and completion ring byte-ring sizes (each a power of two). The
// host's ring loop (crates/reims-graphics/src/device.rs) posts this queue's command-
// buffer completions on its own completion ring, so the queue reads that ring (not the
// device's control completion ring); see newCommandQueue.
constexpr uint32_t kQueueSubmitDataSize = 32768;
constexpr uint32_t kQueueCompleteDataSize = 32768;
} // namespace

@implementation RgfxMetalDevice (CommandBuffer)

- (nullable id<MTLCommandQueue>)newCommandQueue
{
	return [self newCommandQueueWithMaxCommandBufferCount:0];
}

// A queue of the descriptor's command buffer count. The descriptor's log state can only be one
// the device made, and it makes none (newLogStateWithDescriptor:error:).
- (nullable id<MTLCommandQueue>)newCommandQueueWithDescriptor:(MTLCommandQueueDescriptor *)descriptor
{
	return [self newCommandQueueWithMaxCommandBufferCount:descriptor.maxCommandBufferCount];
}

- (nullable id<MTLCommandQueue>)newCommandQueueWithMaxCommandBufferCount:
    (NSUInteger)maxCommandBufferCount
{
	RgfxSerializer *ser = [self serializer];
	RgfxRegion *region = [self region];
	if (ser == nullptr || region == nullptr) {
		// A device whose region did not map (its wire handshake failed) has no ring to
		// place, so no queue stands up. The unit test drives the factory directly with a
		// fixture region (RgfxCommandObjects.h).
		return nil;
	}

	// Place the queue's submission + completion ring pair in the device region, then
	// construct the submission ring's writer (owned by the device) and route an
	// RGFX_CMD_QUEUE_CREATE carrying the allocated offsets.
	RgfxRegion::RingPair rings =
	    region->allocateRingPair(kQueueSubmitDataSize, kQueueCompleteDataSize);
	if (!rings.ok) {
		return nil; // the region had no room for the ring pair
	}
	auto releaseRings = [region, rings] {
		region->release(rings.submit_block);
		region->release(rings.complete_block);
	};
	RgfxRingWriter *submit = [self addQueueSubmitRingAtOffset:rings.submit_offset
	                                                dataSize:kQueueSubmitDataSize];
	if (submit == nullptr) {
		releaseRings();
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
	if (queue == nil) {
		[self dropQueueSubmitRing:submit];
		releaseRings();
		return nil;
	}
	// At the queue's end, once the device has ended it, the writer over its submission ring
	// goes, then its rings' blocks.
	__weak RgfxMetalDevice *weakSelf = self;
	[queue rgfxSetRelease:[weakSelf, submit, releaseRings] {
		RgfxMetalDevice *device = weakSelf;
		[device dropQueueSubmitRing:submit];
		releaseRings();
	}];
	// The host's ring loop posts this queue's command-buffer completions on the queue's
	// own completion ring (device.rs: QueueCompletionPoster on the guest-given
	// complete_offset), raising the queue's vector. So the queue reads THAT ring, not the
	// device's control completion ring; it owns the reader for the queue's lifetime.
	RgfxReplyReader *reader =
	    new RgfxReplyReader(region->base() + rings.complete_offset, rings.complete_data_size);
	// The shared events its command buffers signal are the device's, which they listen on.
	reader->setEventsTo([self replyReader]);
	[queue adoptReplyReader:reader];
	// A command buffer too large for the ring puts its message in a block of the region.
	queue.region = region;
	queue.slabs = [self slabs];
	// Its command buffers that commitAndWaitUntilSubmitted commits ask for their submission
	// where the host answers it.
	const RgfxCaps *caps = [self caps];
	queue.asksSubmitted = caps != nullptr && (caps->features() & RGFX_FEATURE_CB_SUBMITTED) != 0;
	// A commit that finds the queue's ring full waits for the host's room record on the queue's
	// completion ring, which the reader hands to the ring's writer, where the device's word holds
	// SUBMIT_ROOM; else it sleeps between looks (RgfxRingWriter::post).
	if (caps != nullptr && (caps->features() & RGFX_FEATURE_SUBMIT_ROOM) != 0) {
		auto room = std::make_shared<RgfxRingRoomEvents>();
		reader->setRoomEvents(room);
		submit->setRoomEvents(room);
	}
	// Where the host commits a process's command buffers in the order its queues put them on
	// their rings, the queue takes part in the process's commit order (RgfxCommitOrder.hpp): the
	// device made one where its word holds RGFX_FEATURE_COMMIT_ORDER.
	[queue rgfxJoinCommitOrder:[self commitOrder]];
	// The process's completion thread takes the ring's completions from now on, whether or
	// not anything waits for them.
	if (RgfxCompletionThread *completions = [self completionThread])
		completions->add([queue sharedReplyReader]);
	return queue;
}

@end
