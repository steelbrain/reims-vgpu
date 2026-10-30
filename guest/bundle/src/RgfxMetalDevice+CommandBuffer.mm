/*
 * RgfxMetalDevice (CommandBuffer): see the header. Vends the command queue from the
 * device; the queue, command buffer and encoders are in RgfxCommandObjects.mm.
 *
 * Our own ObjC++; the only Apple-private names are the forward-declarations in
 * RgfxMetalPrivate.h.
 */

#import "RgfxMetalDevice+CommandBuffer.h"

#import "RgfxCommandObjects.h"

#include "RgfxSerializer.hpp"

@implementation RgfxMetalDevice (CommandBuffer)

- (nullable id<MTLCommandQueue>)newCommandQueue
{
	return [self newCommandQueueWithMaxCommandBufferCount:0];
}

- (nullable id<MTLCommandQueue>)newCommandQueueWithMaxCommandBufferCount:
    (NSUInteger)maxCommandBufferCount
{
	RgfxSerializer *ser = [self serializer];
	if (ser == nullptr) {
		return nil; // documented stub: no ring transport yet (slice 0's device)
	}
	// When the device-region/ring allocator lands, this allocates the queue's
	// submission (and completion) ring in the device region and hands the submission
	// ring's writer to the factory below, which routes the RGFX_CMD_QUEUE_CREATE and
	// wraps the queue:
	//
	//   RgfxRingWriter *submit = [self allocateSubmissionRing];
	//   return [RgfxCommandQueue queueWithSubmitRing:submit ids:[self ids]
	//                                       maxCount:maxCommandBufferCount
	//                                         device:self serializer:ser];
	//
	// Until that allocator exists there is no ring to place, so the queue cannot be
	// stood up from the device and this returns nil. The unit test drives the queue
	// directly through the factory with a test ring (RgfxCommandObjects.h).
	(void)maxCommandBufferCount;
	return nil;
}

@end
