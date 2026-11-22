/*
 * queue_tests: the command queue's and the command buffer's contract with their encoders,
 * against Metal's answers, part of the Metal shell's check (shell_tests.mm calls
 * rgfxQueueTests; guest/bundle/Makefile `test`). Metal's answers these hold to were measured on
 * the M3 Max (macOS 27) and on the MacBook Pro 2019's AMD Radeon Pro 5300M and Intel UHD 630
 * (macOS 26.7.1), which agree on each.
 */

#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <Metal/Metal.h>
#import <objc/message.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <crt_externs.h>
#include <csignal>
#include <cstdint>
#include <cstring>
#include <functional>
#include <mutex>
#include <spawn.h>
#include <string>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
#include <vector>

#import "RgfxBuffer.h"
#import "RgfxCommandObjects.h"
#import "RgfxEvent.h"
#import "RgfxFunction.h"
#import "RgfxIOSurface.h"
#import "RgfxMetalPrivate.h"
#import "RgfxResidencySet.h"
#import "RgfxResource.h"
#import "RgfxSharedEvent.h"
#import "RgfxTexture.h"
#import "RgfxUnsupported.h"

#include "RgfxCompletionThread.hpp"
#include "RgfxObjectModel.hpp"
#include "RgfxRegion.hpp"
#include "RgfxReplyReader.hpp"
#include "RgfxRingWriter.hpp"
#include "RgfxSerializer.hpp"
#include "RgfxSlabs.hpp"
#include "reims_graphics_wire.h"
#include "rgfx_test_ring.hpp" // CHECK, g_checks, TestRing

void rgfxQueueTests(const char *self);
int rgfxQueueDiesCase(const char *name);
int rgfxQueueTimings(int rounds);

@interface QueueTestHandle : RgfxResource <MTLBuffer, MTLTexture>
@end

// The command-queue SPI WindowServer sets its queues' completion queues with.
@protocol QueueTestCompletionQueueSPI
- (void)setCompletionQueue:(dispatch_queue_t)queue;
- (dispatch_queue_t)completionQueue;
@end

@implementation QueueTestHandle
@end

// A handle that answers its pixel format, as a fill from a color asks the texture's.
@interface QueueTestFormatHandle : QueueTestHandle
@end

@implementation QueueTestFormatHandle
- (MTLPixelFormat)pixelFormat
{
	return MTLPixelFormatBGRA8Unorm;
}
@end

// A handle that answers the resource SPI as the bundle's buffers and textures do
// (RGFX_RESOURCE_SPI).
@interface QueueTestSpiHandle : QueueTestHandle
- (BOOL)isComplete;
- (BOOL)isWriteComplete;
- (void)waitUntilComplete;
@end

@implementation QueueTestSpiHandle
RGFX_RESOURCE_SPI
@end

// The command-buffer SPI the guest's frameworks send (MTLCommandBufferSPI), as the AMD 5300M's and
// the Intel 630's buffers answer it.
@protocol QueueTestCommandBufferSPI
- (void)__waitUntilScheduledAsync:(void (^)(id<MTLCommandBuffer>))block;
- (void)__waitUntilCompletedAsync:(void (^)(id<MTLCommandBuffer>))block;
- (id)privateData;
- (void)setPrivateData:(id)privateData;
- (NSUInteger)privateDataOffset;
- (void)setPrivateDataOffset:(NSUInteger)offset;
- (id)privateLoggingBuffer;
- (void)setPrivateLoggingBuffer:(id)buffer;
- (BOOL)sharedIndirectionTable;
- (void)setSharedIndirectionTable:(BOOL)shared;
- (BOOL)isProfilingEnabled;
- (void)setProfilingEnabled:(BOOL)enabled;
- (NSDictionary *)profilingResults;
- (NSUInteger)getListIndex;
- (id)deadlineProfile;
- (void *)debugBufferContentsWithLength:(NSUInteger *)length;
- (void)setLogs:(id)logs;
- (id)sampledBlitCommandEncoder:(uintptr_t)programInfoBuffer capacity:(NSUInteger)capacity;
- (id)sampledComputeCommandEncoderWithProgramInfoBuffer:(uintptr_t)programInfoBuffer
                                               capacity:(NSUInteger)capacity;
- (void)configWithCommandBufferDescriptor:(MTLCommandBufferDescriptor *)descriptor;
- (void)setResponsibleTaskIDs:(const int *)taskIDs count:(NSUInteger)count;
- (void)useInternalResidencySet:(id)residencySet;
- (void)useInternalResidencySets:(const id _Nonnull[_Nonnull])residencySets count:(NSUInteger)count;
@end

// The command-buffer SPI an Apple GPU's buffers answer, and an arm64 guest's.
@protocol QueueTestScheduledSignalSPI
- (void)encodeSignalEventScheduled:(id<MTLEvent>)event value:(uint64_t)value;
@end

// The command-queue SPI members the queue answers (RgfxCommandQueue, MTLCommandQueueSPI).
@protocol QueueTestCommandQueueSPI
- (NSUInteger)qosLevel;
- (dispatch_queue_t)commitQueue;
- (BOOL)commitSynchronously;
- (BOOL)commitsWithQoS;
- (BOOL)disableCrossQueueHazardTracking;
- (BOOL)isOpenGLQueue;
- (dispatch_queue_t)completionQueue;
- (int)backgroundTrackingPID;
- (void)setBackgroundTrackingPID:(int)pid;
- (BOOL)executionEnabled;
- (BOOL)skipRender;
- (BOOL)isProfilingEnabled;
- (void)setProfilingEnabled:(BOOL)enabled;
- (BOOL)isStatEnabled;
- (void)setStatOptions:(NSUInteger)options;
- (void)setStatLocations:(NSUInteger)locations;
- (NSUInteger)StatOptions;
- (NSUInteger)getStatOptions;
- (NSUInteger)StatLocations;
- (NSUInteger)getStatLocations;
- (NSUInteger)numRequestedCounters;
- (void)setNumRequestedCounters:(NSUInteger)count;
- (NSUInteger)numInternalSampleCounters;
- (void)setNumInternalSampleCounters:(NSUInteger)count;
- (NSMutableDictionary *)counterInfo;
- (void)setCounterInfo:(NSMutableDictionary *)info;
- (NSArray *)availableCounters;
- (NSArray *)getRequestedCounters;
- (int)requestCounters:(NSArray *)counters;
- (NSArray *)subdivideCounterList:(NSArray *)list;
- (void)dispatchAvailableCompletionNotifications;
- (void)finish;
@end

// The command-queue SPI's priority members (MTLCommandQueueSPI), and the queue's GPUPriority.
@protocol QueueTestPrioritySPI
- (BOOL)setGPUPriority:(NSUInteger)priority;
- (BOOL)setGPUPriority:(NSUInteger)priority offset:(unsigned short)offset;
- (NSUInteger)GPUPriority;
- (NSUInteger)getGPUPriority;
- (BOOL)setBackgroundGPUPriority:(NSUInteger)priority;
- (BOOL)setBackgroundGPUPriority:(NSUInteger)priority offset:(unsigned short)offset;
- (NSUInteger)getBackgroundGPUPriority;
@end

// The setters of Metal's queue descriptor's internal subclass, which the Mac's descriptor is.
@protocol QueueTestQueueDescriptorSPI
- (void)setQosLevel:(NSUInteger)level;
- (void)setCommitQueue:(dispatch_queue_t)queue;
- (void)setCommitSynchronously:(BOOL)value;
- (void)setCommitsWithQoS:(BOOL)value;
- (void)setDisableCrossQueueHazardTracking:(BOOL)value;
- (void)setIsOpenGLQueue:(BOOL)value;
- (void)setCompletionQueue:(dispatch_queue_t)queue;
@end

// The members of Metal's encoders' SPI the bundle's encoders answer (src/RgfxCommandObjects.mm).
@protocol QueueTestEncoderSPI
- (id)endEncodingAndRetrieveProgramAddressTable;
- (void)insertSplit;
@end
@protocol QueueTestRenderEncoderSPI <QueueTestEncoderSPI>
- (void)addSplitHandler:(void (^)(void))handler;
- (void)setToolsDispatchBufferSPI:(NSUInteger)buffer
                          atIndex:(NSUInteger)index
                           stages:(NSUInteger)stages;
- (void)setVertexAmplificationMode:(NSUInteger)mode value:(NSUInteger)value;
@end
@protocol QueueTestComputeEncoderSPI <QueueTestEncoderSPI>
- (void)encodeStartIf:(id<MTLBuffer>)buffer
                offset:(NSUInteger)offset
            comparison:(MTLCompareFunction)comparison
        referenceValue:(uint32_t)referenceValue;
- (void)encodeStartElse;
- (BOOL)encodeEndIf;
- (void)encodeStartDoWhile;
- (BOOL)encodeEndDoWhile:(id<MTLBuffer>)buffer
                  offset:(NSUInteger)offset
              comparison:(MTLCompareFunction)comparison
          referenceValue:(uint32_t)referenceValue;
- (void)setThreadgroupDistributionMode:(NSInteger)mode;
- (void)setThreadgroupDistributionModeWithClusterGroupIndex:(uint32_t)index;
- (void)setThreadgroupPackingDisabled:(BOOL)disabled;
- (void)setToolsDispatchBufferSPI:(NSUInteger)buffer atIndex:(NSUInteger)index;
- (id)newKernelDebugInfo;
@end
@protocol QueueTestBlitEncoderSPI <QueueTestEncoderSPI>
- (void)invalidateCompressedTexture:(id<MTLTexture>)texture;
- (void)invalidateCompressedTexture:(id<MTLTexture>)texture
                              slice:(NSUInteger)slice
                              level:(NSUInteger)level;
- (void)optimizeContentsForTexture:(id<MTLTexture>)texture
                 readAccessPattern:(NSInteger)readAccessPattern
                      readAccessor:(NSInteger)readAccessor;
- (void)optimizeContentsForTexture:(id<MTLTexture>)texture
                 readAccessPattern:(NSInteger)readAccessPattern
                      readAccessor:(NSInteger)readAccessor
                             slice:(NSUInteger)slice
                             level:(NSUInteger)level;
- (void)fillBuffer:(id<MTLBuffer>)buffer range:(NSRange)range pattern4:(uint32_t)pattern;
- (void)fillTexture:(id<MTLTexture>)texture
              level:(NSUInteger)level
              slice:(NSUInteger)slice
             region:(MTLRegion)region
              bytes:(const void *)bytes
             length:(NSUInteger)length;
- (void)fillTexture:(id<MTLTexture>)texture
              level:(NSUInteger)level
              slice:(NSUInteger)slice
             region:(MTLRegion)region
              color:(MTLClearColor)color;
- (void)fillTexture:(id<MTLTexture>)texture
              level:(NSUInteger)level
              slice:(NSUInteger)slice
             region:(MTLRegion)region
              color:(MTLClearColor)color
        pixelFormat:(MTLPixelFormat)pixelFormat;
@end
@protocol QueueTestParallelEncoderSPI <QueueTestEncoderSPI>
- (id)sampledRenderCommandEncoderWithProgramInfoBuffer:(void *)programInfoBuffer
                                               capacity:(NSUInteger)capacity;
- (BOOL)separateCommits;
- (void)setSeparateCommits:(BOOL)separateCommits;
@end
// The function SPI member every x86 Mac's function ends the process at.
@protocol QueueTestFunctionSPI
- (id)functionHandle;
@end

namespace {

const char *g_self = nullptr;
constexpr int kAborted = 128 + SIGABRT;

QueueTestHandle *handle(uint64_t objId)
{
	return [[QueueTestHandle alloc] initWithObjId:objId device:nil serializer:nullptr];
}

// A queue over test rings, with or without a reader of its completion ring. The rings and the
// serialiser are never freed: objects a test lets go may outlive it and post their destroys.
struct Queue {
	TestRing &control = *new TestRing(0x10000);
	RgfxRingWriter &controlWriter = *new RgfxRingWriter(
	    control.base(), control.dataSize, RGFX_CONTROL_RING_ID, TestRing::doorbell, nullptr);
	RgfxIdSpace &ids = *new RgfxIdSpace();
	RgfxSerializer &ser = *new RgfxSerializer(controlWriter, ids);
	TestRing &submit = *new TestRing(0x8000);
	RgfxRingWriter &submitWriter =
	    *new RgfxRingWriter(submit.base(), submit.dataSize, 1, TestRing::doorbell, nullptr);
	TestRing &completion = *new TestRing(0x8000);
	RgfxCommandQueue *queue = nil;

	explicit Queue(bool reader = false, NSUInteger maxCount = 0)
	{
		control.setConsumerIdle(1);
		submit.setConsumerIdle(1);
		queue = [RgfxCommandQueue queueWithSubmitRing:&submitWriter
		                                 submitOffset:0
		                               completeOffset:0
		                             completeDataSize:0
		                                          ids:&ids
		                                     maxCount:maxCount
		                                       device:nil
		                                   serializer:&ser];
		if (reader)
			[queue adoptReplyReader:new RgfxReplyReader(completion.base(), completion.dataSize)];
	}
};

// The seqno of the next command buffer posted on `q`'s submission ring, waiting up to 5 s for one
// another thread posts.
uint64_t nextSeqno(Queue &q)
{
	RgfxMsgHeader h = {};
	std::vector<uint8_t> bytes;
	bool got = q.submit.next(&h, &bytes);
	for (int i = 0; i < 5000 && !got; i++) {
		std::this_thread::sleep_for(std::chrono::milliseconds(1));
		got = q.submit.next(&h, &bytes);
	}
	CHECK(got && h.type == RGFX_CMD_COMMAND_BUFFER);
	return h.seqno;
}

// The kext's wait the completion thread sleeps in: a wake count the host's post moves.
struct FakeKext {
	std::mutex lock;
	std::condition_variable changed;
	uint64_t count = 0;
	bool closed = false;
	void wake()
	{
		std::lock_guard<std::mutex> hold(lock);
		count++;
		changed.notify_all();
	}
	void close()
	{
		std::lock_guard<std::mutex> hold(lock);
		closed = true;
		changed.notify_all();
	}
	static RgfxCompletionThread::WaitResult wait(void *context, uint64_t last, uint64_t *now)
	{
		FakeKext *k = static_cast<FakeKext *>(context);
		std::unique_lock<std::mutex> hold(k->lock);
		k->changed.wait(hold, [&] { return k->count != last || k->closed; });
		*now = k->count;
		return k->closed ? RgfxCompletionThread::kStop : RgfxCompletionThread::kWoke;
	}
};

// The host's post (ring.rs Producer::post): the record published, a full fence, then `armed`
// read and cleared, the vector raised where it was set.
void hostPost(TestRing &ring, FakeKext &kext, const void *msg, uint32_t size)
{
	CHECK(ring.postFromHost(msg, size));
	__atomic_thread_fence(__ATOMIC_SEQ_CST);
	uint32_t *armed =
	    reinterpret_cast<uint32_t *>(ring.mem.data() + __builtin_offsetof(RgfxRingHeader, armed));
	if (__atomic_exchange_n(armed, 0u, __ATOMIC_SEQ_CST) == 1)
		kext.wake();
}

void hostComplete(Queue &q, FakeKext &kext, uint64_t seqno, id<MTLCommandBuffer> cb)
{
	RgfxCbComplete done = {};
	done.header.type = RGFX_EVT_CB_COMPLETE;
	done.header.size = sizeof(done);
	done.header.seqno = seqno;
	done.cb_obj_id = ((RgfxResource *)cb).rgfxObjId;
	done.status = RGFX_CB_OK;
	hostPost(q.completion, kext, &done, sizeof(done));
}

void hostScheduled(Queue &q, FakeKext &kext, uint64_t seqno, id<MTLCommandBuffer> cb)
{
	RgfxCbScheduled scheduled = {};
	scheduled.header.type = RGFX_EVT_CB_SCHEDULED;
	scheduled.header.size = sizeof(scheduled);
	scheduled.header.seqno = seqno;
	scheduled.cb_obj_id = ((RgfxResource *)cb).rgfxObjId;
	hostPost(q.completion, kext, &scheduled, sizeof(scheduled));
}

// A completion thread driving `queues`' readers, over a kext of its own; stopped at the end.
struct Driven {
	FakeKext *kext = new FakeKext(); // outlives the thread, which may sleep in it
	RgfxCompletionThread thread{&FakeKext::wait, kext};
	explicit Driven(std::initializer_list<Queue *> queues)
	{
		for (Queue *q : queues)
			thread.add([q->queue sharedReplyReader]);
		CHECK(thread.start());
	}
	~Driven()
	{
		kext->close();
		for (int i = 0; i < 5000 && thread.running(); i++)
			std::this_thread::sleep_for(std::chrono::milliseconds(1));
	}
};

long waitFor(dispatch_semaphore_t s, double seconds)
{
	return dispatch_semaphore_wait(
	    s, dispatch_time(DISPATCH_TIME_NOW, static_cast<int64_t>(seconds * NSEC_PER_SEC)));
}

MTLRenderPassDescriptor *pass()
{
	MTLRenderPassDescriptor *rp = [MTLRenderPassDescriptor renderPassDescriptor];
	rp.colorAttachments[0].texture = handle(42);
	return rp;
}

// An encoder keeps its command buffer, and the stream it writes, until its endEncoding, as
// Metal's does: a buffer made in a pool that drains lives on while its open encoder does, and goes
// once the encoder ends, though the encoder is kept; the ended encoder's commandBuffer is then
// nil. Each kind of encoder, and a parallel pass's sub-encoder.
void test_encoder_keeps_its_buffer_until_it_ends()
{
	Queue q;
	auto check = [&](id<MTLCommandEncoder> (^make)(id<MTLCommandBuffer>)) {
		__weak id<MTLCommandBuffer> weakCB = nil;
		id<MTLCommandEncoder> enc = nil;
		@autoreleasepool {
			id<MTLCommandBuffer> cb = [q.queue commandBuffer];
			weakCB = cb;
			enc = make(cb);
		}
		CHECK(weakCB != nil);
		CHECK([(id)enc commandBuffer] == weakCB);
		// Encoding through the kept encoder writes the live buffer's stream.
		if ([(id)enc conformsToProtocol:@protocol(MTLBlitCommandEncoder)])
			[(id<MTLBlitCommandEncoder>)enc fillBuffer:handle(7) range:NSMakeRange(0, 4) value:1];
		@autoreleasepool {
			[enc endEncoding];
		}
		CHECK(weakCB == nil);
		CHECK([(id)enc commandBuffer] == nil);
	};
	check(^id<MTLCommandEncoder>(id<MTLCommandBuffer> cb) {
		return [cb renderCommandEncoderWithDescriptor:pass()];
	});
	check(^id<MTLCommandEncoder>(id<MTLCommandBuffer> cb) {
		return [cb computeCommandEncoder];
	});
	check(^id<MTLCommandEncoder>(id<MTLCommandBuffer> cb) {
		return [cb blitCommandEncoder];
	});
	check(^id<MTLCommandEncoder>(id<MTLCommandBuffer> cb) {
		return [cb parallelRenderCommandEncoderWithDescriptor:pass()];
	});

	// A parallel pass's sub-encoder keeps the buffer until its end, and the pass until its own.
	__weak id<MTLCommandBuffer> weakCB = nil;
	id<MTLParallelRenderCommandEncoder> pe = nil;
	id<MTLRenderCommandEncoder> sub = nil;
	@autoreleasepool {
		id<MTLCommandBuffer> cb = [q.queue commandBuffer];
		weakCB = cb;
		pe = [cb parallelRenderCommandEncoderWithDescriptor:pass()];
		sub = [pe renderCommandEncoder];
	}
	CHECK(weakCB != nil && [(id)sub commandBuffer] == weakCB);
	@autoreleasepool {
		[sub drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
		[sub endEncoding];
	}
	CHECK(weakCB != nil);
	@autoreleasepool {
		[pe endEncoding];
	}
	CHECK(weakCB == nil);
}

// Whether `body` raises an unrecognized selector, as Metal's base encoders do for the SPI members
// they leave to their subclasses.
bool raisesUnrecognized(void (^body)(void))
{
	@try {
		body();
	} @catch (NSException *e) {
		return [e.name isEqualToString:NSInvalidArgumentException] &&
		       [e.reason containsString:@"unrecognized selector"];
	}
	return false;
}

// The encoders' SPI members (src/RgfxCommandObjects.mm): what Metal's base encoders raise an
// unrecognized selector for, these raise; the ones that end an encoding end it, answering no
// program address table, and so release the buffer, as endEncoding does; the conditionals and
// loops the GPU would evaluate close as none was open; a split handler is never called; and a
// parallel pass makes no sampling sub-encoder.
void test_encoder_spi_members()
{
	Queue q;
	auto ends = [&](id (^make)(id<MTLCommandBuffer>)) {
		__weak id<MTLCommandBuffer> weakCB = nil;
		id enc = nil;
		@autoreleasepool {
			id<MTLCommandBuffer> cb = [q.queue commandBuffer];
			weakCB = cb;
			enc = make(cb);
		}
		CHECK(weakCB != nil);
		id table = @"not asked";
		@autoreleasepool {
			table = [(id<QueueTestEncoderSPI>)enc endEncodingAndRetrieveProgramAddressTable];
		}
		CHECK(table == nil);
		CHECK(weakCB == nil);
	};
	ends(^id(id<MTLCommandBuffer> cb) {
		return [cb renderCommandEncoderWithDescriptor:pass()];
	});
	ends(^id(id<MTLCommandBuffer> cb) {
		return [cb computeCommandEncoder];
	});
	ends(^id(id<MTLCommandBuffer> cb) {
		return [cb blitCommandEncoder];
	});
	ends(^id(id<MTLCommandBuffer> cb) {
		return [cb parallelRenderCommandEncoderWithDescriptor:pass()];
	});

	@autoreleasepool {
		id<MTLCommandBuffer> cb = [q.queue commandBuffer];
		id<QueueTestRenderEncoderSPI> render =
		    (id<QueueTestRenderEncoderSPI>)[cb renderCommandEncoderWithDescriptor:pass()];
		__block bool called = false;
		[render addSplitHandler:^{
			called = true;
		}];
		CHECK(raisesUnrecognized(^{
			[render insertSplit];
		}));
		CHECK(raisesUnrecognized(^{
			[render setToolsDispatchBufferSPI:0 atIndex:0 stages:1];
		}));
		[render setVertexAmplificationMode:1 value:2];
		[(id<MTLRenderCommandEncoder>)render endEncoding];
		CHECK(!called);

		id<QueueTestComputeEncoderSPI> compute =
		    (id<QueueTestComputeEncoderSPI>)[cb computeCommandEncoder];
		[compute encodeStartIf:handle(7) offset:0 comparison:MTLCompareFunctionEqual referenceValue:1];
		[compute encodeStartElse];
		CHECK(![compute encodeEndIf]);
		[compute encodeStartDoWhile];
		CHECK(![compute encodeEndDoWhile:handle(7)
		                          offset:0
		                      comparison:MTLCompareFunctionEqual
		                  referenceValue:1]);
		[compute setThreadgroupDistributionMode:1];
		[compute setThreadgroupDistributionModeWithClusterGroupIndex:2];
		CHECK(raisesUnrecognized(^{
			[compute setThreadgroupPackingDisabled:YES];
		}));
		CHECK(raisesUnrecognized(^{
			[compute setToolsDispatchBufferSPI:0 atIndex:0];
		}));
		CHECK(raisesUnrecognized(^{
			[compute insertSplit];
		}));
		CHECK([compute newKernelDebugInfo] == nil);
		[(id<MTLComputeCommandEncoder>)compute endEncoding];

		id<QueueTestBlitEncoderSPI> blit = (id<QueueTestBlitEncoderSPI>)[cb blitCommandEncoder];
		id<MTLTexture> texture = (id<MTLTexture>)handle(42);
		CHECK(raisesUnrecognized(^{
			[blit invalidateCompressedTexture:texture];
		}));
		CHECK(raisesUnrecognized(^{
			[blit invalidateCompressedTexture:texture slice:0 level:0];
		}));
		CHECK(raisesUnrecognized(^{
			[blit optimizeContentsForTexture:texture readAccessPattern:0 readAccessor:0];
		}));
		CHECK(raisesUnrecognized(^{
			[blit optimizeContentsForTexture:texture readAccessPattern:0 readAccessor:0 slice:0 level:0];
		}));
		CHECK(raisesUnrecognized(^{
			[blit insertSplit];
		}));
		// Each fill raises, those from a color at the fill from bytes they come to.
		const MTLRegion region = MTLRegionMake2D(0, 0, 1, 1);
		const uint32_t word = 0;
		CHECK(raisesUnrecognized(^{
			[blit fillBuffer:(id<MTLBuffer>)handle(43) range:NSMakeRange(0, 4) pattern4:7];
		}));
		CHECK(raisesUnrecognized(^{
			[blit fillTexture:texture level:0 slice:0 region:region bytes:&word length:4];
		}));
		id<MTLTexture> formatted = (id<MTLTexture>)[[QueueTestFormatHandle alloc] initWithObjId:44
		                                                                                 device:nil
		                                                                             serializer:nullptr];
		for (int withFormat = 0; withFormat < 2; withFormat++) {
			NSString *reason = nil;
			@try {
				if (withFormat)
					[blit fillTexture:formatted
					            level:0
					            slice:0
					           region:region
					            color:MTLClearColorMake(1, 0, 0, 1)
					      pixelFormat:MTLPixelFormatBGRA8Unorm];
				else
					[blit fillTexture:formatted
					            level:0
					            slice:0
					           region:region
					            color:MTLClearColorMake(1, 0, 0, 1)];
			} @catch (NSException *e) {
				reason = e.reason;
			}
			CHECK([reason containsString:@"fillTexture:level:slice:region:bytes:length:"]);
		}
		[(id<MTLBlitCommandEncoder>)blit endEncoding];

		id<QueueTestParallelEncoderSPI> parallel =
		    (id<QueueTestParallelEncoderSPI>)[cb parallelRenderCommandEncoderWithDescriptor:pass()];
		CHECK([parallel sampledRenderCommandEncoderWithProgramInfoBuffer:nullptr capacity:0] == nil);
		CHECK(![parallel separateCommits]);
		[parallel setSeparateCommits:YES];
		CHECK(![parallel separateCommits]);
		CHECK(raisesUnrecognized(^{
			[parallel insertSplit];
		}));
		[(id<MTLParallelRenderCommandEncoder>)parallel endEncoding];
	}
}

// Runs the case `name` of rgfxQueueDiesCase in a process of its own: it must end the process
// (SIGABRT) with `words` on stderr.
void checkDies(const char *name, const char *words)
{
	int fds[2];
	CHECK(pipe(fds) == 0);
	posix_spawn_file_actions_t fa;
	posix_spawn_file_actions_init(&fa);
	posix_spawn_file_actions_adddup2(&fa, fds[1], 2);
	posix_spawn_file_actions_addclose(&fa, fds[0]);
	char *args[] = {const_cast<char *>(g_self), const_cast<char *>("--queue-dies"),
	                const_cast<char *>(name), nullptr};
	pid_t pid = 0;
	CHECK(posix_spawn(&pid, g_self, &fa, nullptr, args, *_NSGetEnviron()) == 0);
	posix_spawn_file_actions_destroy(&fa);
	close(fds[1]);
	std::string err;
	char chunk[512];
	ssize_t n;
	while ((n = read(fds[0], chunk, sizeof chunk)) > 0)
		err.append(chunk, static_cast<size_t>(n));
	close(fds[0]);
	int status = 0;
	CHECK(waitpid(pid, &status, 0) == pid);
	const bool aborted = WIFEXITED(status) && WEXITSTATUS(status) == kAborted;
	if (!aborted || err.find(words) == std::string::npos)
		std::fprintf(stderr, "--queue-dies %s ended with status %d:\n%s\n", name, status,
		             err.c_str());
	CHECK(aborted);
	CHECK(err.find(words) != std::string::npos);
}

// The host's completion of the one command buffer `q` holds on its submission ring: posted on
// its completion ring and taken off it (the reader is undriven, so the taker delivers).
void complete(Queue &q, id<MTLCommandBuffer> cb)
{
	RgfxMsgHeader h = {};
	std::vector<uint8_t> bytes;
	CHECK(q.submit.next(&h, &bytes) && h.type == RGFX_CMD_COMMAND_BUFFER);
	RgfxCbComplete done = {};
	done.header.type = RGFX_EVT_CB_COMPLETE;
	done.header.size = sizeof(done);
	done.header.seqno = h.seqno;
	done.cb_obj_id = ((RgfxResource *)cb).rgfxObjId;
	done.status = RGFX_CB_OK;
	CHECK(q.completion.postFromHost(&done, sizeof(done)));
	[cb waitUntilCompleted];
	CHECK(cb.status == MTLCommandBufferStatusCompleted);
}

// A handler added after the commit ends the process at the call with Metal's words, before the
// buffer completes and after, as Metal does; so does presentDrawable: after the commit, which
// adds a scheduled handler. (It used to race the delivering thread's walk of the list, or be
// dropped unrun.)
void test_handlers_after_commit_end_the_process()
{
	const char *completed = "-[_MTLCommandBuffer addCompletedHandler:]: failed assertion "
	                        "`Completed handler provided after commit call'";
	const char *scheduled = "-[_MTLCommandBuffer addScheduledHandler:]: failed assertion "
	                        "`Scheduled handler provided after commit call'";
	checkDies("completed-after-commit", completed);
	checkDies("completed-after-completion", completed);
	checkDies("scheduled-after-commit", scheduled);
	checkDies("scheduled-after-completion", scheduled);
	checkDies("present-after-commit", scheduled);
	// A function's functionHandle, as Metal's base function's on every x86 Mac.
	checkDies("function-handle",
	          "-[_MTLFunction functionHandle]: failed assertion `Driver needs to override this'");

	// Added before the commit, each runs once, in the order added.
	Queue q(true);
	id<MTLCommandBuffer> cb = [q.queue commandBuffer];
	__block std::string order;
	[cb addScheduledHandler:^(id<MTLCommandBuffer> c) {
		(void)c;
		order += "s";
	}];
	[cb addCompletedHandler:^(id<MTLCommandBuffer> c) {
		(void)c;
		order += "1";
	}];
	[cb addCompletedHandler:^(id<MTLCommandBuffer> c) {
		(void)c;
		order += "2";
	}];
	[cb commit];
	complete(q, cb);
	CHECK(order == "s12");
}

// A completed handler of one queue that waits for another queue's completed handler, or for its
// buffer's completion, is released, as Metal's is on all three GPUs: each queue's handlers run on
// a completion queue of its own, so one queue's waiting handler holds back no other queue's.
void test_handlers_of_queues_wait_for_each_other()
{
	@autoreleasepool {
		Queue q1(true), q2(true);
		Driven driven({&q1, &q2});
		dispatch_semaphore_t fromQ2 = dispatch_semaphore_create(0);
		dispatch_semaphore_t done = dispatch_semaphore_create(0);
		__block long waited = -1;
		__block long completedSeen = -1;
		id<MTLCommandBuffer> a = [q1.queue commandBuffer];
		id<MTLCommandBuffer> b = [q2.queue commandBuffer];
		id<MTLCommandBuffer> c = [q2.queue commandBuffer];
		[a addCompletedHandler:^(id<MTLCommandBuffer> x) {
			(void)x;
			waited = waitFor(fromQ2, 3);
			[c waitUntilCompleted];
			completedSeen = (long)c.status;
			dispatch_semaphore_signal(done);
		}];
		[b addCompletedHandler:^(id<MTLCommandBuffer> x) {
			(void)x;
			dispatch_semaphore_signal(fromQ2);
		}];
		[a commit];
		const uint64_t sa = nextSeqno(q1);
		[b commit];
		const uint64_t sb = nextSeqno(q2);
		[c commit];
		const uint64_t sc = nextSeqno(q2);
		hostComplete(q1, *driven.kext, sa, a);
		std::this_thread::sleep_for(std::chrono::milliseconds(20)); // a's handler waits
		hostComplete(q2, *driven.kext, sb, b);
		hostComplete(q2, *driven.kext, sc, c);
		CHECK(waitFor(done, 5) == 0);
		CHECK(waited == 0);
		CHECK(completedSeen == MTLCommandBufferStatusCompleted);
	}
}

// After setCompletionQueue:, the queue's scheduled and completed handlers run on the dispatch
// queue it gave, in the ring's order, as Metal's do (WindowServer sets one on its queues); before
// it, on the queue's own (labelled com.Metal.CompletionQueueDispatch, as Metal's), and never on
// the completion thread. completionQueue answers the queue given.
void test_handlers_run_on_the_completion_queue()
{
	@autoreleasepool {
		Queue q(true);
		Driven driven({&q});
		static char key;
		dispatch_queue_t given = dispatch_queue_create("test.completion", DISPATCH_QUEUE_SERIAL);
		dispatch_queue_set_specific(given, &key, &key, nullptr);
		CHECK([(id<QueueTestCompletionQueueSPI>)q.queue completionQueue] == nil);

		std::mutex lock;
		std::mutex *logLock = &lock;
		std::vector<std::string> log;
		std::vector<std::string> *logp = &log;
		auto where = [] {
			if (RgfxCompletionThread::onCurrentThread())
				return std::string("thread");
			if (dispatch_get_specific(&key) == &key)
				return std::string("given");
			const char *label = dispatch_queue_get_label(DISPATCH_CURRENT_QUEUE_LABEL);
			return std::string(std::strcmp(label, "com.Metal.CompletionQueueDispatch") == 0 ? "own"
			                                                                              : label);
		};
		auto make = [&](const char *name) {
			id<MTLCommandBuffer> cb = [q.queue commandBuffer];
			std::string n = name;
			[cb addScheduledHandler:^(id<MTLCommandBuffer> x) {
				(void)x;
				std::lock_guard<std::mutex> hold(*logLock);
				logp->push_back("s" + n + "@" + where());
			}];
			[cb addCompletedHandler:^(id<MTLCommandBuffer> x) {
				(void)x;
				std::lock_guard<std::mutex> hold(*logLock);
				logp->push_back("c" + n + "@" + where());
			}];
			[cb commit];
			return cb;
		};
		id<MTLCommandBuffer> before = make("0");
		const uint64_t s0 = nextSeqno(q);
		hostScheduled(q, *driven.kext, s0, before);
		hostComplete(q, *driven.kext, s0, before);
		[before waitUntilCompleted];

		[(id<QueueTestCompletionQueueSPI>)q.queue setCompletionQueue:given];
		CHECK([(id<QueueTestCompletionQueueSPI>)q.queue completionQueue] == given);
		id<MTLCommandBuffer> first = make("1");
		const uint64_t s1 = nextSeqno(q);
		id<MTLCommandBuffer> second = make("2");
		const uint64_t s2 = nextSeqno(q);
		hostScheduled(q, *driven.kext, s1, first);
		hostComplete(q, *driven.kext, s1, first);
		hostComplete(q, *driven.kext, s2, second); // no CB_SCHEDULED: the completion stands for it
		[second waitUntilCompleted];
		std::lock_guard<std::mutex> hold(lock);
		std::vector<std::string> want = {"s0@own",   "c0@own",   "s1@given",
		                                 "c1@given", "s2@given", "c2@given"};
		CHECK(log == want);
	}
}

// A buffer with no handler and nothing to write back completes on the thread that takes its
// records off the ring, which wakes its waiter, as Metal's does for a buffer with no completed
// handler: its wait returns while the queue's completion queue runs nothing. One with a handler
// still runs there, and a buffer with no handler behind one whose record is still to run
// completes after it, in the ring's order, its status final before its wait returns.
void test_buffers_without_handlers_complete_where_seen()
{
	@autoreleasepool {
		Queue q(true);
		Driven driven({&q});
		dispatch_queue_t given = dispatch_queue_create("test.completion", DISPATCH_QUEUE_SERIAL);
		[(id<QueueTestCompletionQueueSPI>)q.queue setCompletionQueue:given];
		dispatch_suspend(given);
		bool resumed = false;
		auto within = [](const std::function<bool()> &pred) {
			for (int i = 0; i < 5000 && !pred(); i++)
				std::this_thread::sleep_for(std::chrono::milliseconds(1));
			return pred();
		};

		// No handler: the wait returns, Completed, with the completion queue suspended.
		id<MTLCommandBuffer> plain = [q.queue commandBuffer];
		[plain commit];
		const uint64_t s0 = nextSeqno(q);
		std::atomic<bool> waited{false};
		std::atomic<bool> *waitedp = &waited;
		MTLCommandBufferStatus atReturn = MTLCommandBufferStatusNotEnqueued;
		std::thread waiter([plain, waitedp, &atReturn] {
			[plain waitUntilCompleted];
			atReturn = plain.status;
			waitedp->store(true);
		});
		hostScheduled(q, *driven.kext, s0, plain);
		hostComplete(q, *driven.kext, s0, plain);
		const bool seen = within([&] { return waited.load(); });
		CHECK(seen);
		if (!seen) {
			dispatch_resume(given);
			resumed = true;
		}
		waiter.join();
		CHECK(atReturn == MTLCommandBufferStatusCompleted);

		// A handler: its record waits for the queue, and so does a buffer with none behind it.
		__block MTLCommandBufferStatus behindAtHandler = MTLCommandBufferStatusCompleted;
		id<MTLCommandBuffer> handled = [q.queue commandBuffer];
		id<MTLCommandBuffer> behind = [q.queue commandBuffer];
		__unsafe_unretained id<MTLCommandBuffer> behindRaw = behind; // alive through the test
		[handled addCompletedHandler:^(id<MTLCommandBuffer> x) {
			(void)x;
			behindAtHandler = behindRaw.status;
		}];
		[handled commit];
		const uint64_t s1 = nextSeqno(q);
		[behind commit];
		const uint64_t s2 = nextSeqno(q);
		hostComplete(q, *driven.kext, s1, handled);
		hostComplete(q, *driven.kext, s2, behind);
		// Both taken off the ring; neither delivered while the queue runs nothing.
		CHECK(within([&] { return q.completion.consumerCount() == q.completion.produced; }));
		std::this_thread::sleep_for(std::chrono::milliseconds(20));
		CHECK(resumed || handled.status != MTLCommandBufferStatusCompleted);
		CHECK(resumed || behind.status != MTLCommandBufferStatusCompleted);
		if (!resumed)
			dispatch_resume(given);
		[behind waitUntilCompleted];
		CHECK(handled.status == MTLCommandBufferStatusCompleted);
		CHECK(behindAtHandler != MTLCommandBufferStatusCompleted);
		CHECK(behind.status == MTLCommandBufferStatusCompleted);
	}
}

// A buffer with a completed handler and no scheduled handler is scheduled on the thread that takes
// its CB_SCHEDULED off the ring: Scheduled, and its waitUntilScheduled returned, while the queue's
// completion queue runs nothing, its completed handler still to run there. Behind a record of its
// queue still to run, a later one's scheduling waits its turn, Committed until that record ran.
void test_scheduling_without_a_scheduled_handler_runs_where_seen()
{
	@autoreleasepool {
		Queue q(true);
		Driven driven({&q});
		dispatch_queue_t given = dispatch_queue_create("test.completion", DISPATCH_QUEUE_SERIAL);
		[(id<QueueTestCompletionQueueSPI>)q.queue setCompletionQueue:given];
		dispatch_suspend(given);
		bool resumed = false;
		auto within = [](const std::function<bool()> &pred) {
			for (int i = 0; i < 5000 && !pred(); i++)
				std::this_thread::sleep_for(std::chrono::milliseconds(1));
			return pred();
		};

		__block int firstRan = 0;
		__block MTLCommandBufferStatus laterAtFirst = MTLCommandBufferStatusNotEnqueued;
		id<MTLCommandBuffer> first = [q.queue commandBuffer];
		id<MTLCommandBuffer> later = [q.queue commandBuffer];
		__unsafe_unretained id<MTLCommandBuffer> laterRaw = later; // alive through the test
		[first addCompletedHandler:^(id<MTLCommandBuffer> x) {
			(void)x;
			laterAtFirst = laterRaw.status;
			firstRan = 1;
		}];
		[later addCompletedHandler:^(id<MTLCommandBuffer> x) {
			(void)x;
		}];
		[first commit];
		const uint64_t s0 = nextSeqno(q);
		[later commit];
		const uint64_t s1 = nextSeqno(q);
		std::atomic<bool> waited{false};
		std::atomic<bool> *waitedp = &waited;
		MTLCommandBufferStatus atReturn = MTLCommandBufferStatusNotEnqueued;
		std::thread waiter([first, waitedp, &atReturn] {
			[first waitUntilScheduled];
			atReturn = first.status;
			waitedp->store(true);
		});
		hostScheduled(q, *driven.kext, s0, first);
		const bool seen = within([&] { return waited.load(); });
		CHECK(seen);
		if (!seen) {
			dispatch_resume(given);
			resumed = true;
		}
		waiter.join();
		CHECK(atReturn == MTLCommandBufferStatusScheduled);
		CHECK(resumed || firstRan == 0);

		// The first's completion goes to the queue, still suspended; the later one's scheduling,
		// behind it, waits there too.
		hostComplete(q, *driven.kext, s0, first);
		hostScheduled(q, *driven.kext, s1, later);
		CHECK(within([&] { return q.completion.consumerCount() == q.completion.produced; }));
		std::this_thread::sleep_for(std::chrono::milliseconds(20));
		CHECK(resumed || later.status == MTLCommandBufferStatusCommitted);
		if (!resumed)
			dispatch_resume(given);
		[later waitUntilScheduled];
		CHECK(firstRan == 1);
		CHECK(laterAtFirst == MTLCommandBufferStatusCommitted);
		CHECK(later.status == MTLCommandBufferStatusScheduled);
		hostComplete(q, *driven.kext, s1, later);
		[later waitUntilCompleted];
		CHECK(later.status == MTLCommandBufferStatusCompleted);
	}
}

// One queue's records run one at a time, in the ring's order, the status flips among them: while
// an earlier buffer's completed handler runs, a later buffer whose CB_SCHEDULED came first is
// Scheduled, and one whose CB_SCHEDULED came after is still Committed, its scheduled handler
// unrun; the later buffer's handlers run once that one returns, and a wait for it returns only
// then, as Metal's does on all three GPUs. Another queue's handlers run meanwhile.
void test_one_queues_handlers_run_in_order()
{
	for (int scheduledFirst = 1; scheduledFirst >= 0; scheduledFirst--) {
		@autoreleasepool {
			Queue q1(true), q2(true);
			Driven driven({&q1, &q2});
			std::atomic<int> earlierDone{0};
			std::atomic<int> *earlierDonep = &earlierDone;
			std::atomic<int> laterScheduledRan{0};
			std::atomic<int> *laterScheduledRanp = &laterScheduledRan;
			__block long laterWhileEarlier = -1;
			__block int laterScheduledRanWhileEarlier = -1;
			__block int laterRanAfter = -1;
			__block int otherQueueDuring = -1;
			dispatch_semaphore_t otherRan = dispatch_semaphore_create(0);
			id<MTLCommandBuffer> earlier = [q1.queue commandBuffer];
			id<MTLCommandBuffer> later = [q1.queue commandBuffer];
			id<MTLCommandBuffer> other = [q2.queue commandBuffer];
			[earlier addCompletedHandler:^(id<MTLCommandBuffer> x) {
				(void)x;
				otherQueueDuring = waitFor(otherRan, 3) == 0;
				std::this_thread::sleep_for(std::chrono::milliseconds(50));
				laterWhileEarlier = (long)later.status;
				laterScheduledRanWhileEarlier = laterScheduledRanp->load();
				earlierDonep->store(1);
			}];
			[later addScheduledHandler:^(id<MTLCommandBuffer> x) {
				(void)x;
				laterScheduledRanp->store(1);
			}];
			[later addCompletedHandler:^(id<MTLCommandBuffer> x) {
				(void)x;
				laterRanAfter = earlierDonep->load();
			}];
			[other addCompletedHandler:^(id<MTLCommandBuffer> x) {
				(void)x;
				dispatch_semaphore_signal(otherRan);
			}];
			[earlier commit];
			const uint64_t se = nextSeqno(q1);
			[later commit];
			const uint64_t sl = nextSeqno(q1);
			[other commit];
			const uint64_t so = nextSeqno(q2);
			if (scheduledFirst)
				hostScheduled(q1, *driven.kext, sl, later);
			hostComplete(q1, *driven.kext, se, earlier);
			if (!scheduledFirst)
				hostScheduled(q1, *driven.kext, sl, later);
			hostComplete(q1, *driven.kext, sl, later);
			std::this_thread::sleep_for(std::chrono::milliseconds(10));
			hostComplete(q2, *driven.kext, so, other);
			[later waitUntilCompleted];
			CHECK(earlierDone.load() == 1); // the wait returned after the earlier handler
			CHECK(laterWhileEarlier == (scheduledFirst ? MTLCommandBufferStatusScheduled
			                                           : MTLCommandBufferStatusCommitted));
			CHECK(laterScheduledRanWhileEarlier == scheduledFirst);
			CHECK(laterRanAfter == 1);
			CHECK(otherQueueDuring == 1);
			CHECK(later.status == MTLCommandBufferStatusCompleted);
		}
	}
}

// The queue's slot goes back as the completion thread takes a buffer's completion off the ring,
// not when its handlers run: a completed handler that makes buffers while every slot is held gets
// one as a later buffer of its queue completes, though that buffer's handlers wait behind it. And
// a handler's commitAndWaitUntilSubmitted of a buffer of its own queue returns at the host's
// CB_SCHEDULED, as Metal's returns there.
void test_slots_and_submission_do_not_wait_for_handlers()
{
	@autoreleasepool {
		Queue q(true, 2);
		Driven driven({&q});
		RgfxCommandQueue *queue = q.queue;
		dispatch_semaphore_t made = dispatch_semaphore_create(0);
		dispatch_semaphore_t submitted = dispatch_semaphore_create(0);
		__block id<MTLCommandBuffer> first = nil, second = nil, third = nil;
		__block BOOL submittedOK = NO;
		id<MTLCommandBuffer> a = [queue commandBuffer];
		id<MTLCommandBuffer> b = [queue commandBuffer];
		[a addCompletedHandler:^(id<MTLCommandBuffer> x) {
			(void)x;
			first = [queue commandBuffer]; // a's own slot, back already
			dispatch_semaphore_signal(made);
			second = [queue commandBuffer]; // b's, back as its completion is taken
			dispatch_semaphore_signal(made);
			third = second;
			submittedOK = [(id<MTLCommandBufferSPI>)second commitAndWaitUntilSubmitted];
			dispatch_semaphore_signal(submitted);
		}];
		[a commit];
		const uint64_t sa = nextSeqno(q);
		[b commit];
		const uint64_t sb = nextSeqno(q);
		hostComplete(q, *driven.kext, sa, a);
		CHECK(waitFor(made, 5) == 0);
		CHECK(waitFor(made, 0.05) != 0); // both slots held: b's and first's
		hostComplete(q, *driven.kext, sb, b);
		CHECK(waitFor(made, 5) == 0);
		// The host takes `second` and schedules it while a's handler waits in its submission.
		const uint64_t s2 = nextSeqno(q);
		hostScheduled(q, *driven.kext, s2, third);
		CHECK(waitFor(submitted, 5) == 0);
		CHECK(submittedOK);
		hostComplete(q, *driven.kext, s2, third);
		[third waitUntilCompleted];
		CHECK(b.status == MTLCommandBufferStatusCompleted);
		(void)first;
	}
}

// The cb_obj_ids of the command buffers posted on `q`'s submission ring since the last call, in
// ring order.
std::vector<uint64_t> posted(Queue &q)
{
	std::vector<uint64_t> out;
	RgfxMsgHeader h = {};
	std::vector<uint8_t> bytes;
	while (q.submit.next(&h, &bytes)) {
		if (h.type != RGFX_CMD_COMMAND_BUFFER)
			continue;
		RgfxCommandBuffer cbh = {};
		std::memcpy(&cbh, bytes.data(), sizeof(cbh));
		out.push_back(cbh.cb_obj_id);
	}
	return out;
}

uint64_t objId(id<MTLCommandBuffer> cb)
{
	return ((RgfxResource *)cb).rgfxObjId;
}

// Command buffers run in the order they were enqueued, whatever order they are committed in, as
// Metal's do on all three GPUs: one committed before a buffer enqueued ahead of it is not posted,
// and stays Committed, until that one is committed; then both go, the earlier first. A commit
// with no enqueue enqueues the buffer there. An enqueued buffer let go uncommitted stays and
// holds back the buffers after it, as Metal's does.
void test_buffers_run_in_enqueue_order()
{
	@autoreleasepool {
		Queue q;
		id<MTLCommandBuffer> a = [q.queue commandBuffer];
		id<MTLCommandBuffer> b = [q.queue commandBuffer];
		CHECK(a.status == MTLCommandBufferStatusNotEnqueued);
		[a enqueue];
		[b enqueue];
		CHECK(a.status == MTLCommandBufferStatusEnqueued && b.status == MTLCommandBufferStatusEnqueued);
		[b commit];
		CHECK(b.status == MTLCommandBufferStatusCommitted);
		CHECK(posted(q).empty());
		// Committed without an enqueue, behind both.
		id<MTLCommandBuffer> c = [q.queue commandBuffer];
		[c commit];
		CHECK(posted(q).empty());
		[a commit];
		CHECK((posted(q) == std::vector<uint64_t>{objId(a), objId(b), objId(c)}));
		// The enqueue a commit makes is its only one: nothing waits behind a posted buffer.
		id<MTLCommandBuffer> d = [q.queue commandBuffer];
		[d commit];
		CHECK((posted(q) == std::vector<uint64_t>{objId(d)}));
	}
	@autoreleasepool {
		Queue q;
		__weak id<MTLCommandBuffer> weakA = nil;
		@autoreleasepool {
			id<MTLCommandBuffer> a = [q.queue commandBuffer];
			weakA = a;
			[a enqueue];
		}
		CHECK(weakA != nil);
		id<MTLCommandBuffer> b = [q.queue commandBuffer];
		[b commit];
		CHECK(posted(q).empty());
		CHECK(b.status == MTLCommandBufferStatusCommitted);
		id<MTLCommandBuffer> a = weakA;
		[a commit];
		CHECK((posted(q) == std::vector<uint64_t>{objId(a), objId(b)}));
	}
}

// Buffers enqueued in one order and committed from several threads at once, in any order, go on
// the ring in the enqueue order.
void test_threads_commit_in_enqueue_order()
{
	@autoreleasepool {
		Queue q;
		const int n = 400;
		NSMutableArray<id<MTLCommandBuffer>> *cbs = [NSMutableArray array];
		std::vector<uint64_t> want;
		for (int i = 0; i < n; i++) {
			id<MTLCommandBuffer> cb = [q.queue commandBuffer];
			id<MTLBlitCommandEncoder> e = [cb blitCommandEncoder];
			[e fillBuffer:handle(7) range:NSMakeRange(0, 4) value:static_cast<uint8_t>(i)];
			[e endEncoding];
			[cb enqueue];
			[cbs addObject:cb];
			want.push_back(objId(cb));
		}
		// The host takes the ring as the threads post, so it never fills.
		std::atomic<bool> stop{false};
		std::vector<uint64_t> got;
		std::thread host([&] {
			while (!stop.load()) {
				for (uint64_t id : posted(q))
					got.push_back(id);
				std::this_thread::yield();
			}
			for (uint64_t id : posted(q))
				got.push_back(id);
		});
		std::vector<std::thread> threads;
		for (int t = 0; t < 4; t++) {
			threads.emplace_back([cbs, t] {
				@autoreleasepool {
					// Each thread commits every fourth buffer, from the last.
					for (int i = n - 1 - t; i >= 0; i -= 4)
						[cbs[static_cast<NSUInteger>(i)] commit];
				}
			});
		}
		for (std::thread &th : threads)
			th.join();
		stop.store(true);
		host.join();
		CHECK(got == want);
	}
}

// commitAndWaitUntilSubmitted of a buffer behind one enqueued and not yet committed returns only
// once that one is committed and both are submitted, as Metal's does.
void test_submitted_waits_for_the_buffers_before()
{
	@autoreleasepool {
		Queue q(true);
		Driven driven({&q});
		id<MTLCommandBuffer> a = [q.queue commandBuffer];
		id<MTLCommandBuffer> b = [q.queue commandBuffer];
		[a enqueue];
		std::vector<uint64_t> order;
		std::thread host([&] {
			// The host schedules each buffer as it takes it.
			for (int taken = 0; taken < 2;) {
				RgfxMsgHeader h = {};
				std::vector<uint8_t> bytes;
				if (!q.submit.next(&h, &bytes)) {
					std::this_thread::sleep_for(std::chrono::milliseconds(1));
					continue;
				}
				RgfxCommandBuffer cbh = {};
				std::memcpy(&cbh, bytes.data(), sizeof(cbh));
				order.push_back(cbh.cb_obj_id);
				hostScheduled(q, *driven.kext, h.seqno, cbh.cb_obj_id == objId(a) ? a : b);
				taken++;
			}
		});
		// The wait is counted from before a's commit is scheduled, so it is at least the 50 ms
		// before it however late this thread gets to the call.
		const auto start = std::chrono::steady_clock::now();
		std::thread late([a] {
			std::this_thread::sleep_for(std::chrono::milliseconds(50));
			[a commit];
		});
		CHECK([(id<MTLCommandBufferSPI>)b commitAndWaitUntilSubmitted]);
		CHECK(std::chrono::steady_clock::now() - start >= std::chrono::milliseconds(40));
		late.join();
		host.join();
		CHECK((order == std::vector<uint64_t>{objId(a), objId(b)}));
	}
}

// A resource answers isComplete and isWriteComplete NO, and its waitUntilComplete waits, while a
// command buffer that names it is submitted and its completion is yet to be taken: the Intel
// 630's answers once the buffer is scheduled, for any use (docs/graphics-kext/dgpu-oracle.md,
// Q26). Before (encoded, or committed and not yet submitted) it answers YES at once, as all three
// Macs do, and so does a resource no buffer in flight names. Apple's GL renderer reads a texture
// back with getBytes: after its waitUntilComplete, behind a draw committed on another queue with
// unretained references, as this one is: answered at once, the read saw the texture as it was
// before the draw (macOS 11.7.11's Safari WebGL on a 780M host). The wait ends at the
// completion's take, not behind an earlier buffer's handler on its queue, as the 630's returns
// with the buffer still Scheduled.
void test_resources_wait_for_their_buffers_in_flight()
{
	@autoreleasepool {
		Queue q(true);
		Driven driven({&q});
		auto spiHandle = [](uint64_t objId) {
			return [[QueueTestSpiHandle alloc] initWithObjId:objId device:nil serializer:nullptr];
		};
		auto draw = [](id<MTLCommandBuffer> cb, id<MTLTexture> target) {
			MTLRenderPassDescriptor *rp = [MTLRenderPassDescriptor renderPassDescriptor];
			rp.colorAttachments[0].texture = target;
			[[cb renderCommandEncoderWithDescriptor:rp] endEncoding];
		};
		QueueTestSpiHandle *drawn = spiHandle(51), *other = spiHandle(52);
		id<MTLCommandBuffer> cb = [q.queue commandBufferWithUnretainedReferences];
		draw(cb, (id<MTLTexture>)drawn);
		CHECK([drawn isComplete] && [drawn isWriteComplete]);
		[cb commit];
		CHECK([drawn isComplete] && [drawn isWriteComplete]);
		const uint64_t seqno = nextSeqno(q);
		hostScheduled(q, *driven.kext, seqno, cb);
		[cb waitUntilScheduled];
		CHECK(![drawn isComplete] && ![drawn isWriteComplete]);
		CHECK([other isComplete] && [other isWriteComplete]);
		[other waitUntilComplete];
		std::atomic<bool> waited{false};
		dispatch_semaphore_t done = dispatch_semaphore_create(0);
		std::thread waiter([&] {
			[drawn waitUntilComplete];
			waited.store(true);
			dispatch_semaphore_signal(done);
		});
		CHECK(waitFor(done, 0.1) != 0 && !waited.load());
		hostComplete(q, *driven.kext, seqno, cb);
		CHECK(waitFor(done, 5) == 0);
		waiter.join();
		CHECK([drawn isComplete] && [drawn isWriteComplete]);
		[cb waitUntilCompleted];
		CHECK(cb.status == MTLCommandBufferStatusCompleted);

		// Behind a buffer of its queue whose completed handler has not returned, the wait ends at
		// the take of its own buffer's completion, with that buffer not yet complete.
		dispatch_semaphore_t release = dispatch_semaphore_create(0);
		id<MTLCommandBuffer> first = [q.queue commandBuffer];
		[first addCompletedHandler:^(id<MTLCommandBuffer>) {
			dispatch_semaphore_wait(release, DISPATCH_TIME_FOREVER);
		}];
		[first commit];
		const uint64_t firstSeqno = nextSeqno(q);
		id<MTLCommandBuffer> second = [q.queue commandBuffer];
		draw(second, (id<MTLTexture>)drawn);
		[second commit];
		const uint64_t secondSeqno = nextSeqno(q);
		hostScheduled(q, *driven.kext, firstSeqno, first);
		hostScheduled(q, *driven.kext, secondSeqno, second);
		[second waitUntilScheduled];
		CHECK(![drawn isComplete]);
		hostComplete(q, *driven.kext, firstSeqno, first);
		hostComplete(q, *driven.kext, secondSeqno, second);
		std::thread late([&] {
			[drawn waitUntilComplete];
			dispatch_semaphore_signal(done);
		});
		CHECK(waitFor(done, 5) == 0);
		late.join();
		CHECK(second.status != MTLCommandBufferStatusCompleted);
		dispatch_semaphore_signal(release);
		[second waitUntilCompleted];
		CHECK(first.status == MTLCommandBufferStatusCompleted &&
		      second.status == MTLCommandBufferStatusCompleted);
	}
}

// A second enqueue, an enqueue after the commit and a second commit end the process at the call
// with Metal's words, as Metal does on all three GPUs. (They used to do nothing.)
void test_enqueue_and_commit_are_once()
{
	const char *enqueued = "-[_MTLCommandBuffer enqueue]: failed assertion `enqueue an already "
	                       "enqueued command buffer'";
	checkDies("enqueue-twice", enqueued);
	checkDies("enqueue-after-commit", enqueued);
	checkDies("commit-twice", "-[IOGPUMetalCommandBuffer validate]: failed assertion `commit an "
	                          "already committed command buffer'");
	checkDies("commit-and-wait-after-commit", "commit an already committed command buffer");
}

// The region's blocks, placed one after another past the floor up to `limit`, and those live.
struct Blocks {
	uint64_t next = RGFX_ALLOCATIONS_OFFSET;
	uint64_t limit = 0;
	uint64_t made = 0;
	std::vector<uint64_t> live;
	static bool alloc(void *c, uint64_t size, uint64_t align, uint64_t *offset)
	{
		Blocks *b = static_cast<Blocks *>(c);
		const uint64_t at = (b->next + align - 1) & ~(align - 1);
		if (at + size > b->limit)
			return false;
		b->next = at + size;
		b->made++;
		b->live.push_back(at);
		*offset = at;
		return true;
	}
	static void release(void *c, uint64_t offset)
	{
		std::vector<uint64_t> &live = static_cast<Blocks *>(c)->live;
		for (auto it = live.begin(); it != live.end(); ++it) {
			if (*it == offset) {
				live.erase(it);
				return;
			}
		}
		CHECK(false); // a block released twice, or never placed
	}
};

// The process's one chunk of resource ids, for the textures a test makes.
bool oneIdChunk(void *context, uint64_t *first, uint64_t *ids)
{
	int *calls = static_cast<int *>(context);
	if ((*calls)++ != 0)
		return false;
	*first = 1 + 4 * RGFX_RESOURCE_ID_CHUNK;
	return true;
}

// A buffer that writes only surfaces in place retires its writes as the completion thread takes
// its completion, before its handlers, which run behind the queue's earlier ones: a CPU lock
// waiting for its writes goes on there, as Metal's waits only for the GPU. Its delivery retires
// nothing more. A buffer that writes a copy retires once the queue's write-back queue has written
// the copy back, and so does a later buffer of its queue that writes only surfaces in place,
// behind it, as the kext retires a queue's writes up to a seqno: the copy's would otherwise
// retire before its write-back. Neither waits for the handlers before them, so a handler that
// locks a surface a later buffer of its queue writes waits only for that buffer's write-back;
// their deliveries wait for it. (They used to retire at their deliveries, behind the handlers:
// such a lock waited out the kext's 2 s, then went on without the results.) A held buffer (one
// that waits on an event below its value) that writes the copy and completes with no
// CB_SCHEDULED, its accesses not yet named, retires there too, and its scheduling at its
// delivery names none of its writes to the kext: one named after its retire would hold the
// surface's CPU locks for good.
void test_in_place_writes_retire_at_the_take()
{
	struct Source {
		uint64_t inPlace = RGFX_ALLOCATIONS_OFFSET + 0x40000;
		uint64_t copy = RGFX_ALLOCATIONS_OFFSET + 0x60000;
		uint32_t inPlaceId = 0;
		std::atomic<int> given{0};
		std::mutex lock;
		std::vector<uint64_t> retired; // the seqnos, in the calls' order
		std::vector<uint64_t> named;   // the seqnos whose writes are named to the kext
		// The copied surface, once set, and its first byte at each retire after that.
		IOSurfaceRef copied = nullptr;
		std::vector<uint8_t> copiedAtRetire;
		static RgfxRegion::PagesAnswer pages(void *c, const void *, uint64_t, uint64_t,
		                                     uint32_t surfaceId, bool,
		                                     const RgfxSurfacePagesLayout *, uint64_t *offset,
		                                     uint64_t *serial, bool *fresh, bool *cached)
		{
			Source *src = static_cast<Source *>(c);
			if (surfaceId != src->inPlaceId)
				return RgfxRegion::kPagesRefused;
			src->given++;
			*offset = src->inPlace;
			*serial = 9;
			*fresh = true;
			*cached = false;
			return RgfxRegion::kPagesGiven;
		}
		static bool acquire(void *c, const void *, uint64_t, uint64_t, uint64_t *offset,
		                    uint64_t *serial, bool *fresh)
		{
			*offset = static_cast<Source *>(c)->copy;
			*serial = 10;
			*fresh = true;
			return true;
		}
		static void release(void *, uint64_t) {}
		static void writes(void *c, uint64_t, uint64_t seqno, const uint64_t *, uint32_t)
		{
			Source *src = static_cast<Source *>(c);
			std::lock_guard<std::mutex> hold(src->lock);
			src->named.push_back(seqno);
		}
		bool namedAt(uint64_t seqno)
		{
			std::lock_guard<std::mutex> hold(lock);
			return std::find(named.begin(), named.end(), seqno) != named.end();
		}
		static void retire(void *c, uint64_t, uint64_t seqno)
		{
			Source *src = static_cast<Source *>(c);
			std::lock_guard<std::mutex> hold(src->lock);
			src->retired.push_back(seqno);
			if (src->copied != nullptr)
				src->copiedAtRetire.push_back(
				    *static_cast<const uint8_t *>(IOSurfaceGetBaseAddress(src->copied)));
		}
		std::vector<uint8_t> seenCopied()
		{
			std::lock_guard<std::mutex> hold(lock);
			return copiedAtRetire;
		}
		std::vector<uint64_t> seen()
		{
			std::lock_guard<std::mutex> hold(lock);
			return retired;
		}
	};
	@autoreleasepool {
		// Three slots: the test takes all three back once the completion thread has taken the
		// three buffers it holds, which it gives back at each take, after the retire's choice.
		Queue q(true, 3);
		int &calls = *new int(0);
		q.ids.setResourceIdSource(&oneIdChunk, &calls);
		// Never freed, as the rings: the textures may outlive the test.
		std::vector<uint8_t> &mem = *new std::vector<uint8_t>(RGFX_ALLOCATIONS_OFFSET + 0x100000, 0);
		RgfxRegion &region = *new RgfxRegion(mem.data(), mem.size());
		Source &src = *new Source();
		region.setSurfaceSource(&Source::acquire, &Source::release, &src);
		region.setSurfacePagesSource(&Source::pages, 16, &src);
		region.setSurfaceWriteSource(&Source::writes, &Source::retire, &src);
		NSDictionary *props = @{
			(id)kIOSurfaceWidth : @16,
			(id)kIOSurfaceHeight : @16,
			(id)kIOSurfaceBytesPerElement : @4,
			(id)kIOSurfacePixelFormat : @(static_cast<uint32_t>('BGRA')),
		};
		IOSurfaceRef inPlace = IOSurfaceCreate((__bridge CFDictionaryRef)props);
		IOSurfaceRef copied = IOSurfaceCreate((__bridge CFDictionaryRef)props);
		CHECK(inPlace != nullptr && copied != nullptr);
		src.inPlaceId = IOSurfaceGetID(inPlace);
		auto over = [&](IOSurfaceRef surface) {
			MTLTextureDescriptor *td =
			    [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
			                                                       width:16
			                                                      height:16
			                                                   mipmapped:NO];
			td.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
			return [RgfxIOSurfaceTexture textureWithDescriptor:td
			                                        iosurface:surface
			                                            plane:0
			                                           region:&region
			                                           device:nil
			                                       serializer:&q.ser];
		};
		RgfxIOSurfaceTexture *tp = over(inPlace);
		RgfxIOSurfaceTexture *tc = over(copied);
		CHECK(tp != nil && tc != nil && src.given == 1);
		auto draw = [&](id<MTLTexture> t) {
			id<MTLCommandBuffer> cb = [q.queue commandBuffer];
			MTLRenderPassDescriptor *rp = [MTLRenderPassDescriptor renderPassDescriptor];
			rp.colorAttachments[0].texture = t;
			rp.colorAttachments[0].loadAction = MTLLoadActionClear;
			rp.colorAttachments[0].storeAction = MTLStoreActionStore;
			[[cb renderCommandEncoderWithDescriptor:rp] endEncoding];
			return cb;
		};
		// A buffer whose completed handler holds the queue's later records back until let go.
		dispatch_semaphore_t held = dispatch_semaphore_create(0);
		dispatch_semaphore_t letGo = dispatch_semaphore_create(0);
		auto holding = [&]() {
			id<MTLCommandBuffer> cb = [q.queue commandBuffer];
			[cb addCompletedHandler:^(id<MTLCommandBuffer> c) {
				(void)c;
				dispatch_semaphore_signal(held);
				waitFor(letGo, 10);
			}];
			return cb;
		};
		auto retiredBecomes = [&](const std::vector<uint64_t> &want) {
			for (int i = 0; i < 5000 && src.seen() != want; i++)
				std::this_thread::sleep_for(std::chrono::milliseconds(1));
			return src.seen() == want;
		};
		Driven driven({&q});

		// In place: retired at the take, while the handler before it holds its delivery.
		id<MTLCommandBuffer> hold = holding();
		[hold commit];
		const uint64_t sh = nextSeqno(q);
		id<MTLCommandBuffer> a = draw(tp);
		[a commit];
		const uint64_t sa = nextSeqno(q);
		hostComplete(q, *driven.kext, sh, hold);
		CHECK(waitFor(held, 5) == 0);
		hostComplete(q, *driven.kext, sa, a);
		CHECK(retiredBecomes({sa}));
		CHECK(a.status != MTLCommandBufferStatusCompleted);
		dispatch_semaphore_signal(letGo);
		[a waitUntilCompleted];
		CHECK(a.status == MTLCommandBufferStatusCompleted);
		CHECK(src.seen() == std::vector<uint64_t>({sa}));

		// A copy, then in place: both retire in order once the copy is written back, while the
		// handler before them holds their deliveries.
		hold = holding();
		[hold commit];
		const uint64_t sh2 = nextSeqno(q);
		id<MTLCommandBuffer> b = draw(tc);
		[b commit];
		const uint64_t sb = nextSeqno(q);
		id<MTLCommandBuffer> c = draw(tp);
		[c commit];
		const uint64_t sc = nextSeqno(q);
		{
			std::lock_guard<std::mutex> g(src.lock);
			src.copied = copied;
		}
		// What the GPU drew into the copy: its first row.
		std::memset(&mem[src.copy], 0x5a, 16 * 4);
		hostComplete(q, *driven.kext, sh2, hold);
		CHECK(waitFor(held, 5) == 0);
		hostComplete(q, *driven.kext, sb, b);
		hostComplete(q, *driven.kext, sc, c);
		CHECK(retiredBecomes({sa, sb, sc}));
		CHECK(src.seenCopied() == std::vector<uint8_t>({0x5a, 0x5a}));
		CHECK(b.status != MTLCommandBufferStatusCompleted &&
		      c.status != MTLCommandBufferStatusCompleted);
		dispatch_semaphore_signal(letGo);
		[c waitUntilCompleted];
		CHECK(b.status == MTLCommandBufferStatusCompleted &&
		      c.status == MTLCommandBufferStatusCompleted);
		CHECK(src.seen() == std::vector<uint64_t>({sa, sb, sc}));

		// Held, writing the copy: not named at its post; its completion, with no CB_SCHEDULED,
		// written back and retired while the handler before it holds its delivery; scheduled at
		// its delivery, naming nothing; and the queue holds no buffer after it, whose writes are
		// named at its post.
		RgfxSharedEvent *gate = [RgfxSharedEvent sharedEventWithInitialValue:0
		                                                              device:nil
		                                                          serializer:&q.ser];
		CHECK(gate != nil);
		hold = holding();
		[hold commit];
		const uint64_t sh3 = nextSeqno(q);
		id<MTLCommandBuffer> d = [q.queue commandBuffer];
		[d encodeWaitForEvent:gate value:1];
		{
			MTLRenderPassDescriptor *rp = [MTLRenderPassDescriptor renderPassDescriptor];
			rp.colorAttachments[0].texture = tc;
			rp.colorAttachments[0].loadAction = MTLLoadActionClear;
			rp.colorAttachments[0].storeAction = MTLStoreActionStore;
			[[d renderCommandEncoderWithDescriptor:rp] endEncoding];
		}
		[d commit];
		const uint64_t sd = nextSeqno(q);
		CHECK(src.namedAt(sb) && !src.namedAt(sd));
		std::memset(&mem[src.copy], 0x6b, 16 * 4);
		hostComplete(q, *driven.kext, sh3, hold);
		CHECK(waitFor(held, 5) == 0);
		hostComplete(q, *driven.kext, sd, d);
		CHECK(retiredBecomes({sa, sb, sc, sd}));
		CHECK(src.seenCopied() == std::vector<uint8_t>({0x5a, 0x5a, 0x6b}));
		CHECK(d.status != MTLCommandBufferStatusCompleted);
		dispatch_semaphore_signal(letGo);
		[d waitUntilCompleted];
		CHECK(d.status == MTLCommandBufferStatusCompleted);
		CHECK(!src.namedAt(sd));
		id<MTLCommandBuffer> e = draw(tp);
		[e commit];
		const uint64_t se = nextSeqno(q);
		CHECK(src.namedAt(se));
		hostComplete(q, *driven.kext, se, e);
		[e waitUntilCompleted];
		CHECK(e.status == MTLCommandBufferStatusCompleted && !src.namedAt(sd));
	}
}

// A buffer that writes app memory under a no-copy buffer (its GPU copy a block of the region)
// writes the copy back into the memory as its completion is taken, on its queue's write-back
// queue, while a handler before it on the queue still holds its delivery: the CPU sees the GPU's
// write as the buffer is done, as on Metal's shared memory, not once the handlers before it have
// run. Its delivery, its status and its waiters come after the write-back.
void test_no_copy_write_backs_land_at_the_take()
{
	@autoreleasepool {
		Queue q(true, 3);
		int &calls = *new int(0);
		q.ids.setResourceIdSource(&oneIdChunk, &calls);
		std::vector<uint8_t> &mem = *new std::vector<uint8_t>(RGFX_ALLOCATIONS_OFFSET + 0x200000, 0);
		RgfxRegion &region = *new RgfxRegion(mem.data(), mem.size());
		RgfxRegionSlabSource &slabSource = *new RgfxRegionSlabSource(region, q.ser);
		RgfxSlabs &slabs = *new RgfxSlabs(slabSource);
		vm_address_t app = 0;
		CHECK(vm_allocate(mach_task_self(), &app, 0x4000, VM_FLAGS_ANYWHERE) == KERN_SUCCESS);
		uint8_t *p = reinterpret_cast<uint8_t *>(app);
		std::memset(p, 0x11, 0x4000);
		RgfxBuffer *b = [RgfxBuffer bufferWithBytesNoCopy:p
		                                           length:0x4000
		                                          options:MTLResourceStorageModeShared
		                                      deallocator:nil
		                                        maxLength:0
		                                           region:&region
		                                            slabs:&slabs
		                                           device:nil
		                                       serializer:&q.ser];
		CHECK(b != nil && [b rgfxShadow] != nullptr);
		// Its GPU copy: a range of a slab, a heap over a block of the region.
		RgfxHeapOverHostMemory heap = {};
		RgfxBufferAtOffset create = {};
		{
			RgfxMsgHeader h = {};
			std::vector<uint8_t> bytes;
			bool sawHeap = false, sawBuffer = false;
			while ((!sawHeap || !sawBuffer) && q.control.next(&h, &bytes)) {
				if (h.type == RGFX_CMD_HEAP_OVER_HOST_MEMORY && bytes.size() == sizeof(heap)) {
					std::memcpy(&heap, bytes.data(), sizeof(heap));
					sawHeap = true;
				} else if (h.type == RGFX_CMD_BUFFER_AT_OFFSET && bytes.size() == sizeof(create)) {
					std::memcpy(&create, bytes.data(), sizeof(create));
					sawBuffer = true;
				}
			}
			CHECK(sawHeap && sawBuffer && create.obj_id == b.rgfxObjId);
		}
		uint8_t *block = mem.data() + heap.region_offset + create.offset;
		CHECK(std::memcmp(block, p, 0x4000) == 0);
		dispatch_semaphore_t held = dispatch_semaphore_create(0);
		dispatch_semaphore_t letGo = dispatch_semaphore_create(0);
		Driven driven({&q});

		id<MTLCommandBuffer> hold = [q.queue commandBuffer];
		[hold addCompletedHandler:^(id<MTLCommandBuffer> c) {
			(void)c;
			dispatch_semaphore_signal(held);
			waitFor(letGo, 10);
		}];
		[hold commit];
		const uint64_t sh = nextSeqno(q);
		id<MTLCommandBuffer> fill = [q.queue commandBuffer];
		id<MTLBlitCommandEncoder> blit = [fill blitCommandEncoder];
		[blit fillBuffer:b range:NSMakeRange(0x100, 0x10) value:0x77];
		[blit endEncoding];
		[fill commit];
		const uint64_t sf = nextSeqno(q);
		// What the host's fill wrote into the GPU's copy.
		std::memset(block + 0x100, 0x77, 0x10);
		hostComplete(q, *driven.kext, sh, hold);
		CHECK(waitFor(held, 5) == 0);
		hostComplete(q, *driven.kext, sf, fill);
		bool landed = false;
		for (int i = 0; i < 5000 && !landed; i++) {
			landed = __atomic_load_n(&p[0x10f], __ATOMIC_ACQUIRE) == 0x77;
			if (!landed)
				std::this_thread::sleep_for(std::chrono::milliseconds(1));
		}
		CHECK(landed && p[0x100] == 0x77 && p[0x110] == 0x11);
		CHECK(fill.status != MTLCommandBufferStatusCompleted);
		dispatch_semaphore_signal(letGo);
		[fill waitUntilCompleted];
		CHECK(fill.status == MTLCommandBufferStatusCompleted);
	}
}

// A command buffer let go with no completion gives back the region block its message lies in, as
// its completion does: one posted out of line on a queue whose completion ring nothing reads.
// Buffer after buffer reuse one block, and the queue's going gives it back to the region; one
// encoded past what its queue's ring takes and never committed takes none, its message copied
// into a block only at the commit. (A buffer let go so used to keep its block, and the region
// lost one a buffer.)
void test_buffers_let_go_give_their_blocks_back()
{
	std::vector<uint8_t> mem(RGFX_ALLOCATIONS_OFFSET + 0x400000, 0);
	RgfxRegion region(mem.data(), mem.size());
	Blocks blocks;
	blocks.limit = mem.size();
	region.setBlockSource(&Blocks::alloc, &Blocks::release, &blocks);
	__weak RgfxCommandQueue *weakQueue = nil;
	@autoreleasepool {
		Queue q; // no reader: no completion comes
		q.queue.region = &region;
		weakQueue = q.queue;
		// Far past half the 32 KiB ring.
		auto encode = [&q] {
			id<MTLCommandBuffer> cb = [q.queue commandBuffer];
			id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
			for (int i = 0; i < 2000; i++)
				[blit fillBuffer:handle(40)
				           range:NSMakeRange(4 * static_cast<NSUInteger>(i), 4)
				           value:static_cast<uint8_t>(i)];
			[blit endEncoding];
			return cb;
		};
		const int n = 16;
		for (int i = 0; i < n; i++) {
			@autoreleasepool {
				(void)encode();
			}
		}
		CHECK(blocks.made == 0);
		for (int i = 0; i < n; i++) {
			@autoreleasepool {
				id<MTLCommandBuffer> cb = encode();
				[cb commit];
				RgfxMsgHeader h = {};
				std::vector<uint8_t> bytes;
				CHECK(q.submit.next(&h, &bytes) && h.type == RGFX_CMD_COMMAND_BUFFER_IN_REGION);
			}
		}
		CHECK(blocks.made == 1);
		CHECK(blocks.live.size() == 1);
	}
	CHECK(weakQueue == nil);
	CHECK(blocks.live.empty());
}

} // namespace

#if defined(__arm64__)
int rgfxQueueDiesScheduledPlain();
#endif

int rgfxQueueDiesCase(const char *name)
{
#if defined(__arm64__)
	if (std::strcmp(name, "scheduled-signal-plain") == 0)
		return rgfxQueueDiesScheduledPlain();
#endif
	signal(SIGABRT, [](int) { _exit(kAborted); });
	@autoreleasepool {
		if (std::strcmp(name, "function-handle") == 0) {
			// It ends the process before anything of the function is read.
			id function = [RgfxFunction alloc];
			[(id<QueueTestFunctionSPI>)function functionHandle];
			return 0;
		}
		Queue q(true);
		id<MTLCommandBuffer> cb = [q.queue commandBuffer];
		if (std::strcmp(name, "enqueue-twice") == 0) {
			[cb enqueue];
			[cb enqueue];
			return 0;
		}
		[cb commit];
		if (std::strcmp(name, "enqueue-after-commit") == 0)
			[cb enqueue];
		if (std::strcmp(name, "commit-twice") == 0)
			[cb commit];
		if (std::strcmp(name, "commit-and-wait-after-commit") == 0)
			[(id<MTLCommandBufferSPI>)cb commitAndWaitUntilSubmitted];
		auto late = [&](bool completedFirst, void (^add)(void)) {
			if (completedFirst)
				complete(q, cb);
			add();
		};
		if (std::strcmp(name, "completed-after-commit") == 0 ||
		    std::strcmp(name, "completed-after-completion") == 0)
			late(std::strcmp(name, "completed-after-completion") == 0, ^{
				[cb addCompletedHandler:^(id<MTLCommandBuffer> c) {
					(void)c;
				}];
			});
		if (std::strcmp(name, "scheduled-after-commit") == 0 ||
		    std::strcmp(name, "scheduled-after-completion") == 0)
			late(std::strcmp(name, "scheduled-after-completion") == 0, ^{
				[cb addScheduledHandler:^(id<MTLCommandBuffer> c) {
					(void)c;
				}];
			});
		if (std::strcmp(name, "present-after-commit") == 0)
			[cb presentDrawable:(id<MTLDrawable>)[NSObject new]];
	}
	return 0;
}

#if defined(__arm64__)
// A scheduled signal of a plain event, on a buffer not yet committed: it ends the process.
int rgfxQueueDiesScheduledPlain()
{
	signal(SIGABRT, [](int) { _exit(kAborted); });
	@autoreleasepool {
		Queue q(true);
		id<MTLCommandBuffer> cb = [q.queue commandBufferWithUnretainedReferences];
		RgfxEvent *plain = [RgfxEvent eventWithDevice:nil serializer:&q.ser];
		[(id<QueueTestScheduledSignalSPI>)cb encodeSignalEventScheduled:plain value:1];
	}
	return 0;
}
#endif

// A command buffer's signal of a shared event that the host posts while the buffer runs, ahead of
// its CB_SCHEDULED, reaches the process through the completion thread: a thread waiting on the
// CPU for it wakes while the buffer is still Committed, and the value it then sets goes to the
// host (SHARED_EVENT_SET_VALUE), which lets the buffer's later wait for that value go; the buffer
// then completes. Metal's CPU sees such a signal while the buffer runs, and the pattern ends on
// all three GPUs (docs/graphics-kext/dgpu-oracle.md, Q18).
void test_a_mid_buffer_signal_wakes_a_cpu_wait_before_the_buffer_ends()
{
	@autoreleasepool {
		Queue q(true);
		Driven driven({&q});
		RgfxSharedEvent *ev = [RgfxSharedEvent sharedEventWithInitialValue:0
		                                                             device:nil
		                                                         serializer:&q.ser];
		CHECK(ev != nil);
		[ev rgfxRegisterWithReplyReader:[q.queue sharedReplyReader].get()];
		id<MTLCommandBuffer> cb = [q.queue commandBuffer];
		[[cb blitCommandEncoder] endEncoding];
		[cb encodeSignalEvent:ev value:1];
		[cb encodeWaitForEvent:ev value:2];
		[[cb blitCommandEncoder] endEncoding];
		[cb commit];
		const uint64_t seqno = nextSeqno(q);

		// The app's thread: it waits for 1, then sets 2.
		std::atomic<int> woke{0};
		std::atomic<long> statusAtWake{-1};
		std::thread app([ev, cb, &woke, &statusAtWake] {
			const BOOL got = [ev waitUntilSignaledValue:1 timeoutMS:5000];
			statusAtWake = static_cast<long>(cb.status);
			woke = got ? 1 : -1;
			ev.signaledValue = 2;
		});
		// The host's GPU reaches the signal: its value goes ahead of any record of the buffer's.
		RgfxEventSignalled sig = {};
		sig.header.type = RGFX_EVT_EVENT_SIGNALLED;
		sig.header.size = sizeof(sig);
		sig.obj_id = ev.rgfxObjId;
		sig.value = 1;
		hostPost(q.completion, *driven.kext, &sig, sizeof(sig));
		// The set comes on the control ring.
		bool set = false;
		for (int i = 0; i < 5000 && !set; i++) {
			RgfxMsgHeader h = {};
			std::vector<uint8_t> bytes;
			while (!set && q.control.next(&h, &bytes)) {
				RgfxSharedEventSetValue m = {};
				if (h.type == RGFX_CMD_SHARED_EVENT_SET_VALUE && bytes.size() >= sizeof(m)) {
					std::memcpy(&m, bytes.data(), sizeof(m));
					set = m.obj_id == ev.rgfxObjId && m.value == 2;
				}
			}
			if (!set)
				std::this_thread::sleep_for(std::chrono::milliseconds(1));
		}
		app.join();
		CHECK(woke == 1);
		CHECK(statusAtWake == static_cast<long>(MTLCommandBufferStatusCommitted));
		CHECK(set);
		CHECK(ev.signaledValue == 2);
		// The host's wait is let go: the buffer is scheduled and completes.
		hostScheduled(q, *driven.kext, seqno, cb);
		hostComplete(q, *driven.kext, seqno, cb);
		[cb waitUntilCompleted];
		CHECK(cb.status == MTLCommandBufferStatusCompleted);
	}
}

// The guest's share of a command buffer's scheduling and completion: how long after the host
// posts a buffer's CB_SCHEDULED, then its CB_COMPLETE, its handlers run and its waits return,
// through the queue's reader, the completion thread (woken by a stand-in of the kext's wait, as
// the kext wakes it) and the queue's completion queue, the host's post made by this thread.
// `rounds` buffers of each kind: one with a scheduled and a completed handler, one with a completed
// handler alone, one with none; each record posted 2 ms after the last, so the threads sleep in
// between as between a client's frames. Prints the median, 90th and 99th percentiles and the most,
// in microseconds: a measurement, run by hand (`shell_tests --queue-timing [rounds]`), never in
// `make test`.
int rgfxQueueTimings(int rounds)
{
	using Clock = std::chrono::steady_clock;
	auto us = [](Clock::time_point from, Clock::time_point to) {
		return std::chrono::duration<double, std::micro>(to - from).count();
	};
	auto print = [](const char *name, std::vector<double> v) {
		std::sort(v.begin(), v.end());
		const size_t n = v.size();
		if (n == 0)
			return;
		std::printf("queue-timing: %s: p50 %.1f p90 %.1f p99 %.1f max %.1f us (n %zu)\n", name,
		            v[n / 2], v[n * 9 / 10], v[n * 99 / 100], v[n - 1], n);
	};
	std::vector<double> schedHandler, schedWaitBoth, schedWaitCompleted, doneHandler,
	    doneWaitBoth, doneWaitCompleted, doneWaitNone;
	@autoreleasepool {
		Queue q(true);
		Driven driven({&q});
		for (int round = 0; round < rounds; round++) {
			for (int kind = 0; kind < 3; kind++) {
				@autoreleasepool {
					__block Clock::time_point scheduledRan, completedRan;
					id<MTLCommandBuffer> cb = [q.queue commandBuffer];
					if (kind == 0) {
						[cb addScheduledHandler:^(id<MTLCommandBuffer> x) {
							(void)x;
							scheduledRan = Clock::now();
						}];
					}
					if (kind <= 1) {
						[cb addCompletedHandler:^(id<MTLCommandBuffer> x) {
							(void)x;
							completedRan = Clock::now();
						}];
					}
					[cb commit];
					const uint64_t seqno = nextSeqno(q);
					std::this_thread::sleep_for(std::chrono::milliseconds(2));
					const Clock::time_point scheduledAt = Clock::now();
					hostScheduled(q, *driven.kext, seqno, cb);
					[cb waitUntilScheduled];
					const Clock::time_point scheduledSeen = Clock::now();
					std::this_thread::sleep_for(std::chrono::milliseconds(2));
					const Clock::time_point completedAt = Clock::now();
					hostComplete(q, *driven.kext, seqno, cb);
					[cb waitUntilCompleted];
					const Clock::time_point completedSeen = Clock::now();
					CHECK(cb.status == MTLCommandBufferStatusCompleted);
					if (kind == 0) {
						schedHandler.push_back(us(scheduledAt, scheduledRan));
						schedWaitBoth.push_back(us(scheduledAt, scheduledSeen));
						doneHandler.push_back(us(completedAt, completedRan));
						doneWaitBoth.push_back(us(completedAt, completedSeen));
					} else if (kind == 1) {
						schedWaitCompleted.push_back(us(scheduledAt, scheduledSeen));
						doneWaitCompleted.push_back(us(completedAt, completedSeen));
					} else {
						doneWaitNone.push_back(us(completedAt, completedSeen));
					}
				}
			}
		}
	}
	print("CB_SCHEDULED to its scheduled handler", schedHandler);
	print("CB_SCHEDULED to waitUntilScheduled, both handlers", schedWaitBoth);
	print("CB_SCHEDULED to waitUntilScheduled, a completed handler alone", schedWaitCompleted);
	print("CB_COMPLETE to its completed handler", doneHandler);
	print("CB_COMPLETE to waitUntilCompleted, both handlers", doneWaitBoth);
	print("CB_COMPLETE to waitUntilCompleted, a completed handler alone", doneWaitCompleted);
	print("CB_COMPLETE to waitUntilCompleted, no handler", doneWaitNone);
	return 0;
}

namespace {

// The waits Metal's Swift overlay awaits (scheduled(), completed()): a block given before the
// buffer gets there runs after its handlers of that stage, in the order given, on the queue's
// completion queue; one given after runs at once, on the caller's thread.
void test_async_waits_run_after_their_handlers()
{
	@autoreleasepool {
		Queue q(true);
		Driven driven({&q});
		std::mutex lock;
		std::mutex *logLock = &lock;
		std::vector<std::string> log;
		std::vector<std::string> *logp = &log;
		auto note = [logLock, logp](const char *what) {
			std::lock_guard<std::mutex> hold(*logLock);
			logp->push_back(what);
		};
		id<MTLCommandBuffer> cb = [q.queue commandBuffer];
		id<QueueTestCommandBufferSPI> spi = (id<QueueTestCommandBufferSPI>)cb;
		CHECK([cb respondsToSelector:@selector(__waitUntilScheduledAsync:)] &&
		      [cb respondsToSelector:@selector(__waitUntilCompletedAsync:)]);
		[spi __waitUntilCompletedAsync:^(id<MTLCommandBuffer> x) {
			CHECK(x == cb && x.status == MTLCommandBufferStatusCompleted);
			note("wc1");
		}];
		[spi __waitUntilScheduledAsync:^(id<MTLCommandBuffer> x) {
			CHECK(x == cb);
			note("ws1");
		}];
		[cb addScheduledHandler:^(id<MTLCommandBuffer> x) {
			(void)x;
			note("s");
		}];
		[cb addCompletedHandler:^(id<MTLCommandBuffer> x) {
			(void)x;
			note("c");
		}];
		[cb commit];
		[spi __waitUntilCompletedAsync:^(id<MTLCommandBuffer> x) {
			(void)x;
			note("wc2");
		}];
		const uint64_t seqno = nextSeqno(q);
		hostScheduled(q, *driven.kext, seqno, cb);
		hostComplete(q, *driven.kext, seqno, cb);
		[cb waitUntilCompleted];
		for (int i = 0; i < 5000; i++) {
			{
				std::lock_guard<std::mutex> hold(lock);
				if (log.size() >= 5)
					break;
			}
			std::this_thread::sleep_for(std::chrono::milliseconds(1));
		}
		{
			std::lock_guard<std::mutex> hold(lock);
			std::vector<std::string> want = {"s", "ws1", "c", "wc1", "wc2"};
			CHECK(log == want);
		}
		bool ranHere = false;
		bool *ranp = &ranHere;
		[spi __waitUntilScheduledAsync:^(id<MTLCommandBuffer> x) {
			(void)x;
			*ranp = true;
		}];
		CHECK(ranHere);
		ranHere = false;
		[spi __waitUntilCompletedAsync:^(id<MTLCommandBuffer> x) {
			(void)x;
			*ranp = true;
		}];
		CHECK(ranHere);
	}
}

// Whether `body` raises the NSInvalidArgumentException of a counted set given nil to insert, as
// Metal's queue and command buffer raise at a nil residency set (measured: the M3 Max, macOS
// 27.0.1, oracle_residency).
bool raisesInsertingNil(void (^body)(void))
{
	@try {
		body();
	} @catch (NSException *e) {
		return [e.name isEqualToString:NSInvalidArgumentException] &&
		       [e.reason isEqualToString:@"*** -[NSCountedSet addObject:]: attempt to insert nil"];
	}
	return false;
}

// The SPI's state a buffer keeps, as Metal's base buffer keeps it, and the answers a fresh buffer
// of the AMD 5300M's and the Intel 630's gives; profilingResults times each stage the buffer
// reached while profiling was enabled, under Metal's keys.
void test_command_buffer_spi_state()
{
	@autoreleasepool {
		Queue q(true);
		Driven driven({&q});
		id<MTLCommandBuffer> cb = [q.queue commandBuffer];
		id<QueueTestCommandBufferSPI> spi = (id<QueueTestCommandBufferSPI>)cb;
		CHECK(spi.privateData == nil && spi.privateDataOffset == 0 && spi.privateLoggingBuffer == nil);
		CHECK(!spi.sharedIndirectionTable && !spi.isProfilingEnabled && spi.profilingResults == nil);
		CHECK(spi.getListIndex == 0 && spi.deadlineProfile == nil);
		NSUInteger length = 7;
		CHECK([spi debugBufferContentsWithLength:&length] == nullptr && length == 0);
		CHECK([spi sampledBlitCommandEncoder:0 capacity:16] == nil &&
		      [spi sampledComputeCommandEncoderWithProgramInfoBuffer:0 capacity:16] == nil);
		NSObject *data = [NSObject new];
		spi.privateData = data;
		spi.privateDataOffset = 48;
		spi.privateLoggingBuffer = data;
		spi.sharedIndirectionTable = YES;
		CHECK(spi.privateData == data && spi.privateDataOffset == 48 &&
		      spi.privateLoggingBuffer == data && spi.sharedIndirectionTable);
		NSArray *logs = @[];
		[spi setLogs:logs];
		CHECK((id)cb.logs == logs);
		[spi configWithCommandBufferDescriptor:[MTLCommandBufferDescriptor new]];
		const int tasks[] = {1, 2};
		[spi setResponsibleTaskIDs:tasks count:2];
		// Metal's own residency sets are used as the caller's: none, and nothing logged; nil
		// raises where the device makes sets, as on the M3 Max (oracle_residency's
		// cb-use-internal-nil).
		id noSets[1] = {nil};
		[spi useInternalResidencySets:noSets count:0];
#if RgfxMakesResidencySets
		CHECK(raisesInsertingNil(^{
			[spi useInternalResidencySet:nil];
		}));
#else
		[spi useInternalResidencySet:nil];
#endif
		CHECK(!RgfxUnsupportedWasLogged([RgfxCmdBuffer class], @selector(useResidencySet:)) &&
		      !RgfxUnsupportedWasLogged([RgfxCmdBuffer class], @selector(useResidencySets:count:)));
		[spi setProfilingEnabled:YES];
		CHECK(spi.isProfilingEnabled);
		[cb commit];
		const uint64_t seqno = nextSeqno(q);
		hostScheduled(q, *driven.kext, seqno, cb);
		hostComplete(q, *driven.kext, seqno, cb);
		[cb waitUntilCompleted];
		NSDictionary *results = spi.profilingResults;
		NSArray *keys = @[
			@"MTLCommandBufferCreationTime", @"MTLCommandBufferEnqueueTime", @"MTLCommandBufferCommitTime",
			@"MTLCommandBufferSubmitToKernelTime", @"MTLCommandBufferKernelScheduledTime",
			@"MTLCommandBufferSubmitToHWTime", @"MTLCommandBufferCompletionInterruptTime",
			@"MTLCommandBufferKernelCompleteTime", @"MTLCommandBufferCompletionHandlerEnqueueTime",
			@"MTLCommandBufferCompletionHandlerExecutionTime"
		];
		CHECK(results.count == keys.count);
		uint64_t last = 0;
		for (NSString *key in keys) {
			const uint64_t at = [results[key] unsignedLongLongValue];
			CHECK(at != 0 && at >= last);
			last = at;
		}
	}
}

// The SPI's state a queue keeps, as Metal's base queue keeps it, and what a fresh queue of the AMD
// 5300M's and the Intel 630's answers; buffers made once the queue profiles profile, as Metal's
// take the switch from their queue as they are made.
void test_command_queue_spi_state()
{
	@autoreleasepool {
		Queue q(true);
		id<QueueTestCommandQueueSPI> spi = (id<QueueTestCommandQueueSPI>)q.queue;
		CHECK(spi.qosLevel == 2 && spi.commitQueue == nil && !spi.commitSynchronously &&
		      !spi.commitsWithQoS && !spi.disableCrossQueueHazardTracking && !spi.isOpenGLQueue);
		CHECK(spi.backgroundTrackingPID == getpid() && spi.executionEnabled && !spi.skipRender);
		CHECK(!spi.isProfilingEnabled && !spi.isStatEnabled && spi.StatOptions == 0 &&
		      spi.getStatLocations == 0 && spi.numRequestedCounters == 0 &&
		      spi.numInternalSampleCounters == 0 && spi.counterInfo == nil);
		CHECK(spi.availableCounters != nil && spi.availableCounters.count == 0 &&
		      spi.getRequestedCounters == nil && [spi requestCounters:@[ @"counter" ]] == -1);
		NSArray *list = @[ @"a", @"b" ];
		NSArray *lists = [spi subdivideCounterList:list];
		CHECK(lists.count == 1 && lists[0] == list);
		[spi dispatchAvailableCompletionNotifications];
		spi.backgroundTrackingPID = 7;
		[spi setStatOptions:3];
		[spi setStatLocations:5];
		spi.numRequestedCounters = 2;
		spi.numInternalSampleCounters = 4;
		NSMutableDictionary *info = [NSMutableDictionary dictionary];
		spi.counterInfo = info;
		CHECK(spi.backgroundTrackingPID == 7 && spi.StatOptions == 3 && spi.getStatOptions == 3 &&
		      spi.StatLocations == 5 && spi.getStatLocations == 5 && spi.numRequestedCounters == 2 &&
		      spi.numInternalSampleCounters == 4 && spi.counterInfo == info);
		id<MTLCommandBuffer> before = [q.queue commandBuffer];
		[spi setProfilingEnabled:YES];
		id<MTLCommandBuffer> after = [q.queue commandBuffer];
		CHECK(spi.isProfilingEnabled);
		CHECK(![(id<QueueTestCommandBufferSPI>)before isProfilingEnabled] &&
		      [(id<QueueTestCommandBufferSPI>)after isProfilingEnabled]);
	}
}

// The priorities a new queue reads back, and what each setter leaves, as on the AMD 5300M and the
// Intel 630 (measured, macOS 26.7.1, oracle_prio's readback, docs/graphics-kext/dgpu-oracle.md
// Q50): GPU priority 1 and background priority 2; every setter answers YES; a GPU priority set
// reads back, its offset form's too, and a background priority set of 2 leaves both as they were.
void test_queue_priorities_read_back_as_metals()
{
	@autoreleasepool {
		Queue a, b, c, d;
		id<QueueTestPrioritySPI> first = (id<QueueTestPrioritySPI>)a.queue;
		CHECK(first.getGPUPriority == 1 && first.GPUPriority == 1 &&
		      first.getBackgroundGPUPriority == 2);
		CHECK([first setBackgroundGPUPriority:2]);
		CHECK(first.getGPUPriority == 1 && first.getBackgroundGPUPriority == 2);
		CHECK([first setGPUPriority:2]);
		CHECK(first.getGPUPriority == 2 && first.GPUPriority == 2 &&
		      first.getBackgroundGPUPriority == 2);

		id<QueueTestPrioritySPI> second = (id<QueueTestPrioritySPI>)b.queue;
		CHECK([second setGPUPriority:0]);
		CHECK(second.getGPUPriority == 0 && second.getBackgroundGPUPriority == 2);

		id<QueueTestPrioritySPI> third = (id<QueueTestPrioritySPI>)c.queue;
		CHECK([third setGPUPriority:2 offset:1]);
		CHECK(third.getGPUPriority == 2 && third.getBackgroundGPUPriority == 2);

		id<QueueTestPrioritySPI> fourth = (id<QueueTestPrioritySPI>)d.queue;
		CHECK([fourth setBackgroundGPUPriority:2 offset:1]);
		CHECK(fourth.getGPUPriority == 1 && fourth.getBackgroundGPUPriority == 2);
	}
}

// What a descriptor names beyond the command buffer count the queue takes, as Metal's base queue
// does: its quality of service, its commit queue and switches, and its completion queue, where
// its buffers' handlers then run.
void test_queue_takes_its_descriptor()
{
	@autoreleasepool {
		Queue q(true);
		Driven driven({&q});
		MTLCommandQueueDescriptor *descriptor = [MTLCommandQueueDescriptor new];
		CHECK([descriptor respondsToSelector:@selector(setQosLevel:)] &&
		      [descriptor respondsToSelector:@selector(setIsOpenGLQueue:)] &&
		      [descriptor respondsToSelector:@selector(setCompletionQueue:)]);
		id<QueueTestQueueDescriptorSPI> named = (id<QueueTestQueueDescriptorSPI>)descriptor;
		dispatch_queue_t commits = dispatch_queue_create("queue-test.commits", DISPATCH_QUEUE_SERIAL);
		dispatch_queue_t completions =
		    dispatch_queue_create("queue-test.completions", DISPATCH_QUEUE_SERIAL);
		static int completionsKey;
		dispatch_queue_set_specific(completions, &completionsKey, &completionsKey, nullptr);
		[named setQosLevel:4];
		[named setCommitQueue:commits];
		[named setCommitSynchronously:YES];
		[named setCommitsWithQoS:YES];
		[named setDisableCrossQueueHazardTracking:YES];
		[named setIsOpenGLQueue:YES];
		[named setCompletionQueue:completions];
		[q.queue rgfxTakeDescriptor:descriptor];
		id<QueueTestCommandQueueSPI> spi = (id<QueueTestCommandQueueSPI>)q.queue;
		CHECK(spi.qosLevel == 4 && spi.commitQueue == commits && spi.commitSynchronously &&
		      spi.commitsWithQoS && spi.disableCrossQueueHazardTracking && spi.isOpenGLQueue &&
		      spi.completionQueue == completions);
		id<MTLCommandBuffer> cb = [q.queue commandBuffer];
		std::atomic<bool> onIt{false};
		std::atomic<bool> *onItp = &onIt;
		[cb addCompletedHandler:^(id<MTLCommandBuffer> x) {
			(void)x;
			onItp->store(dispatch_get_specific(&completionsKey) == &completionsKey);
		}];
		[cb commit];
		const uint64_t seqno = nextSeqno(q);
		hostScheduled(q, *driven.kext, seqno, cb);
		hostComplete(q, *driven.kext, seqno, cb);
		[cb waitUntilCompleted];
		CHECK(onIt.load());
	}
}

// finish returns once every buffer committed before it has completed and run its completed
// handlers, as Metal's waits for its buffers; with none in flight, at once.
void test_finish_waits_for_committed_buffers()
{
	@autoreleasepool {
		Queue q(true);
		Driven driven({&q});
		id<QueueTestCommandQueueSPI> spi = (id<QueueTestCommandQueueSPI>)q.queue;
		[spi finish];
		std::atomic<int> ran{0};
		std::atomic<int> *ranp = &ran;
		id<MTLCommandBuffer> first = [q.queue commandBuffer];
		id<MTLCommandBuffer> second = [q.queue commandBuffer];
		for (id<MTLCommandBuffer> cb in @[ first, second ]) {
			[cb addCompletedHandler:^(id<MTLCommandBuffer> x) {
				(void)x;
				std::this_thread::sleep_for(std::chrono::milliseconds(5));
				ranp->fetch_add(1);
			}];
			[cb commit];
		}
		const uint64_t firstSeqno = nextSeqno(q);
		const uint64_t secondSeqno = nextSeqno(q);
		std::atomic<bool> finished{false};
		std::atomic<int> ranAtFinish{-1};
		std::thread waiter([&] {
			[spi finish];
			ranAtFinish.store(ran.load());
			finished.store(true);
		});
		std::this_thread::sleep_for(std::chrono::milliseconds(20));
		CHECK(!finished.load());
		hostScheduled(q, *driven.kext, firstSeqno, first);
		hostComplete(q, *driven.kext, firstSeqno, first);
		[first waitUntilCompleted];
		std::this_thread::sleep_for(std::chrono::milliseconds(20));
		CHECK(!finished.load());
		hostScheduled(q, *driven.kext, secondSeqno, second);
		hostComplete(q, *driven.kext, secondSeqno, second);
		waiter.join();
		CHECK(finished.load() && ranAtFinish.load() == 2);
	}
}

#if RgfxMakesResidencySets
// The arm64 device's queue holds a residency set as many times as it was added, and its command
// buffer holds the sets it uses, and those its render and compute encoders are given, until it
// completes, as Metal's do on the M3 Max (oracle_residency's queue, retain, hold, cb and
// render-use and compute-use cases). A nil set raises the counted set's
// NSInvalidArgumentException at the queue and the buffer, and the encoders pass over it.
void test_residency_sets_are_held_until_let_go()
{
	Queue q(true);
	__weak id held = nil;
	@autoreleasepool {
		RgfxResidencySet *set = [[RgfxResidencySet alloc] initWithDevice:nil descriptor:nil];
		held = set;
		[q.queue addResidencySet:set];
		[q.queue addResidencySet:set];
	}
	CHECK(held != nil);
	@autoreleasepool {
		[q.queue removeResidencySet:held];
	}
	CHECK(held != nil);
	@autoreleasepool {
		[q.queue removeResidencySet:held];
	}
	CHECK(held == nil);
	@autoreleasepool {
		RgfxResidencySet *a = [[RgfxResidencySet alloc] initWithDevice:nil descriptor:nil];
		RgfxResidencySet *b = [[RgfxResidencySet alloc] initWithDevice:nil descriptor:nil];
		held = a;
		id<MTLResidencySet> both[] = {a, b};
		[q.queue removeResidencySet:a]; // never added: nothing
		[q.queue addResidencySets:both count:2];
		[q.queue addResidencySets:both count:0];
		[q.queue removeResidencySets:both count:0];
		[q.queue removeResidencySets:both count:2];
		[q.queue removeResidencySets:both count:2];
	}
	CHECK(held == nil);
	CHECK(raisesInsertingNil(^{
		[q.queue addResidencySet:(id<MTLResidencySet> _Nonnull)nil];
	}));
	@try {
		[q.queue removeResidencySet:(id<MTLResidencySet> _Nonnull)nil];
		CHECK(false);
	} @catch (NSException *e) {
		CHECK([e.name isEqualToString:NSInvalidArgumentException] &&
		      [e.reason isEqualToString:@"*** -[NSCountedSet removeObject:]: attempt to remove nil"]);
	}

	__weak id used = nil, viaRender = nil, viaCompute = nil;
	id<MTLCommandBuffer> cb = [q.queue commandBuffer];
	@autoreleasepool {
		RgfxResidencySet *u = [[RgfxResidencySet alloc] initWithDevice:nil descriptor:nil];
		RgfxResidencySet *r = [[RgfxResidencySet alloc] initWithDevice:nil descriptor:nil];
		RgfxResidencySet *c = [[RgfxResidencySet alloc] initWithDevice:nil descriptor:nil];
		used = u;
		viaRender = r;
		viaCompute = c;
		[cb useResidencySet:u];
		id<MTLResidencySet> twice[] = {u, u};
		[cb useResidencySets:twice count:2];
		id<MTLRenderCommandEncoder> renc = [cb renderCommandEncoderWithDescriptor:pass()];
		id renderSPI = renc;
		((void (*)(id, SEL, id))objc_msgSend)(renderSPI, @selector(useResidencySet:), nil);
		((void (*)(id, SEL, id))objc_msgSend)(renderSPI, @selector(useResidencySet:), r);
		[renc endEncoding];
		id<MTLComputeCommandEncoder> cenc = [cb computeCommandEncoder];
		id<MTLResidencySet> withNil[] = {c, nil};
		((void (*)(id, SEL, const id *, NSUInteger))objc_msgSend)(
		    cenc, @selector(useResidencySets:count:), withNil, 2);
		[cenc endEncoding];
	}
	CHECK(used != nil && viaRender != nil && viaCompute != nil);
	CHECK(raisesInsertingNil(^{
		[cb useResidencySet:(id<MTLResidencySet> _Nonnull)nil];
	}));
	@autoreleasepool {
		[cb commit];
	}
	CHECK(used != nil && viaRender != nil && viaCompute != nil);
	complete(q, cb);
	CHECK(used == nil && viaRender == nil && viaCompute == nil);
	CHECK(!RgfxUnsupportedWasLogged([RgfxCmdBuffer class], @selector(useResidencySet:)) &&
	      !RgfxUnsupportedWasLogged([RgfxCommandQueue class], @selector(addResidencySet:)));
}
#endif

// The obj_ids of the OBJECT_DESTROYs on `control`, in the order sent; every message is taken off
// the ring.
std::vector<uint64_t> destroysOn(TestRing &control)
{
	std::vector<uint64_t> ids;
	RgfxMsgHeader h = {};
	std::vector<uint8_t> bytes;
	while (control.next(&h, &bytes)) {
		RgfxObjectDestroy m = {};
		if (h.type != RGFX_CMD_OBJECT_DESTROY || bytes.size() < sizeof(m))
			continue;
		std::memcpy(&m, bytes.data(), sizeof(m));
		ids.push_back(m.obj_id);
	}
	return ids;
}

// How a case of test_buffers_keep_what_they_name_until_completed makes its command buffer.
enum class KeepCase { kUnretained, kUnretainedDescriptor, kRetained };

// One case of test_buffers_keep_what_they_name_until_completed: a buffer made as `made` says.
void keepsWhatItNamesUntilCompleted(KeepCase made)
{
	TestRing *control = nullptr;
	std::vector<uint64_t> named;
	@autoreleasepool {
		Queue q(true);
		control = &q.control;
		int &calls = *new int(0);
		q.ids.setResourceIdSource(&oneIdChunk, &calls);
		MTLTextureDescriptor *desc =
		    [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
		                                                       width:8
		                                                      height:8
		                                                   mipmapped:NO];
		desc.storageMode = MTLStorageModePrivate;
		auto makeTexture = [&] {
			RgfxTexture *t = [RgfxTexture textureWithDescriptor:desc
			                                               heap:RGFX_OBJ_NONE
			                                         heapOffset:0
			                                             device:nil
			                                         serializer:&q.ser];
			CHECK(t != nil);
			return t;
		};
		auto makeBuffer = [&] {
			RgfxBuffer *b = [RgfxBuffer bufferWithLength:256
			                                     options:MTLResourceStorageModePrivate
			                                      device:nil
			                                  serializer:&q.ser];
			CHECK(b != nil);
			return b;
		};
		id<MTLCommandBuffer> cb = nil;
		switch (made) {
		case KeepCase::kUnretained:
			cb = [q.queue commandBufferWithUnretainedReferences];
			break;
		case KeepCase::kUnretainedDescriptor: {
			MTLCommandBufferDescriptor *d = [MTLCommandBufferDescriptor new];
			d.retainedReferences = NO;
			cb = [q.queue commandBufferWithDescriptor:d];
			break;
		}
		case KeepCase::kRetained:
			cb = [q.queue commandBuffer];
			break;
		}
		CHECK(cb != nil && cb.retainedReferences == (made == KeepCase::kRetained));
		// Its handler, added before the commit as Metal requires, takes what was destroyed by then.
		bool handled = false;
		std::vector<uint64_t> atHandler;
		bool *handledp = &handled;
		std::vector<uint64_t> *atHandlerp = &atHandler;
		[cb addCompletedHandler:^(id<MTLCommandBuffer> c) {
			(void)c;
			*handledp = true;
			*atHandlerp = destroysOn(*control);
		}];
		std::vector<uint64_t> resourceIds;
		@autoreleasepool {
			// A blit names a buffer and a texture; a compute pass binds an argument buffer and
			// makes resident a texture only it reaches (useResource:usage:).
			RgfxTexture *copied = makeTexture();
			RgfxBuffer *source = makeBuffer();
			RgfxBuffer *arguments = makeBuffer();
			RgfxTexture *resident = makeTexture();
			named = {copied.rgfxObjId, source.rgfxObjId, arguments.rgfxObjId, resident.rgfxObjId};
			resourceIds = {[copied gpuResourceID]._impl, [resident gpuResourceID]._impl};
			id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
			[blit copyFromBuffer:source
			           sourceOffset:0
			      sourceBytesPerRow:32
			    sourceBytesPerImage:256
			             sourceSize:MTLSizeMake(8, 8, 1)
			              toTexture:copied
			       destinationSlice:0
			       destinationLevel:0
			      destinationOrigin:MTLOriginMake(0, 0, 0)];
			[blit endEncoding];
			id<MTLComputeCommandEncoder> compute = [cb computeCommandEncoder];
			[compute setBuffer:arguments offset:0 atIndex:0];
			[compute useResource:resident usage:MTLResourceUsageRead];
			[compute endEncoding];
#if RgfxMakesResidencySets
			// The arm64 device's buffer also holds a residency set it uses, and with it a texture
			// committed to the set (test_residency_sets_are_held_until_let_go).
			RgfxResidencySet *set = [[RgfxResidencySet alloc] initWithDevice:nil descriptor:nil];
			RgfxTexture *inSet = makeTexture();
			named.push_back(inSet.rgfxObjId);
			resourceIds.push_back([inSet gpuResourceID]._impl);
			[set addAllocation:inSet];
			[set commit];
			[cb useResidencySet:set];
#endif
			[cb commit];
		} // the app lets them go while the buffer is in flight
		const std::vector<uint64_t> early = destroysOn(q.control);
		for (uint64_t obj : named)
			CHECK(std::count(early.begin(), early.end(), obj) == 0);
		// A texture made meanwhile takes another resource id, and goes at once.
		uint64_t otherId = 0;
		@autoreleasepool {
			RgfxTexture *other = makeTexture();
			otherId = other.rgfxObjId;
			const uint64_t otherResourceId = [other gpuResourceID]._impl;
			CHECK(std::count(resourceIds.begin(), resourceIds.end(), otherResourceId) == 0);
		}
		CHECK(destroysOn(q.control) == std::vector<uint64_t>{otherId});

		complete(q, cb);
		CHECK(handled);
		const std::vector<uint64_t> after = destroysOn(q.control);
		for (uint64_t obj : named) {
			CHECK(std::count(atHandler.begin(), atHandler.end(), obj) == 0);
			CHECK(std::count(after.begin(), after.end(), obj) == 1);
		}
	}
	// Once each: none of them is destroyed again as the buffer and its queue go.
	const std::vector<uint64_t> gone = destroysOn(*control);
	for (uint64_t obj : named)
		CHECK(std::count(gone.begin(), gone.end(), obj) == 0);
}

// A command buffer keeps what its encoders name until the host's completion of it is delivered,
// whether it retains references or not, so the host never sees an object's destroy, nor its
// resource id go to another, before a buffer that names it has run. Each object a buffer names
// (a buffer and a texture a blit names, an argument buffer a compute pass binds and a texture it
// makes resident with useResource:usage:, and on the arm64 device a texture committed to a
// residency set the buffer uses), let go by the app once the buffer is committed, reaches the
// host's OBJECT_DESTROY only after the buffer's completed handlers have run, and once; a texture
// made meanwhile takes another resource id. A buffer made with unretained references, by
// commandBufferWithUnretainedReferences or by a descriptor, keeps them as one made with retained
// references does, though it answers retainedReferences NO (RgfxCmdBuffer's
// commandBufferForQueue:, its keeper).
void test_buffers_keep_what_they_name_until_completed()
{
	keepsWhatItNamesUntilCompleted(KeepCase::kUnretained);
	keepsWhatItNamesUntilCompleted(KeepCase::kUnretainedDescriptor);
	keepsWhatItNamesUntilCompleted(KeepCase::kRetained);
}


#if defined(__arm64__)
// A shared event of `q`'s, registered with its queue's reader, at `initial`.
RgfxSharedEvent *queueEvent(Queue &q, uint64_t initial)
{
	RgfxSharedEvent *ev = [RgfxSharedEvent sharedEventWithInitialValue:initial
	                                                             device:nil
	                                                         serializer:&q.ser];
	CHECK(ev != nil);
	[ev rgfxRegisterWithReplyReader:[q.queue sharedReplyReader].get()];
	return ev;
}

// The values `q`'s control ring carries for `event` (SHARED_EVENT_SET_VALUE), in order; every
// message is taken off the ring.
std::vector<uint64_t> setsOf(Queue &q, RgfxObjId event)
{
	std::vector<uint64_t> values;
	RgfxMsgHeader h = {};
	std::vector<uint8_t> bytes;
	while (q.control.next(&h, &bytes)) {
		RgfxSharedEventSetValue m = {};
		if (h.type != RGFX_CMD_SHARED_EVENT_SET_VALUE || bytes.size() < sizeof(m))
			continue;
		std::memcpy(&m, bytes.data(), sizeof(m));
		if (m.obj_id == event)
			values.push_back(m.value);
	}
	return values;
}

void scheduleSignal(id<MTLCommandBuffer> cb, id<MTLEvent> event, uint64_t value)
{
	[(id<QueueTestScheduledSignalSPI>)cb encodeSignalEventScheduled:event value:value];
}

// An arm64 guest's command buffer answers an Apple GPU's SPI encodeSignalEventScheduled:value:,
// which MPSGraph sends with no respondsToSelector: check. Its shared event takes the value as
// the buffer's queue submits it, as Metal's does (measured on the M3 Max, macOS 27.0.1,
// oracle_cmdbuf schedsig): at the commit's post, before the buffer's work or its scheduling,
// whatever the buffer encoded before the call; the host is told first, as the CPU's set tells
// it, then this process's listeners and waits, and the scheduled handler reads the value. The
// value only rises: an event at 5 stays 5 after a signal of 3, and of two the higher stays, in
// either order. After the commit the call sets nothing.
void test_a_scheduled_signal_is_set_as_its_queue_submits_it()
{
	@autoreleasepool {
		Queue q(true);
		Driven driven({&q});
		RgfxSharedEvent *ev = queueEvent(q, 0);
		id<MTLCommandBuffer> cb = [q.queue commandBuffer];
		CHECK([(id)cb respondsToSelector:@selector(encodeSignalEventScheduled:value:)]);
		[[cb blitCommandEncoder] endEncoding];
		scheduleSignal(cb, ev, 3);
		std::atomic<uint64_t> inHandler{0};
		std::atomic<uint64_t> *inHandlerAt = &inHandler;
		[cb addScheduledHandler:^(id<MTLCommandBuffer> c) {
			(void)c;
			inHandlerAt->store(ev.signaledValue);
		}];
		dispatch_semaphore_t heard = dispatch_semaphore_create(0);
		MTLSharedEventListener *listener = [[MTLSharedEventListener alloc] init];
		[ev notifyListener:listener
		           atValue:3
		             block:^(id<MTLSharedEvent> e, uint64_t v) {
			             (void)e;
			             (void)v;
			             dispatch_semaphore_signal(heard);
		             }];
		CHECK(ev.signaledValue == 0);
		CHECK(setsOf(q, ev.rgfxObjId).empty());
		[cb commit];
		CHECK(ev.signaledValue == 3);
		CHECK((setsOf(q, ev.rgfxObjId) == std::vector<uint64_t>{3}));
		CHECK(waitFor(heard, 5) == 0);
		CHECK(cb.status == MTLCommandBufferStatusCommitted);
		const uint64_t seqno = nextSeqno(q);
		hostScheduled(q, *driven.kext, seqno, cb);
		[cb waitUntilScheduled];
		CHECK(inHandler == 3);
		hostComplete(q, *driven.kext, seqno, cb);
		[cb waitUntilCompleted];
		CHECK(cb.status == MTLCommandBufferStatusCompleted);
		CHECK(ev.signaledValue == 3);

		struct Case {
			uint64_t initial;
			std::vector<uint64_t> signals;
			uint64_t after;
		};
		const Case cases[] = {{5, {3}, 5}, {5, {7}, 7}, {5, {6, 8}, 8}, {5, {8, 6}, 8}};
		for (const Case &c : cases) {
			RgfxSharedEvent *e = queueEvent(q, c.initial);
			id<MTLCommandBuffer> b = [q.queue commandBuffer];
			for (uint64_t v : c.signals)
				scheduleSignal(b, e, v);
			[b commit];
			CHECK(e.signaledValue == c.after);
			CHECK(setsOf(q, e.rgfxObjId) == c.signals); // the host keeps the higher too
			const uint64_t s = nextSeqno(q);
			hostComplete(q, *driven.kext, s, b);
			[b waitUntilCompleted];
		}

		RgfxSharedEvent *late = queueEvent(q, 0);
		id<MTLCommandBuffer> b = [q.queue commandBuffer];
		[b commit];
		scheduleSignal(b, late, 1);
		const uint64_t s = nextSeqno(q);
		hostComplete(q, *driven.kext, s, b);
		[b waitUntilCompleted];
		CHECK(late.signaledValue == 0);
		CHECK(setsOf(q, late.rgfxObjId).empty());
		CHECK(b.status == MTLCommandBufferStatusCompleted);
	}
}

// A scheduled signal behind a wait the process does not know met, its buffer's own encoded
// before it or one of an earlier buffer of its queue, is set at its buffer's scheduling, before
// the scheduled handlers, as Metal's queue reaches it only once that wait is met (measured on
// the M3 Max, macOS 27.0.1, oracle_cmdbuf schedsig: 0.04 to 0.15 ms after the CPU set the
// event waited on, before the scheduled handler). One behind a wait met already, or ahead of a
// wait, is set at the post; a wait after it on its own event is then met, and the buffer holds
// up no later signal of its queue (Metal's buffer runs).
void test_a_scheduled_signal_behind_a_wait_is_set_at_its_scheduling()
{
	@autoreleasepool {
		Queue q(true);
		Driven driven({&q});

		RgfxSharedEvent *w = queueEvent(q, 0), *e = queueEvent(q, 0);
		id<MTLCommandBuffer> cb = [q.queue commandBuffer];
		[cb encodeWaitForEvent:w value:1];
		scheduleSignal(cb, e, 1);
		std::atomic<uint64_t> inHandler{0};
		std::atomic<uint64_t> *inHandlerAt = &inHandler;
		[cb addScheduledHandler:^(id<MTLCommandBuffer> c) {
			(void)c;
			inHandlerAt->store(e.signaledValue);
		}];
		[cb commit];
		const uint64_t s1 = nextSeqno(q);
		CHECK(e.signaledValue == 0);
		CHECK(setsOf(q, e.rgfxObjId).empty());
		w.signaledValue = 1; // the host's wait goes
		hostScheduled(q, *driven.kext, s1, cb);
		[cb waitUntilScheduled];
		CHECK(inHandler == 1);
		CHECK(e.signaledValue == 1);
		CHECK((setsOf(q, e.rgfxObjId) == std::vector<uint64_t>{1}));
		hostComplete(q, *driven.kext, s1, cb);
		[cb waitUntilCompleted];

		// Behind an earlier buffer of its queue that waits.
		RgfxSharedEvent *w2 = queueEvent(q, 0), *e2 = queueEvent(q, 0);
		id<MTLCommandBuffer> a = [q.queue commandBuffer];
		[a encodeWaitForEvent:w2 value:1];
		[a commit];
		const uint64_t sa = nextSeqno(q);
		id<MTLCommandBuffer> b = [q.queue commandBuffer];
		scheduleSignal(b, e2, 1);
		[b commit];
		const uint64_t sb = nextSeqno(q);
		CHECK(e2.signaledValue == 0);
		CHECK(setsOf(q, e2.rgfxObjId).empty());
		w2.signaledValue = 1;
		hostScheduled(q, *driven.kext, sa, a);
		[a waitUntilScheduled];
		hostScheduled(q, *driven.kext, sb, b);
		[b waitUntilScheduled];
		CHECK(e2.signaledValue == 1);
		CHECK((setsOf(q, e2.rgfxObjId) == std::vector<uint64_t>{1}));
		hostComplete(q, *driven.kext, sa, a);
		hostComplete(q, *driven.kext, sb, b);
		[b waitUntilCompleted];

		// Behind a wait met already; and ahead of a wait on its own event, which it meets.
		RgfxSharedEvent *met = queueEvent(q, 1), *e3 = queueEvent(q, 0), *own = queueEvent(q, 0);
		id<MTLCommandBuffer> c = [q.queue commandBuffer];
		[c encodeWaitForEvent:met value:1];
		scheduleSignal(c, e3, 1);
		scheduleSignal(c, own, 2);
		[c encodeWaitForEvent:own value:2];
		[c commit];
		const uint64_t sc = nextSeqno(q);
		CHECK(e3.signaledValue == 1);
		CHECK(own.signaledValue == 2);
		// So the queue holds nothing: a later buffer's signal is set at its post.
		RgfxSharedEvent *e4 = queueEvent(q, 0);
		id<MTLCommandBuffer> d = [q.queue commandBuffer];
		scheduleSignal(d, e4, 1);
		[d commit];
		const uint64_t sd = nextSeqno(q);
		CHECK(e4.signaledValue == 1);
		hostComplete(q, *driven.kext, sc, c);
		hostComplete(q, *driven.kext, sd, d);
		[d waitUntilCompleted];
		CHECK(c.status == MTLCommandBufferStatusCompleted);
	}
}

// What the call does with no shared event, as Metal's does (measured on the M3 Max, macOS
// 27.0.1, oracle_cmdbuf schedsig_plain, schedsig_nil, schedsig_nil_unretained): a plain event
// ends the process with Metal's words; nil raises Metal's NSInvalidArgumentException on a buffer
// that retains its references, and fails one that retains none with Metal's error.
void test_a_scheduled_signal_takes_only_a_shared_event()
{
	checkDies("scheduled-signal-plain", "MTLEvent passed to API expecting MTLSharedEvent");
	@autoreleasepool {
		Queue q(true);
		Driven driven({&q});
		id<MTLCommandBuffer> retained = [q.queue commandBuffer];
		bool raised = false;
		@try {
			scheduleSignal(retained, nil, 1);
		} @catch (NSException *ex) {
			raised = [ex.name isEqualToString:NSInvalidArgumentException] &&
			         [ex.reason isEqualToString:@"*** -[__NSArrayM insertObject:atIndex:]: object "
			                                    @"cannot be nil"];
		}
		CHECK(raised);
		[retained commit];
		const uint64_t sr = nextSeqno(q);
		hostComplete(q, *driven.kext, sr, retained);
		[retained waitUntilCompleted];
		CHECK(retained.status == MTLCommandBufferStatusCompleted);

		id<MTLCommandBuffer> unretained = [q.queue commandBufferWithUnretainedReferences];
		scheduleSignal(unretained, nil, 1);
		[unretained commit];
		const uint64_t su = nextSeqno(q);
		hostComplete(q, *driven.kext, su, unretained);
		[unretained waitUntilCompleted];
		CHECK(unretained.status == MTLCommandBufferStatusError);
		CHECK([unretained.error.domain isEqualToString:MTLCommandBufferErrorDomain]);
		CHECK(unretained.error.code == MTLCommandBufferErrorInternal);
		CHECK([unretained.error.localizedDescription
		    isEqualToString:@"Invalid Input (0000000a:kIOGPUCommandBufferCallbackErrorInvalidInput)"]);
	}
}
#endif

} // namespace

void rgfxQueueTests(const char *self)
{
	g_self = self;
	test_encoder_keeps_its_buffer_until_it_ends();
	test_encoder_spi_members();
	test_handlers_after_commit_end_the_process();
	test_handlers_of_queues_wait_for_each_other();
	test_handlers_run_on_the_completion_queue();
	test_buffers_without_handlers_complete_where_seen();
	test_scheduling_without_a_scheduled_handler_runs_where_seen();
	test_one_queues_handlers_run_in_order();
	test_slots_and_submission_do_not_wait_for_handlers();
	test_buffers_run_in_enqueue_order();
	test_threads_commit_in_enqueue_order();
	test_submitted_waits_for_the_buffers_before();
	test_resources_wait_for_their_buffers_in_flight();
	test_buffers_keep_what_they_name_until_completed();
	test_enqueue_and_commit_are_once();
	test_buffers_let_go_give_their_blocks_back();
	test_in_place_writes_retire_at_the_take();
	test_no_copy_write_backs_land_at_the_take();
	test_a_mid_buffer_signal_wakes_a_cpu_wait_before_the_buffer_ends();
	test_async_waits_run_after_their_handlers();
	test_command_buffer_spi_state();
	test_command_queue_spi_state();
	test_queue_priorities_read_back_as_metals();
	test_queue_takes_its_descriptor();
	test_finish_waits_for_committed_buffers();
#if RgfxMakesResidencySets
	test_residency_sets_are_held_until_let_go();
#endif
#if defined(__arm64__)
	test_a_scheduled_signal_is_set_as_its_queue_submits_it();
	test_a_scheduled_signal_behind_a_wait_is_set_at_its_scheduling();
	test_a_scheduled_signal_takes_only_a_shared_event();
#endif
}
