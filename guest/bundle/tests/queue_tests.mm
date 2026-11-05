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
#import "RgfxIOSurface.h"
#import "RgfxMetalPrivate.h"
#import "RgfxResource.h"
#import "RgfxSharedEvent.h"

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

@interface QueueTestHandle : RgfxResource <MTLBuffer, MTLTexture>
@end

// The command-queue SPI WindowServer sets its queues' completion queues with.
@protocol QueueTestCompletionQueueSPI
- (void)setCompletionQueue:(dispatch_queue_t)queue;
- (dispatch_queue_t)completionQueue;
@end

@implementation QueueTestHandle
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
		std::thread late([a] {
			std::this_thread::sleep_for(std::chrono::milliseconds(50));
			[a commit];
		});
		const auto start = std::chrono::steady_clock::now();
		CHECK([(id<MTLCommandBufferSPI>)b commitAndWaitUntilSubmitted]);
		CHECK(std::chrono::steady_clock::now() - start >= std::chrono::milliseconds(40));
		late.join();
		host.join();
		CHECK((order == std::vector<uint64_t>{objId(a), objId(b)}));
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
bool oneIdChunk(void *context, uint64_t *first)
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
		                                     uint32_t surfaceId, bool, uint64_t *offset,
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
// its completion does: one encoded past what its queue's ring takes and never committed, and one
// posted out of line on a queue whose completion ring nothing reads. Buffer after buffer reuse
// one block, and the queue's going gives it back to the region. (A buffer let go so used to keep
// its block, and the region lost one a buffer.)
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
		CHECK(blocks.made <= 1);
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

int rgfxQueueDiesCase(const char *name)
{
	signal(SIGABRT, [](int) { _exit(kAborted); });
	@autoreleasepool {
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

void rgfxQueueTests(const char *self)
{
	g_self = self;
	test_encoder_keeps_its_buffer_until_it_ends();
	test_handlers_after_commit_end_the_process();
	test_handlers_of_queues_wait_for_each_other();
	test_handlers_run_on_the_completion_queue();
	test_buffers_without_handlers_complete_where_seen();
	test_one_queues_handlers_run_in_order();
	test_slots_and_submission_do_not_wait_for_handlers();
	test_buffers_run_in_enqueue_order();
	test_threads_commit_in_enqueue_order();
	test_submitted_waits_for_the_buffers_before();
	test_enqueue_and_commit_are_once();
	test_buffers_let_go_give_their_blocks_back();
	test_in_place_writes_retire_at_the_take();
	test_no_copy_write_backs_land_at_the_take();
	test_a_mid_buffer_signal_wakes_a_cpu_wait_before_the_buffer_ends();
}
