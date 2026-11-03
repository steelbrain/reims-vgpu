/*
 * queue_tests: the command queue's and the command buffer's contract with their encoders,
 * against Metal's answers, part of the Metal shell's check (shell_tests.mm calls
 * rgfxQueueTests; guest/bundle/Makefile `test`). Metal's answers these hold to were measured on
 * the M3 Max (macOS 27) and on the MacBook Pro 2019's AMD Radeon Pro 5300M and Intel UHD 630
 * (macOS 26.7.1), which agree on each.
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <crt_externs.h>
#include <csignal>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <spawn.h>
#include <string>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
#include <vector>

#import "RgfxCommandObjects.h"
#import "RgfxMetalPrivate.h"
#import "RgfxResource.h"

#include "RgfxCompletionThread.hpp"
#include "RgfxObjectModel.hpp"
#include "RgfxReplyReader.hpp"
#include "RgfxRingWriter.hpp"
#include "RgfxSerializer.hpp"
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

} // namespace

int rgfxQueueDiesCase(const char *name)
{
	signal(SIGABRT, [](int) { _exit(kAborted); });
	@autoreleasepool {
		Queue q(true);
		id<MTLCommandBuffer> cb = [q.queue commandBuffer];
		[cb commit];
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

void rgfxQueueTests(const char *self)
{
	g_self = self;
	test_encoder_keeps_its_buffer_until_it_ends();
	test_handlers_after_commit_end_the_process();
	test_handlers_of_queues_wait_for_each_other();
	test_handlers_run_on_the_completion_queue();
	test_one_queues_handlers_run_in_order();
	test_slots_and_submission_do_not_wait_for_handlers();
}
