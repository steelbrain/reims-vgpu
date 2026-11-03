/*
 * queue_tests: the command queue's and the command buffer's contract with their encoders,
 * against Metal's answers, part of the Metal shell's check (shell_tests.mm calls
 * rgfxQueueTests; guest/bundle/Makefile `test`). Metal's answers these hold to were measured on
 * the M3 Max (macOS 27) and on the MacBook Pro 2019's AMD Radeon Pro 5300M and Intel UHD 630
 * (macOS 26.7.1), which agree on each.
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <crt_externs.h>
#include <csignal>
#include <cstdint>
#include <cstring>
#include <spawn.h>
#include <string>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

#import "RgfxCommandObjects.h"
#import "RgfxResource.h"

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

	explicit Queue(bool reader = false)
	{
		control.setConsumerIdle(1);
		submit.setConsumerIdle(1);
		queue = [RgfxCommandQueue queueWithSubmitRing:&submitWriter
		                                 submitOffset:0
		                               completeOffset:0
		                             completeDataSize:0
		                                          ids:&ids
		                                     maxCount:0
		                                       device:nil
		                                   serializer:&ser];
		if (reader)
			[queue adoptReplyReader:new RgfxReplyReader(completion.base(), completion.dataSize)];
	}
};

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
}
