/*
 * known_pipeline_tests: pipeline creates against the device's known pipelines
 * (RgfxKnownPipelines.hpp), part of the Metal shell's check (shell_tests.mm calls
 * rgfxKnownPipelineTests; guest/bundle/Makefile `test`). The first create of a descriptor waits
 * for its answer, and its answer is kept once the device accepted it; an equal create is posted
 * without waiting, its state taking the kept reflection at once and its gpuResourceID, its own,
 * waiting at the first read. Each create is a new object with its own resource ID, as Metal's
 * are. A refusal is never kept; where the kext keeps no slot, or no known pipelines are given, a
 * create waits as before.
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

#import "RgfxComputePipelineState.h"
#import "RgfxLibrary.h"
#import "RgfxRenderPipelineState.h"

#include "RgfxKnownPipelines.hpp"
#include "RgfxObjectModel.hpp"
#include "RgfxReplyReader.hpp"
#include "RgfxRingWriter.hpp"
#include "RgfxSerializer.hpp"
#include "reims_graphics_wire.h"
#include "rgfx_test_ring.hpp" // CHECK, g_checks

void rgfxKnownPipelineTests();

namespace {

// A kext stand-in: the control ring forwards each message, answering a pipeline create at once
// (rgfxKextSubmit's route) or, posted deferred, keeping its answer until a take (the
// accelerator's TakeAnswers). Each answer gives the next resource ID. Never freed: states a test
// lets go post their destroys through it after the test.
struct Kext {
	RgfxReplyReader &reader = *new RgfxReplyReader();
	uint64_t seqno = 0;
	uint64_t nextResourceId = 100;
	uint32_t status = RGFX_PIPELINE_OK; // the next pipeline answers'
	bool slotsFull = false;             // the deferred post answers at once, as with no slot
	bool takeBringsNothing = false;     // a take answers nothing (the answer never came)
	std::vector<uint32_t> waited;       // the pipeline creates answered at once, by type
	std::vector<uint64_t> deferred;     // the seqnos posted deferred
	std::vector<RgfxPipelineReflection> held;
	int takes = 0;
	RgfxIdSpace &ids = *new RgfxIdSpace();
	RgfxRingWriter &control = *new RgfxRingWriter(&Kext::forward, this, 0x1000);
	RgfxSerializer &ser = *new RgfxSerializer(control, ids);
	std::shared_ptr<RgfxKnownPipelines> known = std::make_shared<RgfxKnownPipelines>();

	explicit Kext(bool defers = true)
	{
		if (defers)
			control.setDeferred(&Kext::post, &Kext::take, this);
	}

	static bool isPipeline(uint32_t type)
	{
		return type == RGFX_CMD_RENDER_PIPELINE_CREATE || type == RGFX_CMD_COMPUTE_PIPELINE_CREATE;
	}

	RgfxPipelineReflection answerOf(const void *message)
	{
		RgfxMsgHeader h = {};
		std::memcpy(&h, message, sizeof(h));
		uint64_t obj = 0;
		std::memcpy(&obj, static_cast<const uint8_t *>(message) + sizeof(h), sizeof(obj));
		RgfxPipelineReflection r = {};
		r.header = { RGFX_EVT_PIPELINE_REFLECTION, sizeof(r), seqno };
		r.obj_id = obj;
		r.status = status;
		if (status == RGFX_PIPELINE_OK) {
			r.gpu_resource_id = nextResourceId++;
			r.max_total_threads_per_threadgroup = 512;
			r.thread_execution_width = 32;
		}
		r.kind = h.type == RGFX_CMD_COMPUTE_PIPELINE_CREATE ? RGFX_PIPELINE_KIND_COMPUTE
		                                                     : RGFX_PIPELINE_KIND_RENDER;
		return r;
	}

	static bool forward(void *context, const void *message, uint32_t size, uint64_t *out_seqno,
	                    uint32_t *out_status)
	{
		(void)size;
		Kext *k = static_cast<Kext *>(context);
		RgfxMsgHeader h = {};
		std::memcpy(&h, message, sizeof(h));
		*out_seqno = ++k->seqno;
		if (isPipeline(h.type)) {
			k->waited.push_back(h.type);
			RgfxPipelineReflection r = k->answerOf(message);
			k->reader.deliver(&r, sizeof(r));
		} else if (out_status) {
			*out_status = RGFX_STATUS_OK;
		}
		return true;
	}

	static bool post(void *context, const void *message, uint32_t size, uint64_t *out_seqno,
	                 bool *out_deferred)
	{
		(void)size;
		Kext *k = static_cast<Kext *>(context);
		RgfxMsgHeader h = {};
		std::memcpy(&h, message, sizeof(h));
		CHECK(isPipeline(h.type)); // only a pipeline create is posted so
		*out_seqno = ++k->seqno;
		RgfxPipelineReflection r = k->answerOf(message);
		if (k->slotsFull) {
			*out_deferred = false;
			k->waited.push_back(h.type);
			k->reader.deliver(&r, sizeof(r));
			return true;
		}
		*out_deferred = true;
		k->deferred.push_back(k->seqno);
		k->held.push_back(r);
		return true;
	}

	static bool take(void *context, uint64_t seqno)
	{
		(void)seqno;
		Kext *k = static_cast<Kext *>(context);
		k->takes++;
		if (k->takeBringsNothing)
			return true;
		std::vector<RgfxPipelineReflection> held;
		held.swap(k->held);
		for (const RgfxPipelineReflection &r : held)
			k->reader.deliver(&r, sizeof(r));
		return true;
	}

	RgfxLibrary *library()
	{
		const uint8_t bytes[] = {7, 7, 7, 7};
		uint32_t libStatus = 0;
		return [RgfxLibrary libraryWithData:bytes
		                             length:sizeof(bytes)
		                                key:nullptr
		                             device:nil
		                         serializer:&ser
		                             status:&libStatus];
	}

	// A render state of `desc`, registered as the device registers it.
	RgfxRenderPipelineState *render(MTLRenderPipelineDescriptor *desc, bool withKnown = true)
	{
		NSError *err = nil;
		RgfxRenderPipelineState *s =
		    [RgfxRenderPipelineState renderPipelineStateWithDescriptor:desc
		                                                        device:nil
		                                                    serializer:&ser
		                                                         known:withKnown ? known : nullptr
		                                                         error:&err];
		CHECK(s != nil && err == nil);
		[s rgfxRegisterWithReplyReader:&reader];
		return s;
	}

	RgfxComputePipelineState *compute(id<MTLFunction> fn)
	{
		NSError *err = nil;
		RgfxComputePipelineState *s =
		    [RgfxComputePipelineState computePipelineStateWithFunction:fn
		                                                        device:nil
		                                                    serializer:&ser
		                                                         known:known
		                                                         error:&err];
		CHECK(s != nil && err == nil);
		[s rgfxRegisterWithReplyReader:&reader];
		return s;
	}
};

MTLRenderPipelineDescriptor *quad(RgfxLibrary *lib, NSString *label)
{
	MTLRenderPipelineDescriptor *desc = [[MTLRenderPipelineDescriptor alloc] init];
	desc.vertexFunction = [lib newFunctionWithName:@"v_quad"];
	desc.fragmentFunction = [lib newFunctionWithName:@"f_tex"];
	desc.label = label;
	desc.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
	return desc;
}

// The first create waits and its answer is kept; an equal one is posted without waiting, takes
// the kept reflection at once, and its gpuResourceID is its own, taken at the first read only.
void test_an_equal_create_does_not_wait()
{
	Kext k;
	RgfxLibrary *lib = k.library();
	RgfxRenderPipelineState *first = k.render(quad(lib, @"quad"));
	CHECK(k.waited.size() == 1 && k.deferred.empty());
	CHECK(!first.rgfxDeferred && !first.rgfxRefused);
	CHECK(first.gpuResourceID._impl == 100);
	CHECK(k.known->size() == 1);

	RgfxRenderPipelineState *second = k.render(quad(lib, @"quad"));
	CHECK(k.waited.size() == 1 && k.deferred.size() == 1); // posted, not waited for
	CHECK(second.rgfxDeferred && !second.rgfxRefused);
	CHECK(second.rgfxObjId != first.rgfxObjId);              // a new object
	CHECK([second maxTotalThreadsPerThreadgroup] == 512);    // the kept reflection, at once
	CHECK(k.takes == 0);
	CHECK(second.gpuResourceID._impl == 101); // its own ID, from its own answer
	CHECK(k.takes == 1);
	CHECK(second.gpuResourceID._impl == 101 && k.takes == 1); // taken once
	CHECK(first.gpuResourceID._impl == 100);
}

// An answer another thread took before the state registered (a drain) is kept by the reader and
// handed over at the registration: the first read needs no take.
void test_an_answer_taken_before_the_registration()
{
	Kext k;
	RgfxLibrary *lib = k.library();
	(void)k.render(quad(lib, @"quad"));
	NSError *err = nil;
	RgfxRenderPipelineState *s =
	    [RgfxRenderPipelineState renderPipelineStateWithDescriptor:quad(lib, @"quad")
	                                                        device:nil
	                                                    serializer:&k.ser
	                                                         known:k.known
	                                                         error:&err];
	CHECK(s.rgfxDeferred);
	Kext::take(&k, 0); // the drain
	[s rgfxRegisterWithReplyReader:&k.reader];
	CHECK(s.gpuResourceID._impl == 101 && k.takes == 1);
}

// A descriptor that differs in any field (the label here) is not known; a create with no known
// pipelines, or over a route that keeps no answers, waits each time.
void test_creates_that_wait()
{
	{
		Kext k;
		RgfxLibrary *lib = k.library();
		(void)k.render(quad(lib, @"quad"));
		RgfxRenderPipelineState *other = k.render(quad(lib, @"other"));
		CHECK(!other.rgfxDeferred && k.waited.size() == 2);
		RgfxRenderPipelineState *none = k.render(quad(lib, @"quad"), false);
		CHECK(!none.rgfxDeferred && k.waited.size() == 3);
	}
	{
		Kext k(false); // the kext keeps no answers: nothing is keyed
		RgfxLibrary *lib = k.library();
		(void)k.render(quad(lib, @"quad"));
		RgfxRenderPipelineState *again = k.render(quad(lib, @"quad"));
		CHECK(!again.rgfxDeferred && k.waited.size() == 2 && k.known->size() == 0);
	}
}

// A refused create is not kept: its equal create waits, and is refused at the create again.
void test_a_refusal_is_not_kept()
{
	Kext k;
	RgfxLibrary *lib = k.library();
	k.status = RGFX_PIPELINE_REFUSED;
	RgfxRenderPipelineState *refused = k.render(quad(lib, @"quad"));
	CHECK(refused.rgfxRefused && k.known->size() == 0);
	RgfxRenderPipelineState *again = k.render(quad(lib, @"quad"));
	CHECK(!again.rgfxDeferred && again.rgfxRefused && k.waited.size() == 2);
}

// Where the kext kept no slot, an equal create is answered at once, as any.
void test_no_slot_answers_at_once()
{
	Kext k;
	RgfxLibrary *lib = k.library();
	(void)k.render(quad(lib, @"quad"));
	k.slotsFull = true;
	RgfxRenderPipelineState *s = k.render(quad(lib, @"quad"));
	CHECK(!s.rgfxDeferred && k.waited.size() == 2 && k.deferred.empty());
	CHECK(s.gpuResourceID._impl == 101 && k.takes == 0);
}

// A deferred create the device refused (the platform could not make its own pipeline) is no
// refusal at the create; its gpuResourceID is 0. One whose answer never comes is 0 too, and is
// not waited for again.
void test_a_deferred_create_with_no_pipeline()
{
	Kext k;
	RgfxLibrary *lib = k.library();
	(void)k.render(quad(lib, @"quad"));
	k.status = RGFX_PIPELINE_REFUSED;
	RgfxRenderPipelineState *refused = k.render(quad(lib, @"quad"));
	CHECK(refused.rgfxDeferred && !refused.rgfxRefused);
	CHECK(refused.gpuResourceID._impl == 0 && k.takes == 1);

	k.status = RGFX_PIPELINE_OK;
	k.takeBringsNothing = true;
	RgfxRenderPipelineState *lost = k.render(quad(lib, @"quad"));
	CHECK(lost.rgfxDeferred);
	CHECK(lost.gpuResourceID._impl == 0 && k.takes == 2);
	CHECK(lost.gpuResourceID._impl == 0 && k.takes == 2); // not waited for again
}

// A compute state of a function the process made a pipeline of before: posted without waiting,
// its reflection at once, its own ID at the first read.
void test_an_equal_compute_create_does_not_wait()
{
	Kext k;
	RgfxLibrary *lib = k.library();
	id<MTLFunction> fn = [lib newFunctionWithName:@"k_blur"];
	RgfxComputePipelineState *first = k.compute(fn);
	CHECK(!first.rgfxDeferred && first.gpuResourceID._impl == 100);
	RgfxComputePipelineState *second = k.compute(fn);
	CHECK(second.rgfxDeferred && k.waited.size() == 1);
	CHECK(second.threadExecutionWidth == 32 && k.takes == 0);
	CHECK(second.gpuResourceID._impl == 101 && k.takes == 1);
}

// The key: every byte of the message but its header and object id, and its type.
void test_the_key()
{
	std::vector<uint8_t> a(64, 0), b(64, 0);
	for (size_t i = 0; i < a.size(); i++)
		a[i] = b[i] = static_cast<uint8_t>(i);
	b[4] = 0xee;  // the header's size
	b[8] = 0xee;  // its seqno
	b[16] = 0xee; // the object id
	const auto ka = RgfxKnownPipelines::keyOf(RGFX_CMD_RENDER_PIPELINE_CREATE, a.data(), 64);
	CHECK(ka == RgfxKnownPipelines::keyOf(RGFX_CMD_RENDER_PIPELINE_CREATE, b.data(), 64));
	CHECK(ka != RgfxKnownPipelines::keyOf(RGFX_CMD_COMPUTE_PIPELINE_CREATE, a.data(), 64));
	b[40] = 0xee; // a descriptor byte
	CHECK(ka != RgfxKnownPipelines::keyOf(RGFX_CMD_RENDER_PIPELINE_CREATE, b.data(), 64));
	CHECK(ka != RgfxKnownPipelines::keyOf(RGFX_CMD_RENDER_PIPELINE_CREATE, a.data(), 56));
}

} // namespace

void rgfxKnownPipelineTests()
{
	@autoreleasepool {
		test_the_key();
		test_an_equal_create_does_not_wait();
		test_an_answer_taken_before_the_registration();
		test_creates_that_wait();
		test_a_refusal_is_not_kept();
		test_no_slot_answers_at_once();
		test_a_deferred_create_with_no_pipeline();
		test_an_equal_compute_create_does_not_wait();
	}
}
