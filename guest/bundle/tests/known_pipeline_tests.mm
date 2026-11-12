/*
 * known_pipeline_tests: pipeline creates against the device's known pipelines
 * (RgfxKnownPipelines.hpp), part of the Metal shell's check (shell_tests.mm calls
 * rgfxKnownPipelineTests; guest/bundle/Makefile `test`). The first create of a descriptor waits
 * for its answer, and its answer is kept once the device accepted it; an equal create is posted
 * without waiting, its state taking the kept reflection at once and its gpuResourceID, its own,
 * waiting at the first read. Each create is a new object with its own resource ID, as Metal's
 * are. A refusal is never kept; where the kext keeps no slot, or no known pipelines are given, a
 * create waits as before. Where the route shares answers (RgfxRingWriter::setShared), a create of
 * another process's equal create is posted without waiting too, by its pipeline key
 * (rgfxSharedPipelineKey), which the host's computes alike.
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
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
// accelerator's TakeAnswers). Each answer gives the next resource ID. Where it shares answers, a
// create posted by its key (rgfxKextSubmitShared's route) finds the answer kept for that key by
// any process's stand-in, and is posted deferred, or is answered at once and its answer kept.
// Never freed: states a test lets go post their destroys through it after the test.
struct Kext {
	// The answers kept by pipeline key, as the kext keeps them (RgfxSharedPipelines.hpp).
	typedef std::map<RgfxPipelineKey, RgfxPipelineReflection> Kept;

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
	std::shared_ptr<Kept> kept;              // where it shares answers, every process's
	std::vector<RgfxPipelineKey> sharedKeys; // the keys creates were posted by
	std::vector<uint8_t> lastShared;         // the last create posted by its key
	std::vector<uint32_t> createFlags;       // each pipeline create's flags, in order

	explicit Kext(bool defers = true, std::shared_ptr<Kept> sharing = nullptr)
	{
		if (defers)
			control.setDeferred(&Kext::post, &Kext::take, this);
		if (sharing) {
			kept = std::move(sharing);
			control.setShared(&Kext::shared);
			ser.setSharesPipelines(control.shares());
		}
	}

	// Notes a pipeline create's flags.
	void noteFlags(const void *message)
	{
		RgfxMsgHeader h = {};
		std::memcpy(&h, message, sizeof(h));
		const size_t at = h.type == RGFX_CMD_COMPUTE_PIPELINE_CREATE
		                      ? offsetof(RgfxComputePipelineCreate, flags)
		                      : offsetof(RgfxRenderPipelineCreate, flags);
		uint32_t flags = 0;
		std::memcpy(&flags, static_cast<const uint8_t *>(message) + at, sizeof(flags));
		createFlags.push_back(flags);
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
			k->noteFlags(message);
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
		k->noteFlags(message);
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

	static bool shared(void *context, const void *message, uint32_t size, const uint8_t *key,
	                   uint64_t *out_seqno, bool *out_deferred,
	                   RgfxPipelineReflection *out_reflection)
	{
		Kext *k = static_cast<Kext *>(context);
		RgfxPipelineKey named;
		std::memcpy(named.data(), key, named.size());
		k->sharedKeys.push_back(named);
		k->lastShared.assign(static_cast<const uint8_t *>(message),
		                     static_cast<const uint8_t *>(message) + size);
		auto found = k->kept->find(named);
		if (found != k->kept->end()) {
			if (!post(context, message, size, out_seqno, out_deferred))
				return false;
			if (*out_deferred) {
				*out_reflection = found->second;
				out_reflection->header.seqno = *out_seqno;
				std::memcpy(&out_reflection->obj_id,
				            static_cast<const uint8_t *>(message) + sizeof(RgfxMsgHeader),
				            sizeof(uint64_t));
			}
			return true;
		}
		// Answered at once, and kept where the device made the pipeline.
		RgfxMsgHeader h = {};
		std::memcpy(&h, message, sizeof(h));
		k->noteFlags(message);
		*out_seqno = ++k->seqno;
		*out_deferred = false;
		k->waited.push_back(h.type);
		RgfxPipelineReflection r = k->answerOf(message);
		k->reader.deliver(&r, sizeof(r));
		if (r.status == RGFX_PIPELINE_OK) {
			r.header.seqno = 0;
			r.obj_id = 0;
			r.gpu_resource_id = 0;
			(*k->kept)[named] = r;
		}
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

	// A library of content key `k` (32 bytes of it) over the same bytes as library()'s.
	RgfxLibrary *keyedLibrary(uint8_t k)
	{
		const uint8_t bytes[] = {7, 7, 7, 7};
		uint8_t key[32];
		std::memset(key, k, sizeof(key));
		uint32_t libStatus = 0;
		return [RgfxLibrary libraryWithData:bytes
		                             length:sizeof(bytes)
		                                key:key
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

// The shared vector of the pipeline key, the host's (pipeline_key.rs,
// pipeline_key_of_the_shared_vector): the two sides key a create alike. Kernel `k` of library
// (1 << 40) | 2, whose content key's bytes are 0x10, 0x11, ..., asking for its key, as object
// (1 << 40) | 7 at seqno 41.
void test_the_shared_key_vector()
{
	RgfxComputePipelineCreate c = {};
	std::vector<uint8_t> m(sizeof(c) + 8, 0);
	c.header.type = RGFX_CMD_COMPUTE_PIPELINE_CREATE;
	c.header.size = static_cast<uint32_t>(m.size());
	c.header.seqno = 41;
	c.obj_id = (1ull << 40) | 7;
	c.function_library_obj_id = (1ull << 40) | 2;
	c.name_offset = sizeof(c);
	c.name_len = 1;
	c.constants_offset = sizeof(c) + 1;
	c.linked_offset = sizeof(c) + 1;
	c.flags = RGFX_COMPUTE_PIPELINE_ANSWER_KEY;
	std::memcpy(m.data(), &c, sizeof(c));
	m[sizeof(c)] = 'k';
	auto keys = [](uint64_t id, RgfxLibraryContentKey *k) {
		if (id != ((1ull << 40) | 2))
			return false;
		for (size_t i = 0; i < k->size(); i++)
			(*k)[i] = static_cast<uint8_t>(0x10 + i);
		return true;
	};
	RgfxPipelineKey key;
	const uint32_t compute = RGFX_CMD_COMPUTE_PIPELINE_CREATE;
	CHECK(rgfxSharedPipelineKey(compute, m.data(), static_cast<uint32_t>(m.size()), keys, &key));
	char hex[65] = {};
	for (size_t i = 0; i < key.size(); i++)
		std::snprintf(hex + 2 * i, 3, "%02x", key[i]);
	CHECK(std::strcmp(hex, "2be3bb91e78f55b505082cb4c42a647817694dcb5cf47b9974d643f958e991f7") == 0);

	// Another process's equal create (other ids and seqno) has the same key.
	RgfxComputePipelineCreate other = c;
	other.header.seqno = 900;
	other.obj_id = (3ull << 40) | 9;
	other.function_library_obj_id = (3ull << 40) | 5;
	std::vector<uint8_t> o = m;
	std::memcpy(o.data(), &other, sizeof(other));
	RgfxPipelineKey same;
	auto sameKeys = [](uint64_t, RgfxLibraryContentKey *k) {
		for (size_t i = 0; i < k->size(); i++)
			(*k)[i] = static_cast<uint8_t>(0x10 + i);
		return true;
	};
	CHECK(rgfxSharedPipelineKey(compute, o.data(), static_cast<uint32_t>(o.size()), sameKeys, &same));
	CHECK(same == key);
	// The header is the ring's, whatever the bytes hold before it is stamped.
	std::vector<uint8_t> unstamped = m;
	std::memset(unstamped.data(), 0, sizeof(RgfxMsgHeader));
	CHECK(rgfxSharedPipelineKey(compute, unstamped.data(), static_cast<uint32_t>(unstamped.size()), keys, &same));
	CHECK(same == key);

	// None where the create does not ask for one, a library has no key, it links dynamic
	// libraries, it is shorter than its create, or it is of another type.
	RgfxPipelineKey none;
	auto noKeys = [](uint64_t, RgfxLibraryContentKey *) { return false; };
	CHECK(!rgfxSharedPipelineKey(compute, m.data(), static_cast<uint32_t>(m.size()), noKeys, &none));
	RgfxComputePipelineCreate unasked = c;
	unasked.flags = 0;
	std::vector<uint8_t> u = m;
	std::memcpy(u.data(), &unasked, sizeof(unasked));
	CHECK(!rgfxSharedPipelineKey(compute, u.data(), static_cast<uint32_t>(u.size()), keys, &none));
	RgfxComputePipelineCreate dynamic = c;
	dynamic.library_count = 1;
	std::vector<uint8_t> d = m;
	std::memcpy(d.data(), &dynamic, sizeof(dynamic));
	CHECK(!rgfxSharedPipelineKey(compute, d.data(), static_cast<uint32_t>(d.size()), keys, &none));
	CHECK(!rgfxSharedPipelineKey(compute, m.data(), 64, keys, &none));
	CHECK(!rgfxSharedPipelineKey(RGFX_CMD_MESH_RENDER_PIPELINE_CREATE, m.data(),
	                             static_cast<uint32_t>(m.size()), keys, &none));
}

// A create another process's equal create was answered for: the first create of a key waits and
// its answer is kept; the equal create of another process, over another library object of the
// same content, is posted without waiting, its reflection the kept one at once and its
// gpuResourceID its own, taken at the first read; that process knows the pipeline from then on.
// A library of other content keys another pipeline, which waits.
void test_a_create_another_process_made_does_not_wait()
{
	auto kept = std::make_shared<Kext::Kept>();
	Kext a(true, kept);
	RgfxComputePipelineState *first = a.compute([a.keyedLibrary(1) newFunctionWithName:@"k_blur"]);
	CHECK(!first.rgfxDeferred && a.waited.size() == 1 && a.sharedKeys.size() == 1);
	CHECK(kept->size() == 1 && first.gpuResourceID._impl == 100);
	CHECK((a.createFlags.back() & RGFX_COMPUTE_PIPELINE_ANSWER_KEY) != 0);

	Kext b(true, kept);
	b.nextResourceId = 200;
	(void)b.keyedLibrary(9); // so b's library has another id than a's
	RgfxLibrary *lib = b.keyedLibrary(1);
	id<MTLFunction> fn = [lib newFunctionWithName:@"k_blur"];
	RgfxComputePipelineState *second = b.compute(fn);
	CHECK(b.sharedKeys.size() == 1 && b.sharedKeys[0] == a.sharedKeys[0]);
	CHECK(second.rgfxDeferred && b.waited.empty() && b.deferred.size() == 1);
	CHECK(!second.rgfxRefused);
	CHECK(second.threadExecutionWidth == 32 && b.takes == 0); // the kept reflection, at once
	CHECK(second.gpuResourceID._impl == 200 && b.takes == 1); // its own ID, from its own answer
	// Known to b from then on: its next equal create is posted without asking by key.
	RgfxComputePipelineState *third = b.compute(fn);
	CHECK(third.rgfxDeferred && b.sharedKeys.size() == 1 && b.deferred.size() == 2);

	// A library of another content key: another pipeline key, which waits and is kept.
	RgfxComputePipelineState *other = b.compute([b.keyedLibrary(2) newFunctionWithName:@"k_blur"]);
	CHECK(!other.rgfxDeferred && b.waited.size() == 1 && b.sharedKeys.size() == 2);
	CHECK(b.sharedKeys[1] != a.sharedKeys[0] && kept->size() == 2);
}

// A render create is shared as a compute create is, and the key the kext is asked by is the
// create's own: rgfxSharedPipelineKey of the message, naming its libraries by their content keys.
// Where the kext keeps no slot, a create it keeps the answer for is answered at once.
void test_a_shared_render_create()
{
	auto kept = std::make_shared<Kext::Kept>();
	Kext a(true, kept);
	RgfxRenderPipelineState *first = a.render(quad(a.keyedLibrary(3), @"quad"));
	CHECK(!first.rgfxDeferred && a.sharedKeys.size() == 1 && kept->size() == 1);
	CHECK((a.createFlags.back() & RGFX_RENDER_PIPELINE_ANSWER_KEY) != 0);
	RgfxPipelineKey key;
	auto keys = [](uint64_t, RgfxLibraryContentKey *k) {
		k->fill(3);
		return true;
	};
	CHECK(rgfxSharedPipelineKey(RGFX_CMD_RENDER_PIPELINE_CREATE, a.lastShared.data(),
	                            static_cast<uint32_t>(a.lastShared.size()), keys, &key));
	CHECK(key == a.sharedKeys.back());

	Kext b(true, kept);
	RgfxRenderPipelineState *second = b.render(quad(b.keyedLibrary(3), @"quad"));
	CHECK(second.rgfxDeferred && b.waited.empty() && b.sharedKeys.back() == key);
	CHECK([second maxTotalThreadsPerThreadgroup] == 512);

	Kext c(true, kept);
	c.slotsFull = true;
	RgfxRenderPipelineState *third = c.render(quad(c.keyedLibrary(3), @"quad"));
	CHECK(!third.rgfxDeferred && c.waited.size() == 1 && c.sharedKeys.back() == key);
	CHECK(third.gpuResourceID._impl == 100 && c.takes == 0);
}

// Creates asked by no key: one naming a library with no content key, which the device keys no
// pipeline of (it still asks the device for its key, which the device gives only where it holds
// every library by key); and any where the route shares nothing, which do not ask the device.
void test_creates_not_shared()
{
	{
		auto kept = std::make_shared<Kext::Kept>();
		Kext k(true, kept);
		RgfxComputePipelineState *s = k.compute([k.library() newFunctionWithName:@"k_blur"]);
		CHECK(!s.rgfxDeferred && k.sharedKeys.empty() && k.waited.size() == 1 && kept->empty());
		CHECK((k.createFlags.back() & RGFX_COMPUTE_PIPELINE_ANSWER_KEY) != 0);
	}
	{
		Kext k; // a route that shares nothing
		RgfxLibrary *lib = k.keyedLibrary(1);
		(void)k.compute([lib newFunctionWithName:@"k_blur"]);
		CHECK(k.sharedKeys.empty() && (k.createFlags.back() & RGFX_COMPUTE_PIPELINE_ANSWER_KEY) == 0);
		(void)k.render(quad(lib, @"quad"));
		CHECK(k.sharedKeys.empty() && (k.createFlags.back() & RGFX_RENDER_PIPELINE_ANSWER_KEY) == 0);
	}
	{
		Kext k(false, std::make_shared<Kext::Kept>()); // no deferred posts: nothing shared
		CHECK(!k.ser.sharesPipelines());
		(void)k.compute([k.keyedLibrary(1) newFunctionWithName:@"k_blur"]);
		CHECK(k.sharedKeys.empty() && (k.createFlags.back() & RGFX_COMPUTE_PIPELINE_ANSWER_KEY) == 0);
	}
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
		test_the_shared_key_vector();
		test_a_create_another_process_made_does_not_wait();
		test_a_shared_render_create();
		test_creates_not_shared();
	}
}
