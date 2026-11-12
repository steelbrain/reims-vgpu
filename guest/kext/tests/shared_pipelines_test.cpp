// The native test of RgfxSharedPipelines.hpp: the device's answer to a pipeline create that
// carries a made pipeline's key is kept by that key, and an equal create of any process finds it,
// stamped with its own seqno and object id, every field the device gives each create alone
// zeroed; a refusal, a mesh answer and an answer with no key are not kept; a set full of other
// keys replaces the one found or learnt longest ago, so the table never holds more than its size.
#include "RgfxSharedPipelines.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {
int checks = 0;

void check(bool ok, const char *what)
{
	checks++;
	if (!ok) {
		std::fprintf(stderr, "shared_pipelines_test: FAILED: %s\n", what);
		std::exit(1);
	}
}

typedef RgfxSharedPipelines<2, 2> Table;
constexpr uint32_t kBytes = Table::kKeyedReplyBytes;

// Key `k` landing in set `set` of a 2-set table: its first four bytes `set`, the rest `k`.
void keyOf(uint8_t key[32], uint8_t k, uint32_t set)
{
	std::memset(key, k, 32);
	std::memcpy(key, &set, sizeof(set));
}

// A device's answer to the create at `seqno` of object `obj`: made, of a compute pipeline, with
// key `key` after it where `keyed`.
void answerOf(uint8_t reply[kBytes], const uint8_t key[32], uint64_t seqno, uint64_t obj,
              bool keyed = true)
{
	RgfxPipelineReflection r = {};
	r.header.type = RGFX_EVT_PIPELINE_REFLECTION;
	r.header.size = keyed ? kBytes : sizeof(r);
	r.header.seqno = seqno;
	r.obj_id = obj;
	r.status = RGFX_PIPELINE_OK;
	r.kind = RGFX_PIPELINE_KIND_COMPUTE;
	r.flags = RGFX_PIPELINE_REFLECT_SUPPORT_ICB | (keyed ? RGFX_PIPELINE_REFLECT_KEYED : 0);
	r.gpu_resource_id = 0x55;
	r.unique_identifier = 0x66;
	r.resource_index = 0x77;
	r.max_total_threads_per_threadgroup = 1024;
	r.thread_execution_width = 32;
	r.static_threadgroup_memory_length = 4096;
	std::memset(reply, 0, kBytes);
	std::memcpy(reply, &r, sizeof(r));
	std::memcpy(reply + RGFX_PIPELINE_REFLECTION_KEY_OFFSET, key, 32);
}
} // namespace

int main()
{
	static_assert(Table::kKeyedReplyBytes == sizeof(RgfxPipelineReflection) + 32,
	              "the key follows the reflection");
	static Table t;
	uint8_t reply[kBytes];
	uint8_t a[32], b[32], c[32], d[32];
	keyOf(a, 0xa0, 0);
	keyOf(b, 0xb0, 0);
	keyOf(c, 0xc0, 0);
	keyOf(d, 0xd0, 1);
	RgfxPipelineReflection out;

	// Kept by its key, found by any create of it, as that create's own.
	answerOf(reply, a, 41, (1ull << 40) | 7);
	check(t.learn(reply, kBytes), "a made pipeline's keyed answer is kept");
	check(t.size() == 1, "one answer kept");
	check(t.find(a, 900, (3ull << 40) | 9, &out), "an equal create finds it");
	check(out.header.type == RGFX_EVT_PIPELINE_REFLECTION &&
	          out.header.size == sizeof(RgfxPipelineReflection),
	      "the plain reflection, no key after it");
	check(out.header.seqno == 900 && out.obj_id == ((3ull << 40) | 9),
	      "stamped with the finding create's seqno and object");
	check(out.gpu_resource_id == 0 && out.unique_identifier == 0 && out.resource_index == 0,
	      "what the device gives each create alone is not kept");
	check(out.status == RGFX_PIPELINE_OK && out.kind == RGFX_PIPELINE_KIND_COMPUTE &&
	          out.max_total_threads_per_threadgroup == 1024 && out.thread_execution_width == 32 &&
	          out.static_threadgroup_memory_length == 4096,
	      "the rest of the reflection is the device's");
	check((out.flags & RGFX_PIPELINE_REFLECT_SUPPORT_ICB) != 0, "its flags are the device's");
	check(!t.find(b, 1, 2, &out), "another key finds nothing");

	// Not kept: a refusal, a mesh answer, an answer without a key, a short one, another event.
	answerOf(reply, b, 1, 2);
	RgfxPipelineReflection r;
	std::memcpy(&r, reply, sizeof(r));
	r.status = RGFX_PIPELINE_REFUSED;
	std::memcpy(reply, &r, sizeof(r));
	check(!t.learn(reply, kBytes), "a refusal is not kept");
	answerOf(reply, b, 1, 2);
	std::memcpy(&r, reply, sizeof(r));
	r.kind = RGFX_PIPELINE_KIND_MESH;
	std::memcpy(reply, &r, sizeof(r));
	check(!t.learn(reply, kBytes), "a mesh answer is not kept");
	answerOf(reply, b, 1, 2, false);
	check(!t.learn(reply, kBytes), "an answer without a key is not kept");
	answerOf(reply, b, 1, 2);
	check(!t.learn(reply, kBytes - 1), "a reply shorter than the key's end is not kept");
	std::memcpy(&r, reply, sizeof(r));
	r.header.size = sizeof(r);
	std::memcpy(reply, &r, sizeof(r));
	check(!t.learn(reply, kBytes), "a reply whose size ends before the key is not kept");
	answerOf(reply, b, 1, 2);
	std::memcpy(&r, reply, sizeof(r));
	r.header.type = RGFX_EVT_PIPELINE_REFLECTION + 1;
	std::memcpy(reply, &r, sizeof(r));
	check(!t.learn(reply, kBytes), "another event is not kept");
	check(t.size() == 1 && !t.find(b, 1, 2, &out), "none of them was kept");

	// Learning a kept key again keeps one entry, the newer answer.
	answerOf(reply, a, 42, 8);
	std::memcpy(&r, reply, sizeof(r));
	r.thread_execution_width = 64;
	std::memcpy(reply, &r, sizeof(r));
	check(t.learn(reply, kBytes), "an answer of a kept key is kept again");
	check(t.size() == 1 && t.find(a, 1, 1, &out) && out.thread_execution_width == 64,
	      "one entry, the newer answer");

	// Set 0 holds two: a third key there replaces the one found or learnt longest ago.
	answerOf(reply, b, 2, 3);
	check(t.learn(reply, kBytes), "b kept");
	check(t.find(a, 1, 1, &out), "a found, so b is the one longest ago");
	answerOf(reply, c, 3, 4);
	check(t.learn(reply, kBytes), "c kept");
	check(t.find(a, 1, 1, &out) && t.find(c, 1, 1, &out) && !t.find(b, 1, 1, &out),
	      "b replaced, a and c kept");
	// Set 1 is untouched by set 0's replacements.
	answerOf(reply, d, 4, 5);
	check(t.learn(reply, kBytes), "d kept in set 1");
	check(t.size() == 3, "a, c and d");

	// The table never holds more than its size, however many keys it learns.
	for (uint32_t i = 0; i < 64; i++) {
		uint8_t k[32];
		keyOf(k, static_cast<uint8_t>(i), i);
		answerOf(reply, k, i, i);
		t.learn(reply, kBytes);
	}
	check(t.size() == 4, "2 sets of 2 hold 4 answers at most");

	// A start forgets every answer.
	t.clear();
	check(t.size() == 0 && !t.find(a, 1, 1, &out), "cleared");

	std::printf("shared_pipelines_test: %d checks passed\n", checks);
	return 0;
}
