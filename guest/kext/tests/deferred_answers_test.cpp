// The native test of RgfxDeferredAnswers.hpp: a deferred pipeline create's answer is kept in the
// slot reserved for it, whatever order answers come in, until its context takes it; a context
// holds at most its share of the slots, the table at most its size; a cancelled or dropped slot is
// free again, and an answer no slot waits for is not kept.
#include "RgfxDeferredAnswers.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {
int checks = 0;

void check(bool ok, const char *what)
{
	checks++;
	if (!ok) {
		std::fprintf(stderr, "deferred_answers_test: FAILED: %s\n", what);
		std::exit(1);
	}
}

constexpr uint32_t kReplyMax = 128;
typedef RgfxDeferredAnswers<8, 4, kReplyMax> Answers;

// Stamps slot `slot`'s seqno, as the post does.
void stamp(Answers &a, int slot, uint64_t seqno)
{
	__atomic_store_n(a.seqnoOf(slot), seqno, __ATOMIC_RELEASE);
}

// The reflection answering the create at `seqno` with resource id `id`, given to `a`.
bool answer(Answers &a, uint64_t seqno, uint64_t id, uint32_t type = RGFX_EVT_PIPELINE_REFLECTION)
{
	RgfxPipelineReflection r = {};
	r.header.type = type;
	r.header.size = sizeof(r);
	r.header.seqno = seqno;
	r.gpu_resource_id = id;
	r.status = RGFX_PIPELINE_OK;
	const auto *bytes = reinterpret_cast<const uint8_t *>(&r);
	return a.answer(r.header, bytes + sizeof(RgfxMsgHeader),
	                static_cast<uint32_t>(sizeof(r) - sizeof(RgfxMsgHeader)));
}

// The answers `out` holds, `length` bytes of whole reflections: their seqnos and ids.
void read(const uint8_t *out, uint32_t length, uint64_t *seqnos, uint64_t *ids, uint32_t *n)
{
	*n = 0;
	for (uint32_t at = 0; at < length;) {
		RgfxPipelineReflection r;
		std::memcpy(&r, out + at, sizeof(r));
		check(r.header.size == sizeof(r), "each answer is whole, its size its own");
		seqnos[*n] = r.header.seqno;
		ids[*n] = r.gpu_resource_id;
		(*n)++;
		at += r.header.size;
	}
}
} // namespace

int main()
{
	static Answers a;
	// A context's slots: its share, then none. Context 0 is no context.
	int s[4];
	for (int i = 0; i < 4; i++) {
		s[i] = a.reserve(7);
		check(s[i] >= 0, "a context takes its share");
	}
	check(a.reserve(7) < 0, "and no more");
	check(a.reserve(0) < 0, "context 0 takes none");
	check(a.held(7) == 4, "the context holds four");

	// Answers in any order find their slots by seqno; a slot not yet stamped finds none.
	stamp(a, s[0], 10);
	stamp(a, s[1], 11);
	stamp(a, s[2], 12);
	check(a.state(7, 11) == Answers::kWaiting, "posted, not answered");
	check(answer(a, 12, 0x112), "the third's answer is kept");
	check(answer(a, 10, 0x110), "the first's");
	check(!answer(a, 13, 0x113), "no slot waits for seqno 13");
	check(!answer(a, 0, 0x100), "nor for seqno 0, an unstamped slot's");
	check(!answer(a, 11, 0x111, RGFX_EVT_COMPLETE), "an answer of another type is no slot's");
	check(!answer(a, 10, 0x999), "a second answer to one create is not kept");
	check(a.state(7, 10) == Answers::kAnswered && a.state(7, 11) == Answers::kWaiting,
	      "the first answered, the second waiting");
	check(a.state(8, 10) == Answers::kAbsent, "another context's create is none of these");

	// A take copies the answers, in seqno order, and frees their slots.
	uint8_t out[512];
	uint32_t count = 0;
	uint32_t length = a.take(7, out, sizeof(out), &count);
	uint64_t seqnos[8], ids[8];
	uint32_t n = 0;
	read(out, length, seqnos, ids, &n);
	check(count == 2 && n == 2, "two answers taken");
	check(seqnos[0] == 10 && ids[0] == 0x110 && seqnos[1] == 12 && ids[1] == 0x112,
	      "in seqno order, each its own");
	check(a.state(7, 10) == Answers::kAbsent && a.held(7) == 2, "their slots are free");
	check(a.take(7, out, sizeof(out), &count) == 0 && count == 0, "nothing more to take");

	// A take of what does not fit leaves it for the next.
	check(answer(a, 11, 0x111), "the second's answer");
	length = a.take(7, out, sizeof(RgfxPipelineReflection) - 8, &count);
	check(length == 0 && count == 0 && a.state(7, 11) == Answers::kAnswered, "kept for later");
	length = a.take(7, out, sizeof(out), &count);
	check(count == 1 && length == sizeof(RgfxPipelineReflection), "taken then");

	// A cancelled slot (its create not posted) is free; a dropped context's are all free, its
	// answers with them; other contexts keep theirs.
	a.cancel(s[3]);
	check(a.held(7) == 0, "every slot of the context free");
	int other = a.reserve(9);
	stamp(a, other, 40);
	int mine = a.reserve(7);
	stamp(a, mine, 41);
	check(answer(a, 41, 0x141), "answered");
	a.drop(7);
	check(a.state(7, 41) == Answers::kAbsent && a.held(7) == 0, "dropped with its answer");
	check(!answer(a, 41, 0x141), "a dropped create's answer is not kept");
	check(a.state(9, 40) == Answers::kWaiting, "another context's stays");

	// A create posted after its context's close: its own slot goes, whichever context holds the
	// number now, and no other.
	int late = a.reserve(7);
	stamp(a, late, 42);
	int next = a.reserve(7);
	stamp(a, next, 43);
	a.forget(42);
	check(a.state(7, 42) == Answers::kAbsent && a.state(7, 43) == Answers::kWaiting,
	      "the late create's slot alone is free");
	a.forget(0);
	check(a.held(7) == 1 && a.state(9, 40) == Answers::kWaiting, "seqno 0 frees none");
	a.drop(7);

	// The table's size bounds every context together.
	int held = 1; // context 9's
	for (uint32_t c = 20; c < 24; c++)
		for (int i = 0; i < 4; i++)
			held += a.reserve(c) >= 0 ? 1 : 0;
	check(held == 8, "the table holds its 8 slots and no more");

	// An answer longer than a slot keeps its first ReplyMax bytes, its size saying so.
	static RgfxDeferredAnswers<1, 1, 24> small;
	int one = small.reserve(3);
	__atomic_store_n(small.seqnoOf(one), uint64_t{5}, __ATOMIC_RELEASE);
	RgfxPipelineReflection r = {};
	r.header.type = RGFX_EVT_PIPELINE_REFLECTION;
	r.header.seqno = 5;
	r.obj_id = 0x77;
	check(small.answer(r.header, reinterpret_cast<const uint8_t *>(&r) + sizeof(RgfxMsgHeader),
	                   static_cast<uint32_t>(sizeof(r) - sizeof(RgfxMsgHeader))),
	      "a long answer is kept");
	length = small.take(3, out, sizeof(out), &count);
	RgfxMsgHeader h;
	std::memcpy(&h, out, sizeof(h));
	uint64_t obj = 0;
	std::memcpy(&obj, out + sizeof(h), sizeof(obj));
	check(length == 24 && h.size == 24 && obj == 0x77, "cut to the slot, its size the cut");

	std::printf("deferred_answers_test: %d checks passed\n", checks);
	return 0;
}
