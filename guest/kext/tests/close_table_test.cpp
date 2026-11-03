// The native test of RgfxCloseTable.hpp and RgfxIdChunks.hpp: a close answered in time is taken
// at once; one not answered in time holds its context's GPU address units and resource-id chunks,
// which no one else takes, until its answer, however late, lets them go as one in time would; and
// one never answered holds them for good.
#include "RgfxBlockTable.hpp"
#include "RgfxCloseTable.hpp"
#include "RgfxContextSlots.hpp"
#include "RgfxIdChunks.hpp"

#include <cstdio>
#include <cstdlib>

namespace {
int checks = 0;

void check(bool ok, const char *what)
{
	checks++;
	if (!ok) {
		std::fprintf(stderr, "close_table_test: FAILED: %s\n", what);
		std::exit(1);
	}
}

struct Closes {
	RgfxCloseEntry entries[4] = {};
	RgfxCloseTable t = {entries, 4};
};

// The kext's state of a client's context, as RgfxAcceleratorContexts.cpp keeps it.
struct State {
	RgfxContextSlots slots = kRgfxContextSlotsEmpty;
	const void *ids[8] = {};
};

const char kDead = 0, kNew = 0;
const void *const dead = &kDead; // the closing client, whose address a new client may take
const void *const newer = &kNew; // a client opened after the close

// A late close of `owner`'s context: posted, not answered in time, its holdings handed to its
// token and parked. Its entry's index.
uint32_t lateClose(Closes &c, State &s, const void *owner, uint32_t first, uint32_t units)
{
	const int32_t at = rgfxCloseReserve(c.t, 7, first, units);
	check(at >= 0, "a close reserved");
	c.entries[at].seqno = 70 + static_cast<uint64_t>(at);
	uint32_t status = 0;
	check(!rgfxCloseInTime(c.t, static_cast<uint32_t>(at), &status), "not answered in time");
	const void *token = rgfxCloseToken(c.t, static_cast<uint32_t>(at));
	rgfxIdChunksHand(s.ids, 8, owner, token);
	check(!rgfxClosePark(c.t, static_cast<uint32_t>(at)), "parked before any answer");
	return static_cast<uint32_t>(at);
}

// A context's holdings: its units and two id chunks.
void open(State &s, const void *owner, uint32_t *first)
{
	check(rgfxContextSlotsTake(s.slots, 4, false, first), "the context's units");
	check(rgfxIdChunkTake(s.ids, 8, owner) == 0 && rgfxIdChunkTake(s.ids, 8, owner) == 1,
	      "its two id chunks");
}
} // namespace

int main()
{
	{
		// Answered while its thread waits: taken at once, the entry free again.
		Closes c;
		const int32_t at = rgfxCloseReserve(c.t, 3, 0, 1);
		c.entries[at].seqno = 40;
		check(!rgfxCloseAnswered(c.t, static_cast<uint32_t>(at)), "not yet answered");
		check(!rgfxCloseAnswer(c.t, 0, 0) && !rgfxCloseAnswer(c.t, 41, 0),
		      "seqno 0 and another message's answer are not the close's");
		check(rgfxCloseAnswer(c.t, 40, 0) && rgfxCloseAnswered(c.t, static_cast<uint32_t>(at)),
		      "its answer");
		check(!rgfxCloseAnswer(c.t, 40, 0), "a second answer is not taken");
		uint32_t status = 9;
		check(rgfxCloseInTime(c.t, static_cast<uint32_t>(at), &status) && status == 0,
		      "in time, with its status");
		check(c.entries[at].state == kRgfxCloseFree && rgfxCloseLateCount(c.t) == 0,
		      "and its entry is free");
		uint32_t taken = 0;
		check(!rgfxCloseTakeLate(c.t, &taken), "the worker has nothing to take");
	}
	{
		// Never answered: the context's units and id chunks stay its token's for good, none taken
		// by a client opened after it, even one at the dead client's address.
		Closes c;
		State s;
		uint32_t first = 0;
		open(s, dead, &first);
		const uint32_t at = lateClose(c, s, dead, first, 4);
		const void *token = rgfxCloseToken(c.t, at);
		uint32_t taken = 0;
		for (int i = 0; i < 100; i++)
			check(!rgfxCloseTakeLate(c.t, &taken), "a close never answered is never taken");
		check(rgfxCloseLateCount(c.t) == 1, "it stays late");
		// The GPU address units: no later context's run meets them.
		for (;;) {
			uint32_t got = 0;
			if (!rgfxContextSlotsTake(s.slots, 4, true, &got))
				break;
			check(got + 4 <= first || got >= first + 4, "a later run never meets the late units");
		}
		check(rgfxContextRunIs(s.slots, first, 4, true), "the late close's units stay held");
		// The id chunks: a client at the dead one's address frees none, and new takes skip them.
		check(rgfxIdChunksHand(s.ids, 8, dead, nullptr) == 0,
		      "a client at the dead one's address frees none of its chunks");
		for (int i = 0; i < 6; i++)
			check(rgfxIdChunkTake(s.ids, 8, newer) >= 2, "a new client takes other chunks");
		check(rgfxIdChunkTake(s.ids, 8, newer) < 0 && s.ids[0] == token && s.ids[1] == token,
		      "the late close's two stay its own");
	}
	{
		// Answered late, before the closing thread parked it: the worker takes it only once
		// parked, then lets go of what it held as an answer in time does.
		Closes c;
		State s;
		uint32_t first = 0;
		open(s, dead, &first);
		const int32_t at = rgfxCloseReserve(c.t, 7, first, 4);
		c.entries[at].seqno = 90;
		uint32_t status = 0;
		check(!rgfxCloseInTime(c.t, static_cast<uint32_t>(at), &status), "late");
		check(rgfxCloseAnswer(c.t, 90, 0), "its late answer finds it");
		uint32_t taken = 0;
		check(!rgfxCloseTakeLate(c.t, &taken), "not taken before its holdings are its token's");
		const void *token = rgfxCloseToken(c.t, static_cast<uint32_t>(at));
		rgfxIdChunksHand(s.ids, 8, dead, token);
		check(rgfxClosePark(c.t, static_cast<uint32_t>(at)), "parked, it asks for the worker");
		check(rgfxCloseTakeLate(c.t, &taken) && taken == static_cast<uint32_t>(at) &&
		          c.entries[at].status == 0,
		      "which takes it, answered OK");
		const int32_t next = rgfxCloseReserve(c.t, 8, 0, 1);
		check(next >= 0 && next != at,
		      "its entry is no other close's while the worker lets go of its holdings");
		rgfxCloseUnposted(c.t, static_cast<uint32_t>(next));
		// The worker lets go of all of it, as the closing thread does for an answer in time.
		check(rgfxIdChunksHand(s.ids, 8, token, nullptr) == 2 && s.ids[0] == nullptr,
		      "its id chunks are free");
		rgfxContextSlotsGive(s.slots, c.entries[at].firstUnit, c.entries[at].units);
		check(rgfxContextRunIs(s.slots, first, 4, false), "its units are given back");
		rgfxCloseRetired(c.t, taken);
		check(c.entries[at].state == kRgfxCloseFree && rgfxCloseLateCount(c.t) == 0,
		      "and its entry is free once it is done with");
		uint32_t again = 0;
		check(rgfxContextSlotsTake(s.slots, 4, true, &again) && rgfxIdChunkTake(s.ids, 8, newer) == 0,
		      "a new context may take them now");
	}
	{
		// Answered late after it was parked: the answer asks for the worker by itself.
		Closes c;
		State s;
		uint32_t first = 0;
		open(s, dead, &first);
		const uint32_t at = lateClose(c, s, dead, first, 4);
		uint32_t taken = 0;
		check(!rgfxCloseTakeLate(c.t, &taken), "nothing to take before its answer");
		check(rgfxCloseAnswer(c.t, c.entries[at].seqno, 2), "a late refusal");
		check(rgfxCloseTakeLate(c.t, &taken) && taken == at && c.entries[at].status == 2,
		      "taken with its status: the worker keeps its holdings from use for good");
		check(rgfxIdChunksHand(s.ids, 8, rgfxCloseToken(c.t, at), rgfxBlockNoOwner()) == 2,
		      "kept for good, no longer the token's");
		rgfxCloseRetired(c.t, taken);
		check(rgfxIdChunksHand(s.ids, 8, rgfxCloseToken(c.t, at), nullptr) == 0,
		      "the token's entry reused holds none of them");
	}
	{
		// A full table reserves nothing; the caller then waits as for any lifecycle message.
		Closes c;
		for (uint32_t i = 0; i < 4; i++)
			check(rgfxCloseReserve(c.t, i, i, 1) >= 0, "a close");
		check(rgfxCloseReserve(c.t, 9, 9, 1) < 0, "the table is full");
		rgfxCloseUnposted(c.t, 2);
		check(rgfxCloseReserve(c.t, 9, 9, 1) == 2, "a close not posted frees its entry");
	}
	std::printf("close_table_test: OK (%d checks)\n", checks);
	return 0;
}
