// The native test of RgfxReleaseTable.hpp: releases in flight are answered in any order, an answer
// that comes before its entry is marked posted still finds it, and one not answered in time is
// told of once and stays in flight, holding its block, until its answer retires it. A release
// whose block's pages are named first waits off the ring, holding its entry and its block, while
// the others are posted, answered and retired, and is posted once named.
#include "RgfxReleaseTable.hpp"

#include <cstdio>
#include <cstdlib>

namespace {
int checks = 0;

void check(bool ok, const char *what)
{
	checks++;
	if (!ok) {
		std::fprintf(stderr, "release_table_test: FAILED: %s\n", what);
		std::exit(1);
	}
}

struct Table {
	RgfxReleaseEntry entries[4] = {};
	RgfxReleaseTable t = {entries, 4, 0};
};

// Reserves an entry for the block at `offset`, posts it as seqno `seqno` at `now`.
int32_t post(Table &x, uint64_t offset, uint64_t seqno, uint64_t now)
{
	const int32_t at = rgfxReleaseReserve(x.t);
	if (at < 0)
		return at;
	x.entries[at].offset = offset;
	x.entries[at].seqno = seqno;
	rgfxReleasePosted(x.t, static_cast<uint32_t>(at), now);
	return at;
}
} // namespace

int main()
{
	constexpr uint64_t kTimeout = 1000;
	{
		// Three releases in flight, answered last first.
		Table x;
		check(post(x, 0x1000, 11, 10) >= 0 && post(x, 0x2000, 12, 10) >= 0 &&
		          post(x, 0x3000, 13, 10) >= 0,
		      "three releases posted");
		check(x.t.inFlight == 3, "all three in flight");
		RgfxReleaseEntry e;
		bool answered = false;
		check(!rgfxReleaseTakeDone(x.t, 20, kTimeout, &e, &answered), "none done yet");
		check(rgfxReleaseAnswer(x.t, 13, 0), "the last posted answered first");
		check(rgfxReleaseTakeDone(x.t, 20, kTimeout, &e, &answered) && answered &&
		          e.offset == 0x3000 && e.status == 0,
		      "and retired first, the others still waiting");
		check(!rgfxReleaseTakeDone(x.t, 20, kTimeout, &e, &answered), "the others are not done");
		check(rgfxReleaseAnswer(x.t, 11, 2) && rgfxReleaseAnswer(x.t, 12, 0),
		      "the first two answered, one refused");
		bool sawRefused = false, sawOk = false;
		while (rgfxReleaseTakeDone(x.t, 20, kTimeout, &e, &answered)) {
			check(answered, "an answered release");
			sawRefused |= e.offset == 0x1000 && e.status == 2;
			sawOk |= e.offset == 0x2000 && e.status == 0;
		}
		check(sawRefused && sawOk, "each answer its own release's");
		check(x.t.inFlight == 0, "nothing in flight");
	}
	{
		// The answer comes before the poster marks the entry posted: the post wrote the seqno
		// before the message was published.
		Table x;
		const int32_t at = rgfxReleaseReserve(x.t);
		x.entries[at].offset = 0x5000;
		x.entries[at].seqno = 21;
		check(rgfxReleaseAnswer(x.t, 21, 0), "an answer to a release still posting");
		rgfxReleasePosted(x.t, static_cast<uint32_t>(at), 30);
		check(x.entries[at].state == kRgfxReleaseAnswered, "stays answered once marked posted");
		RgfxReleaseEntry e;
		bool answered = false;
		check(rgfxReleaseTakeDone(x.t, 30, kTimeout, &e, &answered) && answered &&
		          e.offset == 0x5000,
		      "and is retired");
	}
	{
		// What is not a release in flight is not taken.
		Table x;
		post(x, 0x1000, 31, 10);
		check(!rgfxReleaseAnswer(x.t, 0, 0), "seqno 0 names none");
		check(!rgfxReleaseAnswer(x.t, 99, 0), "another message's answer is not a release's");
		check(rgfxReleaseAnswer(x.t, 31, 0), "its own is");
		check(!rgfxReleaseAnswer(x.t, 31, 0), "a second answer to it is not taken");
		const int32_t at = rgfxReleaseReserve(x.t);
		check(!rgfxReleaseAnswer(x.t, 0, 0), "a reserved entry has no seqno to answer");
		rgfxReleaseUnposted(x.t, static_cast<uint32_t>(at));
		check(x.t.inFlight == 1, "an entry not posted is free again");
	}
	{
		// Not answered in time: told of once, the oldest first in the timer's eyes, and kept in
		// flight, so its block and its surface stay the device's, until its answer comes.
		Table x;
		post(x, 0x1000, 41, 100);
		post(x, 0x2000, 42, 500);
		check(rgfxReleaseOldest(x.t) == 100, "the oldest waiting");
		RgfxReleaseEntry e;
		bool answered = true;
		check(!rgfxReleaseTakeDone(x.t, 100 + kTimeout - 1, kTimeout, &e, &answered),
		      "not before its time");
		check(rgfxReleaseTakeDone(x.t, 100 + kTimeout, kTimeout, &e, &answered) && !answered &&
		          e.offset == 0x1000,
		      "told of as late at its time");
		check(x.t.inFlight == 2 && x.entries[0].state == kRgfxReleaseLate,
		      "and still in flight, not retired");
		check(rgfxReleaseOldest(x.t) == 500, "the next oldest is the timer's");
		check(!rgfxReleaseTakeDone(x.t, 100 + 100 * kTimeout, kTimeout, &e, &answered) ||
		          e.offset != 0x1000,
		      "a late release is told of once, however long it waits");
		check(x.t.inFlight == 2, "and nothing was retired for the wait");
		check(rgfxReleaseAnswer(x.t, 41, 2), "the late answer finds its release");
		check(rgfxReleaseTakeDone(x.t, 100 + 100 * kTimeout, kTimeout, &e, &answered) &&
		          answered && e.offset == 0x1000 && e.status == 2,
		      "which it retires with its status");
		check(x.t.inFlight == 1, "only then is its entry free");
	}
	{
		// Never answered: the release holds its entry for good, and the table's room shrinks by
		// it, rather than its block going out again.
		Table x;
		post(x, 0x1000, 45, 100);
		RgfxReleaseEntry e;
		bool answered = true;
		check(rgfxReleaseTakeDone(x.t, 100 + kTimeout, kTimeout, &e, &answered) && !answered,
		      "late");
		for (uint64_t now = 100 + kTimeout; now < 100 + 1000 * kTimeout; now += kTimeout)
			check(!rgfxReleaseTakeDone(x.t, now, kTimeout, &e, &answered),
			      "a release never answered is never retired");
		check(x.t.inFlight == 1, "it stays in flight");
		check(rgfxReleaseOldest(x.t) == 0, "and arms no timer");
		check(post(x, 0x2000, 46, 200) >= 0 && rgfxReleaseAnswer(x.t, 46, 0) &&
		          rgfxReleaseTakeDone(x.t, 300, kTimeout, &e, &answered) && answered &&
		          e.offset == 0x2000,
		      "the other entries go on");
	}
	{
		// A full table reserves nothing until an entry is retired.
		Table x;
		for (uint64_t i = 0; i < 4; i++)
			check(post(x, 0x1000 * (i + 1), 50 + i, 10) >= 0, "a release posted");
		check(rgfxReleaseReserve(x.t) < 0, "the table is full");
		check(rgfxReleaseAnswer(x.t, 52, 0), "one answered");
		RgfxReleaseEntry e;
		bool answered = false;
		check(rgfxReleaseTakeDone(x.t, 20, kTimeout, &e, &answered), "and retired");
		check(rgfxReleaseReserve(x.t) >= 0, "which frees an entry");
	}
	{
		// A release whose deferred block's pages are named first is handed to the naming worker:
		// no answer, lateness or timer looks at it, and the releases after it go on.
		Table x;
		const int32_t naming = rgfxReleaseReserve(x.t);
		check(naming >= 0, "an entry reserved for a block whose pages are named first");
		x.entries[naming].offset = 0x5000;
		x.entries[naming].serial = 7;
		check(rgfxReleaseNextNaming(x.t) < 0, "nothing to name before it is handed over");
		rgfxReleaseNaming(x.t, static_cast<uint32_t>(naming));
		check(rgfxReleaseNextNaming(x.t) == naming, "the naming worker finds it");
		check(rgfxReleaseTakeNamed(x.t) < 0, "not named yet: the release worker posts nothing");
		check(!rgfxReleaseAnswer(x.t, 0, 0), "no seqno: no answer finds it");
		const int32_t other = post(x, 0x6000, 61, 10);
		check(other >= 0 && other != naming, "the next block's release is posted meanwhile");
		check(rgfxReleaseOldest(x.t) == 10, "the timer is the posted release's alone");
		RgfxReleaseEntry e;
		bool answered = false;
		check(rgfxReleaseTakeDone(x.t, 10 + kTimeout, kTimeout, &e, &answered) && !answered &&
		          e.offset == 0x6000,
		      "the posted one is late, the naming one never");
		check(!rgfxReleaseTakeDone(x.t, 10 + 100 * kTimeout, kTimeout, &e, &answered),
		      "however long the naming takes");
		check(rgfxReleaseAnswer(x.t, 61, 0) &&
		          rgfxReleaseTakeDone(x.t, 20, kTimeout, &e, &answered) && answered &&
		          e.offset == 0x6000,
		      "and the posted one is answered and retired while the naming waits");
		check(x.t.inFlight == 1 && x.entries[naming].state == kRgfxReleaseNaming,
		      "the naming one holds its entry");
		rgfxReleaseNamed(x.t, static_cast<uint32_t>(naming), 2);
		check(rgfxReleaseNextNaming(x.t) < 0, "named: nothing left to name");
		check(rgfxReleaseTakeNamed(x.t) == naming && x.entries[naming].attach == 2 &&
		          x.entries[naming].offset == 0x5000 && x.entries[naming].serial == 7 &&
		          x.entries[naming].state == kRgfxReleasePosting,
		      "the release worker takes it back, posting, with its block and what the naming did");
		check(rgfxReleaseTakeNamed(x.t) < 0, "once");
		x.entries[naming].seqno = 62;
		rgfxReleasePosted(x.t, static_cast<uint32_t>(naming), 30);
		check(rgfxReleaseAnswer(x.t, 62, 0) &&
		          rgfxReleaseTakeDone(x.t, 40, kTimeout, &e, &answered) && answered &&
		          e.offset == 0x5000 && x.t.inFlight == 0,
		      "then it is posted, answered and retired as any other");
		rgfxReleaseNamed(x.t, static_cast<uint32_t>(naming), 1);
		check(rgfxReleaseTakeNamed(x.t) < 0, "a naming done for an entry not naming is not taken");
	}
	std::printf("release_table_test: OK (%d checks)\n", checks);
	return 0;
}
