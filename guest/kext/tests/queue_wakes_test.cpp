// The native test of RgfxQueueWakes.hpp: the filter's look at a watched completion ring and the
// work loop's after it. Every interleaving of the host's posts, each interrupt's two looks (the
// filter's as it arrives, the work loop's pass after it) and the client's completion thread (it
// returns from its wait with the count, then drains the ring, then waits with that count) ends
// with every record taken; one completion's interrupt wakes its client once; and a completion
// posted between an interrupt's two looks is woken for by the work loop's.
//
// The model raises the vector at every post, as the host does while the ring is armed; the arm
// itself is the completion thread's (completion_arm_test.cpp tests ring 0's).
#include "RgfxQueueWakes.hpp"

#include <cstdio>
#include <cstdlib>

namespace {
int checks = 0;

void check(bool ok, const char *what)
{
	checks++;
	if (!ok) {
		std::fprintf(stderr, "queue_wakes_test: FAILED: %s\n", what);
		std::exit(1);
	}
}

struct Model {
	// The ring's header.
	uint64_t producer = 0;
	uint64_t consumer = 0;
	// The looks' last producers, as the kext keeps them (RgfxWatchedRing).
	uint64_t filterSeen = 0;
	uint64_t loopSeen = 0;
	// The client's wake count, and the count its completion thread waits with; whether it has
	// returned from the wait and has the ring to drain.
	uint64_t count = 0;
	uint64_t waitedWith = 0;
	bool draining = false;
	// The vector raised and its filter not yet run; filters run whose work loop pass has not.
	bool raised = false;
	bool loopOwed = false;
	int posts = 0;
	// Whether the work loop's look skips what the filter looked at (the rule), or not (the
	// control).
	bool dedupe = true;

	void post()
	{
		producer++;
		posts++;
		raised = true;
	}
	void filter()
	{
		raised = false;
		loopOwed = true;
		if (rgfxFilterWakes(&filterSeen, producer, [this] { return consumer; }))
			count++;
	}
	void loop()
	{
		loopOwed = false;
		const uint64_t none = ~0ull;
		if (rgfxLoopWakes(&loopSeen, dedupe ? &filterSeen : &none, producer,
		                  [this] { return consumer; }))
			count++;
	}
	bool woken() const { return !draining && count != waitedWith; }
	void wake()
	{
		waitedWith = count;
		draining = true;
	}
	void drain()
	{
		consumer = producer;
		draining = false;
	}
};

constexpr int kMaxPosts = 3;
long runs = 0;

// Every interleaving from `m`: at its end nothing can run, and every record must be taken.
void explore(const Model &m)
{
	bool any = false;
	if (m.posts < kMaxPosts) {
		Model n = m;
		n.post();
		explore(n);
		any = true;
	}
	if (m.raised) {
		Model n = m;
		n.filter();
		explore(n);
		any = true;
	}
	if (m.loopOwed) {
		Model n = m;
		n.loop();
		explore(n);
		any = true;
	}
	if (m.woken()) {
		Model n = m;
		n.wake();
		explore(n);
		any = true;
	}
	if (m.draining) {
		Model n = m;
		n.drain();
		explore(n);
		any = true;
	}
	if (!any) {
		runs++;
		check(m.consumer == m.producer, "a record is left untaken with its client asleep");
	}
}

// One completion, its interrupt's two looks, then the client: one wake, the filter's.
void testOneCompletionWakesOnce(bool dedupe, uint64_t wakes)
{
	Model m;
	m.dedupe = dedupe;
	m.post();
	m.filter();
	check(m.count == 1, "the filter's look wakes the client for a new completion");
	m.loop();
	check(m.count == wakes, dedupe ? "the work loop's look skips what the filter woke for"
	                               : "without the rule, the work loop's look wakes again");
	m.wake();
	m.drain();
	check(m.consumer == m.producer && !m.woken(), "the client takes the record, and sleeps");
}

// A completion posted after the filter's look and before the work loop's: the work loop's look
// wakes the client for it, and the next interrupt's filter, finding the client has taken it,
// does not.
void testPostBetweenTheLooks()
{
	Model m;
	m.post();
	m.filter();
	m.wake();
	m.drain();
	m.post();
	m.loop();
	check(m.count == 2, "the work loop's look wakes for a completion the filter did not see");
	m.wake();
	m.drain();
	m.filter();
	check(m.count == 2, "a filter finding every record taken wakes nothing");
	m.loop();
	check(m.count == 2 && m.consumer == m.producer, "and nor does its work loop's look");
}

// A completion the client took before either look (its pass was running): no wake.
void testTakenBeforeTheLooks()
{
	Model m;
	m.post();
	m.drain();
	m.filter();
	m.loop();
	check(m.count == 0, "a ring whose records are all taken wakes nothing");
}

// Before the filter is hooked, or on an interrupt it did not see, the work loop's look alone
// wakes, as before.
void testWorkLoopAlone()
{
	Model m;
	m.post();
	m.raised = false;
	m.loop();
	check(m.count == 1, "the work loop's look wakes where the filter has not looked");
	m.loop();
	check(m.count == 1, "and only once per producer");
}
} // namespace

int main()
{
	explore(Model{});
	Model control;
	control.dedupe = false;
	explore(control);
	testOneCompletionWakesOnce(true, 1);
	testOneCompletionWakesOnce(false, 2);
	testPostBetweenTheLooks();
	testTakenBeforeTheLooks();
	testWorkLoopAlone();
	std::printf("queue_wakes_test: %d checks passed (%ld interleavings)\n", checks, runs);
	return 0;
}
