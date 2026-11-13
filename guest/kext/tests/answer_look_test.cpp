// The native test of RgfxAnswerLook.hpp: a waiter's look for its answer takes the ring only where
// a record waits and the work loop's gate is free, pauses otherwise, ends as the answer comes,
// and ends at its deadline without one.
#include "RgfxAnswerLook.hpp"

#include <cstdio>
#include <cstdlib>

namespace {
int checks = 0;

void check(bool ok, const char *what)
{
	checks++;
	if (!ok) {
		std::fprintf(stderr, "answer_look_test: FAILED: %s\n", what);
		std::exit(1);
	}
}

// A look's world: a clock that moves one tick a call, a record that lands after `landsAfter`
// pauses, an answer that comes with the take of the record where `answers`, and a gate another
// thread holds for the first `gateHeldFor` takes.
struct World {
	uint64_t clock = 0;
	int pauses = 0;
	int takes = 0;
	int landsAfter = 0;
	int gateHeldFor = 0;
	int refused = 0;
	bool answers = true;
	bool answered = false;
	bool pendingRecord = false;

	bool look(uint64_t deadline)
	{
		return rgfxLookForAnswer(
		    deadline, [this] { return clock++; }, [this] { return answered; },
		    [this] { return pendingRecord; },
		    [this] {
			    if (refused < gateHeldFor) {
				    refused++;
				    return false;
			    }
			    takes++;
			    pendingRecord = false;
			    answered = answers;
			    return true;
		    },
		    [this] {
			    if (++pauses == landsAfter)
				    pendingRecord = true;
		    });
	}
};
} // namespace

int main()
{
	// Answered before the look: nothing taken, no pause.
	{
		World w;
		w.answered = true;
		check(w.look(100), "an answer already there ends the look");
		check(w.takes == 0 && w.pauses == 0, "and takes nothing, pausing never");
	}
	// The record lands after three pauses: one take, which answers.
	{
		World w;
		w.landsAfter = 3;
		check(w.look(100), "a record that lands answers the look");
		check(w.takes == 1 && w.pauses == 3, "with one take, after the pauses before it");
	}
	// No record within the deadline: the look ends unanswered, having taken nothing.
	{
		World w;
		check(!w.look(10), "no answer by the deadline ends the look unanswered");
		check(w.takes == 0 && w.pauses == 10, "having paused until it");
	}
	// A record that answers another waiter: taken, and the look goes on to its deadline.
	{
		World w;
		w.landsAfter = 2;
		w.answers = false;
		check(!w.look(10), "another's answer does not end the look");
		check(w.takes == 1, "but its record is taken");
	}
	// A gate another thread holds: the take does not run, the look pauses and takes the record
	// once the gate is free, never waiting for it.
	{
		World w;
		w.landsAfter = 1;
		w.gateHeldFor = 3;
		check(w.look(100), "a record behind a held gate answers the look once it is free");
		check(w.takes == 1 && w.refused == 3 && w.pauses == 4,
		      "after a pause for each take the held gate refused");
	}
	// A gate held to the deadline: the look ends unanswered, the record left to the work loop.
	{
		World w;
		w.landsAfter = 1;
		w.gateHeldFor = 1000;
		check(!w.look(20), "a gate held to the deadline ends the look unanswered");
		check(w.takes == 0 && w.pendingRecord, "leaving the record for the work loop");
	}
	// The bound a pipeline create's thread spends looking is the constant's.
	check(kRgfxAnswerLookNs == 150 * 1000, "a waiter looks 150 us at most");
	std::printf("answer_look_test: OK (%d checks)\n", checks);
	return 0;
}
