// The native test of RgfxCompletionArm.hpp: the kext's take of ring 0's completion ring touches
// the header only where something changed (a take that finds the ring as its last arm left it
// reads `producer` once, and stores nothing), and no interleaving of the host's post with a take
// leaves a record untaken with no interrupt to come, or the ring unarmed once all are taken.
//
// The host's post, as crates/reims-vgpu-graphics's ring.rs makes it: it publishes `producer`,
// fences, then reads `armed`; where it reads 1 it clears it, then reads `consumer`, and where the
// guest has taken every record published it sets `armed` back to 1 and raises nothing
// (rearm_or_raise), else it raises the vector. Each of its two steps may come before any of the
// take's accesses.
#include "RgfxCompletionArm.hpp"

#include <cstdio>
#include <cstdlib>
#include <functional>
#include <vector>

namespace {
int checks = 0;

void check(bool ok, const char *what)
{
	checks++;
	if (!ok) {
		std::fprintf(stderr, "completion_arm_test: FAILED: %s\n", what);
		std::exit(1);
	}
}

// The ring's header as both sides see it, the guest's own counts, and the host's raised vector.
struct Model {
	uint64_t producer = 0;
	uint64_t headerConsumer = 0;
	uint32_t armed = 0;
	bool raised = false;
	uint64_t consumed = 0;
	uint64_t armedAt = kRgfxNotArmed;
	// The header's accesses the guest made, by kind.
	int producerReads = 0, consumerStores = 0, arms = 0;

	// The host's post, its two steps: the publish, then the look at `armed` after the fence.
	void publish() { producer += 32; }
	void lookAtArmed()
	{
		if (armed != 1)
			return;
		armed = 0;
		if (headerConsumer == producer)
			armed = 1;
		else
			raised = true;
	}
};

// The guest's view of the header, calling `before` ahead of each access, where a test puts the
// host's steps.
struct Ring {
	Model &m;
	std::function<void()> before;
	uint64_t producer()
	{
		before();
		m.producerReads++;
		return m.producer;
	}
	void take(uint64_t end)
	{
		before();
		if (m.consumed != end) {
			m.consumed = end;
			m.headerConsumer = end;
			m.consumerStores++;
		}
	}
	void arm()
	{
		before();
		m.armed = 1;
		m.arms++;
	}
};

void take(Model &m, const std::function<void()> &before)
{
	Ring r{m, before};
	rgfxTakeAndArm(r, m.consumed, m.armedAt);
}

void none() {}

// The guest takes the ring again at each vector raised, until none is: then every record published
// is taken and the ring is armed, so the host's next post raises the vector.
void settle(Model &m, const char *what)
{
	for (int i = 0; m.raised && i < 8; i++) {
		m.raised = false;
		take(m, none);
	}
	check(!m.raised, what);
	check(m.consumed == m.producer, what);
	check(m.armed == 1, what);
}
} // namespace

int main()
{
	// A fresh ring: armed once, the producer read before and after the arm, nothing stored but
	// the arm.
	{
		Model m;
		take(m, none);
		check(m.arms == 1 && m.producerReads == 2 && m.consumerStores == 0,
		      "a fresh empty ring is armed once");
		check(m.armed == 1 && m.armedAt == 0, "the arm found the ring empty");
		// Nothing published since: one read of the producer, no store.
		take(m, none);
		check(m.arms == 1 && m.producerReads == 3 && m.consumerStores == 0,
		      "an armed empty ring is read once and not armed again");
		take(m, none);
		check(m.arms == 1 && m.producerReads == 4, "and again");
	}
	// A record posted to an armed ring: the vector, then one take that stores the consumer once
	// and arms again.
	{
		Model m;
		take(m, none);
		m.publish();
		m.lookAtArmed();
		check(m.raised && m.armed == 0, "the post to an armed ring raises the vector");
		const int arms = m.arms, stores = m.consumerStores;
		m.raised = false;
		take(m, none);
		check(m.consumed == m.producer && m.consumerStores == stores + 1 && m.arms == arms + 1,
		      "the take stores the consumer once and arms again");
		check(m.armed == 1 && m.armedAt == m.consumed, "armed at the new count");
		take(m, none);
		check(m.arms == arms + 1 && m.consumerStores == stores + 1, "then nothing more is stored");
	}
	// Every interleaving of the host's two steps with one take's accesses, from a ring armed
	// empty (the skip) and from one never armed, with 0 to 2 records taken before: no record
	// is left untaken with no vector to come, and the ring ends armed.
	for (int start = 0; start < 2; start++) {
		for (int before = 0; before < 3; before++) {
			// The take's accesses with no host step among them, to bound the steps tried.
			int accesses = 0;
			{
				Model m;
				for (int i = 0; i < before; i++) {
					m.publish();
					m.lookAtArmed();
					m.raised = false;
					take(m, none);
				}
				if (start == 0)
					take(m, none);
				int n = 0;
				take(m, [&] { n++; });
				accesses = n;
			}
			for (int publishAt = 0; publishAt <= accesses + 1; publishAt++) {
				for (int lookAt = publishAt; lookAt <= accesses + 1; lookAt++) {
					Model m;
					for (int i = 0; i < before; i++) {
						m.publish();
						m.lookAtArmed();
						m.raised = false;
						take(m, none);
					}
					if (start == 0)
						take(m, none);
					int step = 0;
					bool published = false, looked = false;
					auto host = [&] {
						if (!published && step == publishAt) {
							m.publish();
							published = true;
						}
						if (published && !looked && step == lookAt) {
							m.lookAtArmed();
							looked = true;
						}
					};
					take(m, [&] {
						host();
						step++;
					});
					// The host's steps left after the take.
					step = accesses + 1;
					host();
					if (!published) {
						m.publish();
						published = true;
					}
					if (!looked)
						m.lookAtArmed();
					settle(m, "an interleaving leaves a record untaken or the ring unarmed");
				}
			}
		}
	}
	std::printf("completion_arm_test: %d checks passed\n", checks);
	return 0;
}
