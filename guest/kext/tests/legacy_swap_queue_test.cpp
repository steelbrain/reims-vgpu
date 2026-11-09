// The native test of RgfxLegacySwapLayout.hpp: each release's layout is told by its class's size,
// and the swap queue's first swap is checked, taken off and read as the paravirtual display does,
// on a queue laid out at each release's offsets.
#include "RgfxLegacySwapLayout.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {
int checks = 0;

void check(bool ok, const char *what, const char *release)
{
	checks++;
	if (!ok) {
		std::fprintf(stderr, "legacy_swap_queue_test: FAILED (%s): %s\n", release, what);
		std::exit(1);
	}
}

// A framebuffer and its swaps in memory, at a layout's offsets, the queue built as TAILQ_INSERT_TAIL
// builds it.
struct Fake {
	explicit Fake(const RgfxLegacySwapLayout &layout) : l(layout), fb(layout.classSize + 64) {}

	uint8_t *swap(size_t k)
	{
		while (swaps.size() <= k)
			swaps.emplace_back(0x1000, 0);
		return swaps[k].data();
	}
	void **at(uint8_t *p, uint32_t offset) { return reinterpret_cast<void **>(p + offset); }
	void **head() { return at(fb.data(), l.current + 8); }
	void **last() { return at(fb.data(), l.current + 16); }
	void **next(uint8_t *s) { return at(s, l.link); }
	void **back(uint8_t *s) { return at(s, l.link + 8); }

	void emptyQueue()
	{
		*head() = nullptr;
		*last() = head();
	}
	void append(uint8_t *s)
	{
		*next(s) = nullptr;
		*back(s) = *last();
		*static_cast<void **>(*last()) = s;
		*last() = next(s);
	}
	void setWord(uint8_t *s, uint32_t offset, uint32_t value) { std::memcpy(s + offset, &value, 4); }

	const RgfxLegacySwapLayout &l;
	std::vector<uint8_t> fb;
	std::vector<std::vector<uint8_t>> swaps;
};

void testLayout(const RgfxLegacySwapLayout &layout)
{
	const char *r = layout.release;
	check(rgfxLegacySwapLayoutOf(layout.classSize) == &layout, "the class's size tells the release",
	      r);
	check(layout.current + 24 <= layout.classSize, "the queue lies in the class", r);
	check(layout.link % 8 == 0 && layout.surfaces % 8 == 0 && layout.current % 8 == 0,
	      "pointers are aligned", r);
	check(layout.state % 4 == 0 && layout.layers % 4 == 0, "words are aligned", r);

	Fake f(layout);
	RgfxLegacySwapQueue q(f.fb.data(), layout);
	f.emptyQueue();
	check(q.current() == nullptr && q.first() == nullptr, "an empty queue has no swap", r);
	check(!q.firstLinked(), "an empty queue has no first swap to take", r);

	uint8_t *a = f.swap(0), *b = f.swap(1), *c = f.swap(2);
	f.append(a);
	f.append(b);
	f.append(c);
	check(q.first() == a && q.firstLinked(), "the first of three is linked", r);

	q.setCurrent(a);
	q.removeFirst();
	check(q.current() == a, "the current swap is set", r);
	check(q.first() == b && *f.back(b) == static_cast<void *>(f.head()),
	      "the second becomes the first, its back link the head's", r);
	check(*f.last() == static_cast<void *>(f.next(c)), "the last stays the last", r);
	check(q.firstLinked(), "the new first is linked", r);

	q.removeFirst();
	q.removeFirst();
	check(q.first() == nullptr && *f.last() == static_cast<void *>(f.head()),
	      "taking the last empties the queue, as TAILQ_REMOVE does", r);

	// A swap appended to the emptied queue is linked again.
	f.append(b);
	check(q.first() == b && q.firstLinked(), "a queue emptied takes swaps again", r);

	// Links that do not hold, as a wrong layout or a damaged queue would leave them.
	f.emptyQueue();
	f.append(a);
	f.append(b);
	*f.back(b) = f.next(c);
	check(!q.firstLinked(), "a next swap whose back link is not the first's is refused", r);
	f.emptyQueue();
	f.append(a);
	*f.back(a) = f.next(c);
	check(!q.firstLinked(), "a first swap whose back link is not the head's is refused", r);

	// The state the paravirtual display begins a swap at: 0 or 8, nothing else.
	f.setWord(a, layout.state, 0);
	check(q.begins(a), "state 0 begins", r);
	f.setWord(a, layout.state, 8);
	check(q.begins(a), "state 8 begins", r);
	for (uint32_t s : {1u, 2u, 4u, 9u, 0x10u, 0x80000000u}) {
		f.setWord(a, layout.state, s);
		check(!q.begins(a), "any other state is dropped", r);
	}

	// A layer's surface where its bit is set.
	void *s0 = &checks, *s1 = &f;
	*f.at(a, layout.surfaces) = s0;
	*f.at(a, layout.surfaces + 8) = s1;
	f.setWord(a, layout.layers, 0);
	check(q.surface(a, 0) == nullptr && q.surface(a, 1) == nullptr, "no bit, no surface", r);
	f.setWord(a, layout.layers, 1);
	check(q.surface(a, 0) == s0 && q.surface(a, 1) == nullptr, "bit 0 gives layer 0's", r);
	f.setWord(a, layout.layers, 0x80000002u);
	check(q.surface(a, 0) == nullptr && q.surface(a, 1) == s1, "bit 1 gives layer 1's", r);
}
} // namespace

int main()
{
	for (const RgfxLegacySwapLayout &layout : kRgfxLegacySwapLayouts)
		testLayout(layout);
	check(rgfxLegacySwapLayoutOf(0x1dd0) == nullptr && rgfxLegacySwapLayoutOf(0) == nullptr,
	      "a size no release has tells none", "-");
	std::printf("legacy_swap_queue_test: %d checks passed over %zu releases\n", checks,
	            sizeof(kRgfxLegacySwapLayouts) / sizeof(kRgfxLegacySwapLayouts[0]));
	return 0;
}
