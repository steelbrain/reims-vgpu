// The native test of RgfxSwapRate.hpp: a second's swaps counted for its one log line, the
// distinct layer-0 surfaces with their counts, and no line due before a second or while idle.
#include "RgfxSwapRate.hpp"

#include <cstdio>
#include <cstdlib>

namespace {
int checks = 0;

void check(bool ok, const char *what)
{
	checks++;
	if (!ok) {
		std::fprintf(stderr, "swap_rate_test: FAILED: %s\n", what);
		std::exit(1);
	}
}
} // namespace

int main()
{
	const uint64_t second = 1000;
	RgfxSwapRate r = {};
	check(!r.due(5000, second), "no swap, no line: an idle display logs nothing");

	// Sixty swaps of WindowServer's surface 4, then an app's surface 9 on twenty of them with a
	// layer-1 surface, one swap with no surface and one dropped.
	for (uint64_t t = 100; t < 160; t++) {
		r.begin(t, 4, false, true);
		r.complete(t);
	}
	for (uint64_t t = 160; t < 180; t++)
		r.begin(t, 9, true, true);
	r.begin(180, 0, false, false);
	r.drop(181);
	check(r.start == 100, "the window opens at its first swap");
	check(r.begun == 81 && r.completed == 60 && r.dropped == 1, "begun, completed and dropped");
	check(r.unshown == 1 && r.layer1 == 20, "a swap with no surface, and layer-1 swaps");
	check(r.ids[0] == 4 && r.counts[0] == 60 && r.ids[1] == 9 && r.counts[1] == 20 &&
	          r.ids[2] == 0 && r.others == 0,
	      "the distinct layer-0 surfaces with their counts");
	check(!r.due(1099, second) && r.due(1100, second),
	      "the line is due a second after the first swap");

	// Past four surfaces, the rest are counted together.
	RgfxSwapRate m = {};
	for (uint32_t id = 1; id <= 6; id++)
		m.begin(10, id, false, true);
	m.begin(11, 6, false, true);
	check(m.ids[3] == 4 && m.counts[3] == 1 && m.others == 3,
	      "surfaces past four counted together");

	// A clear starts the next window at its next swap.
	r.clear();
	check(!r.start && !r.begun && !r.due(1 << 30, second), "a cleared window is idle");
	r.complete(0);
	check(r.start == 1 && r.completed == 1, "a window opened at tick 0 still counts as open");

	std::printf("swap_rate_test: %d checks passed\n", checks);
	return 0;
}
