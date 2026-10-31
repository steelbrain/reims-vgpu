// The display pipe's clients (RgfxPipeClients.hpp), built natively: every associated client is
// served, and the last one's going shows the scanout again where a surface was shown.
#include <cstdio>

#include "RgfxPipeClients.hpp"

static int g_checks = 0, g_failed = 0;
#define CHECK(e)                                                                   \
	do {                                                                           \
		g_checks++;                                                                \
		if (!(e)) {                                                                \
			g_failed++;                                                            \
			std::fprintf(stderr, "%s:%d: CHECK(%s) failed\n", __FILE__, __LINE__, #e); \
		}                                                                          \
	} while (0)

int main()
{
	int a = 0, b = 0, c = 0;
	const int *out[3] = {};
	bool scanout = true;

	// Two clients associate (IOPresentment's pair): both are served, the first not replaced.
	RgfxPipeClients<const int *, 2> pipes;
	CHECK(pipes.add(&a) && pipes.add(&b) && pipes.add(&a));
	CHECK(pipes.count() == 2 && pipes.all(out) == 2 && out[0] == &a && out[1] == &b);
	CHECK(!pipes.add(&c)); // no room
	CHECK(pipes.holds(&a) && pipes.holds(&b) && !pipes.holds(&c));

	// VBLANKs go to the clients registered for them only.
	CHECK(!pipes.anyVblank());
	pipes.setVblank(&b, true);
	CHECK(pipes.anyVblank() && pipes.wantingVblank(out) == 1 && out[0] == &b);
	pipes.setVblank(&c, true); // not held: nothing
	CHECK(pipes.wantingVblank(out) == 1);

	// A surface shown; one client goes: the other still shows it, nothing to ask.
	pipes.shows(0x77);
	CHECK(pipes.remove(&a, &scanout) && !scanout && pipes.shown() == 0x77);
	CHECK(!pipes.remove(&a, &scanout) && !scanout); // gone already
	// The last goes: the scanout is asked for, once.
	CHECK(pipes.remove(&b, &scanout) && scanout && pipes.shown() == 0);
	CHECK(!pipes.anyVblank() && pipes.count() == 0);

	// The last going while the display shows the scanout asks nothing.
	CHECK(pipes.add(&c));
	CHECK(pipes.remove(&c, &scanout) && !scanout);

	// A slot freed takes a new client, with no VBLANK registration of the old one's.
	CHECK(pipes.add(&a));
	pipes.setVblank(&a, true);
	CHECK(pipes.remove(&a, &scanout));
	CHECK(pipes.add(&b) && !pipes.anyVblank());

	if (g_failed) {
		std::fprintf(stderr, "pipe_clients_test: %d of %d checks failed\n", g_failed, g_checks);
		return 1;
	}
	std::printf("pipe_clients_test: OK (%d checks)\n", g_checks);
	return 0;
}
