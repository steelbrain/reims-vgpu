// The native test of RgfxScanoutOwner.hpp: only the task that holds the framebuffer's server
// connection (or whose open of one is in flight) owns the scanout, a new connection's task
// replaces the old one and takes every other task's hold of the scanout back, a closed connection
// owns nothing, the VRAM range goes to the owner on its connection alone (never to IOFramebuffer's
// own setup, for a new mode either), and the scanout's room never reaches the accelerator's blocks.
#include "RgfxScanoutOwner.hpp"

#include <cstdio>
#include <cstdlib>

namespace {
int checks = 0;

void check(bool ok, const char *what)
{
	checks++;
	if (!ok) {
		std::fprintf(stderr, "scanout_owner_test: FAILED: %s\n", what);
		std::exit(1);
	}
}

// Tasks and connections are opaque pointers to the rule: any distinct addresses do.
int windowServer, app, nextWindowServer, kernel;
int connection, nextConnection;
} // namespace

int main()
{
	const void *ws = &windowServer, *other = &app, *ws2 = &nextWindowServer, *k = &kernel;
	const void *oldTask = &app, *oldClient = &app;
	RgfxScanoutOwner o;

	// Before any server connection, no task owns the scanout.
	check(!o.owns(ws, false) && !o.owns(other, false), "no owner before the server connection");
	check(!o.owns(nullptr, true), "no task is no owner");

	// The server connection's open: its task owns the scanout while it is in flight, as
	// IOFramebuffer sets the display up inside it, and no other task does.
	bool marked = o.beginOpen(ws, false);
	check(marked, "the first open is marked");
	check(o.owns(ws, false), "the opening task owns the scanout during its open");
	check(!o.owns(other, false), "another task does not, during the open");
	o.endOpen(ws, &connection, marked, &oldTask, &oldClient);
	check(!oldTask && !oldClient, "the first connection replaces nothing");
	check(o.owns(ws, true), "the connection's task owns the scanout while it is open");
	check(!o.owns(other, true), "another task does not");
	check(o.opening == nullptr, "the open is over");

	// A second open while the connection is held is refused by IOFramebuffer: it marks nothing,
	// makes nothing and changes no owner.
	marked = o.beginOpen(other, true);
	check(!marked, "an open while the connection is held is not marked");
	check(!o.owns(other, true), "nor does its task own the scanout during it");
	o.endOpen(other, nullptr, marked, &oldTask, &oldClient);
	check(!oldTask && !oldClient && o.owns(ws, true) && !o.owns(other, true),
	      "a refused open changes nothing");

	// Its connection closed (the client terminated), the task owns nothing.
	check(!o.owns(ws, false), "a closed connection owns nothing");

	// A failed open after the close: its task owns the scanout only while it is in flight.
	marked = o.beginOpen(other, false);
	check(marked && o.owns(other, false), "a task owns the scanout during its open");
	o.endOpen(other, nullptr, marked, &oldTask, &oldClient);
	check(!o.owns(other, false) && !oldTask && !oldClient, "and not after it failed");

	// The next WindowServer's connection replaces the last: the caller lets the old task and
	// connection go.
	marked = o.beginOpen(ws2, false);
	o.endOpen(ws2, &nextConnection, marked, &oldTask, &oldClient);
	check(oldTask == ws && oldClient == &connection, "the replaced task and connection go back");
	check(o.owns(ws2, true) && !o.owns(ws, true), "the new connection's task is the only owner");

	// An open whose mark another open took (two at once, the second waiting on IOFramebuffer's
	// gate) still records the connection it made, and clears no one else's mark.
	RgfxScanoutOwner racing;
	bool first = racing.beginOpen(ws, false);
	bool second = racing.beginOpen(other, false);
	check(first && !second, "one open in flight is marked at a time");
	racing.endOpen(other, &connection, second, &oldTask, &oldClient);
	check(racing.opening == ws, "the unmarked open leaves the other's mark");
	check(racing.owns(other, true), "and its connection is recorded");
	racing.endOpen(ws, nullptr, first, &oldTask, &oldClient);
	check(racing.opening == nullptr && racing.owns(other, true) && !racing.owns(ws, true),
	      "the marked open, refused, clears its mark");

	// The record's end hands back what it held.
	o.clear(&oldTask, &oldClient);
	check(oldTask == ws2 && oldClient == &nextConnection, "the record's end hands back its own");
	check(!o.owns(ws2, true), "and owns nothing after");

	// The owner moves to the next WindowServer: the former owner's grants and mappings of the
	// scanout go, and the new owner's stay, as does the same task's where it opens again.
	check(!rgfxKeepsScanout(ws, ws2), "the former owner's hold goes when the owner moves");
	check(!rgfxKeepsScanout(other, ws2), "another task's hold goes");
	check(rgfxKeepsScanout(ws2, ws2), "the new owner's hold stays");
	check(!rgfxKeepsScanout(nullptr, ws2), "a hold of no task goes");

	// The VRAM range: the owner on the connection it holds, and no one else: not the kernel, not
	// a task whose open is in flight.
	check(!rgfxMayMapVram(k, k, false, false) && !rgfxMayMapVram(k, k, true, false),
	      "the kernel does not map the VRAM range");
	check(rgfxMayMapVram(ws, k, true, false), "the owner on its connection does");
	check(!rgfxMayMapVram(other, k, false, false), "no other task does");
	check(!rgfxMayMapVram(nullptr, k, true, false), "a null task does not");
	// Nor IOFramebuffer's setup for a new mode, which runs on the thread of the owner that set
	// it: the range it gets there is the memory size it publishes, which it then never does.
	RgfxScanoutOwner setting;
	const void *wsThread = &windowServer, *otherThread = &app;
	check(!setting.inSetup(wsThread) && !setting.inSetup(nullptr), "no setup before one begins");
	const void *outer = setting.beginSetup(wsThread);
	check(outer == nullptr, "the first setup runs inside none");
	check(setting.inSetup(wsThread) && !setting.inSetup(otherThread),
	      "the setup's thread is marked, and no other");
	check(!rgfxMayMapVram(ws, k, true, setting.inSetup(wsThread)),
	      "the owner's setup for a new mode does not map the VRAM range");
	check(rgfxMayMapVram(ws, k, true, setting.inSetup(otherThread)),
	      "the owner's other threads on its connection do, meanwhile");
	const void *inner = setting.beginSetup(otherThread);
	check(inner == wsThread && setting.inSetup(otherThread), "a setup inside it is marked");
	setting.endSetup(inner);
	check(setting.inSetup(wsThread), "and its end puts the outer one back");
	setting.endSetup(outer);
	check(!setting.inSetup(wsThread) && setting.settingUp == nullptr, "no setup after it ends");
	check(rgfxMayMapVram(ws, k, true, setting.inSetup(wsThread)),
	      "and the owner maps the range on its connection again");
	RgfxScanoutOwner held;
	marked = held.beginOpen(ws, false);
	check(held.owns(ws, false) && !held.holdsConnection(ws, false),
	      "an open in flight owns the scanout but holds no connection");
	held.endOpen(ws, &connection, marked, &oldTask, &oldClient);
	check(held.holdsConnection(ws, true), "the made connection is held");
	check(!held.holdsConnection(ws, false), "a closed one is not");
	check(!held.holdsConnection(other, true), "nor is another task's");

	// The scanout's room: the span, inside the region, never past the accelerator's blocks.
	const uint64_t cursor = 256 * 256 * 4;
	check(rgfxScanoutRoom(0x40000000, cursor) == RGFX_SCANOUT_SPAN, "the 1 GiB region's room");
	check(RGFX_ALLOCATIONS_OFFSET + rgfxScanoutRoom(0x40000000, cursor) ==
	          RGFX_CONTEXT_ALLOCATIONS_OFFSET,
	      "the room ends where the accelerator's blocks start");
	check(rgfxScanoutRoom(0x1000000, cursor) == 0x1000000 - RGFX_ALLOCATIONS_OFFSET - cursor,
	      "a region smaller than the span: what it holds before the cursor");
	check(rgfxScanoutRoom(RGFX_ALLOCATIONS_OFFSET + cursor, cursor) == 0,
	      "a region with no room past its head");
	// An 8K frame fits the room; a 16K-wide one, which the old room let through, does not.
	check(30720ull * 4320 <= rgfxScanoutRoom(0x40000000, cursor), "an 8K frame fits");
	check(65536ull * 4320 > rgfxScanoutRoom(0x40000000, cursor), "a 16K-wide one does not");

	std::printf("scanout_owner_test: OK (%d checks)\n", checks);
	return 0;
}
