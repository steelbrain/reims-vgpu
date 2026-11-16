// Native tests of the install pass's decisions (src/rgfx_pass.h): what a pass
// does where it runs, and when the watch that follows a pass in arm64's
// BaseSystem passes again and ends. Build and run with `make test`.
#include "rgfx_pass.h"
#include <cassert>
#include <cstdio>

int main() {
	// --- what a pass does ---------------------------------------------------
	// A full system's Data volume gets the install on either architecture.
	assert(rgfx_pass_kind(false, false) == kRgfxPassInstall);
	assert(rgfx_pass_kind(false, true) == kRgfxPassInstall);
	// x86-64's Recovery and installer: nothing written.
	assert(rgfx_pass_kind(true, false) == kRgfxPassSkip);
	// arm64's BaseSystem: the install into its temporary Data volume, then the
	// watch.
	assert(rgfx_pass_kind(true, true) == kRgfxPassInstallThenWatch);

	// --- when the watch passes again ----------------------------------------
	const uint64_t tmpfs = 0x2a00000001ull, data = 0x1000000002ull, tmpfs2 = 0x2a00000003ull;
	// No Data volume (between one temporary volume's unmount and the next's
	// mount, or before the full system's mounts): no pass.
	assert(!rgfx_watch_passes(false, 0, tmpfs));
	// The volume the last pass ran on: no pass.
	assert(!rgfx_watch_passes(true, tmpfs, tmpfs));
	// The full system's Data volume after FileVault's unlock: a pass.
	assert(rgfx_watch_passes(true, data, tmpfs));
	// A new temporary Data volume (another BaseSystem): a pass.
	assert(rgfx_watch_passes(true, tmpfs2, tmpfs));

	// --- when the watch ends ------------------------------------------------
	// The full system with its Data volume passed: the boot stays there.
	assert(rgfx_watch_ends(false, true, data, data));
	// Still a BaseSystem, whatever its Data volume: the watch goes on (the
	// full system's Data volume can be mounted before the root moves to it).
	assert(!rgfx_watch_ends(true, true, tmpfs, tmpfs));
	assert(!rgfx_watch_ends(true, true, data, data));
	// The full system before its Data volume is there, or before a pass ran on
	// it: the watch goes on.
	assert(!rgfx_watch_ends(false, false, 0, tmpfs));
	assert(!rgfx_watch_ends(false, true, data, tmpfs));

	std::puts("pass_test: ok");
	return 0;
}
