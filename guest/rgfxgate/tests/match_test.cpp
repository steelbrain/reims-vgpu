// Native tests of RgfxGate's matching logic (src/rgfx_match.h): the cdhash
// membership test that gates library validation and file-map-executable, and
// the path-prefix test that gates the boot-time install write. Build and run
// with `make test`.
#include "rgfx_match.h"
#include <cassert>
#include <cstdio>

int main() {
	// --- cdhash membership --------------------------------------------------
	const uint8_t ours[20] = {0x9f,0x64,0x2c,0x63,0x51,0xfc,0xc5,0x85,0x55,0x4e,
	                          0x7f,0x50,0x56,0xa4,0xc5,0xfc,0x90,0x28,0x2f,0xcf};
	const uint8_t other[20] = {0x00,0x64,0x2c,0x63,0x51,0xfc,0xc5,0x85,0x55,0x4e,
	                           0x7f,0x50,0x56,0xa4,0xc5,0xfc,0x90,0x28,0x2f,0xcf};
	const uint8_t allowed[][20] = {
		{0x9f,0x64,0x2c,0x63,0x51,0xfc,0xc5,0x85,0x55,0x4e,
		 0x7f,0x50,0x56,0xa4,0xc5,0xfc,0x90,0x28,0x2f,0xcf},
	};
	// Our cdhash matches; a kernel cdhash is at least 20 bytes.
	assert(rgfx_cdhash_allowed(ours, 20, allowed, 1));
	// A longer candidate still matches on its first 20 bytes.
	uint8_t longer[32]; for (int i = 0; i < 20; i++) longer[i] = ours[i];
	assert(rgfx_cdhash_allowed(longer, 32, allowed, 1));
	// A different hash is rejected (the planted-bundle narrowness check).
	assert(!rgfx_cdhash_allowed(other, 20, allowed, 1));
	// A short or missing candidate is rejected.
	assert(!rgfx_cdhash_allowed(ours, 19, allowed, 1));
	assert(!rgfx_cdhash_allowed(nullptr, 20, allowed, 1));
	// An empty allow-list matches nothing.
	assert(!rgfx_cdhash_allowed(ours, 20, allowed, 0));

	// --- path prefix --------------------------------------------------------
	const char *root = "/Library/GPUBundles";
	assert(rgfx_path_under("/Library/GPUBundles", root));
	assert(rgfx_path_under("/Library/GPUBundles/reims-graphics.bundle", root));
	assert(rgfx_path_under("/Library/GPUBundles/reims-graphics.bundle/Contents/MacOS/reims-graphics", root));
	// A sibling that merely shares the prefix string is not under it.
	assert(!rgfx_path_under("/Library/GPUBundlesX", root));
	assert(!rgfx_path_under("/Library/GPUBundlesEvil/x", root));
	// Unrelated and empty paths are rejected.
	assert(!rgfx_path_under("/System/Library/Extensions/x.bundle", root));
	assert(!rgfx_path_under("/Library", root));
	assert(!rgfx_path_under(nullptr, root));
	assert(!rgfx_path_under("/Library/GPUBundles", ""));

	// --- staging sibling path -----------------------------------------------
	const char *bundle = "/Library/GPUBundles/reims-graphics.bundle";
	char buf[64];
	// The sibling is the bundle path with the suffix appended, under the same
	// container, so the install's writes to it are admitted and the rename
	// stays on one volume.
	assert(rgfx_sibling_path(bundle, ".new", buf, sizeof(buf)));
	assert(strcmp(buf, "/Library/GPUBundles/reims-graphics.bundle.new") == 0);
	assert(rgfx_path_under(buf, root));
	// An exact fit (path + suffix + terminator == cap) is written.
	char tight[46]; // strlen(bundle)=41, +".new"=45, +'\0'=46
	assert(rgfx_sibling_path(bundle, ".new", tight, sizeof(tight)));
	assert(strcmp(tight, "/Library/GPUBundles/reims-graphics.bundle.new") == 0);
	// One byte short leaves the buffer untouched and reports failure.
	char small[45];
	small[0] = 'Z';
	assert(!rgfx_sibling_path(bundle, ".new", small, sizeof(small)));
	assert(small[0] == 'Z');
	// Null arguments and a zero cap are refused.
	assert(!rgfx_sibling_path(nullptr, ".new", buf, sizeof(buf)));
	assert(!rgfx_sibling_path(bundle, nullptr, buf, sizeof(buf)));
	assert(!rgfx_sibling_path(bundle, ".new", nullptr, sizeof(buf)));
	assert(!rgfx_sibling_path(bundle, ".new", buf, 0));

	printf("rgfxgate match_test: all assertions passed\n");
	return 0;
}
