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

	printf("rgfxgate match_test: all assertions passed\n");
	return 0;
}
