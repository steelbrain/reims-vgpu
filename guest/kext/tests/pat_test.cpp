// The native test of RgfxPat.hpp: the PAT the x86-64 kext writes on every CPU, entry 2 write-back
// and every other entry as it was, from xnu's PAT and from the one a CPU comes out of reset with.
#include "RgfxPat.hpp"

#include <cstdio>
#include <cstdlib>

namespace {
int checks = 0;

void check(bool ok, const char *what)
{
	checks++;
	if (!ok) {
		std::fprintf(stderr, "pat_test: FAILED: %s\n", what);
		std::exit(1);
	}
}

// QEMU's PAT at a CPU's reset (target/i386/cpu.c): xnu's, with entry 6 UC- where xnu writes WC.
const uint64_t kQemuReset = 0x0007040600070406ull;

// Whether `after` holds `before`'s entries, entry 2 apart.
bool othersKept(uint64_t before, uint64_t after)
{
	for (uint32_t e = 0; e < 8; e++)
		if (e != RGFX_PAT_DEVICE_ENTRY && rgfxPatEntry(before, e) != rgfxPatEntry(after, e))
			return false;
	return true;
}
} // namespace

int main()
{
	// xnu's PAT, entry by entry: WB, WT, UC-, UC, then the same with WC at 6.
	check(RGFX_PAT_XNU == 0x0001040600070406ull, "xnu's PAT");
	check(rgfxPatEntry(RGFX_PAT_XNU, 0) == RGFX_PAT_WRITE_BACK, "entry 0 write-back");
	check(rgfxPatEntry(RGFX_PAT_XNU, RGFX_PAT_DEVICE_ENTRY) == RGFX_PAT_UNCACHED_MINUS,
	      "entry 2, a page xnu does not manage, UC- in xnu's PAT");
	check(rgfxPatEntry(RGFX_PAT_XNU, 3) == RGFX_PAT_UNCACHEABLE, "entry 3 UC");
	check(rgfxPatEntry(RGFX_PAT_XNU, RGFX_PAT_WRITE_COMBINED_ENTRY) == RGFX_PAT_WRITE_COMBINING,
	      "entry 6, such a page asked write-combined, WC in xnu's PAT");
	check(RGFX_MSR_PAT == 0x277, "IA32_PAT");

	// The kext's PAT: entry 2 write-back, every other entry xnu's.
	const uint64_t kept = rgfxPatWithDeviceWriteBack(RGFX_PAT_XNU);
	check(kept == 0x0001040600060406ull, "xnu's PAT with entry 2 write-back");
	check(rgfxPatEntry(kept, RGFX_PAT_DEVICE_ENTRY) == RGFX_PAT_WRITE_BACK, "entry 2 write-back");
	check(othersKept(RGFX_PAT_XNU, kept), "every other entry as xnu set it, WC at 6 among them");
	check(rgfxPatWithDeviceWriteBack(kept) == kept,
	      "a CPU that holds it already is written the same PAT: the kext skips it");

	// A CPU out of reset that xnu has not started: only entry 2 changes, so its UC- at 6 stays
	// for xnu's pat_init to see.
	const uint64_t reset = rgfxPatWithDeviceWriteBack(kQemuReset);
	check(reset == 0x0007040600060406ull, "the reset PAT with entry 2 write-back");
	check(othersKept(kQemuReset, reset), "entry 6 left UC- for pat_init");

	// Whatever entry 2 held, it ends write-back, and no other byte moves: each of the 256 values.
	for (uint64_t v = 0; v < 256; v++) {
		const uint64_t pat = (RGFX_PAT_XNU & ~(0xffull << 16)) | (v << 16);
		const uint64_t after = rgfxPatWithDeviceWriteBack(pat);
		check(rgfxPatEntry(after, RGFX_PAT_DEVICE_ENTRY) == RGFX_PAT_WRITE_BACK &&
		          othersKept(pat, after),
		      "entry 2 alone, from any value");
	}

	std::printf("pat_test: OK (%d checks)\n", checks);
	return 0;
}
