// The native test of RgfxPat.hpp: the PAT the x86-64 kext writes on every CPU, entry 2 write-back
// and every other entry as it was, from xnu's PAT and from the one a CPU comes out of reset with;
// which PAT counts as held; and the record of every CPU's PAT the user client reads out, by CPU
// number, whatever number a CPU answers with.
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

	// Held: entry 2 write-back and entry 6 write-combining. xnu's PAT is not, the kext's is; a CPU
	// with entry 2 rewritten that xnu never started (6 still UC-) is not; no answer is not.
	check(!rgfxPatHeld(RGFX_PAT_XNU), "xnu's PAT is not held");
	check(rgfxPatHeld(kept), "the kext's PAT is held");
	check(!rgfxPatHeld(reset), "a PAT xnu never wrote is not held, entry 6 UC-");
	check(!rgfxPatHeld(RGFX_PAT_NONE), "no answer is not held");
	check(RGFX_PAT_NONE != RGFX_PAT_XNU && RGFX_PAT_NONE != kept && RGFX_PAT_NONE != kQemuReset,
	      "no PAT a CPU holds reads as no answer");

	// The record: a CPU writes its own entry; entries no CPU answered stay RGFX_PAT_NONE; a CPU
	// whose number lies past the record, 32 and 64 among them where a 32- or 64-bit mask would
	// overflow, is counted and writes nothing.
	{
		uint64_t pats[6];
		for (uint64_t &p : pats)
			p = 0x5a5a5a5a5a5a5a5aull;
		RgfxCpuPats t = {pats, 4, 7, 7};
		rgfxCpuPatsClear(&t);
		check(t.answered == 0 && t.outside == 0, "the counts emptied");
		bool none = true;
		for (uint32_t i = 0; i < 4; i++)
			none = none && pats[i] == RGFX_PAT_NONE;
		check(none, "every entry RGFX_PAT_NONE");
		check(pats[4] == 0x5a5a5a5a5a5a5a5aull, "nothing past the record emptied");
		rgfxCpuPatsRecord(&t, 0, kept);
		rgfxCpuPatsRecord(&t, 3, RGFX_PAT_XNU);
		rgfxCpuPatsRecord(&t, 4, kept);
		rgfxCpuPatsRecord(&t, 32, kept);
		rgfxCpuPatsRecord(&t, 64, kept);
		rgfxCpuPatsRecord(&t, -1, kept);
		check(t.answered == 6 && t.outside == 4, "six answered, four past the record");
		check(pats[0] == kept && pats[3] == RGFX_PAT_XNU, "each CPU's own entry");
		check(pats[1] == RGFX_PAT_NONE && pats[2] == RGFX_PAT_NONE, "no answer at 1 and 2");
		check(pats[4] == 0x5a5a5a5a5a5a5a5aull && pats[5] == 0x5a5a5a5a5a5a5a5aull,
		      "no CPU past the record writes past it");
	}

	// The x86-64 kext offers the read, as CpuPats in its feature word.
	check(kRgfxKextFeatureCpuPatsRead == kRgfxKextFeatureCpuPats, "x86-64 offers CpuPats");

	std::printf("pat_test: OK (%d checks)\n", checks);
	return 0;
}
