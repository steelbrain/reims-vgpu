// The native test of RgfxMemorySize.hpp: the GPU memory the kext states, from the host's working
// set and unified-memory answers, as an Intel Mac's two GPUs state theirs (measured, macOS 26.7.1).
#include "RgfxMemorySize.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {
int checks = 0;

void check(bool ok, const char *what)
{
	checks++;
	if (!ok) {
		std::fprintf(stderr, "memory_size_test: FAILED: %s\n", what);
		std::exit(1);
	}
}

// The answers as the kext reads them: out of a whole device info's bytes, the host's layout.
RgfxMemorySize fromDeviceInfo(uint64_t workingSet, bool unified)
{
	static RgfxDeviceInfo info;
	std::memset(&info, 0xa5, sizeof(info));
	info.sizes[kRgfxDeviceInfoWorkingSetSize] = workingSet;
	info.identity_flags[kRgfxDeviceInfoHasUnifiedMemory] = unified ? 1 : 0;
	return rgfxMemorySizeFrom(reinterpret_cast<const uint8_t *>(&info) + kRgfxMemorySizeFirst);
}
} // namespace

int main()
{
	// The AMD Radeon Pro 5300M: its own memory. VRAM,totalMB 4080 on its IOPCIDevice, and
	// dedicatedMemorySize the working set.
	RgfxMemorySize amd = rgfxMemorySize(4278190080ull, false);
	check(amd.vramMB == 4080, "the AMD's VRAM,totalMB is its working set in MiB");
	check(amd.dedicatedBytes == 4278190080ull, "the AMD's dedicatedMemorySize is its working set");

	// The Intel UHD Graphics 630: unified memory. VRAM,totalMB 1536 on its accelerator, and
	// dedicatedMemorySize 0.
	RgfxMemorySize intel = rgfxMemorySize(1610612736ull, true);
	check(intel.vramMB == 1536, "the Intel's VRAM,totalMB is its working set in MiB");
	check(intel.dedicatedBytes == 0, "a unified GPU dedicates no memory");

	// A host that gave no working set: nothing stated.
	RgfxMemorySize none = rgfxMemorySize(0, false);
	check(none.vramMB == 0 && none.dedicatedBytes == 0, "no working set states none");
	// Less than a MiB rounds down to none, as the shift does.
	check(rgfxMemorySize((1ull << 20) - 1, false).vramMB == 0, "under a MiB is 0 MiB");
	// A working set past what a u32 of MiB holds saturates.
	check(rgfxMemorySize(~0ull, false).vramMB == UINT32_MAX, "the MiB saturate at a u32");

	// Read from the device info's bytes as the host lays them out, every other byte not ours.
	RgfxMemorySize read = fromDeviceInfo(17094934528ull, false);
	check(read.vramMB == 16303, "the working set reads from sizes[0]");
	check(read.dedicatedBytes == 17094934528ull, "a discrete host's dedicated memory reads alike");
	read = fromDeviceInfo(1610612736ull, true);
	check(read.vramMB == 1536 && read.dedicatedBytes == 0,
	      "the unified flag reads from identity_flags[3]");
	// The span the kext reads holds both answers and lies inside the device info.
	check(kRgfxMemorySizeFirst == offsetof(RgfxDeviceInfo, sizes), "the span starts at sizes[0]");
	check(kRgfxMemorySizeEnd <= sizeof(RgfxDeviceInfo), "the span ends inside the device info");
	check(kRgfxMemorySizeEnd > offsetof(RgfxDeviceInfo, identity_flags) +
	                               kRgfxDeviceInfoHasUnifiedMemory,
	      "the span holds identity_flags[3]");

	std::printf("memory_size_test: OK (%d checks)\n", checks);
	return 0;
}
