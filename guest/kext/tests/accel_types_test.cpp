// The native test of RgfxAccelTypes.hpp: the framebuffer's IOAccelTypes is set to the
// accelerator's path as it stands where it has none, where it names the accelerator through the
// PCI device's earlier name (the device renamed after the path was set), and where it names
// anything else; it is left where it is that path already.
#include "RgfxAccelTypes.hpp"

#include <cstdio>
#include <cstdlib>

namespace {
int checks = 0;

void check(bool ok, const char *what)
{
	checks++;
	if (!ok) {
		std::fprintf(stderr, "accel_types_test: FAILED: %s\n", what);
		std::exit(1);
	}
}
} // namespace

int main()
{
	// The service-plane paths of one accelerator before and after its PCI device is renamed.
	const char *before = "IOService:/AppleACPIPlatformExpert/PCI0@0/AppleACPIPCI/S20@4/RgfxAccelerator";
	const char *after = "IOService:/AppleACPIPlatformExpert/PCI0@0/AppleACPIPCI/GFX0@4/RgfxAccelerator";

	check(rgfxAccelTypesStale(nullptr, after), "a framebuffer with no IOAccelTypes is named");
	check(rgfxAccelTypesStale(before, after),
	      "a path through the PCI device's earlier name is named again");
	check(!rgfxAccelTypesStale(after, after), "the path as it stands is left as it is");
	check(rgfxAccelTypesStale("IOService:/AppleACPIPlatformExpert/PCI0@0/AppleACPIPCI/GFX0@4", after),
	      "the PCI device's own path is not the accelerator's");
	check(rgfxAccelTypesStale("", after), "an empty path is named again");

	std::printf("accel_types_test: %d checks passed\n", checks);
	return 0;
}
