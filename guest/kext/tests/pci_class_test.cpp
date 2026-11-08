// The native test of RgfxPciClass.hpp: the `class-code` the kext states on the PCI device reads
// 0x030000, a VGA-compatible display controller's, as both GPUs of an Intel Mac state it
// (measured, macOS 26.7.1); a display controller as the device's own class is, of another
// subclass.
#include "RgfxPciClass.hpp"

#include <cstdio>
#include <cstdlib>

namespace {
int checks = 0;

void check(bool ok, const char *what)
{
	checks++;
	if (!ok) {
		std::fprintf(stderr, "pci_class_test: FAILED: %s\n", what);
		std::exit(1);
	}
}
} // namespace

int main()
{
	const uint32_t stated = rgfxClassCodeValue(kRgfxMacGpuClassCode);
	check(stated == 0x030000, "class-code reads 0x030000");
	check(stated >> 16 == RGFX_PCI_CLASS >> 8, "the same base class as the device's, a display controller");
	check((stated >> 8 & 0xff) != (RGFX_PCI_CLASS & 0xff), "another subclass than the device's");
	const uint8_t device[4] = {0x00, RGFX_PCI_CLASS & 0xff, RGFX_PCI_CLASS >> 8, 0x00};
	check(rgfxClassCodeValue(device) == 0x038000, "the device's own class reads 0x038000");
	std::printf("pci_class_test: %d checks passed\n", checks);
	return 0;
}
