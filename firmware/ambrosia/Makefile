# The reims-graphics device's driver module for Ambrosia, the boot firmware of
# arm64 macOS guests on QEMU's vmapple machine (ReimsGraphicsModule.c, and
# AmbrosiaGraphics.h for the interface), built on the host's wire protocol
# header in ../../crates/reims-vgpu-graphics/include, a configuration bundle
# that starts it, and our configuration's bundle, which adds our kexts.
#
#   make         build/ReimsGraphics.elf, the module
#   make bundle  build/ReimsGraphics.cpio, a configuration bundle: config.plist,
#                which names the module and the vtables whose discriminators Ambrosia
#                hands to Graft (Kernel > ExportVtableDiscriminators: IOMobileFramebufferLegacy's
#                and its superclasses', IOMobileFramebufferService's and IOMobileFramebuffer's),
#                and the module as Drivers/ReimsGraphics.elf, in the cpio "newc" format with
#                config.plist first, as Ambrosia reads it
#   make config-bundle GRAFT=<Graft.kext> REIMS_GRAPHICS=<ReimsGraphics.kext>
#                      RGFXGATE=<RgfxGate.kext>
#                build/Config.cpio, our configuration: the bundle's config.plist with a
#                Kernel > Add entry for each kext, in that order, and the kexts under
#                Kexts/ (tools/stage-config.py). Each must be given, and be a kext bundle
#                with the identifier its role expects and an arm64e executable; where one
#                is not, the target names it and why, and writes no bundle
#   make guest-tools
#                build/guest-tools: the lane gate's guest probes that check-boot.sh runs
#                in an arm64 guest, built on a Mac for arm64 and macOS 13 and later from
#                ../../guest/bundle/tools: bench, whose -l lists the Metal devices, and
#                present_probe, an on-screen Metal app
#   make clean   remove build/
#
# check-boot.sh boots a guest with a bundle and checks it (its header says how).
#
# BUILD moves the output. Needs clang and lld (on macOS, Homebrew's: brew
# install lld), as the option ROM does, cpio, and python3 for config-bundle;
# guest-tools needs Xcode.

CC = clang
LD = ld.lld

BUILD = build
WIRE  = ../../crates/reims-vgpu-graphics/include

# Freestanding aarch64 code that runs at any address. The PCIe window, where
# the device's BARs are, is device memory to the module, so no access may be
# unaligned (-mstrict-align), and the compiler keeps to general registers.
CFLAGS = --target=aarch64-unknown-none-elf -Os -ffreestanding -fPIE \
	-fvisibility=hidden -mstrict-align -mgeneral-regs-only \
	-fno-stack-protector -fno-asynchronous-unwind-tables -fno-unwind-tables \
	-ffunction-sections -fdata-sections \
	-Wall -Wextra -Werror -I$(WIRE)

# An ELF static PIE, the module format Ambrosia loads: no PT_INTERP, its
# relocations R_AARCH64_RELATIVE in a DT_RELA table, none in its code, and the
# entry point as e_entry.
LDFLAGS = -pie --no-dynamic-linker -z text --gc-sections --build-id=none \
	--entry=ReimsGraphicsModuleEntry

all: $(BUILD)/ReimsGraphics.elf

bundle: $(BUILD)/ReimsGraphics.cpio

# The kexts' builds live outside this tree, and make sees none of their changes, so the bundle is
# staged afresh each time; a failed stage leaves no bundle behind.
config-bundle: $(BUILD)/ReimsGraphics.elf
	rm -f $(BUILD)/Config.cpio
	python3 -I -B tools/stage-config.py $(BUILD)/config config.plist $(BUILD)/ReimsGraphics.elf \
		"$(GRAFT)" "$(REIMS_GRAPHICS)" "$(RGFXGATE)"
	cd $(BUILD)/config && { echo config.plist; echo Drivers/ReimsGraphics.elf; \
		find Kexts -type f | LC_ALL=C sort; } | cpio -o -H newc > ../Config.cpio

# As lane-gate.sh builds them for an x86-64 guest, but for arm64, and for macOS 13 and later.
GUEST_TOOLS = ../../guest/bundle/tools
TOOLCC = xcrun clang -arch arm64 -mmacosx-version-min=13.0 -fobjc-arc -O2 -Wall -Wextra -Werror \
	-Wno-unused-parameter

guest-tools: $(BUILD)/guest-tools/bench $(BUILD)/guest-tools/present_probe

$(BUILD)/guest-tools/bench: $(GUEST_TOOLS)/bench/bench.m
	@mkdir -p $(@D)
	$(TOOLCC) $< -framework Foundation -framework Metal -framework MetalPerformanceShaders \
		-framework QuartzCore -framework AppKit -o $@

$(BUILD)/guest-tools/present_probe: $(GUEST_TOOLS)/present_probe.m
	@mkdir -p $(@D)
	$(TOOLCC) $< -framework AppKit -framework Metal -framework QuartzCore -framework CoreVideo -o $@

$(BUILD)/ReimsGraphics.elf: $(BUILD)/ReimsGraphicsModule.o
	$(LD) $(LDFLAGS) -o $@ $^

$(BUILD)/ReimsGraphicsModule.o: ReimsGraphicsModule.c
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -MMD -MP -c $< -o $@

$(BUILD)/ReimsGraphics.cpio: config.plist $(BUILD)/ReimsGraphics.elf
	rm -rf $(BUILD)/bundle
	mkdir -p $(BUILD)/bundle/Drivers
	cp config.plist $(BUILD)/bundle/
	cp $(BUILD)/ReimsGraphics.elf $(BUILD)/bundle/Drivers/
	cd $(BUILD)/bundle && printf 'config.plist\nDrivers/ReimsGraphics.elf\n' | cpio -o -H newc > ../ReimsGraphics.cpio

clean:
	rm -rf $(BUILD)

.PHONY: all bundle config-bundle guest-tools clean

-include $(BUILD)/ReimsGraphicsModule.d
