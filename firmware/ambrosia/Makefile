# The reims-graphics device's driver module for Ambrosia, the boot firmware of
# arm64 macOS guests on QEMU's vmapple machine (ReimsGraphicsModule.c, and
# AmbrosiaGraphics.h for the interface), built on the host's wire protocol
# header in ../../crates/reims-vgpu-graphics/include, and a configuration bundle
# that starts it.
#
#   make         build/ReimsGraphics.elf, the module
#   make bundle  build/ReimsGraphics.cpio, a configuration bundle: config.plist,
#                which names the module, and the module as Drivers/ReimsGraphics.elf,
#                in the cpio "newc" format with config.plist first, as Ambrosia reads it
#   make clean   remove build/
#
# BUILD moves the output. Needs clang and lld (on macOS, Homebrew's: brew
# install lld), as the option ROM does, and cpio.

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

.PHONY: all bundle clean

-include $(BUILD)/ReimsGraphicsModule.d
