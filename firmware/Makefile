# The option ROM: the reims-graphics device's UEFI driver (src/), linked as an
# x86-64 EFI boot-service driver and packaged as a PCI option ROM
# (docs/graphics-kext/design.md), built on the host's wire protocol header in
# ../crates/reims-vgpu-graphics/include.
#
#   make           build/ReimsGraphics.{efi,rom}
#   make prebuilt  on a Mac, also prebuilt/ReimsGraphics.rom, which reims-ctl carries, and
#                  prebuilt/ReimsGraphics.toml, the sha256 of the files it was built from
#                  (prebuilt.sh): commit both with a change to what the ROM is built from
#   make prebuilt-check
#                  fail where the committed ROM is stale: a file its record names changed
#                  (cargo xtask check runs it)
#   make clean     remove build/
#
# BUILD moves the output. Needs clang and lld (on macOS, Homebrew's: brew
# install lld). The ROM's bytes are the compiler's: Xcode's clang and Linux's
# build different ones from the same sources.

CC    = clang
LD    = lld-link
ROMLD = ld.lld

BUILD = build

TARGET = --target=x86_64-unknown-windows

CFLAGS = $(TARGET) -Os -ffreestanding -fshort-wchar -funsigned-char \
	-fno-ms-extensions -fno-stack-protector -mno-stack-arg-probe \
	-mno-red-zone -mno-implicit-float -fno-asynchronous-unwind-tables \
	-ffunction-sections -fdata-sections \
	-ftrap-function=undefined_behavior_has_been_optimized_away_by_clang \
	-Wall -Wextra -Werror -Wno-unused-parameter

LDFLAGS = /nologo /nodefaultlib /dll /driver /base:0 /machine:x64 \
	/subsystem:efi_boot_service_driver /entry:ReimsGraphicsEntryPoint \
	/opt:ref /align:32 /filealign:32 /Brepro

GNAME = ReimsGraphics
WIRE  = ../crates/reims-vgpu-graphics/include

# The ROM's PCI identity is the host wire header's. It is generated C, not
# assembler, so its macro values are read with the preprocessor and passed to
# ReimsGraphicsRom.S. ROM_CLASS_CODE is the 24-bit class code: the header's
# 16-bit class and subclass over a zero programming-interface byte.
RGFX_DUMP   = $(CC) --target=x86_64-unknown-none-elf -E -dM -x c $(WIRE)/reims_graphics_wire.h
ROM_VENDOR  = $(shell $(RGFX_DUMP) 2>/dev/null | sed -n 's/^\#define RGFX_PCI_VENDOR_ID //p')
ROM_DEVICE  = $(shell $(RGFX_DUMP) 2>/dev/null | sed -n 's/^\#define RGFX_PCI_DEVICE_ID //p')
ROM_CLASS16 = $(shell $(RGFX_DUMP) 2>/dev/null | sed -n 's/^\#define RGFX_PCI_CLASS //p')
GROMDEFS    = -DROM_VENDOR_ID=$(ROM_VENDOR) -DROM_DEVICE_ID=$(ROM_DEVICE) \
	'-DROM_CLASS_CODE=(($(ROM_CLASS16)) << 8)'

GSOURCES = \
	src/ReimsGraphics.c \
	src/ReimsGraphicsOutput.c \
	src/ReimsGraphicsName.c

GOBJECTS = $(patsubst src/%,$(BUILD)/obj/%.obj,$(basename $(GSOURCES)))

$(GOBJECTS): CFLAGS += -I$(WIRE)

all: rom

efi: $(BUILD)/$(GNAME).efi

rom: $(BUILD)/$(GNAME).rom

# OptionRom.S lays out the ROM header around the linked image; ld.lld
# flattens it to raw bytes.
$(BUILD)/$(GNAME).rom: src/ReimsGraphicsRom.S src/OptionRom.S \
		$(WIRE)/reims_graphics_wire.h $(BUILD)/$(GNAME).efi
	$(CC) --target=x86_64-unknown-none-elf $(GROMDEFS) -DEFI_IMAGE='"$(BUILD)/$(GNAME).efi"' \
		-c $< -o $(BUILD)/obj/ReimsGraphicsRom.o
	$(ROMLD) --oformat binary -o $@ $(BUILD)/obj/ReimsGraphicsRom.o

$(BUILD)/$(GNAME).efi: $(GOBJECTS)
	$(LD) $(LDFLAGS) /out:$@ $^

$(BUILD)/obj/%.obj: src/%.c
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -MMD -MP -c $< -o $@

prebuilt: $(BUILD)/$(GNAME).rom
	sh prebuilt.sh $(BUILD)

prebuilt-check:
	@sh prebuilt.sh --check

clean:
	rm -rf $(BUILD)

.PHONY: all efi rom prebuilt prebuilt-check clean

-include $(GOBJECTS:.obj=.d)
