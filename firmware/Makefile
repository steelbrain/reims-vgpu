# The option ROMs: each PCI device's UEFI driver (src/), linked as an x86-64
# EFI boot-service driver and packaged as a PCI option ROM
# (docs/architecture.md 11.4). ReimsVgpu is apple-gfx-pci's; ReimsGraphics is
# the reims-graphics device's (docs/graphics-kext/design.md), built on the
# host's wire protocol header in ../crates/reims-graphics/include.
#
#   make           build/ReimsVgpu.{efi,rom} and build/ReimsGraphics.{efi,rom}
#   make prebuilt  on a Mac, also prebuilt/ReimsGraphics.rom, which reims-ctl carries, and
#                  prebuilt/ReimsGraphics.toml, the sha256 of the files it was built from
#                  (prebuilt.sh): commit both with a change to what the ROM is built from
#   make prebuilt-check
#                  fail where the committed ROM is stale: a file its record names changed
#                  (cargo xtask check runs it)
#   make clean     remove build/
#
# BUILD moves the output: cargo xtask dist runs `make rom` with BUILD set to
# <target>/firmware. Needs clang and lld (on macOS, Homebrew's: brew install
# lld). The ROM's bytes are the compiler's: Xcode's clang and Linux's build
# different ones from the same sources.

CC    = clang
LD    = lld-link
ROMLD = ld.lld

BUILD = build
NAME  = ReimsVgpu

SOURCES = \
	src/ReimsVgpu.c \
	src/GraphicsOutput.c \
	src/ComponentName.c

TARGET = --target=x86_64-unknown-windows

CFLAGS = $(TARGET) -Os -ffreestanding -fshort-wchar -funsigned-char \
	-fno-ms-extensions -fno-stack-protector -mno-stack-arg-probe \
	-mno-red-zone -mno-implicit-float -fno-asynchronous-unwind-tables \
	-ffunction-sections -fdata-sections \
	-ftrap-function=undefined_behavior_has_been_optimized_away_by_clang \
	-Wall -Wextra -Werror -Wno-unused-parameter

LDFLAGS = /nologo /nodefaultlib /dll /driver /base:0 /machine:x64 \
	/subsystem:efi_boot_service_driver /entry:ReimsVgpuEntryPoint \
	/opt:ref /align:32 /filealign:32 /Brepro

OBJECTS = $(patsubst src/%,$(BUILD)/obj/%.obj,$(basename $(SOURCES)))

GNAME = ReimsGraphics
WIRE  = ../crates/reims-graphics/include

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
GLDFLAGS = $(filter-out /entry:%,$(LDFLAGS)) /entry:ReimsGraphicsEntryPoint

$(GOBJECTS): CFLAGS += -I$(WIRE)

all: rom

efi: $(BUILD)/$(NAME).efi $(BUILD)/$(GNAME).efi

rom: $(BUILD)/$(NAME).rom $(BUILD)/$(GNAME).rom

# OptionRom.S lays out the ROM header around the linked image; ld.lld
# flattens it to raw bytes.
$(BUILD)/$(NAME).rom: src/OptionRom.S src/ReimsVgpu.h $(BUILD)/$(NAME).efi
	$(CC) --target=x86_64-unknown-none-elf -DEFI_IMAGE='"$(BUILD)/$(NAME).efi"' \
		-c $< -o $(BUILD)/obj/OptionRom.o
	$(ROMLD) --oformat binary -o $@ $(BUILD)/obj/OptionRom.o

$(BUILD)/$(NAME).efi: $(OBJECTS)
	$(LD) $(LDFLAGS) /out:$@ $^

$(BUILD)/$(GNAME).rom: src/ReimsGraphicsRom.S src/OptionRom.S \
		$(WIRE)/reims_graphics_wire.h $(BUILD)/$(GNAME).efi
	$(CC) --target=x86_64-unknown-none-elf $(GROMDEFS) -DEFI_IMAGE='"$(BUILD)/$(GNAME).efi"' \
		-c $< -o $(BUILD)/obj/ReimsGraphicsRom.o
	$(ROMLD) --oformat binary -o $@ $(BUILD)/obj/ReimsGraphicsRom.o

$(BUILD)/$(GNAME).efi: $(GOBJECTS)
	$(LD) $(GLDFLAGS) /out:$@ $^

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

-include $(OBJECTS:.obj=.d) $(GOBJECTS:.obj=.d)
