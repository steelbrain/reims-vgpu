# The option ROMs: each PCI device's UEFI driver (src/), linked as an x86-64
# EFI boot-service driver and packaged as a PCI option ROM
# (docs/architecture.md 11.4). ReimsVgpu is apple-gfx-pci's; ReimsGraphics is
# the reims-graphics device's (docs/graphics-kext/design.md), built on the
# guest's wire protocol header in ../guest/include.
#
#   make        build/ReimsVgpu.{efi,rom} and build/ReimsGraphics.{efi,rom}
#   make clean  remove build/
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
WIRE  = ../guest/include

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

$(BUILD)/$(GNAME).rom: src/ReimsGraphicsRom.S src/OptionRom.S src/ReimsGraphics.h \
		$(WIRE)/reims_graphics_wire.h $(BUILD)/$(GNAME).efi
	$(CC) --target=x86_64-unknown-none-elf -I$(WIRE) -DEFI_IMAGE='"$(BUILD)/$(GNAME).efi"' \
		-c $< -o $(BUILD)/obj/ReimsGraphicsRom.o
	$(ROMLD) --oformat binary -o $@ $(BUILD)/obj/ReimsGraphicsRom.o

$(BUILD)/$(GNAME).efi: $(GOBJECTS)
	$(LD) $(GLDFLAGS) /out:$@ $^

$(BUILD)/obj/%.obj: src/%.c
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -MMD -MP -c $< -o $@

clean:
	rm -rf $(BUILD)

.PHONY: all efi rom clean

-include $(OBJECTS:.obj=.d) $(GOBJECTS:.obj=.d)
