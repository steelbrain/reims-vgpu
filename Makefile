# The Rust library and its tooling. The Cargo workspace is at the
# repository root.
#
#   make        the library as a plain library (cargo xtask dist) in its dist
#               tree, target/dist-vulkan: libreims_vgpu.a and the shared
#               library (.so on Linux, .dylib on a Mac, over MoltenVK), with
#               reims_vgpu.h and reims-vgpu.pc
#   make qemu   the library as a plain library (cargo xtask dist), then
#               vendor/qemu's qemu-system-x86_64 with reims-graphics-pci on
#               its C API, linked statically through pkg-config, in
#               build/qemu (docs/qemu.md)
#   make clean  remove build/, QEMU's build included
#
# QEMU_CONFIGURE adds options to QEMU's configure and runs it again.
# CARGO_TARGET_DIR moves the build, and with it RUST_TARGET, out of target.

BUILD := build
# $(BUILD) made absolute without abspath, which splits a path with spaces.
BUILD_DIR := $(if $(filter /%,$(BUILD)),$(BUILD),$(CURDIR)/$(BUILD))
QEMU_CONFIGURE ?=
QEMU_BUILD := $(BUILD_DIR)/qemu
QEMU_SRC := $(CURDIR)/vendor/qemu
# Cargo's target directory, made absolute the same way: the dist tree is
# under it, which QEMU's config.status records, and cargo resolves a
# relative one against the directory it runs in.
RUST_TARGET_REL := $(or $(CARGO_TARGET_DIR),target)
RUST_TARGET := $(if $(filter /%,$(RUST_TARGET_REL)),$(RUST_TARGET_REL),$(CURDIR)/$(RUST_TARGET_REL))
# The dist tree cargo xtask dist writes (xtask/src/dist.rs).
RUST_DIST := $(RUST_TARGET)/dist-vulkan

.PHONY: all qemu clean

all:
	CARGO_TARGET_DIR="$(RUST_TARGET)" cargo xtask dist

vendor/qemu/configure:
	@echo "vendor/qemu is empty: git submodule update --init vendor/qemu" >&2
	@exit 1

# QEMU finds reims-vgpu.pc in the dist tree and links libreims_vgpu.a, so the
# binary records no path to the library; reims-graphics-pci calls its C API
# (reims_vgpu.h). Configure runs once, out of tree, and again whenever
# QEMU_CONFIGURE is set or config.status was made from another source or with
# another PKG_CONFIG_PATH, or names an option vendor/qemu's configure no longer
# takes (--enable-pvg, --apple-gfx-backend, --pvg-framework) or the
# aarch64-softmmu target an earlier make qemu built. It also runs again when
# reims-vgpu.pc differs from the one it last read: meson keeps the package's
# flags from configure, so a framework the library came to need (its
# Libs.private) would otherwise be missing from QEMU's link.
QEMU_PKG_CONFIG := $(RUST_DIST)/lib/pkgconfig
QEMU_PC_READ := $(QEMU_BUILD)/reims-vgpu.pc.configured

qemu: all vendor/qemu/configure
	@mkdir -p "$(QEMU_BUILD)"
	@if [ ! -f "$(QEMU_BUILD)/build.ninja" ] || [ -n "$(QEMU_CONFIGURE)" ] \
			|| ! grep -qF -- "exec '$(QEMU_SRC)/configure'" "$(QEMU_BUILD)/config.status" 2>/dev/null \
			|| ! grep -qxF -- "PKG_CONFIG_PATH='$(QEMU_PKG_CONFIG)'" "$(QEMU_BUILD)/config.status" \
			|| grep -qE -- "'--(enable-pvg|apple-gfx-backend=|pvg-framework=|target-list=aarch64-softmmu)" "$(QEMU_BUILD)/config.status" \
			|| ! cmp -s "$(QEMU_PKG_CONFIG)/reims-vgpu.pc" "$(QEMU_PC_READ)"; then \
		cd "$(QEMU_BUILD)" && PKG_CONFIG_PATH="$(QEMU_PKG_CONFIG)" "$(QEMU_SRC)/configure" \
			--target-list=x86_64-softmmu --enable-reims-vgpu \
			--disable-docs $(QEMU_CONFIGURE) \
			&& cp "$(QEMU_PKG_CONFIG)/reims-vgpu.pc" "$(QEMU_PC_READ)"; \
	fi
	$(MAKE) -C "$(QEMU_BUILD)" qemu-system-x86_64

clean:
	rm -rf "$(BUILD_DIR)"
