# The guest side of the reims-graphics device (QEMU's reims-graphics-pci) for
# x86-64 macOS 11 to 26 guests, and the kext set their OpenCore image injects
# for it, and the arm64e set for the arm64 macOS 14 and later guests Ambrosia
# boots (prebuilt-arm64e/, which Ambrosia's config bundle places):
# Graft (graft/, vendored from Graft's repository with its record),
# ReimsGraphics (kext/), injected dormant for Graft to bind against the
# System kernel collection's libraries, and RgfxGate (rgfxgate/), a Graft
# client that carries the Metal bundle (bundle/). Builds on a Mac with Xcode
# alone: no KDK and nothing fetched.
#
#   make set [OUT=build/set]
#       build each part in build/set-work/ and write OUT: the set's bundles
#       and set.toml, their Kernel > Add entries in order (tools/prebuilt.py)
#   make prebuilt
#       the set built afresh into prebuilt/, which the repository carries
#       for hosts that cannot build it, and prebuilt/sources.toml, the
#       record of the sources it was built from; commit prebuilt/ with the
#       change to those sources
#   make set-arm64e [OUT64=build/set-arm64e]
#       the arm64e set: Graft's arm64e build, and ReimsGraphics and RgfxGate
#       for arm64e, with the Metal bundle for arm64e and arm64 inside RgfxGate
#   make prebuilt-arm64e
#       that set built afresh into prebuilt-arm64e/ with its own sources.toml
#   make prebuilt-check
#       fail unless prebuilt/ and prebuilt-arm64e/ are each what their
#       sources.toml records, the tree's sources are the ones they were built
#       from, graft/ is the copy its source.toml records, and no kext leaves
#       one of our own symbols undefined that no kext before it defines (which
#       staging a set checks too); hashes and symbol tables only, on any host
#   make set-check KEXTINJECT=<OpenCorePkg's TestKextInject binary>
#               BOOT_KC=<the guest's x86_64 BootKernelExtensions.kc> [SET=prebuilt]
#       link SET into BOOT_KC with OpenCore's linker, in user space, in
#       Kernel > Add order; check every kext injects and every name
#       resolves, and that ReimsGraphics stays bindable by Graft
#       (tools/oc-link-set.sh)
#   make test   the parts' own tests: kext/, rgfxgate/ and bundle/
#   make clean  remove build/
#
# Everything but prebuilt/ lands in build/ or the parts' own build/, never
# committed. To take a new Graft, replace graft/ with Graft's
# tools/vendor.py and rebuild prebuilt/.

OUT ?= build/set
OUT64 ?= build/set-arm64e
SET ?= prebuilt
WORK = $(abspath build/set-work)
WORK64 = $(abspath build/set-work-arm64e)
BUNDLE = $(WORK)/bundle/package/reims-graphics.bundle
BUNDLE64 = $(WORK64)/bundle/package/reims-graphics.bundle
KEXTINJECT ?=
BOOT_KC ?=

# The parts' Makefiles get only the variables given them below, not this
# command line's (OUT would set theirs).
MAKEOVERRIDES =

.PHONY: set prebuilt set-arm64e prebuilt-arm64e prebuilt-check set-check test clean
set:
	$(MAKE) -C kext BUILD=$(WORK)/kext
	$(MAKE) -C bundle BUILD=$(WORK)/bundle package
	$(MAKE) -C rgfxgate BUILD=$(WORK)/rgfxgate BUNDLE_DIR=$(BUNDLE)
	ditto graft/built/Graft.kext $(WORK)/Graft.kext
	python3 -I -B tools/prebuilt.py stage $(OUT) $(WORK)

prebuilt:
	rm -rf $(WORK)
	$(MAKE) set OUT=prebuilt
	python3 -I -B tools/prebuilt.py record prebuilt

set-arm64e:
	$(MAKE) -C kext ARCH_NAME=arm64e BUILD=$(WORK64)/kext
	$(MAKE) -C bundle ARCH=arm64e PACKAGE_ARCHS="arm64e arm64" BUILD=$(WORK64)/bundle package
	$(MAKE) -C rgfxgate ARCH_NAME=arm64e BUILD=$(WORK64)/rgfxgate BUNDLE_DIR=$(BUNDLE64)
	ditto graft/built/Graft-arm64e.kext $(WORK64)/Graft-arm64e.kext
	python3 -I -B tools/prebuilt.py stage-arm64e $(OUT64) $(WORK64)

prebuilt-arm64e:
	rm -rf $(WORK64)
	$(MAKE) set-arm64e OUT64=prebuilt-arm64e
	python3 -I -B tools/prebuilt.py record prebuilt-arm64e

prebuilt-check:
	python3 -I -B tools/prebuilt.py check
	python3 -I -B tools/prebuilt.py check prebuilt-arm64e

set-check:
	bash tools/oc-link-set.sh "$(KEXTINJECT)" "$(BOOT_KC)" $(SET) build/set-check

test:
	$(MAKE) -C kext test
	$(MAKE) -C rgfxgate test
	$(MAKE) -C bundle test

clean:
	rm -rf build
