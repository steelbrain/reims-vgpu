# The guest side of the reims-graphics device (QEMU's reims-graphics-pci) for
# x86-64 macOS 11 to 26 guests, and the kext set their OpenCore image injects
# for it: Graft (graft/, vendored from Graft's repository with its record),
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
#   make prebuilt-check
#       fail unless prebuilt/ is what sources.toml records, the tree's
#       sources are the ones it was built from, and graft/ is the copy its
#       source.toml records; hashes only, on any host
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
SET ?= prebuilt
WORK = $(abspath build/set-work)
BUNDLE = $(WORK)/bundle/package/reims-graphics.bundle
KEXTINJECT ?=
BOOT_KC ?=

# The parts' Makefiles get only the variables given them below, not this
# command line's (OUT would set theirs).
MAKEOVERRIDES =

.PHONY: set prebuilt prebuilt-check set-check test clean
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

prebuilt-check:
	python3 -I -B tools/prebuilt.py check

set-check:
	bash tools/oc-link-set.sh "$(KEXTINJECT)" "$(BOOT_KC)" $(SET) build/set-check

test:
	$(MAKE) -C kext test
	$(MAKE) -C rgfxgate test
	$(MAKE) -C bundle test

clean:
	rm -rf build
