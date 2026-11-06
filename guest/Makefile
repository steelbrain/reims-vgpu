# The guest side of the reims-graphics device (QEMU's reims-graphics-pci) for
# x86-64 macOS 26 guests, and the kext set their OpenCore image injects for it,
# after Lilu: ReimsGraphicsLoader (loader/), its two shims, ReimsGraphics
# (kext/), staged for the loader, and RgfxGate (rgfxgate/), which carries the
# Metal bundle (bundle/). OpenCore links injected kexts against the Boot
# kernel collection only, and ReimsGraphics uses two System-collection
# libraries, so it loads through the loader (loader/README.md). Builds on a Mac
# with Xcode.
#
#   make set KDK=<a macOS 26 KDK's Extensions directory> [OUT=build/set]
#       build each part in build/set-work/ and write OUT: the set's bundles
#       and set.toml, their Kernel > Add entries in order
#       (tools/prebuilt.py). DEPS (default build/deps) holds the pinned Lilu
#       headers and MacKernelSDK the Lilu plugins build against, fetched
#       where it has none (rgfxgate/tools/fetch-lilu.sh). The KDK is only
#       read: the two libraries' exports and class layouts, and the kernel's
#       exports.
#   make prebuilt KDK=...
#       the set built afresh into prebuilt/, which the repository carries
#       for hosts that cannot build it, and prebuilt/sources.toml, the
#       record of the sources it was built from; commit prebuilt/ with the
#       change to those sources
#   make prebuilt-check
#       fail unless prebuilt/ is what sources.toml records and the tree's
#       sources are the ones it was built from; hashes only, on any host
#   make set-check KEXTINJECT=<OpenCorePkg's TestKextInject binary>
#               BOOT_KC=<the guest's x86_64 BootKernelExtensions.kc>
#               LILU=<a RELEASE Lilu.kext> [SET=prebuilt]
#       link Lilu and SET into BOOT_KC with OpenCore's linker, in user space,
#       in Kernel > Add order, and check every kext injects, every name
#       resolves and every site of ReimsGraphics holds its shim's
#       placeholder (tools/oc-link-set.sh)
#   make test   the parts' own tests: kext/, loader/, rgfxgate/ and bundle/
#   make clean  remove build/
#
# Everything but prebuilt/ lands in build/ or the parts' own build/, never
# committed.

KDK    ?=
DEPS   ?= build/deps
OUT    ?= build/set
SET    ?= prebuilt
WORK    = $(abspath build/set-work)
BUNDLE  = $(WORK)/bundle/package/reims-graphics.bundle

KEXTINJECT ?=
BOOT_KC    ?=
LILU       ?=

# The parts' Makefiles get only the variables given them below, not this
# command line's (OUT and KDK would set theirs).
MAKEOVERRIDES =

.PHONY: set prebuilt prebuilt-check set-check test clean
set:
	@test -n "$(KDK)" || { echo "set KDK=<a macOS 26 KDK's Extensions directory>" >&2; exit 2; }
	@# Fetched first: rgfxgate's objects do not wait for its own fetch under -j.
	@test -d "$(DEPS)/Lilu.kext" && test -d "$(DEPS)/MacKernelSDK" || bash rgfxgate/tools/fetch-lilu.sh "$(abspath $(DEPS))"
	$(MAKE) -C kext BUILD=$(WORK)/kext
	$(MAKE) -C bundle BUILD=$(WORK)/bundle package
	$(MAKE) -C rgfxgate BUILD=$(WORK)/rgfxgate DEPS=$(abspath $(DEPS)) BUNDLE_DIR=$(BUNDLE)
	$(MAKE) -C loader BUILD=$(WORK)/loader DEPS=$(abspath $(DEPS)) KDK=$(abspath $(KDK)) \
		KEXT=$(WORK)/kext/ReimsGraphics.kext
	python3 -I -B tools/prebuilt.py stage $(OUT) $(WORK)

prebuilt:
	@test -n "$(KDK)" || { echo "set KDK=<a macOS 26 KDK's Extensions directory>" >&2; exit 2; }
	rm -rf $(WORK)
	$(MAKE) set OUT=prebuilt KDK=$(abspath $(KDK)) DEPS=$(abspath $(DEPS))
	python3 -I -B tools/prebuilt.py record prebuilt $(abspath $(KDK))

prebuilt-check:
	python3 -I -B tools/prebuilt.py check

set-check:
	bash tools/oc-link-set.sh "$(KEXTINJECT)" "$(BOOT_KC)" "$(LILU)" $(SET) \
		$(WORK)/loader/staged/ReimsGraphics.bridge/fixups.json build/set-check

test:
	$(MAKE) -C kext test
	$(MAKE) -C loader test
	$(MAKE) -C rgfxgate test
	$(MAKE) -C bundle test

clean:
	rm -rf build
