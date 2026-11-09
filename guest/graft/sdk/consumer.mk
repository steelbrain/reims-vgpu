# SPDX-License-Identifier: GPL-2.0-or-later OR LGPL-3.0-or-later
# Graft's build rules for a consumer kext, in place of the kext's final link rule. Include
# this file after the consumer Makefile's default target. It links the consumer twice and
# stages it with consumer.py; README.md, beside this file, describes the steps.
#
# The consumer's Makefile defines:
#   NAME      the kext's executable name
#   VERSION   the version that replaces @VERSION@ in src/Info.plist
#   ARCH      the target flags for both links, such as -arch x86_64
#   BUILD     a build directory; these rules write only under $(BUILD)/graft
#   KEXT      the staged bundle these rules write
#   BINARY    $(KEXT)/Contents/MacOS/$(NAME), the target that writes $(KEXT)
#   OBJECTS   the kext's objects, which the consumer's own rules compile
#   HEADERS   the consumer's headers, on which the layout probe depends
#   CXXFLAGS  the flags OBJECTS are compiled with, which the layout probe uses too
# and provides src/Info.plist and the declaration. The declaration path defaults
# to src/graft.json; a Makefile that builds more than one variant from one source
# (a test probe, say) sets GRAFT_DECL to another path before including this file.

GRAFT_TOOLS := $(dir $(lastword $(MAKEFILE_LIST)))
GRAFT_PYTHON = python3 -I -B $(GRAFT_TOOLS)consumer.py
GRAFT_SOURCES = $(GRAFT_TOOLS)consumer.py $(GRAFT_TOOLS)graft_macho.py
GRAFT_DECL ?= src/graft.json
GRAFT_WORK = $(BUILD)/graft
# An arm64e consumer keeps its import GOT and vtable in writable __DATA, which
# Graft writes after lockdown (notes/arm64-boot.md); -no_data_const folds
# __DATA_CONST in. A fold-free x86-64 build links without it.
ifneq (,$(findstring arm64e,$(ARCH)))
GRAFT_LDEXTRA = -Xlinker -no_data_const
endif
GRAFT_LAYOUTS = $(GRAFT_WORK)/layouts.json
GRAFT_CENSUS = $(GRAFT_WORK)/census
GRAFT_RESERVE = $(GRAFT_WORK)/manifest-reserve.bin
GRAFT_UNBOUND = $(GRAFT_WORK)/Unbound.kext/Contents/MacOS/$(NAME)
GRAFT_INFO = $(GRAFT_WORK)/Unbound.kext/Contents/Info.plist

# Each declared class's size and vtable slots, as the consumer's compiler lays them out.
$(GRAFT_LAYOUTS): $(GRAFT_DECL) $(HEADERS) Makefile $(GRAFT_SOURCES)
	@mkdir -p $(@D)
	$(GRAFT_PYTHON) probe $(GRAFT_DECL) $@ -- $(CXXFLAGS) -Isrc

# The first link, without a manifest, for the census of imports and sites.
$(GRAFT_CENSUS): $(OBJECTS) Makefile
	@mkdir -p $(@D)
	xcrun clang++ $(ARCH) -nostdlib -Xlinker -kext $(GRAFT_LDEXTRA) $(OBJECTS) -lcc_kext -o $@

$(GRAFT_INFO): src/Info.plist Makefile
	@mkdir -p $(@D)
	sed 's/@VERSION@/$(VERSION)/g' $< > $@
	plutil -lint $@ > /dev/null

# A zero-filled file the size of the manifest and its import placeholders.
$(GRAFT_RESERVE): $(GRAFT_CENSUS) $(GRAFT_LAYOUTS) $(GRAFT_INFO) $(GRAFT_DECL) $(GRAFT_SOURCES)
	$(GRAFT_PYTHON) reserve $(GRAFT_CENSUS) $(GRAFT_DECL) $(GRAFT_LAYOUTS) $(GRAFT_INFO) $@

# The second link, with the reservation as __DATA,__graft.
$(GRAFT_UNBOUND): $(OBJECTS) $(GRAFT_RESERVE) Makefile
	@mkdir -p $(@D)
	xcrun clang++ $(ARCH) -nostdlib -Xlinker -kext $(GRAFT_LDEXTRA) $(OBJECTS) -lcc_kext \
		-Xlinker -sectcreate -Xlinker __DATA -Xlinker __graft -Xlinker $(GRAFT_RESERVE) -o $@

# The staged bundle. Staging refuses an existing output, so the last one goes first.
$(BINARY): $(GRAFT_UNBOUND) $(GRAFT_INFO) $(GRAFT_LAYOUTS) $(GRAFT_DECL) $(GRAFT_SOURCES)
	python3 -c 'import shutil, sys; shutil.rmtree(sys.argv[1], ignore_errors=True)' "$(KEXT)"
	$(GRAFT_PYTHON) stage $(GRAFT_WORK)/Unbound.kext $(GRAFT_DECL) $(GRAFT_LAYOUTS) $(KEXT)
