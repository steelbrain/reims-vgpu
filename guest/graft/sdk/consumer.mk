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
# and provides src/graft.json, the declaration, and src/Info.plist.

GRAFT_TOOLS := $(dir $(lastword $(MAKEFILE_LIST)))
GRAFT_PYTHON = python3 -I -B $(GRAFT_TOOLS)consumer.py
GRAFT_SOURCES = $(GRAFT_TOOLS)consumer.py $(GRAFT_TOOLS)graft_macho.py
GRAFT_WORK = $(BUILD)/graft
GRAFT_LAYOUTS = $(GRAFT_WORK)/layouts.json
GRAFT_CENSUS = $(GRAFT_WORK)/census
GRAFT_RESERVE = $(GRAFT_WORK)/manifest-reserve.bin
GRAFT_UNBOUND = $(GRAFT_WORK)/Unbound.kext/Contents/MacOS/$(NAME)
GRAFT_INFO = $(GRAFT_WORK)/Unbound.kext/Contents/Info.plist

# Each declared class's size and vtable slots, as the consumer's compiler lays them out.
$(GRAFT_LAYOUTS): src/graft.json $(HEADERS) Makefile $(GRAFT_SOURCES)
	@mkdir -p $(@D)
	$(GRAFT_PYTHON) probe src/graft.json $@ -- $(CXXFLAGS) -Isrc

# The first link, without a manifest, for the census of imports and sites.
$(GRAFT_CENSUS): $(OBJECTS) Makefile
	@mkdir -p $(@D)
	xcrun clang++ $(ARCH) -nostdlib -Xlinker -kext $(OBJECTS) -lcc_kext -o $@

$(GRAFT_INFO): src/Info.plist Makefile
	@mkdir -p $(@D)
	sed 's/@VERSION@/$(VERSION)/g' $< > $@
	plutil -lint $@ > /dev/null

# A zero-filled file the size of the manifest and its import placeholders.
$(GRAFT_RESERVE): $(GRAFT_CENSUS) $(GRAFT_LAYOUTS) $(GRAFT_INFO) src/graft.json $(GRAFT_SOURCES)
	$(GRAFT_PYTHON) reserve $(GRAFT_CENSUS) src/graft.json $(GRAFT_LAYOUTS) $(GRAFT_INFO) $@

# The second link, with the reservation as __DATA,__graft.
$(GRAFT_UNBOUND): $(OBJECTS) $(GRAFT_RESERVE) Makefile
	@mkdir -p $(@D)
	xcrun clang++ $(ARCH) -nostdlib -Xlinker -kext $(OBJECTS) -lcc_kext \
		-Xlinker -sectcreate -Xlinker __DATA -Xlinker __graft -Xlinker $(GRAFT_RESERVE) -o $@

# The staged bundle. Staging refuses an existing output, so the last one goes first.
$(BINARY): $(GRAFT_UNBOUND) $(GRAFT_INFO) $(GRAFT_LAYOUTS) src/graft.json $(GRAFT_SOURCES)
	python3 -c 'import shutil, sys; shutil.rmtree(sys.argv[1], ignore_errors=True)' "$(KEXT)"
	$(GRAFT_PYTHON) stage $(GRAFT_WORK)/Unbound.kext src/graft.json $(GRAFT_LAYOUTS) $(KEXT)
