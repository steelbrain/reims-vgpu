# SPDX-License-Identifier: GPL-2.0-or-later OR LGPL-3.0-or-later
# Graft's build rules for a consumer kext, in place of the kext's final link rule. Include
# this file after the consumer Makefile's default target. It links the consumer twice and
# stages it with one graft-kext build; README.md, beside this file, describes the steps.
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
# and provides the Info.plist and the declaration. The declaration path defaults
# to src/graft.json and the Info.plist to src/Info.plist; a Makefile that builds
# more than one variant from one source (a test probe, say), or a per-architecture
# Info.plist, sets GRAFT_DECL or GRAFT_INFO_PLIST to another path before including
# this file.

GRAFT_TOOLS := $(dir $(lastword $(MAKEFILE_LIST)))
# graft-kext, compiled once into the build directory from its sources beside this file.
# Setting GRAFT_KEXT runs another build of the tool instead, to check that build against
# this one.
GRAFT_TOOL = $(BUILD)/graft/graft-kext
GRAFT_KEXT ?= $(GRAFT_TOOL)
GRAFT_TOOL_SOURCES = $(wildcard $(GRAFT_TOOLS)tool/*.cpp $(GRAFT_TOOLS)tool/*.h) \
	$(GRAFT_TOOLS)include/graft/ManifestFormat.h
GRAFT_DECL ?= src/graft.json
GRAFT_INFO_PLIST ?= src/Info.plist
# The makefile that includes this one, whatever its name, and the SDK's headers,
# which the probe may compile through the consumer's own.
GRAFT_MAKEFILE := $(firstword $(MAKEFILE_LIST))
GRAFT_HEADERS = $(wildcard $(GRAFT_TOOLS)include/graft/*.h)

# The staged bundle. graft-kext probes each declared class's layout with the consumer's
# flags, links the objects once for the census of imports and sites, reserves room for the
# manifest, links again with the reservation as __DATA,__graft (an arm64e link also with
# -no_data_const), and stages that link into $(KEXT), replacing it. Each step's output
# stays under $(BUILD)/graft.
$(BINARY): $(OBJECTS) $(GRAFT_DECL) $(GRAFT_INFO_PLIST) $(HEADERS) $(GRAFT_HEADERS) \
		$(GRAFT_MAKEFILE) $(filter $(GRAFT_TOOL),$(GRAFT_KEXT))
	$(GRAFT_KEXT) build --name=$(NAME) --version=$(VERSION) --arch="$(ARCH)" \
		--declaration=$(GRAFT_DECL) --info-plist=$(GRAFT_INFO_PLIST) --work=$(BUILD)/graft \
		--output=$(KEXT) $(OBJECTS) -- $(CXXFLAGS) -Isrc

# The tool is a host program in standard C++17. Its warnings stay warnings, so that a
# newer compiler's new ones do not stop a consumer's build.
$(GRAFT_TOOL): $(GRAFT_TOOL_SOURCES)
	@mkdir -p $(@D)
	xcrun clang++ -std=c++17 -O2 -Wall -Wextra -pedantic -I$(GRAFT_TOOLS)include \
		$(filter %.cpp,$^) -o $@
