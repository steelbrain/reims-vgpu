# Include after defining NAME, VERSION, SDK, ARCH, BUILD, KEXT, BINARY,
# OBJECTS, HEADERS and CXXFLAGS. The caller owns compilation of OBJECTS and
# src/graft.json; this fragment replaces its ordinary final-link rule.
# All inputs are local, including this tooling and Xcode's kernel headers.

GRAFT_TOOLS := $(dir $(lastword $(MAKEFILE_LIST)))
GRAFT_WORK = $(BUILD)/graft
GRAFT_LAYOUTS = $(GRAFT_WORK)/layouts.json
GRAFT_RESERVE = $(GRAFT_WORK)/manifest-reserve.bin
GRAFT_CENSUS = $(GRAFT_WORK)/census
GRAFT_UNBOUND = $(GRAFT_WORK)/Unbound.kext/Contents/MacOS/$(NAME)
GRAFT_INFO = $(GRAFT_WORK)/Unbound.kext/Contents/Info.plist
GRAFT_PYTHON = python3 -I -B $(GRAFT_TOOLS)consumer.py
GRAFT_SOURCES = $(GRAFT_TOOLS)consumer.py $(GRAFT_TOOLS)graft_macho.py

$(GRAFT_INFO): src/Info.plist Makefile
	@mkdir -p $(@D)
	sed 's/@VERSION@/$(VERSION)/g' $< > $@
	plutil -lint $@ > /dev/null

$(GRAFT_LAYOUTS): src/graft.json $(HEADERS) Makefile $(GRAFT_SOURCES)
	@mkdir -p $(@D)
	$(GRAFT_PYTHON) probe src/graft.json $@ -- $(CXXFLAGS) -Isrc

$(GRAFT_CENSUS): $(OBJECTS) Makefile
	@mkdir -p $(@D)
	xcrun clang++ $(ARCH) -nostdlib -Xlinker -kext $(OBJECTS) -lcc_kext -o $@

$(GRAFT_RESERVE): $(GRAFT_CENSUS) $(GRAFT_LAYOUTS) $(GRAFT_INFO) src/graft.json $(GRAFT_SOURCES)
	$(GRAFT_PYTHON) reserve $(GRAFT_CENSUS) src/graft.json $(GRAFT_LAYOUTS) $(GRAFT_INFO) $@

$(GRAFT_UNBOUND): $(OBJECTS) $(GRAFT_RESERVE) Makefile
	@mkdir -p $(@D)
	xcrun clang++ $(ARCH) -nostdlib -Xlinker -kext $(OBJECTS) -lcc_kext \
		-Xlinker -sectcreate -Xlinker __DATA -Xlinker __graft -Xlinker $(GRAFT_RESERVE) -o $@

$(BINARY): $(GRAFT_UNBOUND) $(GRAFT_INFO) $(GRAFT_LAYOUTS) src/graft.json $(GRAFT_SOURCES)
	python3 -c 'import shutil, sys; shutil.rmtree(sys.argv[1], ignore_errors=True)' "$(KEXT)"
	$(GRAFT_PYTHON) stage $(GRAFT_WORK)/Unbound.kext src/graft.json $(GRAFT_LAYOUTS) $(KEXT)
