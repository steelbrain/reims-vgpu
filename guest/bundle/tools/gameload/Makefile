# gameload, a game-shaped Metal load (gameload.m's header says what each shape is, and every
# option and output line).
#
#   make -C guest/bundle/tools/gameload        one universal binary (arm64 and x86_64: an
#                                              Apple-silicon Mac, an Intel Mac, and an arm64 or
#                                              x86-64 guest), guest/bundle/build/gameload
#   make -C guest/bundle/tools/gameload clean
#
# It runs on a real device, never in the bundle's `make test`. Its shaders compile from source at
# run time (gameload_source.h, from gameload.metal, for newLibraryWithSource:). It has a Makefile
# of its own because guest/bundle/Makefile is one of the sources the prebuilt guest sets record
# (guest/prebuilt*/sources.toml): a target there would mean rebuilding both sets.

SDK   ?= $(shell xcrun --sdk macosx --show-sdk-path)
BUILD  = ../../build

all: $(BUILD)/gameload

$(BUILD)/gameload_source.h: gameload.metal Makefile
	@mkdir -p $(@D)
	xxd -i -n gameload_metal gameload.metal > $@

$(BUILD)/gameload: gameload.m $(BUILD)/gameload_source.h Makefile
	@mkdir -p $(@D)
	xcrun clang -arch arm64 -arch x86_64 -isysroot $(SDK) -mmacosx-version-min=13.0 -fobjc-arc -O2 \
		-Wall -Wextra -Werror -Wno-unused-parameter -I $(BUILD) gameload.m \
		-framework Foundation -framework Metal -o $@

clean:
	rm -f $(BUILD)/gameload $(BUILD)/gameload_source.h

.PHONY: all clean
