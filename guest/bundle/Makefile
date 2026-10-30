# The reims-graphics guest Metal-plugin bundle's serialiser core: the object
# model, the device-info caps accessor, the ring writer and the resource
# create/destroy serialiser (src/Rgfx*.{hpp,cpp}; the protocol is
# docs/graphics-kext/design.md, "Stage 3 wire protocol"). It is our own code and
# depends only on the committed wire header; it is the part of the bundle that
# needs no Apple Metal SPI, so it builds and is checked on the Mac in user space.
#
#   make        build the serialiser objects (x86_64, the guest's arch)
#   make test   build and run the self-contained serialiser check under Rosetta
#   make clean  remove build/
#
# The Metal device shell (the <MTLDevice, MTLDeviceSPI> subclass that calls this
# serialiser) needs Apple's private SPI and a guest boot experiment, and is not
# built here. Everything lands in build/, never committed.

NAME     = serializer_tests
SDK     ?= $(shell xcrun --sdk macosx --show-sdk-path)
ARCH    ?= x86_64
WIRE     = ../../crates/reims-graphics/include
BUILD    = build

CXX      = xcrun clang++
CXXFLAGS = -arch $(ARCH) -isysroot $(SDK) -std=c++17 -Os \
	-Wall -Wextra -Werror -Wno-unused-parameter \
	-I src -I $(WIRE)

HEADERS  = $(wildcard src/*.hpp)
SRCS     = src/RgfxObjectModel.cpp src/RgfxCaps.cpp src/RgfxRingWriter.cpp \
	src/RgfxSerializer.cpp src/RgfxWireLayout.cpp
OBJECTS  = $(SRCS:src/%.cpp=$(BUILD)/obj/%.o)
TESTBIN  = $(BUILD)/$(NAME)

all: $(OBJECTS)

$(BUILD)/obj/%.o: src/%.cpp $(HEADERS) $(WIRE)/reims_graphics_wire.h Makefile
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(TESTBIN): $(OBJECTS) tests/serializer_tests.cpp $(HEADERS) Makefile
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(OBJECTS) tests/serializer_tests.cpp -o $@

test: $(TESTBIN)
	arch -$(ARCH) $(TESTBIN)

clean:
	rm -rf $(BUILD)

.PHONY: all test clean
