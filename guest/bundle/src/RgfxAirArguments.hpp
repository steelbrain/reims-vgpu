// RgfxAirArguments: the arguments of a Metal library's entry points, as the AIR module of each
// declares them in its own metadata: each binding argument's kind, binding index, count and
// access. The guest's device learns from it which bindings a function may write, which the host
// does not tell it (it gives no binding reflection).
//
// A .metallib holds one LLVM bitcode module per function (its function list's OFFT tag places it
// in the bitcode section, its MDSZ tag gives its size). The module lists its entry points in the
// named metadata `air.kernel`, `air.vertex` and `air.fragment`: one node per entry point, of the
// function, a node of its outputs and a node of its arguments, one node per argument. A binding
// argument's node starts with its position and its kind (`air.buffer`, `air.indirect_buffer` for
// an argument buffer, `air.texture`, `air.sampler`), then names its binding with
// `air.location_index` and two integers, the index and the count (an array's length), and its
// access with one of `air.read`, `air.write`, `air.read_write` and `air.sample`. Read from modules
// that Apple's Metal compiler built from our own sources (tests/air_arguments.metal), and as
// LLVM's text form of the same modules prints them.
//
// The reader is LLVM's bitstream format (its documented container: abbreviations, blocks and
// records) over the module's blocks that carry the above: the module's global values and its
// string table (the functions' names), its constants (the integers), and its metadata. It skips
// every other block whole. Anything it cannot follow makes the function's answer unknown rather
// than wrong.
//
// Our own code.
#ifndef RGFX_AIR_ARGUMENTS_HPP_
#define RGFX_AIR_ARGUMENTS_HPP_

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

enum class RgfxAirStage : uint8_t { Kernel, Vertex, Fragment };

enum class RgfxAirKind : uint8_t { Buffer, IndirectBuffer, Texture, Sampler };

enum class RgfxAirAccess : uint8_t { Unknown, Read, Write, ReadWrite, Sample };

// One binding argument of an entry point.
struct RgfxAirArgument {
	RgfxAirKind kind;
	// Its first binding index, and how many it takes from there (an array's length).
	uint32_t index;
	uint32_t count;
	RgfxAirAccess access;
	// For an argument buffer: whether a texture among its members is written (write or
	// read_write).
	bool writesTextures;
};

// One entry point: its name, stage and binding arguments, in the module's order.
struct RgfxAirFunction {
	std::string name;
	RgfxAirStage stage;
	std::vector<RgfxAirArgument> arguments;
};

// The entry points of the AIR module at `bitcode` (raw bitcode, or in LLVM's wrapper header).
// False where it is not one the reader can follow; `out` then holds nothing.
bool rgfxReadAirFunctions(const uint8_t *bitcode, size_t size, std::vector<RgfxAirFunction> &out);

// The bitcode of the function `name` in the .metallib at `lib`: false where the bytes are no
// .metallib, or it has no such function or no bitcode for it.
bool rgfxMetallibBitcode(const uint8_t *lib, size_t size, const std::string &name,
                         const uint8_t **bitcode, size_t *bitcodeSize);

// What `fn` does with the textures bound at `index` of its stage: true where an argument there
// may write them (write or read_write), false where its arguments there only read or sample them
// or none is there.
bool rgfxAirWritesTexture(const RgfxAirFunction &fn, uint32_t index);

#endif // RGFX_AIR_ARGUMENTS_HPP_
