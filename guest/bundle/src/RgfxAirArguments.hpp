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
// access with one of `air.read`, `air.write`, `air.read_write` and `air.sample`. An argument
// declared under a function constant puts `air.function_constant` and the constant's node ahead
// of its kind; it is read as the function's whatever the constant's value. A binding index that
// is an expression of function constants is a global variable the module's initialiser sets in
// place of the integer; the argument is read as one at any index. A binding of a kind the reader
// does not know makes the function's answer unknown; a vertex function's and a kernel's stage
// input attributes, function tables and acceleration structures are named, as bindings of no
// texture. Threadgroup memory is an `air.buffer` of `air.address_space` 3, at an index of its own
// that no buffer is bound to: it is no argument here. Read from modules that Apple's Metal compiler built from our own sources
// (tests/air_arguments.metal), and as LLVM's text form of the same modules prints them.
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
	// Whether its binding index is an expression of function constants (`[[texture(kIndex)]]`),
	// which the function's specialisation sets: it may be at any index, and `index` is 0.
	bool anyIndex;
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

// Narrows a universal .metallib (a big-endian fat header: magic CA FE BA BE, or CA FE BA BF with
// 64-bit offsets) to its slice of generic AIR bitcode, a .metallib itself; other bytes, and a
// universal one with no such slice, are left as they are.
void rgfxMetallibAirSlice(const uint8_t *&bytes, size_t &size);

// The bitcode of the function `name` in the .metallib at `lib` (one slice, as
// rgfxMetallibAirSlice gives): false where the bytes are no .metallib, or it has no such function
// or no bitcode for it.
bool rgfxMetallibBitcode(const uint8_t *lib, size_t size, const std::string &name,
                         const uint8_t **bitcode, size_t *bitcodeSize);

// What `fn` does with the textures bound at `index` of its stage: true where an argument there
// may write them (write or read_write), false where its arguments there only read or sample them
// or none is there. An argument at any index (anyIndex) is at none: see
// rgfxAirWritesTextureAnywhere.
bool rgfxAirWritesTexture(const RgfxAirFunction &fn, uint32_t index);

// Whether `fn` has a texture argument it may write (write or read_write) whose binding index
// function constants set (anyIndex): a texture bound at any index of its stage may be it.
bool rgfxAirWritesTextureAnywhere(const RgfxAirFunction &fn);

#endif // RGFX_AIR_ARGUMENTS_HPP_
