// RgfxAirArguments: the arguments of a Metal library's entry points, as the AIR module of each
// declares them in its own metadata: each binding argument's kind, binding index, count and
// access. The guest's device learns from it which bindings a function may write, which the host
// does not tell it (it gives no binding reflection). It also reads a stitchable function's
// signature, from which the device builds the reflection Metal answers for the function.
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
// place of the integer; the argument is read as one at any index. So is a count (the length of an
// `array_ref` of textures or buffers, which Metal Performance Shaders' kernels bind); the argument
// is read as one of every index from its first. A binding of a kind the reader does not know
// makes the function's answer unknown; a vertex function's and a kernel's stage input attributes,
// a post-tessellation vertex function's patch inputs, function tables and acceleration structures
// are named, as bindings of no texture. Threadgroup memory is an `air.buffer` of
// `air.address_space` 3, at an index of its own that no buffer is bound to: it is no argument
// here. Read from modules that Apple's Metal compiler built from our own sources
// (tests/air_arguments.metal), and as LLVM's text form of the same modules prints them; a count
// set by function constants, which no source of ours declares, from macOS 27's frameworks'
// modules on the M3 Max, where 709 entry points bind one (Metal Performance Shaders' kernels
// among them).
//
// A module built for macOS 10.12 (-std=macos-metal1.2) holds LLVM 3.5's metadata nodes, each
// operand a type and a value: a metadata where the type is metadata, else a value in place of a
// METADATA_VALUE (tests/air_arguments_old.metal).
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
	// Whether its count is an expression of function constants, which the function's
	// specialisation sets: it may take every index from `index` on, and `count` is 0.
	bool anyCount;
};

// One function constant a module declares. The module lists them in the named metadata
// `air.function_constants`: one node per constant, of the global its value is read from, the
// name of its type (`bool`, `float4`, `ushort2`, ...: a scalar's, or a vector's of two to four),
// its name and its index (`[[function_constant(index)]]`). It is required where that global is
// `unnamed_addr`; the global of one the function tests with `is_function_constant_defined` is not.
// A .metallib's module of one function lists that function's constants, those its `CNST` tag
// lists (RgfxFunctionConstants.hpp), and none where it has no CNST tag (measured: the 46,563
// functions of macOS 27's frameworks' .metallibs on the M3 Max, 6,615 of them with constants:
// each constant of the same name, index and type in both, required in CNST for each of the 25,275
// whose global is `unnamed_addr` and for none of the 7,108 others).
struct RgfxAirConstant {
	std::string name;
	uint32_t type = 0; // MTLDataType
	uint32_t index = 0;
	bool required = false;
};

// One entry point: its name, stage and binding arguments, in the module's order, and the function
// constants its module declares (`constantsRead` false where they did not read).
struct RgfxAirFunction {
	std::string name;
	RgfxAirStage stage;
	std::vector<RgfxAirArgument> arguments;
	std::vector<RgfxAirConstant> constants;
	bool constantsRead = false;
};

// The entry points of the AIR module at `bitcode` (raw bitcode, or in LLVM's wrapper header).
// False where it is not one the reader can follow; `out` then holds nothing.
bool rgfxReadAirFunctions(const uint8_t *bitcode, size_t size, std::vector<RgfxAirFunction> &out);

// The signatures of a module's visible functions. The module lists them in the named metadata
// `air.visible`: one node per function, of the function, a node of its outputs and one of its
// inputs, then, for a stitchable function, a node `air.stitching_info` of its signature. That
// node's first operand is the return's `air.stitching_type` (none where it returns void), each
// later one an `air.stitching_argument` of an argument's `air.stitching_type` and its name (none
// for an unnamed argument), in the argument's order. A stitching type's first operand is the type's node: its name
// (`air.float_type`, `air.struct_type`, ...), its size and alignment, and an address space's
// node or none (a node of one `air.address_space` node, of the space's number), then what the
// kind adds:
//   scalars (`air.float_type`, `air.half_type`, `air.bfloat_type`, `air.int_type`, `air.uint_type`,
//     `air.short_type`, `air.ushort_type`, `air.char_type`, `air.uchar_type`, `air.bool_type`,
//     `air.long_type`, `air.ulong_type`) and `air.void_type` (a pointer's pointee): nothing;
//   `air.vector_type`, `air.packed_vector_type`: the element's node and the component count;
//   `air.matrix_type`: the element's node, the column count and the row count;
//   `air.array_type` (an array of C's), `air.array_of_type` (a metal::array): the element's node
//     and the length;
//   `air.struct_type`: the struct's name and a node of its fields (none for an atomic of no
//     size), each an `air.record_field` of its offset, a second integer, its type's node and its
//     name. An atomic is the struct `metal::_atomic<T>`;
//   `air.enum_type`: the enum's name and its underlying scalar's node;
//   `air.pointer_type`, `air.lvalue_reference_type` and `air.rvalue_reference_type` (references,
//     read as pointers): the pointee's node;
//   `air.texture_<shape>_type`, `air.depth_<shape>_type`: the sample type's node and the access
//     (`sample`, `read`, `write`, `read_write`);
//   `air.sampler_type`: nothing;
//   `air.visible_function_table_type`: an `air.function_type` node of the functions' signature;
//     `air.intersection_function_table_type`: the tag's name (`triangle_data`, ...);
//   `air.acceleration_structure_type`: `instancing` for an instance acceleration structure.
// Read from modules Apple's Metal compiler built from our own sources (tests/air_arguments.metal),
// as LLVM's text form of the same modules prints them.

enum class RgfxAirScalar : uint8_t { Float, Half, BFloat, Int, UInt, Short, UShort, Char, UChar, Bool, Long, ULong };

// A texture's shape, as its type's name gives it (`air.texture_2d_array_type`: T2DArray).
enum class RgfxAirTextureShape : uint8_t {
	T1D, T1DArray, T2D, T2DArray, T2DMS, T2DMSArray, Cube, CubeArray, T3D, Buffer1D
};

struct RgfxAirField;

// One type of a stitchable function's signature.
struct RgfxAirType {
	enum class Kind : uint8_t {
		Scalar,
		Vector,
		Matrix,
		Array,
		ArrayOf,
		Struct,
		Pointer,
		Texture,
		Sampler,
		Void,
		Enum,
		VisibleFunctionTable,
		IntersectionFunctionTable,
		AccelerationStructure,
	};
	Kind kind = Kind::Scalar;
	// Scalar, Vector and Matrix: the elements'; Texture: its samples'; Enum: its underlying type.
	RgfxAirScalar scalar = RgfxAirScalar::Float;
	// The type's size and alignment, in bytes, and its address space (0 where it has none).
	uint32_t size = 0, align = 0, addressSpace = 0;
	// Vector: its components; Matrix: its columns; Array and ArrayOf: its length.
	uint32_t count = 0;
	// Matrix: its rows.
	uint32_t rows = 0;
	// Vector: whether it is a packed one.
	bool packed = false;
	// Texture: its shape, whether it is a depth texture, and its access (Sample, Read, Write or
	// ReadWrite).
	RgfxAirTextureShape shape = RgfxAirTextureShape::T2D;
	bool depth = false;
	RgfxAirAccess access = RgfxAirAccess::Unknown;
	// Struct and Enum: its name; Struct: its fields, in their order.
	std::string name;
	std::vector<RgfxAirField> fields;
	// Array, ArrayOf and Pointer (a reference among them): the element's (the pointee's) type, its
	// one entry.
	std::vector<RgfxAirType> element;
};

struct RgfxAirField {
	std::string name;
	uint32_t offset = 0;
	RgfxAirType type;
};

// One visible function: its name and, where the module gives it, its signature.
struct RgfxAirVisible {
	enum class Signature : uint8_t {
		None,    // no `air.stitching_info`: a visible function that is not stitchable
		Read,    // the signature below
		Unknown, // a signature with a type the reader does not know
	};
	std::string name;
	Signature signature = Signature::None;
	// Whether it returns void; else its return type.
	bool returnsVoid = false;
	RgfxAirType returnType;
	// Its arguments' names (empty for an unnamed one) and types, in their order.
	std::vector<std::string> argumentNames;
	std::vector<RgfxAirType> argumentTypes;
	// The function constants its module declares, as RgfxAirFunction's.
	std::vector<RgfxAirConstant> constants;
	bool constantsRead = false;
};

// The visible functions of the AIR module at `bitcode` (as rgfxReadAirFunctions takes it), in
// `air.visible`'s order. False where it is not one the reader can follow; `out` then holds nothing.
bool rgfxReadAirVisible(const uint8_t *bitcode, size_t size, std::vector<RgfxAirVisible> &out);

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
// or none is there. An argument of any count (anyCount) is at every index from its first; one at
// any index (anyIndex) is at none: see rgfxAirWritesTextureAnywhere.
bool rgfxAirWritesTexture(const RgfxAirFunction &fn, uint32_t index);

// Whether `fn` has a texture argument it may write (write or read_write) whose binding index
// function constants set (anyIndex): a texture bound at any index of its stage may be it.
bool rgfxAirWritesTextureAnywhere(const RgfxAirFunction &fn);

#endif // RGFX_AIR_ARGUMENTS_HPP_
