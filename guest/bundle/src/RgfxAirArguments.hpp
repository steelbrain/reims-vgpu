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
// here. An OpenCL C kernel's argument passed by value (a scalar, a vector, a struct or a union)
// is an `air.constant`, bound at an `air.location_index` among the buffers, with no access key;
// the kernel's function takes it as a pointer of address space 2 to the value. It is read as a
// buffer at its index, of no access the reader knows (read: Metal's compiler on macOS 27.0.1, the
// M3 Max, OpenCL C programs of our own compiled as the GL renderer compiles them, for each scalar
// type, vectors of 2 to 16 components, half and half4, structs, a nested struct and a union, as
// LLVM's text form prints the modules; tests/cl_mode_tests.mm). Read from modules that Apple's
// Metal compiler built from our own sources (tests/air_arguments.metal), and as LLVM's text form
// of the same modules prints them; a count set by function constants, which no source of ours
// declares, from macOS 27's frameworks' modules on the M3 Max, where 709 entry points bind one
// (Metal Performance Shaders' kernels among them).
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
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "reims_graphics_wire.h" // RgfxFunctionConstant

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
// constants its module declares (`constantsRead` false where they did not read). `argumentsRead`
// is false where an argument's node is not one the reader follows (a binding of a kind it does
// not know, a texture of no access it knows, a binding it cannot read): which bindings the
// function writes is then unknown, and `arguments` is empty.
struct RgfxAirFunction {
	std::string name;
	RgfxAirStage stage;
	bool argumentsRead = true;
	std::vector<RgfxAirArgument> arguments;
	std::vector<RgfxAirConstant> constants;
	bool constantsRead = false;
};

// The entry points of the AIR module at `bitcode` (raw bitcode, or in LLVM's wrapper header),
// each with its own answer: an entry point whose arguments the reader does not follow is listed,
// `argumentsRead` false, beside the others, and one whose node names no function of the module is
// left out. False where the bytes are not a module the reader can follow; `out` then holds
// nothing.
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
		Unknown, // a signature with a type or a node the reader does not know
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
// `air.visible`'s order, each with its own answer: one whose node is not of the form above has a
// signature Unknown, and one whose node names no function of the module is left out. False where
// the bytes are not a module the reader can follow; `out` then holds nothing.
bool rgfxReadAirVisible(const uint8_t *bitcode, size_t size, std::vector<RgfxAirVisible> &out);

// --- An entry point's nodes, whole, for its pipeline's reflection -----------------------------
// Each argument node of an entry point (`air.kernel`, `air.vertex`, `air.fragment`, `air.object`,
// `air.mesh`), as its metadata states it (measured: the modules Apple's Metal compiler builds from
// tools/oracle/oracle_reflection.metal, macOS 27.2 SDK, as LLVM's text form prints them). A node
// is its parameter's position, then `air.function_constant` and its predicate's node where it is
// declared under a function constant, then its kind (`air.buffer`, `air.texture`, `air.sampler`,
// `air.indirect_buffer`, `air.thread_position_in_grid`, ...), then keys and their operands, in no
// fixed order:
//   `air.location_index` index count   its binding: the index and how many it takes (an array's
//                                      length), each an integer or a global the module's
//                                      initialiser sets, for one function constants compute;
//   `air.buffer_size` n, `air.address_space` n, `air.arg_type_size` n, `air.arg_type_align_size` n;
//   `air.struct_type_info` !layout     its pointee's layout (below);
//   `air.arg_type_name` "T", `air.arg_name` "n";
//   `air.read`, `air.write`, `air.read_write`, `air.sample`: its access;
//   `air.arg_unused`: the function never references it;
//   `air.render_target` n, as its kind: a fragment function's input of colour attachment n's value
//   (`[[color(n)]]`);
//   `air.mesh` !info: a mesh function's mesh, its `air.mesh_type_info` node of its vertex and
//   primitive data (each a node of outputs, below), its vertex and primitive counts and its
//   topology (`air.triangle`, ...).
// An entry's outputs are a node of nodes, each its kind (`air.position`, `air.vertex_output`,
// `air.render_target`, ...) and keys, `air.arg_type_name` "T" and `air.arg_name` "n" among them.
// A predicate's node is the global the initialiser sets, the constant's type and its name: the
// function constant's own name where the predicate is one constant (`[[function_constant(kOn)]]`),
// the name of the constant it is an expression of (`kOn || kOff`) where it is one of several.
// A layout is a run of members, each `[air.struct_type_info !layout] offset size arrayLength
// "type" "name"` and, in an argument buffer's, `air.indirect_argument` and the member's own node
// (or its first id, for a struct of members), or `air.render_target` n, `air.raster_order_group` n.

struct RgfxAirNode;

// One output of a vertex function (its outputs' node), or one member of a mesh's vertex or
// primitive data: its kind (`air.position`, `air.vertex_output`, `air.mesh_vertex_data`,
// `air.render_target`, ...), its type's name and its name (empty where it has none).
struct RgfxAirOutput {
	std::string kind, typeName, name;
};

struct RgfxAirMember {
	uint32_t offset = 0, size = 0, arrayLength = 0;
	std::string typeName, name;
	// `air.struct_type_info` ahead of it: its own type's layout.
	std::shared_ptr<const std::vector<RgfxAirMember>> layout;
	// `air.indirect_argument`: an argument buffer's member's node, or its first id where the
	// member is a struct of members (`hasFirstId`).
	std::shared_ptr<const RgfxAirNode> indirect;
	bool hasFirstId = false;
	uint32_t firstId = 0;
	bool hasRenderTarget = false, hasRasterOrderGroup = false;
	uint32_t renderTarget = 0, rasterOrderGroup = 0;
};

struct RgfxAirNode {
	bool hasPosition = false;
	uint32_t position = 0;
	std::string kind;
	// Declared under a function constant: the predicate's name (the constant's, where it is one),
	// and the global the module's initialiser stores its value in (UINT64_MAX where it names
	// none).
	bool predicated = false;
	std::string predicate;
	uint64_t predicateGlobal = UINT64_MAX;
	// `air.location_index`: its index and count, each an integer, or (`indexIsGlobal`,
	// `countIsGlobal`) one function constants compute, which the module's initialiser stores in
	// the global `indexGlobal` or `countGlobal`, the integer then 0.
	bool hasLocation = false, indexIsGlobal = false, countIsGlobal = false;
	uint32_t index = 0, count = 0;
	uint64_t indexGlobal = UINT64_MAX, countGlobal = UINT64_MAX;
	RgfxAirAccess access = RgfxAirAccess::Unknown;
	bool hasAddressSpace = false, hasBufferSize = false, hasTypeSize = false, hasTypeAlign = false;
	uint32_t addressSpace = 0, bufferSize = 0, typeSize = 0, typeAlign = 0;
	std::string typeName, name;
	bool unused = false;
	std::shared_ptr<const std::vector<RgfxAirMember>> layout;
	// `air.render_target` n, its kind: a fragment function's input of colour attachment n's value.
	bool hasRenderTarget = false;
	uint32_t renderTarget = 0;
	// `air.mesh`: its vertex and primitive data, its vertex and primitive counts and its topology's
	// name.
	bool hasMesh = false;
	std::vector<RgfxAirOutput> meshVertexData, meshPrimitiveData;
	uint32_t meshVertices = 0, meshPrimitives = 0;
	std::string meshTopology;
};

enum class RgfxAirEntryStage : uint8_t { Kernel, Vertex, Fragment, Object, Mesh };

struct RgfxAirEntry {
	std::string name;
	// Its function's value number in the module.
	uint64_t function = 0;
	RgfxAirEntryStage stage = RgfxAirEntryStage::Kernel;
	// False where a node of it (its entry's, an output's, an argument's or a layout's, its work
	// group size's) is not of a form the reader knows: its outputs, arguments and work group size
	// are then unknown, and empty here.
	bool nodesRead = true;
	// Its outputs (the entry's second node), in order: none for a kernel or a void function.
	std::vector<RgfxAirOutput> outputs;
	std::vector<RgfxAirNode> arguments;
	// A kernel's `air.work_group_size` and its three sizes, a node of the entry's after its
	// arguments' (an OpenCL C kernel's reqd_work_group_size, as Metal's compiler writes it for the
	// GL renderer's programs; measured: macOS 26.7.1 and macOS 27).
	bool hasWorkGroupSize = false;
	uint32_t workGroupSize[3] = {1, 1, 1};
};

// The code of a module that declares function constants: its functions' bodies, as the reader
// decodes them for rgfxAirSpecialise. Opaque.
struct RgfxAirCode;

// A module's entry points, its function constants' indices by name (`air.function_constants`:
// each the constant's global, its type, its name and its index; one whose node does not read is
// left out, so a predicate that names it has no value), and, where it declares function
// constants, its code (null where the reader does not follow it).
struct RgfxAirEntries {
	std::vector<RgfxAirEntry> entries;
	std::vector<std::pair<std::string, uint32_t>> constants;
	std::shared_ptr<const RgfxAirCode> code;
};

// The entry points of the AIR module at `bitcode` (as rgfxReadAirFunctions takes it), every node
// of each whole, each entry point with its own answer: one a node or layout of which is not of a
// form the reader knows is listed, `nodesRead` false, beside the others, and one whose node names
// no function of the module is left out. False where the bytes are not a module the reader can
// follow; `out` then holds nothing.
bool rgfxReadAirEntries(const uint8_t *bitcode, size_t size, RgfxAirEntries &out);

// --- One read of a module, kept for the process -----------------------------------------------

// What the three readers above answer for one module, from one read of its bitstream: each
// `...Read` and its value are what that reader returns and puts in `out` for the same bytes.
struct RgfxAirModule {
	bool functionsRead = false;
	std::vector<RgfxAirFunction> functions; // rgfxReadAirFunctions
	bool visibleRead = false;
	std::vector<RgfxAirVisible> visible; // rgfxReadAirVisible
	bool entriesRead = false;
	RgfxAirEntries entries; // rgfxReadAirEntries
};

// Reads the AIR module at `bitcode` (as rgfxReadAirFunctions takes it) once, into `out`.
void rgfxReadAirModule(const uint8_t *bitcode, size_t size, RgfxAirModule &out);

// What a module of `size` bytes read as `module` costs a cache's budget: its bytes, and the bytes
// its decoded code holds (RgfxAirEntries::code: its containers' elements, for a module that
// declares function constants), which grow with its function bodies rather than with its
// metadata.
size_t rgfxAirModuleCost(const RgfxAirModule &module, size_t size);

// The modules a process read, by their bytes: a module's reads are a function of its bytes alone,
// so every lookup of equal bytes, from any library over them, shares one read. It keeps the most
// recently looked-up modules, a copy of each one's bytes (its key, compared whole) with its
// reads, while their costs (rgfxAirModuleCost) total at most `budget`; a module it let go is read
// again at its next lookup. The reads it hands out stay valid while a holder keeps them. Safe on
// any thread; two threads that look up one module at once may each read it, and both are answered
// the first read kept.
class RgfxAirModuleCache {
public:
	explicit RgfxAirModuleCache(size_t budget) : budget_(budget) {}
	RgfxAirModuleCache(const RgfxAirModuleCache &) = delete;
	RgfxAirModuleCache &operator=(const RgfxAirModuleCache &) = delete;

	// The reads of the module at `bitcode`: those kept for equal bytes, else read now (and kept).
	std::shared_ptr<const RgfxAirModule> moduleOf(const uint8_t *bitcode, size_t size);

	// How many modules it read, how many it keeps, and their costs (tests).
	uint64_t reads() const;
	size_t kept() const;
	size_t keptBytes() const;

private:
	struct Kept {
		std::string bytes;
		std::shared_ptr<const RgfxAirModule> module;
		size_t cost;
	};
	using Order = std::list<Kept>;
	mutable std::mutex lock_;
	Order order_; // the most recently looked up first
	std::unordered_map<std::string_view, Order::iterator> byBytes_; // views of each Kept's bytes
	size_t keptBytes_ = 0;
	uint64_t reads_ = 0;
	const size_t budget_;
};

// The costs of the modules the process keeps (rgfxAirModules). measured: the 40,177 distinct
// modules of macOS 27's frameworks' .metallibs on the M3 Max are 11,072 bytes each on average, so
// this keeps about 1,500 such modules of no code. One of no code holds about 1.5 times its bytes
// in all, its copy and its reads (11,770 bytes in use for 7,690 of bitcode, over 2,000 of those
// modules, arm64 and x86_64 alike), so a full cache of them holds about 25 MiB. One of code holds
// about 1.2 times its cost (158 MB in use for costs of 131 MB, over the first 1,000 such modules,
// arm64), so a full cache of them holds about 19 MiB.
constexpr size_t kRgfxAirModuleCacheBytes = 16u << 20;

// The process's cache of modules, of kRgfxAirModuleCacheBytes, which every library reads its
// functions' modules through; never destroyed, as a library may be freed on any thread while the
// process exits.
RgfxAirModuleCache &rgfxAirModules();

// --- An entry point's specialisation by its function constants' values ----------------------
// Metal specialises a function by its constants' values before it compiles it: the module's
// initialisers (its functions of section `air.static_init`, which store each constant's value,
// and each value computed from constants, in a global of their own, from the constant's own
// global of section `air.fc_initializer`) run with the values, the function's code reads those
// globals, and code the values make unreachable or unused is removed. So which arguments the
// specialised function uses, and the binding indices, counts and predicates function constants
// compute, are the code's under those values (measured: an AMD Radeon Pro 5300M, an Intel UHD
// Graphics 630 and an Apple M3 Max answer an argument read only in a branch a false constant
// skips unused, and a buffer at `[[buffer(kIndex)]]` at the index kIndex is given; the modules
// as LLVM's text form prints them).
//
// The reader decodes a module's function bodies (FUNCTION_BLOCK, of LLVM's documented records,
// as a module of typed pointers writes them) where it declares function constants. It runs the
// initialisers over integers and floating-point scalars: a load of a constant's global gives the
// constant's value (none where the function has no value for it, any value where the constant's
// node in `air.function_constants` does not read), `air.is_function_constant_defined` whether it
// has one, `air.normalize_function_constant_predicate` 1 for a value other
// than 0. In the entry point's body it propagates the values those globals then hold through the
// branches, switches, selects and phis they decide (sparse conditional constant propagation), and
// counts as used each argument a live instruction of a reachable block reads: an instruction is
// live where it stores, writes memory by an atomic or a fence, returns, branches on a value no
// constant decides, or calls a function that may write memory, unwind or not return (its
// attributes' `readnone`, `readonly` or `memory`, `nounwind` and `willreturn` say), or where a
// live instruction reads its result. Anything else the reader does not follow leaves the code
// unread, and the AIR's own answers stand.
struct RgfxAirSpecialisation {
	// The integer the module's initialisers leave in each global they store one in, by the
	// global's value number, where every value it is computed from is known.
	std::map<uint64_t, int64_t> globals;
	// Whether the entry point's body was followed; where it was, whether the specialised code may
	// use each of its parameters, by position.
	bool bodyRead = false;
	std::vector<bool> used;
};

// The specialisation of `entry` of `module` by the `count` function constant values at `values`;
// false where the module has no code the reader follows (`out` then empty).
bool rgfxAirSpecialise(const RgfxAirEntries &module, const RgfxAirEntry &entry, const RgfxFunctionConstant *values,
                       uint32_t count, RgfxAirSpecialisation &out);

// Narrows a universal .metallib (a big-endian fat header: magic CA FE BA BE, or CA FE BA BF with
// 64-bit offsets) to its slice of generic AIR bitcode, a .metallib itself; other bytes, and a
// universal one with no such slice, are left as they are.
void rgfxMetallibAirSlice(const uint8_t *&bytes, size_t &size);

// The bitcode of the function `name` in the .metallib at `lib` (one slice, as
// rgfxMetallibAirSlice gives): false where the bytes are no .metallib, or it has no such function
// or no bitcode for it.
bool rgfxMetallibBitcode(const uint8_t *lib, size_t size, const std::string &name,
                         const uint8_t **bitcode, size_t *bitcodeSize);

// rgfxMetallibBitcode's answers for a .metallib, by function name, from one walk of its function
// list: a lookup hashes the name where rgfxMetallibBitcode walks the list to it. The list's first
// function of a name answers for it, and a function past the first one whose group does not read
// answers nothing, as there.
class RgfxMetallibBitcodeIndex {
public:
	// Walks the function list of the `size` bytes at `lib` (one slice, as rgfxMetallibAirSlice
	// gives).
	RgfxMetallibBitcodeIndex(const uint8_t *lib, size_t size);

	// rgfxMetallibBitcode(bytes, size, name, ...), for `bytes` whose function list is the one the
	// index walked: the same bytes, or bytes of the same content key (RgfxLibraryKey.hpp), whose
	// function list and header are those bytes'. Bytes of another length or header are walked.
	bool find(const uint8_t *bytes, size_t size, const std::string &name, const uint8_t **bitcode,
	          size_t *bitcodeSize) const;

private:
	struct Range {
		bool valid;
		uint64_t offset, size;
	};
	bool read_ = false; // the header read: the section lies inside the bytes
	size_t size_ = 0;
	uint64_t list_ = 0, section_ = 0, sectionSize_ = 0;
	std::unordered_map<std::string, Range> byName_;
};

// What `fn` does with the textures bound at `index` of its stage: true where an argument there
// may write them (write or read_write), false where its arguments there only read or sample them
// or none is there. An argument of any count (anyCount) is at every index from its first; one at
// any index (anyIndex) is at none: see rgfxAirWritesTextureAnywhere.
bool rgfxAirWritesTexture(const RgfxAirFunction &fn, uint32_t index);

// Whether `fn` has a texture argument it may write (write or read_write) whose binding index
// function constants set (anyIndex): a texture bound at any index of its stage may be it.
bool rgfxAirWritesTextureAnywhere(const RgfxAirFunction &fn);

#endif // RGFX_AIR_ARGUMENTS_HPP_
