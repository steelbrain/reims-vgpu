// The AIR argument reader's test (src/RgfxAirArguments.*), over the .metallib Apple's Metal
// compiler builds from tests/air_arguments.metal at test time (RGFX_AIR_ARGUMENTS names it):
// every entry point's binding arguments, kind, index, count and access, as the source declares
// them, those under function constants among them; which texture bindings each may write; no
// answer for a binding of a kind the reader does not know; each stitchable function's signature,
// type by type, and an unknown one where it has a type the reader does not know; and no answer,
// nor a crash, for bytes that are not such a module.
#include "RgfxAirArguments.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

namespace {
int checks = 0;

void check(bool ok, const std::string &what)
{
	checks++;
	if (!ok) {
		std::fprintf(stderr, "air_arguments_test: FAILED: %s\n", what.c_str());
		std::exit(1);
	}
}

std::vector<uint8_t> readFile(const char *path)
{
	std::ifstream f(path, std::ios::binary);
	return std::vector<uint8_t>(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
}

// The entry point `name` of the library's function `name`.
RgfxAirFunction entry(const std::vector<uint8_t> &lib, const std::string &name)
{
	const uint8_t *bitcode = nullptr;
	size_t size = 0;
	check(rgfxMetallibBitcode(lib.data(), lib.size(), name, &bitcode, &size),
	      name + ": the library has its bitcode");
	std::vector<RgfxAirFunction> fns;
	check(rgfxReadAirFunctions(bitcode, size, fns), name + ": its module is read");
	check(fns.size() == 1 && fns[0].name == name, name + ": its module's one entry point is it");
	return fns.empty() ? RgfxAirFunction{} : fns[0];
}

const RgfxAirArgument *at(const RgfxAirFunction &fn, RgfxAirKind kind, uint32_t index)
{
	for (const RgfxAirArgument &a : fn.arguments)
		if (a.kind == kind && a.index == index)
			return &a;
	return nullptr;
}

// The visible function `name` of the library's function `name`.
RgfxAirVisible visible(const std::vector<uint8_t> &lib, const std::string &name)
{
	const uint8_t *bitcode = nullptr;
	size_t size = 0;
	check(rgfxMetallibBitcode(lib.data(), lib.size(), name, &bitcode, &size),
	      name + ": the library has its bitcode");
	std::vector<RgfxAirVisible> fns;
	check(rgfxReadAirVisible(bitcode, size, fns), name + ": its module is read");
	check(fns.size() == 1 && fns[0].name == name, name + ": its module's one visible function is it");
	return fns.empty() ? RgfxAirVisible{} : fns[0];
}

using Kind = RgfxAirType::Kind;
using Scalar = RgfxAirScalar;

bool isScalar(const RgfxAirType &t, Scalar s, uint32_t size)
{
	return t.kind == Kind::Scalar && t.scalar == s && t.size == size && t.align == size;
}

bool isVector(const RgfxAirType &t, Scalar s, uint32_t count, uint32_t size, bool packed = false)
{
	return t.kind == Kind::Vector && t.scalar == s && t.count == count && t.size == size && t.packed == packed;
}

bool isMatrix(const RgfxAirType &t, Scalar s, uint32_t columns, uint32_t rows, uint32_t size)
{
	return t.kind == Kind::Matrix && t.scalar == s && t.count == columns && t.rows == rows && t.size == size;
}

// A kind with one element (Array, ArrayOf, Pointer): its element, or null.
const RgfxAirType *elementOf(const RgfxAirType &t, Kind kind, uint32_t count = 0)
{
	return t.kind == kind && t.element.size() == 1 && t.count == count ? &t.element[0] : nullptr;
}

bool isTexture(const RgfxAirType &t, RgfxAirTextureShape shape, Scalar s, RgfxAirAccess access, bool depth = false)
{
	return t.kind == Kind::Texture && t.shape == shape && t.scalar == s && t.access == access && t.depth == depth;
}

// The field `name` at `offset` of the struct `t`, or null.
const RgfxAirType *field(const RgfxAirType &t, const std::string &name, uint32_t offset)
{
	for (const RgfxAirField &f : t.fields)
		if (f.name == name)
			return f.offset == offset ? &f.type : nullptr;
	return nullptr;
}

bool isPair(const RgfxAirType &t)
{
	const RgfxAirType *xy = field(t, "xy", 0), *w = field(t, "w", 8), *n = field(t, "n", 12);
	return t.kind == Kind::Struct && t.name == "SigPair" && t.size == 16 && t.align == 8 && t.fields.size() == 3 &&
	       xy != nullptr && isVector(*xy, Scalar::Float, 2, 8) && w != nullptr && isScalar(*w, Scalar::Float, 4) &&
	       n != nullptr && isScalar(*n, Scalar::Int, 4);
}

void expect(const RgfxAirFunction &fn, RgfxAirKind kind, uint32_t index, uint32_t count,
            RgfxAirAccess access, const char *what)
{
	const RgfxAirArgument *a = at(fn, kind, index);
	check(a != nullptr && a->count == count && a->access == access, fn.name + ": " + what);
}
} // namespace

int main()
{
	const char *path = std::getenv("RGFX_AIR_ARGUMENTS");
	check(path != nullptr, "RGFX_AIR_ARGUMENTS names the test library");
	const std::vector<uint8_t> lib = readFile(path);
	check(lib.size() > 0x58, "the test library is read");

	const RgfxAirFunction all = entry(lib, "every_access");
	check(all.stage == RgfxAirStage::Kernel, "every_access is a kernel");
	check(all.arguments.size() == 11, "every_access: its eleven binding arguments, no built-in");
	expect(all, RgfxAirKind::Texture, 0, 1, RgfxAirAccess::Read, "access::read at texture 0");
	expect(all, RgfxAirKind::Texture, 1, 1, RgfxAirAccess::Sample, "a sampled texture at 1");
	expect(all, RgfxAirKind::Texture, 2, 1, RgfxAirAccess::ReadWrite, "access::read_write at 2");
	expect(all, RgfxAirKind::Texture, 3, 1, RgfxAirAccess::Write, "access::write at 3");
	expect(all, RgfxAirKind::Texture, 4, 4, RgfxAirAccess::Sample, "four sampled from 4");
	expect(all, RgfxAirKind::Texture, 9, 2, RgfxAirAccess::Write, "two written from 9");
	expect(all, RgfxAirKind::Buffer, 0, 1, RgfxAirAccess::ReadWrite, "a device buffer at 0");
	expect(all, RgfxAirKind::Buffer, 1, 1, RgfxAirAccess::Read, "a constant at buffer 1");
	expect(all, RgfxAirKind::Sampler, 0, 1, RgfxAirAccess::Unknown, "a sampler at 0");
	const RgfxAirArgument *writesOne = at(all, RgfxAirKind::IndirectBuffer, 2);
	const RgfxAirArgument *readsOnly = at(all, RgfxAirKind::IndirectBuffer, 3);
	check(writesOne != nullptr && writesOne->writesTextures,
	      "every_access: the argument buffer at 2 writes a member texture");
	check(readsOnly != nullptr && !readsOnly->writesTextures,
	      "every_access: the one at 3 only samples its texture");
	const bool wantWrites[12] = {false, false, true, true, false, false,
	                             false, false, false, true, true, false};
	for (uint32_t i = 0; i < 12; i++)
		check(rgfxAirWritesTexture(all, i) == wantWrites[i],
		      "every_access: texture " + std::to_string(i) + (wantWrites[i] ? " is" : " is not") +
		          " written");

	const RgfxAirFunction none = entry(lib, "no_texture");
	check(none.stage == RgfxAirStage::Kernel && none.arguments.size() == 1, "no_texture: one buffer");
	expect(none, RgfxAirKind::Buffer, 3, 1, RgfxAirAccess::ReadWrite, "a device buffer at 3");
	check(!rgfxAirWritesTexture(none, 0), "no_texture writes no texture");

	const RgfxAirFunction vertex = entry(lib, "reads_in_vertex");
	check(vertex.stage == RgfxAirStage::Vertex && vertex.arguments.size() == 1,
	      "reads_in_vertex: a vertex function of one texture");
	expect(vertex, RgfxAirKind::Texture, 7, 1, RgfxAirAccess::Sample, "a sampled texture at 7");
	check(!rgfxAirWritesTexture(vertex, 7), "reads_in_vertex writes no texture");

	const RgfxAirFunction fragment = entry(lib, "writes_in_fragment");
	check(fragment.stage == RgfxAirStage::Fragment && fragment.arguments.size() == 3,
	      "writes_in_fragment: a fragment function of two textures and a sampler");
	expect(fragment, RgfxAirKind::Texture, 5, 1, RgfxAirAccess::Write, "access::write at 5");
	expect(fragment, RgfxAirKind::Texture, 0, 1, RgfxAirAccess::Sample, "a sampled texture at 0");
	check(rgfxAirWritesTexture(fragment, 5) && !rgfxAirWritesTexture(fragment, 0),
	      "writes_in_fragment writes texture 5 alone");

	// Arguments under function constants are the function's whatever the constants' values: a
	// texture written there is written.
	const RgfxAirFunction planes = entry(lib, "conditional_planes");
	check(planes.arguments.size() == 8,
	      "conditional_planes: four buffers, three textures and a buffer under constants");
	for (uint32_t i = 0; i < 4; i++)
		expect(planes, RgfxAirKind::Buffer, i, 1, RgfxAirAccess::Read,
		       ("a read buffer at " + std::to_string(i)).c_str());
	expect(planes, RgfxAirKind::Texture, 0, 1, RgfxAirAccess::Read, "a read texture at 0 under a constant");
	expect(planes, RgfxAirKind::Texture, 1, 1, RgfxAirAccess::Read, "a read texture at 1 under a constant");
	expect(planes, RgfxAirKind::Texture, 2, 1, RgfxAirAccess::Write,
	       "a written texture at 2 under a constant");
	expect(planes, RgfxAirKind::Buffer, 4, 1, RgfxAirAccess::ReadWrite,
	       "a device buffer at 4 under a constant");
	check(!rgfxAirWritesTexture(planes, 0) && !rgfxAirWritesTexture(planes, 1) &&
	          rgfxAirWritesTexture(planes, 2),
	      "conditional_planes writes texture 2 alone");

	// Binding indices function constants set: arguments at any index, which write no texture at
	// a given index; a function writing one may write a texture bound at any index.
	const RgfxAirFunction anywhere = entry(lib, "planes_where_constants_say");
	check(anywhere.arguments.size() == 3, "planes_where_constants_say: a buffer and two textures");
	expect(anywhere, RgfxAirKind::Buffer, 0, 1, RgfxAirAccess::Read, "a read buffer at 0");
	int anyRead = 0, anyWritten = 0;
	for (const RgfxAirArgument &a : anywhere.arguments) {
		if (a.kind == RgfxAirKind::Texture && a.anyIndex && a.index == 0 && a.count == 1) {
			anyRead += a.access == RgfxAirAccess::Read;
			anyWritten += a.access == RgfxAirAccess::Write;
		}
	}
	check(anyRead == 1 && anyWritten == 1,
	      "planes_where_constants_say: a read texture and a written one, each at any index");
	bool atNone = true;
	for (uint32_t i = 0; i < 32; i++)
		atNone = atNone && !rgfxAirWritesTexture(anywhere, i);
	check(atNone && rgfxAirWritesTextureAnywhere(anywhere),
	      "planes_where_constants_say writes a texture at any index, none at a given one");
	const RgfxAirFunction readsAnywhere = entry(lib, "reads_where_constants_say");
	check(readsAnywhere.arguments.size() == 2 && !rgfxAirWritesTextureAnywhere(readsAnywhere) &&
	          rgfxAirWritesTexture(readsAnywhere, 3) && !rgfxAirWritesTexture(readsAnywhere, 0),
	      "reads_where_constants_say: reads at any index, writes texture 3 alone");
	check(!rgfxAirWritesTextureAnywhere(planes) && !rgfxAirWritesTextureAnywhere(all),
	      "functions of fixed indices write nowhere else");

	// Threadgroup memory is bound by no buffer: the device buffer at its index is read alone.
	const RgfxAirFunction shares = entry(lib, "shares_threadgroup_memory");
	check(shares.arguments.size() == 2,
	      "shares_threadgroup_memory: its two buffers, no threadgroup memory");
	expect(shares, RgfxAirKind::Buffer, 0, 1, RgfxAirAccess::Read,
	       "a read buffer at 0, the threadgroup memory's index");
	expect(shares, RgfxAirKind::Buffer, 1, 1, RgfxAirAccess::ReadWrite, "a device buffer at 1");

	// A stage input's attribute, a function table and an acceleration structure: bindings of no
	// texture, beside those the function has.
	const RgfxAirFunction attributes = entry(lib, "reads_attributes");
	check(attributes.stage == RgfxAirStage::Vertex && attributes.arguments.size() == 1,
	      "reads_attributes: its buffer alone, past its stage input");
	expect(attributes, RgfxAirKind::Buffer, 1, 1, RgfxAirAccess::Read, "a constant at buffer 1");
	const RgfxAirFunction stageIn = entry(lib, "reads_stage_in");
	check(stageIn.stage == RgfxAirStage::Kernel && stageIn.arguments.size() == 2,
	      "reads_stage_in: its buffer and texture alone, past its stage input");
	expect(stageIn, RgfxAirKind::Buffer, 1, 1, RgfxAirAccess::ReadWrite, "a device buffer at 1");
	expect(stageIn, RgfxAirKind::Texture, 0, 1, RgfxAirAccess::Write,
	       "a written texture at 0 under a constant");
	check(rgfxAirWritesTexture(stageIn, 0), "reads_stage_in writes texture 0");
	const RgfxAirFunction table = entry(lib, "calls_through_table");
	check(table.arguments.size() == 2, "calls_through_table: its buffer and texture alone");
	expect(table, RgfxAirKind::Buffer, 0, 1, RgfxAirAccess::ReadWrite, "a device buffer at 0");
	expect(table, RgfxAirKind::Texture, 3, 1, RgfxAirAccess::Write, "access::write at 3");

	// A binding of a kind the reader does not know may be a texture: the function has no answer.
	{
		std::vector<uint8_t> other = lib;
		const char kind[] = "air.visible_function_table";
		auto it = std::search(other.begin(), other.end(), kind, kind + sizeof(kind) - 1);
		check(it != other.end(), "the library names the function table's kind");
		*(it + sizeof(kind) - 2) = '_';
		const uint8_t *bc = nullptr;
		size_t n = 0;
		std::vector<RgfxAirFunction> fns;
		check(rgfxMetallibBitcode(other.data(), other.size(), "calls_through_table", &bc, &n) &&
		          !rgfxReadAirFunctions(bc, n, fns) && fns.empty(),
		      "calls_through_table, its table's kind unknown: no answer");
		check(rgfxMetallibBitcode(other.data(), other.size(), "conditional_planes", &bc, &n) &&
		          rgfxReadAirFunctions(bc, n, fns),
		      "conditional_planes is still read beside it");
	}

	const uint8_t *bitcode = nullptr;
	size_t size = 0;
	check(!rgfxMetallibBitcode(lib.data(), lib.size(), "no_such_function", &bitcode, &size),
	      "a function the library lacks has no bitcode");
	check(!rgfxMetallibBitcode(lib.data(), 0x40, "every_access", &bitcode, &size),
	      "a cut header is no library");

	// A universal .metallib: its AIR slice is the library, in either header; one with no AIR
	// slice, and bytes that are no universal one, are left as they are.
	for (const bool wide : {false, true}) {
		const size_t entry = wide ? 32 : 20, header = 8 + 2 * entry;
		const size_t junk = 64, at = (header + junk + 15) & ~size_t(15);
		std::vector<uint8_t> fat(at + lib.size(), 0xAB);
		auto be = [&](size_t off, uint64_t v, int bytes) {
			for (int k = 0; k < bytes; k++)
				fat[off + k] = static_cast<uint8_t>(v >> (8 * (bytes - 1 - k)));
		};
		be(0, wide ? 0xCAFEBABF : 0xCAFEBABE, 4);
		be(4, 2, 4);
		const uint64_t cputypes[2] = {0x01000018, 0x01000017};
		const uint64_t offsets[2] = {header, at}, sizes[2] = {junk, lib.size()};
		for (int i = 0; i < 2; i++) {
			const size_t e = 8 + i * entry;
			be(e, cputypes[i], 4);
			be(e + 4, 0, 4);
			if (wide) {
				be(e + 8, offsets[i], 8);
				be(e + 16, sizes[i], 8);
			} else {
				be(e + 8, offsets[i], 4);
				be(e + 12, sizes[i], 4);
			}
		}
		std::memcpy(fat.data() + at, lib.data(), lib.size());
		const uint8_t *slice = fat.data();
		size_t sliceSize = fat.size();
		rgfxMetallibAirSlice(slice, sliceSize);
		check(slice == fat.data() + at && sliceSize == lib.size(),
		      std::string(wide ? "a 64-bit" : "a 32-bit") + " universal library's AIR slice");
		check(!rgfxMetallibBitcode(fat.data(), fat.size(), "every_access", &bitcode, &size) &&
		          rgfxMetallibBitcode(slice, sliceSize, "every_access", &bitcode, &size),
		      "the bitcode is the slice's, not the universal file's");
		be(8 + entry, 0x01000018, 4); // no AIR slice now
		slice = fat.data();
		sliceSize = fat.size();
		rgfxMetallibAirSlice(slice, sliceSize);
		check(slice == fat.data() && sliceSize == fat.size(), "no AIR slice: left as it is");
	}
	const uint8_t *plain = lib.data();
	size_t plainSize = lib.size();
	rgfxMetallibAirSlice(plain, plainSize);
	check(plain == lib.data() && plainSize == lib.size(), "a .metallib is left as it is");

	// Stitchable functions' signatures. A visible function that is not stitchable has none.
	{
		const RgfxAirVisible plainVisible = visible(lib, "doubled");
		check(plainVisible.signature == RgfxAirVisible::Signature::None && plainVisible.argumentTypes.empty(),
		      "doubled: visible, with no signature");
		std::vector<RgfxAirFunction> entries;
		const uint8_t *bc = nullptr;
		size_t n = 0;
		check(rgfxMetallibBitcode(lib.data(), lib.size(), "sig_void", &bc, &n) &&
		          rgfxReadAirFunctions(bc, n, entries) && entries.empty(),
		      "sig_void: a stitchable function is no entry point");
		std::vector<RgfxAirVisible> none;
		check(rgfxMetallibBitcode(lib.data(), lib.size(), "every_access", &bc, &n) &&
		          rgfxReadAirVisible(bc, n, none) && none.empty(),
		      "every_access: a kernel is no visible function");
	}
	const RgfxAirVisible sigVoid = visible(lib, "sig_void");
	check(sigVoid.signature == RgfxAirVisible::Signature::Read && sigVoid.returnsVoid &&
	          sigVoid.argumentNames == std::vector<std::string>{"a"} &&
	          isScalar(sigVoid.argumentTypes.at(0), Scalar::Float, 4),
	      "sig_void: void, of a float");
	const RgfxAirVisible sigNone = visible(lib, "sig_none");
	check(sigNone.signature == RgfxAirVisible::Signature::Read && !sigNone.returnsVoid &&
	          isScalar(sigNone.returnType, Scalar::Float, 4) && sigNone.argumentTypes.empty(),
	      "sig_none: a float, of nothing");

	const RgfxAirVisible sigStruct = visible(lib, "sig_struct");
	check(sigStruct.signature == RgfxAirVisible::Signature::Read && isPair(sigStruct.returnType),
	      "sig_struct returns SigPair, its fields by name, offset and type");
	check(sigStruct.argumentNames == std::vector<std::string>{"o", "i", "u", "h", "z"},
	      "sig_struct: its arguments' names, in order");
	if (sigStruct.argumentTypes.size() == 5) {
		const RgfxAirType &o = sigStruct.argumentTypes[0];
		const RgfxAirType *p = field(o, "p", 0), *h = field(o, "h", 16), *u = field(o, "u", 24);
		check(o.kind == Kind::Struct && o.name == "SigOuter" && o.size == 32 && o.fields.size() == 3 &&
		          p != nullptr && isPair(*p) && h != nullptr && isVector(*h, Scalar::Half, 4, 8) && u != nullptr &&
		          isScalar(*u, Scalar::UInt, 4),
		      "sig_struct: SigOuter, a SigPair nested");
		check(isScalar(sigStruct.argumentTypes[1], Scalar::Int, 4) &&
		          isVector(sigStruct.argumentTypes[2], Scalar::UInt, 2, 8) &&
		          isScalar(sigStruct.argumentTypes[3], Scalar::Half, 2) &&
		          isScalar(sigStruct.argumentTypes[4], Scalar::Bool, 1),
		      "sig_struct: int, uint2, half, bool");
	} else {
		check(false, "sig_struct: five arguments");
	}

	const RgfxAirVisible sigScalars = visible(lib, "sig_scalars");
	const std::vector<RgfxAirType> &sc = sigScalars.argumentTypes;
	check(isVector(sigScalars.returnType, Scalar::UChar, 4, 4) && sc.size() == 15, "sig_scalars: uchar4, of 15");
	if (sc.size() == 15) {
		check(isScalar(sc[0], Scalar::Char, 1) && isScalar(sc[1], Scalar::Short, 2) &&
		          isVector(sc[2], Scalar::UShort, 2, 4),
		      "sig_scalars: char, short, ushort2");
		check(isMatrix(sc[3], Scalar::Float, 4, 4, 64) && isMatrix(sc[8], Scalar::Half, 3, 2, 12) &&
		          isMatrix(sc[9], Scalar::Float, 2, 3, 32),
		      "sig_scalars: float4x4, half3x2, float2x3 (columns, rows)");
		check(isVector(sc[4], Scalar::Float, 3, 12, true) && isVector(sc[10], Scalar::Half, 2, 4, true),
		      "sig_scalars: packed_float3, packed_half2");
		check(isScalar(sc[5], Scalar::Long, 8) && isVector(sc[6], Scalar::ULong, 3, 32) &&
		          isVector(sc[7], Scalar::Bool, 2, 2),
		      "sig_scalars: long, ulong3, bool2");
		check(isScalar(sc[11], Scalar::UInt, 4) && isVector(sc[12], Scalar::Int, 4, 16) &&
		          isScalar(sc[13], Scalar::UChar, 1) && isVector(sc[14], Scalar::Char, 3, 4),
		      "sig_scalars: uint, int4, uchar, char3");
	}

	const RgfxAirVisible sigPointers = visible(lib, "sig_pointers");
	const std::vector<RgfxAirType> &pt = sigPointers.argumentTypes;
	check(sigPointers.argumentNames == std::vector<std::string>{"p", "q", "a", "sp", "m", "rows", "t"},
	      "sig_pointers: its arguments' names");
	if (pt.size() == 7) {
		const RgfxAirType *p = elementOf(pt[0], Kind::Pointer), *q = elementOf(pt[1], Kind::Pointer),
		                  *a = elementOf(pt[2], Kind::Pointer), *sp = elementOf(pt[3], Kind::Pointer),
		                  *m = elementOf(pt[4], Kind::Pointer), *rows = elementOf(pt[5], Kind::Pointer),
		                  *t = elementOf(pt[6], Kind::Pointer);
		check(p != nullptr && isScalar(*p, Scalar::Float, 4) && q != nullptr && isVector(*q, Scalar::Int, 4, 16) &&
		          a != nullptr && isScalar(*a, Scalar::Float, 4) && t != nullptr && isScalar(*t, Scalar::Half, 2),
		      "sig_pointers: to float (device, and a declared array), int4 (constant) and half (thread)");
		check(sp != nullptr && isPair(*sp) && m != nullptr && isMatrix(*m, Scalar::Float, 4, 4, 64) &&
		          pt[0].size == 8 && pt[0].align == 8,
		      "sig_pointers: to a struct and a matrix; a pointer's own size and alignment");
		const RgfxAirType *row = rows != nullptr ? elementOf(*rows, Kind::Array, 4) : nullptr;
		check(row != nullptr && isVector(*row, Scalar::Float, 2, 8) && rows->size == 32,
		      "sig_pointers: to an array of four float2");
	} else {
		check(false, "sig_pointers: seven arguments");
	}

	const RgfxAirVisible sigFields = visible(lib, "sig_fields");
	check(sigFields.argumentNames == std::vector<std::string>{"f", "af", "s"} && sigFields.argumentTypes.size() == 3 &&
	          isVector(sigFields.returnType, Scalar::Float, 4, 16),
	      "sig_fields: float4, of SigFields, a metal::array and a sampler");
	if (sigFields.argumentTypes.size() == 3) {
		const RgfxAirType &f = sigFields.argumentTypes[0];
		check(f.kind == Kind::Struct && f.name == "SigFields" && f.size == 256 && f.align == 16 && f.fields.size() == 17,
		      "SigFields: its 17 fields");
		auto arrayField = [&](const char *name, uint32_t offset, uint32_t count) {
			const RgfxAirType *t = field(f, name, offset);
			return t != nullptr ? elementOf(*t, Kind::Array, count) : nullptr;
		};
		auto pointerField = [&](const char *name, uint32_t offset) {
			const RgfxAirType *t = field(f, name, offset);
			return t != nullptr ? elementOf(*t, Kind::Pointer) : nullptr;
		};
		const RgfxAirType *a = arrayField("a", 0, 2), *pairs = arrayField("pairs", 8, 2),
		                  *grid = arrayField("grid", 40, 2), *v = arrayField("v", 64, 2), *m = arrayField("m", 96, 2),
		                  *hs = arrayField("hs", 128, 3), *ts = arrayField("ts", 200, 2), *ss = arrayField("ss", 216, 2),
		                  *ps = arrayField("ps", 232, 2);
		check(a != nullptr && isScalar(*a, Scalar::Float, 4) && pairs != nullptr && isPair(*pairs) && v != nullptr &&
		          isVector(*v, Scalar::Float, 4, 16) && m != nullptr && isMatrix(*m, Scalar::Float, 2, 2, 16) &&
		          hs != nullptr && isScalar(*hs, Scalar::Half, 2),
		      "SigFields: arrays of float, SigPair, float4, float2x2 and half");
		const RgfxAirType *row = grid != nullptr ? elementOf(*grid, Kind::Array, 3) : nullptr;
		check(row != nullptr && isScalar(*row, Scalar::Float, 4) && field(f, "grid", 40)->size == 24,
		      "SigFields: an array of arrays");
		check(ts != nullptr && isTexture(*ts, RgfxAirTextureShape::T2D, Scalar::Float, RgfxAirAccess::Sample) &&
		          ss != nullptr && ss->kind == Kind::Sampler && ps != nullptr && elementOf(*ps, Kind::Pointer) != nullptr,
		      "SigFields: arrays of textures, samplers and pointers");
		const RgfxAirType *fp = pointerField("f", 136), *sp = pointerField("sp", 144),
		                  *transform = pointerField("transform", 152), *rows = pointerField("rows", 160);
		check(fp != nullptr && isScalar(*fp, Scalar::Float, 4) && sp != nullptr && isPair(*sp) &&
		          transform != nullptr && isMatrix(*transform, Scalar::Float, 4, 4, 64) && rows != nullptr &&
		          elementOf(*rows, Kind::Array, 4) != nullptr,
		      "SigFields: pointers to float, SigPair, float4x4 and an array");
		const RgfxAirType *t = field(f, "t", 168), *smp = field(f, "s", 176), *af = field(f, "af", 184),
		                  *origin = field(f, "origin", 248);
		const RgfxAirType *afElement = af != nullptr ? elementOf(*af, Kind::ArrayOf, 3) : nullptr;
		check(t != nullptr && isTexture(*t, RgfxAirTextureShape::T2D, Scalar::Float, RgfxAirAccess::Sample) &&
		          smp != nullptr && smp->kind == Kind::Sampler && afElement != nullptr &&
		          isScalar(*afElement, Scalar::Float, 4) && origin != nullptr && isVector(*origin, Scalar::Float, 2, 8),
		      "SigFields: a texture, a sampler, a metal::array and a float2");
		const RgfxAirType *arg = elementOf(sigFields.argumentTypes[1], Kind::ArrayOf, 3);
		check(arg != nullptr && isScalar(*arg, Scalar::Float, 4) && sigFields.argumentTypes[2].kind == Kind::Sampler,
		      "sig_fields: a metal::array of three floats, a sampler");
	}

	const RgfxAirVisible sigTextures = visible(lib, "sig_textures");
	const std::vector<RgfxAirType> &tx = sigTextures.argumentTypes;
	if (tx.size() == 13) {
		using Shape = RgfxAirTextureShape;
		check(isTexture(tx[0], Shape::T2D, Scalar::Half, RgfxAirAccess::Read) &&
		          isTexture(tx[1], Shape::T2D, Scalar::Float, RgfxAirAccess::Write) &&
		          isTexture(tx[2], Shape::T2D, Scalar::Float, RgfxAirAccess::ReadWrite),
		      "sig_textures: read, write and read_write");
		check(isTexture(tx[3], Shape::T2D, Scalar::Float, RgfxAirAccess::Sample, true) &&
		          isTexture(tx[9], Shape::T2DMSArray, Scalar::Float, RgfxAirAccess::Read, true),
		      "sig_textures: depth textures");
		check(isTexture(tx[4], Shape::T3D, Scalar::Float, RgfxAirAccess::Sample) &&
		          isTexture(tx[5], Shape::CubeArray, Scalar::Float, RgfxAirAccess::Sample) &&
		          isTexture(tx[6], Shape::T1DArray, Scalar::Float, RgfxAirAccess::Sample) &&
		          isTexture(tx[7], Shape::Buffer1D, Scalar::Float, RgfxAirAccess::Read) &&
		          isTexture(tx[8], Shape::T2DMSArray, Scalar::Float, RgfxAirAccess::Read),
		      "sig_textures: each shape");
		check(isTexture(tx[10], Shape::T2D, Scalar::Int, RgfxAirAccess::Sample) &&
		          isTexture(tx[11], Shape::T2D, Scalar::UShort, RgfxAirAccess::Sample) && tx[12].kind == Kind::Sampler,
		      "sig_textures: of int and ushort; a sampler");
	} else {
		check(false, "sig_textures: thirteen arguments");
	}
	const RgfxAirVisible sigReturnsTexture = visible(lib, "sig_returns_texture");
	check(isTexture(sigReturnsTexture.returnType, RgfxAirTextureShape::T2D, Scalar::Float, RgfxAirAccess::Sample),
	      "sig_returns_texture returns a texture");

	// A type the reader does not know: that function's signature is unknown, read beside the rest.
	{
		std::vector<uint8_t> other = lib;
		const char kind[] = "air.sampler_type";
		size_t spoiltKinds = 0;
		for (auto it = other.begin();
		     (it = std::search(it, other.end(), kind, kind + sizeof(kind) - 1)) != other.end(); it++) {
			*(it + sizeof(kind) - 2) = '_';
			spoiltKinds++;
		}
		check(spoiltKinds > 0, "the library names the sampler's type");
		const RgfxAirVisible spoilt = visible(other, "sig_fields"), still = visible(other, "sig_struct");
		check(spoilt.signature == RgfxAirVisible::Signature::Unknown && spoilt.argumentTypes.empty(),
		      "sig_fields, a sampler's type unknown: an unknown signature");
		check(still.signature == RgfxAirVisible::Signature::Read && isPair(still.returnType),
		      "sig_struct is still read beside it");
	}

	// Bytes that are not such a module: none answers, and none crashes the reader.
	check(rgfxMetallibBitcode(lib.data(), lib.size(), "every_access", &bitcode, &size),
	      "every_access's bitcode, to spoil");
	std::vector<RgfxAirFunction> fns;
	check(!rgfxReadAirFunctions(bitcode, 16, fns) && fns.empty(), "a cut module answers nothing");
	check(!rgfxReadAirFunctions(lib.data(), lib.size(), fns), "a .metallib is not a module");
	const std::vector<uint8_t> good(bitcode, bitcode + size);
	uint32_t seed = 12345;
	int read = 0;
	for (int round = 0; round < 4000; round++) {
		std::vector<uint8_t> spoilt = good;
		for (int k = 0; k < 1 + round % 4; k++) {
			seed = seed * 1103515245u + 12345u;
			spoilt[(seed >> 8) % spoilt.size()] ^= static_cast<uint8_t>(1u << ((seed >> 4) & 7));
		}
		const size_t cut = round % 7 == 0 ? (seed >> 3) % spoilt.size() : spoilt.size();
		if (rgfxReadAirFunctions(spoilt.data(), cut, fns))
			read++;
	}
	check(true, "4000 spoilt modules read without a crash (" + std::to_string(read) + " answered)");
	check(rgfxMetallibBitcode(lib.data(), lib.size(), "sig_fields", &bitcode, &size),
	      "sig_fields's bitcode, to spoil");
	const std::vector<uint8_t> goodVisible(bitcode, bitcode + size);
	std::vector<RgfxAirVisible> visibles;
	read = 0;
	for (int round = 0; round < 4000; round++) {
		std::vector<uint8_t> spoilt = goodVisible;
		for (int k = 0; k < 1 + round % 4; k++) {
			seed = seed * 1103515245u + 12345u;
			spoilt[(seed >> 8) % spoilt.size()] ^= static_cast<uint8_t>(1u << ((seed >> 4) & 7));
		}
		const size_t cut = round % 7 == 0 ? (seed >> 3) % spoilt.size() : spoilt.size();
		if (rgfxReadAirVisible(spoilt.data(), cut, visibles))
			read++;
	}
	check(true, "4000 spoilt modules' signatures read without a crash (" + std::to_string(read) + " answered)");

	std::printf("air_arguments_test: OK (%d checks)\n", checks);
	return 0;
}
