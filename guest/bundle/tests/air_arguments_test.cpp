// The AIR argument reader's test (src/RgfxAirArguments.*), over the .metallib Apple's Metal
// compiler builds from tests/air_arguments.metal at test time (RGFX_AIR_ARGUMENTS names it):
// every entry point's binding arguments, kind, index, count and access, as the source declares
// them, those under function constants among them; which texture bindings each may write; no
// answer for a binding of a kind the reader does not know; and no answer, nor a crash, for bytes
// that are not such a module.
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

	std::printf("air_arguments_test: OK (%d checks)\n", checks);
	return 0;
}
