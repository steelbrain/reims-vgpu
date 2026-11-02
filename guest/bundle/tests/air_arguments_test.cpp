// The AIR argument reader's test (src/RgfxAirArguments.*), over the .metallib Apple's Metal
// compiler builds from tests/air_arguments.metal at test time (RGFX_AIR_ARGUMENTS names it):
// every entry point's binding arguments, kind, index, count and access, as the source declares
// them; which texture bindings each may write; and no answer, nor a crash, for bytes that are
// not such a module.
#include "RgfxAirArguments.hpp"

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

	const uint8_t *bitcode = nullptr;
	size_t size = 0;
	check(!rgfxMetallibBitcode(lib.data(), lib.size(), "no_such_function", &bitcode, &size),
	      "a function the library lacks has no bitcode");
	check(!rgfxMetallibBitcode(lib.data(), 0x40, "every_access", &bitcode, &size),
	      "a cut header is no library");

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
