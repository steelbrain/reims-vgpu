// The AIR argument reader's test (src/RgfxAirArguments.*), over the .metallib Apple's Metal
// compiler builds from tests/air_arguments.metal at test time (RGFX_AIR_ARGUMENTS names it):
// every entry point's binding arguments, kind, index, count and access, as the source declares
// them, those under function constants among them; which texture bindings each may write; no
// answer for a binding of a kind the reader does not know; each stitchable function's signature,
// type by type, and an unknown one where it has a type the reader does not know; each module's
// function constants, as its function's CNST tag lists them; the same over the .metallib built
// from tests/air_arguments_old.metal for macOS 10.12 (RGFX_AIR_ARGUMENTS_OLD); every node of an
// entry point whole (rgfxReadAirEntries), over the oracle probe's library (RGFX_ORACLE_REFLECTION,
// built from tools/oracle/oracle_reflection.metal); no answer, nor a crash, for bytes that are
// not such a module; one read of each module answering what the three readers answer, and the
// process's cache of those reads (RgfxAirModuleCache); and every function of the repo's committed
// .metallibs found by the read the device makes for its kind (RGFX_COMMITTED_METALLIBS).
#include "RgfxAirArguments.hpp"
#include "RgfxFunctionConstants.hpp"

#include <dirent.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <thread>
#include <utility>
#include <vector>

// What the readers answer, compared whole: every field, a shared layout or node by its value.
static bool operator==(const RgfxAirArgument &a, const RgfxAirArgument &b);
static bool operator==(const RgfxAirConstant &a, const RgfxAirConstant &b);
static bool operator==(const RgfxAirFunction &a, const RgfxAirFunction &b);
static bool operator==(const RgfxAirType &a, const RgfxAirType &b);
static bool operator==(const RgfxAirField &a, const RgfxAirField &b);
static bool operator==(const RgfxAirVisible &a, const RgfxAirVisible &b);
static bool operator==(const RgfxAirOutput &a, const RgfxAirOutput &b);
static bool operator==(const RgfxAirMember &a, const RgfxAirMember &b);
static bool operator==(const RgfxAirNode &a, const RgfxAirNode &b);
static bool operator==(const RgfxAirEntry &a, const RgfxAirEntry &b);
static bool operator==(const RgfxAirEntries &a, const RgfxAirEntries &b);

template <class T>
static bool sameValue(const std::shared_ptr<const T> &a, const std::shared_ptr<const T> &b)
{
	return a == nullptr ? b == nullptr : b != nullptr && *a == *b;
}

static bool operator==(const RgfxAirArgument &a, const RgfxAirArgument &b)
{
	return a.kind == b.kind && a.index == b.index && a.count == b.count && a.access == b.access &&
	       a.writesTextures == b.writesTextures && a.anyIndex == b.anyIndex && a.anyCount == b.anyCount;
}

static bool operator==(const RgfxAirConstant &a, const RgfxAirConstant &b)
{
	return a.name == b.name && a.type == b.type && a.index == b.index && a.required == b.required;
}

static bool operator==(const RgfxAirFunction &a, const RgfxAirFunction &b)
{
	return a.name == b.name && a.stage == b.stage && a.arguments == b.arguments && a.constants == b.constants &&
	       a.constantsRead == b.constantsRead;
}

static bool operator==(const RgfxAirType &a, const RgfxAirType &b)
{
	return a.kind == b.kind && a.scalar == b.scalar && a.size == b.size && a.align == b.align &&
	       a.addressSpace == b.addressSpace && a.count == b.count && a.rows == b.rows && a.packed == b.packed &&
	       a.shape == b.shape && a.depth == b.depth && a.access == b.access && a.name == b.name &&
	       a.fields == b.fields && a.element == b.element;
}

static bool operator==(const RgfxAirField &a, const RgfxAirField &b)
{
	return a.name == b.name && a.offset == b.offset && a.type == b.type;
}

static bool operator==(const RgfxAirVisible &a, const RgfxAirVisible &b)
{
	return a.name == b.name && a.signature == b.signature && a.returnsVoid == b.returnsVoid &&
	       a.returnType == b.returnType && a.argumentNames == b.argumentNames && a.argumentTypes == b.argumentTypes &&
	       a.constants == b.constants && a.constantsRead == b.constantsRead;
}

static bool operator==(const RgfxAirOutput &a, const RgfxAirOutput &b)
{
	return a.kind == b.kind && a.typeName == b.typeName && a.name == b.name;
}

static bool operator==(const RgfxAirMember &a, const RgfxAirMember &b)
{
	return a.offset == b.offset && a.size == b.size && a.arrayLength == b.arrayLength && a.typeName == b.typeName &&
	       a.name == b.name && sameValue(a.layout, b.layout) && sameValue(a.indirect, b.indirect) &&
	       a.hasFirstId == b.hasFirstId && a.firstId == b.firstId && a.hasRenderTarget == b.hasRenderTarget &&
	       a.hasRasterOrderGroup == b.hasRasterOrderGroup && a.renderTarget == b.renderTarget &&
	       a.rasterOrderGroup == b.rasterOrderGroup;
}

static bool operator==(const RgfxAirNode &a, const RgfxAirNode &b)
{
	return a.hasPosition == b.hasPosition && a.position == b.position && a.kind == b.kind &&
	       a.predicated == b.predicated && a.predicate == b.predicate && a.hasLocation == b.hasLocation &&
	       a.indexIsGlobal == b.indexIsGlobal && a.index == b.index && a.count == b.count && a.access == b.access &&
	       a.hasAddressSpace == b.hasAddressSpace && a.hasBufferSize == b.hasBufferSize &&
	       a.hasTypeSize == b.hasTypeSize && a.hasTypeAlign == b.hasTypeAlign && a.addressSpace == b.addressSpace &&
	       a.bufferSize == b.bufferSize && a.typeSize == b.typeSize && a.typeAlign == b.typeAlign &&
	       a.typeName == b.typeName && a.name == b.name && a.unused == b.unused && sameValue(a.layout, b.layout) &&
	       a.hasMesh == b.hasMesh && a.meshVertexData == b.meshVertexData &&
	       a.meshPrimitiveData == b.meshPrimitiveData && a.meshVertices == b.meshVertices &&
	       a.meshPrimitives == b.meshPrimitives && a.meshTopology == b.meshTopology;
}

static bool operator==(const RgfxAirEntry &a, const RgfxAirEntry &b)
{
	return a.name == b.name && a.stage == b.stage && a.outputs == b.outputs && a.arguments == b.arguments;
}

static bool operator==(const RgfxAirEntries &a, const RgfxAirEntries &b)
{
	return a.entries == b.entries && a.constants == b.constants;
}

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

// The functions of the library's function list: each one's name and the offset of its public
// metadata (its OFFT tag's first field), as RgfxLibrary.mm reads the list.
struct Listed {
	std::string name;
	uint64_t publicOffset = 0;
	int type = -1; // its TYPE tag's byte: 0 to 2 for a vertex, fragment or kernel function
};

std::vector<Listed> listed(const std::vector<uint8_t> &lib)
{
	auto rd = [&](uint64_t at, size_t n) {
		uint64_t v = 0;
		if (at <= lib.size() && n <= lib.size() - at)
			std::memcpy(&v, lib.data() + at, n);
		return v;
	};
	std::vector<Listed> out;
	uint64_t q = rd(0x18, 8);
	const uint64_t count = rd(q, 4);
	q += 4;
	for (uint64_t i = 0; i < count && q < lib.size(); i++) {
		const uint64_t end = q + rd(q, 4);
		Listed f;
		for (uint64_t t = q + 4; t + 6 <= end && std::memcmp(lib.data() + t, "ENDT", 4) != 0;) {
			const uint64_t n = rd(t + 4, 2);
			if (std::memcmp(lib.data() + t, "NAME", 4) == 0)
				f.name.assign(reinterpret_cast<const char *>(lib.data() + t + 6),
				              strnlen(reinterpret_cast<const char *>(lib.data() + t + 6), n));
			else if (std::memcmp(lib.data() + t, "OFFT", 4) == 0)
				f.publicOffset = rd(t + 6, 8);
			else if (std::memcmp(lib.data() + t, "TYPE", 4) == 0)
				f.type = static_cast<int>(rd(t + 6, 1));
			t += 6 + n;
		}
		out.push_back(f);
		q = end;
	}
	return out;
}

bool hasConstant(const std::vector<RgfxAirConstant> &cs, const char *name, uint32_t type,
                 uint32_t index, bool required)
{
	for (const RgfxAirConstant &c : cs)
		if (c.name == name)
			return c.type == type && c.index == index && c.required == required;
	return false;
}

// Every function's constants in its module, as its CNST tag lists them: the same names, types,
// indices and flags. Answers how many functions it compared, and how many of them have constants.
std::pair<size_t, size_t> constantsAsCnst(const std::vector<uint8_t> &lib)
{
	size_t compared = 0, withConstants = 0;
	for (const Listed &f : listed(lib)) {
		std::vector<RgfxConstantDecl> cnst;
		check(rgfxReadConstantTable(lib.data(), lib.size(), f.publicOffset, cnst),
		      f.name + ": its CNST tag reads");
		const uint8_t *bc = nullptr;
		size_t bcSize = 0;
		check(rgfxMetallibBitcode(lib.data(), lib.size(), f.name, &bc, &bcSize),
		      f.name + ": its bitcode");
		std::vector<RgfxAirFunction> fns;
		std::vector<RgfxAirVisible> vis;
		const std::vector<RgfxAirConstant> *air = nullptr;
		if (rgfxReadAirFunctions(bc, bcSize, fns) && fns.size() == 1 && fns[0].constantsRead)
			air = &fns[0].constants;
		else if (rgfxReadAirVisible(bc, bcSize, vis) && vis.size() == 1 && vis[0].constantsRead)
			air = &vis[0].constants;
		if (air == nullptr)
			continue; // a module the reader does not follow (sig_acceleration's among them)
		compared++;
		withConstants += air->empty() ? 0 : 1;
		bool same = air->size() == cnst.size();
		for (const RgfxConstantDecl &d : cnst)
			same = same && hasConstant(*air, d.name.c_str(), d.type, d.index, d.required);
		check(same, f.name + ": its module's constants are its CNST tag's");
	}
	return {compared, withConstants};
}

// The function `name`'s module in `lib`, as bytes: empty where the library has no bitcode for it.
std::vector<uint8_t> moduleBytes(const std::vector<uint8_t> &lib, const std::string &name)
{
	const uint8_t *bc = nullptr;
	size_t n = 0;
	if (!rgfxMetallibBitcode(lib.data(), lib.size(), name, &bc, &n))
		return {};
	return std::vector<uint8_t>(bc, bc + n);
}

// One read of each function's module answers what each of the three readers answers for it, and
// the process's cache hands out that answer. Answers how many modules it compared.
size_t moduleReadsAsReaders(const std::vector<uint8_t> &lib, const std::string &what)
{
	size_t compared = 0;
	for (const Listed &f : listed(lib)) {
		const std::vector<uint8_t> bc = moduleBytes(lib, f.name);
		if (bc.empty())
			continue;
		RgfxAirModule once;
		rgfxReadAirModule(bc.data(), bc.size(), once);
		std::vector<RgfxAirFunction> fns;
		std::vector<RgfxAirVisible> vis;
		RgfxAirEntries entries;
		const bool fnsRead = rgfxReadAirFunctions(bc.data(), bc.size(), fns);
		const bool visRead = rgfxReadAirVisible(bc.data(), bc.size(), vis);
		const bool entriesRead = rgfxReadAirEntries(bc.data(), bc.size(), entries);
		check(once.functionsRead == fnsRead && once.functions == fns,
		      what + " " + f.name + ": one read's arguments are rgfxReadAirFunctions'");
		check(once.visibleRead == visRead && once.visible == vis,
		      what + " " + f.name + ": one read's visible functions are rgfxReadAirVisible's");
		check(once.entriesRead == entriesRead && once.entries == entries,
		      what + " " + f.name + ": one read's nodes are rgfxReadAirEntries'");
		const std::shared_ptr<const RgfxAirModule> kept = rgfxAirModules().moduleOf(bc.data(), bc.size());
		check(kept != nullptr && kept->functionsRead == fnsRead && kept->functions == fns &&
		          kept->visibleRead == visRead && kept->visible == vis && kept->entriesRead == entriesRead &&
		          kept->entries == entries,
		      what + " " + f.name + ": the process's read is the same");
		compared++;
	}
	return compared;
}

// The .metallib files of the directory `dir`, by name.
std::vector<std::string> metallibsIn(const std::string &dir)
{
	std::vector<std::string> out;
	if (DIR *d = opendir(dir.c_str())) {
		while (const dirent *e = readdir(d)) {
			const std::string name = e->d_name;
			if (name.size() > 9 && name.compare(name.size() - 9, 9, ".metallib") == 0)
				out.push_back(dir + "/" + name);
		}
		closedir(d);
	}
	std::sort(out.begin(), out.end());
	return out;
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
	// A count function constants set: an argument of every index from its first, which writes
	// there; at any index too, at none (rgfxAirWritesTextureAnywhere).
	RgfxAirFunction counted;
	counted.arguments.push_back(
	    {RgfxAirKind::Texture, 3, 0, RgfxAirAccess::Write, false, false, true});
	check(!rgfxAirWritesTexture(counted, 2) && rgfxAirWritesTexture(counted, 3) &&
	          rgfxAirWritesTexture(counted, 40) && !rgfxAirWritesTextureAnywhere(counted),
	      "a written texture of any count from 3: written at 3 and up");
	counted.arguments[0].anyIndex = true;
	counted.arguments[0].index = 0;
	check(!rgfxAirWritesTexture(counted, 3) && rgfxAirWritesTextureAnywhere(counted),
	      "at any index too: written at none, anywhere");
	// No source of ours declares one (an `array_ref` binding), but Metal Performance Shaders'
	// kernels do: read from the OS's libraries, where it has them, never kept.
	{
		static const char *const kMps[] = {
		    "/System/Library/Frameworks/MetalPerformanceShaders.framework/Versions/A/Frameworks/"
		    "MPSCore.framework/Versions/A/Resources/default.metallib",
		    "/System/Library/Frameworks/MetalPerformanceShaders.framework/Versions/A/Frameworks/"
		    "MPSImage.framework/Versions/A/Resources/default.metallib",
		    "/System/Library/Frameworks/MetalPerformanceShaders.framework/Versions/A/Frameworks/"
		    "MPSNDArray.framework/Versions/A/Resources/default.metallib",
		};
		size_t libraries = 0, entries = 0, unread = 0, anyCount = 0;
		for (const char *mps : kMps) {
			const std::vector<uint8_t> os = readFile(mps);
			if (os.size() < 0x58)
				continue;
			libraries++;
			for (const Listed &f : listed(os)) {
				const uint8_t *bc = nullptr;
				size_t n = 0;
				if (f.type < 0 || f.type > 2 ||
				    !rgfxMetallibBitcode(os.data(), os.size(), f.name, &bc, &n))
					continue;
				entries++;
				std::vector<RgfxAirFunction> fns;
				if (!rgfxReadAirFunctions(bc, n, fns)) {
					unread++;
					continue;
				}
				for (const RgfxAirFunction &fn : fns)
					for (const RgfxAirArgument &a : fn.arguments)
						anyCount += a.anyCount ? 1 : 0;
			}
		}
		std::printf("air_arguments_test: the OS's MPS libraries: %zu, %zu entry points, %zu unread, "
		            "%zu arguments of any count\n",
		            libraries, entries, unread, anyCount);
		check(libraries == 0 || anyCount > 0,
		      "the OS's MPS kernels' arguments of any count are read");
	}
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
	const RgfxAirFunction patch = entry(lib, "reads_patch");
	check(patch.stage == RgfxAirStage::Vertex && patch.arguments.size() == 1,
	      "reads_patch: its texture alone, past its control points and patch input");
	expect(patch, RgfxAirKind::Texture, 3, 1, RgfxAirAccess::Sample, "a sampled texture at 3");
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

	// References, read as pointers, as arguments and as a struct's fields.
	const RgfxAirVisible sigReferences = visible(lib, "sig_references");
	const std::vector<RgfxAirType> &rf = sigReferences.argumentTypes;
	if (rf.size() == 5) {
		const RgfxAirType &r = rf[0];
		const RgfxAirType *transform = field(r, "transform", 16), *pair = field(r, "pair", 32),
		                  *rows = field(r, "rows", 40);
		const RgfxAirType *m = transform != nullptr ? elementOf(*transform, Kind::Pointer) : nullptr,
		                  *p = pair != nullptr ? elementOf(*pair, Kind::Pointer) : nullptr,
		                  *a = rows != nullptr ? elementOf(*rows, Kind::Pointer) : nullptr;
		const RgfxAirType *row = a != nullptr ? elementOf(*a, Kind::Array, 2) : nullptr;
		check(r.kind == Kind::Struct && r.name == "SigReferences" && r.size == 48 && r.fields.size() == 6 &&
		          m != nullptr && isMatrix(*m, Scalar::Float, 4, 4, 64) && m->addressSpace == 2 && p != nullptr &&
		          isPair(*p) && p->addressSpace == 1 && row != nullptr && isVector(*row, Scalar::Float, 3, 16),
		      "sig_references: SigReferences's reference fields, to float4x4, SigPair and float3[2]");
		const RgfxAirType *v = elementOf(rf[1], Kind::Pointer), *d = elementOf(rf[2], Kind::Pointer),
		                  *cp = elementOf(rf[3], Kind::Pointer), *mm = elementOf(rf[4], Kind::Pointer);
		check(v != nullptr && isVector(*v, Scalar::Float, 4, 16) && v->addressSpace == 0 && d != nullptr &&
		          isScalar(*d, Scalar::Float, 4) && d->addressSpace == 1 && cp != nullptr && isPair(*cp) &&
		          cp->addressSpace == 2 && mm != nullptr && isMatrix(*mm, Scalar::Float, 2, 2, 16),
		      "sig_references: references to float4, float, SigPair and float2x2, in their address spaces");
	} else {
		check(false, "sig_references: five arguments");
	}

	// Pointers in each address space, to pointers and to void; an rvalue reference.
	const RgfxAirVisible sigSpaces = visible(lib, "sig_address_spaces");
	const std::vector<RgfxAirType> &as = sigSpaces.argumentTypes;
	if (as.size() == 17) {
		const RgfxAirType *t = elementOf(as[0], Kind::Pointer), *r = elementOf(as[1], Kind::Pointer),
		                  *rv = elementOf(as[2], Kind::Pointer), *pp = elementOf(as[3], Kind::Pointer),
		                  *v = elementOf(as[5], Kind::Pointer), *cv = elementOf(as[6], Kind::Pointer),
		                  *pbf = elementOf(as[11], Kind::Pointer), *ppf = elementOf(as[12], Kind::Pointer);
		check(t != nullptr && isScalar(*t, Scalar::Float, 4) && t->addressSpace == 3 && r != nullptr &&
		          r->addressSpace == 3 && rv != nullptr && isScalar(*rv, Scalar::Float, 4) && rv->addressSpace == 0,
		      "sig_address_spaces: threadgroup pointers and references, an rvalue reference");
		const RgfxAirType *inner = pp != nullptr ? elementOf(*pp, Kind::Pointer) : nullptr;
		check(inner != nullptr && pp->addressSpace == 1 && isScalar(*inner, Scalar::Float, 4) &&
		          inner->addressSpace == 1,
		      "sig_address_spaces: a pointer to a pointer");
		check(v != nullptr && v->kind == Kind::Void && v->addressSpace == 1 && v->size == 0 && cv != nullptr &&
		          cv->kind == Kind::Void && cv->addressSpace == 2,
		      "sig_address_spaces: pointers to void, device and constant");
		check(pbf != nullptr && isScalar(*pbf, Scalar::BFloat, 2) && ppf != nullptr &&
		          isVector(*ppf, Scalar::Float, 3, 12, true),
		      "sig_address_spaces: pointers to bfloat and packed_float3");
	} else {
		check(false, "sig_address_spaces: seventeen arguments");
	}

	// Atomics: structs of Metal's library, a float's of no size and no field.
	const RgfxAirVisible sigAtomics = visible(lib, "sig_atomics");
	const std::vector<RgfxAirType> &at = sigAtomics.argumentTypes;
	if (at.size() == 5) {
		const RgfxAirType *a = elementOf(at[0], Kind::Pointer), *c = elementOf(at[2], Kind::Pointer),
		                  *w = elementOf(at[3], Kind::Pointer);
		check(a != nullptr && a->kind == Kind::Struct && a->name == "metal::_atomic<int>" && a->size == 4 &&
		          a->fields.size() == 1 && a->fields[0].offset == 0 && isScalar(a->fields[0].type, Scalar::Int, 4),
		      "sig_atomics: an atomic_int, a struct of one int");
		check(c != nullptr && c->kind == Kind::Struct && c->name == "metal::_atomic<float>" && c->size == 0 &&
		          c->fields.empty(),
		      "sig_atomics: an atomic_float, a struct of no size and no field");
		const RgfxAirType *cs = w != nullptr ? field(*w, "c", 8) : nullptr;
		const RgfxAirType *ce = cs != nullptr ? elementOf(*cs, Kind::Array, 2) : nullptr;
		check(w != nullptr && w->name == "SigAtomics" && w->fields.size() == 4 && ce != nullptr &&
		          ce->name == "metal::_atomic<unsigned int>",
		      "sig_atomics: SigAtomics, an array of atomics among its fields");
	} else {
		check(false, "sig_atomics: five arguments");
	}

	// Enums and bfloats.
	const RgfxAirVisible sigEnums = visible(lib, "sig_enums");
	const std::vector<RgfxAirType> &en = sigEnums.argumentTypes;
	const RgfxAirType &e8 = sigEnums.returnType;
	check(e8.kind == Kind::Enum && e8.name == "SigE8" && e8.scalar == Scalar::UChar && e8.size == 1,
	      "sig_enums returns an enum of uchar");
	if (en.size() == 8) {
		const RgfxAirType *pe = elementOf(en[3], Kind::Pointer);
		check(en[0].kind == Kind::Enum && en[0].name == "SigE" && en[0].scalar == Scalar::Int && en[0].size == 4 &&
		          en[2].kind == Kind::Struct && en[2].name == "SigEnums" && en[2].fields.size() == 8 &&
		          pe != nullptr && pe->kind == Kind::Enum,
		      "sig_enums: enums as arguments, fields and a pointee");
		check(isScalar(en[4], Scalar::BFloat, 2) && isVector(en[5], Scalar::BFloat, 2, 4) &&
		          isVector(en[6], Scalar::BFloat, 3, 8) && isVector(en[7], Scalar::BFloat, 4, 8),
		      "sig_enums: bfloat and its vectors");
	} else {
		check(false, "sig_enums: eight arguments");
	}

	// Function tables.
	const RgfxAirVisible sigTables = visible(lib, "sig_tables");
	const std::vector<RgfxAirType> &tb = sigTables.argumentTypes;
	if (tb.size() == 6) {
		const RgfxAirType *ts = field(tb[2], "ts", 16), *its = field(tb[2], "its", 32);
		const RgfxAirType *tse = ts != nullptr ? elementOf(*ts, Kind::Array, 2) : nullptr,
		                  *itse = its != nullptr ? elementOf(*its, Kind::Array, 2) : nullptr,
		                  *of = elementOf(tb[3], Kind::ArrayOf, 2), *pt = elementOf(tb[4], Kind::Pointer);
		check(tb[0].kind == Kind::VisibleFunctionTable && tb[1].kind == Kind::IntersectionFunctionTable,
		      "sig_tables: a visible and an intersection function table");
		check(tse != nullptr && tse->kind == Kind::VisibleFunctionTable && itse != nullptr &&
		          itse->kind == Kind::IntersectionFunctionTable && of != nullptr &&
		          of->kind == Kind::VisibleFunctionTable && pt != nullptr && pt->kind == Kind::VisibleFunctionTable,
		      "sig_tables: arrays of tables, a metal::array of them and a pointer to one");
	} else {
		check(false, "sig_tables: six arguments");
	}

	// Pointers to textures and samplers, in each address space.
	const RgfxAirVisible sigPointees = visible(lib, "sig_pointees");
	const std::vector<RgfxAirType> &pe = sigPointees.argumentTypes;
	if (pe.size() == 10) {
		const RgfxAirType *ptr = elementOf(pe[0], Kind::Pointer), *ptc = elementOf(pe[2], Kind::Pointer),
		                  *ptg = elementOf(pe[3], Kind::Pointer), *rt = elementOf(pe[4], Kind::Pointer),
		                  *ps = elementOf(pe[7], Kind::Pointer);
		check(ptr != nullptr && isTexture(*ptr, RgfxAirTextureShape::T2D, Scalar::Float, RgfxAirAccess::Read) &&
		          ptr->addressSpace == 1 && ptc != nullptr && ptc->addressSpace == 2 && ptg != nullptr &&
		          ptg->addressSpace == 3 && rt != nullptr && rt->addressSpace == 0,
		      "sig_pointees: textures pointed to in each address space");
		check(ps != nullptr && ps->kind == Kind::Sampler && ps->addressSpace == 2, "sig_pointees: a sampler pointed to");
	} else {
		check(false, "sig_pointees: ten arguments");
	}

	const RgfxAirVisible sigUnnamed = visible(lib, "sig_unnamed");
	check(sigUnnamed.signature == RgfxAirVisible::Signature::Read &&
	          sigUnnamed.argumentNames == std::vector<std::string>{"a", "", "b"} &&
	          sigUnnamed.argumentTypes.size() == 3 && isVector(sigUnnamed.argumentTypes[1], Scalar::Float, 4, 16),
	      "sig_unnamed: an unnamed argument, named empty");

	// An acceleration structure is read; Metal answers no reflection of it (RgfxReflection.h).
	const RgfxAirVisible sigAcceleration = visible(lib, "sig_acceleration");
	check(sigAcceleration.signature == RgfxAirVisible::Signature::Read &&
	          sigAcceleration.argumentTypes.size() == 2 &&
	          sigAcceleration.argumentTypes[0].kind == Kind::AccelerationStructure,
	      "sig_acceleration: an acceleration structure");

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

	// Each module's function constants: every_constant's five, of their types (MTLDataType) and
	// indices, the one it tests with is_function_constant_defined not required; every_access's
	// one; none for a function of none; the visible tinted's one.
	const RgfxAirFunction constants = entry(lib, "every_constant");
	check(constants.constantsRead && constants.constants.size() == 5 &&
	          hasConstant(constants.constants, "kTint", 6, 20, true) &&
	          hasConstant(constants.constants, "kSteps", 42, 21, true) &&
	          hasConstant(constants.constants, "kMask", 55, 22, true) &&
	          hasConstant(constants.constants, "kGain", 16, 23, true) &&
	          hasConstant(constants.constants, "kOptional", 29, 24, false),
	      "every_constant: its five constants");
	check(all.constantsRead && all.constants.size() == 1 &&
	          hasConstant(all.constants, "kScale", 53, 0, true),
	      "every_access: kScale");
	check(none.constantsRead && none.constants.empty(), "no_texture: no constants");
	const RgfxAirVisible tinted = visible(lib, "tinted");
	check(tinted.constantsRead && tinted.constants.size() == 1 &&
	          hasConstant(tinted.constants, "kTint", 6, 20, true),
	      "tinted: kTint");
	// Every function's, as its CNST tag lists them.
	const auto compared = constantsAsCnst(lib);
	check(compared.first >= 10 && compared.second >= 5,
	      "every function's constants compared (" + std::to_string(compared.first) + ", " +
	          std::to_string(compared.second) + " with constants)");

	// Modules built for macOS 10.12, of LLVM 3.5's metadata nodes: each entry point's arguments,
	// and its constants as its CNST tag lists them.
	{
		const char *oldPath = std::getenv("RGFX_AIR_ARGUMENTS_OLD");
		check(oldPath != nullptr, "RGFX_AIR_ARGUMENTS_OLD names the older test library");
		const std::vector<uint8_t> old = readFile(oldPath);
		check(old.size() > 0x58, "the older test library is read");
		const RgfxAirFunction oldAll = entry(old, "old_every_access");
		check(oldAll.stage == RgfxAirStage::Kernel && oldAll.arguments.size() == 6,
		      "old_every_access: its six binding arguments");
		expect(oldAll, RgfxAirKind::Texture, 0, 1, RgfxAirAccess::Read, "access::read at texture 0");
		expect(oldAll, RgfxAirKind::Texture, 1, 1, RgfxAirAccess::Sample, "a sampled texture at 1");
		expect(oldAll, RgfxAirKind::Texture, 2, 1, RgfxAirAccess::Write, "access::write at 2");
		expect(oldAll, RgfxAirKind::Buffer, 0, 1, RgfxAirAccess::ReadWrite, "a device buffer at 0");
		expect(oldAll, RgfxAirKind::Buffer, 1, 1, RgfxAirAccess::Read, "a constant at buffer 1");
		expect(oldAll, RgfxAirKind::Sampler, 0, 1, RgfxAirAccess::Unknown, "a sampler at 0");
		check(rgfxAirWritesTexture(oldAll, 2) && !rgfxAirWritesTexture(oldAll, 0) &&
		          !rgfxAirWritesTexture(oldAll, 1),
		      "old_every_access writes texture 2 alone");
		check(oldAll.constantsRead && oldAll.constants.size() == 2 &&
		          hasConstant(oldAll.constants, "kOldScale", 53, 0, true) &&
		          hasConstant(oldAll.constants, "kOldOptional", 29, 1, false),
		      "old_every_access: kOldScale, and kOldOptional not required");
		const RgfxAirFunction oldVertex = entry(old, "old_vertex");
		check(oldVertex.stage == RgfxAirStage::Vertex && oldVertex.arguments.size() == 1,
		      "old_vertex: a vertex function of one texture");
		expect(oldVertex, RgfxAirKind::Texture, 4, 1, RgfxAirAccess::Sample, "a sampled texture at 4");
		const RgfxAirFunction oldFragment = entry(old, "old_fragment");
		check(oldFragment.stage == RgfxAirStage::Fragment && rgfxAirWritesTexture(oldFragment, 5) &&
		          !rgfxAirWritesTexture(oldFragment, 0),
		      "old_fragment writes texture 5 alone");
		const auto oldCompared = constantsAsCnst(old);
		check(oldCompared.first == 3 && oldCompared.second == 1,
		      "the older functions' constants compared (" + std::to_string(oldCompared.first) +
		          ", " + std::to_string(oldCompared.second) + " with constants)");
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

	// Every node of an entry point whole, for its pipeline's reflection (rgfxReadAirEntries), over
	// tools/oracle/oracle_reflection.metal's library (RGFX_ORACLE_REFLECTION).
	const char *oraclePath = std::getenv("RGFX_ORACLE_REFLECTION");
	check(oraclePath != nullptr, "RGFX_ORACLE_REFLECTION names the oracle probe's library");
	const std::vector<uint8_t> oracle = readFile(oraclePath);
	auto entriesOf = [&](const std::string &name) {
		const uint8_t *bc = nullptr;
		size_t n = 0;
		RgfxAirEntries e;
		check(rgfxMetallibBitcode(oracle.data(), oracle.size(), name, &bc, &n) && rgfxReadAirEntries(bc, n, e),
		      name + ": its nodes are read");
		RgfxAirEntries one;
		for (const RgfxAirEntry &x : e.entries)
			if (x.name == name)
				one.entries.push_back(x);
		one.constants = e.constants;
		check(one.entries.size() == 1, name + ": its module declares it");
		return one;
	};
	{
		const RgfxAirEntries basic = entriesOf("k_basic");
		const RgfxAirEntry &k = basic.entries.at(0);
		check(k.stage == RgfxAirEntryStage::Kernel && k.arguments.size() == 11 && k.outputs.empty(),
		      "k_basic: a kernel of 11 nodes");
		const RgfxAirNode &params = k.arguments.at(1);
		check(params.kind == "air.buffer" && params.name == "params" && params.typeName == "Params" &&
		          params.hasLocation && params.index == 1 && params.count == 1 && params.addressSpace == 2 &&
		          params.typeSize == 192 && params.typeAlign == 16 && params.access == RgfxAirAccess::Read,
		      "k_basic: params, a constant Params at buffer 1");
		check(params.layout != nullptr && params.layout->size() == 12, "k_basic: Params's 12 members");
		if (params.layout != nullptr && params.layout->size() == 12) {
			const RgfxAirMember &counts = params.layout->at(6), &inner = params.layout->at(7);
			check(counts.name == "counts" && counts.typeName == "int" && counts.offset == 52 && counts.size == 4 &&
			          counts.arrayLength == 3 && counts.layout == nullptr,
			      "k_basic: counts, an int[3] at 52");
			check(inner.name == "inner" && inner.layout != nullptr && inner.layout->size() == 2 &&
			          inner.layout->at(1).typeName == "int2",
			      "k_basic: inner, an Inner of its own layout");
		}
		check(k.arguments.at(3).kind == "air.texture" && k.arguments.at(3).typeName == "texture2d<float, read>" &&
		          k.arguments.at(6).access == RgfxAirAccess::ReadWrite,
		      "k_basic: its textures' types and access");
		check(k.arguments.at(8).addressSpace == 3 && k.arguments.at(9).kind == "air.thread_position_in_grid" &&
		          k.arguments.at(9).typeName == "uint2" && !k.arguments.at(9).hasLocation,
		      "k_basic: threadgroup memory, then its built-ins");
		const RgfxAirEntry unused = entriesOf("k_unused").entries.at(0);
		check(!unused.arguments.at(0).unused && unused.arguments.at(1).unused && unused.arguments.at(5).unused &&
		          !unused.arguments.at(6).unused,
		      "k_unused: air.arg_unused on the arguments it never references");
		const RgfxAirEntries constants = entriesOf("k_constants");
		const RgfxAirEntry &kc = constants.entries.at(0);
		check(constants.constants.size() == 3 && constants.constants[0].first == "kOn" &&
		          constants.constants[2].first == "kIndex" && constants.constants[2].second == 2,
		      "k_constants: the module's three constants by index");
		check(!kc.arguments.at(0).predicated && kc.arguments.at(1).predicated && kc.arguments.at(1).predicate == "kOn" &&
		          kc.arguments.at(2).predicate == "kOff" && kc.arguments.at(5).indexIsGlobal &&
		          !kc.arguments.at(6).indexIsGlobal,
		      "k_constants: its predicates by constant, and the index kIndex computes");
		const RgfxAirEntry ab = entriesOf("k_argument_buffer").entries.at(0);
		const RgfxAirNode &members = ab.arguments.at(0);
		check(members.kind == "air.indirect_buffer" && members.layout != nullptr && members.layout->size() == 7,
		      "k_argument_buffer: an argument buffer of 7 members");
		if (members.layout != nullptr && members.layout->size() == 7) {
			const RgfxAirMember &textures = members.layout->at(4), &inner = members.layout->at(5);
			check(textures.indirect != nullptr && textures.indirect->kind == "air.texture" &&
			          textures.indirect->index == 4 && textures.indirect->count == 2,
			      "k_argument_buffer: an array of 2 textures at id 4");
			check(inner.indirect != nullptr && inner.indirect->kind == "air.buffer" && inner.indirect->index == 6 &&
			          inner.indirect->addressSpace == 2 && inner.indirect->layout != nullptr,
			      "k_argument_buffer: a constant Inner pointer at id 6");
			check(members.layout->at(3).indirect != nullptr &&
			          members.layout->at(3).indirect->kind == "air.indirect_constant",
			      "k_argument_buffer: a constant member");
		}
		const RgfxAirEntry vb = entriesOf("v_buffers").entries.at(0);
		check(vb.stage == RgfxAirEntryStage::Vertex && vb.outputs.size() == 2 && vb.outputs[0].kind == "air.position" &&
		          vb.outputs[0].typeName == "float4" && vb.outputs[1].kind == "air.vertex_output" &&
		          vb.outputs[1].typeName == "float2" && vb.outputs[1].name == "uv",
		      "v_buffers: its outputs, a position and uv");
		const RgfxAirEntry vs = entriesOf("v_stage_in").entries.at(0);
		check(vs.arguments.at(2).kind == "air.vertex_input" && vs.arguments.at(2).index == 2 && vs.arguments.at(2).unused,
		      "v_stage_in: an attribute it never reads");
		const RgfxAirEntry ft = entriesOf("f_targets").entries.at(0);
		check(ft.stage == RgfxAirEntryStage::Fragment && ft.outputs.size() == 2 &&
		          ft.outputs[1].kind == "air.render_target" && ft.outputs[1].typeName == "half4",
		      "f_targets: its render targets");
		const RgfxAirEntry mp = entriesOf("m_payload").entries.at(0);
		const RgfxAirNode &output = mp.arguments.at(0);
		check(mp.stage == RgfxAirEntryStage::Mesh && output.kind == "air.mesh" && output.hasMesh &&
		          output.meshVertices == 3 && output.meshPrimitives == 1 && output.meshTopology == "air.triangle" &&
		          output.meshVertexData.size() == 2 && output.meshVertexData[1].typeName == "float2" &&
		          output.meshPrimitiveData.empty(),
		      "m_payload: its mesh's data, counts and topology");
		check(mp.arguments.at(1).kind == "air.payload" && mp.arguments.at(1).layout != nullptr &&
		          entriesOf("o_payload").entries.at(0).stage == RgfxAirEntryStage::Object,
		      "o_payload and m_payload: their payload");
		check(entriesOf("k_nothing").entries.at(0).arguments.empty(), "k_nothing: no nodes");
		// A fragment function's reads of its colour attachments: nodes of kind air.render_target and
		// the attachment's index, of no binding.
		const RgfxAirEntry fetch = entriesOf("f_fetch").entries.at(0);
		check(fetch.arguments.size() == 4 && fetch.arguments.at(2).kind == "air.render_target" &&
		          fetch.arguments.at(2).hasRenderTarget && fetch.arguments.at(2).renderTarget == 0 &&
		          fetch.arguments.at(3).renderTarget == 1 && fetch.arguments.at(3).typeName == "half4" &&
		          !fetch.arguments.at(3).hasLocation,
		      "f_fetch: its colour attachments' reads");
		// An argument buffer's struct of members is its first id and its layout, the layout's ids its
		// own from 0; a C array of members is its node of count 1 and the member's array length.
		const RgfxAirEntry nested = entriesOf("k_nested").entries.at(0);
		const RgfxAirNode &nestedBuffer = nested.arguments.at(0);
		check(nestedBuffer.layout != nullptr && nestedBuffer.layout->size() == 9, "k_nested: 9 members");
		if (nestedBuffer.layout != nullptr && nestedBuffer.layout->size() == 9) {
			const RgfxAirMember &leaves = nestedBuffer.layout->at(2), &buffers = nestedBuffer.layout->at(4);
			check(leaves.hasFirstId && leaves.firstId == 3 && leaves.arrayLength == 2 && leaves.layout != nullptr &&
			          leaves.layout->size() == 2 && leaves.layout->at(1).indirect != nullptr &&
			          leaves.layout->at(1).indirect->index == 1,
			      "k_nested: leaves, two structs of members from id 3");
			check(buffers.indirect != nullptr && buffers.indirect->index == 9 && buffers.indirect->count == 1 &&
			          buffers.arrayLength == 2,
			      "k_nested: buffers, a C array of 2 at id 9");
			check(nestedBuffer.layout->at(6).indirect != nullptr &&
			          nestedBuffer.layout->at(6).indirect->kind == "air.indirect_buffer" &&
			          nestedBuffer.layout->at(6).indirect->layout != nullptr,
			      "k_nested: child, a pointer to an argument buffer of its own");
		}

		// Spoilt modules: none crashes the reader.
		const uint8_t *bc = nullptr;
		size_t n = 0;
		check(rgfxMetallibBitcode(oracle.data(), oracle.size(), "k_argument_buffer", &bc, &n), "a module to spoil");
		const std::vector<uint8_t> goodEntries(bc, bc + n);
		RgfxAirEntries spoiltEntries;
		read = 0;
		for (int round = 0; round < 4000; round++) {
			std::vector<uint8_t> spoilt = goodEntries;
			for (int k2 = 0; k2 < 1 + round % 4; k2++) {
				seed = seed * 1103515245u + 12345u;
				spoilt[(seed >> 8) % spoilt.size()] ^= static_cast<uint8_t>(1u << ((seed >> 4) & 7));
			}
			const size_t cut = round % 7 == 0 ? (seed >> 3) % spoilt.size() : spoilt.size();
			if (rgfxReadAirEntries(spoilt.data(), cut, spoiltEntries))
				read++;
		}
		check(true, "4000 spoilt modules' nodes read without a crash (" + std::to_string(read) + " answered)");
	}

	// One read of each module answers what the three readers answer for it, over every library
	// above, and so does the process's cache of modules.
	{
		const char *oldPath = std::getenv("RGFX_AIR_ARGUMENTS_OLD");
		const size_t compared = moduleReadsAsReaders(lib, "air_arguments") +
		                        moduleReadsAsReaders(readFile(oldPath), "air_arguments_old") +
		                        moduleReadsAsReaders(oracle, "oracle_reflection");
		std::printf("air_arguments_test: one read of each module: %zu modules compared\n", compared);
		check(compared >= 40, "every module compared (" + std::to_string(compared) + ")");
	}

	// The cache: equal bytes, at any address, are read once and share their read; it keeps the
	// most recently looked-up modules while their bytes fit its budget, reads one it let go again,
	// and keeps none larger than the budget. A read it let go stays valid while it is held.
	{
		std::vector<std::vector<uint8_t>> three = {moduleBytes(lib, "every_access"),
		                                           moduleBytes(lib, "reads_in_vertex"),
		                                           moduleBytes(lib, "writes_in_fragment")};
		std::sort(three.begin(), three.end(),
		          [](const std::vector<uint8_t> &x, const std::vector<uint8_t> &y) { return x.size() > y.size(); });
		const std::vector<uint8_t> &a = three[0], &b = three[1], &c = three[2]; // a the largest
		check(!c.empty() && a != b && b != c && a != c, "three modules to keep");
		const std::vector<uint8_t> a2 = a; // the same bytes at another address
		RgfxAirModuleCache cache(a.size() + b.size());
		const auto pa = cache.moduleOf(a.data(), a.size());
		check(pa != nullptr && cache.reads() == 1 && cache.kept() == 1 && cache.keptBytes() == a.size(),
		      "the first lookup reads the module, and keeps it");
		check(cache.moduleOf(a2.data(), a2.size()) == pa && cache.reads() == 1,
		      "equal bytes at another address: the same read, not read again");
		RgfxAirModule direct;
		rgfxReadAirModule(a.data(), a.size(), direct);
		check(pa->functionsRead == direct.functionsRead && pa->functions == direct.functions &&
		          pa->visibleRead == direct.visibleRead && pa->visible == direct.visible &&
		          pa->entriesRead == direct.entriesRead && pa->entries == direct.entries,
		      "the read kept is rgfxReadAirModule's");
		const auto pb = cache.moduleOf(b.data(), b.size());
		check(pb != pa && cache.reads() == 2 && cache.kept() == 2 && cache.keptBytes() == a.size() + b.size(),
		      "other bytes are read, and kept");
		check(cache.moduleOf(a.data(), a.size()) == pa && cache.reads() == 2, "a, looked up again");
		const auto pc = cache.moduleOf(c.data(), c.size());
		check(cache.reads() == 3 && cache.kept() == 2 && cache.keptBytes() == a.size() + c.size(),
		      "c past the budget lets b go, the least recently looked up");
		check(cache.moduleOf(c.data(), c.size()) == pc && cache.moduleOf(a.data(), a.size()) == pa &&
		          cache.reads() == 3,
		      "a and c are kept");
		const auto pb2 = cache.moduleOf(b.data(), b.size());
		check(pb2 != pb && cache.reads() == 4 && cache.kept() == 2 && cache.keptBytes() == a.size() + b.size(),
		      "b, let go, is read again, and lets c go");
		check(pb->functions == pb2->functions && pb->entries == pb2->entries && pb->visible == pb2->visible,
		      "a read let go stays valid while it is held, and is the same as the new one");
		RgfxAirModuleCache tiny(a.size() - 1);
		const auto big = tiny.moduleOf(a.data(), a.size());
		check(big != nullptr && big->functions == pa->functions && tiny.kept() == 0 && tiny.keptBytes() == 0,
		      "a module larger than the budget is read, not kept");
		check(tiny.moduleOf(a.data(), a.size()) != big && tiny.reads() == 2, "and is read at each lookup");
		// Threads looking up one module at once: every one is answered the read kept.
		RgfxAirModuleCache shared(1u << 20);
		std::vector<std::shared_ptr<const RgfxAirModule>> got(8);
		std::vector<std::thread> threads;
		for (size_t i = 0; i < got.size(); i++)
			threads.emplace_back([&, i] { got[i] = shared.moduleOf(a.data(), a.size()); });
		for (std::thread &t : threads)
			t.join();
		check(std::all_of(got.begin(), got.end(), [&](const auto &m) { return m != nullptr && m == got[0]; }) &&
		          shared.reads() >= 1 && shared.reads() <= got.size() && shared.kept() == 1 &&
		          shared.moduleOf(a2.data(), a2.size()) == got[0],
		      "threads looking up one module at once share one read");
		// The process's own, which every library reads through.
		const uint64_t before = rgfxAirModules().reads();
		const auto p1 = rgfxAirModules().moduleOf(a.data(), a.size());
		check(rgfxAirModules().moduleOf(a2.data(), a2.size()) == p1 && rgfxAirModules().reads() - before <= 1,
		      "the process's cache reads a module once");
	}

	// Every function of the repo's committed .metallibs (RGFX_COMMITTED_METALLIBS: the suite's
	// fixtures, and the Vulkan platform's metallibs, which Apple's compiler built from our own
	// sources) is found by the read the device makes for its kind: an entry point's arguments and
	// nodes, a visible function's signature, a mesh or object function's nodes. Core Image's
	// kernels (TYPE 5) and intersection functions get no read. kernel_trunc.metallib, which the
	// suite's build.sh cuts to 1000 bytes on purpose, has no bitcode for its function. The nodes
	// of the entry points below do not read: each has an argument of a kind the node reader does
	// not follow, a fragment function's framebuffer-fetch input (`[[color(0)]]`) or an explicit
	// imageblock, so their pipelines answer no reflection (RgfxPipelineReflectionOf.mm).
	{
		const std::vector<std::string> nodesUnread = {
		    "mutdec_imageblock.metallib mutdec_imageblock",
		    "rounding.metallib fetch_float_fragment",
		    "rounding.metallib fetch_half_fragment",
		    "rounding.metallib fetch_seen_fragment",
		    "rounding_safe.metallib fetch_float_fragment",
		    "rounding_safe.metallib fetch_half_fragment",
		    "rounding_safe.metallib fetch_seen_fragment",
		    "slice_write_repro.metallib slice_write",
		    "slice_write_repro.metallib slice_write_l0",
		};
		size_t unreadNodes = 0;
		const char *dirs = std::getenv("RGFX_COMMITTED_METALLIBS");
		check(dirs != nullptr, "RGFX_COMMITTED_METALLIBS names the committed metallibs' directories");
		const std::string list = dirs != nullptr ? dirs : "";
		size_t libraries = 0, functions = 0, notAsked = 0;
		for (size_t at = 0; at < list.size();) {
			size_t colon = list.find(':', at);
			if (colon == std::string::npos)
				colon = list.size();
			const std::string dir = list.substr(at, colon - at);
			at = colon + 1;
			const std::vector<std::string> files = metallibsIn(dir);
			check(!files.empty(), dir + ": holds .metallibs");
			for (const std::string &file : files) {
				const std::vector<uint8_t> bytes = readFile(file.c_str());
				const uint8_t *p = bytes.data();
				size_t n = bytes.size();
				rgfxMetallibAirSlice(p, n);
				const std::vector<uint8_t> one(p, p + n);
				const bool cut = file.size() >= 22 && file.compare(file.size() - 22, 22, "/kernel_trunc.metallib") == 0;
				const std::vector<Listed> fns = listed(one);
				check(!fns.empty(), file + ": its function list reads");
				libraries++;
				for (const Listed &f : fns) {
					functions++;
					const std::string what = file + ": " + f.name;
					const std::vector<uint8_t> bc = moduleBytes(one, f.name);
					if (cut) {
						check(bc.empty(), what + ": cut on purpose, no bitcode");
						continue;
					}
					check(!bc.empty(), what + ": its bitcode");
					auto named = [&](const auto &v) {
						return std::any_of(v.begin(), v.end(), [&](const auto &x) { return x.name == f.name; });
					};
					std::vector<RgfxAirFunction> args;
					std::vector<RgfxAirVisible> vis;
					RgfxAirEntries nodes;
					switch (f.type) {
					case 0:
					case 1:
					case 2: {
						check(rgfxReadAirFunctions(bc.data(), bc.size(), args) && named(args), what + ": its arguments");
						const std::string listedAs = file.substr(file.rfind('/') + 1) + " " + f.name;
						const bool unread =
						    std::find(nodesUnread.begin(), nodesUnread.end(), listedAs) != nodesUnread.end();
						const bool read = rgfxReadAirEntries(bc.data(), bc.size(), nodes) && named(nodes.entries);
						check(read != unread, what + (unread ? ": its nodes do not read" : ": its nodes"));
						unreadNodes += unread ? 1 : 0;
						break;
					}
					case 4: {
						check(rgfxReadAirVisible(bc.data(), bc.size(), vis), what + ": its module's visible functions");
						auto v = std::find_if(vis.begin(), vis.end(), [&](const RgfxAirVisible &x) { return x.name == f.name; });
						check(v != vis.end() && v->signature != RgfxAirVisible::Signature::Unknown, what + ": its signature");
						break;
					}
					case 7:
					case 8:
						check(rgfxReadAirEntries(bc.data(), bc.size(), nodes) && named(nodes.entries), what + ": its nodes");
						break;
					default:
						notAsked++;
						break;
					}
				}
			}
		}
		std::printf("air_arguments_test: the committed metallibs: %zu, %zu functions, %zu of a kind given "
		            "no read\n",
		            libraries, functions, notAsked);
		check(libraries >= 60, "the committed metallibs are read (" + std::to_string(libraries) + ")");
		check(unreadNodes == nodesUnread.size(), "each entry point listed as of unread nodes is met");
	}

	std::printf("air_arguments_test: OK (%d checks)\n", checks);
	return 0;
}
