// The AIR argument reader's test (src/RgfxAirArguments.*), over the .metallib Apple's Metal
// compiler builds from tests/air_arguments.metal at test time (RGFX_AIR_ARGUMENTS names it):
// every entry point's binding arguments, kind, index, count and access, as the source declares
// them, those under function constants among them; which texture bindings each may write; no
// answer for an entry point with a binding of a kind the reader does not know, beside the answers
// of the others of its module, over the compiler's module of every entry point of the same source
// (RGFX_AIR_MODULE, its .air), and the same of every node whole, an entry point of a node the reader
// does not follow listed beside the others; each reader's answer its own for each entry point,
// visible function and function constant of a module written here of malformed nodes beside
// well-formed ones; each stitchable function's signature,
// type by type, and an unknown one where it has a type the reader does not know; each module's
// function constants, as its function's CNST tag lists them; the same over the .metallib built
// from tests/air_arguments_old.metal for macOS 10.12 (RGFX_AIR_ARGUMENTS_OLD); each buffer's
// address space where a module built for macOS 11 names none (RGFX_AIR_ARGUMENTS_11, from
// tests/air_arguments_11.metal); every node of an
// entry point whole (rgfxReadAirEntries), over the oracle probe's library (RGFX_ORACLE_REFLECTION,
// built from tools/oracle/oracle_reflection.metal); no answer, nor a crash, for bytes that are
// not such a module; one read of each module answering what the three readers answer, and the
// process's cache of those reads (RgfxAirModuleCache); the index of a function list's bitcode
// answering as rgfxMetallibBitcode (RgfxMetallibBitcodeIndex); and every function of the repo's
// committed .metallibs found by the read the device makes for its kind (RGFX_COMMITTED_METALLIBS).
#include "RgfxAirArguments.hpp"
#include "RgfxFunctionConstants.hpp"

#include <dirent.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <set>
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
	return a.name == b.name && a.stage == b.stage && a.argumentsRead == b.argumentsRead && a.arguments == b.arguments &&
	       a.constants == b.constants && a.constantsRead == b.constantsRead;
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
	       a.predicateGlobal == b.predicateGlobal && a.countIsGlobal == b.countIsGlobal &&
	       a.indexGlobal == b.indexGlobal && a.countGlobal == b.countGlobal && a.hasRenderTarget == b.hasRenderTarget &&
	       a.renderTarget == b.renderTarget &&
	       a.hasMesh == b.hasMesh && a.meshVertexData == b.meshVertexData &&
	       a.meshPrimitiveData == b.meshPrimitiveData && a.meshVertices == b.meshVertices &&
	       a.meshPrimitives == b.meshPrimitives && a.meshTopology == b.meshTopology;
}

static bool operator==(const RgfxAirEntry &a, const RgfxAirEntry &b)
{
	return a.name == b.name && a.function == b.function && a.stage == b.stage && a.nodesRead == b.nodesRead &&
	       a.outputs == b.outputs && a.arguments == b.arguments && a.hasWorkGroupSize == b.hasWorkGroupSize &&
	       std::equal(a.workGroupSize, a.workGroupSize + 3, b.workGroupSize);
}

// Two modules' code, which is opaque, compared by what it answers: each entry point's
// specialisation by no constant values.
static bool sameCode(const RgfxAirEntries &a, const RgfxAirEntries &b)
{
	if ((a.code == nullptr) != (b.code == nullptr))
		return false;
	for (const RgfxAirEntry &e : a.entries) {
		RgfxAirSpecialisation x, y;
		if (rgfxAirSpecialise(a, e, nullptr, 0, x) != rgfxAirSpecialise(b, e, nullptr, 0, y) ||
		    x.globals != y.globals || x.bodyRead != y.bodyRead || x.used != y.used)
			return false;
	}
	return true;
}

static bool operator==(const RgfxAirEntries &a, const RgfxAirEntries &b)
{
	return a.entries == b.entries && a.constants == b.constants && sameCode(a, b);
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
	check(fns.empty() || fns[0].argumentsRead, name + ": its arguments are read");
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

// The index of a library's function list answers what rgfxMetallibBitcode answers, for every name
// in `names`, over the bytes it walked and over a copy of them at another address. Answers how
// many it found.
size_t indexAsWalk(const std::vector<uint8_t> &lib, const std::vector<std::string> &names,
                   const std::string &what)
{
	const RgfxMetallibBitcodeIndex index(lib.data(), lib.size());
	const std::vector<uint8_t> copy = lib;
	size_t found = 0;
	for (const std::string &name : names) {
		for (const std::vector<uint8_t> *bytes : {&lib, &copy}) {
			const uint8_t *walkedAt = nullptr, *indexedAt = nullptr;
			size_t walkedSize = 0, indexedSize = 0;
			const bool walked = rgfxMetallibBitcode(bytes->data(), bytes->size(), name, &walkedAt, &walkedSize);
			const bool indexed = index.find(bytes->data(), bytes->size(), name, &indexedAt, &indexedSize);
			check(walked == indexed && (!walked || (walkedAt == indexedAt && walkedSize == indexedSize)),
			      what + " " + name + ": the index answers as the walk");
			found += indexed ? 1 : 0;
		}
	}
	return found;
}

// The names `lib`'s function list holds, then one it does not and the empty one.
std::vector<std::string> namesOf(const std::vector<uint8_t> &lib)
{
	std::vector<std::string> names;
	for (const Listed &f : listed(lib))
		names.push_back(f.name);
	names.push_back("no_such_function");
	names.push_back("");
	return names;
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

// LLVM's bitstream (its documented container), written: unabbreviated records only, each block's
// abbreviations 4 bits wide.
class Bitstream {
public:
	Bitstream() : bytes_{'B', 'C', 0xC0, 0xDE}, bit_(32) {}
	void enter(unsigned block)
	{
		put(1, width_); // ENTER_SUBBLOCK
		vbr(block, 8);
		vbr(4, 4);
		align();
		open_.push_back(Open{width_, bit_});
		put(0, 32); // the block's length in words, set at leave()
		width_ = 4;
	}
	void leave()
	{
		put(0, width_); // END_BLOCK
		align();
		const Open o = open_.back();
		open_.pop_back();
		const uint32_t words = static_cast<uint32_t>((bit_ - o.at - 32) / 32);
		for (size_t i = 0; i < 4; i++)
			bytes_[o.at / 8 + i] = static_cast<uint8_t>(words >> (8 * i));
		width_ = o.width;
	}
	void record(unsigned code, const std::vector<uint64_t> &ops)
	{
		put(3, width_); // UNABBREV_RECORD
		vbr(code, 6);
		vbr(ops.size(), 6);
		for (uint64_t op : ops)
			vbr(op, 6);
	}
	const std::vector<uint8_t> &bytes() const { return bytes_; }

private:
	struct Open {
		unsigned width;
		size_t at;
	};
	void put(uint64_t v, unsigned n)
	{
		for (unsigned i = 0; i < n; i++, bit_++) {
			if (bit_ / 8 >= bytes_.size())
				bytes_.push_back(0);
			if ((v >> i) & 1)
				bytes_[bit_ / 8] |= static_cast<uint8_t>(1u << (bit_ % 8));
		}
	}
	void vbr(uint64_t v, unsigned n)
	{
		const uint64_t more = 1ull << (n - 1);
		do {
			const uint64_t piece = v & (more - 1);
			v >>= n - 1;
			put(piece | (v != 0 ? more : 0), n);
		} while (v != 0);
	}
	void align()
	{
		while (bit_ % 32 != 0)
			put(0, 1);
	}
	std::vector<uint8_t> bytes_;
	size_t bit_;
	unsigned width_ = 2;
	std::vector<Open> open_;
};

// A module of our own, written as LLVM writes a module of version 1 (its functions' names in its
// value symbol table): its global variables (constants of `i32 addrspace(2)*`, value numbers from
// 0), its functions (declarations of `void ()`, after them), its i32 constants (after those), and
// its metadata, each string, value and node a metadata of its own, in its order (METADATA_STRING_OLD,
// METADATA_VALUE and METADATA_NODE records), and its named metadata.
struct WrittenModule {
	size_t globals = 0;
	std::vector<std::string> functions; // set before any metadata
	std::vector<int64_t> integers;
	struct Md {
		int kind; // 0 a string, 1 a value, 2 a node
		std::string string;
		uint64_t value;
		std::vector<uint64_t> ops; // a node's metadata, UINT64_MAX for none
	};
	std::vector<Md> md;
	std::vector<std::pair<std::string, std::vector<uint64_t>>> named;

	uint64_t str(const std::string &s)
	{
		md.push_back(Md{0, s, 0, {}});
		return md.size() - 1;
	}
	uint64_t value(uint64_t v)
	{
		md.push_back(Md{1, {}, v, {}});
		return md.size() - 1;
	}
	uint64_t i32(int64_t x)
	{
		integers.push_back(x);
		return value(globals + functions.size() + integers.size() - 1);
	}
	uint64_t global(size_t k) { return value(k); }
	uint64_t fn(const std::string &name)
	{
		const size_t k = static_cast<size_t>(std::find(functions.begin(), functions.end(), name) - functions.begin());
		return value(globals + k);
	}
	uint64_t node(const std::vector<uint64_t> &ops)
	{
		md.push_back(Md{2, {}, 0, ops});
		return md.size() - 1;
	}
	std::vector<uint8_t> write() const
	{
		auto chars = [](uint64_t first, const std::string &s) {
			std::vector<uint64_t> ops;
			if (first != UINT64_MAX)
				ops.push_back(first);
			for (char c : s)
				ops.push_back(static_cast<uint8_t>(c));
			return ops;
		};
		Bitstream b;
		b.enter(8);       // MODULE_BLOCK
		b.record(1, {1}); // VERSION
		b.enter(17);      // TYPE_BLOCK_NEW: void, void (), i32, i32 addrspace(2)*, metadata
		b.record(1, {5});
		b.record(2, {});
		b.record(21, {0, 0});
		b.record(7, {32});
		b.record(8, {2, 2});
		b.record(16, {});
		b.leave();
		for (size_t k = 0; k < globals; k++)
			b.record(7, {3, 1, 0, 0, 0, 0}); // GLOBALVAR: a constant, of no initialiser
		for (size_t k = 0; k < functions.size(); k++)
			b.record(8, {1, 0, 1, 0, 0, 0, 0, 0, 0}); // FUNCTION: a declaration
		b.enter(11); // CONSTANTS_BLOCK
		b.record(1, {2});
		for (int64_t x : integers)
			b.record(4, {x >= 0 ? static_cast<uint64_t>(x) << 1 : (static_cast<uint64_t>(-x) << 1) | 1});
		b.leave();
		b.enter(14); // VALUE_SYMTAB_BLOCK
		for (size_t k = 0; k < functions.size(); k++)
			b.record(1, chars(globals + k, functions[k]));
		b.leave();
		b.enter(15); // METADATA_BLOCK
		for (const Md &d : md) {
			if (d.kind == 0) {
				b.record(1, chars(UINT64_MAX, d.string));
			} else if (d.kind == 1) {
				b.record(2, {2, d.value});
			} else {
				std::vector<uint64_t> ops;
				for (uint64_t op : d.ops)
					ops.push_back(op == UINT64_MAX ? 0 : op + 1);
				b.record(3, ops);
			}
		}
		for (const auto &[name, ids] : named) {
			b.record(4, chars(UINT64_MAX, name));
			b.record(10, ids);
		}
		b.leave();
		b.leave();
		return b.bytes();
	}
};

// Each reader's answers over a module written here, of malformed nodes beside well-formed ones:
// the entry points' arguments (rgfxReadAirFunctions), their nodes whole (rgfxReadAirEntries) and
// the visible functions' signatures (rgfxReadAirVisible) are each one's own, one whose node names
// no function of the module left out; a layout read past the reader's depth from one entry point
// reads from another's nearer node; and a function constant whose node does not read is left out
// of the module's constants and of its code's, the others kept. readConstants reads the module's
// constants whole, so its answer is unread for every function where one does not read.
void checkWrittenModule()
{
	WrittenModule w;
	w.globals = 3;
	w.functions = {"k_good",       "k_bad_arg", "k_bad_wgs", "k_deep", "k_shallow", "k_after",
	               "v_bad_outputs", "v_good",    "f_one",     "f_short", "f_two"};
	const uint64_t none = w.node({});
	auto buffer = [&](int64_t position, int64_t index) {
		return std::vector<uint64_t>{w.i32(position), w.str("air.buffer"), w.str("air.location_index"), w.i32(index),
		                             w.i32(1), w.str("air.read")};
	};
	auto entryOf = [&](const std::string &name, uint64_t outputs, const std::vector<uint64_t> &args,
	                   std::vector<uint64_t> extra = {}) {
		std::vector<uint64_t> ops = {w.fn(name), outputs, w.node(args)};
		ops.insert(ops.end(), extra.begin(), extra.end());
		return w.node(ops);
	};
	// A chain of nine layouts, each a member of the next's type: the ninth past the depth the reader
	// follows from an argument of the first, one deep from an argument of its own.
	uint64_t layout = w.node({w.i32(0), w.i32(4), w.i32(0), w.str("T"), w.str("m")});
	const uint64_t ninth = layout;
	for (int k = 0; k < 8; k++)
		layout = w.node({w.str("air.struct_type_info"), layout, w.i32(0), w.i32(4), w.i32(0), w.str("T"), w.str("m")});
	std::vector<uint64_t> deep = buffer(0, 1), shallow = buffer(0, 2);
	deep.insert(deep.end(), {w.str("air.struct_type_info"), layout});
	shallow.insert(shallow.end(), {w.str("air.struct_type_info"), ninth});
	std::vector<uint64_t> underBad = {w.i32(1), w.str("air.function_constant"),
	                                  w.node({w.global(1), w.str("uint"), w.str("kBad")}), w.str("air.buffer"),
	                                  w.str("air.location_index"), w.i32(5), w.i32(1)};
	w.named.push_back({"air.kernel",
	                   {entryOf("k_good", none, {w.node(buffer(0, 0))}),
	                    entryOf("k_bad_arg", none, {w.node({w.i32(0), w.str("air.buffer"), w.i32(7)})}),
	                    w.node({w.str("k_none"), none, none}),
	                    entryOf("k_bad_wgs", none, {}, {w.node({w.str("air.work_group_size"), w.i32(1), w.i32(2)})}),
	                    entryOf("k_deep", none, {w.node(deep)}), entryOf("k_shallow", none, {w.node(shallow)}),
	                    entryOf("k_after", none, {w.node(underBad)},
	                            {w.node({w.str("air.work_group_size"), w.i32(8), w.i32(4), w.i32(1)})})}});
	w.named.push_back(
	    {"air.vertex",
	     {entryOf("v_bad_outputs", w.node({w.node({w.i32(5)})}), {}),
	      entryOf("v_good",
	              w.node({w.node({w.str("air.position"), w.str("air.arg_type_name"), w.str("float4"),
	                              w.str("air.arg_name"), w.str("p")})}),
	              {})}});
	w.named.push_back({"air.visible",
	                   {w.node({w.fn("f_one"), none, none}), w.node({w.str("f_none"), none, none}),
	                    w.node({w.fn("f_short")}), w.node({w.fn("f_two"), none, none})}});
	// kBad's index is a string; kOdd's type is one readConstants does not know, and rgfxReadAirEntries
	// does not read.
	w.named.push_back({"air.function_constants",
	                   {w.node({w.global(0), w.str("uint"), w.str("kGood"), w.i32(3)}),
	                    w.node({w.global(1), w.str("uint"), w.str("kBad"), w.str("three")}),
	                    w.node({w.global(2), w.str("bfloat"), w.str("kOdd"), w.i32(4)})}});
	const std::vector<uint8_t> bytes = w.write();

	std::vector<RgfxAirFunction> fns;
	check(rgfxReadAirFunctions(bytes.data(), bytes.size(), fns), "the written module's functions read");
	std::vector<std::string> names;
	for (const RgfxAirFunction &f : fns)
		names.push_back(f.name);
	check(names == std::vector<std::string>{"k_good", "k_bad_arg", "k_bad_wgs", "k_deep", "k_shallow", "k_after",
	                                        "v_bad_outputs", "v_good"},
	      "the written module's functions: each entry point that names a function");
	for (const RgfxAirFunction &f : fns) {
		check(!f.constantsRead && f.constants.empty(), f.name + ": its module's constants, one unread, are unread");
		check(f.argumentsRead == (f.name != "k_bad_arg"), f.name + ": its own arguments' answer");
	}
	check(fns.size() == 8 && fns[0].arguments.size() == 1 && fns[0].arguments[0].kind == RgfxAirKind::Buffer &&
	          fns[0].arguments[0].index == 0 && fns[0].arguments[0].access == RgfxAirAccess::Read,
	      "k_good: its buffer");
	check(fns.size() == 8 && fns[5].arguments.size() == 1 && fns[5].arguments[0].index == 5,
	      "k_after: its buffer under kBad, read whatever the constant's value");

	RgfxAirEntries e;
	check(rgfxReadAirEntries(bytes.data(), bytes.size(), e) && e.entries.size() == 8,
	      "the written module's nodes read, each entry point that names a function listed");
	const struct {
		const char *name;
		bool read;
	} kEntries[] = {{"k_good", true},    {"k_bad_arg", false},    {"k_bad_wgs", false}, {"k_deep", false},
	                {"k_shallow", true}, {"k_after", true}, {"v_bad_outputs", false}, {"v_good", true}};
	for (size_t i = 0; i < e.entries.size() && i < 8; i++) {
		const RgfxAirEntry &x = e.entries[i];
		check(x.name == kEntries[i].name && x.nodesRead == kEntries[i].read && x.function == w.globals + i,
		      x.name + ": its own nodes' answer");
		if (!x.nodesRead)
			check(x.arguments.empty() && x.outputs.empty() && !x.hasWorkGroupSize, x.name + ": no nodes");
	}
	if (e.entries.size() == 8) {
		const RgfxAirEntry &good = e.entries[0], &shallowEntry = e.entries[4], &after = e.entries[5],
		                   &vertex = e.entries[7];
		check(good.arguments.size() == 1 && good.arguments[0].kind == "air.buffer" && good.arguments[0].index == 0,
		      "k_good: its buffer's node");
		check(shallowEntry.arguments.size() == 1 && shallowEntry.arguments[0].layout != nullptr &&
		          shallowEntry.arguments[0].layout->size() == 1 &&
		          (*shallowEntry.arguments[0].layout)[0].typeName == "T" &&
		          (*shallowEntry.arguments[0].layout)[0].size == 4,
		      "k_shallow: the ninth layout, read from its own node after k_deep did not read it");
		check(after.hasWorkGroupSize && after.workGroupSize[0] == 8 && after.workGroupSize[1] == 4 &&
		          after.workGroupSize[2] == 1 && after.arguments.size() == 1 && after.arguments[0].predicated &&
		          after.arguments[0].predicate == "kBad" && after.arguments[0].predicateGlobal == 1,
		      "k_after: its work group size and its buffer under kBad");
		check(vertex.outputs.size() == 1 && vertex.outputs[0].kind == "air.position" &&
		          vertex.outputs[0].typeName == "float4" && vertex.outputs[0].name == "p",
		      "v_good: its output");
	}
	check(e.constants == std::vector<std::pair<std::string, uint32_t>>{{"kGood", 3}, {"kOdd", 4}},
	      "the written module's constants: each whose node reads");
	RgfxAirSpecialisation spec;
	check(e.code != nullptr && !e.entries.empty() && rgfxAirSpecialise(e, e.entries[0], nullptr, 0, spec),
	      "the written module's code, beside the constant whose node does not read");

	std::vector<RgfxAirVisible> vis;
	check(rgfxReadAirVisible(bytes.data(), bytes.size(), vis) && vis.size() == 3, "the written module's visible functions");
	if (vis.size() == 3)
		check(vis[0].name == "f_one" && vis[0].signature == RgfxAirVisible::Signature::None && vis[1].name == "f_short" &&
		          vis[1].signature == RgfxAirVisible::Signature::Unknown && vis[2].name == "f_two" &&
		          vis[2].signature == RgfxAirVisible::Signature::None && !vis[0].constantsRead,
		      "the visible functions: each one's own signature, f_none left out");

	// One read of it answers what the three readers answer.
	RgfxAirModule once;
	rgfxReadAirModule(bytes.data(), bytes.size(), once);
	check(once.functionsRead && once.functions == fns && once.visibleRead && once.visible == vis && once.entriesRead &&
	          once.entries == e,
	      "one read of the written module answers what the readers answer");
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
				for (const RgfxAirFunction &fn : fns) {
					unread += fn.argumentsRead ? 0 : 1;
					for (const RgfxAirArgument &a : fn.arguments)
						anyCount += a.anyCount ? 1 : 0;
				}
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

	// A binding of a kind the reader does not know may be a texture: the function has no answer,
	// and its module still reads.
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
		          rgfxReadAirFunctions(bc, n, fns) && fns.size() == 1 && fns[0].name == "calls_through_table" &&
		          !fns[0].argumentsRead && fns[0].arguments.empty(),
		      "calls_through_table, its table's kind unknown: listed, its arguments unread");
		check(rgfxMetallibBitcode(other.data(), other.size(), "conditional_planes", &bc, &n) &&
		          rgfxReadAirFunctions(bc, n, fns) && fns.size() == 1 && fns[0].argumentsRead,
		      "conditional_planes is still read beside it");
	}

	// A module of every entry point of tests/air_arguments.metal (the compiler's .air, before
	// `metallib` splits it into one module a function; RGFX_AIR_MODULE): each entry point's answer
	// is the one its own module gives, and one whose arguments do not read (its function table's
	// kind spoilt) leaves the others' answers as they were.
	if (const char *modulePath = std::getenv("RGFX_AIR_MODULE")) {
		const std::vector<uint8_t> whole = readFile(modulePath);
		std::vector<RgfxAirFunction> all;
		check(rgfxReadAirFunctions(whole.data(), whole.size(), all), "the whole module reads");
		size_t compared = 0;
		for (const RgfxAirFunction &f : all) {
			const RgfxAirFunction own = entry(lib, f.name);
			check(f.argumentsRead && own.stage == f.stage && own.arguments == f.arguments,
			      f.name + ": the whole module's answer is its own module's");
			compared++;
		}
		check(compared >= 12, "the whole module lists every entry point (" + std::to_string(compared) + ")");
		// Each string that starts with the kind's spoilt: the kind's, and the stitching type's
		// (`air.visible_function_table_type`), which only visible functions' signatures name.
		std::vector<uint8_t> spoilt = whole;
		const char tableKind[] = "air.visible_function_table";
		const size_t length = sizeof(tableKind) - 1;
		size_t spoiled = 0;
		for (auto it = spoilt.begin(); (it = std::search(it, spoilt.end(), tableKind, tableKind + length)) != spoilt.end();
		     it += length) {
			*(it + length - 1) = '_';
			spoiled++;
		}
		check(spoiled > 0, "the whole module names the function table's kind");
		std::vector<RgfxAirFunction> left;
		check(rgfxReadAirFunctions(spoilt.data(), spoilt.size(), left) && left.size() == all.size(),
		      "the spoilt module reads, every entry point listed");
		for (size_t i = 0; i < left.size() && i < all.size(); i++) {
			if (left[i].name == "calls_through_table")
				check(!left[i].argumentsRead && left[i].arguments.empty(),
				      "calls_through_table, its table's kind unknown: its arguments unread");
			else
				check(left[i] == all[i], left[i].name + ": its answer beside it is as it was");
		}

		// The same of the nodes whole (rgfxReadAirEntries). The whole module's nodes read, each entry
		// point's but reads_patch's: a post-tessellation vertex function's control point input
		// (`air.patch_control_point_input`) takes two nodes, its control point function's and its
		// attributes', operands no key the reader follows takes, so that entry point is listed with
		// no nodes and the others' read. Then, the key ahead of the kind of each argument declared
		// under a function constant spoilt (`air.function_constant`, so the constant's node is an
		// operand no key takes), the entry points of such an argument are listed with no nodes, and
		// every other entry point's nodes are as they were.
		RgfxAirEntries entriesAll, entriesLeft;
		check(rgfxReadAirEntries(whole.data(), whole.size(), entriesAll), "the whole module's nodes read");
		for (const RgfxAirEntry &e : entriesAll.entries)
			check(e.nodesRead == (e.name != "reads_patch"),
			      e.name + (e.name == "reads_patch" ? ": listed, its control point input's nodes unread"
			                                        : ": its nodes read"));
		check(std::any_of(entriesAll.entries.begin(), entriesAll.entries.end(),
		                  [](const RgfxAirEntry &e) { return e.name == "reads_patch"; }),
		      "the whole module lists reads_patch");
		std::vector<uint8_t> unkeyed = whole;
		const char fcKey[] = "air.function_constant";
		const size_t fcLength = sizeof(fcKey) - 1;
		size_t unkeyedCount = 0;
		for (auto it = unkeyed.begin(); (it = std::search(it, unkeyed.end(), fcKey, fcKey + fcLength)) != unkeyed.end();
		     it += fcLength) {
			*(it + fcLength - 1) = '_';
			unkeyedCount++;
		}
		check(unkeyedCount > 0, "the whole module names the function constants' key");
		check(rgfxReadAirEntries(unkeyed.data(), unkeyed.size(), entriesLeft) &&
		          entriesLeft.entries.size() == entriesAll.entries.size(),
		      "the module of the key spoilt reads, every entry point listed");
		std::vector<RgfxAirFunction> functionsLeft;
		check(rgfxReadAirFunctions(unkeyed.data(), unkeyed.size(), functionsLeft) && functionsLeft.size() == all.size(),
		      "the module of the key spoilt reads its functions, every entry point listed");
		std::set<std::string> predicated;
		for (size_t i = 0; i < entriesLeft.entries.size() && i < entriesAll.entries.size(); i++) {
			const RgfxAirEntry &was = entriesAll.entries[i], &now = entriesLeft.entries[i];
			const bool under = std::any_of(was.arguments.begin(), was.arguments.end(),
			                               [](const RgfxAirNode &n) { return n.predicated; });
			if (under) {
				predicated.insert(was.name);
				check(now.name == was.name && now.function == was.function && now.stage == was.stage &&
				          !now.nodesRead && now.arguments.empty() && now.outputs.empty() && !now.hasWorkGroupSize,
				      was.name + ", its function constant's key unknown: listed, its nodes unread");
			} else {
				check(now == was, was.name + ": its nodes beside it are as they were");
			}
		}
		check(predicated.size() >= 3 && predicated.size() < entriesAll.entries.size(),
		      "some entry points declare an argument under a function constant (" +
		          std::to_string(predicated.size()) + ")");
		for (size_t i = 0; i < functionsLeft.size() && i < all.size(); i++) {
			if (predicated.count(all[i].name) != 0)
				check(!functionsLeft[i].argumentsRead && functionsLeft[i].arguments.empty(),
				      all[i].name + ", its function constant's key unknown: its arguments unread");
			else
				check(functionsLeft[i].argumentsRead && functionsLeft[i].arguments == all[i].arguments,
				      all[i].name + ": its arguments beside it are as they were");
		}
	} else {
		std::printf("air_arguments_test: RGFX_AIR_MODULE unset: no module of several entry points\n");
	}

	// Malformed nodes of a module written here (LLVM's bitstream of a module of our own, of the
	// nodes Apple's compiler never writes so): each entry point's, visible function's and function
	// constant's answer its own.
	checkWrittenModule();

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

	// A module built for macOS 11 (tests/air_arguments_11.metal, RGFX_AIR_ARGUMENTS_11) names no
	// air.address_space on its buffers, as the compiler of macOS 11 and 12 writes them: each
	// buffer's address space is its parameter's (device 1, constant 2, threadgroup 3), and nothing
	// else of a node takes one.
	{
		const char *path11 = std::getenv("RGFX_AIR_ARGUMENTS_11");
		check(path11 != nullptr, "RGFX_AIR_ARGUMENTS_11 names the macOS 11 test library");
		const std::vector<uint8_t> lib11 = readFile(path11);
		auto entry11 = [&](const std::string &name) {
			const uint8_t *bc = nullptr;
			size_t n = 0;
			RgfxAirEntries e;
			const bool read = rgfxMetallibBitcode(lib11.data(), lib11.size(), name, &bc, &n) &&
			                  rgfxReadAirEntries(bc, n, e);
			check(read, name + ": its nodes are read");
			const std::string text(reinterpret_cast<const char *>(bc), read ? n : 0);
			check(read && text.find("air.address_space") == std::string::npos &&
			          text.find("air.arg_type_size") != std::string::npos,
			      name + ": its module names no address space, and names its pointees' sizes");
			for (const RgfxAirEntry &x : e.entries)
				if (x.name == name) {
					check(x.nodesRead, name + ": its own nodes are read");
					return x;
				}
			check(false, name + ": its module declares it");
			return RgfxAirEntry{};
		};
		auto space = [](const RgfxAirEntry &e, const std::string &arg) -> int {
			for (const RgfxAirNode &n : e.arguments)
				if (n.name == arg)
					return n.hasAddressSpace ? static_cast<int>(n.addressSpace) : -1;
			return -2;
		};
		const RgfxAirEntry k = entry11("k_11");
		check(space(k, "out") == 1 && space(k, "params") == 2 && space(k, "tg") == 3,
		      "k_11: a device buffer, a constant struct and threadgroup memory, of their parameters' spaces");
		check(space(k, "tex") == -1 && space(k, "smp") == -1 && space(k, "gid") == -1,
		      "k_11: its texture, sampler and built-in of no address space");
		const RgfxAirEntry v = entry11("v_11");
		check(space(v, "mvp") == 2 && space(v, "pos") == -1, "v_11: a constant matrix; its stage input of none");
		const RgfxAirEntry f = entry11("f_11");
		check(space(f, "tint") == 2 && space(f, "color") == -1, "f_11: a constant colour; its texture of none");
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
		one.code = e.code;
		check(one.entries.size() == 1, name + ": its module declares it");
		check(one.entries.size() == 1 && one.entries[0].nodesRead, name + ": its own nodes are read");
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

		// The specialisation of k_specialised by its constants' values (RgfxAirArguments.hpp):
		// Metal's answers, measured on both x86 Macs (tools/pipeline_reflection_check's
		// c_specialised_on and c_specialised_off).
		const RgfxAirEntries specialised = entriesOf("k_specialised");
		check(specialised.code != nullptr, "k_specialised: its module's code is read");
		auto value = [](uint32_t index, uint32_t type, int32_t v) {
			RgfxFunctionConstant c = {};
			c.index = index;
			c.data_type = type;
			if (type == 53) // MTLDataTypeBool
				c.value[0] = static_cast<uint8_t>(v != 0);
			else
				std::memcpy(c.value, &v, sizeof v);
			return c;
		};
		auto usedOf = [&](const RgfxAirEntries &e, const std::vector<RgfxFunctionConstant> &values,
		                  RgfxAirSpecialisation &spec) {
			std::string out;
			check(rgfxAirSpecialise(e, e.entries.at(0), values.data(), static_cast<uint32_t>(values.size()), spec) &&
			          spec.bodyRead,
			      e.entries.at(0).name + ": its body is followed");
			for (size_t i = 0; i < spec.used.size(); i++)
				out += spec.used[i] ? "1" : "0";
			return out;
		};
		const RgfxAirEntry &ks = specialised.entries.at(0);
		const std::vector<RgfxFunctionConstant> on = {value(0, 53, 1), value(1, 53, 0), value(2, 29, 5),
		                                              value(3, 29, 1)};
		const std::vector<RgfxFunctionConstant> off = {value(0, 53, 0), value(1, 53, 0), value(2, 29, 7),
		                                               value(3, 29, 2), value(4, 53, 1)};
		RgfxAirSpecialisation specOn, specOff;
		// out, plus, either, modeOne, optional, modeTwo, smp, gid.
		check(usedOf(specialised, on, specOn) == "11110001", "k_specialised under kMode 1: what it uses");
		check(usedOf(specialised, off, specOff) == "11001111", "k_specialised under kMode 2 and kOpt: what it uses");
		auto global = [](const RgfxAirSpecialisation &spec, uint64_t g) {
			auto it = spec.globals.find(g);
			return it == spec.globals.end() ? INT64_MIN : it->second;
		};
		check(ks.arguments.at(1).indexIsGlobal && global(specOn, ks.arguments.at(1).indexGlobal) == 6 &&
		          global(specOff, ks.arguments.at(1).indexGlobal) == 8,
		      "k_specialised: plus at kIndex + 1");
		check(ks.arguments.at(2).predicated && global(specOn, ks.arguments.at(2).predicateGlobal) == 1 &&
		          global(specOff, ks.arguments.at(2).predicateGlobal) == 0,
		      "k_specialised: either under kOn || kOff");
		// With no values, the required constants are unknown, so is plus's index, and every argument
		// they decide the reads of may be used; kOpt is not defined, so `optional` is not.
		RgfxAirSpecialisation none;
		check(usedOf(specialised, {}, none) == "11110111" && global(none, ks.arguments.at(1).indexGlobal) == INT64_MIN,
		      "k_specialised with no values: what it may use, plus's index unknown");
		const RgfxAirEntries vertex = entriesOf("v_specialised");
		RgfxAirSpecialisation modeOne, modeTwo;
		// position, uv, unread (no read anywhere: the AIR's air.arg_unused stands), mvp, uvs, vid.
		check(usedOf(vertex, {value(3, 29, 1)}, modeOne) == "111100" &&
		          usedOf(vertex, {value(3, 29, 2)}, modeTwo) == "101111",
		      "v_specialised under kMode 1 and 2: what it uses");
		// No code where the module declares no function constants.
		check(entriesOf("k_basic").code == nullptr, "k_basic: no constants, no code");
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
		// Spoilt code: no module's specialisation crashes or runs on.
		check(rgfxMetallibBitcode(oracle.data(), oracle.size(), "k_specialised", &bc, &n), "code to spoil");
		const std::vector<uint8_t> goodCode(bc, bc + n);
		read = 0;
		int followed = 0;
		for (int round = 0; round < 4000; round++) {
			std::vector<uint8_t> spoilt = goodCode;
			for (int k2 = 0; k2 < 1 + round % 4; k2++) {
				seed = seed * 1103515245u + 12345u;
				spoilt[(seed >> 8) % spoilt.size()] ^= static_cast<uint8_t>(1u << ((seed >> 4) & 7));
			}
			const size_t cut = round % 7 == 0 ? (seed >> 3) % spoilt.size() : spoilt.size();
			RgfxAirEntries e;
			if (!rgfxReadAirEntries(spoilt.data(), cut, e))
				continue;
			read++;
			for (const RgfxAirEntry &x : e.entries) {
				RgfxAirSpecialisation spec;
				if (rgfxAirSpecialise(e, x, round % 2 == 0 ? on.data() : off.data(),
				                      static_cast<uint32_t>(round % 2 == 0 ? on.size() : off.size()), spec) &&
				    spec.bodyRead)
					followed++;
			}
		}
		check(true, "4000 spoilt modules' code specialised without a crash (" + std::to_string(read) + " read, " +
		                std::to_string(followed) + " followed)");
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

	// The bitcode index answers what rgfxMetallibBitcode answers: for every name of every library
	// above and of the committed metallibs, for one of another library's names (bytes of another
	// header, which it walks), and for libraries whose header and function list are spoilt.
	{
		const char *oldPath = std::getenv("RGFX_AIR_ARGUMENTS_OLD");
		const std::vector<uint8_t> old = readFile(oldPath);
		size_t found = indexAsWalk(lib, namesOf(lib), "air_arguments") +
		               indexAsWalk(old, namesOf(old), "air_arguments_old") +
		               indexAsWalk(oracle, namesOf(oracle), "oracle_reflection");
		const char *dirs = std::getenv("RGFX_COMMITTED_METALLIBS");
		const std::string list = dirs != nullptr ? dirs : "";
		for (size_t at = 0; at < list.size();) {
			size_t colon = list.find(':', at);
			if (colon == std::string::npos)
				colon = list.size();
			for (const std::string &file : metallibsIn(list.substr(at, colon - at))) {
				const std::vector<uint8_t> one = readFile(file.c_str());
				found += indexAsWalk(one, namesOf(one), file);
			}
			at = colon + 1;
		}
		check(found >= 2 * 300, "the index finds the libraries' functions (" + std::to_string(found) + ")");
		const RgfxMetallibBitcodeIndex index(lib.data(), lib.size());
		for (const std::string &name : namesOf(oracle)) {
			const uint8_t *walkedAt = nullptr, *indexedAt = nullptr;
			size_t walkedSize = 0, indexedSize = 0;
			const bool walked = rgfxMetallibBitcode(oracle.data(), oracle.size(), name, &walkedAt, &walkedSize);
			check(index.find(oracle.data(), oracle.size(), name, &indexedAt, &indexedSize) == walked &&
			          walkedAt == indexedAt && walkedSize == indexedSize,
			      name + ": bytes of another header are walked");
		}
		uint64_t section = 0;
		std::memcpy(&section, lib.data() + 0x48, sizeof(section));
		check(section > 0x58 && section < lib.size(), "the test library's bitcode section");
		const std::vector<std::string> names = namesOf(lib);
		uint32_t seed = 0x5eed;
		size_t answered = 0;
		for (int round = 0; round < 2000; round++) {
			std::vector<uint8_t> spoilt = lib;
			for (int k = 0; k < 1 + round % 4; k++) {
				seed = seed * 1103515245u + 12345u;
				spoilt[(seed >> 8) % section] ^= static_cast<uint8_t>(1u << ((seed >> 4) & 7));
			}
			answered += indexAsWalk(spoilt, names, "spoilt round " + std::to_string(round));
		}
		check(answered > 0, "2000 spoilt libraries' indices answer as the walk (" + std::to_string(answered) + " found)");
	}

	// The cache: equal bytes, at any address, are read once and share their read; it keeps the
	// most recently looked-up modules while their costs fit its budget, reads one it let go again,
	// and keeps none costlier than the budget. A read it let go stays valid while it is held. A
	// module's cost is its bytes, and its decoded code's where it declares function constants.
	{
		std::vector<std::vector<uint8_t>> three = {moduleBytes(lib, "every_access"),
		                                           moduleBytes(lib, "reads_in_vertex"),
		                                           moduleBytes(lib, "writes_in_fragment")};
		auto costOf = [](const std::vector<uint8_t> &m) {
			RgfxAirModule read;
			rgfxReadAirModule(m.data(), m.size(), read);
			return rgfxAirModuleCost(read, m.size());
		};
		std::sort(three.begin(), three.end(),
		          [&](const std::vector<uint8_t> &x, const std::vector<uint8_t> &y) { return costOf(x) > costOf(y); });
		const std::vector<uint8_t> &a = three[0], &b = three[1], &c = three[2]; // a the costliest
		check(!c.empty() && a != b && b != c && a != c, "three modules to keep");
		const size_t ca = costOf(a), cb = costOf(b), cc = costOf(c);
		RgfxAirModule withCode;
		rgfxReadAirModule(moduleBytes(lib, "every_access").data(), moduleBytes(lib, "every_access").size(), withCode);
		check(withCode.entries.code != nullptr &&
		          rgfxAirModuleCost(withCode, moduleBytes(lib, "every_access").size()) >
		              moduleBytes(lib, "every_access").size(),
		      "every_access declares a constant: its code costs beside its bytes");
		check(ca == costOf(a), "a module's cost is the same at each read");
		const std::vector<uint8_t> a2 = a; // the same bytes at another address
		RgfxAirModuleCache cache(ca + cb);
		const auto pa = cache.moduleOf(a.data(), a.size());
		check(pa != nullptr && cache.reads() == 1 && cache.kept() == 1 && cache.keptBytes() == ca,
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
		check(pb != pa && cache.reads() == 2 && cache.kept() == 2 && cache.keptBytes() == ca + cb,
		      "other bytes are read, and kept");
		check(cache.moduleOf(a.data(), a.size()) == pa && cache.reads() == 2, "a, looked up again");
		const auto pc = cache.moduleOf(c.data(), c.size());
		check(cache.reads() == 3 && cache.kept() == 2 && cache.keptBytes() == ca + cc,
		      "c past the budget lets b go, the least recently looked up");
		check(cache.moduleOf(c.data(), c.size()) == pc && cache.moduleOf(a.data(), a.size()) == pa &&
		          cache.reads() == 3,
		      "a and c are kept");
		const auto pb2 = cache.moduleOf(b.data(), b.size());
		check(pb2 != pb && cache.reads() == 4 && cache.kept() == 2 && cache.keptBytes() == ca + cb,
		      "b, let go, is read again, and lets c go");
		check(pb->functions == pb2->functions && pb->entries == pb2->entries && pb->visible == pb2->visible,
		      "a read let go stays valid while it is held, and is the same as the new one");
		RgfxAirModuleCache tiny(ca - 1);
		const auto big = tiny.moduleOf(a.data(), a.size());
		check(big != nullptr && big->functions == pa->functions && tiny.kept() == 0 && tiny.keptBytes() == 0,
		      "a module costlier than the budget is read, not kept");
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
	// of the entry points below do not read: each has an explicit imageblock, an argument of a
	// kind the node reader does not follow, so each is listed with no nodes and their pipelines
	// answer no reflection (RgfxPipelineReflectionOf.mm). A fragment function's framebuffer-fetch input (`[[color(0)]]`,
	// `air.render_target`) reads, so rounding's fetch fragments are not listed.
	{
		const std::vector<std::string> nodesUnread = {
		    "mutdec_imageblock.metallib mutdec_imageblock",
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
					// The function's entry point in its module's nodes, where they read and list it.
					auto entryIn = [&](RgfxAirEntries &e) -> const RgfxAirEntry * {
						if (!rgfxReadAirEntries(bc.data(), bc.size(), e))
							return nullptr;
						auto it = std::find_if(e.entries.begin(), e.entries.end(),
						                       [&](const RgfxAirEntry &x) { return x.name == f.name; });
						return it != e.entries.end() ? &*it : nullptr;
					};
					std::vector<RgfxAirFunction> args;
					std::vector<RgfxAirVisible> vis;
					RgfxAirEntries nodes;
					switch (f.type) {
					case 0:
					case 1:
					case 2: {
						check(rgfxReadAirFunctions(bc.data(), bc.size(), args) &&
						          std::any_of(args.begin(), args.end(),
						                      [&](const RgfxAirFunction &x) { return x.name == f.name && x.argumentsRead; }),
						      what + ": its arguments");
						const std::string listedAs = file.substr(file.rfind('/') + 1) + " " + f.name;
						const bool unread =
						    std::find(nodesUnread.begin(), nodesUnread.end(), listedAs) != nodesUnread.end();
						const RgfxAirEntry *e = entryIn(nodes);
						check(e != nullptr && e->nodesRead != unread,
						      what + (unread ? ": listed, its nodes unread" : ": its nodes"));
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
					case 8: {
						const RgfxAirEntry *e = entryIn(nodes);
						check(e != nullptr && e->nodesRead, what + ": its nodes");
						break;
					}
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
