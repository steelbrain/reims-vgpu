// RgfxAirArguments: the arguments of a Metal library's entry points, from each one's AIR module
// (RgfxAirArguments.hpp). Our own code, over LLVM's documented bitstream container.
#include "RgfxAirArguments.hpp"

#include <algorithm>
#include <cstring>
#include <map>

namespace {

// LLVM's bitstream: the abbreviation ids every block has, and the blocks and records read here.
constexpr unsigned kEndBlock = 0, kEnterSubblock = 1, kDefineAbbrev = 2, kUnabbrevRecord = 3;
constexpr unsigned kBlockInfo = 0;
constexpr unsigned kModuleBlock = 8, kConstantsBlock = 11, kValueSymtabBlock = 14,
                   kMetadataBlock = 15, kStrtabBlock = 23;
// BLOCKINFO's record that names the block the abbreviations after it are for.
constexpr unsigned kBlockInfoSetBid = 1;
// MODULE_BLOCK's records: the version, and the global values in their value numbering's order.
constexpr unsigned kModuleVersion = 1, kModuleGlobalVar = 7, kModuleFunction = 8,
                   kModuleAliasOld = 9, kModuleAlias = 14, kModuleIfunc = 15;
// CONSTANTS_BLOCK's records: SETTYPE defines no value; NULL one of zero (an integer's 0 is
// written so), INTEGER one of a signed VBR; every other record one value.
constexpr unsigned kConstSetType = 1, kConstNull = 2, kConstInteger = 4;
// VALUE_SYMTAB_BLOCK's records with a name: [value, chars] and [value, offset, chars].
constexpr unsigned kVstEntry = 1, kVstFnEntry = 3;
// METADATA_BLOCK's records.
constexpr unsigned kMdStringOld = 1, kMdValue = 2, kMdNode = 3, kMdName = 4,
                   kMdDistinctNode = 5, kMdKind = 6, kMdNamedNode = 10, kMdStrings = 35,
                   kMdGlobalDeclAttachment = 36, kMdIndexOffset = 38, kMdIndex = 39;
// STRTAB_BLOCK's one record: the blob.
constexpr unsigned kStrtabBlob = 1;

// Reads bits from a byte array, least significant first, as LLVM writes them.
class Bits {
public:
	Bits(const uint8_t *p, size_t n) : p_(p), bits_(n * 8) {}
	bool ok() const { return ok_; }
	bool atEnd() const { return pos_ >= bits_; }
	size_t pos() const { return pos_; }
	void fail() { ok_ = false; pos_ = bits_; }
	uint64_t read(unsigned width)
	{
		if (width > 64 || width > bits_ - pos_) {
			fail();
			return 0;
		}
		uint64_t v = 0;
		for (unsigned got = 0; got < width;) {
			const unsigned bit = pos_ & 7;
			const unsigned take = std::min(8 - bit, width - got);
			v |= static_cast<uint64_t>((p_[pos_ >> 3] >> bit) & ((1u << take) - 1)) << got;
			got += take;
			pos_ += take;
		}
		return v;
	}
	uint64_t vbr(unsigned width)
	{
		if (width < 2 || width > 32) {
			fail();
			return 0;
		}
		const uint64_t more = 1ull << (width - 1);
		uint64_t v = 0;
		for (unsigned shift = 0; ok_; shift += width - 1) {
			const uint64_t piece = read(width);
			if (shift > 63) {
				fail();
				return 0;
			}
			v |= (piece & (more - 1)) << shift;
			if ((piece & more) == 0)
				return v;
		}
		return 0;
	}
	void align32()
	{
		const size_t to = (pos_ + 31) & ~static_cast<size_t>(31);
		if (to > bits_)
			fail();
		else
			pos_ = to;
	}
	// `n` bytes from here (the position on a byte), skipped; null where they run past the end.
	const uint8_t *takeBytes(size_t n)
	{
		if ((pos_ & 7) != 0 || n > (bits_ - pos_) / 8) {
			fail();
			return nullptr;
		}
		const uint8_t *at = p_ + pos_ / 8;
		pos_ += n * 8;
		return at;
	}
	void skipTo(size_t bit)
	{
		if (bit > bits_ || bit < pos_)
			fail();
		else
			pos_ = bit;
	}

private:
	const uint8_t *p_;
	size_t bits_;
	size_t pos_ = 0;
	bool ok_ = true;
};

// One operand of an abbreviation: a literal, or how a value is encoded.
enum : uint8_t { kOpLiteral = 0, kOpFixed = 1, kOpVbr = 2, kOpArray = 3, kOpChar6 = 4, kOpBlob = 5 };
struct AbbrevOp {
	uint8_t encoding;
	uint64_t value; // the literal, or the width of a fixed or VBR field
};
using Abbrev = std::vector<AbbrevOp>;

struct Record {
	unsigned code = 0;
	std::vector<uint64_t> ops;
	const uint8_t *blob = nullptr;
	size_t blobSize = 0;
};

bool readAbbrev(Bits &b, Abbrev &out)
{
	const uint64_t count = b.vbr(5);
	if (!b.ok() || count == 0 || count > 1024)
		return false;
	out.clear();
	for (uint64_t i = 0; i < count && b.ok(); i++) {
		AbbrevOp op = {};
		if (b.read(1) != 0) {
			op.encoding = kOpLiteral;
			op.value = b.vbr(8);
		} else {
			op.encoding = static_cast<uint8_t>(b.read(3));
			if (op.encoding == kOpFixed || op.encoding == kOpVbr)
				op.value = b.vbr(5);
			else if (op.encoding != kOpArray && op.encoding != kOpChar6 &&
			         op.encoding != kOpBlob)
				return false;
		}
		out.push_back(op);
	}
	return b.ok();
}

bool readScalar(Bits &b, const AbbrevOp &op, uint64_t &v)
{
	switch (op.encoding) {
	case kOpLiteral:
		v = op.value;
		return true;
	case kOpFixed:
		v = b.read(static_cast<unsigned>(op.value));
		return b.ok();
	case kOpVbr:
		v = op.value == 0 ? 0 : b.vbr(static_cast<unsigned>(op.value));
		return b.ok();
	case kOpChar6: {
		static const char kChars[] =
		    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._";
		v = static_cast<uint8_t>(kChars[b.read(6)]);
		return b.ok();
	}
	default:
		return false;
	}
}

// One record of abbreviation `a`: its first operand is the record's code.
bool readAbbreviated(Bits &b, const Abbrev &a, Record &r)
{
	r.ops.clear();
	r.blob = nullptr;
	r.blobSize = 0;
	bool haveCode = false;
	auto put = [&](uint64_t v) {
		if (!haveCode) {
			r.code = static_cast<unsigned>(v);
			haveCode = true;
		} else {
			r.ops.push_back(v);
		}
	};
	for (size_t i = 0; i < a.size(); i++) {
		const AbbrevOp &op = a[i];
		if (op.encoding == kOpArray) {
			if (i + 1 != a.size() - 1)
				return false;
			const uint64_t n = b.vbr(6);
			if (!b.ok() || n > (1u << 24))
				return false;
			for (uint64_t k = 0; k < n; k++) {
				uint64_t v = 0;
				if (!readScalar(b, a[i + 1], v))
					return false;
				put(v);
			}
			return b.ok() && haveCode;
		}
		if (op.encoding == kOpBlob) {
			if (i != a.size() - 1)
				return false;
			const uint64_t n = b.vbr(6);
			b.align32();
			r.blob = b.takeBytes(n);
			r.blobSize = n;
			b.align32();
			return b.ok() && r.blob != nullptr && haveCode;
		}
		uint64_t v = 0;
		if (!readScalar(b, op, v))
			return false;
		put(v);
	}
	return b.ok() && haveCode;
}

// The module's parts the arguments are read from.
struct Module {
	uint64_t version = 0;
	// The global values, in their value numbering: a function's name's place in the string
	// table (version 2), or its name from the value symbol table (before).
	struct Global {
		bool function = false;
		uint64_t strtabOffset = 0, strtabSize = 0;
		std::string name;
	};
	std::vector<Global> globals;
	// The module's integer constants, by value number.
	std::map<uint64_t, int64_t> integers;
	const uint8_t *strtab = nullptr;
	size_t strtabSize = 0;
	// The metadata, by metadata number.
	struct Md {
		enum Kind : uint8_t { kOther, kString, kValue, kNode } kind = kOther;
		std::string string;
		uint64_t value = 0;
		std::vector<uint64_t> ops; // a node's operands' numbers; UINT64_MAX for none
	};
	std::vector<Md> md;
	// The named metadata the entry points are listed in.
	std::map<std::string, std::vector<uint64_t>> named;
};

int64_t signedVbr(uint64_t v)
{
	if ((v & 1) == 0)
		return static_cast<int64_t>(v >> 1);
	if (v == 1)
		return INT64_MIN;
	return -static_cast<int64_t>(v >> 1);
}

// METADATA_STRINGS: [count, offset of the characters, blob]; the blob starts with the strings'
// lengths, VBR6 in a bitstream of their own, and holds their characters from `offset`.
bool readMdStrings(Module &m, const Record &r)
{
	if (r.ops.size() < 2 || r.blob == nullptr)
		return false;
	const uint64_t count = r.ops[0], offset = r.ops[1];
	if (offset > r.blobSize || count > r.blobSize)
		return false;
	Bits lengths(r.blob, static_cast<size_t>(offset));
	size_t at = static_cast<size_t>(offset);
	for (uint64_t i = 0; i < count; i++) {
		const uint64_t n = lengths.vbr(6);
		if (!lengths.ok() || n > r.blobSize - at)
			return false;
		Module::Md s;
		s.kind = Module::Md::kString;
		s.string.assign(reinterpret_cast<const char *>(r.blob + at), static_cast<size_t>(n));
		m.md.push_back(std::move(s));
		at += static_cast<size_t>(n);
	}
	return true;
}

std::string chars(const std::vector<uint64_t> &ops, size_t from)
{
	std::string s;
	for (size_t i = from; i < ops.size(); i++)
		s.push_back(static_cast<char>(ops[i]));
	return s;
}

class Reader {
public:
	explicit Reader(Module &m) : m_(m) {}

	// The blocks from the reader's position to the end of the enclosing block (or the stream):
	// `id` is the enclosing block's, `width` its abbreviation width, `abbrevs` its own.
	bool block(Bits &b, unsigned id, unsigned width, std::vector<Abbrev> abbrevs, unsigned depth)
	{
		if (depth > 16)
			return false;
		std::string pendingName;
		bool haveName = false;
		unsigned blockInfoTarget = ~0u;
		Record r;
		while (b.ok() && !b.atEnd()) {
			const uint64_t abbrevId = b.read(width);
			if (!b.ok())
				return false;
			if (abbrevId == kEndBlock) {
				b.align32();
				return b.ok();
			}
			if (abbrevId == kEnterSubblock) {
				const unsigned sub = static_cast<unsigned>(b.vbr(8));
				const unsigned subWidth = static_cast<unsigned>(b.vbr(4));
				b.align32();
				const uint64_t words = b.read(32);
				if (!b.ok() || subWidth == 0 || subWidth > 32)
					return false;
				const size_t end = b.pos() + static_cast<size_t>(words) * 32;
				if (!wanted(id, sub)) {
					b.skipTo(end);
					continue;
				}
				if (sub == kConstantsBlock)
					constantBase_ = m_.globals.size(), constantNext_ = 0;
				if (!block(b, sub, subWidth, blockInfo_[sub], depth + 1))
					return false;
				b.skipTo(end);
				continue;
			}
			if (abbrevId == kDefineAbbrev) {
				Abbrev a;
				if (!readAbbrev(b, a))
					return false;
				if (id == kBlockInfo) {
					if (blockInfoTarget == ~0u)
						return false;
					blockInfo_[blockInfoTarget].push_back(std::move(a));
				} else {
					abbrevs.push_back(std::move(a));
				}
				continue;
			}
			if (abbrevId == kUnabbrevRecord) {
				r.code = static_cast<unsigned>(b.vbr(6));
				const uint64_t n = b.vbr(6);
				if (!b.ok() || n > (1u << 24))
					return false;
				r.ops.resize(static_cast<size_t>(n));
				for (uint64_t i = 0; i < n; i++)
					r.ops[i] = b.vbr(6);
				r.blob = nullptr;
				r.blobSize = 0;
			} else {
				const uint64_t at = abbrevId - 4;
				if (at >= abbrevs.size() || !readAbbreviated(b, abbrevs[at], r))
					return false;
			}
			if (!b.ok())
				return false;
			if (id == kBlockInfo) {
				if (r.code == kBlockInfoSetBid && !r.ops.empty())
					blockInfoTarget = static_cast<unsigned>(r.ops[0]);
				continue;
			}
			if (!record(id, r, pendingName, haveName))
				return false;
		}
		return b.ok();
	}

private:
	// The blocks read: the module and, in it, its constants, value symbol table and metadata;
	// the string table beside it; and BLOCKINFO wherever it is. A function's own blocks, the
	// types and the rest are skipped.
	static bool wanted(unsigned parent, unsigned sub)
	{
		if (sub == kBlockInfo)
			return true;
		if (parent == ~0u)
			return sub == kModuleBlock || sub == kStrtabBlock;
		return parent == kModuleBlock && (sub == kConstantsBlock || sub == kValueSymtabBlock ||
		                                  sub == kMetadataBlock);
	}

	bool record(unsigned id, const Record &r, std::string &pendingName, bool &haveName)
	{
		switch (id) {
		case kModuleBlock:
			switch (r.code) {
			case kModuleVersion:
				if (!r.ops.empty())
					m_.version = r.ops[0];
				break;
			case kModuleGlobalVar:
			case kModuleFunction:
			case kModuleAliasOld:
			case kModuleAlias:
			case kModuleIfunc: {
				Module::Global g;
				g.function = r.code == kModuleFunction;
				if (m_.version >= 2 && r.ops.size() >= 2) {
					g.strtabOffset = r.ops[0];
					g.strtabSize = r.ops[1];
				}
				m_.globals.push_back(std::move(g));
				break;
			}
			default:
				break;
			}
			return true;
		case kConstantsBlock:
			if (r.code == kConstSetType)
				return true;
			if (r.code == kConstInteger && !r.ops.empty())
				m_.integers[constantBase_ + constantNext_] = signedVbr(r.ops[0]);
			else if (r.code == kConstNull)
				m_.integers[constantBase_ + constantNext_] = 0;
			constantNext_++;
			return true;
		case kValueSymtabBlock:
			if ((r.code == kVstEntry && r.ops.size() >= 1) ||
			    (r.code == kVstFnEntry && r.ops.size() >= 2)) {
				const uint64_t v = r.ops[0];
				if (v < m_.globals.size())
					m_.globals[v].name = chars(r.ops, r.code == kVstEntry ? 1 : 2);
			}
			return true;
		case kStrtabBlock:
			if (r.code == kStrtabBlob && r.blob != nullptr) {
				m_.strtab = r.blob;
				m_.strtabSize = r.blobSize;
			}
			return true;
		case kMetadataBlock:
			return metadata(r, pendingName, haveName);
		default:
			return true;
		}
	}

	bool metadata(const Record &r, std::string &pendingName, bool &haveName)
	{
		switch (r.code) {
		case kMdStrings:
			return readMdStrings(m_, r);
		case kMdStringOld: {
			Module::Md s;
			s.kind = Module::Md::kString;
			s.string = chars(r.ops, 0);
			m_.md.push_back(std::move(s));
			return true;
		}
		case kMdValue: {
			Module::Md v;
			if (r.ops.size() >= 2) {
				v.kind = Module::Md::kValue;
				v.value = r.ops[1];
			}
			m_.md.push_back(std::move(v));
			return true;
		}
		case kMdNode:
		case kMdDistinctNode: {
			Module::Md n;
			n.kind = Module::Md::kNode;
			n.ops.reserve(r.ops.size());
			for (uint64_t op : r.ops)
				n.ops.push_back(op == 0 ? UINT64_MAX : op - 1);
			m_.md.push_back(std::move(n));
			return true;
		}
		case kMdName:
			pendingName = chars(r.ops, 0);
			haveName = true;
			return true;
		case kMdNamedNode:
			if (haveName)
				m_.named[pendingName] = r.ops;
			haveName = false;
			return true;
		case kMdKind:
		case kMdGlobalDeclAttachment:
		case kMdIndexOffset:
		case kMdIndex:
			return true;
		default:
			// Every other record (debug info and the like) is one metadata of its own.
			m_.md.emplace_back();
			return true;
		}
	}

	Module &m_;
	std::map<unsigned, std::vector<Abbrev>> blockInfo_;
	uint64_t constantBase_ = 0, constantNext_ = 0;
};

const Module::Md *mdAt(const Module &m, uint64_t i)
{
	return i < m.md.size() ? &m.md[static_cast<size_t>(i)] : nullptr;
}

const std::string *mdString(const Module &m, uint64_t i)
{
	const Module::Md *d = mdAt(m, i);
	return d != nullptr && d->kind == Module::Md::kString ? &d->string : nullptr;
}

bool mdInteger(const Module &m, uint64_t i, int64_t &v)
{
	const Module::Md *d = mdAt(m, i);
	if (d == nullptr || d->kind != Module::Md::kValue)
		return false;
	auto it = m.integers.find(d->value);
	if (it == m.integers.end())
		return false;
	v = it->second;
	return true;
}

RgfxAirAccess accessOf(const std::string &s)
{
	if (s == "air.read")
		return RgfxAirAccess::Read;
	if (s == "air.write")
		return RgfxAirAccess::Write;
	if (s == "air.read_write")
		return RgfxAirAccess::ReadWrite;
	if (s == "air.sample")
		return RgfxAirAccess::Sample;
	return RgfxAirAccess::Unknown;
}

bool writes(RgfxAirAccess a)
{
	return a == RgfxAirAccess::Write || a == RgfxAirAccess::ReadWrite;
}

// An argument node's binding (`air.location_index`, index, count) and access, from operand
// `from` on. False where it has no binding the reader can follow.
bool binding(const Module &m, const Module::Md &node, size_t from, RgfxAirArgument &a)
{
	bool haveIndex = false;
	for (size_t i = from; i < node.ops.size(); i++) {
		const std::string *s = mdString(m, node.ops[i]);
		if (s == nullptr)
			continue;
		if (*s == "air.location_index" && !haveIndex && i + 2 < node.ops.size()) {
			int64_t index = 0, count = 0;
			if (!mdInteger(m, node.ops[i + 1], index) || !mdInteger(m, node.ops[i + 2], count) ||
			    index < 0 || count < 1 || index > UINT32_MAX || count > UINT32_MAX)
				return false;
			a.index = static_cast<uint32_t>(index);
			a.count = static_cast<uint32_t>(count);
			haveIndex = true;
			i += 2;
		} else if (a.access == RgfxAirAccess::Unknown && accessOf(*s) != RgfxAirAccess::Unknown) {
			a.access = accessOf(*s);
		}
	}
	return haveIndex;
}

// Whether an argument buffer's struct (its `air.struct_type_info` node) has a member texture it
// writes: each member names its argument with `air.indirect_argument` and that argument's node.
bool membersWriteTextures(const Module &m, const Module::Md &info, unsigned depth)
{
	if (depth > 8)
		return true;
	for (size_t i = 0; i + 1 < info.ops.size(); i++) {
		const std::string *s = mdString(m, info.ops[i]);
		if (s == nullptr || *s != "air.indirect_argument")
			continue;
		const Module::Md *member = mdAt(m, info.ops[i + 1]);
		if (member == nullptr || member->kind != Module::Md::kNode || member->ops.size() < 2)
			return true;
		const std::string *kind = mdString(m, member->ops[1]);
		if (kind == nullptr)
			return true;
		if (*kind == "air.texture") {
			for (uint64_t op : member->ops) {
				const std::string *a = mdString(m, op);
				if (a != nullptr && writes(accessOf(*a)))
					return true;
			}
		} else if (*kind == "air.indirect_buffer") {
			for (size_t k = 0; k + 1 < member->ops.size(); k++) {
				const std::string *t = mdString(m, member->ops[k]);
				const Module::Md *nested = mdAt(m, member->ops[k + 1]);
				if (t != nullptr && *t == "air.struct_type_info" && nested != nullptr &&
				    nested->kind == Module::Md::kNode && membersWriteTextures(m, *nested, depth + 1))
					return true;
			}
		}
	}
	return false;
}

// An argument buffer's members write a texture: where its node has no struct to read, so it
// could.
bool indirectWritesTextures(const Module &m, const Module::Md &node)
{
	for (size_t i = 0; i + 1 < node.ops.size(); i++) {
		const std::string *s = mdString(m, node.ops[i]);
		if (s != nullptr && *s == "air.struct_type_info") {
			const Module::Md *info = mdAt(m, node.ops[i + 1]);
			return info == nullptr || info->kind != Module::Md::kNode ||
			       membersWriteTextures(m, *info, 0);
		}
	}
	return true;
}

std::string globalName(const Module &m, uint64_t v)
{
	if (v >= m.globals.size())
		return {};
	const Module::Global &g = m.globals[static_cast<size_t>(v)];
	if (!g.name.empty())
		return g.name;
	if (m.strtab != nullptr && g.strtabOffset <= m.strtabSize &&
	    g.strtabSize <= m.strtabSize - g.strtabOffset)
		return std::string(reinterpret_cast<const char *>(m.strtab + g.strtabOffset),
		                   static_cast<size_t>(g.strtabSize));
	return {};
}

// One entry point's node: the function, then two nodes, the second its arguments.
bool entryPoint(const Module &m, uint64_t id, RgfxAirStage stage, RgfxAirFunction &out)
{
	const Module::Md *entry = mdAt(m, id);
	if (entry == nullptr || entry->kind != Module::Md::kNode || entry->ops.size() < 3)
		return false;
	const Module::Md *fn = mdAt(m, entry->ops[0]);
	const Module::Md *args = mdAt(m, entry->ops[2]);
	if (fn == nullptr || fn->kind != Module::Md::kValue || args == nullptr ||
	    args->kind != Module::Md::kNode)
		return false;
	const uint64_t v = fn->value;
	if (v >= m.globals.size() || !m.globals[static_cast<size_t>(v)].function)
		return false;
	out.name = globalName(m, v);
	out.stage = stage;
	out.arguments.clear();
	if (out.name.empty())
		return false;
	for (uint64_t argId : args->ops) {
		const Module::Md *arg = mdAt(m, argId);
		if (arg == nullptr || arg->kind != Module::Md::kNode || arg->ops.size() < 2)
			return false;
		const std::string *kind = mdString(m, arg->ops[1]);
		if (kind == nullptr)
			continue; // an output or a built-in, of no binding
		RgfxAirArgument a = {};
		a.access = RgfxAirAccess::Unknown;
		if (*kind == "air.buffer")
			a.kind = RgfxAirKind::Buffer;
		else if (*kind == "air.indirect_buffer")
			a.kind = RgfxAirKind::IndirectBuffer;
		else if (*kind == "air.texture")
			a.kind = RgfxAirKind::Texture;
		else if (*kind == "air.sampler")
			a.kind = RgfxAirKind::Sampler;
		else
			continue;
		if (!binding(m, *arg, 2, a))
			return false;
		// A texture of no access the reader knows is one it may write.
		if (a.kind == RgfxAirKind::Texture && a.access == RgfxAirAccess::Unknown)
			return false;
		if (a.kind == RgfxAirKind::IndirectBuffer)
			a.writesTextures = indirectWritesTextures(m, *arg);
		out.arguments.push_back(a);
	}
	return true;
}

uint32_t le32(const uint8_t *p)
{
	uint32_t v;
	std::memcpy(&v, p, sizeof(v));
	return v;
}

uint64_t le64(const uint8_t *p)
{
	uint64_t v;
	std::memcpy(&v, p, sizeof(v));
	return v;
}

} // namespace

bool rgfxReadAirFunctions(const uint8_t *bitcode, size_t size, std::vector<RgfxAirFunction> &out)
{
	out.clear();
	if (bitcode == nullptr || size < 8)
		return false;
	// LLVM's wrapper header: magic 0x0B17C0DE, version, then the bitcode's offset and size.
	if (le32(bitcode) == 0x0B17C0DEu) {
		if (size < 20)
			return false;
		const uint32_t at = le32(bitcode + 8), n = le32(bitcode + 12);
		if (at > size || n > size - at || n < 8)
			return false;
		bitcode += at;
		size = n;
	}
	if (bitcode[0] != 'B' || bitcode[1] != 'C' || bitcode[2] != 0xC0 || bitcode[3] != 0xDE)
		return false;
	Module m;
	Reader reader(m);
	Bits b(bitcode + 4, size - 4);
	if (!reader.block(b, ~0u, 2, {}, 0))
		return false;
	static const struct {
		const char *name;
		RgfxAirStage stage;
	} kStages[] = {{"air.kernel", RgfxAirStage::Kernel},
	               {"air.vertex", RgfxAirStage::Vertex},
	               {"air.fragment", RgfxAirStage::Fragment}};
	for (const auto &s : kStages) {
		auto it = m.named.find(s.name);
		if (it == m.named.end())
			continue;
		for (uint64_t id : it->second) {
			RgfxAirFunction f;
			if (!entryPoint(m, id, s.stage, f)) {
				out.clear();
				return false;
			}
			out.push_back(std::move(f));
		}
	}
	return true;
}

bool rgfxMetallibBitcode(const uint8_t *p, size_t n, const std::string &name,
                         const uint8_t **bitcode, size_t *bitcodeSize)
{
	// The header: the function list's offset at 0x18, the bitcode section's offset and size at
	// 0x48 and 0x50 (RgfxLibrary.mm reads the same list); a function's group of tags, each a
	// 4-byte name, a u16 size and its data, up to ENDT.
	if (p == nullptr || n < 0x58 || std::memcmp(p, "MTLB", 4) != 0)
		return false;
	const uint64_t list = le64(p + 0x18), section = le64(p + 0x48), sectionSize = le64(p + 0x50);
	if (list > n || n - list < 4 || section > n || sectionSize > n - section)
		return false;
	const uint32_t count = le32(p + list);
	uint64_t q = list + 4;
	for (uint32_t i = 0; i < count; i++) {
		if (n - q < 4)
			return false;
		const uint32_t group = le32(p + q);
		if (group < 4 || group > n - q)
			return false;
		const uint64_t end = q + group;
		std::string fnName;
		uint64_t offset = 0, length = 0;
		bool haveOffset = false, haveLength = false;
		for (uint64_t t = q + 4; end - t >= 6 && std::memcmp(p + t, "ENDT", 4) != 0;) {
			const uint16_t tagSize = static_cast<uint16_t>(p[t + 4] | (p[t + 5] << 8));
			const uint8_t *data = p + t + 6;
			if (tagSize > end - t - 6)
				return false;
			if (std::memcmp(p + t, "NAME", 4) == 0)
				fnName.assign(reinterpret_cast<const char *>(data),
				              strnlen(reinterpret_cast<const char *>(data), tagSize));
			else if (std::memcmp(p + t, "OFFT", 4) == 0 && tagSize >= 24) {
				offset = le64(data + 16);
				haveOffset = true;
			} else if (std::memcmp(p + t, "MDSZ", 4) == 0 && tagSize >= 8) {
				length = le64(data);
				haveLength = true;
			}
			t += 6 + static_cast<uint64_t>(tagSize);
		}
		if (fnName == name) {
			if (!haveOffset || !haveLength || offset > sectionSize || length > sectionSize - offset)
				return false;
			*bitcode = p + section + offset;
			*bitcodeSize = static_cast<size_t>(length);
			return true;
		}
		q = end;
	}
	return false;
}

bool rgfxAirWritesTexture(const RgfxAirFunction &fn, uint32_t index)
{
	for (const RgfxAirArgument &a : fn.arguments) {
		if (a.kind == RgfxAirKind::Texture && index >= a.index && index - a.index < a.count &&
		    writes(a.access))
			return true;
	}
	return false;
}
