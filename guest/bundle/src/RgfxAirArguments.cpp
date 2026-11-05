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
                   kMetadataBlock = 15, kTypeBlock = 17, kStrtabBlock = 23;
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
// TYPE_BLOCK's records: every one defines a type but NUMENTRY and STRUCT_NAME.
constexpr unsigned kTypeNumEntry = 1, kTypeVoid = 2, kTypeMetadata = 16, kTypeStructName = 19;
// METADATA_BLOCK's records. OLD_NODE and OLD_FN_NODE are LLVM 3.5's NODE and FN_NODE, which
// Apple's compiler writes for macOS 10.12 (-std=macos-metal1.2): operands of type and value pairs.
constexpr unsigned kMdStringOld = 1, kMdValue = 2, kMdNode = 3, kMdName = 4,
                   kMdDistinctNode = 5, kMdKind = 6, kMdOldNode = 8, kMdOldFnNode = 9,
                   kMdNamedNode = 10, kMdStrings = 35,
                   kMdGlobalDeclAttachment = 36, kMdIndexOffset = 38, kMdIndex = 39;
// STRTAB_BLOCK's one record: the blob.
constexpr unsigned kStrtabBlob = 1;
// A GLOBALVAR's unnamed_addr field for `unnamed_addr` (0 is none, 2 `local_unnamed_addr`).
constexpr uint64_t kUnnamedAddr = 1;
// The tag of a node's operand that names a value an old node holds in place (Module), past any
// metadata's number; UINT64_MAX, none, has both top bits.
constexpr uint64_t kInlineValue = 1ull << 62;

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
		// A global variable's unnamed_addr: 0 none, kUnnamedAddr, or 2 local_unnamed_addr.
		uint64_t unnamedAddr = 0;
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
	// The type table's numbers of the metadata and void types, and how many types it has.
	uint64_t types = 0, metadataType = UINT64_MAX, voidType = UINT64_MAX;
	// The values an old node holds in place (kMdOldNode), each a kValue metadata outside the
	// numbering: an operand naming one is kInlineValue | its place here.
	std::vector<Md> inlineValues;
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
		                                  sub == kMetadataBlock || sub == kTypeBlock);
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
				// GLOBALVAR: [type, isconst, initid, linkage, alignment, section, visibility,
				// threadlocal, unnamed_addr, ...], after the name's two fields in version 2.
				const size_t unnamedAt = m_.version >= 2 ? 10 : 8;
				if (r.code == kModuleGlobalVar && r.ops.size() > unnamedAt)
					g.unnamedAddr = r.ops[unnamedAt];
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
		case kTypeBlock:
			if (r.code == kTypeNumEntry || r.code == kTypeStructName)
				return true;
			if (r.code == kTypeMetadata)
				m_.metadataType = m_.types;
			else if (r.code == kTypeVoid)
				m_.voidType = m_.types;
			m_.types++;
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
		case kMdOldNode:
		case kMdOldFnNode: {
			// Each operand a type's number and a value's: a metadata's number where the type is
			// metadata, none where it is void, else a value, held in place.
			if (r.ops.size() % 2 != 0)
				return false;
			Module::Md n;
			n.kind = Module::Md::kNode;
			for (size_t i = 0; i < r.ops.size(); i += 2) {
				const uint64_t type = r.ops[i], value = r.ops[i + 1];
				if (type == m_.metadataType) {
					n.ops.push_back(value);
				} else if (type == m_.voidType) {
					n.ops.push_back(UINT64_MAX);
				} else {
					Module::Md v;
					v.kind = Module::Md::kValue;
					v.value = value;
					n.ops.push_back(kInlineValue | m_.inlineValues.size());
					m_.inlineValues.push_back(std::move(v));
				}
			}
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
	if ((i >> 62) == (kInlineValue >> 62)) {
		const uint64_t at = i & ~kInlineValue;
		return at < m.inlineValues.size() ? &m.inlineValues[static_cast<size_t>(at)] : nullptr;
	}
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

// Whether an `air.buffer` argument's node is of threadgroup memory (`air.address_space` 3,
// `[[threadgroup(n)]]`): memory each threadgroup has of its own, sized by the encoder, which no
// buffer is bound to.
bool threadgroupMemory(const Module &m, const Module::Md &node)
{
	for (size_t i = 0; i + 1 < node.ops.size(); i++) {
		const std::string *s = mdString(m, node.ops[i]);
		int64_t space = 0;
		if (s != nullptr && *s == "air.address_space" && mdInteger(m, node.ops[i + 1], space))
			return space == 3;
	}
	return false;
}

// Whether metadata `i` is a global variable of the module (one no function is): a binding index
// function constants set, which the module's initialiser stores there.
bool mdGlobalVariable(const Module &m, uint64_t i)
{
	const Module::Md *d = mdAt(m, i);
	return d != nullptr && d->kind == Module::Md::kValue && d->value < m.globals.size() &&
	       !m.globals[static_cast<size_t>(d->value)].function;
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
			if (mdGlobalVariable(m, node.ops[i + 1]))
				a.anyIndex = true;
			else if (!mdInteger(m, node.ops[i + 1], index))
				return false;
			if (mdGlobalVariable(m, node.ops[i + 2]))
				a.anyCount = true;
			else if (!mdInteger(m, node.ops[i + 2], count) || count < 1 || count > UINT32_MAX)
				return false;
			if (index < 0 || index > UINT32_MAX)
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

// The binding kinds besides buffers, argument buffers, textures and samplers that our Metal
// compiler gives an `air.location_index`, none of them a texture: a vertex function's and a
// kernel's stage input attribute and a post-tessellation vertex function's patch input (an
// attribute of the stage input descriptor, fetched from buffers), and the function tables and
// acceleration structures, bound among the buffers.
bool bindsNoTexture(const std::string &kind)
{
	return kind == "air.vertex_input" || kind == "air.stage_in" || kind == "air.patch_input" ||
	       kind == "air.visible_function_table" ||
	       kind == "air.intersection_function_table" ||
	       kind == "air.instance_acceleration_structure" ||
	       kind == "air.primitive_acceleration_structure";
}

// Whether an argument node names a binding (`air.location_index`).
bool hasBinding(const Module &m, const Module::Md &node)
{
	for (uint64_t op : node.ops) {
		const std::string *s = mdString(m, op);
		if (s != nullptr && *s == "air.location_index")
			return true;
	}
	return false;
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
		size_t kindAt = 1;
		const std::string *kind = mdString(m, arg->ops[1]);
		// An argument declared under a function constant: the constant's node, then its kind.
		// It is the function's whatever the constant's value.
		if (kind != nullptr && *kind == "air.function_constant") {
			kindAt = 3;
			kind = arg->ops.size() > kindAt ? mdString(m, arg->ops[kindAt]) : nullptr;
			if (kind == nullptr)
				return false;
		}
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
		else if (bindsNoTexture(*kind))
			continue;
		else if (hasBinding(m, *arg))
			return false; // a binding of a kind the reader does not know: it may be a texture
		else
			continue; // a built-in or a stage input, of no binding
		if (!binding(m, *arg, kindAt + 1, a))
			return false;
		if (a.kind == RgfxAirKind::Buffer && threadgroupMemory(m, *arg))
			continue; // threadgroup memory: no buffer at its index
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

// The module at `bitcode`, raw or in LLVM's wrapper header.
bool readModule(const uint8_t *bitcode, size_t size, Module &m)
{
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
	Reader reader(m);
	Bits b(bitcode + 4, size - 4);
	return reader.block(b, ~0u, 2, {}, 0);
}

// An unsigned integer operand of a node.
bool mdUnsigned(const Module &m, const Module::Md &node, size_t i, uint32_t &v)
{
	int64_t x = 0;
	if (i >= node.ops.size() || !mdInteger(m, node.ops[i], x) || x < 0 || x > INT32_MAX)
		return false;
	v = static_cast<uint32_t>(x);
	return true;
}

// The node `id`, where it is one.
const Module::Md *mdNode(const Module &m, uint64_t id)
{
	const Module::Md *d = mdAt(m, id);
	return d != nullptr && d->kind == Module::Md::kNode ? d : nullptr;
}

bool scalarNamed(const std::string &s, RgfxAirScalar &out)
{
	static const struct {
		const char *name;
		RgfxAirScalar scalar;
	} kScalars[] = {
		{"air.float_type", RgfxAirScalar::Float}, {"air.half_type", RgfxAirScalar::Half},
		{"air.bfloat_type", RgfxAirScalar::BFloat},
		{"air.int_type", RgfxAirScalar::Int},     {"air.uint_type", RgfxAirScalar::UInt},
		{"air.short_type", RgfxAirScalar::Short}, {"air.ushort_type", RgfxAirScalar::UShort},
		{"air.char_type", RgfxAirScalar::Char},   {"air.uchar_type", RgfxAirScalar::UChar},
		{"air.bool_type", RgfxAirScalar::Bool},   {"air.long_type", RgfxAirScalar::Long},
		{"air.ulong_type", RgfxAirScalar::ULong},
	};
	for (const auto &k : kScalars)
		if (s == k.name) {
			out = k.scalar;
			return true;
		}
	return false;
}

bool textureNamed(const std::string &s, RgfxAirTextureShape &shape, bool &depth)
{
	static const struct {
		const char *name;
		RgfxAirTextureShape shape;
		bool depth;
	} kTextures[] = {
		{"air.texture_1d_type", RgfxAirTextureShape::T1D, false},
		{"air.texture_1d_array_type", RgfxAirTextureShape::T1DArray, false},
		{"air.texture_2d_type", RgfxAirTextureShape::T2D, false},
		{"air.texture_2d_array_type", RgfxAirTextureShape::T2DArray, false},
		{"air.texture_2d_ms_type", RgfxAirTextureShape::T2DMS, false},
		{"air.texture_2d_ms_array_type", RgfxAirTextureShape::T2DMSArray, false},
		{"air.texture_cube_type", RgfxAirTextureShape::Cube, false},
		{"air.texture_cube_array_type", RgfxAirTextureShape::CubeArray, false},
		{"air.texture_3d_type", RgfxAirTextureShape::T3D, false},
		{"air.texture_buffer_1d_type", RgfxAirTextureShape::Buffer1D, false},
		{"air.depth_2d_type", RgfxAirTextureShape::T2D, true},
		{"air.depth_2d_array_type", RgfxAirTextureShape::T2DArray, true},
		{"air.depth_2d_ms_type", RgfxAirTextureShape::T2DMS, true},
		{"air.depth_2d_ms_array_type", RgfxAirTextureShape::T2DMSArray, true},
		{"air.depth_cube_type", RgfxAirTextureShape::Cube, true},
		{"air.depth_cube_array_type", RgfxAirTextureShape::CubeArray, true},
	};
	for (const auto &k : kTextures)
		if (s == k.name) {
			shape = k.shape;
			depth = k.depth;
			return true;
		}
	return false;
}

// A type node's address space: its operand `space`, none (0) or a node of one
// `air.address_space` node of the space's number. False where it is neither.
bool addressSpace(const Module &m, uint64_t space, uint32_t &out)
{
	out = 0;
	if (space == UINT64_MAX)
		return true;
	const Module::Md *list = mdNode(m, space);
	const Module::Md *as = list != nullptr && list->ops.size() == 1 ? mdNode(m, list->ops[0]) : nullptr;
	const std::string *tag = as != nullptr && as->ops.size() == 2 ? mdString(m, as->ops[0]) : nullptr;
	return tag != nullptr && *tag == "air.address_space" && mdUnsigned(m, *as, 1, out);
}

// The type node `id` (RgfxAirArguments.hpp: its name, size, alignment and address space, then
// what its kind adds). False for a node of a kind the reader does not know.
bool airType(const Module &m, uint64_t id, unsigned depth, RgfxAirType &out)
{
	const Module::Md *node = mdNode(m, id);
	if (depth > 16 || node == nullptr || node->ops.size() < 4)
		return false;
	const std::string *tag = mdString(m, node->ops[0]);
	if (tag == nullptr || !mdUnsigned(m, *node, 1, out.size) || !mdUnsigned(m, *node, 2, out.align) ||
	    !addressSpace(m, node->ops[3], out.addressSpace))
		return false;
	const std::string &t = *tag;
	if (scalarNamed(t, out.scalar)) {
		out.kind = RgfxAirType::Kind::Scalar;
		return true;
	}
	if (t == "air.void_type") {
		out.kind = RgfxAirType::Kind::Void;
		return true;
	}
	if (t == "air.enum_type") {
		const std::string *name = node->ops.size() >= 6 ? mdString(m, node->ops[4]) : nullptr;
		RgfxAirType underlying;
		if (name == nullptr || !airType(m, node->ops[5], depth + 1, underlying) ||
		    underlying.kind != RgfxAirType::Kind::Scalar)
			return false;
		out.kind = RgfxAirType::Kind::Enum;
		out.name = *name;
		out.scalar = underlying.scalar;
		return true;
	}
	if (t == "air.visible_function_table_type") {
		out.kind = RgfxAirType::Kind::VisibleFunctionTable;
		return true;
	}
	if (t == "air.intersection_function_table_type") {
		out.kind = RgfxAirType::Kind::IntersectionFunctionTable;
		return true;
	}
	if (t == "air.acceleration_structure_type") {
		out.kind = RgfxAirType::Kind::AccelerationStructure;
		return true;
	}
	if (t == "air.vector_type" || t == "air.packed_vector_type") {
		RgfxAirType element;
		if (node->ops.size() < 6 || !airType(m, node->ops[4], depth + 1, element) ||
		    element.kind != RgfxAirType::Kind::Scalar || !mdUnsigned(m, *node, 5, out.count))
			return false;
		out.kind = RgfxAirType::Kind::Vector;
		out.scalar = element.scalar;
		out.packed = t == "air.packed_vector_type";
		return true;
	}
	if (t == "air.matrix_type") {
		RgfxAirType element;
		if (node->ops.size() < 7 || !airType(m, node->ops[4], depth + 1, element) ||
		    element.kind != RgfxAirType::Kind::Scalar || !mdUnsigned(m, *node, 5, out.count) ||
		    !mdUnsigned(m, *node, 6, out.rows))
			return false;
		out.kind = RgfxAirType::Kind::Matrix;
		out.scalar = element.scalar;
		return true;
	}
	if (t == "air.array_type" || t == "air.array_of_type") {
		out.element.resize(1);
		if (node->ops.size() < 6 || !airType(m, node->ops[4], depth + 1, out.element[0]) ||
		    !mdUnsigned(m, *node, 5, out.count))
			return false;
		out.kind = t == "air.array_type" ? RgfxAirType::Kind::Array : RgfxAirType::Kind::ArrayOf;
		return true;
	}
	// A reference is a pointer here: Metal reflects them alike (RgfxReflection.h).
	if (t == "air.pointer_type" || t == "air.lvalue_reference_type" || t == "air.rvalue_reference_type") {
		out.element.resize(1);
		if (node->ops.size() < 5 || !airType(m, node->ops[4], depth + 1, out.element[0]))
			return false;
		out.kind = RgfxAirType::Kind::Pointer;
		return true;
	}
	if (t == "air.struct_type") {
		const std::string *name = node->ops.size() >= 6 ? mdString(m, node->ops[4]) : nullptr;
		// A struct of no size has none (`metal::_atomic<float>`, where an argument points to it).
		static const Module::Md kNoFields;
		const Module::Md *fields = node->ops.size() < 6         ? nullptr
		                           : node->ops[5] == UINT64_MAX ? &kNoFields
		                                                        : mdNode(m, node->ops[5]);
		if (name == nullptr || fields == nullptr)
			return false;
		out.kind = RgfxAirType::Kind::Struct;
		out.name = *name;
		for (uint64_t f : fields->ops) {
			const Module::Md *field = mdNode(m, f);
			const std::string *fieldTag = field != nullptr && field->ops.size() >= 5
			                                  ? mdString(m, field->ops[0])
			                                  : nullptr;
			const std::string *fieldName = fieldTag != nullptr ? mdString(m, field->ops[4]) : nullptr;
			RgfxAirField rf;
			if (fieldName == nullptr || *fieldTag != "air.record_field" ||
			    !mdUnsigned(m, *field, 1, rf.offset) || !airType(m, field->ops[3], depth + 1, rf.type))
				return false;
			rf.name = *fieldName;
			out.fields.push_back(std::move(rf));
		}
		return true;
	}
	if (textureNamed(t, out.shape, out.depth)) {
		RgfxAirType sample;
		const std::string *access = node->ops.size() >= 6 ? mdString(m, node->ops[5]) : nullptr;
		if (access == nullptr || !airType(m, node->ops[4], depth + 1, sample) ||
		    sample.kind != RgfxAirType::Kind::Scalar)
			return false;
		out.kind = RgfxAirType::Kind::Texture;
		out.scalar = sample.scalar;
		out.access = accessOf("air." + *access);
		return out.access != RgfxAirAccess::Unknown;
	}
	if (t == "air.sampler_type") {
		out.kind = RgfxAirType::Kind::Sampler;
		return true;
	}
	return false;
}

// An `air.stitching_type` node's type.
bool stitchingType(const Module &m, uint64_t id, RgfxAirType &out)
{
	const Module::Md *node = mdNode(m, id);
	const std::string *tag = node != nullptr && node->ops.size() >= 2 ? mdString(m, node->ops[0]) : nullptr;
	return tag != nullptr && *tag == "air.stitching_type" && airType(m, node->ops[1], 0, out);
}

// One `air.visible` node: the function, its outputs, its inputs and, for a stitchable function,
// its `air.stitching_info`.
bool visibleFunction(const Module &m, uint64_t id, RgfxAirVisible &out)
{
	const Module::Md *entry = mdNode(m, id);
	if (entry == nullptr || entry->ops.size() < 3)
		return false;
	const Module::Md *fn = mdAt(m, entry->ops[0]);
	if (fn == nullptr || fn->kind != Module::Md::kValue || fn->value >= m.globals.size() ||
	    !m.globals[static_cast<size_t>(fn->value)].function)
		return false;
	out.name = globalName(m, fn->value);
	if (out.name.empty())
		return false;
	const Module::Md *info = nullptr;
	for (size_t i = 3; i < entry->ops.size() && info == nullptr; i++) {
		const Module::Md *n = mdNode(m, entry->ops[i]);
		const std::string *tag = n != nullptr && !n->ops.empty() ? mdString(m, n->ops[0]) : nullptr;
		if (tag != nullptr && *tag == "air.stitching_info")
			info = n;
	}
	if (info == nullptr) {
		out.signature = RgfxAirVisible::Signature::None;
		return true;
	}
	out.signature = RgfxAirVisible::Signature::Unknown;
	if (info->ops.size() < 2)
		return true;
	out.returnsVoid = info->ops[1] == UINT64_MAX;
	if (!out.returnsVoid && !stitchingType(m, info->ops[1], out.returnType))
		return true;
	for (size_t i = 2; i < info->ops.size(); i++) {
		const Module::Md *arg = mdNode(m, info->ops[i]);
		const std::string *tag = arg != nullptr && arg->ops.size() >= 2 ? mdString(m, arg->ops[0]) : nullptr;
		// An unnamed argument's node has no name.
		const std::string *name = tag != nullptr && arg->ops.size() >= 3 ? mdString(m, arg->ops[2]) : nullptr;
		RgfxAirType type;
		if (tag == nullptr || *tag != "air.stitching_argument" || (name == nullptr && arg->ops.size() >= 3) ||
		    !stitchingType(m, arg->ops[1], type))
			return true;
		out.argumentNames.push_back(name != nullptr ? *name : std::string());
		out.argumentTypes.push_back(std::move(type));
	}
	out.signature = RgfxAirVisible::Signature::Read;
	return true;
}

// MTLDataType of a function constant's type name: a scalar's, or a vector's of two to four of
// them, which MTLDataType numbers after its scalar's.
bool constantType(const std::string &s, uint32_t &type)
{
	static const struct {
		const char *name;
		uint32_t type;
	} kScalars[] = {{"float", 3},  {"half", 16},   {"int", 29},  {"uint", 33},
	                {"short", 37}, {"ushort", 41}, {"char", 45}, {"uchar", 49},
	                {"bool", 53},  {"long", 81},   {"ulong", 85}};
	for (const auto &k : kScalars) {
		const size_t n = std::strlen(k.name);
		if (s.compare(0, n, k.name) != 0)
			continue;
		if (s.size() == n) {
			type = k.type;
			return true;
		}
		if (s.size() == n + 1 && s[n] >= '2' && s[n] <= '4') {
			type = k.type + static_cast<uint32_t>(s[n] - '1');
			return true;
		}
	}
	return false;
}

// The function constants of the module (`air.function_constants`): false where a node does not
// read, `out` then empty.
bool readConstants(const Module &m, std::vector<RgfxAirConstant> &out)
{
	out.clear();
	auto it = m.named.find("air.function_constants");
	if (it == m.named.end())
		return true;
	for (uint64_t id : it->second) {
		const Module::Md *node = mdNode(m, id);
		const bool whole = node != nullptr && node->ops.size() >= 4;
		const std::string *type = whole ? mdString(m, node->ops[1]) : nullptr;
		const std::string *name = whole ? mdString(m, node->ops[2]) : nullptr;
		RgfxAirConstant c;
		if (!whole || !mdGlobalVariable(m, node->ops[0]) || type == nullptr || name == nullptr ||
		    !constantType(*type, c.type) || !mdUnsigned(m, *node, 3, c.index)) {
			out.clear();
			return false;
		}
		c.name = *name;
		const uint64_t global = mdAt(m, node->ops[0])->value;
		c.required = m.globals[static_cast<size_t>(global)].unnamedAddr == kUnnamedAddr;
		out.push_back(std::move(c));
	}
	return true;
}

// --- whole nodes, for a pipeline's reflection ----------------------------------------------

// A string operand of a node.
bool mdText(const Module &m, const Module::Md &node, size_t i, std::string &out)
{
	const std::string *s = i < node.ops.size() ? mdString(m, node.ops[i]) : nullptr;
	if (s == nullptr)
		return false;
	out = *s;
	return true;
}

bool layoutOf(const Module &m, uint64_t id, unsigned depth,
              std::map<uint64_t, std::shared_ptr<const std::vector<RgfxAirMember>>> &layouts,
              std::shared_ptr<const std::vector<RgfxAirMember>> &out);

// The node `id` of outputs (a vertex function's, a mesh's vertex or primitive data): each a node of
// its kind, then keys, `air.arg_type_name` and `air.arg_name` among them, whose other operands
// (a render target's or a vertex output's numbers, `generated(...)` names) reflection does not
// read. False where it is not a node of such nodes.
bool outputsOf(const Module &m, uint64_t id, std::vector<RgfxAirOutput> &out)
{
	const Module::Md *list = mdNode(m, id);
	if (list == nullptr)
		return false;
	for (uint64_t o : list->ops) {
		const Module::Md *node = mdNode(m, o);
		RgfxAirOutput output;
		if (node == nullptr || !mdText(m, *node, 0, output.kind))
			return false;
		for (size_t i = 1; i + 1 < node->ops.size(); i++) {
			const std::string *key = mdString(m, node->ops[i]);
			if (key != nullptr && (*key == "air.arg_type_name" || *key == "air.arg_name") &&
			    mdText(m, *node, i + 1, *key == "air.arg_name" ? output.name : output.typeName))
				i++;
		}
		out.push_back(std::move(output));
	}
	return true;
}

// The node `id` whole (RgfxAirArguments.hpp, "An entry point's nodes, whole"): false where an
// operand is not of the form its key takes.
bool wholeNode(const Module &m, uint64_t id, unsigned depth,
               std::map<uint64_t, std::shared_ptr<const std::vector<RgfxAirMember>>> &layouts, RgfxAirNode &out)
{
	const Module::Md *node = mdNode(m, id);
	if (node == nullptr || depth > 8)
		return false;
	size_t i = 0;
	int64_t position = 0;
	if (!node->ops.empty() && mdInteger(m, node->ops[0], position)) {
		if (position < 0 || position > INT32_MAX)
			return false;
		out.hasPosition = true;
		out.position = static_cast<uint32_t>(position);
		i = 1;
	}
	for (; i < node->ops.size(); i++) {
		const std::string *key = mdString(m, node->ops[i]);
		if (key == nullptr)
			return false; // an operand no key takes
		const std::string &k = *key;
		if (k == "air.function_constant") {
			const Module::Md *pred = i + 1 < node->ops.size() ? mdNode(m, node->ops[i + 1]) : nullptr;
			if (pred == nullptr || !mdText(m, *pred, 2, out.predicate))
				return false;
			out.predicated = true;
			i += 1;
		} else if (k == "air.location_index") {
			int64_t index = 0, count = 0;
			if (i + 2 >= node->ops.size() || !mdInteger(m, node->ops[i + 2], count) || count < 1 ||
			    count > INT32_MAX)
				return false;
			if (mdGlobalVariable(m, node->ops[i + 1]))
				out.indexIsGlobal = true;
			else if (!mdInteger(m, node->ops[i + 1], index) || index < 0 || index > INT32_MAX)
				return false;
			out.hasLocation = true;
			out.index = static_cast<uint32_t>(index);
			out.count = static_cast<uint32_t>(count);
			i += 2;
		} else if (k == "air.buffer_size" || k == "air.address_space" || k == "air.arg_type_size" ||
		           k == "air.arg_type_align_size") {
			uint32_t v = 0;
			if (!mdUnsigned(m, *node, i + 1, v))
				return false;
			if (k == "air.buffer_size")
				out.hasBufferSize = true, out.bufferSize = v;
			else if (k == "air.address_space")
				out.hasAddressSpace = true, out.addressSpace = v;
			else if (k == "air.arg_type_size")
				out.hasTypeSize = true, out.typeSize = v;
			else
				out.hasTypeAlign = true, out.typeAlign = v;
			i += 1;
		} else if (k == "air.struct_type_info") {
			if (i + 1 >= node->ops.size() || !layoutOf(m, node->ops[i + 1], depth + 1, layouts, out.layout))
				return false;
			i += 1;
		} else if (k == "air.arg_type_name" || k == "air.arg_name") {
			if (!mdText(m, *node, i + 1, k == "air.arg_name" ? out.name : out.typeName))
				return false;
			i += 1;
		} else if (k == "air.arg_unused") {
			out.unused = true;
		} else if (accessOf(k) != RgfxAirAccess::Unknown) {
			out.access = accessOf(k);
		} else if (k == "air.mesh" && out.kind.empty()) {
			out.kind = k;
			const Module::Md *info = i + 1 < node->ops.size() ? mdNode(m, node->ops[i + 1]) : nullptr;
			std::string tag;
			if (info == nullptr || !mdText(m, *info, 0, tag) || tag != "air.mesh_type_info" || info->ops.size() < 6 ||
			    !outputsOf(m, info->ops[1], out.meshVertexData) || !outputsOf(m, info->ops[2], out.meshPrimitiveData) ||
			    !mdUnsigned(m, *info, 3, out.meshVertices) || !mdUnsigned(m, *info, 4, out.meshPrimitives) ||
			    !mdText(m, *info, 5, out.meshTopology))
				return false;
			out.hasMesh = true;
			i += 1;
		} else if (out.kind.empty()) {
			out.kind = k;
		} else {
			// What else a node of a stage input, an output or an interpolated input says (its
			// interpolation, `generated(...)` names): nothing reflection answers.
			continue;
		}
	}
	return !out.kind.empty();
}

// The layout node `id`: its members, each read once a module (`layouts`).
bool layoutOf(const Module &m, uint64_t id, unsigned depth,
              std::map<uint64_t, std::shared_ptr<const std::vector<RgfxAirMember>>> &layouts,
              std::shared_ptr<const std::vector<RgfxAirMember>> &out)
{
	auto known = layouts.find(id);
	if (known != layouts.end()) {
		out = known->second;
		return out != nullptr;
	}
	layouts[id] = nullptr; // a layout that holds itself reads as none
	const Module::Md *node = mdNode(m, id);
	if (node == nullptr || depth > 8)
		return false;
	auto members = std::make_shared<std::vector<RgfxAirMember>>();
	for (size_t i = 0; i < node->ops.size();) {
		RgfxAirMember member;
		const std::string *lead = mdString(m, node->ops[i]);
		if (lead != nullptr && *lead == "air.struct_type_info") {
			if (i + 1 >= node->ops.size() || !layoutOf(m, node->ops[i + 1], depth + 1, layouts, member.layout))
				return false;
			i += 2;
		}
		if (!mdUnsigned(m, *node, i, member.offset) || !mdUnsigned(m, *node, i + 1, member.size) ||
		    !mdUnsigned(m, *node, i + 2, member.arrayLength) || !mdText(m, *node, i + 3, member.typeName) ||
		    !mdText(m, *node, i + 4, member.name))
			return false;
		i += 5;
		while (i < node->ops.size()) {
			const std::string *key = mdString(m, node->ops[i]);
			if (key == nullptr || i + 1 >= node->ops.size())
				break;
			if (*key == "air.indirect_argument") {
				if (mdNode(m, node->ops[i + 1]) != nullptr) {
					auto arg = std::make_shared<RgfxAirNode>();
					if (!wholeNode(m, node->ops[i + 1], depth + 1, layouts, *arg))
						return false;
					member.indirect = std::move(arg);
				} else if (mdUnsigned(m, *node, i + 1, member.firstId)) {
					member.hasFirstId = true;
				} else {
					return false;
				}
			} else if (*key == "air.render_target") {
				if (!mdUnsigned(m, *node, i + 1, member.renderTarget))
					return false;
				member.hasRenderTarget = true;
			} else if (*key == "air.raster_order_group") {
				if (!mdUnsigned(m, *node, i + 1, member.rasterOrderGroup))
					return false;
				member.hasRasterOrderGroup = true;
			} else {
				break; // the next member's `air.struct_type_info`
			}
			i += 2;
		}
		members->push_back(std::move(member));
	}
	out = members;
	layouts[id] = out;
	return true;
}

// The visible functions of the module `m` (rgfxReadAirVisible).
bool visibleOf(const Module &m, std::vector<RgfxAirVisible> &out)
{
	out.clear();
	auto it = m.named.find("air.visible");
	if (it == m.named.end())
		return true;
	std::vector<RgfxAirConstant> constants;
	const bool constantsRead = readConstants(m, constants);
	for (uint64_t id : it->second) {
		RgfxAirVisible v;
		if (!visibleFunction(m, id, v)) {
			out.clear();
			return false;
		}
		v.constants = constants;
		v.constantsRead = constantsRead;
		if (v.signature != RgfxAirVisible::Signature::Read) {
			v.returnsVoid = false;
			v.returnType = {};
			v.argumentNames.clear();
			v.argumentTypes.clear();
		}
		out.push_back(std::move(v));
	}
	return true;
}

// The entry points' arguments of the module `m` (rgfxReadAirFunctions).
bool functionsOf(const Module &m, std::vector<RgfxAirFunction> &out)
{
	out.clear();
	static const struct {
		const char *name;
		RgfxAirStage stage;
	} kStages[] = {{"air.kernel", RgfxAirStage::Kernel},
	               {"air.vertex", RgfxAirStage::Vertex},
	               {"air.fragment", RgfxAirStage::Fragment}};
	std::vector<RgfxAirConstant> constants;
	const bool constantsRead = readConstants(m, constants);
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
			f.constants = constants;
			f.constantsRead = constantsRead;
			out.push_back(std::move(f));
		}
	}
	return true;
}

// The entry points' nodes whole of the module `m` (rgfxReadAirEntries).
bool entriesOf(const Module &m, RgfxAirEntries &out)
{
	out = {};
	static const struct {
		const char *name;
		RgfxAirEntryStage stage;
	} kStages[] = {{"air.kernel", RgfxAirEntryStage::Kernel}, {"air.vertex", RgfxAirEntryStage::Vertex},
	               {"air.fragment", RgfxAirEntryStage::Fragment}, {"air.object", RgfxAirEntryStage::Object},
	               {"air.mesh", RgfxAirEntryStage::Mesh}};
	std::map<uint64_t, std::shared_ptr<const std::vector<RgfxAirMember>>> layouts;
	for (const auto &s : kStages) {
		auto it = m.named.find(s.name);
		if (it == m.named.end())
			continue;
		for (uint64_t id : it->second) {
			const Module::Md *entry = mdAt(m, id);
			const Module::Md *fn = entry != nullptr && entry->kind == Module::Md::kNode && entry->ops.size() >= 3
			                           ? mdAt(m, entry->ops[0])
			                           : nullptr;
			const Module::Md *args = fn != nullptr ? mdNode(m, entry->ops[2]) : nullptr;
			if (fn == nullptr || fn->kind != Module::Md::kValue || args == nullptr ||
			    fn->value >= m.globals.size() || !m.globals[static_cast<size_t>(fn->value)].function) {
				out = {};
				return false;
			}
			RgfxAirEntry e;
			e.name = globalName(m, fn->value);
			e.stage = s.stage;
			if (!outputsOf(m, entry->ops[1], e.outputs)) {
				out = {};
				return false;
			}
			for (uint64_t a : args->ops) {
				RgfxAirNode n;
				if (!wholeNode(m, a, 0, layouts, n)) {
					out = {};
					return false;
				}
				e.arguments.push_back(std::move(n));
			}
			if (e.name.empty()) {
				out = {};
				return false;
			}
			out.entries.push_back(std::move(e));
		}
	}
	auto constants = m.named.find("air.function_constants");
	if (constants != m.named.end()) {
		for (uint64_t id : constants->second) {
			const Module::Md *c = mdNode(m, id);
			std::string name;
			uint32_t index = 0;
			if (c == nullptr || !mdText(m, *c, 2, name) || !mdUnsigned(m, *c, 3, index)) {
				out = {};
				return false;
			}
			out.constants.emplace_back(name, index);
		}
	}
	return true;
}

} // namespace

bool rgfxReadAirVisible(const uint8_t *bitcode, size_t size, std::vector<RgfxAirVisible> &out)
{
	out.clear();
	Module m;
	return readModule(bitcode, size, m) && visibleOf(m, out);
}

bool rgfxReadAirFunctions(const uint8_t *bitcode, size_t size, std::vector<RgfxAirFunction> &out)
{
	out.clear();
	Module m;
	return readModule(bitcode, size, m) && functionsOf(m, out);
}

bool rgfxReadAirEntries(const uint8_t *bitcode, size_t size, RgfxAirEntries &out)
{
	out = {};
	Module m;
	return readModule(bitcode, size, m) && entriesOf(m, out);
}

void rgfxReadAirModule(const uint8_t *bitcode, size_t size, RgfxAirModule &out)
{
	out = {};
	Module m;
	if (!readModule(bitcode, size, m))
		return;
	out.functionsRead = functionsOf(m, out.functions);
	out.visibleRead = visibleOf(m, out.visible);
	out.entriesRead = entriesOf(m, out.entries);
}

std::shared_ptr<const RgfxAirModule> RgfxAirModuleCache::moduleOf(const uint8_t *bitcode, size_t size)
{
	const std::string_view key(reinterpret_cast<const char *>(bitcode), bitcode != nullptr ? size : 0);
	{
		std::lock_guard<std::mutex> hold(lock_);
		auto found = byBytes_.find(key);
		if (found != byBytes_.end()) {
			order_.splice(order_.begin(), order_, found->second);
			return found->second->module;
		}
	}
	auto read = std::make_shared<RgfxAirModule>();
	rgfxReadAirModule(bitcode, size, *read);
	std::shared_ptr<const RgfxAirModule> module = std::move(read);
	std::lock_guard<std::mutex> hold(lock_);
	reads_++;
	auto found = byBytes_.find(key);
	if (found != byBytes_.end()) {
		// Another thread read the same bytes meanwhile: its read is the one kept.
		order_.splice(order_.begin(), order_, found->second);
		return found->second->module;
	}
	order_.push_front(Kept{std::string(key), module});
	byBytes_.emplace(std::string_view(order_.front().bytes), order_.begin());
	keptBytes_ += key.size();
	while (keptBytes_ > budget_ && !order_.empty()) {
		const Kept &last = order_.back();
		keptBytes_ -= last.bytes.size();
		byBytes_.erase(std::string_view(last.bytes));
		order_.pop_back();
	}
	return module;
}

uint64_t RgfxAirModuleCache::reads() const
{
	std::lock_guard<std::mutex> hold(lock_);
	return reads_;
}

size_t RgfxAirModuleCache::kept() const
{
	std::lock_guard<std::mutex> hold(lock_);
	return order_.size();
}

size_t RgfxAirModuleCache::keptBytes() const
{
	std::lock_guard<std::mutex> hold(lock_);
	return keptBytes_;
}

RgfxAirModuleCache &rgfxAirModules()
{
	static auto *modules = new RgfxAirModuleCache(kRgfxAirModuleCacheBytes);
	return *modules;
}

namespace {
// A universal .metallib's slice of generic AIR bitcode: the cputype its fat_arch entry carries.
// measured: macOS 26.6.1 x86_64 guest, QuartzCore's default.metallib (19 slices, one of them
// this, the rest code compiled for particular GPUs).
constexpr uint32_t kMetallibAirCpuType = 0x0100'0017;

uint32_t be32(const uint8_t *p)
{
	return (uint32_t(p[0]) << 24) | (uint32_t(p[1]) << 16) | (uint32_t(p[2]) << 8) | p[3];
}

uint64_t be64(const uint8_t *p)
{
	return (uint64_t(be32(p)) << 32) | be32(p + 4);
}
} // namespace

void rgfxMetallibAirSlice(const uint8_t *&bytes, size_t &size)
{
	const uint8_t *p = bytes;
	if (p == nullptr || size < 8)
		return;
	const uint32_t magic = be32(p);
	if (magic != 0xCAFE'BABE && magic != 0xCAFE'BABF)
		return;
	const bool wide = magic == 0xCAFE'BABF;
	const uint64_t entrySize = wide ? 32 : 20;
	const uint32_t count = be32(p + 4);
	if (count > (size - 8) / entrySize)
		return;
	for (uint32_t i = 0; i < count; i++) {
		const uint8_t *e = p + 8 + i * entrySize;
		if (be32(e) != kMetallibAirCpuType)
			continue;
		const uint64_t offset = wide ? be64(e + 8) : be32(e + 8);
		const uint64_t length = wide ? be64(e + 16) : be32(e + 12);
		if (offset > size || length > size - offset || length == 0)
			return;
		bytes = p + offset;
		size = static_cast<size_t>(length);
		return;
	}
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
		if (a.kind == RgfxAirKind::Texture && !a.anyIndex && index >= a.index &&
		    (a.anyCount || index - a.index < a.count) && writes(a.access))
			return true;
	}
	return false;
}

bool rgfxAirWritesTextureAnywhere(const RgfxAirFunction &fn)
{
	for (const RgfxAirArgument &a : fn.arguments) {
		if (a.kind == RgfxAirKind::Texture && a.anyIndex && writes(a.access))
			return true;
	}
	return false;
}
