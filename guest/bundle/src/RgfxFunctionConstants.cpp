/*
 * RgfxFunctionConstants: see RgfxFunctionConstants.hpp. Our own C++.
 */

#include "RgfxFunctionConstants.hpp"

#include <cstring>

namespace {

// One family of MTLDataType's function-constant types: its scalar's raw value (its two-, three-
// and four-lane vectors follow it in a row), the size of a lane, and its name.
struct Family {
	uint32_t scalar;
	uint32_t lane;
	const char *name;
};

// MTLDataType's raw values (Metal's MTLDataType.h).
constexpr Family kFamilies[] = {
	{ 3, 4, "Float" },  { 16, 2, "Half" },  { 29, 4, "Int" },   { 33, 4, "UInt" },
	{ 37, 2, "Short" }, { 41, 2, "UShort" }, { 45, 1, "Char" },  { 49, 1, "UChar" },
	{ 53, 1, "Bool" },  { 81, 8, "Long" },  { 85, 8, "ULong" },
};

// The family `type` is a scalar or vector of, and its lane count; null for any other type.
const Family *familyOf(uint32_t type, uint32_t *lanes)
{
	for (const Family &f : kFamilies) {
		if (type >= f.scalar && type < f.scalar + 4) {
			*lanes = type - f.scalar + 1;
			return &f;
		}
	}
	return nullptr;
}

uint16_t rd16(const uint8_t *p)
{
	uint16_t v;
	std::memcpy(&v, p, sizeof(v));
	return v;
}

uint32_t rd32(const uint8_t *p)
{
	uint32_t v;
	std::memcpy(&v, p, sizeof(v));
	return v;
}

uint64_t rd64(const uint8_t *p)
{
	uint64_t v;
	std::memcpy(&v, p, sizeof(v));
	return v;
}

// A .metallib header's public metadata section: its offset and size (at 0x28 and 0x30).
constexpr size_t kPublicMetadataOffset = 0x28;
constexpr size_t kPublicMetadataSize = 0x30;

// Reads one `CNST` tag's entries: a u16 count, then per constant its NUL-terminated name, a u8
// MTLDataType, a u16 index and a u8 flag, 1 where the constant is required (measured on the
// macOS 27 compiler's output against Metal's MTLFunctionConstant.required).
bool readCnst(const uint8_t *p, size_t n, std::vector<RgfxConstantDecl> &out)
{
	if (n < 2)
		return false;
	uint16_t count = rd16(p);
	size_t at = 2;
	for (uint16_t i = 0; i < count; i++) {
		const void *nul = at < n ? std::memchr(p + at, 0, n - at) : nullptr;
		if (nul == nullptr)
			return false;
		size_t nameLen = static_cast<const uint8_t *>(nul) - (p + at);
		RgfxConstantDecl d;
		d.name.assign(reinterpret_cast<const char *>(p + at), nameLen);
		at += nameLen + 1;
		if (n - at < 4)
			return false;
		d.type = p[at];
		d.index = rd16(p + at + 1);
		d.required = p[at + 3] != 0;
		at += 4;
		out.push_back(std::move(d));
	}
	return true;
}

} // namespace

uint32_t rgfxConstantSize(uint32_t type)
{
	uint32_t lanes = 0;
	const Family *f = familyOf(type, &lanes);
	return f ? f->lane * lanes : 0;
}

namespace {

// MTLDataType's values that are no scalar or vector of a family, and their names after
// "MTLDataType" (Metal's MTLDataType.h, macOS 27 SDK). Metal's errors name every value of the
// enum so (measured: oracle_names on the AMD Radeon Pro 5300M and the Intel UHD Graphics 630,
// macOS 26.7.1).
struct Named {
	uint32_t type;
	const char *name;
};
constexpr Named kNamed[] = {
	{ 0, "None" },
	{ 1, "Struct" },
	{ 2, "Array" },
	{ 7, "Float2x2" },
	{ 8, "Float2x3" },
	{ 9, "Float2x4" },
	{ 10, "Float3x2" },
	{ 11, "Float3x3" },
	{ 12, "Float3x4" },
	{ 13, "Float4x2" },
	{ 14, "Float4x3" },
	{ 15, "Float4x4" },
	{ 20, "Half2x2" },
	{ 21, "Half2x3" },
	{ 22, "Half2x4" },
	{ 23, "Half3x2" },
	{ 24, "Half3x3" },
	{ 25, "Half3x4" },
	{ 26, "Half4x2" },
	{ 27, "Half4x3" },
	{ 28, "Half4x4" },
	{ 58, "Texture" },
	{ 59, "Sampler" },
	{ 60, "Pointer" },
	{ 62, "R8Unorm" },
	{ 63, "R8Snorm" },
	{ 64, "R16Unorm" },
	{ 65, "R16Snorm" },
	{ 66, "RG8Unorm" },
	{ 67, "RG8Snorm" },
	{ 68, "RG16Unorm" },
	{ 69, "RG16Snorm" },
	{ 70, "RGBA8Unorm" },
	{ 71, "RGBA8Unorm_sRGB" },
	{ 72, "RGBA8Snorm" },
	{ 73, "RGBA16Unorm" },
	{ 74, "RGBA16Snorm" },
	{ 75, "RGB10A2Unorm" },
	{ 76, "RG11B10Float" },
	{ 77, "RGB9E5Float" },
	{ 78, "RenderPipeline" },
	{ 79, "ComputePipeline" },
	{ 80, "IndirectCommandBuffer" },
	{ 115, "VisibleFunctionTable" },
	{ 116, "IntersectionFunctionTable" },
	{ 117, "PrimitiveAccelerationStructure" },
	{ 118, "InstanceAccelerationStructure" },
	{ 121, "BFloat" },
	{ 122, "BFloat2" },
	{ 123, "BFloat3" },
	{ 124, "BFloat4" },
};

} // namespace

std::string rgfxDataTypeName(uint32_t type)
{
	uint32_t lanes = 0;
	const Family *f = familyOf(type, &lanes);
	if (f != nullptr) {
		std::string name = std::string("MTLDataType") + f->name;
		if (lanes > 1)
			name += std::to_string(lanes);
		return name;
	}
	for (const Named &n : kNamed) {
		if (n.type == type)
			return std::string("MTLDataType") + n.name;
	}
	// A value outside the enum: ours.
	return "MTLDataType(" + std::to_string(type) + ")";
}

bool rgfxReadConstantTable(const uint8_t *bytes, size_t length, uint64_t offset,
                           std::vector<RgfxConstantDecl> &out)
{
	out.clear();
	if (bytes == nullptr || length < kPublicMetadataSize + 8)
		return false;
	const uint64_t section = rd64(bytes + kPublicMetadataOffset);
	const uint64_t sectionSize = rd64(bytes + kPublicMetadataSize);
	if (section > length || sectionSize > length - section || offset > sectionSize ||
	    sectionSize - offset < 4)
		return false;
	// The entry: a u32 size, then its tags (a 4-byte name, a u16 size and the data) up to
	// `ENDT`. The size counts its own 4 bytes in a metallib built for macOS 27, and not in one
	// built for macOS 26 (measured: Apple metal 32023.922, -mmacosx-version-min 27.2 and 26.0,
	// the same functions), so the tags are read up to the 4 bytes past it.
	const uint8_t *entry = bytes + section + offset;
	const uint64_t room = sectionSize - offset;
	const uint32_t size = rd32(entry);
	if (size < 4 || size > room)
		return false;
	const size_t end = static_cast<size_t>(size + 4 <= room ? size + 4 : size);
	size_t at = 4;
	while (end - at >= 4) {
		if (std::memcmp(entry + at, "ENDT", 4) == 0)
			return true;
		if (end - at < 6)
			break;
		const uint16_t tagSize = rd16(entry + at + 4);
		if (tagSize > end - at - 6)
			break;
		if (std::memcmp(entry + at, "CNST", 4) == 0 && !readCnst(entry + at + 6, tagSize, out))
			break;
		at += 6 + static_cast<size_t>(tagSize);
	}
	out.clear();
	return false;
}

bool rgfxResolveConstants(const std::vector<RgfxConstantDecl> &decls,
                          const std::map<uint32_t, RgfxConstantValue> &byIndex,
                          const std::map<std::string, RgfxConstantValue> &byName,
                          std::vector<RgfxFunctionConstant> &out, RgfxConstantMismatch *mismatch)
{
	out.clear();
	for (size_t i = 0; i < decls.size(); i++) {
		const RgfxConstantDecl &d = decls[i];
		const RgfxConstantValue *v = nullptr;
		auto at = byIndex.find(d.index);
		if (at != byIndex.end()) {
			v = &at->second;
		} else {
			auto named = byName.find(d.name);
			if (named != byName.end())
				v = &named->second;
		}
		if (v != nullptr && v->type != d.type) {
			if (mismatch != nullptr) {
				mismatch->decl = i;
				mismatch->valueType = v->type;
			}
			out.clear();
			return false;
		}
		if (v == nullptr && !d.required)
			continue; // not defined
		RgfxFunctionConstant k = {};
		k.index = d.index;
		k.data_type = d.type;
		if (v != nullptr) {
			const size_t n = v->bytes.size() < sizeof(k.value) ? v->bytes.size() : sizeof(k.value);
			std::memcpy(k.value, v->bytes.data(), n);
		}
		out.push_back(k);
	}
	return true;
}
