/*
 * RgfxMetallibWriter: see RgfxMetallibWriter.hpp. Our own C++.
 */
#include "RgfxMetallibWriter.hpp"

#include <CommonCrypto/CommonDigest.h>

#include <cstring>

namespace {

constexpr uint32_t kWrapperMagic = 0x0B17C0DE;
constexpr uint32_t kWrapperBytes = 20;
constexpr uint64_t kHeaderBytes = 0x58;

void put(std::vector<uint8_t> &d, const void *p, size_t n)
{
	const uint8_t *b = static_cast<const uint8_t *>(p);
	d.insert(d.end(), b, b + n);
}

void put16(std::vector<uint8_t> &d, uint16_t v)
{
	const uint8_t b[2] = {static_cast<uint8_t>(v), static_cast<uint8_t>(v >> 8)};
	put(d, b, 2);
}

void put32(std::vector<uint8_t> &d, uint32_t v)
{
	for (int i = 0; i < 4; i++)
		d.push_back(static_cast<uint8_t>(v >> (8 * i)));
}

void put64(std::vector<uint8_t> &d, uint64_t v)
{
	for (int i = 0; i < 8; i++)
		d.push_back(static_cast<uint8_t>(v >> (8 * i)));
}

void tag(std::vector<uint8_t> &d, const char name[4], const void *p, uint16_t n)
{
	put(d, name, 4);
	put16(d, n);
	put(d, p, n);
}

uint32_t le32(const uint8_t *p)
{
	return static_cast<uint32_t>(p[0]) | static_cast<uint32_t>(p[1]) << 8 | static_cast<uint32_t>(p[2]) << 16 |
	       static_cast<uint32_t>(p[3]) << 24;
}

} // namespace

std::vector<uint8_t> RgfxWriteMetallib(const std::vector<RgfxMetallibFunction> &functions)
{
	// Each function's bitcode, wrapped and padded, in turn; its tag group places it.
	std::vector<uint8_t> sections;
	std::vector<uint8_t> groups;
	uint64_t metadataAt = 0;
	for (const RgfxMetallibFunction &f : functions) {
		const uint64_t at = sections.size();
		put32(sections, kWrapperMagic);
		put32(sections, 0);
		put32(sections, kWrapperBytes);
		put32(sections, static_cast<uint32_t>(f.bitcodeLength));
		put32(sections, 0xffffffffu);
		put(sections, f.bitcode, f.bitcodeLength);
		while (sections.size() % 16 != 0)
			sections.push_back(0);
		const uint64_t size = sections.size() - at;
		uint8_t hash[CC_SHA256_DIGEST_LENGTH];
		CC_SHA256(sections.data() + at, static_cast<CC_LONG>(size), hash);

		std::vector<uint8_t> tags;
		tag(tags, "NAME", f.name.c_str(), static_cast<uint16_t>(f.name.size() + 1));
		const uint8_t type = static_cast<uint8_t>(f.stage);
		tag(tags, "TYPE", &type, 1);
		tag(tags, "HASH", hash, sizeof hash);
		std::vector<uint8_t> offsets;
		put64(offsets, metadataAt);
		put64(offsets, metadataAt);
		put64(offsets, at);
		tag(tags, "OFFT", offsets.data(), static_cast<uint16_t>(offsets.size()));
		std::vector<uint8_t> vers;
		put16(vers, f.airMajor);
		put16(vers, f.airMinor);
		put16(vers, f.languageMajor);
		put16(vers, f.languageMinor);
		tag(tags, "VERS", vers.data(), static_cast<uint16_t>(vers.size()));
		std::vector<uint8_t> mdsz;
		put64(mdsz, size);
		tag(tags, "MDSZ", mdsz.data(), static_cast<uint16_t>(mdsz.size()));
		put(tags, "ENDT", 4);
		put32(groups, static_cast<uint32_t>(4 + tags.size()));
		put(groups, tags.data(), tags.size());
		metadataAt += 8;
	}
	uint8_t hash[CC_SHA256_DIGEST_LENGTH];
	CC_SHA256(sections.data(), static_cast<CC_LONG>(sections.size()), hash);
	std::vector<uint8_t> extension;
	tag(extension, "UUID", hash, 16);
	put(extension, "ENDT", 4);

	const uint64_t metadataSize = metadataAt;
	const uint64_t listOffset = kHeaderBytes, listSize = groups.size();
	const uint64_t publicOffset = listOffset + 4 + groups.size() + extension.size();
	const uint64_t privateOffset = publicOffset + metadataSize, bitcodeOffset = privateOffset + metadataSize;
	const uint64_t size = bitcodeOffset + sections.size();
	std::vector<uint8_t> out;
	out.reserve(size);
	put(out, "MTLB", 4);
	put16(out, 0x8001);
	put16(out, 2);
	put16(out, 5);
	const uint8_t zeros[6] = {};
	put(out, zeros, sizeof zeros);
	put64(out, size);
	put64(out, listOffset);
	put64(out, listSize);
	put64(out, publicOffset);
	put64(out, metadataSize);
	put64(out, privateOffset);
	put64(out, metadataSize);
	put64(out, bitcodeOffset);
	put64(out, sections.size());
	put32(out, static_cast<uint32_t>(functions.size()));
	put(out, groups.data(), groups.size());
	put(out, extension.data(), extension.size());
	for (int i = 0; i < 2; i++) {
		for (size_t n = 0; n < functions.size(); n++) {
			put32(out, 8);
			put(out, "ENDT", 4);
		}
	}
	put(out, sections.data(), sections.size());
	return out;
}

std::vector<uint8_t> RgfxWriteMetallib(const RgfxMetallibFunction &f)
{
	return RgfxWriteMetallib(std::vector<RgfxMetallibFunction>{f});
}

bool RgfxMetallibOfClLibrary(const uint8_t *bytes, size_t length, std::vector<uint8_t> &out)
{
	constexpr size_t kPairsAt = 0x10, kPairs = 10, kEntryWords = 7;
	if (bytes == nullptr || length < kPairsAt + 8 * kPairs || le32(bytes) != 1 || le32(bytes + 4) != 0 ||
	    le32(bytes + 8) != 0 || le32(bytes + 12) != 0)
		return false;
	// A section's bytes, where its pair places them inside `bytes`.
	auto section = [&](size_t n, const uint8_t **at, size_t *size) {
		const uint64_t offset = le32(bytes + kPairsAt + 8 * n), bytesOf = le32(bytes + kPairsAt + 8 * n + 4);
		if (offset > length || bytesOf > length - offset)
			return false;
		*at = bytes + offset;
		*size = static_cast<size_t>(bytesOf);
		return true;
	};
	const uint8_t *table = nullptr, *modules = nullptr;
	size_t tableSize = 0, modulesSize = 0;
	if (!section(0, &table, &tableSize) || !section(3, &modules, &modulesSize) || tableSize < 12)
		return false;
	const uint16_t head[4] = {8, 2, 1, 1};
	for (int i = 0; i < 4; i++)
		if ((table[2 * i] | table[2 * i + 1] << 8) != head[i])
			return false;
	const uint32_t count = le32(table + 8);
	if (count == 0)
		return false;
	std::vector<RgfxMetallibFunction> functions;
	size_t at = 12;
	for (uint32_t i = 0; i < count; i++) {
		if (tableSize - at < 4 * kEntryWords + 1)
			return false;
		const uint8_t *entry = table + at;
		const uint8_t *name = entry + 4 * kEntryWords;
		const size_t nameLength = strnlen(reinterpret_cast<const char *>(name), tableSize - at - 4 * kEntryWords);
		if (nameLength == 0 || nameLength == tableSize - at - 4 * kEntryWords)
			return false;
		at += 4 * kEntryWords + nameLength + 1;
		const uint32_t stage = le32(entry), module = le32(entry + 16);
		if (stage != static_cast<uint32_t>(RgfxMetallibStage::Kernel) || module > modulesSize ||
		    modulesSize - module < 4)
			return false;
		const uint32_t moduleSize = le32(modules + module);
		if (moduleSize > modulesSize - module - 4)
			return false;
		size_t bitcodeLength = moduleSize;
		const uint8_t *wrapped = modules + module + 4;
		const uint8_t *bitcode = RgfxUnwrapBitcode(wrapped, &bitcodeLength);
		if (bitcode == wrapped)
			return false;
		RgfxMetallibFunction f;
		f.name.assign(reinterpret_cast<const char *>(name), nameLength);
		f.stage = RgfxMetallibStage::Kernel;
		f.bitcode = bitcode;
		f.bitcodeLength = bitcodeLength;
		functions.push_back(f);
	}
	out = RgfxWriteMetallib(functions);
	return true;
}

const uint8_t *RgfxUnwrapBitcode(const uint8_t *bitcode, size_t *length)
{
	if (bitcode == nullptr || *length < kWrapperBytes || le32(bitcode) != kWrapperMagic)
		return bitcode;
	const uint32_t offset = le32(bitcode + 8), size = le32(bitcode + 12);
	if (offset > *length || size > *length - offset)
		return bitcode;
	*length = size;
	return bitcode + offset;
}
