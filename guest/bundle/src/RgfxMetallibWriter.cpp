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

std::vector<uint8_t> RgfxWriteMetallib(const RgfxMetallibFunction &f)
{
	std::vector<uint8_t> section;
	put32(section, kWrapperMagic);
	put32(section, 0);
	put32(section, kWrapperBytes);
	put32(section, static_cast<uint32_t>(f.bitcodeLength));
	put32(section, 0xffffffffu);
	put(section, f.bitcode, f.bitcodeLength);
	while (section.size() % 16 != 0)
		section.push_back(0);
	uint8_t hash[CC_SHA256_DIGEST_LENGTH];
	CC_SHA256(section.data(), static_cast<CC_LONG>(section.size()), hash);

	std::vector<uint8_t> tags;
	tag(tags, "NAME", f.name.c_str(), static_cast<uint16_t>(f.name.size() + 1));
	const uint8_t type = static_cast<uint8_t>(f.stage);
	tag(tags, "TYPE", &type, 1);
	tag(tags, "HASH", hash, sizeof hash);
	const uint8_t offsets[24] = {};
	tag(tags, "OFFT", offsets, sizeof offsets);
	std::vector<uint8_t> vers;
	put16(vers, f.airMajor);
	put16(vers, f.airMinor);
	put16(vers, f.languageMajor);
	put16(vers, f.languageMinor);
	tag(tags, "VERS", vers.data(), static_cast<uint16_t>(vers.size()));
	std::vector<uint8_t> mdsz;
	put64(mdsz, section.size());
	tag(tags, "MDSZ", mdsz.data(), static_cast<uint16_t>(mdsz.size()));
	put(tags, "ENDT", 4);
	std::vector<uint8_t> group;
	put32(group, static_cast<uint32_t>(4 + tags.size()));
	put(group, tags.data(), tags.size());

	std::vector<uint8_t> extension;
	tag(extension, "UUID", hash, 16);
	put(extension, "ENDT", 4);

	const uint64_t listOffset = kHeaderBytes, listSize = group.size();
	const uint64_t publicOffset = listOffset + 4 + group.size() + extension.size();
	const uint64_t privateOffset = publicOffset + 8, bitcodeOffset = privateOffset + 8;
	const uint64_t size = bitcodeOffset + section.size();
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
	put64(out, 8);
	put64(out, privateOffset);
	put64(out, 8);
	put64(out, bitcodeOffset);
	put64(out, section.size());
	put32(out, 1);
	put(out, group.data(), group.size());
	put(out, extension.data(), extension.size());
	for (int i = 0; i < 2; i++) {
		put32(out, 8);
		put(out, "ENDT", 4);
	}
	put(out, section.data(), section.size());
	return out;
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
