/*
 * RgfxBinaryArchiveFile: see RgfxBinaryArchiveFile.hpp. Our own C++.
 */
#include "RgfxBinaryArchiveFile.hpp"

namespace {

constexpr char kMagic[8] = {'R', 'G', 'F', 'X', 'B', 'A', 'R', '1'};

// Metal's archive: a fat header, big-endian, of its magic and its slice count, then a 20-byte
// record per slice (CPU type, subtype, offset, size, alignment). Metal writes 0xcbfebabe; it reads
// the fat files of 0xcafebabe alike, as archives of other GPUs' slices (measured on the AMD 5300M
// and Intel 630 under macOS 26.7.1, oracle_bundle archive.*: IconRendering's binary.metallib, 18
// slices of CPU type 0x01000013, and an archive the device wrote with its magic changed, each an
// archive of no keys with no error; a file of no slices is "an invalid format").
constexpr uint32_t kMetalMagic = 0xcbfebabe;
constexpr uint32_t kFatMagic = 0xcafebabe;
constexpr size_t kMetalSliceBytes = 20;

uint32_t be32(const uint8_t *p)
{
	return (uint32_t(p[0]) << 24) | (uint32_t(p[1]) << 16) | (uint32_t(p[2]) << 8) | p[3];
}

// Reads little-endian fields from a byte range, failing once one would pass its end.
struct Reader {
	const uint8_t *p;
	size_t left;

	bool take(void *out, size_t n)
	{
		if (n > left)
			return false;
		std::memcpy(out, p, n);
		p += n;
		left -= n;
		return true;
	}
	bool u32(uint32_t &v) { return take(&v, sizeof(v)); }
};

void put32(std::vector<uint8_t> &out, uint32_t v)
{
	const uint8_t *b = reinterpret_cast<const uint8_t *>(&v);
	out.insert(out.end(), b, b + sizeof(v));
}

// Whether the bytes are a Metal archive whose every slice lies inside them.
bool metalArchive(const uint8_t *bytes, size_t length)
{
	if (length < 8 || (be32(bytes) != kMetalMagic && be32(bytes) != kFatMagic))
		return false;
	const uint32_t slices = be32(bytes + 4);
	if (slices == 0 || (length - 8) / kMetalSliceBytes < slices)
		return false;
	for (uint32_t i = 0; i < slices; i++) {
		const uint8_t *s = bytes + 8 + i * kMetalSliceBytes;
		const uint64_t offset = be32(s + 8), size = be32(s + 12);
		if (offset + size > length)
			return false;
	}
	return true;
}

// Whether the bytes are a whole .metallib: its magic, and the file size its header states (at
// 0x10) theirs. Both x86 Macs' Metal makes an archive of one (AMD 5300M and Intel 630, macOS
// 26.7.1: oracle_bundle archive.metallib), which holds no slice of this device's.
bool wholeMetallib(const uint8_t *bytes, size_t length)
{
	if (length < 0x18 || std::memcmp(bytes, "MTLB", 4) != 0)
		return false;
	uint64_t size = 0;
	std::memcpy(&size, bytes + 0x10, sizeof(size));
	return size == length;
}

} // namespace

RgfxArchiveFileKind rgfxReadArchiveFile(const uint8_t *bytes, size_t length,
                                        std::vector<RgfxArchiveSection> &sections)
{
	sections.clear();
	if (length >= sizeof(kMagic) && std::memcmp(bytes, kMagic, sizeof(kMagic)) == 0) {
		Reader r{bytes + sizeof(kMagic), length - sizeof(kMagic)};
		uint32_t count = 0;
		if (!r.u32(count))
			return RgfxArchiveFileKind::Invalid;
		std::vector<RgfxArchiveSection> read;
		for (uint32_t i = 0; i < count; i++) {
			RgfxArchiveSection s;
			uint32_t nameLength = 0, entries = 0;
			if (!r.u32(nameLength) || nameLength > r.left)
				return RgfxArchiveFileKind::Invalid;
			s.architecture.assign(reinterpret_cast<const char *>(r.p), nameLength);
			r.p += nameLength;
			r.left -= nameLength;
			if (!r.u32(entries) || entries > r.left / (1 + sizeof(RgfxArchiveEntry::key)))
				return RgfxArchiveFileKind::Invalid;
			for (uint32_t k = 0; k < entries; k++) {
				RgfxArchiveEntry e;
				if (!r.take(&e.kind, 1) || !r.take(e.key, sizeof(e.key)))
					return RgfxArchiveFileKind::Invalid;
				s.entries.insert(e);
			}
			read.push_back(std::move(s));
		}
		if (r.left != 0)
			return RgfxArchiveFileKind::Invalid;
		sections = std::move(read);
		return RgfxArchiveFileKind::Ours;
	}
	return metalArchive(bytes, length) || wholeMetallib(bytes, length) ? RgfxArchiveFileKind::Metal
	                                                                   : RgfxArchiveFileKind::Invalid;
}

std::vector<uint8_t> rgfxWriteArchiveFile(const std::vector<RgfxArchiveSection> &sections)
{
	std::vector<uint8_t> out(kMagic, kMagic + sizeof(kMagic));
	put32(out, static_cast<uint32_t>(sections.size()));
	for (const RgfxArchiveSection &s : sections) {
		put32(out, static_cast<uint32_t>(s.architecture.size()));
		out.insert(out.end(), s.architecture.begin(), s.architecture.end());
		put32(out, static_cast<uint32_t>(s.entries.size()));
		for (const RgfxArchiveEntry &e : s.entries) {
			out.push_back(e.kind);
			out.insert(out.end(), e.key, e.key + sizeof(e.key));
		}
	}
	return out;
}
