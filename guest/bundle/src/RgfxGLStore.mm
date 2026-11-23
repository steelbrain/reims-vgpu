/*
 * RgfxGLStore: see RgfxGLStore.hpp. Its files are read and written through the bundle's own file
 * calls (RgfxFiles.hpp), never Foundation's, so a sandbox that lists the system calls a process
 * may make takes every call the store makes (tests/files_tests.mm).
 */
#include "RgfxGLStore.hpp"

#include <dispatch/dispatch.h>

#include <CommonCrypto/CommonDigest.h>

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <tuple>

#include "RgfxFiles.hpp"

namespace {

constexpr char kEntryMagic[8] = {'R', 'G', 'F', 'X', 'G', 'L', 'S', '1'};
constexpr char kValueMagic[8] = {'R', 'G', 'F', 'X', 'G', 'L', 'V', '1'};
constexpr char kKeyMagic[8] = {'R', 'G', 'F', 'X', 'G', 'L', 'K', '1'};
constexpr size_t kHeader = sizeof kEntryMagic + 16; // magic, key length, value length
constexpr size_t kHash = CC_SHA256_DIGEST_LENGTH;
constexpr char kSuffix[] = ".rgl";
constexpr size_t kNameLength = 2 * kHash + sizeof kSuffix - 1;
// A file of the directory that is no entry (a write's own file, RgfxWriteFileWhole's, left by a
// process that ended in it) goes at the next trim once it is this old.
constexpr double kStaleFile = 3600;

void append(std::string &key, char tag, const void *bytes, size_t length)
{
	key.push_back(tag);
	const uint64_t n = length;
	key.append(reinterpret_cast<const char *>(&n), sizeof n);
	if (length != 0)
		key.append(static_cast<const char *>(bytes), length);
}

void appendBytes(std::vector<uint8_t> &out, const void *bytes, size_t length)
{
	const uint64_t n = length;
	const uint8_t *p = reinterpret_cast<const uint8_t *>(&n);
	out.insert(out.end(), p, p + sizeof n);
	if (length != 0)
		out.insert(out.end(), static_cast<const uint8_t *>(bytes), static_cast<const uint8_t *>(bytes) + length);
}

// Reads a u64 length and that many bytes at `p` (`n` left) into `out`; false where they pass the end.
template <typename T>
bool takeBytes(const uint8_t *&p, size_t &n, T &out)
{
	uint64_t length = 0;
	if (n < sizeof length)
		return false;
	std::memcpy(&length, p, sizeof length);
	p += sizeof length;
	n -= sizeof length;
	if (length > n)
		return false;
	out.assign(reinterpret_cast<const typename T::value_type *>(p),
	           reinterpret_cast<const typename T::value_type *>(p) + length);
	p += length;
	n -= length;
	return true;
}

void sha256(const uint8_t *p, size_t n, uint8_t out[kHash])
{
	CC_SHA256_CTX ctx;
	CC_SHA256_Init(&ctx);
	while (n > 0) {
		const CC_LONG take = n > 0x4000'0000 ? 0x4000'0000 : static_cast<CC_LONG>(n);
		CC_SHA256_Update(&ctx, p, take);
		p += take;
		n -= take;
	}
	CC_SHA256_Final(out, &ctx);
}

std::string describe(const char *what, const std::string &path, int error)
{
	return std::string(what) + " " + path + ": " + std::strerror(error);
}

// A call's errno that says the file is not there (as against one the system refused).
bool isMissing(int error)
{
	return error == ENOENT || error == ENOTDIR;
}

} // namespace

std::string RgfxGLVariantKey(const std::string &identity, const uint8_t *ir, size_t irLength, const uint8_t *plugin,
                             size_t pluginLength, uint8_t bitcodeType, uint64_t functionType)
{
	std::string key(kKeyMagic, sizeof kKeyMagic);
	append(key, 'V', nullptr, 0);
	append(key, 'I', identity.data(), identity.size());
	append(key, 'R', ir, irLength);
	append(key, 'P', plugin, pluginLength);
	append(key, 'B', &bitcodeType, sizeof bitcodeType);
	append(key, 'F', &functionType, sizeof functionType);
	return key;
}

std::string RgfxGLStageInKey(const std::string &variantKey, const std::string &stageInKey)
{
	std::string key(kKeyMagic, sizeof kKeyMagic);
	append(key, 'S', nullptr, 0);
	append(key, 'K', variantKey.data(), variantKey.size());
	append(key, 'D', stageInKey.data(), stageInKey.size());
	return key;
}

std::vector<uint8_t> RgfxGLEncodeLowered(const RgfxGLLowered &lowered)
{
	std::vector<uint8_t> out(kValueMagic, kValueMagic + sizeof kValueMagic);
	const uint32_t stage = static_cast<uint32_t>(lowered.stage);
	const uint8_t *s = reinterpret_cast<const uint8_t *>(&stage);
	out.insert(out.end(), s, s + sizeof stage);
	appendBytes(out, lowered.metallib.data(), lowered.metallib.size());
	appendBytes(out, lowered.entry.data(), lowered.entry.size());
	appendBytes(out, lowered.returnData.data(), lowered.returnData.size());
	appendBytes(out, lowered.stageInText.data(), lowered.stageInText.size());
	return out;
}

bool RgfxGLDecodeLowered(const uint8_t *value, size_t length, RgfxGLLowered &out)
{
	uint32_t stage = 0;
	if (value == nullptr || length < sizeof kValueMagic + sizeof stage ||
	    std::memcmp(value, kValueMagic, sizeof kValueMagic) != 0)
		return false;
	std::memcpy(&stage, value + sizeof kValueMagic, sizeof stage);
	if (stage != static_cast<uint32_t>(RgfxMetallibStage::Vertex) &&
	    stage != static_cast<uint32_t>(RgfxMetallibStage::Fragment) &&
	    stage != static_cast<uint32_t>(RgfxMetallibStage::Kernel))
		return false;
	const uint8_t *p = value + sizeof kValueMagic + sizeof stage;
	size_t n = length - sizeof kValueMagic - sizeof stage;
	RgfxGLLowered l;
	l.stage = static_cast<RgfxMetallibStage>(stage);
	if (!takeBytes(p, n, l.metallib) || !takeBytes(p, n, l.entry) || !takeBytes(p, n, l.returnData) ||
	    !takeBytes(p, n, l.stageInText) || n != 0 || l.metallib.empty() || l.entry.empty())
		return false;
	out = std::move(l);
	return true;
}

struct RgfxGLStore::Writes {
	dispatch_queue_t queue;
	bool held = false; // guarded by the store's lock_
};

RgfxGLStore::RgfxGLStore(std::string directory, uint64_t capBytes, std::string identity)
    : directory_(std::move(directory)), cap_(capBytes), identity_(std::move(identity)),
      writes_(new Writes{dispatch_queue_create(
          "ai.aneesiqbal.reims-graphics.gl-store",
          dispatch_queue_attr_make_with_qos_class(DISPATCH_QUEUE_SERIAL, QOS_CLASS_UTILITY, 0))})
{
}

RgfxGLStore::~RgfxGLStore()
{
	holdWrites(false);
	drain();
}

void RgfxGLStore::putLater(std::string key, std::vector<uint8_t> value)
{
	if (off())
		return;
	const uint64_t size = key.size() + value.size();
	if (queued_.fetch_add(size) + size > cap_) {
		queued_ -= size;
		return;
	}
	auto *write = new std::pair<std::string, std::vector<uint8_t>>(std::move(key), std::move(value));
	dispatch_async(writes_->queue, ^{
		put(write->first, write->second);
		queued_ -= size;
		delete write;
	});
}

void RgfxGLStore::drain()
{
	{
		std::lock_guard<std::mutex> hold(lock_);
		if (writes_->held)
			return;
	}
	dispatch_sync(writes_->queue, ^{});
}

void RgfxGLStore::holdWrites(bool held)
{
	std::lock_guard<std::mutex> hold(lock_);
	if (writes_->held == held)
		return;
	writes_->held = held;
	if (held)
		dispatch_suspend(writes_->queue);
	else
		dispatch_resume(writes_->queue);
}

std::string RgfxGLStore::pathOf(const std::string &key) const
{
	uint8_t h[kHash];
	sha256(reinterpret_cast<const uint8_t *>(key.data()), key.size(), h);
	static const char kHex[] = "0123456789abcdef";
	std::string name;
	for (uint8_t b : h) {
		name.push_back(kHex[b >> 4]);
		name.push_back(kHex[b & 15]);
	}
	return directory_ + "/" + name + kSuffix;
}

void RgfxGLStore::fail(const std::string &why)
{
	std::lock_guard<std::mutex> hold(lock_);
	if (failure_.empty())
		failure_ = why;
}

void RgfxGLStore::refuse(const std::string &why)
{
	std::lock_guard<std::mutex> hold(lock_);
	if (failure_.empty())
		failure_ = why;
	off_ = true;
}

bool RgfxGLStore::off()
{
	return off_;
}

bool RgfxGLStore::takeFirstFailure(std::string &why)
{
	std::lock_guard<std::mutex> hold(lock_);
	if (failure_.empty() || failureTaken_)
		return false;
	failureTaken_ = true;
	why = failure_;
	return true;
}

bool RgfxGLStore::get(const std::string &key, std::vector<uint8_t> &value)
{
	if (off())
		return false;
	const std::string path = pathOf(key);
	std::vector<uint8_t> data;
	if (const int error = RgfxReadFile(path, data)) {
		if (!isMissing(error))
			refuse(describe("could not read", path, error));
		return false;
	}
	const uint8_t *bytes = data.data();
	const size_t size = data.size();
	uint64_t keyLength = 0, valueLength = 0;
	std::string bad;
	if (size < kHeader + kHash)
		bad = "is shorter than an entry";
	else {
		std::memcpy(&keyLength, bytes + sizeof kEntryMagic, sizeof keyLength);
		std::memcpy(&valueLength, bytes + sizeof kEntryMagic + 8, sizeof valueLength);
		uint8_t h[kHash];
		if (std::memcmp(bytes, kEntryMagic, sizeof kEntryMagic) != 0)
			bad = "is no entry";
		else if (keyLength > size || valueLength > size || kHeader + keyLength + valueLength + kHash != size)
			bad = "is not as long as its lengths say";
		else {
			sha256(bytes, size - kHash, h);
			if (std::memcmp(h, bytes + size - kHash, kHash) != 0)
				bad = "does not hash to its hash";
			else if (keyLength != key.size() || std::memcmp(bytes + kHeader, key.data(), key.size()) != 0)
				bad = "holds another key";
		}
	}
	if (!bad.empty()) {
		RgfxRemoveFile(path);
		fail("the entry " + path + " " + bad + "; removed");
		return false;
	}
	// Used now: past the cap the least recently used go first. The entry's first byte, its magic's,
	// written over itself (RgfxTouchFile).
	RgfxTouchFile(path, static_cast<uint8_t>(kEntryMagic[0]));
	value.assign(bytes + kHeader + keyLength, bytes + kHeader + keyLength + valueLength);
	return true;
}

bool RgfxGLStore::put(const std::string &key, const std::vector<uint8_t> &value)
{
	if (off())
		return false;
	std::vector<uint8_t> data;
	data.reserve(kHeader + key.size() + value.size() + kHash);
	const uint64_t lengths[2] = {key.size(), value.size()};
	const uint8_t *l = reinterpret_cast<const uint8_t *>(lengths);
	data.insert(data.end(), kEntryMagic, kEntryMagic + sizeof kEntryMagic);
	data.insert(data.end(), l, l + sizeof lengths);
	data.insert(data.end(), key.begin(), key.end());
	data.insert(data.end(), value.begin(), value.end());
	uint8_t h[kHash];
	sha256(data.data(), data.size(), h);
	data.insert(data.end(), h, h + kHash);

	{
		std::lock_guard<std::mutex> hold(lock_);
		if (!made_) {
			if (const int error = RgfxMakeDirectories(directory_)) {
				if (failure_.empty())
					failure_ = describe("could not make", directory_, error);
				off_ = true;
				return false;
			}
			made_ = true;
		}
	}
	// Written to a file of its own and renamed into place (RgfxWriteFileWhole): a reader sees the
	// old entry or the new one, whole.
	const std::string path = pathOf(key);
	if (const int error = RgfxWriteFileWhole(path, data.data(), data.size())) {
		refuse(describe("could not write", path, error));
		return false;
	}
	std::lock_guard<std::mutex> hold(lock_);
	bytes_ += data.size();
	if (!counted_ || bytes_ > cap_)
		trim();
	return true;
}

void RgfxGLStore::trim()
{
	std::vector<RgfxDirectoryFile> files;
	if (const int error = RgfxListFiles(directory_, files)) {
		if (failure_.empty())
			failure_ = describe("could not list", directory_, error);
		return;
	}
	std::vector<std::tuple<double, uint64_t, std::string>> entries;
	uint64_t total = 0;
	const double now = RgfxFileClockNow();
	for (const RgfxDirectoryFile &f : files) {
		const std::string path = directory_ + "/" + f.name;
		if (f.name.size() != kNameLength || f.name.compare(2 * kHash, std::string::npos, kSuffix) != 0) {
			if (now - f.modified > kStaleFile)
				RgfxRemoveFile(path);
			continue;
		}
		entries.emplace_back(f.modified, f.size, path);
		total += f.size;
	}
	counted_ = true;
	bytes_ = total;
	if (total <= cap_)
		return;
	std::sort(entries.begin(), entries.end(),
	          [](const auto &a, const auto &b) { return std::get<0>(a) < std::get<0>(b); });
	const uint64_t keep = cap_ / 4 * 3;
	for (const auto &e : entries) {
		if (bytes_ <= keep)
			break;
		const int gone = RgfxRemoveFile(std::get<2>(e));
		if (gone == 0 || isMissing(gone))
			bytes_ -= std::get<1>(e);
	}
}

namespace {

uint64_t sizeOf(const std::string &key, const RgfxGLLowered &l)
{
	return key.size() + l.metallib.size() + l.entry.size() + l.returnData.size() + l.stageInText.size();
}

} // namespace

RgfxGLMemo::RgfxGLMemo(uint64_t capBytes) : cap_(capBytes) {}

bool RgfxGLMemo::get(const std::string &key, RgfxGLLowered &out)
{
	std::lock_guard<std::mutex> hold(lock_);
	auto it = entries_.find(key);
	if (it == entries_.end())
		return false;
	out = it->second;
	return true;
}

void RgfxGLMemo::put(const std::string &key, const RgfxGLLowered &lowered)
{
	const uint64_t size = sizeOf(key, lowered);
	std::lock_guard<std::mutex> hold(lock_);
	if (size > cap_)
		return;
	auto made = entries_.emplace(key, lowered);
	if (!made.second)
		return;
	order_.push_back(&made.first->first);
	bytes_ += size;
	while (bytes_ > cap_ && !order_.empty()) {
		auto oldest = entries_.find(*order_.front());
		order_.pop_front();
		bytes_ -= sizeOf(oldest->first, oldest->second);
		entries_.erase(oldest);
	}
}

uint64_t RgfxGLMemo::bytes()
{
	std::lock_guard<std::mutex> hold(lock_);
	return bytes_;
}

bool RgfxGLLowerKept(RgfxGLStore *store, const std::string &key,
                     const std::function<bool(RgfxGLLowered &, std::string &)> &lower, RgfxGLLowered &out,
                     std::string &why, bool *hit, RgfxGLMemo *memo)
{
	if (hit != nullptr)
		*hit = false;
	if (memo != nullptr && memo->get(key, out)) {
		if (hit != nullptr)
			*hit = true;
		return true;
	}
	std::vector<uint8_t> value;
	if (store != nullptr && store->get(key, value)) {
		if (RgfxGLDecodeLowered(value.data(), value.size(), out)) {
			if (memo != nullptr)
				memo->put(key, out);
			if (hit != nullptr)
				*hit = true;
			return true;
		}
		store->fail("the entry " + store->pathOf(key) + " holds no lowered variant");
	}
	RgfxGLLowered lowered;
	if (!lower(lowered, why))
		return false;
	if (store != nullptr)
		store->putLater(key, RgfxGLEncodeLowered(lowered));
	if (memo != nullptr)
		memo->put(key, lowered);
	out = std::move(lowered);
	return true;
}
