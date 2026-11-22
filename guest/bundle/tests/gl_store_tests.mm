/*
 * gl_store_tests: the store of lowered GL shader variants (src/RgfxGLStore.*), over a directory
 * of the test's own and a lowering of the test's own (gl_tests.mm's rgfxGLTests calls
 * rgfxGLStoreTests; guest/bundle/Makefile `test`). A variant from the store is the lowering's
 * byte for byte, in a later store over the same directory too, as in an app's next run; an entry
 * that does not read whole, hash to its hash, hold its key or decode is lowered again, and the
 * first failure is said once; a read or write the system refuses turns the store off; a write is
 * a whole entry or none; past the cap the least recently used go first. A lowering's write is
 * queued, so its caller waits for none; queued writes keep the cap, hold at most its bytes, and a
 * write the process ends before is lost alone. A process's memo answers
 * a variant asked for again with the bytes it answered first, the oldest going past its cap. No GL
 * generator: the guest's runs check the store under the real lowering.
 */
#include <dirent.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <unistd.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#import <Foundation/Foundation.h>

#include "RgfxGLStore.hpp"
#include "rgfx_test_ring.hpp" // CHECK, g_checks

void rgfxGLStoreTests();

namespace {

// A directory of the test's own, removed with what it holds when the test ends.
struct ScratchDirectory {
	std::string path;
	ScratchDirectory()
	{
		const char *base = getenv("TMPDIR");
		std::string t = std::string(base != nullptr && *base != 0 ? base : "/tmp") + "/rgfx-gl-store.XXXXXX";
		std::vector<char> buf(t.begin(), t.end());
		buf.push_back(0);
		CHECK(mkdtemp(buf.data()) != nullptr);
		path = buf.data();
	}
	~ScratchDirectory() { removeTree(path); }
	static void removeTree(const std::string &p)
	{
		struct stat st;
		if (lstat(p.c_str(), &st) != 0)
			return;
		if (S_ISDIR(st.st_mode)) {
			if (DIR *d = opendir(p.c_str())) {
				while (struct dirent *e = readdir(d)) {
					const std::string name = e->d_name;
					if (name != "." && name != "..")
						removeTree(p + "/" + name);
				}
				closedir(d);
			}
			rmdir(p.c_str());
		} else
			unlink(p.c_str());
	}
};

std::vector<std::string> listNames(const std::string &dir)
{
	std::vector<std::string> out;
	if (DIR *d = opendir(dir.c_str())) {
		while (struct dirent *e = readdir(d)) {
			const std::string name = e->d_name;
			if (name != "." && name != "..")
				out.push_back(name);
		}
		closedir(d);
	}
	return out;
}

std::vector<uint8_t> readFile(const std::string &path)
{
	std::vector<uint8_t> out;
	if (FILE *f = std::fopen(path.c_str(), "rb")) {
		int c;
		while ((c = std::fgetc(f)) != EOF)
			out.push_back(static_cast<uint8_t>(c));
		std::fclose(f);
	}
	return out;
}

void writeFile(const std::string &path, const std::vector<uint8_t> &bytes)
{
	FILE *f = std::fopen(path.c_str(), "wb");
	CHECK(f != nullptr);
	CHECK(std::fwrite(bytes.data(), 1, bytes.size(), f) == bytes.size());
	std::fclose(f);
}

void setTime(const std::string &path, time_t seconds)
{
	struct timeval t[2] = {{seconds, 0}, {seconds, 0}};
	CHECK(utimes(path.c_str(), t) == 0);
}

bool sameLowered(const RgfxGLLowered &a, const RgfxGLLowered &b)
{
	return a.metallib == b.metallib && a.entry == b.entry && a.stage == b.stage && a.returnData == b.returnData &&
	       a.stageInText == b.stageInText;
}

// A lowering of the test's own: what it makes depends on `seed` alone; `calls` counts its runs.
struct FakeLowering {
	int calls = 0;
	uint8_t seed = 1;
	bool ok = true;
	bool operator()(RgfxGLLowered &out, std::string &why)
	{
		calls++;
		if (!ok) {
			why = "the fake lowering failed";
			return false;
		}
		out.metallib.assign(4096 + seed, seed);
		std::memcpy(out.metallib.data(), "MTLB", 4);
		out.entry = "__main";
		out.stage = seed % 2 ? RgfxMetallibStage::Fragment : RgfxMetallibStage::Kernel;
		out.returnData.assign(100 + seed, static_cast<uint8_t>(seed ^ 0x5a));
		out.stageInText = seed % 3 == 0 ? std::string("; a kernel's text ") + std::to_string(seed) : std::string();
		return true;
	}
	std::function<bool(RgfxGLLowered &, std::string &)> fn()
	{
		return [this](RgfxGLLowered &out, std::string &why) { return (*this)(out, why); };
	}
};

std::string keyOf(uint8_t n, const std::string &identity = "bundle 1\n")
{
	const uint8_t ir[4] = {1, 2, 3, n};
	const uint8_t plugin[3] = {9, 8, n};
	return RgfxGLVariantKey(identity, ir, sizeof ir, plugin, sizeof plugin, 2, 1);
}

// RgfxGLLowerKept, then the store's queued writes done, as a later process finds them.
bool lowerKeptDone(RgfxGLStore *store, const std::string &key,
                   const std::function<bool(RgfxGLLowered &, std::string &)> &lower, RgfxGLLowered &out,
                   std::string &why, bool *hit)
{
	const bool made = RgfxGLLowerKept(store, key, lower, out, why, hit);
	store->drain();
	return made;
}

void test_keys()
{
	const uint8_t ab[2] = {'a', 'b'}, c[1] = {'c'}, a[1] = {'a'}, bc[2] = {'b', 'c'};
	const std::string id = "bundle 1\n";
	CHECK(RgfxGLVariantKey(id, ab, 2, c, 1, 2, 1) == RgfxGLVariantKey(id, ab, 2, c, 1, 2, 1));
	// A field's bytes never run into the next's.
	CHECK(RgfxGLVariantKey(id, ab, 2, c, 1, 2, 1) != RgfxGLVariantKey(id, a, 1, bc, 2, 2, 1));
	CHECK(RgfxGLVariantKey(id, ab, 2, c, 1, 2, 1) != RgfxGLVariantKey("bundle 2\n", ab, 2, c, 1, 2, 1));
	CHECK(RgfxGLVariantKey(id, ab, 2, c, 1, 2, 1) != RgfxGLVariantKey(id, ab, 2, c, 1, 3, 1));
	CHECK(RgfxGLVariantKey(id, ab, 2, c, 1, 2, 1) != RgfxGLVariantKey(id, ab, 2, c, 1, 2, 2));
	CHECK(RgfxGLVariantKey(id, ab, 2, nullptr, 0, 2, 1) != RgfxGLVariantKey(id, ab, 2, c, 1, 2, 1));
	const std::string v = RgfxGLVariantKey(id, ab, 2, c, 1, 2, 1);
	CHECK(RgfxGLStageInKey(v, "d1") == RgfxGLStageInKey(v, "d1"));
	CHECK(RgfxGLStageInKey(v, "d1") != RgfxGLStageInKey(v, "d2"));
	CHECK(RgfxGLStageInKey(v, "d1") != v);
	CHECK(RgfxGLStageInKey(v, "d1") != RgfxGLStageInKey(RgfxGLVariantKey(id, a, 1, bc, 2, 2, 1), "d1"));
	RgfxGLStore s("/nonexistent", 1);
	CHECK(s.pathOf(v) == s.pathOf(v) && s.pathOf(v) != s.pathOf(RgfxGLStageInKey(v, "d1")));
	CHECK(s.pathOf(v).size() == std::string("/nonexistent/").size() + 64 + 4);
}

void test_value_codec()
{
	FakeLowering f;
	for (uint8_t seed : {1, 2, 3, 6}) {
		f.seed = seed;
		RgfxGLLowered l, back;
		std::string why;
		CHECK(f(l, why));
		const std::vector<uint8_t> v = RgfxGLEncodeLowered(l);
		CHECK(RgfxGLDecodeLowered(v.data(), v.size(), back));
		CHECK(sameLowered(l, back));
		// One byte short, one byte over, another magic, another stage: none decode, and `out` is
		// left as it was.
		RgfxGLLowered untouched = back;
		CHECK(!RgfxGLDecodeLowered(v.data(), v.size() - 1, back));
		std::vector<uint8_t> over = v;
		over.push_back(0);
		CHECK(!RgfxGLDecodeLowered(over.data(), over.size(), back));
		std::vector<uint8_t> magic = v;
		magic[0] ^= 1;
		CHECK(!RgfxGLDecodeLowered(magic.data(), magic.size(), back));
		std::vector<uint8_t> stage = v;
		stage[8] = 9;
		CHECK(!RgfxGLDecodeLowered(stage.data(), stage.size(), back));
		CHECK(sameLowered(untouched, back));
	}
	RgfxGLLowered none;
	CHECK(!RgfxGLDecodeLowered(nullptr, 0, none));
}

// A hit answers what the miss lowered, byte for byte, in this store and in a store of another
// process's (a new store over the same directory); the lowering runs once.
void test_hit_is_miss()
{
	ScratchDirectory scratch;
	const std::string dir = scratch.path + "/app/ai.aneesiqbal.reims-graphics/gl";
	FakeLowering f;
	f.seed = 6;
	RgfxGLStore store(dir, 1 << 20, "bundle 1\n");
	RgfxGLLowered missed, hit, again;
	std::string why;
	bool wasHit = true;
	CHECK(lowerKeptDone(&store, keyOf(1), f.fn(), missed, why, &wasHit));
	CHECK(!wasHit && f.calls == 1);
	CHECK(listNames(dir).size() == 1);
	CHECK(lowerKeptDone(&store, keyOf(1), f.fn(), hit, why, &wasHit));
	CHECK(wasHit && f.calls == 1);
	CHECK(sameLowered(missed, hit));
	CHECK(RgfxGLEncodeLowered(missed) == RgfxGLEncodeLowered(hit));
	RgfxGLStore next(dir, 1 << 20, "bundle 1\n");
	CHECK(lowerKeptDone(&next, keyOf(1), f.fn(), again, why, &wasHit));
	CHECK(wasHit && f.calls == 1 && sameLowered(missed, again));
	// Another key misses, and the lowering it makes is kept beside the first.
	f.seed = 2;
	CHECK(lowerKeptDone(&next, keyOf(2), f.fn(), again, why, &wasHit));
	CHECK(!wasHit && f.calls == 2 && !sameLowered(missed, again));
	CHECK(listNames(dir).size() == 2);
	// A failed lowering keeps nothing, and answers its reason.
	f.ok = false;
	why.clear();
	CHECK(!lowerKeptDone(&next, keyOf(3), f.fn(), again, why, &wasHit));
	CHECK(why == "the fake lowering failed" && listNames(dir).size() == 2);
	// No store lowers each time.
	f.ok = true;
	const int before = f.calls;
	CHECK(RgfxGLLowerKept(nullptr, keyOf(1), f.fn(), again, why, &wasHit));
	CHECK(RgfxGLLowerKept(nullptr, keyOf(1), f.fn(), again, why, &wasHit));
	CHECK(!wasHit && f.calls == before + 2);
	std::string failure;
	CHECK(!store.takeFirstFailure(failure) && !next.takeFirstFailure(failure));
}

// An entry that does not read whole, does not hash to its hash, holds another key or holds no
// variant is removed and lowered again; the store says its first failure once.
void test_bad_entries_lower_again()
{
	ScratchDirectory scratch;
	const std::string dir = scratch.path + "/gl";
	FakeLowering f;
	RgfxGLStore store(dir, 1 << 20);
	RgfxGLLowered first, out;
	std::string why, failure;
	bool hit = false;
	CHECK(lowerKeptDone(&store, keyOf(1), f.fn(), first, why, &hit));
	const std::string path = store.pathOf(keyOf(1));
	const std::vector<uint8_t> good = readFile(path);
	CHECK(good.size() > 56);

	// A flipped byte in the value.
	std::vector<uint8_t> bad = good;
	bad[good.size() / 2] ^= 0x40;
	writeFile(path, bad);
	CHECK(lowerKeptDone(&store, keyOf(1), f.fn(), out, why, &hit));
	CHECK(!hit && f.calls == 2 && sameLowered(first, out));
	CHECK(store.takeFirstFailure(failure) && failure.find("does not hash to its hash") != std::string::npos);
	CHECK(!store.takeFirstFailure(failure));
	CHECK(readFile(path) == good); // written again, whole

	// Short: a write cut off, and an empty file.
	writeFile(path, std::vector<uint8_t>(good.begin(), good.end() - 7));
	CHECK(lowerKeptDone(&store, keyOf(1), f.fn(), out, why, &hit));
	CHECK(!hit && f.calls == 3 && sameLowered(first, out));
	writeFile(path, {});
	CHECK(lowerKeptDone(&store, keyOf(1), f.fn(), out, why, &hit));
	CHECK(!hit && f.calls == 4 && sameLowered(first, out));
	CHECK(readFile(path) == good);

	// Another key's entry under this key's name.
	f.seed = 2;
	RgfxGLLowered second;
	CHECK(lowerKeptDone(&store, keyOf(2), f.fn(), second, why, &hit));
	writeFile(path, readFile(store.pathOf(keyOf(2))));
	f.seed = 1;
	CHECK(lowerKeptDone(&store, keyOf(1), f.fn(), out, why, &hit));
	CHECK(!hit && f.calls == 6 && sameLowered(first, out));

	// An entry, whole and hashed, whose value is no variant.
	CHECK(store.put(keyOf(1), {1, 2, 3}));
	CHECK(lowerKeptDone(&store, keyOf(1), f.fn(), out, why, &hit));
	CHECK(!hit && f.calls == 7 && sameLowered(first, out));
	CHECK(lowerKeptDone(&store, keyOf(1), f.fn(), out, why, &hit));
	CHECK(hit && f.calls == 7 && sameLowered(first, out));

	// The first failure only, once: a fresh store over the same bad bytes says it again.
	CHECK(!store.takeFirstFailure(failure));
	writeFile(path, bad);
	RgfxGLStore other(dir, 1 << 20);
	CHECK(lowerKeptDone(&other, keyOf(1), f.fn(), out, why, &hit));
	CHECK(!hit && other.takeFirstFailure(failure) && !other.takeFirstFailure(failure));
}

// Where the store cannot write, every variant is lowered as without one, and the store says so
// once.
void test_unwritable_store_lowers()
{
	ScratchDirectory scratch;
	writeFile(scratch.path + "/file", {1});
	RgfxGLStore store(scratch.path + "/file/gl", 1 << 20);
	FakeLowering f;
	RgfxGLLowered a, b;
	std::string why, failure;
	bool hit = true;
	CHECK(lowerKeptDone(&store, keyOf(1), f.fn(), a, why, &hit));
	CHECK(!hit && f.calls == 1);
	CHECK(lowerKeptDone(&store, keyOf(1), f.fn(), b, why, &hit));
	CHECK(!hit && f.calls == 2 && sameLowered(a, b));
	CHECK(store.takeFirstFailure(failure) && failure.find("could not make") != std::string::npos);
	CHECK(!store.takeFirstFailure(failure));
	CHECK(!store.put(keyOf(2), {1}));
	CHECK(!store.takeFirstFailure(failure));
}

// A read the system refuses turns the store off: it reads and writes nothing more in the process,
// and the variants are lowered as without it.
void test_refused_read_turns_store_off()
{
	ScratchDirectory scratch;
	const std::string dir = scratch.path + "/gl";
	RgfxGLStore store(dir, 1 << 20);
	FakeLowering f;
	RgfxGLLowered a, b;
	std::string why, failure;
	bool hit = true;
	CHECK(lowerKeptDone(&store, keyOf(1), f.fn(), a, why, &hit));
	const std::string path = store.pathOf(keyOf(1));
	CHECK(chmod(path.c_str(), 0) == 0);
	CHECK(lowerKeptDone(&store, keyOf(1), f.fn(), b, why, &hit));
	CHECK(!hit && f.calls == 2 && sameLowered(a, b));
	CHECK(store.takeFirstFailure(failure) && failure.find("could not read") != std::string::npos);
	CHECK(chmod(path.c_str(), 0644) == 0);
	// Off: the entry, readable again, is not read, and a new one is not written.
	CHECK(lowerKeptDone(&store, keyOf(1), f.fn(), b, why, &hit));
	CHECK(!hit && f.calls == 3);
	CHECK(lowerKeptDone(&store, keyOf(2), f.fn(), b, why, &hit));
	CHECK(!hit && f.calls == 4 && listNames(dir).size() == 1);
	CHECK(!store.takeFirstFailure(failure));
	// The next process's store reads it.
	RgfxGLStore next(dir, 1 << 20);
	CHECK(lowerKeptDone(&next, keyOf(1), f.fn(), b, why, &hit));
	CHECK(hit && f.calls == 4 && sameLowered(a, b));
}

// Past the cap, the least recently used entries go first, until three quarters of it are left; a
// read marks its entry used. A file that is no entry (a write a dead process left) goes too once it
// is an hour old, and a newer one stays.
void test_cap_drops_least_recently_used()
{
	ScratchDirectory scratch;
	const std::string dir = scratch.path + "/gl";
	std::vector<uint8_t> value(1000, 7);
	RgfxGLStore probe(dir, 1 << 20);
	CHECK(probe.put(keyOf(0), value));
	const uint64_t entry = readFile(probe.pathOf(keyOf(0))).size();
	unlink(probe.pathOf(keyOf(0)).c_str());

	// Room for four entries.
	RgfxGLStore store(dir, entry * 4 + entry / 2);
	struct timeval now;
	gettimeofday(&now, nullptr);
	const time_t t0 = now.tv_sec - 10000;
	for (uint8_t n = 1; n <= 4; n++) {
		CHECK(store.put(keyOf(n), value));
		setTime(store.pathOf(keyOf(n)), t0 + n);
	}
	CHECK(listNames(dir).size() == 4);
	// Entry 1, the oldest, is read: it is the newest now.
	std::vector<uint8_t> got;
	CHECK(store.get(keyOf(1), got) && got == value);
	writeFile(dir + "/.tmp.1.0", {1});
	setTime(dir + "/.tmp.1.0", t0);
	writeFile(dir + "/.tmp.2.0", {1});
	// A fifth passes the cap: entries 2 and 3 go (three quarters of the cap hold three).
	CHECK(store.put(keyOf(5), value));
	struct stat st;
	CHECK(stat(store.pathOf(keyOf(1)).c_str(), &st) == 0);
	CHECK(stat(store.pathOf(keyOf(2)).c_str(), &st) != 0);
	CHECK(stat(store.pathOf(keyOf(3)).c_str(), &st) != 0);
	CHECK(stat(store.pathOf(keyOf(4)).c_str(), &st) == 0);
	CHECK(stat(store.pathOf(keyOf(5)).c_str(), &st) == 0);
	CHECK(stat((dir + "/.tmp.1.0").c_str(), &st) != 0);
	CHECK(stat((dir + "/.tmp.2.0").c_str(), &st) == 0);
	unlink((dir + "/.tmp.2.0").c_str());
	CHECK(listNames(dir).size() == 3);
	// A store opened later counts what is there before it writes.
	RgfxGLStore later(dir, entry * 4 + entry / 2);
	CHECK(later.put(keyOf(6), value));
	CHECK(listNames(dir).size() == 4);
	CHECK(later.put(keyOf(7), value));
	CHECK(listNames(dir).size() == 3);
}

// A lowering's write is queued: the lowering returns before its entry is written, and a store over
// the same directory meanwhile misses it. Writes a process ends before (held here, as in a process
// that ends with them queued, with the next process's store opened meanwhile) lose only their own
// entries: the entries written before read whole, and the lost ones are lowered again.
void test_queued_writes_lose_only_their_own()
{
	ScratchDirectory scratch;
	const std::string dir = scratch.path + "/gl";
	FakeLowering f;
	RgfxGLLowered first, second, out;
	std::string why, failure;
	bool hit = true;
	RgfxGLStore store(dir, 1 << 20);
	CHECK(lowerKeptDone(&store, keyOf(1), f.fn(), first, why, &hit) && !hit);
	store.holdWrites(true);
	f.seed = 2;
	CHECK(RgfxGLLowerKept(&store, keyOf(2), f.fn(), second, why, &hit) && !hit && f.calls == 2);
	store.drain(); // waits for none while held
	CHECK(listNames(dir).size() == 1);

	RgfxGLStore next(dir, 1 << 20);
	CHECK(lowerKeptDone(&next, keyOf(1), f.fn(), out, why, &hit) && hit && f.calls == 2);
	CHECK(sameLowered(first, out));
	CHECK(lowerKeptDone(&next, keyOf(2), f.fn(), out, why, &hit) && !hit && f.calls == 3);
	CHECK(sameLowered(second, out));
	CHECK(listNames(dir).size() == 2);

	// The held write, let go, writes the same entry over the next store's, whole.
	const std::vector<uint8_t> written = readFile(next.pathOf(keyOf(2)));
	store.holdWrites(false);
	store.drain();
	CHECK(readFile(store.pathOf(keyOf(2))) == written);
	CHECK(listNames(dir).size() == 2);
	CHECK(!store.takeFirstFailure(failure) && !next.takeFirstFailure(failure));
}

// Queued writes keep the cap and its order: past it the least recently used go first. The writes
// queued at once hold at most the cap's bytes, and one past them is dropped, its variant lowered
// again in a later run.
void test_queued_writes_keep_the_cap()
{
	ScratchDirectory scratch;
	const std::string dir = scratch.path + "/gl";
	const std::vector<uint8_t> value(3000, 7);
	const uint64_t cap = 10000; // three of these writes, queued or kept, and not four
	RgfxGLStore store(dir, cap);
	store.holdWrites(true);
	for (uint8_t n = 1; n <= 4; n++)
		store.putLater(keyOf(n), value);
	store.holdWrites(false);
	store.drain();
	struct stat st;
	CHECK(listNames(dir).size() == 3);
	CHECK(stat(store.pathOf(keyOf(4)).c_str(), &st) != 0);

	// Entry 2 is read: entries 1 and 3 are the least recently used, and a fourth entry, past the
	// cap, drops them (three quarters of the cap hold two).
	struct timeval now;
	gettimeofday(&now, nullptr);
	for (uint8_t n = 1; n <= 3; n++)
		setTime(store.pathOf(keyOf(n)), now.tv_sec - 10000 + n);
	std::vector<uint8_t> got;
	CHECK(store.get(keyOf(2), got) && got == value);
	store.putLater(keyOf(4), value);
	store.drain();
	CHECK(stat(store.pathOf(keyOf(1)).c_str(), &st) != 0);
	CHECK(stat(store.pathOf(keyOf(2)).c_str(), &st) == 0);
	CHECK(stat(store.pathOf(keyOf(3)).c_str(), &st) != 0);
	CHECK(stat(store.pathOf(keyOf(4)).c_str(), &st) == 0);
	CHECK(listNames(dir).size() == 2);
	std::string failure;
	CHECK(!store.takeFirstFailure(failure));
}

// Writers and readers of one directory at once, in threads as in processes, writing now and
// queued: every read answers a whole value of its key, and no temporary file is left.
void test_concurrent_writes_are_whole()
{
	ScratchDirectory scratch;
	const std::string dir = scratch.path + "/gl";
	RgfxGLStore a(dir, 64 << 20), b(dir, 64 << 20);
	std::atomic<int> wrong{0}, reads{0};
	std::vector<std::thread> threads;
	for (int t = 0; t < 8; t++)
		threads.emplace_back([&, t] {
			RgfxGLStore &s = t % 2 ? a : b;
			for (int i = 0; i < 200; i++) {
				const uint8_t n = static_cast<uint8_t>(i % 5);
				const std::vector<uint8_t> value(20000 + n * 1000, n);
				if (t < 2)
					s.put(keyOf(n), value);
				else if (t < 4)
					s.putLater(keyOf(n), value);
				std::vector<uint8_t> got;
				if (s.get(keyOf(n), got)) {
					reads++;
					if (got != value)
						wrong++;
				}
			}
		});
	for (auto &t : threads)
		t.join();
	a.drain();
	b.drain();
	CHECK(wrong == 0 && reads > 0);
	for (const auto &name : listNames(dir))
		CHECK(name.size() == 68 && name.compare(64, 4, ".rgl") == 0);
	CHECK(listNames(dir).size() == 5);
	std::string failure;
	CHECK(!a.takeFirstFailure(failure) && !b.takeFirstFailure(failure));
}

// A variant asked for again in a process is the memo's, the bytes first answered, with no store
// and with one; past the memo's cap the oldest go first and are lowered again.
void test_memo_answers_repeats()
{
	FakeLowering f;
	RgfxGLMemo memo(64 << 20);
	RgfxGLLowered a, b;
	std::string why;
	bool hit = true;
	CHECK(RgfxGLLowerKept(nullptr, keyOf(1), f.fn(), a, why, &hit, &memo));
	CHECK(!hit && f.calls == 1);
	f.seed = 9; // a lowering that would answer other bytes now
	CHECK(RgfxGLLowerKept(nullptr, keyOf(1), f.fn(), b, why, &hit, &memo));
	CHECK(hit && f.calls == 1 && sameLowered(a, b));

	// With a store: its entry is read once, then the memo answers, the entry's file gone or not.
	ScratchDirectory scratch;
	const std::string dir = scratch.path + "/gl";
	{
		RgfxGLStore first(dir, 64 << 20);
		FakeLowering g;
		RgfxGLLowered c;
		CHECK(RgfxGLLowerKept(&first, keyOf(2), g.fn(), c, why, &hit));
		CHECK(!hit && g.calls == 1);
	}
	RgfxGLStore store(dir, 64 << 20);
	FakeLowering g;
	RgfxGLMemo kept(64 << 20);
	RgfxGLLowered c, d;
	CHECK(RgfxGLLowerKept(&store, keyOf(2), g.fn(), c, why, &hit, &kept));
	CHECK(hit && g.calls == 0);
	CHECK(unlink(store.pathOf(keyOf(2)).c_str()) == 0);
	CHECK(RgfxGLLowerKept(&store, keyOf(2), g.fn(), d, why, &hit, &kept));
	CHECK(hit && g.calls == 0 && sameLowered(c, d));

	// The cap: an entry of keyOf's key and FakeLowering's seed 1 is about 4.3 KB, so a memo of
	// 10000 bytes holds two, and a third drops the first.
	FakeLowering h;
	RgfxGLMemo small(10000);
	RgfxGLLowered e;
	for (uint8_t n : {1, 2, 3})
		CHECK(RgfxGLLowerKept(nullptr, keyOf(n), h.fn(), e, why, &hit, &small) && !hit);
	CHECK(h.calls == 3 && small.bytes() > 8000 && small.bytes() <= 10000);
	CHECK(RgfxGLLowerKept(nullptr, keyOf(3), h.fn(), e, why, &hit, &small) && hit && h.calls == 3);
	CHECK(RgfxGLLowerKept(nullptr, keyOf(2), h.fn(), e, why, &hit, &small) && hit && h.calls == 3);
	CHECK(RgfxGLLowerKept(nullptr, keyOf(1), h.fn(), e, why, &hit, &small) && !hit && h.calls == 4);
	// A variant larger than the cap is answered, not kept.
	RgfxGLMemo tiny(100);
	CHECK(RgfxGLLowerKept(nullptr, keyOf(1), h.fn(), e, why, &hit, &tiny) && !hit);
	CHECK(tiny.bytes() == 0);
	// A failed lowering keeps nothing.
	FakeLowering bad;
	bad.ok = false;
	RgfxGLMemo none(64 << 20);
	CHECK(!RgfxGLLowerKept(nullptr, keyOf(4), bad.fn(), e, why, &hit, &none) && none.bytes() == 0);
}

} // namespace

void rgfxGLStoreTests()
{
	@autoreleasepool {
		test_keys();
		test_value_codec();
		test_hit_is_miss();
		test_bad_entries_lower_again();
		test_unwritable_store_lowers();
		test_refused_read_turns_store_off();
		test_cap_drops_least_recently_used();
		test_queued_writes_lose_only_their_own();
		test_queued_writes_keep_the_cap();
		test_concurrent_writes_are_whole();
		test_memo_answers_repeats();
	}
}
