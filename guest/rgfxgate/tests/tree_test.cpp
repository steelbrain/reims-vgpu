// Native tests of the tree removal the boot-time install runs on its staging
// and swapped-out trees (src/rgfx_tree.h), over real directories in a scratch
// directory, with the file system calls a user process has in place of the
// kext's VNOP calls. Build and run with `make test`.
#include "rgfx_tree.h"

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <fcntl.h>
#include <string>
#include <sys/stat.h>
#include <unistd.h>

namespace {

// The file system as the walk sees it, through lstat, readdir, unlink and
// rmdir. Counts the removals it is asked for.
struct PosixFs {
	unsigned removes = 0;

	RgfxNodeKind kind(const char *path) {
		struct stat st;
		if (lstat(path, &st) != 0)
			return kRgfxNodeAbsent;
		return S_ISDIR(st.st_mode) ? kRgfxNodeDir : kRgfxNodeFile;
	}
	bool first(const char *dir, char *name, size_t cap) {
		DIR *d = opendir(dir);
		if (d == nullptr)
			return false;
		bool found = false;
		while (struct dirent *e = readdir(d)) {
			if (strcmp(e->d_name, ".") == 0 || strcmp(e->d_name, "..") == 0)
				continue;
			if (strlen(e->d_name) < cap) {
				strcpy(name, e->d_name);
				found = true;
			}
			break;
		}
		closedir(d);
		return found;
	}
	void remove(const char *path, RgfxNodeKind kind) {
		removes++;
		if (kind == kRgfxNodeDir)
			rmdir(path);
		else
			unlink(path);
	}
};

std::string scratch;

std::string at(const std::string &rel) { return scratch + "/" + rel; }

void mkdirs(const std::string &rel) {
	std::string p = scratch;
	size_t start = 0;
	while (start <= rel.size()) {
		size_t slash = rel.find('/', start);
		if (slash == std::string::npos)
			slash = rel.size();
		p += "/" + rel.substr(start, slash - start);
		mkdir(p.c_str(), 0755);
		start = slash + 1;
	}
}

void touch(const std::string &rel) {
	int fd = open(at(rel).c_str(), O_CREAT | O_WRONLY | O_TRUNC, 0644);
	assert(fd >= 0);
	const ssize_t wrote = write(fd, "x", 1);
	assert(wrote == 1);
	close(fd);
}

bool exists(const std::string &rel) {
	struct stat st;
	return lstat(at(rel).c_str(), &st) == 0;
}

bool removeTree(const std::string &rel, PosixFs &fs) {
	char path[1024];
	snprintf(path, sizeof(path), "%s", at(rel).c_str());
	const bool gone = rgfx_tree_remove(fs, path, sizeof(path));
	// The buffer holds the root again.
	assert(at(rel) == path);
	return gone;
}

// A record of the kernel's `struct direntry` layout (sys/dirent.h's
// __DARWIN_STRUCT_DIRENTRY), which a user process's 64-bit-inode
// `struct dirent` shares.
struct Direntry {
	uint64_t d_ino;
	uint64_t d_seekoff;
	uint16_t d_reclen;
	uint16_t d_namlen;
	uint8_t d_type;
	char d_name[1024];
};
static_assert(offsetof(Direntry, d_reclen) == offsetof(struct dirent, d_reclen), "d_reclen");
static_assert(offsetof(Direntry, d_namlen) == offsetof(struct dirent, d_namlen), "d_namlen");
static_assert(offsetof(Direntry, d_name) == offsetof(struct dirent, d_name), "d_name");

// Appends one record named `name`, `reclen` bytes long (0: the name's own
// length, rounded up to 8), to `buf`; returns its offset.
size_t record(std::string &buf, const char *name, size_t reclen = 0, size_t namlen = SIZE_MAX) {
	const size_t head = offsetof(Direntry, d_name);
	const size_t nl = namlen == SIZE_MAX ? strlen(name) : namlen;
	if (reclen == 0)
		reclen = (head + strlen(name) + 1 + 7) & ~size_t(7);
	const size_t start = buf.size();
	buf.resize(start + (reclen > head ? reclen : head), '\0');
	const uint16_t rl = static_cast<uint16_t>(reclen);
	const uint16_t n16 = static_cast<uint16_t>(nl);
	memcpy(&buf[start + offsetof(Direntry, d_reclen)], &rl, sizeof(rl));
	memcpy(&buf[start + offsetof(Direntry, d_namlen)], &n16, sizeof(n16));
	if (reclen > head)
		memcpy(&buf[start + head], name, std::min(strlen(name), reclen - head));
	return start;
}

std::string next(const std::string &buf, size_t *pos) {
	const char *name = nullptr;
	size_t nl = 0;
	if (!rgfx_dir_next<Direntry>(reinterpret_cast<const uint8_t *>(buf.data()), buf.size(), pos,
	                             &name, &nl))
		return "<end>";
	return std::string(name, nl);
}

void testRecords() {
	// "." and ".." are skipped; each other record is read in order.
	std::string buf;
	record(buf, ".");
	record(buf, "..");
	record(buf, "Contents");
	record(buf, "wire-version-1");
	size_t pos = 0;
	assert(next(buf, &pos) == "Contents");
	assert(next(buf, &pos) == "wire-version-1");
	assert(next(buf, &pos) == "<end>");
	assert(pos == buf.size());
	// Names that only start with dots are entries.
	buf.clear();
	record(buf, "...");
	record(buf, ".hidden");
	pos = 0;
	assert(next(buf, &pos) == "...");
	assert(next(buf, &pos) == ".hidden");
	// An empty read has no entry.
	pos = 0;
	assert(next(std::string(), &pos) == "<end>");
	// A record shorter than its header (a zero d_reclen among them) ends the
	// read, so a bad length cannot loop.
	buf.clear();
	record(buf, "a", 4);
	pos = 0;
	assert(next(buf, &pos) == "<end>");
	assert(pos == 0);
	// So does one that runs past the bytes read, or whose name runs past it.
	buf.clear();
	record(buf, "a");
	buf.resize(buf.size() - 1);
	pos = 0;
	assert(next(buf, &pos) == "<end>");
	buf.clear();
	record(buf, "a", 32, 200);
	pos = 0;
	assert(next(buf, &pos) == "<end>");
	// A record cut off at the end, after a good one, ends the read after it.
	buf.clear();
	record(buf, "good");
	buf.append(10, '\0');
	pos = 0;
	assert(next(buf, &pos) == "good");
	assert(next(buf, &pos) == "<end>");
}

// The bug the walk fixes: a tree holding a file the payload does not name, as
// a tree an older build staged does, is removed whole.
void testLeftoverOfAnOlderBuild() {
	PosixFs fs;
	mkdirs("t1/reims-graphics.bundle.new/Contents/MacOS");
	mkdirs("t1/reims-graphics.bundle.new/Contents/Resources");
	mkdirs("t1/reims-graphics.bundle.new/Contents/_CodeSignature");
	touch("t1/reims-graphics.bundle.new/Contents/Info.plist");
	touch("t1/reims-graphics.bundle.new/Contents/MacOS/reims-graphics");
	touch("t1/reims-graphics.bundle.new/Contents/Resources/wire-version-1");
	touch("t1/reims-graphics.bundle.new/Contents/Resources/dropped-by-the-new-build.metallib");
	touch("t1/reims-graphics.bundle.new/Contents/_CodeSignature/CodeResources");
	assert(removeTree("t1/reims-graphics.bundle.new", fs));
	assert(!exists("t1/reims-graphics.bundle.new"));
	// Its parent, the container, stays.
	assert(exists("t1"));
	// Nine nodes under the root, and the root.
	assert(fs.removes == 10);
}

void testShapes() {
	PosixFs fs;
	// An absent root is gone already, and nothing is removed.
	assert(removeTree("absent", fs));
	assert(fs.removes == 0);
	// A file at the root is removed.
	touch("lone-file");
	assert(removeTree("lone-file", fs));
	assert(!exists("lone-file"));
	// Empty directories, nested and side by side, go too.
	mkdirs("t2/a/b/c");
	mkdirs("t2/d");
	mkdirs("t2/e/f");
	touch("t2/e/f/g");
	assert(removeTree("t2", fs));
	assert(!exists("t2"));
}

// A symbolic link in the tree is removed itself, and what it names stays.
void testLinksAreNotFollowed() {
	PosixFs fs;
	mkdirs("outside/kept");
	touch("outside/kept/file");
	mkdirs("t3/Contents");
	const int dirLink = symlink(at("outside/kept").c_str(), at("t3/Contents/dir-link").c_str());
	const int fileLink = symlink(at("outside/kept/file").c_str(), at("t3/file-link").c_str());
	assert(dirLink == 0 && fileLink == 0);
	assert(removeTree("t3", fs));
	assert(!exists("t3"));
	assert(exists("outside/kept/file"));
}

// A tree deeper than the walk descends stops it, and the walk says so; what
// lies outside the tree is untouched.
void testTooDeep() {
	PosixFs fs;
	std::string deep = "t4";
	for (unsigned i = 0; i < kRgfxTreeMaxDepth + 1; i++)
		deep += "/d";
	mkdirs(deep);
	touch(deep + "/file");
	touch("t4-beside");
	assert(!removeTree("t4", fs));
	assert(exists("t4"));
	assert(exists("t4-beside"));
	// The deepest directory the walk reaches is left, with what is under it.
	assert(exists(deep + "/file"));
}

// A node the walk cannot remove ends it at once, the root kept.
void testStuck() {
	PosixFs fs;
	mkdirs("t5/locked");
	touch("t5/locked/file");
	chmod(at("t5/locked").c_str(), 0555);
	const bool gone = removeTree("t5", fs);
	chmod(at("t5/locked").c_str(), 0755);
	if (geteuid() != 0) {
		assert(!gone);
		assert(exists("t5/locked/file"));
		// One try at the file, and none after it.
		assert(fs.removes == 1);
	}
	PosixFs again;
	assert(removeTree("t5", again));
}

// A path that does not fit the buffer stops the walk where it is.
void testPathTooLong() {
	PosixFs fs;
	mkdirs("t6");
	std::string longName(200, 'n');
	touch("t6/" + longName);
	char path[1024];
	snprintf(path, sizeof(path), "%s", at("t6").c_str());
	const size_t cap = strlen(path) + 1 + 10; // room for a short name only
	assert(!rgfx_tree_remove(fs, path, cap));
	assert(at("t6") == path);
	assert(exists("t6/" + longName));
	assert(removeTree("t6", fs));
}

} // namespace

int main() {
	char tmpl[] = "/tmp/rgfx-tree-test.XXXXXX";
	const char *made = mkdtemp(tmpl);
	assert(made != nullptr);
	scratch = tmpl;
	testRecords();
	testLeftoverOfAnOlderBuild();
	testShapes();
	testLinksAreNotFollowed();
	testTooDeep();
	testStuck();
	testPathTooLong();
	std::string cleanup = "rm -rf '" + scratch + "'";
	const int cleaned = system(cleanup.c_str());
	assert(cleaned == 0);
	printf("tree_test: ok\n");
	return 0;
}
