// Native tests of the link the boot-time install keeps from the HEVC encoder
// plug-in's driver path to the bundle's executable (src/rgfx_alias.h), over
// real files in a scratch directory, with the file system calls a user process
// has in place of the kext's VNOP calls. Build and run with `make test`.
#include "rgfx_alias.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <string>
#include <sys/stat.h>
#include <unistd.h>

namespace {

// The file system as the update sees it, through lstat, unlink, mkdir and
// link. Counts the removals and tags it is asked for.
struct PosixFs {
	unsigned removes = 0;
	unsigned tags = 0;

	bool file_id(const char *path, uint64_t *id) {
		struct stat st;
		if (lstat(path, &st) != 0)
			return false;
		*id = (uint64_t)st.st_ino;
		return true;
	}
	void remove(const char *path) {
		removes++;
		unlink(path);
	}
	bool ensure_dir(const char *path) {
		struct stat st;
		if (mkdir(path, 0755) == 0)
			return true;
		return lstat(path, &st) == 0 && S_ISDIR(st.st_mode);
	}
	bool link(const char *target, const char *path) { return ::link(target, path) == 0; }
	void tag(const char *) { tags++; }
};

std::string g_root;

std::string at(const char *rel) { return g_root + "/" + rel; }

void writeFile(const std::string &path, const char *bytes) {
	const int fd = open(path.c_str(), O_CREAT | O_TRUNC | O_WRONLY, 0644);
	assert(fd >= 0);
	assert(write(fd, bytes, strlen(bytes)) == (ssize_t)strlen(bytes));
	close(fd);
}

// A file written aside and renamed over `path`: a new inode, as each install
// writes the executable.
void replaceFile(const std::string &path, const char *bytes) {
	const std::string aside = path + ".new";
	writeFile(aside, bytes);
	assert(rename(aside.c_str(), path.c_str()) == 0);
}

uint64_t inode(const std::string &path) {
	struct stat st;
	return lstat(path.c_str(), &st) == 0 ? (uint64_t)st.st_ino : 0;
}

bool isFile(const std::string &path) {
	struct stat st;
	return lstat(path.c_str(), &st) == 0 && S_ISREG(st.st_mode);
}

struct Paths {
	std::string target, bundle, contents, macos, link;
	const char *dirs[3];
	Paths()
	    : target(at("reims-graphics")), bundle(at("Driver.bundle")),
	      contents(at("Driver.bundle/Contents")), macos(at("Driver.bundle/Contents/MacOS")),
	      link(at("Driver.bundle/Contents/MacOS/Driver")) {
		dirs[0] = bundle.c_str();
		dirs[1] = contents.c_str();
		dirs[2] = macos.c_str();
	}
	RgfxAliasResult update(PosixFs &fs) const {
		return rgfx_alias_update(fs, target.c_str(), dirs, 3, link.c_str());
	}
};

// No executable: nothing is made.
void testNoTarget() {
	Paths p;
	PosixFs fs;
	assert(p.update(fs) == kRgfxAliasNoTarget);
	assert(inode(p.bundle) == 0 && fs.removes == 0 && fs.tags == 0);
}

// A first update makes the directories and the link, the executable's own
// inode, and tags the directories; a second changes nothing.
void testFirstAndCurrent() {
	Paths p;
	writeFile(p.target, "v1");
	PosixFs fs;
	assert(p.update(fs) == kRgfxAliasLinked);
	assert(isFile(p.link) && inode(p.link) == inode(p.target));
	assert(fs.removes == 0 && fs.tags == 3);
	PosixFs again;
	assert(p.update(again) == kRgfxAliasCurrent);
	assert(again.removes == 0 && again.tags == 0 && inode(p.link) == inode(p.target));
}

// An install writes the executable as a new inode: the link to the old one is
// removed and made again to the new one.
void testNewExecutable() {
	Paths p;
	const uint64_t old = inode(p.target);
	replaceFile(p.target, "v2");
	assert(inode(p.target) != old);
	PosixFs fs;
	assert(p.update(fs) == kRgfxAliasLinked);
	assert(fs.removes == 1 && inode(p.link) == inode(p.target));
}

// Another file at the link's path, a copy or a symbolic link to the
// executable, is replaced by the link itself.
void testOtherFiles() {
	Paths p;
	unlink(p.link.c_str());
	writeFile(p.link, "a copy");
	PosixFs fs;
	assert(p.update(fs) == kRgfxAliasLinked);
	assert(fs.removes == 1 && inode(p.link) == inode(p.target));
	unlink(p.link.c_str());
	assert(symlink(p.target.c_str(), p.link.c_str()) == 0);
	PosixFs fs2;
	assert(p.update(fs2) == kRgfxAliasLinked);
	assert(fs2.removes == 1 && isFile(p.link) && inode(p.link) == inode(p.target));
}

// A file where a directory above the link belongs: no link, no tags.
void testBlockedDirectory() {
	Paths p;
	unlink(p.link.c_str());
	rmdir(p.macos.c_str());
	writeFile(p.macos, "not a directory");
	PosixFs fs;
	assert(p.update(fs) == kRgfxAliasFailed);
	assert(fs.tags == 0 && !isFile(p.link));
	unlink(p.macos.c_str());
}

} // namespace

int main() {
	char tmpl[] = "/tmp/rgfx-alias-test.XXXXXX";
	const char *dir = mkdtemp(tmpl);
	assert(dir != nullptr);
	g_root = dir;
	testNoTarget();
	testFirstAndCurrent();
	testNewExecutable();
	testOtherFiles();
	testBlockedDirectory();
	Paths p;
	unlink(p.link.c_str());
	rmdir(p.macos.c_str());
	rmdir(p.contents.c_str());
	rmdir(p.bundle.c_str());
	unlink(p.target.c_str());
	rmdir(g_root.c_str());
	std::printf("alias_test: OK\n");
	return 0;
}
