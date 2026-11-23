/*
 * files_tests: the bundle's own file calls (src/RgfxFiles.*), over a directory of the test's own
 * (shell_tests' main calls rgfxFilesTests; guest/bundle/Makefile `test`). A whole write makes or
 * replaces its file, with open(2)'s mode, and leaves no file of its own behind, whether it
 * succeeds or fails; a read gives the file's bytes; a directory is made with its parents; a
 * listing gives a directory's regular files, hidden ones too; a touch sets a file's time to now
 * and keeps its bytes.
 *
 * And the rule the bundle's writes keep (src/RgfxFiles.hpp): each function, and the GL store over
 * them, runs in a child of this binary under a sandbox that kills it (SIGKILL) at every file
 * system call some WebKit WebContent profile read leaves out (kLeftOut), as WebKit's profile on
 * macOS 11 and 12 kills its process at a call its list leaves out. The controls show that the
 * sandbox kills at such a call: utimes and setattrlist, which set a file's times, and
 * open_dprotected_np, which Foundation's atomic write makes on macOS 11.
 */
#include <dirent.h>
#include <fcntl.h>
#include <sandbox.h>
#include <spawn.h>
#include <sys/attr.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/time.h>
#include <sys/wait.h>
#include <unistd.h>

#include <crt_externs.h>

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "RgfxFiles.hpp"
#include "RgfxGLStore.hpp"
#include "rgfx_test_ring.hpp" // CHECK, g_checks

void rgfxFilesTests(const char *self);
int rgfxFilesSandboxedCase(const char *directory, const char *name);

namespace {

const char *g_self = nullptr;

// The file system calls of the SDK's sys/syscall.h that the WebContent profile of some release
// leaves out, read from the guests' own profiles (com.apple.WebProcess.sb) on macOS 11.7.11,
// 12.7.6, 13.7.8, 14.8.9, 15.8.2 and 26.7.2 (x86-64) and 27.0.1 (arm64). A call a profile allows
// only outside Lockdown Mode, or only where it leaves IOKit open to the process, counts as
// allowed: WebGL runs on the device only where IOKit is open. Every one of them kills the process
// on 11 and 12.
const int kLeftOut[] = {
	SYS_link, SYS_chdir, SYS_fchdir, SYS_mknod, SYS_chown, SYS_chflags, SYS_fchflags, SYS_symlink,
	SYS_readv, SYS_fchown, SYS_mkfifo, SYS_rmdir, SYS_utimes, SYS_futimes, SYS_pwrite, SYS_statfs,
	SYS_fstatfs, SYS_fdatasync, SYS_stat, SYS_fstat, SYS_lstat, SYS_pathconf, SYS_fpathconf,
	SYS_getdirentries, SYS_truncate, SYS_undelete, SYS_fchflagsat, SYS_open_dprotected_np,
	SYS_fsgetpath_ext, SYS_openat_dprotected_np, SYS_setattrlist, SYS_getdirentriesattr,
	SYS_exchangedata, SYS_searchfs, SYS_delete, SYS_copyfile, SYS_fgetattrlist, SYS_fsetattrlist,
	SYS_setxattr, SYS_removexattr, SYS_fremovexattr, SYS_flistxattr, SYS_fsctl, SYS_ffsctl,
	SYS_fhopen, SYS_fsctlat, SYS_open_extended, SYS_stat_extended, SYS_lstat_extended,
	SYS_fstat_extended, SYS_chmod_extended, SYS_fchmod_extended, SYS_access_extended,
	SYS_mkfifo_extended, SYS_mkdir_extended, SYS_stat64_extended, SYS_lstat64_extended,
	SYS_fstat64_extended, SYS_guarded_ftruncate_np, SYS_lchown, SYS_fsync_nocancel,
	SYS_readv_nocancel, SYS_writev_nocancel, SYS_pread_nocancel, SYS_pwrite_nocancel,
	SYS_clonefileat, SYS_renameat, SYS_fchmodat, SYS_fchownat, SYS_fstatat, SYS_linkat,
	SYS_unlinkat, SYS_readlinkat, SYS_symlinkat, SYS_mkdirat, SYS_getattrlistat, SYS_openbyid_np,
	SYS_guarded_open_dprotected_np, SYS_guarded_write_np, SYS_guarded_writev_np, SYS_renameatx_np,
	SYS_fclonefileat, SYS_fs_snapshot, SYS_setattrlistat, SYS_preadv, SYS_pwritev,
	SYS_preadv_nocancel, SYS_pwritev_nocancel, SYS_freadlink, SYS_mkfifoat, SYS_mknodat,
	SYS_aio_read, SYS_aio_write, SYS_aio_fsync, SYS_aio_readv, SYS_aio_writev, SYS_lio_listio,
};

// A directory of the test's own, removed with what it holds when the test ends.
struct Scratch {
	std::string path;
	Scratch()
	{
		const char *base = getenv("TMPDIR");
		std::string t = std::string(base != nullptr && *base != 0 ? base : "/tmp") + "/rgfx-files.XXXXXX";
		std::vector<char> buf(t.begin(), t.end());
		buf.push_back(0);
		CHECK(mkdtemp(buf.data()) != nullptr);
		path = buf.data();
	}
	~Scratch() { removeTree(path); }
	static void removeTree(const std::string &p)
	{
		struct stat st;
		if (lstat(p.c_str(), &st) != 0)
			return;
		if (S_ISDIR(st.st_mode)) {
			chmod(p.c_str(), 0755);
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

std::vector<std::string> namesIn(const std::string &dir)
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

std::vector<uint8_t> bytesOf(size_t n, uint8_t seed)
{
	std::vector<uint8_t> out(n);
	for (size_t i = 0; i < n; i++)
		out[i] = static_cast<uint8_t>(seed + i * 7);
	return out;
}

std::vector<uint8_t> readAll(const std::string &path)
{
	std::vector<uint8_t> out;
	CHECK(RgfxReadFile(path, out) == 0);
	return out;
}

double mtimeOf(const std::string &path)
{
	struct stat st;
	CHECK(lstat(path.c_str(), &st) == 0);
	return static_cast<double>(st.st_mtimespec.tv_sec) + static_cast<double>(st.st_mtimespec.tv_nsec) / 1e9;
}

void setOld(const std::string &path, time_t ago)
{
	struct timeval now;
	gettimeofday(&now, nullptr);
	struct timeval t[2] = {{now.tv_sec - ago, 0}, {now.tv_sec - ago, 0}};
	CHECK(utimes(path.c_str(), t) == 0);
}

// A whole write makes its file with open(2)'s mode, replaces a file there with the new bytes, of
// any length, none too, and leaves only the file in its directory.
void test_whole_writes_make_and_replace()
{
	Scratch s;
	const std::string path = s.path + "/f.bin";
	const std::vector<uint8_t> first = bytesOf(5000, 1), second = bytesOf(70, 9);
	CHECK(RgfxWriteFileWhole(path, first.data(), first.size()) == 0);
	CHECK(readAll(path) == first);
	const mode_t mask = umask(0);
	umask(mask);
	struct stat st;
	CHECK(stat(path.c_str(), &st) == 0 && S_ISREG(st.st_mode) && (st.st_mode & 0777) == (0666 & ~mask));
	CHECK(RgfxWriteFileWhole(path, second.data(), second.size()) == 0);
	CHECK(readAll(path) == second);
	CHECK(RgfxWriteFileWhole(path, nullptr, 0) == 0);
	CHECK(readAll(path).empty());
	CHECK(namesIn(s.path) == std::vector<std::string>{"f.bin"});
	// A bare name is in the current directory.
	char here[PATH_MAX];
	CHECK(getcwd(here, sizeof here) != nullptr);
	CHECK(chdir(s.path.c_str()) == 0);
	CHECK(RgfxWriteFileWhole("g.bin", first.data(), first.size()) == 0);
	CHECK(chdir(here) == 0);
	CHECK(readAll(s.path + "/g.bin") == first);
}

// A write that fails says why, and whether it got as far as making its file, and leaves no file of
// its own: to a directory that is not there, over a directory, and in a directory it may not write
// in (the file there keeps its bytes).
void test_whole_write_failures_leave_nothing()
{
	Scratch s;
	const std::vector<uint8_t> b = bytesOf(100, 3);
	bool made = true;
	CHECK(RgfxWriteFileWhole(s.path + "/gone/f.bin", b.data(), b.size(), &made) == ENOENT && !made);
	CHECK(RgfxMakeDirectories(s.path + "/d") == 0);
	CHECK(RgfxWriteFileWhole(s.path + "/d", b.data(), b.size(), &made) == EISDIR && made);
	CHECK(namesIn(s.path) == std::vector<std::string>{"d"});
	CHECK(namesIn(s.path + "/d").empty());
	const std::string ro = s.path + "/ro";
	CHECK(RgfxMakeDirectories(ro) == 0);
	made = false;
	CHECK(RgfxWriteFileWhole(ro + "/f.bin", b.data(), b.size(), &made) == 0 && made);
	CHECK(chmod(ro.c_str(), 0555) == 0);
	const std::vector<uint8_t> other = bytesOf(10, 4);
	CHECK(RgfxWriteFileWhole(ro + "/f.bin", other.data(), other.size(), &made) == EACCES && !made);
	CHECK(readAll(ro + "/f.bin") == b);
	CHECK(namesIn(ro) == std::vector<std::string>{"f.bin"});
	CHECK(chmod(ro.c_str(), 0755) == 0);
}

// A read gives the file's bytes, of a file larger than one read takes too, and none of a file
// that is not there or a directory.
void test_reads()
{
	Scratch s;
	std::vector<uint8_t> out = {1};
	CHECK(RgfxReadFile(s.path + "/gone", out) == ENOENT && out.size() == 1);
	CHECK(RgfxReadFile(s.path, out) == EISDIR);
	const std::vector<uint8_t> big = bytesOf(3 << 20, 5);
	CHECK(RgfxWriteFileWhole(s.path + "/big", big.data(), big.size()) == 0);
	CHECK(readAll(s.path + "/big") == big);
	CHECK(RgfxWriteFileWhole(s.path + "/empty", nullptr, 0) == 0);
	CHECK(readAll(s.path + "/empty").empty());
	CHECK(RgfxReadFile(s.path + "/big/x", out) == ENOTDIR);
}

// A directory is made with its parents, and one there already is made; something else at its path
// or its parent's is ENOTDIR.
void test_directories()
{
	Scratch s;
	CHECK(RgfxMakeDirectories(s.path + "/a/b/c") == 0);
	CHECK(RgfxIsDirectory(s.path + "/a/b/c"));
	CHECK(RgfxMakeDirectories(s.path + "/a/b/c") == 0);
	CHECK(RgfxMakeDirectories(s.path + "/a") == 0);
	CHECK(RgfxWriteFileWhole(s.path + "/file", "x", 1) == 0);
	CHECK(RgfxMakeDirectories(s.path + "/file") == ENOTDIR);
	CHECK(RgfxMakeDirectories(s.path + "/file/d") == ENOTDIR);
	CHECK(!RgfxIsDirectory(s.path + "/file"));
	CHECK(!RgfxIsDirectory(s.path + "/gone"));
	CHECK(symlink((s.path + "/a").c_str(), (s.path + "/link").c_str()) == 0);
	CHECK(RgfxIsDirectory(s.path + "/link"));
}

// A listing gives the directory's regular files, hidden ones too, with their sizes and times, and
// leaves out directories and symbolic links.
void test_listing()
{
	Scratch s;
	CHECK(RgfxWriteFileWhole(s.path + "/one", "abc", 3) == 0);
	CHECK(RgfxWriteFileWhole(s.path + "/.hidden", "a", 1) == 0);
	CHECK(RgfxMakeDirectories(s.path + "/dir") == 0);
	CHECK(symlink("one", (s.path + "/link").c_str()) == 0);
	setOld(s.path + "/.hidden", 5000);
	std::vector<RgfxDirectoryFile> files;
	CHECK(RgfxListFiles(s.path, files) == 0);
	CHECK(files.size() == 2);
	const double now = RgfxFileClockNow();
	for (const RgfxDirectoryFile &f : files) {
		CHECK(f.name == "one" || f.name == ".hidden");
		CHECK(f.size == (f.name == "one" ? 3u : 1u));
		if (f.name == "one")
			CHECK(now - f.modified > -2 && now - f.modified < 60);
		else
			CHECK(now - f.modified > 4990 && now - f.modified < 5060);
	}
	CHECK(RgfxListFiles(s.path + "/gone", files) == ENOENT);
}

// A touch sets the file's time to now and keeps its bytes; not through a symbolic link.
void test_touch()
{
	Scratch s;
	const std::string path = s.path + "/e";
	const std::vector<uint8_t> b = bytesOf(64, 'R');
	CHECK(RgfxWriteFileWhole(path, b.data(), b.size()) == 0);
	setOld(path, 3000);
	CHECK(RgfxFileClockNow() - mtimeOf(path) > 2990);
	CHECK(RgfxTouchFile(path, b[0]) == 0);
	const double age = RgfxFileClockNow() - mtimeOf(path);
	CHECK(age > -2 && age < 60);
	CHECK(readAll(path) == b);
	CHECK(RgfxTouchFile(s.path + "/gone", 'R') == ENOENT);
	CHECK(symlink("e", (s.path + "/link").c_str()) == 0);
	CHECK(RgfxTouchFile(s.path + "/link", 'R') == ELOOP);
}

// A removal removes the file, and says ENOENT of one not there.
void test_remove()
{
	Scratch s;
	CHECK(RgfxWriteFileWhole(s.path + "/f", "x", 1) == 0);
	CHECK(RgfxRemoveFile(s.path + "/f") == 0);
	CHECK(namesIn(s.path).empty());
	CHECK(RgfxRemoveFile(s.path + "/f") == ENOENT);
}

// Runs this binary again as `--files-sandboxed DIR name`; its wait status.
int runSandboxed(const std::string &dir, const char *name)
{
	char *args[] = {const_cast<char *>(g_self), const_cast<char *>("--files-sandboxed"),
	                const_cast<char *>(dir.c_str()), const_cast<char *>(name), nullptr};
	pid_t pid = 0;
	CHECK(posix_spawn(&pid, g_self, nullptr, nullptr, args, *_NSGetEnviron()) == 0);
	int status = 0;
	CHECK(waitpid(pid, &status, 0) == pid);
	return status;
}

bool exitedZero(int status) { return WIFEXITED(status) && WEXITSTATUS(status) == 0; }
bool killed(int status) { return WIFSIGNALED(status) && WTERMSIG(status) == SIGKILL; }

// Each function, and the GL store over them, makes only calls every WebContent list read takes:
// under a sandbox that kills at every call some list leaves out, they run to their end. The
// controls are killed.
void test_calls_every_webcontent_list_takes()
{
	Scratch s;
	const int files = runSandboxed(s.path, "files");
	if (!exitedZero(files))
		std::fprintf(stderr, "--files-sandboxed files ended with status %d\n", files);
	CHECK(exitedZero(files));
	// The store's trim removes a file a dead process's write left, once it is an hour old.
	CHECK(RgfxMakeDirectories(s.path + "/gl") == 0);
	CHECK(RgfxWriteFileWhole(s.path + "/gl/.rgfx-0000000000000000.tmp", "x", 1) == 0);
	setOld(s.path + "/gl/.rgfx-0000000000000000.tmp", 7200);
	const int store = runSandboxed(s.path, "store");
	if (!exitedZero(store))
		std::fprintf(stderr, "--files-sandboxed store ended with status %d\n", store);
	CHECK(exitedZero(store));
	for (const char *control : {"utimes", "setattrlist", "open_dprotected_np"}) {
		const int status = runSandboxed(s.path, control);
		if (!killed(status))
			std::fprintf(stderr, "--files-sandboxed %s ended with status %d\n", control, status);
		CHECK(killed(status));
	}
}

std::string sandboxProfile()
{
	std::string p = "(version 1)(allow default)(deny syscall-unix (with send-signal SIGKILL) (syscall-number";
	for (int n : kLeftOut)
		p += " " + std::to_string(n);
	return p + "))";
}

std::string keyOf(uint8_t n)
{
	const uint8_t ir[4] = {1, 2, 3, n};
	const uint8_t plugin[3] = {9, 8, n};
	return RgfxGLVariantKey("files 1\n", ir, sizeof ir, plugin, sizeof plugin, 2, 1);
}

void sandboxedFiles(const std::string &dir)
{
	const std::string d = dir + "/x/y";
	const std::vector<uint8_t> a = bytesOf(9000, 1), b = bytesOf(40, 2);
	CHECK(RgfxMakeDirectories(d) == 0 && RgfxIsDirectory(d));
	CHECK(RgfxWriteFileWhole(d + "/f", a.data(), a.size()) == 0);
	CHECK(RgfxWriteFileWhole(d + "/f", b.data(), b.size()) == 0);
	CHECK(readAll(d + "/f") == b);
	CHECK(RgfxTouchFile(d + "/f", b[0]) == 0);
	std::vector<RgfxDirectoryFile> files;
	CHECK(RgfxListFiles(d, files) == 0 && files.size() == 1 && files[0].size == b.size());
	CHECK(RgfxFileClockNow() - files[0].modified < 60);
	CHECK(RgfxWriteFileWhole(dir + "/gone/f", a.data(), a.size()) == ENOENT);
	CHECK(RgfxRemoveFile(d + "/f") == 0);
}

void sandboxedStore(const std::string &dir)
{
	const std::string gl = dir + "/gl";
	const std::vector<uint8_t> value = bytesOf(1000, 7);
	// An entry's size, from a store of its own.
	uint64_t entry = 0;
	{
		RgfxGLStore probe(dir + "/probe", 1 << 20);
		CHECK(probe.put(keyOf(0), value));
		entry = readAll(probe.pathOf(keyOf(0))).size();
	}
	// Room for three entries and a half.
	RgfxGLStore store(gl, entry * 3 + entry / 2);
	std::vector<uint8_t> got;
	CHECK(!store.get(keyOf(1), got));
	CHECK(store.put(keyOf(1), value)); // its trim removes the dead write's file
	std::vector<RgfxDirectoryFile> files;
	CHECK(RgfxListFiles(gl, files) == 0 && files.size() == 1);
	CHECK(store.put(keyOf(2), value));
	CHECK(store.put(keyOf(3), value));
	CHECK(store.get(keyOf(1), got) && got == value); // the newest now
	// Past the cap: 2 and 3 go, as three quarters of the cap hold two.
	CHECK(store.put(keyOf(4), value));
	CHECK(!store.get(keyOf(2), got) && !store.get(keyOf(3), got));
	CHECK(store.get(keyOf(1), got) && store.get(keyOf(4), got));
	// A bad entry is removed.
	CHECK(RgfxWriteFileWhole(store.pathOf(keyOf(5)), "not an entry", 12) == 0);
	CHECK(!store.get(keyOf(5), got));
	std::string why;
	CHECK(store.takeFirstFailure(why) && why.find("removed") != std::string::npos);
	CHECK(RgfxReadFile(store.pathOf(keyOf(5)), got) == ENOENT);
	// A queued write, on the store's own queue.
	store.putLater(keyOf(6), value);
	store.drain();
	CHECK(store.get(keyOf(6), got) && got == value);
}

} // namespace

int rgfxFilesSandboxedCase(const char *directory, const char *name)
{
	const std::string dir = directory;
	const std::string control = dir + "/control";
	if (std::strcmp(name, "files") != 0 && std::strcmp(name, "store") != 0)
		CHECK(RgfxWriteFileWhole(control, "x", 1) == 0);
	char *error = nullptr;
	if (sandbox_init(sandboxProfile().c_str(), 0, &error) != 0) {
		std::fprintf(stderr, "sandbox_init: %s\n", error != nullptr ? error : "(no message)");
		return 3;
	}
	if (std::strcmp(name, "files") == 0)
		sandboxedFiles(dir);
	else if (std::strcmp(name, "store") == 0)
		sandboxedStore(dir);
	else if (std::strcmp(name, "utimes") == 0)
		utimes(control.c_str(), nullptr);
	else if (std::strcmp(name, "setattrlist") == 0) {
		struct attrlist list = {};
		list.bitmapcount = ATTR_BIT_MAP_COUNT;
		list.commonattr = ATTR_CMN_MODTIME;
		struct timespec t = {1, 0};
		setattrlist(control.c_str(), &list, &t, sizeof t, FSOPT_NOFOLLOW);
	} else if (std::strcmp(name, "open_dprotected_np") == 0)
		syscall(SYS_open_dprotected_np, (dir + "/made").c_str(), O_WRONLY | O_CREAT, 0, 0, 0644);
	else
		return 2;
	_exit(0);
}

void rgfxFilesTests(const char *self)
{
	g_self = self;
	test_whole_writes_make_and_replace();
	test_whole_write_failures_leave_nothing();
	test_reads();
	test_directories();
	test_listing();
	test_touch();
	test_remove();
	test_calls_every_webcontent_list_takes();
}
