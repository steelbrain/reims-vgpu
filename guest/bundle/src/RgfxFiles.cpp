/*
 * RgfxFiles: see RgfxFiles.hpp. Plain POSIX calls only: no Foundation, no open(2) variant but
 * open itself, no temporary directory.
 */
#include "RgfxFiles.hpp"

#include <dirent.h>
#include <fcntl.h>
#include <mach/mach_time.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <cstdio>

namespace {

// The most one write(2) is asked to take.
constexpr size_t kWriteChunk = size_t(1) << 30;

int openRetrying(const char *path, int flags, mode_t mode)
{
	int fd;
	do
		fd = open(path, flags, mode);
	while (fd < 0 && errno == EINTR);
	return fd;
}

// The directory `path` names its file in: "." for a bare name, "/" for one at the root.
std::string parentOf(const std::string &path)
{
	const size_t slash = path.find_last_of('/');
	if (slash == std::string::npos)
		return ".";
	if (slash == 0)
		return "/";
	return path.substr(0, slash);
}

// mkdir(2), with a directory already at `path` answered as made.
int makeOne(const std::string &path)
{
	if (mkdir(path.c_str(), 0777) == 0)
		return 0;
	const int failure = errno;
	if (failure != EEXIST)
		return failure;
	struct stat st;
	if (stat(path.c_str(), &st) != 0)
		return errno;
	return S_ISDIR(st.st_mode) ? 0 : ENOTDIR;
}

} // namespace

int RgfxReadFile(const std::string &path, std::vector<uint8_t> &out)
{
	const int fd = openRetrying(path.c_str(), O_RDONLY | O_CLOEXEC, 0);
	if (fd < 0)
		return errno;
	struct stat st;
	if (fstat(fd, &st) != 0) {
		const int failure = errno;
		close(fd);
		return failure;
	}
	if (S_ISDIR(st.st_mode)) {
		close(fd);
		return EISDIR;
	}
	if (st.st_size < 0 || static_cast<uint64_t>(st.st_size) > SIZE_MAX / 2) {
		close(fd);
		return EFBIG;
	}
	// One byte past the size it has now, so a read at the end sees the end.
	std::vector<uint8_t> bytes(static_cast<size_t>(st.st_size) + 1);
	size_t have = 0;
	for (;;) {
		if (have == bytes.size())
			bytes.resize(bytes.size() * 2);
		const ssize_t n = read(fd, bytes.data() + have, bytes.size() - have);
		if (n < 0) {
			if (errno == EINTR)
				continue;
			const int failure = errno;
			close(fd);
			return failure;
		}
		if (n == 0)
			break;
		have += static_cast<size_t>(n);
	}
	close(fd);
	bytes.resize(have);
	out = std::move(bytes);
	return 0;
}

int RgfxWriteFileWhole(const std::string &path, const void *bytes, size_t length)
{
	static std::atomic<uint64_t> serial{0};
	const std::string directory = parentOf(path);
	std::string temporary;
	int fd = -1;
	// A name no other write uses: the clock, the process's count of its writes and an address of
	// its stack (which ASLR places per process); O_EXCL refuses a name in use, and the next is tried.
	for (int attempt = 0; fd < 0; attempt++) {
		char name[32];
		const uint64_t tag = mach_absolute_time() ^ (serial.fetch_add(1) * 0x9e3779b97f4a7c15ull) ^
		                     static_cast<uint64_t>(reinterpret_cast<uintptr_t>(&name));
		snprintf(name, sizeof name, "/.rgfx-%016llx.tmp", static_cast<unsigned long long>(tag));
		temporary = directory + name;
		fd = openRetrying(temporary.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0666);
		if (fd < 0 && (errno != EEXIST || attempt == 7))
			return errno;
	}
	const uint8_t *p = static_cast<const uint8_t *>(bytes);
	size_t left = length;
	int failure = 0;
	while (left > 0) {
		const ssize_t n = write(fd, p, left > kWriteChunk ? kWriteChunk : left);
		if (n < 0) {
			if (errno == EINTR)
				continue;
			failure = errno;
			break;
		}
		p += n;
		left -= static_cast<size_t>(n);
	}
	if (close(fd) != 0 && failure == 0 && errno != EINTR)
		failure = errno;
	if (failure == 0 && rename(temporary.c_str(), path.c_str()) != 0)
		failure = errno;
	if (failure != 0)
		unlink(temporary.c_str());
	return failure;
}

int RgfxMakeDirectories(const std::string &path)
{
	int failure = makeOne(path);
	if (failure != ENOENT)
		return failure;
	const std::string parent = parentOf(path);
	if (parent == path || parent == "." || parent == "/")
		return failure;
	failure = RgfxMakeDirectories(parent);
	if (failure != 0)
		return failure;
	return makeOne(path);
}

int RgfxRemoveFile(const std::string &path)
{
	return unlink(path.c_str()) == 0 ? 0 : errno;
}

double RgfxFileClockNow()
{
	struct timeval now;
	gettimeofday(&now, nullptr);
	return static_cast<double>(now.tv_sec) + static_cast<double>(now.tv_usec) / 1e6;
}

int RgfxTouchFile(const std::string &path, uint8_t first)
{
	const int fd = openRetrying(path.c_str(), O_WRONLY | O_NOFOLLOW | O_CLOEXEC, 0);
	if (fd < 0)
		return errno;
	ssize_t n;
	do
		n = write(fd, &first, 1);
	while (n < 0 && errno == EINTR);
	const int failure = n == 1 ? 0 : n < 0 ? errno : EIO;
	close(fd);
	return failure;
}

bool RgfxIsDirectory(const std::string &path)
{
	struct stat st;
	return stat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
}

int RgfxListFiles(const std::string &directory, std::vector<RgfxDirectoryFile> &out)
{
	DIR *d = opendir(directory.c_str());
	if (d == nullptr)
		return errno;
	out.clear();
	while (struct dirent *e = readdir(d)) {
		const std::string name = e->d_name;
		if (name == "." || name == "..")
			continue;
		struct stat st;
		if (lstat((directory + "/" + name).c_str(), &st) != 0 || !S_ISREG(st.st_mode))
			continue;
		RgfxDirectoryFile f;
		f.name = name;
		f.size = static_cast<uint64_t>(st.st_size);
		f.modified = static_cast<double>(st.st_mtimespec.tv_sec) + static_cast<double>(st.st_mtimespec.tv_nsec) / 1e9;
		out.push_back(std::move(f));
	}
	closedir(d);
	return 0;
}
