/*
 * RgfxFiles: the bundle's own file calls. Every file the bundle writes from an app's process goes
 * through these: the GL store's (RgfxGLStore.hpp), and a dynamic library's and a binary archive's
 * serializeToURL:.
 *
 * Every Metal process loads the bundle, sandboxed ones too, and a sandbox may list the system
 * calls a process may make. WebKit's WebContent profile does (read from the guests' own profiles:
 * macOS 11.7.11, 12.7.6, 13.7.8, 14.8.9, 15.8.2 and 26.7.2 on x86-64, 27.0.1 on arm64): on macOS 11
 * and 12 a call its list leaves out kills the process (SIGKILL), and from 13 on it fails (EPERM).
 * Foundation's file calls make whatever system calls their release's Foundation picks. Its atomic
 * write (NSDataWritingAtomic, writeToURL:atomically:) makes its file with open_dprotected_np on
 * macOS 11, which 11's list leaves out, and an atomic write from the bundle killed every Safari
 * WebGL page there (docs/lessons.md). So these functions make plain POSIX calls: open, read,
 * write, close, fstat, stat, lstat, rename, unlink, mkdir, and opendir and readdir. Each list read
 * takes their system calls; from 14 on, rename and unlink only where the profile has not closed
 * IOKit to the process, as it has not in any process that runs Metal on the device, whose driver
 * is an IOKit user client, and unlink outside Lockdown Mode. Neither call that sets a file's
 * times is on every list: setattrlist is on 11's alone, utimes on none, so RgfxTouchFile sets a
 * file's time with a write. tests/files_tests.mm runs each function, and the GL store, in a
 * sandbox that kills the process at every file system call some list leaves out.
 *
 * Each function returns 0, or the errno of the call that failed.
 */
#ifndef RGFX_FILES_HPP_
#define RGFX_FILES_HPP_

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

/// The bytes of the file at `path`, in `out`.
int RgfxReadFile(const std::string &path, std::vector<uint8_t> &out);

/// Writes `length` bytes at `bytes` to `path`, whole: to a new file of its own in the same
/// directory (".rgfx-<16 hex digits>.tmp"), then renamed over `path`. A reader sees the old file
/// or the new one, never part of one. The file is made with mode 0666 less the process's umask,
/// as open(2) makes a file. A failed write removes its own file. A process that ends during the
/// write leaves that file behind, and the GL store's trim removes it later. `madeFile`, where
/// given, is set to whether the write made its file in the directory: a failure with it false
/// failed at making that file, one with it true at writing it or renaming it over `path`.
int RgfxWriteFileWhole(const std::string &path, const void *bytes, size_t length, bool *madeFile = nullptr);

/// Makes the directory `path` and any of its parents that are not there (mode 0777 less the
/// umask). 0 where it is there already, and ENOTDIR where something else is at `path`.
int RgfxMakeDirectories(const std::string &path);

/// Removes the file at `path`.
int RgfxRemoveFile(const std::string &path);

/// Sets the modification time of the file at `path` to now, by writing `first` over its first
/// byte, which the caller knows the file holds: the file's bytes stay as they were. Not through a
/// symbolic link.
int RgfxTouchFile(const std::string &path, uint8_t first);

/// Whether `path` names a directory, or a symbolic link to one (stat(2)).
bool RgfxIsDirectory(const std::string &path);

/// A regular file of a directory: its name, size and modification time, in seconds since 1970.
struct RgfxDirectoryFile {
	std::string name;
	uint64_t size = 0;
	double modified = 0;
};

/// The regular files of the directory `directory`, its hidden ones too, in `out`. Symbolic links
/// and directories are left out.
int RgfxListFiles(const std::string &directory, std::vector<RgfxDirectoryFile> &out);

/// Now, in seconds since 1970, on the clock RgfxListFiles' times are on.
double RgfxFileClockNow();

#endif // RGFX_FILES_HPP_
