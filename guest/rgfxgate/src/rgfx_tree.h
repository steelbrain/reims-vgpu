// The removal of a tree the boot-time install staged or swapped aside
// (RgfxGateInstall.cpp's removeTree): every entry under the root, whichever
// build wrote it, then the root. The payload's own file list names only this
// build's files, so a file an older build staged and this one dropped would
// keep its directories, and the root, from going; the next install would then
// stage into that root and swap it in as the live bundle, the old file with
// it. The walk here reads the directories themselves instead.
//
// No kernel or Lilu dependency: the walk takes the file system as a parameter,
// which the kext gives its VNOP calls and the native tests (tests/tree_test.cpp)
// a real directory, so the tests run the exact code the kext runs.
#pragma once
#include <stddef.h>
#include <stdint.h>
#ifdef KERNEL
#include <libkern/libkern.h> // memcpy, strlen in the kext
#else
#include <string.h>
#endif

// What a path names, as a lookup that does not follow a final symbolic link
// sees it: a symbolic link is a file, removed itself, never its target.
enum RgfxNodeKind { kRgfxNodeAbsent = 0, kRgfxNodeFile, kRgfxNodeDir };

// The deepest a removal descends below its root. A bundle's tree is four deep
// (NAME.bundle/Contents/Resources/FILE); a directory below this depth is left,
// and with it the directories above it.
static const unsigned kRgfxTreeMaxDepth = 16;
// The most nodes one removal takes: a bundle holds a few dozen.
static const unsigned kRgfxTreeMaxNodes = 4096;

// Appends '/' and `leaf` to `path`, which holds `len` bytes in a buffer of
// `cap`. False, with `path` as it was, where the result does not fit.
static inline bool rgfx_path_push(char *path, size_t cap, size_t len, const char *leaf) {
	const size_t leaflen = strlen(leaf);
	if (len + 1 + leaflen + 1 > cap)
		return false;
	path[len] = '/';
	memcpy(path + len + 1, leaf, leaflen + 1);
	return true;
}

// The next entry other than "." and ".." in `len` bytes of a directory read,
// from byte `*pos`, which moves past it: records of `Rec`'s layout (the
// kernel's `struct direntry`, VNOP_READDIR's with VNODE_READDIR_EXTENDED), each
// `d_reclen` bytes and its name `d_namlen` bytes from `d_name`. False at the
// end of the bytes, and at a record whose lengths do not fit in them, which
// ends the read there.
template <typename Rec>
static inline bool rgfx_dir_next(const uint8_t *buf, size_t len, size_t *pos,
                                 const char **name, size_t *namelen) {
	const size_t head = offsetof(Rec, d_name);
	while (*pos + head <= len) {
		uint16_t reclen = 0;
		decltype(Rec::d_namlen) nl = 0;
		memcpy(&reclen, buf + *pos + offsetof(Rec, d_reclen), sizeof(reclen));
		memcpy(&nl, buf + *pos + offsetof(Rec, d_namlen), sizeof(nl));
		if (reclen < head || reclen > len - *pos || head + (size_t)nl > reclen)
			return false;
		const char *n = reinterpret_cast<const char *>(buf + *pos + head);
		*pos += reclen;
		if (nl == 0 || (nl == 1 && n[0] == '.') || (nl == 2 && n[0] == '.' && n[1] == '.'))
			continue;
		*name = n;
		*namelen = nl;
		return true;
	}
	return false;
}

// Removes the tree at `path` (a NUL-terminated path in a buffer of `cap`
// bytes, which the walk writes below the root and leaves holding the root
// again). Each pass walks down from the root to a node with nothing under it,
// a file or an empty directory, following the first entry each directory
// reads, and removes it; so no directory is read after the walk has changed
// it. It stops at the first pass that removes nothing: a node it cannot
// remove, a directory past kRgfxTreeMaxDepth, or a path too long for the
// buffer. Whether the root is gone (an absent root counts).
//
// `Fs` gives the file system:
//   RgfxNodeKind kind(const char *path);
//   bool first(const char *dir, char *name, size_t cap);  // an entry of dir
//       other than "." and "..", NUL-terminated; false where it has none or
//       cannot be read
//   void remove(const char *path, RgfxNodeKind kind);     // unlink or rmdir
template <typename Fs>
static inline bool rgfx_tree_remove(Fs &fs, char *path, size_t cap) {
	const size_t root = strlen(path);
	char name[256];
	for (unsigned n = 0; n < kRgfxTreeMaxNodes; n++) {
		path[root] = '\0';
		size_t len = root;
		RgfxNodeKind kind = fs.kind(path);
		if (kind == kRgfxNodeAbsent)
			return true;
		for (unsigned depth = 0; kind == kRgfxNodeDir && depth < kRgfxTreeMaxDepth; depth++) {
			if (!fs.first(path, name, sizeof(name)) || !rgfx_path_push(path, cap, len, name))
				break;
			len = strlen(path);
			kind = fs.kind(path);
		}
		bool removed = false;
		if (kind != kRgfxNodeAbsent) {
			fs.remove(path, kind);
			removed = fs.kind(path) == kRgfxNodeAbsent;
		}
		if (!removed) {
			path[root] = '\0';
			return false;
		}
	}
	path[root] = '\0';
	return fs.kind(path) == kRgfxNodeAbsent;
}
