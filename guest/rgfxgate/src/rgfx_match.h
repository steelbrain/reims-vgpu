// Pure matching logic shared by the kext hooks and the native tests.
// No kernel or Lilu dependency: the two decisions that gate RgfxGate are a
// code-directory-hash membership test (which bundle executable may bypass
// library validation / file-map-executable) and a path-prefix test (which
// directory the boot-time install may write). Keeping them here lets the
// native tests exercise the exact code the kext runs.
#pragma once
#include <stddef.h>
#include <stdint.h>
#ifdef KERNEL
#include <libkern/libkern.h> // memcmp, strlen, strncmp in the kext
#else
#include <string.h>
#endif

// CS_CDHASH_LEN: a code-directory hash is always compared at 20 bytes
// (larger hashes are truncated), matching the kernel's csfg_get_cdhash.
enum { kRgfxCDHashLen = 20 };

// True when cand (cand_len bytes, as the kernel returns it) equals one of the
// n allowed 20-byte cdhashes. A shorter candidate never matches.
static inline bool rgfx_cdhash_allowed(const uint8_t *cand, size_t cand_len,
                                       const uint8_t (*allowed)[kRgfxCDHashLen], size_t n) {
	if (cand == NULL || cand_len < (size_t)kRgfxCDHashLen)
		return false;
	for (size_t i = 0; i < n; i++)
		if (memcmp(cand, allowed[i], kRgfxCDHashLen) == 0)
			return true;
	return false;
}

// True when path is prefix itself or a descendant of it: prefix must be
// followed by end-of-string or a '/', so "/a/bc" is not under "/a/b".
static inline bool rgfx_path_under(const char *path, const char *prefix) {
	if (path == NULL || prefix == NULL)
		return false;
	size_t pl = strlen(prefix);
	if (pl == 0)
		return false;
	if (strncmp(path, prefix, pl) != 0)
		return false;
	char c = path[pl];
	return c == '\0' || c == '/';
}

// out = path immediately followed by suffix, NUL-terminated; true only when the
// whole thing (with its terminator) fits in cap, and nothing is written when it
// does not. The install forms the staging sibling of the bundle path this way
// (".../NAME.bundle" + ".new"): it writes the new tree there and renames it over
// the live bundle, so a reader sees the whole old tree or the whole new one,
// never a half-written one (RgfxGateInstall.cpp).
static inline bool rgfx_sibling_path(const char *path, const char *suffix,
                                     char *out, size_t cap) {
	if (path == NULL || suffix == NULL || out == NULL || cap == 0)
		return false;
	size_t pl = strlen(path), sl = strlen(suffix);
	if (pl + sl + 1 > cap)
		return false;
	memcpy(out, path, pl);
	memcpy(out + pl, suffix, sl);
	out[pl + sl] = '\0';
	return true;
}
