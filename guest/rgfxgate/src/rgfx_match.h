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
