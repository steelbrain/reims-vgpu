// Pure matching logic shared by the kext hooks and the native tests.
// No kernel or Graft dependency: the two decisions that gate RgfxGate are a
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

// The rootless trusted-class query RgfxGate answers on arm64's BaseSystem, so
// macOS 27 loads our GPU bundle from the BaseSystem's tmpfs Data volume. It is
// the Sandbox policy's mac_syscall(2) call 2, operation "file-write-data",
// storage class "KernelExtensionManagement", with a path subject (the selector
// field is 1, not 0xf0 for an fd), and the path under our bundle. True only on
// that exact shape; anything else the hook chains to the Sandbox policy's own
// handler untouched (RgfxGateHooks.cpp).
#define RGFX_ROOTLESS_CALL 2
#define RGFX_ROOTLESS_OP "file-write-data"
#define RGFX_ROOTLESS_CLASS "KernelExtensionManagement"
#define RGFX_ROOTLESS_SELECTOR_PATH 1
#define RGFX_ROOTLESS_BUNDLE "/Library/GPUBundles/reims-graphics.bundle"

// strcmp built from strlen + strncmp (both in libkern and <string.h>), so the
// kext needs no strcmp symbol.
static inline bool rgfx_streq(const char *a, const char *b) {
	if (a == NULL || b == NULL)
		return false;
	size_t bl = strlen(b);
	return strncmp(a, b, bl) == 0 && a[bl] == '\0';
}

static inline bool rgfx_rootless_trust_ours(int call, unsigned long long selector,
                                            const char *op, const char *cls, const char *path) {
	if (call != RGFX_ROOTLESS_CALL || selector != RGFX_ROOTLESS_SELECTOR_PATH)
		return false;
	if (op == NULL || cls == NULL || path == NULL)
		return false;
	if (!rgfx_streq(op, RGFX_ROOTLESS_OP) || !rgfx_streq(cls, RGFX_ROOTLESS_CLASS))
		return false;
	return rgfx_path_under(path, RGFX_ROOTLESS_BUNDLE);
}

// The signature hook's answer where the real policies refused a code signature
// (`refusal`, nonzero): 0 for our bundle's executable (`ours`: one of the
// allowed cdhashes, under the bundles directory), the refusal for anything else.
// A refusal can be fatal: AMFI hands back a description (`*desc`, `*desc_len`
// bytes), and the dispatcher (mac_vnode_check_signature) panics where the hook
// answers 0 with one set. arm64's BaseSystem allows only platform binaries and
// refuses our ad-hoc signed bundle so. Admitting our bundle drops the
// description: the length and the pointer are cleared, and the buffer is left
// as it is, never freed here (RgfxGateHooks.cpp). Returns the answer;
// `*dropped` says whether a description was dropped. Anything not ours keeps
// its refusal and its description.
static inline int rgfx_vsig_answer(int refusal, bool ours, char **desc, size_t *desc_len,
                                   bool *dropped) {
	*dropped = false;
	if (!ours)
		return refusal;
	if (desc_len != NULL && *desc_len != 0) {
		*desc_len = 0;
		if (desc != NULL)
			*desc = NULL;
		*dropped = true;
	}
	return 0;
}
