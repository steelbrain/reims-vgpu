// RgfxGate's boot-time install. On the first normal boot, before the
// accelerator looks for a GPU bundle, the kext writes its embedded
// reims-graphics.bundle to /Library/GPUBundles in kernel context and gives the
// tree the SIP rootless storage class a stock host carries there, so the Metal
// loader trusts it (RgfxGate.hpp). The write is atomic: the whole tree is built
// and tagged in a sibling staging directory (NAME.bundle.new), then renamed
// over the live bundle, so a reader sees either the whole old tree or the whole
// new one, never a half-written one. The container is created first if the Data
// volume lacks it. Later boots are no-ops once the bundle is current and
// tagged. A newer kext whose embedded executable differs -- even at the same
// wire version -- replaces the bundle the same way: the staged tree is swapped
// in with a single rename (VNOP_RENAMEX, exchanging it with the live tree) and
// the old tree, swapped aside, is dropped. The container is never unlocked, so
// SIP is relaxed for nothing even mid-upgrade. A bundle that is current but
// untagged -- staged under relaxed SIP, then booted SIP-on -- is tagged in
// place without a rewrite. The staging writes are admitted by the vnode hooks
// (RgfxGateHooks.cpp) only while gInstalling is set on this thread and the
// target is under the bundle path; the tags, the renames and the removes run
// below the MAC layer, as the directory creates do. In x86-64's installer and
// Recovery the kext does nothing. In arm64's BaseSystem (FileVault's unlock,
// Recovery, the obliteration boots) it installs the bundle into the BaseSystem's
// temporary Data volume, then passes again on each Data volume the boot moves to
// without restarting the kernel (rgfx_pass.h). The pass is published on
// IOResources from its start to its end (RgfxGate.hpp), and the accelerator
// registers only once it has ended.
#include "RgfxGateFile.hpp"

#include <IOKit/IOLib.h>
#include <IOKit/IOService.h>
#include <kern/clock.h>
#include <kern/thread.h>
#include <libkern/c++/OSNumber.h>
#include <libkern/libkern.h>
#include <mach/thread_act.h>
#include <sys/fcntl.h>
#include <sys/mount.h>
#include <sys/stat.h>
#include <sys/dirent.h>
#include <sys/uio.h>
#include <sys/vnode.h>
#include <sys/vnode_if.h>

#include "RgfxGate.hpp"
#include "RgfxGatePayload.h"
#include "RgfxGatePrefs.hpp"
#include "RgfxGateVnop.hpp"
#include "rgfx_match.h"
#include "rgfx_pass.h"
#include "rgfx_tree.h"

namespace rgfxgate {

using namespace rgfx_payload;

static void joinPath(char *out, size_t cap, const char *a, const char *b) {
	size_t n = strlcpy(out, a, cap);
	if (n < cap)
		n += strlcpy(out + n, "/", cap - n);
	if (n < cap)
		strlcpy(out + n, b, cap - n);
}

// Look up the parent directory of `path` and fill `cn` (a componentname with
// the given nameiop) with its leaf; `leafbuf` (leafcap bytes) backs the name
// and must outlive the VNOP call. Returns the parent vnode, which the caller
// puts, or nullptr. Shared by the rename and rmdir helpers, which call the
// VNOP KPIs directly, below the vn_* MAC layer, as the directory creates do.
static vnode_t parentAndLeaf(const char *path, uint32_t nameiop,
                             struct componentname *cn, char *leafbuf,
                             size_t leafcap, vfs_context_t ctx) {
	const char *slash = nullptr;
	for (const char *q = path; *q; q++)
		if (*q == '/')
			slash = q;
	if (slash == nullptr || slash == path)
		return nullptr;
	size_t plen = (size_t)(slash - path);
	char parent[1024];
	if (plen >= sizeof(parent))
		return nullptr;
	memcpy(parent, path, plen);
	parent[plen] = '\0';
	const char *leaf = slash + 1;
	if (strlcpy(leafbuf, leaf, leafcap) >= leafcap)
		return nullptr;
	vnode_t dvp = nullptr;
	if (vnode_lookup(parent, 0, &dvp, ctx) != 0)
		return nullptr;
	bzero(cn, sizeof(*cn));
	cn->cn_nameiop = nameiop;
	cn->cn_flags = 0;
	cn->cn_pnbuf = leafbuf;
	cn->cn_pnlen = (int)strlen(leafbuf);
	cn->cn_nameptr = leafbuf;
	cn->cn_namelen = (int)strlen(leafbuf);
	return dvp;
}

// Create one directory by full path if it is not already there. Returns true
// if it exists afterwards.
static bool ensureDir(const char *path, vfs_context_t ctx) {
	vnode_t vp = nullptr;
	if (vnode_lookup(path, 0, &vp, ctx) == 0) {
		vnode_put(vp);
		return true; // already present
	}
	// Split into parent + leaf.
	const char *slash = nullptr;
	for (const char *p = path; *p; p++)
		if (*p == '/')
			slash = p;
	if (slash == nullptr || slash == path)
		return false;
	char parent[1024];
	size_t plen = (size_t)(slash - path);
	if (plen >= sizeof(parent))
		return false;
	memcpy(parent, path, plen);
	parent[plen] = '\0';
	const char *leaf = slash + 1;

	vnode_t dvp = nullptr;
	if (vnode_lookup(parent, 0, &dvp, ctx) != 0)
		return false;

	struct componentname cn;
	bzero(&cn, sizeof(cn));
	cn.cn_nameiop = CREATE;
	cn.cn_flags = 0;
	cn.cn_pnbuf = (char *)leaf;
	cn.cn_pnlen = (int)strlen(leaf);
	cn.cn_nameptr = (char *)leaf;
	cn.cn_namelen = (int)strlen(leaf);

	struct vnode_attr va;
	VATTR_INIT(&va);
	VATTR_SET(&va, va_type, VDIR);
	VATTR_SET(&va, va_mode, 0755);
	VATTR_SET(&va, va_uid, 0);
	VATTR_SET(&va, va_gid, 0);

	vnode_t nvp = nullptr;
	int err = rgfxVnopMkdir(dvp, &nvp, &cn, &va, ctx);
	if (nvp)
		vnode_put(nvp);
	vnode_put(dvp);
	return err == 0;
}

// The embedded executable's bytes, for the idempotency compare.
static const PayloadFile *embeddedExec() {
	for (uint32_t i = 0; i < kFileCount; i++)
		if (kFiles[i].is_exec)
			return &kFiles[i];
	return nullptr;
}

// True if the on-disk bundle is missing, has a different executable, or is
// missing the wire-version marker.
static bool installNeeded(vfs_context_t ctx) {
	const PayloadFile *ex = embeddedExec();
	if (ex == nullptr)
		return false;
	char execPath[1024], markerPath[1024];
	joinPath(execPath, sizeof(execPath), kBundlePath, kExecRel);
	joinPath(markerPath, sizeof(markerPath), kBundlePath, kMarkerRel);

	size_t sz = 0;
	uint8_t *buf = readFile(execPath, sz, ex->size);
	if (buf == nullptr)
		return true;
	bool differ = (sz != ex->size) || (memcmp(buf, ex->data, ex->size) != 0);
	IOFree(buf, sz);
	if (differ)
		return true;

	// The marker is empty, and readFile refuses an empty file,
	// so test that it exists, as the accelerator does.
	vnode_t vp = nullptr;
	if (vnode_lookup(markerPath, 0, &vp, ctx) != 0)
		return true;
	vnode_put(vp);
	return false;
}

static bool writeOne(const char *root, const PayloadFile *f, vfs_context_t ctx) {
	char path[1024];
	joinPath(path, sizeof(path), root, f->rel);
	int cmode = f->is_exec ? 0755 : 0644;
	int err = writeFile(path, (void *)(uintptr_t)f->data, f->size,
	                                    O_CREAT | O_TRUNC | O_WRONLY | O_NOFOLLOW, cmode);
	if (err != 0)
		graft_client_log(kClientID, "write %s failed (%d)", f->rel, err);
	return err == 0;
}

// The container the bundle lives in, and the SIP rootless storage class a stock
// host gives /Library/GPUBundles. Metal's gpu_bundle_find_trusted loads a GPU
// bundle only from a path whose storage class is this one (or from the sealed
// system volume), so the bundle we write to the Data volume must carry it to be
// loaded with SIP on, exactly as the kext-management tools tag the bundles they
// stage there.
static const char kContainerPath[] = "/Library/GPUBundles";
static const char kRootlessXattr[] = "com.apple.rootless";
static const char kKextMgmtClass[] = "KernelExtensionManagement";

// Give one path that storage class and the restricted flag. The direct
// VNOP_SETXATTR and vnode_setattr run below the vn_* MAC layer, as VNOP_MKDIR
// does for the directory creates, so the attributes are written from kernel
// context without relaxing SIP for anything else; only this path is tagged.
static void tagRootless(const char *path, vfs_context_t ctx) {
	vnode_t vp = nullptr;
	if (vnode_lookup(path, 0, &vp, ctx) != 0) {
		graft_client_log(kClientID, "tag: %s not found", path);
		return;
	}
	uio_t uio = uio_create(1, 0, UIO_SYSSPACE, UIO_WRITE);
	if (uio != nullptr) {
		uio_addiov(uio, (user_addr_t)(uintptr_t)kKextMgmtClass,
		           sizeof(kKextMgmtClass) - 1);
		int xerr = VNOP_SETXATTR(vp, kRootlessXattr, uio, 0, ctx);
		if (xerr != 0)
			graft_client_log(kClientID, "set rootless class on %s failed (%d)", path, xerr);
		uio_free(uio);
	}
	struct vnode_attr va;
	VATTR_INIT(&va);
	VATTR_SET(&va, va_flags, SF_RESTRICTED);
	int ferr = vnode_setattr(vp, &va, ctx);
	if (ferr != 0)
		graft_client_log(kClientID, "set restricted flag on %s failed (%d)", path, ferr);
	vnode_put(vp);
}

// Tag one tree (files, then directories deepest first, then its own root) with
// the GPU-bundle storage class and restricted flag. The root is tagged last, so
// a tagged root means a tagged tree. The container is tagged separately: it is
// never unlocked, so an upgrade does not widen SIP around it.
static void tagTree(const char *root, vfs_context_t ctx) {
	char p[1024];
	for (uint32_t i = 0; i < kFileCount; i++) {
		joinPath(p, sizeof(p), root, kFiles[i].rel);
		tagRootless(p, ctx);
	}
	for (uint32_t i = kDirCount; i-- > 0;) {
		joinPath(p, sizeof(p), root, kDirs[i]);
		tagRootless(p, ctx);
	}
	tagRootless(root, ctx);
}

// True if the bundle directory carries both the storage class and the
// restricted flag. The tag sets the bundle directory after everything inside
// it, so a tagged directory means a tagged tree. The flag alone does not
// count: a bundle copied into the restricted container under relaxed SIP is
// given the restricted flag but never the class, and the Metal loader reads
// the class.
static bool bundleTagged(vfs_context_t ctx) {
	vnode_t vp = nullptr;
	if (vnode_lookup(kBundlePath, 0, &vp, ctx) != 0)
		return false;
	struct vnode_attr va;
	VATTR_INIT(&va);
	VATTR_WANTED(&va, va_flags);
	bool restricted = vnode_getattr(vp, &va, ctx) == 0 &&
	                  VATTR_IS_SUPPORTED(&va, va_flags) &&
	                  (va.va_flags & SF_RESTRICTED) != 0;
	bool classed = false;
	char cls[64];
	uio_t uio = uio_create(1, 0, UIO_SYSSPACE, UIO_READ);
	if (uio != nullptr) {
		uio_addiov(uio, (user_addr_t)(uintptr_t)cls, sizeof(cls));
		size_t xsz = 0;
		if (VNOP_GETXATTR(vp, kRootlessXattr, uio, &xsz, 0, ctx) == 0) {
			size_t got = sizeof(cls) - (size_t)uio_resid(uio);
			classed = got == sizeof(kKextMgmtClass) - 1 &&
			          memcmp(cls, kKextMgmtClass, got) == 0;
		}
		uio_free(uio);
	}
	vnode_put(vp);
	return restricted && classed;
}

// Undo tagRootless for one path so the install can remove it: drop the storage
// class and the restricted flag, below the MAC layer as the tag sets them. A
// symbolic link is cleared itself, never what it names.
static void clearRootless(const char *path, vfs_context_t ctx) {
	vnode_t vp = nullptr;
	if (vnode_lookup(path, VNODE_LOOKUP_NOFOLLOW, &vp, ctx) != 0)
		return;
	rgfxVnopRemovexattr(vp, kRootlessXattr, 0, ctx);
	struct vnode_attr va;
	VATTR_INIT(&va);
	VATTR_SET(&va, va_flags, 0);
	vnode_setattr(vp, &va, ctx);
	vnode_put(vp);
}

// Unlink one file by full path, below the MAC layer as the directory creates
// use VNOP_MKDIR. Already absent counts as success. A symbolic link is
// unlinked itself, never what it names.
static void removeFile(const char *path, vfs_context_t ctx) {
	vnode_t vp = nullptr;
	if (vnode_lookup(path, VNODE_LOOKUP_NOFOLLOW, &vp, ctx) != 0)
		return;
	const char *slash = nullptr;
	for (const char *q = path; *q; q++)
		if (*q == '/')
			slash = q;
	char parent[1024];
	size_t plen = (slash == nullptr) ? 0 : (size_t)(slash - path);
	if (slash == nullptr || slash == path || plen >= sizeof(parent)) {
		vnode_put(vp);
		return;
	}
	memcpy(parent, path, plen);
	parent[plen] = '\0';
	const char *leaf = slash + 1;
	vnode_t dvp = nullptr;
	if (vnode_lookup(parent, 0, &dvp, ctx) == 0) {
		struct componentname cn;
		bzero(&cn, sizeof(cn));
		cn.cn_nameiop = DELETE;
		cn.cn_flags = 0;
		cn.cn_pnbuf = (char *)leaf;
		cn.cn_pnlen = (int)strlen(leaf);
		cn.cn_nameptr = (char *)leaf;
		cn.cn_namelen = (int)strlen(leaf);
		rgfxVnopRemove(dvp, vp, &cn, 0, ctx);
		vnode_put(dvp);
	}
	vnode_put(vp);
}

// Remove one empty directory by full path, below the MAC layer as removeFile
// unlinks files. Non-empty or absent is left as it is (best effort).
static void removeDir(const char *path, vfs_context_t ctx) {
	vnode_t vp = nullptr;
	if (vnode_lookup(path, VNODE_LOOKUP_NOFOLLOW, &vp, ctx) != 0)
		return;
	char leaf[256];
	struct componentname cn;
	vnode_t dvp = parentAndLeaf(path, DELETE, &cn, leaf, sizeof(leaf), ctx);
	if (dvp != nullptr) {
		rgfxVnopRmdir(dvp, vp, &cn, ctx);
		vnode_put(dvp);
	}
	vnode_put(vp);
}

// The VFS's record of a renamed vnode's name and parent, which vn_rename updates
// after the file system's rename and a VNOP called directly does not: without
// it the vnode keeps its old name, so every path the kernel builds for it and
// for what lies under it (vn_getpath, fcntl(F_GETPATH)) names the old path. The
// flags are vn_rename's: the name where it changed, the parent where it moved.
static void renameIdentity(vnode_t vp, vnode_t olddvp, const char *oldleaf, vnode_t newdvp,
                           const char *newleaf) {
	int flags = 0;
	if (strcmp(oldleaf, newleaf) != 0)
		flags |= VNODE_UPDATE_NAME;
	if (olddvp != newdvp)
		flags |= VNODE_UPDATE_PARENT;
	if (flags != 0)
		vnode_update_identity(vp, newdvp, newleaf, (int)strlen(newleaf), 0, flags);
}

// Rename `from` onto `to`, below the vn_* MAC layer as the directory creates
// use VNOP_MKDIR, so the container's restricted flag does not block it. With
// swap (both ends must exist), the two are exchanged atomically
// (VFS_RENAME_SWAP): a reader of `to` always sees one whole tree, never a
// half-swapped one. Without swap, `from` replaces `to`, which need not exist.
// Each renamed vnode then takes its new name and parent (renameIdentity), as
// after vn_rename: the installed bundle's vnodes would otherwise keep the
// staging name, reims-graphics.bundle.new.
// Shared with the AppleGVA preference's atomic write (RgfxGatePrefs.cpp).
bool rgfxRenameNode(const char *from, const char *to, bool swap,
                    vfs_context_t ctx) {
	vnode_t fvp = nullptr;
	if (vnode_lookup(from, 0, &fvp, ctx) != 0)
		return false; // nothing to rename
	vnode_t tvp = nullptr;
	if (vnode_lookup(to, 0, &tvp, ctx) != 0)
		tvp = nullptr; // target absent: a plain rename into a free name
	bool ok = false;
	if (!(swap && tvp == nullptr)) {
		char fleaf[256], tleaf[256];
		struct componentname fcn, tcn;
		vnode_t fdvp = parentAndLeaf(from, RENAME, &fcn, fleaf, sizeof(fleaf), ctx);
		vnode_t tdvp = parentAndLeaf(to, RENAME, &tcn, tleaf, sizeof(tleaf), ctx);
		if (fdvp != nullptr && tdvp != nullptr) {
			int err = swap ? rgfxVnopRenamex(fdvp, fvp, &fcn, tdvp, tvp, &tcn, VFS_RENAME_SWAP, ctx)
			               : rgfxVnopRename(fdvp, fvp, &fcn, tdvp, tvp, &tcn, ctx);
			ok = err == 0;
			if (!ok)
				graft_client_log(kClientID, "rename %s -> %s failed (%d)", from, to, err);
			if (ok) {
				renameIdentity(fvp, fdvp, fleaf, tdvp, tleaf);
				if (swap)
					renameIdentity(tvp, tdvp, tleaf, fdvp, fleaf);
			}
		}
		if (tdvp)
			vnode_put(tdvp);
		if (fdvp)
			vnode_put(fdvp);
	}
	if (tvp)
		vnode_put(tvp);
	vnode_put(fvp);
	return ok;
}

// The file system rgfx_tree_remove walks (rgfx_tree.h): lookups that do not
// follow a final symbolic link, a directory's entries as VNOP_READDIR gives
// them, and each node unlocked (clearRootless) and removed below the MAC layer.
// `buf` holds one read of a directory's records.
struct KernelTree {
	vfs_context_t ctx;
	uint8_t *buf;
	size_t cap;

	RgfxNodeKind kind(const char *path) {
		vnode_t vp = nullptr;
		if (vnode_lookup(path, VNODE_LOOKUP_NOFOLLOW, &vp, ctx) != 0)
			return kRgfxNodeAbsent;
		const RgfxNodeKind k = vnode_isdir(vp) ? kRgfxNodeDir : kRgfxNodeFile;
		vnode_put(vp);
		return k;
	}

	// The first entry of `dir` other than "." and "..": its reads from the
	// directory's start, each as many `struct direntry` records as `buf` holds,
	// until one holds such an entry or the directory ends.
	bool first(const char *dir, char *name, size_t namecap) {
		vnode_t vp = nullptr;
		if (vnode_open(dir, FREAD, 0, VNODE_LOOKUP_NOFOLLOW, &vp, ctx) != 0)
			return false;
		bool found = false;
		uio_t uio = uio_create(1, 0, UIO_SYSSPACE, UIO_READ);
		off_t offset = 0;
		for (int reads = 0; uio != nullptr && reads < 64; reads++) {
			uio_reset(uio, offset, UIO_SYSSPACE, UIO_READ);
			uio_addiov(uio, (user_addr_t)(uintptr_t)buf, cap);
			int eof = 0, count = 0;
			if (rgfxVnopReaddir(vp, uio, VNODE_READDIR_EXTENDED, &eof, &count, ctx) != 0)
				break;
			const size_t got = cap - (size_t)uio_resid(uio);
			size_t pos = 0;
			const char *n = nullptr;
			size_t nl = 0;
			if (rgfx_dir_next<struct direntry>(buf, got, &pos, &n, &nl)) {
				if (nl < namecap) {
					memcpy(name, n, nl);
					name[nl] = '\0';
					found = true;
				}
				break;
			}
			if (eof || got == 0)
				break;
			offset = uio_offset(uio);
		}
		if (uio != nullptr)
			uio_free(uio);
		vnode_close(vp, FREAD, ctx);
		return found;
	}

	void remove(const char *path, RgfxNodeKind kind) {
		clearRootless(path, ctx);
		if (kind == kRgfxNodeDir)
			removeDir(path, ctx);
		else
			removeFile(path, ctx);
	}
};

// Remove a staged or swapped-out tree rooted at `root`: every file and
// directory under it, whichever build staged them, deepest first, then the
// root (rgfx_tree_remove). Reading the directories, rather than the payload's
// list of this build's files, removes a file an older build staged and this
// one dropped: one left would keep the root, and the next install would stage
// into that root and swap it in as the live bundle, the old file with it. Best
// effort: a node that cannot be removed is logged, and the root then stays.
static void removeTree(const char *root, vfs_context_t ctx) {
	vnode_t vp = nullptr;
	if (vnode_lookup(root, VNODE_LOOKUP_NOFOLLOW, &vp, ctx) != 0)
		return; // nothing there
	vnode_put(vp);
	struct Scratch {
		char path[1024];
		uint8_t records[2048];
	};
	Scratch *s = static_cast<Scratch *>(IOMalloc(sizeof(Scratch)));
	if (s == nullptr) {
		graft_client_log(kClientID, "no memory to remove %s", root);
		return;
	}
	if (strlcpy(s->path, root, sizeof(s->path)) < sizeof(s->path)) {
		KernelTree fs = {ctx, s->records, sizeof(s->records)};
		if (!rgfx_tree_remove(fs, s->path, sizeof(s->path)))
			graft_client_log(kClientID, "%s could not be removed whole", root);
	}
	IOFree(s, sizeof(Scratch));
}

// Build the whole bundle tree under `root` (the staging sibling) and write its
// files: directories first, then the non-executable, non-marker files, the
// executable, and the wire-version marker last. The order does not matter in
// staging (nothing reads it), but it mirrors the tag order.
static bool stageTree(const char *root, vfs_context_t ctx) {
	if (!ensureDir(root, ctx))
		return false;
	char p[1024];
	for (uint32_t i = 0; i < kDirCount; i++) {
		joinPath(p, sizeof(p), root, kDirs[i]);
		if (!ensureDir(p, ctx))
			return false;
	}
	bool ok = true;
	for (uint32_t i = 0; ok && i < kFileCount; i++)
		if (!kFiles[i].is_exec && !kFiles[i].is_marker)
			ok = writeOne(root, &kFiles[i], ctx);
	for (uint32_t i = 0; ok && i < kFileCount; i++)
		if (kFiles[i].is_exec)
			ok = writeOne(root, &kFiles[i], ctx);
	for (uint32_t i = 0; ok && i < kFileCount; i++)
		if (kFiles[i].is_marker)
			ok = writeOne(root, &kFiles[i], ctx);
	return ok;
}

// The sibling staging directory's suffix: NAME.bundle.new beside NAME.bundle,
// under the same container, so its writes are admitted (installWriteAllowed)
// and its rename onto the bundle stays on one volume.
static const char kStageSuffix[] = ".new";

// Install or replace the bundle atomically. Build and tag the whole tree in the
// staging sibling, then put it in place with one rename: a live bundle is
// exchanged with it (VFS_RENAME_SWAP) and the old tree dropped, and where there
// is none the staged tree is moved in. A reader never sees a half-written
// bundle, so an upgrade that keeps the wire version (the rewrite race) can no
// longer brick the boot. The container is created if the Data volume lacks it
// and is never unlocked. Returns whether the bundle is in place afterwards.
static bool installBundle(vfs_context_t ctx) {
	char stageRoot[1024];
	if (!rgfx_sibling_path(kBundlePath, kStageSuffix, stageRoot, sizeof(stageRoot))) {
		graft_client_log(kClientID, "staging path too long; no install");
		return false;
	}
	bool ok = ensureDir(kContainerPath, ctx);
	// Clear a staging tree a crashed earlier install may have left.
	removeTree(stageRoot, ctx);
	if (ok)
		ok = stageTree(stageRoot, ctx);
	if (ok)
		tagTree(stageRoot, ctx);
	if (ok) {
		vnode_t live = nullptr;
		bool haveLive = vnode_lookup(kBundlePath, 0, &live, ctx) == 0;
		if (haveLive)
			vnode_put(live);
		ok = rgfxRenameNode(stageRoot, kBundlePath, haveLive, ctx);
	}
	// On a swap the old tree is now at stageRoot; on a move it is gone; on a
	// failure the staged tree is still there. One sweep clears whichever remains.
	removeTree(stageRoot, ctx);
	// The container carries the class on every install (it may be freshly made).
	tagRootless(kContainerPath, ctx);
	return ok;
}

// Records the install's outcome as RgfxGate's Graft client state
// (GraftClient.ai.aneesiqbal.RgfxGate, "<state>: <reason>"), so a refusal is
// visible there and never hidden behind the install-done flag, which only says
// the pass ended. Published on arm64, whose boots read it back; a no-op on
// x86-64, which keeps that build byte-identical.
#if defined(__arm64e__)
static inline void recordInstallState(const char *state, const char *reason) {
	graft_client_state(kClientID, state, reason);
}
#else
// A macro, not an empty inline: it drops its arguments at the preprocessor, so
// the states and reasons compile to nothing at all and the x86-64 build is
// byte-identical to one without this recording.
#define recordInstallState(state, reason) ((void)0)
#endif

// Whether this build installs into a base system's Data volume (rgfx_pass.h):
// arm64e's BaseSystem draws through Metal, x86-64's Recovery and installer do not.
#if defined(__arm64e__)
static constexpr bool kInstallsInBaseSystem = true;
#else
static constexpr bool kInstallsInBaseSystem = false;
#endif

static void doInstall() {
	vfs_context_t ctx = vfs_context_create(nullptr);
	if (ctx == nullptr) {
		recordInstallState("refused", "no VFS context for the install");
		return;
	}

	// A base system (rgfx_pass.h): x86-64's installer and Recovery, where nothing
	// is written; arm64's BaseSystem, where the bundle goes into its temporary
	// Data volume.
	vnode_t inst = nullptr;
	if (vnode_lookup("/System/Installation", 0, &inst, ctx) == 0) {
		vnode_put(inst);
		if (rgfx_pass_kind(true, kInstallsInBaseSystem) == kRgfxPassSkip) {
			recordInstallState("skipped", "installer or Recovery environment");
			vfs_context_rele(ctx);
			return;
		}
#if defined(__arm64e__)
		// The install writes below the VFS, which skips its read-only check (lessons.md): the
		// volume /Library resolves to, the BaseSystem's Data volume, must be one that takes
		// writes, its temporary one.
		vnode_t lib = nullptr;
		bool writable = vnode_lookup("/Library", 0, &lib, ctx) == 0;
		if (writable) {
			writable = !vfs_isrdonly(vnode_mount(lib));
			vnode_put(lib);
		}
		if (!writable) {
			graft_client_log(kClientID, "BaseSystem: its Data volume is read-only; no install");
			recordInstallState("skipped", "the BaseSystem's Data volume is read-only");
			vfs_context_rele(ctx);
			return;
		}
#endif
		graft_client_log(kClientID, "BaseSystem: the bundle goes into its temporary Data volume");
	}

#if defined(__arm64e__)
	// The arm64 install writes through the vnode op vector, whose offset in
	// struct vnode is a measured per-release fact. Confirm it against a live
	// vnode before any op-vector call: the root always exists and carries no
	// probe xattr, so the cross-check against the exported VNOP_GETXATTR is
	// clean. A release outside the measured five, or a failed cross-check,
	// refuses the install rather than dereferencing an unverified layout.
	{
		vnode_t root = nullptr;
		const char *why = "no root vnode";
		bool ready = vnode_lookup("/", 0, &root, ctx) == 0 &&
		             rgfxVnopInstallReady(root, ctx, &why);
		if (root != nullptr)
			vnode_put(root);
		if (!ready) {
			graft_client_log(kClientID, "vnode op vector unverified (%s); no install", why);
			recordInstallState("refused", why);
			vfs_context_rele(ctx);
			return;
		}
	}
#endif

	bool needWrite = installNeeded(ctx);
	bool tagged = bundleTagged(ctx);
	if (!needWrite && tagged) {
		graft_client_log(kClientID, "bundle current and tagged; no install");
		recordInstallState("present", "bundle already installed and tagged");
		vfs_context_rele(ctx);
		return;
	}

	gInstallThread = current_thread();
	gInstalling = true;

	bool ok;
	if (needWrite) {
		// Stage the whole tree and swap it over the live bundle atomically, so a
		// reader never sees a half-written one.
		ok = installBundle(ctx);
	} else {
		// The bundle is byte-current but untagged (staged under relaxed SIP,
		// then first seen SIP-on): tag it in place, with no rewrite and no
		// window, so the stock Metal loader trusts it.
		ensureDir(kContainerPath, ctx);
		tagTree(kBundlePath, ctx);
		tagRootless(kContainerPath, ctx);
		ok = true;
	}

	gInstalling = false;
	gInstallThread = nullptr;
	vfs_context_rele(ctx);
	graft_client_log(kClientID, "bundle %s %s", needWrite ? "install" : "tag",
	       ok ? "complete" : "FAILED");
	if (ok)
		recordInstallState(needWrite ? "installed" : "tagged",
		                   needWrite ? "wrote the bundle to /Library/GPUBundles and tagged it"
		                             : "tagged the current bundle in place");
	else
		recordInstallState("failed", "the bundle write did not complete");
}

// Publishes the install pass on IOResources (RgfxGate.hpp): `done` false while
// it runs, true once it has ended.
static void publishPass(bool done) {
	IOService::publishResource(kInstallDoneKey, done ? kOSBooleanTrue : kOSBooleanFalse);
}

#if defined(__arm64e__)
// The wall-clock time, for log lines read beside launchd's.
static void logAt(const char *what) {
	clock_sec_t secs = 0;
	clock_usec_t usecs = 0;
	clock_get_calendar_microtime(&secs, &usecs);
	graft_client_log(kClientID, "%s at %lu.%06u", what, (unsigned long)secs, (unsigned)usecs);
}

// Whether the root is a base system: the pass's own test (doInstall).
static bool inBaseSystem() {
	vfs_context_t ctx = vfs_context_create(nullptr);
	if (ctx == nullptr)
		return false;
	vnode_t vp = nullptr;
	bool base = vnode_lookup("/System/Installation", 0, &vp, ctx) == 0;
	if (base)
		vnode_put(vp);
	vfs_context_rele(ctx);
	return base;
}

// The Data volume /Library/Preferences is on, by its file system's id; false where that
// directory is not there.
static bool dataVolumeId(uint64_t *id) {
	vfs_context_t ctx = vfs_context_create(nullptr);
	if (ctx == nullptr)
		return false;
	vnode_t vp = nullptr;
	bool have = false;
	if (vnode_lookup("/Library/Preferences", 0, &vp, ctx) == 0) {
		mount_t mp = vnode_mount(vp);
		struct vfsstatfs *st = mp != nullptr ? vfs_statfs(mp) : nullptr;
		if (st != nullptr) {
			*id = (static_cast<uint64_t>(static_cast<uint32_t>(st->f_fsid.val[0])) << 32) |
			      static_cast<uint32_t>(st->f_fsid.val[1]);
			have = true;
		}
		vnode_put(vp);
	}
	vfs_context_rele(ctx);
	return have;
}

// The root file system's id, as the accelerator reads it to hold a read of its MetalPluginName
// (RgfxAccelerator.cpp, rootId, kept in step by hand); false before there is a root.
static bool rootId(uint64_t *id) {
	vnode_t root = vfs_rootvnode();
	if (root == nullptr)
		return false;
	mount_t mp = vnode_mount(root);
	struct vfsstatfs *st = mp != nullptr ? vfs_statfs(mp) : nullptr;
	if (st != nullptr)
		*id = (static_cast<uint64_t>(static_cast<uint32_t>(st->f_fsid.val[0])) << 32) |
		      static_cast<uint32_t>(st->f_fsid.val[1]);
	vnode_put(root);
	return st != nullptr;
}

// Publishes the root the pass that just ran ended under (RgfxGate.hpp, kPassRootKey), before the
// pass itself is published as ended: from then on a read of the accelerator's MetalPluginName on
// that root goes on, where one on any other root waits for its own pass.
static void publishPassRoot() {
	uint64_t root = 0;
	if (!rootId(&root))
		return;
	OSNumber *n = OSNumber::withNumber(root, 64);
	if (n == nullptr)
		return;
	IOService::publishResource(kPassRootKey, n);
	n->release();
}

// How often the watch looks for the next Data volume.
static constexpr uint32_t kDataWatchMs = 100;

// After a pass in arm64's BaseSystem, the boot leaves it by a userspace reboot that keeps the
// kernel, and with it this kext and the accelerator the bundle registered: to the full system
// once FileVault's password unlocks its Data volume, where WindowServer loads the bundle from
// that volume's /Library/GPUBundles, or to another BaseSystem, whose rc.temporaryDataVolume
// makes a new temporary Data volume, without the bundle. So each Data volume the boot moves to
// gets a pass of its own, published as the first was: the full system's bundle is then the
// kext's, as on a boot that starts there. The accelerator stays registered across that reboot,
// so what orders the full system's WindowServer after the pass is the root each pass publishes
// (publishPassRoot): Metal reads the accelerator's MetalPluginName before it loads the bundle,
// and the accelerator holds a read made on a root no pass has ended under yet. The watch ends
// once the root is a full system whose Data volume has had its pass (rgfx_pass.h), and says so
// with the property's false.
static void watchDataVolumes(uint64_t last) {
	for (;;) {
		IOSleep(kDataWatchMs);
		uint64_t id = 0;
		const bool have = dataVolumeId(&id);
		const bool base = inBaseSystem();
		if (rgfx_watch_passes(have, id, last)) {
			last = id;
			logAt(base ? "a new Data volume in the BaseSystem: the pass again"
			           : "the full system's Data volume: the pass again");
			publishPass(false);
			doInstall();
			publishPassRoot();
			publishPass(true);
			keepVideoPreference();
			logAt("the pass ended");
		}
		if (rgfx_watch_ends(base, have, id, last)) {
			IOService::publishResource(kPassRootKey, kOSBooleanFalse);
			logAt("the full system's Data volume has had its pass: the watch ends");
			return;
		}
	}
}
#endif

static void installEntry(void *, wait_result_t) {
	// Wait for the Data volume to mount, then install. /Library/Preferences only
	// exists once Data is up, so waiting on it (rather than /Library, a firmlink
	// that can resolve to the read-only system volume first) means a container
	// we create afterwards always lands on Data, not a placeholder Data later
	// shadows. Up to ~60 s, then give up for this boot. Sleep before the first
	// lookup: this thread is entered as a continuation very early in boot, and
	// IOSleep yields it through the scheduler into a normal preemptible state
	// before it takes the VFS namecache lock, which a development kernel
	// otherwise faults on (lock_rw_old.c, preemption check). The VFS context
	// waits for IOBSD too: Graft starts this kext before BSD is up on some
	// releases, and a credential taken before the kernel task has its BSD
	// process panics (Darwin 20: "thread wants credential but has no BSD
	// process").
	vfs_context_t ctx = nullptr;
	bool ready = false;
	for (int i = 0; i < 600; i++) {
		IOSleep(100);
		if (!ctx) {
			IOService *resources = IOService::getResourceService();
			if (!resources || !resources->getProperty("IOBSD"))
				continue;
			ctx = vfs_context_create(nullptr);
		}
		vnode_t vp = nullptr;
		if (ctx && vnode_lookup("/Library/Preferences", 0, &vp, ctx) == 0) {
			vnode_put(vp);
			ready = true;
			break;
		}
	}
	if (ctx)
		vfs_context_rele(ctx);
#if defined(__arm64e__)
	// The Data volume the first pass runs on, and whether the watch follows it.
	uint64_t firstData = 0;
	const bool watch = ready && dataVolumeId(&firstData) &&
	                   rgfx_pass_kind(inBaseSystem(), kInstallsInBaseSystem) ==
	                       kRgfxPassInstallThenWatch;
	if (watch)
		logAt("BaseSystem: the pass");
#endif
	if (ready)
		doInstall();
	else {
		graft_client_log(kClientID, "Data volume never appeared; no install");
		recordInstallState("refused", "Data volume never mounted");
	}
#if defined(__arm64e__)
	if (watch)
		publishPassRoot();
#endif
	publishPass(true);
	if (ready)
		keepVideoPreference();
#if defined(__arm64e__)
	if (watch) {
		logAt("the pass ended");
		watchDataVolumes(firstData);
	}
#endif
	thread_terminate(current_thread());
}

void startInstaller() {
	publishPass(false);
	recordInstallState("pending", "install thread starting");
	thread_t t = nullptr;
	if (kernel_thread_start(installEntry, nullptr, &t) == KERN_SUCCESS)
		thread_deallocate(t);
	else {
		graft_client_log(kClientID, "could not start install thread");
		recordInstallState("refused", "could not start the install thread");
		publishPass(true);
	}
}

} // namespace rgfxgate
