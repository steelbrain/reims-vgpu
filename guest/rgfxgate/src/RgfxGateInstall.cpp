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
// below the MAC layer, as the directory creates do. In the installer/Recovery
// the kext does nothing. The pass is published on IOResources from its start to
// its end (RgfxGate.hpp), and the accelerator registers only once it has ended.
#include <Headers/kern_file.hpp>
#include <Headers/kern_util.hpp>

#include <IOKit/IOLib.h>
#include <IOKit/IOService.h>
#include <kern/thread.h>
#include <libkern/libkern.h>
#include <mach/thread_act.h>
#include <sys/fcntl.h>
#include <sys/stat.h>
#include <sys/uio.h>
#include <sys/vnode.h>
#include <sys/vnode_if.h>

#include "RgfxGate.hpp"
#include "RgfxGatePayload.h"
#include "RgfxGatePrefs.hpp"
#include "rgfx_match.h"

// MacKernelSDK omits these VNOP prototypes; they are exported BSD KPIs, as
// VNOP_MKDIR is.
extern "C" int VNOP_MKDIR(vnode_t, vnode_t *, struct componentname *,
                          struct vnode_attr *, vfs_context_t);
extern "C" int VNOP_REMOVE(vnode_t, vnode_t, struct componentname *, int,
                           vfs_context_t);
extern "C" int VNOP_RMDIR(vnode_t, vnode_t, struct componentname *,
                          vfs_context_t);
extern "C" int VNOP_RENAME(vnode_t, vnode_t, struct componentname *, vnode_t,
                           vnode_t, struct componentname *, vfs_context_t);
extern "C" int VNOP_RENAMEX(vnode_t, vnode_t, struct componentname *, vnode_t,
                            vnode_t, struct componentname *, vfs_rename_flags_t,
                            vfs_context_t);
extern "C" int VNOP_REMOVEXATTR(vnode_t, const char *, int, vfs_context_t);

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
	int err = VNOP_MKDIR(dvp, &nvp, &cn, &va, ctx);
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
	uint8_t *buf = FileIO::readFileToBuffer(execPath, sz);
	if (buf == nullptr)
		return true;
	bool differ = (sz != ex->size) || (memcmp(buf, ex->data, ex->size) != 0);
	Buffer::deleter(buf);
	if (differ)
		return true;

	// The marker is empty, and FileIO::readFileToBuffer refuses an empty file,
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
	int err = FileIO::writeBufferToFile(path, (void *)(uintptr_t)f->data, f->size,
	                                    O_CREAT | O_TRUNC | O_WRONLY | O_NOFOLLOW, cmode);
	if (err != 0)
		SYSLOG("rgfxgate", "write %s failed (%d)", f->rel, err);
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
		SYSLOG("rgfxgate", "tag: %s not found", path);
		return;
	}
	uio_t uio = uio_create(1, 0, UIO_SYSSPACE, UIO_WRITE);
	if (uio != nullptr) {
		uio_addiov(uio, (user_addr_t)(uintptr_t)kKextMgmtClass,
		           sizeof(kKextMgmtClass) - 1);
		int xerr = VNOP_SETXATTR(vp, kRootlessXattr, uio, 0, ctx);
		if (xerr != 0)
			SYSLOG("rgfxgate", "set rootless class on %s failed (%d)", path, xerr);
		uio_free(uio);
	}
	struct vnode_attr va;
	VATTR_INIT(&va);
	VATTR_SET(&va, va_flags, SF_RESTRICTED);
	int ferr = vnode_setattr(vp, &va, ctx);
	if (ferr != 0)
		SYSLOG("rgfxgate", "set restricted flag on %s failed (%d)", path, ferr);
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

// Undo tagRootless for one path so the install can rewrite it: drop the storage
// class and the restricted flag, below the MAC layer as the tag sets them.
static void clearRootless(const char *path, vfs_context_t ctx) {
	vnode_t vp = nullptr;
	if (vnode_lookup(path, 0, &vp, ctx) != 0)
		return;
	VNOP_REMOVEXATTR(vp, kRootlessXattr, 0, ctx);
	struct vnode_attr va;
	VATTR_INIT(&va);
	VATTR_SET(&va, va_flags, 0);
	vnode_setattr(vp, &va, ctx);
	vnode_put(vp);
}

// Unlink one file by full path, below the MAC layer as the directory creates
// use VNOP_MKDIR. Already absent counts as success.
static void removeFile(const char *path, vfs_context_t ctx) {
	vnode_t vp = nullptr;
	if (vnode_lookup(path, 0, &vp, ctx) != 0)
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
		VNOP_REMOVE(dvp, vp, &cn, 0, ctx);
		vnode_put(dvp);
	}
	vnode_put(vp);
}

// Remove one empty directory by full path, below the MAC layer as removeFile
// unlinks files. Non-empty or absent is left as it is (best effort).
static void removeDir(const char *path, vfs_context_t ctx) {
	vnode_t vp = nullptr;
	if (vnode_lookup(path, 0, &vp, ctx) != 0)
		return;
	char leaf[256];
	struct componentname cn;
	vnode_t dvp = parentAndLeaf(path, DELETE, &cn, leaf, sizeof(leaf), ctx);
	if (dvp != nullptr) {
		VNOP_RMDIR(dvp, vp, &cn, ctx);
		vnode_put(dvp);
	}
	vnode_put(vp);
}

// Rename `from` onto `to`, below the vn_* MAC layer as the directory creates
// use VNOP_MKDIR, so the container's restricted flag does not block it. With
// swap (both ends must exist), the two are exchanged atomically
// (VFS_RENAME_SWAP): a reader of `to` always sees one whole tree, never a
// half-swapped one. Without swap, `from` replaces `to`, which need not exist.
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
			int err = swap ? VNOP_RENAMEX(fdvp, fvp, &fcn, tdvp, tvp, &tcn,
			                              VFS_RENAME_SWAP, ctx)
			               : VNOP_RENAME(fdvp, fvp, &fcn, tdvp, tvp, &tcn, ctx);
			ok = err == 0;
			if (!ok)
				SYSLOG("rgfxgate", "rename %s -> %s failed (%d)", from, to, err);
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

// Remove a staged or swapped-out tree rooted at `root`: unlock and unlink its
// files, then its directories deepest first, then the root. The payload layout
// (kFiles/kDirs) is the layout we ever stage, so a tree of our own build is
// cleared whole; a tree from a differing older build may leave entries this
// kext does not know, and its root rmdir then fails -- clutter, not a brick,
// swept the next time an install stages here. Best effort throughout.
static void removeTree(const char *root, vfs_context_t ctx) {
	char p[1024];
	for (uint32_t i = 0; i < kFileCount; i++) {
		joinPath(p, sizeof(p), root, kFiles[i].rel);
		clearRootless(p, ctx);
		removeFile(p, ctx);
	}
	for (uint32_t i = kDirCount; i-- > 0;) {
		joinPath(p, sizeof(p), root, kDirs[i]);
		clearRootless(p, ctx);
		removeDir(p, ctx);
	}
	clearRootless(root, ctx);
	removeDir(root, ctx);
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
		SYSLOG("rgfxgate", "staging path too long; no install");
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

static void doInstall() {
	vfs_context_t ctx = vfs_context_create(nullptr);
	if (ctx == nullptr)
		return;

	// Installer/Recovery: do nothing.
	vnode_t inst = nullptr;
	if (vnode_lookup("/System/Installation", 0, &inst, ctx) == 0) {
		vnode_put(inst);
		vfs_context_rele(ctx);
		return;
	}
	bool needWrite = installNeeded(ctx);
	bool tagged = bundleTagged(ctx);
	if (!needWrite && tagged) {
		DBGLOG("rgfxgate", "bundle current and tagged; no install");
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
	SYSLOG("rgfxgate", "bundle %s %s", needWrite ? "install" : "tag",
	       ok ? "complete" : "FAILED");
}

// Publishes the install pass on IOResources (RgfxGate.hpp): `done` false while
// it runs, true once it has ended.
static void publishPass(bool done) {
	IOService::publishResource(kInstallDoneKey, done ? kOSBooleanTrue : kOSBooleanFalse);
}

static void installEntry(void *, wait_result_t) {
	// Wait for the Data volume to mount, then install. /Library/Preferences only
	// exists once Data is up, so waiting on it (rather than /Library, a firmlink
	// that can resolve to the read-only system volume first) means a container
	// we create afterwards always lands on Data, not a placeholder Data later
	// shadows. Up to ~60 s, then give up for this boot. Sleep before the first
	// lookup: this thread is entered as a continuation very early in boot, and
	// IOSleep yields it through the scheduler into a normal preemptible state
	// before it takes the VFS namecache lock, which a development kernel
	// otherwise faults on (lock_rw_old.c, preemption check).
	vfs_context_t ctx = vfs_context_create(nullptr);
	bool ready = false;
	for (int i = 0; i < 600; i++) {
		IOSleep(100);
		vnode_t vp = nullptr;
		if (ctx && vnode_lookup("/Library/Preferences", 0, &vp, ctx) == 0) {
			vnode_put(vp);
			ready = true;
			break;
		}
	}
	if (ctx)
		vfs_context_rele(ctx);
	if (ready)
		doInstall();
	else
		SYSLOG("rgfxgate", "Data volume never appeared; no install");
	publishPass(true);
	if (ready)
		keepVideoPreference();
	thread_terminate(current_thread());
}

void startInstaller() {
	publishPass(false);
	thread_t t = nullptr;
	if (kernel_thread_start(installEntry, nullptr, &t) == KERN_SUCCESS)
		thread_deallocate(t);
	else {
		SYSLOG("rgfxgate", "could not start install thread");
		publishPass(true);
	}
}

} // namespace rgfxgate
