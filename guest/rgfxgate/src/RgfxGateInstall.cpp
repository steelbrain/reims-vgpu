// RgfxGate's boot-time install. On the first normal boot, before the
// accelerator looks for a GPU bundle, the kext writes its embedded
// reims-graphics.bundle to /Library/GPUBundles in kernel context and gives the
// tree the SIP rootless storage class a stock host carries there, so the Metal
// loader trusts it (RgfxGate.hpp). It writes the executable before the
// wire-version marker the accelerator keys on, so a half-written bundle never
// registers, and creates the container first if the Data volume lacks it.
// Later boots are no-ops once the bundle is current and tagged. A newer kext
// whose embedded executable differs replaces the bundle: it unlocks the tagged
// tree, drops the marker, then rewrites and re-tags. A bundle that is current
// but untagged -- staged under relaxed SIP, then booted SIP-on -- is tagged in
// place. The writes are admitted by the vnode hooks (RgfxGateHooks.cpp) only
// while gInstalling is set on this thread and the target is under the bundle
// path; the tag, the unlock and the marker drop run below the MAC layer, as the
// directory creates do. In the installer/Recovery the kext does nothing. The
// pass is published on IOResources from its start to its end (RgfxGate.hpp), and
// the accelerator registers only once it has ended.
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

// MacKernelSDK omits these VNOP prototypes; they are exported BSD KPIs, as
// VNOP_MKDIR is.
extern "C" int VNOP_MKDIR(vnode_t, vnode_t *, struct componentname *,
                          struct vnode_attr *, vfs_context_t);
extern "C" int VNOP_REMOVE(vnode_t, vnode_t, struct componentname *, int,
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

static bool writeOne(const PayloadFile *f, vfs_context_t ctx) {
	char path[1024];
	joinPath(path, sizeof(path), kBundlePath, f->rel);
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

// Tag the installed tree and its container, deepest path first so a directory's
// restricted flag is set only after everything inside it is already tagged.
static void tagInstalledTree(vfs_context_t ctx) {
	char p[1024];
	for (uint32_t i = 0; i < kFileCount; i++) {
		joinPath(p, sizeof(p), kBundlePath, kFiles[i].rel);
		tagRootless(p, ctx);
	}
	for (uint32_t i = kDirCount; i-- > 0;) {
		joinPath(p, sizeof(p), kBundlePath, kDirs[i]);
		tagRootless(p, ctx);
	}
	tagRootless(kBundlePath, ctx);
	tagRootless(kContainerPath, ctx);
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

// Unlock the whole tagged tree before an upgrade, shallowest path first so a
// directory is cleared before the entries under it that it protects.
static void clearInstalledTree(vfs_context_t ctx) {
	char p[1024];
	clearRootless(kContainerPath, ctx);
	clearRootless(kBundlePath, ctx);
	for (uint32_t i = 0; i < kDirCount; i++) {
		joinPath(p, sizeof(p), kBundlePath, kDirs[i]);
		clearRootless(p, ctx);
	}
	for (uint32_t i = 0; i < kFileCount; i++) {
		joinPath(p, sizeof(p), kBundlePath, kFiles[i].rel);
		clearRootless(p, ctx);
	}
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

	// Create the container if the Data volume lacks it, then the bundle tree.
	bool ok = ensureDir(kContainerPath, ctx) && ensureDir(kBundlePath, ctx);
	for (uint32_t i = 0; ok && i < kDirCount; i++) {
		char d[1024];
		joinPath(d, sizeof(d), kBundlePath, kDirs[i]);
		ok = ensureDir(d, ctx);
	}

	if (ok && needWrite) {
		// A newer executable replaces an existing one: unlock our own tagged
		// tree first, then drop the marker so a half-written bundle never
		// registers while the files are rewritten.
		if (tagged)
			clearInstalledTree(ctx);
		char marker[1024];
		joinPath(marker, sizeof(marker), kBundlePath, kMarkerRel);
		removeFile(marker, ctx);
		// Non-executable, non-marker files first; then the executable; the
		// wire-version marker last, so the accelerator never sees a partial bundle.
		for (uint32_t i = 0; ok && i < kFileCount; i++)
			if (!kFiles[i].is_exec && !kFiles[i].is_marker)
				ok = writeOne(&kFiles[i], ctx);
		for (uint32_t i = 0; ok && i < kFileCount; i++)
			if (kFiles[i].is_exec)
				ok = writeOne(&kFiles[i], ctx);
		for (uint32_t i = 0; ok && i < kFileCount; i++)
			if (kFiles[i].is_marker)
				ok = writeOne(&kFiles[i], ctx);
	}

	// Tag the tree with the GPU-bundle storage class so the stock Metal loader
	// trusts it with SIP on: after a (re)write, or in place for a current but
	// untagged bundle.
	if (ok)
		tagInstalledTree(ctx);
	else
		// The container is the system's, classed on every install: give it its
		// tags back even when our own tree failed to write, so an update that
		// unlocked it never leaves it open.
		tagRootless(kContainerPath, ctx);

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
