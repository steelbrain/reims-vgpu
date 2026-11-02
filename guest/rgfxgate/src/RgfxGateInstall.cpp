// RgfxGate's boot-time install. On the first normal boot with SIP on, before
// the accelerator looks for a GPU bundle, the kext writes its embedded
// reims-graphics.bundle to /Library/GPUBundles (a SIP-restricted path) in
// kernel context. It writes the executable before the wire-version marker the
// accelerator keys on, so a half-written bundle never registers, and only when
// the on-disk executable differs from the embedded one, so later boots are
// no-ops. The write is admitted by the vnode hooks (RgfxGateHooks.cpp) only
// while gInstalling is set on this thread and the target is under the bundle
// path. In the installer/Recovery the kext does nothing.
#include <Headers/kern_file.hpp>
#include <Headers/kern_util.hpp>

#include <IOKit/IOLib.h>
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

// MacKernelSDK omits the VNOP_MKDIR prototype; it is an exported BSD KPI.
extern "C" int VNOP_MKDIR(vnode_t, vnode_t *, struct componentname *,
                          struct vnode_attr *, vfs_context_t);

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
static bool installNeeded() {
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

	size_t msz = 0;
	uint8_t *mbuf = FileIO::readFileToBuffer(markerPath, msz);
	if (mbuf == nullptr)
		return true;
	Buffer::deleter(mbuf);
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
	if (!installNeeded()) {
		DBGLOG("rgfxgate", "bundle already current; no install");
		vfs_context_rele(ctx);
		return;
	}

	gInstallThread = current_thread();
	gInstalling = true;

	bool ok = ensureDir(kBundlePath, ctx);
	for (uint32_t i = 0; ok && i < kDirCount; i++) {
		char d[1024];
		joinPath(d, sizeof(d), kBundlePath, kDirs[i]);
		ok = ensureDir(d, ctx);
	}
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

	// With every file in place, tag the tree with the GPU-bundle storage class
	// so the stock Metal loader trusts it with SIP on.
	if (ok)
		tagInstalledTree(ctx);

	gInstalling = false;
	gInstallThread = nullptr;
	vfs_context_rele(ctx);
	SYSLOG("rgfxgate", "bundle install %s", ok ? "complete" : "FAILED");
}

static void installEntry(void *, wait_result_t) {
	// Wait for the Data volume's /Library/GPUBundles to appear (launchd mounts
	// it well before WindowServer). Up to ~60 s, then give up for this boot.
	// Sleep before the first lookup: this thread is entered as a continuation
	// very early in boot, and IOSleep yields it through the scheduler into a
	// normal preemptible state before it takes the VFS namecache lock, which a
	// development kernel otherwise faults on (lock_rw_old.c, preemption check).
	vfs_context_t ctx = vfs_context_create(nullptr);
	bool ready = false;
	for (int i = 0; i < 600; i++) {
		IOSleep(100);
		vnode_t vp = nullptr;
		if (ctx && vnode_lookup("/Library/GPUBundles", 0, &vp, ctx) == 0) {
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
		SYSLOG("rgfxgate", "/Library/GPUBundles never appeared; no install");
	thread_terminate(current_thread());
}

void startInstaller() {
	thread_t t = nullptr;
	if (kernel_thread_start(installEntry, nullptr, &t) == KERN_SUCCESS)
		thread_deallocate(t);
	else
		SYSLOG("rgfxgate", "could not start install thread");
}

} // namespace rgfxgate
