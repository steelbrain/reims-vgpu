// The vnode operations RgfxGate's below-MAC install makes, as one set of thin
// wrappers so the installer (RgfxGateInstall.cpp) reads the same on both
// architectures.
//
// On x86-64 every VNOP_* the install uses is an exported kernel symbol, so each
// wrapper is a direct pass-through and the boot linker resolves it.
//
// On arm64 only VNOP_SETXATTR and VNOP_GETXATTR are exported; VNOP_MKDIR,
// VNOP_CREATE, VNOP_REMOVE, VNOP_RMDIR, VNOP_RENAME, VNOP_RENAMEX,
// VNOP_REMOVEXATTR and VNOP_READDIR exist under no external symbol at all (the
// compiler inlined each into its callers), so referencing them would leave the
// kext unresolvable and OpenCore would refuse it. Each is rebuilt exactly as the
// kernel's own VNOP_* wrappers build it: fill the operation's `vnop_*_args`
// struct, then call through the vnode's op vector at the index the operation's
// exported descriptor carries -- `vp->v_op[vnop_*_desc.vdesc_offset](&args)`.
// The descriptors (`_vnop_*_desc`) are exported on every release and name the
// index themselves, so the index is never a carried constant. The op vector's
// byte offset in `struct vnode` is the one private layout fact; it is a checked
// per-release constant (kRgfxVOpOffset), measured on the five config-2 Boot KCs
// (13.6, 14.6.1, 15.6.1, 26.6.2, 27.0.1) and confirmed at install by a
// cross-check against the exported VNOP_GETXATTR (rgfxVnopSelfTest), so a wrong
// offset -- an unknown release, or a layout change on a known one -- refuses the
// install rather than dereferencing a wild pointer.
#pragma once
#include <sys/vnode.h>
#include <sys/vnode_if.h>
#include <sys/uio.h>
#if defined(__arm64e__)
#include <libkern/version.h> // version_major, to key the op-vector offset by release
#endif

#if !defined(__arm64e__)
// The SDK declares only VNOP_GETXATTR / VNOP_SETXATTR as callable; the rest
// appear in doc comments alone. On x86-64 they are exported kernel symbols, so
// declare the ones the install uses and call them directly.
extern "C" int VNOP_MKDIR(vnode_t, vnode_t *, struct componentname *, struct vnode_attr *,
                          vfs_context_t);
extern "C" int VNOP_REMOVE(vnode_t, vnode_t, struct componentname *, int, vfs_context_t);
extern "C" int VNOP_RMDIR(vnode_t, vnode_t, struct componentname *, vfs_context_t);
extern "C" int VNOP_RENAME(vnode_t, vnode_t, struct componentname *, vnode_t, vnode_t,
                           struct componentname *, vfs_context_t);
extern "C" int VNOP_RENAMEX(vnode_t, vnode_t, struct componentname *, vnode_t, vnode_t,
                            struct componentname *, vfs_rename_flags_t, vfs_context_t);
extern "C" int VNOP_REMOVEXATTR(vnode_t, const char *, int, vfs_context_t);
extern "C" int VNOP_READDIR(vnode_t, struct uio *, int, int *, int *, vfs_context_t);
#endif

namespace rgfxgate {

#if defined(__arm64e__)

// struct vnodeop_desc is forward-declared but not defined in the Kernel
// framework SDK. Its first field is `int vdesc_offset`, the op-vector index
// ("first for speed", a stable XNU ABI); that is all the call-through reads.
struct rgfx_vnodeop_desc_head {
	int vdesc_offset;
};

// The op vector (VOPFUNC *v_op) in struct vnode. Byte-identical 0xd0 on all five
// config-2 Boot KCs 13.6..27.0.1 (Ambrosia's survey, 2026-11-08, confirmed via
// the exported VNOP_IOCTL's own op-vector call). Carried as the measured fact;
// rgfxVnopSelfTest confirms it against a live vnode before the install trusts it.
static constexpr size_t kRgfxVOpOffset = 0xd0;

// A vnode op-vector entry, as the kernel types and signs it (int (*)(void *)).
typedef int (*rgfx_vopfunc)(void *);

// Call vnode `opVnode`'s filesystem implementation of operation `desc` with the
// prepared `args` (whose a_desc the caller set to &desc). `opVnode` is the
// vnode whose op vector the kernel's own wrapper would use -- the first vnode
// argument of the operation (the parent directory for create/mkdir/remove/
// rmdir/rename, the file for xattr/readdir). The indirect call authenticates the
// slot pointer the same way the kernel's VNOP_* wrappers do.
static inline int rgfxVnodeCall(vnode_t opVnode, struct vnodeop_desc *desc, void *args) {
	const int index = reinterpret_cast<const rgfx_vnodeop_desc_head *>(desc)->vdesc_offset;
	rgfx_vopfunc *vops =
	    *reinterpret_cast<rgfx_vopfunc *const *>(reinterpret_cast<const uint8_t *>(opVnode) +
	                                             kRgfxVOpOffset);
	return vops[index](args);
}

// Whether `vops` reads as a kernel-space op-vector pointer: non-null, 8-aligned,
// and in the arm64 kernel's high half. A cheap gate before any dereference, so a
// grossly wrong offset refuses rather than faults.
static inline bool rgfxKernelPointer(const void *p) {
	uintptr_t v = reinterpret_cast<uintptr_t>(p);
	return v != 0 && (v & 7) == 0 && (v >> 48) == 0xffff;
}

// Confirm kRgfxVOpOffset against a live vnode before the install uses the op
// vector for the absent operations. GETXATTR is exported *and* reachable through
// the op vector, so it validates the path end to end: read the op vector at the
// measured offset, sanity-check the pointer and the getxattr slot, then compare
// a getxattr through the op vector against the exported VNOP_GETXATTR for a name
// that is absent on `probe` -- both must return the same errno (ENOATTR on a
// clean probe). Equal returns prove the op vector is wired to the same
// implementation the exported wrapper reaches, i.e. the offset is right. False
// (refuse the install) on any mismatch or implausible pointer.
static inline bool rgfxVnopSelfTest(vnode_t probe, vfs_context_t ctx) {
	if (probe == nullptr)
		return false;
	rgfx_vopfunc *vops =
	    *reinterpret_cast<rgfx_vopfunc *const *>(reinterpret_cast<const uint8_t *>(probe) +
	                                             kRgfxVOpOffset);
	if (!rgfxKernelPointer(vops))
		return false;
	const int gx = reinterpret_cast<const rgfx_vnodeop_desc_head *>(&vnop_getxattr_desc)->vdesc_offset;
	if (gx < 0 || gx > 4096 || !rgfxKernelPointer(reinterpret_cast<void *>(vops[gx])))
		return false;
	// A probe name no file carries; both paths should answer ENOATTR (93).
	static const char kProbeName[] = "com.steelbrain.rgfxgate.voptest";
	char scratch[4];
	int errVector = -1, errExport = -1;
	uio_t uio = uio_create(1, 0, UIO_SYSSPACE, UIO_READ);
	if (uio != nullptr) {
		uio_addiov(uio, reinterpret_cast<user_addr_t>(scratch), sizeof(scratch));
		size_t size = 0;
		struct vnop_getxattr_args a;
		a.a_desc = &vnop_getxattr_desc;
		a.a_vp = probe;
		a.a_name = kProbeName;
		a.a_uio = uio;
		a.a_size = &size;
		a.a_options = 0;
		a.a_context = ctx;
		errVector = vops[gx](&a);
		uio_free(uio);
	}
	uio = uio_create(1, 0, UIO_SYSSPACE, UIO_READ);
	if (uio != nullptr) {
		uio_addiov(uio, reinterpret_cast<user_addr_t>(scratch), sizeof(scratch));
		size_t size = 0;
		errExport = VNOP_GETXATTR(probe, kProbeName, uio, &size, 0, ctx);
		uio_free(uio);
	}
	return uio != nullptr && errVector == errExport;
}

// Whether the running release is one of the five the op-vector offset was
// measured on, by Darwin major: 13->22, 14->23, 15->24, 26->25, 27->26. The
// offset is byte-identical (0xd0) on all five; a release outside the five has no
// measured offset, so the install refuses rather than read a guessed layout.
static inline bool rgfxVnopKnownRelease() {
	switch (version_major) {
	case 22:
	case 23:
	case 24:
	case 25:
	case 26:
		return true;
	default:
		return false;
	}
}

// The gate the arm64 install passes before it uses the op vector for the absent
// operations: the release is one of the five measured ones, and the live
// cross-check against the exported VNOP_GETXATTR confirms the offset on `probe`.
// `*why` names the reason on a refusal.
static inline bool rgfxVnopInstallReady(vnode_t probe, vfs_context_t ctx, const char **why) {
	if (!rgfxVnopKnownRelease()) {
		if (why)
			*why = "macOS release not among the five with a measured v_op offset";
		return false;
	}
	if (!rgfxVnopSelfTest(probe, ctx)) {
		if (why)
			*why = "v_op offset self-check against VNOP_GETXATTR failed";
		return false;
	}
	return true;
}

#endif // __arm64e__

// --- the wrappers the install calls -----------------------------------------
// Named rgfxVnop* so the x86 build keeps calling the real VNOP_* and the arm64
// build takes the op-vector path, with no change at the call sites.

static inline int rgfxVnopMkdir(vnode_t dvp, vnode_t *vpp, struct componentname *cnp,
                                struct vnode_attr *vap, vfs_context_t ctx) {
#if defined(__arm64e__)
	struct vnop_mkdir_args a;
	a.a_desc = &vnop_mkdir_desc;
	a.a_dvp = dvp;
	a.a_vpp = vpp;
	a.a_cnp = cnp;
	a.a_vap = vap;
	a.a_context = ctx;
	return rgfxVnodeCall(dvp, &vnop_mkdir_desc, &a);
#else
	return VNOP_MKDIR(dvp, vpp, cnp, vap, ctx);
#endif
}

#if defined(__arm64e__)
// Create one file node below the MAC layer: VNOP_CREATE rebuilt through the op
// vector, for the arm64 writeFile (RgfxGateFile.hpp). x86 creates files through
// vnode_open(O_CREAT), which its create/open/truncate MAC routes admit.
static inline int rgfxVnopCreate(vnode_t dvp, vnode_t *vpp, struct componentname *cnp,
                                 struct vnode_attr *vap, vfs_context_t ctx) {
	struct vnop_create_args a;
	a.a_desc = &vnop_create_desc;
	a.a_dvp = dvp;
	a.a_vpp = vpp;
	a.a_cnp = cnp;
	a.a_vap = vap;
	a.a_context = ctx;
	return rgfxVnodeCall(dvp, &vnop_create_desc, &a);
}
#endif

static inline int rgfxVnopRemove(vnode_t dvp, vnode_t vp, struct componentname *cnp, int flags,
                                 vfs_context_t ctx) {
#if defined(__arm64e__)
	struct vnop_remove_args a;
	a.a_desc = &vnop_remove_desc;
	a.a_dvp = dvp;
	a.a_vp = vp;
	a.a_cnp = cnp;
	a.a_flags = flags;
	a.a_context = ctx;
	return rgfxVnodeCall(dvp, &vnop_remove_desc, &a);
#else
	return VNOP_REMOVE(dvp, vp, cnp, flags, ctx);
#endif
}

static inline int rgfxVnopRmdir(vnode_t dvp, vnode_t vp, struct componentname *cnp,
                                vfs_context_t ctx) {
#if defined(__arm64e__)
	struct vnop_rmdir_args a;
	a.a_desc = &vnop_rmdir_desc;
	a.a_dvp = dvp;
	a.a_vp = vp;
	a.a_cnp = cnp;
	a.a_context = ctx;
	return rgfxVnodeCall(dvp, &vnop_rmdir_desc, &a);
#else
	return VNOP_RMDIR(dvp, vp, cnp, ctx);
#endif
}

static inline int rgfxVnopRename(vnode_t fdvp, vnode_t fvp, struct componentname *fcnp,
                                 vnode_t tdvp, vnode_t tvp, struct componentname *tcnp,
                                 vfs_context_t ctx) {
#if defined(__arm64e__)
	struct vnop_rename_args a;
	a.a_desc = &vnop_rename_desc;
	a.a_fdvp = fdvp;
	a.a_fvp = fvp;
	a.a_fcnp = fcnp;
	a.a_tdvp = tdvp;
	a.a_tvp = tvp;
	a.a_tcnp = tcnp;
	a.a_context = ctx;
	return rgfxVnodeCall(fdvp, &vnop_rename_desc, &a);
#else
	return VNOP_RENAME(fdvp, fvp, fcnp, tdvp, tvp, tcnp, ctx);
#endif
}

static inline int rgfxVnopRenamex(vnode_t fdvp, vnode_t fvp, struct componentname *fcnp,
                                  vnode_t tdvp, vnode_t tvp, struct componentname *tcnp,
                                  vfs_rename_flags_t flags, vfs_context_t ctx) {
#if defined(__arm64e__)
	struct vnop_renamex_args a;
	a.a_desc = &vnop_renamex_desc;
	a.a_fdvp = fdvp;
	a.a_fvp = fvp;
	a.a_fcnp = fcnp;
	a.a_tdvp = tdvp;
	a.a_tvp = tvp;
	a.a_tcnp = tcnp;
	a.a_vap = nullptr; // reserved; the kernel's VNOP_RENAMEX passes none
	a.a_flags = flags;
	a.a_context = ctx;
	return rgfxVnodeCall(fdvp, &vnop_renamex_desc, &a);
#else
	return VNOP_RENAMEX(fdvp, fvp, fcnp, tdvp, tvp, tcnp, flags, ctx);
#endif
}

static inline int rgfxVnopRemovexattr(vnode_t vp, const char *name, int options,
                                      vfs_context_t ctx) {
#if defined(__arm64e__)
	struct vnop_removexattr_args a;
	a.a_desc = &vnop_removexattr_desc;
	a.a_vp = vp;
	a.a_name = name;
	a.a_options = options;
	a.a_context = ctx;
	return rgfxVnodeCall(vp, &vnop_removexattr_desc, &a);
#else
	return VNOP_REMOVEXATTR(vp, name, options, ctx);
#endif
}

static inline int rgfxVnopReaddir(vnode_t vp, struct uio *uio, int flags, int *eofflag,
                                  int *numdirent, vfs_context_t ctx) {
#if defined(__arm64e__)
	struct vnop_readdir_args a;
	a.a_desc = &vnop_readdir_desc;
	a.a_vp = vp;
	a.a_uio = uio;
	a.a_flags = flags;
	a.a_eofflag = eofflag;
	a.a_numdirent = numdirent;
	a.a_context = ctx;
	return rgfxVnodeCall(vp, &vnop_readdir_desc, &a);
#else
	return VNOP_READDIR(vp, uio, flags, eofflag, numdirent, ctx);
#endif
}

} // namespace rgfxgate
