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
// byte offset in `struct vnode` is the one private layout fact (kRgfxVOpOffset,
// 0xd0, measured on the config-2 Boot KCs from 13.6 to 27.0.1 and on 27.2.1's).
// The install does not gate on a release allow-list for it -- Apple's version
// numbering makes one brittle, and a release outside the list would refuse an
// offset that is in fact correct (27.0.1's VMAPPLE kernel reports version_major
// 27, not the 26 the list expected). Instead it proves the offset on the live
// kernel with data reads alone (rgfxVnopProveOffset), never calling or
// authenticating through a candidate pointer, so a wrong offset can only
// refuse, never fault. Only a proven offset is used for the absent operations'
// op-vector calls.
#pragma once
#include <sys/vnode.h>
#include <sys/vnode_if.h>
#include <sys/uio.h>
#if defined(__arm64e__)
#include <ptrauth.h>
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

// The op vector (VOPFUNC *v_op) in struct vnode. Byte-identical 0xd0 on the
// config-2 Boot KCs 13.6..27.0.1 (Ambrosia's survey; on 27.0.1's VMAPPLE kernel,
// build 26A434, the exported _VNOP_GETXATTR's only x0-relative load is
// [x0, #0xd0] before it indexes the op vector, and so it is on 27.2.1's, build
// 26B131). Carried as the measured fact;
// rgfxVnopProveOffset confirms it on the live kernel, by data reads alone,
// before the install trusts it.
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

// Whether `p` reads as a kernel-space pointer: non-null, `align`-aligned, and in
// the arm64 kernel's high half (canonical 0xffff...). The op-vector table is
// 8-aligned; a code entry is 4-aligned. A cheap gate before any dereference, so a
// grossly wrong offset refuses rather than faults.
static inline bool rgfxKernelAligned(const void *p, uintptr_t align) {
	uintptr_t v = reinterpret_cast<uintptr_t>(p);
	return v != 0 && (v & (align - 1)) == 0 && (v >> 48) == 0xffff;
}

// Strip any pointer-authentication code from `p`, giving its plain address. A
// pure bit operation (xpac): it never authenticates, so it cannot trap, and it
// is harmless on a value that is not a signed pointer.
static inline const void *rgfxStripPAC(const void *p) {
	return ptrauth_strip(p, ptrauth_key_function_pointer);
}

// Read struct vnode's op-vector word (v_op) at the measured offset, PAC stripped.
// A plain data read within the vnode struct; no call, no authentication.
static inline const void *const *rgfxReadVOp(vnode_t vp) {
	const void *raw = *reinterpret_cast<const void *const *>(
	    reinterpret_cast<const uint8_t *>(vp) + kRgfxVOpOffset);
	return reinterpret_cast<const void *const *>(rgfxStripPAC(raw));
}

// How many of the op vector's leading slots to read, and how many must look like
// kernel code pointers. The op vector is a dense array -- one slot per vnop,
// every slot a real implementation -- so 96 slots stay well inside it, and inside
// struct mount (the only other field a wrong offset survives to here, and larger
// still), so the read never runs off the end. ~90% tolerates a few absent slots.
static constexpr int kRgfxProbeSlots = 96;
static constexpr int kRgfxProbeMin = 86;

// Prove kRgfxVOpOffset names struct vnode's op vector on the live kernel, by data
// reads alone -- never calling or authenticating through a candidate pointer, so
// a wrong offset can only refuse, never fault. `probe` and `other` are two vnodes
// of the same filesystem. Steps, and what a wrong offset does at each:
//  1. Read the word at the offset from both vnodes, PAC stripped. Both reads are
//     inside the vnode struct (0xd0 is well within it), so neither faults
//     whatever field the offset names.
//  2. The two must be equal and a canonical 8-aligned kernel pointer. Only a
//     field shared by same-filesystem vnodes survives -- v_op or v_mount; a
//     per-vnode field (v_data, v_id, ...) differs and refuses here. Both
//     survivors are mapped, so the array reads in step 3 cannot reach unmapped
//     memory.
//  3. Over the first kRgfxProbeSlots words, at least kRgfxProbeMin must strip to
//     canonical 4-aligned kernel pointers. The op vector is a dense array of
//     signed code pointers; the other survivor, a mount struct, is a mix of
//     counts, flags and locks and fails this. Each word is read as data and
//     stripped (xpac), never called.
//  4. The getxattr slot the install will use -- its index in range, inside the
//     region step 3 proved -- must itself strip to a kernel pointer.
// Passing all four proves the offset; the install's op-vector calls then run on a
// layout proven here. Any failure refuses, having made no call.
static inline bool rgfxVnopProveOffset(vnode_t probe, vnode_t other) {
	if (probe == nullptr || other == nullptr)
		return false;
	const void *const *vops = rgfxReadVOp(probe);
	if (vops != rgfxReadVOp(other) || !rgfxKernelAligned(vops, 8))
		return false; // not a field shared by same-filesystem vnodes
	int good = 0;
	for (int i = 0; i < kRgfxProbeSlots; i++)
		if (rgfxKernelAligned(rgfxStripPAC(vops[i]), 4))
			good++;
	if (good < kRgfxProbeMin)
		return false; // not a dense array of code pointers (e.g. v_mount)
	const int gx = reinterpret_cast<const rgfx_vnodeop_desc_head *>(&vnop_getxattr_desc)->vdesc_offset;
	if (gx < 0 || gx >= kRgfxProbeSlots)
		return false; // keep the slot read inside the region step 3 proved
	return rgfxKernelAligned(rgfxStripPAC(vops[gx]), 4);
}

// The gate the arm64 install passes before it uses the op vector for the absent
// operations: a second vnode of the same filesystem, then rgfxVnopProveOffset's
// data-only proof of the offset across the two. The whole gate -- it holds on any
// release where the offset is right and refuses where it is not, with no release
// list to keep and no call through an unproven pointer. `*why` names the reason
// on a refusal.
static inline bool rgfxVnopInstallReady(vnode_t probe, vfs_context_t ctx, const char **why) {
	vnode_t other = nullptr;
	if (vnode_lookup("/System", 0, &other, ctx) != 0) {
		if (why)
			*why = "could not open a second vnode to confirm the v_op offset";
		return false;
	}
	bool ok = rgfxVnopProveOffset(probe, other);
	vnode_put(other);
	if (!ok) {
		if (why)
			*why = "v_op offset proof failed; op-vector layout not confirmed";
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
