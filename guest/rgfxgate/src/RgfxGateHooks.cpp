// The kernel hooks RgfxGate installs. Each is a Graft route of an XNU MAC
// dispatcher; every one short-circuits only for our embedded bundle (by its
// code-directory hash) or, for the install writes, only for our own install
// thread on the bundle path. Everything else calls the original unchanged.

#include <libkern/libkern.h>
#include <sys/vnode.h>

#include "RgfxGate.hpp"
#include "rgfx_match.h"
#include "RgfxGatePayload.h"

#if defined(__arm64e__)
#include <ptrauth.h>
#include "rgfx_amfi.h"
#endif

namespace rgfxgate {

t_csfg_get_cdhash gCsfgGetCdhash = nullptr;
volatile bool gInstalling = false;
thread_t gInstallThread = nullptr;

// csblob_get_cdhash(struct cs_blob *) -> the blob's CS_CDHASH_LEN-byte cdhash.
// Resolved once at client initialization; it is not a KPI symbol.
using t_csblob_get_cdhash = const uint8_t *(*)(void *cs_blob);
static t_csblob_get_cdhash gCsblobGetCdhash = nullptr;

#if defined(__arm64e__)
// The allowed cdhashes: every slice of the embedded bundle executable's. A process maps the
// slice of its own architecture, so platform processes (arm64e, WindowServer among them) map the
// arm64e slice and third-party ones (arm64) the arm64 slice, each with its own cdhash
// (tools/pack-payload.py).
static const uint8_t (*const kAllowed)[kRgfxCDHashLen] = rgfx_payload::kExecCDHashes;
static constexpr size_t kAllowedCount = rgfx_payload::kExecCDHashCount;
#else
// The one allowed cdhash: the embedded bundle executable's.
static const uint8_t (*const kAllowed)[kRgfxCDHashLen] = &rgfx_payload::kExecCDHash;
static constexpr size_t kAllowedCount = 1;
#endif

// The directory every install write stays under (the SIP-restricted path).
static const char kGpuBundlesDir[] = "/Library/GPUBundles";

bool cdhashIsOurs(void *fg, uint64_t offset) {
	if (gCsfgGetCdhash == nullptr || fg == nullptr)
		return false;
	size_t size = 0;
	const uint8_t *cd = gCsfgGetCdhash(fg, offset, &size);
	return rgfx_cdhash_allowed(cd, size, kAllowed, kAllowedCount);
}

extern "C" thread_t current_thread(void);
extern "C" int vn_getpath(struct vnode *vp, char *pathbuf, int *len);

// True when vp's path is under /Library/GPUBundles. Used to pin both the
// install writes and the signature-trust exemption to the installed bundle's
// own directory, so neither is granted anywhere else on the system.
static bool vnodeUnderBundles(vnode_t vp) {
	if (vp == nullptr)
		return false;
	char path[1024];
	int len = (int)sizeof(path);
	if (vn_getpath(vp, path, &len) != 0)
		return false;
	return rgfx_path_under(path, kGpuBundlesDir);
}

#if defined(__arm64e__)

// --- arm64: the three AMFI admission ops, called as raw mpo_* --------------
// Graft shadows AMFI's mac_policy_ops table and places each replacement
// directly in the operation's slot, so these wrappers are called as the raw
// policy ops (mpo_*), not the x86 dispatchers below. Only the three admission
// ops are routed: the install writes go below the MAC layer on arm64
// (RgfxGateVnop.hpp), so there is no create/open/truncate route. The mpo
// signatures insert a `struct label *` (file_check_mmap arg3,
// vnode_check_signature arg2, library_validation none; Ambrosia's disasm of the
// five config-2 KCs, 2026-11-08), forwarded unchanged. Each saved original is
// AMFI's own mpo pointer, signed key IA under the operation's constant
// discriminator with no address diversity (rgfx_amfi.h), so it is declared as a
// matching __ptrauth function pointer and called through directly.

// library validation (wall B): no label, the dispatcher's shape.
static int (*__ptrauth(ptrauth_key_asia, 0, kRgfxAmfiDiscLibraryValidation) orgLibVal)(
    void *proc, void *fg, off_t slice_offset, uint64_t error_message, size_t error_message_size);
static int wrapLibVal(void *proc, void *fg, off_t slice_offset, uint64_t error_message,
                      size_t error_message_size) {
	if (cdhashIsOurs(fg, (uint64_t)slice_offset))
		return 0; // our bundle is valid to load into a library-validated process
	return orgLibVal(proc, fg, slice_offset, error_message, error_message_size);
}

// file-map-executable (wall C): label at arg3 (NULL from the dispatcher; never
// dereferenced). The *maxprot clamp is the x86 wrapper's, for the same reason.
static int (*__ptrauth(ptrauth_key_asia, 0, kRgfxAmfiDiscFileCheckMmap) orgMmap)(
    void *cred, void *fg, void *label, int prot, int flags, uint64_t file_pos, int *maxprot);
static int wrapMmap(void *cred, void *fg, void *label, int prot, int flags, uint64_t file_pos,
                    int *maxprot) {
	if ((prot & kVmProtExecute) && cdhashIsOurs(fg, file_pos)) {
		if (maxprot != nullptr)
			*maxprot &= prot;
		return 0; // the GPU process may map our driver executable
	}
	return orgMmap(cred, fg, label, prot, flags, file_pos, maxprot);
}

// code-signature trust (wall B, before library validation): label at arg2
// (real, from the vnode). The x86 wrapVSig's comment holds for this one: the
// real policies run first, and only our bundle's own executable under
// /Library/GPUBundles is trusted when they reject it.
static int (*__ptrauth(ptrauth_key_asia, 0, kRgfxAmfiDiscVnodeCheckSignature) orgVSig)(
    vnode_t vp, void *label, void *cs_blob, void *imgp, unsigned int *cs_flags,
    unsigned int *signer_type, int flags, unsigned int platform);
static int wrapVSig(vnode_t vp, void *label, void *cs_blob, void *imgp, unsigned int *cs_flags,
                    unsigned int *signer_type, int flags, unsigned int platform) {
	int r = orgVSig(vp, label, cs_blob, imgp, cs_flags, signer_type, flags, platform);
	if (r == 0)
		return 0; // already trusted; leave everything to the real policies
	const uint8_t *cd = (gCsblobGetCdhash && cs_blob) ? gCsblobGetCdhash(cs_blob) : nullptr;
	if (cd && rgfx_cdhash_allowed(cd, kRgfxCDHashLen, kAllowed, kAllowedCount) &&
	    vnodeUnderBundles(vp))
		return 0; // trust our bundle's own ad-hoc signature, nothing more
	return r;
}

bool installHooks() {
	uint64_t csfg = 0, csblob = 0;
	if (!graft_kernel_symbol(kClientID, "_csfg_get_cdhash", &csfg) ||
	    !graft_kernel_symbol(kClientID, "_csblob_get_cdhash", &csblob))
		return false;
	gCsfgGetCdhash = reinterpret_cast<t_csfg_get_cdhash>(csfg);
	gCsblobGetCdhash = reinterpret_cast<t_csblob_get_cdhash>(csblob);
	// The replacements go as plain function pointers: Graft strips and re-signs
	// each for its slot. Each original's storage is the __ptrauth variable, so
	// AMFI's signed slot value, stored there raw, is called through as is.
	const graft_route_request requests[] = {
	    {RGFX_OP_LIBRARY_VALIDATION, reinterpret_cast<void *>(wrapLibVal),
	     reinterpret_cast<void **>(&orgLibVal)},
	    {RGFX_OP_FILE_CHECK_MMAP, reinterpret_cast<void *>(wrapMmap),
	     reinterpret_cast<void **>(&orgMmap)},
	    {RGFX_OP_VNODE_CHECK_SIGNATURE, reinterpret_cast<void *>(wrapVSig),
	     reinterpret_cast<void **>(&orgVSig)},
	};
	if (!graft_route_kernel(kClientID, requests, sizeof(requests) / sizeof(requests[0])))
		return false;
	graft_client_log(kClientID, "hooks installed (3 AMFI ops; writes below MAC)");
	return true;
}

#else // x86-64: Graft inline-hooks the six MAC dispatchers.

// True on exactly our install thread while an install is in flight, for a
// vnode (a directory being created in, or a file being written) under
// /Library/GPUBundles. The gInstalling short-circuit keeps the common path
// (every open/create/truncate in the system) a single load.
static bool installWriteAllowed(vnode_t vp) {
	if (!gInstalling || current_thread() != gInstallThread)
		return false;
	return vnodeUnderBundles(vp);
}

// --- library validation (wall B) ------------------------------------------
// int mac_file_check_library_validation(proc, fileglob, off_t slice_offset,
//                                       user_long_t error_message, size_t len)
using t_lib_val = int (*)(void *, void *, off_t, uint64_t, size_t);
static void *orgLibVal;
static int wrapLibVal(void *proc, void *fg, off_t slice_offset,
                      uint64_t error_message, size_t error_message_size) {
	if (cdhashIsOurs(fg, (uint64_t)slice_offset))
		return 0; // our bundle is valid to load into a library-validated process
	return reinterpret_cast<t_lib_val>(orgLibVal)(proc, fg, slice_offset, error_message, error_message_size);
}

// --- file-map-executable (wall C) ------------------------------------------
// int mac_file_check_mmap(cred, fileglob, int prot, int flags,
//                         uint64_t file_pos, int *maxprot)
using t_mmap = int (*)(void *, void *, int, int, uint64_t, int *);
static void *orgMmap;
static int wrapMmap(void *cred, void *fg, int prot, int flags,
                    uint64_t file_pos, int *maxprot) {
	if ((prot & kVmProtExecute) && cdhashIsOurs(fg, file_pos)) {
		// Short-circuiting skips the original's W^X downgrade of *maxprot, so
		// clamp it to the requested prot ourselves (our r-x mapping needs no
		// more), rather than leaving the caller's wider maxprot in place.
		if (maxprot != nullptr)
			*maxprot &= prot;
		return 0; // the GPU process may map our driver executable
	}
	return reinterpret_cast<t_mmap>(orgMmap)(cred, fg, prot, flags, file_pos, maxprot);
}

// --- code-signature trust (wall B, the gate before library validation) -----
// Loading our ad-hoc, team-less bundle into a library-validated process is
// refused at the code-signature check before library validation ever runs, so
// our executable is never mapped. int mac_vnode_check_signature(vnode, cs_blob,
// image_params, uint *cs_flags, uint *signer_type, int flags, uint platform):
// a nonzero return makes ubc_cs_blob_add reject the blob. We let the real
// policies run; only if they reject AND the blob is our bundle's do we accept
// it (return 0). We set no flags: CS_VALID is already on from the code
// directory, and the library-validation exception is wrapLibVal's, so the blob
// stays non-platform, team-less -- just a trusted ad-hoc signature, no more.
// The exemption is pinned to the path as well as the cdhash: only our bundle's
// own executable under /Library/GPUBundles is trusted, so a byte-identical copy
// of it placed anywhere else is not, exactly as the install writes are pinned
// to that directory (installWriteAllowed). Since the signature wall gates
// library validation and the executable mapping, a copy elsewhere fails here
// and never reaches wrapLibVal or wrapMmap, which need no path check of their
// own.
using t_vsig = int (*)(vnode_t, void *, void *, unsigned int *, unsigned int *,
                       int, unsigned int);
static void *orgVSig;
static int wrapVSig(vnode_t vp, void *cs_blob, void *imgp, unsigned int *cs_flags,
                    unsigned int *signer_type, int flags, unsigned int platform) {
	int r = reinterpret_cast<t_vsig>(orgVSig)(vp, cs_blob, imgp, cs_flags, signer_type, flags, platform);
	if (r == 0)
		return 0; // already trusted; leave everything to the real policies
	const uint8_t *cd = (gCsblobGetCdhash && cs_blob) ? gCsblobGetCdhash(cs_blob) : nullptr;
	if (cd && rgfx_cdhash_allowed(cd, kRgfxCDHashLen, kAllowed, kAllowedCount) &&
	    vnodeUnderBundles(vp))
		return 0; // trust our bundle's own ad-hoc signature, nothing more
	return r;
}

// --- install writes (wall P) -----------------------------------------------
// int mac_vnode_check_create(ctx, dvp, componentname, vnode_attr)
using t_vcreate = int (*)(vfs_context_t, vnode_t, struct componentname *, struct vnode_attr *);
static void *orgVCreate;
static int wrapVCreate(vfs_context_t ctx, vnode_t dvp,
                       struct componentname *cnp, struct vnode_attr *vap) {
	if (installWriteAllowed(dvp))
		return 0;
	return reinterpret_cast<t_vcreate>(orgVCreate)(ctx, dvp, cnp, vap);
}

// int mac_vnode_check_open(ctx, vp, int acc_mode)
using t_vopen = int (*)(vfs_context_t, vnode_t, int);
static void *orgVOpen;
static int wrapVOpen(vfs_context_t ctx, vnode_t vp, int acc_mode) {
	if (installWriteAllowed(vp))
		return 0;
	return reinterpret_cast<t_vopen>(orgVOpen)(ctx, vp, acc_mode);
}

// int mac_vnode_check_truncate(ctx, kauth_cred_t file_cred, vp)
using t_vtrunc = int (*)(vfs_context_t, void *, vnode_t);
static void *orgVTrunc;
static int wrapVTrunc(vfs_context_t ctx, void *file_cred, vnode_t vp) {
	if (installWriteAllowed(vp))
		return 0;
	return reinterpret_cast<t_vtrunc>(orgVTrunc)(ctx, file_cred, vp);
}

bool installHooks() {
    uint64_t csfg = 0, csblob = 0;
    if (!graft_kernel_symbol(kClientID, "_csfg_get_cdhash", &csfg) ||
        !graft_kernel_symbol(kClientID, "_csblob_get_cdhash", &csblob))
        return false;
    gCsfgGetCdhash = reinterpret_cast<t_csfg_get_cdhash>(csfg);
    gCsblobGetCdhash = reinterpret_cast<t_csblob_get_cdhash>(csblob);
    const graft_route_request requests[] = {
        {"_mac_file_check_library_validation", reinterpret_cast<void *>(wrapLibVal), &orgLibVal},
        {"_mac_file_check_mmap", reinterpret_cast<void *>(wrapMmap), &orgMmap},
        {"_mac_vnode_check_create", reinterpret_cast<void *>(wrapVCreate), &orgVCreate},
        {"_mac_vnode_check_open", reinterpret_cast<void *>(wrapVOpen), &orgVOpen},
        {"_mac_vnode_check_truncate", reinterpret_cast<void *>(wrapVTrunc), &orgVTrunc},
        {"_mac_vnode_check_signature", reinterpret_cast<void *>(wrapVSig), &orgVSig},
    };
    if (!graft_route_kernel(kClientID, requests, sizeof(requests) / sizeof(requests[0]))) {
        return false;
    }
    graft_client_log(kClientID, "hooks installed");
    return true;
}


#endif // __arm64e__

} // namespace rgfxgate
