// The kernel hooks RgfxGate installs. Each is a Lilu route of an XNU MAC
// dispatcher; every one short-circuits only for our embedded bundle (by its
// code-directory hash) or, for the install writes, only for our own install
// thread on the bundle path. Everything else calls the original unchanged.
#include <Headers/kern_patcher.hpp>
#include <Headers/kern_util.hpp>

#include <libkern/libkern.h>
#include <sys/vnode.h>

#include "RgfxGate.hpp"
#include "rgfx_match.h"
#include "RgfxGatePayload.h"

namespace rgfxgate {

t_csfg_get_cdhash gCsfgGetCdhash = nullptr;
volatile bool gInstalling = false;
thread_t gInstallThread = nullptr;

// csblob_get_cdhash(struct cs_blob *) -> the blob's CS_CDHASH_LEN-byte cdhash.
// Resolved once at patcher load; it is not a KPI symbol.
using t_csblob_get_cdhash = const uint8_t *(*)(void *cs_blob);
static t_csblob_get_cdhash gCsblobGetCdhash = nullptr;

// The one allowed cdhash: the embedded bundle executable's.
static const uint8_t (*const kAllowed)[kRgfxCDHashLen] = &rgfx_payload::kExecCDHash;
static constexpr size_t kAllowedCount = 1;

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
static t_lib_val orgLibVal;
static int wrapLibVal(void *proc, void *fg, off_t slice_offset,
                      uint64_t error_message, size_t error_message_size) {
	if (cdhashIsOurs(fg, (uint64_t)slice_offset))
		return 0; // our bundle is valid to load into a library-validated process
	return orgLibVal(proc, fg, slice_offset, error_message, error_message_size);
}

// --- file-map-executable (wall C) ------------------------------------------
// int mac_file_check_mmap(cred, fileglob, int prot, int flags,
//                         uint64_t file_pos, int *maxprot)
using t_mmap = int (*)(void *, void *, int, int, uint64_t, int *);
static t_mmap orgMmap;
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
	return orgMmap(cred, fg, prot, flags, file_pos, maxprot);
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
static t_vsig orgVSig;
static int wrapVSig(vnode_t vp, void *cs_blob, void *imgp, unsigned int *cs_flags,
                    unsigned int *signer_type, int flags, unsigned int platform) {
	int r = orgVSig(vp, cs_blob, imgp, cs_flags, signer_type, flags, platform);
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
static t_vcreate orgVCreate;
static int wrapVCreate(vfs_context_t ctx, vnode_t dvp,
                       struct componentname *cnp, struct vnode_attr *vap) {
	if (installWriteAllowed(dvp))
		return 0;
	return orgVCreate(ctx, dvp, cnp, vap);
}

// int mac_vnode_check_open(ctx, vp, int acc_mode)
using t_vopen = int (*)(vfs_context_t, vnode_t, int);
static t_vopen orgVOpen;
static int wrapVOpen(vfs_context_t ctx, vnode_t vp, int acc_mode) {
	if (installWriteAllowed(vp))
		return 0;
	return orgVOpen(ctx, vp, acc_mode);
}

// int mac_vnode_check_truncate(ctx, kauth_cred_t file_cred, vp)
using t_vtrunc = int (*)(vfs_context_t, void *, vnode_t);
static t_vtrunc orgVTrunc;
static int wrapVTrunc(vfs_context_t ctx, void *file_cred, vnode_t vp) {
	if (installWriteAllowed(vp))
		return 0;
	return orgVTrunc(ctx, file_cred, vp);
}

void installHooks(KernelPatcher &patcher) {
	// Resolve the two cdhash accessors the hooks need; neither is a KPI symbol.
	// csfg_get_cdhash gives a fileglob's cdhash (library validation, mmap);
	// csblob_get_cdhash gives a cs_blob's (the signature check).
	auto csfg = patcher.solveSymbol(KernelPatcher::KernelID, "_csfg_get_cdhash");
	if (csfg == 0) {
		SYSLOG("rgfxgate", "could not resolve _csfg_get_cdhash; gate disabled");
		patcher.clearError();
		return;
	}
	gCsfgGetCdhash = reinterpret_cast<t_csfg_get_cdhash>(csfg);
	auto csblob = patcher.solveSymbol(KernelPatcher::KernelID, "_csblob_get_cdhash");
	if (csblob != 0)
		gCsblobGetCdhash = reinterpret_cast<t_csblob_get_cdhash>(csblob);
	else
		patcher.clearError();

	// Build the routes. The signature hook is useful only with
	// csblob_get_cdhash: without it wrapVSig cannot know our blob and would just
	// pass every signature check through. Where that symbol did not resolve (a
	// future macOS renames or drops it), leave the signature hook out and say so
	// loudly: our bundle then fails the signature wall and WindowServer
	// crash-loops with no GPU, so the inert gate is named in the log rather than
	// left a silent brick, exactly as the csfg miss disables the whole gate.
	KernelPatcher::RouteRequest reqs[] = {
		{"_mac_file_check_library_validation", wrapLibVal, orgLibVal},
		{"_mac_file_check_mmap", wrapMmap, orgMmap},
		{"_mac_vnode_check_create", wrapVCreate, orgVCreate},
		{"_mac_vnode_check_open", wrapVOpen, orgVOpen},
		{"_mac_vnode_check_truncate", wrapVTrunc, orgVTrunc},
		{"_mac_vnode_check_signature", wrapVSig, orgVSig}, // last: dropped below if no csblob
	};
	size_t count = arrsize(reqs);
	if (gCsblobGetCdhash == nullptr) {
		SYSLOG("rgfxgate", "could not resolve _csblob_get_cdhash; signature gate "
		                   "inert, our bundle will not load");
		count -= 1; // drop the trailing signature route
	}
	if (!patcher.routeMultiple(KernelPatcher::KernelID, reqs, count, 0, 0, true, true)) {
		SYSLOG("rgfxgate", "routeMultiple failed (err %d)", patcher.getError());
		patcher.clearError();
		return;
	}
	DBGLOG("rgfxgate", "hooks installed");
}

} // namespace rgfxgate
