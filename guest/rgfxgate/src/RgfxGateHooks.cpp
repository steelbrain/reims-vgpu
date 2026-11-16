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
#include <mach/machine.h>
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
#if defined(__arm64e__)
// The same directory as a vnode's path names it in arm64's BaseSystem, where /Library is a
// symbolic link to the temporary Data volume's Library, not a firmlink (rgfx_pass.h).
static const char kGpuBundlesDataDir[] = "/System/Volumes/Data/Library/GPUBundles";
volatile bool gBaseSystemRoot = false;
#endif

bool cdhashIsOurs(void *fg, uint64_t offset) {
	if (gCsfgGetCdhash == nullptr || fg == nullptr)
		return false;
	size_t size = 0;
	const uint8_t *cd = gCsfgGetCdhash(fg, offset, &size);
	return rgfx_cdhash_allowed(cd, size, kAllowed, kAllowedCount);
}

extern "C" thread_t current_thread(void);
extern "C" int vn_getpath(struct vnode *vp, char *pathbuf, int *len);

// True when vp's path is under /Library/GPUBundles (on arm64 also under its
// BaseSystem's form of that path, kGpuBundlesDataDir). Used to pin both the
// install writes and the signature-trust exemption to the installed bundle's
// own directory, so neither is granted anywhere else on the system.
static bool vnodeUnderBundles(vnode_t vp) {
	if (vp == nullptr)
		return false;
	char path[1024];
	int len = (int)sizeof(path);
	if (vn_getpath(vp, path, &len) != 0)
		return false;
#if defined(__arm64e__)
	return rgfx_path_under(path, kGpuBundlesDir) || rgfx_path_under(path, kGpuBundlesDataDir);
#else
	return rgfx_path_under(path, kGpuBundlesDir);
#endif
}

#if defined(__arm64e__)

// --- arm64: the three AMFI admission ops, called as raw mpo_* --------------
// Graft shadows AMFI's mac_policy_ops table and places each replacement
// directly in the operation's slot, so these wrappers are called as the raw
// policy ops (mpo_*), not the x86 dispatchers below. Only the three admission
// ops are routed: the install writes go below the MAC layer on arm64
// (RgfxGateVnop.hpp), so there is no create/open/truncate route. Each wrapper
// has its operation's mpo shape, as XNU's mac_policy.h declares it, and forwards
// every argument unchanged. Graft hands back each original, AMFI's own mpo
// pointer, as a void * holds a function pointer (discriminator 0, Graft.h), so
// each is kept in a void * and converted to its mpo type at the call, which
// re-signs it for that type. A typed function-pointer variable written through a
// void ** would be authenticated with the type's discriminator and fault. AMFI
// signs each slot with its mpo type's own discriminator (function-type
// discrimination, rgfx_amfi.h), so each wrapper's type is checked against its
// slot's at compile time: a wrapper of another shape reads the wrong arguments,
// and fails the build instead.

// library validation (wall B): mpo_file_check_library_validation_t, no label.
using MpoLibVal = int(void *proc, void *fg, off_t slice_offset, uint64_t error_message,
                      size_t error_message_size);
static void *orgLibVal;
static int wrapLibVal(void *proc, void *fg, off_t slice_offset, uint64_t error_message,
                      size_t error_message_size) {
	if (cdhashIsOurs(fg, (uint64_t)slice_offset))
		return 0; // our bundle is valid to load into a library-validated process
	return reinterpret_cast<MpoLibVal *>(orgLibVal)(proc, fg, slice_offset, error_message,
	                                                error_message_size);
}

// file-map-executable (wall C): mpo_file_check_mmap_t, the label at arg3 (NULL
// from the dispatcher; never dereferenced). The *maxprot clamp is the x86
// wrapper's, for the same reason. The dispatcher asks every policy's op on its
// own, so the mapping needs both AMFI's and the Sandbox policy's: a sandboxed
// process whose profile reads /Library/GPUBundles but maps nothing executable
// there (WebKit's GPU process, replayd) is refused by Sandbox's op alone. Each
// policy's op gets the same rule and calls that policy's own original.
using MpoMmap = int(void *cred, void *fg, void *label, int prot, int flags, uint64_t file_pos,
                    int *maxprot);
static int admitMmap(void *original, void *cred, void *fg, void *label, int prot, int flags,
                     uint64_t file_pos, int *maxprot) {
	if ((prot & kVmProtExecute) && cdhashIsOurs(fg, file_pos)) {
		if (maxprot != nullptr)
			*maxprot &= prot;
		return 0; // the GPU process may map our driver executable
	}
	return reinterpret_cast<MpoMmap *>(original)(cred, fg, label, prot, flags, file_pos, maxprot);
}
static void *orgMmap;
static int wrapMmap(void *cred, void *fg, void *label, int prot, int flags, uint64_t file_pos,
                    int *maxprot) {
	return admitMmap(orgMmap, cred, fg, label, prot, flags, file_pos, maxprot);
}
static void *orgSandboxMmap;
static int wrapSandboxMmap(void *cred, void *fg, void *label, int prot, int flags,
                           uint64_t file_pos, int *maxprot) {
	return admitMmap(orgSandboxMmap, cred, fg, label, prot, flags, file_pos, maxprot);
}

// The rootless trust query (the Sandbox policy's policy_syscall, mpo_policy_syscall_t: proc,
// call, arg). On arm64's BaseSystem, IOGPU asks libsystem_sandbox whether our GPU bundle's path
// is trusted for the storage class KernelExtensionManagement, and libsystem_sandbox asks the
// Sandbox policy through mac_syscall(2), call 2. macOS 27 answers "not trusted" for the bundle
// on the BaseSystem's tmpfs Data volume, so WindowServer finds no Metal device and FileVault's
// unlock screen is never drawn; 14, 15 and 26 answer "trusted" for the same tagged file. While
// the BaseSystem is the root (gBaseSystemRoot, which each install pass clears and the
// BaseSystem's pass sets), this hook answers "trusted" for exactly that query on a path under
// our bundle, and passes every other call, and every call once the full system's pass has run,
// to the Sandbox policy's own handler. arg is a user address: eight words, with the
// answer's address at +0, the operation string's at +0x10, the subject selector at +0x18 (1 for
// a path), the path's at +0x20 and the storage class's at +0x38 (rgfx_match.h). A failed copy
// passes the call on rather than answering.
using MpoPolicySyscall = int(void *p, int call, uint64_t arg);
static void *orgSandboxPolicySyscall;
static int wrapSandboxPolicySyscall(void *p, int call, uint64_t arg) {
	MpoPolicySyscall *original = reinterpret_cast<MpoPolicySyscall *>(orgSandboxPolicySyscall);
	if (!gBaseSystemRoot || call != RGFX_ROOTLESS_CALL)
		return original(p, call, arg);
	uint64_t a[8];
	if (copyin(arg, a, sizeof(a)) != 0)
		return original(p, call, arg);
	const uint64_t outPtr = a[0], opPtr = a[2], selector = a[3], pathPtr = a[4], clsPtr = a[7];
	if (selector != RGFX_ROOTLESS_SELECTOR_PATH || outPtr == 0 || opPtr == 0 || pathPtr == 0 ||
	    clsPtr == 0)
		return original(p, call, arg);
	char op[64], cls[64], path[1024];
	size_t done = 0;
	if (copyinstr(opPtr, op, sizeof(op), &done) != 0 ||
	    copyinstr(clsPtr, cls, sizeof(cls), &done) != 0 ||
	    copyinstr(pathPtr, path, sizeof(path), &done) != 0)
		return original(p, call, arg);
	if (!rgfx_rootless_trust_ours(call, selector, op, cls, path))
		return original(p, call, arg);
	const uint64_t trusted = 1;
	if (copyout(&trusted, outPtr, sizeof(trusted)) != 0)
		return original(p, call, arg);
	static volatile bool logged = false;
	if (!logged) {
		logged = true;
		graft_client_log(kClientID, "answered the BaseSystem's rootless trust query for the bundle");
	}
	return 0;
}

// code-signature trust (wall B, before library validation):
// mpo_vnode_check_signature_t, (vp, label, cpu_type, cs_blob, cs_flags,
// signer_type, flags, platform, fatal_failure_desc, fatal_failure_desc_len):
// the label (real, from the vnode) at arg2, the slice's CPU type at arg3, the
// blob at arg4, and the two out-parameters of a fatal failure's description
// last, all forwarded. The x86 wrapVSig's comment holds for this one: the real
// policies run first, and only our bundle's own executable under
// /Library/GPUBundles is trusted when they reject it.
using MpoVnodeCheckSignature = int(vnode_t vp, void *label, cpu_type_t cpu_type, void *cs_blob,
                                   unsigned int *cs_flags, unsigned int *signer_type, int flags,
                                   unsigned int platform, char **fatal_failure_desc,
                                   size_t *fatal_failure_desc_len);
static void *orgVSig;
static int wrapVSig(vnode_t vp, void *label, cpu_type_t cpu_type, void *cs_blob,
                    unsigned int *cs_flags, unsigned int *signer_type, int flags,
                    unsigned int platform, char **fatal_failure_desc,
                    size_t *fatal_failure_desc_len) {
	int r = reinterpret_cast<MpoVnodeCheckSignature *>(orgVSig)(
	    vp, label, cpu_type, cs_blob, cs_flags, signer_type, flags, platform, fatal_failure_desc,
	    fatal_failure_desc_len);
	if (r == 0)
		return 0; // already trusted; leave everything to the real policies
	const uint8_t *cd = (gCsblobGetCdhash && cs_blob) ? gCsblobGetCdhash(cs_blob) : nullptr;
	// Our bundle's own ad-hoc signature is trusted, nothing more. arm64's BaseSystem allows
	// only platform binaries and makes its refusal fatal; admitting our bundle there drops the
	// fatal description, whose buffer (1024 bytes on 26.6.2) is left allocated, not freed: the
	// heap it comes from is AMFI's choice, which no header states, and a free into the wrong
	// heap panics the kernel. It is one buffer each time a process there adds the bundle's
	// signature: the few processes of a BaseSystem boot that load Metal (rgfx_vsig_answer).
	const bool ours = cd && rgfx_cdhash_allowed(cd, kRgfxCDHashLen, kAllowed, kAllowedCount) &&
	                  vnodeUnderBundles(vp);
	bool dropped = false;
	r = rgfx_vsig_answer(r, ours, fatal_failure_desc, fatal_failure_desc_len, &dropped);
	if (dropped)
		graft_client_log(kClientID, "the bundle admitted past a fatal signature refusal "
		                            "(the BaseSystem's platform-binary rule)");
	return r;
}

// Each wrapper's type, and so its mpo type, is signed under its slot's discriminator.
static_assert(__builtin_ptrauth_type_discriminator(decltype(wrapLibVal)) ==
                  kRgfxAmfiDiscLibraryValidation,
              "wrapLibVal has file_check_library_validation's mpo shape");
static_assert(__builtin_ptrauth_type_discriminator(decltype(wrapMmap)) ==
                  kRgfxAmfiDiscFileCheckMmap,
              "wrapMmap has file_check_mmap's mpo shape");
static_assert(__builtin_ptrauth_type_discriminator(decltype(wrapSandboxMmap)) ==
                  kRgfxAmfiDiscFileCheckMmap,
              "wrapSandboxMmap has file_check_mmap's mpo shape");
static_assert(__builtin_ptrauth_type_discriminator(decltype(wrapSandboxPolicySyscall)) ==
                  kRgfxAmfiDiscPolicySyscall,
              "wrapSandboxPolicySyscall has policy_syscall's mpo shape");
static_assert(__builtin_ptrauth_type_discriminator(decltype(wrapVSig)) ==
                  kRgfxAmfiDiscVnodeCheckSignature,
              "wrapVSig has vnode_check_signature's mpo shape");

bool installHooks() {
	uint64_t csfg = 0, csblob = 0;
	if (!graft_kernel_symbol(kClientID, "_csfg_get_cdhash", &csfg) ||
	    !graft_kernel_symbol(kClientID, "_csblob_get_cdhash", &csblob))
		return false;
	// Graft returns each symbol's raw address (Graft.h). On arm64e a C function
	// pointer is signed (IA key, zero discriminator for these plain fn-ptr
	// types), so sign the raw address for that schema here; the call sites
	// authenticate it. Casting the integer straight to the pointer type would
	// make the compiler authenticate a raw value and trap under FEAT_FPAC.
	gCsfgGetCdhash = reinterpret_cast<t_csfg_get_cdhash>(ptrauth_sign_unauthenticated(
	    reinterpret_cast<void *>(csfg), ptrauth_key_function_pointer, 0));
	gCsblobGetCdhash = reinterpret_cast<t_csblob_get_cdhash>(ptrauth_sign_unauthenticated(
	    reinterpret_cast<void *>(csblob), ptrauth_key_function_pointer, 0));
	// The replacements go as plain function pointers: Graft strips and re-signs
	// each for its slot. Each original comes back as a void *, converted to its
	// mpo type where its wrapper calls it.
	const graft_route_request requests[] = {
	    {RGFX_OP_LIBRARY_VALIDATION, reinterpret_cast<void *>(wrapLibVal), &orgLibVal},
	    {RGFX_OP_FILE_CHECK_MMAP, reinterpret_cast<void *>(wrapMmap), &orgMmap},
	    {RGFX_OP_VNODE_CHECK_SIGNATURE, reinterpret_cast<void *>(wrapVSig), &orgVSig},
	};
	if (!graft_route_kernel(kClientID, requests, sizeof(requests) / sizeof(requests[0])))
		return false;
	// The Sandbox policy's op goes in a call of its own: one call names one policy (Graft.h).
	// Without it, everything but sandboxed processes still maps the bundle, so a refusal (a
	// Graft that shadows AMFI alone) is logged and the gate goes on.
	const graft_route_request sandbox[] = {
	    {RGFX_OP_SANDBOX_FILE_CHECK_MMAP, reinterpret_cast<void *>(wrapSandboxMmap),
	     &orgSandboxMmap},
	};
	if (!graft_route_kernel(kClientID, sandbox, sizeof(sandbox) / sizeof(sandbox[0]))) {
		graft_client_log(kClientID, "hooks installed (3 AMFI ops; writes below MAC); Sandbox's "
		                            "file_check_mmap not routed: sandboxed processes (WebKit's GPU "
		                            "process) cannot map the bundle");
		return true;
	}
	// The rootless trust query goes in a call of its own, so a Graft that cannot route it (one
	// before 0.4.0) leaves the routes above in place: only macOS 27's BaseSystem needs it.
	const graft_route_request trust[] = {
	    {RGFX_OP_SANDBOX_POLICY_SYSCALL, reinterpret_cast<void *>(wrapSandboxPolicySyscall),
	     &orgSandboxPolicySyscall},
	};
	if (!graft_route_kernel(kClientID, trust, sizeof(trust) / sizeof(trust[0]))) {
		graft_client_log(kClientID, "hooks installed (3 AMFI ops, Sandbox's file_check_mmap; "
		                            "writes below MAC); Sandbox's policy_syscall not routed: "
		                            "macOS 27's BaseSystem will not trust the bundle");
		return true;
	}
	graft_client_log(kClientID, "hooks installed (3 AMFI ops, Sandbox's file_check_mmap and "
	                            "policy_syscall; writes below MAC)");
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
