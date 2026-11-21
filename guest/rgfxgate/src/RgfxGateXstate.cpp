// The xstate route (x86-64): every user thread runs at XCR0 with the AVX-512
// state components on, where XNU took AVX-512 state from CPUID leaf 0xD while
// leaf 7 says the CPU has no AVX-512 (QEMU's x-avx512-xstate, which reims-ctl
// sets where the host's own layout is not Intel's).
//
// A Zen 4 core runs a thread's code from its decoders, its op cache unused,
// while the thread's XCR0 leaves the AVX-512 state components (opmask,
// ZMM_Hi256, Hi16_ZMM) off, and XNU runs every thread at XCR0 0x7 until the
// thread takes a #UD on an EVEX or VEX instruction (fpu.c, fpUDflt and
// "On-demand AVX512 support"). This routes XNU's #NM handler, fpnoextflt,
// which XNU's lazy FPU enters on a thread's first FP/SIMD use after each
// switch, and promotes a user thread there through XNU's own promotion.
// fpu_thread_promote_avx512 is static, so not in the kernel's symbol table;
// its caller fpu_set_fxstate is: given the AVX-512 flavor for a thread at AVX
// state, it promotes the thread first (the AVX-512 save area, XCR0 0xe7, the
// task's xstate), then stores the state it was given. The thread's own state
// goes in, read with fpu_get_fxstate's AVX flavor, the AVX-512 fields zero.
// Apps are told nothing new: XNU's AVX-512 capability bits, and so
// hw.optional.avx512*, need leaf 7's bits too (commpage.c).
//
// fpu_get_fxstate and fpu_set_fxstate copy k0-7, zmm0-15's upper halves and
// zmm16-31 at Intel's XSAVE offsets, and the save area holds the CPU's own
// layout, which on AMD differs (rgfx_xstate.h). So where CPUID leaf 0xD's
// offsets are not Intel's, both are routed as well. For the calling thread (a
// signal's frame in sendsig, sigreturn's restore, an exception's state,
// thread_get/set_state on itself), the AVX-512 flavor's three fields come
// from XSAVE of components 5-7, and go back with XRSTOR of them, at CPUID's
// offsets. The original call leaves CR0.TS set, so the XSAVE or XRSTOR takes
// a #NM in the kernel; kernel_trap's T_NO_FPU case calls fpnoextflt, which
// loads the thread's save area before the instruction runs again. The XSAVE
// then reads the thread's registers; the XRSTOR overwrites what the original
// stored at the wrong offsets, and the next save writes the area in the CPU's
// layout. Another thread's save area is out of reach (pcb->ifps has no
// accessor), so the AVX-512 flavor for another thread is refused with
// KERN_FAILURE, XNU's answer for a thread with no AVX-512 state. Where the
// offsets are Intel's, XNU's copy is right and only fpnoextflt is routed.
// Where the layout does not fit XNU's save area nothing is routed: no thread
// is promoted whose state could not be copied.
//
// Graft's log (GraftLog on the registry root) gets one line for what was
// installed or why nothing was, and counts at 1, 10, 100, ...

#include <IOKit/IOLib.h>
#include <mach/thread_status.h>

#include "RgfxGate.hpp"
#include "rgfx_xstate.h"

#if !defined(__arm64e__)

namespace rgfxgate {
namespace {

using t_void = void (*)(void);
using t_fxstate = kern_return_t (*)(thread_t, thread_state_t, thread_flavor_t);
using t_bool = boolean_t (*)(void);
using t_task = task_t (*)(void);
using t_thread = thread_t (*)(void);

// XNU's own size for the AVX-512 save area (struct x86_avx512_thread_state):
// XSAVE's image must fit in it.
constexpr uint32_t kXnuAvx512Area = 2688;
// CR0.TS: set, the thread's FP/SIMD state is not live in the registers.
constexpr uint64_t kCr0Ts = 0x8;

void *gOrgNoExt;
void *gOrgGetFx;
void *gOrgSetFx;
// XNU's fpu_get_fxstate and fpu_set_fxstate for the promotion: the originals
// once they are routed, so promote() never enters the routes.
t_fxstate gGetFx;
t_fxstate gSetFx;
t_bool gAtInterrupt;
t_bool gPreemptionOn;
t_task gCurrentTask;
t_thread gCurrentThread;
task_t gKernelTask;
RgfxXstateLayout gLayout;
uint32_t gImageSize;
uint64_t gPromotions;
uint64_t gRefusals;
uint64_t gGets;
uint64_t gSets;
uint64_t gOtherThread;
uint64_t gLeft;

inline uint64_t readXcr0() {
	uint32_t lo, hi;
	__asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
	return (static_cast<uint64_t>(hi) << 32) | lo;
}

inline uint64_t readCr0() {
	uint64_t v;
	__asm__ volatile("mov %%cr0, %0" : "=r"(v));
	return v;
}

inline void cpuid(uint32_t leaf, uint32_t sub, uint32_t r[4]) {
	__asm__ volatile("cpuid" : "=a"(r[0]), "=b"(r[1]), "=c"(r[2]), "=d"(r[3]) : "a"(leaf), "c"(sub));
}

// XSAVE and XRSTOR of components 5-7 alone (RFBM 0xe0), in the standard
// form, on a 64-byte-aligned image.
inline void xsaveAvx512(uint8_t *image) {
	__asm__ volatile("xsave64 %0"
	                 : "+m"(*image)
	                 : "a"(static_cast<uint32_t>(kRgfxXstateAvx512)), "d"(0u)
	                 : "memory");
}

inline void xrstorAvx512(const uint8_t *image) {
	__asm__ volatile("xrstor64 %0"
	                 :
	                 : "m"(*image), "a"(static_cast<uint32_t>(kRgfxXstateAvx512)), "d"(0u)
	                 : "memory");
}

// Whether the thread may block: preemption on (level 0, interrupts on) and
// not in an interrupt.
inline bool mayBlock() { return gPreemptionOn() && !gAtInterrupt(); }

// 1, 10, 100, ...: when a count is logged.
inline bool decade(uint64_t n) {
	uint64_t p = 1;
	while (p < n)
		p *= 10;
	return p == n;
}

// Counts one of what and logs the count at 1, 10, 100, ..., where the thread
// may block.
void count(uint64_t *n, const char *what) {
	uint64_t v = __atomic_add_fetch(n, 1, __ATOMIC_RELAXED);
	if (decade(v) && mayBlock())
		graft_client_log(kClientID, "xstate: %llu %s", static_cast<unsigned long long>(v), what);
}

// The current thread's state, read with the AVX flavor, stored back with the
// AVX-512 one, which promotes the thread (see the top).
bool promote() {
	auto *st = static_cast<x86_avx512_state64_t *>(IOMalloc(sizeof(x86_avx512_state64_t)));
	if (st == nullptr)
		return false;
	bzero(st, sizeof(*st));
	thread_t th = gCurrentThread();
	auto ts = reinterpret_cast<thread_state_t>(st);
	kern_return_t g = gGetFx(th, ts, x86_AVX_STATE64);
	kern_return_t s = g == KERN_SUCCESS ? gSetFx(th, ts, x86_AVX512_STATE64) : KERN_FAILURE;
	IOFree(st, sizeof(*st));
	if (g == KERN_SUCCESS && s == KERN_SUCCESS)
		return true;
	uint64_t n = __atomic_add_fetch(&gRefusals, 1, __ATOMIC_RELAXED);
	if (decade(n))
		graft_client_log(kClientID, "xstate: %llu promotions refused (last: get %d, set %d)",
		                 static_cast<unsigned long long>(n), g, s);
	return false;
}

void wrapNoExt(void) {
	// Only where XNU's own promotion runs: user_trap's T_INVALID_OPCODE case
	// (fpUDflt) and its T_NO_FPU case (fpnoextflt) sit in one switch, after
	// pal_sti, on the faulting thread (trap.c). kernel_trap also enters
	// fpnoextflt, with whatever the interrupted code held; the promotion's
	// save area allocation waits (fp_state_alloc, Z_WAITOK), so the thread
	// must be preemptible.
	if ((readXcr0() & kRgfxXstateAvx512) == 0 && mayBlock() && gCurrentTask() != gKernelTask &&
	    promote()) {
		count(&gPromotions, "promotions");
		// fpu_set_fxstate leaves the stored state valid and CR0.TS set (with
		// no save area yet, the promotion ran fpnoextflt itself, and the
		// store then set TS again), and the original loads it; TS clear
		// means it is live.
		if (!(readCr0() & kCr0Ts))
			return;
	}
	reinterpret_cast<t_void>(gOrgNoExt)();
}

// An image for components 5-7, or nullptr where the thread may not wait for
// one; XNU's copy then stands, and is counted.
uint8_t *newImage() {
	if (!mayBlock()) {
		count(&gLeft, "AVX-512 copies left at Intel's offsets (thread not preemptible)");
		return nullptr;
	}
	auto *image = static_cast<uint8_t *>(IOMallocAligned(gImageSize, 64));
	if (image == nullptr)
		count(&gLeft, "AVX-512 copies left at Intel's offsets (no memory)");
	return image;
}

kern_return_t wrapGetFx(thread_t th, thread_state_t ts, thread_flavor_t f) {
	auto org = reinterpret_cast<t_fxstate>(gOrgGetFx);
	if (f != x86_AVX512_STATE64 || ts == nullptr)
		return org(th, ts, f);
	if (th != gCurrentThread()) {
		count(&gOtherThread, "AVX-512 state requests for another thread refused");
		return KERN_FAILURE;
	}
	kern_return_t r = org(th, ts, f);
	// XNU answers the flavor only for a thread at AVX-512 state, whose XCR0
	// has components 5-7.
	if (r != KERN_SUCCESS || (readXcr0() & kRgfxXstateAvx512) != kRgfxXstateAvx512)
		return r;
	uint8_t *image = newImage();
	if (image == nullptr)
		return r;
	bzero(image, gImageSize);
	xsaveAvx512(image);
	auto *st = reinterpret_cast<x86_avx512_state64_t *>(ts);
	rgfx_xstate_to_flavor(image, gLayout, reinterpret_cast<uint8_t *>(&st->fpu_k0),
	                      reinterpret_cast<uint8_t *>(&st->fpu_zmmh0),
	                      reinterpret_cast<uint8_t *>(&st->fpu_zmm16));
	IOFreeAligned(image, gImageSize);
	count(&gGets, "AVX-512 states read at CPUID's offsets");
	return r;
}

kern_return_t wrapSetFx(thread_t th, thread_state_t ts, thread_flavor_t f) {
	auto org = reinterpret_cast<t_fxstate>(gOrgSetFx);
	if (f != x86_AVX512_STATE64 || ts == nullptr)
		return org(th, ts, f);
	if (th != gCurrentThread()) {
		count(&gOtherThread, "AVX-512 state requests for another thread refused");
		return KERN_FAILURE;
	}
	kern_return_t r = org(th, ts, f);
	// The original promotes a thread at AVX state first, so a success leaves
	// XCR0 with components 5-7.
	if (r != KERN_SUCCESS || (readXcr0() & kRgfxXstateAvx512) != kRgfxXstateAvx512)
		return r;
	uint8_t *image = newImage();
	if (image == nullptr)
		return r;
	const auto *st = reinterpret_cast<const x86_avx512_state64_t *>(ts);
	rgfx_xstate_from_flavor(image, gLayout, reinterpret_cast<const uint8_t *>(&st->fpu_k0),
	                        reinterpret_cast<const uint8_t *>(&st->fpu_zmmh0),
	                        reinterpret_cast<const uint8_t *>(&st->fpu_zmm16));
	xrstorAvx512(image);
	IOFreeAligned(image, gImageSize);
	count(&gSets, "AVX-512 states stored at CPUID's offsets");
	return r;
}

} // namespace

void installXstate() {
	uint32_t r[4];
	cpuid(7, 0, r);
	if (r[1] & (1u << 16)) {
		graft_client_log(kClientID, "xstate: not installed: leaf 7 has avx512f, so XNU "
		                            "promotes AVX-512 threads itself");
		return;
	}
	uint64_t en = 0, gf = 0, sf = 0, ai = 0, pe = 0, ct = 0, cth = 0, kt = 0;
	const struct {
		const char *name;
		uint64_t *at;
	} syms[] = {
		{"_ml_fpu_avx512_enabled", &en}, {"_fpu_get_fxstate", &gf},
		{"_fpu_set_fxstate", &sf},       {"_ml_at_interrupt_context", &ai},
		{"_preemption_enabled", &pe},    {"_current_task", &ct},
		{"_current_thread", &cth},       {"_kernel_task", &kt},
	};
	for (const auto &y : syms) {
		if (!graft_kernel_symbol(kClientID, y.name, y.at) || *y.at == 0) {
			graft_client_log(kClientID, "xstate: not installed: no %s in the kernel's symbol table",
			                 y.name);
			return;
		}
	}
	if (!reinterpret_cast<t_bool>(en)()) {
		graft_client_log(kClientID, "xstate: not installed: XNU has no AVX-512 state (leaf 0xD "
		                            "lists none)");
		return;
	}
	// Where XSAVE puts components 5-7: leaf 0xD, sub-leaves 5-7, EBX the
	// offset and EAX the size.
	uint32_t c5[4], c6[4], c7[4];
	cpuid(0xd, 5, c5);
	cpuid(0xd, 6, c6);
	cpuid(0xd, 7, c7);
	gLayout = {c5[1], c5[0], c6[1], c6[0], c7[1], c7[0]};
	if (!rgfx_xstate_layout_ok(gLayout, kXnuAvx512Area)) {
		graft_client_log(kClientID, "xstate: not installed: leaf 0xD's AVX-512 components (k %u+%u, "
		                            "zmm_hi256 %u+%u, hi16_zmm %u+%u) do not fit XNU's save area",
		                 gLayout.opmask, gLayout.opmask_size, gLayout.zmm_hi256,
		                 gLayout.zmm_hi256_size, gLayout.hi16_zmm, gLayout.hi16_zmm_size);
		return;
	}
	const bool intel = rgfx_xstate_is_intel(gLayout);
	gImageSize = (rgfx_xstate_end(gLayout) + 63) & ~63u;
	gGetFx = reinterpret_cast<t_fxstate>(gf);
	gSetFx = reinterpret_cast<t_fxstate>(sf);
	gAtInterrupt = reinterpret_cast<t_bool>(ai);
	gPreemptionOn = reinterpret_cast<t_bool>(pe);
	gCurrentTask = reinterpret_cast<t_task>(ct);
	gCurrentThread = reinterpret_cast<t_thread>(cth);
	gKernelTask = *reinterpret_cast<task_t *>(kt);
	// A layout's routes go in together or none do (graft_route_kernel).
	const graft_route_request req[] = {
		{"_fpnoextflt", reinterpret_cast<void *>(wrapNoExt), &gOrgNoExt},
		{"_fpu_get_fxstate", reinterpret_cast<void *>(wrapGetFx), &gOrgGetFx},
		{"_fpu_set_fxstate", reinterpret_cast<void *>(wrapSetFx), &gOrgSetFx},
	};
	if (!graft_route_kernel(kClientID, req, intel ? 1 : 3)) {
		graft_client_log(kClientID, "xstate: not installed: Graft refused the routes");
		return;
	}
	if (!intel) {
		gGetFx = reinterpret_cast<t_fxstate>(gOrgGetFx);
		gSetFx = reinterpret_cast<t_fxstate>(gOrgSetFx);
	}
	graft_client_log(kClientID, "xstate: installed: user threads promote to AVX-512 state at first "
	                            "FP use; their AVX-512 state is copied %s (k %u, zmm_hi256 %u, "
	                            "hi16_zmm %u)",
	                 intel ? "by XNU (Intel's offsets)"
	                       : "at CPUID's offsets, another thread's refused",
	                 gLayout.opmask, gLayout.zmm_hi256, gLayout.hi16_zmm);
}

} // namespace rgfxgate

#endif
